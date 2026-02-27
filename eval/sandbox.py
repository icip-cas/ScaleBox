import os
import sys
import signal
from utils.set_env import set_hf_cache
set_hf_cache()
import time
from tqdm.auto import tqdm
from datasets import load_dataset
import argparse
import logging
import json
import asyncio
import re
from openai import AsyncOpenAI as OpenAI

import numpy as np
from utils.vllm_runner import VLLMRunner
from utils.api_runner import APIRunner
from utils.multi_api_runner import MultiAPIRunner
from utils.vllm_server_manager import VLLMServerManager

from utils.template import get_template_data
from collections import defaultdict
import requests
import copy
import aiohttp

def test_assert(url):
    payload = {
        "completion": "```python\nimport re\ndef text_match_three(text):\n        patterns = 'ab{3}?'\n        return re.search(patterns,  text)\n```",
        "config": {
            "language": "python",
            "provided_data": { 
                "test_cases": {
                    "type": "assert", 
                    "test":  "def check(text_match_three):\n    assert not text_match_three(\"ac\")", 
                    "entry_point": "text_match_three",
                },            
            },
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        print(json.dumps(result, indent=2))
        assert result['accepted'] == True
    except requests.exceptions.Timeout:
        print(f"[Error] Sandbox endpoint {url} response timed out (30s)")
        raise
    except requests.exceptions.RequestException as e:
        print(f"[Error] Sandbox endpoint {url} request failed: {repr(e)}")
        raise
    except Exception as e:
        print(f"[Error] Sandbox endpoint {url} test failed: {repr(e)}")
        raise

async def get_sandbox_result(dataset_type, data, completion, config, url, session):
    payload = {}
    provided_data = {}
    payload["completion"] = completion
    config_copy = copy.deepcopy(config)

    if dataset_type == "MultiPLEDataset":
        config_copy["language"] = data['language']

    if dataset_type == "MBPPDataset":
        config_copy["language"] = "python"

    if dataset_type == "HumanEvalDataset":
        config_copy["language"] = "python"

    if dataset_type == "LiveCodeBenchDataset":
        config_copy["language"] = "python"

    if dataset_type == "AetherCodeDataset":
        config_copy["language"] = "cpp"
        config_copy['extra']['special_judge_program'] = data['checker']
        config_copy['extra']['special_judge_language'] = 'cpp'
        config_copy['extra']['force_special_judge'] = True

    provided_data["test_cases"] = data['test']
    config_copy["provided_data"] = provided_data
    payload["config"] = config_copy

    async with session.post(url, json=payload) as response:
        res = await response.json()
    return res 

MAX_CONCURRENCY = 8

async def _eval_one(raw_completion, info, args, data_i, config, session):
    if args.reasoning_model:
        completion = re.split(r"</think>\s*", raw_completion)[-1]
    else:
        completion = raw_completion

    outputlines = completion.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    
    if len(indexlines) < 2:
        if "```" not in completion and "def" in completion:
            if 'language' in data_i:
                completion = f"```{data_i['language']}\n"+completion+"\n```\n"
            else:
                completion = "```python\n"+completion+"\n```\n"
        else:
            completion = None
    else:
        # By default, take the last code block first in case no later block contains "def"
        completion = "\n".join(outputlines[indexlines[-2] : indexlines[-1] + 1])
        
        i = len(indexlines) - 1
        while i >= 1:
            start = indexlines[i - 1]
            end = indexlines[i]
            temp_completion = "\n".join(outputlines[start : end + 1])
            
            # Check whether it contains "def"
            if "def" in temp_completion:
                completion = temp_completion
                break
            
            # Continue searching in the previous code block
            i -= 2

    if completion is not None:
        try:
            res = await get_sandbox_result(info["datasetType"], data_i, completion, config, args.endpoint, session)
        except Exception as e:
            res = {'accepted': False, 'error': repr(e)}
    else:
        res = {'accepted': False}

    res['raw_id'] = data_i['id']
    return res

async def evaluate_all_async(results, info, data, config, args):
    results_sandbox = [[None] * len(completions) for completions in results]

    timeout = aiohttp.ClientTimeout(total=4000)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []

        for instance_idx, completions in enumerate(results):
            for sample_idx, raw_completion in enumerate(completions):
                async def _task(rc=raw_completion, idx=instance_idx, sidx=sample_idx):
                    async with sem:
                        res = await _eval_one(rc, info, args, data[idx], config, session)
                        return idx, sidx, res
                tasks.append(asyncio.create_task(_task()))

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Sandbox evaluation"):
            idx, sidx, res = await fut
            results_sandbox[idx][sidx] = res

    accepted_sandbox = [
        [r.get('accepted', False) for r in row]
        for row in results_sandbox
    ]

    return results_sandbox, accepted_sandbox

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default=None, help="Model name for code generation, use dataset provided code by default")
argparser.add_argument("--dataset_config", type=str, default="config/multi_humaneval_mbpp.json")
argparser.add_argument("--endpoint", type=str, default="http://0.0.0.0:8080")
argparser.add_argument("--prompt_type", type=str, default="chatml", help="Prompt type for the model")
argparser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
argparser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
argparser.add_argument("--top_k", type=int, default=20, help="Top-k sampling")
argparser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling")
argparser.add_argument("--max_completion_tokens", type=int, default=8192, help="Max new tokens for the model")
argparser.add_argument("--max_model_len", type=int, default=None, help="Max context length for vLLM server (default: same as max_completion_tokens * 2 or model's max)")
argparser.add_argument("--n_sample", type=int, default=1, help="Number of samples to generate for each instance")
argparser.add_argument("--stop_token", type=str, default='</s>,<|im_end|>,<|endoftext|>', help="Stop token for the model")
argparser.add_argument("--num_gpus_total", type=int, default=1, help="Total number of GPUs/NPUs")
argparser.add_argument("--num_gpus_per_model", type=int, default=1, help="Number of GPUs/NPUs per model")
argparser.add_argument("--npu", action="store_true", default=False, help="Use NPU instead of GPU")
argparser.add_argument("--reasoning_model", action="store_true", default=False, help="For reasoning model, remove text before '</think>'.")
argparser.add_argument("--output_dir", type=str, default="res/multi_language", help="Output directory for the results")
argparser.add_argument("--batch_size", type=int, default=0, help="Batch size for the model")
argparser.add_argument("--api_url", type=str, default=None, help="API URL for OpenAI-compatible endpoint (set to enable API)")
argparser.add_argument("--api_key", type=str, default="EMPTY", help="API key for authentication")
argparser.add_argument("--model_name", type=str, default="model", help="Model name to use in API calls")
argparser.add_argument("--rpm", type=int, default=0, help="API requests per minute limit (0 disables rate limiting)")
argparser.add_argument("--sample_only", action="store_true", default=False, help="Only sample without evaluation")
argparser.add_argument("--sample_file", type=str, default=None, help="Load samples from file instead of sampling (no new sampling)")
argparser.add_argument("--resume_from", type=str, default=None, help="Resume sampling from specified file, append new samples to output")
# vLLM Server mode arguments
argparser.add_argument("--use_vllm_server", action="store_true", default=False, help="Use vLLM server mode for multi-GPU deployment")
argparser.add_argument("--vllm_server_base_port", type=int, default=8000, help="Base port for vLLM servers")
argparser.add_argument("--vllm_server_host", type=str, default="0.0.0.0", help="Host for vLLM servers")
argparser.add_argument("--vllm_server_dtype", type=str, default="auto", help="Data type for vLLM server (auto/float16/bfloat16/float32)")
argparser.add_argument("--vllm_server_wait_timeout", type=int, default=600, help="Timeout for waiting vLLM server to be ready (seconds)")
argparser.add_argument("--mem_fraction", type=float, default=0.9, help="GPU/NPU memory utilization fraction (0.0-1.0)")
args = argparser.parse_args()

with open(args.dataset_config, "r", encoding="utf-8") as file:
    dataset_config = json.load(file)

# vLLM Server mode: start servers
vllm_server_manager = None
vllm_server_endpoints = []

# Define signal handler to ensure proper cleanup on Ctrl+C
def signal_handler(signum, frame):
    print("\n" + "=" * 60, flush=True)
    print("Termination signal received, cleaning up resources...", flush=True)
    if vllm_server_manager is not None:
        print("Shutting down vLLM Server...", flush=True)
        vllm_server_manager.stop_servers()
        print("vLLM Server stopped", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command

if args.use_vllm_server:
    print("=" * 60)
    print("Starting vLLM Server mode")
    print("=" * 60)
    
    # If model_path is not provided via CLI, read it from dataset_config
    model_path = args.model_path
    if not model_path:
        # Read model_path from the first infer_parameters entry of the first dataset
        for dataset_name, info in dataset_config.items():
            if "infer_parameters" in info and len(info["infer_parameters"]) > 0:
                model_path = info["infer_parameters"][0].get("model_path")
                if model_path:
                    print(f"Loaded model_path from config: {model_path}")
                    break
    
    if not model_path:
        raise ValueError("In vLLM Server mode, you must provide --model_path or set model_path in dataset_config")
    
    # Also read other vLLM Server parameters from config (if not specified via CLI)
    first_infer_params = None
    for dataset_name, info in dataset_config.items():
        if "infer_parameters" in info and len(info["infer_parameters"]) > 0:
            first_infer_params = info["infer_parameters"][0]
            break
    
    # Resolve parameters: CLI args > config values > defaults
    num_gpus_total = args.num_gpus_total if args.num_gpus_total != 1 else (first_infer_params.get("num_gpus_total", 1) if first_infer_params else 1)
    num_gpus_per_model = args.num_gpus_per_model if args.num_gpus_per_model != 1 else (first_infer_params.get("num_gpus_per_model", 1) if first_infer_params else 1)
    max_completion_tokens = args.max_completion_tokens if args.max_completion_tokens != 8192 else (first_infer_params.get("max_completion_tokens", 8192) if first_infer_params else 8192)
    batch_size = args.batch_size if args.batch_size != 0 else (first_infer_params.get("batch_size", 16) if first_infer_params else 16)
    
    # max_model_len: maximum context length for the vLLM service
    # Priority: CLI arg > config value > default (None, let vLLM auto-detect model max length)
    max_model_len = args.max_model_len
    if max_model_len is None and first_infer_params:
        max_model_len = first_infer_params.get("max_model_len", None)
    
    # Update args for subsequent usage
    args.model_path = model_path
    args.num_gpus_total = num_gpus_total
    args.num_gpus_per_model = num_gpus_per_model
    args.max_completion_tokens = max_completion_tokens
    args.batch_size = batch_size
    
    vllm_server_manager = VLLMServerManager(
        model_path=model_path,
        num_gpus_total=num_gpus_total,
        num_gpus_per_model=num_gpus_per_model,
        base_port=args.vllm_server_base_port,
        host=args.vllm_server_host,
        max_model_len=max_model_len,  # Use dedicated max_model_len; None lets vLLM auto-detect
        dtype=args.vllm_server_dtype,
        trust_remote_code=True,
        api_key=args.api_key,
        served_model_name=args.model_name,
        use_npu=args.npu,
        mem_fraction=args.mem_fraction,
        wait_timeout=args.vllm_server_wait_timeout,
    )
    
    print(f"vLLM Server configuration:")
    print(f"  max_model_len (server context length): {max_model_len if max_model_len else 'auto-detect'}")
    print(f"  max_completion_tokens (request generation length): {max_completion_tokens}")
    
    try:
        vllm_server_endpoints = vllm_server_manager.start_servers(wait_ready=True)
        print(f"vLLM Server started successfully with {len(vllm_server_endpoints)} instance(s)")
        for i, ep in enumerate(vllm_server_endpoints):
            print(f"  Instance {i}: {ep}")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to start vLLM Server: {e}")
        if vllm_server_manager:
            vllm_server_manager.stop_servers()
        raise

try:
    for dataset_name, info in dataset_config.items():

        for infer_parameters in info["infer_parameters"]:

            all_accepted_results = defaultdict(lambda: defaultdict(dict))

            for k, v in infer_parameters.items():
                setattr(args, k, v)
            
            test_assert(args.endpoint)

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            config = {
                'run_timeout': 10,
                'compile_timeout': 10,
            }
            if 'language' in info:
                config['language'] = info['language']
            if 'extra' in info:
                config['extra'] = info['extra']
            else:
                config['extra'] = {}
            config['extra']['total_timeout'] = 300
            config['extra']['run_all_cases'] = False

            for sub_dataset in info["datasets"]:
                if info["datasetType"] == "MultiPLEDataset":
                    dataset_idf = "multiple_" + sub_dataset["huggingFace"]["subset"].split("-")[-1]
                else:
                    dataset_idf = sub_dataset["dataset"]

                prompts, data = get_template_data(sub_dataset, info["datasetType"], args.prompt_type, args.reasoning_model)
                print("###len(data)###", len(data))

                if 'stop_tokens' in data[0]:
                    args.stop_token = ','.join(data[0]['stop_tokens'])
                print("###args.stop_token###", args.stop_token)

                # Define sampled output file path
                sample_output_path = os.path.join(args.output_dir, dataset_idf + "_samples.jsonl")
                
                # Define save callback (called after each prompt finishes sampling)
                def save_sample_callback(idx, samples):
                    """Save one prompt's sampled results, format: {"id": ..., "sample": [...]}."""
                    with open(sample_output_path, "a", encoding="utf-8") as f:
                        record = {
                            "id": data[idx]['id'],
                            "sample": samples
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Decide whether to load from file or run sampling
                if args.sample_file:
                    # Load sampled results from file, format: {"id": ..., "sample": [...]}
                    print(f"Loading sampled results from file: {args.sample_file}")
                    results_dict = {}
                    with open(args.sample_file, "r", encoding="utf-8") as f:
                        for line in f:
                            record = json.loads(line)
                            results_dict[record['id']] = record['sample']
                    
                    # Extract samples in the order of IDs in data
                    results = [results_dict.get(data_item['id'], []) for data_item in data]
                    
                else:
                    # If resume-from is specified
                    if args.resume_from:
                        if not os.path.exists(args.resume_from):
                            print(f"[Error] Resume file does not exist: {args.resume_from}")
                            exit(1)
                        
                        print(f"[Resume] Resuming from file: {args.resume_from}")
                        
                        # Load existing sampled results
                        existing_dict = {}
                        with open(args.resume_from, "r", encoding="utf-8") as f:
                            for line in f:
                                record = json.loads(line)
                                existing_dict[record['id']] = record['sample']
                        
                        # Build initial results
                        results = [existing_dict.get(data_item['id'], []) for data_item in data]
                        
                        # Find prompts that still need sampling
                        pending_indices = [i for i, r in enumerate(results) if len(r) == 0]
                        
                        print(f"[Resume] Completed: {len(data) - len(pending_indices)}/{len(data)} prompts")
                        print(f"[Resume] Pending: {len(pending_indices)} prompts")
                        
                        # Copy existing data into output file
                        with open(sample_output_path, "w", encoding="utf-8") as f:
                            for data_item in data:
                                if data_item['id'] in existing_dict:
                                    record = {
                                        "id": data_item['id'],
                                        "sample": existing_dict[data_item['id']]
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        
                        if len(pending_indices) == 0:
                            print(f"[Resume] All prompts are already completed, no further sampling needed")
                        else:
                            # Sample only unfinished prompts
                            pending_prompts = [prompts[i] for i in pending_indices]
                            
                            # Create callback for pending indices
                            def save_pending_callback(local_idx, samples):
                                """Save sampled results for a pending prompt."""
                                global_idx = pending_indices[local_idx]
                                with open(sample_output_path, "a", encoding="utf-8") as f:
                                    record = {
                                        "id": data[global_idx]['id'],
                                        "sample": samples
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            
                            if args.use_vllm_server:
                                runner = MultiAPIRunner(
                                    args=args,
                                    model=args.model_name,
                                    api_endpoints=vllm_server_endpoints,
                                    api_key=args.api_key,
                                )
                                pending_results = runner.run_batch(pending_prompts, save_callback=save_pending_callback)
                            elif args.api_url is not None:
                                runner = APIRunner(args, args.model_path)
                                pending_results = runner.run_batch(pending_prompts, save_callback=save_pending_callback)
                            else:
                                runner = VLLMRunner(args, args.model_path)
                                pending_results = runner.run_batch(pending_prompts, save_callback=save_pending_callback)
                            
                            # Merge results
                            for i, idx in enumerate(pending_indices):
                                results[idx] = pending_results[i]
                    else:
                        # Clear previous sample file (if it exists)
                        if os.path.exists(sample_output_path):
                            os.remove(sample_output_path)
                        
                        # Select API or VLLM based on arguments
                        if args.use_vllm_server:
                            runner = MultiAPIRunner(
                                args=args,
                                model=args.model_name,
                                api_endpoints=vllm_server_endpoints,
                                api_key=args.api_key,
                            )
                            results = runner.run_batch(prompts, save_callback=save_sample_callback)
                        elif args.api_url is not None:
                            runner = APIRunner(args, args.model_path)
                            results = runner.run_batch(prompts, save_callback=save_sample_callback)
                        else:
                            runner = VLLMRunner(args, args.model_path)
                            results = runner.run_batch(prompts, save_callback=save_sample_callback)
                
                # If sample-only mode is enabled, skip evaluation
                if args.sample_only:
                    print(f"Sampling completed, results saved to: {sample_output_path}")
                    continue
                
                # Run sandbox evaluation
                start = time.perf_counter()
                results_sandbox, accepted_sandbox = asyncio.run(evaluate_all_async(results, info, data, config, args))
                elapsed_s = time.perf_counter() - start
                elapsed_min = elapsed_s / 60
                print(f"Sandbox elapsed time: {elapsed_min:.2f} minutes")
                res_output_path = os.path.join(args.output_dir, dataset_idf + ".jsonl")
                with open(res_output_path, "w", encoding="utf-8") as f:
                    for res in results_sandbox:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                
                min_samples = min(len(acc) for acc in accepted_sandbox)
                avg_acc = 0
                for sample_idx in range(min_samples):
                    accepted_count = 0
                    for instance_idx in range(len(accepted_sandbox)):
                        if accepted_sandbox[instance_idx][sample_idx]:
                            accepted_count += 1
                    accuracy = accepted_count / len(accepted_sandbox)
                    avg_acc += accuracy
                    print(f"Sample {sample_idx} accuracy: {accuracy}")
                    all_accepted_results[dataset_name][sub_dataset['id']][sample_idx] = accuracy

                avg_acc /= min_samples
                all_accepted_results[dataset_name][sub_dataset['id']]["avg_acc"] = avg_acc

            with open(os.path.join(args.output_dir, "accuracy.json"), "w", encoding="utf-8") as f:
                json.dump(all_accepted_results, f, ensure_ascii=False, indent=4)

            for dataset_name in all_accepted_results:
                print(f"{dataset_name}")
                for sub_dataset in all_accepted_results[dataset_name]:
                    print(f"{sub_dataset}\t{all_accepted_results[dataset_name][sub_dataset]['avg_acc']}")
                print("--------------------------------")

finally:
    # Ensure vLLM server is properly closed
    if vllm_server_manager is not None:
        print("=" * 60)
        print("Shutting down vLLM Server...")
        vllm_server_manager.stop_servers()
        print("vLLM Server stopped")
        print("=" * 60)
