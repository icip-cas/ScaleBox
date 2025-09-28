import os

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
from utils.template import get_template_data
from collections import defaultdict
import asyncio
import requests
import copy
import asyncio
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
    response = requests.post(url, json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

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

    provided_data["test_cases"] = data['test']
    config_copy["provided_data"] = provided_data
    payload["config"] = config_copy

    async with session.post(url, json=payload) as response:
        res = await response.json()
    return res 

MAX_CONCURRENCY = 32

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
        for i in range(len(indexlines) - 1, 0, -1):
            start = indexlines[i-1]
            end = indexlines[i]
            temp_completion = "\n".join(outputlines[start : end + 1])
            
            if "def" in temp_completion:
                completion = temp_completion
                break

    if completion is not None:
        try:
            res = await get_sandbox_result(info["datasetType"], data_i, completion, config, args.endpoint, session)
        except Exception as e:
            res = {'accepted': False, 'error': repr(e)}
    else:
        res = {'accepted': False}

    res['llm_raw_completion'] = raw_completion
    return res

async def evaluate_all_async(results, info, data, config, args):
    results_sandbox = []
    accepted_sandbox = []

    timeout = aiohttp.ClientTimeout(total=60)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for instance_idx, completions in enumerate(results):
            tasks = []
            for raw_completion in completions:
                async def _task(rc=raw_completion, idx=instance_idx):
                    async with sem:
                        return await _eval_one(rc, info, args, data[idx], config, session)
                tasks.append(asyncio.create_task(_task()))

            tmp_res = await asyncio.gather(*tasks)
            tmp_accepted = [r.get('accepted', False) for r in tmp_res]

            results_sandbox.append(tmp_res)
            accepted_sandbox.append(tmp_accepted)

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
argparser.add_argument("--n_sample", type=int, default=1, help="Number of samples to generate for each instance")
argparser.add_argument("--stop_token", type=str, default='</s>,<|im_end|>,<|endoftext|>', help="Stop token for the model")
argparser.add_argument("--num_gpus_total", type=int, default=1, help="Total number of GPUs")
argparser.add_argument("--num_gpus_per_model", type=int, default=1, help="Number of GPUs per model")
argparser.add_argument("--reasoning_model", action="store_true", default=False, help="For reasoning model, remove text before '</think>'.")
argparser.add_argument("--output_dir", type=str, default="res/multi_language", help="Output directory for the results")
args = argparser.parse_args()

with open(args.dataset_config, "r", encoding="utf-8") as file:
    dataset_config = json.load(file)

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
        config['extra']['total_timeout'] = 8
        config['extra']['run_all_cases'] = True

        for sub_dataset in info["datasets"]:
            if info["datasetType"] == "MultiPLEDataset":
                dataset_idf = "multiple_" + sub_dataset["huggingFace"]["subset"].split("-")[-1]
            else:
                dataset_idf = sub_dataset["dataset"]

            prompts, data = get_template_data(sub_dataset, info["datasetType"], args.prompt_type, args.reasoning_model)
            print("###len(data)###", len(data))

            if 'stop_tokens' in data[0]:
                args.stop_token = ','.join(data[0]['stop_tokens'])
            print("###args.stop_token###",args.stop_token)

            runner = VLLMRunner(args, args.model_path)
            results = runner.run_batch(prompts) # [[sample 1, ... ,sample n],[]]
            
            start = time.perf_counter()
            results_sandbox, accepted_sandbox = asyncio.run(evaluate_all_async(results, info, data, config, args))
            elapsed_s = time.perf_counter() - start
            elapsed_min = elapsed_s / 60
            print(f"sandbox耗时:{elapsed_min:.2f} 分钟")
            res_output_path = os.path.join(args.output_dir, dataset_idf + ".jsonl")
            with open(res_output_path, "w", encoding="utf-8") as f:
                for res in results_sandbox:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            avg_acc = 0
            for sample_idx in range(len(accepted_sandbox[0])):
                accepted_count = 0
                for instance_idx in range(len(accepted_sandbox)):
                    if accepted_sandbox[instance_idx][sample_idx]:
                        accepted_count += 1
                accuracy = accepted_count / len(accepted_sandbox)
                avg_acc += accuracy
                print(f"Sample {sample_idx} accuracy: {accuracy}")
                all_accepted_results[dataset_name][sub_dataset['id']][sample_idx] = accuracy

            avg_acc /= len(accepted_sandbox[0])
            all_accepted_results[dataset_name][sub_dataset['id']]["avg_acc"] = avg_acc

        with open(os.path.join(args.output_dir, "accuracy.json"), "w", encoding="utf-8") as f:
            json.dump(all_accepted_results, f, ensure_ascii=False, indent=4)

        for dataset_name in all_accepted_results:
            print(f"{dataset_name}")
            for sub_dataset in all_accepted_results[dataset_name]:
                print(f"{sub_dataset}\t{all_accepted_results[dataset_name][sub_dataset]['avg_acc']}")
            print("--------------------------------")