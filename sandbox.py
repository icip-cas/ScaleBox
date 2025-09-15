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
        "completion": "```python\nn = int(input())\na = list(map(int, input().split()))\nsorted_desc = sorted(a, reverse=True)\nsecond_largest = sorted_desc[1]\nindex = a.index(second_largest) + 1\nprint(index)\n```",
        "config": {
            "language": "python",
            "provided_data": { 
                "test_cases": {
                    "type": "stdin_stdout", 
                    "input": ["8\n1 2 3 4 5 10 9 11\n"], 
                    "output": ["6\n"],        
                    "fn_name": None, 
                }   
            },
            "extra": {
                "run_all_cases": True,
                "total_timeout": 10
            }
        }
    }
    response = requests.post(url, json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

# 实现并行
async def get_sandbox_result(dataset_type, data, completion, config, url, session):
    payload = {}
    provided_data = {}
    payload["completion"] = completion
    config_copy = copy.deepcopy(config)

    if dataset_type == "MultiPLEDataset":
        config_copy["language"] = data['language']
        
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
    # 保持你原先的后处理逻辑
    if args.reasoning_model:
        completion = re.split(r"</think>\s*", raw_completion)[-1]
    else:
        completion = raw_completion
    
    # 后处理部分
    # if info["datasetType"] == "MultiPLEDataset":
    #     m = re.search(r"```.*?```", completion, re.S)
    # else:
    #     m = re.search(r"```python\s*(.*?)```", completion, re.S)
    # completion = m.group(0) if m else None
    outputlines = completion.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        completion = None
    else:
        start = indexlines[-2]
        end = indexlines[-1]
        completion = "\n".join(outputlines[start : end + 1])

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

    timeout = aiohttp.ClientTimeout(total=60)  # 按需调整
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for instance_idx, completions in enumerate(results):
            # 针对同一个 instance 内的样本并行跑
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

# 用于收集所有子数据集的accepted结果
all_accepted_results = defaultdict(lambda: defaultdict(dict))

for dataset_name, info in dataset_config.items():

    for k, v in info["infer_parameters"].items():
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
        # 这里dataset_idf指明是什么语言种类的数据 idf是identifier(标识符)
        if info["datasetType"] == "MultiPLEDataset": # 这里是如果评测数据是多语言数据
            dataset_idf = "multiple_" + sub_dataset["huggingFace"]["subset"].split("-")[-1]
        else:
            dataset_idf = sub_dataset["dataset"]

        prompts, data = get_template_data(sub_dataset, info["datasetType"], args.prompt_type, args.reasoning_model)
        print("###len(data)###", len(data))

        # 上面数据已经处理好了,接下来是开始调用model进行推理

        # 如果data[0]中有stop_tokens，则将stop_tokens保存到args.stop_token
        if 'stop_tokens' in data[0]:
            args.stop_token = ','.join(data[0]['stop_tokens'])
        print("###args.stop_token###",args.stop_token)

        runner = VLLMRunner(args, args.model_path)
        results = runner.run_batch(prompts) # [[sample 1, ... ,sample n],[]]
        print("####推理完成####")
        
        start = time.perf_counter()
        results_sandbox, accepted_sandbox = asyncio.run(evaluate_all_async(results, info, data, config, args))
        elapsed_s = time.perf_counter() - start
        elapsed_min = elapsed_s / 60
        print(f"sandbox耗时:{elapsed_min:.2f} 分钟")
        # # 数据后处理并调用sandbox进行评测
        # results_sandbox = []
        # accepted_sandbox = []
        # for instance_idx, completions in enumerate(results):
        #     tmp_res = []
        #     tmp_accepted = []
        #     for sample_idx, raw_completion in enumerate(completions):

        #         if args.reasoning_model:
        #             # For reasoning model, remove text before '</think>'.
        #             completion = re.sub(r".*</think>\n*", "", raw_completion, flags=re.DOTALL)
        #         else:
        #             completion = raw_completion
                
        #         if info["datasetType"] == "MultiPLEDataset":
        #             m = re.search(r"```.*?```", completion, re.S)
        #             completion = m.group(0) if m else None
        #         else:
        #             m = re.search(r"```python\s*(.*?)```", completion, re.S)
        #             completion = m.group(0) if m else None

        #         if completion is not None:
        #             res = get_sandbox_result(info["datasetType"], data[instance_idx], completion, config, args.endpoint)
        #         else:
        #             res = {'accepted': False}
        #         res['llm_raw_completion'] = raw_completion

        #         tmp_accepted.append(res['accepted'])
        #         tmp_res.append(res)

        #     results_sandbox.append(tmp_res)
        #     accepted_sandbox.append(tmp_accepted)

        # 将results_sandbox保存到output_path
        res_output_path = os.path.join(args.output_dir, dataset_idf + ".jsonl")
        with open(res_output_path, "w", encoding="utf-8") as f:
            for res in results_sandbox:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        
        # 计算accepted_sandbox的准确率
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

# 最后将all_accepted_results保存到文件
with open(os.path.join(args.output_dir, "accuracy.json"), "w", encoding="utf-8") as f:
    json.dump(all_accepted_results, f, ensure_ascii=False, indent=4)

# 最后将all_accepted_results中所有dataset_name中的sub_dataset中的avg_acc在命令行中显示出来
for dataset_name in all_accepted_results:
    print(f"{dataset_name}")
    for sub_dataset in all_accepted_results[dataset_name]:
        print(f"{sub_dataset}\t{all_accepted_results[dataset_name][sub_dataset]['avg_acc']}")
    print("--------------------------------")


