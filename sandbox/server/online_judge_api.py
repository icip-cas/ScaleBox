# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, List, Optional

import json

from fastapi import APIRouter, HTTPException

from datetime import datetime

import os

import tempfile

from sandbox.datasets.types import (
    CodingDataset,
    EvalResult,
    GetMetricsFunctionRequest,
    GetMetricsFunctionResult,
    GetMetricsRequest,
    GetPromptByIdRequest,
    GetPromptsRequest,
    Prompt,
    SubmitRequest,
    TestConfig,
    GeneralStdioTest,
)
from sandbox.registry import get_all_dataset_ids, get_coding_class_by_dataset, get_coding_class_by_name

from sandbox.utils.common import ensure_json
from sandbox.utils.extraction import default_extract_helper
from sandbox.utils.testing import check_stdio_test_cases_parallel, check_function_call_test_cases_parallel
from sandbox.database import get_row_by_id_in_table

oj_router = APIRouter()


def get_dataset_cls(dataset_id: str, config: Optional[TestConfig] = None) -> CodingDataset:
    internal_cls = get_coding_class_by_dataset(dataset_id)
    if internal_cls is not None:
        return internal_cls
    if config is None or config.dataset_type is None:
        raise HTTPException(status_code=400, detail=f'no eval class found for dataset {dataset_id}')
    config_cls = get_coding_class_by_name(config.dataset_type)
    if config_cls is None:
        raise HTTPException(status_code=400, detail=f'eval class {config.dataset_type} not found')
    return config_cls


@oj_router.get("/list_datasets", description='List all registered datasets', tags=['datasets'])
async def list_datasets() -> List[str]:
    return get_all_dataset_ids()


@oj_router.post("/list_ids", description='List all ids of a dataset', tags=['datasets'])
async def list_ids(request: GetPromptsRequest) -> List[int | str]:
    dataset = get_dataset_cls(request.dataset, request.config)
    ids = await dataset.get_ids(request)
    return ids


@oj_router.post("/get_prompts", description='Get prompts of a dataset', tags=['datasets'])
async def get_prompt(request: GetPromptsRequest) -> List[Prompt]:
    dataset = get_dataset_cls(request.dataset, request.config)
    prompts = await dataset.get_prompts(request)
    return prompts


@oj_router.post("/get_prompt_by_id", description='Get a single prompt given id in a dataset', tags=['datasets'])
async def get_prompt_by_id(request: GetPromptByIdRequest) -> Prompt:
    dataset = get_dataset_cls(request.dataset, request.config)
    prompt = await dataset.get_prompt_by_id(request)
    return prompt


@oj_router.post("/submit", description='Submit a single problem in a dataset', tags=['datasets'])
async def submit(request: SubmitRequest) -> EvalResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    result = await dataset.evaluate_single(request)
    
    # if os.getenv('SAVE_BAD_CASES') == 'true':
    # if all([o.exec_info.status == 'SandboxError' for o in result.tests]):
    # Try to write the code to folder `output/{current_date}/`
    if os.getenv('SAVE_BAD_CASES', default='') == 'true' and any([o.exec_info.status == 'SandboxError' for o in result.tests]):
        # Create the folder if it does not exist
        os.makedirs('output', exist_ok=True)
        output_dir = os.path.join('output', datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)
        # Write the code to a file
        with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, suffix='.json', delete=False) as f:
            f.write(result.model_dump_json(indent=2))

    return result

def convert_stdio_data_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert 
    {'test_cases': [{'type': 'stdin_stdout', 'input': 'n = 6\r\nA[] = {16,17,4,3,5,2}', 'output': '17 5 2'}, {'type': 'stdin_stdout', 'input': 'n = 5\r\nA[] = {1,2,3,4,0}', 'output': '4 0'}]}
    to the format 
    {
        "test": [                           # Test data format for "Standard" problems
        {
            "input": {                  # Input data, filename -> content, stdin is the standard stream
                "stdin": "xxx"
            },
            "output": {                 # Expected output, filename -> content 
                "stdout": "xxx"
            }
        },
        ...
        ]
    }
    """
    test_cases = data['test_cases']
    test = []
    for case in test_cases:
        test.append({
            "input": {
                "stdin": case['input']
            },
            "output": {
                "stdout": case['output']
            }
        })
    return {'test': test}


@oj_router.post("/common_evaluate", description='Submit a single problem and evaluate in the form of stdio common judge or function eval', tags=['datasets'])
async def common_evaluate(request: SubmitRequest) -> EvalResult:
    if not request.config.language:
        raise HTTPException(status_code=400,
                            detail=f'config.language field must exist, got None')
    if request.config.language in ['javascript', 'js']:
        request.config.language = 'nodejs'
    code = default_extract_helper(request.completion, request.config.language, request.config.custom_extract_logic)

    data_format = "stdin_stdout"
    if 'test_cases' in request.config.provided_data:
        first_case = request.config.provided_data['test_cases'][0]
        if 'type' in first_case:
            data_format = first_case['type']
        elif 'fn_name' in first_case and first_case['fn_name'] is not None:
            data_format = "function_call"

    if data_format == "stdin_stdout":
        if 'test_cases' in request.config.provided_data:
            request.config.provided_data = convert_stdio_data_format(request.config.provided_data)
        row = await get_row_by_id_in_table(request, table_name=None, columns=['test'])
        cases = [GeneralStdioTest(**case) for case in ensure_json(row, 'test')]
        outcomes = await check_stdio_test_cases_parallel(code, cases, request.config)
    # function call eval
    else:
        cases = request.config.provided_data['test_cases']
        outcomes = await check_function_call_test_cases_parallel(code, cases, request.config)
    
    result = EvalResult(id=request.id,
                            accepted=all([o.passed for o in outcomes]),
                            extracted_code=code,
                            tests=outcomes)
    
    # if os.getenv('SAVE_BAD_CASES') == 'true':
    # if all([o.exec_info.status == 'SandboxError' for o in outcomes]):
    # Try to write the code to folder `output/{current_date}/`
    if os.getenv('SAVE_BAD_CASES', default='') == 'true' and any([o.exec_info.status == 'SandboxError' for o in outcomes]):
        # Create the folder if it does not exist
        os.makedirs('output', exist_ok=True)
        output_dir = os.path.join('output', datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)
        # Write the code to a file
        with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, suffix='.json', delete=False) as f:
            f.write(result.model_dump_json(indent=2))

    return result


def convert_batch_data_format(data: Dict[str, Any], data_format='stdin_stdout') -> Dict[str, Any]:
    """
    Convert 
    {'test_cases': {'type': 'xxx', 'fn_name': 'xxx', 'input': ['n = 6\r\nA[] = {16,17,4,3,5,2}', 'n = 5\r\nA[] = {1,2,3,4,0}'], 'output': ['17 5 2', '4 0']}}
    to the format 
    {'test_cases': [{'type': 'xxx', 'input': 'n = 6\r\nA[] = {16,17,4,3,5,2}', 'output': '17 5 2'}, {'type': 'xxx', 'input': 'n = 5\r\nA[] = {1,2,3,4,0}', 'output': '4 0'}]}
    """
    test_cases = data['test_cases']
    test = []
    json_input = test_cases.get('json_input', False)
    for i in range(len(test_cases['input'])):
        the_input = test_cases['input'][i]
        the_output = test_cases['output'][i]
        if isinstance(the_input, str) and data_format == 'function_call' and json_input is True:
            # For json_input data, if the input is a string, we need to load it as json and convert it to a list
            try:
                the_input = [json.loads(line) for line in the_input.split("\n")]
                the_output = [json.loads(the_output)]
            except json.JSONDecodeError:
                pass

        test.append({
            "type": test_cases['type'],
            "fn_name": test_cases['fn_name'],
            "input": the_input,
            "output": the_output,
        })
    return {'test_cases': test}


dataset_map = {
    'python': 'humaneval_python',
    'cpp': 'multiple_cpp',
    'typescript': 'multiple_ts',
    'bash': 'multiple_sh',
    'csharp': 'multiple_cs',
    'go': 'multiple_go',
    'java': 'multiple_java',
    'lua': 'multiple_lua',
    'javascript': 'multiple_js',
    'nodejs': 'multiple_js',
    'php': 'multiple_php',
    'perl': 'multiple_pl',
    'racket': 'multiple_rkt',
    'r': 'multiple_r',
    'rust': 'multiple_rs',
    'scala': 'multiple_scala',
    'swift': 'multiple_swift',
    'ruby': 'multiple_rb',
    'd': 'multiple_d',
    'julia': 'multiple_jl',
}

@oj_router.post("/common_evaluate_batch", description='Submit a single problem with a batch of test cases and evaluate in the form of stdio common judge or function eval', tags=['datasets'])
async def common_evaluate_batch(request: SubmitRequest) -> EvalResult:
    if not request.config.language:
        raise HTTPException(status_code=400,
                            detail=f'config.language field must exist, got None')
    if request.config.language == 'javascript':
        request.config.language = 'nodejs'
    code = default_extract_helper(request.completion, request.config.language, request.config.custom_extract_logic)

    data_format = "stdin_stdout"
    case = request.config.provided_data['test_cases']
    if 'type' in case:
        data_format = case['type']
    elif 'fn_name' in case and case['fn_name'] is not None:
        data_format = "function_call"
    if data_format == "assert":
        dataset = dataset_map.get(request.config.language, 'multiple_python')
        request.dataset = dataset
        logging.info(f"Using dataset {dataset} for language {request.config.language}")
        dataset = get_dataset_cls(dataset, request.config)
        for key, value in request.config.provided_data['test_cases'].items():
            request.config.provided_data[key] = value
        request.config.extra['is_freeform'] = True
        result = await dataset.evaluate_single(request)
    else:
        request.config.provided_data = convert_batch_data_format(request.config.provided_data, data_format)
        if data_format == "stdin_stdout":
            request.config.provided_data = convert_stdio_data_format(request.config.provided_data)
            row = await get_row_by_id_in_table(request, table_name=None, columns=['test'])
            cases = [GeneralStdioTest(**case) for case in ensure_json(row, 'test')]
            outcomes = await check_stdio_test_cases_parallel(code, cases, request.config)
        # function call eval
        else:
            cases = request.config.provided_data['test_cases']
            outcomes = await check_function_call_test_cases_parallel(code, cases, request.config)
        
        result = EvalResult(id=request.id,
                                accepted=all([o.passed for o in outcomes]),
                                extracted_code=code,
                                tests=outcomes)
    
    # if os.getenv('SAVE_BAD_CASES') == 'true':
    # if all([o.exec_info.status == 'SandboxError' for o in outcomes]):
    # Try to write the code to folder `output/{current_date}/`
    if os.getenv('SAVE_BAD_CASES', default='') == 'true' and any([o.exec_info.status == 'SandboxError' for o in outcomes]):
        # Create the folder if it does not exist
        os.makedirs('output', exist_ok=True)
        output_dir = os.path.join('output', datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)
        # Write the code to a file
        with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, suffix='.json', delete=False) as f:
            f.write(result.model_dump_json(indent=2))

    return result


@oj_router.post("/get_metrics",
                description='Get the metrics given all problem results in a dataset (partially supported)',
                tags=['datasets'])
async def get_metrics(request: GetMetricsRequest) -> Dict[str, Any]:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, 'get_metrics'):
        result = await dataset.get_metrics(request.results)
        return result
    else:
        return {}


@oj_router.post("/get_metrics_function",
                description='Get the function to generate the metrics given results (partially supported)',
                tags=['datasets'])
async def get_metrics_function(request: GetMetricsFunctionRequest) -> GetMetricsFunctionResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, 'get_metrics_function'):
        func = dataset.get_metrics_function()
        return GetMetricsFunctionResult(function=func)
    else:
        return GetMetricsFunctionResult(function=None)

if __name__ == "__main__":
    dataset = get_coding_class_by_dataset("code_contests")
    print("dataset", dataset)