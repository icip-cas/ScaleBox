from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from datasets import load_dataset
from utils.livecodebench.generation import load_code_generation_dataset, load_code_cpp_generation_dataset
import os
import re
import polars as pl
import re
import ast

class Role(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "gpt"

@dataclass
class ConversationTemplate:
    name: str
    role_starts: Optional[Dict[Role, str]] = None
    role_ends: Optional[Dict[Role, str]] = None
    offset: Optional[int] = 0
    default_system_message: Optional[str] = None
    stop_str: Optional[str] = None

    def get_attributes(self) -> Dict:
        return {
            "name": self.name,
            "role_starts": self.role_starts,
            "role_ends": self.role_ends,
            "offset": self.offset,
            "default_system_message": self.default_system_message,
        }

language_mappings = {
    "cs": "csharp",
    "jl": "julia",
    "js": "nodejs",
    "pl": "perl",
    "rb": "ruby",
    "rkt": "racket",
    "rs": "rust",
    "sh": "bash",
    "ts": "typescript",
    "go_test.go": "go",
    "r": "R",
    "d": "D_ut"
}

TEMPLATES = {
    "chatml": ConversationTemplate(
        name="chatml",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "chatml_qwen3": ConversationTemplate(
        name="chatml_qwen3",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "deepseek": ConversationTemplate(
        name="deepseek",
        role_starts={
            Role.SYSTEM: "<｜begin▁of▁sentence｜>",
            Role.HUMAN: "<｜User｜>",
            Role.ASSISTANT: "<｜Assistant｜>",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "",
            Role.ASSISTANT: "<｜end▁of▁sentence｜>",
        },
        default_system_message="",
        offset=0,
        stop_str="<｜end▁of▁sentence｜>",
    ),
    "llama-3-instruct": ConversationTemplate(
        name="llama-3-instruct",
        role_starts={
            Role.SYSTEM: "<|start_header_id|>system<|end_header_id|>\n\n",
            Role.HUMAN: "<|start_header_id|>user<|end_header_id|>\n\n",
            Role.ASSISTANT: "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        role_ends={
            Role.SYSTEM: "<|eot_id|>",
            Role.HUMAN: "<|eot_id|>",
            Role.ASSISTANT: "<|eot_id|>",
        },
        default_system_message="",
        offset=0,
        stop_str="<|eot_id|>",
    ),
}

# load lcb dataset
def load_lcb_dataset(dataset):
    raw_data = load_code_generation_dataset(
        release_version=f'release_{dataset["version"]}',
        start_date=dataset["begin_date"],
        end_date=dataset["end_date"],
    )
    data = []
    for id, sample in enumerate(raw_data):
        data.append({
            "id": id+1,
            "raw_data": sample,
            "content": sample.question_content,
            "test": sample.get_evaluation_sample(),
        })
    return data

def load_lcb_cpp_dataset(dataset):
    raw_data = load_code_cpp_generation_dataset(
        release_version=f'release_{dataset["version"]}',
        start_date=dataset["begin_date"],
        end_date=dataset["end_date"],
    )
    data = []
    for id, sample in enumerate(raw_data):
        data.append({
            "id": id+1,
            "raw_data": sample,
            "content": sample.question_content,
            "test": sample.get_evaluation_sample(),
        })
    return data

def load_multiple_dataset(dataset):
    raw_data = load_dataset(
        "json",
        data_files=f"data/MultiPL-E/{dataset['huggingFace']['subset']}.jsonl"
    )["train"]
    data = []
    for id, sample in enumerate(raw_data):
        language = sample['language']
        if language in language_mappings:
            language = language_mappings[language]
        data.append({
            "id": id+1,
            "raw_data": sample,
            "prompt": sample['prompt'],
            "language": language,
            "test": {"type": "assert", "tests": sample['tests'], "stop_tokens": sample['stop_tokens']},
        })
    return data

def convert_test_format(test_code: str, entry_point: str, use_set: bool = False) -> str:
    """
    将原始测试代码格式转换为目标格式
    
    原始格式:
        inputs = [[(arg1), (arg2)], ...]
        results = [(result1), (result2), ...]
        for i, (inp, exp) in enumerate(zip(inputs, results)):
            assertion(func_name(*inp), exp, 0)
    
    目标格式:
        def check(func_name):
            assert set(func_name(arg1, arg2)) == set(result1)  # use_set=True
            assert func_name(arg1, arg2) == result1            # use_set=False
            ...
        check(func_name)
    
    参数:
        test_code: 原始测试代码
        entry_point: 函数名
        use_set: 是否使用 set() 包装，通过 sample['test_list'][0] 中是否有 'set' 判断
    """
    try:
        # 提取 inputs
        inputs_match = re.search(r'inputs\s*=\s*(\[.*?\])\s*\nresults', test_code, re.DOTALL)
        if not inputs_match:
            return test_code  # 如果格式不匹配，返回原始代码
        inputs_str = inputs_match.group(1)
        
        # 提取 results
        results_match = re.search(r'results\s*=\s*(\[.*?\])\s*\n(?:for|$)', test_code, re.DOTALL)
        if not results_match:
            return test_code
        results_str = results_match.group(1)

        # 使用安全的 eval 环境解析列表（支持 inf, nan, set 等）
        safe_globals = {
            "inf": float("inf"),
            "nan": float("nan"),
            "True": True,
            "False": False,
            "None": None,
            "set": set,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "frozenset": frozenset,
        }
        inputs = eval(inputs_str, {"__builtins__": {}}, safe_globals)
        results = eval(results_str, {"__builtins__": {}}, safe_globals)
        
        # 生成新格式
        lines = [f"def check({entry_point}):"]
        for inp, exp in zip(inputs, results):
            args_str = ", ".join(repr(arg) for arg in inp)
            
            if use_set:
                lines.append(f"    assert set({entry_point}({args_str})) == set({repr(exp)})")
            else:
                lines.append(f"    assert {entry_point}({args_str}) == {repr(exp)}")
        
        lines.append(f"check({entry_point})")
        
        return "\n".join(lines)
    
    except Exception as e:
        print(f"Warning: Failed to convert test format: {e}")
        return test_code  # 转换失败时返回原始代码


def load_mbpp_dataset(dataset):
    if dataset['id'] == 'mbpp':
        raw_data = load_dataset(
            "json",
            data_files=f"data/FusedMBPP/mbpp.jsonl"
        )["train"]
    elif dataset['id'] == 'mbppplus':
        raw_data = load_dataset(
            "json",
            data_files=f"data/FusedMBPP/mbppplus.jsonl"
        )["train"]
    else:
        raise ValueError(f"Invalid dataset id: {dataset['id']}")
    
    data = []
    for id, sample in enumerate(raw_data):
        # 提取 entry_point
        if "math.isclose" in sample['test_list'][0]:
            entry_point = re.search(r"math\.isclose\((\w+)\(", sample['test_list'][0]).group(1)
        elif "text_match_three" in sample['test_list'][0]:
            entry_point = "text_match_three"
        elif "is_perfect_square" in sample['test_list'][0]:
            entry_point = "is_perfect_square"
        elif "similar_elements" in sample['test_list'][0]:
            entry_point = "similar_elements"
        elif "find_char_long" in sample['test_list'][0]:
            entry_point = "find_char_long"
        elif "common_in_nested_lists" in sample['test_list'][0]:
            entry_point = "common_in_nested_lists"
        elif "extract_singly" in sample['test_list'][0]:
            entry_point = "extract_singly"
        elif "larg_nnum" in sample['test_list'][0]:
            entry_point = "larg_nnum" 
        elif "Diff" in sample['test_list'][0]:
            entry_point = "Diff"
        elif "max_height" in sample['test_list'][0]:
            entry_point = "max_height"
        else:
            entry_point = re.search(r'assert\s+([A-Za-z_]\w*)\s*\(', sample['test_list'][0]).group(1)
        
        # 构建 test
        test = "def check(" + entry_point + "):\n    "
        test += "\n    ".join(sample['test_list'])
        
        # mbppplus 格式转换
        if dataset['id'] == 'mbppplus':
            use_set = 'set(' in sample['test_list'][0]
            test = convert_test_format(sample['test'], entry_point, use_set)
        
        # 提取 prefix_template
        lines = sample['code'].split('\n')
        for i, line in enumerate(lines):
            if entry_point in line:
                def_line_index = i
                break
        result_lines = lines[def_line_index:def_line_index + 1]
        prefix_template = '\n'.join(result_lines)
        
        data.append({
            "id": id + 1,
            "raw_data": sample,
            "content": sample['prompt'],
            "test": {"type": "assert", "test": test, "entry_point": entry_point},
            "prefix_template": prefix_template,
        })
    
    return data

def load_humaneval_dataset(dataset):
    if dataset['id'] == 'humaneval':
        raw_data = load_dataset(
            "json",
            data_files=f"data/openai_humaneval/humaneval.jsonl"
        )["train"]
    elif dataset['id'] == 'humanevalplus':
        raw_data = load_dataset(
            "json",
            data_files=f"data/openai_humaneval/humanevalplus.jsonl"
        )["train"]
    else:
        raise ValueError(f"Invalid dataset id: {dataset['id']}")
    data = []
    for id, sample in enumerate(raw_data):
        data.append({
            "id": id+1,
            "raw_data": sample,
            "prompt": sample['prompt'],
            "test": {"type": "assert", "test": sample['test'], "entry_point": sample['entry_point']},
        })
    return data

def load_aethercode_dataset(dataset):
    # 由于aethercode数据是parquet类型，直接使用load_dataset会遇到parquet的bug，所以使用polars.read_parquet来加载数据
    version = dataset['version'].strip('"').split(',')
    raw_data = []
    for v in version:
        df = pl.read_parquet(f"./AetherCode/{v}/test-*.parquet")
        for i in df.iter_rows(named=True):
            raw_data.append({'id':i['id'],'prompt':i['description'],'test_cases':i['test_cases']})
    
    # # 从 jsonl 文件读取数据
    # import json
    # raw_data = []
    
    # # 读取需要保留的 ID 列表
    # zero_true_ids_path = "/141nfs/wangpengbo/sandbox/sandbox_eval/res/aethercode_deepseek_reasoner/zero_true_ids.txt"
    # with open(zero_true_ids_path, "r") as f:
    #     target_ids = set(line.strip() for line in f if line.strip())
    
    # jsonl_path = "/141nfs/wangpengbo/aethercode/AetherCode/generate_judge_program/data/v1_2024_classified_special_judge.jsonl"
    # with open(jsonl_path, "r") as f:
    #     for line in f:
    #         item = json.loads(line)
    #         # # 只保留 ID 在 zero_true_ids.txt 中的数据
    #         # if str(item['id']) in target_ids:
    #         raw_data.append({'id': item['id'], 'prompt': item['description'], 'test_cases': item['test_cases']})
    
    import json
    special_judge_file = dataset.get("special_judge_file")
    if not special_judge_file:
        candidate_paths = [
            "./AetherCode/checker.jsonl",
            "./eval/data/checker.jsonl",
            "./data/checker.jsonl",
        ]
        checker_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                checker_path = path
                break
        if checker_path is None:
            raise FileNotFoundError(
                "Cannot find default checker.jsonl. Please provide `special_judge_file` in config "
                "or place checker file in one of: ./AetherCode/checker.jsonl, ./eval/data/checker.jsonl, ./data/checker.jsonl"
            )

        checker_data = []
        with open(checker_path, "r") as f:
            for line in f:
                checker_data.append(json.loads(line))

        checker_dict = {item['id']: item['checker'] for item in checker_data}
    else:
        checker_data = []
        with open(special_judge_file, "r") as f:
            for line in f:
                checker_data.append(json.loads(line))

        checker_dict = {item['problem_id']: item['special_judge_program'] for item in checker_data}
    
    data = []
    for id, sample in enumerate(raw_data):
        checker = checker_dict.get(sample['id'])
        if checker is None:
            continue
        
        input = []
        output = []
        for t in sample['test_cases']:
            input.append(t['input'])
            output.append(t['output'])
        data.append({
            "id": sample['id'],
            "raw_data": sample,
            "prompt": sample['prompt'],
            "test": {"type": "stdin_stdout", "input": input,"output": output, "fn_name": None},
            "checker": checker,
        })
    print("###len(data)###",len(data))
    return data

# 不同数据集的模板
def get_lcb_prompt(
    question, prompt_type, think
) -> str:
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    if prompt_type != "chatml_qwen3":
        full_prompt += "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\nQuestion: "
    else:
        full_prompt += "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nQuestion: "

    full_prompt += question.question_content + '\n\n'
    if question.starter_code:
        full_prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        full_prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        full_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        full_prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if think:
        full_prompt += '<think>\n'

    return full_prompt

def get_lcb_cpp_prompt(
    question, prompt_type, think
) -> str:
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the cpp program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    if prompt_type != "chatml_qwen3":
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\nQuestion: "
    else:
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests.\n\nQuestion: "
    full_prompt += question.question_content + '\n\n'
    if question.starter_code:
        full_prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        full_prompt += f"```cpp\n{question.starter_code}\n```\n\n"
    else:
        full_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        full_prompt += f"```cpp\n# YOUR CODE HERE\n```\n\n"
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if think:
        full_prompt += '<think>\n'

    return full_prompt

# mbpp模板有问题 应该告诉模型程序以什么函数名开头（这里通过用例子告诉模型函数名） 此外还要告诉模型不要写注释
def get_mbpp_prompt(
    instance, prompt_type, think
) -> str:
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\"\"\"\n"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    test_list = "\n".join(instance['raw_data']['test_list'])
    # full_prompt += f"You are an expert Python programmer, and here is your task: {instance['content']} Your code should pass these tests:\n\n{instance['raw_data']['test_list'][0]}"
    # full_prompt += f"Please provide a self-contained Python script that solves the following problem in a markdown code block: {instance['content']} Your code should pass these tests:\n\n{instance['raw_data']['test_list'][0]}"
    # full_prompt += f"You are an expert Python programmer, and here is your task: {instance['content']} You will use the following starter code: {instance['prefix_template']}"
    # full_prompt += f"You are an expert Python programmer, and here is your task: ```python\n{instance['content']}\n```\n\nYour code should pass these tests:\n\n{test_list}\n\nYou should submit your final solution in the following format: ```python\n\n```"

    # full_prompt += instruction_prefix + "```python\n" + instance['prefix_template'] + "\n" + instance['content'] + "\n```"
    # full_prompt += instruction_prefix + instance['content'] + "Function named " + instance['prefix_template']
    full_prompt += instruction_prefix + instance['content'] + "\n" + instance['raw_data']['test_list'][0] + "\n\"\"\"\n"

    # full_prompt += f"You are an expert Python programmer, and here is your task:\n{instance['content']}\nYour code should pass these tests:\n\n{test_list}\n You should submit your final solution in the following format: ```python\n\n```"
    # \n\n```python\n# YOUR CODE HERE\n```\n\nYou DON'T NEED TO write an example of how to use this function. The code should not contain comments. The code should not contain test cases.
    # full_prompt += f"""```python\n{instance['content']}\n{instance['prefix_template']}\n```\n\nPlease think step by step, then complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown syntax. The code should not contain `Main` function. You DON'T NEED TO write an example of how to use this function."""
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    # full_prompt += response_prefix
    if think:
        full_prompt += '<think>\n'
    return full_prompt

def get_humaneval_prompt(
    instance, prompt_type, think
) -> str:
    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    full_prompt += "Complete the following python code:\n"
    # full_prompt += f"```python\n{instance['prompt']}\n```\n"
    full_prompt += instance['prompt'] + "You should submit your final solution in the following format: ```python\n\n```"
    # full_prompt += f"You are an intelligent programming assistant to produce Python algorithmic solutions.\nCan you complete the following Python function?\n```python\n{instance['prompt']}\n```"
    # \nYou should submit your final solution in the following format: ```python\n\n```
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if think:
        full_prompt += '<think>\n'
    return full_prompt

def get_multiple_prompt(
    instance, prompt_type, think
) -> str:
    temp_obj = TEMPLATES[prompt_type]
    language = instance['language']
    if language in language_mappings:
        language = language_mappings[language]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    full_prompt += f"""```{language}\n{instance['prompt']}\n```\n\nPlease think step by step, then complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown syntax. The code should not contain `Main` function. You DON'T NEED TO write an example of how to use this function."""
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if think:
        full_prompt += '<think>\n'
    return full_prompt

# 模板好像有问题，模型总是输出python代码而不是cpp代码
def get_aethercode_prompt(
    instance, prompt_type, think
) -> str:
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the cpp program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    if prompt_type != "chatml_qwen3":
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\nQuestion: "
    else:
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests.\n\nQuestion: "
    full_prompt += instance['prompt'] + '\n\n'
    full_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
    full_prompt += f"```cpp\n#include # YOUR CODE HERE\n```\n\n"
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if think:
        full_prompt += '<think>\n'

    return full_prompt

def get_template_data(dataset, dataset_type, prompt_type, reasoning_model):
    # 获取数据
    if dataset_type == "LiveCodeBenchDataset":
        data = load_lcb_dataset(dataset)
    elif dataset_type == "LiveCodeBenchDataset-cpp":
        data = load_lcb_cpp_dataset(dataset)
    elif dataset_type == "MultiPLEDataset":
        data = load_multiple_dataset(dataset)
    elif dataset_type == "MBPPDataset":
        data = load_mbpp_dataset(dataset)
    elif dataset_type == "HumanEvalDataset":
        data = load_humaneval_dataset(dataset)
    elif dataset_type == "AetherCodeDataset":
        data = load_aethercode_dataset(dataset)
    
    # 给数据套模板
    prompts = []
    for instance in data:
        if dataset_type == "LiveCodeBenchDataset":
            prompt = get_lcb_prompt(instance["raw_data"], prompt_type, reasoning_model)
        elif dataset_type == "LiveCodeBenchDataset-cpp":
            prompt = get_lcb_cpp_prompt(instance["raw_data"], prompt_type, reasoning_model)
        elif dataset_type == 'MBPPDataset':
            prompt = get_mbpp_prompt(instance, prompt_type, reasoning_model)
        elif dataset_type == 'HumanEvalDataset':
            prompt = get_humaneval_prompt(instance, prompt_type, reasoning_model)
        elif dataset_type == 'MultiPLEDataset':
            prompt = get_multiple_prompt(instance, prompt_type, reasoning_model)
        elif dataset_type == "AetherCodeDataset":
            prompt = get_aethercode_prompt(instance, prompt_type, reasoning_model)
        prompts.append(prompt)

    return prompts, data
