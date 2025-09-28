from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from datasets import load_dataset
from utils.livecodebench.generation import load_code_generation_dataset
import re
import polars as pl

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
    "go_test.go": "go"
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
            Role.SYSTEM: "",
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

def load_mbpp_dataset(dataset):
    raw_data = load_dataset(
        "json",
        data_files=f"data/FusedMBPP/mbppplus.jsonl"
    )["train"]
    data = []
    for id, sample in enumerate(raw_data):
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
        else:   
            entry_point = re.search(r'assert\s+([A-Za-z_]\w*)\s*\(', sample['test_list'][0]).group(1)
        test = "def check(" + entry_point + "):\n    "
        # test += "\n    ".join(sample['labels']['challenge_test_list'])
        test += "\n    ".join(sample['test_list'])
        lines = sample['code'].split('\n')
        for i, line in enumerate(lines):
            if entry_point in line:
                def_line_index = i
                break
        result_lines = lines[def_line_index:def_line_index + 1]
        prefix_template = '\n'.join(result_lines)
        if "test" in sample:
            data.append({
                "id": id+1,
                "raw_data": sample,
                "content": sample['prompt'],
                "test": {"type": "assert", "test": sample['test'] + test, "entry_point": entry_point},
                "prefix_template": prefix_template,
            })
        else:    
            data.append({
                "id": id+1,
                "raw_data": sample,
                "content": sample['prompt'],
                "test": {"type": "assert", "test": test, "entry_point": entry_point},
                "prefix_template": prefix_template,
            })
    return data

def load_humaneval_dataset(dataset):
    raw_data = load_dataset(
        "json",
        data_files=f"data/openai_humaneval/humanevalplus.jsonl"
    )["train"]
    data = []
    for id, sample in enumerate(raw_data):
        data.append({
            "id": id+1,
            "raw_data": sample,
            "prompt": sample['prompt'],
            "test": {"type": "assert", "test": sample['test'], "entry_point": sample['entry_point']},
        })
    return data

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

def get_mbpp_prompt(
    instance, prompt_type, think
) -> str:
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\"\"\"\n"
    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    test_list = "\n".join(instance['raw_data']['test_list'])
    full_prompt += instruction_prefix + instance['content'] + "\n" + instance['raw_data']['test_list'][0] + "\n\"\"\"\n"
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
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
    full_prompt += instance['prompt'] + "You should submit your final solution in the following format: ```python\n\n```"
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

def get_template_data(dataset, dataset_type, prompt_type, reasoning_model):
    if dataset_type == "LiveCodeBenchDataset":
        data = load_lcb_dataset(dataset)
    elif dataset_type == "MultiPLEDataset":
        data = load_multiple_dataset(dataset)
    elif dataset_type == "MBPPDataset":
        data = load_mbpp_dataset(dataset)
    elif dataset_type == "HumanEvalDataset":
        data = load_humaneval_dataset(dataset)
    
    prompts = []
    for instance in data:
        if dataset_type == "LiveCodeBenchDataset":
            prompt = get_lcb_prompt(instance["raw_data"], prompt_type, reasoning_model)
        elif dataset_type == 'MBPPDataset':
            prompt = get_mbpp_prompt(instance, prompt_type, reasoning_model)
        elif dataset_type == 'HumanEvalDataset':
            prompt = get_humaneval_prompt(instance, prompt_type, reasoning_model)
        elif dataset_type == 'MultiPLEDataset':
            prompt = get_multiple_prompt(instance, prompt_type, reasoning_model)
        prompts.append(prompt)

    return prompts, data
