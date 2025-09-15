from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from datasets import load_dataset
from utils.livecodebench.generation import load_code_generation_dataset

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
    "ts": "typescript",
    "sh": "bash",
    "js": "nodejs",
    "pl": "perl",
    "rkt": "racket",
    "rs": "rust",
    "rb": "ruby",
    "jl": "julia"
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

def get_mbpp_prompt(
    instance, prompt_type, think
) -> str:
    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    full_prompt += f"You are an expert Python programmer, and here is your task: {instance['content']} Your code should pass these tests:\n\n{instance['test_list'][0]}\n{instance['test_list'][1]}"
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
    full_prompt += instance['prompt']
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
    # 获取数据
    if dataset_type == "LiveCodeBenchDataset":
        data = load_lcb_dataset(dataset)
    elif dataset_type == "MultiPLEDataset":
        data = load_dataset(
            "json",
            data_files=f"data/MultiPL-E/{dataset['huggingFace']['subset']}.jsonl"
        )["train"]
    elif dataset_type == "MBPPDataset":
        data = load_dataset(
            "json",
            data_files=f"data/FusedMBPP/test_mbpp.jsonl"
        )["train"]
    elif dataset_type == "HumanEvalDataset":
        data = load_dataset(
            "json",
            data_files=f"data/openai_humaneval/humaneval.jsonl"
        )["train"]
    
    # 给数据套模板
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
