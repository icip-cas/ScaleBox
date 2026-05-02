from dataclasses import dataclass
import json
from typing import Dict, Optional
from enum import Enum
import os
import re

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
    "d": "D_ut",
}

LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
LCB_SYSTEM_MESSAGE_DEEPSEEK_R1 = (
    "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
)
LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

LCB_LLAMA_PROMPT_DIR = os.path.join(
    os.path.dirname(__file__),
    "livecodebench",
    "few_shot_examples",
    "generation",
)

def _resolve_lcb_llama_prompt_dir():
    required_files = ("func.json", "stdin.json")
    if all(os.path.isfile(os.path.join(LCB_LLAMA_PROMPT_DIR, file_name)) for file_name in required_files):
        return LCB_LLAMA_PROMPT_DIR

    raise FileNotFoundError(
        "Could not find LiveCodeBench few-shot example files in: "
        f"{LCB_LLAMA_PROMPT_DIR}"
    )

LCB_LLAMA_PROMPT_DIR = _resolve_lcb_llama_prompt_dir()

def _load_lcb_llama_examples(file_name):
    with open(os.path.join(LCB_LLAMA_PROMPT_DIR, file_name), "r", encoding="utf-8") as file:
        return json.load(file)

LCB_LLAMA_FUNC_EXAMPLES = _load_lcb_llama_examples("func.json")
LCB_LLAMA_STDIN_EXAMPLES = _load_lcb_llama_examples("stdin.json")

def _get_lcb_llama_prompt(question):
    starter_code = getattr(question, "starter_code", "")
    examples = LCB_LLAMA_FUNC_EXAMPLES if starter_code else LCB_LLAMA_STDIN_EXAMPLES

    def get_example_prompt(example):
        prompt = "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if starter_code:
            prompt += "### Starter Code\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Answer\n\n"
        prompt += example["answer"]
        if example["answer"]:
            prompt += "\n\n"
        return prompt

    return get_example_prompt(examples[0]) + get_example_prompt(
        {
            "question": question.question_content,
            "sample_code": starter_code,
            "answer": "",
        }
    )

def _get_lcb_deepseek_r1_user_prompt(question):
    starter_code = getattr(question, "starter_code", "")
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if starter_code:
        prompt += f"{LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"{LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "<｜Assistant｜>"
    return prompt

def _get_qwen3_thinking_prefix(think):
    if think:
        return ""
    return "<think>\n\n</think>\n\n"

def _get_lcb_user_prompt(question, prompt_type):
    starter_code = getattr(question, "starter_code", "")

    if prompt_type == "qwen2.5-instruct":
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
        prompt += f"Question: {question.question_content}\n\n"
        if starter_code:
            prompt += f"{LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"{LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        return prompt

    if prompt_type == "qwen3":
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
        prompt += f"Question: {question.question_content}\n\n"
        if starter_code:
            prompt += f"{LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"{LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        return prompt

    prompt = f"### Question:\n{question.question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

TEMPLATES = {
    "qwen2.5-instruct": ConversationTemplate(
        name="qwen2.5-instruct",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="You are a helpful assistant.",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "qwen3": ConversationTemplate(
        name="qwen3",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="You are a helpful assistant.",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "qwen2.5-distill": ConversationTemplate(
        name="qwen2.5-distill",
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
    "llama3": ConversationTemplate(
        name="llama3",
        role_starts={
            Role.SYSTEM: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
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
    "llama3-instruct": ConversationTemplate(
        name="llama3-instruct",
        role_starts={
            Role.SYSTEM: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
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

def get_lcb_prompt(question, prompt_type, think) -> str:
    if prompt_type == "llama3":
        return _get_lcb_llama_prompt(question)

    temp_obj = TEMPLATES[prompt_type]

    if prompt_type == "qwen2.5-distill":
        return LCB_SYSTEM_MESSAGE_DEEPSEEK_R1 + _get_lcb_deepseek_r1_user_prompt(question)

    system_message = temp_obj.default_system_message if prompt_type in {"qwen2.5-instruct", "qwen3"} else LCB_SYSTEM_MESSAGE_GENERIC
    if prompt_type == "llama3-instruct":
        system_message = system_message.strip()
        user_message = _get_lcb_user_prompt(question, prompt_type).strip()
    else:
        user_message = _get_lcb_user_prompt(question, prompt_type)

    full_prompt = ""
    if system_message:
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]

    full_prompt += temp_obj.role_starts[Role.HUMAN]
    full_prompt += user_message
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if prompt_type == "qwen3":
        full_prompt += _get_qwen3_thinking_prefix(think)
    return full_prompt

def get_mbpp_prompt(instance, prompt_type, think) -> str:
    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    full_prompt += f"""```python\n{instance['prefix_template']}\n```\n\nPlease think step by step, then complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown syntax. The code should not contain `Main` function. You DON'T NEED TO write an example of how to use this function."""
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if prompt_type == "qwen3":
        full_prompt += _get_qwen3_thinking_prefix(think)
    return full_prompt

def get_humaneval_prompt(instance, prompt_type, think) -> str:
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
    if prompt_type == "qwen3":
        full_prompt += _get_qwen3_thinking_prefix(think)
    return full_prompt

def get_multipl_e_prompt(instance, prompt_type, think) -> str:
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
    if prompt_type == "qwen3":
        full_prompt += _get_qwen3_thinking_prefix(think)
    return full_prompt

# The template seems problematic: the model often outputs Python instead of C++.
def get_aethercode_prompt(instance, prompt_type, think) -> str:
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the cpp program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    temp_obj = TEMPLATES[prompt_type]
    full_prompt = ""
    if temp_obj.default_system_message != "":
        full_prompt += temp_obj.role_starts[Role.SYSTEM]
        full_prompt += temp_obj.default_system_message
        full_prompt += temp_obj.role_ends[Role.SYSTEM]
    full_prompt += temp_obj.role_starts[Role.HUMAN]
    if prompt_type != "qwen3":
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\nQuestion: "
    else:
        full_prompt += "You will be given a question (problem specification) and will generate a correct cpp program that matches the specification and passes all tests.\n\nQuestion: "
    full_prompt += instance['prompt'] + '\n\n'
    full_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
    full_prompt += "```cpp\n#include # YOUR CODE HERE\n```\n\n"
    full_prompt += temp_obj.role_ends[Role.HUMAN]
    full_prompt += temp_obj.role_starts[Role.ASSISTANT]
    if prompt_type == "qwen3":
        full_prompt += _get_qwen3_thinking_prefix(think)
    return full_prompt
