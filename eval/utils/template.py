from dataclasses import dataclass, field
import json
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from datasets import load_dataset
from tqdm import tqdm
from utils.livecodebench.generation import load_local_code_generation_dataset
from datetime import datetime
import glob
import os
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
    os.path.dirname(os.path.dirname(__file__)),
    "LiveCodeBench",
    "lcb_runner",
    "prompts",
    "few_shot_examples",
    "generation",
)


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

def require_sample_id(sample):
    """中文：读取样本 id，仅兼容 id 和 question_id。
    English: Read the sample id and only support id and question_id.
    """
    for key in ["id", "question_id"]:
        try:
            return sample[key]
        except (TypeError, KeyError):
            continue

    for attr in ["id", "question_id"]:
        if hasattr(sample, attr):
            return getattr(sample, attr)

    raise KeyError(f"Sample is missing required field `id`: {sample}")


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


def _build_lcb_dataset_from_path(dataset_path, release_version="release_v1", start_date=None, end_date=None):
    return load_local_code_generation_dataset(
        dataset_path,
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
    )


# load lcb dataset
def load_lcb_dataset(dataset):
    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError("`livecodebench` requires `benchmark_data_path`.")
    version = dataset.get("version")
    if not version:
        raise ValueError("`livecodebench` requires `version`.")
    release_version = f"release_{version}"
    raw_data = _build_lcb_dataset_from_path(
        benchmark_data_path,
        release_version=release_version,
        start_date=dataset.get("begin_date"),
        end_date=dataset.get("end_date"),
    )
    data = []
    for sample in tqdm(raw_data, desc="Building benchmark cases", ncols=120):
        data.append({
            "id": require_sample_id(sample),
            "raw_data": sample,
            "prompt": sample.question_content,
            "test": sample.get_evaluation_sample(),
        })
    return data


def load_multipl_e_dataset(dataset):
    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError("`multipl_e` requires `benchmark_data_path`.")
    raw_data = load_dataset("json", data_files=benchmark_data_path)["train"]
    data = []
    for sample in raw_data:
        language = sample["language"]
        if language in language_mappings:
            language = language_mappings[language]
        data.append({
            "id": require_sample_id(sample),
            "raw_data": sample,
            "prompt": sample["prompt"],
            "language": language,
            "test": {"type": "assert", "tests": sample["tests"], "stop_tokens": sample["stop_tokens"]},
        })
    return data


def convert_test_format(test_code: str, entry_point: str, use_set: bool = False) -> str:
    """
    Convert raw test code format to the target format.

    Raw format:
        inputs = [[(arg1), (arg2)], ...]
        results = [(result1), (result2), ...]
        for i, (inp, exp) in enumerate(zip(inputs, results)):
            assertion(func_name(*inp), exp, 0)

    Target format:
        def check(func_name):
            assert set(func_name(arg1, arg2)) == set(result1)  # use_set=True
            assert func_name(arg1, arg2) == result1            # use_set=False
            ...
        check(func_name)

    Args:
        test_code: Raw test code
        entry_point: Function name
        use_set: Whether to wrap with set(); determined by whether sample['test_list'][0] contains 'set'
    """
    try:
        inputs_match = re.search(r'inputs\s*=\s*(\[.*?\])\s*\nresults', test_code, re.DOTALL)
        if not inputs_match:
            return test_code
        inputs_str = inputs_match.group(1)

        results_match = re.search(r'results\s*=\s*(\[.*?\])\s*\n(?:for|$)', test_code, re.DOTALL)
        if not results_match:
            return test_code
        results_str = results_match.group(1)

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

        lines = [f"def check({entry_point}):"]
        for inp, exp in zip(inputs, results):
            if not isinstance(inp, (list, tuple)):
                inp = [inp]
            args_str = ", ".join(repr(x) for x in inp)
            if use_set:
                lines.append(f"    assert set({entry_point}({args_str})) == set({repr(exp)})")
            else:
                lines.append(f"    assert {entry_point}({args_str}) == {repr(exp)}")

        lines.append(f"check({entry_point})")
        return "\n".join(lines)
    except Exception as e:
        print(f"Warning: Failed to convert test format: {e}")
        return test_code


def load_mbpp_dataset(dataset):
    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError(f"`{dataset['id']}` requires `benchmark_data_path`.")
    raw_data = load_dataset("json", data_files=benchmark_data_path)["train"]

    data = []
    for sample in raw_data:
        if "math.isclose" in sample["test_list"][0]:
            entry_point = re.search(r"math\.isclose\((\w+)\(", sample["test_list"][0]).group(1)
        elif "text_match_three" in sample["test_list"][0]:
            entry_point = "text_match_three"
        elif "is_perfect_square" in sample["test_list"][0]:
            entry_point = "is_perfect_square"
        elif "similar_elements" in sample["test_list"][0]:
            entry_point = "similar_elements"
        elif "find_char_long" in sample["test_list"][0]:
            entry_point = "find_char_long"
        elif "common_in_nested_lists" in sample["test_list"][0]:
            entry_point = "common_in_nested_lists"
        elif "extract_singly" in sample["test_list"][0]:
            entry_point = "extract_singly"
        elif "larg_nnum" in sample["test_list"][0]:
            entry_point = "larg_nnum"
        elif "Diff" in sample["test_list"][0]:
            entry_point = "Diff"
        elif "max_height" in sample["test_list"][0]:
            entry_point = "max_height"
        else:
            entry_point = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", sample["test_list"][0]).group(1)

        test = "def check(" + entry_point + "):\n    "
        test += "\n    ".join(sample["test_list"])

        if dataset["id"] == "mbppplus":
            use_set = "set(" in sample["test_list"][0]
            test = convert_test_format(sample["test"], entry_point, use_set)

        lines = sample["code"].split("\n")
        def_line_index = 0
        for index, line in enumerate(lines):
            if entry_point in line:
                def_line_index = index
                break
        result_lines = lines[def_line_index:def_line_index + 1]
        prefix_template = "\n".join(result_lines)

        data.append({
            "id": require_sample_id(sample),
            "raw_data": sample,
            "prompt": sample["prompt"],
            "test": {"type": "assert", "test": test, "entry_point": entry_point},
            "prefix_template": prefix_template,
        })
    return data


def load_humaneval_dataset(dataset):
    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError(f"`{dataset['id']}` requires `benchmark_data_path`.")
    raw_data = load_dataset("json", data_files=benchmark_data_path)["train"]
    data = []
    for sample in raw_data:
        data.append({
            "id": require_sample_id(sample),
            "raw_data": sample,
            "prompt": sample["prompt"],
            "test": {"type": "assert", "test": sample["test"], "entry_point": sample["entry_point"]},
        })
    return data


def load_aethercode_dataset(dataset):
    import polars as pl

    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError("`aethercode` requires `benchmark_data_path`.")
    input_root = benchmark_data_path
    version = dataset.get("version")
    if not version:
        raise ValueError("`aethercode` requires `version`.")
    raw_data = []
    versions = [v.strip() for v in str(version).strip('"').split(",") if v.strip()]
    version_dirs = [os.path.join(input_root, v) for v in versions]

    for version_dir in version_dirs:
        parquet_pattern = os.path.join(version_dir, "test-*.parquet")
        parquet_files = sorted(glob.glob(parquet_pattern))
        if not parquet_files:
            continue
        df = pl.read_parquet(parquet_pattern)
        for row in df.iter_rows(named=True):
            raw_data.append({"id": row["id"], "prompt": row["description"], "test_cases": row["test_cases"]})

    import json

    special_judge_file = dataset.get("special_judge_file")
    if not special_judge_file:
        raise ValueError("`aethercode` requires `special_judge_file`.")
    checker_data = []
    with open(special_judge_file, "r", encoding="utf-8") as file:
        for line in file:
            checker_data.append(json.loads(line))

    checker_dict = {item["id"]: item["checker"] for item in checker_data}

    data = []
    for sample in raw_data:
        sample_id = require_sample_id(sample)
        checker = checker_dict.get(sample_id)
        if checker is None:
            continue

        input_values = []
        output_values = []
        for test_case in sample["test_cases"]:
            input_values.append(test_case["input"])
            output_values.append(test_case["output"])
        data.append({
            "id": sample_id,
            "raw_data": sample,
            "prompt": sample["prompt"],
            "test": {"type": "stdin_stdout", "input": input_values, "output": output_values, "fn_name": None},
            "checker": checker,
        })
    print("###len(data)###", len(data))
    return data


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


def get_template_data(dataset, benchmark, prompt_type, thinking):
    if benchmark == "livecodebench":
        data = load_lcb_dataset(dataset)
    elif benchmark == "multipl_e":
        data = load_multipl_e_dataset(dataset)
    elif benchmark in {"mbpp", "mbppplus"}:
        data = load_mbpp_dataset(dataset)
    elif benchmark in {"humaneval", "humanevalplus"}:
        data = load_humaneval_dataset(dataset)
    elif benchmark == "aethercode":
        data = load_aethercode_dataset(dataset)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    prompts = []
    for instance in data:
        if benchmark == "livecodebench":
            prompt = get_lcb_prompt(instance["raw_data"], prompt_type, thinking)
        elif benchmark in {"mbpp", "mbppplus"}:
            prompt = get_mbpp_prompt(instance, prompt_type, thinking)
        elif benchmark in {"humaneval", "humanevalplus"}:
            prompt = get_humaneval_prompt(instance, prompt_type, thinking)
        elif benchmark == "multipl_e":
            prompt = get_multipl_e_prompt(instance, prompt_type, thinking)
        elif benchmark == "aethercode":
            prompt = get_aethercode_prompt(instance, prompt_type, thinking)
        prompts.append(prompt)

    return prompts, data
