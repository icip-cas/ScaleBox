import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError:
    pl = None

from datasets import load_dataset
from tqdm import tqdm
from utils.livecodebench.generation import load_local_code_generation_dataset
from utils.template import (
    get_aethercode_prompt,
    get_humaneval_prompt,
    get_lcb_prompt,
    get_mbpp_prompt,
    get_multipl_e_prompt,
    language_mappings,
)
from utils.validate import validate_benchmark_data_path, validate_eval_path
from utils.logger import get_logger

logger = get_logger(__name__)

def require_sample_id(sample):
    # 中文：读取样本 id，仅兼容 id 和 question_id。
    # English: Read the sample id and only support id and question_id.
    for key in ["id", "question_id"]:
        try:
            return sample[key]
        except (TypeError, KeyError):
            continue

    for attr in ["id", "question_id"]:
        if hasattr(sample, attr):
            return getattr(sample, attr)

    raise KeyError(f"Sample is missing required field `id`: {sample}")

def build_dataset_config(benchmark, benchmark_data_path, config):
    if benchmark == "livecodebench":
        return {
            "id": "livecodebench",
            "dataset": "livecodebench",
            "benchmark_data_path": benchmark_data_path,
            "version": config["version"],
            "begin_date": config.get("begin_date"),
            "end_date": config.get("end_date"),
        }
    if benchmark == "aethercode":
        return {
            "id": "aethercode_cpp",
            "dataset": "aethercode_cpp",
            "benchmark_data_path": benchmark_data_path,
            "version": config["version"],
            "special_judge_file": config["special_judge_file"],
        }
    return {
        "id": benchmark,
        "dataset": benchmark,
        "benchmark_data_path": benchmark_data_path,
    }

def get_data_dir() -> Path:
    return Path(__file__).resolve().parent

def get_default_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT") or "https://huggingface.co"

def get_auto_download_target(benchmark, config):
    data_dir = get_data_dir()
    if benchmark == "livecodebench":
        return {
            "script_path": data_dir / "download_livecodebench.py",
            "benchmark_data_path": data_dir / "livecodebench" / "code_generation_lite",
        }
    if benchmark == "mbpp":
        return {
            "script_path": data_dir / "download_mbpp.py",
            "benchmark_data_path": data_dir / "mbpp.jsonl",
        }
    if benchmark == "mbppplus":
        return {
            "script_path": data_dir / "download_mbppplus.py",
            "benchmark_data_path": data_dir / "mbppplus.jsonl",
        }
    if benchmark == "humaneval":
        return {
            "script_path": data_dir / "download_humaneval.py",
            "benchmark_data_path": data_dir / "openai_humaneval" / "humaneval.jsonl",
        }
    if benchmark == "humanevalplus":
        return {
            "script_path": data_dir / "download_humanevalplus.py",
            "benchmark_data_path": data_dir / "humanevalplus.jsonl",
        }
    if benchmark == "aethercode":
        return {
            "script_path": data_dir / "download_aethercode.py",
            "benchmark_data_path": data_dir / "aethercode",
        }
    return None

def run_download_script(script_path: Path) -> None:
    command = [sys.executable, str(script_path), "--hf-endpoint", get_default_hf_endpoint()]
    logger.info(f"Auto-downloading benchmark data via: {' '.join(command)}")
    subprocess.run(command, cwd=str(get_data_dir()), check=True)

def resolve_benchmark_data_path(args, config):
    benchmark = config["benchmark"]

    if args.benchmark_data_path:
        return args.benchmark_data_path

    if benchmark == "multipl_e":
        raise ValueError(
            "`benchmark.data_path` is required for `multipl_e`; automatic download is not supported."
        )

    target = get_auto_download_target(benchmark, config)
    if target is None:
        raise ValueError(f"Unsupported benchmark for auto download: {benchmark}")

    benchmark_data_path = Path(target["benchmark_data_path"])
    if not benchmark_data_path.exists():
        run_download_script(Path(target["script_path"]))

    args.benchmark_data_path = str(benchmark_data_path)
    logger.info(f"Using benchmark data path: {args.benchmark_data_path}")
    return args.benchmark_data_path

def _build_lcb_dataset_from_path(dataset_path, release_version="release_v1", start_date=None, end_date=None):
    return load_local_code_generation_dataset(
        dataset_path,
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
    )

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
        logger.warning(f"Failed to convert test format: {e}")
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
    benchmark_data_path = dataset.get("benchmark_data_path")
    if not benchmark_data_path:
        raise ValueError("`aethercode` requires `benchmark_data_path`.")
    if pl is None:
        raise RuntimeError("polars is required to read the downloaded parquet files.")
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
    return data

def build_model_prompt(instance, benchmark, prompt_type, thinking):
    if benchmark == "livecodebench":
        return get_lcb_prompt(instance["raw_data"], prompt_type, thinking)
    if benchmark in {"mbpp", "mbppplus"}:
        return get_mbpp_prompt(instance, prompt_type, thinking)
    if benchmark in {"humaneval", "humanevalplus"}:
        return get_humaneval_prompt(instance, prompt_type, thinking)
    if benchmark == "multipl_e":
        return get_multipl_e_prompt(instance, prompt_type, thinking)
    if benchmark == "aethercode":
        return get_aethercode_prompt(instance, prompt_type, thinking)
    raise ValueError(f"Unsupported benchmark: {benchmark}")

def load_benchmark_cases(benchmark, benchmark_data_path, config, prompt_type, thinking):
    dataset_config = build_dataset_config(benchmark, benchmark_data_path, config)
    if benchmark == "livecodebench":
        data = load_lcb_dataset(dataset_config)
    elif benchmark == "multipl_e":
        data = load_multipl_e_dataset(dataset_config)
    elif benchmark in {"mbpp", "mbppplus"}:
        data = load_mbpp_dataset(dataset_config)
    elif benchmark in {"humaneval", "humanevalplus"}:
        data = load_humaneval_dataset(dataset_config)
    elif benchmark == "aethercode":
        data = load_aethercode_dataset(dataset_config)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    cases = []
    for data_item in data:
        case = {
            "id": data_item["id"],
            "prompt": data_item["prompt"],
            "model_prompt": build_model_prompt(data_item, benchmark, prompt_type, thinking),
            "test": data_item["test"],
        }
        if "language" in data_item:
            case["language"] = data_item["language"]
        if "checker" in data_item:
            case["checker"] = data_item["checker"]
        cases.append(case)

    return cases

def load_eval_cases(eval_path):
    cases = []
    with open(eval_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            case = json.loads(line)
            for field in ["id", "prompt", "response"]:
                if field not in case:
                    raise ValueError(f"Missing `{field}` in eval_only input at line {line_number}")

            # Only accept new format: response must be a list of strings
            response = case["response"]
            if not isinstance(response, list):
                raise ValueError(
                    f"`response` must be a list of strings in eval_only input at line {line_number}"
                )
            if not all(isinstance(r, str) for r in response):
                raise ValueError(
                    f"`response` list must contain only strings in eval_only input at line {line_number}"
                )

            # Expand list of responses into multiple cases
            for resp in response:
                cases.append({
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "response": resp,
                })
    return cases

def merge_eval_with_benchmark_cases(benchmark_cases, eval_cases):
    benchmark_case_map = {case["id"]: case for case in benchmark_cases}
    merged_cases = []
    for eval_case in eval_cases:
        if eval_case["id"] not in benchmark_case_map:
            raise ValueError(f"Cannot find benchmark case for eval_only id={eval_case['id']}")
        benchmark_case = benchmark_case_map[eval_case["id"]]
        merged_case = {
            "id": eval_case["id"],
            "prompt": eval_case["prompt"],
            "response": eval_case["response"],
            "model_prompt": benchmark_case["model_prompt"],
            "test": benchmark_case["test"],
        }
        if "language" in benchmark_case:
            merged_case["language"] = benchmark_case["language"]
        if "checker" in benchmark_case:
            merged_case["checker"] = benchmark_case["checker"]
        merged_cases.append(merged_case)
    return merged_cases

def load_cases(args, config):
    benchmark = config["benchmark"]
    benchmark_data_path = resolve_benchmark_data_path(args, config)
    validate_benchmark_data_path(benchmark_data_path, benchmark)
    if args.eval_only:
        validate_eval_path(args.eval_path)
        eval_cases = load_eval_cases(args.eval_path)
        benchmark_cases = load_benchmark_cases(
            benchmark,
            benchmark_data_path,
            config,
            args.prompt_type,
            args.thinking,
        )
        return merge_eval_with_benchmark_cases(benchmark_cases, eval_cases)
    return load_benchmark_cases(
        benchmark,
        benchmark_data_path,
        config,
        args.prompt_type,
        args.thinking,
    )
