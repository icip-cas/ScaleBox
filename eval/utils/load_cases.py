import json

from utils.template import get_template_data
from utils.validate import validate_benchmark_data_path, validate_eval_path


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



def load_benchmark_cases(benchmark, benchmark_data_path, config, prompt_type, thinking):
    dataset_config = build_dataset_config(benchmark, benchmark_data_path, config)
    prompts, data = get_template_data(dataset_config, benchmark, prompt_type, thinking)

    cases = []
    for model_prompt, data_item in zip(prompts, data):
        case = {
            "id": data_item["id"],
            "prompt": data_item["prompt"],
            "model_prompt": model_prompt,
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
            if not isinstance(case["response"], str):
                raise ValueError(f"`response` must be a string in eval_only input at line {line_number}")
            cases.append({
                "id": case["id"],
                "prompt": case["prompt"],
                "response": case["response"],
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
    validate_benchmark_data_path(args.benchmark_data_path, benchmark)
    if args.eval_only:
        validate_eval_path(args.eval_path)
        eval_cases = load_eval_cases(args.eval_path)
        benchmark_cases = load_benchmark_cases(
            benchmark,
            args.benchmark_data_path,
            config,
            args.prompt_type,
            args.thinking,
        )
        return merge_eval_with_benchmark_cases(benchmark_cases, eval_cases)
    return load_benchmark_cases(
        benchmark,
        args.benchmark_data_path,
        config,
        args.prompt_type,
        args.thinking,
    )
