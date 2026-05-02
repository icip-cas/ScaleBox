import os
import json
import requests
from .logger import get_logger

logger = get_logger(__name__)

MODE_FIELDS = ("use_server", "use_ray")

def validate_mode_args(args):
    # Validate sampling mode arguments and required path arguments.
    if args.sample_only and args.eval_only:
        raise ValueError("`--sample_only` and `--eval_only` cannot be used together.")

    if args.eval_only:
        if not args.eval_path:
            raise ValueError("`--eval_path` is required when using `--eval_only`.")
        return

    if args.eval_path:
        raise ValueError("`--eval_path` can only be used with `--eval_only`.")

    enabled_modes = [field for field in MODE_FIELDS if getattr(args, field, False)]
    if len(enabled_modes) != 1:
        raise ValueError("Exactly one of --use_server or --use_ray must be set.")

    if args.use_server and not args.model_path:
        raise ValueError("`--model_path` is required when using `--use_server`.")
    if args.use_ray and not args.model_path:
        raise ValueError("`--model_path` is required when using `--use_ray`.")

def validate_benchmark_data_path(benchmark_data_path, benchmark):
    if not os.path.exists(benchmark_data_path):
        raise FileNotFoundError(f"Benchmark data path does not exist: {benchmark_data_path}")
    if benchmark in {"livecodebench", "aethercode"}:
        if not os.path.isdir(benchmark_data_path):
            raise ValueError(f"Benchmark `{benchmark}` expects `--benchmark_data_path` to be a directory")
    else:
        if not os.path.isfile(benchmark_data_path):
            raise ValueError(f"Benchmark `{benchmark}` expects `--benchmark_data_path` to be a JSONL file")

def validate_eval_path(eval_path):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Eval path does not exist: {eval_path}")
    if not os.path.isfile(eval_path):
        raise ValueError("`--eval_path` must point to a JSONL file with id/prompt/response")

def find_latest_samples_file(output_dir):
    # Find the most recent samples.jsonl file in the output directory.
    # Args:
    # output_dir: Base output directory containing timestamped subdirectories
    # Returns:
    # Path to the most recent samples.jsonl file, or None if not found
    if not os.path.exists(output_dir):
        return None

    if not os.path.isdir(output_dir):
        return None

    # Find all timestamped subdirectories
    subdirs = []
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path):
            samples_path = os.path.join(entry_path, "samples.jsonl")
            if os.path.isfile(samples_path):
                subdirs.append((entry, entry_path, samples_path))

    if not subdirs:
        return None

    # Sort by directory name (which is timestamp-based) in descending order
    subdirs.sort(key=lambda x: x[0], reverse=True)

    # Return the most recent samples.jsonl
    return subdirs[0][2]

def validate_resume_sample_args(args):
    if args.resume_sample and args.eval_only:
        raise ValueError("`--resume_sample` cannot be used with `--eval_only`.")

    if args.resume_sample_path and not args.resume_sample:
        raise ValueError("`--resume_sample_path` can only be used with `--resume_sample`.")

    if args.resume_sample:
        # If resume_sample_path is not provided, try to find the latest samples.jsonl
        if not args.resume_sample_path:
            latest_samples = find_latest_samples_file(args.output_dir)
            if latest_samples:
                args.resume_sample_path = latest_samples
                logger.info(f"Auto-detected resume sample path: {args.resume_sample_path}")
            else:
                raise ValueError(
                    f"`--resume_sample_path` is required when using `--resume_sample`, "
                    f"or there must be existing samples.jsonl files in `--output_dir` ({args.output_dir}). "
                    f"No samples.jsonl files found in the output directory."
                )

        # Validate the resume_sample_path
        if not os.path.exists(args.resume_sample_path):
            raise FileNotFoundError(f"Resume sample path does not exist: {args.resume_sample_path}")
        if not os.path.isfile(args.resume_sample_path):
            raise ValueError("`--resume_sample_path` must point to a JSONL file.")

def validate_thinking(prompt_type, thinking):
    # Validate that thinking is only used with qwen3.
    if thinking and prompt_type != "qwen3":
        raise ValueError("`--thinking` is only supported when `--prompt_type=qwen3`.")

def print_prompt_preview(cases, args):
    if not cases:
        return

    preview_case = cases[0]
    logger.info("Prompt preview")
    logger.info(f"id: {preview_case['id']}")
    logger.info(f"prompt_type: {args.prompt_type}")
    logger.info(f"thinking: {args.thinking}")
    logger.info("model_prompt:")
    logger.info(preview_case["model_prompt"])

def check_sandbox_endpoint(url):
    # Send a minimal test request to the sandbox endpoint and check whether it is available.
    from .evaluate import is_local_endpoint, is_run_code_endpoint, get_endpoint_path

    if is_local_endpoint(url):
        logger.info("[sandbox check] Using local evaluator mode; skipped remote endpoint health check.")
        return

    endpoint_path = get_endpoint_path(url)
    is_run_code = is_run_code_endpoint(url)

    if is_run_code:
        payload = {
            "language": "python",
            "code": "print(1 + 1)",
            "run_timeout": 10,
            "compile_timeout": 10,
        }
    else:
        payload = {
            "completion": "```python\nimport re\ndef text_match_three(text):\n        patterns = 'ab{3}?'\n        return re.search(patterns,  text)\n```",
            "config": {
                "language": "python",
                "provided_data": {
                    "test_cases": {
                        "type": "assert",
                        "test": "def check(text_match_three):\n    assert not text_match_three(\"abc\")",
                        "entry_point": "text_match_three",
                    },
                },
            },
        }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))

        if is_run_code:
            run_result = result.get("run_result") or {}
            if result.get("status") != "Success" or str(run_result.get("stdout", "")).strip() != "2":
                raise RuntimeError(f"Unexpected /run_code check result: {result}")
        else:
            if result.get("accepted") is not True:
                raise RuntimeError(f"Unexpected /common_evaluate_batch check result: {result}")
    except Exception as e:
        logger.error(f"[sandbox check] {type(e).__name__}: {e} | url={url}")
        raise
