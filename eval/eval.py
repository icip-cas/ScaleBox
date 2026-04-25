import json
import os
import signal
import sys
from datetime import datetime

import requests

from utils.set_env import set_hf_cache
from utils.cli import parse_args
from utils.config import load_config, merge_config_into_args
from utils.validate import validate_mode_args, validate_thinking, validate_resume_sample_args
from utils.load_cases import load_cases
from utils.evaluate import build_sandbox_config, evaluate_cases
from utils.resume import sample_cases_with_resume
from utils.template import TEMPLATES


SAMPLES_FILENAME = "samples.jsonl"
RESULTS_FILENAME = "results.jsonl"
ACCURACY_FILENAME = "accuracy.json"


def print_prompt_preview(cases, args):
    if not cases:
        return

    preview_case = cases[0]
    print("=" * 60)
    print("Prompt preview")
    print(f"id: {preview_case['id']}")
    print(f"prompt_type: {args.prompt_type}")
    print(f"thinking: {args.thinking}")
    print("model_prompt:")
    print(preview_case["model_prompt"])
    print("=" * 60)


def check_sandbox_endpoint(url):
    """中文：向 sandbox 接口发送一个最小测试请求，检查服务是否可用。
    English: Send a minimal test request to the sandbox endpoint and check whether it is available.
    """
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
        print(json.dumps(result, indent=2, ensure_ascii=False))
        assert result.get("accepted") is True, result
    except Exception as e:
        print(f"[sandbox check] {type(e).__name__}: {e} | url={url}")
        raise


def create_timestamped_output_dir(output_dir):
    """中文：在用户指定目录下创建时间戳子目录，并将本次结果写入其中。
    English: Create a timestamped subdirectory under the user output directory for this run.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(output_dir, timestamp)
    suffix = 1
    while os.path.exists(run_output_dir):
        run_output_dir = os.path.join(output_dir, f"{timestamp}_{suffix}")
        suffix += 1
    os.makedirs(run_output_dir)
    print(f"[Output] Results will be saved to: {run_output_dir}")
    return run_output_dir


def start_vllm_server(args):
    """中文：按当前配置启动 vLLM Server。
    English: Start vLLM server instances from the runtime config.
    """
    if not args.use_server:
        return None, []

    from utils.vllm_server import VLLMServer

    print("=" * 60)
    print("Starting vLLM Server mode")
    print("=" * 60)

    if not args.model_path:
        raise ValueError("`--model_path` is required when using `--use_server`.")
    model_path = args.model_path
    vllm_server = VLLMServer(
        model_path=model_path,
        num_gpus_total=args.num_gpus_total,
        num_gpus_per_model=args.num_gpus_per_model,
        base_port=args.vllm_server_base_port,
        host=args.vllm_server_host,
        max_model_len=args.max_model_len,
        dtype=args.vllm_server_dtype,
        trust_remote_code=True,
        served_model_name=args.model_name,
        use_npu=args.npu,
        mem_fraction=args.mem_fraction,
        wait_timeout=args.vllm_server_wait_timeout,
    )

    try:
        vllm_server_endpoints = vllm_server.start_servers(wait_ready=True)
        print(f"vLLM Server started successfully with {len(vllm_server_endpoints)} instance(s)")
        for index, endpoint in enumerate(vllm_server_endpoints):
            print(f"  Instance {index}: {endpoint}")
        print("=" * 60)
        return vllm_server, vllm_server_endpoints
    except Exception:
        cleanup_vllm_server(vllm_server)
        raise


def cleanup_vllm_server(vllm_server):
    if vllm_server is None:
        return None

    print("=" * 60)
    print("Shutting down vLLM Server...")
    vllm_server.stop_servers()
    print("vLLM Server stopped")
    print("=" * 60)
    return None


def register_signal_handlers(manager_holder):
    """中文：注册中断/终止信号处理器。
    用于在程序被 Ctrl+C 或系统终止时，先关闭已启动的本地 vLLM 服务，再退出。

    English: Register interrupt/termination signal handlers.
    This ensures the started local vLLM server is stopped before the program exits
    when interrupted by Ctrl+C or terminated by the system.
    """

    def signal_handler(signum, frame):
        print("\n" + "=" * 60, flush=True)
        print("Termination signal received, cleaning up resources...", flush=True)
        manager_holder["manager"] = cleanup_vllm_server(manager_holder["manager"])
        print("=" * 60, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def load_runner(args, vllm_server_endpoints):
    if args.use_server:
        from utils.multi_api_runner import MultiAPIRunner

        return MultiAPIRunner(
            args=args,
            model=args.model_name,
            api_endpoints=vllm_server_endpoints,
            )
    if args.use_ray:
        from utils.vllm_ray import VLLMRay

        return VLLMRay(args, args.model_path)
    raise ValueError("Exactly one of --use_server or --use_ray must be set.")


def _normalize_case_stop_tokens(case):
    stop_tokens = case.get("test", {}).get("stop_tokens")
    if not stop_tokens:
        return None
    if isinstance(stop_tokens, str):
        stop_tokens = stop_tokens.split(",")
    return [str(token) for token in stop_tokens if str(token).strip()]


def sample_cases(cases, args, vllm_server_endpoints, save_callback=None):
    prompts = [case["model_prompt"] for case in cases]
    stop_tokens_by_prompt = [_normalize_case_stop_tokens(case) for case in cases]
    runner = load_runner(args, vllm_server_endpoints)
    raw_results = runner.run_batch(
        prompts,
        stop_tokens_by_prompt=stop_tokens_by_prompt,
        save_callback=save_callback,
    )

    if len(raw_results) != len(cases):
        raise RuntimeError(
            f"Runner returned {len(raw_results)} results, but {len(cases)} cases were provided."
        )

    expanded_cases = []
    for case, samples in zip(cases, raw_results):
        if not isinstance(samples, list):
            raise RuntimeError(
                f"Runner result for case id={case['id']} must be a list, got {type(samples).__name__}."
            )
        for sample in samples:
            expanded_case = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": sample if isinstance(sample, str) else str(sample),
                "scalebox": None,
                "test": case["test"],
            }
            if "language" in case:
                expanded_case["language"] = case["language"]
            if "checker" in case:
                expanded_case["checker"] = case["checker"]
            expanded_cases.append(expanded_case)
    return expanded_cases


def write_sample_results(output_path, cases):
    with open(output_path, "w", encoding="utf-8") as file:
        for case in cases:
            output_case = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": case["response"],
            }
            file.write(json.dumps(output_case, ensure_ascii=False) + "\n")


def write_results(output_path, cases):
    with open(output_path, "w", encoding="utf-8") as file:
        for case in cases:
            output_case = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": case["response"],
                "scalebox": case["scalebox"],
            }
            file.write(json.dumps(output_case, ensure_ascii=False) + "\n")


def write_accuracy(output_path, cases):
    accuracy = 0.0
    if cases:
        accuracy = sum(float(case.get("scalebox", 0.0)) for case in cases) / len(cases)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"accuracy": round(accuracy, 4)}, file, ensure_ascii=False, indent=2)


def main():
    """中文：程序主入口，负责单 benchmark 的采样与评测流程。
    English: Main entry point for single-benchmark sampling and evaluation.
    """
    set_hf_cache()
    args = parse_args()
    config = load_config(args.config)
    args = merge_config_into_args(args, config)
    validate_mode_args(args)
    validate_thinking(args.prompt_type, args.thinking)
    validate_resume_sample_args(args)
    if not args.stop_token:
        args.stop_token = TEMPLATES[args.prompt_type].stop_str
    args.output_dir = create_timestamped_output_dir(args.output_dir)

    manager_holder = {"manager": None}
    register_signal_handlers(manager_holder)

    if not args.sample_only:
        check_sandbox_endpoint(args.endpoint)

    try:
        if args.use_server and not args.eval_only:
            manager_holder["manager"], vllm_server_endpoints = start_vllm_server(args)
        else:
            vllm_server_endpoints = []

        cases = load_cases(args, config)
        print_prompt_preview(cases, args)

        if args.eval_only:
            run_cases = cases
            sandbox_config = build_sandbox_config(config)
            run_cases = evaluate_cases(run_cases, config["benchmark"], sandbox_config, args)

            results_path = os.path.join(args.output_dir, RESULTS_FILENAME)
            write_results(results_path, run_cases)
            print(f"Results saved to: {results_path}")

            accuracy_path = os.path.join(args.output_dir, ACCURACY_FILENAME)
            write_accuracy(accuracy_path, run_cases)
            print(f"Accuracy saved to: {accuracy_path}")
        elif args.sample_only:
            samples_path = os.path.join(args.output_dir, SAMPLES_FILENAME)
            if args.resume_sample:
                sampled_cases = sample_cases_with_resume(
                    cases,
                    args,
                    vllm_server_endpoints,
                    args.resume_sample_path,
                    samples_path,
                    sample_cases,
                    write_sample_results,
                )
            else:
                sampled_cases = sample_cases(cases, args, vllm_server_endpoints)
                write_sample_results(samples_path, sampled_cases)
            print(f"Sample results saved to: {samples_path}")
        else:
            samples_path = os.path.join(args.output_dir, SAMPLES_FILENAME)
            if args.resume_sample:
                sampled_cases = sample_cases_with_resume(
                    cases,
                    args,
                    vllm_server_endpoints,
                    args.resume_sample_path,
                    samples_path,
                    sample_cases,
                    write_sample_results,
                )
            else:
                sampled_cases = sample_cases(cases, args, vllm_server_endpoints)
                write_sample_results(samples_path, sampled_cases)
            print(f"Sample results saved to: {samples_path}")

            sandbox_config = build_sandbox_config(config)
            run_cases = evaluate_cases(sampled_cases, config["benchmark"], sandbox_config, args)

            results_path = os.path.join(args.output_dir, RESULTS_FILENAME)
            write_results(results_path, run_cases)
            print(f"Results saved to: {results_path}")

            accuracy_path = os.path.join(args.output_dir, ACCURACY_FILENAME)
            write_accuracy(accuracy_path, run_cases)
            print(f"Accuracy saved to: {accuracy_path}")
    finally:
        manager_holder["manager"] = cleanup_vllm_server(manager_holder["manager"])


if __name__ == "__main__":
    main()
