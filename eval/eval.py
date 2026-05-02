import json
import os
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import requests

from utils.set_env import set_hf_cache
from utils.cli import parse_args
from utils.config import load_config, merge_config_into_args
from utils.validate import (
    validate_mode_args,
    validate_thinking,
    validate_resume_sample_args,
    print_prompt_preview,
    check_sandbox_endpoint,
)
from data.load_data import load_cases
from utils.evaluate import build_sandbox_config, evaluate_cases
from utils.multi_api_runner import MultiAPIRunner
from utils.resume import append_sample_results, sample_cases_with_resume, validate_unique_case_ids
from utils.template import TEMPLATES
from utils.vllm_ray import VLLMRay
from utils.vllm_server import VLLMServer
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

SAMPLES_FILENAME = "samples.jsonl"
RESULTS_FILENAME = "results.jsonl"
ACCURACY_FILENAME = "accuracy.json"

def create_timestamped_output_dir(output_dir):
    # Create a timestamped subdirectory under the user output directory for this run.
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_output_dir)
    logger.info(f"Results will be saved to: {run_output_dir}")
    return run_output_dir

def start_vllm_server(args):
    # Start vLLM server instances from the runtime config.
    if not args.use_server:
        return None, []

    logger.info("Starting vLLM Server mode")

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
        logger.info(f"vLLM Server started successfully with {len(vllm_server_endpoints)} instance(s)")
        for index, endpoint in enumerate(vllm_server_endpoints):
            logger.info(f"  Instance {index}: {endpoint}")
        return vllm_server, vllm_server_endpoints
    except Exception:
        cleanup_vllm_server(vllm_server)
        raise

def cleanup_vllm_server(vllm_server):
    if vllm_server is None:
        return None

    logger.info("Shutting down vLLM Server...")
    vllm_server.stop_servers()
    logger.info("vLLM Server stopped")
    return None

def register_signal_handlers(manager_holder):
    # Register interrupt/termination signal handlers.
    # This ensures the started local vLLM server is stopped before the program exits
    # when interrupted by Ctrl+C or terminated by the system.

    def signal_handler(signum, frame):
        logger.info("Termination signal received, cleaning up resources...")
        manager_holder["manager"] = cleanup_vllm_server(manager_holder["manager"])
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def load_runner(args, vllm_server_endpoints):
    if args.use_server:
        return MultiAPIRunner(
            args=args,
            model=args.model_name,
            api_endpoints=vllm_server_endpoints,
            )
    if args.use_ray:
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
    # Aggregate cases by id to group multiple samples together
    cases_by_id = {}
    for case in cases:
        case_id = case["id"]
        if case_id not in cases_by_id:
            cases_by_id[case_id] = {
                "id": case_id,
                "prompt": case["prompt"],
                "responses": []
            }
        cases_by_id[case_id]["responses"].append(case["response"])

    # Write aggregated results
    with open(output_path, "w", encoding="utf-8") as file:
        for case_id in cases_by_id:
            output_case = {
                "id": cases_by_id[case_id]["id"],
                "prompt": cases_by_id[case_id]["prompt"],
                "response": cases_by_id[case_id]["responses"],
            }
            file.write(json.dumps(output_case, ensure_ascii=False) + "\n")

def write_results(output_path, cases):
    # Aggregate cases by id to group multiple samples together
    cases_by_id = {}
    case_order = []  # Track the order of first appearance

    for case in cases:
        case_id = case["id"]
        if case_id not in cases_by_id:
            cases_by_id[case_id] = {
                "id": case_id,
                "prompt": case["prompt"],
                "responses": [],
                "scalebox": []
            }
            case_order.append(case_id)

        # Append response and scalebox in order
        cases_by_id[case_id]["responses"].append(case["response"])
        cases_by_id[case_id]["scalebox"].append(case["scalebox"])

    # Write aggregated results in the order they first appeared
    with open(output_path, "w", encoding="utf-8") as file:
        for case_id in case_order:
            output_case = {
                "id": cases_by_id[case_id]["id"],
                "prompt": cases_by_id[case_id]["prompt"],
                "responses": cases_by_id[case_id]["responses"],
                "scalebox": cases_by_id[case_id]["scalebox"],
            }
            file.write(json.dumps(output_case, ensure_ascii=False) + "\n")

def write_accuracy(output_path, cases):
    # Aggregate cases by id to collect scalebox scores
    cases_by_id = {}

    for case in cases:
        case_id = case["id"]
        if case_id not in cases_by_id:
            cases_by_id[case_id] = []

        # Collect scalebox scores for this prompt
        scalebox_score = float(case.get("scalebox_score", case.get("scalebox", 0.0)))
        cases_by_id[case_id].append(scalebox_score)

    # Calculate accuracy for each sample position (column-wise average)
    # accuracy[i] = average of all prompts' i-th sample
    if not cases_by_id:
        accuracy_list = []
        mean_accuracy = 0.0
    else:
        # Get the number of samples (assume all prompts have the same number of samples)
        n_samples = len(next(iter(cases_by_id.values())))

        accuracy_list = []
        for sample_idx in range(n_samples):
            # Calculate average accuracy at this sample position across all prompts
            scores_at_position = [scores[sample_idx] for scores in cases_by_id.values() if sample_idx < len(scores)]
            avg_accuracy = sum(scores_at_position) / len(scores_at_position) if scores_at_position else 0.0
            accuracy_list.append(round(avg_accuracy, 4))

        # Calculate mean of all accuracies
        mean_accuracy = round(sum(accuracy_list) / len(accuracy_list), 4) if accuracy_list else 0.0

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"accuracy": accuracy_list, "mean_accuracy": mean_accuracy}, file, ensure_ascii=False, indent=2)

def convert_jsonl_to_json(jsonl_path):
    """Convert JSONL file to formatted JSON and delete the JSONL file.

    Uses atomic write (temp file + os.replace) for safety.
    Only logs warning if JSONL deletion fails (keeps the JSON result).
    """
    jsonl_path = Path(jsonl_path)
    json_path = jsonl_path.with_suffix('.json')

    try:
        # Read all lines from JSONL
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # Write to temporary file first (atomic write)
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            dir=json_path.parent,
            text=True
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic replace
            os.replace(temp_path, json_path)
        except:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # Try to delete JSONL (only warning if fails)
        try:
            os.remove(jsonl_path)
        except Exception as e:
            logger.warning(
                f"Successfully created {json_path}, but failed to delete {jsonl_path}: {e}"
            )

        return str(json_path)
    except Exception as e:
        logger.error(f"Failed to convert {jsonl_path} to JSON: {e}")
        raise

def build_sample_save_callback(cases, samples_path):
    def save_callback(index, samples, run_cases=cases):
        if not samples:
            return
        case = run_cases[index]
        append_sample_results(samples_path, case["id"], case["prompt"], samples)

    return save_callback

def sample_cases_with_incremental_save(cases, args, vllm_server_endpoints, samples_path):
    with open(samples_path, "w", encoding="utf-8"):
        pass

    save_callback = build_sample_save_callback(cases, samples_path)
    sampled_cases = sample_cases(cases, args, vllm_server_endpoints, save_callback=save_callback)
    write_sample_results(samples_path, sampled_cases)
    return sampled_cases

def main():
    # Main entry point for single-benchmark sampling and evaluation.
    setup_logger()
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
        if not args.eval_only:
            validate_unique_case_ids(cases)
        print_prompt_preview(cases, args)

        if args.eval_only:
            run_cases = cases
            sandbox_config = build_sandbox_config(config)
            run_cases = evaluate_cases(run_cases, config["benchmark"], sandbox_config, args)

            results_path = os.path.join(args.output_dir, RESULTS_FILENAME)
            write_results(results_path, run_cases)

            accuracy_path = os.path.join(args.output_dir, ACCURACY_FILENAME)
            write_accuracy(accuracy_path, run_cases)
            logger.info(f"Accuracy saved to: {accuracy_path}")

            # Convert results.jsonl to results.json and delete jsonl
            results_json_path = convert_jsonl_to_json(results_path)
            logger.info(f"Results saved to: {results_json_path}")
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
                sampled_cases = sample_cases_with_incremental_save(
                    cases,
                    args,
                    vllm_server_endpoints,
                    samples_path,
                )
            logger.info(f"Sample results saved to: {samples_path}")
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
                sampled_cases = sample_cases_with_incremental_save(
                    cases,
                    args,
                    vllm_server_endpoints,
                    samples_path,
                )
            logger.info(f"Sample results saved to: {samples_path}")

            sandbox_config = build_sandbox_config(config)
            run_cases = evaluate_cases(sampled_cases, config["benchmark"], sandbox_config, args)

            results_path = os.path.join(args.output_dir, RESULTS_FILENAME)
            write_results(results_path, run_cases)

            accuracy_path = os.path.join(args.output_dir, ACCURACY_FILENAME)
            write_accuracy(accuracy_path, run_cases)
            logger.info(f"Accuracy saved to: {accuracy_path}")

            # Convert results.jsonl to results.json and delete jsonl
            results_json_path = convert_jsonl_to_json(results_path)
            logger.info(f"Results saved to: {results_json_path}")
    finally:
        manager_holder["manager"] = cleanup_vllm_server(manager_holder["manager"])

if __name__ == "__main__":
    main()
