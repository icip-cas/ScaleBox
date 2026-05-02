import json
import os
from .logger import get_logger

logger = get_logger(__name__)

def validate_unique_case_ids(cases):
    seen_ids = set()
    duplicate_ids = []
    for case in cases:
        case_id = case["id"]
        if case_id in seen_ids and case_id not in duplicate_ids:
            duplicate_ids.append(case_id)
        seen_ids.add(case_id)

    if duplicate_ids:
        duplicate_ids_str = ", ".join(str(case_id) for case_id in duplicate_ids[:10])
        raise ValueError(f"Duplicate case ids are not supported: {duplicate_ids_str}")

def load_sample_results(sample_path):
    sampled_by_id = {}
    if not os.path.exists(sample_path):
        return sampled_by_id

    with open(sample_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample_case = json.loads(line)
            except json.JSONDecodeError as error:
                has_remaining_non_empty = any(remaining_line.strip() for remaining_line in file)
                if not has_remaining_non_empty:
                    logger.warning(
                        f"Ignoring incomplete trailing line in {sample_path} "
                        f"at line {line_number}: {error}"
                    )
                    break
                raise ValueError(
                    f"Invalid JSON in sample resume file at line {line_number}: {error}"
                ) from error
            if "id" not in sample_case:
                raise ValueError(f"Missing `id` in sample resume file at line {line_number}")
            if "response" not in sample_case:
                raise ValueError(f"Missing `response` in sample resume file at line {line_number}")

            response = sample_case["response"]
            if not isinstance(response, list):
                raise ValueError(
                    f"`response` must be a list of strings in sample resume file at line {line_number}"
                )
            if not all(isinstance(r, str) for r in response):
                raise ValueError(
                    f"`response` list must contain only strings in sample resume file at line {line_number}"
                )
            sampled_by_id.setdefault(sample_case["id"], []).extend(response)

    return sampled_by_id

def append_sample_results(sample_path, case_id, prompt, samples):
    with open(sample_path, "a", encoding="utf-8") as file:
        output_case = {
            "id": case_id,
            "prompt": prompt,
            "response": [sample if isinstance(sample, str) else str(sample) for sample in samples],
        }
        file.write(json.dumps(output_case, ensure_ascii=False) + "\n")
        file.flush()

def build_sampled_cases_from_sample_results(cases, sampled_by_id, n_sample, skip_validation=False):
    if not skip_validation:
        validate_unique_case_ids(cases)
    sampled_cases = []
    for case in cases:
        responses = sampled_by_id.get(case["id"], [])[:n_sample]
        for response in responses:
            sampled_case = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": response,
                "scalebox": None,
                "test": case["test"],
            }
            if "language" in case:
                sampled_case["language"] = case["language"]
            if "checker" in case:
                sampled_case["checker"] = case["checker"]
            sampled_cases.append(sampled_case)
    return sampled_cases

def sample_cases_with_resume(
    cases,
    args,
    vllm_server_endpoints,
    resume_sample_path,
    samples_path,
    sample_cases_fn,
    write_sample_results_fn,
):
    # Validate once at the beginning
    validate_unique_case_ids(cases)
    sampled_by_id = load_sample_results(resume_sample_path)

    # Initialize samples_path with existing completed prompts (those that don't need more samples)
    completed_cases = []
    for case in cases:
        existing_samples = sampled_by_id.get(case["id"], [])
        if len(existing_samples) >= args.n_sample:
            # This prompt already has enough samples, write it immediately
            for sample in existing_samples[:args.n_sample]:
                completed_cases.append({
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "response": sample,
                })
    write_sample_results_fn(samples_path, completed_cases)

    pending_groups = {}
    for case in cases:
        existing_count = min(len(sampled_by_id.get(case["id"], [])), args.n_sample)
        remaining_count = max(0, args.n_sample - existing_count)
        if remaining_count > 0:
            pending_groups.setdefault(remaining_count, []).append(case)

    original_n_sample = args.n_sample
    try:
        for remaining_count in sorted(pending_groups):
            grouped_cases = pending_groups[remaining_count]

            def save_callback(index, new_samples, group_cases=grouped_cases, existing_by_id=sampled_by_id):
                group_case = group_cases[index]
                case_id = group_case["id"]
                # Merge old samples with new samples
                old_samples = existing_by_id.get(case_id, [])
                all_samples = old_samples + new_samples
                # Write merged samples as a single line
                append_sample_results(samples_path, case_id, group_case["prompt"], all_samples)
                # Update in-memory state to avoid reloading from disk
                sampled_by_id[case_id] = all_samples

            args.n_sample = remaining_count
            sample_cases_fn(grouped_cases, args, vllm_server_endpoints, save_callback=save_callback)
    finally:
        args.n_sample = original_n_sample

    # Build final results from in-memory state, skip validation since we already validated
    sampled_cases = build_sampled_cases_from_sample_results(cases, sampled_by_id, original_n_sample, skip_validation=True)
    return sampled_cases
