import json
import os


def load_sample_results(sample_path):
    sampled_by_id = {}
    if not os.path.exists(sample_path):
        return sampled_by_id

    with open(sample_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            sample_case = json.loads(line)
            if "id" not in sample_case:
                raise ValueError(f"Missing `id` in sample resume file at line {line_number}")
            if "response" not in sample_case:
                raise ValueError(f"Missing `response` in sample resume file at line {line_number}")
            if not isinstance(sample_case["response"], str):
                raise ValueError(f"`response` must be a string in sample resume file at line {line_number}")
            sampled_by_id.setdefault(sample_case["id"], []).append(sample_case["response"])

    return sampled_by_id



def append_sample_results(sample_path, case_id, prompt, samples):
    with open(sample_path, "a", encoding="utf-8") as file:
        for sample in samples:
            output_case = {
                "id": case_id,
                "prompt": prompt,
                "response": sample if isinstance(sample, str) else str(sample),
            }
            file.write(json.dumps(output_case, ensure_ascii=False) + "\n")



def build_sampled_cases_from_sample_results(cases, sampled_by_id, n_sample):
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
    sampled_by_id = load_sample_results(resume_sample_path)
    existing_sampled_cases = build_sampled_cases_from_sample_results(cases, sampled_by_id, args.n_sample)
    write_sample_results_fn(samples_path, existing_sampled_cases)

    pending_groups = {}
    for case in cases:
        existing_count = len(sampled_by_id.get(case["id"], []))
        remaining_count = max(0, args.n_sample - existing_count)
        if remaining_count > 0:
            pending_groups.setdefault(remaining_count, []).append(case)

    original_n_sample = args.n_sample
    try:
        for remaining_count in sorted(pending_groups):
            grouped_cases = pending_groups[remaining_count]

            def save_callback(index, samples, group_cases=grouped_cases):
                group_case = group_cases[index]
                append_sample_results(samples_path, group_case["id"], group_case["prompt"], samples)

            args.n_sample = remaining_count
            sample_cases_fn(grouped_cases, args, vllm_server_endpoints, save_callback=save_callback)
    finally:
        args.n_sample = original_n_sample

    sampled_by_id = load_sample_results(samples_path)
    sampled_cases = build_sampled_cases_from_sample_results(cases, sampled_by_id, original_n_sample)
    write_sample_results_fn(samples_path, sampled_cases)
    return sampled_cases
