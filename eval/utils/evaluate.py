import asyncio
import copy
import re

import aiohttp
from tqdm import tqdm


MAX_CONCURRENCY = 32
DEFAULT_TOTAL_TIMEOUT = 300


def summarize_sandbox_result(result, run_all_cases):
    if run_all_cases:
        tests = result.get("tests") if isinstance(result, dict) else None
        if isinstance(tests, list) and tests:
            passed_count = sum(1 for test in tests if isinstance(test, dict) and test.get("passed") is True)
            return passed_count / len(tests)
    return 1.0 if isinstance(result, dict) and result.get("accepted") else 0.0


def build_sandbox_config(config):
    sandbox_config = {
        "run_timeout": config.get("run_timeout", 10),
        "compile_timeout": config.get("compile_timeout", 10),
        "extra": copy.deepcopy(config.get("extra", {})),
    }
    sandbox_config["extra"].setdefault("total_timeout", config.get("total_timeout", DEFAULT_TOTAL_TIMEOUT))
    sandbox_config["extra"].setdefault("run_all_cases", config.get("run_all_cases", False))
    if "language" in config:
        sandbox_config["language"] = config["language"]
    return sandbox_config



def extract_completion(response_text, language, thinking):
    completion = re.split(r"</think>\s*", response_text)[-1] if thinking else response_text

    output_lines = completion.split("\n")
    fence_lines = [index for index, line in enumerate(output_lines) if "```" in line]

    if len(fence_lines) < 2:
        if "```" not in completion and "def" in completion:
            fence_language = language or "python"
            return f"```{fence_language}\n{completion}\n```\n"
        return None

    completion = "\n".join(output_lines[fence_lines[-2]: fence_lines[-1] + 1])

    index = len(fence_lines) - 1
    while index >= 1:
        start = fence_lines[index - 1]
        end = fence_lines[index]
        candidate = "\n".join(output_lines[start: end + 1])
        if "def" in candidate:
            completion = candidate
            break
        index -= 2

    return completion



def get_case_language(benchmark, case):
    if benchmark == "aethercode":
        return "cpp"
    if benchmark == "multipl_e":
        return case.get("language")
    return "python"


async def get_sandbox_result(benchmark, case, completion, config, url, session):
    """中文：组装 sandbox 请求并返回执行结果。
    English: Build a sandbox request and return the execution result.
    """
    config_copy = copy.deepcopy(config)
    config_copy["language"] = get_case_language(benchmark, case)

    if benchmark == "aethercode":
        extra = config_copy.setdefault("extra", {})
        extra["special_judge_program"] = case["checker"]
        extra["special_judge_language"] = "cpp"
        extra["force_special_judge"] = True

    payload = {
        "completion": completion,
        "config": {**config_copy, "provided_data": {"test_cases": case["test"]}},
    }

    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def _evaluate_case(case, benchmark, sandbox_config, args, session):
    language = get_case_language(benchmark, case)
    completion = extract_completion(case["response"], language, args.thinking)
    if completion is None:
        return {"score": 0.0, "raw_result": None}
    try:
        result = await get_sandbox_result(benchmark, case, completion, sandbox_config, args.endpoint, session)
    except Exception:
        return {"score": 0.0, "raw_result": None}
    return {
        "score": summarize_sandbox_result(result, sandbox_config["extra"].get("run_all_cases", False)),
        "raw_result": result,
    }


async def evaluate_cases_async(cases, benchmark, sandbox_config, args):
    timeout = aiohttp.ClientTimeout(total=4000)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    pbar_lock = asyncio.Lock()
    results = [None] * len(cases)
    pbar = tqdm(total=len(cases), desc="Evaluating", ncols=120) if cases else None

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for index, case in enumerate(cases):
                async def _task(idx=index, current_case=case):
                    async with sem:
                        results[idx] = await _evaluate_case(current_case, benchmark, sandbox_config, args, session)
                    if pbar is not None:
                        async with pbar_lock:
                            pbar.update(1)
                tasks.append(_task())
            await asyncio.gather(*tasks)
    finally:
        if pbar is not None:
            pbar.close()
    return results



def evaluate_cases(cases, benchmark, sandbox_config, args):
    sandbox_results = asyncio.run(evaluate_cases_async(cases, benchmark, sandbox_config, args))
    for case, sandbox_result in zip(cases, sandbox_results):
        case["scalebox_score"] = sandbox_result["score"]
        case["scalebox_raw"] = sandbox_result["raw_result"]
        case["scalebox"] = sandbox_result["raw_result"] if args.save_full_scalebox_result else sandbox_result["score"]
    return cases
