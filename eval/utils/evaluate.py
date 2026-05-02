import copy
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse

import requests
from tqdm import tqdm

MAX_CONCURRENCY = 32
DEFAULT_TOTAL_TIMEOUT = 300
DEFAULT_REQUEST_TIMEOUT = 4000

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

def get_endpoint_path(url):
    parsed = urlparse(url)
    return parsed.path.rstrip("/")

def is_local_endpoint(url):
    endpoint = (url or "").strip().lower()
    if endpoint in {"local", "local://", "local://run_code"}:
        return True
    return endpoint.startswith("local://")

def is_run_code_endpoint(url):
    return get_endpoint_path(url).endswith("/run_code")

def strip_code_fence(code):
    text = code.strip()
    if not text.startswith("```"):
        return code
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip("\n")

def normalize_stdout(text):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).rstrip("\n")

def run_code_success(response):
    if not isinstance(response, dict):
        return False
    if response.get("status") != "Success":
        return False
    run_result = response.get("run_result") or {}
    return run_result.get("return_code", 0) == 0

def build_requests_session(pool_size):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def post_json_sync(url, payload, timeout, session=None):
    client = session if session is not None else requests
    response = client.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

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

def execute_python_code_locally(code, stdin_data, run_timeout):
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input=stdin_data if stdin_data is not None else "",
            text=True,
            capture_output=True,
            timeout=run_timeout,
            check=False,
        )
        return {
            "status": "Success" if completed.returncode == 0 else "RuntimeError",
            "run_result": {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "return_code": completed.returncode,
            },
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "Timeout",
            "run_result": {
                "stdout": error.stdout or "",
                "stderr": (error.stderr or "") + f"\nTimed out after {run_timeout}s",
                "return_code": -1,
            },
        }
    except Exception as error:
        return {
            "status": "Error",
            "run_result": {
                "stdout": "",
                "stderr": repr(error),
                "return_code": -1,
            },
        }

def build_python_function_call_harness(code, fn_name, case_input, case_output):
    args = case_input if isinstance(case_input, list) else [case_input]
    expected = case_output[0] if isinstance(case_output, list) and len(case_output) == 1 else case_output
    return f"""
_user_code = {repr(code)}
_globals = {{}}
exec(_user_code, _globals, _globals)
_fn_name = {repr(fn_name)}
_args = {repr(args)}
_expected = {repr(expected)}

if "Solution" in _globals and hasattr(_globals["Solution"], _fn_name):
    _callable = getattr(_globals["Solution"](), _fn_name)
else:
    _callable = _globals[_fn_name]

_result = _callable(*_args)
assert _result == _expected, f"Expected {{_expected!r}}, got {{_result!r}}"
"""

def evaluate_via_local_run_code(case, code, config):
    test_cases = case["test"]
    test_type = test_cases.get("type", "stdin_stdout")
    run_all_cases = config.get("extra", {}).get("run_all_cases", False)
    language = config.get("language", "python")
    run_timeout = config.get("run_timeout", 10)

    inputs = test_cases.get("input", [])
    outputs = test_cases.get("output", [])
    total_tests = min(len(inputs), len(outputs))
    length_aligned = len(inputs) == len(outputs)
    tests = []
    accepted = length_aligned

    if language != "python":
        return {"accepted": False, "tests": [{"passed": False, "exec_info": {"status": "UnsupportedLanguage"}}]}

    if total_tests == 0:
        return {"accepted": False, "tests": []}

    if test_type == "stdin_stdout":
        for idx in range(total_tests):
            exec_info = execute_python_code_locally(code, inputs[idx], run_timeout)
            passed = run_code_success(exec_info) and (
                normalize_stdout((exec_info.get("run_result") or {}).get("stdout")) == normalize_stdout(outputs[idx])
            )
            tests.append({"passed": passed, "exec_info": exec_info})
            if not passed:
                accepted = False
                if not run_all_cases:
                    break
    elif test_type == "function_call":
        fn_name = test_cases.get("fn_name")
        if not fn_name:
            return {"accepted": False, "tests": [{"passed": False}]}
        for idx in range(total_tests):
            harness = build_python_function_call_harness(code, fn_name, inputs[idx], outputs[idx])
            exec_info = execute_python_code_locally(harness, "", run_timeout)
            passed = run_code_success(exec_info)
            tests.append({"passed": passed, "exec_info": exec_info})
            if not passed:
                accepted = False
                if not run_all_cases:
                    break
    else:
        return {"accepted": False, "tests": [{"passed": False}]}

    return {"accepted": accepted and all(test.get("passed") for test in tests), "tests": tests}

def get_sandbox_result_sync(benchmark, case, completion, config, url, session=None):
    config_copy = copy.deepcopy(config)
    config_copy["language"] = get_case_language(benchmark, case)
    request_timeout = config_copy.get("extra", {}).get("request_timeout", DEFAULT_REQUEST_TIMEOUT)

    if benchmark == "aethercode":
        extra = config_copy.setdefault("extra", {})
        extra["special_judge_program"] = case["checker"]
        extra["special_judge_language"] = "cpp"
        extra["force_special_judge"] = True

    payload = {
        "completion": completion,
        "config": {**config_copy, "provided_data": {"test_cases": case["test"]}},
    }

    return post_json_sync(url, payload, request_timeout, session=session)

def _evaluate_case_sync(case, benchmark, sandbox_config, args):
    try:
        language = get_case_language(benchmark, case)
        completion = extract_completion(case.get("response", ""), language, args.thinking)
        if completion is None:
            return {"score": 0.0, "raw_result": None}
        result = get_sandbox_result_sync(benchmark, case, completion, sandbox_config, args.endpoint)
    except Exception:
        return {"score": 0.0, "raw_result": None}
    return {
        "score": summarize_sandbox_result(result, sandbox_config["extra"].get("run_all_cases", False)),
        "raw_result": result,
    }

def evaluate_cases_threadpool(cases, benchmark, sandbox_config, args):
    results = [{"score": 0.0, "raw_result": None} for _ in cases]
    pbar = tqdm(total=len(cases), desc="Evaluating", ncols=120) if cases else None
    pbar_lock = Lock()

    def _task(idx, current_case):
        result = _evaluate_case_sync(current_case, benchmark, sandbox_config, args)
        results[idx] = result
        if pbar is not None:
            with pbar_lock:
                pbar.update(1)
        return result

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            futures = {executor.submit(_task, idx, case): idx for idx, case in enumerate(cases)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception:
                    results[idx] = {"score": 0.0, "raw_result": None}
    finally:
        if pbar is not None:
            pbar.close()
    return results

def evaluate_cases(cases, benchmark, sandbox_config, args):
    # Use threadpool for evaluation
    sandbox_results = evaluate_cases_threadpool(cases, benchmark, sandbox_config, args)

    for case, sandbox_result in zip(cases, sandbox_results):
        case["scalebox_score"] = sandbox_result["score"]
        case["scalebox_raw"] = sandbox_result["raw_result"]
        case["scalebox"] = sandbox_result["raw_result"] if args.save_full_scalebox_result else sandbox_result["score"]
    return cases
