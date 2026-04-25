# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3  # Number of retries for API calls
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    memory_limit_mb=1024,
    timeout=30,
    **kwargs,
):
    """
    Computes the code score by executing it against test cases in a remote sandbox.

    Args:
        data_source (any): This parameter is not currently used.
        solution_str (str): The string containing the code solution to be evaluated.
            It may include a language identifier in a markdown code block (e.g., ```python).
        ground_truth (str or dict): A JSON string or a dictionary containing the test cases.
            It should have 'input' and 'output' keys.
        extra_info (any): This parameter is not currently used.
        sandbox_fusion_url (str, optional): The URL of the sandbox service.
            Example: "https://<your service endpoint>/common_batch_evaluate".
        memory_limit_mb (int, optional): The memory limit in megabytes for code execution.
        timeout (int, optional): The timeout in seconds for both compilation and execution
            for each test case. Defaults to 30.
        **kwargs: Additional keyword arguments that are not currently used.

    Returns:
        tuple[float, list[dict]]: A tuple containing:
            - score (float): A score from 0.0 to 1.0, representing the fraction of
              test cases that passed.
            - metadata_list (list[dict]): A list containing a dictionary with detailed
              metadata about the execution, including API responses, status, errors,
              and individual test case results.
    """
    # 1. Extract code and language from solution_str
    # Remove <think>.*</think> tags if they exist
    solution = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    language_str = re.search(r"```(\w+)", solution_str)
    if language_str:
        language = language_str.group(1).strip()
    else:
        # Default to Python if no language is specified
        language = "python"

    try:
        # 2. Parse test cases
        test_cases = ground_truth
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if not test_cases or "input" not in test_cases or "output" not in test_cases:
            logger.error("Invalid test_cases structure.")
            logger.error(f"{str(test_cases)[:100]} ...")
            return 0.0, [
                {"error": "Invalid test_cases structure (missing inputs/outputs)"}
            ]

        # 3. Call sandbox API
        api_response, error_msg = call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url,
            code=solution,
            in_outs=test_cases,
            compile_timeout=timeout,
            run_timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language=language,
        )

        # 4. Process API response
        metadata = {
            "input": str(test_cases),
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "stdout": None,
            "stderr": None,
            "exit_code": None,
            "duration": None,
            "compile_duration": None,
            "compile_stderr": None,
            "api_status": None,
            "compile_status": None,
            "run_status": None,
            "score": 0.0,
        }

        if error_msg:
            metadata["status"] = "api_error"
            logger.error(f"Sandbox Error Report: API error occurred: {error_msg}")
            generation_to_log = (
                solution[:200] + "..." if len(solution) > 200 else solution
            )
            logger.error(f"Sandbox Error Report: Generation: {generation_to_log}")
        elif api_response:
            logger.debug(f"Sandbox Debug Report: API Response: {api_response}")
            metadata["api_response"] = api_response
            metadata["api_status"] = api_response.get("status")
            compile_result = api_response.get("compile_result")
            run_result = api_response.get("run_result")

            if compile_result:
                metadata["compile_status"] = compile_result.get("status")
                metadata["compile_duration"] = compile_result.get("execution_time")
                metadata["compile_stderr"] = compile_result.get("stderr")

            if run_result:
                metadata["run_status"] = run_result.get("status")
                metadata["stdout"] = run_result.get("stdout")
                metadata["stderr"] = run_result.get("stderr")
                metadata["exit_code"] = run_result.get("return_code")
                metadata["duration"] = run_result.get("execution_time")

            if api_response.get("accepted", None) is True:
                metadata["status"] = "success"
                metadata["score"] = 1.0
            else:
                metadata["status"] = "wrong_answer"
                cases = api_response.get("tests", [])
                total_cases = len(cases)
                passed_cases = sum(
                    1 for test in cases if test and test.get("passed", False)
                )
                if total_cases > 0:
                    metadata["score"] = passed_cases / total_cases

        score = metadata.get("score", 0.0)
        final_metadata = [metadata]

        logger.info(f"Sandbox Info Report: Results: {score}")

    except Exception as e:
        score = 0.0
        final_metadata = [{"error": f"Unhandled exception: {e}"}]

    return float(score), (
        final_metadata if isinstance(final_metadata, list) else [final_metadata]
    )


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    in_outs: any,
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str,
    run_all_cases: bool = True,
    total_timeout: int = 30,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    """
    Calls the remote sandbox API to execute code and retries on specific HTTP errors.

    Args:
        sandbox_fusion_url (str): The URL of the sandbox fusion API endpoint.
        code (str): The source code to be executed.
        in_outs (any): The test cases to be used for evaluation. This is typically
            a dictionary or a JSON string representing the test data.
        compile_timeout (int): The maximum time in seconds allowed for code compilation.
        run_timeout (int): The maximum time in seconds allowed for code execution.
        memory_limit_mb (int): The memory limit in megabytes for code execution.
        language (str, optional): The programming language of the code. Defaults to "python".
            This is used for validation against a list of supported languages.
        run_all_cases (bool, optional): Whether to run all test cases or stop on the first failure.
        total_timeout (int, optional): The total timeout in seconds for the entire API call.

    Returns:
        tuple[Optional[dict[str, Any]], Optional[str]]: A tuple containing:
            - response_json (dict or None): The JSON response from the API if the call
              is successful, otherwise None.
            - error_message (str or None): A string containing an error message if the
              call fails after all retries, otherwise None.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    # For special judge programs
    if "special_judge_program" in in_outs:
        special_judge_program = in_outs["special_judge_program"]
        special_judge_language = in_outs.get("special_judge_language", "python")
    else:
        special_judge_program = None
        special_judge_language = None

    payload = json.dumps(
        {
            "completion": code,
            "config": {
                "language": language,
                "compile_timeout": compile_timeout,
                "run_timeout": run_timeout,
                'memory_limit_mb': memory_limit_mb,
                "provided_data": {
                    "test_cases": in_outs
                },
                "extra": {
                    "run_all_cases": run_all_cases, 
                    "total_timeout": total_timeout,
                    'special_judge_program': special_judge_program,
                    'special_judge_language': special_judge_language,
                },
            },
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: "
                f"Calling sandbox API at {sandbox_fusion_url}"
            )  # <-- Use internal log_prefix
            requests._client_max_size = (
                100 * 1024 * 1024
            )  # Set max request size to 100MB
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for retryable status codes (e.g., 500, 504)
            if response.status_code in [500, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Received status code {response.status_code} "
                    f"on attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = (
                f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            )
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = (
                f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            )
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(
        f"{log_prefix}Sandbox API call failed. Last error: {last_error}"
    )  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, (
        last_error.replace(log_prefix, "API Call Failed: ")
        if last_error
        else "API Call Failed after retries"
    )
