# Copyright 2025 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import atexit
import asyncio
import builtins
import json
import multiprocessing
import os
import re
import signal
import sys
import base64
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import HTTPException

from sandbox.configs.run_config import RunConfig
from sandbox.datasets.types import EvalTestCase, GeneralStdioTest, RunStatus, TestConfig
from sandbox.runners.types import CommandRunResult, CommandRunStatus, compile_languages
from sandbox.utils.common import truncate_str
from sandbox.utils.execution import max_concurrency
from sandbox.utils.sandbox_client import RunCodeRequest, run_code_in_sandbox, run_code_in_sandbox_w_retry
from sandbox.server.sandbox_api import RunCodeResponse

sandbox_config = RunConfig.get_instance_sync()
logger = structlog.stdlib.get_logger()
DEFAULT_COMMON_BATCH_TOTAL_TIMEOUT = 1000


async def check_auto_test_case(code: str, config: TestConfig) -> EvalTestCase:
    '''
    auto test: run the code and check if the return value is 0
    '''
    result = await run_code_in_sandbox(RunCodeRequest(code=code, language=config.language))
    return EvalTestCase(passed=result.status == RunStatus.Success, exec_info=result)


async def check_function_call_test_case(code: str, case: GeneralStdioTest, config: TestConfig) -> EvalTestCase:
    '''
    auto test: run the code and check if the return value is 0
    '''
    result = await run_code_in_sandbox(RunCodeRequest(code=code, 
                                                      language=config.language,
                                                      compile_timeout=config.compile_timeout or 10,
                                                      run_timeout=config.run_timeout or 10,
                                                      mem_limit=config.mem_limit,))
    return EvalTestCase(passed=result.status == RunStatus.Success, exec_info=result, test_info=case)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(a, b, rel_tol=1e-5):
    return abs(a - b) / max(abs(b), 1e-10) < rel_tol

async def check_special_judge(stdin: str, stdout: str, answer: str, special_judge_program: str, special_judge_language: str = "python") -> bool:
    stdin_base64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
    stdout_base64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
    answer_base64 = base64.b64encode(answer.encode('utf-8')).decode('utf-8')
    files = {
        "stdin.txt": stdin_base64,
        "stdout.txt": stdout_base64,
        "answer.txt": answer_base64,
    }
    
    result = await run_code_in_sandbox_w_retry(
        RunCodeRequest(code=special_judge_program,
                       language=special_judge_language,
                       files=files,
                       stdin=None,
                       compile_timeout=10,
                       run_timeout=10,))
    logger.debug(f"check_special_judge result: {result}")

    return result.status == RunStatus.Success and result.run_result.return_code == 0


async def check_stdio_test_case(code: str, case: GeneralStdioTest, config: TestConfig, lower_cmp=True) -> EvalTestCase:
    if config.language in compile_languages:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           compile_timeout=config.compile_timeout or 10,
                           run_timeout=config.run_timeout or 10,
                           mem_limit=config.mem_limit,))
    else:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           run_timeout=config.run_timeout or 20,
                           mem_limit=config.mem_limit,))
    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != 'Success':
        return fail_case
    result_lines = result.run_result.stdout.strip().split('\n')
    expected_lines = case.output['stdout'].strip().split('\n')
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]
    # if len(result_lines) != len(expected_lines):
    #     return fail_case
    special_judge_program = config.extra.get('special_judge_program', None)
    special_judge_language = config.extra.get('special_judge_language', 'python')
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl):
                if float_equal(float(rl), float(el)):
                    continue
            if special_judge_program is not None:
                correct = await check_special_judge(
                    stdin=case.input['stdin'],
                    stdout=case.output['stdout'],
                    answer=result.run_result.stdout,
                    special_judge_program=special_judge_program,
                    special_judge_language=special_judge_language
                )
                if correct:
                    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())
                else:
                    return fail_case
            else:
                return fail_case
    if not config.extra.get('return_full_case', False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


async def check_stdio_test_cases(code: str,
                                 cases: List[GeneralStdioTest],
                                 config: TestConfig,
                                 lower_cmp=True) -> List[EvalTestCase]:
    result = []
    for case in cases:
        outcome = await check_stdio_test_case(code, case, config, lower_cmp)
        result.append(outcome)
        if not outcome.passed:
            break
    return result


async def check_stdio_test_cases_parallel(code: str,
                                          cases: List[GeneralStdioTest],
                                          config: TestConfig,
                                          lower_cmp=True) -> List[EvalTestCase]:
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    result = []
    tasks: List[asyncio.Task[EvalTestCase]] = []

    check_stdio_test_case_limited = check_stdio_test_case
    runner_concurrency = sandbox_config.dataset.max_runner_concurrency
    if config.language == 'csharp':
        # dotnet runtime startup is memory-heavy; cap C# stdio fanout to avoid CoreCLR OOM.
        csharp_max_concurrency = int(os.getenv('SANDBOX_CSHARP_CASE_CONCURRENCY', '1'))
        if runner_concurrency > 0:
            runner_concurrency = min(runner_concurrency, csharp_max_concurrency)
        else:
            runner_concurrency = csharp_max_concurrency

    if runner_concurrency > 0:
        check_stdio_test_case_limited = max_concurrency(
            runner_concurrency)(check_stdio_test_case)

    for case in cases:
        task = asyncio.create_task(check_stdio_test_case_limited(code, case, config, lower_cmp))
        tasks.append(task)

    run_all_cases = config.extra.get("run_all_cases", False)
    total_timeout = config.extra.get("total_timeout", 300)
    deadline = asyncio.get_running_loop().time() + total_timeout

    for task_idx, task in enumerate(tasks):
        instance_logger.info(f"check_stdio_test_cases_parallel, total_timeout: {total_timeout}, task_idx: {task_idx}, deadline: {deadline}, current_time: {asyncio.get_running_loop().time()}")
        if asyncio.get_running_loop().time() > deadline:
            instance_logger.error("check_stdio_test_cases_parallel timeout", total_timeout=total_timeout, task_idx=task_idx)
            for remaining_task in tasks:
                if not remaining_task.done():
                    time_out_outcome = EvalTestCase(passed=False, exec_info=RunCodeResponse(status=RunStatus.SandboxError, message= "Total Timeout"), test_info=None)
                    result.append(time_out_outcome)
                    remaining_task.cancel()
            break
        try:
            outcome = await task
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to check stdio call test case: {e}')
        result.append(outcome)

        if not run_all_cases and not outcome.passed:
            for remaining_task in tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            break

    return result

def concat_function_assertion(function, fn_name, input, output, language):
    original_output = output
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if type(output) is str:
        output = f'"{output}"'
    elif type(output) is bool:
        if language in ["cpp", "java", "csharp", "go", "rust", "D_ut", "lua", "julia", "nodejs", "typescript", "php", "ruby", "scala", "swift", "kotlin_script"]:
            output = "true" if output else "false"
        elif language == "racket":
            output = "#t" if output else "#f"
        elif language == "R":
            output = "TRUE" if output else "FALSE"
        elif language in ["bash", "perl", "verilog"]:
            output = "1" if output else "0"
    input_str = ', '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
    # python, racket, D_ut, lua, julia, nodejs, cpp, go, java, typescript, csharp, rust, php, bash, ruby, perl, scala, kotlin_script, R, verilog, swift
    if language == "python":
        if "class Solution" not in function:
            full_code = f'''
{function}
res = {fn_name}(*{input})
assert res == {original_output}[0]
'''
        else:
            full_code = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
            full_code += f'''
{function}
sol = Solution()
res = sol.{fn_name}(*{input})
assert res == {original_output}[0]
'''
    elif language == "racket":
        input_str_racket = ' '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
        full_code = f'''
{function}
(define (main)
    (define res ({fn_name} {input_str_racket}))
    (define expected {output})
    (if (equal? res expected)
        (display "Test passed")
        (error "Test failed")))
(main)
'''
    elif language == "D_ut":
        full_code = f'''
import std.stdio;
{function}
void main() {{
    auto res = {fn_name}({input_str});
    auto expected = {output};
    if (res != expected) {{
        throw new Exception("Test failed");
    }} else {{
        writeln("Test passed");
    }}
}}
'''
    elif language == "lua":
        full_code = f'''
{function}
res = {fn_name}({input_str})
assert(res =={output}, "Test failed: expected 8, got " .. res)
'''
    elif language == "julia":
        full_code = f'''
{function}
res = {fn_name}({input_str})
if res != {output}
    error("Test failed")
end
'''
    elif language == "nodejs":
        full_code = f'''
{function}
const res = {fn_name}({input_str});
const expected = {output};
if (res !== expected) {{
    throw new Error(`Test failed`);
}}
'''
    elif language == "cpp":
        full_code = f'''
#include <iostream>
#include <cassert>
{function}
int main() {{
    auto res = {fn_name}({input_str});
    auto expected = {output};
    assert(res == expected);
    return 0;
}}
'''
    elif language == "go":
        full_code = f'''
package main
import (
    "fmt"
    "testing"
)
{function}
func Test{fn_name}(t *testing.T) {{
    res := {fn_name}({input_str})
    expected := {output}
    if res != expected {{
        t.Errorf("Expected %v, got %v", expected, res)
    }}
}}
func main() {{
    res := {fn_name}({input_str})
    expected := {output}
    if res != expected {{
        panic(fmt.Sprintf("Test failed: Expected %v, got %v", expected, res))
    }} else {{
        fmt.Println("Test passed")
    }}
}}
'''
    elif language == "java":
        full_code = f'''
public class Main {{
    {function}
    public static void main(String[] args) {{
        var res = {fn_name}({input_str});
        var expected = {output};
        if (res != expected) {{
            throw new RuntimeException("Test failed");
        }} else {{
            System.out.println("Test passed");
        }}
    }}
}}
'''
    elif language == "typescript":
        full_code = f'''
{function}
const res = {fn_name}({input_str});
const expected = {output};
if (res !== expected) {{
    throw new Error(`Test failed`);
}}
'''
    elif language == "csharp":
        full_code = f'''
using System;

public class Test{fn_name} {{
    {function}
    public static void Main(string[] args) {{
        var res = {fn_name}({input_str});
        var expected = {output};
        if (res != expected) {{
            throw new Exception("Test failed");
        }} else {{
            Console.WriteLine("Test passed");
        }}
    }}
}}
'''
    elif language == "rust":
        full_code = f'''
{function}
fn main() {{
    let res = {fn_name}({input_str});
    let expected = {output};
    if res != expected {{
        panic!("Test failed");
    }} else {{
        println!("Test passed");
    }}
}}
'''
    elif language == "php":
        full_code = f'''
<?php
{function}
$res = {fn_name}({input_str});
$expected = {output};
if ($res != $expected) {{
    throw new Exception("Test failed");
}}
?>
'''
    elif language == "bash":
        input_str = ' '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
        full_code = f'''
{function}
res=$({fn_name} {input_str})
expected={output}
if [ "$res" != "$expected" ]; then
    echo "Test failed, res: $res, expected: $expected" 1>&2
    exit 1
fi
'''
    elif language == "ruby":
        full_code = f'''
{function}
res = {fn_name}({input_str})
expected = {output}
if res != expected
    raise "Test failed"
end
'''
    elif language == "perl":
        full_code = f'''
{function}
$res = {fn_name}({input_str});
$expected = {output};
if ($res != $expected) {{
    die "Test failed";
}}
'''
    elif language == "scala":
        full_code = f'''
object Test{fn_name} {{
    {function}
    def main(args: Array[String]): Unit = {{
        val res = {fn_name}({input_str})
        val expected = {output}
        if (res != expected) {{
            throw new Exception(s"Test failed: Expected $expected, got $res")
        }} else {{
            println("Test passed")
        }}
    }}
}}
'''
    elif language == "kotlin_script":
        kotlin_input_str = ', '.join([f'"{i}"' if type(i) is str else ("true" if i else "false") if type(i) is bool else str(i) for i in input])
        full_code = f'''
{function}
val res = {fn_name}({kotlin_input_str})
val expected = {output}
if (res != expected) {{
    throw Exception("Test failed")
}}
'''
    elif language == "R":
        r_input_str = ', '.join([f'"{i}"' if type(i) is str else ("TRUE" if i else "FALSE") if type(i) is bool else str(i) for i in input])
        full_code = f'''
{function}
res <- {fn_name}({r_input_str})
expected <- {output}
stopifnot(isTRUE(all.equal(res, expected)))
'''
    elif language == "verilog":
        verilog_input_str = ', '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
        full_code = f'''
module tb;
{function}
    integer res;
    integer expected;
    initial begin
        res = {fn_name}({verilog_input_str});
        expected = {output};
        if (res !== expected) begin
            $display("Test failed");
            $fatal;
        end
        $display("Test passed");
        $finish;
    end
endmodule
'''
    elif language == "swift":
        # For Swift, we need to add parameter labels in function call
        # Convert "a, b" to "a: a, b: b" format
        swift_input_args_str = re.findall(f'func {fn_name}\((.*)\)', function)[0]
        swift_input_args = re.findall(r'(\w+):', swift_input_args_str)
        swift_input_params = []
        for arg, value in zip(swift_input_args, input):
            if isinstance(value, str):
                value = f'"{value}"'
            if len(input) > 1:
                swift_input_params.append(f'{arg}: {value}')
            else:
                swift_input_params.append(f'{value}')
        swift_input_str = ', '.join(swift_input_params)

        
        full_code = f'''
{function}

let res = {fn_name}({swift_input_str})
let expected = {output}
if res != expected {{
    fatalError("Test failed: Expected \\(expected), got \\(res)")
}} else {{
    print("Test passed")
}}
'''
    return full_code

async def check_function_call_test_cases_parallel(code: str,
                                                  cases: List[Dict[str, Any]],
                                                  config: TestConfig) -> List[EvalTestCase]:
    """
    Given the function call input and output, append `assert fn(input) == output` to the code and run it.
    cases = [
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[1, 2, 3, 4, 5, 6, 7, 8, 9]], 'output': [[1, 2, 7, 4, 5, 6, 3, 8, 9]]}, 
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[12, 13, 14]], 'output': [[12, 14, 13]]}, 
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[9, 2, 4, 7, 3]], 'output': [[2, 7, 4, 3, 9]]}
    ]
    """
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    result = []
    tasks: List[asyncio.Task[EvalTestCase]] = []

    check_function_call_test_case_limited = check_function_call_test_case
    runner_concurrency = sandbox_config.dataset.max_runner_concurrency
    if config.language == 'csharp':
        # dotnet runtime startup is memory-heavy; cap C# stdio fanout to avoid CoreCLR OOM.
        csharp_max_concurrency = int(os.getenv('SANDBOX_CSHARP_CASE_CONCURRENCY', '1'))
        if runner_concurrency > 0:
            runner_concurrency = min(runner_concurrency, csharp_max_concurrency)
        else:
            runner_concurrency = csharp_max_concurrency

    if runner_concurrency > 0:
        check_function_call_test_case_limited = max_concurrency(
            runner_concurrency)(check_function_call_test_case)
            
    language = config.language
    for case in cases:
        fn_name = case['fn_name']
        input = case['input']
        output = case['output']
        full_code = concat_function_assertion(code, fn_name, input, output, language)
        task = asyncio.create_task(check_function_call_test_case_limited(full_code, case, config))
        tasks.append(task)
    
    run_all_cases = config.extra.get("run_all_cases", False)
    total_timeout = config.extra.get("total_timeout", 300)
    deadline = asyncio.get_running_loop().time() + total_timeout

    for task_idx, task in enumerate(tasks):
        instance_logger.info(f"check_function_call_test_cases_parallel, total_timeout: {total_timeout}, task_idx: {task_idx}, deadline: {deadline}, current_time: {asyncio.get_running_loop().time()}")
        if asyncio.get_running_loop().time() > deadline:
            instance_logger.error("check_function_call_test_cases_parallel timeout", total_timeout=total_timeout, task_idx=task_idx)
            for remaining_task in tasks:
                if not remaining_task.done():
                    time_out_outcome = EvalTestCase(passed=False, exec_info=RunCodeResponse(status=RunStatus.SandboxError, message= "Total Timeout"), test_info=None)
                    result.append(time_out_outcome)
                    remaining_task.cancel()
            break
        try:
            outcome = await task
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to check function call test case: {e}')
        result.append(outcome)

        if not run_all_cases and not outcome.passed:
            for remaining_task in tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            break

    return result


class _PythonExecCaseTimeout(Exception):
    pass


def _raise_python_exec_case_timeout(signum, frame):
    raise _PythonExecCaseTimeout("Case Timeout")


def _build_python_stdio_exec_response(
    elapsed: float,
    stdout: str,
    stderr: str,
    *,
    timed_out: bool = False,
    failed: bool = False,
    failure_message: str = "",
) -> RunCodeResponse:
    if timed_out:
        run_result = CommandRunResult(
            status=CommandRunStatus.TimeLimitExceeded,
            execution_time=elapsed,
            stdout=stdout,
            stderr=stderr,
        )
        return RunCodeResponse(
            status=RunStatus.Failed,
            message="Case Timeout",
            run_result=run_result,
            executor_pod_name=os.environ.get("MY_POD_NAME"),
        )

    run_result = CommandRunResult(
        status=CommandRunStatus.Finished,
        execution_time=elapsed,
        return_code=1 if failed else 0,
        stdout=stdout,
        stderr=stderr,
    )
    return RunCodeResponse(
        status=RunStatus.Failed if failed else RunStatus.Success,
        message=failure_message if failed else "",
        run_result=run_result,
        executor_pod_name=os.environ.get("MY_POD_NAME"),
    )


def _should_fallback_stdio_case(
    response: RunCodeResponse,
    case: GeneralStdioTest,
) -> bool:
    if response.status != RunStatus.Success or response.run_result is None:
        return False
    if response.run_result.return_code not in (None, 0):
        return False
    stdout = response.run_result.stdout or ""
    stderr = response.run_result.stderr or ""
    expected_stdout = case.output.get("stdout", "") if case.output else ""
    return stdout == "" and stderr == "" and expected_stdout.strip() != ""


def _finalize_stdio_outcome_sync(
    result: RunCodeResponse, case: GeneralStdioTest, config: TestConfig, lower_cmp: bool = True
) -> EvalTestCase:
    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != RunStatus.Success:
        return fail_case

    result_lines = result.run_result.stdout.strip().split('\n')
    expected_lines = case.output['stdout'].strip().split('\n')
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]

    special_judge_program = config.extra.get('special_judge_program', None)
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl) and float_equal(float(rl), float(el)):
                continue
            if special_judge_program is not None:
                raise RuntimeError("special_judge_program is not supported in sync stdio helper")
            return fail_case

    if not config.extra.get('return_full_case', False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


async def _finalize_stdio_outcome(
    result: RunCodeResponse, case: GeneralStdioTest, config: TestConfig, lower_cmp: bool = True
) -> EvalTestCase:
    special_judge_program = config.extra.get('special_judge_program', None)
    if special_judge_program is None:
        return _finalize_stdio_outcome_sync(result, case, config, lower_cmp)

    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != RunStatus.Success:
        return fail_case

    result_lines = result.run_result.stdout.strip().split('\n')
    expected_lines = case.output['stdout'].strip().split('\n')
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]

    special_judge_language = config.extra.get('special_judge_language', 'python')
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl) and float_equal(float(rl), float(el)):
                continue
            correct = await check_special_judge(
                stdin=case.input['stdin'],
                stdout=case.output['stdout'],
                answer=result.run_result.stdout,
                special_judge_program=special_judge_program,
                special_judge_language=special_judge_language
            )
            if correct:
                return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())
            return fail_case

    if not config.extra.get('return_full_case', False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


def _execute_python_code_with_real_stdio(
    compiled_code,
    stdin_text: str,
    per_case_timeout: int,
    *,
    code_text: Optional[str] = None,
) -> Dict[str, Any]:
    case_start = time.time()
    exception_type = None
    failure_message = ""
    stdout = ""
    stderr = ""
    exit_hook_failure = None

    try:
        signal.alarm(max(int(per_case_timeout), 1))
        with tempfile.TemporaryDirectory() as tmp_dir:
            main_path = os.path.join(tmp_dir, "__main__.py")
            stdin_path = os.path.join(tmp_dir, "stdin.txt")
            stdout_path = os.path.join(tmp_dir, "stdout.txt")
            stderr_path = os.path.join(tmp_dir, "stderr.txt")
            if code_text is not None:
                with open(main_path, "w", encoding="utf-8") as f:
                    f.write(code_text)
            with open(stdin_path, "w", encoding="utf-8") as f:
                f.write(stdin_text)
            original_cwd = os.getcwd()
            original_stdin = sys.stdin
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            original_sys_argv = sys.argv[:]
            original_stdin_fd = os.dup(0)
            original_stdout_fd = os.dup(1)
            original_stderr_fd = os.dup(2)

            def _safe_flush_stream_graph(stream, visited=None):
                if stream is None:
                    return
                if visited is None:
                    visited = set()
                stream_id = id(stream)
                if stream_id in visited:
                    return
                visited.add(stream_id)

                try:
                    flush = getattr(stream, "flush", None)
                    if callable(flush):
                        flush()
                except Exception:
                    pass

                for attr in ("buffer", "raw", "file"):
                    try:
                        child = getattr(stream, attr, None)
                    except Exception:
                        child = None
                    if child is not None and child is not stream:
                        _safe_flush_stream_graph(child, visited)

            def _safe_close_custom_stream(stream, protected_ids, visited=None):
                if stream is None:
                    return
                if visited is None:
                    visited = set()
                stream_id = id(stream)
                if stream_id in visited or stream_id in protected_ids:
                    return
                visited.add(stream_id)

                for attr in ("buffer", "raw", "file"):
                    try:
                        child = getattr(stream, attr, None)
                    except Exception:
                        child = None
                    if child is not None and child is not stream:
                        _safe_close_custom_stream(child, protected_ids, visited)

                _safe_flush_stream_graph(stream)
                try:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass

            try:
                with open(stdin_path, "r", encoding="utf-8", errors="replace") as stdin_file, \
                     open(stdout_path, "w", encoding="utf-8", errors="replace") as stdout_file, \
                     open(stderr_path, "w", encoding="utf-8", errors="replace") as stderr_file:
                    os.chdir(tmp_dir)
                    os.dup2(stdin_file.fileno(), 0)
                    os.dup2(stdout_file.fileno(), 1)
                    os.dup2(stderr_file.fileno(), 2)
                    sys.stdin = stdin_file
                    sys.stdout = stdout_file
                    sys.stderr = stderr_file
                    sys.argv = [main_path]
                    exec_globals = {"__name__": "__main__", "__file__": main_path}
                    local_exit_funcs = []
                    exec_timed_out = False
                    original_atexit_register = atexit.register
                    original_atexit_unregister = getattr(atexit, "unregister", None)
                    original_sys_exit = sys.exit
                    original_builtins_exit = getattr(builtins, "exit", None)
                    original_builtins_quit = getattr(builtins, "quit", None)

                    def _local_atexit_register(func, *args, **kwargs):
                        local_exit_funcs.append((func, args, kwargs))
                        return func

                    def _local_atexit_unregister(func):
                        local_exit_funcs[:] = [
                            item for item in local_exit_funcs if item[0] != func
                        ]

                    def _local_exit(code=0):
                        _safe_flush_stream_graph(sys.stdout)
                        _safe_flush_stream_graph(sys.stderr)
                        raise SystemExit(code)

                    atexit.register = _local_atexit_register
                    if original_atexit_unregister is not None:
                        atexit.unregister = _local_atexit_unregister
                    sys.exit = _local_exit
                    builtins.exit = _local_exit
                    builtins.quit = _local_exit
                    try:
                        try:
                            exec(compiled_code, exec_globals, exec_globals)
                        except _PythonExecCaseTimeout:
                            exec_timed_out = True
                            raise
                    finally:
                        protected_stream_ids = {
                            id(stdin_file),
                            id(stdout_file),
                            id(stderr_file),
                            id(original_stdin),
                            id(original_stdout),
                            id(original_stderr),
                        }
                        if not exec_timed_out:
                            while local_exit_funcs:
                                func, args, kwargs = local_exit_funcs.pop()
                                try:
                                    func(*args, **kwargs)
                                except Exception:
                                    if exit_hook_failure is None:
                                        exit_hook_failure = traceback.format_exc()

                            _safe_flush_stream_graph(sys.stdout)
                            _safe_flush_stream_graph(sys.stderr)
                            _safe_close_custom_stream(sys.stdout, protected_stream_ids)
                            _safe_close_custom_stream(sys.stderr, protected_stream_ids)
                            _safe_flush_stream_graph(stdout_file)
                            _safe_flush_stream_graph(stderr_file)

                        atexit.register = original_atexit_register
                        if original_atexit_unregister is not None:
                            atexit.unregister = original_atexit_unregister
                        sys.exit = original_sys_exit
                        if original_builtins_exit is not None:
                            builtins.exit = original_builtins_exit
                        if original_builtins_quit is not None:
                            builtins.quit = original_builtins_quit
            finally:
                for stream in (sys.stdout, sys.stderr):
                    try:
                        stream.flush()
                    except Exception:
                        pass
                os.dup2(original_stdin_fd, 0)
                os.dup2(original_stdout_fd, 1)
                os.dup2(original_stderr_fd, 2)
                os.close(original_stdin_fd)
                os.close(original_stdout_fd)
                os.close(original_stderr_fd)
                sys.stdin = original_stdin
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                sys.argv = original_sys_argv
                os.chdir(original_cwd)
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                stderr = f.read()
            if exit_hook_failure:
                exception_type = exception_type or "exception"
                if failure_message:
                    stderr = f"{stderr}\n{exit_hook_failure}".strip() if stderr else exit_hook_failure
                else:
                    failure_message = exit_hook_failure
    except _PythonExecCaseTimeout:
        exception_type = "timeout"
        failure_message = "Case Timeout"
    except AssertionError:
        exception_type = "assertion"
        failure_message = "Wrong Answer"
    except SystemExit as exc:
        code_value = exc.code
        if code_value not in (None, 0):
            exception_type = "system_exit"
            failure_message = f"SystemExit: {code_value}"
    except BaseException:
        exception_type = "exception"
        failure_message = traceback.format_exc()
    finally:
        signal.alarm(0)

    return {
        "elapsed": time.time() - case_start,
        "stdout": stdout,
        "stderr": stderr,
        "exception_type": exception_type,
        "failure_message": failure_message,
    }


def _execute_python_stdio_code_once(
    code: str,
    compiled_code,
    stdin_text: str,
    per_case_timeout: int,
) -> Dict[str, Any]:
    raw = _execute_python_code_with_real_stdio(
        compiled_code,
        stdin_text,
        per_case_timeout,
        code_text=code,
    )
    timed_out = raw["exception_type"] == "timeout"
    failed = raw["exception_type"] not in (None, "timeout")
    stderr = raw["stderr"]
    if raw["failure_message"]:
        stderr = f"{stderr}\n{raw['failure_message']}".strip() if stderr else raw["failure_message"]
    return {
        "elapsed": raw["elapsed"],
        "stdout": raw["stdout"],
        "stderr": stderr,
        "timed_out": timed_out,
        "failed": failed,
        "failure_message": raw["failure_message"],
    }


def _python_stdio_exec_worker(
    code: str,
    cases: List[GeneralStdioTest],
    start_idx: int,
    per_case_timeout: int,
    run_all_cases: bool,
    result_conn,
) -> None:
    signal.signal(signal.SIGALRM, _raise_python_exec_case_timeout)

    try:
        compiled_code = compile(code, "__main__.py", "exec")
    except BaseException:
        compile_error = traceback.format_exc()
        cases_to_emit = cases if run_all_cases else cases[:1]
        for idx, _case in enumerate(cases_to_emit, start_idx):
            result_conn.send(
                {
                    "idx": idx,
                    "elapsed": 0.0,
                    "stdout": "",
                    "stderr": compile_error,
                    "timed_out": False,
                    "failed": True,
                    "failure_message": compile_error,
                }
            )
        result_conn.send({"done": True})
        result_conn.close()
        return

    for idx, case in enumerate(cases, start_idx):
        raw = _execute_python_stdio_code_once(code, compiled_code, case.input["stdin"], per_case_timeout)
        result_conn.send(
            {
                "idx": idx,
                **raw,
            }
        )
        if not run_all_cases and (raw["timed_out"] or raw["failed"]):
            break
    result_conn.send({"done": True})
    result_conn.close()


def _build_python_function_call_exec_response(
    passed: bool,
    elapsed: float,
    stdout: str,
    stderr: str,
    *,
    failure_message: str = "Wrong Answer",
    timed_out: bool = False,
) -> RunCodeResponse:
    if timed_out:
        run_result = CommandRunResult(
            status=CommandRunStatus.TimeLimitExceeded,
            execution_time=elapsed,
            stdout=stdout,
            stderr=stderr,
        )
        return RunCodeResponse(
            status=RunStatus.Failed,
            message="Case Timeout",
            run_result=run_result,
            executor_pod_name=os.environ.get("MY_POD_NAME"),
        )

    run_result = CommandRunResult(
        status=CommandRunStatus.Finished,
        execution_time=elapsed,
        return_code=0 if passed else 1,
        stdout=stdout,
        stderr=stderr,
    )
    return RunCodeResponse(
        status=RunStatus.Success if passed else RunStatus.Failed,
        message="" if passed else failure_message,
        run_result=run_result,
        executor_pod_name=os.environ.get("MY_POD_NAME"),
    )


def _should_fallback_function_call_case(message: Dict[str, Any]) -> bool:
    if message.get("passed") or message.get("timed_out"):
        return False
    stderr = message.get("stderr") or ""
    return "ModuleNotFoundError" in stderr


def _python_function_call_exec_worker(
    full_codes: List[str],
    cases: List[Dict[str, Any]],
    start_idx: int,
    per_case_timeout: int,
    run_all_cases: bool,
    result_conn,
) -> None:
    signal.signal(signal.SIGALRM, _raise_python_exec_case_timeout)

    for idx, (full_code, _case) in enumerate(zip(full_codes, cases), start_idx):
        passed = False
        timed_out = False
        error_message = None
        failure_message = "Wrong Answer"

        try:
            compiled_full_code = compile(full_code, "__main__.py", "exec")
            raw = _execute_python_code_with_real_stdio(
                compiled_full_code,
                "",
                per_case_timeout,
                code_text=full_code,
            )
            if raw["exception_type"] is None:
                passed = True
            elif raw["exception_type"] == "timeout":
                timed_out = True
                error_message = "Case Timeout"
            elif raw["exception_type"] == "assertion":
                error_message = "Wrong Answer"
            else:
                failure_message = "Execution Failed"
                error_message = raw["failure_message"]
            elapsed = raw["elapsed"]
            stdout = raw["stdout"]
            stderr = raw["stderr"]
        except BaseException:
            elapsed = 0.0
            stdout = ""
            stderr = traceback.format_exc()
            failure_message = "Execution Failed"
            error_message = None

        if error_message:
            stderr = f"{stderr}\n{error_message}".strip() if stderr else error_message

        result_conn.send(
            {
                "idx": idx,
                "passed": passed,
                "timed_out": timed_out,
                "elapsed": elapsed,
                "stdout": stdout,
                "stderr": stderr,
                "failure_message": failure_message,
            }
        )

        if not run_all_cases and not passed:
            break
    result_conn.send({"done": True})
    result_conn.close()


def _chunk_cases(cases: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]


async def _consume_python_subworkers(
    worker_specs: List[Dict[str, Any]],
    max_active_workers: int,
    total_cases: int,
    total_timeout: int,
    run_all_cases: bool,
    build_outcome,
):
    active_workers = []
    next_worker_idx = 0
    stop_dispatch = False

    def start_worker(spec):
        recv_conn, send_conn = spec["mp_ctx"].Pipe(duplex=False)
        process = spec["mp_ctx"].Process(target=spec["target"], args=(*spec["args"], send_conn))
        process.start()
        send_conn.close()
        return {"process": process, "conn": recv_conn, "done": False}

    while next_worker_idx < len(worker_specs) and len(active_workers) < max_active_workers:
        active_workers.append(start_worker(worker_specs[next_worker_idx]))
        next_worker_idx += 1

    outcomes = []
    seen_indices = set()
    total_timed_out = False
    deadline = time.time() + total_timeout
    while active_workers:
        remaining_timeout = deadline - time.time()
        if remaining_timeout <= 0:
            total_timed_out = True
            for worker in active_workers:
                if worker["process"].is_alive():
                    worker["process"].kill()
            break

        made_progress = False
        for worker in active_workers:
            if worker["done"]:
                continue
            while True:
                try:
                    if not worker["conn"].poll(0):
                        break
                    message = worker["conn"].recv()
                    made_progress = True
                except (EOFError, OSError):
                    worker["done"] = True
                    break
                if message.get("done"):
                    worker["done"] = True
                    break
                seen_indices.add(message["idx"])
                outcome = await build_outcome(message)
                outcomes.append(outcome)
                if not run_all_cases and not outcome.passed:
                    stop_dispatch = True
                    for item in active_workers:
                        if item["process"].is_alive():
                            item["process"].kill()
                        item["done"] = True
                    break
            if all(item["done"] for item in active_workers):
                break
            if not worker["process"].is_alive() and not worker["conn"].poll():
                worker["done"] = True

        finished = []
        for idx, worker in enumerate(active_workers):
            if worker["done"]:
                worker["conn"].close()
                worker["process"].join(timeout=1)
                if worker["process"].is_alive():
                    worker["process"].kill()
                    worker["process"].join()
                finished.append(idx)
        for idx in reversed(finished):
            del active_workers[idx]
        while (
            not stop_dispatch
            and next_worker_idx < len(worker_specs)
            and len(active_workers) < max_active_workers
        ):
            active_workers.append(start_worker(worker_specs[next_worker_idx]))
            next_worker_idx += 1

        if not made_progress and active_workers:
            await asyncio.sleep(min(0.05, max(remaining_timeout, 0.0)))

    for worker in active_workers:
        worker["conn"].close()
        worker["process"].join(timeout=1)
        if worker["process"].is_alive():
            worker["process"].kill()
            worker["process"].join()

    if total_timed_out and len(outcomes) < total_cases:
        missing = [idx for idx in range(total_cases) if idx not in seen_indices]
        return outcomes, total_timed_out, missing
    return outcomes, total_timed_out, []


async def check_python_stdio_test_cases_single_worker_exec(
    code: str,
    cases: List[GeneralStdioTest],
    config: TestConfig,
    lower_cmp: bool = True,
) -> List[EvalTestCase]:
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    all_tests_start_time = time.time()
    total_timeout = int(config.extra.get("total_timeout", DEFAULT_COMMON_BATCH_TOTAL_TIMEOUT))
    per_case_timeout = int(config.run_timeout or 10)
    run_all_cases = config.extra.get("run_all_cases", False)
    cases_per_subworker = max(int(getattr(sandbox_config.dataset, "cases_per_subworker", 1) or 1), 1)
    max_subworkers = max(int(getattr(sandbox_config.dataset, "max_runner_concurrency", 1) or 1), 1)
    
    if config.language == 'csharp':
        # dotnet runtime startup is memory-heavy; cap C# function-call fanout to avoid CoreCLR OOM.
        csharp_max_concurrency = int(os.getenv('SANDBOX_CSHARP_CASE_CONCURRENCY', '1'))
        if max_subworkers > 0:
            max_subworkers = min(max_subworkers, csharp_max_concurrency)
        else:
            max_subworkers = csharp_max_concurrency

    try:
        mp_ctx = multiprocessing.get_context("fork")
    except ValueError:
        mp_ctx = multiprocessing.get_context()

    serializable_cases = [GeneralStdioTest(**case.model_dump()) for case in cases]
    chunks = _chunk_cases(serializable_cases, cases_per_subworker)
    worker_specs = []
    start_idx = 0
    for chunk in chunks:
        worker_specs.append(
            {
                "mp_ctx": mp_ctx,
                "target": _python_stdio_exec_worker,
                "args": (code, chunk, start_idx, per_case_timeout, run_all_cases),
            }
        )
        start_idx += len(chunk)

    config.extra = config.extra or {}
    fallback_case_indices = config.extra.setdefault("debug_stdio_exec_fallback_case_indices", [])

    async def build_outcome(message):
        case = cases[message["idx"]]
        response = _build_python_stdio_exec_response(
            message["elapsed"],
            message["stdout"],
            message["stderr"],
            timed_out=message["timed_out"],
            failed=message["failed"],
            failure_message=message["failure_message"],
        )
        if _should_fallback_stdio_case(response, case):
            fallback_case_indices.append(message["idx"])
            return await check_stdio_test_case(code, case, config, lower_cmp)
        return await _finalize_stdio_outcome(response, case, config, lower_cmp)

    outcomes, total_timed_out, missing = await _consume_python_subworkers(
        worker_specs, max_subworkers, len(cases), total_timeout, run_all_cases, build_outcome
    )

    if total_timed_out and missing:
        for idx in missing:
            case = cases[idx]
            outcomes.append(
                EvalTestCase(
                    passed=False,
                    exec_info=RunCodeResponse(status=RunStatus.SandboxError, message="Total Timeout"),
                    test_info=case.model_dump(),
                )
            )

    all_tests_execution_time = time.time() - all_tests_start_time
    if not hasattr(config, "extra") or config.extra is None:
        config.extra = {}
    config.extra["all_tests_execution_time"] = all_tests_execution_time
    config.extra["debug_stdio_exec_fallback_case_count"] = len(fallback_case_indices)
    instance_logger.info(
        f"Python stdio execution time: {all_tests_execution_time:.3f} seconds for {len(cases)} test cases "
        f"(subworkers={min(len(chunks), max_subworkers)}, cases_per_subworker={cases_per_subworker}, "
        f"fallback_cases={len(fallback_case_indices)})"
    )
    return outcomes


async def check_python_function_call_test_cases_single_worker_exec(
    code: str,
    cases: List[Dict[str, Any]],
    config: TestConfig,
) -> List[EvalTestCase]:
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    all_tests_start_time = time.time()
    total_timeout = int(config.extra.get("total_timeout", DEFAULT_COMMON_BATCH_TOTAL_TIMEOUT))
    per_case_timeout = int(config.run_timeout or 10)
    run_all_cases = config.extra.get("run_all_cases", False)
    cases_per_subworker = max(int(getattr(sandbox_config.dataset, "cases_per_subworker", 1) or 1), 1)
    max_subworkers = max(int(getattr(sandbox_config.dataset, "max_runner_concurrency", 1) or 1), 1)
    
    if config.language == 'csharp':
        # dotnet runtime startup is memory-heavy; cap C# function-call fanout to avoid CoreCLR OOM.
        csharp_max_concurrency = int(os.getenv('SANDBOX_CSHARP_CASE_CONCURRENCY', '1'))
        if max_subworkers > 0:
            max_subworkers = min(max_subworkers, csharp_max_concurrency)
        else:
            max_subworkers = csharp_max_concurrency

    full_codes = []
    for case in cases:
        full_codes.append(
            concat_function_assertion(
                code,
                case["fn_name"],
                case["input"],
                case["output"],
                config.language,
            )
        )

    try:
        mp_ctx = multiprocessing.get_context("fork")
    except ValueError:
        mp_ctx = multiprocessing.get_context()

    case_chunks = _chunk_cases(cases, cases_per_subworker)
    code_chunks = _chunk_cases(full_codes, cases_per_subworker)
    worker_specs = []
    start_idx = 0
    for chunk_cases, chunk_codes in zip(case_chunks, code_chunks):
        worker_specs.append(
            {
                "mp_ctx": mp_ctx,
                "target": _python_function_call_exec_worker,
                "args": (chunk_codes, chunk_cases, start_idx, per_case_timeout, run_all_cases),
            }
        )
        start_idx += len(chunk_cases)

    config.extra = config.extra or {}
    fallback_case_indices = config.extra.setdefault("debug_function_call_exec_fallback_case_indices", [])

    async def build_outcome(message):
        if _should_fallback_function_call_case(message):
            idx = message["idx"]
            fallback_case_indices.append(idx)
            return await check_function_call_test_case(full_codes[idx], cases[idx], config)
        return EvalTestCase(
            passed=message["passed"],
            exec_info=_build_python_function_call_exec_response(
                message["passed"],
                message["elapsed"],
                message["stdout"],
                message["stderr"],
                failure_message=message["failure_message"],
                timed_out=message["timed_out"],
            ),
            test_info=cases[message["idx"]],
        )

    outcomes, total_timed_out, missing = await _consume_python_subworkers(
        worker_specs, max_subworkers, len(cases), total_timeout, run_all_cases, build_outcome
    )

    if total_timed_out and missing:
        for idx in missing:
            case = cases[idx]
            outcomes.append(
                EvalTestCase(
                    passed=False,
                    exec_info=RunCodeResponse(status=RunStatus.SandboxError, message="Total Timeout"),
                    test_info=case,
                )
            )

    all_tests_execution_time = time.time() - all_tests_start_time
    if not hasattr(config, "extra") or config.extra is None:
        config.extra = {}
    config.extra["all_tests_execution_time"] = all_tests_execution_time
    config.extra["debug_function_call_exec_fallback_case_count"] = len(fallback_case_indices)
    instance_logger.info(
        f"Python function_call execution time: {all_tests_execution_time:.3f} seconds for {len(cases)} test cases "
        f"(subworkers={min(len(case_chunks), max_subworkers)}, cases_per_subworker={cases_per_subworker}, "
        f"fallback_cases={len(fallback_case_indices)})"
    )
    return outcomes


def parse_jest_cases(report_data: str) -> List[Dict[str, Any]]:
    if isinstance(report_data, str):
        report = json.loads(report_data)
    else:
        report = report_data

    test_cases = []

    for test_suite in report['testResults']:
        file_path = test_suite['testFilePath']

        for test_case in test_suite['testResults']:
            result = {
                'passed': test_case['status'] == 'passed',
                'full_name': test_case['fullName'],
                'file': file_path,
                'suite': ' > '.join(test_case['ancestorTitles']),
                'test': test_case['title'],
                'failure_messages': test_case['failureMessages']
            }
            test_cases.append(result)

    return test_cases
