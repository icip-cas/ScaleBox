<div align="center">

<img src="./docs/static/img/scalebox_logo.png" width="70%">

<div align="center">
    ⚡ A scalable sandbox for distributed code execution, RL training and unified benchmarking
    <br>
    <br>
</div>

[![GitHub Repo stars](https://img.shields.io/github/stars/icip-cas/ScaleBox)](https://github.com/icip-cas/ScaleBox/stargazers)
[![LICENSE](https://img.shields.io/pypi/l/evalplus)](https://github.com/icip-cas/ScaleBox/blob/main/LICENSE)
</div>

## 📋 Table of Contents
- [✨ Highlights](#-highlights)
- [🎯 Features](#-features)
- [🚀 Usage](#-usage)
- [📊 Unified Evaluation](#-unified-evaluation)
- [🧾 Special Judge Generation](#-special-judge-generation)
- [📝 Citation](#-citation)
- [🙏 Acknowledgement](#-acknowledgement)
- [📄 License](#-license)

## ✨ Highlights

- **High-Performance Distributed Engine**
    - Elastic Scaling: Optimized for large-scale multi-machine distributed deployment with built-in load balancing and real-time service monitoring.
    - Massive Parallelization: Native support for instance-level and unit-test parallelization.
    - Superior Throughput: ScaleBox achieves up to 2.63× (x86) and 2.53× (ARM) throughput compared to SandboxFusion, and 1.59× over verl <sub>Prime</sub> in large-scale settings. Detailed results are available in [efficency test report](./reports/efficiency_test_report.md).

- **RL-Native Integration**
    - Full Ecosystem Compatibility: Out-of-the-box support for mainstream RL frameworks including [verl](./integrations/verl/verl_integration_guide.md) (NVIDIA/Ascend) and [MindSpeed-RL](./integrations/mindspeed_rl/mindspeed_rl_integration_guide.md) (Ascend).
    - Unified RL Interface: A single request interface for diverse training paradigms: stdin-stdout (with Special Judge), function call, and assert (MultiPL-E format).
    - Hybrid Deployment: Seamless [one-click deployment](deploy/start_verl_sandbox.sh) using Docker environments that bridge RL training and sandbox execution.

- **Precision Evaluation & Automation**
    - Automated Special Judge: A lightweight pipeline that automatically classifies complex problems and generates custom checkers for floating-point tolerance or multiple valid solutions.
    - Unified Benchmark Suite: Simplified one-click evaluation for common code benchmarks, supporting both Instruct-type and Reasoning-type models.

- **Production-Ready Usability**
    - Robust Monitoring: Integrated with Nginx log integration, automated service restarts, and detailed error tracking.
    - Modern Ecosystem: Native support for MCP (Model Context Protocol) and flexible parameter configuration for rapid research iterations.

## 🎯 Features

**Code Runner**: Run and return the result of a code snippet.

<details>
<summary>Supported 21 languages:</summary>

- Python (python, pytest)
- C++
- C#
- Go (go, go test)
- Java (javac, junit)
- NodeJS
- Typescript (tsx, jest)
- Scala
- Kotlin
- PHP
- Rust
- Bash
- Lua
- R
- Perl
- D
- Ruby
- Julia
- Verilog
- CUDA (GPU)
- Python (GPU)

</details>

</br>

**Unified Evaluation**: A unified evaluation interface for code generation tasks.

- [common_evaluate_batch](#calling-the-sandbox)

## 🚀 Usage

### 📦 Installation

**Docker**

Use the provided docker image:
- x86 platform: `quay.io/jszheng/scalebox:x86-20260331` 
- arm64 platform: `quay.io/jszheng/scalebox:arm64-20260331` 

Or, build the docker image locally:

```bash
# x86 platform
docker build --rm -f ./scripts/Dockerfile.x86 -t scalebox:v1 .

# arm64 platform
docker build --rm -f ./scripts/Dockerfile.arm64 -t scalebox:v1 .
```

### 🌐 Deployment

#### 🔧 Environment Variables
Before deployment, configure the following environment variables:
```bash
export HOST=0.0.0.0           # Server host address
export PORT=8080              # Server port
export WORKERS=32             # Number of parallel workers for uvicorn (set 1 for single CPU)
export MAX_MEM=50000000       # Maximum memory limit per process in KB (50GB), or 'unlimited'
export SAVE_BAD_CASES=false   # Set 'true' to save bad cases for debugging in 'output/{datetime}/'
```

#### 💻 Single-Node Deployment
For running the sandbox on a single machine:

```bash
# Start the server with basic configuration
make run-online

# OR use supervisor for automatic restart on failure
bash deploy/start_sandbox_with_supervisor.sh
```

#### 🌍 Distributed Deployment
Running the following command, then by checking `MASTER_IP` it will deploy nginx on the main node, and deploy sandbox on each worker node:

```bash
export NGINX_PORT=8081              # nginx will run on this port
bash deploy/start_distributed.sh
```

#### 🔄 Hot-Update the Distributed Setup
To dynamically add or remove worker nodes with hot updates (without full service restart):

1. Start/stop worker nodes using the worker node setup instructions above.
2. Re-run `bash deploy/start_distributed_nginx.sh` on the main node to trigger a dynamic hot-update. The nginx upstream configuration is refreshed in place and will automatically include all currently available worker nodes.

#### 🐳 Docker Deployment
To run the sandbox server using Docker with health check and automatic restart on failure:

```bash
docker run \
    --privileged \
    -p 8080:8080 \
    -p 8081:8081 \
    --volume ~/scalebox:/scalebox \
    -w /scalebox \
    --health-cmd='python /scalebox/deploy/a_plus_b.py || exit 1' \
    --health-interval=2s \
    -itd \
    --restart unless-stopped \
    quay.io/jszheng/scalebox:x86-20260331 \
    make run-online
```

### 🔌 Calling the sandbox
In addition to the originally provided dataset-specific evaluation APIs, we also provide a unified evaluation API, which includes both stdio and function call evaluation modes on various languages.

<details>
<summary>The description of API parameters are as follows:</summary>

- completion: The code to be evaluated, in the form of markdown code block.
- config: The configuration for the evaluation
    - language: The language of the code.
    - compile_timeout: The timeout for the code to be compiled. Default to 10.
    - run_timeout: The timeout for the code to be run. Default to 10.
    - provided_data: The data for the evaluation.
        - test_cases: The test cases for the evaluation.
            - type: The type of the test cases, either `stdin_stdout` or `function_call`.
            - input: The input for the test cases. For `stdin_stdout`, the format is `["input_1", "input_2", ..., "input_n"]`; for `function_call`, the format is `[[input_1_1, input_1_2, ..., input_1_k], [input_2_1, input_2_2, ..., input_2_k], ..., [input_n_1, input_n_2, ..., input_n_k]]`.
            - output: The output for the test cases. For `stdin_stdout`, the format is `["output_1", "output_2", ..., "output_n"]`; for `function_call`, the format is `[[output_1], [output_2], .., [output_n]]`.
            - fn_name: The name of the function to be evaluated.
            - json_input: Whether the input needs to be [split by '\n' and loaded as json](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py#L246). Default to False.
- extra: The extra configuration for the evaluation.
    - run_all_cases: Whether to run all test cases if one test case failed.
    - total_timeout: After which the unit tests will not be executed, while the already running unit tests will continue to run until `run_timeout` is reached. Default to 300.

</details>

#### Uasge Example

Here is an example of how to use the `common_evaluate_batch` API for testing a+b problem with standard stdin-stdout format.
```python
# stdio evaluate
payload = {
    "completion": """```python\na, b = map(int, input().split())\nprint(a + b)\n```""",
    "config": {
        "language": "python",
        "run_timeout": 10,
        "provided_data": { 
            "test_cases": 
                {"type": "stdin_stdout", "input": ["1 2", "3 4"], "output": ["3", "7"], "fn_name": None},            
        },
        "extra": {
            "run_all_cases": True
        }
    }
}

response = requests.post('http://0.0.0.0:8080/common_evaluate_batch', json=payload)
result = response.json()
```

<details>
<summary>Response</summary>

```json
{
    "id": 0,
    "accepted": true,
    "extracted_code": "a, b = map(int, input().split())\nprint(a + b)",
    "full_code": null,
    "test_code": null,
    "tests": [
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": null,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.0040967464447021484,
                    "return_code": 0,
                    "stdout": "3\n",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": {
                "input": {
                    "stdin": "1 2"
                },
                "output": {
                    "stdout": "3"
                }
            }
        },
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": null,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.017037630081176758,
                    "return_code": 0,
                    "stdout": "7\n",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": {
                "input": {
                    "stdin": "3 4"
                },
                "output": {
                    "stdout": "7"
                }
            }
        }
    ],
    "extracted_type": null,
    "extra": null
}
```

</details>

Also an example of function call evaluation for the same problem:
```python
# function evaluate batch
payload = {
    "completion": """```python\ndef add(a, b):\n    return a + b\n```""",
    "config": {
        "language": "python",
        "provided_data": { 
            "test_cases": 
                {"type": "function_call", "input": [[1, 2], [3, 4]], "output": [[3], [7]], "fn_name": "add", "json_input": False},            
        },
        "extra": {
            "run_all_cases": True,
            "total_timeout": 1
        }
    }
}

response = requests.post('http://0.0.0.0:8080/common_evaluate_batch', json=payload)
result = response.json()
```

<details>
<summary>Response</summary>

```json
{
    "id": 0,
    "accepted": true,
    "extracted_code": "def add(a, b):\n    return a + b",
    "full_code": null,
    "test_code": null,
    "tests": [
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": null,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.00021147727966308594,
                    "return_code": 0,
                    "stdout": "",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": {
                "type": "function_call",
                "fn_name": "add",
                "input": [1, 2],
                "output": [3]
            }
        },
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": null,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.01851511001586914,
                    "return_code": 0,
                    "stdout": "",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": {
                "type": "function_call",
                "fn_name": "add",
                "input": [3, 4],
                "output": [7]
            }
        }
    ],
    "extracted_type": null,
    "extra": null
}
```

</details>

#### Other Usage Examples

<details>
<summary>Special Judge Evaluation Examples:</summary>

Here is an example of stdin-stdout special judge evaluation. Given the input number `c`, the output number `a` and `b` should satisfy `a + b == c`. 
The special judge program should read the file path of `stdin.txt`, `stdout.txt` and `answer.txt` to get the input, output and answer, and return exit code 0 if the output is correct, otherwise return exit code 1.

```python
payload = {
    "completion": """```python\nc = int(input())\nprint(c-1, 1)\n```""",
    "config": {
        "language": "python",
        "run_timeout": 10,
        "provided_data": { 
            "test_cases": 
                {"type": "stdin_stdout", "output": ["1 2", "3 4"], "input": ["3", "7"], "fn_name": None},            
        },
        "extra": {
            "run_all_cases": True,
            "special_judge_program": '''import sys\n\ndef read_file(filepath):\n    """Read file content and return lines."""\n    with open(filepath, 'r') as f:\n        return f.read().strip().split('\\n')\n\n\ndef validate_solution(stdin_path, stdout_path, answer_path):\n    """Validate the participant's solution."""\n    \n    stdin_lines = read_file(stdin_path)\n    stdout_lines = read_file(stdout_path)\n    participant_output = read_file(answer_path)\n\n    a, b = map(int, participant_output[0].split())\n    c = a + b\n    expected_output = int(stdin_lines[0])\n    return c == expected_output\n\n    \nstdin_path = "stdin.txt"\nstdout_path = "stdout.txt"\nanswer_path = "answer.txt"\n\nis_valid = validate_solution(stdin_path, stdout_path, answer_path)\n\nif is_valid:\n    sys.exit(0)\nelse:\n    sys.exit(1)''',
            "special_judge_language": "python",
        }
    }
}

response = requests.post('http://0.0.0.0:8080/common_evaluate_batch', json=payload)
result = response.json()
```

<details>
<summary>Response</summary>

```json
{
  "id": 0,
  "accepted": true,
  "extracted_code": "c = int(input())\nprint(c-1, 1)",
  "full_code": null,
  "test_code": null,
  "tests": [
    {
      "passed": true,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.004090547561645508,
          "return_code": 0,
          "stdout": "2 1\n",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": {
        "input": {
          "stdin": "3"
        },
        "output": {
          "stdout": "1 2"
        }
      }
    },
    {
      "passed": true,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.027703046798706055,
          "return_code": 0,
          "stdout": "6 1\n",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": {
        "input": {
          "stdin": "7"
        },
        "output": {
          "stdout": "3 4"
        }
      }
    }
  ],
  "extracted_type": null,
  "extra": null
}
```

</details>

</details>


<details>
<summary>MultiPL-E Evaluation Examples:</summary>


An example of assert evaluation from MultiPL-E cpp:
```python
# function evaluate batch
payload = {
    "completion": "```cpp\n#include <bits/stdc++.h>\nusing namespace std;\n\n// Write a cpp function to identify non-prime numbers.\nbool is_not_prime(long n) {\n    // Handle corner cases\n    if (n <= 1) return true;\n    if (n <= 3) return false;\n\n    // This is checked so that we can skip \n    // middle five numbers in below loop\n    if (n % 2 == 0 || n % 3 == 0) return true;\n\n    for (long i = 5; i * i <= n; i += 6)\n        if (n % i == 0 || n % (i + 2) == 0)\n            return true;\n\n    return false;\n}",
    "config": {
        "language": "cpp",
        "provided_data": { 
            "test_cases": {
                "type": "assert", 
                "tests": "}\nint main() {\n    auto candidate = is_not_prime;\n    assert(candidate((2)) == (false));\n    assert(candidate((10)) == (true));\n    assert(candidate((35)) == (true));\n    assert(candidate((37)) == (false));\n}\n", 
                "stop_tokens": ["\n}"]},            
        },
        "extra": {
            "run_all_cases": True,
            "total_timeout": 1
        }
    }
}

response = requests.post('http://0.0.0.0:8080/common_evaluate_batch', json=payload)
result = response.json()
```

<details>
<summary>Response</summary>

```json
{
    "id": 0,
    "accepted": true,
    "extracted_code": "#include <bits/stdc++.h>\nusing namespace std;\n\n// Write a cpp function to identify non-prime numbers.\nbool is_not_prime(long n) {\n    // Handle corner cases\n    if (n <= 1) return true;\n    if (n <= 3) return false;\n\n    // This is checked so that we can skip \n    // middle five numbers in below loop\n    if (n % 2 == 0 || n % 3 == 0) return true;\n\n    for (long i = 5; i * i <= n; i += 6)\n        if (n % i == 0 || n % (i + 2) == 0)\n            return true;\n\n    return false;",
    "full_code": "using namespace std;\n#include<optional>\n#include<cassert>\n#include<stdlib.h>\n#include<algorithm>\n#include<cmath>\n#include<math.h>\n#include<numeric>\n#include<stdio.h>\n#include<vector>\n#include<set>\n#include<map>\n#include<queue>\n#include<stack>\n#include<list>\n#include<deque>\n#include<boost/any.hpp>\n#include<string>\n#include<climits>\n#include<cstring>\n#include<iostream>\n#include<sstream>\n#include<fstream>\n#include <bits/stdc++.h>\nusing namespace std;\n\n// Write a cpp function to identify non-prime numbers.\nbool is_not_prime(long n) {\n    // Handle corner cases\n    if (n <= 1) return true;\n    if (n <= 3) return false;\n\n    // This is checked so that we can skip \n    // middle five numbers in below loop\n    if (n % 2 == 0 || n % 3 == 0) return true;\n\n    for (long i = 5; i * i <= n; i += 6)\n        if (n % i == 0 || n % (i + 2) == 0)\n            return true;\n\n    return false;\n}\nint main() {\n    auto candidate = is_not_prime;\n    assert(candidate((2)) == (false));\n    assert(candidate((10)) == (true));\n    assert(candidate((35)) == (true));\n    assert(candidate((37)) == (false));\n}\n",
    "test_code": null,
    "tests": [
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": {
                    "status": "Finished",
                    "execution_time": 1.4092826843261719,
                    "return_code": 0,
                    "stdout": "",
                    "stderr": ""
                },
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.0036695003509521484,
                    "return_code": 0,
                    "stdout": "",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": null
        }
    ],
    "extracted_type": null,
    "extra": null
}
```

</details>


</details>



<details>
<summary>HumanEval Evaluation Examples:</summary>


An example of assert evaluation from HumanEval:
```python
# function evaluate batch
payload = {
    "completion": "```python\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```",
    "config": {
        "language": "python",
        "provided_data": { 
            "test_cases": {
                "type": "assert", 
                "test":  "\n\nMETADATA = {}\n\n\ndef check(candidate):\n    assert candidate(6) == False\n    assert candidate(101) == True\n    assert candidate(11) == True\n    assert candidate(13441) == True\n    assert candidate(61) == True\n    assert candidate(4) == False\n    assert candidate(1) == False\n    assert candidate(5) == True\n    assert candidate(11) == True\n    assert candidate(17) == True\n    assert candidate(5 * 17) == False\n    assert candidate(11 * 7) == False\n    assert candidate(13441 * 19) == False\n\n", 
                "entry_point": "is_prime",
            },            
        },
    }
}


response = requests.post('http://0.0.0.0:8080/common_evaluate_batch', json=payload)
result = response.json()
```

<details>
<summary>Response</summary>

```json
{
    "id": 0,
    "accepted": true,
    "extracted_code": "def is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True",
    "full_code": "import math\nimport re\nimport sys\nimport copy\nimport datetime\nimport itertools\nimport collections\nimport heapq\nimport statistics\nimport functools\nimport hashlib\nimport numpy\nimport numpy as np\nimport string\nfrom typing import *\nfrom collections import *\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n\nMETADATA = {}\n\n\ndef check(candidate):\n    assert candidate(6) == False\n    assert candidate(101) == True\n    assert candidate(11) == True\n    assert candidate(13441) == True\n    assert candidate(61) == True\n    assert candidate(4) == False\n    assert candidate(1) == False\n    assert candidate(5) == True\n    assert candidate(11) == True\n    assert candidate(17) == True\n    assert candidate(5 * 17) == False\n    assert candidate(11 * 7) == False\n    assert candidate(13441 * 19) == False\n\n\ncheck(is_prime)",
    "test_code": null,
    "tests": [
        {
            "passed": true,
            "exec_info": {
                "status": "Success",
                "message": "",
                "compile_result": null,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.12065744400024414,
                    "return_code": 0,
                    "stdout": "",
                    "stderr": ""
                },
                "executor_pod_name": null,
                "files": {}
            },
            "test_info": null
        }
    ],
    "extracted_type": null,
    "extra": null
}
```

</details>

</details>


### 🛠️ Dev & Test

Refer to installation section for the setup of development environment.

Run all unit tests:

```bash
make test
```

Run the test of `common_evaluate_batch` API:
```bash
export URL="http://0.0.0.0:8080"
PYTHONPATH=$(pwd):$PYTHONPATH pytest -s -vv -k test_sandbox_common_evaluate
```

Run a specific unit test (allows you to see stdout):

```bash
make test-case CASE=test_java_assert
```

Run a specific unit test with pdb:

```bash
make test-case-pdb CASE=test_java_assert
```

Format the code:

```bash
make format
```

### 🤖 Model Context Protocol
Install [fastmcp](https://gofastmcp.com/getting-started/installation), then start the mcp server, which connects to the sandbox `run_code` API:

```bash
cd mcp_server

export SANDBOX_URL="http://0.0.0.0:8080/run_code"
fastmcp run server.py --transport="http" --host 0.0.0.0 --port="8765"
```

Then, add the following to your MCP client:

```json
{
    "mcpServers": {
        "sandbox": {
            "httpUrl": "http://124.16.138.150:8765/mcp"
        }
    }
}
```

## 📊 Unified Evaluation

An evaluation framework that uses the sandbox within this codebase for assessment.

### ⚙️ Sandbox Eval Usage Guide

Environment Setup.

```bash
conda create --name sandbox_eval python=3.10 -y
conda activate sandbox_eval

pip install -r requirements.txt
```

Quick Start.

First, create a JSON parameter file in the config directory, which includes benchmark parameter settings and model inference parameter settings.

Then run one of the following commands for evaluation.
```bash
cd eval

# Option 1: Ray-based multi-GPU inference
python3 sandbox.py --dataset_config <path/to/config.json>

# Option 2: vLLM server-based multi-GPU inference
python3 sandbox.py --dataset_config <path/to/config.json> --use_vllm_server

# Option 3: OpenAI-compatible API inference
python3 sandbox.py --dataset_config <path/to/config.json> \
    --api_url <openai_compatible_api_url> \
    --api_key <your_api_key> \
    --model_name <your_model_name> \
    --rpm <requests_per_minute>
```
- Option 1 uses Ray-based multi-GPU inference.
- Option 2 first deploys the model across multi-GPU with multiple vLLM servers, then performs concurrent inference via multiple API endpoints.
- Option 3 sends requests to an external OpenAI-compatible API endpoint (`--rpm` is optional; use `0` to disable rate limiting).

<details>
<summary><strong>Notes</strong></summary>

- If you are running evaluation on NPUs, add the `--npu` flag to the command (applies to all options above).
- If you only want to generate samples without running evaluation, add the `--sample_only` flag.
- If you want to reuse an existing sample file, add `--sample_file <path/to/sample_file.jsonl>`.
- Important config fields (mainly under `infer_parameters`):
  - `model_path`: model path or model ID used for inference/deployment.
  - `output_dir`: output directory for sampled results and evaluation outputs.
  - `endpoint`: ScaleBox evaluation endpoint, typically `http://<ip:port>/common_evaluate_batch`.
  - `prompt_type`: prompt template/token format by model family. Supported values: `llama-3-instruct`, `deepseek`, `chatml`, `chatml_qwen3`. To add a new type, extend `TEMPLATES` in `eval/utils/template.py`.
  - `max_completion_tokens`: maximum generated tokens per sample.
  - `n_sample`: number of sampled completions per prompt.
  - `num_gpus_total`: total number of GPUs/NPUs used for inference.
  - `num_gpus_per_model`: number of GPUs/NPUs allocated to one model instance.
  - `reasoning_model`: set to `true` for reasoning models.

</details>

### 📊 Evaluation results

| Model | HumanEval | MBPP | HumanEval+ | MBPP+ | LiveCodeBench | AetherCode |
|-------|-----------|------|------------|-------|---------------|------------|
| Llama-3-8B-Instruct | 60.98 | 62.76 | 57.93 | 54.76 | 10.48 | 0.20 |
| Llama-3.1-8B-Instruct | 70.73 | 66.74 | 65.24 | 57.67 | 6.18 | 0.20 |
| DeepSeek-R1-Distill-Qwen-1.5B | 47.56 | 40.28 | 44.51 | 37.30 | 16.13 | 0.07 |
| Qwen3-4B | 89.63 | 82.67 | 85.37 | 73.81 | 53.92 | 8.07 |
| Qwen3-8B | 88.41 | 85.95 | 80.48 | 73.28 | 60.09 | 9.18 |

To reproduce the results in the table, reuse the config files under `eval/config/<model>` and run with `--use_vllm_server` enabled.

## 🧾 Special Judge Generation

Some programming problems require a “special judge” (custom checker) instead of exact-match outputs. This repo provides a lightweight pipeline to:

- Automatically classify which problems need a special judge, summarizing categories like multiple valid solutions or floating-point tolerance
- Generate Python judge programs tailored to each problem, and filter with gold reference solutions and empty submissions to ensure correctness

Quick start (requires an OpenAI-compatible model API and a running sandbox):

```bash
cd special_judge

# 0) Preprocess `PrimeIntellect/verifiable-coding-problems` dataset and filter out python problems with gold solutions
python3 preprocess.py

# 1) Classify problems for special-judge need
python3 special_judge/classify.py \
    --api_key $API_KEY \
    --base_url https://api.deepseek.com \
    --model deepseek-chat \
    --data_path data/PrimeIntellect-verifiable-coding-problems-python.parquet \
    --split train \
    --text_field prompt \
    --output_path data/classified_deepseek-chat.jsonl \
    --batch_size 16

# 2) Filter and summarize the classification results
python3 special_judge/filter_special_judge.py \
    --input data/classified_deepseek-chat.jsonl \
    --output-jsonl data/require_special_judge.jsonl \
    --output-ids data/require_special_judge_ids.json

# 3) Generate custom judge programs and evaluate via sandbox
#    Provide your running sandbox URL (e.g., http://0.0.0.0:8080)
python3 special_judge/generate_judge_program.py \
    --sandbox_url $SANDBOX_URL \
    --api_key $API_KEY \
    --base_url https://api.deepseek.com \
    --model deepseek-chat \
    --data_path data/require_special_judge.parquet \
    --split train \
    --text_field prompt \
    --output_path data/special_judge_deepseek-chat.jsonl \
    --run_timeout 30
```

Notes:
- The classifier and generator stream LLM outputs and include simple retry/backoff for rate limits/timeouts.
- Generated judge programs follow the stdin/stdout/answer interface required by the sandbox’s special judge mode.


## 📚 Logging
### ⚠️ Error Programs Recording
Enable by setting `SAVE_BAD_CASES` to `true` in the environment variables, and disabled by default.
```bash
export SAVE_BAD_CASES=true
make run-online
```
If the unit test running status is `SandboxError`, the result would be written to `output/{datetime}/xxx.json`.

<details>
<summary>Example</summary>
```json
{
  "id": 0,
  "accepted": false,
  "extracted_code": "__author__ = 'Admin'\n\ndef f(n):\n\treturn max(n[0], n[1])\nt = True\n(x1, y1, x2, y2, x3, y3) = map(int, input().split())\nm = [x1, y1, x2, y2, x3, y3]\nm1 = [[x1, y1, 'A'], [x2, y2, 'B'], [x3, y3, 'C']]\nm1.sort(key=f)\nmaxi = max(m1[-1][0], m1[-1][1])\nmini = min(m1[-1][0], m1[-1][1])\nmaxj = max(m1[-2][1], m1[-2][0])\nminj = min(m1[-2][1], m1[-2][0])\nmaxk = max(m1[0][1], m1[0][0])\nmink = min(m1[0][1], m1[0][0])\ns = m1[-1][2]\ns1 = m1[-2][2]\ns2 = m1[0][2]\nmatr = [[0] * maxi for i in range(maxi)]\nfor i in range(mini):\n\tfor j in range(maxi):\n\t\tmatr[i][j] = s\nif maxj == maxi and mini + minj <= maxi:\n\tfor i in range(mini, minj + mini):\n\t\tfor j in range(maxj):\n\t\t\tmatr[i][j] = s1\n\tif maxk == maxi and mini + minj + mink == maxi:\n\t\tfor i in range(minj + mini, mink + minj + mini):\n\t\t\tfor j in range(maxk):\n\t\t\t\tmatr[i][j] = s2\n\telse:\n\t\tt = False\nelif maxj == maxi - mini:\n\tfor i in range(mini, mini + maxj):\n\t\tfor j in range(minj):\n\t\t\tmatr[i][j] = s1\n\tif maxk == maxj and mink == maxi - minj:\n\t\tfor i in range(mini, mini + maxk):\n\t\t\tfor j in range(minj, minj + mink):\n\t\t\t\tmatr[i][j] = s2\n\telse:\n\t\tt = False\nelif minj == maxi - mini:\n\tfor i in range(mini, mini + minj):\n\t\tfor j in range(maxj):\n\t\t\tmatr[i][j] = s1\n\tif mink == minj and maxk == maxi - maxj:\n\t\tfor i in range(mini, mini + mink):\n\t\t\tfor j in range(maxj, maxj + maxk):\n\t\t\t\tmatr[i][j] = s2\n\telif maxk == minj and mink == maxi - maxj:\n\t\tfor i in range(mini, mini + maxk):\n\t\t\tfor j in range(maxj, maxj + mink):\n\t\t\t\tmatr[i][j] = s2\n\telse:\n\t\tt = False\nelse:\n\tt = False\nif t == True:\n\tprint(maxi)\n\tfor i in range(maxi):\n\t\tprint(*matr[i], sep='')\nelse:\n\tprint(-1)",
  "full_code": null,
  "test_code": null,
  "tests": [
    {
      "passed": false,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.8049399852752686,
          "return_code": 0,
          "stdout": "5\nCCCCC\nCCCCC\nBBBBB\nBBBBB\nAAAAA\n",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": {
        "input": {
          "stdin": "5 1 2 5 5 2\n"
        },
        "output": {
          "stdout": "5\nAAAAA\nBBBBB\nBBBBB\nCCCCC\nCCCCC\n"
        }
      }
    },
    {
      "passed": false,
      "exec_info": {
        "status": "SandboxError",
        "message": "Total Timeout",
        "compile_result": null,
        "run_result": null,
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": null
    }
  ],
  "extracted_type": null,
  "extra": null
}
```
</details>

### 🔗  Nignx Connection Logging
Running the command below to test the availability of the upstream servers and count the connections to each server.
```bash
bash deploy/test_available_server.sh
```

The output will be like this:
```bash
Active connections per upstream server:
=========== Active connections ============
Address [IP1]:[PORT1]: 1 connections
Address [IP2]:[PORT2]: 4 connections
Address [IP3]:[PORT3]: 3 connections
Address [IP4]:[PORT4]: 2 connections
===========================================
========= Active server addresses =========
Address [IP1]:[PORT1] is working
Address [IP2]:[PORT2] is working
Address [IP3]:[PORT3] is working
Address [IP4]:[PORT4] is working
===========================================
```

## 📝 Citation
```bibtex
@software{scalebox_2025,
  title = {scalebox},
  url = {https://github.com/icip-cas/scalebox},
  year = {2025}
}
```

## 🙏 Acknowledgement

This project is modified from [SandboxFusion](https://github.com/bytedance/SandboxFusion), an open-source secure sandbox for running and judging code generated by LLMs. We extend our gratitude to the original authors and contributors of SandboxFusion for their excellent work in creating a robust foundation for code execution and evaluation.

The original SandboxFusion project is licensed under the Apache License 2.0 and is maintained by ByteDance. For more information about the original project, please visit their [GitHub repository](https://github.com/bytedance/SandboxFusion).

## 📄 License

```
Copyright 2025 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences.
Copyright 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
