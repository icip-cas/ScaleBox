import requests
import json
import os

URL = os.getenv('URL', 'http://localhost:8000')

def test_special_judge():
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

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_special_judge_2():
    payload = {
        "completion": """```python\nc = int(input())\nprint(c-1, c)\n```""",
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

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == False

def test_cpp_assert():
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

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_python_assert():
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

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

# "python", "java", "cpp", "csharp", "go", "rust", "lua", "julia", "nodejs", "typescript", "php", "ruby", "scala", "swift", "perl", "racket", "D_ut"

def test_stdio_2():
    payload = {
        'completion': "```python\n\nn = int(input())\na = list(map(int, input().split()))\n\n# Initialize DP: list of tuples (value, cost)\ndp = [(a[0], 0)]\n\nfor i in range(1, n):\n    new_dp = []\n    current = a[i]\n    for (prev_val, prev_cost) in dp:\n        if current > prev_val + 1:\n            new_val = current\n            new_cost = prev_cost\n        else:\n            new_val = prev_val + 1\n            new_cost = prev_cost + (new_val - current)\n        new_dp.append((new_val, new_cost))\n    \n    # Sort the new_dp by value\n    new_dp.sort()\n    \n    # Process to keep only the minimal cost entries\n    min_cost = float('inf')\n    processed = []\n    for val, cost in new_dp:\n        if cost < min_cost:\n            processed.append((val, cost))\n            min_cost = cost\n    dp = processed\n\n# The minimal cost is the minimal in the last dp list\nif dp:\n    print(min(dp, key=lambda x: x[1])[1])\nelse:\n    print(0)\n\n```", 
        'config': {
            'language': 'python', 
            'compile_timeout': 20, 'run_timeout': 30, 
            'provided_data': {
                'test_cases': {
                    'fn_name': None, 
                    'input': ['7\n2 1 5 11 5 9 11', '5\n5 4 3 2 1', '2\n1 1000', '2\n1000 1', '5\n100 80 60 70 90', '10\n10 16 17 11 1213 1216 1216 1209 3061 3062', '20\n103 103 110 105 107 119 113 121 116 132 128 124 128 125 138 137 140 136 154 158', '1\n1', '5\n1 1 1 2 3', '1\n1000', '50\n499 780 837 984 481 526 944 482 862 136 265 605 5 631 974 967 574 293 969 467 573 845 102 224 17 873 648 120 694 996 244 313 404 129 899 583 541 314 525 496 443 857 297 78 575 2 430 137 387 319', '75\n392 593 98 533 515 448 220 310 386 79 539 294 208 828 75 534 875 493 94 205 656 105 546 493 60 188 222 108 788 504 809 621 934 455 307 212 630 298 938 62 850 421 839 134 950 256 934 817 209 559 866 67 990 835 534 672 468 768 757 516 959 893 275 315 692 927 321 554 801 805 885 12 67 245 495', '10\n26 723 970 13 422 968 875 329 234 983', '20\n245 891 363 6 193 704 420 447 237 947 664 894 512 194 513 616 671 623 686 378', '5\n850 840 521 42 169'], 
                    'output': ['9', '12', '0', '1000', '54', '16', '43', '0', '3', '0', '12423', '17691', '2546', '3208', '1485'], 
                    'type': 'stdin_stdout'
                    }
                }, 
            'extra': {'run_all_cases': True, 'total_timeout': 30}
        }
    }
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] in [True, False]



def test_python_function_call_str_input():
    payload = {"completion": """
```
def double_check(strng):
    if len(strng) < 2:
        return False
    for i in range(len(strng) - 1):
        if strng[i].lower() == strng[i+1].lower():
            return True
    return False
```
""", "config": {"language": "python", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_java_function_call_str_input():
    payload = {"completion": """
```
public static boolean double_check(String strng) {
    if (strng.length() < 2) {
        return false;
    }
    for (int i = 0; i < strng.length() - 1; i++) {
        if (Character.toLowerCase(strng.charAt(i)) == Character.toLowerCase(strng.charAt(i + 1))) {
            return true;
        }
    }
    return false;
}
```
""", "config": {"language": "java", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_cpp_function_call_str_input():
    payload = {"completion": """```
#include <iostream>
#include <string>
using namespace std;
bool double_check(string strng) {
    if (strng.length() < 2) {
        return false;
    }
    for (int i = 0; i < strng.length() - 1; i++) {
        if (tolower(strng[i]) == tolower(strng[i + 1])) {
            return true;
        }
    }
    return false;
}
```
""", "config": {"language": "cpp", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_csharp_function_call_str_input():
    payload = {"completion": """```
public static bool double_check(string strng) {
    if (strng.Length < 2) {
        return false;
    }
    for (int i = 0; i < strng.Length - 1; i++) {
        if (char.ToLower(strng[i]) == char.ToLower(strng[i + 1])) {
            return true;
        }
    }
    return false;
}
```
""", "config": {"language": "csharp", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()    
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_go_function_call_str_input():
    payload = {"completion": """```
import (
    "unicode"
)
func double_check(strng string) bool {
    if len(strng) < 2 {
        return false
    }
    for i := 0; i < len(strng)-1; i++ {
        if unicode.ToLower(rune(strng[i])) == unicode.ToLower(rune(strng[i+1])) {
            return true
        }
    }
    return false
}
```
""", "config": {"language": "go", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_rust_function_call_str_input():
    payload = {"completion": """```
fn double_check(strng: &str) -> bool {
    if strng.len() < 2 {
        return false;
    }
    let chars: Vec<char> = strng.chars().collect();
    for i in 0..chars.len() - 1 {
        if chars[i].to_lowercase().to_string() == chars[i + 1].to_lowercase().to_string() {
            return true;
        }
    }
    return false;
}
```
""", "config": {"language": "rust", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_php_function_call_str_input():
    payload = {"completion": """```
function double_check($strng) {
    if (strlen($strng) < 2) {
        return false;
    }
    for ($i = 0; $i < strlen($strng) - 1; $i++) {
        if (strtolower($strng[$i]) == strtolower($strng[$i + 1])) {
            return true;
        }
    }
    return false;
}
```
""", "config": {"language": "php", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_ruby_function_call_str_input():
    payload = {"completion": """```
def double_check(strng)
    if strng.length < 2
        return false
    end
    (0..strng.length - 2).each do |i|
        if strng[i].downcase == strng[i + 1].downcase
            return true
        end
    end
    return false
end
```
""", "config": {"language": "ruby", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_perl_function_call_str_input():
    payload = {"completion": """```
sub double_check {
    my ($strng) = @_;
    if (length($strng) < 2) {
        return 0;
    }
    for (my $i = 0; $i < length($strng) - 1; $i++) {
        if (lc(substr($strng, $i, 1)) eq lc(substr($strng, $i + 1, 1))) {
            return 1;
        }
    }
    return 0;
}
```
""", "config": {"language": "perl", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[0], [1], [1], [1], [1], [0], [0], [0]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_swift_function_call_str_input():
    payload = {"completion": """```
func double_check(_ strng: String) -> Bool {
    if strng.count < 2 {
        return false
    }
    for i in 0..<strng.count - 1 {
        let firstChar = strng[strng.index(strng.startIndex, offsetBy: i)]
        let secondChar = strng[strng.index(strng.startIndex, offsetBy: i + 1)]
        if firstChar.lowercased() == secondChar.lowercased() {
            return true
        }
    }
    return false
}
```
""", "config": {"language": "swift", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[False], [True], [True], [True], [True], [False], [False], [False]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_D_ut_function_call_str_input():
    payload = {"completion": """```
import std.uni;

int double_check(string strng) {
    if (strng.length < 2) {
        return 0;
    }
    foreach (size_t i; 0 .. strng.length - 1) {
        if (toLower(strng[i]) == toLower(strng[i + 1])) {
            return 1;
        }
    }
    return 0;
}
```
""", "config": {"language": "D_ut", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[0], [1], [1], [1], [1], [0], [0], [0]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_live_code_bench_data():
    payload = {
        "completion": """class Solution:
        def matrixSum(self, nums: List[List[int]], dum_param: int = 1, dum_str: str = "") -> int:
            # 1) Sort each row in descending order
            for row in nums:
                row.sort(reverse=True)
            
            # 2) For each position j = 0..m-1, take the max across all rows at that position
            #    and add to the score.
            #    zip(*nums) groups the j-th elements of every row into a tuple.
            return sum(max(col) for col in zip(*nums))""", 
        "config": {
            "language": "python", 
            "compile_timeout": 60, 
            "run_timeout": 60, 
            "provided_data": {
                "test_cases": {
                    "fn_name": "matrixSum", 
                    "input": ["[[7, 2, 1], [6, 4, 2], [6, 5, 3], [3, 2, 1]]\n1\n\"abc\"", "[[1]]\n2\n\"'''\""], 
                    "output": ["15", "1"], 
                    "type": "function_call",
                    "json_input": True,
                }
            }, 
            "extra": {
                "run_all_cases": True, 
                "total_timeout": 60
            }
        }
    }

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_total_timeout():
    false = False
    true = True
    payload = {"completion": "```\nimport time\ntime.sleep(1)\ndef double_check(strng):\n    if len(strng) < 2:\n        return False\n    for i in range(len(strng) - 1):\n        if strng[i].lower() == strng[i+1].lower():\n            return True\n    return False\n```", "config": {"language": "python", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]] * 100, "output": [[false], [true], [true], [true], [true], [false], [false], [false]] * 100, "type": "function_call"}}, "extra": {"run_all_cases": true, "total_timeout": 1}}}


    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == False
    assert result['tests'][-1]['passed'] == False
    assert result['tests'][-1]['exec_info']['status'] == 'SandboxError'
    assert result['tests'][-1]['exec_info']['message'] == 'Total Timeout'

def function_call_evaluate(completion, language):
    payload = {
        "completion": f"""```{language}\n{completion}\n```""",
        "config": {
            "language": language,
            "compile_timeout": 10,
            "run_timeout": 10,
            # "custom_extract_logic": "string",
            "provided_data": { 
                "test_cases": [
                    {"type": "function_call", "input": [1, 2], "output": [3], "fn_name": "add"},
                    {"type": "function_call", "input": [3, 4], "output": [7], "fn_name": "add"}
                ]
            },
            "extra": {
                "run_all_cases": True
            }
        }
    }
    response = requests.post(f'{URL}/common_evaluate', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

# racket, D_ut, lua, julia, nodejs, python, cpp, go, java, typescript, csharp, rust, php, bash, ruby, perl, scala, swift, kotlin_script, R, verilog
def test_racket_function_call():
    completion = """
#lang racket
(define (add a b)
  (+ a b))
"""
    result = function_call_evaluate(completion, "racket")
    assert result['accepted'] == True

def test_D_ut_function_call():
    completion = """
int add(int a, int b) {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "D_ut")
    assert result['accepted'] == True

def test_lua_function_call():
    completion = """
function add(a, b)
    return a + b
end
"""
    result = function_call_evaluate(completion, "lua")
    assert result['accepted'] == True

def test_julia_function_call():
    completion = """
function add(a, b)
    return a + b
end
"""
    result = function_call_evaluate(completion, "julia")
    assert result['accepted'] == True

def test_nodejs_function_call():
    completion = """
function add(a, b) {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "nodejs")
    assert result['accepted'] == True

def test_python_function_call():
    completion = """
def add(a, b):
    return a + b
"""
    result = function_call_evaluate(completion, "python")
    assert result['accepted'] == True

def test_cpp_function_call():
    completion = """
int add(int a, int b) {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "cpp")
    assert result['accepted'] == True

def test_go_function_call():
    completion = """
func add(a int, b int) int {
    return a + b
}
"""
    result = function_call_evaluate(completion, "go")
    assert result['accepted'] == True

def test_java_function_call():
    completion = """
public static int add(int a, int b) {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "java")
    assert result['accepted'] == True

def test_typescript_function_call():
    completion = """
function add(a: number, b: number): number {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "typescript")
    assert result['accepted'] == True

def test_csharp_function_call():
    completion = """
public static int add(int a, int b) {
    return a + b;
}
"""
    result = function_call_evaluate(completion, "csharp")
    assert result['accepted'] == True

def test_rust_function_call():
    completion = """
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
    result = function_call_evaluate(completion, "rust")
    assert result['accepted'] == True

def test_php_function_call():
    completion = """
function add($a, $b) {
    return $a + $b;
}
"""
    result = function_call_evaluate(completion, "php")
    assert result['accepted'] == True

def test_bash_function_call():
    completion = """
add() {
    echo $(($1 + $2))
}
"""
    result = function_call_evaluate(completion, "bash")
    assert result['accepted'] == True

def test_ruby_function_call():
    completion = """
def add(a, b)
  a + b
end
"""
    result = function_call_evaluate(completion, "ruby")
    assert result['accepted'] == True

def test_perl_function_call():
    completion = """
sub add {
    my ($a, $b) = @_;
    return $a + $b;
}
"""
    result = function_call_evaluate(completion, "perl")
    assert result['accepted'] == True

def test_scala_function_call():
    completion = """
def add(a: Int, b: Int): Int = {
  a + b
}
"""
    result = function_call_evaluate(completion, "scala")
    assert result['accepted'] == True

def test_swift_function_call():
    completion = """
func add(a: Int, b: Int) -> Int {
    return a + b
}
"""
    result = function_call_evaluate(completion, "swift")
    assert result['accepted'] == True

def test_kotlin_function_call():
    completion = """
fun add(a: Int, b: Int): Int {
    return a + b
}
"""
    result = function_call_evaluate(completion, "kotlin_script")
    assert result['accepted'] == True

def test_r_function_call():
    completion = """
add <- function(a, b) {
    return(a + b)
}
"""
    result = function_call_evaluate(completion, "R")
    assert result['accepted'] == True

def test_verilog_function_call():
    completion = """
function integer add;
    input integer a;
    input integer b;
    begin
        add = a + b;
    end
endfunction
"""
    result = function_call_evaluate(completion, "verilog")
    assert result['accepted'] == True

def stdio_evaluate(completion, language):
    payload = {
        "completion": f"""```{language}\n{completion}\n```""",
        "config": {
            "language": language,
            "compile_timeout": 10,
            "run_timeout": 60,
            # "custom_extract_logic": "string",
            "provided_data": { 
                "test": [
                    {
                        "input": {   
                            "stdin": "1 2"
                        },
                        "output": {  
                            "stdout": "3"
                        }
                    },
                    {
                        "input": {   
                            "stdin": "3 4"
                        },
                        "output": {  
                            "stdout": "7"
                        }
                    }
                ],
            },
            "extra": {
                "run_all_cases": True
            }
        }
    }

    # result = {"accepted": True}
    response = requests.post(f'{URL}/common_evaluate', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

# racket, D_ut, lua, julia, nodejs, python, cpp, go, java, typescript, csharp, rust, php, bash, ruby, perl, scala, swift, kotlin_script, R, verilog

def test_racket_stdio_evaluate():
    completion = """
#lang racket
(define (add a b)
  (+ a b))
(define (main)
  (define input (read-line))
  (define numbers (map string->number (string-split input)))
  (define result (apply add numbers))
  (displayln result))
(main)
"""
    result = stdio_evaluate(completion, "racket")
    assert result['accepted'] == True

def test_D_ut_stdio_evaluate():
    completion = """
import std.stdio;
int add(int a, int b) {
    return a + b;
}
void main() {
    int a, b;
    readf(" %d %d", a, b);
    writeln(add(a, b));
}
"""
    result = stdio_evaluate(completion, "D_ut")
    assert result['accepted'] == True

def test_lua_stdio_evaluate():
    completion = """
function add(a, b)
    return a + b
end
a, b = io.read("*n", "*n")
print(add(a, b))
"""
    result = stdio_evaluate(completion, "lua")
    assert result['accepted'] == True

def test_julia_stdio_evaluate():
    completion = """
function add(a, b)
    return a + b
end
a, b = parse.(Int, split(readline()))
println(add(a, b))
"""
    result = stdio_evaluate(completion, "julia")
    assert result['accepted'] == True

def test_nodejs_stdio_evaluate():
    completion = """
function add(a, b) {
    return a + b;
}
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
rl.on('line', (line) => {
    const [a, b] = line.split(' ').map(Number);
    console.log(add(a, b));
    rl.close();
});
"""
    result = stdio_evaluate(completion, "nodejs")
    assert result['accepted'] == True

def test_python_stdio_evaluate():
    completion = """
def add(a, b):
    return a + b

a, b = map(int, input().split())
print(add(a, b))
"""
    result = stdio_evaluate(completion, "python")
    assert result['accepted'] == True

def test_cpp_stdio_evaluate():
    completion = """
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int a, b;
    cin >> a >> b;
    cout << add(a, b) << endl;
    return 0;
}
"""
    result = stdio_evaluate(completion, "cpp")
    assert result['accepted'] == True

def test_go_stdio_evaluate():
    completion = """
package main

import "fmt"

func add(a, b int) int {
    return a + b
}

func main() {
    var a, b int
    fmt.Scan(&a, &b)
    fmt.Println(add(a, b))
}
"""
    result = stdio_evaluate(completion, "go")
    assert result['accepted'] == True

def test_java_stdio_evaluate():
    completion = """
import java.util.Scanner;

public class Main {
    public static int add(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int a = scanner.nextInt();
        int b = scanner.nextInt();
        System.out.println(add(a, b));
    }
}
"""
    result = stdio_evaluate(completion, "java")
    assert result['accepted'] == True

def test_typescript_stdio_evaluate():
    completion = """
const readline = require('readline');

function add(a: number, b: number): number {
    return a + b;
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

rl.on('line', (line: string) => {
    const [a, b] = line.split(' ').map(Number);
    console.log(add(a, b));
    rl.close();
});
"""
    result = stdio_evaluate(completion, "typescript")
    assert result['accepted'] == True

def test_csharp_stdio_evaluate():
    completion = """
using System;

public class APlusB{
    private static void Main(){
        string? line = Console.ReadLine();
        if (line != null) {
            string[] input = line.Split(' ');
            Console.WriteLine(int.Parse(input[0]) + int.Parse(input[1]));
        }
    }
}
"""
    result = stdio_evaluate(completion, "csharp")
    assert result['accepted'] == True

def test_rust_stdio_evaluate():
    completion = """
use std::io;

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let numbers: Vec<i32> = input
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    println!("{}", add(numbers[0], numbers[1]));
}
"""
    result = stdio_evaluate(completion, "rust")
    assert result['accepted'] == True

def test_php_stdio_evaluate():
    completion = """
<?php
function add($a, $b) {
    return $a + $b;
}

$input = fgets(STDIN);
list($a, $b) = explode(" ", $input);
echo add((int)$a, (int)$b);
?>
"""
    result = stdio_evaluate(completion, "php")
    assert result['accepted'] == True

def test_bash_stdio_evaluate():
    completion = """
#!/bin/bash

add() {
    echo $(($1 + $2))
}

read a b
add $a $b
"""
    result = stdio_evaluate(completion, "bash")
    assert result['accepted'] == True

def test_ruby_stdio_evaluate():
    completion = """
def add(a, b)
    return a + b
end

a, b = gets.split.map(&:to_i)
puts add(a, b)
"""
    result = stdio_evaluate(completion, "ruby")
    assert result['accepted'] == True

def test_perl_stdio_evaluate():
    completion = """
sub add {
    my ($a, $b) = @_;
    return $a + $b;
}

my $input = <STDIN>;
my @numbers = split / /, $input;
print add($numbers[0], $numbers[1]) . "\n";
"""
    result = stdio_evaluate(completion, "perl")
    assert result['accepted'] == True

def test_scala_stdio_evaluate():
    completion = """
object Main {
    def add(a: Int, b: Int): Int = {
    a + b
    }
    
    def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    val nums = input.split(" ").map(_.toInt)
    println(add(nums(0), nums(1)))
    }
}
"""
    result = stdio_evaluate(completion, "scala")
    assert result['accepted'] == True

def test_swift_stdio_evaluate():
    completion = """
func add(a: Int, b: Int) -> Int {
    return a + b
}

if let input = readLine() {
    let numbers = input.split(separator: " ").compactMap { Int($0) }
    print(add(a: numbers[0], b: numbers[1]))
}
"""
    result = stdio_evaluate(completion, "swift")
    assert result['accepted'] == True

def test_kotlin_stdio_evaluate():
    completion = """
fun add(a: Int, b: Int): Int {
    return a + b
}

val input = readLine()!!.trim().split(" ").map { it.toInt() }
println(add(input[0], input[1]))
"""
    result = stdio_evaluate(completion, "kotlin_script")
    assert result['accepted'] == True

def test_r_stdio_evaluate():
    completion = """
add <- function(a, b) {
    return(a + b)
}

line <- readLines("stdin", n = 1)
nums <- as.integer(strsplit(trimws(line), "\\\\s+")[[1]])
cat(add(nums[1], nums[2]))
"""
    result = stdio_evaluate(completion, "R")
    assert result['accepted'] == True

def test_verilog_stdio_evaluate():
    completion = """
module tb;
    integer a, b;
    integer res;

    function integer add;
        input integer x;
        input integer y;
        begin
            add = x + y;
        end
    endfunction

    initial begin
        if ($fscanf(32'h8000_0000, "%d %d", a, b) != 2) begin
            $display("0");
            $finish;
        end
        res = add(a, b);
        $display("%0d", res);
        $finish;
    end
endmodule
"""
    result = stdio_evaluate(completion, "verilog")
    assert result['accepted'] == True

def stdio_batch_evaluate(completion, language):
    payload = {
        "completion": f"""```{language}\n{completion}\n```""",
        "config": {
            "language": language,
            "compile_timeout": 10,
            "run_timeout": 60,
            # "custom_extract_logic": "string",
            "provided_data": { 
                "test_cases": 
                    {"type": "stdin_stdout", "input": ["1 2", "3 4"], "output": ["3", "7"], "fn_name": None},            
            },
            "extra": {
                "run_all_cases": True
            }
        }
    }

    # result = {"accepted": True}
    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def test_python_stdio_batch_evaluate():
    completion = """
def add(a, b):
    return a + b
a, b = map(int, input().split())
print(add(a, b))
"""
    result = stdio_batch_evaluate(completion, "python")
    assert result['accepted'] == True

def test_cpp_stdio_batch_evaluate():
    completion = """
#include <iostream>
using namespace std;
int add(int a, int b) {
    return a + b;
}
int main() {
    int a, b;
    cin >> a >> b;
    cout << add(a, b) << endl;
    return 0;
}
"""
    result = stdio_batch_evaluate(completion, "cpp")
    assert result['accepted'] == True

def test_1():
    payload = {
    # "dataset": "string",
    # "id": 0,
    "completion": """```python\ndef add(a, b):\n    return a + b\n```""",
    "config": {
        "language": "python",
        # "compile_timeout": 0,
        # "run_timeout": 0,
        # "custom_extract_logic": "string",
        "provided_data": { 
            "test_cases": 
                {"type": "function_call", "input": [[1, 2], [3, 4]], "output": [[3], [7]], "fn_name": "add"},            
        },
        "extra": {
            "run_all_cases": True
        }
    }
}

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_2():
    payload = {
        # "dataset": "string",
        # "id": 0,
    "completion": """```python\na, b = map(int, input().split())\nprint(a + b)\n```""",
        "config": {
            "language": "python",
            # "compile_timeout": 0,
            "run_timeout": 10,
            # "custom_extract_logic": "string",
            "provided_data": { 
                "test_cases": 
                    {"type": "stdin_stdout", "input": ["1 2", "3 4"], "output": ["3", "7"], "fn_name": None},            
            },
            "extra": {
                "run_all_cases": True
            }
        }
    }

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_3():
    false = False
    true = True
    payload = {"completion": "\ndef double_check(strng):\n    if len(strng) < 2:\n        return False\n    for i in range(len(strng) - 1):\n        if strng[i].lower() == strng[i+1].lower():\n            return True\n    return False\n", "config": {"language": "python", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "double_check", "input": [["abca"], ["aabc"], ["a 11 c d"], ["AabBcC"], ["a b  c"], ["a b c d e f g h i h k"], ["2020"], ["a!@€£#$%^&*()_-+=}]{[|':;?/>.<,~"]], "output": [[false], [true], [true], [true], [true], [false], [false], [false]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}


    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True

def test_4():
    false = False
    true = True
    payload = {"completion": "\nclass Solution:\n    def addDigits(self, num: int) -> List[List[int]]:\n        if num == 0:\n            return 0\n        remainder = num % 9\n        return 9 if remainder == 0 else remainder\n", "config": {"language": "python", "compile_timeout": 60, "run_timeout": 60, "provided_data": {"test_cases": {"fn_name": "addDigits", "input": [[38]], "output": [[2]], "type": "function_call"}}, "extra": {"run_all_cases": True, "total_timeout": 60}}}

    response = requests.post(f'{URL}/common_evaluate_batch', json=payload)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert result['accepted'] == True
