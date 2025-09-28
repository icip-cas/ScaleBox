import requests
import json
import os 

# obtain local address
local_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()

URL = f"http://{local_ip}:8080"
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
    # print(json.dumps(result, indent=2))
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

if __name__ == "__main__":
    test_python_stdio_batch_evaluate()