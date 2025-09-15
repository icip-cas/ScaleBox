# Code Benchmark Evaluation

```bash
# Sandbox URL
URL=http://10.0.1.3:8080/common_evaluate_batch

# Set a while loop to test if URL is reachable
while true; do
    # Check if the URL is reachable
    if curl -s ${URL} --max-time 2; then
        echo "URL is reachable"
        break
    else
        echo "URL is not reachable, retrying in 5 seconds..."
        sleep 5
    fi
done

# Infer
# qwen3 model
python3 sandbox.py --dataset_config config/livecodebench-qwen3-4b.json

# deepseek distll model
python3 sandbox.py --dataset_config config/livecodebench-qwen2.5-1.5b-distill.json
```