# ScaleBox Eval Configuration Reference

This document only explains the configuration structure and field meanings for `eval/config/*.yaml`.

## 1. Configuration Structure

Configuration files must use nested YAML. The top level must contain these 6 sections:

```yaml
run:
benchmark:
model:
sampling:
sandbox:
output:
```

## 2. Key Rules

- If the same parameter is set in both CLI and YAML, the CLI value takes priority.
- `run.sample_only` and `run.eval_only` cannot both be `true`.
- When not using `eval_only`, exactly one of `run.use_server` and `run.use_ray` must be `true`.
- If `run.eval_only: true`, then `run.eval_path` is required.
- For `benchmark.data_path`:
  - `livecodebench` and `aethercode` require a directory path.
  - Other benchmarks require a `.jsonl` file path.
- `livecodebench` requires `benchmark.version`.
- `aethercode` requires both `benchmark.version` and `benchmark.special_judge_file`.

## 3. Auto-Download Rules

When `benchmark.data_path: null`:

- Auto-download is supported for: `mbpp`, `mbppplus`, `humaneval`, `humanevalplus`, `livecodebench`, `aethercode`
- Auto-download is not supported for: `multipl_e`

The workflow for `multipl_e` is:

1. Run `eval/data/download_multiple.py` to download all data.
2. Set the specific `.jsonl` file path in `benchmark.data_path`.

## 4. Field Reference

### `run`

- `sample_only`: Run sampling only, without evaluation.
- `eval_only`: Run evaluation only on existing sampling results.
- `eval_path`: Path to the input result file used in `eval_only` mode.
- `resume_sample`: Continue sampling from an existing `samples.jsonl` file.
- `resume_sample_path`: Path to the `samples.jsonl` file used for resume.
- `use_server`: Use vLLM server mode for sampling.
- `use_ray`: Use ray-based parallel sampling.

### `benchmark`

- `name`: Benchmark name. Supported values: `mbpp`, `mbppplus`, `humaneval`, `humanevalplus`, `livecodebench`, `aethercode`, `multipl_e`.
- `data_path`: Path to the benchmark data. If it is `null`, the program will either auto-download the data or raise an error, depending on the benchmark.
- `version`: Version used by `livecodebench` or `aethercode`.
- `begin_date`: Used only by `livecodebench`. Start date in `YYYY-MM-DD` format.
- `end_date`: Used only by `livecodebench`. End date in `YYYY-MM-DD` format.
- `special_judge_file`: Used only by `aethercode`. Required.
- `language`: Reserved field. Usually does not need to be set manually.

### `model`

- `model_path`: Model path or model name.
- `model_name`: Model name exposed by the sampling service.
- `prompt_type`: Prompt template type.
- `thinking`: Whether to enable thinking mode.
- `num_gpus_total`: Total number of GPUs/NPUs.
- `num_gpus_per_model`: Number of GPUs/NPUs used by each model instance.
- `npu`: Whether to use NPU.
- `mem_fraction`: Memory usage fraction.
- `batch_size`: Sampling batch size.
- `max_model_len`: Maximum vLLM context length. `null` means using the default value.

### `model.vllm_server`

- `base_port`: Starting port.
- `host`: Listening address.
- `dtype`: Inference precision, such as `auto`, `float16`, or `bfloat16`.
- `wait_timeout`: Timeout in seconds for waiting for the service to start.

### `sampling`

- `temperature`: Sampling temperature.
- `top_p`: Top-p.
- `top_k`: Top-k.
- `min_p`: Min-p.
- `max_completion_tokens`: Maximum number of tokens generated per completion.
- `n_sample`: Number of generations per case.
- `stop_token`: Stop token. `null` means using the default setting.

### `sandbox`

- `endpoint`: ScaleBox evaluation service endpoint.
- `run_timeout`: Execution timeout per case, in seconds.
- `compile_timeout`: Compilation timeout, in seconds.
- `total_timeout`: Total sandbox timeout, in seconds.
- `run_all_cases`:
  - `true`: Save the test pass rate.
  - `false`: Save a `0/1` score.
- `save_full_scalebox_result`:
  - `true`: Save the full ScaleBox response body in `results.jsonl`.
  - `false`: Save only the summarized score in `scalebox`.
- `extra`: Extra parameters passed through to the sandbox.

### `output`

- `output_dir`: Output directory.

## 5. Output Files

The program creates a result directory under `output.output_dir`. Common output files are:

- `samples.jsonl`: Sampling results.
- `results.jsonl`: Evaluation results.
- `accuracy.json`: Summary metrics.