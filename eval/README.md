# Eval Usage Guide

This directory contains the unified evaluation entry (`sandbox.py`) and config examples for different inference/evaluation modes.

## Quick Start

Run with any config file under `eval/config`:

```bash
cd ScaleBox/eval
python sandbox.py --dataset_config config/livecodebench-qwen3-4b.json
```

## New Config Examples

### 1) Single API mode (`APIRunner`)

Config: `config/api.json`

Use when your model is served by an OpenAI-compatible endpoint:

```bash
python sandbox.py --dataset_config config/api.json
```

Key fields:
- `api_url`: e.g. `http://127.0.0.1:8000/v1/completions`
- `api_key`
- `model_name`
- `rpm`: request-per-minute limiter

### 2) Multi-instance vLLM Server mode (`VLLMServerManager` + `MultiAPIRunner`)

Config: `config/vllm-server.json`

```bash
python sandbox.py --dataset_config config/vllm-server.json
```

Key fields:
- `use_vllm_server`: `true`
- `vllm_server_base_port`
- `vllm_server_host`
- `vllm_server_dtype`
- `vllm_server_wait_timeout`
- `mem_fraction`

### 3) NPU vLLM Server mode

Config: `config/vllm-server-npu.json`

```bash
python sandbox.py --dataset_config config/vllm-server-npu.json
```

Key field:
- `npu`: `true`

### 4) Sampling workflow examples (`sample_only` / `resume_from` / `sample_file`)

Config: `config/sample_only-resume_from-sample_file.json`

This file contains three `infer_parameters` examples:
- step1: `sample_only=true` (generate samples only)
- step2: `resume_from=...` (resume unfinished sampling)
- step3: `sample_file=...` (evaluate from existing samples)

```bash
python sandbox.py --dataset_config config/sample_only-resume_from-sample_file.json
```

### 5) AetherCode special-judge dataset

Config: `config/aethercode-qwen3-4b.json`

```bash
python sandbox.py --dataset_config config/aethercode-qwen3-4b.json
```

This mode uses `datasetType: AetherCodeDataset` and C++ special judge evaluation in sandbox.

## Notes

- Replace placeholder values in config before running (for example `api_url`, `api_key`, model path, endpoints).
- The sandbox endpoint (`endpoint`) should point to `/common_evaluate_batch`.
- For large jobs, prefer sample-first workflow to support resume and offline evaluation.
