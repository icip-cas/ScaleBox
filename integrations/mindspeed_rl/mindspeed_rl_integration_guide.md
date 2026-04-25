# MindSpeed-RL Integration Guide

## Repository and Version

We provide the MindSpeed-RL related version in this repository:
https://github.com/jszheng21/MindSpeed-RL

All operations in this guide should be executed under the MindSpeed-RL repository root.

```bash
git clone https://github.com/jszheng21/MindSpeed-RL.git
cd MindSpeed-RL
```

## Install

Follow the official [installation guide](./docs/install_guide.md) to install MindSpeed-RL. The specific MindSpeed-RL version and related dependency versions are as follows:

- `MindSpeed-RL`@`master-c57ff51`

- `MindSpeed`@`master-c99f34c0`

- `Megatron-LM`@`main-core_v0.12.1`

- `MindSpeed-LLM`@`2.1.0-0fd71133`

After installation, please first run the official math RL demo (`grpo_qwen25_7b_A3`) to ensure the framework is running correctly.

## Code RL demo

### Summary

- Model: DeepSeek-R1-Distill-Qwen-1.5B
- Dataset: verifiable-coding-problems-python-only
- Algorithm: GRPO

### Prepare Data

1. Prepare Raw Data

To obtain the `verifiable-coding-problems-python-only` data, refer to the data processing script `examples/data/build_verifiable-coding-problems-python-only.py`, which reads data from `PrimeIntellect/verifiable-coding-problems`, filters for valid Python data, and simultaneously converts the test case format to be compatible with icip-sandbox.

2. Tokenize

To obtain data that can be directly read by MindSpeed-RL, the raw data needs to be tokenized. Specifically, refer to the configuration file `configs/datasets/verifiable-coding-problems-python-only.yaml`, update its `input`, `tokenizer_name_or_path`, and `output_prefix` fields, then execute the following command to tokenize the raw data from `input` and output it to `output_prefix`.

```bash
bash examples/data/preprocess_data.sh verifiable-coding-problems-python-only
```

### Prepare Model

The model parameter format read by MindSpeed-RL is Megatron-mcore. Refer to the weight conversion script in the [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM) repository (MindSpeed-LLM/examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh) to convert the model weight format from HF to mcore.

### Run Script

After completing the above preparations, configure the following properties in the `configs/grpo_qwen25_1.5b_8k_code.yaml` file:

- `tokenizer_name_or_path`: Path to the original HF format model

- `data_path`: The output_prefix of the tokenized dataset

- `load`: Path to the mcore format model

- `save`: Path where the model will be saved

- `sandbox_fusion_url`: The deployment URL for icip-sandbox

Then, run

```bash
bash examples/grpo/grpo_trainer_qwen25_1.5b_8k_code.sh
```