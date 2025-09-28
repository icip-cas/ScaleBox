We evaluated the performance of our sandbox API against prime_code local evaluation method. Our results demonstrate that it's more efficent and scalable, which is highly suitable for CodeRL training workloads.

## Experimental Setup

### Dataset
- **Source**: PrimeIntellect/verifiable-coding-problems
- **Sample Size**: 2,048 / 4.096 / 8,192 Python problem instances with reference code

### Hardware Environment
- **CPU**: Intel Xeon Platinum 8378A @ 3.00GHz
- **Cores**: 128 physical cores
- **Memory**: 1 TB

### Sandbox Configuration
- **Load Balancer**: NGINX
- **Unit Test Parallelism**: `max_runner_concurrency=16`

### Client Configuration
- **Request Concurrency**: 128
- **Batch Processing**: Enabled via `common_evaluate_batch`

## Performance Results

Our results demonstrate that the sandbox API achieves a **14.57% performance improvement** over the baseline and **36.31% performance improvement** over the `run_code` API. Compared with `run_code` API which send each unit test seperately, our `common_evaluate_batch` API pack the whole test cases, and is more efficent. Our sandbox also scales linearly across multiple nodes. With two sandbox nodes, we achieved a **38.29% reduction** in evaluation time. 

### 2048 Cases

| Evaluation Method | Time (s) | Client Concurrency | Workers | Total CPU Cores |
|-------------------|----------|-------------------|---------|-----------------|
| **Baseline** (No Sandbox) | 164.89 | 128 | - | 128 |
| **Sandbox/run_code** (1 node) | 221.18 | 128 | 32 | 128 |
| **Sandbox/common_evaluate_batch** (1 node) | 140.86 | 128 | 32 | 128 |
| **Sandbox/common_evaluate_batch** (2 nodes) | 86.93 | 128 | 64 | 256 |
| **Sandbox/common_evaluate_batch** (3 nodes) | 66.84 | 128 | 96 | 384 |

### 4096 Cases

| Evaluation Method | Time (s) | Client Concurrency | Workers | Total CPU Cores |
|-------------------|----------|-------------------|---------|-----------------|
| **Baseline** (No Sandbox) | 297.03 | 128 | - | 128 |
| **Sandbox/run_code** (1 node) | 516.16 | 128 | 128 | 128 |
| **Sandbox/common_evaluate_batch** (1 node) | 181.74 | 128 | 32 | 128 |
| **Sandbox/common_evaluate_batch** (2 nodes) | 155.21 | 128 | 64 | 256 |
| **Sandbox/common_evaluate_batch** (3 nodes) | 118.80 | 128 | 96 | 384 |

### 8192 Cases

| Evaluation Method | Time (s) | Client Concurrency | Workers | Total CPU Cores |
|-------------------|----------|-------------------|---------|-----------------|
| **Baseline** (No Sandbox) | 556.48 | 128 | - | 128 |
| **Sandbox/run_code** (1 node) | 989.05 | 128 | 128 | 128 |
| **Sandbox/common_evaluate_batch** (1 node) | 324.01 | 128 | 32 | 128 |
| **Sandbox/common_evaluate_batch** (2 nodes) | 258.85 | 128 | 64 | 256 |
| **Sandbox/common_evaluate_batch** (3 nodes) | 183.8 | 128 | 96 | 384 |