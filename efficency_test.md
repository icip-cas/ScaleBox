## Experimental Setup

### Dataset
- **Source**: PrimeIntellect/verifiable-coding-problems
- **Sample Size**: 2,048 / 4.096 / 8,192 Python problem instances with reference code

### Hardware Environment
- **CPU**: Intel Xeon 6700P Series
- **Cores**: 64 Cores / 128 Threads
- **Memory**: 256 GB

- **CPU**: HiSilicon Kunpeng 920 7285Z (ARM64)
- **Cores**: 320 Cores / 320 Threads
- **Memory**: 2.0 TB

### Sandbox Configuration
- **Load Balancer**: NGINX
- **Unit Test Parallelism**: `max_runner_concurrency=32, cases_per_subworker=6`

### Client Configuration
- **Request Concurrency**: `SCALEBOX: nodes * workers` (1/2/3 nodes: 32/64/96); `SandboxFusion: 128`; `verl <sub>Prime</sub>: 128`

## Performance Results

SCALEBOX consistently outperforms both baselines. On x86 (single node), SCALEBOX achieves **1.59×-2.88×** throughput versus verl <sub>Prime</sub> and **1.60×-2.63×** versus SandboxFusion; scaling from 1 to 2/3 nodes further improves throughput by **11.9%-27.5%** and **22.4%-58.0%**. On ARM (single node), SCALEBOX reaches **1.47×-3.32×** throughput over verl <sub>Prime</sub> and **2.25×-2.89×** over SandboxFusion.

### x86 CPU

### 2048 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 271.61 | 7.54 (1.00×) | 128 | - | 64 |
| SandboxFusion | 1 | 150.99 | 13.56 (1.80×) | 128 | 128 | 64 |
| SCALEBOX | 1 | 94.45 | 21.68 (2.88×) | 32 | 32 | 64 |
| SCALEBOX | 2 | 84.41 | 24.26 (3.22×) | 64 | 64 | 64 |
| SCALEBOX | 3 | 77.16 | 26.54 (3.52×) | 96 | 96 | 64 |

### 4096 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 293.85 | 13.94 (1.00×) | 128 | - | 64 |
| SandboxFusion | 1 | 278.24 | 14.72 (1.06×) | 128 | 128 | 64 |
| SCALEBOX | 1 | 132.82 | 30.84 (2.21×) | 32 | 32 | 64 |
| SCALEBOX | 2 | 107.70 | 38.03 (2.73×) | 64 | 64 | 64 |
| SCALEBOX | 3 | 92.12 | 44.46 (3.19×) | 96 | 96 | 64 |

### 8192 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 331.25 | 24.73 (1.00×) | 128 | - | 64 |
| SandboxFusion | 1 | 548.93 | 14.92 (0.60×) | 128 | 128 | 64 |
| SCALEBOX | 1 | 208.38 | 39.31 (1.59×) | 32 | 32 | 64 |
| SCALEBOX | 2 | 163.40 | 50.13 (2.03×) | 64 | 64 | 64 |
| SCALEBOX | 3 | 131.92 | 62.10 (2.51×) | 96 | 96 | 64 |

### ARM CPU

### 2048 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 470.95 | 4.35 (1.00×) | 128 | - | 320 |
| SandboxFusion | 1 | 319.82 | 6.40 (1.47×) | 128 | 128 | 128 |
| SCALEBOX | 1 | 141.94 | 14.43 (3.32×) | 32 | 32 | 128 |

### 4096 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 495.34 | 8.27 (1.00×) | 128 | - | 320 |
| SandboxFusion | 1 | 634.70 | 6.45 (0.78×) | 128 | 128 | 128 |
| SCALEBOX | 1 | 220.13 | 18.61 (2.25×) | 32 | 32 | 128 |

### 8192 Cases

| Method | Nodes | Time (s) | Throughput (tasks/s) | Concurrency | Workers | CPU |
|--------|-------|----------|----------------------|-------------|---------|-----|
| verl <sub>Prime</sub> | 1 | 594.98 | 13.77 (1.00×) | 128 | - | 320 |
| SandboxFusion | 1 | 1026.52 | 7.98 (0.58×) | 128 | 128 | 128 |
| SCALEBOX | 1 | 404.92 | 20.23 (1.47×) | 32 | 32 | 128 |
