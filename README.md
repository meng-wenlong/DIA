# DIA

## Setup

First, install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
or install through [docker](https://hub.docker.com/r/ollama/ollama).

```bash
conda create -n ri python=3.12
conda activate ri
cd DIA
bash install.sh
```

## Quick Start

Run experiments with 10 examples:

- DIA-I
  ```bash
  ollama pull llama3:latest
  bash scripts/run_attack_dia-1_llama3:latest_advbench_test.sh
  ```
- DIA-II
  ```
  ollama pull llama3:latest
  bash scripts/run_attack_dia-2_llama3:latest_advbench_test.sh
  ```

If you want reproduce the results in Table 2 and Figure 5:

- DIA-I
  ```bash
  bash scripts/run_attack_dia-1_llama3:latest_advbench.sh
  ```
- DIA-II
  ```bash
  bash scripts/run_attack_dia-2_llama3:latest_advbench.sh
  ```

It will take 4~5 hours.