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

## Template Inference Attack (TIA)

Here are steps to reproduce Figure 2.

Step 1: Generate inference results

```bash
cd tia
python main.py --model_name_a qwen2:latest --model_name_b llama3:latest --model_name_other gemma2:latest
python main.py --model_name_a llama3:latest --model_name_b qwen2:latest --model_name_other gemma2:latest
python main.py --model_name_a gemma2:latest --model_name_b qwen2:latest --model_name_other llama3:latest
python main.py --model_name_a qwen2:latest --model_name_b gemma2:latest --model_name_other llama3:latest
python main.py --model_name_a gemma2:latest --model_name_b llama3:latest --model_name_other qwen2:latest
python main.py --model_name_a llama3:latest --model_name_b gemma2:latest --model_name_other qwen2:latest
```

Step 2: Compute accuracy

```bash
python compute_metrics.py
```

The results are stored in `template_infer_results.json`.