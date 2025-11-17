<p align="center">
  <img src="assets/logo.png" width="200"/>
</p>

# Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation

This repository contains the official implementation of DIA (Dialogue Injection Attack), a new jailbreak attack paradigm that leverages fabricated dialogue history to enhance the effectiveness of adversarial prompts against large language models (LLMs). Unlike prior jailbreak studies that focus primarily on single-turn interactions or assume the attacker can only modify user inputs, DIA reveals a broader and more powerful attack surface, **the ability to manipulate a model’s historical outputs**.

DIA operates entirely in a black-box setting, requiring only access to a chat API or knowledge of the model’s chat template. We introduce two methods for constructing adversarial dialogue histories: an adaptation of gray-box prefilling attacks (DIA-I) and an approach based on deferred responses (DIA-II).

## Setup

First, install and serve Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```
or serve it through [docker](https://hub.docker.com/r/ollama/ollama).

```bash
conda create -n ri python=3.12
conda activate ri
cd DIA
chmod a+x install.sh
./install.sh
```

## Quick Start

Run experiments with 10 examples:

- DIA-I
  ```bash
  ollama pull llama3:latest
  bash scripts/run_attack_dia-1_llama3:latest_advbench_test.sh
  ```
- DIA-II
  ```bash
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
python main.py --model_name_a qwen2:latest --model_name_b llama3:latest --model_name_other gemma2:latest --save_path infer_output_qwen2_llama3.json
python main.py --model_name_a llama3:latest --model_name_b qwen2:latest --model_name_other gemma2:latest --save_path infer_output_llama3_qwen2.json
python main.py --model_name_a gemma2:latest --model_name_b qwen2:latest --model_name_other llama3:latest --save_path infer_output_gemma2_qwen2.json
python main.py --model_name_a qwen2:latest --model_name_b gemma2:latest --model_name_other llama3:latest --save_path infer_output_qwen2_gemma2.json
python main.py --model_name_a gemma2:latest --model_name_b llama3:latest --model_name_other qwen2:latest --save_path infer_output_gemma2_llama3.json
python main.py --model_name_a llama3:latest --model_name_b gemma2:latest --model_name_other qwen2:latest --save_path infer_output_llama3_qwen2.json
```

Step 2: Compute accuracy

```bash
python compute_metrics.py
```

The results are stored in `template_infer_results.json`.