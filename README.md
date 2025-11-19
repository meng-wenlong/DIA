<p align="center">
  <img src="assets/logo.png" width="300"/>
</p>

# Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-0.12.6-brightgreen.svg)


This repository contains the official implementation of DIA (Dialogue Injection Attack), a new jailbreak attack paradigm that leverages fabricated dialogue history to enhance the effectiveness of adversarial prompts against large language models (LLMs). Unlike prior jailbreak studies that focus primarily on single-turn interactions or assume the attacker can only modify user inputs, DIA reveals a broader and more powerful attack surface, **the ability to manipulate a model’s historical outputs**.

DIA operates entirely in a black-box setting, requiring only access to a chat API or knowledge of the model’s chat template. We introduce two methods for constructing adversarial dialogue histories: an adaptation of gray-box prefilling attacks (DIA-I) and an approach based on deferred responses (DIA-II).

## Set-up

### Installation

First, install and serve Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```
or serve it through [docker](https://hub.docker.com/r/ollama/ollama).

Then intall dependicies

```bash
conda create -n di python=3.12
conda activate di
cd DIA
chmod a+x install.sh
./install.sh
```

### Basic Test

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

If you encounter a Java version error, you can install Java via conda
```bash
conda install conda-forge::openjdk
```

<!-- If you want reproduce the results in Table 2 and Figure 5:

- DIA-I
  ```bash
  bash scripts/run_attack_dia-1_llama3:latest_advbench.sh
  ```
- DIA-II
  ```bash
  bash scripts/run_attack_dia-2_llama3:latest_advbench.sh
  ```

It will take 4~5 hours. -->

## Main Experiments

To reproduce our main experiments, you can use our provided scripts
```bash
bash scripts/run_attack_dia-1_llama3:latest_advbench.sh
```
for DIA-I and
```bash
bash scripts/run_attack_dia-2_llama3:latest_advbench.sh
```
for DIA-II.

Below is the content of `run_attack_dia-2_llama3:latest_advbench.sh`.
```bash
DATA=advbench  # HEx-PHI
MODEL=llama3:latest
PRESET=llamaguard2  # llamaguard3 

python -u exps/demo_task_solve_rewrite.py \
--seed 42 \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results/dia2/${MODEL}/${DATA}.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json
```

Usually, you only need to change `DATA`, `MODEL`, and `PRESET`. `DATA` determines the dataset we evaluate on. `MODEL` is the victim model. It can be an Ollama model, a GPT model, or a Claude model. `PRESET` is the preset for JailbreakEval. You can choose llamaguad2 or llamaguard3.
For example, if you want to test gpt-4o on HEx-PHI dataset using llamaguard3 as the evaluator, you should set
```bash
DATA=HEx-PHI
MODEL=gpt-4o-2024-08-06
PRESET=llamaguard3
```

If you want to do more, here are argument explaination for advanced users.

- `preset_path (str)`: The preset file path for JailbreakEval. JailbreakEval requires a specific format of preset to config evaluation.
- `prefix_path (str)`: Affirmative beginnings path. If the file does not exist, DIA will regenerate affirmative beginnings and save them to the path. If it is None, DIA will regenrate but not save.
- `similar_prefix_path (str)`: Affirmative beginning path of similar unharmful questions. If the file does not exist, DIA will regenerate affirmative beginnings and save them to the path. If it is None, DIA will regenerate but not save.
- `similar_response_path (str)`: Response path of similar unharmful questions. If the file does not exist, DIA will regenerate responses and save them to the path. If it is None, DIA will regenerate but not save.
- `data_path (str)`: Harmful instruction dataset. A single line is a harmful instruction.
- `target_model (str)`: Target victim model. Can be an Ollama model, an OpenAI gpt model, or an Anthropic claude model.
- `max_rewrite_tries (int)`: Max times of rewrite. Default to 9, which means for each instruction we will rewrite it 9 times, and query the victim model 10 times. It set to 0, will diable rewrite.
- `output_path (str)`: Path to save model responses.
- `ablation_rewritten (bool)`: Whether to apply rewritten ablation. If True, DIA will query the victim model multi-times but does not rewrite the original instruction.
- `seed (int)`: Random seed. Only effective for the first round.

You can also evaluate after performing attacks using `eval_results_multi_iter.py`.

```bash
MODEL=llama3.1:70b
DATA=advbench
PRESET=llamaguard3

python -u exps/eval_results_multi_iter.py \
--result_path results/dia2/$MODEL/${DATA}.json \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/dia2/$MODEL/${DATA}_eval-${PRESET}.json \
--override
```

Ensure that `result_path` already exists.

## WebUI Attack

You can use bellow command to generate dialogue injection prompts
```bash
DATA=advbench
MODEL=llama3.1:70b

# DIA-I
python -u exps/gen_prompt_continue.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results/dia1-prompts/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json

# DIA-II
python -u exps/gen_prompt_task-solve.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results/dia2-prompts/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
```

It will generate a json file. You can paste "webui_attack_prompt" to attack WebUIs.

You can also evaluate after performing attacks using `eval_results_multi_iter.py`.

```bash
MODEL=llama3.1:70b
DATA=advbench
PRESET=llamaguard3

python -u exps/eval_results_multi_iter.py \
--result_path results/dia2/$MODEL/${DATA}.json \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/dia2/$MODEL/${DATA}_eval-${PRESET}.json
```

## Ablation Study

### Ablate System replacement

Set `--ablate_system_prompt` to True.

```bash
DATA=advbench
MODEL=llama3:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_ablate_system/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--ablate_system_prompt True
```

### Ablate Hypnosis

Set `--ablate_hypnosis` to True.

```bash
DATA=advbench
MODEL=llama3:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_ablate_hypnosis/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--ablate_hypnosis True
```

### Ablate Answer Guidance

Set `--ablate_answer_guidance` to True.

```bash
DATA=advbench
MODEL=llama3:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_answer_guidance/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--ablate_answer_guidance True
```

### Ablate Rewrite

Set `--ablate_rewritten` to True.

```bash
DATA=advbench
MODEL=llama3:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_ablate_rewritten/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--ablate_rewritten True
```

### Ablate Auxialiary Model

Go to `src/di/utils.py`. Set `DEFAULT_AUX_MODEL` to the auxiliary model you want to use. Then, re-run attack.

```bash
DATA=advbench
MODEL=llama3.1:70b

python -u exps/gen_prompt_task-solve.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_ablate_aux/dia2/$MODEL/$DATA.json \
--prefix_path temp_ablate_aux/$DATA-prefixes.json \
--similar_prefix_path temp_ablate_aux/similar-prefixes.json \
--max_rewrite_tries 0
```

## Defenses

### OpenAI Moderation

Set OPENAI_API_KEY.
```bash
export OPENAI_API_KEY=your_api_key
```

Run `eval_results_defense_openai_moderation.py`

```bash
MODEL=gemma2:latest
DATA=advbench
PRESET=llamaguard2

python -u exps/eval_results_defense_openai_moderation.py \
--result_path results/dia2/$MODEL/${DATA}.json \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/dia2/$MODEL/${DATA}_eval-moderation-${PRESET}.json
```

### Perplexity Filter

Run `eval_results_defense_perplexity_filter.py`

```bash
MODEL=gemma2:latest
DATA=advbench
PRESET=llamaguard2

python -u exps/eval_results_defense_perplexity_filter.py \
--result_path results/dia2/$MODEL/${DATA}.json \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/dia2/$MODEL/${DATA}_eval-perplexity-${PRESET}.json
```

### Defensive System Prompt

Set `--prompt_defense_method` to system_prompt.

```bash
DATA=advbench
MODEL=gemma2:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_defense_system/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--prompt_defense_method system_prompt
```

### Defensive Prompt Patch

Set `--prompt_defense_method` to prompt_patch.

```bash
DATA=advbench
MODEL=gemma2:latest

python -u exps/demo_task_solve_rewrite.py \
--use_initial_system_prompt False \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_defense_patch/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--prompt_defense_method prompt_patch
```

### Bergeron

Set `--prompt_defense_method` to bergeron.

```bash
DATA=advbench
MODEL=gemma2:latest

python -u exps/demo_task_solve_rewrite.py \
--use_initial_system_prompt False \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_defense_bergeron/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--prompt_defense_method bergeron
```

### Input Canonicalization

Set `--prompt_defense_method` to canonicalize.

```bash
DATA=advbench
MODEL=gemma2:latest

python -u exps/demo_task_solve_rewrite.py \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results_defense_canonicalize/dia2/$MODEL/$DATA.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0 \
--prompt_defense_method canonicalize
```

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

## Baselines

We reproduce baseline in a simillar manner to DIA by unifying their interface and changing their backend to Ollama.
Detailed instructions are placed in [baselines/](baselines).