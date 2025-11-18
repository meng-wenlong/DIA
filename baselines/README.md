# DeepInception

Run attack

```bash
cd DeepInception

DATA=advbench
MODEL=llama3:latest

python -u main.py \
--target_model $MODEL \
--data_path ../../data/$DATA.txt \
--output_path results/$MODEL/$DATA.json
```

Eval attack

```bash
MODEL=llama3:latest
DATA=advbench
PRESET=llamaguard2

python -u eval_results.py \
--result_path results/$MODEL/${DATA}.json \
--preset_path ../../presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/$MODEL/${DATA}_eval-${PRESET}.json
```


# DRA

Run attack

```bash
cd DRA
pip install detoxify

MODEL=llama3:latest
DATASET=advbench

python -u main.py \
--model $MODEL \
--data_path ../../data/$DATASET.txt \
--output_path results/$MODEL/$DATASET.json
```

Eval attack

```bash
MODEL=llama3:latest
DATA=advbench
PRESET=llamaguard2

python -u eval_results.py \
--result_path results/$MODEL/${DATA}.json \
--preset_path ../../presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/$MODEL/${DATA}_eval-${PRESET}.json
```


# PAIR

Run attack

```bash
cd PAIR
pip install wandb

DATA=advbench
MODEL=llama3:latest

python -u main.py \
--data_path ../../data/${DATA}.txt \
--prefix_path ../../temp/${DATA}-prefixes.json \
--output_path results/${MODEL}/${DATA}.json \
--target_model ${MODEL} \
--attack_model gemma2:27b \
--judge_model gemma2:27b \
--n_streams 1
```

Eval attack

```bash
MODEL=llama3:latest
DATA=advbench
PRESET=llamaguard2

python -u eval_results.py \
--result_path results/$MODEL/${DATA}.json \
--preset_path ../../presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/$MODEL/${DATA}_eval-${PRESET}.json
```

# ReneLLM

Run attack

```bash
cd ReNeLLM

DATA=advbench
MODEL=llama3:latest

python -u get_responses_ollama.py \
--target_model ${MODEL} \
--prompt_path temp/${DATA}/renellm_nested_prompt.json \
--output_path results/${MODEL}/${DATA}.json
```

Eval attack

```bash
MODEL=llama3:latest
DATA=advbench
PRESET=llamaguard2

python -u eval_results.py \
--result_path results/$MODEL/${DATA}.json \
--preset_path ../../presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/$MODEL/${DATA}_eval-${PRESET}.json
```