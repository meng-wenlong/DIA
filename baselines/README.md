# DRA

```bash
cd DRA
pip install detoxify

python -u main.py \
--model gpt-4o-mini \
--data_path ../../data/advbench.txt \
--output_path results/gpt-4o-mini/advbench.json
```


# DeepInception


# PAIR

```bash
cd PAIR
pip install wandb

python main.py \
--data_path ../../data/advbench.txt \
--prefix_path ../../data/advbench_official_prefix.json \
--output_path outputs/advbench_results.json \
--target_model vicuna:13b \
--attack_model vicuna:13b \
--judge_model gpt-4o-mini \
```

# ReneLLM

```bash
cd ReNeLLM
pip install anthropic==0.32.0
```