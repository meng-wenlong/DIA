DATA=advbench
MODEL=llama3:latest
PRESET=llamaguard2

python -u exps/demo_task_solve_rewrite.py \
--seed 42 \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--data_path data/$DATA.txt \
--target_model $MODEL \
--output_path results/dia2/${MODEL}/${DATA}.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json