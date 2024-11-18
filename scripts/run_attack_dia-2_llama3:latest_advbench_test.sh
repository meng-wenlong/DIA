DATA=advbench
MODEL=llama3:latest
PRESET=llamaguard2

python -u exps/demo_continue_task_solve.py \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--data_path data/$DATA-test.txt \
--target_model $MODEL \
--output_path results/dia1/${MODEL}/${DATA}-test.json \
--prefix_path temp/$DATA-prefixes.json \
--similar_prefix_path temp/similar-prefixes.json \
--max_rewrite_tries 0