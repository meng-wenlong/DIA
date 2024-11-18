DATA=MaliciousInstructs
MODEL=llama3:latest

python -u main.py \
--data_path ../../data/${DATA}.txt \
--prefix_path ../../temp/${DATA}-prefixes.json \
--output_path results/${MODEL}/${DATA}.json \
--target_model ${MODEL} \
--attack_model gemma2:27b \
--judge_model gemma2:27b