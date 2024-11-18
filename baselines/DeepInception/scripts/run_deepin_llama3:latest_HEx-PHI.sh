cd ../baselines/DeepInception

DATA=HEx-PHI
MODEL=llama3:latest

python -u main.py \
--target_model $MODEL \
--data_path ../../data/$DATA.txt \
--output_path results/$MODEL/$DATA.json