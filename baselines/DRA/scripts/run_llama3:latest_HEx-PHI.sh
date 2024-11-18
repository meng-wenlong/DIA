MODEL=llama3:latest
DATASET=HEx-PHI

python -u main.py \
--model $MODEL \
--data_path ../../data/$DATASET.txt \
--output_path results/$MODEL/$DATASET.json \