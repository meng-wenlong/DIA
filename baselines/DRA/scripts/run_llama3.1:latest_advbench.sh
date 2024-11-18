MODEL=llama3.1:latest
DATASET=advbench

python -u main.py \
--model $MODEL \
--data_path ../../data/$DATASET.txt \
--output_path results/$MODEL/$DATASET.json \