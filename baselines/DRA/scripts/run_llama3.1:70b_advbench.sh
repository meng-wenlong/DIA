MODEL=llama3.1:70b
DATASET=advbench

python -u main.py \
--model $MODEL \
--data_path ../../data/$DATASET.txt \
--output_path results/$MODEL/$DATASET.json \