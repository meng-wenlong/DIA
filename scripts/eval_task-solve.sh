MODEL=llama3.1:latest
DATA=advbench
PRESET=llamaguard2

python -u exps/eval_results_multi_iter.py \
--result_path results/task-solve/$MODEL/${DATA}.json \
--preset_path presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/task-solve/$MODEL/${DATA}_eval-${PRESET}.json