MODEL=llama3:70b

python -u eval_results.py \
--result_path results/$MODEL/advbench.json \
--preset_path ../../presets/HFChat-inan2023llama-llamaguard2.yaml \