MODEL=llama3.1:70b

# python -u eval_results.py \
# --result_path results/$MODEL/advbench.json \
# --preset_path ../../presets/HFChat-inan2023llama-llamaguard2.yaml \
# --output_path results/$MODEL/advbench_eval-llamaguard2.json

# python -u eval_results.py \
# --result_path results/$MODEL/advbench.json \
# --preset_path ../../presets/HFChat-inan2023llama-llamaguard3.yaml \
# --output_path results/$MODEL/advbench_eval-llamaguard3.json

python -u eval_results.py \
--result_path results/$MODEL/advbench.json \
--preset_path ../../presets/OpenAITextClassification-flagged-answer.yaml \
--output_path results/$MODEL/advbench_eval-openai-moderation.json