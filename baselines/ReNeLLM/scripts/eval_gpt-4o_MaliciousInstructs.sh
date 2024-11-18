DATA=MaliciousInstructs
MODEL=gpt-4o-2024-08-06
PRESET=llamaguard2

python -u eval_results.py \
--result_path results/$MODEL/${DATA}.json \
--preset_path ../../presets/HFChat-inan2023llama-${PRESET}.yaml \
--output_path results/$MODEL/${DATA}_eval-${PRESET}.json \
--override