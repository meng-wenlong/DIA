MODEL=llama3:latest
DATASET=HEx-PHI

# python -u eval_results.py \
# --result_path outputs/$MODEL/$DATASET.json \
# --preset_path ../../presets/HFChat-inan2023llama-llamaguard2.yaml \
# --output_path outputs/$MODEL/${DATASET}_eval-llamaguard2.json

# python -u eval_results.py \
# --result_path outputs/$MODEL/$DATASET.json \
# --preset_path ../../presets/HFChat-inan2023llama-llamaguard3.yaml \
# --output_path outputs/$MODEL/${DATASET}_eval-llamaguard3.json

python -u eval_results.py \
--result_path outputs/$MODEL/$DATASET.json \
--preset_path ../../presets/OpenAITextClassification-flagged-answer.yaml \
--output_path outputs/$MODEL/${DATASET}_eval-openai-moderation.json