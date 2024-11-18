DATA=advbench
MODEL=llama3:latest

python -u get_responses_ollama.py \
--target_model ${MODEL} \
--prompt_path temp/${DATA}/renellm_nested_prompt.json \
--output_path results/${MODEL}/${DATA}.json