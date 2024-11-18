import json
import os
from utils.scenario_nest_utils import extract_instruct


datasets = [
    "advbench",
    "HEx-PHI",
    "MaliciousInstructs",
]


def read_nested_prompts(dataset: str, rewritten: int = 0):
    if rewritten == 0:
        file_path = os.path.join("temp", dataset, "renellm_nested_prompt.json")
    else:
        file_path = os.path.join("temp", dataset, f"renellm_nested_prompt_rewritten{rewritten}.json")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        return data
    

def get_output_path(dataset: str, rewritten: int = 0):
    if rewritten == 0:
        file_path = os.path.join("temp", dataset, "renellm_unnested_prompt.json")
    else:
        file_path = os.path.join("temp", dataset, f"renellm_unnested_prompt_rewritten{rewritten}.json")
    return file_path


def main():
    for dataset in datasets:
        for rewritten in range(10):
            nested_prompts = read_nested_prompts(dataset, rewritten)
            unnested_prompts = []
            for item in nested_prompts:
                nested_prompt = item["nested_prompt"]
                unnested_prompt = extract_instruct(nested_prompt)
                if unnested_prompt is None:
                    raise ValueError(f"Failed to extract instruct from nested prompt: {nested_prompt}")
                unnested_prompts.append({
                    "instruct": item["instruct"],
                    "unnested_prompt": unnested_prompt
                })
            
            output_path = get_output_path(dataset, rewritten)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w") as f:
                json.dump(unnested_prompts, f, indent=2)


if __name__ == "__main__":
    main()