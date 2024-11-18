import re
import os
import json
from main import generate_inception_prompt


# read advbench.json or advbench_rewritten{n}.json
def update_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(r'advbench(_rewritten\d+)?\.json', file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                updated = False
                for item in data:
                    if 'attack_prompt' not in item:
                        if 'instruct' in item:
                            item['attack_prompt'] = generate_inception_prompt(item['instruct'])
                            updated = True
                
                if updated:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=4)
                        print(f"Updated {file_path}")


if __name__ == "__main__":
    update_json_files("results")