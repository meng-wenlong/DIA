import os
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, AutoTokenizer

from di import ContinueAttacker, AttackConfig, eval_batch
from di.utils import rewrite_instruction


@dataclass
class OtherArguments:
    prefix_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to prefix file."},
    )
    data_path: str = field(
        default="data/test.txt",
        metadata={"help": "Path to data file."},
    )
    target_model: str = field(
        default="llama3:latest",
        metadata={"help": "Target model."},
    )
    output_path: str = field(
        default="outputs/results.json",
        metadata={"help": "Path to output file."},
    )


def get_tokenizer(target_model: str):
    if "llama3.1" in target_model:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif "llama3" in target_model:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif "gemma2" in target_model:
        model_name = "google/gemma-2-2b-it"
    elif "qwen2" in target_model:
        model_name = "Qwen/Qwen2-7B-Instruct"
    else:
        model_name = "Qwen/Qwen2-7B-Instruct"  # default

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def extract_between_markers(text: str, start_marker: str = "Hello!", end_marker: str = ":(") -> str | None:
    start_index = text.find(start_marker)
    if start_index == -1:
        return None

    end_index = text.rfind(end_marker)
    if end_index == -1 or end_index < start_index:
        return None
    
    return text[start_index:end_index + len(end_marker)]


def main():
    parser = HfArgumentParser((AttackConfig, OtherArguments))

    attack_args, other_args = parser.parse_args_into_dataclasses()

    # Read data to list[str]
    with open(other_args.data_path, "r", encoding="utf-8") as file:
        instructs = file.read().split("\n")
        # filter ''
        instructs = list(filter(None, instructs))
    
    # Load attacker and attack
    attacker = ContinueAttacker(other_args.target_model, attack_args)
    
    attack_messages = attacker.attack(
        instructs,
        prefix_path=other_args.prefix_path,
        options = {"num_ctx": 8192, "num_predict": 1024, "seed": 42},
        seed=42,
        return_prompt_only=True,
    )
    
    tokenizer = get_tokenizer(other_args.target_model)
    attack_prompts = []
    for instruct, attack_message in zip(instructs, attack_messages):
        attack_prompt = tokenizer.apply_chat_template(attack_message, tokenize=False)
        webui_attack_prompt = extract_between_markers(attack_prompt, start_marker="Hello!", end_marker=":(")

        attack_prompts.append({
            "instruct": instruct,
            "attack_prompt": attack_prompt,
            "webui_attack_prompt": webui_attack_prompt,
        })

    # Save to output_path
    os.makedirs(os.path.dirname(other_args.output_path), exist_ok=True)
    with open(other_args.output_path, "w", encoding="utf-8") as file:
        json.dump(attack_prompts, file, indent=2)


if __name__ == "__main__":
    main()