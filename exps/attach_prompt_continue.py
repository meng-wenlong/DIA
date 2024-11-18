import os
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from ri import ContinueAttacker, AttackConfig
from ri.utils import (
    rewrite_instruction,
    add_suffix,
)


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
    max_rewrite_tries: int = field(
        default=0,
        metadata={"help": "Max tries for rewriting."},
    )
    output_path: str = field(
        default="outputs/results.json",
        metadata={"help": "Path to output file."},
    )


def main():
    parser = HfArgumentParser((AttackConfig, OtherArguments))
    attack_args, other_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(other_args.output_path):
        raise FileNotFoundError(f"Output file {other_args.output_path} not found.")

    # Read data to list[str]
    with open(other_args.data_path, "r", encoding="utf-8") as file:
        instructs = file.read().split("\n")
        # filter ''
        instructs = list(filter(None, instructs))
    if 'gpt-' in other_args.target_model and 'mini' not in other_args.target_model:
        instructs = instructs[:180]

    # Load attacker and get prompts
    attacker = ContinueAttacker(other_args.target_model, attack_args)
    prompts = attacker.attack(
        instructs,
        prefix_path=other_args.prefix_path,
        return_prompt_only=True,
    )

    # Attack prompts to output file
    with open(other_args.output_path, "r", encoding="utf-8") as file:
        results = json.load(file)
    for i, result in enumerate(results):
        result["last_prompt"] = prompts[i][-1]['content']
    with open(other_args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    assert other_args.max_rewrite_tries >= 0 and other_args.max_rewrite_tries <= 11
    for i in range(other_args.max_rewrite_tries):
        rewritten_output_path = add_suffix(other_args.output_path, f"_rewrite_{i}")

        rewritten_data_path = add_suffix(other_args.data_path, f"_rewrite_{i}")
        if not os.path.exists(rewritten_data_path):
            rewritten_instructs = []
            for instruct in instructs:
                rewritten_instructs.append(rewrite_instruction(instruct))
            with open(rewritten_data_path, "w", encoding="utf-8") as file:
                file.write("\n".join(rewritten_instructs))
        else:
            with open(rewritten_data_path, "r", encoding="utf-8") as file:
                rewritten_instructs = file.read().split("\n")
                if len(rewritten_instructs) > len(instructs):
                    rewritten_instructs = rewritten_instructs[:len(instructs)]
        
        rewritten_prefix_path = add_suffix(other_args.prefix_path, f"_rewrite_{i}") if other_args.prefix_path else None

        rewritten_prompts = attacker.attack(
            rewritten_instructs,
            prefix_path=rewritten_prefix_path,
            return_prompt_only=True,
        )

        # Attach prompts to output file
        with open(rewritten_output_path, "r", encoding="utf-8") as file:
            results = json.load(file)
        for j, result in enumerate(results):
            result["last_prompt"] = rewritten_prompts[j][-1]['content']
        with open(rewritten_output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()