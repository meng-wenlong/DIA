import os
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from renellm import rene
from typing import Optional
from tqdm import tqdm

from di.utils import add_suffix


@dataclass
class ReneArguments:
    rewrite_model: str = field(
        default="gemma2:27b",
        metadata={"help": "Model used for rewriting the prompt."},
    )
    gpt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "GPT API key."},
    )
    gpt_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "GPT base URL."},
    )
    round_sleep: int = field(
        default=1,
        metadata={"help": "Sleep time between every round."},
    )
    fail_sleep: int = field(
        default=1,
        metadata={"help": "Sleep time for fail response."},
    )
    retry_times: int = field(
        default=100,
        metadata={"help": "Retry times when exception occurs."},
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Model temperature."},
    )
    judge_model: str = field(
        default="gemma2:27b",
        metadata={"help": "Model used for harmful classification."},
    )


@dataclass
class OtherArguments:
    data_path: str = field(
        default="../../data/HEx-PHI.txt",
        metadata={"help": "Path to the data file."},
    )
    max_rewrite_tries: int = field(
        default=9,
        metadata={"help": "Max tries for rewriting the prompt."},
    )
    output_path: str = field(
        default="temp/HEx-PHI/renellm_nested_prompt.json",
        metadata={"help": "Path to the output file."},
    )


def main():
    parser = HfArgumentParser((ReneArguments, OtherArguments))
    rene_args, other_args = parser.parse_args_into_dataclasses()

    with open(other_args.data_path, "r") as file:
        instructs = file.read().split("\n")
        # filter ''
        instructs = list(filter(None, instructs))

    results = []
    for instruct in tqdm(instructs):
        instruct = instruct.strip()
        nested_prompt, temp_harm_behavior = rene(instruct, rene_args)

        results.append({
            "instruct": instruct,
            "nested_prompt": nested_prompt,
            "temp_harm_behavior": temp_harm_behavior,
        })
    
    os.makedirs(os.path.dirname(other_args.output_path), exist_ok=True)
    with open(other_args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # Begin rewrite
    for i in range(1, other_args.max_rewrite_tries + 1):
        results = []
        for instruct in tqdm(instructs):
            instruct = instruct.strip()
            nested_prompt, temp_harm_behavior = rene(instruct, rene_args)

            results.append({
                "instruct": instruct,
                "nested_prompt": nested_prompt,
                "temp_harm_behavior": temp_harm_behavior,
            })
        
        output_path = add_suffix(other_args.output_path, f"_rewritten{i}")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()