import os
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
from jailbreakeval import JailbreakEvaluator

from ri import TaskSolvingAttacker, AttackConfig, eval_batch
from ri.utils import (
    rewrite_instruction,
    add_suffix,
)


@dataclass
class EvalArguments:
    preset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to preset file."},
    )


@dataclass
class OtherArguments:
    prefix_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to prefix file."},
    )
    similar_prefix_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to similar prefix file."},
    )
    similar_response_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to similar response file. Deprecated."},
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
        default=9,
        metadata={"help": "Max tries for rewriting."},
    )
    output_path: str = field(
        default="outputs/results.json",
        metadata={"help": "Path to output file."},
    )
    ablate_rewritten: bool = field(
        default=False,
        metadata={"help": "Ablate rewritten instructions."},
    )


def main():
    parser = HfArgumentParser((AttackConfig, EvalArguments, OtherArguments))

    attack_args, eval_args, other_args = parser.parse_args_into_dataclasses()

    # Read data to list[str]
    with open(other_args.data_path, "r", encoding="utf-8") as file:
        instructs = file.read().split("\n")
        # filter ''
        instructs = list(filter(None, instructs))
    if 'gpt-' in other_args.target_model and 'mini' not in other_args.target_model:
        instructs = instructs[:180]

    # Load attacker and attack
    attacker = TaskSolvingAttacker(other_args.target_model, attack_args)
    if os.path.exists(other_args.output_path):
        with open(other_args.output_path, "r", encoding="utf-8") as file:
            results = json.load(file)
            responses = [result["response"] for result in results]
            eval_results = [result["eval_result"] for result in results]
    else:
        responses = attacker.attack(
            instructs, 
            prefix_path=other_args.prefix_path,
            similar_prefix_path=other_args.similar_prefix_path,
            similar_response_path=other_args.similar_response_path,
            options = {"num_ctx": 8192, "num_predict": 1024, "seed": 42},
            seed=42, # For the first run, we ensure the same random seed
        )

        # Eval
        if eval_args.preset_path:
            evaluator = JailbreakEvaluator.from_preset(eval_args.preset_path)
            eval_results = eval_batch(evaluator, questions=instructs, answers=responses)
        else:
            eval_results = [None] * len(instructs)

        # Save instructs, responses, and eval_results to json
        results = []
        for instruct, response, eval_result in zip(instructs, responses, eval_results):
            results.append({
                "instruct": instruct,
                "response": response,
                "eval_result": eval_result,
            })
        os.makedirs(os.path.dirname(other_args.output_path), exist_ok=True)
        with open(other_args.output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    responses_ls = [responses] # list[list[str]]
    eval_results_ls = [eval_results] # list[list[bool]]
    assert other_args.max_rewrite_tries >= 0 and other_args.max_rewrite_tries < 11
    for i in range(other_args.max_rewrite_tries):
        rewritten_output_path = add_suffix(other_args.output_path, f"_rewritten{i+1}")
        if os.path.exists(rewritten_output_path):
            with open(rewritten_output_path, "r", encoding="utf-8") as file:
                results = json.load(file)
                rewritten_responses = [result["response"] for result in results]
                rewritten_eval_results = [result["eval_result"] for result in results]
            responses_ls.append(rewritten_responses)
            eval_results_ls.append(rewritten_eval_results)
            continue

        # Generate rewritten instructions
        if other_args.ablate_rewritten:
            rewritten_data_path = other_args.data_path
        else:
            rewritten_data_path = add_suffix(other_args.data_path, f"_rewritten{i+1}")
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
        
        # Name rewritten cache file(s)
        if other_args.ablate_rewritten:
            rewritten_prefix_path = other_args.prefix_path
        else:
            rewritten_prefix_path = add_suffix(other_args.prefix_path, f"_rewritten{i+1}") if other_args.prefix_path else None
        rewritten_responses = attacker.attack(
            rewritten_instructs, 
            prefix_path=rewritten_prefix_path,
            similar_prefix_path=other_args.similar_prefix_path,
            similar_response_path=other_args.similar_response_path,
        )
        responses_ls.append(rewritten_responses)

        if eval_args.preset_path:
            rewritten_eval_results = eval_batch(evaluator, questions=instructs, answers=rewritten_responses) # NOTE: use original instructs
        else:
            rewritten_eval_results = [None] * len(instructs)
        eval_results_ls.append(rewritten_eval_results)

        # Save rewritten results to json
        results = []
        for instruct, response, eval_result in zip(instructs, rewritten_responses, rewritten_eval_results):
            results.append({
                "instruct": instruct,
                "response": response,
                "eval_result": eval_result,
            })
        with open(rewritten_output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
    
    # Compute ASR (OR gate)
    if eval_args.preset_path:
        or_eval_results = [any(eval_res) for eval_res in zip(*eval_results_ls)]
        asr = sum(or_eval_results) / len(or_eval_results)
        print(f"ASR: {asr}")


if __name__ == "__main__":
    main()