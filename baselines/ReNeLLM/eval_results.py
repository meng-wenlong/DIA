import argparse
import json
import os
from jailbreakeval import JailbreakEvaluator
from ri.eval import eval_batch, OllamaChatService
from ri.utils import add_suffix


def find_rewritten_files(filepath: str) -> list[str]:
    '''return a list of rewritten paths
    ''' 
    rewritten_paths = []
    i = 1
    while True:
        rewritten_path = add_suffix(filepath, f"_rewritten{i}")
        if not os.path.exists(rewritten_path):
            break
        rewritten_paths.append(rewritten_path)
        i += 1
    return rewritten_paths


def eval_result_with_evaluator(
    result_path: str,
    evaluator: JailbreakEvaluator,
    instruct_column: str = "nested_prompt",
):
    with open(result_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    instructs = []
    responses = []
    for result in results:
        instructs.append(result[instruct_column])
        responses.append(result["response"])

    eval_results = eval_batch(evaluator, questions=instructs, answers=responses)
    return eval_results


def merge_results(results_list: list[list[bool]]) -> list[bool]:
    accumulated_results = [False] * len(results_list[0])

    for results in results_list:
        for index, result in enumerate(results):
            if result:
                accumulated_results[index] = True
    return accumulated_results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, required=True, help="Path to result file.")
    parser.add_argument("--preset_path", type=str, default=None, help="Path to preset file.")
    parser.add_argument("--output_path", type=str, default="output.json", help="output path")
    parser.add_argument("--override", action="store_true", help="override output file if exists")
    parser.add_argument("--instruct_column", type=str, default="nested_prompt", help="column name of instructs")

    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.override:
        print(f"Output file {args.output_path} exists. Use --override to override.")
        return

    rewritten_paths = find_rewritten_files(args.result_path)
    # insert result_path to the first place
    rewritten_paths.insert(0, args.result_path)

    if args.preset_path is not None:
        evaluator = JailbreakEvaluator.from_preset(args.preset_path)
    else:
        evaluator = None

    asr_list: list[float] = []
    eval_results_list:list[list[bool]] = []
    if evaluator:
        for rewritten_path in rewritten_paths:
            eval_results = eval_result_with_evaluator(rewritten_path, evaluator, args.instruct_column)
            eval_results_list.append(eval_results)
            
            accumulated_results = merge_results(eval_results_list)
            asr = sum(accumulated_results) / len(accumulated_results)
            asr_list.append(asr)
    else:
        for rewritten_path in rewritten_paths:
            with open(rewritten_path, "r") as file:
                results = json.load(file)

            eval_results = [
                result['eval_result'] for result in results
            ]
            eval_results_list.append(eval_results)

            accumulated_results = merge_results(eval_results_list)
            asr = sum(accumulated_results) / len(accumulated_results)
            asr_list.append(asr)

    for i in range(len(asr_list)):
        print(f"ASR@{i + 1}: {asr_list[i]}")

    asr_dict = {f"ASR@{i + 1}": asr_list[i] for i in range(len(asr_list))}
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(asr_dict, file, indent=2)


if __name__ == "__main__":
    main()
