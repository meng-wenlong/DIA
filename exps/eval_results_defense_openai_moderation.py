import argparse
import json
import os
from jailbreakeval import JailbreakEvaluator
from di.eval import eval_batch, OllamaChatService
from di.utils import add_suffix


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
    question_column: str = "instruct",
    answer_column: str = "response",
):
    with open(result_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    questions = []
    answers = []
    for result in results:
        questions.append(result[question_column])
        answers.append(result[answer_column])

    eval_results = eval_batch(evaluator, questions=questions, answers=answers)
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
    parser.add_argument("--max_rewrite_tries", type=int, default=0, help="max number of rewrite tries")

    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.override:
        print(f"Output file {args.output_path} exists. Use --override to override.")
        return
    
    rewritten_paths = find_rewritten_files(args.result_path)
    # insert result_path to the first place
    rewritten_paths.insert(0, args.result_path)
    rewritten_paths = rewritten_paths[:args.max_rewrite_tries + 1]

    if args.preset_path is not None:
        evaluator = JailbreakEvaluator.from_preset(args.preset_path)
    else:
        evaluator = None

    # Load OpenAI Moderation model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    moderation_preset_path = os.path.join(parent_dir, "presets", "OpenAITextClassification-flagged-input.yaml")
    moderator = JailbreakEvaluator.from_preset(moderation_preset_path)

    asr_list: list[float] = []
    eval_results_list: list[list[bool]] = []
    if evaluator:
        for rewritten_path in rewritten_paths:
            eval_results = eval_result_with_evaluator(rewritten_path, evaluator)
            eval_results_list.append(eval_results)
    else:
        for rewritten_path in rewritten_paths:
            with open(rewritten_path, "r") as file:
                results = json.load(file)

            eval_results = [
                result['eval_result'] for result in results
            ]
            eval_results_list.append(eval_results)
    
    for i, rewritten_path in enumerate(rewritten_paths):
        # TODO: attach prompt column
        moderation_results = eval_result_with_evaluator(rewritten_path, moderator, question_column="last_prompt")
        # If moderation_result is True, we set eval_results_list[i] to False
        for j, moderation_result in enumerate(moderation_results):
            if moderation_result:
                eval_results_list[i][j] = False

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