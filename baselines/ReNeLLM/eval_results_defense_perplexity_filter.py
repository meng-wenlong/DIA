import argparse
import json
import os
import evaluate
from jailbreakeval import JailbreakEvaluator
from ri.eval import eval_batch, OllamaChatService
from ri.utils import add_suffix
from typing import Optional


class PerplexityFilter:
    def __init__(self, model_name: str = "google/gemma-2-2b", perplexity_threshold: Optional[float] = None,baseline_file_paths: Optional[list[str]] = None) -> None:
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        self.model_name = model_name

        if perplexity_threshold is None:
            # Compute default perplexity_threshold
            if baseline_file_paths is None:
                baseline_file_paths = [
                    "../data/advbench.txt",
                    "../data/HEx-PHI.txt",
                    "../data/MaliciousInstructs.txt"
                ]
            # read baseline files
            baseline_sentences = []
            for baseline_file_path in baseline_file_paths:
                with open(baseline_file_path, "r", encoding="utf-8") as file:
                    instructs = file.read().split("\n")
                    instructs = list(filter(None, instructs))
                baseline_sentences.extend(instructs)
            import pdb; pdb.set_trace()
            perplexity_results = self.perplexity.compute(predictions=baseline_sentences, model_id=self.model_name)
            self.perplexity_threshold = max(perplexity_results["perplexities"])
        else:
            self.perplexity_threshold = perplexity_threshold
    
    def __call__(self, data: dict[str, str]) -> bool:
        perplexity_results = self.perplexity.compute(predictions=[data["question"]], model_id=self.model_name)
        return perplexity_results["mean_perplexity"] > self.perplexity_threshold
    
    def evaluate_batch(self, questions: list[str]) -> list[bool]:
        perplexity_results = self.perplexity.compute(predictions=questions, model_id=self.model_name)
        return [perplexity > self.perplexity_threshold for perplexity in perplexity_results['perplexities']]


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

    if isinstance(evaluator, PerplexityFilter):
        eval_results = evaluator.evaluate_batch(questions)
    else:
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
    work_dir = os.path.dirname(os.path.dirname(current_dir))
    baseline_file_paths = [
        os.path.join(work_dir, "data", "advbench.txt"),
        # os.path.join(work_dir, "data", "HEx-PHI.txt"),
        # os.path.join(work_dir, "data", "MaliciousInstructs.txt")
    ]
    moderator = PerplexityFilter(model_name="google/gemma-2-2b", baseline_file_paths=baseline_file_paths)

    asr_list: list[float] = []
    eval_results_list: list[list[bool]] = []
    if evaluator:
        for rewritten_path in rewritten_paths:
            eval_results = eval_result_with_evaluator(rewritten_path, evaluator, question_column="nested_prompt")
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
        moderation_results = eval_result_with_evaluator(rewritten_path, moderator, question_column="nested_prompt")
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