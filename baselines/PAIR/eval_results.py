import os
import argparse
import json
from tqdm import tqdm
from jailbreakeval import JailbreakEvaluator
from ri.eval import OllamaChatService


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, required=True, help="Path to result file.")
    parser.add_argument("--preset_path", type=str, required=True, help="Path to preset file.")
    parser.add_argument("--iter", type=int, default=10, help="iter time")
    parser.add_argument("--output_path", type=str, default="output.json", help="output path")
    parser.add_argument("--n_streams", type=int, default=1, help="number of streams")

    args = parser.parse_args()

    with open(args.result_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    evaluator = JailbreakEvaluator.from_preset(args.preset_path)

    success_counts = [0] * args.iter

    for result in tqdm(results):
        attacks = result["attacks"]
        for i, attack in enumerate(attacks):
            if i >= args.iter:
                break

            adv_prompts = attack["adv_prompt_list"][:args.n_streams]
            target_responses = attack["target_responses"][:args.n_streams]
            # handle the case where the target response is too long
            target_responses = [" ".join(target_response.split()[:1024]) if len(target_response.split()) > 1024 else target_response for target_response in target_responses]

            if any(evaluator({
                "question": adv_prompt,
                "answer": target_response,
            }) for adv_prompt, target_response in zip(adv_prompts, target_responses)):
                for j in range(i, args.iter):
                    success_counts[j] += 1
                break

    asr_list = [success_count / len(results) for success_count in success_counts]
    for i in range(len(asr_list)):
        print(f"ASR@{i + 1}: {asr_list[i]}")

    asr_dict = {f"ASR@{i + 1}": asr_list[i] for i in range(len(asr_list))}
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(asr_dict, file, indent=2)


if __name__ == "__main__":
    main()