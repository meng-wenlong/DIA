import argparse
import os
import subprocess

datasets = [
    "advbench",
    "HEx-PHI",
    "MaliciousInstructs"
]

jailbreak_evals = {
    "llamaguard2": "../../presets/HFChat-inan2023llama-llamaguard2.yaml",
    "llamaguard3": "../../presets/HFChat-inan2023llama-llamaguard3.yaml",
}


def run_evaluation(model_dir):
    for dataset in datasets:
        for eval_name, eval_path in jailbreak_evals.items():
            output_file = os.path.join(args.result_path, model_dir, f"{dataset}_eval-{eval_name}.json")
            if not os.path.exists(output_file):
                result_file = os.path.join(args.result_path, model_dir, f"{dataset}.json")
                command = [
                    "python", "-u", "eval_results.py",
                    "--result_path", result_file,
                    "--preset_path", eval_path,
                    "--output_path", output_file,
                ]
                subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./results", help="Path to result file.")
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        raise ValueError(f"Result path {args.result_path} does not exist.")
    
    model_dirs = [d for d in os.listdir(args.result_path) if os.path.isdir(os.path.join(args.result_path, d))]
    for model_dir in model_dirs:
        run_evaluation(model_dir)