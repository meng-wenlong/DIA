import argparse
import os
from ri.utils import rewrite_instruction, add_suffix
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/advbench.txt", help="Path to data file.")
    parser.add_argument("--max_rewrite_tries", type=int, default=9)
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as file:
        instructs = file.read().split("\n")
        # filter ''
        instructs = list(filter(None, instructs))

    for i in range(args.max_rewrite_tries):
        rewritten_data_path = add_suffix(args.data_path, f"_rewritten{i+1}")
        if os.path.exists(rewritten_data_path):
            continue

        rewritten_instructs = []
        for instruct in tqdm(instructs):
            rewritten_instructs.append(rewrite_instruction(instruct))
            print("rewritten:\n", rewritten_instructs[-1])

        with open(rewritten_data_path, "w", encoding="utf-8") as file:
            file.write("\n".join(rewritten_instructs))