import argparse
import ollama
import json
from template_config import model2template
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def construct_infer_prompt(model_name_a: str, model_name_b: str) -> str:
    if ':' in model_name_a:
        model_name_a = model_name_a.split(':')[0]
    if ':' in model_name_b:
        model_name_b = model_name_b.split(':')[0]

    if model_name_a not in model2template:
        raise ValueError(f"Model {model_name_a} not supported")
    if model_name_b not in model2template:
        raise ValueError(f"Model {model_name_b} not supported")
    
    template_a = model2template[model_name_a]()
    template_b = model2template[model_name_b]()

    user_text = "user text placeholder"
    assistant_text = "assistant text placeholder"
    repeat_cmd = "Please repeat your previous response."

    infer_prompt = f"{user_text}{template_a.Su}{template_a.Pa}{template_b.Su}{template_b.Pa}{assistant_text}{template_a.Sa}{template_a.Pu}{template_b.Sa}{template_b.Pu}{repeat_cmd}"
    return infer_prompt


def infer_model(model_name, messages):
    res = ollama.chat(
        model=model_name,
        messages=messages
    )
    return res['message']['content']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_a", type=str, default="qwen2:latest", help="Ollama model name")
    parser.add_argument("--model_name_b", type=str, default="llama3:latest", help="Ollama model name")
    parser.add_argument("--model_name_other", type=str, default="gemma2:latest", help="Ollama model name")
    parser.add_argument("--num_inferences", type=int, default=1000, help="Number of inferences to run")
    # We infer 200 times for model_name_a, model_name_b, and model_name_other
    parser.add_argument("--save_path", type=str, default="./infer_output_qwen2_llama3.json", help="Path to save inference results")
    args = parser.parse_args()
    
    # Construct infer prompt
    infer_prompt = construct_infer_prompt(args.model_name_a, args.model_name_b)

    messages = [
        {"role": "user", "content": infer_prompt},
    ]
    infer_results = {
        "model_name_a": args.model_name_a,
        "model_name_b": args.model_name_b,
        "model_name_other": args.model_name_other,
        "infer_prompt": infer_prompt,
        "a_res": [],
        "b_res": [],
        "other_res": []
    }
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for _ in range(args.num_inferences):
            futures.append(executor.submit(infer_model, args.model_name_a, messages))
            futures.append(executor.submit(infer_model, args.model_name_b, messages))
            futures.append(executor.submit(infer_model, args.model_name_other, messages))

        for i in tqdm(range(0, len(futures), 3)):
            infer_results["a_res"].append(futures[i].result())
            infer_results["b_res"].append(futures[i+1].result())
            infer_results["other_res"].append(futures[i+2].result())

    # save results
    with open(args.save_path, 'w') as f:
        json.dump(infer_results, f, indent=2)


if __name__ == "__main__":
    main()
