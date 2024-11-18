import argparse
import json
import re
import random
import evaluate
from template_config import model2template
from tqdm import tqdm


accuracy_metric = evaluate.load("accuracy")


def template_infer(
    response: str,
    Pa2: str,
    Sa1: str,
    assistant_text: str = "assistant text placeholder",
) -> int:
    Pa2 = Pa2.strip()
    Sa1 = Sa1.strip()

    if Pa2 in response and Sa1 in response:
        return 0 # neither 1 nor 2
    
    # assistant_text*Sa1
    pattern = re.compile(r"(?i:" + re.escape(assistant_text) + r").*" + re.escape(Sa1), re.DOTALL)
    if pattern.search(response):
        return 2
    # if response.lower() == assistant_text.lower():
    #     return 2

    # Pa2*assistant_text
    pattern = re.compile(re.escape(Pa2) + r".*(?i:" + re.escape(assistant_text) + r")", re.DOTALL)
    if pattern.search(response):
        return 1
    
    return -1 # cannot determine, maybe others


def compute_metrics(
    model_a: str = "qwen2",
    model_b: str = "gemma2",
    max_try_times: int = 5,
    model_query_times: int = 200,
):
    # Load two files
    infer_output_path = f"./infer_output_{model_a}_{model_b}.json"
    aux_infer_output_path = f"./infer_output_{model_b}_{model_a}.json"
    with open(infer_output_path, 'r') as f:
        infer_output = json.load(f)
    with open(aux_infer_output_path, 'r') as f:
        aux_infer_output = json.load(f)

    model_a_template = model2template[model_a]()
    model_b_template = model2template[model_b]()
    
    def query_one_example(
        model_column: str = "a_res",
    ) -> int:
        exchange_a_b = False
        for _ in range(max_try_times):
            query_response = random.choice(infer_output[model_column])
            infer_res = template_infer(
                response=query_response,
                Pa2=model_b_template.Pa,
                Sa1=model_a_template.Sa,
            )
            if infer_res != -1:
                break
        
        if infer_res == -1:
            exchange_a_b = True
            model_column_switch = {
                "a_res": "b_res",
                "b_res": "a_res",
                'other_res': 'other_res'
            }
            model_column = model_column_switch[model_column]
            for _ in range(max_try_times):
                query_response = random.choice(aux_infer_output[model_column])
                infer_res = template_infer(
                    response=query_response,
                    Pa2=model_a_template.Pa,
                    Sa1=model_b_template.Sa,
                )
                if infer_res != -1:
                    break

        if exchange_a_b and infer_res > 0:
            switch = {1: 2, 2: 1}
            infer_res = switch[infer_res]
        elif infer_res == -1:
            infer_res = 0

        return infer_res

    # query model a, b, and other
    # Query a
    a_query_res = []
    for i in tqdm(range(model_query_times)):
        a_query_res.append(query_one_example(model_column="a_res"))
    
    # Query b
    b_query_res = []
    for i in tqdm(range(model_query_times)):
        b_query_res.append(query_one_example(model_column="b_res"))

    # Query other
    other_query_res = []
    for i in tqdm(range(model_query_times)):
        other_query_res.append(query_one_example(model_column="other_res"))

    references = [1] * model_query_times + [2] * model_query_times + [0] * model_query_times
    predictions = a_query_res + b_query_res + other_query_res
    acc = accuracy_metric.compute(references=references, predictions=predictions)['accuracy']
    
    return {
        "accuracy": acc,
        "a_query_res": a_query_res,
        "b_query_res": b_query_res,
        "other_query_res": other_query_res
    }


if __name__ == "__main__":
    models = ['qwen2', 'gemma2', 'llama3']

    template_infer_results = []
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            for max_try_times in [1, 2, 3, 4, 5]:
                infer_metrics = compute_metrics(
                    model_a=model_a,
                    model_b=model_b,
                    max_try_times=max_try_times,
                )
                template_infer_results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "max_try_times": max_try_times,
                    "accuracy": infer_metrics["accuracy"]
                })

    with open("./template_infer_results.json", 'w') as f:
        json.dump(template_infer_results, f, indent=2)