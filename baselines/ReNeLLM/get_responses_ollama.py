import argparse
import copy
import json
import os
import ollama
import logging
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from ri.utils import add_suffix
from ri.prompts import DEFENSE_PROMPT_PATCH, DEFENSE_SYSTEM_PROMPT, PROMPT_CRITIQUE_PROMPT, CONSCIENCE_DISCLAIMER_PROMPT


logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(self, target_model):
        self.target_model = target_model
        if 'gpt-' in self.target_model:
            self.openai_client = OpenAI()
        else:
            self.openai_client = None

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=60))
    def get_response(
        self,
        messages,
        prompt_defense_method: Optional[str] = None, # ['prompt_patch', 'system_prompt', 'bergeron']
        bergeron_secondary_model: Optional[str] = None,
        options: Optional[dict] = None,
        **kwargs
    ) -> str:
        
        if prompt_defense_method is not None:
            processed_messages = copy.deepcopy(messages)
            if prompt_defense_method == 'prompt_patch':
                processed_messages[-1]['content'] += '\n\n' + DEFENSE_PROMPT_PATCH
            elif prompt_defense_method == 'system_prompt':
                if processed_messages[0]['role'] == 'system':
                    processed_messages[0]['content'] = DEFENSE_SYSTEM_PROMPT
                else:
                    processed_messages.insert(0, {'role': 'system', 'content': DEFENSE_SYSTEM_PROMPT})
            elif prompt_defense_method == 'bergeron':
                def is_valid_critique(critique: str):
                    no_critique_flags = ["no change", "not change", "not adversarial"]
                    for flag in no_critique_flags:
                        if flag.lower() in critique.lower():
                            return False
                    return True
                
                def get_critique_response(prompt):
                    critique_messages = [
                        {"role": "user", "content": PROMPT_CRITIQUE_PROMPT.format(prompt=prompt)}
                    ]
                    if bergeron_secondary_model is None:
                        critique_response = self.get_response(critique_messages)
                    else:
                        critique_response = ollama.chat(
                            model=bergeron_secondary_model,
                            messages=critique_messages,
                        )
                    return critique_response if is_valid_critique(critique_response) else ""
                
                def make_conscience_prompt(prompt: str, prompt_critique: str):
                    return CONSCIENCE_DISCLAIMER_PROMPT.format(prompt_critique=prompt_critique, prompt=prompt)
                
                input_critique = get_critique_response(processed_messages[-1]['content'])
                if len(input_critique) > 0:
                    logger.debug("Generating conscience...")
                    sanitized = make_conscience_prompt(processed_messages[-1]['content'], input_critique)
                    processed_messages[-1]['content'] = sanitized
                
            messages = processed_messages

        if self.openai_client:
            response = self.openai_client.chat.completions.create(
                model=self.target_model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        
        response = ollama.chat(
            model=self.target_model,
            messages=messages,
            options=options or {"num_ctx": 8192, "num_predict": 1024},
        ) # type: ignore
        return response['message']['content']
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="llama3:latest")
    parser.add_argument("--prompt_path", type=str, default="temp/advbench/renellm_nested_prompt.json")
    parser.add_argument("--output_path", type=str, default="results/llama3:latest/advbench.json")
    parser.add_argument("--max_rewrite_tries", type=int, default=9)
    parser.add_argument("--prompt_column", type=str, default="nested_prompt")
    parser.add_argument("--prompt_defense_method", type=str, default=None)
    args = parser.parse_args()

    assert args.max_rewrite_tries >= 0 and args.max_rewrite_tries < 11
    
    chatbot = ChatBot(args.target_model)

    with open(args.prompt_path, "r", encoding="utf-8") as file:
        rene_prompts = json.load(file)
    if 'gpt-' in args.target_model and 'mini' not in args.target_model:
        rene_prompts = rene_prompts[:180]

    # try to get system message
    if 'gpt-' not in args.target_model and 'system' in ollama.show(args.target_model):
        system_message = ollama.show(args.target_model)['system']
    else:
        system_message = None
    
    attack_results = []
    for rene_prompt in tqdm(rene_prompts):
        user_prompt = rene_prompt[args.prompt_column]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ] if system_message else [{"role": "user", "content": user_prompt}]

        response = chatbot.get_response(
            messages=messages,
            prompt_defense_method=args.prompt_defense_method,
            options={"num_ctx": 8192, "num_predict": 1024, "seed": 42},
            seed=42,
        )
        attack_results.append({
            "instruct": rene_prompt["instruct"],
            args.prompt_column: user_prompt,
            "response": response,
        })
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(attack_results, file, indent=2)

    for i in range(args.max_rewrite_tries):
        rewritten_prompt_path = add_suffix(args.prompt_path, f"_rewritten{i+1}")
        with open(rewritten_prompt_path, "r", encoding="utf-8") as file:
            rewritten_rene_prompts = json.load(file)
            if len(rewritten_rene_prompts) > len(rene_prompts):
                rewritten_rene_prompts = rewritten_rene_prompts[:len(rene_prompts)]

        attack_results = []
        for rene_prompt in tqdm(rewritten_rene_prompts):
            user_prompt = rene_prompt[args.prompt_column]

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ] if system_message else [{"role": "user", "content": user_prompt}]

            response = chatbot.get_response(
                messages=messages,
                prompt_defense_method=args.prompt_defense_method,
            )
            attack_results.append({
                "instruct": rene_prompt["instruct"],
                args.prompt_column: user_prompt,
                "response": response,
            })

        rewritten_output_path = add_suffix(args.output_path, f"_rewritten{i+1}")
        with open(rewritten_output_path, "w", encoding="utf-8") as file:
            json.dump(attack_results, file, indent=2)


if __name__ == "__main__":
    main()