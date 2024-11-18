import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import List, Optional


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    

class Ollama(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
    
    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        outputs_list = []
        for full_prompts in full_prompts_list:
            response = ollama.chat(
                model=self.model_name,
                messages=full_prompts,
                options={
                    "num_predict": max_n_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            ) # type: ignore
            outputs_list.append(response['message']['content'])
        
        return outputs_list
    

class GPT(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI()

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=60))
    def generate(
        self,
        conv: list[dict],
        max_n_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=conv,
            max_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content
    
    def batched_generate(
        self,
        convs_list: list[list[dict[str, str]]],
        max_n_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        return [
            self.generate(
                conv=convs,
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for convs in convs_list
        ]