import ollama
import copy
import logging
from openai import OpenAI
from anthropic import Anthropic
from abc import ABC, abstractmethod
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from di.prompts import DEFENSE_SYSTEM_PROMPT, DEFENSE_PROMPT_PATCH, PROMPT_CRITIQUE_PROMPT, CONSCIENCE_DISCLAIMER_PROMPT


logger = logging.getLogger(__name__)


class AbstractAttacker(ABC):
    def __init__(self, target_model):
        self.target_model = target_model
        self.openai_client = None
        self.anthropic_client = None
        if 'gpt-' in self.target_model:
            self.openai_client = OpenAI()
        elif 'claude-' in self.target_model:
            self.anthropic_client = Anthropic()

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=60))
    def get_response(
        self, 
        messages,
        prompt_defense_method: Optional[str] = None, # ['prompt_patch', 'system_prompt', 'bergeron', 'canonicalize']
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
            elif prompt_defense_method == 'canonicalize':
                combined_content = ''
                for msg in messages:
                    combined_content += msg['role'] + '\n' + msg['content'] + '\n\n'
                combined_content = combined_content.strip()
                processed_messages = [{'role': 'user', 'content': combined_content}]

            messages = processed_messages

        if self.openai_client:
            response = self.openai_client.chat.completions.create(
                model=self.target_model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        elif self.anthropic_client:
            if 'max_tokens' not in kwargs:
                kwargs['max_tokens'] = 1024
            kwargs.pop('seed', None)

            # Handle system messages
            anthropic_messages = copy.deepcopy(messages)
            if anthropic_messages[0]['role'] == 'system':
                kwargs['system'] = anthropic_messages[0]['content']
                anthropic_messages = anthropic_messages[1:]
            
            for msg in anthropic_messages:
                if msg['role'] == 'system':
                    msg['role'] = 'user'  # Anthropic does not have system role

            response = self.anthropic_client.messages.create(
                model=self.target_model,
                messages=anthropic_messages,
                **kwargs
            )
            return response.content[0].text

        response = ollama.chat(
            model=self.target_model,
            messages=messages,
            options=options or {"num_ctx": 8192, "num_predict": 1024},
        ) # type: ignore
        return response['message']['content']
    
    @abstractmethod
    def attack(self, harm_instructs: list[str]) -> list[str]:
        pass
        