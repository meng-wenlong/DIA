import copy
import ollama
import common
import logging
from typing import Optional
from language_models import GPT, Ollama
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P
from di.prompts import DEFENSE_PROMPT_PATCH_LLAMA2_2, DEFENSE_SYSTEM_PROMPT, PROMPT_CRITIQUE_PROMPT, CONSCIENCE_DISCLAIMER_PROMPT


logger = logging.getLogger(__name__)


def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(
        model_name = args.attack_model, 
        max_n_tokens = args.attack_max_n_tokens, 
        max_n_attack_attempts = args.max_n_attack_attempts, 
        temperature = ATTACK_TEMP, # init to 1
        top_p = ATTACK_TOP_P, # init to 0.9
    )
    
    targetLM = TargetLM(
        model_name = args.target_model, 
        max_n_tokens = args.target_max_n_tokens,
        temperature = TARGET_TEMP, # init to 0
        top_p = TARGET_TOP_P, # init to 1
    )
    return attackLM, targetLM


class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self, 
        model_name: str, 
        max_n_tokens: int, 
        max_n_attack_attempts: int, 
        temperature: float,
        top_p: float
    ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model = load_indiv_model(model_name)

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0]) <= 1:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """Here are the improvement and prompt.\n{\"improvement\": \"""" 

        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append({"role": "user", "content": prompt})
            # Get prompts
            if "gpt-" in self.model_name:
                pass
            else:
                conv.append({"role": "assistant", "content": init_message})
            
        for attempt in range(self.max_n_attack_attempts):
            # Generate outputs 
            outputs_list = self.model.batched_generate(
                [convs_list[idx] for idx in indices_to_regenerate],
                max_n_tokens = self.max_n_tokens,  
                temperature = self.temperature,
                top_p = self.top_p
            )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt-" in self.model_name:
                    pass
                else:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index][-1]['content'] = json_str  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs


class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self, 
        model_name: str, 
        max_n_tokens: int, 
        temperature: float,
        top_p: float,
    ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.model = load_indiv_model(model_name)

    def get_response(
        self, 
        prompts_list, 
        prompt_defense_method: Optional[str] = None, # ['prompt_patch', 'system_prompt', 'bergeron']
        bergeron_secondary_model: Optional[str] = None,
    ):
        batchsize = len(prompts_list)

        # get system prompt
        system_prompt = "You are a helpfull assistant."
        if "gpt-" in self.model_name:
            pass
        else:
            model_info = ollama.show(self.model_name)
            if "system" in model_info:
                system_prompt = model_info["system"]

        convs_list = [[{"role": "system", "content": system_prompt}] for _ in range(batchsize)]
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append({"role": "user", "content": prompt})

        if prompt_defense_method is not None:
            processed_convs_list = copy.deepcopy(convs_list)
            if prompt_defense_method == 'prompt_patch':
                for i in range(batchsize):
                    processed_convs_list[i][-1]['content'] += '\n\n' + DEFENSE_PROMPT_PATCH_LLAMA2_2
            elif prompt_defense_method == 'system_prompt':
                # PAIR system prompt is for attack, we just append defense system prompt
                for i in range(batchsize):
                    processed_convs_list[i].insert(0, {'role': 'system', 'content': DEFENSE_SYSTEM_PROMPT})
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
                    critique_response = ollama.chat(
                        model=bergeron_secondary_model or self.model_name,
                        messages=critique_messages,
                    )['message']['content']
                    return critique_response if is_valid_critique(critique_response) else ""
                
                def make_conscience_prompt(prompt: str, prompt_critique: str):
                    return CONSCIENCE_DISCLAIMER_PROMPT.format(prompt_critique=prompt_critique, prompt=prompt)
                
                for i in range(batchsize):
                    input_critique = get_critique_response(processed_convs_list[i][-1]['content'])
                    if len(input_critique) > 0:
                        logger.debug("Generating conscience...")
                        sanitized = make_conscience_prompt(processed_convs_list[i][-1]['content'], input_critique)
                        processed_convs_list[i][-1]['content'] = sanitized

            convs_list = processed_convs_list
        
        outputs_list = self.model.batched_generate(
            convs_list,
            max_n_tokens = self.max_n_tokens,  
            temperature = self.temperature,
            top_p = self.top_p
        )
        return outputs_list


def load_indiv_model(model_name):
    if "gpt-" in model_name:
        lm = GPT(model_name)
    else:
        lm = Ollama(model_name)
    
    return lm
