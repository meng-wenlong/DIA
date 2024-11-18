from dataclasses import dataclass, field
from transformers import AutoTokenizer
from typing import Type


@dataclass
class ModelTemplate:
    model_name_or_path: str = ""
    tokenizer: AutoTokenizer = field(init=False)
    Pu: str = "prefix user"
    Su: str = "suffix user"
    Pa: str = "prefix assistant"
    Sa: str = "suffix assistant"

    def __post_init__(self):
        if self.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            raise ValueError("model_name_or_path cannot be empty")


@dataclass
class Qwen2Template(ModelTemplate):
    model_name_or_path: str = "Qwen/Qwen2-7B-Instruct"
    Pu: str = "<|im_start|>user\n"
    Su: str = "<|im_end|>\n"
    Pa: str = "<|im_start|>assistant\n"
    Sa: str = "<|im_end|>\n"


@dataclass
class Qwen2_5Template(ModelTemplate):
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    Pu: str = "<|im_start|>user\n"
    Su: str = "<|im_end|>\n"
    Pa: str = "<|im_start|>assistant\n"
    Sa: str = "<|im_end|>\n"


@dataclass
class Llama3Template(ModelTemplate):
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    Pu: str = "<|start_header_id|>user<|end_header_id|>\n\n"
    Su: str = "<|eot_id|>"
    Pa: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    Sa: str = "<|eot_id|>"


@dataclass
class Gemma2Template(ModelTemplate):
    model_name_or_path: str = "google/gemma-2-9b-it"
    Pu: str = "<start_of_turn>user\n"
    Su: str = "<end_of_turn>\n"
    Pa: str = "<start_of_turn>model\n"
    Sa: str = "<end_of_turn>\n"


model2template: dict[str, Type[ModelTemplate]] = {
    "qwen2": Qwen2Template,
    "llama3": Llama3Template,
    "gemma2": Gemma2Template,
    "qwen2.5": Qwen2_5Template
}