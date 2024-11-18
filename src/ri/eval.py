import copy
import logging
import ollama
from typing import Optional, Any, Sequence
from tqdm import tqdm

from jailbreakeval import JailbreakEvaluator
from jailbreakeval.services.chat.service_base import ChatService
from jailbreakeval.services.chat.types import Message


logger = logging.getLogger(__name__)


def eval_batch(
    evaluator: JailbreakEvaluator,
    questions: list[str],
    answers: list[str],
) -> list[bool]:
    results = []
    assert len(questions) == len(answers)

    for question, answer in tqdm(zip(questions, answers)):
        attempt_to_evaluate = {
            "question": question,
            "answer": answer,
        }
        evaluation_result = evaluator(attempt_to_evaluate)
        results.append(evaluation_result)

    return results


class OllamaChatService(ChatService, service_type="ollama"): # type:ignore
    def __init__(
        self,
        model: Optional[str] = None,
        chat_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(model, chat_kwargs=chat_kwargs)

    def chat(self, conversation: Sequence[Message], **override_chat_kwargs):
        chat_kwargs = copy.deepcopy(self.chat_kwargs)
        for k, v in override_chat_kwargs.items():
            chat_kwargs[k] = v
        response = ollama.chat(
            model=self.model,
            messages=conversation,
            options=chat_kwargs,
        )
        return response["message"]["content"]