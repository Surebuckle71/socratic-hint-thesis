from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.llm_config import (
    DEFAULT_MODEL,
    GENERATION_MAX_TOKENS,
    extract_text,
    require_api_key,
    thinking_kwargs_for_model,
)
from socratic_hint.output_format import format_prompt, parse_model_output
from socratic_hint.types import DialogueTurn, HintResult

__all__ = ["DEFAULT_MODEL", "PromptedBackend"]


class PromptedBackend(HintBackend):
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=require_api_key())
        self.model = model

    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        prompt = format_prompt(dialogue_history, problem, suppress_state=suppress_state)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=GENERATION_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **thinking_kwargs_for_model(self.model),
        )
        raw_text = extract_text(response)
        return parse_model_output(raw_text, suppress_state=suppress_state)
