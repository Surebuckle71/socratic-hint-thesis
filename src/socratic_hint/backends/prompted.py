import os

from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.output_format import format_prompt, parse_model_output
from socratic_hint.types import DialogueTurn, HintResult

DEFAULT_MODEL = "claude-sonnet-5"


class PromptedBackend(HintBackend):
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
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
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        return parse_model_output(raw_text, suppress_state=suppress_state)
