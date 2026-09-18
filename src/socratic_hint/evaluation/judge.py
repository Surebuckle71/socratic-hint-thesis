import json
from dataclasses import dataclass

from anthropic import Anthropic

from socratic_hint.llm_config import (
    DEFAULT_MODEL,
    GENERATION_MAX_TOKENS,
    extract_text,
    require_api_key,
    thinking_kwargs_for_model,
)

JUDGE_DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]

__all__ = ["DEFAULT_MODEL", "JUDGE_DIMENSIONS", "JudgeScore", "PedagogicalQualityJudge"]


@dataclass(frozen=True)
class JudgeScore:
    scores: dict[str, int]
    rationale: str


class PedagogicalQualityJudge:
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=require_api_key())
        self.model = model

    def score(self, problem: str, dialogue_context: str, hint: str) -> JudgeScore:
        prompt = (
            "You are scoring a tutoring hint on three dimensions, each an integer 1-5:\n"
            f"{', '.join(JUDGE_DIMENSIONS)}.\n\n"
            f"Problem: {problem}\nDialogue so far: {dialogue_context}\nHint given: {hint}\n\n"
            "Respond ONLY with JSON in exactly this shape: "
            '{"scaffolding_vs_telling": <1-5>, "correctness": <1-5>, '
            '"appropriateness": <1-5>, "rationale": "<one sentence>"}\n\n'
            "Output must start directly with `{` and end with `}` — no markdown code fences, "
            "no ```json``` blocks, no explanatory text before or after the JSON."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=GENERATION_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **thinking_kwargs_for_model(self.model),
        )
        raw = extract_text(response)
        parsed = json.loads(raw)
        scores = {dim: int(parsed[dim]) for dim in JUDGE_DIMENSIONS}
        return JudgeScore(scores=scores, rationale=parsed["rationale"])
