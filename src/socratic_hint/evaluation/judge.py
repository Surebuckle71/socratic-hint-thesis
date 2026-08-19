import json
import os
from dataclasses import dataclass

from anthropic import Anthropic

JUDGE_DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class JudgeScore:
    scores: dict[str, int]
    rationale: str


class PedagogicalQualityJudge:
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def score(self, problem: str, dialogue_context: str, hint: str) -> JudgeScore:
        prompt = (
            "You are scoring a tutoring hint on three dimensions, each an integer 1-5:\n"
            f"{', '.join(JUDGE_DIMENSIONS)}.\n\n"
            f"Problem: {problem}\nDialogue so far: {dialogue_context}\nHint given: {hint}\n\n"
            "Respond ONLY with JSON in exactly this shape: "
            '{"scaffolding_vs_telling": <1-5>, "correctness": <1-5>, '
            '"appropriateness": <1-5>, "rationale": "<one sentence>"}'
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        parsed = json.loads(raw)
        scores = {dim: int(parsed[dim]) for dim in JUDGE_DIMENSIONS}
        return JudgeScore(scores=scores, rationale=parsed["rationale"])
