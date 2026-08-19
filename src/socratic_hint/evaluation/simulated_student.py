import os
from dataclasses import dataclass

from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.types import DialogueTurn

DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class SimulationResult:
    converged: bool
    turns_taken: int
    transcript: list[DialogueTurn]


class SimulatedStudentEvaluator:
    def __init__(
        self,
        student_client: Anthropic | None = None,
        student_model: str = DEFAULT_MODEL,
        max_turns: int = 5,
    ):
        self.student_client = student_client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.student_model = student_model
        self.max_turns = max_turns

    def _simulate_student_reply(self, problem: str, transcript: list[DialogueTurn]) -> str:
        history_text = "\n".join(f"{t.speaker}: {t.text}" for t in transcript)
        prompt = (
            f"You are a student working on: {problem}\n"
            f"Conversation so far:\n{history_text}\n\n"
            "Reply as the student would, in 1-2 sentences, attempting to use the tutor's last hint."
        )
        response = self.student_client.messages.create(
            model=self.student_model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _is_correct(self, problem: str, ground_truth: str, student_reply: str) -> bool:
        prompt = (
            f"Problem: {problem}\nCorrect answer: {ground_truth}\nStudent's latest reply: {student_reply}\n\n"
            "Does the student's reply arrive at the correct final answer? Respond with exactly one word, YES or NO, with no other text, punctuation, or formatting."
        )
        response = self.student_client.messages.create(
            model=self.student_model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")

    def run(self, backend: HintBackend, problem: str, ground_truth: str) -> SimulationResult:
        transcript: list[DialogueTurn] = []
        for turn_number in range(1, self.max_turns + 1):
            hint_result = backend.infer_and_hint(transcript, problem)
            transcript.append(DialogueTurn(speaker="tutor", text=hint_result.hint))
            student_reply = self._simulate_student_reply(problem, transcript)
            transcript.append(DialogueTurn(speaker="student", text=student_reply))
            if self._is_correct(problem, ground_truth, student_reply):
                return SimulationResult(converged=True, turns_taken=turn_number, transcript=transcript)
        return SimulationResult(converged=False, turns_taken=self.max_turns, transcript=transcript)
