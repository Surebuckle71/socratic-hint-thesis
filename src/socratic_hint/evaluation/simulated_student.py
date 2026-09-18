from dataclasses import dataclass

from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.llm_config import (
    DEFAULT_MODEL,
    MIN_MAX_TOKENS,
    extract_text,
    require_api_key,
    thinking_kwargs_for_model,
)
from socratic_hint.types import DialogueTurn

__all__ = ["DEFAULT_MODEL", "SimulationResult", "SimulatedStudentEvaluator"]


# How far into the reply to look for the YES verdict. The judge is asked for a
# bare YES/NO, but real replies arrive wrapped: `**YES**`, `"YES"`, `YES.`, a
# leading bullet. A strict `startswith("YES")` treats every one of those as NO,
# which depresses `convergence_rate` in one direction only — a silent,
# systematic under-count. A short window keeps that from happening while still
# refusing to find "yes" buried inside a paragraph of prose.
_YES_WINDOW_CHARS = 16


def _reads_as_yes(raw: str) -> bool:
    """True when the verdict text affirms, tolerating trivial formatting."""
    return "YES" in raw.strip().upper()[:_YES_WINDOW_CHARS]


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
        self.student_client = student_client or Anthropic(api_key=require_api_key())
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
            max_tokens=MIN_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **thinking_kwargs_for_model(self.student_model),
        )
        return extract_text(response).strip()

    def _is_correct(self, problem: str, ground_truth: str, student_reply: str) -> bool:
        prompt = (
            f"Problem: {problem}\nCorrect answer: {ground_truth}\nStudent's latest reply: {student_reply}\n\n"
            "Does the student's reply arrive at the correct final answer? Respond with exactly one word, YES or NO, with no other text, punctuation, or formatting."
        )
        response = self.student_client.messages.create(
            model=self.student_model,
            max_tokens=MIN_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **thinking_kwargs_for_model(self.student_model),
        )
        return _reads_as_yes(extract_text(response))

    def run(
        self,
        backend: HintBackend,
        problem: str,
        ground_truth: str,
        suppress_state: bool = False,
        initial_history: list[DialogueTurn] | None = None,
    ) -> SimulationResult:
        """Run a simulated tutoring dialogue until the student converges.

        `suppress_state` is forwarded to the backend on every tutor turn, so
        the ablation condition (fine-tuned model with state-conditioning
        suppressed) is simulated with the same weights but without state.

        `initial_history` seeds the transcript with real recorded dialogue
        (e.g. the opening turns of a MathDial conversation) so the first
        generated hint has genuine conversational grounding. It defaults to an
        empty history, preserving the original cold-start behaviour. The
        returned transcript includes the seed turns at the front.
        """
        transcript: list[DialogueTurn] = list(initial_history or [])
        for turn_number in range(1, self.max_turns + 1):
            hint_result = backend.infer_and_hint(
                transcript, problem, suppress_state=suppress_state
            )
            transcript.append(DialogueTurn(speaker="tutor", text=hint_result.hint))
            student_reply = self._simulate_student_reply(problem, transcript)
            transcript.append(DialogueTurn(speaker="student", text=student_reply))
            if self._is_correct(problem, ground_truth, student_reply):
                return SimulationResult(converged=True, turns_taken=turn_number, transcript=transcript)
        return SimulationResult(converged=False, turns_taken=self.max_turns, transcript=transcript)
