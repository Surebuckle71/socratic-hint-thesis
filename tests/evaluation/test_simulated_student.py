from unittest.mock import MagicMock

from socratic_hint.backends.base import HintBackend
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.types import HintResult


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeTextBlock(text)]


class FixedHintBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="Try again.", state=None)


def test_run_converges_when_student_reply_judged_correct():
    client = MagicMock()
    client.messages.create.side_effect = [
        FakeResponse("I think it's 4."),  # student reply
        FakeResponse("YES"),  # correctness judgment
    ]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=5)

    result = evaluator.run(FixedHintBackend(), "Solve 2+2.", "4")

    assert result.converged is True
    assert result.turns_taken == 1
    assert len(result.transcript) == 2


def test_run_stops_at_max_turns_when_never_correct():
    client = MagicMock()
    client.messages.create.side_effect = [
        FakeResponse("I don't know."), FakeResponse("NO"),
        FakeResponse("Still unsure."), FakeResponse("NO"),
    ]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=2)

    result = evaluator.run(FixedHintBackend(), "Solve 2+2.", "4")

    assert result.converged is False
    assert result.turns_taken == 2
