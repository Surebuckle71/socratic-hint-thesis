from unittest.mock import MagicMock

from socratic_hint.backends.base import HintBackend
import pytest

from socratic_hint.evaluation.simulated_student import (
    SimulatedStudentEvaluator,
    _reads_as_yes,
)
from socratic_hint.types import DialogueTurn, HintResult


class FakeThinkingBlock:
    """Mirrors a real ThinkingBlock: has `.thinking`, NOT `.text`."""

    def __init__(self, thinking: str = "reasoning..."):
        self.type = "thinking"
        self.thinking = thinking


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeResponse:
    """Adaptive thinking is on by default for claude-sonnet-5, so a real
    response leads with a ThinkingBlock — content[0] is not the answer."""

    def __init__(self, text: str):
        self.content = [FakeThinkingBlock(), FakeTextBlock(text)]


class FixedHintBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="Try again.", state=None)


class RecordingBackend(HintBackend):
    """Records the kwargs it was called with, for threading assertions."""

    def __init__(self):
        self.calls: list[dict] = []

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        self.calls.append(
            {
                "history": list(dialogue_history),
                "problem": problem,
                "suppress_state": suppress_state,
            }
        )
        return HintResult(hint="Try again.", state=None)


@pytest.mark.parametrize(
    "raw",
    [
        "YES",
        "yes",
        "  YES  ",
        "**YES**",
        '"YES"',
        "YES.",
        "- YES",
        "YES\nThe student computed 4.",
    ],
)
def test_reads_as_yes_tolerates_trivial_formatting(raw):
    """A strict startswith("YES") read every one of these as NO, which only
    ever depressed convergence_rate — a silent, one-directional undercount."""
    assert _reads_as_yes(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "NO",
        "no",
        "**NO**",
        "NO.",
        "The student did not reach the correct answer.",
        "",
    ],
)
def test_reads_as_yes_rejects_negatives(raw):
    assert _reads_as_yes(raw) is False


def test_reads_as_yes_ignores_yes_buried_deep_in_prose():
    """The window is short on purpose: a discursive answer that merely
    mentions "yes" later on is not a verdict."""
    raw = "The student made an arithmetic slip, so the answer is not yes at all."
    assert _reads_as_yes(raw) is False


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


def test_student_calls_request_thinking_with_adequate_budget():
    client = MagicMock()
    client.messages.create.side_effect = [FakeResponse("I think it's 4."), FakeResponse("YES")]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=1)

    evaluator.run(FixedHintBackend(), "Solve 2+2.", "4")

    # Both the reply generation and the YES/NO correctness check. The check
    # previously used max_tokens=8, which adaptive thinking would exhaust
    # before emitting any text — silently making _is_correct always False.
    assert client.messages.create.call_count == 2
    for call in client.messages.create.call_args_list:
        assert call.kwargs["thinking"] == {"type": "adaptive"}
        assert call.kwargs["output_config"] == {"effort": "low"}
        assert call.kwargs["max_tokens"] >= 1024


def test_run_threads_suppress_state_to_backend():
    client = MagicMock()
    client.messages.create.side_effect = [FakeResponse("I think it's 4."), FakeResponse("YES")]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=1)
    backend = RecordingBackend()

    evaluator.run(backend, "Solve 2+2.", "4", suppress_state=True)

    assert backend.calls
    assert all(call["suppress_state"] is True for call in backend.calls)


def test_run_defaults_to_state_conditioning_on():
    client = MagicMock()
    client.messages.create.side_effect = [FakeResponse("I think it's 4."), FakeResponse("YES")]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=1)
    backend = RecordingBackend()

    evaluator.run(backend, "Solve 2+2.", "4")

    assert all(call["suppress_state"] is False for call in backend.calls)


def test_run_seeds_transcript_with_initial_history():
    client = MagicMock()
    client.messages.create.side_effect = [FakeResponse("I think it's 4."), FakeResponse("YES")]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=1)
    backend = RecordingBackend()

    seed = [
        DialogueTurn(speaker="tutor", text="What do we know?"),
        DialogueTurn(speaker="student", text="Not much."),
    ]
    result = evaluator.run(backend, "Solve 2+2.", "4", initial_history=seed)

    # The first generation sees the real recorded context, not a blank history.
    assert backend.calls[0]["history"] == seed
    # ...and the returned transcript keeps the seed at the front.
    assert result.transcript[:2] == seed


def test_initial_history_defaults_to_empty_and_is_not_mutated():
    client = MagicMock()
    client.messages.create.side_effect = [FakeResponse("I think it's 4."), FakeResponse("YES")]
    evaluator = SimulatedStudentEvaluator(student_client=client, max_turns=1)
    backend = RecordingBackend()

    seed = [DialogueTurn(speaker="tutor", text="What do we know?")]
    evaluator.run(backend, "Solve 2+2.", "4", initial_history=seed)

    assert len(seed) == 1  # caller's list untouched
    assert backend.calls[0]["history"] == seed
