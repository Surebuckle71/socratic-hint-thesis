from socratic_hint.backends.base import HintBackend
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic, StateAdaptivityPair
from socratic_hint.types import DialogueTurn, HintResult


class HistorySensitiveBackend(HintBackend):
    """Returns a different hint depending on dialogue history length — stands
    in for a genuinely state-adaptive backend."""

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        hint = "Detailed hint." if len(dialogue_history) == 0 else "Brief nudge."
        return HintResult(hint=hint, state=None)


class ConstantBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="Same hint every time.", state=None)


def make_pair() -> StateAdaptivityPair:
    return StateAdaptivityPair(
        problem="Solve 2+2.",
        low_mastery_history=[],
        high_mastery_history=[
            DialogueTurn(speaker="tutor", text="prior hint"),
            DialogueTurn(speaker="student", text="prior reply"),
        ],
    )


def test_run_detects_differing_hints():
    diagnostic = StateAdaptivityDiagnostic()
    result = diagnostic.run(HistorySensitiveBackend(), make_pair())
    assert result.hints_differ is True
    assert result.low_mastery_hint == "Detailed hint."
    assert result.high_mastery_hint == "Brief nudge."


def test_run_detects_identical_hints():
    diagnostic = StateAdaptivityDiagnostic()
    result = diagnostic.run(ConstantBackend(), make_pair())
    assert result.hints_differ is False


def test_run_batch_computes_adaptivity_rate():
    diagnostic = StateAdaptivityDiagnostic()
    pairs = [make_pair(), make_pair()]
    rate = diagnostic.run_batch(HistorySensitiveBackend(), pairs)
    assert rate == 1.0

    rate_constant = diagnostic.run_batch(ConstantBackend(), pairs)
    assert rate_constant == 0.0
