import itertools

import pytest

from socratic_hint.backends.base import HintBackend
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic, StateAdaptivityPair
from socratic_hint.types import DialogueTurn, HintResult


class HistorySensitiveBackend(HintBackend):
    """Deterministic but genuinely history-sensitive — stands in for a truly
    state-adaptive backend. Signal high, noise floor zero."""

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        hint = "Detailed hint." if len(dialogue_history) == 0 else "Brief nudge."
        return HintResult(hint=hint, state=None)


class ConstantBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="Same hint every time.", state=None)


class StochasticNonAdaptiveBackend(HintBackend):
    """Ignores history entirely but returns a different hint every call — the
    exact failure mode the noise-floor control exists to cancel out. Cycles a
    fixed list rather than using real randomness so the test is deterministic."""

    def __init__(self):
        self._hints = itertools.cycle(["Hint A.", "Hint B.", "Hint C."])

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint=next(self._hints), state=None)


class RecordingBackend(HintBackend):
    def __init__(self):
        self.calls: list[dict] = []

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        self.calls.append({"history": list(dialogue_history), "suppress_state": suppress_state})
        return HintResult(hint="a hint", state=None)


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
    # Deterministic backend: repeating the same history changes nothing.
    assert result.noise_control_differ is False


def test_run_detects_identical_hints():
    diagnostic = StateAdaptivityDiagnostic()
    result = diagnostic.run(ConstantBackend(), make_pair())
    assert result.hints_differ is False
    assert result.noise_control_differ is False


def test_run_records_noise_control_for_stochastic_backend():
    diagnostic = StateAdaptivityDiagnostic()
    result = diagnostic.run(StochasticNonAdaptiveBackend(), make_pair())
    # Every generation differs, including the same-history repeat — so the
    # apparent "adaptivity" is entirely sampling noise.
    assert result.hints_differ is True
    assert result.noise_control_differ is True


def test_run_issues_three_generations_per_pair():
    backend = RecordingBackend()
    StateAdaptivityDiagnostic().run(backend, make_pair())
    assert len(backend.calls) == 3
    # Third call is the noise control: same history as the first.
    assert backend.calls[2]["history"] == backend.calls[0]["history"]


def test_run_batch_nets_out_sampling_noise():
    diagnostic = StateAdaptivityDiagnostic()
    pairs = [make_pair(), make_pair()]

    # Genuine adaptivity: differ_rate 1.0, noise floor 0.0.
    assert diagnostic.run_batch(HistorySensitiveBackend(), pairs) == 1.0

    # No variation at all: differ_rate 0.0, noise floor 0.0.
    assert diagnostic.run_batch(ConstantBackend(), pairs) == 0.0

    # Stochastic but non-adaptive: differ_rate 1.0, noise floor 1.0 — the
    # metric must report ~0, not the ~100% the old implementation gave.
    assert diagnostic.run_batch(StochasticNonAdaptiveBackend(), pairs) == 0.0


def test_run_batch_raises_on_empty_pairs():
    # Silently returning 0.0 would be indistinguishable from a measured
    # zero-adaptivity result.
    with pytest.raises(ValueError, match="must not be empty"):
        StateAdaptivityDiagnostic().run_batch(ConstantBackend(), [])


def test_run_batch_threads_suppress_state():
    backend = RecordingBackend()
    StateAdaptivityDiagnostic().run_batch(backend, [make_pair()], suppress_state=True)
    assert backend.calls
    assert all(call["suppress_state"] is True for call in backend.calls)


def test_run_batch_defaults_to_state_conditioning_on():
    backend = RecordingBackend()
    StateAdaptivityDiagnostic().run_batch(backend, [make_pair()])
    assert all(call["suppress_state"] is False for call in backend.calls)
