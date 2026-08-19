from dataclasses import dataclass

from socratic_hint.backends.base import HintBackend
from socratic_hint.types import DialogueTurn

# Note: this uses exact-text inequality as the "hints differ" signal, which is
# a conservative proxy — semantically-identical hints with different wording
# would count as "different". This limitation is accepted for scoping
# purposes (see docs/superpowers/specs/2026-08-18-system-architecture-design.md)
# rather than adding an LLM-judge semantic-difference check, which would add
# API cost/complexity not required to support the core claim.


@dataclass(frozen=True)
class StateAdaptivityPair:
    problem: str
    low_mastery_history: list[DialogueTurn]
    high_mastery_history: list[DialogueTurn]


@dataclass(frozen=True)
class StateAdaptivityResult:
    low_mastery_hint: str
    high_mastery_hint: str
    hints_differ: bool


class StateAdaptivityDiagnostic:
    def run(self, backend: HintBackend, pair: StateAdaptivityPair) -> StateAdaptivityResult:
        low_result = backend.infer_and_hint(pair.low_mastery_history, pair.problem)
        high_result = backend.infer_and_hint(pair.high_mastery_history, pair.problem)
        return StateAdaptivityResult(
            low_mastery_hint=low_result.hint,
            high_mastery_hint=high_result.hint,
            hints_differ=low_result.hint.strip() != high_result.hint.strip(),
        )

    def run_batch(self, backend: HintBackend, pairs: list[StateAdaptivityPair]) -> float:
        if not pairs:
            return 0.0
        results = [self.run(backend, pair) for pair in pairs]
        return sum(1 for r in results if r.hints_differ) / len(results)
