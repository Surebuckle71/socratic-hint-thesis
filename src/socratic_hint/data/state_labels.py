from dataclasses import dataclass

from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.output_format import SUBSKILLS

# Silver-label priors: a tutor's chosen pedagogical move is treated as a weak
# signal for the student's mastery state at that point in the dialogue (e.g. a
# "telling" move suggests the tutor judged scaffolding unlikely to work, i.e.
# low mastery). These are NOT verified ground-truth labels — documented as a
# limitation in the thesis (see docs/superpowers/specs/2026-08-18-system-architecture-design.md).
MOVE_TO_MASTERY_PRIOR: dict[str, float] = {
    "telling": 0.2,
    "probing": 0.5,
    "focus": 0.5,
    "generic": 0.7,
}

DEFAULT_PRIOR = 0.5


@dataclass(frozen=True)
class DerivedStateLabel:
    turn_index: int
    subskills: dict[str, float]


def derive_state_labels(example: MathDialExample) -> list[DerivedStateLabel]:
    labels: list[DerivedStateLabel] = []
    for i, turn in enumerate(example.turns):
        if turn.speaker != "tutor":
            continue
        prior = MOVE_TO_MASTERY_PRIOR.get(turn.move or "", DEFAULT_PRIOR)
        subskills = {skill: prior for skill in SUBSKILLS}
        labels.append(DerivedStateLabel(turn_index=i, subskills=subskills))
    return labels
