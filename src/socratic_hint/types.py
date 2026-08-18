from dataclasses import dataclass
from typing import Literal

Speaker = Literal["tutor", "student"]


@dataclass(frozen=True)
class DialogueTurn:
    speaker: Speaker
    text: str
    move: str | None = None


@dataclass(frozen=True)
class StateEstimate:
    subskills: dict[str, float]


@dataclass(frozen=True)
class HintResult:
    hint: str
    state: StateEstimate | None
