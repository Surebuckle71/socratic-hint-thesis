from abc import ABC, abstractmethod

from socratic_hint.types import DialogueTurn, HintResult


class HintBackend(ABC):
    @abstractmethod
    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        ...
