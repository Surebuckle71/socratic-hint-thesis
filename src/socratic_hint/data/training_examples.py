from dataclasses import dataclass

from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.state_labels import derive_state_labels
from socratic_hint.output_format import format_completion, format_prompt
from socratic_hint.types import StateEstimate


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    completion: str


def build_training_examples(example: MathDialExample) -> list[TrainingExample]:
    labels = derive_state_labels(example)
    training_examples: list[TrainingExample] = []
    for label in labels:
        history = example.turns[: label.turn_index]
        target_turn = example.turns[label.turn_index]
        prompt = format_prompt(history, example.problem)
        completion = format_completion(StateEstimate(subskills=label.subskills), target_turn.text)
        training_examples.append(TrainingExample(prompt=prompt, completion=completion))
    return training_examples
