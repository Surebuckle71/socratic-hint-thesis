from dataclasses import dataclass

from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.state_labels import derive_state_labels
from socratic_hint.output_format import (
    format_completion,
    format_hint_only_completion,
    format_prompt,
)
from socratic_hint.types import StateEstimate


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    completion: str


def build_training_examples(example: MathDialExample) -> list[TrainingExample]:
    """Build TWO training examples per tutor turn: state-conditioned and
    state-suppressed.

    The evaluation harness runs a "state-suppressed" ablation
    (`suppress_state=True`) against the SAME fine-tuned weights as the
    state-conditioned condition. If training only ever showed the model
    state-conditioned prompts, that ablation would test a prompt format the
    model had never seen: it would either ignore the unfamiliar instruction and
    emit `State:`/`Hint:` anyway (making conditions 2 and 3 differ by sampling
    noise alone — a null result by construction), or half-follow it and produce
    out-of-distribution output whose lower quality says nothing about losing
    state conditioning.

    Emitting both variants makes `suppress_state` a learned behaviour, so the
    ablation isolates state conditioning rather than prompt-format novelty.
    The suppressed completion is hint-only — exactly the shape
    `parse_model_output(..., suppress_state=True)` expects.

    This roughly doubles the training set relative to the state-conditioned-only
    version; that is intended.
    """
    labels = derive_state_labels(example)
    training_examples: list[TrainingExample] = []
    for label in labels:
        history = example.turns[: label.turn_index]
        target_turn = example.turns[label.turn_index]

        training_examples.append(
            TrainingExample(
                prompt=format_prompt(history, example.problem),
                completion=format_completion(
                    StateEstimate(subskills=label.subskills), target_turn.text
                ),
            )
        )
        training_examples.append(
            TrainingExample(
                prompt=format_prompt(history, example.problem, suppress_state=True),
                completion=format_hint_only_completion(target_turn.text),
            )
        )
    return training_examples
