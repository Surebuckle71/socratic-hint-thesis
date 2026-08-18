from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset

from socratic_hint.types import DialogueTurn

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class MathDialExample:
    qid: int
    problem: str
    ground_truth: str
    student_incorrect_solution: str
    turns: list[DialogueTurn]


def parse_conversation(raw: str) -> list[DialogueTurn]:
    turns: list[DialogueTurn] = []
    for segment in raw.split("|EOM|"):
        segment = segment.strip()
        if not segment:
            continue
        speaker_part, _, rest = segment.partition(":")
        speaker_label = speaker_part.strip().lower()
        speaker = "tutor" if speaker_label == "teacher" else "student"
        rest = rest.strip()
        move = None
        if rest.startswith("(") and ")" in rest:
            move_part, _, text = rest.partition(")")
            move = move_part[1:].strip()
            rest = text.strip()
        turns.append(DialogueTurn(speaker=speaker, text=rest, move=move))
    return turns


def _to_example(row: dict) -> MathDialExample:
    return MathDialExample(
        qid=row["qid"],
        problem=row["question"],
        ground_truth=row["ground_truth"],
        student_incorrect_solution=row["student_incorrect_solution"],
        turns=parse_conversation(row["conversation"]),
    )


def load_mathdial(split: Split) -> list[MathDialExample]:
    hf_split = "test" if split == "test" else "train"
    rows = load_dataset("eth-nlped/mathdial", split=hf_split)
    examples = [_to_example(row) for row in rows]

    if split == "test":
        return examples

    # HF only publishes train/test; carve a dialogue-level 90/10 validation
    # split from train, sorted by qid for a deterministic, reproducible split.
    examples = sorted(examples, key=lambda ex: ex.qid)
    cutoff = int(len(examples) * 0.9)
    return examples[:cutoff] if split == "train" else examples[cutoff:]
