# System Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core hint-generation system — a backend-agnostic module that infers student state from tutoring dialogue and generates a state-conditioned Socratic hint — with a prompted-API backend, a QLoRA fine-tuned local backend, and the 3-condition evaluation harness defined in the spec.

**Architecture:** One shared `HintBackend` interface implemented by two backends (prompted-API and fine-tuned-local) that both consume/produce the same structured types, so the evaluation harness and the fallback plan work identically regardless of which backend is active. Data pipeline turns MathDial into (dialogue-history, derived-state, hint) training triples; a QLoRA fine-tuning script trains the local backend's model on them.

**Tech Stack:** Python 3.11+, pytest, `anthropic` SDK (prompted backend + LLM-as-judge + simulated student, model id `claude-sonnet-5`), HuggingFace `datasets` (MathDial), `unsloth` + `peft` + `trl` + `transformers` + `bitsandbytes` (QLoRA fine-tuning on Qwen2.5-3B-Instruct).

**Spec:** `docs/superpowers/specs/2026-08-18-system-architecture-design.md`

## Global Constraints

- Base model for fine-tuning: `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` (Unsloth's Dynamic 4-bit quant — better accuracy than plain `bnb-4bit` for ~10% more VRAM; verified to exist on HuggingFace). Fall back to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` only if the dynamic variant OOMs on the 6GB card.
- Prompted-backend / judge / simulated-student model: `claude-sonnet-5` (Anthropic API), API key read from `ANTHROPIC_API_KEY` env var.
- MathDial dataset: HuggingFace `eth-nlped/mathdial` — verified schema: columns `qid`, `question`, `ground_truth`, `student_incorrect_solution`, `conversation`. Splits: `train` (2,264 rows), `test` (599 rows) — **no native validation split**; carve one from `train` (see Task 2).
- `conversation` field format (verified): turns joined by `|EOM|`, each turn `"Teacher: (move)text"` or `"Student: text"` (no move label on student turns).
- Derived student-state labels are silver/proxy labels, not ground truth — this is a documented thesis limitation, not a bug to "fix" during implementation.
- All external-API-calling code (Anthropic client, in Tasks 5, 8, 9) must accept an injectable client in `__init__` so tests can supply a fake/mock instead of hitting the network.

---

### Task 1: Core types and structured-output format

**Files:**
- Create: `pyproject.toml`
- Create: `src/socratic_hint/__init__.py`
- Create: `src/socratic_hint/types.py`
- Create: `src/socratic_hint/output_format.py`
- Create: `src/socratic_hint/backends/__init__.py`
- Create: `src/socratic_hint/backends/base.py`
- Test: `tests/test_types.py`
- Test: `tests/test_output_format.py`

**Interfaces:**
- Produces: `DialogueTurn(speaker: Literal["tutor","student"], text: str, move: str | None = None)`, `StateEstimate(subskills: dict[str, float])`, `HintResult(hint: str, state: StateEstimate | None)`, `SUBSKILLS: list[str]`, `format_prompt(dialogue_history: list[DialogueTurn], problem: str, suppress_state: bool = False) -> str`, `format_completion(state: StateEstimate, hint: str) -> str`, `parse_model_output(raw_output: str, suppress_state: bool = False) -> HintResult`, abstract `HintBackend.infer_and_hint(dialogue_history: list[DialogueTurn], problem: str, suppress_state: bool = False) -> HintResult`.

- [ ] **Step 1: Create the project scaffold**

`pyproject.toml`:
```toml
[project]
name = "socratic-hint"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "datasets>=3.0.0",
]

[project.optional-dependencies]
train = [
    "unsloth",
    "peft>=0.13.0",
    "trl>=0.12.0",
    "transformers>=4.46.0",
    "bitsandbytes>=0.44.0",
    "torch>=2.4.0",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["gpu: requires a local GPU and trained checkpoint"]
```

`src/socratic_hint/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test for types**

`tests/test_types.py`:
```python
from socratic_hint.types import DialogueTurn, StateEstimate, HintResult


def test_dialogue_turn_defaults_move_to_none():
    turn = DialogueTurn(speaker="student", text="I think it's 12.")
    assert turn.move is None
    assert turn.speaker == "student"


def test_dialogue_turn_tutor_with_move():
    turn = DialogueTurn(speaker="tutor", text="What's 4 times 3?", move="probing")
    assert turn.move == "probing"


def test_state_estimate_holds_subskill_mapping():
    state = StateEstimate(subskills={"arithmetic_execution": 0.7})
    assert state.subskills["arithmetic_execution"] == 0.7


def test_hint_result_state_can_be_none():
    result = HintResult(hint="Try breaking the problem into steps.", state=None)
    assert result.state is None
    assert "steps" in result.hint
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.types'`

- [ ] **Step 4: Implement types**

`src/socratic_hint/types.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Write the failing test for output_format**

`tests/test_output_format.py`:
```python
import pytest

from socratic_hint.output_format import (
    SUBSKILLS,
    format_completion,
    format_prompt,
    parse_model_output,
)
from socratic_hint.types import DialogueTurn, StateEstimate


def test_format_prompt_includes_problem_and_turns():
    history = [
        DialogueTurn(speaker="tutor", text="What's 4 times 3?", move="probing"),
        DialogueTurn(speaker="student", text="12"),
    ]
    prompt = format_prompt(history, "Solve 4 * 3 + 2.")
    assert "Solve 4 * 3 + 2." in prompt
    assert "What's 4 times 3?" in prompt
    assert "12" in prompt


def test_format_prompt_suppress_state_omits_state_instruction():
    prompt = format_prompt([], "problem", suppress_state=True)
    assert "State:" not in prompt
    assert "Hint:" in prompt


def test_format_prompt_default_asks_for_state_and_hint():
    prompt = format_prompt([], "problem", suppress_state=False)
    assert "State:" in prompt
    assert "Hint:" in prompt
    for skill in SUBSKILLS:
        assert skill in prompt


def test_format_completion_produces_state_and_hint_lines():
    state = StateEstimate(subskills={"arithmetic_execution": 0.65})
    completion = format_completion(state, "Try the next step.")
    assert "State:" in completion
    assert "arithmetic_execution=0.65" in completion
    assert "Hint: Try the next step." in completion


def test_parse_model_output_round_trips_format_completion():
    state = StateEstimate(subskills={"arithmetic_execution": 0.65, "step_sequencing": 0.40})
    raw = format_completion(state, "Try breaking it into two steps.")
    result = parse_model_output(raw)
    assert result.hint == "Try breaking it into two steps."
    assert result.state.subskills["arithmetic_execution"] == pytest.approx(0.65)
    assert result.state.subskills["step_sequencing"] == pytest.approx(0.40)


def test_parse_model_output_suppress_state_returns_none_state():
    raw = "Hint: Try breaking it into two steps."
    result = parse_model_output(raw, suppress_state=True)
    assert result.state is None
    assert result.hint == "Try breaking it into two steps."


def test_parse_model_output_raises_on_missing_fields():
    with pytest.raises(ValueError):
        parse_model_output("This is not formatted correctly at all.")
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/test_output_format.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.output_format'`

- [ ] **Step 8: Implement output_format**

`src/socratic_hint/output_format.py`:
```python
from socratic_hint.types import DialogueTurn, HintResult, StateEstimate

SUBSKILLS = [
    "problem_comprehension",
    "arithmetic_execution",
    "step_sequencing",
    "self_correction",
]


def format_prompt(
    dialogue_history: list[DialogueTurn], problem: str, suppress_state: bool = False
) -> str:
    lines = [f"Problem: {problem}", ""]
    for turn in dialogue_history:
        label = "Tutor" if turn.speaker == "tutor" else "Student"
        lines.append(f"{label}: {turn.text}")
    lines.append("")
    if suppress_state:
        lines.append("Respond with the tutor's next Socratic hint only. Do not include a state estimate.")
        lines.append("Format your response exactly as:")
        lines.append("Hint: <hint text>")
    else:
        lines.append(
            "First estimate the student's current mastery of each subskill "
            f"({', '.join(SUBSKILLS)}) as a probability between 0 and 1, "
            "then give the tutor's next Socratic hint, conditioned on that estimate."
        )
        lines.append("Format your response exactly as:")
        lines.append("State: <subskill1>=<0.xx>, <subskill2>=<0.xx>, ...")
        lines.append("Hint: <hint text>")
    return "\n".join(lines)


def format_completion(state: StateEstimate, hint: str) -> str:
    state_str = ", ".join(f"{k}={v:.2f}" for k, v in state.subskills.items())
    return f"State: {state_str}\nHint: {hint}"


def parse_model_output(raw_output: str, suppress_state: bool = False) -> HintResult:
    hint_line = None
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("hint:"):
            hint_line = stripped.split(":", 1)[1].strip()
            break

    if suppress_state:
        if hint_line is None:
            raise ValueError(f"Could not parse hint from model output: {raw_output!r}")
        return HintResult(hint=hint_line, state=None)

    state_line = None
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("state:"):
            state_line = stripped.split(":", 1)[1].strip()
            break

    if state_line is None or hint_line is None:
        raise ValueError(f"Could not parse state/hint from model output: {raw_output!r}")

    subskills: dict[str, float] = {}
    for pair in state_line.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        subskills[name.strip()] = float(value.strip())

    return HintResult(hint=hint_line, state=StateEstimate(subskills=subskills))
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_output_format.py -v`
Expected: PASS (7 tests)

- [ ] **Step 10: Add the HintBackend interface**

`src/socratic_hint/backends/__init__.py`: empty file.

`src/socratic_hint/backends/base.py`:
```python
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
```

No test file for this step — it's an abstract interface with no behavior of its own; Tasks 5 and 7 test their concrete implementations against it.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml src/socratic_hint/__init__.py src/socratic_hint/types.py src/socratic_hint/output_format.py src/socratic_hint/backends/ tests/test_types.py tests/test_output_format.py
git commit -m "Add core types, structured output format, and HintBackend interface"
```

---

### Task 2: MathDial data loader

**Files:**
- Create: `src/socratic_hint/data/__init__.py`
- Create: `src/socratic_hint/data/mathdial_loader.py`
- Test: `tests/data/__init__.py`
- Test: `tests/data/test_mathdial_loader.py`

**Interfaces:**
- Consumes: `DialogueTurn` from Task 1 (`socratic_hint.types`)
- Produces: `MathDialExample(qid: int, problem: str, ground_truth: str, student_incorrect_solution: str, turns: list[DialogueTurn])`, `parse_conversation(raw: str) -> list[DialogueTurn]`, `load_mathdial(split: Literal["train","validation","test"]) -> list[MathDialExample]`

- [ ] **Step 1: Write the failing test for conversation parsing**

`tests/data/__init__.py`: empty file.

`tests/data/test_mathdial_loader.py`:
```python
from unittest.mock import patch

from socratic_hint.data.mathdial_loader import (
    MathDialExample,
    load_mathdial,
    parse_conversation,
)


def test_parse_conversation_splits_on_eom_and_extracts_moves():
    raw = (
        "Teacher: (probing)Steven, If you had 4 of something and tripled that "
        "amount, how much would you have?|EOM|"
        "Student: I would have 12 of something.|EOM|"
        "Teacher: (generic)Exactly right!|EOM|"
    )
    turns = parse_conversation(raw)
    assert len(turns) == 3
    assert turns[0].speaker == "tutor"
    assert turns[0].move == "probing"
    assert turns[0].text == "Steven, If you had 4 of something and tripled that amount, how much would you have?"
    assert turns[1].speaker == "student"
    assert turns[1].move is None
    assert turns[1].text == "I would have 12 of something."
    assert turns[2].move == "generic"


def test_parse_conversation_ignores_empty_segments():
    raw = "Teacher: (telling)Look at the total again.|EOM||EOM|Student: Oh I see.|EOM|"
    turns = parse_conversation(raw)
    assert len(turns) == 2


FAKE_ROWS = [
    {
        "qid": 5000012,
        "question": "Nancy is filling an aquarium...",
        "ground_truth": "72 cubic feet",
        "student_incorrect_solution": "wrong answer text",
        "conversation": "Teacher: (probing)hint one|EOM|Student: reply one|EOM|",
    },
    {
        "qid": 5000084,
        "question": "John does push-ups...",
        "ground_truth": "9 weeks",
        "student_incorrect_solution": "wrong answer text 2",
        "conversation": "Teacher: (telling)hint two|EOM|Student: reply two|EOM|",
    },
]


def test_load_mathdial_test_split_maps_rows_to_examples():
    with patch("socratic_hint.data.mathdial_loader.load_dataset", return_value=FAKE_ROWS):
        examples = load_mathdial("test")
    assert len(examples) == 2
    assert isinstance(examples[0], MathDialExample)
    assert examples[0].qid == 5000012
    assert examples[0].problem == "Nancy is filling an aquarium..."
    assert len(examples[0].turns) == 2


def test_load_mathdial_train_and_validation_are_disjoint_and_cover_train_rows():
    many_rows = [dict(row, qid=row["qid"] + i) for i in range(10) for row in FAKE_ROWS]
    with patch("socratic_hint.data.mathdial_loader.load_dataset", return_value=many_rows):
        train = load_mathdial("train")
        validation = load_mathdial("validation")
    train_qids = {ex.qid for ex in train}
    val_qids = {ex.qid for ex in validation}
    assert train_qids.isdisjoint(val_qids)
    assert len(train_qids) + len(val_qids) == len(many_rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_mathdial_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.data'`

- [ ] **Step 3: Implement the loader**

`src/socratic_hint/data/__init__.py`: empty file.

`src/socratic_hint/data/mathdial_loader.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_mathdial_loader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/data/__init__.py src/socratic_hint/data/mathdial_loader.py tests/data/__init__.py tests/data/test_mathdial_loader.py
git commit -m "Add MathDial data loader with verified conversation-format parsing"
```

---

### Task 3: Derived student-state labels

**Files:**
- Create: `src/socratic_hint/data/state_labels.py`
- Test: `tests/data/test_state_labels.py`

**Interfaces:**
- Consumes: `MathDialExample`, `DialogueTurn` from Task 2; `SUBSKILLS` from Task 1 (`socratic_hint.output_format`)
- Produces: `DerivedStateLabel(turn_index: int, subskills: dict[str, float])`, `derive_state_labels(example: MathDialExample) -> list[DerivedStateLabel]`, `MOVE_TO_MASTERY_PRIOR: dict[str, float]`

- [ ] **Step 1: Write the failing test**

`tests/data/test_state_labels.py`:
```python
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.state_labels import MOVE_TO_MASTERY_PRIOR, derive_state_labels
from socratic_hint.output_format import SUBSKILLS
from socratic_hint.types import DialogueTurn


def make_example(turns: list[DialogueTurn]) -> MathDialExample:
    return MathDialExample(
        qid=1, problem="p", ground_truth="42",
        student_incorrect_solution="wrong", turns=turns,
    )


def test_derive_state_labels_only_labels_tutor_turns():
    example = make_example([
        DialogueTurn(speaker="tutor", text="hint", move="telling"),
        DialogueTurn(speaker="student", text="reply"),
        DialogueTurn(speaker="tutor", text="hint2", move="probing"),
    ])
    labels = derive_state_labels(example)
    assert len(labels) == 2
    assert labels[0].turn_index == 0
    assert labels[1].turn_index == 2


def test_derive_state_labels_uses_move_prior():
    example = make_example([
        DialogueTurn(speaker="tutor", text="hint", move="telling"),
    ])
    labels = derive_state_labels(example)
    expected = MOVE_TO_MASTERY_PRIOR["telling"]
    assert labels[0].subskills == {skill: expected for skill in SUBSKILLS}


def test_derive_state_labels_defaults_unknown_move_to_generic_prior():
    example = make_example([
        DialogueTurn(speaker="tutor", text="hint", move="focus"),
    ])
    labels = derive_state_labels(example)
    assert labels[0].subskills[SUBSKILLS[0]] == MOVE_TO_MASTERY_PRIOR["focus"]


def test_derive_state_labels_empty_turns_returns_empty_list():
    example = make_example([])
    assert derive_state_labels(example) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_state_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.data.state_labels'`

- [ ] **Step 3: Implement state label derivation**

`src/socratic_hint/data/state_labels.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_state_labels.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/data/state_labels.py tests/data/test_state_labels.py
git commit -m "Add derived student-state labels from tutor move priors"
```

---

### Task 4: Training example builder

**Files:**
- Create: `src/socratic_hint/data/training_examples.py`
- Test: `tests/data/test_training_examples.py`

**Interfaces:**
- Consumes: `MathDialExample` from Task 2; `derive_state_labels`, `DerivedStateLabel` from Task 3; `format_prompt`, `format_completion`, `parse_model_output` from Task 1
- Produces: `TrainingExample(prompt: str, completion: str)`, `build_training_examples(example: MathDialExample) -> list[TrainingExample]`

- [ ] **Step 1: Write the failing test**

`tests/data/test_training_examples.py`:
```python
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.training_examples import build_training_examples
from socratic_hint.output_format import parse_model_output
from socratic_hint.types import DialogueTurn


def test_build_training_examples_one_per_tutor_turn():
    example = MathDialExample(
        qid=1, problem="Solve 2+2.", ground_truth="4", student_incorrect_solution="5",
        turns=[
            DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing"),
            DialogueTurn(speaker="student", text="5"),
            DialogueTurn(speaker="tutor", text="Try counting on your fingers.", move="telling"),
        ],
    )
    training_examples = build_training_examples(example)
    assert len(training_examples) == 2


def test_build_training_examples_prompt_excludes_target_and_later_turns():
    example = MathDialExample(
        qid=1, problem="Solve 2+2.", ground_truth="4", student_incorrect_solution="5",
        turns=[
            DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing"),
            DialogueTurn(speaker="student", text="5"),
            DialogueTurn(speaker="tutor", text="Try counting on your fingers.", move="telling"),
        ],
    )
    training_examples = build_training_examples(example)
    first_prompt = training_examples[0].prompt
    assert "What's 2 plus 2?" not in first_prompt
    assert "Solve 2+2." in first_prompt


def test_build_training_examples_completion_is_parseable():
    example = MathDialExample(
        qid=1, problem="Solve 2+2.", ground_truth="4", student_incorrect_solution="5",
        turns=[
            DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing"),
        ],
    )
    training_examples = build_training_examples(example)
    result = parse_model_output(training_examples[0].completion)
    assert result.hint == "What's 2 plus 2?"
    assert result.state is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_training_examples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.data.training_examples'`

- [ ] **Step 3: Implement the builder**

`src/socratic_hint/data/training_examples.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_training_examples.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/data/training_examples.py tests/data/test_training_examples.py
git commit -m "Add training example builder producing (prompt, completion) pairs"
```

---

### Task 5: Prompted-API backend

**Files:**
- Create: `src/socratic_hint/backends/prompted.py`
- Test: `tests/backends/__init__.py`
- Test: `tests/backends/test_prompted.py`

**Interfaces:**
- Consumes: `HintBackend` from Task 1 (`socratic_hint.backends.base`); `format_prompt`, `parse_model_output` from Task 1 (`socratic_hint.output_format`); `DialogueTurn`, `HintResult` from Task 1 (`socratic_hint.types`)
- Produces: `PromptedBackend(client, model: str = "claude-sonnet-5")` implementing `HintBackend`

- [ ] **Step 1: Write the failing test**

`tests/backends/__init__.py`: empty file.

`tests/backends/test_prompted.py`:
```python
from unittest.mock import MagicMock

from socratic_hint.backends.prompted import PromptedBackend
from socratic_hint.types import DialogueTurn


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeTextBlock(text)]


def make_fake_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = FakeResponse(response_text)
    return client


def test_infer_and_hint_parses_state_and_hint_from_response():
    client = make_fake_client("State: arithmetic_execution=0.60\nHint: Try the next step.")
    backend = PromptedBackend(client=client, model="claude-sonnet-5")

    result = backend.infer_and_hint([], "Solve 2+2.")

    assert result.hint == "Try the next step."
    assert result.state.subskills["arithmetic_execution"] == 0.60
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert "Solve 2+2." in call_kwargs["messages"][0]["content"]


def test_infer_and_hint_suppress_state_passes_through():
    client = make_fake_client("Hint: Try the next step.")
    backend = PromptedBackend(client=client, model="claude-sonnet-5")

    result = backend.infer_and_hint(
        [DialogueTurn(speaker="tutor", text="hi")], "Solve 2+2.", suppress_state=True
    )

    assert result.state is None
    assert result.hint == "Try the next step."
    call_kwargs = client.messages.create.call_args.kwargs
    assert "State:" not in call_kwargs["messages"][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backends/test_prompted.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.backends.prompted'`

- [ ] **Step 3: Implement the prompted backend**

`src/socratic_hint/backends/prompted.py`:
```python
import os

from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.output_format import format_prompt, parse_model_output
from socratic_hint.types import DialogueTurn, HintResult

DEFAULT_MODEL = "claude-sonnet-5"


class PromptedBackend(HintBackend):
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        prompt = format_prompt(dialogue_history, problem, suppress_state=suppress_state)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        return parse_model_output(raw_text, suppress_state=suppress_state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backends/test_prompted.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/backends/prompted.py tests/backends/__init__.py tests/backends/test_prompted.py
git commit -m "Add prompted-API backend (also the fallback / baseline-1 system)"
```

---

### Task 6: QLoRA fine-tuning script

**Files:**
- Create: `src/socratic_hint/training/__init__.py`
- Create: `src/socratic_hint/training/train_qlora.py`
- Test: `tests/training/__init__.py`
- Test: `tests/training/test_train_qlora.py`

**Interfaces:**
- Consumes: `TrainingExample` from Task 4 (`socratic_hint.data.training_examples`)
- Produces: `TrainConfig(output_dir, max_steps, per_device_train_batch_size=2, learning_rate=2e-4, lora_r=16, lora_alpha=16, max_seq_length=2048)`, `build_lora_config(config: TrainConfig)`, `examples_to_hf_dataset(examples: list[TrainingExample])`, `train(examples: list[TrainingExample], config: TrainConfig) -> Path` — used by Task 7's `FinetunedBackend`, which loads whatever checkpoint directory `train()` returns.

**Before implementing:** this task uses fast-moving library APIs (Unsloth, PEFT, TRL) that may have changed since this plan was written. Step 1 verifies the current API against live docs before writing code that depends on it — do not skip this even if the code below looks plausible.

- [ ] **Step 1: Verify the current Unsloth QLoRA fine-tuning API**

Fetch Unsloth's official fine-tuning guide (e.g. via WebFetch on `https://docs.unsloth.ai/get-started/fine-tuning-guide` or the current equivalent — search if that URL has moved) and confirm: the exact signature of `FastLanguageModel.from_pretrained`, how LoRA adapters are attached (`FastLanguageModel.get_peft_model` and its parameters), and the current recommended trainer (`trl.SFTTrainer` or its replacement). If the API below has changed, update it to match what the docs show — the tests in Step 4-5 are what must pass, not the literal code in Step 2.

- [ ] **Step 2: Write the failing smoke test**

This is a real, fast (~1-2 minutes on the RTX 4050) integration test — not a placeholder — that trains for 2 steps on a 2-example toy dataset and checks the pipeline runs end-to-end without crashing. It requires the `train` optional dependency group and a local GPU, so it's marked `gpu` and skipped by default in fast test runs.

`tests/training/__init__.py`: empty file.

`tests/training/test_train_qlora.py`:
```python
import pytest

from socratic_hint.data.training_examples import TrainingExample
from socratic_hint.training.train_qlora import TrainConfig, train


@pytest.mark.gpu
def test_train_smoke_produces_checkpoint(tmp_path):
    examples = [
        TrainingExample(
            prompt="Problem: Solve 2+2.\n\nState: <s1>=<0.xx>\nHint:",
            completion="State: arithmetic_execution=0.50\nHint: What's 2 plus 2?",
        ),
        TrainingExample(
            prompt="Problem: Solve 3+3.\n\nState: <s1>=<0.xx>\nHint:",
            completion="State: arithmetic_execution=0.60\nHint: What's 3 plus 3?",
        ),
    ]
    config = TrainConfig(output_dir=tmp_path / "checkpoint", max_steps=2)

    output_dir = train(examples, config)

    assert output_dir == config.output_dir
    assert (output_dir / "adapter_config.json").exists() or (output_dir / "config.json").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/training/test_train_qlora.py -v -m gpu`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.training'`

- [ ] **Step 4: Implement the training script**

Install training deps first: `pip install -e ".[train]"`

`src/socratic_hint/training/__init__.py`: empty file.

`src/socratic_hint/training/train_qlora.py` — starting point; adjust to match whatever Step 1's doc check found:
```python
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from socratic_hint.data.training_examples import TrainingExample

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    max_steps: int
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 16
    max_seq_length: int = 2048


def examples_to_hf_dataset(examples: list[TrainingExample]) -> Dataset:
    return Dataset.from_dict(
        {"text": [f"{ex.prompt}\n\n{ex.completion}" for ex in examples]}
    )


def train(examples: list[TrainingExample], config: TrainConfig) -> Path:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    dataset = examples_to_hf_dataset(examples)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=TrainingArguments(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=config.per_device_train_batch_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        ),
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return config.output_dir
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/training/test_train_qlora.py -v -m gpu`
Expected: PASS (1 test, ~1-2 min). If it fails due to an API mismatch not caught in Step 1, re-check the current Unsloth docs and adjust — do not silently skip or weaken the test's assertions to make it pass.

- [ ] **Step 6: Commit**

```bash
git add src/socratic_hint/training/ tests/training/
git commit -m "Add QLoRA fine-tuning script for Qwen2.5-3B-Instruct with GPU smoke test"
```

---

### Task 7: Fine-tuned local backend

**Files:**
- Create: `src/socratic_hint/backends/finetuned.py`
- Test: `tests/backends/test_finetuned.py`

**Interfaces:**
- Consumes: `HintBackend` from Task 1; `format_prompt`, `parse_model_output` from Task 1; the checkpoint directory produced by Task 6's `train()`
- Produces: `FinetunedBackend(checkpoint_dir: Path, max_seq_length: int = 2048)` implementing `HintBackend`

- [ ] **Step 1: Write the failing tests — one fast (mocked), one GPU-marked (real checkpoint)**

`tests/backends/test_finetuned.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from socratic_hint.backends.finetuned import FinetunedBackend
from socratic_hint.types import DialogueTurn


def test_infer_and_hint_parses_generated_text(tmp_path):
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    fake_inputs = {"input_ids": MagicMock(shape=(1, 10))}
    fake_tokenizer.return_value.to.return_value = fake_inputs
    fake_model.generate.return_value = [MagicMock()]
    fake_tokenizer.decode.return_value = "State: arithmetic_execution=0.55\nHint: Try again."
    fake_model.device = "cpu"

    with patch("socratic_hint.backends.finetuned.FastLanguageModel") as fake_flm:
        fake_flm.from_pretrained.return_value = (fake_model, fake_tokenizer)
        backend = FinetunedBackend(checkpoint_dir=tmp_path)

    result = backend.infer_and_hint([], "Solve 2+2.")

    assert result.hint == "Try again."
    assert result.state.subskills["arithmetic_execution"] == 0.55


@pytest.mark.gpu
def test_infer_and_hint_against_real_smoke_checkpoint(tmp_path):
    """Requires Task 6's smoke test to have been run first, producing a
    checkpoint at the given path. Run manually with a real checkpoint dir:
    pytest tests/backends/test_finetuned.py -v -m gpu --checkpoint-dir=<path>
    This test intentionally has no fixture wiring here — see the task's
    manual-run note below."""
    pytest.skip("Run manually against a real checkpoint; see docstring.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backends/test_finetuned.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.backends.finetuned'`

- [ ] **Step 3: Implement the fine-tuned backend**

`src/socratic_hint/backends/finetuned.py`:
```python
from pathlib import Path

from unsloth import FastLanguageModel

from socratic_hint.backends.base import HintBackend
from socratic_hint.output_format import format_prompt, parse_model_output
from socratic_hint.types import DialogueTurn, HintResult


class FinetunedBackend(HintBackend):
    def __init__(self, checkpoint_dir: Path, max_seq_length: int = 2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        prompt = format_prompt(dialogue_history, problem, suppress_state=suppress_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        raw_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_model_output(raw_text, suppress_state=suppress_state)
```

Note: `FastLanguageModel.for_inference(self.model)` is called for side effect in `__init__` and is not separately mocked in the fast test above — the mocked `FastLanguageModel` patch covers it since it's an attribute of the same mock.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backends/test_finetuned.py -v` (this runs the fast mocked test only; the `gpu`-marked test is a documented manual step)
Expected: PASS (1 test passes, 1 skipped)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/backends/finetuned.py tests/backends/test_finetuned.py
git commit -m "Add fine-tuned local backend implementing HintBackend"
```

---

### Task 8: LLM-as-judge pedagogical quality scorer

**Files:**
- Create: `src/socratic_hint/evaluation/__init__.py`
- Create: `src/socratic_hint/evaluation/judge.py`
- Test: `tests/evaluation/__init__.py`
- Test: `tests/evaluation/test_judge.py`

**Interfaces:**
- Produces: `JUDGE_DIMENSIONS: list[str]`, `JudgeScore(scores: dict[str, int], rationale: str)`, `PedagogicalQualityJudge(client, model: str = "claude-sonnet-5").score(problem: str, dialogue_context: str, hint: str) -> JudgeScore`

- [ ] **Step 1: Write the failing test**

`src/socratic_hint/evaluation/__init__.py`: empty file.
`tests/evaluation/__init__.py`: empty file.

`tests/evaluation/test_judge.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from socratic_hint.evaluation.judge import JUDGE_DIMENSIONS, PedagogicalQualityJudge


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeTextBlock(text)]


def make_fake_client(payload: dict) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = FakeResponse(json.dumps(payload))
    return client


def test_score_parses_all_dimensions():
    payload = {
        "scaffolding_vs_telling": 4,
        "correctness": 5,
        "appropriateness": 3,
        "rationale": "Good scaffolding hint.",
    }
    client = make_fake_client(payload)
    judge = PedagogicalQualityJudge(client=client)

    score = judge.score("Solve 2+2.", "Tutor: hi", "Try again.")

    for dim in JUDGE_DIMENSIONS:
        assert score.scores[dim] == payload[dim]
    assert score.rationale == "Good scaffolding hint."


def test_score_raises_on_malformed_json():
    client = MagicMock()
    client.messages.create.return_value = FakeResponse("not json at all")
    judge = PedagogicalQualityJudge(client=client)

    with pytest.raises(json.JSONDecodeError):
        judge.score("problem", "context", "hint")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.evaluation'`

- [ ] **Step 3: Implement the judge**

`src/socratic_hint/evaluation/judge.py`:
```python
import json
import os
from dataclasses import dataclass

from anthropic import Anthropic

JUDGE_DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class JudgeScore:
    scores: dict[str, int]
    rationale: str


class PedagogicalQualityJudge:
    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def score(self, problem: str, dialogue_context: str, hint: str) -> JudgeScore:
        prompt = (
            "You are scoring a tutoring hint on three dimensions, each an integer 1-5:\n"
            f"{', '.join(JUDGE_DIMENSIONS)}.\n\n"
            f"Problem: {problem}\nDialogue so far: {dialogue_context}\nHint given: {hint}\n\n"
            "Respond ONLY with JSON in exactly this shape: "
            '{"scaffolding_vs_telling": <1-5>, "correctness": <1-5>, '
            '"appropriateness": <1-5>, "rationale": "<one sentence>"}'
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        parsed = json.loads(raw)
        scores = {dim: int(parsed[dim]) for dim in JUDGE_DIMENSIONS}
        return JudgeScore(scores=scores, rationale=parsed["rationale"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_judge.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/evaluation/__init__.py src/socratic_hint/evaluation/judge.py tests/evaluation/__init__.py tests/evaluation/test_judge.py
git commit -m "Add LLM-as-judge pedagogical quality scorer"
```

---

### Task 9: Simulated-student outcome-proxy evaluator

**Files:**
- Create: `src/socratic_hint/evaluation/simulated_student.py`
- Test: `tests/evaluation/test_simulated_student.py`

**Interfaces:**
- Consumes: `HintBackend` from Task 1; `DialogueTurn` from Task 1
- Produces: `SimulationResult(converged: bool, turns_taken: int, transcript: list[DialogueTurn])`, `SimulatedStudentEvaluator(student_client, student_model="claude-sonnet-5", max_turns=5).run(backend: HintBackend, problem: str, ground_truth: str) -> SimulationResult`

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_simulated_student.py`:
```python
from unittest.mock import MagicMock

from socratic_hint.backends.base import HintBackend
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.types import HintResult


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeTextBlock(text)]


class FixedHintBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="Try again.", state=None)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_simulated_student.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.evaluation.simulated_student'`

- [ ] **Step 3: Implement the evaluator**

`src/socratic_hint/evaluation/simulated_student.py`:
```python
import os
from dataclasses import dataclass

from anthropic import Anthropic

from socratic_hint.backends.base import HintBackend
from socratic_hint.types import DialogueTurn

DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class SimulationResult:
    converged: bool
    turns_taken: int
    transcript: list[DialogueTurn]


class SimulatedStudentEvaluator:
    def __init__(
        self,
        student_client: Anthropic | None = None,
        student_model: str = DEFAULT_MODEL,
        max_turns: int = 5,
    ):
        self.student_client = student_client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.student_model = student_model
        self.max_turns = max_turns

    def _simulate_student_reply(self, problem: str, transcript: list[DialogueTurn]) -> str:
        history_text = "\n".join(f"{t.speaker}: {t.text}" for t in transcript)
        prompt = (
            f"You are a student working on: {problem}\n"
            f"Conversation so far:\n{history_text}\n\n"
            "Reply as the student would, in 1-2 sentences, attempting to use the tutor's last hint."
        )
        response = self.student_client.messages.create(
            model=self.student_model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _is_correct(self, problem: str, ground_truth: str, student_reply: str) -> bool:
        prompt = (
            f"Problem: {problem}\nCorrect answer: {ground_truth}\nStudent's latest reply: {student_reply}\n\n"
            "Does the student's reply arrive at the correct final answer? Respond with exactly YES or NO."
        )
        response = self.student_client.messages.create(
            model=self.student_model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")

    def run(self, backend: HintBackend, problem: str, ground_truth: str) -> SimulationResult:
        transcript: list[DialogueTurn] = []
        for turn_number in range(1, self.max_turns + 1):
            hint_result = backend.infer_and_hint(transcript, problem)
            transcript.append(DialogueTurn(speaker="tutor", text=hint_result.hint))
            student_reply = self._simulate_student_reply(problem, transcript)
            transcript.append(DialogueTurn(speaker="student", text=student_reply))
            if self._is_correct(problem, ground_truth, student_reply):
                return SimulationResult(converged=True, turns_taken=turn_number, transcript=transcript)
        return SimulationResult(converged=False, turns_taken=self.max_turns, transcript=transcript)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_simulated_student.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/evaluation/simulated_student.py tests/evaluation/test_simulated_student.py
git commit -m "Add simulated-student outcome-proxy evaluator"
```

---

### Task 10: State-adaptivity diagnostic

**Files:**
- Create: `src/socratic_hint/evaluation/state_adaptivity.py`
- Test: `tests/evaluation/test_state_adaptivity.py`

**Interfaces:**
- Consumes: `HintBackend`, `DialogueTurn` from Task 1
- Produces: `StateAdaptivityPair(problem: str, low_mastery_history: list[DialogueTurn], high_mastery_history: list[DialogueTurn])`, `StateAdaptivityResult(low_mastery_hint: str, high_mastery_hint: str, hints_differ: bool)`, `StateAdaptivityDiagnostic().run(backend, pair) -> StateAdaptivityResult`, `.run_batch(backend, pairs: list[StateAdaptivityPair]) -> float`

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_state_adaptivity.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_state_adaptivity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.evaluation.state_adaptivity'`

- [ ] **Step 3: Implement the diagnostic**

`src/socratic_hint/evaluation/state_adaptivity.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_state_adaptivity.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/evaluation/state_adaptivity.py tests/evaluation/test_state_adaptivity.py
git commit -m "Add state-adaptivity diagnostic (the metric specific to this thesis's contribution)"
```

---

### Task 11: Evaluation orchestrator

**Files:**
- Create: `src/socratic_hint/evaluation/run_evaluation.py`
- Test: `tests/evaluation/test_run_evaluation.py`

**Interfaces:**
- Consumes: `HintBackend` from Task 1; `MathDialExample` from Task 2; `PedagogicalQualityJudge`, `JudgeScore` from Task 8; `SimulatedStudentEvaluator`, `SimulationResult` from Task 9; `StateAdaptivityDiagnostic`, `StateAdaptivityPair` from Task 10
- Produces: `ConditionResult(condition_name: str, mean_judge_scores: dict[str, float], convergence_rate: float, state_adaptivity_rate: float)`, `evaluate_condition(condition_name, backend, test_examples, adaptivity_pairs, judge, student_evaluator, adaptivity_diagnostic) -> ConditionResult`

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_run_evaluation.py`:
```python
from unittest.mock import MagicMock

from socratic_hint.backends.base import HintBackend
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.evaluation.judge import JudgeScore
from socratic_hint.evaluation.run_evaluation import evaluate_condition
from socratic_hint.evaluation.simulated_student import SimulationResult
from socratic_hint.types import HintResult


class FixedHintBackend(HintBackend):
    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        return HintResult(hint="a hint", state=None)


def make_example(qid: int) -> MathDialExample:
    return MathDialExample(
        qid=qid, problem="p", ground_truth="42", student_incorrect_solution="x", turns=[]
    )


def test_evaluate_condition_aggregates_all_three_metrics():
    judge = MagicMock()
    judge.score.side_effect = [
        JudgeScore(scores={"correctness": 4, "appropriateness": 2}, rationale="r1"),
        JudgeScore(scores={"correctness": 2, "appropriateness": 4}, rationale="r2"),
    ]

    student_evaluator = MagicMock()
    student_evaluator.run.side_effect = [
        SimulationResult(converged=True, turns_taken=1, transcript=[]),
        SimulationResult(converged=False, turns_taken=5, transcript=[]),
    ]

    adaptivity_diagnostic = MagicMock()
    adaptivity_diagnostic.run_batch.return_value = 0.75

    result = evaluate_condition(
        condition_name="fine_tuned_with_state",
        backend=FixedHintBackend(),
        test_examples=[make_example(1), make_example(2)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=adaptivity_diagnostic,
    )

    assert result.condition_name == "fine_tuned_with_state"
    assert result.mean_judge_scores["correctness"] == 3.0
    assert result.mean_judge_scores["appropriateness"] == 3.0
    assert result.convergence_rate == 0.5
    assert result.state_adaptivity_rate == 0.75


def test_evaluate_condition_handles_empty_test_examples():
    judge = MagicMock()
    student_evaluator = MagicMock()
    adaptivity_diagnostic = MagicMock()
    adaptivity_diagnostic.run_batch.return_value = 0.0

    result = evaluate_condition(
        condition_name="empty",
        backend=FixedHintBackend(),
        test_examples=[],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=adaptivity_diagnostic,
    )

    assert result.mean_judge_scores == {}
    assert result.convergence_rate == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_run_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'socratic_hint.evaluation.run_evaluation'`

- [ ] **Step 3: Implement the orchestrator**

`src/socratic_hint/evaluation/run_evaluation.py`:
```python
from dataclasses import dataclass

from socratic_hint.backends.base import HintBackend
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.evaluation.judge import PedagogicalQualityJudge
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic, StateAdaptivityPair


@dataclass(frozen=True)
class ConditionResult:
    condition_name: str
    mean_judge_scores: dict[str, float]
    convergence_rate: float
    state_adaptivity_rate: float


def evaluate_condition(
    condition_name: str,
    backend: HintBackend,
    test_examples: list[MathDialExample],
    adaptivity_pairs: list[StateAdaptivityPair],
    judge: PedagogicalQualityJudge,
    student_evaluator: SimulatedStudentEvaluator,
    adaptivity_diagnostic: StateAdaptivityDiagnostic,
) -> ConditionResult:
    judge_scores: list[dict[str, int]] = []
    convergences: list[bool] = []

    for example in test_examples:
        hint_result = backend.infer_and_hint([], example.problem)
        score = judge.score(example.problem, "", hint_result.hint)
        judge_scores.append(score.scores)

        sim_result = student_evaluator.run(backend, example.problem, example.ground_truth)
        convergences.append(sim_result.converged)

    mean_scores: dict[str, float] = {}
    if judge_scores:
        for dim in judge_scores[0]:
            mean_scores[dim] = sum(s[dim] for s in judge_scores) / len(judge_scores)

    convergence_rate = sum(convergences) / len(convergences) if convergences else 0.0
    adaptivity_rate = adaptivity_diagnostic.run_batch(backend, adaptivity_pairs)

    return ConditionResult(
        condition_name=condition_name,
        mean_judge_scores=mean_scores,
        convergence_rate=convergence_rate,
        state_adaptivity_rate=adaptivity_rate,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_run_evaluation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/socratic_hint/evaluation/run_evaluation.py tests/evaluation/test_run_evaluation.py
git commit -m "Add evaluation orchestrator tying backends, judge, simulator, and diagnostic together"
```

---

## After this plan

Once all 11 tasks pass review, the system is ready for two follow-up actions that are execution steps, not further TDD tasks:

1. **Run the real QLoRA fine-tune** on the full MathDial-derived training set (not just Task 6's 2-example smoke test) — a long-running (likely multi-hour) job on the RTX 4050, executed and monitored separately from this plan's task loop.
2. **Run the full 3-condition evaluation** from the spec, using the real fine-tuned checkpoint plus the prompted backend, across the MathDial test split — produces the results that go into the thesis's results chapter.

Both depend on this plan's code being correct and reviewed first.
