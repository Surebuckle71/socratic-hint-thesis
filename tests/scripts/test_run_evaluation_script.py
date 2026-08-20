"""Tests for the evaluation entry point in `scripts/run_evaluation.py`.

`scripts/run_evaluation.py` imports `FinetunedBackend` (and therefore
`unsloth`) lazily inside `main()`, not at module level, specifically so this
module — and the pure functions it tests, `build_adaptivity_pairs` and
`filter_leaked_test_examples` — can be imported and tested on a base install
with no training stack present. Do not reintroduce a module-level
`FinetunedBackend`/`unsloth` import in the script without also re-adding an
`importorskip` guard here; that combination is what silently skipped this
whole file's coverage before.
"""

import importlib.util
import sys
from pathlib import Path

from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.types import DialogueTurn

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_evaluation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_run_evaluation_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def tutor(text: str, move: str) -> DialogueTurn:
    return DialogueTurn(speaker="tutor", text=text, move=move)


def student(text: str) -> DialogueTurn:
    return DialogueTurn(speaker="student", text=text)


def make_example(turns: list[DialogueTurn], qid: int = 1) -> MathDialExample:
    return MathDialExample(
        qid=qid,
        problem="Solve 2+2.",
        ground_truth="4",
        student_incorrect_solution="5",
        turns=turns,
    )


# --- build_adaptivity_pairs -------------------------------------------------


def test_adaptivity_pair_never_uses_the_opening_greeting():
    """Turn 0 is a `(generic)` greeting carrying the table's highest prior.

    Taking a plain argmax made turn 0 the "high mastery" pick in 84.4% of real
    test dialogues, i.e. an EMPTY history contrasted against a long one. It
    must never be selected now.
    """
    example = make_example([
        tutor("Hi there!", "generic"),          # 0: prior 0.7, highest in table
        student("hello"),                       # 1
        tutor("What's the total?", "telling"),  # 2: prior 0.2
        student("no idea"),                     # 3
        tutor("Which part is unclear?", "probing"),  # 4: prior 0.5
    ])
    pairs, skipped = script.build_adaptivity_pairs([example])

    assert skipped == 0
    assert len(pairs) == 1
    pair = pairs[0]
    # Neither history is empty, which is what selecting turn 0 would produce.
    assert pair.low_mastery_history
    assert pair.high_mastery_history
    # low = turn 2 (0.2), high = turn 4 (0.5).
    assert len(pair.low_mastery_history) == 2
    assert len(pair.high_mastery_history) == 4


def test_adaptivity_pair_histories_are_comparable_in_length():
    example = make_example([
        tutor("Hi there!", "generic"),
        student("hello"),
        tutor("Just tell me the total.", "telling"),   # 2: 0.2
        student("um"),
        tutor("What do we know?", "probing"),          # 4: 0.5
        student("not much"),
        tutor("Focus on the first step.", "focus"),    # 6: 0.5
    ])
    pairs, _ = script.build_adaptivity_pairs([example])

    assert len(pairs) == 1
    pair = pairs[0]
    diff = abs(len(pair.low_mastery_history) - len(pair.high_mastery_history))
    assert diff <= script.MAX_HISTORY_LENGTH_DIFF
    # The nearer of the two equally-high candidates (turn 4, not turn 6) wins.
    assert len(pair.high_mastery_history) == 4


def test_dialogue_with_no_valid_contrast_is_skipped_not_degenerate():
    """No fallback to a degenerate pair — the dialogue is dropped and counted."""
    example = make_example([
        tutor("Hi there!", "generic"),
        student("hello"),
        tutor("What do we know?", "probing"),   # 0.5
        student("not much"),
        tutor("Focus on step one.", "focus"),   # 0.5 — no contrast
    ])
    pairs, skipped = script.build_adaptivity_pairs([example])

    assert pairs == []
    assert skipped == 1


def test_dialogue_whose_only_contrast_is_too_far_apart_is_skipped():
    example = make_example([
        tutor("Hi there!", "generic"),
        student("hello"),
        tutor("Just tell me the total.", "telling"),  # 2: 0.2
        student("um"),
        student("still um"),
        student("really stuck"),
        tutor("What do we know?", "probing"),         # 6: 0.5 — 4 turns apart
    ])
    pairs, skipped = script.build_adaptivity_pairs([example])

    assert pairs == []
    assert skipped == 1


def test_dialogue_with_only_the_greeting_as_tutor_turn_is_skipped():
    example = make_example([
        tutor("Hi there!", "generic"),
        student("hello"),
        tutor("What's the total?", "telling"),
    ])
    # Only one non-opening tutor turn -> no pair possible.
    pairs, skipped = script.build_adaptivity_pairs([example])
    assert pairs == []
    assert skipped == 1


def test_build_adaptivity_pairs_counts_skips_across_dialogues():
    good = make_example([
        tutor("Hi!", "generic"),
        student("hello"),
        tutor("Just tell me.", "telling"),
        student("um"),
        tutor("What do we know?", "probing"),
    ], qid=1)
    bad = make_example([tutor("Hi!", "generic")], qid=2)

    pairs, skipped = script.build_adaptivity_pairs([good, bad, bad])
    assert len(pairs) == 1
    assert skipped == 2


def test_selected_histories_are_prefixes_of_the_dialogue():
    example = make_example([
        tutor("Hi!", "generic"),
        student("hello"),
        tutor("Just tell me.", "telling"),
        student("um"),
        tutor("What do we know?", "probing"),
    ])
    pairs, _ = script.build_adaptivity_pairs([example])
    pair = pairs[0]
    assert pair.low_mastery_history == example.turns[:2]
    assert pair.high_mastery_history == example.turns[:4]
    assert pair.problem == example.problem


# --- filter_leaked_test_examples --------------------------------------------


def test_leakage_filter_drops_test_examples_sharing_a_train_qid():
    test_examples = [make_example([], qid=q) for q in (1, 2, 3, 4)]
    kept = script.filter_leaked_test_examples(test_examples, train_qids={2, 4})
    assert [ex.qid for ex in kept] == [1, 3]


def test_leakage_filter_keeps_everything_when_no_overlap():
    test_examples = [make_example([], qid=q) for q in (10, 11)]
    kept = script.filter_leaked_test_examples(test_examples, train_qids={1, 2})
    assert [ex.qid for ex in kept] == [10, 11]


def test_leakage_filter_can_empty_the_test_set():
    test_examples = [make_example([], qid=1)]
    assert script.filter_leaked_test_examples(test_examples, train_qids={1}) == []
