from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.state_labels import (
    DEFAULT_PRIOR,
    MOVE_TO_MASTERY_PRIOR,
    derive_state_labels,
)
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


def test_derive_state_labels_defaults_unrecognised_move_to_default_prior():
    """A move with no entry in the prior table must fall back to DEFAULT_PRIOR.

    The previous version of this test used `move="focus"`, which IS a key in
    MOVE_TO_MASTERY_PRIOR — so it duplicated the known-move test above and
    never exercised the fallback branch at all.
    """
    move = "revealing_answer"
    assert move not in MOVE_TO_MASTERY_PRIOR  # guards against the old mistake

    example = make_example([
        DialogueTurn(speaker="tutor", text="hint", move=move),
    ])
    labels = derive_state_labels(example)
    assert labels[0].subskills == {skill: DEFAULT_PRIOR for skill in SUBSKILLS}


def test_derive_state_labels_defaults_missing_move_to_default_prior():
    """MathDial turns can carry no move tag at all; `move=None` must not crash."""
    example = make_example([
        DialogueTurn(speaker="tutor", text="hint", move=None),
    ])
    labels = derive_state_labels(example)
    assert labels[0].subskills == {skill: DEFAULT_PRIOR for skill in SUBSKILLS}


def test_derive_state_labels_empty_turns_returns_empty_list():
    example = make_example([])
    assert derive_state_labels(example) == []
