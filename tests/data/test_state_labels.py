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
