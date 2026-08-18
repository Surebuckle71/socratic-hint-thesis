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
