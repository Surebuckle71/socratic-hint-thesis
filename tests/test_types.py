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
