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


def test_parse_model_output_keeps_a_multi_line_hint_whole():
    """A hint spanning several lines must not be truncated to its first line.

    Generation allows thousands of tokens, so multi-line hints are realistic —
    and both the simulated student and the judge only ever see what this
    returns.
    """
    raw = (
        "State: arithmetic_execution=0.50\n"
        "Hint: Start by finding the total number of apples.\n"
        "Then think about how many are left after Sam takes three.\n"
        "What operation connects those two numbers?"
    )
    result = parse_model_output(raw)
    assert result.hint == (
        "Start by finding the total number of apples.\n"
        "Then think about how many are left after Sam takes three.\n"
        "What operation connects those two numbers?"
    )


def test_parse_model_output_multi_line_hint_when_state_suppressed():
    raw = "Hint: First line of the hint.\nSecond line of the hint."
    result = parse_model_output(raw, suppress_state=True)
    assert result.hint == "First line of the hint.\nSecond line of the hint."
    assert result.state is None


def test_parse_model_output_multi_line_hint_stops_at_a_later_field():
    """A recognised field prefix after the hint terminates the hint block."""
    raw = "Hint: Think about the total first.\nAnd then the remainder.\nState: foo=0.10"
    result = parse_model_output(raw, suppress_state=True)
    assert result.hint == "Think about the total first.\nAnd then the remainder."


def test_parse_model_output_single_line_hint_is_unchanged():
    raw = "State: arithmetic_execution=0.50\nHint: Just one line."
    assert parse_model_output(raw).hint == "Just one line."
