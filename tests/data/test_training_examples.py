from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.data.training_examples import build_training_examples
from socratic_hint.output_format import parse_model_output
from socratic_hint.types import DialogueTurn


def make_example(turns: list[DialogueTurn]) -> MathDialExample:
    return MathDialExample(
        qid=1,
        problem="Solve 2+2.",
        ground_truth="4",
        student_incorrect_solution="5",
        turns=turns,
    )


TWO_TUTOR_TURNS = [
    DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing"),
    DialogueTurn(speaker="student", text="5"),
    DialogueTurn(speaker="tutor", text="Try counting on your fingers.", move="telling"),
]


def test_build_training_examples_two_per_tutor_turn():
    """One state-conditioned variant plus one state-suppressed variant.

    The suppressed variant exists so the evaluation harness's
    `suppress_state=True` ablation tests a format the model was actually
    trained on, rather than an unseen prompt shape.
    """
    training_examples = build_training_examples(make_example(TWO_TUTOR_TURNS))
    assert len(training_examples) == 4


def test_build_training_examples_prompt_excludes_target_and_later_turns():
    training_examples = build_training_examples(make_example(TWO_TUTOR_TURNS))
    first_prompt = training_examples[0].prompt
    assert "What's 2 plus 2?" not in first_prompt
    assert "Solve 2+2." in first_prompt


def test_build_training_examples_completion_is_parseable():
    turns = [DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing")]
    training_examples = build_training_examples(make_example(turns))
    result = parse_model_output(training_examples[0].completion)
    assert result.hint == "What's 2 plus 2?"
    assert result.state is not None


def test_build_training_examples_emits_a_state_suppressed_variant():
    turns = [DialogueTurn(speaker="tutor", text="What's 2 plus 2?", move="probing")]
    training_examples = build_training_examples(make_example(turns))
    assert len(training_examples) == 2

    suppressed = training_examples[1]

    # The suppressed prompt must not ask for a state estimate at all.
    assert "State:" not in suppressed.prompt
    assert "Hint:" in suppressed.prompt

    # ...and its completion must be hint-only, parsing back exactly the way the
    # suppressed evaluation condition parses real generations.
    assert "State:" not in suppressed.completion
    result = parse_model_output(suppressed.completion, suppress_state=True)
    assert result.hint == "What's 2 plus 2?"
    assert result.state is None


def test_build_training_examples_variants_share_history_and_target():
    """Both variants of a turn differ only in state conditioning."""
    training_examples = build_training_examples(make_example(TWO_TUTOR_TURNS))
    conditioned, suppressed = training_examples[2], training_examples[3]

    # Same dialogue history in both prompts.
    assert "What's 2 plus 2?" in conditioned.prompt
    assert "What's 2 plus 2?" in suppressed.prompt

    # Same target hint in both completions.
    target = "Try counting on your fingers."
    assert parse_model_output(conditioned.completion).hint == target
    assert parse_model_output(suppressed.completion, suppress_state=True).hint == target


def test_build_training_examples_no_tutor_turns_yields_nothing():
    turns = [DialogueTurn(speaker="student", text="I'm stuck.")]
    assert build_training_examples(make_example(turns)) == []
