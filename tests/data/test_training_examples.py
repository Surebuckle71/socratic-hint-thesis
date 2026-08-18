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
