import json
from unittest.mock import MagicMock

from socratic_hint.backends.base import HintBackend
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.evaluation.judge import JudgeScore
from socratic_hint.evaluation.run_evaluation import (
    CONTEXT_TURNS,
    evaluate_condition,
    format_dialogue_context,
)
from socratic_hint.evaluation.simulated_student import SimulationResult
from socratic_hint.types import DialogueTurn, HintResult


class FixedHintBackend(HintBackend):
    def __init__(self):
        self.call_count = 0

    def infer_and_hint(self, dialogue_history, problem, suppress_state=False):
        self.call_count += 1
        return HintResult(hint="a hint", state=None)


def make_example(qid: int, turns: list[DialogueTurn] | None = None) -> MathDialExample:
    return MathDialExample(
        qid=qid,
        problem="p",
        ground_truth="42",
        student_incorrect_solution="x",
        turns=turns if turns is not None else [],
    )


def make_transcript(hint: str = "a hint") -> list[DialogueTurn]:
    return [
        DialogueTurn(speaker="tutor", text=hint),
        DialogueTurn(speaker="student", text="ok"),
    ]


def test_evaluate_condition_aggregates_all_three_metrics():
    judge = MagicMock()
    judge.score.side_effect = [
        JudgeScore(scores={"correctness": 4, "appropriateness": 2}, rationale="r1"),
        JudgeScore(scores={"correctness": 2, "appropriateness": 4}, rationale="r2"),
    ]

    student_evaluator = MagicMock()
    student_evaluator.run.side_effect = [
        SimulationResult(converged=True, turns_taken=1, transcript=make_transcript()),
        SimulationResult(converged=False, turns_taken=5, transcript=make_transcript()),
    ]

    adaptivity_diagnostic = MagicMock()
    adaptivity_diagnostic.run_batch.return_value = 0.75

    result = evaluate_condition(
        condition_name="fine_tuned_with_state",
        backend=FixedHintBackend(),
        test_examples=[make_example(1), make_example(2)],
        adaptivity_pairs=[object()],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=adaptivity_diagnostic,
    )

    assert result.condition_name == "fine_tuned_with_state"
    assert result.mean_judge_scores["correctness"] == 3.0
    assert result.mean_judge_scores["appropriateness"] == 3.0
    assert result.convergence_rate == 0.5
    assert result.state_adaptivity_rate == 0.75
    assert result.failed_examples == 0


def test_evaluate_condition_scores_the_simulations_own_first_hint():
    """The judge must see the hint the simulated student actually received —
    not a second, independently generated one."""
    judge = MagicMock()
    judge.score.return_value = JudgeScore(scores={"correctness": 5}, rationale="r")

    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=make_transcript("the real first hint")
    )

    backend = FixedHintBackend()
    evaluate_condition(
        condition_name="c",
        backend=backend,
        test_examples=[make_example(1)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=MagicMock(),
    )

    # No extra generation outside the simulation.
    assert backend.call_count == 0
    assert judge.score.call_args.args[2] == "the real first hint"


def test_evaluate_condition_gives_judge_real_dialogue_context():
    judge = MagicMock()
    judge.score.return_value = JudgeScore(scores={"correctness": 5}, rationale="r")

    turns = [
        DialogueTurn(speaker="tutor", text="What do we know?"),
        DialogueTurn(speaker="student", text="Not much."),
        DialogueTurn(speaker="tutor", text="later turn, beyond the seed"),
    ]
    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=turns[:CONTEXT_TURNS] + make_transcript()
    )

    evaluate_condition(
        condition_name="c",
        backend=FixedHintBackend(),
        test_examples=[make_example(1, turns=turns)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=MagicMock(),
    )

    context = judge.score.call_args.args[1]
    assert context != ""
    assert "What do we know?" in context
    assert "Not much." in context
    # Only the first CONTEXT_TURNS turns are used as seed context.
    assert "beyond the seed" not in context

    # The simulator was seeded with the same turns the judge sees.
    assert student_evaluator.run.call_args.kwargs["initial_history"] == turns[:CONTEXT_TURNS]


def test_evaluate_condition_threads_suppress_state():
    judge = MagicMock()
    judge.score.return_value = JudgeScore(scores={"correctness": 5}, rationale="r")
    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=make_transcript()
    )
    adaptivity_diagnostic = MagicMock()
    adaptivity_diagnostic.run_batch.return_value = 0.1

    evaluate_condition(
        condition_name="fine_tuned_without_state",
        backend=FixedHintBackend(),
        test_examples=[make_example(1)],
        adaptivity_pairs=[object()],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=adaptivity_diagnostic,
        suppress_state=True,
    )

    assert student_evaluator.run.call_args.kwargs["suppress_state"] is True
    assert adaptivity_diagnostic.run_batch.call_args.kwargs["suppress_state"] is True


def test_evaluate_condition_defaults_suppress_state_false():
    judge = MagicMock()
    judge.score.return_value = JudgeScore(scores={"correctness": 5}, rationale="r")
    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=make_transcript()
    )

    evaluate_condition(
        condition_name="c",
        backend=FixedHintBackend(),
        test_examples=[make_example(1)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=MagicMock(),
    )

    assert student_evaluator.run.call_args.kwargs["suppress_state"] is False


def test_evaluate_condition_continues_after_a_failing_example():
    judge = MagicMock()
    judge.score.side_effect = [
        JudgeScore(scores={"correctness": 4}, rationale="r1"),
        ValueError("malformed judge JSON"),
        JudgeScore(scores={"correctness": 2}, rationale="r3"),
    ]
    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=make_transcript()
    )

    result = evaluate_condition(
        condition_name="c",
        backend=FixedHintBackend(),
        test_examples=[make_example(1), make_example(2), make_example(3)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=MagicMock(),
    )

    # The run completed; only the bad example was dropped, and it is reported.
    assert result.failed_examples == 1
    assert result.mean_judge_scores["correctness"] == 3.0
    assert result.convergence_rate == 1.0


def test_evaluate_condition_writes_incremental_results(tmp_path):
    judge = MagicMock()
    judge.score.side_effect = [
        JudgeScore(scores={"correctness": 4}, rationale="r1"),
        RuntimeError("boom"),
    ]
    student_evaluator = MagicMock()
    student_evaluator.run.return_value = SimulationResult(
        converged=True, turns_taken=1, transcript=make_transcript()
    )

    results_path = tmp_path / "results.jsonl"
    evaluate_condition(
        condition_name="c",
        backend=FixedHintBackend(),
        test_examples=[make_example(1), make_example(2)],
        adaptivity_pairs=[],
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=MagicMock(),
        results_path=results_path,
    )

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["qid"] == 1
    assert records[0]["judge_scores"] == {"correctness": 4}
    # Failures are persisted too, so partial output never hides data loss.
    assert records[1]["qid"] == 2
    assert "RuntimeError" in records[1]["error"]


def test_evaluate_condition_reports_none_adaptivity_when_no_pairs():
    judge = MagicMock()
    student_evaluator = MagicMock()
    adaptivity_diagnostic = MagicMock()

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
    # Not measured — distinct from a measured 0.0.
    assert result.state_adaptivity_rate is None
    adaptivity_diagnostic.run_batch.assert_not_called()


def test_format_dialogue_context_renders_turns():
    turns = [
        DialogueTurn(speaker="tutor", text="What do we know?"),
        DialogueTurn(speaker="student", text="Not much."),
    ]
    assert format_dialogue_context(turns) == "Tutor: What do we know?\nStudent: Not much."


def test_format_dialogue_context_handles_no_turns():
    assert format_dialogue_context([]) == "(no prior dialogue)"
