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
