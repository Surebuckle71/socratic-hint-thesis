from dataclasses import dataclass

from socratic_hint.backends.base import HintBackend
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.evaluation.judge import PedagogicalQualityJudge
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic, StateAdaptivityPair


@dataclass(frozen=True)
class ConditionResult:
    condition_name: str
    mean_judge_scores: dict[str, float]
    convergence_rate: float
    state_adaptivity_rate: float


def evaluate_condition(
    condition_name: str,
    backend: HintBackend,
    test_examples: list[MathDialExample],
    adaptivity_pairs: list[StateAdaptivityPair],
    judge: PedagogicalQualityJudge,
    student_evaluator: SimulatedStudentEvaluator,
    adaptivity_diagnostic: StateAdaptivityDiagnostic,
) -> ConditionResult:
    judge_scores: list[dict[str, int]] = []
    convergences: list[bool] = []

    for example in test_examples:
        hint_result = backend.infer_and_hint([], example.problem)
        score = judge.score(example.problem, "", hint_result.hint)
        judge_scores.append(score.scores)

        sim_result = student_evaluator.run(backend, example.problem, example.ground_truth)
        convergences.append(sim_result.converged)

    mean_scores: dict[str, float] = {}
    if judge_scores:
        for dim in judge_scores[0]:
            mean_scores[dim] = sum(s[dim] for s in judge_scores) / len(judge_scores)

    convergence_rate = sum(convergences) / len(convergences) if convergences else 0.0
    adaptivity_rate = adaptivity_diagnostic.run_batch(backend, adaptivity_pairs)

    return ConditionResult(
        condition_name=condition_name,
        mean_judge_scores=mean_scores,
        convergence_rate=convergence_rate,
        state_adaptivity_rate=adaptivity_rate,
    )
