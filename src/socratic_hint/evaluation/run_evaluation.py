import json
from dataclasses import dataclass
from pathlib import Path

from socratic_hint.backends.base import HintBackend
from socratic_hint.data.mathdial_loader import MathDialExample
from socratic_hint.evaluation.judge import PedagogicalQualityJudge
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic, StateAdaptivityPair
from socratic_hint.types import DialogueTurn

# How many recorded MathDial turns to use as seed dialogue context. Hints are
# generated from, and judged against, this real conversational grounding
# rather than a blank history — which both matches the non-empty-history
# distribution the fine-tuned model was trained on and gives the judge
# something to score `appropriateness` against.
CONTEXT_TURNS = 2


@dataclass(frozen=True)
class ConditionResult:
    condition_name: str
    mean_judge_scores: dict[str, float]
    convergence_rate: float
    # None when no adaptivity pairs were supplied, or when every pair failed —
    # distinct from a measured 0.0.
    state_adaptivity_rate: float | None
    # Examples that raised and were skipped, so partial runs are never silent.
    failed_examples: int = 0
    # Adaptivity pairs that raised and were skipped, same rationale.
    failed_adaptivity_pairs: int = 0


def format_dialogue_context(turns: list[DialogueTurn]) -> str:
    """Render seed dialogue turns as the judge's `dialogue_context` string."""
    if not turns:
        return "(no prior dialogue)"
    labels = {"tutor": "Tutor", "student": "Student"}
    return "\n".join(f"{labels.get(t.speaker, t.speaker)}: {t.text}" for t in turns)


def evaluate_condition(
    condition_name: str,
    backend: HintBackend,
    test_examples: list[MathDialExample],
    adaptivity_pairs: list[StateAdaptivityPair],
    judge: PedagogicalQualityJudge,
    student_evaluator: SimulatedStudentEvaluator,
    adaptivity_diagnostic: StateAdaptivityDiagnostic,
    suppress_state: bool = False,
    results_path: Path | None = None,
) -> ConditionResult:
    """Evaluate one experimental condition across all three metrics.

    Args:
        suppress_state: forwarded to every backend generation. This is what
            makes the spec's "condition 2: fine-tuned model, state-conditioning
            suppressed" ablation runnable — the same weights, generating
            without state conditioning.
        results_path: optional JSONL file; truncated once at the start of the
            condition, then one line is appended per example as it completes,
            so a multi-hour run yields partial results incrementally instead of
            only at the end.

    A failure on one example (unparseable generation, malformed judge JSON) is
    recorded and skipped rather than aborting the run; the count is reported
    on the result as `failed_examples`.
    """
    # Truncate once, here — not per example. Appending without truncating means
    # re-running into the same --results-dir silently interleaves two runs'
    # records into one file with no way to tell them apart.
    if results_path is not None:
        results_path.write_text("", encoding="utf-8")

    judge_scores: list[dict[str, int]] = []
    convergences: list[bool] = []
    failed = 0

    for example in test_examples:
        try:
            seed_turns = example.turns[:CONTEXT_TURNS]

            # Run the simulation FIRST and score its actual opening hint. The
            # judge must see the hint the simulated student really received —
            # generating a separate one for scoring would double API cost and
            # score a different hint than the one that drove the dialogue.
            sim_result = student_evaluator.run(
                backend,
                example.problem,
                example.ground_truth,
                suppress_state=suppress_state,
                initial_history=seed_turns,
            )

            first_hint = _first_tutor_hint(sim_result.transcript, seed_turns)
            dialogue_context = format_dialogue_context(seed_turns)
            score = judge.score(example.problem, dialogue_context, first_hint)

            judge_scores.append(score.scores)
            convergences.append(sim_result.converged)

            _append_result(
                results_path,
                {
                    "condition": condition_name,
                    "qid": example.qid,
                    "suppress_state": suppress_state,
                    "converged": sim_result.converged,
                    "turns_taken": sim_result.turns_taken,
                    "first_hint": first_hint,
                    "judge_scores": score.scores,
                    "rationale": score.rationale,
                },
            )
        except Exception as exc:  # noqa: BLE001 - one bad example must not kill the run
            failed += 1
            _append_result(
                results_path,
                {
                    "condition": condition_name,
                    "qid": example.qid,
                    "suppress_state": suppress_state,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            continue

    mean_scores: dict[str, float] = {}
    if judge_scores:
        for dim in judge_scores[0]:
            mean_scores[dim] = sum(s[dim] for s in judge_scores) / len(judge_scores)

    convergence_rate = sum(convergences) / len(convergences) if convergences else 0.0

    # Empty pairs would make run_batch raise; report "not measured" instead of
    # inventing a real-looking 0.0.
    adaptivity_rate: float | None = None
    failed_pairs = 0
    if adaptivity_pairs:
        batch = adaptivity_diagnostic.run_batch(
            backend, adaptivity_pairs, suppress_state=suppress_state
        )
        adaptivity_rate = batch.net_rate
        failed_pairs = batch.failed_pairs
        _append_result(
            results_path,
            {
                "condition": condition_name,
                "record_type": "adaptivity_batch",
                "suppress_state": suppress_state,
                "net_rate": batch.net_rate,
                "differ_rate": batch.differ_rate,
                "noise_rate": batch.noise_rate,
                "evaluated_pairs": batch.evaluated_pairs,
                "failed_pairs": batch.failed_pairs,
            },
        )

    return ConditionResult(
        condition_name=condition_name,
        mean_judge_scores=mean_scores,
        convergence_rate=convergence_rate,
        state_adaptivity_rate=adaptivity_rate,
        failed_examples=failed,
        failed_adaptivity_pairs=failed_pairs,
    )


def _first_tutor_hint(transcript: list[DialogueTurn], seed_turns: list[DialogueTurn]) -> str:
    """The first hint the backend generated, i.e. the first tutor turn after the seed."""
    for turn in transcript[len(seed_turns):]:
        if turn.speaker == "tutor":
            return turn.text
    return ""


def _append_result(results_path: Path | None, record: dict) -> None:
    if results_path is None:
        return
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
