"""Run the 3-condition evaluation on the MathDial test split.

Meant to be run MANUALLY: it makes real Anthropic API calls (judge + simulated
student, so ANTHROPIC_API_KEY must be set) and loads a local fine-tuned
checkpoint on a CUDA GPU. It is deliberately not part of the automated test
suite — see README.md.

    .venv313\\Scripts\\python.exe scripts/run_evaluation.py checkpoints/run1 \\
        --limit 100 --results-dir results/

The three conditions are:
  1. prompted_only              - PromptedBackend (Anthropic API baseline)
  2. finetuned_without_state    - FinetunedBackend, suppress_state=True
  3. finetuned_with_state       - FinetunedBackend, suppress_state=False

Conditions 2 and 3 use the SAME weights; only state conditioning differs.
That contrast is the thesis's central claim.
"""

import argparse
from pathlib import Path

from socratic_hint.backends.finetuned import FinetunedBackend

from socratic_hint.backends.prompted import PromptedBackend
from socratic_hint.data.mathdial_loader import MathDialExample, load_mathdial
from socratic_hint.data.state_labels import derive_state_labels
from socratic_hint.evaluation.judge import PedagogicalQualityJudge
from socratic_hint.evaluation.run_evaluation import ConditionResult, evaluate_condition
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.evaluation.state_adaptivity import (
    StateAdaptivityDiagnostic,
    StateAdaptivityPair,
)


def build_adaptivity_pairs(examples: list[MathDialExample]) -> list[StateAdaptivityPair]:
    """Contrast, within one dialogue, the histories preceding the tutor's
    lowest- and highest-inferred-mastery turns.

    LIMITATION: the two histories differ in length and content, not only in
    inferred mastery, so a hint difference is not attributable to mastery
    alone. The noise-floor control in StateAdaptivityDiagnostic removes
    sampling noise but not this confound. Treat the resulting rate as a
    directional diagnostic, not a clean causal estimate.
    """
    pairs: list[StateAdaptivityPair] = []
    for example in examples:
        labels = derive_state_labels(example)
        if len(labels) < 2:
            continue
        # All subskills share one prior per turn, so any value identifies it.
        def prior(label):
            return next(iter(label.subskills.values()))

        low = min(labels, key=prior)
        high = max(labels, key=prior)
        if prior(low) >= prior(high):
            continue  # no contrast in this dialogue
        pairs.append(
            StateAdaptivityPair(
                problem=example.problem,
                low_mastery_history=example.turns[: low.turn_index],
                high_mastery_history=example.turns[: high.turn_index],
            )
        )
    return pairs


def format_results_table(results: list[ConditionResult]) -> str:
    dims: list[str] = []
    for result in results:
        for dim in result.mean_judge_scores:
            if dim not in dims:
                dims.append(dim)

    headers = ["condition"] + [f"judge:{d}" for d in dims] + [
        "convergence",
        "adaptivity(net)",
        "failed",
    ]
    rows = []
    for result in results:
        row = [result.condition_name]
        row += [f"{result.mean_judge_scores.get(d, float('nan')):.2f}" for d in dims]
        row.append(f"{result.convergence_rate:.2f}")
        row.append(
            "n/a" if result.state_adaptivity_rate is None
            else f"{result.state_adaptivity_rate:+.2f}"
        )
        row.append(str(result.failed_examples))
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Directory holding the fine-tuned LoRA adapter (from run_training.py).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test examples (useful for a cheap pilot run).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=5, help="Max simulated dialogue turns per example."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="If set, write per-example JSONL results here as the run progresses.",
    )
    args = parser.parse_args()

    print("Loading MathDial test split...")
    test_examples = load_mathdial("test")
    if args.limit is not None:
        test_examples = test_examples[: args.limit]
    adaptivity_pairs = build_adaptivity_pairs(test_examples)
    print(f"  {len(test_examples)} test examples, {len(adaptivity_pairs)} adaptivity pairs")

    judge = PedagogicalQualityJudge()
    student_evaluator = SimulatedStudentEvaluator(max_turns=args.max_turns)
    adaptivity_diagnostic = StateAdaptivityDiagnostic()

    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fine-tuned checkpoint from {args.checkpoint_dir}...")
    finetuned = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)

    conditions = [
        ("prompted_only", PromptedBackend(), False),
        ("finetuned_without_state", finetuned, True),
        ("finetuned_with_state", finetuned, False),
    ]

    results: list[ConditionResult] = []
    for name, backend, suppress_state in conditions:
        print(f"\n=== Condition: {name} (suppress_state={suppress_state}) ===")
        results_path = (
            args.results_dir / f"{name}.jsonl" if args.results_dir is not None else None
        )
        result = evaluate_condition(
            condition_name=name,
            backend=backend,
            test_examples=test_examples,
            adaptivity_pairs=adaptivity_pairs,
            judge=judge,
            student_evaluator=student_evaluator,
            adaptivity_diagnostic=adaptivity_diagnostic,
            suppress_state=suppress_state,
            results_path=results_path,
        )
        results.append(result)
        if result.failed_examples:
            print(f"  WARNING: {result.failed_examples} example(s) failed and were skipped")

    print("\n" + format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
