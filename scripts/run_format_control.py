"""
Format-control ablation: does Condition 2's quality gap against Condition 3
come from missing state information, or just from generating fewer tokens
before the hint?

Runs a new condition ("finetuned_format_control") using the same fine-tuned
weights, the same 241 leakage-filtered test dialogues, and the same
evaluation protocol (judge + simulated student) as scripts/run_evaluation.py,
but with FormatControlBackend: every hint is generated after a forced,
dialogue-independent `State: 0.50, 0.50, 0.50, 0.50` line, matching Condition
3's output length and format while carrying zero real information about the
conversation.

Makes real Anthropic API calls (judge + simulated student), same as
run_evaluation.py. Meant to be run manually:

    .venv313\\Scripts\\python.exe scripts/run_format_control.py checkpoints/run1 \\
        --results-dir results/full_run/
"""

import argparse
import sys
from pathlib import Path

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.evaluation.judge import PedagogicalQualityJudge
from socratic_hint.evaluation.run_evaluation import evaluate_condition
from socratic_hint.evaluation.simulated_student import SimulatedStudentEvaluator
from socratic_hint.evaluation.state_adaptivity import StateAdaptivityDiagnostic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import build_adaptivity_pairs, filter_leaked_test_examples  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adaptivity-limit", type=int, default=None)
    parser.add_argument("--student-model", default="claude-haiku-4-5")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    from socratic_hint.backends.finetuned import FinetunedBackend
    from socratic_hint.backends.format_control import FormatControlBackend

    print("Loading MathDial test split...")
    test_examples = load_mathdial("test")
    train_qids = {ex.qid for ex in load_mathdial("train")}
    train_qids |= {ex.qid for ex in load_mathdial("validation")}
    before = len(test_examples)
    test_examples = filter_leaked_test_examples(test_examples, train_qids)
    print(f"  Leakage filter: {before} -> {len(test_examples)} dialogues")

    if args.limit is not None:
        test_examples = test_examples[: args.limit]
    adaptivity_pairs, skipped = build_adaptivity_pairs(test_examples)
    if args.adaptivity_limit is not None:
        adaptivity_pairs = adaptivity_pairs[: args.adaptivity_limit]
    print(f"  {len(test_examples)} test examples, {len(adaptivity_pairs)} adaptivity pairs ({skipped} skipped)")

    judge = PedagogicalQualityJudge()
    student_evaluator = SimulatedStudentEvaluator(
        student_model=args.student_model, max_turns=args.max_turns
    )
    adaptivity_diagnostic = StateAdaptivityDiagnostic()

    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fine-tuned checkpoint from {args.checkpoint_dir}...")
    finetuned = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)
    backend = FormatControlBackend(finetuned)

    results_path = (
        args.results_dir / "finetuned_format_control.jsonl"
        if args.results_dir is not None
        else None
    )
    result = evaluate_condition(
        condition_name="finetuned_format_control",
        backend=backend,
        test_examples=test_examples,
        adaptivity_pairs=adaptivity_pairs,
        judge=judge,
        student_evaluator=student_evaluator,
        adaptivity_diagnostic=adaptivity_diagnostic,
        suppress_state=False,  # ignored by FormatControlBackend; kept for interface parity
        results_path=results_path,
    )

    print("\n=== Format-control ablation result ===")
    print(f"Mean judge scores: {result.mean_judge_scores}")
    print(f"Convergence rate: {result.convergence_rate:.4f}")
    print(f"State-adaptivity net rate: {result.state_adaptivity_rate}")
    print(f"Failed examples: {result.failed_examples}, failed adaptivity pairs: {result.failed_adaptivity_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
