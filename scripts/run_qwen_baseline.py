"""
Qwen2.5-3B-Instruct prompted (non-fine-tuned) baseline - the missing piece for
RQ3. Runs two new conditions using QwenPromptedBackend:

  - qwen_prompted_no_state:   base model, state-suppressed prompt format
  - qwen_prompted_with_state: base model, state-conditioned prompt format

against the same 241 leakage-filtered test dialogues, same adaptivity pairs,
and same judge + simulated-student evaluation protocol as
scripts/run_evaluation.py. With this, the full 2x2 the professor asked for is
available: Qwen prompted/no-state, Qwen prompted/state, Qwen fine-tuned/
no-state (existing Condition 2), Qwen fine-tuned/state (existing Condition
3), plus Claude Sonnet 5 prompted as an external reference (existing
Condition 1).

Makes real Anthropic API calls (judge + simulated student). Meant to be run
manually:

    .venv313\\Scripts\\python.exe scripts/run_qwen_baseline.py \\
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adaptivity-limit", type=int, default=None)
    parser.add_argument("--student-model", default="claude-haiku-4-5")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    from socratic_hint.backends.qwen_prompted import QwenPromptedBackend

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

    print("Loading base Qwen2.5-3B-Instruct (no fine-tuning)...")
    qwen = QwenPromptedBackend()

    conditions = [
        ("qwen_prompted_no_state", True),
        ("qwen_prompted_with_state", False),
    ]

    for name, suppress_state in conditions:
        print(f"\n=== Condition: {name} (suppress_state={suppress_state}) ===")
        results_path = args.results_dir / f"{name}.jsonl" if args.results_dir is not None else None
        result = evaluate_condition(
            condition_name=name,
            backend=qwen,
            test_examples=test_examples,
            adaptivity_pairs=adaptivity_pairs,
            judge=judge,
            student_evaluator=student_evaluator,
            adaptivity_diagnostic=adaptivity_diagnostic,
            suppress_state=suppress_state,
            results_path=results_path,
        )
        print(f"Mean judge scores: {result.mean_judge_scores}")
        print(f"Convergence rate: {result.convergence_rate:.4f}")
        print(f"State-adaptivity net rate: {result.state_adaptivity_rate}")
        print(f"Failed examples: {result.failed_examples}, failed adaptivity pairs: {result.failed_adaptivity_pairs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
