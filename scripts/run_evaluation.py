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

The test split is filtered for problem-level leakage before anything is
evaluated: MathDial's official split separates dialogues, not problems, so most
test problems were also seen during fine-tuning. See
`filter_leaked_test_examples`.
"""

import argparse
import sys
from pathlib import Path

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


# Maximum allowed difference in history length (in dialogue turns) between the
# low- and high-mastery members of an adaptivity pair.
#
# Why 2 and not 1: MathDial dialogues alternate strictly between teacher and
# student, so two DISTINCT tutor turns are at least two turns apart and a
# threshold of 1 is unsatisfiable in practice (measured: only 17 of 599 test
# dialogues admit any pair at all under `<= 1`, versus 489 under `<= 2`). A
# difference of 2 is one extra exchange (one tutor turn plus one student reply)
# — the tightest length match the corpus structure actually permits.
MAX_HISTORY_LENGTH_DIFF = 2


def build_adaptivity_pairs(
    examples: list[MathDialExample],
) -> tuple[list[StateAdaptivityPair], int]:
    """Contrast, within one dialogue, two histories that differ in inferred
    mastery while being as close as possible in length.

    Returns `(pairs, skipped_dialogues)`.

    Two constraints keep this measuring mastery rather than context volume:

    1. **Turn 0 is never a candidate.** MathDial dialogues open with a
       `(generic)` teacher greeting, and `generic` carries the highest prior in
       `MOVE_TO_MASTERY_PRIOR` (0.7). Taking the argmax over all tutor turns
       therefore selected turn 0 in 84.4% of test dialogues (482/571 pairs) —
       i.e. the "high mastery" history was usually EMPTY, contrasted against a
       "low mastery" history averaging ~5 turns. That diagnostic measured
       "hint with no context" vs "hint with lots of context", not mastery. A
       dialogue-opening greeting also carries no diagnostic signal about the
       student: no information has been exchanged yet that could justify
       calling the state "high mastery".

    2. **Both histories are non-empty and within `MAX_HISTORY_LENGTH_DIFF`
       turns of each other**, so a length confound cannot re-enter through a
       different mechanism. Among the candidate pairs that satisfy this, the
       one with the largest prior gap is chosen (ties broken toward the closer
       length, then the earlier turns, so the selection is deterministic).

    A dialogue admitting no such pair is skipped and counted rather than
    contributing a degenerate pair.

    Residual limitation: the two histories still differ in CONTENT, not only in
    inferred mastery, and the mastery priors themselves are silver labels
    derived from teacher move tags. The metric remains a directional
    diagnostic — but it is no longer dominated by a length artefact.
    """
    pairs: list[StateAdaptivityPair] = []
    skipped = 0

    # All subskills share one prior per turn, so any value identifies it.
    def prior(label) -> float:
        return next(iter(label.subskills.values()))

    for example in examples:
        # turn_index == 0 is the dialogue-opening greeting — see docstring.
        labels = [lab for lab in derive_state_labels(example) if lab.turn_index > 0]
        if len(labels) < 2:
            skipped += 1
            continue

        best = None  # (prior_gap, -length_diff, -low_index, -high_index, low, high)
        for low in labels:
            for high in labels:
                gap = prior(high) - prior(low)
                if gap <= 0:
                    continue
                length_diff = abs(high.turn_index - low.turn_index)
                if length_diff > MAX_HISTORY_LENGTH_DIFF:
                    continue
                key = (gap, -length_diff, -low.turn_index, -high.turn_index)
                if best is None or key > best[0]:
                    best = (key, low, high)

        if best is None:
            skipped += 1
            continue

        _, low, high = best
        pairs.append(
            StateAdaptivityPair(
                problem=example.problem,
                low_mastery_history=example.turns[: low.turn_index],
                high_mastery_history=example.turns[: high.turn_index],
            )
        )
    return pairs, skipped


def filter_leaked_test_examples(
    test_examples: list[MathDialExample], train_qids: set[int]
) -> list[MathDialExample]:
    """Drop test dialogues whose `qid` also appears in the training pool.

    MathDial's published train/test split is dialogue-level, not problem-level.
    `train_qids` here is the qid set of train + validation (both carved from
    HF's single published `train` split — see `load_mathdial`): 80.7% of test
    qids also occur in that pool, and this filter drops 358 of 599 test
    dialogues, leaving 241. A stricter, independent measure — test dialogues
    whose exact `(question, student_incorrect_solution)` pair is present in
    train ALONE (not train+validation) — finds 317 (52.9%), confirming the
    leakage is not just qid reuse with a cosmetically different problem
    statement. Evaluating the fine-tuned conditions on leaked problems would
    give them a systematic advantage in exactly the comparison the thesis
    rests on.
    """
    return [ex for ex in test_examples if ex.qid not in train_qids]


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
        "failed_pairs",
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
        row.append(str(result.failed_adaptivity_pairs))
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

    # Imported lazily (not at module level, and not at the top of main() —
    # after arg parsing, so `--help` and bad-argument errors exit before this
    # runs) so that this module — including the pure functions above
    # (`build_adaptivity_pairs`, `filter_leaked_test_examples`), which cover
    # two of the fixes this audit found — can be imported and tested without
    # unsloth installed. A module-level import here made the whole test file
    # depend on `pytest.importorskip("unsloth")`, silently skipping coverage
    # of those two functions on any environment without the training stack
    # (the base install this repo explicitly supports).
    from socratic_hint.backends.finetuned import FinetunedBackend

    print("Loading MathDial test split...")
    test_examples = load_mathdial("test")

    # Problem-level leakage filter. MathDial's official split separates
    # dialogues, not problems, so most test problems were also fine-tuned on.
    # Both the train and validation splits are carved from HF's `train` split
    # and both are consumed by run_training.py, so both are excluded here.
    print("Loading MathDial train/validation splits to filter problem-level leakage...")
    train_qids = {ex.qid for ex in load_mathdial("train")}
    train_qids |= {ex.qid for ex in load_mathdial("validation")}
    before = len(test_examples)
    test_examples = filter_leaked_test_examples(test_examples, train_qids)
    print(
        f"  Leakage filter: dropped {before - len(test_examples)} of {before} test "
        f"dialogues whose problem (qid) also appears in the training pool; "
        f"{len(test_examples)} held-out dialogues remain"
    )
    if not test_examples:
        print("ERROR: no test examples left after leakage filtering; aborting.", file=sys.stderr)
        return 1

    if args.limit is not None:
        test_examples = test_examples[: args.limit]
    adaptivity_pairs, skipped_dialogues = build_adaptivity_pairs(test_examples)
    print(f"  {len(test_examples)} test examples, {len(adaptivity_pairs)} adaptivity pairs")
    print(
        f"  {skipped_dialogues} dialogue(s) yielded no valid adaptivity pair "
        f"(no non-opening turn contrast within {MAX_HISTORY_LENGTH_DIFF} turns of length)"
    )

    # Construct every API-backed component BEFORE the multi-GB checkpoint load,
    # so a missing ANTHROPIC_API_KEY fails in seconds rather than after the GPU
    # load has already completed.
    judge = PedagogicalQualityJudge()
    student_evaluator = SimulatedStudentEvaluator(max_turns=args.max_turns)
    prompted = PromptedBackend()
    adaptivity_diagnostic = StateAdaptivityDiagnostic()

    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fine-tuned checkpoint from {args.checkpoint_dir}...")
    finetuned = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)

    conditions = [
        ("prompted_only", prompted, False),
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
        if result.failed_adaptivity_pairs:
            print(
                f"  WARNING: {result.failed_adaptivity_pairs} adaptivity pair(s) failed "
                "and were skipped"
            )

    print("\n" + format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
