"""
Same-history counterfactual adaptivity diagnostic.

Addresses the residual confound in the original state-adaptivity diagnostic
(state_adaptivity.py / Section 3.7 of the thesis): there, the "low-mastery"
and "high-mastery" histories are two DIFFERENT portions of the same dialogue,
so a hint difference could come from the state prior changing OR from the
underlying dialogue content changing, since the two are never independent
when both come from naturally-occurring turns.

This script instead holds ONE dialogue history fixed per dialogue and
manually injects two different state vectors into the prompt (bypassing the
model's own self-estimation step entirely, via
FinetunedBackend.infer_hint_given_state). Only the state changes; the history
is byte-identical between the two generations.

It also logs PER-PAIR outcomes (not just aggregate differ-rate / noise-rate
counts, which is what scripts/significance_analysis.py had to work around
with an unpaired two-proportion z-test approximation). With per-pair data, a
proper paired McNemar test on "differs under different state" vs "differs
under same state" becomes possible.

No Anthropic API calls are made - this is pure local GPU generation, reusing
the already-fine-tuned checkpoint. Meant to be run manually:

    .venv313\\Scripts\\python.exe scripts/run_counterfactual_adaptivity.py \\
        checkpoints/run1 --limit 20 --results-dir results/counterfactual/
"""

import argparse
import json
import sys
from pathlib import Path

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.output_format import SUBSKILLS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import (  # noqa: E402
    build_adaptivity_pairs,
    filter_leaked_test_examples,
)

LOW_STATE = {s: 0.2 for s in SUBSKILLS}
HIGH_STATE = {s: 0.8 for s in SUBSKILLS}


def run_one(backend, problem: str, history) -> dict:
    hint_low = backend.infer_hint_given_state(history, problem, LOW_STATE)
    hint_high = backend.infer_hint_given_state(history, problem, HIGH_STATE)
    # Noise-floor control: repeat sample from the SAME injected state.
    hint_control = backend.infer_hint_given_state(history, problem, LOW_STATE)
    return {
        "hint_low": hint_low,
        "hint_high": hint_high,
        "hint_control": hint_control,
        "differs_between_states": hint_low.strip() != hint_high.strip(),
        "differs_noise_control": hint_low.strip() != hint_control.strip(),
    }


def mcnemar_paired(pairs: list[dict]) -> dict:
    """Paired McNemar table on the two binary outcomes actually logged per
    dialogue: 'differs when state changes' vs 'differs under same state'."""
    a = b = c = d = 0  # both-true, state-only, noise-only, both-false
    for p in pairs:
        s, n = p["differs_between_states"], p["differs_noise_control"]
        if s and n:
            a += 1
        elif s and not n:
            b += 1
        elif not s and n:
            c += 1
        else:
            d += 1
    n_discordant = b + c
    if n_discordant == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_discordant  # continuity-corrected
        # Two-sided p-value from chi-square(1) via the standard normal tail,
        # avoiding a scipy dependency: p = erfc(sqrt(chi2/2)).
        import math

        p_value = math.erfc((chi2 / 2) ** 0.5)
    return {"a": a, "b": b, "c": c, "d": d, "chi2": chi2, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Test examples to use before pair-building")
    parser.add_argument("--pair-limit", type=int, default=None, help="Cap on counterfactual pairs actually run")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    from socratic_hint.backends.finetuned import FinetunedBackend

    print("Loading MathDial test split...")
    test_examples = load_mathdial("test")
    train_qids = {ex.qid for ex in load_mathdial("train")}
    train_qids |= {ex.qid for ex in load_mathdial("validation")}
    test_examples = filter_leaked_test_examples(test_examples, train_qids)
    if args.limit is not None:
        test_examples = test_examples[: args.limit]

    pairs, skipped = build_adaptivity_pairs(test_examples)
    if args.pair_limit is not None:
        pairs = pairs[: args.pair_limit]
    print(f"{len(pairs)} counterfactual pairs to run ({skipped} dialogues skipped, same selection as the original diagnostic)")

    print(f"Loading fine-tuned checkpoint from {args.checkpoint_dir}...")
    backend = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)

    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.results_dir / "counterfactual_adaptivity.jsonl"
        out_file = open(out_path, "w", encoding="utf-8")
    else:
        out_file = None

    results = []
    failed = 0
    for i, pair in enumerate(pairs):
        try:
            r = run_one(backend, pair.problem, pair.low_mastery_history)
        except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the run
            failed += 1
            print(f"  [{i+1}/{len(pairs)}] FAILED: {exc}")
            continue
        results.append(r)
        if out_file is not None:
            out_file.write(json.dumps(r) + "\n")
            out_file.flush()
        print(f"  [{i+1}/{len(pairs)}] differs_between_states={r['differs_between_states']} differs_noise_control={r['differs_noise_control']}")

    if out_file is not None:
        out_file.close()

    n = len(results)
    if n == 0:
        print("No pairs succeeded; aborting.")
        return 1

    differ_rate = sum(r["differs_between_states"] for r in results) / n
    noise_rate = sum(r["differs_noise_control"] for r in results) / n
    net_rate = differ_rate - noise_rate
    mcnemar = mcnemar_paired(results)

    print(f"\n=== Counterfactual adaptivity results (n={n}, {failed} failed) ===")
    print(f"Differ rate (different injected state): {differ_rate:.4f}")
    print(f"Noise rate  (same injected state, repeat sample): {noise_rate:.4f}")
    print(f"Net rate: {net_rate:.4f}")
    print(f"Paired McNemar 2x2: a={mcnemar['a']} b={mcnemar['b']} c={mcnemar['c']} d={mcnemar['d']}")
    print(f"McNemar chi2={mcnemar['chi2']:.3f}, p={mcnemar['p_value']:.4f}")

    if args.results_dir is not None:
        summary_path = args.results_dir / "counterfactual_adaptivity_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n": n,
                    "failed": failed,
                    "differ_rate": differ_rate,
                    "noise_rate": noise_rate,
                    "net_rate": net_rate,
                    "mcnemar": mcnemar,
                },
                f,
                indent=2,
            )
        print(f"\nSummary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
