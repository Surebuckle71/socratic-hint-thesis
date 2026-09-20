"""
Placebo version of the same-history counterfactual diagnostic.

The counterfactual compares hints generated after an injected State: line of 0.20 for every
subskill with hints after 0.80 for every subskill. Any edit to that line might change the
hint, so the excess over resampling does not show that the meaning of the state matters.
This script repeats the design with a PLACEBO pair, 0.48 versus 0.52 for every subskill, two
states that carry almost the same meaning but are different text. The state effect is then
the excess of the 0.20-vs-0.80 result over the placebo excess.

For each of the same 191 pairs it generates, from the same fixed history:
  sampled:  hint_a (0.48), hint_b (0.52), hint_control (0.48 again)
  greedy:   hint_a (0.48), hint_b (0.52), hint_control (0.48 again)

No API calls; local GPU only.

    .venv313\\Scripts\\python.exe scripts/run_placebo_counterfactual.py checkpoints/run1 \\
        --results-dir results/placebo/
"""

import argparse
import json
import sys
from pathlib import Path

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.output_format import SUBSKILLS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import build_adaptivity_pairs, filter_leaked_test_examples  # noqa: E402
from run_greedy_counterfactual import greedy_hint  # noqa: E402

STATE_A = {s: 0.48 for s in SUBSKILLS}
STATE_B = {s: 0.52 for s in SUBSKILLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", type=Path)
    ap.add_argument("--pair-limit", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    from socratic_hint.backends.finetuned import FinetunedBackend

    train_qids = {ex.qid for ex in load_mathdial("train")} | {ex.qid for ex in load_mathdial("validation")}
    test = filter_leaked_test_examples(load_mathdial("test"), train_qids)
    pairs, skipped = build_adaptivity_pairs(test)
    if args.pair_limit:
        pairs = pairs[: args.pair_limit]
    print(f"{len(pairs)} pairs ({skipped} skipped)")

    backend = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sampled_path = args.results_dir / "placebo_sampled.jsonl"
    greedy_path = args.results_dir / "placebo_greedy.jsonl"
    samp_rows, greedy_rows = [], []
    with open(sampled_path, "w", encoding="utf-8") as fs, open(greedy_path, "w", encoding="utf-8") as fg:
        for i, pair in enumerate(pairs):
            h, prob = pair.low_mastery_history, pair.problem
            a = backend.infer_hint_given_state(h, prob, STATE_A)
            b = backend.infer_hint_given_state(h, prob, STATE_B)
            c = backend.infer_hint_given_state(h, prob, STATE_A)
            rs = {"hint_low": a, "hint_high": b, "hint_control": c,
                  "differs_between_states": a.strip() != b.strip(),
                  "differs_noise_control": a.strip() != c.strip()}
            ga = greedy_hint(backend, h, prob, STATE_A)
            gb = greedy_hint(backend, h, prob, STATE_B)
            gc = greedy_hint(backend, h, prob, STATE_A)
            rg = {"hint_low": ga, "hint_high": gb, "hint_control": gc,
                  "differs_between_states": ga.strip() != gb.strip(),
                  "differs_noise_control": ga.strip() != gc.strip()}
            samp_rows.append(rs)
            greedy_rows.append(rg)
            fs.write(json.dumps(rs) + "\n"); fs.flush()
            fg.write(json.dumps(rg) + "\n"); fg.flush()
            print(f"[{i+1}/{len(pairs)}] sampled differs={rs['differs_between_states']} noise={rs['differs_noise_control']} | "
                  f"greedy differs={rg['differs_between_states']}", flush=True)

    n = len(samp_rows)
    summary = {
        "n": n,
        "placebo_states": [0.48, 0.52],
        "sampled": {"differ_rate": sum(r["differs_between_states"] for r in samp_rows) / n,
                    "noise_rate": sum(r["differs_noise_control"] for r in samp_rows) / n},
        "greedy": {"differ_rate": sum(r["differs_between_states"] for r in greedy_rows) / n,
                   "determinism_violations": sum(r["differs_noise_control"] for r in greedy_rows)},
    }
    (args.results_dir / "placebo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
