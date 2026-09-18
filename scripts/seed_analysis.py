"""
Across-run summary of the fine-tuned conditions: mean, SD and 95% t-interval
over the original run plus the seeded repeat runs in results/seeds/seed*/.

For each run and condition it takes the mean judge scores, the convergence
rate, and the original diagnostic's net adaptivity rate; it also summarizes the
paired Condition 3 minus Condition 2 gap per run. The original run (results/
full_run/) was not seeded; the repeats use --seed 1, 2, ... The intervals
describe run-to-run variation in generation, not sampling uncertainty within
one run (those are the bootstrap intervals elsewhere in the thesis). With few
runs the t-interval is wide.

    python scripts/seed_analysis.py
"""

import json
import math
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"
DIMS = ["scaffolding_vs_telling", "correctness", "appropriateness"]
COND = {"finetuned_without_state": "Condition 2 (state suppressed)", "finetuned_with_state": "Condition 3 (state-conditioned)"}


def load(path):
    recs, adapt = [], None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("record_type") == "adaptivity_batch":
            adapt = d
        else:
            recs.append(d)
    return recs, adapt


def run_stats(recs, adapt):
    ok = [r for r in recs if "error" not in r]
    out = {d: sum(r["judge_scores"][d] for r in ok) / len(ok) for d in DIMS}
    out["convergence"] = sum(r["converged"] for r in ok) / len(ok)
    out["net_adaptivity"] = adapt["net_rate"] if adapt else float("nan")
    out["n_ok"] = len(ok)
    out["failed"] = len(recs) - len(ok)
    return out


def paired_gap(r2, r3):
    bad = {i for i, r in enumerate(r2) if "error" in r} | {i for i, r in enumerate(r3) if "error" in r}
    a = [r for i, r in enumerate(r2) if i not in bad]
    b = [r for i, r in enumerate(r3) if i not in bad]
    gap = {d: sum(y["judge_scores"][d] - x["judge_scores"][d] for x, y in zip(a, b)) / len(a) for d in DIMS}
    gap["convergence"] = sum(y["converged"] for y in b) / len(b) - sum(x["converged"] for x in a) / len(a)
    return gap


def summarize(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float("nan"), (float("nan"), float("nan"))
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, (m - half, m + half)


def main():
    runs = {"original (unseeded)": R / "full_run"}
    for d in sorted((R / "seeds").glob("seed*")):
        if all((d / f"{c}.jsonl").exists() for c in COND) and json.loads(
            open(d / "finetuned_with_state.jsonl", encoding="utf-8").readlines()[-1]
        ).get("record_type") == "adaptivity_batch":
            runs[d.name] = d
    print(f"Completed runs: {list(runs)}")
    per = {c: [] for c in COND}
    gaps = []
    for name, d in runs.items():
        loaded = {c: load(d / f"{c}.jsonl") for c in COND}
        for c in COND:
            per[c].append(run_stats(*loaded[c]))
        gaps.append(paired_gap(loaded["finetuned_without_state"][0], loaded["finetuned_with_state"][0]))
        print(f"  {name}: failed C2={per['finetuned_without_state'][-1]['failed']} C3={per['finetuned_with_state'][-1]['failed']}")

    keys = DIMS + ["convergence", "net_adaptivity"]
    for c, label in COND.items():
        print(f"\n{label}: mean, SD, 95% CI across {len(runs)} runs")
        for k in keys:
            m, sd, (lo, hi) = summarize([x[k] for x in per[c]])
            print(f"  {k:24s} {m:7.3f}  SD {sd:6.3f}  [{lo:7.3f}, {hi:7.3f}]   values: " + ", ".join(f"{x[k]:.3f}" for x in per[c]))
    print("\nCondition 3 minus Condition 2 (paired within each run):")
    for k in DIMS + ["convergence"]:
        m, sd, (lo, hi) = summarize([g[k] for g in gaps])
        print(f"  {k:24s} {m:+7.3f}  SD {sd:6.3f}  [{lo:+7.3f}, {hi:+7.3f}]   values: " + ", ".join(f"{g[k]:+.3f}" for g in gaps))


if __name__ == "__main__":
    main()
