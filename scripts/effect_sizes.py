"""
Paired effect sizes for the Condition 2 vs Condition 3 comparison, computed
from the same already-collected results/full_run/*.jsonl used by
significance_analysis.py. No new experiment runs.

- Matched-pairs rank-biserial correlation for each Wilcoxon signed-rank test
  (scaffolding, correctness, appropriateness).
- Risk difference and odds ratio for convergence (McNemar 2x2 table).
"""

import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_run"


def rankdata(values):
    """Average ranks (1-indexed), ties get the mean rank of their block."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks
COND2_FILE = RESULTS_DIR / "finetuned_without_state.jsonl"
COND3_FILE = RESULTS_DIR / "finetuned_with_state.jsonl"
DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                continue
            records.append(d)
    return records


def align_by_position(recs2, recs3):
    """Same approach as significance_analysis.py: both files were run over the
    identical, identically-ordered test_examples list, so pair by position and
    drop any position where either side failed."""
    bad_positions = {i for i, r in enumerate(recs2) if "error" in r} | {
        i for i, r in enumerate(recs3) if "error" in r
    }
    c2 = [r for i, r in enumerate(recs2) if i not in bad_positions]
    c3 = [r for i, r in enumerate(recs3) if i not in bad_positions]
    assert len(c2) == len(c3)
    return list(zip(c2, c3))


def rank_biserial_from_wilcoxon(diffs):
    """Matched-pairs rank-biserial correlation: (sum of + ranks - sum of - ranks) / total rank sum."""
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 0
    abs_diffs = [abs(d) for d in diffs]
    ranks = rankdata(abs_diffs)
    pos_rank_sum = sum(r for r, d in zip(ranks, diffs) if d > 0)
    neg_rank_sum = sum(r for r, d in zip(ranks, diffs) if d < 0)
    total = pos_rank_sum + neg_rank_sum
    r_rb = (pos_rank_sum - neg_rank_sum) / total
    return r_rb, n


def main():
    recs2 = load_records(COND2_FILE)
    recs3 = load_records(COND3_FILE)
    pairs = align_by_position(recs2, recs3)
    print(f"Aligned pairs: {len(pairs)}")

    for dim in DIMENSIONS:
        diffs = [p3["judge_scores"][dim] - p2["judge_scores"][dim] for p2, p3 in pairs]
        r_rb, n_nonzero = rank_biserial_from_wilcoxon(diffs)
        print(f"{dim}: rank-biserial r = {r_rb:.3f} (n_nonzero={n_nonzero}, n_total={len(diffs)})")

    # Convergence: 2x2 McNemar table, Cond2 vs Cond3
    a = b = c = d = 0  # a: both converge, b: 2 conv/3 not, c: 2 not/3 conv, d: neither
    for p2, p3 in pairs:
        c2, c3 = p2["converged"], p3["converged"]
        if c2 and c3:
            a += 1
        elif c2 and not c3:
            b += 1
        elif not c2 and c3:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    p2_rate = (a + b) / n
    p3_rate = (a + c) / n
    risk_diff = p3_rate - p2_rate
    # Odds ratio for paired discordant pairs (b, c)
    odds_ratio = c / b if b > 0 else float("inf")
    print(f"\nConvergence 2x2 (a={a}, b={b}, c={c}, d={d}, n={n})")
    print(f"Cond2 rate={p2_rate:.4f}, Cond3 rate={p3_rate:.4f}")
    print(f"Risk difference (Cond3 - Cond2) = {risk_diff:.4f}")
    print(f"Discordant-pairs odds ratio (c/b) = {odds_ratio:.3f}")


if __name__ == "__main__":
    main()
