"""
Post-hoc significance tests on already-collected evaluation data.

Does not re-run any experiment - just analyzes results/full_run/*.jsonl,
which already exist. Two things are tested:

1. Judge-score gaps between Condition 2 (finetuned_without_state) and
   Condition 3 (finetuned_with_state), paired by dialogue position, since
   both conditions were run over the identical, identically-ordered
   `test_examples` list (confirmed in scripts/run_evaluation.py).
   - Wilcoxon signed-rank test per judge dimension
   - Paired t-test per judge dimension, as a cross-check
   - McNemar's test on convergence (binary, paired)

2. Whether the state-adaptivity net rates (0.52% vs 2.09%) are
   statistically distinguishable from zero, and from each other. Only
   aggregate counts are logged (no raw per-pair outcomes), so this uses an
   unpaired two-proportion z-test approximation - conservative, and
   explicitly weaker than a true paired McNemar test would be. Flagged
   in the output.
"""

import json
import math
from pathlib import Path

from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_run"

COND2_FILE = RESULTS_DIR / "finetuned_without_state.jsonl"  # state-suppressed
COND3_FILE = RESULTS_DIR / "finetuned_with_state.jsonl"     # state-conditioned

DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                continue
            records.append(d)
    return records


def load_adaptivity_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                return d
    raise ValueError(f"No adaptivity_batch record found in {path}")


def two_proportion_ztest(count1, n1, count2, n2):
    """Two-sided z-test for a difference in two independent proportions."""
    p1, p2 = count1 / n1, count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, float("nan"), float("nan")
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1 - p2, z, p_value


def one_proportion_vs_proportion_ztest(diff_count, n_diff, same_count, n_same):
    """Two-sided z-test that two independent proportions differ (used here to
    test whether the 'net rate' - differ_rate minus noise_rate - is
    distinguishable from zero within one condition)."""
    return two_proportion_ztest(diff_count, n_diff, same_count, n_same)


def main():
    cond2_records = load_records(COND2_FILE)
    cond3_records = load_records(COND3_FILE)

    print(f"Condition 2 (state-suppressed): {len(cond2_records)} records")
    print(f"Condition 3 (state-conditioned): {len(cond3_records)} records")

    # Drop any record with an 'error' key (failed generation), and its
    # positional counterpart in the other file, so the pairing stays valid.
    bad_positions = {
        i for i, r in enumerate(cond2_records) if "error" in r
    } | {
        i for i, r in enumerate(cond3_records) if "error" in r
    }
    if bad_positions:
        print(f"Dropping {len(bad_positions)} position(s) with a failed record: {sorted(bad_positions)}")

    c2 = [r for i, r in enumerate(cond2_records) if i not in bad_positions]
    c3 = [r for i, r in enumerate(cond3_records) if i not in bad_positions]
    assert len(c2) == len(c3), "Paired lists must be the same length after dropping failures"
    n_paired = len(c2)
    print(f"Paired sample size after dropping failures: n = {n_paired}\n")

    print("=" * 70)
    print("1. JUDGE SCORES - Condition 2 (state-suppressed) vs Condition 3 (state-conditioned)")
    print("=" * 70)
    for dim in DIMENSIONS:
        x = [r["judge_scores"][dim] for r in c2]
        y = [r["judge_scores"][dim] for r in c3]
        diffs = [b - a for a, b in zip(x, y)]
        mean_diff = sum(diffs) / len(diffs)

        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(x, y)
        except ValueError as exc:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
            print(f"  [wilcoxon failed for {dim}: {exc}]")

        ttest_stat, ttest_p = stats.ttest_rel(x, y)

        print(f"\n{dim}:")
        print(f"  mean (Cond2 suppressed) = {sum(x)/len(x):.3f}, mean (Cond3 conditioned) = {sum(y)/len(y):.3f}")
        print(f"  mean paired difference (Cond3 - Cond2) = {mean_diff:+.3f}")
        print(f"  Wilcoxon signed-rank: statistic={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")
        print(f"  Paired t-test:        t={ttest_stat:.3f}, p={ttest_p:.4f}")

    print("\n" + "=" * 70)
    print("2. CONVERGENCE - McNemar's test (paired binary outcome)")
    print("=" * 70)
    conv2 = [r["converged"] for r in c2]
    conv3 = [r["converged"] for r in c3]
    # McNemar 2x2: only discordant pairs matter
    both_true = sum(1 for a, b in zip(conv2, conv3) if a and b)
    only_c2 = sum(1 for a, b in zip(conv2, conv3) if a and not b)
    only_c3 = sum(1 for a, b in zip(conv2, conv3) if not a and b)
    both_false = sum(1 for a, b in zip(conv2, conv3) if not a and not b)
    print(f"  Both converged: {both_true}, only Cond2 converged: {only_c2}, "
          f"only Cond3 converged: {only_c3}, neither converged: {both_false}")
    n_discordant = only_c2 + only_c3
    if n_discordant > 0:
        # exact McNemar via binomial test on discordant pairs
        mcnemar_p = stats.binomtest(min(only_c2, only_c3), n_discordant, 0.5).pvalue
        print(f"  Discordant pairs: {n_discordant}. Exact McNemar p-value = {mcnemar_p:.4f}")
    else:
        print("  No discordant pairs - conditions agree on every dialogue, test is degenerate.")
    print(f"  Convergence rate Cond2 = {sum(conv2)/len(conv2):.3f}, Cond3 = {sum(conv3)/len(conv3):.3f}")

    print("\n" + "=" * 70)
    print("3. STATE-ADAPTIVITY NET RATES")
    print("   (unpaired two-proportion z-test approximation - only aggregate")
    print("    counts were logged, not raw per-pair outcomes, so this is a")
    print("    conservative substitute for a true paired McNemar test)")
    print("=" * 70)

    summaries = {}
    for name, path in [
        ("Condition 1 (prompted)", RESULTS_DIR / "prompted_only.jsonl"),
        ("Condition 2 (state-suppressed)", COND2_FILE),
        ("Condition 3 (state-conditioned)", COND3_FILE),
    ]:
        summaries[name] = load_adaptivity_summary(path)

    for name, s in summaries.items():
        n = s["evaluated_pairs"]
        differ_count = round(s["differ_rate"] * n)
        noise_count = round(s["noise_rate"] * n)
        net, z, p = one_proportion_vs_proportion_ztest(differ_count, n, noise_count, n)
        print(f"\n{name}: n={n}")
        print(f"  differ_rate={s['differ_rate']:.4f} ({differ_count}/{n}), "
              f"noise_rate={s['noise_rate']:.4f} ({noise_count}/{n}), "
              f"net_rate={s['net_rate']:.4f}")
        print(f"  Is net rate != 0? z={z:.3f}, p={p:.4f}")

    # Compare Cond2's net rate to Cond3's net rate directly.
    s2, s3 = summaries["Condition 2 (state-suppressed)"], summaries["Condition 3 (state-conditioned)"]
    n2, n3 = s2["evaluated_pairs"], s3["evaluated_pairs"]
    d2, s2n = round(s2["differ_rate"] * n2), round(s2["noise_rate"] * n2)
    d3, s3n = round(s3["differ_rate"] * n3), round(s3["noise_rate"] * n3)
    p_net2 = d2 / n2 - s2n / n2
    p_net3 = d3 / n3 - s3n / n3
    # variance of a difference of two independent proportions, summed for two
    # independent differences being compared
    var2 = (d2 / n2) * (1 - d2 / n2) / n2 + (s2n / n2) * (1 - s2n / n2) / n2
    var3 = (d3 / n3) * (1 - d3 / n3) / n3 + (s3n / n3) * (1 - s3n / n3) / n3
    se_diff = math.sqrt(var2 + var3)
    z_diff = (p_net3 - p_net2) / se_diff if se_diff > 0 else float("nan")
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff))) if se_diff > 0 else float("nan")
    print(f"\nCond3 net rate ({p_net3:.4f}) vs Cond2 net rate ({p_net2:.4f}):")
    print(f"  difference = {p_net3 - p_net2:+.4f}, z = {z_diff:.3f}, p = {p_diff:.4f}")


if __name__ == "__main__":
    main()
