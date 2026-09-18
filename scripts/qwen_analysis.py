"""
Analysis of the Qwen2.5-3B-Instruct prompted (non-fine-tuned) baseline
(scripts/run_qwen_baseline.py) against the existing conditions. No new runs.

Judge/convergence comparisons pair results by position (qids repeat across
dialogues), dropping any position where either side failed, matching
significance_analysis.py. Needs scipy (global Python; not installed in .venv313):

    python scripts/qwen_analysis.py
"""

import json
import math
from pathlib import Path

from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_run"
DIMENSIONS = ["scaffolding_vs_telling", "correctness", "appropriateness"]

FILES = {
    "claude": "prompted_only.jsonl",
    "qwen_p_nostate": "qwen_prompted_no_state.jsonl",
    "qwen_p_state": "qwen_prompted_with_state.jsonl",
    "ft_nostate": "finetuned_without_state.jsonl",
    "ft_state": "finetuned_with_state.jsonl",
}


def load(name):
    recs, adapt = [], None
    with open(RESULTS_DIR / FILES[name], encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                adapt = d
            else:
                recs.append(d)
    return recs, adapt


def rank_biserial(diffs):
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 0.0
    ranks = stats.rankdata([abs(d) for d in diffs])
    pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    return (pos - neg) / (pos + neg)


def paired(a_name, b_name, data):
    ra, rb = data[a_name][0], data[b_name][0]
    assert len(ra) == len(rb), (a_name, b_name, len(ra), len(rb))
    bad = {i for i, r in enumerate(ra) if "error" in r} | {i for i, r in enumerate(rb) if "error" in r}
    pa = [r for i, r in enumerate(ra) if i not in bad]
    pb = [r for i, r in enumerate(rb) if i not in bad]
    out = {"n": len(pa)}
    for dim in DIMENSIONS:
        x = [r["judge_scores"][dim] for r in pa]
        y = [r["judge_scores"][dim] for r in pb]
        diffs = [q - p for p, q in zip(x, y)]
        try:
            wp = stats.wilcoxon(x, y).pvalue
        except ValueError:
            wp = float("nan")
        out[dim] = {"mean_diff_b_minus_a": sum(diffs) / len(diffs), "wilcoxon_p": wp, "rank_biserial": rank_biserial(diffs)}
    b = sum(1 for p, q in zip(pa, pb) if p["converged"] and not q["converged"])
    c = sum(1 for p, q in zip(pa, pb) if not p["converged"] and q["converged"])
    n = len(pa)
    ra_rate = sum(p["converged"] for p in pa) / n
    rb_rate = sum(q["converged"] for q in pb) / n
    mp = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    out["convergence"] = {"a_rate": ra_rate, "b_rate": rb_rate, "risk_diff_b_minus_a": rb_rate - ra_rate,
                          "only_a": b, "only_b": c, "mcnemar_exact_p": mp}
    return out


def describe(name, data):
    recs, adapt = data[name]
    ok = [r for r in recs if "error" not in r]
    d = {"n_records": len(recs), "n_ok": len(ok), "failed": len(recs) - len(ok)}
    for dim in DIMENSIONS:
        v = [r["judge_scores"][dim] for r in ok]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        d[dim] = {"mean": m, "sd": sd}
    d["convergence"] = sum(r["converged"] for r in ok) / len(ok)
    d["mean_turns"] = sum(r["turns_taken"] for r in ok) / len(ok)
    d["adaptivity"] = adapt
    return d


def two_prop(c1, n1, c2, n2):
    p1, p2 = c1 / n1, c2 / n2
    pool = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    data = {k: load(k) for k in FILES}
    summary = {"descriptive": {k: describe(k, data) for k in FILES}, "paired": {}, "adaptivity_tests": {}}
    for label, a, b in [
        ("state_suppressed: qwen_prompted -> finetuned", "qwen_p_nostate", "ft_nostate"),
        ("state_conditioned: qwen_prompted -> finetuned", "qwen_p_state", "ft_state"),
        ("qwen_prompted: state_suppressed -> state_conditioned", "qwen_p_nostate", "qwen_p_state"),
        ("state_conditioned: qwen_prompted -> claude", "qwen_p_state", "claude"),
    ]:
        summary["paired"][label] = paired(a, b, data)

    for k in ("qwen_p_nostate", "qwen_p_state"):
        a = data[k][1]
        if a is None:
            continue
        n = a["evaluated_pairs"]
        dc, nc = round(a["differ_rate"] * n), round(a["noise_rate"] * n)
        z, p = two_prop(dc, n, nc, n)
        summary["adaptivity_tests"][k] = {"net_rate": a["net_rate"], "differ_rate": a["differ_rate"],
                                          "noise_rate": a["noise_rate"], "pairs": n, "z_vs_zero": z, "p_vs_zero": p}
    out = RESULTS_DIR.parent / "qwen_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
