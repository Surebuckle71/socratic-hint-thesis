"""
Figures for the results chapter, drawn from the saved result files.

  five_conditions.pdf   judge scores and convergence for all five conditions, with 95% intervals
  adaptivity_story.pdf  share of dialogues whose hints differ under a different injected state
                        versus a repeat of the same state, for sampled and greedy decoding

    python scripts/make_figures.py OUTPUT_DIR
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else R
OUT.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------- Figure A: five conditions ----------------
q = json.load(open(R / "qwen_analysis_summary.json", encoding="utf-8"))["descriptive"]
order = [("claude", "1. Claude, prompted", "#4C72B0"),
         ("ft_nostate", "2. FT, no state", "#DD8452"),
         ("ft_state", "3. FT + state", "#55A868"),
         ("qwen_p_nostate", "4. Qwen, no state", "#8172B3"),
         ("qwen_p_state", "5. Qwen + state", "#B9AED6")]
dims = [("scaffolding_vs_telling", "Scaffolding"), ("correctness", "Correctness"), ("appropriateness", "Appropriateness")]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [3, 1.35]})
w = 0.16
for j, (key, label, color) in enumerate(order):
    d = q[key]
    means = [d[k]["mean"] for k, _ in dims]
    errs = [1.96 * d[k]["sd"] / math.sqrt(d["n_ok"]) for k, _ in dims]
    xs = [i + (j - 2) * w for i in range(3)]
    ax.bar(xs, means, w, yerr=errs, color=color, label=label.replace("\n", " "), capsize=2, error_kw={"lw": 0.8})
    k = round(d["convergence"] * d["n_ok"])
    lo, hi = wilson(k, d["n_ok"])
    ax2.bar(j, d["convergence"], 0.7, color=color, yerr=[[d["convergence"] - lo], [hi - d["convergence"]]], capsize=2, error_kw={"lw": 0.8})
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in dims])
ax.set_ylim(0, 6.3); ax.set_yticks(range(6)); ax.set_ylabel("Mean judge score (1-5)")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
ax2.set_ylim(0.6, 1.0); ax2.set_ylabel("Convergence rate")
ax2.set_xticks(range(5)); ax2.set_xticklabels(["1", "2", "3", "4", "5"])
ax2.set_xlabel("Condition"); ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "five_conditions.pdf"); plt.close(fig)

# ---------------- Figure B: adaptivity story ----------------
cf = json.load(open(R / "counterfactual" / "counterfactual_adaptivity_summary.json"))
s1 = json.load(open(R / "counterfactual" / "semantic_adaptivity_summary.json"))
s2 = json.load(open(R / "counterfactual" / "semantic_adaptivity_summary_claude-opus-5.json"))
g = json.load(open(R / "counterfactual" / "greedy_counterfactual_summary.json"))
g1 = json.load(open(R / "counterfactual" / "semantic_greedy_summary.json"))
g2 = json.load(open(R / "counterfactual" / "semantic_greedy_summary_claude-opus-5.json"))
groups = [
    ("Sampled\nexact text", cf["differ_rate"], cf["noise_rate"]),
    ("Sampled\nsemantic\n(Sonnet 5)", s1["semantic_differ_rate"], s1["semantic_noise_rate"]),
    ("Sampled\nsemantic\n(Opus 5)", s2["semantic_differ_rate"], s2["semantic_noise_rate"]),
    ("Greedy\nexact text", g["differ_rate"], g["determinism_violations"] / g["n"]),
    ("Greedy\nsemantic\n(Sonnet 5)", g1["semantic_differ_rate_greedy"], 0.0),
    ("Greedy\nsemantic\n(Opus 5)", g2["semantic_differ_rate_greedy"], 0.0),
]
fig, ax = plt.subplots(figsize=(6.8, 3.5))
w = 0.36
for i, (lab, diff, noise) in enumerate(groups):
    b1 = ax.bar(i - w / 2, diff * 100, w, color="#4C72B0", label="Different injected state" if i == 0 else None)
    b2 = ax.bar(i + w / 2, noise * 100, w, color="#B0B0B0", label="Same injected state (repeat)" if i == 0 else None)
    ax.text(i - w / 2, diff * 100 + 1.5, f"{diff*100:.1f}", ha="center", fontsize=6.5)
    ax.text(i + w / 2, noise * 100 + 1.5, f"{noise*100:.1f}", ha="center", fontsize=6.5)
ax.axvline(2.5, color="black", lw=0.6, ls=":")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g_[0] for g_ in groups], fontsize=8)
ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Dialogues whose two hints differ (%)")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(OUT / "adaptivity_story.pdf"); plt.close(fig)

# ---------------- Figure C: judge scores of the three main conditions (mean +/- 1 SD) ----------------
def _scores(fname):
    rows = []
    for line in open(R / "full_run" / fname, encoding="utf-8"):
        r = json.loads(line)
        if r.get("record_type") != "adaptivity_batch" and r.get("judge_scores"):
            rows.append(r["judge_scores"])
    return rows


main3 = [("1. Claude, prompted", "prompted_only.jsonl", "#4C72B0"),
         ("2. FT, no state", "finetuned_without_state.jsonl", "#DD8452"),
         ("3. FT + state", "finetuned_with_state.jsonl", "#55A868")]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
w = 0.26
for j, (label, fname, color) in enumerate(main3):
    rows = _scores(fname)
    means, sds = [], []
    for k, _ in dims:
        v = [r[k] for r in rows]
        m = sum(v) / len(v)
        means.append(m)
        sds.append(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)))
    ax.bar([i + (j - 1) * w for i in range(3)], means, w, yerr=sds, color=color, label=label, capsize=3, error_kw={"lw": 1.0})
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in dims])
ax.set_ylim(0, 6.4); ax.set_yticks(range(6)); ax.set_ylabel("Mean judge score (1-5)")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2)
fig.tight_layout()
fig.savefig(OUT / "judge_scores.pdf"); plt.close(fig)

# ---------------- Figure D: raw differ rate versus noise floor (original content-varying design) ----------------
def _adaptivity(fname):
    for line in open(R / "full_run" / fname, encoding="utf-8"):
        r = json.loads(line)
        if r.get("record_type") == "adaptivity_batch":
            return r
    raise ValueError(fname)


adapt = [("1. Claude,\nprompted", "prompted_only.jsonl"),
         ("2. FT,\nno state", "finetuned_without_state.jsonl"),
         ("3. FT +\nstate", "finetuned_with_state.jsonl")]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
w = 0.36
for i, (lab, fname) in enumerate(adapt):
    a = _adaptivity(fname)
    ax.bar(i - w / 2, a["differ_rate"], w, color="#4C72B0", label="Differ rate" if i == 0 else None)
    ax.bar(i + w / 2, a["noise_rate"], w, color="#C44E52", label="Noise floor" if i == 0 else None)
    ax.text(i, 1.03, f"net={a['net_rate']:.4f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in adapt])
ax.set_ylim(0, 1.15); ax.set_ylabel("Rate")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(fontsize=8, loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "adaptivity.pdf"); plt.close(fig)
print("wrote", OUT / "five_conditions.pdf,", OUT / "adaptivity_story.pdf,", OUT / "judge_scores.pdf", "and", OUT / "adaptivity.pdf")
