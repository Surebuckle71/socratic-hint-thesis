"""
Turn the repeated-run results into a LaTeX table (and a JSON of the numbers) for the thesis.

Runs = the original run (results/full_run/, unseeded) plus every completed seed in
results/seeds/seed*/. For each fine-tuned condition and for the paired Condition 3 minus
Condition 2 gap it lists the per-run values, the mean, the SD and the 95% t-interval.

    python scripts/seed_latex.py [output.tex]
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_analysis import COND, DIMS, load, paired_gap, run_stats, summarize  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
LABEL = {"scaffolding_vs_telling": "Scaffolding", "correctness": "Correctness", "appropriateness": "Appropriateness",
         "convergence": "Convergence rate", "net_adaptivity": "Net adaptivity rate"}


def completed_runs():
    runs = {"original": R / "full_run"}
    for d in sorted((R / "seeds").glob("seed*")):
        f3 = d / "finetuned_with_state.jsonl"
        if f3.exists() and all((d / f"{c}.jsonl").exists() for c in COND):
            last = json.loads(open(f3, encoding="utf-8").readlines()[-1])
            if last.get("record_type") == "adaptivity_batch":
                runs[d.name] = d
    return runs


def fmt(x, dec=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"


def row(name, vals, dec, sign=False):
    m, sd, (lo, hi) = summarize(vals)
    vs = ", ".join(fmt(v, dec, sign) for v in vals)
    ci = "n/a" if math.isnan(lo) else f"[{fmt(lo, dec, sign)}, {fmt(hi, dec, sign)}]"
    return f"    {name} & {vs} & {fmt(m, dec, sign)} & {fmt(sd, dec)} & {ci} \\\\", (m, sd, lo, hi)


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "seeds" / "seed_table.tex"
    runs = completed_runs()
    per = {c: [] for c in COND}
    gaps = []
    for name, d in runs.items():
        loaded = {c: load(d / f"{c}.jsonl") for c in COND}
        for c in COND:
            per[c].append(run_stats(*loaded[c]))
        gaps.append(paired_gap(loaded["finetuned_without_state"][0], loaded["finetuned_with_state"][0]))
    n = len(runs)
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        rf"  \caption{{Run-to-run variation of the two fine-tuned conditions over {n} runs (the original run and {n - 1} repeat runs with different sampling seeds): the value in each run, the mean, the standard deviation, and the 95\% $t$-interval of the mean across runs. The lower block is the paired Condition~3 minus Condition~2 difference within each run.}}",
        r"  \label{tab:repeated-runs}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}lXccc@{}}",
        r"    \toprule",
        r"    Measure & Values across runs & Mean & SD & 95\% CI \\",
    ]
    numbers = {"runs": list(runs)}
    for c, title in (("finetuned_without_state", "Condition~2 (state suppressed)"), ("finetuned_with_state", "Condition~3 (state-conditioned)")):
        lines += [r"    \midrule", rf"    \multicolumn{{5}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for k in DIMS + ["convergence", "net_adaptivity"]:
            text, stats = row(LABEL[k], [x[k] for x in per[c]], 2 if k in DIMS else 3)
            lines.append(text); numbers[f"{c}.{k}"] = stats
    lines += [r"    \midrule", r"    \multicolumn{5}{@{}l}{\emph{Condition~3 minus Condition~2 (paired within each run)}} \\"]
    for k in DIMS + ["convergence"]:
        text, stats = row(LABEL[k], [g[k] for g in gaps], 2 if k in DIMS else 3, sign=True)
        lines.append(text); numbers[f"gap.{k}"] = stats
    lines += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    json.dump(numbers, open(out_path.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out_path} ({n} runs: {list(runs)})")


if __name__ == "__main__":
    main()
