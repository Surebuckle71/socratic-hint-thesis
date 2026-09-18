"""
Compare completed human mastery ratings against the silver labels.

Ratings are integers 1 (low), 2 (medium), 3 (high), or blank/0 for "cannot
tell". Pass one or two completed annotation sheets (two enables inter-rater
agreement). Needs scipy (global Python):

    python scripts/analyze_annotation.py annotation/annotation_sheet_A.csv [annotation/annotation_sheet_B.csv]

Reports, per annotator and for the mean of annotators:
  - Spearman correlation between the item's mean subskill rating and the
    silver prior (0.2 / 0.5 / 0.7);
  - mean rating by tutor move (does the move type track rated mastery?);
  - how much the four subskill ratings differ within an item (do the four
    subskills actually vary independently, which the silver labels assume
    they do not);
  - with two annotators, quadratic-weighted Cohen's kappa per subskill.
"""

import csv
import sys
from pathlib import Path

from scipy import stats

ANN_DIR = Path(__file__).resolve().parent.parent / "annotation"
SUBSKILLS = ["problem_comprehension", "arithmetic_execution", "step_sequencing", "self_correction"]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["item_id"]: r for r in csv.DictReader(fh)}


def rating(row, s):
    v = (row.get(f"{s}_rating") or "").strip()
    return int(v) if v in {"1", "2", "3"} else None


def item_mean(row):
    vals = [rating(row, s) for s in SUBSKILLS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def weighted_kappa(a, b, k=3):
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x - 1][y - 1] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: ((i - j) / (k - 1)) ** 2
    num = sum(w(i, j) * obs[i][j] / n for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n ** 2 for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def main():
    sheets = [read(p) for p in sys.argv[1:]]
    if not sheets:
        sys.exit(__doc__)
    key = {r["item_id"]: r for r in csv.DictReader(open(ANN_DIR / "annotation_key_DO_NOT_OPEN.csv", encoding="utf-8-sig"))}

    for idx, sheet in enumerate(sheets, 1):
        xs, ys, by_move, spread = [], [], {}, []
        for item, row in sheet.items():
            m = item_mean(row)
            if m is None:
                continue
            silver = float(key[item]["silver_prior"])
            xs.append(silver); ys.append(m)
            by_move.setdefault(key[item]["tutor_move"], []).append(m)
            vals = [rating(row, s) for s in SUBSKILLS]
            vals = [v for v in vals if v is not None]
            spread.append(max(vals) - min(vals))
        rho, p = stats.spearmanr(xs, ys)
        print(f"\nAnnotator {idx}: n={len(xs)} items rated")
        print(f"  Spearman(silver prior, mean rated mastery) = {rho:.2f} (p={p:.3f})")
        for move, v in sorted(by_move.items()):
            print(f"  mean rated mastery when tutor move = {move:8s}: {sum(v)/len(v):.2f} (n={len(v)})")
        print(f"  items where the four subskill ratings are all identical: {sum(1 for s in spread if s == 0)}/{len(spread)}")

    if len(sheets) == 2:
        print("\nInter-rater agreement (quadratic-weighted kappa):")
        for s in SUBSKILLS:
            pairs = [(rating(sheets[0][i], s), rating(sheets[1][i], s)) for i in sheets[0] if i in sheets[1]]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            print(f"  {s:24s} kappa_w = {weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]):.2f} (n={len(pairs)})")


if __name__ == "__main__":
    main()
