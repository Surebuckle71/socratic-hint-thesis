"""
Label-level sensitivity of the silver mapping (tutor move -> mastery prior)
against a completed annotation sheet. No model retraining: this asks how well
alternative mappings track independently RATED mastery on the 60-item pilot,
and where the thesis's mapping ranks among all possible move orderings.

Spearman correlation is invariant to monotone transformations of the prior, so
only the ORDER of (and ties between) the four moves matters; values are shown
for readability.

    python scripts/label_sensitivity.py annotation/annotation_sheet_claude.csv
"""

import csv
import itertools
import sys
from pathlib import Path

from scipy import stats

ANN = Path(__file__).resolve().parent.parent / "annotation"
SUB = ["problem_comprehension", "arithmetic_execution", "step_sequencing", "self_correction"]
MOVES = ["telling", "probing", "focus", "generic"]

MAPPINGS = {
    "thesis (tell .2, probe .5, focus .5, generic .7)": {"telling": .2, "probing": .5, "focus": .5, "generic": .7},
    "probing < focus (tell .2, probe .4, focus .6, generic .7)": {"telling": .2, "probing": .4, "focus": .6, "generic": .7},
    "focus < probing (tell .2, focus .4, probe .6, generic .7)": {"telling": .2, "probing": .6, "focus": .4, "generic": .7},
    "generic neutral (tell .2, probe .5, focus .5, generic .5)": {"telling": .2, "probing": .5, "focus": .5, "generic": .5},
    "telling vs rest (tell .2, all others .6)": {"telling": .2, "probing": .6, "focus": .6, "generic": .6},
    "flat (all .5)": {m: .5 for m in MOVES},
}


def item_mean(row):
    v = [int(row[f"{s}_rating"]) for s in SUB if row[f"{s}_rating"]]
    return sum(v) / len(v) if v else None


def main():
    sheet = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
    key = {r["item_id"]: r for r in csv.DictReader(open(ANN / "annotation_key_DO_NOT_OPEN.csv", encoding="utf-8-sig"))}
    data = [(key[r["item_id"]]["tutor_move"], item_mean(r)) for r in sheet if item_mean(r) is not None]
    ys = [y for _, y in data]
    print(f"n={len(data)}")
    for m in MOVES:
        v = [y for mv, y in data if mv == m]
        print(f"  mean rated mastery | {m:8s}: {sum(v)/len(v):.2f} (n={len(v)})")
    print("\nSpearman(mapping, mean rated mastery):")
    res = {}
    for name, mp in MAPPINGS.items():
        xs = [mp[m] for m, _ in data]
        rho = stats.spearmanr(xs, ys)[0] if len(set(xs)) > 1 else float("nan")
        res[name] = rho
        print(f"  {rho:+.2f}  {name}")

    base = res[list(MAPPINGS)[0]]
    all_rho = []
    for perm in itertools.permutations(range(4)):
        mp = {m: perm[i] for i, m in enumerate(MOVES)}
        all_rho.append(stats.spearmanr([mp[m] for m, _ in data], ys)[0])
    better = sum(1 for r in all_rho if r > base + 1e-9)
    print(f"\nAll 24 strict orderings of the four moves: best {max(all_rho):+.2f}, worst {min(all_rho):+.2f}; "
          f"{better} orderings correlate more strongly than the thesis mapping.")


if __name__ == "__main__":
    main()
