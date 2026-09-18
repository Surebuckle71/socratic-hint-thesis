"""
Systematic error-analysis taxonomy over the judge rationales already collected
in results/full_run/*.jsonl. No new experiment runs - this only re-reads and
categorizes already-generated judge rationale text for Condition 2 and
Condition 3's hints, keyword-matched against categories derived by inspecting
the most divergent pairs by hand first.

This is a heuristic, keyword-based classification of free-text judge
rationales, not a human-annotated taxonomy: a rationale can match more than
one category, or none. It is reported as a best-effort systematic pass across
all 240 pairs, not a precise or exhaustive one.
"""

import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_run"
COND2_FILE = RESULTS_DIR / "finetuned_without_state.jsonl"
COND3_FILE = RESULTS_DIR / "finetuned_with_state.jsonl"

CATEGORIES = {
    "factual_math_error": [
        r"factual error", r"mathematically wrong", r"mathematically unsound",
        r"misstates", r"incorrectly implies", r"incorrect statement",
        r"is actually correct", r"is actually incorrect",
    ],
    "affirms_wrong_answer": [
        r"incorrectly affirms", r"confirm(s|ing) it as correct",
        r"repeats the student's incorrect answer",
    ],
    "gives_away_answer": [
        r"gives? away", r"reveals? the answer", r"tells? rather than",
        r"telling rather than guiding",
    ],
    "vague_unhelpful": [
        r"\bvague\b", r"no mathematical guidance", r"offers no guidance",
        r"fails to address",
    ],
    "off_topic_confusing": [
        r"off-topic", r"irrelevant", r"unrelated", r"confusing", r"unclear",
        r"misdirect",
    ],
    "appropriate_scaffolding": [
        r"without giving (the|away the) answer", r"without giving away",
        r"appropriately", r"guides? the student", r"prompts? reflection",
    ],
}


def load(path):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                continue
            recs.append(d)
    return recs


def classify(rationale):
    rationale_lower = rationale.lower()
    hits = []
    for cat, patterns in CATEGORIES.items():
        if any(re.search(p, rationale_lower) for p in patterns):
            hits.append(cat)
    return hits


def main():
    c2 = load(COND2_FILE)
    c3 = load(COND3_FILE)
    bad = {i for i, r in enumerate(c2) if "error" in r} | {
        i for i, r in enumerate(c3) if "error" in r
    }
    c2 = [r for i, r in enumerate(c2) if i not in bad]
    c3 = [r for i, r in enumerate(c3) if i not in bad]
    assert len(c2) == len(c3) == 240

    counts2 = {cat: 0 for cat in CATEGORIES}
    counts3 = {cat: 0 for cat in CATEGORIES}
    for r in c2:
        for cat in classify(r["rationale"]):
            counts2[cat] += 1
    for r in c3:
        for cat in classify(r["rationale"]):
            counts3[cat] += 1

    n = len(c2)
    print(f"n = {n} dialogues per condition\n")
    print(f"{'Category':30s} {'Cond2 (n, %)':18s} {'Cond3 (n, %)':18s}")
    for cat in CATEGORIES:
        c2n, c3n = counts2[cat], counts3[cat]
        print(f"{cat:30s} {c2n:3d} ({100*c2n/n:4.1f}%)     {c3n:3d} ({100*c3n/n:4.1f}%)")


if __name__ == "__main__":
    main()
