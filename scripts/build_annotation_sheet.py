"""
Build a human mastery-annotation sheet for the silver-label validity check.

For each sampled tutor turn, the annotator sees the problem, the reference
answer, and the dialogue UP TO (not including) that tutor turn, and rates the
student's mastery on each of the four subskills. The tutor's own move and
message at that turn are hidden, so ratings cannot simply echo the silver
label's source (tutor move type). The silver labels are written to a separate
key file that annotators must not open until they are done.

Samples SAMPLES_PER_MOVE tutor turns per move type from the 241 leakage-
filtered test dialogues (fixed seed), one turn per dialogue, and requires at
least one prior student turn.

    .venv313\\Scripts\\python.exe scripts/build_annotation_sheet.py
"""

import csv
import random
from pathlib import Path

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.data.state_labels import MOVE_TO_MASTERY_PRIOR
from socratic_hint.output_format import SUBSKILLS

SEED = 3407
SAMPLES_PER_MOVE = 15
OUT_DIR = Path(__file__).resolve().parent.parent / "annotation"


def main():
    train_qids = {ex.qid for ex in load_mathdial("train")} | {ex.qid for ex in load_mathdial("validation")}
    test = [ex for ex in load_mathdial("test") if ex.qid not in train_qids]
    print(f"{len(test)} leakage-filtered test dialogues")

    candidates: dict[str, list[tuple[int, int]]] = {m: [] for m in MOVE_TO_MASTERY_PRIOR}
    for d, ex in enumerate(test):
        for i, turn in enumerate(ex.turns):
            if turn.speaker != "tutor" or turn.move not in candidates:
                continue
            if any(t.speaker == "student" for t in ex.turns[:i]):
                candidates[turn.move].append((d, i))

    rng = random.Random(SEED)
    chosen: list[tuple[int, int, str]] = []
    used_dialogues: set[int] = set()
    for move, pool in candidates.items():
        rng.shuffle(pool)
        picked = 0
        for d, i in pool:
            if d in used_dialogues:
                continue
            chosen.append((d, i, move))
            used_dialogues.add(d)
            picked += 1
            if picked == SAMPLES_PER_MOVE:
                break
        assert picked == SAMPLES_PER_MOVE, (move, picked)
    rng.shuffle(chosen)

    OUT_DIR.mkdir(exist_ok=True)
    sheet_rows, key_rows = [], []
    for n, (d, i, move) in enumerate(chosen, 1):
        ex = test[d]
        history = "\n".join(f"[{'TUTOR' if t.speaker == 'tutor' else 'STUDENT'}] {t.text}" for t in ex.turns[:i])
        item = f"item{n:02d}"
        row = {"item_id": item, "problem": ex.problem, "reference_answer": ex.ground_truth,
               "dialogue_so_far": history}
        for s in SUBSKILLS:
            row[f"{s}_rating"] = ""
        row["notes"] = ""
        sheet_rows.append(row)
        key_rows.append({"item_id": item, "qid": ex.qid, "turn_index": i, "tutor_move": move,
                         "silver_prior": MOVE_TO_MASTERY_PRIOR[move]})

    def write(path, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR / "annotation_sheet.csv", sheet_rows)
    write(OUT_DIR / "annotation_key_DO_NOT_OPEN.csv", key_rows)
    print(f"Wrote {len(sheet_rows)} items to {OUT_DIR}")


if __name__ == "__main__":
    main()
