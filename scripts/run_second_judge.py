"""
Second-judge validation: re-score every already-generated hint with a
different judge model (claude-opus-5, vs. the original claude-sonnet-5), and
report inter-rater agreement.

This directly answers two of the professor's points at once:
  - "LLM-as-judge" (High): the original judge is the same model as Condition 1
    (self-enhancement bias risk). Re-scoring with a different model gives an
    independent read on whether Condition 1's advantage survives a different
    judge.
  - No new hints are generated - this reuses `first_hint` from
    results/full_run/*.jsonl, so the only cost is judge-scoring API calls
    (no GPU, no simulated-student calls).

The original per-example records do not store the problem statement or
dialogue context, only qid + first_hint + judge_scores. Both are
reconstructed here from the MathDial test split, matched by POSITION (not
qid, since qids repeat across dialogues) - the same technique
scripts/significance_analysis.py and scripts/effect_sizes.py use, since
run_evaluation.py iterates test_examples in one fixed, deterministic order
and writes exactly one JSONL line per example in that order, including
failed ones.

Meant to be run manually (needs ANTHROPIC_API_KEY):

    .venv313\\Scripts\\python.exe scripts/run_second_judge.py
"""

import json
import sys
from pathlib import Path

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.evaluation.judge import JUDGE_DIMENSIONS, PedagogicalQualityJudge
from socratic_hint.evaluation.run_evaluation import CONTEXT_TURNS, format_dialogue_context

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import filter_leaked_test_examples  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_run"
CONDITIONS = ["prompted_only", "finetuned_without_state", "finetuned_with_state"]
SECOND_JUDGE_MODEL = "claude-opus-5"


def load_records_in_order(path):
    """All records in file order, including failed ones (for position alignment)."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("record_type") == "adaptivity_batch":
                continue
            records.append(d)
    return records


def cohen_kappa_like_agreement(a: list[int], b: list[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / (var_a * var_b) ** 0.5


def main():
    print("Loading MathDial test split for problem/context reconstruction...")
    test_examples = load_mathdial("test")
    train_qids = {ex.qid for ex in load_mathdial("train")}
    train_qids |= {ex.qid for ex in load_mathdial("validation")}
    test_examples = filter_leaked_test_examples(test_examples, train_qids)

    judge2 = PedagogicalQualityJudge(model=SECOND_JUDGE_MODEL)
    out_dir = RESULTS_DIR.parent / "second_judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_orig = {dim: [] for dim in JUDGE_DIMENSIONS}
    all_new = {dim: [] for dim in JUDGE_DIMENSIONS}

    for cond in CONDITIONS:
        path = RESULTS_DIR / f"{cond}.jsonl"
        records = load_records_in_order(path)
        assert len(records) == len(test_examples), (
            f"{cond}: {len(records)} records vs {len(test_examples)} test examples "
            "- position alignment would be wrong, aborting"
        )
        out_path = out_dir / f"{cond}.jsonl"
        print(f"\n=== {cond}: {len(records)} hints to re-score with {SECOND_JUDGE_MODEL} ===")

        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, (r, example) in enumerate(zip(records, test_examples)):
                if "error" in r:
                    continue
                seed_turns = example.turns[:CONTEXT_TURNS]
                dialogue_context = format_dialogue_context(seed_turns)
                try:
                    score = judge2.score(
                        problem=example.problem,
                        dialogue_context=dialogue_context,
                        hint=r["first_hint"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{i+1}/{len(records)}] FAILED: {exc}")
                    continue

                for dim in JUDGE_DIMENSIONS:
                    all_orig[dim].append(r["judge_scores"][dim])
                    all_new[dim].append(score.scores[dim])

                out_f.write(
                    json.dumps(
                        {
                            "qid": r["qid"],
                            "condition": cond,
                            "orig_scores": r["judge_scores"],
                            "second_judge_scores": score.scores,
                            "second_judge_rationale": score.rationale,
                        }
                    )
                    + "\n"
                )
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(records)}]")

    print("\n=== Inter-rater agreement (Claude Sonnet 5 vs Claude Opus 5) ===")
    for dim in JUDGE_DIMENSIONS:
        agree = cohen_kappa_like_agreement(all_orig[dim], all_new[dim])
        corr = pearson([float(x) for x in all_orig[dim]], [float(x) for x in all_new[dim]])
        print(f"{dim}: exact-match rate={agree:.3f}, Pearson r={corr:.3f}, n={len(all_orig[dim])}")


if __name__ == "__main__":
    main()
