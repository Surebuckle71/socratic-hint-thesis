"""
Semantic re-scoring of the same-history counterfactual adaptivity data.

The original diagnostic (and scripts/run_counterfactual_adaptivity.py) counts
two hints as "different" on exact-text inequality, which stochastic decoding
makes true almost always (differ and noise rates both ~99%). This script
replaces exact-text inequality with a semantic one: an LLM judge is asked
whether two hints carry the same pedagogical content and strategy, ignoring
wording. The same noise-floor logic then applies:

    semantic net rate = P(semantically differ | different injected state)
                      - P(semantically differ | same injected state)

No new hints are generated: it reads results/counterfactual/
counterfactual_adaptivity.jsonl. API calls only (no GPU). Note the stored
records do not include the problem statement or dialogue history, so the judge
compares the two hints on their own, which is a stated limitation.

    .venv313\\Scripts\\python.exe scripts/semantic_adaptivity.py
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic

from socratic_hint.llm_config import (
    DEFAULT_MODEL,
    MIN_MAX_TOKENS,
    extract_text,
    require_api_key,
    thinking_kwargs_for_model,
)

IN_PATH = Path(__file__).resolve().parent.parent / "results" / "counterfactual" / "counterfactual_adaptivity.jsonl"
OUT_PATH = IN_PATH.parent / "semantic_adaptivity.jsonl"
SUMMARY_PATH = IN_PATH.parent / "semantic_adaptivity_summary.json"


def semantically_different(client: Anthropic, hint_a: str, hint_b: str, model: str = DEFAULT_MODEL) -> bool:
    prompt = (
        "Two tutoring hints for the same math problem follow. Ignore wording, "
        "phrasing, and sentence order. Decide whether they carry the SAME "
        "pedagogical content and strategy (they point the student at the same "
        "step, ask essentially the same question or make essentially the same "
        "move), or DIFFERENT content or strategy (they target different steps, "
        "one gives away or confirms something the other does not, or one asks "
        "a materially different question).\n\n"
        f"Hint A: {hint_a}\n\nHint B: {hint_b}\n\n"
        "Respond with exactly one word: SAME or DIFFERENT."
    )
    response = client.messages.create(
        model=model,
        max_tokens=MIN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        **thinking_kwargs_for_model(model),
    )
    verdict = extract_text(response).strip().upper()
    if verdict.startswith("DIFFERENT"):
        return True
    if verdict.startswith("SAME"):
        return False
    raise ValueError(f"Unparseable verdict: {verdict!r}")


def mcnemar(rows: list[dict]) -> dict:
    b = sum(1 for r in rows if r["sem_differs_between_states"] and not r["sem_differs_noise"])
    c = sum(1 for r in rows if not r["sem_differs_between_states"] and r["sem_differs_noise"])
    a = sum(1 for r in rows if r["sem_differs_between_states"] and r["sem_differs_noise"])
    d = len(rows) - a - b - c
    nd = b + c
    if nd == 0:
        return {"a": a, "b": b, "c": c, "d": d, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / nd
    return {"a": a, "b": b, "c": c, "d": d, "chi2": chi2, "p_value": math.erfc((chi2 / 2) ** 0.5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    suffix = "" if args.model == DEFAULT_MODEL else "_" + args.model
    out_path = OUT_PATH.with_name(f"semantic_adaptivity{suffix}.jsonl")
    summary_path = SUMMARY_PATH.with_name(f"semantic_adaptivity_summary{suffix}.json")
    client = Anthropic(api_key=require_api_key())
    records = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    print(f"{len(records)} counterfactual pairs; judging {2 * len(records)} hint pairs semantically")

    def work(rec):
        try:
            between = semantically_different(client, rec["hint_low"], rec["hint_high"], args.model)
            noise = semantically_different(client, rec["hint_low"], rec["hint_control"], args.model)
            return {
                "hint_low": rec["hint_low"],
                "hint_high": rec["hint_high"],
                "hint_control": rec["hint_control"],
                "sem_differs_between_states": between,
                "sem_differs_noise": noise,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, records))

    ok = [r for r in results if "error" not in r]
    failed = len(results) - len(ok)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(ok)
    differ = sum(r["sem_differs_between_states"] for r in ok) / n
    noise = sum(r["sem_differs_noise"] for r in ok) / n
    mc = mcnemar(ok)
    summary = {
        "n": n, "failed": failed, "semantic_differ_rate": differ,
        "semantic_noise_rate": noise, "semantic_net_rate": differ - noise, "mcnemar": mc,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
