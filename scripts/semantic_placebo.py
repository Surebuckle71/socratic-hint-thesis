"""
Semantic scoring of the placebo counterfactual hints (results/placebo/).

Run for the thesis with Claude Sonnet 5 only (the placebo hints were not scored by Opus 5, to save API cost).

Same judge prompt as scripts/semantic_adaptivity.py. For each of the 191 pairs it judges:
  sampled: (0.48 hint vs 0.52 hint) and (0.48 hint vs its resample)  -> placebo net rate
  greedy:  (0.48 hint vs 0.52 hint)                                  -> placebo differ rate
It logs the input and output tokens of every call and stops early once --max-calls is reached,
so the spend can be checked before a second judge model is run.

    .venv313\\Scripts\\python.exe scripts/semantic_placebo.py [--model claude-opus-5] [--limit 20]
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_adaptivity import mcnemar  # noqa: E402
from socratic_hint.llm_config import (  # noqa: E402
    DEFAULT_MODEL,
    MIN_MAX_TOKENS,
    extract_text,
    require_api_key,
    thinking_kwargs_for_model,
)

D = Path(__file__).resolve().parent.parent / "results" / "placebo"
TOK = {"in": 0, "out": 0, "calls": 0}


def judge(client, a, b, model):
    prompt = (
        "Two tutoring hints for the same math problem follow. Ignore wording, "
        "phrasing, and sentence order. Decide whether they carry the SAME "
        "pedagogical content and strategy (they point the student at the same "
        "step, ask essentially the same question or make essentially the same "
        "move), or DIFFERENT content or strategy (they target different steps, "
        "one gives away or confirms something the other does not, or one asks "
        "a materially different question).\n\n"
        f"Hint A: {a}\n\nHint B: {b}\n\n"
        "Respond with exactly one word: SAME or DIFFERENT."
    )
    response = client.messages.create(
        model=model, max_tokens=MIN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}], **thinking_kwargs_for_model(model),
    )
    TOK["in"] += response.usage.input_tokens
    TOK["out"] += response.usage.output_tokens
    TOK["calls"] += 1
    verdict = extract_text(response).strip().upper()
    if verdict.startswith("DIFFERENT"):
        return True
    if verdict.startswith("SAME"):
        return False
    raise ValueError(f"Unparseable verdict: {verdict!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=700)
    args = ap.parse_args()
    client = Anthropic(api_key=require_api_key())
    samp = [json.loads(l) for l in open(D / "placebo_sampled.jsonl", encoding="utf-8")]
    greedy = [json.loads(l) for l in open(D / "placebo_greedy.jsonl", encoding="utf-8")]
    if args.limit:
        samp, greedy = samp[: args.limit], greedy[: args.limit]
    planned = 2 * len(samp) + len(greedy)
    assert planned <= args.max_calls, f"{planned} calls planned, above --max-calls {args.max_calls}"
    print(f"model {args.model}: {planned} judgments planned")

    def work_s(r):
        try:
            return {"between": judge(client, r["hint_low"], r["hint_high"], args.model),
                    "noise": judge(client, r["hint_low"], r["hint_control"], args.model)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def work_g(r):
        try:
            return judge(client, r["hint_low"], r["hint_high"], args.model)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        rs = list(pool.map(work_s, samp))
        rg = list(pool.map(work_g, greedy))
    ok = [r for r in rs if "error" not in r]
    rows = [{"sem_differs_between_states": r["between"], "sem_differs_noise": r["noise"]} for r in ok]
    n = len(ok)
    differ = sum(r["between"] for r in ok) / n
    noise = sum(r["noise"] for r in ok) / n
    gok = [x for x in rg if isinstance(x, bool)]
    out = {
        "model": args.model, "placebo_states": [0.48, 0.52],
        "sampled": {"n": n, "failed": len(rs) - n, "semantic_differ_rate": differ, "semantic_noise_rate": noise,
                    "semantic_net_rate": differ - noise, "mcnemar": mcnemar(rows)},
        "greedy": {"n": len(gok), "failed": len(rg) - len(gok), "semantic_differ_rate_greedy": sum(gok) / len(gok)},
        "tokens": TOK,
    }
    suffix = "" if args.model == DEFAULT_MODEL else "_" + args.model
    (D / f"semantic_placebo_summary{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
