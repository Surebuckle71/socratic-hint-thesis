"""
Semantic scoring of the placebo counterfactual hints (results/placebo/), with per-pair verdicts saved.

Run for the thesis with Claude Sonnet 5 only (the placebo hints were not scored by Opus 5, to save API cost).

Same judge prompt as scripts/semantic_adaptivity.py. For each of the 191 pairs it judges:
  placebo, sampled: (0.48 hint vs 0.52 hint) and (0.48 hint vs its resample)   -> placebo net rate
  placebo, greedy:  (0.48 hint vs 0.52 hint)
  real, greedy:     (0.20 hint vs 0.80 hint), re-scored so that a paired test against the placebo
                    is possible (the first greedy scoring stored only the summary rate)
Every verdict is written per pair, so the real and placebo results can be compared pair by pair.
The tokens of every call are logged, and the run refuses to start above --max-calls.

    .venv313\\Scripts\\python.exe scripts/semantic_placebo.py [--model claude-opus-5] [--limit 20]
"""

import argparse
import json
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

ROOT = Path(__file__).resolve().parent.parent / "results"
D = ROOT / "placebo"
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
    ap.add_argument("--max-calls", type=int, default=900)
    args = ap.parse_args()
    client = Anthropic(api_key=require_api_key())
    samp = [json.loads(l) for l in open(D / "placebo_sampled.jsonl", encoding="utf-8")]
    greedy = [json.loads(l) for l in open(D / "placebo_greedy.jsonl", encoding="utf-8")]
    real_greedy = [json.loads(l) for l in open(ROOT / "counterfactual" / "greedy_counterfactual.jsonl", encoding="utf-8")]
    if args.limit:
        samp, greedy, real_greedy = samp[: args.limit], greedy[: args.limit], real_greedy[: args.limit]
    planned = 2 * len(samp) + len(greedy) + len(real_greedy)
    assert planned <= args.max_calls, f"{planned} calls planned, above --max-calls {args.max_calls}"
    print(f"model {args.model}: {planned} judgments planned")

    def safe(fn):
        def run(r):
            try:
                return fn(r)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"{type(exc).__name__}: {exc}"}
        return run

    work_s = safe(lambda r: {"between": judge(client, r["hint_low"], r["hint_high"], args.model),
                             "noise": judge(client, r["hint_low"], r["hint_control"], args.model)})
    work_g = safe(lambda r: {"between": judge(client, r["hint_low"], r["hint_high"], args.model)})

    with ThreadPoolExecutor(max_workers=8) as pool:
        rs = list(pool.map(work_s, samp))
        rg = list(pool.map(work_g, greedy))
        rr = list(pool.map(work_g, real_greedy))

    suffix = "" if args.model == DEFAULT_MODEL else "_" + args.model
    for name, rows in (("semantic_placebo_sampled_pairs", rs), ("semantic_placebo_greedy_pairs", rg),
                       ("semantic_real_greedy_pairs", rr)):
        with open(D / f"{name}{suffix}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    ok = [r for r in rs if "error" not in r]
    n = len(ok)
    differ = sum(r["between"] for r in ok) / n
    noise = sum(r["noise"] for r in ok) / n
    rows = [{"sem_differs_between_states": r["between"], "sem_differs_noise": r["noise"]} for r in ok]
    gok = [r["between"] for r in rg if "error" not in r]
    rok = [r["between"] for r in rr if "error" not in r]
    out = {
        "model": args.model, "placebo_states": [0.48, 0.52],
        "sampled": {"n": n, "failed": len(rs) - n, "semantic_differ_rate": differ, "semantic_noise_rate": noise,
                    "semantic_net_rate": differ - noise, "mcnemar": mcnemar(rows)},
        "greedy_placebo": {"n": len(gok), "failed": len(rg) - len(gok), "semantic_differ_rate": sum(gok) / len(gok)},
        "greedy_real": {"n": len(rok), "failed": len(rr) - len(rok), "semantic_differ_rate": sum(rok) / len(rok)},
        "tokens": TOK,
    }
    (D / f"semantic_placebo_summary{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
