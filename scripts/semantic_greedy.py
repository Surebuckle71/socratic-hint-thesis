"""
Semantic re-scoring of the greedy-decoding counterfactual hints
(results/counterfactual/greedy_counterfactual.jsonl). Under greedy decoding
there is no sampling noise, so the fraction of dialogues whose low-state and
high-state hints differ in pedagogical content is attributable to the injected
state. Compare with the sampled-decoding same-state semantic noise rates
(67.0% Sonnet 5, 71.7% Opus 5) in semantic_adaptivity_summary*.json.

    .venv313\Scripts\python.exe scripts/semantic_greedy.py [--model claude-opus-5]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_adaptivity import semantically_different  # noqa: E402
from socratic_hint.llm_config import DEFAULT_MODEL, require_api_key  # noqa: E402

D = Path(__file__).resolve().parent.parent / "results" / "counterfactual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(D / "greedy_counterfactual.jsonl", encoding="utf-8")]
    client = Anthropic(api_key=require_api_key())

    def work(r):
        try:
            return semantically_different(client, r["hint_low"], r["hint_high"], args.model)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(work, rows))
    ok = [x for x in res if isinstance(x, bool)]
    out = {"model": args.model, "n": len(ok), "failed": len(res) - len(ok),
           "semantic_differ_rate_greedy": sum(ok) / len(ok)}
    suffix = "" if args.model == DEFAULT_MODEL else "_" + args.model
    (D / f"semantic_greedy_summary{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
