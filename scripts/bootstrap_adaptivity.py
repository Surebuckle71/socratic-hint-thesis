"""
Paired bootstrap 95% confidence intervals for the net-adaptivity difference in
the same-history counterfactual diagnostic.

net = mean(differs under different injected state) - mean(differs under same
injected state), both outcomes logged per dialogue, so resampling dialogues
with replacement gives a paired bootstrap. Reads results/counterfactual/*.jsonl.

    python scripts/bootstrap_adaptivity.py
"""

import json
import random
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "results" / "counterfactual"
B = 10_000
SEED = 3407


def load(name):
    return [json.loads(l) for l in open(D / name, encoding="utf-8")]


def paired_bootstrap(between, noise):
    n = len(between)
    obs = sum(between) / n - sum(noise) / n
    rng = random.Random(SEED)
    stats = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(between[i] for i in idx) / n - sum(noise[i] for i in idx) / n)
    stats.sort()
    return obs, stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def main():
    out = {}
    exact = [r for r in load("counterfactual_adaptivity.jsonl")]
    out["exact_text"] = paired_bootstrap([int(r["differs_between_states"]) for r in exact],
                                         [int(r["differs_noise_control"]) for r in exact])
    for label, name in [("semantic_sonnet5", "semantic_adaptivity.jsonl"),
                        ("semantic_opus5", "semantic_adaptivity_claude-opus-5.jsonl")]:
        rows = [r for r in load(name) if "error" not in r]
        out[label] = paired_bootstrap([int(r["sem_differs_between_states"]) for r in rows],
                                      [int(r["sem_differs_noise"]) for r in rows]) + (len(rows),)
    for k, v in out.items():
        n = f", n={v[3]}" if len(v) > 3 else f", n={len(exact)}"
        print(f"{k}: net={v[0]:.4f}  95% CI [{v[1]:.4f}, {v[2]:.4f}]{n}")
    (D / "bootstrap_adaptivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
