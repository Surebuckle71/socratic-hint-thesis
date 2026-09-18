"""
Deterministic-decoding version of the same-history counterfactual diagnostic.

With sampling, two hints differ almost always even for identical inputs, so
exact-text inequality cannot separate a state effect from decoding noise. Here
generation is GREEDY (do_sample=False), so decoding noise is removed: the same
prompt must give the same hint. For each dialogue the history is held fixed
and only the injected state vector changes (0.2 vs 0.8 on every subskill,
exactly as in run_counterfactual_adaptivity.py). A third generation repeats
the low state to verify determinism. Any difference between the low- and
high-state hints is then caused by the state and nothing else.

No API calls; local GPU only.

    .venv313\\Scripts\\python.exe scripts/run_greedy_counterfactual.py \\
        checkpoints/run1 --results-dir results/counterfactual/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR, format_prompt
from socratic_hint.output_format import SUBSKILLS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import build_adaptivity_pairs, filter_leaked_test_examples  # noqa: E402

LOW_STATE = {s: 0.2 for s in SUBSKILLS}
HIGH_STATE = {s: 0.8 for s in SUBSKILLS}


def greedy_hint(backend, history, problem, state) -> str:
    state_str = ", ".join(f"{k}={v:.2f}" for k, v in state.items())
    prompt = (
        format_prompt(history, problem, suppress_state=False)
        + PROMPT_COMPLETION_SEPARATOR
        + f"State: {state_str}\nHint:"
    )
    inputs = backend.tokenizer(prompt, return_tensors="pt").to(backend.model.device)
    with torch.no_grad():
        out = backend.model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    raw = backend.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return backend._extract_forced_hint(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", type=Path)
    ap.add_argument("--pair-limit", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    from socratic_hint.backends.finetuned import FinetunedBackend

    train_qids = {ex.qid for ex in load_mathdial("train")} | {ex.qid for ex in load_mathdial("validation")}
    test = filter_leaked_test_examples(load_mathdial("test"), train_qids)
    pairs, skipped = build_adaptivity_pairs(test)
    if args.pair_limit:
        pairs = pairs[: args.pair_limit]
    print(f"{len(pairs)} pairs ({skipped} skipped)")

    backend = FinetunedBackend(checkpoint_dir=args.checkpoint_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "greedy_counterfactual.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, pair in enumerate(pairs):
            h = pair.low_mastery_history
            low = greedy_hint(backend, h, pair.problem, LOW_STATE)
            high = greedy_hint(backend, h, pair.problem, HIGH_STATE)
            control = greedy_hint(backend, h, pair.problem, LOW_STATE)
            r = {"hint_low": low, "hint_high": high, "hint_control": control,
                 "differs_between_states": low.strip() != high.strip(),
                 "differs_noise_control": low.strip() != control.strip()}
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"[{i+1}/{len(pairs)}] differs={r['differs_between_states']} control_differs={r['differs_noise_control']}", flush=True)

    n = len(rows)
    summary = {"n": n, "differ_rate": sum(r["differs_between_states"] for r in rows) / n,
               "determinism_violations": sum(r["differs_noise_control"] for r in rows)}
    (args.results_dir / "greedy_counterfactual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
