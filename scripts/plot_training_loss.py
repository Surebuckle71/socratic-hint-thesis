"""
Plot the training and validation loss recorded over the full QLoRA run.

Reads log_history from the final checkpoint's trainer_state.json (the run was resumed six times,
and the history is carried across resumes) and marks two points: the validation-loss minimum
(step 1,600, not retained) and the checkpoint that was actually evaluated (final step 3,371).

    python scripts/plot_training_loss.py [output.pdf]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "checkpoints" / "run1" / "checkpoint-3371" / "trainer_state.json"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "training_loss.pdf"
    hist = json.load(open(STATE, encoding="utf-8"))["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    best = min(ev, key=lambda x: x[1])
    final = ev[-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(*zip(*tr), color="#b8c4e0", lw=0.8, label="Training loss")
    ax.plot(*zip(*ev), color="#d1495b", lw=1.6, marker="o", ms=3, label="Validation loss")
    ax.axvline(best[0], color="0.4", lw=0.8, ls=":")
    ax.plot(*best, "o", mfc="white", mec="black", ms=7, mew=1.4, ls="none",
            label=f"Validation-loss minimum (step {best[0]:,}, not retained)")
    ax.plot(*final, "s", color="black", ms=6.5, ls="none", label=f"Evaluated checkpoint (final step {final[0]:,})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(0.15, 2.9)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}; minimum {best}, final {final}")


if __name__ == "__main__":
    main()
