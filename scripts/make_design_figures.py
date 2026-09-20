"""
Redraws the two design diagrams of the methodology chapter with larger text:

  conditions.pdf   the experimental conditions and the shared evaluation protocol
  pipeline.pdf     the state-estimation-to-hint pipeline for a single dialogue turn

    python scripts/make_design_figures.py OUTPUT_DIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
FS = 10


def box(ax, x, y, w, h, text, fc, ec, fs=FS, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.5, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333333", ls=ls, shrinkA=0, shrinkB=0))


# ---------------- Figure: conditions ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.1))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
ax.add_patch(Rectangle((3.5, 2.5), 6.3, 2.1, fill=False, ec="#555555", lw=1.4, ls="--"))
ax.text(6.65, 4.78, "identical fine-tuned weights (ablation pair)", ha="center", va="bottom", fontsize=FS - 1, style="italic", color="#444444")
box(ax, 0.2, 2.75, 3.0, 1.6, "Condition 1\nPrompted baseline\nClaude Sonnet 5\n(no fine-tuning)", "#E8EEF8", "#4C72B0")
box(ax, 3.7, 2.75, 2.8, 1.6, "Condition 2\nFine-tuned, 3B\nstate suppressed", "#EFEFEF", "#8A8A8A")
box(ax, 6.8, 2.75, 2.8, 1.6, "Condition 3\nFine-tuned, 3B\nstate-conditioned", "#E3F0E2", "#55A868")
for x in (1.7, 5.1, 8.2):
    arrow(ax, x, 2.7, x, 1.75)
box(ax, 0.2, 0.15, 9.4, 1.55, "Same evaluation protocol for all conditions:\nsimulated-student dialogue, pedagogical judge,\nstate-adaptivity diagnostic", "#FBE6D6", "#DD8452")
fig.savefig(OUT / "conditions.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---------------- Figure: pipeline ----------------
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")
box(ax, 0.1, 1.9, 2.3, 1.5, "Problem +\ndialogue\nhistory", "#EFEFEF", "#8A8A8A")
box(ax, 3.3, 3.2, 2.7, 1.5, "State estimation\n(Conditions 1\nand 3 only)", "#FBE6D6", "#DD8452")
box(ax, 3.3, 0.3, 2.7, 1.3, "Skipped\n(Condition 2)", "#EFEFEF", "#8A8A8A", ls="--")
box(ax, 6.9, 1.9, 3.1, 1.5, "Hint generation\n(conditioned on\nstate, if estimated)", "#E3F0E2", "#55A868")
box(ax, 10.5, 1.9, 1.4, 1.5, "Next\nhint\n(State: /\nHint:)", "#E8EEF8", "#4C72B0")
arrow(ax, 2.4, 2.9, 3.3, 3.8)
arrow(ax, 2.4, 2.4, 3.3, 1.2, ls="--")
arrow(ax, 6.0, 3.8, 6.9, 3.0)
arrow(ax, 6.0, 1.2, 6.9, 2.3, ls="--")
arrow(ax, 10.0, 2.65, 10.5, 2.65)
fig.savefig(OUT / "pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("wrote", OUT / "conditions.pdf", "and", OUT / "pipeline.pdf")
