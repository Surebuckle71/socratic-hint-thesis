"""Run a real QLoRA fine-tuning job on the MathDial train split.

Meant to be run MANUALLY on a machine with a CUDA GPU, using the training
environment (.venv313 + the `train` extra). It is deliberately not part of the
automated test suite — see README.md.

    .venv313\\Scripts\\python.exe scripts/run_training.py --output-dir checkpoints/run1

The GPU smoke test in tests/training/test_train_qlora.py covers the same
`train()` entry point with 2 examples and 2 steps.
"""

import argparse
import random
import sys
from pathlib import Path

# Unsloth must be imported before trl/transformers/peft so its patches apply;
# importing train_qlora first guarantees that ordering.
from socratic_hint.training.train_qlora import TrainConfig, train

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.data.training_examples import build_training_examples

# Cap on validation examples used for the periodic eval-loss signal. The full
# validation split would make every eval pass slow enough to dominate the run;
# a few hundred examples is plenty for a loss curve.
MAX_EVAL_EXAMPLES = 256

# Fixed seed for the validation-subset shuffle, so the eval-loss curve is
# comparable across runs.
EVAL_SUBSET_SEED = 3407

# Roughly how many passes over the training data to make.
TARGET_EPOCHS = 2
# Guard rails on the derived step count, so an unexpected dataset size cannot
# produce a run that is trivially short or absurdly long.
MIN_STEPS = 200
# `build_training_examples` now emits two variants per tutor turn (a
# state-conditioned one and a state-suppressed one — see
# src/socratic_hint/data/training_examples.py), roughly doubling the example
# count from what MAX_STEPS was originally sized for. Left at the old 2000,
# TARGET_EPOCHS=2 would silently resolve to ~1.19 real epochs, meaning
# condition 3 (the thesis's headline, state-conditioned condition) would see
# its own prompt format ~40% less than the design intended. Raised to give
# headroom above the ~3371 steps that 2 real epochs over the full (post-fix)
# 2035-dialogue train split actually needs; if the train split size changes
# materially, re-derive this number rather than trusting it blindly.
MAX_STEPS = 3500


def build_examples(split: str):
    dialogues = load_mathdial(split)
    examples = []
    for dialogue in dialogues:
        examples.extend(build_training_examples(dialogue))
    return dialogues, examples


def derive_max_steps(num_examples: int, effective_batch_size: int) -> int:
    """Steps for ~TARGET_EPOCHS passes, clamped to a sane range.

    One optimizer step consumes `per_device_train_batch_size *
    gradient_accumulation_steps` examples, so epochs -> steps is
    (examples * epochs) / effective_batch_size.
    """
    steps = (num_examples * TARGET_EPOCHS) // max(effective_batch_size, 1)
    return max(MIN_STEPS, min(steps, MAX_STEPS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the trained LoRA adapter checkpoint to.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Training steps. Defaults to ~%d epochs over the train split "
        "(clamped to [%d, %d])." % (TARGET_EPOCHS, MIN_STEPS, MAX_STEPS),
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the validation-loss signal (faster, but flies blind).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint-N under --output-dir (auto-detected) "
        "instead of starting fresh. --max-steps, if given, is still the target "
        "TOTAL step count, not additional steps from here.",
    )
    args = parser.parse_args()

    print("Loading MathDial train split...")
    train_dialogues, train_examples = build_examples("train")
    print(f"  {len(train_dialogues)} dialogues -> {len(train_examples)} training examples")

    if not train_examples:
        print("ERROR: no training examples built; aborting.", file=sys.stderr)
        return 1

    eval_examples = None
    if not args.no_eval:
        print("Loading MathDial validation split...")
        eval_dialogues, eval_examples = build_examples("validation")
        # Shuffle before slicing. build_examples emits in qid order, so a plain
        # head-slice would compute the whole validation-loss curve on a narrow
        # band of qid-adjacent problems. Seeded, so the subset is reproducible.
        random.Random(EVAL_SUBSET_SEED).shuffle(eval_examples)
        eval_examples = eval_examples[:MAX_EVAL_EXAMPLES]
        print(
            f"  {len(eval_dialogues)} dialogues -> {len(eval_examples)} eval examples "
            f"(capped at {MAX_EVAL_EXAMPLES})"
        )

    # Build a config first so the derived step count uses the real batch sizes.
    base = TrainConfig(output_dir=args.output_dir, max_steps=1)
    effective_batch = base.per_device_train_batch_size * base.gradient_accumulation_steps
    # `is not None`, not truthiness: `--max-steps 0` is falsy, so `or` silently
    # substituted the derived default instead of honouring what was asked.
    if args.max_steps is not None:
        if args.max_steps < 1:
            print(
                f"ERROR: --max-steps must be >= 1 (got {args.max_steps}).",
                file=sys.stderr,
            )
            return 1
        max_steps = args.max_steps
    else:
        max_steps = derive_max_steps(len(train_examples), effective_batch)

    config = TrainConfig(output_dir=args.output_dir, max_steps=max_steps)

    print(
        f"Training: max_steps={config.max_steps}, effective batch={effective_batch} "
        f"({config.per_device_train_batch_size} x {config.gradient_accumulation_steps} accum), "
        f"lr={config.learning_rate}, checkpoint every {config.save_steps} steps"
    )

    output_dir = train(
        train_examples,
        config,
        eval_examples=eval_examples,
        resume_from_checkpoint=True if args.resume else None,
    )
    print(f"Done. Checkpoint written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
