"""
Measure peak GPU memory of the real QLoRA training configuration.

Training memory was not logged during the original run. This runs the same
`train()` code path and TrainConfig (batch 2 x 8 accumulation, max_seq_length
2048, 8-bit AdamW, LoRA r=16) for a few optimizer steps on the LONGEST
training examples, so the peak reflects a worst-case batch rather than an
average one. It also reports the token-length distribution of the whole
training set. Output goes to a temporary directory; checkpoints/ is not
touched.

    powershell -File scripts/run_with_msvc_env.ps1 -Command "python scripts/measure_training_memory.py"
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training import build_examples  # noqa: E402

from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR
from socratic_hint.training.train_qlora import BASE_MODEL, TrainConfig, train

OUT = Path(__file__).resolve().parent.parent / "results" / "training_memory_measurement.json"
STEPS = 2
MB = 1024 ** 2


def smi_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    used, temp = [float(x) for x in r.stdout.strip().split(",")]
    return used, temp


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak_used = self.peak_temp = 0.0

    def run(self):
        while not self.stop:
            used, temp = smi_used()
            self.peak_used = max(self.peak_used, used)
            self.peak_temp = max(self.peak_temp, temp)
            time.sleep(0.5)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    _, examples = build_examples("train")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    lengths = [len(tok(f"{e.prompt}{PROMPT_COMPLETION_SEPARATOR}{e.completion}")["input_ids"]) for e in examples]
    order = sorted(range(len(examples)), key=lambda i: -lengths[i])
    cfg_probe = TrainConfig(output_dir=Path("."), max_steps=STEPS)
    per_step = cfg_probe.per_device_train_batch_size * cfg_probe.gradient_accumulation_steps
    worst = [examples[i] for i in order[: STEPS * per_step]]
    worst_lengths = [lengths[i] for i in order[: STEPS * per_step]]

    baseline_used, baseline_temp = smi_used()
    props = torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()
    sampler = Sampler(); sampler.start()
    with tempfile.TemporaryDirectory() as tmp:
        config = TrainConfig(output_dir=Path(tmp) / "mem_probe", max_steps=STEPS, save_steps=10_000)
        t0 = time.perf_counter()
        train(worst, config)
        seconds = time.perf_counter() - t0
    sampler.stop = True; time.sleep(1)

    summary = {
        "gpu": props.name, "gpu_total_mb": props.total_memory / MB,
        "optimizer_steps": STEPS, "examples_used": len(worst),
        "per_device_batch": cfg_probe.per_device_train_batch_size,
        "grad_accum": cfg_probe.gradient_accumulation_steps, "max_seq_length": cfg_probe.max_seq_length,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
        "smi_baseline_used_mb_before": baseline_used, "smi_peak_used_mb": sampler.peak_used,
        "smi_peak_temp_c": sampler.peak_temp, "smi_baseline_temp_c": baseline_temp,
        "seconds_total_including_model_load": seconds,
        "train_set_examples": len(examples),
        "train_token_lengths": {"median": pct(lengths, 0.5), "p95": pct(lengths, 0.95), "p99": pct(lengths, 0.99),
                                "max": max(lengths), "over_2048": sum(1 for x in lengths if x > 2048)},
        "worst_case_batch_token_lengths": {"min": min(worst_lengths), "median": pct(worst_lengths, 0.5), "max": max(worst_lengths)},
        "torch": torch.__version__,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
