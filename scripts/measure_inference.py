"""
Measure inference-time GPU memory, latency, and throughput of the fine-tuned
checkpoint on real held-out dialogue prompts (the consumer-hardware claim).

Loads the checkpoint the same way FinetunedBackend does, then generates hints
for N leakage-filtered test dialogues in each output format (state suppressed,
state-conditioned), using the same prompt construction and max_new_tokens=512
as the evaluation. Records:
  - GPU memory after loading, and PyTorch peak allocated/reserved memory
    during generation;
  - driver-level peak memory.used and peak temperature sampled with nvidia-smi
    (includes anything else using the GPU, e.g. the desktop);
  - per-hint latency, generated tokens, and tokens/second;
  - on-disk sizes of the adapter and the checkpoint folder.
This measures INFERENCE only, not training.

    .venv313\\Scripts\\python.exe scripts/measure_inference.py checkpoints/run1
"""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from socratic_hint.data.mathdial_loader import load_mathdial
from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR, format_prompt

OUT = Path(__file__).resolve().parent.parent / "results" / "inference_measurement.json"
HISTORY_TURNS = 6
MB = 1024 ** 2


def smi():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    used, temp, power = [float(x) for x in r.stdout.strip().split(",")]
    return used, temp, power


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak_used = self.peak_temp = self.peak_power = 0.0

    def run(self):
        while not self.stop:
            used, temp, power = smi()
            self.peak_used = max(self.peak_used, used)
            self.peak_temp = max(self.peak_temp, temp)
            self.peak_power = max(self.peak_power, power)
            time.sleep(0.5)


def summarize(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    train_qids = {ex.qid for ex in load_mathdial("train")} | {ex.qid for ex in load_mathdial("validation")}
    test = [ex for ex in load_mathdial("test") if ex.qid not in train_qids][: args.n + args.warmup]
    assert len(test) == args.n + args.warmup

    props = torch.cuda.get_device_properties(0)
    baseline_used, baseline_temp, _ = smi()
    torch.cuda.reset_peak_memory_stats()
    sampler = Sampler(); sampler.start()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.checkpoint), max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    torch.cuda.synchronize()
    after_load = {
        "allocated_mb": torch.cuda.memory_allocated() / MB,
        "reserved_mb": torch.cuda.memory_reserved() / MB,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
    }

    results = {}
    for name, suppress in [("state_suppressed", True), ("state_conditioned", False)]:
        rows = []
        torch.cuda.reset_peak_memory_stats()
        for i, ex in enumerate(test):
            n = HISTORY_TURNS if len(ex.turns) >= HISTORY_TURNS else len(ex.turns) - (len(ex.turns) % 2)
            prompt = format_prompt(ex.turns[:n], ex.problem, suppress_state=suppress) + PROMPT_COMPLETION_SEPARATOR
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=512)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if i < args.warmup:
                continue
            new = out.shape[1] - inputs["input_ids"].shape[1]
            rows.append({"prompt_tokens": inputs["input_ids"].shape[1], "new_tokens": new,
                         "seconds": dt, "tokens_per_second": new / dt})
        results[name] = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
            "prompt_tokens": summarize([r["prompt_tokens"] for r in rows]),
            "new_tokens": summarize([r["new_tokens"] for r in rows]),
            "seconds_per_hint": summarize([r["seconds"] for r in rows]),
            "tokens_per_second": summarize([r["tokens_per_second"] for r in rows]),
            "hit_512_token_limit": sum(1 for r in rows if r["new_tokens"] >= 512),
        }
        print(name, json.dumps(results[name], indent=1))

    sampler.stop = True; time.sleep(1)
    ckpt = args.checkpoint
    summary = {
        "gpu": props.name, "gpu_total_mb": props.total_memory / MB,
        "history_turns": HISTORY_TURNS, "n_dialogues": args.n, "warmup": args.warmup,
        "smi_baseline_used_mb_before_load": baseline_used,
        "smi_peak_used_mb": sampler.peak_used, "smi_peak_temp_c": sampler.peak_temp,
        "smi_baseline_temp_c": baseline_temp, "smi_peak_power_w": sampler.peak_power,
        "after_load": after_load, "generation": results,
        "adapter_size_mb": (ckpt / "adapter_model.safetensors").stat().st_size / MB,
        "torch": torch.__version__,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
