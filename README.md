# socratic-hint-thesis

Research code for an M.Sc. thesis on **LLM-guided Socratic hint generation conditioned on an
explicit student-state estimate**. A model is asked to first estimate the student's mastery of
several subskills, then produce the next tutoring hint conditioned on that estimate. The central
claim — that state conditioning improves hint quality — is tested by an ablation that suppresses
the state estimate at inference time (`suppress_state=True`) and compares against state-conditioned
generation **using the same model weights**.

The code has two hint backends behind one `HintBackend` interface
(`src/socratic_hint/backends/`): a prompted Anthropic-API baseline and a locally fine-tuned
QLoRA adapter over Qwen2.5-3B-Instruct. A three-metric evaluation harness
(`src/socratic_hint/evaluation/`) compares three conditions — prompted-only,
fine-tuned-without-state, fine-tuned-with-state — on LLM-judged pedagogical quality, simulated
student convergence, and a state-adaptivity diagnostic.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/socratic_hint/data/` | MathDial loading, derived state labels, training-example construction |
| `src/socratic_hint/backends/` | `PromptedBackend` (Anthropic API), `FinetunedBackend` (local QLoRA) |
| `src/socratic_hint/training/` | QLoRA fine-tuning entry point (`train`) |
| `src/socratic_hint/evaluation/` | Judge, simulated student, state-adaptivity diagnostic, orchestrator |
| `src/socratic_hint/llm_config.py` | Shared model id and request shape for every real API call |
| `scripts/` | Runnable end-to-end training and evaluation scripts |
| `tests/` | Unit tests (fast, mocked) plus one `-m gpu` smoke test |

## Environments

There are **two** environments, because the training stack cannot live in the base install.

### Base environment — everything except training

```bash
pip install -e .
```

This gets you the data loaders, output format, prompted backend, and the whole evaluation
harness. The fast test suite runs here. Set `ANTHROPIC_API_KEY` for anything that makes real
API calls.

### Training environment — `.venv313`

Fine-tuning needs `unsloth`, which pulls in a CUDA-enabled `torch`, `bitsandbytes`, `peft`, and
`trl`. Those are kept in a **separate Python 3.13 virtualenv** at `.venv313` so a plain
`pip install -e .` stays lightweight and installable on a machine without a GPU.

```bash
py -3.13 -m venv .venv313
.venv313\Scripts\python.exe -m pip install -e ".[train]"
```

Two platform notes for native Windows:

- **CUDA torch.** `pip` resolves a CPU-only `torch` by default. Install the CUDA build that
  matches your driver from the PyTorch index *before* installing the `train` extra, or the
  4-bit loading path will not find a GPU.
- **MSVC / Triton.** Unsloth uses Triton, which JIT-compiles a CUDA driver stub at runtime and
  therefore needs the MSVC toolchain and Windows SDK on `PATH`/`INCLUDE`/`LIB`. Rather than
  configuring that by hand, wrap GPU commands in **`scripts/run_with_msvc_env.ps1`**, which sets
  those variables and then runs your command (its header comments explain why `vcvarsall.bat`
  is bypassed on this machine):

  ```powershell
  powershell -File scripts\run_with_msvc_env.ps1 -Command "pytest tests/training -v -m gpu"
  ```

## Running tests

The fast suite is the default. `pyproject.toml` sets `addopts = "-m 'not gpu'"`, so a bare
`pytest` never tries to run GPU tests:

```bash
pytest                      # fast suite (all mocked, no GPU, no API key needed)
pytest -v                   # same, verbose
```

Tests that import `unsloth` (`tests/backends/test_finetuned.py`,
`tests/training/test_train_qlora.py`) call `pytest.importorskip("unsloth")` at module level, so
on a base install they are **skipped**, not collection errors. On `.venv313` they run normally.

The GPU tests are opt-in — an explicit `-m gpu` on the command line overrides `addopts`:

```powershell
powershell -File scripts\run_with_msvc_env.ps1 -Command ".venv313\Scripts\python.exe -m pytest tests/training/test_train_qlora.py -v -m gpu"
```

Because `tests/backends/test_finetuned.py` imports `unsloth` at module level, running the
**whole** `tests/` tree exercises that import during collection. Run the full suite under the
training environment:

```powershell
.venv313\Scripts\python.exe -m pytest -m "not gpu" -v
```

## Running the pipeline

Both scripts are **manual** — they need a real API key and/or a GPU, and are intentionally not
part of the automated test suite.

### Training

```powershell
powershell -File scripts\run_with_msvc_env.ps1 -Command ".venv313\Scripts\python.exe scripts\run_training.py --output-dir checkpoints\run1"
```

Loads the MathDial train split, builds `(prompt, completion)` pairs from every tutor turn, and
fine-tunes with QLoRA. Step count defaults to roughly two epochs over the built examples (clamped
to 200–2000); override with `--max-steps`. A validation split drives a periodic eval-loss signal —
disable it with `--no-eval`. Checkpoints are written periodically, so a crash mid-run does not
lose the whole job.

Every tutor turn yields **two** training examples, not one: the state-conditioned variant
(`State:` + `Hint:`) and a state-suppressed variant (hint only). The evaluation harness runs a
`suppress_state=True` ablation against these same weights, so the suppressed prompt format has to
be something the model was actually trained on — otherwise that condition would be testing an
unseen prompt format rather than the absence of state conditioning. On the real train split this
gives 2035 dialogues → 26,968 examples (13,484 of each variant).

Loss is computed on the **completion only**. The dataset is handed to TRL as separate
`prompt`/`completion` columns, which makes `SFTTrainer` resolve `completion_only_loss` to `True`
and mask prompt tokens out of the loss; with a flattened single `text` field, ~90% of the loss
signal went into reproducing dialogue history the model is never asked to generate.

### Evaluation

```powershell
.venv313\Scripts\python.exe scripts\run_evaluation.py checkpoints\run1 --limit 100 --results-dir results\
```

Requires `ANTHROPIC_API_KEY` (the judge and simulated student are API-backed). Runs all three
conditions and prints a comparison table across all three metrics. Use `--limit` for a cheap
pilot run before committing to the full test split. With `--results-dir`, per-example results are
written as JSONL **as the run progresses**, so a multi-hour run yields partial output rather than
nothing-until-the-end; a single malformed generation is recorded and skipped instead of aborting
the run, and the counts surface in the results table's `failed` and `failed_pairs` columns. The
file is truncated once per condition, so re-running into the same `--results-dir` replaces the
previous run rather than interleaving with it.

#### Problem-level leakage filtering

Before anything is evaluated, the script drops every test dialogue whose `qid` also appears in
the training pool (train + validation, both carved from MathDial's published `train` split).
MathDial's official split separates **dialogues**, not **problems**: an audit of the real dataset
found 80.7% of test qids also present in train, and 59.8% of test dialogues (358/599) repeating an
identical `(question, student_incorrect_solution)` pair from train. Evaluating the fine-tuned
conditions on problems they were fine-tuned on would hand them a systematic advantage in exactly
the comparison the thesis rests on. The filter removes **358 of 599** test dialogues, leaving 241
genuinely held-out ones, and prints those counts so the reduction is never silent. `load_mathdial`
itself is deliberately left alone — it still returns MathDial's official splits faithfully; the
filtering is explicit at the point where it matters.

## Interpreting the state-adaptivity metric

The reported adaptivity rate is **net of a sampling-noise floor**. Both backends decode
stochastically, so two samples of the same prompt usually differ in wording — a naive
"do the hints differ?" check would score a completely non-adaptive model at ~100%. For every pair
the diagnostic therefore also generates a second hint from the *same* history and reports

```
adaptivity = P(differ | different history) - P(differ | same history)
```

This value can be negative when noise exceeds signal. It is reported as-is rather than clamped,
because a negative value is genuine evidence against the thesis's claim.

### How the contrasted histories are chosen

Two constraints keep the diagnostic measuring mastery rather than context volume:

- **The dialogue-opening turn is never a candidate.** MathDial dialogues open with a `(generic)`
  teacher greeting, and `generic` happens to carry the highest prior in `MOVE_TO_MASTERY_PRIOR`
  (0.7). A plain argmax therefore picked turn 0 as the "high mastery" history in 84.4% of test
  dialogues (482/571 pairs) — meaning that history was **empty**, contrasted against a
  "low mastery" history averaging ~5 turns. The old metric was measuring *hint with no context*
  vs *hint with lots of context*. A greeting also carries no diagnostic signal about the student:
  nothing has been exchanged yet that could justify calling the state "high mastery".
- **Both histories must be non-empty and within 2 turns of each other in length.** MathDial
  alternates strictly between teacher and student, so two distinct tutor turns are at least two
  turns apart — a threshold of 1 is unsatisfiable in practice (17 of 599 dialogues admit any pair,
  versus 489 at `<= 2`). Two turns is one extra exchange, the tightest length match the corpus
  structure permits.

A dialogue admitting no such pair is **skipped and counted**, never replaced by a degenerate
fallback pair. On the leakage-filtered test set this yields 191 pairs from 241 dialogues (50
skipped), with mean history lengths of 7.70 (low) vs 7.52 (high) and zero empty histories.

The residual limitation is that the two histories still differ in *content*, and the mastery
priors are silver labels derived from teacher move tags — so this remains a directional
diagnostic. It is no longer dominated by a length artefact.
