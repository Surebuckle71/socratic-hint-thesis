# Audit fix report — core technical codebase

**Date:** 2026-08-20
**Branch:** `audit-fixes-code` (branched from `master` at `843d359`)
**Status:** DONE_WITH_CONCERNS

All nine numbered findings and all four Minor findings are fixed. One judgment call
departs from the audit's suggested parameter (fix #1's length threshold) and one
additional pre-existing bug was found and fixed while verifying fix #5; both are
called out below and repeated in the Concerns section.

Every empirical number in this report was measured against the real MathDial dataset
via the installed `datasets` stack in `.venv313`, not estimated.

---

## Test evidence (whole-suite)

| Suite | Command | Result |
| --- | --- | --- |
| Fast | `.venv313\Scripts\python.exe -m pytest -m "not gpu" -q` | **101 passed, 1 deselected** |
| GPU | `.venv313\Scripts\python.exe -m pytest tests/training/test_train_qlora.py -v -m gpu` | **1 passed, 2 deselected** (36.7s) |

Before this work the fast suite had 88 tests; it now has 101. Output is pristine — no
warnings, no skips beyond the intended `-m gpu` deselection.

---

## 1. (CRITICAL) State-adaptivity metric measured dialogue length, not mastery state

**File:** `scripts/run_evaluation.py` — `build_adaptivity_pairs`

### What was wrong

The function took the argmax over inferred-mastery priors to pick the "high mastery"
history. MathDial dialogues open with a `(generic)` teacher greeting, and
`MOVE_TO_MASTERY_PRIOR["generic"] = 0.7` is the largest value in the table — so the
argmax landed on turn 0, whose history is **empty**.

I reproduced the audit's measurement exactly on the real test split:

```
OLD build_adaptivity_pairs on unfiltered test:
  571 pairs, 482 (84.4%) with EMPTY high-mastery history
```

So in 84.4% of dialogues the diagnostic compared *a hint generated from no context at
all* against *a hint generated from ~5 turns of context*. It was measuring context
volume, not mastery.

### What I did

1. **Turn 0 is excluded from candidates for both the low and high picks.** Beyond
   fixing the artefact, this is the methodologically right call on its own terms: a
   dialogue-opening greeting carries no diagnostic signal about the student, because
   no information has been exchanged yet that could justify calling the state "high
   mastery".
2. **Both histories must be non-empty and comparable in length**, within
   `MAX_HISTORY_LENGTH_DIFF` turns.
3. Among candidate pairs satisfying both constraints, the one with the **largest prior
   gap** is chosen; ties break toward the closer length, then the earlier turns, so
   selection is deterministic.
4. **Dialogues admitting no valid pair are skipped and counted.** `build_adaptivity_pairs`
   now returns `(pairs, skipped)` and `main()` prints the skip count, so the reduction
   is visible rather than silent. No degenerate fallback pair is ever constructed.
5. The docstring no longer carries the old "differ in length and content" caveat. It
   now describes what was actually done and why, and states the honest residual
   limitation: the histories still differ in *content*, and the priors are silver
   labels, so the metric stays directional — but it is no longer dominated by a length
   artefact.

### Judgment call: threshold is 2, not the suggested 1

The audit suggested `abs(len(low) - len(high)) <= 1`. **This is unsatisfiable on
MathDial in practice.** The corpus alternates strictly between teacher and student, so
two *distinct* tutor turns are at least two turns apart. Measured — for each test
dialogue, the smallest achievable length gap between any valid low/high prior pair:

| Smallest achievable gap | Dialogues |
| --- | --- |
| 1 | 17 |
| 2 | 472 |
| no valid contrast at all | 110 |

A threshold of 1 would discard 97% of the test set and leave 17 pairs — not a usable
diagnostic. I set `MAX_HISTORY_LENGTH_DIFF = 2`, which is one extra exchange (one
tutor turn plus one student reply) and the tightest length match the corpus structure
permits. The constant is module-level with a comment recording this measurement, so
the choice is inspectable rather than buried.

### Result on real data (after fix #4's leakage filter)

```
ADAPTIVITY (filtered set): 191 pairs, 50 dialogues skipped
  empty-history pairs: 0
  mean len low=7.70  high=7.52
  max |len diff| = 2
```

Empty histories: 482 → **0**. Mean length difference: ~5 turns → **0.18 turns**.

### Tests

Seven new tests in `tests/scripts/test_run_evaluation_script.py`, covering: turn 0
never selected; histories length-comparable; nearer of two equally-high candidates
wins; no-contrast dialogue skipped not degenerate; too-far-apart contrast skipped;
greeting-only dialogue skipped; skip counting across dialogues; histories are genuine
dialogue prefixes.

---

## 2. (CRITICAL) One malformed generation could kill an entire multi-hour run

**Files:** `src/socratic_hint/evaluation/state_adaptivity.py`,
`src/socratic_hint/evaluation/run_evaluation.py`

### What was wrong

`evaluate_condition`'s per-example loop was hardened in an earlier fix wave, but
`adaptivity_diagnostic.run_batch(...)` sat outside that protection and calls
`backend.infer_and_hint` three times per pair with no error handling.
`parse_model_output` raises `ValueError` on unparseable output. At 191 pairs × 3
generations (571 × 3 before fix #1), a single bad generation late in a run aborted the
whole condition — losing the final `ConditionResult` and every remaining condition.

### What I did

`run_batch` wraps each pair's `self.run(...)` in try/except, counting failures the same
way `evaluate_condition` already does.

I extended the return type rather than leaving the count unreported. `run_batch`
previously returned a bare `float`; it now returns a new frozen dataclass
`StateAdaptivityBatchResult` with `net_rate`, `differ_rate`, `noise_rate`,
`evaluated_pairs`, `failed_pairs`. Reporting `differ_rate` and `noise_rate`
separately costs nothing and makes the net rate auditable after the fact.

`net_rate` is `None` when every pair failed, keeping "not measured" distinguishable
from a measured 0.0 — the same principle the existing empty-pairs handling already
used.

The count propagates to `ConditionResult.failed_adaptivity_pairs`, into a
`record_type: "adaptivity_batch"` line in the results JSONL, into a printed warning,
and into a new `failed_pairs` column in the results table.

### Tests

Three new tests: component rates and counts reported; batch survives a pair that raises
partway through (2 evaluated, 1 failed, surviving pairs still measured); all-fail case
returns `net_rate is None`. Two new backends model the failure modes
(`AlwaysFailingBackend`, `FailOnSecondPairBackend`). Existing `run_batch` assertions
updated to `.net_rate`; `test_run_evaluation.py`'s mocks updated via a `make_batch`
helper, plus a new test asserting `failed_adaptivity_pairs` propagates.

---

## 3. (CRITICAL) Condition 2 tested a prompt format the model was never trained on

**Files:** `src/socratic_hint/data/training_examples.py`,
`src/socratic_hint/output_format.py`

### What was wrong

`build_training_examples` only ever constructed state-conditioned examples, while the
harness runs `suppress_state=True` against those same weights. The model had never seen
a suppress_state-formatted prompt. Either failure mode invalidates the thesis's central
ablation: the model ignores the unfamiliar instruction (conditions 2 and 3 differ only
by sampling noise — a null result by construction), or half-follows it and produces
out-of-distribution output whose lower quality says nothing about losing state
conditioning.

### What I did

Each tutor turn now emits **both** variants:

- the existing state-conditioned example: `format_prompt(history, problem)` +
  `format_completion(state, hint)`;
- a suppressed example: `format_prompt(history, problem, suppress_state=True)` +
  `format_hint_only_completion(hint)`.

I added `format_hint_only_completion(hint) -> f"Hint: {hint}"` to `output_format.py`
rather than inlining the f-string, so the training target and the parser live in the
same module and cannot drift apart. I verified against `parse_model_output` that this
is exactly the shape it expects with `suppress_state=True` — the test asserts the
round-trip rather than assuming it.

### Result on real data

```
TRAINING EXAMPLES: 2035 dialogues -> 26,968 examples
  (13,484 state-conditioned + 13,484 suppressed)
```

Exactly doubled, as intended.

### Tests

`tests/data/test_training_examples.py` rewritten. Count assertion 2 → 4 for a
two-tutor-turn dialogue. New tests: the suppressed variant's prompt contains no state
instruction and its completion parses correctly with
`parse_model_output(..., suppress_state=True)` returning `state is None`; both variants
of a turn share history and target hint, differing only in state conditioning; a
dialogue with no tutor turns yields nothing.

---

## 4. (CRITICAL) 60% of the test set shared its (problem, misconception) pair with train

**Files:** `scripts/run_evaluation.py`, `README.md`,
`docs/superpowers/specs/2026-08-18-system-architecture-design.md`

### What was wrong

The spec claimed the split "avoids leaking... across splits". True at the turn level,
false at the problem level. Confirmed on the real dataset:

```
train dialogues 2035, validation 227, test 599
unique test qids 394; 281 also in train, 318 also in train|validation
test dialogues with qid in train:            317
test dialogues with qid in train|validation: 358
test dialogues with identical (question, student_incorrect_solution) in train: 317
```

### What I did

Added `filter_leaked_test_examples(test_examples, train_qids)` to
`scripts/run_evaluation.py`, applied after loading and **before** `--limit`, with the
counts printed. `load_mathdial` is untouched, per the audit's instruction — it keeps its
contract of faithfully returning MathDial's official splits, and the filtering is
explicit at the point where it matters. The script aborts with a clear error if the
filter empties the test set.

**Judgment call: the exclusion pool is train ∪ validation, not train alone.** Filtering
against `load_mathdial("train")` alone drops 317; against train ∪ validation it drops
**358**, matching the audit's expected figure. Both splits are carved from HF's single
`train` split and both are consumed by `run_training.py` (validation drives the
eval-loss signal), so both were seen during training and both must be excluded. The 358
match confirms this reading.

```
LEAKAGE: 599 test -> dropped 358 -> 241 kept
```

Documentation:

- `README.md` gained a "Problem-level leakage filtering" subsection under Evaluation,
  giving the measured numbers and explicitly framing this as fixing a leakage issue
  found during audit.
- The spec's Data pipeline "Split" line now says the guarantee holds at the turn level
  only and points to Open risks; a new Open risks bullet states the measured overlap,
  why it biases the condition-1-vs-2 comparison, the script-level mitigation, and the
  residual risk (reduced statistical power on 241 dialogues — a fair-but-smaller
  evaluation preferred to a larger contaminated one). A second bullet records fix #1's
  residual limitation.

### Tests

Three tests covering: overlapping qids dropped; no-overlap case keeps everything; the
filter can legitimately empty the test set.

---

## 5. (Important) 90% of training loss was spent on prompt tokens

**Files:** `src/socratic_hint/training/train_qlora.py`,
`src/socratic_hint/backends/finetuned.py`, `src/socratic_hint/output_format.py`

### What I did

`examples_to_hf_dataset` now builds `Dataset.from_dict({"prompt": [...], "completion":
[...]})` instead of a flattened `text` field, and `dataset_text_field="text"` is dropped
from `SFTConfig`.

I checked TRL 0.24's actual behaviour in the installed `.venv313` rather than assuming.
`sft_trainer.py:736`:

```python
if args.completion_only_loss is None:
    self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
```

So it resolves to `True` automatically from the dataset shape. I deliberately left
`completion_only_loss` **unset** rather than passing `True`: hardcoding it would
silently do nothing useful if the dataset shape ever regressed to a single `text`
column, whereas leaving it to resolve keeps the dataset shape as the single source of
truth. The reasoning is recorded in a comment at the `SFTConfig` call site.

Verified on real training examples with the base model's tokenizer:

```
completion_only_loss resolves to: True
rows=524  tokens=249,106  unmasked(completion)=22,758  = 9.1% of loss
```

9.1% — matching the audit's ~90/10 estimate. Previously 100% of tokens carried loss.

### Additional bug found while verifying (pre-existing, now fixed)

The same verification surfaced a real defect the audit did not mention. With the
separator on the completion side, `tokenize(prompt)` was **not** a prefix of
`tokenize(prompt + completion)` on **524 of 524 rows**. Diagnosed:

```
first diff at index 193 of prompt length 194
prompt tail    : ['hint', ' text', '>']
joined at same : ['hint', ' text', '>\n\n', 'State', ':', ' problem']
```

Qwen's BPE merges the prompt's trailing `>` with the following `\n\n` into a single
`>\n\n` token. TRL warns about exactly this, and the loss-mask boundary — which fix #5
makes load-bearing — straddled a token.

Moving `PROMPT_COMPLETION_SEPARATOR` to the **prompt** side drops mismatches to
**0/524** while keeping the joined text byte-identical and the mask at 9.1%. The
constant lives in `output_format.py` so the trainer and the backend share one
definition.

This exposed a second-order consequence that was **already wrong before my change**:
training saw contexts ending in the merged `>\n\n` token, while `FinetunedBackend` fed
the model a context ending in a bare `>` — a different final token, putting every
generation slightly off-distribution. `FinetunedBackend.infer_and_hint` now appends the
same separator, making inference byte-identical to training.

### Tests

Two new fast tests: the dataset has `prompt`/`completion` columns and no `text` column;
the concatenation preserves the original flattened text, with the separator on the
prompt side. Plus the GPU smoke test, **re-run after the change: passed** (36.7s).

---

## 6. (Important) `_is_correct` was brittle to trivial formatting

**File:** `src/socratic_hint/evaluation/simulated_student.py`

`extract_text(response).strip().upper().startswith("YES")` read `"**YES**"`, `'"YES"'`,
`"- YES"` and similar as NO. The failure is one-directional: it only ever depressed
`convergence_rate`, never inflated it — a silent, systematic undercount.

Extracted a module-level `_reads_as_yes(raw)` that checks whether `"YES"` appears in the
first 16 characters of the stripped, uppercased text. Kept deliberately simple, per the
audit's instruction. Extracting it as a named function (rather than editing the
expression in place) is what makes it directly unit-testable without mocking an API
client.

**Tests:** parametrised over 8 affirmative forms and 6 negative forms, plus a test that
a discursive answer merely mentioning "yes" later in the sentence is not read as a
verdict — the case that justifies the window being short.

---

## 7. (Important) Multi-line hints were silently truncated

**File:** `src/socratic_hint/output_format.py`

The loop broke at the first line starting with `hint:` and returned only that line's
remainder. With `GENERATION_MAX_TOKENS = 4096`, multi-line hints are entirely realistic,
and both the simulated student and the judge only ever saw the fragment.

Extracted `_extract_hint(raw_output)`, which captures from the `Hint:` line to the end
of the output, joining subsequent lines and stopping at a later recognised field prefix
(`_FIELD_PREFIXES = ("state:", "hint:")`). Behaviour on single-line hints is unchanged,
including the empty-hint edge case.

**Tests:** four new tests — a three-line hint kept whole; multi-line under
`suppress_state=True`; termination at a later `State:` line; single-line hint unchanged.

---

## 8. (Important) The default-prior fallback test didn't test the fallback

**File:** `tests/data/test_state_labels.py`

`test_derive_state_labels_defaults_unknown_move_to_generic_prior` used `move="focus"`,
which *is* a key in `MOVE_TO_MASTERY_PRIOR` — a duplicate of the known-move test that
never exercised `DEFAULT_PRIOR`.

Renamed to `..._defaults_unrecognised_move_to_default_prior`, switched to
`move="revealing_answer"`, and asserted the result equals `DEFAULT_PRIOR` across all
subskills. Added `assert move not in MOVE_TO_MASTERY_PRIOR` as a guard so the same
mistake cannot recur silently if the table gains that key later. Added a companion test
for `move=None`, which is a real MathDial case.

---

## 9. (Important) `pyproject.toml`'s `trl` floor was incompatible with the code

**File:** `pyproject.toml`

`trl>=0.12.0` could resolve a TRL that predates `SFTConfig(max_length=...)` (older TRL
used `max_seq_length`), crashing at config construction. Fix #5 adds a second
requirement: prompt/completion datasets auto-resolving `completion_only_loss`.

Raised to `trl>=0.20.0` — the audit's suggested reasonable floor, and the version at
which `max_length` replaced `max_seq_length`. Recorded in a comment that development
and smoke-testing were against the installed 0.24.0, so the tested-versus-declared gap
is visible.

---

## Minor findings

| Finding | Fix |
| --- | --- |
| Results JSONL opened `"a"`, never truncated — re-running interleaved two runs' records | `evaluate_condition` truncates `results_path` once at the start (once per condition, not per example). New test runs the condition twice and asserts 2 records, not 4. |
| Eval subset was `eval_examples[:256]` in qid order | Shuffled with a fixed seed (`EVAL_SUBSET_SEED = 3407`) before slicing — representative and still reproducible. |
| `args.max_steps or derive_max_steps(...)` — `--max-steps 0` silently fell back | Now `if args.max_steps is not None`, with values below 1 rejected with a clear error rather than silently substituted. |
| Bare `KeyError` on missing `ANTHROPIC_API_KEY` at three sites | All three route through `llm_config.require_api_key()`, raising a `RuntimeError` naming the variable, which components need it, and the `client=` escape hatch. |
| Missing key discovered only after the multi-GB checkpoint load | `PromptedBackend()` is constructed before `FinetunedBackend(...)` in `main()`. Two-line change (the judge and simulated student were already ahead of the load), so I did the reordering as well as the message. |

---

## Commits

| SHA | Subject |
| --- | --- |
| `a07225a` | Train the state-suppressed format; stop truncating multi-line hints (#3, #7) |
| `2e92d6a` | Compute training loss on the completion only; raise the trl floor (#5, #9) |
| `601b660` | Stop one bad generation from killing an adaptivity batch (#2, JSONL minor) |
| `23fb814` | Make state-adaptivity measure mastery, and filter leaked test problems (#1, #4) |
| `de766ff` | Harden the convergence check and the missing-API-key error (#6, API-key minors) |
| `db6d9ef` | Test the real default-prior fallback; fix two run_training bugs (#8, run_training minors) |

Fixes #3/#7 and #5/#9 share commits because they touch the same files and could not be
split without interactive staging.

---

## Concerns

1. **`MAX_HISTORY_LENGTH_DIFF = 2`, not the audit's suggested 1.** Forced by MathDial's
   strict speaker alternation — `<= 1` yields 17 pairs across 599 dialogues. I believe 2
   is correct and it removes the confound the audit identified (mean length gap 0.18
   turns, zero empty histories), but it is a deliberate departure from the written
   instruction and should be confirmed.

2. **The adaptivity sample is now much smaller: 571 pairs → 191.** This is the combined
   effect of fixes #1 and #4 and is unavoidable if both are correct, but it materially
   reduces the statistical power of the adaptivity diagnostic. Worth deciding
   explicitly whether 191 pairs supports the claim the thesis wants to make, rather than
   discovering it at write-up.

3. **The test set is now 241 dialogues, down from 599.** Same trade-off, deliberately
   made in favour of a fair-but-smaller evaluation. Flagged in the spec's Open risks.
   If power proves inadequate, the honest alternative is re-splitting MathDial at the
   problem level rather than relaxing the filter — that would change `load_mathdial`'s
   contract and is a bigger decision than this fix wave should make unilaterally.

4. **Fix #3 doubles training set size, so the derived step count changes.**
   `derive_max_steps` targets ~2 epochs but clamps to `[200, 2000]`. At 26,968 examples
   and an effective batch of 16, two epochs is 3,371 steps — clamped down to
   `MAX_STEPS = 2000`, i.e. ~1.19 epochs rather than the intended 2. This was already
   true before my change (13,484 examples → 1,685 steps, within range), so doubling the
   data pushed it past the ceiling. I did **not** change `MAX_STEPS`: it is a
   deliberate runtime guard rail tied to the 6GB VRAM budget and multi-hour run
   duration, and raising it is a training-budget decision, not a bug fix. Flagging it
   because the effective epoch count silently changed as a side effect of fix #3.

5. **The `>\n\n` tokenizer-boundary bug (see fix #5) was not in the audit's list.** I
   fixed it because fix #5 makes that boundary define the loss mask, and because it
   also meant `FinetunedBackend` was feeding the model a final token it never saw in
   training. Both changes are verified (0/524 mismatches; GPU smoke test passes), but
   this is scope beyond the written brief.

6. **Work is on branch `audit-fixes-code`, not `master`.** The repo was on its default
   branch; I branched rather than committing directly to it. Merge or fast-forward as
   preferred.

7. **No end-to-end evaluation run was performed.** `scripts/run_evaluation.py` needs a
   trained checkpoint plus real API spend, so fixes #1, #2 and #4 are verified by unit
   tests plus direct measurement against the real dataset, not by an actual evaluation
   run. Fix #3's real effect — whether the model genuinely learns the suppressed format
   — can only be confirmed by retraining and re-evaluating.
