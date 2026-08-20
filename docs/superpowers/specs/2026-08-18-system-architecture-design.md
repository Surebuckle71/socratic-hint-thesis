# System Architecture — Design

**Date:** 2026-08-18
**Thesis:** "LLM-Guided Socratic Hint Generation with Student-State Awareness for Intelligent Tutoring Systems" — M.Sc. Data Science, University of Europe for Applied Sciences (UE), Potsdam. Deadline: 2026-10-05. Total thesis length: ~30 pages.

This is the second design slice for the thesis (after [literature scoping](2026-08-18-literature-scoping-design.md)). It defines the core system: how student dialogue becomes an inferred state, and how that state conditions the next Socratic hint.

## Purpose

Build and evaluate the thesis's central technical contribution — an LLM that infers student state from ongoing tutoring dialogue and conditions its next Socratic hint on that state — with enough empirical rigor to support the novelty claim in `literature/positioning.md`, within a 7-week timeline and a 6GB-VRAM laptop GPU.

## Guiding principle

Impressive through rigor and polish, not through scope. Every component must directly strengthen the empirical claim ("state-conditioning improves hint quality/adaptivity") or the thesis's defensibility. Capabilities that would be interesting but don't serve that claim (a live chat UI, a larger fine-tuned model, a 4-condition evaluation grid) are deliberately deferred or cut — see "Out of scope" below.

## Architecture

**Single fine-tuned model, structured two-stage output, no UI coupling.** The core is a callable Python module — no web framework, no session layer — that takes a dialogue history and a current problem, and returns `(state_estimate, hint)`.

Internally, one fine-tuned LLM produces both outputs in a single generation: it first emits an explicit, structured state estimate (per-subskill mastery signal derived from the dialogue so far), then generates the next hint conditioned on that estimate. This is a single fine-tuning job (not two separate models for state and hints), and — critically for evaluation — the state-conditioning step can be suppressed at inference time on the *same* model weights, giving a controlled ablation without training a second model.

**Base model:** Qwen2.5-3B-Instruct, fine-tuned via QLoRA (4-bit quantized base + LoRA adapters) using Unsloth, chosen specifically because Unsloth's memory optimizations make QLoRA fine-tuning reliable on 6GB VRAM (vanilla HuggingFace+PEFT is tighter and riskier at this budget). Qwen2.5-3B was chosen over similarly-sized alternatives (Phi-3.5-mini, Llama-3.2-3B) for its stronger math/reasoning performance at this parameter count, which matters directly given MathDial's math-tutoring domain.

**Backend abstraction:** the module's public interface (`infer_and_hint(dialogue_history, problem) -> (state, hint)`) is backend-agnostic. The fine-tuned local model is the primary backend; a prompted-API backend (see "Fallback" below) implements the same interface. Switching backends is a config change, not a rearchitecture — this is what makes the fallback plan and the evaluation's baseline condition the same artifact rather than separate work.

## Data pipeline

**MathDial** (`macina2023mathdial`, already verified in `literature/references.bib`) is the dialogue backbone: ~2,800 real teacher-authored tutoring dialogues on grade-school math (GSM8k-grounded), each turn labeled with a teacher pedagogical move (scaffolding, telling, probing, generic).

**State labels are derived, not native to MathDial** — constructed per-turn from (a) the student's running correctness pattern on the dialogue's underlying math sub-skills, and (b) the teacher's move label as weak supervision (e.g. a "telling" move is more likely to follow a low-mastery state). These are **silver labels, not ground truth**, and this is documented explicitly as a limitation in the thesis rather than glossed over — no public tutoring dataset has verified per-turn knowledge-state labels, and the same limitation is accepted by the closest prior work (`scarlatos2025exploring`).

**Split:** MathDial's existing dialogue-level train/val/test partitioning is used as-is (no re-splitting), to avoid leaking turns from the same dialogue across splits. **This guarantee holds at the turn level only, not the problem level** — see Open risks.

## Training

QLoRA fine-tuning of Qwen2.5-3B-Instruct on (dialogue-history, derived-state, next-hint) triples constructed from the MathDial training split. Training and hyperparameter specifics (LoRA rank, learning rate, epochs, sequence length) are implementation-plan-level detail, not architecture-level — deferred to the implementation plan, to be tuned empirically within the 6GB budget.

## Evaluation

Three conditions, each isolating one specific question — chosen deliberately over a full 2×2 ablation grid to keep the empirical story tight rather than sprawling:

1. **Prompted-only pipeline** (state-inference + hint generation via prompting an API model, no fine-tuning). This is simultaneously the fallback system (see below) and baseline condition 1 — no separate work required for either purpose.
2. **Fine-tuned model, state-conditioning suppressed** (same weights as condition 3, ablation control).
3. **Fine-tuned model, full state-conditioned pipeline** (the thesis's system).

Condition 1→2 isolates the value of fine-tuning; condition 2→3 isolates the value of state-awareness specifically, which is the thesis's core contribution claim.

**Metrics:**
- **Pedagogical quality** — LLM-as-judge scoring against a subset of `maurya2025unifying`'s taxonomy dimensions (scaffolding-vs-telling, correctness, appropriateness), following `zheng2023judging`'s validated LLM-as-judge protocol.
- **Outcome proxy** — an LLM-simulated student responds to each generated hint; measure whether the simulated dialogue converges to a correct answer within N turns, following `scarlatos2025training`'s approach. Substitutes for a real learning-gain RCT, which is infeasible within 7 weeks (no IRB/recruitment time).
- **State-adaptivity (the metric specific to this thesis's contribution)** — a targeted diagnostic over constructed dialogue pairs where the true underlying state differs; does generated hint content/specificity differ appropriately between them? This directly tests the positioning doc's claim, not just general hint quality.

## Fallback plan

If QLoRA fine-tuning stalls or underperforms within the timeline, the prompted-only backend (already built as evaluation condition 1, not a separate contingency) becomes the primary system instead of a baseline. No wasted work under either outcome — this was a deliberate design goal, not an afterthought.

## Out of scope for this slice

- **Live chat UI.** The core module has no UI coupling by design, so a thin Gradio/Streamlit wrapper can be added later (portfolio/demo value, discussed separately) without touching the architecture defined here. Not committed to now, to protect the timeline.
- **A 4th/5th evaluation condition** (e.g. an off-the-shelf no-fine-tune-no-state floor baseline, or a large-API-model upper-bound reference). Legitimate if time permits late in the project; not required to support the core claim.
- **Real human-subject learning-gain study.** Infeasible in 7 weeks; the simulated-student outcome proxy substitutes, consistent with prior work in the bibliography.
- **Full fine-tuning or a larger (7B+) base model.** Ruled out by the 6GB VRAM budget; QLoRA on a 3-4B model was chosen specifically to keep this reliable rather than risky.

## Open risks

- **Derived state labels are silver, not verified ground truth** — carried over as a limitation into the thesis write-up, consistent with how the closest prior work handles the same gap.
- **6GB VRAM is a hard ceiling.** QLoRA on a 3-4B model is expected to fit, but actual memory usage depends on sequence length and batch size choices made at the implementation-plan stage; if training proves infeasible even at this size, the fallback plan absorbs that risk.
- **MathDial's move-label-derived state signal is a proxy**, not a validated knowledge-tracing output — its quality as a supervision signal for the state-estimation head is an empirical question to be checked early in implementation, not assumed.
- **The published split avoids turn-level leakage but NOT problem-level leakage.** The Data pipeline section's claim that using MathDial's partitioning as-is "avoids leaking" is correct only for turns from the same dialogue. Measured on the real dataset: 80.7% of test `qid`s also appear in train, and 59.8% of test dialogues (358/599) repeat an *identical* `(question, student_incorrect_solution)` pair from train. Left unmitigated, the fine-tuned conditions would be evaluated on math problems — with the same student misconception — that they were fine-tuned on, for ~60% of the test set, biasing precisely the condition-1-vs-2 comparison meant to show that fine-tuning helps. **Mitigation:** the evaluation script (`scripts/run_evaluation.py`) filters out every test dialogue whose `qid` occurs in the training pool before evaluating anything, dropping 358 of 599 and leaving 241 held-out dialogues, and logs the counts. This is done at the script level on purpose: `load_mathdial` keeps its contract of faithfully returning MathDial's official splits. The residual risk is reduced statistical power on a 241-dialogue test set — a fair-but-smaller evaluation is preferred to a larger contaminated one.
- **The state-adaptivity diagnostic's paired histories differ in content, not only in inferred mastery.** Turn-0 greetings are excluded and the two histories are length-matched to within two turns (see README), which removes the dominant length confound, but the metric remains directional rather than a clean causal estimate.
