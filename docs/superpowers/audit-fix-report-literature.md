# Audit-Fix Report — Literature Review Deliverables

Scope: fixes to `literature/positioning.md` and `literature/verification-log.md` from the fresh-eyes audit of the 32-paper bibliography and the gap-analysis positioning document. Every fix was checked against the subarea source files (`literature/01-its-foundations.md` … `05-evaluation-methodology.md`) and `references.bib`, not from recollection of the papers.

---

## 1. (CRITICAL) `jung2024clst` misattributed a dialogue-reading capability — FIXED

**Source of truth.** `04-student-state-modeling.md` entry 5 (`scarlatos2025exploring`) says the LLM reads "raw tutor-student dialogue transcripts and infer/update per-skill mastery estimates, rather than relying on structured correct/incorrect interaction logs." Entry 6 (`jung2024clst`) says CLST "Reformulates knowledge tracing as a natural-language generation task and aligns a generative language model to act as the knowledge tracer, aiming to mitigate the classic cold-start problem," with relevance framed as "an LLM's world knowledge and language understanding can substitute for large amounts of per-student interaction data." No mention of dialogue transcripts. The audit finding is confirmed.

**Location A — "Student-state modeling has moved from structured to language-native…" paragraph.**

Before:
> The lineage from `corbett1995knowledge` (BKT) → `piech2015deep` (DKT) → `pandey2019selfattentive` / `ghosh2020contextaware` (attention-based) → `scarlatos2025exploring` and `jung2024clst` (LLMs inferring state directly from raw dialogue, not structured correct/incorrect logs) demonstrates that LLM-based state inference from free-form tutoring dialogue is now feasible and competitive.

After:
> The lineage runs from `corbett1995knowledge` (BKT) → `piech2015deep` (DKT) → `pandey2019selfattentive` / `ghosh2020contextaware` (attention-based) → `scarlatos2025exploring`, which has an LLM read **raw tutor-student dialogue transcripts** and infer per-skill mastery from them rather than from structured correct/incorrect logs. `jung2024clst` is a complementary rather than equivalent result: it reformulates conventional knowledge-tracing interaction data as a natural-language generation task, showing that an LLM's world knowledge can substitute for large amounts of per-student interaction history under cold-start conditions — not that state can be read off free-form dialogue. Together they establish that LLM-based state inference is feasible both from dialogue (Scarlatos et al.) and from thin interaction histories (Jung et al.).

**Location B — "This thesis's contribution" section.**

Before:
> (extending the feasibility shown by `scarlatos2025exploring` and `jung2024clst` from prediction into a live pedagogical signal)

After:
> (extending the dialogue-based state inference `scarlatos2025exploring` shows to be feasible from a prediction target into a live pedagogical signal, and drawing on `jung2024clst` for the separate result that LLM world knowledge can carry state estimation when per-student interaction history is thin — as it necessarily is early in any single tutoring session)

**Location C (additional, same error, found while reading) — "The gap" section.** The gap section also bundled the two papers as one capability ("The other line (`scarlatos2025exploring`, `jung2024clst`, and the DKT lineage behind them) shows LLMs can infer a usable student-state representation directly from dialogue"). Rewritten to separate them the same way; see fix 2 below for the full replacement text of that sentence.

`jung2024clst` remains cited in all three places, now for what its source entry actually claims. The cold-start point is arguably *more* useful to the thesis than the misattributed one, since a live tutoring session starts cold by construction — that framing is now in the contribution section.

---

## 2. (IMPORTANT) "None of them close the loop" overclaimed against `corbett1995knowledge` — FIXED

**Source of truth.** `04-student-state-modeling.md` line 7: BKT was "Evaluated within the ACT Programming Tutor, showing the model's per-skill mastery estimates could drive individualized problem selection." The absolute "none of them" was contradicted by the document's own cited source. Confirmed.

Before:
> But in every one of these papers, state inference is evaluated as a **standalone prediction task** (does the model predict the next answer correctly?) — none of them close the loop by using the inferred state to generate the next pedagogical action.

After (new paragraph, scoped explicitly):
> In the post-BKT papers of that lineage (`piech2015deep`, `pandey2019selfattentive`, `ghosh2020contextaware`, `scarlatos2025exploring`, `jung2024clst`), state inference is evaluated as a **standalone prediction task** (does the model predict the next answer correctly?), with no pedagogical action taken on the inferred state. `corbett1995knowledge` is a partial precedent in the other direction: BKT's per-skill mastery estimates were shown within the ACT Programming Tutor to drive individualized *problem selection*. But problem selection is a coarse-grained pedagogical action — choosing what to work on next — not generation of the next conversational turn. Narrowing the claim to what the sources support: within the surveyed literature, none of these papers feed an inferred student state into **hint or utterance generation** specifically.

Both remedies suggested by the audit were applied together: BKT is excluded from the standalone-prediction list *and* named as a partial precedent, *and* the surviving negative claim is narrowed to hint/utterance generation. The "The gap" section was updated to match:

> But that inference is evaluated in isolation as next-answer prediction, never fed forward into hint or utterance generation (with `corbett1995knowledge`'s BKT-driven problem selection as the partial, coarser-grained precedent noted above).

---

## 3. (IMPORTANT) Effect-size range conflated two comparisons — FIXED

**Source of truth.** `01-its-foundations.md` entry 1: VanLehn finds "human tutors achieve a mean effect size of d ≈ 0.79 over no-tutoring control conditions, while step-based/substep-based ITS achieve d ≈ 0.76." Entry 2: Kulik & Fletcher is "A meta-analysis of 50 controlled evaluations of ITS finding a median effect size of 0.66 SD … relative to conventional instruction." Two different estimands, two different baselines. Confirmed.

Before:
> Two independent meta-analyses (`vanlehn2011relative`, `kulik2016effectiveness`) put step/substep-based ITS effect sizes at d ≈ 0.66–0.76, closing most of the gap to human tutors (d ≈ 0.79).

After:
> `vanlehn2011relative`'s meta-analysis puts step-based/substep-based ITS at d ≈ 0.76 against a no-tutoring baseline, closing most of the gap to human tutors (d ≈ 0.79, measured against the same no-tutoring baseline). `kulik2016effectiveness`'s independent meta-analysis of 50 evaluations reports a median of d ≈ 0.66 across ITS of all kinds relative to *conventional instruction* — a different estimand with a different comparison baseline, so the two figures are not a single range, but they converge on the same conclusion that ITS effectiveness approaches human-tutor levels.

**Consistency check against `05-evaluation-methodology.md`.** Its VanLehn entry says: "compares effect sizes … relative to no-tutoring controls. It found human tutoring's advantage (d ≈ 0.79) was much smaller than earlier folklore suggested (d ≈ 2.0), and that step-based ITSs (d ≈ 0.76) nearly match human tutors." Same numbers, same baseline, different emphasis (`05` foregrounds the methodology and the 2.0-folklore correction because that file is about evaluation design). **No real inconsistency — left as-is.** The new `positioning.md` phrasing is consistent with both `01` and `05`.

Small follow-on edit in the same paragraph: "The mechanism identified as driving this" became "The mechanism `vanlehn2011relative` identifies as driving the ITS effect", because after the split the pronoun "this" would otherwise have pointed at the Kulik sentence, where the interaction-granularity finding does not live.

---

## 4. (IMPORTANT) `cho2024systematic` credited with naming the wrong open problem — FIXED

**Source of truth.** `04-student-state-modeling.md` line 37: the survey identifies "shared limitations such as dependence on structured datasets and underuse of richer contextual/textual student data." That is not the state-inference-to-pedagogical-action gap. Confirmed.

Before:
> `cho2024systematic`'s survey explicitly names this as an open problem: KT-LLM integration underuses richer contextual/textual student data.

After:
> `cho2024systematic`'s survey names a related open problem — KT-LLM integration's dependence on structured datasets and underuse of richer contextual/textual student data — which is adjacent to, though not identical with, the closed-loop gap argued here.

The survey now stands as corroborating context rather than as an endorsement of the thesis's specific gap claim.

---

## 5. (Minor) `liu2024socraticlm` provenance URL — CHECKED, AUDIT FINDING WAS WRONG; NOTE ADDED INSTEAD

The audit called `https://proceedings.neurips.cc/paper_files/paper/26554-/bibtex` malformed on the grounds that it does not match the `.../paper/2024/hash/<hash>-Abstract-Conference.html` shape of the entry's `url` field. I checked it rather than "correcting" it:

```
$ curl -s -o /dev/null -w "%{http_code}" https://proceedings.neurips.cc/paper_files/paper/26554-/bibtex
200
```

and the body is the SocraticLM record:

```
@inproceedings{NEURIPS2024_9bae399d,
 author = {Liu, Jiayu and Huang, Zhenya and Xiao, Tong and Sha, Jing and Wu, Jinze and Liu, Qi and Wang, Shijin and Chen, Enhong},
 doi = {10.52202/079017-2721},
 pages = {85693--85721},
 title = {SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models},
 ...
}
```

The URL is live, well-formed, and lists exactly the two manually-added fields (`pages = {85693--85721}`, `doi = {10.52202/079017-2721}`) it is cited as the source for. NeurIPS simply serves BibTeX from a separate internal paper-id route (`/paper_files/paper/<id>-/bibtex`) rather than from the hash route used for abstract pages — the two shapes coexist by design. **Changing the URL would have replaced a working citation with a guess.** Instead I appended a note to the log recording the re-check, the HTTP 200, the returned values, and the corresponding hash-route abstract page, so the same false positive is not re-raised by a future audit.

---

## 6. (Minor) HTTP status typo in the `wood1976role` note — FIXED (verified, not guessed)

The audit asked for verification rather than a silent guess, so I fetched the DOI:

```
$ curl -s -o /dev/null -w "%{http_code} %{url_effective}" -L https://onlinelibrary.wiley.com/doi/10.1111/j.1469-7610.1976.tb00381.x
403 https://onlinelibrary.wiley.com/doi/10.1111/j.1469-7610.1976.tb00381.x
```

Wiley returns **403 Forbidden**, not 402. Changed `(DOI cross-check; Wiley was 402)` → `(DOI cross-check; Wiley was 403)`, now matching both the observed behaviour and the parallel ACM DL note in the same table. No uncertainty remains.

---

## 7. (Minor) `pardos2024chatgpt` uncited despite being the strongest causal evidence — FIXED

**Source of truth.** `03-llm-hint-generation.md` entry 5: "A randomized efficacy study (N=274, 3×4 design) … Finds ChatGPT-generated help produced statistically significant learning gains over no-help control, with no significant difference from human-tutor-authored help."

Added to the front of the "LLM-generated Socratic questioning exists…" paragraph, where it establishes the premise the rest of the paragraph then qualifies:

> The underlying premise — that LLM-generated hints can be pedagogically effective at all — has direct causal support: `pardos2024chatgpt`'s randomized efficacy study (N=274, 3×4 design across four algebra/statistics topics) found ChatGPT-generated help produced significant learning gains over a no-help control and *no* significant difference from human-tutor-authored help. That is the strongest causal evidence in this bibliography for the thesis's basic premise, though the help it tested was pre-authored per problem rather than generated live and adapted to a student's current state.

The closing qualifier keeps the citation load-bearing for the gap argument rather than merely decorative: Pardos & Bhandari establish that LLM hints work, and the "pre-authored, not state-adapted" contrast feeds directly into the paragraph's existing thesis.

`wang2024tutor`, `lin2023using`, and `phung2023generative` were left uncited as instructed. Having read `03-llm-hint-generation.md` in full, none of them attaches to a claim already being made without stretching it — `phung2023generative` is closest (LLM hints vs. human tutors in programming) but is a benchmarking study, not causal learning-gain evidence, so adding it alongside Pardos would have diluted the point rather than strengthened it. Not forced.

---

## 8. (Minor) Superlatives and a dropped qualifier — FIXED

**8a — competing superlatives.** The two claims were on different axes but read as a contradiction side by side. Both were disambiguated in place, without restructuring:

- `shridhar2022automatic` / `alhossami2023socratic`: "are the two closest neighbors to this thesis's core technique" → "On the Socratic-generation technique specifically, … are the nearest neighbours" (the axis is now stated before the claim rather than buried in it).
- `liu2024socraticlm`: "is the closest competitor in the entire bibliography" → "is the closest competitor overall — nearest not on any single technique but on the whole system-level ambition of this thesis".

**8b — dropped qualifier in "The gap".**

Before:
> SocraticLM's fixed-persona student profile is the closest approach to this and is verified to fall short of it (see above).

After:
> SocraticLM's fixed-persona student profile — re-sampled independently on each reply, rather than carried forward across the dialogue — is the closest approach to this and is verified to fall short of it (see above).

This matches the earlier, fuller characterization in the same document and is consistent with `01-its-foundations.md`'s verified note ("the pipeline will 'set Student to simulate one of them each time it replies'"). A reader skimming only "The gap" no longer gets the one-persona-per-dialogue impression.

---

## Post-fix verification

**1. Citation keys.** All 26 keys cited in `positioning.md` exist in `references.bib` (32 entries); zero unresolved. `pardos2024chatgpt` is newly among them (25 → 26 used). The six bib keys not cited in `positioning.md` — `heffernan2014assistments`, `koedinger2005cognitive`, `lin2023using`, `nye2014autotutor`, `phung2023generative`, `wang2024tutor` — remain intentional background-only citations carried by the subarea files.

**2. Full re-read of `positioning.md`.** The argument still runs: established evidence → gap → contribution. Two coherence follow-ons surfaced during the re-read and were fixed (both noted above): the dangling "this" in the effect-size paragraph after the VanLehn/Kulik split, and "deep-KT papers" → "post-BKT papers" as the label for the standalone-prediction list, since that list now excludes BKT itself and includes `jung2024clst`, which is a generative LM rather than a deep-KT model. Nothing changed contradicts anything else in the document; the section header "Student-state modeling has moved from structured to language-native, but is evaluated in isolation from hint generation" remains accurate under the newly narrowed claim.

**3. Not changed.** `references.bib` and the five subarea files were read as source-of-truth only and are untouched. No citation was re-verified against external sources beyond the two URL checks in fixes 5 and 6.

## Concerns

One audit finding (#5) did not survive contact with the source: the URL flagged as garbled is valid and returns exactly the values it is cited for. The log now records that check so the finding is not re-raised. No other finding was disputed — 1, 2, 3, 4, 6, 7, and 8 all reproduced against the subarea files as described.
