# Literature Scoping — Design

**Date:** 2026-08-18
**Thesis:** "LLM-Guided Socratic Hint Generation with Student-State Awareness for Intelligent Tutoring Systems" — M.Sc. Data Science, University of Europe for Applied Sciences (UE), Potsdam. Deadline: 2026-10-05. Total thesis length: ~30 pages.

## Purpose

Produce the research foundation for the thesis's related-work section: a verified bibliography and a positioning document arguing where this thesis's contribution sits relative to existing work. This is the first of several design slices for the thesis (system architecture, hint-generation approach, student-state modeling, and evaluation design are separate, later slices).

## Scope

Five subareas, derived directly from the thesis title:

1. **Intelligent Tutoring Systems (ITS)** — architectures, pedagogical foundations, effectiveness evidence
2. **Socratic method / hint-based tutoring** — pedagogical theory and prior (non-LLM) hint-generation systems
3. **LLMs for educational hint/feedback generation** — prompting strategies, LLM tutoring systems, prior evaluations
4. **Student/knowledge-state modeling** — Bayesian Knowledge Tracing, Deep Knowledge Tracing, LLM-based state inference
5. **Evaluation methodology for tutoring/hint quality** — how prior work measures hint quality, learning gains, engagement

Target: **6-8 papers per subarea, 30+ total**, sized to support a ~4-6 page related-work section in a 30-page thesis. Bias toward highly-cited and recent (last 3-4 years) work, plus 1-2 canonical/foundational citations per subarea where one clearly exists (e.g. VanLehn 2011 for ITS meta-analysis).

Papers only (peer-reviewed or arXiv preprints from credible research groups) — no textbooks or blog posts, unless a subarea proves genuinely thin on peer-reviewed work, in which case note the gap explicitly rather than padding with weak sources.

## Repo structure

New git repo: `Documents\GitHub\socratic-hint-thesis`

```
socratic-hint-thesis/
  literature/
    01-its-foundations.md
    02-socratic-pedagogy.md
    03-llm-hint-generation.md
    04-student-state-modeling.md
    05-evaluation-methodology.md
    references.bib
    positioning.md
  docs/superpowers/specs/
    2026-08-18-literature-scoping-design.md
```

Each subarea markdown file contains, per paper: full citation, 2-4 sentence summary, and a note on relevance to the thesis.

## Process

1. **Parallel search** — five research agents, one per subarea, each search primary sources (arXiv, ACL Anthology, AIED/EDM/L@S proceedings, Google Scholar) and write their subarea file plus BibTeX entries for their papers.
2. **Merge** — combine all agent output into one `references.bib`, deduplicating overlapping citations (a paper may be relevant to more than one subarea).
3. **Verification pass (mandatory, non-negotiable)** — LLM research agents can fabricate plausible-sounding citations (invented papers, wrong year/venue, real authors with wrong title). To eliminate this risk:
   - Every `.bib` entry must carry a source link the agent actually fetched (arXiv ID, DOI, or ACL Anthology/ACM/IEEE URL) — not one recalled from memory.
   - After agents return, a separate pass confirms each link resolves and that title/authors/year in the `.bib` entry match what's actually published at that link.
   - Any entry that fails this check is dropped, not corrected by guessing.
   - No paper enters `positioning.md` or `references.bib` without passing verification.
4. **Synthesis** — read all five verified subarea files and write `positioning.md`: what's established in each subarea, where the gap sits, and how "LLM-guided Socratic hints with student-state awareness" fills it. Done directly, not delegated, since this is the argument that carries the thesis's contribution claim.

## Out of scope for this slice

- System architecture design
- Hint-generation prompting/model design
- Student-state modeling implementation
- Evaluation study design for the thesis's own experiments
- Drafting actual thesis prose (this comes after positioning is validated, ideally with advisor input)

## Open risk

Paper-count and structure targets here (30+ total, 6-8/subarea) are reasonable estimates, not verified against UE Potsdam program requirements or advisor guidance. If the user obtains a program citation-count norm or advisor feedback later, this target should be revisited.
