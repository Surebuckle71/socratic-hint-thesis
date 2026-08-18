# Literature Scoping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified 30+ paper bibliography and gap-analysis positioning doc for the thesis's related-work section.

**Architecture:** Five research agents run in parallel, one per subarea, each producing a markdown notes file + BibTeX fragment sourced from real, fetched URLs. Outputs are merged, then every citation is verified against its source link before being trusted. A synthesis pass (done directly, not delegated) turns the verified material into a positioning argument.

**Tech Stack:** Markdown files, BibTeX, git. No code/tests in the traditional sense — "tests" below are verification checks on research output.

## Global Constraints

- Target: 30+ total verified papers, 6-8 per subarea, across 5 subareas.
- Every `.bib` entry must carry a source link (arXiv ID, DOI, or ACL Anthology/ACM/IEEE URL) the agent actually fetched — not recalled from memory.
- No paper enters `positioning.md` or the final `references.bib` without passing the verification pass (Task 3).
- Papers only (peer-reviewed or credible arXiv preprints) — no textbooks/blogs unless a subarea is genuinely thin, in which case note the gap rather than pad with weak sources.
- Repo root: `C:\Users\rampr\Documents\GitHub\socratic-hint-thesis`

---

### Task 1: Five parallel subarea research agents

**Files:**
- Create: `literature/01-its-foundations.md`
- Create: `literature/02-socratic-pedagogy.md`
- Create: `literature/03-llm-hint-generation.md`
- Create: `literature/04-student-state-modeling.md`
- Create: `literature/05-evaluation-methodology.md`
- Create: `literature/raw-bib/01.bib` ... `literature/raw-bib/05.bib` (one per agent, pre-merge)

**Interfaces:**
- Produces: for later tasks — 5 markdown files each containing a numbered list of papers with `Citation:`, `Link:`, `Summary:` (2-4 sentences), `Relevance:` (1-2 sentences on connection to the thesis) fields; 5 raw `.bib` fragments with one entry per paper, each entry's `url`/`doi`/`eprint` field populated from a link the agent actually fetched.

- [ ] **Step 1: Dispatch 5 agents in parallel, one message, one Agent call each**

  Use the `mattpocock-skills:research` skill pattern (or a general-purpose research agent if that skill isn't invocable directly) with this prompt template per subarea — fill in `{SUBAREA}`, `{FOCUS}`, `{OUTPUT_MD}`, `{OUTPUT_BIB}`:

  ```
  Research subarea for an M.Sc. thesis titled "LLM-Guided Socratic Hint
  Generation with Student-State Awareness for Intelligent Tutoring Systems."

  Subarea: {SUBAREA}
  Focus: {FOCUS}

  Find 6-8 real, verifiable papers (peer-reviewed or credible arXiv
  preprints). For each paper you MUST actually fetch its page (arXiv,
  ACL Anthology, ACM DL, IEEE Xplore, DOI resolver, or Google Scholar
  result) and confirm title/authors/year from that fetched page — do
  not rely on memory alone, LLMs are known to hallucinate citations.

  Bias toward highly-cited and recent (2022-2026) work, plus 1-2
  foundational/classic papers if the subarea has an obvious canonical
  reference.

  Write two files:
  1. {OUTPUT_MD} — numbered list, one entry per paper:
     - Citation: Author(s) (Year). Title. Venue.
     - Link: <the exact URL you fetched>
     - Summary: 2-4 sentences on what the paper does/finds
     - Relevance: 1-2 sentences on how it connects to {SUBAREA} in
       this thesis's context
  2. {OUTPUT_BIB} — one BibTeX entry per paper, `url` or `doi` or
     `eprint` field populated with the link you fetched.

  Report back the paper count and confirm every entry has a fetched
  link.
  ```

  The 5 subareas (`{SUBAREA}` / `{FOCUS}` / file pairs):

  1. **Intelligent Tutoring Systems (ITS)** / architectures, pedagogical foundations, effectiveness evidence / `literature/01-its-foundations.md` + `literature/raw-bib/01.bib`
  2. **Socratic method / hint-based tutoring** / pedagogical theory and prior non-LLM hint-generation systems / `literature/02-socratic-pedagogy.md` + `literature/raw-bib/02.bib`
  3. **LLMs for educational hint/feedback generation** / prompting strategies, LLM tutoring systems, prior evaluations / `literature/03-llm-hint-generation.md` + `literature/raw-bib/03.bib`
  4. **Student/knowledge-state modeling** / Bayesian Knowledge Tracing, Deep Knowledge Tracing, LLM-based state inference / `literature/04-student-state-modeling.md` + `literature/raw-bib/04.bib`
  5. **Evaluation methodology for tutoring/hint quality** / how prior work measures hint quality, learning gains, engagement / `literature/05-evaluation-methodology.md` + `literature/raw-bib/05.bib`

- [ ] **Step 2: Verify all 5 files exist with content**

  Check: `literature/0{1..5}-*.md` and `literature/raw-bib/0{1..5}.bib` all exist and are non-empty.
  Expected: 5 markdown files, each with 6-8 numbered paper entries; 5 bib files, each with matching entry count.

- [ ] **Step 3: Commit**

  ```bash
  git add literature/
  git commit -m "Add raw subarea research from 5 parallel agents"
  ```

---

### Task 2: Merge and deduplicate bibliography

**Files:**
- Create: `literature/references.bib`
- Modify: none (raw-bib files stay as historical record)

**Interfaces:**
- Consumes: `literature/raw-bib/01.bib` ... `05.bib` from Task 1
- Produces: `literature/references.bib` — single merged file, later consumed by Task 3 (verification) and thesis writing

- [ ] **Step 1: Concatenate and dedupe**

  Read all 5 raw `.bib` files. A paper cited by multiple subareas will appear more than once — keep one entry, using the most complete version (prefer the one with a DOI over one with only a URL). Assign each entry a citation key in `author_year_firstword` format (e.g. `vanlehn2011relative`).

  Write merged result to `literature/references.bib`.

- [ ] **Step 2: Verify merge**

  Check: entry count in `references.bib` is between 25 and 45 (5 subareas × 6-8, minus expected overlap). No duplicate citation keys.
  Expected: PASS. If entry count is below 25, return to Task 1 for the thinnest subarea(s) and request 2-3 more papers before proceeding.

- [ ] **Step 3: Commit**

  ```bash
  git add literature/references.bib
  git commit -m "Merge and deduplicate subarea bibliographies"
  ```

---

### Task 3: Verification pass (mandatory, non-negotiable)

**Files:**
- Modify: `literature/references.bib` (remove failed entries)
- Modify: `literature/0{1..5}-*.md` (remove corresponding failed entries)
- Create: `literature/verification-log.md`

**Interfaces:**
- Consumes: `literature/references.bib` from Task 2
- Produces: a fully verified `references.bib` (every remaining entry's link fetched and confirmed) — consumed by Task 4 (synthesis) and, later, by thesis writing

- [ ] **Step 1: Fetch and check every entry**

  For each entry in `references.bib`: fetch the URL/DOI/arXiv link. Confirm the title, author list, and year at that link match the `.bib` entry (allow minor formatting differences, e.g. "et al." vs full author list). Record result (PASS/FAIL + reason) in `literature/verification-log.md`.

- [ ] **Step 2: Drop failures**

  Remove any FAILED entry from `references.bib` and from its corresponding subarea markdown file. Do not attempt to "fix" a failed entry by guessing correct details — if the paper is real but details were wrong, re-fetch and confirm before re-adding; if it can't be confirmed, drop it.

- [ ] **Step 3: Verify final count still meets target**

  Check: post-verification entry count is still 25+.
  Expected: PASS. If it drops below 25 due to failures, dispatch a targeted follow-up research agent for the affected subarea(s) to replace dropped papers, then repeat Step 1 for the new entries only.

- [ ] **Step 4: Commit**

  ```bash
  git add literature/references.bib literature/verification-log.md literature/0*.md
  git commit -m "Verify bibliography against fetched sources; drop unverifiable entries"
  ```

---

### Task 4: Synthesize positioning document

**Files:**
- Create: `literature/positioning.md`

**Interfaces:**
- Consumes: all 5 verified subarea markdown files from Task 3
- Produces: `literature/positioning.md` — the gap-analysis deliverable, later consumed when drafting the thesis's related-work chapter

- [ ] **Step 1: Read all 5 verified subarea files**

  Done directly (not delegated) since this is the argument that carries the thesis's contribution claim.

- [ ] **Step 2: Write positioning.md**

  Structure:
  ```markdown
  # Positioning: LLM-Guided Socratic Hint Generation with Student-State Awareness

  ## What's established
  [Per subarea, 1 short paragraph: what the literature already covers]

  ## The gap
  [1-2 paragraphs: what combination of these subareas is NOT yet
  addressed by existing work — cite specific papers by key showing
  what's close but missing]

  ## This thesis's contribution
  [1 paragraph: how the thesis's approach fills that specific gap,
  referencing the gap paragraph directly]
  ```

  Every claim must cite a specific `.bib` key from `references.bib` — no unsupported claims about "the literature."

- [ ] **Step 3: Verify no unsupported claims**

  Check: every citation key mentioned in `positioning.md` exists in `references.bib`. No sentence asserts a gap without naming the papers that come closest to filling it.
  Expected: PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add literature/positioning.md
  git commit -m "Add positioning/gap-analysis synthesis"
  ```

---

### Task 5: Final review pass

**Files:**
- Modify: none (read-only review)

**Interfaces:**
- Consumes: everything from Tasks 1-4

- [ ] **Step 1: Spot-check 5 random entries**

  Pick 5 entries at random across subareas from `references.bib`, manually re-fetch their links, confirm they resolve and match.
  Expected: 5/5 PASS. If any fail, the verification pass in Task 3 was insufficient — re-run Task 3's Step 1 across the full bibliography, not just the failures found so far.

- [ ] **Step 2: Confirm target met**

  Check: `references.bib` has 30+ entries (per user's "around 30, 30+ for sure" instruction), all subareas have 6+ verified papers, `positioning.md` exists and cites only verified keys.
  Expected: PASS.

- [ ] **Step 3: Final commit**

  ```bash
  git add -A
  git commit -m "Complete literature scoping: verified bibliography + positioning doc"
  ```
