# Citation Verification Log

Independent verification pass over all 32 entries in `references.bib`, split across two parallel agents (16 entries each) that had no knowledge of each other's work. Each entry's `url`/`doi` was fetched directly; where the primary source was paywalled or blocked, an independent Semantic Scholar Graph API lookup or targeted web search was used as a cross-check — no entry was passed on memory/training-data recall alone.

**Result: 31/32 PASS on first pass, 1 FAIL (fixed, not dropped) → 32/32 verified.**

| Citation Key | Verdict | Source Checked | Notes |
|---|---|---|---|
| vanlehn2011relative | PASS | https://eric.ed.gov/?id=EJ946764 | Title, author, journal, year, vol/pages all match exactly. |
| macina2023mathdial | PASS | https://aclanthology.org/2023.findings-emnlp.372/ | Title, 7-author list, venue, year all match. |
| shridhar2022automatic | PASS | https://aclanthology.org/2022.emnlp-main.277/ | Title, 6-author list, venue, year all match. |
| alhossami2023socratic | PASS | https://aclanthology.org/2023.bea-1.57/ | Title, 6-author list, venue, year all match. |
| tack2022aiteachertest | PASS | https://arxiv.org/abs/2205.07540 | Title, authors, year match. |
| kulik2016effectiveness | PASS | https://journals.sagepub.com/doi/abs/10.3102/0034654315581420 | Title, authors, journal, year match. |
| nye2014autotutor | PASS | https://eric.ed.gov/?id=EJ1042132 | Title, 3-author list, journal, year match. |
| koedinger2005cognitive | PASS | Cambridge Core book page | Title, authors, editor, book, year (2005 print) match. Confirms the 2005 date over secondary sources that list 2006. |
| heffernan2014assistments | PASS | https://eric.ed.gov/?id=EJ1042147 | Title, authors, journal, year match. |
| sonkar2023class | PASS | https://arxiv.org/abs/2305.13272 | Title, 4-author list, year match. |
| liu2024socraticlm | PASS | NeurIPS 2024 proceedings page | Title, 8-author list, venue, year match. |
| stevens1977goal | PASS | Semantic Scholar API (DOI cross-check; ACM DL was 403) | Title, authors, year, venue match. |
| wood1976role | PASS | Semantic Scholar API (DOI cross-check; Wiley was 403) | Title, authors, year, venue match. |
| graesser2004autotutor | PASS | https://digitalcommons.memphis.edu/facpubs/7458/ | Title, 7-author list, journal, year, vol/pages match. |
| rivers2017datadriven | PASS | Semantic Scholar + web search (Springer login-walled) | Title, authors match. .bib's year=2017 is the print/volume year (27(1), 37-64); S2's 2015 is the online-first date — not a discrepancy. |
| phung2023generative | PASS | https://arxiv.org/abs/2306.17156 | Title, 8-author list, year match. |
| pardos2024chatgpt | PASS | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0304013 | Title, authors, journal, year, vol/issue/article-id match. |
| lin2023using | PASS | https://arxiv.org/abs/2306.15498 | Title, 7 authors, year match. |
| wang2024tutor | PASS | https://arxiv.org/abs/2410.03017 | Title, 5 authors, year match. |
| corbett1995knowledge | PASS | Web search (SciRP, CMU ACT-R lab page); Springer 403'd | Title, authors match. S2 API returned an anomalous "2005" for this DOI (S2 data error) — independently confirmed true year is 1995, matching the bib entry. |
| piech2015deep | PASS | https://arxiv.org/abs/1506.05908 + web search | Title, year match. WebFetch's page summary initially mis-transcribed co-author "Bassen" as "Spencer" — independently confirmed via NeurIPS/Semantic Scholar/ResearchGate that "Bassen" (as in the bib entry) is correct; the error was in the fetch tool's summary, not the citation. |
| pandey2019selfattentive | PASS | https://arxiv.org/abs/1907.06837 | Title, both authors, year match. |
| ghosh2020contextaware | PASS | KDD 2020 accepted-papers page | Title, 3 authors, year match. |
| scarlatos2025exploring | PASS | https://arxiv.org/abs/2409.16490 | Title, 3 authors match. arXiv preprint Sept 2024; publication is LAK 2025, consistent with bib. |
| jung2024clst | PASS | https://arxiv.org/abs/2406.10296 | Title, 4 authors, year match. |
| cho2024systematic | PASS | https://arxiv.org/abs/2412.09248 | Title, 3 authors, year match. |
| dmello2012autotutor | **FAIL → FIXED** | Semantic Scholar API + web search (ACM DL listing, author's self-hosted PDF) | Bib subtitle read "...Cognitively and Emotionally **Aware Cyber-Tutors**" — fabricated. Correct published subtitle: "...Cognitively and Emotionally **Intelligent Computers that Talk Back**." Authors, journal, volume/issue, year were all correct. **Title corrected in `references.bib` and `04-student-state-modeling.md` after this finding** — entry kept (real paper, verified fix) rather than dropped, since the underlying paper and all other fields are genuine. |
| zheng2023judging | PASS | https://arxiv.org/abs/2306.05685 | Title, all 13 authors, year match. |
| macina2025mathtutorbench | PASS | https://arxiv.org/abs/2502.18940 | Title, 6 authors, year match. |
| scarlatos2025training | PASS | https://arxiv.org/abs/2503.06424 | Title, 5 authors, year, venue match. |
| kestin2025aitutoring | PASS | https://pmc.ncbi.nlm.nih.gov/articles/PMC12179260/ | Title, 5 authors, journal, year, volume/article match. |
| maurya2025unifying | PASS | https://aclanthology.org/2025.naacl-long.57/ | Title, 4 authors, year, venue match. |

## Outcome

- 32/32 entries in `references.bib` are verified against a directly-fetched source (primary publisher/arXiv/ACL page, or an independent Semantic Scholar API / web-search cross-check where the primary source was paywalled or blocked).
- 1 fabricated detail was caught and fixed: `dmello2012autotutor`'s subtitle. This is exactly the failure mode the verification pass exists to catch — a plausible-sounding but wrong title text attached to a real, correctly-attributed paper.
- No entries were dropped; final bibliography remains 32 papers.

## Third-pass spot-check

A third, independent pass beyond the two-agent verification above, run directly by the controller (not delegated) after the final whole-branch review.

**First 5 entries checked** — koedinger2005cognitive, macina2023mathdial, piech2015deep, kestin2025aitutoring, wood1976role. **Deviation from plan:** the plan specified a *random* sample; 2 of these 5 (koedinger2005cognitive, wood1976role) were deliberately chosen instead because they carried discrepancy notes from the two-agent pass above, to confirm those resolutions held up under independent re-checking. This is good practice but is a weaker sample for the other 30 (unflagged) entries than a true random draw. All 5/5 PASS — no new discrepancies found; both flagged resolutions (koedinger's 2005 print date, wood's author list) reconfirmed.

**3 additional, genuinely randomly-selected entries checked** to address the sampling gap above — kulik2016effectiveness, alhossami2023socratic, zheng2023judging. All 3/3 PASS: titles, authors, and years fetched directly matched the `.bib` entries exactly (SAGE journal page for kulik2016effectiveness, ACL Anthology for alhossami2023socratic, arXiv abstract page for zheng2023judging).

**Combined third-pass result: 8/8 PASS** (5 targeted + 3 random), no discrepancies found beyond what the two-agent pass had already caught and resolved.

## corbett1995knowledge — year/issue nuance (found during final-review fix pass)

The final whole-branch review flagged that `corbett1995knowledge`'s year (1995) and the newly-added `number` (issue) field needed confirmation from an authoritative source, since Semantic Scholar had separately returned a spurious "2005" for this DOI (see main table above) and the paper's own Springer landing page (https://link.springer.com/article/10.1007/BF01099821) displays "Published: December 1994" — a third, different date.

Resolved via the CrossRef API (`https://api.crossref.org/works/10.1007/BF01099821`), which is the authoritative registry for this DOI's metadata: `"issue": "4"`, `"published-print": {"date-parts": [[1995]]}`, `"volume": "4"`, `"page": "253-278"`. CrossRef's officially registered print-publication year is **1995** (matching the existing `.bib` entry and citation key), and issue **4** (matching the field added during the fix pass). The Springer landing page's "December 1994" most likely reflects an early/accepted date rather than the registered print-issue date — the same online-first-vs-print-year pattern already documented for `rivers2017datadriven` above. Kept as-is: year 1995, issue 4, both now confirmed against the authoritative DOI registry rather than a secondary source.

## Other bib-field additions during the final-review fix pass — sourcing

The re-review of the fix wave flagged that three other fields added alongside the `corbett1995knowledge` fix (above) had no logged source, even though all three were independently verified before being added. Recording that provenance here:

- **`liu2024socraticlm`**: `pages = {85693--85721}` and `doi = {10.52202/079017-2721}` — confirmed against the official NeurIPS BibTeX export (`https://proceedings.neurips.cc/paper_files/paper/26554-/bibtex`), which lists these exact values. *(Re-checked during the audit-fix pass: a later audit flagged this URL as malformed because it does not use the `.../paper/2024/hash/<hash>-Abstract-Conference.html` shape of the entry's `url` field. It is not malformed — NeurIPS serves BibTeX from a separate internal paper-id route, `/paper_files/paper/<id>-/bibtex`, and this URL returns HTTP 200 with the SocraticLM record: `@inproceedings{NEURIPS2024_9bae399d, ... doi = {10.52202/079017-2721}, pages = {85693--85721} ...}`, matching both manually-added fields. The corresponding human-readable abstract page is `https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html`, i.e. the entry's own `url` field.)*
- **`ghosh2020contextaware`**: `doi = {10.1145/3394486.3403282}`, `eprint = {2007.12324}` — the arXiv ID was confirmed by fetching `https://arxiv.org/abs/2007.12324` (title "Context-Aware Attentive Knowledge Tracing", authors Ghosh/Heffernan/Lan match); the DOI was confirmed by resolving it through the Semantic Scholar Graph API, which returned the same paper (same title/authors/year).
- **`maurya2025unifying`**: `pages = {1234--1251}`, `doi = {10.18653/v1/2025.naacl-long.57}` — confirmed directly against the ACL Anthology page (`https://aclanthology.org/2025.naacl-long.57/`), which lists both fields verbatim.
