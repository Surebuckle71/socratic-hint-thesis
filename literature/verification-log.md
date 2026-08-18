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
| wood1976role | PASS | Semantic Scholar API (DOI cross-check; Wiley was 402) | Title, authors, year, venue match. |
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
