# ITS Foundations — Architectures, Pedagogical Foundations, Effectiveness Evidence

Literature gathered for the "Intelligent Tutoring Systems (ITS)" subarea of the thesis
*LLM-Guided Socratic Hint Generation with Student-State Awareness for Intelligent Tutoring Systems*.
All entries below were verified by fetching the linked page directly (title/authors/year
confirmed from the fetched content, not from memory).

1. **VanLehn, K. (2011). The Relative Effectiveness of Human Tutoring, Intelligent Tutoring Systems, and Other Tutoring Systems. Educational Psychologist, 46(4), 197–221.**
   - Link: https://eric.ed.gov/?id=EJ946764
   - Summary: A landmark meta-analysis synthesizing decades of controlled studies comparing human tutoring, intelligent tutoring systems (ITS), and non-interactive computer-aided instruction. It finds that human tutors achieve a mean effect size of d ≈ 0.79 over no-tutoring control conditions, while step-based/substep-based ITS achieve d ≈ 0.76 — nearly matching human tutors. It also shows that interaction granularity (answer-based vs. step-based vs. substep-based) is a key driver of effectiveness.
   - Relevance: This is the canonical effectiveness benchmark for ITS research; any claim that an LLM-based Socratic hint generator narrows the gap toward human-tutor-level effectiveness should be framed against these effect-size baselines.

2. **Kulik, J. A., & Fletcher, J. D. (2016). Effectiveness of Intelligent Tutoring Systems: A Meta-Analytic Review. Review of Educational Research, 86(1), 42–78.**
   - Link: https://journals.sagepub.com/doi/abs/10.3102/0034654315581420
   - Summary: A meta-analysis of 50 controlled evaluations of ITS finding a median effect size of 0.66 SD (raising performance from the 50th to roughly the 75th percentile) relative to conventional instruction. The review highlights that effect sizes are strongly moderated by whether outcomes are measured with locally-developed vs. standardized tests, cautioning against overgeneralizing large effects.
   - Relevance: Provides a second, independently-derived effectiveness baseline and a methodological warning (test-alignment moderators) directly relevant to designing and reporting learning-gain evaluations for an LLM-based hint generator.

3. **Nye, B. D., Graesser, A. C., & Hu, X. (2014). AutoTutor and Family: A Review of 17 Years of Natural Language Tutoring. International Journal of Artificial Intelligence in Education, 24(4), 427–469.**
   - Link: https://eric.ed.gov/?id=EJ1042132
   - Summary: A comprehensive retrospective on AutoTutor and its derivative systems, covering the natural-language dialogue architecture, expectation-and-misconception-tailored dialogue moves, pedagogical agents, and the empirical learning-gains evidence accumulated across domains (physics, computer literacy, critical thinking).
   - Relevance: AutoTutor is the canonical natural-language ITS and its dialogue-move taxonomy (hints, prompts, pumps, assertions) is a direct conceptual ancestor of Socratic hint-generation strategies this thesis formalizes with LLMs.

4. **Koedinger, K. R., & Corbett, A. (2005). Cognitive Tutors: Technology Bringing Learning Science to the Classroom. In R. K. Sawyer (Ed.), The Cambridge Handbook of the Learning Sciences (pp. 61–78). Cambridge University Press.**
   - Link: https://www.cambridge.org/core/books/abs/cambridge-handbook-of-the-learning-sciences/cognitive-tutors/0E237DCF86B4AF2F847410BECC950754
   - Summary: Describes the Cognitive Tutor architecture (model tracing and knowledge tracing built on ACT-R cognitive theory), which underlies large-scale classroom deployments and randomized field evaluations showing substantial gains over traditional instruction in mathematics.
   - Relevance: Cognitive Tutor is the canonical example of a student-model-driven ITS architecture; its model-tracing/knowledge-tracing approach is the classical precursor to the "student-state awareness" component this thesis pairs with LLM-generated hints.

5. **Heffernan, N. T., & Heffernan, C. L. (2014). The ASSISTments Ecosystem: Building a Platform that Brings Scientists and Teachers Together for Minimally Invasive Research on Human Learning and Teaching. International Journal of Artificial Intelligence in Education, 24(4), 470–497.**
   - Link: https://eric.ed.gov/?id=EJ1042147
   - Summary: Describes the ASSISTments platform, which combines formative assessment with tutoring (hints, scaffolding, immediate feedback) and doubles as a large-scale infrastructure for randomized controlled experiments on learning interventions in real classrooms.
   - Relevance: ASSISTments is a canonical, widely-used web-based ITS whose hint/scaffolding delivery mechanism and experiment-embedding design are directly relevant as a deployment and evaluation model for a Socratic hint-generation system.

6. **Macina, J., Daheim, N., Chowdhury, S. P., Sinha, T., Kapur, M., Gurevych, I., & Sachan, M. (2023). MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems. Findings of the Association for Computational Linguistics: EMNLP 2023.**
   - Link: https://arxiv.org/abs/2305.14536
   - Summary: Introduces a dataset of ~3,000 one-to-one tutoring dialogues pairing human teachers with an LLM prompted to simulate a student holding specific math misconceptions, annotated with pedagogical tutor moves (probing, focus, telling, generic). Also shows that LLMs tend to give away answers prematurely rather than scaffold, motivating pedagogically-constrained generation.
   - Relevance: A direct architectural and evaluation testbed for LLM-based Socratic hint generation — its tutor-move taxonomy and misconception-grounded dialogues are a template for constructing and evaluating student-state-aware hints.

7. **Sonkar, S., Liu, N., Mallick, D. B., & Baraniuk, R. G. (2023). CLASS: A Design Framework for Building Intelligent Tutoring Systems Based on Learning Science Principles. Findings of the Association for Computational Linguistics: EMNLP 2023.**
   - Link: https://arxiv.org/abs/2305.13272
   - Summary: Proposes a two-dataset design framework (a scaffolding dataset for step-by-step problem-solving guidance and a conversational dataset for natural interaction) for building LLM-powered ITS, demonstrated via SPOCK, a proof-of-concept biology tutor.
   - Relevance: A recent, concrete architecture for translating classic ITS pedagogical principles (scaffolding, step-by-step guidance) into an LLM-based system — directly analogous to the architecture this thesis proposes for Socratic hint generation.

8. **Liu, J., Huang, Z., Xiao, T., Sha, J., Wu, J., Liu, Q., Wang, S., & Chen, E. (2024). SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models. Advances in Neural Information Processing Systems 37 (NeurIPS 2024).**
   - Link: https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html
   - Summary: Introduces a "Thought-Provoking" teaching paradigm (contrasted with passive question-answering) implemented via a multi-agent "Dean-Teacher-Student" architecture and the SocraTeach dataset (~35,000 multi-round math dialogues), showing gains over GPT-4 on pedagogical quality metrics.
   - Relevance: The most directly on-topic prior work — an LLM system explicitly designed around Socratic questioning rather than answer-giving — making it a key point of comparison and differentiation for this thesis's own hint-generation approach.
