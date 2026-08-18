# Evaluation Methodology for Tutoring / Hint Quality

Literature on how prior work measures hint quality, learning gains, engagement, and dialogue/tutoring quality — including LLM-as-judge approaches, human evaluation protocols, and learning-gain study designs. Compiled for the thesis "LLM-Guided Socratic Hint Generation with Student-State Awareness for Intelligent Tutoring Systems."

---

1. **Citation:** VanLehn, K. (2011). The Relative Effectiveness of Human Tutoring, Intelligent Tutoring Systems, and Other Tutoring Systems. *Educational Psychologist*, 46(4), 197–221.
   **Link:** https://asu.elsevierpure.com/en/publications/the-relative-effectiveness-of-human-tutoring-intelligent-tutoring/
   **Summary:** This foundational meta-analytic review compares effect sizes (measured as pre-test/post-test learning-gain differences, in standard-deviation units) across human tutoring, step-based and substep-based intelligent tutoring systems, and answer-based systems relative to no-tutoring controls. It found human tutoring's advantage (d ≈ 0.79) was much smaller than earlier folklore suggested (d ≈ 2.0), and that step-based ITSs (d ≈ 0.76) nearly match human tutors. The paper established the now-standard methodology of using controlled pre/post learning-gain effect sizes to compare tutoring interventions.
   **Relevance:** Provides the canonical learning-gain measurement methodology (effect-size comparison against no-tutoring and human-tutoring baselines) that any RCT-style evaluation of the thesis's Socratic hint generator would need to follow or cite as precedent.

2. **Citation:** Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 (Datasets and Benchmarks Track)*.
   **Link:** https://arxiv.org/abs/2306.05685
   **Summary:** This paper establishes and validates the "LLM-as-a-judge" paradigm, showing that strong LLM judges (e.g., GPT-4) can reach over 80% agreement with human preference judgments — matching human-human agreement levels — on open-ended chat quality. It also catalogs systematic judge biases (position, verbosity, self-enhancement) and limited reasoning ability, and proposes mitigations, alongside introducing the MT-Bench and Chatbot Arena benchmarks.
   **Relevance:** Foundational methodological reference for using an LLM as an automatic judge of hint/tutoring quality in this thesis; its bias catalog (verbosity, position, self-enhancement) is directly relevant to designing a trustworthy LLM-as-judge protocol for scoring Socratic hints.

3. **Citation:** Tack, A., & Piech, C. (2022). The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues. *Proceedings of the 15th International Conference on Educational Data Mining (EDM 2022)*.
   **Link:** https://arxiv.org/abs/2205.07540
   **Summary:** The authors propose a comparative-judgment evaluation protocol in which conversational agents (Blender, GPT-3) generate counterpart responses to real teacher turns in authentic tutoring dialogues, which are then rated by human judges (and models) along three ability dimensions — speaking like a teacher, understanding a student, and helping a student — via a Bayesian ability-estimation model. They find GPT-3 and Blender substantially underperform real teachers, especially on helpfulness.
   **Relevance:** Directly relevant as an early, rigorous human-evaluation protocol specifically for LLM-generated tutoring turns, including a decomposition of tutoring quality into sub-dimensions (helpfulness, student-understanding) that maps closely onto what a Socratic hint generator needs to be evaluated on.

4. **Citation:** Macina, J., Daheim, N., Chowdhury, S. P., Sinha, T., Kapur, M., Gurevych, I., & Sachan, M. (2023). MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems. *Findings of EMNLP 2023*.
   **Link:** https://arxiv.org/abs/2305.14536
   **Summary:** MathDial pairs human teachers with an LLM prompted to simulate common student errors, producing ~3K one-to-one tutoring dialogues annotated with a taxonomy of teacher "moves" (e.g., scaffolding/sense-making questions vs. giving away answers). The paper evaluates models fine-tuned on this data with both automatic metrics and human evaluation, including an interactive setting that measures the trade-off between guiding students to solve problems themselves versus revealing solutions.
   **Relevance:** Offers a directly applicable dataset-plus-evaluation design (teacher-move taxonomy, automatic + human evaluation, and an interactive success-vs-telling trade-off metric) for assessing whether an LLM Socratic hint generator scaffolds rather than gives away answers.

5. **Citation:** Macina, J., Daheim, N., Hakimi, I., Kapur, M., Gurevych, I., & Sachan, M. (2025). MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors. *arXiv preprint*.
   **Link:** https://arxiv.org/abs/2502.18940
   **Summary:** MathTutorBench is a unified benchmark evaluating open-ended LLM tutoring across three high-level teacher skills and seven concrete tasks, using a reward model trained to discriminate expert from novice teacher responses. The authors find that strong problem-solving ability does not translate into good teaching, and that tutoring quality degrades over longer dialogues.
   **Relevance:** Provides a recent, holistic benchmark methodology (multi-task, reward-model-based scoring of open-ended tutor turns) that is a strong template for evaluating hint quality across dialogue turns in a student-state-aware Socratic hint system.

6. **Citation:** Scarlatos, A., Liu, N., Lee, J., Baraniuk, R., & Lan, A. (2025). Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues. *Proceedings of AIED 2025* / arXiv preprint.
   **Link:** https://arxiv.org/abs/2503.06424
   **Summary:** The authors train an LLM tutor policy using preference optimization (DPO), where candidate tutor utterances are scored by (1) an LLM-based simulated student model predicting likelihood of a correct subsequent student response, and (2) a GPT-4o-evaluated pedagogical rubric. This combines outcome-based (predicted learning gain proxy) and process-based (pedagogical rubric via LLM-as-judge) evaluation signals to optimize tutoring dialogue quality.
   **Relevance:** Demonstrates a concrete method for combining an LLM-as-judge pedagogical rubric with a learning-outcome proxy (simulated-student correctness) — a template applicable to evaluating and training a student-state-aware Socratic hint generator toward both pedagogical soundness and effectiveness.

7. **Citation:** Kestin, G., Miller, K., Klales, A., Milbourne, T., & Ponti, G. (2025). AI tutoring outperforms in-class active learning: an RCT introducing a novel research-based design in an authentic educational setting. *Scientific Reports*, 15, 17458.
   **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12179260/
   **Summary:** This randomized controlled trial with a crossover design compared an AI-tutor-supported lesson against an instructor-led active-learning classroom lesson for 194 university physics students, measuring learning gains via pre/post assessments as well as engagement and motivation self-reports. Students using the AI tutor learned significantly more in less time and reported higher engagement and motivation than in the active-learning condition.
   **Relevance:** Supplies a current, rigorous learning-gain and engagement RCT design (crossover, pre/post testing, self-reported engagement/motivation) directly usable as a template for an empirical evaluation of the thesis's Socratic hint system in a real classroom setting.

8. **Citation:** Maurya, K. K., Srivatsa, K. V. A., Petukhova, K., & Kochmar, E. (2025). Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors. *Proceedings of NAACL 2025 (Long Papers)*.
   **Link:** https://aclanthology.org/2025.naacl-long.57/
   **Summary:** The paper proposes a unified evaluation taxonomy of eight pedagogical dimensions grounded in learning-science principles, and introduces MRBench, a dataset of ~1,600 tutor responses annotated for pedagogical quality across these dimensions, used to assess both LLM-based and human tutors in mathematics education. It finds that some LLMs excel at tutoring while others are better suited to plain question-answering than pedagogical dialogue.
   **Relevance:** Provides a directly transferable, multi-dimensional evaluation taxonomy (grounded in learning science) and annotated benchmark for scoring pedagogical quality of tutor/hint responses — highly relevant as a structured rubric basis for evaluating this thesis's Socratic hints.
