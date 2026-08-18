# LLMs for Educational Hint/Feedback Generation

Subarea literature list: prompting strategies for hints/feedback, LLM-based tutoring systems, and prior evaluations of LLM tutors (GPT-based tutoring, Socratic prompting with LLMs, LLM feedback generation for programming/math education).

1. **Citation:** Shridhar, K., Macina, J., El-Assady, M., Sinha, T., Kapur, M., & Sachan, M. (2022). Automatic Generation of Socratic Subquestions for Teaching Math Word Problems. EMNLP 2022.
   **Link:** https://arxiv.org/abs/2211.12835
   **Summary:** Proposes methods for LLMs to generate sequences of Socratic subquestions that scaffold a student toward solving a math word problem, using input-conditioning and reinforcement-learning-based generation schemes. Shows that the generated Socratic questions both read as more didactically sound to human judges and improve downstream math word problem solver performance.
   **Relevance:** This is a very close neighbor to the thesis's core contribution — it directly tackles LLM-generated Socratic subquestions for math tutoring, though it does not incorporate explicit student-state modeling/adaptation, which is the thesis's added dimension.

2. **Citation:** Macina, J., Daheim, N., Chowdhury, S., Sinha, T., Kapur, M., Gurevych, I., & Sachan, M. (2023). MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems. Findings of the Association for Computational Linguistics: EMNLP 2023.
   **Link:** https://aclanthology.org/2023.findings-emnlp.372/
   **Summary:** Introduces a dataset of ~2,848 one-to-one tutoring dialogues in which human teachers guide an LLM simulating a student (with realistic errors) through GSM8k-style math word problems, annotated with a taxonomy of teacher scaffolding moves. Shows the data can be used to fine-tune models toward more effective (scaffolding, not answer-giving) tutoring behavior.
   **Relevance:** Provides a pedagogically-annotated dialogue resource and teacher-move taxonomy directly usable for training/evaluating LLM tutors that scaffold rather than solve, relevant to designing and benchmarking this thesis's hint-generation prompts.

3. **Citation:** Tack, A., & Piech, C. (2022). The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues. Proceedings of the 15th International Conference on Educational Data Mining (EDM 2022).
   **Link:** https://arxiv.org/abs/2205.07540
   **Summary:** Proposes an evaluation framework that runs conversational agents (Blender, GPT-3) in parallel with real human teachers on the same student dialogue turns, then compares responses on three pedagogical abilities: speaking like a teacher, understanding the student, and helping the student. Finds these general-purpose conversational models are quantifiably worse than real teachers, especially on helpfulness.
   **Relevance:** One of the earliest rigorous evaluations of general-purpose LLMs as tutoring dialogue agents; establishes an evaluation methodology (pedagogical-dimension comparison against human teachers) relevant to how this thesis should evaluate its own hint generator.

4. **Citation:** Phung, T., Pădurean, V.-A., Cambronero, J., Gulwani, S., Kohn, T., Majumdar, R., Singla, A., & Soares, G. (2023). Generative AI for Programming Education: Benchmarking ChatGPT, GPT-4, and Human Tutors. Proceedings of the 2023 ACM Conference on International Computing Education Research (ICER 2023), Vol. 2 (extended version on arXiv).
   **Link:** https://arxiv.org/abs/2306.17156
   **Summary:** Systematically benchmarks ChatGPT (GPT-3.5) and GPT-4 against human tutors on programming-education tasks such as bug localization/repair and generating hints for novice programmers. Finds GPT-4 substantially outperforms ChatGPT and approaches human-tutor quality on several scenarios, though human tutors remain more reliable overall.
   **Relevance:** Directly benchmarks LLM-generated programming feedback/hints against human tutors across multiple task types, giving a comparative baseline for the quality expectations and failure modes of LLM hint generators in programming education.

5. **Citation:** Pardos, Z. A., & Bhandari, S. (2024). ChatGPT-generated help produces learning gains equivalent to human tutor-authored help on mathematics skills. PLOS ONE, 19(5), e0304013.
   **Link:** https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0304013
   **Summary:** A randomized efficacy study (N=274, 3×4 design) comparing student learning gains from no-help, human-tutor-authored hints, and ChatGPT-generated hints across four algebra/statistics topic areas. Finds ChatGPT-generated help produced statistically significant learning gains over no-help control, with no significant difference from human-tutor-authored help.
   **Relevance:** Provides direct causal/experimental evidence that LLM-generated hints can produce learning gains comparable to human-authored hints, directly supporting the premise and motivation of this thesis's LLM-based hint generation approach.

6. **Citation:** Al-Hossami, E., Bunescu, R., Teehan, R., Powell, L., Mahajan, K., & Dorodchi, M. (2023). Socratic Questioning of Novice Debuggers: A Benchmark Dataset and Preliminary Evaluations. Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023).
   **Link:** https://aclanthology.org/2023.bea-1.57/
   **Summary:** Introduces a benchmark dataset of Socratic tutoring conversations in which an instructor guides a novice programmer to discover and fix bugs through questioning rather than direct answers, and evaluates GPT-based models' ability to generate such Socratic questions. Reports preliminary results on how well LLMs can imitate this Socratic debugging-tutor behavior.
   **Relevance:** Very close neighbor to the thesis — combines LLMs with explicit Socratic questioning for a tutoring task (debugging), though it lacks the thesis's explicit student-state-awareness/adaptation component and targets programming debugging rather than general hint generation.

7. **Citation:** Lin, J., Thomas, D. R., Han, F., Gupta, S., Tan, W., Nguyen, N. D., & Koedinger, K. R. (2023). Using Large Language Models to Provide Explanatory Feedback to Human Tutors. Proceedings of the 24th International Conference on Artificial Intelligence in Education (AIED 2023).
   **Link:** https://arxiv.org/abs/2306.15498
   **Summary:** Develops and compares two LLM-based approaches (fine-tuned classification and LLM-facilitated named entity recognition) for giving real-time explanatory feedback to human tutors on their use of effective praise during online lessons. Reports strong classification performance for identifying effective vs. ineffective praise and progress toward explanation-generating feedback.
   **Relevance:** Illustrates a concrete LLM prompting/feedback-generation pipeline for a pedagogical feedback subtask (evaluating and explaining tutor praise), relevant as a methodological precedent for structured LLM feedback generation in tutoring contexts.

8. **Citation:** Wang, R. E., Ribeiro, A. T., Robinson, C. D., Loeb, S., & Demszky, D. (2024). Tutor CoPilot: A Human-AI Approach for Scaling Real-Time Expertise. arXiv preprint.
   **Link:** https://arxiv.org/abs/2410.03017
   **Summary:** Reports a large-scale randomized controlled trial (900 tutors, 1,800 K-12 students) of an LLM-based "copilot" that gives human tutors real-time expert-modeled suggestions during live math tutoring sessions. Finds the tool increased use of pedagogically effective strategies (e.g., probing questions) and improved student topic mastery, with the largest gains for lower-rated tutors.
   **Relevance:** A large-scale, real-world evaluation of an LLM system embedded in live tutoring that promotes questioning-based (rather than answer-giving) pedagogy; relevant as evidence for the real-world efficacy and evaluation methodology of LLM-guided pedagogical support, though it augments human tutors rather than generating hints directly to students.
