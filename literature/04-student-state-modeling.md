# Student/Knowledge-State Modeling

Subarea literature list: Bayesian Knowledge Tracing (BKT), Deep Knowledge Tracing (DKT) and successors, LLM-based student state inference/modeling, and affect/engagement state tracking in tutoring systems.

1. **Citation:** Corbett, A. T., & Anderson, J. R. (1995). Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. User Modeling and User-Adapted Interaction, 4, 253–278.
   **Link:** https://doi.org/10.1007/BF01099821
   **Summary:** Introduces Bayesian Knowledge Tracing (BKT), a two-state (known/unknown) Hidden Markov Model that updates the probability a student has mastered each skill in an ideal model based on their observed correctness on tutor problems, with parameters for guess, slip, initial knowledge, and learning transition. Evaluated within the ACT Programming Tutor, showing the model's per-skill mastery estimates could drive individualized problem selection.
   **Relevance:** This is the foundational, canonical formalism for probabilistic student-knowledge-state estimation in intelligent tutoring systems; it establishes the "latent mastery state updated from evidence" framing that any student-state-aware hint generator (including LLM-based ones) must either build on or explicitly depart from.

2. **Citation:** Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., & Sohl-Dickstein, J. (2015). Deep Knowledge Tracing. Advances in Neural Information Processing Systems 28 (NeurIPS 2015).
   **Link:** https://arxiv.org/abs/1506.05908
   **Summary:** Proposes Deep Knowledge Tracing (DKT), which replaces BKT's per-skill Hidden Markov Model with a single recurrent neural network (LSTM) that ingests a student's full interaction sequence and outputs a joint, continuous representation of mastery across all skills without hand-engineered skill dependencies. Shows large AUC improvements over BKT-family baselines on several real-world tutoring datasets.
   **Relevance:** The canonical origin point for neural/sequential student-state modeling; establishes the "student state as a learned latent vector updated turn-by-turn" paradigm that this thesis's LLM-based state-aware hint generator implicitly extends into an LLM context rather than an RNN.

3. **Citation:** Pandey, S., & Karypis, G. (2019). A Self-Attentive Model for Knowledge Tracing. Proceedings of the 12th International Conference on Educational Data Mining (EDM 2019).
   **Link:** https://arxiv.org/abs/1907.06837
   **Summary:** Introduces SAKT, the first transformer/self-attention-based knowledge tracing model, which identifies which past interactions are most relevant to a student's upcoming exercise via attention weights rather than a recurrent hidden state, and predicts performance from that weighted history. Reports an average 4.43% AUC improvement over prior RNN-based DKT variants, especially on sparse interaction data.
   **Relevance:** Represents the shift from recurrent to attention-based student-state architectures, directly relevant background for using attention/transformer (i.e., LLM-native) mechanisms to represent which prior student interactions matter most when generating the next Socratic hint.

4. **Citation:** Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2020).
   **Link:** https://www.kdd.org/kdd2020/accepted-papers/view/context-aware-attentive-knowledge-tracing.html
   **Summary:** Proposes AKT, which combines a monotonic, distance-aware self-attention mechanism with Rasch-model-inspired interpretable embeddings for questions and concepts, so that a student's predicted future performance is related to past performance with psychometrically motivated regularization. Outperforms prior deep knowledge tracing models (including DKT and SAKT) by up to 6% AUC while retaining more interpretable parameters.
   **Relevance:** A leading DKT successor that explicitly balances deep-learning predictive power with psychometric interpretability of the student-state representation, informative for designing a student-state module whose internal state can plausibly be surfaced to or reasoned about by an LLM hint generator.

5. **Citation:** Scarlatos, A., Baker, R. S., & Lan, A. (2025). Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs. Proceedings of the 15th International Learning Analytics and Knowledge Conference (LAK 2025).
   **Link:** https://arxiv.org/abs/2409.16490
   **Summary:** Investigates using LLMs directly as knowledge tracing engines by having them read raw tutor-student dialogue transcripts and infer/update per-skill mastery estimates, rather than relying on structured correct/incorrect interaction logs as in classical KT. Compares LLM-inferred knowledge states against traditional KT model outputs and analyzes where LLM-based inference over free-form dialogue succeeds or diverges.
   **Relevance:** Directly relevant precedent for this thesis's core mechanism — inferring a student's knowledge/understanding state from unstructured tutoring dialogue using an LLM rather than a structured KT model — which is exactly the student-state-inference step needed before generating an adaptive Socratic hint.

6. **Citation:** Jung, H., Yoo, J., Yoon, Y., & Jang, Y. (2025). CLST: Cold-Start Mitigation in Knowledge Tracing by Aligning a Generative Language Model as a Students' Knowledge Tracer. Journal of Educational Data Mining, 17(2), 86-117.
   **Link:** https://jedm.educationaldatamining.org/index.php/JEDM/article/view/854 (preprint: https://arxiv.org/abs/2406.10296)
   **Summary:** Reformulates knowledge tracing as a natural-language generation task and aligns a generative language model to act as the knowledge tracer, aiming to mitigate the classic cold-start problem (new students/skills with little interaction history) that hurts standard DKT-family models. Reports improved prediction accuracy, reliability, and cross-domain transfer in low-data regimes compared to conventional KT baselines.
   **Relevance:** Demonstrates that an LLM's world knowledge and language understanding can substitute for large amounts of per-student interaction data when estimating knowledge state, supporting the feasibility of this thesis's LLM-driven student-state inference even with limited interaction history per student.

7. **Citation:** Cho, Y., AlMamlook, R. E., & Gharaibeh, T. (2024). A Systematic Review of Knowledge Tracing and Large Language Models in Education: Opportunities, Issues, and Future Research. arXiv preprint.
   **Link:** https://arxiv.org/abs/2412.09248
   **Summary:** Systematically reviews the intersection of knowledge tracing and LLMs in education, covering how LLMs can enhance or replace structured KT models, generate/adapt educational content, and act as tracing systems themselves, while identifying shared limitations such as dependence on structured datasets and underuse of richer contextual/textual student data. Synthesizes open issues and future research directions for LLM-integrated student modeling.
   **Relevance:** Provides a current map of the LLM+knowledge-tracing landscape and its open problems (e.g., moving beyond structured logs to richer signals), useful for positioning this thesis's LLM-based, dialogue-driven student-state-aware hint generator relative to the field's identified gaps.

8. **Citation:** D'Mello, S. K., & Graesser, A. (2012). AutoTutor and Affective AutoTutor: Learning by Talking with Cognitively and Emotionally Intelligent Computers that Talk Back. ACM Transactions on Interactive Intelligent Systems, 2(4), Article 23.
   **Link:** https://doi.org/10.1145/2395123.2395128
   **Summary:** Describes AutoTutor and its affect-sensitive extension, an intelligent tutoring system that holds natural-language dialogue with students and additionally detects affective states (boredom, confusion, flow/engagement, frustration) from dialogue and posture/facial cues to adapt its tutorial responses. Finds that confusion, in particular, is significantly related to learning outcomes, motivating tutor responses that productively sustain rather than eliminate confusion.
   **Relevance:** Establishes that a complete "student state" for adaptive tutoring includes affective/engagement dimensions alongside knowledge mastery, directly relevant to broadening this thesis's student-state representation beyond pure correctness-based knowledge tracing toward affect-aware Socratic hint adaptation.
