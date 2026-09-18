from unsloth import FastLanguageModel

from socratic_hint.backends.base import HintBackend
from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR, format_prompt, parse_model_output
from socratic_hint.training.train_qlora import BASE_MODEL
from socratic_hint.types import DialogueTurn, HintResult


class QwenPromptedBackend(HintBackend):
    """Qwen2.5-3B-Instruct, prompted with the same instructions and output
    format as the fine-tuned conditions, but with NO fine-tuning applied - the
    base checkpoint's own weights, unmodified.

    This is the missing piece Condition 1 (Claude Sonnet 5) cannot supply on
    its own: RQ3 asks about the added value of fine-tuning, but comparing the
    fine-tuned Qwen model only against a much larger, differently-trained,
    differently-provided model (Claude) confounds fine-tuning with model
    size, architecture, training investment, and provider all at once. This
    backend holds the base model and prompt format fixed and removes only
    fine-tuning, giving a same-model-family comparison that Condition 1
    cannot.
    """

    def __init__(self, max_seq_length: int = 2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        prompt = (
            format_prompt(dialogue_history, problem, suppress_state=suppress_state)
            + PROMPT_COMPLETION_SEPARATOR
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        raw_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_model_output(raw_text, suppress_state=suppress_state)
