from pathlib import Path

from unsloth import FastLanguageModel

from socratic_hint.backends.base import HintBackend
from socratic_hint.output_format import (
    PROMPT_COMPLETION_SEPARATOR,
    format_prompt,
    parse_model_output,
)
from socratic_hint.types import DialogueTurn, HintResult


class FinetunedBackend(HintBackend):
    def __init__(self, checkpoint_dir: Path, max_seq_length: int = 2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
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
        # The training prompts end with this separator (see
        # `examples_to_hf_dataset`), so appending it here makes the inference
        # context byte-identical to what the model was fine-tuned on. Without
        # it the final token differs (`>` vs the merged `>\n\n`), putting every
        # generation slightly off-distribution.
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
