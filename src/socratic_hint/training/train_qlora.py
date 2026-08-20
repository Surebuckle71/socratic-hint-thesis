# Unsloth must be imported before trl/transformers/peft so its patches apply.
from unsloth import FastLanguageModel

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from socratic_hint.data.training_examples import TrainingExample
from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR

# Pre-quantized Qwen2.5-3B-Instruct checkpoint, for fast download + 4-bit loading.
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    max_steps: int
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 16
    max_seq_length: int = 2048
    # Effective batch size = per_device_train_batch_size * this. The
    # per-device batch is tiny because of the 6GB VRAM budget, so accumulation
    # is what makes the gradient signal stable.
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 20
    # Memory-efficient optimizer; standard for QLoRA on constrained VRAM.
    optim: str = "adamw_8bit"
    # Periodic checkpointing so a crash in a multi-hour run does not lose
    # everything. Defaults are larger than the smoke test's max_steps=2, so a
    # smoke run performs no mid-run saves (the final save_pretrained still
    # runs) and stays fast.
    save_steps: int = 100
    save_total_limit: int = 3
    # Validation loss signal during real training. Disabled when no eval
    # dataset is passed to `train`.
    eval_steps: int = 100


def build_lora_config(config: TrainConfig) -> dict:
    """LoRA adapter kwargs for FastLanguageModel.get_peft_model, derived from a TrainConfig."""
    return {
        "r": config.lora_r,
        "target_modules": LORA_TARGET_MODULES,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        # Unsloth's own gradient-checkpointing variant: more memory-efficient
        # than the generic True, which matters under the 6GB VRAM budget that
        # drove the choice of a 3B base model.
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
    }


def examples_to_hf_dataset(examples: list[TrainingExample]) -> Dataset:
    """Build a TRL prompt-completion dataset (not a flattened `text` field).

    With one `text` column, SFTTrainer computes loss over every token — and the
    prompt is ~90% of the tokens here (measured: 9.1% completion), so most of
    the gradient signal went into reproducing dialogue history the model is
    never asked to produce. Separate `prompt`/`completion` columns make TRL
    resolve `completion_only_loss` to True (SFTTrainer: `if
    args.completion_only_loss is None: self.completion_only_loss = "prompt" in
    sample and "completion" in sample`), masking prompt tokens out of the loss
    so training optimises the state estimate and hint only.

    TRL concatenates `prompt + completion` verbatim, so the joined text is
    byte-identical to the previous flattened form. The separator goes on the
    PROMPT side: Qwen's BPE merges the prompt's trailing `>` with a following
    `\\n\\n` into one `>\\n\\n` token, so putting it on the completion side would
    make `tokenize(prompt)` not a prefix of `tokenize(prompt + completion)` —
    TRL warns about that, and the loss boundary would straddle a token.
    `FinetunedBackend` appends the same separator at inference time.
    """
    return Dataset.from_dict(
        {
            "prompt": [
                f"{ex.prompt}{PROMPT_COMPLETION_SEPARATOR}" for ex in examples
            ],
            "completion": [ex.completion for ex in examples],
        }
    )


def train(
    examples: list[TrainingExample],
    config: TrainConfig,
    eval_examples: list[TrainingExample] | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> Path:
    """Fine-tune the base model on `examples`.

    `eval_examples` (typically built from `load_mathdial("validation")`) turns
    on periodic evaluation so a real run has a validation-loss signal. When it
    is omitted — as in the GPU smoke test — evaluation is disabled entirely.

    `resume_from_checkpoint`: `True` auto-detects the latest `checkpoint-N`
    under `config.output_dir` (the periodic saves `save_steps` already
    produces); a path string resumes from that specific checkpoint directory.
    Restores model/optimizer/scheduler/RNG state and the global step count, so
    training continues rather than restarting — `config.max_steps` is still
    the target *total* step count, not additional steps from here.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **build_lora_config(config))

    dataset = examples_to_hf_dataset(examples)
    eval_dataset = examples_to_hf_dataset(eval_examples) if eval_examples else None

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        # Eval defaults to 8 in TrainingArguments if unset — match the train
        # batch size instead, since it was deliberately kept tiny for the
        # 6GB VRAM budget and an unconstrained eval pass would OOM at the
        # first eval checkpoint.
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        max_length=config.max_seq_length,
        # No `dataset_text_field`: the dataset carries `prompt`/`completion`
        # columns, which is what makes TRL mask prompt tokens out of the loss.
        # `completion_only_loss` is left unset so TRL resolves it to True from
        # the dataset shape; setting it explicitly would silently do nothing if
        # the dataset shape ever regressed to a single `text` column.
        logging_steps=1,
        # Periodic checkpointing: a crash mid-run resumes from the last save
        # instead of losing the whole run.
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return config.output_dir
