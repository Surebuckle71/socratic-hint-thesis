# Unsloth must be imported before trl/transformers/peft so its patches apply.
from unsloth import FastLanguageModel

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from socratic_hint.data.training_examples import TrainingExample

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


def build_lora_config(config: TrainConfig) -> dict:
    """LoRA adapter kwargs for FastLanguageModel.get_peft_model, derived from a TrainConfig."""
    return {
        "r": config.lora_r,
        "target_modules": LORA_TARGET_MODULES,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": True,
        "random_state": 3407,
    }


def examples_to_hf_dataset(examples: list[TrainingExample]) -> Dataset:
    return Dataset.from_dict(
        {"text": [f"{ex.prompt}\n\n{ex.completion}" for ex in examples]}
    )


def train(examples: list[TrainingExample], config: TrainConfig) -> Path:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **build_lora_config(config))

    dataset = examples_to_hf_dataset(examples)

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        max_length=config.max_seq_length,
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return config.output_dir
