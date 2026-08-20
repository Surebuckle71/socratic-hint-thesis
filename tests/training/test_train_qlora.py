import pytest

# socratic_hint.training.train_qlora imports unsloth at module level, and
# unsloth is only installed in the training environment (.venv313). Skip the
# whole module on a base install so `pytest` does not fail at COLLECTION time.
pytest.importorskip("unsloth")

from socratic_hint.data.training_examples import TrainingExample  # noqa: E402
from socratic_hint.output_format import PROMPT_COMPLETION_SEPARATOR  # noqa: E402
from socratic_hint.training.train_qlora import (  # noqa: E402
    TrainConfig,
    examples_to_hf_dataset,
    train,
)


def test_dataset_has_prompt_and_completion_columns_not_flattened_text():
    """TRL resolves `completion_only_loss` to True only when the dataset
    carries separate `prompt`/`completion` columns. A single `text` column
    would put ~90% of the loss on prompt tokens the model never has to
    produce."""
    dataset = examples_to_hf_dataset(
        [TrainingExample(prompt="P", completion="C")]
    )
    assert set(dataset.column_names) == {"prompt", "completion"}
    assert "text" not in dataset.column_names


def test_dataset_concatenation_preserves_the_original_training_text():
    """TRL concatenates prompt+completion verbatim, so the joined text must
    match the previously flattened `f"{prompt}\\n\\n{completion}"` form."""
    example = TrainingExample(prompt="Problem: x", completion="State: a=0.50\nHint: h")
    row = examples_to_hf_dataset([example])[0]
    assert row["prompt"] + row["completion"] == (
        f"{example.prompt}{PROMPT_COMPLETION_SEPARATOR}{example.completion}"
    )
    # The separator sits on the PROMPT side, so `tokenize(prompt)` stays a
    # clean prefix of `tokenize(prompt + completion)` despite Qwen's BPE
    # merging `>` with a following `\n\n`.
    assert row["prompt"].endswith(PROMPT_COMPLETION_SEPARATOR)
    assert not row["completion"].startswith(PROMPT_COMPLETION_SEPARATOR)


@pytest.mark.gpu
def test_train_smoke_produces_checkpoint(tmp_path):
    examples = [
        TrainingExample(
            prompt="Problem: Solve 2+2.\n\nState: <s1>=<0.xx>\nHint:",
            completion="State: arithmetic_execution=0.50\nHint: What's 2 plus 2?",
        ),
        TrainingExample(
            prompt="Problem: Solve 3+3.\n\nState: <s1>=<0.xx>\nHint:",
            completion="State: arithmetic_execution=0.60\nHint: What's 3 plus 3?",
        ),
    ]
    config = TrainConfig(output_dir=tmp_path / "checkpoint", max_steps=2)

    output_dir = train(examples, config)

    assert output_dir == config.output_dir
    assert (output_dir / "adapter_config.json").exists() or (output_dir / "config.json").exists()
