import pytest

from socratic_hint.data.training_examples import TrainingExample
from socratic_hint.training.train_qlora import TrainConfig, train


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
