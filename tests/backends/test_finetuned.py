from unittest.mock import MagicMock, patch

import pytest

from socratic_hint.backends.finetuned import FinetunedBackend
from socratic_hint.types import DialogueTurn


def test_infer_and_hint_parses_generated_text(tmp_path):
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    fake_inputs = {"input_ids": MagicMock(shape=(1, 10))}
    fake_tokenizer.return_value.to.return_value = fake_inputs
    fake_model.generate.return_value = [MagicMock()]
    fake_tokenizer.decode.return_value = "State: arithmetic_execution=0.55\nHint: Try again."
    fake_model.device = "cpu"

    with patch("socratic_hint.backends.finetuned.FastLanguageModel") as fake_flm:
        fake_flm.from_pretrained.return_value = (fake_model, fake_tokenizer)
        backend = FinetunedBackend(checkpoint_dir=tmp_path)

    result = backend.infer_and_hint([], "Solve 2+2.")

    assert result.hint == "Try again."
    assert result.state.subskills["arithmetic_execution"] == 0.55


@pytest.mark.gpu
def test_infer_and_hint_against_real_smoke_checkpoint(tmp_path):
    """Requires Task 6's smoke test to have been run first, producing a
    checkpoint at the given path. Run manually with a real checkpoint dir:
    pytest tests/backends/test_finetuned.py -v -m gpu --checkpoint-dir=<path>
    This test intentionally has no fixture wiring here — see the task's
    manual-run note below."""
    pytest.skip("Run manually against a real checkpoint; see docstring.")
