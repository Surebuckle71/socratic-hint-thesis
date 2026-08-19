from unittest.mock import MagicMock, patch

import pytest

from socratic_hint.backends.finetuned import FinetunedBackend
from socratic_hint.types import DialogueTurn


def test_infer_and_hint_parses_generated_text(tmp_path):
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    fake_inputs = {"input_ids": MagicMock(shape=(1, 10))}
    fake_tokenizer.return_value.to.return_value = fake_inputs
    output_row = list(range(20))
    fake_model.generate.return_value = [output_row]
    fake_tokenizer.decode.return_value = "State: arithmetic_execution=0.55\nHint: Try again."
    fake_model.device = "cpu"

    with patch("socratic_hint.backends.finetuned.FastLanguageModel") as fake_flm:
        fake_flm.from_pretrained.return_value = (fake_model, fake_tokenizer)
        backend = FinetunedBackend(checkpoint_dir=tmp_path)

    result = backend.infer_and_hint([], "Solve 2+2.")

    assert result.hint == "Try again."
    assert result.state.subskills["arithmetic_execution"] == 0.55

    # generate() was actually called with max_new_tokens=512
    assert fake_model.generate.call_args.kwargs["max_new_tokens"] == 512

    # decode() received the correctly-sliced generated tokens (offset by the
    # prompt length, i.e. inputs["input_ids"].shape[1] == 10), not just
    # whatever MagicMock's auto __getitem__ happened to hand back
    decoded_tokens = fake_tokenizer.decode.call_args.args[0]
    assert list(decoded_tokens) == output_row[10:]
    assert fake_tokenizer.decode.call_args.kwargs["skip_special_tokens"] is True

    # prompt sent to the tokenizer reflects suppress_state=False (default)
    prompt_arg = fake_tokenizer.call_args.args[0]
    assert "Solve 2+2." in prompt_arg
    assert "State:" in prompt_arg


def test_infer_and_hint_suppress_state_passes_through(tmp_path):
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    fake_inputs = {"input_ids": MagicMock(shape=(1, 10))}
    fake_tokenizer.return_value.to.return_value = fake_inputs
    output_row = list(range(20))
    fake_model.generate.return_value = [output_row]
    fake_tokenizer.decode.return_value = "Hint: Try again."
    fake_model.device = "cpu"

    with patch("socratic_hint.backends.finetuned.FastLanguageModel") as fake_flm:
        fake_flm.from_pretrained.return_value = (fake_model, fake_tokenizer)
        backend = FinetunedBackend(checkpoint_dir=tmp_path)

    result = backend.infer_and_hint(
        [DialogueTurn(speaker="tutor", text="hi")], "Solve 2+2.", suppress_state=True
    )

    assert result.state is None
    assert result.hint == "Try again."

    # prompt sent to the tokenizer reflects suppress_state=True
    prompt_arg = fake_tokenizer.call_args.args[0]
    assert "State:" not in prompt_arg


@pytest.mark.gpu
def test_infer_and_hint_against_real_smoke_checkpoint(tmp_path):
    """Requires Task 6's smoke test to have been run first, producing a
    checkpoint at the given path. Run manually with a real checkpoint dir:
    pytest tests/backends/test_finetuned.py -v -m gpu --checkpoint-dir=<path>
    This test intentionally has no fixture wiring here — see the task's
    manual-run note below."""
    pytest.skip("Run manually against a real checkpoint; see docstring.")
