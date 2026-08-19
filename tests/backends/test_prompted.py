from unittest.mock import MagicMock

from socratic_hint.backends.prompted import PromptedBackend
from socratic_hint.types import DialogueTurn


class FakeThinkingBlock:
    """Mirrors a real ThinkingBlock: has `.thinking`, NOT `.text`."""

    def __init__(self, thinking: str = "reasoning..."):
        self.type = "thinking"
        self.thinking = thinking


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeResponse:
    """Adaptive thinking is on by default for claude-sonnet-5, so a real
    response leads with a ThinkingBlock — content[0] is not the answer."""

    def __init__(self, text: str):
        self.content = [FakeThinkingBlock(), FakeTextBlock(text)]


def make_fake_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = FakeResponse(response_text)
    return client


def test_infer_and_hint_parses_state_and_hint_from_response():
    client = make_fake_client("State: arithmetic_execution=0.60\nHint: Try the next step.")
    backend = PromptedBackend(client=client, model="claude-sonnet-5")

    result = backend.infer_and_hint([], "Solve 2+2.")

    assert result.hint == "Try the next step."
    assert result.state.subskills["arithmetic_execution"] == 0.60
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert "Solve 2+2." in call_kwargs["messages"][0]["content"]
    # Thinking is requested explicitly, and max_tokens leaves room for it.
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "low"}
    assert call_kwargs["max_tokens"] >= 1024


def test_infer_and_hint_suppress_state_passes_through():
    client = make_fake_client("Hint: Try the next step.")
    backend = PromptedBackend(client=client, model="claude-sonnet-5")

    result = backend.infer_and_hint(
        [DialogueTurn(speaker="tutor", text="hi")], "Solve 2+2.", suppress_state=True
    )

    assert result.state is None
    assert result.hint == "Try the next step."
    call_kwargs = client.messages.create.call_args.kwargs
    assert "State:" not in call_kwargs["messages"][0]["content"]
