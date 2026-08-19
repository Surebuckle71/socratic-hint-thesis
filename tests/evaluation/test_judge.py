import json
from unittest.mock import MagicMock

import pytest

from socratic_hint.evaluation.judge import JUDGE_DIMENSIONS, PedagogicalQualityJudge


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


def make_fake_client(payload: dict) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = FakeResponse(json.dumps(payload))
    return client


def test_score_parses_all_dimensions():
    payload = {
        "scaffolding_vs_telling": 4,
        "correctness": 5,
        "appropriateness": 3,
        "rationale": "Good scaffolding hint.",
    }
    client = make_fake_client(payload)
    judge = PedagogicalQualityJudge(client=client)

    score = judge.score("Solve 2+2.", "Tutor: hi", "Try again.")

    for dim in JUDGE_DIMENSIONS:
        assert score.scores[dim] == payload[dim]
    assert score.rationale == "Good scaffolding hint."

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "low"}
    # Thinking tokens count against max_tokens; 256 would risk emitting no
    # text block at all.
    assert call_kwargs["max_tokens"] >= 1024


def test_score_raises_on_malformed_json():
    client = MagicMock()
    client.messages.create.return_value = FakeResponse("not json at all")
    judge = PedagogicalQualityJudge(client=client)

    with pytest.raises(json.JSONDecodeError):
        judge.score("problem", "context", "hint")
