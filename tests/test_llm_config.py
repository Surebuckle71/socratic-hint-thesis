import pytest

from socratic_hint.llm_config import (
    DEFAULT_MODEL,
    MIN_MAX_TOKENS,
    extract_text,
    thinking_kwargs_for_model,
)


class Block:
    def __init__(self, type_: str, **fields):
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


class Response:
    def __init__(self, *blocks):
        self.content = list(blocks)


def test_extract_text_skips_leading_thinking_block():
    # This is the exact shape claude-sonnet-5 returns by default: adaptive
    # thinking is on, so content[0] is a ThinkingBlock with no `.text`.
    response = Response(
        Block("thinking", thinking="let me reason about this"),
        Block("text", text="the answer"),
    )
    assert extract_text(response) == "the answer"


def test_extract_text_handles_text_only_response():
    assert extract_text(Response(Block("text", text="hi"))) == "hi"


def test_extract_text_returns_first_text_block():
    response = Response(
        Block("thinking", thinking="..."),
        Block("text", text="first"),
        Block("text", text="second"),
    )
    assert extract_text(response) == "first"


def test_extract_text_raises_when_no_text_block():
    # Happens when max_tokens is exhausted by thinking tokens.
    response = Response(Block("thinking", thinking="..."))
    with pytest.raises(ValueError, match="No text block"):
        extract_text(response)


def test_thinking_kwargs_for_model_adaptive_shape():
    kwargs = thinking_kwargs_for_model("claude-sonnet-5")
    assert kwargs == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }


def test_thinking_kwargs_for_model_returns_fresh_dict():
    # Mutating one call's result must not leak into the next call site.
    first = thinking_kwargs_for_model("claude-sonnet-5")
    first["thinking"]["type"] = "mutated"
    assert thinking_kwargs_for_model("claude-sonnet-5")["thinking"]["type"] == "adaptive"


def test_thinking_kwargs_for_model_haiku_uses_budget_tokens():
    # Confirmed against a live API 400 during the pilot run: Haiku 4.5
    # rejects adaptive thinking outright and needs the older
    # enabled/budget_tokens form, with no `effort` field at all.
    kwargs = thinking_kwargs_for_model("claude-haiku-4-5")
    assert kwargs["thinking"]["type"] == "enabled"
    assert isinstance(kwargs["thinking"]["budget_tokens"], int)
    assert kwargs["thinking"]["budget_tokens"] < MIN_MAX_TOKENS
    assert "output_config" not in kwargs


def test_shared_constants():
    assert DEFAULT_MODEL == "claude-sonnet-5"
    # Thinking tokens are billed against max_tokens, so the floor must leave
    # room for both the reasoning and the visible answer.
    assert MIN_MAX_TOKENS >= 1024
