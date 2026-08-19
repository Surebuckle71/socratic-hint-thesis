"""Shared configuration for every real Anthropic API call in this project.

All API-calling modules (the prompted backend, the LLM judge, the simulated
student) import from here so that model choice and request shape stay
consistent in one place.

Why `thinking` and `output_config` are set explicitly
-----------------------------------------------------
`claude-sonnet-5` runs adaptive thinking BY DEFAULT when the `thinking`
parameter is omitted. That has two consequences that silently break naive
call sites:

1. `response.content` is a list of typed blocks. With thinking on, the first
   block is a `ThinkingBlock` (which exposes `.thinking`, not `.text`), so
   `response.content[0].text` raises `AttributeError`. Always locate the text
   block by its `type` — that is what `extract_text` does.
2. Thinking tokens count against `max_tokens`. A budget sized for the visible
   answer alone (e.g. 8 tokens for a YES/NO classification) is consumed by
   thinking and never emits any text block at all.

We keep thinking ON at `effort: "low"` rather than disabling it: these are
short classification / short-generation tasks that do not need deep
reasoning, and disabling thinking outright has its own documented failure
modes (leaked reasoning tags, tool calls written into visible text).
`MIN_MAX_TOKENS` is the floor every call site uses so thinking always has
room to finish before the visible answer is produced.
"""

DEFAULT_MODEL = "claude-sonnet-5"

# Floor for max_tokens on every call. Thinking tokens are billed against
# max_tokens, so anything smaller risks exhausting the budget on thinking
# alone and returning no text block.
MIN_MAX_TOKENS = 1024


def default_thinking_kwargs() -> dict:
    """Request kwargs enabling adaptive thinking at low effort.

    Spread into `client.messages.create(...)` at every call site.
    """
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }


def extract_text(response) -> str:
    """Return the text of the first `text` block in an API response.

    Skips `thinking` (and any other non-text) blocks, which is required
    whenever thinking is enabled — `response.content[0]` is not the answer.

    Raises:
        ValueError: if the response contains no text block at all (e.g. the
            entire `max_tokens` budget was spent on thinking).
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(
        "No text block in API response; content block types were "
        f"{[getattr(b, 'type', None) for b in response.content]}. "
        "This usually means max_tokens was exhausted by thinking tokens."
    )
