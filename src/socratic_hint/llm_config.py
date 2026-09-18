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

Why thinking kwargs are chosen PER MODEL, not one-size-fits-all
----------------------------------------------------------------
Confirmed against a live 400 during the pilot run: adaptive thinking
(`{"type": "adaptive"}`) is supported on every current model EXCEPT
Haiku 4.5, which rejects it outright ("adaptive thinking is not supported
on this model") and still requires the older `{"type": "enabled",
"budget_tokens": N}` form. Haiku 4.5 also does not accept
`output_config.effort` at all. Any call site that lets its model be
swapped at runtime (the simulated student, via `--student-model`) must
route through `thinking_kwargs_for_model`, not a single hardcoded shape.
"""

import os

DEFAULT_MODEL = "claude-sonnet-5"

# Floor for max_tokens on every call. Thinking tokens are billed against
# max_tokens, so anything smaller risks exhausting the budget on thinking
# alone and returning no text block. Also large enough that Haiku 4.5's
# fixed `budget_tokens` (below) is comfortably less than max_tokens, which
# the API requires.
MIN_MAX_TOKENS = 2048

# Higher ceiling for call sites that must emit a non-trivial visible answer
# (a state line + hint, or a full JSON object) rather than a single word —
# at MIN_MAX_TOKENS, a longer-than-usual thinking pass can truncate the
# visible answer before parsing, turning a normal example into a silent
# per-example failure in evaluate_condition's error handling rather than a
# clear signal. Pure short-answer classification (e.g. YES/NO) can stay at
# MIN_MAX_TOKENS.
GENERATION_MAX_TOKENS = 4096

# Haiku 4.5's fixed thinking budget for the `enabled`/`budget_tokens` form
# (see `thinking_kwargs_for_model`). Comfortably under MIN_MAX_TOKENS, as
# the API requires `budget_tokens < max_tokens`.
_HAIKU_THINKING_BUDGET = 1024


def thinking_kwargs_for_model(model: str) -> dict:
    """Request kwargs enabling thinking, in the form the given model accepts.

    Spread into `client.messages.create(...)` at every call site. Do not use
    `default_thinking_kwargs` (removed) for a call site whose model can be
    swapped at runtime — Haiku 4.5 needs a different shape than every other
    current model (see module docstring).
    """
    if model.startswith("claude-haiku"):
        return {"thinking": {"type": "enabled", "budget_tokens": _HAIKU_THINKING_BUDGET}}
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }


def require_api_key() -> str:
    """The Anthropic API key, or a clear error explaining what to set.

    `os.environ["ANTHROPIC_API_KEY"]` raises a bare `KeyError:
    'ANTHROPIC_API_KEY'`, which says nothing about which component needed it or
    what to do — and it surfaces at construction time, potentially after a
    multi-GB checkpoint load. Every API-backed component routes through here so
    the failure is actionable.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. The prompted backend, the LLM judge, "
            "and the simulated student all make real Anthropic API calls. Set "
            "ANTHROPIC_API_KEY in your environment, or pass an explicit "
            "`client=Anthropic(...)` to the component."
        )
    return key


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
