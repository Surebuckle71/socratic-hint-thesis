from socratic_hint.types import DialogueTurn, HintResult, StateEstimate

# Immutable on purpose: this is shared module-level state read by the prompt
# formatter, the state-label deriver, and the tests. A tuple prevents a
# consumer from mutating the canonical subskill list in place.
SUBSKILLS: tuple[str, ...] = (
    "problem_comprehension",
    "arithmetic_execution",
    "step_sequencing",
    "self_correction",
)


# Text placed between a prompt and its completion when the two are joined into
# one training sequence. It belongs to the PROMPT side: Qwen's BPE merges the
# prompt's final `>` with a following `\n\n` into a single `>\n\n` token, so a
# separator on the completion side leaves the prompt/completion token boundary
# straddling one token — TRL warns about exactly this, and the local
# fine-tuned backend must append it too so its inference context is
# byte-identical to what training saw.
PROMPT_COMPLETION_SEPARATOR = "\n\n"


def format_prompt(
    dialogue_history: list[DialogueTurn], problem: str, suppress_state: bool = False
) -> str:
    lines = [f"Problem: {problem}", ""]
    for turn in dialogue_history:
        label = "Tutor" if turn.speaker == "tutor" else "Student"
        lines.append(f"{label}: {turn.text}")
    lines.append("")
    if suppress_state:
        lines.append("Respond with the tutor's next Socratic hint only. Do not include a state estimate.")
        lines.append("Format your response exactly as:")
        lines.append("Hint: <hint text>")
    else:
        lines.append(
            "First estimate the student's current mastery of each subskill "
            f"({', '.join(SUBSKILLS)}) as a probability between 0 and 1, "
            "then give the tutor's next Socratic hint, conditioned on that estimate."
        )
        lines.append("Format your response exactly as:")
        lines.append("State: <subskill1>=<0.xx>, <subskill2>=<0.xx>, ...")
        lines.append("Hint: <hint text>")
    return "\n".join(lines)


def format_completion(state: StateEstimate, hint: str) -> str:
    state_str = ", ".join(f"{k}={v:.2f}" for k, v in state.subskills.items())
    return f"State: {state_str}\nHint: {hint}"


def format_hint_only_completion(hint: str) -> str:
    """The state-suppressed training target: exactly what
    `parse_model_output(..., suppress_state=True)` expects to parse back."""
    return f"Hint: {hint}"


# Field prefixes that terminate a multi-line hint. A `Hint:` block runs to the
# end of the generation unless one of these starts a later line.
_FIELD_PREFIXES = ("state:", "hint:")


def _extract_hint(raw_output: str) -> str | None:
    """Text of the `Hint:` field, spanning every line up to the next field.

    Hints routinely run to several sentences or lines (generation allows
    thousands of tokens). Returning only the remainder of the `Hint:` line
    would silently truncate them, and both the simulated student and the judge
    would then see only the fragment.
    """
    lines = raw_output.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().lower().startswith("hint:"):
            continue
        collected = [line.strip().split(":", 1)[1].strip()]
        for later in lines[i + 1:]:
            if later.strip().lower().startswith(_FIELD_PREFIXES):
                break
            collected.append(later)
        return "\n".join(collected).strip()
    return None


def parse_model_output(raw_output: str, suppress_state: bool = False) -> HintResult:
    hint_line = _extract_hint(raw_output)

    if suppress_state:
        if hint_line is None:
            raise ValueError(f"Could not parse hint from model output: {raw_output!r}")
        return HintResult(hint=hint_line, state=None)

    state_line = None
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("state:"):
            state_line = stripped.split(":", 1)[1].strip()
            break

    if state_line is None or hint_line is None:
        raise ValueError(f"Could not parse state/hint from model output: {raw_output!r}")

    subskills: dict[str, float] = {}
    for pair in state_line.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        subskills[name.strip()] = float(value.strip())

    return HintResult(hint=hint_line, state=StateEstimate(subskills=subskills))
