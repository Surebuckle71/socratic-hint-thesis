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


def parse_model_output(raw_output: str, suppress_state: bool = False) -> HintResult:
    hint_line = None
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("hint:"):
            hint_line = stripped.split(":", 1)[1].strip()
            break

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
