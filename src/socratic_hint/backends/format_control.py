from socratic_hint.backends.base import HintBackend
from socratic_hint.backends.finetuned import FinetunedBackend
from socratic_hint.output_format import SUBSKILLS
from socratic_hint.types import DialogueTurn, HintResult, StateEstimate

# Fixed, uninformative state: identical for every dialogue and every turn, so
# it carries zero information about the actual conversation. This is the
# "neutral" state referred to throughout - not derived from the dialogue in
# any way, unlike Condition 3's self-estimated state.
NEUTRAL_STATE = {s: 0.5 for s in SUBSKILLS}


class FormatControlBackend(HintBackend):
    """Format-control ablation for Condition 2 (state-suppressed).

    Condition 2 and Condition 3 differ in two ways at once, not one: Condition
    3 both (a) conditions on state information and (b) emits a longer
    generation (a `State:` line before the `Hint:` line). Condition 2's
    quality gap against Condition 3 could therefore come from the missing
    state information, or just from generating fewer tokens before the hint -
    the original design cannot separate these.

    This backend isolates (b): it forces the SAME fine-tuned weights to emit
    a `State:` line of the same length and format as Condition 3, but with a
    fixed, dialogue-independent value that carries no real information about
    the conversation. Any quality difference between this condition and
    Condition 2 proper is attributable to the extra generation step alone,
    holding state-informativeness at zero throughout.
    """

    def __init__(self, finetuned: FinetunedBackend):
        self.finetuned = finetuned

    def infer_and_hint(
        self,
        dialogue_history: list[DialogueTurn],
        problem: str,
        suppress_state: bool = False,
    ) -> HintResult:
        # suppress_state is part of the HintBackend interface but is not
        # meaningful here: this condition always emits the format-matched,
        # neutral-state completion regardless of the caller's flag, since the
        # whole point is to hold state-informativeness fixed at zero while
        # keeping the Condition-3-shaped output format.
        hint = self.finetuned.infer_hint_given_state(
            dialogue_history, problem, NEUTRAL_STATE
        )
        return HintResult(hint=hint, state=StateEstimate(subskills=dict(NEUTRAL_STATE)))
