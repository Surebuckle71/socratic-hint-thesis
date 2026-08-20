from dataclasses import dataclass

from socratic_hint.backends.base import HintBackend
from socratic_hint.types import DialogueTurn

# How this metric works, and why it needs a control
# --------------------------------------------------
# "Adaptivity" here means: does the hint change when the student-state history
# changes? The signal is exact-text inequality between the hint generated from
# a low-mastery history and the hint generated from a high-mastery history.
#
# Exact-text inequality on its own is NOT a usable measure. Both backends decode
# stochastically, so two independent samples of the SAME prompt almost always
# differ in wording — a completely non-adaptive model would score ~100%
# "adaptivity", and the number would be uninterpretable.
#
# We therefore measure a same-history noise floor alongside the signal: a third
# generation from the *same* (low-mastery) history, compared against the first.
# That control captures exactly the sampling variation that is not attributable
# to the state conditioning. The reported rate is the excess of the differ rate
# over the noise floor:
#
#     adaptivity = P(differ | different history) - P(differ | same history)
#
# A non-adaptive stochastic model nets out to ~0. A genuinely state-sensitive
# model shows a positive excess. The net rate can be negative when noise
# exceeds signal; it is reported as-is rather than clamped, because a negative
# value is real evidence against adaptivity and hiding it would overstate the
# thesis's central claim.
#
# Exact-text comparison remains a conservative proxy — semantically identical
# hints with different wording count as "different". That limitation is
# accepted for scoping purposes (see
# docs/superpowers/specs/2026-08-18-system-architecture-design.md) rather than
# adding an LLM-judge semantic-difference check; the noise-floor control is
# what makes the proxy interpretable, since re-wording inflates the signal and
# the control equally and cancels in the difference.


@dataclass(frozen=True)
class StateAdaptivityPair:
    problem: str
    low_mastery_history: list[DialogueTurn]
    high_mastery_history: list[DialogueTurn]


@dataclass(frozen=True)
class StateAdaptivityResult:
    low_mastery_hint: str
    high_mastery_hint: str
    hints_differ: bool
    # Same-history repeat sample, for comparison against `hints_differ`.
    noise_control_differ: bool


@dataclass(frozen=True)
class StateAdaptivityBatchResult:
    """Batch-level adaptivity measurement, with failures made visible.

    `net_rate` is None when every pair failed — distinct from a measured 0.0.
    """

    net_rate: float | None
    differ_rate: float | None
    noise_rate: float | None
    evaluated_pairs: int
    failed_pairs: int


class StateAdaptivityDiagnostic:
    def run(
        self,
        backend: HintBackend,
        pair: StateAdaptivityPair,
        suppress_state: bool = False,
    ) -> StateAdaptivityResult:
        """Generate the signal pair plus a same-history noise control.

        Costs three generations per pair: low-mastery, high-mastery, and a
        repeat of low-mastery that establishes the sampling-noise floor.
        """
        low_result = backend.infer_and_hint(
            pair.low_mastery_history, pair.problem, suppress_state=suppress_state
        )
        high_result = backend.infer_and_hint(
            pair.high_mastery_history, pair.problem, suppress_state=suppress_state
        )
        # Noise-floor control: a second independent sample from the SAME
        # history as `low_result`. Any difference here is pure decoding noise.
        control_result = backend.infer_and_hint(
            pair.low_mastery_history, pair.problem, suppress_state=suppress_state
        )
        return StateAdaptivityResult(
            low_mastery_hint=low_result.hint,
            high_mastery_hint=high_result.hint,
            hints_differ=low_result.hint.strip() != high_result.hint.strip(),
            noise_control_differ=low_result.hint.strip() != control_result.hint.strip(),
        )

    def run_batch(
        self,
        backend: HintBackend,
        pairs: list[StateAdaptivityPair],
        suppress_state: bool = False,
    ) -> StateAdaptivityBatchResult:
        """Adaptivity rate net of the sampling-noise floor.

        `net_rate` is `P(differ | different history) - P(differ | same
        history)`. It can be negative when noise exceeds signal — reported
        honestly rather than clamped to zero.

        Each pair costs three generations, so a full run issues thousands of
        them. A single unparseable generation (`parse_model_output` raises
        `ValueError` on, say, a bolded hint line) must not destroy a multi-hour
        run, so per-pair failures are caught, counted, and excluded from the
        rates — mirroring how `evaluate_condition` handles its per-example
        loop.

        Raises:
            ValueError: if `pairs` is empty. Returning 0.0 for no input would
                be indistinguishable from "ran and measured zero adaptivity".
        """
        if not pairs:
            raise ValueError("adaptivity_pairs must not be empty")

        results: list[StateAdaptivityResult] = []
        failed = 0
        for pair in pairs:
            try:
                results.append(self.run(backend, pair, suppress_state=suppress_state))
            except Exception:  # noqa: BLE001 - one bad pair must not kill the batch
                failed += 1

        if not results:
            return StateAdaptivityBatchResult(
                net_rate=None,
                differ_rate=None,
                noise_rate=None,
                evaluated_pairs=0,
                failed_pairs=failed,
            )

        differ_rate = sum(1 for r in results if r.hints_differ) / len(results)
        noise_rate = sum(1 for r in results if r.noise_control_differ) / len(results)
        return StateAdaptivityBatchResult(
            net_rate=differ_rate - noise_rate,
            differ_rate=differ_rate,
            noise_rate=noise_rate,
            evaluated_pairs=len(results),
            failed_pairs=failed,
        )
