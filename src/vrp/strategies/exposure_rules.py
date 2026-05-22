from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final, cast


MIN_EXPOSURE: Final[float] = -1.0
MAX_EXPOSURE: Final[float] = 0.0
FULL_SHORT_VOL_EXPOSURE: Final[float] = -1.0
FLAT_EXPOSURE: Final[float] = 0.0
DEFAULT_TRANSITION_EXPOSURE: Final[float] = -0.25
DEFAULT_STRESS_PROBABILITY_CUTOFF: Final[float] = 0.40
DEFAULT_PROBABILITY_SUM_TOLERANCE: Final[float] = 0.001

STATE_CALM: Final[str] = "calm"
STATE_TRANSITION: Final[str] = "transition"
STATE_STRESS: Final[str] = "stress"

VALID_STATE_NAMES: Final[set[str]] = {
    STATE_CALM,
    STATE_TRANSITION,
    STATE_STRESS,
}

BLOCK_NONE: Final[str] = "none"
DECISION_UNAVAILABLE: Final[str] = "unavailable"

BLOCK_MISSING_PROBABILITIES: Final[str] = "missing_probabilities"
BLOCK_INVALID_PROBABILITY_VALUE: Final[str] = "invalid_probability_value"
BLOCK_INVALID_PROBABILITY_SUM: Final[str] = "invalid_probability_sum"
BLOCK_MISSING_HAR_FORECAST: Final[str] = "missing_har_forecast"
BLOCK_MISSING_VRP_HAR: Final[str] = "missing_vrp_har_gk"


@dataclass(frozen=True)
class ExposureDecision:
    """
    Single-rule exposure decision.

    strategy_available:
        False means the rule could not be evaluated because required inputs
        were missing or invalid.

    target_exposure:
        Exposure convention:
            -1.0 = full short-vol exposure
             0.0 = flat/no short-vol exposure

        Unavailable decisions use NaN.

    blocked_reason:
        Explains missing/invalid inputs only.

    decision_reason:
        Explains a valid rule decision, including valid flat decisions.
    """

    strategy_available: bool
    target_exposure: float
    blocked_reason: str
    decision_reason: str


class ProbabilityValidationError(ValueError):
    """
    Raised when HMM/MAR probability inputs are unavailable or invalid.
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def unavailable_decision(blocked_reason: str) -> ExposureDecision:
    return ExposureDecision(
        strategy_available=False,
        target_exposure=math.nan,
        blocked_reason=blocked_reason,
        decision_reason=DECISION_UNAVAILABLE,
    )


def available_decision(target_exposure: float, decision_reason: str) -> ExposureDecision:
    return ExposureDecision(
        strategy_available=True,
        target_exposure=clip_short_vol_exposure(target_exposure),
        blocked_reason=BLOCK_NONE,
        decision_reason=decision_reason,
    )


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _coerce_finite_float(value: object, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be a real numeric value, got {value!r}.")

    result = float(cast(Real, value))

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")

    return result


def _coerce_probability(value: object, name: str) -> float:
    if not _is_number(value):
        raise ProbabilityValidationError(
            reason=BLOCK_MISSING_PROBABILITIES,
            message=f"{name} must be a finite probability, got {value!r}.",
        )

    result = float(cast(Real, value))

    if not math.isfinite(result):
        raise ProbabilityValidationError(
            reason=BLOCK_MISSING_PROBABILITIES,
            message=f"{name} must be finite, got {value!r}.",
        )

    if result < 0.0 or result > 1.0:
        raise ProbabilityValidationError(
            reason=BLOCK_INVALID_PROBABILITY_VALUE,
            message=f"{name} must lie in [0, 1], got {result}.",
        )

    return result


def normalise_state_name(state_name: object) -> str:
    if state_name is None:
        raise ValueError("state_name must not be None.")

    state = str(state_name).strip().lower()

    if state not in VALID_STATE_NAMES:
        raise ValueError(
            f"Unknown regime state '{state_name}'. "
            f"Expected one of {sorted(VALID_STATE_NAMES)}."
        )

    return state


def clip_short_vol_exposure(
    exposure: object,
    min_exposure: float = MIN_EXPOSURE,
    max_exposure: float = MAX_EXPOSURE,
) -> float:
    """
    Clip target exposure to the Phase 9 short-vol interval [-1.0, 0.0].

    Positive values are clipped to 0.0.
    Values below -1.0 are clipped to -1.0.
    """
    value = _coerce_finite_float(exposure, "exposure")

    if min_exposure > max_exposure:
        raise ValueError(
            f"min_exposure must be <= max_exposure, got "
            f"{min_exposure} > {max_exposure}."
        )

    return min(max(value, min_exposure), max_exposure)


def unconditional_full_exposure() -> float:
    """
    S0: unconditional full short-vol benchmark.
    """
    return FULL_SHORT_VOL_EXPOSURE


def threshold_hard_filter_exposure(state_name: object) -> float:
    """
    S1: threshold hard stress filter.

    stress      -> 0.0
    calm        -> -1.0
    transition  -> -1.0
    """
    state = normalise_state_name(state_name)

    if state == STATE_STRESS:
        return FLAT_EXPOSURE

    return FULL_SHORT_VOL_EXPOSURE


def threshold_defensive_exposure(
    state_name: object,
    transition_exposure: float = DEFAULT_TRANSITION_EXPOSURE,
) -> float:
    """
    S2: threshold defensive mapping.

    calm        -> -1.0
    transition  -> -0.25
    stress      -> 0.0
    """
    state = normalise_state_name(state_name)
    transition_value = clip_short_vol_exposure(transition_exposure)

    if state == STATE_CALM:
        return FULL_SHORT_VOL_EXPOSURE

    if state == STATE_TRANSITION:
        return transition_value

    return FLAT_EXPOSURE


def validate_probability_triplet(
    p_calm: object,
    p_transition: object,
    p_stress: object,
    probability_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
) -> tuple[float, float, float]:
    """
    Validate HMM/MAR filtered probabilities.

    Requirements:
        - all three probabilities are finite
        - each lies in [0, 1]
        - sum is approximately one
        - no silent renormalisation
    """
    tolerance = _coerce_finite_float(
        probability_sum_tolerance,
        "probability_sum_tolerance",
    )

    if tolerance <= 0.0:
        raise ValueError("probability_sum_tolerance must be positive.")

    calm = _coerce_probability(p_calm, "p_calm")
    transition = _coerce_probability(p_transition, "p_transition")
    stress = _coerce_probability(p_stress, "p_stress")

    probability_sum = calm + transition + stress

    if abs(probability_sum - 1.0) > tolerance:
        raise ProbabilityValidationError(
            reason=BLOCK_INVALID_PROBABILITY_SUM,
            message=(
                "Probability triplet must sum to one within tolerance. "
                f"Got sum={probability_sum:.12f}, "
                f"tolerance={tolerance:.12f}."
            ),
        )

    return calm, transition, stress


def probability_linear_exposure(
    p_calm: object,
    p_stress: object,
) -> float:
    """
    S3 primary probability-sizing rule.

    target_exposure = -clip(p_calm - p_stress, 0.0, 1.0)

    This function validates the two probabilities used by the linear margin
    rule but does not validate the full probability triplet. Use
    validate_probability_triplet() before this function when p_transition is
    available.
    """
    calm = _coerce_probability(p_calm, "p_calm")
    stress = _coerce_probability(p_stress, "p_stress")

    calm_minus_stress = calm - stress
    clipped_margin = min(max(calm_minus_stress, 0.0), 1.0)

    return clip_short_vol_exposure(-clipped_margin)


def probability_linear_decision(
    p_calm: object,
    p_transition: object,
    p_stress: object,
    probability_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
) -> ExposureDecision:
    """
    Valid non-carry HMM/MAR probability-sizing decision.
    """
    try:
        calm, _transition, stress = validate_probability_triplet(
            p_calm=p_calm,
            p_transition=p_transition,
            p_stress=p_stress,
            probability_sum_tolerance=probability_sum_tolerance,
        )
    except ProbabilityValidationError as exc:
        return unavailable_decision(exc.reason)

    exposure = probability_linear_exposure(calm, stress)
    return available_decision(exposure, "probability_linear")


def _coerce_har_forecast_available(value: object) -> bool:
    return bool(value) is True and value is True


def _coerce_vrp_har(value: object) -> float:
    if not _is_number(value):
        raise ValueError(f"vrp_har_gk must be finite numeric, got {value!r}.")

    result = float(cast(Real, value))

    if not math.isfinite(result):
        raise ValueError(f"vrp_har_gk must be finite numeric, got {value!r}.")

    return result


def apply_probability_carry_gate(
    p_calm: object,
    p_transition: object,
    p_stress: object,
    vrp_har_gk: object,
    har_forecast_available: object,
    stress_probability_cutoff: float = DEFAULT_STRESS_PROBABILITY_CUTOFF,
    probability_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
) -> ExposureDecision:
    """
    Carry-aware probability exposure decision.

    Required Phase 9 behaviour:
        - if HAR forecast is unavailable: unavailable
        - if probabilities are missing/invalid: unavailable
        - if vrp_har_gk <= 0: valid flat decision
        - if p_stress > 0.40: valid flat decision
        - otherwise: use calm-minus-stress probability sizing

    Important:
        p_stress == 0.40 does not trigger the stress-probability veto.
    """
    cutoff = _coerce_finite_float(
        stress_probability_cutoff,
        "stress_probability_cutoff",
    )

    if cutoff < 0.0 or cutoff > 1.0:
        raise ValueError("stress_probability_cutoff must lie in [0, 1].")

    if har_forecast_available is not True:
        return unavailable_decision(BLOCK_MISSING_HAR_FORECAST)

    try:
        calm, _transition, stress = validate_probability_triplet(
            p_calm=p_calm,
            p_transition=p_transition,
            p_stress=p_stress,
            probability_sum_tolerance=probability_sum_tolerance,
        )
    except ProbabilityValidationError as exc:
        return unavailable_decision(exc.reason)

    try:
        vrp_har = _coerce_vrp_har(vrp_har_gk)
    except ValueError:
        return unavailable_decision(BLOCK_MISSING_VRP_HAR)

    if vrp_har <= 0.0:
        return available_decision(
            target_exposure=FLAT_EXPOSURE,
            decision_reason="negative_or_zero_vrp_har",
        )

    if stress > cutoff:
        return available_decision(
            target_exposure=FLAT_EXPOSURE,
            decision_reason="stress_probability_veto",
        )

    exposure = probability_linear_exposure(calm, stress)

    return available_decision(
        target_exposure=exposure,
        decision_reason="probability_linear_carry",
    )


def apply_carry_gate(
    p_calm: object,
    p_transition: object,
    p_stress: object,
    vrp_har_gk: object,
    har_forecast_available: object,
    stress_probability_cutoff: float = DEFAULT_STRESS_PROBABILITY_CUTOFF,
    probability_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
) -> ExposureDecision:
    """
    Backward-compatible alias for the Phase 9 carry-aware probability gate.
    """
    return apply_probability_carry_gate(
        p_calm=p_calm,
        p_transition=p_transition,
        p_stress=p_stress,
        vrp_har_gk=vrp_har_gk,
        har_forecast_available=har_forecast_available,
        stress_probability_cutoff=stress_probability_cutoff,
        probability_sum_tolerance=probability_sum_tolerance,
    )


__all__ = [
    "MIN_EXPOSURE",
    "MAX_EXPOSURE",
    "FULL_SHORT_VOL_EXPOSURE",
    "FLAT_EXPOSURE",
    "DEFAULT_TRANSITION_EXPOSURE",
    "DEFAULT_STRESS_PROBABILITY_CUTOFF",
    "DEFAULT_PROBABILITY_SUM_TOLERANCE",
    "STATE_CALM",
    "STATE_TRANSITION",
    "STATE_STRESS",
    "VALID_STATE_NAMES",
    "BLOCK_NONE",
    "DECISION_UNAVAILABLE",
    "BLOCK_MISSING_PROBABILITIES",
    "BLOCK_INVALID_PROBABILITY_VALUE",
    "BLOCK_INVALID_PROBABILITY_SUM",
    "BLOCK_MISSING_HAR_FORECAST",
    "BLOCK_MISSING_VRP_HAR",
    "ExposureDecision",
    "ProbabilityValidationError",
    "unavailable_decision",
    "available_decision",
    "normalise_state_name",
    "clip_short_vol_exposure",
    "unconditional_full_exposure",
    "threshold_hard_filter_exposure",
    "threshold_defensive_exposure",
    "validate_probability_triplet",
    "probability_linear_exposure",
    "probability_linear_decision",
    "apply_probability_carry_gate",
    "apply_carry_gate",
]