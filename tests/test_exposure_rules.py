from __future__ import annotations

import math

import pytest

from vrp.strategies.exposure_rules import (
    BLOCK_INVALID_PROBABILITY_SUM,
    BLOCK_MISSING_HAR_FORECAST,
    BLOCK_MISSING_PROBABILITIES,
    BLOCK_MISSING_VRP_HAR,
    BLOCK_NONE,
    DEFAULT_STRESS_PROBABILITY_CUTOFF,
    FLAT_EXPOSURE,
    FULL_SHORT_VOL_EXPOSURE,
    ProbabilityValidationError,
    apply_probability_carry_gate,
    clip_short_vol_exposure,
    probability_linear_decision,
    probability_linear_exposure,
    threshold_defensive_exposure,
    threshold_hard_filter_exposure,
    unconditional_full_exposure,
    validate_probability_triplet,
)


def test_unconditional_full_returns_minus_one() -> None:
    assert unconditional_full_exposure() == -1.0


def test_clip_short_vol_exposure_bounds_positive_to_zero() -> None:
    assert clip_short_vol_exposure(0.75) == 0.0


def test_clip_short_vol_exposure_bounds_too_negative_to_minus_one() -> None:
    assert clip_short_vol_exposure(-1.75) == -1.0


def test_clip_short_vol_exposure_keeps_internal_value() -> None:
    assert clip_short_vol_exposure(-0.37) == pytest.approx(-0.37)


def test_clip_short_vol_exposure_rejects_nan() -> None:
    with pytest.raises(ValueError):
        clip_short_vol_exposure(math.nan)


def test_threshold_hard_filter_blocks_stress() -> None:
    assert threshold_hard_filter_exposure("stress") == 0.0


def test_threshold_hard_filter_keeps_full_exposure_for_calm() -> None:
    assert threshold_hard_filter_exposure("calm") == -1.0


def test_threshold_hard_filter_keeps_full_exposure_for_transition() -> None:
    assert threshold_hard_filter_exposure("transition") == -1.0


def test_threshold_hard_filter_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        threshold_hard_filter_exposure("panic")


def test_threshold_defensive_maps_all_states() -> None:
    assert threshold_defensive_exposure("calm") == -1.0
    assert threshold_defensive_exposure("transition") == -0.25
    assert threshold_defensive_exposure("stress") == 0.0


def test_threshold_defensive_is_case_and_space_tolerant() -> None:
    assert threshold_defensive_exposure(" Calm ") == -1.0
    assert threshold_defensive_exposure(" TRANSITION ") == -0.25
    assert threshold_defensive_exposure(" Stress ") == 0.0


def test_probability_linear_uses_calm_minus_stress() -> None:
    exposure = probability_linear_exposure(p_calm=0.70, p_stress=0.20)
    assert exposure == pytest.approx(-0.50)


def test_probability_linear_goes_flat_when_stress_exceeds_calm() -> None:
    exposure = probability_linear_exposure(p_calm=0.20, p_stress=0.70)
    assert exposure == 0.0


def test_probability_linear_clips_to_full_short_vol() -> None:
    exposure = probability_linear_exposure(p_calm=1.0, p_stress=0.0)
    assert exposure == -1.0


def test_probability_linear_rejects_probability_above_one() -> None:
    with pytest.raises(ProbabilityValidationError):
        probability_linear_exposure(p_calm=1.2, p_stress=0.0)


def test_probability_triplet_accepts_exact_sum_one() -> None:
    result = validate_probability_triplet(
        p_calm=0.50,
        p_transition=0.30,
        p_stress=0.20,
    )
    assert result == pytest.approx((0.50, 0.30, 0.20))


def test_probability_triplet_accepts_small_tolerance_error() -> None:
    result = validate_probability_triplet(
        p_calm=0.50,
        p_transition=0.30,
        p_stress=0.2005,
        probability_sum_tolerance=0.001,
    )
    assert result == pytest.approx((0.50, 0.30, 0.2005))


def test_probability_triplet_rejects_sum_outside_tolerance() -> None:
    with pytest.raises(ProbabilityValidationError) as excinfo:
        validate_probability_triplet(
            p_calm=0.50,
            p_transition=0.30,
            p_stress=0.205,
            probability_sum_tolerance=0.001,
        )

    assert excinfo.value.reason == BLOCK_INVALID_PROBABILITY_SUM


def test_probability_triplet_rejects_missing_probability() -> None:
    with pytest.raises(ProbabilityValidationError) as excinfo:
        validate_probability_triplet(
            p_calm=None,
            p_transition=0.30,
            p_stress=0.20,
        )

    assert excinfo.value.reason == BLOCK_MISSING_PROBABILITIES


def test_probability_linear_decision_marks_invalid_sum_unavailable() -> None:
    decision = probability_linear_decision(
        p_calm=0.50,
        p_transition=0.30,
        p_stress=0.205,
        probability_sum_tolerance=0.001,
    )

    assert decision.strategy_available is False
    assert math.isnan(decision.target_exposure)
    assert decision.blocked_reason == BLOCK_INVALID_PROBABILITY_SUM
    assert decision.decision_reason == "unavailable"


def test_probability_linear_decision_available_for_valid_inputs() -> None:
    decision = probability_linear_decision(
        p_calm=0.60,
        p_transition=0.25,
        p_stress=0.15,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == pytest.approx(-0.45)
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "probability_linear"


def test_carry_gate_missing_har_forecast_is_unavailable() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.60,
        p_transition=0.25,
        p_stress=0.15,
        vrp_har_gk=0.02,
        har_forecast_available=False,
    )

    assert decision.strategy_available is False
    assert math.isnan(decision.target_exposure)
    assert decision.blocked_reason == BLOCK_MISSING_HAR_FORECAST
    assert decision.decision_reason == "unavailable"


def test_carry_gate_missing_vrp_har_is_unavailable() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.60,
        p_transition=0.25,
        p_stress=0.15,
        vrp_har_gk=None,
        har_forecast_available=True,
    )

    assert decision.strategy_available is False
    assert math.isnan(decision.target_exposure)
    assert decision.blocked_reason == BLOCK_MISSING_VRP_HAR


def test_carry_gate_uses_numeric_vrp_har_gk_not_boolean_flag() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.60,
        p_transition=0.25,
        p_stress=0.15,
        vrp_har_gk=-0.01,
        har_forecast_available=True,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == FLAT_EXPOSURE
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "negative_or_zero_vrp_har"


def test_carry_gate_zero_vrp_har_is_valid_flat_not_unavailable() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.60,
        p_transition=0.25,
        p_stress=0.15,
        vrp_har_gk=0.0,
        har_forecast_available=True,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == FLAT_EXPOSURE
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "negative_or_zero_vrp_har"


def test_carry_gate_blocks_stress_probability_greater_than_cutoff() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.50,
        p_transition=0.09,
        p_stress=0.41,
        vrp_har_gk=0.02,
        har_forecast_available=True,
        stress_probability_cutoff=DEFAULT_STRESS_PROBABILITY_CUTOFF,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == FLAT_EXPOSURE
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "stress_probability_veto"


def test_carry_gate_does_not_block_equal_040() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.50,
        p_transition=0.10,
        p_stress=0.40,
        vrp_har_gk=0.02,
        har_forecast_available=True,
        stress_probability_cutoff=DEFAULT_STRESS_PROBABILITY_CUTOFF,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == pytest.approx(-0.10)
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "probability_linear_carry"


def test_carry_gate_applies_probability_exposure_when_all_gates_pass() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.70,
        p_transition=0.20,
        p_stress=0.10,
        vrp_har_gk=0.03,
        har_forecast_available=True,
    )

    assert decision.strategy_available is True
    assert decision.target_exposure == pytest.approx(-0.60)
    assert decision.blocked_reason == BLOCK_NONE
    assert decision.decision_reason == "probability_linear_carry"


def test_carry_gate_invalid_probability_sum_is_unavailable() -> None:
    decision = apply_probability_carry_gate(
        p_calm=0.70,
        p_transition=0.20,
        p_stress=0.20,
        vrp_har_gk=0.03,
        har_forecast_available=True,
    )

    assert decision.strategy_available is False
    assert math.isnan(decision.target_exposure)
    assert decision.blocked_reason == BLOCK_INVALID_PROBABILITY_SUM
    assert decision.decision_reason == "unavailable"


def test_all_available_decisions_are_bounded() -> None:
    decisions = [
        apply_probability_carry_gate(
            p_calm=0.70,
            p_transition=0.20,
            p_stress=0.10,
            vrp_har_gk=0.03,
            har_forecast_available=True,
        ),
        apply_probability_carry_gate(
            p_calm=0.50,
            p_transition=0.09,
            p_stress=0.41,
            vrp_har_gk=0.03,
            har_forecast_available=True,
        ),
        probability_linear_decision(
            p_calm=0.60,
            p_transition=0.30,
            p_stress=0.10,
        ),
    ]

    for decision in decisions:
        assert decision.strategy_available is True
        assert -1.0 <= decision.target_exposure <= 0.0


def test_exposure_constants_match_phase9_convention() -> None:
    assert FULL_SHORT_VOL_EXPOSURE == -1.0
    assert FLAT_EXPOSURE == 0.0