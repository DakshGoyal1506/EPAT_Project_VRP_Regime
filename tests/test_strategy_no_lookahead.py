from __future__ import annotations

import math

import pandas as pd
import pytest

from vrp.strategies.signal_schema import (
    PHASE9_OUTPUT_COLUMNS,
    REQUIRED_HMM_COLUMNS,
    assert_no_forbidden_columns_consumed,
    assert_no_performance_columns,
    build_no_lookahead_audit_records,
    find_forbidden_columns,
    normalise_market,
    require_columns,
    require_source_columns,
    sanitize_input_frames,
    sanitize_strategy_input_frame,
    validate_availability_consistency,
    validate_exposure_bounds,
    validate_long_format_keys,
    validate_phase9_output_schema,
    validate_phase9_signal_panel,
)


def _minimal_phase9_output_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": "US",
                "strategy_name": "unconditional_full",
                "regime_model": "unconditional",
                "signal_observation_date": pd.Timestamp("2020-01-02"),
                "signal_available_after_close_date": pd.Timestamp("2020-01-02"),
                "target_trade_date": pd.Timestamp("2020-01-03"),
                "target_exposure": -1.0,
                "strategy_available": True,
                "blocked_reason": "none",
                "decision_reason": "unconditional_full",
                "state_name": pd.NA,
                "p_calm": pd.NA,
                "p_transition": pd.NA,
                "p_stress": pd.NA,
                "vrp_har_gk": pd.NA,
                "har_forecast_available": pd.NA,
                "source_signal_date_column": "date",
                "source_model": "unconditional",
            },
            {
                "market": "US",
                "strategy_name": "hmm_prob_linear",
                "regime_model": "gaussian_hmm",
                "signal_observation_date": pd.Timestamp("2020-01-02"),
                "signal_available_after_close_date": pd.Timestamp("2020-01-02"),
                "target_trade_date": pd.Timestamp("2020-01-03"),
                "target_exposure": math.nan,
                "strategy_available": False,
                "blocked_reason": "missing_hmm_probabilities",
                "decision_reason": "unavailable",
                "state_name": pd.NA,
                "p_calm": pd.NA,
                "p_transition": pd.NA,
                "p_stress": pd.NA,
                "vrp_har_gk": pd.NA,
                "har_forecast_available": pd.NA,
                "source_signal_date_column": "hmm_signal_trade_date",
                "source_model": "gaussian_hmm",
            },
            {
                "market": "US",
                "strategy_name": "hmm_prob_linear_carry",
                "regime_model": "gaussian_hmm",
                "signal_observation_date": pd.Timestamp("2020-01-02"),
                "signal_available_after_close_date": pd.Timestamp("2020-01-02"),
                "target_trade_date": pd.Timestamp("2020-01-03"),
                "target_exposure": 0.0,
                "strategy_available": True,
                "blocked_reason": "none",
                "decision_reason": "stress_probability_veto",
                "state_name": "stress",
                "p_calm": 0.20,
                "p_transition": 0.30,
                "p_stress": 0.50,
                "vrp_har_gk": 0.02,
                "har_forecast_available": True,
                "source_signal_date_column": "hmm_signal_trade_date",
                "source_model": "gaussian_hmm",
            },
        ],
        columns=list(PHASE9_OUTPUT_COLUMNS),
    )


def test_normalise_market_accepts_us_and_india() -> None:
    assert normalise_market("us") == "US"
    assert normalise_market(" india ") == "INDIA"


def test_normalise_market_rejects_unknown_market() -> None:
    with pytest.raises(ValueError):
        normalise_market("EUROPE")


def test_require_columns_passes_when_columns_exist() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_columns(df, ["a", "b"], frame_name="test_frame")


def test_require_columns_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(KeyError):
        require_columns(df, ["a", "b"], frame_name="test_frame")


def test_har_source_columns_required() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "vrp_har_gk": [0.01],
            "har_forecast_available": [True],
        }
    )

    require_source_columns(df, "har")


def test_hmm_next_session_columns_are_required_and_allowed() -> None:
    df = pd.DataFrame({column: [1] for column in REQUIRED_HMM_COLUMNS})

    require_source_columns(df, "gaussian_hmm")
    assert find_forbidden_columns(df.columns) == ()


def test_forbidden_columns_are_detected() -> None:
    columns = [
        "date",
        "vrp_har_gk",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "some_smoothed_probability",
        "crisis_window_label",
        "msvol_state",
    ]

    forbidden = find_forbidden_columns(columns)

    assert "rv_gk_22d_forward_ann_label" in forbidden
    assert "vrp_forward_expost_gk_label" in forbidden
    assert "some_smoothed_probability" in forbidden
    assert "crisis_window_label" in forbidden
    assert "msvol_state" in forbidden


def test_forbidden_columns_are_dropped_not_consumed() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "vrp_har_gk": [0.01],
            "har_forecast_available": [True],
            "rv_gk_22d_forward_ann_label": [0.25],
            "vrp_forward_expost_gk_label": [0.03],
            "some_smoothed_probability": [0.70],
            "crisis_window_label": [1],
            "msvol_state": [2],
        }
    )

    result = sanitize_strategy_input_frame(df, frame_name="har")

    assert "date" in result.frame.columns
    assert "vrp_har_gk" in result.frame.columns
    assert "har_forecast_available" in result.frame.columns

    assert "rv_gk_22d_forward_ann_label" not in result.frame.columns
    assert "vrp_forward_expost_gk_label" not in result.frame.columns
    assert "some_smoothed_probability" not in result.frame.columns
    assert "crisis_window_label" not in result.frame.columns
    assert "msvol_state" not in result.frame.columns

    assert set(result.forbidden_columns_present_but_excluded) == {
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "some_smoothed_probability",
        "crisis_window_label",
        "msvol_state",
    }
    assert result.forbidden_columns_used == ()


def test_sanitize_input_frames_tracks_excluded_columns_by_frame() -> None:
    frames = {
        "har": pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-02")],
                "vrp_har_gk": [0.01],
                "har_forecast_available": [True],
                "vrp_forward_expost_gk_label": [0.03],
            }
        ),
        "hmm": pd.DataFrame(
            {
                "hmm_signal_trade_date": [pd.Timestamp("2020-01-03")],
                "hmm_smoothed_prob_stress": [0.20],
            }
        ),
    }

    sanitized, excluded = sanitize_input_frames(frames)

    assert "vrp_forward_expost_gk_label" not in sanitized["har"].columns
    assert "hmm_smoothed_prob_stress" not in sanitized["hmm"].columns
    assert excluded["har"] == ("vrp_forward_expost_gk_label",)
    assert excluded["hmm"] == ("hmm_smoothed_prob_stress",)


def test_assert_no_forbidden_columns_consumed_rejects_labels() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_columns_consumed(
            ["date", "vrp_forward_expost_gk_label"]
        )


def test_assert_no_forbidden_columns_consumed_allows_next_session_columns() -> None:
    assert_no_forbidden_columns_consumed(
        [
            "hmm_signal_trade_date",
            "hmm_state_name_for_next_session",
            "hmm_filtered_prob_calm_for_next_session",
            "hmm_filtered_prob_transition_for_next_session",
            "hmm_filtered_prob_stress_for_next_session",
        ]
    )


def test_msvol_columns_rejected_for_strategy_use() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_columns_consumed(["date", "msvol_state"])


def test_smoothed_columns_rejected_for_strategy_use() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_columns_consumed(["date", "hmm_smoothed_prob_stress"])


def test_crisis_columns_rejected_for_strategy_use() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_columns_consumed(["date", "crisis_window_flag"])


def test_label_columns_rejected_for_strategy_use() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_columns_consumed(["date", "future_vrp_label"])


def test_phase9_output_schema_contains_required_columns() -> None:
    df = _minimal_phase9_output_frame()
    validate_phase9_output_schema(df)


def test_phase9_output_schema_rejects_missing_column() -> None:
    df = _minimal_phase9_output_frame().drop(columns=["target_trade_date"])

    with pytest.raises(KeyError):
        validate_phase9_output_schema(df)


def test_phase9_output_schema_rejects_extra_column_by_default() -> None:
    df = _minimal_phase9_output_frame()
    df["debug_column"] = 1

    with pytest.raises(ValueError):
        validate_phase9_output_schema(df)


def test_phase9_output_schema_allows_extra_column_when_requested() -> None:
    df = _minimal_phase9_output_frame()
    df["debug_column"] = 1

    validate_phase9_output_schema(df, allow_extra_columns=True)


def test_output_schema_rejects_returns_field() -> None:
    df = _minimal_phase9_output_frame()
    df["strategy_return"] = 0.01

    with pytest.raises(ValueError):
        validate_phase9_output_schema(df, allow_extra_columns=True)


def test_output_schema_rejects_pnl_field() -> None:
    df = _minimal_phase9_output_frame()
    df["pnl"] = 10.0

    with pytest.raises(ValueError):
        validate_phase9_output_schema(df, allow_extra_columns=True)


def test_assert_no_performance_columns_rejects_sharpe() -> None:
    with pytest.raises(ValueError):
        assert_no_performance_columns(["strategy_name", "sharpe_ratio"])


def test_validate_long_format_keys_passes_for_unique_keys() -> None:
    df = _minimal_phase9_output_frame()
    validate_long_format_keys(df)


def test_validate_long_format_keys_rejects_duplicates() -> None:
    row = _minimal_phase9_output_frame().iloc[[0]].copy()
    duplicated = pd.concat([row, row], ignore_index=True)

    with pytest.raises(ValueError):
        validate_long_format_keys(duplicated)


def test_validate_exposure_bounds_passes_for_available_and_unavailable_rows() -> None:
    df = _minimal_phase9_output_frame()
    validate_exposure_bounds(df)


def test_validate_exposure_bounds_rejects_positive_available_exposure() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[0, "target_exposure"] = 0.25

    with pytest.raises(ValueError):
        validate_exposure_bounds(df)


def test_validate_exposure_bounds_rejects_too_negative_available_exposure() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[0, "target_exposure"] = -1.25

    with pytest.raises(ValueError):
        validate_exposure_bounds(df)


def test_validate_exposure_bounds_rejects_nan_available_exposure() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[0, "target_exposure"] = math.nan

    with pytest.raises(ValueError):
        validate_exposure_bounds(df)


def test_validate_exposure_bounds_rejects_non_nan_unavailable_exposure() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[1, "target_exposure"] = 0.0

    with pytest.raises(ValueError):
        validate_exposure_bounds(df)


def test_validate_availability_consistency_accepts_valid_flat_decision() -> None:
    df = _minimal_phase9_output_frame()
    valid_flat = df[df["strategy_name"] == "hmm_prob_linear_carry"]

    validate_availability_consistency(valid_flat)


def test_validate_availability_consistency_rejects_available_blocked_reason() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[0, "blocked_reason"] = "stress_veto"

    with pytest.raises(ValueError):
        validate_availability_consistency(df)


def test_validate_availability_consistency_rejects_available_unavailable_reason() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[0, "decision_reason"] = "unavailable"

    with pytest.raises(ValueError):
        validate_availability_consistency(df)


def test_validate_availability_consistency_rejects_unavailable_none_block() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[1, "blocked_reason"] = "none"

    with pytest.raises(ValueError):
        validate_availability_consistency(df)


def test_validate_availability_consistency_rejects_unavailable_decision_reason() -> None:
    df = _minimal_phase9_output_frame()
    df.loc[1, "decision_reason"] = "probability_linear"

    with pytest.raises(ValueError):
        validate_availability_consistency(df)


def test_validate_full_phase9_signal_panel() -> None:
    df = _minimal_phase9_output_frame()
    validate_phase9_signal_panel(df)


def test_build_no_lookahead_audit_records() -> None:
    records = build_no_lookahead_audit_records(
        present_but_excluded={
            "har": ["vrp_forward_expost_gk_label"],
            "hmm": ["hmm_smoothed_prob_stress"],
        },
        forbidden_columns_used=[],
    )

    assert records == [
        {
            "frame_name": "har",
            "column_name": "vrp_forward_expost_gk_label",
            "audit_status": "present_but_excluded",
            "used_by_strategy": False,
        },
        {
            "frame_name": "hmm",
            "column_name": "hmm_smoothed_prob_stress",
            "audit_status": "present_but_excluded",
            "used_by_strategy": False,
        },
    ]


def test_build_no_lookahead_audit_records_records_forbidden_use() -> None:
    records = build_no_lookahead_audit_records(
        present_but_excluded={},
        forbidden_columns_used=["vrp_forward_expost_gk_label"],
    )

    assert records == [
        {
            "frame_name": "strategy_consumed_columns",
            "column_name": "vrp_forward_expost_gk_label",
            "audit_status": "forbidden_column_used",
            "used_by_strategy": True,
        }
    ]