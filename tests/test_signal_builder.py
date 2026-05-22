from __future__ import annotations

import math

import pandas as pd
import pytest

from vrp.strategies.signal_builder import (
    MISSING_TARGET_TRADE_DATE,
    build_phase9_signal_panel,
)
from vrp.strategies.signal_schema import validate_phase9_signal_panel
from vrp.strategies.strategy_registry import (
    APPROVED_STRATEGY_NAMES,
    REJECTED_STRATEGY_NAMES,
)


def _dates() -> list[pd.Timestamp]:
    return [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-06"),
    ]


def _har_frame() -> pd.DataFrame:
    d0, d1, d2, d3 = _dates()
    return pd.DataFrame(
        {
            "date": [d0, d1, d2, d3],
            "vrp_har_gk": [0.02, -0.01, 0.02, 0.03],
            "vrp_har_gk_positive": [True, False, True, True],
            "har_forecast_available": [True, True, True, True],
            "vrp_forward_expost_gk_label": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _threshold_frame() -> pd.DataFrame:
    d0, d1, d2, d3 = _dates()
    return pd.DataFrame(
        {
            "date": [d0, d1, d2, d3],
            "threshold_state": [0, 2, 1, 0],
            "threshold_state_name": ["calm", "stress", "transition", "calm"],
            "threshold_regime_available": [True, True, True, True],
            "threshold_trigger_reason": [
                "low_vol",
                "stress_trigger",
                "transition_trigger",
                "low_vol",
            ],
            "crisis_window_label": [0, 1, 0, 0],
        }
    )


def _hmm_frame() -> pd.DataFrame:
    d0, d1, d2, d3 = _dates()
    return pd.DataFrame(
        {
            "hmm_signal_observation_date": [d0, d1, d2, d3],
            "hmm_signal_available_after_close_date": [d0, d1, d2, d3],
            "hmm_signal_trade_date": [d1, d2, d3, pd.NaT],
            "hmm_state_for_next_session": [0, 0, 2, 0],
            "hmm_state_name_for_next_session": [
                "calm",
                "calm",
                "stress",
                "calm",
            ],
            "hmm_filtered_prob_calm_for_next_session": [0.70, 0.70, 0.50, 0.70],
            "hmm_filtered_prob_transition_for_next_session": [
                0.20,
                0.10,
                0.09,
                0.20,
            ],
            "hmm_filtered_prob_stress_for_next_session": [
                0.10,
                0.20,
                0.41,
                0.10,
            ],
            "hmm_smoothed_prob_stress": [0.99, 0.99, 0.99, 0.99],
        }
    )


def _mar_frame() -> pd.DataFrame:
    d0, d1, d2, d3 = _dates()
    return pd.DataFrame(
        {
            "mar_signal_observation_date": [d0, d1, d2, d3],
            "mar_signal_available_after_close_date": [d0, d1, d2, d3],
            "mar_signal_trade_date": [d1, d2, d3, pd.NaT],
            "mar_state_for_next_session": [0, 0, 2, 0],
            "mar_state_name_for_next_session": [
                "calm",
                "calm",
                "stress",
                "calm",
            ],
            "mar_filtered_prob_calm_for_next_session": [0.60, 0.70, 0.50, 0.70],
            "mar_filtered_prob_transition_for_next_session": [
                0.25,
                0.10,
                0.09,
                0.20,
            ],
            "mar_filtered_prob_stress_for_next_session": [
                0.15,
                0.20,
                0.41,
                0.10,
            ],
            "mar_diagnostic_smoothed_state": [1, 1, 1, 1],
        }
    )


def _build() -> pd.DataFrame:
    result = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=_threshold_frame(),
        hmm=_hmm_frame(),
        mar=_mar_frame(),
    )
    return result.signals


def _row(
    signals: pd.DataFrame,
    *,
    strategy_name: str,
    observation_date: str,
) -> pd.Series:
    mask = (
        (signals["strategy_name"] == strategy_name)
        & (signals["signal_observation_date"] == pd.Timestamp(observation_date))
    )
    selected = signals.loc[mask]
    assert len(selected) == 1
    return selected.iloc[0]


def test_builds_exactly_seven_strategies() -> None:
    signals = _build()
    assert set(signals["strategy_name"].unique()) == set(APPROVED_STRATEGY_NAMES)


def test_each_strategy_has_one_row_per_observation_date() -> None:
    signals = _build()
    counts = signals.groupby("strategy_name").size().to_dict()

    for strategy in APPROVED_STRATEGY_NAMES:
        assert counts[strategy] == 4


def test_no_rejected_strategy_names_appear() -> None:
    signals = _build()
    names = set(signals["strategy_name"].unique())

    assert names.isdisjoint(set(REJECTED_STRATEGY_NAMES))


def test_output_is_long_format_and_schema_valid() -> None:
    signals = _build()
    validate_phase9_signal_panel(signals)

    key_columns = [
        "market",
        "signal_observation_date",
        "target_trade_date",
        "strategy_name",
    ]
    assert not signals.duplicated(key_columns).any()


def test_threshold_is_shifted_to_next_trade_date() -> None:
    signals = _build()

    first = _row(
        signals,
        strategy_name="threshold_hard_filter",
        observation_date="2020-01-01",
    )
    second = _row(
        signals,
        strategy_name="threshold_hard_filter",
        observation_date="2020-01-02",
    )
    third = _row(
        signals,
        strategy_name="threshold_hard_filter",
        observation_date="2020-01-03",
    )

    assert first["target_trade_date"] == pd.Timestamp("2020-01-02")
    assert second["target_trade_date"] == pd.Timestamp("2020-01-03")
    assert third["target_trade_date"] == pd.Timestamp("2020-01-06")


def test_hmm_is_not_double_shifted() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="hmm_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["target_trade_date"] == pd.Timestamp("2020-01-02")


def test_mar_is_not_double_shifted() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="mar_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["target_trade_date"] == pd.Timestamp("2020-01-02")


def test_missing_trade_date_is_unavailable() -> None:
    signals = _build()

    for strategy_name in APPROVED_STRATEGY_NAMES:
        row = _row(
            signals,
            strategy_name=strategy_name,
            observation_date="2020-01-06",
        )
        assert row["strategy_available"] is False or row["strategy_available"] == False
        assert math.isnan(row["target_exposure"])
        assert row["blocked_reason"] == MISSING_TARGET_TRADE_DATE
        assert row["decision_reason"] == "unavailable"


def test_threshold_stress_veto_is_valid_flat_not_unavailable() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="threshold_hard_filter",
        observation_date="2020-01-02",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == 0.0
    assert row["blocked_reason"] == "none"
    assert row["decision_reason"] == "stress_veto"


def test_threshold_defensive_transition_is_partial_exposure() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="threshold_defensive",
        observation_date="2020-01-03",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == pytest.approx(-0.25)
    assert row["blocked_reason"] == "none"
    assert row["decision_reason"] == "transition_partial_exposure"


def test_hmm_probability_linear_exposure() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="hmm_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == pytest.approx(-0.60)
    assert row["decision_reason"] == "probability_linear"


def test_mar_probability_linear_exposure() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="mar_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == pytest.approx(-0.45)
    assert row["decision_reason"] == "probability_linear"


def test_negative_vrp_har_is_valid_flat_not_unavailable() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="hmm_prob_linear_carry",
        observation_date="2020-01-02",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == 0.0
    assert row["blocked_reason"] == "none"
    assert row["decision_reason"] == "negative_or_zero_vrp_har"


def test_stress_probability_veto_is_valid_flat_not_unavailable() -> None:
    signals = _build()

    row = _row(
        signals,
        strategy_name="hmm_prob_linear_carry",
        observation_date="2020-01-03",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == 0.0
    assert row["blocked_reason"] == "none"
    assert row["decision_reason"] == "stress_probability_veto"


def test_all_available_exposures_are_bounded() -> None:
    signals = _build()
    available = signals[signals["strategy_available"] == True]

    assert not available["target_exposure"].isna().any()
    assert (available["target_exposure"] >= -1.0).all()
    assert (available["target_exposure"] <= 0.0).all()


def test_forbidden_columns_are_excluded_and_recorded() -> None:
    result = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=_threshold_frame(),
        hmm=_hmm_frame(),
        mar=_mar_frame(),
    )

    signals = result.signals
    excluded = result.forbidden_columns_present_but_excluded

    assert "vrp_forward_expost_gk_label" in excluded["har"]
    assert "crisis_window_label" in excluded["threshold"]
    assert "hmm_smoothed_prob_stress" in excluded["gaussian_hmm"]
    assert "mar_diagnostic_smoothed_state" in excluded["markov_autoreg"]
    assert result.forbidden_columns_used == ()

    for forbidden_column in [
        "vrp_forward_expost_gk_label",
        "crisis_window_label",
        "hmm_smoothed_prob_stress",
        "mar_diagnostic_smoothed_state",
    ]:
        assert forbidden_column not in signals.columns


def test_missing_probabilities_are_unavailable() -> None:
    hmm = _hmm_frame()
    hmm.loc[0, "hmm_filtered_prob_calm_for_next_session"] = pd.NA

    signals = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=_threshold_frame(),
        hmm=hmm,
        mar=_mar_frame(),
    ).signals

    row = _row(
        signals,
        strategy_name="hmm_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["strategy_available"] is False or row["strategy_available"] == False
    assert math.isnan(row["target_exposure"])
    assert row["blocked_reason"] == "missing_probabilities"
    assert row["decision_reason"] == "unavailable"


def test_invalid_probability_sum_is_unavailable() -> None:
    hmm = _hmm_frame()
    hmm.loc[0, "hmm_filtered_prob_stress_for_next_session"] = 0.20

    signals = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=_threshold_frame(),
        hmm=hmm,
        mar=_mar_frame(),
    ).signals

    row = _row(
        signals,
        strategy_name="hmm_prob_linear",
        observation_date="2020-01-01",
    )

    assert row["strategy_available"] is False or row["strategy_available"] == False
    assert math.isnan(row["target_exposure"])
    assert row["blocked_reason"] == "invalid_probability_sum"
    assert row["decision_reason"] == "unavailable"


def test_requested_single_strategy_builds_only_that_strategy() -> None:
    result = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=_threshold_frame(),
        hmm=_hmm_frame(),
        mar=_mar_frame(),
        requested_strategy="hmm_prob_linear",
    )

    assert set(result.signals["strategy_name"].unique()) == {"hmm_prob_linear"}


def test_unapproved_requested_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        build_phase9_signal_panel(
            market="US",
            har=_har_frame(),
            threshold=_threshold_frame(),
            hmm=_hmm_frame(),
            mar=_mar_frame(),
            requested_strategy="probability_product",
        )


def test_no_returns_or_pnl_columns_are_created() -> None:
    signals = _build()

    forbidden_substrings = [
        "return",
        "pnl",
        "profit",
        "sharpe",
        "drawdown",
        "transaction_cost",
        "performance",
        "backtest",
    ]

    for column in signals.columns:
        lowered = column.lower()
        assert not any(token in lowered for token in forbidden_substrings)
    
def test_threshold_regime_available_accepts_numpy_bool_true() -> None:
    threshold = _threshold_frame()
    threshold["threshold_regime_available"] = threshold[
        "threshold_regime_available"
    ].astype("bool")

    signals = build_phase9_signal_panel(
        market="US",
        har=_har_frame(),
        threshold=threshold,
        hmm=_hmm_frame(),
        mar=_mar_frame(),
    ).signals

    row = _row(
        signals,
        strategy_name="threshold_hard_filter",
        observation_date="2020-01-01",
    )

    assert row["strategy_available"] is True or row["strategy_available"] == True
    assert row["target_exposure"] == -1.0
    assert row["blocked_reason"] == "none"