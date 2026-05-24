from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vrp.backtest.backtest_registry import (
    BacktestRegistryError,
    assert_no_outcome_labels_used_as_signals,
)
from vrp.backtest.payoff_proxies import (
    PRIMARY_PAYOFF_LABEL,
    build_forward_vrp_outcome_panel,
    compute_forward_vrp_strategy_payoff,
    join_strategy_with_outcome,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_toy_signals() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")


def _load_toy_outcomes() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")


def _build_joined_panel() -> pd.DataFrame:
    signals = _load_toy_signals()
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    return join_strategy_with_outcome(
        signals,
        outcomes,
        alignment="signal_observation_date",
    )


def test_target_trade_date_is_after_signal_observation_date_for_eligible_rows() -> None:
    joined = _build_joined_panel()
    panel = compute_forward_vrp_strategy_payoff(joined)

    eligible = panel["is_backtest_eligible"].astype(bool)

    assert (
        panel.loc[eligible, "target_trade_date"]
        > panel.loc[eligible, "signal_observation_date"]
    ).all()


def test_default_outcome_label_date_equals_signal_observation_date() -> None:
    joined = _build_joined_panel()

    assert (
        joined["outcome_label_date"]
        == pd.to_datetime(joined["signal_observation_date"])
    ).all()


def test_no_same_day_signal_to_trade_use_in_toy_fixture() -> None:
    signals = _load_toy_signals()

    signal_dates = pd.to_datetime(signals["signal_observation_date"])
    trade_dates = pd.to_datetime(signals["target_trade_date"])

    assert (trade_dates > signal_dates).all()


def test_join_does_not_shift_hmm_or_mar_rows_again() -> None:
    joined = _build_joined_panel()

    regime_rows = joined[
        joined["strategy_name"].str.startswith(("hmm_", "mar_"), na=False)
    ]

    assert not regime_rows.empty
    assert (
        regime_rows["outcome_label_date"]
        == regime_rows["signal_observation_date"]
    ).all()


def test_forward_expost_labels_forbidden_as_signal_features() -> None:
    with pytest.raises(BacktestRegistryError):
        assert_no_outcome_labels_used_as_signals(
            ["hmm_filtered_stress_prob", PRIMARY_PAYOFF_LABEL]
        )


def test_outcome_label_is_used_only_after_join() -> None:
    signals = _load_toy_signals()

    assert PRIMARY_PAYOFF_LABEL not in signals.columns

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)

    assert PRIMARY_PAYOFF_LABEL in joined.columns


def test_target_exposure_is_not_overwritten() -> None:
    joined = _build_joined_panel()

    before = joined["target_exposure"].copy()
    panel = compute_forward_vrp_strategy_payoff(joined)

    pd.testing.assert_series_equal(
        before.reset_index(drop=True),
        panel["target_exposure"].reset_index(drop=True),
        check_names=False,
    )

    assert "target_exposure_for_backtest" in panel.columns


def test_payoff_uses_outcome_from_signal_observation_date_not_target_trade_date() -> None:
    signals = pd.DataFrame(
        {
            "market": ["US"],
            "strategy_name": ["unconditional_full"],
            "signal_observation_date": ["2020-01-02"],
            "target_trade_date": ["2020-01-03"],
            "target_exposure": [-1.0],
            "strategy_available": [True],
        }
    )

    outcomes = pd.DataFrame(
        {
            "market": ["US", "US"],
            "date": ["2020-01-02", "2020-01-03"],
            PRIMARY_PAYOFF_LABEL: [0.04, -0.99],
        }
    )

    outcome_panel = build_forward_vrp_outcome_panel(outcomes)
    joined = join_strategy_with_outcome(
        signals,
        outcome_panel,
        alignment="signal_observation_date",
    )
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert row["outcome_label_date"] == pd.Timestamp("2020-01-02")
    assert float(row[PRIMARY_PAYOFF_LABEL]) == 0.04
    assert float(row["gross_return_proxy"]) == 0.04