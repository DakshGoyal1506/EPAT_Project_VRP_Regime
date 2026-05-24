from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vrp.backtest.costs import (
    BacktestCostError,
    apply_costs_to_backtest_panel,
    compute_exposure_change_costs,
)
from vrp.backtest.payoff_proxies import (
    EXCLUSION_REASONS,
    PRIMARY_PAYOFF_LABEL,
    PayoffProxyError,
    build_forward_vrp_outcome_panel,
    compute_forward_vrp_strategy_payoff,
    join_strategy_with_outcome,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_toy_signals() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")


def _load_toy_outcomes() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")


def _build_toy_panel() -> pd.DataFrame:
    signals = _load_toy_signals()
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    payoff = compute_forward_vrp_strategy_payoff(joined)
    return apply_costs_to_backtest_panel(payoff, cost_bps=5.0)


def test_build_forward_vrp_outcome_panel() -> None:
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())

    assert list(outcomes.columns) == [
        "market",
        "outcome_label_date",
        PRIMARY_PAYOFF_LABEL,
    ]
    assert len(outcomes) == 3
    assert outcomes[PRIMARY_PAYOFF_LABEL].dtype.kind in {"f", "i"}
    assert not outcomes.duplicated(["market", "outcome_label_date"]).any()


def test_join_uses_signal_observation_date_alignment() -> None:
    signals = _load_toy_signals()
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())

    joined = join_strategy_with_outcome(
        signals,
        outcomes,
        alignment="signal_observation_date",
    )

    assert "outcome_label_date" in joined.columns
    assert "_outcome_merge" in joined.columns
    assert joined["_outcome_merge"].eq("both").all()

    assert (
        joined["outcome_label_date"]
        == pd.to_datetime(joined["signal_observation_date"])
    ).all()


def test_primary_payoff_sign_convention_for_toy_cases() -> None:
    panel = _build_toy_panel()

    unconditional = panel[panel["strategy_name"] == "unconditional_full"].sort_values(
        "signal_observation_date"
    )

    gross = unconditional["gross_return_proxy"].round(10).tolist()

    assert gross == [0.04, 0.0, -0.01]


def test_valid_flat_row_is_eligible_with_zero_payoff() -> None:
    panel = _build_toy_panel()

    row = panel[
        (panel["strategy_name"] == "unconditional_full")
        & (panel["target_exposure"].astype(float) == 0.0)
    ].iloc[0]

    assert bool(row["is_backtest_eligible"]) is True
    assert row["exclusion_reason"] == "available"
    assert float(row["target_exposure_for_backtest"]) == 0.0
    assert float(row["gross_return_proxy"]) == 0.0


def test_cost_accounting_uses_first_exposure_from_zero() -> None:
    panel = _build_toy_panel()

    unconditional = panel[panel["strategy_name"] == "unconditional_full"].sort_values(
        "target_trade_date"
    )

    deltas = unconditional["delta_exposure"].round(10).tolist()
    costs = unconditional["cost_proxy"].round(10).tolist()

    assert deltas == [-1.0, 1.0, -0.5]
    assert costs == [0.0005, 0.0005, 0.00025]


def test_net_return_proxy_subtracts_costs() -> None:
    panel = _build_toy_panel()

    unconditional = panel[panel["strategy_name"] == "unconditional_full"].sort_values(
        "target_trade_date"
    )

    net = unconditional["net_return_proxy"].round(10).tolist()

    assert net == [0.0395, -0.0005, -0.01025]


def test_costs_disabled_sets_zero_cost_on_eligible_rows() -> None:
    signals = _load_toy_signals()
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    payoff = compute_forward_vrp_strategy_payoff(joined)

    panel = apply_costs_to_backtest_panel(payoff, enabled=False)

    eligible = panel["is_backtest_eligible"].astype(bool)

    assert panel.loc[eligible, "cost_proxy"].eq(0.0).all()
    assert np.allclose(
        panel.loc[eligible, "net_return_proxy"],
        panel.loc[eligible, "gross_return_proxy"],
    )


def test_negative_cost_bps_fails() -> None:
    panel = _build_toy_panel()

    with pytest.raises(BacktestCostError):
        compute_exposure_change_costs(panel, cost_bps=-1)


def test_exclusion_reason_strategy_unavailable() -> None:
    signals = _load_toy_signals()
    signals.loc[0, "strategy_available"] = False

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "strategy_unavailable"
    assert pd.isna(row["target_exposure_for_backtest"])
    assert pd.isna(row["gross_return_proxy"])


def test_exclusion_reason_missing_target_trade_date() -> None:
    signals = _load_toy_signals()
    signals.loc[0, "target_trade_date"] = None

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "missing_target_trade_date"


def test_exclusion_reason_non_finite_exposure() -> None:
    signals = _load_toy_signals()
    signals["target_exposure"] = signals["target_exposure"].astype(object)
    signals.loc[0, "target_exposure"] = "not_a_number"

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "non_finite_exposure"


def test_exclusion_reason_missing_outcome_join() -> None:
    signals = _load_toy_signals()
    signals.loc[0, "signal_observation_date"] = "1999-01-01"

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "missing_outcome_join"


def test_exclusion_reason_missing_payoff_label() -> None:
    signals = _load_toy_signals()
    outcomes = _load_toy_outcomes()
    outcomes.loc[0, PRIMARY_PAYOFF_LABEL] = np.nan

    outcome_panel = build_forward_vrp_outcome_panel(outcomes)
    joined = join_strategy_with_outcome(signals, outcome_panel)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "missing_payoff_label"


def test_exclusion_reason_invalid_strategy_name() -> None:
    signals = _load_toy_signals()
    signals.loc[0, "strategy_name"] = "unapproved_strategy"

    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())
    joined = join_strategy_with_outcome(signals, outcomes)
    panel = compute_forward_vrp_strategy_payoff(joined)

    row = panel.iloc[0]

    assert bool(row["is_backtest_eligible"]) is False
    assert row["exclusion_reason"] == "invalid_strategy_name"


def test_exclusion_reasons_are_from_allowed_set() -> None:
    panel = _build_toy_panel()

    assert set(panel["exclusion_reason"].unique()).issubset(set(EXCLUSION_REASONS))


def test_duplicate_outcome_rows_fail() -> None:
    outcomes = _load_toy_outcomes()
    outcomes = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)

    with pytest.raises(PayoffProxyError):
        build_forward_vrp_outcome_panel(outcomes)