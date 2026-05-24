from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vrp.backtest.costs import apply_costs_to_backtest_panel
from vrp.backtest.metrics import (
    ANNUALIZED_METRICS_INTERPRETATION,
    BacktestMetricsError,
    build_availability_summary,
    build_strategy_metric_table,
    compute_cte,
    compute_downside_deviation,
    compute_equity_curve,
    compute_max_drawdown,
    compute_strategy_metrics,
    get_metrics_metadata,
)
from vrp.backtest.payoff_proxies import (
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


def test_metrics_metadata_contains_overlap_caveat() -> None:
    metadata = get_metrics_metadata(horizon_trading_days=22)

    assert metadata["uses_overlapping_forward_labels"] is True
    assert metadata["horizon_trading_days"] == 22
    assert metadata["research_proxy_not_trade_pnl"] is True
    assert (
        metadata["annualized_metrics_interpretation"]
        == ANNUALIZED_METRICS_INTERPRETATION
    )


def test_compute_equity_curve_is_additive_and_uses_eligible_rows() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"]

    curve = compute_equity_curve(unconditional)

    assert curve["net_return_proxy"].round(10).tolist() == [
        0.0395,
        -0.0005,
        -0.01025,
    ]
    assert curve["equity_curve"].round(10).tolist() == [
        0.0395,
        0.039,
        0.02875,
    ]
    assert curve["drawdown"].round(10).tolist() == [
        0.0,
        -0.0005,
        -0.01075,
    ]


def test_compute_max_drawdown_includes_initial_zero_peak() -> None:
    returns = pd.Series([-0.10, 0.03, -0.02])

    max_drawdown = compute_max_drawdown(returns)

    assert round(max_drawdown, 10) == -0.10


def test_compute_cte_95_uses_loss_tail() -> None:
    returns = pd.Series([0.01, 0.02, -0.10, -0.05, 0.03])

    cte = compute_cte(returns, alpha=0.95)

    assert cte <= returns.quantile(0.05)


def test_compute_downside_deviation_positive_when_losses_exist() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])

    downside = compute_downside_deviation(returns, annualization_periods=252)

    assert downside > 0


def test_strategy_metrics_for_unconditional_toy_panel() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"]

    metrics = compute_strategy_metrics(unconditional)

    assert metrics["n_obs"] == 3
    assert metrics["n_eligible"] == 3
    assert metrics["availability_rate"] == 1.0
    assert metrics["n_return_obs"] == 3

    assert round(metrics["total_gross_return"], 10) == 0.03
    assert round(metrics["total_cost"], 10) == 0.00125
    assert round(metrics["total_return_proxy"], 10) == 0.02875

    assert round(metrics["mean_return"], 10) == round(0.02875 / 3.0, 10)
    assert metrics["annualized_return"] == metrics["mean_return"] * 252

    assert metrics["max_drawdown"] < 0
    assert metrics["hit_rate"] == pytest.approx(1.0 / 3.0)

    assert metrics["uses_overlapping_forward_labels"] is True
    assert metrics["horizon_trading_days"] == 22
    assert metrics["research_proxy_not_trade_pnl"] is True


def test_strategy_metrics_include_exposure_adjusted_fields() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"]

    metrics = compute_strategy_metrics(unconditional)

    assert "sum_abs_exposure" in metrics
    assert "return_per_abs_exposure" in metrics
    assert "drawdown_per_abs_exposure" in metrics
    assert "return_to_drawdown" in metrics

    assert metrics["sum_abs_exposure"] == pytest.approx(1.5)
    assert metrics["return_per_abs_exposure"] == pytest.approx(
        metrics["total_return_proxy"] / 1.5
    )
    assert metrics["return_to_drawdown"] == pytest.approx(
        metrics["total_return_proxy"] / abs(metrics["max_drawdown"])
    )


def test_strategy_metrics_ignore_ineligible_rows_for_returns() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"].copy()

    bad_extra = unconditional.iloc[[0]].copy()
    bad_extra["is_backtest_eligible"] = False
    bad_extra["exclusion_reason"] = "strategy_unavailable"
    bad_extra["net_return_proxy"] = 999.0
    bad_extra["gross_return_proxy"] = 999.0

    modified = pd.concat([unconditional, bad_extra], ignore_index=True)

    metrics = compute_strategy_metrics(modified)

    assert metrics["n_obs"] == 4
    assert metrics["n_eligible"] == 3
    assert metrics["availability_rate"] == 0.75
    assert round(metrics["total_return_proxy"], 10) == 0.02875


def test_strategy_metrics_raise_if_eligible_return_missing() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"].copy()

    unconditional.loc[unconditional.index[0], "net_return_proxy"] = np.nan

    with pytest.raises(BacktestMetricsError):
        compute_strategy_metrics(unconditional)


def test_strategy_metrics_all_ineligible_returns_nan_metrics() -> None:
    panel = _build_toy_panel()
    unconditional = panel[panel["strategy_name"] == "unconditional_full"].copy()

    unconditional["is_backtest_eligible"] = False
    unconditional["exclusion_reason"] = "strategy_unavailable"
    unconditional["net_return_proxy"] = np.nan

    metrics = compute_strategy_metrics(unconditional)

    assert metrics["n_obs"] == 3
    assert metrics["n_eligible"] == 0
    assert metrics["availability_rate"] == 0.0
    assert np.isnan(metrics["total_return_proxy"])
    assert metrics["uses_overlapping_forward_labels"] is True


def test_build_strategy_metric_table_one_row_per_market_strategy() -> None:
    panel = _build_toy_panel()

    metrics = build_strategy_metric_table(panel)

    assert set(metrics["strategy_name"]) == set(panel["strategy_name"].unique())
    assert set(metrics["market"]) == {"US"}
    assert len(metrics) == panel["strategy_name"].nunique()

    required_cols = {
        "market",
        "strategy_name",
        "n_obs",
        "n_eligible",
        "availability_rate",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "uses_overlapping_forward_labels",
        "annualized_metrics_interpretation",
        "research_proxy_not_trade_pnl",
    }

    assert required_cols.issubset(set(metrics.columns))


def test_build_availability_summary_counts_exclusion_reasons() -> None:
    panel = _build_toy_panel()

    modified = panel.copy()
    modified.loc[modified.index[0], "is_backtest_eligible"] = False
    modified.loc[modified.index[0], "exclusion_reason"] = "strategy_unavailable"

    summary = build_availability_summary(modified)

    unconditional = summary[summary["strategy_name"] == "unconditional_full"].iloc[0]

    assert unconditional["n_obs"] == 3
    assert unconditional["n_eligible"] == 2
    assert unconditional["availability_rate"] == pytest.approx(2.0 / 3.0)
    assert unconditional["strategy_unavailable"] == 1


def test_build_strategy_metric_table_empty_panel_returns_empty_df() -> None:
    panel = pd.DataFrame(
        columns=[
            "market",
            "strategy_name",
            "is_backtest_eligible",
            "net_return_proxy",
        ]
    )

    metrics = build_strategy_metric_table(panel)

    assert metrics.empty