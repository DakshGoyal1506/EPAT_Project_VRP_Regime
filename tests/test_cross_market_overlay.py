from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.reports.cross_market import CrossMarketLeakageError
from vrp.strategies.cross_market_overlay import (
    CrossMarketOverlayError,
    apply_us_stress_overlay_to_exposure,
    build_all_india_cross_market_overlays,
    build_india_cross_market_overlay_panel,
    compute_overlay_returns,
    select_base_strategy_exposure,
    select_base_strategy_returns,
    summarize_overlay_vs_base,
    validate_overlay_analysis_only,
    validate_overlay_summary_schema,
)


def _overlay_config(tmp_path=None) -> dict:
    cfg = {
        "overlay": {
            "enabled": True,
            "analysis_only": True,
            "not_part_of_phase9_strategy_universe": True,
            "primary_india_strategy": "mar_prob_linear_carry",
            "secondary_india_strategy": "hmm_prob_linear_carry",
            "us_stress_cutoffs": [0.50, 0.60],
            "default_us_stress_cutoff": 0.60,
            "cost_bps": 5,
            "no_phase9_mutation": True,
            "no_phase10_mutation": True,
            "no_phase11_usage": True,
        },
        "required_overlay_summary_columns": [
            "model",
            "strategy",
            "cutoff",
            "n_obs",
            "base_mean_return",
            "overlay_mean_return",
            "base_vol",
            "overlay_vol",
            "base_sharpe",
            "overlay_sharpe",
            "base_sortino",
            "overlay_sortino",
            "base_max_drawdown",
            "overlay_max_drawdown",
            "base_turnover",
            "overlay_turnover",
            "base_exposure_mean",
            "overlay_exposure_mean",
            "blocked_days",
            "blocked_day_fraction",
            "analysis_only",
        ],
        "forbidden_inputs": {
            "phase11": [
                "reports/tables/phase_11/daily_paper_signal.csv",
                "reports/tables/phase_11/paper_order_intents.csv",
                "reports/tables/phase_11/risk_check_report.csv",
            ]
        },
        "forbidden_keywords": [
            "iBridgePy",
            "paper_order_intents",
            "daily_paper_signal",
            "risk_check_report",
            "paper_signal",
        ],
        "models": ["markov_autoreg", "gaussian_hmm"],
        "outputs": {
            "india_overlay_panel": "data/processed/india_cross_market_overlay_panel.parquet",
            "overlay_summary": "reports/tables/phase_13/india_overlay_summary.csv",
        },
    }

    if tmp_path is not None:
        cfg["input_files"] = {
            "INDIA": {
                "strategy_signals": "data/processed/india_strategy_signals.parquet",
                "backtest": "data/processed/india_backtest_panel.parquet",
            }
        }

    return cfg


def _predictive_panel(n: int = 20, model: str = "markov_autoreg") -> pd.DataFrame:
    india_dates = pd.date_range("2024-01-02", periods=n, freq="B")
    us_lagged_dates = india_dates - pd.Timedelta(days=1)

    return pd.DataFrame(
        {
            "model": [model] * n,
            "panel_type": ["predictive_lagged"] * n,
            "india_date": india_dates,
            "us_lagged_date": us_lagged_dates,
            "lag_calendar_days": [1] * n,
            "lag_is_strictly_prior": [True] * n,
            "us_stress_prob_lag1": [0.2, 0.7] * (n // 2),
        }
    )


def _strategy_signals(n: int = 20, strategy: str = "mar_prob_linear_carry") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "strategy": [strategy] * n,
            "target_exposure": [1.0] * n,
        }
    )


def _backtest_panel(n: int = 20, strategy: str = "mar_prob_linear_carry") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "strategy": [strategy] * n,
            "net_return": np.linspace(-0.01, 0.01, n),
        }
    )


def test_select_base_strategy_exposure_uses_requested_strategy() -> None:
    df = pd.concat(
        [
            _strategy_signals(strategy="mar_prob_linear_carry"),
            _strategy_signals(strategy="other_strategy"),
        ],
        ignore_index=True,
    )

    out = select_base_strategy_exposure(df, "mar_prob_linear_carry")

    assert not out.empty
    assert out["strategy"].eq("mar_prob_linear_carry").all()
    assert out["base_exposure"].eq(1.0).all()


def test_select_base_strategy_exposure_rejects_missing_strategy() -> None:
    df = _strategy_signals(strategy="some_other_strategy")

    with pytest.raises(CrossMarketOverlayError):
        select_base_strategy_exposure(df, "mar_prob_linear_carry")


def test_select_base_strategy_returns_uses_requested_strategy() -> None:
    df = pd.concat(
        [
            _backtest_panel(strategy="mar_prob_linear_carry"),
            _backtest_panel(strategy="other_strategy"),
        ],
        ignore_index=True,
    )

    out = select_base_strategy_returns(df, "mar_prob_linear_carry")

    assert not out.empty
    assert out["strategy"].eq("mar_prob_linear_carry").all()
    assert "base_return" in out.columns


def test_overlay_sets_exposure_to_zero_when_lagged_us_stress_exceeds_cutoff() -> None:
    exposure = select_base_strategy_exposure(
        _strategy_signals(),
        "mar_prob_linear_carry",
    )

    out = apply_us_stress_overlay_to_exposure(
        exposure,
        _predictive_panel(),
        model="markov_autoreg",
        cutoff=0.60,
    )

    assert not out.empty
    assert out.loc[out["us_stress_prob_lagged"] > 0.60, "overlay_exposure"].eq(0.0).all()
    assert out.loc[out["us_stress_prob_lagged"] <= 0.60, "overlay_exposure"].eq(1.0).all()
    assert out["analysis_only"].fillna(False).astype(bool).all()
    assert not out["phase9_mutation"].fillna(True).astype(bool).any()
    assert not out["phase10_mutation"].fillna(True).astype(bool).any()
    assert not out["phase11_usage"].fillna(True).astype(bool).any()


def test_overlay_rejects_same_date_us_predictive_leakage() -> None:
    exposure = select_base_strategy_exposure(
        _strategy_signals(n=2),
        "mar_prob_linear_carry",
    )

    predictive = pd.DataFrame(
        {
            "model": ["markov_autoreg", "markov_autoreg"],
            "panel_type": ["predictive_lagged", "predictive_lagged"],
            "india_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "lag_calendar_days": [0, 1],
            "lag_is_strictly_prior": [False, True],
            "us_stress_prob_lag1": [0.8, 0.2],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        apply_us_stress_overlay_to_exposure(
            exposure,
            predictive,
            model="markov_autoreg",
            cutoff=0.60,
        )


def test_compute_overlay_returns_scales_return_and_charges_turnover_cost() -> None:
    exposure = select_base_strategy_exposure(
        _strategy_signals(),
        "mar_prob_linear_carry",
    )

    overlay = apply_us_stress_overlay_to_exposure(
        exposure,
        _predictive_panel(),
        model="markov_autoreg",
        cutoff=0.60,
    )

    returns = select_base_strategy_returns(
        _backtest_panel(),
        "mar_prob_linear_carry",
    )

    out = compute_overlay_returns(
        overlay,
        returns,
        cost_bps=5,
    )

    assert not out.empty
    assert {"overlay_return", "overlay_cost", "base_equity", "overlay_equity"}.issubset(
        out.columns
    )
    assert out["overlay_cost"].ge(0.0).all()

    blocked = out["blocked_by_us_stress"]
    assert out.loc[blocked, "overlay_return_gross"].abs().le(1e-12).all()


def test_build_india_cross_market_overlay_panel_returns_panel_and_summary() -> None:
    panel, summary = build_india_cross_market_overlay_panel(
        predictive_panel=_predictive_panel(),
        strategy_signals=_strategy_signals(),
        backtest_panel=_backtest_panel(),
        model="markov_autoreg",
        strategy_name="mar_prob_linear_carry",
        cutoff=0.60,
        cost_bps=5,
        config=_overlay_config(),
    )

    assert not panel.empty
    assert not summary.empty
    assert summary["model"].iloc[0] == "markov_autoreg"
    assert summary["strategy"].iloc[0] == "mar_prob_linear_carry"
    assert float(summary["cutoff"].iloc[0]) == 0.60
    assert int(summary["blocked_days"].iloc[0]) == 10
    assert bool(summary["analysis_only"].iloc[0]) is True

    validate_overlay_summary_schema(summary, _overlay_config())


def test_validate_overlay_analysis_only_rejects_mutation_flag() -> None:
    panel, _ = build_india_cross_market_overlay_panel(
        predictive_panel=_predictive_panel(),
        strategy_signals=_strategy_signals(),
        backtest_panel=_backtest_panel(),
        model="markov_autoreg",
        strategy_name="mar_prob_linear_carry",
        cutoff=0.60,
        cost_bps=5,
        config=_overlay_config(),
    )

    bad = panel.copy()
    bad["phase9_mutation"] = True

    with pytest.raises(CrossMarketOverlayError):
        validate_overlay_analysis_only(bad, _overlay_config())


def test_validate_overlay_summary_schema_rejects_missing_required_column() -> None:
    _, summary = build_india_cross_market_overlay_panel(
        predictive_panel=_predictive_panel(),
        strategy_signals=_strategy_signals(),
        backtest_panel=_backtest_panel(),
        model="markov_autoreg",
        strategy_name="mar_prob_linear_carry",
        cutoff=0.60,
        cost_bps=5,
        config=_overlay_config(),
    )

    bad = summary.drop(columns=["analysis_only"])

    with pytest.raises(CrossMarketOverlayError):
        validate_overlay_summary_schema(bad, _overlay_config())


def test_build_all_india_cross_market_overlays_from_parquet_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    signals = pd.concat(
        [
            _strategy_signals(strategy="mar_prob_linear_carry"),
            _strategy_signals(strategy="hmm_prob_linear_carry"),
        ],
        ignore_index=True,
    )
    backtest = pd.concat(
        [
            _backtest_panel(strategy="mar_prob_linear_carry"),
            _backtest_panel(strategy="hmm_prob_linear_carry"),
        ],
        ignore_index=True,
    )

    signals.to_parquet(data_dir / "india_strategy_signals.parquet", index=False)
    backtest.to_parquet(data_dir / "india_backtest_panel.parquet", index=False)

    predictive = pd.concat(
        [
            _predictive_panel(model="markov_autoreg"),
            _predictive_panel(model="gaussian_hmm"),
        ],
        ignore_index=True,
    )

    cfg = _overlay_config(tmp_path)

    out = build_all_india_cross_market_overlays(
        predictive_panel=predictive,
        config=cfg,
        root=tmp_path,
        models=["markov_autoreg", "gaussian_hmm"],
    )

    overlay_panel = out["india_overlay_panel"]
    summary = out["overlay_summary"]

    assert not overlay_panel.empty
    assert not summary.empty
    assert set(summary["model"]) == {"markov_autoreg", "gaussian_hmm"}
    assert set(summary["cutoff"].astype(float)) == {0.50, 0.60}
    assert summary["analysis_only"].fillna(False).astype(bool).all()