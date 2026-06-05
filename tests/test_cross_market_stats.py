from __future__ import annotations

import numpy as np
import pandas as pd

from vrp.reports.cross_market import (
    build_lead_lag_table,
    compute_granger_diagnostics,
    compute_logistic_diagnostic_tables,
    compute_regime_probability_correlations,
    compute_state_label_agreement,
    compute_vrp_change_correlations,
    compute_vrp_level_correlations,
)


def _stats_config(min_observations: int = 20) -> dict:
    return {
        "correlations": {
            "methods": ["pearson", "spearman"],
            "min_observations": min_observations,
        },
        "lead_lag": {
            "enabled": True,
            "max_lag": 3,
            "lag_direction": "us_leads_india",
            "descriptive_only": True,
        },
        "linear_regression": {
            "enabled": True,
            "robust_standard_errors": "HAC",
            "hac_maxlags": 2,
            "min_observations": min_observations,
            "descriptive_only": True,
        },
        "granger_diagnostics": {
            "enabled": True,
            "max_lag": 2,
            "descriptive_only": True,
            "reject_causal_language": True,
            "continuous_series_only": True,
        },
        "predictive_features": {
            "local_india_lags": [
                "india_vrp_har_gk_lag1",
                "india_iv_ann_lag1",
                "india_rv_gk_22d_ann_lag1",
                "india_stress_prob_lag1",
            ],
            "lagged_us_features": [
                "us_stress_prob_lag1",
                "us_vrp_har_gk_lag1",
                "us_iv_ann_lag1",
                "us_rv_gk_22d_ann_lag1",
            ],
        },
        "logistic_regression": {
            "enabled": True,
            "dependent_variable": "india_stress_indicator",
            "local_only_model": True,
            "local_plus_us_model": True,
            "robust_standard_errors": "HC1",
            "min_observations": min_observations,
            "handle_perfect_separation": "skip_with_reason",
        },
        "predictive_validation": {
            "enabled": True,
            "split_method": "chronological",
            "train_fraction": 0.70,
            "min_train_observations": min_observations,
            "min_test_observations": 10,
            "no_cutoff_tuning": True,
        },
    }


def _descriptive_panel(n: int = 80) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "model": ["gaussian_hmm"] * n,
            "panel_type": ["descriptive_same_date"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "us_vrp_har_gk": x,
            "india_vrp_har_gk": 0.5 * x + 3.0,
            "us_stress_prob": np.clip(0.2 + x / (2.0 * n), 0.0, 1.0),
            "india_stress_prob": np.clip(0.1 + x / (2.5 * n), 0.0, 1.0),
            "us_state_name": ["calm"] * (n // 2) + ["stress"] * (n - n // 2),
            "india_state_name": ["calm"] * (n // 2) + ["stress"] * (n - n // 2),
        }
    )


def _predictive_panel(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    india_lag = rng.normal(size=n)
    us_lag = rng.normal(size=n)
    z = -0.2 + 0.8 * india_lag + 0.7 * us_lag
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(float)

    india_dates = pd.date_range("2020-01-02", periods=n, freq="B")
    us_lagged_dates = india_dates - pd.Timedelta(days=1)

    return pd.DataFrame(
        {
            "model": ["gaussian_hmm"] * n,
            "panel_type": ["predictive_lagged"] * n,
            "india_date": india_dates,
            "us_lagged_date": us_lagged_dates,
            "lag_calendar_days": [1] * n,
            "lag_is_strictly_prior": [True] * n,
            "india_vrp_har_gk": india_lag + rng.normal(scale=0.1, size=n),
            "us_vrp_har_gk": us_lag + rng.normal(scale=0.1, size=n),
            "india_stress_prob": np.clip(p + rng.normal(scale=0.05, size=n), 0.01, 0.99),
            "us_stress_prob": np.clip(0.5 + 0.2 * us_lag, 0.01, 0.99),
            "us_stress_prob_lag1": np.clip(0.5 + 0.2 * us_lag, 0.01, 0.99),
            "us_vrp_har_gk_lag1": us_lag,
            "us_iv_ann_lag1": rng.normal(size=n),
            "us_rv_gk_22d_ann_lag1": rng.normal(size=n),
            "india_stress_indicator": y,
            "india_vrp_har_gk_lag1": india_lag,
            "india_iv_ann_lag1": rng.normal(size=n),
            "india_rv_gk_22d_ann_lag1": rng.normal(size=n),
            "india_stress_prob_lag1": np.clip(0.5 + 0.2 * india_lag, 0.01, 0.99),
        }
    )


def test_vrp_level_correlations_have_expected_columns() -> None:
    panel = _descriptive_panel()
    out = compute_vrp_level_correlations(panel, config=_stats_config())

    required = {
        "model",
        "panel_type",
        "pair",
        "x",
        "y",
        "method",
        "n_obs",
        "correlation",
        "p_value",
        "status",
        "reason",
    }

    assert required.issubset(out.columns)
    assert set(out["method"]) == {"pearson", "spearman"}
    assert out["status"].eq("ok").all()
    assert out["correlation"].notna().all()


def test_vrp_change_correlations_skip_constant_diff_without_crashing() -> None:
    panel = _descriptive_panel()
    out = compute_vrp_change_correlations(panel, config=_stats_config())

    assert {"status", "reason", "pair"}.issubset(out.columns)
    assert out["pair"].eq("us_india_vrp_change").all()

    # The synthetic data is linear, so first differences are constant.
    assert out["status"].isin(["ok", "skipped"]).all()


def test_regime_probability_correlations_use_lagged_us_column_when_available() -> None:
    panel = _predictive_panel()
    out = compute_regime_probability_correlations(panel, config=_stats_config())

    assert not out.empty
    assert out["x"].eq("us_stress_prob_lag1").all()
    assert out["y"].eq("india_stress_prob").all()
    assert out["status"].isin(["ok", "skipped"]).all()


def test_state_label_agreement_outputs_summary_and_pairs() -> None:
    panel = _descriptive_panel()
    out = compute_state_label_agreement(panel)

    assert not out.empty
    assert {"summary", "state_pair"}.issubset(set(out["table_type"]))
    summary = out[out["table_type"] == "summary"].iloc[0]
    assert float(summary["exact_label_agreement_rate"]) == 1.0


def test_lead_lag_table_respects_us_leads_india_direction() -> None:
    panel = _predictive_panel()
    out = build_lead_lag_table(panel, max_lag=3, config=_stats_config())

    required = {
        "pair",
        "us_lag_trading_rows",
        "x",
        "x_lagged_for_test",
        "y",
        "method",
        "descriptive_only",
        "causal_interpretation_allowed",
    }

    assert required.issubset(out.columns)
    assert set(out["us_lag_trading_rows"].dropna().astype(int)) == {1, 2, 3}
    assert out["descriptive_only"].fillna(False).astype(bool).all()
    assert not out["causal_interpretation_allowed"].fillna(True).astype(bool).any()

    stress_rows = out[out["pair"] == "us_stress_prob_leads_india_stress_prob"]
    assert not stress_rows.empty
    assert stress_rows["x"].eq("us_stress_prob_lag1").all()
    assert stress_rows["y"].eq("india_stress_prob").all()


def test_granger_diagnostics_are_descriptive_only_and_noncausal() -> None:
    panel = _predictive_panel(n=150)
    out = compute_granger_diagnostics(panel, max_lag=2, config=_stats_config())

    assert not out.empty
    assert {"descriptive_only", "causal_interpretation_allowed", "status"}.issubset(
        out.columns
    )
    assert out["descriptive_only"].fillna(False).astype(bool).all()
    assert not out["causal_interpretation_allowed"].fillna(True).astype(bool).any()
    assert out["status"].isin(["ok", "skipped", "error"]).all()


def test_granger_diagnostics_reject_non_numeric_label_like_series() -> None:
    n = 80
    panel = pd.DataFrame(
        {
            "model": ["gaussian_hmm"] * n,
            "panel_type": ["predictive_lagged"] * n,
            "india_date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "india_stress_prob": ["calm", "stress"] * (n // 2),
            "us_stress_prob_lag1": ["calm", "stress"] * (n // 2),
            "india_vrp_har_gk": np.random.default_rng(1).normal(size=n),
            "us_vrp_har_gk_lag1": np.random.default_rng(2).normal(size=n),
        }
    )

    out = compute_granger_diagnostics(panel, max_lag=2, config=_stats_config())

    target = out[out["pair"] == "us_stress_prob_lagged_to_india_stress_prob"]
    assert not target.empty
    assert target["status"].eq("skipped").all()
    assert target["reason"].str.contains("non_numeric_or_label_like_series").all()


def test_logistic_diagnostics_handle_too_few_observations() -> None:
    panel = _predictive_panel(n=30)
    cfg = _stats_config(min_observations=250)

    tables = compute_logistic_diagnostic_tables(panel, cfg)

    summary = tables["logistic_model_summary"]
    comparison = tables["logistic_model_comparison"]
    oos = tables["logistic_oos_diagnostics"]

    assert not summary.empty
    assert summary["status"].eq("skipped").all()
    assert summary["reason"].str.contains("insufficient_observations").all()

    assert not comparison.empty
    assert "delta_pseudo_r2" in comparison.columns
    assert "likelihood_ratio_p_value" in comparison.columns

    assert not oos.empty
    assert oos["status"].eq("skipped").all()


def test_logistic_diagnostics_include_required_comparison_metrics() -> None:
    panel = _predictive_panel(n=300)
    cfg = _stats_config(min_observations=50)

    tables = compute_logistic_diagnostic_tables(panel, cfg)

    comparison = tables["logistic_model_comparison"]
    required = {
        "local_pseudo_r2",
        "plus_us_pseudo_r2",
        "delta_pseudo_r2",
        "local_aic",
        "plus_us_aic",
        "delta_aic_plus_minus_local",
        "local_bic",
        "plus_us_bic",
        "delta_bic_plus_minus_local",
        "local_log_likelihood",
        "plus_us_log_likelihood",
        "delta_log_likelihood",
        "local_auc",
        "plus_us_auc",
        "delta_auc",
        "local_brier_score",
        "plus_us_brier_score",
        "delta_brier_score_plus_minus_local",
        "likelihood_ratio_p_value",
    }

    assert required.issubset(comparison.columns)


def test_logistic_perfect_separation_is_reported_not_crashed() -> None:
    n = 120
    x = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    y = x.copy()

    panel = pd.DataFrame(
        {
            "model": ["gaussian_hmm"] * n,
            "panel_type": ["predictive_lagged"] * n,
            "india_date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "india_stress_indicator": y,
            "india_vrp_har_gk_lag1": x,
            "india_iv_ann_lag1": np.random.default_rng(3).normal(size=n),
            "india_rv_gk_22d_ann_lag1": np.random.default_rng(4).normal(size=n),
            "india_stress_prob_lag1": x,
            "us_stress_prob_lag1": x,
            "us_vrp_har_gk_lag1": x,
            "us_iv_ann_lag1": np.random.default_rng(5).normal(size=n),
            "us_rv_gk_22d_ann_lag1": np.random.default_rng(6).normal(size=n),
        }
    )

    cfg = _stats_config(min_observations=50)
    tables = compute_logistic_diagnostic_tables(panel, cfg)

    summary = tables["logistic_model_summary"]

    assert not summary.empty
    assert summary["status"].isin(["ok", "skipped", "error"]).all()

    # Most statsmodels versions report perfect separation as skipped/error.
    # If a version fits anyway, the test still ensures the diagnostic table exists.
    if not summary["status"].eq("ok").all():
        assert summary["reason"].astype(str).str.len().gt(0).any()