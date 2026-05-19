from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.regimes.hmm_features import build_hmm_feature_panel  # noqa: E402
from vrp.regimes.hmm_registry import (  # noqa: E402
    get_hmm_filtered_economic_probability_columns,
)
from vrp.regimes.hmm_scaling import (  # noqa: E402
    assert_scaled_prefix_invariance,
    assert_scaler_fit_uses_train_only,
    scale_hmm_feature_panel,
    transform_hmm_feature_panel_with_fitted_scaler,
)
from vrp.regimes.hmm_validation import (  # noqa: E402
    assert_hmm_feature_columns_are_legal,
    assert_output_probability_policy_is_safe,
    validate_crisis_windows_usage,
    validate_threshold_comparison_usage,
)
from vrp.regimes.gaussian_hmm import (  # noqa: E402
    HMMCandidateSpec,
    HMMFitConfig,
    fit_and_build_hmm_candidate_output,
)
from vrp.regimes.online_filter import (  # noqa: E402
    assert_prefix_filter_invariance,
    forward_filter_gaussian,
)
from vrp.reports.hmm_diagnostics import (  # noqa: E402
    build_hmm_no_lookahead_audit_table,
    build_hmm_probability_audit_table,
)


def _make_panel(
    *,
    n: int = 1200,
    seed: int = 7,
    extreme_tail: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regime = np.tile(np.r_[np.zeros(60), np.ones(60)], n // 120).astype(int)
    if len(regime) < n:
        regime = np.r_[regime, regime[: n - len(regime)]]
    regime = regime[:n]

    vrp = np.where(
        regime == 0,
        rng.normal(0.04, 0.01, n),
        rng.normal(-0.02, 0.02, n),
    )
    rv = np.where(
        regime == 0,
        rng.normal(0.10, 0.02, n),
        rng.normal(0.35, 0.05, n),
    )
    iv = np.where(
        regime == 0,
        rng.normal(0.15, 0.02, n),
        rng.normal(0.45, 0.05, n),
    )
    ret = np.where(
        regime == 0,
        rng.normal(0.001, 0.005, n),
        rng.normal(-0.002, 0.02, n),
    )

    if extreme_tail:
        tail = min(30, n)
        vrp[-tail:] = 100.0
        rv[-tail:] = 200.0
        iv[-tail:] = 300.0
        ret[-tail:] = -50.0

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "vrp_har_gk": vrp,
            "rv_gk_22d_ann_lag1": rv,
            "iv_ann": iv,
            "log_return": ret,
            "har_forecast_available": True,
        }
    )


def _fit_output(df: pd.DataFrame):
    feature_panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    scaled = scale_hmm_feature_panel(
        feature_panel,
        train_fraction=0.70,
        min_train_observations=750,
        min_test_observations=250,
    )

    return fit_and_build_hmm_candidate_output(
        scaled,
        spec=HMMCandidateSpec("F3", 2, "diag"),
        fit_config=HMMFitConfig(
            n_init=2,
            n_iter=150,
            random_seed=42,
        ),
    )


def test_hmm_feature_guard_rejects_forward_expost_label_threshold_and_crisis_names() -> None:
    bad_cols = [
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "future_rv",
        "threshold_state",
        "threshold_trigger_reason",
        "crisis_dummy",
        "hmm_filtered_prob_stress",
    ]

    for bad_col in bad_cols:
        with pytest.raises(ValueError):
            assert_hmm_feature_columns_are_legal(["iv_ann", bad_col])


def test_hmm_crisis_windows_are_diagnostic_only() -> None:
    validate_crisis_windows_usage(
        used_for="crisis_stress_overlap",
        diagnostics_only=True,
    )

    validate_crisis_windows_usage(
        used_for="crisis_lead_lag",
        diagnostics_only=True,
    )

    with pytest.raises(ValueError):
        validate_crisis_windows_usage(
            used_for="model_selection",
            diagnostics_only=True,
        )

    with pytest.raises(ValueError):
        validate_crisis_windows_usage(
            used_for="state_mapping",
            diagnostics_only=True,
        )


def test_hmm_threshold_state_is_diagnostic_only() -> None:
    validate_threshold_comparison_usage(
        threshold_state_as_feature=False,
        threshold_state_as_target=False,
        choose_model_by_threshold_match=False,
    )

    with pytest.raises(ValueError):
        validate_threshold_comparison_usage(threshold_state_as_feature=True)

    with pytest.raises(ValueError):
        validate_threshold_comparison_usage(threshold_state_as_target=True)

    with pytest.raises(ValueError):
        validate_threshold_comparison_usage(choose_model_by_threshold_match=True)


def test_hmm_probability_policy_rejects_smoothed_backtest_usage() -> None:
    assert_output_probability_policy_is_safe(
        uses_custom_forward_filter=True,
        uses_hmmlearn_predict_proba_for_backtest=False,
        uses_smoothed_probabilities_for_backtest=False,
    )

    with pytest.raises(ValueError):
        assert_output_probability_policy_is_safe(
            uses_custom_forward_filter=False,
            uses_hmmlearn_predict_proba_for_backtest=False,
            uses_smoothed_probabilities_for_backtest=False,
        )

    with pytest.raises(ValueError):
        assert_output_probability_policy_is_safe(
            uses_custom_forward_filter=True,
            uses_hmmlearn_predict_proba_for_backtest=True,
            uses_smoothed_probabilities_for_backtest=False,
        )

    with pytest.raises(ValueError):
        assert_output_probability_policy_is_safe(
            uses_custom_forward_filter=True,
            uses_hmmlearn_predict_proba_for_backtest=False,
            uses_smoothed_probabilities_for_backtest=True,
        )


def test_hmm_scaler_fit_uses_train_only_not_future_tail() -> None:
    df = _make_panel(extreme_tail=True)

    feature_panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    scaled = scale_hmm_feature_panel(
        feature_panel,
        train_fraction=0.70,
        min_train_observations=750,
        min_test_observations=250,
    )

    assert_scaler_fit_uses_train_only(scaled)

    train_mean = scaled.X_raw[scaled.train_indices].mean(axis=0)
    full_mean = scaled.X_raw.mean(axis=0)

    np.testing.assert_allclose(np.asarray(scaled.scaler.mean_), train_mean)
    assert not np.allclose(np.asarray(scaled.scaler.mean_), full_mean)


def test_hmm_scaled_prefix_invariance_when_future_rows_are_appended() -> None:
    prefix_df = _make_panel(n=1000, seed=11)
    future_df = _make_panel(n=250, seed=99, extreme_tail=True)
    future_df["date"] = pd.date_range(prefix_df["date"].max() + pd.Timedelta(days=1), periods=250)

    full_df = pd.concat([prefix_df, future_df], ignore_index=True)

    prefix_panel = build_hmm_feature_panel(
        prefix_df,
        market="US",
        feature_set="F3",
        min_eligible_observations=800,
        min_eligible_fraction=0.50,
    )
    full_panel = build_hmm_feature_panel(
        full_df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    prefix_scaled = scale_hmm_feature_panel(
        prefix_panel,
        train_fraction=0.70,
        min_train_observations=600,
        min_test_observations=200,
    )

    full_transformed = transform_hmm_feature_panel_with_fitted_scaler(
        full_panel,
        scaler=prefix_scaled.scaler,
        source_scaler_metadata=prefix_scaled.metadata.to_dict(),
    )

    assert_scaled_prefix_invariance(
        prefix_scaled.scaled_panel,
        full_transformed.scaled_panel,
        scaled_feature_cols=prefix_scaled.scaled_feature_cols,
        n_prefix_rows=len(prefix_scaled.scaled_panel),
        atol=1.0e-12,
    )


def test_hmm_custom_forward_filter_prefix_invariance_after_fit() -> None:
    output = _fit_output(_make_panel())

    assert output.fit_result.model is not None
    assert output.filter_result is not None

    prefix_len = 300

    prefix_filter = forward_filter_gaussian(
        X=output.fit_result.scaled_panel.X_scaled[:prefix_len],
        startprob=output.fit_result.model.startprob_,
        transmat=output.fit_result.model.transmat_,
        means=output.fit_result.model.means_,
        covars=output.fit_result.model.covars_,
        covariance_type=output.fit_result.spec.covariance_type,
        min_covar=output.fit_result.fit_config.min_covar,
    )

    assert_prefix_filter_invariance(
        output.filter_result.filtered_probs,
        prefix_filter.filtered_probs,
        atol=1.0e-10,
    )


def test_hmm_signal_availability_is_next_session_only() -> None:
    output = _fit_output(_make_panel())

    assert output.output_panel is not None

    panel = output.output_panel.copy()

    dates = pd.to_datetime(panel["date"])
    trade_dates = pd.to_datetime(panel["hmm_signal_trade_date"], errors="coerce")

    usable = trade_dates.notna()

    assert usable.any()
    assert (trade_dates.loc[usable] > dates.loc[usable]).all()

    assert pd.isna(panel.loc[len(panel) - 1, "hmm_signal_trade_date"])


def test_hmm_no_lookahead_audit_passes_core_checks() -> None:
    output = _fit_output(_make_panel())

    audit = build_hmm_no_lookahead_audit_table(output)

    assert not audit.empty

    required_checks = {
        "hmm_feature_names_point_in_time",
        "chronological_train_test_split",
        "signal_availability_columns_present",
        "signal_trade_date_after_observation_date",
        "backtest_probability_columns_are_filtered_economic_only",
        "hmmlearn_predict_proba_not_used_for_backtest",
        "crisis_and_threshold_diagnostics_not_training_inputs",
    }

    assert required_checks.issubset(set(audit["check_name"]))
    assert audit["passed"].all()
    assert audit["overall_passed"].all()


def test_hmm_probability_audit_passes_and_uses_filtered_economic_backtest_columns() -> None:
    output = _fit_output(_make_panel())

    audit = build_hmm_probability_audit_table(output)

    assert len(audit) == 1
    row = audit.iloc[0]

    assert bool(row["uses_custom_forward_filter"])
    assert not bool(row["uses_hmmlearn_predict_proba_for_backtest"])
    assert not bool(row["uses_smoothed_probabilities_for_backtest"])
    assert bool(row["future_invariance_passed"])
    assert bool(row["passed"])

    backtest_cols = set(str(row["backtest_probability_columns"]).split(","))
    expected = set(get_hmm_filtered_economic_probability_columns())

    assert backtest_cols == expected