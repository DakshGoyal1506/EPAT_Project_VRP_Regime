from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vrp.regimes.markov_autoreg_registry import (
    MARModelSpec,
    load_markov_autoreg_config,
    raw_filtered_probability_columns,
)
from vrp.regimes.markov_autoreg import (
    DATE_COL,
    TRANSFORMED_TARGET_COL,
    MARCandidateFit,
    MARFitAttempt,
    MARFitFirewallSummary,
    MARFullFilterResult,
    MARParameterLookaheadAudit,
    MARProbabilityAudit,
    add_economic_state_probabilities,
    add_next_session_signal_columns,
    align_mar_probabilities_to_eligible_frame,
    audit_aligned_probabilities,
    build_mar_signal_output,
    build_parameter_lookahead_audit,
    build_train_state_economic_summary,
    compute_mar_stress_scores,
    economic_state_code,
    fit_apply_target_transform_train_only,
    get_endog,
    label_mar_states_economically,
    prepare_mar_model_data,
    rank_as_unit_interval,
    state_occupancy_from_aligned_frame,
    validate_mar_signal_output,
)


def make_synthetic_panel(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    base = np.r_[
        rng.normal(0.04, 0.01, n // 2),
        rng.normal(-0.01, 0.03, n - n // 2),
    ]

    rv = np.r_[
        rng.normal(0.08, 0.01, n // 2),
        rng.normal(0.20, 0.04, n - n // 2),
    ]

    iv = np.r_[
        rng.normal(0.12, 0.02, n // 2),
        rng.normal(0.35, 0.07, n - n // 2),
    ]

    returns = np.r_[
        rng.normal(0.0004, 0.006, n // 2),
        rng.normal(-0.0008, 0.015, n - n // 2),
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "vrp_har_gk": base,
            "rv_gk_22d_ann_lag1": rv,
            "iv_ann": iv,
            "index_return": returns,
            "har_forecast_available": True,
        }
    )


def make_config_for_tests():
    cfg = load_markov_autoreg_config("configs/model_markov_autoreg.yaml")

    # Keep tests fast and independent of the user's full sample size.
    object.__setattr__(
        cfg,
        "train_test_split",
        replace(
            cfg.train_test_split,
            min_train_observations=50,
            min_test_observations=25,
        ),
    )

    object.__setattr__(
        cfg,
        "validation",
        replace(
            cfg.validation,
            min_available_fraction_after_warmup=0.20,
        ),
    )

    return cfg


def test_config_loads_and_primary_spec_is_phase7_default():
    cfg = load_markov_autoreg_config("configs/model_markov_autoreg.yaml")
    spec = cfg.primary_model

    assert cfg.model_name == "markov_autoreg_v1"
    assert cfg.implementation == "statsmodels_markov_autoregression"
    assert spec.target == "vrp_har"
    assert spec.order == 1
    assert spec.n_states == 2
    assert spec.switching_ar is True
    assert spec.switching_trend is True
    assert spec.switching_variance is True


def test_prepare_mar_model_data_uses_only_allowed_target_and_availability_rule():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(300)

    panel.loc[:9, "har_forecast_available"] = False
    panel["hmm_state_for_next_session"] = 1
    panel["rv_gk_22d_forward_ann_label"] = 0.2

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    assert prepared.target_col == "vrp_har_gk"
    assert prepared.eligibility.n_eligible == 290
    assert "hmm_state_for_next_session" in prepared.forbidden_columns_present_in_panel
    assert "rv_gk_22d_forward_ann_label" in prepared.forbidden_columns_present_in_panel
    assert TRANSFORMED_TARGET_COL in prepared.eligible_frame.columns


def test_get_endog_returns_transformed_target_without_nan_or_inf():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(300)

    prepared = prepare_mar_model_data(
        df=panel,
        market="INDIA",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    train_y = get_endog(prepared, "train")
    full_y = get_endog(prepared, "full")

    assert len(train_y) == prepared.split.n_train
    assert len(full_y) == prepared.eligibility.n_eligible
    assert np.isfinite(train_y.to_numpy()).all()
    assert np.isfinite(full_y.to_numpy()).all()


def test_train_only_winsorize_standardize_transform_is_fit_on_train_only():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(300)

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    train_idx = prepared.eligible_frame.index[: prepared.split.n_train]

    transformed, summary = fit_apply_target_transform_train_only(
        eligible_frame=prepared.eligible_frame,
        train_idx=train_idx,
        spec=spec,
        cfg=cfg,
        target_col=prepared.target_col,
    )

    assert len(transformed) == len(prepared.eligible_frame)
    assert np.isfinite(transformed.to_numpy()).all()

    if summary.method == "winsorize_train_quantiles_then_standardize":
        assert "lower_cap_estimated_on_train" in summary.params
        assert "upper_cap_estimated_on_train" in summary.params


def test_probability_alignment_blanks_order_one_warmup_row():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(100)

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=False,
    )

    n = len(prepared.eligible_frame)
    raw_filtered = pd.DataFrame(
        {
            "mar_filtered_prob_raw_state_0": np.linspace(0.2, 0.8, n),
            "mar_filtered_prob_raw_state_1": 1.0 - np.linspace(0.2, 0.8, n),
        }
    )

    raw_smoothed = pd.DataFrame(
        {
            "mar_diagnostic_smoothed_prob_raw_state_0": np.linspace(0.1, 0.9, n),
            "mar_diagnostic_smoothed_prob_raw_state_1": 1.0 - np.linspace(0.1, 0.9, n),
        }
    )

    aligned = align_mar_probabilities_to_eligible_frame(
        prepared=prepared,
        spec=spec,
        raw_filtered_probabilities=raw_filtered,
        raw_smoothed_probabilities_diagnostic=raw_smoothed,
    )

    assert bool(aligned.loc[0, "mar_model_observation_available"]) is False
    assert pd.isna(aligned.loc[0, "mar_filtered_prob_raw_state_0"])
    assert pd.isna(aligned.loc[0, "mar_filtered_prob_raw_state_1"])

    assert bool(aligned.loc[1:, "mar_model_observation_available"].all())
    assert aligned.loc[1:, raw_filtered_probability_columns(2)].notna().all().all()


def test_probability_alignment_accepts_n_minus_order_probability_rows():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(100)

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=False,
    )

    n = len(prepared.eligible_frame)
    m = n - spec.order

    raw_filtered = pd.DataFrame(
        {
            "mar_filtered_prob_raw_state_0": np.full(m, 0.25),
            "mar_filtered_prob_raw_state_1": np.full(m, 0.75),
        }
    )

    raw_smoothed = pd.DataFrame(
        {
            "mar_diagnostic_smoothed_prob_raw_state_0": np.full(m, 0.20),
            "mar_diagnostic_smoothed_prob_raw_state_1": np.full(m, 0.80),
        }
    )

    aligned = align_mar_probabilities_to_eligible_frame(
        prepared=prepared,
        spec=spec,
        raw_filtered_probabilities=raw_filtered,
        raw_smoothed_probabilities_diagnostic=raw_smoothed,
    )

    assert pd.isna(aligned.loc[0, "mar_filtered_prob_raw_state_0"])
    assert aligned.loc[1, "mar_filtered_prob_raw_state_0"] == pytest.approx(0.25)
    assert aligned.loc[1, "mar_filtered_prob_raw_state_1"] == pytest.approx(0.75)


def test_probability_audit_passes_for_valid_aligned_probabilities():
    cfg = make_config_for_tests()
    spec = cfg.primary_model
    panel = make_synthetic_panel(100)

    prepared = prepare_mar_model_data(
        df=panel,
        market="INDIA",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=False,
    )

    n = len(prepared.eligible_frame)
    raw_filtered = pd.DataFrame(
        {
            "mar_filtered_prob_raw_state_0": np.full(n, 0.35),
            "mar_filtered_prob_raw_state_1": np.full(n, 0.65),
        }
    )
    raw_smoothed = pd.DataFrame(
        {
            "mar_diagnostic_smoothed_prob_raw_state_0": np.full(n, 0.35),
            "mar_diagnostic_smoothed_prob_raw_state_1": np.full(n, 0.65),
        }
    )

    aligned = align_mar_probabilities_to_eligible_frame(
        prepared=prepared,
        spec=spec,
        raw_filtered_probabilities=raw_filtered,
        raw_smoothed_probabilities_diagnostic=raw_smoothed,
    )

    audit = audit_aligned_probabilities(aligned, spec, cfg)

    assert audit.passed is True
    assert audit.n_warmup_rows == 1
    assert audit.n_model_available_rows == n - 1
    assert audit.max_row_sum_abs_error == pytest.approx(0.0)


def test_rank_as_unit_interval_direction():
    values = pd.Series([10.0, 20.0, 30.0])

    high_stress = rank_as_unit_interval(values, higher_is_stress=True)
    low_stress = rank_as_unit_interval(values, higher_is_stress=False)

    assert high_stress.tolist() == [0.0, 0.5, 1.0]
    assert low_stress.tolist() == [1.0, 0.5, 0.0]


def test_economic_state_code_mapping():
    assert economic_state_code("calm") == 0
    assert economic_state_code("transition") == 1
    assert economic_state_code("stress") == 2

    with pytest.raises(ValueError):
        economic_state_code("unknown")


def test_compute_stress_scores_for_vrp_har_prefers_lower_vrp_and_higher_vol():
    cfg = make_config_for_tests()
    spec = cfg.primary_model

    summary = pd.DataFrame(
        {
            "raw_state": [0, 1],
            "target_mean_train": [-0.01, 0.04],
            "target_std_train": [0.05, 0.01],
            "index_return_mean_train": [-0.001, 0.001],
            "iv_mean_train": [0.30, 0.12],
            "rv_mean_train": [0.25, 0.08],
            "sigma2": [0.20, 0.01],
        }
    )

    scores = compute_mar_stress_scores(summary, spec, cfg)

    assert scores.iloc[0] > scores.iloc[1]


def test_label_mar_states_k2_maps_lowest_score_to_calm_highest_to_stress():
    cfg = make_config_for_tests()
    spec = cfg.primary_model

    state_summary = pd.DataFrame(
        {
            "raw_state": [0, 1],
            "stress_score": [0.9, 0.1],
            "target_mean_train": [-0.02, 0.04],
            "target_std_train": [0.05, 0.01],
            "sigma2": [0.2, 0.01],
            "index_return_mean_train": [-0.001, 0.001],
            "iv_mean_train": [0.30, 0.12],
            "rv_mean_train": [0.20, 0.08],
        }
    )

    mapping = label_mar_states_economically(state_summary, spec, cfg)

    assert mapping.raw_state_to_name[1] == "calm"
    assert mapping.raw_state_to_name[0] == "stress"
    assert mapping.name_to_raw_state["transition"] is None
    assert mapping.transition_state_modelled is False


def test_add_economic_probabilities_k2_sets_transition_zero_after_warmup():
    cfg = make_config_for_tests()
    spec = cfg.primary_model

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "mar_eligible_observation_number": [0, 1, 2, 3],
            "mar_model_observation_available": [False, True, True, True],
            "mar_raw_state_filtered": [np.nan, 1.0, 0.0, 1.0],
            "mar_filtered_prob_raw_state_0": [np.nan, 0.2, 0.9, 0.3],
            "mar_filtered_prob_raw_state_1": [np.nan, 0.8, 0.1, 0.7],
        }
    )

    state_summary = pd.DataFrame(
        {
            "raw_state": [0, 1],
            "stress_score": [0.9, 0.1],
            "target_mean_train": [-0.02, 0.04],
            "target_std_train": [0.05, 0.01],
            "sigma2": [0.2, 0.01],
            "index_return_mean_train": [-0.001, 0.001],
            "iv_mean_train": [0.30, 0.12],
            "rv_mean_train": [0.20, 0.08],
        }
    )

    mapping = label_mar_states_economically(state_summary, spec, cfg)

    out = add_economic_state_probabilities(frame, spec, mapping)

    assert pd.isna(out.loc[0, "mar_filtered_prob_transition"])
    assert out.loc[1, "mar_filtered_prob_transition"] == pytest.approx(0.0)
    assert out.loc[2, "mar_filtered_prob_stress"] == pytest.approx(0.9)
    assert out.loc[1, "mar_filtered_prob_calm"] == pytest.approx(0.8)


def test_next_session_signal_columns_use_current_observation_for_next_trade_date():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "mar_model_observation_available": [False, True, True, True],
            "mar_state": [np.nan, 0.0, 2.0, 0.0],
            "mar_state_name": [np.nan, "calm", "stress", "calm"],
            "mar_filtered_prob_calm": [np.nan, 0.9, 0.1, 0.8],
            "mar_filtered_prob_transition": [np.nan, 0.0, 0.0, 0.0],
            "mar_filtered_prob_stress": [np.nan, 0.1, 0.9, 0.2],
        }
    )

    out = add_next_session_signal_columns(frame)

    assert out.loc[1, "mar_signal_observation_date"] == out.loc[1, "date"]
    assert out.loc[1, "mar_signal_available_after_close_date"] == out.loc[1, "date"]
    assert out.loc[1, "mar_signal_trade_date"] == out.loc[2, "date"]

    assert pd.isna(out.loc[0, "mar_state_for_next_session"])
    assert out.loc[1, "mar_state_name_for_next_session"] == "calm"
    assert out.loc[2, "mar_state_name_for_next_session"] == "stress"


def test_validate_mar_signal_output_accepts_valid_k2_frame():
    cfg = make_config_for_tests()
    spec = cfg.primary_model

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "mar_eligible_observation_number": [0, 1, 2, 3],
            "mar_model_observation_available": [False, True, True, True],
            "mar_state_for_next_session": [np.nan, 0.0, 2.0, 0.0],
            "mar_state_name_for_next_session": [np.nan, "calm", "stress", "calm"],
            "mar_signal_observation_date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "mar_signal_available_after_close_date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "mar_signal_trade_date": pd.date_range("2020-01-02", periods=4, freq="B"),
            "mar_filtered_prob_calm": [np.nan, 0.9, 0.1, 0.8],
            "mar_filtered_prob_transition": [np.nan, 0.0, 0.0, 0.0],
            "mar_filtered_prob_stress": [np.nan, 0.1, 0.9, 0.2],
            "mar_filtered_prob_calm_for_next_session": [np.nan, 0.9, 0.1, 0.8],
            "mar_filtered_prob_transition_for_next_session": [np.nan, 0.0, 0.0, 0.0],
            "mar_filtered_prob_stress_for_next_session": [np.nan, 0.1, 0.9, 0.2],
        }
    )

    validate_mar_signal_output(frame, spec, cfg)
    
    