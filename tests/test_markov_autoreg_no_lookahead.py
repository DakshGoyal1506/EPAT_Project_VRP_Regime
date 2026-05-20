from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vrp.regimes.markov_autoreg_registry import load_markov_autoreg_config
from vrp.regimes.markov_autoreg import (
    DATE_COL,
    MARCandidateFit,
    MARFitAttempt,
    MARFitFirewallSummary,
    MARFullFilterResult,
    MARParameterLookaheadAudit,
    MARProbabilityAudit,
    add_next_session_signal_columns,
    audit_aligned_probabilities,
    build_parameter_lookahead_audit,
    prepare_mar_model_data,
)


def make_panel(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    return pd.DataFrame(
        {
            "date": dates,
            "vrp_har_gk": rng.normal(0.03, 0.02, n),
            "rv_gk_22d_ann_lag1": rng.normal(0.12, 0.02, n),
            "iv_ann": rng.normal(0.18, 0.03, n),
            "index_return": rng.normal(0.0002, 0.01, n),
            "har_forecast_available": True,
        }
    )


def make_cfg():
    cfg = load_markov_autoreg_config("configs/model_markov_autoreg.yaml")

    object.__setattr__(
        cfg,
        "train_test_split",
        replace(
            cfg.train_test_split,
            min_train_observations=20,
            min_test_observations=10,
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


def make_dummy_candidate(prepared):
    spec = prepared.spec

    fit_summary = MARFitFirewallSummary(
        fit_converged=True,
        fit_exception=None,
        llf=1.0,
        aic=2.0,
        bic=3.0,
        hqic=4.0,
        nobs=len(prepared.train_frame),
        n_params=8,
        warnflag=0,
        mle_retvals={"converged": True},
        valid_candidate=True,
        invalid_reason="",
        warnings=tuple(),
    )

    return MARCandidateFit(
        market=prepared.market,
        spec=spec,
        prepared=prepared,
        model=None,
        result=type("DummyResult", (), {"params": np.array([0.1, 0.2])})(),
        train_filtered_probabilities=pd.DataFrame(
            {
                "mar_filtered_prob_raw_state_0": [0.2, 0.8],
                "mar_filtered_prob_raw_state_1": [0.8, 0.2],
            }
        ),
        train_state_assignments=pd.Series([1, 0], name="mar_raw_state"),
        train_state_occupancy={0: 0.5, 1: 0.5},
        transition_matrix=pd.DataFrame([[0.9, 0.1], [0.2, 0.8]]),
        ar_coefficients={0: 0.5, 1: 0.9},
        sigma2_by_state={0: 0.2, 1: 0.1},
        fit_summary=fit_summary,
        fit_attempts=(
            MARFitAttempt(
                attempt_id=1,
                method="bfgs",
                cov_type="approx",
                maxiter=100,
                em_iter=5,
                search_reps=0,
                disp=False,
                success=True,
                exception=None,
            ),
        ),
    )


def test_parameter_lookahead_audit_explicitly_marks_train_only_params():
    cfg = make_cfg()
    spec = cfg.primary_model
    panel = make_panel()

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    candidate = make_dummy_candidate(prepared)

    aligned = prepared.eligible_frame.copy()
    audit = build_parameter_lookahead_audit(candidate, aligned)

    assert audit.params_estimated_using_full_sample is False
    assert audit.filtered_probabilities_use_train_params_only is True
    assert audit.smoothed_probabilities_used_for_backtest is False
    assert audit.passed is True

    assert audit.mar_fit_window_start == prepared.split.train_start_date
    assert audit.mar_fit_window_end == prepared.split.train_end_date
    assert audit.mar_filter_window_start == str(prepared.eligible_frame[DATE_COL].iloc[0].date())
    assert audit.mar_filter_window_end == str(prepared.eligible_frame[DATE_COL].iloc[-1].date())


def test_probability_audit_rejects_warmup_probabilities():
    cfg = make_cfg()
    spec = cfg.primary_model
    panel = make_panel()

    prepared = prepare_mar_model_data(
        df=panel,
        market="INDIA",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    aligned = prepared.eligible_frame.copy()
    aligned["mar_model_observation_available"] = True
    aligned["mar_filtered_prob_raw_state_0"] = 0.4
    aligned["mar_filtered_prob_raw_state_1"] = 0.6

    audit = audit_aligned_probabilities(aligned, spec, cfg)

    assert audit.passed is False
    assert "AR warmup rows contain filtered probabilities" in audit.invalid_reason


def test_probability_audit_rejects_bad_row_sums_after_warmup():
    cfg = make_cfg()
    spec = cfg.primary_model
    panel = make_panel()

    prepared = prepare_mar_model_data(
        df=panel,
        market="US",
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    aligned = prepared.eligible_frame.copy()
    aligned["mar_model_observation_available"] = False
    aligned.loc[1:, "mar_model_observation_available"] = True

    aligned["mar_filtered_prob_raw_state_0"] = np.nan
    aligned["mar_filtered_prob_raw_state_1"] = np.nan

    aligned.loc[1:, "mar_filtered_prob_raw_state_0"] = 0.7
    aligned.loc[1:, "mar_filtered_prob_raw_state_1"] = 0.7

    audit = audit_aligned_probabilities(aligned, spec, cfg)

    assert audit.passed is False
    assert "probability row sums exceed tolerance" in audit.invalid_reason


def test_next_session_signal_uses_next_available_row_not_calendar_day():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])

    frame = pd.DataFrame(
        {
            "date": dates,
            "mar_model_observation_available": [True, True, True],
            "mar_state": [0.0, 2.0, 0.0],
            "mar_state_name": ["calm", "stress", "calm"],
            "mar_filtered_prob_calm": [0.9, 0.2, 0.8],
            "mar_filtered_prob_transition": [0.0, 0.0, 0.0],
            "mar_filtered_prob_stress": [0.1, 0.8, 0.2],
        }
    )

    out = add_next_session_signal_columns(frame)

    assert out.loc[0, "mar_signal_observation_date"] == pd.Timestamp("2024-01-05")
    assert out.loc[0, "mar_signal_available_after_close_date"] == pd.Timestamp("2024-01-05")
    assert out.loc[0, "mar_signal_trade_date"] == pd.Timestamp("2024-01-08")
    assert out.loc[1, "mar_signal_trade_date"] == pd.Timestamp("2024-01-09")
    assert pd.isna(out.loc[2, "mar_signal_trade_date"])


def test_no_lookahead_audit_required_fields_match_phase7_policy():
    audit = MARParameterLookaheadAudit(
        mar_fit_window_start="2020-01-01",
        mar_fit_window_end="2021-01-01",
        mar_filter_window_start="2020-01-01",
        mar_filter_window_end="2024-01-01",
        params_estimated_using_full_sample=False,
        filtered_probabilities_use_train_params_only=True,
        smoothed_probabilities_used_for_backtest=False,
        passed=True,
    )

    assert audit.params_estimated_using_full_sample is False
    assert audit.filtered_probabilities_use_train_params_only is True
    assert audit.smoothed_probabilities_used_for_backtest is False
    assert audit.passed is True


def test_mar_probability_audit_dataclass_records_failure_reason():
    audit = MARProbabilityAudit(
        n_rows=10,
        n_model_available_rows=9,
        n_warmup_rows=1,
        n_probability_rows=9,
        n_nan_rows_after_warmup=1,
        max_row_sum_abs_error=0.2,
        probability_row_sum_tolerance=1.0e-8,
        passed=False,
        invalid_reason="bad rows",
    )

    assert audit.passed is False
    assert audit.invalid_reason == "bad rows"