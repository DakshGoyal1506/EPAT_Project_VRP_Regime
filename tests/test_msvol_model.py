from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vrp.regimes import msvol_model as m


def make_config(tmp_path: Path, min_observations: int = 20) -> dict:
    config = {
        "model_name": "msvol_appendix_v1",
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "optional_phase": True,
        "required_for_main_pipeline": False,
        "input_policy": {
            "primary_input": "index_log_return",
            "prefilter_mean": True,
            "prefilter_method": "AR1",
            "return_scale": "percent",
            "min_observations": min_observations,
        },
        "markets": {
            "US": {
                "input_csv": "data/interim/msgarch/us_msgarch_input.csv",
                "raw_output_csv": "data/interim/msvol/us_msvol_raw_output.csv",
                "preflight_json": "data/interim/msvol/us_msvol_preflight.json",
                "skip_report_json": "data/interim/msvol/us_msvol_skip_report.json",
                "model_summary_json": "data/interim/msvol/us_msvol_model_summary.json",
            }
        },
        "model_spec": {
            "library": "statsmodels",
            "model_class": "MarkovRegression",
            "k_regimes": 2,
            "trend": "c",
            "switching_variance": True,
        },
        "fit_policy": {
            "maxiter": 20,
            "em_iter": 2,
            "search_reps": 0,
            "search_iter": 0,
            "disp": False,
        },
    }

    path = tmp_path / "configs" / "model_msvol.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return config


def write_input_csv(tmp_path: Path, n: int = 30, market: str = "US") -> Path:
    rng = np.random.default_rng(123)

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0, 0.01, size=n)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "market": market,
            "log_return": returns,
            "return_for_msgarch": 100.0 * returns,
            "source_return_column": "log_return",
            "input_available": True,
        }
    )

    path = tmp_path / "data" / "interim" / "msgarch" / "us_msgarch_input.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_prefilter_ar1_length_and_finite_values():
    x = np.array([0.1, 0.2, 0.15, -0.1, 0.05, 0.08, -0.02, 0.03, 0.07, -0.04])
    result = m.prefilter_ar1(x)

    assert result.residuals.shape == x.shape
    assert result.fitted.shape == x.shape
    assert np.all(np.isfinite(result.residuals))
    assert np.isfinite(result.intercept)
    assert np.isfinite(result.phi)
    assert result.method in {"AR1", "demean_fallback"}


def test_prefilter_rejects_short_series():
    with pytest.raises(m.MSVolError, match="at least 10"):
        m.prefilter_ar1([1.0, 2.0, 3.0])


def test_probability_validation_accepts_valid_rows():
    prob = np.array(
        [
            [0.2, 0.8],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )

    m.validate_probability_matrix(prob)


def test_probability_validation_rejects_bad_rows():
    prob = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(m.MSVolError, match="not summing"):
        m.validate_probability_matrix(prob)


def test_estimate_state_variances_from_probabilities():
    residuals = np.array([1.0, 1.0, 2.0, 4.0])
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    variances = m.estimate_state_variances_from_probabilities(residuals, probs)

    assert variances.shape == (2,)
    assert np.all(variances > 0)
    assert variances[1] > variances[0]


def test_build_raw_output_frame_schema():
    input_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "market": ["US", "US", "US"],
        }
    )

    fit_result = m.MSVolFitResult(
        filtered_probabilities=np.array(
            [
                [0.8, 0.2],
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        ),
        smoothed_probabilities=np.array(
            [
                [0.7, 0.3],
                [0.5, 0.5],
                [0.2, 0.8],
            ]
        ),
        state_variance_estimates=np.array([1.0, 4.0]),
        conditional_variance=np.array([1.6, 2.2, 3.7]),
        fit_status="ok",
        convergence_status="converged",
        selected_spec="test_spec",
        probability_extraction_method="synthetic",
        loglike=-10.0,
        aic=24.0,
        bic=30.0,
        nobs=3,
    )

    out = m.build_raw_output_frame(input_df=input_df, fit_result=fit_result)

    expected = {
        "date",
        "market",
        "msvol_raw_state_0_prob_filtered",
        "msvol_raw_state_1_prob_filtered",
        "msvol_raw_state_0_variance_estimate",
        "msvol_raw_state_1_variance_estimate",
        "msvol_conditional_variance",
        "msvol_conditional_volatility",
        "msvol_model_valid",
        "msvol_fit_status",
        "msvol_skip_reason",
        "msvol_raw_state_0_prob_smoothed_diagnostic",
        "msvol_raw_state_1_prob_smoothed_diagnostic",
    }

    assert set(out.columns) == expected
    assert len(out) == 3
    assert np.allclose(
        out["msvol_raw_state_0_prob_filtered"] + out["msvol_raw_state_1_prob_filtered"],
        1.0,
    )


def test_read_msvol_input_csv_validates_schema(tmp_path: Path):
    path = write_input_csv(tmp_path, n=20)

    df = m.read_msvol_input_csv(path, expected_market="US")

    assert len(df) == 20
    assert list(df.columns) == m.REQUIRED_INPUT_COLUMNS
    assert df["date"].is_monotonic_increasing


def test_read_msvol_input_csv_rejects_wrong_market(tmp_path: Path):
    path = write_input_csv(tmp_path, n=20, market="INDIA")

    with pytest.raises(m.MSVolError, match="not matching market"):
        m.read_msvol_input_csv(path, expected_market="US")


def test_run_market_with_injected_fake_fit(monkeypatch, tmp_path: Path):
    config = make_config(tmp_path, min_observations=20)
    write_input_csv(tmp_path, n=25)

    def fake_fit(residuals, config):
        n = len(residuals)
        filtered = np.column_stack(
            [
                np.linspace(0.9, 0.1, n),
                np.linspace(0.1, 0.9, n),
            ]
        )

        filtered = filtered / filtered.sum(axis=1, keepdims=True)

        return m.MSVolFitResult(
            filtered_probabilities=filtered,
            smoothed_probabilities=None,
            state_variance_estimates=np.array([1.0, 5.0]),
            conditional_variance=filtered @ np.array([1.0, 5.0]),
            fit_status="ok",
            convergence_status="converged",
            selected_spec="fake_spec",
            probability_extraction_method="fake",
            loglike=-1.0,
            aic=2.0,
            bic=3.0,
            nobs=n,
        )

    monkeypatch.setattr(m, "fit_msvol_markov_regression", fake_fit)

    result = m.run_msvol_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=False,
    )

    assert result.status == "ok"
    assert result.raw_output_csv.exists()
    assert result.preflight_json.exists()
    assert result.model_summary_json.exists()

    raw = pd.read_csv(result.raw_output_csv)
    assert len(raw) == 25
    assert "msvol_raw_state_0_prob_filtered" in raw.columns
    assert "msvol_raw_state_1_prob_filtered" in raw.columns

    with result.model_summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["true_msgarch"] is False
    assert summary["lower_variance_raw_state"] == 0
    assert summary["higher_variance_raw_state"] == 1


def test_run_market_allow_skip_writes_skip_report(tmp_path: Path):
    config = make_config(tmp_path, min_observations=20)

    result = m.run_msvol_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=True,
    )

    assert result.status == "skipped"
    assert result.skip_report_json.exists()
    assert result.preflight_json.exists()

    with result.skip_report_json.open("r", encoding="utf-8") as f:
        skip = json.load(f)

    assert skip["skipped"] is True
    assert skip["true_msgarch"] is False