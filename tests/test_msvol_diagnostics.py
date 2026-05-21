from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vrp.reports import msvol_diagnostics as d


def make_config(tmp_path: Path) -> tuple[dict, Path]:
    config = {
        "model_name": "msvol_appendix_v1",
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "markets": {
            "US": {
                "input_csv": "data/interim/msgarch/us_msgarch_input.csv",
                "raw_output_csv": "data/interim/msvol/us_msvol_raw_output.csv",
                "preflight_json": "data/interim/msvol/us_msvol_preflight.json",
                "skip_report_json": "data/interim/msvol/us_msvol_skip_report.json",
                "model_summary_json": "data/interim/msvol/us_msvol_model_summary.json",
            }
        },
        "output_policy": {
            "processed_dir": "data/processed",
            "phase8_report_dir": "reports/tables/phase_8",
        },
    }

    path = tmp_path / "configs" / "model_msvol.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return config, path


def make_msvol_processed() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")

    return pd.DataFrame(
        {
            "date": dates,
            "market": ["US"] * 6,
            "msvol_signal_observation_date": dates,
            "msvol_signal_available_after_close_date": dates,
            "msvol_signal_trade_date": list(dates[1:]) + [pd.NaT],
            "msvol_state_for_next_session": [0, 0, 2, 2, 0, 2],
            "msvol_state_name_for_next_session": ["calm", "calm", "stress", "stress", "calm", "stress"],
            "msvol_filtered_prob_calm_for_next_session": [0.9, 0.8, 0.2, 0.1, 0.7, 0.3],
            "msvol_filtered_prob_transition_for_next_session": [0.0] * 6,
            "msvol_filtered_prob_stress_for_next_session": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "msvol_transition_state_modelled": [False] * 6,
            "msvol_calm_raw_state": [0] * 6,
            "msvol_stress_raw_state": [1] * 6,
            "msvol_lower_variance_raw_state": [0] * 6,
            "msvol_higher_variance_raw_state": [1] * 6,
            "msvol_raw_state_0_variance_estimate": [1.0] * 6,
            "msvol_raw_state_1_variance_estimate": [4.0] * 6,
            "msvol_conditional_variance": [1.3, 1.5, 3.1, 3.6, 1.9, 3.2],
            "msvol_conditional_volatility": np.sqrt([1.3, 1.5, 3.1, 3.6, 1.9, 3.2]),
            "msvol_model_valid": [True] * 6,
            "msvol_fit_status": ["ok"] * 6,
            "msvol_skip_reason": [""] * 6,
        }
    )


def write_msvol_processed(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "data" / "processed" / "us_msvol_regimes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_comparator(tmp_path: Path, name: str, states: list[str]) -> Path:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    path = tmp_path / "data" / "processed" / f"us_{name}_regimes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": dates,
            f"{name}_state_name_for_next_session": states,
        }
    )
    df.to_parquet(path, index=False)
    return path


def write_feature_panel(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    path = tmp_path / "data" / "processed" / "us_vrp_har.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": dates,
            "market": ["US"] * 6,
            "log_return": [0.01, 0.02, -0.03, -0.02, 0.01, -0.04],
            "rv_gk_22d_ann": [0.10, 0.11, 0.30, 0.32, 0.14, 0.28],
            "vrp_har_gk": [0.03, 0.04, -0.01, -0.02, 0.02, -0.03],
        }
    )
    df.to_parquet(path, index=False)
    return path


def test_validate_msvol_processed_schema_accepts_valid_frame():
    df = make_msvol_processed()
    d.validate_msvol_processed_schema(df, expected_market="US")


def test_validate_msvol_processed_schema_rejects_bad_probability_sum():
    df = make_msvol_processed()
    df.loc[0, "msvol_filtered_prob_stress_for_next_session"] = 0.2

    with pytest.raises(d.MSVolDiagnosticsError, match="summing"):
        d.validate_msvol_processed_schema(df, expected_market="US")


def test_extract_stress_indicator_from_state_name():
    comparator = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "threshold_state_name_for_next_session": ["calm", "stress", "transition"],
        }
    )

    out = d.extract_stress_indicator(comparator, "threshold")

    assert out["threshold_stress"].tolist() == [0.0, 1.0, 0.0]


def test_compute_pairwise_comparison():
    msvol = make_msvol_processed()
    comparator = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="B"),
            "threshold_state_name_for_next_session": ["calm", "calm", "stress", "stress", "calm", "stress"],
        }
    )

    metrics = d.compute_pairwise_comparison(msvol, comparator, "threshold")

    assert metrics["n_overlap_days_threshold"] == 6
    assert metrics["threshold_agreement_rate"] == pytest.approx(1.0)
    assert metrics["avg_msvol_stress_prob_in_threshold_stress"] == pytest.approx((0.8 + 0.9 + 0.7) / 3)


def test_state_duration_summary():
    msvol = make_msvol_processed()

    duration = d.build_state_duration_summary("US", msvol)

    assert set(duration["state_name"]) == {"calm", "stress"}

    calm = duration[duration["state_name"] == "calm"].iloc[0]
    stress = duration[duration["state_name"] == "stress"].iloc[0]

    assert calm["n_runs"] == 2
    assert stress["n_runs"] == 2
    assert calm["total_days"] == 3
    assert stress["total_days"] == 3


def test_run_diagnostics_writes_outputs(tmp_path: Path):
    config, _ = make_config(tmp_path)

    write_msvol_processed(tmp_path, make_msvol_processed())
    write_comparator(tmp_path, "threshold", ["calm", "calm", "stress", "stress", "calm", "stress"])
    write_comparator(tmp_path, "hmm", ["calm", "stress", "stress", "calm", "calm", "stress"])

    mar_path = tmp_path / "data" / "processed" / "us_mar_regimes.parquet"
    mar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="B"),
            "mar_state_name_for_next_session": ["stress", "calm", "stress", "stress", "calm", "calm"],
        }
    ).to_parquet(mar_path, index=False)

    write_feature_panel(tmp_path)

    result = d.run_msvol_diagnostics_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=False,
    )

    assert result.status == "ok"
    assert result.comparison_summary_csv.exists()
    assert result.state_duration_summary_csv.exists()
    assert result.appendix_csv.exists()

    summary = pd.read_csv(result.comparison_summary_csv)
    assert summary.loc[0, "market"] == "US"
    assert summary.loc[0, "status"] == "ok"
    assert summary.loc[0, "n_msvol_days"] == 6
    assert summary.loc[0, "n_overlap_days_threshold"] == 6
    assert summary.loc[0, "threshold_agreement_rate"] == pytest.approx(1.0)
    assert summary.loc[0, "used_for_strategy"] == False
    assert summary.loc[0, "used_for_backtest"] == False
    assert summary.loc[0, "selected_return_column"] == "log_return"
    assert summary.loc[0, "selected_rv_column"] == "rv_gk_22d_ann"
    assert summary.loc[0, "selected_vrp_column"] == "vrp_har_gk"

    duration = pd.read_csv(result.state_duration_summary_csv)
    assert set(duration["state_name"]) == {"calm", "stress"}

    appendix = pd.read_csv(result.appendix_csv)
    assert "US" in appendix["market"].tolist()


def test_run_diagnostics_with_missing_comparators_does_not_fail(tmp_path: Path):
    config, _ = make_config(tmp_path)

    write_msvol_processed(tmp_path, make_msvol_processed())

    result = d.run_msvol_diagnostics_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=False,
    )

    summary = pd.read_csv(result.comparison_summary_csv)

    assert summary.loc[0, "n_msvol_days"] == 6
    assert summary.loc[0, "n_overlap_days_threshold"] == 0
    assert pd.isna(summary.loc[0, "threshold_agreement_rate"])


def test_run_diagnostics_allow_skip_when_msvol_missing(tmp_path: Path):
    config, _ = make_config(tmp_path)

    result = d.run_msvol_diagnostics_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=True,
    )

    assert result.status == "skipped"
    assert result.comparison_summary_csv.exists()
    assert result.state_duration_summary_csv.exists()

    summary = pd.read_csv(result.comparison_summary_csv)
    assert summary.loc[0, "status"] == "skipped"
    assert summary.loc[0, "used_for_strategy"] == False
    assert summary.loc[0, "used_for_backtest"] == False