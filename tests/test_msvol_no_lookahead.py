from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vrp.reports import msvol_no_lookahead as n


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


def make_processed_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")

    return pd.DataFrame(
        {
            "date": dates,
            "market": ["US"] * 5,
            "msvol_signal_observation_date": dates,
            "msvol_signal_available_after_close_date": dates,
            "msvol_signal_trade_date": list(dates[1:]) + [pd.NaT],
            "msvol_state_for_next_session": [0, 0, 2, 2, 0],
            "msvol_state_name_for_next_session": ["calm", "calm", "stress", "stress", "calm"],
            "msvol_filtered_prob_calm_for_next_session": [0.9, 0.8, 0.2, 0.1, 0.7],
            "msvol_filtered_prob_transition_for_next_session": [0.0] * 5,
            "msvol_filtered_prob_stress_for_next_session": [0.1, 0.2, 0.8, 0.9, 0.3],
            "msvol_transition_state_modelled": [False] * 5,
            "msvol_calm_raw_state": [0] * 5,
            "msvol_stress_raw_state": [1] * 5,
            "msvol_lower_variance_raw_state": [0] * 5,
            "msvol_higher_variance_raw_state": [1] * 5,
            "msvol_raw_state_0_variance_estimate": [1.0] * 5,
            "msvol_raw_state_1_variance_estimate": [4.0] * 5,
            "msvol_conditional_variance": [1.2, 1.5, 3.4, 3.8, 1.6],
            "msvol_conditional_volatility": np.sqrt([1.2, 1.5, 3.4, 3.8, 1.6]),
            "msvol_model_valid": [True] * 5,
            "msvol_fit_status": ["ok"] * 5,
            "msvol_skip_reason": [""] * 5,
            "msvol_smoothed_prob_calm_diagnostic": [0.85, 0.75, 0.25, 0.15, 0.65],
            "msvol_smoothed_prob_stress_diagnostic": [0.15, 0.25, 0.75, 0.85, 0.35],
        }
    )


def write_processed(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "data" / "processed" / "us_msvol_regimes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_metadata_files(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "tables" / "phase_8" / "us"
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "market": "US",
        "true_msgarch": False,
        "used_for_strategy": False,
        "used_for_backtest": False,
        "status": "ok",
    }

    diagnostics_metadata = {
        "market": "US",
        "diagnostic_only": True,
        "used_for_strategy": False,
        "used_for_backtest": False,
        "status": "ok",
    }

    with (report_dir / "msvol_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    with (report_dir / "msvol_diagnostics_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics_metadata, f)

    comparison = pd.DataFrame(
        {
            "market": ["US"],
            "diagnostic_only": [True],
            "used_for_strategy": [False],
            "used_for_backtest": [False],
        }
    )
    comparison.to_csv(report_dir / "msvol_comparison_summary.csv", index=False)


def failed_error_checks(audit: pd.DataFrame) -> pd.DataFrame:
    return audit[(audit["severity"] == "error") & (~audit["passed"].astype(bool))]


def test_build_no_lookahead_audit_passes_valid_frame(tmp_path: Path):
    processed = make_processed_frame()

    audit = n.build_no_lookahead_audit(
        market="US",
        processed=processed,
        metadata={
            "true_msgarch": False,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
        comparison_summary=pd.DataFrame(
            {
                "diagnostic_only": [True],
                "used_for_strategy": [False],
                "used_for_backtest": [False],
            }
        ),
        diagnostics_metadata={
            "diagnostic_only": True,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
    )

    assert len(audit) > 0
    assert failed_error_checks(audit).empty


def test_audit_rejects_same_day_trade_date():
    processed = make_processed_frame()
    processed.loc[0, "msvol_signal_trade_date"] = processed.loc[0, "date"]

    audit = n.build_no_lookahead_audit(
        market="US",
        processed=processed,
        metadata={
            "true_msgarch": False,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
        comparison_summary=pd.DataFrame(
            {
                "diagnostic_only": [True],
                "used_for_strategy": [False],
                "used_for_backtest": [False],
            }
        ),
        diagnostics_metadata={
            "diagnostic_only": True,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
    )

    failures = failed_error_checks(audit)
    assert "trade_date_is_next_available_row_date" in failures["check_name"].tolist()
    assert "trade_date_strictly_after_observation_for_tradable_rows" in failures["check_name"].tolist()


def test_audit_rejects_smoothed_next_session_column():
    processed = make_processed_frame()
    processed["msvol_smoothed_prob_stress_for_next_session"] = processed[
        "msvol_smoothed_prob_stress_diagnostic"
    ]

    audit = n.build_no_lookahead_audit(
        market="US",
        processed=processed,
        metadata={
            "true_msgarch": False,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
        comparison_summary=pd.DataFrame(
            {
                "diagnostic_only": [True],
                "used_for_strategy": [False],
                "used_for_backtest": [False],
            }
        ),
        diagnostics_metadata={
            "diagnostic_only": True,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
    )

    failures = failed_error_checks(audit)
    assert "no_smoothed_probability_next_session_columns" in failures["check_name"].tolist()
    assert "smoothed_columns_are_diagnostic_only" in failures["check_name"].tolist()


def test_audit_rejects_probability_sum_error():
    processed = make_processed_frame()
    processed.loc[0, "msvol_filtered_prob_stress_for_next_session"] = 0.2

    audit = n.build_no_lookahead_audit(
        market="US",
        processed=processed,
        metadata={
            "true_msgarch": False,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
        comparison_summary=pd.DataFrame(
            {
                "diagnostic_only": [True],
                "used_for_strategy": [False],
                "used_for_backtest": [False],
            }
        ),
        diagnostics_metadata={
            "diagnostic_only": True,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
    )

    failures = failed_error_checks(audit)
    assert "filtered_probabilities_sum_to_one" in failures["check_name"].tolist()


def test_audit_rejects_strategy_like_columns():
    processed = make_processed_frame()
    processed["msvol_strategy_return"] = 0.0

    audit = n.build_no_lookahead_audit(
        market="US",
        processed=processed,
        metadata={
            "true_msgarch": False,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
        comparison_summary=pd.DataFrame(
            {
                "diagnostic_only": [True],
                "used_for_strategy": [False],
                "used_for_backtest": [False],
            }
        ),
        diagnostics_metadata={
            "diagnostic_only": True,
            "used_for_strategy": False,
            "used_for_backtest": False,
        },
    )

    failures = failed_error_checks(audit)
    assert "processed_panel_contains_no_strategy_or_backtest_columns" in failures["check_name"].tolist()


def test_run_no_lookahead_audit_writes_outputs(tmp_path: Path):
    config, _ = make_config(tmp_path)
    write_processed(tmp_path, make_processed_frame())
    write_metadata_files(tmp_path)

    result = n.run_msvol_no_lookahead_audit_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=False,
    )

    assert result.status == "ok"
    assert result.n_failed_error_checks == 0
    assert result.market_audit_csv.exists()
    assert result.combined_audit_csv.exists()

    audit = pd.read_csv(result.market_audit_csv)
    assert failed_error_checks(audit).empty

    combined = pd.read_csv(result.combined_audit_csv)
    assert "US" in combined["market"].tolist()


def test_run_no_lookahead_audit_allow_skip_when_processed_missing(tmp_path: Path):
    config, _ = make_config(tmp_path)

    result = n.run_msvol_no_lookahead_audit_for_market(
        market="US",
        config=config,
        project_root=tmp_path,
        allow_skip=True,
    )

    assert result.status == "failed"
    assert result.n_failed_error_checks > 0
    assert result.market_audit_csv.exists()

    audit = pd.read_csv(result.market_audit_csv)
    failures = failed_error_checks(audit)
    assert "processed_panel_non_empty" in failures["check_name"].tolist()