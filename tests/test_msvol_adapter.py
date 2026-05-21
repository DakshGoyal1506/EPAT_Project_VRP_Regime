from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vrp.regimes import msvol_adapter as a


def make_config(tmp_path: Path) -> tuple[dict, Path]:
    config = {
        "model_name": "msvol_appendix_v1",
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "optional_phase": True,
        "required_for_main_pipeline": False,
        "markets": {
            "US": {
                "input_csv": "data/interim/msgarch/us_msgarch_input.csv",
                "raw_output_csv": "data/interim/msvol/us_msvol_raw_output.csv",
                "preflight_json": "data/interim/msvol/us_msvol_preflight.json",
                "skip_report_json": "data/interim/msvol/us_msvol_skip_report.json",
                "model_summary_json": "data/interim/msvol/us_msvol_model_summary.json",
            }
        },
        "state_policy": {
            "transition_state_modelled": False,
            "transition_probability_for_valid_rows": 0.0,
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


def make_raw_df(
    state0_var: float = 1.0,
    state1_var: float = 4.0,
    include_smoothed: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B").strftime("%Y-%m-%d"),
            "market": ["US", "US", "US", "US"],
            "msvol_raw_state_0_prob_filtered": [0.9, 0.7, 0.4, 0.2],
            "msvol_raw_state_1_prob_filtered": [0.1, 0.3, 0.6, 0.8],
            "msvol_raw_state_0_variance_estimate": [state0_var] * 4,
            "msvol_raw_state_1_variance_estimate": [state1_var] * 4,
            "msvol_conditional_variance": [1.3, 1.9, 2.8, 3.4],
            "msvol_conditional_volatility": np.sqrt([1.3, 1.9, 2.8, 3.4]),
            "msvol_model_valid": [True, True, True, True],
            "msvol_fit_status": ["ok", "ok", "ok", "ok"],
            "msvol_skip_reason": ["", "", "", ""],
        }
    )

    if include_smoothed:
        df["msvol_raw_state_0_prob_smoothed_diagnostic"] = [0.85, 0.65, 0.35, 0.25]
        df["msvol_raw_state_1_prob_smoothed_diagnostic"] = [0.15, 0.35, 0.65, 0.75]

    return df


def write_raw_output(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "data" / "interim" / "msvol" / "us_msvol_raw_output.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_validate_probability_rows_accepts_valid_raw():
    df = make_raw_df()
    a.validate_msvol_raw_schema(df, expected_market="US")
    a.validate_msvol_probability_rows(df)


def test_validate_probability_rows_rejects_bad_sum():
    df = make_raw_df()
    df.loc[0, "msvol_raw_state_1_prob_filtered"] = 0.2

    with pytest.raises(a.MSVolAdapterError, match="not summing"):
        a.validate_msvol_probability_rows(df)


def test_map_state0_low_variance_to_calm():
    df = make_raw_df(state0_var=1.0, state1_var=4.0)

    mapping = a.map_msvol_states_by_variance(df)

    assert mapping.calm_raw_state == 0
    assert mapping.stress_raw_state == 1
    assert mapping.lower_variance_raw_state == 0
    assert mapping.higher_variance_raw_state == 1


def test_map_state1_low_variance_to_calm():
    df = make_raw_df(state0_var=5.0, state1_var=2.0)

    mapping = a.map_msvol_states_by_variance(df)

    assert mapping.calm_raw_state == 1
    assert mapping.stress_raw_state == 0
    assert mapping.lower_variance_raw_state == 1
    assert mapping.higher_variance_raw_state == 0


def test_standardize_output_timing_and_schema():
    config, _ = make_config(Path("/tmp"))
    df = make_raw_df(state0_var=1.0, state1_var=4.0)

    processed, mapping = a.standardize_msvol_output(df, market="US", config=config)

    assert mapping.calm_raw_state == 0
    assert mapping.stress_raw_state == 1

    for col in a.PROCESSED_COLUMNS:
        assert col in processed.columns

    assert len(processed) == 4

    assert processed.loc[0, "msvol_signal_observation_date"] == pd.Timestamp("2020-01-01")
    assert processed.loc[0, "msvol_signal_available_after_close_date"] == pd.Timestamp("2020-01-01")
    assert processed.loc[0, "msvol_signal_trade_date"] == pd.Timestamp("2020-01-02")

    assert pd.isna(processed.loc[3, "msvol_signal_trade_date"])

    assert processed.loc[0, "msvol_state_name_for_next_session"] == "calm"
    assert processed.loc[0, "msvol_state_for_next_session"] == 0
    assert processed.loc[3, "msvol_state_name_for_next_session"] == "stress"
    assert processed.loc[3, "msvol_state_for_next_session"] == 2

    prob_sum = (
        processed["msvol_filtered_prob_calm_for_next_session"]
        + processed["msvol_filtered_prob_transition_for_next_session"]
        + processed["msvol_filtered_prob_stress_for_next_session"]
    )

    assert np.allclose(prob_sum, 1.0)
    assert processed["msvol_filtered_prob_transition_for_next_session"].eq(0.0).all()
    assert processed["msvol_transition_state_modelled"].eq(False).all()


def test_smoothed_probabilities_remain_diagnostic_only():
    config, _ = make_config(Path("/tmp"))
    df = make_raw_df(include_smoothed=True)

    processed, _ = a.standardize_msvol_output(df, market="US", config=config)

    smoothed_cols = [col for col in processed.columns if "smoothed" in col.lower()]
    assert smoothed_cols
    assert all("diagnostic" in col.lower() for col in smoothed_cols)
    assert not any("for_next_session" in col.lower() for col in smoothed_cols)


def test_import_outputs_writes_parquet_audit_and_metadata(tmp_path: Path):
    config, config_path = make_config(tmp_path)
    write_raw_output(tmp_path, make_raw_df())

    result = a.import_msvol_outputs_for_market(
        market="US",
        config=config,
        config_path=config_path,
        project_root=tmp_path,
        allow_skip=False,
    )

    assert result.status == "ok"
    assert result.processed_output_parquet.exists()
    assert result.metadata_json.exists()
    assert result.probability_audit_csv.exists()

    processed = pd.read_parquet(result.processed_output_parquet)
    assert len(processed) == 4
    assert "msvol_filtered_prob_calm_for_next_session" in processed.columns

    audit = pd.read_csv(result.probability_audit_csv)
    assert audit.loc[0, "validation_status"] == "ok"
    assert audit.loc[0, "n_rows"] == 4
    assert audit.loc[0, "n_missing_signal_trade_date"] == 1

    with result.metadata_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["true_msgarch"] is False
    assert metadata["used_for_strategy"] is False
    assert metadata["used_for_backtest"] is False
    assert metadata["status"] == "ok"
    assert metadata["calm_raw_state"] == 0
    assert metadata["stress_raw_state"] == 1


def test_import_allow_skip_writes_empty_processed_output(tmp_path: Path):
    config, config_path = make_config(tmp_path)

    skip_path = tmp_path / "data" / "interim" / "msvol" / "us_msvol_skip_report.json"
    skip_path.parent.mkdir(parents=True, exist_ok=True)

    with skip_path.open("w", encoding="utf-8") as f:
        json.dump({"skip_reason": "synthetic skip"}, f)

    result = a.import_msvol_outputs_for_market(
        market="US",
        config=config,
        config_path=config_path,
        project_root=tmp_path,
        allow_skip=True,
    )

    assert result.status == "skipped"
    assert result.processed_output_parquet.exists()
    assert result.metadata_json.exists()
    assert result.probability_audit_csv.exists()

    processed = pd.read_parquet(result.processed_output_parquet)
    assert len(processed) == 0

    audit = pd.read_csv(result.probability_audit_csv)
    assert audit.loc[0, "validation_status"] == "skipped"
    assert audit.loc[0, "skip_reason"] == "synthetic skip"

    with result.metadata_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["status"] == "skipped"
    assert metadata["skip_reason"] == "synthetic skip"
    assert metadata["true_msgarch"] is False