from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vrp.reports.strategy_diagnostics import (
    DEFAULT_MSVOL_POLICY,
    DEFAULT_REPORT_NOTE,
    assert_diagnostics_are_phase9_only,
    build_phase9_metadata,
    create_all_phase9_diagnostics,
    create_no_lookahead_audit_table,
    create_strategy_blocked_reason_summary,
    create_strategy_exposure_by_year,
    create_strategy_exposure_change_summary,
    create_strategy_signal_summary,
    file_sha256,
    write_diagnostic_tables,
    write_metadata_json,
)
from vrp.strategies.signal_schema import PHASE9_OUTPUT_COLUMNS


def _sample_signal_panel() -> pd.DataFrame:
    rows = [
        {
            "market": "US",
            "strategy_name": "unconditional_full",
            "regime_model": "unconditional",
            "signal_observation_date": pd.Timestamp("2020-01-01"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-01"),
            "target_trade_date": pd.Timestamp("2020-01-02"),
            "target_exposure": -1.0,
            "strategy_available": True,
            "blocked_reason": "none",
            "decision_reason": "unconditional_full",
            "state_name": pd.NA,
            "p_calm": pd.NA,
            "p_transition": pd.NA,
            "p_stress": pd.NA,
            "vrp_har_gk": 0.02,
            "har_forecast_available": True,
            "source_signal_date_column": "date",
            "source_model": "unconditional",
        },
        {
            "market": "US",
            "strategy_name": "unconditional_full",
            "regime_model": "unconditional",
            "signal_observation_date": pd.Timestamp("2020-01-02"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-02"),
            "target_trade_date": pd.Timestamp("2020-01-03"),
            "target_exposure": -1.0,
            "strategy_available": True,
            "blocked_reason": "none",
            "decision_reason": "unconditional_full",
            "state_name": pd.NA,
            "p_calm": pd.NA,
            "p_transition": pd.NA,
            "p_stress": pd.NA,
            "vrp_har_gk": -0.01,
            "har_forecast_available": True,
            "source_signal_date_column": "date",
            "source_model": "unconditional",
        },
        {
            "market": "US",
            "strategy_name": "hmm_prob_linear_carry",
            "regime_model": "gaussian_hmm",
            "signal_observation_date": pd.Timestamp("2020-01-01"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-01"),
            "target_trade_date": pd.Timestamp("2020-01-02"),
            "target_exposure": -0.60,
            "strategy_available": True,
            "blocked_reason": "none",
            "decision_reason": "probability_linear_carry",
            "state_name": "calm",
            "p_calm": 0.70,
            "p_transition": 0.20,
            "p_stress": 0.10,
            "vrp_har_gk": 0.02,
            "har_forecast_available": True,
            "source_signal_date_column": "hmm_signal_observation_date",
            "source_model": "gaussian_hmm",
        },
        {
            "market": "US",
            "strategy_name": "hmm_prob_linear_carry",
            "regime_model": "gaussian_hmm",
            "signal_observation_date": pd.Timestamp("2020-01-02"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-02"),
            "target_trade_date": pd.Timestamp("2020-01-03"),
            "target_exposure": 0.0,
            "strategy_available": True,
            "blocked_reason": "none",
            "decision_reason": "negative_or_zero_vrp_har",
            "state_name": "calm",
            "p_calm": 0.70,
            "p_transition": 0.10,
            "p_stress": 0.20,
            "vrp_har_gk": -0.01,
            "har_forecast_available": True,
            "source_signal_date_column": "hmm_signal_observation_date",
            "source_model": "gaussian_hmm",
        },
        {
            "market": "US",
            "strategy_name": "hmm_prob_linear_carry",
            "regime_model": "gaussian_hmm",
            "signal_observation_date": pd.Timestamp("2020-01-03"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-03"),
            "target_trade_date": pd.Timestamp("2020-01-06"),
            "target_exposure": 0.0,
            "strategy_available": True,
            "blocked_reason": "none",
            "decision_reason": "stress_probability_veto",
            "state_name": "stress",
            "p_calm": 0.50,
            "p_transition": 0.09,
            "p_stress": 0.41,
            "vrp_har_gk": 0.02,
            "har_forecast_available": True,
            "source_signal_date_column": "hmm_signal_observation_date",
            "source_model": "gaussian_hmm",
        },
        {
            "market": "US",
            "strategy_name": "hmm_prob_linear_carry",
            "regime_model": "gaussian_hmm",
            "signal_observation_date": pd.Timestamp("2020-01-06"),
            "signal_available_after_close_date": pd.Timestamp("2020-01-06"),
            "target_trade_date": pd.NaT,
            "target_exposure": pd.NA,
            "strategy_available": False,
            "blocked_reason": "missing_target_trade_date",
            "decision_reason": "unavailable",
            "state_name": "calm",
            "p_calm": 0.70,
            "p_transition": 0.20,
            "p_stress": 0.10,
            "vrp_har_gk": 0.02,
            "har_forecast_available": True,
            "source_signal_date_column": "hmm_signal_observation_date",
            "source_model": "gaussian_hmm",
        },
    ]

    return pd.DataFrame(rows, columns=list(PHASE9_OUTPUT_COLUMNS))


def _present_but_excluded() -> dict[str, tuple[str, ...]]:
    return {
        "har": ("vrp_forward_expost_gk_label",),
        "gaussian_hmm": ("hmm_smoothed_prob_stress",),
        "markov_autoreg": (),
        "threshold": ("crisis_window_label",),
    }


def test_signal_summary_has_allowed_columns_only() -> None:
    summary = create_strategy_signal_summary(_sample_signal_panel())

    assert "strategy_name" in summary.columns
    assert "available_count" in summary.columns
    assert "mean_target_exposure" in summary.columns

    forbidden_tokens = ["return", "pnl", "sharpe", "drawdown", "transaction_cost"]
    for column in summary.columns:
        lowered = column.lower()
        assert not any(token in lowered for token in forbidden_tokens)


def test_signal_summary_counts_valid_flat_as_available() -> None:
    summary = create_strategy_signal_summary(_sample_signal_panel())
    row = summary[
        summary["strategy_name"] == "hmm_prob_linear_carry"
    ].iloc[0]

    assert row["row_count"] == 4
    assert row["available_count"] == 3
    assert row["unavailable_count"] == 1
    assert row["flat_count"] == 2
    assert row["partial_short_vol_count"] == 1


def test_exposure_by_year_created_without_performance_fields() -> None:
    table = create_strategy_exposure_by_year(_sample_signal_panel())

    assert "target_year" in table.columns
    assert "mean_target_exposure" in table.columns

    for column in table.columns:
        lowered = column.lower()
        assert "return" not in lowered
        assert "pnl" not in lowered
        assert "sharpe" not in lowered
        assert "drawdown" not in lowered


def test_exposure_change_summary_uses_safe_name_logic() -> None:
    table = create_strategy_exposure_change_summary(_sample_signal_panel())

    assert "absolute_exposure_change_count" in table.columns
    assert "absolute_exposure_change_sum" in table.columns

    for column in table.columns:
        assert "turnover" not in column.lower()
        assert "cost" not in column.lower()


def test_exposure_change_summary_counts_changes() -> None:
    table = create_strategy_exposure_change_summary(_sample_signal_panel())
    row = table[
        table["strategy_name"] == "hmm_prob_linear_carry"
    ].iloc[0]

    assert row["available_row_count"] == 3
    assert row["exposure_change_observation_count"] == 2
    assert row["absolute_exposure_change_count"] == 1
    assert row["absolute_exposure_change_sum"] == pytest.approx(0.60)


def test_blocked_reason_summary_contains_blocked_and_decision_reasons() -> None:
    table = create_strategy_blocked_reason_summary(_sample_signal_panel())

    assert set(table["summary_type"].unique()) == {
        "blocked_reason",
        "decision_reason",
    }

    reasons = set(table["reason"].astype(str))
    assert "missing_target_trade_date" in reasons
    assert "negative_or_zero_vrp_har" in reasons
    assert "stress_probability_veto" in reasons


def test_no_lookahead_audit_table_records_excluded_columns() -> None:
    table = create_no_lookahead_audit_table(
        present_but_excluded=_present_but_excluded(),
        forbidden_columns_used=[],
    )

    assert "vrp_forward_expost_gk_label" in set(table["column_name"])
    assert "hmm_smoothed_prob_stress" in set(table["column_name"])
    assert "crisis_window_label" in set(table["column_name"])
    assert table["used_by_strategy"].eq(False).all()


def test_all_diagnostics_created() -> None:
    tables = create_all_phase9_diagnostics(
        signal_panel=_sample_signal_panel(),
        present_but_excluded=_present_but_excluded(),
        forbidden_columns_used=[],
    )

    assert set(tables) == {
        "strategy_signal_summary",
        "strategy_exposure_by_year",
        "strategy_exposure_change_summary",
        "strategy_blocked_reason_summary",
        "strategy_no_lookahead_audit",
    }


def test_diagnostics_do_not_include_returns_pnl_sharpe_drawdown() -> None:
    tables = create_all_phase9_diagnostics(
        signal_panel=_sample_signal_panel(),
        present_but_excluded=_present_but_excluded(),
        forbidden_columns_used=[],
    )

    forbidden_tokens = [
        "return",
        "pnl",
        "profit",
        "sharpe",
        "drawdown",
        "transaction_cost",
        "performance",
        "backtest",
    ]

    for table in tables.values():
        for column in table.columns:
            lowered = column.lower()
            assert not any(token in lowered for token in forbidden_tokens)


def test_assert_diagnostics_rejects_performance_column() -> None:
    bad = pd.DataFrame({"strategy_name": ["x"], "sharpe_ratio": [1.2]})

    with pytest.raises(ValueError):
        assert_diagnostics_are_phase9_only({"bad_table": bad})


def test_metadata_contains_required_fields(tmp_path: Path) -> None:
    input_file = tmp_path / "us_vrp_har.parquet"
    input_file.write_bytes(b"test-input")

    metadata = build_phase9_metadata(
        market="US",
        strategy_config_hash="abc123",
        input_file_paths={"har": input_file},
        input_file_hashes=None,
        row_counts_by_input={"har": 10},
        strategy_names=["unconditional_full", "hmm_prob_linear"],
        forbidden_columns_present_but_excluded=_present_but_excluded(),
        forbidden_columns_used=[],
        msvol_policy=DEFAULT_MSVOL_POLICY,
        timing_policy={
            "threshold_shift_to_next_trade_date": True,
            "do_not_double_shift_hmm_or_mar": True,
        },
        exposure_bounds={"min_exposure": -1.0, "max_exposure": 0.0},
        report_note=DEFAULT_REPORT_NOTE,
        run_timestamp="2026-01-01T00:00:00+00:00",
    )

    assert metadata["phase"] == 9
    assert metadata["market"] == "US"
    assert metadata["strategy_config_hash"] == "abc123"
    assert metadata["input_file_paths"]["har"] == str(input_file)
    assert metadata["input_file_hashes"]["har"] == file_sha256(input_file)
    assert metadata["row_counts_by_input"]["har"] == 10
    assert metadata["forbidden_columns_used"] == []
    assert metadata["msvol_policy"] == "excluded_diagnostic_only"
    assert "Phase 9 defines ex-ante exposure intentions only" in metadata[
        "phase_9_report_note"
    ]


def test_metadata_rejects_forbidden_columns_used() -> None:
    with pytest.raises(ValueError):
        build_phase9_metadata(
            market="US",
            strategy_config_hash="abc123",
            input_file_paths={},
            input_file_hashes={},
            row_counts_by_input={},
            strategy_names=["unconditional_full"],
            forbidden_columns_present_but_excluded={},
            forbidden_columns_used=["vrp_forward_expost_gk_label"],
            timing_policy={},
            exposure_bounds={"min_exposure": -1.0, "max_exposure": 0.0},
        )


def test_metadata_records_msvol_policy_excluded() -> None:
    metadata = build_phase9_metadata(
        market="INDIA",
        strategy_config_hash="abc123",
        input_file_paths={},
        input_file_hashes={},
        row_counts_by_input={},
        strategy_names=["unconditional_full"],
        forbidden_columns_present_but_excluded={},
        forbidden_columns_used=[],
        timing_policy={},
        exposure_bounds={"min_exposure": -1.0, "max_exposure": 0.0},
    )

    assert metadata["msvol_policy"] == "excluded_diagnostic_only"


def test_write_diagnostic_tables(tmp_path: Path) -> None:
    tables = create_all_phase9_diagnostics(
        signal_panel=_sample_signal_panel(),
        present_but_excluded=_present_but_excluded(),
        forbidden_columns_used=[],
    )

    output_paths = {
        "signal_summary": tmp_path / "strategy_signal_summary.csv",
        "exposure_by_year": tmp_path / "strategy_exposure_by_year.csv",
        "exposure_change_summary": tmp_path / "strategy_exposure_change_summary.csv",
        "blocked_reason_summary": tmp_path / "strategy_blocked_reason_summary.csv",
        "no_lookahead_audit": tmp_path / "strategy_no_lookahead_audit.csv",
    }

    written = write_diagnostic_tables(tables=tables, output_paths=output_paths)

    assert set(written) == set(tables)
    for path in written.values():
        assert path.exists()
        assert path.suffix == ".csv"

    assert (tmp_path / "strategy_exposure_change_summary.csv").exists()
    assert not (tmp_path / "strategy_turnover_preview.csv").exists()


def test_write_metadata_json(tmp_path: Path) -> None:
    metadata = build_phase9_metadata(
        market="US",
        strategy_config_hash="abc123",
        input_file_paths={},
        input_file_hashes={},
        row_counts_by_input={},
        strategy_names=["unconditional_full"],
        forbidden_columns_present_but_excluded={},
        forbidden_columns_used=[],
        timing_policy={},
        exposure_bounds={"min_exposure": -1.0, "max_exposure": 0.0},
        run_timestamp="2026-01-01T00:00:00+00:00",
    )

    output_path = tmp_path / "strategy_metadata.json"
    written = write_metadata_json(metadata=metadata, output_path=output_path)

    assert written.exists()

    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["market"] == "US"
    assert loaded["forbidden_columns_used"] == []


def test_file_sha256_missing_file_returns_none(tmp_path: Path) -> None:
    assert file_sha256(tmp_path / "missing.txt") is None


def test_file_sha256_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("abc", encoding="utf-8")

    assert file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )