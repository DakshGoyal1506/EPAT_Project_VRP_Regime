from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vrp.reports.cross_market import (
    CrossMarketInputError,
    CrossMarketMutationError,
    collect_locked_artifact_hashes,
    guarded_read_csv,
    guarded_read_parquet,
    hash_locked_artifacts_before_after,
)


def _mutation_config() -> dict:
    return {
        "locked_artifacts": {
            "phase9": [
                "data/processed/us_strategy_signals.parquet",
                "data/processed/india_strategy_signals.parquet",
            ],
            "phase10": [
                "data/processed/us_backtest_panel.parquet",
                "data/processed/india_backtest_panel.parquet",
                "reports/tables/phase_10/backtest_summary.csv",
                "reports/tables/phase_10/backtest_metadata.json",
            ],
        },
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
    }


def _write_locked_artifacts(root: Path) -> None:
    processed = root / "data" / "processed"
    phase10 = root / "reports" / "tables" / "phase_10"

    processed.mkdir(parents=True, exist_ok=True)
    phase10.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "value": [1.0, 2.0, 3.0],
        }
    )

    df.to_parquet(processed / "us_strategy_signals.parquet", index=False)
    df.to_parquet(processed / "india_strategy_signals.parquet", index=False)
    df.to_parquet(processed / "us_backtest_panel.parquet", index=False)
    df.to_parquet(processed / "india_backtest_panel.parquet", index=False)

    (phase10 / "backtest_summary.csv").write_text(
        "metric,value\nsharpe,1.0\n",
        encoding="utf-8",
    )
    (phase10 / "backtest_metadata.json").write_text(
        '{"phase": "phase_10"}\n',
        encoding="utf-8",
    )


def test_locked_artifact_hashes_pass_when_unchanged(tmp_path) -> None:
    cfg = _mutation_config()
    _write_locked_artifacts(tmp_path)

    before = collect_locked_artifact_hashes(cfg, root=tmp_path)
    after = collect_locked_artifact_hashes(cfg, root=tmp_path)

    hash_locked_artifacts_before_after(before, after)


def test_locked_artifact_hashes_fail_when_phase9_file_changes(tmp_path) -> None:
    cfg = _mutation_config()
    _write_locked_artifacts(tmp_path)

    before = collect_locked_artifact_hashes(cfg, root=tmp_path)

    changed = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    changed.to_parquet(
        tmp_path / "data" / "processed" / "india_strategy_signals.parquet",
        index=False,
    )

    after = collect_locked_artifact_hashes(cfg, root=tmp_path)

    with pytest.raises(CrossMarketMutationError):
        hash_locked_artifacts_before_after(before, after)


def test_locked_artifact_hashes_fail_when_phase10_file_changes(tmp_path) -> None:
    cfg = _mutation_config()
    _write_locked_artifacts(tmp_path)

    before = collect_locked_artifact_hashes(cfg, root=tmp_path)

    path = tmp_path / "reports" / "tables" / "phase_10" / "backtest_summary.csv"
    path.write_text("metric,value\nsharpe,2.0\n", encoding="utf-8")

    after = collect_locked_artifact_hashes(cfg, root=tmp_path)

    with pytest.raises(CrossMarketMutationError):
        hash_locked_artifacts_before_after(before, after)


def test_locked_artifact_hashes_fail_when_existing_file_is_deleted(tmp_path) -> None:
    cfg = _mutation_config()
    _write_locked_artifacts(tmp_path)

    before = collect_locked_artifact_hashes(cfg, root=tmp_path)

    path = tmp_path / "data" / "processed" / "us_backtest_panel.parquet"
    path.unlink()

    after = collect_locked_artifact_hashes(cfg, root=tmp_path)

    with pytest.raises(CrossMarketMutationError):
        hash_locked_artifacts_before_after(before, after)


def test_missing_locked_artifacts_are_stable_if_missing_before_and_after(tmp_path) -> None:
    cfg = _mutation_config()

    before = collect_locked_artifact_hashes(cfg, root=tmp_path)
    after = collect_locked_artifact_hashes(cfg, root=tmp_path)

    hash_locked_artifacts_before_after(before, after)


def test_guarded_read_parquet_rejects_forbidden_phase11_path_before_file_check(tmp_path) -> None:
    cfg = _mutation_config()

    with pytest.raises(CrossMarketInputError):
        guarded_read_parquet(
            "reports/tables/phase_11/daily_paper_signal.csv",
            config=cfg,
            root=tmp_path,
        )


def test_guarded_read_csv_rejects_forbidden_phase11_path_before_file_check(tmp_path) -> None:
    cfg = _mutation_config()

    with pytest.raises(CrossMarketInputError):
        guarded_read_csv(
            "reports/tables/phase_11/paper_order_intents.csv",
            config=cfg,
            root=tmp_path,
        )


def test_guarded_read_rejects_broker_like_keyword(tmp_path) -> None:
    cfg = _mutation_config()

    with pytest.raises(CrossMarketInputError):
        guarded_read_parquet(
            "reports/tables/phase_11/broker_signal.parquet",
            config=cfg,
            root=tmp_path,
        )