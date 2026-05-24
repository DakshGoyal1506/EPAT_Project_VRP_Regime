from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.backtest.backtest_config import load_backtest_config
from vrp.backtest.vectorized_engine import (
    VectorizedBacktestError,
    resolve_markets,
    run_market_backtest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _base_config_dict() -> dict:
    return {
        "backtest_phase": "phase_10",
        "description": "test config",
        "input_files": {
            "US": {
                "strategy_signals": "data/processed/us_strategy_signals.parquet",
                "vrp_har": "data/processed/us_vrp_har.parquet",
                "vrp": "data/processed/us_vrp.parquet",
                "threshold": "data/processed/us_threshold_regimes.parquet",
                "hmm": "data/processed/us_hmm_regimes.parquet",
                "mar": "data/processed/us_markov_autoreg_regimes.parquet",
            },
            "INDIA": {
                "strategy_signals": "data/processed/india_strategy_signals.parquet",
                "vrp_har": "data/processed/india_vrp_har.parquet",
                "vrp": "data/processed/india_vrp.parquet",
                "threshold": "data/processed/india_threshold_regimes.parquet",
                "hmm": "data/processed/india_hmm_regimes.parquet",
                "mar": "data/processed/india_markov_autoreg_regimes.parquet",
            },
        },
        "strategy_universe": [
            "unconditional_full",
            "threshold_hard_filter",
            "threshold_defensive",
            "hmm_prob_linear",
            "hmm_prob_linear_carry",
            "mar_prob_linear",
            "mar_prob_linear_carry",
        ],
        "primary_payoff": {
            "name": "forward_vrp_research_proxy",
            "label_col": "vrp_forward_expost_gk_label",
            "label_role": "realised_outcome_only",
            "outcome_alignment": "signal_observation_date",
            "payoff_formula": "-target_exposure * label",
            "annualization_periods": 252,
            "horizon_trading_days": 22,
            "allow_horizon_override": False,
            "overlapping_labels": True,
            "report_as_research_proxy": True,
        },
        "costs": {
            "enabled": True,
            "default_cost_bps": 5,
            "apply_to_abs_exposure_change": True,
            "cost_formula": "abs(delta_exposure) * cost_bps / 10000",
        },
        "robustness": {
            "cost_bps_grid": [0, 2.5, 5, 10, 20],
            "rebalance_frequencies": ["daily", "weekly"],
            "subperiods": {
                "US": [["2020-02-01", "2020-06-30", "COVID"]],
                "INDIA": [["2020-02-01", "2020-06-30", "COVID"]],
            },
        },
        "output_files": {
            "US": {
                "backtest_panel": "data/processed/us_backtest_panel.parquet",
                "metadata": "data/processed/us_backtest_panel_metadata.json",
            },
            "INDIA": {
                "backtest_panel": "data/processed/india_backtest_panel.parquet",
                "metadata": "data/processed/india_backtest_panel_metadata.json",
            },
        },
        "reporting": {
            "table_dir": "reports/tables/phase_10",
            "figure_dir": "reports/figures/phase_10",
        },
    }


def _prepare_tmp_repo(tmp_path: Path) -> Path:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    outcomes = pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")

    signals.to_parquet(processed / "us_strategy_signals.parquet", index=False)
    outcomes.to_parquet(processed / "us_vrp_har.parquet", index=False)
    outcomes.to_parquet(processed / "us_vrp.parquet", index=False)

    # Same fixture duplicated for INDIA to satisfy ALL-style config validation when needed.
    india_signals = signals.copy()
    india_signals["market"] = "INDIA"
    india_outcomes = outcomes.copy()
    india_outcomes["market"] = "INDIA"

    india_signals.to_parquet(processed / "india_strategy_signals.parquet", index=False)
    india_outcomes.to_parquet(processed / "india_vrp_har.parquet", index=False)
    india_outcomes.to_parquet(processed / "india_vrp.parquet", index=False)

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "backtest.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config_dict(), sort_keys=False),
        encoding="utf-8",
    )

    return config_path


def test_resolve_markets() -> None:
    assert resolve_markets("US") == ["US"]
    assert resolve_markets("INDIA") == ["INDIA"]
    assert resolve_markets("ALL") == ["US", "INDIA"]

    with pytest.raises(VectorizedBacktestError):
        resolve_markets("EU")


def test_run_market_backtest_writes_panel_and_metadata(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=5.0,
        force=False,
        write=True,
    )

    assert result.market == "US"
    assert result.strategy == "all"
    assert result.n_rows == 9
    assert result.n_eligible == 9
    assert result.n_strategies == 7
    assert result.wrote_files is True

    assert result.output_path.exists()
    assert result.metadata_path.exists()

    panel = pd.read_parquet(result.output_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    required_cols = {
        "market",
        "strategy_name",
        "signal_observation_date",
        "target_trade_date",
        "outcome_label_date",
        "target_exposure",
        "target_exposure_for_backtest",
        "gross_return_proxy",
        "delta_exposure",
        "cost_proxy",
        "net_return_proxy",
        "is_backtest_eligible",
        "exclusion_reason",
    }

    assert required_cols.issubset(panel.columns)

    assert metadata["phase"] == "phase_10"
    assert metadata["market"] == "US"
    assert metadata["payoff_type"] == "forward_vrp_research_proxy"
    assert metadata["payoff_label"] == "vrp_forward_expost_gk_label"
    assert metadata["label_role"] == "realised_outcome_only"
    assert metadata["outcome_alignment"] == "signal_observation_date"
    assert metadata["cost_bps"] == 5.0
    assert metadata["annualization_periods"] == 252
    assert metadata["horizon_trading_days"] == 22
    assert metadata["overlapping_labels"] is True
    assert metadata["research_proxy_not_trade_pnl"] is True
    assert metadata["strategy_universe_locked"] is True
    assert metadata["n_target_not_after_signal_violations"] == 0
    assert metadata["n_outcome_not_equal_signal_date_violations"] == 0


def test_run_market_backtest_rejects_existing_outputs_without_force(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=5.0,
        force=False,
        write=True,
    )

    with pytest.raises(VectorizedBacktestError):
        run_market_backtest(
            market="US",
            config=config,
            repo_root=tmp_path,
            strategy="all",
            cost_bps=5.0,
            force=False,
            write=True,
        )


def test_run_market_backtest_allows_force_overwrite(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    first = run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=5.0,
        force=False,
        write=True,
    )

    second = run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=10.0,
        force=True,
        write=True,
    )

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))

    assert first.output_path == second.output_path
    assert metadata["cost_bps"] == 10.0


def test_run_market_backtest_single_strategy_filter(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="unconditional_full",
        cost_bps=5.0,
        force=False,
        write=False,
    )

    assert result.n_rows == 3
    assert result.n_eligible == 3
    assert result.n_strategies == 1
    assert result.wrote_files is False
    assert not result.output_path.exists()
    assert not result.metadata_path.exists()


def test_run_market_backtest_rejects_writing_single_strategy_to_canonical_output(
    tmp_path: Path,
) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    with pytest.raises(VectorizedBacktestError):
        run_market_backtest(
            market="US",
            config=config,
            repo_root=tmp_path,
            strategy="unconditional_full",
            cost_bps=5.0,
            force=True,
            write=True,
        )


def test_run_market_backtest_dry_run_does_not_write(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_market_backtest(
        market="US",
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=5.0,
        force=False,
        write=False,
    )

    assert result.wrote_files is False
    assert result.n_rows == 9
    assert not result.output_path.exists()
    assert not result.metadata_path.exists()


def test_run_market_backtest_rejects_bad_strategy(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    with pytest.raises(VectorizedBacktestError):
        run_market_backtest(
            market="US",
            config=config,
            repo_root=tmp_path,
            strategy="bad_strategy",
            cost_bps=5.0,
            force=False,
            write=False,
        )


def test_run_market_backtest_rejects_negative_cost(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    with pytest.raises(VectorizedBacktestError):
        run_market_backtest(
            market="US",
            config=config,
            repo_root=tmp_path,
            strategy="all",
            cost_bps=-1.0,
            force=False,
            write=False,
        )