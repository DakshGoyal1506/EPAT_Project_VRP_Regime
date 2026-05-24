from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from vrp.backtest.backtest_config import load_backtest_config
from vrp.backtest.robustness import (
    run_cost_sensitivity_robustness,
    run_robustness_suite,
    run_subperiod_robustness,
    run_tradable_proxy_detection,
    write_weekly_rebalance_skip_report,
)
from vrp.backtest.tradable_proxy_detector import detect_tradable_proxy_data


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
            "default_cost_bps": 5.0,
            "apply_to_abs_exposure_change": True,
            "cost_formula": "abs(delta_exposure) * cost_bps / 10000",
        },
        "robustness": {
            "cost_bps_grid": [0, 5, 10],
            "rebalance_frequencies": ["daily", "weekly"],
            "subperiods": {
                "US": [["2020-01-01", "2020-12-31", "ToyWindow"]],
                "INDIA": [["2020-01-01", "2020-12-31", "ToyWindow"]],
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

    us_signals = signals.copy()
    us_outcomes = outcomes.copy()
    us_signals["market"] = "US"
    us_outcomes["market"] = "US"

    india_signals = signals.copy()
    india_outcomes = outcomes.copy()
    india_signals["market"] = "INDIA"
    india_outcomes["market"] = "INDIA"

    us_signals.to_parquet(processed / "us_strategy_signals.parquet", index=False)
    us_outcomes.to_parquet(processed / "us_vrp_har.parquet", index=False)
    us_outcomes.to_parquet(processed / "us_vrp.parquet", index=False)

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


def test_cost_sensitivity_robustness_writes_csv(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_cost_sensitivity_robustness(
        config=config,
        repo_root=tmp_path,
        market="US",
        force=True,
    )

    assert result.status == "completed"

    output_path = result.output_paths["cost_sensitivity"]
    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert not df.empty
    assert set(df["cost_bps"]) == {0.0, 5.0, 10.0}
    assert set(df["market"]) == {"US"}
    assert "total_return_proxy" in df.columns
    assert "return_per_abs_exposure" in df.columns


def test_subperiod_robustness_writes_csv(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_subperiod_robustness(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        force=True,
    )

    assert result.status == "completed"

    output_path = result.output_paths["subperiod"]
    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert not df.empty
    assert set(df["market"]) == {"US", "INDIA"}
    assert set(df["subperiod"]) == {"ToyWindow"}
    assert set(df["strategy_name"]) == set(config.strategy_universe)
    assert "total_return_proxy" in df.columns
    assert "uses_overlapping_forward_labels" in df.columns


def test_weekly_rebalance_skip_report(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = write_weekly_rebalance_skip_report(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        force=True,
    )

    assert result.status == "skipped"

    output_path = result.output_paths["weekly_rebalance_skip"]
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "skipped"
    assert payload["robustness_test"] == "weekly_rebalance"
    assert "carry exposure only across eligible rows" in payload["reason"]


def test_tradable_proxy_detector_skips_when_no_files(tmp_path: Path) -> None:
    result = detect_tradable_proxy_data(tmp_path)

    assert result.status == "skipped"
    assert result.n_candidates == 0


def test_tradable_proxy_detector_finds_existing_candidate(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    candidate = processed / "us_vix_futures_proxy.parquet"
    pd.DataFrame({"date": ["2020-01-01"], "close": [1.0]}).to_parquet(
        candidate,
        index=False,
    )

    result = detect_tradable_proxy_data(tmp_path)

    assert result.status == "available"
    assert result.n_candidates >= 1
    assert any("vix_futures" in item.matched_token for item in result.candidates)


def test_run_tradable_proxy_detection_writes_report(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_tradable_proxy_detection(
        config=config,
        repo_root=tmp_path,
        force=True,
    )

    assert result.status == "skipped"

    output_path = result.output_paths["tradable_proxy_detection"]
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "skipped"
    assert "does not download" in payload["reason"]


def test_run_full_robustness_suite_writes_metadata(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    results = run_robustness_suite(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        test="all",
        force=True,
    )

    assert {result.test_name for result in results} == {
        "costs",
        "subperiod",
        "weekly",
        "tradable_proxy",
    }

    metadata_path = tmp_path / "reports" / "tables" / "phase_10" / "robustness_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["phase"] == "phase_10"
    assert metadata["rules"]["no_new_data_downloads"] is True
    assert metadata["rules"]["tradable_proxy_detection_only"] is True
    assert metadata["rules"]["weekly_rebalance_skip_safe_by_default"] is True