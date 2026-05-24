from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from vrp.backtest.backtest_config import load_backtest_config
from vrp.backtest.final_audit import (
    assert_final_audit_passed,
    run_phase10_final_audit,
    write_final_audit_report,
)
from vrp.backtest.robustness import run_robustness_suite
from vrp.backtest.vectorized_engine import run_backtests
from vrp.reports.backtest_diagnostics import generate_backtest_diagnostics


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


def test_full_phase10_pipeline_passes_final_audit(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    run_backtests(
        markets=["US", "INDIA"],
        config=config,
        repo_root=tmp_path,
        strategy="all",
        cost_bps=5.0,
        force=True,
        write=True,
    )

    generate_backtest_diagnostics(
        config=config,
        repo_root=tmp_path,
        market="ALL",
    )

    run_robustness_suite(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        test="all",
        force=True,
    )

    result = run_phase10_final_audit(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        require_robustness=True,
    )

    assert result.status == "passed"
    assert result.n_errors == 0

    output_path = tmp_path / "reports" / "tables" / "phase_10" / "phase10_final_audit.json"
    write_final_audit_report(result, output_path)

    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["n_errors"] == 0

    assert_final_audit_passed(result)


def test_final_audit_fails_when_required_output_missing(tmp_path: Path) -> None:
    config_path = _prepare_tmp_repo(tmp_path)
    config = load_backtest_config(config_path)

    result = run_phase10_final_audit(
        config=config,
        repo_root=tmp_path,
        market="ALL",
        require_robustness=True,
    )

    assert result.status == "failed"
    assert result.n_errors > 0
    assert any("Missing panel" in issue.message for issue in result.issues)