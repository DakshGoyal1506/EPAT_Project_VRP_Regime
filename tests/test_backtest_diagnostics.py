from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from vrp.backtest.backtest_config import load_backtest_config
from vrp.backtest.costs import apply_costs_to_backtest_panel
from vrp.backtest.payoff_proxies import (
    build_forward_vrp_outcome_panel,
    compute_forward_vrp_strategy_payoff,
    join_strategy_with_outcome,
)
from vrp.reports.backtest_diagnostics import (
    REPORT_LIMITATIONS,
    VISUAL_INTERPRETATION_WARNING,
    build_backtest_by_strategy_year_table,
    build_backtest_summary_table,
    build_common_start_panel,
    build_common_start_summary_table,
    build_crisis_window_performance_table,
    build_no_lookahead_audit_table,
    build_tail_summary_table,
    generate_backtest_diagnostics,
    get_common_start_dates,
)


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


def _write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "backtest.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _build_toy_panel(market: str = "US") -> pd.DataFrame:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    outcomes = pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")

    signals["market"] = market
    outcomes["market"] = market

    outcome_panel = build_forward_vrp_outcome_panel(outcomes)
    joined = join_strategy_with_outcome(signals, outcome_panel)
    payoff = compute_forward_vrp_strategy_payoff(joined)

    return apply_costs_to_backtest_panel(payoff, cost_bps=5.0)


def test_backtest_summary_table_contains_caveats(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_backtest_config(config_path)

    panel = _build_toy_panel()
    summary = build_backtest_summary_table(panel, config=config)

    assert not summary.empty
    assert "uses_overlapping_forward_labels" in summary.columns
    assert "annualized_metrics_interpretation" in summary.columns
    assert "research_proxy_not_trade_pnl" in summary.columns
    assert summary["uses_overlapping_forward_labels"].all()
    assert summary["research_proxy_not_trade_pnl"].all()


def test_year_table_and_crisis_table_build(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_backtest_config(config_path)

    panel = _build_toy_panel()

    by_year = build_backtest_by_strategy_year_table(panel, config=config)
    crisis = build_crisis_window_performance_table(panel, config=config)

    assert not by_year.empty
    assert "year" in by_year.columns
    assert set(by_year["year"]) == {2020}

    assert not crisis.empty
    assert set(crisis["subperiod"]) == {"ToyWindow"}


def test_no_lookahead_audit_passes_on_toy_panel(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_backtest_config(config_path)

    panel = _build_toy_panel()
    audit = build_no_lookahead_audit_table(panel, config=config)

    assert not audit.empty
    assert audit["passes_no_lookahead_audit"].all()
    assert audit["n_target_not_after_signal_violations"].sum() == 0
    assert audit["n_outcome_not_equal_signal_date_violations"].sum() == 0


def test_common_start_panel_and_summary_build(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_backtest_config(config_path)

    panel = _build_toy_panel()

    common_dates = get_common_start_dates(panel)
    common_panel = build_common_start_panel(panel)
    common_summary = build_common_start_summary_table(panel, config=config)

    assert common_dates == {"US": "2020-01-03"}
    assert not common_panel.empty
    assert "common_start_date" in common_panel.columns
    assert set(common_panel["common_start_date"]) == {"2020-01-03"}

    assert not common_summary.empty
    assert "common_start_date" in common_summary.columns
    assert "return_per_abs_exposure" in common_summary.columns
    assert "return_to_drawdown" in common_summary.columns


def test_tail_summary_table_builds() -> None:
    panel = _build_toy_panel()

    tail = build_tail_summary_table(panel)

    assert not tail.empty

    required = {
        "market",
        "strategy_name",
        "n_eligible",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "worst_return",
        "best_return",
        "cte_95",
    }
    assert required.issubset(set(tail.columns))
    assert set(tail["strategy_name"]) == set(panel["strategy_name"].unique())


def test_generate_backtest_diagnostics_writes_tables_figures_and_metadata(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_backtest_config(config_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    us_panel = _build_toy_panel("US")
    india_panel = _build_toy_panel("INDIA")

    us_panel.to_parquet(processed / "us_backtest_panel.parquet", index=False)
    india_panel.to_parquet(processed / "india_backtest_panel.parquet", index=False)

    result = generate_backtest_diagnostics(
        config=config,
        repo_root=tmp_path,
        market="ALL",
    )

    for path in result.table_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    for path in result.figure_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    assert result.metadata_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["phase"] == "phase_10"
    assert metadata["markets"] == ["US", "INDIA"]
    assert metadata["payoff_label"] == "vrp_forward_expost_gk_label"
    assert metadata["label_role"] == "realised_outcome_only"
    assert metadata["overlapping_labels"] is True
    assert metadata["research_proxy_not_trade_pnl"] is True
    assert metadata["no_lookahead_audit_passed"] is True
    assert metadata["visual_interpretation_warning"] == VISUAL_INTERPRETATION_WARNING
    assert metadata["common_start_dates"] == {
        "INDIA": "2020-01-03",
        "US": "2020-01-03",
    }

    for limitation in REPORT_LIMITATIONS:
        assert limitation in metadata["limitations"]

    expected_tables = {
        "backtest_summary",
        "backtest_common_start_summary",
        "backtest_tail_summary",
        "backtest_by_strategy_year",
        "crisis_window_performance",
        "backtest_availability_summary",
        "backtest_no_lookahead_audit",
    }
    assert expected_tables.issubset(set(result.table_paths))

    expected_figures = {
        "equity_curves_us",
        "equity_curves_common_start_us",
        "drawdowns_us",
        "return_distribution_us",
        "equity_curves_india",
        "equity_curves_common_start_india",
        "drawdowns_india",
        "return_distribution_india",
    }
    assert expected_figures.issubset(set(result.figure_paths))