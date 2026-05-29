from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_ibkr_paper_signal.py"


def _base_config(tmp_path: Path) -> dict:
    return {
        "phase": "phase_11",
        "mode": "paper_signal_only",
        "paper_only": True,
        "kill_switch": True,
        "live_orders_enabled": False,
        "allow_order_placement": False,
        "default_strategy": "mar_prob_linear_carry",
        "approved_strategies": [
            "unconditional_full",
            "threshold_hard_filter",
            "threshold_defensive",
            "hmm_prob_linear",
            "hmm_prob_linear_carry",
            "mar_prob_linear",
            "mar_prob_linear_carry",
        ],
        "forbidden_active_strategy_models": ["msvol", "msgarch"],
        "signal_inputs": {
            "US": str(tmp_path / "us_strategy_signals.csv"),
            "INDIA": str(tmp_path / "india_strategy_signals.csv"),
        },
        "signal_schema": {
            "canonical_required_columns": [
                "target_trade_date",
                "target_exposure",
                "strategy_available",
                "blocked_reason",
                "decision_reason",
            ],
            "strategy_column_candidates": ["strategy_name", "strategy"],
            "optional_columns": [
                "market",
                "signal_observation_date",
                "signal_date",
                "source_model",
                "vrp_har_gk",
                "har_forecast_available",
                "state_name",
                "p_calm",
                "p_transition",
                "p_stress",
            ],
            "aliases": {
                "strategy": "strategy_name",
                "signal_date": "signal_observation_date",
            },
            "validation": {
                "target_exposure_min": -1.0,
                "target_exposure_max": 0.0,
                "latest_signal_sort_column": "target_trade_date",
                "require_latest_by_target_trade_date": True,
                "allow_boolean_like_strategy_available": True,
            },
        },
        "signal_freshness": {
            "max_signal_age_days": 5,
            "block_stale_signal": True,
            "allow_weekend_gap": True,
        },
        "research_proxy_warning": RESEARCH_PROXY_WARNING,
        "broker": {
            "provider": "ibkr",
            "adapter": "ibridgepy",
            "optional_dependency": True,
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 11,
            "account": "PAPER_ACCOUNT_PLACEHOLDER",
            "require_env_account": True,
        },
        "env": {
            "account_var": "IBKR_PAPER_ACCOUNT",
            "host_var": "IBKR_HOST",
            "port_var": "IBKR_PORT",
            "client_id_var": "IBKR_CLIENT_ID",
        },
        "broker_status_taxonomy": [
            "IBRIDGEPY_NOT_INSTALLED",
            "IBRIDGEPY_IMPORT_OK",
            "BROKER_CONNECTION_NOT_ATTEMPTED",
            "BROKER_CONNECTION_FAILED",
            "BROKER_DATA_UNAVAILABLE",
            "BROKER_DATA_AVAILABLE",
        ],
        "final_status_taxonomy": [
            "ALLOWED_PAPER_INTENT",
            "BLOCKED_BY_KILL_SWITCH",
            "BLOCKED_STALE_SIGNAL",
            "BLOCKED_CONFIG_SAFETY",
            "BLOCKED_RISK_LIMIT",
            "BLOCKED_MISSING_SIGNAL",
            "BLOCKED_BROKER_DATA",
            "NO_SIGNAL",
            "STAY_FLAT",
            "BROKER_INSPECTION_ONLY",
        ],
        "allowed_markets": ["US", "INDIA"],
        "allowed_instruments": {
            "US": ["SPY", "VXX", "SVXY"],
            "INDIA": ["NIFTY_PROXY_MANUAL_ONLY"],
        },
        "blocked_instruments": [
            "naked_short_options",
            "live_short_options",
            "futures_without_permission",
            "margin_unknown_products",
        ],
        "instrument_mapping": {
            "US": {
                "research_underlying": "SPX_or_SPY",
                "default_paper_proxy": "SPY",
                "short_vol_proxy_candidates": ["VXX", "SVXY"],
                "notes": "Paper signal only.",
            },
            "INDIA": {
                "research_underlying": "NIFTY",
                "default_paper_proxy": "NIFTY_PROXY_MANUAL_ONLY",
                "notes": "Signal-only by default.",
            },
        },
        "manual_overrides": {
            "INDIA": {
                "manual_instrument_verified": False,
            },
        },
        "paper_sizing": {
            "paper_notional_per_full_exposure": 10000,
            "min_order_notional": 0,
            "round_shares": True,
            "allow_fractional_shares": False,
            "max_contracts": 0,
            "max_shares": 100,
            "max_notional": 10000,
            "max_delta": 0,
            "max_vega": 0,
            "max_margin_usage": 0.0,
            "allow_options": False,
            "allow_naked_short_options": False,
        },
        "risk_checks": {
            "require_paper_only": True,
            "require_kill_switch_off_for_allowed_intent": False,
            "block_if_kill_switch_on": True,
            "require_market_open": False,
            "max_quote_age_seconds": 300,
            "max_bid_ask_spread_bps": 100,
            "require_quote_for_allowed_intent": False,
            "block_stale_quote_if_quote_available": True,
            "block_missing_quote": False,
            "block_live_order_functions": True,
            "block_stale_signal": True,
            "block_naked_short_options": True,
            "block_india_unverified_instrument": True,
        },
        "state": {
            "use_position_state": False,
            "paper_position_state": str(tmp_path / "paper_position_state.csv"),
        },
        "audit": {
            "write_config_snapshot": True,
            "write_run_metadata": True,
            "include_phase10_proxy_warning": True,
            "scan_for_live_order_functions": True,
        },
        "outputs": {
            "broker_cache_dir": str(tmp_path / "broker_cache"),
            "tables_dir": str(tmp_path / "phase_11"),
            "logs_dir": str(tmp_path / "logs"),
            "latest_signal_table": str(tmp_path / "phase_11" / "daily_paper_signal.csv"),
            "paper_order_intents": str(tmp_path / "phase_11" / "paper_order_intents.csv"),
            "risk_check_report": str(tmp_path / "phase_11" / "risk_check_report.csv"),
            "broker_metadata": str(tmp_path / "phase_11" / "broker_metadata.json"),
            "run_metadata": str(tmp_path / "phase_11" / "run_metadata.json"),
            "config_snapshot": str(tmp_path / "phase_11" / "ibkr_paper_config_snapshot.yaml"),
        },
    }


def _write_config(tmp_path: Path, config: dict | None = None) -> Path:
    path = tmp_path / "ibkr_paper.yaml"
    data = config or _base_config(tmp_path)

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return path


def _write_signal(path: Path, *, market: str = "US", exposure: float = -0.6) -> Path:
    pd.DataFrame(
        {
            "market": [market],
            "strategy_name": ["mar_prob_linear_carry"],
            "signal_observation_date": ["2026-05-22"],
            "target_trade_date": ["2026-05-23"],
            "target_exposure": [exposure],
            "strategy_available": [True],
            "blocked_reason": [""],
            "decision_reason": ["carry gate valid"],
            "vrp_har_gk": [0.03],
            "har_forecast_available": [True],
            "state_name": ["calm"],
            "p_calm": [0.9],
            "p_transition": [0.05],
            "p_stress": [0.05],
        }
    ).to_csv(path, index=False)

    return path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_dry_run_writes_no_outputs(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--dry-run",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["market"] == "US"
    assert payload["strategy"] == "mar_prob_linear_carry"
    assert payload["live_order_sent"] is False
    assert not (tmp_path / "phase_11" / "daily_paper_signal.csv").exists()


def test_cli_end_to_end_missing_quote_blocks_by_kill_switch(tmp_path: Path) -> None:
    signal_path = _write_signal(tmp_path / "us_strategy_signals.csv")
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--as-of-date",
            "2026-05-24",
            "--run-timestamp-utc",
            "2026-05-24T00:00:00+00:00",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is False
    assert payload["market"] == "US"
    assert payload["daily_signal_final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert payload["paper_intent_written"] is True
    assert payload["paper_intent_final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert payload["final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert payload["live_order_sent"] is False

    daily_signal = pd.read_csv(tmp_path / "phase_11" / "daily_paper_signal.csv")
    intents = pd.read_csv(tmp_path / "phase_11" / "paper_order_intents.csv")
    risk_report = pd.read_csv(tmp_path / "phase_11" / "risk_check_report.csv")
    run_metadata = json.loads(
        (tmp_path / "phase_11" / "run_metadata.json").read_text(encoding="utf-8")
    )
    broker_metadata = json.loads(
        (tmp_path / "phase_11" / "broker_metadata.json").read_text(encoding="utf-8")
    )

    assert len(daily_signal) == 1
    assert len(intents) == 1
    assert len(risk_report) >= 1
    assert intents.loc[0, "symbol"] == "VXX"
    assert pd.isna(intents.loc[0, "paper_quantity"])
    assert run_metadata["live_order_sent"] is False
    assert broker_metadata["live_order_sent"] is False
    assert (tmp_path / "phase_11" / "ibkr_paper_config_snapshot.yaml").exists()


def test_cli_end_to_end_manual_quote_blocks_risk_limit(tmp_path: Path) -> None:
    signal_path = _write_signal(tmp_path / "us_strategy_signals.csv")
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--as-of-date",
            "2026-05-24",
            "--run-timestamp-utc",
            "2026-05-24T00:00:00+00:00",
            "--quote-symbol",
            "VXX",
            "--quote-bid",
            "14.95",
            "--quote-ask",
            "15.05",
            "--quote-age-seconds",
            "30",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["paper_intent_final_status"] == "BLOCKED_RISK_LIMIT"
    assert payload["final_status"] == "BLOCKED_RISK_LIMIT"
    assert payload["live_order_sent"] is False

    intents = pd.read_csv(tmp_path / "phase_11" / "paper_order_intents.csv")
    assert intents.loc[0, "symbol"] == "VXX"
    assert intents.loc[0, "side"] == "SELL"
    assert intents.loc[0, "paper_quantity"] == 400
    assert intents.loc[0, "final_status"] == "BLOCKED_RISK_LIMIT"


def test_cli_end_to_end_manual_quote_blocks_kill_switch_when_limits_ok(
    tmp_path: Path,
) -> None:
    config_data = _base_config(tmp_path)
    config_data["paper_sizing"]["max_shares"] = 1000

    signal_path = _write_signal(tmp_path / "us_strategy_signals.csv")
    config_path = _write_config(tmp_path, config_data)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--as-of-date",
            "2026-05-24",
            "--run-timestamp-utc",
            "2026-05-24T00:00:00+00:00",
            "--quote-symbol",
            "VXX",
            "--quote-bid",
            "14.95",
            "--quote-ask",
            "15.05",
            "--quote-age-seconds",
            "30",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["paper_intent_final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert payload["final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert payload["live_order_sent"] is False


def test_cli_no_signal_writes_empty_intent_file(tmp_path: Path) -> None:
    signal_path = tmp_path / "us_strategy_signals.csv"
    pd.DataFrame(
        {
            "market": ["US"],
            "strategy_name": ["mar_prob_linear_carry"],
            "signal_observation_date": ["2026-05-22"],
            "target_trade_date": ["2026-05-23"],
            "target_exposure": [-0.6],
            "strategy_available": [False],
            "blocked_reason": ["HAR unavailable"],
            "decision_reason": ["strategy unavailable"],
        }
    ).to_csv(signal_path, index=False)

    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--as-of-date",
            "2026-05-24",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["daily_signal_final_status"] == "NO_SIGNAL"
    assert payload["paper_intent_written"] is False
    assert payload["final_status"] == "NO_SIGNAL"

    intents = pd.read_csv(tmp_path / "phase_11" / "paper_order_intents.csv")
    assert intents.empty


def test_cli_stale_signal_blocks(tmp_path: Path) -> None:
    signal_path = _write_signal(tmp_path / "us_strategy_signals.csv")
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--as-of-date",
            "2026-06-30",
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["daily_signal_final_status"] == "BLOCKED_STALE_SIGNAL"
    assert payload["paper_intent_final_status"] == "BLOCKED_STALE_SIGNAL"
    assert payload["final_status"] == "BLOCKED_STALE_SIGNAL"
    assert payload["live_order_sent"] is False


def test_cli_rejects_bad_as_of_date(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--as-of-date",
            "bad-date",
        ]
    )

    assert result.returncode == 2
    assert "as-of-date" in result.stderr


def test_cli_rejects_bad_quote_without_symbol(tmp_path: Path) -> None:
    signal_path = _write_signal(tmp_path / "us_strategy_signals.csv")
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--market",
            "US",
            "--signal-path",
            str(signal_path),
            "--quote-bid",
            "10",
            "--quote-ask",
            "11",
        ]
    )

    assert result.returncode == 2
    assert "quote-symbol" in result.stderr