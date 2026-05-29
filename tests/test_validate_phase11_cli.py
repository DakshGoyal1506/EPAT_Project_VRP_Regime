from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from vrp.broker import RESEARCH_PROXY_WARNING


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_phase11.py"


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
            "US": str(tmp_path / "us_strategy_signals.parquet"),
            "INDIA": str(tmp_path / "india_strategy_signals.parquet"),
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


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "ibkr_paper.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_base_config(tmp_path), handle, sort_keys=False)

    return path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_phase11_source_guard_only(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    guard_report = tmp_path / "live_order_guard_report.json"

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--skip-artifacts",
            "--source-guard-report",
            str(guard_report),
            "--print-json",
        ]
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)

    assert payload["passed"] is True
    assert payload["source_guard_passed"] is True
    assert payload["integration_passed"] is None
    assert payload["live_order_sent"] is False
    assert guard_report.exists()


def test_validate_phase11_rejects_skipping_everything(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--skip-artifacts",
            "--skip-source-guard",
            "--print-json",
        ]
    )

    assert result.returncode == 2
    assert "At least one validation" in result.stderr


def test_validate_phase11_artifacts_fail_when_missing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = _run_cli(
        [
            "--config",
            str(config_path),
            "--skip-source-guard",
            "--print-json",
        ]
    )

    assert result.returncode == 2
    assert "integration checks failed" in result.stderr