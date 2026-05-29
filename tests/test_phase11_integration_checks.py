from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.phase11_integration_checks import (
    Phase11IntegrationError,
    assert_phase11_artifacts_valid,
    check_phase11_artifacts,
    write_phase11_integration_report,
)


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


def _load_config(tmp_path: Path):
    return load_broker_config(_write_config(tmp_path))


def _write_common_json_artifacts(
    tmp_path: Path,
    *,
    daily_status: str,
    intent_status: str | None,
    final_status: str,
    live_order_sent: bool = False,
) -> None:
    root = tmp_path / "phase_11"
    root.mkdir(parents=True, exist_ok=True)

    broker_metadata = {
        "ibridgepy_available": False,
        "imported_module_name": None,
        "broker_connection_attempted": False,
        "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
        "broker_data_status": "BROKER_DATA_UNAVAILABLE",
        "paper_only": True,
        "kill_switch": True,
        "live_orders_enabled": False,
        "allow_order_placement": False,
        "live_order_sent": live_order_sent,
        "research_proxy_warning": RESEARCH_PROXY_WARNING,
    }

    run_metadata = {
        "run_timestamp_utc": "2026-05-24T00:00:00+00:00",
        "market": "US",
        "strategy": "mar_prob_linear_carry",
        "config_path": str(tmp_path / "ibkr_paper.yaml"),
        "config_hash": "a" * 64,
        "input_signal_path": str(tmp_path / "us_strategy_signals.parquet"),
        "input_signal_mtime": None,
        "latest_target_trade_date": "2026-05-23",
        "daily_signal_final_status": daily_status,
        "paper_intent_final_status": intent_status,
        "final_status": final_status,
        "ibridgepy_available": False,
        "broker_connection_attempted": False,
        "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
        "broker_data_status": "BROKER_DATA_UNAVAILABLE",
        "paper_only": True,
        "kill_switch": True,
        "live_orders_enabled": False,
        "allow_order_placement": False,
        "live_order_sent": live_order_sent,
        "research_proxy_warning": RESEARCH_PROXY_WARNING,
    }

    (root / "broker_metadata.json").write_text(
        json.dumps(broker_metadata, indent=2),
        encoding="utf-8",
    )
    (root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    (root / "ibkr_paper_config_snapshot.yaml").write_text(
        "paper_only: true\nkill_switch: true\nlive_orders_enabled: false\nallow_order_placement: false\n",
        encoding="utf-8",
    )


def _write_daily_signal(
    tmp_path: Path,
    *,
    final_status: str,
    live_order_sent: bool = False,
) -> None:
    root = tmp_path / "phase_11"
    root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "market": "US",
                "strategy_name": "mar_prob_linear_carry",
                "recommended_action": "PAPER_SHORT_VOL_INTENT",
                "target_exposure": -0.6,
                "paper_only": True,
                "kill_switch": True,
                "live_orders_enabled": False,
                "allow_order_placement": False,
                "final_status": final_status,
                "live_order_sent": live_order_sent,
                "research_proxy_warning": RESEARCH_PROXY_WARNING,
            }
        ]
    ).to_csv(root / "daily_paper_signal.csv", index=False)


def _write_empty_intents_and_risk(tmp_path: Path) -> None:
    root = tmp_path / "phase_11"
    root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        columns=[
            "market",
            "strategy_name",
            "recommended_action",
            "symbol",
            "side",
            "target_exposure",
            "paper_target_notional",
            "final_status",
            "live_order_sent",
            "research_proxy_warning",
        ]
    ).to_csv(root / "paper_order_intents.csv", index=False)

    pd.DataFrame(
        columns=[
            "market",
            "symbol",
            "recommended_action",
            "final_status",
            "check_name",
            "status",
            "blocks_intent",
            "reason",
        ]
    ).to_csv(root / "risk_check_report.csv", index=False)


def _write_intent_and_risk(
    tmp_path: Path,
    *,
    final_status: str,
    live_order_sent: bool = False,
) -> None:
    root = tmp_path / "phase_11"
    root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "market": "US",
                "strategy_name": "mar_prob_linear_carry",
                "recommended_action": "PAPER_SHORT_VOL_INTENT",
                "symbol": "VXX",
                "side": "SELL",
                "target_exposure": -0.6,
                "paper_target_notional": 6000.0,
                "final_status": final_status,
                "live_order_sent": live_order_sent,
                "research_proxy_warning": RESEARCH_PROXY_WARNING,
            }
        ]
    ).to_csv(root / "paper_order_intents.csv", index=False)

    pd.DataFrame(
        [
            {
                "market": "US",
                "symbol": "VXX",
                "recommended_action": "PAPER_SHORT_VOL_INTENT",
                "final_status": final_status,
                "check_name": "kill_switch",
                "status": "FAILED",
                "blocks_intent": True,
                "reason": "kill_switch is on and blocks paper intent",
            }
        ]
    ).to_csv(root / "risk_check_report.csv", index=False)


def test_valid_terminal_stale_no_intent_passes(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(tmp_path, final_status="BLOCKED_STALE_SIGNAL")
    _write_empty_intents_and_risk(tmp_path)
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_STALE_SIGNAL",
        intent_status=None,
        final_status="BLOCKED_STALE_SIGNAL",
    )

    report = check_phase11_artifacts(config)

    assert report.passed is True
    assert report.violations == ()


def test_valid_intent_blocked_by_kill_switch_passes(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(tmp_path, final_status="BLOCKED_BY_KILL_SWITCH")
    _write_intent_and_risk(tmp_path, final_status="BLOCKED_BY_KILL_SWITCH")
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_BY_KILL_SWITCH",
        intent_status="BLOCKED_BY_KILL_SWITCH",
        final_status="BLOCKED_BY_KILL_SWITCH",
    )

    report = check_phase11_artifacts(config)

    assert report.passed is True


def test_missing_artifacts_fail(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    report = check_phase11_artifacts(config)

    assert report.passed is False
    assert any(v.check_name == "artifact_exists" for v in report.violations)


def test_terminal_status_with_intent_fails(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(tmp_path, final_status="BLOCKED_STALE_SIGNAL")
    _write_intent_and_risk(tmp_path, final_status="BLOCKED_STALE_SIGNAL")
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_STALE_SIGNAL",
        intent_status="BLOCKED_STALE_SIGNAL",
        final_status="BLOCKED_STALE_SIGNAL",
    )

    report = check_phase11_artifacts(config)

    assert report.passed is False
    assert any(v.check_name == "terminal_status_intent_written" for v in report.violations)


def test_run_final_status_mismatch_fails(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(tmp_path, final_status="BLOCKED_BY_KILL_SWITCH")
    _write_intent_and_risk(tmp_path, final_status="BLOCKED_RISK_LIMIT")
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_BY_KILL_SWITCH",
        intent_status="BLOCKED_RISK_LIMIT",
        final_status="BLOCKED_BY_KILL_SWITCH",
    )

    report = check_phase11_artifacts(config)

    assert report.passed is False
    assert any(v.check_name == "run_final_status_priority" for v in report.violations)


def test_live_order_sent_fails(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(
        tmp_path,
        final_status="BLOCKED_BY_KILL_SWITCH",
        live_order_sent=True,
    )
    _write_intent_and_risk(tmp_path, final_status="BLOCKED_BY_KILL_SWITCH")
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_BY_KILL_SWITCH",
        intent_status="BLOCKED_BY_KILL_SWITCH",
        final_status="BLOCKED_BY_KILL_SWITCH",
    )

    report = check_phase11_artifacts(config)

    assert report.passed is False
    assert any(v.check_name == "daily_signal_live_order_sent" for v in report.violations)


def test_assert_phase11_artifacts_valid_raises(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    with pytest.raises(Phase11IntegrationError, match="integration checks failed"):
        assert_phase11_artifacts_valid(config)


def test_write_phase11_integration_report(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    _write_daily_signal(tmp_path, final_status="BLOCKED_STALE_SIGNAL")
    _write_empty_intents_and_risk(tmp_path)
    _write_common_json_artifacts(
        tmp_path,
        daily_status="BLOCKED_STALE_SIGNAL",
        intent_status=None,
        final_status="BLOCKED_STALE_SIGNAL",
    )

    report = check_phase11_artifacts(config)
    output_path = tmp_path / "phase_11" / "phase11_integration_report.json"

    written = write_phase11_integration_report(report, output_path)

    assert written == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["violations"] == []