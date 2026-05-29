from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.market_data import quote_from_mapping
from vrp.broker.paper_trader import build_paper_order_intent
from vrp.broker.signal_publisher import DailyPaperSignal
from vrp.reports.broker_diagnostics import (
    build_broker_metadata_record,
    build_run_metadata,
    read_json_file,
    write_broker_metadata,
    write_phase11_diagnostics,
    write_run_metadata,
)


def _base_config() -> dict:
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
            "US": "data/processed/us_strategy_signals.parquet",
            "INDIA": "data/processed/india_strategy_signals.parquet",
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
            "paper_position_state": "reports/tables/phase_11/paper_position_state.csv",
        },
        "audit": {
            "write_config_snapshot": True,
            "write_run_metadata": True,
            "include_phase10_proxy_warning": True,
            "scan_for_live_order_functions": True,
        },
        "outputs": {
            "broker_cache_dir": "data/broker_cache",
            "tables_dir": "reports/tables/phase_11",
            "logs_dir": "logs",
            "latest_signal_table": "reports/tables/phase_11/daily_paper_signal.csv",
            "paper_order_intents": "reports/tables/phase_11/paper_order_intents.csv",
            "risk_check_report": "reports/tables/phase_11/risk_check_report.csv",
            "broker_metadata": "reports/tables/phase_11/broker_metadata.json",
            "run_metadata": "reports/tables/phase_11/run_metadata.json",
            "config_snapshot": "reports/tables/phase_11/ibkr_paper_config_snapshot.yaml",
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "ibkr_paper.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _load_config(tmp_path: Path, config_data: dict | None = None):
    return load_broker_config(_write_config(tmp_path, config_data or _base_config()))


def _daily_signal() -> DailyPaperSignal:
    return DailyPaperSignal(
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
        market="US",
        strategy_name="mar_prob_linear_carry",
        signal_observation_date="2026-05-22",
        target_trade_date="2026-05-23",
        target_exposure=-0.6,
        strategy_available=True,
        recommended_action="PAPER_SHORT_VOL_INTENT",
        blocked_reason="",
        decision_reason="carry gate valid",
        signal_age_days=1,
        signal_is_stale=False,
        signal_freshness_reason="fresh",
        paper_only=True,
        kill_switch=True,
        live_orders_enabled=False,
        allow_order_placement=False,
        intent_allowed_before_kill_switch=True,
        intent_allowed_after_kill_switch=False,
        final_status="BLOCKED_BY_KILL_SWITCH",
        live_order_sent=False,
        research_proxy_warning=RESEARCH_PROXY_WARNING,
        source_signal_path="data/processed/us_strategy_signals.parquet",
        source_signal_mtime_utc="2026-05-24T00:00:00+00:00",
        source_strategy_row={},
    )


def test_build_broker_metadata_record_defaults(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    record = build_broker_metadata_record(
        config,
        broker_metadata={
            "ibridgepy_available": False,
            "broker_connection_attempted": False,
            "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
            "broker_data_status": "BROKER_DATA_UNAVAILABLE",
        },
    )

    assert record["ibridgepy_available"] is False
    assert record["broker_connection_attempted"] is False
    assert record["broker_connection_status"] == "IBRIDGEPY_NOT_INSTALLED"
    assert record["broker_data_status"] == "BROKER_DATA_UNAVAILABLE"
    assert record["paper_only"] is True
    assert record["kill_switch"] is True
    assert record["live_orders_enabled"] is False
    assert record["allow_order_placement"] is False
    assert record["live_order_sent"] is False
    assert "research-layer proxy units" in record["research_proxy_warning"]


def test_build_run_metadata_without_intent(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal()

    metadata = build_run_metadata(
        config=config,
        market="US",
        strategy="mar_prob_linear_carry",
        daily_signal=signal,
        broker_metadata={
            "ibridgepy_available": False,
            "broker_connection_attempted": False,
            "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
            "broker_data_status": "BROKER_DATA_UNAVAILABLE",
            "live_order_sent": False,
        },
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert metadata.market == "US"
    assert metadata.strategy == "mar_prob_linear_carry"
    assert metadata.latest_target_trade_date == "2026-05-23"
    assert metadata.daily_signal_final_status == "BLOCKED_BY_KILL_SWITCH"
    assert metadata.paper_intent_final_status is None
    assert metadata.final_status == "BLOCKED_BY_KILL_SWITCH"
    assert metadata.paper_only is True
    assert metadata.kill_switch is True
    assert metadata.live_order_sent is False
    assert len(metadata.config_hash) == 64


def test_build_run_metadata_uses_paper_intent_status_when_available(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal()
    quote = quote_from_mapping(
        {
            "symbol": "VXX",
            "market": "US",
            "bid": 14.95,
            "ask": 15.05,
            "quote_age_seconds": 30,
        }
    )

    paper_result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=quote,
    )

    metadata = build_run_metadata(
        config=config,
        market="US",
        strategy="mar_prob_linear_carry",
        daily_signal=signal,
        paper_result=paper_result,
        broker_metadata={
            "ibridgepy_available": False,
            "broker_connection_attempted": False,
            "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
            "broker_data_status": "BROKER_DATA_UNAVAILABLE",
            "live_order_sent": False,
        },
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert metadata.paper_intent_final_status == "BLOCKED_RISK_LIMIT"
    assert metadata.final_status == "BLOCKED_RISK_LIMIT"
    assert metadata.live_order_sent is False


def test_write_broker_metadata(tmp_path: Path) -> None:
    path = tmp_path / "broker_metadata.json"

    written = write_broker_metadata(
        {
            "ibridgepy_available": False,
            "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
        },
        path,
    )

    assert written == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ibridgepy_available"] is False


def test_write_run_metadata(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal()

    metadata = build_run_metadata(
        config=config,
        market="US",
        daily_signal=signal,
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    path = tmp_path / "run_metadata.json"
    written = write_run_metadata(metadata, path)

    assert written == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["market"] == "US"
    assert payload["final_status"] == "BLOCKED_BY_KILL_SWITCH"


def test_write_phase11_diagnostics_writes_all_files(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["outputs"]["broker_metadata"] = str(tmp_path / "broker_metadata.json")
    config_data["outputs"]["run_metadata"] = str(tmp_path / "run_metadata.json")
    config_data["outputs"]["config_snapshot"] = str(tmp_path / "snapshot.yaml")
    config_data["outputs"]["broker_cache_dir"] = str(tmp_path / "broker_cache")
    config_data["outputs"]["tables_dir"] = str(tmp_path / "tables")
    config_data["outputs"]["logs_dir"] = str(tmp_path / "logs")

    config = _load_config(tmp_path, config_data)
    signal = _daily_signal()

    result = write_phase11_diagnostics(
        config=config,
        market="US",
        strategy="mar_prob_linear_carry",
        daily_signal=signal,
        broker_metadata={
            "ibridgepy_available": False,
            "broker_connection_attempted": False,
            "broker_connection_status": "IBRIDGEPY_NOT_INSTALLED",
            "broker_data_status": "BROKER_DATA_UNAVAILABLE",
            "live_order_sent": False,
        },
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert result.broker_metadata_path.exists()
    assert result.run_metadata_path.exists()
    assert result.config_snapshot_path is not None
    assert result.config_snapshot_path.exists()

    broker_payload = read_json_file(result.broker_metadata_path)
    run_payload = read_json_file(result.run_metadata_path)

    assert broker_payload["broker_connection_status"] == "IBRIDGEPY_NOT_INSTALLED"
    assert run_payload["final_status"] == "BLOCKED_BY_KILL_SWITCH"
    assert "research-layer proxy units" in run_payload["research_proxy_warning"]


def test_write_phase11_diagnostics_respects_snapshot_disabled(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["audit"]["write_config_snapshot"] = False
    config_data["outputs"]["broker_metadata"] = str(tmp_path / "broker_metadata.json")
    config_data["outputs"]["run_metadata"] = str(tmp_path / "run_metadata.json")
    config_data["outputs"]["config_snapshot"] = str(tmp_path / "snapshot.yaml")
    config_data["outputs"]["broker_cache_dir"] = str(tmp_path / "broker_cache")
    config_data["outputs"]["tables_dir"] = str(tmp_path / "tables")
    config_data["outputs"]["logs_dir"] = str(tmp_path / "logs")

    config = _load_config(tmp_path, config_data)

    result = write_phase11_diagnostics(
        config=config,
        market="US",
        daily_signal=_daily_signal(),
    )

    assert result.broker_metadata_path.exists()
    assert result.run_metadata_path.exists()
    assert result.config_snapshot_path is None
    assert not (tmp_path / "snapshot.yaml").exists()


def test_read_json_file(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    assert read_json_file(path) == {"a": 1}