from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.ibridgepy_adapter import (
    BROKER_CONNECTION_FAILED,
    BROKER_CONNECTION_NOT_ATTEMPTED,
    BROKER_DATA_UNAVAILABLE,
    IBRIDGEPY_IMPORT_OK,
    IBRIDGEPY_NOT_INSTALLED,
    BrokerAdapterError,
    IBridgePyAdapter,
    get_broker_availability,
    get_broker_metadata,
    validate_broker_status_value,
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


def _load_config(tmp_path: Path):
    return load_broker_config(_write_config(tmp_path, _base_config()))


def test_adapter_reports_not_installed_without_crashing(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    def fake_import(_: str):
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    availability = adapter.check_availability()

    assert availability.ibridgepy_available is False
    assert availability.imported_module_name is None
    assert availability.broker_connection_attempted is False
    assert availability.broker_connection_status == IBRIDGEPY_NOT_INSTALLED
    assert availability.broker_data_status == BROKER_DATA_UNAVAILABLE


def test_adapter_reports_import_ok(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    fake_module = types.ModuleType("IBridgePy")

    def fake_import(name: str):
        if name == "IBridgePy":
            return fake_module
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    availability = adapter.check_availability()

    assert availability.ibridgepy_available is True
    assert availability.imported_module_name == "IBridgePy"
    assert availability.broker_connection_attempted is False
    assert availability.broker_connection_status == IBRIDGEPY_IMPORT_OK
    assert availability.broker_data_status == BROKER_CONNECTION_NOT_ATTEMPTED


def test_adapter_tries_lowercase_candidate(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    fake_module = types.ModuleType("ibridgepy")

    def fake_import(name: str):
        if name == "ibridgepy":
            return fake_module
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    availability = adapter.check_availability()

    assert availability.ibridgepy_available is True
    assert availability.imported_module_name == "ibridgepy"


def test_adapter_caches_import_result(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    fake_module = types.ModuleType("IBridgePy")
    calls = {"count": 0}

    def fake_import(name: str):
        calls["count"] += 1
        if name == "IBridgePy":
            return fake_module
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)

    first = adapter.check_availability()
    second = adapter.check_availability()

    assert first.ibridgepy_available is True
    assert second.ibridgepy_available is True
    assert calls["count"] == 1


def test_inspect_broker_readiness_default_does_not_attempt_connection(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)

    def fake_import(_: str):
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    result = adapter.inspect_broker_readiness()

    assert result.availability.broker_connection_attempted is False
    assert result.live_order_sent is False
    assert result.paper_only is True
    assert result.kill_switch is True
    assert result.live_orders_enabled is False
    assert result.allow_order_placement is False


def test_inspect_broker_readiness_attempt_connection_is_structured_failure(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)
    fake_module = types.ModuleType("IBridgePy")

    def fake_import(_: str):
        return fake_module

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    result = adapter.inspect_broker_readiness(attempt_connection=True)

    assert result.availability.ibridgepy_available is True
    assert result.availability.broker_connection_attempted is True
    assert result.availability.broker_connection_status == BROKER_CONNECTION_FAILED
    assert result.availability.broker_data_status == BROKER_DATA_UNAVAILABLE
    assert result.live_order_sent is False


def test_broker_metadata_is_json_friendly(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    def fake_import(_: str):
        raise ImportError("missing")

    adapter = IBridgePyAdapter(config, import_func=fake_import)
    metadata = adapter.broker_metadata()

    assert metadata["ibridgepy_available"] is False
    assert metadata["broker_connection_attempted"] is False
    assert metadata["broker_connection_status"] == IBRIDGEPY_NOT_INSTALLED
    assert metadata["live_order_sent"] is False
    assert metadata["paper_only"] is True
    assert metadata["kill_switch"] is True


def test_convenience_get_broker_availability_runs_without_real_ibridgepy(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)

    availability = get_broker_availability(config)

    assert availability.broker_connection_attempted is False
    assert availability.broker_connection_status in {
        IBRIDGEPY_NOT_INSTALLED,
        IBRIDGEPY_IMPORT_OK,
    }


def test_convenience_get_broker_metadata_runs_without_real_ibridgepy(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)

    metadata = get_broker_metadata(config)

    assert metadata["broker_connection_attempted"] is False
    assert metadata["live_order_sent"] is False


def test_validate_broker_status_value_accepts_known_status() -> None:
    assert validate_broker_status_value(IBRIDGEPY_NOT_INSTALLED) == (
        IBRIDGEPY_NOT_INSTALLED
    )


def test_validate_broker_status_value_rejects_unknown_status() -> None:
    with pytest.raises(BrokerAdapterError, match="Unknown broker status"):
        validate_broker_status_value("BAD_STATUS")


def test_adapter_class_does_not_expose_execution_method_names(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    adapter = IBridgePyAdapter(config)

    forbidden_method_names = {
        "place_order",
        "order_target",
        "order_value",
        "order_percent",
        "submit_order",
        "market_order",
        "limit_order",
        "bracket_order",
        "buy",
        "sell",
    }

    exposed = set(dir(adapter))

    assert not forbidden_method_names.intersection(exposed)