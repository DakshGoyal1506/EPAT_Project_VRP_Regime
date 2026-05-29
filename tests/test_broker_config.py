from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import (
    BrokerConfigError,
    compute_config_hash,
    ensure_output_directories,
    get_market_signal_path,
    get_output_paths,
    load_broker_config,
    validate_broker_config,
    write_config_snapshot,
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
            "broker_cache_dir": "data/broker_cache",
            "tables_dir": "reports/tables/phase_11",
            "logs_dir": "logs",
            "latest_signal_table": "reports/tables/phase_11/daily_paper_signal.csv",
            "paper_order_intents": "reports/tables/phase_11/paper_order_intents.csv",
            "risk_check_report": "reports/tables/phase_11/risk_check_report.csv",
            "broker_metadata": "reports/tables/phase_11/broker_metadata.json",
            "run_metadata": "reports/tables/phase_11/run_metadata.json",
            "config_snapshot": (
                "reports/tables/phase_11/ibkr_paper_config_snapshot.yaml"
            ),
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "ibkr_paper.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def test_load_broker_config_valid_defaults(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_config())

    config = load_broker_config(path)

    assert config.phase == "phase_11"
    assert config.mode == "paper_signal_only"
    assert config.paper_only is True
    assert config.kill_switch is True
    assert config.live_orders_enabled is False
    assert config.allow_order_placement is False
    assert config.default_strategy == "mar_prob_linear_carry"
    assert config.paper_sizing.allow_options is False
    assert config.paper_sizing.allow_naked_short_options is False
    assert config.paper_sizing.max_contracts == 0
    assert config.paper_sizing.max_margin_usage == 0.0


def test_env_overrides_are_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(tmp_path, _base_config())

    monkeypatch.setenv("IBKR_PAPER_ACCOUNT", "DU1234567")
    monkeypatch.setenv("IBKR_HOST", "localhost")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")

    config = load_broker_config(path)

    assert config.broker.account == "DU1234567"
    assert config.broker.host == "localhost"
    assert config.broker.port == 4002
    assert config.broker.client_id == 42


def test_invalid_env_integer_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(tmp_path, _base_config())
    monkeypatch.setenv("IBKR_PORT", "not-an-int")

    with pytest.raises(BrokerConfigError, match="IBKR_PORT must be an integer"):
        load_broker_config(path)


def test_validation_rejects_live_orders(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["live_orders_enabled"] = True

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="live_orders_enabled must be false"):
        load_broker_config(path)


def test_validation_rejects_order_placement(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["allow_order_placement"] = True

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="allow_order_placement must be false"):
        load_broker_config(path)


def test_validation_rejects_non_paper_mode(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_only"] = False

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="paper_only must be true"):
        load_broker_config(path)


def test_validation_rejects_bad_strategy(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["default_strategy"] = "unknown_strategy"

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="approved strategy universe"):
        load_broker_config(path)


def test_validation_rejects_forbidden_strategy_model(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["default_strategy"] = "msvol"

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="forbidden active model"):
        load_broker_config(path)


def test_validation_rejects_options_contract_mismatch(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["allow_options"] = False
    config_data["paper_sizing"]["max_contracts"] = 1

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="max_contracts must be 0"):
        load_broker_config(path)


def test_validation_rejects_naked_short_options(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["allow_naked_short_options"] = True

    path = _write_config(tmp_path, config_data)

    with pytest.raises(
        BrokerConfigError,
        match="allow_naked_short_options must be false",
    ):
        load_broker_config(path)


def test_validation_rejects_margin_usage(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_margin_usage"] = 0.5

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="max_margin_usage must be 0.0"):
        load_broker_config(path)


def test_validation_rejects_missing_output_key(tmp_path: Path) -> None:
    config_data = _base_config()
    del config_data["outputs"]["run_metadata"]

    path = _write_config(tmp_path, config_data)

    with pytest.raises(BrokerConfigError, match="outputs missing required keys"):
        load_broker_config(path)


def test_get_market_signal_path(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_config())
    config = load_broker_config(path)

    us_path = get_market_signal_path(config, "US")
    india_path = get_market_signal_path(config, "india")

    assert us_path == Path("data/processed/us_strategy_signals.parquet")
    assert india_path == Path("data/processed/india_strategy_signals.parquet")


def test_get_market_signal_path_rejects_unknown_market(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_config())
    config = load_broker_config(path)

    with pytest.raises(BrokerConfigError, match="not allowed"):
        get_market_signal_path(config, "EUROPE")


def test_get_output_paths(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_config())
    config = load_broker_config(path)

    output_paths = get_output_paths(config)

    assert output_paths["latest_signal_table"] == Path(
        "reports/tables/phase_11/daily_paper_signal.csv"
    )
    assert output_paths["broker_metadata"] == Path(
        "reports/tables/phase_11/broker_metadata.json"
    )


def test_config_hash_is_stable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _base_config())
    config = load_broker_config(path)

    first_hash = compute_config_hash(config)
    second_hash = compute_config_hash(config)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_write_config_snapshot_redacts_env_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(tmp_path, _base_config())
    monkeypatch.setenv("IBKR_PAPER_ACCOUNT", "DU1234567")

    config = load_broker_config(path)
    snapshot_path = tmp_path / "snapshot.yaml"

    written = write_config_snapshot(config, snapshot_path)

    assert written == snapshot_path
    text = snapshot_path.read_text(encoding="utf-8")
    assert "DU1234567" not in text
    assert "REDACTED_IBKR_PAPER_ACCOUNT" in text


def test_ensure_output_directories(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["outputs"] = {
        "broker_cache_dir": str(tmp_path / "data" / "broker_cache"),
        "tables_dir": str(tmp_path / "reports" / "tables" / "phase_11"),
        "logs_dir": str(tmp_path / "logs"),
        "latest_signal_table": str(tmp_path / "reports" / "tables" / "phase_11" / "daily_paper_signal.csv"),
        "paper_order_intents": str(tmp_path / "reports" / "tables" / "phase_11" / "paper_order_intents.csv"),
        "risk_check_report": str(tmp_path / "reports" / "tables" / "phase_11" / "risk_check_report.csv"),
        "broker_metadata": str(tmp_path / "reports" / "tables" / "phase_11" / "broker_metadata.json"),
        "run_metadata": str(tmp_path / "reports" / "tables" / "phase_11" / "run_metadata.json"),
        "config_snapshot": str(tmp_path / "reports" / "tables" / "phase_11" / "ibkr_paper_config_snapshot.yaml"),
    }

    path = _write_config(tmp_path, config_data)
    config = load_broker_config(path)

    ensure_output_directories(config)

    assert (tmp_path / "data" / "broker_cache").exists()
    assert (tmp_path / "reports" / "tables" / "phase_11").exists()
    assert (tmp_path / "logs").exists()