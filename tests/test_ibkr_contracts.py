from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.contracts import (
    BLOCKED,
    PAPER_ALLOWED,
    PAPER_INSPECTION_ONLY,
    PAPER_SIGNAL_ONLY,
    BrokerInstrumentRegistry,
    ContractSpec,
    ContractValidationError,
    InstrumentMapping,
    choose_paper_proxy_for_action,
    contract_status_reason,
    get_allowed_contracts,
    get_default_proxy_contract,
    get_short_vol_proxy_candidates,
    is_broker_intent_eligible,
    validate_contract_allowed,
    validate_contract_not_blocked,
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


def _load_config(tmp_path: Path, config_data: dict | None = None):
    return load_broker_config(_write_config(tmp_path, config_data or _base_config()))


def test_registry_contains_default_contracts() -> None:
    registry = BrokerInstrumentRegistry()

    assert registry.get("US", "SPY").symbol == "SPY"
    assert registry.get("US", "VXX").symbol == "VXX"
    assert registry.get("US", "SVXY").symbol == "SVXY"
    assert registry.get("INDIA", "NIFTY_PROXY_MANUAL_ONLY").symbol == (
        "NIFTY_PROXY_MANUAL_ONLY"
    )


def test_registry_symbols_are_market_scoped() -> None:
    registry = BrokerInstrumentRegistry()

    assert registry.symbols_for_market("US") == ("SPY", "SVXY", "VXX")
    assert registry.symbols_for_market("INDIA") == ("NIFTY_PROXY_MANUAL_ONLY",)


def test_registry_rejects_unknown_contract() -> None:
    registry = BrokerInstrumentRegistry()

    with pytest.raises(ContractValidationError, match="Unknown contract"):
        registry.get("US", "UNKNOWN")


def test_contract_spec_as_dict() -> None:
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    record = contract.as_dict()

    assert record["symbol"] == "SPY"
    assert record["sec_type"] == "STK"
    assert record["exchange"] == "SMART"
    assert record["currency"] == "USD"
    assert record["market"] == "US"


def test_instrument_mapping_from_config_us(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    mapping = InstrumentMapping.from_config("US", config)

    assert mapping.market == "US"
    assert mapping.research_underlying == "SPX_or_SPY"
    assert mapping.default_paper_proxy == "SPY"
    assert mapping.short_vol_proxy_candidates == ("VXX", "SVXY")


def test_get_allowed_contracts_us(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    contracts = get_allowed_contracts("US", config)
    symbols = {contract.symbol for contract in contracts}

    assert symbols == {"SPY", "VXX", "SVXY"}
    assert all(contract.tradability_status == PAPER_ALLOWED for contract in contracts)


def test_get_default_proxy_contract_us(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    contract = get_default_proxy_contract("US", config)

    assert contract.symbol == "SPY"
    assert contract.market == "US"
    assert contract.tradability_status == PAPER_ALLOWED


def test_get_short_vol_proxy_candidates_us(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    candidates = get_short_vol_proxy_candidates("US", config)

    assert [contract.symbol for contract in candidates] == ["VXX", "SVXY"]


def test_choose_paper_proxy_for_short_vol_uses_first_candidate(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    contract = choose_paper_proxy_for_action(
        "US",
        "PAPER_SHORT_VOL_INTENT",
        config,
    )

    assert contract.symbol == "VXX"


def test_choose_paper_proxy_for_other_action_uses_default(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    contract = choose_paper_proxy_for_action(
        "US",
        "STAY_FLAT",
        config,
    )

    assert contract.symbol == "SPY"


def test_validate_contract_allowed_accepts_us_allowed_contract(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")

    assert validate_contract_allowed(contract, config) == contract


def test_validate_contract_allowed_rejects_market_not_allowed(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = ContractSpec(
        symbol="EWG",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        multiplier=1.0,
        market="EUROPE",
        tradability_status=PAPER_ALLOWED,
        notes="Not configured.",
    )

    with pytest.raises(ContractValidationError, match="not allowed"):
        validate_contract_allowed(contract, config)


def test_validate_contract_allowed_rejects_symbol_not_allowed(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = ContractSpec(
        symbol="AAPL",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        multiplier=1.0,
        market="US",
        tradability_status=PAPER_ALLOWED,
        notes="Not part of Phase 11 allowed list.",
    )

    with pytest.raises(ContractValidationError, match="allowed_instruments"):
        validate_contract_allowed(contract, config)


def test_validate_contract_not_blocked_rejects_option_contract(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = ContractSpec(
        symbol="SPX_OPTION_PAPER_ONLY",
        sec_type="OPT",
        exchange="CBOE",
        currency="USD",
        multiplier=100.0,
        market="US",
        tradability_status=PAPER_INSPECTION_ONLY,
        notes="Option inspection placeholder.",
    )

    with pytest.raises(ContractValidationError, match="Options are blocked"):
        validate_contract_not_blocked(contract, config)


def test_validate_contract_not_blocked_rejects_future_contract(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = ContractSpec(
        symbol="VIX_FUTURE_PAPER_ONLY",
        sec_type="FUT",
        exchange="CFE",
        currency="USD",
        multiplier=1000.0,
        market="US",
        tradability_status=PAPER_INSPECTION_ONLY,
        notes="Future inspection placeholder.",
    )

    with pytest.raises(ContractValidationError, match="Futures are blocked"):
        validate_contract_not_blocked(contract, config)


def test_validate_contract_not_blocked_rejects_blocked_status(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = ContractSpec(
        symbol="BLOCKED_THING",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        multiplier=1.0,
        market="US",
        tradability_status=BLOCKED,
        notes="Blocked placeholder.",
    )

    with pytest.raises(ContractValidationError, match="explicitly blocked"):
        validate_contract_not_blocked(contract, config)


def test_india_default_proxy_is_blocked_without_manual_verification(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)

    with pytest.raises(ContractValidationError, match="signal-only by default"):
        get_default_proxy_contract("INDIA", config)


def test_india_default_proxy_allowed_after_manual_verification(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["manual_overrides"]["INDIA"]["manual_instrument_verified"] = True

    config = _load_config(tmp_path, config_data)

    contract = get_default_proxy_contract("INDIA", config)

    assert contract.symbol == "NIFTY_PROXY_MANUAL_ONLY"
    assert contract.tradability_status == PAPER_SIGNAL_ONLY


def test_india_not_broker_intent_eligible_even_after_manual_verification(
    tmp_path: Path,
) -> None:
    config_data = _base_config()
    config_data["manual_overrides"]["INDIA"]["manual_instrument_verified"] = True

    config = _load_config(tmp_path, config_data)
    contract = get_default_proxy_contract("INDIA", config)

    assert is_broker_intent_eligible(contract, config) is False


def test_us_contract_is_broker_intent_eligible(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")

    assert is_broker_intent_eligible(contract, config) is True


def test_contract_status_reason_for_us_allowed(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")

    reason = contract_status_reason(contract, config)

    assert "eligible for paper intent" in reason


def test_contract_status_reason_for_india_signal_only(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("INDIA", "NIFTY_PROXY_MANUAL_ONLY")

    reason = contract_status_reason(contract, config)

    assert "signal-only by default" in reason


def test_config_missing_instrument_mapping_rejected(tmp_path: Path) -> None:
    config_data = _base_config()
    del config_data["instrument_mapping"]

    config = _load_config(tmp_path, config_data)

    with pytest.raises(ContractValidationError, match="instrument_mapping"):
        InstrumentMapping.from_config("US", config)


def test_config_missing_allowed_instruments_rejected(tmp_path: Path) -> None:
    config_data = _base_config()
    del config_data["allowed_instruments"]

    config = _load_config(tmp_path, config_data)
    contract = BrokerInstrumentRegistry().get("US", "SPY")

    with pytest.raises(ContractValidationError, match="allowed_instruments"):
        validate_contract_allowed(contract, config)