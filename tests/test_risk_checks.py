from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import BrokerConfig, load_broker_config
from vrp.broker.contracts import BrokerInstrumentRegistry
from vrp.broker.market_data import missing_quote, quote_from_mapping
from vrp.broker.paper_sizing import build_paper_sizing
from vrp.broker.risk_checks import (
    ALLOWED_PAPER_INTENT,
    BLOCKED_BY_KILL_SWITCH,
    BLOCKED_BROKER_DATA,
    BLOCKED_CONFIG_SAFETY,
    BLOCKED_RISK_LIMIT,
    BLOCKED_STALE_SIGNAL,
    NO_SIGNAL,
    STAY_FLAT,
    run_phase11_risk_checks,
    write_risk_check_report,
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


def _load_config(tmp_path: Path, config_data: dict | None = None) -> BrokerConfig:
    return load_broker_config(_write_config(tmp_path, config_data or _base_config()))


def _valid_inputs(tmp_path: Path, config_data: dict | None = None):
    config = _load_config(tmp_path, config_data)
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99.95,
            "ask": 100.05,
            "quote_age_seconds": 30,
        }
    )
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )
    return config, contract, quote, sizing


def test_no_signal_terminal_status(tmp_path: Path) -> None:
    config, contract, quote, sizing = _valid_inputs(tmp_path)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action=NO_SIGNAL,
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == NO_SIGNAL
    assert summary.intent_allowed_before_kill_switch is False


def test_stay_flat_terminal_status(tmp_path: Path) -> None:
    config, contract, quote, sizing = _valid_inputs(tmp_path)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action=STAY_FLAT,
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == STAY_FLAT
    assert summary.intent_allowed_before_kill_switch is False


def test_default_blocks_by_kill_switch(tmp_path: Path) -> None:
    config, contract, quote, sizing = _valid_inputs(tmp_path)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_BY_KILL_SWITCH
    assert summary.intent_allowed_before_kill_switch is True
    assert summary.intent_allowed_after_kill_switch is False
    assert summary.live_order_sent is False


def test_allowed_when_kill_switch_off(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["kill_switch"] = False
    config, contract, quote, sizing = _valid_inputs(tmp_path, config_data)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == ALLOWED_PAPER_INTENT
    assert summary.intent_allowed_before_kill_switch is True
    assert summary.intent_allowed_after_kill_switch is True


def test_stale_signal_blocks_before_other_checks(tmp_path: Path) -> None:
    config, contract, quote, sizing = _valid_inputs(tmp_path)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
        signal_is_stale=True,
    )

    assert summary.final_status == BLOCKED_STALE_SIGNAL
    assert summary.intent_allowed_before_kill_switch is False


def test_max_notional_blocks(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_notional"] = 1000
    config, contract, quote, sizing = _valid_inputs(tmp_path, config_data)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_RISK_LIMIT
    assert "notional" in summary.primary_block_reason


def test_max_shares_blocks(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_shares"] = 1
    config, contract, quote, sizing = _valid_inputs(tmp_path, config_data)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_RISK_LIMIT
    assert "quantity" in summary.primary_block_reason


def test_missing_quote_allowed_by_default_then_kill_switch_blocks(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    quote = missing_quote(symbol="SPY", market="US")
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.quote_status == "QUOTE_MISSING"
    assert summary.final_status == BLOCKED_BY_KILL_SWITCH


def test_missing_quote_blocks_when_required(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["risk_checks"]["require_quote_for_allowed_intent"] = True

    config = _load_config(tmp_path, config_data)
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    quote = missing_quote(symbol="SPY", market="US")
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_BROKER_DATA


def test_stale_quote_blocks(tmp_path: Path) -> None:
    config, contract, _, _ = _valid_inputs(tmp_path)
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99.95,
            "ask": 100.05,
            "quote_age_seconds": 301,
        }
    )
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_RISK_LIMIT
    assert "stale" in summary.primary_block_reason


def test_wide_spread_blocks(tmp_path: Path) -> None:
    config, contract, _, _ = _valid_inputs(tmp_path)
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99,
            "ask": 101,
            "quote_age_seconds": 30,
        }
    )
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_RISK_LIMIT
    assert "spread" in summary.primary_block_reason


def test_india_blocks_as_signal_only(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("INDIA", "NIFTY_PROXY_MANUAL_ONLY")
    quote = missing_quote(symbol="NIFTY_PROXY_MANUAL_ONLY", market="INDIA")
    sizing = build_paper_sizing(
        target_exposure=-0.5,
        contract=contract,
        config=config,
        quote=quote,
    )

    summary = run_phase11_risk_checks(
        market="INDIA",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    assert summary.final_status == BLOCKED_RISK_LIMIT
    assert "signal-only" in summary.primary_block_reason or "not eligible" in summary.primary_block_reason


def test_write_risk_check_report(tmp_path: Path) -> None:
    config, contract, quote, sizing = _valid_inputs(tmp_path)

    summary = run_phase11_risk_checks(
        market="US",
        recommended_action="PAPER_SHORT_VOL_INTENT",
        contract=contract,
        sizing=sizing,
        config=config,
        quote=quote,
    )

    output_path = tmp_path / "risk_check_report.csv"
    written = write_risk_check_report(summary, output_path)

    assert written == output_path
    assert output_path.exists()

    report = pd.read_csv(output_path)
    assert len(report) >= 1
    assert "check_name" in report.columns
    assert "final_status" in report.columns
    assert report["final_status"].iloc[0] == BLOCKED_BY_KILL_SWITCH