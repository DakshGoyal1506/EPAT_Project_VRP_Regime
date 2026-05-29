from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.contracts import BrokerInstrumentRegistry
from vrp.broker.market_data import quote_from_mapping
from vrp.broker.paper_sizing import (
    PaperSizingError,
    build_paper_sizing,
    calculate_paper_target_notional,
    calculate_share_quantity,
    validate_no_phase10_sizing_inputs,
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


def test_calculate_paper_target_notional() -> None:
    assert calculate_paper_target_notional(-0.6, 10000) == 6000


def test_calculate_paper_target_notional_zero() -> None:
    assert calculate_paper_target_notional(0.0, 10000) == 0


def test_calculate_paper_target_notional_rejects_positive_exposure() -> None:
    with pytest.raises(PaperSizingError, match="between -1.0 and 0.0"):
        calculate_paper_target_notional(0.1, 10000)


def test_calculate_paper_target_notional_rejects_too_negative_exposure() -> None:
    with pytest.raises(PaperSizingError, match="between -1.0 and 0.0"):
        calculate_paper_target_notional(-1.1, 10000)


def test_calculate_share_quantity_floor() -> None:
    quantity = calculate_share_quantity(
        6000,
        90,
        round_shares=True,
        allow_fractional_shares=False,
    )

    assert quantity == 66


def test_calculate_share_quantity_fractional() -> None:
    quantity = calculate_share_quantity(
        6000,
        90,
        round_shares=True,
        allow_fractional_shares=True,
    )

    assert quantity == pytest.approx(66.6666666667)


def test_calculate_share_quantity_rejects_bad_price() -> None:
    with pytest.raises(PaperSizingError, match="reference_price must be positive"):
        calculate_share_quantity(
            6000,
            0,
            round_shares=True,
            allow_fractional_shares=False,
        )


def test_build_paper_sizing_notional_only_without_quote(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")

    sizing = build_paper_sizing(
        target_exposure=-0.6,
        contract=contract,
        config=config,
    )

    assert sizing.paper_target_notional == 6000
    assert sizing.reference_price is None
    assert sizing.paper_quantity is None
    assert sizing.sizing_status == "SIZED_NOTIONAL_ONLY"
    assert sizing.used_phase10_performance is False


def test_build_paper_sizing_with_quote(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 89.5,
            "ask": 90.5,
            "last": 90,
        }
    )

    sizing = build_paper_sizing(
        target_exposure=-0.6,
        contract=contract,
        config=config,
        quote=quote,
    )

    assert sizing.reference_price == 90
    assert sizing.paper_quantity == 66
    assert sizing.sizing_status == "SIZED_NOTIONAL_AND_QUANTITY"


def test_build_paper_sizing_zero_exposure(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = BrokerInstrumentRegistry().get("US", "SPY")
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 100,
        }
    )

    sizing = build_paper_sizing(
        target_exposure=0.0,
        contract=contract,
        config=config,
        quote=quote,
    )

    assert sizing.paper_target_notional == 0
    assert sizing.paper_quantity == 0
    assert sizing.sizing_status == "ZERO_NOTIONAL"


def test_validate_no_phase10_sizing_inputs_rejects_forbidden_key() -> None:
    with pytest.raises(PaperSizingError, match="Phase 10 performance"):
        validate_no_phase10_sizing_inputs({"sharpe": 1.2})


def test_validate_no_phase10_sizing_inputs_allows_clean_payload() -> None:
    validate_no_phase10_sizing_inputs(
        {
            "target_exposure": -0.5,
            "paper_notional_per_full_exposure": 10000,
        }
    )