from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.market_data import (
    QUOTE_AVAILABLE,
    QUOTE_INVALID,
    QUOTE_MISSING,
    QUOTE_SPREAD_TOO_WIDE,
    QUOTE_STALE,
    MarketDataError,
    evaluate_quote_readiness,
    missing_quote,
    quote_from_mapping,
    quote_is_stale,
    quote_record_for_report,
    quote_spread_too_wide,
    reference_price_from_quote,
    validate_quote_snapshot,
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


def test_quote_from_mapping_computes_mid_spread_and_age() -> None:
    now = datetime(2026, 5, 20, 12, 5, 0, tzinfo=timezone.utc)

    quote = quote_from_mapping(
        {
            "symbol": "spy",
            "market": "us",
            "bid": 499.0,
            "ask": 501.0,
            "last": 500.0,
            "timestamp_utc": "2026-05-20T12:00:00+00:00",
            "source": "test",
        },
        now_utc=now,
    )

    assert quote.symbol == "SPY"
    assert quote.market == "US"
    assert quote.bid == 499.0
    assert quote.ask == 501.0
    assert quote.mid == 500.0
    assert quote.spread_bps == 40.0
    assert quote.quote_age_seconds == 300.0
    assert quote.status == QUOTE_AVAILABLE


def test_quote_from_mapping_accepts_z_timestamp() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 101,
            "timestamp_utc": "2026-05-20T12:00:00Z",
        },
        now_utc=datetime(2026, 5, 20, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert quote.timestamp_utc == datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    assert quote.quote_age_seconds == 60.0


def test_quote_from_mapping_accepts_explicit_age() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 101,
            "quote_age_seconds": 12,
        }
    )

    assert quote.quote_age_seconds == 12.0


def test_quote_from_mapping_uses_last_without_bid_ask() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": None,
            "ask": None,
            "last": 500,
        }
    )

    assert quote.last == 500.0
    assert quote.mid is None
    assert quote.spread_bps is None
    assert reference_price_from_quote(quote) == 500.0


def test_quote_from_mapping_rejects_missing_symbol() -> None:
    with pytest.raises(MarketDataError, match="missing symbol"):
        quote_from_mapping({"market": "US", "bid": 1, "ask": 2})


def test_quote_from_mapping_rejects_missing_market() -> None:
    with pytest.raises(MarketDataError, match="missing market"):
        quote_from_mapping({"symbol": "SPY", "bid": 1, "ask": 2})


def test_quote_from_mapping_rejects_negative_bid() -> None:
    with pytest.raises(MarketDataError, match="bid must be non-negative"):
        quote_from_mapping({"symbol": "SPY", "market": "US", "bid": -1, "ask": 2})


def test_quote_from_mapping_rejects_negative_ask() -> None:
    with pytest.raises(MarketDataError, match="ask must be non-negative"):
        quote_from_mapping({"symbol": "SPY", "market": "US", "bid": 1, "ask": -2})


def test_quote_from_mapping_rejects_ask_below_bid() -> None:
    with pytest.raises(MarketDataError, match="ask must be greater"):
        quote_from_mapping({"symbol": "SPY", "market": "US", "bid": 102, "ask": 101})


def test_quote_from_mapping_rejects_bad_numeric_value() -> None:
    with pytest.raises(MarketDataError, match="must be numeric"):
        quote_from_mapping({"symbol": "SPY", "market": "US", "bid": "bad", "ask": 101})


def test_quote_from_mapping_rejects_bad_timestamp() -> None:
    with pytest.raises(MarketDataError, match="not parseable"):
        quote_from_mapping(
            {
                "symbol": "SPY",
                "market": "US",
                "bid": 100,
                "ask": 101,
                "timestamp_utc": "not-a-time",
            }
        )


def test_missing_quote_is_structured() -> None:
    quote = missing_quote(symbol="SPY", market="US")

    assert quote.status == QUOTE_MISSING
    assert quote.symbol == "SPY"
    assert quote.market == "US"
    assert quote.bid is None
    assert quote.ask is None
    assert quote.mid is None


def test_validate_quote_snapshot_allows_missing_quote() -> None:
    quote = missing_quote(symbol="SPY", market="US")

    assert validate_quote_snapshot(quote) == quote


def test_reference_price_prefers_mid() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99,
            "ask": 101,
            "last": 150,
        }
    )

    assert reference_price_from_quote(quote) == 100.0


def test_reference_price_returns_none_for_missing_quote() -> None:
    quote = missing_quote(symbol="SPY", market="US")

    assert reference_price_from_quote(quote) is None


def test_quote_is_stale() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 101,
            "quote_age_seconds": 301,
        }
    )

    assert quote_is_stale(quote, max_quote_age_seconds=300) is True


def test_quote_spread_too_wide() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99,
            "ask": 101,
        }
    )

    assert quote_spread_too_wide(quote, max_bid_ask_spread_bps=100) is True


def test_evaluate_quote_readiness_passes_valid_quote(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99.95,
            "ask": 100.05,
            "quote_age_seconds": 30,
        }
    )

    result = evaluate_quote_readiness(quote, config)

    assert result.quote_status == QUOTE_AVAILABLE
    assert result.quote_available is True
    assert result.quote_valid is True
    assert result.blocks_allowed_intent is False


def test_evaluate_quote_readiness_allows_missing_quote_by_default(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    quote = missing_quote(symbol="SPY", market="US")

    result = evaluate_quote_readiness(quote, config)

    assert result.quote_status == QUOTE_MISSING
    assert result.quote_available is False
    assert result.quote_valid is False
    assert result.blocks_allowed_intent is False


def test_evaluate_quote_readiness_blocks_missing_quote_when_required(
    tmp_path: Path,
) -> None:
    config_data = _base_config()
    config_data["risk_checks"]["require_quote_for_allowed_intent"] = True

    config = _load_config(tmp_path, config_data)

    result = evaluate_quote_readiness(None, config)

    assert result.quote_status == QUOTE_MISSING
    assert result.blocks_allowed_intent is True
    assert "blocks missing quote" in result.block_reason


def test_evaluate_quote_readiness_blocks_stale_quote(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 101,
            "quote_age_seconds": 301,
        }
    )

    result = evaluate_quote_readiness(quote, config)

    assert result.quote_status == QUOTE_STALE
    assert result.is_stale is True
    assert result.blocks_allowed_intent is True


def test_evaluate_quote_readiness_blocks_wide_spread(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 99,
            "ask": 101,
            "quote_age_seconds": 30,
        }
    )

    result = evaluate_quote_readiness(quote, config)

    assert result.quote_status == QUOTE_SPREAD_TOO_WIDE
    assert result.spread_too_wide is True
    assert result.blocks_allowed_intent is True


def test_evaluate_quote_readiness_reports_none_quote(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    result = evaluate_quote_readiness(None, config)

    assert result.quote_status == QUOTE_MISSING
    assert result.quote_available is False


def test_quote_record_for_report_handles_none() -> None:
    record = quote_record_for_report(None, symbol="SPY", market="US")

    assert record["symbol"] == "SPY"
    assert record["market"] == "US"
    assert record["status"] == QUOTE_MISSING


def test_quote_record_for_report_handles_quote() -> None:
    quote = quote_from_mapping(
        {
            "symbol": "SPY",
            "market": "US",
            "bid": 100,
            "ask": 101,
        }
    )

    record = quote_record_for_report(quote, symbol="SPY", market="US")

    assert record["symbol"] == "SPY"
    assert record["mid"] == 100.5
    assert record["status"] == QUOTE_AVAILABLE


def test_evaluate_quote_readiness_invalid_quote_object_is_blocked(
    tmp_path: Path,
) -> None:
    from vrp.broker.market_data import QuoteSnapshot

    config = _load_config(tmp_path)
    quote = QuoteSnapshot(
        symbol="SPY",
        market="US",
        bid=101,
        ask=100,
        last=None,
        mid=100.5,
        timestamp_utc=None,
        quote_age_seconds=None,
        spread_bps=-99.5,
        source="bad-test",
        status=QUOTE_AVAILABLE,
    )

    result = evaluate_quote_readiness(quote, config)

    assert result.quote_status == QUOTE_INVALID
    assert result.blocks_allowed_intent is True