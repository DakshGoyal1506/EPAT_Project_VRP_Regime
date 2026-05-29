from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.market_data import missing_quote, quote_from_mapping
from vrp.broker.paper_state import (
    PaperPositionState,
    get_latest_position_state,
    has_prior_short_state,
    write_paper_position_state,
)
from vrp.broker.paper_trader import (
    PAPER_ORDER_INTENT_COLUMNS,
    build_paper_order_intent,
    infer_paper_side,
    publish_paper_order_intent,
    read_daily_paper_signal,
    select_contract_for_paper_signal,
    write_paper_order_intents,
)
from vrp.broker.signal_publisher import (
    DailyPaperSignal,
    NO_SIGNAL,
    PAPER_SHORT_VOL_INTENT,
    STAY_FLAT,
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


def _daily_signal(
    *,
    action: str = PAPER_SHORT_VOL_INTENT,
    exposure: float = -0.6,
    market: str = "US",
    final_status: str = "BLOCKED_BY_KILL_SWITCH",
) -> DailyPaperSignal:
    return DailyPaperSignal(
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
        market=market,
        strategy_name="mar_prob_linear_carry",
        signal_observation_date="2026-05-22",
        target_trade_date="2026-05-23",
        target_exposure=exposure,
        strategy_available=True,
        recommended_action=action,
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
        final_status=final_status,
        live_order_sent=False,
        research_proxy_warning=RESEARCH_PROXY_WARNING,
        source_signal_path="data/processed/us_strategy_signals.parquet",
        source_signal_mtime_utc="2026-05-24T00:00:00+00:00",
        source_strategy_row={},
    )


def test_select_contract_for_short_vol_uses_vxx(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    contract = select_contract_for_paper_signal(
        market="US",
        recommended_action=PAPER_SHORT_VOL_INTENT,
        config=config,
    )

    assert contract.symbol == "VXX"


def test_infer_side_for_vxx_short_vol(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    contract = select_contract_for_paper_signal(
        market="US",
        recommended_action=PAPER_SHORT_VOL_INTENT,
        config=config,
    )

    assert infer_paper_side(
        recommended_action=PAPER_SHORT_VOL_INTENT,
        contract=contract,
    ) == "SELL"


def test_build_paper_order_intent_no_signal_returns_none(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal(action=NO_SIGNAL, exposure=-0.6, final_status=NO_SIGNAL)

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
    )

    assert result.intent is None
    assert "NO_SIGNAL" in result.reason


def test_build_paper_order_intent_stay_flat_returns_none(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal(action=STAY_FLAT, exposure=0.0, final_status=STAY_FLAT)

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
    )

    assert result.intent is None
    assert "STAY_FLAT" in result.reason


def test_build_paper_order_intent_default_blocks_by_kill_switch(tmp_path: Path) -> None:
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

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=quote,
    )

    assert result.intent is not None
    assert result.contract is not None
    assert result.sizing is not None
    assert result.risk_summary is not None

    intent = result.intent
    assert intent.symbol == "VXX"
    assert intent.side == "SELL"
    assert intent.paper_target_notional == 6000
    assert intent.reference_price == 15
    assert intent.paper_quantity == 400
    assert intent.final_status == "BLOCKED_RISK_LIMIT"
    assert intent.live_order_sent is False


def test_build_paper_order_intent_blocks_by_kill_switch_when_size_within_limit(
    tmp_path: Path,
) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_shares"] = 1000
    config = _load_config(tmp_path, config_data)
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

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=quote,
    )

    assert result.intent is not None
    assert result.intent.final_status == "BLOCKED_BY_KILL_SWITCH"
    assert result.intent.intent_allowed_before_kill_switch is True
    assert result.intent.intent_allowed_after_kill_switch is False


def test_build_paper_order_intent_allowed_when_kill_switch_off_and_limits_ok(
    tmp_path: Path,
) -> None:
    config_data = _base_config()
    config_data["kill_switch"] = False
    config_data["paper_sizing"]["max_shares"] = 1000
    config = _load_config(tmp_path, config_data)
    signal = _daily_signal(final_status="BROKER_INSPECTION_ONLY")
    quote = quote_from_mapping(
        {
            "symbol": "VXX",
            "market": "US",
            "bid": 14.95,
            "ask": 15.05,
            "quote_age_seconds": 30,
        }
    )

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=quote,
    )

    assert result.intent is not None
    assert result.intent.final_status == "ALLOWED_PAPER_INTENT"
    assert result.intent.intent_allowed_after_kill_switch is True
    assert result.intent.live_order_sent is False


def test_build_paper_order_intent_missing_quote_keeps_quantity_blank(
    tmp_path: Path,
) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_shares"] = 1000
    config = _load_config(tmp_path, config_data)
    signal = _daily_signal()

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=missing_quote(symbol="VXX", market="US"),
    )

    assert result.intent is not None
    assert result.intent.reference_price is None
    assert result.intent.paper_quantity is None
    assert result.intent.final_status == "BLOCKED_BY_KILL_SWITCH"


def test_write_paper_order_intents_empty_writes_header(tmp_path: Path) -> None:
    output_path = tmp_path / "paper_order_intents.csv"

    write_paper_order_intents([], output_path)

    frame = pd.read_csv(output_path)
    assert list(frame.columns) == PAPER_ORDER_INTENT_COLUMNS
    assert frame.empty


def test_write_paper_order_intents_writes_row(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_shares"] = 1000
    config = _load_config(tmp_path, config_data)
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

    result = build_paper_order_intent(
        daily_signal=signal,
        config=config,
        quote=quote,
    )

    output_path = tmp_path / "paper_order_intents.csv"
    # result.intent may be None; ensure we pass a list of PaperOrderIntent only
    if result.intent is not None:
        write_paper_order_intents([result.intent], output_path)
    else:
        write_paper_order_intents([], output_path)

    frame = pd.read_csv(output_path)
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "VXX"
    assert frame.loc[0, "side"] == "SELL"
    assert frame.loc[0, "live_order_sent"] == False


def test_read_daily_paper_signal(tmp_path: Path) -> None:
    signal = _daily_signal()
    path = tmp_path / "daily_paper_signal.csv"
    pd.DataFrame([signal.as_dict()]).to_csv(path, index=False)

    loaded = read_daily_paper_signal(path)

    assert loaded.market == "US"
    assert loaded.strategy_name == "mar_prob_linear_carry"
    assert loaded.recommended_action == PAPER_SHORT_VOL_INTENT
    assert loaded.target_exposure == -0.6


def test_publish_paper_order_intent_writes_files(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["paper_sizing"]["max_shares"] = 1000
    config = _load_config(tmp_path, config_data)
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

    intent_path = tmp_path / "paper_order_intents.csv"
    risk_path = tmp_path / "risk_check_report.csv"

    result = publish_paper_order_intent(
        config=config,
        daily_signal=signal,
        quote=quote,
        output_path=intent_path,
        risk_report_path=risk_path,
    )

    assert result.intent is not None
    assert intent_path.exists()
    assert risk_path.exists()

    intent_frame = pd.read_csv(intent_path)
    risk_frame = pd.read_csv(risk_path)

    assert intent_frame.loc[0, "symbol"] == "VXX"
    assert "check_name" in risk_frame.columns


def test_publish_paper_order_intent_no_signal_writes_empty_intent_file(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)
    signal = _daily_signal(action=NO_SIGNAL, exposure=-0.6, final_status=NO_SIGNAL)

    intent_path = tmp_path / "paper_order_intents.csv"

    result = publish_paper_order_intent(
        config=config,
        daily_signal=signal,
        output_path=intent_path,
    )

    assert result.intent is None
    frame = pd.read_csv(intent_path)
    assert frame.empty
    assert list(frame.columns) == PAPER_ORDER_INTENT_COLUMNS


def test_position_state_load_and_prior_short(tmp_path: Path) -> None:
    state_path = tmp_path / "paper_position_state.csv"

    state = PaperPositionState(
        updated_at_utc="2026-05-24T00:00:00+00:00",
        market="US",
        strategy_name="mar_prob_linear_carry",
        symbol="VXX",
        target_trade_date="2026-05-23",
        target_exposure=-0.6,
        paper_quantity=400,
        side="SELL",
        status="BLOCKED_BY_KILL_SWITCH",
    )

    write_paper_position_state([state], state_path)

    loaded = get_latest_position_state(
        state_path,
        market="US",
        strategy_name="mar_prob_linear_carry",
    )

    assert loaded is not None
    assert loaded.symbol == "VXX"
    assert loaded.target_exposure == -0.6
    assert has_prior_short_state(
        state_path,
        market="US",
        strategy_name="mar_prob_linear_carry",
    ) is True