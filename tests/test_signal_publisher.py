from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import load_broker_config
from vrp.broker.signal_publisher import (
    BLOCKED_BY_KILL_SWITCH,
    BLOCKED_MISSING_SIGNAL,
    BLOCKED_STALE_SIGNAL,
    BROKER_INSPECTION_ONLY,
    NO_SIGNAL,
    PAPER_SHORT_VOL_INTENT,
    REDUCE_TO_ZERO,
    STAY_FLAT,
    build_daily_paper_signal,
    build_missing_signal_record,
    determine_signal_final_status,
    interpret_signal_action,
    publish_daily_paper_signal,
    write_daily_paper_signal,
)
from vrp.broker.signal_schema import check_signal_freshness, select_latest_signal


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
        "signal_schema": {
            "canonical_required_columns": [
                "target_trade_date",
                "target_exposure",
                "strategy_available",
                "blocked_reason",
                "decision_reason",
            ],
            "strategy_column_candidates": ["strategy_name", "strategy"],
            "optional_columns": [
                "market",
                "signal_observation_date",
                "signal_date",
                "source_model",
                "vrp_har_gk",
                "har_forecast_available",
                "state_name",
                "p_calm",
                "p_transition",
                "p_stress",
            ],
            "aliases": {
                "strategy": "strategy_name",
                "signal_date": "signal_observation_date",
            },
            "validation": {
                "target_exposure_min": -1.0,
                "target_exposure_max": 0.0,
                "latest_signal_sort_column": "target_trade_date",
                "require_latest_by_target_trade_date": True,
                "allow_boolean_like_strategy_available": True,
            },
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


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["US", "US", "US"],
            "strategy_name": [
                "mar_prob_linear_carry",
                "mar_prob_linear_carry",
                "mar_prob_linear_carry",
            ],
            "signal_observation_date": [
                "2026-05-20",
                "2026-05-21",
                "2026-05-22",
            ],
            "target_trade_date": [
                "2026-05-21",
                "2026-05-22",
                "2026-05-23",
            ],
            "target_exposure": [-0.25, -0.50, -0.60],
            "strategy_available": [True, True, True],
            "blocked_reason": ["", "", ""],
            "decision_reason": [
                "carry gate valid",
                "carry gate valid",
                "carry gate valid",
            ],
            "vrp_har_gk": [0.01, 0.02, 0.03],
            "har_forecast_available": [True, True, True],
            "state_name": ["calm", "calm", "calm"],
            "p_calm": [0.7, 0.8, 0.9],
            "p_transition": [0.2, 0.1, 0.05],
            "p_stress": [0.1, 0.1, 0.05],
        }
    )


def test_interpret_signal_action_no_signal() -> None:
    row = {
        "strategy_available": False,
        "target_exposure": -0.5,
    }

    assert interpret_signal_action(row) == NO_SIGNAL


def test_interpret_signal_action_stay_flat() -> None:
    row = {
        "strategy_available": True,
        "target_exposure": 0.0,
    }

    assert interpret_signal_action(row) == STAY_FLAT


def test_interpret_signal_action_reduce_to_zero_with_state() -> None:
    row = {
        "strategy_available": True,
        "target_exposure": 0.0,
    }

    assert (
        interpret_signal_action(
            row,
            use_position_state=True,
            prior_target_exposure=-0.5,
        )
        == REDUCE_TO_ZERO
    )


def test_interpret_signal_action_short_vol_intent() -> None:
    row = {
        "strategy_available": True,
        "target_exposure": -0.5,
    }

    assert interpret_signal_action(row) == PAPER_SHORT_VOL_INTENT


def test_determine_status_blocks_by_kill_switch(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    status, before, after = determine_signal_final_status(
        recommended_action=PAPER_SHORT_VOL_INTENT,
        config=config,
        freshness=None,
    )

    assert status == BLOCKED_BY_KILL_SWITCH
    assert before is True
    assert after is False


def test_determine_status_allows_inspection_when_kill_switch_off(tmp_path: Path) -> None:
    config_data = _base_config()
    config_data["kill_switch"] = False

    config = _load_config(tmp_path, config_data)

    status, before, after = determine_signal_final_status(
        recommended_action=PAPER_SHORT_VOL_INTENT,
        config=config,
        freshness=None,
    )

    assert status == BROKER_INSPECTION_ONLY
    assert before is True
    assert after is True


def test_determine_status_stale_signal_overrides_kill_switch(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    freshness = check_signal_freshness(
        {"target_trade_date": date(2026, 5, 1)},
        as_of_date=date(2026, 5, 20),
        max_signal_age_days=5,
        block_stale_signal=True,
    )

    status, before, after = determine_signal_final_status(
        recommended_action=PAPER_SHORT_VOL_INTENT,
        config=config,
        freshness=freshness,
    )

    assert status == BLOCKED_STALE_SIGNAL
    assert before is False
    assert after is False


def test_build_daily_paper_signal_default_blocks_by_kill_switch(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    signal_path = tmp_path / "signals.csv"
    _signal_frame().to_csv(signal_path, index=False)

    latest = select_latest_signal(_signal_frame(), strategy_name="mar_prob_linear_carry")
    freshness = check_signal_freshness(
        latest,
        as_of_date=date(2026, 5, 24),
        max_signal_age_days=5,
    )

    signal = build_daily_paper_signal(
        market="US",
        strategy_name="mar_prob_linear_carry",
        latest_signal=latest,
        config=config,
        signal_path=signal_path,
        freshness=freshness,
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert signal.market == "US"
    assert signal.strategy_name == "mar_prob_linear_carry"
    assert signal.target_trade_date == "2026-05-23"
    assert signal.target_exposure == -0.6
    assert signal.recommended_action == PAPER_SHORT_VOL_INTENT
    assert signal.paper_only is True
    assert signal.kill_switch is True
    assert signal.live_orders_enabled is False
    assert signal.allow_order_placement is False
    assert signal.intent_allowed_before_kill_switch is True
    assert signal.intent_allowed_after_kill_switch is False
    assert signal.final_status == BLOCKED_BY_KILL_SWITCH
    assert signal.live_order_sent is False
    assert "research-layer proxy units" in signal.research_proxy_warning


def test_build_daily_paper_signal_no_signal(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    frame = _signal_frame()
    frame.loc[2, "strategy_available"] = False
    frame.loc[2, "blocked_reason"] = "Phase 9 unavailable"
    frame.loc[2, "decision_reason"] = "HAR unavailable"

    latest = select_latest_signal(frame, strategy_name="mar_prob_linear_carry")

    signal = build_daily_paper_signal(
        market="US",
        strategy_name="mar_prob_linear_carry",
        latest_signal=latest,
        config=config,
        signal_path=tmp_path / "signals.csv",
    )

    assert signal.recommended_action == NO_SIGNAL
    assert signal.final_status == NO_SIGNAL
    assert signal.intent_allowed_before_kill_switch is False


def test_build_daily_paper_signal_stay_flat(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    frame = _signal_frame()
    frame.loc[2, "target_exposure"] = 0.0

    latest = select_latest_signal(frame, strategy_name="mar_prob_linear_carry")

    signal = build_daily_paper_signal(
        market="US",
        strategy_name="mar_prob_linear_carry",
        latest_signal=latest,
        config=config,
        signal_path=tmp_path / "signals.csv",
    )

    assert signal.recommended_action == STAY_FLAT
    assert signal.final_status == STAY_FLAT
    assert signal.intent_allowed_before_kill_switch is False


def test_build_missing_signal_record(tmp_path: Path) -> None:
    config = _load_config(tmp_path)

    signal = build_missing_signal_record(
        market="US",
        strategy_name="mar_prob_linear_carry",
        config=config,
        signal_path=tmp_path / "missing.csv",
        reason="file missing",
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert signal.recommended_action == NO_SIGNAL
    assert signal.final_status == BLOCKED_MISSING_SIGNAL
    assert signal.blocked_reason == "file missing"
    assert signal.live_order_sent is False


def test_write_daily_paper_signal(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    latest = select_latest_signal(_signal_frame(), strategy_name="mar_prob_linear_carry")

    signal = build_daily_paper_signal(
        market="US",
        strategy_name="mar_prob_linear_carry",
        latest_signal=latest,
        config=config,
        signal_path=tmp_path / "signals.csv",
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    out_path = tmp_path / "daily_paper_signal.csv"
    written = write_daily_paper_signal(signal, out_path)

    assert written == out_path
    assert out_path.exists()

    out = pd.read_csv(out_path)
    assert len(out) == 1
    assert out.loc[0, "market"] == "US"
    assert out.loc[0, "recommended_action"] == PAPER_SHORT_VOL_INTENT
    assert out.loc[0, "final_status"] == BLOCKED_BY_KILL_SWITCH

    source_row = json.loads(str(out.loc[0, "source_strategy_row"]))
    assert source_row["target_trade_date"] == "2026-05-23"


def test_publish_daily_paper_signal_writes_output(tmp_path: Path) -> None:
    signal_path = tmp_path / "signals.csv"
    _signal_frame().to_csv(signal_path, index=False)

    config = _load_config(tmp_path)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        strategy_name="mar_prob_linear_carry",
        signal_path=signal_path,
        output_path=output_path,
        as_of_date=date(2026, 5, 24),
        run_timestamp_utc="2026-05-24T00:00:00+00:00",
    )

    assert signal.final_status == BLOCKED_BY_KILL_SWITCH
    assert output_path.exists()

    out = pd.read_csv(output_path)
    assert out.loc[0, "target_trade_date"] == "2026-05-23"
    assert out.loc[0, "recommended_action"] == PAPER_SHORT_VOL_INTENT


def test_publish_daily_paper_signal_stale_signal(tmp_path: Path) -> None:
    signal_path = tmp_path / "signals.csv"
    _signal_frame().to_csv(signal_path, index=False)

    config = _load_config(tmp_path)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        strategy_name="mar_prob_linear_carry",
        signal_path=signal_path,
        output_path=output_path,
        as_of_date=date(2026, 6, 30),
    )

    assert signal.signal_is_stale is True
    assert signal.final_status == BLOCKED_STALE_SIGNAL

    out = pd.read_csv(output_path)
    assert out.loc[0, "final_status"] == BLOCKED_STALE_SIGNAL


def test_publish_daily_paper_signal_missing_file_writes_blocked_record(
    tmp_path: Path,
) -> None:
    config = _load_config(tmp_path)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        strategy_name="mar_prob_linear_carry",
        signal_path=tmp_path / "missing.csv",
        output_path=output_path,
    )

    assert signal.final_status == BLOCKED_MISSING_SIGNAL
    assert signal.recommended_action == NO_SIGNAL
    assert output_path.exists()


def test_publish_daily_paper_signal_bad_schema_writes_blocked_record(
    tmp_path: Path,
) -> None:
    signal_path = tmp_path / "signals.csv"
    pd.DataFrame({"bad": [1]}).to_csv(signal_path, index=False)

    config = _load_config(tmp_path)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        strategy_name="mar_prob_linear_carry",
        signal_path=signal_path,
        output_path=output_path,
    )

    assert signal.final_status == BLOCKED_MISSING_SIGNAL
    assert "strategy_name or strategy" in signal.blocked_reason


def test_publish_daily_paper_signal_uses_default_strategy(tmp_path: Path) -> None:
    signal_path = tmp_path / "signals.csv"
    _signal_frame().to_csv(signal_path, index=False)

    config = _load_config(tmp_path)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        signal_path=signal_path,
        output_path=output_path,
        as_of_date=date(2026, 5, 24),
    )

    assert signal.strategy_name == "mar_prob_linear_carry"


def test_publish_daily_paper_signal_uses_configured_signal_path(tmp_path: Path) -> None:
    signal_path = tmp_path / "us_strategy_signals.csv"
    _signal_frame().to_csv(signal_path, index=False)

    config_data = _base_config()
    config_data["signal_inputs"]["US"] = str(signal_path)

    config = _load_config(tmp_path, config_data)
    output_path = tmp_path / "daily_paper_signal.csv"

    signal = publish_daily_paper_signal(
        config=config,
        market="US",
        output_path=output_path,
        as_of_date=date(2026, 5, 24),
    )

    assert signal.source_signal_path == str(signal_path)
    assert signal.final_status == BLOCKED_BY_KILL_SWITCH