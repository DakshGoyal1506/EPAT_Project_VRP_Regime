"""
Phase 11 daily paper-signal publisher.

This module reads the latest validated Phase 9 strategy signal and writes
reports/tables/phase_11/daily_paper_signal.csv.

It does not create order intents.
It does not size positions.
It does not inspect broker quotes.
It does not place live orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import (
    BrokerConfig,
    get_market_signal_path,
    get_output_paths,
)
from vrp.broker.signal_schema import (
    Phase9SignalSchema,
    SignalFreshnessResult,
    SignalSchemaError,
    latest_signal_to_record,
    load_validate_select_latest_signal,
    signal_freshness_from_config,
)


NO_SIGNAL = "NO_SIGNAL"
STAY_FLAT = "STAY_FLAT"
REDUCE_TO_ZERO = "REDUCE_TO_ZERO"
PAPER_SHORT_VOL_INTENT = "PAPER_SHORT_VOL_INTENT"

BLOCKED_BY_KILL_SWITCH = "BLOCKED_BY_KILL_SWITCH"
BLOCKED_STALE_SIGNAL = "BLOCKED_STALE_SIGNAL"
BLOCKED_MISSING_SIGNAL = "BLOCKED_MISSING_SIGNAL"
BROKER_INSPECTION_ONLY = "BROKER_INSPECTION_ONLY"


class SignalPublisherError(RuntimeError):
    """Raised when the Phase 11 paper signal cannot be published."""


@dataclass(frozen=True)
class DailyPaperSignal:
    """Daily paper signal record for Phase 11."""

    run_timestamp_utc: str
    market: str
    strategy_name: str
    signal_observation_date: str | None
    target_trade_date: str | None
    target_exposure: float | None
    strategy_available: bool
    recommended_action: str
    blocked_reason: str
    decision_reason: str
    signal_age_days: int | None
    signal_is_stale: bool
    signal_freshness_reason: str
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    intent_allowed_before_kill_switch: bool
    intent_allowed_after_kill_switch: bool
    final_status: str
    live_order_sent: bool
    research_proxy_warning: str
    source_signal_path: str
    source_signal_mtime_utc: str | None
    source_strategy_row: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp_utc": self.run_timestamp_utc,
            "market": self.market,
            "strategy_name": self.strategy_name,
            "signal_observation_date": self.signal_observation_date,
            "target_trade_date": self.target_trade_date,
            "target_exposure": self.target_exposure,
            "strategy_available": self.strategy_available,
            "recommended_action": self.recommended_action,
            "blocked_reason": self.blocked_reason,
            "decision_reason": self.decision_reason,
            "signal_age_days": self.signal_age_days,
            "signal_is_stale": self.signal_is_stale,
            "signal_freshness_reason": self.signal_freshness_reason,
            "paper_only": self.paper_only,
            "kill_switch": self.kill_switch,
            "live_orders_enabled": self.live_orders_enabled,
            "allow_order_placement": self.allow_order_placement,
            "intent_allowed_before_kill_switch": self.intent_allowed_before_kill_switch,
            "intent_allowed_after_kill_switch": self.intent_allowed_after_kill_switch,
            "final_status": self.final_status,
            "live_order_sent": self.live_order_sent,
            "research_proxy_warning": self.research_proxy_warning,
            "source_signal_path": self.source_signal_path,
            "source_signal_mtime_utc": self.source_signal_mtime_utc,
            "source_strategy_row": self.source_strategy_row,
        }


def interpret_signal_action(
    latest_signal: pd.Series | Mapping[str, Any],
    *,
    use_position_state: bool = False,
    prior_target_exposure: float | None = None,
) -> str:
    """Map Phase 9 signal semantics to Phase 11 paper action.

    No risk checks are performed here.
    """

    strategy_available = bool(latest_signal["strategy_available"])
    target_exposure = float(latest_signal["target_exposure"])

    if not strategy_available:
        return NO_SIGNAL

    if target_exposure == 0.0:
        if use_position_state and prior_target_exposure is not None:
            if prior_target_exposure < 0.0:
                return REDUCE_TO_ZERO

        return STAY_FLAT

    if target_exposure < 0.0:
        return PAPER_SHORT_VOL_INTENT

    raise SignalPublisherError(
        f"Unexpected target_exposure={target_exposure}. "
        "Phase 11 expects exposure in [-1.0, 0.0]."
    )


def determine_signal_final_status(
    *,
    recommended_action: str,
    config: BrokerConfig,
    freshness: SignalFreshnessResult | None,
) -> tuple[str, bool, bool]:
    """Determine signal-level final status.

    Returns:
        final_status,
        intent_allowed_before_kill_switch,
        intent_allowed_after_kill_switch

    Full risk validation is handled in later chunks.
    """

    if freshness is not None and freshness.is_stale and freshness.final_status_if_blocked:
        return freshness.final_status_if_blocked, False, False

    if recommended_action == NO_SIGNAL:
        return NO_SIGNAL, False, False

    if recommended_action == STAY_FLAT:
        return STAY_FLAT, False, False

    if recommended_action == REDUCE_TO_ZERO:
        before_kill = True
        after_kill = _allowed_after_kill_switch(before_kill, config)
        final_status = BLOCKED_BY_KILL_SWITCH if before_kill and not after_kill else BROKER_INSPECTION_ONLY
        return final_status, before_kill, after_kill

    if recommended_action == PAPER_SHORT_VOL_INTENT:
        before_kill = True
        after_kill = _allowed_after_kill_switch(before_kill, config)
        final_status = BLOCKED_BY_KILL_SWITCH if before_kill and not after_kill else BROKER_INSPECTION_ONLY
        return final_status, before_kill, after_kill

    return BLOCKED_MISSING_SIGNAL, False, False


def build_daily_paper_signal(
    *,
    market: str,
    strategy_name: str,
    latest_signal: pd.Series | Mapping[str, Any],
    config: BrokerConfig,
    signal_path: str | Path,
    freshness: SignalFreshnessResult | None = None,
    run_timestamp_utc: str | None = None,
    use_position_state: bool | None = None,
    prior_target_exposure: float | None = None,
) -> DailyPaperSignal:
    """Build the daily paper signal record."""

    market_code = market.upper()
    signal_record = latest_signal_to_record(latest_signal)

    if use_position_state is None:
        state_config = config.raw.get("state")
        use_position_state = (
            bool(state_config.get("use_position_state", False))
            if isinstance(state_config, Mapping)
            else False
        )

    recommended_action = interpret_signal_action(
        latest_signal,
        use_position_state=use_position_state,
        prior_target_exposure=prior_target_exposure,
    )

    final_status, before_kill, after_kill = determine_signal_final_status(
        recommended_action=recommended_action,
        config=config,
        freshness=freshness,
    )

    path = Path(signal_path)
    source_mtime = _file_mtime_utc(path)

    return DailyPaperSignal(
        run_timestamp_utc=run_timestamp_utc or _utc_now_iso(),
        market=market_code,
        strategy_name=strategy_name,
        signal_observation_date=_optional_str(signal_record.get("signal_observation_date")),
        target_trade_date=_optional_str(signal_record.get("target_trade_date")),
        target_exposure=_optional_float(signal_record.get("target_exposure")),
        strategy_available=bool(latest_signal["strategy_available"]),
        recommended_action=recommended_action,
        blocked_reason=str(signal_record.get("blocked_reason", "")),
        decision_reason=str(signal_record.get("decision_reason", "")),
        signal_age_days=freshness.age_days if freshness is not None else None,
        signal_is_stale=bool(freshness.is_stale) if freshness is not None else False,
        signal_freshness_reason=freshness.reason if freshness is not None else "",
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        intent_allowed_before_kill_switch=before_kill,
        intent_allowed_after_kill_switch=after_kill,
        final_status=final_status,
        live_order_sent=False,
        research_proxy_warning=str(config.raw.get("research_proxy_warning", RESEARCH_PROXY_WARNING)),
        source_signal_path=str(path),
        source_signal_mtime_utc=source_mtime,
        source_strategy_row=signal_record,
    )


def write_daily_paper_signal(
    signal: DailyPaperSignal,
    output_path: str | Path,
) -> Path:
    """Write one-row daily paper signal CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = signal.as_dict().copy()
    row["source_strategy_row"] = _jsonish_string(row["source_strategy_row"])

    pd.DataFrame([row]).to_csv(path, index=False)

    return path


def publish_daily_paper_signal(
    *,
    config: BrokerConfig,
    market: str,
    strategy_name: str | None = None,
    signal_path: str | Path | None = None,
    output_path: str | Path | None = None,
    as_of_date: date | datetime | str | None = None,
    run_timestamp_utc: str | None = None,
    schema: Phase9SignalSchema | None = None,
) -> DailyPaperSignal:
    """End-to-end daily paper signal publisher.

    Reads the configured Phase 9 signal file, selects the latest row, checks
    freshness, builds the Phase 11 paper-signal record, and writes CSV.
    """

    market_code = market.upper()
    active_strategy = strategy_name or config.default_strategy
    active_signal_path = Path(signal_path) if signal_path is not None else get_market_signal_path(config, market_code)

    active_schema = schema or Phase9SignalSchema.from_broker_config(config)

    try:
        latest = load_validate_select_latest_signal(
            active_signal_path,
            schema=active_schema,
            strategy_name=active_strategy,
            market=market_code,
        )
    except SignalSchemaError as exc:
        signal = build_missing_signal_record(
            market=market_code,
            strategy_name=active_strategy,
            config=config,
            signal_path=active_signal_path,
            reason=str(exc),
            run_timestamp_utc=run_timestamp_utc,
        )
        _write_signal_to_configured_output(signal, config, output_path)
        return signal

    freshness = signal_freshness_from_config(
        latest,
        config,
        as_of_date=as_of_date,
    )

    signal = build_daily_paper_signal(
        market=market_code,
        strategy_name=active_strategy,
        latest_signal=latest,
        config=config,
        signal_path=active_signal_path,
        freshness=freshness,
        run_timestamp_utc=run_timestamp_utc,
    )

    _write_signal_to_configured_output(signal, config, output_path)
    return signal


def build_missing_signal_record(
    *,
    market: str,
    strategy_name: str,
    config: BrokerConfig,
    signal_path: str | Path,
    reason: str,
    run_timestamp_utc: str | None = None,
) -> DailyPaperSignal:
    """Build a BLOCKED_MISSING_SIGNAL record when Phase 9 input cannot load."""

    path = Path(signal_path)

    return DailyPaperSignal(
        run_timestamp_utc=run_timestamp_utc or _utc_now_iso(),
        market=market.upper(),
        strategy_name=strategy_name,
        signal_observation_date=None,
        target_trade_date=None,
        target_exposure=None,
        strategy_available=False,
        recommended_action=NO_SIGNAL,
        blocked_reason=reason,
        decision_reason="Phase 9 signal could not be loaded or validated",
        signal_age_days=None,
        signal_is_stale=False,
        signal_freshness_reason="signal missing or invalid",
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        intent_allowed_before_kill_switch=False,
        intent_allowed_after_kill_switch=False,
        final_status=BLOCKED_MISSING_SIGNAL,
        live_order_sent=False,
        research_proxy_warning=str(config.raw.get("research_proxy_warning", RESEARCH_PROXY_WARNING)),
        source_signal_path=str(path),
        source_signal_mtime_utc=_file_mtime_utc(path),
        source_strategy_row={},
    )


def _write_signal_to_configured_output(
    signal: DailyPaperSignal,
    config: BrokerConfig,
    output_path: str | Path | None,
) -> Path:
    if output_path is None:
        output_paths = get_output_paths(config)
        output_path = output_paths["latest_signal_table"]

    return write_daily_paper_signal(signal, output_path)


def _allowed_after_kill_switch(before_kill: bool, config: BrokerConfig) -> bool:
    if not before_kill:
        return False

    risk = config.risk_checks

    if config.kill_switch and risk.block_if_kill_switch_on:
        return False

    if not config.paper_only:
        return False

    if config.live_orders_enabled:
        return False

    if config.allow_order_placement:
        return False

    return True


def _file_mtime_utc(path: Path) -> str | None:
    if not path.exists():
        return None

    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.replace(microsecond=0).isoformat()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    return float(value)


def _jsonish_string(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str)