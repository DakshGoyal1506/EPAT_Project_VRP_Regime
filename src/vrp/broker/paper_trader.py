"""
Paper intent writer for Phase 11.

This module creates paper_order_intents.csv. It does not create live broker
orders and does not call broker execution APIs.

Rules:
- NO_SIGNAL creates no paper intent.
- STAY_FLAT creates no paper intent unless future position-state rules require it.
- REDUCE_TO_ZERO requires prior paper position state.
- PAPER_SHORT_VOL_INTENT may create a paper intent, then risk checks decide status.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import BrokerConfig, get_output_paths
from vrp.broker.contracts import (
    BrokerInstrumentRegistry,
    ContractSpec,
    InstrumentMapping,
)
from vrp.broker.market_data import QuoteSnapshot, missing_quote
from vrp.broker.paper_sizing import PaperSizingResult, build_paper_sizing
from vrp.broker.paper_state import (
    PaperPositionState,
    get_latest_position_state,
)
from vrp.broker.risk_checks import (
    RiskCheckSummary,
    run_phase11_risk_checks,
    write_empty_risk_check_report,
    write_risk_check_report,
)
from vrp.broker.signal_publisher import (
    DailyPaperSignal,
    NO_SIGNAL,
    PAPER_SHORT_VOL_INTENT,
    REDUCE_TO_ZERO,
    STAY_FLAT,
)


PAPER_ORDER_INTENT_COLUMNS = [
    "run_timestamp_utc",
    "market",
    "strategy_name",
    "target_trade_date",
    "signal_observation_date",
    "recommended_action",
    "symbol",
    "instrument_type",
    "side",
    "target_exposure",
    "paper_target_notional",
    "reference_price",
    "paper_quantity",
    "quantity_type",
    "sizing_status",
    "quote_status",
    "risk_final_status",
    "risk_block_reason",
    "paper_only",
    "kill_switch",
    "live_orders_enabled",
    "allow_order_placement",
    "intent_allowed_before_kill_switch",
    "intent_allowed_after_kill_switch",
    "final_status",
    "live_order_sent",
    "research_proxy_warning",
]

TERMINAL_NO_INTENT_FINAL_STATUSES = {
    "BLOCKED_MISSING_SIGNAL",
    "BLOCKED_STALE_SIGNAL",
    "NO_SIGNAL",
    "STAY_FLAT",
}


class PaperTraderError(RuntimeError):
    """Raised when paper intent construction fails."""


@dataclass(frozen=True)
class PaperOrderIntent:
    """Hypothetical paper intent row.

    This is not a live broker order.
    """

    run_timestamp_utc: str
    market: str
    strategy_name: str
    target_trade_date: str | None
    signal_observation_date: str | None
    recommended_action: str
    symbol: str
    instrument_type: str
    side: str
    target_exposure: float
    paper_target_notional: float
    reference_price: float | None
    paper_quantity: int | float | None
    quantity_type: str
    sizing_status: str
    quote_status: str | None
    risk_final_status: str
    risk_block_reason: str
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    intent_allowed_before_kill_switch: bool
    intent_allowed_after_kill_switch: bool
    final_status: str
    live_order_sent: bool
    research_proxy_warning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp_utc": self.run_timestamp_utc,
            "market": self.market,
            "strategy_name": self.strategy_name,
            "target_trade_date": self.target_trade_date,
            "signal_observation_date": self.signal_observation_date,
            "recommended_action": self.recommended_action,
            "symbol": self.symbol,
            "instrument_type": self.instrument_type,
            "side": self.side,
            "target_exposure": self.target_exposure,
            "paper_target_notional": self.paper_target_notional,
            "reference_price": self.reference_price,
            "paper_quantity": self.paper_quantity,
            "quantity_type": self.quantity_type,
            "sizing_status": self.sizing_status,
            "quote_status": self.quote_status,
            "risk_final_status": self.risk_final_status,
            "risk_block_reason": self.risk_block_reason,
            "paper_only": self.paper_only,
            "kill_switch": self.kill_switch,
            "live_orders_enabled": self.live_orders_enabled,
            "allow_order_placement": self.allow_order_placement,
            "intent_allowed_before_kill_switch": self.intent_allowed_before_kill_switch,
            "intent_allowed_after_kill_switch": self.intent_allowed_after_kill_switch,
            "final_status": self.final_status,
            "live_order_sent": self.live_order_sent,
            "research_proxy_warning": self.research_proxy_warning,
        }


@dataclass(frozen=True)
class PaperIntentBuildResult:
    """Result of paper-intent construction."""

    intent: PaperOrderIntent | None
    contract: ContractSpec | None
    sizing: PaperSizingResult | None
    risk_summary: RiskCheckSummary | None
    reason: str

    def has_intent(self) -> bool:
        return self.intent is not None


def read_daily_paper_signal(path: str | Path) -> DailyPaperSignal:
    """Read one-row daily_paper_signal.csv into DailyPaperSignal."""

    signal_path = Path(path)
    if not signal_path.exists():
        raise PaperTraderError(f"daily_paper_signal.csv not found: {signal_path}")

    frame = pd.read_csv(signal_path)

    if frame.empty:
        raise PaperTraderError(f"daily_paper_signal.csv is empty: {signal_path}")

    if len(frame) != 1:
        raise PaperTraderError(
            f"daily_paper_signal.csv must contain exactly one row: {signal_path}"
        )

    row = frame.iloc[0].to_dict()

    return DailyPaperSignal(
        run_timestamp_utc=str(row["run_timestamp_utc"]),
        market=str(row["market"]).upper(),
        strategy_name=str(row["strategy_name"]),
        signal_observation_date=_optional_str(row.get("signal_observation_date")),
        target_trade_date=_optional_str(row.get("target_trade_date")),
        target_exposure=_optional_float(row.get("target_exposure")),
        strategy_available=_to_bool(row.get("strategy_available")),
        recommended_action=str(row["recommended_action"]),
        blocked_reason=str(row.get("blocked_reason", "")),
        decision_reason=str(row.get("decision_reason", "")),
        signal_age_days=_optional_int(row.get("signal_age_days")),
        signal_is_stale=_to_bool(row.get("signal_is_stale")),
        signal_freshness_reason=str(row.get("signal_freshness_reason", "")),
        paper_only=_to_bool(row.get("paper_only")),
        kill_switch=_to_bool(row.get("kill_switch")),
        live_orders_enabled=_to_bool(row.get("live_orders_enabled")),
        allow_order_placement=_to_bool(row.get("allow_order_placement")),
        intent_allowed_before_kill_switch=_to_bool(
            row.get("intent_allowed_before_kill_switch")
        ),
        intent_allowed_after_kill_switch=_to_bool(
            row.get("intent_allowed_after_kill_switch")
        ),
        final_status=str(row.get("final_status", "")),
        live_order_sent=_to_bool(row.get("live_order_sent")),
        research_proxy_warning=str(
            row.get("research_proxy_warning", RESEARCH_PROXY_WARNING)
        ),
        source_signal_path=str(row.get("source_signal_path", "")),
        source_signal_mtime_utc=_optional_str(row.get("source_signal_mtime_utc")),
        source_strategy_row={},
    )


def build_paper_order_intent(
    *,
    daily_signal: DailyPaperSignal,
    config: BrokerConfig,
    quote: QuoteSnapshot | None = None,
    registry: BrokerInstrumentRegistry | None = None,
    prior_state: PaperPositionState | None = None,
) -> PaperIntentBuildResult:
    """Build a paper order-intent candidate from a daily paper signal."""

    action = daily_signal.recommended_action

    if daily_signal.final_status in TERMINAL_NO_INTENT_FINAL_STATUSES:
        return PaperIntentBuildResult(
            intent=None,
            contract=None,
            sizing=None,
            risk_summary=None,
            reason=(
                f"{daily_signal.final_status} is terminal and does not create "
                "paper order intent"
            ),
        )

    if action == NO_SIGNAL:
        return PaperIntentBuildResult(
            intent=None,
            contract=None,
            sizing=None,
            risk_summary=None,
            reason="NO_SIGNAL does not create paper order intent",
        )

    if action == STAY_FLAT:
        return PaperIntentBuildResult(
            intent=None,
            contract=None,
            sizing=None,
            risk_summary=None,
            reason="STAY_FLAT does not create paper order intent without position state",
        )

    if action == REDUCE_TO_ZERO and prior_state is None:
        return PaperIntentBuildResult(
            intent=None,
            contract=None,
            sizing=None,
            risk_summary=None,
            reason="REDUCE_TO_ZERO requires prior paper position state",
        )

    registry = registry or BrokerInstrumentRegistry()
    contract = select_contract_for_paper_signal(
        market=daily_signal.market,
        recommended_action=action,
        config=config,
        registry=registry,
    )

    active_quote = quote
    if active_quote is None:
        active_quote = missing_quote(symbol=contract.symbol, market=contract.market)

    target_exposure = float(daily_signal.target_exposure or 0.0)

    sizing = build_paper_sizing(
        target_exposure=target_exposure,
        contract=contract,
        config=config,
        quote=active_quote,
    )

    if action == REDUCE_TO_ZERO and prior_state is not None:
        sizing_quantity = prior_state.paper_quantity
    else:
        sizing_quantity = sizing.paper_quantity

    risk_summary = run_phase11_risk_checks(
        market=daily_signal.market,
        recommended_action=action,
        contract=contract,
        sizing=sizing,
        config=config,
        quote=active_quote,
        signal_final_status=daily_signal.final_status,
        signal_is_stale=daily_signal.signal_is_stale,
    )

    side = infer_paper_side(
        recommended_action=action,
        contract=contract,
        prior_state=prior_state,
    )

    intent = PaperOrderIntent(
        run_timestamp_utc=daily_signal.run_timestamp_utc or _utc_now_iso(),
        market=daily_signal.market,
        strategy_name=daily_signal.strategy_name,
        target_trade_date=daily_signal.target_trade_date,
        signal_observation_date=daily_signal.signal_observation_date,
        recommended_action=action,
        symbol=contract.symbol,
        instrument_type=contract.sec_type,
        side=side,
        target_exposure=target_exposure,
        paper_target_notional=sizing.paper_target_notional,
        reference_price=sizing.reference_price,
        paper_quantity=sizing_quantity,
        quantity_type=sizing.quantity_type,
        sizing_status=sizing.sizing_status,
        quote_status=risk_summary.quote_status,
        risk_final_status=risk_summary.final_status,
        risk_block_reason=risk_summary.primary_block_reason,
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        intent_allowed_before_kill_switch=risk_summary.intent_allowed_before_kill_switch,
        intent_allowed_after_kill_switch=risk_summary.intent_allowed_after_kill_switch,
        final_status=risk_summary.final_status,
        live_order_sent=False,
        research_proxy_warning=str(
            config.raw.get("research_proxy_warning", RESEARCH_PROXY_WARNING)
        ),
    )

    return PaperIntentBuildResult(
        intent=intent,
        contract=contract,
        sizing=sizing,
        risk_summary=risk_summary,
        reason="paper order intent built",
    )


def publish_paper_order_intent(
    *,
    config: BrokerConfig,
    daily_signal: DailyPaperSignal | None = None,
    daily_signal_path: str | Path | None = None,
    quote: QuoteSnapshot | None = None,
    output_path: str | Path | None = None,
    risk_report_path: str | Path | None = None,
    registry: BrokerInstrumentRegistry | None = None,
) -> PaperIntentBuildResult:
    """Build and write paper_order_intents.csv.

    If signal action does not require an intent, an empty CSV with the correct
    header is written.
    """

    if daily_signal is None:
        if daily_signal_path is None:
            output_paths = get_output_paths(config)
            daily_signal_path = output_paths["latest_signal_table"]

        daily_signal = read_daily_paper_signal(daily_signal_path)

    prior_state = resolve_prior_state_if_enabled(config, daily_signal)

    result = build_paper_order_intent(
        daily_signal=daily_signal,
        config=config,
        quote=quote,
        registry=registry,
        prior_state=prior_state,
    )

    if output_path is None:
        output_path = get_output_paths(config)["paper_order_intents"]

    write_paper_order_intents(
        [result.intent] if result.intent is not None else [],
        output_path,
    )

    if risk_report_path is None:
        risk_report_path = get_output_paths(config)["risk_check_report"]

    if result.risk_summary is not None:
        write_risk_check_report(result.risk_summary, risk_report_path)
    else:
        write_empty_risk_check_report(risk_report_path)

    return result


def write_paper_order_intents(
    intents: list[PaperOrderIntent],
    output_path: str | Path,
) -> Path:
    """Write paper_order_intents.csv.

    Empty list writes a header-only CSV.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [intent.as_dict() for intent in intents]
    frame = pd.DataFrame(records, columns=PAPER_ORDER_INTENT_COLUMNS)
    frame.to_csv(path, index=False)

    return path


def select_contract_for_paper_signal(
    *,
    market: str,
    recommended_action: str,
    config: BrokerConfig,
    registry: BrokerInstrumentRegistry | None = None,
) -> ContractSpec:
    """Select paper proxy without hiding later risk blocks.

    This selector intentionally does not pre-block India. Risk checks handle
    India signal-only status explicitly and record the reason.
    """

    registry = registry or BrokerInstrumentRegistry()
    market_code = market.upper()
    mapping = InstrumentMapping.from_config(market_code, config)

    if recommended_action == PAPER_SHORT_VOL_INTENT:
        for symbol in mapping.short_vol_proxy_candidates:
            return registry.get(market_code, symbol)

    return registry.get(market_code, mapping.default_paper_proxy)


def infer_paper_side(
    *,
    recommended_action: str,
    contract: ContractSpec,
    prior_state: PaperPositionState | None = None,
) -> str:
    """Infer paper side label.

    These are labels in CSV output only, not broker calls.
    """

    action = recommended_action.upper()
    symbol = contract.symbol.upper()

    if action == REDUCE_TO_ZERO:
        return "REDUCE_TO_ZERO"

    if action == PAPER_SHORT_VOL_INTENT:
        if symbol == "VXX":
            return "SELL"

        if symbol == "SVXY":
            return "BUY"

        return "PAPER_SHORT_VOL"

    return "NONE"


def resolve_prior_state_if_enabled(
    config: BrokerConfig,
    daily_signal: DailyPaperSignal,
) -> PaperPositionState | None:
    """Load prior paper state only when config enables it."""

    state_config = config.raw.get("state")
    if not isinstance(state_config, Mapping):
        return None

    if not bool(state_config.get("use_position_state", False)):
        return None

    state_path = state_config.get("paper_position_state")
    if not state_path:
        return None

    return get_latest_position_state(
        state_path,
        market=daily_signal.market,
        strategy_name=daily_signal.strategy_name,
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "t", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "f", "0", "no", "n"}:
            return False

    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if value == "":
        return None

    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if value == "":
        return None

    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if value == "":
        return None

    return int(float(value))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()