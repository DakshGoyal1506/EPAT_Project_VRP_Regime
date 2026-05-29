"""
Risk checks for Phase 11 paper signals and paper intents.

This module aggregates safety checks. It does not create broker orders and does
not write order-intent rows. It prepares auditable risk-check output for later
paper-intent construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vrp.broker.broker_config import BrokerConfig
from vrp.broker.contracts import (
    ContractSpec,
    ContractValidationError,
    is_broker_intent_eligible,
    validate_contract_allowed,
)
from vrp.broker.market_data import (
    QuoteReadinessResult,
    QuoteSnapshot,
    evaluate_quote_readiness,
)
from vrp.broker.paper_sizing import PaperSizingResult


PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"

ALLOWED_PAPER_INTENT = "ALLOWED_PAPER_INTENT"
BLOCKED_BY_KILL_SWITCH = "BLOCKED_BY_KILL_SWITCH"
BLOCKED_STALE_SIGNAL = "BLOCKED_STALE_SIGNAL"
BLOCKED_CONFIG_SAFETY = "BLOCKED_CONFIG_SAFETY"
BLOCKED_RISK_LIMIT = "BLOCKED_RISK_LIMIT"
BLOCKED_MISSING_SIGNAL = "BLOCKED_MISSING_SIGNAL"
BLOCKED_BROKER_DATA = "BLOCKED_BROKER_DATA"
NO_SIGNAL = "NO_SIGNAL"
STAY_FLAT = "STAY_FLAT"
BROKER_INSPECTION_ONLY = "BROKER_INSPECTION_ONLY"

RISK_CHECK_REPORT_COLUMNS = [
    "market",
    "symbol",
    "recommended_action",
    "paper_only",
    "kill_switch",
    "live_orders_enabled",
    "allow_order_placement",
    "intent_allowed_before_kill_switch",
    "intent_allowed_after_kill_switch",
    "final_status",
    "primary_block_reason",
    "quote_status",
    "live_order_sent",
    "check_name",
    "status",
    "blocks_intent",
    "reason",
    "observed_value",
    "limit_value",
]


class RiskCheckError(RuntimeError):
    """Raised when risk-check inputs are invalid."""


@dataclass(frozen=True)
class RiskCheckResult:
    """Single risk check result."""

    check_name: str
    status: str
    blocks_intent: bool
    reason: str
    observed_value: Any = None
    limit_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status,
            "blocks_intent": self.blocks_intent,
            "reason": self.reason,
            "observed_value": self.observed_value,
            "limit_value": self.limit_value,
        }


@dataclass(frozen=True)
class RiskCheckSummary:
    """Aggregate Phase 11 risk decision."""

    market: str
    symbol: str
    recommended_action: str
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    intent_allowed_before_kill_switch: bool
    intent_allowed_after_kill_switch: bool
    final_status: str
    primary_block_reason: str
    quote_status: str | None
    live_order_sent: bool
    checks: tuple[RiskCheckResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "recommended_action": self.recommended_action,
            "paper_only": self.paper_only,
            "kill_switch": self.kill_switch,
            "live_orders_enabled": self.live_orders_enabled,
            "allow_order_placement": self.allow_order_placement,
            "intent_allowed_before_kill_switch": self.intent_allowed_before_kill_switch,
            "intent_allowed_after_kill_switch": self.intent_allowed_after_kill_switch,
            "final_status": self.final_status,
            "primary_block_reason": self.primary_block_reason,
            "quote_status": self.quote_status,
            "live_order_sent": self.live_order_sent,
        }

    def checks_as_records(self) -> list[dict[str, Any]]:
        base = self.as_dict()
        records = []
        for check in self.checks:
            row = base.copy()
            row.update(check.as_dict())
            records.append(row)
        return records


def run_phase11_risk_checks(
    *,
    market: str,
    recommended_action: str,
    contract: ContractSpec,
    sizing: PaperSizingResult,
    config: BrokerConfig,
    quote: QuoteSnapshot | None = None,
    signal_final_status: str | None = None,
    signal_is_stale: bool = False,
) -> RiskCheckSummary:
    """Run Phase 11 paper-intent risk checks."""

    market_code = market.upper()
    action = str(recommended_action)

    checks: list[RiskCheckResult] = []

    if action == NO_SIGNAL:
        return _terminal_summary(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            final_status=NO_SIGNAL,
            primary_block_reason="no available Phase 9 signal",
            checks=(
                RiskCheckResult(
                    "signal_available",
                    FAILED,
                    True,
                    "strategy_available is false",
                ),
            ),
            quote_status=None,
        )

    if action == STAY_FLAT:
        return _terminal_summary(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            final_status=STAY_FLAT,
            primary_block_reason="target exposure is zero; no paper intent needed",
            checks=(
                RiskCheckResult(
                    "stay_flat",
                    PASSED,
                    False,
                    "target exposure is zero",
                ),
            ),
            quote_status=None,
        )

    if signal_final_status == BLOCKED_MISSING_SIGNAL:
        return _terminal_summary(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            final_status=BLOCKED_MISSING_SIGNAL,
            primary_block_reason="Phase 9 signal missing or invalid",
            checks=(
                RiskCheckResult(
                    "signal_schema",
                    FAILED,
                    True,
                    "Phase 9 signal missing or invalid",
                ),
            ),
            quote_status=None,
        )

    if signal_is_stale or signal_final_status == BLOCKED_STALE_SIGNAL:
        return _terminal_summary(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            final_status=BLOCKED_STALE_SIGNAL,
            primary_block_reason="latest Phase 9 signal is stale",
            checks=(
                RiskCheckResult(
                    "signal_freshness",
                    FAILED,
                    True,
                    "latest Phase 9 signal is stale",
                ),
            ),
            quote_status=None,
        )

    checks.extend(_config_safety_checks(config))
    checks.extend(_contract_checks(contract, config))
    checks.extend(_sizing_checks(sizing, config))

    quote_readiness = evaluate_quote_readiness(quote, config)
    checks.extend(_quote_checks(quote_readiness))

    config_block = _first_blocking_check(
        checks,
        names={
            "paper_only",
            "live_orders_enabled",
            "allow_order_placement",
            "naked_short_options",
            "max_contracts",
            "max_margin_usage",
        },
    )

    if config_block is not None:
        return _summary_from_checks(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            checks=tuple(checks),
            quote_status=quote_readiness.quote_status,
            final_status=BLOCKED_CONFIG_SAFETY,
            primary_block_reason=config_block.reason,
        )

    risk_block = _first_blocking_check(
        checks,
        names={
            "contract_allowed",
            "broker_intent_eligible",
            "india_manual_verification",
            "max_notional",
            "max_shares",
            "max_delta",
            "max_vega",
            "quote_staleness",
            "quote_spread",
        },
    )

    if risk_block is not None:
        return _summary_from_checks(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            checks=tuple(checks),
            quote_status=quote_readiness.quote_status,
            final_status=BLOCKED_RISK_LIMIT,
            primary_block_reason=risk_block.reason,
        )

    broker_data_block = _first_blocking_check(
        checks,
        names={"quote_available"},
    )

    if broker_data_block is not None:
        return _summary_from_checks(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            checks=tuple(checks),
            quote_status=quote_readiness.quote_status,
            final_status=BLOCKED_BROKER_DATA,
            primary_block_reason=broker_data_block.reason,
        )

    before_kill = True
    after_kill = _allowed_after_kill_switch(before_kill, config)

    if before_kill and not after_kill:
        checks.append(
            RiskCheckResult(
                "kill_switch",
                FAILED,
                True,
                "kill_switch is on and blocks paper intent",
                observed_value=config.kill_switch,
                limit_value=False,
            )
        )
        return _summary_from_checks(
            market=market_code,
            symbol=contract.symbol,
            recommended_action=action,
            config=config,
            checks=tuple(checks),
            quote_status=quote_readiness.quote_status,
            final_status=BLOCKED_BY_KILL_SWITCH,
            primary_block_reason="kill_switch is on and blocks paper intent",
            intent_allowed_before_kill_switch=True,
            intent_allowed_after_kill_switch=False,
        )

    checks.append(
        RiskCheckResult(
            "kill_switch",
            PASSED,
            False,
            "kill_switch does not block paper intent",
            observed_value=config.kill_switch,
            limit_value=False,
        )
    )

    return _summary_from_checks(
        market=market_code,
        symbol=contract.symbol,
        recommended_action=action,
        config=config,
        checks=tuple(checks),
        quote_status=quote_readiness.quote_status,
        final_status=ALLOWED_PAPER_INTENT,
        primary_block_reason="all configured Phase 11 checks passed",
        intent_allowed_before_kill_switch=True,
        intent_allowed_after_kill_switch=True,
    )


def write_risk_check_report(
    summary: RiskCheckSummary,
    output_path: str | Path,
) -> Path:
    """Write detailed risk-check report to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = summary.checks_as_records()
    if not records:
        records = [summary.as_dict()]

    pd.DataFrame(records, columns=RISK_CHECK_REPORT_COLUMNS).to_csv(path, index=False)
    return path


def write_empty_risk_check_report(output_path: str | Path) -> Path:
    """Write header-only risk-check report.

    Used when no paper intent is allowed to reach risk checks.
    This prevents stale risk_check_report.csv content from previous runs.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=RISK_CHECK_REPORT_COLUMNS).to_csv(path, index=False)
    return path


def _config_safety_checks(config: BrokerConfig) -> list[RiskCheckResult]:
    sizing = config.paper_sizing
    checks = [
        RiskCheckResult(
            "paper_only",
            PASSED if config.paper_only else FAILED,
            not config.paper_only,
            "paper_only must be true",
            observed_value=config.paper_only,
            limit_value=True,
        ),
        RiskCheckResult(
            "live_orders_enabled",
            PASSED if not config.live_orders_enabled else FAILED,
            bool(config.live_orders_enabled),
            "live_orders_enabled must be false",
            observed_value=config.live_orders_enabled,
            limit_value=False,
        ),
        RiskCheckResult(
            "allow_order_placement",
            PASSED if not config.allow_order_placement else FAILED,
            bool(config.allow_order_placement),
            "allow_order_placement must be false",
            observed_value=config.allow_order_placement,
            limit_value=False,
        ),
        RiskCheckResult(
            "naked_short_options",
            PASSED if not sizing.allow_naked_short_options else FAILED,
            bool(sizing.allow_naked_short_options),
            "allow_naked_short_options must be false",
            observed_value=sizing.allow_naked_short_options,
            limit_value=False,
        ),
        RiskCheckResult(
            "max_contracts",
            PASSED if sizing.max_contracts == 0 else FAILED,
            sizing.max_contracts != 0,
            "max_contracts must be 0 while options are disabled",
            observed_value=sizing.max_contracts,
            limit_value=0,
        ),
        RiskCheckResult(
            "max_margin_usage",
            PASSED if sizing.max_margin_usage == 0.0 else FAILED,
            sizing.max_margin_usage != 0.0,
            "max_margin_usage must be 0.0 in Phase 11",
            observed_value=sizing.max_margin_usage,
            limit_value=0.0,
        ),
    ]
    return checks


def _contract_checks(
    contract: ContractSpec,
    config: BrokerConfig,
) -> list[RiskCheckResult]:
    checks: list[RiskCheckResult] = []

    try:
        validate_contract_allowed(contract, config)
        contract_allowed = True
        reason = "contract passes allowed-instrument validation"
    except ContractValidationError as exc:
        contract_allowed = False
        reason = str(exc)

    checks.append(
        RiskCheckResult(
            "contract_allowed",
            PASSED if contract_allowed else FAILED,
            not contract_allowed,
            reason,
            observed_value=contract.symbol,
            limit_value="configured allowed instruments",
        )
    )

    broker_eligible = is_broker_intent_eligible(contract, config)
    checks.append(
        RiskCheckResult(
            "broker_intent_eligible",
            PASSED if broker_eligible else FAILED,
            not broker_eligible,
            (
                "contract is eligible for paper broker intent"
                if broker_eligible
                else "contract is not eligible for allowed paper broker intent"
            ),
            observed_value=contract.tradability_status,
            limit_value="PAPER_ALLOWED",
        )
    )

    if contract.market.upper() == "INDIA":
        manual_overrides = config.raw.get("manual_overrides", {})
        india_overrides = (
            manual_overrides.get("INDIA", {})
            if isinstance(manual_overrides, dict)
            else {}
        )
        verified = bool(india_overrides.get("manual_instrument_verified", False))

        checks.append(
            RiskCheckResult(
                "india_manual_verification",
                PASSED if verified else FAILED,
                not verified,
                (
                    "India manual instrument verification enabled"
                    if verified
                    else "India remains signal-only until manual verification is enabled"
                ),
                observed_value=verified,
                limit_value=True,
            )
        )

    return checks


def _sizing_checks(
    sizing: PaperSizingResult,
    config: BrokerConfig,
) -> list[RiskCheckResult]:
    limits = config.paper_sizing

    checks = [
        RiskCheckResult(
            "phase10_sizing_inputs",
            PASSED if not sizing.used_phase10_performance else FAILED,
            bool(sizing.used_phase10_performance),
            "Phase 10 performance must not be used for sizing",
            observed_value=sizing.used_phase10_performance,
            limit_value=False,
        ),
        RiskCheckResult(
            "max_notional",
            PASSED if sizing.paper_target_notional <= limits.max_notional else FAILED,
            sizing.paper_target_notional > limits.max_notional,
            "paper target notional must not exceed max_notional",
            observed_value=sizing.paper_target_notional,
            limit_value=limits.max_notional,
        ),
        RiskCheckResult(
            "max_delta",
            PASSED if limits.max_delta == 0 else FAILED,
            limits.max_delta != 0,
            "max_delta must remain 0 in Phase 11 default no-greeks mode",
            observed_value=limits.max_delta,
            limit_value=0,
        ),
        RiskCheckResult(
            "max_vega",
            PASSED if limits.max_vega == 0 else FAILED,
            limits.max_vega != 0,
            "max_vega must remain 0 in Phase 11 default no-options mode",
            observed_value=limits.max_vega,
            limit_value=0,
        ),
    ]

    if sizing.paper_quantity is None:
        checks.append(
            RiskCheckResult(
                "max_shares",
                SKIPPED,
                False,
                "paper quantity unavailable because reference price is unavailable",
                observed_value=None,
                limit_value=limits.max_shares,
            )
        )
    else:
        checks.append(
            RiskCheckResult(
                "max_shares",
                PASSED if float(sizing.paper_quantity) <= limits.max_shares else FAILED,
                float(sizing.paper_quantity) > limits.max_shares,
                "paper quantity must not exceed max_shares",
                observed_value=sizing.paper_quantity,
                limit_value=limits.max_shares,
            )
        )

    return checks


def _quote_checks(readiness: QuoteReadinessResult) -> list[RiskCheckResult]:
    checks = [
        RiskCheckResult(
            "quote_available",
            PASSED if readiness.quote_available else FAILED,
            readiness.blocks_allowed_intent and not readiness.quote_available,
            readiness.block_reason,
            observed_value=readiness.quote_available,
            limit_value=True,
        ),
        RiskCheckResult(
            "quote_staleness",
            PASSED if not readiness.is_stale else FAILED,
            readiness.blocks_allowed_intent and readiness.is_stale,
            readiness.block_reason if readiness.is_stale else "quote is not stale",
            observed_value=readiness.is_stale,
            limit_value=readiness.max_quote_age_seconds,
        ),
        RiskCheckResult(
            "quote_spread",
            PASSED if not readiness.spread_too_wide else FAILED,
            readiness.blocks_allowed_intent and readiness.spread_too_wide,
            (
                readiness.block_reason
                if readiness.spread_too_wide
                else "quote spread is within configured limit or unavailable"
            ),
            observed_value=readiness.spread_too_wide,
            limit_value=readiness.max_bid_ask_spread_bps,
        ),
    ]
    return checks


def _allowed_after_kill_switch(before_kill: bool, config: BrokerConfig) -> bool:
    if not before_kill:
        return False

    if config.kill_switch and config.risk_checks.block_if_kill_switch_on:
        return False

    if not config.paper_only:
        return False

    if config.live_orders_enabled:
        return False

    if config.allow_order_placement:
        return False

    return True


def _terminal_summary(
    *,
    market: str,
    symbol: str,
    recommended_action: str,
    config: BrokerConfig,
    final_status: str,
    primary_block_reason: str,
    checks: tuple[RiskCheckResult, ...],
    quote_status: str | None,
) -> RiskCheckSummary:
    return RiskCheckSummary(
        market=market,
        symbol=symbol,
        recommended_action=recommended_action,
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        intent_allowed_before_kill_switch=False,
        intent_allowed_after_kill_switch=False,
        final_status=final_status,
        primary_block_reason=primary_block_reason,
        quote_status=quote_status,
        live_order_sent=False,
        checks=checks,
    )


def _summary_from_checks(
    *,
    market: str,
    symbol: str,
    recommended_action: str,
    config: BrokerConfig,
    checks: tuple[RiskCheckResult, ...],
    quote_status: str | None,
    final_status: str,
    primary_block_reason: str,
    intent_allowed_before_kill_switch: bool = False,
    intent_allowed_after_kill_switch: bool = False,
) -> RiskCheckSummary:
    return RiskCheckSummary(
        market=market,
        symbol=symbol,
        recommended_action=recommended_action,
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        intent_allowed_before_kill_switch=intent_allowed_before_kill_switch,
        intent_allowed_after_kill_switch=intent_allowed_after_kill_switch,
        final_status=final_status,
        primary_block_reason=primary_block_reason,
        quote_status=quote_status,
        live_order_sent=False,
        checks=checks,
    )


def _first_blocking_check(
    checks: list[RiskCheckResult],
    *,
    names: set[str],
) -> RiskCheckResult | None:
    for check in checks:
        if check.check_name in names and check.blocks_intent:
            return check

    return None