"""
Paper sizing for Phase 11.

Sizing is intentionally limited to configured paper notional. It must not use
Phase 10 backtest performance, Sharpe, drawdown, proxy PnL, or inferred margin.

Allowed formula:
    paper_target_notional = abs(target_exposure) * paper_notional_per_full_exposure

Optional share quantity:
    shares = floor(paper_target_notional / reference_price)

Only computed when a reliable reference price exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from vrp.broker.broker_config import BrokerConfig
from vrp.broker.contracts import ContractSpec
from vrp.broker.market_data import QuoteSnapshot, reference_price_from_quote


class PaperSizingError(ValueError):
    """Raised when paper sizing input is invalid."""


@dataclass(frozen=True)
class PaperSizingResult:
    """Paper sizing result for Phase 11."""

    market: str
    symbol: str
    target_exposure: float
    paper_notional_per_full_exposure: float
    paper_target_notional: float
    reference_price: float | None
    paper_quantity: int | float | None
    quantity_type: str
    sizing_status: str
    sizing_reason: str
    max_notional: float
    max_shares: int
    max_contracts: int
    allow_fractional_shares: bool
    round_shares: bool
    used_phase10_performance: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "target_exposure": self.target_exposure,
            "paper_notional_per_full_exposure": self.paper_notional_per_full_exposure,
            "paper_target_notional": self.paper_target_notional,
            "reference_price": self.reference_price,
            "paper_quantity": self.paper_quantity,
            "quantity_type": self.quantity_type,
            "sizing_status": self.sizing_status,
            "sizing_reason": self.sizing_reason,
            "max_notional": self.max_notional,
            "max_shares": self.max_shares,
            "max_contracts": self.max_contracts,
            "allow_fractional_shares": self.allow_fractional_shares,
            "round_shares": self.round_shares,
            "used_phase10_performance": self.used_phase10_performance,
        }


def calculate_paper_target_notional(
    target_exposure: float,
    paper_notional_per_full_exposure: float,
) -> float:
    """Calculate configured paper notional from Phase 9 exposure only."""

    exposure = float(target_exposure)
    notional_per_full = float(paper_notional_per_full_exposure)

    if exposure < -1.0 or exposure > 0.0:
        raise PaperSizingError(
            "target_exposure must be between -1.0 and 0.0 for Phase 11 sizing"
        )

    if notional_per_full < 0.0:
        raise PaperSizingError("paper_notional_per_full_exposure cannot be negative")

    return abs(exposure) * notional_per_full


def calculate_share_quantity(
    paper_target_notional: float,
    reference_price: float,
    *,
    round_shares: bool,
    allow_fractional_shares: bool,
) -> int | float:
    """Calculate paper share quantity from notional and reference price."""

    notional = float(paper_target_notional)
    price = float(reference_price)

    if notional < 0.0:
        raise PaperSizingError("paper_target_notional cannot be negative")

    if price <= 0.0:
        raise PaperSizingError("reference_price must be positive")

    raw_quantity = notional / price

    if allow_fractional_shares:
        return raw_quantity

    if round_shares:
        return int(math.floor(raw_quantity))

    raise PaperSizingError(
        "Cannot compute non-fractional shares when round_shares is false"
    )


def build_paper_sizing(
    *,
    target_exposure: float,
    contract: ContractSpec,
    config: BrokerConfig,
    quote: QuoteSnapshot | None = None,
    reference_price: float | None = None,
) -> PaperSizingResult:
    """Build sizing result for paper intent.

    If quote/reference price is unavailable, this still computes intended
    notional but leaves quantity blank.
    """

    sizing = config.paper_sizing

    paper_target_notional = calculate_paper_target_notional(
        target_exposure=target_exposure,
        paper_notional_per_full_exposure=sizing.paper_notional_per_full_exposure,
    )

    active_reference_price = reference_price
    if active_reference_price is None and quote is not None:
        active_reference_price = reference_price_from_quote(quote)

    paper_quantity: int | float | None = None
    quantity_type = "shares" if contract.sec_type.upper() in {"STK", "ETF", "ETN"} else "units"
    status = "SIZED_NOTIONAL_ONLY"
    reason = "reference price unavailable; quantity not computed"

    if active_reference_price is not None:
        paper_quantity = calculate_share_quantity(
            paper_target_notional=paper_target_notional,
            reference_price=active_reference_price,
            round_shares=sizing.round_shares,
            allow_fractional_shares=sizing.allow_fractional_shares,
        )
        status = "SIZED_NOTIONAL_AND_QUANTITY"
        reason = "notional and paper quantity computed from reference price"

    if paper_target_notional == 0.0:
        status = "ZERO_NOTIONAL"
        reason = "target exposure is zero"
        paper_quantity = 0 if active_reference_price is not None else None

    return PaperSizingResult(
        market=contract.market,
        symbol=contract.symbol,
        target_exposure=float(target_exposure),
        paper_notional_per_full_exposure=sizing.paper_notional_per_full_exposure,
        paper_target_notional=paper_target_notional,
        reference_price=active_reference_price,
        paper_quantity=paper_quantity,
        quantity_type=quantity_type,
        sizing_status=status,
        sizing_reason=reason,
        max_notional=sizing.max_notional,
        max_shares=sizing.max_shares,
        max_contracts=sizing.max_contracts,
        allow_fractional_shares=sizing.allow_fractional_shares,
        round_shares=sizing.round_shares,
        used_phase10_performance=False,
    )


def validate_no_phase10_sizing_inputs(payload: Mapping[str, Any]) -> None:
    """Reject any accidental Phase 10 performance-based sizing inputs."""

    forbidden_keys = {
        "phase_10_total_return_proxy",
        "phase_10_sharpe",
        "phase_10_drawdown",
        "phase_10_net_return_proxy",
        "phase_10_strategy_ranking",
        "phase_10_proxy_pnl",
        "total_return_proxy",
        "sharpe",
        "drawdown",
        "net_return_proxy",
        "strategy_ranking",
        "proxy_pnl",
    }

    present = sorted(forbidden_keys.intersection(set(payload.keys())))

    if present:
        raise PaperSizingError(
            "Phase 11 sizing cannot use Phase 10 performance fields: "
            f"{present}"
        )