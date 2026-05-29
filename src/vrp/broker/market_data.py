"""
Quote schema and market-data validation for Phase 11.

This module does not fetch live broker data. It only defines the normalized
quote shape used by Phase 11 risk checks and paper sizing.

Quotes are optional. Missing broker data must be represented structurally
rather than crashing the Phase 11 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from vrp.broker.broker_config import BrokerConfig


class MarketDataError(ValueError):
    """Raised when quote data is malformed."""


QUOTE_AVAILABLE = "QUOTE_AVAILABLE"
QUOTE_MISSING = "QUOTE_MISSING"
QUOTE_INVALID = "QUOTE_INVALID"
QUOTE_STALE = "QUOTE_STALE"
QUOTE_SPREAD_TOO_WIDE = "QUOTE_SPREAD_TOO_WIDE"


@dataclass(frozen=True)
class QuoteSnapshot:
    """Normalized quote snapshot for Phase 11.

    bid/ask/last/mid may be None only for missing or unavailable quote records.
    timestamp_utc must be timezone-aware when present.
    """

    symbol: str
    market: str
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    timestamp_utc: datetime | None
    quote_age_seconds: float | None
    spread_bps: float | None
    source: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "timestamp_utc": (
                self.timestamp_utc.isoformat() if self.timestamp_utc is not None else None
            ),
            "quote_age_seconds": self.quote_age_seconds,
            "spread_bps": self.spread_bps,
            "source": self.source,
            "status": self.status,
        }


@dataclass(frozen=True)
class QuoteReadinessResult:
    """Result of quote checks against Phase 11 risk config."""

    quote_status: str
    quote_available: bool
    quote_valid: bool
    is_stale: bool
    spread_too_wide: bool
    blocks_allowed_intent: bool
    block_reason: str
    max_quote_age_seconds: int
    max_bid_ask_spread_bps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "quote_status": self.quote_status,
            "quote_available": self.quote_available,
            "quote_valid": self.quote_valid,
            "is_stale": self.is_stale,
            "spread_too_wide": self.spread_too_wide,
            "blocks_allowed_intent": self.blocks_allowed_intent,
            "block_reason": self.block_reason,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "max_bid_ask_spread_bps": self.max_bid_ask_spread_bps,
        }


def quote_from_mapping(
    data: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    default_source: str = "manual_or_broker",
) -> QuoteSnapshot:
    """Build a normalized QuoteSnapshot from a mapping.

    Expected keys:
    symbol, market, bid, ask, last, timestamp_utc, source

    mid, spread_bps, and quote_age_seconds are computed when possible.
    """

    symbol = str(data.get("symbol", "")).strip().upper()
    market = str(data.get("market", "")).strip().upper()

    if not symbol:
        raise MarketDataError("Quote is missing symbol")

    if not market:
        raise MarketDataError("Quote is missing market")

    bid = _optional_float(data.get("bid"), field_name="bid")
    ask = _optional_float(data.get("ask"), field_name="ask")
    last = _optional_float(data.get("last"), field_name="last")
    source = str(data.get("source", default_source))

    timestamp_raw = data.get("timestamp_utc")
    timestamp = _optional_datetime_utc(timestamp_raw, field_name="timestamp_utc")

    status = str(data.get("status", "") or QUOTE_AVAILABLE)

    mid = _compute_mid(bid, ask, data.get("mid"))
    spread_bps = _compute_spread_bps(bid, ask, mid, data.get("spread_bps"))

    quote_age_seconds = _compute_quote_age_seconds(
        timestamp,
        now_utc=now_utc,
        provided_age=data.get("quote_age_seconds"),
    )

    quote = QuoteSnapshot(
        symbol=symbol,
        market=market,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        timestamp_utc=timestamp,
        quote_age_seconds=quote_age_seconds,
        spread_bps=spread_bps,
        source=source,
        status=status,
    )

    return validate_quote_snapshot(quote)


def missing_quote(
    *,
    symbol: str,
    market: str,
    source: str = "broker_unavailable",
    status: str = QUOTE_MISSING,
) -> QuoteSnapshot:
    """Create a structured missing-quote snapshot."""

    symbol_clean = str(symbol).strip().upper()
    market_clean = str(market).strip().upper()

    if not symbol_clean:
        raise MarketDataError("Missing quote requires symbol")

    if not market_clean:
        raise MarketDataError("Missing quote requires market")

    return QuoteSnapshot(
        symbol=symbol_clean,
        market=market_clean,
        bid=None,
        ask=None,
        last=None,
        mid=None,
        timestamp_utc=None,
        quote_age_seconds=None,
        spread_bps=None,
        source=source,
        status=status,
    )


def validate_quote_snapshot(quote: QuoteSnapshot) -> QuoteSnapshot:
    """Validate quote fields.

    Full bid/ask validation is required for QUOTE_AVAILABLE. Missing quotes are
    allowed as structured unavailable data.
    """

    if quote.status == QUOTE_MISSING:
        return quote

    if quote.bid is None and quote.ask is None and quote.last is None:
        raise MarketDataError("Available quote must contain bid/ask or last")

    if quote.bid is not None and quote.bid < 0:
        raise MarketDataError("Quote bid must be non-negative")

    if quote.ask is not None and quote.ask < 0:
        raise MarketDataError("Quote ask must be non-negative")

    if quote.last is not None and quote.last < 0:
        raise MarketDataError("Quote last must be non-negative")

    if quote.bid is not None and quote.ask is not None:
        if quote.ask < quote.bid:
            raise MarketDataError("Quote ask must be greater than or equal to bid")

        expected_mid = (quote.bid + quote.ask) / 2.0

        if quote.mid is None:
            raise MarketDataError("Quote mid could not be computed")

        if abs(quote.mid - expected_mid) > 1e-9:
            raise MarketDataError("Quote mid must equal average of bid and ask")

        if expected_mid > 0:
            expected_spread = (quote.ask - quote.bid) / expected_mid * 10000.0
            if quote.spread_bps is None:
                raise MarketDataError("Quote spread_bps could not be computed")

            if abs(quote.spread_bps - expected_spread) > 1e-6:
                raise MarketDataError(
                    "Quote spread_bps must equal (ask - bid) / mid * 10000"
                )

    if quote.quote_age_seconds is not None and quote.quote_age_seconds < 0:
        raise MarketDataError("Quote age cannot be negative")

    return quote


def reference_price_from_quote(quote: QuoteSnapshot) -> float | None:
    """Return usable reference price for paper sizing.

    Preference:
    1. mid
    2. last
    """

    if quote.status == QUOTE_MISSING:
        return None

    if quote.mid is not None and quote.mid > 0:
        return quote.mid

    if quote.last is not None and quote.last > 0:
        return quote.last

    return None


def quote_is_stale(
    quote: QuoteSnapshot,
    *,
    max_quote_age_seconds: int,
) -> bool:
    """Return whether a quote is stale."""

    if quote.status == QUOTE_MISSING:
        return False

    if quote.quote_age_seconds is None:
        return False

    return quote.quote_age_seconds > max_quote_age_seconds


def quote_spread_too_wide(
    quote: QuoteSnapshot,
    *,
    max_bid_ask_spread_bps: float,
) -> bool:
    """Return whether bid-ask spread exceeds configured threshold."""

    if quote.status == QUOTE_MISSING:
        return False

    if quote.spread_bps is None:
        return False

    return quote.spread_bps > max_bid_ask_spread_bps


def evaluate_quote_readiness(
    quote: QuoteSnapshot | None,
    config: BrokerConfig,
) -> QuoteReadinessResult:
    """Evaluate quote readiness under Phase 11 risk config.

    Missing quotes may be allowed depending on config. Even when missing quotes
    are allowed, share quantity must not be computed later unless a reliable
    reference price is available.
    """

    risk = config.risk_checks

    max_age = int(risk.max_quote_age_seconds)
    max_spread = float(risk.max_bid_ask_spread_bps)

    if quote is None:
        quote_available = False
        quote_valid = False
        blocks = bool(risk.block_missing_quote or risk.require_quote_for_allowed_intent)
        reason = (
            "quote missing and config blocks missing quote"
            if blocks
            else "quote missing but config permits signal-only continuation"
        )

        return QuoteReadinessResult(
            quote_status=QUOTE_MISSING,
            quote_available=quote_available,
            quote_valid=quote_valid,
            is_stale=False,
            spread_too_wide=False,
            blocks_allowed_intent=blocks,
            block_reason=reason,
            max_quote_age_seconds=max_age,
            max_bid_ask_spread_bps=max_spread,
        )

    if quote.status == QUOTE_MISSING:
        blocks = bool(risk.block_missing_quote or risk.require_quote_for_allowed_intent)
        reason = (
            "quote missing and config blocks missing quote"
            if blocks
            else "quote missing but config permits signal-only continuation"
        )

        return QuoteReadinessResult(
            quote_status=QUOTE_MISSING,
            quote_available=False,
            quote_valid=False,
            is_stale=False,
            spread_too_wide=False,
            blocks_allowed_intent=blocks,
            block_reason=reason,
            max_quote_age_seconds=max_age,
            max_bid_ask_spread_bps=max_spread,
        )

    try:
        validate_quote_snapshot(quote)
    except MarketDataError as exc:
        return QuoteReadinessResult(
            quote_status=QUOTE_INVALID,
            quote_available=True,
            quote_valid=False,
            is_stale=False,
            spread_too_wide=False,
            blocks_allowed_intent=True,
            block_reason=str(exc),
            max_quote_age_seconds=max_age,
            max_bid_ask_spread_bps=max_spread,
        )

    is_stale = quote_is_stale(quote, max_quote_age_seconds=max_age)
    spread_wide = quote_spread_too_wide(
        quote,
        max_bid_ask_spread_bps=max_spread,
    )

    if is_stale and risk.block_stale_quote_if_quote_available:
        return QuoteReadinessResult(
            quote_status=QUOTE_STALE,
            quote_available=True,
            quote_valid=True,
            is_stale=True,
            spread_too_wide=spread_wide,
            blocks_allowed_intent=True,
            block_reason="quote available but stale",
            max_quote_age_seconds=max_age,
            max_bid_ask_spread_bps=max_spread,
        )

    if spread_wide:
        return QuoteReadinessResult(
            quote_status=QUOTE_SPREAD_TOO_WIDE,
            quote_available=True,
            quote_valid=True,
            is_stale=is_stale,
            spread_too_wide=True,
            blocks_allowed_intent=True,
            block_reason="quote bid-ask spread exceeds configured limit",
            max_quote_age_seconds=max_age,
            max_bid_ask_spread_bps=max_spread,
        )

    return QuoteReadinessResult(
        quote_status=QUOTE_AVAILABLE,
        quote_available=True,
        quote_valid=True,
        is_stale=is_stale,
        spread_too_wide=False,
        blocks_allowed_intent=False,
        block_reason="quote passed configured readiness checks",
        max_quote_age_seconds=max_age,
        max_bid_ask_spread_bps=max_spread,
    )


def quote_record_for_report(
    quote: QuoteSnapshot | None,
    *,
    symbol: str,
    market: str,
) -> dict[str, Any]:
    """Return JSON/CSV-friendly quote record for reports."""

    if quote is None:
        return missing_quote(symbol=symbol, market=market).as_dict()

    return quote.as_dict()


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MarketDataError(f"Quote field {field_name!r} must be numeric") from exc


def _optional_datetime_utc(value: Any, *, field_name: str) -> datetime | None:
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError as exc:
            raise MarketDataError(f"Quote field {field_name!r} is not parseable") from exc
    else:
        raise MarketDataError(f"Quote field {field_name!r} must be datetime or string")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _compute_mid(
    bid: float | None,
    ask: float | None,
    provided_mid: Any,
) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0

    return _optional_float(provided_mid, field_name="mid")


def _compute_spread_bps(
    bid: float | None,
    ask: float | None,
    mid: float | None,
    provided_spread_bps: Any,
) -> float | None:
    if bid is not None and ask is not None and mid is not None and mid > 0:
        return (ask - bid) / mid * 10000.0

    return _optional_float(provided_spread_bps, field_name="spread_bps")


def _compute_quote_age_seconds(
    timestamp: datetime | None,
    *,
    now_utc: datetime | None,
    provided_age: Any,
) -> float | None:
    explicit_age = _optional_float(provided_age, field_name="quote_age_seconds")
    if explicit_age is not None:
        return explicit_age

    if timestamp is None:
        return None

    effective_now = now_utc or datetime.now(timezone.utc)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)

    effective_now = effective_now.astimezone(timezone.utc)
    return (effective_now - timestamp).total_seconds()