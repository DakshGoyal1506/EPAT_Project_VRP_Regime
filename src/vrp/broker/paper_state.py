"""
Optional paper position state for Phase 11.

Position state is off by default. It exists only so Phase 11 can distinguish:

    target_exposure == 0 -> STAY_FLAT

from:

    previous paper exposure < 0 and current target_exposure == 0 -> REDUCE_TO_ZERO

No live broker position is read here.
No live broker order is placed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class PaperStateError(ValueError):
    """Raised when paper position state is malformed."""


PAPER_POSITION_STATE_COLUMNS = [
    "updated_at_utc",
    "market",
    "strategy_name",
    "symbol",
    "target_trade_date",
    "target_exposure",
    "paper_quantity",
    "side",
    "status",
]


@dataclass(frozen=True)
class PaperPositionState:
    """Latest tracked paper position state."""

    updated_at_utc: str
    market: str
    strategy_name: str
    symbol: str
    target_trade_date: str | None
    target_exposure: float
    paper_quantity: float | None
    side: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "updated_at_utc": self.updated_at_utc,
            "market": self.market,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "target_trade_date": self.target_trade_date,
            "target_exposure": self.target_exposure,
            "paper_quantity": self.paper_quantity,
            "side": self.side,
            "status": self.status,
        }


def load_paper_position_state(path: str | Path) -> pd.DataFrame:
    """Load optional paper position state.

    Missing file returns an empty state frame.
    """

    state_path = Path(path)
    if not state_path.exists():
        return pd.DataFrame(columns=PAPER_POSITION_STATE_COLUMNS)

    frame = pd.read_csv(state_path)

    missing = sorted(set(PAPER_POSITION_STATE_COLUMNS).difference(frame.columns))
    if missing:
        raise PaperStateError(f"Paper position state missing columns: {missing}")

    if frame.empty:
        return frame

    frame = frame.copy()
    frame["market"] = frame["market"].astype(str).str.upper()
    frame["strategy_name"] = frame["strategy_name"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["target_exposure"] = pd.to_numeric(frame["target_exposure"], errors="coerce")
    frame["paper_quantity"] = pd.to_numeric(frame["paper_quantity"], errors="coerce")

    if frame["target_exposure"].isna().any():
        raise PaperStateError("Paper position state contains invalid target_exposure")

    return frame


def get_latest_position_state(
    path: str | Path,
    *,
    market: str,
    strategy_name: str,
) -> PaperPositionState | None:
    """Return latest paper position state for market and strategy."""

    frame = load_paper_position_state(path)

    if frame.empty:
        return None

    filtered = frame[
        (frame["market"].astype(str).str.upper() == market.upper())
        & (frame["strategy_name"].astype(str) == strategy_name)
    ].copy()

    if filtered.empty:
        return None

    filtered["_updated_sort"] = pd.to_datetime(
        filtered["updated_at_utc"],
        errors="coerce",
    )

    if filtered["_updated_sort"].isna().all():
        filtered["_updated_sort"] = pd.RangeIndex(len(filtered))

    filtered = filtered.sort_values("_updated_sort", ascending=True)
    latest = filtered.iloc[-1].to_dict()

    quantity = latest.get("paper_quantity")
    if pd.isna(quantity):
        quantity = None

    target_trade_date = latest.get("target_trade_date")
    if pd.isna(target_trade_date):
        target_trade_date = None

    return PaperPositionState(
        updated_at_utc=str(latest["updated_at_utc"]),
        market=str(latest["market"]).upper(),
        strategy_name=str(latest["strategy_name"]),
        symbol=str(latest["symbol"]).upper(),
        target_trade_date=str(target_trade_date) if target_trade_date is not None else None,
        target_exposure=float(latest["target_exposure"]),
        paper_quantity=float(quantity) if quantity is not None else None,
        side=str(latest["side"]),
        status=str(latest["status"]),
    )


def has_prior_short_state(
    path: str | Path,
    *,
    market: str,
    strategy_name: str,
) -> bool:
    """Return whether prior paper state shows short-vol exposure."""

    state = get_latest_position_state(
        path,
        market=market,
        strategy_name=strategy_name,
    )

    if state is None:
        return False

    return state.target_exposure < 0.0


def write_paper_position_state(
    states: list[PaperPositionState],
    path: str | Path,
) -> Path:
    """Write paper position states to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [state.as_dict() for state in states]
    pd.DataFrame(records, columns=PAPER_POSITION_STATE_COLUMNS).to_csv(
        output_path,
        index=False,
    )

    return output_path


def make_position_state_from_intent_record(
    intent_record: dict[str, Any],
) -> PaperPositionState:
    """Create paper state from a paper-intent record."""

    return PaperPositionState(
        updated_at_utc=str(
            intent_record.get("run_timestamp_utc") or _utc_now_iso()
        ),
        market=str(intent_record["market"]).upper(),
        strategy_name=str(intent_record["strategy_name"]),
        symbol=str(intent_record["symbol"]).upper(),
        target_trade_date=_optional_str(intent_record.get("target_trade_date")),
        target_exposure=float(intent_record.get("target_exposure", 0.0)),
        paper_quantity=_optional_float(intent_record.get("paper_quantity")),
        side=str(intent_record.get("side", "")),
        status=str(intent_record.get("final_status", "")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if value == "":
        return None

    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if value == "":
        return None

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()