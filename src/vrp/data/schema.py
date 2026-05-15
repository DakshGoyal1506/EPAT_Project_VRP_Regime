"""Canonical data schemas for the VRP project.

Phase 0 defines only the minimal OHLCV schema. Later phases will add schemas for
realised variance, implied variance, VRP, regimes, signals, and backtest results.
"""

from __future__ import annotations

from typing import Final

OHLCV_COLUMNS: Final[list[str]] = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "market",
    "symbol",
]

OHLC_PRICE_COLUMNS: Final[list[str]] = [
    "open",
    "high",
    "low",
    "close",
]

NUMERIC_OHLCV_COLUMNS: Final[list[str]] = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

METADATA_COLUMNS: Final[list[str]] = [
    "source",
    "market",
    "symbol",
]

DATE_COLUMN: Final[str] = "date"
