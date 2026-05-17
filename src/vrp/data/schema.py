"""Canonical data schemas for the VRP project.

Phase 1 defines the canonical OHLCV schema and data-audit schema used by all
public daily data sources.
"""

from __future__ import annotations

from typing import Final

DATE_COLUMN: Final[str] = "date"

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

PRICE_COLUMNS: Final[list[str]] = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
]

METADATA_COLUMNS: Final[list[str]] = [
    "source",
    "market",
    "symbol",
]

GROUP_COLUMNS: Final[list[str]] = [
    "market",
    "symbol",
]

DATA_AUDIT_COLUMNS: Final[list[str]] = [
    "market",
    "dataset",
    "source",
    "symbol",
    "start_date",
    "end_date",
    "n_rows",
    "n_missing_close",
    "n_duplicate_dates",
    "min_close",
    "max_close",
    "validation_status",
]