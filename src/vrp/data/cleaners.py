"""Column standardisation utilities for Phase 1 data ingestion.

Cleaners normalize source-specific DataFrames into the canonical OHLCV schema:

date, open, high, low, close, adj_close, volume, source, market, symbol

Cleaning rules:
- Do not forward-fill prices.
- Do not merge sources.
- Do not infer missing market data silently.
- Close-only sources are allowed only through explicit close-series conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from vrp.data.schema import OHLCV_COLUMNS


COMMON_DATE_COLUMNS = [
    "date",
    "Date",
    "DATE",
    "timestamp",
    "Timestamp",
    "TIMESTAMP",
    "HistoricalDate",
    "historical_date",
]

COMMON_OHLCV_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low": ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE", "Adj Close", "Adj_Close", "Last", "LAST"],
    "adj_close": ["adj_close", "Adj Close", "Adj_Close", "adjusted_close"],
    "volume": ["volume", "Volume", "VOLUME"],
}


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, which can appear in yfinance output."""
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        flattened: list[str] = []
        for column_tuple in out.columns:
            non_empty_parts = [str(part) for part in column_tuple if str(part)]
            flattened.append("_".join(non_empty_parts))
        out.columns = flattened
    else:
        out.columns = [str(col) for col in out.columns]

    return out


def normalize_column_label(label: str) -> str:
    """Normalize source labels for matching."""
    return (
        str(label)
        .strip()
        .replace("\ufeff", "")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with whitespace-stripped string column names."""
    out = flatten_columns(df)
    out.columns = [str(col).strip() for col in out.columns]
    return out


def find_first_existing_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Find the first matching column using exact and normalized matching."""
    exact_lookup = {col: col for col in columns}
    normalized_lookup = {normalize_column_label(col): col for col in columns}

    for candidate in candidates:
        if candidate in exact_lookup:
            return exact_lookup[candidate]

        normalized_candidate = normalize_column_label(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    return None


def require_column(
    columns: list[str],
    candidates: list[str],
    canonical_name: str,
) -> str:
    """Find a source column or raise a clear error."""
    found = find_first_existing_column(columns, candidates)
    if found is None:
        raise ValueError(
            f"Could not find required column for '{canonical_name}'. "
            f"Tried candidates: {candidates}. Available columns: {columns}"
        )
    return found


def standardize_ohlcv_columns(
    df: pd.DataFrame,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: Mapping[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Standardize a source DataFrame that contains full OHLCV-like data.

    Parameters
    ----------
    df:
        Source DataFrame.
    market:
        Market code.
    source:
        Source identifier.
    symbol:
        Source symbol.
    column_map:
        Optional mapping from canonical column names to source column names or
        candidate lists. Use this for CBOE/NSE CSVs with non-standard headers.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    source_df = normalize_source_columns(df)

    if source_df.empty:
        raise ValueError(f"Source DataFrame is empty for source={source}, symbol={symbol}")

    if _has_datetime_index(source_df):
        source_df = source_df.reset_index()

    columns = [str(col) for col in source_df.columns]

    date_column = _resolve_column(
        columns=columns,
        canonical_name="date",
        default_candidates=COMMON_DATE_COLUMNS,
        column_map=column_map,
    )
    open_column = _resolve_column(
        columns=columns,
        canonical_name="open",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["open"],
        column_map=column_map,
    )
    high_column = _resolve_column(
        columns=columns,
        canonical_name="high",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["high"],
        column_map=column_map,
    )
    low_column = _resolve_column(
        columns=columns,
        canonical_name="low",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["low"],
        column_map=column_map,
    )
    close_column = _resolve_column(
        columns=columns,
        canonical_name="close",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["close"],
        column_map=column_map,
    )

    adj_close_column = _resolve_optional_column(
        columns=columns,
        canonical_name="adj_close",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["adj_close"],
        column_map=column_map,
    )
    volume_column = _resolve_optional_column(
        columns=columns,
        canonical_name="volume",
        default_candidates=COMMON_OHLCV_COLUMN_CANDIDATES["volume"],
        column_map=column_map,
    )

    out = pd.DataFrame(
        {
            "date": _to_date_series(source_df[date_column]),
            "open": pd.to_numeric(source_df[open_column], errors="coerce"),
            "high": pd.to_numeric(source_df[high_column], errors="coerce"),
            "low": pd.to_numeric(source_df[low_column], errors="coerce"),
            "close": pd.to_numeric(source_df[close_column], errors="coerce"),
            "adj_close": (
                pd.to_numeric(source_df[adj_close_column], errors="coerce")
                if adj_close_column is not None
                else pd.to_numeric(source_df[close_column], errors="coerce")
            ),
            "volume": (
                pd.to_numeric(source_df[volume_column], errors="coerce")
                if volume_column is not None
                else 0
            ),
            "source": source,
            "market": market,
            "symbol": symbol,
        }
    )

    return _finalize_canonical_frame(out)


def standardize_yahoo_ohlcv(
    df: pd.DataFrame,
    *,
    market: str,
    source: str,
    symbol: str,
) -> pd.DataFrame:
    """Standardize yfinance OHLCV output into canonical schema."""
    return standardize_ohlcv_columns(
        df,
        market=market,
        source=source,
        symbol=symbol,
        column_map={
            "date": ["Date", "Datetime", "index", "date"],
            "open": ["Open", "open"],
            "high": ["High", "high"],
            "low": ["Low", "low"],
            "close": ["Close", "close"],
            "adj_close": ["Adj Close", "Adj_Close", "adj_close", "Close", "close"],
            "volume": ["Volume", "volume"],
        },
    )


def standardize_single_close_series(
    df: pd.DataFrame,
    *,
    date_column: str,
    close_column: str,
    market: str,
    source: str,
    symbol: str,
) -> pd.DataFrame:
    """Convert a date/close-only series into canonical OHLCV shape.

    This is used for sources such as FRED VIXCLS. It is explicit, not silent:
    open, high, low, close, and adj_close are all set equal to close, while
    volume is set to 0.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    source_df = normalize_source_columns(df)

    if source_df.empty:
        raise ValueError(f"Close-only DataFrame is empty for source={source}, symbol={symbol}")

    if _has_datetime_index(source_df):
        source_df = source_df.reset_index()

    columns = [str(col) for col in source_df.columns]
    resolved_date = require_column(columns, [date_column], "date")
    resolved_close = require_column(columns, [close_column], "close")

    close = pd.to_numeric(source_df[resolved_close], errors="coerce")

    out = pd.DataFrame(
        {
            "date": _to_date_series(source_df[resolved_date]),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": 0,
            "source": source,
            "market": market,
            "symbol": symbol,
        }
    )

    return _finalize_canonical_frame(out)


def _resolve_column(
    *,
    columns: list[str],
    canonical_name: str,
    default_candidates: list[str],
    column_map: Mapping[str, str | list[str]] | None,
) -> str:
    candidates = _get_candidates(
        canonical_name=canonical_name,
        default_candidates=default_candidates,
        column_map=column_map,
    )
    return require_column(columns, candidates, canonical_name)


def _resolve_optional_column(
    *,
    columns: list[str],
    canonical_name: str,
    default_candidates: list[str],
    column_map: Mapping[str, str | list[str]] | None,
) -> str | None:
    candidates = _get_candidates(
        canonical_name=canonical_name,
        default_candidates=default_candidates,
        column_map=column_map,
    )
    return find_first_existing_column(columns, candidates)


def _get_candidates(
    *,
    canonical_name: str,
    default_candidates: list[str],
    column_map: Mapping[str, str | list[str]] | None,
) -> list[str]:
    if column_map is None or canonical_name not in column_map:
        return default_candidates

    mapped = column_map[canonical_name]
    if isinstance(mapped, str):
        return [mapped]
    return [str(item) for item in mapped]


def _has_datetime_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None


def _to_date_series(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.tz_localize(None).dt.normalize()


def _finalize_canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, OHLCV_COLUMNS].copy()
    out = out.sort_values(["market", "symbol", "date"]).reset_index(drop=True)
    return out