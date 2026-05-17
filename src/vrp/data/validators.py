"""Validation utilities for canonical market data.

Validators fail loudly. They do not repair data. Cleaning and repair must be
done explicitly in source-specific ingestion or cleaner modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from vrp.data.schema import (
    DATA_AUDIT_COLUMNS,
    DATE_COLUMN,
    GROUP_COLUMNS,
    NUMERIC_OHLCV_COLUMNS,
    OHLC_PRICE_COLUMNS,
    OHLCV_COLUMNS,
    PRICE_COLUMNS,
)


@dataclass(frozen=True)
class MissingValueReport:
    """Missing-value summary for one DataFrame column."""

    column: str
    n_missing: int
    missing_fraction: float


def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame contains the canonical OHLCV columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    missing_columns = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")


def validate_required_numeric_columns(df: pd.DataFrame) -> None:
    """Validate that numeric OHLCV columns can be parsed as numeric."""
    validate_ohlcv_schema(df)

    bad_columns: list[str] = []
    for column in NUMERIC_OHLCV_COLUMNS:
        parsed = pd.to_numeric(df[column], errors="coerce")
        original_missing = df[column].isna()
        introduced_missing = parsed.isna() & ~original_missing
        if introduced_missing.any():
            bad_columns.append(column)

    if bad_columns:
        raise ValueError(f"Non-numeric values detected in columns: {bad_columns}")


def validate_no_duplicate_dates(df: pd.DataFrame) -> None:
    """Validate no duplicate dates within each market-symbol pair."""
    validate_ohlcv_schema(df)

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Date column contains missing values.")

    duplicated_mask = df.duplicated(subset=GROUP_COLUMNS + [DATE_COLUMN], keep=False)

    if duplicated_mask.any():
        duplicate_rows = df.loc[duplicated_mask, GROUP_COLUMNS + [DATE_COLUMN]]
        raise ValueError(
            "Duplicate dates detected within market-symbol group: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )


def validate_sorted_dates(df: pd.DataFrame) -> None:
    """Validate that dates are sorted ascending within each market-symbol pair."""
    validate_ohlcv_schema(df)

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Date column contains missing values.")

    grouped = df.groupby(GROUP_COLUMNS, sort=False, dropna=False)

    unsorted_groups: list[dict[str, Any]] = []
    for (market, symbol), group in grouped:
        dates = pd.to_datetime(group[DATE_COLUMN], errors="coerce")
        if dates.isna().any():
            unsorted_groups.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "reason": "unparseable_date",
                }
            )
            continue

        if not dates.is_monotonic_increasing:
            unsorted_groups.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "reason": "dates_not_sorted",
                }
            )

    if unsorted_groups:
        raise ValueError(f"Dates are not sorted within groups: {unsorted_groups}")


def validate_non_negative_prices(df: pd.DataFrame) -> None:
    """Validate that price columns are non-negative."""
    validate_ohlcv_schema(df)
    validate_required_numeric_columns(df)

    price_frame = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if price_frame.isna().any().any():
        missing_columns = price_frame.columns[price_frame.isna().any()].tolist()
        raise ValueError(f"Price columns contain missing values: {missing_columns}")

    if (price_frame < 0).any().any():
        bad_columns = price_frame.columns[(price_frame < 0).any()].tolist()
        raise ValueError(f"Negative price values detected in columns: {bad_columns}")


def validate_non_negative_volume(df: pd.DataFrame) -> None:
    """Validate that volume is non-negative.

    Close-only volatility-index sources may set volume to 0.
    """
    validate_ohlcv_schema(df)
    validate_required_numeric_columns(df)

    volume = pd.to_numeric(df["volume"], errors="coerce")

    if volume.isna().any():
        raise ValueError("Volume column contains missing values.")

    if (volume < 0).any():
        raise ValueError("Negative volume values detected.")


def validate_ohlc_bounds(df: pd.DataFrame) -> None:
    """Validate basic OHLC bounds.

    Rules
    -----
    1. OHLC prices must be non-negative.
    2. high >= max(open, close).
    3. low <= min(open, close).
    4. high >= low.
    """
    validate_ohlcv_schema(df)
    validate_non_negative_prices(df)

    ohlc = df[OHLC_PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    high_below_open_or_close = ohlc["high"] < ohlc[["open", "close"]].max(axis=1)
    if high_below_open_or_close.any():
        bad_rows = df.loc[high_below_open_or_close, ["date", "open", "high", "close"]]
        raise ValueError(
            "Invalid OHLC bounds: high is below open or close. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )

    low_above_open_or_close = ohlc["low"] > ohlc[["open", "close"]].min(axis=1)
    if low_above_open_or_close.any():
        bad_rows = df.loc[low_above_open_or_close, ["date", "open", "low", "close"]]
        raise ValueError(
            "Invalid OHLC bounds: low is above open or close. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )

    high_below_low = ohlc["high"] < ohlc["low"]
    if high_below_low.any():
        bad_rows = df.loc[high_below_low, ["date", "high", "low"]]
        raise ValueError(
            "Invalid OHLC bounds: high is below low. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )


def validate_ohlcv_frame(df: pd.DataFrame) -> None:
    """Run all canonical OHLCV validations."""
    validate_ohlcv_schema(df)
    validate_required_numeric_columns(df)
    validate_no_duplicate_dates(df)
    validate_sorted_dates(df)
    validate_non_negative_prices(df)
    validate_non_negative_volume(df)
    validate_ohlc_bounds(df)


def build_missing_value_report(df: pd.DataFrame) -> list[MissingValueReport]:
    """Build a missing-value report for all columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    n_rows = len(df)
    reports: list[MissingValueReport] = []

    for column in df.columns:
        n_missing = int(df[column].isna().sum())
        missing_fraction = 0.0 if n_rows == 0 else n_missing / n_rows
        reports.append(
            MissingValueReport(
                column=str(column),
                n_missing=n_missing,
                missing_fraction=float(missing_fraction),
            )
        )

    return reports


def build_data_audit_row(
    df: pd.DataFrame,
    *,
    market: str,
    dataset: str,
    source: str,
    symbol: str | None,
) -> dict[str, Any]:
    """Build one audit-table row for a canonical data frame.

    The audit row is produced even when validation fails. In that case,
    `validation_status` contains the error message.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    n_rows = len(df)

    if DATE_COLUMN in df.columns and n_rows > 0:
        parsed_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        start_date = _date_to_string(parsed_dates.min())
        end_date = _date_to_string(parsed_dates.max())
    else:
        start_date = None
        end_date = None

    if "close" in df.columns and n_rows > 0:
        close = pd.to_numeric(df["close"], errors="coerce")
        n_missing_close = int(close.isna().sum())
        min_close = _float_or_none(close.min(skipna=True))
        max_close = _float_or_none(close.max(skipna=True))
    else:
        n_missing_close = n_rows
        min_close = None
        max_close = None

    n_duplicate_dates = _count_duplicate_market_symbol_dates(df)

    try:
        validate_ohlcv_frame(df)
        validation_status = "PASS"
    except Exception as exc:  # noqa: BLE001
        validation_status = f"FAIL: {exc}"

    row = {
        "market": market,
        "dataset": dataset,
        "source": source,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": n_rows,
        "n_missing_close": n_missing_close,
        "n_duplicate_dates": n_duplicate_dates,
        "min_close": min_close,
        "max_close": max_close,
        "validation_status": validation_status,
    }

    return {column: row[column] for column in DATA_AUDIT_COLUMNS}


def build_missing_value_report_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame version of the missing-value report."""
    reports = build_missing_value_report(df)
    return pd.DataFrame(
        [
            {
                "column": report.column,
                "n_missing": report.n_missing,
                "missing_fraction": report.missing_fraction,
            }
            for report in reports
        ]
    )


def _count_duplicate_market_symbol_dates(df: pd.DataFrame) -> int:
    if not all(column in df.columns for column in GROUP_COLUMNS + [DATE_COLUMN]):
        return 0

    duplicated_mask = df.duplicated(subset=GROUP_COLUMNS + [DATE_COLUMN], keep=False)
    return int(duplicated_mask.sum())


def _date_to_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)