"""Validation utilities for canonical market data.

These functions intentionally fail loudly. Silent repair belongs in explicit cleaning
modules, not validators.
"""

from __future__ import annotations

import pandas as pd

from vrp.data.schema import (
    DATE_COLUMN,
    NUMERIC_OHLCV_COLUMNS,
    OHLC_PRICE_COLUMNS,
    OHLCV_COLUMNS,
)


def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame contains the canonical OHLCV columns.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If any required column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    missing_columns = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")


def validate_no_duplicate_dates(df: pd.DataFrame) -> None:
    """Validate that no duplicate dates exist within each market-symbol pair.

    The canonical dataset may contain multiple symbols. Duplicate dates are checked
    within (`market`, `symbol`) when those columns exist; otherwise the date column
    alone is checked.
    """
    validate_ohlcv_schema(df)

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Date column contains missing values.")

    group_columns = ["market", "symbol"]
    if all(col in df.columns for col in group_columns):
        duplicated_mask = df.duplicated(subset=group_columns + [DATE_COLUMN], keep=False)
    else:
        duplicated_mask = df.duplicated(subset=[DATE_COLUMN], keep=False)

    if duplicated_mask.any():
        duplicate_rows = df.loc[duplicated_mask, group_columns + [DATE_COLUMN]]
        raise ValueError(
            "Duplicate dates detected within market-symbol group: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )


def validate_monotonic_dates(df: pd.DataFrame) -> None:
    """Validate that dates are strictly increasing within each market-symbol pair."""
    validate_ohlcv_schema(df)

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Date column contains missing values.")

    parsed_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Date column contains non-datetime values.")

    frame = df.copy()
    frame[DATE_COLUMN] = parsed_dates
    group_columns = ["market", "symbol"]

    if all(col in frame.columns for col in group_columns):
        for _, group in frame.groupby(group_columns, sort=False):
            if not group[DATE_COLUMN].is_monotonic_increasing:
                raise ValueError("Dates must be monotonic increasing within market-symbol groups.")
    elif not frame[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Dates must be monotonic increasing.")


def validate_numeric_columns_present(df: pd.DataFrame) -> None:
    """Validate numeric OHLCV columns are present and numeric-castable."""
    validate_ohlcv_schema(df)

    for column in NUMERIC_OHLCV_COLUMNS:
        series = df[column]
        converted = pd.to_numeric(series, errors="coerce")

        # Permit nulls for now, but reject non-null values that fail numeric coercion.
        non_null_invalid = series.notna() & converted.isna()
        if non_null_invalid.any():
            raise ValueError(f"Column '{column}' contains non-numeric values.")


def validate_non_negative_volume(df: pd.DataFrame) -> None:
    """Validate volume is present, numeric-castable, and non-negative."""
    validate_ohlcv_schema(df)

    converted = pd.to_numeric(df["volume"], errors="coerce")
    if converted.isna().any():
        raise ValueError("Volume column contains missing or non-numeric values.")
    if (converted < 0).any():
        raise ValueError("Volume must be non-negative.")


def validate_adj_close_positive_or_nullable_policy(df: pd.DataFrame) -> None:
    """Validate adj_close follows the current policy: nullable, but positive when present."""
    validate_ohlcv_schema(df)

    converted = pd.to_numeric(df["adj_close"], errors="coerce")
    invalid_non_null = df["adj_close"].notna() & converted.isna()
    if invalid_non_null.any():
        raise ValueError("adj_close contains non-numeric values.")

    non_null_values = converted[df["adj_close"].notna()]
    if (non_null_values <= 0).any():
        raise ValueError("adj_close must be strictly positive when present.")


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

    price_frame = df[OHLC_PRICE_COLUMNS]

    if price_frame.isna().any().any():
        raise ValueError("OHLC price columns contain missing values.")

    if (price_frame < 0).any().any():
        bad_columns = price_frame.columns[(price_frame < 0).any()].tolist()
        raise ValueError(f"Negative OHLC values detected in columns: {bad_columns}")

    high_below_open_or_close = df["high"] < df[["open", "close"]].max(axis=1)
    if high_below_open_or_close.any():
        bad_rows = df.loc[high_below_open_or_close, ["date", "open", "high", "close"]]
        raise ValueError(
            "Invalid OHLC bounds: high is below open or close. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )

    low_above_open_or_close = df["low"] > df[["open", "close"]].min(axis=1)
    if low_above_open_or_close.any():
        bad_rows = df.loc[low_above_open_or_close, ["date", "open", "low", "close"]]
        raise ValueError(
            "Invalid OHLC bounds: low is above open or close. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )

    high_below_low = df["high"] < df["low"]
    if high_below_low.any():
        bad_rows = df.loc[high_below_low, ["date", "high", "low"]]
        raise ValueError(
            "Invalid OHLC bounds: high is below low. "
            f"Rows: {bad_rows.to_dict(orient='records')}"
        )
