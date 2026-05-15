"""Validation utilities for canonical market data.

These functions intentionally fail loudly. Silent repair belongs in explicit cleaning
modules, not validators.
"""

from __future__ import annotations

import pandas as pd

from vrp.data.schema import DATE_COLUMN, OHLC_PRICE_COLUMNS, OHLCV_COLUMNS


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
