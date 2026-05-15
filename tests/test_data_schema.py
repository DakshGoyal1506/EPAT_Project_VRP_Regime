from __future__ import annotations

import pandas as pd
import pytest

from vrp.data.schema import OHLCV_COLUMNS
from vrp.data.validators import (
    validate_no_duplicate_dates,
    validate_ohlc_bounds,
    validate_ohlcv_schema,
)


def make_valid_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [105.0, 103.0],
            "low": [99.0, 100.0],
            "close": [102.0, 102.5],
            "adj_close": [102.0, 102.5],
            "volume": [1_000_000, 1_100_000],
            "source": ["unit_test", "unit_test"],
            "market": ["US", "US"],
            "symbol": ["TEST", "TEST"],
        }
    )


def test_ohlcv_columns_are_defined() -> None:
    expected_columns = [
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

    assert OHLCV_COLUMNS == expected_columns


def test_validate_ohlcv_schema_accepts_valid_frame() -> None:
    df = make_valid_ohlcv_frame()

    validate_ohlcv_schema(df)


def test_validate_ohlcv_schema_rejects_missing_column() -> None:
    df = make_valid_ohlcv_frame().drop(columns=["source"])

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        validate_ohlcv_schema(df)


def test_validate_no_duplicate_dates_accepts_unique_dates() -> None:
    df = make_valid_ohlcv_frame()

    validate_no_duplicate_dates(df)


def test_validate_no_duplicate_dates_rejects_duplicate_market_symbol_date() -> None:
    df = make_valid_ohlcv_frame()
    duplicate_row = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate dates detected"):
        validate_no_duplicate_dates(df)


def test_validate_ohlc_bounds_accepts_valid_prices() -> None:
    df = make_valid_ohlcv_frame()

    validate_ohlc_bounds(df)


def test_validate_ohlc_bounds_rejects_negative_prices() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "open"] = -100.0

    with pytest.raises(ValueError, match="Negative OHLC values"):
        validate_ohlc_bounds(df)


def test_validate_ohlc_bounds_rejects_high_below_open() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "high"] = 99.5

    with pytest.raises(ValueError, match="high is below open or close"):
        validate_ohlc_bounds(df)


def test_validate_ohlc_bounds_rejects_low_above_close() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "low"] = 103.0

    with pytest.raises(ValueError, match="low is above open or close"):
        validate_ohlc_bounds(df)


def test_validate_ohlc_bounds_rejects_high_below_low() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "high"] = 98.0
    df.loc[0, "low"] = 99.0

    with pytest.raises(ValueError):
        validate_ohlc_bounds(df)
