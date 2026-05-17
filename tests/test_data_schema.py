from __future__ import annotations

import pandas as pd
import pytest

from vrp.data.schema import DATA_AUDIT_COLUMNS, OHLCV_COLUMNS
from vrp.data.validators import (
    build_data_audit_row,
    build_missing_value_report,
    build_missing_value_report_frame,
    validate_no_duplicate_dates,
    validate_non_negative_prices,
    validate_non_negative_volume,
    validate_ohlc_bounds,
    validate_ohlcv_frame,
    validate_ohlcv_schema,
    validate_required_numeric_columns,
    validate_sorted_dates,
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


def test_data_audit_columns_are_defined() -> None:
    expected_columns = [
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

    assert DATA_AUDIT_COLUMNS == expected_columns


def test_validate_ohlcv_schema_accepts_valid_frame() -> None:
    df = make_valid_ohlcv_frame()

    validate_ohlcv_schema(df)


def test_validate_ohlcv_schema_rejects_missing_column() -> None:
    df = make_valid_ohlcv_frame().drop(columns=["source"])

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        validate_ohlcv_schema(df)


def test_validate_required_numeric_columns_rejects_non_numeric_price() -> None:
    df = make_valid_ohlcv_frame()
    df["close"] = df["close"].astype("object")
    df.loc[0, "close"] = "bad_value"

    with pytest.raises(ValueError, match="Non-numeric values detected"):
        validate_required_numeric_columns(df)


def test_validate_no_duplicate_dates_accepts_unique_dates() -> None:
    df = make_valid_ohlcv_frame()

    validate_no_duplicate_dates(df)


def test_validate_no_duplicate_dates_rejects_duplicate_market_symbol_date() -> None:
    df = make_valid_ohlcv_frame()
    duplicate_row = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate dates detected"):
        validate_no_duplicate_dates(df)


def test_validate_sorted_dates_accepts_sorted_dates() -> None:
    df = make_valid_ohlcv_frame()

    validate_sorted_dates(df)


def test_validate_sorted_dates_rejects_unsorted_dates_within_group() -> None:
    df = make_valid_ohlcv_frame().iloc[[1, 0]].reset_index(drop=True)

    with pytest.raises(ValueError, match="Dates are not sorted"):
        validate_sorted_dates(df)


def test_validate_non_negative_prices_accepts_valid_prices() -> None:
    df = make_valid_ohlcv_frame()

    validate_non_negative_prices(df)


def test_validate_non_negative_prices_rejects_negative_adj_close() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "adj_close"] = -1.0

    with pytest.raises(ValueError, match="Negative price values"):
        validate_non_negative_prices(df)


def test_validate_non_negative_volume_accepts_zero_volume() -> None:
    df = make_valid_ohlcv_frame()
    df["volume"] = 0

    validate_non_negative_volume(df)


def test_validate_non_negative_volume_rejects_negative_volume() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="Negative volume values"):
        validate_non_negative_volume(df)


def test_validate_ohlc_bounds_accepts_valid_prices() -> None:
    df = make_valid_ohlcv_frame()

    validate_ohlc_bounds(df)


def test_validate_ohlc_bounds_rejects_negative_prices() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "open"] = -100.0

    with pytest.raises(ValueError, match="Negative price values"):
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


def test_validate_ohlcv_frame_accepts_valid_frame() -> None:
    df = make_valid_ohlcv_frame()

    validate_ohlcv_frame(df)


def test_build_missing_value_report_counts_missing_values() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "close"] = None

    report = build_missing_value_report(df)
    close_report = next(item for item in report if item.column == "close")

    assert close_report.n_missing == 1
    assert close_report.missing_fraction == 0.5


def test_build_missing_value_report_frame_returns_dataframe() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "close"] = None

    report = build_missing_value_report_frame(df)

    assert {"column", "n_missing", "missing_fraction"}.issubset(report.columns)
    assert report.loc[report["column"] == "close", "n_missing"].iloc[0] == 1


def test_build_data_audit_row_for_valid_frame() -> None:
    df = make_valid_ohlcv_frame()

    row = build_data_audit_row(
        df,
        market="US",
        dataset="us_underlying",
        source="unit_test",
        symbol="TEST",
    )

    assert list(row.keys()) == DATA_AUDIT_COLUMNS
    assert row["market"] == "US"
    assert row["dataset"] == "us_underlying"
    assert row["source"] == "unit_test"
    assert row["symbol"] == "TEST"
    assert row["start_date"] == "2024-01-02"
    assert row["end_date"] == "2024-01-03"
    assert row["n_rows"] == 2
    assert row["n_missing_close"] == 0
    assert row["n_duplicate_dates"] == 0
    assert row["min_close"] == 102.0
    assert row["max_close"] == 102.5
    assert row["validation_status"] == "PASS"


def test_build_data_audit_row_records_validation_failure() -> None:
    df = make_valid_ohlcv_frame()
    df.loc[0, "high"] = 50.0

    row = build_data_audit_row(
        df,
        market="US",
        dataset="us_underlying",
        source="unit_test",
        symbol="TEST",
    )

    assert row["validation_status"].startswith("FAIL:")
