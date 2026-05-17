# tests/test_implied_variance.py

from __future__ import annotations

import pandas as pd
import pytest

from vrp.features.implied_variance import (
    build_implied_variance,
    infer_iv_close_column,
    validate_vix_values,
)


def test_infer_iv_close_column_prefers_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "close": [10.0, 20.0, 30.0],
        }
    )

    assert infer_iv_close_column(df) == "close"


def test_infer_iv_close_column_accepts_vix_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "vix_close": [10.0, 20.0, 30.0],
        }
    )

    assert infer_iv_close_column(df) == "vix_close"


def test_infer_iv_close_column_accepts_india_vix_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "india_vix_close": [10.0, 20.0, 30.0],
        }
    )

    assert infer_iv_close_column(df) == "india_vix_close"


def test_infer_iv_close_column_uses_single_numeric_non_date_column() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "PX_LAST": [10.0, 20.0, 30.0],
        }
    )

    assert infer_iv_close_column(df) == "PX_LAST"


def test_infer_iv_close_column_accepts_adj_close_with_space() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "Adj Close": [10.0, 20.0, 30.0],
        }
    )

    assert infer_iv_close_column(df) == "Adj Close"


def test_infer_iv_close_column_rejects_ambiguous_numeric_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "a": [10.0, 20.0, 30.0],
            "b": [11.0, 21.0, 31.0],
        }
    )

    with pytest.raises(ValueError, match="Could not infer"):
        infer_iv_close_column(df)


def test_validate_vix_values_accepts_valid_percent_levels() -> None:
    df = pd.DataFrame({"iv_close": [12.5, 20.0, 45.0]})

    validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_zero() -> None:
    df = pd.DataFrame({"iv_close": [12.5, 0.0, 45.0]})

    with pytest.raises(ValueError, match="<="):
        validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_negative() -> None:
    df = pd.DataFrame({"iv_close": [12.5, -1.0, 45.0]})

    with pytest.raises(ValueError, match="<="):
        validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_missing_values() -> None:
    df = pd.DataFrame({"iv_close": [12.5, None, 45.0]})

    with pytest.raises(ValueError, match="missing"):
        validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_non_numeric_strings() -> None:
    df = pd.DataFrame({"iv_close": [12.5, "bad", 45.0]})

    with pytest.raises(ValueError, match="non-numeric"):
        validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_decimal_scaled_values() -> None:
    df = pd.DataFrame({"iv_close": [0.12, 0.15, 0.20]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_vix_values(df, iv_col="iv_close")


def test_validate_vix_values_rejects_extreme_values() -> None:
    df = pd.DataFrame({"iv_close": [12.5, 201.0, 45.0]})

    with pytest.raises(ValueError, match=">="):
        validate_vix_values(df, iv_col="iv_close", max_value=200.0)


def test_build_implied_variance_us_formula_and_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "close": [10.0, 20.0, 30.0],
        }
    )

    out = build_implied_variance(df, market="US")

    assert list(out.columns) == [
        "date",
        "market",
        "iv_symbol",
        "iv_close",
        "iv_ann",
    ]
    assert out["market"].eq("US").all()
    assert out["iv_symbol"].eq("VIX").all()

    assert out.loc[0, "iv_ann"] == pytest.approx((10.0 / 100.0) ** 2)
    assert out.loc[1, "iv_ann"] == pytest.approx((20.0 / 100.0) ** 2)
    assert out.loc[2, "iv_ann"] == pytest.approx((30.0 / 100.0) ** 2)


def test_build_implied_variance_india_symbol() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "close": [10.0, 20.0, 30.0],
        }
    )

    out = build_implied_variance(df, market="INDIA")

    assert out["market"].eq("INDIA").all()
    assert out["iv_symbol"].eq("INDIA_VIX").all()


def test_build_implied_variance_sorts_by_date() -> None:
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "close": [30.0, 10.0, 20.0],
        }
    )

    out = build_implied_variance(df, market="US")

    assert out["date"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    assert out["iv_close"].tolist() == [10.0, 20.0, 30.0]


def test_build_implied_variance_rejects_duplicate_dates() -> None:
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
            ],
            "close": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="duplicated"):
        build_implied_variance(df, market="US")


def test_build_implied_variance_rejects_invalid_market() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "close": [10.0, 20.0, 30.0],
        }
    )

    with pytest.raises(ValueError, match="Unsupported market"):
        build_implied_variance(df, market="EU")