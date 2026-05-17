from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vrp.features.realized_variance import (
    annualize_variance,
    annualize_vol,
    build_rv_panel,
    close_to_close_daily_var,
    garman_klass_daily_var,
    parkinson_daily_var,
    rogers_satchell_daily_var,
    rolling_realized_variance,
    validate_ohlc,
    yang_zhang_rolling_var,
)
from vrp.features.returns import (
    add_all_returns,
    add_gap_return,
    add_intraday_return,
    compute_log_returns,
    compute_simple_returns,
)


def make_valid_ohlc(n: int = 30) -> pd.DataFrame:
    """
    Deterministic valid OHLC data for tests.

    Prices are constructed so:
    - open/high/low/close are positive
    - high >= max(open, close)
    - low <= min(open, close)
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    base = pd.Series(np.linspace(100.0, 120.0, n))
    open_ = base
    close = base * 1.001
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": np.arange(n) + 1000,
        }
    )


def make_constant_ohlc(n: int = 30, price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.arange(n) + 1000,
        }
    )


def test_compute_log_returns_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "close": [100.0, 110.0, 121.0],
        }
    )

    out = compute_log_returns(df)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(math.log(110.0 / 100.0))
    assert out.iloc[2] == pytest.approx(math.log(121.0 / 110.0))


def test_compute_simple_returns_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "close": [100.0, 110.0, 121.0],
        }
    )

    out = compute_simple_returns(df)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.10)
    assert out.iloc[2] == pytest.approx(0.10)


def test_add_gap_return_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "open": [100.0, 103.0, 108.0],
            "close": [101.0, 106.0, 109.0],
        }
    )

    out = add_gap_return(df)

    assert pd.isna(out.loc[0, "gap_return"])
    assert out.loc[1, "gap_return"] == pytest.approx(math.log(103.0 / 101.0))
    assert out.loc[2, "gap_return"] == pytest.approx(math.log(108.0 / 106.0))


def test_add_intraday_return_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="B"),
            "open": [100.0, 105.0],
            "close": [102.0, 103.0],
        }
    )

    out = add_intraday_return(df)

    assert out.loc[0, "intraday_return"] == pytest.approx(math.log(102.0 / 100.0))
    assert out.loc[1, "intraday_return"] == pytest.approx(math.log(103.0 / 105.0))


def test_add_all_returns_expected_nan_policy() -> None:
    df = make_valid_ohlc(n=5)

    out = add_all_returns(df)

    assert pd.isna(out.loc[0, "log_return"])
    assert pd.isna(out.loc[0, "simple_return"])
    assert pd.isna(out.loc[0, "gap_return"])
    assert pd.notna(out.loc[0, "intraday_return"])

    assert pd.notna(out.loc[1, "log_return"])
    assert pd.notna(out.loc[1, "simple_return"])
    assert pd.notna(out.loc[1, "gap_return"])
    assert pd.notna(out.loc[1, "intraday_return"])


def test_constant_prices_give_zero_daily_variance() -> None:
    df = make_constant_ohlc(n=30)

    cc = close_to_close_daily_var(df)
    pk = parkinson_daily_var(df)
    gk = garman_klass_daily_var(df)
    rs = rogers_satchell_daily_var(df)

    assert pd.isna(cc.iloc[0])
    assert (cc.dropna() == 0.0).all()
    assert (pk == 0.0).all()
    assert (gk == 0.0).all()
    assert (rs == 0.0).all()


def test_estimators_non_negative_for_valid_ohlc() -> None:
    df = make_valid_ohlc(n=30)

    estimators = [
        close_to_close_daily_var(df),
        parkinson_daily_var(df),
        garman_klass_daily_var(df),
        rogers_satchell_daily_var(df),
    ]

    for series in estimators:
        assert (series.dropna() >= 0.0).all()


def test_validate_ohlc_rejects_non_positive_prices() -> None:
    df = make_valid_ohlc(n=5)
    df.loc[2, "close"] = 0.0

    with pytest.raises(ValueError, match="close"):
        validate_ohlc(df)


def test_validate_ohlc_rejects_high_below_open_or_close() -> None:
    df = make_valid_ohlc(n=5)
    df.loc[2, "high"] = min(df.loc[2, "open"], df.loc[2, "close"]) * 0.99

    with pytest.raises(ValueError, match="high < max"):
        validate_ohlc(df)


def test_validate_ohlc_rejects_low_above_open_or_close() -> None:
    df = make_valid_ohlc(n=5)
    df.loc[2, "low"] = max(df.loc[2, "open"], df.loc[2, "close"]) * 1.01

    with pytest.raises(ValueError, match="low > min"):
        validate_ohlc(df)


def test_close_to_close_daily_var_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "close": [100.0, 110.0, 121.0],
        }
    )

    out = close_to_close_daily_var(df)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(math.log(110.0 / 100.0) ** 2)
    assert out.iloc[2] == pytest.approx(math.log(121.0 / 110.0) ** 2)


def test_parkinson_daily_var_formula() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
        }
    )

    out = parkinson_daily_var(df)

    expected = (math.log(110.0 / 90.0) ** 2) / (4.0 * math.log(2.0))
    assert out.iloc[0] == pytest.approx(expected)


def test_garman_klass_daily_var_formula() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        }
    )

    out = garman_klass_daily_var(df)

    log_hl = math.log(110.0 / 90.0)
    log_co = math.log(105.0 / 100.0)
    expected = 0.5 * (log_hl**2) - ((2.0 * math.log(2.0) - 1.0) * (log_co**2))

    assert out.iloc[0] == pytest.approx(max(expected, 0.0))


def test_rogers_satchell_daily_var_formula() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        }
    )

    out = rogers_satchell_daily_var(df)

    expected = (
        math.log(110.0 / 100.0) * math.log(110.0 / 105.0)
        + math.log(90.0 / 100.0) * math.log(90.0 / 105.0)
    )

    assert out.iloc[0] == pytest.approx(max(expected, 0.0))


def test_rolling_realized_variance_is_trailing_no_lookahead() -> None:
    df = pd.DataFrame({"rv_test_daily": [1.0, 2.0, 3.0, 1000.0]})

    out = rolling_realized_variance(df, "rv_test_daily", window=3)

    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert out.iloc[3] == pytest.approx((2.0 + 3.0 + 1000.0) / 3.0)


def test_rolling_realized_variance_preserves_expected_nan() -> None:
    df = pd.DataFrame({"rv_test_daily": [np.nan, 1.0, 2.0, 3.0, 4.0]})

    out = rolling_realized_variance(df, "rv_test_daily", window=3)

    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert out.iloc[4] == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)


def test_rolling_realized_variance_rejects_non_numeric_non_missing() -> None:
    df = pd.DataFrame({"rv_test_daily": [0.1, "bad", 0.2, 0.3]})

    with pytest.raises(ValueError, match="non-numeric"):
        rolling_realized_variance(df, "rv_test_daily", window=2)


def test_rolling_realized_variance_rejects_negative_variance() -> None:
    df = pd.DataFrame({"rv_test_daily": [0.1, -0.2, 0.3]})

    with pytest.raises(ValueError, match="negative"):
        rolling_realized_variance(df, "rv_test_daily", window=2)


def test_annualize_variance_scalar() -> None:
    assert annualize_variance(0.01, periods=252) == pytest.approx(2.52)


def test_annualize_variance_series() -> None:
    series = pd.Series([0.01, 0.02])

    out = annualize_variance(series, periods=252)

    assert out.iloc[0] == pytest.approx(2.52)
    assert out.iloc[1] == pytest.approx(5.04)


def test_annualize_vol_scalar() -> None:
    assert annualize_vol(0.01, periods=252) == pytest.approx(math.sqrt(2.52))


def test_annualize_vol_rejects_negative_variance() -> None:
    with pytest.raises(ValueError, match="negative"):
        annualize_vol(-0.01, periods=252)


def test_yang_zhang_rolling_var_expected_nan_alignment() -> None:
    df = make_valid_ohlc(n=30)

    out = yang_zhang_rolling_var(df, window=22)

    assert out.name == "rv_yz_22d"
    assert out.iloc[:22].isna().all()
    assert pd.notna(out.iloc[22])
    assert (out.dropna() >= 0.0).all()


def test_yang_zhang_uses_sample_variance_components() -> None:
    df = make_valid_ohlc(n=30)
    window = 22

    out = yang_zhang_rolling_var(df, window=window)

    sorted_df = df.sort_values("date").reset_index(drop=True)
    open_ = pd.Series(sorted_df["open"].to_numpy(), index=sorted_df.index)
    high = pd.Series(sorted_df["high"].to_numpy(), index=sorted_df.index)
    low = pd.Series(sorted_df["low"].to_numpy(), index=sorted_df.index)
    close = pd.Series(sorted_df["close"].to_numpy(), index=sorted_df.index)

    open_return = np.log(open_ / close.shift(1))
    close_return = np.log(close / open_)
    rs = (
        np.log(high / open_) * np.log(high / close)
        + np.log(low / open_) * np.log(low / close)
    )

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    expected = (
        open_return.rolling(window=window, min_periods=window, center=False).var(ddof=1)
        + k * close_return.rolling(window=window, min_periods=window, center=False).var(ddof=1)
        + (1 - k) * rs.rolling(window=window, min_periods=window, center=False).mean()
    )
    expected = expected.clip(lower=0.0).rename("rv_yz_22d")

    pd.testing.assert_series_equal(
        out,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_build_rv_panel_contains_required_columns_and_no_yz_daily() -> None:
    df = make_valid_ohlc(n=30)

    out = build_rv_panel(
        df=df,
        market="US",
        symbol="US_UNDERLYING",
        window=22,
    )

    required = [
        "date",
        "market",
        "symbol",
        "log_return",
        "simple_return",
        "gap_return",
        "intraday_return",
        "rv_cc_daily",
        "rv_parkinson_daily",
        "rv_gk_daily",
        "rv_rs_daily",
        "rv_cc_22d_ann",
        "rv_parkinson_22d_ann",
        "rv_gk_22d_ann",
        "rv_rs_22d_ann",
        "rv_yz_22d_ann",
    ]

    for col in required:
        assert col in out.columns

    assert "rv_yz_daily" not in out.columns
    assert out["market"].eq("US").all()
    assert out["symbol"].eq("US_UNDERLYING").all()


def test_build_rv_panel_expected_nan_policy() -> None:
    df = make_valid_ohlc(n=30)

    out = build_rv_panel(
        df=df,
        market="US",
        symbol="US_UNDERLYING",
        window=22,
    )

    assert pd.isna(out.loc[0, "rv_cc_daily"])
    assert pd.notna(out.loc[0, "rv_parkinson_daily"])
    assert pd.notna(out.loc[0, "rv_gk_daily"])
    assert pd.notna(out.loc[0, "rv_rs_daily"])

    assert out["rv_gk_22d_ann"].iloc[:21].isna().all()
    assert pd.notna(out["rv_gk_22d_ann"].iloc[21])

    assert out["rv_yz_22d_ann"].iloc[:22].isna().all()
    assert pd.notna(out["rv_yz_22d_ann"].iloc[22])


def test_build_rv_panel_primary_column_is_gk_22d_ann() -> None:
    df = make_valid_ohlc(n=30)

    out = build_rv_panel(
        df=df,
        market="INDIA",
        symbol="INDIA_UNDERLYING",
        window=22,
    )

    assert "rv_gk_22d_ann" in out.columns
    assert out["rv_gk_22d_ann"].first_valid_index() == 21


def test_build_rv_panel_invalid_ohlc_raises() -> None:
    df = make_valid_ohlc(n=30)
    df.loc[5, "high"] = df.loc[5, "low"] * 0.99

    with pytest.raises(ValueError):
        build_rv_panel(
            df=df,
            market="US",
            symbol="US_UNDERLYING",
            window=22,
        )


def test_build_rv_panel_window_parameter_changes_column_names() -> None:
    df = make_valid_ohlc(n=15)

    out = build_rv_panel(
        df=df,
        market="US",
        symbol="US_UNDERLYING",
        window=10,
    )

    assert "rv_gk_10d_ann" in out.columns
    assert "rv_yz_10d_ann" in out.columns
    assert "rv_gk_22d_ann" not in out.columns
    assert "rv_yz_22d_ann" not in out.columns