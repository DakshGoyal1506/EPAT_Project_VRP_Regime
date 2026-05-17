# src/vrp/features/realized_variance.py

"""
Realised variance estimators for Phase 2.

Scope:
- Daily close-to-close variance
- Daily Parkinson variance
- Daily Garman-Klass variance
- Daily Rogers-Satchell variance
- Rolling Yang-Zhang variance
- Rolling annualised realised variance panels

Conventions:
- All ratios are log ratios using natural logarithms.
- Daily estimator outputs are daily variance estimates, not volatility.
- Rolling realised variance uses trailing rolling mean of daily variance.
- Annualised variance = daily variance rate * 252.
- Primary project feature = rv_gk_22d_ann.
- Yang-Zhang is rolling-only. No rv_yz_daily column is created.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from vrp.features.returns import add_all_returns

OHLC_COLUMNS = ["open", "high", "low", "close"]

def _required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copied DataFrame sorted by date if a date column exists.

    The returned index is reset so downstream rolling logic uses clean row order.
    """
    out = df.copy()
    
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors='coerce')
        out = out.sort_values("date").reset_index(drop=True)
    
    return out

def _validate_window(window: int) -> None:
    """
    Validate rolling window length.

    Yang-Zhang requires window > 1 because the k coefficient uses (window - 1).
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError(f"Window must be an integer >= 2. Got: {window}")

def _validate_annualization_factor(factor: float) -> None:
    """
    Validate annualisation factor.

    Must be positive and finite.
    """
    if not isinstance(factor, (int, float, np.integer, np.floating)) or factor <= 0 or not np.isfinite(factor):
        raise ValueError(f"Annualisation factor must be a positive finite number. Got: {factor}")

def _coerce_ohlc_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce OHLC columns to numeric, converting invalid entries to NaN.

    This allows the variance calculations to proceed while naturally handling invalid data.
    """
    _required_columns(df, OHLC_COLUMNS)
    out = df.copy()
    
    for col in OHLC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    
    return out

def validate_ohlc(df: pd.DataFrame) -> None:
    """
    Validate OHLC prices before realised-variance construction.

    Required:
    - open/high/low/close exist
    - all OHLC values are positive and finite
    - high >= max(open, close)
    - low <= min(open, close)
    - high >= low
    """
    _required_columns(df, OHLC_COLUMNS)

    ohlc = _coerce_ohlc_to_numeric(df)

    for col in OHLC_COLUMNS:
        values = ohlc[col]
        invalid_mask = values.isna() | ~np.isfinite(values) | (values <= 0)

        if invalid_mask.any():
            bad_count = int(invalid_mask.sum())
            raise ValueError(f"Column '{col}' contains {bad_count} non-positive or non-finite values at indices: {values[invalid_mask].index.tolist()}")

    high = ohlc["high"]
    low = ohlc["low"]
    open_ = ohlc["open"]
    close = ohlc["close"]

    bad_high_mask = high < np.maximum(open_, close)
    if bad_high_mask.any():
        bad_count = int(bad_high_mask.sum())
        raise ValueError(
            f"Found {bad_count} row(s) where high < max(open, close)."
        )

    bad_low_mask = low > np.minimum(open_, close)
    if bad_low_mask.any():
        bad_count = int(bad_low_mask.sum())
        raise ValueError(
            f"Found {bad_count} row(s) where low > min(open, close)."
        )

    bad_range_mask = high < low
    if bad_range_mask.any():
        bad_count = int(bad_range_mask.sum())
        raise ValueError(f"Found {bad_count} row(s) where high < low.")

def _non_negative_variance(series: pd.Series, name: str) -> pd.Series:
    """
    Clip tiny negative numerical artefacts to zero.

    The estimators should be non-negative under valid OHLC data. Floating-point
    arithmetic can still create very small negative values around zero.
    """
    tolerance = 1e-14
    materially_negative = series < -tolerance

    if materially_negative.any():
        bad_count = int(materially_negative.sum())
        raise ValueError(
            f"Estimator '{name}' produced {bad_count} materially negative variance value(s)."
        )

    out = series.clip(lower=0.0)
    out.name = name
    return out

def close_to_close_daily_var(df: pd.DataFrame) -> pd.Series:
    """
    Compute close-to-close daily realised variance.

    Formula:
        rv_cc_daily_t = [log(close_t / close_{t-1})]^2

    Returns
    -------
    pd.Series
        Daily variance series named "rv_cc_daily".
    """
    _required_columns(df, ["close"])

    out = _sort_by_date(df)
    close = pd.to_numeric(out["close"], errors="coerce")

    invalid_mask = close.isna() | ~np.isfinite(close) | (close <= 0)
    if invalid_mask.any():
        bad_count = int(invalid_mask.sum())
        raise ValueError(
            f"Column 'close' contains {bad_count} invalid value(s). "
            "Close prices must be positive and finite."
        )

    var = pd.Series(np.log(close / close.shift(1)) ** 2, index=close.index, name="rv_cc_daily")
    return _non_negative_variance(var, "rv_cc_daily")

def parkinson_daily_var(df: pd.DataFrame) -> pd.Series:
    """
    Compute Parkinson daily variance estimator.

    Formula:
        rv_parkinson_daily_t =
            [log(high_t / low_t)]^2 / [4 * log(2)]

    Returns
    -------
    pd.Series
        Daily variance series named "rv_parkinson_daily".
    """
    
    _required_columns(df, ["high", "low"])
    
    out = _sort_by_date(df)
    out = _coerce_ohlc_to_numeric(out)
    validate_ohlc(out)
    
    high = out["high"]
    low = out["low"]
    
    log_hl = np.log(high / low)
    var = pd.Series((1 / (4 * np.log(2))) * log_hl ** 2, index=out.index, name="rv_parkinson_daily")
    return _non_negative_variance(var, "rv_parkinson_daily")

def garman_klass_daily_var(df: pd.DataFrame) -> pd.Series:
    """
    Compute Garman-Klass daily variance estimator.
    
    Formula:
        rv_gk_daily_t =
            0.5 * [log(high_t / low_t)]^2
            - (2 * log(2) - 1) * [log(close_t / open_t)]^2
    Returns
    -------
    pd.Series
        Daily variance series named "rv_gk_daily".
    """
    
    _required_columns(df, ["high", "low", "open", "close"])
    
    out = _sort_by_date(df)
    out = _coerce_ohlc_to_numeric(out)
    validate_ohlc(out)
    
    high = out["high"]
    low = out["low"]
    open = out["open"]
    close = out["close"]
    
    log_hl = np.log(high / low)
    log_oc = np.log(close / open)
    
    var = pd.Series(0.5 * (log_hl ** 2) - ((2.0 * np.log(2.0) - 1) * (log_oc ** 2)), index = out.index, name = "rv_gk_daily")
    return _non_negative_variance(var, "rv_gk_daily")

def rogers_satchell_daily_var(df: pd.DataFrame) -> pd.Series:
    """
    Compute Rogers-Stachel daily variance estimator.

    Formula:
        rv_rs_daily_t =
            [log(high_t / open_t) * log(high_t / close_t)] +
            [log(low_t / open_t) * log(low_t / close_t)]
    Returns
    -------
    pd.Series
        Daily variance series named "rv_rs_daily".
    """
    
    _required_columns(df, ["high", "low", "open", "close"])
    
    out = _sort_by_date(df)
    out = _coerce_ohlc_to_numeric(out)
    validate_ohlc(out)
    
    high = out["high"]
    low = out["low"]
    open = out["open"]
    close = out["close"]
    
    log_ho = np.log(high / open)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open)
    log_lc = np.log(low / close)
    
    var = pd.Series((log_ho * log_hc) + (log_lo * log_lc), index = out.index, name="rv_rs_daily")
    return _non_negative_variance(var, "rv_rs_daily")

def rolling_realized_variance(
    df: pd.DataFrame,
    var_col: str,
    window: int = 22,
) -> pd.Series:
    
    """
    Compute trailing rolling mean of a daily variance column.

    Convention:
        rolling_rv_t = mean(daily_var_{t-window+1}, ..., daily_var_t)

    This returns a daily variance rate, not annualised variance.

    Parameters
    ----------
    df:
        DataFrame containing `var_col`.
    var_col:
        Name of daily variance column.
    window:
        Trailing window length in trading days.

    Returns
    -------
    pd.Series
        Rolling daily variance-rate series.
    """
    
    _validate_window(window)
    _required_columns(df, [var_col])
    
    raw = df[var_col]
    values = pd.to_numeric(raw, errors='coerce')

    bad_parse_mask = raw.notna() & values.isna()
    if bad_parse_mask.any():
        bad_count = int(bad_parse_mask.sum())
        raise ValueError(f"Column '{var_col}' contains {bad_count} non-numeric value(s) that cannot be coerced to numeric.")

    if (values.dropna() < 0).any():
        raise ValueError(f"Column '{var_col}' contains negative variance values.")
    
    rolling_var = values.rolling(window=window, min_periods=window, center=False).mean()
    
    rolling_var.name = f"{var_col.replace('_daily', '')}_{window}d"
    return rolling_var

def annualize_variance(var: pd.Series | float, periods: int = 252) -> pd.Series | float:
    """
    Annualise variance.

    Formula:
        annualised_variance = variance_per_period * periods
    """
    _validate_annualization_factor(periods)

    return var * periods

def annualize_vol(var: pd.Series | float, periods: int = 252) -> pd.Series | float:
    """
    Annualise volatility from a variance input.

    Formula:
        annualised_volatility = sqrt(variance_per_period * periods)
    """
    _validate_annualization_factor(periods)

    annulaised_var = annualize_variance(var, periods)
    
    if isinstance(annulaised_var, pd.Series):
        if(annulaised_var.dropna() < 0).any():
            raise ValueError("Annualised variance contains negative values, cannot compute volatility.")
        return pd.Series(np.sqrt(annulaised_var), index=annulaised_var.index)
    
    if annulaised_var < 0:
        raise ValueError("Annualised variance is negative, cannot compute volatility.")
    
    return float(np.sqrt(annulaised_var))

def yang_zhang_rolling_var(df: pd.DataFrame, window: int = 22) -> pd.Series:
    """
    Compute rolling Yang-Zhang variance estimator.

    Yang-Zhang is rolling-only here. It is not emitted as rv_yz_daily.

    Definitions:
        open_return_t  = log(open_t / close_{t-1})
        close_return_t = log(close_t / open_t)

        rs_t =
            log(high_t / close_t) * log(high_t / open_t)
            +
            log(low_t / close_t) * log(low_t / open_t)

        k =
            0.34 / [1.34 + (window + 1) / (window - 1)]

        yz_rolling_var_t =
            rolling_var(open_return, window)
            + k * rolling_var(close_return, window)
            + (1 - k) * rolling_mean(rs_t, window)

    Returns
    -------
    pd.Series
        Rolling daily variance-rate series named "rv_yz_{window}d".
        Annualisation is handled separately.
    """
    _validate_window(window)
    _required_columns(df, ["open", "high", "low", "close"])
    
    out = _sort_by_date(df)
    out = _coerce_ohlc_to_numeric(out)
    validate_ohlc(out)

    open = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]

    open_return = pd.Series(np.log(open / close.shift(1)), index=out.index, name="open_return")
    close_return = pd.Series(np.log(close / open), index=out.index, name="close_return")

    log_ho = np.log(high / open)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open)
    log_lc = np.log(low / close)

    rs = (log_ho * log_hc) + (log_lo * log_lc)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_open = open_return.rolling(
        window=window,
        min_periods=window,
        center=False,
    ).var(ddof=1)
    var_close = close_return.rolling(
        window=window,
        min_periods=window,
        center=False,
    ).var(ddof=1)
    rs_series = pd.Series(rs, index=out.index)
    mean_rs = rs_series.rolling(window=window, min_periods=window, center=False).mean()

    yz_var = var_open + (k * var_close) + ((1 - k) * mean_rs)
    
    yz_var.name = f"rv_yz_{window}d"
    
    return _non_negative_variance(yz_var, yz_var.name)

def build_rv_panel(
    df: pd.DataFrame,
    market: str,
    symbol: str,
    window: int = 22,
    annualization_periods: int = 252,
) -> pd.DataFrame:
    """
    Build complete realised-variance panel for one market.

    Output columns:
        date
        market
        symbol
        log_return
        simple_return
        gap_return
        intraday_return
        rv_cc_daily
        rv_parkinson_daily
        annualized_var = annualize_variance(var, periods)
        rv_rs_daily
        if isinstance(annualized_var, pd.Series):
            if(annualized_var.dropna() < 0).any():
        rv_gk_{window}d_ann
            return pd.Series(np.sqrt(annualized_var), index=annualized_var.index)
        rv_yz_{window}d_ann
        if annualized_var < 0:
    Notes
    -----
        return float(np.sqrt(annualized_var))
    - No rv_yz_daily column is created.
    """
    _required_columns(df, ["date", "open", "high", "low", "close"])
    
    out = _sort_by_date(df)
    out = _coerce_ohlc_to_numeric(out)
    validate_ohlc(out)

    out["market"] = market
    out["symbol"] = symbol

    out = add_all_returns(out, price_col="close")

    out["rv_cc_daily"] = close_to_close_daily_var(out)
    out["rv_parkinson_daily"] = parkinson_daily_var(out)
    out["rv_gk_daily"] = garman_klass_daily_var(out)
    out["rv_rs_daily"] = rogers_satchell_daily_var(out)

    daily_to_rolling = {
        "rv_cc_daily": f"rv_cc_{window}d_ann",
        "rv_parkinson_daily": f"rv_parkinson_{window}d_ann",
        "rv_gk_daily": f"rv_gk_{window}d_ann",
        "rv_rs_daily": f"rv_rs_{window}d_ann",
    }

    for daily_col, ann_col in daily_to_rolling.items():
        rolling_var = rolling_realized_variance(
            out,
            var_col=daily_col,
            window=window,
        )
        out[ann_col] = annualize_variance(
            rolling_var,
            periods=annualization_periods,
        )

    yz_rolling_var = yang_zhang_rolling_var(out, window=window)
    out[f"rv_yz_{window}d_ann"] = annualize_variance(
        yz_rolling_var,
        periods=annualization_periods,
    )

    out["market"] = str(market).upper()
    out["symbol"] = str(symbol)

    ordered_columns = [
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
        f"rv_cc_{window}d_ann",
        f"rv_parkinson_{window}d_ann",
        f"rv_gk_{window}d_ann",
        f"rv_rs_{window}d_ann",
        f"rv_yz_{window}d_ann",
    ]

    return out[ordered_columns].copy()