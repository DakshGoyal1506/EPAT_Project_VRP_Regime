# src/vrp/features/returns.py

"""
Return construction utilities

Scope:
- Daily close-to-close log returns
- Daily close-to-close simple returns
- Overnight/gap log returns: log(open_t / close_{t-1})
- Intraday log returns: log(close_t / open_t)

Conventions:
- All log returns use natural logarithms.
- The first close-to-close return is NaN because close_{t-1} is unavailable.
- The first gap return is NaN because close_{t-1} is unavailable.
- Intraday return is available whenever open_t and close_t are valid.
"""

from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd

def _required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _validate_positive_prices(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """
    Raise ValueError if price columns contain zero, negative, or non-finite values.

    Log ratios are undefined for non-positive prices.
    """
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce')
        
        invalid_mask = series.isna() | (series <= 0) | ~np.isfinite(series)
        if invalid_mask.any():
            bad_count = int(invalid_mask.sum())
            raise ValueError(f"Column '{col}' contains {bad_count} non-positive or non-finite values at indices: {series[invalid_mask].index.tolist()}")
        
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

def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Compute daily close-to-close log returns.
    
    Formula: log_return_t = log(price_t / price_{t-1})

    Parameters:
    - df: DataFrame containing price data.
    - price_col: Column name for closing prices (default "close").

    Returns:
    - Series of log returns, indexed the same as input DataFrame.
      The first return is NaN due to lack of prior price.
    """
    _required_columns(df, [price_col])
    
    sorted_df = _sort_by_date(df)
    _validate_positive_prices(sorted_df, [price_col])
    
    prices = pd.to_numeric(sorted_df[price_col], errors='coerce')
    log_returns = pd.Series(
        np.log(prices / prices.shift(1)),
        index=prices.index,
        name="log_return",
    )
    
    return log_returns
    
def compute_simple_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Compute daily close-to-close simple returns.
    
    Formula: simple_return_t = (price_t / price_{t-1}) - 1

    Parameters:
    - df: DataFrame containing price data.
    - price_col: Column name for closing prices (default "close").

    Returns:
    - Series of simple returns, indexed the same as input DataFrame.
      The first return is NaN due to lack of prior price.
    """
    _required_columns(df, [price_col])
    
    sorted_df = _sort_by_date(df)
    _validate_positive_prices(sorted_df, [price_col])
    
    prices = pd.to_numeric(sorted_df[price_col], errors='coerce')
    simple_returns = pd.Series(
        (prices / prices.shift(1)) - 1,
        index=prices.index,
        name="simple_return",
    )
    
    return simple_returns

def add_gap_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overnight/gap log returns.
    
    Formula: gap_return_t = log(open_t / close_{t-1})

    Parameters:
    - df: DataFrame containing price data.
    Returns:
    - DataFrame with added gap log return column.
    """
    _required_columns(df, ["open", "close"])
    
    sorted_df = _sort_by_date(df)
    _validate_positive_prices(sorted_df, ["open", "close"])
    
    opens = pd.to_numeric(sorted_df["open"], errors='coerce')
    closes = pd.to_numeric(sorted_df["close"], errors='coerce')
    
    sorted_df["gap_return"] = np.log(opens / closes.shift(1))
    
    return sorted_df

def add_intraday_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intraday log returns.

    Args:
        df (pd.DataFrame): DataFrame containing 'open' and 'close' price columns.

    Returns:
        pd.DataFrame: DataFrame with added intraday log return column.
    """
    
    _required_columns(df, ["open", "close"])
    
    sorted_df = _sort_by_date(df)
    _validate_positive_prices(sorted_df, ["open", "close"])
    
    opens = pd.to_numeric(sorted_df["open"], errors='coerce')
    closes = pd.to_numeric(sorted_df["close"], errors='coerce')
    
    sorted_df["intraday_return"] = np.log(closes / opens)
    
    return sorted_df

def add_all_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute all types of returns for the given DataFrame.
    - log returns
    - simple returns
    - gap returns
    - intraday returns

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        price_col (str): Column name for closing prices (default "close").

    Returns:
        pd.DataFrame: DataFrame with added log and simple returns columns.
    """
    df = add_gap_return(df)
    df = add_intraday_return(df)
    df["log_return"] = compute_log_returns(df, price_col)
    df["simple_return"] = compute_simple_returns(df, price_col)
    return df