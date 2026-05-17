# src/vrp/features/implied_variance.py

"""
Implied variance construction for Phase 3.

Scope:
- Read cleaned VIX / India VIX panels from Phase 1.
- Infer the VIX close column.
- Validate VIX index values.
- Convert VIX index close into annualised implied variance.

Definitions:
    iv_ann_t = (iv_close_t / 100)^2

Output columns:
    date
    market
    iv_symbol
    iv_close
    iv_ann
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_IV_SYMBOLS = {
    "US": "VIX",
    "INDIA": "INDIA_VIX",
}

IV_CLOSE_CANDIDATES = [
    "iv_close",
    "vix_close",
    "india_vix_close",
    "close",
    "Close",
    "CLOSE",
    "adj_close",
    "Adj Close",
    "Adj_Close",
    "ADJ_CLOSE",
    "Adj_close",
]

def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    
def _normalize_market(market: str) -> str:
    """
    Normalize supported market labels.
    """
    market_norm = str(market).upper().strip()

    if market_norm not in DEFAULT_IV_SYMBOLS:
        raise ValueError(
            f"Unsupported market '{market}'. "
            f"Expected one of: {sorted(DEFAULT_IV_SYMBOLS)}"
        )

    return market_norm

def infer_iv_close_column(vix_df: pd.DataFrame) -> str:
    """
    Infer the VIX / India VIX close column.

    Accepted candidates:
    - iv_close
    - vix_close
    - india_vix_close
    - close
    - Close
    - CLOSE
    - adj_close
    - Adj Close
    - Adj_Close

    Parameters
    ----------
    vix_df:
        Input VIX panel.

    Returns
    -------
    str
        Inferred close column name.

    Raises
    ------
    ValueError
        If no usable close column can be found.
    """
    if not isinstance(vix_df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(vix_df)}")
    
    for col in IV_CLOSE_CANDIDATES:
        if col in vix_df.columns:
            return col
    
    numeric_candidates: list[str] = []
    
    for col in vix_df.columns:
        if col == "date":
            continue
        
        numeric = pd.to_numeric(vix_df[col], errors="coerce")
        not_missing = int(numeric.notna().sum())
        
        if not_missing > 0:
            numeric_candidates.append(col)
        
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
        
    raise ValueError(
    "Could not infer VIX close column. "
    f"Expected one of {IV_CLOSE_CANDIDATES}, or exactly one numeric non-date column. "
    f"Found columns: {list(vix_df.columns)}"
    )

def validate_vix_values(
    df: pd.DataFrame,
    *,
    iv_col: str = "iv_close",
    min_value: float = 0.0,
    max_value: float = 200.0,
) -> None:
    """
    Validate VIX / India VIX close values.

    Rules:
    - Values must be numeric.
    - Values must be finite.
    - Values must be strictly greater than min_value.
    - Values must be below max_value.
    - If all non-missing values are between 0 and 1, raise an error because
      the data is probably already divided by 100.

    Parameters
    ----------
    df:
        DataFrame containing implied-volatility index close values.
    iv_col:
        Column containing VIX / India VIX close levels.
    min_value:
        Lower bound. Default 0.
    max_value:
        Upper bound. Default 200.

    Raises
    ------
    ValueError
        If validation fails.
    """
    
    _require_columns(df, [iv_col])
    
    if max_value <= min_value:
        raise ValueError(f"max_value must be greater than min_value. Got min_value={min_value}, max_value={max_value}")
    
    raw = df[iv_col]
    values = pd.to_numeric(raw, errors="coerce")
    
    bad_parse_mask = values.isna() & raw.notna()
    if bad_parse_mask.any():
        bad_count = int(bad_parse_mask.sum())
        bad_indices = values[bad_parse_mask].index.tolist()
        raise ValueError(
            f"Failed to parse column '{iv_col}' as numeric. "
            f"Found {bad_count} non-numeric values at indices: {bad_indices}"
        )
    
    missing_mask = values.isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        missing_indices = values[missing_mask].index.tolist()
        raise ValueError(
            f"Column '{iv_col}' contains {missing_count} missing value(s) at indices: {missing_indices}"
        )
    
    non_missing_values = values.dropna()

    non_finite_mask = ~np.isfinite(non_missing_values)
    if non_finite_mask.any():
        non_finite_count = int(non_finite_mask.sum())
        non_finite_indices = non_missing_values[non_finite_mask].index.tolist()
        raise ValueError(
            f"Column '{iv_col}' contains {non_finite_count} non-finite values at indices: {non_finite_indices}"
        )
    
    to_high_mask = values >= max_value
    if to_high_mask.any():
        to_high_count = int(to_high_mask.sum())
        to_high_indices = values[to_high_mask].index.tolist()
        raise ValueError(
            f"Column '{iv_col}' contains {to_high_count} values >= {max_value} at indices: {to_high_indices}"
        )
    
    to_low_mask = values <= min_value
    if to_low_mask.any():
        to_low_count = int(to_low_mask.sum())
        to_low_indices = values[to_low_mask].index.tolist()
        raise ValueError(
            f"Column '{iv_col}' contains {to_low_count} values <= {min_value} at indices: {to_low_indices}"
        )
    
    decimal_scale_mask = (non_missing_values > 0) & (non_missing_values < 1)
    if decimal_scale_mask.any():
        decimal_count = int(decimal_scale_mask.sum())
        decimal_indices = non_missing_values[decimal_scale_mask].index.tolist()
        raise ValueError(
            f"Column '{iv_col}' contains {decimal_count} values between 0 and 1 at indices: {decimal_indices}. "
            "This suggests the data may already be divided by 100. Please check the source data."
        )
    
def build_implied_variance(
    vix_df: pd.DataFrame,
    market: str,
    *,
    iv_symbol: str | None = None,
    iv_close_col: str | None = None,
    max_vix_value: float = 200.0,
) -> pd.DataFrame:
    """
    Build an implied-variance panel from VIX / India VIX close levels.

    Formula:
        iv_ann_t = (iv_close_t / 100)^2

    Parameters
    ----------
    vix_df:
        Input VIX / India VIX panel. Must contain "date" and a close column.
    market:
        Market label: "US" or "INDIA".
    iv_symbol:
        Optional explicit IV symbol. Defaults:
            US -> VIX
            INDIA -> INDIA_VIX
    iv_close_col:
        Optional explicit input close column. If None, inferred.
    max_vix_value:
        Maximum accepted VIX-style close value. Default 200.

    Returns
    -------
    pd.DataFrame
        Columns:
            date
            market
            iv_symbol
            iv_close
            iv_ann
    """
    if not isinstance(vix_df, pd.DataFrame):
        raise TypeError("vix_df must be a pandas DataFrame.")
    
    market_norm = _normalize_market(market)
    
    if iv_symbol is None:
        iv_symbol = DEFAULT_IV_SYMBOLS[market_norm]
    
    _require_columns(vix_df, ["date"])
    
    out = vix_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    
    bad_date_mask = out["date"].isna()
    if bad_date_mask.any():
        bad_count = int(bad_date_mask.sum())
        bad_indices = out[bad_date_mask].index.tolist()
        raise ValueError(
            f"Failed to parse 'date' column as datetime. "
            f"Found {bad_count} non-parsable values at indices: {bad_indices}"
        )
    
    if iv_close_col is None:
        iv_close_col = infer_iv_close_column(out)
    
    _require_columns(out, [iv_close_col])
    
    result = pd.DataFrame({
        "date": out["date"],
        "market": market_norm,
        "iv_symbol": str(iv_symbol),
        "iv_close": pd.to_numeric(out[iv_close_col], errors="coerce"),
    })
    
    result = result.sort_values("date").reset_index(drop=True)
    
    duplicate_dates = result["date"].duplicated(keep=False)
    
    if duplicate_dates.any():
        bad_count = int(duplicate_dates.sum())
        first_bad_dates = (
            result.loc[duplicate_dates, "date"]
            .drop_duplicates()
            .head(10)
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(
            f"Found {bad_count} duplicated date entries in the input data. "
            f"First few duplicate dates: {first_bad_dates}"
        )
    
    validate_vix_values(result, iv_col="iv_close", min_value=0.0, max_value=max_vix_value)
    
    result["iv_ann"] = (result["iv_close"] / 100) ** 2 ##Formula for annualised implied variance
    
    order = ["date", "market", "iv_symbol", "iv_close", "iv_ann"]
    return result[order].copy()