# src/vrp/features/calendars.py

"""
Calendar alignment utilities for Phase 3.

Scope:
- Align IV and RV panels on market-specific dates.
- Report IV/RV calendar mismatches.
- Write calendar mismatch diagnostics.

Rules:
- Use inner join for VRP construction.
- Do not forward-fill IV.
- Do not forward-fill RV.
- Do not merge US and India calendars into a combined trading calendar yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from vrp.features.feature_io import ensure_parent_dir


CALENDAR_MISMATCH_COLUMNS = [
    "market",
    "iv_start",
    "iv_end",
    "rv_start",
    "rv_end",
    "iv_rows",
    "rv_rows",
    "common_dates",
    "iv_only_dates",
    "rv_only_dates",
    "first_iv_only_date",
    "first_rv_only_date",
]

def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")
    
def _format_date(value: pd.Timestamp | None) -> str | None:
    """
    Convert Timestamp to YYYY-MM-DD string.
    """
    if value is None or pd.isna(value):
        return None

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _prepare_date_panel(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    Validate and normalize a panel with a date column.

    Rules:
    - Must contain date.
    - date must parse cleanly.
    - duplicate dates are not allowed.
    - output is sorted by date and index reset.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    _require_columns(df, ["date"], name=name)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    bad_date_mask = out["date"].isna()
    if bad_date_mask.any():
        bad_count = int(bad_date_mask.sum())
        bad_indices = out[bad_date_mask].index.tolist()[:10]
        raise ValueError(
            f"{name} contains {bad_count} invalid date value(s). "
            f"First bad indices: {bad_indices}"
        )

    duplicate_mask = out["date"].duplicated(keep=False)
    if duplicate_mask.any():
        bad_count = int(duplicate_mask.sum())
        bad_dates = (
            out.loc[duplicate_mask, "date"]
            .dt.strftime("%Y-%m-%d")
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{name} contains {bad_count} duplicated date row(s). "
            f"First duplicated dates: {bad_dates}"
        )

    out = out.sort_values("date").reset_index(drop=True)

    return out

def _date_set(df: pd.DataFrame, *, name: str) -> set[pd.Timestamp]:
    """
    Return set of normalized dates from a panel.
    """
    prepared = _prepare_date_panel(df, name=name)
    return set(pd.Timestamp(x) for x in prepared["date"])

def _first_date_or_none(dates: set[pd.Timestamp]) -> pd.Timestamp | None:
    """
    Return earliest date from a set, or None.
    """
    if not dates:
        return None

    return min(dates)

def align_market_dates(
    iv_df: pd.DataFrame,
    rv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align IV and RV panels on common market dates using inner join.

    Parameters
    ----------
    iv_df:
        Implied variance panel. Expected columns include:
            date, market, iv_symbol, iv_close, iv_ann
    rv_df:
        Realised variance panel. Expected columns include:
            date, market, symbol, rv_gk_daily, rv_gk_22d_ann

    Returns
    -------
    pd.DataFrame
        Inner-joined panel on date.

    Notes
    -----
    - No forward-fill.
    - No backward-fill.
    - No outer join.
    - If both IV and RV panels contain market columns, they must match on all
      aligned rows.
    """
    iv_prepared = _prepare_date_panel(iv_df, name="IV panel")
    rv_prepared = _prepare_date_panel(rv_df, name="RV panel")

    aligned = iv_prepared.merge(
        rv_prepared,
        on="date",
        how="inner",
        suffixes=("_iv", "_rv"),
        sort=True,
    )

    if "market_iv" in aligned.columns and "market_rv" in aligned.columns:
        market_iv = aligned["market_iv"].astype(str).str.upper()
        market_rv = aligned["market_rv"].astype(str).str.upper()

        mismatch_mask = market_iv != market_rv
        if mismatch_mask.any():
            bad_count = int(mismatch_mask.sum())
            bad_dates = (
                aligned.loc[mismatch_mask, "date"]
                .dt.strftime("%Y-%m-%d")
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"IV/RV market mismatch on {bad_count} aligned row(s). "
                f"First bad dates: {bad_dates}"
            )

        aligned["market"] = market_iv
        aligned = aligned.drop(columns=["market_iv", "market_rv"])

    elif "market_iv" in aligned.columns:
        aligned["market"] = aligned["market_iv"].astype(str).str.upper()
        aligned = aligned.drop(columns=["market_iv"])

    elif "market_rv" in aligned.columns:
        aligned["market"] = aligned["market_rv"].astype(str).str.upper()
        aligned = aligned.drop(columns=["market_rv"])

    # Keep date first, market second when available.
    leading_cols = ["date"]
    if "market" in aligned.columns:
        leading_cols.append("market")

    remaining_cols = [col for col in aligned.columns if col not in leading_cols]
    aligned = aligned[leading_cols + remaining_cols].copy()

    return aligned.sort_values("date").reset_index(drop=True)

def report_calendar_mismatches(
    iv_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    market: str,
) -> dict[str, object]:
    """
    Report date overlap and mismatches between IV and RV panels.

    Parameters
    ----------
    iv_df:
        Implied variance panel.
    rv_df:
        Realised variance panel.
    market:
        Market label.

    Returns
    -------
    dict[str, object]
        Calendar mismatch summary with stable output columns:
            market
            iv_start
            iv_end
            rv_start
            rv_end
            iv_rows
            rv_rows
            common_dates
            iv_only_dates
            rv_only_dates
            first_iv_only_date
            first_rv_only_date
    """
    iv = _prepare_date_panel(iv_df, name=f"{market} iv_df")
    rv = _prepare_date_panel(rv_df, name=f"{market} rv_df")

    iv_dates = set(pd.Timestamp(x) for x in iv["date"])
    rv_dates = set(pd.Timestamp(x) for x in rv["date"])

    common_dates = iv_dates & rv_dates
    iv_only_dates = iv_dates - rv_dates
    rv_only_dates = rv_dates - iv_dates

    row = {
        "market": str(market).upper(),
        "iv_start": _format_date(iv["date"].min()) if len(iv) > 0 else None,
        "iv_end": _format_date(iv["date"].max()) if len(iv) > 0 else None,
        "rv_start": _format_date(rv["date"].min()) if len(rv) > 0 else None,
        "rv_end": _format_date(rv["date"].max()) if len(rv) > 0 else None,
        "iv_rows": int(len(iv)),
        "rv_rows": int(len(rv)),
        "common_dates": int(len(common_dates)),
        "iv_only_dates": int(len(iv_only_dates)),
        "rv_only_dates": int(len(rv_only_dates)),
        "first_iv_only_date": _format_date(_first_date_or_none(iv_only_dates)),
        "first_rv_only_date": _format_date(_first_date_or_none(rv_only_dates)),
    }

    return row

def build_calendar_mismatch_table(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Build a stable calendar mismatch DataFrame from row dictionaries.
    """
    return pd.DataFrame(rows, columns=CALENDAR_MISMATCH_COLUMNS)


def write_calendar_mismatch_report(
    rows: list[dict[str, object]],
    output_path: str | Path = "reports/tables/calendar_mismatches.csv",
) -> Path:
    """
    Write calendar mismatch report to CSV.

    Parameters
    ----------
    rows:
        List of calendar mismatch row dictionaries.
    output_path:
        Output CSV path.

    Returns
    -------
    Path
        Written CSV path.
    """
    out_path = ensure_parent_dir(output_path)

    table = build_calendar_mismatch_table(rows)
    table.to_csv(out_path, index=False)

    return out_path