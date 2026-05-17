# src/vrp/features/vrp.py

"""
Variance Risk Premium construction for Phase 3.

Scope:
- Merge implied variance and realised variance on market-specific dates.
- Compute backward VRP using lagged realised variance.
- Compute forward ex-post VRP as a non-tradable evaluation label.
- Mark feature-allowed rows.
- Enforce feature/label separation.

Definitions:
    iv_ann_t = (iv_close_t / 100)^2

    vrp_backward_gk_t =
        iv_ann_t - rv_gk_22d_ann_{t-1}

    rv_gk_22d_forward_ann_label_t =
        252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})

    vrp_forward_expost_gk_label_t =
        iv_ann_t - rv_gk_22d_forward_ann_label_t
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from vrp.features.calendars import align_market_dates
from vrp.features.feature_registry import (
    VRP_FEATURE_COLUMNS,
    VRP_LABEL_COLUMNS,
    assert_registry_is_valid,
    assert_no_lookahead_feature_columns,
)


PRIMARY_RV_COL = "rv_gk_22d_ann"
PRIMARY_RV_DAILY_COL = "rv_gk_daily"
PRIMARY_RV_LAG_COL = "rv_gk_22d_ann_lag1"
PRIMARY_BACKWARD_VRP_COL = "vrp_backward_gk"
PRIMARY_BACKWARD_POSITIVE_COL = "vrp_backward_gk_positive"
PRIMARY_FORWARD_RV_LABEL_COL = "rv_gk_22d_forward_ann_label"
PRIMARY_FORWARD_VRP_LABEL_COL = "vrp_forward_expost_gk_label"

ROBUSTNESS_RV_COLS = {
    "cc": "rv_cc_22d_ann",
    "parkinson": "rv_parkinson_22d_ann",
    "rs": "rv_rs_22d_ann",
    "yz": "rv_yz_22d_ann",
}


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    name: str,
) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def _sort_by_date(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    Return date-sorted copy with clean date values.
    """
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

    return out.sort_values("date").reset_index(drop=True)


def _coerce_numeric_column(
    df: pd.DataFrame,
    column: str,
    *,
    name: str,
    allow_missing: bool = True,
) -> pd.Series:
    """
    Convert a column to numeric while detecting bad non-missing strings.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to coerce.
    name:
        Human-readable panel name for error messages.
    allow_missing:
        If False, missing values are rejected.
    """
    _require_columns(df, [column], name=name)

    raw = df[column]
    values = pd.to_numeric(raw, errors="coerce")

    bad_parse_mask = raw.notna() & values.isna()
    if bad_parse_mask.any():
        bad_count = int(bad_parse_mask.sum())
        bad_indices = values[bad_parse_mask].index.tolist()[:10]
        raise ValueError(
            f"{name}.{column} contains {bad_count} non-numeric non-missing "
            f"value(s). First bad indices: {bad_indices}"
        )

    if not allow_missing:
        missing_mask = values.isna()
        if missing_mask.any():
            bad_count = int(missing_mask.sum())
            bad_indices = values[missing_mask].index.tolist()[:10]
            raise ValueError(
                f"{name}.{column} contains {bad_count} missing value(s). "
                f"First bad indices: {bad_indices}"
            )

    return values


def _validate_positive_finite(
    values: pd.Series,
    *,
    column: str,
    name: str,
    allow_missing: bool = True,
) -> None:
    """
    Validate numeric column is positive and finite.
    """
    check = values.copy()

    if allow_missing:
        check = check.dropna()

    missing_mask = values.isna()
    if not allow_missing and missing_mask.any():
        bad_count = int(missing_mask.sum())
        raise ValueError(f"{name}.{column} contains {bad_count} missing value(s).")

    non_finite_mask = ~np.isfinite(check)
    if non_finite_mask.any():
        bad_count = int(non_finite_mask.sum())
        raise ValueError(f"{name}.{column} contains {bad_count} non-finite value(s).")

    non_positive_mask = check <= 0
    if non_positive_mask.any():
        bad_count = int(non_positive_mask.sum())
        raise ValueError(f"{name}.{column} contains {bad_count} non-positive value(s).")


def merge_iv_rv(
    iv_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    """
    Merge implied variance and realised variance panels by common dates.

    Parameters
    ----------
    iv_df:
        IV panel with:
            date, market, iv_symbol, iv_close, iv_ann
    rv_df:
        RV panel with:
            date, market, symbol, rv_gk_daily, rv_gk_22d_ann
    market:
        Market label.

    Returns
    -------
    pd.DataFrame
        Date-aligned panel. Uses inner join only.

    Notes
    -----
    - No forward-fill.
    - No backward-fill.
    - No outer join.
    """
    market_norm = str(market).upper().strip()

    iv = _sort_by_date(iv_df, name=f"{market_norm} IV panel")
    rv = _sort_by_date(rv_df, name=f"{market_norm} RV panel")

    _require_columns(
        iv,
        ["date", "market", "iv_symbol", "iv_close", "iv_ann"],
        name=f"{market_norm} IV panel",
    )
    _require_columns(
        rv,
        ["date", "market", PRIMARY_RV_DAILY_COL, PRIMARY_RV_COL],
        name=f"{market_norm} RV panel",
    )

    if "symbol" not in rv.columns and "underlying_symbol" not in rv.columns:
        raise ValueError(
            f"{market_norm} RV panel must contain either 'symbol' or "
            "'underlying_symbol'."
        )

    iv_market_values = iv["market"].astype(str).str.upper().unique().tolist()
    rv_market_values = rv["market"].astype(str).str.upper().unique().tolist()

    if iv_market_values != [market_norm]:
        raise ValueError(
            f"{market_norm} IV panel contains unexpected market values: "
            f"{iv_market_values}"
        )

    if rv_market_values != [market_norm]:
        raise ValueError(
            f"{market_norm} RV panel contains unexpected market values: "
            f"{rv_market_values}"
        )

    aligned = align_market_dates(iv, rv)

    if aligned.empty:
        raise ValueError(
            f"{market_norm} IV/RV merge has zero common dates. "
            "Check calendar coverage."
        )

    if "symbol" in aligned.columns and "underlying_symbol" not in aligned.columns:
        aligned = aligned.rename(columns={"symbol": "underlying_symbol"})

    _require_columns(
        aligned,
        [
            "date",
            "market",
            "underlying_symbol",
            "iv_symbol",
            "iv_close",
            "iv_ann",
            PRIMARY_RV_DAILY_COL,
            PRIMARY_RV_COL,
        ],
        name=f"{market_norm} merged IV/RV panel",
    )

    aligned["iv_close"] = _coerce_numeric_column(
        aligned,
        "iv_close",
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=False,
    )
    aligned["iv_ann"] = _coerce_numeric_column(
        aligned,
        "iv_ann",
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=False,
    )
    aligned[PRIMARY_RV_DAILY_COL] = _coerce_numeric_column(
        aligned,
        PRIMARY_RV_DAILY_COL,
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=True,
    )
    aligned[PRIMARY_RV_COL] = _coerce_numeric_column(
        aligned,
        PRIMARY_RV_COL,
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=True,
    )

    _validate_positive_finite(
        aligned["iv_close"],
        column="iv_close",
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=False,
    )
    _validate_positive_finite(
        aligned["iv_ann"],
        column="iv_ann",
        name=f"{market_norm} merged IV/RV panel",
        allow_missing=False,
    )

    aligned["market"] = market_norm

    return aligned.sort_values("date").reset_index(drop=True)


def compute_backward_vrp(
    panel: pd.DataFrame,
    rv_col: str = PRIMARY_RV_COL,
) -> pd.DataFrame:
    """
    Compute backward-looking, point-in-time VRP.

    Formula:
        rv_gk_22d_ann_lag1_t = rv_gk_22d_ann_{t-1}
        vrp_backward_gk_t = iv_ann_t - rv_gk_22d_ann_lag1_t

    Parameters
    ----------
    panel:
        Merged IV/RV panel.
    rv_col:
        Realised variance column to lag.

    Returns
    -------
    pd.DataFrame
        Panel with:
            rv_gk_22d_ann_lag1
            vrp_backward_gk
            vrp_backward_gk_positive
    """
    out = _sort_by_date(panel, name="merged IV/RV panel")

    _require_columns(out, ["iv_ann", rv_col], name="merged IV/RV panel")

    iv_ann = _coerce_numeric_column(
        out,
        "iv_ann",
        name="merged IV/RV panel",
        allow_missing=False,
    )
    rv = _coerce_numeric_column(
        out,
        rv_col,
        name="merged IV/RV panel",
        allow_missing=True,
    )

    negative_mask = rv.dropna() < 0
    if negative_mask.any():
        bad_count = int(negative_mask.sum())
        raise ValueError(
            f"{rv_col} contains {bad_count} negative variance value(s)."
        )

    lag_col = f"{rv_col}_lag1"

    out[lag_col] = rv.shift(1)
    out[PRIMARY_BACKWARD_VRP_COL] = iv_ann - out[lag_col]

    out[PRIMARY_BACKWARD_POSITIVE_COL] = (
        out[PRIMARY_BACKWARD_VRP_COL] > 0
    ).where(out[PRIMARY_BACKWARD_VRP_COL].notna(), None).astype("boolean")

    return out


def compute_backward_vrp_robustness(
    panel: pd.DataFrame,
    rv_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compute lagged backward VRP robustness columns for non-primary estimators.
    """
    out = _sort_by_date(panel, name="merged IV/RV panel")

    if rv_cols is None:
        rv_cols = ROBUSTNESS_RV_COLS

    _require_columns(out, ["iv_ann"], name="merged IV/RV panel")

    iv_ann = _coerce_numeric_column(
        out,
        "iv_ann",
        name="merged IV/RV panel",
        allow_missing=False,
    )

    for estimator_name, rv_col in rv_cols.items():
        if rv_col not in out.columns:
            continue

        rv = _coerce_numeric_column(
            out,
            rv_col,
            name="merged IV/RV panel",
            allow_missing=True,
        )

        negative_mask = rv.dropna() < 0
        if negative_mask.any():
            bad_count = int(negative_mask.sum())
            raise ValueError(
                f"{rv_col} contains {bad_count} negative variance value(s)."
            )

        lag_col = f"{rv_col}_lag1"
        vrp_col = f"vrp_backward_{estimator_name}"
        positive_col = f"{vrp_col}_positive"

        out[lag_col] = rv.shift(1)
        out[vrp_col] = iv_ann - out[lag_col]

        out[positive_col] = pd.Series(
            [pd.NA if pd.isna(value) else bool(value > 0) for value in out[vrp_col]],
            index=out.index,
            dtype="boolean",
        )

    return out


def _future_annualized_rv_from_daily(
    daily_rv: pd.Series,
    *,
    horizon: int,
    annualization_periods: int,
) -> pd.Series:
    """
    Compute future annualised realised variance label.

    At date t, use strictly future values:
        daily_rv_{t+1}, ..., daily_rv_{t+horizon}

    Formula:
        future_rv_ann_t =
            annualization_periods * mean(daily_rv_{t+1:t+horizon})

    This avoids same-day leakage.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1. Got: {horizon}")

    if annualization_periods <= 0:
        raise ValueError(
            f"annualization_periods must be positive. Got: {annualization_periods}"
        )

    future_columns = [
        daily_rv.shift(-step)
        for step in range(1, horizon + 1)
    ]

    future_matrix = pd.concat(future_columns, axis=1)
    valid_count = future_matrix.notna().sum(axis=1)

    future_mean = future_matrix.mean(axis=1)
    future_mean = future_mean.where(valid_count == horizon)

    future_ann = future_mean * annualization_periods
    future_ann.name = PRIMARY_FORWARD_RV_LABEL_COL

    return future_ann


def compute_forward_expost_vrp(
    panel: pd.DataFrame,
    rv_daily_col: str = PRIMARY_RV_DAILY_COL,
    *,
    horizon: int = 22,
    annualization_periods: int = 252,
) -> pd.DataFrame:
    """
    Compute forward ex-post realised variance and VRP labels.

    Label formula:
        rv_gk_22d_forward_ann_label_t =
            252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})

        vrp_forward_expost_gk_label_t =
            iv_ann_t - rv_gk_22d_forward_ann_label_t

    These columns are non-tradable labels only.

    Parameters
    ----------
    panel:
        Merged IV/RV panel.
    rv_daily_col:
        Daily realised variance column.
    horizon:
        Number of future trading observations. Default 22.
    annualization_periods:
        Annualisation factor. Default 252.

    Returns
    -------
    pd.DataFrame
        Panel with forward ex-post label columns.
    """
    out = _sort_by_date(panel, name="merged IV/RV panel")

    _require_columns(out, ["iv_ann", rv_daily_col], name="merged IV/RV panel")

    iv_ann = _coerce_numeric_column(
        out,
        "iv_ann",
        name="merged IV/RV panel",
        allow_missing=False,
    )
    daily_rv = _coerce_numeric_column(
        out,
        rv_daily_col,
        name="merged IV/RV panel",
        allow_missing=True,
    )

    negative_daily_mask = daily_rv.dropna() < 0
    if negative_daily_mask.any():
        bad_count = int(negative_daily_mask.sum())
        raise ValueError(
            f"{rv_daily_col} contains {bad_count} negative daily variance value(s)."
        )

    out[PRIMARY_FORWARD_RV_LABEL_COL] = _future_annualized_rv_from_daily(
        daily_rv,
        horizon=horizon,
        annualization_periods=annualization_periods,
    )

    out[PRIMARY_FORWARD_VRP_LABEL_COL] = (
        iv_ann - out[PRIMARY_FORWARD_RV_LABEL_COL]
    )

    return out


def flag_feature_columns_vs_label_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature_allowed flag and enforce feature/label separation.

    feature_allowed is True only when all registered feature columns are present
    and non-missing for the row.

    This function does not make labels tradable. It only marks whether the row is
    usable for later live-feature construction.
    """
    assert_registry_is_valid()
    assert_no_lookahead_feature_columns(VRP_FEATURE_COLUMNS)

    out = panel.copy()

    missing_features = [
        col for col in VRP_FEATURE_COLUMNS
        if col not in out.columns
    ]
    missing_labels = [
        col for col in VRP_LABEL_COLUMNS
        if col not in out.columns
    ]

    if missing_features:
        raise ValueError(f"Panel is missing registered feature column(s): {missing_features}")

    if missing_labels:
        raise ValueError(f"Panel is missing registered label column(s): {missing_labels}")

    out["feature_allowed"] = out[VRP_FEATURE_COLUMNS].notna().all(axis=1)

    return out


def build_vrp_panel(
    iv_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    market: str,
    *,
    horizon: int = 22,
    annualization_periods: int = 252,
    rv_col: str = PRIMARY_RV_COL,
    rv_daily_col: str = PRIMARY_RV_DAILY_COL,
) -> pd.DataFrame:
    """
    Build complete Phase 3 VRP panel for one market.

    Steps:
    1. Inner-join IV and RV on date.
    2. Compute backward VRP from lagged RV.
    3. Compute forward ex-post RV and VRP labels.
    4. Enforce feature/label separation.
    5. Return stable ordered columns.

    Returns
    -------
    pd.DataFrame
        VRP panel.
    """
    merged = merge_iv_rv(iv_df, rv_df, market=market)

    out = compute_backward_vrp(
        merged,
        rv_col=rv_col,
    )

    out = compute_backward_vrp_robustness(out)

    out = compute_forward_expost_vrp(
        out,
        rv_daily_col=rv_daily_col,
        horizon=horizon,
        annualization_periods=annualization_periods,
    )

    out = flag_feature_columns_vs_label_columns(out)

    base_columns = [
        "date",
        "market",
        "underlying_symbol",
        "iv_symbol",
        "iv_close",
        "iv_ann",
        rv_daily_col,
        rv_col,
        f"{rv_col}_lag1",
        PRIMARY_BACKWARD_VRP_COL,
        PRIMARY_BACKWARD_POSITIVE_COL,
        PRIMARY_FORWARD_RV_LABEL_COL,
        PRIMARY_FORWARD_VRP_LABEL_COL,
        "feature_allowed",
    ]

    optional_diagnostic_columns = [
        "log_return",
        "simple_return",
        "rv_cc_22d_ann",
        "rv_parkinson_22d_ann",
        "rv_rs_22d_ann",
        "rv_yz_22d_ann",
        "rv_cc_22d_ann_lag1",
        "vrp_backward_cc",
        "vrp_backward_cc_positive",
        "rv_parkinson_22d_ann_lag1",
        "vrp_backward_parkinson",
        "vrp_backward_parkinson_positive",
        "rv_rs_22d_ann_lag1",
        "vrp_backward_rs",
        "vrp_backward_rs_positive",
        "rv_yz_22d_ann_lag1",
        "vrp_backward_yz",
        "vrp_backward_yz_positive",
    ]

    ordered_columns = [
        col for col in base_columns + optional_diagnostic_columns
        if col in out.columns
    ]

    return out[ordered_columns].copy()