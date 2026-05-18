# src/vrp/forecasting/forecast_evaluation.py

"""
Forecast evaluation utilities for Phase 4 HAR-RV forecasts.

Scope:
- Evaluate HAR-RV forecasts against realised forward variance labels.
- Compare HAR against naive and timing-safe historical-mean baselines.
- Use metrics appropriate for variance forecasts.

Important:
- Forecasts and targets are variances.
- QLIKE clips only local metric arrays for numerical stability.
- Original input data is never overwritten.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "market",
    "forecast_col",
    "target_col",
    "n_obs",
    "mse",
    "rmse",
    "mae",
    "qlike",
    "bias",
    "correlation",
    "directional_accuracy_vs_baseline",
]


DEFAULT_FORECAST_COLUMNS = [
    "har_rv_gk_22d_forecast_ann",
    "naive_lagged_22d_rv_ann",
    "expanding_mean_forward_rv_baseline",
    "rolling_mean_forward_rv_baseline",
]


def _as_numeric_series(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """
    Convert input values to a numeric pandas Series.
    """
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")

    return pd.to_numeric(pd.Series(values), errors="coerce")


def _paired_clean_arrays(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return aligned finite arrays after dropping NaN and non-finite pairs.
    """
    true = _as_numeric_series(y_true)
    pred = _as_numeric_series(y_pred)

    if len(true) != len(pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(true)} and {len(pred)}."
        )

    data = pd.DataFrame(
        {
            "y_true": true,
            "y_pred": pred,
        }
    )

    mask = (
        data["y_true"].notna()
        & data["y_pred"].notna()
        & np.isfinite(data["y_true"])
        & np.isfinite(data["y_pred"])
    )

    clean = data.loc[mask]

    return (
        clean["y_true"].to_numpy(dtype=float),
        clean["y_pred"].to_numpy(dtype=float),
    )


def _triple_clean_arrays(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
    baseline_pred: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return aligned finite arrays for target, forecast, and baseline.
    """
    true = _as_numeric_series(y_true)
    pred = _as_numeric_series(y_pred)
    base = _as_numeric_series(baseline_pred)

    if not (len(true) == len(pred) == len(base)):
        raise ValueError(
            "y_true, y_pred, and baseline_pred must have same length. "
            f"Got {len(true)}, {len(pred)}, and {len(base)}."
        )

    data = pd.DataFrame(
        {
            "y_true": true,
            "y_pred": pred,
            "baseline_pred": base,
        }
    )

    mask = (
        data["y_true"].notna()
        & data["y_pred"].notna()
        & data["baseline_pred"].notna()
        & np.isfinite(data["y_true"])
        & np.isfinite(data["y_pred"])
        & np.isfinite(data["baseline_pred"])
    )

    clean = data.loc[mask]

    return (
        clean["y_true"].to_numpy(dtype=float),
        clean["y_pred"].to_numpy(dtype=float),
        clean["baseline_pred"].to_numpy(dtype=float),
    )


def mse(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Mean squared error.
    """
    true, pred = _paired_clean_arrays(y_true, y_pred)

    if len(true) == 0:
        return float("nan")

    return float(np.mean((true - pred) ** 2))


def rmse(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Root mean squared error.
    """
    value = mse(y_true, y_pred)

    if np.isnan(value):
        return float("nan")

    return float(np.sqrt(value))


def mae(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Mean absolute error.
    """
    true, pred = _paired_clean_arrays(y_true, y_pred)

    if len(true) == 0:
        return float("nan")

    return float(np.mean(np.abs(true - pred)))


def qlike(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
    eps: float = 1e-8,
) -> float:
    """
    QLIKE loss for variance forecasts.

    Formula used:
        QLIKE = mean(y_true / y_pred - log(y_true / y_pred) - 1)

    This form is non-negative and equals zero when y_true == y_pred.

    For numerical stability, y_true and y_pred are clipped by eps inside this
    function only. The original data is not modified.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive. Got: {eps}")

    true, pred = _paired_clean_arrays(y_true, y_pred)

    if len(true) == 0:
        return float("nan")

    true_clip = np.clip(true, eps, None)
    pred_clip = np.clip(pred, eps, None)

    ratio = true_clip / pred_clip
    loss = ratio - np.log(ratio) - 1.0

    return float(np.mean(loss))


def forecast_bias(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Mean forecast error.

    Positive value means forecast is higher than realised target on average.
    """
    true, pred = _paired_clean_arrays(y_true, y_pred)

    if len(true) == 0:
        return float("nan")

    return float(np.mean(pred - true))


def forecast_correlation(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Pearson correlation between forecast and target.
    """
    true, pred = _paired_clean_arrays(y_true, y_pred)

    if len(true) < 2:
        return float("nan")

    if np.std(true) == 0 or np.std(pred) == 0:
        return float("nan")

    return float(np.corrcoef(true, pred)[0, 1])


def directional_accuracy_against_baseline(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
    baseline_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Directional accuracy of forecast changes versus baseline changes.

    This answers:
        When the forecast says the target should be above/below the baseline,
        how often is the realised target actually above/below that baseline?

    Ties are excluded.
    """
    true, pred, base = _triple_clean_arrays(y_true, y_pred, baseline_pred)

    if len(true) == 0:
        return float("nan")

    forecast_direction = np.sign(pred - base)
    realized_direction = np.sign(true - base)

    non_tie_mask = (forecast_direction != 0) & (realized_direction != 0)

    if not np.any(non_tie_mask):
        return float("nan")

    correct = forecast_direction[non_tie_mask] == realized_direction[non_tie_mask]

    return float(np.mean(correct))


def _count_clean_pairs(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> int:
    """
    Count usable target/forecast pairs.
    """
    true, pred = _paired_clean_arrays(y_true, y_pred)
    return int(len(true))


def _get_market(df: pd.DataFrame, fallback: str | None = None) -> str:
    """
    Extract market label from a forecast panel.
    """
    if "market" in df.columns and len(df) > 0:
        values = df["market"].dropna().astype(str).str.upper().unique().tolist()
        if len(values) == 1:
            return values[0]

    if fallback is not None:
        return str(fallback).upper()

    return "UNKNOWN"


def evaluate_forecasts(
    df: pd.DataFrame,
    target_col: str,
    forecast_cols: Iterable[str],
    *,
    market: str | None = None,
    baseline_col: str = "naive_lagged_22d_rv_ann",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Evaluate one panel across one or more forecast columns.

    Parameters
    ----------
    df:
        Forecast panel.
    target_col:
        Target column. Primary Phase 4 target:
            rv_gk_22d_forward_ann_label
    forecast_cols:
        Forecast columns to evaluate.
    market:
        Optional market override.
    baseline_col:
        Baseline column used for directional accuracy.
    eps:
        Numerical floor for QLIKE calculation.

    Returns
    -------
    pd.DataFrame
        One row per forecast column.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    forecast_cols = list(forecast_cols)
    missing_forecasts = [col for col in forecast_cols if col not in df.columns]

    if missing_forecasts:
        raise ValueError(f"Missing forecast column(s): {missing_forecasts}")

    market_value = _get_market(df, fallback=market)

    rows: list[dict[str, object]] = []

    for forecast_col in forecast_cols:
        y_true = df[target_col]
        y_pred = df[forecast_col]

        if baseline_col in df.columns and forecast_col != baseline_col:
            directional_accuracy = directional_accuracy_against_baseline(
                y_true,
                y_pred,
                df[baseline_col],
            )
        else:
            directional_accuracy = float("nan")

        rows.append(
            {
                "market": market_value,
                "forecast_col": forecast_col,
                "target_col": target_col,
                "n_obs": _count_clean_pairs(y_true, y_pred),
                "mse": mse(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "qlike": qlike(y_true, y_pred, eps=eps),
                "bias": forecast_bias(y_true, y_pred),
                "correlation": forecast_correlation(y_true, y_pred),
                "directional_accuracy_vs_baseline": directional_accuracy,
            }
        )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_forecast_accuracy_table(
    us_df: pd.DataFrame,
    india_df: pd.DataFrame,
    *,
    target_col: str = "rv_gk_22d_forward_ann_label",
    forecast_cols: list[str] | None = None,
    baseline_col: str = "naive_lagged_22d_rv_ann",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Build combined US/India forecast accuracy table.
    """
    if forecast_cols is None:
        forecast_cols = [
            col for col in DEFAULT_FORECAST_COLUMNS
            if col in us_df.columns or col in india_df.columns
        ]

    us_available = [col for col in forecast_cols if col in us_df.columns]
    india_available = [col for col in forecast_cols if col in india_df.columns]

    tables: list[pd.DataFrame] = []

    if us_available:
        tables.append(
            evaluate_forecasts(
                us_df,
                target_col=target_col,
                forecast_cols=us_available,
                market="US",
                baseline_col=baseline_col,
                eps=eps,
            )
        )

    if india_available:
        tables.append(
            evaluate_forecasts(
                india_df,
                target_col=target_col,
                forecast_cols=india_available,
                market="INDIA",
                baseline_col=baseline_col,
                eps=eps,
            )
        )

    if not tables:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    return pd.concat(tables, ignore_index=True)


def rank_forecasts_by_metric(
    accuracy_table: pd.DataFrame,
    *,
    metric: str = "qlike",
) -> pd.DataFrame:
    """
    Rank forecasts within each market by a loss metric.

    Lower is better for:
        mse, rmse, mae, qlike

    This function does not claim superiority. It only ranks observed metrics.
    """
    if metric not in accuracy_table.columns:
        raise ValueError(f"Metric column not found: {metric}")

    if "market" not in accuracy_table.columns:
        raise ValueError("accuracy_table must contain 'market' column.")

    out = accuracy_table.copy()
    out[f"{metric}_rank"] = out.groupby("market")[metric].rank(
        method="dense",
        ascending=True,
    )

    return out