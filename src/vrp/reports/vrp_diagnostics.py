# src/vrp/reports/vrp_diagnostics.py

"""
Diagnostics and reporting utilities for Phase 3 VRP panels.

Outputs:
- reports/tables/vrp_summary.csv
- reports/tables/vrp_metadata.json
- reports/figures/us_iv_rv_vrp.png
- reports/figures/india_iv_rv_vrp.png

Scope:
- Descriptive summary of IV, lagged RV, backward VRP, and forward labels.
- Positivity diagnostics.
- IV vs RV and VRP plots.

Rules:
- Do not mutate input panels.
- Do not create signals.
- Do not compute HAR.
- Do not compute regimes.
- Do not forward-fill missing values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.features.feature_io import ensure_parent_dir
from vrp.features.feature_registry import make_vrp_feature_metadata


VRP_SUMMARY_COLUMNS = [
    "market",
    "column",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p05",
    "p95",
    "count",
    "missing",
    "positive_count",
    "positive_ratio",
]


DEFAULT_SUMMARY_TARGET_COLUMNS = [
    "iv_ann",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
    "rv_cc_22d_ann_lag1",
    "vrp_backward_cc",
    "rv_parkinson_22d_ann_lag1",
    "vrp_backward_parkinson",
    "rv_rs_22d_ann_lag1",
    "vrp_backward_rs",
    "rv_yz_22d_ann_lag1",
    "vrp_backward_yz",
]


def _get_market(df: pd.DataFrame, fallback: str) -> str:
    """
    Extract market label from panel if possible.
    """
    if "market" in df.columns and len(df) > 0:
        non_missing = df["market"].dropna()
        if len(non_missing) > 0:
            return str(non_missing.iloc[0]).upper()

    return str(fallback).upper()


def _coerce_numeric_for_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Coerce a column to numeric for summary reporting.

    Non-numeric values become NaN here because this is a reporting module.
    Validation should happen earlier in feature construction.
    """
    return pd.to_numeric(df[column], errors="coerce")


def make_vrp_summary(
    panels: dict[str, pd.DataFrame],
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build VRP descriptive summary table.

    Parameters
    ----------
    panels:
        Mapping from market name to VRP panel.
        Example: {"US": us_vrp, "INDIA": india_vrp}
    columns:
        Numeric columns to summarize. Defaults to primary Phase 3 columns.

    Returns
    -------
    pd.DataFrame
        Summary table with:
            market
            column
            mean
            median
            std
            min
            max
            p05
            p95
            count
            missing
            positive_count
            positive_ratio
    """
    if not panels:
        raise ValueError("No VRP panels supplied.")

    if columns is None:
        columns = DEFAULT_SUMMARY_TARGET_COLUMNS

    rows: list[dict[str, object]] = []

    for market_key, panel in panels.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"Panel for market '{market_key}' must be a DataFrame.")

        market = _get_market(panel, fallback=market_key)

        for col in columns:
            if col not in panel.columns:
                continue

            values = _coerce_numeric_for_summary(panel, col)

            count = int(values.count())
            missing = int(values.isna().sum())
            positive_count = int((values.dropna() > 0).sum())
            positive_ratio = positive_count / count if count > 0 else float("nan")

            rows.append(
                {
                    "market": market,
                    "column": col,
                    "mean": values.mean(skipna=True),
                    "median": values.median(skipna=True),
                    "std": values.std(skipna=True),
                    "min": values.min(skipna=True),
                    "max": values.max(skipna=True),
                    "p05": values.quantile(0.05),
                    "p95": values.quantile(0.95),
                    "count": count,
                    "missing": missing,
                    "positive_count": positive_count,
                    "positive_ratio": positive_ratio,
                }
            )

    return pd.DataFrame(rows, columns=VRP_SUMMARY_COLUMNS)


def write_vrp_metadata(
    output_path: str | Path,
    *,
    horizon: int = 22,
    annualization_periods: int = 252,
    iv_horizon_calendar_days: int = 30,
    rv_horizon_trading_days: int = 22,
) -> Path:
    """
    Write Phase 3 VRP metadata JSON.

    Parameters
    ----------
    output_path:
        Metadata output path.
    horizon:
        Forward ex-post label horizon in trading observations.
    annualization_periods:
        Annualisation factor for daily realised variance.
    iv_horizon_calendar_days:
        IV index horizon in calendar days.
    rv_horizon_trading_days:
        RV approximation horizon in trading days.

    Returns
    -------
    Path
        Written metadata path.
    """
    out_path = ensure_parent_dir(output_path)

    metadata = {
        "phase": "Phase 3 - implied variance and VRP construction",
        "primary_estimator": "garman_klass",
        "robustness_estimators": [
            "close_to_close",
            "parkinson",
            "rogers_satchell",
            "yang_zhang",
        ],
        "robustness_usage": (
            "Backward VRP robustness diagnostics only; not live trading features in Phase 3."
        ),
        "inputs": [
            "data/processed/us_vix.parquet",
            "data/processed/india_vix.parquet",
            "data/processed/us_rv.parquet",
            "data/processed/india_rv.parquet",
        ],
        "outputs": [
            "data/processed/us_iv.parquet",
            "data/processed/india_iv.parquet",
            "data/processed/us_vrp.parquet",
            "data/processed/india_vrp.parquet",
        ],
        "primary_iv_column": "iv_ann",
        "primary_rv_column": "rv_gk_22d_ann",
        "primary_backward_vrp_column": "vrp_backward_gk",
        "forward_expost_rv_label_column": "rv_gk_22d_forward_ann_label",
        "forward_expost_vrp_label_column": "vrp_forward_expost_gk_label",
        "iv_formula": "iv_ann_t = (iv_close_t / 100)^2",
        "backward_vrp_formula": "vrp_backward_gk_t = iv_ann_t - rv_gk_22d_ann_lag1_t",
        "forward_rv_label_formula": (
            "rv_gk_22d_forward_ann_label_t = "
            "annualization_periods * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+horizon})"
        ),
        "forward_expost_vrp_formula": (
            "vrp_forward_expost_gk_label_t = "
            "iv_ann_t - rv_gk_22d_forward_ann_label_t"
        ),
        "backward_vrp_timing": (
            "iv_ann_t minus rv_gk_22d_ann_lag1; conservative point-in-time convention"
        ),
        "forward_expost_usage": (
            "evaluation label only; never allowed as live feature or trading signal"
        ),
        "horizon": horizon,
        "annualization_periods": annualization_periods,
        "iv_horizon_calendar_days": iv_horizon_calendar_days,
        "rv_horizon_trading_days": rv_horizon_trading_days,
        "horizon_alignment_note": (
            "22 trading days approximates the 30-calendar-day VIX / India VIX horizon."
        ),
        "no_forward_fill_policy": (
            "IV and RV panels are aligned by inner join on exact dates. "
            "No IV or RV forward-fill is used."
        ),
        "feature_registry": make_vrp_feature_metadata(),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_path


def plot_iv_rv_vrp(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    market: str,
) -> Path:
    """
    Plot IV, lagged RV, backward VRP, and forward ex-post VRP label.

    Parameters
    ----------
    df:
        VRP panel.
    output_path:
        Figure output path.
    market:
        Market label.

    Returns
    -------
    Path
        Written figure path.
    """
    required_columns = [
        "date",
        "iv_ann",
        "rv_gk_22d_ann_lag1",
        "vrp_backward_gk",
        "vrp_forward_expost_gk_label",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{market} VRP panel missing required plot column(s): {missing}"
        )

    plot_df = df[required_columns].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")

    bad_dates = plot_df["date"].isna()
    if bad_dates.any():
        bad_count = int(bad_dates.sum())
        raise ValueError(
            f"{market} VRP panel has {bad_count} invalid date value(s)."
        )

    for col in required_columns:
        if col == "date":
            continue
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    out_path = ensure_parent_dir(output_path)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(13, 10),
        sharex=True,
    )

    axes[0].plot(plot_df["date"], plot_df["iv_ann"], label="iv_ann")
    axes[0].plot(
        plot_df["date"],
        plot_df["rv_gk_22d_ann_lag1"],
        label="rv_gk_22d_ann_lag1",
    )
    axes[0].set_title(f"{str(market).upper()} IV vs lagged realised variance")
    axes[0].set_ylabel("Annualised variance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_df["date"],
        plot_df["vrp_backward_gk"],
        label="vrp_backward_gk",
    )
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title(f"{str(market).upper()} backward VRP")
    axes[1].set_ylabel("Variance spread")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        plot_df["date"],
        plot_df["vrp_forward_expost_gk_label"],
        label="vrp_forward_expost_gk_label",
    )
    axes[2].axhline(0.0, linewidth=1)
    axes[2].set_title(f"{str(market).upper()} forward ex-post VRP label")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Variance spread")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_vrp_diagnostics(
    panels: dict[str, pd.DataFrame],
    *,
    table_dir: str | Path = "reports/tables",
    figure_dir: str | Path = "reports/figures",
    horizon: int = 22,
    annualization_periods: int = 252,
) -> dict[str, Path]:
    """
    Write all Phase 3 VRP diagnostics.

    Outputs:
    - reports/tables/vrp_summary.csv
    - reports/tables/vrp_metadata.json
    - reports/figures/us_iv_rv_vrp.png
    - reports/figures/india_iv_rv_vrp.png

    Parameters
    ----------
    panels:
        Mapping from market name to VRP panel.
    table_dir:
        Directory for CSV/JSON outputs.
    figure_dir:
        Directory for figures.
    horizon:
        Forward label horizon.
    annualization_periods:
        Annualisation factor.

    Returns
    -------
    dict[str, Path]
        Created output paths.
    """
    if not panels:
        raise ValueError("No VRP panels supplied for diagnostics.")

    table_path = Path(table_dir)
    figure_path = Path(figure_dir)

    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    created: dict[str, Path] = {}

    summary = make_vrp_summary(panels)
    summary_path = table_path / "vrp_summary.csv"
    summary.to_csv(summary_path, index=False)
    created["summary"] = summary_path

    metadata_path = write_vrp_metadata(
        table_path / "vrp_metadata.json",
        horizon=horizon,
        annualization_periods=annualization_periods,
    )
    created["metadata"] = metadata_path

    for market, panel in panels.items():
        market_lower = str(market).lower()
        figure_output = figure_path / f"{market_lower}_iv_rv_vrp.png"

        fig_path = plot_iv_rv_vrp(
            panel,
            figure_output,
            market=str(market).upper(),
        )

        created[f"figure_{str(market).upper()}"] = fig_path

    return created