# src/vrp/reports/rv_diagnostics.py

"""
Diagnostics and reporting utilities for Phase 2 realised variance panels.

Outputs:
- reports/tables/rv_summary.csv
- reports/tables/rv_estimator_correlations.csv
- reports/tables/rv_metadata.json
- reports/figures/rv_estimators_us.png
- reports/figures/rv_estimators_india.png
"""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from vrp.features.feature_io import ensure_parent_dir


DEFAULT_INPUT_FILES = [
    "data/processed/us_underlying.parquet",
    "data/processed/india_underlying.parquet",
]


def rv_annualized_columns(window: int = 22) -> list[str]:
    """
    Return annualised realised-variance columns for a given window.
    """
    return [
        f"rv_cc_{window}d_ann",
        f"rv_parkinson_{window}d_ann",
        f"rv_gk_{window}d_ann",
        f"rv_rs_{window}d_ann",
        f"rv_yz_{window}d_ann",
    ]


def _get_symbol(df: pd.DataFrame, market: str) -> str:
    """
    Safely extract symbol from a panel.
    """
    if "symbol" in df.columns and len(df) > 0:
        first_symbol = df["symbol"].dropna()
        if len(first_symbol) > 0:
            return str(first_symbol.iloc[0])

    return str(market)


def make_rv_summary(
    panels: dict[str, pd.DataFrame],
    *,
    window: int = 22,
) -> pd.DataFrame:
    """
    Build realised-variance summary statistics.

    Parameters
    ----------
    panels:
        Mapping from market name to RV panel.
        Example: {"US": us_rv, "INDIA": india_rv}
    window:
        Rolling annualised RV window.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per market and RV column.
    """
    columns = rv_annualized_columns(window)
    rows: list[dict[str, object]] = []

    for market, panel in panels.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"Panel for market '{market}' must be a DataFrame.")

        symbol = _get_symbol(panel, market)

        for col in columns:
            if col not in panel.columns:
                continue

            values = pd.to_numeric(panel[col], errors="coerce")

            rows.append(
                {
                    "market": str(market).upper(),
                    "symbol": symbol,
                    "column": col,
                    "mean": values.mean(skipna=True),
                    "median": values.median(skipna=True),
                    "std": values.std(skipna=True),
                    "min": values.min(skipna=True),
                    "max": values.max(skipna=True),
                    "p95": values.quantile(0.95),
                    "count": int(values.count()),
                    "missing": int(values.isna().sum()),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "market",
            "symbol",
            "column",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "p95",
            "count",
            "missing",
        ],
    )


def make_estimator_correlations(
    panels: dict[str, pd.DataFrame],
    *,
    window: int = 22,
) -> pd.DataFrame:
    """
    Build long-format estimator correlation table.

    Parameters
    ----------
    panels:
        Mapping from market name to RV panel.
    window:
        Rolling annualised RV window.

    Returns
    -------
    pd.DataFrame
        Long-format correlation matrix.
    """
    columns = rv_annualized_columns(window)
    rows: list[dict[str, object]] = []

    for market, panel in panels.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"Panel for market '{market}' must be a DataFrame.")

        available_cols = [col for col in columns if col in panel.columns]
        if len(available_cols) < 2:
            continue

        symbol = _get_symbol(panel, market)

        numeric = panel[available_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        corr = numeric.corr()

        for estimator_1 in corr.index:
            for estimator_2 in corr.columns:
                value = corr.loc[estimator_1, estimator_2]

                if pd.isna(value):
                    continue

                if not isinstance(value, Real):
                    continue

                rows.append(
                    {
                        "market": str(market).upper(),
                        "symbol": symbol,
                        "estimator_1": estimator_1,
                        "estimator_2": estimator_2,
                        "correlation": float(cast(Real, value)),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "market",
            "symbol",
            "estimator_1",
            "estimator_2",
            "correlation",
        ],
    )


def write_rv_metadata(
    output_path: str | Path,
    *,
    window: int = 22,
    annualization_periods: int = 252,
    input_files: list[str] | None = None,
) -> Path:
    """
    Write Phase 2 realised-variance metadata JSON.

    Parameters
    ----------
    output_path:
        Metadata output path.
    window:
        Primary rolling window.
    annualization_periods:
        Annualisation factor.
    input_files:
        Phase 2 input files. Defaults to frozen Phase 1 processed files.

    Returns
    -------
    Path
        Written metadata path.
    """
    if input_files is None:
        input_files = DEFAULT_INPUT_FILES

    out_path = ensure_parent_dir(output_path)

    metadata = {
        "phase": "Phase 2 - realised variance construction",
        "phase1_frozen": True,
        "input_files": input_files,
        "primary_estimator": "garman_klass",
        "primary_column": f"rv_gk_{window}d_ann",
        "primary_window": window,
        "rolling_convention": (
            "trailing mean of daily variance, annualized by multiplying by 252"
        ),
        "annualization_periods": annualization_periods,
        "yang_zhang_convention": (
            "rolling window estimator only; no rv_yz_daily column"
        ),
        "no_lookahead_rule": (
            "rolling features use only observations up to and including current date"
        ),
        "us_underlying_note": (
            "Uses Phase 1 frozen US underlying file. If this is SPY, it is treated "
            "as ETF proxy for SPX and documented as a limitation."
        ),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_path


def plot_rv_estimators(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    market: str,
    window: int = 22,
) -> Path:
    """
    Plot annualised realised variance estimators.

    Parameters
    ----------
    df:
        RV panel.
    output_path:
        Figure output path.
    market:
        Market label.
    window:
        Rolling annualised RV window.

    Returns
    -------
    Path
        Written figure path.
    """
    if "date" not in df.columns:
        raise ValueError("RV panel must contain a 'date' column for plotting.")

    columns = rv_annualized_columns(window)
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        raise ValueError(
            f"No annualised RV columns found for window={window}. "
            f"Expected one of: {columns}"
        )

    plot_df = df[["date", *available_cols]].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    for col in available_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    out_path = ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in available_cols:
        ax.plot(plot_df["date"], plot_df[col], label=col)

    ax.set_title(f"{str(market).upper()} realised variance estimators ({window}d annualised)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised variance")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_rv_diagnostics(
    panels: dict[str, pd.DataFrame],
    *,
    table_dir: str | Path = "reports/tables",
    figure_dir: str | Path = "reports/figures",
    window: int = 22,
    annualization_periods: int = 252,
) -> dict[str, Path]:
    """
    Write all Phase 2 realised-variance diagnostics.

    Parameters
    ----------
    panels:
        Mapping from market name to RV panel.
    table_dir:
        Directory for CSV/JSON diagnostics.
    figure_dir:
        Directory for figure outputs.
    window:
        Rolling annualised RV window.
    annualization_periods:
        Annualisation factor.

    Returns
    -------
    dict[str, Path]
        Mapping from diagnostic name to written path.
    """
    if not panels:
        raise ValueError("No panels supplied for RV diagnostics.")

    table_path = Path(table_dir)
    figure_path = Path(figure_dir)

    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    created: dict[str, Path] = {}

    summary = make_rv_summary(panels, window=window)
    summary_path = table_path / "rv_summary.csv"
    summary.to_csv(summary_path, index=False)
    created["summary"] = summary_path

    correlations = make_estimator_correlations(panels, window=window)
    correlations_path = table_path / "rv_estimator_correlations.csv"
    correlations.to_csv(correlations_path, index=False)
    created["correlations"] = correlations_path

    metadata_path = write_rv_metadata(
        table_path / "rv_metadata.json",
        window=window,
        annualization_periods=annualization_periods,
    )
    created["metadata"] = metadata_path

    for market, panel in panels.items():
        market_lower = str(market).lower()
        fig_path = plot_rv_estimators(
            panel,
            figure_path / f"rv_estimators_{market_lower}.png",
            market=str(market).upper(),
            window=window,
        )
        created[f"figure_{str(market).upper()}"] = fig_path

    return created