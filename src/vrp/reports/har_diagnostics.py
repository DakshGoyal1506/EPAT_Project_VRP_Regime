# src/vrp/reports/har_diagnostics.py

"""
Diagnostics and reporting utilities for Phase 4 HAR-RV forecasts.

Outputs:
- reports/tables/har_forecast_accuracy.csv
- reports/tables/har_coefficients.csv
- reports/tables/har_vrp_summary.csv
- reports/tables/har_metadata.json
- reports/tables/har_no_lookahead_audit.csv
- reports/figures/har_forecast_us.png
- reports/figures/har_forecast_india.png
- reports/figures/har_residuals_us.png
- reports/figures/har_residuals_india.png
- reports/figures/har_vrp_us.png
- reports/figures/har_vrp_india.png

Scope:
- Forecast accuracy tables.
- HAR coefficient history.
- HAR-VRP descriptive summary.
- Row-level no-lookahead audit table.
- Simple inspection plots.

Rules:
- Do not fit models here.
- Do not create trading signals.
- Do not create regimes.
- Do not run backtests.
- Do not forward-fill or backfill forecasts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp.features.feature_io import ensure_parent_dir
from vrp.forecasting.forecast_evaluation import (
    build_forecast_accuracy_table,
    evaluate_forecasts,
)
from vrp.forecasting.har_registry import (
    HAR_BASELINE_COLUMNS,
    HAR_FEATURE_COLUMNS,
    HAR_FORECAST_COLUMNS,
    HAR_OUTPUT_FEATURE_COLUMNS,
    HAR_TARGET_COLUMNS,
    make_har_feature_metadata,
)


DEFAULT_TARGET_COL = "rv_gk_22d_forward_ann_label"
DEFAULT_FORECAST_COL = "har_rv_gk_22d_forecast_ann"


def _normalize_market_name(market: str) -> str:
    """
    Normalize market name for output display.
    """
    return str(market).upper().strip()


def _market_file_suffix(market: str) -> str:
    """
    Return lowercase market suffix for figure names.
    """
    return _normalize_market_name(market).lower()


def _get_market(panel: pd.DataFrame, fallback: str) -> str:
    """
    Extract market label from a panel.
    """
    if "market" in panel.columns and len(panel) > 0:
        values = panel["market"].dropna().astype(str).str.upper().unique().tolist()
        if len(values) == 1:
            return values[0]

    return _normalize_market_name(fallback)


def _to_datetime_sorted(panel: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    Return date-sorted copy.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if "date" not in panel.columns:
        raise ValueError(f"{name} is missing required column: date")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    bad_date_mask = out["date"].isna()
    if bad_date_mask.any():
        bad_count = int(bad_date_mask.sum())
        raise ValueError(f"{name} contains {bad_count} invalid date value(s).")

    return out.sort_values("date").reset_index(drop=True)


def _require_columns(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
) -> None:
    """
    Raise if required columns are missing.
    """
    missing = [col for col in columns if col not in panel.columns]

    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def _numeric_series(panel: pd.DataFrame, column: str) -> pd.Series:
    """
    Return numeric series for reporting.
    """
    return pd.to_numeric(panel[column], errors="coerce")


def _write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Write CSV and create parent directory.
    """
    out_path = ensure_parent_dir(path)
    df.to_csv(out_path, index=False)
    return out_path


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """
    Write JSON and create parent directory.
    """
    out_path = ensure_parent_dir(path)

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    return out_path


def _save_figure(path: str | Path) -> Path:
    """
    Save current matplotlib figure and close it.
    """
    out_path = ensure_parent_dir(path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def summarize_har_coefficients(
    coefficient_rows: pd.DataFrame | list[dict[str, object]],
) -> pd.DataFrame:
    """
    Return coefficient history in stable column order.

    The input should contain one row per successful forecast date.
    """
    columns = [
        "date",
        "market",
        "model_name",
        "train_start_date",
        "train_end_date",
        "n_train",
        "coef_const",
        "coef_har_rv_d_lag1_ann",
        "coef_har_rv_w_lag1_ann",
        "coef_har_rv_m_lag1_ann",
        "se_const",
        "se_har_rv_d_lag1_ann",
        "se_har_rv_w_lag1_ann",
        "se_har_rv_m_lag1_ann",
        "t_const",
        "t_har_rv_d_lag1_ann",
        "t_har_rv_w_lag1_ann",
        "t_har_rv_m_lag1_ann",
        "hac_maxlags",
        "hac_available",
    ]

    if isinstance(coefficient_rows, list):
        out = pd.DataFrame(coefficient_rows)
    else:
        out = coefficient_rows.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = np.nan

    if len(out) > 0 and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["market", "date"]).reset_index(drop=True)
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    return out[columns].copy()


def build_har_forecast_summary(
    forecast_panels: dict[str, pd.DataFrame],
    *,
    target_col: str = DEFAULT_TARGET_COL,
    forecast_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build forecast accuracy summary table.

    If both US and INDIA are available, use the combined helper.
    Otherwise evaluate available markets independently.
    """
    if forecast_cols is None:
        forecast_cols = [
            DEFAULT_FORECAST_COL,
            *HAR_BASELINE_COLUMNS,
        ]

    normalized = {
        _normalize_market_name(market): panel
        for market, panel in forecast_panels.items()
        if isinstance(panel, pd.DataFrame)
    }

    if "US" in normalized and "INDIA" in normalized:
        available_cols = [
            col for col in forecast_cols
            if col in normalized["US"].columns or col in normalized["INDIA"].columns
        ]

        return build_forecast_accuracy_table(
            normalized["US"],
            normalized["INDIA"],
            target_col=target_col,
            forecast_cols=available_cols,
        )

    tables: list[pd.DataFrame] = []

    for market, panel in normalized.items():
        available_cols = [col for col in forecast_cols if col in panel.columns]

        if not available_cols:
            continue

        tables.append(
            evaluate_forecasts(
                panel,
                target_col=target_col,
                forecast_cols=available_cols,
                market=market,
            )
        )

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


def build_har_vrp_summary(
    vrp_har_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build HAR-VRP descriptive summary table.
    """
    summary_cols = [
        "iv_ann",
        "vrp_backward_gk",
        "vrp_forward_expost_gk_label",
        "har_rv_gk_22d_forecast_ann",
        "vrp_har_gk",
    ]

    rows: list[dict[str, object]] = []

    for market_key, panel in vrp_har_panels.items():
        market = _get_market(panel, fallback=market_key)

        for col in summary_cols:
            if col not in panel.columns:
                continue

            values = _numeric_series(panel, col)

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

        if "har_forecast_available" in panel.columns:
            available = panel["har_forecast_available"].astype(bool)
            rows.append(
                {
                    "market": market,
                    "column": "har_forecast_available",
                    "mean": float(available.mean()) if len(available) else float("nan"),
                    "median": float(available.median()) if len(available) else float("nan"),
                    "std": float(available.astype(float).std()) if len(available) else float("nan"),
                    "min": float(available.min()) if len(available) else float("nan"),
                    "max": float(available.max()) if len(available) else float("nan"),
                    "p05": float(available.astype(float).quantile(0.05)) if len(available) else float("nan"),
                    "p95": float(available.astype(float).quantile(0.95)) if len(available) else float("nan"),
                    "count": int(len(available)),
                    "missing": int(panel["har_forecast_available"].isna().sum()),
                    "positive_count": int(available.sum()),
                    "positive_ratio": float(available.mean()) if len(available) else float("nan"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )


def build_har_no_lookahead_audit(
    audit_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine row-level no-lookahead audit tables.
    """
    frames: list[pd.DataFrame] = []

    for market, frame in audit_frames.items():
        if not isinstance(frame, pd.DataFrame):
            continue

        out = frame.copy()

        if "market" not in out.columns:
            out["market"] = _normalize_market_name(market)

        frames.append(out)

    columns = [
        "market",
        "forecast_date",
        "n_candidate_train_rows",
        "n_valid_train_rows",
        "min_train_required",
        "max_training_row_date",
        "max_training_target_end_date",
        "forecast_date_minus_max_target_end_days",
        "rule_target_end_before_forecast_date",
        "forecast_available",
        "blocked_reason",
    ]

    if not frames:
        return pd.DataFrame(columns=columns)

    audit = pd.concat(frames, ignore_index=True)

    for col in columns:
        if col not in audit.columns:
            audit[col] = np.nan

    return audit[columns].copy()


def plot_har_forecast_vs_realized(
    forecast_panel: pd.DataFrame,
    output_path: str | Path,
    *,
    market: str,
    target_col: str = DEFAULT_TARGET_COL,
    forecast_col: str = DEFAULT_FORECAST_COL,
) -> Path:
    """
    Plot realised forward RV label vs HAR forecast and naive baseline.
    """
    panel = _to_datetime_sorted(forecast_panel, name=f"{market} HAR forecast panel")

    _require_columns(
        panel,
        ["date", target_col, forecast_col],
        name=f"{market} HAR forecast panel",
    )

    target = _numeric_series(panel, target_col)
    forecast = _numeric_series(panel, forecast_col)

    plt.figure(figsize=(12, 6))
    plt.plot(panel["date"], target, label="Forward realised RV label")
    plt.plot(panel["date"], forecast, label="HAR forecast")

    if "naive_lagged_22d_rv_ann" in panel.columns:
        naive = _numeric_series(panel, "naive_lagged_22d_rv_ann")
        plt.plot(panel["date"], naive, label="Naive lagged 22d RV")

    plt.title(f"{_normalize_market_name(market)} HAR forecast vs realised forward RV")
    plt.xlabel("Date")
    plt.ylabel("Annualized variance")
    plt.legend()

    return _save_figure(output_path)


def plot_har_residuals(
    forecast_panel: pd.DataFrame,
    output_path: str | Path,
    *,
    market: str,
    target_col: str = DEFAULT_TARGET_COL,
    forecast_col: str = DEFAULT_FORECAST_COL,
) -> Path:
    """
    Plot HAR residuals: target - forecast.
    """
    panel = _to_datetime_sorted(forecast_panel, name=f"{market} HAR forecast panel")

    _require_columns(
        panel,
        ["date", target_col, forecast_col],
        name=f"{market} HAR forecast panel",
    )

    target = _numeric_series(panel, target_col)
    forecast = _numeric_series(panel, forecast_col)
    residual = target - forecast

    plt.figure(figsize=(12, 5))
    plt.plot(panel["date"], residual, label="Forward RV label - HAR forecast")
    plt.axhline(0.0, linewidth=1)
    plt.title(f"{_normalize_market_name(market)} HAR forecast residuals")
    plt.xlabel("Date")
    plt.ylabel("Annualized variance residual")
    plt.legend()

    return _save_figure(output_path)


def plot_har_vrp(
    vrp_har_panel: pd.DataFrame,
    output_path: str | Path,
    *,
    market: str,
) -> Path:
    """
    Plot backward VRP and HAR-based prospective VRP.
    """
    panel = _to_datetime_sorted(vrp_har_panel, name=f"{market} HAR-VRP panel")

    _require_columns(
        panel,
        ["date", "vrp_har_gk"],
        name=f"{market} HAR-VRP panel",
    )

    plt.figure(figsize=(12, 6))

    if "vrp_backward_gk" in panel.columns:
        backward = _numeric_series(panel, "vrp_backward_gk")
        plt.plot(panel["date"], backward, label="Backward VRP GK")

    har_vrp = _numeric_series(panel, "vrp_har_gk")
    plt.plot(panel["date"], har_vrp, label="HAR prospective VRP GK")
    plt.axhline(0.0, linewidth=1)

    plt.title(f"{_normalize_market_name(market)} backward VRP vs HAR-based VRP")
    plt.xlabel("Date")
    plt.ylabel("Annualized variance spread")
    plt.legend()

    return _save_figure(output_path)


def write_har_metadata(
    output_path: str | Path,
    *,
    config: Any,
) -> Path:
    """
    Write Phase 4 HAR metadata JSON.
    """
    feature_metadata = make_har_feature_metadata()

    metadata = {
        "phase": "Phase 4 - HAR-RV forecasting and HAR-based VRP construction",
        "model_type": getattr(config, "model_type", "direct_har_22d"),
        "target_definition": (
            "rv_gk_22d_forward_ann_label_t = 252 * mean("
            "rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})"
        ),
        "target_source": (
            "existing Phase 3 column rv_gk_22d_forward_ann_label"
        ),
        "target_validation_policy": (
            "Recompute forward RV target from rv_gk_daily only for validation; "
            "fail if it differs from the Phase 3 label beyond configured tolerance."
        ),
        "forecast_horizon": getattr(config, "forecast_horizon", 22),
        "annualization_periods": getattr(config, "annualization_periods", 252),
        "timing_mode": getattr(config, "timing_mode", "conservative_lag1"),
        "feature_columns": list(HAR_FEATURE_COLUMNS),
        "target_column": getattr(config, "target_col", DEFAULT_TARGET_COL),
        "forecast_column": DEFAULT_FORECAST_COL,
        "min_train_observations": getattr(config, "min_train_observations", 500),
        "oos_mode": getattr(config, "oos_mode", "expanding"),
        "no_lookahead_rule": (
            "At forecast date t, HAR features use realised variance only through t-1."
        ),
        "training_label_availability_rule": (
            "Training row s is allowed for forecast date t only when "
            "target_end_date_s < t."
        ),
        "forecast_floor": getattr(config, "forecast_floor", 1.0e-8),
        "hac_maxlags": getattr(config, "hac_maxlags", 22),
        "baseline_forecasts": list(HAR_BASELINE_COLUMNS),
        "har_registry": feature_metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return _write_json(metadata, output_path)


def _extract_result_maps(
    results: dict[str, dict[str, pd.DataFrame]],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """
    Extract forecast/coefficient/audit/vrp_har maps from train_har.py results.
    """
    forecast_panels: dict[str, pd.DataFrame] = {}
    coefficient_panels: dict[str, pd.DataFrame] = {}
    audit_panels: dict[str, pd.DataFrame] = {}
    vrp_har_panels: dict[str, pd.DataFrame] = {}

    for market, payload in results.items():
        market_key = _normalize_market_name(market)

        if "forecast" in payload:
            forecast_panels[market_key] = payload["forecast"]

        if "coefficients" in payload:
            coefficient_panels[market_key] = payload["coefficients"]

        if "audit" in payload:
            audit_panels[market_key] = payload["audit"]

        if "vrp_har" in payload:
            vrp_har_panels[market_key] = payload["vrp_har"]

    return forecast_panels, coefficient_panels, audit_panels, vrp_har_panels


def _report_path(
    config: Any,
    key: str,
    default_path: str,
    *,
    table_dir: Path,
) -> Path:
    """
    Resolve report table path from config or default.
    """
    report_paths = getattr(config, "report_paths", {}) or {}

    if key in report_paths:
        path = Path(report_paths[key])
        if path.is_absolute():
            return path
        return Path.cwd() / path

    return table_dir / Path(default_path).name


def _figure_path(
    config: Any,
    key: str,
    default_path: str,
    *,
    figure_dir: Path,
) -> Path:
    """
    Resolve report figure path from config or default.
    """
    figure_paths = getattr(config, "figure_paths", {}) or {}

    if key in figure_paths:
        path = Path(figure_paths[key])
        if path.is_absolute():
            return path
        return Path.cwd() / path

    return figure_dir / Path(default_path).name


def write_har_diagnostics(
    results: dict[str, dict[str, pd.DataFrame]],
    *,
    config: Any,
    table_dir: str | Path,
    figure_dir: str | Path,
) -> dict[str, Path]:
    """
    Write all HAR diagnostics from train_har.py result payload.

    Parameters
    ----------
    results:
        Dictionary produced by scripts/train_har.py:
            {
                "US": {
                    "forecast": ...,
                    "coefficients": ...,
                    "audit": ...,
                    "vrp_har": ...,
                },
                "INDIA": {...},
            }
    config:
        HARConfig or config-like object.
    table_dir:
        Report table directory.
    figure_dir:
        Report figure directory.

    Returns
    -------
    dict[str, Path]
        Mapping of output name to written path.
    """
    if not results:
        raise ValueError("No HAR results supplied for diagnostics.")

    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)

    (
        forecast_panels,
        coefficient_panels,
        audit_panels,
        vrp_har_panels,
    ) = _extract_result_maps(results)

    written: dict[str, Path] = {}

    target_col = getattr(config, "target_col", DEFAULT_TARGET_COL)

    forecast_accuracy = build_har_forecast_summary(
        forecast_panels,
        target_col=target_col,
        forecast_cols=[
            DEFAULT_FORECAST_COL,
            *HAR_BASELINE_COLUMNS,
        ],
    )

    forecast_accuracy_path = _report_path(
        config,
        "forecast_accuracy",
        "reports/tables/har_forecast_accuracy.csv",
        table_dir=table_dir,
    )
    written["forecast_accuracy"] = _write_csv(
        forecast_accuracy,
        forecast_accuracy_path,
    )

    all_coefficients = pd.concat(
        [
            summarize_har_coefficients(frame)
            for frame in coefficient_panels.values()
        ],
        ignore_index=True,
    ) if coefficient_panels else summarize_har_coefficients([])

    coefficients_path = _report_path(
        config,
        "coefficients",
        "reports/tables/har_coefficients.csv",
        table_dir=table_dir,
    )
    written["coefficients"] = _write_csv(all_coefficients, coefficients_path)

    vrp_summary = build_har_vrp_summary(vrp_har_panels)

    vrp_summary_path = _report_path(
        config,
        "vrp_summary",
        "reports/tables/har_vrp_summary.csv",
        table_dir=table_dir,
    )
    written["vrp_summary"] = _write_csv(vrp_summary, vrp_summary_path)

    audit = build_har_no_lookahead_audit(audit_panels)

    audit_path = _report_path(
        config,
        "no_lookahead_audit",
        "reports/tables/har_no_lookahead_audit.csv",
        table_dir=table_dir,
    )
    written["no_lookahead_audit"] = _write_csv(audit, audit_path)

    metadata_path = _report_path(
        config,
        "metadata",
        "reports/tables/har_metadata.json",
        table_dir=table_dir,
    )
    written["metadata"] = write_har_metadata(
        metadata_path,
        config=config,
    )

    for market, forecast_panel in forecast_panels.items():
        suffix = _market_file_suffix(market)

        forecast_fig_path = _figure_path(
            config,
            f"forecast_{market}",
            f"reports/figures/har_forecast_{suffix}.png",
            figure_dir=figure_dir,
        )
        written[f"forecast_figure_{market}"] = plot_har_forecast_vs_realized(
            forecast_panel,
            forecast_fig_path,
            market=market,
            target_col=target_col,
            forecast_col=DEFAULT_FORECAST_COL,
        )

        residual_fig_path = _figure_path(
            config,
            f"residuals_{market}",
            f"reports/figures/har_residuals_{suffix}.png",
            figure_dir=figure_dir,
        )
        written[f"residuals_figure_{market}"] = plot_har_residuals(
            forecast_panel,
            residual_fig_path,
            market=market,
            target_col=target_col,
            forecast_col=DEFAULT_FORECAST_COL,
        )

    for market, vrp_har_panel in vrp_har_panels.items():
        suffix = _market_file_suffix(market)

        vrp_fig_path = _figure_path(
            config,
            f"vrp_{market}",
            f"reports/figures/har_vrp_{suffix}.png",
            figure_dir=figure_dir,
        )
        written[f"vrp_figure_{market}"] = plot_har_vrp(
            vrp_har_panel,
            vrp_fig_path,
            market=market,
        )

    return written