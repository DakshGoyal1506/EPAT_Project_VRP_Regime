from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vrp.reports.cross_market import (
    CrossMarketInputError,
    _repo_path,
)


class CrossMarketDiagnosticsError(RuntimeError):
    """Raised when Phase 13 diagnostics/report generation fails."""


def _ensure_dir(path: str | Path, root: str | Path | None = None) -> Path:
    out = _repo_path(path, root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ensure_parent(path: str | Path, root: str | Path | None = None) -> Path:
    out = _repo_path(path, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _output_path(
    config: Mapping[str, Any],
    section: str,
    key: str,
) -> str:
    values = config.get(section, {})
    if key not in values:
        raise CrossMarketDiagnosticsError(f"Missing {section}.{key} in config.")
    return str(values[key])


def _table_output_path(config: Mapping[str, Any], key: str) -> str:
    outputs = config.get("outputs", {})
    if key not in outputs:
        raise CrossMarketDiagnosticsError(f"Missing outputs.{key} in config.")
    return str(outputs[key])


def _figure_output_path(config: Mapping[str, Any], key: str) -> str:
    return _output_path(config, "figures", key)


def ensure_phase13_dirs(
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Ensure Phase 13 report directories exist.
    """
    outputs = config.get("outputs", {})
    tables_dir = outputs.get("tables_dir", "reports/tables/phase_13")
    figures_dir = outputs.get("figures_dir", "reports/figures/phase_13")

    return {
        "tables_dir": _ensure_dir(tables_dir, root=root),
        "figures_dir": _ensure_dir(figures_dir, root=root),
    }


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    root: str | Path | None = None,
    *,
    index: bool = False,
) -> Path:
    """
    Write a Phase 13 CSV table.
    """
    out = _ensure_parent(path, root=root)
    df.to_csv(out, index=index)
    return out


def _coerce_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise CrossMarketDiagnosticsError(
            f"Missing date column {col!r}. Available columns: {list(df.columns)}"
        )
    return pd.to_datetime(df[col], errors="coerce").dt.normalize()


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise CrossMarketDiagnosticsError(
            f"Missing numeric column {col!r}. Available columns: {list(df.columns)}"
        )
    return pd.to_numeric(df[col], errors="coerce")


def _select_model_panel(
    df: pd.DataFrame,
    model: str | None = None,
    panel_type: str | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if model is not None and "model" in out.columns:
        out = out[out["model"].astype(str) == str(model)].copy()

    if panel_type is not None and "panel_type" in out.columns:
        out = out[out["panel_type"].astype(str) == str(panel_type)].copy()

    return out


def _prepare_plot_data(
    df: pd.DataFrame,
    *,
    date_col: str,
    columns: list[str],
    model: str | None = None,
    panel_type: str | None = None,
) -> pd.DataFrame:
    out = _select_model_panel(df, model=model, panel_type=panel_type)
    if out.empty:
        raise CrossMarketDiagnosticsError(
            f"No rows available for model={model!r}, panel_type={panel_type!r}."
        )

    out = out.copy()
    out[date_col] = _coerce_date_col(out, date_col)

    for col in columns:
        out[col] = _numeric_col(out, col)

    out = out[[date_col] + columns].dropna(subset=[date_col])
    out = out.sort_values(date_col)
    return out


def _save_current_figure(path: str | Path, root: str | Path | None = None) -> Path:
    """
    Save the active matplotlib figure.

    Matplotlib's savefig writes the current figure to a file path, with format
    inferred from the extension when possible.
    """
    import matplotlib.pyplot as plt

    out = _ensure_parent(path, root=root)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _available_models(df: pd.DataFrame) -> list[str | None]:
    if "model" not in df.columns:
        return [None]
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    return models or [None]


def plot_us_india_vrp(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    model: str | None = None,
) -> Path:
    """
    Plot US and India VRP from the same-date descriptive panel.
    """
    import matplotlib.pyplot as plt

    plot_df = _prepare_plot_data(
        panel,
        date_col="date",
        columns=["us_vrp_har_gk", "india_vrp_har_gk"],
        model=model,
        panel_type="descriptive_same_date",
    )

    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["us_vrp_har_gk"], label="US VRP")
    plt.plot(plot_df["date"], plot_df["india_vrp_har_gk"], label="India VRP")
    plt.title(f"US and India VRP — {model or 'all models'}")
    plt.xlabel("Date")
    plt.ylabel("VRP")
    plt.legend()

    return _save_current_figure(
        _figure_output_path(config, "us_india_vrp"),
        root=root,
    )


def plot_us_india_stress_prob(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    model: str | None = None,
) -> Path:
    """
    Plot same-date US and India stress probabilities.
    """
    import matplotlib.pyplot as plt

    plot_df = _prepare_plot_data(
        panel,
        date_col="date",
        columns=["us_stress_prob", "india_stress_prob"],
        model=model,
        panel_type="descriptive_same_date",
    )

    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["us_stress_prob"], label="US stress probability")
    plt.plot(
        plot_df["date"],
        plot_df["india_stress_prob"],
        label="India stress probability",
    )
    plt.title(f"US and India stress probabilities — {model or 'all models'}")
    plt.xlabel("Date")
    plt.ylabel("Stress probability")
    plt.legend()

    return _save_current_figure(
        _figure_output_path(config, "us_india_stress_prob"),
        root=root,
    )


def plot_lagged_us_vs_india_stress(
    predictive_panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    model: str | None = None,
) -> Path:
    """
    Plot lagged US stress probability against India stress probability.
    """
    import matplotlib.pyplot as plt

    x_col = (
        "us_stress_prob_lag1"
        if "us_stress_prob_lag1" in predictive_panel.columns
        else "us_stress_prob"
    )

    plot_df = _prepare_plot_data(
        predictive_panel,
        date_col="india_date",
        columns=[x_col, "india_stress_prob"],
        model=model,
        panel_type="predictive_lagged",
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        plot_df["india_date"],
        plot_df[x_col],
        label="Lagged US stress probability",
    )
    plt.plot(
        plot_df["india_date"],
        plot_df["india_stress_prob"],
        label="India stress probability",
    )
    plt.title(f"Lagged US stress vs India stress — {model or 'all models'}")
    plt.xlabel("India date")
    plt.ylabel("Stress probability")
    plt.legend()

    return _save_current_figure(
        _figure_output_path(config, "lagged_us_vs_india_stress"),
        root=root,
    )


def _overlay_filter(
    overlay_panel: pd.DataFrame,
    *,
    model: str | None = None,
    strategy: str | None = None,
    cutoff: float | None = None,
) -> pd.DataFrame:
    out = overlay_panel.copy()

    if model is not None and "model" in out.columns:
        out = out[out["model"].astype(str) == str(model)].copy()

    if strategy is not None and "strategy" in out.columns:
        out = out[out["strategy"].astype(str) == str(strategy)].copy()

    if cutoff is not None and "cutoff" in out.columns:
        out = out[
            np.isclose(pd.to_numeric(out["cutoff"], errors="coerce"), cutoff)
        ].copy()

    if out.empty:
        raise CrossMarketDiagnosticsError(
            f"No overlay rows for model={model!r}, strategy={strategy!r}, "
            f"cutoff={cutoff!r}."
        )

    return out


def _default_overlay_selection(
    overlay_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[str | None, str | None, float | None]:
    overlay_cfg = config.get("overlay", {})
    default_cutoff = overlay_cfg.get("default_us_stress_cutoff", None)

    model = None
    strategy = None

    if "model" in overlay_panel.columns:
        models = sorted(overlay_panel["model"].dropna().astype(str).unique().tolist())
        if "markov_autoreg" in models:
            model = "markov_autoreg"
        elif models:
            model = models[0]

    if "strategy" in overlay_panel.columns:
        strategies = sorted(
            overlay_panel["strategy"].dropna().astype(str).unique().tolist()
        )
        preferred = overlay_cfg.get("primary_india_strategy")
        if preferred in strategies:
            strategy = str(preferred)
        elif strategies:
            strategy = strategies[0]

    cutoff = float(default_cutoff) if default_cutoff is not None else None

    return model, strategy, cutoff


def plot_overlay_equity_curves(
    overlay_panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    model: str | None = None,
    strategy: str | None = None,
    cutoff: float | None = None,
) -> Path:
    """
    Plot base and overlay equity curves for one model/strategy/cutoff.
    """
    import matplotlib.pyplot as plt

    if model is None and strategy is None and cutoff is None:
        model, strategy, cutoff = _default_overlay_selection(overlay_panel, config)

    plot_df = _overlay_filter(
        overlay_panel,
        model=model,
        strategy=strategy,
        cutoff=cutoff,
    )

    required = ["date", "base_equity", "overlay_equity"]
    missing = sorted(set(required) - set(plot_df.columns))
    if missing:
        raise CrossMarketDiagnosticsError(
            f"Overlay equity plot missing columns: {missing}"
        )

    plot_df = plot_df.copy()
    plot_df["date"] = _coerce_date_col(plot_df, "date")
    plot_df["base_equity"] = _numeric_col(plot_df, "base_equity")
    plot_df["overlay_equity"] = _numeric_col(plot_df, "overlay_equity")
    plot_df = plot_df.sort_values("date")

    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["base_equity"], label="Base")
    plt.plot(plot_df["date"], plot_df["overlay_equity"], label="Overlay")
    plt.title(
        "India overlay equity curves"
        f" — {model or 'model'} / {strategy or 'strategy'} / cutoff={cutoff}"
    )
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()

    return _save_current_figure(
        _figure_output_path(config, "india_overlay_equity_curves"),
        root=root,
    )


def plot_overlay_exposure(
    overlay_panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    model: str | None = None,
    strategy: str | None = None,
    cutoff: float | None = None,
) -> Path:
    """
    Plot base and overlay exposure for one model/strategy/cutoff.
    """
    import matplotlib.pyplot as plt

    if model is None and strategy is None and cutoff is None:
        model, strategy, cutoff = _default_overlay_selection(overlay_panel, config)

    plot_df = _overlay_filter(
        overlay_panel,
        model=model,
        strategy=strategy,
        cutoff=cutoff,
    )

    required = ["date", "base_exposure", "overlay_exposure"]
    missing = sorted(set(required) - set(plot_df.columns))
    if missing:
        raise CrossMarketDiagnosticsError(
            f"Overlay exposure plot missing columns: {missing}"
        )

    plot_df = plot_df.copy()
    plot_df["date"] = _coerce_date_col(plot_df, "date")
    plot_df["base_exposure"] = _numeric_col(plot_df, "base_exposure")
    plot_df["overlay_exposure"] = _numeric_col(plot_df, "overlay_exposure")
    plot_df = plot_df.sort_values("date")

    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["base_exposure"], label="Base exposure")
    plt.plot(plot_df["date"], plot_df["overlay_exposure"], label="Overlay exposure")
    plt.title(
        "India overlay exposure"
        f" — {model or 'model'} / {strategy or 'strategy'} / cutoff={cutoff}"
    )
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.legend()

    return _save_current_figure(
        _figure_output_path(config, "india_overlay_exposure"),
        root=root,
    )


def _table_status_counts(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "n_rows": 0,
            "n_ok": 0,
            "n_skipped": 0,
            "n_error": 0,
        }

    if "status" not in df.columns:
        return {
            "n_rows": int(len(df)),
            "n_ok": np.nan,
            "n_skipped": np.nan,
            "n_error": np.nan,
        }

    status = df["status"].astype(str).str.lower()
    return {
        "n_rows": int(len(df)),
        "n_ok": int((status == "ok").sum()),
        "n_skipped": int((status == "skipped").sum()),
        "n_error": int((status == "error").sum()),
    }


def _audit_pass_value(df: pd.DataFrame, table_name: str) -> bool | float:
    if df is None or df.empty:
        return False

    if table_name == "alignment_audit" and "n_same_date_violations" in df.columns:
        return bool(
            pd.to_numeric(df["n_same_date_violations"], errors="coerce")
            .fillna(1)
            .eq(0)
            .all()
        )

    if table_name == "no_lookahead_audit" and "passes_no_lookahead" in df.columns:
        return bool(df["passes_no_lookahead"].fillna(False).astype(bool).all())

    if table_name == "overlay_summary" and "analysis_only" in df.columns:
        return bool(df["analysis_only"].fillna(False).astype(bool).all())

    return np.nan


def write_phase13_summary_index(
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, str | Path] | None,
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Write a compact index of generated Phase 13 outputs.

    This is useful for final report review.
    """
    rows: list[dict[str, Any]] = []

    for name, df in tables.items():
        counts = _table_status_counts(df)
        path = config.get("outputs", {}).get(name, "")
        rows.append(
            {
                "artifact_type": "table",
                "artifact_name": name,
                "configured_path": str(path),
                "exists_after_write": (
                    bool(_repo_path(path, root).exists()) if path else False
                ),
                "n_rows": counts["n_rows"],
                "n_ok": counts["n_ok"],
                "n_skipped": counts["n_skipped"],
                "n_error": counts["n_error"],
                "audit_pass": _audit_pass_value(df, name),
            }
        )

    for name, path in (figures or {}).items():
        rows.append(
            {
                "artifact_type": "figure",
                "artifact_name": name,
                "configured_path": str(path),
                "exists_after_write": bool(Path(path).exists()),
                "n_rows": np.nan,
                "n_ok": np.nan,
                "n_skipped": np.nan,
                "n_error": np.nan,
                "audit_pass": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out_path = _table_output_path(config, "phase13_summary_index")
    write_table(out, out_path, root=root)
    return out


def write_all_stat_tables(
    stat_tables: Mapping[str, pd.DataFrame],
    logistic_tables: Mapping[str, pd.DataFrame],
    overlay_tables: Mapping[str, pd.DataFrame] | None,
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Write all Phase 13 diagnostic CSV tables that are provided.
    """
    ensure_phase13_dirs(config, root=root)
    written: dict[str, Path] = {}

    table_sources: dict[str, pd.DataFrame] = {}
    table_sources.update(dict(stat_tables))
    table_sources.update(dict(logistic_tables))

    if overlay_tables:
        table_sources.update(dict(overlay_tables))

    for key, df in table_sources.items():
        if key not in config.get("outputs", {}):
            continue
        path = _table_output_path(config, key)
        written[key] = write_table(df, path, root=root)

    return written


def generate_core_figures(
    descriptive_panel: pd.DataFrame,
    predictive_panel: pd.DataFrame,
    overlay_panel: pd.DataFrame | None,
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Generate the standard Phase 13 figures.

    Figures are skipped if required inputs are empty or missing; the error is
    encoded in the returned path table by absence, not by crashing the full run.
    """
    ensure_phase13_dirs(config, root=root)

    written: dict[str, Path] = {}

    model = None
    models = _available_models(descriptive_panel)
    if "markov_autoreg" in models:
        model = "markov_autoreg"
    elif models:
        model = models[0]

    plotters = [
        (
            "us_india_vrp",
            lambda: plot_us_india_vrp(
                descriptive_panel,
                config,
                root=root,
                model=model,
            ),
        ),
        (
            "us_india_stress_prob",
            lambda: plot_us_india_stress_prob(
                descriptive_panel,
                config,
                root=root,
                model=model,
            ),
        ),
        (
            "lagged_us_vs_india_stress",
            lambda: plot_lagged_us_vs_india_stress(
                predictive_panel,
                config,
                root=root,
                model=model,
            ),
        ),
    ]

    for name, fn in plotters:
        try:
            written[name] = fn()
        except Exception:
            continue

    if overlay_panel is not None and not overlay_panel.empty:
        overlay_plotters = [
            (
                "india_overlay_equity_curves",
                lambda: plot_overlay_equity_curves(overlay_panel, config, root=root),
            ),
            (
                "india_overlay_exposure",
                lambda: plot_overlay_exposure(overlay_panel, config, root=root),
            ),
        ]

        for name, fn in overlay_plotters:
            try:
                written[name] = fn()
            except Exception:
                continue

    return written


def validate_phase13_required_tables(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> None:
    """
    Validate required Phase 13 diagnostic table contracts.
    """
    required = [
        "alignment_audit",
        "no_lookahead_audit",
        "vrp_level_correlations",
        "vrp_change_correlations",
        "regime_probability_correlations",
        "state_label_agreement",
        "lead_lag_table",
        "granger_diagnostics",
        "logistic_model_summary",
        "logistic_model_comparison",
        "logistic_oos_diagnostics",
    ]

    missing_tables = [name for name in required if name not in tables]
    if missing_tables:
        raise CrossMarketDiagnosticsError(
            f"Missing required Phase 13 tables: {missing_tables}"
        )

    alignment = tables["alignment_audit"]
    if "n_same_date_violations" not in alignment.columns:
        raise CrossMarketDiagnosticsError(
            "alignment_audit missing n_same_date_violations."
        )
    if not pd.to_numeric(
        alignment["n_same_date_violations"],
        errors="coerce",
    ).fillna(1).eq(0).all():
        raise CrossMarketDiagnosticsError(
            "alignment_audit has same-date violations."
        )

    no_lookahead = tables["no_lookahead_audit"]
    if "passes_no_lookahead" not in no_lookahead.columns:
        raise CrossMarketDiagnosticsError(
            "no_lookahead_audit missing passes_no_lookahead."
        )
    if not no_lookahead["passes_no_lookahead"].fillna(False).astype(bool).all():
        raise CrossMarketDiagnosticsError("no_lookahead_audit failed.")

    granger = tables["granger_diagnostics"]
    for col in ("descriptive_only", "causal_interpretation_allowed"):
        if col not in granger.columns:
            raise CrossMarketDiagnosticsError(
                f"granger_diagnostics missing {col}."
            )
    if not granger["descriptive_only"].fillna(False).astype(bool).all():
        raise CrossMarketDiagnosticsError(
            "granger_diagnostics must be descriptive_only."
        )
    if granger["causal_interpretation_allowed"].fillna(True).astype(bool).any():
        raise CrossMarketDiagnosticsError(
            "granger_diagnostics cannot allow causal interpretation."
        )

    comparison = tables["logistic_model_comparison"]
    required_comparison_cols = [
        "delta_pseudo_r2",
        "delta_aic_plus_minus_local",
        "delta_bic_plus_minus_local",
        "delta_log_likelihood",
        "delta_auc",
        "delta_brier_score_plus_minus_local",
        "likelihood_ratio_p_value",
    ]
    missing = sorted(set(required_comparison_cols) - set(comparison.columns))
    if missing:
        raise CrossMarketDiagnosticsError(
            f"logistic_model_comparison missing columns: {missing}"
        )


def build_phase13_report_bundle(
    *,
    descriptive_panel: pd.DataFrame,
    predictive_panel: pd.DataFrame,
    stat_tables: Mapping[str, pd.DataFrame],
    logistic_tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    root: str | Path | None = None,
    overlay_panel: pd.DataFrame | None = None,
    overlay_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Write tables, figures, and summary index for Phase 13.

    This function does not build modelling outputs. It only reports them.
    """
    ensure_phase13_dirs(config, root=root)

    all_tables: dict[str, pd.DataFrame] = {}
    all_tables.update(dict(stat_tables))
    all_tables.update(dict(logistic_tables))

    if overlay_summary is not None:
        all_tables["overlay_summary"] = overlay_summary

    written_tables = write_all_stat_tables(
        stat_tables=stat_tables,
        logistic_tables=logistic_tables,
        overlay_tables={"overlay_summary": overlay_summary}
        if overlay_summary is not None
        else None,
        config=config,
        root=root,
    )

    written_figures = generate_core_figures(
        descriptive_panel=descriptive_panel,
        predictive_panel=predictive_panel,
        overlay_panel=overlay_panel,
        config=config,
        root=root,
    )

    summary_index = write_phase13_summary_index(
        tables=all_tables,
        figures={k: str(v) for k, v in written_figures.items()},
        config=config,
        root=root,
    )

    return {
        "written_tables": written_tables,
        "written_figures": written_figures,
        "phase13_summary_index": summary_index,
    }
