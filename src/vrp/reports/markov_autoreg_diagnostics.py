"""
Phase 7 Markov autoregression diagnostics.

Diagnostics are post-assignment only. This module must never feed HMM states,
threshold states, crisis labels, or forward/ex-post labels into the MAR model.

Inputs:
    MARSignalOutput from src/vrp/regimes/markov_autoreg.py

Outputs:
    reports/tables/phase_7/{market}/mar_duration_summary.csv
    reports/tables/phase_7/{market}/mar_state_by_year.csv
    reports/tables/phase_7/{market}/mar_hmm_agreement.csv
    reports/tables/phase_7/{market}/mar_threshold_agreement.csv
    reports/tables/regime_model_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vrp.regimes.markov_autoreg_registry import (
    MARConfig,
    duration_summary_path,
    global_regime_model_comparison_path,
    hmm_agreement_path,
    hmm_regimes_path,
    state_by_year_path,
    threshold_agreement_path,
    threshold_regimes_path,
)

from vrp.regimes.markov_autoreg import (
    MARSignalOutput,
    fit_summary_dict,
)


DATE_COL = "date"


def write_phase7_diagnostics(
    signal: MARSignalOutput,
    cfg: MARConfig,
    *,
    force: bool,
) -> dict[str, str | None]:
    """
    Write Phase 7 diagnostic tables for one market.

    This function is intentionally called after MAR states are assigned.
    """
    market = signal.full_filter.candidate.market

    duration = build_mar_duration_summary(signal.output_frame)
    state_year = build_mar_state_by_year(signal.output_frame)

    write_csv(duration, duration_summary_path(market, cfg), force=force)
    write_csv(state_year, state_by_year_path(market, cfg), force=force)

    hmm_agreement = build_hmm_agreement_table(signal.output_frame, market, cfg)
    threshold_agreement = build_threshold_agreement_table(signal.output_frame, market, cfg)

    if hmm_agreement is not None:
        write_csv(hmm_agreement, hmm_agreement_path(market, cfg), force=force)

    if threshold_agreement is not None:
        write_csv(threshold_agreement, threshold_agreement_path(market, cfg), force=force)

    comparison_row = build_regime_model_comparison_row(signal, cfg)
    upsert_global_regime_model_comparison(
        row=comparison_row,
        path=global_regime_model_comparison_path(cfg),
    )

    return {
        "duration_summary": str(duration_summary_path(market, cfg)),
        "state_by_year": str(state_by_year_path(market, cfg)),
        "hmm_agreement": str(hmm_agreement_path(market, cfg)) if hmm_agreement is not None else None,
        "threshold_agreement": (
            str(threshold_agreement_path(market, cfg))
            if threshold_agreement is not None
            else None
        ),
        "global_regime_model_comparison": str(global_regime_model_comparison_path(cfg)),
    }


def build_mar_duration_summary(output_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise consecutive economic-state runs.

    Uses mar_state_name_for_next_session because this is the backtest-facing
    state assignment.
    """
    required = [
        DATE_COL,
        "mar_signal_trade_date",
        "mar_model_observation_available",
        "mar_state_name_for_next_session",
    ]
    missing = [col for col in required if col not in output_frame.columns]
    if missing:
        raise ValueError(f"Cannot build duration summary. Missing columns: {missing}")

    df = output_frame.copy()
    df = df[df["mar_model_observation_available"].astype(bool)].copy()
    df = df.dropna(subset=["mar_state_name_for_next_session"])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "state_name",
                "n_runs",
                "mean_duration_days",
                "median_duration_days",
                "max_duration_days",
                "min_duration_days",
                "total_days",
            ]
        )

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    state = df["mar_state_name_for_next_session"].astype(str)
    run_id = (state != state.shift(1)).cumsum()

    runs = (
        df.assign(_run_id=run_id)
        .groupby("_run_id", as_index=False)
        .agg(
            state_name=("mar_state_name_for_next_session", "first"),
            start_date=(DATE_COL, "first"),
            end_date=(DATE_COL, "last"),
            start_trade_date=("mar_signal_trade_date", "first"),
            end_trade_date=("mar_signal_trade_date", "last"),
            duration_days=(DATE_COL, "size"),
        )
    )

    summary = (
        runs.groupby("state_name", as_index=False)
        .agg(
            n_runs=("duration_days", "count"),
            mean_duration_days=("duration_days", "mean"),
            median_duration_days=("duration_days", "median"),
            max_duration_days=("duration_days", "max"),
            min_duration_days=("duration_days", "min"),
            total_days=("duration_days", "sum"),
        )
        .sort_values("state_name")
        .reset_index(drop=True)
    )

    return summary


def build_mar_state_by_year(output_frame: pd.DataFrame) -> pd.DataFrame:
    """Annual distribution of MAR economic states."""
    required = [
        "mar_signal_trade_date",
        "mar_model_observation_available",
        "mar_state_name_for_next_session",
    ]
    missing = [col for col in required if col not in output_frame.columns]
    if missing:
        raise ValueError(f"Cannot build state-by-year summary. Missing columns: {missing}")

    df = output_frame.copy()
    df = df[df["mar_model_observation_available"].astype(bool)].copy()
    df = df.dropna(subset=["mar_signal_trade_date", "mar_state_name_for_next_session"])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "state_name",
                "n_days",
                "year_total_days",
                "state_fraction",
            ]
        )

    df["year"] = pd.to_datetime(df["mar_signal_trade_date"]).dt.year
    counts = (
        df.groupby(["year", "mar_state_name_for_next_session"], as_index=False)
        .size()
        .rename(
            columns={
                "mar_state_name_for_next_session": "state_name",
                "size": "n_days",
            }
        )
    )

    totals = counts.groupby("year", as_index=False)["n_days"].sum()
    totals = totals.rename({"n_days": "year_total_days"}, axis="columns")

    out = counts.merge(totals, on="year", how="left")
    out["state_fraction"] = out["n_days"] / out["year_total_days"]

    return out.sort_values(["year", "state_name"]).reset_index(drop=True)


def build_hmm_agreement_table(
    mar_output: pd.DataFrame,
    market: str,
    cfg: MARConfig,
) -> pd.DataFrame | None:
    """
    Compare MAR economic state with Phase 6 HMM state.

    Diagnostic only. HMM agreement is not a model-selection objective.
    """
    path = hmm_regimes_path(market, cfg)

    if not path.exists():
        return None

    hmm = pd.read_parquet(path)
    hmm = standardize_date_columns(hmm)

    hmm_state_col = first_existing_column(
        hmm,
        [
            "hmm_state_name_for_next_session",
            "hmm_state_name",
            "hmm_state_for_next_session",
            "hmm_state",
        ],
    )

    if hmm_state_col is None:
        return None

    hmm_date_col = first_existing_column(
        hmm,
        [
            "hmm_signal_trade_date",
            "signal_trade_date",
            DATE_COL,
        ],
    )

    if hmm_date_col is None:
        return None

    left = mar_output.copy()
    left = standardize_date_columns(left)

    left = left[
        [
            "mar_signal_trade_date",
            "mar_model_observation_available",
            "mar_state_name_for_next_session",
        ]
    ].copy()

    left = left[left["mar_model_observation_available"].astype(bool)]
    left = left.dropna(subset=["mar_signal_trade_date", "mar_state_name_for_next_session"])
    left = left.rename(columns={"mar_signal_trade_date": "_trade_date"})

    right = hmm[[hmm_date_col, hmm_state_col]].copy()
    right = right.rename(
        columns={
            hmm_date_col: "_trade_date",
            hmm_state_col: "hmm_state_name_for_next_session",
        }
    )

    right["_trade_date"] = pd.to_datetime(right["_trade_date"], errors="coerce")
    right = right.dropna(subset=["_trade_date"])

    merged = left.merge(right, on="_trade_date", how="inner")

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "market": market,
                    "comparison": "MAR_vs_HMM",
                    "n_overlap": 0,
                    "agreement_rate": np.nan,
                    "note": "No overlapping trade dates.",
                }
            ]
        )

    merged["mar_state_norm"] = merged["mar_state_name_for_next_session"].map(normalize_state_name)
    merged["hmm_state_norm"] = merged["hmm_state_name_for_next_session"].map(normalize_state_name)
    merged["agree"] = merged["mar_state_norm"] == merged["hmm_state_norm"]

    summary = pd.DataFrame(
        [
            {
                "market": market,
                "comparison": "MAR_vs_HMM",
                "n_overlap": int(len(merged)),
                "agreement_rate": float(merged["agree"].mean()),
                "mar_state_col": "mar_state_name_for_next_session",
                "hmm_state_col": hmm_state_col,
                "join_date_col": hmm_date_col,
                "note": "Diagnostic only. Not used for MAR model selection.",
            }
        ]
    )

    ctab = pd.crosstab(
        merged["mar_state_norm"],
        merged["hmm_state_norm"],
        dropna=False,
    ).reset_index()

    ctab.insert(0, "market", market)
    ctab.insert(1, "comparison", "MAR_vs_HMM_crosstab")

    return pd.concat([summary, ctab], ignore_index=True, sort=False)


def build_threshold_agreement_table(
    mar_output: pd.DataFrame,
    market: str,
    cfg: MARConfig,
) -> pd.DataFrame | None:
    """
    Compare MAR economic state with Phase 5 threshold regime.

    Diagnostic only. Threshold agreement is not a model-selection objective.
    """
    path = threshold_regimes_path(market, cfg)

    if not path.exists():
        return None

    threshold = pd.read_parquet(path)
    threshold = standardize_date_columns(threshold)

    threshold_state_col = first_existing_column(
        threshold,
        [
            "threshold_state_name_for_next_session",
            "threshold_state_name",
            "threshold_regime_name",
            "threshold_state_for_next_session",
            "threshold_state",
        ],
    )

    if threshold_state_col is None:
        return None

    threshold_date_col = first_existing_column(
        threshold,
        [
            "threshold_signal_trade_date",
            "signal_trade_date",
            DATE_COL,
        ],
    )

    if threshold_date_col is None:
        return None

    left = mar_output.copy()
    left = standardize_date_columns(left)

    left = left[
        [
            "mar_signal_trade_date",
            "mar_model_observation_available",
            "mar_state_name_for_next_session",
        ]
    ].copy()

    left = left[left["mar_model_observation_available"].astype(bool)]
    left = left.dropna(subset=["mar_signal_trade_date", "mar_state_name_for_next_session"])
    left = left.rename(columns={"mar_signal_trade_date": "_trade_date"})

    right = threshold[[threshold_date_col, threshold_state_col]].copy()
    right = right.rename(
        columns={
            threshold_date_col: "_trade_date",
            threshold_state_col: "threshold_state_name_for_next_session",
        }
    )

    right["_trade_date"] = pd.to_datetime(right["_trade_date"], errors="coerce")
    right = right.dropna(subset=["_trade_date"])

    merged = left.merge(right, on="_trade_date", how="inner")

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "market": market,
                    "comparison": "MAR_vs_threshold",
                    "n_overlap": 0,
                    "agreement_rate": np.nan,
                    "note": "No overlapping trade dates.",
                }
            ]
        )

    merged["mar_state_norm"] = merged["mar_state_name_for_next_session"].map(normalize_state_name)
    merged["threshold_state_norm"] = merged["threshold_state_name_for_next_session"].map(
        normalize_state_name
    )
    merged["agree"] = merged["mar_state_norm"] == merged["threshold_state_norm"]

    summary = pd.DataFrame(
        [
            {
                "market": market,
                "comparison": "MAR_vs_threshold",
                "n_overlap": int(len(merged)),
                "agreement_rate": float(merged["agree"].mean()),
                "mar_state_col": "mar_state_name_for_next_session",
                "threshold_state_col": threshold_state_col,
                "join_date_col": threshold_date_col,
                "note": "Diagnostic only. Not used for MAR model selection.",
            }
        ]
    )

    ctab = pd.crosstab(
        merged["mar_state_norm"],
        merged["threshold_state_norm"],
        dropna=False,
    ).reset_index()

    ctab.insert(0, "market", market)
    ctab.insert(1, "comparison", "MAR_vs_threshold_crosstab")

    return pd.concat([summary, ctab], ignore_index=True, sort=False)


def build_regime_model_comparison_row(
    signal: MARSignalOutput,
    cfg: MARConfig,
) -> dict[str, Any]:
    """One row for reports/tables/regime_model_comparison.csv."""
    candidate = signal.full_filter.candidate
    fit = fit_summary_dict(candidate)
    output = signal.output_frame

    available = output["mar_model_observation_available"].astype(bool)
    state_col = "mar_state_name_for_next_session"

    fractions = {}
    if available.any() and state_col in output.columns:
        state_counts = output.loc[available, state_col].value_counts(normalize=True)
        fractions = {
            f"fraction_{str(k)}": float(v)
            for k, v in state_counts.items()
        }

    row = {
        "model_family": "markov_autoreg",
        "model_name": cfg.model_name,
        "market": candidate.market,
        "target": candidate.spec.target,
        "target_col": candidate.prepared.target_col,
        "order": candidate.spec.order,
        "n_states": candidate.spec.n_states,
        "switching_ar": candidate.spec.switching_ar,
        "switching_trend": candidate.spec.switching_trend,
        "switching_variance": candidate.spec.switching_variance,
        "suffix": candidate.spec.suffix(),
        "valid_candidate": candidate.fit_summary.valid_candidate,
        "fit_converged": candidate.fit_summary.fit_converged,
        "llf": candidate.fit_summary.llf,
        "aic": candidate.fit_summary.aic,
        "bic": candidate.fit_summary.bic,
        "hqic": candidate.fit_summary.hqic,
        "nobs": candidate.fit_summary.nobs,
        "n_params": candidate.fit_summary.n_params,
        "n_output_rows": int(len(output)),
        "n_model_available_rows": int(available.sum()),
        "probability_audit_passed": signal.full_filter.probability_audit.passed,
        "lookahead_audit_passed": signal.full_filter.lookahead_audit.passed,
        "economic_check_passed": bool(
            signal.state_mapping.economic_check.get("passed", False)
        ),
        "economic_check_invalid_reason": str(
            signal.state_mapping.economic_check.get("invalid_reason", "")
        ),
        "calm_raw_state": signal.state_mapping.name_to_raw_state.get("calm"),
        "transition_raw_state": signal.state_mapping.name_to_raw_state.get("transition"),
        "stress_raw_state": signal.state_mapping.name_to_raw_state.get("stress"),
        "transition_state_modelled": signal.state_mapping.transition_state_modelled,
    }

    row.update(fractions)
    return row


def upsert_global_regime_model_comparison(
    row: dict[str, Any],
    path: Path,
) -> None:
    """
    Upsert one MAR row into global model-comparison table.

    Existing non-MAR rows are preserved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    key_cols = [
        "model_family",
        "market",
        "target",
        "order",
        "n_states",
        "switching_ar",
        "switching_trend",
        "switching_variance",
        "suffix",
    ]

    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)
        for col in key_cols:
            if col not in old.columns:
                old[col] = np.nan

        old_key = make_key_frame(old, key_cols)
        new_key = make_key_frame(new, key_cols).iloc[0]
        keep_mask = ~(old_key == tuple(new_key)).all(axis=1)
        out = pd.concat([old.loc[keep_mask], new], ignore_index=True, sort=False)
    else:
        out = new

    out.to_csv(path, index=False)


def make_key_frame(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Normalize key columns for upsert comparison."""
    out = pd.DataFrame(index=df.index)
    for col in key_cols:
        out[col] = df[col].astype(str)
    return out


def normalize_state_name(value: Any) -> str:
    """Normalize regime names across MAR/HMM/threshold outputs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"

    text = str(value).strip().lower()

    if text in {"0", "calm", "low", "low_vol", "low volatility", "normal"}:
        return "calm"

    if text in {"1", "transition", "mid", "medium", "neutral", "mixed"}:
        return "transition"

    if text in {"2", "stress", "high", "high_vol", "high volatility", "crisis"}:
        return "stress"

    if "calm" in text or "low" in text:
        return "calm"

    if "transition" in text or "middle" in text or "medium" in text:
        return "transition"

    if "stress" in text or "crisis" in text or "high" in text:
        return "stress"

    return text


def standardize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns containing 'date' to datetime where possible."""
    out = df.copy()
    for col in out.columns:
        if "date" in str(col).lower():
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def write_csv(df: pd.DataFrame, path: Path, *, force: bool) -> None:
    """Write CSV with overwrite control."""
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Pass --force to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)