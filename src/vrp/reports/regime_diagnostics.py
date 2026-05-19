from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp.regimes.regime_registry import (
    REGIME_FORBIDDEN_FEATURE_SUBSTRINGS,
    get_allowed_diagnostic_labels,
    get_allowed_regime_features,
)
from vrp.regimes.state_labeling import CALM, STRESS, TRANSITION, STATE_ID_TO_NAME


STATE_ORDER = [CALM, TRANSITION, STRESS]
STATE_NAME_ORDER = ["calm", "transition", "stress"]

COMPONENT_STATE_COLUMNS = [
    "iv_percentile_state",
    "rv_percentile_state",
    "drawdown_state",
    "iv_slope_state",
    "vrp_har_state",
]

COMPONENT_NAME_COLUMNS = [
    "iv_percentile_state_name",
    "rv_percentile_state_name",
    "drawdown_state_name",
    "iv_slope_state_name",
    "vrp_har_state_name",
]

SUMMARY_NUMERIC_COLUMNS = [
    "iv_ann",
    "iv_close",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "vrp_har_gk",
    "har_rv_gk_22d_forecast_ann",
    "log_return",
    "simple_return",
]

FORWARD_LABEL_COLUMNS = [
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
]


def build_threshold_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final-regime summary table.

    Output:
        reports/tables/threshold_regime_summary.csv
    """
    data = _prepare_threshold_df(df)
    available = data[data["threshold_regime_available"]].copy()

    if available.empty:
        return _empty_regime_summary()

    rows = []
    group_cols = ["market", "threshold_model_name", "threshold_state", "threshold_state_name"]

    total_days_by_market = available.groupby(["market", "threshold_model_name"]).size().to_dict()

    for keys, group in available.groupby(group_cols, dropna=False):
        market, model_name, state, state_name = keys
        total_days = int(total_days_by_market.get((market, model_name), 0))
        n_days = int(len(group))

        row = {
            "market": market,
            "threshold_model_name": model_name,
            "state": int(state) if state is not None and pd.notna(state) else state,
            "state_name": state_name,
            "n_days": n_days,
            "fraction_days": n_days / total_days if total_days else np.nan,
            "vrp_har_positive_ratio": _positive_ratio(group, "vrp_har_gk"),
        }

        for col in SUMMARY_NUMERIC_COLUMNS:
            if col in group.columns:
                row[f"avg_{col}"] = _safe_mean(group[col])
            elif col in {"log_return", "simple_return"}:
                row[f"avg_{col}"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    return _sort_state_table(result)


def build_threshold_component_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize component-state distributions by market.

    Output:
        reports/tables/threshold_component_summary.csv
    """
    data = _prepare_threshold_df(df)

    rows = []

    for component_col in COMPONENT_STATE_COLUMNS:
        if component_col not in data.columns:
            continue

        available_col = _availability_col(component_col)
        blocked_col = _blocked_reason_col(component_col)

        component_name = component_col.removesuffix("_state")

        for (market, model_name), market_df in data.groupby(
            ["market", "threshold_model_name"],
            dropna=False,
        ):
            n_total = int(len(market_df))
            n_available = (
                int(market_df[available_col].sum())
                if available_col in market_df.columns
                else int(market_df[component_col].notna().sum())
            )
            n_unavailable = n_total - n_available

            state_counts = (
                market_df[component_col]
                .dropna()
                .astype(int)
                .value_counts()
                .reindex(STATE_ORDER, fill_value=0)
            )

            for state in STATE_ORDER:
                n_days = int(state_counts.loc[state])
                rows.append(
                    {
                        "market": market,
                        "threshold_model_name": model_name,
                        "component": component_name,
                        "component_state_col": component_col,
                        "state": state,
                        "state_name": STATE_ID_TO_NAME[state],
                        "n_days": n_days,
                        "fraction_available_days": (
                            n_days / n_available if n_available else np.nan
                        ),
                        "n_component_available": n_available,
                        "n_component_unavailable": n_unavailable,
                        "fraction_component_available": (
                            n_available / n_total if n_total else np.nan
                        ),
                    }
                )

            if blocked_col in market_df.columns:
                blocked = market_df.loc[
                    market_df[component_col].isna(),
                    blocked_col,
                ].fillna("unknown")
                for reason, count in blocked.value_counts().items():
                    rows.append(
                        {
                            "market": market,
                            "threshold_model_name": model_name,
                            "component": component_name,
                            "component_state_col": component_col,
                            "state": pd.NA,
                            "state_name": "unavailable",
                            "n_days": int(count),
                            "fraction_available_days": np.nan,
                            "n_component_available": n_available,
                            "n_component_unavailable": n_unavailable,
                            "fraction_component_available": (
                                n_available / n_total if n_total else np.nan
                            ),
                            "blocked_reason": reason,
                        }
                    )

    return pd.DataFrame(rows)


def build_threshold_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build transition matrix of final threshold states.

    Output:
        reports/tables/threshold_transition_matrix.csv
    """
    data = _prepare_threshold_df(df)
    data = data[data["threshold_regime_available"]].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "threshold_model_name",
                "from_state",
                "from_state_name",
                "to_state",
                "to_state_name",
                "count",
                "probability",
            ]
        )

    rows = []

    for (market, model_name), group in data.groupby(
        ["market", "threshold_model_name"],
        dropna=False,
    ):
        group = group.sort_values("date").copy()
        group["from_state"] = group["threshold_state"].shift(1)
        group["to_state"] = group["threshold_state"]

        transitions = group.dropna(subset=["from_state", "to_state"]).copy()
        if transitions.empty:
            continue

        transitions["from_state"] = transitions["from_state"].astype(int)
        transitions["to_state"] = transitions["to_state"].astype(int)

        counts = pd.crosstab(
            transitions["from_state"],
            transitions["to_state"],
            dropna=False,
        )

        counts = counts.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)

        probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

        for from_state in STATE_ORDER:
            for to_state in STATE_ORDER:
                rows.append(
                    {
                        "market": market,
                        "threshold_model_name": model_name,
                        "from_state": from_state,
                        "from_state_name": STATE_ID_TO_NAME[from_state],
                        "to_state": to_state,
                        "to_state_name": STATE_ID_TO_NAME[to_state],
                        "count": int(counts.loc[from_state, to_state]),
                        "probability": probabilities.loc[from_state, to_state],
                    }
                )

    return pd.DataFrame(rows)


def build_threshold_state_duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize final-regime episode duration.

    Output:
        reports/tables/threshold_state_duration_summary.csv
    """
    data = _prepare_threshold_df(df)
    data = data[data["threshold_regime_available"]].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "threshold_model_name",
                "state",
                "state_name",
                "n_episodes",
                "avg_duration_days",
                "median_duration_days",
                "max_duration_days",
                "p90_duration_days",
            ]
        )

    episode_rows = []

    for (market, model_name), group in data.groupby(
        ["market", "threshold_model_name"],
        dropna=False,
    ):
        group = group.sort_values("date").copy()
        group["state_change"] = group["threshold_state"].ne(
            group["threshold_state"].shift(1)
        )
        group["episode_id"] = group["state_change"].cumsum()

        for _, ep in group.groupby("episode_id"):
            state = int(ep["threshold_state"].iloc[0])
            episode_rows.append(
                {
                    "market": market,
                    "threshold_model_name": model_name,
                    "state": state,
                    "state_name": STATE_ID_TO_NAME[state],
                    "duration_days": int(len(ep)),
                    "start_date": ep["date"].min(),
                    "end_date": ep["date"].max(),
                }
            )

    episodes = pd.DataFrame(episode_rows)

    if episodes.empty:
        return pd.DataFrame()

    summary = (
        episodes.groupby(
            ["market", "threshold_model_name", "state", "state_name"],
            dropna=False,
        )["duration_days"]
        .agg(
            n_episodes="count",
            avg_duration_days="mean",
            median_duration_days="median",
            max_duration_days="max",
            p90_duration_days=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )

    return _sort_state_table(summary)


def build_threshold_state_by_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build annual state distribution table.

    Output:
        reports/tables/threshold_state_by_year.csv
    """
    data = _prepare_threshold_df(df)
    data = data[data["threshold_regime_available"]].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "year",
                "state_name",
                "n_days",
                "fraction_days",
                "avg_iv_ann",
                "avg_rv_gk_22d_ann_lag1",
                "avg_vrp_har_gk",
            ]
        )

    data["year"] = data["date"].dt.year

    rows = []

    for (market, model_name, year), year_df in data.groupby(
        ["market", "threshold_model_name", "year"],
        dropna=False,
    ):
        total_days = int(len(year_df))

        for state in STATE_ORDER:
            state_df = year_df[year_df["threshold_state"] == state]
            n_days = int(len(state_df))

            rows.append(
                {
                    "market": market,
                    "threshold_model_name": model_name,
                    "year": int(year),
                    "state": state,
                    "state_name": STATE_ID_TO_NAME[state],
                    "n_days": n_days,
                    "fraction_days": n_days / total_days if total_days else np.nan,
                    "avg_iv_ann": _safe_mean(state_df.get("iv_ann")),
                    "avg_rv_gk_22d_ann_lag1": _safe_mean(
                        state_df.get("rv_gk_22d_ann_lag1")
                    ),
                    "avg_vrp_har_gk": _safe_mean(state_df.get("vrp_har_gk")),
                }
            )

    return _sort_state_table(pd.DataFrame(rows), extra_sort_cols=["year"])


def build_threshold_crisis_hit_table(
    df: pd.DataFrame,
    crisis_windows: Mapping[str, Sequence[Sequence[str]]],
    skip_windows_outside_sample: bool = True,
) -> pd.DataFrame:
    """
    Build crisis-window hit table.

    Crisis windows are reporting annotations only. They are never used to construct
    threshold states.

    Parameters:
        skip_windows_outside_sample: if True, skip crisis windows with no sample coverage.

    Output:
        reports/tables/threshold_crisis_hit_table.csv
    """
    data = _prepare_threshold_df(df)
    rows = []

    for market, windows in crisis_windows.items():
        market = str(market).upper()
        market_df = data[
            (data["market"].astype(str).str.upper() == market)
            & (data["threshold_regime_available"])
        ].copy()

        if market_df.empty:
            continue

        model_name = _single_or_first(market_df["threshold_model_name"])

        for start_date, end_date, crisis_name in windows:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            window_df = market_df[
                (market_df["date"] >= start_ts)
                & (market_df["date"] <= end_ts)
            ].copy()

            n_days = int(len(window_df))
            if n_days == 0 and skip_windows_outside_sample:
                continue
            stress_days = int((window_df["threshold_state"] == STRESS).sum())
            transition_days = int((window_df["threshold_state"] == TRANSITION).sum())
            calm_days = int((window_df["threshold_state"] == CALM).sum())

            rows.append(
                {
                    "market": market,
                    "threshold_model_name": model_name,
                    "crisis_name": crisis_name,
                    "start_date": start_ts.date().isoformat(),
                    "end_date": end_ts.date().isoformat(),
                    "n_days": n_days,
                    "stress_days": stress_days,
                    "transition_days": transition_days,
                    "calm_days": calm_days,
                    "stress_fraction": stress_days / n_days if n_days else np.nan,
                    "transition_or_stress_fraction": (
                        (stress_days + transition_days) / n_days
                        if n_days
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_threshold_crisis_lead_lag_table(
    df: pd.DataFrame,
    crisis_windows: Mapping[str, Sequence[Sequence[str]]],
    skip_windows_outside_sample: bool = True,
) -> pd.DataFrame:
    """
    Build crisis lead/lag diagnostics.

    Parameters:
        skip_windows_outside_sample: if True, skip crisis windows with no sample coverage.

    Output:
        reports/tables/threshold_crisis_lead_lag_table.csv
    """
    data = _prepare_threshold_df(df)
    rows = []

    for market, windows in crisis_windows.items():
        market = str(market).upper()
        market_df = data[
            (data["market"].astype(str).str.upper() == market)
            & (data["threshold_regime_available"])
        ].copy()

        if market_df.empty:
            continue

        model_name = _single_or_first(market_df["threshold_model_name"])

        for start_date, end_date, crisis_name in windows:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            window_df = market_df[
                (market_df["date"] >= start_ts)
                & (market_df["date"] <= end_ts)
            ].sort_values("date")

            if len(window_df) == 0 and skip_windows_outside_sample:
                continue

            pre_window_df = market_df[
                (market_df["date"] >= start_ts - pd.Timedelta(days=21))
                & (market_df["date"] < start_ts)
            ].sort_values("date")

            stress_df = window_df[window_df["threshold_state"] == STRESS]
            first_stress_date = (
                stress_df["date"].min() if not stress_df.empty else pd.NaT
            )

            first_available_window_date = (
                window_df["date"].min().date().isoformat()
                if not window_df.empty
                else pd.NA
            )

            start_row = window_df[window_df["date"] == start_ts]
            if start_row.empty and not window_df.empty:
                start_row = window_df.iloc[[0]]

            stress_on_first_available = (
                bool((start_row["threshold_state"] == STRESS).iloc[0])
                if not start_row.empty
                else False
            )
            transition_or_stress_on_first_available = (
                bool(start_row["threshold_state"].isin([TRANSITION, STRESS]).iloc[0])
                if not start_row.empty
                else False
            )

            rows.append(
                {
                    "market": market,
                    "threshold_model_name": model_name,
                    "crisis_name": crisis_name,
                    "start_date": start_ts.date().isoformat(),
                    "end_date": end_ts.date().isoformat(),
                    "first_available_window_date": first_available_window_date,
                    "n_days": int(len(window_df)),
                    "first_stress_date": (
                        first_stress_date.date().isoformat()
                        if pd.notna(first_stress_date)
                        else pd.NA
                    ),
                    "days_from_window_start_to_first_stress": (
                        int((first_stress_date - start_ts).days)
                        if pd.notna(first_stress_date)
                        else pd.NA
                    ),
                    "max_threshold_stress_score": _safe_max(
                        window_df.get("threshold_stress_score")
                    ),
                    "stress_on_first_available_window_date": stress_on_first_available,
                    "transition_or_stress_on_first_available_window_date": transition_or_stress_on_first_available,
                    "pre_window_21d_stress_days": int(
                        (pre_window_df["threshold_state"] == STRESS).sum()
                    ),
                    "pre_window_21d_transition_or_stress_days": int(
                        pre_window_df["threshold_state"].isin([TRANSITION, STRESS]).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_threshold_vrp_by_state_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build VRP distribution table by final threshold state.

    Output:
        reports/tables/threshold_vrp_by_state.csv
    """
    data = _prepare_threshold_df(df)
    data = data[data["threshold_regime_available"]].copy()

    rows = []

    for vrp_col in ["vrp_har_gk", "vrp_backward_gk"]:
        if vrp_col not in data.columns:
            continue

        for keys, group in data.groupby(
            ["market", "threshold_model_name", "threshold_state", "threshold_state_name"],
            dropna=False,
        ):
            market, model_name, state, state_name = keys
            values = pd.to_numeric(group[vrp_col], errors="coerce").dropna()

            rows.append(
                {
                    "market": market,
                    "threshold_model_name": model_name,
                    "state": int(state),
                    "state_name": state_name,
                    "vrp_col": vrp_col,
                    "n_days": int(len(values)),
                    "mean": values.mean() if not values.empty else np.nan,
                    "median": values.median() if not values.empty else np.nan,
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "p05": values.quantile(0.05) if not values.empty else np.nan,
                    "p25": values.quantile(0.25) if not values.empty else np.nan,
                    "p75": values.quantile(0.75) if not values.empty else np.nan,
                    "p95": values.quantile(0.95) if not values.empty else np.nan,
                    "positive_ratio": (
                        float((values > 0).mean()) if not values.empty else np.nan
                    ),
                }
            )

    return _sort_state_table(pd.DataFrame(rows))


def build_threshold_forward_label_by_state_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build forward-label diagnostics after regime assignment.

    These labels are forbidden as construction inputs. They are allowed here only
    because threshold_state already exists.

    Output:
        reports/tables/threshold_forward_label_by_state.csv
    """
    data = _prepare_threshold_df(df)
    data = data[data["threshold_regime_available"]].copy()

    available_label_cols = [col for col in FORWARD_LABEL_COLUMNS if col in data.columns]

    if not available_label_cols:
        return pd.DataFrame(
            columns=[
                "market",
                "threshold_model_name",
                "state",
                "state_name",
                "n_days",
                "avg_forward_rv_label",
                "avg_forward_expost_vrp_label",
                "forward_vrp_positive_ratio",
                "forward_rv_p95",
            ]
        )

    rows = []

    for keys, group in data.groupby(
        ["market", "threshold_model_name", "threshold_state", "threshold_state_name"],
        dropna=False,
    ):
        market, model_name, state, state_name = keys

        rv = (
            pd.to_numeric(group["rv_gk_22d_forward_ann_label"], errors="coerce")
            if "rv_gk_22d_forward_ann_label" in group.columns
            else pd.Series(dtype="float64")
        )
        vrp = (
            pd.to_numeric(group["vrp_forward_expost_gk_label"], errors="coerce")
            if "vrp_forward_expost_gk_label" in group.columns
            else pd.Series(dtype="float64")
        )

        rows.append(
            {
                "market": market,
                "threshold_model_name": model_name,
                "state": int(state),
                "state_name": state_name,
                "n_days": int(len(group)),
                "avg_forward_rv_label": _safe_mean(rv),
                "avg_forward_expost_vrp_label": _safe_mean(vrp),
                "forward_vrp_positive_ratio": (
                    float((vrp.dropna() > 0).mean()) if not vrp.dropna().empty else np.nan
                ),
                "forward_rv_p95": (
                    rv.dropna().quantile(0.95) if not rv.dropna().empty else np.nan
                ),
            }
        )

    return _sort_state_table(pd.DataFrame(rows))


def build_threshold_no_lookahead_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level no-lookahead audit.

    This audit strengthens no-lookahead enforcement by explicitly logging which
    construction features are allowed, which diagnostic labels are present,
    and confirming that no forbidden columns were used.

    Output:
        reports/tables/threshold_no_lookahead_audit.csv
    """
    data = _prepare_threshold_df(df)

    forbidden_tokens = REGIME_FORBIDDEN_FEATURE_SUBSTRINGS
    allowed_construction_features = get_allowed_regime_features()
    allowed_diagnostic_labels = get_allowed_diagnostic_labels()

    construction_uses_forbidden = any(
        any(token in col.lower() for token in forbidden_tokens)
        for col in allowed_construction_features
    )

    rows = []

    for _, row in data.iterrows():
        regime_available = bool(row.get("threshold_regime_available", False))

        feature_available = bool(
            row.get(
                "threshold_primary_components_available",
                regime_available,
            )
        )

        rows.append(
            {
                "market": row.get("market", pd.NA),
                "date": row.get("date", pd.NaT),
                "threshold_model_name": row.get("threshold_model_name", pd.NA),
                "threshold_source": "strict_prior_rolling_thresholds",
                "uses_strict_prior_thresholds": True,
                "threshold_history_start_date": row.get(
                    "threshold_history_start_date",
                    pd.NaT,
                ),
                "threshold_history_end_date": row.get(
                    "threshold_history_end_date",
                    pd.NaT,
                ),
                "threshold_n_history": row.get("threshold_n_history", pd.NA),
                "feature_available": feature_available,
                "har_forecast_available": bool(row.get("har_forecast_available", False)),
                "construction_feature_cols": ";".join(sorted(allowed_construction_features)),
                "diagnostic_label_cols_available": ";".join(sorted(allowed_diagnostic_labels)),
                "uses_forbidden_columns": bool(construction_uses_forbidden),
                "forbidden_columns_used_for_construction": False,
                "regime_available": regime_available,
                "blocked_reason": row.get("threshold_blocked_reason", pd.NA),
            }
        )

    return pd.DataFrame(rows)


def write_threshold_metadata(
    config: Mapping[str, Any],
    output_path: str | Path,
    panels: Optional[Mapping[str, pd.DataFrame]] = None,
    input_paths: Optional[Mapping[str, str | Path]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write threshold metadata JSON.

    Output:
        reports/tables/threshold_regime_metadata.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "model_name": config.get("model_name"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256_json(config),
        "config_path": "configs/regime_threshold.yaml",
        "git_commit_if_available": _get_git_commit(),
        "score_method": config.get("combined_filter", {}).get("score_method"),
        "calm_score_cutoff": config.get("combined_filter", {}).get("calm_score_cutoff"),
        "stress_score_cutoff": config.get("combined_filter", {}).get(
            "stress_score_cutoff"
        ),
        "max_possible_score": _max_possible_score(config),
        "uses_manual_crisis_labels_for_training": config.get(
            "regime_learning_policy", {}
        ).get("uses_manual_crisis_labels_for_training"),
        "uses_crisis_windows_for_reporting_only": config.get(
            "regime_learning_policy", {}
        ).get("uses_crisis_windows_for_reporting_only"),
        "uses_forward_labels_for_training": config.get("regime_learning_policy", {}).get(
            "uses_forward_labels_for_training"
        ),
        "uses_forward_labels_for_reporting_only": config.get(
            "regime_learning_policy", {}
        ).get("uses_forward_labels_for_reporting_only"),
        "no_manual_state_overrides": config.get("regime_learning_policy", {}).get(
            "no_manual_state_overrides"
        ),
        "interpretation_note": (
            "The threshold baseline is intentionally conservative. Calm is rare because it requires "
            "simultaneous low IV, non-stress RV, non-stress drawdown, positive HAR-VRP compensation, "
            "and low aggregate stress score. This makes the filter useful as a risk-blocking benchmark, "
            "but not as a balanced clustering model. Stress and transition regimes are persistent; "
            "calm episodes are sparse and flickery."
        ),
    }

    input_file_metadata = {}

    if input_paths is not None:
        for market, path in input_paths.items():
            path_obj = Path(path)
            input_file_metadata[str(market)] = {
                "input_file_path": str(path_obj),
                "input_file_exists": path_obj.exists(),
                "input_file_modified_time": (
                    datetime.fromtimestamp(
                        path_obj.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat()
                    if path_obj.exists()
                    else None
                ),
            }

    panel_metadata = {}

    if panels is not None:
        for market, panel in panels.items():
            prepared = _prepare_threshold_df(panel)
            available = prepared[prepared["threshold_regime_available"]].copy()

            panel_metadata[str(market)] = {
                "input_row_count": int(len(prepared)),
                "output_columns": list(prepared.columns),
                "output_row_count": int(len(prepared)),
                "first_available_regime_date": (
                    available["date"].min().date().isoformat()
                    if not available.empty
                    else None
                ),
                "last_available_regime_date": (
                    available["date"].max().date().isoformat()
                    if not available.empty
                    else None
                ),
                "n_total_rows": int(len(prepared)),
                "n_available_regime_rows": int(len(available)),
                "available_fraction": (
                    float(len(available) / len(prepared)) if len(prepared) else np.nan
                ),
                "n_har_unavailable_rows": int(
                    (~prepared["har_forecast_available"].astype(bool)).sum()
                    if "har_forecast_available" in prepared.columns
                    else 0
                ),
                "n_required_component_missing_rows": int(
                    (~prepared["threshold_primary_components_available"].astype(bool)).sum()
                    if "threshold_primary_components_available" in prepared.columns
                    else 0
                ),
            }

    metadata["input_files"] = input_file_metadata
    metadata["panels"] = panel_metadata

    if extra:
        metadata.update(dict(extra))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)

    return metadata


def plot_threshold_regimes(
    df: pd.DataFrame,
    market: str,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot final threshold states over time.

    Output:
        reports/figures/threshold_regimes_{market}.png
    """
    data = _market_df(df, market)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(data["date"], data["iv_ann"], linewidth=1.0, label="iv_ann")
    ax1.plot(
        data["date"],
        data["rv_gk_22d_ann_lag1"],
        linewidth=1.0,
        label="rv_gk_22d_ann_lag1",
    )
    ax1.set_ylabel("Annualized variance / volatility proxy")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.step(
        data["date"],
        data["threshold_state"],
        where="post",
        linewidth=1.0,
        label="threshold_state",
    )
    ax2.set_yticks(STATE_ORDER)
    ax2.set_yticklabels(STATE_NAME_ORDER)
    ax2.set_ylabel("Threshold state")
    ax2.legend(loc="upper right")

    ax1.set_title(f"{market.upper()} threshold regimes")
    fig.tight_layout()

    _save_figure_if_requested(fig, output_path)

    return fig


def plot_threshold_regime_vrp_boxplots(
    df: pd.DataFrame,
    market: str,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot HAR-VRP distribution by final threshold state.

    Output:
        reports/figures/threshold_regime_vrp_boxplots_{market}.png
    """
    data = _market_df(df, market)
    data = data[data["threshold_regime_available"]].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = []
    labels = []

    for state in STATE_ORDER:
        values = pd.to_numeric(
            data.loc[data["threshold_state"] == state, "vrp_har_gk"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            box_data.append(values)
            labels.append(STATE_ID_TO_NAME[state])

    if box_data:
        ax.boxplot(box_data, labels=labels, showfliers=False)
    else:
        ax.text(0.5, 0.5, "No available VRP data", ha="center", va="center")

    ax.set_title(f"{market.upper()} HAR-VRP by threshold state")
    ax.set_ylabel("vrp_har_gk")
    fig.tight_layout()

    _save_figure_if_requested(fig, output_path)

    return fig


def plot_threshold_component_states(
    df: pd.DataFrame,
    market: str,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot all component states over time.

    Output:
        reports/figures/threshold_component_states_{market}.png
    """
    data = _market_df(df, market)

    fig, ax = plt.subplots(figsize=(12, 6))

    y_offsets = {
        "iv_percentile_state": 0,
        "rv_percentile_state": 4,
        "drawdown_state": 8,
        "iv_slope_state": 12,
        "vrp_har_state": 16,
    }

    for col, offset in y_offsets.items():
        if col not in data.columns:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        ax.step(
            data["date"],
            values + offset,
            where="post",
            linewidth=1.0,
            label=col,
        )

    ax.set_title(f"{market.upper()} threshold component states")
    ax.set_ylabel("Component state + offset")
    ax.legend(loc="upper left")
    fig.tight_layout()

    _save_figure_if_requested(fig, output_path)

    return fig


def _prepare_threshold_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        return df.copy()

    out = df.copy()

    if "date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("threshold DataFrame must contain 'date' column.")

    out["date"] = pd.to_datetime(out["date"], errors="raise")

    required = [
        "market",
        "threshold_model_name",
        "threshold_state",
        "threshold_state_name",
        "threshold_regime_available",
    ]
    missing = sorted(set(required) - set(out.columns))
    if missing:
        raise ValueError(f"threshold DataFrame missing required column(s): {missing}")

    out = out.sort_values(["market", "date"]).reset_index(drop=True)

    return out


def _empty_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market",
            "threshold_model_name",
            "state",
            "state_name",
            "n_days",
            "fraction_days",
            "avg_iv_ann",
            "avg_iv_close",
            "avg_rv_gk_22d_ann_lag1",
            "avg_vrp_backward_gk",
            "avg_vrp_har_gk",
            "avg_har_rv_gk_22d_forecast_ann",
            "vrp_har_positive_ratio",
            "avg_log_return",
            "avg_simple_return",
        ]
    )


def _sort_state_table(
    df: pd.DataFrame,
    extra_sort_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    sort_cols = []
    for col in ["market", "threshold_model_name"]:
        if col in out.columns:
            sort_cols.append(col)

    if extra_sort_cols:
        sort_cols.extend([col for col in extra_sort_cols if col in out.columns])

    if "state" in out.columns:
        out["_state_sort"] = pd.to_numeric(out["state"], errors="coerce")
        sort_cols.append("_state_sort")

    out = out.sort_values(sort_cols).drop(columns=["_state_sort"], errors="ignore")
    return out.reset_index(drop=True)


def _availability_col(component_state_col: str) -> str:
    return component_state_col.removesuffix("_state") + "_available"


def _blocked_reason_col(component_state_col: str) -> str:
    return component_state_col.removesuffix("_state") + "_blocked_reason"


def _safe_mean(series: Optional[pd.Series]) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _safe_max(series: Optional[pd.Series]) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def _positive_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float((values > 0).mean()) if not values.empty else np.nan


def _single_or_first(series: pd.Series) -> Any:
    values = series.dropna().unique().tolist()
    return values[0] if values else pd.NA


def _market_df(df: pd.DataFrame, market: str) -> pd.DataFrame:
    data = _prepare_threshold_df(df)
    market = str(market).upper()

    out = data[data["market"].astype(str).str.upper() == market].copy()

    if out.empty:
        raise ValueError(f"No rows found for market {market!r}.")

    return out.sort_values("date").reset_index(drop=True)


def _save_figure_if_requested(fig: plt.Figure, output_path: Optional[str | Path]) -> None:
    if output_path is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def _sha256_json(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, set)) else False:
        return None
    return obj


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _max_possible_score(config: Mapping[str, Any]) -> Optional[float]:
    try:
        weights = config["combined_filter"]["component_weights"]
        return float(sum(float(weight) * STRESS for weight in weights.values()))
    except Exception:
        return None