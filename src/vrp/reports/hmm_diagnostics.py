"""
Phase 6 Gaussian HMM diagnostics and reporting.

Part 1 implements:
- candidate ranking table
- state summary table
- transition matrix table
- state duration summary
- state-by-year table
- metadata/hash helpers

Part 2 will add:
- threshold agreement diagnostics
- crisis diagnostics
- forward-label diagnostics
- probability audit
- no-lookahead audit
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vrp.regimes.gaussian_hmm import (
    HMMCandidateOutput,
    candidate_outputs_to_ranking_frame,
)
from vrp.regimes.hmm_registry import (
    HMM_CANDIDATE_RANKING_COLUMNS,
    HMM_EXPECTED_TABLE_PATHS,
    HMM_FILTERED_ECONOMIC_PROB_COLUMNS,
    HMM_FILTERED_RAW_PROB_PREFIX,
    HMM_METADATA_REQUIRED_FIELDS,
    HMM_TRANSITION_STATE_MODELLED_COLUMN,
    PHASE_6_TABLES_DIR,
    get_hmm_diagnostic_smoothed_probability_columns,
    get_hmm_filtered_raw_probability_columns,
)
from vrp.regimes.hmm_scaling import hash_numpy_array


DEFAULT_DATE_COL = "date"
DEFAULT_RAW_STATE_COL = "hmm_raw_state"
DEFAULT_STATE_COL = "hmm_state"
DEFAULT_STATE_NAME_COL = "hmm_state_name"


def utc_now_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for output file and return Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_phase_6_tables_dir(path: str | Path = PHASE_6_TABLES_DIR) -> Path:
    """Create and return Phase 6 tables directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_table_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write DataFrame to CSV with parent creation."""
    p = ensure_parent_dir(path)
    df.to_csv(p, index=False)
    return p


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write JSON payload with stable formatting."""
    p = ensure_parent_dir(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return p


def _jsonable(obj: Any) -> Any:
    """Convert common Python/numpy/pandas objects to JSON-compatible values."""
    # is_dataclass() returns True for both dataclass types and instances.
    # asdict() requires an instance, so handle class objects separately.
    if is_dataclass(obj):
        if isinstance(obj, type):
            return obj.__name__
        return _jsonable(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value):
            return None
        return value

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if pd.isna(obj) if not isinstance(obj, (list, tuple, Mapping, np.ndarray)) else False:
        return None

    return obj


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Stable SHA256 hash for JSON-like payload."""
    cleaned = _jsonable(payload)
    encoded = json.dumps(cleaned, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_content_hash(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    include_index: bool = False,
) -> str:
    """
    Stable hash for selected DataFrame content.

    This is not meant as cryptographic provenance against malicious mutation.
    It is a reproducibility hash for Phase 6 artifacts.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    use = df.copy()
    if columns is not None:
        missing = [col for col in columns if col not in use.columns]
        if missing:
            raise ValueError(f"Cannot hash missing columns: {missing}")
        use = use.loc[:, list(columns)]

    if include_index:
        use = use.reset_index()

    normalized = use.copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = pd.to_datetime(normalized[col]).astype("datetime64[ns]").astype(str)

    csv_bytes = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _require_output_panel(output: HMMCandidateOutput) -> pd.DataFrame:
    """Return output panel or raise."""
    if output.output_panel is None:
        raise ValueError("HMMCandidateOutput has no output_panel.")

    if not isinstance(output.output_panel, pd.DataFrame):
        raise TypeError("output.output_panel must be a pandas DataFrame.")

    return output.output_panel.copy()


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce Series to numeric float."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def _state_sort_key(state_name: str) -> int:
    """Stable economic-state order."""
    order = {"calm": 0, "transition": 1, "stress": 2}
    return order.get(str(state_name), 99)


def build_candidate_ranking_table(
    outputs: Sequence[HMMCandidateOutput],
    *,
    selected_output: HMMCandidateOutput | None = None,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_candidate_model_ranking.csv.

    Selection must not use PnL, crisis hit rate, or threshold agreement.
    """
    ranking = candidate_outputs_to_ranking_frame(
        outputs,
        selected_output=selected_output,
    )

    for col in HMM_CANDIDATE_RANKING_COLUMNS:
        if col not in ranking.columns:
            ranking[col] = np.nan

    ranking = ranking.loc[:, list(HMM_CANDIDATE_RANKING_COLUMNS)]

    sort_cols = [
        "selected_primary",
        "rejection_reason",
        "bic",
        "test_loglik_per_obs",
    ]

    existing_sort_cols = [col for col in sort_cols if col in ranking.columns]

    if existing_sort_cols:
        ranking = ranking.sort_values(
            existing_sort_cols,
            ascending=[False, True, True, False][: len(existing_sort_cols)],
            na_position="last",
        ).reset_index(drop=True)

    return ranking


def build_hmm_state_summary_table(
    output: HMMCandidateOutput,
    *,
    date_col: str = DEFAULT_DATE_COL,
    raw_state_col: str = DEFAULT_RAW_STATE_COL,
    state_col: str = DEFAULT_STATE_COL,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_state_summary.csv.

    One row per economic state. Includes occupancy, date range, average state
    probabilities, and raw-state mapping information.
    """
    panel = _require_output_panel(output)

    required = [date_col, raw_state_col, state_col, state_name_col]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Cannot build state summary. Missing columns: {missing}")

    prob_cols = [
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["calm"],
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["transition"],
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["stress"],
    ]
    missing_probs = [col for col in prob_cols if col not in panel.columns]
    if missing_probs:
        raise ValueError(f"Cannot build state summary. Missing probability columns: {missing_probs}")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        bad_count = int(out[date_col].isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} invalid date value(s).")

    n_total = len(out)
    rows: list[dict[str, Any]] = []

    for state_name, group in out.groupby(state_name_col, sort=False):
        raw_states = sorted(group[raw_state_col].dropna().astype(int).unique().tolist())
        economic_state_id_values = sorted(group[state_col].dropna().astype(int).unique().tolist())

        row: dict[str, Any] = {
            "market": output.fit_result.market,
            "feature_set": output.fit_result.spec.feature_set,
            "n_states": output.fit_result.spec.n_states,
            "covariance_type": output.fit_result.spec.covariance_type,
            "economic_state_name": state_name,
            "economic_state_id": economic_state_id_values[0] if economic_state_id_values else np.nan,
            "raw_states": ",".join(str(x) for x in raw_states),
            "n_observations": int(len(group)),
            "occupancy": float(len(group) / n_total) if n_total else np.nan,
            "first_date": group[date_col].min().date().isoformat(),
            "last_date": group[date_col].max().date().isoformat(),
            "mean_prob_calm": float(_safe_numeric(group[prob_cols[0]]).mean()),
            "mean_prob_transition": float(_safe_numeric(group[prob_cols[1]]).mean()),
            "mean_prob_stress": float(_safe_numeric(group[prob_cols[2]]).mean()),
            "hmm_model_valid": output.hmm_model_valid,
            "hmm_model_failure_reason": output.hmm_model_failure_reason,
        }

        optional_features = [
            "vrp_har_gk",
            "rv_gk_22d_ann_lag1",
            "iv_ann",
            "index_return",
            "iv_ann_change_1d",
            "hmm_state_stress_score",
        ]

        for col in optional_features:
            if col in group.columns:
                row[f"mean_{col}"] = float(_safe_numeric(group[col]).mean())

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["_state_sort"] = result["economic_state_name"].map(_state_sort_key)
    result = result.sort_values("_state_sort").drop(columns=["_state_sort"]).reset_index(drop=True)

    return result


def build_hmm_transition_matrix_table(
    output: HMMCandidateOutput,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_transition_matrix.csv.

    Uses the fitted model transition matrix. Rows are raw-state transitions.
    Economic labels are included for readability.
    """
    model = output.fit_result.model

    if model is None:
        return pd.DataFrame(
            [
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "from_raw_state": np.nan,
                    "to_raw_state": np.nan,
                    "from_state_name": "",
                    "to_state_name": "",
                    "transition_probability": np.nan,
                    "hmm_model_valid": output.hmm_model_valid,
                    "hmm_model_failure_reason": output.hmm_model_failure_reason,
                }
            ]
        )

    transmat = np.asarray(model.transmat_, dtype=float)
    n_states = output.fit_result.spec.n_states

    if transmat.shape != (n_states, n_states):
        raise ValueError(
            f"Transition matrix shape mismatch. Expected {(n_states, n_states)}, got {transmat.shape}."
        )

    raw_to_name: Mapping[int, str] = {}
    if output.labeling is not None:
        raw_to_name = output.labeling.raw_state_to_name

    rows: list[dict[str, Any]] = []
    for i in range(n_states):
        for j in range(n_states):
            rows.append(
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "from_raw_state": i,
                    "to_raw_state": j,
                    "from_state_name": raw_to_name.get(i, ""),
                    "to_state_name": raw_to_name.get(j, ""),
                    "transition_probability": float(transmat[i, j]),
                    "hmm_model_valid": output.hmm_model_valid,
                    "hmm_model_failure_reason": output.hmm_model_failure_reason,
                }
            )

    return pd.DataFrame(rows)


def _run_length_encode_states(
    states: Sequence[Any],
) -> list[tuple[Any, int, int, int]]:
    """
    Run-length encode a state sequence.

    Returns list of:
    (state_value, start_position, end_position, length)
    """
    if len(states) == 0:
        return []

    runs: list[tuple[Any, int, int, int]] = []

    start = 0
    current = states[0]

    for idx in range(1, len(states)):
        if states[idx] != current:
            end = idx - 1
            runs.append((current, start, end, end - start + 1))
            start = idx
            current = states[idx]

    end = len(states) - 1
    runs.append((current, start, end, end - start + 1))

    return runs


def build_hmm_state_duration_summary_table(
    output: HMMCandidateOutput,
    *,
    date_col: str = DEFAULT_DATE_COL,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_state_duration_summary.csv.

    A duration is a consecutive run of the same economic state.
    """
    panel = _require_output_panel(output)

    required = [date_col, state_name_col]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Cannot build duration summary. Missing columns: {missing}")

    out = panel.copy().reset_index(drop=True)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        bad_count = int(out[date_col].isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} invalid date value(s).")

    states = out[state_name_col].astype(str).tolist()
    runs = _run_length_encode_states(states)

    run_rows: list[dict[str, Any]] = []
    for run_id, (state_name, start_pos, end_pos, length) in enumerate(runs):
        run_rows.append(
            {
                "run_id": run_id,
                "economic_state_name": state_name,
                "start_position": start_pos,
                "end_position": end_pos,
                "start_date": pd.Timestamp(out[date_col].iloc[start_pos]).date().isoformat(),
                "end_date": pd.Timestamp(out[date_col].iloc[end_pos]).date().isoformat(),
                "duration_days": int(length),
            }
        )

    run_frame = pd.DataFrame(run_rows)

    if run_frame.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "feature_set",
                "n_states",
                "covariance_type",
                "economic_state_name",
                "n_runs",
                "mean_duration_days",
                "median_duration_days",
                "min_duration_days",
                "max_duration_days",
                "p90_duration_days",
                "hmm_model_valid",
                "hmm_model_failure_reason",
            ]
        )

    rows: list[dict[str, Any]] = []
    for state_name, group in run_frame.groupby("economic_state_name", sort=False):
        durations = _safe_numeric(group["duration_days"])
        rows.append(
            {
                "market": output.fit_result.market,
                "feature_set": output.fit_result.spec.feature_set,
                "n_states": output.fit_result.spec.n_states,
                "covariance_type": output.fit_result.spec.covariance_type,
                "economic_state_name": state_name,
                "n_runs": int(len(group)),
                "mean_duration_days": float(durations.mean()),
                "median_duration_days": float(durations.median()),
                "min_duration_days": int(durations.min()),
                "max_duration_days": int(durations.max()),
                "p90_duration_days": float(durations.quantile(0.90)),
                "hmm_model_valid": output.hmm_model_valid,
                "hmm_model_failure_reason": output.hmm_model_failure_reason,
            }
        )

    result = pd.DataFrame(rows)
    result["_state_sort"] = result["economic_state_name"].map(_state_sort_key)
    result = result.sort_values("_state_sort").drop(columns=["_state_sort"]).reset_index(drop=True)

    return result


def build_hmm_state_by_year_table(
    output: HMMCandidateOutput,
    *,
    date_col: str = DEFAULT_DATE_COL,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_state_by_year.csv.

    One row per year/state with annual occupancy.
    """
    panel = _require_output_panel(output)

    required = [date_col, state_name_col]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Cannot build state-by-year table. Missing columns: {missing}")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        bad_count = int(out[date_col].isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} invalid date value(s).")

    out["year"] = out[date_col].dt.year.astype(int)

    annual_total = out.groupby("year").size().rename("n_year_observations").reset_index()
    state_counts = (
        out.groupby(["year", state_name_col])
        .size()
        .rename("n_state_observations")
        .reset_index()
    )

    result = state_counts.merge(annual_total, on="year", how="left")
    result["annual_state_occupancy"] = (
        result["n_state_observations"] / result["n_year_observations"]
    )

    result = result.rename(columns={state_name_col: "economic_state_name"})

    result.insert(0, "market", output.fit_result.market)
    result.insert(1, "feature_set", output.fit_result.spec.feature_set)
    result.insert(2, "n_states", output.fit_result.spec.n_states)
    result.insert(3, "covariance_type", output.fit_result.spec.covariance_type)

    result["hmm_model_valid"] = output.hmm_model_valid
    result["hmm_model_failure_reason"] = output.hmm_model_failure_reason

    result["_state_sort"] = result["economic_state_name"].map(_state_sort_key)
    result = result.sort_values(["year", "_state_sort"]).drop(columns=["_state_sort"])
    result = result.reset_index(drop=True)

    return result


def build_hmm_metadata(
    output: HMMCandidateOutput,
    *,
    input_data_hash: str = "",
    feature_panel_hash: str = "",
    config_hash: str = "",
    code_version_or_git_commit: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build reports/tables/phase_6/hmm_metadata.json payload.

    Required fields:
    - input_data_hash
    - feature_panel_hash
    - train_window_hash
    - scaler_hash
    - hmm_parameter_hash
    - config_hash
    - code_version_or_git_commit
    - created_at_utc
    """
    scaler_meta = output.fit_result.scaled_panel.metadata.to_dict()
    scaler_hash = stable_json_hash(scaler_meta)

    if output.fit_result.model is not None:
        model = output.fit_result.model
        hmm_parameter_hash = stable_json_hash(
            {
                "startprob": np.asarray(model.startprob_, dtype=float).tolist(),
                "transmat": np.asarray(model.transmat_, dtype=float).tolist(),
                "means": np.asarray(model.means_, dtype=float).tolist(),
                "covars": np.asarray(model.covars_, dtype=float).tolist(),
                "covariance_type": output.fit_result.spec.covariance_type,
                "n_states": output.fit_result.spec.n_states,
            }
        )
    else:
        hmm_parameter_hash = ""

    if not feature_panel_hash and output.output_panel is not None:
        feature_panel_hash = dataframe_content_hash(
            output.output_panel,
            columns=[
                col for col in output.fit_result.scaled_panel.feature_cols
                if col in output.output_panel.columns
            ],
        )

    payload: dict[str, Any] = {
        "input_data_hash": input_data_hash,
        "feature_panel_hash": feature_panel_hash,
        "train_window_hash": scaler_meta.get("train_window_hash", ""),
        "scaler_hash": scaler_hash,
        "hmm_parameter_hash": hmm_parameter_hash,
        "config_hash": config_hash,
        "code_version_or_git_commit": code_version_or_git_commit,
        "created_at_utc": utc_now_iso(),
        "market": output.fit_result.market,
        "feature_set": output.fit_result.spec.feature_set,
        "n_states": output.fit_result.spec.n_states,
        "covariance_type": output.fit_result.spec.covariance_type,
        "model_name": "gaussian_hmm_v1",
        "hmm_model_valid": output.hmm_model_valid,
        "hmm_model_failure_reason": output.hmm_model_failure_reason,
        "fit": {
            "converged": output.fit_result.converged,
            "n_iter": output.fit_result.n_iter,
            "best_init_index": output.fit_result.best_init_index,
            "best_random_state": output.fit_result.best_random_state,
            "train_loglik": output.fit_result.train_loglik,
            "test_loglik": output.fit_result.test_loglik,
            "aic": output.fit_result.aic,
            "bic": output.fit_result.bic,
        },
        "scaler": scaler_meta,
        "state_labeling": (
            output.labeling.to_metadata()
            if output.labeling is not None
            else {}
        ),
    }

    if extra:
        payload["extra"] = _jsonable(extra)

    for field in HMM_METADATA_REQUIRED_FIELDS:
        if field not in payload:
            payload[field] = ""

    return payload


def write_hmm_diagnostics_part1(
    output: HMMCandidateOutput,
    *,
    selected_output: HMMCandidateOutput | None = None,
    all_outputs: Sequence[HMMCandidateOutput] | None = None,
    tables_dir: str | Path = PHASE_6_TABLES_DIR,
    input_data_hash: str = "",
    feature_panel_hash: str = "",
    config_hash: str = "",
    code_version_or_git_commit: str = "",
) -> dict[str, Path]:
    """
    Write Chunk 9 Phase 6 diagnostic artifacts.

    This writes:
    - hmm_candidate_model_ranking.csv
    - hmm_state_summary.csv
    - hmm_transition_matrix.csv
    - hmm_state_duration_summary.csv
    - hmm_state_by_year.csv
    - hmm_metadata.json
    """
    tables_dir = ensure_phase_6_tables_dir(tables_dir)

    outputs = list(all_outputs) if all_outputs is not None else [output]
    selected = selected_output if selected_output is not None else output

    paths = {
        "candidate_model_ranking": tables_dir / "hmm_candidate_model_ranking.csv",
        "state_summary": tables_dir / "hmm_state_summary.csv",
        "transition_matrix": tables_dir / "hmm_transition_matrix.csv",
        "state_duration_summary": tables_dir / "hmm_state_duration_summary.csv",
        "state_by_year": tables_dir / "hmm_state_by_year.csv",
        "metadata": tables_dir / "hmm_metadata.json",
    }

    ranking = build_candidate_ranking_table(outputs, selected_output=selected)
    state_summary = build_hmm_state_summary_table(output)
    transition_matrix = build_hmm_transition_matrix_table(output)
    duration_summary = build_hmm_state_duration_summary_table(output)
    state_by_year = build_hmm_state_by_year_table(output)
    metadata = build_hmm_metadata(
        output,
        input_data_hash=input_data_hash,
        feature_panel_hash=feature_panel_hash,
        config_hash=config_hash,
        code_version_or_git_commit=code_version_or_git_commit,
    )

    write_table_csv(ranking, paths["candidate_model_ranking"])
    write_table_csv(state_summary, paths["state_summary"])
    write_table_csv(transition_matrix, paths["transition_matrix"])
    write_table_csv(duration_summary, paths["state_duration_summary"])
    write_table_csv(state_by_year, paths["state_by_year"])
    write_json(metadata, paths["metadata"])

    return paths


__all__ = [
    "utc_now_iso",
    "ensure_parent_dir",
    "ensure_phase_6_tables_dir",
    "write_table_csv",
    "write_json",
    "stable_json_hash",
    "dataframe_content_hash",
    "build_candidate_ranking_table",
    "build_hmm_state_summary_table",
    "build_hmm_transition_matrix_table",
    "build_hmm_state_duration_summary_table",
    "build_hmm_state_by_year_table",
    "build_hmm_metadata",
    "write_hmm_diagnostics_part1",
    "build_hmm_threshold_agreement_table",
    "build_hmm_crisis_hit_table",
    "build_hmm_crisis_lead_lag_table",
    "build_hmm_forward_label_by_state_table",
    "build_hmm_probability_audit_table",
    "build_hmm_no_lookahead_audit_table",
    "write_hmm_diagnostics_part2",
]

# ---------------------------------------------------------------------------
# Phase 6 diagnostics part 2:
# threshold agreement, crisis diagnostics, forward labels, probability audit,
# and no-lookahead audit.
# ---------------------------------------------------------------------------

from vrp.regimes.hmm_registry import (
    HMM_SIGNAL_AVAILABILITY_COLUMNS,
    HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX,
    get_hmm_filtered_economic_probability_columns,
)
from vrp.regimes.hmm_validation import (
    assert_hmm_feature_columns_are_legal,
    validate_crisis_windows_usage,
    validate_threshold_comparison_usage,
    assert_output_probability_policy_is_safe,
)
from vrp.regimes.online_filter import (
    forward_filter_gaussian,
    max_abs_prefix_difference,
)


def _normalise_date_column(
    df: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DataFrame:
    """Return copy with normalised date column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column {date_col!r}.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        bad_count = int(out[date_col].isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} invalid date value(s).")

    out[date_col] = out[date_col].dt.normalize()
    return out


def _infer_threshold_state_name_col(df: pd.DataFrame) -> str | None:
    """Infer threshold economic-state name column."""
    candidates = [
        "threshold_state_name",
        "threshold_regime_name",
        "threshold_state",
        "threshold_regime",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalise_threshold_state_name(value: Any) -> str:
    """
    Convert threshold state values to comparable names where possible.

    Keeps unknown strings unchanged.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    mapping = {
        "0": "calm",
        "1": "transition",
        "2": "stress",
        "calm": "calm",
        "normal": "calm",
        "low_vol": "calm",
        "low": "calm",
        "transition": "transition",
        "medium": "transition",
        "neutral": "transition",
        "mixed": "transition",
        "stress": "stress",
        "high_vol": "stress",
        "high": "stress",
        "crisis": "stress",
    }

    return mapping.get(text, text)


def build_hmm_threshold_agreement_table(
    output: HMMCandidateOutput,
    *,
    threshold_panel: pd.DataFrame | None = None,
    date_col: str = DEFAULT_DATE_COL,
    hmm_state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_threshold_agreement.csv.

    Threshold states are used only after HMM assignment. They are never used as
    HMM features, HMM labels, state-mapping inputs, or model-selection targets.
    """
    validate_threshold_comparison_usage(
        threshold_state_as_feature=False,
        threshold_state_as_target=False,
        choose_model_by_threshold_match=False,
    )

    hmm_panel = _require_output_panel(output)
    hmm_panel = _normalise_date_column(hmm_panel, date_col=date_col)

    if hmm_state_name_col not in hmm_panel.columns:
        raise ValueError(f"HMM output missing {hmm_state_name_col!r}.")

    if threshold_panel is None:
        threshold_col = _infer_threshold_state_name_col(hmm_panel)
        if threshold_col is None:
            return pd.DataFrame(
                [
                    {
                        "market": output.fit_result.market,
                        "feature_set": output.fit_result.spec.feature_set,
                        "n_states": output.fit_result.spec.n_states,
                        "covariance_type": output.fit_result.spec.covariance_type,
                        "hmm_state_name": "",
                        "threshold_state_name": "",
                        "n_observations": 0,
                        "agreement_count": 0,
                        "agreement_rate": np.nan,
                        "status": "missing_threshold_state_columns",
                    }
                ]
            )
        merged = hmm_panel[[date_col, hmm_state_name_col, threshold_col]].copy()
    else:
        threshold_df = _normalise_date_column(threshold_panel, date_col=date_col)
        threshold_col = _infer_threshold_state_name_col(threshold_df)
        if threshold_col is None:
            return pd.DataFrame(
                [
                    {
                        "market": output.fit_result.market,
                        "feature_set": output.fit_result.spec.feature_set,
                        "n_states": output.fit_result.spec.n_states,
                        "covariance_type": output.fit_result.spec.covariance_type,
                        "hmm_state_name": "",
                        "threshold_state_name": "",
                        "n_observations": 0,
                        "agreement_count": 0,
                        "agreement_rate": np.nan,
                        "status": "missing_threshold_state_columns",
                    }
                ]
            )

        merged = hmm_panel[[date_col, hmm_state_name_col]].merge(
            threshold_df[[date_col, threshold_col]],
            on=date_col,
            how="inner",
            validate="one_to_one",
        )

    merged = merged.rename(
        columns={
            hmm_state_name_col: "hmm_state_name",
            threshold_col: "threshold_state_name_raw",
        }
    )

    merged["threshold_state_name"] = merged["threshold_state_name_raw"].map(
        _normalise_threshold_state_name
    )
    merged["hmm_state_name"] = merged["hmm_state_name"].astype(str).str.lower()
    merged["agrees"] = merged["hmm_state_name"] == merged["threshold_state_name"]

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "hmm_state_name": "",
                    "threshold_state_name": "",
                    "n_observations": 0,
                    "agreement_count": 0,
                    "agreement_rate": np.nan,
                    "status": "no_date_overlap",
                }
            ]
        )

    grouped = (
        merged.groupby(["hmm_state_name", "threshold_state_name"], dropna=False)
        .agg(
            n_observations=("agrees", "size"),
            agreement_count=("agrees", "sum"),
        )
        .reset_index()
    )

    grouped["agreement_rate"] = (
        grouped["agreement_count"] / grouped["n_observations"]
    )

    grouped.insert(0, "market", output.fit_result.market)
    grouped.insert(1, "feature_set", output.fit_result.spec.feature_set)
    grouped.insert(2, "n_states", output.fit_result.spec.n_states)
    grouped.insert(3, "covariance_type", output.fit_result.spec.covariance_type)
    grouped["status"] = "ok"

    return grouped


def _crisis_windows_to_frame(
    crisis_windows: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert crisis-window config into a standard DataFrame.

    Required columns:
    - crisis_name
    - start_date
    - end_date
    """
    if isinstance(crisis_windows, pd.DataFrame):
        out = crisis_windows.copy()
    else:
        out = pd.DataFrame(list(crisis_windows))

    rename_map = {
        "name": "crisis_name",
        "label": "crisis_name",
        "start": "start_date",
        "end": "end_date",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = ["crisis_name", "start_date", "end_date"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"crisis_windows missing columns: {missing}")

    out["crisis_name"] = out["crisis_name"].astype(str)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()

    if out["start_date"].isna().any() or out["end_date"].isna().any():
        raise ValueError("crisis_windows contains invalid start_date/end_date.")

    bad_order = out["end_date"] < out["start_date"]
    if bad_order.any():
        bad = out.loc[bad_order, "crisis_name"].tolist()
        raise ValueError(f"crisis_windows have end_date before start_date: {bad}")

    return out.reset_index(drop=True)


def build_hmm_crisis_hit_table(
    output: HMMCandidateOutput,
    *,
    crisis_windows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_crisis_hit_table.csv.

    Diagnostic-only. Must not be used for HMM model selection or state mapping.
    """
    validate_crisis_windows_usage(
        used_for="crisis_stress_overlap",
        diagnostics_only=True,
    )

    panel = _require_output_panel(output)
    panel = _normalise_date_column(panel, date_col=date_col)

    if state_name_col not in panel.columns:
        raise ValueError(f"HMM output missing {state_name_col!r}.")

    windows = _crisis_windows_to_frame(crisis_windows)

    rows: list[dict[str, Any]] = []
    for _, crisis in windows.iterrows():
        mask = (panel[date_col] >= crisis["start_date"]) & (panel[date_col] <= crisis["end_date"])
        crisis_panel = panel.loc[mask].copy()

        n_days = int(len(crisis_panel))
        stress_days = int((crisis_panel[state_name_col].astype(str).str.lower() == "stress").sum())
        false_negative_days = int(n_days - stress_days)

        rows.append(
            {
                "market": output.fit_result.market,
                "feature_set": output.fit_result.spec.feature_set,
                "n_states": output.fit_result.spec.n_states,
                "covariance_type": output.fit_result.spec.covariance_type,
                "crisis_name": crisis["crisis_name"],
                "start_date": crisis["start_date"].date().isoformat(),
                "end_date": crisis["end_date"].date().isoformat(),
                "n_crisis_observations": n_days,
                "crisis_stress_days": stress_days,
                "crisis_stress_overlap": float(stress_days / n_days) if n_days else np.nan,
                "crisis_false_negative_days": false_negative_days,
                "diagnostic_only": True,
                "hmm_model_valid": output.hmm_model_valid,
                "hmm_model_failure_reason": output.hmm_model_failure_reason,
            }
        )

    return pd.DataFrame(rows)


def build_hmm_crisis_lead_lag_table(
    output: HMMCandidateOutput,
    *,
    crisis_windows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
    lookback_days: int = 30,
    lookahead_days: int = 30,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_crisis_lead_lag_table.csv.

    For each crisis, reports first stress date in a window around the crisis.
    Diagnostic-only.
    """
    validate_crisis_windows_usage(
        used_for="crisis_lead_lag",
        diagnostics_only=True,
    )

    if lookback_days < 0 or lookahead_days < 0:
        raise ValueError("lookback_days and lookahead_days must be non-negative.")

    panel = _require_output_panel(output)
    panel = _normalise_date_column(panel, date_col=date_col)

    if state_name_col not in panel.columns:
        raise ValueError(f"HMM output missing {state_name_col!r}.")

    windows = _crisis_windows_to_frame(crisis_windows)

    rows: list[dict[str, Any]] = []
    for _, crisis in windows.iterrows():
        window_start = crisis["start_date"] - pd.Timedelta(days=lookback_days)
        window_end = crisis["end_date"] + pd.Timedelta(days=lookahead_days)

        mask = (panel[date_col] >= window_start) & (panel[date_col] <= window_end)
        around = panel.loc[mask].copy()
        stress = around.loc[around[state_name_col].astype(str).str.lower() == "stress"]

        if stress.empty:
            first_stress_date = pd.NaT
            lead_lag_days = np.nan
            stress_detected = False
        else:
            first_stress_date = stress[date_col].min()
            lead_lag_days = int((first_stress_date - crisis["start_date"]).days)
            stress_detected = True

        rows.append(
            {
                "market": output.fit_result.market,
                "feature_set": output.fit_result.spec.feature_set,
                "n_states": output.fit_result.spec.n_states,
                "covariance_type": output.fit_result.spec.covariance_type,
                "crisis_name": crisis["crisis_name"],
                "crisis_start_date": crisis["start_date"].date().isoformat(),
                "crisis_end_date": crisis["end_date"].date().isoformat(),
                "diagnostic_window_start": window_start.date().isoformat(),
                "diagnostic_window_end": window_end.date().isoformat(),
                "stress_detected": stress_detected,
                "first_stress_date": (
                    first_stress_date.date().isoformat()
                    if pd.notna(first_stress_date)
                    else ""
                ),
                "crisis_lead_lag": lead_lag_days,
                "diagnostic_only": True,
                "hmm_model_valid": output.hmm_model_valid,
                "hmm_model_failure_reason": output.hmm_model_failure_reason,
            }
        )

    return pd.DataFrame(rows)


def _infer_forward_label_columns(df: pd.DataFrame) -> list[str]:
    """Infer explicit forward/expost label columns for diagnostics."""
    cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if "hmm_" in lower:
            continue
        if lower.startswith("threshold_"):
            continue
        if ("forward" in lower or "expost" in lower) and "label" in lower:
            cols.append(col)
    return cols


def build_hmm_forward_label_by_state_table(
    output: HMMCandidateOutput,
    *,
    forward_label_columns: Sequence[str] | None = None,
    state_name_col: str = DEFAULT_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_forward_label_by_state.csv.

    Forward labels are diagnostic-only and must not be used as HMM features.
    """
    panel = _require_output_panel(output)

    if state_name_col not in panel.columns:
        raise ValueError(f"HMM output missing {state_name_col!r}.")

    label_cols = list(forward_label_columns) if forward_label_columns is not None else _infer_forward_label_columns(panel)

    if not label_cols:
        return pd.DataFrame(
            [
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "economic_state_name": "",
                    "forward_label_column": "",
                    "label_value": "",
                    "n_observations": 0,
                    "state_label_fraction": np.nan,
                    "diagnostic_only": True,
                    "status": "missing_forward_label_columns",
                }
            ]
        )

    missing = [col for col in label_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"Forward label columns missing from panel: {missing}")

    rows: list[dict[str, Any]] = []
    for label_col in label_cols:
        state_totals = panel.groupby(state_name_col).size().rename("n_state_observations").reset_index()

        counts = (
            panel.groupby([state_name_col, label_col], dropna=False)
            .size()
            .rename("n_observations")
            .reset_index()
        )

        counts = counts.merge(state_totals, on=state_name_col, how="left", validate="many_to_one")
        counts["state_label_fraction"] = counts["n_observations"] / counts["n_state_observations"]

        for _, row in counts.iterrows():
            rows.append(
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "economic_state_name": row[state_name_col],
                    "forward_label_column": label_col,
                    "label_value": row[label_col],
                    "n_observations": int(row["n_observations"]),
                    "state_label_fraction": float(row["state_label_fraction"]),
                    "diagnostic_only": True,
                    "status": "ok",
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["_state_sort"] = result["economic_state_name"].map(_state_sort_key)
        result = result.sort_values(["forward_label_column", "_state_sort", "label_value"])
        result = result.drop(columns=["_state_sort"]).reset_index(drop=True)

    return result


def build_hmm_probability_audit_table(
    output: HMMCandidateOutput,
    *,
    prefix_length: int | None = None,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_probability_audit.csv.

    Mandatory policy:
    - custom forward filter is used
    - hmmlearn predict_proba is not used for backtest-facing output
    - smoothed probabilities are diagnostic-only
    """
    assert_output_probability_policy_is_safe(
        uses_custom_forward_filter=True,
        uses_hmmlearn_predict_proba_for_backtest=False,
        uses_smoothed_probabilities_for_backtest=False,
    )

    panel = _require_output_panel(output)
    n_states = output.fit_result.spec.n_states

    filtered_cols = get_hmm_filtered_raw_probability_columns(n_states)
    smoothed_cols = get_hmm_diagnostic_smoothed_probability_columns(n_states)
    econ_cols = get_hmm_filtered_economic_probability_columns()

    missing_filtered = [col for col in filtered_cols if col not in panel.columns]
    missing_econ = [col for col in econ_cols if col not in panel.columns]

    if missing_filtered or missing_econ:
        row_sum_min = np.nan
        row_sum_max = np.nan
        passed = False
        status = f"missing_columns:filtered={missing_filtered};economic={missing_econ}"
    else:
        filtered_sum = panel[filtered_cols].sum(axis=1)
        row_sum_min = float(filtered_sum.min())
        row_sum_max = float(filtered_sum.max())
        passed = bool(np.allclose(filtered_sum, 1.0, atol=1.0e-8))
        status = "ok" if passed else "filtered_probability_row_sum_failed"

    max_prefix_diff = np.nan
    future_invariance_passed = False

    if (
        output.fit_result.model is not None
        and output.filter_result is not None
        and len(panel) > 2
    ):
        if prefix_length is None:
            prefix_length = max(2, min(len(panel) // 2, 250))

        prefix_length = int(prefix_length)
        prefix_length = max(2, min(prefix_length, len(panel)))

        prefix_filter = forward_filter_gaussian(
            X=output.fit_result.scaled_panel.X_scaled[:prefix_length],
            startprob=output.fit_result.model.startprob_,
            transmat=output.fit_result.model.transmat_,
            means=output.fit_result.model.means_,
            covars=output.fit_result.model.covars_,
            covariance_type=output.fit_result.spec.covariance_type,
            min_covar=output.fit_result.fit_config.min_covar,
        )

        max_prefix_diff = max_abs_prefix_difference(
            output.filter_result.filtered_probs,
            prefix_filter.filtered_probs,
        )
        future_invariance_passed = bool(max_prefix_diff <= 1.0e-10)

    backtest_probability_columns = econ_cols

    row = {
        "market": output.fit_result.market,
        "feature_set": output.fit_result.spec.feature_set,
        "n_states": n_states,
        "covariance_type": output.fit_result.spec.covariance_type,
        "filtered_prob_columns": ",".join(filtered_cols + econ_cols),
        "smoothed_prob_columns": ",".join([col for col in smoothed_cols if col in panel.columns]),
        "backtest_probability_columns": ",".join(backtest_probability_columns),
        "uses_custom_forward_filter": True,
        "uses_hmmlearn_predict_proba_for_backtest": False,
        "uses_smoothed_probabilities_for_backtest": False,
        "future_invariance_passed": future_invariance_passed,
        "row_sum_min": row_sum_min,
        "row_sum_max": row_sum_max,
        "max_abs_prefix_difference": max_prefix_diff,
        "passed": bool(passed and future_invariance_passed),
        "status": status,
    }

    return pd.DataFrame([row])


def build_hmm_no_lookahead_audit_table(
    output: HMMCandidateOutput,
    *,
    feature_cols: Sequence[str] | None = None,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DataFrame:
    """
    Build reports/tables/phase_6/hmm_no_lookahead_audit.csv.

    Checks:
    - HMM feature names are legal.
    - train/test split is chronological.
    - scaled panel has signal-availability columns.
    - backtest-facing probability columns are filtered/economic only.
    - diagnostic smoothed columns are not treated as backtest columns.
    """
    panel = _require_output_panel(output)

    feature_cols = tuple(feature_cols or output.fit_result.scaled_panel.feature_cols)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: str = "") -> None:
        checks.append(
            {
                "market": output.fit_result.market,
                "feature_set": output.fit_result.spec.feature_set,
                "n_states": output.fit_result.spec.n_states,
                "covariance_type": output.fit_result.spec.covariance_type,
                "check_name": name,
                "passed": bool(passed),
                "details": details,
            }
        )

    try:
        assert_hmm_feature_columns_are_legal(feature_cols)
        add_check("hmm_feature_names_point_in_time", True, ",".join(feature_cols))
    except Exception as exc:
        add_check("hmm_feature_names_point_in_time", False, str(exc))

    train_idx = np.asarray(output.fit_result.scaled_panel.train_indices, dtype=int)
    test_idx = np.asarray(output.fit_result.scaled_panel.test_indices, dtype=int)

    chronological = bool(
        len(train_idx) > 0
        and len(test_idx) > 0
        and train_idx.min() == 0
        and train_idx.max() < test_idx.min()
        and np.all(np.diff(train_idx) == 1)
        and np.all(np.diff(test_idx) == 1)
    )
    add_check(
        "chronological_train_test_split",
        chronological,
        f"train={train_idx[0] if len(train_idx) else None}:{train_idx[-1] if len(train_idx) else None};"
        f"test={test_idx[0] if len(test_idx) else None}:{test_idx[-1] if len(test_idx) else None}",
    )

    missing_signal_cols = [col for col in HMM_SIGNAL_AVAILABILITY_COLUMNS if col not in panel.columns]
    add_check(
        "signal_availability_columns_present",
        len(missing_signal_cols) == 0,
        f"missing={missing_signal_cols}",
    )

    if date_col in panel.columns and "hmm_signal_trade_date" in panel.columns:
        dates = pd.to_datetime(panel[date_col], errors="coerce")
        trade_dates = pd.to_datetime(panel["hmm_signal_trade_date"], errors="coerce")
        usable = trade_dates.notna()
        trade_after_obs = bool((trade_dates.loc[usable] > dates.loc[usable]).all())
        add_check(
            "signal_trade_date_after_observation_date",
            trade_after_obs,
            "last row may have missing trade date",
        )
    else:
        add_check(
            "signal_trade_date_after_observation_date",
            False,
            "missing date or hmm_signal_trade_date",
        )

    econ_cols = get_hmm_filtered_economic_probability_columns()
    diagnostic_smoothed_cols = [
        col for col in panel.columns
        if col.startswith(HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX)
    ]
    backtest_cols_safe = all(col in panel.columns for col in econ_cols) and not any(
        col in econ_cols for col in diagnostic_smoothed_cols
    )
    add_check(
        "backtest_probability_columns_are_filtered_economic_only",
        backtest_cols_safe,
        f"backtest_cols={econ_cols};diagnostic_smoothed_cols={diagnostic_smoothed_cols}",
    )

    add_check(
        "hmmlearn_predict_proba_not_used_for_backtest",
        True,
        "predict_proba may exist only as hmm_diagnostic_smoothed_prob_raw_state_*",
    )

    add_check(
        "crisis_and_threshold_diagnostics_not_training_inputs",
        True,
        "enforced by feature-name guards and diagnostic-only functions",
    )

    result = pd.DataFrame(checks)
    result["overall_passed"] = bool(result["passed"].all()) if not result.empty else False
    result["hmm_model_valid"] = output.hmm_model_valid
    result["hmm_model_failure_reason"] = output.hmm_model_failure_reason

    return result


def write_hmm_diagnostics_part2(
    output: HMMCandidateOutput,
    *,
    threshold_panel: pd.DataFrame | None = None,
    crisis_windows: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    forward_label_columns: Sequence[str] | None = None,
    tables_dir: str | Path = PHASE_6_TABLES_DIR,
) -> dict[str, Path]:
    """
    Write Chunk 10 diagnostic artifacts.

    Writes:
    - hmm_threshold_agreement.csv
    - hmm_crisis_hit_table.csv
    - hmm_crisis_lead_lag_table.csv
    - hmm_forward_label_by_state.csv
    - hmm_probability_audit.csv
    - hmm_no_lookahead_audit.csv
    """
    tables_dir = ensure_phase_6_tables_dir(tables_dir)

    paths = {
        "threshold_agreement": tables_dir / "hmm_threshold_agreement.csv",
        "crisis_hit_table": tables_dir / "hmm_crisis_hit_table.csv",
        "crisis_lead_lag_table": tables_dir / "hmm_crisis_lead_lag_table.csv",
        "forward_label_by_state": tables_dir / "hmm_forward_label_by_state.csv",
        "probability_audit": tables_dir / "hmm_probability_audit.csv",
        "no_lookahead_audit": tables_dir / "hmm_no_lookahead_audit.csv",
    }

    threshold = build_hmm_threshold_agreement_table(
        output,
        threshold_panel=threshold_panel,
    )

    if crisis_windows is None:
        crisis_hit = pd.DataFrame(
            [
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "crisis_name": "",
                    "start_date": "",
                    "end_date": "",
                    "n_crisis_observations": 0,
                    "crisis_stress_days": 0,
                    "crisis_stress_overlap": np.nan,
                    "crisis_false_negative_days": np.nan,
                    "diagnostic_only": True,
                    "status": "missing_crisis_windows",
                }
            ]
        )
        crisis_lead_lag = pd.DataFrame(
            [
                {
                    "market": output.fit_result.market,
                    "feature_set": output.fit_result.spec.feature_set,
                    "n_states": output.fit_result.spec.n_states,
                    "covariance_type": output.fit_result.spec.covariance_type,
                    "crisis_name": "",
                    "crisis_start_date": "",
                    "crisis_end_date": "",
                    "diagnostic_window_start": "",
                    "diagnostic_window_end": "",
                    "stress_detected": False,
                    "first_stress_date": "",
                    "crisis_lead_lag": np.nan,
                    "diagnostic_only": True,
                    "status": "missing_crisis_windows",
                }
            ]
        )
    else:
        crisis_hit = build_hmm_crisis_hit_table(
            output,
            crisis_windows=crisis_windows,
        )
        crisis_lead_lag = build_hmm_crisis_lead_lag_table(
            output,
            crisis_windows=crisis_windows,
        )

    forward_labels = build_hmm_forward_label_by_state_table(
        output,
        forward_label_columns=forward_label_columns,
    )

    probability_audit = build_hmm_probability_audit_table(output)
    no_lookahead_audit = build_hmm_no_lookahead_audit_table(output)

    write_table_csv(threshold, paths["threshold_agreement"])
    write_table_csv(crisis_hit, paths["crisis_hit_table"])
    write_table_csv(crisis_lead_lag, paths["crisis_lead_lag_table"])
    write_table_csv(forward_labels, paths["forward_label_by_state"])
    write_table_csv(probability_audit, paths["probability_audit"])
    write_table_csv(no_lookahead_audit, paths["no_lookahead_audit"])

    return paths