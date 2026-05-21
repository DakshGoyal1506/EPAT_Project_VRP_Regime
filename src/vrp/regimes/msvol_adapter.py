from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class MSVolAdapterError(RuntimeError):
    """Raised when MSVOL raw output cannot be standardized safely."""


RAW_REQUIRED_COLUMNS = [
    "date",
    "market",
    "msvol_raw_state_0_prob_filtered",
    "msvol_raw_state_1_prob_filtered",
    "msvol_raw_state_0_variance_estimate",
    "msvol_raw_state_1_variance_estimate",
    "msvol_conditional_variance",
    "msvol_conditional_volatility",
    "msvol_model_valid",
    "msvol_fit_status",
    "msvol_skip_reason",
]


PROCESSED_COLUMNS = [
    "date",
    "market",
    "msvol_signal_observation_date",
    "msvol_signal_available_after_close_date",
    "msvol_signal_trade_date",
    "msvol_state_for_next_session",
    "msvol_state_name_for_next_session",
    "msvol_filtered_prob_calm_for_next_session",
    "msvol_filtered_prob_transition_for_next_session",
    "msvol_filtered_prob_stress_for_next_session",
    "msvol_transition_state_modelled",
    "msvol_calm_raw_state",
    "msvol_stress_raw_state",
    "msvol_lower_variance_raw_state",
    "msvol_higher_variance_raw_state",
    "msvol_raw_state_0_variance_estimate",
    "msvol_raw_state_1_variance_estimate",
    "msvol_conditional_variance",
    "msvol_conditional_volatility",
    "msvol_model_valid",
    "msvol_fit_status",
    "msvol_skip_reason",
]


@dataclass(frozen=True)
class MSVolStateMapping:
    calm_raw_state: int
    stress_raw_state: int
    lower_variance_raw_state: int
    higher_variance_raw_state: int
    state_0_variance: float
    state_1_variance: float


@dataclass(frozen=True)
class MSVolImportResult:
    market: str
    status: str
    raw_output_csv: Path
    processed_output_parquet: Path
    metadata_json: Path
    probability_audit_csv: Path
    n_processed_rows: int
    skip_reason: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_project_path(path_like: str | Path, project_root: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / path


def market_slug(market: str) -> str:
    return market.lower().strip()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        return ""

    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return out


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def load_msvol_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path, project_root)

    if not config_path.exists():
        raise MSVolAdapterError(f"MSVOL config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise MSVolAdapterError(f"Invalid MSVOL config: {config_path}")

    return config


def normalize_market_arg(market: str, config: dict[str, Any]) -> list[str]:
    market = market.upper().strip()
    configured = set(config.get("markets", {}).keys())

    if market == "ALL":
        return sorted(configured)

    if market not in configured:
        allowed = sorted(configured | {"ALL"})
        raise MSVolAdapterError(f"Unknown market '{market}'. Allowed values: {allowed}")

    return [market]


def get_market_paths(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    market = market.upper().strip()
    slug = market_slug(market)

    market_cfg = config.get("markets", {}).get(market)
    if not isinstance(market_cfg, dict):
        raise MSVolAdapterError(f"Missing MSVOL market config for {market}")

    processed_dir = resolve_project_path(
        config.get("output_policy", {}).get("processed_dir", "data/processed"),
        project_root,
    )
    report_root = resolve_project_path(
        config.get("output_policy", {}).get("phase8_report_dir", "reports/tables/phase_8"),
        project_root,
    )

    raw_output_csv = resolve_project_path(market_cfg["raw_output_csv"], project_root)
    skip_report_json = resolve_project_path(market_cfg["skip_report_json"], project_root)
    preflight_json = resolve_project_path(market_cfg["preflight_json"], project_root)
    model_summary_json = resolve_project_path(market_cfg["model_summary_json"], project_root)

    report_dir = report_root / slug

    return {
        "raw_output_csv": raw_output_csv,
        "skip_report_json": skip_report_json,
        "preflight_json": preflight_json,
        "model_summary_json": model_summary_json,
        "processed_output_parquet": processed_dir / f"{slug}_msvol_regimes.parquet",
        "metadata_json": report_dir / "msvol_metadata.json",
        "probability_audit_csv": report_dir / "msvol_probability_audit.csv",
    }


def load_raw_msvol_output(market: str, config: dict[str, Any], project_root: str | Path | None = None) -> pd.DataFrame:
    paths = get_market_paths(market, config, project_root)
    raw_path = paths["raw_output_csv"]

    if not raw_path.exists():
        raise MSVolAdapterError(f"MSVOL raw output not found: {raw_path}")

    df = pd.read_csv(raw_path)
    validate_msvol_raw_schema(df, expected_market=market)
    validate_msvol_probability_rows(df)

    return df


def load_msvol_skip_report(market: str, config: dict[str, Any], project_root: str | Path | None = None) -> dict[str, Any]:
    paths = get_market_paths(market, config, project_root)
    return load_json_if_exists(paths["skip_report_json"])


def validate_msvol_raw_schema(df: pd.DataFrame, expected_market: str | None = None) -> None:
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MSVolAdapterError(f"MSVOL raw output missing required column(s): {missing}")

    if len(df) == 0:
        raise MSVolAdapterError("MSVOL raw output is empty.")

    date = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(date.isna().sum())
    if bad_dates:
        raise MSVolAdapterError(f"MSVOL raw output contains {bad_dates} invalid date value(s).")

    if int(date.duplicated().sum()):
        raise MSVolAdapterError("MSVOL raw output contains duplicate date rows.")

    if not date.is_monotonic_increasing:
        raise MSVolAdapterError("MSVOL raw output dates must be sorted ascending.")

    market_values = df["market"].astype(str).str.upper()
    if expected_market is not None:
        expected = expected_market.upper()
        bad_market = int((market_values != expected).sum())
        if bad_market:
            raise MSVolAdapterError(
                f"MSVOL raw output contains {bad_market} row(s) not matching market {expected}."
            )

    numeric_cols = [
        "msvol_raw_state_0_prob_filtered",
        "msvol_raw_state_1_prob_filtered",
        "msvol_raw_state_0_variance_estimate",
        "msvol_raw_state_1_variance_estimate",
        "msvol_conditional_variance",
        "msvol_conditional_volatility",
    ]

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int(values.isna().sum())
        if bad:
            raise MSVolAdapterError(f"Column '{col}' contains {bad} invalid numeric value(s).")

        if not np.all(np.isfinite(values.to_numpy(dtype=float))):
            raise MSVolAdapterError(f"Column '{col}' contains non-finite value(s).")

    for col in [
        "msvol_raw_state_0_variance_estimate",
        "msvol_raw_state_1_variance_estimate",
        "msvol_conditional_variance",
        "msvol_conditional_volatility",
    ]:
        values = pd.to_numeric(df[col], errors="coerce")
        bad_positive = int((values <= 0.0).sum())
        if bad_positive:
            raise MSVolAdapterError(f"Column '{col}' contains {bad_positive} non-positive value(s).")


def validate_msvol_probability_rows(df: pd.DataFrame, atol: float = 1e-5) -> None:
    p0 = pd.to_numeric(df["msvol_raw_state_0_prob_filtered"], errors="coerce").to_numpy(dtype=float)
    p1 = pd.to_numeric(df["msvol_raw_state_1_prob_filtered"], errors="coerce").to_numpy(dtype=float)

    prob = np.column_stack([p0, p1])

    if not np.all(np.isfinite(prob)):
        raise MSVolAdapterError("MSVOL filtered probabilities contain non-finite values.")

    if np.any(prob < -atol) or np.any(prob > 1.0 + atol):
        raise MSVolAdapterError("MSVOL filtered probabilities contain values outside [0, 1].")

    row_sums = prob.sum(axis=1)
    bad_rows = np.abs(row_sums - 1.0) > atol

    if np.any(bad_rows):
        raise MSVolAdapterError(
            f"MSVOL filtered probabilities contain {int(bad_rows.sum())} row(s) not summing to 1."
        )

    smoothed_cols = [
        "msvol_raw_state_0_prob_smoothed_diagnostic",
        "msvol_raw_state_1_prob_smoothed_diagnostic",
    ]

    if all(col in df.columns for col in smoothed_cols):
        s0 = pd.to_numeric(df[smoothed_cols[0]], errors="coerce").to_numpy(dtype=float)
        s1 = pd.to_numeric(df[smoothed_cols[1]], errors="coerce").to_numpy(dtype=float)
        smoothed = np.column_stack([s0, s1])

        if not np.all(np.isfinite(smoothed)):
            raise MSVolAdapterError("MSVOL smoothed diagnostic probabilities contain non-finite values.")

        if np.any(smoothed < -atol) or np.any(smoothed > 1.0 + atol):
            raise MSVolAdapterError("MSVOL smoothed diagnostic probabilities contain values outside [0, 1].")

        smoothed_sums = smoothed.sum(axis=1)
        bad_smoothed = np.abs(smoothed_sums - 1.0) > atol

        if np.any(bad_smoothed):
            raise MSVolAdapterError(
                f"MSVOL smoothed diagnostic probabilities contain {int(bad_smoothed.sum())} row(s) not summing to 1."
            )


def map_msvol_states_by_variance(df: pd.DataFrame) -> MSVolStateMapping:
    v0 = float(pd.to_numeric(df["msvol_raw_state_0_variance_estimate"], errors="coerce").mean())
    v1 = float(pd.to_numeric(df["msvol_raw_state_1_variance_estimate"], errors="coerce").mean())

    if not np.isfinite(v0) or not np.isfinite(v1):
        raise MSVolAdapterError("Cannot map MSVOL states because variance estimates are non-finite.")

    if v0 <= 0.0 or v1 <= 0.0:
        raise MSVolAdapterError("Cannot map MSVOL states because variance estimates are non-positive.")

    if np.isclose(v0, v1, rtol=1e-8, atol=1e-12):
        lower = 0
        higher = 1
    elif v0 < v1:
        lower = 0
        higher = 1
    else:
        lower = 1
        higher = 0

    return MSVolStateMapping(
        calm_raw_state=lower,
        stress_raw_state=higher,
        lower_variance_raw_state=lower,
        higher_variance_raw_state=higher,
        state_0_variance=v0,
        state_1_variance=v1,
    )


def _filtered_prob_col(raw_state: int) -> str:
    return f"msvol_raw_state_{raw_state}_prob_filtered"


def _smoothed_diag_col(raw_state: int) -> str:
    return f"msvol_raw_state_{raw_state}_prob_smoothed_diagnostic"


def standardize_msvol_output(df: pd.DataFrame, market: str, config: dict[str, Any]) -> tuple[pd.DataFrame, MSVolStateMapping]:
    market = market.upper().strip()

    validate_msvol_raw_schema(df, expected_market=market)
    validate_msvol_probability_rows(df)

    raw = df.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("date").reset_index(drop=True)

    mapping = map_msvol_states_by_variance(raw)

    calm_prob = pd.to_numeric(raw[_filtered_prob_col(mapping.calm_raw_state)], errors="coerce")
    stress_prob = pd.to_numeric(raw[_filtered_prob_col(mapping.stress_raw_state)], errors="coerce")

    stress_selected = stress_prob >= calm_prob

    transition_prob = float(
        config.get("state_policy", {}).get("transition_probability_for_valid_rows", 0.0)
    )

    transition_state_modelled = bool(
        config.get("state_policy", {}).get("transition_state_modelled", False)
    )

    processed = pd.DataFrame(
        {
            "date": raw["date"],
            "market": market,
            "msvol_signal_observation_date": raw["date"],
            "msvol_signal_available_after_close_date": raw["date"],
            "msvol_signal_trade_date": raw["date"].shift(-1),
            "msvol_state_for_next_session": np.where(stress_selected, 2, 0).astype(int),
            "msvol_state_name_for_next_session": np.where(stress_selected, "stress", "calm"),
            "msvol_filtered_prob_calm_for_next_session": calm_prob.astype(float),
            "msvol_filtered_prob_transition_for_next_session": transition_prob,
            "msvol_filtered_prob_stress_for_next_session": stress_prob.astype(float),
            "msvol_transition_state_modelled": transition_state_modelled,
            "msvol_calm_raw_state": mapping.calm_raw_state,
            "msvol_stress_raw_state": mapping.stress_raw_state,
            "msvol_lower_variance_raw_state": mapping.lower_variance_raw_state,
            "msvol_higher_variance_raw_state": mapping.higher_variance_raw_state,
            "msvol_raw_state_0_variance_estimate": pd.to_numeric(
                raw["msvol_raw_state_0_variance_estimate"], errors="coerce"
            ),
            "msvol_raw_state_1_variance_estimate": pd.to_numeric(
                raw["msvol_raw_state_1_variance_estimate"], errors="coerce"
            ),
            "msvol_conditional_variance": pd.to_numeric(raw["msvol_conditional_variance"], errors="coerce"),
            "msvol_conditional_volatility": pd.to_numeric(raw["msvol_conditional_volatility"], errors="coerce"),
            "msvol_model_valid": raw["msvol_model_valid"].astype(bool),
            "msvol_fit_status": raw["msvol_fit_status"].astype(str),
            "msvol_skip_reason": raw["msvol_skip_reason"].fillna("").astype(str),
        }
    )

    calm_smoothed_col = _smoothed_diag_col(mapping.calm_raw_state)
    stress_smoothed_col = _smoothed_diag_col(mapping.stress_raw_state)

    if calm_smoothed_col in raw.columns and stress_smoothed_col in raw.columns:
        processed["msvol_smoothed_prob_calm_diagnostic"] = pd.to_numeric(
            raw[calm_smoothed_col], errors="coerce"
        )
        processed["msvol_smoothed_prob_stress_diagnostic"] = pd.to_numeric(
            raw[stress_smoothed_col], errors="coerce"
        )

    forbidden_smoothed_next_session = [
        col
        for col in processed.columns
        if "smoothed" in col.lower() and "for_next_session" in col.lower()
    ]
    if forbidden_smoothed_next_session:
        raise MSVolAdapterError(
            "Smoothed probabilities cannot be used in next-session columns: "
            f"{forbidden_smoothed_next_session}"
        )

    prob_sum = (
        processed["msvol_filtered_prob_calm_for_next_session"]
        + processed["msvol_filtered_prob_transition_for_next_session"]
        + processed["msvol_filtered_prob_stress_for_next_session"]
    )

    bad_sum = np.abs(prob_sum.to_numpy(dtype=float) - 1.0) > 1e-5
    if np.any(bad_sum):
        raise MSVolAdapterError(
            f"Processed MSVOL probabilities contain {int(bad_sum.sum())} row(s) not summing to 1."
        )

    return processed, mapping


def empty_processed_msvol_frame() -> pd.DataFrame:
    dtypes: dict[str, Any] = {
        "date": "datetime64[ns]",
        "market": "object",
        "msvol_signal_observation_date": "datetime64[ns]",
        "msvol_signal_available_after_close_date": "datetime64[ns]",
        "msvol_signal_trade_date": "datetime64[ns]",
        "msvol_state_for_next_session": "int64",
        "msvol_state_name_for_next_session": "object",
        "msvol_filtered_prob_calm_for_next_session": "float64",
        "msvol_filtered_prob_transition_for_next_session": "float64",
        "msvol_filtered_prob_stress_for_next_session": "float64",
        "msvol_transition_state_modelled": "bool",
        "msvol_calm_raw_state": "int64",
        "msvol_stress_raw_state": "int64",
        "msvol_lower_variance_raw_state": "int64",
        "msvol_higher_variance_raw_state": "int64",
        "msvol_raw_state_0_variance_estimate": "float64",
        "msvol_raw_state_1_variance_estimate": "float64",
        "msvol_conditional_variance": "float64",
        "msvol_conditional_volatility": "float64",
        "msvol_model_valid": "bool",
        "msvol_fit_status": "object",
        "msvol_skip_reason": "object",
    }

    data = {col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()}
    return pd.DataFrame(data)


def build_msvol_skipped_output(market: str, skip_reason: str) -> pd.DataFrame:
    out = empty_processed_msvol_frame()
    return out


def build_probability_audit(
    processed: pd.DataFrame,
    mapping: MSVolStateMapping | None,
    market: str,
    status: str,
    skip_reason: str = "",
) -> pd.DataFrame:
    if processed.empty:
        return pd.DataFrame(
            [
                {
                    "market": market.upper(),
                    "status": status,
                    "n_rows": 0,
                    "min_prob_sum": np.nan,
                    "max_prob_sum": np.nan,
                    "max_abs_prob_sum_error": np.nan,
                    "min_calm_prob": np.nan,
                    "max_calm_prob": np.nan,
                    "min_stress_prob": np.nan,
                    "max_stress_prob": np.nan,
                    "n_missing_signal_trade_date": 0,
                    "calm_raw_state": np.nan,
                    "stress_raw_state": np.nan,
                    "state_0_variance_estimate": np.nan,
                    "state_1_variance_estimate": np.nan,
                    "validation_status": "skipped" if status == "skipped" else "empty",
                    "skip_reason": skip_reason,
                    "created_at_utc": utc_now_iso(),
                }
            ]
        )

    prob_sum = (
        processed["msvol_filtered_prob_calm_for_next_session"]
        + processed["msvol_filtered_prob_transition_for_next_session"]
        + processed["msvol_filtered_prob_stress_for_next_session"]
    )

    return pd.DataFrame(
        [
            {
                "market": market.upper(),
                "status": status,
                "n_rows": int(len(processed)),
                "min_prob_sum": float(prob_sum.min()),
                "max_prob_sum": float(prob_sum.max()),
                "max_abs_prob_sum_error": float(np.abs(prob_sum - 1.0).max()),
                "min_calm_prob": float(processed["msvol_filtered_prob_calm_for_next_session"].min()),
                "max_calm_prob": float(processed["msvol_filtered_prob_calm_for_next_session"].max()),
                "min_stress_prob": float(processed["msvol_filtered_prob_stress_for_next_session"].min()),
                "max_stress_prob": float(processed["msvol_filtered_prob_stress_for_next_session"].max()),
                "n_missing_signal_trade_date": int(processed["msvol_signal_trade_date"].isna().sum()),
                "calm_raw_state": None if mapping is None else int(mapping.calm_raw_state),
                "stress_raw_state": None if mapping is None else int(mapping.stress_raw_state),
                "state_0_variance_estimate": None if mapping is None else float(mapping.state_0_variance),
                "state_1_variance_estimate": None if mapping is None else float(mapping.state_1_variance),
                "validation_status": "ok",
                "skip_reason": skip_reason,
                "created_at_utc": utc_now_iso(),
            }
        ]
    )


def build_metadata_payload(
    market: str,
    status: str,
    paths: dict[str, Path],
    processed: pd.DataFrame,
    mapping: MSVolStateMapping | None,
    config_path: Path,
    model_summary: dict[str, Any],
    preflight: dict[str, Any],
    skip_report: dict[str, Any],
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "market": market.upper(),
        "phase": 8,
        "model_name": "msvol_appendix_v1",
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "true_msgarch": False,
        "appendix_only": True,
        "used_for_strategy": False,
        "used_for_backtest": False,
        "status": status,
        "skip_reason": skip_reason,
        "config_path": str(config_path),
        "config_hash_sha256": sha256_file(config_path),
        "raw_output_csv": str(paths["raw_output_csv"]),
        "raw_output_hash_sha256": sha256_file(paths["raw_output_csv"]),
        "processed_output_parquet": str(paths["processed_output_parquet"]),
        "processed_output_hash_sha256": sha256_file(paths["processed_output_parquet"]),
        "probability_audit_csv": str(paths["probability_audit_csv"]),
        "model_summary_json": str(paths["model_summary_json"]),
        "preflight_json": str(paths["preflight_json"]),
        "skip_report_json": str(paths["skip_report_json"]),
        "n_processed_rows": int(len(processed)),
        "start_date": None if processed.empty else str(processed["date"].min().date()),
        "end_date": None if processed.empty else str(processed["date"].max().date()),
        "n_missing_signal_trade_date": int(processed["msvol_signal_trade_date"].isna().sum()) if not processed.empty else 0,
        "state_mapping_rule": "lower variance raw state -> calm; higher variance raw state -> stress",
        "calm_raw_state": None if mapping is None else int(mapping.calm_raw_state),
        "stress_raw_state": None if mapping is None else int(mapping.stress_raw_state),
        "lower_variance_raw_state": None if mapping is None else int(mapping.lower_variance_raw_state),
        "higher_variance_raw_state": None if mapping is None else int(mapping.higher_variance_raw_state),
        "state_0_variance_estimate": None if mapping is None else float(mapping.state_0_variance),
        "state_1_variance_estimate": None if mapping is None else float(mapping.state_1_variance),
        "transition_state_modelled": False,
        "transition_probability_for_valid_rows": 0.0,
        "filtered_probabilities_used_for_next_session": True,
        "smoothed_probabilities_diagnostic_only": True,
        "model_summary": model_summary,
        "preflight": preflight,
        "skip_report": skip_report,
        "report_note": (
            "This is not true MSGARCH. True MSGARCH remains optional and requires "
            "R's MSGARCH package. This Python model is used only as a volatility-regime robustness proxy."
        ),
        "created_at_utc": utc_now_iso(),
    }


def import_msvol_outputs_for_market(
    market: str,
    config: dict[str, Any],
    config_path: str | Path,
    project_root: str | Path | None = None,
    allow_skip: bool = False,
) -> MSVolImportResult:
    market = market.upper().strip()
    config_path_resolved = resolve_project_path(config_path, project_root)
    paths = get_market_paths(market, config, project_root)

    raw_path = paths["raw_output_csv"]
    processed_path = paths["processed_output_parquet"]
    metadata_path = paths["metadata_json"]
    audit_path = paths["probability_audit_csv"]

    preflight = load_json_if_exists(paths["preflight_json"])
    model_summary = load_json_if_exists(paths["model_summary_json"])
    skip_report = load_json_if_exists(paths["skip_report_json"])

    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        processed, mapping = standardize_msvol_output(raw, market, config)

        status = "ok"
        skip_reason = ""

    else:
        skip_reason = (
            skip_report.get("skip_reason")
            or preflight.get("skip_reason")
            or f"MSVOL raw output missing: {raw_path}"
        )

        if not allow_skip:
            raise MSVolAdapterError(
                f"MSVOL raw output missing for {market}. "
                f"Use --allow-skip to write skipped metadata. Reason: {skip_reason}"
            )

        processed = build_msvol_skipped_output(market, skip_reason)
        mapping = None
        status = "skipped"

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_path, index=False)

    audit = build_probability_audit(
        processed=processed,
        mapping=mapping,
        market=market,
        status=status,
        skip_reason=skip_reason,
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    metadata = build_metadata_payload(
        market=market,
        status=status,
        paths=paths,
        processed=processed,
        mapping=mapping,
        config_path=config_path_resolved,
        model_summary=model_summary,
        preflight=preflight,
        skip_report=skip_report,
        skip_reason=skip_reason,
    )
    write_json(metadata_path, metadata)

    return MSVolImportResult(
        market=market,
        status=status,
        raw_output_csv=raw_path,
        processed_output_parquet=processed_path,
        metadata_json=metadata_path,
        probability_audit_csv=audit_path,
        n_processed_rows=int(len(processed)),
        skip_reason=skip_reason,
    )