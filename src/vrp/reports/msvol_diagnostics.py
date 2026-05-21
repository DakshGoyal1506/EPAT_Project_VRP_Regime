from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class MSVolDiagnosticsError(RuntimeError):
    """Raised when MSVOL diagnostics cannot be produced safely."""


MSVOL_REQUIRED_COLUMNS = [
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
    "msvol_model_valid",
    "msvol_fit_status",
    "msvol_skip_reason",
]


SUMMARY_COLUMNS = [
    "market",
    "status",
    "n_msvol_days",
    "n_overlap_days",
    "n_overlap_days_threshold",
    "n_overlap_days_hmm",
    "n_overlap_days_mar",
    "corr_msvol_stress_vs_threshold_stress",
    "corr_msvol_stress_vs_hmm_stress",
    "corr_msvol_stress_vs_mar_stress",
    "threshold_agreement_rate",
    "hmm_agreement_rate",
    "mar_agreement_rate",
    "avg_msvol_stress_prob_in_threshold_stress",
    "avg_msvol_stress_prob_in_hmm_stress",
    "avg_msvol_stress_prob_in_mar_stress",
    "msvol_stress_days_pct",
    "avg_return_in_msvol_calm",
    "avg_return_in_msvol_stress",
    "avg_rv_in_msvol_calm",
    "avg_rv_in_msvol_stress",
    "avg_vrp_har_in_msvol_calm",
    "avg_vrp_har_in_msvol_stress",
    "selected_return_column",
    "selected_rv_column",
    "selected_vrp_column",
    "threshold_source_path",
    "hmm_source_path",
    "mar_source_path",
    "feature_source_path",
    "diagnostic_only",
    "used_for_strategy",
    "used_for_backtest",
    "skip_reason",
    "created_at_utc",
]


DURATION_COLUMNS = [
    "market",
    "state_name",
    "n_runs",
    "total_days",
    "share_days",
    "mean_duration_days",
    "median_duration_days",
    "min_duration_days",
    "max_duration_days",
    "diagnostic_only",
    "created_at_utc",
]


@dataclass(frozen=True)
class LoadedTable:
    path: Path | None
    df: pd.DataFrame


@dataclass(frozen=True)
class MSVolDiagnosticsResult:
    market: str
    status: str
    comparison_summary_csv: Path
    state_duration_summary_csv: Path
    appendix_csv: Path
    n_summary_rows: int
    n_duration_rows: int
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


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return out


def load_msvol_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path, project_root)
    if not config_path.exists():
        raise MSVolDiagnosticsError(f"MSVOL config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise MSVolDiagnosticsError(f"Invalid MSVOL config: {config_path}")

    return config


def normalize_market_arg(market: str, config: dict[str, Any]) -> list[str]:
    market = market.upper().strip()
    configured = set(config.get("markets", {}).keys())

    if market == "ALL":
        return sorted(configured)

    if market not in configured:
        allowed = sorted(configured | {"ALL"})
        raise MSVolDiagnosticsError(f"Unknown market '{market}'. Allowed values: {allowed}")

    return [market]


def _read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)

    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    raise MSVolDiagnosticsError(f"Unsupported table format: {file_path}")


def _first_existing(paths: list[Path]) -> LoadedTable:
    for path in paths:
        if path.exists():
            return LoadedTable(path=path, df=_read_table(path))

    return LoadedTable(path=None, df=pd.DataFrame())


def get_msvol_processed_path(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    slug = market_slug(market)
    processed_dir = resolve_project_path(
        config.get("output_policy", {}).get("processed_dir", "data/processed"),
        project_root,
    )
    return processed_dir / f"{slug}_msvol_regimes.parquet"


def get_report_paths(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    slug = market_slug(market)
    report_root = resolve_project_path(
        config.get("output_policy", {}).get("phase8_report_dir", "reports/tables/phase_8"),
        project_root,
    )

    return {
        "comparison_summary_csv": report_root / slug / "msvol_comparison_summary.csv",
        "state_duration_summary_csv": report_root / slug / "msvol_state_duration_summary.csv",
        "appendix_csv": report_root / "msvol_model_comparison_appendix.csv",
        "diagnostics_metadata_json": report_root / slug / "msvol_diagnostics_metadata.json",
    }


def _custom_candidate_paths(
    config: dict[str, Any],
    market: str,
    role: str,
    project_root: str | Path | None,
) -> list[Path]:
    diagnostics = config.get("diagnostics", {})
    comparator_paths = diagnostics.get("comparator_paths", {})

    if not isinstance(comparator_paths, dict):
        return []

    market_cfg = comparator_paths.get(market.upper(), {})
    if not isinstance(market_cfg, dict):
        return []

    raw = market_cfg.get(role, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    return [resolve_project_path(item, project_root) for item in raw]


def candidate_paths_for_market(
    market: str,
    role: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> list[Path]:
    market = market.upper().strip()
    slug = market_slug(market)

    custom = _custom_candidate_paths(config, market, role, project_root)
    if custom:
        return custom

    if role == "threshold":
        rels = [
            f"data/processed/{slug}_threshold_regimes.parquet",
            f"data/processed/{slug}_threshold_regime.parquet",
            f"data/processed/{slug}_threshold_panel.parquet",
            f"data/processed/{slug}_threshold_states.parquet",
        ]
    elif role == "hmm":
        rels = [
            f"data/processed/{slug}_hmm_regimes.parquet",
            f"data/processed/{slug}_gaussian_hmm_regimes.parquet",
            f"data/processed/{slug}_hmm_states.parquet",
        ]
    elif role == "mar":
        rels = [
            f"data/processed/{slug}_markov_autoreg_regimes.parquet",
            f"data/processed/{slug}_mar_regimes.parquet",
            f"data/processed/{slug}_markov_autoreg_states.parquet",
            f"data/processed/{slug}_arhmm_regimes.parquet",
        ]
    elif role == "features":
        rels = [
            f"data/processed/{slug}_vrp_har.parquet",
            f"data/processed/{slug}_vrp.parquet",
            f"data/processed/{slug}_rv.parquet",
        ]
    else:
        raise MSVolDiagnosticsError(f"Unknown diagnostics role: {role}")

    return [resolve_project_path(rel, project_root) for rel in rels]


def load_optional_role_table(
    market: str,
    role: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> LoadedTable:
    return _first_existing(candidate_paths_for_market(market, role, config, project_root))


def load_msvol_processed(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    path = get_msvol_processed_path(market, config, project_root)

    if not path.exists():
        raise MSVolDiagnosticsError(f"MSVOL processed regime panel not found: {path}")

    df = pd.read_parquet(path)
    validate_msvol_processed_schema(df, expected_market=market)

    return df


def validate_msvol_processed_schema(df: pd.DataFrame, expected_market: str | None = None) -> None:
    missing = [col for col in MSVOL_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MSVolDiagnosticsError(f"MSVOL processed panel missing required column(s): {missing}")

    if len(df) == 0:
        return

    date = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(date.isna().sum())
    if bad_dates:
        raise MSVolDiagnosticsError(f"MSVOL processed panel contains {bad_dates} invalid date value(s).")

    if int(date.duplicated().sum()):
        raise MSVolDiagnosticsError("MSVOL processed panel contains duplicate date rows.")

    market_values = df["market"].astype(str).str.upper()
    if expected_market is not None:
        expected = expected_market.upper()
        bad_market = int((market_values != expected).sum())
        if bad_market:
            raise MSVolDiagnosticsError(
                f"MSVOL processed panel contains {bad_market} row(s) not matching market {expected}."
            )

    prob_cols = [
        "msvol_filtered_prob_calm_for_next_session",
        "msvol_filtered_prob_transition_for_next_session",
        "msvol_filtered_prob_stress_for_next_session",
    ]

    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce")
    if int(probs.isna().sum().sum()):
        raise MSVolDiagnosticsError("MSVOL processed probabilities contain invalid numeric value(s).")

    prob_sum = probs.sum(axis=1)
    bad_sum = np.abs(prob_sum.to_numpy(dtype=float) - 1.0) > 1e-5
    if np.any(bad_sum):
        raise MSVolDiagnosticsError(
            f"MSVOL processed probabilities contain {int(bad_sum.sum())} row(s) not summing to 1."
        )

    smoothed_next_cols = [
        col
        for col in df.columns
        if "smoothed" in col.lower() and "for_next_session" in col.lower()
    ]
    if smoothed_next_cols:
        raise MSVolDiagnosticsError(
            f"Smoothed probabilities cannot be next-session columns: {smoothed_next_cols}"
        )


def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if "date" not in out.columns:
        date_like = [col for col in out.columns if col.lower().endswith("date")]
        if not date_like:
            raise MSVolDiagnosticsError("Table does not contain a date column.")
        out = out.rename(columns={date_like[0]: "date"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    out["date"] = out["date"].dt.normalize()

    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def _contains_stress_label(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return text.str.contains("stress|crisis|high_vol|high-vol|high volatility|risk_off|risk-off", regex=True)


def _find_stress_probability_column(df: pd.DataFrame, model_name: str) -> str | None:
    cols = list(df.columns)
    lower_map = {col: col.lower() for col in cols}

    preferred = [
        f"{model_name}_filtered_prob_stress_for_next_session",
        f"{model_name}_prob_stress_for_next_session",
        f"{model_name}_stress_prob_for_next_session",
        f"{model_name}_prob_stress",
        f"{model_name}_stress_probability",
    ]

    for candidate in preferred:
        for col, lower in lower_map.items():
            if lower == candidate.lower():
                return col

    for col, lower in lower_map.items():
        if "prob" in lower and "stress" in lower and "smoothed" not in lower:
            return col

    return None


def _find_state_name_column(df: pd.DataFrame, model_name: str) -> str | None:
    cols = list(df.columns)
    lower_map = {col: col.lower() for col in cols}

    preferred = [
        f"{model_name}_state_name_for_next_session",
        f"{model_name}_state_name",
        f"{model_name}_regime_name_for_next_session",
        f"{model_name}_regime_name",
        "state_name_for_next_session",
        "state_name",
        "regime_name_for_next_session",
        "regime_name",
        "regime",
    ]

    for candidate in preferred:
        for col, lower in lower_map.items():
            if lower == candidate.lower():
                return col

    for col, lower in lower_map.items():
        if "state" in lower and "name" in lower:
            return col

    for col, lower in lower_map.items():
        if "regime" in lower and "prob" not in lower:
            return col

    return None


def extract_stress_indicator(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", f"{model_name}_stress"])

    data = _standardize_date_column(df)

    prob_col = _find_stress_probability_column(data, model_name)
    if prob_col is not None:
        stress_prob = pd.to_numeric(data[prob_col], errors="coerce")
        out = pd.DataFrame(
            {
                "date": data["date"],
                f"{model_name}_stress": np.where(stress_prob >= 0.5, 1.0, 0.0),
            }
        )
        return out.dropna(subset=[f"{model_name}_stress"])

    state_col = _find_state_name_column(data, model_name)
    if state_col is not None:
        out = pd.DataFrame(
            {
                "date": data["date"],
                f"{model_name}_stress": _contains_stress_label(data[state_col]).astype(float),
            }
        )
        return out.dropna(subset=[f"{model_name}_stress"])

    return pd.DataFrame(columns=["date", f"{model_name}_stress"])


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(pair) < 2:
        return np.nan

    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan

    return float(pair["x"].corr(pair["y"]))


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return np.nan
    return float(values.mean())


def _safe_agreement(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) == 0:
        return np.nan
    return float((pair["x"].astype(int) == pair["y"].astype(int)).mean())


def compute_pairwise_comparison(
    msvol: pd.DataFrame,
    comparator: pd.DataFrame,
    model_name: str,
) -> dict[str, Any]:
    model_name = model_name.lower().strip()

    if comparator.empty:
        return {
            f"n_overlap_days_{model_name}": 0,
            f"corr_msvol_stress_vs_{model_name}_stress": np.nan,
            f"{model_name}_agreement_rate": np.nan,
            f"avg_msvol_stress_prob_in_{model_name}_stress": np.nan,
        }

    ms = msvol[
        [
            "date",
            "msvol_state_name_for_next_session",
            "msvol_filtered_prob_stress_for_next_session",
        ]
    ].copy()
    ms["date"] = pd.to_datetime(ms["date"], errors="coerce").dt.normalize()
    ms["msvol_stress"] = (
        ms["msvol_state_name_for_next_session"].astype(str).str.lower().eq("stress").astype(float)
    )

    comp = extract_stress_indicator(comparator, model_name)

    if comp.empty:
        return {
            f"n_overlap_days_{model_name}": 0,
            f"corr_msvol_stress_vs_{model_name}_stress": np.nan,
            f"{model_name}_agreement_rate": np.nan,
            f"avg_msvol_stress_prob_in_{model_name}_stress": np.nan,
        }

    merged = ms.merge(comp, on="date", how="inner")
    stress_col = f"{model_name}_stress"

    if merged.empty:
        return {
            f"n_overlap_days_{model_name}": 0,
            f"corr_msvol_stress_vs_{model_name}_stress": np.nan,
            f"{model_name}_agreement_rate": np.nan,
            f"avg_msvol_stress_prob_in_{model_name}_stress": np.nan,
        }

    comparator_stress_mask = merged[stress_col] == 1.0

    return {
        f"n_overlap_days_{model_name}": int(len(merged)),
        f"corr_msvol_stress_vs_{model_name}_stress": _safe_corr(
            merged["msvol_stress"],
            merged[stress_col],
        ),
        f"{model_name}_agreement_rate": _safe_agreement(
            merged["msvol_stress"],
            merged[stress_col],
        ),
        f"avg_msvol_stress_prob_in_{model_name}_stress": _safe_mean(
            merged.loc[comparator_stress_mask, "msvol_filtered_prob_stress_for_next_session"]
        ),
    }


def _select_numeric_column(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str]:
    if df.empty:
        return pd.Series(dtype=float), ""

    lower_to_col = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        col = lower_to_col.get(candidate.lower())
        if col is not None:
            return pd.to_numeric(df[col], errors="coerce"), col

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in df.columns:
            if candidate_lower in col.lower():
                return pd.to_numeric(df[col], errors="coerce"), col

    return pd.Series(index=df.index, dtype=float), ""


def merge_feature_context(msvol: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    base = msvol.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()

    if feature_df.empty:
        return base

    features = _standardize_date_column(feature_df)

    drop_cols = [col for col in ["market"] if col in features.columns]
    features = features.drop(columns=drop_cols)

    return base.merge(features, on="date", how="left")


def compute_state_feature_means(
    msvol: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> dict[str, Any]:
    joined = merge_feature_context(msvol, feature_df)

    return_series, return_col = _select_numeric_column(
        joined,
        [
            "log_return",
            "index_return",
            "simple_return",
            "return_for_msgarch",
        ],
    )

    rv_series, rv_col = _select_numeric_column(
        joined,
        [
            "rv_gk_22d_ann",
            "rv_parkinson_22d_ann",
            "rv_cc_22d_ann",
            "rv_rs_22d_ann",
            "rv_22d_ann",
            "realized_variance",
            "rv",
        ],
    )

    vrp_series, vrp_col = _select_numeric_column(
        joined,
        [
            "vrp_har_gk",
            "vrp_har",
            "vrp_gk_har",
            "vrp_gk",
            "vrp",
        ],
    )

    state_name = joined["msvol_state_name_for_next_session"].astype(str).str.lower()
    calm = state_name.eq("calm")
    stress = state_name.eq("stress")

    return {
        "avg_return_in_msvol_calm": _safe_mean(return_series.loc[calm]),
        "avg_return_in_msvol_stress": _safe_mean(return_series.loc[stress]),
        "avg_rv_in_msvol_calm": _safe_mean(rv_series.loc[calm]),
        "avg_rv_in_msvol_stress": _safe_mean(rv_series.loc[stress]),
        "avg_vrp_har_in_msvol_calm": _safe_mean(vrp_series.loc[calm]),
        "avg_vrp_har_in_msvol_stress": _safe_mean(vrp_series.loc[stress]),
        "selected_return_column": return_col,
        "selected_rv_column": rv_col,
        "selected_vrp_column": vrp_col,
    }


def build_msvol_comparison_summary(
    market: str,
    msvol: pd.DataFrame,
    threshold: LoadedTable,
    hmm: LoadedTable,
    mar: LoadedTable,
    features: LoadedTable,
    status: str = "ok",
    skip_reason: str = "",
) -> pd.DataFrame:
    if msvol.empty:
        row: dict = {col: np.nan for col in SUMMARY_COLUMNS}
        row.update(
            {
                "market": market.upper(),
                "status": status,
                "n_msvol_days": 0,
                "n_overlap_days": 0,
                "n_overlap_days_threshold": 0,
                "n_overlap_days_hmm": 0,
                "n_overlap_days_mar": 0,
                "diagnostic_only": True,
                "used_for_strategy": False,
                "used_for_backtest": False,
                "skip_reason": skip_reason,
                "created_at_utc": utc_now_iso(),
            }
        )
        return pd.DataFrame([row], columns=SUMMARY_COLUMNS)

    ms = msvol.copy()
    ms["date"] = pd.to_datetime(ms["date"], errors="coerce").dt.normalize()

    threshold_metrics = compute_pairwise_comparison(ms, threshold.df, "threshold")
    hmm_metrics = compute_pairwise_comparison(ms, hmm.df, "hmm")
    mar_metrics = compute_pairwise_comparison(ms, mar.df, "mar")
    feature_metrics = compute_state_feature_means(ms, features.df)

    msvol_stress = ms["msvol_state_name_for_next_session"].astype(str).str.lower().eq("stress")

    overlap_counts = [
        threshold_metrics["n_overlap_days_threshold"],
        hmm_metrics["n_overlap_days_hmm"],
        mar_metrics["n_overlap_days_mar"],
    ]

    row = {
        "market": market.upper(),
        "status": status,
        "n_msvol_days": int(len(ms)),
        "n_overlap_days": int(max(overlap_counts)) if overlap_counts else 0,
        "n_overlap_days_threshold": threshold_metrics["n_overlap_days_threshold"],
        "n_overlap_days_hmm": hmm_metrics["n_overlap_days_hmm"],
        "n_overlap_days_mar": mar_metrics["n_overlap_days_mar"],
        "corr_msvol_stress_vs_threshold_stress": threshold_metrics[
            "corr_msvol_stress_vs_threshold_stress"
        ],
        "corr_msvol_stress_vs_hmm_stress": hmm_metrics[
            "corr_msvol_stress_vs_hmm_stress"
        ],
        "corr_msvol_stress_vs_mar_stress": mar_metrics[
            "corr_msvol_stress_vs_mar_stress"
        ],
        "threshold_agreement_rate": threshold_metrics["threshold_agreement_rate"],
        "hmm_agreement_rate": hmm_metrics["hmm_agreement_rate"],
        "mar_agreement_rate": mar_metrics["mar_agreement_rate"],
        "avg_msvol_stress_prob_in_threshold_stress": threshold_metrics[
            "avg_msvol_stress_prob_in_threshold_stress"
        ],
        "avg_msvol_stress_prob_in_hmm_stress": hmm_metrics[
            "avg_msvol_stress_prob_in_hmm_stress"
        ],
        "avg_msvol_stress_prob_in_mar_stress": mar_metrics[
            "avg_msvol_stress_prob_in_mar_stress"
        ],
        "msvol_stress_days_pct": float(msvol_stress.mean()),
        "avg_return_in_msvol_calm": feature_metrics["avg_return_in_msvol_calm"],
        "avg_return_in_msvol_stress": feature_metrics["avg_return_in_msvol_stress"],
        "avg_rv_in_msvol_calm": feature_metrics["avg_rv_in_msvol_calm"],
        "avg_rv_in_msvol_stress": feature_metrics["avg_rv_in_msvol_stress"],
        "avg_vrp_har_in_msvol_calm": feature_metrics["avg_vrp_har_in_msvol_calm"],
        "avg_vrp_har_in_msvol_stress": feature_metrics["avg_vrp_har_in_msvol_stress"],
        "selected_return_column": feature_metrics["selected_return_column"],
        "selected_rv_column": feature_metrics["selected_rv_column"],
        "selected_vrp_column": feature_metrics["selected_vrp_column"],
        "threshold_source_path": "" if threshold.path is None else str(threshold.path),
        "hmm_source_path": "" if hmm.path is None else str(hmm.path),
        "mar_source_path": "" if mar.path is None else str(mar.path),
        "feature_source_path": "" if features.path is None else str(features.path),
        "diagnostic_only": True,
        "used_for_strategy": False,
        "used_for_backtest": False,
        "skip_reason": skip_reason,
        "created_at_utc": utc_now_iso(),
    }

    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def build_state_duration_summary(market: str, msvol: pd.DataFrame) -> pd.DataFrame:
    if msvol.empty:
        return pd.DataFrame(columns=DURATION_COLUMNS)

    df = msvol.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["state_name"] = df["msvol_state_name_for_next_session"].astype(str).str.lower()

    if df.empty:
        return pd.DataFrame(columns=DURATION_COLUMNS)

    group_id = df["state_name"].ne(df["state_name"].shift()).cumsum()
    runs = (
        df.groupby(group_id)
        .agg(
            state_name=("state_name", "first"),
            duration_days=("state_name", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    total_days = int(len(df))

    for state_name, grp in runs.groupby("state_name"):
        durations = grp["duration_days"].astype(float)

        rows.append(
            {
                "market": market.upper(),
                "state_name": state_name,
                "n_runs": int(len(grp)),
                "total_days": int(durations.sum()),
                "share_days": float(durations.sum() / total_days) if total_days else np.nan,
                "mean_duration_days": float(durations.mean()),
                "median_duration_days": float(durations.median()),
                "min_duration_days": float(durations.min()),
                "max_duration_days": float(durations.max()),
                "diagnostic_only": True,
                "created_at_utc": utc_now_iso(),
            }
        )

    return pd.DataFrame(rows, columns=DURATION_COLUMNS).sort_values("state_name").reset_index(drop=True)


def run_msvol_diagnostics_for_market(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
    allow_skip: bool = False,
) -> MSVolDiagnosticsResult:
    market = market.upper().strip()
    report_paths = get_report_paths(market, config, project_root)

    for path in report_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        msvol = load_msvol_processed(market, config, project_root)
        status = "ok"
        skip_reason = ""

    except Exception as exc:
        if not allow_skip:
            raise

        msvol = pd.DataFrame(columns=MSVOL_REQUIRED_COLUMNS)
        status = "skipped"
        skip_reason = str(exc)

    threshold = load_optional_role_table(market, "threshold", config, project_root)
    hmm = load_optional_role_table(market, "hmm", config, project_root)
    mar = load_optional_role_table(market, "mar", config, project_root)
    features = load_optional_role_table(market, "features", config, project_root)

    comparison = build_msvol_comparison_summary(
        market=market,
        msvol=msvol,
        threshold=threshold,
        hmm=hmm,
        mar=mar,
        features=features,
        status=status,
        skip_reason=skip_reason,
    )

    duration = build_state_duration_summary(market, msvol)

    comparison.to_csv(report_paths["comparison_summary_csv"], index=False)
    duration.to_csv(report_paths["state_duration_summary_csv"], index=False)

    metadata = {
        "market": market,
        "status": status,
        "diagnostic_only": True,
        "used_for_strategy": False,
        "used_for_backtest": False,
        "msvol_processed_path": str(get_msvol_processed_path(market, config, project_root)),
        "comparison_summary_csv": str(report_paths["comparison_summary_csv"]),
        "state_duration_summary_csv": str(report_paths["state_duration_summary_csv"]),
        "threshold_source_path": "" if threshold.path is None else str(threshold.path),
        "hmm_source_path": "" if hmm.path is None else str(hmm.path),
        "mar_source_path": "" if mar.path is None else str(mar.path),
        "feature_source_path": "" if features.path is None else str(features.path),
        "skip_reason": skip_reason,
        "created_at_utc": utc_now_iso(),
    }
    write_json(report_paths["diagnostics_metadata_json"], metadata)

    update_appendix_comparison_table(
        market=market,
        comparison=comparison,
        appendix_csv=report_paths["appendix_csv"],
    )

    return MSVolDiagnosticsResult(
        market=market,
        status=status,
        comparison_summary_csv=report_paths["comparison_summary_csv"],
        state_duration_summary_csv=report_paths["state_duration_summary_csv"],
        appendix_csv=report_paths["appendix_csv"],
        n_summary_rows=int(len(comparison)),
        n_duration_rows=int(len(duration)),
        skip_reason=skip_reason,
    )


def update_appendix_comparison_table(
    market: str,
    comparison: pd.DataFrame,
    appendix_csv: Path,
) -> Path:
    appendix_csv.parent.mkdir(parents=True, exist_ok=True)

    if appendix_csv.exists():
        existing = pd.read_csv(appendix_csv)
    else:
        existing = pd.DataFrame(columns=SUMMARY_COLUMNS)

    market = market.upper().strip()

    if "market" in existing.columns and not existing.empty:
        existing = existing[existing["market"].astype(str).str.upper() != market].copy()

    combined = pd.concat([existing, comparison], ignore_index=True)
    combined = combined.reindex(columns=SUMMARY_COLUMNS)
    combined = combined.sort_values("market").reset_index(drop=True)

    combined.to_csv(appendix_csv, index=False)
    return appendix_csv