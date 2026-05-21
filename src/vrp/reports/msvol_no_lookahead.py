from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class MSVolNoLookaheadError(RuntimeError):
    """Raised when MSVOL no-lookahead audit cannot be completed."""


REQUIRED_MSVOL_COLUMNS = [
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


FORBIDDEN_STRATEGY_FRAGMENTS = [
    "position",
    "exposure",
    "weight",
    "leverage",
    "pnl",
    "profit",
    "strategy_return",
    "backtest",
    "trade_size",
    "allocation",
]


@dataclass(frozen=True)
class MSVolNoLookaheadAuditResult:
    market: str
    status: str
    market_audit_csv: Path
    combined_audit_csv: Path
    n_checks: int
    n_failed_error_checks: int
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


def load_msvol_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path, project_root)

    if not config_path.exists():
        raise MSVolNoLookaheadError(f"MSVOL config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise MSVolNoLookaheadError(f"Invalid MSVOL config: {config_path}")

    return config


def normalize_market_arg(market: str, config: dict[str, Any]) -> list[str]:
    market = market.upper().strip()
    configured = set(config.get("markets", {}).keys())

    if market == "ALL":
        return sorted(configured)

    if market not in configured:
        allowed = sorted(configured | {"ALL"})
        raise MSVolNoLookaheadError(f"Unknown market '{market}'. Allowed values: {allowed}")

    return [market]


def get_phase8_report_root(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    return resolve_project_path(
        config.get("output_policy", {}).get("phase8_report_dir", "reports/tables/phase_8"),
        project_root,
    )


def get_processed_path(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    processed_dir = resolve_project_path(
        config.get("output_policy", {}).get("processed_dir", "data/processed"),
        project_root,
    )
    return processed_dir / f"{market_slug(market)}_msvol_regimes.parquet"


def get_report_paths(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    report_root = get_phase8_report_root(config, project_root)
    slug = market_slug(market)

    return {
        "market_audit_csv": report_root / slug / "msvol_no_lookahead_audit.csv",
        "combined_audit_csv": report_root / "msvol_no_lookahead_audit.csv",
        "metadata_json": report_root / slug / "msvol_metadata.json",
        "comparison_summary_csv": report_root / slug / "msvol_comparison_summary.csv",
        "diagnostics_metadata_json": report_root / slug / "msvol_diagnostics_metadata.json",
    }


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    return None


def _add_check(
    rows: list[dict[str, Any]],
    market: str,
    check_name: str,
    passed: bool,
    severity: str,
    detail: str,
) -> None:
    rows.append(
        {
            "market": market.upper(),
            "check_name": check_name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
            "created_at_utc": utc_now_iso(),
        }
    )


def _load_processed_or_empty(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None,
    allow_skip: bool,
) -> tuple[pd.DataFrame, str]:
    processed_path = get_processed_path(market, config, project_root)

    if not processed_path.exists():
        reason = f"MSVOL processed panel not found: {processed_path}"
        if allow_skip:
            return pd.DataFrame(), reason
        raise MSVolNoLookaheadError(reason)

    df = pd.read_parquet(processed_path)
    return df, ""


def build_no_lookahead_audit(
    market: str,
    processed: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    comparison_summary: pd.DataFrame | None = None,
    diagnostics_metadata: dict[str, Any] | None = None,
    skip_reason: str = "",
) -> pd.DataFrame:
    market = market.upper().strip()
    metadata = metadata or {}
    diagnostics_metadata = diagnostics_metadata or {}

    rows: list[dict[str, Any]] = []

    if processed.empty:
        _add_check(
            rows,
            market,
            "processed_panel_non_empty",
            False,
            "error",
            skip_reason or "Processed MSVOL panel is empty.",
        )
        return pd.DataFrame(rows)

    missing = [col for col in REQUIRED_MSVOL_COLUMNS if col not in processed.columns]
    _add_check(
        rows,
        market,
        "required_columns_present",
        not missing,
        "error",
        "missing=[]" if not missing else f"missing={missing}",
    )

    if missing:
        return pd.DataFrame(rows)

    df = processed.copy()

    dates = pd.to_datetime(df["date"], errors="coerce")
    obs_dates = pd.to_datetime(df["msvol_signal_observation_date"], errors="coerce")
    available_dates = pd.to_datetime(df["msvol_signal_available_after_close_date"], errors="coerce")
    trade_dates = pd.to_datetime(df["msvol_signal_trade_date"], errors="coerce")

    _add_check(
        rows,
        market,
        "date_columns_parse",
        not dates.isna().any() and not obs_dates.isna().any() and not available_dates.isna().any(),
        "error",
        (
            f"bad_date={int(dates.isna().sum())}, "
            f"bad_observation={int(obs_dates.isna().sum())}, "
            f"bad_available={int(available_dates.isna().sum())}"
        ),
    )

    _add_check(
        rows,
        market,
        "dates_sorted_unique",
        dates.is_monotonic_increasing and not dates.duplicated().any(),
        "error",
        f"is_sorted={dates.is_monotonic_increasing}, duplicate_dates={int(dates.duplicated().sum())}",
    )

    _add_check(
        rows,
        market,
        "observation_date_equals_row_date",
        bool((dates == obs_dates).all()),
        "error",
        "MSVOL signal at row t must be observed using row date t.",
    )

    _add_check(
        rows,
        market,
        "available_after_close_not_before_observation",
        bool((available_dates >= obs_dates).all()),
        "error",
        "Signal availability cannot occur before observation date.",
    )

    _add_check(
        rows,
        market,
        "available_after_close_equals_observation_date",
        bool((available_dates == obs_dates).all()),
        "error",
        "Daily close signal should be available after the same date's close.",
    )

    expected_trade_dates = dates.shift(-1)

    if len(df) > 1:
        non_last_ok = bool((trade_dates.iloc[:-1].reset_index(drop=True) == expected_trade_dates.iloc[:-1].reset_index(drop=True)).all())
        trade_after_observation = bool((trade_dates.iloc[:-1].reset_index(drop=True) > dates.iloc[:-1].reset_index(drop=True)).all())
    else:
        non_last_ok = True
        trade_after_observation = True

    last_missing = bool(pd.isna(trade_dates.iloc[-1]))

    _add_check(
        rows,
        market,
        "trade_date_is_next_available_row_date",
        non_last_ok and last_missing,
        "error",
        (
            "For rows except the last, trade date must equal the next row's date; "
            "last row must have missing trade date."
        ),
    )

    _add_check(
        rows,
        market,
        "trade_date_strictly_after_observation_for_tradable_rows",
        trade_after_observation,
        "error",
        "Every tradable MSVOL signal must trade strictly after the observation date.",
    )

    prob_cols = [
        "msvol_filtered_prob_calm_for_next_session",
        "msvol_filtered_prob_transition_for_next_session",
        "msvol_filtered_prob_stress_for_next_session",
    ]

    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce")
    prob_sum = probs.sum(axis=1)

    _add_check(
        rows,
        market,
        "filtered_probabilities_are_finite",
        bool(np.isfinite(probs.to_numpy(dtype=float)).all()),
        "error",
        "Filtered next-session probabilities must be finite.",
    )

    _add_check(
        rows,
        market,
        "filtered_probabilities_sum_to_one",
        bool((np.abs(prob_sum.to_numpy(dtype=float) - 1.0) <= 1e-5).all()),
        "error",
        f"max_abs_error={float(np.abs(prob_sum - 1.0).max())}",
    )

    _add_check(
        rows,
        market,
        "transition_probability_zero",
        bool((pd.to_numeric(df["msvol_filtered_prob_transition_for_next_session"], errors="coerce") == 0.0).all()),
        "error",
        "MSVOL has exactly two states, so transition probability must be 0.",
    )

    transition_modelled = df["msvol_transition_state_modelled"].map(_as_bool)
    _add_check(
        rows,
        market,
        "transition_state_not_modelled",
        bool((transition_modelled == False).all()),
        "error",
        "MSVOL must remain a calm/stress two-state appendix model.",
    )

    smoothed_next_session_cols = [
        col
        for col in df.columns
        if "smoothed" in col.lower() and "for_next_session" in col.lower()
    ]
    _add_check(
        rows,
        market,
        "no_smoothed_probability_next_session_columns",
        len(smoothed_next_session_cols) == 0,
        "error",
        f"forbidden_columns={smoothed_next_session_cols}",
    )

    smoothed_cols = [col for col in df.columns if "smoothed" in col.lower()]
    bad_smoothed_names = [col for col in smoothed_cols if "diagnostic" not in col.lower()]
    _add_check(
        rows,
        market,
        "smoothed_columns_are_diagnostic_only",
        len(bad_smoothed_names) == 0,
        "error",
        f"bad_smoothed_columns={bad_smoothed_names}",
    )

    strategy_like_columns = [
        col
        for col in df.columns
        if any(fragment in col.lower() for fragment in FORBIDDEN_STRATEGY_FRAGMENTS)
    ]
    _add_check(
        rows,
        market,
        "processed_panel_contains_no_strategy_or_backtest_columns",
        len(strategy_like_columns) == 0,
        "error",
        f"strategy_like_columns={strategy_like_columns}",
    )

    meta_strategy = _as_bool(metadata.get("used_for_strategy"))
    meta_backtest = _as_bool(metadata.get("used_for_backtest"))
    meta_true_msgarch = _as_bool(metadata.get("true_msgarch"))

    _add_check(
        rows,
        market,
        "metadata_marks_not_used_for_strategy",
        meta_strategy is False,
        "warning" if metadata else "warning",
        f"used_for_strategy={metadata.get('used_for_strategy')}",
    )

    _add_check(
        rows,
        market,
        "metadata_marks_not_used_for_backtest",
        meta_backtest is False,
        "warning" if metadata else "warning",
        f"used_for_backtest={metadata.get('used_for_backtest')}",
    )

    _add_check(
        rows,
        market,
        "metadata_marks_not_true_msgarch",
        meta_true_msgarch is False,
        "warning" if metadata else "warning",
        f"true_msgarch={metadata.get('true_msgarch')}",
    )

    if comparison_summary is not None and not comparison_summary.empty:
        strategy_values = comparison_summary.get("used_for_strategy", pd.Series(dtype=object)).map(_as_bool)
        backtest_values = comparison_summary.get("used_for_backtest", pd.Series(dtype=object)).map(_as_bool)
        diagnostic_values = comparison_summary.get("diagnostic_only", pd.Series(dtype=object)).map(_as_bool)

        _add_check(
            rows,
            market,
            "comparison_summary_not_used_for_strategy",
            bool((strategy_values == False).all()),
            "error",
            "MSVOL comparison summary must be diagnostic only.",
        )

        _add_check(
            rows,
            market,
            "comparison_summary_not_used_for_backtest",
            bool((backtest_values == False).all()),
            "error",
            "MSVOL comparison summary must not include backtest use.",
        )

        _add_check(
            rows,
            market,
            "comparison_summary_diagnostic_only",
            bool((diagnostic_values == True).all()),
            "error",
            "MSVOL comparison summary must be marked diagnostic_only=True.",
        )
    else:
        _add_check(
            rows,
            market,
            "comparison_summary_available",
            False,
            "warning",
            "MSVOL comparison summary missing or empty.",
        )

    diag_strategy = _as_bool(diagnostics_metadata.get("used_for_strategy"))
    diag_backtest = _as_bool(diagnostics_metadata.get("used_for_backtest"))
    diag_only = _as_bool(diagnostics_metadata.get("diagnostic_only"))

    _add_check(
        rows,
        market,
        "diagnostics_metadata_diagnostic_only",
        (diag_only is True and diag_strategy is False and diag_backtest is False),
        "warning",
        (
            f"diagnostic_only={diagnostics_metadata.get('diagnostic_only')}, "
            f"used_for_strategy={diagnostics_metadata.get('used_for_strategy')}, "
            f"used_for_backtest={diagnostics_metadata.get('used_for_backtest')}"
        ),
    )

    return pd.DataFrame(rows)


def run_msvol_no_lookahead_audit_for_market(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
    allow_skip: bool = False,
) -> MSVolNoLookaheadAuditResult:
    market = market.upper().strip()
    paths = get_report_paths(market, config, project_root)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    processed, skip_reason = _load_processed_or_empty(
        market=market,
        config=config,
        project_root=project_root,
        allow_skip=allow_skip,
    )

    metadata = load_json_if_exists(paths["metadata_json"])
    diagnostics_metadata = load_json_if_exists(paths["diagnostics_metadata_json"])

    if paths["comparison_summary_csv"].exists():
        comparison = pd.read_csv(paths["comparison_summary_csv"])
    else:
        comparison = pd.DataFrame()

    audit = build_no_lookahead_audit(
        market=market,
        processed=processed,
        metadata=metadata,
        comparison_summary=comparison,
        diagnostics_metadata=diagnostics_metadata,
        skip_reason=skip_reason,
    )

    paths["market_audit_csv"].parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(paths["market_audit_csv"], index=False)

    combined_path = update_combined_audit_table(
        market=market,
        audit=audit,
        combined_audit_csv=paths["combined_audit_csv"],
    )

    failed_error_checks = int(((audit["severity"] == "error") & (~audit["passed"].astype(bool))).sum())

    status = "ok" if failed_error_checks == 0 else "failed"

    if status == "failed" and not allow_skip:
        # Do not raise here. The script should write the audit table first.
        pass

    return MSVolNoLookaheadAuditResult(
        market=market,
        status=status,
        market_audit_csv=paths["market_audit_csv"],
        combined_audit_csv=combined_path,
        n_checks=int(len(audit)),
        n_failed_error_checks=failed_error_checks,
        skip_reason=skip_reason,
    )


def update_combined_audit_table(
    market: str,
    audit: pd.DataFrame,
    combined_audit_csv: Path,
) -> Path:
    combined_audit_csv.parent.mkdir(parents=True, exist_ok=True)

    market = market.upper().strip()

    if combined_audit_csv.exists():
        existing = pd.read_csv(combined_audit_csv)
        if "market" in existing.columns:
            existing = existing[existing["market"].astype(str).str.upper() != market].copy()
    else:
        existing = pd.DataFrame(columns=audit.columns)

    combined = pd.concat([existing, audit], ignore_index=True)
    combined = combined.sort_values(["market", "severity", "check_name"]).reset_index(drop=True)

    combined.to_csv(combined_audit_csv, index=False)
    return combined_audit_csv