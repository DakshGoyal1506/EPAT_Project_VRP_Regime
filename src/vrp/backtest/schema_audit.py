from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


APPROVED_PHASE9_STRATEGIES: tuple[str, ...] = (
    "unconditional_full",
    "threshold_hard_filter",
    "threshold_defensive",
    "hmm_prob_linear",
    "hmm_prob_linear_carry",
    "mar_prob_linear",
    "mar_prob_linear_carry",
)

REQUIRED_SIGNAL_COLUMNS: tuple[str, ...] = (
    "market",
    "strategy_name",
    "signal_observation_date",
    "target_trade_date",
    "target_exposure",
    "strategy_available",
)

SIGNAL_DUPLICATE_KEY: tuple[str, ...] = (
    "market",
    "signal_observation_date",
    "target_trade_date",
    "strategy_name",
)

PRIMARY_PAYOFF_LABEL = "vrp_forward_expost_gk_label"

ACCEPTED_OUTCOME_DATE_COLUMNS: tuple[str, ...] = (
    "signal_observation_date",
    "date",
    "outcome_label_date",
)

SUPPORTED_MARKETS: tuple[str, ...] = ("US", "INDIA")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    market: str
    component: str
    message: str


@dataclass(frozen=True)
class MarketInputPaths:
    market: str
    strategy_signals: Path
    vrp_har: Path
    vrp: Path

    def outcome_candidates(self) -> dict[str, Path]:
        return {
            "vrp_har": self.vrp_har,
            "vrp": self.vrp,
        }


@dataclass
class MarketAuditResult:
    market: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


def default_market_input_paths(repo_root: Path, market: str) -> MarketInputPaths:
    market = market.upper()
    if market not in SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported market: {market}. Expected one of {SUPPORTED_MARKETS}.")

    prefix = market.lower()
    processed_dir = repo_root / "data" / "processed"

    return MarketInputPaths(
        market=market,
        strategy_signals=processed_dir / f"{prefix}_strategy_signals.parquet",
        vrp_har=processed_dir / f"{prefix}_vrp_har.parquet",
        vrp=processed_dir / f"{prefix}_vrp.parquet",
    )


def resolve_markets(market: str) -> list[str]:
    market = market.upper()
    if market == "ALL":
        return list(SUPPORTED_MARKETS)
    if market not in SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported market: {market}. Expected US, INDIA, or ALL.")
    return [market]


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension for {path}. Use .parquet or .csv.")


def missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
    return [col for col in required_columns if col not in df.columns]


def find_outcome_date_column(df: pd.DataFrame) -> str | None:
    for col in ACCEPTED_OUTCOME_DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def _add_issue(
    result: MarketAuditResult,
    *,
    severity: str,
    component: str,
    message: str,
) -> None:
    if severity not in {"error", "warning", "info"}:
        raise ValueError(f"Invalid audit severity: {severity}")

    result.issues.append(
        AuditIssue(
            severity=severity,
            market=result.market,
            component=component,
            message=message,
        )
    )


def _add_row(
    result: MarketAuditResult,
    *,
    component: str,
    path: Path,
    exists: bool,
    n_rows: int | None = None,
    n_columns: int | None = None,
    validation_status: str,
    details: str = "",
) -> None:
    result.rows.append(
        {
            "market": result.market,
            "component": component,
            "path": str(path),
            "exists": exists,
            "n_rows": n_rows,
            "n_columns": n_columns,
            "validation_status": validation_status,
            "details": details,
        }
    )


def _safe_read_component(
    result: MarketAuditResult,
    *,
    component: str,
    path: Path,
) -> pd.DataFrame | None:
    path = Path(path)

    if not path.exists():
        _add_row(
            result,
            component=component,
            path=path,
            exists=False,
            validation_status="error",
            details="file does not exist",
        )
        _add_issue(
            result,
            severity="error",
            component=component,
            message=f"Required file does not exist: {path}",
        )
        return None

    try:
        df = read_table(path)
    except Exception as exc:
        _add_row(
            result,
            component=component,
            path=path,
            exists=True,
            validation_status="error",
            details=f"failed to read file: {exc}",
        )
        _add_issue(
            result,
            severity="error",
            component=component,
            message=f"Failed to read {path}: {exc}",
        )
        return None

    _add_row(
        result,
        component=component,
        path=path,
        exists=True,
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        validation_status="read_ok",
        details="file read successfully",
    )
    return df


def audit_signal_frame(
    df: pd.DataFrame,
    *,
    market: str,
    result: MarketAuditResult,
    component: str = "strategy_signals",
) -> None:
    missing = missing_columns(df, REQUIRED_SIGNAL_COLUMNS)
    if missing:
        _add_issue(
            result,
            severity="error",
            component=component,
            message=f"Missing required signal columns: {missing}",
        )
        return

    strategies = sorted(df["strategy_name"].dropna().astype(str).unique().tolist())
    expected = sorted(APPROVED_PHASE9_STRATEGIES)

    missing_strategies = sorted(set(expected) - set(strategies))
    extra_strategies = sorted(set(strategies) - set(expected))

    if missing_strategies or extra_strategies:
        _add_issue(
            result,
            severity="error",
            component=component,
            message=(
                "Strategy universe mismatch. "
                f"Missing={missing_strategies}; Extra={extra_strategies}"
            ),
        )

    msvol_mask = df["strategy_name"].astype(str).str.contains("msvol", case=False, na=False)
    n_msvol = int(msvol_mask.sum())
    if n_msvol > 0:
        _add_issue(
            result,
            severity="error",
            component=component,
            message=f"MSVOL strategy rows are forbidden in Phase 10. Found {n_msvol} rows.",
        )

    duplicate_missing = missing_columns(df, SIGNAL_DUPLICATE_KEY)
    if duplicate_missing:
        _add_issue(
            result,
            severity="error",
            component=component,
            message=f"Cannot run duplicate-key check. Missing columns: {duplicate_missing}",
        )
    else:
        duplicate_mask = df.duplicated(list(SIGNAL_DUPLICATE_KEY), keep=False)
        n_duplicates = int(duplicate_mask.sum())
        if n_duplicates > 0:
            _add_issue(
                result,
                severity="error",
                component=component,
                message=(
                    "Duplicate signal rows found by key "
                    f"{list(SIGNAL_DUPLICATE_KEY)}. Duplicate row count={n_duplicates}."
                ),
            )

    exposure_numeric = pd.to_numeric(df["target_exposure"], errors="coerce")
    n_non_finite_exposure = int(exposure_numeric.isna().sum())

    n_available = int(df["strategy_available"].fillna(False).astype(bool).sum())
    n_rows = int(len(df))

    result.rows.append(
        {
            "market": market,
            "component": f"{component}_schema",
            "path": "",
            "exists": True,
            "n_rows": n_rows,
            "n_columns": int(len(df.columns)),
            "validation_status": "checked",
            "details": json.dumps(
                {
                    "n_unique_strategies": len(strategies),
                    "strategies": strategies,
                    "n_available_rows": n_available,
                    "n_non_finite_target_exposure": n_non_finite_exposure,
                    "duplicate_key": list(SIGNAL_DUPLICATE_KEY),
                },
                sort_keys=True,
            ),
        }
    )


def audit_outcome_frame(
    df: pd.DataFrame,
    *,
    market: str,
    result: MarketAuditResult,
    component: str,
    label_col: str = PRIMARY_PAYOFF_LABEL,
) -> bool:
    has_label = label_col in df.columns
    date_col = find_outcome_date_column(df)

    if not has_label:
        _add_issue(
            result,
            severity="warning",
            component=component,
            message=f"Outcome label {label_col!r} not found in {component}.",
        )

    if date_col is None:
        _add_issue(
            result,
            severity="warning",
            component=component,
            message=(
                f"No accepted outcome date column found in {component}. "
                f"Accepted columns: {list(ACCEPTED_OUTCOME_DATE_COLUMNS)}"
            ),
        )

    n_missing_label = None
    if has_label:
        label_numeric = pd.to_numeric(df[label_col], errors="coerce")
        n_missing_label = int(label_numeric.isna().sum())

    n_duplicate_dates = None
    if date_col is not None:
        duplicate_key = ["market", date_col] if "market" in df.columns else [date_col]
        n_duplicate_dates = int(df.duplicated(duplicate_key, keep=False).sum())
        if n_duplicate_dates > 0:
            _add_issue(
                result,
                severity="error",
                component=component,
                message=(
                    f"Duplicate outcome rows found by key {duplicate_key}. "
                    f"Duplicate row count={n_duplicate_dates}."
                ),
            )

    result.rows.append(
        {
            "market": market,
            "component": f"{component}_schema",
            "path": "",
            "exists": True,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "validation_status": "checked",
            "details": json.dumps(
                {
                    "has_payoff_label": has_label,
                    "payoff_label": label_col,
                    "outcome_date_column": date_col,
                    "n_missing_payoff_label": n_missing_label,
                    "n_duplicate_outcome_dates": n_duplicate_dates,
                },
                sort_keys=True,
            ),
        }
    )

    return has_label and date_col is not None


def audit_market_inputs(paths: MarketInputPaths) -> MarketAuditResult:
    market = paths.market.upper()
    result = MarketAuditResult(market=market)

    signal_df = _safe_read_component(
        result,
        component="strategy_signals",
        path=paths.strategy_signals,
    )
    if signal_df is not None:
        audit_signal_frame(
            signal_df,
            market=market,
            result=result,
            component="strategy_signals",
        )

    valid_outcome_candidates: list[str] = []
    for component, path in paths.outcome_candidates().items():
        outcome_df = _safe_read_component(
            result,
            component=component,
            path=path,
        )
        if outcome_df is None:
            continue

        is_valid_outcome = audit_outcome_frame(
            outcome_df,
            market=market,
            result=result,
            component=component,
            label_col=PRIMARY_PAYOFF_LABEL,
        )
        if is_valid_outcome:
            valid_outcome_candidates.append(component)

    if not valid_outcome_candidates:
        _add_issue(
            result,
            severity="error",
            component="outcome_candidates",
            message=(
                f"No valid outcome candidate contains both a usable date column and "
                f"{PRIMARY_PAYOFF_LABEL!r}."
            ),
        )
    else:
        result.rows.append(
            {
                "market": market,
                "component": "selected_outcome_candidates",
                "path": "",
                "exists": True,
                "n_rows": None,
                "n_columns": None,
                "validation_status": "checked",
                "details": json.dumps(
                    {"valid_outcome_candidates": valid_outcome_candidates},
                    sort_keys=True,
                ),
            }
        )

    return result


def audit_phase10_inputs(
    repo_root: Path,
    *,
    market: str = "ALL",
) -> list[MarketAuditResult]:
    repo_root = Path(repo_root)
    markets = resolve_markets(market)

    results: list[MarketAuditResult] = []
    for market_code in markets:
        paths = default_market_input_paths(repo_root, market_code)
        results.append(audit_market_inputs(paths))

    return results


def write_audit_reports(
    results: Sequence[MarketAuditResult],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for result in results:
        rows.extend(result.rows)
        issues.extend(asdict(issue) for issue in result.issues)

    csv_path = output_dir / "phase10_input_audit.csv"
    json_path = output_dir / "phase10_input_audit.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "has_errors": any(result.has_errors() for result in results),
        "n_errors": sum(result.error_count() for result in results),
        "n_warnings": sum(result.warning_count() for result in results),
        "rows": rows,
        "issues": issues,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "csv": csv_path,
        "json": json_path,
    }


def render_audit_summary(results: Sequence[MarketAuditResult]) -> str:
    lines: list[str] = []

    for result in results:
        status = "FAILED" if result.has_errors() else "OK"
        lines.append(
            f"[{result.market}] {status} "
            f"errors={result.error_count()} warnings={result.warning_count()}"
        )

        for issue in result.issues:
            lines.append(
                f"  - {issue.severity.upper()} "
                f"{issue.component}: {issue.message}"
            )

    if not lines:
        return "No audit results."

    return "\n".join(lines)


def assert_no_audit_errors(results: Sequence[MarketAuditResult]) -> None:
    messages: list[str] = []

    for result in results:
        for issue in result.issues:
            if issue.severity == "error":
                messages.append(f"[{issue.market}] {issue.component}: {issue.message}")

    if messages:
        raise AssertionError("Phase 10 input audit failed:\n" + "\n".join(messages))