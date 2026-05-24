from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from vrp.backtest.backtest_config import BacktestConfig, get_market_output_path
from vrp.backtest.backtest_registry import (
    BACKTEST_FORBIDDEN_SIGNAL_COLUMNS,
    BACKTEST_STRATEGY_UNIVERSE,
    assert_no_msvol_strategy_use,
    assert_no_smoothed_probability_use,
    assert_strategy_universe_locked,
)
from vrp.backtest.vectorized_engine import resolve_markets


class Phase10FinalAuditError(ValueError):
    """Raised when the final Phase 10 audit cannot be completed."""


@dataclass(frozen=True)
class FinalAuditIssue:
    severity: str
    component: str
    message: str


@dataclass(frozen=True)
class FinalAuditResult:
    status: str
    n_errors: int
    n_warnings: int
    issues: list[FinalAuditIssue]
    generated_at_utc: str

    def has_errors(self) -> bool:
        return self.n_errors > 0


REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    "market",
    "strategy_name",
    "signal_observation_date",
    "target_trade_date",
    "outcome_label_date",
    "target_exposure",
    "target_exposure_for_backtest",
    "strategy_available",
    "vrp_forward_expost_gk_label",
    "gross_return_proxy",
    "delta_exposure",
    "cost_proxy",
    "net_return_proxy",
    "is_backtest_eligible",
    "exclusion_reason",
)

REQUIRED_PER_MARKET_METADATA_KEYS: tuple[str, ...] = (
    "phase",
    "payoff_type",
    "payoff_label",
    "label_role",
    "outcome_alignment",
    "cost_bps",
    "annualization_periods",
    "horizon_trading_days",
    "overlapping_labels",
    "research_proxy_not_trade_pnl",
    "strategy_universe_locked",
    "n_target_not_after_signal_violations",
    "n_outcome_not_equal_signal_date_violations",
)

REQUIRED_DIAGNOSTIC_TABLES: tuple[str, ...] = (
    "backtest_summary.csv",
    "backtest_common_start_summary.csv",
    "backtest_tail_summary.csv",
    "backtest_by_strategy_year.csv",
    "crisis_window_performance.csv",
    "backtest_availability_summary.csv",
    "backtest_no_lookahead_audit.csv",
    "backtest_metadata.json",
)

REQUIRED_DIAGNOSTIC_FIGURE_PATTERNS: tuple[str, ...] = (
    "equity_curves_{market}.png",
    "equity_curves_common_start_{market}.png",
    "drawdowns_{market}.png",
    "return_distribution_{market}.png",
)

REQUIRED_ROBUSTNESS_OUTPUTS: tuple[str, ...] = (
    "robustness_cost_sensitivity.csv",
    "robustness_subperiods.csv",
    "robustness_weekly_rebalance_skipped.json",
    "tradable_proxy_detection.json",
    "robustness_metadata.json",
)

REQUIRED_REPORT_METADATA_KEYS: tuple[str, ...] = (
    "phase",
    "markets",
    "payoff_type",
    "payoff_label",
    "label_role",
    "outcome_alignment",
    "annualization_periods",
    "horizon_trading_days",
    "overlapping_labels",
    "research_proxy_not_trade_pnl",
    "visual_interpretation_warning",
    "common_start_dates",
    "no_lookahead_audit_passed",
    "limitations",
)

REQUIRED_ROBUSTNESS_RULE_FLAGS: tuple[str, ...] = (
    "no_new_data_downloads",
    "no_new_strategies",
    "no_msvol_strategy_use",
    "tradable_proxy_detection_only",
    "weekly_rebalance_skip_safe_by_default",
    "research_proxy_not_trade_pnl",
    "overlapping_labels",
)


def _resolve_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(float(value)):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    return value


def _add_issue(
    issues: list[FinalAuditIssue],
    *,
    severity: str,
    component: str,
    message: str,
) -> None:
    if severity not in {"error", "warning", "info"}:
        raise Phase10FinalAuditError(f"Invalid severity: {severity}")

    issues.append(
        FinalAuditIssue(
            severity=severity,
            component=component,
            message=message,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase10FinalAuditError(f"Failed to read JSON {path}: {exc}") from exc


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise Phase10FinalAuditError(f"Backtest panel missing: {path}")

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise Phase10FinalAuditError(f"Failed to read panel {path}: {exc}") from exc


def audit_backtest_panel(
    panel: pd.DataFrame,
    *,
    market: str,
    issues: list[FinalAuditIssue],
) -> None:
    component = f"{market.lower()}_backtest_panel"

    missing = sorted(set(REQUIRED_PANEL_COLUMNS) - set(panel.columns))
    if missing:
        _add_issue(
            issues,
            severity="error",
            component=component,
            message=f"Missing required panel columns: {missing}",
        )
        return

    strategies = sorted(panel["strategy_name"].dropna().astype(str).unique().tolist())
    try:
        assert_strategy_universe_locked(strategies)
    except Exception as exc:
        _add_issue(
            issues,
            severity="error",
            component=component,
            message=f"Strategy universe mismatch: {exc}",
        )

    try:
        assert_no_msvol_strategy_use(panel)
    except Exception as exc:
        _add_issue(
            issues,
            severity="error",
            component=component,
            message=f"MSVOL strategy violation: {exc}",
        )

    try:
        assert_no_smoothed_probability_use(panel)
    except Exception as exc:
        _add_issue(
            issues,
            severity="error",
            component=component,
            message=f"Smoothed probability violation: {exc}",
        )

    forbidden_signal_cols = sorted(
        set(panel.columns).intersection(set(BACKTEST_FORBIDDEN_SIGNAL_COLUMNS))
    )
    if "vrp_forward_expost_gk_label" not in forbidden_signal_cols:
        _add_issue(
            issues,
            severity="warning",
            component=component,
            message="Payoff label is not visible in panel columns; expected realised outcome label.",
        )

    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    if eligible.any():
        signal_dates = pd.to_datetime(
            panel.loc[eligible, "signal_observation_date"],
            errors="coerce",
        )
        trade_dates = pd.to_datetime(
            panel.loc[eligible, "target_trade_date"],
            errors="coerce",
        )
        outcome_dates = pd.to_datetime(
            panel.loc[eligible, "outcome_label_date"],
            errors="coerce",
        )

        n_bad_trade_dates = int((trade_dates <= signal_dates).sum())
        if n_bad_trade_dates > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=(
                    "No-lookahead violation: eligible target_trade_date must be "
                    f"after signal_observation_date. Count={n_bad_trade_dates}."
                ),
            )

        n_bad_outcome_dates = int((outcome_dates != signal_dates).sum())
        if n_bad_outcome_dates > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=(
                    "Outcome alignment violation: outcome_label_date should equal "
                    f"signal_observation_date. Count={n_bad_outcome_dates}."
                ),
            )

        n_missing_net = int(
            pd.to_numeric(
                panel.loc[eligible, "net_return_proxy"],
                errors="coerce",
            ).isna().sum()
        )
        if n_missing_net > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=f"Eligible rows contain missing net_return_proxy. Count={n_missing_net}.",
            )

        n_missing_exposure = int(
            pd.to_numeric(
                panel.loc[eligible, "target_exposure_for_backtest"],
                errors="coerce",
            ).isna().sum()
        )
        if n_missing_exposure > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=(
                    "Eligible rows contain missing target_exposure_for_backtest. "
                    f"Count={n_missing_exposure}."
                ),
            )

        exposure = pd.to_numeric(
            panel.loc[eligible, "target_exposure_for_backtest"],
            errors="coerce",
        )
        label = pd.to_numeric(
            panel.loc[eligible, "vrp_forward_expost_gk_label"],
            errors="coerce",
        )
        gross = pd.to_numeric(
            panel.loc[eligible, "gross_return_proxy"],
            errors="coerce",
        )
        cost = pd.to_numeric(
            panel.loc[eligible, "cost_proxy"],
            errors="coerce",
        )
        net = pd.to_numeric(
            panel.loc[eligible, "net_return_proxy"],
            errors="coerce",
        )

        expected_gross = -exposure * label
        expected_net = gross - cost.fillna(0.0)

        n_bad_gross = int(
            (~np.isclose(
                gross.to_numpy(dtype=float),
                expected_gross.to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            )).sum()
        )
        if n_bad_gross > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=(
                    "Payoff identity violation: gross_return_proxy must equal "
                    "-target_exposure_for_backtest * vrp_forward_expost_gk_label. "
                    f"Count={n_bad_gross}."
                ),
            )

        n_bad_net = int(
            (~np.isclose(
                net.to_numpy(dtype=float),
                expected_net.to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            )).sum()
        )
        if n_bad_net > 0:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=(
                    "Net return identity violation: net_return_proxy must equal "
                    "gross_return_proxy - cost_proxy. "
                    f"Count={n_bad_net}."
                ),
            )

    ineligible = ~eligible
    if ineligible.any():
        n_bad_ineligible_exposure = int(
            pd.to_numeric(
                panel.loc[ineligible, "target_exposure_for_backtest"],
                errors="coerce",
            ).notna().sum()
        )
        if n_bad_ineligible_exposure > 0:
            _add_issue(
                issues,
                severity="warning",
                component=component,
                message=(
                    "Ineligible rows contain non-null target_exposure_for_backtest. "
                    f"Count={n_bad_ineligible_exposure}."
                ),
            )


def audit_per_market_metadata(
    metadata: dict[str, Any],
    *,
    market: str,
    issues: list[FinalAuditIssue],
) -> None:
    component = f"{market.lower()}_metadata"

    missing = sorted(set(REQUIRED_PER_MARKET_METADATA_KEYS) - set(metadata))
    if missing:
        _add_issue(
            issues,
            severity="error",
            component=component,
            message=f"Missing required metadata keys: {missing}",
        )
        return

    expected_values = {
        "phase": "phase_10",
        "payoff_label": "vrp_forward_expost_gk_label",
        "label_role": "realised_outcome_only",
        "outcome_alignment": "signal_observation_date",
        "horizon_trading_days": 22,
        "overlapping_labels": True,
        "research_proxy_not_trade_pnl": True,
        "strategy_universe_locked": True,
        "n_target_not_after_signal_violations": 0,
        "n_outcome_not_equal_signal_date_violations": 0,
    }

    for key, expected in expected_values.items():
        actual = metadata.get(key)
        if actual != expected:
            _add_issue(
                issues,
                severity="error",
                component=component,
                message=f"Metadata key {key!r} expected {expected!r}, got {actual!r}.",
            )


def audit_diagnostic_outputs(
    *,
    config: BacktestConfig,
    repo_root: Path,
    markets: list[str],
    issues: list[FinalAuditIssue],
) -> None:
    table_dir = _resolve_path(repo_root, config.reporting["table_dir"])
    figure_dir = _resolve_path(repo_root, config.reporting["figure_dir"])

    for filename in REQUIRED_DIAGNOSTIC_TABLES:
        path = table_dir / filename
        if not path.exists():
            _add_issue(
                issues,
                severity="error",
                component="diagnostic_tables",
                message=f"Missing diagnostic table: {path}",
            )
            continue

        if path.stat().st_size == 0:
            _add_issue(
                issues,
                severity="error",
                component="diagnostic_tables",
                message=f"Diagnostic table is empty: {path}",
            )

    for market in markets:
        market_lower = market.lower()
        for pattern in REQUIRED_DIAGNOSTIC_FIGURE_PATTERNS:
            path = figure_dir / pattern.format(market=market_lower)
            if not path.exists():
                _add_issue(
                    issues,
                    severity="error",
                    component="diagnostic_figures",
                    message=f"Missing diagnostic figure: {path}",
                )
                continue

            if path.stat().st_size == 0:
                _add_issue(
                    issues,
                    severity="error",
                    component="diagnostic_figures",
                    message=f"Diagnostic figure is empty: {path}",
                )

    report_metadata_path = table_dir / "backtest_metadata.json"
    if report_metadata_path.exists():
        metadata = _read_json(report_metadata_path)
        missing = sorted(set(REQUIRED_REPORT_METADATA_KEYS) - set(metadata))
        if missing:
            _add_issue(
                issues,
                severity="error",
                component="backtest_metadata",
                message=f"Missing report metadata keys: {missing}",
            )

        if metadata.get("no_lookahead_audit_passed") is not True:
            _add_issue(
                issues,
                severity="error",
                component="backtest_metadata",
                message="no_lookahead_audit_passed is not true.",
            )

        if metadata.get("research_proxy_not_trade_pnl") is not True:
            _add_issue(
                issues,
                severity="error",
                component="backtest_metadata",
                message="research_proxy_not_trade_pnl is not true.",
            )

        if metadata.get("overlapping_labels") is not True:
            _add_issue(
                issues,
                severity="error",
                component="backtest_metadata",
                message="overlapping_labels is not true.",
            )

        warning = str(metadata.get("visual_interpretation_warning", ""))
        if "not executable account equity curves" not in warning:
            _add_issue(
                issues,
                severity="error",
                component="backtest_metadata",
                message="visual_interpretation_warning does not state account-equity caveat.",
            )


def audit_robustness_outputs(
    *,
    config: BacktestConfig,
    repo_root: Path,
    issues: list[FinalAuditIssue],
) -> None:
    table_dir = _resolve_path(repo_root, config.reporting["table_dir"])

    for filename in REQUIRED_ROBUSTNESS_OUTPUTS:
        path = table_dir / filename
        if not path.exists():
            _add_issue(
                issues,
                severity="error",
                component="robustness_outputs",
                message=f"Missing robustness output: {path}",
            )
            continue

        if path.stat().st_size == 0:
            _add_issue(
                issues,
                severity="error",
                component="robustness_outputs",
                message=f"Robustness output is empty: {path}",
            )

    metadata_path = table_dir / "robustness_metadata.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        rules = metadata.get("rules", {})

        for key in REQUIRED_ROBUSTNESS_RULE_FLAGS:
            if rules.get(key) is not True:
                _add_issue(
                    issues,
                    severity="error",
                    component="robustness_metadata",
                    message=f"Robustness rule flag {key!r} is not true.",
                )

    tradable_proxy_path = table_dir / "tradable_proxy_detection.json"
    if tradable_proxy_path.exists():
        detection = _read_json(tradable_proxy_path)
        status = detection.get("status")
        if status not in {"skipped", "available"}:
            _add_issue(
                issues,
                severity="error",
                component="tradable_proxy_detection",
                message=f"Unexpected tradable proxy status: {status!r}.",
            )


def run_phase10_final_audit(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
    require_robustness: bool = True,
) -> FinalAuditResult:
    repo_root = Path(repo_root)
    markets = resolve_markets(market)

    issues: list[FinalAuditIssue] = []

    for market_code in markets:
        panel_path = _resolve_path(
            repo_root,
            get_market_output_path(config, market_code, "backtest_panel"),
        )
        metadata_path = _resolve_path(
            repo_root,
            get_market_output_path(config, market_code, "metadata"),
        )

        if not panel_path.exists():
            _add_issue(
                issues,
                severity="error",
                component=f"{market_code.lower()}_backtest_panel",
                message=f"Missing panel: {panel_path}",
            )
        else:
            panel = _read_panel(panel_path)
            audit_backtest_panel(panel, market=market_code, issues=issues)

        if not metadata_path.exists():
            _add_issue(
                issues,
                severity="error",
                component=f"{market_code.lower()}_metadata",
                message=f"Missing metadata sidecar: {metadata_path}",
            )
        else:
            metadata = _read_json(metadata_path)
            audit_per_market_metadata(metadata, market=market_code, issues=issues)

    audit_diagnostic_outputs(
        config=config,
        repo_root=repo_root,
        markets=markets,
        issues=issues,
    )

    if require_robustness:
        audit_robustness_outputs(
            config=config,
            repo_root=repo_root,
            issues=issues,
        )

    n_errors = sum(issue.severity == "error" for issue in issues)
    n_warnings = sum(issue.severity == "warning" for issue in issues)

    return FinalAuditResult(
        status="failed" if n_errors else "passed",
        n_errors=int(n_errors),
        n_warnings=int(n_warnings),
        issues=issues,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


def write_final_audit_report(
    result: FinalAuditResult,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return output_path


def render_final_audit_summary(result: FinalAuditResult) -> str:
    lines = [
        f"Phase 10 final audit: {result.status.upper()} "
        f"errors={result.n_errors} warnings={result.n_warnings}"
    ]

    for issue in result.issues:
        lines.append(
            f"  - {issue.severity.upper()} {issue.component}: {issue.message}"
        )

    return "\n".join(lines)


def assert_final_audit_passed(result: FinalAuditResult) -> None:
    if result.has_errors():
        raise AssertionError(render_final_audit_summary(result))