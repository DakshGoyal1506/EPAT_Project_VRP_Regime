"""
Final Phase 11 integration checks.

These checks validate the artifacts already written by the Phase 11 CLI.

They do not create signals.
They do not create paper intents.
They do not connect to a broker.
They do not place orders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import BrokerConfig, get_output_paths


TERMINAL_NO_INTENT_STATUSES = {
    "BLOCKED_MISSING_SIGNAL",
    "BLOCKED_STALE_SIGNAL",
    "NO_SIGNAL",
    "STAY_FLAT",
}

FINAL_STATUS_PRIORITY_FROM_INTENT = {
    "ALLOWED_PAPER_INTENT",
    "BLOCKED_BY_KILL_SWITCH",
    "BLOCKED_CONFIG_SAFETY",
    "BLOCKED_RISK_LIMIT",
    "BLOCKED_BROKER_DATA",
    "BLOCKED_STALE_SIGNAL",
}

REQUIRED_DAILY_SIGNAL_COLUMNS = {
    "market",
    "strategy_name",
    "recommended_action",
    "target_exposure",
    "paper_only",
    "kill_switch",
    "live_orders_enabled",
    "allow_order_placement",
    "final_status",
    "live_order_sent",
    "research_proxy_warning",
}

REQUIRED_INTENT_COLUMNS = {
    "market",
    "strategy_name",
    "recommended_action",
    "symbol",
    "side",
    "target_exposure",
    "paper_target_notional",
    "final_status",
    "live_order_sent",
    "research_proxy_warning",
}

REQUIRED_RISK_COLUMNS = {
    "market",
    "symbol",
    "recommended_action",
    "final_status",
    "check_name",
    "status",
    "blocks_intent",
    "reason",
}

REQUIRED_BROKER_METADATA_KEYS = {
    "ibridgepy_available",
    "broker_connection_attempted",
    "broker_connection_status",
    "broker_data_status",
    "paper_only",
    "kill_switch",
    "live_orders_enabled",
    "allow_order_placement",
    "live_order_sent",
    "research_proxy_warning",
}

REQUIRED_RUN_METADATA_KEYS = {
    "market",
    "strategy",
    "config_hash",
    "daily_signal_final_status",
    "paper_intent_final_status",
    "final_status",
    "paper_only",
    "kill_switch",
    "live_orders_enabled",
    "allow_order_placement",
    "live_order_sent",
    "research_proxy_warning",
}


class Phase11IntegrationError(RuntimeError):
    """Raised when final Phase 11 integration checks fail."""


@dataclass(frozen=True)
class IntegrationViolation:
    """One artifact-level integration violation."""

    check_name: str
    artifact: str
    severity: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "check_name": self.check_name,
            "artifact": self.artifact,
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Phase11IntegrationReport:
    """Final Phase 11 integration report."""

    passed: bool
    output_root: str
    violations: tuple[IntegrationViolation, ...]
    artifacts_checked: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_root": self.output_root,
            "violations": [violation.as_dict() for violation in self.violations],
            "artifacts_checked": list(self.artifacts_checked),
        }


def check_phase11_artifacts(
    config: BrokerConfig,
    *,
    output_root: str | Path | None = None,
) -> Phase11IntegrationReport:
    """Validate final Phase 11 output artifacts."""

    output_paths = get_output_paths(config)

    if output_root is None:
        root = Path(output_paths.get("tables_dir", "reports/tables/phase_11"))
    else:
        root = Path(output_root)

    paths = {
        "daily_signal": Path(output_paths["latest_signal_table"]),
        "paper_intents": Path(output_paths["paper_order_intents"]),
        "risk_report": Path(output_paths["risk_check_report"]),
        "broker_metadata": Path(output_paths["broker_metadata"]),
        "run_metadata": Path(output_paths["run_metadata"]),
        "config_snapshot": Path(output_paths["config_snapshot"]),
    }

    violations: list[IntegrationViolation] = []
    artifacts_checked: list[str] = []

    for artifact_name, artifact_path in paths.items():
        artifacts_checked.append(str(artifact_path))
        if not artifact_path.exists():
            violations.append(
                IntegrationViolation(
                    check_name="artifact_exists",
                    artifact=artifact_name,
                    severity="ERROR",
                    reason=f"missing artifact: {artifact_path}",
                )
            )

    if violations:
        return Phase11IntegrationReport(
            passed=False,
            output_root=str(root),
            violations=tuple(violations),
            artifacts_checked=tuple(artifacts_checked),
        )

    daily_signal = _read_csv(paths["daily_signal"], artifact="daily_signal")
    paper_intents = _read_csv(paths["paper_intents"], artifact="paper_intents")
    risk_report = _read_csv(paths["risk_report"], artifact="risk_report")
    broker_metadata = _read_json(paths["broker_metadata"], artifact="broker_metadata")
    run_metadata = _read_json(paths["run_metadata"], artifact="run_metadata")

    _check_daily_signal(daily_signal, violations)
    _check_paper_intents(paper_intents, violations)
    _check_risk_report(risk_report, paper_intents, violations)
    _check_broker_metadata(broker_metadata, config, violations)
    _check_run_metadata(run_metadata, config, violations)
    _check_cross_artifact_consistency(
        daily_signal=daily_signal,
        paper_intents=paper_intents,
        risk_report=risk_report,
        broker_metadata=broker_metadata,
        run_metadata=run_metadata,
        violations=violations,
    )
    _check_config_snapshot(paths["config_snapshot"], config, violations)

    return Phase11IntegrationReport(
        passed=len(violations) == 0,
        output_root=str(root),
        violations=tuple(violations),
        artifacts_checked=tuple(artifacts_checked),
    )


def assert_phase11_artifacts_valid(
    config: BrokerConfig,
    *,
    output_root: str | Path | None = None,
) -> Phase11IntegrationReport:
    """Raise if Phase 11 artifact checks fail."""

    report = check_phase11_artifacts(config, output_root=output_root)

    if not report.passed:
        preview = "\n".join(
            f"- {v.artifact}: {v.check_name}: {v.reason}"
            for v in report.violations[:20]
        )
        raise Phase11IntegrationError(
            f"Phase 11 integration checks failed with "
            f"{len(report.violations)} violation(s).\n{preview}"
        )

    return report


def write_phase11_integration_report(
    report: Phase11IntegrationReport,
    output_path: str | Path,
) -> Path:
    """Write final Phase 11 integration report to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return path


def _check_daily_signal(
    frame: pd.DataFrame,
    violations: list[IntegrationViolation],
) -> None:
    missing = sorted(REQUIRED_DAILY_SIGNAL_COLUMNS.difference(frame.columns))
    if missing:
        _add(
            violations,
            "daily_signal_schema",
            "daily_signal",
            f"missing columns: {missing}",
        )
        return

    if len(frame) != 1:
        _add(
            violations,
            "daily_signal_row_count",
            "daily_signal",
            f"expected exactly one row, got {len(frame)}",
        )
        return

    row = frame.iloc[0]

    if _boolish(row["live_order_sent"]):
        _add(
            violations,
            "daily_signal_live_order_sent",
            "daily_signal",
            "daily signal has live_order_sent=true",
        )

    if not _boolish(row["paper_only"]):
        _add(
            violations,
            "daily_signal_paper_only",
            "daily_signal",
            "paper_only must be true",
        )

    if _boolish(row["live_orders_enabled"]):
        _add(
            violations,
            "daily_signal_live_orders_enabled",
            "daily_signal",
            "live_orders_enabled must be false",
        )

    if _boolish(row["allow_order_placement"]):
        _add(
            violations,
            "daily_signal_allow_order_placement",
            "daily_signal",
            "allow_order_placement must be false",
        )

    if "research-layer proxy units" not in str(row["research_proxy_warning"]):
        _add(
            violations,
            "daily_signal_proxy_warning",
            "daily_signal",
            "research proxy warning missing or malformed",
        )


def _check_paper_intents(
    frame: pd.DataFrame,
    violations: list[IntegrationViolation],
) -> None:
    missing = sorted(REQUIRED_INTENT_COLUMNS.difference(frame.columns))
    if missing:
        _add(
            violations,
            "paper_intent_schema",
            "paper_intents",
            f"missing columns: {missing}",
        )
        return

    if frame.empty:
        return

    for index, row in frame.iterrows():
        if _boolish(row["live_order_sent"]):
            _add(
                violations,
                "paper_intent_live_order_sent",
                "paper_intents",
                f"row {index} has live_order_sent=true",
            )

        if "research-layer proxy units" not in str(row["research_proxy_warning"]):
            _add(
                violations,
                "paper_intent_proxy_warning",
                "paper_intents",
                f"row {index} missing research proxy warning",
            )

        if str(row["symbol"]).upper() in {"", "NAN"}:
            _add(
                violations,
                "paper_intent_symbol",
                "paper_intents",
                f"row {index} missing symbol",
            )

        if str(row["final_status"]) == "ALLOWED_PAPER_INTENT":
            if not _boolish(row.get("paper_only", True)):
                _add(
                    violations,
                    "paper_intent_allowed_not_paper_only",
                    "paper_intents",
                    f"row {index} allowed while paper_only is false",
                )


def _check_risk_report(
    risk_report: pd.DataFrame,
    paper_intents: pd.DataFrame,
    violations: list[IntegrationViolation],
) -> None:
    missing = sorted(REQUIRED_RISK_COLUMNS.difference(risk_report.columns))
    if missing:
        _add(
            violations,
            "risk_report_schema",
            "risk_report",
            f"missing columns: {missing}",
        )
        return

    if paper_intents.empty and not risk_report.empty:
        _add(
            violations,
            "risk_report_should_be_empty",
            "risk_report",
            "risk report must be header-only when no paper intent is written",
        )

    if not paper_intents.empty and risk_report.empty:
        _add(
            violations,
            "risk_report_should_have_checks",
            "risk_report",
            "risk report must contain checks when paper intent is written",
        )


def _check_broker_metadata(
    payload: Mapping[str, Any],
    config: BrokerConfig,
    violations: list[IntegrationViolation],
) -> None:
    missing = sorted(REQUIRED_BROKER_METADATA_KEYS.difference(payload.keys()))
    if missing:
        _add(
            violations,
            "broker_metadata_schema",
            "broker_metadata",
            f"missing keys: {missing}",
        )
        return

    if _boolish(payload["live_order_sent"]):
        _add(
            violations,
            "broker_metadata_live_order_sent",
            "broker_metadata",
            "broker metadata has live_order_sent=true",
        )

    if not _boolish(payload["paper_only"]):
        _add(
            violations,
            "broker_metadata_paper_only",
            "broker_metadata",
            "paper_only must be true",
        )

    if _boolish(payload["live_orders_enabled"]):
        _add(
            violations,
            "broker_metadata_live_orders_enabled",
            "broker_metadata",
            "live_orders_enabled must be false",
        )

    if _boolish(payload["allow_order_placement"]):
        _add(
            violations,
            "broker_metadata_allow_order_placement",
            "broker_metadata",
            "allow_order_placement must be false",
        )

    allowed_statuses = set(str(x) for x in config.raw.get("broker_status_taxonomy", ()))
    status = str(payload["broker_connection_status"])

    if allowed_statuses and status not in allowed_statuses:
        _add(
            violations,
            "broker_metadata_status_taxonomy",
            "broker_metadata",
            f"broker_connection_status not in taxonomy: {status}",
        )


def _check_run_metadata(
    payload: Mapping[str, Any],
    config: BrokerConfig,
    violations: list[IntegrationViolation],
) -> None:
    missing = sorted(REQUIRED_RUN_METADATA_KEYS.difference(payload.keys()))
    if missing:
        _add(
            violations,
            "run_metadata_schema",
            "run_metadata",
            f"missing keys: {missing}",
        )
        return

    if _boolish(payload["live_order_sent"]):
        _add(
            violations,
            "run_metadata_live_order_sent",
            "run_metadata",
            "run metadata has live_order_sent=true",
        )

    if not _boolish(payload["paper_only"]):
        _add(
            violations,
            "run_metadata_paper_only",
            "run_metadata",
            "paper_only must be true",
        )

    if _boolish(payload["live_orders_enabled"]):
        _add(
            violations,
            "run_metadata_live_orders_enabled",
            "run_metadata",
            "live_orders_enabled must be false",
        )

    if _boolish(payload["allow_order_placement"]):
        _add(
            violations,
            "run_metadata_allow_order_placement",
            "run_metadata",
            "allow_order_placement must be false",
        )

    final_status = str(payload["final_status"])
    allowed_statuses = set(str(x) for x in config.raw.get("final_status_taxonomy", ()))

    if allowed_statuses and final_status not in allowed_statuses:
        _add(
            violations,
            "run_metadata_final_status_taxonomy",
            "run_metadata",
            f"final_status not in taxonomy: {final_status}",
        )

    if "research-layer proxy units" not in str(payload["research_proxy_warning"]):
        _add(
            violations,
            "run_metadata_proxy_warning",
            "run_metadata",
            "research proxy warning missing or malformed",
        )


def _check_cross_artifact_consistency(
    *,
    daily_signal: pd.DataFrame,
    paper_intents: pd.DataFrame,
    risk_report: pd.DataFrame,
    broker_metadata: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    violations: list[IntegrationViolation],
) -> None:
    if daily_signal.empty:
        return

    daily_row = daily_signal.iloc[0]
    daily_status = str(daily_row["final_status"])
    run_status = str(run_metadata.get("final_status", ""))
    run_daily_status = str(run_metadata.get("daily_signal_final_status", ""))

    if run_daily_status != daily_status:
        _add(
            violations,
            "daily_vs_run_status",
            "run_metadata",
            f"run daily_signal_final_status={run_daily_status} differs from daily final_status={daily_status}",
        )

    if daily_status in TERMINAL_NO_INTENT_STATUSES and not paper_intents.empty:
        _add(
            violations,
            "terminal_status_intent_written",
            "paper_intents",
            f"daily terminal status {daily_status} must not write paper intent",
        )

    if daily_status in TERMINAL_NO_INTENT_STATUSES:
        if run_status != daily_status:
            _add(
                violations,
                "terminal_run_status",
                "run_metadata",
                f"run final_status={run_status} must equal terminal daily status={daily_status}",
            )
        return

    if paper_intents.empty:
        if run_status != daily_status:
            _add(
                violations,
                "no_intent_run_status",
                "run_metadata",
                f"no intent written, so run final_status={run_status} must equal daily final_status={daily_status}",
            )
        return

    intent_row = paper_intents.iloc[0]
    intent_status = str(intent_row["final_status"])
    run_intent_status = run_metadata.get("paper_intent_final_status")

    if str(run_intent_status) != intent_status:
        _add(
            violations,
            "intent_vs_run_status",
            "run_metadata",
            f"paper_intent_final_status={run_intent_status} differs from intent final_status={intent_status}",
        )

    if intent_status in FINAL_STATUS_PRIORITY_FROM_INTENT and run_status != intent_status:
        _add(
            violations,
            "run_final_status_priority",
            "run_metadata",
            f"run final_status={run_status} must use paper intent status={intent_status}",
        )

    if not risk_report.empty:
        risk_statuses = set(str(x) for x in risk_report["final_status"].dropna().unique())
        if intent_status not in risk_statuses:
            _add(
                violations,
                "risk_vs_intent_status",
                "risk_report",
                f"intent final_status={intent_status} not found in risk report statuses={sorted(risk_statuses)}",
            )

    if _boolish(broker_metadata.get("live_order_sent", False)) or _boolish(
        run_metadata.get("live_order_sent", False)
    ):
        _add(
            violations,
            "cross_artifact_live_order_sent",
            "run_metadata",
            "broker/run metadata indicates a live order was sent",
        )


def _check_config_snapshot(
    snapshot_path: Path,
    config: BrokerConfig,
    violations: list[IntegrationViolation],
) -> None:
    audit = config.raw.get("audit", {})
    write_snapshot = bool(audit.get("write_config_snapshot", True)) if isinstance(audit, dict) else True

    if write_snapshot and not snapshot_path.exists():
        _add(
            violations,
            "config_snapshot_exists",
            "config_snapshot",
            f"config snapshot missing: {snapshot_path}",
        )
        return

    if not snapshot_path.exists():
        return

    text = snapshot_path.read_text(encoding="utf-8")

    if "live_orders_enabled: true" in text:
        _add(
            violations,
            "config_snapshot_live_orders",
            "config_snapshot",
            "snapshot contains live_orders_enabled: true",
        )

    if "allow_order_placement: true" in text:
        _add(
            violations,
            "config_snapshot_order_placement",
            "config_snapshot",
            "snapshot contains allow_order_placement: true",
        )


def _read_csv(path: Path, *, artifact: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise Phase11IntegrationError(f"Could not read {artifact}: {path}: {exc}") from exc


def _read_json(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise Phase11IntegrationError(f"Could not read {artifact}: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise Phase11IntegrationError(f"{artifact} must contain a JSON object")

    return payload


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in {"true", "t", "1", "yes", "y"}:
            return True

        if cleaned in {"false", "f", "0", "no", "n", "", "nan"}:
            return False

    return bool(value)


def _add(
    violations: list[IntegrationViolation],
    check_name: str,
    artifact: str,
    reason: str,
    *,
    severity: str = "ERROR",
) -> None:
    violations.append(
        IntegrationViolation(
            check_name=check_name,
            artifact=artifact,
            severity=severity,
            reason=reason,
        )
    )