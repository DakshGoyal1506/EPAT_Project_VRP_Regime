#!/usr/bin/env python
"""
Phase 11 completion validator.

This script validates that Phase 11 is still a readiness-only layer.

It checks:
- no executable live-order source patterns in src/ and scripts/
- Phase 11 output artifacts are internally consistent
- live_order_sent remains false across artifacts

It does not create signals.
It does not create paper intents.
It does not connect to a broker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    src_path = repo_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from vrp.broker.broker_config import BrokerConfigError, load_broker_config  # noqa: E402
from vrp.broker.live_order_guard import (  # noqa: E402
    LiveOrderGuardError,
    assert_no_live_order_code,
    write_live_order_guard_report,
)
from vrp.broker.phase11_integration_checks import (  # noqa: E402
    Phase11IntegrationError,
    assert_phase11_artifacts_valid,
    write_phase11_integration_report,
)


class Phase11ValidationCliError(RuntimeError):
    """Raised when validation CLI inputs are invalid."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Phase 11 readiness-only safety and artifact consistency."
    )

    parser.add_argument(
        "--config",
        default="configs/ibkr_paper.yaml",
        help="Path to Phase 11 broker paper config.",
    )
    parser.add_argument(
        "--skip-source-guard",
        action="store_true",
        help="Skip source-code live-order guard.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip artifact integration checks.",
    )
    parser.add_argument(
        "--source-guard-report",
        default="reports/tables/phase_11/live_order_guard_report.json",
        help="Output path for source guard JSON report.",
    )
    parser.add_argument(
        "--integration-report",
        default="reports/tables/phase_11/phase11_integration_report.json",
        help="Output path for integration JSON report.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print machine-readable validation summary.",
    )

    return parser


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_source_guard and args.skip_artifacts:
        raise Phase11ValidationCliError(
            "At least one validation must run; both guards cannot be skipped."
        )

    config = load_broker_config(args.config)

    source_guard_passed: bool | None = None
    integration_passed: bool | None = None
    source_guard_report_path: str | None = None
    integration_report_path: str | None = None

    if not args.skip_source_guard:
        source_report = assert_no_live_order_code([Path("src"), Path("scripts")])
        source_report_path = write_live_order_guard_report(
            source_report,
            args.source_guard_report,
        )
        source_guard_passed = source_report.passed
        source_guard_report_path = str(source_report_path)

    if not args.skip_artifacts:
        integration_report = assert_phase11_artifacts_valid(config)
        integration_path = write_phase11_integration_report(
            integration_report,
            args.integration_report,
        )
        integration_passed = integration_report.passed
        integration_report_path = str(integration_path)

    passed = all(
        value is not False
        for value in (
            source_guard_passed,
            integration_passed,
        )
    )

    return {
        "passed": passed,
        "source_guard_passed": source_guard_passed,
        "integration_passed": integration_passed,
        "source_guard_report": source_guard_report_path,
        "integration_report": integration_report_path,
        "live_order_sent": False,
    }


def print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("Phase 11 validation summary")
    print(f"passed: {summary['passed']}")
    print(f"source_guard_passed: {summary['source_guard_passed']}")
    print(f"integration_passed: {summary['integration_passed']}")
    print(f"live_order_sent: {summary['live_order_sent']}")

    if summary["source_guard_report"] is not None:
        print(f"source_guard_report: {summary['source_guard_report']}")

    if summary["integration_report"] is not None:
        print(f"integration_report: {summary['integration_report']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_validation(args)
    except (
        BrokerConfigError,
        LiveOrderGuardError,
        Phase11IntegrationError,
        Phase11ValidationCliError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_summary(summary, as_json=args.print_json)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())