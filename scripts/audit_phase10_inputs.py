from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.backtest.schema_audit import (  # noqa: E402
    audit_phase10_inputs,
    render_audit_summary,
    write_audit_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight schema audit for Phase 10 VRP backtest inputs."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        default="ALL",
        help="Market to audit.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to parent of scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "tables" / "phase_10",
        help="Directory for audit CSV/JSON outputs.",
    )
    parser.add_argument(
        "--no-write-report",
        action="store_true",
        help="Run audit but do not write CSV/JSON reports.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit with non-zero status on audit errors.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results = audit_phase10_inputs(
        repo_root=args.repo_root,
        market=args.market,
    )

    print(render_audit_summary(results))

    if not args.no_write_report:
        paths = write_audit_reports(results, args.output_dir)
        print(f"Wrote audit CSV:  {paths['csv']}")
        print(f"Wrote audit JSON: {paths['json']}")

    has_errors = any(result.has_errors() for result in results)
    if has_errors and not args.no_strict:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())