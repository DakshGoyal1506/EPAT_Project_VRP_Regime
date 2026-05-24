from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.backtest.backtest_config import load_backtest_config  # noqa: E402
from vrp.backtest.final_audit import (  # noqa: E402
    assert_final_audit_passed,
    render_final_audit_summary,
    run_phase10_final_audit,
    write_final_audit_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final Phase 10 integration and artifact audit."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        default="ALL",
        help="Market to audit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "backtest.yaml",
        help="Backtest config path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "tables" / "phase_10" / "phase10_final_audit.json",
        help="Final audit JSON output path.",
    )
    parser.add_argument(
        "--no-require-robustness",
        action="store_true",
        help="Do not fail if robustness outputs are missing.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit non-zero when audit has errors.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_backtest_config(args.config)

    result = run_phase10_final_audit(
        config=config,
        repo_root=args.repo_root,
        market=args.market,
        require_robustness=not args.no_require_robustness,
    )

    write_final_audit_report(result, args.output)
    print(render_final_audit_summary(result))
    print(f"Final audit JSON: {args.output}")

    if not args.no_strict:
        assert_final_audit_passed(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())