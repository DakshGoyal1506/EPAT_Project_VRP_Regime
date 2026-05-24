from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.backtest.backtest_config import load_backtest_config  # noqa: E402
from vrp.backtest.backtest_registry import BACKTEST_STRATEGY_UNIVERSE  # noqa: E402
from vrp.backtest.schema_audit import (  # noqa: E402
    assert_no_audit_errors,
    audit_phase10_inputs,
    render_audit_summary,
)
from vrp.backtest.vectorized_engine import (  # noqa: E402
    render_run_summary,
    resolve_markets,
    run_backtests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 10 vectorised research-layer VRP backtest."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        default="ALL",
        help="Market to run.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", *BACKTEST_STRATEGY_UNIVERSE],
        default="all",
        help="Strategy to run. Default runs all approved Phase 9 strategies.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Override default cost in basis points. If omitted, config default is used.",
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
        help="Repository root. Relative config paths resolve against this directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate panels but do not write output files.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip Chunk 0 input audit before running.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_backtest_config(args.config)
    markets = resolve_markets(args.market)

    if not args.skip_audit:
        audit_results = audit_phase10_inputs(
            repo_root=args.repo_root,
            market=args.market,
        )
        print(render_audit_summary(audit_results))
        assert_no_audit_errors(audit_results)

    results = run_backtests(
        markets=markets,
        config=config,
        repo_root=args.repo_root,
        strategy=args.strategy,
        cost_bps=args.cost_bps,
        force=args.force,
        write=not args.dry_run,
    )

    print(render_run_summary(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())