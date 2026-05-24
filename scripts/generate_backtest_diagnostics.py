from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.backtest.backtest_config import load_backtest_config  # noqa: E402
from vrp.reports.backtest_diagnostics import (  # noqa: E402
    generate_backtest_diagnostics,
    render_diagnostics_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 10 backtest diagnostic tables and figures."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        default="ALL",
        help="Market to report.",
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

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_backtest_config(args.config)
    result = generate_backtest_diagnostics(
        config=config,
        repo_root=args.repo_root,
        market=args.market,
    )

    print(render_diagnostics_summary(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())