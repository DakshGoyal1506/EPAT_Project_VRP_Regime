from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrp.reports.msvol_diagnostics import (
    load_msvol_config,
    normalize_market_arg,
    run_msvol_diagnostics_for_market,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 8 Python MSVOL diagnostics. "
            "Diagnostics only: no strategy, no backtest, no exposure sizing."
        )
    )

    parser.add_argument(
        "--market",
        required=True,
        choices=["US", "INDIA", "ALL"],
        help="Market to process.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_msvol.yaml",
        help="Path to MSVOL config YAML.",
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Write skipped diagnostics if MSVOL processed panel is missing.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_msvol_config(args.config, project_root=PROJECT_ROOT)
        markets = normalize_market_arg(args.market, config)

        for market in markets:
            result = run_msvol_diagnostics_for_market(
                market=market,
                config=config,
                project_root=PROJECT_ROOT,
                allow_skip=args.allow_skip,
            )

            print(
                json.dumps(
                    {
                        "market": result.market,
                        "status": result.status,
                        "comparison_summary_csv": str(result.comparison_summary_csv),
                        "state_duration_summary_csv": str(result.state_duration_summary_csv),
                        "appendix_csv": str(result.appendix_csv),
                        "n_summary_rows": result.n_summary_rows,
                        "n_duration_rows": result.n_duration_rows,
                        "skip_reason": result.skip_reason,
                    },
                    indent=2,
                )
            )

        return 0

    except Exception as exc:
        print(f"[MSVOL DIAGNOSTICS ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())