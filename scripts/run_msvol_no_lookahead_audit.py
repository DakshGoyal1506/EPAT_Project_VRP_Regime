from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrp.reports.msvol_no_lookahead import (
    load_msvol_config,
    normalize_market_arg,
    run_msvol_no_lookahead_audit_for_market,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 8 MSVOL no-lookahead audit. "
            "This checks timing only; it does not create signals, positions, or backtests."
        )
    )

    parser.add_argument(
        "--market",
        required=True,
        choices=["US", "INDIA", "ALL"],
        help="Market to audit.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_msvol.yaml",
        help="Path to MSVOL config YAML.",
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Write failed/skipped audit rows instead of stopping when outputs are missing.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_msvol_config(args.config, project_root=PROJECT_ROOT)
        markets = normalize_market_arg(args.market, config)

        overall_failed_error_checks = 0

        for market in markets:
            result = run_msvol_no_lookahead_audit_for_market(
                market=market,
                config=config,
                project_root=PROJECT_ROOT,
                allow_skip=args.allow_skip,
            )

            overall_failed_error_checks += result.n_failed_error_checks

            print(
                json.dumps(
                    {
                        "market": result.market,
                        "status": result.status,
                        "n_checks": result.n_checks,
                        "n_failed_error_checks": result.n_failed_error_checks,
                        "market_audit_csv": str(result.market_audit_csv),
                        "combined_audit_csv": str(result.combined_audit_csv),
                        "skip_reason": result.skip_reason,
                    },
                    indent=2,
                )
            )

        if overall_failed_error_checks > 0:
            print(
                f"[MSVOL NO-LOOKAHEAD AUDIT FAILED] failed_error_checks={overall_failed_error_checks}",
                file=sys.stderr,
            )
            return 1

        return 0

    except Exception as exc:
        print(f"[MSVOL NO-LOOKAHEAD AUDIT ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())