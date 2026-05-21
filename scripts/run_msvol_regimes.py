from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrp.regimes.msvol_model import (
    MSVolError,
    load_msvol_config,
    normalize_market_arg,
    run_msvol_for_market,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Python-only Markov-switching volatility robustness model "
            "for Phase 8. This is MSVOL, not true MSGARCH."
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
        help="Write skip reports instead of failing when fitting/input errors occur.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_msvol_config(args.config, project_root=PROJECT_ROOT)
        markets = normalize_market_arg(args.market, config)

        results = []
        for market in markets:
            result = run_msvol_for_market(
                market=market,
                config=config,
                project_root=PROJECT_ROOT,
                allow_skip=args.allow_skip,
            )
            results.append(result)

            print(
                json.dumps(
                    {
                        "market": result.market,
                        "status": result.status,
                        "n_observations": result.n_observations,
                        "input_csv": str(result.input_csv),
                        "raw_output_csv": str(result.raw_output_csv),
                        "preflight_json": str(result.preflight_json),
                        "model_summary_json": str(result.model_summary_json),
                        "skip_reason": result.skip_reason,
                    },
                    indent=2,
                )
            )

        if any(r.status == "skipped" for r in results):
            print(
                "One or more markets were skipped. This is allowed only because --allow-skip was used.",
                file=sys.stderr,
            )

        return 0

    except Exception as exc:
        print(f"[MSVOL RUN ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())