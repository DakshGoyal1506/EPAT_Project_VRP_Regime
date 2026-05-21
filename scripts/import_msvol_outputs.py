from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrp.regimes.msvol_adapter import (
    import_msvol_outputs_for_market,
    load_msvol_config,
    normalize_market_arg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import raw Python MSVOL outputs and write standardized Phase 8 regime panels. "
            "This is not true MSGARCH."
        )
    )

    parser.add_argument(
        "--market",
        required=True,
        choices=["US", "INDIA", "ALL"],
        help="Market to import.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_msvol.yaml",
        help="Path to MSVOL config YAML.",
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Write skipped metadata/output if raw output is missing.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_msvol_config(args.config, project_root=PROJECT_ROOT)
        markets = normalize_market_arg(args.market, config)

        for market in markets:
            result = import_msvol_outputs_for_market(
                market=market,
                config=config,
                config_path=args.config,
                project_root=PROJECT_ROOT,
                allow_skip=args.allow_skip,
            )

            print(
                json.dumps(
                    {
                        "market": result.market,
                        "status": result.status,
                        "n_processed_rows": result.n_processed_rows,
                        "processed_output_parquet": str(result.processed_output_parquet),
                        "metadata_json": str(result.metadata_json),
                        "probability_audit_csv": str(result.probability_audit_csv),
                        "skip_reason": result.skip_reason,
                    },
                    indent=2,
                )
            )

        return 0

    except Exception as exc:
        print(f"[MSVOL IMPORT ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())