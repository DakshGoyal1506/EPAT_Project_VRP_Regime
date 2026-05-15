"""Phase 0 data download entry point.

Phase 0 supports dry-run only. It prints configured data sources and performs no
network requests and no file writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from vrp.data.schema import OHLCV_COLUMNS


DEFAULT_CONFIG_PATH = Path("configs/data_sources.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download configured data sources. Phase 0 supports --dry-run only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to data source configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended data sources without downloading anything.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return loaded


def get_enabled_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Expected 'sources' to be a list in data source config.")

    enabled_sources = [
        source for source in sources if isinstance(source, dict) and source.get("enabled")
    ]
    return enabled_sources


def print_dry_run(config: dict[str, Any]) -> None:
    print("EPAT VRP data download dry run")
    print("No downloads will be performed.")
    print("No files will be written.")
    print()

    print("Canonical OHLCV columns:")
    for column in OHLCV_COLUMNS:
        print(f"  - {column}")
    print()

    enabled_sources = get_enabled_sources(config)
    print(f"Enabled sources: {len(enabled_sources)}")
    for source in enabled_sources:
        source_id = source.get("source_id")
        provider = source.get("provider")
        market = source.get("market")
        symbol = source.get("symbol")
        access_method = source.get("access_method")
        raw_output = source.get("raw_output")
        interim_output = source.get("interim_output")

        print(f"- {source_id}")
        print(f"    provider: {provider}")
        print(f"    market: {market}")
        print(f"    symbol: {symbol}")
        print(f"    access_method: {access_method}")
        print(f"    raw_output: {raw_output}")
        print(f"    interim_output: {interim_output}")

    print()
    print("Dry run complete.")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    if not args.dry_run:
        raise NotImplementedError(
            "Phase 0 supports dry-run only. Run: python scripts/download_data.py --dry-run"
        )

    print_dry_run(config)


if __name__ == "__main__":
    main()
