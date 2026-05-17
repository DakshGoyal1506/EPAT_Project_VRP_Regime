"""Phase 1 data ingestion CLI.

Examples
--------
Dry run all configured public sources:

    python scripts/download_data.py --market ALL --source all --dry-run

Download Yahoo sources for US:

    python scripts/download_data.py --market US --source yahoo

Download Yahoo sources for India:

    python scripts/download_data.py --market INDIA --source yahoo

Use a local CBOE CSV override:

    python scripts/download_data.py --market US --source cboe --source-id cboe_vix --local-csv data/manual/cboe/VIX_History.csv

Use a local NSE CSV override:

    python scripts/download_data.py --market INDIA --source nse --source-id nse_india_vix --local-csv data/manual/nse/india_vix.csv

Rules
-----
- No source mixing.
- No forward-fill.
- No US/India calendar merge.
- iBridgePy/IBKR is not used in Phase 1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from vrp.data.base import BaseDataLoader, DataIngestionError, DataLoadResult
from vrp.data.cboe_loader import CboeVixLoader
from vrp.data.fred_loader import FredLoader
from vrp.data.io import ensure_parent_dir, save_processed, save_raw
from vrp.data.nse_loader import NseLocalCsvLoader
from vrp.data.yahoo_loader import YahooFinanceLoader
from vrp.data.validators import build_data_audit_row


DEFAULT_CONFIG_PATH = Path("configs/data_sources.yaml")
DEFAULT_AUDIT_PATH = Path("reports/tables/data_audit.csv")

MARKET_CHOICES = {"US", "INDIA", "ALL"}
SOURCE_CHOICES = {"yahoo", "fred", "cboe", "nse", "all"}


@dataclass(frozen=True)
class SourceAttempt:
    """Result of attempting one configured source."""

    source_config: dict[str, Any]
    result: DataLoadResult | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/load Phase 1 public market data sources."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to data source YAML config.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="ALL",
        choices=sorted(MARKET_CHOICES),
        help="Market filter: US, INDIA, or ALL.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=sorted(SOURCE_CHOICES),
        help="Source family filter: yahoo, fred, cboe, nse, or all.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Optional exact source_id filter, for example cboe_vix or nse_india_vix.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional override start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional override end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing raw/processed/audit outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected sources and planned outputs without downloading or writing.",
    )
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=None,
        help=(
            "Optional local CSV override. Use with --source-id for CBOE/NSE sources. "
            "Example: --source-id cboe_vix --local-csv data/manual/cboe/VIX_History.csv"
        ),
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


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    selected_sources = select_sources(
        config=config,
        market=args.market,
        source_family=args.source,
        source_id=args.source_id,
    )

    selected_sources = apply_local_csv_override(
        selected_sources=selected_sources,
        local_csv=args.local_csv,
        source_id=args.source_id,
    )

    if args.dry_run:
        print_dry_run(
            config=config,
            selected_sources=selected_sources,
            market=args.market,
            source_family=args.source,
            source_id=args.source_id,
            start=args.start,
            end=args.end,
        )
        return

    attempts = run_source_attempts(
        selected_sources=selected_sources,
        start=args.start,
        end=args.end,
        force=args.force,
    )

    processed_results = write_processed_outputs(
        config=config,
        attempts=attempts,
        market_filter=args.market,
        force=args.force,
    )

    audit_rows = build_audit_rows(attempts=attempts)
    write_audit_table(audit_rows=audit_rows, config=config, force=args.force)

    print_run_summary(attempts=attempts, processed_results=processed_results)


def select_sources(
    *,
    config: dict[str, Any],
    market: str,
    source_family: str,
    source_id: str | None,
) -> list[dict[str, Any]]:
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Expected top-level 'sources' list in config.")

    selected: list[dict[str, Any]] = []

    for source_config in sources:
        if not isinstance(source_config, dict):
            continue

        if not source_config.get("enabled", False):
            continue

        if source_config.get("provider") == "ibkr_ibridgepy":
            continue

        if market != "ALL" and source_config.get("market") != market:
            continue

        if source_family != "all" and not source_matches_family(
            source_config=source_config,
            source_family=source_family,
        ):
            continue

        if source_id is not None and source_config.get("source_id") != source_id:
            continue

        selected.append(dict(source_config))

    selected = sorted(
        selected,
        key=lambda item: (
            str(item.get("market")),
            str(item.get("dataset")),
            int(item.get("priority", 999)),
            str(item.get("source_id")),
        ),
    )

    if not selected:
        raise ValueError(
            "No enabled sources matched the filters. "
            f"market={market}, source={source_family}, source_id={source_id}"
        )

    return selected


def source_matches_family(
    *,
    source_config: dict[str, Any],
    source_family: str,
) -> bool:
    provider = str(source_config.get("provider"))
    access_method = str(source_config.get("access_method"))

    if source_family == "yahoo":
        return provider == "yahoo_finance"

    if source_family == "fred":
        return provider == "fred"

    if source_family == "cboe":
        return provider == "cboe"

    if source_family == "nse":
        return provider == "nse"

    if source_family == "all":
        return True

    return source_family in {provider, access_method}


def apply_local_csv_override(
    *,
    selected_sources: list[dict[str, Any]],
    local_csv: Path | None,
    source_id: str | None,
) -> list[dict[str, Any]]:
    if local_csv is None:
        return selected_sources

    if source_id is None:
        if len(selected_sources) != 1:
            raise ValueError(
                "--local-csv without --source-id is allowed only when exactly one "
                "source is selected. Provide --source-id cboe_vix or --source-id nse_india_vix."
            )

    if not local_csv.exists():
        raise FileNotFoundError(f"Local CSV override not found: {local_csv}")

    updated: list[dict[str, Any]] = []
    matched = False

    for source_config in selected_sources:
        item = dict(source_config)
        if source_id is None or item.get("source_id") == source_id:
            provider = item.get("provider")
            if provider not in {"cboe", "nse"}:
                raise ValueError(
                    "--local-csv is intended only for CBOE/NSE manual CSV sources. "
                    f"Got source_id={item.get('source_id')}, provider={provider}"
                )
            item["local_csv_path"] = str(local_csv)
            matched = True
        updated.append(item)

    if not matched:
        raise ValueError(f"No selected source matched --source-id {source_id}")

    return updated


def print_dry_run(
    *,
    config: dict[str, Any],
    selected_sources: list[dict[str, Any]],
    market: str,
    source_family: str,
    source_id: str | None,
    start: str | None,
    end: str | None,
) -> None:
    print("EPAT VRP Phase 1 data ingestion dry run")
    print("No downloads will be performed.")
    print("No files will be written.")
    print()
    print(f"market filter: {market}")
    print(f"source filter: {source_family}")
    print(f"source_id filter: {source_id}")
    print(f"start override: {start}")
    print(f"end override: {end}")
    print()

    print(f"Selected sources: {len(selected_sources)}")
    for item in selected_sources:
        print(f"- {item.get('source_id')}")
        print(f"    market: {item.get('market')}")
        print(f"    dataset: {item.get('dataset')}")
        print(f"    role: {item.get('role')}")
        print(f"    provider: {item.get('provider')}")
        print(f"    access_method: {item.get('access_method')}")
        print(f"    priority: {item.get('priority')}")
        print(f"    symbol: {item.get('symbol')}")
        print(f"    start_date: {item.get('start_date')}")
        print(f"    end_date: {item.get('end_date')}")
        print(f"    official_url: {item.get('official_url')}")
        print(f"    local_csv_path: {item.get('local_csv_path')}")
        print(f"    raw_path: {item.get('raw_path')}")
        print(f"    processed_path: {item.get('processed_path')}")

    print()
    print("Processed output priority plan:")
    processed_outputs = config.get("processed_outputs", {})
    if isinstance(processed_outputs, dict):
        for market_key, market_config in processed_outputs.items():
            if market != "ALL" and market_key != market:
                continue
            if not isinstance(market_config, dict):
                continue
            for dataset_kind, dataset_config in market_config.items():
                if not isinstance(dataset_config, dict):
                    continue
                print(f"- {market_key}.{dataset_kind}")
                print(f"    dataset: {dataset_config.get('dataset')}")
                print(f"    processed_path: {dataset_config.get('processed_path')}")
                print(f"    source_priority: {dataset_config.get('source_priority')}")

    print()
    print("Dry run complete.")


def run_source_attempts(
    *,
    selected_sources: list[dict[str, Any]],
    start: str | None,
    end: str | None,
    force: bool,
) -> list[SourceAttempt]:
    attempts: list[SourceAttempt] = []

    for source_config in selected_sources:
        source_id = source_config.get("source_id")
        print(f"Loading source: {source_id}")

        try:
            loader = build_loader(source_config)
            result = loader.load(start=start, end=end)

            save_raw(result.frame, result.raw_path, force=force)

            attempts.append(
                SourceAttempt(
                    source_config=source_config,
                    result=result,
                    error=None,
                )
            )
            print(f"  PASS raw_path={result.raw_path}")

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            attempts.append(
                SourceAttempt(
                    source_config=source_config,
                    result=None,
                    error=error_message,
                )
            )
            print(f"  FAIL {source_id}: {error_message}")

    return attempts


def build_loader(source_config: dict[str, Any]) -> BaseDataLoader:
    provider = str(source_config.get("provider"))

    if provider == "yahoo_finance":
        return YahooFinanceLoader(source_config)

    if provider == "fred":
        return FredLoader(source_config)

    if provider == "cboe":
        return CboeVixLoader(source_config)

    if provider == "nse":
        return NseLocalCsvLoader(source_config)

    raise DataIngestionError(
        f"No loader implemented for source_id={source_config.get('source_id')}, "
        f"provider={provider}"
    )


def write_processed_outputs(
    *,
    config: dict[str, Any],
    attempts: list[SourceAttempt],
    market_filter: str,
    force: bool,
) -> list[DataLoadResult]:
    successful_by_source_id = {
        attempt.result.source_id: attempt.result
        for attempt in attempts
        if attempt.result is not None
    }

    errors_by_source_id = {
        str(attempt.source_config.get("source_id")): attempt.error
        for attempt in attempts
        if attempt.error is not None
    }

    processed_outputs_config = config.get("processed_outputs")
    if not isinstance(processed_outputs_config, dict):
        raise ValueError("Expected top-level processed_outputs mapping in config.")

    written_results: list[DataLoadResult] = []

    for market, market_outputs in processed_outputs_config.items():
        if market_filter != "ALL" and market != market_filter:
            continue

        if not isinstance(market_outputs, dict):
            continue

        for output_name, output_config in market_outputs.items():
            if not isinstance(output_config, dict):
                continue

            selected_result = select_processed_source(
                output_config=output_config,
                successful_by_source_id=successful_by_source_id,
            )

            if selected_result is None:
                selected_source_ids = output_config.get("source_priority", [])
                matching_failures = {
                    source_id: errors_by_source_id.get(source_id)
                    for source_id in selected_source_ids
                    if source_id in errors_by_source_id
                }

                # If this output had no selected/attempted source in this run, skip it.
                # Example: --source fred should not require india_underlying.
                attempted_source_ids = {
                    str(attempt.source_config.get("source_id")) for attempt in attempts
                }
                relevant_attempted = [
                    source_id
                    for source_id in selected_source_ids
                    if source_id in attempted_source_ids
                ]

                if not relevant_attempted:
                    continue

                raise DataIngestionError(
                    f"All attempted sources failed for processed output "
                    f"{market}.{output_name}. "
                    f"Dataset={output_config.get('dataset')}. "
                    f"Failures={matching_failures}"
                )

            processed_path = output_config.get("processed_path")
            if processed_path is None:
                raise ValueError(f"Missing processed_path for {market}.{output_name}")

            save_processed(selected_result.frame, processed_path, force=force)
            written_results.append(selected_result)

            print(
                "Processed output written: "
                f"{processed_path} using source_id={selected_result.source_id}"
            )

    return written_results


def select_processed_source(
    *,
    output_config: dict[str, Any],
    successful_by_source_id: dict[str, DataLoadResult],
) -> DataLoadResult | None:
    source_priority = output_config.get("source_priority")

    if not isinstance(source_priority, list):
        raise ValueError(
            f"Expected source_priority list for output_config={output_config}"
        )

    for source_id in source_priority:
        source_id_str = str(source_id)
        if source_id_str in successful_by_source_id:
            return successful_by_source_id[source_id_str]

    return None


def build_audit_rows(attempts: list[SourceAttempt]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for attempt in attempts:
        source_config = attempt.source_config
        source_id = str(source_config.get("source_id"))
        market = str(source_config.get("market"))
        dataset = str(source_config.get("dataset"))
        symbol = source_config.get("symbol")
        symbol_str = None if symbol is None else str(symbol)

        if attempt.result is not None:
            row = build_data_audit_row(
                attempt.result.frame,
                market=market,
                dataset=dataset,
                source=source_id,
                symbol=symbol_str,
            )
            rows.append(row)
            continue

        rows.append(
            {
                "market": market,
                "dataset": dataset,
                "source": source_id,
                "symbol": symbol_str,
                "start_date": None,
                "end_date": None,
                "n_rows": 0,
                "n_missing_close": 0,
                "n_duplicate_dates": 0,
                "min_close": None,
                "max_close": None,
                "validation_status": f"FAIL: {attempt.error}",
            }
        )

    return rows


def write_audit_table(
    *,
    audit_rows: list[dict[str, Any]],
    config: dict[str, Any],
    force: bool,
) -> Path:
    paths = config.get("paths", {})
    audit_path = Path(str(paths.get("audit_table", DEFAULT_AUDIT_PATH)))

    if audit_path.exists() and not force:
        raise FileExistsError(
            f"Audit table already exists and force=False: {audit_path}. "
            "Use --force to overwrite."
        )

    ensure_parent_dir(audit_path)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_path, index=False)

    print(f"Audit table written: {audit_path}")
    return audit_path


def print_run_summary(
    *,
    attempts: list[SourceAttempt],
    processed_results: list[DataLoadResult],
) -> None:
    n_success = sum(1 for attempt in attempts if attempt.result is not None)
    n_failed = sum(1 for attempt in attempts if attempt.error is not None)

    print()
    print("Phase 1 data ingestion summary")
    print(f"Sources attempted: {len(attempts)}")
    print(f"Sources succeeded: {n_success}")
    print(f"Sources failed: {n_failed}")
    print(f"Processed outputs written: {len(processed_results)}")

    if n_failed:
        print()
        print("Failed sources:")
        for attempt in attempts:
            if attempt.error is not None:
                print(f"- {attempt.source_config.get('source_id')}: {attempt.error}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()