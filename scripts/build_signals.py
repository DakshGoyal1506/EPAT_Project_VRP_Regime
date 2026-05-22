#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from vrp.reports.strategy_diagnostics import (
    DEFAULT_MSVOL_POLICY,
    DEFAULT_REPORT_NOTE,
    build_input_file_hashes,
    build_phase9_metadata,
    build_row_counts_by_input,
    create_all_phase9_diagnostics,
    write_diagnostic_tables,
    write_metadata_json,
)
from vrp.strategies.signal_builder import build_phase9_signal_panel
from vrp.strategies.strategy_config import (
    StrategyConfig,
    get_market_input_paths,
    get_market_output_path,
    get_strategy_definitions,
    load_strategy_config,
    strategy_config_hash,
)
from vrp.strategies.strategy_registry import APPROVED_STRATEGY_NAMES


DEFAULT_CONFIG_PATH = Path("configs/strategies.yaml")
SUPPORTED_MARKETS = ("US", "INDIA")
MARKET_ALL = "ALL"

REQUIRED_INPUT_KEYS = (
    "har",
    "threshold",
    "gaussian_hmm",
    "markov_autoreg",
)

DIAGNOSTIC_OUTPUT_KEYS = (
    "signal_summary",
    "exposure_by_year",
    "exposure_change_summary",
    "blocked_reason_summary",
    "no_lookahead_audit",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Phase 9 strategy signal panels. This creates ex-ante "
            "short-vol exposure intentions only. It does not run a backtest."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs/strategies.yaml.",
    )

    parser.add_argument(
        "--market",
        type=str.upper,
        choices=(*SUPPORTED_MARKETS, MARKET_ALL),
        required=True,
        help="Market to process: US, INDIA, or ALL.",
    )

    parser.add_argument(
        "--strategy",
        default="all",
        choices=("all", *APPROVED_STRATEGY_NAMES),
        help="Approved Phase 9 strategy to build, or all.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Phase 9 outputs.",
    )

    return parser.parse_args(argv)


def _markets_from_arg(market: str) -> tuple[str, ...]:
    market_key = market.strip().upper()

    if market_key == MARKET_ALL:
        return SUPPORTED_MARKETS

    if market_key in SUPPORTED_MARKETS:
        return (market_key,)

    raise ValueError(f"Unsupported market: {market}")


def _ensure_no_msvol_paths(paths: Mapping[str, Path]) -> None:
    for source_name, path in paths.items():
        lowered = str(path).lower()
        if "msvol" in lowered or "msgarch" in lowered:
            raise ValueError(
                "Phase 9 must not read MSVOL/MSGARCH files. "
                f"Found source={source_name}, path={path}."
            )


def _check_required_input_keys(paths: Mapping[str, Path]) -> None:
    supplied = set(paths)
    required = set(REQUIRED_INPUT_KEYS)

    if supplied != required:
        missing = sorted(required.difference(supplied))
        extra = sorted(supplied.difference(required))
        raise ValueError(
            "Phase 9 input paths must contain exactly the required Phase 4/5/6/7 "
            f"sources. Missing={missing}. Extra={extra}."
        )


def _read_input_frames(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    _check_required_input_keys(paths)
    _ensure_no_msvol_paths(paths)

    frames: dict[str, pd.DataFrame] = {}

    for source_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Phase 9 input file for source '{source_name}': {path}"
            )

        frames[source_name] = pd.read_parquet(path)

    return frames


def _ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. Re-run with --force to overwrite."
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_signal_panel(
    *,
    signal_panel: pd.DataFrame,
    output_path: Path,
    force: bool,
) -> None:
    _ensure_can_write(output_path, force=force)
    _ensure_parent(output_path)
    signal_panel.to_parquet(output_path, index=False)


def _select_report_paths(config: StrategyConfig, force: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    for key in DIAGNOSTIC_OUTPUT_KEYS:
        if key not in config.report_files:
            raise KeyError(f"Missing report_files.{key} in strategy config.")

        path = Path(config.report_files[key])
        _ensure_can_write(path, force=force)
        paths[key] = path

    if "metadata" not in config.report_files:
        raise KeyError("Missing report_files.metadata in strategy config.")

    metadata_path = Path(config.report_files["metadata"])
    _ensure_can_write(metadata_path, force=force)
    paths["metadata"] = metadata_path

    return paths


def _concat_diagnostic_tables(
    per_market_tables: Mapping[str, Mapping[str, pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    combined: dict[str, pd.DataFrame] = {}

    table_names = {
        table_name
        for tables in per_market_tables.values()
        for table_name in tables.keys()
    }

    for table_name in sorted(table_names):
        frames: list[pd.DataFrame] = []

        for market, tables in per_market_tables.items():
            if table_name not in tables:
                continue

            table = tables[table_name].copy()

            if "market" not in table.columns:
                table.insert(0, "market", market)

            frames.append(table)

        if frames:
            combined[table_name] = pd.concat(frames, ignore_index=True)
        else:
            combined[table_name] = pd.DataFrame()

    return combined


def _timing_policy_dict(config: StrategyConfig) -> dict[str, Any]:
    return asdict(config.timing_policy)


def _exposure_bounds_dict(config: StrategyConfig) -> dict[str, Any]:
    return asdict(config.exposure_bounds)


def _build_market(
    *,
    config: StrategyConfig,
    market: str,
    requested_strategy: str,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    input_paths = get_market_input_paths(config, market)
    output_path = get_market_output_path(config, market)

    frames = _read_input_frames(input_paths)

    strategy_definitions = get_strategy_definitions(
        config,
        requested_strategy=requested_strategy,
    )

    build_result = build_phase9_signal_panel(
        market=market,
        har=frames["har"],
        threshold=frames["threshold"],
        hmm=frames["gaussian_hmm"],
        mar=frames["markov_autoreg"],
        strategy_definitions=strategy_definitions,
        requested_strategy=requested_strategy,
        validate_output=True,
    )

    _write_signal_panel(
        signal_panel=build_result.signals,
        output_path=output_path,
        force=force,
    )

    diagnostics = create_all_phase9_diagnostics(
        signal_panel=build_result.signals,
        present_but_excluded=build_result.forbidden_columns_present_but_excluded,
        forbidden_columns_used=build_result.forbidden_columns_used,
    )

    metadata = build_phase9_metadata(
        market=market,
        strategy_config_hash=strategy_config_hash(config),
        input_file_paths=input_paths,
        input_file_hashes=build_input_file_hashes(input_paths),
        row_counts_by_input=build_row_counts_by_input(frames),
        strategy_names=build_result.signals["strategy_name"].drop_duplicates().tolist(),
        forbidden_columns_present_but_excluded=(
            build_result.forbidden_columns_present_but_excluded
        ),
        forbidden_columns_used=build_result.forbidden_columns_used,
        msvol_policy=DEFAULT_MSVOL_POLICY,
        timing_policy=_timing_policy_dict(config),
        exposure_bounds=_exposure_bounds_dict(config),
        report_note=config.report_note or DEFAULT_REPORT_NOTE,
    )

    metadata["output_file"] = str(output_path)
    metadata["signal_row_count"] = int(len(build_result.signals))
    metadata["requested_strategy"] = requested_strategy

    return build_result.signals, diagnostics, metadata


def _combined_metadata(
    *,
    config: StrategyConfig,
    requested_market: str,
    requested_strategy: str,
    per_market_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    all_forbidden_used: list[str] = []

    for metadata in per_market_metadata.values():
        all_forbidden_used.extend(metadata.get("forbidden_columns_used", []))

    if all_forbidden_used:
        raise ValueError(
            "Forbidden columns were consumed by strategy logic: "
            f"{sorted(set(all_forbidden_used))}"
        )

    return {
        "phase": 9,
        "requested_market": requested_market,
        "requested_strategy": requested_strategy,
        "strategy_config_hash": strategy_config_hash(config),
        "strategy_names": (
            list(APPROVED_STRATEGY_NAMES)
            if requested_strategy == "all"
            else [requested_strategy]
        ),
        "forbidden_columns_used": [],
        "msvol_policy": DEFAULT_MSVOL_POLICY,
        "timing_policy": _timing_policy_dict(config),
        "exposure_bounds": _exposure_bounds_dict(config),
        "phase_9_report_note": config.report_note or DEFAULT_REPORT_NOTE,
        "markets": {
            market: dict(metadata)
            for market, metadata in per_market_metadata.items()
        },
    }


def run(
    *,
    config_path: Path,
    market: str,
    strategy: str,
    force: bool,
) -> int:
    config = load_strategy_config(config_path)
    markets = _markets_from_arg(market)

    report_paths = _select_report_paths(config, force=force)

    per_market_diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    per_market_metadata: dict[str, dict[str, Any]] = {}

    print("Phase 9 signal build started.")
    print(f"Config: {config_path}")
    print(f"Market request: {market}")
    print(f"Strategy request: {strategy}")
    print("MSVOL policy: excluded_diagnostic_only; MSVOL files will not be read.")

    for market_key in markets:
        print(f"\nBuilding market: {market_key}")

        signals, diagnostics, metadata = _build_market(
            config=config,
            market=market_key,
            requested_strategy=strategy,
            force=force,
        )

        per_market_diagnostics[market_key] = diagnostics
        per_market_metadata[market_key] = metadata

        output_path = get_market_output_path(config, market_key)

        print(f"Signal rows: {len(signals)}")
        print(f"Strategies: {sorted(signals['strategy_name'].unique().tolist())}")
        print(f"Wrote signal panel: {output_path}")

    combined_diagnostics = _concat_diagnostic_tables(per_market_diagnostics)

    diagnostic_paths = {
        key: path
        for key, path in report_paths.items()
        if key != "metadata"
    }

    written_tables = write_diagnostic_tables(
        tables=combined_diagnostics,
        output_paths=diagnostic_paths,
    )

    metadata = _combined_metadata(
        config=config,
        requested_market=market,
        requested_strategy=strategy,
        per_market_metadata=per_market_metadata,
    )

    metadata_path = write_metadata_json(
        metadata=metadata,
        output_path=report_paths["metadata"],
    )

    print("\nWrote diagnostics:")
    for name, path in written_tables.items():
        print(f"  {name}: {path}")

    print(f"Wrote metadata: {metadata_path}")
    print("Phase 9 signal build completed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    return run(
        config_path=args.config,
        market=args.market,
        strategy=args.strategy,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
