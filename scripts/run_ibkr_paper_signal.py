#!/usr/bin/env python
"""
End-to-end Phase 11 CLI.

This runner is paper-only. It writes:
- daily_paper_signal.csv
- paper_order_intents.csv
- risk_check_report.csv
- broker_metadata.json
- run_metadata.json
- ibkr_paper_config_snapshot.yaml

It does not place orders.
It does not expose broker execution functions.
It does not infer live sizing from Phase 10.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Allow `python scripts/run_ibkr_paper_signal.py` from repo root."""

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    src_path = repo_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from vrp.broker.broker_config import (  # noqa: E402
    BrokerConfigError,
    ensure_output_directories,
    get_output_paths,
    load_broker_config,
)
from vrp.broker.market_data import MarketDataError, QuoteSnapshot, quote_from_mapping  # noqa: E402
from vrp.broker.paper_trader import publish_paper_order_intent  # noqa: E402
from vrp.broker.signal_publisher import DailyPaperSignal, publish_daily_paper_signal  # noqa: E402
from vrp.reports.broker_diagnostics import (  # noqa: E402
    read_json_file,
    write_phase11_diagnostics,
)


class CliError(RuntimeError):
    """Raised when CLI inputs are invalid."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 11 IBKR/iBridgePy execution-readiness paper-signal layer. "
            "This writes paper-only audit artifacts and never sends live orders."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/ibkr_paper.yaml",
        help="Path to Phase 11 broker paper config.",
    )
    parser.add_argument(
        "--market",
        required=True,
        choices=["US", "INDIA"],
        help="Market to run.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy name. Defaults to config.default_strategy.",
    )
    parser.add_argument(
        "--signal-path",
        default=None,
        help="Optional override for Phase 9 signal input path.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional YYYY-MM-DD date for signal freshness evaluation.",
    )
    parser.add_argument(
        "--run-timestamp-utc",
        default=None,
        help="Optional ISO timestamp for reproducible run metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print intended paths without writing outputs.",
    )

    parser.add_argument(
        "--quote-symbol",
        default=None,
        help="Optional quote symbol. If omitted, quote is treated as missing.",
    )
    parser.add_argument(
        "--quote-bid",
        type=float,
        default=None,
        help="Optional quote bid.",
    )
    parser.add_argument(
        "--quote-ask",
        type=float,
        default=None,
        help="Optional quote ask.",
    )
    parser.add_argument(
        "--quote-last",
        type=float,
        default=None,
        help="Optional quote last.",
    )
    parser.add_argument(
        "--quote-timestamp-utc",
        default=None,
        help="Optional quote timestamp in ISO UTC format.",
    )
    parser.add_argument(
        "--quote-age-seconds",
        type=float,
        default=None,
        help="Optional quote age in seconds.",
    )
    parser.add_argument(
        "--quote-source",
        default="manual_cli",
        help="Quote source label.",
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )

    return parser


def parse_as_of_date(value: str | None) -> date | None:
    if value is None or value == "":
        return None

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CliError(f"--as-of-date must be YYYY-MM-DD, got {value!r}") from exc


def validate_run_timestamp(value: str | None) -> str | None:
    if value is None or value == "":
        return None

    cleaned = value.strip()
    parse_value = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned

    try:
        datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise CliError(
            f"--run-timestamp-utc must be an ISO datetime, got {value!r}"
        ) from exc

    return cleaned


def build_optional_quote(args: argparse.Namespace, market: str) -> QuoteSnapshot | None:
    """Build quote only when user supplies quote fields.

    If no quote fields are supplied, return None and let downstream code create
    a structured missing quote for the selected contract.
    """

    quote_fields_supplied = any(
        value is not None
        for value in (
            args.quote_symbol,
            args.quote_bid,
            args.quote_ask,
            args.quote_last,
            args.quote_timestamp_utc,
            args.quote_age_seconds,
        )
    )

    if not quote_fields_supplied:
        return None

    if not args.quote_symbol:
        raise CliError("--quote-symbol is required when quote fields are supplied")

    try:
        return quote_from_mapping(
            {
                "symbol": args.quote_symbol,
                "market": market,
                "bid": args.quote_bid,
                "ask": args.quote_ask,
                "last": args.quote_last,
                "timestamp_utc": args.quote_timestamp_utc,
                "quote_age_seconds": args.quote_age_seconds,
                "source": args.quote_source,
            }
        )
    except MarketDataError as exc:
        raise CliError(f"Invalid quote input: {exc}") from exc


def dry_run_summary(
    *,
    config_path: Path,
    market: str,
    strategy: str | None,
    signal_path: str | None,
    config_outputs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "config_path": str(config_path),
        "market": market,
        "strategy": strategy,
        "signal_path_override": signal_path,
        "would_write": {
            "daily_paper_signal": str(config_outputs["latest_signal_table"]),
            "paper_order_intents": str(config_outputs["paper_order_intents"]),
            "risk_check_report": str(config_outputs["risk_check_report"]),
            "broker_metadata": str(config_outputs["broker_metadata"]),
            "run_metadata": str(config_outputs["run_metadata"]),
            "config_snapshot": str(config_outputs["config_snapshot"]),
        },
        "live_order_sent": False,
    }


def run_phase11_cli(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_broker_config(config_path)
    ensure_output_directories(config)

    output_paths = get_output_paths(config)
    active_strategy = args.strategy or config.default_strategy
    market = args.market.upper()
    as_of_date = parse_as_of_date(args.as_of_date)
    run_timestamp = validate_run_timestamp(args.run_timestamp_utc)

    if args.dry_run:
        return dry_run_summary(
            config_path=config_path,
            market=market,
            strategy=active_strategy,
            signal_path=args.signal_path,
            config_outputs=output_paths,
        )

    quote = build_optional_quote(args, market)

    daily_signal = publish_daily_paper_signal(
        config=config,
        market=market,
        strategy_name=active_strategy,
        signal_path=args.signal_path,
        output_path=output_paths["latest_signal_table"],
        as_of_date=as_of_date,
        run_timestamp_utc=run_timestamp,
    )

    paper_result = publish_paper_order_intent(
        config=config,
        daily_signal=daily_signal,
        quote=quote,
        output_path=output_paths["paper_order_intents"],
        risk_report_path=output_paths["risk_check_report"],
    )

    diagnostics = write_phase11_diagnostics(
        config=config,
        market=market,
        strategy=active_strategy,
        input_signal_path=args.signal_path,
        daily_signal=daily_signal,
        paper_result=paper_result,
        run_timestamp_utc=run_timestamp,
    )

    run_metadata = read_json_file(diagnostics.run_metadata_path)
    broker_metadata = read_json_file(diagnostics.broker_metadata_path)

    return {
        "dry_run": False,
        "market": market,
        "strategy": active_strategy,
        "daily_signal_final_status": daily_signal.final_status,
        "paper_intent_written": paper_result.intent is not None,
        "paper_intent_final_status": (
            paper_result.intent.final_status if paper_result.intent is not None else None
        ),
        "final_status": run_metadata.get("final_status"),
        "live_order_sent": bool(run_metadata.get("live_order_sent", False)),
        "ibridgepy_available": bool(broker_metadata.get("ibridgepy_available", False)),
        "broker_connection_status": broker_metadata.get("broker_connection_status"),
        "outputs": {
            "daily_paper_signal": str(output_paths["latest_signal_table"]),
            "paper_order_intents": str(output_paths["paper_order_intents"]),
            "risk_check_report": str(output_paths["risk_check_report"]),
            "broker_metadata": str(diagnostics.broker_metadata_path),
            "run_metadata": str(diagnostics.run_metadata_path),
            "config_snapshot": (
                str(diagnostics.config_snapshot_path)
                if diagnostics.config_snapshot_path is not None
                else None
            ),
        },
    }


def print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("Phase 11 run summary")
    print(f"dry_run: {summary['dry_run']}")
    print(f"market: {summary['market']}")
    print(f"strategy: {summary['strategy']}")

    if summary["dry_run"]:
        print("live_order_sent: False")
        print("would_write:")
        for key, path in summary["would_write"].items():
            print(f"  {key}: {path}")
        return

    print(f"daily_signal_final_status: {summary['daily_signal_final_status']}")
    print(f"paper_intent_written: {summary['paper_intent_written']}")
    print(f"paper_intent_final_status: {summary['paper_intent_final_status']}")
    print(f"final_status: {summary['final_status']}")
    print(f"live_order_sent: {summary['live_order_sent']}")
    print(f"ibridgepy_available: {summary['ibridgepy_available']}")
    print(f"broker_connection_status: {summary['broker_connection_status']}")
    print("outputs:")
    for key, path in summary["outputs"].items():
        print(f"  {key}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_phase11_cli(args)
    except (BrokerConfigError, CliError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_summary(summary, as_json=args.print_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())