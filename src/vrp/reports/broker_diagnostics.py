"""
Phase 11 broker diagnostics and run metadata.

This module writes audit artifacts only:
- broker_metadata.json
- run_metadata.json
- config snapshot YAML

It does not create signals.
It does not create paper order intents.
It does not place broker orders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vrp.broker import RESEARCH_PROXY_WARNING
from vrp.broker.broker_config import (
    BrokerConfig,
    compute_config_hash,
    ensure_output_directories,
    get_market_signal_path,
    get_output_paths,
    write_config_snapshot,
)
from vrp.broker.ibridgepy_adapter import (
    BROKER_CONNECTION_NOT_ATTEMPTED,
    BROKER_DATA_UNAVAILABLE,
    IBRIDGEPY_NOT_INSTALLED,
    get_broker_metadata,
)
from vrp.broker.paper_trader import PaperIntentBuildResult
from vrp.broker.signal_publisher import DailyPaperSignal


class BrokerDiagnosticsError(RuntimeError):
    """Raised when Phase 11 diagnostics cannot be written."""


@dataclass(frozen=True)
class RunMetadata:
    """Audit metadata for one Phase 11 run."""

    run_timestamp_utc: str
    market: str
    strategy: str
    config_path: str
    config_hash: str
    input_signal_path: str
    input_signal_mtime: str | None
    latest_target_trade_date: str | None
    daily_signal_final_status: str | None
    paper_intent_final_status: str | None
    final_status: str
    ibridgepy_available: bool
    broker_connection_attempted: bool
    broker_connection_status: str
    broker_data_status: str
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    live_order_sent: bool
    research_proxy_warning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp_utc": self.run_timestamp_utc,
            "market": self.market,
            "strategy": self.strategy,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "input_signal_path": self.input_signal_path,
            "input_signal_mtime": self.input_signal_mtime,
            "latest_target_trade_date": self.latest_target_trade_date,
            "daily_signal_final_status": self.daily_signal_final_status,
            "paper_intent_final_status": self.paper_intent_final_status,
            "final_status": self.final_status,
            "ibridgepy_available": self.ibridgepy_available,
            "broker_connection_attempted": self.broker_connection_attempted,
            "broker_connection_status": self.broker_connection_status,
            "broker_data_status": self.broker_data_status,
            "paper_only": self.paper_only,
            "kill_switch": self.kill_switch,
            "live_orders_enabled": self.live_orders_enabled,
            "allow_order_placement": self.allow_order_placement,
            "live_order_sent": self.live_order_sent,
            "research_proxy_warning": self.research_proxy_warning,
        }


@dataclass(frozen=True)
class DiagnosticsWriteResult:
    """Paths written by diagnostics module."""

    broker_metadata_path: Path
    run_metadata_path: Path
    config_snapshot_path: Path | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "broker_metadata_path": str(self.broker_metadata_path),
            "run_metadata_path": str(self.run_metadata_path),
            "config_snapshot_path": (
                str(self.config_snapshot_path)
                if self.config_snapshot_path is not None
                else None
            ),
        }


def build_broker_metadata_record(
    config: BrokerConfig,
    *,
    broker_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized broker metadata record.

    If no broker metadata is supplied, this checks optional iBridgePy dependency
    without attempting a broker connection.
    """

    if broker_metadata is None:
        record = get_broker_metadata(config)
    else:
        record = dict(broker_metadata)

    record.setdefault("ibridgepy_available", False)
    record.setdefault("imported_module_name", None)
    record.setdefault("broker_connection_attempted", False)
    record.setdefault("broker_connection_status", IBRIDGEPY_NOT_INSTALLED)
    record.setdefault("broker_data_status", BROKER_DATA_UNAVAILABLE)
    record.setdefault("message", "")
    record.setdefault("checked_at_utc", _utc_now_iso())

    record["paper_only"] = config.paper_only
    record["kill_switch"] = config.kill_switch
    record["live_orders_enabled"] = config.live_orders_enabled
    record["allow_order_placement"] = config.allow_order_placement
    record["live_order_sent"] = bool(record.get("live_order_sent", False))
    record["research_proxy_warning"] = str(
        config.raw.get("research_proxy_warning", RESEARCH_PROXY_WARNING)
    )

    return _json_ready(record)


def build_run_metadata(
    *,
    config: BrokerConfig,
    market: str,
    strategy: str | None = None,
    input_signal_path: str | Path | None = None,
    daily_signal: DailyPaperSignal | None = None,
    paper_result: PaperIntentBuildResult | None = None,
    broker_metadata: Mapping[str, Any] | None = None,
    run_timestamp_utc: str | None = None,
) -> RunMetadata:
    """Build run metadata for one Phase 11 execution."""

    market_code = market.upper()
    active_strategy = strategy or config.default_strategy

    signal_path = (
        Path(input_signal_path)
        if input_signal_path is not None
        else get_market_signal_path(config, market_code)
    )

    broker_record = build_broker_metadata_record(
        config,
        broker_metadata=broker_metadata,
    )

    paper_intent_final_status = None
    paper_intent_live_order_sent = False

    if paper_result is not None and paper_result.intent is not None:
        paper_intent_final_status = paper_result.intent.final_status
        paper_intent_live_order_sent = bool(paper_result.intent.live_order_sent)

    daily_signal_final_status = daily_signal.final_status if daily_signal else None

    final_status = _resolve_final_status(
        daily_signal_final_status=daily_signal_final_status,
        paper_intent_final_status=paper_intent_final_status,
    )

    latest_target_trade_date = (
        daily_signal.target_trade_date
        if daily_signal is not None
        else None
    )

    live_order_sent = bool(
        paper_intent_live_order_sent
        or broker_record.get("live_order_sent", False)
    )

    return RunMetadata(
        run_timestamp_utc=run_timestamp_utc or _utc_now_iso(),
        market=market_code,
        strategy=active_strategy,
        config_path=str(config.config_path),
        config_hash=compute_config_hash(config),
        input_signal_path=str(signal_path),
        input_signal_mtime=_file_mtime_utc(signal_path),
        latest_target_trade_date=latest_target_trade_date,
        daily_signal_final_status=daily_signal_final_status,
        paper_intent_final_status=paper_intent_final_status,
        final_status=final_status,
        ibridgepy_available=bool(broker_record.get("ibridgepy_available", False)),
        broker_connection_attempted=bool(
            broker_record.get("broker_connection_attempted", False)
        ),
        broker_connection_status=str(
            broker_record.get("broker_connection_status", BROKER_CONNECTION_NOT_ATTEMPTED)
        ),
        broker_data_status=str(
            broker_record.get("broker_data_status", BROKER_DATA_UNAVAILABLE)
        ),
        paper_only=config.paper_only,
        kill_switch=config.kill_switch,
        live_orders_enabled=config.live_orders_enabled,
        allow_order_placement=config.allow_order_placement,
        live_order_sent=live_order_sent,
        research_proxy_warning=str(
            config.raw.get("research_proxy_warning", RESEARCH_PROXY_WARNING)
        ),
    )


def write_broker_metadata(
    metadata: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write broker metadata JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(path, metadata)
    return path


def write_run_metadata(
    metadata: RunMetadata | Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write run metadata JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = metadata.as_dict() if isinstance(metadata, RunMetadata) else dict(metadata)
    _write_json(path, payload)
    return path


def write_phase11_diagnostics(
    *,
    config: BrokerConfig,
    market: str,
    strategy: str | None = None,
    input_signal_path: str | Path | None = None,
    daily_signal: DailyPaperSignal | None = None,
    paper_result: PaperIntentBuildResult | None = None,
    broker_metadata: Mapping[str, Any] | None = None,
    run_timestamp_utc: str | None = None,
    broker_metadata_path: str | Path | None = None,
    run_metadata_path: str | Path | None = None,
    config_snapshot_path: str | Path | None = None,
) -> DiagnosticsWriteResult:
    """Write all Phase 11 diagnostics artifacts."""

    ensure_output_directories(config)

    output_paths = get_output_paths(config)

    if broker_metadata_path is None:
        broker_metadata_path = output_paths["broker_metadata"]

    if run_metadata_path is None:
        run_metadata_path = output_paths["run_metadata"]

    if config_snapshot_path is None:
        config_snapshot_path = output_paths["config_snapshot"]

    broker_record = build_broker_metadata_record(
        config,
        broker_metadata=broker_metadata,
    )

    run_record = build_run_metadata(
        config=config,
        market=market,
        strategy=strategy,
        input_signal_path=input_signal_path,
        daily_signal=daily_signal,
        paper_result=paper_result,
        broker_metadata=broker_record,
        run_timestamp_utc=run_timestamp_utc,
    )

    broker_path = write_broker_metadata(broker_record, broker_metadata_path)
    run_path = write_run_metadata(run_record, run_metadata_path)

    snapshot_path = None
    audit = config.raw.get("audit")
    write_snapshot = (
        bool(audit.get("write_config_snapshot", True))
        if isinstance(audit, Mapping)
        else True
    )

    if write_snapshot:
        snapshot_path = write_config_snapshot(config, config_snapshot_path)

    return DiagnosticsWriteResult(
        broker_metadata_path=broker_path,
        run_metadata_path=run_path,
        config_snapshot_path=snapshot_path,
    )


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read JSON object from path."""

    json_path = Path(path)

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise BrokerDiagnosticsError(f"Expected JSON object in {json_path}")

    return payload


def _resolve_final_status(
    *,
    daily_signal_final_status: str | None,
    paper_intent_final_status: str | None,
) -> str:
    """Resolve run-level final status.

    Paper-intent status is more specific when available.
    """

    if paper_intent_final_status:
        return paper_intent_final_status

    if daily_signal_final_status:
        return daily_signal_final_status

    return "BROKER_INSPECTION_ONLY"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    json_payload = _json_ready(dict(payload))

    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_ready(item) for item in value]

    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def _file_mtime_utc(path: Path) -> str | None:
    if not path.exists():
        return None

    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.replace(microsecond=0).isoformat()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()