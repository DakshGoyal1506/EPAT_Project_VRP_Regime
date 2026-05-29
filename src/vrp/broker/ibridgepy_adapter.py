"""
Optional iBridgePy adapter for Phase 11.

This module is intentionally narrow:
- import-safe when iBridgePy is not installed
- no broker connection by default
- no execution functions
- no live account action
- structured metadata only

Phase 11 is paper-signal and paper-intent only.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

from vrp.broker import BROKER_STATUS_TAXONOMY
from vrp.broker.broker_config import BrokerConfig


IBRIDGEPY_NOT_INSTALLED = "IBRIDGEPY_NOT_INSTALLED"
IBRIDGEPY_IMPORT_OK = "IBRIDGEPY_IMPORT_OK"
BROKER_CONNECTION_NOT_ATTEMPTED = "BROKER_CONNECTION_NOT_ATTEMPTED"
BROKER_CONNECTION_FAILED = "BROKER_CONNECTION_FAILED"
BROKER_DATA_UNAVAILABLE = "BROKER_DATA_UNAVAILABLE"
BROKER_DATA_AVAILABLE = "BROKER_DATA_AVAILABLE"


class BrokerAdapterError(RuntimeError):
    """Raised when optional broker inspection cannot proceed safely."""


@dataclass(frozen=True)
class BrokerAvailability:
    """Structured iBridgePy / broker availability state."""

    ibridgepy_available: bool
    imported_module_name: str | None
    broker_connection_attempted: bool
    broker_connection_status: str
    broker_data_status: str
    message: str
    checked_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ibridgepy_available": self.ibridgepy_available,
            "imported_module_name": self.imported_module_name,
            "broker_connection_attempted": self.broker_connection_attempted,
            "broker_connection_status": self.broker_connection_status,
            "broker_data_status": self.broker_data_status,
            "message": self.message,
            "checked_at_utc": self.checked_at_utc,
        }


@dataclass(frozen=True)
class BrokerInspectionResult:
    """Result of optional broker inspection.

    This does not include live account action and does not imply tradability.
    """

    availability: BrokerAvailability
    account: str
    host: str
    port: int
    client_id: int
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    live_order_sent: bool
    notes: str

    def as_dict(self) -> dict[str, Any]:
        result = self.availability.as_dict()
        result.update(
            {
                "account": self.account,
                "host": self.host,
                "port": self.port,
                "client_id": self.client_id,
                "paper_only": self.paper_only,
                "kill_switch": self.kill_switch,
                "live_orders_enabled": self.live_orders_enabled,
                "allow_order_placement": self.allow_order_placement,
                "live_order_sent": self.live_order_sent,
                "notes": self.notes,
            }
        )
        return result


class IBridgePyAdapter:
    """Import-safe adapter for optional iBridgePy inspection.

    This class deliberately has no methods for execution. It only reports
    whether the optional dependency is importable and records broker config
    metadata for audit reports.
    """

    def __init__(
        self,
        config: BrokerConfig,
        *,
        module_candidates: Iterable[str] | None = None,
        import_func: Callable[[str], ModuleType] | None = None,
    ) -> None:
        self.config = config
        self.module_candidates = tuple(module_candidates or ("IBridgePy", "ibridgepy"))
        self._import_func = import_func or importlib.import_module
        self._cached_module: ModuleType | None = None
        self._cached_module_name: str | None = None

    def check_availability(self) -> BrokerAvailability:
        """Check whether iBridgePy can be imported.

        No broker connection is attempted.
        """

        module_name, module = self._try_import_module()
        checked_at = _utc_now_iso()

        if module is None:
            return BrokerAvailability(
                ibridgepy_available=False,
                imported_module_name=None,
                broker_connection_attempted=False,
                broker_connection_status=IBRIDGEPY_NOT_INSTALLED,
                broker_data_status=BROKER_DATA_UNAVAILABLE,
                message=(
                    "iBridgePy is not installed or not importable. "
                    "Phase 11 can still generate paper signals and audit reports."
                ),
                checked_at_utc=checked_at,
            )

        return BrokerAvailability(
            ibridgepy_available=True,
            imported_module_name=module_name,
            broker_connection_attempted=False,
            broker_connection_status=IBRIDGEPY_IMPORT_OK,
            broker_data_status=BROKER_CONNECTION_NOT_ATTEMPTED,
            message=(
                "iBridgePy import succeeded. Broker connection was not attempted "
                "by this adapter."
            ),
            checked_at_utc=checked_at,
        )

    def inspect_broker_readiness(self, *, attempt_connection: bool = False) -> BrokerInspectionResult:
        """Return structured broker readiness metadata.

        attempt_connection defaults to False. Current Phase 11 keeps this as
        metadata-only; later chunks may use market-data wrappers, but this
        adapter itself does not perform account actions.
        """

        availability = self.check_availability()

        if attempt_connection:
            availability = BrokerAvailability(
                ibridgepy_available=availability.ibridgepy_available,
                imported_module_name=availability.imported_module_name,
                broker_connection_attempted=True,
                broker_connection_status=BROKER_CONNECTION_FAILED,
                broker_data_status=BROKER_DATA_UNAVAILABLE,
                message=(
                    "Connection attempt requested, but direct broker connection "
                    "is intentionally not implemented in the Phase 11 adapter. "
                    "Use a separate market-data inspection wrapper later."
                ),
                checked_at_utc=_utc_now_iso(),
            )

        notes = (
            "Phase 11 adapter is dependency-inspection only. It does not expose "
            "broker execution functions and does not send live orders."
        )

        return BrokerInspectionResult(
            availability=availability,
            account=self.config.broker.account,
            host=self.config.broker.host,
            port=self.config.broker.port,
            client_id=self.config.broker.client_id,
            paper_only=self.config.paper_only,
            kill_switch=self.config.kill_switch,
            live_orders_enabled=self.config.live_orders_enabled,
            allow_order_placement=self.config.allow_order_placement,
            live_order_sent=False,
            notes=notes,
        )

    def broker_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly broker metadata for Phase 11 reports."""

        return self.inspect_broker_readiness(attempt_connection=False).as_dict()

    def _try_import_module(self) -> tuple[str | None, ModuleType | None]:
        if self._cached_module is not None:
            return self._cached_module_name, self._cached_module

        for module_name in self.module_candidates:
            try:
                module = self._import_func(module_name)
            except ImportError:
                continue

            self._cached_module = module
            self._cached_module_name = module_name
            return module_name, module

        return None, None


def get_broker_availability(config: BrokerConfig) -> BrokerAvailability:
    """Convenience wrapper for dependency availability checks."""

    return IBridgePyAdapter(config).check_availability()


def get_broker_metadata(config: BrokerConfig) -> dict[str, Any]:
    """Convenience wrapper for broker metadata reports."""

    return IBridgePyAdapter(config).broker_metadata()


def validate_broker_status_value(status: str) -> str:
    """Validate a broker status against configured taxonomy constants."""

    if status not in BROKER_STATUS_TAXONOMY:
        raise BrokerAdapterError(f"Unknown broker status: {status}")

    return status


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()