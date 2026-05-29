"""
Phase 11 paper contract registry.

This module defines internal paper instrument specifications only. It does not
create broker orders, does not request broker execution, and does not assume
that any instrument is tradable in a real account.

The registry is intentionally conservative:
- US equity/ETP paper proxies are eligible for paper intent.
- India is signal-only unless manual verification is enabled in config.
- Options and futures are blocked by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from vrp.broker.broker_config import BrokerConfig


class ContractValidationError(ValueError):
    """Raised when a paper contract is not allowed by Phase 11 rules."""


PAPER_ALLOWED = "PAPER_ALLOWED"
PAPER_SIGNAL_ONLY = "PAPER_SIGNAL_ONLY"
PAPER_INSPECTION_ONLY = "PAPER_INSPECTION_ONLY"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ContractSpec:
    """Internal paper instrument specification.

    sec_type follows IBKR-style naming where relevant, but this object is not
    an IBKR Contract object.
    """

    symbol: str
    sec_type: str
    exchange: str
    currency: str
    multiplier: float
    market: str
    tradability_status: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sec_type": self.sec_type,
            "exchange": self.exchange,
            "currency": self.currency,
            "multiplier": self.multiplier,
            "market": self.market,
            "tradability_status": self.tradability_status,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class InstrumentMapping:
    """Config-backed mapping from research market to paper proxy."""

    market: str
    research_underlying: str
    default_paper_proxy: str
    short_vol_proxy_candidates: tuple[str, ...]
    notes: str

    @classmethod
    def from_config(
        cls,
        market: str,
        config: BrokerConfig,
    ) -> "InstrumentMapping":
        market_code = market.upper()
        mapping_root = config.raw.get("instrument_mapping")

        if not isinstance(mapping_root, Mapping):
            raise ContractValidationError("Config missing instrument_mapping section")

        market_mapping = mapping_root.get(market_code)

        if not isinstance(market_mapping, Mapping):
            raise ContractValidationError(
                f"Config missing instrument_mapping for market {market_code}"
            )

        candidates = market_mapping.get("short_vol_proxy_candidates", ())
        if candidates is None:
            candidates = ()

        return cls(
            market=market_code,
            research_underlying=str(market_mapping.get("research_underlying", "")),
            default_paper_proxy=str(market_mapping.get("default_paper_proxy", "")),
            short_vol_proxy_candidates=tuple(str(x) for x in candidates),
            notes=str(market_mapping.get("notes", "")),
        )


class BrokerInstrumentRegistry:
    """Registry of internal paper contract specs."""

    def __init__(self, contracts: Iterable[ContractSpec] | None = None) -> None:
        if contracts is None:
            contracts = default_contract_specs()

        self._contracts: dict[tuple[str, str], ContractSpec] = {}
        for contract in contracts:
            key = _contract_key(contract.market, contract.symbol)
            self._contracts[key] = contract

    def get(self, market: str, symbol: str) -> ContractSpec:
        key = _contract_key(market, symbol)

        try:
            return self._contracts[key]
        except KeyError as exc:
            raise ContractValidationError(
                f"Unknown contract symbol={symbol!r} for market={market.upper()!r}"
            ) from exc

    def list_market(self, market: str) -> tuple[ContractSpec, ...]:
        market_code = market.upper()
        return tuple(
            contract
            for (contract_market, _), contract in sorted(self._contracts.items())
            if contract_market == market_code
        )

    def symbols_for_market(self, market: str) -> tuple[str, ...]:
        return tuple(contract.symbol for contract in self.list_market(market))

    def as_records(self) -> list[dict[str, Any]]:
        return [contract.as_dict() for contract in self._contracts.values()]


def default_contract_specs() -> tuple[ContractSpec, ...]:
    """Return default internal Phase 11 paper contract specs.

    SPY, VXX, and SVXY are represented as US stock-like ETP paper proxies.
    NIFTY_PROXY_MANUAL_ONLY is intentionally signal-only by default.
    """

    return (
        ContractSpec(
            symbol="SPY",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            market="US",
            tradability_status=PAPER_ALLOWED,
            notes=(
                "US equity ETF paper proxy. Internal paper spec only; verify "
                "actual broker permissions before any future execution work."
            ),
        ),
        ContractSpec(
            symbol="VXX",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            market="US",
            tradability_status=PAPER_ALLOWED,
            notes=(
                "US volatility-linked ETN paper proxy candidate. Internal paper "
                "spec only; not equivalent to variance swap or option PnL."
            ),
        ),
        ContractSpec(
            symbol="SVXY",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            market="US",
            tradability_status=PAPER_ALLOWED,
            notes=(
                "US short-volatility ETP paper proxy candidate. Internal paper "
                "spec only; not equivalent to direct option-selling exposure."
            ),
        ),
        ContractSpec(
            symbol="NIFTY_PROXY_MANUAL_ONLY",
            sec_type="MANUAL",
            exchange="MANUAL",
            currency="INR",
            multiplier=1.0,
            market="INDIA",
            tradability_status=PAPER_SIGNAL_ONLY,
            notes=(
                "India signal-only placeholder. Broker intent remains blocked "
                "unless manual instrument verification is enabled in config."
            ),
        ),
    )


def get_allowed_contracts(
    market: str,
    config: BrokerConfig,
    registry: BrokerInstrumentRegistry | None = None,
) -> tuple[ContractSpec, ...]:
    """Return contracts allowed by config for a market.

    This returns config-allowed instruments. A returned contract may still be
    signal-only or inspection-only depending on its tradability status.
    """

    market_code = market.upper()
    _validate_market_allowed(market_code, config)

    registry = registry or BrokerInstrumentRegistry()
    allowed_symbols = _allowed_symbols_for_market(market_code, config)

    contracts = []
    for symbol in allowed_symbols:
        contract = registry.get(market_code, symbol)
        validate_contract_not_blocked(contract, config)
        contracts.append(contract)

    return tuple(contracts)


def get_default_proxy_contract(
    market: str,
    config: BrokerConfig,
    registry: BrokerInstrumentRegistry | None = None,
) -> ContractSpec:
    """Return configured default paper proxy for a market."""

    market_code = market.upper()
    _validate_market_allowed(market_code, config)

    registry = registry or BrokerInstrumentRegistry()
    mapping = InstrumentMapping.from_config(market_code, config)

    if not mapping.default_paper_proxy:
        raise ContractValidationError(
            f"No default paper proxy configured for market {market_code}"
        )

    contract = registry.get(market_code, mapping.default_paper_proxy)
    validate_contract_allowed(contract, config)
    return contract


def get_short_vol_proxy_candidates(
    market: str,
    config: BrokerConfig,
    registry: BrokerInstrumentRegistry | None = None,
) -> tuple[ContractSpec, ...]:
    """Return configured short-vol paper proxy candidates.

    For India this can be empty because the default is manual signal-only.
    """

    market_code = market.upper()
    _validate_market_allowed(market_code, config)

    registry = registry or BrokerInstrumentRegistry()
    mapping = InstrumentMapping.from_config(market_code, config)

    candidates: list[ContractSpec] = []
    for symbol in mapping.short_vol_proxy_candidates:
        contract = registry.get(market_code, symbol)
        validate_contract_allowed(contract, config)
        candidates.append(contract)

    return tuple(candidates)


def choose_paper_proxy_for_action(
    market: str,
    recommended_action: str,
    config: BrokerConfig,
    registry: BrokerInstrumentRegistry | None = None,
) -> ContractSpec:
    """Choose a conservative paper proxy for a Phase 11 action.

    For PAPER_SHORT_VOL_INTENT, prefer the first configured short-vol proxy
    candidate. If no candidate exists, use the default paper proxy.
    """

    market_code = market.upper()
    action = str(recommended_action).upper()
    registry = registry or BrokerInstrumentRegistry()

    if action == "PAPER_SHORT_VOL_INTENT":
        candidates = get_short_vol_proxy_candidates(market_code, config, registry)
        if candidates:
            return candidates[0]

    return get_default_proxy_contract(market_code, config, registry)


def validate_contract_allowed(
    contract: ContractSpec,
    config: BrokerConfig,
) -> ContractSpec:
    """Validate that a contract is allowed by market, symbol, and status."""

    _validate_market_allowed(contract.market, config)

    allowed_symbols = _allowed_symbols_for_market(contract.market, config)
    if contract.symbol not in allowed_symbols:
        raise ContractValidationError(
            f"Contract {contract.symbol!r} is not listed under "
            f"allowed_instruments for market {contract.market!r}"
        )

    validate_contract_not_blocked(contract, config)

    if contract.tradability_status == BLOCKED:
        raise ContractValidationError(
            f"Contract {contract.symbol!r} has blocked tradability status"
        )

    if contract.market.upper() == "INDIA":
        _validate_india_manual_gate(contract, config)

    return contract


def validate_contract_not_blocked(
    contract: ContractSpec,
    config: BrokerConfig,
) -> ContractSpec:
    """Validate that contract does not violate blocked-product rules."""

    blocked_instruments = {
        str(item).lower() for item in config.raw.get("blocked_instruments", ())
    }

    sec_type = contract.sec_type.upper()

    if sec_type in {"OPT", "FOP"}:
        if "naked_short_options" in blocked_instruments:
            raise ContractValidationError(
                f"Options are blocked by default in Phase 11: {contract.symbol}"
            )

    if sec_type == "FUT":
        if "futures_without_permission" in blocked_instruments:
            raise ContractValidationError(
                f"Futures are blocked by default in Phase 11: {contract.symbol}"
            )

    if contract.tradability_status == BLOCKED:
        raise ContractValidationError(
            f"Contract is explicitly blocked: {contract.symbol}"
        )

    return contract


def is_broker_intent_eligible(
    contract: ContractSpec,
    config: BrokerConfig,
) -> bool:
    """Return whether a contract may become an allowed paper broker intent."""

    try:
        validate_contract_allowed(contract, config)
    except ContractValidationError:
        return False

    return contract.tradability_status == PAPER_ALLOWED


def contract_status_reason(
    contract: ContractSpec,
    config: BrokerConfig,
) -> str:
    """Return human-readable reason for contract readiness status."""

    try:
        validate_contract_allowed(contract, config)
    except ContractValidationError as exc:
        return str(exc)

    if contract.tradability_status == PAPER_ALLOWED:
        return "contract is eligible for paper intent subject to risk checks"

    if contract.tradability_status == PAPER_SIGNAL_ONLY:
        return "contract is signal-only and cannot produce broker intent by default"

    if contract.tradability_status == PAPER_INSPECTION_ONLY:
        return "contract is broker-inspection-only"

    return f"contract status is {contract.tradability_status}"


def _contract_key(market: str, symbol: str) -> tuple[str, str]:
    return market.upper(), symbol.upper()


def _validate_market_allowed(market: str, config: BrokerConfig) -> None:
    market_code = market.upper()
    allowed_markets = {str(item).upper() for item in config.raw.get("allowed_markets", ())}

    if market_code not in allowed_markets:
        raise ContractValidationError(
            f"Market {market_code!r} is not allowed. "
            f"Allowed markets: {sorted(allowed_markets)}"
        )


def _allowed_symbols_for_market(market: str, config: BrokerConfig) -> tuple[str, ...]:
    market_code = market.upper()
    allowed_instruments = config.raw.get("allowed_instruments")

    if not isinstance(allowed_instruments, Mapping):
        raise ContractValidationError("Config missing allowed_instruments section")

    symbols = allowed_instruments.get(market_code)
    if symbols is None:
        raise ContractValidationError(
            f"Config missing allowed_instruments for market {market_code}"
        )

    return tuple(str(symbol).upper() for symbol in symbols)


def _validate_india_manual_gate(
    contract: ContractSpec,
    config: BrokerConfig,
) -> None:
    if contract.symbol != "NIFTY_PROXY_MANUAL_ONLY":
        return

    manual_overrides = config.raw.get("manual_overrides")
    if not isinstance(manual_overrides, Mapping):
        manual_verified = False
    else:
        india_overrides = manual_overrides.get("INDIA")
        if not isinstance(india_overrides, Mapping):
            manual_verified = False
        else:
            manual_verified = bool(india_overrides.get("manual_instrument_verified", False))

    if not manual_verified:
        raise ContractValidationError(
            "NIFTY_PROXY_MANUAL_ONLY is signal-only by default. "
            "Set manual_overrides.INDIA.manual_instrument_verified=true only "
            "after separate broker/instrument verification."
        )