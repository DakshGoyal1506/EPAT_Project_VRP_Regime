"""
Phase 11 broker configuration loader and validator.

This module validates that the broker-readiness layer remains paper-only,
fail-closed, and safe to run without iBridgePy or IBKR connectivity.

It does not connect to a broker.
It does not place orders.
It does not infer live sizing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from vrp.broker import (
    APPROVED_STRATEGIES,
    FINAL_STATUS_TAXONOMY,
    FORBIDDEN_ACTIVE_STRATEGY_MODELS,
    MODE,
    PHASE,
    RESEARCH_PROXY_WARNING,
)


class BrokerConfigError(ValueError):
    """Raised when Phase 11 broker configuration is unsafe or invalid."""


@dataclass(frozen=True)
class BrokerConnectionConfig:
    """Broker connection settings.

    These are placeholders for optional broker inspection only. They are not
    sufficient to enable live order placement.
    """

    provider: str
    adapter: str
    optional_dependency: bool
    host: str
    port: int
    client_id: int
    account: str
    require_env_account: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BrokerConnectionConfig":
        return cls(
            provider=str(data.get("provider", "")),
            adapter=str(data.get("adapter", "")),
            optional_dependency=bool(data.get("optional_dependency", True)),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 7497)),
            client_id=int(data.get("client_id", 11)),
            account=str(data.get("account", "PAPER_ACCOUNT_PLACEHOLDER")),
            require_env_account=bool(data.get("require_env_account", True)),
        )


@dataclass(frozen=True)
class PaperSizingConfig:
    """Paper sizing limits.

    Phase 11 sizing is intentionally narrow:
    abs(target_exposure) * configured paper notional.
    """

    paper_notional_per_full_exposure: float
    min_order_notional: float
    round_shares: bool
    allow_fractional_shares: bool
    max_contracts: int
    max_shares: int
    max_notional: float
    max_delta: float
    max_vega: float
    max_margin_usage: float
    allow_options: bool
    allow_naked_short_options: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PaperSizingConfig":
        return cls(
            paper_notional_per_full_exposure=float(
                data.get("paper_notional_per_full_exposure", 0.0)
            ),
            min_order_notional=float(data.get("min_order_notional", 0.0)),
            round_shares=bool(data.get("round_shares", True)),
            allow_fractional_shares=bool(data.get("allow_fractional_shares", False)),
            max_contracts=int(data.get("max_contracts", 0)),
            max_shares=int(data.get("max_shares", 0)),
            max_notional=float(data.get("max_notional", 0.0)),
            max_delta=float(data.get("max_delta", 0.0)),
            max_vega=float(data.get("max_vega", 0.0)),
            max_margin_usage=float(data.get("max_margin_usage", 0.0)),
            allow_options=bool(data.get("allow_options", False)),
            allow_naked_short_options=bool(
                data.get("allow_naked_short_options", False)
            ),
        )


@dataclass(frozen=True)
class RiskCheckConfig:
    """Risk check switches and limits."""

    require_paper_only: bool
    require_kill_switch_off_for_allowed_intent: bool
    block_if_kill_switch_on: bool
    require_market_open: bool
    max_quote_age_seconds: int
    max_bid_ask_spread_bps: float
    require_quote_for_allowed_intent: bool
    block_stale_quote_if_quote_available: bool
    block_missing_quote: bool
    block_live_order_functions: bool
    block_stale_signal: bool
    block_naked_short_options: bool
    block_india_unverified_instrument: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RiskCheckConfig":
        return cls(
            require_paper_only=bool(data.get("require_paper_only", True)),
            require_kill_switch_off_for_allowed_intent=bool(
                data.get("require_kill_switch_off_for_allowed_intent", False)
            ),
            block_if_kill_switch_on=bool(data.get("block_if_kill_switch_on", True)),
            require_market_open=bool(data.get("require_market_open", False)),
            max_quote_age_seconds=int(data.get("max_quote_age_seconds", 300)),
            max_bid_ask_spread_bps=float(data.get("max_bid_ask_spread_bps", 100.0)),
            require_quote_for_allowed_intent=bool(
                data.get("require_quote_for_allowed_intent", False)
            ),
            block_stale_quote_if_quote_available=bool(
                data.get("block_stale_quote_if_quote_available", True)
            ),
            block_missing_quote=bool(data.get("block_missing_quote", False)),
            block_live_order_functions=bool(
                data.get("block_live_order_functions", True)
            ),
            block_stale_signal=bool(data.get("block_stale_signal", True)),
            block_naked_short_options=bool(
                data.get("block_naked_short_options", True)
            ),
            block_india_unverified_instrument=bool(
                data.get("block_india_unverified_instrument", True)
            ),
        )


@dataclass(frozen=True)
class BrokerConfig:
    """Validated Phase 11 config object."""

    config_path: Path
    raw: dict[str, Any]
    phase: str
    mode: str
    paper_only: bool
    kill_switch: bool
    live_orders_enabled: bool
    allow_order_placement: bool
    default_strategy: str
    approved_strategies: tuple[str, ...]
    forbidden_active_strategy_models: tuple[str, ...]
    broker: BrokerConnectionConfig
    paper_sizing: PaperSizingConfig
    risk_checks: RiskCheckConfig

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        config_path: str | Path,
    ) -> "BrokerConfig":
        raw = copy.deepcopy(dict(data))

        broker_data = _require_mapping(raw, "broker")
        sizing_data = _require_mapping(raw, "paper_sizing")
        risk_data = _require_mapping(raw, "risk_checks")

        approved = tuple(
            str(x) for x in raw.get("approved_strategies", APPROVED_STRATEGIES)
        )
        forbidden = tuple(
            str(x).lower()
            for x in raw.get(
                "forbidden_active_strategy_models",
                FORBIDDEN_ACTIVE_STRATEGY_MODELS,
            )
        )

        return cls(
            config_path=Path(config_path),
            raw=raw,
            phase=str(raw.get("phase", "")),
            mode=str(raw.get("mode", "")),
            paper_only=bool(raw.get("paper_only", False)),
            kill_switch=bool(raw.get("kill_switch", False)),
            live_orders_enabled=bool(raw.get("live_orders_enabled", True)),
            allow_order_placement=bool(raw.get("allow_order_placement", True)),
            default_strategy=str(raw.get("default_strategy", "")),
            approved_strategies=approved,
            forbidden_active_strategy_models=forbidden,
            broker=BrokerConnectionConfig.from_mapping(broker_data),
            paper_sizing=PaperSizingConfig.from_mapping(sizing_data),
            risk_checks=RiskCheckConfig.from_mapping(risk_data),
        )


def load_broker_config(path: str | Path) -> BrokerConfig:
    """Load, apply environment overrides, and validate Phase 11 config."""

    config_path = Path(path)
    if not config_path.exists():
        raise BrokerConfigError(f"Broker config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise BrokerConfigError(f"Broker config must be a mapping: {config_path}")

    config = BrokerConfig.from_mapping(data, config_path=config_path)
    config = load_env_overrides(config)
    validate_broker_config(config)
    return config


def load_env_overrides(config: BrokerConfig) -> BrokerConfig:
    """Apply optional environment overrides to broker connection fields.

    Missing environment variables are not fatal. Phase 11 must run without
    iBridgePy and without a broker account configured.
    """

    raw = copy.deepcopy(config.raw)
    env_config = _require_mapping(raw, "env")
    broker_config = cast(dict[str, Any], _require_mapping(raw, "broker"))

    account_var = str(env_config.get("account_var", "IBKR_PAPER_ACCOUNT"))
    host_var = str(env_config.get("host_var", "IBKR_HOST"))
    port_var = str(env_config.get("port_var", "IBKR_PORT"))
    client_id_var = str(env_config.get("client_id_var", "IBKR_CLIENT_ID"))

    account_value = os.getenv(account_var)
    host_value = os.getenv(host_var)
    port_value = os.getenv(port_var)
    client_id_value = os.getenv(client_id_var)

    if account_value:
        broker_config["account"] = account_value

    if host_value:
        broker_config["host"] = host_value

    if port_value:
        broker_config["port"] = _parse_int_env(port_var, port_value)

    if client_id_value:
        broker_config["client_id"] = _parse_int_env(client_id_var, client_id_value)

    return BrokerConfig.from_mapping(raw, config_path=config.config_path)


def validate_broker_config(config: BrokerConfig) -> BrokerConfig:
    """Validate hard safety invariants.

    Returns the config for convenient chaining. Raises BrokerConfigError if any
    invariant fails.
    """

    errors: list[str] = []

    if config.phase != PHASE:
        errors.append(f"phase must be {PHASE!r}, got {config.phase!r}")

    if config.mode != MODE:
        errors.append(f"mode must be {MODE!r}, got {config.mode!r}")

    if config.paper_only is not True:
        errors.append("paper_only must be true")

    if config.live_orders_enabled is not False:
        errors.append("live_orders_enabled must be false")

    if config.allow_order_placement is not False:
        errors.append("allow_order_placement must be false")

    if config.default_strategy.lower() in config.forbidden_active_strategy_models:
        errors.append(
            f"default_strategy cannot be a forbidden active model: "
            f"{config.default_strategy}"
        )

    if config.default_strategy not in config.approved_strategies:
        errors.append(
            f"default_strategy must be in approved strategy universe: "
            f"{config.default_strategy}"
        )

    for forbidden in FORBIDDEN_ACTIVE_STRATEGY_MODELS:
        if forbidden in {x.lower() for x in config.approved_strategies}:
            errors.append(f"approved_strategies cannot include forbidden model {forbidden}")

    sizing = config.paper_sizing

    if sizing.allow_naked_short_options is not False:
        errors.append("paper_sizing.allow_naked_short_options must be false")

    if sizing.allow_options is False and sizing.max_contracts != 0:
        errors.append("max_contracts must be 0 when allow_options is false")

    if sizing.max_margin_usage != 0.0:
        errors.append("max_margin_usage must be 0.0 in Phase 11 default safety mode")

    if sizing.paper_notional_per_full_exposure < 0:
        errors.append("paper_notional_per_full_exposure cannot be negative")

    if sizing.max_notional < 0:
        errors.append("max_notional cannot be negative")

    if sizing.max_shares < 0:
        errors.append("max_shares cannot be negative")

    risk = config.risk_checks

    if risk.require_paper_only is not True:
        errors.append("risk_checks.require_paper_only must be true")

    if risk.block_if_kill_switch_on is not True:
        errors.append("risk_checks.block_if_kill_switch_on must be true")

    if risk.block_live_order_functions is not True:
        errors.append("risk_checks.block_live_order_functions must be true")

    if risk.block_naked_short_options is not True:
        errors.append("risk_checks.block_naked_short_options must be true")

    signal_freshness = _require_mapping(config.raw, "signal_freshness")
    max_signal_age_days = int(signal_freshness.get("max_signal_age_days", 0))

    if max_signal_age_days <= 0:
        errors.append("signal_freshness.max_signal_age_days must be positive")

    if bool(signal_freshness.get("block_stale_signal", True)) is not True:
        errors.append("signal_freshness.block_stale_signal must be true")

    outputs = _require_mapping(config.raw, "outputs")
    required_output_keys = {
        "latest_signal_table",
        "paper_order_intents",
        "risk_check_report",
        "broker_metadata",
        "run_metadata",
        "config_snapshot",
    }
    missing_output_keys = sorted(required_output_keys.difference(outputs.keys()))

    if missing_output_keys:
        errors.append(f"outputs missing required keys: {missing_output_keys}")

    signal_inputs = _require_mapping(config.raw, "signal_inputs")
    allowed_markets = tuple(str(x).upper() for x in config.raw.get("allowed_markets", ()))

    for market in allowed_markets:
        if market not in signal_inputs:
            errors.append(f"signal_inputs missing path for market {market}")

    status_taxonomy = tuple(str(x) for x in config.raw.get("final_status_taxonomy", ()))
    for required_status in FINAL_STATUS_TAXONOMY:
        if required_status not in status_taxonomy:
            errors.append(f"final_status_taxonomy missing {required_status}")

    warning = str(config.raw.get("research_proxy_warning", ""))
    if "research-layer proxy units" not in warning:
        errors.append("research_proxy_warning must state that Phase 10 is proxy-only")

    if RESEARCH_PROXY_WARNING != warning:
        errors.append("research_proxy_warning must match package-level warning text")

    audit = _require_mapping(config.raw, "audit")
    write_snapshot = audit.get("write_config_snapshot", False)
    if not isinstance(write_snapshot, bool):
        errors.append("audit.write_config_snapshot must be a boolean")

    if bool(audit.get("write_run_metadata", False)) is not True:
        errors.append("audit.write_run_metadata must be true")

    if bool(audit.get("include_phase10_proxy_warning", False)) is not True:
        errors.append("audit.include_phase10_proxy_warning must be true")

    if errors:
        joined = "\n- ".join(errors)
        raise BrokerConfigError(f"Invalid Phase 11 broker config:\n- {joined}")

    return config


def get_market_signal_path(config: BrokerConfig, market: str) -> Path:
    """Return configured Phase 9 signal path for a market."""

    market_code = market.upper()
    allowed_markets = {str(x).upper() for x in config.raw.get("allowed_markets", ())}

    if market_code not in allowed_markets:
        raise BrokerConfigError(
            f"Market {market_code!r} is not allowed. "
            f"Allowed markets: {sorted(allowed_markets)}"
        )

    signal_inputs = _require_mapping(config.raw, "signal_inputs")

    if market_code not in signal_inputs:
        raise BrokerConfigError(f"No signal input configured for market {market_code!r}")

    return Path(str(signal_inputs[market_code]))


def get_output_paths(config: BrokerConfig) -> dict[str, Path]:
    """Return configured output paths as Path objects."""

    outputs = _require_mapping(config.raw, "outputs")
    return {str(key): Path(str(value)) for key, value in outputs.items()}


def compute_config_hash(config: BrokerConfig) -> str:
    """Compute a stable hash of the effective config."""

    payload = json.dumps(config.raw, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_config_snapshot(
    config: BrokerConfig,
    output_path: str | Path | None = None,
    *,
    redact_sensitive: bool = True,
) -> Path:
    """Write the effective config used for the run.

    Broker account is redacted by default when it came from an environment
    override or differs from the placeholder.
    """

    if output_path is None:
        output_paths = get_output_paths(config)
        output_path = output_paths["config_snapshot"]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = _redacted_raw_config(config.raw) if redact_sensitive else config.raw

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)

    return path


def ensure_output_directories(config: BrokerConfig) -> None:
    """Create configured output directories."""

    output_paths = get_output_paths(config)

    for key, path in output_paths.items():
        if key.endswith("_dir") or path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise BrokerConfigError(f"Missing or invalid mapping in config: {key}")
    return value


def _parse_int_env(var_name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise BrokerConfigError(
            f"Environment variable {var_name} must be an integer, got {value!r}"
        ) from exc


def _redacted_raw_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(dict(raw))
    broker = redacted.get("broker")

    if isinstance(broker, dict):
        account = str(broker.get("account", ""))
        if account and account != "PAPER_ACCOUNT_PLACEHOLDER":
            broker["account"] = "REDACTED_IBKR_PAPER_ACCOUNT"

    return redacted