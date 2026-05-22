from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vrp.strategies.strategy_registry import (
    ALLOWED_STRATEGY_MODELS,
    APPROVED_STRATEGY_NAMES,
    FORBIDDEN_STRATEGY_MODELS,
    REJECTED_STRATEGY_NAMES,
    assert_model_allowed_for_strategy,
    assert_no_msvol_strategy_use,
    validate_strategy_model_map,
    validate_strategy_names,
)


@dataclass(frozen=True)
class ExposureBounds:
    min_exposure: float
    max_exposure: float


@dataclass(frozen=True)
class TimingPolicy:
    signal_computed_after_close: bool
    threshold_shift_to_next_trade_date: bool
    hmm_next_session_columns_are_pre_shifted: bool
    mar_next_session_columns_are_pre_shifted: bool
    do_not_double_shift_hmm_or_mar: bool


@dataclass(frozen=True)
class FrozenConstants:
    transition_exposure: float
    stress_probability_cutoff: float
    probability_sum_tolerance: float
    primary_probability_rule: str
    carry_gate_vrp_column: str
    carry_gate_operator: str
    carry_gate_threshold: float
    use_vrp_har_positive_as_source_of_truth: bool


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    active: bool
    model_family: str
    rule_name: str
    requires_probabilities: bool
    requires_har: bool
    probability_rule: str | None
    transition_exposure: float | None
    max_stress_probability: float | None
    require_vrp_har_positive: bool
    description: str


@dataclass(frozen=True)
class StrategyConfig:
    phase: int
    version: int
    description: str
    report_note: str
    exposure_bounds: ExposureBounds
    timing_policy: TimingPolicy
    frozen_constants: FrozenConstants
    msvol_policy: dict[str, Any]
    allowed_strategy_models: tuple[str, ...]
    forbidden_strategy_models: tuple[str, ...]
    approved_strategy_names: tuple[str, ...]
    rejected_strategy_names: tuple[str, ...]
    input_files: dict[str, dict[str, str]]
    output_files: dict[str, str]
    report_files: dict[str, str]
    strategies: dict[str, StrategyDefinition]
    raw: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if payload is None:
        raise ValueError(f"Strategy config is empty: {path}")

    if not isinstance(payload, dict):
        raise TypeError(f"Strategy config must parse to a mapping: {path}")

    return payload


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Config key '{key}' must be a mapping.")
    return value


def _require_sequence(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Config key '{key}' must be a list.")
    return tuple(str(item) for item in value)


def _require_float(raw: dict[str, Any], key: str) -> float:
    if key not in raw:
        raise KeyError(f"Missing required numeric config key '{key}'.")
    value = raw[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Config key '{key}' must be numeric.")
    return float(value)


def _require_int(raw: dict[str, Any], key: str) -> int:
    if key not in raw:
        raise KeyError(f"Missing required integer config key '{key}'.")
    value = raw[key]
    if not isinstance(value, int):
        raise TypeError(f"Config key '{key}' must be an integer.")
    return value


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    if key not in raw:
        raise KeyError(f"Missing required boolean config key '{key}'.")
    value = raw[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{key}' must be boolean.")
    return value


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"Config key '{key}' must be numeric or null.")
    return float(value)


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return str(value)


def _parse_exposure_bounds(raw: dict[str, Any]) -> ExposureBounds:
    bounds = _require_mapping(raw, "exposure_bounds")
    return ExposureBounds(
        min_exposure=_require_float(bounds, "min_exposure"),
        max_exposure=_require_float(bounds, "max_exposure"),
    )


def _parse_timing_policy(raw: dict[str, Any]) -> TimingPolicy:
    timing = _require_mapping(raw, "timing_policy")
    return TimingPolicy(
        signal_computed_after_close=_require_bool(
            timing, "signal_computed_after_close"
        ),
        threshold_shift_to_next_trade_date=_require_bool(
            timing, "threshold_shift_to_next_trade_date"
        ),
        hmm_next_session_columns_are_pre_shifted=_require_bool(
            timing, "hmm_next_session_columns_are_pre_shifted"
        ),
        mar_next_session_columns_are_pre_shifted=_require_bool(
            timing, "mar_next_session_columns_are_pre_shifted"
        ),
        do_not_double_shift_hmm_or_mar=_require_bool(
            timing, "do_not_double_shift_hmm_or_mar"
        ),
    )


def _parse_frozen_constants(raw: dict[str, Any]) -> FrozenConstants:
    constants = _require_mapping(raw, "frozen_constants")
    return FrozenConstants(
        transition_exposure=_require_float(constants, "transition_exposure"),
        stress_probability_cutoff=_require_float(
            constants, "stress_probability_cutoff"
        ),
        probability_sum_tolerance=_require_float(
            constants, "probability_sum_tolerance"
        ),
        primary_probability_rule=str(constants.get("primary_probability_rule")),
        carry_gate_vrp_column=str(constants.get("carry_gate_vrp_column")),
        carry_gate_operator=str(constants.get("carry_gate_operator")),
        carry_gate_threshold=_require_float(constants, "carry_gate_threshold"),
        use_vrp_har_positive_as_source_of_truth=_require_bool(
            constants, "use_vrp_har_positive_as_source_of_truth"
        ),
    )


def _parse_strategy_definition(name: str, raw: dict[str, Any]) -> StrategyDefinition:
    if not isinstance(raw, dict):
        raise TypeError(f"Strategy definition for '{name}' must be a mapping.")

    return StrategyDefinition(
        name=name,
        active=_require_bool(raw, "active"),
        model_family=str(raw.get("model_family")),
        rule_name=str(raw.get("rule_name")),
        requires_probabilities=_require_bool(raw, "requires_probabilities"),
        requires_har=_require_bool(raw, "requires_har"),
        probability_rule=_optional_string(raw, "probability_rule"),
        transition_exposure=_optional_float(raw, "transition_exposure"),
        max_stress_probability=_optional_float(raw, "max_stress_probability"),
        require_vrp_har_positive=_require_bool(raw, "require_vrp_har_positive"),
        description=str(raw.get("description", "")),
    )


def _parse_strategies(raw: dict[str, Any]) -> dict[str, StrategyDefinition]:
    strategies_raw = _require_mapping(raw, "strategies")
    return {
        str(name): _parse_strategy_definition(str(name), strategy_raw)
        for name, strategy_raw in strategies_raw.items()
    }


def load_strategy_config(path: str | Path) -> StrategyConfig:
    """
    Load and validate Phase 9 strategy configuration.
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)

    config = StrategyConfig(
        phase=_require_int(raw, "phase"),
        version=_require_int(raw, "version"),
        description=str(raw.get("description", "")),
        report_note=str(raw.get("report_note", "")),
        exposure_bounds=_parse_exposure_bounds(raw),
        timing_policy=_parse_timing_policy(raw),
        frozen_constants=_parse_frozen_constants(raw),
        msvol_policy=_require_mapping(raw, "msvol_policy"),
        allowed_strategy_models=_require_sequence(raw, "allowed_strategy_models"),
        forbidden_strategy_models=_require_sequence(raw, "forbidden_strategy_models"),
        approved_strategy_names=_require_sequence(raw, "approved_strategy_names"),
        rejected_strategy_names=_require_sequence(raw, "rejected_strategy_names"),
        input_files={
            str(market).upper(): {
                str(source_name): str(source_path)
                for source_name, source_path in source_paths.items()
            }
            for market, source_paths in _require_mapping(raw, "input_files").items()
        },
        output_files={
            str(market).upper(): str(path_value)
            for market, path_value in _require_mapping(raw, "output_files").items()
        },
        report_files={
            str(name): str(path_value)
            for name, path_value in _require_mapping(raw, "report_files").items()
        },
        strategies=_parse_strategies(raw),
        raw=raw,
    )

    validate_strategy_config(config)
    return config


def validate_strategy_config(config: StrategyConfig) -> None:
    """
    Validate the Phase 9 config contract.

    This deliberately fails loudly if the strategy universe, frozen constants,
    exposure bounds, timing flags, or model-family rules drift from the approved
    Phase 9 design.
    """
    if config.phase != 9:
        raise ValueError(f"Expected phase=9, got phase={config.phase}.")

    if set(config.approved_strategy_names) != set(APPROVED_STRATEGY_NAMES):
        raise ValueError(
            "Config approved_strategy_names does not match registry-approved "
            f"strategy names. Config={sorted(config.approved_strategy_names)}, "
            f"Registry={sorted(APPROVED_STRATEGY_NAMES)}."
        )

    if set(config.rejected_strategy_names) != set(REJECTED_STRATEGY_NAMES):
        raise ValueError(
            "Config rejected_strategy_names does not match registry rejected "
            f"strategy names. Config={sorted(config.rejected_strategy_names)}, "
            f"Registry={sorted(REJECTED_STRATEGY_NAMES)}."
        )

    if set(config.allowed_strategy_models) != set(ALLOWED_STRATEGY_MODELS):
        raise ValueError(
            "Config allowed_strategy_models does not match registry allowed "
            f"models. Config={sorted(config.allowed_strategy_models)}, "
            f"Registry={sorted(ALLOWED_STRATEGY_MODELS)}."
        )

    if set(config.forbidden_strategy_models) != set(FORBIDDEN_STRATEGY_MODELS):
        raise ValueError(
            "Config forbidden_strategy_models does not match registry forbidden "
            f"models. Config={sorted(config.forbidden_strategy_models)}, "
            f"Registry={sorted(FORBIDDEN_STRATEGY_MODELS)}."
        )

    active_strategies = {
        name: strategy
        for name, strategy in config.strategies.items()
        if strategy.active
    }

    validate_strategy_names(active_strategies.keys())

    validate_strategy_model_map(
        {
            strategy.name: strategy.model_family
            for strategy in active_strategies.values()
        }
    )

    if config.exposure_bounds.min_exposure != -1.0:
        raise ValueError("Phase 9 requires min_exposure == -1.0.")

    if config.exposure_bounds.max_exposure != 0.0:
        raise ValueError("Phase 9 requires max_exposure == 0.0.")

    if not (-1.0 <= config.frozen_constants.transition_exposure <= 0.0):
        raise ValueError("transition_exposure must lie between -1.0 and 0.0.")

    if config.frozen_constants.transition_exposure != -0.25:
        raise ValueError("Phase 9 frozen transition_exposure must equal -0.25.")

    if not (0.0 <= config.frozen_constants.stress_probability_cutoff <= 1.0):
        raise ValueError("stress_probability_cutoff must lie between 0.0 and 1.0.")

    if config.frozen_constants.stress_probability_cutoff != 0.40:
        raise ValueError("Phase 9 frozen stress_probability_cutoff must equal 0.40.")

    if not (0.0 < config.frozen_constants.probability_sum_tolerance < 0.01):
        raise ValueError("probability_sum_tolerance must be positive and small.")

    if config.frozen_constants.probability_sum_tolerance != 0.001:
        raise ValueError("Phase 9 frozen probability_sum_tolerance must equal 0.001.")

    if config.frozen_constants.primary_probability_rule != "calm_minus_stress":
        raise ValueError("Phase 9 primary_probability_rule must be calm_minus_stress.")

    if config.frozen_constants.carry_gate_vrp_column != "vrp_har_gk":
        raise ValueError("Carry gate must use numeric vrp_har_gk as source of truth.")

    if config.frozen_constants.carry_gate_operator != "gt":
        raise ValueError("Carry gate operator must be 'gt'.")

    if config.frozen_constants.carry_gate_threshold != 0.0:
        raise ValueError("Carry gate threshold must be 0.0.")

    if config.frozen_constants.use_vrp_har_positive_as_source_of_truth:
        raise ValueError(
            "vrp_har_gk_positive must not be used as source of truth. "
            "Use numeric vrp_har_gk > 0."
        )

    if not config.timing_policy.signal_computed_after_close:
        raise ValueError("Signals must be computed after close.")

    if not config.timing_policy.threshold_shift_to_next_trade_date:
        raise ValueError("Threshold regimes must be shifted to next trade date.")

    if not config.timing_policy.hmm_next_session_columns_are_pre_shifted:
        raise ValueError("HMM next-session-safe columns must be treated as pre-shifted.")

    if not config.timing_policy.mar_next_session_columns_are_pre_shifted:
        raise ValueError("MAR next-session-safe columns must be treated as pre-shifted.")

    if not config.timing_policy.do_not_double_shift_hmm_or_mar:
        raise ValueError("HMM/MAR signals must not be double-shifted.")

    _validate_msvol_policy(config)
    _validate_strategy_definitions(config)
    _validate_paths(config)


def _validate_msvol_policy(config: StrategyConfig) -> None:
    policy = config.msvol_policy

    if str(policy.get("status")) != "excluded_diagnostic_only":
        raise ValueError("msvol_policy.status must be excluded_diagnostic_only.")

    forbidden_true_flags = [
        "read_input_files",
        "hash_input_files",
        "merge_columns",
        "include_in_diagnostics",
    ]

    for key in forbidden_true_flags:
        if bool(policy.get(key)):
            raise ValueError(f"msvol_policy.{key} must be false in Phase 9.")


def _validate_strategy_definitions(config: StrategyConfig) -> None:
    for strategy in config.strategies.values():
        strategy_name = strategy.name

        if strategy_name in REJECTED_STRATEGY_NAMES:
            raise ValueError(f"Rejected strategy present in config: {strategy_name}")

        assert_model_allowed_for_strategy(strategy.model_family)
        assert_no_msvol_strategy_use(strategy.model_family)

        if strategy.requires_probabilities:
            if strategy.model_family not in {"gaussian_hmm", "markov_autoreg"}:
                raise ValueError(
                    f"{strategy_name} requires probabilities but model_family is "
                    f"{strategy.model_family}."
                )

            if strategy.probability_rule != "calm_minus_stress":
                raise ValueError(
                    f"{strategy_name} must use probability_rule='calm_minus_stress'."
                )

        if strategy.probability_rule is not None:
            if strategy.model_family not in {"gaussian_hmm", "markov_autoreg"}:
                raise ValueError(
                    f"Probability rule is not allowed for model_family "
                    f"{strategy.model_family} in strategy {strategy_name}."
                )

            if strategy.probability_rule == "probability_product":
                raise ValueError("probability_product is rejected for Phase 9.")

        if strategy.transition_exposure is not None:
            if strategy.name != "threshold_defensive":
                raise ValueError(
                    "Only threshold_defensive may define transition_exposure."
                )

            if strategy.transition_exposure != -0.25:
                raise ValueError("threshold_defensive transition_exposure must be -0.25.")

        if strategy.max_stress_probability is not None:
            if not (0.0 <= strategy.max_stress_probability <= 1.0):
                raise ValueError(
                    f"{strategy_name} max_stress_probability must lie in [0, 1]."
                )

            if strategy.max_stress_probability != 0.40:
                raise ValueError(
                    f"{strategy_name} max_stress_probability must equal 0.40."
                )

        if strategy.require_vrp_har_positive and not strategy.requires_har:
            raise ValueError(
                f"{strategy_name} requires positive VRP_HAR but requires_har=False."
            )

        if strategy.requires_har and not strategy.name.endswith("_carry"):
            raise ValueError(
                f"{strategy_name} requires HAR input but is not a carry-aware strategy."
            )


def _validate_paths(config: StrategyConfig) -> None:
    expected_markets = {"US", "INDIA"}

    if set(config.input_files) != expected_markets:
        raise ValueError(
            f"input_files must contain exactly {sorted(expected_markets)}, "
            f"got {sorted(config.input_files)}."
        )

    if set(config.output_files) != expected_markets:
        raise ValueError(
            f"output_files must contain exactly {sorted(expected_markets)}, "
            f"got {sorted(config.output_files)}."
        )

    required_input_keys = {"har", "threshold", "gaussian_hmm", "markov_autoreg"}

    for market, input_map in config.input_files.items():
        if set(input_map) != required_input_keys:
            raise ValueError(
                f"input_files.{market} must contain exactly "
                f"{sorted(required_input_keys)}, got {sorted(input_map)}."
            )

        for key, path_value in input_map.items():
            lowered_path = str(path_value).lower()
            if "msvol" in lowered_path or "msgarch" in lowered_path:
                raise ValueError(
                    f"Phase 9 must not read MSVOL/MSGARCH input files. "
                    f"Found {key}={path_value}."
                )

    for market, output_path in config.output_files.items():
        lowered_path = str(output_path).lower()
        if not lowered_path.endswith("_strategy_signals.parquet"):
            raise ValueError(
                f"Output path for {market} must end with "
                f"_strategy_signals.parquet, got {output_path}."
            )


def get_strategy_definitions(
    config: StrategyConfig,
    requested_strategy: str | None = None,
) -> dict[str, StrategyDefinition]:
    """
    Return active strategy definitions.

    If requested_strategy is provided, return only that strategy after checking
    that it belongs to the approved seven-strategy universe.
    """
    active = {
        name: definition
        for name, definition in config.strategies.items()
        if definition.active
    }

    if requested_strategy is None or requested_strategy == "all":
        return active

    requested = str(requested_strategy).strip()

    if requested not in APPROVED_STRATEGY_NAMES:
        raise ValueError(
            f"Requested strategy '{requested}' is not in the approved Phase 9 "
            f"strategy universe: {list(APPROVED_STRATEGY_NAMES)}."
        )

    if requested not in active:
        raise ValueError(f"Requested strategy '{requested}' is not active in config.")

    return {requested: active[requested]}


def get_market_input_paths(config: StrategyConfig, market: str) -> dict[str, Path]:
    market_key = str(market).strip().upper()

    if market_key not in config.input_files:
        raise ValueError(
            f"Unsupported market '{market}'. Available markets: "
            f"{sorted(config.input_files)}."
        )

    return {
        source_name: Path(source_path)
        for source_name, source_path in config.input_files[market_key].items()
    }


def get_market_output_path(config: StrategyConfig, market: str) -> Path:
    market_key = str(market).strip().upper()

    if market_key not in config.output_files:
        raise ValueError(
            f"Unsupported market '{market}'. Available markets: "
            f"{sorted(config.output_files)}."
        )

    return Path(config.output_files[market_key])


def strategy_config_hash(config: StrategyConfig) -> str:
    """
    Return a stable SHA-256 hash of the raw strategy config payload.
    """
    payload = json.dumps(config.raw, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()