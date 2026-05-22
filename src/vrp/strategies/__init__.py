"""Strategy signal and exposure-rule modules."""
"""
Phase 9 strategy-signal construction package.

This package defines ex-ante short-vol exposure-intention rules only.
It does not compute realised returns, PnL, Sharpe, drawdown, transaction
costs, live orders, or option-chain execution.
"""

from vrp.strategies.strategy_config import (
    ExposureBounds,
    FrozenConstants,
    StrategyConfig,
    StrategyDefinition,
    TimingPolicy,
    get_market_input_paths,
    get_market_output_path,
    get_strategy_definitions,
    load_strategy_config,
    strategy_config_hash,
    validate_strategy_config,
)

from vrp.strategies.strategy_registry import (
    ALLOWED_CARRY_COLUMNS,
    ALLOWED_HMM_PROBABILITY_COLUMNS,
    ALLOWED_MAR_PROBABILITY_COLUMNS,
    ALLOWED_STRATEGY_MODELS,
    APPROVED_STRATEGY_NAMES,
    FORBIDDEN_STRATEGY_MODELS,
    REJECTED_STRATEGY_NAMES,
    STRATEGY_FORBIDDEN_EXACT_COLUMNS,
    STRATEGY_FORBIDDEN_FEATURE_SUBSTRINGS,
    assert_model_allowed_for_strategy,
    assert_no_msvol_strategy_use,
    assert_no_strategy_forbidden_columns,
    assert_strategy_inputs_are_point_in_time,
    get_allowed_probability_columns,
    get_forbidden_columns,
    validate_strategy_model_map,
    validate_strategy_names,
)

__all__ = [
    "ExposureBounds",
    "FrozenConstants",
    "StrategyConfig",
    "StrategyDefinition",
    "TimingPolicy",
    "get_market_input_paths",
    "get_market_output_path",
    "get_strategy_definitions",
    "load_strategy_config",
    "strategy_config_hash",
    "validate_strategy_config",
    "ALLOWED_CARRY_COLUMNS",
    "ALLOWED_HMM_PROBABILITY_COLUMNS",
    "ALLOWED_MAR_PROBABILITY_COLUMNS",
    "ALLOWED_STRATEGY_MODELS",
    "APPROVED_STRATEGY_NAMES",
    "FORBIDDEN_STRATEGY_MODELS",
    "REJECTED_STRATEGY_NAMES",
    "STRATEGY_FORBIDDEN_EXACT_COLUMNS",
    "STRATEGY_FORBIDDEN_FEATURE_SUBSTRINGS",
    "assert_model_allowed_for_strategy",
    "assert_no_msvol_strategy_use",
    "assert_no_strategy_forbidden_columns",
    "assert_strategy_inputs_are_point_in_time",
    "get_allowed_probability_columns",
    "get_forbidden_columns",
    "validate_strategy_model_map",
    "validate_strategy_names",
]