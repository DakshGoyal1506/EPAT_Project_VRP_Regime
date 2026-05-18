from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from vrp.regimes.regime_registry import (
    assert_no_forbidden_regime_features,
    assert_regime_features_are_point_in_time,
    get_allowed_diagnostic_labels,
    get_allowed_regime_construction_inputs,
)
from vrp.regimes.state_labeling import (
    CALM,
    STRESS,
    TRANSITION,
    map_state_id_to_name,
)


VALID_MARKETS = frozenset({"US", "INDIA"})
VALID_SCORE_METHODS = frozenset({"weighted_sum"})
VALID_PERCENTILE_INTERPOLATIONS = frozenset(
    {"linear", "lower", "higher", "midpoint", "nearest"}
)

REQUIRED_CONFIG_TOP_LEVEL_KEYS = [
    "model_name",
    "primary_input_files",
    "primary_output_files",
    "annualization_periods",
    "states",
    "regime_learning_policy",
    "diagnostic_window_policy",
    "threshold_policy",
    "iv_filter",
    "rv_filter",
    "drawdown_filter",
    "iv_slope_filter",
    "vrp_har_filter",
    "component_policy",
    "combined_filter",
    "sample_coverage_policy",
    "metadata_policy",
    "diagnostics",
    "crisis_windows",
]

REQUIRED_INPUT_COLUMNS = [
    "iv_ann",
    "iv_close",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "har_rv_gk_22d_forecast_ann",
    "vrp_har_gk",
    "har_forecast_available",
]


def load_threshold_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate the Phase 5 threshold-regime YAML config.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Threshold config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Threshold config must parse to a dictionary: {config_path}")

    validate_threshold_config(config)

    return config


def validate_threshold_config(config: Mapping[str, Any]) -> bool:
    """
    Validate the Phase 5 threshold-regime configuration.

    This function catches config-level mistakes before any regime labels are built:
    - missing required sections
    - wrong state mapping
    - use of old vix_slope naming
    - crisis-window leakage into model construction
    - forward-label leakage into model construction
    - unreachable score cutoffs
    - invalid percentile settings
    - cross-market threshold pooling
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping/dictionary.")

    _require_keys(config, REQUIRED_CONFIG_TOP_LEVEL_KEYS, context="config")
    _reject_legacy_vix_slope_naming(config)
    _validate_state_config(config["states"])
    _validate_primary_files(config)
    _validate_regime_learning_policy(config["regime_learning_policy"])
    _validate_diagnostic_window_policy(config["diagnostic_window_policy"])
    _validate_threshold_policy(config["threshold_policy"])
    _validate_component_filter_configs(config)
    _validate_component_policy(config["component_policy"], config["combined_filter"])
    _validate_combined_filter(config["combined_filter"])
    _validate_sample_coverage_policy(config["sample_coverage_policy"])
    _validate_metadata_policy(config["metadata_policy"])
    _validate_diagnostics_policy(config["diagnostics"])
    _validate_crisis_windows(config["crisis_windows"])

    return True


def validate_threshold_input_panel(panel: pd.DataFrame, market: str) -> bool:
    """
    Validate the input panel used to build Phase 5 threshold regimes.

    The panel may contain forward/ex-post/label columns because Phase 3 labels are
    preserved for later diagnostics. This function does not approve those columns
    for construction. It only validates the columns required for construction.
    """
    market = _normalise_market(market)

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    if panel.empty:
        raise ValueError(f"{market} threshold input panel is empty.")

    missing_cols = sorted(set(REQUIRED_INPUT_COLUMNS) - set(panel.columns))
    if missing_cols:
        raise ValueError(
            f"{market} threshold input panel is missing required column(s): "
            f"{missing_cols}"
        )

    construction_cols = [
        "iv_ann",
        "iv_close",
        "rv_gk_22d_ann_lag1",
        "vrp_backward_gk",
        "har_rv_gk_22d_forecast_ann",
        "vrp_har_gk",
        "har_forecast_available",
    ]
    assert_regime_features_are_point_in_time(construction_cols)

    if "market" in panel.columns:
        non_missing_market = panel["market"].dropna().astype(str).str.upper().unique()
        bad_markets = sorted(set(non_missing_market) - {market})
        if bad_markets:
            raise ValueError(
                f"{market} input panel contains rows from unexpected market(s): "
                f"{bad_markets}"
            )

    _validate_panel_date_axis(panel, market)
    _validate_no_duplicate_dates(panel, market)
    _validate_har_availability_column(panel, market)

    numeric_required_cols = [
        "iv_ann",
        "iv_close",
        "rv_gk_22d_ann_lag1",
        "vrp_backward_gk",
        "har_rv_gk_22d_forecast_ann",
        "vrp_har_gk",
    ]

    for col in numeric_required_cols:
        coerced = pd.to_numeric(panel[col], errors="coerce")
        bad_count = int(panel[col].notna().sum() - coerced.notna().sum())
        if bad_count > 0:
            raise ValueError(
                f"{market} column {col!r} contains {bad_count} non-numeric "
                "value(s) that cannot be coerced."
            )

    return True


def rolling_percentile_threshold(
    series: pd.Series,
    window: int,
    percentile: float,
    min_periods: int,
    strict_prior: bool = True,
    interpolation: str = "linear",
) -> pd.Series:
    """
    Compute a rolling percentile threshold.

    If strict_prior=True, the threshold at row t is computed using only rows before t
    by shifting the rolling percentile by one row.
    """
    numeric = _coerce_numeric_series(series, name="series")
    _validate_positive_int(window, "window")
    _validate_positive_int(min_periods, "min_periods")
    _validate_percentile(percentile, "percentile")
    _validate_interpolation(interpolation)

    if min_periods > window:
        raise ValueError(
            f"min_periods ({min_periods}) must be <= window ({window})."
        )

    threshold = numeric.rolling(
        window=window,
        min_periods=min_periods,
    ).quantile(percentile, interpolation=interpolation)

    if strict_prior:
        threshold = threshold.shift(1)

    return threshold.astype("float64")


def expanding_percentile_threshold(
    series: pd.Series,
    percentile: float,
    min_periods: int,
    strict_prior: bool = True,
    interpolation: str = "linear",
) -> pd.Series:
    """
    Compute an expanding percentile threshold.

    If strict_prior=True, the threshold at row t is computed using only rows before t
    by shifting the expanding percentile by one row.
    """
    numeric = _coerce_numeric_series(series, name="series")
    _validate_percentile(percentile, "percentile")
    _validate_positive_int(min_periods, "min_periods")
    _validate_interpolation(interpolation)

    threshold = numeric.expanding(
        min_periods=min_periods,
    ).quantile(percentile, interpolation=interpolation)

    if strict_prior:
        threshold = threshold.shift(1)

    return threshold.astype("float64")


def trailing_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
    strict_prior_mean_std: bool = True,
    ddof: int = 1,
) -> pd.Series:
    """
    Compute trailing z-score.

    With strict_prior_mean_std=True:

        z_t = (x_t - mean(x before t)) / std(x before t)

    This is the required behavior for HAR-VRP z-score classification.
    """
    numeric = _coerce_numeric_series(series, name="series")
    _validate_positive_int(window, "window")
    _validate_positive_int(min_periods, "min_periods")

    if min_periods > window:
        raise ValueError(
            f"min_periods ({min_periods}) must be <= window ({window})."
        )

    if ddof < 0:
        raise ValueError("ddof must be non-negative.")

    rolling = numeric.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=ddof)

    if strict_prior_mean_std:
        mean = mean.shift(1)
        std = std.shift(1)

    std = std.replace(0.0, np.nan)

    zscore = (numeric - mean) / std

    return zscore.astype("float64")


def compute_drawdown_from_price(
    price_series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute rolling-window drawdown from a price/equity curve.

    drawdown_t = price_t / rolling_max(price)_t - 1

    The rolling maximum uses only current and past values, never future values.
    """
    price = _coerce_numeric_series(price_series, name="price_series")
    _validate_positive_int(window, "window")

    if min_periods is None:
        min_periods = window

    _validate_positive_int(min_periods, "min_periods")

    if min_periods > window:
        raise ValueError(
            f"min_periods ({min_periods}) must be <= window ({window})."
        )

    if (price.dropna() <= 0).any():
        raise ValueError("price_series must contain only positive prices.")

    rolling_peak = price.rolling(window=window, min_periods=min_periods).max()
    drawdown = (price / rolling_peak) - 1.0

    return drawdown.astype("float64")


def compute_drawdown_from_returns(
    return_series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    return_type: str = "simple",
) -> pd.Series:
    """
    Reconstruct a pseudo-equity curve from returns and compute drawdown.

    return_type:
    - "simple": equity_t = cumulative product of (1 + r_t)
    - "log":    equity_t = exp(cumulative sum of r_t)

    Missing returns are treated as missing observations, not as zero returns.
    """
    returns = _coerce_numeric_series(return_series, name="return_series")
    _validate_positive_int(window, "window")

    if min_periods is None:
        min_periods = window

    _validate_positive_int(min_periods, "min_periods")

    return_type = str(return_type).strip().lower()
    if return_type not in {"simple", "log"}:
        raise ValueError("return_type must be either 'simple' or 'log'.")

    if return_type == "simple":
        if (returns.dropna() <= -1.0).any():
            raise ValueError(
                "simple returns must be greater than -1.0 to build equity curve."
            )
        equity = (1.0 + returns).cumprod()
    else:
        equity = np.exp(returns.cumsum())

    return compute_drawdown_from_price(
        price_series=equity,
        window=window,
        min_periods=min_periods,
    )


def classify_percentile_state(
    series: pd.Series,
    calm_threshold: pd.Series | float,
    stress_threshold: pd.Series | float,
    high_is_stress: bool = True,
) -> pd.Series:
    """
    Classify a numeric series using calm/stress thresholds.

    For high_is_stress=True:
        calm       if x_t <= calm_threshold_t
        stress     if x_t >= stress_threshold_t
        transition otherwise

    For high_is_stress=False:
        calm       if x_t >= calm_threshold_t
        stress     if x_t <= stress_threshold_t
        transition otherwise

    Missing series values or missing thresholds produce missing state.
    """
    x = _coerce_numeric_series(series, name="series")
    calm = _as_aligned_threshold_series(calm_threshold, x.index, "calm_threshold")
    stress = _as_aligned_threshold_series(stress_threshold, x.index, "stress_threshold")

    state = pd.Series(pd.NA, index=x.index, dtype="Int64")
    valid = x.notna() & calm.notna() & stress.notna()

    if not valid.any():
        return state

    if high_is_stress:
        calm_mask = valid & (x <= calm)
        stress_mask = valid & (x >= stress)
    else:
        calm_mask = valid & (x >= calm)
        stress_mask = valid & (x <= stress)

    transition_mask = valid & ~(calm_mask | stress_mask)

    state.loc[calm_mask] = CALM
    state.loc[transition_mask] = TRANSITION
    state.loc[stress_mask] = STRESS

    return state.astype("Int64")


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = sorted(set(keys) - set(mapping.keys()))
    if missing:
        raise ValueError(f"{context} is missing required key(s): {missing}")


def _reject_legacy_vix_slope_naming(config: Mapping[str, Any]) -> None:
    text_keys = _flatten_mapping_keys(config)

    legacy_keys = sorted(
        key for key in text_keys if "vix_slope" in key.lower()
    )

    if legacy_keys:
        raise ValueError(
            "Legacy vix_slope naming detected. Use iv_slope naming instead. "
            f"Problem key(s): {legacy_keys}"
        )


def _flatten_mapping_keys(mapping: Mapping[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []

    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)

        if isinstance(value, Mapping):
            keys.extend(_flatten_mapping_keys(value, prefix=full_key))

    return keys


def _validate_state_config(states: Mapping[str, Any]) -> None:
    expected = {
        "calm": CALM,
        "transition": TRANSITION,
        "stress": STRESS,
    }

    if dict(states) != expected:
        raise ValueError(
            f"states must be exactly {expected}. Got {dict(states)}."
        )


def _validate_primary_files(config: Mapping[str, Any]) -> None:
    for section in ["primary_input_files", "primary_output_files"]:
        value = config[section]

        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a mapping.")

        missing_markets = sorted(VALID_MARKETS - set(value.keys()))
        if missing_markets:
            raise ValueError(
                f"{section} must contain paths for markets {sorted(VALID_MARKETS)}. "
                f"Missing: {missing_markets}"
            )

        for market, path in value.items():
            _normalise_market(market)
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{section}.{market} must be a non-empty path string.")


def _validate_regime_learning_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "regime_type",
            "uses_manual_crisis_labels_for_training",
            "uses_crisis_windows_for_reporting_only",
            "uses_forward_labels_for_training",
            "uses_forward_labels_for_reporting_only",
            "no_manual_state_overrides",
        ],
        context="regime_learning_policy",
    )

    if policy["regime_type"] != "deterministic_unsupervised_threshold":
        raise ValueError(
            "regime_learning_policy.regime_type must be "
            "'deterministic_unsupervised_threshold'."
        )

    expected_bools = {
        "uses_manual_crisis_labels_for_training": False,
        "uses_crisis_windows_for_reporting_only": True,
        "uses_forward_labels_for_training": False,
        "uses_forward_labels_for_reporting_only": True,
        "no_manual_state_overrides": True,
    }

    for key, expected in expected_bools.items():
        if bool(policy[key]) is not expected:
            raise ValueError(f"regime_learning_policy.{key} must be {expected}.")


def _validate_diagnostic_window_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "use_crisis_windows_only_for_reporting",
            "skip_windows_outside_sample",
            "do_not_tune_thresholds_on_crisis_windows",
        ],
        context="diagnostic_window_policy",
    )

    expected_bools = {
        "use_crisis_windows_only_for_reporting": True,
        "skip_windows_outside_sample": True,
        "do_not_tune_thresholds_on_crisis_windows": True,
    }

    for key, expected in expected_bools.items():
        if bool(policy[key]) is not expected:
            raise ValueError(f"diagnostic_window_policy.{key} must be {expected}.")


def _validate_threshold_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "method",
            "use_strict_prior_thresholds",
            "min_history",
            "rolling_window",
            "percentile_interpolation",
            "prohibit_full_sample_percentiles",
            "prohibit_cross_market_threshold_pooling",
        ],
        context="threshold_policy",
    )

    if policy["method"] != "rolling_expanding":
        raise ValueError("threshold_policy.method must be 'rolling_expanding'.")

    if bool(policy["use_strict_prior_thresholds"]) is not True:
        raise ValueError("threshold_policy.use_strict_prior_thresholds must be true.")

    if bool(policy["prohibit_full_sample_percentiles"]) is not True:
        raise ValueError("threshold_policy.prohibit_full_sample_percentiles must be true.")

    if bool(policy["prohibit_cross_market_threshold_pooling"]) is not True:
        raise ValueError(
            "threshold_policy.prohibit_cross_market_threshold_pooling must be true."
        )

    _validate_positive_int(policy["min_history"], "threshold_policy.min_history")
    _validate_positive_int(policy["rolling_window"], "threshold_policy.rolling_window")

    if int(policy["min_history"]) > int(policy["rolling_window"]):
        raise ValueError("threshold_policy.min_history must be <= rolling_window.")

    _validate_interpolation(policy["percentile_interpolation"])


def _validate_component_filter_configs(config: Mapping[str, Any]) -> None:
    threshold_policy = config["threshold_policy"]

    iv_filter = config["iv_filter"]
    rv_filter = config["rv_filter"]
    vrp_har_filter = config["vrp_har_filter"]
    drawdown_filter = config["drawdown_filter"]
    iv_slope_filter = config["iv_slope_filter"]

    _validate_percentile_filter_config(iv_filter, "iv_filter")
    _validate_percentile_filter_config(rv_filter, "rv_filter")
    _validate_vrp_har_filter_config(vrp_har_filter)
    _validate_drawdown_filter_config(drawdown_filter)
    _validate_iv_slope_filter_config(iv_slope_filter)

    construction_features = [
        iv_filter["feature_col"],
        rv_filter["feature_col"],
        vrp_har_filter["feature_col"],
        iv_slope_filter["feature_col"],
        vrp_har_filter["availability_col"],
    ]

    assert_regime_features_are_point_in_time(construction_features)

    if iv_filter["feature_col"] != "iv_ann":
        raise ValueError("iv_filter.feature_col must be 'iv_ann'.")

    if rv_filter["feature_col"] != "rv_gk_22d_ann_lag1":
        raise ValueError("rv_filter.feature_col must be 'rv_gk_22d_ann_lag1'.")

    if vrp_har_filter["feature_col"] != "vrp_har_gk":
        raise ValueError("vrp_har_filter.feature_col must be 'vrp_har_gk'.")

    if iv_slope_filter["feature_col"] != "iv_ann":
        raise ValueError("iv_slope_filter.feature_col must be 'iv_ann'.")

    if int(vrp_har_filter["min_history"]) > int(vrp_har_filter["zscore_window"]):
        raise ValueError("vrp_har_filter.min_history must be <= zscore_window.")

    if int(threshold_policy["min_history"]) > int(threshold_policy["rolling_window"]):
        raise ValueError("threshold_policy.min_history must be <= rolling_window.")


def _validate_percentile_filter_config(filter_config: Mapping[str, Any], name: str) -> None:
    _require_keys(
        filter_config,
        [
            "feature_col",
            "calm_percentile",
            "stress_percentile",
            "high_is_stress",
            "required",
        ],
        context=name,
    )

    assert_no_forbidden_regime_features([filter_config["feature_col"]])

    _validate_percentile(filter_config["calm_percentile"], f"{name}.calm_percentile")
    _validate_percentile(filter_config["stress_percentile"], f"{name}.stress_percentile")

    calm = float(filter_config["calm_percentile"])
    stress = float(filter_config["stress_percentile"])

    if bool(filter_config["high_is_stress"]):
        if calm >= stress:
            raise ValueError(
                f"{name}.calm_percentile must be < stress_percentile when "
                "high_is_stress=true."
            )
    else:
        if calm <= stress:
            raise ValueError(
                f"{name}.calm_percentile must be > stress_percentile when "
                "high_is_stress=false."
            )


def _validate_vrp_har_filter_config(filter_config: Mapping[str, Any]) -> None:
    _require_keys(
        filter_config,
        [
            "feature_col",
            "require_har_forecast_available",
            "availability_col",
            "zscore_window",
            "min_history",
            "calm_zscore",
            "stress_zscore",
            "required",
        ],
        context="vrp_har_filter",
    )

    if bool(filter_config["require_har_forecast_available"]) is not True:
        raise ValueError(
            "vrp_har_filter.require_har_forecast_available must be true."
        )

    if filter_config["availability_col"] != "har_forecast_available":
        raise ValueError(
            "vrp_har_filter.availability_col must be 'har_forecast_available'."
        )

    _validate_positive_int(filter_config["zscore_window"], "vrp_har_filter.zscore_window")
    _validate_positive_int(filter_config["min_history"], "vrp_har_filter.min_history")

    if float(filter_config["calm_zscore"]) <= float(filter_config["stress_zscore"]):
        raise ValueError(
            "vrp_har_filter.calm_zscore must be greater than stress_zscore."
        )


def _validate_drawdown_filter_config(filter_config: Mapping[str, Any]) -> None:
    _require_keys(
        filter_config,
        [
            "price_col_candidates",
            "fallback_return_col_candidates",
            "lookback_window",
            "transition_drawdown",
            "stress_drawdown",
            "required",
        ],
        context="drawdown_filter",
    )

    _validate_positive_int(filter_config["lookback_window"], "drawdown_filter.lookback_window")

    if not isinstance(filter_config["price_col_candidates"], list):
        raise ValueError("drawdown_filter.price_col_candidates must be a list.")

    if not isinstance(filter_config["fallback_return_col_candidates"], list):
        raise ValueError(
            "drawdown_filter.fallback_return_col_candidates must be a list."
        )

    transition_drawdown = float(filter_config["transition_drawdown"])
    stress_drawdown = float(filter_config["stress_drawdown"])

    if stress_drawdown >= transition_drawdown:
        raise ValueError(
            "drawdown_filter.stress_drawdown must be more negative than "
            "transition_drawdown."
        )


def _validate_iv_slope_filter_config(filter_config: Mapping[str, Any]) -> None:
    _require_keys(
        filter_config,
        [
            "feature_col",
            "short_ma_window",
            "long_ma_window",
            "transition_slope_threshold",
            "stress_slope_threshold",
            "required",
        ],
        context="iv_slope_filter",
    )

    assert_no_forbidden_regime_features([filter_config["feature_col"]])

    _validate_positive_int(filter_config["short_ma_window"], "iv_slope_filter.short_ma_window")
    _validate_positive_int(filter_config["long_ma_window"], "iv_slope_filter.long_ma_window")

    if int(filter_config["short_ma_window"]) >= int(filter_config["long_ma_window"]):
        raise ValueError("iv_slope_filter.short_ma_window must be < long_ma_window.")

    if float(filter_config["transition_slope_threshold"]) >= float(
        filter_config["stress_slope_threshold"]
    ):
        raise ValueError(
            "iv_slope_filter.transition_slope_threshold must be < "
            "stress_slope_threshold."
        )


def _validate_component_policy(
    component_policy: Mapping[str, Any],
    combined_filter: Mapping[str, Any],
) -> None:
    _require_keys(
        component_policy,
        [
            "required_components",
            "optional_components",
            "do_not_fill_missing_component_states",
            "do_not_backfill_component_states",
            "do_not_forward_fill_component_states",
            "missing_required_blocks_final_regime",
            "score_optional_components_when_available_only",
            "score_missing_components_as",
        ],
        context="component_policy",
    )

    required_components = list(component_policy["required_components"])
    optional_components = list(component_policy["optional_components"])
    all_policy_components = set(required_components + optional_components)
    weighted_components = set(combined_filter["component_weights"].keys())

    expected_required = {
        "iv_percentile_state",
        "rv_percentile_state",
        "vrp_har_state",
    }

    if set(required_components) != expected_required:
        raise ValueError(
            "component_policy.required_components must be exactly "
            f"{sorted(expected_required)}."
        )

    expected_optional = {
        "drawdown_state",
        "iv_slope_state",
    }

    if set(optional_components) != expected_optional:
        raise ValueError(
            "component_policy.optional_components must be exactly "
            f"{sorted(expected_optional)}."
        )

    if all_policy_components != weighted_components:
        raise ValueError(
            "component_policy required+optional components must match "
            "combined_filter.component_weights keys."
        )

    expected_bools = {
        "do_not_fill_missing_component_states": True,
        "do_not_backfill_component_states": True,
        "do_not_forward_fill_component_states": True,
        "missing_required_blocks_final_regime": True,
        "score_optional_components_when_available_only": True,
    }

    for key, expected in expected_bools.items():
        if bool(component_policy[key]) is not expected:
            raise ValueError(f"component_policy.{key} must be {expected}.")

    if component_policy["score_missing_components_as"] is not None:
        raise ValueError("component_policy.score_missing_components_as must be null.")


def _validate_combined_filter(combined_filter: Mapping[str, Any]) -> None:
    _require_keys(
        combined_filter,
        [
            "score_method",
            "component_weights",
            "calm_score_cutoff",
            "stress_score_cutoff",
            "hard_stress_components",
            "calm_required_conditions",
            "require_score_cutoffs_reachable",
            "write_trigger_explanations",
        ],
        context="combined_filter",
    )

    score_method = combined_filter["score_method"]
    if score_method not in VALID_SCORE_METHODS:
        raise ValueError(
            f"combined_filter.score_method must be one of {sorted(VALID_SCORE_METHODS)}."
        )

    weights = combined_filter["component_weights"]
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("combined_filter.component_weights must be a non-empty mapping.")

    for component, weight in weights.items():
        if not isinstance(component, str) or not component.endswith("_state"):
            raise ValueError(
                f"Invalid component weight key {component!r}. Expected *_state."
            )
        if "vix_slope" in component:
            raise ValueError("Use iv_slope_state, not vix_slope_state.")
        if float(weight) < 0:
            raise ValueError(f"Weight for {component} must be non-negative.")

    calm_score_cutoff = float(combined_filter["calm_score_cutoff"])
    stress_score_cutoff = float(combined_filter["stress_score_cutoff"])

    if calm_score_cutoff < 0:
        raise ValueError("combined_filter.calm_score_cutoff must be >= 0.")

    if calm_score_cutoff >= stress_score_cutoff:
        raise ValueError(
            "combined_filter.calm_score_cutoff must be < stress_score_cutoff."
        )

    max_possible_score = sum(float(weight) * STRESS for weight in weights.values())

    if bool(combined_filter["require_score_cutoffs_reachable"]) is not True:
        raise ValueError("combined_filter.require_score_cutoffs_reachable must be true.")

    if stress_score_cutoff > max_possible_score:
        raise ValueError(
            "combined_filter.stress_score_cutoff is unreachable. "
            f"Cutoff={stress_score_cutoff}, max_possible_score={max_possible_score}."
        )

    hard_stress_components = set(combined_filter["hard_stress_components"])
    unknown_hard_components = sorted(hard_stress_components - set(weights.keys()))

    if unknown_hard_components:
        raise ValueError(
            "combined_filter.hard_stress_components contains unknown component(s): "
            f"{unknown_hard_components}"
        )

    calm_required_conditions = combined_filter["calm_required_conditions"]
    _require_keys(
        calm_required_conditions,
        [
            "iv_percentile_state",
            "rv_percentile_state_not",
            "drawdown_state",
            "vrp_har_state",
        ],
        context="combined_filter.calm_required_conditions",
    )

    if calm_required_conditions["iv_percentile_state"] != "calm":
        raise ValueError("Calm condition for iv_percentile_state must be 'calm'.")

    if calm_required_conditions["rv_percentile_state_not"] != "stress":
        raise ValueError("Calm condition for rv_percentile_state_not must be 'stress'.")

    if calm_required_conditions["drawdown_state"] != "calm_or_unavailable":
        raise ValueError(
            "Calm condition for drawdown_state must be 'calm_or_unavailable'."
        )

    if calm_required_conditions["vrp_har_state"] != "calm":
        raise ValueError("Calm condition for vrp_har_state must be 'calm'.")

    if bool(combined_filter["write_trigger_explanations"]) is not True:
        raise ValueError("combined_filter.write_trigger_explanations must be true.")


def _validate_sample_coverage_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "warn_if_available_fraction_after_warmup_below",
            "hard_fail_if_no_available_regimes",
        ],
        context="sample_coverage_policy",
    )

    value = float(policy["warn_if_available_fraction_after_warmup_below"])
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            "sample_coverage_policy.warn_if_available_fraction_after_warmup_below "
            "must be between 0 and 1."
        )

    if bool(policy["hard_fail_if_no_available_regimes"]) is not True:
        raise ValueError(
            "sample_coverage_policy.hard_fail_if_no_available_regimes must be true."
        )


def _validate_metadata_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "write_config_sha256",
            "write_input_file_metadata",
            "write_git_commit_if_available",
            "write_created_at_utc",
        ],
        context="metadata_policy",
    )

    for key in policy:
        if not isinstance(policy[key], bool):
            raise ValueError(f"metadata_policy.{key} must be boolean.")


def _validate_diagnostics_policy(policy: Mapping[str, Any]) -> None:
    _require_keys(
        policy,
        [
            "write_reports",
            "write_figures",
            "allow_forward_labels_only_after_regime_assignment",
            "duration_diagnostics",
            "annual_state_distribution",
            "crisis_hit_table",
            "crisis_lead_lag_table",
            "forward_label_by_state_table",
            "no_lookahead_audit",
        ],
        context="diagnostics",
    )

    for key in policy:
        if bool(policy[key]) is not True:
            raise ValueError(f"diagnostics.{key} must be true for Phase 5.")


def _validate_crisis_windows(crisis_windows: Mapping[str, Any]) -> None:
    if not isinstance(crisis_windows, Mapping):
        raise ValueError("crisis_windows must be a mapping.")

    missing_markets = sorted(VALID_MARKETS - set(crisis_windows.keys()))
    if missing_markets:
        raise ValueError(f"crisis_windows missing market(s): {missing_markets}")

    for market, windows in crisis_windows.items():
        market = _normalise_market(market)

        if not isinstance(windows, list):
            raise ValueError(f"crisis_windows.{market} must be a list.")

        for window in windows:
            if not isinstance(window, list) or len(window) != 3:
                raise ValueError(
                    f"Each crisis window for {market} must be "
                    "[start_date, end_date, crisis_name]. Got: {window}"
                )

            start_date, end_date, crisis_name = window

            start_ts = pd.to_datetime(start_date, errors="raise")
            end_ts = pd.to_datetime(end_date, errors="raise")

            if end_ts < start_ts:
                raise ValueError(
                    f"Crisis window end date is before start date for {market}: "
                    f"{window}"
                )

            if not isinstance(crisis_name, str) or not crisis_name.strip():
                raise ValueError(f"Crisis name must be a non-empty string: {window}")


def _validate_panel_date_axis(panel: pd.DataFrame, market: str) -> None:
    if "date" in panel.columns:
        parsed = pd.to_datetime(panel["date"], errors="coerce")
        bad_count = int(parsed.isna().sum())
        if bad_count > 0:
            raise ValueError(
                f"{market} input panel contains {bad_count} invalid date value(s)."
            )
        return

    if isinstance(panel.index, pd.DatetimeIndex):
        return

    raise ValueError(
        f"{market} input panel must contain a 'date' column or use a DatetimeIndex."
    )


def _validate_no_duplicate_dates(panel: pd.DataFrame, market: str) -> None:
    if "date" in panel.columns:
        dates = pd.to_datetime(panel["date"], errors="coerce")
    else:
        dates = pd.Series(panel.index, index=panel.index)

    duplicate_count = int(dates.duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"{market} input panel contains {duplicate_count} duplicate date(s)."
        )


def _validate_har_availability_column(panel: pd.DataFrame, market: str) -> None:
    col = "har_forecast_available"
    values = panel[col]

    if values.isna().any():
        missing_count = int(values.isna().sum())
        raise ValueError(
            f"{market} {col!r} contains {missing_count} missing value(s)."
        )

    allowed_values = {True, False, 0, 1, "True", "False", "true", "false", "0", "1"}
    observed = set(values.dropna().unique().tolist())

    invalid = sorted(
        [value for value in observed if value not in allowed_values],
        key=lambda x: str(x),
    )

    if invalid:
        raise ValueError(
            f"{market} {col!r} contains non-boolean-like value(s): {invalid}"
        )


def _coerce_numeric_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    numeric.name = series.name

    return numeric


def _as_aligned_threshold_series(
    threshold: pd.Series | float,
    index: pd.Index,
    name: str,
) -> pd.Series:
    if isinstance(threshold, pd.Series):
        aligned = pd.to_numeric(threshold.reindex(index), errors="coerce")
        return aligned.astype("float64")

    if np.isscalar(threshold):
        return pd.Series(float(threshold), index=index, dtype="float64")

    raise TypeError(f"{name} must be a pandas Series or scalar numeric value.")


def _validate_percentile(value: Any, name: str) -> None:
    numeric = float(value)

    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1. Got {value}.")


def _validate_interpolation(value: Any) -> None:
    interpolation = str(value).strip()

    if interpolation not in VALID_PERCENTILE_INTERPOLATIONS:
        raise ValueError(
            "percentile interpolation must be one of "
            f"{sorted(VALID_PERCENTILE_INTERPOLATIONS)}. Got {value!r}."
        )


def _validate_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not boolean.")

    try:
        numeric = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.") from exc

    if numeric <= 0:
        raise ValueError(f"{name} must be positive. Got {value!r}.")


def _normalise_market(market: str) -> str:
    if not isinstance(market, str):
        raise TypeError("market must be a string.")

    normalised = market.strip().upper()

    if normalised not in VALID_MARKETS:
        raise ValueError(
            f"Invalid market {market!r}. Expected one of {sorted(VALID_MARKETS)}."
        )

    return normalised

def classify_iv_percentile(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Classify implied-variance level using strict-prior rolling percentiles.

    Rules:
        calm       if iv_ann_t <= prior rolling calm percentile
        stress     if iv_ann_t >= prior rolling stress percentile
        transition otherwise

    Output columns:
        iv_percentile_state
        iv_percentile_state_name
        iv_percentile_available
        iv_percentile_blocked_reason
        iv_calm_threshold
        iv_stress_threshold
        threshold_history_start_date
        threshold_history_end_date
        threshold_n_history
    """
    _require_keys(config, ["threshold_policy", "iv_filter"], context="config")

    out = _prepare_component_panel(panel)
    filter_config = config["iv_filter"]
    threshold_policy = config["threshold_policy"]

    feature_col = filter_config["feature_col"]
    _require_panel_columns(out, [feature_col], context="IV percentile classification")

    window = int(threshold_policy["rolling_window"])
    min_history = int(threshold_policy["min_history"])
    interpolation = threshold_policy["percentile_interpolation"]

    iv = _coerce_numeric_series(out[feature_col], name=feature_col)

    calm_threshold = rolling_percentile_threshold(
        iv,
        window=window,
        percentile=float(filter_config["calm_percentile"]),
        min_periods=min_history,
        strict_prior=bool(threshold_policy["use_strict_prior_thresholds"]),
        interpolation=interpolation,
    )

    stress_threshold = rolling_percentile_threshold(
        iv,
        window=window,
        percentile=float(filter_config["stress_percentile"]),
        min_periods=min_history,
        strict_prior=bool(threshold_policy["use_strict_prior_thresholds"]),
        interpolation=interpolation,
    )

    state = classify_percentile_state(
        iv,
        calm_threshold=calm_threshold,
        stress_threshold=stress_threshold,
        high_is_stress=bool(filter_config["high_is_stress"]),
    )

    available = state.notna()
    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    blocked_reason.loc[iv.isna()] = "missing_iv_ann"
    blocked_reason.loc[
        iv.notna() & (calm_threshold.isna() | stress_threshold.isna())
    ] = "insufficient_prior_iv_history"

    out["iv_calm_threshold"] = calm_threshold
    out["iv_stress_threshold"] = stress_threshold
    out["iv_percentile_state"] = state
    out["iv_percentile_state_name"] = map_state_id_to_name(state)
    out["iv_percentile_available"] = available.astype(bool)
    out["iv_percentile_blocked_reason"] = blocked_reason

    _attach_threshold_history_metadata(
        out=out,
        source=iv,
        window=window,
        strict_prior=bool(threshold_policy["use_strict_prior_thresholds"]),
    )

    return out


def classify_rv_percentile(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Classify realised-volatility level using strict-prior rolling percentiles.

    Rules:
        calm       if rv_gk_22d_ann_lag1_t <= prior rolling calm percentile
        stress     if rv_gk_22d_ann_lag1_t >= prior rolling stress percentile
        transition otherwise

    Output columns:
        rv_percentile_state
        rv_percentile_state_name
        rv_percentile_available
        rv_percentile_blocked_reason
        rv_calm_threshold
        rv_stress_threshold
    """
    _require_keys(config, ["threshold_policy", "rv_filter"], context="config")

    out = _prepare_component_panel(panel)
    filter_config = config["rv_filter"]
    threshold_policy = config["threshold_policy"]

    feature_col = filter_config["feature_col"]
    _require_panel_columns(out, [feature_col], context="RV percentile classification")

    window = int(threshold_policy["rolling_window"])
    min_history = int(threshold_policy["min_history"])
    interpolation = threshold_policy["percentile_interpolation"]

    rv = _coerce_numeric_series(out[feature_col], name=feature_col)

    calm_threshold = rolling_percentile_threshold(
        rv,
        window=window,
        percentile=float(filter_config["calm_percentile"]),
        min_periods=min_history,
        strict_prior=bool(threshold_policy["use_strict_prior_thresholds"]),
        interpolation=interpolation,
    )

    stress_threshold = rolling_percentile_threshold(
        rv,
        window=window,
        percentile=float(filter_config["stress_percentile"]),
        min_periods=min_history,
        strict_prior=bool(threshold_policy["use_strict_prior_thresholds"]),
        interpolation=interpolation,
    )

    state = classify_percentile_state(
        rv,
        calm_threshold=calm_threshold,
        stress_threshold=stress_threshold,
        high_is_stress=bool(filter_config["high_is_stress"]),
    )

    available = state.notna()
    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    blocked_reason.loc[rv.isna()] = "missing_rv_gk_22d_ann_lag1"
    blocked_reason.loc[
        rv.notna() & (calm_threshold.isna() | stress_threshold.isna())
    ] = "insufficient_prior_rv_history"

    out["rv_calm_threshold"] = calm_threshold
    out["rv_stress_threshold"] = stress_threshold
    out["rv_percentile_state"] = state
    out["rv_percentile_state_name"] = map_state_id_to_name(state)
    out["rv_percentile_available"] = available.astype(bool)
    out["rv_percentile_blocked_reason"] = blocked_reason

    return out


def classify_drawdown(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Classify drawdown from price if possible, otherwise from returns.

    Preferred source:
        close / adj_close / underlying_close

    Fallback source:
        log_return / simple_return

    Rules:
        calm       if drawdown >= transition_drawdown
        transition if stress_drawdown < drawdown < transition_drawdown
        stress     if drawdown <= stress_drawdown

    Output columns:
        drawdown
        drawdown_state
        drawdown_state_name
        drawdown_available
        drawdown_blocked_reason
    """
    _require_keys(config, ["drawdown_filter"], context="config")

    out = _prepare_component_panel(panel)
    filter_config = config["drawdown_filter"]

    window = int(filter_config["lookback_window"])
    transition_drawdown = float(filter_config["transition_drawdown"])
    stress_drawdown = float(filter_config["stress_drawdown"])

    source_col = _first_existing_column(out, filter_config["price_col_candidates"])
    drawdown_source = None

    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")

    if source_col is not None:
        price = _coerce_numeric_series(out[source_col], name=source_col)

        try:
            drawdown = compute_drawdown_from_price(
                price_series=price,
                window=window,
                min_periods=window,
            )
            drawdown_source = f"price:{source_col}"
        except ValueError as exc:
            drawdown = pd.Series(np.nan, index=out.index, dtype="float64")
            blocked_reason.loc[:] = f"invalid_price_for_drawdown:{source_col}:{exc}"

    else:
        return_col = _first_existing_column(
            out,
            filter_config["fallback_return_col_candidates"],
        )

        if return_col is None:
            drawdown = pd.Series(np.nan, index=out.index, dtype="float64")
            blocked_reason.loc[:] = "missing_price_or_return_for_drawdown"
        else:
            returns = _coerce_numeric_series(out[return_col], name=return_col)
            return_type = "log" if return_col == "log_return" else "simple"

            try:
                drawdown = compute_drawdown_from_returns(
                    return_series=returns,
                    window=window,
                    min_periods=window,
                    return_type=return_type,
                )
                drawdown_source = f"{return_type}_return:{return_col}"
            except ValueError as exc:
                drawdown = pd.Series(np.nan, index=out.index, dtype="float64")
                blocked_reason.loc[:] = f"invalid_return_for_drawdown:{return_col}:{exc}"

    state = pd.Series(pd.NA, index=out.index, dtype="Int64")
    valid = drawdown.notna()

    calm_mask = valid & (drawdown >= transition_drawdown)
    stress_mask = valid & (drawdown <= stress_drawdown)
    transition_mask = valid & ~(calm_mask | stress_mask)

    state.loc[calm_mask] = CALM
    state.loc[transition_mask] = TRANSITION
    state.loc[stress_mask] = STRESS

    blocked_reason.loc[valid] = pd.NA
    blocked_reason.loc[drawdown.isna() & blocked_reason.isna()] = (
        "insufficient_drawdown_history"
    )

    out["drawdown"] = drawdown
    out["drawdown_source"] = drawdown_source
    out["drawdown_state"] = state
    out["drawdown_state_name"] = map_state_id_to_name(state)
    out["drawdown_available"] = state.notna().astype(bool)
    out["drawdown_blocked_reason"] = blocked_reason

    return out


def classify_iv_slope(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Classify implied-volatility slope using strict-prior moving averages.

    Compute:
        short_ma_t = mean(iv_ann before t over short window)
        long_ma_t  = mean(iv_ann before t over long window)
        iv_slope_t = short_ma_t / long_ma_t - 1

    Rules:
        stress     if iv_slope >= stress_slope_threshold
        transition if iv_slope > transition_slope_threshold
        calm       otherwise

    Output columns:
        iv_slope
        iv_slope_state
        iv_slope_state_name
        iv_slope_available
        iv_slope_blocked_reason
    """
    _require_keys(config, ["iv_slope_filter"], context="config")

    out = _prepare_component_panel(panel)
    filter_config = config["iv_slope_filter"]

    feature_col = filter_config["feature_col"]
    _require_panel_columns(out, [feature_col], context="IV slope classification")

    short_window = int(filter_config["short_ma_window"])
    long_window = int(filter_config["long_ma_window"])
    transition_threshold = float(filter_config["transition_slope_threshold"])
    stress_threshold = float(filter_config["stress_slope_threshold"])

    iv = _coerce_numeric_series(out[feature_col], name=feature_col)

    short_ma = iv.rolling(
        window=short_window,
        min_periods=short_window,
    ).mean().shift(1)

    long_ma = iv.rolling(
        window=long_window,
        min_periods=long_window,
    ).mean().shift(1)

    long_ma = long_ma.replace(0.0, np.nan)
    iv_slope = (short_ma / long_ma) - 1.0

    state = pd.Series(pd.NA, index=out.index, dtype="Int64")
    valid = iv.notna() & short_ma.notna() & long_ma.notna() & iv_slope.notna()

    stress_mask = valid & (iv_slope >= stress_threshold)
    transition_mask = valid & (iv_slope > transition_threshold) & ~stress_mask
    calm_mask = valid & ~(transition_mask | stress_mask)

    state.loc[calm_mask] = CALM
    state.loc[transition_mask] = TRANSITION
    state.loc[stress_mask] = STRESS

    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    blocked_reason.loc[iv.isna()] = "missing_iv_ann"
    blocked_reason.loc[
        iv.notna() & ~valid
    ] = "insufficient_prior_iv_slope_history"

    out["iv_short_ma"] = short_ma
    out["iv_long_ma"] = long_ma
    out["iv_slope"] = iv_slope
    out["iv_slope_state"] = state
    out["iv_slope_state_name"] = map_state_id_to_name(state)
    out["iv_slope_available"] = state.notna().astype(bool)
    out["iv_slope_blocked_reason"] = blocked_reason

    return out


def classify_vrp_zscore(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Classify HAR-based prospective VRP using a strict-prior trailing z-score.

    HAR availability rule:
        vrp_har_gk is usable only when har_forecast_available == True.

    Compute:
        z_t = (vrp_har_gk_t - prior rolling mean) / prior rolling std

    Rules:
        calm       if z_t >= calm_zscore
        stress     if z_t <= stress_zscore
        transition otherwise

    Output columns:
        vrp_har_zscore
        vrp_har_state
        vrp_har_state_name
        vrp_har_available
        vrp_har_blocked_reason
    """
    _require_keys(config, ["vrp_har_filter"], context="config")

    out = _prepare_component_panel(panel)
    filter_config = config["vrp_har_filter"]

    feature_col = filter_config["feature_col"]
    availability_col = filter_config["availability_col"]

    _require_panel_columns(
        out,
        [feature_col, availability_col],
        context="VRP-HAR z-score classification",
    )

    zscore_window = int(filter_config["zscore_window"])
    min_history = int(filter_config["min_history"])
    calm_zscore = float(filter_config["calm_zscore"])
    stress_zscore = float(filter_config["stress_zscore"])

    har_available = _coerce_bool_series(out[availability_col], availability_col)
    raw_vrp = _coerce_numeric_series(out[feature_col], name=feature_col)

    usable_vrp = raw_vrp.where(har_available, np.nan)

    zscore = trailing_zscore(
        usable_vrp,
        window=zscore_window,
        min_periods=min_history,
        strict_prior_mean_std=True,
        ddof=1,
    )

    state = pd.Series(pd.NA, index=out.index, dtype="Int64")
    valid = har_available & raw_vrp.notna() & zscore.notna()

    calm_mask = valid & (zscore >= calm_zscore)
    stress_mask = valid & (zscore <= stress_zscore)
    transition_mask = valid & ~(calm_mask | stress_mask)

    state.loc[calm_mask] = CALM
    state.loc[transition_mask] = TRANSITION
    state.loc[stress_mask] = STRESS

    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    blocked_reason.loc[~har_available] = "har_forecast_unavailable"
    blocked_reason.loc[har_available & raw_vrp.isna()] = "missing_vrp_har_gk"
    blocked_reason.loc[
        har_available & raw_vrp.notna() & zscore.isna()
    ] = "insufficient_prior_vrp_har_history"

    out["vrp_har_zscore"] = zscore
    out["vrp_har_state"] = state
    out["vrp_har_state_name"] = map_state_id_to_name(state)
    out["vrp_har_available"] = state.notna().astype(bool)
    out["vrp_har_blocked_reason"] = blocked_reason

    return out


def classify_all_threshold_components(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Apply all five Phase 5 component classifiers.

    This function does not build the final threshold_state. It only creates
    component states, component availability flags, blocked reasons, and diagnostics.
    """
    out = _prepare_component_panel(panel)

    out = classify_iv_percentile(out, config)
    out = classify_rv_percentile(out, config)
    out = classify_drawdown(out, config)
    out = classify_iv_slope(out, config)
    out = classify_vrp_zscore(out, config)

    return out


def _prepare_component_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Return a date-sorted defensive copy for component classification.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    out = panel.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out = out.sort_values("date").reset_index(drop=True)
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index().copy()
        out = out.reset_index().rename(columns={"index": "date"})
    else:
        raise ValueError(
            "panel must contain a 'date' column or use a DatetimeIndex."
        )

    return out


def _require_panel_columns(
    panel: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    missing = sorted(set(columns) - set(panel.columns))

    if missing:
        raise ValueError(f"{context} missing required column(s): {missing}")


def _first_existing_column(
    panel: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for col in candidates:
        if col in panel.columns:
            return col
    return None


def _coerce_bool_series(series: pd.Series, name: str) -> pd.Series:
    """
    Convert common boolean-like values to bool without allowing missing values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }

    converted = series.map(mapping)

    if converted.isna().any():
        bad_values = sorted(
            series[converted.isna()].dropna().unique().tolist(),
            key=lambda x: str(x),
        )
        raise ValueError(
            f"{name} contains missing or non-boolean-like value(s): {bad_values}"
        )

    return converted.astype(bool)


def _attach_threshold_history_metadata(
    out: pd.DataFrame,
    source: pd.Series,
    window: int,
    strict_prior: bool,
) -> None:
    """
    Attach generic threshold-history metadata.

    These fields are primarily tied to the rolling threshold policy. They record the
    approximate prior window available to a row, after excluding the current row
    when strict_prior=True.
    """
    _require_panel_columns(out, ["date"], context="threshold history metadata")

    dates = pd.to_datetime(out["date"], errors="raise")
    valid_obs = source.notna().astype(int)

    n_history = valid_obs.rolling(
        window=window,
        min_periods=1,
    ).sum()

    if strict_prior:
        n_history = n_history.shift(1)

    history_start = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    history_end = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    for i in range(len(out)):
        end_pos = i - 1 if strict_prior else i
        start_pos = max(0, end_pos - window + 1)

        if end_pos < 0:
            continue

        history_start.iloc[i] = dates.iloc[start_pos]
        history_end.iloc[i] = dates.iloc[end_pos]

    out["threshold_history_start_date"] = history_start
    out["threshold_history_end_date"] = history_end
    out["threshold_n_history"] = n_history.astype("Int64")
    
THRESHOLD_CORE_OUTPUT_COLUMNS = [
    "date",
    "market",
    "threshold_model_name",
    "iv_percentile_state",
    "iv_percentile_state_name",
    "rv_percentile_state",
    "rv_percentile_state_name",
    "drawdown_state",
    "drawdown_state_name",
    "iv_slope_state",
    "iv_slope_state_name",
    "vrp_har_state",
    "vrp_har_state_name",
    "iv_percentile_available",
    "rv_percentile_available",
    "drawdown_available",
    "iv_slope_available",
    "vrp_har_available",
    "iv_percentile_blocked_reason",
    "rv_percentile_blocked_reason",
    "drawdown_blocked_reason",
    "iv_slope_blocked_reason",
    "vrp_har_blocked_reason",
    "threshold_stress_score",
    "threshold_state",
    "threshold_state_name",
    "threshold_regime_available",
    "threshold_blocked_reason",
    "threshold_hard_stress_trigger",
    "threshold_score_trigger",
    "threshold_calm_conditions_pass",
    "threshold_primary_components_available",
    "threshold_trigger_reason",
    "iv_ann",
    "iv_close",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "har_rv_gk_22d_forecast_ann",
    "vrp_har_gk",
    "har_forecast_available",
    "iv_calm_threshold",
    "iv_stress_threshold",
    "rv_calm_threshold",
    "rv_stress_threshold",
    "vrp_har_zscore",
    "drawdown",
    "iv_slope",
    "threshold_history_start_date",
    "threshold_history_end_date",
    "threshold_n_history",
]

THRESHOLD_OPTIONAL_OUTPUT_COLUMNS = [
    "iv_short_ma",
    "iv_long_ma",
    "drawdown_source",
    "vrp_backward_gk_positive",
    "vrp_har_gk_positive",
    "feature_allowed",
    "har_blocked_reason",
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
    "log_return",
    "simple_return",
    "close",
    "adj_close",
    "underlying_close",
]


def combine_threshold_regimes(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Combine component threshold states into the final Phase 5 threshold regime.

    Required components:
        iv_percentile_state
        rv_percentile_state
        vrp_har_state

    Optional components:
        drawdown_state
        iv_slope_state

    Final-state rules:
        unavailable if any required component is missing
        stress      if any hard-stress component is stress
        stress      if threshold_stress_score >= stress_score_cutoff
        calm        if calm conditions pass and score <= calm_score_cutoff
        transition  otherwise

    This function does not use crisis windows, forward labels, ex-post labels,
    strategy returns, or HMM outputs.
    """
    _require_keys(
        config,
        ["model_name", "component_policy", "combined_filter"],
        context="config",
    )

    out = _prepare_component_panel(panel)

    component_policy = config["component_policy"]
    combined_filter = config["combined_filter"]

    required_components = list(component_policy["required_components"])
    optional_components = list(component_policy["optional_components"])
    component_weights = dict(combined_filter["component_weights"])
    hard_stress_components = list(combined_filter["hard_stress_components"])

    all_components = required_components + optional_components

    _require_panel_columns(
        out,
        all_components,
        context="combine_threshold_regimes",
    )

    _require_panel_columns(
        out,
        [_availability_col_for_component(component) for component in all_components],
        context="combine_threshold_regimes availability",
    )

    _require_panel_columns(
        out,
        [_blocked_reason_col_for_component(component) for component in all_components],
        context="combine_threshold_regimes blocked reasons",
    )

    _validate_no_forward_labels_in_combination_features(all_components)

    out["threshold_model_name"] = str(config["model_name"])

    if "market" not in out.columns:
        out["market"] = pd.NA

    stress_score = _compute_threshold_stress_score(
        out=out,
        component_weights=component_weights,
    )

    primary_available = _compute_primary_components_available(
        out=out,
        required_components=required_components,
    )

    missing_required_reason = _build_missing_required_reason(
        out=out,
        required_components=required_components,
    )

    hard_stress_trigger = _compute_hard_stress_trigger(
        out=out,
        hard_stress_components=hard_stress_components,
    )

    hard_stress_flag = hard_stress_trigger.notna()

    calm_conditions_pass = _compute_calm_conditions_pass(out)

    stress_score_cutoff = float(combined_filter["stress_score_cutoff"])
    calm_score_cutoff = float(combined_filter["calm_score_cutoff"])

    score_trigger = stress_score >= stress_score_cutoff

    final_state = pd.Series(pd.NA, index=out.index, dtype="Int64")
    regime_available = primary_available.copy()

    stress_mask = regime_available & (hard_stress_flag | score_trigger)
    calm_mask = (
        regime_available
        & ~stress_mask
        & calm_conditions_pass
        & (stress_score <= calm_score_cutoff)
    )
    transition_mask = regime_available & ~(stress_mask | calm_mask)

    final_state.loc[stress_mask] = STRESS
    final_state.loc[calm_mask] = CALM
    final_state.loc[transition_mask] = TRANSITION

    blocked_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    blocked_reason.loc[~regime_available] = missing_required_reason.loc[
        ~regime_available
    ]

    trigger_reason = pd.Series(pd.NA, index=out.index, dtype="object")
    trigger_reason.loc[~regime_available] = blocked_reason.loc[~regime_available]
    trigger_reason.loc[stress_mask & hard_stress_flag] = hard_stress_trigger.loc[
        stress_mask & hard_stress_flag
    ]
    trigger_reason.loc[stress_mask & ~hard_stress_flag & score_trigger] = (
        "score_ge_cutoff"
    )
    trigger_reason.loc[calm_mask] = "calm_conditions_pass"
    trigger_reason.loc[transition_mask] = "transition_default"

    out["threshold_stress_score"] = stress_score
    out["threshold_state"] = final_state
    out["threshold_state_name"] = map_state_id_to_name(final_state)
    out["threshold_regime_available"] = regime_available.astype(bool)
    out["threshold_blocked_reason"] = blocked_reason
    out["threshold_hard_stress_trigger"] = hard_stress_trigger
    out["threshold_score_trigger"] = score_trigger.astype(bool)
    out["threshold_calm_conditions_pass"] = calm_conditions_pass.astype(bool)
    out["threshold_primary_components_available"] = primary_available.astype(bool)
    out["threshold_trigger_reason"] = trigger_reason

    _validate_combined_regime_output(out)

    return _order_threshold_output_columns(out)


def build_threshold_regime_panel(
    market: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build the complete Phase 5 threshold-regime panel for one market.

    Reads:
        data/processed/us_vrp_har.parquet
        or
        data/processed/india_vrp_har.parquet

    Returns a DataFrame. Writing to disk is handled by scripts/train_regimes.py.
    """
    market = _normalise_market(market)
    validate_threshold_config(config)

    input_files = config["primary_input_files"]
    input_path = Path(input_files[market])

    if not input_path.exists():
        raise FileNotFoundError(
            f"{market} Phase 5 input file not found: {input_path}"
        )

    panel = pd.read_parquet(input_path)
    validate_threshold_input_panel(panel, market=market)

    panel = _prepare_component_panel(panel)
    panel["market"] = market

    component_panel = classify_all_threshold_components(panel, config)
    regime_panel = combine_threshold_regimes(component_panel, config)

    regime_panel["market"] = market

    _validate_no_crisis_window_columns_in_regime_panel(regime_panel)
    _validate_no_component_state_backfill(regime_panel)

    return regime_panel


def build_all_threshold_regimes(
    config: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Build Phase 5 threshold-regime panels for all configured markets.

    Returns:
        {
            "US": us_threshold_regime_panel,
            "INDIA": india_threshold_regime_panel,
        }
    """
    validate_threshold_config(config)

    panels: Dict[str, pd.DataFrame] = {}

    for market in ["US", "INDIA"]:
        panels[market] = build_threshold_regime_panel(market=market, config=config)

    return panels


def _compute_threshold_stress_score(
    out: pd.DataFrame,
    component_weights: Mapping[str, Any],
) -> pd.Series:
    """
    Compute weighted-sum stress score.

    Component state values:
        calm       = 0
        transition = 1
        stress     = 2

    Missing components contribute nothing to the score. They are not filled.
    Required missing components are handled separately by availability logic.
    """
    score = pd.Series(0.0, index=out.index, dtype="float64")

    for component, weight in component_weights.items():
        if component not in out.columns:
            raise ValueError(f"Missing component state column: {component}")

        state = pd.to_numeric(out[component], errors="coerce")
        valid = state.notna()

        score.loc[valid] = score.loc[valid] + (state.loc[valid] * float(weight))

    return score.astype("float64")


def _compute_primary_components_available(
    out: pd.DataFrame,
    required_components: Sequence[str],
) -> pd.Series:
    """
    A final regime is available only when all required component states exist.
    """
    available = pd.Series(True, index=out.index, dtype=bool)

    for component in required_components:
        if component not in out.columns:
            raise ValueError(f"Missing required component state column: {component}")

        component_state_available = out[component].notna()
        component_available_col = _availability_col_for_component(component)

        if component_available_col in out.columns:
            component_state_available = (
                component_state_available
                & _coerce_bool_series(out[component_available_col], component_available_col)
            )

        available = available & component_state_available

    return available.astype(bool)


def _build_missing_required_reason(
    out: pd.DataFrame,
    required_components: Sequence[str],
) -> pd.Series:
    """
    Create row-level blocked reason when final regime is unavailable.
    """
    reason = pd.Series(pd.NA, index=out.index, dtype="object")

    for component in required_components:
        component_missing = out[component].isna()
        availability_col = _availability_col_for_component(component)

        if availability_col in out.columns:
            component_missing = component_missing | ~_coerce_bool_series(
                out[availability_col],
                availability_col,
            )

        component_reason_col = _blocked_reason_col_for_component(component)

        if component_reason_col in out.columns:
            component_reason = out[component_reason_col].astype("object")
        else:
            component_reason = pd.Series(pd.NA, index=out.index, dtype="object")

        fill_mask = component_missing & reason.isna()
        reason.loc[fill_mask] = (
            "missing_required:"
            + component
            + ":"
            + component_reason.loc[fill_mask].fillna("component_unavailable").astype(str)
        )

    return reason


def _compute_hard_stress_trigger(
    out: pd.DataFrame,
    hard_stress_components: Sequence[str],
) -> pd.Series:
    """
    Return first hard-stress trigger per row.

    Example:
        hard_stress:iv_percentile_state
        hard_stress:drawdown_state
    """
    trigger = pd.Series(pd.NA, index=out.index, dtype="object")

    for component in hard_stress_components:
        if component not in out.columns:
            raise ValueError(f"Unknown hard-stress component: {component}")

        state = pd.to_numeric(out[component], errors="coerce")
        stress_mask = state == STRESS
        fill_mask = stress_mask & trigger.isna()

        trigger.loc[fill_mask] = f"hard_stress:{component}"

    return trigger


def _compute_calm_conditions_pass(out: pd.DataFrame) -> pd.Series:
    """
    Implement approved calm logic:

        iv_percentile_state == calm
        rv_percentile_state != stress
        drawdown_state == calm OR drawdown unavailable
        vrp_har_state == calm
    """
    _require_panel_columns(
        out,
        [
            "iv_percentile_state",
            "rv_percentile_state",
            "drawdown_state",
            "vrp_har_state",
        ],
        context="calm condition computation",
    )

    iv_state = pd.to_numeric(out["iv_percentile_state"], errors="coerce")
    rv_state = pd.to_numeric(out["rv_percentile_state"], errors="coerce")
    drawdown_state = pd.to_numeric(out["drawdown_state"], errors="coerce")
    vrp_state = pd.to_numeric(out["vrp_har_state"], errors="coerce")

    drawdown_available = (
        _coerce_bool_series(out["drawdown_available"], "drawdown_available")
        if "drawdown_available" in out.columns
        else drawdown_state.notna()
    )

    drawdown_calm_or_unavailable = (
        ((drawdown_available) & (drawdown_state == CALM))
        | (~drawdown_available)
    )

    calm_conditions = (
        (iv_state == CALM)
        & (rv_state.notna())
        & (rv_state != STRESS)
        & drawdown_calm_or_unavailable
        & (vrp_state == CALM)
    )

    return calm_conditions.fillna(False).astype(bool)


def _availability_col_for_component(component: str) -> str:
    if not component.endswith("_state"):
        raise ValueError(f"Component must end with '_state'. Got: {component}")

    return component[: -len("_state")] + "_available"


def _blocked_reason_col_for_component(component: str) -> str:
    if not component.endswith("_state"):
        raise ValueError(f"Component must end with '_state'. Got: {component}")

    return component[: -len("_state")] + "_blocked_reason"


def _validate_no_forward_labels_in_combination_features(
    component_columns: Sequence[str],
) -> None:
    forbidden_components = [
        col
        for col in component_columns
        if any(token in col.lower() for token in ["future", "forward", "expost", "label"])
    ]

    if forbidden_components:
        raise ValueError(
            "Forward/ex-post/label columns cannot be used as threshold components: "
            f"{forbidden_components}"
        )


def _validate_combined_regime_output(out: pd.DataFrame) -> None:
    """
    Validate final combined regime integrity.
    """
    _require_panel_columns(
        out,
        [
            "threshold_state",
            "threshold_regime_available",
            "threshold_blocked_reason",
            "threshold_trigger_reason",
            "threshold_stress_score",
        ],
        context="combined regime output validation",
    )

    available = _coerce_bool_series(
        out["threshold_regime_available"],
        "threshold_regime_available",
    )

    state = out["threshold_state"]

    if state.loc[available].isna().any():
        bad_count = int(state.loc[available].isna().sum())
        raise ValueError(
            f"{bad_count} available threshold regime row(s) have missing threshold_state."
        )

    if state.loc[~available].notna().any():
        bad_count = int(state.loc[~available].notna().sum())
        raise ValueError(
            f"{bad_count} unavailable threshold regime row(s) have non-missing "
            "threshold_state."
        )

    if out["threshold_blocked_reason"].loc[~available].isna().any():
        bad_count = int(out["threshold_blocked_reason"].loc[~available].isna().sum())
        raise ValueError(
            f"{bad_count} unavailable threshold regime row(s) lack blocked reason."
        )

    if out["threshold_trigger_reason"].isna().any():
        bad_count = int(out["threshold_trigger_reason"].isna().sum())
        raise ValueError(
            f"{bad_count} threshold regime row(s) lack trigger reason."
        )

    numeric_state = pd.to_numeric(state.dropna(), errors="coerce")
    invalid_state_values = sorted(
        set(numeric_state.dropna().astype(int).tolist()) - {CALM, TRANSITION, STRESS}
    )

    if invalid_state_values:
        raise ValueError(
            "Invalid threshold_state value(s): "
            f"{invalid_state_values}. Expected {CALM}, {TRANSITION}, {STRESS}."
        )


def _order_threshold_output_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Put important Phase 5 output columns first while preserving extra columns.
    """
    ordered_columns = []

    for col in THRESHOLD_CORE_OUTPUT_COLUMNS + THRESHOLD_OPTIONAL_OUTPUT_COLUMNS:
        if col in out.columns and col not in ordered_columns:
            ordered_columns.append(col)

    remaining_columns = [col for col in out.columns if col not in ordered_columns]

    return out[ordered_columns + remaining_columns].copy()


def _validate_no_crisis_window_columns_in_regime_panel(out: pd.DataFrame) -> None:
    """
    Crisis windows must not be injected into the row-level regime panel.

    They are only consumed later by diagnostics functions.
    """
    forbidden_tokens = [
        "crisis_name",
        "crisis_window",
        "known_crisis",
        "manual_crisis",
        "event_window",
    ]

    bad_cols = [
        col
        for col in out.columns
        if any(token in col.lower() for token in forbidden_tokens)
    ]

    if bad_cols:
        raise ValueError(
            "Crisis-window columns detected in regime panel construction output. "
            "Crisis windows are reporting-only. Bad columns: "
            f"{bad_cols}"
        )


def _validate_no_component_state_backfill(out: pd.DataFrame) -> None:
    """
    Guard against accidental component-state fill/backfill behavior.

    If a component is unavailable, its state must be missing.
    """
    component_cols = [
        "iv_percentile_state",
        "rv_percentile_state",
        "drawdown_state",
        "iv_slope_state",
        "vrp_har_state",
    ]

    for state_col in component_cols:
        if state_col not in out.columns:
            continue

        available_col = _availability_col_for_component(state_col)

        if available_col not in out.columns:
            continue

        available = _coerce_bool_series(out[available_col], available_col)
        invalid_mask = (~available) & out[state_col].notna()

        if invalid_mask.any():
            bad_count = int(invalid_mask.sum())
            raise ValueError(
                f"{bad_count} row(s) have {state_col} populated while "
                f"{available_col}=False. This suggests fill/backfill leakage."
            )