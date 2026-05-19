"""
Registry and safety rules for Phase 6 Gaussian HMM regime modelling.

This module defines:
- approved HMM feature sets
- feature-name aliases used in config files
- scaled output column names
- forbidden feature rules
- conditional feature availability rules
- expected Phase 6 report paths

No model fitting is implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final


HMM_MODEL_NAME: Final[str] = "gaussian_hmm_v1"
HMM_IMPLEMENTATION: Final[str] = "hmmlearn"


# ---------------------------------------------------------------------------
# Approved feature sets
# ---------------------------------------------------------------------------

HMM_FEATURE_SETS: Final[dict[str, list[str]]] = {
    "F1": ["vrp_har_gk"],
    "F2": ["vrp_har_gk", "rv_gk_22d_ann_lag1", "index_return"],
    "F3": ["vrp_har_gk", "rv_gk_22d_ann_lag1", "iv_ann", "index_return"],
    "F4": [
        "vrp_har_gk",
        "rv_gk_22d_ann_lag1",
        "iv_ann",
        "iv_ann_change_1d",
        "index_return",
    ],
}

HMM_PRIMARY_MODEL: Final[dict[str, object]] = {
    "feature_set": "F3",
    "n_states": 3,
    "covariance_type": "diag",
}

HMM_FALLBACK_MODEL: Final[dict[str, object]] = {
    "feature_set": "F3",
    "n_states": 2,
    "covariance_type": "diag",
}

HMM_ALLOWED_N_STATES: Final[tuple[int, ...]] = (2, 3)
HMM_ALLOWED_COVARIANCE_TYPES: Final[tuple[str, ...]] = ("diag", "full")


# ---------------------------------------------------------------------------
# Economic alias support
# ---------------------------------------------------------------------------

HMM_FEATURE_ALIASES: Final[dict[str, str]] = {
    "VRP_HAR": "vrp_har_gk",
    "RV": "rv_gk_22d_ann_lag1",
    "IV": "iv_ann",
    "IV_CHANGE": "iv_ann_change_1d",
    "INDEX_RETURN": "index_return",
}

HMM_CANONICAL_FEATURES: Final[tuple[str, ...]] = tuple(
    sorted({feature for features in HMM_FEATURE_SETS.values() for feature in features})
)


# ---------------------------------------------------------------------------
# Conditional features
# ---------------------------------------------------------------------------

HMM_CONDITIONAL_FEATURES: Final[dict[str, str]] = {
    "vrp_har_gk": "har_forecast_available",
    "har_rv_gk_22d_forecast_ann": "har_forecast_available",
}


# ---------------------------------------------------------------------------
# Forbidden feature rules
# ---------------------------------------------------------------------------

HMM_FORBIDDEN_FEATURE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "future",
    "forward",
    "expost",
    "label",
)

HMM_FORBIDDEN_FEATURE_EXACT: Final[tuple[str, ...]] = (
    "threshold_state",
    "threshold_state_name",
    "threshold_stress_score",
    "threshold_regime_available",
    "threshold_trigger_reason",
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
)

HMM_FORBIDDEN_FEATURE_PREFIXES: Final[tuple[str, ...]] = (
    "threshold_",
    "crisis_",
    "hmm_",
)

HMM_FORBIDDEN_FEATURE_CONTAINS: Final[tuple[str, ...]] = (
    "crisis",
)


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

HMM_SCALED_FEATURE_PREFIX: Final[str] = "hmm_feature_"
HMM_SCALED_FEATURE_SUFFIX: Final[str] = "_scaled"

HMM_SCALED_FEATURE_COLUMNS: Final[dict[str, str]] = {
    "vrp_har_gk": "hmm_feature_vrp_har_gk_scaled",
    "rv_gk_22d_ann_lag1": "hmm_feature_rv_gk_22d_ann_lag1_scaled",
    "iv_ann": "hmm_feature_iv_ann_scaled",
    "iv_ann_change_1d": "hmm_feature_iv_ann_change_1d_scaled",
    "index_return": "hmm_feature_index_return_scaled",
}

HMM_FILTERED_RAW_PROB_PREFIX: Final[str] = "hmm_filtered_prob_raw_state_"
HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX: Final[str] = (
    "hmm_diagnostic_smoothed_prob_raw_state_"
)

HMM_ECONOMIC_STATE_NAMES: Final[tuple[str, ...]] = ("calm", "transition", "stress")

HMM_FILTERED_ECONOMIC_PROB_COLUMNS: Final[dict[str, str]] = {
    "calm": "hmm_filtered_prob_calm",
    "transition": "hmm_filtered_prob_transition",
    "stress": "hmm_filtered_prob_stress",
}

HMM_TRANSITION_STATE_MODELLED_COLUMN: Final[str] = "hmm_transition_state_modelled"

HMM_SIGNAL_AVAILABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "hmm_signal_observation_date",
    "hmm_signal_available_after_close_date",
    "hmm_signal_trade_date",
    "hmm_state_for_next_session",
    "hmm_state_name_for_next_session",
    "hmm_filtered_prob_calm_for_next_session",
    "hmm_filtered_prob_transition_for_next_session",
    "hmm_filtered_prob_stress_for_next_session",
)


# ---------------------------------------------------------------------------
# Phase 6 report paths
# ---------------------------------------------------------------------------

PHASE_6_TABLES_DIR: Final[Path] = Path("reports/tables/phase_6")
PHASE_6_FIGURES_DIR: Final[Path] = Path("reports/figures/phase_6")

HMM_EXPECTED_TABLE_PATHS: Final[dict[str, Path]] = {
    "candidate_model_ranking": PHASE_6_TABLES_DIR / "hmm_candidate_model_ranking.csv",
    "feature_availability": PHASE_6_TABLES_DIR / "hmm_feature_availability.csv",
    "state_summary": PHASE_6_TABLES_DIR / "hmm_state_summary.csv",
    "transition_matrix": PHASE_6_TABLES_DIR / "hmm_transition_matrix.csv",
    "state_duration_summary": PHASE_6_TABLES_DIR / "hmm_state_duration_summary.csv",
    "state_by_year": PHASE_6_TABLES_DIR / "hmm_state_by_year.csv",
    "threshold_agreement": PHASE_6_TABLES_DIR / "hmm_threshold_agreement.csv",
    "crisis_hit_table": PHASE_6_TABLES_DIR / "hmm_crisis_hit_table.csv",
    "crisis_lead_lag_table": PHASE_6_TABLES_DIR / "hmm_crisis_lead_lag_table.csv",
    "forward_label_by_state": PHASE_6_TABLES_DIR / "hmm_forward_label_by_state.csv",
    "probability_audit": PHASE_6_TABLES_DIR / "hmm_probability_audit.csv",
    "no_lookahead_audit": PHASE_6_TABLES_DIR / "hmm_no_lookahead_audit.csv",
    "metadata": PHASE_6_TABLES_DIR / "hmm_metadata.json",
}


# ---------------------------------------------------------------------------
# Required report schemas
# ---------------------------------------------------------------------------

HMM_CANDIDATE_RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "market",
    "feature_set",
    "n_states",
    "covariance_type",
    "n_features",
    "n_observations",
    "n_train",
    "n_test",
    "train_loglik",
    "test_loglik",
    "train_loglik_per_obs",
    "test_loglik_per_obs",
    "aic",
    "bic",
    "converged",
    "n_iter",
    "min_state_occupancy_train",
    "min_state_occupancy_test",
    "economic_monotonicity_passed",
    "selected_primary",
    "rejection_reason",
)

HMM_FEATURE_AVAILABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "market",
    "feature_set",
    "required_feature",
    "source_column",
    "required_condition",
    "n_total_rows",
    "n_missing_feature",
    "n_condition_failed",
    "n_eligible_rows_after_feature",
    "first_available_date",
    "last_available_date",
    "blocked_reason",
)

HMM_PROBABILITY_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "market",
    "feature_set",
    "n_states",
    "covariance_type",
    "filtered_prob_columns",
    "smoothed_prob_columns",
    "backtest_probability_columns",
    "uses_custom_forward_filter",
    "uses_hmmlearn_predict_proba_for_backtest",
    "uses_smoothed_probabilities_for_backtest",
    "future_invariance_passed",
    "row_sum_min",
    "row_sum_max",
    "max_abs_prefix_difference",
    "passed",
)

HMM_METADATA_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "input_data_hash",
    "feature_panel_hash",
    "train_window_hash",
    "scaler_hash",
    "hmm_parameter_hash",
    "config_hash",
    "code_version_or_git_commit",
    "created_at_utc",
)


# ---------------------------------------------------------------------------
# Model failure reasons
# ---------------------------------------------------------------------------

HMM_ALLOWED_FAILURE_REASONS: Final[tuple[str, ...]] = (
    "insufficient_data",
    "state_collapse",
    "non_convergence",
    "unstable_covariance",
    "economically_uninterpretable_states",
    "poor_oos_likelihood",
    "feature_availability_too_low",
    "filtered_probability_invalid",
    "unknown_failure",
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def normalize_hmm_feature_name(feature: str) -> str:
    """
    Convert a feature alias or canonical feature name to the canonical name.

    Examples
    --------
    "VRP_HAR" -> "vrp_har_gk"
    "vrp_har_gk" -> "vrp_har_gk"
    """
    if not isinstance(feature, str):
        raise TypeError(f"Feature name must be a string. Got {type(feature)!r}.")

    stripped = feature.strip()
    if not stripped:
        raise ValueError("Feature name cannot be empty.")

    upper = stripped.upper()
    if upper in HMM_FEATURE_ALIASES:
        return HMM_FEATURE_ALIASES[upper]

    return stripped


def get_hmm_feature_set_names() -> list[str]:
    """Return approved HMM feature-set names."""
    return list(HMM_FEATURE_SETS.keys())


def assert_hmm_feature_set_is_valid(feature_set: str) -> None:
    """Raise if feature_set is not one of the approved Phase 6 feature sets."""
    if feature_set not in HMM_FEATURE_SETS:
        valid = ", ".join(sorted(HMM_FEATURE_SETS))
        raise ValueError(f"Unknown HMM feature_set={feature_set!r}. Valid values: {valid}.")


def get_hmm_feature_columns(feature_set: str | Iterable[str]) -> list[str]:
    """
    Return canonical feature columns.

    Parameters
    ----------
    feature_set:
        Either one of F1/F2/F3/F4 or an explicit iterable of feature names/aliases.
    """
    if isinstance(feature_set, str) and feature_set in HMM_FEATURE_SETS:
        feature_cols = list(HMM_FEATURE_SETS[feature_set])
    elif isinstance(feature_set, str):
        raise ValueError(
            f"Unknown HMM feature_set={feature_set!r}. "
            f"Use one of {sorted(HMM_FEATURE_SETS)} or pass an iterable of features."
        )
    else:
        feature_cols = [normalize_hmm_feature_name(feature) for feature in feature_set]

    assert_no_forbidden_hmm_features(feature_cols)
    return feature_cols


def get_hmm_conditional_features() -> dict[str, str]:
    """Return mapping: feature -> required availability/condition column."""
    return dict(HMM_CONDITIONAL_FEATURES)


def get_hmm_required_condition(feature_col: str) -> str | None:
    """Return the condition column required for a feature, if any."""
    canonical = normalize_hmm_feature_name(feature_col)
    return HMM_CONDITIONAL_FEATURES.get(canonical)


def is_hmm_conditional_feature(feature_col: str) -> bool:
    """Return True if the feature has a required availability condition."""
    return get_hmm_required_condition(feature_col) is not None


def get_hmm_scaled_feature_column(feature_col: str) -> str:
    """Return the explicit scaled output column name for a canonical feature."""
    canonical = normalize_hmm_feature_name(feature_col)
    if canonical not in HMM_SCALED_FEATURE_COLUMNS:
        raise ValueError(f"No scaled HMM output column registered for feature {feature_col!r}.")
    return HMM_SCALED_FEATURE_COLUMNS[canonical]


def get_hmm_scaled_feature_columns(feature_set: str | Iterable[str]) -> list[str]:
    """Return scaled feature column names for a feature set or explicit feature list."""
    return [get_hmm_scaled_feature_column(feature) for feature in get_hmm_feature_columns(feature_set)]


def get_hmm_feature_output_name_map(feature_set: str | Iterable[str]) -> dict[str, str]:
    """Return mapping: raw canonical feature -> scaled HMM output feature."""
    feature_cols = get_hmm_feature_columns(feature_set)
    return {feature: get_hmm_scaled_feature_column(feature) for feature in feature_cols}


def get_hmm_filtered_raw_probability_columns(n_states: int) -> list[str]:
    """Return raw-state filtered probability column names."""
    assert_hmm_n_states_is_valid(n_states)
    return [f"{HMM_FILTERED_RAW_PROB_PREFIX}{state}" for state in range(n_states)]


def get_hmm_diagnostic_smoothed_probability_columns(n_states: int) -> list[str]:
    """Return diagnostic-only smoothed raw-state probability column names."""
    assert_hmm_n_states_is_valid(n_states)
    return [f"{HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX}{state}" for state in range(n_states)]


def get_hmm_filtered_economic_probability_columns() -> list[str]:
    """Return economic-state filtered probability column names."""
    return [HMM_FILTERED_ECONOMIC_PROB_COLUMNS[state] for state in HMM_ECONOMIC_STATE_NAMES]


def assert_hmm_n_states_is_valid(n_states: int) -> None:
    """Raise if n_states is not approved for Phase 6."""
    if n_states not in HMM_ALLOWED_N_STATES:
        raise ValueError(
            f"Invalid HMM n_states={n_states!r}. "
            f"Allowed values: {HMM_ALLOWED_N_STATES}."
        )


def assert_hmm_covariance_type_is_valid(covariance_type: str) -> None:
    """Raise if covariance_type is not approved for Phase 6."""
    if covariance_type not in HMM_ALLOWED_COVARIANCE_TYPES:
        raise ValueError(
            f"Invalid HMM covariance_type={covariance_type!r}. "
            f"Allowed values: {HMM_ALLOWED_COVARIANCE_TYPES}."
        )


def assert_no_forbidden_hmm_features(feature_cols: Iterable[str]) -> None:
    """
    Raise if any feature violates Phase 6 no-lookahead / no-threshold / no-crisis rules.
    """
    for feature in feature_cols:
        canonical = normalize_hmm_feature_name(feature)
        lower = canonical.lower()

        if lower in HMM_FORBIDDEN_FEATURE_EXACT:
            raise ValueError(f"Forbidden HMM feature: {feature!r}.")

        for prefix in HMM_FORBIDDEN_FEATURE_PREFIXES:
            if lower.startswith(prefix):
                raise ValueError(
                    f"Forbidden HMM feature {feature!r}: starts with forbidden prefix {prefix!r}."
                )

        for substring in HMM_FORBIDDEN_FEATURE_SUBSTRINGS:
            if substring in lower:
                raise ValueError(
                    f"Forbidden HMM feature {feature!r}: contains forbidden substring {substring!r}."
                )

        for substring in HMM_FORBIDDEN_FEATURE_CONTAINS:
            if substring in lower:
                raise ValueError(
                    f"Forbidden HMM feature {feature!r}: contains forbidden token {substring!r}."
                )


def assert_hmm_features_are_point_in_time(feature_cols: Iterable[str]) -> None:
    """
    Validate that the proposed HMM feature columns are legal point-in-time features.

    This function intentionally does not inspect DataFrame values. It only checks names.
    Row-level availability and conditional checks are handled in hmm_features.py.
    """
    cols = [normalize_hmm_feature_name(feature) for feature in feature_cols]
    assert_no_forbidden_hmm_features(cols)


def validate_hmm_model_spec(
    feature_set: str,
    n_states: int,
    covariance_type: str,
) -> None:
    """Validate a Phase 6 candidate model specification."""
    assert_hmm_feature_set_is_valid(feature_set)
    assert_hmm_n_states_is_valid(n_states)
    assert_hmm_covariance_type_is_valid(covariance_type)
    assert_hmm_features_are_point_in_time(get_hmm_feature_columns(feature_set))


def iter_hmm_candidate_specs(
    feature_sets: Iterable[str] | None = None,
    n_states_values: Iterable[int] | None = None,
    covariance_types: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """
    Return validated candidate model specs.

    Defaults to the approved Phase 6 grid:
    F1-F4 x K in {2,3} x covariance in {diag,full}.
    """
    feature_sets = list(feature_sets) if feature_sets is not None else list(HMM_FEATURE_SETS)
    n_states_values = (
        list(n_states_values) if n_states_values is not None else list(HMM_ALLOWED_N_STATES)
    )
    covariance_types = (
        list(covariance_types)
        if covariance_types is not None
        else list(HMM_ALLOWED_COVARIANCE_TYPES)
    )

    specs: list[dict[str, object]] = []
    for feature_set in feature_sets:
        for n_states in n_states_values:
            for covariance_type in covariance_types:
                validate_hmm_model_spec(
                    feature_set=feature_set,
                    n_states=int(n_states),
                    covariance_type=str(covariance_type),
                )
                specs.append(
                    {
                        "feature_set": feature_set,
                        "n_states": int(n_states),
                        "covariance_type": str(covariance_type),
                    }
                )
    return specs


def is_primary_hmm_spec(
    feature_set: str,
    n_states: int,
    covariance_type: str,
    primary_model: Mapping[str, object] | None = None,
) -> bool:
    """Return True if the supplied spec matches the configured primary model."""
    primary = dict(HMM_PRIMARY_MODEL if primary_model is None else primary_model)
    return (
        feature_set == str(primary["feature_set"])
        and int(n_states) == int(str(primary["n_states"]))
        and covariance_type == str(primary["covariance_type"])
    )


def is_fallback_hmm_spec(
    feature_set: str,
    n_states: int,
    covariance_type: str,
    fallback_model: Mapping[str, object] | None = None,
) -> bool:
    """Return True if the supplied spec matches the configured fallback model."""
    fallback = dict(HMM_FALLBACK_MODEL if fallback_model is None else fallback_model)
    return (
        feature_set == str(fallback["feature_set"])
        and int(n_states) == int(str(fallback["n_states"]))
        and covariance_type == str(fallback["covariance_type"])
    )


def get_phase_6_tables_dir() -> Path:
    """Return Phase 6 table output directory."""
    return PHASE_6_TABLES_DIR


def get_phase_6_figures_dir() -> Path:
    """Return Phase 6 figure output directory."""
    return PHASE_6_FIGURES_DIR


def get_expected_hmm_table_paths() -> dict[str, Path]:
    """Return expected Phase 6 diagnostic table paths."""
    return dict(HMM_EXPECTED_TABLE_PATHS)


def make_model_specific_output_stem(
    market: str,
    feature_set: str,
    n_states: int,
    covariance_type: str,
) -> str:
    """
    Build a stable model-specific filename stem.

    Example
    -------
    us_hmm_F3_k3_diag
    """
    if not market:
        raise ValueError("market cannot be empty.")
    validate_hmm_model_spec(feature_set, n_states, covariance_type)
    return f"{market.lower()}_hmm_{feature_set}_k{int(n_states)}_{covariance_type}"


def get_model_specific_processed_path(
    market: str,
    feature_set: str,
    n_states: int,
    covariance_type: str,
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Return model-specific HMM regime panel path."""
    stem = make_model_specific_output_stem(market, feature_set, n_states, covariance_type)
    return Path(processed_dir) / f"{stem}.parquet"


def get_model_specific_pickle_path(
    market: str,
    feature_set: str,
    n_states: int,
    covariance_type: str,
    model_dir: str | Path = "models",
) -> Path:
    """Return model-specific HMM pickle path."""
    stem = make_model_specific_output_stem(market, feature_set, n_states, covariance_type)
    return Path(model_dir) / f"{stem}.pkl"


def get_primary_alias_processed_path(
    market: str,
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Return primary alias output path for selected HMM regime panel."""
    if not market:
        raise ValueError("market cannot be empty.")
    return Path(processed_dir) / f"{market.lower()}_hmm_regimes.parquet"


def get_primary_alias_model_path(
    market: str,
    model_dir: str | Path = "models",
) -> Path:
    """Return primary alias output path for selected HMM model pickle."""
    if not market:
        raise ValueError("market cannot be empty.")
    return Path(model_dir) / f"{market.lower()}_gaussian_hmm.pkl"


__all__ = [
    "HMM_MODEL_NAME",
    "HMM_IMPLEMENTATION",
    "HMM_FEATURE_SETS",
    "HMM_PRIMARY_MODEL",
    "HMM_FALLBACK_MODEL",
    "HMM_ALLOWED_N_STATES",
    "HMM_ALLOWED_COVARIANCE_TYPES",
    "HMM_FEATURE_ALIASES",
    "HMM_CANONICAL_FEATURES",
    "HMM_CONDITIONAL_FEATURES",
    "HMM_FORBIDDEN_FEATURE_SUBSTRINGS",
    "HMM_FORBIDDEN_FEATURE_EXACT",
    "HMM_FORBIDDEN_FEATURE_PREFIXES",
    "HMM_FORBIDDEN_FEATURE_CONTAINS",
    "HMM_SCALED_FEATURE_COLUMNS",
    "HMM_FILTERED_RAW_PROB_PREFIX",
    "HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX",
    "HMM_ECONOMIC_STATE_NAMES",
    "HMM_FILTERED_ECONOMIC_PROB_COLUMNS",
    "HMM_TRANSITION_STATE_MODELLED_COLUMN",
    "HMM_SIGNAL_AVAILABILITY_COLUMNS",
    "PHASE_6_TABLES_DIR",
    "PHASE_6_FIGURES_DIR",
    "HMM_EXPECTED_TABLE_PATHS",
    "HMM_CANDIDATE_RANKING_COLUMNS",
    "HMM_FEATURE_AVAILABILITY_COLUMNS",
    "HMM_PROBABILITY_AUDIT_COLUMNS",
    "HMM_METADATA_REQUIRED_FIELDS",
    "HMM_ALLOWED_FAILURE_REASONS",
    "normalize_hmm_feature_name",
    "get_hmm_feature_set_names",
    "assert_hmm_feature_set_is_valid",
    "get_hmm_feature_columns",
    "get_hmm_conditional_features",
    "get_hmm_required_condition",
    "is_hmm_conditional_feature",
    "get_hmm_scaled_feature_column",
    "get_hmm_scaled_feature_columns",
    "get_hmm_feature_output_name_map",
    "get_hmm_filtered_raw_probability_columns",
    "get_hmm_diagnostic_smoothed_probability_columns",
    "get_hmm_filtered_economic_probability_columns",
    "assert_hmm_n_states_is_valid",
    "assert_hmm_covariance_type_is_valid",
    "assert_no_forbidden_hmm_features",
    "assert_hmm_features_are_point_in_time",
    "validate_hmm_model_spec",
    "iter_hmm_candidate_specs",
    "is_primary_hmm_spec",
    "is_fallback_hmm_spec",
    "get_phase_6_tables_dir",
    "get_phase_6_figures_dir",
    "get_expected_hmm_table_paths",
    "make_model_specific_output_stem",
    "get_model_specific_processed_path",
    "get_model_specific_pickle_path",
    "get_primary_alias_processed_path",
    "get_primary_alias_model_path",
]