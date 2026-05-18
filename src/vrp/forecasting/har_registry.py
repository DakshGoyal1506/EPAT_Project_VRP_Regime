# src/vrp/forecasting/har_registry.py

"""
HAR-RV registry for Phase 4.

Purpose:
- Define the exact primary HAR predictor columns.
- Define HAR target and forecast-output columns.
- Prevent forward-looking / ex-post / label columns from entering HAR predictors.
- Keep Phase 4 forecast outputs separate from the Phase 3 VRP registry.

Primary HAR model:
    y_t = beta_0
          + beta_d * har_rv_d_lag1_ann_t
          + beta_w * har_rv_w_lag1_ann_t
          + beta_m * har_rv_m_lag1_ann_t
          + error_t

Primary target:
    rv_gk_22d_forward_ann_label

Primary forecast:
    har_rv_gk_22d_forecast_ann
"""

from __future__ import annotations


HAR_FEATURE_COLUMNS = [
    "har_rv_d_lag1_ann",
    "har_rv_w_lag1_ann",
    "har_rv_m_lag1_ann",
]


HAR_TARGET_COLUMNS = [
    "rv_gk_22d_forward_ann_label",
]


HAR_FORECAST_COLUMNS = [
    "har_rv_gk_22d_forecast_ann",
]


HAR_BASELINE_COLUMNS = [
    "naive_lagged_22d_rv_ann",
    "expanding_mean_forward_rv_baseline",
    "rolling_mean_forward_rv_baseline",
]


HAR_OUTPUT_FEATURE_COLUMNS = [
    "har_rv_gk_22d_forecast_ann",
    "vrp_har_gk",
]


HAR_METADATA_COLUMNS = [
    "target_col",
    "target_start_date",
    "target_end_date",
    "har_model_name",
    "har_train_start_date",
    "har_train_end_date",
    "har_n_train",
    "har_oos_flag",
    "har_forecast_available",
    "har_blocked_reason",
]


HAR_FORBIDDEN_FEATURE_SUBSTRINGS = [
    "future",
    "forward",
    "expost",
    "label",
]


HAR_EXPLICITLY_FORBIDDEN_PREDICTORS = [
    "iv_ann",
    "iv_close",
    "vrp_backward_gk",
    "vrp_backward_gk_positive",
    "vrp_forward_expost_gk_label",
    "rv_gk_22d_forward_ann_label",
    "har_rv_gk_22d_forecast_ann",
    "vrp_har_gk",
]


def get_har_feature_columns() -> list[str]:
    """
    Return the exact primary HAR predictor columns.
    """
    return list(HAR_FEATURE_COLUMNS)


def get_har_target_columns() -> list[str]:
    """
    Return HAR target columns.
    """
    return list(HAR_TARGET_COLUMNS)


def get_har_forecast_columns() -> list[str]:
    """
    Return HAR forecast columns.
    """
    return list(HAR_FORECAST_COLUMNS)


def get_har_baseline_columns() -> list[str]:
    """
    Return HAR baseline forecast columns.
    """
    return list(HAR_BASELINE_COLUMNS)


def get_har_output_feature_columns() -> list[str]:
    """
    Return Phase 4 outputs that may become later-phase features.

    These are not model inputs for the primary HAR regression.
    """
    return list(HAR_OUTPUT_FEATURE_COLUMNS)


def get_har_metadata_columns() -> list[str]:
    """
    Return HAR forecast-panel metadata columns.
    """
    return list(HAR_METADATA_COLUMNS)


def get_har_forbidden_feature_substrings() -> list[str]:
    """
    Return forbidden substrings for HAR predictor names.
    """
    return list(HAR_FORBIDDEN_FEATURE_SUBSTRINGS)


def get_har_explicitly_forbidden_predictors() -> list[str]:
    """
    Return explicit columns that cannot enter the primary HAR model.
    """
    return list(HAR_EXPLICITLY_FORBIDDEN_PREDICTORS)


def is_forbidden_har_feature_column(column: str) -> bool:
    """
    Return True if a column name is forbidden as a HAR predictor.

    A predictor is forbidden if:
    - it contains future / forward / expost / label, or
    - it is explicitly blocked, such as iv_ann or vrp_backward_gk.
    """
    column_str = str(column)
    column_lower = column_str.lower()

    if column_str in HAR_EXPLICITLY_FORBIDDEN_PREDICTORS:
        return True

    return any(
        forbidden in column_lower
        for forbidden in HAR_FORBIDDEN_FEATURE_SUBSTRINGS
    )


def assert_no_forbidden_har_features(feature_columns: list[str]) -> None:
    """
    Validate that HAR predictors contain no forbidden columns.
    """
    bad_columns = [
        col for col in feature_columns
        if is_forbidden_har_feature_column(col)
    ]

    if bad_columns:
        raise ValueError(
            "HAR feature list contains forbidden or forward-looking column(s): "
            f"{bad_columns}"
        )


def assert_primary_har_features(feature_columns: list[str]) -> None:
    """
    Validate that the primary HAR model uses exactly the three approved predictors.

    This intentionally rejects iv_ann, backward VRP, labels, forecasts, and any
    other extra columns.
    """
    supplied = list(feature_columns)
    expected = list(HAR_FEATURE_COLUMNS)

    if supplied != expected:
        raise ValueError(
            "Primary HAR predictors must equal exactly "
            f"{expected}. Got: {supplied}"
        )

    assert_no_forbidden_har_features(supplied)


def assert_har_registry_is_valid() -> None:
    """
    Validate HAR registry consistency.
    """
    feature_set = set(HAR_FEATURE_COLUMNS)
    target_set = set(HAR_TARGET_COLUMNS)
    forecast_set = set(HAR_FORECAST_COLUMNS)
    output_set = set(HAR_OUTPUT_FEATURE_COLUMNS)
    baseline_set = set(HAR_BASELINE_COLUMNS)

    feature_target_overlap = sorted(feature_set & target_set)
    feature_forecast_overlap = sorted(feature_set & forecast_set)
    feature_output_overlap = sorted(feature_set & output_set)
    target_forecast_overlap = sorted(target_set & forecast_set)
    baseline_feature_overlap = sorted(baseline_set & feature_set)

    if feature_target_overlap:
        raise ValueError(
            "HAR columns cannot be both features and targets: "
            f"{feature_target_overlap}"
        )

    if feature_forecast_overlap:
        raise ValueError(
            "HAR columns cannot be both features and forecasts: "
            f"{feature_forecast_overlap}"
        )

    if feature_output_overlap:
        raise ValueError(
            "HAR output columns cannot be primary HAR predictors: "
            f"{feature_output_overlap}"
        )

    if target_forecast_overlap:
        raise ValueError(
            "HAR columns cannot be both targets and forecasts: "
            f"{target_forecast_overlap}"
        )

    if baseline_feature_overlap:
        raise ValueError(
            "Baseline forecast columns cannot be primary HAR predictors: "
            f"{baseline_feature_overlap}"
        )

    assert_primary_har_features(HAR_FEATURE_COLUMNS)

    target_label_columns = [
        col for col in HAR_TARGET_COLUMNS
        if "label" not in col.lower()
    ]
    if target_label_columns:
        raise ValueError(
            "HAR target columns must be explicitly named as labels: "
            f"{target_label_columns}"
        )


def make_har_feature_metadata() -> dict[str, object]:
    """
    Return metadata describing HAR feature, target, forecast, and timing policy.
    """
    assert_har_registry_is_valid()

    return {
        "har_feature_columns": get_har_feature_columns(),
        "har_target_columns": get_har_target_columns(),
        "har_forecast_columns": get_har_forecast_columns(),
        "har_baseline_columns": get_har_baseline_columns(),
        "har_output_feature_columns": get_har_output_feature_columns(),
        "har_metadata_columns": get_har_metadata_columns(),
        "forbidden_feature_substrings": get_har_forbidden_feature_substrings(),
        "explicitly_forbidden_predictors": get_har_explicitly_forbidden_predictors(),
        "primary_predictor_policy": (
            "The primary HAR-RV model must use exactly daily, weekly, and "
            "monthly lagged realised variance features. It must not use IV, "
            "VRP, forecast, forward, ex-post, or label columns as predictors."
        ),
        "target_source_policy": (
            "The primary target is the existing Phase 3 column "
            "rv_gk_22d_forward_ann_label. Forward-target recomputation is "
            "validation-only."
        ),
        "training_label_availability_rule": (
            "For forecast date t, a training row s is allowed only when "
            "target_end_date_s < t."
        ),
    }