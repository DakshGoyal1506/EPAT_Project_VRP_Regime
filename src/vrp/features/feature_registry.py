# src/vrp/features/feature_registry.py

"""
Feature registry for Phase 3 VRP construction.

Purpose:
- Define which VRP columns are allowed as model/trading features.
- Define which VRP columns are labels only.
- Prevent forward-looking / ex-post / label columns from entering feature sets.

This file acts as an anti-lookahead firewall for later phases.
"""

from __future__ import annotations


VRP_FEATURE_COLUMNS = [
    "iv_ann",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "vrp_backward_gk_positive",
]


VRP_LABEL_COLUMNS = [
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
]


VRP_ROBUSTNESS_COLUMNS = [
    "rv_cc_22d_ann_lag1",
    "vrp_backward_cc",
    "vrp_backward_cc_positive",
    "rv_parkinson_22d_ann_lag1",
    "vrp_backward_parkinson",
    "vrp_backward_parkinson_positive",
    "rv_rs_22d_ann_lag1",
    "vrp_backward_rs",
    "vrp_backward_rs_positive",
    "rv_yz_22d_ann_lag1",
    "vrp_backward_yz",
    "vrp_backward_yz_positive",
]


FORBIDDEN_FEATURE_SUBSTRINGS = [
    "future",
    "forward",
    "expost",
    "label",
]


def get_vrp_feature_columns() -> list[str]:
    """
    Return allowed VRP feature columns.
    """
    return list(VRP_FEATURE_COLUMNS)


def get_vrp_label_columns() -> list[str]:
    """
    Return VRP label-only columns.
    """
    return list(VRP_LABEL_COLUMNS)


def get_vrp_robustness_columns() -> list[str]:
    """
    Return VRP robustness-only columns.
    """
    return list(VRP_ROBUSTNESS_COLUMNS)


def get_forbidden_feature_substrings() -> list[str]:
    """
    Return forbidden substrings for feature names.
    """
    return list(FORBIDDEN_FEATURE_SUBSTRINGS)


def is_forbidden_feature_column(column: str) -> bool:
    """
    Return True if a column name contains any forbidden feature substring.

    Examples of forbidden names:
    - future_rv
    - rv_forward
    - vrp_expost
    - target_label
    """
    column_lower = str(column).lower()

    return any(
        forbidden in column_lower
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS
    )


def assert_no_lookahead_feature_columns(feature_columns: list[str]) -> None:
    """
    Validate that feature columns do not contain forward-looking labels.

    Raises
    ------
    ValueError
        If any feature column contains a forbidden substring.
    """
    bad_columns = [
        col for col in feature_columns
        if is_forbidden_feature_column(col)
    ]

    if bad_columns:
        raise ValueError(
            "Feature list contains forward-looking or label column(s): "
            f"{bad_columns}"
        )


def assert_registry_is_valid() -> None:
    """
    Validate internal feature/label registry consistency.
    """
    overlap = sorted(set(VRP_FEATURE_COLUMNS) & set(VRP_LABEL_COLUMNS))
    robustness_label_overlap = sorted(set(VRP_ROBUSTNESS_COLUMNS) & set(VRP_LABEL_COLUMNS))

    if overlap:
        raise ValueError(
            f"Columns cannot be both features and labels: {overlap}"
        )

    if robustness_label_overlap:
        raise ValueError(
            f"Columns cannot be both robustness diagnostics and labels: {robustness_label_overlap}"
        )

    assert_no_lookahead_feature_columns(VRP_FEATURE_COLUMNS)


def make_vrp_feature_metadata() -> dict[str, object]:
    """
    Return metadata dictionary describing VRP feature/label separation.
    """
    assert_registry_is_valid()

    return {
        "allowed_feature_columns": get_vrp_feature_columns(),
        "robustness_columns": get_vrp_robustness_columns(),
        "label_columns": get_vrp_label_columns(),
        "forbidden_feature_substrings": get_forbidden_feature_substrings(),
        "lookahead_policy": (
            "Columns containing future, forward, expost, or label are never "
            "allowed as live model/trading features."
        ),
        "robustness_policy": (
            "Robustness columns are diagnostic only in Phase 3 and are not part of the primary live feature registry."
        ),
    }