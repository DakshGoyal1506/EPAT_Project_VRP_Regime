# tests/test_no_lookahead.py

from __future__ import annotations

import pandas as pd
import pytest

from vrp.features.feature_registry import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    VRP_FEATURE_COLUMNS,
    VRP_LABEL_COLUMNS,
    VRP_ROBUSTNESS_COLUMNS,
    assert_no_lookahead_feature_columns,
    assert_registry_is_valid,
    get_vrp_feature_columns,
    get_vrp_label_columns,
    get_vrp_robustness_columns,
    is_forbidden_feature_column,
)
from vrp.features.vrp import flag_feature_columns_vs_label_columns


def test_registry_has_no_feature_label_overlap() -> None:
    assert_registry_is_valid()

    overlap = set(VRP_FEATURE_COLUMNS) & set(VRP_LABEL_COLUMNS)
    assert overlap == set()


def test_registry_has_no_feature_robustness_overlap() -> None:
    overlap = set(VRP_FEATURE_COLUMNS) & set(VRP_ROBUSTNESS_COLUMNS)
    assert overlap == set()


def test_feature_columns_do_not_contain_forbidden_substrings() -> None:
    for col in VRP_FEATURE_COLUMNS:
        col_lower = col.lower()
        for token in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert token not in col_lower


def test_label_columns_are_not_features() -> None:
    features = get_vrp_feature_columns()
    labels = get_vrp_label_columns()

    for label in labels:
        assert label not in features


def test_robustness_columns_are_not_primary_features() -> None:
    features = get_vrp_feature_columns()
    labels = get_vrp_label_columns()
    robustness = get_vrp_robustness_columns()

    assert robustness == VRP_ROBUSTNESS_COLUMNS

    for column in robustness:
        assert column not in features
        assert column not in labels


def test_robustness_columns_have_no_forward_or_label_names() -> None:
    for column in VRP_ROBUSTNESS_COLUMNS:
        column_lower = column.lower()
        assert "future" not in column_lower
        assert "forward" not in column_lower
        assert "expost" not in column_lower
        assert "label" not in column_lower


def test_label_columns_have_label_naming() -> None:
    labels = get_vrp_label_columns()

    assert "rv_gk_22d_forward_ann_label" in labels
    assert "vrp_forward_expost_gk_label" in labels

    for label in labels:
        label_lower = label.lower()
        assert "label" in label_lower
        assert ("forward" in label_lower) or ("expost" in label_lower)


def test_is_forbidden_feature_column_detects_bad_names() -> None:
    assert is_forbidden_feature_column("future_rv")
    assert is_forbidden_feature_column("rv_forward")
    assert is_forbidden_feature_column("vrp_expost")
    assert is_forbidden_feature_column("target_label")

    assert not is_forbidden_feature_column("iv_ann")
    assert not is_forbidden_feature_column("rv_gk_22d_ann_lag1")
    assert not is_forbidden_feature_column("vrp_backward_gk")


def test_assert_no_lookahead_feature_columns_rejects_bad_names() -> None:
    with pytest.raises(ValueError, match="forward-looking"):
        assert_no_lookahead_feature_columns(
            [
                "iv_ann",
                "rv_gk_22d_ann_lag1",
                "vrp_forward_expost_gk_label",
            ]
        )


def test_flag_feature_columns_vs_label_columns_separates_features_and_labels() -> None:
    df = pd.DataFrame(
        {
            "iv_ann": [0.04, 0.05, None],
            "rv_gk_22d_ann_lag1": [0.03, 0.04, 0.05],
            "vrp_backward_gk": [0.01, 0.01, None],
            "vrp_backward_gk_positive": [True, True, None],
            "rv_gk_22d_forward_ann_label": [0.05, None, None],
            "vrp_forward_expost_gk_label": [-0.01, None, None],
        }
    )

    out = flag_feature_columns_vs_label_columns(df)

    assert "feature_allowed" in out.columns
    assert bool(out.loc[0, "feature_allowed"])
    assert bool(out.loc[1, "feature_allowed"])
    assert not bool(out.loc[2, "feature_allowed"])


def test_flag_feature_columns_vs_label_columns_requires_labels_present() -> None:
    df = pd.DataFrame(
        {
            "iv_ann": [0.04],
            "rv_gk_22d_ann_lag1": [0.03],
            "vrp_backward_gk": [0.01],
            "vrp_backward_gk_positive": [True],
        }
    )

    with pytest.raises(ValueError, match="label"):
        flag_feature_columns_vs_label_columns(df)


def test_flag_feature_columns_vs_label_columns_requires_features_present() -> None:
    df = pd.DataFrame(
        {
            "rv_gk_22d_forward_ann_label": [0.05],
            "vrp_forward_expost_gk_label": [-0.01],
        }
    )

    with pytest.raises(ValueError, match="feature"):
        flag_feature_columns_vs_label_columns(df)