import numpy as np
import pandas as pd
import pytest

from vrp.regimes.hmm_features import build_hmm_feature_panel
from vrp.regimes.hmm_registry import get_hmm_feature_output_name_map
from vrp.regimes.hmm_scaling import (
    assert_scaled_prefix_invariance,
    assert_scaler_fit_uses_train_only,
    chronological_train_test_indices,
    hash_numpy_array,
    scale_hmm_feature_panel,
    scaler_metadata_to_json_dict,
    transform_hmm_feature_panel_with_fitted_scaler,
)


def _make_panel_df(n=120, seed=7, extreme_tail=False):
    rng = np.random.default_rng(seed)

    vrp = rng.normal(0.02, 0.03, n)
    rv = rng.uniform(0.05, 0.35, n)
    iv = rng.uniform(0.10, 0.40, n)
    ret = rng.normal(0.0, 0.01, n)

    if extreme_tail:
        tail = min(10, n)
        vrp[-tail:] = 100.0
        rv[-tail:] = 200.0
        iv[-tail:] = 300.0
        ret[-tail:] = -50.0

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "vrp_har_gk": vrp,
            "rv_gk_22d_ann_lag1": rv,
            "iv_ann": iv,
            "log_return": ret,
            "har_forecast_available": True,
        }
    )


def _build_panel(df, feature_set="F3"):
    return build_hmm_feature_panel(
        df,
        market="US",
        feature_set=feature_set,
        min_eligible_observations=20,
        min_eligible_fraction=0.50,
    )


def test_chronological_train_test_indices_no_shuffle():
    train_idx, test_idx = chronological_train_test_indices(
        100,
        train_fraction=0.70,
        min_train_observations=10,
        min_test_observations=10,
    )

    np.testing.assert_array_equal(train_idx, np.arange(70))
    np.testing.assert_array_equal(test_idx, np.arange(70, 100))


def test_chronological_train_test_indices_rejects_small_train():
    with pytest.raises(ValueError, match="Insufficient train observations"):
        chronological_train_test_indices(
            20,
            train_fraction=0.50,
            min_train_observations=15,
            min_test_observations=5,
        )


def test_hmm_scaler_fit_uses_train_only():
    df = _make_panel_df(n=120, extreme_tail=True)
    panel = _build_panel(df)

    scaled = scale_hmm_feature_panel(
        panel,
        train_fraction=0.70,
        min_train_observations=20,
        min_test_observations=20,
    )

    assert_scaler_fit_uses_train_only(scaled)

    train_mean = scaled.X_raw[scaled.train_indices].mean(axis=0)
    full_mean = scaled.X_raw.mean(axis=0)

    assert scaled.scaler.mean_ is not None
    np.testing.assert_allclose(scaled.scaler.mean_, train_mean)
    assert not np.allclose(scaled.scaler.mean_, full_mean)


def test_scaled_feature_columns_are_written_with_explicit_names():
    df = _make_panel_df(n=120)
    panel = _build_panel(df)

    scaled = scale_hmm_feature_panel(
        panel,
        train_fraction=0.70,
        min_train_observations=20,
        min_test_observations=20,
    )

    expected_map = get_hmm_feature_output_name_map(panel.feature_cols)
    expected_cols = tuple(expected_map[col] for col in panel.feature_cols)

    assert scaled.scaled_feature_cols == expected_cols

    for col in expected_cols:
        assert col in scaled.scaled_panel.columns

    assert scaled.X_scaled.shape == (len(panel.eligible_panel), len(panel.feature_cols))


def test_scaler_metadata_written():
    df = _make_panel_df(n=120)
    panel = _build_panel(df)

    scaled = scale_hmm_feature_panel(
        panel,
        train_fraction=0.70,
        min_train_observations=20,
        min_test_observations=20,
    )

    meta = scaler_metadata_to_json_dict(scaled.metadata)

    required = {
        "scaler_fit_start_date",
        "scaler_fit_end_date",
        "scaler_feature_means",
        "scaler_feature_scales",
        "scaler_input_hash",
        "train_window_hash",
    }

    assert required.issubset(meta.keys())
    assert meta["scaler_fit_start_date"] == "2020-01-01"
    assert meta["scaler_fit_end_date"] == "2020-03-24"
    assert set(meta["scaler_feature_means"]) == set(panel.feature_cols)
    assert set(meta["scaler_feature_scales"]) == set(panel.feature_cols)
    assert isinstance(meta["scaler_input_hash"], str)
    assert len(meta["scaler_input_hash"]) == 64


def test_scaled_feature_prefix_invariance_when_reusing_same_scaler():
    prefix_df = _make_panel_df(n=100, seed=11)
    future_df = _make_panel_df(n=25, seed=99, extreme_tail=True)
    future_df["date"] = pd.date_range("2020-04-10", periods=25)

    full_df = pd.concat([prefix_df, future_df], ignore_index=True)

    prefix_panel = _build_panel(prefix_df)
    full_panel = _build_panel(full_df)

    prefix_scaled = scale_hmm_feature_panel(
        prefix_panel,
        train_fraction=0.70,
        min_train_observations=20,
        min_test_observations=20,
    )

    full_transformed = transform_hmm_feature_panel_with_fitted_scaler(
        full_panel,
        scaler=prefix_scaled.scaler,
        source_scaler_metadata=prefix_scaled.metadata.to_dict(),
    )

    assert_scaled_prefix_invariance(
        prefix_scaled.scaled_panel,
        full_transformed.scaled_panel,
        scaled_feature_cols=prefix_scaled.scaled_feature_cols,
        n_prefix_rows=len(prefix_scaled.scaled_panel),
    )


def test_hash_numpy_array_changes_when_values_change():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.1]])

    assert hash_numpy_array(a) != hash_numpy_array(b)


def test_hash_numpy_array_same_for_same_values():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert hash_numpy_array(a) == hash_numpy_array(b)