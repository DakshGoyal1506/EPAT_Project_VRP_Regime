# tests/test_forecast_evaluation.py

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vrp.forecasting.forecast_evaluation import (
    METRIC_COLUMNS,
    build_forecast_accuracy_table,
    directional_accuracy_against_baseline,
    evaluate_forecasts,
    forecast_bias,
    forecast_correlation,
    mae,
    mse,
    qlike,
    rmse,
)


def test_mse_rmse_mae_basic_values() -> None:
    y_true = [1.0, 2.0, 4.0]
    y_pred = [1.0, 3.0, 2.0]

    expected_mse = (0.0**2 + 1.0**2 + 2.0**2) / 3.0
    expected_rmse = math.sqrt(expected_mse)
    expected_mae = (0.0 + 1.0 + 2.0) / 3.0

    assert mse(y_true, y_pred) == pytest.approx(expected_mse)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)
    assert mae(y_true, y_pred) == pytest.approx(expected_mae)


def test_metrics_filter_nan_and_non_finite_pairs() -> None:
    y_true = pd.Series([1.0, np.nan, 3.0, np.inf, 5.0])
    y_pred = pd.Series([2.0, 4.0, np.nan, 8.0, -np.inf])

    # Only first pair is valid.
    assert mse(y_true, y_pred) == pytest.approx(1.0)
    assert rmse(y_true, y_pred) == pytest.approx(1.0)
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert forecast_bias(y_true, y_pred) == pytest.approx(1.0)


def test_metrics_return_nan_when_no_valid_pairs() -> None:
    y_true = [np.nan, np.inf]
    y_pred = [1.0, 2.0]

    assert math.isnan(mse(y_true, y_pred))
    assert math.isnan(rmse(y_true, y_pred))
    assert math.isnan(mae(y_true, y_pred))
    assert math.isnan(forecast_bias(y_true, y_pred))
    assert math.isnan(forecast_correlation(y_true, y_pred))


def test_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        mse([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="same length"):
        directional_accuracy_against_baseline(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0],
        )


def test_qlike_positive_values() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 1.0])

    # Formula:
    #   mean(y_true / y_pred - log(y_true / y_pred) - 1)
    expected = np.mean(
        [
            1.0 / 1.0 - np.log(1.0 / 1.0) - 1.0,
            2.0 / 1.0 - np.log(2.0 / 1.0) - 1.0,
        ]
    )

    assert qlike(y_true, y_pred) == pytest.approx(expected)


def test_qlike_clips_for_stability_without_overwriting_inputs() -> None:
    y_true = pd.Series([0.0, 1.0])
    y_pred = pd.Series([0.0, 1.0])

    original_true = y_true.copy()
    original_pred = y_pred.copy()

    value = qlike(y_true, y_pred, eps=1e-8)

    assert np.isfinite(value)
    pd.testing.assert_series_equal(y_true, original_true)
    pd.testing.assert_series_equal(y_pred, original_pred)


def test_qlike_rejects_non_positive_eps() -> None:
    with pytest.raises(ValueError, match="eps must be positive"):
        qlike([1.0], [1.0], eps=0.0)


def test_forecast_bias_and_correlation() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.0, 3.0, 4.0, 5.0]

    assert forecast_bias(y_true, y_pred) == pytest.approx(1.0)
    assert forecast_correlation(y_true, y_pred) == pytest.approx(1.0)


def test_forecast_correlation_returns_nan_for_constant_series() -> None:
    assert math.isnan(forecast_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(forecast_correlation([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]))


def test_directional_accuracy_against_baseline() -> None:
    y_true = [2.0, 0.5, 1.0, 2.0]
    y_pred = [1.5, 0.4, 1.0, 0.5]
    baseline = [1.0, 1.0, 1.0, 1.0]

    # Forecast directions: +, -, 0, -
    # Realized directions: +, -, 0, +
    # Ties are excluded, so rows 0, 1, 3 remain.
    # Correct rows: 0, 1 => 2/3.
    assert directional_accuracy_against_baseline(
        y_true,
        y_pred,
        baseline,
    ) == pytest.approx(2.0 / 3.0)


def test_directional_accuracy_returns_nan_when_only_ties() -> None:
    y_true = [1.0, 1.0]
    y_pred = [1.0, 1.0]
    baseline = [1.0, 1.0]

    assert math.isnan(
        directional_accuracy_against_baseline(y_true, y_pred, baseline)
    )


def make_forecast_panel(market: str = "US") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "market": [market] * 6,
            "rv_gk_22d_forward_ann_label": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "har_rv_gk_22d_forecast_ann": [1.1, 1.9, 3.2, 3.8, 4.9, np.nan],
            "naive_lagged_22d_rv_ann": [1.0, 2.2, 2.8, 4.5, 5.0, 6.1],
            "expanding_mean_forward_rv_baseline": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "rolling_mean_forward_rv_baseline": [1.0, 1.7, 2.2, 3.0, 3.6, 4.1],
        }
    )


def test_evaluate_forecasts_schema_and_values() -> None:
    df = make_forecast_panel("US")

    result = evaluate_forecasts(
        df,
        target_col="rv_gk_22d_forward_ann_label",
        forecast_cols=[
            "har_rv_gk_22d_forecast_ann",
            "naive_lagged_22d_rv_ann",
        ],
        market="US",
    )

    assert list(result.columns) == METRIC_COLUMNS
    assert len(result) == 2

    har_row = result.loc[
        result["forecast_col"] == "har_rv_gk_22d_forecast_ann"
    ].iloc[0]

    assert har_row["market"] == "US"
    assert har_row["target_col"] == "rv_gk_22d_forward_ann_label"

    # Valid HAR pairs are rows 0, 1, 2, 3.
    assert int(har_row["n_obs"]) == 4
    assert np.isfinite(har_row["mse"])
    assert np.isfinite(har_row["rmse"])
    assert np.isfinite(har_row["mae"])
    assert np.isfinite(har_row["qlike"])


def test_evaluate_forecasts_missing_target_raises() -> None:
    df = make_forecast_panel("US").drop(columns=["rv_gk_22d_forward_ann_label"])

    with pytest.raises(ValueError, match="Missing target column"):
        evaluate_forecasts(
            df,
            target_col="rv_gk_22d_forward_ann_label",
            forecast_cols=["har_rv_gk_22d_forecast_ann"],
        )


def test_evaluate_forecasts_missing_forecast_raises() -> None:
    df = make_forecast_panel("US")

    with pytest.raises(ValueError, match="Missing forecast column"):
        evaluate_forecasts(
            df,
            target_col="rv_gk_22d_forward_ann_label",
            forecast_cols=["missing_forecast"],
        )


def test_build_forecast_accuracy_table_combines_us_and_india() -> None:
    us_df = make_forecast_panel("US")
    india_df = make_forecast_panel("INDIA")

    table = build_forecast_accuracy_table(
        us_df,
        india_df,
        target_col="rv_gk_22d_forward_ann_label",
        forecast_cols=[
            "har_rv_gk_22d_forecast_ann",
            "naive_lagged_22d_rv_ann",
            "expanding_mean_forward_rv_baseline",
            "rolling_mean_forward_rv_baseline",
        ],
    )

    assert list(table.columns) == METRIC_COLUMNS
    assert set(table["market"]) == {"US", "INDIA"}
    assert set(table["forecast_col"]) == {
        "har_rv_gk_22d_forecast_ann",
        "naive_lagged_22d_rv_ann",
        "expanding_mean_forward_rv_baseline",
        "rolling_mean_forward_rv_baseline",
    }

    assert len(table) == 8
    assert (table["n_obs"] > 0).all()


def test_build_forecast_accuracy_table_handles_missing_optional_forecast_cols() -> None:
    us_df = make_forecast_panel("US").drop(columns=["rolling_mean_forward_rv_baseline"])
    india_df = make_forecast_panel("INDIA")

    table = build_forecast_accuracy_table(
        us_df,
        india_df,
        target_col="rv_gk_22d_forward_ann_label",
    )

    assert not table.empty
    assert "har_rv_gk_22d_forecast_ann" in set(table["forecast_col"])
    assert "naive_lagged_22d_rv_ann" in set(table["forecast_col"])