import pandas as pd
import numpy as np
from pathlib import Path

from vrp.forecasting.har_rv import (
    load_har_config,
    expanding_window_har_forecast,
    HARConfig,
)


def make_synthetic_panel(n=80, horizon=3):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "market": ["US"] * n,
    })

    # create daily rv with positive numbers
    df["rv_gk_daily"] = 0.0001 + 0.00001 * np.arange(n)

    # create lagged 22d ann rv as simple moving average of daily
    df["rv_gk_22d_ann"] = (
        df["rv_gk_daily"].shift(1).rolling(window=22, min_periods=1).mean() * 252
    )

    # lag1
    df["rv_gk_22d_ann_lag1"] = df["rv_gk_22d_ann"].shift(1)

    df["iv_ann"] = df["rv_gk_22d_ann"] + 0.00005

    # construct forward target using horizon-day rolling mean
    future_rv = df["rv_gk_daily"].shift(-1).rolling(window=horizon, min_periods=horizon).mean() * 252
    df["rv_gk_22d_forward_ann_label"] = future_rv

    # HAR features
    df["har_rv_d_lag1_ann"] = df["rv_gk_daily"].shift(1) * 252
    df["har_rv_w_lag1_ann"] = df["rv_gk_daily"].shift(1).rolling(window=5, min_periods=5).mean() * 252
    df["har_rv_m_lag1_ann"] = df["rv_gk_daily"].shift(1).rolling(window=22, min_periods=22).mean() * 252

    # set target_start_date and target_end_date
    df["target_start_date"] = df["date"].shift(-1)
    df["target_end_date"] = df["date"].shift(-horizon)
    df["target_col"] = "rv_gk_22d_forward_ann_label"

    df = df.reset_index(drop=True)
    return df


def compare_backends():
    panel = make_synthetic_panel()

    cfg = load_har_config(Path("configs/har_rv.yaml"))
    # small sample overrides
    cfg = cfg.model_copy(update={"min_train_observations": 10, "rolling_train_window": 20})

    cfg_stats = cfg.model_copy(update={"compute_backend": "cpu_statsmodels"})
    cfg_numpy = cfg.model_copy(update={"compute_backend": "cpu_numpy_batched"})

    f_stats, c_stats, a_stats = expanding_window_har_forecast(panel.copy(), cfg_stats)
    f_np, c_np, a_np = expanding_window_har_forecast(panel.copy(), cfg_numpy)

    y1 = f_stats["har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    y2 = f_np["har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)

    # compare available mask
    mask1 = np.isfinite(y1)
    mask2 = np.isfinite(y2)
    assert np.array_equal(mask1, mask2)

    # compare values where available
    if mask1.sum() > 0:
        np.testing.assert_allclose(y1[mask1], y2[mask1], rtol=1e-6, atol=1e-12)


def test_batched_vs_statsmodels():
    compare_backends()
