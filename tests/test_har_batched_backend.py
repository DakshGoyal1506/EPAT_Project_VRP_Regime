import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from vrp.forecasting.har_rv import (
    load_har_config,
    expanding_window_har_forecast,
    add_forward_target_metadata,
    HARConfig,
)


def make_synthetic_panel(n=80, horizon=22):
    """
    Create synthetic HAR panel with properly constructed forward target.
    
    Forward target is built as:
        rv_gk_22d_forward_ann_label_t = 252 * mean(rv_gk_daily[t+1], ..., rv_gk_daily[t+h])
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "market": ["US"] * n,
    })

    # Create daily rv with positive numbers
    df["rv_gk_daily"] = 0.0001 + 0.00001 * np.arange(n)

    # Create lagged 22d ann rv as simple moving average of daily
    df["rv_gk_22d_ann"] = (
        df["rv_gk_daily"].shift(1).rolling(window=22, min_periods=1).mean() * 252
    )

    # lag1
    df["rv_gk_22d_ann_lag1"] = df["rv_gk_22d_ann"].shift(1)

    df["iv_ann"] = df["rv_gk_22d_ann"] + 0.00005

    # Build forward target: 252 * mean(daily[t+1], ..., daily[t+h])
    # Use concat([daily.shift(-i) for i in 1..horizon])
    future_dailies = [df["rv_gk_daily"].shift(-i) for i in range(1, horizon + 1)]
    future_stack = pd.concat(future_dailies, axis=1)
    # Only compute mean where all horizon values are available
    valid = future_stack.notna().sum(axis=1) == horizon
    df["rv_gk_22d_forward_ann_label"] = (
        252 * future_stack.mean(axis=1)
    ).where(valid)

    # HAR features
    df["har_rv_d_lag1_ann"] = df["rv_gk_daily"].shift(1) * 252
    df["har_rv_w_lag1_ann"] = df["rv_gk_daily"].shift(1).rolling(window=5, min_periods=5).mean() * 252
    df["har_rv_m_lag1_ann"] = df["rv_gk_daily"].shift(1).rolling(window=22, min_periods=22).mean() * 252

    # Add proper target metadata using the canonical function
    df = add_forward_target_metadata(
        df,
        horizon=horizon,
        target_col="rv_gk_22d_forward_ann_label",
    )

    df = df.reset_index(drop=True)
    return df


def compare_backends():
    """Compare cpu_statsmodels vs cpu_numpy_batched backends."""
    panel = make_synthetic_panel(n=80, horizon=22)

    cfg = load_har_config(Path("configs/har_rv.yaml"))
    # Small sample overrides
    cfg = cfg.model_copy(update={
        "min_train_observations": 10,
        "rolling_train_window": 20,
        "forecast_horizon": 22,
    })

    cfg_stats = cfg.model_copy(update={"compute_backend": "cpu_statsmodels"})
    cfg_numpy = cfg.model_copy(update={"compute_backend": "cpu_numpy_batched"})

    f_stats, c_stats, a_stats = expanding_window_har_forecast(panel.copy(), cfg_stats)
    f_np, c_np, a_np = expanding_window_har_forecast(panel.copy(), cfg_numpy)

    # Check forecast_available masks match
    avail_stats = f_stats["har_forecast_available"].to_numpy(dtype=bool)
    avail_np = f_np["har_forecast_available"].to_numpy(dtype=bool)
    
    assert np.array_equal(avail_stats, avail_np), \
        f"Forecast availability masks differ: statsmodels={avail_stats.sum()}, numpy={avail_np.sum()}"

    # Check blocked reasons match for blocked rows
    blocked_stats = ~avail_stats
    blocked_np = ~avail_np
    blocked_both = blocked_stats & blocked_np
    
    if blocked_both.any():
        reasons_stats = f_stats.loc[blocked_both, "har_blocked_reason"].to_numpy()
        reasons_np = f_np.loc[blocked_both, "har_blocked_reason"].to_numpy()
        assert np.array_equal(reasons_stats, reasons_np), \
            f"Blocked reasons differ at blocked indices"

    # Compare forecast values where available
    y_stats = f_stats.loc[avail_stats, "har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    y_np = f_np.loc[avail_np, "har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    
    if len(y_stats) > 0:
        np.testing.assert_allclose(y_stats, y_np, rtol=1e-6, atol=1e-12,
                                   err_msg="Forecast values diverge on available rows")


def test_batched_vs_statsmodels():
    """Test cpu_statsmodels vs cpu_numpy_batched equivalence."""
    compare_backends()


def test_torch_batched_equivalence():
    """Test torch_batched backend equivalence vs numpy if torch is available."""
    torch = pytest.importorskip("torch")
    
    panel = make_synthetic_panel(n=80, horizon=22)
    
    cfg = load_har_config(Path("configs/har_rv.yaml"))
    cfg = cfg.model_copy(update={
        "min_train_observations": 10,
        "rolling_train_window": 20,
        "forecast_horizon": 22,
    })
    
    cfg_numpy = cfg.model_copy(update={"compute_backend": "cpu_numpy_batched"})
    cfg_torch = cfg.model_copy(update={
        "compute_backend": "torch_batched",
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_dtype": "float64",
    })
    
    f_np, c_np, a_np = expanding_window_har_forecast(panel.copy(), cfg_numpy)
    f_torch, c_torch, a_torch = expanding_window_har_forecast(panel.copy(), cfg_torch)
    
    # Check forecast_available masks match
    avail_np = f_np["har_forecast_available"].to_numpy(dtype=bool)
    avail_torch = f_torch["har_forecast_available"].to_numpy(dtype=bool)
    
    assert np.array_equal(avail_np, avail_torch), \
        f"Torch forecast availability differs: numpy={avail_np.sum()}, torch={avail_torch.sum()}"
    
    # Compare forecast values where available
    y_np = f_np.loc[avail_np, "har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    y_torch = f_torch.loc[avail_torch, "har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    
    if len(y_np) > 0:
        np.testing.assert_allclose(y_np, y_torch, rtol=1e-5, atol=1e-10,
                                   err_msg="Torch forecast values diverge from numpy")
