"""
Smoke parity script: compare `cpu_numpy_batched` vs `torch_batched` HAR backends on a synthetic panel.

Run:
    conda activate epat
    python scripts/smoke_backend_parity.py

The script builds a synthetic increasing RV series, computes HAR features and forward target,
then runs the batched forecast for `cpu_numpy_batched` and `torch_batched` (if torch importable),
and reports max absolute differences in forecasts and coefficients.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from vrp.forecasting.har_rv import (
    load_har_config,
    make_har_features,
    add_forward_target_metadata,
    prepare_har_model_frame,
    _initialize_forecast_columns,
    batched_har_forecast,
)


def make_synthetic_panel(n=600, start_date="2018-01-01"):
    dates = pd.date_range(start_date, periods=n, freq="D")
    panel = pd.DataFrame({"date": dates, "market": ["US"] * n})
    panel["rv_gk_daily"] = 0.0001 + 0.000001 * np.arange(n)

    panel = make_har_features(panel, daily_rv_col="rv_gk_daily", horizon=22, annualization_periods=252)

    horizon = 22
    future_cols = [panel["rv_gk_daily"].shift(-i) for i in range(1, horizon + 1)]
    future_stack = pd.concat(future_cols, axis=1)
    valid = future_stack.notna().sum(axis=1) == horizon
    panel["rv_gk_22d_forward_ann_label"] = (252 * future_stack.mean(axis=1)).where(valid)

    # add required IV column
    panel["iv_ann"] = panel["har_rv_m_lag1_ann"].fillna(0.0) + 0.00005

    panel = add_forward_target_metadata(panel, horizon=22, target_col="rv_gk_22d_forward_ann_label")

    return panel


def run_batched(panel: pd.DataFrame, cfg, backend: str):
    cfg2 = cfg.model_copy(update={"compute_backend": backend})
    work = prepare_har_model_frame(panel, config=cfg2, validate_target=False)
    out = _initialize_forecast_columns(work)
    forecast, coeff, audit = batched_har_forecast(out, cfg2, mode="expanding")
    return forecast, coeff, audit


def main():
    cfg = load_har_config(Path("configs/har_rv.yaml"))
    # make config friendly for fast smoke test
    cfg = cfg.model_copy(update={
        "min_train_observations": 5,
        "rolling_train_window": 1000,
        "coefficient_hac_frequency": "none",
    })

    panel = make_synthetic_panel(n=500)

    print("Running cpu_numpy_batched...")
    f_n, c_n, a_n = run_batched(panel, cfg, "cpu_numpy_batched")

    try:
        import importlib, torch  # noqa: F401
        have_torch = True
    except Exception:
        have_torch = False

    if not have_torch:
        print("Torch not available; skipping torch_batched comparison.")
        return

    print("Running torch_batched...")
    f_t, c_t, a_t = run_batched(panel, cfg, "torch_batched")

    # forecasts comparison
    mask_n = f_n["har_forecast_available"].to_numpy(dtype=bool)
    mask_t = f_t["har_forecast_available"].to_numpy(dtype=bool)
    common = mask_n & mask_t
    print(f"Rows={len(f_n)}, forecasts (numpy)={mask_n.sum()}, (torch)={mask_t.sum()}, common={common.sum()}")

    fvals_n = f_n["har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    fvals_t = f_t["har_rv_gk_22d_forecast_ann"].to_numpy(dtype=float)
    max_forecast_diff = np.nanmax(np.abs(fvals_n - fvals_t))
    print(f"Max forecast abs diff: {max_forecast_diff:.6e}")

    # coefficients: merge on identifying columns
    if len(c_n) == 0 or len(c_t) == 0:
        print("No coefficient rows to compare.")
        return

    merge_cols = ["date", "market", "model_name", "train_start_date", "train_end_date", "n_train"]
    merged = c_n.merge(c_t, on=merge_cols, suffixes=("_n", "_t"))
    if len(merged) == 0:
        print("No matching coefficient rows to compare after merge.")
        return

    coef_cols = [
        "coef_const",
        "coef_har_rv_d_lag1_ann",
        "coef_har_rv_w_lag1_ann",
        "coef_har_rv_m_lag1_ann",
    ]

    diffs = {}
    for col in coef_cols:
        arr = np.abs(merged[f"{col}_n"].to_numpy(float) - merged[f"{col}_t"].to_numpy(float))
        diffs[col] = float(np.nanmax(arr))

    print("Max coefficient abs diffs:")
    for k, v in diffs.items():
        print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()