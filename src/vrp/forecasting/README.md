# Forecasting

## Purpose

This package owns Phase 4 HAR-RV forecasting and HAR-based prospective VRP construction.

It turns Phase 3 VRP panels into point-in-time forecasts of future realised variance and then constructs:

```text
vrp_har_gk = iv_ann - har_rv_gk_22d_forecast_ann
```

## Phase Ownership

Primary phase:

```text
Phase 4 - HAR-RV forecasting and HAR-based prospective VRP
```

Downstream consumers:

```text
Phase 5 threshold regimes
Phase 6 Gaussian HMM
Phase 7 Markov autoregression
Phase 9 strategy signal construction
Phase 10 backtest and robustness
```

## Responsibilities

This module must:

1. Build conservative lagged HAR predictors.
2. Use the existing Phase 3 forward RV label as the supervised target.
3. Validate target construction against `rv_gk_daily`.
4. Run expanding or rolling walk-forward HAR forecasts.
5. Enforce `target_end_date_s < forecast_date_t`.
6. Support statsmodels, NumPy batched, and Torch batched backends.
7. Produce coefficient history and optional HAC checkpoint inference.
8. Write no-lookahead audit data through the script/report pipeline.
9. Support forecast accuracy diagnostics.
10. Feed HAR forecasts into HAR-VRP construction.

## Main Modules

```text
har_rv.py
    HARConfig, config loading, feature construction, target metadata,
    walk-forward forecasting, backend solvers, audit rows, coefficient rows.

har_registry.py
    Strict registry of HAR predictors, target columns, forecast columns,
    forbidden predictor names, and output features.

forecast_evaluation.py
    MSE, RMSE, MAE, QLIKE, bias, correlation, directional accuracy,
    forecast accuracy table construction.
```

## Expected Inputs

Phase 4 consumes Phase 3 VRP panels:

```text
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
```

Required columns:

```text
date
market
iv_ann
rv_gk_daily
rv_gk_22d_ann_lag1
rv_gk_22d_forward_ann_label
```

## Expected Outputs

Forecast panels:

```text
data/processed/us_har_forecast.parquet
data/processed/india_har_forecast.parquet
```

HAR-VRP panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Report tables:

```text
reports/tables/har_forecast_accuracy.csv
reports/tables/har_coefficients.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_metadata.json
reports/tables/har_no_lookahead_audit.csv
```

Figures:

```text
reports/figures/har_forecast_us.png
reports/figures/har_forecast_india.png
reports/figures/har_residuals_us.png
reports/figures/har_residuals_india.png
reports/figures/har_vrp_us.png
reports/figures/har_vrp_india.png
```

Generated outputs are local-only by default.

## Commands

Primary GPU-capable run:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend cpu_numpy_batched --coefficient-hac-frequency none
```

Backend parity smoke check:

```bash
python scripts/smoke_backend_parity.py
```

CLI help:

```bash
python scripts/train_har.py --help
```

## Tests

```bash
pytest tests/test_har_rv.py
pytest tests/test_har_batched_backend.py
pytest tests/test_forecast_evaluation.py
pytest tests/test_no_lookahead.py
```

## Safety and No-Lookahead Boundaries

The HAR target is:

```text
rv_gk_22d_forward_ann_label_t
= 252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})
```

This target is label-only.

HAR predictors are:

```text
har_rv_d_lag1_ann
har_rv_w_lag1_ann
har_rv_m_lag1_ann
```

These use realised variance only through `t-1`.

A training row `s` is allowed for forecast date `t` only when:

```text
target_end_date_s < forecast_date_t
```

`vrp_har_gk` is valid only when:

```text
har_forecast_available == True
```

No HAR forecast is forward-filled or backfilled.

## This Module Must Not Do

This module must not:

1. Download data.
2. Rebuild Phase 1-3 data.
3. Create threshold regimes.
4. Train HMMs.
5. Create strategy signals.
6. Run backtests.
7. Run broker or IBKR logic.
8. Use forward/ex-post/label columns as live predictors.
9. Use full-sample future information.
10. Fill missing HAR forecasts silently.
