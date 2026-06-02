# Phase 4 - HAR-RV Forecasting and HAR-Based Prospective VRP

## Status

Complete / frozen.

This phase is implemented, tested, and ready for downstream regime phases. Future edits should be limited to bug fixes, safety/no-lookahead fixes, reproducibility fixes, documentation updates, or tests for existing behaviour.

## Objective

Phase 4 builds a point-in-time HAR-RV forecasting engine and uses it to construct a prospective variance risk premium.

The primary forecast is:

```text
har_rv_gk_22d_forecast_ann
```

The primary HAR-based VRP output is:

```text
vrp_har_gk = iv_ann - har_rv_gk_22d_forecast_ann
```

## Phase Boundary

This phase does:

- build HAR-RV predictors
- validate the Phase 3 forward realised-variance label
- estimate direct 22-trading-day HAR-RV forecasts
- compare HAR forecasts against naive baselines
- construct HAR-based prospective VRP
- write forecast panels, HAR-VRP panels, diagnostics, metadata, and audit tables

This phase does not:

- build threshold regimes
- train HMMs
- train Markov autoregressions
- construct trading signals
- run backtests
- run broker or IBKR logic
- download new data
- use option-chain data

## Files Owned by This Phase

```text
configs/har_rv.yaml
scripts/train_har.py
scripts/smoke_backend_parity.py
src/vrp/forecasting/har_rv.py
src/vrp/forecasting/har_registry.py
src/vrp/forecasting/forecast_evaluation.py
src/vrp/reports/har_diagnostics.py
tests/test_har_rv.py
tests/test_har_batched_backend.py
tests/test_forecast_evaluation.py
```

Phase 4 also extends:

```text
src/vrp/features/vrp.py
tests/test_no_lookahead.py
```

## Main Functions, Classes, and Scripts

Config:

```text
HARConfig
load_har_config
```

Feature and target construction:

```text
make_har_features
add_forward_target_metadata
recompute_forward_target_for_validation
validate_phase3_target_matches_recomputed_target
prepare_har_model_frame
```

Estimation and forecasting:

```text
fit_har_ols
predict_har
expanding_window_har_forecast
rolling_window_har_forecast
batched_har_forecast
batched_solve_ols_numpy
batched_solve_ols_torch
```

Backend selection:

```text
resolve_compute_backend
resolve_torch_device
resolve_torch_dtype
```

Evaluation:

```text
mse
rmse
mae
qlike
forecast_bias
forecast_correlation
evaluate_forecasts
build_forecast_accuracy_table
```

HAR-VRP construction:

```text
compute_har_vrp
```

Diagnostics:

```text
write_har_diagnostics
build_har_forecast_summary
build_har_vrp_summary
plot_har_forecast_vs_realized
plot_har_residuals
plot_har_vrp
write_har_metadata
```

## Config Files Used

```text
configs/har_rv.yaml
```

Key configuration fields:

```text
primary_estimator: garman_klass
primary_daily_rv_col: rv_gk_daily
primary_forward_label_col: rv_gk_22d_forward_ann_label
primary_iv_col: iv_ann
forecast_horizon: 22
annualization_periods: 252
timing_mode: conservative_lag1
model_type: direct_har_22d
oos_mode: expanding
min_train_observations: 500
compute_backend: torch_batched
torch_device: auto
torch_dtype: float64
coefficient_hac_frequency: month_end
forecast_floor: 1.0e-8
```

## Input Files

Phase 4 consumes Phase 3 VRP panels:

```text
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
```

Required columns include:

```text
date
market
iv_ann
rv_gk_daily
rv_gk_22d_ann_lag1
rv_gk_22d_forward_ann_label
vrp_forward_expost_gk_label
feature_allowed
```

## Generated Output Files

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

## Commit Policy

Commit:

```text
source code
configs
scripts
tests
documentation
README files
.gitkeep placeholders
```

Keep local-only unless explicitly approved:

```text
data/processed/*_har_forecast.parquet
data/processed/*_vrp_har.parquet
reports/tables/har_*.csv
reports/tables/har_*.json
reports/figures/har_*.png
```

## Regeneration Commands

GPU-capable primary run:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend cpu_numpy_batched --coefficient-hac-frequency none
```

Single-market runs:

```bash
python scripts/train_har.py --market US --mode expanding --force
python scripts/train_har.py --market INDIA --mode expanding --force
```

Backend parity smoke check:

```bash
python scripts/smoke_backend_parity.py
```

## Tests to Run

```bash
pytest tests/test_har_rv.py
pytest tests/test_har_batched_backend.py
pytest tests/test_forecast_evaluation.py
pytest tests/test_no_lookahead.py
pytest
```

## Validation Checklist

1. HAR feature columns are strictly lagged.
2. The HAR target is the existing Phase 3 forward realised-variance label.
3. Target validation recomputes `t+1 ... t+22` only.
4. No current-day realised variance enters HAR predictors.
5. No IV, VRP, forecast, forward, ex-post, or label column enters HAR predictors.
6. Training rows satisfy `target_end_date_s < forecast_date_t`.
7. Forecasts are not forward-filled or backfilled.
8. `vrp_har_gk` is null whenever `har_forecast_available` is false.
9. Forecast accuracy table compares HAR against naive baselines.
10. Audit table is generated and passes the no-lookahead rule.

## No-Lookahead and Safety Rules

At date `t`, HAR predictors may use realised variance only through `t-1`.

The supervised target is:

```text
rv_gk_22d_forward_ann_label_t
= 252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})
```

This target is allowed only for training and evaluation. It is forbidden as a tradable feature, regime feature, or strategy signal.

For a forecast at date `t`, a training row `s` is allowed only when:

```text
target_end_date_s < forecast_date_t
```

`vrp_har_gk` is valid only when:

```text
har_forecast_available == True
```

## Known Limitations

1. HAR forecasts are point estimates, not distribution forecasts.
2. Forecast quality can differ materially by market and sample.
3. GPU acceleration is optional; CPU fallback is supported.
4. HAC coefficient inference can be disabled for faster production forecast runs.
5. The primary HAR model intentionally excludes IV and VRP predictors to keep the specification clean.
6. The outputs are daily research features, not executable trading instructions.

## Review Checklist

Before closing Phase 4, run the tests and inspect the generated audit if local outputs exist:

```bash
pytest
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```

Expected tracked generated-artifact result:

```text
.env.example
```
