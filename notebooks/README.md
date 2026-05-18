# Notebooks

Notebooks are for inspection, diagnostics, and presentation only.

## Available Notebooks

- `01_data_audit.ipynb` - audit and validation exploration for processed data.
- `02_build_features.ipynb` - realised-variance feature build walkthrough.
- `03.ipynb` - implied variance and VRP inspection, including robustness diagnostics.
 - `03_har_rv.ipynb` - HAR-RV forecast, baseline comparison, residual, coefficient, audit, and HAR-VRP inspection.

## Phase 4 note

`03_har_rv.ipynb` is inspection-only. It must not fit HAR models, tune parameters, write production outputs, create regimes, create trading signals, or run backtests.

## Rules

1. Do not place production logic here.
2. Do not hardcode finance logic here.
3. Do not manually repair data here.
4. Use notebooks only to call functions from `src/vrp/`.
5. If logic becomes useful, move it into `src/vrp/` and add tests.