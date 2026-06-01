# Forecasting

This package implements the HAR-RV forecasting engine used to build prospective variance risk premium (VRP) forecasts (Phase 4). Forecasting logic is kept testable and reusable for downstream regime and backtest components.

Responsibilities

- Build HAR predictors from Phase 2/3 feature panels using only information available at t-1.
- Produce walk-forward forecasts and coefficient history with optional HAC checkpointing.
- Emit no-lookahead audit tables that prove the forecast availability rule.

Key modules

- `har_rv.py` — HAR predictor construction, batched estimation backends, walk-forward orchestration.
- `har_registry.py` — registry of primary predictors and supervised targets; enforces naming and no-lookahead constraints.

Quick commands

```bash
# Train HAR across all markets (expanding window; GPU optional)
python scripts/train_har.py --market ALL --mode expanding --force

# Run HAR unit tests
pytest tests/test_har_batched_backend.py tests/test_har_rv.py
```

No-lookahead audit (concept)

Audit rows must satisfy `max_training_target_end_date < forecast_date` for every forecast that is marked available. See `reports/tables/har_no_lookahead_audit.csv` for the canonical audit output and the python snippet in the repository for a runnable check.

Primary outputs

- Forecast panels: `data/processed/*_har_forecast.parquet`
- HAR-VRP panels: `data/processed/*_vrp_har.parquet`
- Diagnostics: `reports/tables/har_forecast_accuracy.csv`, `reports/tables/har_coefficients.csv`, `reports/tables/har_vrp_summary.csv`, `reports/tables/har_no_lookahead_audit.csv`

Notes

- Supported estimation backends: `cpu_numpy_batched` and `torch_batched` (optional GPU).
- Configure HAC checkpointing and coefficient frequency via `configs/har_rv.yaml`.
