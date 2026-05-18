# HAR Forecasting (Phase 4)

This package implements the Phase 4 HAR-RV forecasting engine and walk-forward
orchestration used to build prospective variance risk premium (VRP) panels.

Contents
- `har_rv.py` — HAR feature construction, walk-forward forecasting, batched
  closed-form OLS backends (NumPy, optional Torch GPU), audit and coefficient
  history utilities, HAC checkpointing.
- `har_registry.py` — strict registry of primary HAR predictors and targets.

Phase 4 summary
- Builds conservative lagged HAR predictors that use information only through
  t-1.
- Uses Phase 3 forward 22-day realised-variance label as the supervised target.
- Enforces the no-lookahead rule: training row s is allowed only if
  `target_end_date_s < forecast_date_t`.
- Produces forecast panels, coefficient history (with `hac_available` flag),
  and no-lookahead audit tables.

Freeze checklist (commands)

```bash
# Full-market expanding run (GPU optional)
python scripts/train_har.py --market ALL --mode expanding --force \
  --backend torch_batched --torch-device cuda --torch-dtype float64 \
  --coefficient-hac-frequency none

# Run tests
pytest
```

Audit validation (python snippet)

```bash
python - << 'PY'
import pandas as pd

audit = pd.read_csv("reports/tables/har_no_lookahead_audit.csv")
available = audit[audit["forecast_available"].astype(bool)].copy()

available["forecast_date"] = pd.to_datetime(available["forecast_date"])
available["max_training_target_end_date"] = pd.to_datetime(
    available["max_training_target_end_date"]
)

bad = available[available["max_training_target_end_date"] >= available["forecast_date"]]
assert bad.empty
assert available["rule_target_end_before_forecast_date"].astype(bool).all()
print("HAR no-lookahead audit passed.")
print("Available rows:", len(available))
PY
```

Outputs
- Forecast panels: `data/processed/us_har_forecast.parquet`, `data/processed/india_har_forecast.parquet`
- HAR-VRP panels: `data/processed/us_vrp_har.parquet`, `data/processed/india_vrp_har.parquet`
- Reports: `reports/tables/har_forecast_accuracy.csv`, `reports/tables/har_coefficients.csv`, `reports/tables/har_vrp_summary.csv`, `reports/tables/har_no_lookahead_audit.csv`

Notes
- The batched solver supports `cpu_numpy_batched` and `torch_batched` backends.
- HAC checkpointing uses the last available coefficient date per month/quarter
  (not calendar month-end) and can be configured via `coefficient_hac_frequency`.
- See `scripts/smoke_backend_parity.py` for a quick parity check between NumPy
  and Torch backends.
