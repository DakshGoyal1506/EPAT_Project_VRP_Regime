# Commands

Run all commands from the repository root.

## Setup

```bash
pip install -e .
pip install -e ".[dev]"
pytest
```

## Git Hygiene

```bash
git status --short
git ls-files data reports docs | sort
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.log \.env"
```

Expected: no generated parquet/model/log/env files tracked.

## Phase 0 — Scaffold and Governance

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
```

## Phase 1 — Data Ingestion

Dry run:

```bash
python scripts/download_data.py --market ALL --source all --dry-run
```

US:

```bash
python scripts/download_data.py --market US --source all --force
```

India Yahoo fallback:

```bash
python scripts/download_data.py --market INDIA --source yahoo --force
```

Manual NSE override:

```bash
python scripts/download_data.py --market INDIA --source nse --source-id nse_india_vix --local-csv data/manual/nse/india_vix.csv --force
```

Tests:

```bash
pytest tests/test_data_loaders.py tests/test_data_schema.py
```

## Phase 2 — Realised Variance

Build realised variance panels and diagnostics:

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

Run Phase 2 tests:

```bash
pytest tests/test_rv_estimators.py
```

Generated outputs are local-only by default:

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet
reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json
reports/figures/rv_estimators_us.png
reports/figures/rv_estimators_india.png
```

## Phase 3 — Implied Variance and VRP

Build implied variance panels only:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build implied variance, VRP panels, and VRP diagnostics:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

Run Phase 3 tests:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Generated outputs are local-only by default:

```text
data/processed/us_iv.parquet
data/processed/india_iv.parquet
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/calendar_mismatches.csv
reports/figures/us_iv_rv_vrp.png
reports/figures/india_iv_rv_vrp.png
```

## Phase 4 — HAR-RV Forecasting

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --coefficient-hac-frequency none
pytest tests/test_har_rv.py
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend numpy --coefficient-hac-frequency none
```

## Phase 5–6 — Threshold and Gaussian HMM Regimes

```bash
python scripts/train_regimes.py --help
pytest tests/test_hmm_model.py tests/test_hmm_filtering.py
```

## Phase 7 — Markov Autoregression / AR-HMM

```bash
python scripts/train_markov_autoreg.py --help
pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py
```

## Phase 8 — Strategy and Backtest

```bash
python scripts/run_backtest.py --help
pytest tests/test_backtest_accounting.py tests/test_backtest_metrics.py
```

## Phase 9 — Robustness

```bash
python scripts/run_robustness.py --help
pytest tests/test_robustness.py
```

## Phase 10 — Cross-Market Analysis

```bash
python scripts/audit_phase10_inputs.py --help
python scripts/audit_phase10_final.py --help
```

## Phase 11 — Broker Paper-Signal Layer

```bash
python scripts/run_ibkr_paper_signal.py --help
python scripts/validate_phase11.py --help
pytest tests/test_paper_trader.py tests/test_live_order_guard.py tests/test_risk_checks.py
```

## Full Test Suite

```bash
pytest
```

## Notebook Cleanup Before Commit

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
