# Commands

Run all commands from the repository root.

## Setup

```bash
pip install -e .
pip install -e "[dev]"
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

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
pytest tests/test_rv_estimators.py
```

## Phase 3 — Implied Variance and VRP

```bash
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
pytest tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
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
