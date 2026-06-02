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
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
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

Primary GPU-capable run:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
pytest tests/test_har_rv.py tests/test_har_batched_backend.py tests/test_forecast_evaluation.py tests/test_no_lookahead.py
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend cpu_numpy_batched --coefficient-hac-frequency none
```

Backend parity smoke check:

```bash
python scripts/smoke_backend_parity.py
```

## Phase 5 - Threshold Baseline Regimes

CLI help:

```bash
python scripts/train_regimes.py --help
```

Build threshold regimes:

```bash
python scripts/train_regimes.py --model threshold --market US --force
python scripts/train_regimes.py --model threshold --market INDIA --force
python scripts/train_regimes.py --model threshold --market ALL --force
```

Tests:

```bash
pytest tests/test_threshold_regimes.py
pytest tests/test_regime_no_lookahead.py
pytest tests/test_no_lookahead.py
```

Generated outputs are local-only by default:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
reports/tables/threshold_*.csv
reports/tables/threshold_*.json
reports/figures/threshold_*.png
```

## Phase 6 - Gaussian HMM Regimes

CLI help:

```bash
python scripts/train_regimes.py --help
```

Run configured primary plus fallback:

```bash
python scripts/train_regimes.py --market US --model gaussian_hmm --primary --force
python scripts/train_regimes.py --market INDIA --model gaussian_hmm --primary --force
```

Run full Phase 6 candidate grid:

```bash
python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force
```

Tests:

```bash
pytest tests/test_hmm_filtering.py
pytest tests/test_hmm_scaling.py
pytest tests/test_hmm_model.py
pytest tests/test_hmm_no_lookahead.py
pytest tests/test_no_lookahead.py
```

Generated outputs are local-only by default:

```text
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
data/processed/*_hmm_*.parquet
models/*_hmm_*.pkl
models/us_gaussian_hmm.pkl
models/india_gaussian_hmm.pkl
reports/tables/phase_6/us/*
reports/tables/phase_6/india/*
reports/figures/phase_6/*
```

## Phase 7 - Markov Autoregression / AR-Aware Regimes

CLI help:

```bash
python scripts/train_markov_autoreg.py --help
```

Primary K=2 model:

```bash
python scripts/train_markov_autoreg.py --market US --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market INDIA --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force
```

Approved candidate grid:

```bash
python scripts/train_markov_autoreg.py --market ALL --run-grid --force
```

Tests:

```bash
pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py
```

Generated outputs are local-only by default:

```text
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
data/processed/markov_autoreg/*.parquet
models/us_markov_autoreg.pkl
models/india_markov_autoreg.pkl
models/markov_autoreg/*.pkl
reports/tables/phase_7/us/*
reports/tables/phase_7/india/*
reports/figures/phase_7/*
```

## Phase 8 - Python-only MSVOL Robustness Appendix

Phase 8 is diagnostic-only. It is not true MSGARCH and is not used for strategy construction.

Help commands:

```bash
python scripts/export_msgarch_inputs.py --help
python scripts/run_msvol_regimes.py --help
python scripts/import_msvol_outputs.py --help
python scripts/run_msvol_diagnostics.py --help
python scripts/run_msvol_no_lookahead_audit.py --help
```

Local regeneration sequence:

```bash
python scripts/export_msgarch_inputs.py --market ALL
python scripts/run_msvol_regimes.py --market ALL
python scripts/import_msvol_outputs.py --market ALL
python scripts/run_msvol_diagnostics.py --market ALL
python scripts/run_msvol_no_lookahead_audit.py --market ALL
```

Tests:

```bash
pytest tests/test_msgarch_export.py
pytest tests/test_msvol_model.py
pytest tests/test_msvol_adapter.py
pytest tests/test_msvol_diagnostics.py
pytest tests/test_msvol_no_lookahead.py
```

Generated outputs are local-only by default:

```text
data/interim/msgarch/*_msgarch_input.csv
data/interim/msvol/*
data/processed/*_msvol_regimes.parquet
reports/tables/phase_8/*
reports/figures/phase_8/*
```

## Phase 9 - Strategy Signal Construction

CLI help:

```bash
python scripts/build_signals.py --help
```

Build Phase 9 signals:

```bash
python scripts/build_signals.py --market US --strategy all --force
python scripts/build_signals.py --market INDIA --strategy all --force
python scripts/build_signals.py --market ALL --strategy all --force
```

Tests:

```bash
pytest tests/test_exposure_rules.py tests/test_signal_builder.py tests/test_strategy_no_lookahead.py tests/test_phase9_diagnostics.py
```

Generated outputs are local-only by default:

```text
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
reports/tables/phase_9/strategy_*.csv
reports/tables/phase_9/strategy_metadata.json
reports/figures/phase_9/*
```

## Phase 10 - Vectorised Research Backtest and Robustness

CLI help:

```bash
python scripts/audit_phase10_inputs.py --help
python scripts/run_backtest.py --help
python scripts/generate_backtest_diagnostics.py --help
python scripts/run_robustness.py --help
python scripts/audit_phase10_final.py --help
```

Regenerate local Phase 10 outputs:

```bash
python scripts/audit_phase10_inputs.py --market ALL
python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force
python scripts/generate_backtest_diagnostics.py --market ALL
python scripts/run_robustness.py --market ALL --test all --force
python scripts/audit_phase10_final.py --market ALL
```

Single-strategy inspection must use dry run:

```bash
python scripts/run_backtest.py --market US --strategy unconditional_full --cost-bps 5 --dry-run
```

Tests:

```bash
pytest tests/test_phase10_input_schema.py tests/test_backtest_config_registry.py tests/test_backtest_accounting.py tests/test_backtest_no_lookahead.py
pytest tests/test_backtest_metrics.py tests/test_vectorized_engine.py tests/test_backtest_diagnostics.py tests/test_robustness.py tests/test_phase10_integration.py
```

Generated outputs are local-only by default:

```text
data/processed/*_backtest_panel.parquet
data/processed/*_backtest_panel_metadata.json
reports/tables/phase_10/*
reports/figures/phase_10/*
```

## Phase 11 - IBKR Paper-Signal Readiness Layer

```bash
python scripts/run_ibkr_paper_signal.py --help
python scripts/validate_phase11.py --help
pytest tests/test_paper_trader.py tests/test_live_order_guard.py tests/test_risk_checks.py tests/test_validate_phase11_cli.py
```

## Phase 12 - Optional Future IBKR Paper Execution Adapter

Phase 12 is intentionally not implemented. It requires explicit re-scoping before any broker execution adapter is added.

## Phase 13 - Cross-Market US-India Analysis

```bash
python scripts/audit_phase13_inputs.py --help
```

If Phase 13 scripts are not present yet, keep this phase in design/review status and do not reuse Phase 10 backtest commands as cross-market validation.

## Phase 14 - Final Report and Release Package

```bash
pytest
git status --short
git ls-files data reports docs | sort
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```

## Full Test Suite

```bash
pytest
```

## Notebook Cleanup Before Commit

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
