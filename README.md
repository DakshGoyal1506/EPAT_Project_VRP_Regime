# EPAT VRP Regime Project

## Project Title

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Objective

This project studies the variance risk premium across two equity-index volatility markets:

- US: SPX / VIX
- India: NIFTY / India VIX

The objective is to measure the variance risk premium, decompose it across volatility regimes, and test whether regime-conditioned short-volatility exposure improves risk-adjusted performance relative to unconditional short-volatility harvesting.

The project is designed as a reproducible research pipeline, not a notebook-only analysis.

## Core Research Questions

1. Is the variance risk premium structurally positive in both US and Indian markets?
2. Does the premium behave differently across calm, transition, and stress regimes?
3. Can regime filters reduce drawdowns relative to unconditional short-volatility exposure?
4. Does a Gaussian HMM add value beyond simple threshold regimes?
5. Does an AR-HMM / Markov autoregression improve regime modelling when volatility and VRP are autocorrelated?
6. Are US and Indian volatility regimes synchronized, lagged, or structurally different?

## Pipeline

```text
public daily data
	↓
clean OHLC / VIX / India VIX series
	↓
realised variance estimators
	↓
implied variance construction
	↓
variance risk premium construction
	↓
HAR-RV forecast
	↓
threshold regimes
	↓
Gaussian HMM regimes
	↓
AR-HMM / Markov autoregression upgrade
	↓
regime-conditioned strategy
	↓
vectorised backtest
	↓
robustness tests
	↓
cross-market lead-lag analysis
	↓
iBridgePy / IBKR paper-signal layer
	↓
final report
```

## Repository Structure

```text
epat-vrp-regime/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── broker_cache/
├── src/vrp/
│   ├── data/
│   ├── features/
│   ├── forecasting/
│   ├── regimes/
│   ├── strategies/
│   ├── backtest/
│   ├── broker/
│   └── reports/
├── notebooks/
├── scripts/
├── tests/
├── reports/
├── pyproject.toml
├── README.md
└── .env.example
```

## Documentation Index

```text
docs/phase_status.md              Current status by phase
docs/artifact_inventory.md        Local artifact manifest template and tracked/local policy
docs/reproducibility.md           Environment setup and rerun protocol
docs/generated_artifact_policy.md  Commit vs local-only artifact rules
docs/commands.md                  Canonical commands by phase
docs/known_limitations.md         Current limitations and non-goals
```

## Notebooks

The `notebooks/` folder is for inspection and presentation only. Production logic should stay in `src/vrp/`.

Current notebook index:

- `01_data_audit.ipynb` - data-quality checks and audit exploration.
- `02_build_features.ipynb` - realized-variance feature build walkthrough for the processed OHLC panels.
- `03.ipynb` - implied variance and VRP inspection.
- `03_har_rv.ipynb` - HAR-RV forecast and HAR-based VRP inspection.

## Phase Roadmap

### Completed Phases

#### Phase 0 - Repo Foundation

Status: complete.

What is implemented:

```text
installable Python package
project configuration and script entry points
schema definitions and validators
basic test harness
reproducible workspace structure
```

Use this from the repository root:

```bash
pip install -e .
pip install -e ".[dev]"
pytest
```

#### Phase 1 - Data Ingestion

Status: complete.

What is implemented:

```text
CBOE VIX loader
FRED VIXCLS loader
Yahoo Finance OHLC loaders for US and India markets
NSE India VIX loader
data schema validation and audit support
```

Key commands:

```bash
python scripts/download_data.py --dry-run
pytest tests/test_data_loaders.py tests/test_data_schema.py
```

#### Phase 2 - Realised Variance

Status: complete.

What is implemented:

```text
close-to-close daily variance
Parkinson daily variance
Garman-Klass daily variance
Rogers-Satchell daily variance
Yang-Zhang rolling variance
trailing rolling realised variance windows
annualized RV panels
RV validation and diagnostics
```

Key modules and functions:

```text
vrp.features.returns.compute_log_returns
vrp.features.returns.compute_simple_returns
vrp.features.returns.add_gap_return
vrp.features.returns.add_intraday_return
vrp.features.returns.add_all_returns
vrp.features.realized_variance.validate_ohlc
vrp.features.realized_variance.close_to_close_daily_var
vrp.features.realized_variance.parkinson_daily_var
vrp.features.realized_variance.garman_klass_daily_var
vrp.features.realized_variance.rogers_satchell_daily_var
vrp.features.realized_variance.rolling_realized_variance
vrp.features.realized_variance.yang_zhang_rolling_var
vrp.features.realized_variance.annualize_variance
vrp.features.realized_variance.annualize_vol
vrp.features.realized_variance.build_rv_panel
```

Build and test commands:

```bash
python scripts/build_features.py --market US --feature rv --window 22
python scripts/build_features.py --market INDIA --feature rv --window 22
python scripts/build_features.py --market ALL --feature rv --window 22
pytest tests/test_rv_estimators.py
```

Expected outputs:

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet
reports/tables/
reports/figures/
```

Notebook support for this phase lives in `notebooks/02_build_features.ipynb`.

#### Phase 3 - Implied Variance and VRP

Status: complete.

What is implemented:

```text
VIX / India VIX implied variance construction
exact-date IV/RV alignment
backward-looking point-in-time VRP
forward ex-post RV and VRP labels
feature/label separation registry
calendar mismatch diagnostics
VRP metadata and plots
```

Key commands:

```bash
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
pytest tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Expected outputs:

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

#### Phase 4 - HAR-RV Forecasting
Build HAR-RV forecasts using only information available at time `t`.

Status: complete and ready for freeze (Phase 4). The HAR-RV engine produces
walk-forward expanding and rolling forecasts under strict no-lookahead rules,
coefficient histories with optional HAC inference checkpoints, and HAR-based
prospective VRP panels for both US and India markets.

Freeze / Run commands (appendix):

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
pytest
```

Audit and validation snippets are preserved in the repository's Phase 4
appendix (see `src/vrp/forecasting/README.md` and the `scripts/` folder).

### Next Phases

#### Phase 5 - Threshold Regimes

Build simple interpretable regime filters using VIX, realised volatility, and VRP thresholds.

#### Phase 6 - Gaussian HMM

Train Gaussian HMM regime models using expanding or walk-forward logic.

Critical rule:

```text
Backtests must use filtered probabilities available at time t.
Do not use full-sample smoothed probabilities for strategy decisions.
```

#### Phase 7 - AR-HMM / Markov Autoregression

Upgrade regime modelling to account for autocorrelation in volatility and VRP series.

#### Phase 8 - Strategy and Backtest

Test unconditional versus regime-conditioned short-volatility exposure.

#### Phase 9 - Robustness

Test sensitivity to estimator choice, regime model, state count, training window, transaction costs, and sample period.

#### Phase 10 - Cross-Market Analysis

Compare US and India VRP behaviour and regime transitions.

#### Phase 11 - Broker Paper-Signal Layer

Create an optional iBridgePy / IBKR paper-signal adapter.

#### Phase 12 - Final Report

Generate final tables, figures, diagnostics, and written conclusions.

## Installation

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quick Start

From the repository root:

```powershell
conda activate epat
```

```bash
python scripts/download_data.py --dry-run
python scripts/build_features.py --market ALL --feature rv --window 22
pytest
```

## Current Phase Status

The source of truth for phase status is:

```text
docs/phase_status.md
```

This README gives the project overview only. Phase-specific completion status, generated artifacts, validation commands, and known limitations are tracked in `docs/`.


## Data Policy

```text
data/raw/          source-format downloaded files
data/interim/      cleaned but not final research panels
data/processed/    final model-ready datasets
data/broker_cache/ optional broker-side cached data
```

Rules:

1. Do not commit large raw data files.
2. Do not commit broker cache files.
3. Do not commit private credentials.
4. Keep every transformation reproducible through scripts and source modules.
5. Notebooks are for inspection only.
6. Production logic must live inside `src/vrp/`.
7. Every phase must include tests.

## No-Lookahead Policy

1. Do not use future realised variance as a tradable signal.
2. Do not use future returns in feature construction.
3. Do not use full-sample HMM-smoothed probabilities in backtests.
4. Use only filtered probabilities available at time `t`.
5. Fit, transform, and evaluate using explicit train/test or walk-forward boundaries.
6. All strategy signals must be timestamped before returns are realized.

## No-Live-Trading Warning

This repository is for academic research and paper-signal generation.

Default broker mode:

```text
paper_signal_only: true
live_trading_enabled: false
```

Any future broker integration must include:

```text
explicit paper mode
risk checks
position limits
order preview logs
manual review checkpoint
```

