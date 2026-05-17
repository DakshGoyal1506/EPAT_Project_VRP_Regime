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

## Phase Roadmap

### Phase 0 — Repo Foundation

Create the installable Python package, configuration files, script entry points, schema definitions, validation utilities, tests, and reproducibility conventions.

Output:

```text
installable repo skeleton
minimal OHLCV schema
schema validators
dry-run data script
passing tests
```

### Phase 1 — Data Ingestion

Implement loaders for:

```text
CBOE VIX
FRED VIXCLS
Yahoo Finance SPX / SPY / NIFTY / India VIX
NSE India VIX
optional IBKR / iBridgePy broker cache
```

Output:

```text
raw data files
standardized interim files
data audit report

```

### Phase 2 — Realised Variance

Implement realised variance estimators:

```text
close-to-close
Parkinson
Garman-Klass
Rogers-Satchell
Yang-Zhang optional
rolling realised variance windows
```

Output:

```text
daily realised variance panels
estimator comparison tables
```

### Phase 3 — Implied Variance and VRP

Construct implied variance proxies from VIX and India VIX, align them to realised variance, and construct VRP.

Output:

```text
US VRP panel
India VRP panel
summary statistics
alignment checks
```

### Phase 4 — HAR-RV Forecasting

Build HAR-RV forecasts using only information available at time `t`.

Output:

```text
forecasted realised variance
prospective VRP measure
forecast diagnostics
```

### Phase 5 — Threshold Regimes

Build simple interpretable regime filters using VIX, realised volatility, and VRP thresholds.

Output:

```text
threshold regime labels
baseline regime strategy signals
```

### Phase 6 — Gaussian HMM

Train Gaussian HMM regime models using expanding or walk-forward logic.

Critical rule:

```text
Backtests must use filtered probabilities available at time t.
Do not use full-sample smoothed probabilities for strategy decisions.
```

Output:

```text
HMM filtered regime probabilities
state diagnostics
regime labels
```

### Phase 7 — AR-HMM / Markov Autoregression

Upgrade regime modelling to account for autocorrelation in volatility and VRP series.

Output:

```text
Markov autoregression regime probabilities
comparison versus Gaussian HMM
```

### Phase 8 — Strategy and Backtest

Test unconditional versus regime-conditioned short-volatility exposure.

Output:

```text
equity curves
performance metrics
drawdown diagnostics
transaction cost sensitivity
```

### Phase 9 — Robustness

Test sensitivity to:

```text
realised variance estimator
regime model
state count
training window
transaction cost assumptions
market sample period
```

Output:

```text
robustness tables
failure cases
parameter sensitivity plots
```

### Phase 10 — Cross-Market Analysis

Compare US and India VRP behaviour and regime transitions.

Output:

```text
lead-lag diagnostics
cross-market regime transition analysis
correlation tables
```

### Phase 11 — Broker Paper-Signal Layer

Create an optional iBridgePy / IBKR paper-signal adapter.

This layer must not be required for core research reproducibility.

Output:

```text
daily broker-aware signal
paper-only exposure instruction
risk checks
no live orders
```

### Phase 12 — Final Report

Generate final tables, figures, diagnostics, and written conclusions.

Output:

```text
reports/final_report.md
reports/figures/
reports/tables/
```

## Installation

From the repository root:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Phase 0 Dry Run

```bash
python scripts/download_data.py --dry-run
```

Expected behaviour:

```text
Print intended data sources.
Do not download anything.
Do not write files.
Exit successfully.
```

## Data Policy

This repository separates data into four levels:

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

The project must obey these rules:

1. Do not use future realised variance as a tradable signal.
2. Do not use future returns in feature construction.
3. Do not use full-sample HMM-smoothed probabilities in backtests.
4. Use only filtered probabilities available at time `t`.
5. Fit, transform, and evaluate using explicit train/test or walk-forward boundaries.
6. All strategy signals must be timestamped before returns are realized.

## No-Live-Trading Warning

This repository is for academic research and paper-signal generation.

The broker layer is disabled by default.

The project must not place live orders unless explicitly changed later.

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

## Current Phase Status

```text
Phase: 0
Status: repo foundation
Finance logic: not implemented yet
Broker logic: paper-signal placeholder only
Tests: minimal schema and validation tests
```

## Quick Start — Run data scripts and tests

Follow these steps from the repository root.

- Activate the Conda environment (example):

	```powershell
	conda activate epat
	```

- Install editable package + dev deps (one-time):

	```bash
	pip install -e .
	pip install -e ".[dev]"
	```

- Dry-run the ingestion (no downloads or writes):

	```bash
	python scripts/download_data.py --dry-run
	```

- Download and write specific source families (examples):

	```bash
	# Download all Yahoo US sources
	python scripts/download_data.py --market US --source yahoo --force

	# Download only FRED sources for US
	python scripts/download_data.py --market US --source fred --force

	# Download only CBOE sources for US
	python scripts/download_data.py --market US --source cboe --force

	# Use a local CSV override for CBOE/NSE
	python scripts/download_data.py --market US --source cboe --source-id cboe_vix --local-csv data/manual/cboe/VIX_History.csv --force
	```

- Run the test suite relevant to data and loaders:

	```bash
	pytest tests/test_data_loaders.py tests/test_data_schema.py -q
	```

## Data & Scripts

- Ingested raw files are written to `data/raw/`.
- Canonical processed datasets are written to `data/processed/`.
- Data audit table is `reports/tables/data_audit.csv`.

See `src/vrp/data/README.md` for loader-specific commands and `data/README.md` for dataset descriptions.

