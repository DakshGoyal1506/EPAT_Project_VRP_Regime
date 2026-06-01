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
docs/commands.md
docs/phases/
scripts/README.md
src/vrp/*/README.md
```
```text
docs/phase_status.md
```

High-level project phases:

| Phase | Scope                                                         |
| ----: | ------------------------------------------------------------- |
|     0 | Repo scaffold, packaging, configuration, directory governance |
|     1 | Public data ingestion                                         |
|     2 | Realised variance estimators                                  |
|     3 | Implied variance and VRP construction                         |
|     4 | HAR-RV forecasting                                            |
|     5 | Threshold regimes                                             |
|     6 | Gaussian HMM regimes                                          |
|     7 | AR-HMM / Markov autoregression                                |
|     8 | Regime-conditioned strategy construction                      |
|     9 | Robustness analysis                                           |
|    10 | Cross-market diagnostics                                      |
|    11 | Broker paper-signal layer                                     |
|    12 | Final report assembly                                         |
|    13 | Final repo freeze and release checklist                       |

For phase status, validation commands, generated artifact policy, and known limitations, use the `docs/` folder.

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

## Common Commands

Complete command documentation lives in:

```text
docs/commands.md
```

### Setup

```bash
pip install -e .
pip install -e ".[dev]"
pytest
```

### Phase 1 — Data Ingestion

Dry run:

```bash
python scripts/download_data.py --market ALL --source all --dry-run
```

US data refresh:

```bash
python scripts/download_data.py --market US --source all --force
```

India Yahoo fallback refresh:

```bash
python scripts/download_data.py --market INDIA --source yahoo --force
```

### Phase 2 — Realised Variance

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

### Phase 3 — Implied Variance and VRP

```bash
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
```

### Phase 4 — HAR-RV Forecasting

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --coefficient-hac-frequency none
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend numpy --coefficient-hac-frequency none
```

### Later Phase Entry Points

```bash
python scripts/train_regimes.py --help
python scripts/train_markov_autoreg.py --help
python scripts/run_backtest.py --help
python scripts/run_robustness.py --help
python scripts/run_ibkr_paper_signal.py --help
```

### Hygiene Checks Before Commit

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.log \.env"
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

