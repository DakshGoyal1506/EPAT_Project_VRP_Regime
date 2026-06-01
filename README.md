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

# EPAT VRP Regime Project

## Project Title

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Executive Summary

This repository implements a reproducible research pipeline for measuring the variance risk premium (VRP), decomposing it across volatility regimes, and evaluating whether regime-conditioned short-volatility exposure improves risk-adjusted performance relative to unconditional short-volatility harvesting. The project compares the US SPX/VIX market with the Indian NIFTY/India VIX market using public daily data, point-in-time feature construction, forecasting, regime detection, vectorised backtesting, robustness checks, cross-market diagnostics, and an optional paper-signal broker layer.

## Research Objective

The project studies whether the variance risk premium is persistent across markets, whether it changes meaningfully across volatility regimes, and whether regime-aware exposure rules improve drawdown control and risk-adjusted returns.

Markets covered:

| Market | Underlying | Implied-volatility proxy |
|---|---|---|
| US | SPX / SPY | VIX |
| India | NIFTY 50 | India VIX |

## Core Research Questions

1. Is the variance risk premium structurally positive in both US and Indian markets?
2. Does the premium behave differently across calm, transition, and stress regimes?
3. Can regime filters reduce drawdowns relative to unconditional short-volatility exposure?
4. Does a Gaussian HMM add value beyond simple threshold regimes?
5. Does a Markov autoregression improve regime modelling when volatility and VRP are autocorrelated?
6. Are US and Indian volatility regimes synchronized, lagged, or structurally different?

## Research Pipeline

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
AR-HMM / Markov autoregression
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

## Current Implementation Status

| Phase | Status                                      | Summary                                                                                               |
| ----: | ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
|     0 | Complete / frozen                           | Repo scaffold, package layout, environment setup, documentation governance, generated artifact policy |
|     1 | Complete / frozen                           | Public daily data ingestion for US and India, canonical OHLCV schema, audit table, loader tests       |
|    2+ | Implemented or in progress in later modules | See `docs/phase_status.md` for the authoritative phase ledger                                         |

## Phase 0 Summary — Scaffold and Governance

Phase 0 created the research-grade repository foundation:

```text
installable Python package
src-layout package under src/vrp/
configs/scripts/tests/docs structure
data/report/model/log artifact policy
.env.example with no private credentials
notebook inspection policy
no-live-trading baseline
generated-vs-tracked file rules
```

Phase 0 does not implement finance logic.

Details:

```text
docs/phases/phase_00_scaffold_governance.md
```

## Phase 1 Summary — Public Data Ingestion

Phase 1 implements the public daily data layer:

```text
CBOE VIX loader
FRED VIXCLS loader
Yahoo Finance OHLCV loaders for US and India
NSE/manual CSV ingestion path
canonical OHLCV schema
raw source Parquet outputs generated locally
processed canonical Parquet outputs generated locally
data audit table
internet-free loader tests with mocks/fixtures
```

Canonical OHLCV schema:

```text
date
open
high
low
close
adj_close
volume
source
market
symbol
```

Phase 1 explicitly excludes realised variance, implied variance, VRP, forecasting, regimes, strategy logic, backtesting, and broker integration.

Details:

```text
docs/phases/phase_01_data_ingestion.md
src/vrp/data/README.md
```

## Repository Structure

```text
EPAT_Project_VRP_Regime/
├── configs/
│   ├── README.md
│   ├── data_sources.yaml
│   ├── markets.yaml
│   ├── har_rv.yaml
   ├── model_hmm.yaml
   ├── model_arhmm.yaml
   ├── model_markov_autoreg.yaml
   ├── strategies.yaml
   ├── backtest.yaml
   └── ibkr_paper.yaml
├── data/
│   ├── README.md
   ├── raw/
   ├── interim/
   ├── processed/
   ├── manual/
   └── broker_cache/
├── docs/
│   ├── phase_status.md
   ├── artifact_inventory.md
   ├── reproducibility.md
   ├── generated_artifact_policy.md
   ├── commands.md
   ├── known_limitations.md
   └── phases/
├── notebooks/
│   ├── README.md
   ├── 01_data_audit.ipynb
   ├── 02_build_features.ipynb
   ├── 03.ipynb
   └── 03_har_rv.ipynb
├── reports/
│   ├── README.md
   ├── figures/
   └── tables/
├── scripts/
│   ├── README.md
   ├── download_data.py
   ├── build_features.py
   ├── train_har.py
   ├── train_regimes.py
   ├── train_markov_autoreg.py
   ├── run_backtest.py
   ├── run_robustness.py
   └── run_ibkr_paper_signal.py
├── src/
│   └── vrp/
│       ├── README.md
      ├── data/
      ├── features/
      ├── forecasting/
      ├── regimes/
      ├── strategies/
      ├── backtest/
      ├── broker/
      └── reports/
├── tests/
│   └── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Documentation Map

| File                                | Purpose                                         |
| ----------------------------------- | ----------------------------------------------- |
| `docs/phase_status.md`              | Authoritative phase ledger                      |
| `docs/phases/`                      | Detailed phase-level documentation              |
| `docs/commands.md`                  | Global command index                            |
| `docs/reproducibility.md`           | Environment setup and rerun protocol            |
| `docs/artifact_inventory.md`        | Local artifact inventory and review substitutes |
| `docs/generated_artifact_policy.md` | Commit vs local-only artifact rules             |
| `docs/known_limitations.md`         | Current limitations and non-goals               |
| `scripts/README.md`                 | Script entry points and CLI usage               |
| `src/vrp/*/README.md`               | Module-specific contracts and commands          |
```
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

