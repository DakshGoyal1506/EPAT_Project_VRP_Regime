# EPAT VRP Regime Project

## Project Title

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Executive Summary

This repository implements a reproducible research pipeline for measuring the variance risk premium (VRP), decomposing it across volatility regimes, and evaluating whether regime-conditioned short-volatility exposure improves risk-adjusted performance relative to unconditional short-volatility harvesting. The project compares US SPX/VIX with Indian NIFTY/India VIX using public daily data, point-in-time feature construction, forecasting, regime detection, vectorised backtesting, robustness checks, cross-market diagnostics, and an optional paper-signal broker layer.

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

| Phase | Status | Summary |
|---:|---|---|
| 0 | Complete / frozen | Repo scaffold, package layout, environment setup, documentation governance, generated artifact policy |
| 1 | Complete / frozen | Public daily data ingestion for US and India; canonical OHLCV schema; audit table; loader tests |
| 2 | Complete / frozen | Realised variance estimators and RV panels; primary `rv_gk_22d_ann` |
| 3 | Complete / frozen | Implied variance, exact-date IV/RV alignment, VRP features, and no-lookahead labels |
| 4 | Complete / frozen | HAR-RV forecasting and HAR-based prospective VRP |
| 5 | Complete / frozen | Threshold baseline regime construction |
| 6 | Complete / frozen | Gaussian HMM regime model with train-only scaling, train-only fitting, custom filtered probabilities, diagnostic-only smoothed probabilities, and no-lookahead audit |
| 7 | Complete / frozen | Markov autoregression regime model with train-only fitting, filtered probabilities, diagnostic-only smoothing, and no-lookahead audit |
| 8 | Complete / frozen | Python-only MSVOL robustness appendix; diagnostic-only return-volatility regime comparison; true R MSGARCH optional/future only |
| 9 | Complete / frozen | Strategy signal construction |
| 10 | Complete / needs final review | Vectorised research backtest and robustness |
| 11 | Complete / needs final review | IBKR paper-signal readiness layer |
| 12 | Not started | Optional future IBKR paper execution adapter |
| 13 | Not started | Cross-market US-India analysis |
| 14 | Blocked | Final report / release package |

## Repository Structure

```text
EPAT_Project_VRP_Regime/
├── configs/
│   ├── README.md
│   ├── data_sources.yaml
│   ├── markets.yaml
│   ├── har_rv.yaml
│   ├── model_hmm.yaml
│   ├── model_arhmm.yaml
│   ├── model_markov_autoreg.yaml
│   ├── model_msvol.yaml
│   ├── model_msgarch.yaml
│   ├── strategies.yaml
│   ├── backtest.yaml
│   └── ibkr_paper.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── manual/
│   └── broker_cache/
├── docs/
│   ├── phase_status.md
│   ├── artifact_inventory.md
│   ├── reproducibility.md
│   ├── generated_artifact_policy.md
│   ├── commands.md
│   ├── known_limitations.md
│   └── phases/
├── notebooks/
│   ├── README.md
│   ├── 01_data_audit.ipynb
│   ├── 02_build_features.ipynb
│   ├── 03.ipynb
│   └── 03_har_rv.ipynb
├── reports/
│   ├── README.md
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── README.md
│   ├── download_data.py
│   ├── build_features.py
│   ├── train_har.py
│   ├── train_regimes.py
│   ├── train_markov_autoreg.py
│   ├── build_signals.py
│   ├── run_backtest.py
│   ├── run_robustness.py
│   └── run_ibkr_paper_signal.py
├── src/
│   └── vrp/
│       ├── README.md
│       ├── data/
│       ├── features/
│       ├── forecasting/
│       ├── regimes/
│       ├── strategies/
│       ├── backtest/
│       ├── broker/
│       └── reports/
├── tests/
│   └── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Documentation Map

| File | Purpose |
|---|---|
| `docs/phase_status.md` | Authoritative phase ledger |
| `docs/phases/` | Detailed per-phase documentation |
| `docs/artifacts/` | Detailed per-phase generated-artifact documentation |
| `docs/commands.md` | Global command index |
| `docs/reproducibility.md` | Environment setup and rerun protocol |
| `docs/artifact_inventory.md` | Local artifact inventory and review substitutes |
| `docs/generated_artifact_policy.md` | Commit vs local-only artifact rules |
| `docs/known_limitations.md` | Current limitations and non-goals |
| `scripts/README.md` | Script entry points and CLI usage |
| `src/vrp/*/README.md` | Module-specific contracts and commands |

## Installation

Run from the repository root:

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quick Start

Minimal smoke check:

```bash
pytest
python scripts/download_data.py --dry-run
```

For phase-specific commands:

```text
docs/commands.md
scripts/README.md
src/vrp/*/README.md
docs/phases/
```

## Generated Artifact Policy

Generated data, model outputs, broker cache, logs, large panels, and diagnostic figures are local-only by default.

Tracked:

```text
source code
configs
scripts
tests
docs
README files
.gitkeep placeholders
.env.example
pyproject.toml
.gitignore
```

Local-only:

```text
data/raw/*
data/interim/*
data/processed/*
data/manual/*
data/broker_cache/*
reports/tables/*
reports/figures/*
logs/*
*.parquet
*.pkl
*.pickle
*.joblib
*.log
.env
```

Detailed policy:

```text
docs/generated_artifact_policy.md
docs/artifact_inventory.md
```

## No-Lookahead Policy

1. Do not use future realised variance as a tradable signal.
2. Do not use future returns in feature construction.
3. Do not use full-sample HMM-smoothed probabilities in backtests.
4. Use only filtered probabilities available at time `t`.
5. Fit, transform, and evaluate using explicit train/test or walk-forward boundaries.
6. Timestamp strategy signals before returns are realised.

## Notebook Policy

Notebooks are for inspection, diagnostics, and presentation only. Production logic belongs in `src/vrp/`, `scripts/`, and `tests/`.

## No-Live-Trading Warning

This repository is for academic research and paper-signal generation.

Default broker policy:

```text
paper_signal_only: true
live_trading_enabled: false
allow_order_placement: false
```

The broker layer is optional and must not be used as the primary historical research data source.
