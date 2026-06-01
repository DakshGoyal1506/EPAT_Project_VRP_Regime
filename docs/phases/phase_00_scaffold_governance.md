# Phase 0 — Project Scaffold, Environment, and Governance

## Status

Complete / frozen.

Phase 0 defines the project foundation. It does not implement research logic.

## Objective

Create a research-grade Python repository for the EPAT VRP regime project with:

```text
installable package
source package layout
configuration directory
script entry points
tests
notebooks policy
data/report artifact policy
environment setup
safe broker placeholders
documentation governance
```

## Research Project

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting:
A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Phase Boundary

Phase 0 includes:

```text
repo scaffold
pyproject setup
README structure
.gitignore
.env.example
src/vrp package structure
empty module folders
initial schema and validators
script placeholders
test harness
docs governance
artifact policy
```

Phase 0 excludes:

```text
data download
realised variance
implied variance
VRP
forecasting
regime modelling
strategy logic
backtesting
broker execution
final report conclusions
```

## Files Owned by This Phase

```text
README.md
pyproject.toml
.env.example
.gitignore
configs/
data/README.md
data/raw/.gitkeep
data/interim/.gitkeep
data/processed/.gitkeep
data/manual/.gitkeep
data/manual/cboe/.gitkeep
data/manual/nse/.gitkeep
data/broker_cache/.gitkeep
docs/
docs/phases/
notebooks/README.md
reports/README.md
reports/figures/README.md
reports/figures/.gitkeep
reports/tables/README.md
reports/tables/.gitkeep
scripts/README.md
tests/README.md
src/vrp/__init__.py
src/vrp/*/__init__.py
src/vrp/*/README.md
```

## Package Layout

```text
src/vrp/
├── data/
├── features/
├── forecasting/
├── regimes/
├── strategies/
├── backtest/
├── broker/
└── reports/
```

## Environment Setup

Run from repository root:

```bash
pip install -e .
pip install -e ".[dev]"
pytest
```

## Reproducibility Commands

Minimum smoke check:

```bash
pytest
python scripts/download_data.py --dry-run
```

Git hygiene:

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.log \.env"
```

Expected generated-artifact check:

```text
.env.example
```

No `.parquet`, `.pkl`, `.pickle`, `.joblib`, `.log`, or real `.env` files should be tracked.

## Artifact Policy

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
model binaries
broker cache
private credentials
notebook outputs
```

## No-Lookahead Governance

The project-wide rules begin in Phase 0:

1. Do not use future realised variance as a tradable signal.
2. Do not use future returns in feature construction.
3. Do not use full-sample HMM-smoothed probabilities in backtests.
4. Use filtered probabilities available at time `t`.
5. Keep production logic in `src/vrp/`.
6. Keep notebooks inspection-only.

## No-Live-Trading Governance

Broker layer is paper/signal-only by default.

Required defaults:

```text
paper_signal_only: true
live_trading_enabled: false
allow_order_placement: false
```

## Validation Checklist

Phase 0 is closed when:

```text
repo installs
pytest passes
dry-run command works
.gitignore blocks generated artifacts
README explains the project
docs explain phase status and artifact policy
no private credentials are tracked
no generated panels are tracked
no model binaries are tracked
```

## Review Packet

Send:

```text
README.md
pyproject.toml
.gitignore
.env.example
docs/
configs/
src/vrp/ tree
tests/ tree
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.log \.env"
pytest output
```
