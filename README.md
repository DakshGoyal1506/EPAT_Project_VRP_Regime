# EPAT VRP Regime Project

## Project Title

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Executive Summary

This repository implements a reproducible research pipeline for measuring the variance risk premium (VRP), decomposing it across volatility regimes, and evaluating whether regime-conditioned short-volatility exposure improves research-proxy behaviour relative to unconditional short-volatility harvesting. The project compares US SPX/VIX with Indian NIFTY/India VIX using public daily data, point-in-time feature construction, forecasting, regime detection, vectorised backtesting, robustness checks, cross-market diagnostics, and a paper-signal readiness layer.

## Research Objective

The project studies whether the variance risk premium is persistent across markets, whether it changes meaningfully across volatility regimes, and whether regime-aware exposure rules improve drawdown control and risk-adjusted research-proxy behaviour.

Markets covered:

| Market | Underlying | Implied-volatility proxy |
|---|---|---|
| US | SPX / SPY | VIX |
| India | NIFTY 50 | India VIX |

## Research Pipeline

```text
public daily data
  -> clean OHLC / VIX / India VIX series
  -> realised variance estimators
  -> implied variance construction
  -> variance risk premium construction
  -> HAR-RV forecast
  -> threshold regimes
  -> Gaussian HMM regimes
  -> AR-HMM / Markov autoregression
  -> regime-conditioned strategy
  -> vectorised research backtest
  -> robustness tests
  -> iBridgePy / IBKR paper-signal readiness layer
  -> cross-market lead-lag analysis
  -> final report and release package
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
| 10 | Complete / frozen | Vectorised research backtest and robustness |
| 11 | Complete / frozen | IBKR paper-signal readiness layer |
| 12 | Skipped / future optional | IBKR paper execution adapter intentionally left out of current submission scope |
| 13 | Complete / frozen | Cross-market US-India analysis with strict previous-US-date alignment, descriptive same-date diagnostics, lagged-US predictive diagnostics, and analysis-only India overlay |
| 14 | In progress | Final report, PDF export, presentation package, claims audit, release checklist, and submission package |

## Repository Structure

```text
EPAT_Project_VRP_Regime/
├── configs/
├── data/
├── docs/
│   ├── phase_status.md
│   ├── artifact_inventory.md
│   ├── reproducibility.md
│   ├── generated_artifact_policy.md
│   ├── commands.md
│   ├── known_limitations.md
│   ├── phases/
│   └── artifacts/
├── notebooks/
├── reports/
│   ├── README.md
│   ├── final/
│   │   ├── README.md
│   │   ├── final_report.md
│   │   ├── final_report.pdf
│   │   ├── executive_summary.md
│   │   ├── presentation_outline.md
│   │   ├── selected_artifacts.md
│   │   ├── limitations.md
│   │   ├── reproducibility_note.md
│   │   ├── future_work.md
│   │   ├── result_claims_audit.md
│   │   ├── table_inventory.md
│   │   └── figure_inventory.md
│   ├── figures/
│   └── tables/
├── scripts/
├── src/
├── tests/
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
| `docs/release_checklist.md` | Final release checklist |
| `docs/final_report_checklist.md` | Final report and PDF QA checklist |
| `docs/submission_package.md` | Reviewer route map for final submission |
| `docs/phases/phase_11_ibkr_readiness.md` | Phase 11 implementation boundary and validation |
| `docs/artifacts/phase_11_artifacts.md` | Phase 11 local artifact schema and commit policy |
| `docs/phase11_runbook.md` | Phase 11 operational runbook |
| `docs/phases/phase_13_cross_market.md` | Phase 13 cross-market analysis boundary, commands, validation, and safety rules |
| `docs/artifacts/phase_13_artifacts.md` | Phase 13 generated artifact documentation |
| `reports/final/final_report.md` | Final report Markdown source |
| `reports/final/final_report.pdf` | Final report PDF export |
| `reports/final/executive_summary.md` | Final executive summary |
| `reports/final/presentation_outline.md` | Evidence-first presentation outline |
| `reports/final/result_claims_audit.md` | Claim-to-evidence audit |
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

## Final Report Package

Final submission files live under:

```text
reports/final/
```

Key entry points:

```text
reports/final/final_report.md
reports/final/final_report.pdf
reports/final/executive_summary.md
reports/final/presentation_outline.md
docs/submission_package.md
docs/release_checklist.md
```

`reports/final/final_report.md` is the source of truth.
`reports/final/final_report.pdf` is the export deliverable.

## Generated Artifact Policy

Generated data, model outputs, broker cache, logs, large panels, report tables, and diagnostic figures are local-only by default.

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
reports/final/*.md
reports/final/final_report.pdf
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
reports/final/selected_artifacts.md
```

## Research-Proxy and No-Account-Return Warning

Phase 10 backtest outputs are research-layer VRP proxy results. They are not executable option-chain PnL, account returns, or live trading results.

Cumulative curves are additive research-proxy curves, not account equity curves.

Strategy outputs are exposure intentions, not broker orders.

## No-Lookahead Policy

1. Do not use future realised variance as a tradable signal.
2. Do not use future returns in feature construction.
3. Do not use full-sample HMM-smoothed probabilities in backtests.
4. Use only filtered probabilities available at time `t`.
5. Fit, transform, and evaluate using explicit train/test or walk-forward boundaries.
6. Timestamp strategy signals before returns are realised.

## Notebook Policy

Notebooks are for inspection, diagnostics, and presentation only. Production logic belongs in `src/vrp/`, `scripts/`, and `tests/`.

## License, Attribution, and Commercial Use

This repository is public for portfolio, academic review, and professional evaluation purposes.

It is not released under a permissive open-source license.

License structure:

```text
Code:
    PolyForm Noncommercial License 1.0.0

Documentation, reports, figures, and presentation material:
    Creative Commons Attribution-NonCommercial 4.0 International

Commercial/profit use:
    Separate written permission required from Daksh Goyal
```

Attribution is required for non-commercial reuse.

Commercial use, profit-generating use, trading-desk use, product integration, paid training reuse, paid consulting reuse, redistribution in paid material, or use in live/paper trading systems requires separate written permission.

See:

```text
LICENSE.md
NOTICE.md
COMMERCIAL_USE.md
CITATION.cff
```

## No-Live-Trading Warning

This repository is for academic research, research-proxy evaluation, and paper-signal readiness. It does not claim live-trading profitability, true option-chain PnL, executable account returns, or broker execution.

Default broker policy:

```text
mode: paper_signal_only
paper_only: true
kill_switch: true
live_orders_enabled: false
allow_order_placement: false
live_order_sent: false
```

The broker layer is optional and must not be used as the primary historical research data source.
