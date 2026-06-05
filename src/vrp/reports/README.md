# Reports

This package provides reporting and diagnostic utilities that assemble reproducible tables, metadata, audits, and figures from canonical pipeline outputs.

## Phase Ownership

Primary reporting support:

```text
Phase 2 - realised variance diagnostics
Phase 3 - implied variance and VRP diagnostics
Phase 4 - HAR-RV diagnostics
Phase 10 - backtest diagnostics
Phase 11 - broker readiness diagnostics
Phase 13 - cross-market diagnostics
```

## Responsibilities

- Summarise generated research panels into stable review tables.
- Emit diagnostic audit tables that verify no-lookahead conditions and data availability.
- Produce reproducible figures under `reports/figures/`.
- Write metadata files that describe run configuration and output contracts.
- Keep generated artifacts local-only unless explicitly approved as final-report artifacts.

## Main Modules

| Module | Purpose |
|---|---|
| `rv_diagnostics.py` | Realised-variance diagnostics and calendar mismatch reports |
| `vrp_diagnostics.py` | VRP summaries, metadata writers, and plotting helpers |
| `backtest_diagnostics.py` | Phase 10 backtest diagnostics |
| `broker_diagnostics.py` | Phase 11 broker-readiness diagnostics |
| `cross_market.py` | Phase 13 cross-market panels, audits, correlations, lead-lag diagnostics, Granger diagnostics, and logistic diagnostics |
| `cross_market_diagnostics.py` | Phase 13 report tables, figures, summary index, and validation helpers |

## Phase 13 Commands

```bash
python scripts/run_cross_market_analysis.py --validate-inputs-only
python scripts/run_cross_market_analysis.py --model ALL --force
pytest tests/test_cross_market_alignment.py tests/test_cross_market_no_lookahead.py tests/test_cross_market_stats.py
```

## Safety Boundaries

- Reporting modules must not create tradable signals from future or outcome columns.
- Full-sample smoothed probabilities are diagnostic-only.
- Phase 13 predictive panels must use `us_lagged_date < india_date`.
- Granger diagnostics are descriptive only and must not be reported as causal proof.
- Reports must not read Phase 11 broker artifacts for Phase 13.
