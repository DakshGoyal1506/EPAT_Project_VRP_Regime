# `src/vrp`

This is the production source package for the EPAT VRP regime project.

Production logic belongs here, not in notebooks.

## Package Layout

| Folder | Purpose |
|---|---|
| `data/` | Data ingestion, schema, cleaners, validators, IO |
| `features/` | Returns, realised variance, implied variance, VRP, feature registries |
| `forecasting/` | HAR-RV forecasting and forecast diagnostics |
| `regimes/` | Threshold regimes, HMM, Markov autoregression, regime validation |
| `strategies/` | Regime-conditioned exposure rules and signal construction |
| `backtest/` | Vectorised backtest engine, metrics, costs, robustness |
| `broker/` | Paper-signal broker layer and safety guards |
| `reports/` | Diagnostic tables, plots, and report helpers |

## Rules

1. Keep reusable logic in this package.
2. Keep scripts as thin orchestration layers.
3. Keep notebooks inspection-only.
4. Add tests for every phase-level contract.
5. Do not introduce lookahead leakage.
6. Do not place live-order logic outside explicit broker safety guards.
7. Do not read private credentials directly in research modules.

## Phase Boundary

This package spans the full project pipeline, but each module must respect its phase contract.

Do not add downstream logic to upstream modules. For example:

- `data/` should not compute realised variance.
- `features/` should not train regimes.
- `regimes/` should not run backtests.
- `backtest/` should not place broker orders.
- `broker/` should not become a research data source.
