# vrp

This is the canonical Python package for the EPAT VRP regime project.

Subpackages:

- `data/` - ingestion, schema, and validation helpers.
- `features/` - realised variance, implied variance, VRP, and feature registry logic.
- `forecasting/` - HAR-RV and related forecasting models.
- `regimes/` - threshold, HMM, and autoregressive regime models.
- `strategies/` - regime-conditioned strategy logic.
- `backtest/` - vectorised backtest and accounting utilities.
- `broker/` - paper-signal and broker boundary code.
- `reports/` - reporting helpers and output assembly.

Project policy, phase status, and artifact rules live in `docs/`.
