# Tests

This folder contains the project test suite.

## Rules

1. Tests must not rely on live internet unless explicitly marked and separated.
2. Data-loader tests should use fixtures or mocks.
3. No-lookahead tests are required for forecasting, regimes, strategies, and backtests.
4. Broker tests must not place live orders.
5. Tests should verify phase contracts, not only function-level behaviour.

## Common Commands

Full suite:

```bash
pytest
```

Phase 1 ingestion/schema:

```bash
pytest tests/test_data_loaders.py tests/test_data_schema.py
```

Phase 2/3 feature and no-lookahead checks:

```bash
pytest tests/test_rv_estimators.py tests/test_vrp_alignment.py tests/test_no_lookahead.py
```

Regime and backtest checks:

```bash
pytest tests/test_hmm_model.py tests/test_hmm_filtering.py
pytest tests/test_backtest_accounting.py tests/test_backtest_metrics.py
```

Broker safety checks:

```bash
pytest tests/test_live_order_guard.py tests/test_risk_checks.py
```
