# Broker Module

## Purpose

`src/vrp/broker/` owns the Phase 11 IBKR paper-signal readiness layer.

It converts validated Phase 9 strategy signals into audited paper-signal and paper-intent artifacts. It does not submit broker orders.

## Phase Ownership

```text
Phase 11 - IBKR paper-signal readiness layer
```

## Responsibilities

This module is responsible for:

- Loading and validating `configs/ibkr_paper.yaml`.
- Validating Phase 9 signal schema.
- Selecting the latest valid signal by `target_trade_date`.
- Detecting stale or missing signals.
- Publishing `daily_paper_signal.csv`.
- Selecting internal paper proxy contracts.
- Computing paper notional and optional paper quantity.
- Running Phase 11 safety/risk checks.
- Writing `paper_order_intents.csv`.
- Writing risk-check reports.
- Providing optional iBridgePy dependency metadata.
- Scanning source files for forbidden live-order execution patterns.
- Validating Phase 11 runtime artifacts.

## Main Modules

```text
__init__.py
broker_config.py
contracts.py
ibridgepy_adapter.py
live_order_guard.py
market_data.py
paper_sizing.py
paper_state.py
paper_trader.py
phase11_integration_checks.py
risk_checks.py
signal_publisher.py
signal_schema.py
```

## Expected Inputs

```text
configs/ibkr_paper.yaml
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
optional manual quote fields from CLI
```

## Expected Outputs

```text
reports/tables/phase_11/daily_paper_signal.csv
reports/tables/phase_11/paper_order_intents.csv
reports/tables/phase_11/risk_check_report.csv
reports/tables/phase_11/broker_metadata.json
reports/tables/phase_11/run_metadata.json
reports/tables/phase_11/ibkr_paper_config_snapshot.yaml
reports/tables/phase_11/phase11_integration_report.json
reports/tables/phase_11/live_order_guard_report.json
```

These outputs are local-only by default.

## Commands

Run Phase 11 readiness:

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json
```

Run final validator:

```bash
python scripts/validate_phase11.py --print-json
```

Run source guard only:

```bash
python scripts/validate_phase11.py --skip-artifacts --print-json
```

Run artifact integration checks only:

```bash
python scripts/validate_phase11.py --skip-source-guard --print-json
```

## Tests

```bash
pytest tests/test_broker_config.py tests/test_signal_schema.py tests/test_ibkr_contracts.py tests/test_ibridgepy_adapter.py tests/test_market_data.py tests/test_signal_publisher.py tests/test_paper_sizing.py tests/test_risk_checks.py tests/test_paper_trader.py tests/test_broker_diagnostics.py tests/test_run_ibkr_paper_signal_cli.py tests/test_live_order_guard.py tests/test_phase11_integration_checks.py tests/test_validate_phase11_cli.py
```

## Safety Boundaries

This module must keep:

```text
paper_only = true
live_orders_enabled = false
allow_order_placement = false
live_order_sent = false
```

## This Module Must Not

- Submit broker orders.
- Expose `placeOrder`, `submit_order`, `buy`, or `sell` execution functions.
- Treat paper intents as executed trades.
- Use Phase 10 performance metrics for sizing.
- Trade options or futures.
- Use stale or missing signals to create paper intents.
- Store broker account identifiers in tracked files.
