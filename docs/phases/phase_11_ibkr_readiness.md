# Phase 11 - IBKR Paper-Signal Readiness Layer

## Status

Complete / frozen.

Phase 11 has been implemented and validated. The final validator returned `passed=true`, `source_guard_passed=true`, `integration_passed=true`, and `live_order_sent=false`. The full project test suite passed locally.

## Objective

Phase 11 converts validated Phase 9 strategy signals into audited paper-signal and paper-intent artifacts.

It is a readiness layer. It does not submit broker orders.

## Phase Boundary

Phase 11 does:

- Load Phase 9 strategy signal files.
- Validate the signal schema.
- Select the latest valid signal by `target_trade_date`.
- Detect stale or missing signals.
- Publish `daily_paper_signal.csv`.
- Convert valid paper-signal recommendations into paper-intent artifacts.
- Route US short-vol paper intent to configured proxy instruments.
- Compute paper notional from configured notional only.
- Optionally compute paper quantity from a quote.
- Run safety and risk checks.
- Write broker metadata, run metadata, config snapshots, source-guard reports, and integration reports.
- Enforce `live_order_sent=false`.

Phase 11 does not:

- Connect to IBKR for execution.
- Submit orders to IBKR.
- Submit live orders.
- Call `placeOrder`.
- Trade options or futures.
- Infer capital or margin from Phase 10 backtests.
- Use Phase 10 performance for sizing.
- Use stale or missing signals to create paper intents.

## Files Owned by This Phase

### Config

```text
configs/ibkr_paper.yaml
```

### Scripts

```text
scripts/run_ibkr_paper_signal.py
scripts/validate_phase11.py
```

### Source

```text
src/vrp/broker/__init__.py
src/vrp/broker/broker_config.py
src/vrp/broker/contracts.py
src/vrp/broker/ibridgepy_adapter.py
src/vrp/broker/live_order_guard.py
src/vrp/broker/market_data.py
src/vrp/broker/paper_sizing.py
src/vrp/broker/paper_state.py
src/vrp/broker/paper_trader.py
src/vrp/broker/phase11_integration_checks.py
src/vrp/broker/risk_checks.py
src/vrp/broker/signal_publisher.py
src/vrp/broker/signal_schema.py
src/vrp/reports/broker_diagnostics.py
```

### Tests

```text
tests/test_broker_config.py
tests/test_ibkr_contracts.py
tests/test_ibridgepy_adapter.py
tests/test_live_order_guard.py
tests/test_market_data.py
tests/test_paper_sizing.py
tests/test_paper_trader.py
tests/test_phase11_integration_checks.py
tests/test_risk_checks.py
tests/test_run_ibkr_paper_signal_cli.py
tests/test_signal_publisher.py
tests/test_signal_schema.py
tests/test_validate_phase11_cli.py
tests/test_broker_diagnostics.py
```

## Main Functions / Classes / Scripts

```text
load_broker_config
validate_broker_config
Phase9SignalSchema
validate_signal_schema
select_latest_signal
DailyPaperSignal
publish_daily_paper_signal
ContractSpec
BrokerInstrumentRegistry
IBridgePyAdapter
QuoteSnapshot
quote_from_mapping
PaperSizingResult
build_paper_sizing
RiskCheckSummary
run_phase11_risk_checks
PaperOrderIntent
publish_paper_order_intent
assert_no_live_order_code
assert_phase11_artifacts_valid
write_phase11_diagnostics
scripts/run_ibkr_paper_signal.py
scripts/validate_phase11.py
```

## Config Files Used

```text
configs/ibkr_paper.yaml
```

Important defaults:

```yaml
paper_only: true
kill_switch: true
live_orders_enabled: false
allow_order_placement: false
```

## Input Files

Expected local inputs:

```text
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
```

The Phase 9 signal file must include at least:

```text
strategy_name
target_trade_date
target_exposure
strategy_available
blocked_reason
decision_reason
```

Unavailable terminal rows are allowed when:

```text
strategy_available = false
target_trade_date = NaT
target_exposure = NaN
```

Those rows are not selectable for Phase 11 paper-intent decisions.

## Generated Output Files

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

Optional state file:

```text
reports/tables/phase_11/paper_position_state.csv
```

## Commit vs Local-Only

Commit:

```text
source code
config
tests
docs
README files
```

Keep local by default:

```text
reports/tables/phase_11/*.csv
reports/tables/phase_11/*.json
reports/tables/phase_11/*.yaml
data/broker_cache/*
logs/*
paper_position_state.csv
```

## Commands to Regenerate Outputs

Current readiness run:

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json
```

Historical deterministic run:

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --signal-path data/processed/us_strategy_signals.parquet --as-of-date 2026-04-15 --print-json
```

Historical deterministic run with manual VXX quote:

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --signal-path data/processed/us_strategy_signals.parquet --as-of-date 2026-04-15 --quote-symbol VXX --quote-bid 14.95 --quote-ask 15.05 --quote-age-seconds 30 --print-json
```

Final validation:

```bash
python scripts/validate_phase11.py --print-json
```

## Tests to Run

Phase-specific tests:

```bash
pytest tests/test_broker_config.py tests/test_signal_schema.py tests/test_ibkr_contracts.py tests/test_ibridgepy_adapter.py tests/test_market_data.py tests/test_signal_publisher.py tests/test_paper_sizing.py tests/test_risk_checks.py tests/test_paper_trader.py tests/test_broker_diagnostics.py tests/test_run_ibkr_paper_signal_cli.py tests/test_live_order_guard.py tests/test_phase11_integration_checks.py tests/test_validate_phase11_cli.py
```

Full test suite:

```bash
pytest
```

## Validation Checklist

- `python scripts/validate_phase11.py --print-json` returns `passed=true`.
- `source_guard_passed=true`.
- `integration_passed=true`.
- `live_order_sent=false`.
- Current-date stale run writes no paper intent.
- Historical fresh run reaches kill-switch block.
- Historical quote run may block on risk limits before kill switch.
- No generated broker artifacts are committed.

## No-Lookahead / Safety Rules

- Do not use future realised variance as a signal.
- Do not use future returns as features.
- Do not use full-sample smoothed HMM probabilities as tradable probabilities.
- Do not use Phase 10 backtest returns for sizing.
- Do not infer option contracts from research proxy PnL.
- Do not create paper intents from stale or missing signals.
- Do not permit `live_order_sent=true`.

## Known Limitations

- iBridgePy is optional and may not be installed.
- No broker execution connection is attempted.
- Quotes are optional manual/inspection inputs.
- VXX/SVXY are paper proxies and not direct variance-swap exposure.
- India remains signal-only by default.
- Runtime broker artifacts are local-only and sensitive by default.

## Review Checklist

- Review `configs/ibkr_paper.yaml`.
- Review `scripts/run_ibkr_paper_signal.py`.
- Review `scripts/validate_phase11.py`.
- Review `src/vrp/broker/`.
- Review `src/vrp/reports/broker_diagnostics.py`.
- Run Phase 11 tests.
- Run `python scripts/validate_phase11.py --print-json`.
- Check that generated artifacts are untracked.
