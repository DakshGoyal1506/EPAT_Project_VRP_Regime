# Phase 11 Runbook

## Purpose

Phase 11 runs the IBKR paper-signal readiness layer.

It produces paper-signal and paper-intent artifacts. It does not submit broker orders.

## Main Run

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json
```

Expected current-date stale state when Phase 9 signal is older than freshness limit:

```text
daily_signal_final_status = BLOCKED_STALE_SIGNAL
paper_intent_written = false
paper_intent_final_status = null
final_status = BLOCKED_STALE_SIGNAL
live_order_sent = false
```

## Historical Deterministic Run

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --signal-path data/processed/us_strategy_signals.parquet --as-of-date 2026-04-15 --print-json
```

Expected:

```text
daily_signal_final_status = BLOCKED_BY_KILL_SWITCH
paper_intent_written = true
paper_intent_final_status = BLOCKED_BY_KILL_SWITCH
final_status = BLOCKED_BY_KILL_SWITCH
live_order_sent = false
```

## Historical Quote Run

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --signal-path data/processed/us_strategy_signals.parquet --as-of-date 2026-04-15 --quote-symbol VXX --quote-bid 14.95 --quote-ask 15.05 --quote-age-seconds 30 --print-json
```

Expected when `max_shares=100`:

```text
paper_intent_final_status = BLOCKED_RISK_LIMIT
final_status = BLOCKED_RISK_LIMIT
live_order_sent = false
```

## Final Validator

```bash
python scripts/validate_phase11.py --print-json
```

Expected:

```text
passed = true
source_guard_passed = true
integration_passed = true
live_order_sent = false
```

## Output Files

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

## Troubleshooting

`BLOCKED_MISSING_SIGNAL` means Phase 9 signal file is missing or schema-invalid.

`BLOCKED_STALE_SIGNAL` means latest valid `target_trade_date` is older than the configured freshness limit.

`BLOCKED_BY_KILL_SWITCH` is expected under default config.

`BLOCKED_RISK_LIMIT` means risk checks blocked before the kill switch.

`IBRIDGEPY_NOT_INSTALLED` is allowed. iBridgePy is optional in Phase 11.

## Safety Rules

- Do not submit broker orders.
- Keep `live_order_sent=false`.
- Do not use Phase 10 performance for sizing.
- Do not commit runtime broker artifacts.
