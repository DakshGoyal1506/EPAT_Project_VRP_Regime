# Phase 11 Artifacts - IBKR Paper-Signal Readiness Layer

All Phase 11 runtime artifacts are local-only by default.

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / keys | Review substitute | Sensitivity / reproducibility notes |
|---|---|---|---:|---|---|---|---|
| Daily paper signal | `reports/tables/phase_11/daily_paper_signal.csv` | `python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json` | No | Runtime signal artifact | `market`, `strategy_name`, `target_trade_date`, `target_exposure`, `recommended_action`, `final_status`, `live_order_sent` | Redacted terminal JSON and one-row preview | Must keep `live_order_sent=false` |
| Paper order intents | `reports/tables/phase_11/paper_order_intents.csv` | Same | No | Runtime paper-intent artifact | `symbol`, `side`, `paper_target_notional`, `paper_quantity`, `final_status`, `live_order_sent` | Redacted terminal JSON and header/one-row preview | Not a broker order |
| Risk check report | `reports/tables/phase_11/risk_check_report.csv` | Same | No | Runtime risk audit | `check_name`, `status`, `blocks_intent`, `reason`, `final_status` | CSV preview | Header-only when no intent is written |
| Broker metadata | `reports/tables/phase_11/broker_metadata.json` | Same | No | Broker dependency/status metadata | `ibridgepy_available`, `broker_connection_status`, `broker_data_status`, `live_order_sent` | Redacted JSON preview | Must not expose account identifiers |
| Run metadata | `reports/tables/phase_11/run_metadata.json` | Same | No | Run audit summary | `config_hash`, `daily_signal_final_status`, `paper_intent_final_status`, `final_status`, `live_order_sent` | Redacted JSON preview | Good review substitute |
| Config snapshot | `reports/tables/phase_11/ibkr_paper_config_snapshot.yaml` | Same | No | Effective runtime config snapshot | Phase 11 config keys | Redacted YAML preview | Account value must be redacted if environment override is used |
| Integration report | `reports/tables/phase_11/phase11_integration_report.json` | `python scripts/validate_phase11.py --print-json` | No | Artifact consistency report | `passed`, `violations`, `artifacts_checked` | JSON preview | Expected `passed=true`, `violations=[]` |
| Live-order guard report | `reports/tables/phase_11/live_order_guard_report.json` | `python scripts/validate_phase11.py --print-json` | No | Source safety scan output | `passed`, `violations`, `scanned_paths` | JSON preview | Expected `violations=[]` |
| Paper position state | `reports/tables/phase_11/paper_position_state.csv` | Optional state flow only | No | Local state file | `market`, `strategy_name`, `symbol`, `target_exposure`, `paper_quantity`, `status` | Redacted preview only | Local state; do not commit |

## Commit Policy

Commit only:

```text
source code
configs
tests
documentation
README files
.gitkeep placeholders
```

Do not commit:

```text
reports/tables/phase_11/*.csv
reports/tables/phase_11/*.json
reports/tables/phase_11/*.yaml
data/broker_cache/*
logs/*
broker account identifiers
paper account identifiers
```

## Expected Runtime States

Current-date stale signal:

```text
daily_signal_final_status = BLOCKED_STALE_SIGNAL
paper_intent_written = false
paper_intent_final_status = null
final_status = BLOCKED_STALE_SIGNAL
live_order_sent = false
```

Historical non-stale signal:

```text
daily_signal_final_status = BLOCKED_BY_KILL_SWITCH
paper_intent_written = true
paper_intent_final_status = BLOCKED_BY_KILL_SWITCH
final_status = BLOCKED_BY_KILL_SWITCH
live_order_sent = false
```

Historical quote path:

```text
daily_signal_final_status = BLOCKED_BY_KILL_SWITCH
paper_intent_written = true
paper_intent_final_status = BLOCKED_RISK_LIMIT
final_status = BLOCKED_RISK_LIMIT
live_order_sent = false
```

## Validation Commands

```bash
python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json
```

```bash
python scripts/validate_phase11.py --print-json
```

```bash
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```

Expected allowed match:

```text
.env.example
```
