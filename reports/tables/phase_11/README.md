# Phase 11 Runtime Tables

This folder is for local Phase 11 runtime artifacts.

These files are generated and should not be committed by default.

## Local Artifacts

```text
daily_paper_signal.csv
paper_order_intents.csv
risk_check_report.csv
broker_metadata.json
run_metadata.json
ibkr_paper_config_snapshot.yaml
phase11_integration_report.json
live_order_guard_report.json
paper_position_state.csv
```

## Commit Policy

Commit:

```text
README.md
.gitkeep if needed
```

Do not commit by default:

```text
*.csv
*.json
*.yaml
```

## Reason

Phase 11 artifacts may contain runtime timing, broker configuration context, local file paths, local quote snapshots, or paper-account setup details.

Use redacted terminal output or selected schema previews for review instead.

## Validation

Run:

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
