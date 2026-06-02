# Phase 8 Artifacts - Python-only MSVOL Robustness Appendix

## Policy

Phase 8 generated artifacts are local-only by default.

Commit:

```text
code
configs
scripts
tests
docs
README files
.gitkeep placeholders
```

Do not commit by default:

```text
data/interim/msgarch/*
data/interim/msvol/*
data/processed/*_msvol_regimes.parquet
reports/tables/phase_8/*
reports/figures/phase_8/*
```

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute | Notes |
|---|---|---|---:|---|---|---|---|
| US MSVOL input CSV | `data/interim/msgarch/us_msgarch_input.csv` | `python scripts/export_msgarch_inputs.py --market US` | No | Generated return-only model input | `date`, `market`, `log_return`, `return_for_msgarch`, `source_return_column`, `input_available` | `reports/tables/phase_8/msgarch_input_summary.csv` preview | Legacy folder name from original MSGARCH plan |
| India MSVOL input CSV | `data/interim/msgarch/india_msgarch_input.csv` | `python scripts/export_msgarch_inputs.py --market INDIA` | No | Generated return-only model input | Same as US | input summary preview | Legacy folder name from original MSGARCH plan |
| US raw MSVOL output | `data/interim/msvol/us_msvol_raw_output.csv` | `python scripts/run_msvol_regimes.py --market US` | No | Generated model probabilities | `date`, `market`, raw filtered probabilities, state variance estimates, conditional variance/volatility | terminal output and model summary JSON preview | Python MSVOL, not true MSGARCH |
| India raw MSVOL output | `data/interim/msvol/india_msvol_raw_output.csv` | `python scripts/run_msvol_regimes.py --market INDIA` | No | Generated model probabilities | Same as US | terminal output and model summary JSON preview | Python MSVOL, not true MSGARCH |
| US processed MSVOL regime panel | `data/processed/us_msvol_regimes.parquet` | `python scripts/import_msvol_outputs.py --market US` | No | Generated processed regime panel | `date`, `market`, `msvol_signal_trade_date`, state/probability fields | schema/head/tail preview and probability audit | Local-only parquet |
| India processed MSVOL regime panel | `data/processed/india_msvol_regimes.parquet` | `python scripts/import_msvol_outputs.py --market INDIA` | No | Generated processed regime panel | Same as US | schema/head/tail preview and probability audit | Local-only parquet |
| US MSVOL metadata | `reports/tables/phase_8/us/msvol_metadata.json` | `python scripts/import_msvol_outputs.py --market US` | No by default | Generated metadata | model validity, state mapping, hashes, status | selected JSON preview | May be selected later only if final-report-ready |
| India MSVOL metadata | `reports/tables/phase_8/india/msvol_metadata.json` | `python scripts/import_msvol_outputs.py --market INDIA` | No by default | Generated metadata | same as US | selected JSON preview | May be selected later only if final-report-ready |
| US probability audit | `reports/tables/phase_8/us/msvol_probability_audit.csv` | `python scripts/import_msvol_outputs.py --market US` | No by default | Generated probability audit | probability sums, min/max probabilities, state mapping | CSV preview | Useful review substitute |
| India probability audit | `reports/tables/phase_8/india/msvol_probability_audit.csv` | `python scripts/import_msvol_outputs.py --market INDIA` | No by default | Generated probability audit | same as US | CSV preview | Useful review substitute |
| US comparison summary | `reports/tables/phase_8/us/msvol_comparison_summary.csv` | `python scripts/run_msvol_diagnostics.py --market US` | No by default | Generated diagnostic comparison | agreement/correlation/overlap with threshold/HMM/MAR where available | CSV preview | Diagnostic-only |
| India comparison summary | `reports/tables/phase_8/india/msvol_comparison_summary.csv` | `python scripts/run_msvol_diagnostics.py --market INDIA` | No by default | Generated diagnostic comparison | same as US | CSV preview | Diagnostic-only |
| US state duration summary | `reports/tables/phase_8/us/msvol_state_duration_summary.csv` | `python scripts/run_msvol_diagnostics.py --market US` | No by default | Generated duration diagnostics | `state_name`, `n_runs`, `total_days`, duration stats | CSV preview | Diagnostic-only |
| India state duration summary | `reports/tables/phase_8/india/msvol_state_duration_summary.csv` | `python scripts/run_msvol_diagnostics.py --market INDIA` | No by default | Generated duration diagnostics | same as US | CSV preview | Diagnostic-only |
| Combined model comparison appendix | `reports/tables/phase_8/msvol_model_comparison_appendix.csv` | `python scripts/run_msvol_diagnostics.py --market ALL` | No by default | Generated appendix table | one row per market | CSV preview | Diagnostic-only |
| US no-lookahead audit | `reports/tables/phase_8/us/msvol_no_lookahead_audit.csv` | `python scripts/run_msvol_no_lookahead_audit.py --market US` | No by default | Generated timing/safety audit | check name, passed, severity, detail | CSV preview; zero failed error checks | Critical review substitute |
| India no-lookahead audit | `reports/tables/phase_8/india/msvol_no_lookahead_audit.csv` | `python scripts/run_msvol_no_lookahead_audit.py --market INDIA` | No by default | Generated timing/safety audit | same as US | CSV preview; zero failed error checks | Critical review substitute |
| Combined no-lookahead audit | `reports/tables/phase_8/msvol_no_lookahead_audit.csv` | `python scripts/run_msvol_no_lookahead_audit.py --market ALL` | No by default | Generated combined safety audit | market/check rows | CSV preview; zero failed error checks | Critical review substitute |
| Phase 8 figures | `reports/figures/phase_8/*` | Future diagnostics if added | No by default | Generated figures | image files | screenshots only | No required Phase 8 figure currently |

## Sensitivity and Reproducibility Notes

```text
MSVOL artifacts are generated from public-market-data-derived panels.
They are not broker artifacts.
They may still be large and should stay local.
No broker account identifiers are expected.
No strategy positions, exposure, PnL, or order data belong in Phase 8 artifacts.
```

## Review Substitute Packet

Use terminal or small previews instead of committing artifacts:

```text
pytest output
python scripts/run_msvol_no_lookahead_audit.py --market ALL output
head/tail of data/processed/*_msvol_regimes.parquet
reports/tables/phase_8/*/msvol_probability_audit.csv preview
reports/tables/phase_8/*/msvol_comparison_summary.csv preview
reports/tables/phase_8/*/msvol_no_lookahead_audit.csv preview
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```
