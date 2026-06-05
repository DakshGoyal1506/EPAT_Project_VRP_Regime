# Final Reproducibility Note

This note explains how the final report package should be reproduced, reviewed, and validated.

It complements:

```text
docs/reproducibility.md
docs/commands.md
docs/generated_artifact_policy.md
docs/artifact_inventory.md
reports/final/selected_artifacts.md
```

---

## 1. Reproducibility Boundary

The project is reproducible at two levels:

| Level             | Contents                                                                                                                 | Git-tracked? |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | -----------: |
| Repository design | Source code, configs, scripts, tests, docs, README files, final report files                                             |          Yes |
| Empirical outputs | Raw data, processed panels, model outputs, backtest panels, signal panels, cross-market panels, broker runtime artifacts |           No |

A fresh clone is sufficient to inspect the research design, phase boundaries, tests, commands, and final report package. Full empirical regeneration requires local data downloads and script execution.

---

## 2. Tracked Reproducible Layer

The tracked repository should include:

```text
source code under src/
configuration files under configs/
scripts under scripts/
tests under tests/
documentation under docs/
README files
.env.example
pyproject.toml
.gitignore
directory placeholders
reports/final/*.md
reports/final/final_report.pdf
```

The final report package is part of the tracked reproducible layer.

---

## 3. Local-Only Generated Layer

The following remain local-only:

```text
data/raw/*
data/interim/*
data/processed/*
data/manual/*
data/broker_cache/*
models/*
logs/*
reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
full backtest panels
full strategy signal panels
full regime probability panels
full cross-market panels
broker runtime artifacts
```

Small generated tables or selected figures may be committed only if they are explicitly approved as final-report artifacts and satisfy the generated-artifact policy.

---

## 4. Environment Setup

Run from the repository root:

```bash
pip install -e .
pip install -e ".[dev]"
```

Optional GPU support for HAR acceleration:

```bash
pip install -e ".[gpu]"
```

The project should remain understandable and testable without GPU support.

---

## 5. Minimal Reviewer Workflow

A reviewer can inspect the project without receiving generated data files.

Recommended review order:

```text
README.md
docs/submission_package.md
reports/final/executive_summary.md
reports/final/final_report.md
reports/final/result_claims_audit.md
reports/final/selected_artifacts.md
docs/reproducibility.md
docs/commands.md
docs/known_limitations.md
docs/release_checklist.md
```

Minimal validation:

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
```

---

## 6. Full Local Regeneration Outline

Full empirical regeneration requires data-source availability and local generated outputs.

General sequence:

```bash
python scripts/download_data.py --market ALL --source all --dry-run
python scripts/download_data.py --market US --source all --force
python scripts/download_data.py --market INDIA --source yahoo --force

python scripts/build_features.py --market ALL --feature rv --window 22
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp

python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none

python scripts/train_regimes.py --market ALL --model threshold --force
python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force

python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force

python scripts/export_msgarch_inputs.py --market ALL
python scripts/run_msvol_regimes.py --market ALL
python scripts/import_msvol_outputs.py --market ALL
python scripts/run_msvol_diagnostics.py --market ALL
python scripts/run_msvol_no_lookahead_audit.py --market ALL

python scripts/build_signals.py --market ALL --strategy all --force

python scripts/audit_phase10_inputs.py --market ALL
python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force
python scripts/generate_backtest_diagnostics.py --market ALL
python scripts/run_robustness.py --market ALL --test all --force
python scripts/audit_phase10_final.py --market ALL

python scripts/validate_phase11.py --print-json

python scripts/run_cross_market_analysis.py --validate-inputs-only
python scripts/run_cross_market_analysis.py --model ALL --force
```

If GPU support is unavailable, use the documented CPU fallback for HAR-RV.

---

## 7. Final Package Validation

Before release, run:

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
git ls-files | findstr /i "broker_cache data/raw data/interim data/processed"
pytest
python scripts/validate_phase11.py --print-json
python scripts/run_cross_market_analysis.py --validate-inputs-only
pytest tests/test_cross_market_alignment.py tests/test_cross_market_no_lookahead.py tests/test_cross_market_stats.py tests/test_cross_market_overlay.py tests/test_phase13_artifact_mutation.py tests/test_phase13_datetime_dtype.py
```

Expected policy result:

```text
no heavy generated artifacts tracked
no credentials tracked
no broker-sensitive artifacts tracked
no live-order evidence
no unaudited numeric report claims
```

---

## 8. Report Reproducibility

The Markdown report is the source of truth:

```text
reports/final/final_report.md
```

The PDF is an export:

```text
reports/final/final_report.pdf
```

PDF generation must happen only after:

```text
reports/final/result_claims_audit.md is complete
reports/final/table_inventory.md is complete
reports/final/figure_inventory.md is complete
reports/final/selected_artifacts.md is complete
reports/final/limitations.md is complete
reports/final/future_work.md is complete
```

Do not manually diverge the PDF from the Markdown source.

---

## 9. Claims Reproducibility

Every major claim must appear in:

```text
reports/final/result_claims_audit.md
```

Every numeric claim must point to:

```text
evidence file
evidence column / metric
report section
```

If a result table has not been inspected, leave the report placeholder intact:

```text
[INSERT VALUE FROM reports/tables/<path>: metric_name]
```

---

## 10. Table and Figure Reproducibility

Every table used in the final report or PDF must appear in:

```text
reports/final/table_inventory.md
```

Every figure used in the final report or PDF must appear in:

```text
reports/final/figure_inventory.md
```

Every selected artifact must appear in:

```text
reports/final/selected_artifacts.md
```

Large local outputs should be summarized, not pasted.

---

## 11. Broker Reproducibility Boundary

Phase 11 is a paper-signal readiness appendix only.

Broker-related review should use:

```text
reports/tables/phase_11/risk_check_report.csv
reports/tables/phase_11/phase11_integration_report.json
reports/tables/phase_11/live_order_guard_report.json
```

These files remain local unless a redacted summary is explicitly approved.

The report must state that no broker orders were placed.

---

## 12. Cross-Market Reproducibility Boundary

Phase 13 generated panels remain local.

Preferred review substitutes:

```text
reports/tables/phase_13/alignment_audit.csv
reports/tables/phase_13/no_lookahead_audit.csv
reports/tables/phase_13/logistic_model_comparison.csv
reports/tables/phase_13/lead_lag_table.csv
reports/tables/phase_13/india_overlay_summary.csv
```

Same-date diagnostics are descriptive only. Lagged-US diagnostics are predictive/statistical diagnostics only. The India overlay is analysis-only and outside the locked Phase 9 strategy universe.

---

## 13. Local Review Packet

When sending the project for review, include or paste:

```text
git status --short
git diff --check output
pytest output
Phase 11 validation output
Phase 13 validation output
selected CSV previews
selected JSON previews
screenshots or selected figures if needed
```

Do not send:

```text
raw data folders
processed parquet panels
broker cache
private account identifiers
full model binaries
logs
```
