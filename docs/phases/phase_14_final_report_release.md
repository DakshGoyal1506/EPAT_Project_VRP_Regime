# Phase 14 — Final Report, Presentation Package, and Release Cleanup

## Status

`in progress`

Phase 14 converts the completed EPAT VRP regime project into a submission-ready release package.

This is a documentation, reporting, reproducibility, and release-cleanup phase. It does not add new research logic, model logic, strategy rules, broker execution, or backtests.

---

## Objective

Create a professional final EPAT submission package containing:

```text
final report source
PDF final report
executive summary
presentation outline
selected artifact inventory
table inventory
figure inventory
claims audit
limitations
future work
reproducibility note
release checklist
submission package map
```

Phase 14 is a release package phase, not only a report-writing phase.

---

## Inputs

Phase 14 depends on frozen outputs and documentation from:

| Phase | Input role                                           |
| ----: | ---------------------------------------------------- |
|     0 | Repo scaffold, governance, generated artifact policy |
|     1 | Data ingestion and audit structure                   |
|     2 | Realised variance construction                       |
|     3 | Implied variance and VRP construction                |
|     4 | HAR-RV forecasting and HAR-based VRP                 |
|     5 | Threshold regimes                                    |
|     6 | Gaussian HMM regimes                                 |
|     7 | Markov autoregression                                |
|     8 | MSVOL robustness appendix                            |
|     9 | Strategy signal construction                         |
|    10 | Vectorised research backtest and robustness          |
|    11 | IBKR paper-signal readiness                          |
|    13 | Cross-market US-India analysis                       |

Phase 12 is not an input because it was skipped.

Correct Phase 12 wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

---

## Non-Scope

Do not do any of the following in Phase 14:

```text
implement new research logic
change model logic
change strategy rules
rerun or alter previous phase outputs unless required for documentation/reproducibility
tune results
add new backtests
add new strategy variants
add broker execution
claim live-trading profitability
claim true option-chain PnL
claim account returns
claim causal US-to-India transmission
claim MSVOL is true MSGARCH
claim Phase 11 sent broker orders
imply Phase 12 was implemented
commit full generated panels
commit broker-sensitive artifacts
```

---

## Required Report Deliverables

```text
reports/final/README.md
reports/final/final_report.md
reports/final/final_report.pdf
reports/final/executive_summary.md
reports/final/presentation_outline.md
reports/final/selected_artifacts.md
reports/final/limitations.md
reports/final/reproducibility_note.md
reports/final/future_work.md
reports/final/result_claims_audit.md
reports/final/table_inventory.md
reports/final/figure_inventory.md
```

---

## Required Documentation Deliverables

```text
docs/phases/phase_14_final_report_release.md
docs/artifacts/phase_14_artifacts.md
docs/release_checklist.md
docs/final_report_checklist.md
docs/submission_package.md
```

---

## Required Updates to Existing Files

These are updated after the new Phase 14 files are created:

```text
README.md
docs/phase_status.md
docs/artifact_inventory.md
docs/commands.md
docs/known_limitations.md
docs/phases/README.md
docs/artifacts/README.md
reports/README.md
reports/tables/README.md
reports/figures/README.md
.gitignore
```

---

## Report Source and PDF Rule

Markdown is the source of truth:

```text
reports/final/final_report.md
```

PDF is an export deliverable:

```text
reports/final/final_report.pdf
```

Do not manually diverge the PDF from the Markdown source.

Generate the PDF only after:

```text
result_claims_audit.md is complete
table_inventory.md is complete
figure_inventory.md is complete
selected_artifacts.md is complete
limitations.md is complete
future_work.md is complete
reproducibility_note.md is complete
final_report_checklist.md is complete
release_checklist.md is complete
```

---

## Claims Audit Rule

`reports/final/result_claims_audit.md` is mandatory.

Each major report claim must map to:

```text
claim
evidence file
evidence column / metric
allowed wording
forbidden overclaim
report section
```

No major conclusion should appear in the report, PDF, executive summary, or presentation outline unless it is represented in the claims audit.

Numeric findings must remain placeholders until the local result table has been inspected.

Placeholder format:

```text
[INSERT VALUE FROM reports/tables/<path>: metric_name]
```

---

## Table and Figure Inventory Rule

Every table used in the final report or PDF must appear in:

```text
reports/final/table_inventory.md
```

Every figure used in the final report or PDF must appear in:

```text
reports/final/figure_inventory.md
```

Every selected artifact used as evidence must appear in:

```text
reports/final/selected_artifacts.md
```

---

## Artifact Policy

Commit by default:

```text
reports/final/*.md
reports/final/final_report.pdf
docs/phases/phase_14_final_report_release.md
docs/artifacts/phase_14_artifacts.md
docs/release_checklist.md
docs/final_report_checklist.md
docs/submission_package.md
updated governance docs
```

Local-only by default:

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

Selected small tables or figures may be committed only if explicitly approved as stable, reproducible, non-sensitive final-report artifacts.

---

## Required Caveats

The final report must clearly state:

1. VIX and India VIX are implied-volatility proxies, not variance swap quotes.
2. Daily OHLC realised variance estimators are proxies for true realised variance.
3. A 22-trading-day horizon approximates one month but is not the same as 30 calendar days.
4. Forward ex-post VRP labels are outcomes only, not tradable features.
5. HAR forecasts are model-dependent and trained under point-in-time constraints.
6. Gaussian HMM does not directly model observed autocorrelation.
7. Markov autoregression addresses observed-series autocorrelation but remains reduced-form.
8. MSVOL is Python-only Markov-switching volatility robustness, not true R MSGARCH.
9. Strategy outputs are exposure intentions.
10. Phase 10 returns are research-layer VRP proxy units, not executable account returns.
11. Overlapping 22-day outcome labels make annualised metrics approximate.
12. Phase 11 did not place broker orders.
13. Phase 12 paper execution was skipped / future optional.
14. Phase 13 cross-market lead-lag diagnostics are predictive/statistical diagnostics, not causal proof.
15. Generated data panels remain local and are not committed to GitHub.

---

## Terminology Lock

The final report must use these terms precisely:

| Term                               | Meaning                                             | Not meant as                             |
| ---------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| Research-proxy return              | Additive VRP proxy backtest unit                    | Account return                           |
| Exposure intention                 | Signal target from strategy layer                   | Broker order                             |
| Filtered probability               | Time-t available regime probability                 | Full-sample smoothed probability         |
| Cross-market predictive diagnostic | Statistical lead-lag or predictive association test | Causal proof                             |
| MSVOL robustness                   | Python Markov-switching volatility check            | True R MSGARCH                           |
| Paper-signal readiness             | Guarded signal-format and risk-check demonstration  | Broker execution or paper trading result |

---

## Presentation Package Rule

The presentation outline must be evidence-first.

Each slide must include:

```text
slide title
3-5 bullets
suggested figure/table
source artifact path
speaker note
```

The presentation must not contain unaudited numeric claims.

---

## PDF Workflow

Recommended workflow:

```text
Phase 14A:
    final_report.md
    executive_summary.md
    result_claims_audit.md
    table_inventory.md
    figure_inventory.md
    selected_artifacts.md

Phase 14B:
    insert selected audited tables and figures into final_report.md

Phase 14C:
    export final_report.md to final_report.pdf

Phase 14D:
    visually inspect PDF
    fix broken tables, clipped figures, and bad page breaks

Phase 14E:
    run final release checklist
```

If Markdown-to-PDF layout is insufficient:

```text
final_report.md
    ↓
DOCX polish
    ↓
PDF export
```

No separate Phase 15 is required.

---

## Validation Commands

Final package validation:

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

---

## Acceptance Criteria

Phase 14 is complete only when:

* [ ] `reports/final/final_report.md` exists.
* [ ] `reports/final/final_report.pdf` exists.
* [ ] `reports/final/executive_summary.md` exists.
* [ ] `reports/final/presentation_outline.md` exists.
* [ ] `reports/final/result_claims_audit.md` exists.
* [ ] `reports/final/table_inventory.md` exists.
* [ ] `reports/final/figure_inventory.md` exists.
* [ ] `reports/final/selected_artifacts.md` exists.
* [ ] `reports/final/limitations.md` exists.
* [ ] `reports/final/reproducibility_note.md` exists.
* [ ] `reports/final/future_work.md` exists.
* [ ] `docs/release_checklist.md` exists.
* [ ] `docs/final_report_checklist.md` exists.
* [ ] `docs/submission_package.md` exists.
* [ ] Phase 12 is labelled `skipped / future optional`.
* [ ] Phase 14 is labelled `complete / frozen` after final validation.
* [ ] No heavy generated artifacts are tracked.
* [ ] No credentials are tracked.
* [ ] No broker-sensitive artifacts are tracked.
* [ ] All numeric claims are audited.
* [ ] PDF has been visually inspected.
* [ ] Final validation commands pass or failures are documented.

---

## Review Packet

Send for review:

```text
git status --short
git diff --check output
pytest output
Phase 11 validation output
Phase 13 validation output
reports/final/*.md
docs/release_checklist.md
docs/final_report_checklist.md
docs/submission_package.md
selected local CSV/JSON previews only when needed
selected figure screenshots only when needed
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
