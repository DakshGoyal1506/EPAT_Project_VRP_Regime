# Phase 14 Artifacts — Final Report and Release Package

## Purpose

This file documents Phase 14 artifacts, their commit policy, and their review substitutes.

Phase 14 creates the final EPAT submission package. It is a release and reporting phase, not a research-logic phase.

---

## Artifact Policy Summary

| Artifact type | Commit? | Reason |
|---|---:|---|
| Final report Markdown | Yes | Source of truth for report |
| Final report PDF | Yes | Submission deliverable |
| Executive summary | Yes | Reviewer-facing summary |
| Presentation outline | Yes | Submission/presentation package |
| Claims audit | Yes | Prevents overclaiming |
| Table inventory | Yes | Evidence governance |
| Figure inventory | Yes | Evidence governance |
| Selected artifacts inventory | Yes | Artifact governance |
| Final limitations | Yes | Report caveats |
| Reproducibility note | Yes | Reviewer workflow |
| Future work | Yes | Scope boundary |
| Release checklist | Yes | Freeze governance |
| Submission package map | Yes | Reviewer routing |
| Raw data | No | Generated/source-refreshable |
| Processed panels | No | Generated/local |
| Model binaries | No | Generated/local |
| Full backtest panels | No | Generated/local |
| Full signal panels | No | Generated/local |
| Full cross-market panels | No | Generated/local |
| Broker runtime artifacts | No | Sensitive/local |
| Selected small summary CSV/JSON | Optional | Only if approved, stable, reproducible, and non-sensitive |
| Selected final PNG figures | Optional | Only if approved and used in final report |

---

## Required Phase 14 Artifacts

| Artifact | Path | Producer | Commit? | Review substitute |
|---|---|---|---:|---|
| Final package README | `reports/final/README.md` | Manual Phase 14 | Yes | File review |
| Final report source | `reports/final/final_report.md` | Manual Phase 14 | Yes | File review |
| Final report PDF | `reports/final/final_report.pdf` | Export from `final_report.md` | Yes | Visual PDF review |
| Executive summary | `reports/final/executive_summary.md` | Manual Phase 14 | Yes | File review |
| Presentation outline | `reports/final/presentation_outline.md` | Manual Phase 14 | Yes | File review |
| Selected artifacts inventory | `reports/final/selected_artifacts.md` | Manual Phase 14 | Yes | File review |
| Final limitations | `reports/final/limitations.md` | Curated from `docs/known_limitations.md` | Yes | File review |
| Reproducibility note | `reports/final/reproducibility_note.md` | Curated from `docs/reproducibility.md` | Yes | File review |
| Future work | `reports/final/future_work.md` | Manual Phase 14 | Yes | File review |
| Result claims audit | `reports/final/result_claims_audit.md` | Manual Phase 14 | Yes | File review |
| Table inventory | `reports/final/table_inventory.md` | Manual Phase 14 | Yes | File review |
| Figure inventory | `reports/final/figure_inventory.md` | Manual Phase 14 | Yes | File review |
| Phase 14 phase doc | `docs/phases/phase_14_final_report_release.md` | Manual Phase 14 | Yes | File review |
| Phase 14 artifact doc | `docs/artifacts/phase_14_artifacts.md` | Manual Phase 14 | Yes | File review |
| Release checklist | `docs/release_checklist.md` | Manual Phase 14 | Yes | File review |
| Final report checklist | `docs/final_report_checklist.md` | Manual Phase 14 | Yes | File review |
| Submission package map | `docs/submission_package.md` | Manual Phase 14 | Yes | File review |

---

## Local-Only Evidence Artifacts

The final report may reference these local artifacts, but they remain local by default:

```text
reports/tables/data_audit.csv
reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/har_forecast_accuracy.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_no_lookahead_audit.csv
reports/tables/threshold_regime_summary.csv
reports/tables/threshold_vrp_by_state.csv
reports/tables/threshold_no_lookahead_audit.csv
reports/tables/phase_6/**/*
reports/tables/phase_7/**/*
reports/tables/phase_8/**/*
reports/tables/phase_9/**/*
reports/tables/phase_10/**/*
reports/tables/phase_11/**/*
reports/tables/phase_13/**/*
reports/figures/**/*.png
```

Do not commit these unless explicitly approved as final-report-selected artifacts.

---

## Never Commit

```text
data/raw/*
data/interim/*
data/processed/*
data/manual/*
data/broker_cache/*
models/*
logs/*
*.parquet
*.pkl
*.pickle
*.joblib
*.pt
*.pth
.env
.env.*
broker logs
TWS logs
paper order logs
private account identifiers
full generated panels
```

---

## Optional Final-Report Artifact Exception

A small generated artifact may be committed only if all conditions hold:

1. It is required for final report review.
2. It is small.
3. It is non-sensitive.
4. It is stable and reproducible.
5. Its producer command is documented.
6. It does not expose broker account, order, or private runtime data.
7. It appears in `reports/final/selected_artifacts.md`.
8. It appears in either `reports/final/table_inventory.md` or `reports/final/figure_inventory.md`.

Recommended optional path if selected figures are committed:

```text
reports/final/figures/
```

Do not commit entire phase figure directories.

---

## PDF Artifact Rule

`reports/final/final_report.pdf` is a required Phase 14 deliverable.

It must be generated from:

```text
reports/final/final_report.md
```

It must not include:

```text
full raw data
full generated panels
huge unformatted tables
unverified numeric claims
broker-sensitive artifacts
local-only paths without explanation
```

Every figure/table in the PDF must appear in:

```text
reports/final/figure_inventory.md
reports/final/table_inventory.md
reports/final/selected_artifacts.md
```

Every major conclusion in the PDF must appear in:

```text
reports/final/result_claims_audit.md
```

---

## Broker Artifact Rule

Broker artifacts are sensitive by default.

Keep local:

```text
reports/tables/phase_11/broker_metadata.json
reports/tables/phase_11/run_metadata.json
reports/tables/phase_11/ibkr_paper_config_snapshot.yaml
data/broker_cache/*
ibkr_logs/*
tws_logs/*
broker_logs/*
paper_order_logs/*
order_preview_logs/*
```

Only redacted summaries may be shared for review.

The final report may reference:

```text
reports/tables/phase_11/risk_check_report.csv
reports/tables/phase_11/phase11_integration_report.json
reports/tables/phase_11/live_order_guard_report.json
```

But these remain local unless explicitly redacted and approved.

---

## Phase 10 Artifact Wording

Phase 10 artifacts must be described as:

```text
research-proxy backtest diagnostics
proxy return units
additive proxy cumulative curves
proxy drawdowns
cost-sensitivity diagnostics
```

Do not describe Phase 10 artifacts as:

```text
account returns
option-chain PnL
live trading results
paper trading results
executable equity curves
```

---

## Phase 13 Artifact Wording

Phase 13 artifacts must be described as:

```text
same-date descriptive diagnostics
lagged-US predictive diagnostics
lead-lag association
analysis-only India overlay
```

Do not describe Phase 13 artifacts as:

```text
causal transmission proof
US-causes-India evidence
new deployed strategy
live trading evidence
```

---

## Review Packet

Recommended Phase 14 review packet:

```text
reports/final/README.md
reports/final/final_report.md
reports/final/executive_summary.md
reports/final/presentation_outline.md
reports/final/limitations.md
reports/final/reproducibility_note.md
reports/final/future_work.md
reports/final/result_claims_audit.md
reports/final/table_inventory.md
reports/final/figure_inventory.md
reports/final/selected_artifacts.md
docs/release_checklist.md
docs/final_report_checklist.md
docs/submission_package.md
git status --short output
git diff --check output
pytest output
Phase 11 validation output
Phase 13 validation output
```

Optional local previews:

```text
selected CSV previews
selected JSON previews
selected figure screenshots
```

Do not send:

```text
raw data folders
processed parquet files
broker cache
private account identifiers
model binaries
logs
```
