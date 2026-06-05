# Final Report Package

This directory contains the final EPAT submission package for:

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

## Package Contents

| File | Purpose | Status |
|---|---|---|
| `final_report.md` | Markdown source of truth for the final report | To be created in Phase 14 |
| `final_report.pdf` | PDF export generated from `final_report.md` after claims audit | To be generated in Phase 14 |
| `executive_summary.md` | Short reviewer-facing project summary | To be created in Phase 14 |
| `presentation_outline.md` | Evidence-first presentation outline | To be created in Phase 14 |
| `selected_artifacts.md` | Final-report artifact and evidence inventory | Created in Phase 14 |
| `limitations.md` | Final curated limitations and caveats | To be created in Phase 14 |
| `reproducibility_note.md` | Final report reproducibility note | To be created in Phase 14 |
| `future_work.md` | Future work and explicitly out-of-scope extensions | To be created in Phase 14 |
| `result_claims_audit.md` | Claim-to-evidence audit for final report conclusions | Created in Phase 14 |
| `table_inventory.md` | Table inventory for final report and PDF | Created in Phase 14 |
| `figure_inventory.md` | Figure inventory for final report and PDF | Created in Phase 14 |

## Selected Final Figures

Selected final-report figure copies live under:

```text
reports/final/figures/
```

These are copied from local generated figure outputs after Phase 14 evidence review. Only the selected copies are final-package artifacts. Original generated figure directories remain local-only by default.

## Source of Truth

The Markdown report is the source of truth:

```text
reports/final/final_report.md
```

The PDF is an export deliverable:

```text
reports/final/final_report.pdf
```

Do not manually diverge the PDF from the Markdown source. If the PDF needs edits, update the Markdown or the Markdown-to-PDF export path first.

## Phase 14 Package Rule

Phase 14 is a release package phase. It includes:

```text
final report
PDF export
executive summary
presentation outline
selected artifact inventory
claims audit
table inventory
figure inventory
limitations
future work
reproducibility note
release checklist
submission package map
```

## Claims Audit Rule

Every major conclusion in the final report and PDF must appear in:

```text
reports/final/result_claims_audit.md
```

Each claim must map to:

```text
claim
evidence file
evidence column / metric
allowed wording
forbidden overclaim
report section
```

Do not include numeric findings unless the corresponding evidence table has been inspected.

Use placeholders until verified:

```text
[INSERT VALUE FROM reports/tables/<path>: metric_name]
```

## Table and Figure Rule

Every table or figure used in the final report or PDF must appear in:

```text
reports/final/table_inventory.md
reports/final/figure_inventory.md
reports/final/selected_artifacts.md
```

Do not include huge raw tables, full generated panels, broker-sensitive outputs, or unverified local artifacts in the PDF.

## Artifact Commit Policy

Commit by default:

```text
reports/final/*.md
reports/final/final_report.pdf
reports/final/figures/*.png
```

Do not commit by default:

```text
data/raw/*
data/interim/*
data/processed/*
data/manual/*
data/broker_cache/*
models/*
logs/*
full backtest panels
full strategy signal panels
full cross-market panels
broker runtime artifacts
```

Selected small tables or figures may be committed only if they are stable, non-sensitive, reproducible, and explicitly selected for final report review.

## Required Caveats

The final report package must not claim:

```text
live-trading profitability
true option-chain PnL
account returns
causal US-to-India transmission
true R MSGARCH implementation
broker order execution
Phase 12 implementation
```

The final report must clearly state:

```text
VIX and India VIX are implied-volatility proxies, not variance swap quotes.
Daily OHLC realised variance estimators are proxies for true realised variance.
Phase 10 outputs are research-layer VRP proxy results, not executable account returns.
Phase 11 is paper-signal readiness only and did not place broker orders.
Phase 12 was skipped / future optional.
Phase 13 cross-market diagnostics are predictive/statistical diagnostics, not causal proof.
Generated data panels remain local and are not committed to GitHub.
```
