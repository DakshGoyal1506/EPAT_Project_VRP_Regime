# Final Report Checklist

This checklist applies specifically to:

```text
reports/final/final_report.md
reports/final/final_report.pdf
reports/final/executive_summary.md
reports/final/presentation_outline.md
```

It prevents report-level overclaiming and PDF export mistakes.

---

## 1. Source Control

* [ ] `reports/final/final_report.md` is the source of truth.
* [ ] `reports/final/final_report.pdf` is generated from the Markdown source.
* [ ] The PDF does not contain material absent from the Markdown source.
* [ ] Manual PDF-only edits have not introduced unaudited claims.

---

## 2. Required Report Sections

`final_report.md` includes:

* [ ] Title page.
* [ ] Abstract.
* [ ] Executive summary.
* [ ] Motivation.
* [ ] Research questions.
* [ ] Data.
* [ ] Methodology overview.
* [ ] Realised variance construction.
* [ ] Implied variance and VRP construction.
* [ ] HAR-RV forecasting.
* [ ] Regime modelling.
* [ ] Strategy construction.
* [ ] Vectorised research backtest.
* [ ] Robustness checks.
* [ ] Cross-market US-India analysis.
* [ ] IBKR paper-signal readiness appendix.
* [ ] Main findings.
* [ ] Terminology lock.
* [ ] Limitations.
* [ ] Future work.
* [ ] Reproducibility.
* [ ] Appendix.

---

## 3. Claims Audit

* [ ] Every major conclusion appears in `reports/final/result_claims_audit.md`.
* [ ] Every numeric claim maps to an evidence file.
* [ ] Every numeric claim maps to a metric or column.
* [ ] Every result claim has allowed wording.
* [ ] Every result claim has forbidden overclaim wording.
* [ ] Claims audit covers Phase 10.
* [ ] Claims audit covers Phase 11.
* [ ] Claims audit covers Phase 13.
* [ ] Claims audit covers MSVOL wording.
* [ ] Claims audit covers Phase 12 skipped/future wording.

---

## 4. Numeric Findings

* [ ] No invented numbers.
* [ ] No numbers copied from memory.
* [ ] No numbers inserted without inspecting local artifacts.
* [ ] Placeholder remains if evidence table was not inspected.
* [ ] Placeholder format is preserved where needed:

```text
[INSERT VALUE FROM reports/tables/<path>: metric_name]
```

* [ ] All remaining placeholders are intentional before PDF export.
* [ ] No unresolved placeholders remain in final PDF unless explicitly accepted.

---

## 5. Table Inventory

* [ ] Every report table appears in `reports/final/table_inventory.md`.
* [ ] Every PDF table appears in `reports/final/table_inventory.md`.
* [ ] Table source path is listed.
* [ ] Table producer command is listed where applicable.
* [ ] Table status is marked.
* [ ] Table commit policy is marked.
* [ ] Large generated tables are not pasted directly into the report.

---

## 6. Figure Inventory

* [ ] Every report figure appears in `reports/final/figure_inventory.md`.
* [ ] Every PDF figure appears in `reports/final/figure_inventory.md`.
* [ ] Figure source path is listed.
* [ ] Figure producer command is listed where applicable.
* [ ] Figure status is marked.
* [ ] Figure commit policy is marked.
* [ ] Figures are readable.
* [ ] Figures are not clipped in PDF.
* [ ] Figures have accurate captions.

---

## 7. Required Caveats

The final report includes these caveats:

* [ ] VIX and India VIX are implied-volatility proxies, not variance swap quotes.
* [ ] Daily OHLC realised variance estimators are proxies for true realised variance.
* [ ] 22 trading days approximate a one-month horizon but are not the same as 30 calendar days.
* [ ] Forward ex-post VRP labels are outcomes only, not tradable features.
* [ ] HAR forecasts are model-dependent and trained under point-in-time constraints.
* [ ] Gaussian HMM does not directly model observed autocorrelation.
* [ ] Markov autoregression addresses observed-series autocorrelation but remains reduced-form.
* [ ] MSVOL is Python-only Markov-switching volatility robustness, not true R MSGARCH.
* [ ] Strategy outputs are exposure intentions.
* [ ] Phase 10 returns are research-layer VRP proxy units, not executable account returns.
* [ ] Overlapping 22-day outcome labels make annualised metrics approximate.
* [ ] Phase 11 did not place broker orders.
* [ ] Phase 12 paper execution was skipped/future optional.
* [ ] Phase 13 cross-market lead-lag diagnostics are predictive/statistical diagnostics, not causal proof.
* [ ] Generated data panels remain local and are not committed to GitHub.

---

## 8. Forbidden Wording Check

Search the final report, PDF source, executive summary, and presentation outline for forbidden phrases.

Forbidden:

```text
profitable live strategy
live-trading profitability
account return
account returns
option-chain PnL
real trading returns
broker execution results
paper trading results
causal transmission
US causes India
true MSGARCH
Phase 12 implementation
orders were sent
guaranteed risk reduction
future crisis protection
```

Replace with approved wording where needed.

---

## 9. Approved Wording Check

Use these terms:

```text
research-proxy backtest
proxy return units
exposure intention
paper-signal readiness
live-order guard
predictive/statistical diagnostic
lead-lag association
Python-only MSVOL robustness
future optional paper execution adapter
no broker orders placed
drawdown behaviour in the tested proxy sample
```

---

## 10. Terminology Lock

* [ ] Terminology lock table exists in `final_report.md`.
* [ ] It includes `Research-proxy return`.
* [ ] It includes `Exposure intention`.
* [ ] It includes `Filtered probability`.
* [ ] It includes `Cross-market predictive diagnostic`.
* [ ] It includes `MSVOL robustness`.
* [ ] It includes `Paper-signal readiness`.

---

## 11. Phase 10 Wording

* [ ] Phase 10 is described as vectorised research-proxy backtest.
* [ ] Phase 10 results are described as proxy units.
* [ ] Cumulative curves are described as additive proxy curves.
* [ ] Drawdowns are described as proxy drawdowns.
* [ ] No account-return wording.
* [ ] No true option-chain PnL wording.
* [ ] No live-trading profitability wording.
* [ ] No executable equity-curve wording.

---

## 12. Phase 11 Wording

Allowed wording:

```text
paper-signal readiness layer
order-guard validation
configuration and risk-check demonstration
no broker orders placed
```

Checklist:

* [ ] Phase 11 is appendix-only.
* [ ] Phase 11 is not described as an execution layer.
* [ ] Phase 11 is not described as paper trading results.
* [ ] Phase 11 is not described as broker backtest.
* [ ] Report states no broker orders were placed.
* [ ] Broker-sensitive fields are excluded.

---

## 13. Phase 12 Wording

Required wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

Checklist:

* [ ] Phase 12 is not described as not started.
* [ ] Phase 12 is not described as implemented.
* [ ] Phase 12 is not described as partially implemented.
* [ ] Phase 12 appears only as skipped/future optional.

---

## 14. Phase 13 Wording

* [ ] Same-date diagnostics are described as descriptive only.
* [ ] Lagged-US diagnostics are described as predictive/statistical only.
* [ ] Granger-style diagnostics are not described as causal proof.
* [ ] Logistic tests are described as tested-sample predictive diagnostics.
* [ ] India overlay is described as analysis-only.
* [ ] India overlay is not described as a new Phase 9 strategy.
* [ ] No causal wording.

---

## 15. MSVOL Wording

* [ ] MSVOL is described as Python-only Markov-switching volatility robustness.
* [ ] MSVOL is not described as true MSGARCH.
* [ ] True R MSGARCH is listed as optional/future.
* [ ] MSVOL is diagnostic-only.
* [ ] MSVOL is not used as strategy construction evidence.

---

## 16. PDF Visual Inspection

After PDF export:

* [ ] Title page is readable.
* [ ] Abstract is readable.
* [ ] Executive summary is readable.
* [ ] Tables fit page width.
* [ ] Figures are not clipped.
* [ ] Captions are readable.
* [ ] Code blocks do not overflow badly.
* [ ] Page breaks are acceptable.
* [ ] No huge raw tables appear.
* [ ] No local-only artifact path appears without explanation.
* [ ] No unresolved placeholder appears unless explicitly accepted.
* [ ] No broker-sensitive information appears.

---

## 17. Executive Summary

* [ ] Executive summary is concise.
* [ ] Executive summary does not include unaudited numbers.
* [ ] Executive summary does not overclaim.
* [ ] Executive summary states non-scope.
* [ ] Executive summary includes key caveats.

---

## 18. Presentation Outline

* [ ] Presentation outline has 10-14 main slides.
* [ ] Each slide has 3-5 bullets.
* [ ] Each slide has suggested figure/table.
* [ ] Each slide has source artifact path.
* [ ] Each slide has speaker note.
* [ ] It does not include unaudited numeric claims.
* [ ] It keeps Phase 11 appendix-only.
* [ ] It keeps Phase 13 non-causal.
* [ ] It labels Phase 10 as research-proxy.

---

## Final Approval

Only export and commit the final PDF after all checklist items above are satisfied or explicitly documented.
