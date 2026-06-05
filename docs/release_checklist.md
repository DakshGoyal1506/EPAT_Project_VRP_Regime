# Release Checklist

This checklist must be completed before marking Phase 14 as `complete / frozen`.

Use this checklist after all final report files, release docs, and PDF export are complete.

---

## 1. Git Hygiene

- [ ] `git status --short` reviewed.
- [ ] Only intended Phase 14 files are new or modified.
- [ ] `git diff --check` returns no whitespace errors.
- [ ] No accidental generated artifact changes are present.
- [ ] No accidental source-code or model-logic changes are present.

Commands:

```bash
git status --short
git diff --check
```

---

## 2. Heavy Artifact Tracking Check

* [ ] No generated heavy artifacts are tracked.
* [ ] No parquet panels are tracked.
* [ ] No model binaries are tracked.
* [ ] No logs are tracked.
* [ ] No environment files are tracked.

Commands:

```bash
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
git ls-files | findstr /i "broker_cache data/raw data/interim data/processed"
```

Expected result:

```text
No unexpected tracked files.
```

Allowed tracked exceptions:

```text
.env.example
data/*/.gitkeep
data/README.md
```

---

## 3. Credential and Broker-Sensitive Check

* [ ] `.env` is not tracked.
* [ ] Broker account identifiers are not tracked.
* [ ] Broker cache is not tracked.
* [ ] TWS / IBKR logs are not tracked.
* [ ] Paper order logs are not tracked.
* [ ] Broker metadata is not committed unless redacted and explicitly approved.

Check:

```bash
git status --short
git ls-files | findstr /i "\.env broker ibkr tws account order"
```

---

## 4. `.env.example`

* [ ] `.env.example` exists.
* [ ] `.env.example` contains placeholders only.
* [ ] `.env.example` does not contain private keys, account IDs, or tokens.

---

## 5. README Current

* [ ] Root `README.md` contains project title.
* [ ] Root `README.md` contains executive summary.
* [ ] Root `README.md` contains high-level pipeline.
* [ ] Root `README.md` contains current phase status.
* [ ] Root `README.md` links to final report location.
* [ ] Root `README.md` links to submission package map.
* [ ] Root `README.md` includes generated artifact policy.
* [ ] Root `README.md` includes no-lookahead policy.
* [ ] Root `README.md` includes no-live-trading warning.
* [ ] Root `README.md` does not contain full report text.
* [ ] Root `README.md` does not contain long command blocks.
* [ ] Root `README.md` does not contain internal chunk notes.

---

## 6. Phase Status Current

* [ ] `docs/phase_status.md` marks Phases 0-11 as `complete / frozen`.
* [ ] `docs/phase_status.md` marks Phase 12 as `skipped / future optional`.
* [ ] `docs/phase_status.md` marks Phase 13 as `complete / frozen`.
* [ ] `docs/phase_status.md` marks Phase 14 as `complete / frozen` only after this checklist is complete.
* [ ] Phase 12 wording is exactly scoped as skipped/future optional.

Required wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

---

## 7. Artifact Inventory Current

* [ ] `docs/artifact_inventory.md` includes Phase 14 final report artifacts.
* [ ] `docs/artifact_inventory.md` identifies final report Markdown as committed.
* [ ] `docs/artifact_inventory.md` identifies final report PDF as committed.
* [ ] `docs/artifact_inventory.md` identifies claims audit as committed.
* [ ] `docs/artifact_inventory.md` identifies inventories as committed.
* [ ] `docs/artifact_inventory.md` keeps generated panels local-only.
* [ ] `docs/artifact_inventory.md` keeps broker artifacts local/redacted.

---

## 8. Final Report Exists

* [ ] `reports/final/final_report.md` exists.
* [ ] Report has title page.
* [ ] Report has abstract.
* [ ] Report has executive summary.
* [ ] Report has methodology overview.
* [ ] Report has realised variance section.
* [ ] Report has implied variance and VRP section.
* [ ] Report has HAR-RV section.
* [ ] Report has regime modelling section.
* [ ] Report has strategy construction section.
* [ ] Report has vectorised research backtest section.
* [ ] Report has robustness section.
* [ ] Report has cross-market analysis section.
* [ ] Report has paper-signal readiness appendix.
* [ ] Report has limitations.
* [ ] Report has future work.
* [ ] Report has reproducibility.
* [ ] Report has appendix.

---

## 9. PDF Report Exists

* [ ] `reports/final/final_report.pdf` exists.
* [ ] PDF was generated from `reports/final/final_report.md`.
* [ ] PDF title page is readable.
* [ ] PDF tables are not clipped.
* [ ] PDF figures are not clipped.
* [ ] PDF page breaks are acceptable.
* [ ] PDF does not contain huge raw tables.
* [ ] PDF does not contain broker-sensitive outputs.
* [ ] PDF does not contain unverified numeric claims.
* [ ] PDF does not diverge from Markdown source.

---

## 10. Presentation Outline Exists

* [ ] `reports/final/presentation_outline.md` exists.
* [ ] It has 10-14 main slides.
* [ ] Each slide has 3-5 bullets.
* [ ] Each slide has suggested figure/table.
* [ ] Each slide has source artifact path.
* [ ] Each slide has speaker note.
* [ ] It does not include unaudited numeric claims.
* [ ] It does not use forbidden wording.

---

## 11. Selected Artifacts Inventory Exists

* [ ] `reports/final/selected_artifacts.md` exists.
* [ ] It contains evidence inventory.
* [ ] It distinguishes commit vs local-only artifacts.
* [ ] It identifies selected table candidates.
* [ ] It identifies selected figure candidates.
* [ ] It keeps full generated panels local-only.
* [ ] It keeps broker artifacts local/redacted.

---

## 12. Claims Audit Complete

* [ ] `reports/final/result_claims_audit.md` exists.
* [ ] Every major final report conclusion appears in the claims audit.
* [ ] Every numeric claim maps to evidence file and metric.
* [ ] Every claim has allowed wording.
* [ ] Every claim has forbidden overclaim wording.
* [ ] No claim asserts live-trading profitability.
* [ ] No claim asserts account returns.
* [ ] No claim asserts true option-chain PnL.
* [ ] No claim asserts causal US-to-India transmission.
* [ ] No claim asserts true MSGARCH.
* [ ] No claim asserts broker order execution.

---

## 13. Table and Figure Inventories Complete

* [ ] `reports/final/table_inventory.md` exists.
* [ ] `reports/final/figure_inventory.md` exists.
* [ ] Every final report table appears in table inventory.
* [ ] Every final report figure appears in figure inventory.
* [ ] Every PDF table appears in table inventory.
* [ ] Every PDF figure appears in figure inventory.
* [ ] Local-only status is marked correctly.
* [ ] Optional selected artifacts are marked correctly.

---

## 14. Tests Pass or Failures Documented

Run:

```bash
pytest
```

* [ ] Full test suite passes, or failures are documented with reason.
* [ ] No failure is caused by Phase 14 documentation edits.
* [ ] No source-code logic was changed in Phase 14.

---

## 15. Phase 11 Live-Order Guard Passes

Run:

```bash
python scripts/validate_phase11.py --print-json
```

* [ ] Validator runs.
* [ ] Live-order guard passes.
* [ ] No output indicates broker orders were placed.
* [ ] Any broker-sensitive fields remain local/redacted.

---

## 16. Phase 13 No-Lookahead Checks Pass

Run:

```bash
python scripts/run_cross_market_analysis.py --validate-inputs-only
pytest tests/test_cross_market_alignment.py tests/test_cross_market_no_lookahead.py tests/test_cross_market_stats.py tests/test_cross_market_overlay.py tests/test_phase13_artifact_mutation.py tests/test_phase13_datetime_dtype.py
```

* [ ] Input validation passes.
* [ ] Cross-market alignment tests pass.
* [ ] No-lookahead tests pass.
* [ ] Overlay mutation tests pass.
* [ ] Datetime dtype tests pass.
* [ ] No Phase 13 output is described causally.

---

## 17. Notebooks Cleared or Marked Inspection-Only

* [ ] `notebooks/README.md` states notebooks are inspection-only.
* [ ] Notebooks do not define production logic.
* [ ] Notebooks do not contain broker orders.
* [ ] Notebook outputs are cleared if committing notebooks.

Optional command:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## 18. Final Limitations Documented

* [ ] `reports/final/limitations.md` exists.
* [ ] VIX proxy limitation included.
* [ ] OHLC RV proxy limitation included.
* [ ] HAR model dependency included.
* [ ] HMM limitation included.
* [ ] MAR reduced-form limitation included.
* [ ] MSVOL-not-true-MSGARCH limitation included.
* [ ] Research-proxy-not-account-return limitation included.
* [ ] Cross-market-not-causal limitation included.
* [ ] Phase 11 no-orders limitation included.
* [ ] Phase 12 skipped/future limitation included.

---

## 19. Future Work Documented

* [ ] `reports/final/future_work.md` exists.
* [ ] True option-chain PnL is future work.
* [ ] Instrument-level execution is future work.
* [ ] IBKR paper execution adapter is future work.
* [ ] True R MSGARCH is future work.
* [ ] Intraday realised variance is future work.
* [ ] Production broker deployment is future work.
* [ ] No future item is implied as already implemented.

---

## 20. Submission Package Map Exists

* [ ] `docs/submission_package.md` exists.
* [ ] It routes to README.
* [ ] It routes to final report Markdown.
* [ ] It routes to final report PDF.
* [ ] It routes to executive summary.
* [ ] It routes to presentation outline.
* [ ] It routes to reproducibility docs.
* [ ] It routes to artifact inventory.
* [ ] It routes to limitations.
* [ ] It routes to validation checklist.

---

## Final Freeze Decision

Phase 14 can be marked `complete / frozen` only after all required checklist items are complete or documented.

Final status wording:

```text
Phase 14 = complete / frozen — final report, PDF export, presentation package, claims audit, release checklist, and submission package complete.
```
