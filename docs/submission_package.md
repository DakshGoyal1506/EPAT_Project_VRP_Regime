# Submission Package

This file tells a reviewer where to start and how to inspect the final EPAT project package.

Project:

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

---

## 1. Start Here

| Purpose | Path |
|---|---|
| Project overview | `README.md` |
| Final report source | `reports/final/final_report.md` |
| Final report PDF | `reports/final/final_report.pdf` |
| Executive summary | `reports/final/executive_summary.md` |
| Presentation outline | `reports/final/presentation_outline.md` |

Recommended first read:

```text
README.md
reports/final/executive_summary.md
reports/final/final_report.md
```

---

## 2. Final Report Package

| File                                    | Purpose                         |
| --------------------------------------- | ------------------------------- |
| `reports/final/README.md`               | Final package index and rules   |
| `reports/final/final_report.md`         | Markdown source of truth        |
| `reports/final/final_report.pdf`        | PDF submission deliverable      |
| `reports/final/executive_summary.md`    | Short summary for reviewers     |
| `reports/final/presentation_outline.md` | Evidence-first slide outline    |
| `reports/final/result_claims_audit.md`  | Claim-to-evidence audit         |
| `reports/final/table_inventory.md`      | Table inventory                 |
| `reports/final/figure_inventory.md`     | Figure inventory                |
| `reports/final/selected_artifacts.md`   | Selected artifact evidence map  |
| `reports/final/limitations.md`          | Final report limitations        |
| `reports/final/reproducibility_note.md` | Final reproducibility note      |
| `reports/final/future_work.md`          | Future extensions and non-scope |

---

## 3. Reproducibility

| Purpose                           | Path                                    |
| --------------------------------- | --------------------------------------- |
| Main reproducibility guide        | `docs/reproducibility.md`               |
| Final-report reproducibility note | `reports/final/reproducibility_note.md` |
| Command index                     | `docs/commands.md`                      |
| Python package config             | `pyproject.toml`                        |
| Environment placeholders          | `.env.example`                          |

Minimal validation:

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
```

---

## 4. Artifact Policy and Inventory

| Purpose                   | Path                                   |
| ------------------------- | -------------------------------------- |
| Generated artifact policy | `docs/generated_artifact_policy.md`    |
| Full artifact inventory   | `docs/artifact_inventory.md`           |
| Final selected artifacts  | `reports/final/selected_artifacts.md`  |
| Final table inventory     | `reports/final/table_inventory.md`     |
| Final figure inventory    | `reports/final/figure_inventory.md`    |
| Phase 14 artifact docs    | `docs/artifacts/phase_14_artifacts.md` |

Generated data panels are not committed.

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
```

---

## 5. Limitations and Scope Boundaries

| Purpose                   | Path                                   |
| ------------------------- | -------------------------------------- |
| Full limitations ledger   | `docs/known_limitations.md`            |
| Final limitations summary | `reports/final/limitations.md`         |
| Future work               | `reports/final/future_work.md`         |
| Claims audit              | `reports/final/result_claims_audit.md` |

Core scope boundaries:

```text
No live-trading profitability claim.
No true option-chain PnL claim.
No account-return claim.
No causal US-to-India transmission claim.
No true MSGARCH claim.
No broker order execution claim.
No Phase 12 implementation claim.
```

---

## 6. Phase Documentation

| Phase | Path                                           |
| ----: | ---------------------------------------------- |
|     0 | `docs/phases/phase_00_scaffold_governance.md`  |
|     1 | `docs/phases/phase_01_data_ingestion.md`       |
|     2 | `docs/phases/phase_02_realised_variance.md`    |
|     3 | `docs/phases/phase_03_implied_variance_vrp.md` |
|     4 | `docs/phases/phase_04_har_rv.md`               |
|     5 | `docs/phases/phase_05_threshold_regimes.md`    |
|     6 | `docs/phases/phase_06_gaussian_hmm.md`         |
|     7 | `docs/phases/phase_07_markov_autoreg.md`       |
|     8 | `docs/phases/phase_08_msvol_appendix.md`       |
|     9 | `docs/phases/phase_09_strategy_signals.md`     |
|    10 | `docs/phases/phase_10_backtest.md`             |
|    11 | `docs/phases/phase_11_ibkr_readiness.md`       |
|    13 | `docs/phases/phase_13_cross_market.md`         |
|    14 | `docs/phases/phase_14_final_report_release.md` |

Phase 12 status:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

---

## 7. Validation and Release Checklist

| Purpose                | Path                             |
| ---------------------- | -------------------------------- |
| Release checklist      | `docs/release_checklist.md`      |
| Final report checklist | `docs/final_report_checklist.md` |
| Phase status ledger    | `docs/phase_status.md`           |
| Command index          | `docs/commands.md`               |

Final validation commands:

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

## 8. Suggested Reviewer Route

### Fast review

```text
README.md
reports/final/executive_summary.md
reports/final/final_report.pdf
reports/final/limitations.md
```

### Technical review

```text
README.md
docs/phase_status.md
docs/reproducibility.md
docs/commands.md
reports/final/final_report.md
reports/final/result_claims_audit.md
reports/final/table_inventory.md
reports/final/figure_inventory.md
reports/final/selected_artifacts.md
```

### Validation review

```text
docs/release_checklist.md
docs/final_report_checklist.md
tests/README.md
scripts/README.md
```

### Artifact-policy review

```text
docs/generated_artifact_policy.md
docs/artifact_inventory.md
docs/artifacts/phase_14_artifacts.md
reports/final/selected_artifacts.md
.gitignore
```

---

## 9. What Is Not Included

The GitHub repository does not include:

```text
raw downloaded market data
processed parquet panels
trained model binaries
full regime panels
full strategy signal panels
full backtest panels
full cross-market panels
broker cache
broker logs
private account identifiers
```

These are intentionally local-only.

---

## 10. Reviewer Notes

Interpretation rules:

1. Treat Phase 10 as research-proxy backtest evidence only.
2. Treat Phase 11 as paper-signal readiness only.
3. Treat Phase 12 as skipped/future optional.
4. Treat Phase 13 as descriptive/predictive cross-market diagnostics only.
5. Treat MSVOL as Python-only robustness, not true MSGARCH.
6. Treat all generated panels as local-only.
