# Artifact Inventory

This file documents generated artifacts, whether they should be committed, and what a reviewer should inspect instead of large local files.

## Policy Summary

| Artifact type | Commit? | Reason |
|---|---:|---|
| Source code | Yes | Required for reproducibility |
| Config files | Yes | Required for reproducibility |
| Tests | Yes | Required for validation |
| Documentation | Yes | Required for review |
| `.gitkeep` placeholders | Yes | Preserve directory structure |
| Raw downloaded data | No | Large/generated/source-refreshable |
| Processed parquet panels | No | Generated/local research outputs |
| Model binaries | No | Generated/local and potentially large |
| Broker cache/state/logs | No | Sensitive and local |
| Full backtest panels | No | Generated/local |
| Full strategy signal panels | No | Generated/local |
| Selected small summary CSV/JSON | Optional | Only if stable, non-sensitive, and useful for final review |
| Selected final figures | Optional | Only if small, stable, and used in final report |

## Phase Artifact Table

| Phase | Artifact | Path | Producer command | Commit? | Reason | Review substitute |
|---|---|---|---|---|---|---|
| 0 | Repo scaffold | `src/`, `configs/`, `scripts/`, `tests/`, `docs/` | Manual scaffold | Yes | Core project structure | Git tree |
| 0 | Environment template | `.env.example` | Manual scaffold | Yes | Safe placeholder only | File review |
| 0 | Package metadata | `pyproject.toml` | Manual scaffold | Yes | Install/test reproducibility | File review |
| 1 | US VIX raw CBOE | `data/raw/us_vix_cboe.parquet` | `python scripts/download_data.py --market US --source cboe --force` | No | Generated data | `reports/tables/data_audit.csv` row |
| 1 | US VIX raw FRED | `data/raw/us_vix_fred.parquet` | `python scripts/download_data.py --market US --source fred --force` | No | Generated data | `reports/tables/data_audit.csv` row |
| 1 | US SPX/SPY raw Yahoo | `data/raw/us_spx_yahoo.parquet`, `data/raw/us_spy_yahoo.parquet` | `python scripts/download_data.py --market US --source yahoo --force` | No | Generated data | `reports/tables/data_audit.csv` row |
| 1 | India VIX raw Yahoo | `data/raw/india_vix_yahoo.parquet` | `python scripts/download_data.py --market INDIA --source yahoo --force` | No | Generated data | `reports/tables/data_audit.csv` row |
| 1 | India NIFTY raw Yahoo | `data/raw/india_nifty_yahoo.parquet` | `python scripts/download_data.py --market INDIA --source yahoo --force` | No | Generated data | `reports/tables/data_audit.csv` row |
| 1 | Official NSE manual files | `data/manual/nse/*` | Manual download | No | Source files; local/manual | Document source/date in notes |
| 1 | Processed VIX/underlying panels | `data/processed/us_vix.parquet`, `data/processed/us_underlying.parquet`, `data/processed/india_vix.parquet`, `data/processed/india_underlying.parquet` | `python scripts/download_data.py ...` | No | Generated panels | Audit table and head/tail printout |
| 1 | Data audit | `reports/tables/data_audit.csv` | `python scripts/download_data.py ...` | Optional | Small summary | CSV or terminal preview |
| 2 | Realised variance panels | `data/processed/us_rv.parquet`, `data/processed/india_rv.parquet` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No | Generated panels | RV summary table, tests |
| 2 | RV figures | `reports/figures/rv_*.png` | RV diagnostics | Optional final only | Generated figures | Small selected final figure |
| 3 | Implied variance panels | `data/processed/us_iv.parquet`, `data/processed/india_iv.parquet` | `python scripts/build_features.py --market ALL --feature iv` | No | Generated panels | IV summary |
| 3 | VRP panels | `data/processed/us_vrp.parquet`, `data/processed/india_vrp.parquet` | `python scripts/build_features.py --market ALL --feature vrp` | No | Generated panels | VRP summary, metadata |
| 3 | VRP metadata | `reports/tables/vrp_metadata.json` | VRP build | Optional | Small metadata | JSON review |
| 4 | HAR forecast panels | `data/processed/us_har_forecast.parquet`, `data/processed/india_har_forecast.parquet` | `python scripts/train_har.py ...` | No | Generated panels | Forecast accuracy table |
| 4 | HAR coefficients | `reports/tables/har_coefficients.csv` | HAR training | Optional | Small diagnostic | Summary excerpt |
| 5 | Threshold regime panels | `data/processed/*threshold*regimes*.parquet` | Regime training command | No | Generated regime panels | Regime summary table |
| 6 | HMM regime panels | `data/processed/*hmm*regimes*.parquet` | `python scripts/train_regimes.py ...` | No | Generated regime panels | HMM diagnostics |
| 6 | HMM model binaries | `data/processed/models/hmm/*` | HMM training | No | Model artifacts | Config + diagnostics |
| 7 | Markov autoregression outputs | `data/processed/*markov_autoreg*.parquet` | `python scripts/train_markov_autoreg.py ...` | No | Generated regime panels | MAR diagnostics |
| 8 | Strategy signal panels | `data/processed/*strategy_signals*.parquet` | Strategy/backtest scripts | No | Generated signal panels | Strategy summary |
| 9 | Robustness outputs | `reports/tables/*robustness*.csv` | `python scripts/run_robustness.py ...` | Optional | Small summaries only | CSV/summary excerpt |
| 10 | Cross-market panels | `data/processed/*cross_market*.parquet` | Phase 10 scripts | No | Generated panels | Phase 10 audit summary |
| 10 | Cross-market summary | `reports/tables/phase_10/*.csv` | Phase 10 scripts | Optional | Small summary | Selected CSV |
| 11 | Paper signal output | `reports/tables/phase_11/*` | `python scripts/run_ibkr_paper_signal.py ...` | No by default | Broker-sensitive runtime context | Redacted terminal output |
| 11 | Broker cache | `data/broker_cache/*` | Broker/paper signal layer | No | Sensitive/local | Redacted status taxonomy only |
| 12 | Final report | `reports/final_report.md` or final report export | Final report generation | Yes when ready | Deliverable | Full review |
| 13 | Release checklist | `docs/release_checklist.md` if created | Manual cleanup | Yes | Freeze governance | File review |

## Local Artifact Review Packet

When asked for review, send these instead of committing heavy files:

```text
pytest output
selected script command output
git status --short
git ls-files data reports docs | sort
reports/tables/*.csv previews if small
head/tail of generated parquet files printed in terminal
screenshots of selected figures if needed
```

## Local Artifact Manifest Template

Use this template locally if a run produces many files:

```text
local_artifacts/YYYY-MM-DD_run_manifest.md
```

Template:

```markdown
# Local Run Manifest

Date:
Git commit:
Conda/env:
Command run:

## Generated Files

| Path | Rows/Size | Purpose | Keep local? | Notes |
|---|---:|---|---|---|

## Validation

| Check | Result |
|---|---|

## Review Substitutes

| Heavy file | Lightweight substitute |
|---|---|
```
