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
| 2 | Realised variance panels | `data/processed/us_rv.parquet`, `data/processed/india_rv.parquet` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No | Generated processed panels | `rv_summary.csv`, tests, small preview |
| 2 | RV summary | `reports/tables/rv_summary.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic table | CSV preview |
| 2 | RV estimator correlations | `reports/tables/rv_estimator_correlations.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic table | CSV preview |
| 2 | RV metadata | `reports/tables/rv_metadata.json` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated run metadata | JSON preview |
| 2 | RV figures | `reports/figures/rv_estimators_us.png`, `reports/figures/rv_estimators_india.png` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic figures | Screenshot if needed |
| 3 | Implied variance panels | `data/processed/us_iv.parquet`, `data/processed/india_iv.parquet` | `python scripts/build_features.py --market ALL --feature iv` | No | Generated processed panels | IV head/tail and formula check |
| 3 | VRP panels | `data/processed/us_vrp.parquet`, `data/processed/india_vrp.parquet` | `python scripts/build_features.py --market ALL --feature vrp` | No | Generated processed panels | `vrp_summary.csv`, metadata, tests |
| 3 | VRP summary | `reports/tables/vrp_summary.csv` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated diagnostic table | CSV preview |
| 3 | VRP metadata | `reports/tables/vrp_metadata.json` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated run metadata | JSON preview |
| 3 | Calendar mismatch report | `reports/tables/calendar_mismatches.csv` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated alignment diagnostic | CSV preview |
| 3 | VRP figures | `reports/figures/us_iv_rv_vrp.png`, `reports/figures/india_iv_rv_vrp.png` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated diagnostic figures | Screenshot if needed |
| 4 | HAR forecast panels | `data/processed/us_har_forecast.parquet`, `data/processed/india_har_forecast.parquet` | `python scripts/train_har.py --market ALL --mode expanding --force ...` | No | Generated processed panels | Schema/head/tail preview |
| 4 | HAR-VRP panels | `data/processed/us_vrp_har.parquet`, `data/processed/india_vrp_har.parquet` | Same | No | Generated processed panels | HAR-VRP unavailable-row validation |
| 4 | HAR forecast accuracy | `reports/tables/har_forecast_accuracy.csv` | Same | No by default | Generated diagnostic table | CSV preview |
| 4 | HAR coefficients | `reports/tables/har_coefficients.csv` | Same | No by default | Generated diagnostic table | CSV preview |
| 4 | HAR-VRP summary | `reports/tables/har_vrp_summary.csv` | Same | No by default | Generated diagnostic table | CSV preview |
| 4 | HAR metadata | `reports/tables/har_metadata.json` | Same | No by default | Generated metadata | JSON preview |
| 4 | HAR no-lookahead audit | `reports/tables/har_no_lookahead_audit.csv` | Same | No by default | Critical timing audit | Audit validation output |
| 4 | HAR figures | `reports/figures/har_*.png` | Same | No by default | Generated diagnostic figures | Screenshots if needed |
| 5 | Threshold regime panels | `data/processed/us_threshold_regimes.parquet`, `data/processed/india_threshold_regimes.parquet` | `python scripts/train_regimes.py --model threshold --market ALL --force` | No | Generated regime panels | `threshold_regime_summary.csv`, schema/head preview |
| 5 | Threshold diagnostics | `reports/tables/threshold_*.csv`, `reports/tables/threshold_regime_metadata.json` | Same | No by default | Generated diagnostics and metadata | CSV/JSON preview |
| 5 | Threshold figures | `reports/figures/threshold_*.png` | Same | No by default | Generated diagnostic figures | Screenshot if needed |
| 5 | Detailed Phase 5 artifact documentation | `docs/artifacts/phase_05_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 6 | HMM primary regime panels | `data/processed/us_hmm_regimes.parquet`, `data/processed/india_hmm_regimes.parquet` | `python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force` | No | Generated regime/probability panels | `hmm_candidate_model_ranking.csv`, `hmm_no_lookahead_audit.csv`, schema/head preview |
| 6 | HMM model-specific panels | `data/processed/*_hmm_*.parquet` | Same | No | Generated candidate panels | Candidate ranking table and selected model summary |
| 6 | HMM primary model binaries | `models/us_gaussian_hmm.pkl`, `models/india_gaussian_hmm.pkl` | Same | No | Generated model/scaler/state-label bundles | Config, metadata, candidate ranking, no-lookahead audit |
| 6 | HMM model-specific binaries | `models/*_hmm_*.pkl` | Same | No | Generated model binaries | Metadata and candidate ranking |
| 6 | HMM diagnostics | `reports/tables/phase_6/{market}/*.csv`, `reports/tables/phase_6/{market}/*.json` | Same | No by default | Generated diagnostics and audit reports | Selected CSV previews; especially candidate ranking and no-lookahead audit |
| 6 | HMM figures | `reports/figures/phase_6/*` | Phase 6 diagnostics if figures are later added | No by default | Generated figures | Screenshots or final-report-selected figures only |
| 7 | MAR primary regime panels | `data/processed/us_markov_autoreg_regimes.parquet`, `data/processed/india_markov_autoreg_regimes.parquet` | `python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force` | No | Generated regime/probability panels | Schema/head/tail preview and probability audit |
| 7 | MAR model-specific panels | `data/processed/markov_autoreg/*.parquet` | Same or `python scripts/train_markov_autoreg.py --market ALL --run-grid --force` | No | Generated model-specific outputs | Candidate ranking and metadata preview |
| 7 | MAR model binaries | `models/us_markov_autoreg.pkl`, `models/india_markov_autoreg.pkl`, `models/markov_autoreg/*.pkl` | Same | No | Generated model payloads | Metadata JSON and fit-summary preview |
| 7 | MAR diagnostics | `reports/tables/phase_7/{market}/*.csv`, `reports/tables/phase_7/{market}/*.json` | Same | No by default | Generated diagnostics and audit reports | Selected CSV/JSON previews, especially probability and no-lookahead audits |
| 7 | MAR figures | `reports/figures/phase_7/*` | Future Phase 7 diagnostics if enabled | No by default | Generated figures | Screenshots or selected final-report figures only |
| 7 | Detailed Phase 7 artifact documentation | `docs/artifacts/phase_07_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 8 | MSVOL input exports | `data/interim/msgarch/*_msgarch_input.csv` | `python scripts/export_msgarch_inputs.py --market ALL` | No | Generated return-only model inputs; legacy msgarch folder name | `reports/tables/phase_8/msgarch_input_summary.csv` preview |
| 8 | MSVOL raw model outputs | `data/interim/msvol/*` | `python scripts/run_msvol_regimes.py --market ALL` | No | Generated model probabilities and metadata | terminal output, model summary JSON preview |
| 8 | MSVOL processed regime panels | `data/processed/us_msvol_regimes.parquet`, `data/processed/india_msvol_regimes.parquet` | `python scripts/import_msvol_outputs.py --market ALL` | No | Generated processed regime panels | schema/head/tail preview and probability audit |
| 8 | MSVOL diagnostics | `reports/tables/phase_8/**/*` | `python scripts/run_msvol_diagnostics.py --market ALL` and `python scripts/run_msvol_no_lookahead_audit.py --market ALL` | No by default | Generated appendix diagnostics and audits | selected CSV/JSON previews, especially no-lookahead audit |
| 8 | MSVOL figures | `reports/figures/phase_8/**/*` | Future Phase 8 diagnostic plotting if added | No by default | Generated figures | screenshots or final-report-selected figures only |
| 8 | Detailed Phase 8 artifact documentation | `docs/artifacts/phase_08_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 9 | Strategy signal panels | `data/processed/*strategy_signals*.parquet` | `python scripts/build_signals.py --market ALL --strategy all --force` | No | Generated signal panels | `strategy_signal_summary.csv`, schema/head preview |
| 9 | Strategy diagnostics | `reports/tables/phase_9/*`, `reports/figures/phase_9/*` | `python scripts/build_signals.py --market ALL --strategy all --force` | No by default | Generated signal diagnostics | Selected CSV/JSON preview |
| 9 | Detailed Phase 9 artifact documentation | `docs/artifacts/phase_09_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 10 | Backtest panels | `data/processed/*_backtest_panel.parquet`, `data/processed/*_backtest_panel_metadata.json` | `python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force` | No | Generated research proxy panels | Head/tail/schema preview and final audit JSON |
| 10 | Backtest diagnostics | `reports/tables/phase_10/backtest_*.csv`, `reports/tables/phase_10/backtest_metadata.json`, `reports/tables/phase_10/phase10_*audit*` | `python scripts/generate_backtest_diagnostics.py --market ALL` and `python scripts/audit_phase10_final.py --market ALL` | No by default | Generated diagnostics and audits | Selected CSV/JSON previews |
| 10 | Robustness outputs | `reports/tables/phase_10/robustness_*.csv`, `reports/tables/phase_10/robustness_*.json`, `reports/tables/phase_10/tradable_proxy_detection.json` | `python scripts/run_robustness.py --market ALL --test all --force` | No by default | Generated robustness diagnostics | CSV/JSON preview |
| 10 | Backtest figures | `reports/figures/phase_10/*` | `python scripts/generate_backtest_diagnostics.py --market ALL` | No by default | Generated research proxy figures | Screenshots or final-report-selected figures only |
| 10 | Detailed Phase 10 artifact documentation | `docs/artifacts/phase_10_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 11 | Daily paper signal | `reports/tables/phase_11/daily_paper_signal.csv` | `python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json` | No | Runtime paper-signal artifact | Redacted JSON summary and one-row preview |
| 11 | Paper order intents | `reports/tables/phase_11/paper_order_intents.csv` | Same | No | Runtime paper-intent artifact, not executed orders | Redacted JSON summary and one-row preview |
| 11 | Risk check report | `reports/tables/phase_11/risk_check_report.csv` | Same | No | Runtime safety/risk audit | CSV preview |
| 11 | Broker metadata | `reports/tables/phase_11/broker_metadata.json` | Same | No | Broker status may be sensitive/local | Redacted JSON preview |
| 11 | Run metadata | `reports/tables/phase_11/run_metadata.json` | Same | No | Runtime audit metadata | Redacted JSON preview |
| 11 | Config snapshot | `reports/tables/phase_11/ibkr_paper_config_snapshot.yaml` | Same | No | Effective broker config snapshot | Redacted YAML preview |
| 11 | Integration report | `reports/tables/phase_11/phase11_integration_report.json` | `python scripts/validate_phase11.py --print-json` | No | Runtime artifact consistency check | JSON preview |
| 11 | Live-order guard report | `reports/tables/phase_11/live_order_guard_report.json` | `python scripts/validate_phase11.py --print-json` | No | Source safety scan output | JSON preview |
| 11 | Broker cache | `data/broker_cache/*` | Broker/paper-signal layer | No | Sensitive/local | Redacted status taxonomy only |
| 11 | Detailed Phase 11 artifact documentation | `docs/artifacts/phase_11_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 12 | Optional paper execution outputs | `reports/tables/phase_12/*`, `broker_logs/*` | Future paper-execution scripts if explicitly scoped | No | Broker-sensitive/local runtime context | Redacted execution summary |
| 13 | Cross-market descriptive panel | `data/processed/cross_market_same_date_descriptive_panel.parquet` | `python scripts/run_cross_market_analysis.py --model ALL --force` | No | Generated processed panel | `alignment_audit.csv`, schema/head preview |
| 13 | Cross-market predictive panel | `data/processed/cross_market_predictive_panel.parquet` | Same | No | Generated lagged predictive panel | `no_lookahead_audit.csv`, schema/head preview |
| 13 | Cross-market combined panel | `data/processed/cross_market_panel.parquet` | Same | No | Generated convenience panel | Separate descriptive/predictive panel previews |
| 13 | India cross-market overlay panel | `data/processed/india_cross_market_overlay_panel.parquet` | Same | No | Generated overlay diagnostic panel | `india_overlay_summary.csv` |
| 13 | Cross-market diagnostics | `reports/tables/phase_13/*.csv`, `reports/tables/phase_13/*.json` | Same | No by default | Generated diagnostics and metadata | Selected CSV/JSON previews |
| 13 | Cross-market figures | `reports/figures/phase_13/*.png` | Same | No by default | Generated diagnostic figures | Screenshots or final-report-selected figures |
| 13 | Detailed Phase 13 artifact documentation | `docs/artifacts/phase_13_artifacts.md` | Manual docs | Yes | Phase artifact documentation | File review |
| 14 | Final report | `reports/final_report.md` or final report export | Final report generation | Yes when ready | Deliverable | Full review |
| 14 | Release checklist | `docs/release_checklist.md` if created | Manual cleanup | Yes | Freeze governance | File review |

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
