# Phase 7 Artifacts - Markov Autoregression

This document describes Phase 7 generated artifacts. These files are reproducible local outputs and are not committed by default.

## Commit Policy

Commit:

```text
configs/model_markov_autoreg.yaml
scripts/train_markov_autoreg.py
src/vrp/regimes/markov_autoreg.py
src/vrp/regimes/markov_autoreg_registry.py
src/vrp/reports/markov_autoreg_diagnostics.py
tests/test_markov_autoreg.py
tests/test_markov_autoreg_no_lookahead.py
docs/phases/phase_07_markov_autoreg.md
docs/artifacts/phase_07_artifacts.md
```

Do not commit:

```text
data/processed/*markov_autoreg*.parquet
models/*markov_autoreg*.pkl
reports/tables/phase_7/**/*
reports/figures/phase_7/**/*
logs
broker cache
```

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute |
|---|---|---|---:|---|---|---|
| US primary MAR regime panel | `data/processed/us_markov_autoreg_regimes.parquet` | `python scripts/train_markov_autoreg.py --market US --target vrp_har --order 1 --states 2 --primary --force` | No | Generated model output | `date`, `mar_signal_trade_date`, `mar_model_observation_available`, `mar_state_name_for_next_session`, `mar_filtered_prob_*_for_next_session` | Schema/head/tail printout and probability audit |
| India primary MAR regime panel | `data/processed/india_markov_autoreg_regimes.parquet` | `python scripts/train_markov_autoreg.py --market INDIA --target vrp_har --order 1 --states 2 --primary --force` | No | Generated model output | Same as US panel | Schema/head/tail printout and probability audit |
| US model-specific MAR panel | `data/processed/markov_autoreg/us_markov_autoreg_vrp_har_order1_k2_sv.parquet` | Same US primary command | No | Generated auditable model-specific output | Same as primary panel plus raw-state probability columns | Candidate ranking and metadata preview |
| India model-specific MAR panel | `data/processed/markov_autoreg/india_markov_autoreg_vrp_har_order1_k2_sv.parquet` | Same India primary command | No | Generated auditable model-specific output | Same as primary panel plus raw-state probability columns | Candidate ranking and metadata preview |
| Candidate-grid MAR panels | `data/processed/markov_autoreg/*.parquet` | `python scripts/train_markov_autoreg.py --market ALL --run-grid --force` | No | Generated robustness outputs | Same MAR panel schema | Candidate ranking preview |
| US primary model payload | `models/us_markov_autoreg.pkl` | US or ALL primary command | No | Generated model binary | Train-fitted params, spec, fit summary, state mapping, target transform metadata | Metadata JSON and fit-summary preview |
| India primary model payload | `models/india_markov_autoreg.pkl` | India or ALL primary command | No | Generated model binary | Same as US model payload | Metadata JSON and fit-summary preview |
| Model-specific payloads | `models/markov_autoreg/*.pkl` | Primary or grid command | No | Generated model binaries | Train-fitted params and metadata | Candidate ranking and metadata preview |
| Metadata | `reports/tables/phase_7/{market}/mar_metadata.json` | Primary or grid command | No by default | Generated run metadata | Config, prepared data, fit, full filter, signal, output paths | JSON preview |
| Candidate ranking | `reports/tables/phase_7/{market}/mar_candidate_model_ranking.csv` | Primary or grid command | No by default | Generated diagnostic table | `target`, `order`, `n_states`, `switching_variance`, `valid_candidate`, `aic`, `bic`, `invalid_reason` | CSV head |
| State summary | `reports/tables/phase_7/{market}/mar_state_summary.csv` | Primary or grid command | No by default | Generated diagnostic table | `raw_state`, `economic_state_name`, `target_mean_train`, `target_std_train`, `sigma2`, `ar_lag1_phi`, `stress_score` | CSV head |
| AR stability | `reports/tables/phase_7/{market}/mar_ar_stability.csv` | Primary or grid command | No by default | Generated diagnostic table | `raw_state`, `economic_state_name`, `ar_lag1_phi`, `sigma2`, `persistence_prob`, `half_life_days`, `ar_stable`, `ar_warning` | CSV head |
| Transition matrix | `reports/tables/phase_7/{market}/mar_transition_matrix.csv` | Primary or grid command | No by default | Generated diagnostic table | `from_state`, `to_state_*` | CSV preview |
| Probability audit | `reports/tables/phase_7/{market}/mar_probability_audit.csv` | Primary or grid command | No by default | No-lookahead/probability validation | `passed`, `n_model_available_rows`, `n_warmup_rows`, `max_row_sum_abs_error` | CSV preview |
| No-lookahead audit | `reports/tables/phase_7/{market}/mar_no_lookahead_audit.csv` | Primary or grid command | No by default | Critical timing audit | `params_estimated_using_full_sample`, `filtered_probabilities_use_train_params_only`, `smoothed_probabilities_used_for_backtest`, `passed` | CSV preview |
| Duration summary | `reports/tables/phase_7/{market}/mar_duration_summary.csv` | Primary or grid command | No by default | Generated regime-duration diagnostic | `state_name`, `n_runs`, `mean_duration_days`, `max_duration_days`, `total_days` | CSV head |
| State-by-year summary | `reports/tables/phase_7/{market}/mar_state_by_year.csv` | Primary or grid command | No by default | Generated annual state distribution | `year`, `state_name`, `n_days`, `year_total_days`, `state_fraction` | CSV head |
| MAR/HMM agreement | `reports/tables/phase_7/{market}/mar_hmm_agreement.csv` | Primary or grid command if Phase 6 outputs exist | No by default | Diagnostic comparison only | `comparison`, `n_overlap`, `agreement_rate` plus crosstab columns | CSV preview |
| MAR/threshold agreement | `reports/tables/phase_7/{market}/mar_threshold_agreement.csv` | Primary or grid command if Phase 5 outputs exist | No by default | Diagnostic comparison only | `comparison`, `n_overlap`, `agreement_rate` plus crosstab columns | CSV preview |
| Global regime comparison | `reports/tables/regime_model_comparison.csv` | Primary or grid command | No by default | Generated model-comparison summary | `model_family`, `market`, `target`, `bic`, `lookahead_audit_passed`, `fraction_calm`, `fraction_stress` | CSV preview |
| Phase 7 figures | `reports/figures/phase_7/**/*` | Future diagnostics if enabled | No by default | Generated figures | N/A | Screenshot if selected |

## Sensitivity and Reproducibility Notes

1. MAR panels are generated from public market data but can be large and should stay local.
2. Model payloads contain generated model parameters and should stay local.
3. Generated report tables are reproducible and should stay local unless selected as final-report artifacts.
4. No broker account, order, or live-trading information belongs in Phase 7 artifacts.
5. Diagnostic smoothed probabilities must not be used as strategy or backtest signals.

## Review Substitute Commands

```bash
python scripts/train_markov_autoreg.py --help
pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py
```

If local artifacts exist, reviewers can inspect schema/head/tail previews instead of committing generated files.

No generated artifact should be committed for this review.
