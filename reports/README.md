# Reports

This directory stores generated report artifacts.

Most report outputs are generated locally and are intentionally not tracked by Git.

## Directory Contract

| Path | Purpose | Commit? |
|---|---|---:|
| `reports/tables/` | Generated CSV/JSON/Markdown diagnostic tables | No by default |
| `reports/figures/` | Generated figures | No by default |
| `reports/final_report.md` | Final report draft/output when ready | Yes when final |

## Commit Policy

Commit:

```text
reports/README.md
reports/tables/README.md
reports/tables/.gitkeep
reports/figures/README.md
reports/figures/.gitkeep
selected final-report-ready artifacts only if explicitly approved
```

Do not commit by default:

```text
reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
reports/figures/*.svg
reports/figures/*.pdf
```

## Rules

1. Generated tables must be reproducible from scripts.
2. Generated figures must be reproducible from scripts or report modules.
3. Do not manually edit generated artifacts without documenting the change.
4. Final report conclusions must trace back to reproducible outputs.
5. Broker-sensitive outputs must remain local or be redacted before review.

## Phase 4 HAR-RV Outputs

Phase 4 may generate:

```text
reports/tables/har_forecast_accuracy.csv
reports/tables/har_coefficients.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_metadata.json
reports/tables/har_no_lookahead_audit.csv
reports/figures/har_forecast_us.png
reports/figures/har_forecast_india.png
reports/figures/har_residuals_us.png
reports/figures/har_residuals_india.png
reports/figures/har_vrp_us.png
reports/figures/har_vrp_india.png
```

These are local-only by default. Use previews, selected excerpts, or screenshots for review unless explicitly approved as final-report artifacts.

## Phase 5 Threshold-Regime Outputs

Phase 5 may generate:

```text
reports/tables/threshold_regime_summary.csv
reports/tables/threshold_component_summary.csv
reports/tables/threshold_transition_matrix.csv
reports/tables/threshold_state_duration_summary.csv
reports/tables/threshold_state_by_year.csv
reports/tables/threshold_crisis_hit_table.csv
reports/tables/threshold_crisis_lead_lag_table.csv
reports/tables/threshold_vrp_by_state.csv
reports/tables/threshold_forward_label_by_state.csv
reports/tables/threshold_no_lookahead_audit.csv
reports/tables/threshold_regime_metadata.json
reports/figures/threshold_regimes_us.png
reports/figures/threshold_regimes_india.png
reports/figures/threshold_regime_vrp_boxplots_us.png
reports/figures/threshold_regime_vrp_boxplots_india.png
reports/figures/threshold_component_states_us.png
reports/figures/threshold_component_states_india.png
```

These are generated diagnostics and stay local by default. Use previews, selected excerpts, or screenshots for review unless explicitly approved as final-report artifacts.

## Phase 6 Gaussian HMM Outputs

Phase 6 may generate:

```text
reports/tables/phase_6/us/hmm_candidate_model_ranking.csv
reports/tables/phase_6/us/hmm_feature_availability.csv
reports/tables/phase_6/us/hmm_state_summary.csv
reports/tables/phase_6/us/hmm_transition_matrix.csv
reports/tables/phase_6/us/hmm_state_duration_summary.csv
reports/tables/phase_6/us/hmm_state_by_year.csv
reports/tables/phase_6/us/hmm_threshold_agreement.csv
reports/tables/phase_6/us/hmm_crisis_hit_table.csv
reports/tables/phase_6/us/hmm_crisis_lead_lag_table.csv
reports/tables/phase_6/us/hmm_forward_label_by_state.csv
reports/tables/phase_6/us/hmm_probability_audit.csv
reports/tables/phase_6/us/hmm_no_lookahead_audit.csv
reports/tables/phase_6/us/hmm_metadata.json

reports/tables/phase_6/india/*.csv
reports/tables/phase_6/india/*.json
reports/figures/phase_6/*
```

These are generated diagnostics and stay local by default. Use selected CSV/JSON previews for review unless explicitly approved as final-report artifacts.

