# Phase 13 - Cross-Market US-India Analysis

## Status

Complete / frozen.

Phase 13 has been implemented, locally validated, and documented for the Phase 14 report baseline.

## Objective

Phase 13 evaluates whether US volatility, VRP, and regime information has descriptive or predictive value for Indian VRP regimes.

The phase separates:

1. Same-date US/India descriptive diagnostics.
2. Lagged-US predictive India diagnostics.
3. Analysis-only India overlay diagnostics using lagged US stress probability.

## Phase Boundary

Phase 13 does:

- Build a same-date descriptive US/India panel.
- Build a predictive India panel using only US observations strictly before the India date.
- Audit cross-market date alignment and stale lags.
- Compute VRP level/change correlations.
- Compute regime probability correlations.
- Compute state-label agreement diagnostics.
- Compute lead-lag and Granger-style descriptive diagnostics.
- Compare local-only versus local-plus-US logistic models.
- Run chronological OOS predictive diagnostics.
- Build an analysis-only India overlay that blocks base exposure during lagged US stress.
- Hash locked Phase 9 and Phase 10 artifacts before and after the run.

Phase 13 does not:

- Use same-date US information for India prediction.
- Claim Granger causality.
- Modify Phase 9 strategy signals.
- Modify Phase 10 backtest outputs.
- Read Phase 11 paper-signal or broker artifacts.
- Add a new official strategy to the Phase 9 universe.
- Use broker, iBridgePy, or paper-trading artifacts.
- Download new data.

## Files Owned by This Phase

### Config

```text
configs/cross_market.yaml
```

### Scripts

```text
scripts/run_cross_market_analysis.py
```

### Source

```text
src/vrp/reports/cross_market.py
src/vrp/reports/cross_market_diagnostics.py
src/vrp/strategies/cross_market_overlay.py
```

### Tests

```text
tests/test_cross_market_alignment.py
tests/test_cross_market_no_lookahead.py
tests/test_cross_market_stats.py
tests/test_cross_market_overlay.py
tests/test_phase13_artifact_mutation.py
tests/test_phase13_datetime_dtype.py
```

## Main Functions / Classes / Scripts

```text
load_cross_market_config
validate_cross_market_inputs
validate_no_forbidden_phase11_inputs
collect_locked_artifact_hashes
hash_locked_artifacts_before_after
previous_us_trading_date_for_india
align_us_india_predictive_panel
assert_no_same_date_us_leakage
build_alignment_audit
build_cross_market_no_lookahead_audit
build_descriptive_same_date_panel
build_predictive_panel
compute_vrp_level_correlations
compute_vrp_change_correlations
compute_regime_probability_correlations
compute_state_label_agreement
build_lead_lag_table
compute_granger_diagnostics
compute_logistic_diagnostic_tables
build_all_india_cross_market_overlays
validate_overlay_summary_schema
scripts/run_cross_market_analysis.py
```

## Config Files Used

```text
configs/cross_market.yaml
```

Important config rules:

```yaml
alignment:
  method: asof_backward_strict
  allow_exact_matches: false
  require_us_lagged_date_lt_india_date: true

overlay:
  analysis_only: true
  no_phase9_mutation: true
  no_phase10_mutation: true
  no_phase11_usage: true
```

## Input Files

Expected local inputs:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
data/processed/us_backtest_panel.parquet
data/processed/india_backtest_panel.parquet
reports/tables/phase_10/backtest_summary.csv
reports/tables/phase_10/backtest_metadata.json
```

Forbidden inputs:

```text
reports/tables/phase_11/daily_paper_signal.csv
reports/tables/phase_11/paper_order_intents.csv
reports/tables/phase_11/risk_check_report.csv
```

## Generated Output Files

```text
data/processed/cross_market_same_date_descriptive_panel.parquet
data/processed/cross_market_predictive_panel.parquet
data/processed/cross_market_panel.parquet
data/processed/india_cross_market_overlay_panel.parquet

reports/tables/phase_13/alignment_audit.csv
reports/tables/phase_13/no_lookahead_audit.csv
reports/tables/phase_13/phase13_metadata.json
reports/tables/phase_13/phase13_run_status.json
reports/tables/phase_13/vrp_level_correlations.csv
reports/tables/phase_13/vrp_change_correlations.csv
reports/tables/phase_13/regime_probability_correlations.csv
reports/tables/phase_13/state_label_agreement.csv
reports/tables/phase_13/lead_lag_table.csv
reports/tables/phase_13/granger_diagnostics.csv
reports/tables/phase_13/logistic_model_summary.csv
reports/tables/phase_13/logistic_parameter_summary.csv
reports/tables/phase_13/logistic_model_comparison.csv
reports/tables/phase_13/logistic_oos_diagnostics.csv
reports/tables/phase_13/india_overlay_summary.csv
reports/tables/phase_13/phase13_summary_index.csv

reports/figures/phase_13/us_india_vrp.png
reports/figures/phase_13/us_india_stress_prob.png
reports/figures/phase_13/lagged_us_vs_india_stress.png
reports/figures/phase_13/india_overlay_equity_curves.png
reports/figures/phase_13/india_overlay_exposure.png
```

## Commit vs Local-Only

Commit:

```text
source code
config
tests
docs
README files
```

Keep local by default:

```text
data/processed/cross_market_*.parquet
data/processed/india_cross_market_overlay_panel.parquet
reports/tables/phase_13/*.csv
reports/tables/phase_13/*.json
reports/figures/phase_13/*.png
```

## Commands to Regenerate Outputs

Validate inputs:

```bash
python scripts/run_cross_market_analysis.py --validate-inputs-only
```

Run without overlay:

```bash
python scripts/run_cross_market_analysis.py --model ALL --skip-overlay --force
```

Run full Phase 13:

```bash
python scripts/run_cross_market_analysis.py --model ALL --force
```

## Tests to Run

Phase-specific tests:

```bash
pytest tests/test_cross_market_alignment.py tests/test_cross_market_no_lookahead.py tests/test_cross_market_stats.py tests/test_cross_market_overlay.py tests/test_phase13_artifact_mutation.py tests/test_phase13_datetime_dtype.py
```

Full suite:

```bash
pytest
```

## Validation Checklist

- `alignment_audit.csv` has `n_same_date_violations = 0`.
- `no_lookahead_audit.csv` has `passes_no_lookahead = True`.
- Predictive panel has no rows where `us_lagged_date >= india_date`.
- Same-date panel has `predictive_allowed = False`.
- Granger diagnostics have `descriptive_only = True`.
- Granger diagnostics have `causal_interpretation_allowed = False`.
- Logistic comparison includes local-only versus local-plus-US models.
- Logistic comparison includes pseudo-R2, AIC, BIC, log likelihood, LR test, AUC, and Brier score.
- Overlay summary has `analysis_only = True`.
- Phase 9/10 locked artifact hashes do not change.
- Phase 11 artifacts are not read.

## No-Lookahead / Safety Rules

- Same-date US/India correlations are descriptive only.
- Predictive India analysis uses lagged US information only.
- Every matched predictive row must satisfy `us_lagged_date < india_date`.
- Do not use US same-date close, VIX, VRP, or stress probability for India prediction.
- Do not use outcome labels or future realised variance as tradable features.
- Do not use full-sample smoothed probabilities as backtest signals.
- Do not use Phase 11 broker or paper-signal artifacts.
- Overlay is analysis-only and not part of the Phase 9 strategy universe.

## Known Limitations

- Daily close-level data does not model intraday US/India timezone mechanics.
- US and India holidays create stale-lag gaps.
- Granger-style tests are statistical lead-lag diagnostics, not causal proof.
- Logistic OOS diagnostics may be mixed even when in-sample LR tests improve.
- Overlay performance is a robustness diagnostic and not an executable strategy.
- Cross-market analysis depends on locally generated upstream Phase 4, 6, 7, 9, and 10 artifacts.

## Review Checklist

- Review `configs/cross_market.yaml`.
- Review `scripts/run_cross_market_analysis.py`.
- Review `src/vrp/reports/cross_market.py`.
- Review `src/vrp/reports/cross_market_diagnostics.py`.
- Review `src/vrp/strategies/cross_market_overlay.py`.
- Run Phase 13 tests.
- Run `python scripts/run_cross_market_analysis.py --validate-inputs-only`.
- Check that generated Phase 13 artifacts are untracked.
