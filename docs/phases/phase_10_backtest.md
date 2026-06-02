# Phase 10 - Vectorised Research Backtest and Robustness

## 1. Status

Complete / needs final review.

Phase 10 is implemented, tested, and audited. This document records the phase boundary, owned files, commands, outputs, validation checks, and limitations before Phase 13 work begins.

## 2. Objective

Phase 10 converts fixed Phase 9 strategy signals into a vectorised research-layer backtest for US and India.

The phase evaluates whether regime-conditioned short-volatility exposure improves risk-adjusted performance relative to unconditional short-volatility exposure.

This is a research proxy backtest, not executable account PnL.

## 3. Phase Boundary

Phase 10 may:

- load fixed Phase 9 strategy signal panels;
- load VRP/HAR-VRP outcome panels;
- construct auditable backtest panels;
- compute research proxy payoffs;
- compute exposure-change costs;
- compute metrics, diagnostics, robustness tables, and figures;
- run input and final artifact audits.

Phase 10 must not:

- create new strategies;
- change Phase 9 signals;
- tune regimes after seeing backtest results;
- use MSVOL as a strategy source;
- use forward/ex-post labels as tradable features;
- use full-sample smoothed HMM probabilities as tradable signals;
- place, preview, or simulate broker orders;
- define executable options PnL;
- define percentage return on invested capital without an explicit capital model.

## 4. Files Owned by This Phase

### Config

```text
configs/backtest.yaml
```

### Scripts

```text
scripts/audit_phase10_inputs.py
scripts/run_backtest.py
scripts/generate_backtest_diagnostics.py
scripts/run_robustness.py
scripts/audit_phase10_final.py
```

### Source Modules

```text
src/vrp/backtest/__init__.py
src/vrp/backtest/backtest_config.py
src/vrp/backtest/backtest_registry.py
src/vrp/backtest/schema_audit.py
src/vrp/backtest/payoff_proxies.py
src/vrp/backtest/costs.py
src/vrp/backtest/metrics.py
src/vrp/backtest/vectorized_engine.py
src/vrp/backtest/robustness.py
src/vrp/backtest/tradable_proxy_detector.py
src/vrp/backtest/final_audit.py
src/vrp/reports/backtest_diagnostics.py
```

### Tests

```text
tests/test_phase10_input_schema.py
tests/test_backtest_config_registry.py
tests/test_backtest_accounting.py
tests/test_backtest_no_lookahead.py
tests/test_backtest_metrics.py
tests/test_vectorized_engine.py
tests/test_backtest_diagnostics.py
tests/test_robustness.py
tests/test_phase10_integration.py
tests/test_no_lookahead.py
```

## 5. Main Functions / Classes / Scripts

### Input Audit

```text
audit_phase10_inputs()
audit_market_inputs()
assert_no_audit_errors()
```

Script:

```bash
python scripts/audit_phase10_inputs.py --market ALL
```

### Config and Registry

```text
load_backtest_config()
validate_backtest_config()
assert_strategy_universe_locked()
assert_no_outcome_labels_used_as_signals()
assert_no_msvol_strategy_use()
assert_no_smoothed_probability_use()
```

### Payoff Construction

```text
build_forward_vrp_outcome_panel()
join_strategy_with_outcome()
compute_forward_vrp_strategy_payoff()
build_research_backtest_panel()
```

Primary payoff formula:

```text
gross_return_proxy = -target_exposure_for_backtest * vrp_forward_expost_gk_label
```

### Cost Accounting

```text
compute_exposure_change_costs()
apply_costs_to_backtest_panel()
```

Cost formula:

```text
cost_proxy = abs(delta_exposure) * cost_bps / 10000
net_return_proxy = gross_return_proxy - cost_proxy
```

### Metrics

```text
compute_equity_curve()
compute_strategy_metrics()
build_strategy_metric_table()
build_availability_summary()
```

### Vectorised Engine

```text
run_market_backtest()
run_backtests()
write_backtest_outputs()
validate_backtest_panel_integrity()
build_backtest_metadata()
```

### Diagnostics

```text
generate_backtest_diagnostics()
build_backtest_summary_table()
build_common_start_summary_table()
build_tail_summary_table()
build_crisis_window_performance_table()
build_no_lookahead_audit_table()
```

### Robustness

```text
run_cost_sensitivity_robustness()
run_subperiod_robustness()
write_weekly_rebalance_skip_report()
run_tradable_proxy_detection()
run_robustness_suite()
```

### Final Audit

```text
run_phase10_final_audit()
write_final_audit_report()
assert_final_audit_passed()
```

## 6. Config Files Used

```text
configs/backtest.yaml
```

This config defines:

- input file paths;
- approved seven-strategy universe;
- payoff label and role;
- outcome alignment;
- cost model;
- robustness cost grid;
- crisis/subperiod windows;
- output paths;
- report paths.

## 7. Input Files

Required local input files:

```text
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
```

Additional configured context inputs:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
```

## 8. Generated Output Files

### Processed Panels

```text
data/processed/us_backtest_panel.parquet
data/processed/us_backtest_panel_metadata.json
data/processed/india_backtest_panel.parquet
data/processed/india_backtest_panel_metadata.json
```

### Tables

```text
reports/tables/phase_10/phase10_input_audit.csv
reports/tables/phase_10/phase10_input_audit.json
reports/tables/phase_10/backtest_summary.csv
reports/tables/phase_10/backtest_common_start_summary.csv
reports/tables/phase_10/backtest_tail_summary.csv
reports/tables/phase_10/backtest_by_strategy_year.csv
reports/tables/phase_10/crisis_window_performance.csv
reports/tables/phase_10/backtest_availability_summary.csv
reports/tables/phase_10/backtest_no_lookahead_audit.csv
reports/tables/phase_10/backtest_metadata.json
reports/tables/phase_10/robustness_cost_sensitivity.csv
reports/tables/phase_10/robustness_subperiods.csv
reports/tables/phase_10/robustness_weekly_rebalance_skipped.json
reports/tables/phase_10/tradable_proxy_detection.json
reports/tables/phase_10/robustness_metadata.json
reports/tables/phase_10/phase10_final_audit.json
```

### Figures

```text
reports/figures/phase_10/equity_curves_us.png
reports/figures/phase_10/equity_curves_india.png
reports/figures/phase_10/equity_curves_common_start_us.png
reports/figures/phase_10/equity_curves_common_start_india.png
reports/figures/phase_10/drawdowns_us.png
reports/figures/phase_10/drawdowns_india.png
reports/figures/phase_10/return_distribution_us.png
reports/figures/phase_10/return_distribution_india.png
```

## 9. Commit vs Local-only

Commit:

```text
code
configs
scripts
tests
documentation
README files
.gitkeep placeholders
```

Do not commit by default:

```text
data/processed/*_backtest_panel.parquet
data/processed/*_backtest_panel_metadata.json
reports/tables/phase_10/*
reports/figures/phase_10/*
```

Selected small summaries or figures may be committed later only if explicitly approved for the final report.

## 10. Commands to Regenerate Outputs

Run only after all Phase 9 and upstream inputs exist locally.

```bash
python scripts/audit_phase10_inputs.py --market ALL
python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force
python scripts/generate_backtest_diagnostics.py --market ALL
python scripts/run_robustness.py --market ALL --test all --force
python scripts/audit_phase10_final.py --market ALL
```

Single-strategy inspection must use dry run:

```bash
python scripts/run_backtest.py --market US --strategy unconditional_full --cost-bps 5 --dry-run
```

Single-strategy writes to canonical output paths are disabled.

## 11. Tests to Run

```bash
pytest tests/test_phase10_input_schema.py
pytest tests/test_backtest_config_registry.py
pytest tests/test_backtest_accounting.py
pytest tests/test_backtest_no_lookahead.py
pytest tests/test_backtest_metrics.py
pytest tests/test_vectorized_engine.py
pytest tests/test_backtest_diagnostics.py
pytest tests/test_robustness.py
pytest tests/test_phase10_integration.py
pytest tests/test_no_lookahead.py
```

Full suite:

```bash
pytest
```

## 12. Validation Checklist

- [ ] `configs/backtest.yaml` loads successfully.
- [ ] Strategy universe has exactly seven approved Phase 9 strategies.
- [ ] MSVOL strategies are not used.
- [ ] Forward/ex-post labels are not used as signal features.
- [ ] `target_trade_date > signal_observation_date` on eligible rows.
- [ ] `outcome_label_date == signal_observation_date` under default alignment.
- [ ] Valid flat rows are eligible and have zero return.
- [ ] Unavailable rows are excluded and counted.
- [ ] Cost accounting uses eligible rows only.
- [ ] `gross_return_proxy = -target_exposure_for_backtest * vrp_forward_expost_gk_label`.
- [ ] `net_return_proxy = gross_return_proxy - cost_proxy`.
- [ ] Common-start summaries are generated.
- [ ] Tail summary is generated.
- [ ] Robustness outputs are generated or skip-safe.
- [ ] Final audit passes with zero errors.

## 13. No-lookahead and Safety Rules

- The forward VRP label is outcome-only.
- It is joined only after signals exist.
- It must not enter Phase 9 signal construction.
- Full-sample smoothed HMM probabilities are diagnostic-only.
- Strategy decisions must use filtered probabilities available at time `t`.
- MSVOL outputs are diagnostic appendix artifacts only.
- Broker logic is out of scope for Phase 10.
- Phase 10 does not place, preview, or route orders.

## 14. Known Limitations

- The payoff is a research-layer proxy, not executable options PnL.
- No initial capital or margin model is defined.
- Cumulative proxy curves are additive sums, not account equity curves.
- The 22-trading-day forward labels overlap.
- Annualized metrics are approximate and intended for relative comparison.
- Transaction costs are simplified exposure-change costs.
- Weekly rebalance robustness is skip-safe by default unless separately audited.
- Tradable proxy detection does not download new data.

## 15. Review Checklist

- [ ] Inspect `configs/backtest.yaml`.
- [ ] Inspect `src/vrp/backtest/`.
- [ ] Inspect `src/vrp/reports/backtest_diagnostics.py`.
- [ ] Inspect Phase 10 tests.
- [ ] Run `pytest`.
- [ ] Run `git diff --check`.
- [ ] Confirm no generated parquet/model/log/env artifacts are tracked.
- [ ] Confirm generated Phase 10 reports remain local unless explicitly approved.
