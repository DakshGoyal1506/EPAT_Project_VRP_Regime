# `vrp.backtest`

## Purpose

This package owns Phase 10 of the EPAT VRP regime project: vectorised research-layer backtesting and robustness diagnostics.

It converts fixed Phase 9 strategy signals into audited research payoff panels, diagnostics, robustness summaries, and final Phase 10 artifact checks.

## Phase Ownership

Primary owner:

```text
Phase 10 - vectorised research-layer backtest and robustness
```

Related upstream dependencies:

```text
Phase 3 - VRP labels and outcome construction
Phase 4 - HAR-RV and HAR-based VRP
Phase 5 - threshold regimes
Phase 6 - Gaussian HMM regimes
Phase 7 - Markov autoregression regimes
Phase 9 - strategy signals
```

## Responsibilities

This package may:

- load fixed Phase 9 strategy signals;
- load VRP/HAR-VRP outcome panels;
- validate Phase 10 input schemas;
- enforce the seven-strategy universe;
- build research backtest panels;
- compute proxy payoff, cost, and net proxy return;
- compute metrics and availability diagnostics;
- run cost/subperiod robustness;
- write skip-safe robustness reports;
- run final artifact audits.

This package must not:

- create new strategies;
- mutate Phase 9 signal construction;
- use future labels as tradable features;
- use full-sample smoothed HMM probabilities as backtest signals;
- use MSVOL as a strategy source;
- download tradable proxy data;
- place broker orders;
- define executable options PnL;
- report proxy sums as percentage return on invested capital.

## Main Modules

| Module | Purpose |
|---|---|
| `backtest_config.py` | Load and validate `configs/backtest.yaml`. |
| `backtest_registry.py` | Lock strategy universe and forbid unsafe signal columns/sources. |
| `schema_audit.py` | Preflight audit for Phase 10 input files. |
| `payoff_proxies.py` | Join signals to outcome labels and compute research payoff proxy. |
| `costs.py` | Compute exposure-change transaction cost proxies. |
| `metrics.py` | Compute strategy metrics, additive equity curves, drawdowns, tail metrics, and availability. |
| `vectorized_engine.py` | Run market-level backtests and write panel/metadata outputs. |
| `robustness.py` | Run cost sensitivity, subperiod robustness, and skip-safe optional reports. |
| `tradable_proxy_detector.py` | Detect existing local tradable proxy files without downloading data. |
| `final_audit.py` | Validate final Phase 10 panels, reports, figures, metadata, robustness outputs, and payoff identities. |

## Expected Inputs

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

## Expected Outputs

Generated local-only outputs:

```text
data/processed/us_backtest_panel.parquet
data/processed/us_backtest_panel_metadata.json
data/processed/india_backtest_panel.parquet
data/processed/india_backtest_panel_metadata.json
reports/tables/phase_10/*
reports/figures/phase_10/*
```

## Payoff Definition

Primary research payoff:

```text
gross_return_proxy = -target_exposure_for_backtest * vrp_forward_expost_gk_label
```

Cost proxy:

```text
cost_proxy = abs(delta_exposure) * cost_bps / 10000
```

Net proxy return:

```text
net_return_proxy = gross_return_proxy - cost_proxy
```

This is not executable account PnL.

## Eligibility Accounting

Eligible rows:

```text
target_exposure_for_backtest = target_exposure
```

Valid flat rows:

```text
target_exposure_for_backtest = 0.0
gross_return_proxy = 0.0
```

Unavailable rows:

```text
target_exposure_for_backtest = NaN
is_backtest_eligible = False
exclusion_reason records why the row is excluded
```

## CLI Entry Points

```bash
python scripts/audit_phase10_inputs.py --market ALL
python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force
python scripts/generate_backtest_diagnostics.py --market ALL
python scripts/run_robustness.py --market ALL --test all --force
python scripts/audit_phase10_final.py --market ALL
```

Single-strategy inspection:

```bash
python scripts/run_backtest.py --market US --strategy unconditional_full --cost-bps 5 --dry-run
```

Single-strategy writes to canonical output paths are disabled.

## Tests

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

## Safety and No-lookahead Boundaries

- `vrp_forward_expost_gk_label` is outcome-only.
- The label must not enter tradable features, regimes, or strategies.
- Strategy decisions must be timestamped before returns are realised.
- HMM/MAR backtest decisions must use filtered probabilities available at time `t`.
- Full-sample smoothed probabilities are diagnostic-only.
- MSVOL outputs are appendix diagnostics only.
- Phase 10 is not a broker or execution layer.
- No live trading is permitted.
