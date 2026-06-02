# Phase 8 - Python-only MSVOL Robustness Appendix

## 1. Status

Complete / frozen.

Phase 8 implements a diagnostic-only volatility-regime robustness appendix. The active implementation is Python-only MSVOL using `statsmodels.MarkovRegression`, not true MSGARCH.

## 2. Objective

Phase 8 checks whether independently estimated return-volatility regimes broadly agree with stress regimes from:

```text
Phase 5 threshold regimes
Phase 6 Gaussian HMM regimes
Phase 7 Markov autoregression regimes
```

The active model is:

```text
statsmodels MarkovRegression
k_regimes = 2
switching_variance = true
switching_trend = false
```

This phase is an appendix robustness layer only.

## 3. Phase Boundary

Phase 8 owns:

```text
Python-only Markov-switching volatility robustness modelling
legacy return input export
raw MSVOL fitting
MSVOL processed regime panels
MSVOL diagnostics
MSVOL no-lookahead audit
optional archival R MSGARCH script documentation
```

Phase 8 does not own:

```text
strategy signal construction
backtesting
position sizing
VaR / ES
cross-market analysis
IBKR broker logic
live or paper execution
```

## 4. Python MSVOL Is Not True MSGARCH

True MSGARCH was considered as an optional robustness appendix, but it requires the R `MSGARCH` package.

The active Phase 8 implementation does not estimate GARCH recursion:

```text
sigma_t^2 = omega_s + alpha_s * epsilon_{t-1}^2 + beta_s * sigma_{t-1}^2
```

Instead, it uses Python `statsmodels.MarkovRegression` with regime-specific variance as a volatility-regime robustness proxy.

True R MSGARCH remains optional/future only.

## 5. Files Owned by Phase 8

Configs:

```text
configs/model_msvol.yaml
configs/model_msgarch.yaml
```

`model_msvol.yaml` is the active config. `model_msgarch.yaml` is legacy/archival for optional R MSGARCH input/export compatibility.

Scripts:

```text
scripts/export_msgarch_inputs.py
scripts/run_msvol_regimes.py
scripts/import_msvol_outputs.py
scripts/run_msvol_diagnostics.py
scripts/run_msvol_no_lookahead_audit.py
scripts/run_msgarch.R
```

`run_msgarch.R` is archival optional code only. It is not part of the active Python Phase 8 path.

Source modules:

```text
src/vrp/regimes/msvol_model.py
src/vrp/regimes/msvol_adapter.py
src/vrp/reports/msvol_diagnostics.py
src/vrp/reports/msvol_no_lookahead.py
```

Tests:

```text
tests/test_msgarch_export.py
tests/test_msvol_model.py
tests/test_msvol_adapter.py
tests/test_msvol_diagnostics.py
tests/test_msvol_no_lookahead.py
```

## 6. Main Functions and Classes

`src/vrp/regimes/msvol_model.py`:

```text
MSVolError
AR1PrefilterResult
MSVolFitResult
RunMarketResult
load_msvol_config
read_msvol_input_csv
prefilter_ar1
fit_msvol_markov_regression
build_raw_output_frame
run_msvol_for_market
```

`src/vrp/regimes/msvol_adapter.py`:

```text
MSVolAdapterError
MSVolStateMapping
MSVolImportResult
validate_msvol_raw_schema
validate_msvol_probability_rows
map_msvol_states_by_variance
standardize_msvol_output
import_msvol_outputs_for_market
```

`src/vrp/reports/msvol_diagnostics.py`:

```text
MSVolDiagnosticsError
LoadedTable
MSVolDiagnosticsResult
run_msvol_diagnostics_for_market
build_msvol_comparison_summary
build_state_duration_summary
```

`src/vrp/reports/msvol_no_lookahead.py`:

```text
MSVolNoLookaheadError
MSVolNoLookaheadAuditResult
build_no_lookahead_audit
run_msvol_no_lookahead_audit_for_market
```

## 7. Config Files Used

Active config:

```text
configs/model_msvol.yaml
```

Legacy/archival optional config:

```text
configs/model_msgarch.yaml
```

Important active settings:

```text
implementation = PYTHON_STATSMODELS_MARKOV_REGRESSION
true_msgarch = false
k_regimes = 2
trend = c
switching_variance = true
switching_trend = false
use_for_strategy = false
use_for_backtest = false
```

## 8. Input Files

Phase 8 uses Phase 4 HAR-VRP panels as source panels for return export:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

The export script writes return-only input CSVs:

```text
data/interim/msgarch/us_msgarch_input.csv
data/interim/msgarch/india_msgarch_input.csv
```

The `msgarch` folder name is legacy. The active model is MSVOL.

Input policy:

```text
Use index returns only.
Do not use HAR residuals.
Do not use VRP_HAR as the model target.
Do not use future/outcome/label columns.
```

## 9. Generated Output Files

Interim outputs:

```text
data/interim/msgarch/us_msgarch_input.csv
data/interim/msgarch/india_msgarch_input.csv
data/interim/msvol/us_msvol_raw_output.csv
data/interim/msvol/india_msvol_raw_output.csv
data/interim/msvol/us_msvol_preflight.json
data/interim/msvol/india_msvol_preflight.json
data/interim/msvol/us_msvol_model_summary.json
data/interim/msvol/india_msvol_model_summary.json
data/interim/msvol/us_msvol_skip_report.json
data/interim/msvol/india_msvol_skip_report.json
```

Processed outputs:

```text
data/processed/us_msvol_regimes.parquet
data/processed/india_msvol_regimes.parquet
```

Report outputs:

```text
reports/tables/phase_8/us/msvol_metadata.json
reports/tables/phase_8/india/msvol_metadata.json
reports/tables/phase_8/us/msvol_probability_audit.csv
reports/tables/phase_8/india/msvol_probability_audit.csv
reports/tables/phase_8/us/msvol_comparison_summary.csv
reports/tables/phase_8/india/msvol_comparison_summary.csv
reports/tables/phase_8/us/msvol_state_duration_summary.csv
reports/tables/phase_8/india/msvol_state_duration_summary.csv
reports/tables/phase_8/us/msvol_no_lookahead_audit.csv
reports/tables/phase_8/india/msvol_no_lookahead_audit.csv
reports/tables/phase_8/msvol_model_comparison_appendix.csv
reports/tables/phase_8/msvol_no_lookahead_audit.csv
reports/figures/phase_8/*
```

## 10. Commit vs Local-only Policy

Commit:

```text
configs
scripts
src modules
tests
docs
README files
.gitkeep placeholders
```

Do not commit by default:

```text
data/interim/msgarch/*
data/interim/msvol/*
data/processed/*_msvol_regimes.parquet
reports/tables/phase_8/*
reports/figures/phase_8/*
```

## 11. Commands to Regenerate Outputs

Run from repository root:

```bash
python scripts/export_msgarch_inputs.py --market ALL
python scripts/run_msvol_regimes.py --market ALL
python scripts/import_msvol_outputs.py --market ALL
python scripts/run_msvol_diagnostics.py --market ALL
python scripts/run_msvol_no_lookahead_audit.py --market ALL
```

## 12. Tests to Run

```bash
pytest tests/test_msgarch_export.py
pytest tests/test_msvol_model.py
pytest tests/test_msvol_adapter.py
pytest tests/test_msvol_diagnostics.py
pytest tests/test_msvol_no_lookahead.py
```

Optional full suite:

```bash
pytest
```

## 13. Validation Checklist

```text
MSVOL config says true_msgarch = false.
MSVOL config says use_for_strategy = false.
MSVOL config says use_for_backtest = false.
MSVOL config and msvol_model.py agree on switching_trend = false.
No R command is required for active Phase 8.
Raw MSVOL outputs exist locally after running model script.
Processed MSVOL regime panels exist locally after import script.
Probability audit reports probability sums near 1.
No-lookahead audit has zero failed error checks.
Smoothed probabilities are diagnostic-only.
Transition probability is zero.
No transition state is modelled.
Generated outputs remain untracked.
```

## 14. No-lookahead and Safety Rules

```text
Use filtered probabilities only for next-session regime timing.
Do not use smoothed probabilities as tradable signals.
Signal observation date is date t.
Signal availability is after date t close.
Signal trade date is the next available trading session.
Last row has missing trade date.
No future/outcome/label columns can enter model inputs.
No strategy/backtest/exposure/PnL columns are created in Phase 8.
```

## 15. Known Limitations

```text
Python MSVOL is not true MSGARCH.
True R MSGARCH remains optional/future.
MSVOL models return-volatility regimes only.
It does not model full GARCH recursion.
State labels are economic interpretations.
Comparator agreement is diagnostic only.
Missing threshold/HMM/MAR comparator files produce NaN or zero-overlap diagnostics rather than failure.
```

## 16. Review Checklist

```text
Read configs/model_msvol.yaml.
Confirm model_msvol.yaml and msvol_model.py agree on switching_trend.
Read scripts/run_msvol_regimes.py.
Read scripts/import_msvol_outputs.py.
Read scripts/run_msvol_diagnostics.py.
Read scripts/run_msvol_no_lookahead_audit.py.
Read src/vrp/regimes/msvol_model.py.
Read src/vrp/regimes/msvol_adapter.py.
Read src/vrp/reports/msvol_diagnostics.py.
Read src/vrp/reports/msvol_no_lookahead.py.
Run Phase 8 tests.
Run git hygiene check.
Confirm generated outputs are not tracked.
```
