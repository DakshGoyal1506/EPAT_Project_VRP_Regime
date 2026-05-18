# Scripts

This folder contains the command-line entry points for the project. The scripts are thin orchestration layers around the reusable code in `src/vrp/` and should be run from the repository root.

## Entry Points

- `build_features.py` - Phase 2 and 3 feature builds and diagnostics.
- `download_data.py` - source-data ingestion and refresh.
- `run_backtest.py` - vectorised strategy backtests.
- `run_ibkr_paper_signal.py` - optional paper-signal integration.
- `run_robustness.py` - robustness evaluation workflows.
- `train_regimes.py` - regime-model training workflows.
 - `train_har.py` - Phase 4 HAR-RV forecasting orchestration and HAR-based VRP outputs.

## Build Features

`build_features.py` is the central Phase 2 / Phase 3 orchestration script.

It can build:

- realised variance panels (`rv`)
- implied variance panels (`iv`)
- implied variance plus VRP panels (`vrp`)

The `vrp` mode is intentionally locked to the Phase 3 22-day contract so the output columns stay aligned with the registry and downstream notebooks.

### Key Arguments

- `--market` - `US`, `INDIA`, or `ALL`
- `--feature` - `rv`, `iv`, or `vrp`
- `--window` - RV window length; Phase 3 VRP requires `22`
- `--horizon` - forward label horizon; Phase 3 VRP requires `22`
- `--annualization-periods` - annualization factor, default `252`
- `--max-vix-value` - upper bound check for VIX-style closes, default `200`
- `--skip-diagnostics` - save parquet outputs without writing reports

### Typical Commands

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp

### HAR Forecasting (Phase 4)

Run the Phase 4 HAR forecasting workflow:

```bash
python scripts/train_har.py --market US --mode expanding
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64
```

Quick backend parity check (synthetic data):

```bash
python scripts/smoke_backend_parity.py
```
```

### Expected VRP Outputs

- `data/processed/us_iv.parquet`
- `data/processed/india_iv.parquet`
- `data/processed/us_vrp.parquet`
- `data/processed/india_vrp.parquet`
- `reports/tables/vrp_summary.csv`
- `reports/tables/vrp_metadata.json`
- `reports/tables/calendar_mismatches.csv`
- `reports/figures/us_iv_rv_vrp.png`
- `reports/figures/india_iv_rv_vrp.png`

## Validation

Run the full suite after changing orchestration logic:

```bash
pytest
```

For Phase 3-specific edits, this narrower check is useful:

```bash
pytest tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```
