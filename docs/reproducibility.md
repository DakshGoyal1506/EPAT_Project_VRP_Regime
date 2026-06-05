# Reproducibility

This repository is designed so a reviewer can understand and validate the research pipeline without receiving large local data files.

## Environment Setup

Run from the repository root.

```bash
pip install -e .
pip install -e ".[dev]"
```

Optional GPU support for Phase 4:

```bash
pip install -e ".[gpu]"
```

PyTorch CUDA installation may require using the official PyTorch wheel selector for the local CUDA/runtime environment.

If using conda:

```bash
conda activate epat
pip install -e .
pip install -e ".[dev]"
```

## Required Runtime Assumptions

1. Python version follows `pyproject.toml`.
2. Production logic is imported from `src/vrp/`.
3. Scripts are run from the repository root.
4. Notebooks do not define production logic.
5. Local generated data may be absent in a fresh clone.
6. `.env` is local-only and must not be committed.

## Smoke Validation

```bash
pytest
python scripts/download_data.py --dry-run
python scripts/build_features.py --help
python scripts/train_har.py --help
python scripts/train_regimes.py --help
python scripts/run_backtest.py --help
python scripts/run_ibkr_paper_signal.py --help
python scripts/run_cross_market_analysis.py --help
```

## Full Local Regeneration Sequence

This sequence assumes internet access and public data-source availability.

```bash
python scripts/download_data.py --market ALL --source all --dry-run
python scripts/download_data.py --market US --source all --force
python scripts/download_data.py --market INDIA --source yahoo --force
python scripts/build_features.py --market ALL --feature rv --window 22
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
```

Later-phase commands should be run only after confirming their inputs exist locally.

Phase 4 CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend cpu_numpy_batched --coefficient-hac-frequency none
```

Phase 6 Gaussian HMM regeneration requires Phase 4 HAR-VRP panels and optional Phase 5 threshold panels:

```bash
python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force
```

Generated HMM panels, model binaries, and reports remain local-only.

Phase 7 Markov autoregression regeneration requires Phase 4 HAR-VRP panels and may optionally read Phase 5/6 regime outputs for diagnostics:

```bash
python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force
```

Generated MAR panels, model binaries, and reports remain local-only.

Phase 8 MSVOL robustness appendix requires Phase 4 HAR-VRP panels and writes generated diagnostics locally:

```bash
python scripts/export_msgarch_inputs.py --market ALL
python scripts/run_msvol_regimes.py --market ALL
python scripts/import_msvol_outputs.py --market ALL
python scripts/run_msvol_diagnostics.py --market ALL
python scripts/run_msvol_no_lookahead_audit.py --market ALL
```

Phase 8 is Python-only MSVOL, not true MSGARCH. True R MSGARCH remains optional/future and is not required for reproducibility.

Phase 13 cross-market analysis requires locally generated Phase 4, 6, 7, 9, and 10 outputs:

```bash
python scripts/run_cross_market_analysis.py --validate-inputs-only
python scripts/run_cross_market_analysis.py --model ALL --force
```

Generated Phase 13 panels, diagnostics, and figures remain local-only.

## Reproducibility Boundaries

### Reproducible from tracked repo

```text
source code
configs
tests
scripts
docs
README files
environment template
directory placeholders
```

### Requires local/generated artifacts

```text
raw market data
processed feature panels
trained model outputs
regime panels
strategy signal panels
backtest panels
broker/paper-signal cache
```

### Requires manual/local intervention

```text
NSE manual CSV files if scripted access is blocked
broker account identifiers
iBridgePy / TWS / IBKR setup
optional R/MSGARCH environment
GPU-specific HAR acceleration
```

## Data Refresh Policy

1. Refresh raw source data through scripts where possible.
2. Do not manually edit processed parquet files.
3. Do not forward-fill missing prices silently.
4. Do not silently mix sources.
5. Source priority must be defined in config.
6. Keep US and India calendars separate until explicit cross-market phases.

## Notebook Reproducibility

Notebooks are inspection-only.

Allowed:

```text
read generated outputs
plot diagnostics
display head/tail
summarize missingness
call src/vrp functions
```

Forbidden:

```text
download data
repair production data manually
define production formulas
fit production models
generate final signals
place or preview broker orders
```

Before committing notebooks:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Reviewer Workflow

A reviewer should be able to inspect:

```text
README.md
docs/
configs/
src/vrp/
scripts/
tests/
```

Then run:

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
```

Generated data is not required to understand repository design.
