# Reproducibility

## Environment Setup

Use the repository root and install the package in editable mode:

```bash
pip install -e .
pip install -e ".[dev]"
```

If you are using the project conda environment, activate it first before installing or running commands.

## Regeneration Sequence

Recommended local rerun order:

```bash
python scripts/download_data.py --dry-run
python scripts/build_features.py --market ALL --feature rv --window 22
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --coefficient-hac-frequency none
python scripts/train_regimes.py --help
python scripts/train_markov_autoreg.py --help
python scripts/run_backtest.py --help
python scripts/run_robustness.py --help
python scripts/run_ibkr_paper_signal.py --help
```

## Review Assumptions

Reviewers should not assume local raw data exists. The repository must remain understandable from tracked code, configs, docs, tests, and small summaries only.

When reviewing results, prefer the documented summary tables and command outputs over local generated panels.
