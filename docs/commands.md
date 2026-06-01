# Commands

## Setup

```bash
pip install -e .
pip install -e ".[dev]"
pytest
```

## Phase Commands

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

## Validation Shortcuts

```bash
pytest tests/test_data_schema.py tests/test_data_loaders.py
pytest tests/test_hmm_model.py tests/test_hmm_filtering.py
pytest tests/test_backtest_accounting.py tests/test_backtest_metrics.py
```
