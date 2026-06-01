# Phase Status

This file is the authoritative status ledger for phases 0 through 12.

| Phase | Status | Scope | Validation command |
|---|---|---|---|
| 0 | complete / frozen | Repo foundation, packaging, configuration, schema, and test harness | `pip install -e .` then `pytest` |
| 1 | complete / frozen | Public data ingestion and schema validation | `python scripts/download_data.py --dry-run` |
| 2 | complete / frozen | Realised variance features and diagnostics | `python scripts/build_features.py --market ALL --feature rv --window 22` |
| 3 | complete / frozen | Implied variance and VRP construction | `python scripts/build_features.py --market ALL --feature iv` and `python scripts/build_features.py --market ALL --feature vrp` |
| 4 | complete / frozen | HAR-RV forecasting and prospective VRP panels | `python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --coefficient-hac-frequency none` |
| 5 | in-progress | Threshold regime filters | `python scripts/train_regimes.py --help` |
| 6 | in-progress | Gaussian HMM regime modelling | `python scripts/train_regimes.py --help` and `pytest tests/test_hmm_model.py tests/test_hmm_filtering.py` |
| 7 | in-progress | AR-HMM / Markov autoregression | `python scripts/train_markov_autoreg.py --help` and `pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py` |
| 8 | in-progress | Regime-conditioned strategy and backtest | `python scripts/run_backtest.py --help` and `pytest tests/test_backtest_accounting.py tests/test_backtest_metrics.py` |
| 9 | in-progress | Robustness analysis | `python scripts/run_robustness.py --help` |
| 10 | in-progress | Cross-market analysis | `python scripts/audit_phase10_inputs.py --help` and `python scripts/audit_phase10_final.py --help` |
| 11 | in-progress | Broker paper-signal layer | `python scripts/run_ibkr_paper_signal.py --help` and `pytest tests/test_paper_trader.py tests/test_live_order_guard.py` |
| 12 | blocked | Final report assembly and publication packaging | `python scripts/generate_backtest_diagnostics.py --help` |

Rules:

1. Update this table when a phase is frozen, unblocked, or materially re-scoped.
2. Keep phase-specific implementation details in code, scripts, and tests.
3. Use `docs/artifact_inventory.md` to track generated outputs and review substitutes.
