# Phase Status

This file is the authoritative phase-status ledger for the EPAT VRP regime project.

The top-level `README.md` gives the project overview. This file tracks phase status, scope, validation commands, and freeze readiness.

## Status Legend

| Status | Meaning |
|---|---|
| `complete / frozen` | Implemented, validated, and should not be changed except for bug, safety, or documentation fixes. |
| `complete / needs final review` | Implemented, but needs final artifact/doc/test review before freezing. |
| `in progress` | Under active development or cleanup. |
| `blocked` | Waiting on prior phases, data, review, or design decision. |
| `not started` | Planned but not implemented. |

## Phase Ledger

| Phase | Status | Scope | Primary validation command |
|---|---|---|---|
| 0 | complete / frozen | Repo scaffold, package structure, environment setup, configs, `.gitignore`, README, governance docs, source/data/report directory policy | `pip install -e .` and `pytest` |
| 1 | complete / frozen | Public daily data ingestion: CBOE VIX, FRED VIXCLS, Yahoo OHLCV, NSE/manual override support, canonical OHLCV schema, data audit | `python scripts/download_data.py --dry-run` and `pytest tests/test_data_loaders.py tests/test_data_schema.py` |
| 2 | complete / frozen | Realised variance estimators and RV panels | `python scripts/build_features.py --market ALL --feature rv --window 22` and `pytest tests/test_rv_estimators.py` |
| 3 | complete / frozen | Implied variance, IV/RV alignment, VRP construction, no-lookahead labels | `python scripts/build_features.py --market ALL --feature iv` and `python scripts/build_features.py --market ALL --feature vrp` |
| 4 | complete / frozen | HAR-RV forecasting and HAR-based prospective VRP | `python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none` |
| 5 | complete / frozen | Threshold baseline regimes with strict-prior rolling thresholds, component blocked reasons, trigger diagnostics, reporting-only crisis windows, and no-lookahead audit | `python scripts/train_regimes.py --help` and `pytest tests/test_threshold_regimes.py tests/test_regime_no_lookahead.py tests/test_no_lookahead.py` |
| 6 | complete / frozen | Gaussian HMM latent-regime baseline with train-only scaling/fitting, custom filtered probabilities, diagnostic-only smoothed probabilities, train-period economic state mapping, candidate validation, and no-lookahead audit | `pytest tests/test_hmm_filtering.py tests/test_hmm_scaling.py tests/test_hmm_model.py tests/test_hmm_no_lookahead.py` |
| 7 | complete / frozen | Markov autoregression regime layer with train-only fitting, train-only target transform, full-series filtering using train-fitted parameters, filtered-probability signal timing, diagnostic-only smoothing, and no-lookahead audit | `python scripts/train_markov_autoreg.py --help` and `pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py` |
| 8 | complete / frozen | Python-only MSVOL robustness appendix; diagnostic-only return-volatility regime comparison; true R MSGARCH kept optional/future only | `python scripts/run_msvol_no_lookahead_audit.py --help` and `pytest tests/test_msgarch_export.py tests/test_msvol_model.py tests/test_msvol_adapter.py tests/test_msvol_diagnostics.py tests/test_msvol_no_lookahead.py` |
| 9 | complete / frozen | Strategy signal construction: fixed seven-strategy universe, long-format next-session exposure intentions, carry-aware HAR-VRP gate, signal diagnostics, and no-lookahead audit | `python scripts/build_signals.py --help` and `pytest tests/test_exposure_rules.py tests/test_signal_builder.py tests/test_strategy_no_lookahead.py tests/test_phase9_diagnostics.py` |
| 10 | complete / needs final review | Vectorised research backtest and robustness | `python scripts/run_backtest.py --help` and `python scripts/run_robustness.py --help` |
| 11 | complete / frozen | IBKR paper-signal readiness layer; no live orders | `python scripts/run_ibkr_paper_signal.py --help` and `python scripts/validate_phase11.py --help` |
| 12 | not started | Optional future IBKR paper execution adapter | Explicit re-scope required before implementation |
| 13 | not started | Cross-market US-India analysis | Starts after Phase 0-12 cleanup baseline |
| 14 | blocked | Final report / release package | Wait for frozen Phase 0-13 baseline |

## Frozen Phase Rules

Once a phase is marked `complete / frozen`:

1. Do not rewrite working logic.
2. Do not move phase boundaries.
3. Do not add new research logic to that phase.
4. Allow only:
	- correctness bug fixes,
	- safety/no-lookahead fixes,
	- reproducibility fixes,
	- documentation updates,
	- tests for existing behaviour.

## Global Non-Negotiables

1. Production logic lives in `src/vrp/`.
2. Notebooks are inspection-only.
3. Generated panels stay local unless explicitly approved as small release artifacts.
4. No private credentials or broker account identifiers are committed.
5. Backtests must not use future realised variance as a tradable signal.
6. HMM/AR-HMM backtests must use filtered probabilities available at time `t`, not full-sample smoothed probabilities.
7. Broker layer remains paper/signal-only unless explicitly re-scoped later.
