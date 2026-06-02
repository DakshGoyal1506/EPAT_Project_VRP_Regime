# Phase Documentation

This folder contains detailed phase-level documentation.

The top-level `README.md` gives only the project overview.  
`docs/phase_status.md` gives the phase ledger.  
This folder gives the detailed implementation record for each phase.

## Phase Files

| Phase | File | Scope |
|---:|---|---|
| 0 | `phase_00_scaffold_governance.md` | Repo scaffold, environment, governance |
| 1 | `phase_01_data_ingestion.md` | Public data ingestion |
| 2 | `phase_02_realised_variance.md` | Realised variance |
| 3 | `phase_03_implied_variance_vrp.md` | Implied variance and VRP |
| 4 | `phase_04_har_rv.md` | HAR-RV forecasting and HAR-based prospective VRP |
| 5 | `phase_05_threshold_regimes.md` | Threshold regimes |
| 6 | `phase_06_gaussian_hmm.md` | Gaussian HMM |
| 7 | `phase_07_markov_autoreg.md` | Markov autoregression |
| 8 | `phase_08_msvol_appendix.md` | Python-only MSVOL robustness appendix |
| 9 | `phase_09_strategy_signals.md` | Strategy signal construction |
| 10 | `phase_10_backtest.md` | Vectorised research backtest and robustness |
| 11 | `phase_11_broker_paper_signal.md` | IBKR paper-signal readiness layer |
| 12 | `phase_12_paper_execution_adapter.md` | Optional future IBKR paper execution adapter |
| 13 | `phase_13_cross_market.md` | Cross-market US-India analysis |
| 14 | `phase_14_final_report_release.md` | Final report / release package |

## Rules

1. Phase docs describe what was implemented, not new research logic.
2. Generated artifacts are documented but not committed unless explicitly approved.
3. Commands in phase docs must be runnable from the repository root.
4. Phase docs must state inputs, outputs, tests, and review packet.
5. Phase docs must preserve no-lookahead and no-live-trading rules.
