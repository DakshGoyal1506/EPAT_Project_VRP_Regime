# Artifact Inventory

This file tracks the expected generated outputs by phase and whether they are intended to be committed.

Rule:

Full panels stay local. Small summary CSV/JSON may be committed only if stable and useful. For now, keep generated summaries local unless a final-report artifact is selected deliberately.

| Phase | Artifact | Path | Producer command | Commit? | Reason | Review substitute |
|---|---|---|---|---|---|---|
| 1 | US VIX raw CBOE | data/raw/us_vix_cboe.parquet | `python scripts/download_data.py --market US --source cboe --force` | No | Generated data | data_audit.csv row |
| 1 | India VIX raw NSE | data/raw/india_vix_nse.parquet | `python scripts/download_data.py --market INDIA --source nse --force` | No | Generated data | data_audit.csv row |
| 1 | Data audit table | reports/tables/data_audit.csv | `python scripts/download_data.py --dry-run` or the ingestion audit path | Optional small | Summary artifact | CSV or screenshot |
| 2 | Realised variance panel | data/processed/us_rv.parquet | `python scripts/build_features.py --market US --feature rv --window 22` | No | Generated data | feature summary table |
| 2 | Realised variance panel | data/processed/india_rv.parquet | `python scripts/build_features.py --market INDIA --feature rv --window 22` | No | Generated data | feature summary table |
| 3 | Implied variance panel | data/processed/us_iv.parquet | `python scripts/build_features.py --market ALL --feature iv` | No | Generated data | alignment summary |
| 3 | VRP panel | data/processed/us_vrp.parquet | `python scripts/build_features.py --market ALL --feature vrp` | No | Generated data | VRP summary table |
| 4 | HAR forecast panel | data/processed/har_forecasts.parquet | `python scripts/train_har.py --market ALL --mode expanding --force` | No | Generated model output | forecast diagnostics |
| 8 | Backtest diagnostics | reports/tables/backtest_diagnostics.csv | `python scripts/run_backtest.py --help` then the backtest run command | Optional small | Reviewer-friendly summary | CSV or screenshot |
| 11 | Paper-signal summary | reports/tables/paper_signal_summary.csv | `python scripts/run_ibkr_paper_signal.py --help` then the paper-signal run command | No | Broker-sensitive runtime output | log excerpt |
