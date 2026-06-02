# Phase 09 Artifacts - Strategy Signal Construction

This document lists Phase 9 generated artifacts, their producer commands, commit policy, schemas, and lightweight review substitutes.

## Commit Policy Summary

Phase 9 generated outputs are local-only by default.

Commit:

```text
code
configs
scripts
tests
docs
README files
.gitkeep placeholders
```

Do not commit:

```text
data/processed/*strategy_signals*.parquet
reports/tables/phase_9/*.csv
reports/tables/phase_9/*.json
reports/figures/phase_9/*
```

## Artifacts

| Artifact | Local path | Producer command | Commit? | Reason | Review substitute |
|---|---|---|---:|---|---|
| US strategy signal panel | `data/processed/us_strategy_signals.parquet` | `python scripts/build_signals.py --market US --strategy all --force` or `--market ALL` | No | Full generated signal panel | Printed schema, row count, strategy list, head/tail |
| India strategy signal panel | `data/processed/india_strategy_signals.parquet` | `python scripts/build_signals.py --market INDIA --strategy all --force` or `--market ALL` | No | Full generated signal panel | Printed schema, row count, strategy list, head/tail |
| Strategy signal summary | `reports/tables/phase_9/strategy_signal_summary.csv` | `python scripts/build_signals.py --market ALL --strategy all --force` | No by default | Generated diagnostic table | Small CSV preview or pasted table |
| Exposure by year | `reports/tables/phase_9/strategy_exposure_by_year.csv` | Same | No by default | Generated diagnostic table | CSV preview |
| Exposure change summary | `reports/tables/phase_9/strategy_exposure_change_summary.csv` | Same | No by default | Signal-path diagnostic, not cost/turnover model | CSV preview |
| Blocked reason summary | `reports/tables/phase_9/strategy_blocked_reason_summary.csv` | Same | No by default | Availability and decision reason audit | CSV preview |
| No-lookahead audit | `reports/tables/phase_9/strategy_no_lookahead_audit.csv` | Same | No by default | Records forbidden columns excluded before strategy construction | CSV preview |
| Metadata | `reports/tables/phase_9/strategy_metadata.json` | Same | No by default | Reproducibility metadata | JSON preview |
| Optional figures | `reports/figures/phase_9/**/*` | Future Phase 9 figure-producing diagnostics, if added | No by default | Generated figures | Screenshot or selected final-report figure only |

## Strategy Signal Panel Schema

Expected columns:

```text
market
strategy_name
regime_model
signal_observation_date
signal_available_after_close_date
target_trade_date
target_exposure
strategy_available
blocked_reason
decision_reason
state_name
p_calm
p_transition
p_stress
vrp_har_gk
har_forecast_available
source_signal_date_column
source_model
```

Long-format key:

```text
market
signal_observation_date
target_trade_date
strategy_name
```

Exposure convention:

```text
target_exposure = -1.0 -> full short-vol exposure
target_exposure =  0.0 -> flat / no exposure
```

Unavailable rows:

```text
strategy_available = False
target_exposure = NaN
decision_reason = unavailable
```

Valid flat rows:

```text
strategy_available = True
target_exposure = 0.0
blocked_reason = none
```

## Approved Strategy Names

```text
unconditional_full
threshold_hard_filter
threshold_defensive
hmm_prob_linear
hmm_prob_linear_carry
mar_prob_linear
mar_prob_linear_carry
```

No other strategy names should appear.

## No-lookahead Artifact Rules

Phase 9 artifacts must not include strategy-consumed columns derived from:

```text
forward realised variance labels
ex-post VRP labels
future/outcome/label columns
smoothed HMM/MAR probabilities
crisis-window labels
MSVOL/MSGARCH outputs
```

Allowed next-session-safe columns:

```text
hmm_*_for_next_session
mar_*_for_next_session
hmm_signal_trade_date
mar_signal_trade_date
```

## MSVOL Policy

MSVOL / MSGARCH is Phase 8 appendix-only.

Phase 9 must not:

```text
read MSVOL files
hash MSVOL files
merge MSVOL columns
build MSVOL strategies
report MSVOL state distributions
```

Expected metadata:

```text
msvol_policy = excluded_diagnostic_only
forbidden_columns_used = []
```

## Review Substitute Commands

Use these instead of committing generated artifacts:

```bash
python - <<'PY'
import pandas as pd

expected = {
    "unconditional_full",
    "threshold_hard_filter",
    "threshold_defensive",
    "hmm_prob_linear",
    "hmm_prob_linear_carry",
    "mar_prob_linear",
    "mar_prob_linear_carry",
}

for market in ["us", "india"]:
    path = f"data/processed/{market}_strategy_signals.parquet"
    df = pd.read_parquet(path)

    print("\n==", market.upper(), "==")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("strategies:", sorted(df["strategy_name"].unique()))

    assert set(df["strategy_name"].unique()) == expected

    available = df[df["strategy_available"] == True]
    unavailable = df[df["strategy_available"] == False]

    assert available["target_exposure"].notna().all()
    assert unavailable["target_exposure"].isna().all()
    assert (available["target_exposure"] >= -1.0).all()
    assert (available["target_exposure"] <= 0.0).all()

    print("available rows:", len(available))
    print("unavailable rows:", len(unavailable))
    print("min exposure:", available["target_exposure"].min())
    print("max exposure:", available["target_exposure"].max())
PY
```

No-lookahead artifact check:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

for market in ["us", "india"]:
    df = pd.read_parquet(f"data/processed/{market}_strategy_signals.parquet")

    bad_tokens = [
        "future",
        "expost",
        "ex_post",
        "smoothed",
        "diagnostic",
        "crisis",
        "msvol",
        "msgarch",
        "label",
    ]

    bad_columns = [
        col for col in df.columns
        if any(token in col.lower() for token in bad_tokens)
    ]

    print(market.upper(), "bad output columns:", bad_columns)
    assert bad_columns == []

metadata = json.loads(Path("reports/tables/phase_9/strategy_metadata.json").read_text())

assert metadata["forbidden_columns_used"] == []
assert metadata["msvol_policy"] == "excluded_diagnostic_only"
PY
```

Tracked-artifact hygiene check:

```bash
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```

Expected allowed match:

```text
.env.example
```

If generated artifacts are accidentally tracked, remove them from the Git index without deleting local files:

```powershell
git ls-files | Select-String '\.parquet$|\.pkl$|\.pickle$|\.joblib$|\.pt$|\.pth$|\.log$' | ForEach-Object { git rm --cached $_.Line }
```
