# Phase 05 Artifacts - Threshold Regimes

## Commit Policy

Phase 5 generated artifacts are local-only by default.

Commit:

```text
code
config
tests
documentation
README files
```

Do not commit by default:

```text
data/processed/*threshold*regimes*.parquet
reports/tables/threshold_*.csv
reports/tables/threshold_*.json
reports/figures/threshold_*.png
```

## Artifact Table

| Artifact name | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute | Notes |
|---|---|---|---:|---|---|---|---|
| US threshold regime panel | `data/processed/us_threshold_regimes.parquet` | `python scripts/train_regimes.py --model threshold --market US --force` | No | Generated processed panel | `date`, `market`, `threshold_state`, `threshold_state_name`, `threshold_stress_score`, component states, availability flags, blocked reasons, trigger reasons | `head(20)`, schema printout, summary table | Local-only; regenerated from Phase 4 HAR-VRP panel |
| India threshold regime panel | `data/processed/india_threshold_regimes.parquet` | `python scripts/train_regimes.py --model threshold --market INDIA --force` | No | Generated processed panel | Same as US threshold panel | `head(20)`, schema printout, summary table | Local-only; regenerated from Phase 4 HAR-VRP panel |
| Regime summary | `reports/tables/threshold_regime_summary.csv` | `python scripts/train_regimes.py --model threshold --market ALL --force` | No by default | Generated diagnostic table | `market`, `state`, `state_name`, `n_days`, `fraction_days`, average IV/RV/VRP columns | CSV preview | Small but still generated; commit only if explicitly approved |
| Component summary | `reports/tables/threshold_component_summary.csv` | Same | No by default | Generated diagnostic table | `market`, `component`, `state_name`, availability and blocked-reason fields | CSV preview | Shows component coverage and missingness |
| Transition matrix | `reports/tables/threshold_transition_matrix.csv` | Same | No by default | Generated diagnostic table | `market`, `from_state_name`, `to_state_name`, `count`, `probability` | CSV preview | Used to inspect regime persistence |
| State duration summary | `reports/tables/threshold_state_duration_summary.csv` | Same | No by default | Generated diagnostic table | `market`, `state_name`, `n_episodes`, duration statistics | CSV preview | Used to detect flickering |
| State by year | `reports/tables/threshold_state_by_year.csv` | Same | No by default | Generated diagnostic table | `market`, `year`, `state_name`, `n_days`, `fraction_days` | CSV preview | Used to inspect era/year dependence |
| Crisis hit table | `reports/tables/threshold_crisis_hit_table.csv` | Same | No by default | Generated diagnostic table | `market`, `crisis_name`, `n_days`, `stress_fraction`, `transition_or_stress_fraction` | CSV preview | Crisis windows are reporting-only |
| Crisis lead/lag table | `reports/tables/threshold_crisis_lead_lag_table.csv` | Same | No by default | Generated diagnostic table | `first_available_window_date`, `first_stress_date`, `days_from_window_start_to_first_stress` | CSV preview | Checks timing of stress detection |
| VRP by state | `reports/tables/threshold_vrp_by_state.csv` | Same | No by default | Generated diagnostic table | `market`, `state_name`, `vrp_col`, distribution stats, `positive_ratio` | CSV preview | Shows compensation separation |
| Forward label by state | `reports/tables/threshold_forward_label_by_state.csv` | Same | No by default | Generated diagnostic table | `avg_forward_rv_label`, `avg_forward_expost_vrp_label`, `forward_vrp_positive_ratio` | CSV preview | Forward labels used only after regime assignment |
| No-lookahead audit | `reports/tables/threshold_no_lookahead_audit.csv` | Same | No by default | Generated audit table | `uses_strict_prior_thresholds`, `uses_forbidden_columns`, `forbidden_columns_used_for_construction`, `regime_available` | CSV preview and value counts | Critical review artifact |
| Regime metadata | `reports/tables/threshold_regime_metadata.json` | Same | No by default | Generated run metadata | `model_name`, `config_sha256`, `score_method`, `output_columns`, coverage fields | JSON preview | Captures reproducibility metadata |
| Final regime plot - US | `reports/figures/threshold_regimes_us.png` | Same | No by default | Generated figure | Plot of IV/RV and threshold state | Screenshot | Commit only if selected for final report |
| Final regime plot - India | `reports/figures/threshold_regimes_india.png` | Same | No by default | Generated figure | Plot of IV/RV and threshold state | Screenshot | Commit only if selected for final report |
| VRP boxplot - US | `reports/figures/threshold_regime_vrp_boxplots_us.png` | Same | No by default | Generated figure | HAR-VRP by threshold state | Screenshot | Commit only if selected for final report |
| VRP boxplot - India | `reports/figures/threshold_regime_vrp_boxplots_india.png` | Same | No by default | Generated figure | HAR-VRP by threshold state | Screenshot | Commit only if selected for final report |
| Component states - US | `reports/figures/threshold_component_states_us.png` | Same | No by default | Generated figure | Component states over time | Screenshot | Commit only if selected for final report |
| Component states - India | `reports/figures/threshold_component_states_india.png` | Same | No by default | Generated figure | Component states over time | Screenshot | Commit only if selected for final report |

## Review Substitute Commands

Run locally if artifacts exist:

```bash
python - <<'PY'
import pandas as pd

for path in [
    "data/processed/us_threshold_regimes.parquet",
    "data/processed/india_threshold_regimes.parquet",
]:
    df = pd.read_parquet(path)
    print("\n", path)
    print(df.head(20))
    print(df.dtypes)
    print(df[["threshold_state", "threshold_state_name", "threshold_regime_available"]].value_counts(dropna=False))
PY
```

Preview key reports:

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

for path in [
    "reports/tables/threshold_regime_summary.csv",
    "reports/tables/threshold_component_summary.csv",
    "reports/tables/threshold_crisis_hit_table.csv",
    "reports/tables/threshold_crisis_lead_lag_table.csv",
    "reports/tables/threshold_no_lookahead_audit.csv",
]:
    p = Path(path)
    if p.exists():
        print("\n", path)
        print(pd.read_csv(p).head(20))

p = Path("reports/tables/threshold_regime_metadata.json")
if p.exists():
    print("\n", p)
    print(json.dumps(json.loads(p.read_text()), indent=2)[:4000])
PY
```

## Sensitivity and Reproducibility Notes

1. These artifacts are generated from local market data and should not be committed by default.
2. Crisis-window diagnostics are reporting-only and must not affect threshold labels.
3. Forward/ex-post labels are diagnostic only and must not enter construction.
4. The no-lookahead audit should show no forbidden construction columns.
5. Regime panels can be regenerated from Phase 4 HAR-VRP panels and `configs/regime_threshold.yaml`.
