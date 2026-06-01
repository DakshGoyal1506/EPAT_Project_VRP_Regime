# Generated Artifact Policy

This document defines what belongs in Git and what must stay local.

## Commit These

```text
source code under src/
configuration files under configs/
scripts under scripts/
tests under tests/
documentation under docs/
README files
.env.example
pyproject.toml
.gitignore
.gitkeep placeholders
small final-report-ready summaries when explicitly approved
selected final figures when explicitly approved
```

## Do Not Commit These

```text
.env
data/raw/*
data/interim/*
data/processed/*
data/broker_cache/*
data/manual/* source downloads
reports/tables/* generated run outputs
reports/figures/* generated run figures
logs/*
*.log
*.parquet
*.pkl
*.pickle
*.joblib
*.pt
*.pth
*.duckdb
*.sqlite
*.h5
broker logs
TWS logs
private account identifiers
full generated panels
full backtest panels
full strategy signal panels
full cross-market panels
```

## Generated Directory Rules

Tracked:

```text
data/README.md
data/raw/.gitkeep
data/interim/.gitkeep
data/processed/.gitkeep
data/manual/.gitkeep
data/manual/cboe/.gitkeep
data/manual/nse/.gitkeep
data/broker_cache/.gitkeep
reports/README.md
reports/figures/.gitkeep
reports/figures/README.md
reports/tables/.gitkeep
reports/tables/README.md
```

Local-only:

```text
data/raw/*.parquet
data/interim/*.parquet
data/processed/*.parquet
data/manual/**/*.csv
data/broker_cache/*
reports/figures/*.png
reports/figures/*.svg
reports/tables/*.csv
reports/tables/*.json
```

## Exception Rule for Final Report Artifacts

A small generated artifact may be committed only if all conditions hold:

1. It is needed for final report review.
2. It is small.
3. It is non-sensitive.
4. It is stable and reproducible.
5. Its producer command is documented.
6. It does not expose broker account, order, or private runtime data.

Examples that may be approved later:

```text
reports/tables/final_summary.csv
reports/tables/final_model_comparison.csv
reports/figures/final_vrp_regime_summary.png
```

Examples that should stay local:

```text
full backtest panel parquet
full strategy signal panel parquet
full model probability panel
all raw source files
all broker signal logs
```

## Broker Artifact Rule

Broker artifacts are treated as sensitive by default.

Do not commit:

```text
account ids
paper account ids
TWS logs
broker connection logs
raw quote cache
paper order preview logs
position state
broker cache files
```

Only redacted summaries may be shared for review.
