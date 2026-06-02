# Phase 0 Artifacts

## Status

Complete / frozen.

Phase 0 artifacts are repository scaffold, governance, and reproducibility files. Phase 0 does not produce research data, model outputs, strategy signals, backtests, or broker artifacts.

## Commit Policy

Commit:

```text
README.md
pyproject.toml
.env.example
.gitignore
configs/
docs/
scripts/README.md
tests/README.md
notebooks/README.md
reports/README.md
reports/figures/README.md
reports/tables/README.md
data/README.md
.gitkeep placeholders
src/vrp/ package scaffold and module README files
```

Do not commit:

```text
real .env files
generated data
generated reports
model binaries
logs
broker cache or credentials
notebook outputs
```

## Expected Repository Artifacts

| Artifact | Path | Commit? | Review substitute |
|---|---|---:|---|
| Package metadata | `pyproject.toml` | Yes | File review |
| Environment placeholder | `.env.example` | Yes | File review |
| Generated-artifact policy | `.gitignore`, `docs/generated_artifact_policy.md`, `docs/artifact_inventory.md` | Yes | File review |
| Phase ledger | `docs/phase_status.md` | Yes | File review |
| Command index | `docs/commands.md` | Yes | File review |
| Source scaffold | `src/vrp/` | Yes | Git tree |
| Config scaffold | `configs/` | Yes | Git tree |
| Data/report placeholders | `data/**/.gitkeep`, `reports/**/.gitkeep`, README files | Yes | Git tree |

## Validation Commands

Run from the repository root:

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.log \.env"
```

Expected generated-artifact check result:

```text
.env.example
```

No real `.env`, generated parquet, model binary, or log file should be tracked.

## Reviewer Notes

Phase 0 review should focus on repository structure, generated-artifact exclusions, documentation placement, package installability, and absence of credentials or local generated outputs.
