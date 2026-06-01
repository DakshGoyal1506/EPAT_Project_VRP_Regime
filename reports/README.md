# Reports

This directory stores generated report artifacts.

Most report outputs are generated locally and are intentionally not tracked by Git.

## Directory Contract

| Path | Purpose | Commit? |
|---|---|---:|
| `reports/tables/` | Generated CSV/JSON/Markdown diagnostic tables | No by default |
| `reports/figures/` | Generated figures | No by default |
| `reports/final_report.md` | Final report draft/output when ready | Yes when final |

## Commit Policy

Commit:

```text
reports/README.md
reports/tables/README.md
reports/tables/.gitkeep
reports/figures/README.md
reports/figures/.gitkeep
selected final-report-ready artifacts only if explicitly approved
```

Do not commit by default:

```text
reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
reports/figures/*.svg
reports/figures/*.pdf
```

## Rules

1. Generated tables must be reproducible from scripts.
2. Generated figures must be reproducible from scripts or report modules.
3. Do not manually edit generated artifacts without documenting the change.
4. Final report conclusions must trace back to reproducible outputs.
5. Broker-sensitive outputs must remain local or be redacted before review.

