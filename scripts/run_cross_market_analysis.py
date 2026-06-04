from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.reports.cross_market import (  # noqa: E402
    CrossMarketError,
    CrossMarketInputError,
    CrossMarketLeakageError,
    build_phase13_core_panels_all_models,
    collect_locked_artifact_hashes,
    compute_cross_market_stat_tables,
    compute_logistic_diagnostic_tables,
    hash_locked_artifacts_before_after,
    load_cross_market_config,
    validate_cross_market_inputs,
    write_cross_market_metadata,
    write_cross_market_stat_tables,
    write_logistic_diagnostic_tables,
    write_phase13_panel_outputs,
)
from vrp.reports.cross_market_diagnostics import (  # noqa: E402
    generate_core_figures,
    validate_phase13_required_tables,
    write_phase13_summary_index,
)
from vrp.strategies.cross_market_overlay import (  # noqa: E402
    build_all_india_cross_market_overlays,
    validate_overlay_summary_schema,
    write_india_cross_market_overlay_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 13 cross-market US-India VRP/regime analysis."
    )

    parser.add_argument(
        "--config",
        default="configs/cross_market.yaml",
        help="Path to Phase 13 cross-market YAML config.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root. Relative config paths are resolved from this root.",
    )
    parser.add_argument(
        "--model",
        default="ALL",
        choices=["ALL", "gaussian_hmm", "markov_autoreg"],
        help="Regime model to run. Use ALL for both primary Phase 13 models.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Predictive lag setting. Phase 13 currently supports lag=1 only.",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip analysis-only India US-stress overlay construction.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PNG diagnostic figure generation.",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Only validate config and required inputs, then exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Phase 13 outputs.",
    )

    return parser.parse_args()


def _selected_models(config: dict[str, Any], model_arg: str) -> list[str]:
    if model_arg == "ALL":
        models = list(config.get("models", []))
    else:
        models = [model_arg]

    if not models:
        raise RuntimeError("No Phase 13 models selected.")

    supported = {"gaussian_hmm", "markov_autoreg"}
    bad = sorted(set(models) - supported)
    if bad:
        raise RuntimeError(f"Unsupported model(s): {bad}")

    return models


def _configured_output_paths(config: dict[str, Any]) -> list[str]:
    outputs = config.get("outputs", {})
    figures = config.get("figures", {})

    paths: list[str] = []
    for value in outputs.values():
        if isinstance(value, str) and (
            value.endswith(".csv")
            or value.endswith(".json")
            or value.endswith(".parquet")
        ):
            paths.append(value)

    for value in figures.values():
        if isinstance(value, str):
            paths.append(value)

    return paths


def _check_existing_outputs(
    config: dict[str, Any],
    root: Path,
    *,
    force: bool,
) -> None:
    if force:
        return

    existing: list[str] = []
    for rel_path in _configured_output_paths(config):
        abs_path = root / rel_path
        if abs_path.exists():
            existing.append(str(abs_path))

    if existing:
        message = (
            "Refusing to overwrite existing Phase 13 outputs without --force. "
            "Existing outputs:\n"
            + "\n".join(existing[:50])
        )
        if len(existing) > 50:
            message += f"\n... and {len(existing) - 50} more"
        raise RuntimeError(message)


def _combine_written_outputs(*parts: dict[str, str]) -> dict[str, str]:
    combined: dict[str, str] = {}
    for part in parts:
        combined.update({str(k): str(v) for k, v in part.items()})
    return combined


def _write_run_status(
    *,
    root: Path,
    config: dict[str, Any],
    status: str,
    reason: str,
) -> None:
    outputs = config.get("outputs", {})
    tables_dir = root / str(outputs.get("tables_dir", "reports/tables/phase_13"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    path = tables_dir / "phase13_run_status.json"
    payload = {
        "phase": config.get("phase"),
        "name": config.get("name"),
        "status": status,
        "reason": reason,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _print_written_outputs(written_outputs: dict[str, str]) -> None:
    print("\nPhase 13 written outputs:")
    for key in sorted(written_outputs):
        print(f"  {key}: {written_outputs[key]}")


def run_phase13(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    config = load_cross_market_config(config_path)
    models = _selected_models(config, args.model)

    validate_cross_market_inputs(
        config,
        root=root,
        require_exists=True,
    )

    if args.validate_inputs_only:
        print("Phase 13 config/input validation passed.")
        print("Selected models:", ", ".join(models))
        return 0

    if args.lag != 1:
        raise RuntimeError("Phase 13 currently supports --lag 1 only.")

    _check_existing_outputs(
        config,
        root=root,
        force=bool(args.force),
    )

    locked_before = collect_locked_artifact_hashes(config, root=root)

    core = build_phase13_core_panels_all_models(
        config=config,
        root=root,
        models=models,
        lag_days=args.lag,
    )

    descriptive_panel = core["descriptive_panel"]
    predictive_panel = core["predictive_panel"]
    combined_panel = core["combined_panel"]
    alignment_audit = core["alignment_audit"]
    no_lookahead_audit = core["no_lookahead_audit"]

    panel_outputs = write_phase13_panel_outputs(
        descriptive_panel=descriptive_panel,
        predictive_panel=predictive_panel,
        combined_panel=combined_panel,
        alignment_audit=alignment_audit,
        no_lookahead_audit=no_lookahead_audit,
        config=config,
        root=root,
    )

    stat_tables = compute_cross_market_stat_tables(
        descriptive_panel=descriptive_panel,
        predictive_panel=predictive_panel,
        config=config,
    )

    stat_outputs = write_cross_market_stat_tables(
        tables=stat_tables,
        config=config,
        root=root,
    )

    logistic_tables = compute_logistic_diagnostic_tables(
        predictive_panel=predictive_panel,
        config=config,
    )

    logistic_outputs = write_logistic_diagnostic_tables(
        tables=logistic_tables,
        config=config,
        root=root,
    )

    overlay_panel = pd.DataFrame()
    overlay_summary = pd.DataFrame()
    overlay_outputs: dict[str, str] = {}

    if not args.skip_overlay:
        overlay_built = build_all_india_cross_market_overlays(
            predictive_panel=predictive_panel,
            config=config,
            root=root,
            models=models,
        )
        overlay_panel = overlay_built.get("india_overlay_panel", pd.DataFrame())
        overlay_summary = overlay_built.get("overlay_summary", pd.DataFrame())

        if not overlay_summary.empty:
            validate_overlay_summary_schema(overlay_summary, config)

        overlay_outputs = write_india_cross_market_overlay_outputs(
            overlay_outputs=overlay_built,
            config=config,
            root=root,
        )

    required_tables: dict[str, pd.DataFrame] = {}
    required_tables.update(
        {
            "alignment_audit": alignment_audit,
            "no_lookahead_audit": no_lookahead_audit,
        }
    )
    required_tables.update(stat_tables)
    required_tables.update(logistic_tables)

    validate_phase13_required_tables(
        required_tables,
        config=config,
    )

    figure_outputs: dict[str, str] = {}
    if not args.skip_figures:
        figures = generate_core_figures(
            descriptive_panel=descriptive_panel,
            predictive_panel=predictive_panel,
            overlay_panel=overlay_panel if not overlay_panel.empty else None,
            config=config,
            root=root,
        )
        figure_outputs = {k: str(v) for k, v in figures.items()}

    summary_tables: dict[str, pd.DataFrame] = dict(required_tables)
    if not overlay_summary.empty:
        summary_tables["overlay_summary"] = overlay_summary

    summary_index = write_phase13_summary_index(
        tables=summary_tables,
        figures=figure_outputs,
        config=config,
        root=root,
    )

    all_outputs = _combine_written_outputs(
        panel_outputs,
        stat_outputs,
        logistic_outputs,
        overlay_outputs,
        figure_outputs,
        {
            "phase13_summary_index": str(
                root / config["outputs"]["phase13_summary_index"]
            )
        },
    )

    metadata_path = write_cross_market_metadata(
        config=config,
        outputs=all_outputs,
        root=root,
    )
    all_outputs["phase13_metadata"] = str(metadata_path)

    locked_after = collect_locked_artifact_hashes(config, root=root)
    hash_locked_artifacts_before_after(locked_before, locked_after)

    _write_run_status(
        root=root,
        config=config,
        status="ok",
        reason="",
    )

    _print_written_outputs(all_outputs)

    print("\nPhase 13 completed.")
    print(f"Models: {', '.join(models)}")
    print(f"Descriptive rows: {len(descriptive_panel)}")
    print(f"Predictive rows: {len(predictive_panel)}")
    print(f"Alignment audit rows: {len(alignment_audit)}")
    print(
        "Logistic summary rows: "
        f"{len(logistic_tables.get('logistic_model_summary', []))}"
    )

    if not overlay_summary.empty:
        print(f"Overlay summary rows: {len(overlay_summary)}")
    else:
        print("Overlay skipped or empty.")

    print(f"Summary index rows: {len(summary_index)}")

    return 0


def main() -> int:
    args = _parse_args()

    try:
        return run_phase13(args)
    except (CrossMarketError, CrossMarketInputError, CrossMarketLeakageError) as exc:
        root = Path(args.root).resolve()
        try:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = root / config_path
            config = load_cross_market_config(config_path)
            _write_run_status(
                root=root,
                config=config,
                status="failed",
                reason=str(exc),
            )
        except Exception:
            pass

        print(f"Phase 13 failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        root = Path(args.root).resolve()
        try:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = root / config_path
            config = load_cross_market_config(config_path)
            _write_run_status(
                root=root,
                config=config,
                status="failed",
                reason=str(exc),
            )
        except Exception:
            pass

        print(f"Phase 13 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
