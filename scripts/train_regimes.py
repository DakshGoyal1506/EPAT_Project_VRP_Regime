from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp.regimes.threshold import (  # noqa: E402
    build_threshold_regime_panel,
    load_threshold_config,
)
from vrp.reports.regime_diagnostics import (  # noqa: E402
    build_threshold_component_summary,
    build_threshold_crisis_hit_table,
    build_threshold_crisis_lead_lag_table,
    build_threshold_forward_label_by_state_table,
    build_threshold_no_lookahead_audit,
    build_threshold_regime_summary,
    build_threshold_state_by_year_table,
    build_threshold_state_duration_summary,
    build_threshold_transition_matrix,
    build_threshold_vrp_by_state_table,
    plot_threshold_component_states,
    plot_threshold_regime_vrp_boxplots,
    plot_threshold_regimes,
    write_threshold_metadata,
)


DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

VALID_MARKETS = ("US", "INDIA", "ALL")
VALID_MODELS = ("threshold",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/build Phase 5 regime models."
    )

    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        required=True,
        help="Regime model to build. Phase 5 supports only 'threshold'.",
    )

    parser.add_argument(
        "--market",
        choices=VALID_MARKETS,
        required=True,
        help="Market to build: US, INDIA, or ALL.",
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "regime_threshold.yaml"),
        help="Path to threshold-regime config YAML.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed threshold-regime parquet outputs.",
    )

    parser.add_argument(
        "--write-reports",
        dest="write_reports",
        action="store_true",
        default=True,
        help="Write report tables and figures. Default: true.",
    )

    parser.add_argument(
        "--no-write-reports",
        dest="write_reports",
        action="store_false",
        help="Skip report tables and figures.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.model != "threshold":
            raise ValueError(
                f"Unsupported model {args.model!r}. Phase 5 supports only 'threshold'."
            )

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

        config = load_threshold_config(config_path)

        markets = _resolve_markets(args.market)

        _ensure_output_dirs()

        panels = _build_and_write_market_panels(
            markets=markets,
            config=config,
            force=args.force,
        )

        if args.write_reports:
            _write_reports(
                panels=panels,
                config=config,
                config_path=config_path,
            )

        _print_success_summary(
            markets=markets,
            panels=panels,
            write_reports=args.write_reports,
        )

        return 0

    except Exception as exc:
        print(f"[ERROR] Phase 5 regime build failed: {exc}", file=sys.stderr)
        return 1


def _resolve_markets(market_arg: str) -> List[str]:
    market = str(market_arg).upper()

    if market == "ALL":
        return ["US", "INDIA"]

    if market not in {"US", "INDIA"}:
        raise ValueError(f"Invalid market {market_arg!r}. Expected US, INDIA, or ALL.")

    return [market]


def _ensure_output_dirs() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _build_and_write_market_panels(
    markets: Iterable[str],
    config: Mapping,
    force: bool,
) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}

    for market in markets:
        output_path = _output_path_for_market(market, config)

        if output_path.exists() and not force:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Use --force to overwrite."
            )

        print(f"[INFO] Building threshold regimes for {market}...")

        panel = build_threshold_regime_panel(market=market, config=config)

        if panel.empty:
            raise ValueError(f"{market} threshold regime panel is empty.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path, index=False)

        print(f"[INFO] Wrote {market} threshold regimes: {output_path}")
        print(f"[INFO] {market} rows: {len(panel):,}")

        available_count = int(panel["threshold_regime_available"].sum())
        available_fraction = available_count / len(panel) if len(panel) else 0.0

        print(
            "[INFO] "
            f"{market} available regimes: {available_count:,} "
            f"({available_fraction:.2%})"
        )

        _warn_if_low_coverage(
            market=market,
            panel=panel,
            config=config,
        )

        panels[market] = panel

    return panels


def _write_reports(
    panels: Mapping[str, pd.DataFrame],
    config: Mapping,
    config_path: Path,
) -> None:
    if not panels:
        raise ValueError("No panels supplied for report generation.")

    combined = pd.concat(list(panels.values()), ignore_index=True)

    if combined.empty:
        raise ValueError("Combined threshold regime panel is empty.")

    print("[INFO] Writing Phase 5 report tables...")

    _write_csv(
        build_threshold_regime_summary(combined),
        REPORT_TABLE_DIR / "threshold_regime_summary.csv",
    )

    _write_csv(
        build_threshold_component_summary(combined),
        REPORT_TABLE_DIR / "threshold_component_summary.csv",
    )

    _write_csv(
        build_threshold_transition_matrix(combined),
        REPORT_TABLE_DIR / "threshold_transition_matrix.csv",
    )

    _write_csv(
        build_threshold_state_duration_summary(combined),
        REPORT_TABLE_DIR / "threshold_state_duration_summary.csv",
    )

    _write_csv(
        build_threshold_state_by_year_table(combined),
        REPORT_TABLE_DIR / "threshold_state_by_year.csv",
    )

    crisis_windows = config.get("crisis_windows", {})
    skip_windows_outside_sample = config.get("diagnostic_window_policy", {}).get(
        "skip_windows_outside_sample", True
    )

    _write_csv(
        build_threshold_crisis_hit_table(
            combined, crisis_windows, skip_windows_outside_sample=skip_windows_outside_sample
        ),
        REPORT_TABLE_DIR / "threshold_crisis_hit_table.csv",
    )

    _write_csv(
        build_threshold_crisis_lead_lag_table(
            combined, crisis_windows, skip_windows_outside_sample=skip_windows_outside_sample
        ),
        REPORT_TABLE_DIR / "threshold_crisis_lead_lag_table.csv",
    )

    _write_csv(
        build_threshold_vrp_by_state_table(combined),
        REPORT_TABLE_DIR / "threshold_vrp_by_state.csv",
    )

    _write_csv(
        build_threshold_forward_label_by_state_table(combined),
        REPORT_TABLE_DIR / "threshold_forward_label_by_state.csv",
    )

    _write_csv(
        build_threshold_no_lookahead_audit(combined),
        REPORT_TABLE_DIR / "threshold_no_lookahead_audit.csv",
    )

    print("[INFO] Writing Phase 5 metadata...")

    input_paths = {
        market: config["primary_input_files"][market]
        for market in panels.keys()
    }

    metadata_extra = {
        "config_path": str(config_path.relative_to(PROJECT_ROOT))
        if config_path.is_relative_to(PROJECT_ROOT)
        else str(config_path),
        "output_files": {
            market: str(_output_path_for_market(market, config))
            for market in panels.keys()
        },
    }

    write_threshold_metadata(
        config=config,
        output_path=REPORT_TABLE_DIR / "threshold_regime_metadata.json",
        panels=panels,
        input_paths=input_paths,
        extra=metadata_extra,
    )

    print("[INFO] Writing Phase 5 figures...")

    for market, panel in panels.items():
        market_lower = market.lower()

        fig = plot_threshold_regimes(
            panel,
            market=market,
            output_path=REPORT_FIGURE_DIR / f"threshold_regimes_{market_lower}.png",
        )
        _close_figure(fig)

        fig = plot_threshold_regime_vrp_boxplots(
            panel,
            market=market,
            output_path=(
                REPORT_FIGURE_DIR
                / f"threshold_regime_vrp_boxplots_{market_lower}.png"
            ),
        )
        _close_figure(fig)

        fig = plot_threshold_component_states(
            panel,
            market=market,
            output_path=(
                REPORT_FIGURE_DIR
                / f"threshold_component_states_{market_lower}.png"
            ),
        )
        _close_figure(fig)

    print("[INFO] Report generation complete.")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        raise ValueError(f"Cannot write None as CSV: {path}")

    df.to_csv(path, index=False)
    print(f"[INFO] Wrote table: {path}")


def _output_path_for_market(market: str, config: Mapping) -> Path:
    market = str(market).upper()

    configured_outputs = config.get("primary_output_files", {})

    if market in configured_outputs:
        output_path = Path(configured_outputs[market])
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        return output_path

    fallback_name = f"{market.lower()}_threshold_regimes.parquet"
    return DATA_PROCESSED_DIR / fallback_name


def _warn_if_low_coverage(
    market: str,
    panel: pd.DataFrame,
    config: Mapping,
) -> None:
    if panel.empty:
        return

    threshold_policy = config.get("threshold_policy", {})
    sample_coverage_policy = config.get("sample_coverage_policy", {})

    min_history = int(threshold_policy.get("min_history", 0))
    warn_cutoff = float(
        sample_coverage_policy.get("warn_if_available_fraction_after_warmup_below", 0.0)
    )

    after_warmup = panel.iloc[min_history:].copy() if len(panel) > min_history else panel

    if after_warmup.empty:
        print(
            f"[WARN] {market} has no rows after warmup length {min_history}.",
            file=sys.stderr,
        )
        return

    available_fraction = float(after_warmup["threshold_regime_available"].mean())

    if available_fraction < warn_cutoff:
        print(
            "[WARN] "
            f"{market} available regime fraction after warmup is "
            f"{available_fraction:.2%}, below configured warning cutoff "
            f"{warn_cutoff:.2%}.",
            file=sys.stderr,
        )


def _close_figure(fig) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def _print_success_summary(
    markets: Iterable[str],
    panels: Mapping[str, pd.DataFrame],
    write_reports: bool,
) -> None:
    print("[INFO] Phase 5 threshold-regime build completed.")

    for market in markets:
        panel = panels[market]

        n_rows = len(panel)
        n_available = int(panel["threshold_regime_available"].sum())
        first_date = pd.to_datetime(panel["date"]).min()
        last_date = pd.to_datetime(panel["date"]).max()

        print(
            "[INFO] "
            f"{market}: rows={n_rows:,}, available={n_available:,}, "
            f"date_range={first_date.date()} to {last_date.date()}"
        )

    if write_reports:
        print(f"[INFO] Report tables directory: {REPORT_TABLE_DIR}")
        print(f"[INFO] Report figures directory: {REPORT_FIGURE_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())