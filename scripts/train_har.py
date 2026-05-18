# scripts/train_har.py

"""
Train Phase 4 HAR-RV forecasts and construct HAR-based prospective VRP.

Main commands:
    python scripts/train_har.py --market US --mode expanding
    python scripts/train_har.py --market INDIA --mode expanding
    python scripts/train_har.py --market ALL --mode expanding

Outputs:
    data/processed/us_har_forecast.parquet
    data/processed/india_har_forecast.parquet
    data/processed/us_vrp_har.parquet
    data/processed/india_vrp_har.parquet

Report outputs, when --write-reports is enabled:
    reports/tables/har_forecast_accuracy.csv
    reports/tables/har_coefficients.csv
    reports/tables/har_vrp_summary.csv
    reports/tables/har_metadata.json
    reports/tables/har_no_lookahead_audit.csv
    reports/figures/har_forecast_us.png
    reports/figures/har_forecast_india.png
    reports/figures/har_residuals_us.png
    reports/figures/har_residuals_india.png
    reports/figures/har_vrp_us.png
    reports/figures/har_vrp_india.png

Core no-lookahead rule:
    For forecast date t, training row s is allowed only when:
        target_end_date_s < t
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.features.feature_io import (  # noqa: E402
    ensure_parent_dir,
    load_feature_panel,
    save_feature_panel,
)
from vrp.features.vrp import compute_har_vrp  # noqa: E402
from vrp.forecasting.har_rv import (  # noqa: E402
    HARConfig,
    expanding_window_har_forecast,
    load_har_config,
    rolling_window_har_forecast,
    resolve_compute_backend,
    resolve_torch_device,
    resolve_torch_dtype,
)
from vrp.forecasting.forecast_evaluation import (  # noqa: E402
    build_forecast_accuracy_table,
)
from vrp.forecasting.har_registry import (  # noqa: E402
    HAR_BASELINE_COLUMNS,
    HAR_FEATURE_COLUMNS,
    HAR_FORECAST_COLUMNS,
    HAR_OUTPUT_FEATURE_COLUMNS,
    HAR_TARGET_COLUMNS,
    assert_har_registry_is_valid,
)


DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


MARKET_CONFIG = {
    "US": {
        "market": "US",
        "vrp_input_path": DATA_PROCESSED_DIR / "us_vrp.parquet",
        "har_forecast_output_path": DATA_PROCESSED_DIR / "us_har_forecast.parquet",
        "vrp_har_output_path": DATA_PROCESSED_DIR / "us_vrp_har.parquet",
    },
    "INDIA": {
        "market": "INDIA",
        "vrp_input_path": DATA_PROCESSED_DIR / "india_vrp.parquet",
        "har_forecast_output_path": DATA_PROCESSED_DIR / "india_har_forecast.parquet",
        "vrp_har_output_path": DATA_PROCESSED_DIR / "india_vrp_har.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train HAR-RV forecasts and build HAR-based VRP panels."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        required=True,
        help="Market to process.",
    )

    parser.add_argument(
        "--mode",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Walk-forward mode. Default: expanding.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/har_rv.yaml",
        help="Path to HAR config YAML. Default: configs/har_rv.yaml.",
    )

    parser.add_argument(
        "--min-train-observations",
        type=int,
        default=None,
        help="Override config min_train_observations.",
    )

    parser.add_argument(
        "--rolling-train-window",
        type=int,
        default=None,
        help="Override config rolling_train_window.",
    )

    parser.add_argument(
        "--hac-maxlags",
        type=int,
        default=None,
        help="Override config hac_maxlags.",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cpu_statsmodels", "cpu_numpy_batched", "torch_batched"],
        default=None,
        help="Compute backend for HAR OLS. Default: use config value or auto.",
    )

    parser.add_argument(
        "--torch-device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Torch device selection when using torch_batched backend.",
    )

    parser.add_argument(
        "--torch-dtype",
        choices=["float64", "float32"],
        default=None,
        help="Torch dtype for batched computation.",
    )

    parser.add_argument(
        "--coefficient-hac-frequency",
        choices=["daily", "month_end", "quarter_end", "final", "none"],
        default=None,
        help="Frequency for HAC coefficient inference checkpoints.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    report_group = parser.add_mutually_exclusive_group()

    report_group.add_argument(
        "--write-reports",
        dest="write_reports",
        action="store_true",
        default=True,
        help="Write HAR diagnostics and report tables. Default: true.",
    )

    report_group.add_argument(
        "--no-write-reports",
        dest="write_reports",
        action="store_false",
        help="Skip HAR diagnostics and report tables.",
    )

    return parser.parse_args()


def markets_to_process(market_arg: str) -> list[str]:
    """
    Resolve CLI market argument into market list.
    """
    if market_arg == "ALL":
        return ["US", "INDIA"]

    return [market_arg]


def _resolve_path(path_value: str | Path) -> Path:
    """
    Resolve relative paths from project root.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _path_from_config_or_default(
    config: HARConfig,
    *,
    market: str,
    mapping_name: str,
    default_path: Path,
) -> Path:
    """
    Resolve a market path from config mapping, falling back to MARKET_CONFIG.
    """
    mapping = getattr(config, mapping_name)

    if isinstance(mapping, dict) and market in mapping:
        return _resolve_path(mapping[market])

    return default_path


def apply_cli_overrides(config: HARConfig, args: argparse.Namespace) -> HARConfig:
    """
    Apply CLI overrides to HARConfig.
    """
    updates: dict[str, object] = {
        "oos_mode": args.mode,
    }

    if args.min_train_observations is not None:
        if args.min_train_observations < 1:
            raise ValueError("--min-train-observations must be >= 1")
        updates["min_train_observations"] = args.min_train_observations

    if args.rolling_train_window is not None:
        if args.rolling_train_window < 1:
            raise ValueError("--rolling-train-window must be >= 1")
        updates["rolling_train_window"] = args.rolling_train_window

    if args.hac_maxlags is not None:
        if args.hac_maxlags < 0:
            raise ValueError("--hac-maxlags must be >= 0")
        updates["hac_maxlags"] = args.hac_maxlags

    # Backend overrides
    if getattr(args, "backend", None) is not None:
        updates["compute_backend"] = args.backend

    if getattr(args, "torch_device", None) is not None:
        updates["torch_device"] = args.torch_device

    if getattr(args, "torch_dtype", None) is not None:
        updates["torch_dtype"] = args.torch_dtype

    if getattr(args, "coefficient_hac_frequency", None) is not None:
        updates["coefficient_hac_frequency"] = args.coefficient_hac_frequency

    return config.model_copy(update=updates)


def _require_existing_input(path: Path, *, description: str) -> None:
    """
    Raise if required input path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _check_output_write_allowed(path: Path, *, force: bool) -> None:
    """
    Prevent accidental overwrite unless --force is supplied.
    """
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Use --force to overwrite."
        )


def _validate_market_panel(panel: pd.DataFrame, *, market: str, name: str) -> None:
    """
    Validate a one-market input/output panel.
    """
    if "market" not in panel.columns:
        raise ValueError(f"{name} is missing required column: market")

    values = panel["market"].dropna().astype(str).str.upper().unique().tolist()

    if values != [market]:
        raise ValueError(
            f"{name} must contain only market={market}. Found: {values}"
        )


def _print_config_summary(config: HARConfig, selected_markets: list[str]) -> None:
    """
    Print compact run summary.
    """
    print("[har] Phase 4 HAR-RV forecast run")
    print(f"[har] markets: {selected_markets}")
    print(f"[har] mode: {config.oos_mode}")
    print(f"[har] target: {config.primary_forward_label_col}")
    print(f"[har] daily RV: {config.primary_daily_rv_col}")
    print(f"[har] features: {HAR_FEATURE_COLUMNS}")
    print(f"[har] min_train_observations: {config.min_train_observations}")
    print(f"[har] rolling_train_window: {config.rolling_train_window}")
    print(f"[har] hac_maxlags: {config.hac_maxlags}")
    print(f"[har] forecast_floor: {config.forecast_floor}")
    # Backend info (config values)
    print(f"[har] compute_backend: {config.compute_backend}")
    print(f"[har] torch_device: {config.torch_device}")
    print(f"[har] torch_dtype: {config.torch_dtype}")
    print(f"[har] coefficient_hac_frequency: {config.coefficient_hac_frequency}")
    # Backend info (resolved values)
    try:
        resolved_backend = resolve_compute_backend(config)
        print(f"[har] resolved_compute_backend: {resolved_backend}")
    except Exception as e:
        print(f"[har] resolved_compute_backend: ERROR ({e})")
    
    try:
        resolved_device = resolve_torch_device(config)
        print(f"[har] resolved_torch_device: {resolved_device}")
    except Exception as e:
        print(f"[har] resolved_torch_device: ERROR ({e})")
    
    try:
        resolved_dtype = resolve_torch_dtype(config)
        print(f"[har] resolved_torch_dtype: {resolved_dtype}")
    except Exception as e:
        print(f"[har] resolved_torch_dtype: ERROR ({e})")
    
    print(
        "[har] timing rule: training row s allowed only when "
        "target_end_date_s < forecast_date_t"
    )


def _run_har_forecast_for_market(
    vrp_panel: pd.DataFrame,
    *,
    config: HARConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run expanding or rolling HAR forecast for one market.
    """
    if config.oos_mode == "expanding":
        return expanding_window_har_forecast(vrp_panel, config)

    if config.oos_mode == "rolling":
        return rolling_window_har_forecast(vrp_panel, config)

    raise ValueError(
        f"Unsupported HAR mode: {config.oos_mode}. "
        "Expected expanding or rolling."
    )


def build_market_outputs(
    market: str,
    *,
    config: HARConfig,
    force: bool,
) -> dict[str, pd.DataFrame]:
    """
    Build and write Phase 4 outputs for one market.

    Returns
    -------
    dict
        {
            "vrp": original Phase 3 VRP panel,
            "forecast": HAR forecast panel,
            "coefficients": coefficient history,
            "audit": no-lookahead audit,
            "vrp_har": HAR-VRP panel,
        }
    """
    market_key = str(market).upper()

    if market_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market: {market_key}")

    market_defaults = MARKET_CONFIG[market_key]

    vrp_input_path = _path_from_config_or_default(
        config,
        market=market_key,
        mapping_name="input_paths",
        default_path=market_defaults["vrp_input_path"],
    )

    forecast_output_path = _path_from_config_or_default(
        config,
        market=market_key,
        mapping_name="forecast_output_paths",
        default_path=market_defaults["har_forecast_output_path"],
    )

    vrp_har_output_path = _path_from_config_or_default(
        config,
        market=market_key,
        mapping_name="vrp_har_output_paths",
        default_path=market_defaults["vrp_har_output_path"],
    )

    _require_existing_input(
        vrp_input_path,
        description=f"Phase 3 VRP input for {market_key}",
    )

    _check_output_write_allowed(forecast_output_path, force=force)
    _check_output_write_allowed(vrp_har_output_path, force=force)

    print(f"[{market_key}] Reading Phase 3 VRP panel: {vrp_input_path}")
    vrp_panel = load_feature_panel(vrp_input_path, sort_by_date=True)
    _validate_market_panel(vrp_panel, market=market_key, name=f"{market_key} VRP panel")

    print(f"[{market_key}] Running {config.oos_mode} HAR forecast")
    forecast_panel, coefficient_frame, audit_frame = _run_har_forecast_for_market(
        vrp_panel,
        config=config,
    )

    _validate_market_panel(
        forecast_panel,
        market=market_key,
        name=f"{market_key} HAR forecast panel",
    )

    print(f"[{market_key}] Building HAR-VRP panel")
    vrp_har_panel = compute_har_vrp(
        vrp_panel=vrp_panel,
        har_forecast_panel=forecast_panel,
    )

    _validate_market_panel(
        vrp_har_panel,
        market=market_key,
        name=f"{market_key} HAR-VRP panel",
    )

    print(f"[{market_key}] Saving HAR forecast panel: {forecast_output_path}")
    save_feature_panel(forecast_panel, forecast_output_path, index=False)

    print(f"[{market_key}] Saving HAR-VRP panel: {vrp_har_output_path}")
    save_feature_panel(vrp_har_panel, vrp_har_output_path, index=False)

    n_available = int(forecast_panel["har_forecast_available"].sum())
    n_rows = int(len(forecast_panel))

    print(f"[{market_key}] Forecast rows: {n_rows:,}")
    print(f"[{market_key}] Available HAR forecasts: {n_available:,}")

    blocked_counts = (
        forecast_panel["har_blocked_reason"]
        .fillna("available")
        .value_counts()
        .to_dict()
    )
    print(f"[{market_key}] Forecast status counts: {blocked_counts}")

    return {
        "vrp": vrp_panel,
        "forecast": forecast_panel,
        "coefficients": coefficient_frame,
        "audit": audit_frame,
        "vrp_har": vrp_har_panel,
    }


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate frames, returning empty frame if input list is empty.
    """
    non_empty = [frame for frame in frames if isinstance(frame, pd.DataFrame)]

    if not non_empty:
        return pd.DataFrame()

    return pd.concat(non_empty, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Write CSV with parent directory creation.
    """
    out_path = ensure_parent_dir(path)
    df.to_csv(out_path, index=False)
    return out_path


def _write_basic_report_tables(
    results: dict[str, dict[str, pd.DataFrame]],
    *,
    config: HARConfig,
) -> dict[str, Path]:
    """
    Write report tables available before har_diagnostics.py is added.

    The full plotting/metadata report module is added in the next chunk.
    """
    written: dict[str, Path] = {}

    coefficient_frames = [
        market_result["coefficients"]
        for market_result in results.values()
        if "coefficients" in market_result
    ]

    audit_frames = [
        market_result["audit"]
        for market_result in results.values()
        if "audit" in market_result
    ]

    all_coefficients = _concat_or_empty(coefficient_frames)
    all_audits = _concat_or_empty(audit_frames)

    coefficient_path = _resolve_path(
        config.report_paths.get(
            "coefficients",
            "reports/tables/har_coefficients.csv",
        )
    )
    audit_path = _resolve_path(
        config.report_paths.get(
            "no_lookahead_audit",
            "reports/tables/har_no_lookahead_audit.csv",
        )
    )

    written["coefficients"] = _write_csv(all_coefficients, coefficient_path)
    written["no_lookahead_audit"] = _write_csv(all_audits, audit_path)

    if "US" in results and "INDIA" in results:
        accuracy = build_forecast_accuracy_table(
            results["US"]["forecast"],
            results["INDIA"]["forecast"],
            target_col=config.target_col,
            forecast_cols=[
                HAR_FORECAST_COLUMNS[0],
                *HAR_BASELINE_COLUMNS,
            ],
        )

        accuracy_path = _resolve_path(
            config.report_paths.get(
                "forecast_accuracy",
                "reports/tables/har_forecast_accuracy.csv",
            )
        )
        written["forecast_accuracy"] = _write_csv(accuracy, accuracy_path)

    elif "US" in results:
        from vrp.forecasting.forecast_evaluation import evaluate_forecasts

        accuracy = evaluate_forecasts(
            results["US"]["forecast"],
            target_col=config.target_col,
            forecast_cols=[
                HAR_FORECAST_COLUMNS[0],
                *HAR_BASELINE_COLUMNS,
            ],
            market="US",
        )

        accuracy_path = _resolve_path(
            config.report_paths.get(
                "forecast_accuracy",
                "reports/tables/har_forecast_accuracy.csv",
            )
        )
        written["forecast_accuracy"] = _write_csv(accuracy, accuracy_path)

    elif "INDIA" in results:
        from vrp.forecasting.forecast_evaluation import evaluate_forecasts

        accuracy = evaluate_forecasts(
            results["INDIA"]["forecast"],
            target_col=config.target_col,
            forecast_cols=[
                HAR_FORECAST_COLUMNS[0],
                *HAR_BASELINE_COLUMNS,
            ],
            market="INDIA",
        )

        accuracy_path = _resolve_path(
            config.report_paths.get(
                "forecast_accuracy",
                "reports/tables/har_forecast_accuracy.csv",
            )
        )
        written["forecast_accuracy"] = _write_csv(accuracy, accuracy_path)

    return written


def _try_write_full_diagnostics(
    results: dict[str, dict[str, pd.DataFrame]],
    *,
    config: HARConfig,
) -> dict[str, Path]:
    """
    Write full HAR diagnostics if src/vrp/reports/har_diagnostics.py exists.

    Until Chunk 7 is pasted, this falls back to basic CSV report tables.
    """
    try:
        from vrp.reports.har_diagnostics import write_har_diagnostics  # type: ignore
    except (ModuleNotFoundError, ImportError):
        print(
            "[diagnostics] vrp.reports.har_diagnostics not found yet. "
            "Writing basic report tables only."
        )
        return _write_basic_report_tables(results, config=config)

    return write_har_diagnostics(
        results,
        config=config,
        table_dir=REPORT_TABLE_DIR,
        figure_dir=REPORT_FIGURE_DIR,
    )


def write_reports_if_requested(
    results: dict[str, dict[str, pd.DataFrame]],
    *,
    config: HARConfig,
    write_reports: bool,
) -> None:
    """
    Write report tables/figures if requested.
    """
    if not write_reports:
        print("[diagnostics] HAR reports skipped by --no-write-reports")
        return

    if not results:
        raise ValueError("No HAR results available for reporting.")

    print("[diagnostics] Writing HAR reports")
    written = _try_write_full_diagnostics(results, config=config)

    for name, path in written.items():
        print(f"[diagnostics] {name}: {path}")


def validate_final_results(
    results: dict[str, dict[str, pd.DataFrame]],
    *,
    selected_markets: list[str],
) -> None:
    """
    Validate final result dictionary.
    """
    missing_markets = [
        market for market in selected_markets
        if market not in results
    ]

    if missing_markets:
        raise ValueError(f"Missing market result(s): {missing_markets}")

    for market, result in results.items():
        required_keys = ["forecast", "coefficients", "audit", "vrp_har"]
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            raise ValueError(
                f"{market} result is missing required output(s): {missing_keys}"
            )

        forecast = result["forecast"]
        vrp_har = result["vrp_har"]

        for col in HAR_TARGET_COLUMNS + HAR_FEATURE_COLUMNS + HAR_FORECAST_COLUMNS:
            if col not in forecast.columns:
                raise ValueError(
                    f"{market} forecast panel missing required column: {col}"
                )

        for col in HAR_OUTPUT_FEATURE_COLUMNS:
            if col not in vrp_har.columns:
                raise ValueError(
                    f"{market} HAR-VRP panel missing required column: {col}"
                )

        invalid_vrp_har = (
            (~vrp_har["har_forecast_available"].astype(bool))
            & vrp_har["vrp_har_gk"].notna()
        )

        if invalid_vrp_har.any():
            bad_count = int(invalid_vrp_har.sum())
            raise ValueError(
                f"{market} HAR-VRP panel has {bad_count} unavailable forecast "
                "row(s) with non-null vrp_har_gk."
            )

        audit = result["audit"]

        available_audit = audit.loc[audit["forecast_available"].astype(bool)]
        if len(available_audit) > 0:
            bad_audit = ~available_audit[
                "rule_target_end_before_forecast_date"
            ].astype(bool)

            if bad_audit.any():
                bad_count = int(bad_audit.sum())
                raise ValueError(
                    f"{market} no-lookahead audit failed for {bad_count} "
                    "available forecast row(s)."
                )


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()

    config_path = _resolve_path(args.config)
    config = load_har_config(config_path)
    config = apply_cli_overrides(config, args)

    assert_har_registry_is_valid()

    selected_markets = markets_to_process(args.market)
    _print_config_summary(config, selected_markets)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for market in selected_markets:
        result = build_market_outputs(
            market,
            config=config,
            force=args.force,
        )
        results[market] = result

    validate_final_results(results, selected_markets=selected_markets)

    write_reports_if_requested(
        results,
        config=config,
        write_reports=args.write_reports,
    )

    print(
        f"[done] HAR Phase 4 complete: market={args.market}, "
        f"mode={config.oos_mode}"
    )


if __name__ == "__main__":
    main()