# scripts/build_features.py

"""
Build feature panels for the EPAT VRP project.

Supported feature builds:
- rv  : Phase 2 realised variance panels
- iv  : Phase 3 implied variance panels
- vrp : Phase 3 implied variance + VRP panels

Main commands:
    python scripts/build_features.py --market ALL --feature rv --window 22
    python scripts/build_features.py --market ALL --feature iv
    python scripts/build_features.py --market ALL --feature vrp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Make `src/` importable when running:
#     python scripts/build_features.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_features_cli import validate_args as validate_cli_args  # noqa: E402


from vrp.features.calendars import (  # noqa: E402
    report_calendar_mismatches,
    write_calendar_mismatch_report,
)
from vrp.features.feature_io import (  # noqa: E402
    assert_required_columns,
    load_feature_panel,
    save_feature_panel,
)
from vrp.features.feature_registry import (  # noqa: E402
    get_vrp_robustness_columns,
)
from vrp.features.implied_variance import build_implied_variance  # noqa: E402
from vrp.features.realized_variance import build_rv_panel  # noqa: E402
from vrp.features.vrp import build_vrp_panel  # noqa: E402
from vrp.reports.rv_diagnostics import write_rv_diagnostics  # noqa: E402
from vrp.reports.vrp_diagnostics import write_vrp_diagnostics  # noqa: E402


DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


MARKET_CONFIG = {
    "US": {
        "market": "US",
        "underlying_symbol": "US_UNDERLYING",
        "iv_symbol": "VIX",
        "underlying_input_path": DATA_PROCESSED_DIR / "us_underlying.parquet",
        "vix_input_path": DATA_PROCESSED_DIR / "us_vix.parquet",
        "rv_output_path": DATA_PROCESSED_DIR / "us_rv.parquet",
        "iv_output_path": DATA_PROCESSED_DIR / "us_iv.parquet",
        "vrp_output_path": DATA_PROCESSED_DIR / "us_vrp.parquet",
    },
    "INDIA": {
        "market": "INDIA",
        "underlying_symbol": "INDIA_UNDERLYING",
        "iv_symbol": "INDIA_VIX",
        "underlying_input_path": DATA_PROCESSED_DIR / "india_underlying.parquet",
        "vix_input_path": DATA_PROCESSED_DIR / "india_vix.parquet",
        "rv_output_path": DATA_PROCESSED_DIR / "india_rv.parquet",
        "iv_output_path": DATA_PROCESSED_DIR / "india_iv.parquet",
        "vrp_output_path": DATA_PROCESSED_DIR / "india_vrp.parquet",
    },
}


REQUIRED_OHLC_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
]


REQUIRED_IV_COLUMNS = [
    "date",
    "market",
    "iv_symbol",
    "iv_close",
    "iv_ann",
]


def required_rv_columns(window: int) -> list[str]:
    """
    Required realised-variance columns for Phase 3.
    """
    return [
        "date",
        "market",
        "symbol",
        "log_return",
        "simple_return",
        "gap_return",
        "intraday_return",
        "rv_gk_daily",
        f"rv_gk_{window}d_ann",
    ]


def required_vrp_columns(window: int) -> list[str]:
    """
    Required VRP output columns.
    """
    return [
        "date",
        "market",
        "underlying_symbol",
        "iv_symbol",
        "iv_close",
        "iv_ann",
        "rv_gk_daily",
        f"rv_gk_{window}d_ann",
        f"rv_gk_{window}d_ann_lag1",
        "vrp_backward_gk",
        "vrp_backward_gk_positive",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "feature_allowed",
    ]


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build feature panels for the EPAT VRP project."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        required=True,
        help="Market to process.",
    )

    parser.add_argument(
        "--feature",
        choices=["rv", "iv", "vrp"],
        required=True,
        help="Feature set to build: rv, iv, or vrp.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=22,
        help="Trailing RV window in trading days. Default: 22.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=22,
        help="Forward ex-post VRP label horizon in trading observations. Default: 22.",
    )

    parser.add_argument(
        "--annualization-periods",
        type=int,
        default=252,
        help="Annualization factor for daily variance. Default: 252.",
    )

    parser.add_argument(
        "--max-vix-value",
        type=float,
        default=200.0,
        help="Maximum accepted VIX / India VIX close value. Default: 200.",
    )

    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="If set, save feature parquet files but do not write reports/diagnostics.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments beyond argparse choices.
    """
    validate_cli_args(args)


def markets_to_process(market_arg: str) -> list[str]:
    """
    Resolve CLI market argument into concrete market list.
    """
    if market_arg == "ALL":
        return ["US", "INDIA"]

    return [market_arg]


def _require_existing_file(path: Path, *, description: str) -> None:
    """
    Raise FileNotFoundError if a required input file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_underlying_panel(input_path: Path, market: str) -> pd.DataFrame:
    """
    Load and validate a frozen Phase 1 processed OHLC panel.
    """
    _require_existing_file(
        input_path,
        description=f"Phase 1 processed underlying input for {market}",
    )

    df = load_feature_panel(input_path, sort_by_date=True)

    assert_required_columns(
        df,
        REQUIRED_OHLC_COLUMNS,
        panel_name=f"{market} underlying panel",
    )

    return df


def load_vix_panel(input_path: Path, market: str) -> pd.DataFrame:
    """
    Load Phase 1 processed VIX / India VIX panel.
    """
    _require_existing_file(
        input_path,
        description=f"Phase 1 processed VIX input for {market}",
    )

    df = load_feature_panel(input_path, sort_by_date=True)

    assert_required_columns(
        df,
        ["date"],
        panel_name=f"{market} VIX panel",
    )

    return df


def load_rv_panel(input_path: Path, market: str, *, window: int) -> pd.DataFrame:
    """
    Load Phase 2 realised-variance panel.
    """
    _require_existing_file(
        input_path,
        description=f"Phase 2 RV input for {market}",
    )

    df = load_feature_panel(input_path, sort_by_date=True)

    assert_required_columns(
        df,
        required_rv_columns(window),
        panel_name=f"{market} RV panel",
    )

    return df


def build_rv_for_market(
    market_key: str,
    *,
    window: int,
    annualization_periods: int,
) -> pd.DataFrame:
    """
    Build and save one market's realised variance panel.
    """
    if market_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market: {market_key}")

    config = MARKET_CONFIG[market_key]

    market = config["market"]
    symbol = config["underlying_symbol"]
    input_path = config["underlying_input_path"]
    output_path = config["rv_output_path"]

    print(f"[{market}] Reading underlying input: {input_path}")
    df = load_underlying_panel(input_path, market)

    print(f"[{market}] Building RV panel with window={window}")
    rv = build_rv_panel(
        df=df,
        market=market,
        symbol=symbol,
        window=window,
        annualization_periods=annualization_periods,
    )

    primary_col = f"rv_gk_{window}d_ann"
    yz_col = f"rv_yz_{window}d_ann"

    assert_required_columns(
        rv,
        [
            "date",
            "market",
            "symbol",
            "log_return",
            "simple_return",
            "gap_return",
            "intraday_return",
            "rv_cc_daily",
            "rv_parkinson_daily",
            "rv_gk_daily",
            "rv_rs_daily",
            primary_col,
            yz_col,
        ],
        panel_name=f"{market} RV panel",
    )

    if "rv_yz_daily" in rv.columns:
        raise ValueError(
            f"{market} RV panel contains forbidden column: rv_yz_daily"
        )

    print(f"[{market}] Saving RV output: {output_path}")
    save_feature_panel(rv, output_path, index=False)

    print(f"[{market}] RV rows: {len(rv):,}")
    print(f"[{market}] Primary RV column: {primary_col}")
    print(f"[{market}] First valid {primary_col}: {rv[primary_col].first_valid_index()}")

    return rv


def build_iv_for_market(
    market_key: str,
    *,
    max_vix_value: float,
) -> pd.DataFrame:
    """
    Build and save one market's implied-variance panel.
    """
    if market_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market: {market_key}")

    config = MARKET_CONFIG[market_key]

    market = config["market"]
    iv_symbol = config["iv_symbol"]
    input_path = config["vix_input_path"]
    output_path = config["iv_output_path"]

    print(f"[{market}] Reading VIX input: {input_path}")
    vix_df = load_vix_panel(input_path, market)

    print(f"[{market}] Building IV panel")
    iv = build_implied_variance(
        vix_df=vix_df,
        market=market,
        iv_symbol=iv_symbol,
        max_vix_value=max_vix_value,
    )

    assert_required_columns(
        iv,
        REQUIRED_IV_COLUMNS,
        panel_name=f"{market} IV panel",
    )

    print(f"[{market}] Saving IV output: {output_path}")
    save_feature_panel(iv, output_path, index=False)

    print(f"[{market}] IV rows: {len(iv):,}")
    print(f"[{market}] IV symbol: {iv_symbol}")

    return iv


def build_vrp_for_market(
    market_key: str,
    *,
    window: int,
    horizon: int,
    annualization_periods: int,
    max_vix_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Build IV and VRP panels for one market.

    Behavior:
    - Read *_vix.parquet.
    - Rebuild IV in memory.
    - Save *_iv.parquet.
    - Read *_rv.parquet.
    - Build *_vrp.parquet.
    - Return VRP panel, IV panel, calendar mismatch row.
    """
    if market_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market: {market_key}")

    config = MARKET_CONFIG[market_key]

    market = config["market"]
    iv_symbol = config["iv_symbol"]
    vix_input_path = config["vix_input_path"]
    rv_input_path = config["rv_output_path"]
    iv_output_path = config["iv_output_path"]
    vrp_output_path = config["vrp_output_path"]

    print(f"[{market}] Reading VIX input: {vix_input_path}")
    vix_df = load_vix_panel(vix_input_path, market)

    print(f"[{market}] Rebuilding IV panel for VRP")
    iv = build_implied_variance(
        vix_df=vix_df,
        market=market,
        iv_symbol=iv_symbol,
        max_vix_value=max_vix_value,
    )

    assert_required_columns(
        iv,
        REQUIRED_IV_COLUMNS,
        panel_name=f"{market} IV panel",
    )

    print(f"[{market}] Saving IV output: {iv_output_path}")
    save_feature_panel(iv, iv_output_path, index=False)

    print(f"[{market}] Reading RV input: {rv_input_path}")
    rv = load_rv_panel(rv_input_path, market, window=window)

    print(f"[{market}] Reporting IV/RV calendar mismatches")
    calendar_row = report_calendar_mismatches(
        iv_df=iv,
        rv_df=rv,
        market=market,
    )

    print(f"[{market}] Building VRP panel with horizon={horizon}")
    vrp = build_vrp_panel(
        iv_df=iv,
        rv_df=rv,
        market=market,
        horizon=horizon,
        annualization_periods=annualization_periods,
        rv_col=f"rv_gk_{window}d_ann",
        rv_daily_col="rv_gk_daily",
    )

    assert_required_columns(
        vrp,
        required_vrp_columns(window),
        panel_name=f"{market} VRP panel",
    )

    forbidden_live_feature_substrings = [
        "future",
        "forward",
        "expost",
        "label",
    ]
    feature_like_columns = [
        "iv_ann",
        f"rv_gk_{window}d_ann_lag1",
        "vrp_backward_gk",
        "vrp_backward_gk_positive",
    ]

    bad_feature_columns = [
        col for col in feature_like_columns
        if any(token in col.lower() for token in forbidden_live_feature_substrings)
    ]

    if bad_feature_columns:
        raise ValueError(
            f"{market} live feature columns contain forbidden substrings: "
            f"{bad_feature_columns}"
        )

    if "vrp_forward_expost_gk_label" in feature_like_columns:
        raise ValueError(
            f"{market} forward ex-post label was incorrectly included as a feature."
        )

    existing_robustness_cols = [
        col for col in get_vrp_robustness_columns()
        if col in vrp.columns
    ]
    print(f"[{market}] Robustness columns generated: {existing_robustness_cols}")

    print(f"[{market}] Saving VRP output: {vrp_output_path}")
    save_feature_panel(vrp, vrp_output_path, index=False)

    print(f"[{market}] VRP rows: {len(vrp):,}")
    print(f"[{market}] First feature_allowed row: {vrp['feature_allowed'].idxmax() if vrp['feature_allowed'].any() else None}")

    return vrp, iv, calendar_row


def write_rv_diagnostics_if_requested(
    panels: dict[str, pd.DataFrame],
    *,
    window: int,
    annualization_periods: int,
    skip_diagnostics: bool,
) -> None:
    """
    Write Phase 2 RV diagnostics unless skipped.
    """
    if skip_diagnostics:
        print("[diagnostics] RV diagnostics skipped by --skip-diagnostics")
        return

    if not panels:
        raise ValueError("No RV panels available for diagnostics.")

    print("[diagnostics] Writing RV diagnostics")

    paths = write_rv_diagnostics(
        panels,
        table_dir=REPORT_TABLE_DIR,
        figure_dir=REPORT_FIGURE_DIR,
        window=window,
        annualization_periods=annualization_periods,
    )

    for name, path in paths.items():
        print(f"[diagnostics] {name}: {path}")


def write_vrp_diagnostics_if_requested(
    panels: dict[str, pd.DataFrame],
    calendar_rows: list[dict[str, object]],
    *,
    horizon: int,
    annualization_periods: int,
    skip_diagnostics: bool,
) -> None:
    """
    Write Phase 3 VRP diagnostics unless skipped.
    """
    if skip_diagnostics:
        print("[diagnostics] VRP diagnostics skipped by --skip-diagnostics")
        return

    if not panels:
        raise ValueError("No VRP panels available for diagnostics.")

    print("[diagnostics] Writing VRP diagnostics")

    paths = write_vrp_diagnostics(
        panels,
        table_dir=REPORT_TABLE_DIR,
        figure_dir=REPORT_FIGURE_DIR,
        horizon=horizon,
        annualization_periods=annualization_periods,
    )

    for name, path in paths.items():
        print(f"[diagnostics] {name}: {path}")

    calendar_path = write_calendar_mismatch_report(
        calendar_rows,
        REPORT_TABLE_DIR / "calendar_mismatches.csv",
    )

    print(f"[diagnostics] calendar_mismatches: {calendar_path}")


def run_rv_build(args: argparse.Namespace, selected_markets: list[str]) -> None:
    """
    Run Phase 2 RV build.
    """
    built_panels: dict[str, pd.DataFrame] = {}

    for market_key in selected_markets:
        rv = build_rv_for_market(
            market_key,
            window=args.window,
            annualization_periods=args.annualization_periods,
        )
        built_panels[market_key] = rv

    write_rv_diagnostics_if_requested(
        built_panels,
        window=args.window,
        annualization_periods=args.annualization_periods,
        skip_diagnostics=args.skip_diagnostics,
    )


def run_iv_build(args: argparse.Namespace, selected_markets: list[str]) -> None:
    """
    Run Phase 3 IV build.
    """
    for market_key in selected_markets:
        build_iv_for_market(
            market_key,
            max_vix_value=args.max_vix_value,
        )


def run_vrp_build(args: argparse.Namespace, selected_markets: list[str]) -> None:
    """
    Run Phase 3 VRP build.

    This rebuilds IV from VIX input in memory and saves IV output before building VRP.
    This avoids stale IV files.
    """
    vrp_panels: dict[str, pd.DataFrame] = {}
    calendar_rows: list[dict[str, object]] = []

    for market_key in selected_markets:
        vrp, _iv, calendar_row = build_vrp_for_market(
            market_key,
            window=args.window,
            horizon=args.horizon,
            annualization_periods=args.annualization_periods,
            max_vix_value=args.max_vix_value,
        )
        vrp_panels[market_key] = vrp
        calendar_rows.append(calendar_row)

    write_vrp_diagnostics_if_requested(
        vrp_panels,
        calendar_rows,
        horizon=args.horizon,
        annualization_periods=args.annualization_periods,
        skip_diagnostics=args.skip_diagnostics,
    )


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    validate_args(args)

    selected_markets = markets_to_process(args.market)

    if args.feature == "rv":
        run_rv_build(args, selected_markets)
    elif args.feature == "iv":
        run_iv_build(args, selected_markets)
    elif args.feature == "vrp":
        run_vrp_build(args, selected_markets)
    else:
        raise ValueError(f"Unsupported feature: {args.feature}")

    print(f"[done] Feature build complete: feature={args.feature}, market={args.market}")


if __name__ == "__main__":
    main()