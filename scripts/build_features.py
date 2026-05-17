# scripts/build_features.py

"""
Build feature panels for the EPAT VRP project.

Phase 2 scope:
- Read frozen Phase 1 processed OHLC files.
- Build realised variance panels.
- Save Phase 2 RV outputs.
- Write RV diagnostics.

Main command:
    python scripts/build_features.py --market ALL --feature rv --window 22
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


from vrp.features.feature_io import (  # noqa: E402
    load_feature_panel,
    save_feature_panel,
    assert_required_columns,
)
from vrp.features.realized_variance import build_rv_panel  # noqa: E402
from vrp.reports.rv_diagnostics import write_rv_diagnostics  # noqa: E402


DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


MARKET_CONFIG = {
    "US": {
        "input_path": DATA_PROCESSED_DIR / "us_underlying.parquet",
        "output_path": DATA_PROCESSED_DIR / "us_rv.parquet",
        "market": "US",
        "symbol": "US_UNDERLYING",
    },
    "INDIA": {
        "input_path": DATA_PROCESSED_DIR / "india_underlying.parquet",
        "output_path": DATA_PROCESSED_DIR / "india_rv.parquet",
        "market": "INDIA",
        "symbol": "INDIA_UNDERLYING",
    },
}


REQUIRED_OHLC_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
]


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build Phase 2 feature panels for the VRP project."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        required=True,
        help="Market to process.",
    )

    parser.add_argument(
        "--feature",
        choices=["rv"],
        required=True,
        help="Feature set to build. Phase 2 currently supports only 'rv'.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=22,
        help="Trailing rolling window in trading days. Default: 22.",
    )

    parser.add_argument(
        "--annualization-periods",
        type=int,
        default=252,
        help="Annualization factor for daily variance. Default: 252.",
    )

    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="If set, save RV parquet files but do not write reports/diagnostics.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments beyond argparse choices.
    """
    if args.window < 2:
        raise ValueError("--window must be an integer >= 2.")

    if args.annualization_periods <= 0:
        raise ValueError("--annualization-periods must be positive.")

    if args.feature != "rv":
        raise ValueError("Only --feature rv is supported in Phase 2.")


def markets_to_process(market_arg: str) -> list[str]:
    """
    Resolve CLI market argument into concrete market list.
    """
    if market_arg == "ALL":
        return ["US", "INDIA"]

    return [market_arg]


def load_underlying_panel(input_path: Path, market: str) -> pd.DataFrame:
    """
    Load and validate a frozen Phase 1 processed OHLC panel.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 1 processed input for {market}: {input_path}"
        )

    df = load_feature_panel(input_path, sort_by_date=True)

    assert_required_columns(
        df,
        REQUIRED_OHLC_COLUMNS,
        panel_name=f"{market} underlying panel",
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

    Parameters
    ----------
    market_key:
        "US" or "INDIA".
    window:
        Rolling RV window.
    annualization_periods:
        Annualization factor.

    Returns
    -------
    pd.DataFrame
        Built RV panel.
    """
    if market_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market: {market_key}")

    config = MARKET_CONFIG[market_key]

    input_path = config["input_path"]
    output_path = config["output_path"]
    market = config["market"]
    symbol = config["symbol"]

    print(f"[{market}] Reading input: {input_path}")
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

    print(f"[{market}] Saving output: {output_path}")
    save_feature_panel(rv, output_path, index=False)

    print(f"[{market}] Rows: {len(rv):,}")
    print(f"[{market}] Primary RV column: {primary_col}")
    print(f"[{market}] First valid {primary_col}: {rv[primary_col].first_valid_index()}")

    return rv


def write_diagnostics_if_requested(
    panels: dict[str, pd.DataFrame],
    *,
    window: int,
    annualization_periods: int,
    skip_diagnostics: bool,
) -> None:
    """
    Write Phase 2 diagnostics unless skipped.
    """
    if skip_diagnostics:
        print("[diagnostics] Skipped by --skip-diagnostics")
        return

    if not panels:
        raise ValueError("No panels available for diagnostics.")

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


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    validate_args(args)

    selected_markets = markets_to_process(args.market)

    built_panels: dict[str, pd.DataFrame] = {}

    for market_key in selected_markets:
        rv = build_rv_for_market(
            market_key,
            window=args.window,
            annualization_periods=args.annualization_periods,
        )
        built_panels[market_key] = rv

    write_diagnostics_if_requested(
        built_panels,
        window=args.window,
        annualization_periods=args.annualization_periods,
        skip_diagnostics=args.skip_diagnostics,
    )

    print("[done] Phase 2 RV feature build complete.")


if __name__ == "__main__":
    main()