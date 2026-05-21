import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MSGARCHExportError(RuntimeError):
    """Raised when MSGARCH input export cannot be completed safely."""


@dataclass(frozen=True)
class ReturnBuildResult:
    log_return: pd.Series
    source_return_column: str
    derivation_method: str


@dataclass(frozen=True)
class ExportResult:
    market: str
    source_panel: Path
    output_csv: Path
    source_return_column: str
    derivation_method: str
    n_source_rows: int
    n_export_rows: int
    n_missing_before_drop: int
    start_date: str | None
    end_date: str | None
    return_scale: str
    min_observations: int
    validation_status: str
    input_hash_sha256: str
    created_at_utc: str


def resolve_project_path(path_like: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def load_msgarch_config(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = resolve_project_path(path, project_root)
    if not config_path.exists():
        raise MSGARCHExportError(f"MSGARCH config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise MSGARCHExportError(f"MSGARCH config is empty or invalid: {config_path}")

    return config


def normalize_market_arg(market: str, config: dict[str, Any]) -> list[str]:
    market = market.upper().strip()
    configured = set(config.get("markets", {}).keys())

    if market == "ALL":
        return sorted(configured)

    if market not in configured:
        allowed = ", ".join(sorted(configured | {"ALL"}))
        raise MSGARCHExportError(f"Unknown market '{market}'. Allowed values: {allowed}")

    return [market]


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    h = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_date_column(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise MSGARCHExportError("Source panel must contain a 'date' column.")

    date = pd.to_datetime(df["date"], errors="coerce")
    bad = int(date.isna().sum())
    if bad:
        raise MSGARCHExportError(f"Source panel contains {bad} invalid date value(s).")

    return date.dt.normalize()


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise MSGARCHExportError(f"Column '{col}' not found.")

    return pd.to_numeric(df[col], errors="coerce")


def _assert_no_forbidden_input_col(col: str) -> None:
    lower = col.lower()
    forbidden_fragments = [
        "vrp",
        "har_resid",
        "har_residual",
        "future",
        "forward_label",
        "next_",
        "state_for_next_session",
        "signal_trade_date",
    ]
    for fragment in forbidden_fragments:
        if fragment in lower:
            raise MSGARCHExportError(
                f"Forbidden MSGARCH input column selected: '{col}'. "
                "Phase 8 MSGARCH input must be index returns only."
            )


def build_index_log_return(
    df: pd.DataFrame,
    return_column_candidates: list[str],
    price_column_candidates: list[str],
) -> ReturnBuildResult:
    """
    Build the index log-return series according to Phase 8 rules.

    Priority:
    1. log_return
    2. index_return
    3. simple_return -> log1p(simple_return)
    4. diff(log(price_col))
    """
    for col in return_column_candidates:
        if col not in df.columns:
            continue

        _assert_no_forbidden_input_col(col)
        raw = _numeric_series(df, col)

        if col == "simple_return":
            invalid = raw <= -1.0
            invalid_count = int(invalid.fillna(False).sum())
            if invalid_count:
                raise MSGARCHExportError(
                    f"simple_return contains {invalid_count} value(s) <= -1, "
                    "so log1p(simple_return) is invalid."
                )

            return ReturnBuildResult(
                log_return=pd.Series(np.log1p(raw), index=raw.index),
                source_return_column=col,
                derivation_method="log1p(simple_return)",
            )

        return ReturnBuildResult(
            log_return=raw,
            source_return_column=col,
            derivation_method=f"direct:{col}",
        )

    for col in price_column_candidates:
        if col not in df.columns:
            continue

        _assert_no_forbidden_input_col(col)
        price = _numeric_series(df, col)

        invalid = price <= 0.0
        invalid_count = int(invalid.fillna(False).sum())
        if invalid_count:
            raise MSGARCHExportError(
                f"Price column '{col}' contains {invalid_count} non-positive value(s)."
            )

        return ReturnBuildResult(
            log_return=pd.Series(np.log(price), index=price.index).diff(),
            source_return_column=f"price:{col}",
            derivation_method=f"diff(log({col}))",
        )

    return_candidates = ", ".join(return_column_candidates)
    price_candidates = ", ".join(price_column_candidates)
    raise MSGARCHExportError(
        "Could not build MSGARCH input return. "
        f"Missing return candidates [{return_candidates}] and price candidates [{price_candidates}]."
    )


def validate_return_series(log_return: pd.Series) -> None:
    finite_or_missing = log_return.isna() | np.isfinite(log_return.astype(float))
    bad_count = int((~finite_or_missing).sum())
    if bad_count:
        raise MSGARCHExportError(f"log_return contains {bad_count} non-finite non-missing value(s).")


def build_msgarch_input_frame(
    source_df: pd.DataFrame,
    market: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, ReturnBuildResult, int]:
    date = require_date_column(source_df)

    return_candidates = list(config.get("return_column_candidates", []))
    price_candidates = list(config.get("price_column_candidates", []))

    if not return_candidates:
        raise MSGARCHExportError("Config field 'return_column_candidates' cannot be empty.")
    if not price_candidates:
        raise MSGARCHExportError("Config field 'price_column_candidates' cannot be empty.")

    result = build_index_log_return(
        df=source_df,
        return_column_candidates=return_candidates,
        price_column_candidates=price_candidates,
    )

    validate_return_series(result.log_return)

    out = pd.DataFrame(
        {
            "date": date,
            "market": market,
            "log_return": result.log_return.astype(float),
            "source_return_column": result.source_return_column,
            "input_available": True,
        }
    )

    n_missing_before_drop = int(out["log_return"].isna().sum())

    out = out.dropna(subset=["log_return"]).copy()
    out = out[np.isfinite(out["log_return"].astype(float))].copy()

    return_scale = str(config.get("input_policy", {}).get("return_scale", "percent")).lower()
    if return_scale == "percent":
        out["return_for_msgarch"] = 100.0 * out["log_return"]
    elif return_scale in {"raw", "decimal"}:
        out["return_for_msgarch"] = out["log_return"]
    else:
        raise MSGARCHExportError(
            f"Unsupported return_scale '{return_scale}'. Use 'percent' or 'raw'."
        )

    out = out.sort_values("date").reset_index(drop=True)

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out[
        [
            "date",
            "market",
            "log_return",
            "return_for_msgarch",
            "source_return_column",
            "input_available",
        ]
    ]

    return out, result, n_missing_before_drop


def load_source_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MSGARCHExportError(f"Source panel not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise MSGARCHExportError(
        f"Unsupported source panel format: {path}. Expected .parquet or .csv."
    )


def export_market(
    market: str,
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    dry_run: bool = False,
) -> ExportResult:
    market = market.upper().strip()

    market_cfg = config.get("markets", {}).get(market)
    if not isinstance(market_cfg, dict):
        raise MSGARCHExportError(f"Missing config for market '{market}'.")

    source_panel = resolve_project_path(market_cfg["source_panel"], project_root)
    output_csv = resolve_project_path(market_cfg["output_input_csv"], project_root)

    min_observations = int(config.get("input_policy", {}).get("min_observations", 1000))
    return_scale = str(config.get("input_policy", {}).get("return_scale", "percent")).lower()

    if dry_run:
        return ExportResult(
            market=market,
            source_panel=source_panel,
            output_csv=output_csv,
            source_return_column="DRY_RUN",
            derivation_method="DRY_RUN",
            n_source_rows=0,
            n_export_rows=0,
            n_missing_before_drop=0,
            start_date=None,
            end_date=None,
            return_scale=return_scale,
            min_observations=min_observations,
            validation_status="dry_run",
            input_hash_sha256="",
            created_at_utc=utc_now_iso(),
        )

    source_df = load_source_panel(source_panel)
    input_df, build_result, n_missing_before_drop = build_msgarch_input_frame(
        source_df=source_df,
        market=market,
        config=config,
    )

    n_export_rows = int(len(input_df))
    if n_export_rows < min_observations:
        raise MSGARCHExportError(
            f"{market} MSGARCH input has {n_export_rows} valid observations, "
            f"below min_observations={min_observations}."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_csv, index=False)

    file_hash = sha256_file(output_csv)

    start_date = None if input_df.empty else str(input_df["date"].iloc[0])
    end_date = None if input_df.empty else str(input_df["date"].iloc[-1])

    return ExportResult(
        market=market,
        source_panel=source_panel,
        output_csv=output_csv,
        source_return_column=build_result.source_return_column,
        derivation_method=build_result.derivation_method,
        n_source_rows=int(len(source_df)),
        n_export_rows=n_export_rows,
        n_missing_before_drop=n_missing_before_drop,
        start_date=start_date,
        end_date=end_date,
        return_scale=return_scale,
        min_observations=min_observations,
        validation_status="ok",
        input_hash_sha256=file_hash,
        created_at_utc=utc_now_iso(),
    )


def export_results_to_summary_frame(results: list[ExportResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for r in results:
        rows.append(
            {
                "market": r.market,
                "source_panel": str(r.source_panel),
                "output_csv": str(r.output_csv),
                "source_return_column": r.source_return_column,
                "derivation_method": r.derivation_method,
                "n_source_rows": r.n_source_rows,
                "n_export_rows": r.n_export_rows,
                "n_missing_before_drop": r.n_missing_before_drop,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "return_scale": r.return_scale,
                "min_observations": r.min_observations,
                "validation_status": r.validation_status,
                "input_hash_sha256": r.input_hash_sha256,
                "created_at_utc": r.created_at_utc,
            }
        )

    return pd.DataFrame(rows)


def write_input_summary(
    results: list[ExportResult],
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    summary_rel = config.get("output_policy", {}).get(
        "input_summary_csv",
        "reports/tables/phase_8/msgarch_input_summary.csv",
    )
    summary_path = resolve_project_path(summary_rel, project_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = export_results_to_summary_frame(results)
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def run_export(
    market_arg: str,
    config_path: str | Path,
    project_root: Path = PROJECT_ROOT,
    dry_run: bool = False,
) -> tuple[list[ExportResult], Path]:
    config = load_msgarch_config(config_path, project_root)
    markets = normalize_market_arg(market_arg, config)

    results = [
        export_market(
            market=market,
            config=config,
            project_root=project_root,
            dry_run=dry_run,
        )
        for market in markets
    ]

    summary_path = write_input_summary(results, config, project_root)
    return results, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export clean index return CSV inputs for optional R MSGARCH Phase 8."
    )

    parser.add_argument(
        "--market",
        required=True,
        choices=["US", "INDIA", "ALL"],
        help="Market to export.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_msgarch.yaml",
        help="Path to MSGARCH config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended inputs/outputs and write only a dry-run summary.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        results, summary_path = run_export(
            market_arg=args.market,
            config_path=args.config,
            project_root=PROJECT_ROOT,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[MSGARCH EXPORT ERROR] {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(
            json.dumps(
                {
                    "market": result.market,
                    "source_panel": str(result.source_panel),
                    "output_csv": str(result.output_csv),
                    "n_export_rows": result.n_export_rows,
                    "validation_status": result.validation_status,
                    "input_hash_sha256": result.input_hash_sha256,
                },
                indent=2,
            )
        )

    print(f"Input summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())