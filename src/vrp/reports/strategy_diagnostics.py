from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import pandas as pd

from vrp.strategies.signal_schema import (
    FORBIDDEN_PERFORMANCE_OUTPUT_SUBSTRINGS,
    PHASE9_OUTPUT_COLUMNS,
    assert_no_performance_columns,
    build_no_lookahead_audit_records,
    validate_phase9_signal_panel,
)
from vrp.strategies.strategy_registry import APPROVED_STRATEGY_NAMES


DEFAULT_REPORT_NOTE = (
    "Phase 9 defines ex-ante exposure intentions only. It does not evaluate "
    "realised strategy performance. The rules were fixed before Phase 10 "
    "backtesting. Reported diagnostics describe signal availability, exposure "
    "distribution, and rule activation only; they are not evidence of profitability."
)

DEFAULT_MSVOL_POLICY = "excluded_diagnostic_only"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_count(series: pd.Series) -> int:
    return int(series.size)


def _safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.sum())


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def _safe_min(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.min())


def _safe_max(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.max())


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(bool)


def _ensure_no_performance_content(df: pd.DataFrame, frame_name: str) -> None:
    try:
        assert_no_performance_columns(df.columns)
    except ValueError as exc:
        raise ValueError(f"{frame_name} contains forbidden Phase 10 fields.") from exc


def _normalise_strategy_order(df: pd.DataFrame) -> pd.DataFrame:
    if "strategy_name" not in df.columns:
        return df

    strategy_rank = {name: index for index, name in enumerate(APPROVED_STRATEGY_NAMES)}
    result = df.copy()
    result["_strategy_rank"] = result["strategy_name"].map(strategy_rank).fillna(999)
    sort_columns = [
        column
        for column in ["market", "_strategy_rank", "strategy_name"]
        if column in result.columns
    ]
    result = result.sort_values(sort_columns).drop(columns=["_strategy_rank"])
    return result.reset_index(drop=True)


def _available_exposures(df: pd.DataFrame) -> pd.Series:
    available = _to_bool_series(df["strategy_available"])
    return pd.to_numeric(df.loc[available, "target_exposure"], errors="coerce")


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def validate_diagnostic_table(df: pd.DataFrame, frame_name: str) -> None:
    """
    Validate that a Phase 9 diagnostic table does not contain Phase 10 fields.
    """
    _ensure_no_performance_content(df, frame_name=frame_name)


def create_strategy_signal_summary(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build one summary row per market/strategy.

    This is a signal-distribution diagnostic only.
    """
    validate_phase9_signal_panel(signal_panel)

    records: list[dict[str, Any]] = []

    group_cols = ["market", "strategy_name", "regime_model"]
    for keys, group in signal_panel.groupby(group_cols, dropna=False):
        market, strategy_name, regime_model = keys
        available_mask = _to_bool_series(group["strategy_available"])
        unavailable_mask = ~available_mask
        available_exposure = pd.to_numeric(
            group.loc[available_mask, "target_exposure"],
            errors="coerce",
        )

        available_count = int(available_mask.sum())
        unavailable_count = int(unavailable_mask.sum())
        total_count = int(len(group))

        flat_count = int((available_exposure == 0.0).sum())
        full_short_count = int((available_exposure == -1.0).sum())
        partial_short_count = int(
            ((available_exposure > -1.0) & (available_exposure < 0.0)).sum()
        )

        records.append(
            {
                "market": market,
                "strategy_name": strategy_name,
                "regime_model": regime_model,
                "row_count": total_count,
                "available_count": available_count,
                "unavailable_count": unavailable_count,
                "available_fraction": _fraction(available_count, total_count),
                "unavailable_fraction": _fraction(unavailable_count, total_count),
                "mean_target_exposure": _safe_mean(available_exposure.dropna()),
                "min_target_exposure": _safe_min(available_exposure.dropna()),
                "max_target_exposure": _safe_max(available_exposure.dropna()),
                "flat_count": flat_count,
                "full_short_vol_count": full_short_count,
                "partial_short_vol_count": partial_short_count,
                "flat_fraction": _fraction(flat_count, available_count),
                "full_short_vol_fraction": _fraction(
                    full_short_count,
                    available_count,
                ),
                "partial_short_vol_fraction": _fraction(
                    partial_short_count,
                    available_count,
                ),
                "first_signal_observation_date": group[
                    "signal_observation_date"
                ].min(),
                "last_signal_observation_date": group[
                    "signal_observation_date"
                ].max(),
                "first_target_trade_date": group["target_trade_date"].min(),
                "last_target_trade_date": group["target_trade_date"].max(),
            }
        )

    result = pd.DataFrame.from_records(records)
    result = _normalise_strategy_order(result)
    validate_diagnostic_table(result, "strategy_signal_summary")
    return result


def create_strategy_exposure_by_year(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build annual exposure-distribution diagnostics.

    Uses target_trade_date year when available. Rows without target_trade_date
    are excluded because they are unavailable for next-session application.
    """
    validate_phase9_signal_panel(signal_panel)

    df = signal_panel.copy()
    df = df[df["target_trade_date"].notna()].copy()
    df["target_year"] = pd.to_datetime(df["target_trade_date"]).dt.year

    records: list[dict[str, Any]] = []

    group_cols = ["market", "strategy_name", "regime_model", "target_year"]
    for keys, group in df.groupby(group_cols, dropna=False):
        market, strategy_name, regime_model, target_year = keys
        available_mask = _to_bool_series(group["strategy_available"])
        available_exposure = pd.to_numeric(
            group.loc[available_mask, "target_exposure"],
            errors="coerce",
        )

        available_count = int(available_mask.sum())
        total_count = int(len(group))

        records.append(
            {
                "market": market,
                "strategy_name": strategy_name,
                "regime_model": regime_model,
                "target_year": int(cast(Any, target_year)),
                "row_count": total_count,
                "available_count": available_count,
                "unavailable_count": total_count - available_count,
                "mean_target_exposure": _safe_mean(available_exposure.dropna()),
                "min_target_exposure": _safe_min(available_exposure.dropna()),
                "max_target_exposure": _safe_max(available_exposure.dropna()),
                "flat_count": int((available_exposure == 0.0).sum()),
                "full_short_vol_count": int((available_exposure == -1.0).sum()),
                "partial_short_vol_count": int(
                    ((available_exposure > -1.0) & (available_exposure < 0.0)).sum()
                ),
            }
        )

    result = pd.DataFrame.from_records(records)
    result = _normalise_strategy_order(result)

    if not result.empty:
        result = result.sort_values(
            ["market", "strategy_name", "target_year"]
        ).reset_index(drop=True)

    validate_diagnostic_table(result, "strategy_exposure_by_year")
    return result


def create_strategy_exposure_change_summary(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build exposure-path diagnostics.

    This is deliberately not called turnover. It does not estimate costs.
    It counts and sums absolute changes in target exposure among available
    signal rows only.
    """
    validate_phase9_signal_panel(signal_panel)

    available = signal_panel[signal_panel["strategy_available"].astype(bool)].copy()
    available = available.sort_values(
        ["market", "strategy_name", "target_trade_date", "signal_observation_date"]
    )
    available["target_exposure"] = pd.to_numeric(
        available["target_exposure"],
        errors="coerce",
    )

    records: list[dict[str, Any]] = []

    group_cols = ["market", "strategy_name", "regime_model"]
    for keys, group in available.groupby(group_cols, dropna=False):
        market, strategy_name, regime_model = keys
        exposures = group["target_exposure"].dropna()
        changes = exposures.diff().dropna()
        absolute_changes = changes.abs()

        nonzero_change_count = int((absolute_changes > 0.0).sum())
        absolute_change_sum = _safe_sum(absolute_changes)

        records.append(
            {
                "market": market,
                "strategy_name": strategy_name,
                "regime_model": regime_model,
                "available_row_count": int(len(group)),
                "exposure_change_observation_count": int(len(changes)),
                "absolute_exposure_change_count": nonzero_change_count,
                "absolute_exposure_change_sum": absolute_change_sum,
                "mean_absolute_exposure_change": _safe_mean(absolute_changes),
                "max_absolute_exposure_change": _safe_max(absolute_changes),
            }
        )

    result = pd.DataFrame.from_records(records)
    result = _normalise_strategy_order(result)
    validate_diagnostic_table(result, "strategy_exposure_change_summary")
    return result


def create_strategy_blocked_reason_summary(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Count blocked reasons and decision reasons by market/strategy.

    Valid flat decisions should appear under decision_reason, not blocked_reason.
    """
    validate_phase9_signal_panel(signal_panel)

    blocked = (
        signal_panel.groupby(
            ["market", "strategy_name", "regime_model", "blocked_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
    )
    blocked["summary_type"] = "blocked_reason"
    blocked = blocked.rename(columns={"blocked_reason": "reason"})

    decision = (
        signal_panel.groupby(
            ["market", "strategy_name", "regime_model", "decision_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
    )
    decision["summary_type"] = "decision_reason"
    decision = decision.rename(columns={"decision_reason": "reason"})

    result = pd.concat([blocked, decision], ignore_index=True)
    result = result[
        [
            "market",
            "strategy_name",
            "regime_model",
            "summary_type",
            "reason",
            "row_count",
        ]
    ]
    result = _normalise_strategy_order(result)
    validate_diagnostic_table(result, "strategy_blocked_reason_summary")
    return result


def create_no_lookahead_audit_table(
    present_but_excluded: Mapping[str, Iterable[str]],
    forbidden_columns_used: Iterable[str] | None = None,
) -> pd.DataFrame:
    records = build_no_lookahead_audit_records(
        present_but_excluded=present_but_excluded,
        forbidden_columns_used=forbidden_columns_used or (),
    )

    result = pd.DataFrame.from_records(
        records,
        columns=[
            "frame_name",
            "column_name",
            "audit_status",
            "used_by_strategy",
        ],
    )

    validate_diagnostic_table(result, "strategy_no_lookahead_audit")
    return result


def file_sha256(path: str | Path) -> str | None:
    """
    Return SHA-256 hash of a file, or None if the path does not exist.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_input_file_hashes(input_file_paths: Mapping[str, str | Path]) -> dict[str, str | None]:
    return {
        str(name): file_sha256(path)
        for name, path in input_file_paths.items()
    }


def build_row_counts_by_input(input_frames: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    return {
        str(name): int(len(frame))
        for name, frame in input_frames.items()
    }


def build_phase9_metadata(
    *,
    market: str,
    strategy_config_hash: str,
    input_file_paths: Mapping[str, str | Path],
    input_file_hashes: Mapping[str, str | None] | None,
    row_counts_by_input: Mapping[str, int],
    strategy_names: Iterable[str],
    forbidden_columns_present_but_excluded: Mapping[str, Iterable[str]],
    forbidden_columns_used: Iterable[str],
    timing_policy: Mapping[str, Any],
    exposure_bounds: Mapping[str, Any],
    report_note: str = DEFAULT_REPORT_NOTE,
    msvol_policy: str = DEFAULT_MSVOL_POLICY,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Build Phase 9 run metadata.

    The metadata explicitly records that MSVOL is excluded and that forbidden
    columns were either excluded or not used.
    """
    forbidden_used = tuple(str(column) for column in forbidden_columns_used)

    if forbidden_used:
        raise ValueError(
            "Phase 9 metadata cannot report forbidden columns as used. "
            f"Found: {list(forbidden_used)}."
        )

    metadata = {
        "run_timestamp": run_timestamp or _utc_timestamp(),
        "phase": 9,
        "market": str(market).upper(),
        "strategy_config_hash": strategy_config_hash,
        "input_file_paths": {
            str(name): str(path)
            for name, path in input_file_paths.items()
        },
        "input_file_hashes": (
            dict(input_file_hashes)
            if input_file_hashes is not None
            else build_input_file_hashes(input_file_paths)
        ),
        "row_counts_by_input": {
            str(name): int(count)
            for name, count in row_counts_by_input.items()
        },
        "strategy_names": sorted(str(name) for name in strategy_names),
        "forbidden_columns_present_but_excluded": {
            str(frame_name): sorted(str(column) for column in columns)
            for frame_name, columns in forbidden_columns_present_but_excluded.items()
        },
        "forbidden_columns_used": [],
        "msvol_policy": msvol_policy,
        "timing_policy": dict(timing_policy),
        "exposure_bounds": dict(exposure_bounds),
        "phase_9_report_note": report_note,
    }

    return metadata


def assert_diagnostics_are_phase9_only(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Fail if diagnostics contain Phase 10/backtest concepts.
    """
    for table_name, table in tables.items():
        validate_diagnostic_table(table, frame_name=table_name)

    if metadata is not None:
        forbidden_metadata_tokens = [
            "sharpe_ratio",
            "max_drawdown",
            "strategy_return",
            "strategy_returns",
            "realised_return",
            "realized_return",
            "pnl",
            "profit_loss",
            "transaction_cost",
            "transaction_costs",
            "performance_ranking",
        ]
        metadata_text = json.dumps(metadata, sort_keys=True).lower()

        for token in forbidden_metadata_tokens:
            if token in metadata_text:
                raise ValueError(
                    f"Metadata contains forbidden Phase 10 token '{token}'."
                )


def create_all_phase9_diagnostics(
    *,
    signal_panel: pd.DataFrame,
    present_but_excluded: Mapping[str, Iterable[str]],
    forbidden_columns_used: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Create all Phase 9 diagnostic tables except metadata.
    """
    tables = {
        "strategy_signal_summary": create_strategy_signal_summary(signal_panel),
        "strategy_exposure_by_year": create_strategy_exposure_by_year(signal_panel),
        "strategy_exposure_change_summary": create_strategy_exposure_change_summary(
            signal_panel
        ),
        "strategy_blocked_reason_summary": create_strategy_blocked_reason_summary(
            signal_panel
        ),
        "strategy_no_lookahead_audit": create_no_lookahead_audit_table(
            present_but_excluded=present_but_excluded,
            forbidden_columns_used=forbidden_columns_used or (),
        ),
    }

    assert_diagnostics_are_phase9_only(tables)
    return tables


def write_diagnostic_tables(
    *,
    tables: Mapping[str, pd.DataFrame],
    output_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """
    Write selected diagnostic tables to CSV.

    Expected output path keys:
        signal_summary
        exposure_by_year
        exposure_change_summary
        blocked_reason_summary
        no_lookahead_audit
    """
    key_map = {
        "strategy_signal_summary": "signal_summary",
        "strategy_exposure_by_year": "exposure_by_year",
        "strategy_exposure_change_summary": "exposure_change_summary",
        "strategy_blocked_reason_summary": "blocked_reason_summary",
        "strategy_no_lookahead_audit": "no_lookahead_audit",
    }

    written: dict[str, Path] = {}

    for table_name, table in tables.items():
        output_key = key_map.get(table_name)
        if output_key is None:
            raise ValueError(f"Unknown diagnostic table name: {table_name}")

        if output_key not in output_paths:
            raise KeyError(f"Missing output path for diagnostic key '{output_key}'.")

        output_path = Path(output_paths[output_key])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        written[table_name] = output_path

    return written


def write_metadata_json(
    *,
    metadata: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Write Phase 9 metadata JSON.
    """
    assert_diagnostics_are_phase9_only(tables={}, metadata=metadata)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True, default=str)

    return path


__all__ = [
    "DEFAULT_REPORT_NOTE",
    "DEFAULT_MSVOL_POLICY",
    "validate_diagnostic_table",
    "create_strategy_signal_summary",
    "create_strategy_exposure_by_year",
    "create_strategy_exposure_change_summary",
    "create_strategy_blocked_reason_summary",
    "create_no_lookahead_audit_table",
    "file_sha256",
    "build_input_file_hashes",
    "build_row_counts_by_input",
    "build_phase9_metadata",
    "assert_diagnostics_are_phase9_only",
    "create_all_phase9_diagnostics",
    "write_diagnostic_tables",
    "write_metadata_json",
]