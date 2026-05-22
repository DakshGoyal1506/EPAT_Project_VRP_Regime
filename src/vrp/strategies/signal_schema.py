from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterable, Mapping

import pandas as pd

from vrp.strategies.exposure_rules import (
    BLOCK_NONE,
    DECISION_UNAVAILABLE,
    MAX_EXPOSURE,
    MIN_EXPOSURE,
)
from vrp.strategies.strategy_registry import (
    ALLOWED_CARRY_COLUMNS,
    ALLOWED_HMM_SIGNAL_COLUMNS,
    ALLOWED_MAR_SIGNAL_COLUMNS,
    ALLOWED_THRESHOLD_COLUMNS,
    assert_no_strategy_forbidden_columns,
    get_forbidden_columns,
)


MARKET_US: Final[str] = "US"
MARKET_INDIA: Final[str] = "INDIA"
VALID_MARKETS: Final[tuple[str, ...]] = (MARKET_US, MARKET_INDIA)

DATE_COLUMN: Final[str] = "date"

REQUIRED_HAR_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    "vrp_har_gk",
    "har_forecast_available",
)

OPTIONAL_HAR_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "vrp_har_gk_positive",
)

REQUIRED_THRESHOLD_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    *ALLOWED_THRESHOLD_COLUMNS,
)

REQUIRED_HMM_COLUMNS: Final[tuple[str, ...]] = (
    *ALLOWED_HMM_SIGNAL_COLUMNS,
)

REQUIRED_MAR_COLUMNS: Final[tuple[str, ...]] = (
    *ALLOWED_MAR_SIGNAL_COLUMNS,
)

PHASE9_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "market",
    "strategy_name",
    "regime_model",
    "signal_observation_date",
    "signal_available_after_close_date",
    "target_trade_date",
    "target_exposure",
    "strategy_available",
    "blocked_reason",
    "decision_reason",
    "state_name",
    "p_calm",
    "p_transition",
    "p_stress",
    "vrp_har_gk",
    "har_forecast_available",
    "source_signal_date_column",
    "source_model",
)

PHASE9_OUTPUT_DATE_COLUMNS: Final[tuple[str, ...]] = (
    "signal_observation_date",
    "signal_available_after_close_date",
    "target_trade_date",
)

PHASE9_OUTPUT_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "p_calm",
    "p_transition",
    "p_stress",
)

PHASE9_REQUIRED_LONG_FORMAT_KEYS: Final[tuple[str, ...]] = (
    "market",
    "signal_observation_date",
    "target_trade_date",
    "strategy_name",
)

FORBIDDEN_PERFORMANCE_OUTPUT_SUBSTRINGS: Final[tuple[str, ...]] = (
    "return",
    "returns",
    "pnl",
    "profit",
    "sharpe",
    "drawdown",
    "transaction_cost",
    "transaction_costs",
    "cost_estimate",
    "performance",
    "backtest",
)

VALID_BLOCKED_REASON_FOR_AVAILABLE: Final[str] = BLOCK_NONE
VALID_DECISION_REASON_FOR_UNAVAILABLE: Final[str] = DECISION_UNAVAILABLE


@dataclass(frozen=True)
class SanitizedFrame:
    """
    Result of dropping no-lookahead-forbidden columns from an upstream panel.

    frame:
        Input DataFrame after forbidden columns have been removed.

    forbidden_columns_present_but_excluded:
        Forbidden columns found in the upstream panel and removed before
        strategy construction.

    forbidden_columns_used:
        Columns that were still consumed after sanitation. This should remain
        empty. A non-empty value is a Phase 9 failure.
    """

    frame: pd.DataFrame
    forbidden_columns_present_but_excluded: tuple[str, ...]
    forbidden_columns_used: tuple[str, ...]


def normalise_market(market: object) -> str:
    value = str(market).strip().upper()

    if value not in VALID_MARKETS:
        raise ValueError(
            f"Unsupported market '{market}'. "
            f"Expected one of {list(VALID_MARKETS)}."
        )

    return value


def _as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _normalise_column_name(column: object) -> str:
    return str(column).strip().lower()


def require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    frame_name: str = "dataframe",
) -> None:
    required = _as_tuple(required_columns)
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(
            f"{frame_name} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}."
        )


def required_columns_for_source(source_name: str) -> tuple[str, ...]:
    source = str(source_name).strip().lower()

    if source == "har":
        return REQUIRED_HAR_COLUMNS

    if source == "threshold":
        return REQUIRED_THRESHOLD_COLUMNS

    if source in {"hmm", "gaussian_hmm"}:
        return REQUIRED_HMM_COLUMNS

    if source in {"mar", "markov_autoreg", "markov_autoregression"}:
        return REQUIRED_MAR_COLUMNS

    raise ValueError(
        f"Unknown source_name '{source_name}'. "
        "Expected one of: har, threshold, gaussian_hmm, markov_autoreg."
    )


def require_source_columns(
    df: pd.DataFrame,
    source_name: str,
    frame_name: str | None = None,
) -> None:
    source = str(source_name).strip().lower()
    label = frame_name if frame_name is not None else source
    require_columns(
        df=df,
        required_columns=required_columns_for_source(source),
        frame_name=label,
    )


def find_forbidden_columns(columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(get_forbidden_columns(columns))


def sanitize_strategy_input_frame(
    df: pd.DataFrame,
    frame_name: str = "dataframe",
) -> SanitizedFrame:
    """
    Drop forbidden no-lookahead columns from a raw upstream panel.

    This function is intentionally permissive toward raw upstream panels:
    they may contain ex-post labels or diagnostic fields created by previous
    phases. Phase 9 must remove them before strategy logic consumes inputs.
    """
    forbidden = find_forbidden_columns(df.columns)

    sanitized = df.drop(columns=list(forbidden), errors="ignore").copy()

    forbidden_used = find_forbidden_columns(sanitized.columns)

    if forbidden_used:
        raise ValueError(
            f"{frame_name} still contains forbidden strategy column(s) after "
            f"sanitation: {list(forbidden_used)}."
        )

    return SanitizedFrame(
        frame=sanitized,
        forbidden_columns_present_but_excluded=forbidden,
        forbidden_columns_used=forbidden_used,
    )


def sanitize_input_frames(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, ...]]]:
    """
    Sanitize multiple input frames.

    Returns:
        sanitized_frames:
            Mapping from original frame key to sanitized DataFrame.

        present_but_excluded:
            Mapping from original frame key to forbidden columns removed from
            that frame.
    """
    sanitized_frames: dict[str, pd.DataFrame] = {}
    present_but_excluded: dict[str, tuple[str, ...]] = {}

    for frame_name, frame in frames.items():
        result = sanitize_strategy_input_frame(frame, frame_name=frame_name)
        sanitized_frames[frame_name] = result.frame
        present_but_excluded[frame_name] = (
            result.forbidden_columns_present_but_excluded
        )

    return sanitized_frames, present_but_excluded


def assert_no_forbidden_columns_consumed(columns: Iterable[str]) -> None:
    """
    Fail if strategy logic attempts to consume forbidden no-lookahead columns.
    """
    assert_no_strategy_forbidden_columns(columns)


def assert_no_performance_columns(columns: Iterable[str]) -> None:
    bad_columns: list[str] = []

    for column in columns:
        normalised = _normalise_column_name(column)
        if any(token in normalised for token in FORBIDDEN_PERFORMANCE_OUTPUT_SUBSTRINGS):
            bad_columns.append(str(column))

    if bad_columns:
        raise ValueError(
            "Phase 9 signal outputs must not contain performance/backtest "
            f"columns. Found: {bad_columns}."
        )


def validate_phase9_output_schema(
    df: pd.DataFrame,
    allow_extra_columns: bool = False,
) -> None:
    """
    Validate the canonical long-format Phase 9 signal panel schema.
    """
    require_columns(
        df=df,
        required_columns=PHASE9_OUTPUT_COLUMNS,
        frame_name="phase9_signal_panel",
    )
    assert_no_performance_columns(df.columns)
    assert_no_forbidden_columns_consumed(df.columns)

    if not allow_extra_columns:
        expected = set(PHASE9_OUTPUT_COLUMNS)
        actual = set(str(column) for column in df.columns)
        extra = sorted(actual.difference(expected))
        if extra:
            raise ValueError(
                "Phase 9 output contains unexpected column(s): "
                f"{extra}. Set allow_extra_columns=True only for explicit "
                "debugging."
            )


def validate_long_format_keys(df: pd.DataFrame) -> None:
    require_columns(
        df=df,
        required_columns=PHASE9_REQUIRED_LONG_FORMAT_KEYS,
        frame_name="phase9_signal_panel",
    )

    duplicated = df.duplicated(list(PHASE9_REQUIRED_LONG_FORMAT_KEYS), keep=False)

    if duplicated.any():
        duplicate_rows = df.loc[
            duplicated,
            list(PHASE9_REQUIRED_LONG_FORMAT_KEYS),
        ].head(10)
        raise ValueError(
            "Phase 9 output must be long-format with one row per "
            "market/signal_observation_date/target_trade_date/strategy_name. "
            f"Duplicate key rows found:\n{duplicate_rows}"
        )


def validate_exposure_bounds(
    df: pd.DataFrame,
    exposure_column: str = "target_exposure",
    available_column: str = "strategy_available",
    min_exposure: float = MIN_EXPOSURE,
    max_exposure: float = MAX_EXPOSURE,
) -> None:
    require_columns(
        df=df,
        required_columns=(exposure_column, available_column),
        frame_name="phase9_signal_panel",
    )

    available_mask = df[available_column].astype(bool)
    available_exposures = df.loc[available_mask, exposure_column]

    if available_exposures.isna().any():
        raise ValueError(
            "Available Phase 9 strategy rows must have finite target_exposure."
        )

    numeric_available = pd.to_numeric(available_exposures, errors="coerce")

    if numeric_available.isna().any():
        raise ValueError(
            "Available Phase 9 strategy rows must have numeric target_exposure."
        )

    if not numeric_available.map(math.isfinite).all():
        raise ValueError(
            "Available Phase 9 strategy rows must have finite target_exposure."
        )

    if (numeric_available < min_exposure).any() or (
        numeric_available > max_exposure
    ).any():
        raise ValueError(
            f"Available Phase 9 target_exposure values must lie in "
            f"[{min_exposure}, {max_exposure}]."
        )

    unavailable_exposures = df.loc[~available_mask, exposure_column]

    if unavailable_exposures.notna().any():
        raise ValueError(
            "Unavailable Phase 9 strategy rows must use NaN target_exposure."
        )


def validate_availability_consistency(
    df: pd.DataFrame,
    available_column: str = "strategy_available",
    exposure_column: str = "target_exposure",
    blocked_reason_column: str = "blocked_reason",
    decision_reason_column: str = "decision_reason",
) -> None:
    require_columns(
        df=df,
        required_columns=(
            available_column,
            exposure_column,
            blocked_reason_column,
            decision_reason_column,
        ),
        frame_name="phase9_signal_panel",
    )

    available_mask = df[available_column].astype(bool)

    available_bad_blocked = df.loc[
        available_mask
        & (df[blocked_reason_column].astype(str) != VALID_BLOCKED_REASON_FOR_AVAILABLE)
    ]

    if not available_bad_blocked.empty:
        raise ValueError(
            "Available Phase 9 rows must use blocked_reason='none'."
        )

    available_bad_decision = df.loc[
        available_mask
        & (df[decision_reason_column].astype(str) == VALID_DECISION_REASON_FOR_UNAVAILABLE)
    ]

    if not available_bad_decision.empty:
        raise ValueError(
            "Available Phase 9 rows must not use decision_reason='unavailable'."
        )

    unavailable_bad_blocked = df.loc[
        (~available_mask)
        & (df[blocked_reason_column].astype(str) == VALID_BLOCKED_REASON_FOR_AVAILABLE)
    ]

    if not unavailable_bad_blocked.empty:
        raise ValueError(
            "Unavailable Phase 9 rows must have a real blocked_reason."
        )

    unavailable_bad_decision = df.loc[
        (~available_mask)
        & (
            df[decision_reason_column].astype(str)
            != VALID_DECISION_REASON_FOR_UNAVAILABLE
        )
    ]

    if not unavailable_bad_decision.empty:
        raise ValueError(
            "Unavailable Phase 9 rows must use decision_reason='unavailable'."
        )


def validate_phase9_signal_panel(
    df: pd.DataFrame,
    allow_extra_columns: bool = False,
) -> None:
    validate_phase9_output_schema(df, allow_extra_columns=allow_extra_columns)
    validate_long_format_keys(df)
    validate_exposure_bounds(df)
    validate_availability_consistency(df)


def build_no_lookahead_audit_records(
    present_but_excluded: Mapping[str, Iterable[str]],
    forbidden_columns_used: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """
    Build records for strategy_no_lookahead_audit.csv.

    This creates metadata/audit records only. It does not allow strategy logic
    to consume forbidden columns.
    """
    used = tuple(forbidden_columns_used or ())

    records: list[dict[str, object]] = []

    for frame_name, columns in present_but_excluded.items():
        for column in columns:
            records.append(
                {
                    "frame_name": frame_name,
                    "column_name": str(column),
                    "audit_status": "present_but_excluded",
                    "used_by_strategy": False,
                }
            )

    for column in used:
        records.append(
            {
                "frame_name": "strategy_consumed_columns",
                "column_name": str(column),
                "audit_status": "forbidden_column_used",
                "used_by_strategy": True,
            }
        )

    return records


__all__ = [
    "MARKET_US",
    "MARKET_INDIA",
    "VALID_MARKETS",
    "DATE_COLUMN",
    "REQUIRED_HAR_COLUMNS",
    "OPTIONAL_HAR_AUDIT_COLUMNS",
    "REQUIRED_THRESHOLD_COLUMNS",
    "REQUIRED_HMM_COLUMNS",
    "REQUIRED_MAR_COLUMNS",
    "PHASE9_OUTPUT_COLUMNS",
    "PHASE9_OUTPUT_DATE_COLUMNS",
    "PHASE9_OUTPUT_PROBABILITY_COLUMNS",
    "PHASE9_REQUIRED_LONG_FORMAT_KEYS",
    "FORBIDDEN_PERFORMANCE_OUTPUT_SUBSTRINGS",
    "SanitizedFrame",
    "normalise_market",
    "require_columns",
    "required_columns_for_source",
    "require_source_columns",
    "find_forbidden_columns",
    "sanitize_strategy_input_frame",
    "sanitize_input_frames",
    "assert_no_forbidden_columns_consumed",
    "assert_no_performance_columns",
    "validate_phase9_output_schema",
    "validate_long_format_keys",
    "validate_exposure_bounds",
    "validate_availability_consistency",
    "validate_phase9_signal_panel",
    "build_no_lookahead_audit_records",
]