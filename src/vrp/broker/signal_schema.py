"""
Phase 9 signal schema validation for Phase 11.

Phase 11 must consume Phase 9 strategy signals exactly as next-session-safe
research outputs. It must not recompute regimes, recompute exposures, or
double-lag signals.

This module validates and canonicalizes the Phase 9 signal input before any
broker-readiness logic runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vrp.broker import APPROVED_STRATEGIES
from vrp.broker.broker_config import BrokerConfig


class SignalSchemaError(ValueError):
    """Raised when a Phase 9 strategy signal file is invalid."""


@dataclass(frozen=True)
class Phase9SignalSchema:
    """Schema contract for Phase 9 signal files."""

    required_columns: tuple[str, ...]
    strategy_column_candidates: tuple[str, ...]
    optional_columns: tuple[str, ...]
    aliases: Mapping[str, str]
    target_exposure_min: float
    target_exposure_max: float
    latest_signal_sort_column: str
    approved_strategies: tuple[str, ...]

    @classmethod
    def from_broker_config(cls, config: BrokerConfig) -> "Phase9SignalSchema":
        raw_schema = config.raw.get("signal_schema")
        if not isinstance(raw_schema, Mapping):
            raise SignalSchemaError("Config missing signal_schema mapping")

        validation = raw_schema.get("validation")
        if not isinstance(validation, Mapping):
            validation = {}

        canonical_required = tuple(
            str(x) for x in raw_schema.get("canonical_required_columns", ())
        )
        strategy_candidates = tuple(
            str(x) for x in raw_schema.get("strategy_column_candidates", ())
        )
        optional_columns = tuple(str(x) for x in raw_schema.get("optional_columns", ()))
        aliases = {
            str(source): str(target)
            for source, target in dict(raw_schema.get("aliases", {})).items()
        }

        return cls(
            required_columns=canonical_required,
            strategy_column_candidates=strategy_candidates,
            optional_columns=optional_columns,
            aliases=aliases,
            target_exposure_min=float(validation.get("target_exposure_min", -1.0)),
            target_exposure_max=float(validation.get("target_exposure_max", 0.0)),
            latest_signal_sort_column=str(
                validation.get("latest_signal_sort_column", "target_trade_date")
            ),
            approved_strategies=tuple(config.approved_strategies),
        )

    @classmethod
    def default(cls) -> "Phase9SignalSchema":
        return cls(
            required_columns=(
                "target_trade_date",
                "target_exposure",
                "strategy_available",
                "blocked_reason",
                "decision_reason",
            ),
            strategy_column_candidates=("strategy_name", "strategy"),
            optional_columns=(
                "market",
                "signal_observation_date",
                "signal_date",
                "source_model",
                "vrp_har_gk",
                "har_forecast_available",
                "state_name",
                "p_calm",
                "p_transition",
                "p_stress",
            ),
            aliases={
                "strategy": "strategy_name",
                "signal_date": "signal_observation_date",
            },
            target_exposure_min=-1.0,
            target_exposure_max=0.0,
            latest_signal_sort_column="target_trade_date",
            approved_strategies=APPROVED_STRATEGIES,
        )


@dataclass(frozen=True)
class SignalFreshnessResult:
    """Signal freshness evaluation."""

    is_stale: bool
    signal_date: date
    as_of_date: date
    age_days: int
    max_signal_age_days: int
    allow_weekend_gap: bool
    final_status_if_blocked: str | None
    reason: str


def load_signal_frame(path: str | Path) -> pd.DataFrame:
    """Load a Phase 9 signal frame from parquet or CSV.

    Parquet is the expected production format. CSV support exists only for
    tests and quick manual inspection.
    """

    signal_path = Path(path)
    if not signal_path.exists():
        raise SignalSchemaError(f"Signal file not found: {signal_path}")

    suffix = signal_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(signal_path)

    if suffix == ".csv":
        return pd.read_csv(signal_path)

    raise SignalSchemaError(
        f"Unsupported signal file format {suffix!r}. "
        "Expected .parquet or .csv."
    )


def canonicalize_signal_frame(
    df: pd.DataFrame,
    schema: Phase9SignalSchema | None = None,
) -> pd.DataFrame:
    """Return a copy of the signal frame using canonical Phase 9 names."""

    schema = schema or Phase9SignalSchema.default()

    if not isinstance(df, pd.DataFrame):
        raise SignalSchemaError("Signal input must be a pandas DataFrame")

    out = df.copy()

    for source, target in schema.aliases.items():
        if source in out.columns and target not in out.columns:
            out = out.rename(columns={source: target})

    if "strategy_name" not in out.columns:
        available = [col for col in schema.strategy_column_candidates if col in out.columns]
        if not available:
            raise SignalSchemaError(
                "Signal frame must include strategy_name or strategy column"
            )

        first_strategy_col = available[0]
        out = out.rename(columns={first_strategy_col: "strategy_name"})

    return out


def validate_signal_schema(
    df: pd.DataFrame,
    schema: Phase9SignalSchema | None = None,
) -> pd.DataFrame:
    """Validate and normalize a Phase 9 signal frame.

    Important Phase 9 convention:
    unavailable terminal rows may have:
        target_trade_date = NaT
        target_exposure = NaN
        strategy_available = False

    Those rows are allowed in the file, but they are not selectable for a
    latest target_trade_date run. Available rows must still have valid dates
    and exposures.
    """

    schema = schema or Phase9SignalSchema.default()
    out = canonicalize_signal_frame(df, schema)

    required = set(schema.required_columns).union({"strategy_name"})
    missing = sorted(required.difference(out.columns))

    if missing:
        raise SignalSchemaError(f"Signal frame missing required columns: {missing}")

    if out.empty:
        raise SignalSchemaError("Signal frame is empty")

    out["strategy_name"] = out["strategy_name"].astype(str)

    invalid_strategies = sorted(
        set(out["strategy_name"].dropna().astype(str)).difference(
            schema.approved_strategies
        )
    )

    if invalid_strategies:
        raise SignalSchemaError(
            "Signal frame contains strategies outside approved universe: "
            f"{invalid_strategies}"
        )

    out["strategy_available"] = _parse_boolean_like_series(
        out["strategy_available"],
        column_name="strategy_available",
    )

    out["target_trade_date"] = _parse_date_series(
        out["target_trade_date"],
        column_name="target_trade_date",
        allow_missing=True,
    )

    if "signal_observation_date" in out.columns:
        out["signal_observation_date"] = _parse_date_series(
            out["signal_observation_date"],
            column_name="signal_observation_date",
            allow_missing=True,
        )

    out["target_exposure"] = _parse_numeric_series(
        out["target_exposure"],
        column_name="target_exposure",
        allow_missing=True,
    )

    available_mask = out["strategy_available"]

    available_missing_date = out[available_mask & out["target_trade_date"].isna()]
    if not available_missing_date.empty:
        sample = available_missing_date.head(5).to_dict("records")
        raise SignalSchemaError(
            "Available strategy rows must have valid target_trade_date. "
            f"Bad sample: {sample}"
        )

    available_missing_exposure = out[available_mask & out["target_exposure"].isna()]
    if not available_missing_exposure.empty:
        sample = available_missing_exposure.head(5).to_dict("records")
        raise SignalSchemaError(
            "Available strategy rows must have valid target_exposure. "
            f"Bad sample: {sample}"
        )

    exposure_mask = out["target_exposure"].notna()
    exposure_values = out.loc[exposure_mask, "target_exposure"]

    invalid_exposure = out[
        exposure_mask
        & (
            (out["target_exposure"] < schema.target_exposure_min)
            | (out["target_exposure"] > schema.target_exposure_max)
        )
    ]

    if not invalid_exposure.empty:
        min_seen = exposure_values.min()
        max_seen = exposure_values.max()
        raise SignalSchemaError(
            "target_exposure out of allowed range "
            f"[{schema.target_exposure_min}, {schema.target_exposure_max}]. "
            f"Observed min={min_seen}, max={max_seen}."
        )

    for text_col in ("blocked_reason", "decision_reason"):
        out[text_col] = out[text_col].fillna("").astype(str)

    if schema.latest_signal_sort_column not in out.columns:
        raise SignalSchemaError(
            f"Missing latest signal sort column: {schema.latest_signal_sort_column}"
        )

    return out


def select_latest_signal(
    df: pd.DataFrame,
    *,
    strategy_name: str | None = None,
    market: str | None = None,
    schema: Phase9SignalSchema | None = None,
) -> pd.Series:
    """Select the latest Phase 9 signal by target_trade_date.

    This deliberately ignores file row order.

    Rows with missing target_trade_date are allowed in the source file but are
    not selectable for a target-trade-date-based Phase 11 run.
    """

    schema = schema or Phase9SignalSchema.default()
    validated = validate_signal_schema(df, schema)

    filtered = validated

    if strategy_name is not None:
        if strategy_name not in schema.approved_strategies:
            raise SignalSchemaError(
                f"Requested strategy {strategy_name!r} is not approved"
            )

        filtered = filtered[filtered["strategy_name"] == strategy_name]

    if market is not None and "market" in filtered.columns:
        market_upper = market.upper()
        filtered = filtered[filtered["market"].astype(str).str.upper() == market_upper]

    if filtered.empty:
        details = []
        if strategy_name is not None:
            details.append(f"strategy={strategy_name}")
        if market is not None:
            details.append(f"market={market.upper()}")
        suffix = f" for {', '.join(details)}" if details else ""
        raise SignalSchemaError(f"No Phase 9 signal rows found{suffix}")

    sort_column = schema.latest_signal_sort_column
    selectable = filtered[filtered[sort_column].notna()].copy()

    if selectable.empty:
        details = []
        if strategy_name is not None:
            details.append(f"strategy={strategy_name}")
        if market is not None:
            details.append(f"market={market.upper()}")
        suffix = f" for {', '.join(details)}" if details else ""
        raise SignalSchemaError(
            f"No selectable Phase 9 rows with valid {sort_column}{suffix}"
        )

    selectable = selectable.sort_values([sort_column], ascending=True)

    latest_date = selectable[sort_column].max()
    latest_rows = selectable[selectable[sort_column] == latest_date]

    if len(latest_rows) > 1:
        latest_rows = _break_latest_tie(latest_rows)

    return latest_rows.iloc[-1].copy()


def check_signal_freshness(
    latest_signal: pd.Series | Mapping[str, Any],
    *,
    as_of_date: date | datetime | str | None = None,
    max_signal_age_days: int = 5,
    allow_weekend_gap: bool = True,
    block_stale_signal: bool = True,
) -> SignalFreshnessResult:
    """Check whether the selected latest signal is stale."""

    if max_signal_age_days <= 0:
        raise SignalSchemaError("max_signal_age_days must be positive")

    signal_value = latest_signal["target_trade_date"]
    signal_date = _coerce_single_date(signal_value, field_name="target_trade_date")
    today = _resolve_as_of_date(as_of_date)

    age_days = (today - signal_date).days

    if age_days < 0:
        return SignalFreshnessResult(
            is_stale=False,
            signal_date=signal_date,
            as_of_date=today,
            age_days=age_days,
            max_signal_age_days=max_signal_age_days,
            allow_weekend_gap=allow_weekend_gap,
            final_status_if_blocked=None,
            reason="signal target_trade_date is in the future relative to as_of_date",
        )

    effective_max_age = max_signal_age_days
    if allow_weekend_gap and max_signal_age_days < 3:
        effective_max_age = 3

    is_stale = age_days > effective_max_age
    final_status = "BLOCKED_STALE_SIGNAL" if is_stale and block_stale_signal else None

    if is_stale:
        reason = (
            f"latest signal target_trade_date={signal_date.isoformat()} is "
            f"{age_days} days old, exceeding max allowed age "
            f"{effective_max_age} days"
        )
    else:
        reason = (
            f"latest signal target_trade_date={signal_date.isoformat()} is "
            f"{age_days} days old, within max allowed age "
            f"{effective_max_age} days"
        )

    return SignalFreshnessResult(
        is_stale=is_stale,
        signal_date=signal_date,
        as_of_date=today,
        age_days=age_days,
        max_signal_age_days=max_signal_age_days,
        allow_weekend_gap=allow_weekend_gap,
        final_status_if_blocked=final_status,
        reason=reason,
    )


def load_validate_select_latest_signal(
    signal_path: str | Path,
    *,
    config: BrokerConfig | None = None,
    schema: Phase9SignalSchema | None = None,
    strategy_name: str | None = None,
    market: str | None = None,
) -> pd.Series:
    """Load a Phase 9 signal file and return the latest valid signal."""

    active_schema = schema
    if active_schema is None and config is not None:
        active_schema = Phase9SignalSchema.from_broker_config(config)
    if active_schema is None:
        active_schema = Phase9SignalSchema.default()

    frame = load_signal_frame(signal_path)
    return select_latest_signal(
        frame,
        strategy_name=strategy_name,
        market=market,
        schema=active_schema,
    )


def signal_freshness_from_config(
    latest_signal: pd.Series | Mapping[str, Any],
    config: BrokerConfig,
    *,
    as_of_date: date | datetime | str | None = None,
) -> SignalFreshnessResult:
    """Run signal freshness check using config values."""

    freshness_config = config.raw.get("signal_freshness")
    if not isinstance(freshness_config, Mapping):
        raise SignalSchemaError("Config missing signal_freshness mapping")

    return check_signal_freshness(
        latest_signal,
        as_of_date=as_of_date,
        max_signal_age_days=int(freshness_config.get("max_signal_age_days", 5)),
        allow_weekend_gap=bool(freshness_config.get("allow_weekend_gap", True)),
        block_stale_signal=bool(freshness_config.get("block_stale_signal", True)),
    )


def latest_signal_to_record(latest_signal: pd.Series | Mapping[str, Any]) -> dict[str, Any]:
    """Convert latest signal row into JSON/CSV-friendly dictionary."""

    if isinstance(latest_signal, pd.Series):
        raw = latest_signal.to_dict()
    else:
        raw = dict(latest_signal)

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, pd.Timestamp):
            out[str(key)] = value.date().isoformat()
        elif isinstance(value, (datetime, date)):
            out[str(key)] = value.isoformat()
        else:
            out[str(key)] = value

    return out


def _parse_date_series(
    series: pd.Series,
    *,
    column_name: str,
    allow_missing: bool = False,
) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")

    invalid_mask = parsed.isna() & ~series.isna()
    if invalid_mask.any():
        bad_values = series[invalid_mask].head(5).tolist()
        raise SignalSchemaError(
            f"Column {column_name!r} contains unparseable dates: {bad_values}"
        )

    return parsed.dt.date


def _parse_numeric_series(
    series: pd.Series,
    *,
    column_name: str,
    allow_missing: bool = False,
) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")

    invalid_mask = parsed.isna() & ~series.isna()
    if invalid_mask.any():
        bad_values = series[invalid_mask].head(5).tolist()
        raise SignalSchemaError(
            f"Column {column_name!r} contains non-numeric values: {bad_values}"
        )

    return parsed.astype(float)


def _parse_boolean_like_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.map(_coerce_bool_like)

    invalid_mask = normalized.isna()
    if invalid_mask.any():
        bad_values = series[invalid_mask].head(5).tolist()
        raise SignalSchemaError(
            f"Column {column_name!r} contains non-boolean values: {bad_values}"
        )

    return normalized.astype(bool)


def _coerce_bool_like(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in {"true", "t", "1", "yes", "y"}:
            return True

        if cleaned in {"false", "f", "0", "no", "n"}:
            return False

    return None


def _break_latest_tie(latest_rows: pd.DataFrame) -> pd.DataFrame:
    """Deterministically break ties among same target_trade_date rows."""

    rows = latest_rows.copy()

    if "signal_observation_date" in rows.columns:
        rows = rows.sort_values(["signal_observation_date"], ascending=True)

    if "strategy_available" in rows.columns:
        rows = rows.sort_values(["strategy_available"], ascending=True)

    return rows


def _coerce_single_date(value: Any, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        raise SignalSchemaError(f"Could not parse {field_name}: {value!r}")

    if isinstance(parsed, pd.Timestamp):
        return parsed.date()

    raise SignalSchemaError(f"Could not parse {field_name}: {value!r}")


def _resolve_as_of_date(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()

    return _coerce_single_date(value, field_name="as_of_date")