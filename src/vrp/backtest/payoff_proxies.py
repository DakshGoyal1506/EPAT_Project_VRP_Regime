from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vrp.backtest.backtest_config import (
    BacktestConfig,
    get_market_backtest_inputs,
)
from vrp.backtest.backtest_registry import (
    BACKTEST_STRATEGY_UNIVERSE,
    assert_no_msvol_strategy_use,
    assert_no_outcome_labels_used_as_signals,
    assert_no_smoothed_probability_use,
    assert_outcome_alignment_allowed,
    assert_strategy_universe_locked,
)
from vrp.backtest.costs import apply_costs_to_backtest_panel


PRIMARY_PAYOFF_LABEL = "vrp_forward_expost_gk_label"

ACCEPTED_OUTCOME_DATE_COLUMNS: tuple[str, ...] = (
    "signal_observation_date",
    "date",
    "outcome_label_date",
)

REQUIRED_SIGNAL_COLUMNS: tuple[str, ...] = (
    "market",
    "strategy_name",
    "signal_observation_date",
    "target_trade_date",
    "target_exposure",
    "strategy_available",
)

EXCLUSION_REASONS: tuple[str, ...] = (
    "available",
    "strategy_unavailable",
    "missing_target_trade_date",
    "non_finite_exposure",
    "missing_payoff_label",
    "missing_outcome_join",
    "invalid_strategy_name",
)


class PayoffProxyError(ValueError):
    """Raised when Phase 10 payoff construction is invalid."""


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise PayoffProxyError(f"Unsupported input file extension: {path}")


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise PayoffProxyError(f"{label} missing required columns: {missing}")


def _find_outcome_date_column(df: pd.DataFrame) -> str:
    for col in ACCEPTED_OUTCOME_DATE_COLUMNS:
        if col in df.columns:
            return col

    raise PayoffProxyError(
        "Outcome panel has no accepted date column. "
        f"Accepted columns: {list(ACCEPTED_OUTCOME_DATE_COLUMNS)}"
    )


def _normalize_date_series(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all() and len(parsed) > 0:
        raise PayoffProxyError(f"Could not parse any values in date column {name!r}.")
    return parsed.dt.normalize()


def _parse_strategy_available(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    return normalized.isin(true_values)


def _validate_exclusion_reasons(df: pd.DataFrame) -> None:
    if "exclusion_reason" not in df.columns:
        raise PayoffProxyError("Missing exclusion_reason column.")

    observed = set(df["exclusion_reason"].dropna().astype(str).unique())
    allowed = set(EXCLUSION_REASONS)

    extra = sorted(observed - allowed)
    if extra:
        raise PayoffProxyError(f"Unknown exclusion_reason values: {extra}")


def build_forward_vrp_outcome_panel(
    vrp_df: pd.DataFrame,
    label_col: str = PRIMARY_PAYOFF_LABEL,
) -> pd.DataFrame:
    """
    Build realised-outcome panel for Phase 10.

    The forward/ex-post VRP label is allowed here only as realised outcome,
    not as a strategy signal.
    """
    if label_col not in vrp_df.columns:
        raise PayoffProxyError(f"Outcome label column not found: {label_col!r}")

    date_col = _find_outcome_date_column(vrp_df)

    keep_cols = [date_col, label_col]
    if "market" in vrp_df.columns:
        keep_cols.insert(0, "market")

    out = vrp_df.loc[:, keep_cols].copy()
    out["outcome_label_date"] = _normalize_date_series(out[date_col], date_col)
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")

    if "market" in out.columns:
        out["market"] = out["market"].astype(str).str.upper()
        duplicate_key = ["market", "outcome_label_date"]
    else:
        duplicate_key = ["outcome_label_date"]

    duplicate_mask = out.duplicated(duplicate_key, keep=False)
    if bool(duplicate_mask.any()):
        n_duplicates = int(duplicate_mask.sum())
        raise PayoffProxyError(
            f"Duplicate outcome rows found by key {duplicate_key}. "
            f"Duplicate row count={n_duplicates}."
        )

    final_cols = []
    if "market" in out.columns:
        final_cols.append("market")
    final_cols.extend(["outcome_label_date", label_col])

    return out.loc[:, final_cols].sort_values(final_cols[:-1]).reset_index(drop=True)


def join_strategy_with_outcome(
    signals_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    alignment: str = "signal_observation_date",
    label_col: str = PRIMARY_PAYOFF_LABEL,
) -> pd.DataFrame:
    """
    Join Phase 9 signal rows to realised outcome labels.

    Default alignment:
        outcome_label_date = signal_observation_date

    This preserves:
        signal_observation_date
        target_trade_date
        outcome_label_date
    """
    assert_outcome_alignment_allowed(alignment)
    _require_columns(signals_df, REQUIRED_SIGNAL_COLUMNS, "signals_df")

    if "outcome_label_date" not in outcome_df.columns:
        outcome_df = build_forward_vrp_outcome_panel(outcome_df, label_col=label_col)

    if label_col not in outcome_df.columns:
        raise PayoffProxyError(f"outcome_df missing label column: {label_col}")

    out = signals_df.copy()
    out["market"] = out["market"].astype(str).str.upper()
    out["strategy_name"] = out["strategy_name"].astype(str)

    out["signal_observation_date"] = _normalize_date_series(
        out["signal_observation_date"],
        "signal_observation_date",
    )
    out["target_trade_date"] = _normalize_date_series(
        out["target_trade_date"],
        "target_trade_date",
    )

    out["outcome_label_date"] = _normalize_date_series(out[alignment], alignment)

    outcome = outcome_df.copy()
    outcome["outcome_label_date"] = _normalize_date_series(
        outcome["outcome_label_date"],
        "outcome_label_date",
    )

    if "market" in outcome.columns:
        outcome["market"] = outcome["market"].astype(str).str.upper()
        merge_keys = ["market", "outcome_label_date"]
    else:
        merge_keys = ["outcome_label_date"]

    merged = out.merge(
        outcome,
        how="left",
        on=merge_keys,
        indicator="_outcome_merge",
        validate="many_to_one",
    )

    return merged


def compute_forward_vrp_strategy_payoff(
    df: pd.DataFrame,
    exposure_col: str = "target_exposure",
    label_col: str = PRIMARY_PAYOFF_LABEL,
) -> pd.DataFrame:
    """
    Compute Phase 10 research payoff proxy.

    gross_return_proxy = -target_exposure_for_backtest * vrp_forward_expost_gk_label

    Unavailable rows are excluded from active strategy return calculations.
    Valid flat rows remain eligible with target_exposure_for_backtest = 0.0.
    """
    required = (
        "market",
        "strategy_name",
        "signal_observation_date",
        "target_trade_date",
        "outcome_label_date",
        exposure_col,
        "strategy_available",
        label_col,
    )
    _require_columns(df, required, "backtest join panel")

    out = df.copy()

    out["market"] = out["market"].astype(str).str.upper()
    out["strategy_name"] = out["strategy_name"].astype(str)

    for col in ("signal_observation_date", "target_trade_date", "outcome_label_date"):
        out[col] = _normalize_date_series(out[col], col)

    exposure_numeric = pd.to_numeric(out[exposure_col], errors="coerce")
    label_numeric = pd.to_numeric(out[label_col], errors="coerce")
    strategy_available = _parse_strategy_available(out["strategy_available"])

    valid_strategy = out["strategy_name"].isin(BACKTEST_STRATEGY_UNIVERSE)
    has_target_trade_date = out["target_trade_date"].notna()
    finite_exposure = np.isfinite(exposure_numeric.to_numpy(dtype=float, na_value=np.nan))
    has_label = label_numeric.notna()

    if "_outcome_merge" in out.columns:
        outcome_joined = out["_outcome_merge"].astype(str).eq("both")
    else:
        outcome_joined = has_label.copy()

    exclusion_reason = pd.Series("available", index=out.index, dtype="object")

    exclusion_reason.loc[~valid_strategy] = "invalid_strategy_name"
    exclusion_reason.loc[valid_strategy & ~strategy_available] = "strategy_unavailable"
    exclusion_reason.loc[
        valid_strategy & strategy_available & ~has_target_trade_date
    ] = "missing_target_trade_date"
    exclusion_reason.loc[
        valid_strategy & strategy_available & has_target_trade_date & ~finite_exposure
    ] = "non_finite_exposure"
    exclusion_reason.loc[
        valid_strategy
        & strategy_available
        & has_target_trade_date
        & finite_exposure
        & ~outcome_joined
    ] = "missing_outcome_join"
    exclusion_reason.loc[
        valid_strategy
        & strategy_available
        & has_target_trade_date
        & finite_exposure
        & outcome_joined
        & ~has_label
    ] = "missing_payoff_label"

    is_eligible = exclusion_reason.eq("available")

    out["target_exposure_for_backtest"] = np.nan
    out.loc[is_eligible, "target_exposure_for_backtest"] = exposure_numeric.loc[is_eligible]

    out["gross_return_proxy"] = np.nan
    out.loc[is_eligible, "gross_return_proxy"] = (
        -out.loc[is_eligible, "target_exposure_for_backtest"].astype(float)
        * label_numeric.loc[is_eligible].astype(float)
    )

    out["is_backtest_eligible"] = is_eligible.astype(bool)
    out["exclusion_reason"] = exclusion_reason

    _validate_exclusion_reasons(out)
    validate_payoff_sign_convention(out, exposure_col="target_exposure_for_backtest", label_col=label_col)

    return out


def validate_payoff_sign_convention(
    df: pd.DataFrame,
    *,
    exposure_col: str = "target_exposure_for_backtest",
    label_col: str = PRIMARY_PAYOFF_LABEL,
    gross_col: str = "gross_return_proxy",
    eligible_col: str = "is_backtest_eligible",
    atol: float = 1e-12,
) -> None:
    required = (exposure_col, label_col, gross_col, eligible_col)
    _require_columns(df, required, "payoff panel")

    eligible = df[eligible_col].fillna(False).astype(bool)
    if not bool(eligible.any()):
        return

    expected = (
        -pd.to_numeric(df.loc[eligible, exposure_col], errors="coerce")
        * pd.to_numeric(df.loc[eligible, label_col], errors="coerce")
    )
    observed = pd.to_numeric(df.loc[eligible, gross_col], errors="coerce")

    if not np.allclose(observed.to_numpy(), expected.to_numpy(), atol=atol, rtol=0.0):
        raise PayoffProxyError(
            "Payoff sign convention violated. Expected "
            "gross_return_proxy = -target_exposure_for_backtest * payoff_label."
        )


def build_research_backtest_panel(
    market: str,
    config: BacktestConfig,
    *,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    """
    Build the full Phase 10 research backtest panel for one market.

    This function does not write files. The vectorised engine in Chunk 4 will
    handle persistence and metadata sidecars.
    """
    market_key = market.upper()
    inputs = get_market_backtest_inputs(config, market_key)

    signals = _read_table(inputs["strategy_signals"])

    assert_no_msvol_strategy_use(signals)
    assert_no_smoothed_probability_use(signals)

    signal_feature_candidates = [
        col for col in signals.columns
        if col not in {
            "market",
            "strategy_name",
            "signal_observation_date",
            "target_trade_date",
            "target_exposure",
            "strategy_available",
        }
    ]
    assert_no_outcome_labels_used_as_signals(signal_feature_candidates)

    strategies = sorted(signals["strategy_name"].dropna().astype(str).unique().tolist())
    assert_strategy_universe_locked(strategies)

    outcome = _load_first_valid_outcome(inputs, config.primary_payoff.label_col)

    outcome_panel = build_forward_vrp_outcome_panel(
        outcome,
        label_col=config.primary_payoff.label_col,
    )
    joined = join_strategy_with_outcome(
        signals,
        outcome_panel,
        alignment=config.primary_payoff.outcome_alignment,
        label_col=config.primary_payoff.label_col,
    )
    payoff_panel = compute_forward_vrp_strategy_payoff(
        joined,
        exposure_col="target_exposure",
        label_col=config.primary_payoff.label_col,
    )

    effective_cost_bps = (
        float(config.costs.default_cost_bps)
        if cost_bps is None
        else float(cost_bps)
    )

    final_panel = apply_costs_to_backtest_panel(
        payoff_panel,
        enabled=config.costs.enabled,
        cost_bps=effective_cost_bps,
    )

    return final_panel


def _load_first_valid_outcome(
    inputs: dict[str, Path],
    label_col: str,
) -> pd.DataFrame:
    """
    Prefer vrp_har, then vrp. Both are audited in Chunk 0.

    This function does not download or construct any new data.
    """
    candidate_keys = ("vrp_har", "vrp")
    errors: list[str] = []

    for key in candidate_keys:
        path = inputs[key]
        try:
            df = _read_table(path)
        except Exception as exc:
            errors.append(f"{key}={path}: read failed: {exc}")
            continue

        if label_col not in df.columns:
            errors.append(f"{key}={path}: missing {label_col}")
            continue

        try:
            _find_outcome_date_column(df)
        except Exception as exc:
            errors.append(f"{key}={path}: missing outcome date column: {exc}")
            continue

        return df

    raise PayoffProxyError(
        "No valid Phase 10 outcome input found. Tried vrp_har then vrp. "
        + " | ".join(errors)
    )