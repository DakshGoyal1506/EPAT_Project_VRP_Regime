"""
Feature preparation for Phase 6 Gaussian HMM regimes.

This module builds the eligible HMM feature panel without fitting any scaler or
model. It enforces:

- approved HMM feature sets only
- no forward/expost/label/crisis/threshold leakage
- no silent feature forward-fill
- point-in-time return construction
- conditional HAR feature availability
- feature availability reporting before row drops

The output from this file feeds Chunk 5 scaling and Chunk 7 Gaussian HMM fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vrp.regimes.hmm_registry import (
    HMM_FEATURE_AVAILABILITY_COLUMNS,
    get_hmm_feature_columns,
    get_hmm_required_condition,
)
from vrp.regimes.hmm_validation import (
    FeatureAvailabilityRecord,
    FeatureAvailabilitySummary,
    assert_hmm_feature_columns_are_legal,
    feature_availability_records_to_frame,
)

DEFAULT_INDEX_RETURN_PREFERENCE: tuple[str, ...] = ("log_return", "simple_return")
DEFAULT_PRICE_COL_CANDIDATES: tuple[str, ...] = (
    "close",
    "adj_close",
    "underlying_close",
)
DEFAULT_IV_CHANGE_LAG: int = 1
DEFAULT_DATE_COL: str = "date"


@dataclass(frozen=True)
class HMMFeatureConstructionConfig:
    """
    Feature-construction configuration for Phase 6.

    Parameters mirror configs/model_hmm.yaml.
    """

    index_return_preference: tuple[str, ...] = DEFAULT_INDEX_RETURN_PREFERENCE
    price_col_candidates: tuple[str, ...] = DEFAULT_PRICE_COL_CANDIDATES
    iv_change_lag: int = DEFAULT_IV_CHANGE_LAG
    require_har_forecast_available_for_vrp_har: bool = True
    require_har_forecast_available_for_har_rv_forecast: bool = True
    block_rows_without_required_condition: bool = True
    do_not_forward_fill_features: bool = True
    date_col: str = DEFAULT_DATE_COL


@dataclass(frozen=True)
class HMMFeaturePanel:
    """
    Prepared HMM feature panel for one market/model feature set.

    Attributes
    ----------
    market:
        Market code, e.g. "US" or "INDIA".
    feature_set:
        HMM feature set name, e.g. "F3".
    feature_cols:
        Raw feature columns used for the HMM.
    source_column_map:
        Mapping from HMM feature to source column or construction expression.
    full_panel:
        Input panel after feature construction, before eligibility row dropping.
    eligible_panel:
        Rows eligible for fitting/filtering after feature and condition checks.
    eligibility_mask:
        Boolean mask aligned to full_panel.
    availability_summary:
        Feature availability summary.
    availability_table:
        DataFrame with the approved feature availability schema.
    blocked_rows:
        DataFrame with row-level blocked reasons.
    """

    market: str
    feature_set: str
    feature_cols: tuple[str, ...]
    source_column_map: Mapping[str, str]
    full_panel: pd.DataFrame
    eligible_panel: pd.DataFrame
    eligibility_mask: pd.Series
    availability_summary: FeatureAvailabilitySummary
    availability_table: pd.DataFrame
    blocked_rows: pd.DataFrame


def _require_dataframe(df: pd.DataFrame) -> None:
    """Validate DataFrame input."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame. Got {type(df)!r}.")


def _require_non_empty_string(value: str, *, name: str) -> None:
    """Validate non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _coerce_numeric(series: pd.Series, *, column: str) -> pd.Series:
    """Coerce a Series to numeric float values without filling missing values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float)


def _coerce_bool_condition(series: pd.Series) -> pd.Series:
    """
    Convert a condition column to boolean.

    Accepted truthy values:
    True, 1, "true", "yes", "y", "available"
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "available"})


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return first column from candidates present in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_date_string(value: Any) -> str:
    """Return ISO date string for reporting."""
    if pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def ensure_date_sorted_panel(
    df: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DataFrame:
    """
    Return a copy sorted by date.

    This is required before constructing returns, changes, and eligibility.
    """
    _require_dataframe(df)

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column {date_col!r}.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        bad_count = int(out[date_col].isna().sum())
        raise ValueError(f"Column {date_col!r} contains {bad_count} invalid date value(s).")

    out = out.sort_values(date_col).reset_index(drop=True)

    if out[date_col].duplicated().any():
        duplicate_count = int(out[date_col].duplicated().sum())
        raise ValueError(f"Column {date_col!r} contains {duplicate_count} duplicate date(s).")

    return out


def construct_index_return(
    df: pd.DataFrame,
    *,
    output_col: str = "index_return",
    return_preference: Sequence[str] = DEFAULT_INDEX_RETURN_PREFERENCE,
    price_col_candidates: Sequence[str] = DEFAULT_PRICE_COL_CANDIDATES,
) -> tuple[pd.DataFrame, str]:
    """
    Construct index_return using the approved priority:

    1. Use log_return if present.
    2. Else use simple_return if present.
    3. Else reconstruct log return from close/adj_close/underlying_close.

    No missing values are filled. The first reconstructed return is NaN.
    """
    _require_dataframe(df)
    out = df.copy()

    existing_return_col = _first_existing_column(out, list(return_preference))
    if existing_return_col is not None:
        out[output_col] = _coerce_numeric(out[existing_return_col], column=existing_return_col)
        return out, existing_return_col

    price_col = _first_existing_column(out, list(price_col_candidates))
    if price_col is None:
        out[output_col] = np.nan
        candidates = ", ".join(price_col_candidates)
        return out, f"unavailable:no_return_or_price_column:{candidates}"

    price = _coerce_numeric(out[price_col], column=price_col)

    if (price <= 0).any():
        bad_count = int((price <= 0).sum())
        raise ValueError(
            f"Cannot reconstruct index_return from {price_col!r}: "
            f"{bad_count} non-positive price value(s)."
        )

    out[output_col] = np.log(price / price.shift(1))
    return out, f"log({price_col}/{price_col}.shift(1))"


def construct_iv_ann_change(
    df: pd.DataFrame,
    *,
    iv_col: str = "iv_ann",
    output_col: str = "iv_ann_change_1d",
    lag: int = DEFAULT_IV_CHANGE_LAG,
) -> tuple[pd.DataFrame, str]:
    """
    Construct iv_ann_change_1d as iv_ann.diff(lag).

    No missing values are filled. This is a backward-looking/current-minus-prior
    feature.
    """
    _require_dataframe(df)

    if lag <= 0:
        raise ValueError(f"lag must be positive. Got {lag}.")

    out = df.copy()

    if iv_col not in out.columns:
        out[output_col] = np.nan
        return out, f"unavailable:missing:{iv_col}"

    out[output_col] = _coerce_numeric(out[iv_col], column=iv_col).diff(periods=lag)
    return out, f"{iv_col}.diff({lag})"


def prepare_hmm_base_panel(
    df: pd.DataFrame,
    *,
    config: HMMFeatureConstructionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Construct derived HMM features on a date-sorted panel.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        Prepared panel and source/construction mapping for derived features.
    """
    config = config or HMMFeatureConstructionConfig()
    out = ensure_date_sorted_panel(df, date_col=config.date_col)

    source_map: dict[str, str] = {}

    out, index_source = construct_index_return(
        out,
        output_col="index_return",
        return_preference=config.index_return_preference,
        price_col_candidates=config.price_col_candidates,
    )
    source_map["index_return"] = index_source

    out, iv_change_source = construct_iv_ann_change(
        out,
        iv_col="iv_ann",
        output_col="iv_ann_change_1d",
        lag=config.iv_change_lag,
    )
    source_map["iv_ann_change_1d"] = iv_change_source

    return out, source_map


def _feature_condition_column(
    feature: str,
    *,
    config: HMMFeatureConstructionConfig,
) -> str:
    """
    Return required condition column for a feature.

    The registry supplies base conditions. The config allows stricter HAR gating.
    """
    condition = get_hmm_required_condition(feature)

    if feature == "vrp_har_gk" and not config.require_har_forecast_available_for_vrp_har:
        return ""

    if (
        feature == "har_rv_gk_22d_forecast_ann"
        and not config.require_har_forecast_available_for_har_rv_forecast
    ):
        return ""

    return condition or ""


def build_hmm_eligibility_mask(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    config: HMMFeatureConstructionConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build row eligibility mask and blocked-row diagnostics.

    A row is eligible only if:
    - every required feature is present and non-missing
    - every required condition column exists and is True
    """
    _require_dataframe(df)
    config = config or HMMFeatureConstructionConfig()

    feature_cols = tuple(assert_hmm_feature_columns_are_legal(feature_cols))
    mask = pd.Series(True, index=df.index, dtype=bool)
    row_reasons: list[list[str]] = [[] for _ in range(len(df))]

    for feature in feature_cols:
        if feature not in df.columns:
            feature_ok = pd.Series(False, index=df.index, dtype=bool)
            reason = f"missing_feature:{feature}"
            for reasons in row_reasons:
                reasons.append(reason)
        else:
            feature_ok = df[feature].notna()
            bad_index = feature_ok.index[~feature_ok]
            for idx in bad_index:
                row_reasons[int(idx)].append(f"missing_feature_value:{feature}")

        mask &= feature_ok

        condition_col = _feature_condition_column(feature, config=config)
        if condition_col and config.block_rows_without_required_condition:
            if condition_col not in df.columns:
                condition_ok = pd.Series(False, index=df.index, dtype=bool)
                reason = f"missing_condition_column:{condition_col}"
                for reasons in row_reasons:
                    reasons.append(reason)
            else:
                condition_ok = _coerce_bool_condition(df[condition_col])
                bad_index = condition_ok.index[~condition_ok]
                for idx in bad_index:
                    row_reasons[int(idx)].append(f"condition_false:{condition_col}")

            mask &= condition_ok

    blocked = pd.DataFrame(
        {
            "row_index": df.index,
            config.date_col: df[config.date_col] if config.date_col in df.columns else pd.NaT,
            "hmm_feature_row_eligible": mask.values,
            "hmm_feature_blocked_reason": [
                "; ".join(dict.fromkeys(reasons)) for reasons in row_reasons
            ],
        }
    )

    blocked = blocked.loc[~blocked["hmm_feature_row_eligible"]].reset_index(drop=True)

    return mask, blocked


def _availability_record_for_feature(
    df: pd.DataFrame,
    *,
    market: str,
    feature_set: str,
    feature: str,
    source_column: str,
    condition_col: str,
    full_eligibility_mask_after_all_features: pd.Series,
    config: HMMFeatureConstructionConfig,
) -> FeatureAvailabilityRecord:
    """Build one feature availability report row."""
    n_total = int(len(df))

    if feature not in df.columns:
        n_missing_feature = n_total
        feature_present_mask = pd.Series(False, index=df.index, dtype=bool)
        blocked_reasons = ["missing_feature_column"]
    else:
        feature_present_mask = df[feature].notna()
        n_missing_feature = int((~feature_present_mask).sum())
        blocked_reasons = []

    if condition_col:
        if condition_col not in df.columns:
            condition_mask = pd.Series(False, index=df.index, dtype=bool)
            n_condition_failed = n_total
            blocked_reasons.append("missing_required_condition_column")
        else:
            condition_mask = _coerce_bool_condition(df[condition_col])
            n_condition_failed = int((~condition_mask).sum())
    else:
        condition_mask = pd.Series(True, index=df.index, dtype=bool)
        n_condition_failed = 0

    own_feature_mask = feature_present_mask & condition_mask
    own_dates = df.loc[own_feature_mask, config.date_col] if config.date_col in df.columns else []

    if len(own_dates) > 0:
        first_available = _safe_date_string(pd.Series(own_dates).min())
        last_available = _safe_date_string(pd.Series(own_dates).max())
    else:
        first_available = ""
        last_available = ""

    if n_missing_feature > 0:
        blocked_reasons.append("missing_feature_values")
    if n_condition_failed > 0:
        blocked_reasons.append("condition_failed")

    blocked_reason = "; ".join(dict.fromkeys(blocked_reasons))

    return FeatureAvailabilityRecord(
        market=market,
        feature_set=feature_set,
        required_feature=feature,
        source_column=source_column,
        required_condition=condition_col,
        n_total_rows=n_total,
        n_missing_feature=n_missing_feature,
        n_condition_failed=n_condition_failed,
        n_eligible_rows_after_feature=int(full_eligibility_mask_after_all_features.sum()),
        first_available_date=first_available,
        last_available_date=last_available,
        blocked_reason=blocked_reason,
    )


def build_hmm_feature_availability_summary(
    df: pd.DataFrame,
    *,
    market: str,
    feature_set: str,
    feature_cols: Sequence[str],
    source_column_map: Mapping[str, str],
    eligibility_mask: pd.Series,
    config: HMMFeatureConstructionConfig | None = None,
    min_eligible_observations: int = 1000,
    min_eligible_fraction: float = 0.50,
) -> FeatureAvailabilitySummary:
    """
    Build feature availability summary after construction but before dropping rows.
    """
    _require_dataframe(df)
    _require_non_empty_string(market, name="market")
    _require_non_empty_string(feature_set, name="feature_set")

    config = config or HMMFeatureConstructionConfig()
    feature_cols = tuple(assert_hmm_feature_columns_are_legal(feature_cols))

    if len(df) != len(eligibility_mask):
        raise ValueError("eligibility_mask length must match df length.")

    n_total = int(len(df))
    n_eligible = int(eligibility_mask.sum())
    eligible_fraction = float(n_eligible / n_total) if n_total > 0 else 0.0

    if n_eligible > 0 and config.date_col in df.columns:
        eligible_dates = df.loc[eligibility_mask, config.date_col]
        first_available_date = _safe_date_string(eligible_dates.min())
        last_available_date = _safe_date_string(eligible_dates.max())
    else:
        first_available_date = ""
        last_available_date = ""

    records: list[FeatureAvailabilityRecord] = []
    for feature in feature_cols:
        condition_col = _feature_condition_column(feature, config=config)
        source_column = source_column_map.get(feature, feature)
        records.append(
            _availability_record_for_feature(
                df,
                market=market,
                feature_set=feature_set,
                feature=feature,
                source_column=source_column,
                condition_col=condition_col,
                full_eligibility_mask_after_all_features=eligibility_mask,
                config=config,
            )
        )

    blocked_reasons: list[str] = []
    if n_eligible < min_eligible_observations:
        blocked_reasons.append("insufficient_eligible_observations")
    if eligible_fraction < min_eligible_fraction:
        blocked_reasons.append("feature_availability_too_low")

    for record in records:
        if record.blocked_reason:
            blocked_reasons.append(record.blocked_reason)

    passed = len(blocked_reasons) == 0
    blocked_reason = "; ".join(dict.fromkeys(blocked_reasons))

    return FeatureAvailabilitySummary(
        market=market,
        feature_set=feature_set,
        passed=passed,
        n_total_rows=n_total,
        n_eligible_rows=n_eligible,
        eligible_fraction=eligible_fraction,
        first_available_date=first_available_date,
        last_available_date=last_available_date,
        blocked_reason=blocked_reason,
        records=tuple(records),
    )


def build_hmm_feature_panel(
    df: pd.DataFrame,
    *,
    market: str,
    feature_set: str,
    config: HMMFeatureConstructionConfig | None = None,
    min_eligible_observations: int = 1000,
    min_eligible_fraction: float = 0.50,
) -> HMMFeaturePanel:
    """
    Build the full and eligible HMM feature panels for one market/feature set.

    Steps:
    1. Validate requested feature set.
    2. Sort by date.
    3. Construct derived features.
    4. Validate legal HMM feature names.
    5. Build eligibility mask.
    6. Build availability report before dropping rows.
    7. Return eligible panel.
    """
    _require_dataframe(df)
    _require_non_empty_string(market, name="market")
    _require_non_empty_string(feature_set, name="feature_set")

    config = config or HMMFeatureConstructionConfig()

    feature_cols = tuple(get_hmm_feature_columns(feature_set))
    feature_cols = tuple(assert_hmm_feature_columns_are_legal(feature_cols))

    panel, derived_source_map = prepare_hmm_base_panel(df, config=config)

    source_column_map: dict[str, str] = {feature: feature for feature in feature_cols}
    source_column_map.update(
        {
            feature: source
            for feature, source in derived_source_map.items()
            if feature in feature_cols
        }
    )

    eligibility_mask, blocked_rows = build_hmm_eligibility_mask(
        panel,
        feature_cols=feature_cols,
        config=config,
    )

    availability_summary = build_hmm_feature_availability_summary(
        panel,
        market=market,
        feature_set=feature_set,
        feature_cols=feature_cols,
        source_column_map=source_column_map,
        eligibility_mask=eligibility_mask,
        config=config,
        min_eligible_observations=min_eligible_observations,
        min_eligible_fraction=min_eligible_fraction,
    )

    availability_table = feature_availability_records_to_frame(
        availability_summary.records
    )

    eligible_panel = panel.loc[eligibility_mask].copy().reset_index(drop=True)

    if config.do_not_forward_fill_features:
        # Explicitly do nothing. This branch exists so callers can see that
        # missing values are intentionally dropped, not filled.
        pass

    return HMMFeaturePanel(
        market=market,
        feature_set=feature_set,
        feature_cols=feature_cols,
        source_column_map=source_column_map,
        full_panel=panel.reset_index(drop=True),
        eligible_panel=eligible_panel,
        eligibility_mask=eligibility_mask.reset_index(drop=True),
        availability_summary=availability_summary,
        availability_table=availability_table.loc[:, list(HMM_FEATURE_AVAILABILITY_COLUMNS)],
        blocked_rows=blocked_rows,
    )


def build_hmm_feature_panels_for_grid(
    df: pd.DataFrame,
    *,
    market: str,
    feature_sets: Iterable[str],
    config: HMMFeatureConstructionConfig | None = None,
    min_eligible_observations: int = 1000,
    min_eligible_fraction: float = 0.50,
) -> dict[str, HMMFeaturePanel]:
    """Build feature panels for multiple feature sets."""
    panels: dict[str, HMMFeaturePanel] = {}
    for feature_set in feature_sets:
        panels[feature_set] = build_hmm_feature_panel(
            df,
            market=market,
            feature_set=feature_set,
            config=config,
            min_eligible_observations=min_eligible_observations,
            min_eligible_fraction=min_eligible_fraction,
        )
    return panels


def concatenate_feature_availability_tables(
    panels: Iterable[HMMFeaturePanel],
) -> pd.DataFrame:
    """Concatenate feature availability tables from several panels."""
    frames = [panel.availability_table for panel in panels]
    if not frames:
        return pd.DataFrame(columns=list(HMM_FEATURE_AVAILABILITY_COLUMNS))

    out = pd.concat(frames, ignore_index=True)

    for col in HMM_FEATURE_AVAILABILITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")

    return out.loc[:, list(HMM_FEATURE_AVAILABILITY_COLUMNS)]


def get_feature_matrix(
    panel: HMMFeaturePanel,
    *,
    as_numpy: bool = True,
) -> pd.DataFrame | np.ndarray:
    """
    Extract the eligible raw feature matrix.

    Scaling happens in Chunk 5. This function returns unscaled raw features.
    """
    feature_frame = panel.eligible_panel.loc[:, list(panel.feature_cols)].copy()

    for col in panel.feature_cols:
        feature_frame[col] = _coerce_numeric(feature_frame[col], column=col)

    if feature_frame.isna().any().any():
        missing_counts = feature_frame.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Eligible feature matrix still contains missing values: {missing_counts}")

    if as_numpy:
        return feature_frame.to_numpy(dtype=float)

    return feature_frame


def assert_hmm_feature_panel_is_usable(
    panel: HMMFeaturePanel,
    *,
    min_observations: int = 1000,
) -> None:
    """Raise if a prepared HMM feature panel is not usable for model fitting."""
    if len(panel.eligible_panel) < min_observations:
        raise ValueError(
            f"HMM feature panel has insufficient eligible observations: "
            f"{len(panel.eligible_panel)} < {min_observations}. "
            f"Reason: {panel.availability_summary.blocked_reason}"
        )

    _ = get_feature_matrix(panel, as_numpy=True)


__all__ = [
    "HMMFeatureConstructionConfig",
    "HMMFeaturePanel",
    "ensure_date_sorted_panel",
    "construct_index_return",
    "construct_iv_ann_change",
    "prepare_hmm_base_panel",
    "build_hmm_eligibility_mask",
    "build_hmm_feature_availability_summary",
    "build_hmm_feature_panel",
    "build_hmm_feature_panels_for_grid",
    "concatenate_feature_availability_tables",
    "get_feature_matrix",
    "assert_hmm_feature_panel_is_usable",
]