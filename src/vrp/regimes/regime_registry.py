from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd


REGIME_ALLOWED_BASE_FEATURES: List[str] = [
    "iv_ann",
    "iv_close",
    "rv_gk_22d_ann_lag1",
    "vrp_backward_gk",
    "vrp_backward_gk_positive",
    "har_rv_gk_22d_forecast_ann",
    "vrp_har_gk",
    "vrp_har_gk_positive",
]

REGIME_CONDITIONAL_FEATURES: Dict[str, str] = {
    "har_rv_gk_22d_forecast_ann": "har_forecast_available",
    "vrp_har_gk": "har_forecast_available",
    "vrp_har_gk_positive": "har_forecast_available",
}

REGIME_FORBIDDEN_FEATURE_SUBSTRINGS: List[str] = [
    "future",
    "forward",
    "expost",
    "label",
]

REGIME_ALLOWED_AUXILIARY_INPUTS: List[str] = [
    "date",
    "market",
    "close",
    "adj_close",
    "underlying_close",
    "log_return",
    "simple_return",
    "har_forecast_available",
    "har_blocked_reason",
    "feature_allowed",
]

REGIME_ALLOWED_DIAGNOSTIC_LABELS: List[str] = [
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
]

REGIME_CONSTRUCTION_INPUTS: List[str] = sorted(
    set(REGIME_ALLOWED_BASE_FEATURES + REGIME_ALLOWED_AUXILIARY_INPUTS)
)


def _normalise_feature_cols(feature_cols: Iterable[str]) -> List[str]:
    if isinstance(feature_cols, str):
        raise TypeError(
            "feature_cols must be an iterable of column names, not a single string."
        )

    normalised: List[str] = []

    for col in feature_cols:
        if not isinstance(col, str):
            raise TypeError(f"Feature column names must be strings. Got {type(col)}.")
        stripped = col.strip()
        if not stripped:
            raise ValueError("Feature column names must not be empty strings.")
        normalised.append(stripped)

    return normalised


def _load_phase3_forbidden_substrings() -> List[str]:
    """
    Best-effort import of Phase 3 forbidden-substring policy.

    Phase 5 keeps its own registry, but imports the earlier policy when available
    so future changes to Phase 3 forbidden naming are not silently missed.
    """
    candidate_attr_names = [
        "FORBIDDEN_FEATURE_SUBSTRINGS",
        "FORBIDDEN_COLUMN_SUBSTRINGS",
        "FORBIDDEN_LABEL_SUBSTRINGS",
        "FEATURE_FORBIDDEN_SUBSTRINGS",
    ]

    try:
        from vrp.features import feature_registry as phase3_feature_registry
    except Exception:
        return []

    found: List[str] = []

    for attr_name in candidate_attr_names:
        value = getattr(phase3_feature_registry, attr_name, None)
        if value is None:
            continue
        if isinstance(value, str):
            found.append(value)
        else:
            found.extend([str(item) for item in value])

    return sorted({item.strip().lower() for item in found if str(item).strip()})


def _load_har_registry_output_columns() -> List[str]:
    """
    Best-effort import of Phase 4 HAR output column names.

    The registry does not depend on exact names inside har_registry.py. It scans
    common constant names and adds any string/list/tuple/set values it finds.
    """
    candidate_attr_names = [
        "HAR_OUTPUT_COLUMNS",
        "HAR_FORECAST_COLUMNS",
        "HAR_ALLOWED_OUTPUT_COLUMNS",
        "HAR_REGISTRY_OUTPUT_COLUMNS",
        "HAR_FEATURE_COLUMNS",
    ]

    try:
        from vrp.forecasting import har_registry
    except Exception:
        return []

    found: List[str] = []

    for attr_name in candidate_attr_names:
        value = getattr(har_registry, attr_name, None)
        if value is None:
            continue

        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                if isinstance(item, str):
                    found.append(item)
                elif isinstance(item, Iterable):
                    found.extend([str(x) for x in item])
        elif isinstance(value, Iterable):
            found.extend([str(item) for item in value])

    return sorted({item.strip() for item in found if str(item).strip()})


PHASE3_FORBIDDEN_FEATURE_SUBSTRINGS: List[str] = _load_phase3_forbidden_substrings()

HAR_REGISTRY_OUTPUT_COLUMNS: List[str] = _load_har_registry_output_columns()

ALL_REGIME_FORBIDDEN_FEATURE_SUBSTRINGS: List[str] = sorted(
    set(
        item.lower()
        for item in (
            REGIME_FORBIDDEN_FEATURE_SUBSTRINGS
            + PHASE3_FORBIDDEN_FEATURE_SUBSTRINGS
        )
        if item
    )
)


def assert_no_forbidden_regime_features(feature_cols: Iterable[str]) -> bool:
    """
    Reject columns that are forbidden for regime construction.

    Any feature containing future, forward, expost, or label is forbidden as a
    regime-construction feature. Those columns may only be used later in reporting
    diagnostics after threshold_state has already been assigned.
    """
    cols = _normalise_feature_cols(feature_cols)

    violations: Dict[str, List[str]] = {}

    for col in cols:
        lower_col = col.lower()
        matched = [
            token
            for token in ALL_REGIME_FORBIDDEN_FEATURE_SUBSTRINGS
            if token in lower_col
        ]
        if matched:
            violations[col] = matched

    if violations:
        details = "; ".join(
            f"{col}: {tokens}" for col, tokens in sorted(violations.items())
        )
        raise ValueError(
            "Forbidden regime-construction feature(s) detected. "
            "Forward/ex-post/label/future columns must not be used to construct "
            f"regime states. Violations: {details}"
        )

    return True


def assert_regime_features_are_point_in_time(feature_cols: Iterable[str]) -> bool:
    """
    Validate that regime-construction features are explicitly allowed and point-in-time.

    This function is intentionally stricter than a generic schema check. It is for
    features used to construct threshold/HMM/Markov regimes, not for post-regime
    diagnostics.
    """
    cols = _normalise_feature_cols(feature_cols)
    assert_no_forbidden_regime_features(cols)

    allowed = set(REGIME_CONSTRUCTION_INPUTS)
    allowed.update(HAR_REGISTRY_OUTPUT_COLUMNS)

    unknown = sorted(set(cols) - allowed)

    if unknown:
        raise ValueError(
            "Unknown or unapproved regime-construction feature(s): "
            f"{unknown}. Approved construction inputs are: {sorted(allowed)}."
        )

    return True


def get_allowed_regime_features() -> List[str]:
    """
    Return the core Phase 5+ regime features.

    This excludes auxiliary date/price/return/control columns and excludes all
    forward/ex-post labels.
    """
    return list(REGIME_ALLOWED_BASE_FEATURES)


def get_allowed_regime_construction_inputs(
    include_auxiliary: bool = True,
) -> List[str]:
    """
    Return approved point-in-time construction inputs.

    Auxiliary inputs include date, market, price/return columns for drawdown, and
    HAR availability flags. They are not forward labels.
    """
    if include_auxiliary:
        return list(REGIME_CONSTRUCTION_INPUTS)
    return list(REGIME_ALLOWED_BASE_FEATURES)


def get_conditional_regime_features() -> Dict[str, str]:
    """
    Return features that are valid only when an availability column is true.
    """
    return dict(REGIME_CONDITIONAL_FEATURES)


def get_allowed_diagnostic_labels() -> List[str]:
    """
    Return forward/ex-post labels allowed only for post-regime diagnostics.

    These must never be passed into assert_regime_features_are_point_in_time().
    """
    return list(REGIME_ALLOWED_DIAGNOSTIC_LABELS)


def is_allowed_diagnostic_label(column: str) -> bool:
    """
    Check whether a column is approved for post-regime diagnostics only.
    """
    return str(column).strip() in REGIME_ALLOWED_DIAGNOSTIC_LABELS


def _is_missing_scalar(value: object) -> bool:
    """
    Typed scalar missing check compatible with strict type checkers.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    return False


def is_regime_feature_available(row: Mapping[str, object] | pd.Series, feature_col: str) -> bool:
    """
    Check row-level availability of a regime feature.

    HAR-derived features are available only when har_forecast_available is true.
    Non-conditional features are available when the feature exists and is non-missing.
    """
    if not isinstance(feature_col, str) or not feature_col.strip():
        raise ValueError("feature_col must be a non-empty string.")

    feature_col = feature_col.strip()

    if feature_col not in REGIME_ALLOWED_BASE_FEATURES:
        raise ValueError(
            f"{feature_col!r} is not an approved core regime feature. "
            f"Approved features: {REGIME_ALLOWED_BASE_FEATURES}"
        )

    if feature_col not in row:
        return False

    value = row[feature_col]

    if _is_missing_scalar(value):
        return False

    availability_col = REGIME_CONDITIONAL_FEATURES.get(feature_col)

    if availability_col is None:
        return True

    if availability_col not in row:
        return False

    availability_value = row[availability_col]

    if _is_missing_scalar(availability_value):
        return False

    return bool(availability_value)


def get_forbidden_regime_feature_substrings() -> List[str]:
    """
    Return the effective forbidden-substring policy for regime construction.
    """
    return list(ALL_REGIME_FORBIDDEN_FEATURE_SUBSTRINGS)


def assert_diagnostic_labels_not_used_for_construction(
    feature_cols: Iterable[str],
) -> bool:
    """
    Explicit guard against accidentally passing approved diagnostic labels into
    construction functions.
    """
    cols = _normalise_feature_cols(feature_cols)
    diagnostic_used = sorted(set(cols) & set(REGIME_ALLOWED_DIAGNOSTIC_LABELS))

    if diagnostic_used:
        raise ValueError(
            "Diagnostic forward/ex-post label(s) were passed into regime "
            f"construction: {diagnostic_used}. These are allowed only after "
            "threshold_state has already been assigned."
        )

    return True


def split_construction_and_diagnostic_columns(
    columns: Iterable[str],
) -> Dict[str, List[str]]:
    """
    Split a set of columns into construction inputs, diagnostic labels, and unknowns.

    This is useful for no-lookahead audits and report-generation checks.
    """
    cols = _normalise_feature_cols(columns)

    construction = []
    diagnostic = []
    unknown = []

    construction_allowed = set(REGIME_CONSTRUCTION_INPUTS)
    diagnostic_allowed = set(REGIME_ALLOWED_DIAGNOSTIC_LABELS)

    for col in cols:
        if col in construction_allowed:
            construction.append(col)
        elif col in diagnostic_allowed:
            diagnostic.append(col)
        else:
            unknown.append(col)

    return {
        "construction": sorted(construction),
        "diagnostic": sorted(diagnostic),
        "unknown": sorted(unknown),
    }