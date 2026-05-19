"""
Validation utilities for Phase 6 Gaussian HMM regime modelling.

This module centralizes:
- feature legality checks
- feature availability checks
- candidate model rejection rules
- covariance and transition-matrix validation
- filtered probability validation
- model-failure reason mapping
- threshold/crisis leakage guards

It deliberately does not fit models and does not write files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vrp.regimes.hmm_registry import (
    HMM_ALLOWED_FAILURE_REASONS,
    HMM_ALLOWED_COVARIANCE_TYPES,
    HMM_ALLOWED_N_STATES,
    HMM_CANDIDATE_RANKING_COLUMNS,
    HMM_FEATURE_AVAILABILITY_COLUMNS,
    HMM_FORBIDDEN_FEATURE_CONTAINS,
    HMM_FORBIDDEN_FEATURE_EXACT,
    HMM_FORBIDDEN_FEATURE_PREFIXES,
    HMM_FORBIDDEN_FEATURE_SUBSTRINGS,
    assert_hmm_covariance_type_is_valid,
    assert_hmm_feature_set_is_valid,
    assert_hmm_features_are_point_in_time,
    assert_hmm_n_states_is_valid,
    get_hmm_conditional_features,
    get_hmm_feature_columns,
    get_hmm_required_condition,
    normalize_hmm_feature_name,
    validate_hmm_model_spec,
)


@dataclass(frozen=True)
class HMMValidationRules:
    """
    Default Phase 6 candidate rejection thresholds.

    These values mirror the approved Phase 6 plan. Keep the defaults simple and
    overridable by config later.
    """

    min_train_state_occupancy: float = 0.05
    min_test_state_occupancy: float = 0.02
    near_absorbing_transition_threshold: float = 0.995
    probability_row_sum_atol: float = 1.0e-8
    min_covariance_eigenvalue: float = 1.0e-10
    min_covariance_diagonal: float = 1.0e-10
    max_covariance_condition_number: float = 1.0e12
    min_eligible_observations: int = 1000
    min_eligible_fraction: float = 0.50
    reject_non_converged: bool = True
    reject_economic_monotonicity_failure: bool = True
    reject_feature_availability_failure: bool = True


@dataclass(frozen=True)
class ProbabilityValidationResult:
    """Summary of filtered-probability validation."""

    passed: bool
    row_sum_min: float
    row_sum_max: float
    max_abs_row_sum_error: float
    rejection_reason: str = ""


@dataclass(frozen=True)
class StateOccupancyResult:
    """State occupancy diagnostics for one segment."""

    counts: dict[int, int]
    fractions: dict[int, float]
    min_occupancy: float
    n_observations: int


@dataclass(frozen=True)
class FeatureAvailabilityRecord:
    """
    One row for reports/tables/phase_6/hmm_feature_availability.csv.
    """

    market: str
    feature_set: str
    required_feature: str
    source_column: str
    required_condition: str
    n_total_rows: int
    n_missing_feature: int
    n_condition_failed: int
    n_eligible_rows_after_feature: int
    first_available_date: str
    last_available_date: str
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return record as a dict using the approved table schema."""
        return {
            "market": self.market,
            "feature_set": self.feature_set,
            "required_feature": self.required_feature,
            "source_column": self.source_column,
            "required_condition": self.required_condition,
            "n_total_rows": self.n_total_rows,
            "n_missing_feature": self.n_missing_feature,
            "n_condition_failed": self.n_condition_failed,
            "n_eligible_rows_after_feature": self.n_eligible_rows_after_feature,
            "first_available_date": self.first_available_date,
            "last_available_date": self.last_available_date,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class FeatureAvailabilitySummary:
    """Feature-set level availability summary."""

    market: str
    feature_set: str
    passed: bool
    n_total_rows: int
    n_eligible_rows: int
    eligible_fraction: float
    first_available_date: str
    last_available_date: str
    blocked_reason: str
    records: tuple[FeatureAvailabilityRecord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateValidationResult:
    """
    Candidate-level validation result.

    This object is later converted into the candidate ranking table.
    """

    market: str
    feature_set: str
    n_states: int
    covariance_type: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    hmm_model_valid: bool
    hmm_model_failure_reason: str
    train_occupancy: StateOccupancyResult | None = None
    test_occupancy: StateOccupancyResult | None = None
    probability_validation: ProbabilityValidationResult | None = None
    covariance_valid: bool | None = None
    transition_matrix_valid: bool | None = None
    economic_monotonicity_passed: bool | None = None
    feature_availability_passed: bool | None = None
    converged: bool | None = None

    @property
    def rejection_reason_text(self) -> str:
        """Return semicolon-separated rejection reasons for CSV output."""
        return "; ".join(self.rejection_reasons)

    def base_ranking_fields(self) -> dict[str, Any]:
        """
        Return validation fields compatible with hmm_candidate_model_ranking.csv.

        The model-fitting layer will add likelihood/AIC/BIC fields.
        """
        train_min = (
            self.train_occupancy.min_occupancy
            if self.train_occupancy is not None
            else np.nan
        )
        test_min = (
            self.test_occupancy.min_occupancy
            if self.test_occupancy is not None
            else np.nan
        )

        row = {
            "market": self.market,
            "feature_set": self.feature_set,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "min_state_occupancy_train": train_min,
            "min_state_occupancy_test": test_min,
            "economic_monotonicity_passed": self.economic_monotonicity_passed,
            "selected_primary": False,
            "rejection_reason": self.rejection_reason_text,
        }

        return row


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_numeric_array(values: Any, *, name: str) -> np.ndarray:
    """Convert values to a finite numeric numpy array."""
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def _safe_date_bounds(df: pd.DataFrame, mask: pd.Series, date_col: str) -> tuple[str, str]:
    """Return first/last available date for rows where mask is True."""
    if date_col not in df.columns or not bool(mask.any()):
        return "", ""

    dates = pd.to_datetime(df.loc[mask, date_col], errors="coerce").dropna()
    if dates.empty:
        return "", ""

    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _to_bool_condition(series: pd.Series) -> pd.Series:
    """
    Convert a condition column to boolean.

    Handles bools, 0/1, and string representations. Missing values become False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "available"})


def _clean_reason_text(reasons: Sequence[str]) -> str:
    """Return a semicolon-separated reason string without blanks."""
    clean = [str(reason).strip() for reason in reasons if str(reason).strip()]
    return "; ".join(clean)


# ---------------------------------------------------------------------------
# Leakage and feature legality guards
# ---------------------------------------------------------------------------


def assert_no_threshold_or_crisis_feature_use(
    feature_cols: Iterable[str],
    *,
    context: str = "hmm_features",
) -> None:
    """
    Raise if threshold/crisis/HMM-derived columns are used as HMM features.

    Phase 6 may compare against threshold/crisis diagnostics after HMM states exist.
    These columns must not enter the HMM feature matrix.
    """
    for feature in feature_cols:
        lower = normalize_hmm_feature_name(feature).lower()

        if lower in HMM_FORBIDDEN_FEATURE_EXACT:
            raise ValueError(
                f"{context}: forbidden HMM feature {feature!r}; exact forbidden match."
            )

        for prefix in HMM_FORBIDDEN_FEATURE_PREFIXES:
            if lower.startswith(prefix):
                raise ValueError(
                    f"{context}: forbidden HMM feature {feature!r}; "
                    f"starts with forbidden prefix {prefix!r}."
                )

        for token in HMM_FORBIDDEN_FEATURE_CONTAINS:
            if token in lower:
                raise ValueError(
                    f"{context}: forbidden HMM feature {feature!r}; "
                    f"contains forbidden token {token!r}."
                )


def assert_no_forward_or_label_feature_use(
    feature_cols: Iterable[str],
    *,
    context: str = "hmm_features",
) -> None:
    """
    Raise if forward/expost/future/label-like columns are used as HMM features.
    """
    for feature in feature_cols:
        lower = normalize_hmm_feature_name(feature).lower()

        for token in HMM_FORBIDDEN_FEATURE_SUBSTRINGS:
            if token in lower:
                raise ValueError(
                    f"{context}: forbidden HMM feature {feature!r}; "
                    f"contains forbidden substring {token!r}."
                )


def assert_hmm_feature_columns_are_legal(
    feature_cols: Iterable[str],
    *,
    context: str = "hmm_features",
) -> list[str]:
    """
    Normalize and validate HMM feature columns.

    Returns canonical feature names if valid.
    """
    canonical = [normalize_hmm_feature_name(feature) for feature in feature_cols]
    assert_hmm_features_are_point_in_time(canonical)
    assert_no_forward_or_label_feature_use(canonical, context=context)
    assert_no_threshold_or_crisis_feature_use(canonical, context=context)
    return canonical


def validate_crisis_windows_usage(
    *,
    used_for: str,
    diagnostics_only: bool = True,
) -> None:
    """
    Validate that crisis windows are diagnostic-only.

    Allowed uses:
    - crisis_stress_overlap
    - crisis_lead_lag
    - crisis_false_negative_days
    """
    allowed = {
        "crisis_stress_overlap",
        "crisis_lead_lag",
        "crisis_false_negative_days",
        "diagnostic_reporting",
    }
    forbidden = {
        "tuning_feature_set",
        "tuning_n_states",
        "tuning_covariance_type",
        "state_mapping",
        "manual_stress_assignment",
        "model_selection",
        "training_target",
        "training_feature",
    }

    normalized = str(used_for).strip().lower()

    if not diagnostics_only:
        raise ValueError("Crisis windows must remain diagnostics-only in Phase 6.")

    if normalized in forbidden:
        raise ValueError(f"Crisis-window use is forbidden for {used_for!r}.")

    if normalized not in allowed:
        raise ValueError(
            f"Unknown crisis-window use {used_for!r}. "
            f"Allowed diagnostic uses: {sorted(allowed)}."
        )


def validate_threshold_comparison_usage(
    *,
    threshold_state_as_feature: bool = False,
    threshold_state_as_target: bool = False,
    choose_model_by_threshold_match: bool = False,
) -> None:
    """
    Validate Phase 5 threshold state usage.

    Threshold regimes are allowed only as post-HMM comparison diagnostics.
    """
    if threshold_state_as_feature:
        raise ValueError("threshold_state cannot be used as an HMM feature.")

    if threshold_state_as_target:
        raise ValueError("threshold_state cannot be used as a supervised HMM target.")

    if choose_model_by_threshold_match:
        raise ValueError("HMM model selection cannot be based on threshold-state matching.")


# ---------------------------------------------------------------------------
# Feature availability
# ---------------------------------------------------------------------------


def build_feature_availability_summary(
    df: pd.DataFrame,
    *,
    market: str,
    feature_set: str,
    feature_cols: Iterable[str] | None = None,
    date_col: str = "date",
    conditional_features: Mapping[str, str] | None = None,
    rules: HMMValidationRules | None = None,
) -> FeatureAvailabilitySummary:
    """
    Build feature availability records for one market and feature set.

    This function does not drop rows. It reports how many rows would remain after
    applying feature-missingness and conditional availability rules.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not market:
        raise ValueError("market cannot be empty.")

    assert_hmm_feature_set_is_valid(feature_set)

    rules = rules or HMMValidationRules()
    conditional_features = dict(
        get_hmm_conditional_features()
        if conditional_features is None
        else conditional_features
    )

    if feature_cols is None:
        features = get_hmm_feature_columns(feature_set)
    else:
        features = assert_hmm_feature_columns_are_legal(feature_cols)

    n_total = int(len(df))
    if n_total == 0:
        records = [
            FeatureAvailabilityRecord(
                market=market,
                feature_set=feature_set,
                required_feature=feature,
                source_column=feature,
                required_condition=conditional_features.get(feature, ""),
                n_total_rows=0,
                n_missing_feature=0,
                n_condition_failed=0,
                n_eligible_rows_after_feature=0,
                first_available_date="",
                last_available_date="",
                blocked_reason="insufficient_data",
            )
            for feature in features
        ]
        return FeatureAvailabilitySummary(
            market=market,
            feature_set=feature_set,
            passed=False,
            n_total_rows=0,
            n_eligible_rows=0,
            eligible_fraction=0.0,
            first_available_date="",
            last_available_date="",
            blocked_reason="insufficient_data",
            records=tuple(records),
        )

    eligible_mask = pd.Series(True, index=df.index, dtype=bool)
    per_feature_stats: dict[str, dict[str, Any]] = {}

    for feature in features:
        condition_col = conditional_features.get(feature, "")
        blocked_reasons: list[str] = []

        if feature not in df.columns:
            feature_present_mask = pd.Series(False, index=df.index, dtype=bool)
            n_missing_feature = n_total
            blocked_reasons.append("missing_feature_column")
        else:
            feature_present_mask = df[feature].notna()
            n_missing_feature = int((~feature_present_mask).sum())

        if condition_col:
            if condition_col not in df.columns:
                condition_mask = pd.Series(False, index=df.index, dtype=bool)
                n_condition_failed = n_total
                blocked_reasons.append("missing_required_condition_column")
            else:
                condition_mask = _to_bool_condition(df[condition_col])
                n_condition_failed = int((~condition_mask).sum())
        else:
            condition_mask = pd.Series(True, index=df.index, dtype=bool)
            n_condition_failed = 0

        feature_eligible_mask = feature_present_mask & condition_mask
        eligible_mask &= feature_eligible_mask

        per_feature_stats[feature] = {
            "condition_col": condition_col,
            "n_missing_feature": n_missing_feature,
            "n_condition_failed": n_condition_failed,
            "blocked_reasons": tuple(blocked_reasons),
        }

    n_eligible = int(eligible_mask.sum())
    eligible_fraction = float(n_eligible / n_total) if n_total > 0 else 0.0
    first_date, last_date = _safe_date_bounds(df, eligible_mask, date_col)

    summary_reasons: list[str] = []
    if n_eligible < rules.min_eligible_observations:
        summary_reasons.append("insufficient_eligible_observations")
    if eligible_fraction < rules.min_eligible_fraction:
        summary_reasons.append("feature_availability_too_low")
    for feature, stats in per_feature_stats.items():
        summary_reasons.extend(stats["blocked_reasons"])

    passed = len(summary_reasons) == 0

    records: list[FeatureAvailabilityRecord] = []
    for feature in features:
        stats = per_feature_stats[feature]
        feature_reasons = list(stats["blocked_reasons"])
        if n_eligible < rules.min_eligible_observations:
            feature_reasons.append("insufficient_eligible_observations")
        if eligible_fraction < rules.min_eligible_fraction:
            feature_reasons.append("feature_availability_too_low")

        records.append(
            FeatureAvailabilityRecord(
                market=market,
                feature_set=feature_set,
                required_feature=feature,
                source_column=feature,
                required_condition=str(stats["condition_col"]),
                n_total_rows=n_total,
                n_missing_feature=int(stats["n_missing_feature"]),
                n_condition_failed=int(stats["n_condition_failed"]),
                n_eligible_rows_after_feature=n_eligible,
                first_available_date=first_date,
                last_available_date=last_date,
                blocked_reason=_clean_reason_text(feature_reasons),
            )
        )

    return FeatureAvailabilitySummary(
        market=market,
        feature_set=feature_set,
        passed=passed,
        n_total_rows=n_total,
        n_eligible_rows=n_eligible,
        eligible_fraction=eligible_fraction,
        first_available_date=first_date,
        last_available_date=last_date,
        blocked_reason=_clean_reason_text(summary_reasons),
        records=tuple(records),
    )


def feature_availability_records_to_frame(
    records: Iterable[FeatureAvailabilityRecord],
) -> pd.DataFrame:
    """Convert feature availability records to the approved report schema."""
    rows = [record.to_dict() for record in records]
    frame = pd.DataFrame(rows)

    for col in HMM_FEATURE_AVAILABILITY_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.Series(dtype="object")

    return frame.loc[:, list(HMM_FEATURE_AVAILABILITY_COLUMNS)]


# ---------------------------------------------------------------------------
# State occupancy validation
# ---------------------------------------------------------------------------


def compute_state_occupancy(
    states: Sequence[int] | np.ndarray,
    *,
    n_states: int,
) -> StateOccupancyResult:
    """Compute count and fraction per raw HMM state."""
    assert_hmm_n_states_is_valid(n_states)

    arr = np.asarray(states)
    if arr.size == 0:
        counts = {state: 0 for state in range(n_states)}
        fractions = {state: 0.0 for state in range(n_states)}
        return StateOccupancyResult(
            counts=counts,
            fractions=fractions,
            min_occupancy=0.0,
            n_observations=0,
        )

    if not np.issubdtype(arr.dtype, np.integer):
        if np.all(np.equal(arr, arr.astype(int))):
            arr = arr.astype(int)
        else:
            raise ValueError("states must contain integer state labels.")

    invalid = sorted(set(arr.tolist()) - set(range(n_states)))
    if invalid:
        raise ValueError(f"states contain invalid labels for n_states={n_states}: {invalid}")

    n_obs = int(arr.size)
    counts = {state: int((arr == state).sum()) for state in range(n_states)}
    fractions = {state: float(counts[state] / n_obs) for state in range(n_states)}
    min_occupancy = min(fractions.values()) if fractions else 0.0

    return StateOccupancyResult(
        counts=counts,
        fractions=fractions,
        min_occupancy=float(min_occupancy),
        n_observations=n_obs,
    )


def state_occupancy_rejection_reasons(
    train_states: Sequence[int] | np.ndarray | None,
    test_states: Sequence[int] | np.ndarray | None,
    *,
    n_states: int,
    rules: HMMValidationRules | None = None,
) -> tuple[list[str], StateOccupancyResult | None, StateOccupancyResult | None]:
    """
    Validate train/test state occupancy.

    Returns (reasons, train_occupancy, test_occupancy).
    """
    rules = rules or HMMValidationRules()
    assert_hmm_n_states_is_valid(n_states)

    reasons: list[str] = []
    train_occ: StateOccupancyResult | None = None
    test_occ: StateOccupancyResult | None = None

    if train_states is None:
        reasons.append("missing_train_state_sequence")
    else:
        train_occ = compute_state_occupancy(train_states, n_states=n_states)
        if train_occ.min_occupancy < rules.min_train_state_occupancy:
            reasons.append(
                f"min_train_state_occupancy_lt_{rules.min_train_state_occupancy:g}"
            )

    if test_states is None:
        reasons.append("missing_test_state_sequence")
    else:
        test_occ = compute_state_occupancy(test_states, n_states=n_states)
        if test_occ.min_occupancy < rules.min_test_state_occupancy:
            reasons.append(f"min_test_state_occupancy_lt_{rules.min_test_state_occupancy:g}")

    return reasons, train_occ, test_occ


# ---------------------------------------------------------------------------
# Transition-matrix validation
# ---------------------------------------------------------------------------


def validate_transition_matrix(
    transmat: Any,
    *,
    n_states: int,
    rules: HMMValidationRules | None = None,
) -> tuple[bool, list[str]]:
    """Validate HMM transition matrix shape, row sums, and near-absorbing states."""
    rules = rules or HMMValidationRules()
    assert_hmm_n_states_is_valid(n_states)

    reasons: list[str] = []

    try:
        matrix = _as_numeric_array(transmat, name="transmat")
    except ValueError as exc:
        return False, [f"invalid_transition_matrix:{exc}"]

    if matrix.shape != (n_states, n_states):
        return False, [
            f"invalid_transition_matrix_shape:{matrix.shape};expected:{(n_states, n_states)}"
        ]

    if np.any(matrix < -rules.probability_row_sum_atol):
        reasons.append("transition_matrix_has_negative_entries")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=rules.probability_row_sum_atol):
        reasons.append("transition_matrix_row_sum_failed")

    diagonal = np.diag(matrix)
    if np.any(diagonal > rules.near_absorbing_transition_threshold):
        reasons.append(
            f"near_absorbing_transition_gt_{rules.near_absorbing_transition_threshold:g}"
        )

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Covariance validation
# ---------------------------------------------------------------------------


def _state_covariance_matrix(
    covars: np.ndarray,
    *,
    covariance_type: str,
    state: int,
) -> np.ndarray:
    """
    Return full covariance matrix for one state.

    Supports hmmlearn-style diag and full covariance layouts.
    """
    if covariance_type == "diag":
        if covars.ndim == 2:
            return np.diag(covars[state])
        if covars.ndim == 3:
            return np.diag(np.diag(covars[state]))
        raise ValueError(f"Unsupported diag covariance shape: {covars.shape}")

    if covariance_type == "full":
        if covars.ndim != 3:
            raise ValueError(f"Full covariance must have 3 dimensions. Got {covars.shape}")
        return covars[state]

    raise ValueError(f"Unsupported covariance_type={covariance_type!r}.")


def validate_covariances(
    covars: Any,
    *,
    covariance_type: str,
    n_states: int,
    n_features: int,
    rules: HMMValidationRules | None = None,
) -> tuple[bool, list[str]]:
    """Validate HMM covariance arrays for numerical stability."""
    rules = rules or HMMValidationRules()
    assert_hmm_n_states_is_valid(n_states)
    assert_hmm_covariance_type_is_valid(covariance_type)

    reasons: list[str] = []

    try:
        arr = _as_numeric_array(covars, name="covars")
    except ValueError as exc:
        return False, [f"unstable_covariance:{exc}"]

    if covariance_type == "diag":
        valid_shape = arr.shape in {
            (n_states, n_features),
            (n_states, n_features, n_features),
        }
    else:
        valid_shape = arr.shape == (n_states, n_features, n_features)

    if not valid_shape:
        return False, [
            f"invalid_covariance_shape:{arr.shape};"
            f"n_states:{n_states};n_features:{n_features};type:{covariance_type}"
        ]

    for state in range(n_states):
        try:
            cov = _state_covariance_matrix(
                arr,
                covariance_type=covariance_type,
                state=state,
            )
        except ValueError as exc:
            return False, [f"unstable_covariance:{exc}"]

        if cov.shape != (n_features, n_features):
            reasons.append(f"invalid_state_covariance_shape_state_{state}:{cov.shape}")
            continue

        diagonal = np.diag(cov)
        if np.any(diagonal <= rules.min_covariance_diagonal):
            reasons.append(f"covariance_diagonal_too_small_state_{state}")

        if not np.allclose(cov, cov.T, atol=1.0e-10):
            reasons.append(f"covariance_not_symmetric_state_{state}")

        try:
            eigvals = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            reasons.append(f"covariance_eigendecomposition_failed_state_{state}")
            continue

        if np.any(eigvals <= rules.min_covariance_eigenvalue):
            reasons.append(f"covariance_non_positive_definite_state_{state}")

        cond = np.linalg.cond(cov)
        if not np.isfinite(cond) or cond > rules.max_covariance_condition_number:
            reasons.append(f"covariance_condition_number_unstable_state_{state}")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Probability validation
# ---------------------------------------------------------------------------


def validate_filtered_probability_matrix(
    probs: Any,
    *,
    n_states: int,
    rules: HMMValidationRules | None = None,
) -> ProbabilityValidationResult:
    """Validate custom forward-filtered probability matrix."""
    rules = rules or HMMValidationRules()
    assert_hmm_n_states_is_valid(n_states)

    try:
        arr = _as_numeric_array(probs, name="filtered_probs")
    except ValueError as exc:
        return ProbabilityValidationResult(
            passed=False,
            row_sum_min=np.nan,
            row_sum_max=np.nan,
            max_abs_row_sum_error=np.nan,
            rejection_reason=f"filtered_probability_invalid:{exc}",
        )

    if arr.ndim != 2 or arr.shape[1] != n_states:
        return ProbabilityValidationResult(
            passed=False,
            row_sum_min=np.nan,
            row_sum_max=np.nan,
            max_abs_row_sum_error=np.nan,
            rejection_reason=(
                f"filtered_probability_invalid_shape:{arr.shape};"
                f"expected_second_dim:{n_states}"
            ),
        )

    if np.any(arr < -rules.probability_row_sum_atol):
        return ProbabilityValidationResult(
            passed=False,
            row_sum_min=np.nan,
            row_sum_max=np.nan,
            max_abs_row_sum_error=np.nan,
            rejection_reason="filtered_probability_negative_entries",
        )

    row_sums = arr.sum(axis=1)
    row_sum_min = float(row_sums.min()) if row_sums.size else np.nan
    row_sum_max = float(row_sums.max()) if row_sums.size else np.nan
    max_abs_error = float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else np.nan

    passed = bool(np.allclose(row_sums, 1.0, atol=rules.probability_row_sum_atol))
    reason = "" if passed else "filtered_probability_row_sum_failed"

    return ProbabilityValidationResult(
        passed=passed,
        row_sum_min=row_sum_min,
        row_sum_max=row_sum_max,
        max_abs_row_sum_error=max_abs_error,
        rejection_reason=reason,
    )


# ---------------------------------------------------------------------------
# Convergence and failure reason mapping
# ---------------------------------------------------------------------------


def extract_hmm_convergence_status(model: Any) -> tuple[bool | None, int | None]:
    """
    Extract convergence status and iteration count from a fitted hmmlearn model.

    Returns (converged, n_iter). If monitor metadata is unavailable, returns
    (None, None).
    """
    monitor = getattr(model, "monitor_", None)
    if monitor is None:
        return None, None

    converged = getattr(monitor, "converged", None)
    n_iter = getattr(monitor, "iter", None)

    if converged is not None:
        converged = bool(converged)

    if n_iter is not None:
        n_iter = int(n_iter)

    return converged, n_iter


def infer_hmm_model_failure_reason(rejection_reasons: Iterable[str]) -> str:
    """
    Map detailed rejection reasons to approved high-level failure reason.
    """
    text = " ".join(str(reason).lower() for reason in rejection_reasons)

    priority_map = (
        ("insufficient_data", ("insufficient_data", "insufficient_eligible_observations")),
        ("feature_availability_too_low", ("feature_availability_too_low",)),
        ("state_collapse", ("occupancy", "state_collapse")),
        ("non_convergence", ("non_convergence", "does_not_converge")),
        ("unstable_covariance", ("covariance",)),
        (
            "economically_uninterpretable_states",
            ("economic_monotonicity_failed", "economically_uninterpretable"),
        ),
        ("poor_oos_likelihood", ("poor_oos_likelihood",)),
        ("filtered_probability_invalid", ("filtered_probability",)),
    )

    for failure_reason, tokens in priority_map:
        if any(token in text for token in tokens):
            return failure_reason

    return "unknown_failure" if text.strip() else ""


def assert_valid_hmm_failure_reason(reason: str) -> None:
    """Raise if reason is not in the approved Phase 6 model-failure vocabulary."""
    if reason and reason not in HMM_ALLOWED_FAILURE_REASONS:
        raise ValueError(
            f"Invalid HMM failure reason {reason!r}. "
            f"Allowed values: {HMM_ALLOWED_FAILURE_REASONS}."
        )


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


def validate_hmm_candidate_result(
    *,
    market: str,
    feature_set: str,
    n_states: int,
    covariance_type: str,
    train_states: Sequence[int] | np.ndarray | None = None,
    test_states: Sequence[int] | np.ndarray | None = None,
    transmat: Any | None = None,
    covars: Any | None = None,
    n_features: int | None = None,
    filtered_probs: Any | None = None,
    converged: bool | None = None,
    economic_monotonicity_passed: bool | None = None,
    feature_availability_passed: bool | None = None,
    rules: HMMValidationRules | None = None,
) -> CandidateValidationResult:
    """
    Validate one fitted HMM candidate.

    This function is intentionally conservative. Missing diagnostics are treated
    as rejection reasons only where the approved Phase 6 rules require them.
    """
    if not market:
        raise ValueError("market cannot be empty.")

    validate_hmm_model_spec(feature_set, n_states, covariance_type)
    rules = rules or HMMValidationRules()

    rejection_reasons: list[str] = []

    occupancy_reasons, train_occ, test_occ = state_occupancy_rejection_reasons(
        train_states,
        test_states,
        n_states=n_states,
        rules=rules,
    )
    rejection_reasons.extend(occupancy_reasons)

    transition_valid: bool | None = None
    if transmat is None:
        rejection_reasons.append("missing_transition_matrix")
    else:
        transition_valid, transition_reasons = validate_transition_matrix(
            transmat,
            n_states=n_states,
            rules=rules,
        )
        rejection_reasons.extend(transition_reasons)

    covariance_valid: bool | None = None
    if covars is None:
        rejection_reasons.append("missing_covariance")
    elif n_features is None:
        rejection_reasons.append("missing_n_features_for_covariance_validation")
    else:
        covariance_valid, covariance_reasons = validate_covariances(
            covars,
            covariance_type=covariance_type,
            n_states=n_states,
            n_features=int(n_features),
            rules=rules,
        )
        rejection_reasons.extend(covariance_reasons)

    probability_validation: ProbabilityValidationResult | None = None
    if filtered_probs is None:
        rejection_reasons.append("missing_filtered_probabilities")
    else:
        probability_validation = validate_filtered_probability_matrix(
            filtered_probs,
            n_states=n_states,
            rules=rules,
        )
        if not probability_validation.passed:
            rejection_reasons.append(probability_validation.rejection_reason)

    if rules.reject_non_converged:
        if converged is None:
            rejection_reasons.append("missing_convergence_status")
        elif not bool(converged):
            rejection_reasons.append("non_convergence")

    if rules.reject_economic_monotonicity_failure:
        if economic_monotonicity_passed is None:
            rejection_reasons.append("missing_economic_monotonicity_check")
        elif not bool(economic_monotonicity_passed):
            rejection_reasons.append("economic_monotonicity_failed")

    if rules.reject_feature_availability_failure:
        if feature_availability_passed is None:
            rejection_reasons.append("missing_feature_availability_check")
        elif not bool(feature_availability_passed):
            rejection_reasons.append("feature_availability_too_low")

    clean_reasons = tuple(reason for reason in rejection_reasons if reason)
    failure_reason = infer_hmm_model_failure_reason(clean_reasons)
    assert_valid_hmm_failure_reason(failure_reason)

    passed = len(clean_reasons) == 0

    return CandidateValidationResult(
        market=market,
        feature_set=feature_set,
        n_states=n_states,
        covariance_type=covariance_type,
        passed=passed,
        rejection_reasons=clean_reasons,
        hmm_model_valid=passed,
        hmm_model_failure_reason=failure_reason,
        train_occupancy=train_occ,
        test_occupancy=test_occ,
        probability_validation=probability_validation,
        covariance_valid=covariance_valid,
        transition_matrix_valid=transition_valid,
        economic_monotonicity_passed=economic_monotonicity_passed,
        feature_availability_passed=feature_availability_passed,
        converged=converged,
    )


def candidate_validation_results_to_frame(
    results: Iterable[CandidateValidationResult],
) -> pd.DataFrame:
    """
    Convert candidate validation results to a partial candidate-ranking frame.

    The model-fitting layer will fill likelihood/AIC/BIC and selected_primary.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {col: np.nan for col in HMM_CANDIDATE_RANKING_COLUMNS}
        row.update(result.base_ranking_fields())
        # store converged as float (1.0, 0.0) or NaN to satisfy numeric column expectations
        if result.converged is True:
            row["converged"] = 1.0
        elif result.converged is False:
            row["converged"] = 0.0
        else:
            row["converged"] = np.nan
        row["rejection_reason"] = result.rejection_reason_text
        rows.append(row)

    frame = pd.DataFrame(rows)
    for col in HMM_CANDIDATE_RANKING_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    return frame.loc[:, list(HMM_CANDIDATE_RANKING_COLUMNS)]


# ---------------------------------------------------------------------------
# Schema guards for later chunks
# ---------------------------------------------------------------------------


def assert_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Raise if required columns are missing from a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{context}: df must be a pandas DataFrame.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def assert_candidate_grid_values_are_valid(
    *,
    feature_sets: Iterable[str],
    n_states_values: Iterable[int],
    covariance_types: Iterable[str],
) -> None:
    """Validate candidate-grid config values."""
    for feature_set in feature_sets:
        assert_hmm_feature_set_is_valid(str(feature_set))

    for n_states in n_states_values:
        assert_hmm_n_states_is_valid(int(n_states))

    for covariance_type in covariance_types:
        assert_hmm_covariance_type_is_valid(str(covariance_type))


def assert_output_probability_policy_is_safe(
    *,
    uses_custom_forward_filter: bool,
    uses_hmmlearn_predict_proba_for_backtest: bool,
    uses_smoothed_probabilities_for_backtest: bool,
) -> None:
    """Validate mandatory Phase 6 probability-output policy."""
    if not uses_custom_forward_filter:
        raise ValueError("Phase 6 requires custom forward-filtered probabilities.")

    if uses_hmmlearn_predict_proba_for_backtest:
        raise ValueError("hmmlearn predict_proba cannot be used for backtest-facing output.")

    if uses_smoothed_probabilities_for_backtest:
        raise ValueError("Smoothed probabilities are diagnostic-only, not backtest-facing.")


__all__ = [
    "HMMValidationRules",
    "ProbabilityValidationResult",
    "StateOccupancyResult",
    "FeatureAvailabilityRecord",
    "FeatureAvailabilitySummary",
    "CandidateValidationResult",
    "assert_no_threshold_or_crisis_feature_use",
    "assert_no_forward_or_label_feature_use",
    "assert_hmm_feature_columns_are_legal",
    "validate_crisis_windows_usage",
    "validate_threshold_comparison_usage",
    "build_feature_availability_summary",
    "feature_availability_records_to_frame",
    "compute_state_occupancy",
    "state_occupancy_rejection_reasons",
    "validate_transition_matrix",
    "validate_covariances",
    "validate_filtered_probability_matrix",
    "extract_hmm_convergence_status",
    "infer_hmm_model_failure_reason",
    "assert_valid_hmm_failure_reason",
    "validate_hmm_candidate_result",
    "candidate_validation_results_to_frame",
    "assert_required_columns",
    "assert_candidate_grid_values_are_valid",
    "assert_output_probability_policy_is_safe",
]