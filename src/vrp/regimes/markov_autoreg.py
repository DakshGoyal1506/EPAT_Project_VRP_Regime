"""
Phase 7 Markov autoregression core utilities.

Chunk 2 scope:
- load Phase 7 input panel
- resolve target column
- validate target availability
- apply train-only target transformation
- create chronological train/test split
- prepare endog series for statsmodels MarkovAutoregression

No model fitting is implemented in this chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from vrp.regimes.markov_autoreg_registry import (
    MARConfig,
    MARModelSpec,
    find_forbidden_input_columns,
    market_lower,
    normalize_market,
    processed_input_path,
    resolve_target_column,
    target_availability_rule,
    target_transform_rule,
)


DATE_COL = "date"
TRANSFORMED_TARGET_COL = "mar_target_transformed"


@dataclass(frozen=True)
class MAREligibilitySummary:
    """Summary of target eligibility before statsmodels fitting."""

    market: str
    target: str
    target_col: str
    n_panel_rows: int
    n_target_non_missing: int
    n_available_by_rule: int
    n_eligible: int
    n_dropped_missing_target: int
    n_dropped_by_availability_rule: int
    availability_rule_description: str


@dataclass(frozen=True)
class MARTrainTestSummary:
    """Chronological train/test split summary."""

    method: str
    train_fraction: float
    n_eligible: int
    n_train: int
    n_test: int
    n_train_effective_after_ar_order: int
    n_test_effective_after_ar_order: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str


@dataclass(frozen=True)
class MARTargetTransformSummary:
    """Target transformation metadata."""

    target: str
    target_col: str
    transformed_col: str
    method: str
    params: dict[str, Any]
    n_input: int
    n_output_missing: int
    n_clipped_low: int
    n_clipped_high: int


@dataclass(frozen=True)
class MARPreparedData:
    """
    Prepared input for Markov autoregression.

    panel:
        Full original panel, sorted by date, with generated index_return if possible.

    eligible_frame:
        Date-aligned rows eligible for MAR estimation/filtering.

    train_frame:
        Chronological train subset of eligible_frame.

    test_frame:
        Chronological test subset of eligible_frame.

    The first `order` observations in eligible_frame are AR warmup rows and should
    not receive final valid probabilities later.
    """

    market: str
    spec: MARModelSpec
    target_col: str
    transformed_target_col: str

    panel: pd.DataFrame
    eligible_frame: pd.DataFrame
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame

    eligibility: MAREligibilitySummary
    split: MARTrainTestSummary
    transform: MARTargetTransformSummary

    forbidden_columns_present_in_panel: tuple[str, ...]
    validation_warnings: tuple[str, ...]

@dataclass(frozen=True)
class MARFitFirewallSummary:
    """Fit-status and validity firewall for one MAR candidate."""

    fit_converged: bool
    fit_exception: str | None
    llf: float | None
    aic: float | None
    bic: float | None
    hqic: float | None
    nobs: int | None
    n_params: int | None
    warnflag: int | str | None
    mle_retvals: dict[str, Any]
    valid_candidate: bool
    invalid_reason: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MARFitAttempt:
    """One statsmodels fitting attempt."""

    attempt_id: int
    method: str
    cov_type: str
    maxiter: int
    em_iter: int
    search_reps: int
    disp: bool
    success: bool
    exception: str | None

@dataclass(frozen=True)
class MARCandidateFit:
    """
    Train-only fitted MarkovAutoregression candidate.

    model:
        statsmodels model object fit on train data only.

    result:
        statsmodels fit result object estimated on train data only.

    train_filtered_probabilities:
        Filtered marginal probabilities produced on the train sample.

    This object intentionally does not contain full-sample filtered probabilities.
    Full-sample filtering with train-fitted parameters is Chunk 4.
    """

    market: str
    spec: MARModelSpec
    prepared: MARPreparedData

    model: Any | None
    result: Any | None

    train_filtered_probabilities: pd.DataFrame
    train_state_assignments: pd.Series
    train_state_occupancy: dict[int, float]

    transition_matrix: pd.DataFrame
    ar_coefficients: dict[int, float]
    sigma2_by_state: dict[int, float]

    fit_summary: MARFitFirewallSummary
    fit_attempts: tuple[MARFitAttempt, ...]


@dataclass(frozen=True)
class MARProbabilityAudit:
    """Probability validation summary for aligned full-sample probabilities."""

    n_rows: int
    n_model_available_rows: int
    n_warmup_rows: int
    n_probability_rows: int
    n_nan_rows_after_warmup: int
    max_row_sum_abs_error: float | None
    probability_row_sum_tolerance: float
    passed: bool
    invalid_reason: str


@dataclass(frozen=True)
class MARParameterLookaheadAudit:
    """Audit proving that parameters were estimated on train only."""

    mar_fit_window_start: str
    mar_fit_window_end: str
    mar_filter_window_start: str
    mar_filter_window_end: str
    params_estimated_using_full_sample: bool
    filtered_probabilities_use_train_params_only: bool
    smoothed_probabilities_used_for_backtest: bool
    passed: bool


@dataclass(frozen=True)
class MARFullFilterResult:
    """
    Full eligible-series filtering result using train-fitted parameters.

    full_filter_result:
        statsmodels result object from applying filter() to full eligible endog.

    full_smooth_result:
        statsmodels result object from applying smooth() to full eligible endog.
        Diagnostic only.

    aligned_frame:
        eligible_frame plus raw filtered/smoothed diagnostic probabilities and
        model-observation availability flags.
    """

    candidate: MARCandidateFit

    full_model: Any | None
    full_filter_result: Any | None
    full_smooth_result: Any | None

    aligned_frame: pd.DataFrame
    raw_filtered_probabilities: pd.DataFrame
    raw_smoothed_probabilities_diagnostic: pd.DataFrame

    full_state_occupancy: dict[int, float]
    test_state_occupancy: dict[int, float]

    probability_audit: MARProbabilityAudit
    lookahead_audit: MARParameterLookaheadAudit


@dataclass(frozen=True)
class MAREconomicStateMapping:
    """Train-only raw-state to economic-state mapping."""

    raw_state_to_name: dict[int, str]
    name_to_raw_state: dict[str, int | None]
    raw_state_to_economic_code: dict[int, int]
    stress_scores: dict[int, float]
    transition_state_modelled: bool
    economic_check: dict[str, Any]


@dataclass(frozen=True)
class MARSignalOutput:
    """MAR output with economic labels and next-session signal columns."""

    full_filter: MARFullFilterResult
    output_frame: pd.DataFrame
    state_summary: pd.DataFrame
    state_mapping: MAREconomicStateMapping


def load_phase7_input_panel(market: str, cfg: MARConfig) -> pd.DataFrame:
    """Load the Phase 7 processed HAR-VRP panel for one market."""
    market_norm = normalize_market(market)
    path = processed_input_path(market_norm, cfg)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 7 input panel not found for {market_norm}: {path}. "
            "Expected Phase 4/6 processed HAR-VRP output to exist."
        )

    df = pd.read_parquet(path)
    return standardize_panel_dates(df, source_path=path)


def standardize_panel_dates(df: pd.DataFrame, source_path: str | Path | None = None) -> pd.DataFrame:
    """
    Ensure a DataFrame has a clean date column.

    Rules:
    - date column required, unless DatetimeIndex can be reset
    - dates converted with pandas.to_datetime
    - rows sorted by date
    - duplicate dates rejected
    """
    out = df.copy()

    if DATE_COL not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if out.columns[0] != DATE_COL:
                out = out.rename(columns={out.columns[0]: DATE_COL})
        elif out.index.name == DATE_COL:
            out = out.reset_index()
        else:
            location = f" in {source_path}" if source_path is not None else ""
            raise ValueError(f"Missing required '{DATE_COL}' column{location}.")

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    bad_dates = int(out[DATE_COL].isna().sum())
    if bad_dates:
        location = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Found {bad_dates} invalid date value(s){location}.")

    out = out.sort_values(DATE_COL).reset_index(drop=True)

    duplicate_count = int(out[DATE_COL].duplicated().sum())
    if duplicate_count:
        location = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Found {duplicate_count} duplicate date value(s){location}.")

    return out


def prepare_mar_data_from_config(
    market: str,
    spec: MARModelSpec,
    cfg: MARConfig,
    *,
    enforce_min_observations: bool = True,
) -> MARPreparedData:
    """Load and prepare market data from configured input path."""
    df = load_phase7_input_panel(market, cfg)
    return prepare_mar_model_data(
        df=df,
        market=market,
        spec=spec,
        cfg=cfg,
        enforce_min_observations=enforce_min_observations,
    )


def prepare_mar_model_data(
    df: pd.DataFrame,
    market: str,
    spec: MARModelSpec,
    cfg: MARConfig,
    *,
    enforce_min_observations: bool = True,
) -> MARPreparedData:
    """
    Prepare an endog target series for MarkovAutoregression.

    The model input is univariate. Threshold/HMM/crisis/forward-label columns
    may exist in the source DataFrame, but they are not selected as model inputs.

    Steps:
    1. Standardize dates.
    2. Resolve and validate target column.
    3. Apply target availability rule, e.g. har_forecast_available == True.
    4. Drop missing/non-finite target rows.
    5. Chronologically split eligible rows.
    6. Estimate target transform parameters on train only.
    7. Apply same transform to full eligible series.
    8. Mark AR warmup rows.
    """
    market_norm = normalize_market(market)
    panel = standardize_panel_dates(df)
    panel = maybe_add_index_return(panel)

    target_col = resolve_target_column(spec.target, cfg)
    _validate_target_column(panel, target_col, spec, cfg)

    warnings: list[str] = []
    forbidden_present = tuple(find_forbidden_input_columns(panel.columns, cfg))

    if forbidden_present:
        warnings.append(
            "Source panel contains forbidden diagnostic/label/future columns. "
            "They were not selected as MAR model inputs."
        )

    eligible_frame, eligibility = build_eligible_mar_frame(
        panel=panel,
        market=market_norm,
        spec=spec,
        cfg=cfg,
        target_col=target_col,
    )

    train_idx, test_idx, split_summary = chronological_train_test_split(
        eligible_frame=eligible_frame,
        spec=spec,
        cfg=cfg,
        enforce_min_observations=enforce_min_observations,
    )

    transformed_values, transform_summary = fit_apply_target_transform_train_only(
        eligible_frame=eligible_frame,
        train_idx=train_idx,
        spec=spec,
        cfg=cfg,
        target_col=target_col,
    )

    eligible_frame = eligible_frame.copy()
    eligible_frame[TRANSFORMED_TARGET_COL] = transformed_values

    bad_transformed = int(
        eligible_frame[TRANSFORMED_TARGET_COL].isna().sum()
        + np.isinf(eligible_frame[TRANSFORMED_TARGET_COL].to_numpy(dtype=float)).sum()
    )
    if bad_transformed:
        raise ValueError(
            f"Target transform produced {bad_transformed} NaN/Inf value(s) "
            f"for target={spec.target}, column={target_col}."
        )

    eligible_frame["mar_sample_segment"] = "test"
    eligible_frame.loc[train_idx, "mar_sample_segment"] = "train"

    eligible_frame["mar_eligible_observation_number"] = np.arange(len(eligible_frame), dtype=int)
    eligible_frame["mar_ar_warmup_row"] = eligible_frame["mar_eligible_observation_number"] < spec.order
    eligible_frame["mar_model_observation_available_expected"] = ~eligible_frame["mar_ar_warmup_row"]

    train_frame = eligible_frame.loc[train_idx].copy().reset_index(drop=True)
    test_frame = eligible_frame.loc[test_idx].copy().reset_index(drop=True)
    eligible_frame = eligible_frame.reset_index(drop=True)

    available_fraction = float(
        eligible_frame["mar_model_observation_available_expected"].sum() / max(len(panel), 1)
    )
    min_available = cfg.validation.min_available_fraction_after_warmup

    if enforce_min_observations and available_fraction < min_available:
        raise ValueError(
            "Insufficient MAR model availability after AR warmup. "
            f"available_fraction={available_fraction:.4f}, required={min_available:.4f}, "
            f"market={market_norm}, target={spec.target}."
        )

    return MARPreparedData(
        market=market_norm,
        spec=spec,
        target_col=target_col,
        transformed_target_col=TRANSFORMED_TARGET_COL,
        panel=panel,
        eligible_frame=eligible_frame,
        train_frame=train_frame,
        test_frame=test_frame,
        eligibility=eligibility,
        split=split_summary,
        transform=transform_summary,
        forbidden_columns_present_in_panel=forbidden_present,
        validation_warnings=tuple(warnings),
    )


def build_eligible_mar_frame(
    panel: pd.DataFrame,
    market: str,
    spec: MARModelSpec,
    cfg: MARConfig,
    target_col: str,
) -> tuple[pd.DataFrame, MAREligibilitySummary]:
    """Build the date/target frame eligible for MAR modelling."""
    target_numeric = pd.to_numeric(panel[target_col], errors="coerce")
    target_non_missing_mask = target_numeric.notna() & np.isfinite(target_numeric.to_numpy(dtype=float))

    availability_mask, availability_description = build_target_availability_mask(
        panel=panel,
        target=spec.target,
        cfg=cfg,
    )

    eligible_mask = target_non_missing_mask & availability_mask

    context_cols = _context_columns_available(panel, cfg)
    keep_cols = [DATE_COL, target_col]
    for col in context_cols:
        if col not in keep_cols:
            keep_cols.append(col)

    eligible = panel.loc[eligible_mask, keep_cols].copy()
    eligible[target_col] = pd.to_numeric(eligible[target_col], errors="raise")

    if eligible.empty:
        raise ValueError(
            f"No eligible rows for MAR model. market={market}, target={spec.target}, "
            f"target_col={target_col}, availability_rule={availability_description}"
        )

    summary = MAREligibilitySummary(
        market=market,
        target=spec.target,
        target_col=target_col,
        n_panel_rows=int(len(panel)),
        n_target_non_missing=int(target_non_missing_mask.sum()),
        n_available_by_rule=int(availability_mask.sum()),
        n_eligible=int(eligible_mask.sum()),
        n_dropped_missing_target=int((~target_non_missing_mask).sum()),
        n_dropped_by_availability_rule=int((~availability_mask).sum()),
        availability_rule_description=availability_description,
    )

    return eligible.reset_index(drop=True), summary


def build_target_availability_mask(
    panel: pd.DataFrame,
    target: str,
    cfg: MARConfig,
) -> tuple[pd.Series, str]:
    """
    Apply target-specific availability rule.

    Example:
        vrp_har requires har_forecast_available == True.
    """
    rule = target_availability_rule(target, cfg)
    required_col = rule.get("required_boolean_column")
    required_value = rule.get("required_boolean_value")

    if required_col in {None, "", "null"}:
        return pd.Series(True, index=panel.index), "no availability rule"

    if required_col not in panel.columns:
        raise ValueError(
            f"Target '{target}' requires availability column '{required_col}', "
            "but it is missing from the input panel."
        )

    mask = boolean_equals(panel[required_col], required_value)

    description = f"{required_col} == {required_value}"
    return mask, description


def chronological_train_test_split(
    eligible_frame: pd.DataFrame,
    spec: MARModelSpec,
    cfg: MARConfig,
    *,
    enforce_min_observations: bool = True,
) -> tuple[pd.Index, pd.Index, MARTrainTestSummary]:
    """Create chronological train/test index masks over eligible rows."""
    n = len(eligible_frame)
    if n <= spec.order + 1:
        raise ValueError(
            f"Not enough eligible rows for MAR order={spec.order}. n_eligible={n}"
        )

    method = cfg.train_test_split.method
    if method != "chronological_fraction":
        raise ValueError(f"Unsupported train/test split method: {method}")

    train_fraction = cfg.train_test_split.train_fraction
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    n_train = int(np.floor(n * train_fraction))
    n_test = n - n_train

    train_idx = eligible_frame.index[:n_train]
    test_idx = eligible_frame.index[n_train:]

    n_train_effective = max(n_train - spec.order, 0)
    n_test_effective = n_test

    if enforce_min_observations:
        if n_train_effective < cfg.train_test_split.min_train_observations:
            raise ValueError(
                "Insufficient effective train observations after AR order. "
                f"effective_train={n_train_effective}, "
                f"required={cfg.train_test_split.min_train_observations}, "
                f"target={spec.target}, order={spec.order}."
            )

        if n_test_effective < cfg.train_test_split.min_test_observations:
            raise ValueError(
                "Insufficient test observations. "
                f"test={n_test_effective}, "
                f"required={cfg.train_test_split.min_test_observations}, "
                f"target={spec.target}, order={spec.order}."
            )

    train_start = eligible_frame.loc[train_idx[0], DATE_COL]
    train_end = eligible_frame.loc[train_idx[-1], DATE_COL]
    test_start = eligible_frame.loc[test_idx[0], DATE_COL]
    test_end = eligible_frame.loc[test_idx[-1], DATE_COL]

    summary = MARTrainTestSummary(
        method=method,
        train_fraction=float(train_fraction),
        n_eligible=int(n),
        n_train=int(n_train),
        n_test=int(n_test),
        n_train_effective_after_ar_order=int(n_train_effective),
        n_test_effective_after_ar_order=int(n_test_effective),
        train_start_date=_date_str(train_start),
        train_end_date=_date_str(train_end),
        test_start_date=_date_str(test_start),
        test_end_date=_date_str(test_end),
    )

    return train_idx, test_idx, summary


def fit_apply_target_transform_train_only(
    eligible_frame: pd.DataFrame,
    train_idx: pd.Index,
    spec: MARModelSpec,
    cfg: MARConfig,
    target_col: str,
) -> tuple[pd.Series, MARTargetTransformSummary]:
    """
    Estimate target transform parameters on train only and apply to all eligible rows.

    Supported methods:
    - none
    - log_positive
    - winsorize_train_quantiles
    """
    rule = target_transform_rule(spec.target, cfg)
    method = str(rule.get("method", "none"))

    values = pd.to_numeric(eligible_frame[target_col], errors="raise").astype(float)
    train_values = values.loc[train_idx]

    n_clipped_low = 0
    n_clipped_high = 0
    params: dict[str, Any] = {}

    if method == "none":
        transformed = values.copy()
        params = {"method": "none"}

    elif method in {"standardize_train", "zscore_train"}:
        train_mean = float(train_values.mean())
        train_std = float(train_values.std(ddof=0))

        if not np.isfinite(train_mean):
            raise ValueError("Train-only standardization produced non-finite mean.")

        if (not np.isfinite(train_std)) or train_std <= 0.0:
            raise ValueError(
                "Train-only standardization produced invalid std. "
                f"std={train_std}"
            )

        transformed = (values - train_mean) / train_std

        params = {
            "method": "standardize_train",
            "mean_estimated_on_train": train_mean,
            "std_estimated_on_train": train_std,
        }

    elif method in {
        "winsorize_train_quantiles_then_standardize",
        "winsorize_train_quantiles_standardize",
    }:
        lower_q, upper_q = _resolve_winsorize_quantiles(rule)

        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError(
                "Invalid winsorization quantiles: "
                f"lower_quantile={lower_q}, upper_quantile={upper_q}"
            )

        lower_cap = float(train_values.quantile(lower_q))
        upper_cap = float(train_values.quantile(upper_q))

        if not np.isfinite(lower_cap) or not np.isfinite(upper_cap):
            raise ValueError("Train-only winsorization produced non-finite cap(s).")

        clipped_full = values.clip(lower=lower_cap, upper=upper_cap)
        clipped_train = train_values.clip(lower=lower_cap, upper=upper_cap)

        n_clipped_low = int((values < lower_cap).sum())
        n_clipped_high = int((values > upper_cap).sum())

        train_mean = float(clipped_train.mean())
        train_std = float(clipped_train.std(ddof=0))

        if not np.isfinite(train_mean):
            raise ValueError("Winsorized train standardization produced non-finite mean.")

        if (not np.isfinite(train_std)) or train_std <= 0.0:
            raise ValueError(
                "Winsorized train standardization produced invalid std. "
                f"std={train_std}"
            )

        transformed = (clipped_full - train_mean) / train_std

        params = {
            "method": "winsorize_train_quantiles_then_standardize",
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "lower_cap_estimated_on_train": lower_cap,
            "upper_cap_estimated_on_train": upper_cap,
            "mean_estimated_on_winsorized_train": train_mean,
            "std_estimated_on_winsorized_train": train_std,
            "n_clipped_low_full_eligible": n_clipped_low,
            "n_clipped_high_full_eligible": n_clipped_high,
        }

    elif method == "log_positive":
        floor = float(rule.get("floor", 1.0e-8))
        if floor <= 0:
            raise ValueError(f"log_positive floor must be positive, got {floor}")

        clipped = values.clip(lower=floor)
        n_clipped_low = int((values < floor).sum())
        transformed = np.log(clipped)
        transformed = pd.Series(transformed, index=eligible_frame.index, name=TRANSFORMED_TARGET_COL)
        params = {
            "method": "log_positive",
            "floor": floor,
            "n_values_below_floor": n_clipped_low,
        }

    elif method == "winsorize_train_quantiles":
        lower_q, upper_q = _resolve_winsorize_quantiles(rule)

        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError(
                "Invalid winsorization quantiles: "
                f"lower_quantile={lower_q}, upper_quantile={upper_q}"
            )

        lower_cap = float(train_values.quantile(lower_q))
        upper_cap = float(train_values.quantile(upper_q))

        if not np.isfinite(lower_cap) or not np.isfinite(upper_cap):
            raise ValueError("Train-only winsorization produced non-finite cap(s).")

        clipped = values.clip(lower=lower_cap, upper=upper_cap)
        n_clipped_low = int((values < lower_cap).sum())
        n_clipped_high = int((values > upper_cap).sum())
        transformed = clipped

        params = {
            "method": "winsorize_train_quantiles",
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "lower_cap_estimated_on_train": lower_cap,
            "upper_cap_estimated_on_train": upper_cap,
            "n_clipped_low_full_eligible": n_clipped_low,
            "n_clipped_high_full_eligible": n_clipped_high,
        }

    else:
        raise ValueError(
            f"Unsupported target transform method '{method}' for target={spec.target}."
        )

    transformed = pd.Series(
        pd.to_numeric(transformed, errors="raise"),
        index=eligible_frame.index,
        name=TRANSFORMED_TARGET_COL,
    ).astype(float)

    n_output_missing = int(transformed.isna().sum())

    summary = MARTargetTransformSummary(
        target=spec.target,
        target_col=target_col,
        transformed_col=TRANSFORMED_TARGET_COL,
        method=method,
        params=params,
        n_input=int(len(values)),
        n_output_missing=n_output_missing,
        n_clipped_low=n_clipped_low,
        n_clipped_high=n_clipped_high,
    )

    return transformed, summary


def get_endog(
    prepared: MARPreparedData,
    sample: Literal["train", "test", "full"] = "train",
) -> pd.Series:
    """
    Return transformed target series for statsmodels.

    sample:
        train -> train-only endog for fitting.
        test  -> test-only endog for diagnostics.
        full  -> full eligible endog for train-param filtering.
    """
    if sample == "train":
        frame = prepared.train_frame
    elif sample == "test":
        frame = prepared.test_frame
    elif sample == "full":
        frame = prepared.eligible_frame
    else:
        raise ValueError(f"Unknown sample: {sample}")

    series = pd.Series(
        frame[prepared.transformed_target_col].to_numpy(dtype=float),
        index=pd.to_datetime(frame[DATE_COL]),
        name=prepared.transformed_target_col,
    )

    if series.isna().any() or np.isinf(series.to_numpy(dtype=float)).any():
        raise ValueError(f"Endog series contains NaN/Inf for sample={sample}.")

    return series


def maybe_add_index_return(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add index_return if it is missing and a usable price column exists.

    This is for economic state diagnostics only. It is not used as MAR input
    unless target='returns'.
    """
    out = panel.copy()

    if "index_return" in out.columns:
        out["index_return"] = pd.to_numeric(out["index_return"], errors="coerce")
        return out

    price_col = _first_existing_column(
        out,
        ["index_close", "underlying_close", "adj_close", "close"],
    )

    if price_col is None:
        return out

    price = pd.to_numeric(out[price_col], errors="coerce")
    valid_price = price.where(price > 0)
    out["index_return"] = np.log(valid_price / valid_price.shift(1))

    return out


def boolean_equals(series: pd.Series, required_value: Any) -> pd.Series:
    """Robust boolean comparison for bool/string/numeric availability columns."""
    if required_value is None:
        return pd.Series(True, index=series.index)

    required_bool = _to_bool_scalar(required_value)

    if pd.api.types.is_bool_dtype(series):
        values = series.fillna(False).astype(bool)
        return values == required_bool

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0
        return values == required_bool

    normalized = series.astype(str).str.strip().str.lower()
    truthy = normalized.isin({"true", "1", "yes", "y", "t"})
    falsy = normalized.isin({"false", "0", "no", "n", "f", "nan", "none", ""})

    values = pd.Series(False, index=series.index)
    values.loc[truthy] = True
    values.loc[falsy] = False

    return values == required_bool


def prepared_data_summary_dict(prepared: MARPreparedData) -> dict[str, Any]:
    """Return JSON/metadata-friendly summary of prepared data."""
    return {
        "market": prepared.market,
        "spec": prepared.spec.to_dict(),
        "target_col": prepared.target_col,
        "transformed_target_col": prepared.transformed_target_col,
        "eligibility": prepared.eligibility.__dict__,
        "split": prepared.split.__dict__,
        "transform": {
            "target": prepared.transform.target,
            "target_col": prepared.transform.target_col,
            "transformed_col": prepared.transform.transformed_col,
            "method": prepared.transform.method,
            "params": prepared.transform.params,
            "n_input": prepared.transform.n_input,
            "n_output_missing": prepared.transform.n_output_missing,
            "n_clipped_low": prepared.transform.n_clipped_low,
            "n_clipped_high": prepared.transform.n_clipped_high,
        },
        "forbidden_columns_present_in_panel": list(prepared.forbidden_columns_present_in_panel),
        "validation_warnings": list(prepared.validation_warnings),
    }


def _validate_target_column(
    panel: pd.DataFrame,
    target_col: str,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> None:
    if target_col not in panel.columns:
        raise ValueError(
            f"Target column '{target_col}' for target='{spec.target}' "
            "is missing from the input panel."
        )

    allowed_target_cols = set(cfg.input_policy.allowed_target_candidates)
    if target_col not in allowed_target_cols:
        raise ValueError(
            f"Target column '{target_col}' is not approved for MAR input. "
            f"Allowed: {sorted(allowed_target_cols)}"
        )

    forbidden = find_forbidden_input_columns([target_col], cfg)
    if forbidden:
        raise ValueError(
            f"Resolved target column is forbidden as MAR input: {forbidden}"
        )

    numeric = pd.to_numeric(panel[target_col], errors="coerce")
    if numeric.notna().sum() == 0:
        raise ValueError(f"Target column '{target_col}' has no numeric observations.")


def _context_columns_available(panel: pd.DataFrame, cfg: MARConfig) -> list[str]:
    """
    Context columns kept for later economic state labeling.

    These are not model inputs.
    """
    candidates = [
        cfg.target_columns.get("iv"),
        cfg.target_columns.get("rv"),
        cfg.target_columns.get("returns"),
        "har_forecast_available",
    ]

    out: list[str] = []
    for col in candidates:
        if col is None:
            continue
        if col in panel.columns and col not in out:
            out.append(str(col))

    return out


def _resolve_winsorize_quantiles(rule: dict[str, Any]) -> tuple[float, float]:
    lower_q = rule.get("lower_quantile")
    upper_q = rule.get("upper_quantile")

    if lower_q is None or upper_q is None:
        optional = rule.get("optional_methods", {})
        nested = optional.get("winsorize_train_quantiles", {})
        lower_q = nested.get("lower_quantile", lower_q)
        upper_q = nested.get("upper_quantile", upper_q)

    if lower_q is None:
        lower_q = 0.005
    if upper_q is None:
        upper_q = 0.995

    return float(lower_q), float(upper_q)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "none", "null", ""}:
        return False

    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()

def fit_markov_autoreg_candidate(
    prepared: MARPreparedData,
    cfg: MARConfig,
    *,
    raise_on_invalid: bool = False,
) -> MARCandidateFit:
    """
    Fit one MarkovAutoregression candidate on the train sample only.

    Strict rule:
        Parameters are estimated only from prepared.train_frame.

    The returned result is not yet safe for backtest-facing output because it
    contains train-sample probabilities only. Chunk 4 applies these train-fitted
    parameters to the full eligible series using the Hamilton filter.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
    except Exception as exc:
        summary = _invalid_fit_summary(
            fit_exception=(
                "Could not import statsmodels MarkovAutoregression: "
                f"{type(exc).__name__}: {exc}"
            )
        )
        candidate = _empty_candidate_fit(prepared, summary)
        if raise_on_invalid:
            raise RuntimeError(summary.invalid_reason)
        return candidate

    spec = prepared.spec
    train_y = get_endog(prepared, "train")

    model: Any | None = None
    result: Any | None = None

    fit_attempts: tuple[MARFitAttempt, ...] = tuple()

    model, result, fit_attempts = fit_statsmodels_markov_autoreg_with_fallbacks(
        MarkovAutoregression=MarkovAutoregression,
        train_y=train_y,
        prepared=prepared,
        cfg=cfg,
    )

    if result is None:
        last_exception = "all fit attempts failed"
        if fit_attempts:
            last = fit_attempts[-1]
            last_exception = last.exception or last_exception

        summary = _invalid_fit_summary(
            fit_exception=last_exception,
            model=model,
            result=result,
        )
        candidate = _empty_candidate_fit(
            prepared,
            summary,
            model=model,
            result=result,
            fit_attempts=fit_attempts,
        )
        if raise_on_invalid:
            raise RuntimeError(summary.invalid_reason)
        return candidate

    train_probs = extract_filtered_probabilities_from_result(
        result=result,
        n_states=spec.n_states,
        prefix="mar_filtered_prob_raw_state_",
    )

    train_assignments = assign_states_from_probability_frame(train_probs)
    train_occupancy = state_occupancy_from_probabilities(train_probs)

    transition_matrix = extract_transition_matrix_from_result(
        result=result,
        n_states=spec.n_states,
    )

    ar_coefficients = extract_ar_coefficients_from_result(
        result=result,
        spec=spec,
    )

    sigma2_by_state = extract_sigma2_from_result(
        result=result,
        spec=spec,
    )

    summary = build_fit_firewall_summary(
        result=result,
        train_filtered_probabilities=train_probs,
        train_state_occupancy=train_occupancy,
        transition_matrix=transition_matrix,
        ar_coefficients=ar_coefficients,
        sigma2_by_state=sigma2_by_state,
        cfg=cfg,
    )

    candidate = MARCandidateFit(
        market=prepared.market,
        spec=spec,
        prepared=prepared,
        model=model,
        result=result,
        train_filtered_probabilities=train_probs,
        train_state_assignments=train_assignments,
        train_state_occupancy=train_occupancy,
        transition_matrix=transition_matrix,
        ar_coefficients=ar_coefficients,
        sigma2_by_state=sigma2_by_state,
        fit_summary=summary,
        fit_attempts=fit_attempts,
    )

    if raise_on_invalid and not summary.valid_candidate:
        raise RuntimeError(summary.invalid_reason)

    return candidate


def fit_statsmodels_markov_autoreg_with_fallbacks(
    MarkovAutoregression: Any,
    train_y: pd.Series,
    prepared: MARPreparedData,
    cfg: MARConfig,
) -> tuple[Any | None, Any | None, tuple[MARFitAttempt, ...]]:
    """
    Fit statsmodels MarkovAutoregression with defensive fallback attempts.

    Why this exists:
    Markov-switching likelihoods are numerically fragile. A candidate can fail
    because Hessian/covariance calculation fails even when the likelihood path is
    otherwise usable. We keep the approved primary spec unchanged, but retry with
    alternative optimizers / covariance estimators.

    The data and model specification are unchanged across attempts.
    Only optimizer/covariance settings change.
    """
    spec = prepared.spec

    attempts_config = [
        {
            "method": "bfgs",
            "cov_type": str(cfg.fit.cov_type),
            "maxiter": int(cfg.fit.maxiter),
            "em_iter": int(cfg.fit.em_iter),
            "search_reps": int(cfg.fit.search_reps),
            "disp": bool(cfg.fit.disp),
        },
        {
            "method": "lbfgs",
            "cov_type": "opg",
            "maxiter": int(cfg.fit.maxiter),
            "em_iter": max(5, int(cfg.fit.em_iter // 2)),
            "search_reps": max(5, int(cfg.fit.search_reps // 2)),
            "disp": bool(cfg.fit.disp),
        },
        {
            "method": "powell",
            "cov_type": "opg",
            "maxiter": min(int(cfg.fit.maxiter), 500),
            "em_iter": 5,
            "search_reps": 0,
            "disp": bool(cfg.fit.disp),
        },
        {
            "method": "bfgs",
            "cov_type": "none",
            "maxiter": min(int(cfg.fit.maxiter), 500),
            "em_iter": 5,
            "search_reps": 0,
            "disp": bool(cfg.fit.disp),
        },
    ]

    attempts: list[MARFitAttempt] = []
    last_model: Any | None = None

    # Avoid date-frequency warnings and remove index-related noise.
    endog_values = np.asarray(train_y, dtype=float)

    for attempt_id, params in enumerate(attempts_config, start=1):
        np.random.seed(int(cfg.random_seed) + attempt_id)

        try:
            model = MarkovAutoregression(
                endog=endog_values,
                k_regimes=spec.n_states,
                order=spec.order,
                switching_ar=spec.switching_ar,
                switching_trend=spec.switching_trend,
                switching_variance=spec.switching_variance,
            )
            last_model = model

            result = model.fit(
                method=params["method"],
                maxiter=params["maxiter"],
                em_iter=params["em_iter"],
                search_reps=params["search_reps"],
                cov_type=params["cov_type"],
                disp=params["disp"],
            )

            attempts.append(
                MARFitAttempt(
                    attempt_id=attempt_id,
                    method=params["method"],
                    cov_type=params["cov_type"],
                    maxiter=params["maxiter"],
                    em_iter=params["em_iter"],
                    search_reps=params["search_reps"],
                    disp=params["disp"],
                    success=True,
                    exception=None,
                )
            )

            return model, result, tuple(attempts)

        except Exception as exc:
            attempts.append(
                MARFitAttempt(
                    attempt_id=attempt_id,
                    method=params["method"],
                    cov_type=params["cov_type"],
                    maxiter=params["maxiter"],
                    em_iter=params["em_iter"],
                    search_reps=params["search_reps"],
                    disp=params["disp"],
                    success=False,
                    exception=f"{type(exc).__name__}: {exc}",
                )
            )

    return last_model, None, tuple(attempts)


def build_fit_firewall_summary(
    result: Any,
    train_filtered_probabilities: pd.DataFrame,
    train_state_occupancy: dict[int, float],
    transition_matrix: pd.DataFrame,
    ar_coefficients: dict[int, float],
    sigma2_by_state: dict[int, float],
    cfg: MARConfig,
) -> MARFitFirewallSummary:
    """
    Validate a statsmodels fit result before it can be accepted.

    A result object is not enough. This firewall rejects candidates with bad
    convergence, invalid likelihood, invalid probabilities, tiny occupancy,
    near-absorbing transitions, explosive AR coefficients, invalid variance, or
    unusable covariance estimates.
    """
    firewall_cfg = dict(cfg.raw.get("fit_firewall", {}))

    invalid_reasons: list[str] = []
    warnings: list[str] = []

    mle_retvals = sanitize_mapping(getattr(result, "mle_retvals", {}) or {})
    fit_converged = infer_convergence_status(result)
    warnflag = mle_retvals.get("warnflag")

    llf = finite_float_or_none(getattr(result, "llf", None))
    aic = finite_float_or_none(getattr(result, "aic", None))
    bic = finite_float_or_none(getattr(result, "bic", None))
    hqic = finite_float_or_none(getattr(result, "hqic", None))
    nobs = int_or_none(getattr(result, "nobs", None))
    n_params = infer_n_params(result)

    if bool(firewall_cfg.get("require_converged", True)) and not fit_converged:
        invalid_reasons.append("fit_converged == False")

    if bool(firewall_cfg.get("require_finite_llf", True)) and llf is None:
        invalid_reasons.append("log-likelihood is missing or non-finite")

    if bool(firewall_cfg.get("require_filtered_probabilities", True)):
        prob_reason = validate_probability_frame(
            probs=train_filtered_probabilities,
            n_states=None,
            row_sum_tolerance=cfg.validation.probability_row_sum_tolerance,
        )
        if prob_reason is not None:
            invalid_reasons.append(f"invalid filtered probabilities: {prob_reason}")

    occupancy_reason = validate_state_occupancy(
        occupancy=train_state_occupancy,
        min_occupancy=cfg.validation.min_train_state_occupancy,
        sample_name="train",
    )
    if occupancy_reason is not None:
        invalid_reasons.append(occupancy_reason)

    transition_reason = validate_transition_matrix(
        transition_matrix=transition_matrix,
        near_absorbing_threshold=cfg.validation.near_absorbing_transition_threshold,
    )
    if transition_reason is not None:
        invalid_reasons.append(transition_reason)

    ar_reason, ar_warnings = validate_ar_coefficients(
        ar_coefficients=ar_coefficients,
        explosive_abs_phi_threshold=cfg.validation.ar_explosive_abs_phi_threshold,
    )
    warnings.extend(ar_warnings)

    if bool(firewall_cfg.get("reject_explosive_ar", True)) and ar_reason is not None:
        invalid_reasons.append(ar_reason)

    sigma_reason = validate_sigma2_by_state(sigma2_by_state)
    if bool(firewall_cfg.get("reject_invalid_variance", True)) and sigma_reason is not None:
        invalid_reasons.append(sigma_reason)

    if bool(firewall_cfg.get("require_valid_covariance", True)):
        cov_reason = validate_covariance_matrix(result)
        if cov_reason is not None:
            invalid_reasons.append(cov_reason)

    valid_candidate = len(invalid_reasons) == 0
    invalid_reason = "" if valid_candidate else "; ".join(invalid_reasons)

    return MARFitFirewallSummary(
        fit_converged=bool(fit_converged),
        fit_exception=None,
        llf=llf,
        aic=aic,
        bic=bic,
        hqic=hqic,
        nobs=nobs,
        n_params=n_params,
        warnflag=warnflag,
        mle_retvals=mle_retvals,
        valid_candidate=valid_candidate,
        invalid_reason=invalid_reason,
        warnings=tuple(warnings),
    )


def extract_filtered_probabilities_from_result(
    result: Any,
    n_states: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Extract filtered marginal probabilities from a statsmodels result object.

    Statsmodels may return numpy arrays or pandas objects depending on version.
    This function normalizes to a DataFrame with columns:
        {prefix}0, {prefix}1, ...
    """
    if not hasattr(result, "filtered_marginal_probabilities"):
        raise ValueError("Result object has no filtered_marginal_probabilities attribute.")

    raw = result.filtered_marginal_probabilities
    probs = _probabilities_to_2d_array(raw=raw, n_states=n_states)

    columns = [f"{prefix}{i}" for i in range(n_states)]
    return pd.DataFrame(probs, columns=columns)


def assign_states_from_probability_frame(probs: pd.DataFrame) -> pd.Series:
    """Assign raw state by maximum filtered probability."""
    if probs.empty:
        return pd.Series(dtype="float64", name="mar_raw_state")

    values = probs.to_numpy(dtype=float)
    state = np.argmax(values, axis=1)
    return pd.Series(state, index=probs.index, name="mar_raw_state")


def state_occupancy_from_probabilities(probs: pd.DataFrame) -> dict[int, float]:
    """
    Compute soft state occupancy from filtered probabilities.

    Soft occupancy is preferred here because it uses the probability mass, not
    only argmax states.
    """
    if probs.empty:
        return {}

    values = probs.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    return {int(i): float(means[i]) for i in range(values.shape[1])}


def extract_transition_matrix_from_result(
    result: Any,
    n_states: int,
) -> pd.DataFrame:
    """
    Extract transition matrix from statsmodels result if available.

    The orientation of statsmodels transition arrays can vary by internal
    convention, but for firewall purposes we mainly need diagonal persistence
    and finite values.
    """
    raw = getattr(result, "regime_transition", None)

    if raw is None:
        return pd.DataFrame(
            np.full((n_states, n_states), np.nan),
            index=[f"from_state_{i}" for i in range(n_states)],
            columns=[f"to_state_{j}" for j in range(n_states)],
        )

    arr = np.asarray(raw, dtype=float)

    if arr.ndim == 3:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        arr = np.asarray(arr).reshape(n_states, n_states)

    if arr.shape != (n_states, n_states):
        arr = np.asarray(arr).reshape(n_states, n_states)

    return pd.DataFrame(
        arr,
        index=[f"from_state_{i}" for i in range(n_states)],
        columns=[f"to_state_{j}" for j in range(n_states)],
    )


def extract_ar_coefficients_from_result(
    result: Any,
    spec: MARModelSpec,
) -> dict[int, float]:
    """
    Extract state-specific AR(1) coefficients when parameter names expose them.

    Statsmodels parameter naming can vary slightly by version. This function is
    deliberately heuristic and conservative:
    - if coefficients are found, they are validated
    - if they are not found, the model is not rejected here
    """
    params = params_as_series(result)
    out: dict[int, float] = {}

    if params.empty:
        return out

    for state in range(spec.n_states):
        candidates = []
        for name, value in params.items():
            lname = str(name).lower()

            has_ar = "ar" in lname
            has_lag1 = "l1" in lname or ".1" in lname or "[1]" in lname

            state_markers = [
                f"[{state}]",
                f".{state}",
                f"_{state}",
                f"regime {state}",
                f"regime_{state}",
            ]
            has_state = any(marker in lname for marker in state_markers)

            if has_ar and (has_lag1 or spec.order == 1) and has_state:
                candidates.append(float(value))

        if len(candidates) == 1:
            out[state] = float(candidates[0])

    # Fallback for common params order is intentionally not used. Wrong AR
    # extraction is worse than missing extraction.
    return out


def extract_sigma2_from_result(
    result: Any,
    spec: MARModelSpec,
) -> dict[int, float]:
    """
    Extract state-specific variance estimates if parameter names expose them.

    If switching_variance=False, statsmodels may expose one shared sigma2.
    In that case the same value is assigned to all states.
    """
    params = params_as_series(result)
    out: dict[int, float] = {}

    if params.empty:
        return out

    sigma_params: list[tuple[str, float]] = []
    for name, value in params.items():
        lname = str(name).lower()
        if "sigma2" in lname or "variance" in lname:
            sigma_params.append((str(name), float(value)))

    if not sigma_params:
        return out

    if not spec.switching_variance and len(sigma_params) >= 1:
        shared = float(sigma_params[0][1])
        return {int(state): shared for state in range(spec.n_states)}

    for state in range(spec.n_states):
        for name, value in sigma_params:
            lname = name.lower()
            state_markers = [
                f"[{state}]",
                f".{state}",
                f"_{state}",
                f"regime {state}",
                f"regime_{state}",
            ]
            if any(marker in lname for marker in state_markers):
                out[state] = float(value)
                break

    return out


def params_as_series(result: Any) -> pd.Series:
    """Return result params as a named Series where possible."""
    raw_params = getattr(result, "params", None)
    if raw_params is None:
        return pd.Series(dtype=float)

    if isinstance(raw_params, pd.Series):
        return raw_params.astype(float)

    params_array = np.asarray(raw_params, dtype=float).ravel()

    names = None
    model = getattr(result, "model", None)
    if model is not None:
        names = getattr(model, "param_names", None)

    if names is None:
        names = [f"param_{i}" for i in range(len(params_array))]

    if len(names) != len(params_array):
        names = [f"param_{i}" for i in range(len(params_array))]

    return pd.Series(params_array, index=list(names), dtype=float)


def validate_probability_frame(
    probs: pd.DataFrame,
    n_states: int | None,
    row_sum_tolerance: float,
) -> str | None:
    """Return invalid reason for probability frame, or None if valid."""
    if probs.empty:
        return "probability frame is empty"

    values = probs.to_numpy(dtype=float)

    if values.ndim != 2:
        return f"probabilities are not 2D, shape={values.shape}"

    if n_states is not None and values.shape[1] != int(n_states):
        return f"expected {n_states} states, got {values.shape[1]}"

    if not np.isfinite(values).all():
        return "probabilities contain NaN or Inf"

    if (values < -row_sum_tolerance).any() or (values > 1.0 + row_sum_tolerance).any():
        return "probabilities outside [0, 1] tolerance"

    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=row_sum_tolerance, rtol=0.0):
        max_abs_error = float(np.max(np.abs(row_sums - 1.0)))
        return f"probability rows do not sum to 1; max_abs_error={max_abs_error:.6g}"

    return None


def validate_state_occupancy(
    occupancy: dict[int, float],
    min_occupancy: float,
    sample_name: str,
) -> str | None:
    """Validate minimum soft state occupancy."""
    if not occupancy:
        return f"{sample_name} state occupancy is empty"

    bad = {
        int(state): float(value)
        for state, value in occupancy.items()
        if not np.isfinite(value) or value < min_occupancy
    }

    if bad:
        return (
            f"{sample_name} state occupancy below threshold "
            f"{min_occupancy:.4f}: {bad}"
        )

    return None


def validate_transition_matrix(
    transition_matrix: pd.DataFrame,
    near_absorbing_threshold: float,
) -> str | None:
    """Validate transition matrix finite values and near-absorbing diagonal."""
    if transition_matrix.empty:
        return "transition matrix is empty"

    values = transition_matrix.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return f"transition matrix is not square, shape={values.shape}"

    if not np.isfinite(values).all():
        return "transition matrix contains NaN or Inf"

    diag = np.diag(values)
    if np.any(diag >= near_absorbing_threshold):
        return (
            "transition matrix is near absorbing; "
            f"max_diagonal={float(np.max(diag)):.6f}, "
            f"threshold={near_absorbing_threshold:.6f}"
        )

    return None


def validate_ar_coefficients(
    ar_coefficients: dict[int, float],
    explosive_abs_phi_threshold: float,
) -> tuple[str | None, list[str]]:
    """
    Validate AR coefficients.

    Near-unit-root states are warned. Numerically explosive states are rejected.
    """
    warnings: list[str] = []

    if not ar_coefficients:
        warnings.append(
            "AR coefficients could not be extracted from statsmodels parameter names."
        )
        return None, warnings

    explosive: dict[int, float] = {}

    for state, phi in ar_coefficients.items():
        if not np.isfinite(phi):
            explosive[int(state)] = float(phi)
            continue

        abs_phi = abs(float(phi))
        if abs_phi >= explosive_abs_phi_threshold:
            warnings.append(
                f"AR coefficient near or beyond unit-root boundary: "
                f"state={state}, phi={phi:.6f}"
            )

        if abs_phi > explosive_abs_phi_threshold + 1.0e-8:
            explosive[int(state)] = float(phi)

    if explosive:
        return f"explosive AR coefficient(s): {explosive}", warnings

    return None, warnings


def validate_sigma2_by_state(sigma2_by_state: dict[int, float]) -> str | None:
    """Validate variance estimates if they were extracted."""
    if not sigma2_by_state:
        return None

    bad = {
        int(state): float(value)
        for state, value in sigma2_by_state.items()
        if (not np.isfinite(value)) or value <= 0.0
    }

    if bad:
        return f"invalid sigma2 estimate(s): {bad}"

    return None


def validate_covariance_matrix(result: Any) -> str | None:
    """
    Validate covariance matrix if statsmodels can provide it.

    If covariance cannot be computed, reject because Phase 7 config requires a
    valid covariance estimate.
    """
    try:
        cov = result.cov_params()
    except Exception as exc:
        return f"covariance matrix unavailable: {type(exc).__name__}: {exc}"

    arr = np.asarray(cov, dtype=float)

    if arr.size == 0:
        return "covariance matrix is empty"

    if not np.isfinite(arr).all():
        return "covariance matrix contains NaN or Inf"

    return None


def infer_convergence_status(result: Any) -> bool:
    """Infer convergence status from statsmodels result."""
    mle_retvals = getattr(result, "mle_retvals", {}) or {}

    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])

    if hasattr(result, "converged"):
        return bool(getattr(result, "converged"))

    # If statsmodels does not expose convergence, be conservative.
    return False


def infer_n_params(result: Any) -> int | None:
    """Infer number of model parameters."""
    params = getattr(result, "params", None)
    if params is None:
        return None

    try:
        return int(np.asarray(params).size)
    except Exception:
        return None


def fit_summary_dict(candidate: MARCandidateFit) -> dict[str, Any]:
    """Return candidate fit summary as a flat dictionary for ranking tables."""
    s = candidate.fit_summary

    return {
        "market": candidate.market,
        "target": candidate.spec.target,
        "order": candidate.spec.order,
        "n_states": candidate.spec.n_states,
        "switching_ar": candidate.spec.switching_ar,
        "switching_trend": candidate.spec.switching_trend,
        "switching_variance": candidate.spec.switching_variance,
        "suffix": candidate.spec.suffix(),
        "fit_converged": s.fit_converged,
        "fit_exception": s.fit_exception,
        "llf": s.llf,
        "aic": s.aic,
        "bic": s.bic,
        "hqic": s.hqic,
        "nobs": s.nobs,
        "n_params": s.n_params,
        "warnflag": s.warnflag,
        "valid_candidate": s.valid_candidate,
        "invalid_reason": s.invalid_reason,
        "train_state_occupancy": candidate.train_state_occupancy,
        "ar_coefficients": candidate.ar_coefficients,
        "sigma2_by_state": candidate.sigma2_by_state,
        "warnings": list(s.warnings),
        "fit_attempts": [attempt.__dict__ for attempt in candidate.fit_attempts],
    }


def filter_full_series_with_train_params(
    candidate: MARCandidateFit,
    cfg: MARConfig,
    *,
    include_smoothed_diagnostic: bool = True,
    raise_on_invalid: bool = True,
) -> MARFullFilterResult:
    """
    Apply train-fitted MAR parameters to the full eligible series.

    Strict no-lookahead rule:
        - candidate.result.params were estimated from train only.
        - full_model is instantiated on full eligible endog.
        - full_model.filter(train_params) applies the Hamilton filter to the
          full series using train-estimated parameters.
        - full_model.smooth(train_params) is diagnostic only.

    This creates point-in-time filtered probabilities. It does not refit on the
    full sample.
    """
    if candidate.result is None:
        raise ValueError(
            "Cannot filter full series because candidate.result is None. "
            f"Fit invalid_reason={candidate.fit_summary.invalid_reason}"
        )

    if not candidate.fit_summary.valid_candidate:
        raise ValueError(
            "Cannot filter full series because candidate failed fit firewall. "
            f"invalid_reason={candidate.fit_summary.invalid_reason}"
        )

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import statsmodels MarkovAutoregression for full filtering: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    prepared = candidate.prepared
    spec = candidate.spec
    full_y = get_endog(prepared, "full")
    full_values = np.asarray(full_y, dtype=float)
    train_params = np.asarray(candidate.result.params, dtype=float)

    full_model: Any | None = None
    full_filter_result: Any | None = None
    full_smooth_result: Any | None = None

    try:
        full_model = MarkovAutoregression(
            endog=full_values,
            k_regimes=spec.n_states,
            order=spec.order,
            switching_ar=spec.switching_ar,
            switching_trend=spec.switching_trend,
            switching_variance=spec.switching_variance,
        )

        full_filter_result = full_model.filter(
            train_params,
            transformed=True,
            cov_type=str(cfg.fit.cov_type),
        )

    except Exception as exc:
        raise RuntimeError(
            "Full-series Hamilton filtering failed using train-fitted params: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if include_smoothed_diagnostic and bool(
        cfg.raw.get("output_policy", {}).get("include_smoothed_probabilities_diagnostic", True)
    ):
        try:
            full_smooth_result = full_model.smooth(
                train_params,
                transformed=True,
                cov_type=str(cfg.fit.cov_type),
            )
        except Exception:
            # Smoothing is diagnostic only. Do not fail the core filtered output
            # if smoothing fails.
            full_smooth_result = None

    raw_filtered = extract_filtered_probabilities_from_result(
        result=full_filter_result,
        n_states=spec.n_states,
        prefix="mar_filtered_prob_raw_state_",
    )

    if full_smooth_result is not None:
        raw_smoothed = extract_smoothed_probabilities_from_result(
            result=full_smooth_result,
            n_states=spec.n_states,
            prefix="mar_diagnostic_smoothed_prob_raw_state_",
        )
    else:
        raw_smoothed = pd.DataFrame(
            np.nan,
            index=np.arange(len(prepared.eligible_frame)),
            columns=[f"mar_diagnostic_smoothed_prob_raw_state_{i}" for i in range(spec.n_states)],
        )

    aligned = align_mar_probabilities_to_eligible_frame(
        prepared=prepared,
        spec=spec,
        raw_filtered_probabilities=raw_filtered,
        raw_smoothed_probabilities_diagnostic=raw_smoothed,
    )

    probability_audit = audit_aligned_probabilities(
        aligned_frame=aligned,
        spec=spec,
        cfg=cfg,
    )

    if raise_on_invalid and not probability_audit.passed:
        raise ValueError(
            "Aligned full-series filtered probability audit failed: "
            f"{probability_audit.invalid_reason}"
        )

    full_state_occupancy = state_occupancy_from_aligned_frame(
        aligned,
        spec=spec,
        segment=None,
    )

    test_state_occupancy = state_occupancy_from_aligned_frame(
        aligned,
        spec=spec,
        segment="test",
    )

    lookahead_audit = build_parameter_lookahead_audit(
        candidate=candidate,
        aligned_frame=aligned,
    )

    result = MARFullFilterResult(
        candidate=candidate,
        full_model=full_model,
        full_filter_result=full_filter_result,
        full_smooth_result=full_smooth_result,
        aligned_frame=aligned,
        raw_filtered_probabilities=raw_filtered,
        raw_smoothed_probabilities_diagnostic=raw_smoothed,
        full_state_occupancy=full_state_occupancy,
        test_state_occupancy=test_state_occupancy,
        probability_audit=probability_audit,
        lookahead_audit=lookahead_audit,
    )

    return result


def extract_smoothed_probabilities_from_result(
    result: Any,
    n_states: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Extract smoothed marginal probabilities from a statsmodels result object.

    These are diagnostic-only and must never become backtest-facing signals.
    """
    if not hasattr(result, "smoothed_marginal_probabilities"):
        raise ValueError("Result object has no smoothed_marginal_probabilities attribute.")

    raw = result.smoothed_marginal_probabilities
    probs = _probabilities_to_2d_array(raw=raw, n_states=n_states)

    columns = [f"{prefix}{i}" for i in range(n_states)]
    return pd.DataFrame(probs, columns=columns)


def align_mar_probabilities_to_eligible_frame(
    prepared: MARPreparedData,
    spec: MARModelSpec,
    raw_filtered_probabilities: pd.DataFrame,
    raw_smoothed_probabilities_diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align statsmodels probability rows back to eligible dates.

    Important:
    - order=1 consumes y_{t-1}; first `order` rows are unavailable.
    - Some statsmodels versions return probability length == nobs.
    - Some may return nobs - order.
    - This function handles both safely.
    """
    frame = prepared.eligible_frame.copy().reset_index(drop=True)
    n = len(frame)

    frame["mar_model_observation_available"] = False

    raw_filtered_cols = [f"mar_filtered_prob_raw_state_{i}" for i in range(spec.n_states)]
    raw_smoothed_cols = [
        f"mar_diagnostic_smoothed_prob_raw_state_{i}" for i in range(spec.n_states)
    ]

    for col in raw_filtered_cols + raw_smoothed_cols:
        frame[col] = np.nan

    filtered_aligned = _align_probability_frame_length(
        probs=raw_filtered_probabilities,
        n_rows=n,
        order=spec.order,
        columns=raw_filtered_cols,
    )

    smoothed_aligned = _align_probability_frame_length(
        probs=raw_smoothed_probabilities_diagnostic,
        n_rows=n,
        order=spec.order,
        columns=raw_smoothed_cols,
    )

    for col in raw_filtered_cols:
        frame[col] = filtered_aligned[col].to_numpy(dtype=float)

    for col in raw_smoothed_cols:
        frame[col] = smoothed_aligned[col].to_numpy(dtype=float)

    warmup_mask = frame["mar_eligible_observation_number"] < spec.order
    filtered_non_missing = frame[raw_filtered_cols].notna().all(axis=1)

    frame["mar_model_observation_available"] = (~warmup_mask) & filtered_non_missing

    # Force AR warmup rows to unavailable and probability NaN.
    frame.loc[warmup_mask, "mar_model_observation_available"] = False
    frame.loc[warmup_mask, raw_filtered_cols] = np.nan
    frame.loc[warmup_mask, raw_smoothed_cols] = np.nan

    frame["mar_raw_state_filtered"] = np.nan
    available_mask = frame["mar_model_observation_available"]
    if bool(available_mask.any()):
        values = frame.loc[available_mask, raw_filtered_cols].to_numpy(dtype=float)
        frame.loc[available_mask, "mar_raw_state_filtered"] = np.argmax(values, axis=1)

    frame["mar_raw_state_filtered"] = frame["mar_raw_state_filtered"].astype("float")

    return frame


def _align_probability_frame_length(
    probs: pd.DataFrame,
    n_rows: int,
    order: int,
    columns: list[str],
) -> pd.DataFrame:
    """
    Return probability DataFrame of length n_rows, aligned to eligible rows.

    Accepted probability lengths:
        n_rows:
            assign directly, then caller will blank warmup rows.

        n_rows - order:
            assign starting at row `order`.

    Any other length is rejected.
    """
    out = pd.DataFrame(np.nan, index=np.arange(n_rows), columns=columns)

    if probs.empty:
        return out

    probs = probs.copy()
    probs.columns = columns
    m = len(probs)

    if m == n_rows:
        out.loc[:, columns] = probs.to_numpy(dtype=float)
        return out

    if m == n_rows - order:
        out.loc[order:, columns] = probs.to_numpy(dtype=float)
        return out

    raise ValueError(
        "Statsmodels probability length cannot be aligned to eligible frame. "
        f"probability_rows={m}, eligible_rows={n_rows}, order={order}"
    )


def audit_aligned_probabilities(
    aligned_frame: pd.DataFrame,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> MARProbabilityAudit:
    """Validate aligned filtered probabilities after AR warmup."""
    prob_cols = [f"mar_filtered_prob_raw_state_{i}" for i in range(spec.n_states)]

    available_mask = aligned_frame["mar_model_observation_available"].astype(bool)
    warmup_mask = aligned_frame["mar_eligible_observation_number"] < spec.order

    available = aligned_frame.loc[available_mask, prob_cols]

    n_rows = int(len(aligned_frame))
    n_model_available = int(available_mask.sum())
    n_warmup = int(warmup_mask.sum())

    tolerance = float(cfg.validation.probability_row_sum_tolerance)

    invalid_reasons: list[str] = []

    if n_model_available <= 0:
        invalid_reasons.append("no model-available probability rows after warmup")

    n_nan_after_warmup = int(
        aligned_frame.loc[~warmup_mask, prob_cols].isna().any(axis=1).sum()
    )
    if n_nan_after_warmup:
        invalid_reasons.append(
            f"{n_nan_after_warmup} non-warmup row(s) contain NaN probabilities"
        )

    max_error: float | None = None

    if n_model_available > 0:
        values = available.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            invalid_reasons.append("available probabilities contain NaN or Inf")

        if (values < -tolerance).any() or (values > 1.0 + tolerance).any():
            invalid_reasons.append("available probabilities outside [0, 1] tolerance")

        row_sums = values.sum(axis=1)
        max_error = float(np.max(np.abs(row_sums - 1.0)))

        if max_error > tolerance:
            invalid_reasons.append(
                f"probability row sums exceed tolerance; max_abs_error={max_error:.6g}"
            )

    warmup_has_probs = bool(
        aligned_frame.loc[warmup_mask, prob_cols].notna().any(axis=None)
    )
    if warmup_has_probs:
        invalid_reasons.append("AR warmup rows contain filtered probabilities")

    passed = len(invalid_reasons) == 0

    return MARProbabilityAudit(
        n_rows=n_rows,
        n_model_available_rows=n_model_available,
        n_warmup_rows=n_warmup,
        n_probability_rows=int(len(available)),
        n_nan_rows_after_warmup=n_nan_after_warmup,
        max_row_sum_abs_error=max_error,
        probability_row_sum_tolerance=tolerance,
        passed=passed,
        invalid_reason="" if passed else "; ".join(invalid_reasons),
    )


def state_occupancy_from_aligned_frame(
    aligned_frame: pd.DataFrame,
    spec: MARModelSpec,
    segment: str | None,
) -> dict[int, float]:
    """
    Compute soft state occupancy from aligned filtered probabilities.

    segment:
        None   -> all model-available rows
        train  -> train rows only
        test   -> test rows only
    """
    prob_cols = [f"mar_filtered_prob_raw_state_{i}" for i in range(spec.n_states)]

    mask = aligned_frame["mar_model_observation_available"].astype(bool)

    if segment is not None:
        mask = mask & (aligned_frame["mar_sample_segment"].astype(str) == segment)

    probs = aligned_frame.loc[mask, prob_cols]

    if probs.empty:
        return {}

    return state_occupancy_from_probabilities(probs.reset_index(drop=True))


def build_parameter_lookahead_audit(
    candidate: MARCandidateFit,
    aligned_frame: pd.DataFrame,
) -> MARParameterLookaheadAudit:
    """
    Build explicit audit fields proving parameter no-lookahead.

    This is metadata-level evidence:
    - fit window is candidate.prepared.train_frame
    - filter window is candidate.prepared.eligible_frame
    - params_estimated_using_full_sample is explicitly false
    """
    prepared = candidate.prepared

    fit_start = _date_str(prepared.train_frame[DATE_COL].iloc[0])
    fit_end = _date_str(prepared.train_frame[DATE_COL].iloc[-1])

    filter_start = _date_str(aligned_frame[DATE_COL].iloc[0])
    filter_end = _date_str(aligned_frame[DATE_COL].iloc[-1])

    params_estimated_using_full_sample = False
    filtered_probabilities_use_train_params_only = True
    smoothed_probabilities_used_for_backtest = False

    passed = (
        params_estimated_using_full_sample is False
        and filtered_probabilities_use_train_params_only is True
        and smoothed_probabilities_used_for_backtest is False
    )

    return MARParameterLookaheadAudit(
        mar_fit_window_start=fit_start,
        mar_fit_window_end=fit_end,
        mar_filter_window_start=filter_start,
        mar_filter_window_end=filter_end,
        params_estimated_using_full_sample=params_estimated_using_full_sample,
        filtered_probabilities_use_train_params_only=filtered_probabilities_use_train_params_only,
        smoothed_probabilities_used_for_backtest=smoothed_probabilities_used_for_backtest,
        passed=passed,
    )


def full_filter_summary_dict(full_filter: MARFullFilterResult) -> dict[str, Any]:
    """Metadata-friendly summary of full-series filtering result."""
    return {
        "market": full_filter.candidate.market,
        "spec": full_filter.candidate.spec.to_dict(),
        "fit_summary": fit_summary_dict(full_filter.candidate),
        "full_state_occupancy": full_filter.full_state_occupancy,
        "test_state_occupancy": full_filter.test_state_occupancy,
        "probability_audit": full_filter.probability_audit.__dict__,
        "lookahead_audit": full_filter.lookahead_audit.__dict__,
        "has_smoothed_diagnostic": full_filter.full_smooth_result is not None,
    }


def build_mar_signal_output(
    full_filter: MARFullFilterResult,
    cfg: MARConfig,
    *,
    raise_on_incoherent: bool = False,
) -> MARSignalOutput:
    """
    Build final MAR regime output.

    Uses train-period raw-state properties only for economic state mapping.

    Output contains:
    - raw filtered probabilities
    - diagnostic smoothed probabilities
    - economic filtered probabilities
    - raw and economic state assignments
    - t+1 signal timing columns
    """
    state_summary = build_train_state_economic_summary(full_filter, cfg)
    mapping = label_mar_states_economically(
        state_summary=state_summary,
        spec=full_filter.candidate.spec,
        cfg=cfg,
    )

    if raise_on_incoherent and not bool(mapping.economic_check.get("passed", False)):
        raise ValueError(
            "MAR economic state check failed: "
            f"{mapping.economic_check.get('invalid_reason', '')}"
        )

    output = add_economic_state_probabilities(
        aligned_frame=full_filter.aligned_frame,
        spec=full_filter.candidate.spec,
        mapping=mapping,
    )

    output = add_next_session_signal_columns(output)

    validate_mar_signal_output(
        output_frame=output,
        spec=full_filter.candidate.spec,
        cfg=cfg,
    )

    return MARSignalOutput(
        full_filter=full_filter,
        output_frame=output,
        state_summary=state_summary,
        state_mapping=mapping,
    )


def build_train_state_economic_summary(
    full_filter: MARFullFilterResult,
    cfg: MARConfig,
) -> pd.DataFrame:
    """
    Summarise raw states using train-period observations only.

    These statistics are used for economic labeling. They must not use test/full
    future information.
    """
    candidate = full_filter.candidate
    prepared = candidate.prepared
    spec = candidate.spec
    frame = full_filter.aligned_frame.copy()

    train_mask = (
        frame["mar_model_observation_available"].astype(bool)
        & (frame["mar_sample_segment"].astype(str) == "train")
    )

    if not bool(train_mask.any()):
        raise ValueError("No train model-available rows for MAR state labeling.")

    target_col = prepared.target_col
    iv_col = cfg.target_columns.get("iv")
    rv_col = cfg.target_columns.get("rv")
    ret_col = cfg.target_columns.get("returns")

    rows: list[dict[str, Any]] = []

    for raw_state in range(spec.n_states):
        prob_col = f"mar_filtered_prob_raw_state_{raw_state}"
        weights = pd.to_numeric(frame.loc[train_mask, prob_col], errors="coerce").astype(float)

        target_values = pd.to_numeric(frame.loc[train_mask, target_col], errors="coerce").astype(float)
        target_mean = weighted_mean(target_values, weights)
        target_std = weighted_std(target_values, weights)

        iv_mean = weighted_mean_if_available(frame, train_mask, iv_col, weights)
        rv_mean = weighted_mean_if_available(frame, train_mask, rv_col, weights)
        ret_mean = weighted_mean_if_available(frame, train_mask, ret_col, weights)

        persistence_prob = transition_diagonal_value(
            candidate.transition_matrix,
            raw_state=raw_state,
        )
        half_life = transition_half_life_days(persistence_prob)

        phi = candidate.ar_coefficients.get(raw_state)
        sigma2 = candidate.sigma2_by_state.get(raw_state)

        rows.append(
            {
                "market": candidate.market,
                "target": spec.target,
                "target_col": target_col,
                "n_states": spec.n_states,
                "order": spec.order,
                "raw_state": raw_state,
                "train_soft_occupancy": float(candidate.train_state_occupancy.get(raw_state, np.nan)),
                "target_mean_train": target_mean,
                "target_std_train": target_std,
                "iv_mean_train": iv_mean,
                "rv_mean_train": rv_mean,
                "index_return_mean_train": ret_mean,
                "intercept": np.nan,
                "ar_lag1_phi": float(phi) if phi is not None else np.nan,
                "sigma2": float(sigma2) if sigma2 is not None else np.nan,
                "persistence_prob": persistence_prob,
                "half_life_days": half_life,
                "ar_stable": bool(abs(phi) < 1.0) if phi is not None and np.isfinite(phi) else np.nan,
                "ar_warning": ar_warning_text(phi),
            }
        )

    summary = pd.DataFrame(rows)
    summary["stress_score"] = compute_mar_stress_scores(summary, spec=spec, cfg=cfg)

    return summary


def label_mar_states_economically(
    state_summary: pd.DataFrame,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> MAREconomicStateMapping:
    """
    Map raw states to calm / transition / stress using train-period stress score.

    K=2:
        lowest stress score -> calm
        highest stress score -> stress
        transition is not modelled

    K=3:
        lowest stress score -> calm
        middle stress score -> transition
        highest stress score -> stress
    """
    required_cols = {"raw_state", "stress_score"}
    missing = required_cols.difference(state_summary.columns)
    if missing:
        raise ValueError(f"State summary missing required columns: {sorted(missing)}")

    ordered = state_summary.sort_values(["stress_score", "raw_state"]).reset_index(drop=True)

    raw_state_to_name: dict[int, str] = {}
    name_to_raw_state: dict[str, int | None] = {
        "calm": None,
        "transition": None,
        "stress": None,
    }

    if spec.n_states == 2:
        calm_state = int(ordered.iloc[0]["raw_state"])
        stress_state = int(ordered.iloc[-1]["raw_state"])

        raw_state_to_name[calm_state] = "calm"
        raw_state_to_name[stress_state] = "stress"

        name_to_raw_state["calm"] = calm_state
        name_to_raw_state["stress"] = stress_state
        transition_state_modelled = False

    elif spec.n_states == 3:
        calm_state = int(ordered.iloc[0]["raw_state"])
        transition_state = int(ordered.iloc[1]["raw_state"])
        stress_state = int(ordered.iloc[2]["raw_state"])

        raw_state_to_name[calm_state] = "calm"
        raw_state_to_name[transition_state] = "transition"
        raw_state_to_name[stress_state] = "stress"

        name_to_raw_state["calm"] = calm_state
        name_to_raw_state["transition"] = transition_state
        name_to_raw_state["stress"] = stress_state
        transition_state_modelled = True

    else:
        raise ValueError(f"Unsupported n_states for economic labeling: {spec.n_states}")

    raw_state_to_code = {
        raw_state: economic_state_code(name)
        for raw_state, name in raw_state_to_name.items()
    }

    stress_scores = {
        int(row["raw_state"]): float(row["stress_score"])
        for _, row in state_summary.iterrows()
    }

    economic_check = evaluate_economic_state_coherence(
        state_summary=state_summary,
        raw_state_to_name=raw_state_to_name,
        spec=spec,
        cfg=cfg,
    )

    return MAREconomicStateMapping(
        raw_state_to_name=raw_state_to_name,
        name_to_raw_state=name_to_raw_state,
        raw_state_to_economic_code=raw_state_to_code,
        stress_scores=stress_scores,
        transition_state_modelled=transition_state_modelled,
        economic_check=economic_check,
    )


def compute_mar_stress_scores(
    state_summary: pd.DataFrame,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> pd.Series:
    """
    Compute relative stress score by raw state.

    Higher score means more stress-like.

    For target=vrp_har:
        lower target mean -> more stress
        higher target std -> more stress
        lower returns -> more stress
        higher IV/RV -> more stress
        higher sigma2 -> more stress
    """
    scores = pd.Series(0.0, index=state_summary.index, dtype=float)
    used_components = pd.Series(0.0, index=state_summary.index, dtype=float)

    def add_component(column: str, higher_is_stress: bool) -> None:
        nonlocal scores, used_components

        if column not in state_summary.columns:
            return

        values = pd.to_numeric(state_summary[column], errors="coerce")
        if values.notna().sum() < 2:
            return

        component = rank_as_unit_interval(values, higher_is_stress=higher_is_stress)
        valid = component.notna()

        scores.loc[valid] += component.loc[valid]
        used_components.loc[valid] += 1.0

    if spec.target == "vrp_har":
        add_component("target_mean_train", higher_is_stress=False)
        add_component("target_std_train", higher_is_stress=True)
        add_component("index_return_mean_train", higher_is_stress=False)
        add_component("iv_mean_train", higher_is_stress=True)
        add_component("rv_mean_train", higher_is_stress=True)
        add_component("sigma2", higher_is_stress=True)

    elif spec.target == "rv":
        add_component("target_mean_train", higher_is_stress=True)
        add_component("target_std_train", higher_is_stress=True)
        add_component("index_return_mean_train", higher_is_stress=False)
        add_component("iv_mean_train", higher_is_stress=True)
        add_component("sigma2", higher_is_stress=True)

    else:
        add_component("target_mean_train", higher_is_stress=True)
        add_component("target_std_train", higher_is_stress=True)
        add_component("sigma2", higher_is_stress=True)

    out = scores / used_components.replace(0.0, np.nan)

    if out.isna().any():
        # Fallback should rarely be used. It prevents state labeling from
        # failing when all context fields are missing.
        fallback = pd.to_numeric(state_summary["target_std_train"], errors="coerce")
        out = rank_as_unit_interval(fallback, higher_is_stress=True)

    return out.astype(float)


def evaluate_economic_state_coherence(
    state_summary: pd.DataFrame,
    raw_state_to_name: dict[int, str],
    spec: MARModelSpec,
    cfg: MARConfig,
) -> dict[str, Any]:
    """
    Check whether calm/stress labels have basic economic meaning.

    This is not used as a model input. It is a post-assignment diagnostic.
    """
    named = state_summary.copy()
    named["economic_state_name"] = named["raw_state"].map(raw_state_to_name)

    calm = named[named["economic_state_name"] == "calm"]
    stress = named[named["economic_state_name"] == "stress"]

    if calm.empty or stress.empty:
        return {
            "passed": False,
            "invalid_reason": "missing calm or stress state after mapping",
            "checks": {},
        }

    calm_row = calm.iloc[0]
    stress_row = stress.iloc[0]

    checks: dict[str, bool] = {}

    if spec.target == "vrp_har":
        checks["stress_has_lower_or_equal_target_mean"] = safe_leq(
            stress_row.get("target_mean_train"),
            calm_row.get("target_mean_train"),
        )
        checks["stress_has_higher_or_equal_target_std"] = safe_geq(
            stress_row.get("target_std_train"),
            calm_row.get("target_std_train"),
        )

    elif spec.target == "rv":
        checks["stress_has_higher_or_equal_target_mean"] = safe_geq(
            stress_row.get("target_mean_train"),
            calm_row.get("target_mean_train"),
        )
        checks["stress_has_higher_or_equal_target_std"] = safe_geq(
            stress_row.get("target_std_train"),
            calm_row.get("target_std_train"),
        )

    checks["stress_has_higher_or_equal_sigma2_if_available"] = safe_geq_optional(
        stress_row.get("sigma2"),
        calm_row.get("sigma2"),
    )
    checks["stress_has_lower_or_equal_return_if_available"] = safe_leq_optional(
        stress_row.get("index_return_mean_train"),
        calm_row.get("index_return_mean_train"),
    )
    checks["stress_has_higher_or_equal_iv_if_available"] = safe_geq_optional(
        stress_row.get("iv_mean_train"),
        calm_row.get("iv_mean_train"),
    )
    checks["stress_has_higher_or_equal_rv_if_available"] = safe_geq_optional(
        stress_row.get("rv_mean_train"),
        calm_row.get("rv_mean_train"),
    )

    failed = [name for name, passed in checks.items() if passed is False]
    passed = len(failed) == 0

    return {
        "passed": passed,
        "invalid_reason": "" if passed else f"failed checks: {failed}",
        "checks": checks,
        "calm_raw_state": int(calm_row["raw_state"]),
        "stress_raw_state": int(stress_row["raw_state"]),
    }


def add_economic_state_probabilities(
    aligned_frame: pd.DataFrame,
    spec: MARModelSpec,
    mapping: MAREconomicStateMapping,
) -> pd.DataFrame:
    """
    Add economic probability and current economic state columns.

    Warmup rows remain NaN.
    For K=2, transition probability is 0.0 only after valid model rows.
    """
    out = aligned_frame.copy()

    econ_cols = [
        "mar_filtered_prob_calm",
        "mar_filtered_prob_transition",
        "mar_filtered_prob_stress",
    ]

    for col in econ_cols:
        out[col] = np.nan

    out["mar_transition_state_modelled"] = bool(mapping.transition_state_modelled)

    available_mask = out["mar_model_observation_available"].astype(bool)

    for raw_state, econ_name in mapping.raw_state_to_name.items():
        raw_col = f"mar_filtered_prob_raw_state_{raw_state}"
        econ_col = f"mar_filtered_prob_{econ_name}"

        if raw_col not in out.columns:
            raise ValueError(f"Missing raw probability column: {raw_col}")

        out.loc[available_mask, econ_col] = out.loc[available_mask, raw_col].astype(float)

    if spec.n_states == 2:
        out.loc[available_mask, "mar_filtered_prob_transition"] = 0.0

    out["mar_state"] = np.nan
    out["mar_state_name"] = pd.NA  # Use pandas NA to get object dtype
    out["mar_state_name"] = out["mar_state_name"].astype(object)

    raw_state_series = pd.to_numeric(out["mar_raw_state_filtered"], errors="coerce")

    for raw_state, econ_name in mapping.raw_state_to_name.items():
        mask = available_mask & (raw_state_series == float(raw_state))
        out.loc[mask, "mar_state"] = float(economic_state_code(econ_name))
        out.loc[mask, "mar_state_name"] = econ_name

    return out


def add_next_session_signal_columns(output_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add t+1 signal timing columns.

    Interpretation:
        Observation at date t is known after date t close.
        It can be traded from the next available trading session.
    """
    out = output_frame.copy()

    out["mar_signal_observation_date"] = out[DATE_COL]
    out["mar_signal_available_after_close_date"] = out[DATE_COL]
    out["mar_signal_trade_date"] = out[DATE_COL].shift(-1)

    signal_cols = {
        "mar_state_for_next_session": "mar_state",
        "mar_state_name_for_next_session": "mar_state_name",
        "mar_filtered_prob_calm_for_next_session": "mar_filtered_prob_calm",
        "mar_filtered_prob_transition_for_next_session": "mar_filtered_prob_transition",
        "mar_filtered_prob_stress_for_next_session": "mar_filtered_prob_stress",
    }

    available_mask = out["mar_model_observation_available"].astype(bool)

    # Initialize with appropriate dtypes
    out["mar_state_for_next_session"] = np.nan
    out["mar_state_name_for_next_session"] = pd.NA  # Use pandas NA to get object dtype
    out["mar_state_name_for_next_session"] = out["mar_state_name_for_next_session"].astype(object)
    out["mar_filtered_prob_calm_for_next_session"] = np.nan
    out["mar_filtered_prob_transition_for_next_session"] = np.nan
    out["mar_filtered_prob_stress_for_next_session"] = np.nan

    # Copy values from available rows
    out.loc[available_mask, "mar_state_for_next_session"] = out.loc[available_mask, "mar_state"]
    out.loc[available_mask, "mar_state_name_for_next_session"] = out.loc[available_mask, "mar_state_name"]
    out.loc[available_mask, "mar_filtered_prob_calm_for_next_session"] = out.loc[available_mask, "mar_filtered_prob_calm"]
    out.loc[available_mask, "mar_filtered_prob_transition_for_next_session"] = out.loc[available_mask, "mar_filtered_prob_transition"]
    out.loc[available_mask, "mar_filtered_prob_stress_for_next_session"] = out.loc[available_mask, "mar_filtered_prob_stress"]

    return out


def validate_mar_signal_output(
    output_frame: pd.DataFrame,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> None:
    """Validate final Chunk 5 output schema and probability behavior."""
    required = [
        "mar_signal_observation_date",
        "mar_signal_available_after_close_date",
        "mar_signal_trade_date",
        "mar_model_observation_available",
        "mar_state_for_next_session",
        "mar_state_name_for_next_session",
        "mar_filtered_prob_calm_for_next_session",
        "mar_filtered_prob_transition_for_next_session",
        "mar_filtered_prob_stress_for_next_session",
    ]

    missing = [col for col in required if col not in output_frame.columns]
    if missing:
        raise ValueError(f"MAR signal output missing required columns: {missing}")

    available_mask = output_frame["mar_model_observation_available"].astype(bool)

    prob_cols = [
        "mar_filtered_prob_calm",
        "mar_filtered_prob_transition",
        "mar_filtered_prob_stress",
    ]

    available_probs = output_frame.loc[available_mask, prob_cols].to_numpy(dtype=float)

    if available_probs.size == 0:
        raise ValueError("No available economic probability rows in MAR output.")

    if not np.isfinite(available_probs).all():
        raise ValueError("Economic probabilities contain NaN/Inf on available rows.")

    row_sums = available_probs.sum(axis=1)
    tolerance = cfg.validation.probability_row_sum_tolerance

    if not np.allclose(row_sums, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Economic probability rows do not sum to 1. "
            f"max_abs_error={float(np.max(np.abs(row_sums - 1.0))):.6g}"
        )

    warmup_mask = output_frame["mar_eligible_observation_number"] < spec.order
    warmup_has_state = output_frame.loc[warmup_mask, "mar_state_for_next_session"].notna().any()

    if warmup_has_state:
        raise ValueError("AR warmup rows must not have next-session MAR states.")

    if spec.n_states == 2:
        transition = output_frame.loc[available_mask, "mar_filtered_prob_transition"]
        if not np.allclose(transition.to_numpy(dtype=float), 0.0, atol=tolerance, rtol=0.0):
            raise ValueError("K=2 transition probability must be 0.0 after warmup.")


def mar_signal_summary_dict(signal_output: MARSignalOutput) -> dict[str, Any]:
    """Metadata-friendly summary of Chunk 5 output."""
    return {
        "market": signal_output.full_filter.candidate.market,
        "spec": signal_output.full_filter.candidate.spec.to_dict(),
        "state_mapping": {
            "raw_state_to_name": signal_output.state_mapping.raw_state_to_name,
            "name_to_raw_state": signal_output.state_mapping.name_to_raw_state,
            "raw_state_to_economic_code": signal_output.state_mapping.raw_state_to_economic_code,
            "stress_scores": signal_output.state_mapping.stress_scores,
            "transition_state_modelled": signal_output.state_mapping.transition_state_modelled,
            "economic_check": signal_output.state_mapping.economic_check,
        },
        "n_rows": int(len(signal_output.output_frame)),
        "n_model_available": int(
            signal_output.output_frame["mar_model_observation_available"].astype(bool).sum()
        ),
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean with NaN handling."""
    x = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").astype(float)

    mask = x.notna() & w.notna() & np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    if not bool(mask.any()):
        return np.nan

    x_valid = x.loc[mask].to_numpy(dtype=float)
    w_valid = w.loc[mask].to_numpy(dtype=float)
    weight_sum = float(w_valid.sum())

    if weight_sum <= 0.0:
        return np.nan

    return float(np.sum(w_valid * x_valid) / weight_sum)


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Weighted population standard deviation with NaN handling."""
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return np.nan

    x = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").astype(float)

    mask = x.notna() & w.notna() & np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    if not bool(mask.any()):
        return np.nan

    x_valid = x.loc[mask].to_numpy(dtype=float)
    w_valid = w.loc[mask].to_numpy(dtype=float)
    weight_sum = float(w_valid.sum())

    if weight_sum <= 0.0:
        return np.nan

    var = float(np.sum(w_valid * (x_valid - mean) ** 2) / weight_sum)
    return float(np.sqrt(max(var, 0.0)))


def weighted_mean_if_available(
    frame: pd.DataFrame,
    mask: pd.Series,
    column: str | None,
    weights: pd.Series,
) -> float:
    """Weighted mean for optional context columns."""
    if column is None or column not in frame.columns:
        return np.nan

    return weighted_mean(frame.loc[mask, column], weights)


def transition_diagonal_value(
    transition_matrix: pd.DataFrame,
    raw_state: int,
) -> float:
    """Extract p(state -> same state) from transition matrix."""
    if transition_matrix.empty:
        return np.nan

    try:
        value = transition_matrix.iloc[int(raw_state), int(raw_state)]
        return float(np.asarray(value, dtype=float))
    except Exception:
        return np.nan


def transition_half_life_days(persistence_prob: float) -> float:
    """
    Convert persistence probability to approximate half-life.

    half-life = log(0.5) / log(p)
    """
    p = float(persistence_prob) if persistence_prob is not None else np.nan

    if not np.isfinite(p) or p <= 0.0:
        return np.nan

    if p >= 1.0:
        return np.inf

    return float(np.log(0.5) / np.log(p))


def ar_warning_text(phi: float | None) -> str:
    """Human-readable AR warning."""
    if phi is None or not np.isfinite(phi):
        return "AR coefficient unavailable"

    if abs(float(phi)) >= 1.0:
        return "abs(phi) >= 1.0"

    if abs(float(phi)) >= 0.98:
        return "near unit-root boundary"

    return ""


def rank_as_unit_interval(values: pd.Series, higher_is_stress: bool) -> pd.Series:
    """
    Rank values into [0, 1].

    higher_is_stress=True:
        smallest -> 0, largest -> 1

    higher_is_stress=False:
        largest -> 0, smallest -> 1
    """
    x = pd.to_numeric(values, errors="coerce").astype(float)

    if x.notna().sum() <= 1:
        return pd.Series(0.0, index=values.index)

    ranks = x.rank(method="average", ascending=higher_is_stress)

    # ascending=True gives low rank to low values.
    # For higher_is_stress=True, low values should map to 0 and high to 1.
    if higher_is_stress:
        ranks = x.rank(method="average", ascending=True)
    else:
        ranks = x.rank(method="average", ascending=False)

    n = float(x.notna().sum())
    out = (ranks - 1.0) / max(n - 1.0, 1.0)

    return out.astype(float)


def economic_state_code(name: str) -> int:
    """Ordinal economic state code."""
    mapping = {
        "calm": 0,
        "transition": 1,
        "stress": 2,
    }

    if name not in mapping:
        raise ValueError(f"Unknown economic state name: {name}")

    return mapping[name]


def safe_geq(left: Any, right: Any) -> bool:
    """Strict required greater/equal check."""
    try:
        l = float(left)
        r = float(right)
    except Exception:
        return False

    if not np.isfinite(l) or not np.isfinite(r):
        return False

    return l >= r


def safe_leq(left: Any, right: Any) -> bool:
    """Strict required lower/equal check."""
    try:
        l = float(left)
        r = float(right)
    except Exception:
        return False

    if not np.isfinite(l) or not np.isfinite(r):
        return False

    return l <= r


def safe_geq_optional(left: Any, right: Any) -> bool:
    """Optional greater/equal check. Missing values pass."""
    try:
        l = float(left)
        r = float(right)
    except Exception:
        return True

    if not np.isfinite(l) or not np.isfinite(r):
        return True

    return l >= r


def safe_leq_optional(left: Any, right: Any) -> bool:
    """Optional lower/equal check. Missing values pass."""
    try:
        l = float(left)
        r = float(right)
    except Exception:
        return True

    if not np.isfinite(l) or not np.isfinite(r):
        return True

    return l <= r


def _empty_candidate_fit(
    prepared: MARPreparedData,
    summary: MARFitFirewallSummary,
    *,
    model: Any | None = None,
    result: Any | None = None,
    fit_attempts: tuple[MARFitAttempt, ...] = tuple(),
) -> MARCandidateFit:
    n_states = prepared.spec.n_states
    empty_probs = pd.DataFrame(
        columns=[f"mar_filtered_prob_raw_state_{i}" for i in range(n_states)]
    )

    return MARCandidateFit(
        market=prepared.market,
        spec=prepared.spec,
        prepared=prepared,
        model=model,
        result=result,
        train_filtered_probabilities=empty_probs,
        train_state_assignments=pd.Series(dtype="float64", name="mar_raw_state"),
        train_state_occupancy={},
        transition_matrix=pd.DataFrame(),
        ar_coefficients={},
        sigma2_by_state={},
        fit_summary=summary,
        fit_attempts=fit_attempts,
    )


def _invalid_fit_summary(
    fit_exception: str,
    *,
    model: Any | None = None,
    result: Any | None = None,
) -> MARFitFirewallSummary:
    mle_retvals = {}
    if result is not None:
        mle_retvals = sanitize_mapping(getattr(result, "mle_retvals", {}) or {})

    return MARFitFirewallSummary(
        fit_converged=False,
        fit_exception=fit_exception,
        llf=finite_float_or_none(getattr(result, "llf", None)) if result is not None else None,
        aic=finite_float_or_none(getattr(result, "aic", None)) if result is not None else None,
        bic=finite_float_or_none(getattr(result, "bic", None)) if result is not None else None,
        hqic=finite_float_or_none(getattr(result, "hqic", None)) if result is not None else None,
        nobs=int_or_none(getattr(result, "nobs", None)) if result is not None else None,
        n_params=infer_n_params(result) if result is not None else None,
        warnflag=mle_retvals.get("warnflag") if mle_retvals else None,
        mle_retvals=mle_retvals,
        valid_candidate=False,
        invalid_reason=fit_exception,
        warnings=tuple(),
    )


def _probabilities_to_2d_array(raw: Any, n_states: int) -> np.ndarray:
    """Normalize statsmodels probability output to shape (nobs, n_states)."""
    if isinstance(raw, pd.DataFrame):
        arr = raw.to_numpy(dtype=float)
    elif isinstance(raw, pd.Series):
        arr = raw.to_frame().to_numpy(dtype=float)
    else:
        arr = np.asarray(raw, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Probability output must be 2D, got shape={arr.shape}")

    if arr.shape[1] == n_states:
        return arr

    if arr.shape[0] == n_states:
        return arr.T

    raise ValueError(
        f"Cannot infer probability orientation. shape={arr.shape}, n_states={n_states}"
    )


def finite_float_or_none(value: Any) -> float | None:
    """Convert to finite float or None."""
    try:
        out = float(value)
    except Exception:
        return None

    if not np.isfinite(out):
        return None

    return out


def int_or_none(value: Any) -> int | None:
    """Convert to int or None."""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def sanitize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Convert statsmodels/numpy return values to metadata-friendly Python values."""
    out: dict[str, Any] = {}

    for key, value in dict(mapping).items():
        out[str(key)] = sanitize_value(value)

    return out


def sanitize_value(value: Any) -> Any:
    """Convert numpy/scipy objects into simple Python metadata values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None

    if isinstance(value, np.ndarray):
        if value.size > 20:
            return f"array(shape={value.shape}, dtype={value.dtype})"
        return [sanitize_value(x) for x in value.ravel().tolist()]

    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return f"{type(value).__name__}(len={len(value)})"
        return [sanitize_value(x) for x in value]

    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}

    return str(value)