# src/vrp/forecasting/har_rv.py

"""
HAR-RV forecasting module for Phase 4.
- HARConfig
- Config loading
- Input validation
- HAR lagged feature construction
- Target window metadata
- Validation-only recomputation of Phase 3 forward target

Important Phase 4 rule:
The primary HAR target is the existing Phase 3 column:

    rv_gk_22d_forward_ann_label

This module may recompute the forward target only for validation/testing.
The recomputed target must match the Phase 3 label, otherwise Phase 4 fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import importlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from pydantic import BaseModel, Field

from vrp.forecasting.har_registry import (
    HAR_FEATURE_COLUMNS,
    HAR_TARGET_COLUMNS,
    assert_har_registry_is_valid,
    assert_primary_har_features,
)


DEFAULT_HAR_MODEL_NAME = "direct_har_22d_gk_conservative_lag1"

DEFAULT_TARGET_COL = "rv_gk_22d_forward_ann_label"
DEFAULT_DAILY_RV_COL = "rv_gk_daily"
DEFAULT_IV_COL = "iv_ann"


class HARConfig(BaseModel):
    """
    Configuration for Phase 4 HAR-RV forecasting.
    """

    primary_estimator: str = "garman_klass"
    primary_daily_rv_col: str = DEFAULT_DAILY_RV_COL
    primary_forward_label_col: str = DEFAULT_TARGET_COL
    primary_iv_col: str = DEFAULT_IV_COL

    forecast_horizon: int = Field(default=22, ge=1)
    annualization_periods: int = Field(default=252, ge=1)

    timing_mode: str = "conservative_lag1"
    model_type: str = "direct_har_22d"

    oos_mode: str = "expanding"
    min_train_observations: int = Field(default=500, ge=1)
    rolling_train_window: int = Field(default=1000, ge=1)

    hac_maxlags: int = Field(default=22, ge=0)
    forecast_floor: float = Field(default=1.0e-8, gt=0)

    target_source: str = "phase3_existing_label"
    target_validation_mode: str = "recompute_and_compare"
    target_validation_atol: float = Field(default=1.0e-12, ge=0)
    target_validation_rtol: float = Field(default=1.0e-10, ge=0)

    write_rolling_baseline: bool = True

    markets: list[str] = Field(default_factory=lambda: ["US", "INDIA"])

    input_paths: dict[str, str] = Field(default_factory=dict)
    forecast_output_paths: dict[str, str] = Field(default_factory=dict)
    vrp_har_output_paths: dict[str, str] = Field(default_factory=dict)
    report_paths: dict[str, str] = Field(default_factory=dict)
    figure_paths: dict[str, str] = Field(default_factory=dict)
    # Compute backend selection for batched closed-form OLS
    compute_backend: str = "auto"
    # choices: auto | cpu_statsmodels | cpu_numpy_batched | torch_batched

    # Torch device selection when torch backend is used or auto-detected
    torch_device: str = "auto"
    # choices: auto | cuda | cpu

    # Torch dtype for batched computation
    torch_dtype: str = "float64"
    # choices: float64 | float32

    # Frequency for HAC covariance estimation for coefficient inference
    coefficient_hac_frequency: str = "month_end"
    # choices: daily | month_end | quarter_end | final | none

    # How often to store coefficient history values
    coefficient_history_frequency: str = "daily"
    # choices: daily | month_end | quarter_end | final

    # Matrix singularity handling policy
    matrix_singularity_policy: str = "pinv"
    # choices: pinv | skip

    # Condition number threshold for numeric stability checks
    condition_number_threshold: float = 1.0e12

    @property
    def feature_cols(self) -> list[str]:
        """
        Primary HAR predictors.
        """
        return list(HAR_FEATURE_COLUMNS)

    @property
    def target_col(self) -> str:
        """
        Primary HAR target column.
        """
        return self.primary_forward_label_col

    @property
    def daily_rv_col(self) -> str:
        """
        Primary daily realised variance column.
        """
        return self.primary_daily_rv_col

    @property
    def iv_col(self) -> str:
        """
        Primary implied variance column.
        """
        return self.primary_iv_col


def load_har_config(path: str | Path) -> HARConfig:
    """
    Load HAR configuration from YAML.

    Parameters
    ----------
    path:
        Path to configs/har_rv.yaml.

    Returns
    -------
    HARConfig
        Parsed config object.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"HAR config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError(f"HAR config file is empty: {config_path}")

    if not isinstance(raw_config, dict):
        raise TypeError(
            f"HAR config must parse to a dictionary. Got: {type(raw_config)}"
        )

    config = HARConfig(**raw_config)

    assert_har_registry_is_valid()
    assert_primary_har_features(config.feature_cols)

    if config.target_col not in HAR_TARGET_COLUMNS:
        raise ValueError(
            f"Configured target column {config.target_col!r} is not registered "
            f"as a HAR target. Registered targets: {HAR_TARGET_COLUMNS}"
        )

    return config


def resolve_compute_backend(config: HARConfig) -> str:
    """
    Resolve compute backend choice given a config.

    Returns one of: cpu_statsmodels, cpu_numpy_batched, torch_batched
    """
    requested = str(config.compute_backend).lower().strip()

    if requested == "auto":
        # torch_batched only if torch + CUDA available; otherwise cpu_numpy_batched
        try:
            torch_spec = importlib.import_module("torch")
            cuda_available = getattr(torch_spec.cuda, "is_available", lambda: False)()
            if cuda_available:
                return "torch_batched"
        except Exception:
            pass
        return "cpu_numpy_batched"

    if requested == "torch_batched":
        try:
            importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "compute_backend=\"torch_batched\" requested but torch is not importable"
            ) from exc

    if requested in {"cpu_statsmodels", "cpu_numpy_batched", "torch_batched"}:
        return requested

    raise ValueError(f"Unknown compute_backend: {config.compute_backend}")


def resolve_torch_device(config: HARConfig) -> str:
    """Return resolved torch device string: 'cuda' or 'cpu' or 'none' if torch missing."""
    device = str(config.torch_device).lower().strip()

    if device == "auto":
        try:
            torch_spec = importlib.import_module("torch")
            if getattr(torch_spec.cuda, "is_available", lambda: False)():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    if device in {"cuda", "cpu"}:
        return device

    raise ValueError(f"Unknown torch_device: {config.torch_device}")


def resolve_torch_dtype(config: HARConfig):
    """Return torch dtype (if torch installed) or numpy dtype string."""
    dtype = str(config.torch_dtype).lower().strip()
    if dtype == "float64":
        return "float64"
    if dtype == "float32":
        return "float32"
    raise ValueError(f"Unknown torch_dtype: {config.torch_dtype}")


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    name: str,
) -> None:
    """
    Raise ValueError if required columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def _sort_by_date(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    Return date-sorted copy with datetime dates.
    """
    _require_columns(df, ["date"], name=name)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    bad_date_mask = out["date"].isna()
    if bad_date_mask.any():
        bad_count = int(bad_date_mask.sum())
        bad_indices = out.index[bad_date_mask].tolist()[:10]
        raise ValueError(
            f"{name} contains {bad_count} invalid date value(s). "
            f"First bad indices: {bad_indices}"
        )

    out = out.sort_values("date").reset_index(drop=True)

    duplicate_mask = out["date"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_dates = (
            out.loc[duplicate_mask, "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()[:10]
        )
        raise ValueError(
            f"{name} contains duplicate date(s). "
            f"First duplicate dates: {duplicate_dates}"
        )

    return out


def _coerce_numeric_column(
    df: pd.DataFrame,
    column: str,
    *,
    name: str,
    allow_missing: bool = True,
) -> pd.Series:
    """
    Convert a column to numeric while detecting bad non-missing values.
    """
    _require_columns(df, [column], name=name)

    raw = df[column]
    values = pd.to_numeric(raw, errors="coerce")

    bad_parse_mask = raw.notna() & values.isna()
    if bad_parse_mask.any():
        bad_count = int(bad_parse_mask.sum())
        bad_indices = values.index[bad_parse_mask].tolist()[:10]
        raise ValueError(
            f"{name}.{column} contains {bad_count} non-numeric non-missing "
            f"value(s). First bad indices: {bad_indices}"
        )

    if not allow_missing:
        missing_mask = values.isna()
        if missing_mask.any():
            bad_count = int(missing_mask.sum())
            bad_indices = values.index[missing_mask].tolist()[:10]
            raise ValueError(
                f"{name}.{column} contains {bad_count} missing value(s). "
                f"First bad indices: {bad_indices}"
            )

    return values


def _validate_non_negative_variance(
    values: pd.Series,
    *,
    column: str,
    name: str,
    allow_missing: bool = True,
) -> None:
    """
    Validate that a variance column is finite and non-negative.
    """
    check = values.copy()

    if allow_missing:
        check = check.dropna()

    if not allow_missing and values.isna().any():
        bad_count = int(values.isna().sum())
        raise ValueError(f"{name}.{column} contains {bad_count} missing value(s).")

    non_finite_mask = ~np.isfinite(check)
    if non_finite_mask.any():
        bad_count = int(non_finite_mask.sum())
        raise ValueError(f"{name}.{column} contains {bad_count} non-finite value(s).")

    negative_mask = check < 0
    if negative_mask.any():
        bad_count = int(negative_mask.sum())
        raise ValueError(f"{name}.{column} contains {bad_count} negative value(s).")


def validate_har_input_panel(panel: pd.DataFrame, config: HARConfig | None = None) -> pd.DataFrame:
    """
    Validate a Phase 3 VRP panel before HAR feature construction.

    Required Phase 3 columns:
    - date
    - market
    - rv_gk_daily
    - rv_gk_22d_forward_ann_label
    - iv_ann

    Returns
    -------
    pd.DataFrame
        Clean date-sorted copy with numeric RV/target/IV columns.
    """
    if config is None:
        config = HARConfig()

    name = "HAR input panel"

    out = _sort_by_date(panel, name=name)

    required_cols = [
        "date",
        "market",
        config.daily_rv_col,
        config.target_col,
        config.iv_col,
    ]

    _require_columns(out, required_cols, name=name)

    out[config.daily_rv_col] = _coerce_numeric_column(
        out,
        config.daily_rv_col,
        name=name,
        allow_missing=True,
    )
    out[config.target_col] = _coerce_numeric_column(
        out,
        config.target_col,
        name=name,
        allow_missing=True,
    )
    out[config.iv_col] = _coerce_numeric_column(
        out,
        config.iv_col,
        name=name,
        allow_missing=True,
    )

    _validate_non_negative_variance(
        out[config.daily_rv_col],
        column=config.daily_rv_col,
        name=name,
        allow_missing=True,
    )
    _validate_non_negative_variance(
        out[config.target_col],
        column=config.target_col,
        name=name,
        allow_missing=True,
    )
    _validate_non_negative_variance(
        out[config.iv_col],
        column=config.iv_col,
        name=name,
        allow_missing=True,
    )

    market_values = out["market"].dropna().astype(str).str.upper().unique().tolist()
    if len(market_values) != 1:
        raise ValueError(
            "HAR input panel must contain exactly one market. "
            f"Found: {market_values}"
        )

    out["market"] = market_values[0]

    assert_har_registry_is_valid()

    return out


def make_har_features(
    panel: pd.DataFrame,
    daily_rv_col: str = DEFAULT_DAILY_RV_COL,
    horizon: int = 22,
    annualization_periods: int = 252,
) -> pd.DataFrame:
    """
    Build conservative lagged HAR-RV predictors.

    At date t, features use realised variance only through t-1.

    Formulas:
        har_rv_d_lag1_ann_t =
            annualization_periods * rv_daily_{t-1}

        har_rv_w_lag1_ann_t =
            annualization_periods * mean(rv_daily_{t-5}, ..., rv_daily_{t-1})

        har_rv_m_lag1_ann_t =
            annualization_periods * mean(rv_daily_{t-22}, ..., rv_daily_{t-1})

    Notes
    -----
    - Current-day rv_gk_daily_t is not used.
    - IV and VRP columns are not used.
    - Target columns are not used.
    - Rows are retained with NaN features where history is incomplete.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1. Got: {horizon}")

    if annualization_periods <= 0:
        raise ValueError(
            f"annualization_periods must be positive. Got: {annualization_periods}"
        )

    out = _sort_by_date(panel, name="HAR feature input panel")

    daily_rv = _coerce_numeric_column(
        out,
        daily_rv_col,
        name="HAR feature input panel",
        allow_missing=True,
    )

    _validate_non_negative_variance(
        daily_rv,
        column=daily_rv_col,
        name="HAR feature input panel",
        allow_missing=True,
    )

    lagged_daily = daily_rv.shift(1)

    out["har_rv_d_lag1_ann"] = annualization_periods * lagged_daily
    out["har_rv_w_lag1_ann"] = (
        annualization_periods
        * lagged_daily.rolling(window=5, min_periods=5).mean()
    )
    out["har_rv_m_lag1_ann"] = (
        annualization_periods
        * lagged_daily.rolling(window=horizon, min_periods=horizon).mean()
    )

    assert_primary_har_features(HAR_FEATURE_COLUMNS)

    return out


def add_forward_target_metadata(
    panel: pd.DataFrame,
    horizon: int = 22,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """
    Add forward target-window metadata.

    For each row t:
        target_start_date_t = first trading date after t
        target_end_date_t   = horizon-th trading date after t

    The target value itself is not constructed here. The primary target remains
    the existing Phase 3 column rv_gk_22d_forward_ann_label.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1. Got: {horizon}")

    out = _sort_by_date(panel, name="HAR target metadata input panel")

    _require_columns(out, ["date", target_col], name="HAR target metadata input panel")

    out["target_col"] = target_col
    out["target_start_date"] = out["date"].shift(-1)
    out["target_end_date"] = out["date"].shift(-horizon)

    return out


def recompute_forward_target_for_validation(
    panel: pd.DataFrame,
    daily_rv_col: str = DEFAULT_DAILY_RV_COL,
    horizon: int = 22,
    annualization_periods: int = 252,
    output_col: str = "_recomputed_rv_gk_22d_forward_ann_label",
) -> pd.DataFrame:
    """
    Recompute the forward RV target only for validation/testing.

    Formula:
        recomputed_target_t =
            annualization_periods
            * mean(rv_daily_{t+1}, ..., rv_daily_{t+horizon})

    This must match the existing Phase 3 target:
        rv_gk_22d_forward_ann_label
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1. Got: {horizon}")

    if annualization_periods <= 0:
        raise ValueError(
            f"annualization_periods must be positive. Got: {annualization_periods}"
        )

    out = _sort_by_date(panel, name="HAR target validation input panel")

    daily_rv = _coerce_numeric_column(
        out,
        daily_rv_col,
        name="HAR target validation input panel",
        allow_missing=True,
    )

    _validate_non_negative_variance(
        daily_rv,
        column=daily_rv_col,
        name="HAR target validation input panel",
        allow_missing=True,
    )

    future_columns = [
        daily_rv.shift(-step)
        for step in range(1, horizon + 1)
    ]

    future_matrix = pd.concat(future_columns, axis=1)
    valid_count = future_matrix.notna().sum(axis=1)

    future_mean = future_matrix.mean(axis=1)
    future_mean = future_mean.where(valid_count == horizon)

    out[output_col] = annualization_periods * future_mean

    return out


def validate_phase3_target_matches_recomputed_target(
    panel: pd.DataFrame,
    config: HARConfig | None = None,
    *,
    recomputed_col: str = "_recomputed_rv_gk_22d_forward_ann_label",
) -> dict[str, Any]:
    """
    Validate that the existing Phase 3 forward RV label matches recomputation.

    This is a fail-loud check. It should be run before model training.

    Returns
    -------
    dict
        Validation summary.
    """
    if config is None:
        config = HARConfig()

    out = recompute_forward_target_for_validation(
        panel,
        daily_rv_col=config.daily_rv_col,
        horizon=config.forecast_horizon,
        annualization_periods=config.annualization_periods,
        output_col=recomputed_col,
    )

    _require_columns(
        out,
        [config.target_col, recomputed_col],
        name="HAR target validation panel",
    )

    existing = _coerce_numeric_column(
        out,
        config.target_col,
        name="HAR target validation panel",
        allow_missing=True,
    )
    recomputed = _coerce_numeric_column(
        out,
        recomputed_col,
        name="HAR target validation panel",
        allow_missing=True,
    )

    existing_available = existing.notna()
    recomputed_available = recomputed.notna()

    missing_mismatch_mask = existing_available != recomputed_available
    if missing_mismatch_mask.any():
        bad_count = int(missing_mismatch_mask.sum())
        bad_dates = (
            out.loc[missing_mismatch_mask, "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()[:10]
        )
        raise ValueError(
            "Phase 3 target availability does not match validation "
            f"recomputation. Bad rows: {bad_count}. First dates: {bad_dates}"
        )

    compare_mask = existing_available & recomputed_available

    if compare_mask.any():
        is_close = np.isclose(
            existing.loc[compare_mask].to_numpy(dtype=float),
            recomputed.loc[compare_mask].to_numpy(dtype=float),
            atol=config.target_validation_atol,
            rtol=config.target_validation_rtol,
            equal_nan=True,
        )

        if not bool(np.all(is_close)):
            bad_positions = np.where(~is_close)[0]
            compare_indices = out.index[compare_mask].to_numpy()
            bad_indices = compare_indices[bad_positions][:10]

            bad_rows = out.loc[
                bad_indices,
                ["date", config.target_col, recomputed_col],
            ].copy()

            bad_rows["date"] = bad_rows["date"].dt.strftime("%Y-%m-%d")

            raise ValueError(
                "Existing Phase 3 target does not match recomputed validation "
                "target. First mismatches: "
                f"{bad_rows.to_dict(orient='records')}"
            )

    return {
        "target_col": config.target_col,
        "recomputed_col": recomputed_col,
        "n_rows": int(len(out)),
        "n_compared": int(compare_mask.sum()),
        "n_missing_target": int(existing.isna().sum()),
        "n_missing_recomputed": int(recomputed.isna().sum()),
        "atol": config.target_validation_atol,
        "rtol": config.target_validation_rtol,
        "passed": True,
    }


def prepare_har_model_frame(
    panel: pd.DataFrame,
    config: HARConfig | None = None,
    *,
    validate_target: bool = True,
) -> pd.DataFrame:
    """
    Build the Phase 4 HAR modelling frame.

    Steps:
    1. Validate Phase 3 input panel.
    2. Add conservative lagged HAR features.
    3. Add target_start_date and target_end_date.
    4. Optionally validate Phase 3 target against recomputation.

    Returns
    -------
    pd.DataFrame
        Date-sorted frame containing original Phase 3 columns plus:
            har_rv_d_lag1_ann
            har_rv_w_lag1_ann
            har_rv_m_lag1_ann
            target_col
            target_start_date
            target_end_date
    """
    if config is None:
        config = HARConfig()

    out = validate_har_input_panel(panel, config=config)

    out = make_har_features(
        out,
        daily_rv_col=config.daily_rv_col,
        horizon=config.forecast_horizon,
        annualization_periods=config.annualization_periods,
    )

    out = add_forward_target_metadata(
        out,
        horizon=config.forecast_horizon,
        target_col=config.target_col,
    )

    if validate_target:
        validate_phase3_target_matches_recomputed_target(out, config=config)

    return out

def validate_primary_har_feature_cols(feature_cols: list[str]) -> None:
    """
    Validate primary HAR predictor columns.

    The primary HAR-RV model is intentionally strict. It must use exactly:
        har_rv_d_lag1_ann
        har_rv_w_lag1_ann
        har_rv_m_lag1_ann

    This rejects IV, backward VRP, forward labels, forecast outputs, and any
    extra variables.
    """
    assert_primary_har_features(list(feature_cols))


def _fit_feature_target_frame(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Build a clean modelling frame for HAR OLS.

    Rows are retained only when all features and target are finite.
    """
    validate_primary_har_feature_cols(feature_cols)

    required_cols = ["date", target_col] + feature_cols
    _require_columns(train_df, required_cols, name="HAR training frame")

    out = train_df[required_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in feature_cols + [target_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    finite_mask = out[feature_cols + [target_col]].notna().all(axis=1)

    for col in feature_cols + [target_col]:
        finite_mask &= np.isfinite(out[col])

    out = out.loc[finite_mask].sort_values("date").reset_index(drop=True)

    negative_target_mask = out[target_col] < 0
    if negative_target_mask.any():
        bad_count = int(negative_target_mask.sum())
        raise ValueError(
            f"HAR target column {target_col} contains "
            f"{bad_count} negative value(s)."
        )

    negative_feature_counts = {
        col: int((out[col] < 0).sum())
        for col in feature_cols
    }
    bad_feature_counts = {
        col: count for col, count in negative_feature_counts.items()
        if count > 0
    }

    if bad_feature_counts:
        raise ValueError(
            "HAR feature columns contain negative variance value(s): "
            f"{bad_feature_counts}"
        )

    return out


def fit_har_ols(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    hac_maxlags: int = 22,
):
    """
    Fit primary direct HAR-RV OLS with HAC/Newey-West robust covariance.

    Regression:
        y_t = beta_0
              + beta_d * har_rv_d_lag1_ann_t
              + beta_w * har_rv_w_lag1_ann_t
              + beta_m * har_rv_m_lag1_ann_t
              + error_t

    Parameters
    ----------
    train_df:
        Training rows already filtered by no-lookahead timing:
            target_end_date < forecast_date
    feature_cols:
        Must equal HAR_FEATURE_COLUMNS exactly.
    target_col:
        Primary target:
            rv_gk_22d_forward_ann_label
    hac_maxlags:
        HAC/Newey-West lag length.

    Returns
    -------
    statsmodels regression result
        Fitted OLS result using HAC robust covariance.
    """
    if hac_maxlags < 0:
        raise ValueError(f"hac_maxlags must be >= 0. Got: {hac_maxlags}")

    validate_primary_har_feature_cols(feature_cols)

    fit_df = _fit_feature_target_frame(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    min_required_rows = len(feature_cols) + 2
    if len(fit_df) < min_required_rows:
        raise ValueError(
            "Insufficient rows to fit HAR OLS after dropping missing values. "
            f"Need at least {min_required_rows}, got {len(fit_df)}."
        )

    y = fit_df[target_col].astype(float)
    x = fit_df[feature_cols].astype(float)
    x = sm.add_constant(x, has_constant="add")

    model = sm.OLS(y, x, missing="drop")
    result = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(hac_maxlags)},
    )

    return result


def predict_har(
    model_result,
    feature_df: pd.DataFrame,
    forecast_floor: float = 1.0e-8,
) -> pd.Series:
    """
    Predict HAR-RV forecast and apply a positive forecast floor.

    Parameters
    ----------
    model_result:
        Fitted statsmodels OLS result.
    feature_df:
        DataFrame containing HAR_FEATURE_COLUMNS.
    forecast_floor:
        Minimum allowed variance forecast.

    Returns
    -------
    pd.Series
        Positive forecast values indexed like feature_df.
    """
    if forecast_floor <= 0:
        raise ValueError(f"forecast_floor must be positive. Got: {forecast_floor}")

    feature_cols = list(HAR_FEATURE_COLUMNS)
    validate_primary_har_feature_cols(feature_cols)

    _require_columns(feature_df, feature_cols, name="HAR prediction frame")

    x = feature_df[feature_cols].copy()

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    missing_mask = x[feature_cols].isna().any(axis=1)
    _non_finite_arr = np.isfinite(x[feature_cols].to_numpy(dtype=float)).all(axis=1)
    # ensure a plain boolean ndarray to avoid pandas/typing overload issues
    non_finite_mask = pd.Series(~np.asarray(_non_finite_arr, dtype=bool), index=x.index)
    invalid_mask = missing_mask | non_finite_mask

    predictions = pd.Series(np.nan, index=feature_df.index, dtype=float)

    if invalid_mask.all():
        return predictions

    valid_x = x.loc[~invalid_mask, feature_cols].astype(float)

    negative_feature_mask = valid_x < 0
    if negative_feature_mask.any().any():
        bad_cols = negative_feature_mask.any(axis=0)
        bad_col_names = bad_cols[bad_cols].index.tolist()
        raise ValueError(
            "HAR prediction features contain negative variance value(s): "
            f"{bad_col_names}"
        )

    valid_x_const = sm.add_constant(valid_x, has_constant="add")
    raw_pred = model_result.predict(valid_x_const)

    clipped_pred = np.maximum(np.asarray(raw_pred, dtype=float), forecast_floor)

    predictions.loc[valid_x.index] = clipped_pred

    return predictions


def _safe_result_lookup(
    values: Any,
    key: str,
    default: float = float("nan"),
) -> float:
    """
    Read a statsmodels params/bse/tvalues entry by key.
    """
    try:
        if hasattr(values, "index"):
            if key in values.index:
                return float(values.loc[key])
            return float(default)

        names = getattr(values, "model", None)
        _ = names
        return float(default)
    except Exception:
        return float(default)


def _result_value_by_name(
    model_result,
    attr_name: str,
    param_name: str,
) -> float:
    """
    Extract a named value from model_result.params/bse/tvalues.

    Handles pandas Series and numpy arrays.
    """
    values = getattr(model_result, attr_name)

    if hasattr(values, "index"):
        if param_name in values.index:
            return float(values.loc[param_name])
        return float("nan")

    param_names = getattr(model_result.model, "exog_names", None)
    if param_names is None:
        return float("nan")

    if param_name not in param_names:
        return float("nan")

    idx = param_names.index(param_name)
    return float(values[idx])


def _format_date_for_row(value: Any) -> str | None:
    """
    Convert date-like value to YYYY-MM-DD string for coefficient rows.
    """
    if value is None or pd.isna(value):
        return None

    timestamp = pd.to_datetime(value, errors="coerce")

    if pd.isna(timestamp):
        return None

    return timestamp.strftime("%Y-%m-%d")


def extract_har_coefficient_row(
    model_result,
    *,
    forecast_date: Any,
    market: str,
    train_start_date: Any,
    train_end_date: Any,
    n_train: int,
    hac_maxlags: int,
    model_name: str = DEFAULT_HAR_MODEL_NAME,
) -> dict[str, object]:
    """
    Extract one coefficient-history row from a fitted HAR model.

    One row should be written for each forecast date with a successfully fitted
    expanding/rolling HAR model.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be positive. Got: {n_train}")

    param_names = [
        "const",
        "har_rv_d_lag1_ann",
        "har_rv_w_lag1_ann",
        "har_rv_m_lag1_ann",
    ]

    row: dict[str, object] = {
        "date": _format_date_for_row(forecast_date),
        "market": str(market).upper(),
        "model_name": model_name,
        "train_start_date": _format_date_for_row(train_start_date),
        "train_end_date": _format_date_for_row(train_end_date),
        "n_train": int(n_train),
        "hac_maxlags": int(hac_maxlags),
    }

    output_name_map = {
        "const": "const",
        "har_rv_d_lag1_ann": "har_rv_d_lag1_ann",
        "har_rv_w_lag1_ann": "har_rv_w_lag1_ann",
        "har_rv_m_lag1_ann": "har_rv_m_lag1_ann",
    }

    for param_name in param_names:
        suffix = output_name_map[param_name]

        row[f"coef_{suffix}"] = _result_value_by_name(
            model_result,
            "params",
            param_name,
        )
        row[f"se_{suffix}"] = _result_value_by_name(
            model_result,
            "bse",
            param_name,
        )
        row[f"t_{suffix}"] = _result_value_by_name(
            model_result,
            "tvalues",
            param_name,
        )

    return row


def make_batched_coefficient_row(
    *,
    beta: np.ndarray,
    forecast_date: Any,
    market: str,
    train_start_date: Any,
    train_end_date: Any,
    n_train: int,
    hac_maxlags: int,
    model_name: str = DEFAULT_HAR_MODEL_NAME,
    hac_available: bool = False,
) -> dict[str, object]:
    """
    Create a coefficient-history row directly from batched OLS results.
    
    Used by batched_har_forecast() to avoid calling extract_har_coefficient_row()
    with model_result=None, which is unsafe.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be positive. Got: {n_train}")
    
    return {
        "date": _format_date_for_row(forecast_date),
        "market": str(market).upper(),
        "model_name": model_name,
        "train_start_date": _format_date_for_row(train_start_date),
        "train_end_date": _format_date_for_row(train_end_date),
        "n_train": int(n_train),
        "coef_const": float(beta[0]),
        "coef_har_rv_d_lag1_ann": float(beta[1]),
        "coef_har_rv_w_lag1_ann": float(beta[2]),
        "coef_har_rv_m_lag1_ann": float(beta[3]),
        "se_const": np.nan,
        "se_har_rv_d_lag1_ann": np.nan,
        "se_har_rv_w_lag1_ann": np.nan,
        "se_har_rv_m_lag1_ann": np.nan,
        "t_const": np.nan,
        "t_har_rv_d_lag1_ann": np.nan,
        "t_har_rv_w_lag1_ann": np.nan,
        "t_har_rv_m_lag1_ann": np.nan,
        "hac_maxlags": int(hac_maxlags),
        "hac_available": bool(hac_available),
    }


def coefficient_rows_to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """
    Convert coefficient-history rows to a stable schema DataFrame.
    """
    columns = [
        "date",
        "market",
        "model_name",
        "train_start_date",
        "train_end_date",
        "n_train",
        "coef_const",
        "coef_har_rv_d_lag1_ann",
        "coef_har_rv_w_lag1_ann",
        "coef_har_rv_m_lag1_ann",
        "se_const",
        "se_har_rv_d_lag1_ann",
        "se_har_rv_w_lag1_ann",
        "se_har_rv_m_lag1_ann",
        "t_const",
        "t_har_rv_d_lag1_ann",
        "t_har_rv_w_lag1_ann",
        "t_har_rv_m_lag1_ann",
        "hac_maxlags",
        "hac_available",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)

    for col in columns:
        if col not in out.columns:
            out[col] = np.nan

    return out[columns].copy()

def _primary_naive_baseline_col(config: HARConfig) -> str:
    """
    Return the Phase 3 lagged 22-day RV baseline column.

    For the primary Garman-Klass setup, this is:
        rv_gk_22d_ann_lag1
    """
    if config.primary_estimator != "garman_klass":
        raise ValueError(
            "Only primary_estimator='garman_klass' is currently supported "
            f"for Phase 4. Got: {config.primary_estimator}"
        )

    return f"rv_gk_{config.forecast_horizon}d_ann_lag1"


def _forecast_panel_columns(config: HARConfig) -> list[str]:
    """
    Stable output schema for HAR forecast panels.
    """
    return [
        "date",
        "market",
        "target_col",
        "target_start_date",
        "target_end_date",
        config.target_col,
        "har_rv_d_lag1_ann",
        "har_rv_w_lag1_ann",
        "har_rv_m_lag1_ann",
        "naive_lagged_22d_rv_ann",
        "expanding_mean_forward_rv_baseline",
        "rolling_mean_forward_rv_baseline",
        "har_rv_gk_22d_forecast_ann",
        "har_model_name",
        "har_train_start_date",
        "har_train_end_date",
        "har_n_train",
        "har_oos_flag",
        "har_forecast_available",
        "har_blocked_reason",
    ]


def _audit_columns() -> list[str]:
    """
    Stable output schema for no-lookahead audit rows.
    """
    return [
        "market",
        "forecast_date",
        "n_candidate_train_rows",
        "n_valid_train_rows",
        "min_train_required",
        "max_training_row_date",
        "max_training_target_end_date",
        "forecast_date_minus_max_target_end_days",
        "rule_target_end_before_forecast_date",
        "forecast_available",
        "blocked_reason",
    ]


def _market_from_panel(df: pd.DataFrame) -> str:
    """
    Extract single market label from a panel.
    """
    if "market" not in df.columns:
        raise ValueError("Panel is missing required column: market")

    values = df["market"].dropna().astype(str).str.upper().unique().tolist()

    if len(values) != 1:
        raise ValueError(f"Panel must contain exactly one market. Found: {values}")

    return values[0]


def _to_date_string_or_none(value: Any) -> str | None:
    """
    Convert date-like value to YYYY-MM-DD string, or None.
    """
    return _format_date_for_row(value)


def _days_between_dates(later: Any, earlier: Any) -> int | None:
    """
    Calendar-day difference between two date-like values.
    """
    if later is None or earlier is None or pd.isna(later) or pd.isna(earlier):
        return None

    later_ts = pd.to_datetime(later, errors="coerce")
    earlier_ts = pd.to_datetime(earlier, errors="coerce")

    if pd.isna(later_ts) or pd.isna(earlier_ts):
        return None

    return int((later_ts.normalize() - earlier_ts.normalize()).days)


def _has_complete_current_features(
    row: pd.Series,
    feature_cols: list[str],
) -> bool:
    """
    Return True if current forecast row has usable HAR predictors.
    """
    for col in feature_cols:
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]

        if pd.isna(value):
            return False

        if not np.isfinite(value):
            return False

        if value < 0:
            return False

    return True


def get_available_training_rows(
    df: pd.DataFrame,
    forecast_date: Any,
    feature_cols: list[str],
    target_col: str,
    config: HARConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return candidate and valid training rows for a forecast date.

    Candidate rows:
        date < forecast_date

    Valid rows:
        target_end_date < forecast_date
        all HAR predictors available
        target available
        all values finite
        all variance values non-negative

    The strict no-lookahead condition is:
        target_end_date_s < forecast_date_t
    """
    validate_primary_har_feature_cols(feature_cols)

    required_cols = ["date", "target_end_date", target_col] + feature_cols
    _require_columns(df, required_cols, name="HAR available-training input")

    forecast_ts = pd.to_datetime(forecast_date, errors="coerce")
    if pd.isna(forecast_ts):
        raise ValueError(f"Invalid forecast_date: {forecast_date}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["target_end_date"] = pd.to_datetime(
        work["target_end_date"],
        errors="coerce",
    )

    candidate_mask = work["date"] < forecast_ts
    candidate_df = work.loc[candidate_mask].copy()

    numeric_cols = feature_cols + [target_col]
    for col in numeric_cols:
        candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")

    valid_mask = candidate_df["target_end_date"].notna()
    valid_mask &= candidate_df["target_end_date"] < forecast_ts
    valid_mask &= candidate_df[numeric_cols].notna().all(axis=1)

    for col in numeric_cols:
        valid_mask &= np.isfinite(candidate_df[col])

    valid_mask &= (candidate_df[numeric_cols] >= 0).all(axis=1)

    valid_df = candidate_df.loc[valid_mask].sort_values("date").reset_index(drop=True)

    return candidate_df.reset_index(drop=True), valid_df


def build_timing_safe_baselines(
    df: pd.DataFrame,
    forecast_date: Any,
    config: HARConfig,
    *,
    valid_train_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Build naive and historical-mean baselines for one forecast date.

    Baseline timing rule:
    Historical mean baselines use only rows where:
        target_end_date < forecast_date

    Returned columns:
        naive_lagged_22d_rv_ann
        expanding_mean_forward_rv_baseline
        rolling_mean_forward_rv_baseline
    """
    forecast_ts = pd.to_datetime(forecast_date, errors="coerce")
    if pd.isna(forecast_ts):
        raise ValueError(f"Invalid forecast_date: {forecast_date}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    current_rows = work.loc[work["date"] == forecast_ts]
    if len(current_rows) != 1:
        raise ValueError(
            "Expected exactly one current row for forecast date "
            f"{forecast_ts}. Found: {len(current_rows)}"
        )

    current_row = current_rows.iloc[0]

    naive_source_col = _primary_naive_baseline_col(config)

    naive_value = float("nan")
    if naive_source_col in current_row.index:
        naive_value_raw = pd.to_numeric(
            pd.Series([current_row[naive_source_col]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(naive_value_raw) and np.isfinite(naive_value_raw):
            naive_value = float(naive_value_raw)

    if valid_train_df is None:
        _, valid_train_df = get_available_training_rows(
            df=work,
            forecast_date=forecast_ts,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            config=config,
        )

    target_values = pd.to_numeric(
        valid_train_df[config.target_col],
        errors="coerce",
    ).dropna()

    target_values = target_values[np.isfinite(target_values)]
    target_values = target_values[target_values >= 0]

    expanding_mean = float("nan")
    rolling_mean = float("nan")

    if len(target_values) > 0:
        expanding_mean = float(target_values.mean())

        rolling_values = target_values.tail(config.rolling_train_window)
        if len(rolling_values) > 0:
            rolling_mean = float(rolling_values.mean())

    return {
        "naive_lagged_22d_rv_ann": naive_value,
        "expanding_mean_forward_rv_baseline": expanding_mean,
        "rolling_mean_forward_rv_baseline": rolling_mean,
    }


def build_no_lookahead_audit_row(
    *,
    market: str,
    forecast_date: Any,
    candidate_train_df: pd.DataFrame,
    valid_train_df: pd.DataFrame,
    min_train_required: int,
    forecast_available: bool,
    blocked_reason: str | None,
) -> dict[str, object]:
    """
    Build one row of the HAR no-lookahead audit table.

    The decisive audit condition is:
        max_training_target_end_date < forecast_date
    """
    forecast_ts = pd.to_datetime(forecast_date, errors="coerce")
    if pd.isna(forecast_ts):
        raise ValueError(f"Invalid forecast_date: {forecast_date}")

    max_training_row_date = None
    max_training_target_end_date = None
    rule_passed = False

    if len(valid_train_df) > 0:
        train_dates = pd.to_datetime(valid_train_df["date"], errors="coerce")
        train_target_end_dates = pd.to_datetime(
            valid_train_df["target_end_date"],
            errors="coerce",
        )

        max_training_row_date = train_dates.max()
        max_training_target_end_date = train_target_end_dates.max()

        if pd.notna(max_training_target_end_date):
            rule_passed = bool(max_training_target_end_date < forecast_ts)

    day_gap = _days_between_dates(forecast_ts, max_training_target_end_date)

    return {
        "market": str(market).upper(),
        "forecast_date": _to_date_string_or_none(forecast_ts),
        "n_candidate_train_rows": int(len(candidate_train_df)),
        "n_valid_train_rows": int(len(valid_train_df)),
        "min_train_required": int(min_train_required),
        "max_training_row_date": _to_date_string_or_none(max_training_row_date),
        "max_training_target_end_date": _to_date_string_or_none(
            max_training_target_end_date
        ),
        "forecast_date_minus_max_target_end_days": day_gap,
        "rule_target_end_before_forecast_date": bool(rule_passed),
        "forecast_available": bool(forecast_available),
        "blocked_reason": blocked_reason,
    }


def _initialize_forecast_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard HAR forecast columns to a model frame.
    """
    out = out.copy()

    out["naive_lagged_22d_rv_ann"] = np.nan
    out["expanding_mean_forward_rv_baseline"] = np.nan
    out["rolling_mean_forward_rv_baseline"] = np.nan
    out["har_rv_gk_22d_forecast_ann"] = np.nan
    out["har_model_name"] = DEFAULT_HAR_MODEL_NAME
    out["har_train_start_date"] = pd.NaT
    out["har_train_end_date"] = pd.NaT
    out["har_n_train"] = 0
    out["har_oos_flag"] = True
    out["har_forecast_available"] = False
    out["har_blocked_reason"] = pd.NA

    return out


def _finalize_forecast_panel(out: pd.DataFrame, config: HARConfig) -> pd.DataFrame:
    """
    Return forecast panel with stable schema and safe dtypes.
    """
    columns = _forecast_panel_columns(config)

    for col in columns:
        if col not in out.columns:
            out[col] = np.nan

    out = out[columns].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["target_start_date"] = pd.to_datetime(
        out["target_start_date"],
        errors="coerce",
    )
    out["target_end_date"] = pd.to_datetime(
        out["target_end_date"],
        errors="coerce",
    )
    out["har_train_start_date"] = pd.to_datetime(
        out["har_train_start_date"],
        errors="coerce",
    )
    out["har_train_end_date"] = pd.to_datetime(
        out["har_train_end_date"],
        errors="coerce",
    )

    out["har_forecast_available"] = out["har_forecast_available"].astype(bool)
    out["har_oos_flag"] = out["har_oos_flag"].astype(bool)
    out["har_n_train"] = pd.to_numeric(
        out["har_n_train"],
        errors="coerce",
    ).fillna(0).astype(int)

    return out


def _current_row_has_target_metadata(row: pd.Series) -> bool:
    """
    Return True if current row has target_start_date and target_end_date metadata.
    """
    target_start_date = row.get("target_start_date")
    target_end_date = row.get("target_end_date")

    start_date = (
        pd.to_datetime(target_start_date, errors="coerce")
        if target_start_date is not None
        else pd.NaT
    )
    end_date = (
        pd.to_datetime(target_end_date, errors="coerce")
        if target_end_date is not None
        else pd.NaT
    )

    return bool(pd.notna(start_date) and pd.notna(end_date))


def _apply_successful_forecast_to_row(
    out: pd.DataFrame,
    row_index: int,
    *,
    forecast_value: float,
    train_df: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Mutate one output row with a successful HAR forecast.
    """
    out.loc[row_index, "har_rv_gk_22d_forecast_ann"] = float(forecast_value)
    out.loc[row_index, "har_model_name"] = model_name
    out.loc[row_index, "har_train_start_date"] = train_df["date"].min()
    out.loc[row_index, "har_train_end_date"] = train_df["date"].max()
    out.loc[row_index, "har_n_train"] = int(len(train_df))
    out.loc[row_index, "har_oos_flag"] = True
    out.loc[row_index, "har_forecast_available"] = True
    out.loc[row_index, "har_blocked_reason"] = pd.NA


def _apply_blocked_forecast_to_row(
    out: pd.DataFrame,
    row_index: int,
    *,
    blocked_reason: str,
    train_df: pd.DataFrame | None = None,
) -> None:
    """
    Mutate one output row with a blocked HAR forecast.
    """
    out.loc[row_index, "har_rv_gk_22d_forecast_ann"] = np.nan
    out.loc[row_index, "har_oos_flag"] = True
    out.loc[row_index, "har_forecast_available"] = False
    out.loc[row_index, "har_blocked_reason"] = blocked_reason

    if train_df is not None and len(train_df) > 0:
        out.loc[row_index, "har_train_start_date"] = train_df["date"].min()
        out.loc[row_index, "har_train_end_date"] = train_df["date"].max()
        out.loc[row_index, "har_n_train"] = int(len(train_df))


def build_har_design_arrays(
    out: pd.DataFrame,
    config: HARConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build batched design arrays for HAR model from an initialized forecast panel.

    Returns:
        dates, target_end_dates, market, X, y,
        current_feature_valid, train_row_valid, target_available
    """
    required = ["date", "target_start_date", "target_end_date", config.target_col]
    required += config.feature_cols
    _require_columns(out, required, name="HAR design input")

    n = len(out)

    dates = pd.to_datetime(out["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    target_end_dates = pd.to_datetime(out["target_end_date"], errors="coerce").to_numpy(
        dtype="datetime64[ns]",
    )

    market = _market_from_panel(out)

    p = 1 + len(config.feature_cols)
    X = np.full((n, p), np.nan, dtype=float)
    X[:, 0] = 1.0

    for j, col in enumerate(config.feature_cols, start=1):
        X[:, j] = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)

    y = pd.to_numeric(out[config.target_col], errors="coerce").to_numpy(dtype=float)

    # current features valid: current row has all features finite and non-negative
    current_feature_valid = np.all(np.isfinite(X[:, 1:]), axis=1) & (X[:, 1:] >= 0).all(axis=1)

    # target available
    target_available = np.isfinite(y) & (y >= 0)

    # train_row_valid requires: current features complete, target available, target_end_date finite
    target_end_finite = ~pd.isna(pd.to_datetime(out["target_end_date"], errors="coerce")).to_numpy()

    # also require target_start_date present
    target_start_finite = ~pd.isna(pd.to_datetime(out["target_start_date"], errors="coerce")).to_numpy()

    train_row_valid = (
        current_feature_valid
        & target_available
        & target_end_finite
        & target_start_finite
    )

    return (
        dates,
        target_end_dates,
        market,
        X,
        y,
        current_feature_valid,
        train_row_valid,
        target_available,
    )


def compute_training_bounds(
    dates: np.ndarray,
    target_end_dates: np.ndarray,
    train_row_valid: np.ndarray,
    config: HARConfig,
    mode: str = "expanding",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute start/end bounds (in positions of eligible training rows) for each forecast row.

    Returns:
        eligible_indices_sorted: indices into original rows of eligible training rows
        window_starts: start positions (inclusive) in eligible_indices_sorted for each forecast row
        window_ends: end positions (exclusive)
    """
    # eligible indices sorted by target_end_dates
    eligible_idx = np.flatnonzero(train_row_valid)

    if len(eligible_idx) == 0:
        n = len(dates)
        return eligible_idx, np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    eligible_target_ends = target_end_dates[eligible_idx]

    # Sort eligible by target_end_date
    order = np.argsort(eligible_target_ends)
    eligible_sorted = eligible_idx[order]
    eligible_target_sorted = eligible_target_ends[order]

    # For searchsorted we need numpy datetime64 array
    # compute for each forecast date the insertion point
    m = len(eligible_sorted)
    n_dates = len(dates)
    window_starts = np.zeros(n_dates, dtype=int)
    window_ends = np.zeros(n_dates, dtype=int)

    # Use searchsorted with side='left' to enforce target_end_date < date
    positions = np.searchsorted(eligible_target_sorted, dates, side="left")

    for i in range(n_dates):
        end_pos = int(positions[i])
        if mode == "expanding":
            start_pos = 0
        else:
            start_pos = max(0, end_pos - config.rolling_train_window)

        window_starts[i] = start_pos
        window_ends[i] = end_pos

    return eligible_sorted, window_starts, window_ends


def build_prefix_sufficient_statistics(X_valid: np.ndarray, y_valid: np.ndarray):
    """
    Build prefix sums for XtX, Xty, y and count over valid training rows.

    X_valid: (m, p)
    y_valid: (m,)

    Returns prefix_xtx (m+1,p,p), prefix_xty (m+1,p), prefix_y (m+1,), prefix_count (m+1,)
    """
    m, p = X_valid.shape

    # compute outer products per row: (m, p, p)
    outer = X_valid[:, :, None] * X_valid[:, None, :]
    prefix_xtx = np.zeros((m + 1, p, p), dtype=float)
    if m > 0:
        prefix_xtx[1:] = np.cumsum(outer, axis=0)

    xy = X_valid * y_valid[:, None]
    prefix_xty = np.zeros((m + 1, p), dtype=float)
    if m > 0:
        prefix_xty[1:] = np.cumsum(xy, axis=0)

    prefix_y = np.zeros((m + 1,), dtype=float)
    if m > 0:
        prefix_y[1:] = np.cumsum(y_valid, axis=0)

    prefix_count = np.arange(m + 1, dtype=int)

    return prefix_xtx, prefix_xty, prefix_y, prefix_count


def batched_solve_ols_numpy(
    XtX_batch: np.ndarray,
    Xty_batch: np.ndarray,
    *,
    matrix_singularity_policy: str = "pinv",
    condition_number_threshold: float = 1.0e12,
):
    """
    Solve small OLS systems in batch using NumPy.

    XtX_batch: (k, p, p)
    Xty_batch: (k, p)

    Returns beta_batch (k, p) and status array (k,) True if solved.
    """
    k, p, _ = XtX_batch.shape
    betas = np.full((k, p), np.nan, dtype=float)
    status = np.zeros(k, dtype=bool)

    for i in range(k):
        A = XtX_batch[i]
        b = Xty_batch[i]
        try:
            cond = np.linalg.cond(A)
        except Exception:
            cond = float("inf")

        if not np.isfinite(cond) or cond > condition_number_threshold:
            if matrix_singularity_policy == "pinv":
                try:
                    pinv = np.linalg.pinv(A)
                    betas[i] = pinv.dot(b)
                    status[i] = True
                except Exception:
                    status[i] = False
            else:
                status[i] = False
        else:
            try:
                betas[i] = np.linalg.solve(A, b)
                status[i] = True
            except Exception:
                if matrix_singularity_policy == "pinv":
                    try:
                        pinv = np.linalg.pinv(A)
                        betas[i] = pinv.dot(b)
                        status[i] = True
                    except Exception:
                        status[i] = False
                else:
                    status[i] = False

    return betas, status


def batched_solve_ols_torch(
    XtX_batch: np.ndarray,
    Xty_batch: np.ndarray,
    *,
    torch_device: str = "cpu",
    torch_dtype: str = "float64",
    matrix_singularity_policy: str = "pinv",
    condition_number_threshold: float = 1.0e12,
):
    """
    Solve batch OLS using torch if available. Falls back to numpy for failures.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - environment dependent
        return batched_solve_ols_numpy(
            XtX_batch, Xty_batch, matrix_singularity_policy=matrix_singularity_policy,
            condition_number_threshold=condition_number_threshold,
        )

    dtype = torch.float64 if torch_dtype == "float64" else torch.float32
    device = torch.device("cuda" if torch_device == "cuda" else "cpu")

    A = torch.from_numpy(XtX_batch).to(device=device, dtype=dtype)
    b = torch.from_numpy(Xty_batch).to(device=device, dtype=dtype)

    k = A.shape[0]
    p = A.shape[1]

    betas = torch.full((k, p), float("nan"), device=device, dtype=dtype)
    status = np.zeros(k, dtype=bool)

    for i in range(k):
        Ai = A[i]
        bi = b[i]
        try:
            cond = torch.linalg.cond(Ai).item()
        except Exception:
            cond = float("inf")

        if not np.isfinite(cond) or cond > condition_number_threshold:
            if matrix_singularity_policy == "pinv":
                try:
                    pinv = torch.linalg.pinv(Ai)
                    betas[i] = pinv.matmul(bi)
                    status[i] = True
                except Exception:
                    status[i] = False
            else:
                status[i] = False
        else:
            try:
                sol = torch.linalg.solve(Ai, bi)
                betas[i] = sol
                status[i] = True
            except Exception:
                if matrix_singularity_policy == "pinv":
                    try:
                        pinv = torch.linalg.pinv(Ai)
                        betas[i] = pinv.matmul(bi)
                        status[i] = True
                    except Exception:
                        status[i] = False
                else:
                    status[i] = False

    return betas.cpu().numpy(), status


def _batched_audit_row_from_arrays(
    market: str,
    forecast_date: Any,
    forecast_idx: int,
    dates: np.ndarray,
    target_end_dates: np.ndarray,
    train_row_valid: np.ndarray,
    eligible_sorted: np.ndarray,
    window_start: int,
    window_end: int,
    min_train_required: int,
    forecast_available: bool,
    blocked_reason: str | None,
) -> dict[str, object]:
    """
    Build a no-lookahead audit row directly from arrays (avoids DataFrame construction).
    
    Uses the actual training window defined by eligible_sorted[window_start:window_end]
    to compute training statistics, making this accurate for both expanding and rolling modes.
    """
    forecast_ts = pd.to_datetime(forecast_date, errors="coerce")
    if pd.isna(forecast_ts):
        raise ValueError(f"Invalid forecast_date: {forecast_date}")

    # Count candidate rows: date < forecast_date
    candidate_mask = dates < np.datetime64(forecast_ts.isoformat(), "ns")
    n_candidate = int(candidate_mask.sum())

    # Training stats from actual window: eligible_sorted[window_start:window_end]
    train_indices = eligible_sorted[window_start:window_end]
    n_valid = len(train_indices)
    
    max_training_row_date = None
    max_training_target_end_date = None
    rule_passed = False

    if n_valid > 0:
        train_dates = dates[train_indices]
        train_target_end_dates = target_end_dates[train_indices]
        
        max_training_row_date = pd.Timestamp(train_dates.max())
        max_training_target_end_date = pd.Timestamp(train_target_end_dates.max())

        if pd.notna(max_training_target_end_date):
            rule_passed = bool(max_training_target_end_date < forecast_ts)

    day_gap = _days_between_dates(forecast_ts, max_training_target_end_date)

    return {
        "market": str(market).upper(),
        "forecast_date": _to_date_string_or_none(forecast_ts),
        "n_candidate_train_rows": int(n_candidate),
        "n_valid_train_rows": int(n_valid),
        "min_train_required": int(min_train_required),
        "max_training_row_date": _to_date_string_or_none(max_training_row_date),
        "max_training_target_end_date": _to_date_string_or_none(max_training_target_end_date),
        "forecast_date_minus_max_target_end_days": day_gap,
        "rule_target_end_before_forecast_date": bool(rule_passed),
        "forecast_available": bool(forecast_available),
        "blocked_reason": blocked_reason,
    }


def batched_har_forecast(
    out: pd.DataFrame,
    config: HARConfig,
    *,
    mode: str = "expanding",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Batched HAR forecast implementation using closed-form OLS.

    Returns forecast_panel, coefficient_frame, audit_frame
    """
    # Build design arrays
    (
        dates,
        target_end_dates,
        market,
        X,
        y,
        current_feature_valid,
        train_row_valid,
        target_available,
    ) = build_har_design_arrays(out, config)

    eligible_sorted, window_starts, window_ends = compute_training_bounds(
        dates, target_end_dates, train_row_valid, config, mode=mode
    )

    # Prepare ordered valid training X/y
    if len(eligible_sorted) > 0:
        X_valid = X[eligible_sorted]
        y_valid = y[eligible_sorted]
    else:
        X_valid = np.zeros((0, X.shape[1]), dtype=float)
        y_valid = np.zeros((0,), dtype=float)

    prefix_xtx, prefix_xty, prefix_y, prefix_count = build_prefix_sufficient_statistics(
        X_valid, y_valid
    )

    n = len(out)
    p = X.shape[1]

    coefficient_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    # prepare outputs
    forecast_vals = np.full(n, np.nan, dtype=float)
    baseline_expanding = np.full(n, np.nan, dtype=float)
    baseline_rolling = np.full(n, np.nan, dtype=float)
    naive_baseline = np.full(n, np.nan, dtype=float)
    har_model_names = [DEFAULT_HAR_MODEL_NAME] * n
    har_train_start = [pd.NaT] * n
    har_train_end = [pd.NaT] * n
    har_n_train = np.zeros(n, dtype=int)
    har_oos_flag = np.zeros(n, dtype=bool)
    har_forecast_available = np.zeros(n, dtype=bool)
    # use None for missing blocked reasons to avoid pandas' pd.NA boolean ambiguity
    har_blocked_reason = [None] * n

    # Precompute naive baseline column if present
    naive_col = _primary_naive_baseline_col(config)
    if naive_col in out.columns:
        # ensure a writable numpy array (pandas may return a read-only view)
        naive_baseline = pd.to_numeric(out[naive_col], errors="coerce").to_numpy(dtype=float).copy()

    # Determine which rows need solving
    solve_indices: list[int] = []
    solve_XtX = []
    solve_Xty = []
    solve_X_current = []
    solve_train_bounds = []

    for i in range(n):
        # Baselines from prefix sums
        start = int(window_starts[i])
        end = int(window_ends[i])
        count = int(prefix_count[end] - prefix_count[start])
        mean_y = float("nan")
        if count > 0:
            mean_y = float((prefix_y[end] - prefix_y[start]) / count)
        baseline_expanding[i] = mean_y
        # rolling mean
        if count > 0:
            # compute rolling mean by taking last min(rolling_window, count)
            last_k = min(config.rolling_train_window, count)
            if last_k > 0:
                # indices in eligible_sorted: end-last_k ... end-1
                last_start = max(0, end - last_k)
                rolling_count = int(prefix_count[end] - prefix_count[last_start])
                if rolling_count > 0:
                    rolling_mean = float((prefix_y[end] - prefix_y[last_start]) / rolling_count)
                else:
                    rolling_mean = float("nan")
            else:
                rolling_mean = float("nan")
        else:
            rolling_mean = float("nan")

        baseline_rolling[i] = rolling_mean
        naive_baseline_val = float(naive_baseline[i]) if not np.isnan(naive_baseline[i]) else float("nan")
        naive_baseline[i] = naive_baseline_val

        blocked_reason = None
        forecast_avail = False

        # metadata checks
        if not _current_row_has_target_metadata(out.iloc[i]):
            blocked_reason = "missing_target_metadata"
        elif count < config.min_train_observations:
            blocked_reason = "insufficient_training_history"
        elif not current_feature_valid[i]:
            blocked_reason = "missing_har_features"
        # record blocked reason for audit parity with statsmodels path
        if blocked_reason is not None:
            har_blocked_reason[i] = blocked_reason
        else:
            # prepare XtX and Xty for this train window
            XtX = prefix_xtx[end] - prefix_xtx[start]
            Xty = prefix_xty[end] - prefix_xty[start]
            # push to batch
            solve_indices.append(i)
            solve_XtX.append(XtX)
            solve_Xty.append(Xty)
            solve_X_current.append(X[i])
            solve_train_bounds.append((start, end))

        # Build audit row from arrays (not empty DataFrames)
        audit_rows.append(
            _batched_audit_row_from_arrays(
                market=market,
                forecast_date=out.loc[i, "date"],
                forecast_idx=i,
                dates=dates,
                target_end_dates=target_end_dates,
                train_row_valid=train_row_valid,
                eligible_sorted=eligible_sorted,
                window_start=int(start),
                window_end=int(end),
                min_train_required=config.min_train_observations,
                forecast_available=False,
                blocked_reason=blocked_reason,
            )
        )

    # Solve batch
    if len(solve_indices) > 0:
        XtX_batch = np.stack(solve_XtX, axis=0)
        Xty_batch = np.stack(solve_Xty, axis=0)

        backend = resolve_compute_backend(config)
        betas = None
        status = None

        if backend == "torch_batched":
            td = resolve_torch_device(config)
            tdt = resolve_torch_dtype(config)
            betas, status = batched_solve_ols_torch(
                XtX_batch,
                Xty_batch,
                torch_device=td,
                torch_dtype=tdt,
                matrix_singularity_policy=config.matrix_singularity_policy,
                condition_number_threshold=config.condition_number_threshold,
            )
        else:
            betas, status = batched_solve_ols_numpy(
                XtX_batch,
                Xty_batch,
                matrix_singularity_policy=config.matrix_singularity_policy,
                condition_number_threshold=config.condition_number_threshold,
            )

        # apply predictions
        for idx, solved in enumerate(status):
            i = solve_indices[idx]
            start, end = solve_train_bounds[idx]
            if not solved:
                har_blocked_reason[i] = "model_fit_failed"
                har_forecast_available[i] = False
                har_oos_flag[i] = True
                continue

            beta = betas[idx]
            xi = solve_X_current[idx]
            pred = float(np.dot(xi, beta))
            if not np.isfinite(pred):
                har_blocked_reason[i] = "prediction_failed"
                har_forecast_available[i] = False
                har_oos_flag[i] = True
                continue

            pred = max(pred, config.forecast_floor)
            forecast_vals[i] = pred
            har_forecast_available[i] = True
            har_oos_flag[i] = True
            har_n_train[i] = int(prefix_count[end] - prefix_count[start])
            if har_n_train[i] > 0:
                # set train start/end dates from eligible_sorted indices
                train_indices = eligible_sorted[start:end]
                har_train_start[i] = pd.to_datetime(out.loc[train_indices, "date"]).min()
                har_train_end[i] = pd.to_datetime(out.loc[train_indices, "date"]).max()

            # coefficient row from batched OLS solution
            coef_row = make_batched_coefficient_row(
                beta=beta,
                forecast_date=out.loc[i, "date"],
                market=market,
                train_start_date=har_train_start[i],
                train_end_date=har_train_end[i],
                n_train=har_n_train[i],
                hac_maxlags=config.hac_maxlags,
                model_name=DEFAULT_HAR_MODEL_NAME,
                hac_available=False,
            )
            coefficient_rows.append(coef_row)

    # Update audit rows with forecast_available status
    for i in range(len(audit_rows)):
        audit_rows[i]["forecast_available"] = bool(har_forecast_available[i])
        if har_blocked_reason[i] is not None:
            audit_rows[i]["blocked_reason"] = har_blocked_reason[i]

    # Construct output frames
    out2 = out.copy()
    out2["har_rv_gk_22d_forecast_ann"] = forecast_vals
    out2["expanding_mean_forward_rv_baseline"] = baseline_expanding
    out2["rolling_mean_forward_rv_baseline"] = baseline_rolling
    out2["naive_lagged_22d_rv_ann"] = naive_baseline

    out2["har_train_start_date"] = har_train_start
    out2["har_train_end_date"] = har_train_end
    out2["har_n_train"] = har_n_train
    out2["har_oos_flag"] = har_oos_flag
    out2["har_forecast_available"] = har_forecast_available
    out2["har_blocked_reason"] = har_blocked_reason

    coefficient_frame = coefficient_rows_to_frame(coefficient_rows) if coefficient_rows else pd.DataFrame()
    audit_frame = pd.DataFrame(audit_rows, columns=_audit_columns()) if audit_rows else pd.DataFrame()

    return _finalize_forecast_panel(out2, config), coefficient_frame, audit_frame


def _walk_forward_har_forecast(
    df: pd.DataFrame,
    config: HARConfig,
    *,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shared expanding/rolling HAR walk-forward engine.

    Returns
    -------
    tuple
        forecast_panel
        coefficient_history
        no_lookahead_audit
    """
    if mode not in {"expanding", "rolling"}:
        raise ValueError(f"mode must be expanding or rolling. Got: {mode}")

    assert_primary_har_features(config.feature_cols)

    work = prepare_har_model_frame(
        df,
        config=config,
        validate_target=True,
    )

    market = _market_from_panel(work)

    out = _initialize_forecast_columns(work)

    # Fast-path: use batched closed-form OLS backends when configured
    backend = resolve_compute_backend(config)
    if backend in {"cpu_numpy_batched", "torch_batched"}:
        return batched_har_forecast(out, config, mode=mode)

    coefficient_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for row_index, row in out.iterrows():
        # row_index can be a label (e.g. Timestamp) which mypy/type-checkers
        # may consider Hashable. Convert to an integer positional index safely.
        if isinstance(row_index, (int, np.integer)):
            row_index = int(row_index)
        else:
            # get_loc may return int, slice, or boolean/integer ndarray
            # depending on index uniqueness. Convert to a single position.
            loc = out.index.get_loc(row_index)
            if isinstance(loc, (int, np.integer)):
                row_index = int(loc)
            elif isinstance(loc, slice):
                if loc.start is None:
                    raise KeyError(f"Could not determine integer position for index label: {row_index}")
                row_index = int(loc.start)
            elif isinstance(loc, np.ndarray):
                positions = np.flatnonzero(loc) if loc.dtype == bool else loc
                if len(positions) == 0:
                    raise KeyError(f"Could not determine integer position for index label: {row_index}")
                row_index = int(positions[0])
            else:
                row_index = int(loc)
        forecast_date = row["date"]

        candidate_train_df, valid_train_df = get_available_training_rows(
            df=out,
            forecast_date=forecast_date,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            config=config,
        )

        if mode == "rolling" and len(valid_train_df) > 0:
            train_df = valid_train_df.tail(config.rolling_train_window).copy()
        else:
            train_df = valid_train_df.copy()

        baselines = build_timing_safe_baselines(
            out,
            forecast_date,
            config,
            valid_train_df=valid_train_df,
        )

        for baseline_col, baseline_value in baselines.items():
            out.loc[row_index, baseline_col] = baseline_value

        blocked_reason: str | None = None
        forecast_available = False

        if not _current_row_has_target_metadata(row):
            blocked_reason = "missing_target_metadata"
            _apply_blocked_forecast_to_row(
                out,
                row_index,
                blocked_reason=blocked_reason,
                train_df=train_df,
            )

        elif len(train_df) < config.min_train_observations:
            blocked_reason = "insufficient_training_history"
            _apply_blocked_forecast_to_row(
                out,
                row_index,
                blocked_reason=blocked_reason,
                train_df=train_df,
            )

        elif not _has_complete_current_features(row, config.feature_cols):
            blocked_reason = "missing_har_features"
            _apply_blocked_forecast_to_row(
                out,
                row_index,
                blocked_reason=blocked_reason,
                train_df=train_df,
            )

        else:
            try:
                model_result = fit_har_ols(
                    train_df=train_df,
                    feature_cols=config.feature_cols,
                    target_col=config.target_col,
                    hac_maxlags=config.hac_maxlags,
                )

                forecast_series = predict_har(
                    model_result,
                    out.loc[[row_index], :],
                    forecast_floor=config.forecast_floor,
                )

                forecast_value = forecast_series.loc[row_index]

                if pd.isna(forecast_value) or not np.isfinite(forecast_value):
                    blocked_reason = "prediction_failed"
                    _apply_blocked_forecast_to_row(
                        out,
                        row_index,
                        blocked_reason=blocked_reason,
                        train_df=train_df,
                    )
                else:
                    _apply_successful_forecast_to_row(
                        out,
                        row_index,
                        forecast_value=float(forecast_value),
                        train_df=train_df,
                        model_name=DEFAULT_HAR_MODEL_NAME,
                    )

                    forecast_available = True

                    coefficient_rows.append(
                        extract_har_coefficient_row(
                            model_result,
                            forecast_date=forecast_date,
                            market=market,
                            train_start_date=train_df["date"].min(),
                            train_end_date=train_df["date"].max(),
                            n_train=len(train_df),
                            hac_maxlags=config.hac_maxlags,
                            model_name=DEFAULT_HAR_MODEL_NAME,
                        )
                    )

            except Exception:
                blocked_reason = "model_fit_failed"
                _apply_blocked_forecast_to_row(
                    out,
                    row_index,
                    blocked_reason=blocked_reason,
                    train_df=train_df,
                )

        audit_rows.append(
            build_no_lookahead_audit_row(
                market=market,
                forecast_date=forecast_date,
                candidate_train_df=candidate_train_df,
                valid_train_df=train_df,
                min_train_required=config.min_train_observations,
                forecast_available=forecast_available,
                blocked_reason=blocked_reason,
            )
        )

    forecast_panel = _finalize_forecast_panel(out, config)
    coefficient_frame = coefficient_rows_to_frame(coefficient_rows)

    audit_frame = pd.DataFrame(audit_rows)
    for col in _audit_columns():
        if col not in audit_frame.columns:
            audit_frame[col] = np.nan
    audit_frame = audit_frame[_audit_columns()].copy()

    return forecast_panel, coefficient_frame, audit_frame


def expanding_window_har_forecast(
    df: pd.DataFrame,
    config: HARConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run conservative expanding-window HAR forecast.

    For forecast date t:
        train only on rows s where target_end_date_s < t
    """
    return _walk_forward_har_forecast(
        df=df,
        config=config,
        mode="expanding",
    )


def rolling_window_har_forecast(
    df: pd.DataFrame,
    config: HARConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run conservative rolling-window HAR forecast.

    For forecast date t:
        train only on rows s where target_end_date_s < t

    Then keep only the most recent config.rolling_train_window valid rows.
    """
    return _walk_forward_har_forecast(
        df=df,
        config=config,
        mode="rolling",
    )


def _resolve_project_path(path_value: str | Path, *, project_root: Path | None = None) -> Path:
    """
    Resolve config path values relative to project root/current working directory.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    if project_root is not None:
        return project_root / path

    return Path.cwd() / path


def build_har_forecast_panel(
    market: str,
    config: HARConfig,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build one market's HAR forecast panel from the Phase 3 VRP parquet.

    This function reads:
        data/processed/{market_lower}_vrp.parquet

    It does not write files. File writing is handled by scripts/train_har.py.
    """
    market_key = str(market).upper()

    if market_key not in config.markets:
        raise ValueError(
            f"Market {market_key} is not listed in config.markets: {config.markets}"
        )

    if market_key not in config.input_paths:
        raise ValueError(f"Missing input path for market {market_key}")

    root = Path(project_root) if project_root is not None else None
    input_path = _resolve_project_path(config.input_paths[market_key], project_root=root)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 VRP input for {market_key}: {input_path}")

    panel = pd.read_parquet(input_path)

    panel_market = _market_from_panel(panel)
    if panel_market != market_key:
        raise ValueError(
            f"Input panel market mismatch. Expected {market_key}, found {panel_market}"
        )

    if config.oos_mode == "expanding":
        return expanding_window_har_forecast(panel, config)

    if config.oos_mode == "rolling":
        return rolling_window_har_forecast(panel, config)

    raise ValueError(
        f"Unsupported config.oos_mode: {config.oos_mode}. "
        "Expected expanding or rolling."
    )


def build_all_har_forecasts(
    config: HARConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build HAR forecasts for all markets in config.markets.

    Returns
    -------
    dict
        {
            "US": {
                "forecast": forecast_panel,
                "coefficients": coefficient_frame,
                "audit": audit_frame,
            },
            "INDIA": {...},
        }
    """
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for market in config.markets:
        forecast_panel, coefficient_frame, audit_frame = build_har_forecast_panel(
            market=market,
            config=config,
            project_root=project_root,
        )

        results[str(market).upper()] = {
            "forecast": forecast_panel,
            "coefficients": coefficient_frame,
            "audit": audit_frame,
        }

    return results