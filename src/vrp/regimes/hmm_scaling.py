"""
Train-only scaling for Phase 6 Gaussian HMM features.

This module enforces the Phase 6 scaling rule:

    Fit scaler only on the HMM training window.
    Apply that fitted scaler to train/test/full eligible rows.
    Appending future rows must not change earlier scaled values when the same
    fitted scaler is reused.

No model fitting happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vrp.regimes.hmm_features import HMMFeaturePanel, get_feature_matrix
from vrp.regimes.hmm_registry import get_hmm_feature_output_name_map


DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_MIN_TRAIN_OBSERVATIONS = 750
DEFAULT_MIN_TEST_OBSERVATIONS = 250
DEFAULT_DATE_COL = "date"


@dataclass(frozen=True)
class HMMScalerMetadata:
    """Metadata needed to audit train-only scaling."""

    scaler_method: str
    market: str
    feature_set: str
    feature_cols: tuple[str, ...]
    scaled_feature_cols: tuple[str, ...]
    n_observations: int
    n_train: int
    n_test: int
    train_fraction: float
    scaler_fit_start_date: str
    scaler_fit_end_date: str
    scaler_feature_means: Mapping[str, float]
    scaler_feature_scales: Mapping[str, float]
    scaler_input_hash: str
    train_window_hash: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""
        return {
            "scaler_method": self.scaler_method,
            "market": self.market,
            "feature_set": self.feature_set,
            "feature_cols": list(self.feature_cols),
            "scaled_feature_cols": list(self.scaled_feature_cols),
            "n_observations": self.n_observations,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": self.train_fraction,
            "scaler_fit_start_date": self.scaler_fit_start_date,
            "scaler_fit_end_date": self.scaler_fit_end_date,
            "scaler_feature_means": dict(self.scaler_feature_means),
            "scaler_feature_scales": dict(self.scaler_feature_scales),
            "scaler_input_hash": self.scaler_input_hash,
            "train_window_hash": self.train_window_hash,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class HMMScaledFeaturePanel:
    """
    Scaled HMM panel after train-only scaler fitting.

    Attributes
    ----------
    market, feature_set:
        Market/model identifiers.
    feature_cols:
        Raw HMM feature columns.
    scaled_feature_cols:
        Output scaled feature columns.
    scaled_panel:
        Eligible panel with scaled feature columns appended.
    scaler:
        Fitted StandardScaler.
    metadata:
        Train-only scaler metadata.
    train_indices:
        Positional indices into scaled_panel used for scaler fitting.
    test_indices:
        Positional indices into scaled_panel used as out-of-sample rows.
    X_raw:
        Raw eligible feature matrix.
    X_scaled:
        Scaled eligible feature matrix.
    """

    market: str
    feature_set: str
    feature_cols: tuple[str, ...]
    scaled_feature_cols: tuple[str, ...]
    scaled_panel: pd.DataFrame
    scaler: StandardScaler
    metadata: HMMScalerMetadata
    train_indices: np.ndarray
    test_indices: np.ndarray
    X_raw: np.ndarray
    X_scaled: np.ndarray

    @property
    def X_train_scaled(self) -> np.ndarray:
        return self.X_scaled[self.train_indices]

    @property
    def X_test_scaled(self) -> np.ndarray:
        return self.X_scaled[self.test_indices]


@dataclass(frozen=True)
class HMMTransformedFeaturePanel:
    """
    Panel transformed using an already-fitted scaler.

    Used for prefix-invariance / future-append audits.
    """

    market: str
    feature_set: str
    feature_cols: tuple[str, ...]
    scaled_feature_cols: tuple[str, ...]
    scaled_panel: pd.DataFrame
    X_raw: np.ndarray
    X_scaled: np.ndarray
    source_scaler_metadata: Mapping[str, Any]


def _utc_now_iso() -> str:
    """Return UTC timestamp string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_hash(payload: Mapping[str, Any]) -> str:
    """Stable SHA256 hash for JSON-serialisable metadata."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_numpy_array(arr: np.ndarray, *, extra: Mapping[str, Any] | None = None) -> str:
    """Stable SHA256 hash of a numeric numpy array plus optional metadata."""
    numeric = np.asarray(arr, dtype=float)
    contiguous = np.ascontiguousarray(numeric)

    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())

    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode("utf-8"))

    return h.hexdigest()


def hash_dataframe_values(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Hash selected DataFrame values in row order."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot hash missing columns: {missing}")

    values = df.loc[:, list(columns)].to_numpy(dtype=float)
    return hash_numpy_array(values, extra=extra)


def chronological_train_test_indices(
    n_observations: int,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    min_train_observations: int = DEFAULT_MIN_TRAIN_OBSERVATIONS,
    min_test_observations: int = DEFAULT_MIN_TEST_OBSERVATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build chronological train/test positional indices.

    No shuffling. Train rows are the first chronological block.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1). Got {train_fraction}.")

    train_end = int(np.floor(n_observations * train_fraction))
    n_train = train_end
    n_test = n_observations - train_end

    if n_train < min_train_observations:
        raise ValueError(
            f"Insufficient train observations: {n_train} < {min_train_observations}."
        )

    if n_test < min_test_observations:
        raise ValueError(
            f"Insufficient test observations: {n_test} < {min_test_observations}."
        )

    train_indices = np.arange(0, train_end, dtype=int)
    test_indices = np.arange(train_end, n_observations, dtype=int)

    return train_indices, test_indices


def _safe_date_bounds(
    df: pd.DataFrame,
    indices: np.ndarray,
    *,
    date_col: str = DEFAULT_DATE_COL,
) -> tuple[str, str]:
    """Return first/last date for positional indices."""
    if date_col not in df.columns or len(indices) == 0:
        return "", ""

    dates = pd.to_datetime(df.iloc[indices][date_col], errors="coerce").dropna()
    if dates.empty:
        return "", ""

    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _validate_feature_matrix(
    X: pd.DataFrame | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate numeric feature matrix."""
    arr = np.asarray(X, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {arr.shape}.")

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} cannot have zero rows or columns.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def fit_train_only_standard_scaler(
    X_raw: np.ndarray,
    *,
    train_indices: np.ndarray,
) -> StandardScaler:
    """
    Fit StandardScaler only on train rows.

    This function must never receive test rows inside fit except through X_raw
    indexing with train_indices.
    """
    X = _validate_feature_matrix(X_raw, name="X_raw")

    train_indices = np.asarray(train_indices, dtype=int)
    if train_indices.ndim != 1 or train_indices.size == 0:
        raise ValueError("train_indices must be a non-empty 1D array.")

    if train_indices.min() < 0 or train_indices.max() >= X.shape[0]:
        raise ValueError("train_indices are out of bounds for X_raw.")

    scaler = StandardScaler()
    scaler.fit(X[train_indices])
    return scaler


def transform_with_fitted_scaler(
    X_raw: np.ndarray,
    *,
    scaler: StandardScaler,
) -> np.ndarray:
    """Transform raw features with an already-fitted scaler."""
    X = _validate_feature_matrix(X_raw, name="X_raw")

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("scaler must be a fitted StandardScaler.")

    X_scaled = scaler.transform(X)

    if not np.all(np.isfinite(X_scaled)):
        raise ValueError("Scaled feature matrix contains non-finite values.")

    return np.asarray(X_scaled, dtype=float)


def _scaler_metadata(
    *,
    panel: HMMFeaturePanel,
    scaler: StandardScaler,
    X_raw: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_fraction: float,
    date_col: str,
) -> HMMScalerMetadata:
    """Build scaler metadata."""
    feature_cols = tuple(panel.feature_cols)
    feature_name_map = get_hmm_feature_output_name_map(feature_cols)
    scaled_feature_cols = tuple(feature_name_map[col] for col in feature_cols)

    train_start, train_end = _safe_date_bounds(
        panel.eligible_panel,
        train_indices,
        date_col=date_col,
    )

    train_X = X_raw[train_indices]
    scaler_input_hash = hash_numpy_array(
        train_X,
        extra={
            "market": panel.market,
            "feature_set": panel.feature_set,
            "feature_cols": list(feature_cols),
            "n_train": int(len(train_indices)),
        },
    )

    train_window_hash = _json_hash(
        {
            "market": panel.market,
            "feature_set": panel.feature_set,
            "train_start": train_start,
            "train_end": train_end,
            "train_indices_first": int(train_indices[0]),
            "train_indices_last": int(train_indices[-1]),
            "n_train": int(len(train_indices)),
        }
    )

    scaler_means = scaler.mean_
    scaler_scales = scaler.scale_
    if scaler_means is None or scaler_scales is None:
        raise ValueError("scaler must be a fitted StandardScaler.")

    means = {
        feature: float(value)
        for feature, value in zip(feature_cols, scaler_means, strict=True)
    }
    scales = {
        feature: float(value)
        for feature, value in zip(feature_cols, scaler_scales, strict=True)
    }

    return HMMScalerMetadata(
        scaler_method="standard",
        market=panel.market,
        feature_set=panel.feature_set,
        feature_cols=feature_cols,
        scaled_feature_cols=scaled_feature_cols,
        n_observations=int(X_raw.shape[0]),
        n_train=int(len(train_indices)),
        n_test=int(len(test_indices)),
        train_fraction=float(train_fraction),
        scaler_fit_start_date=train_start,
        scaler_fit_end_date=train_end,
        scaler_feature_means=means,
        scaler_feature_scales=scales,
        scaler_input_hash=scaler_input_hash,
        train_window_hash=train_window_hash,
        created_at_utc=_utc_now_iso(),
    )


def append_scaled_columns(
    df: pd.DataFrame,
    *,
    X_scaled: np.ndarray,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Append explicit HMM scaled feature columns to a panel."""
    out = df.copy().reset_index(drop=True)
    feature_cols = tuple(feature_cols)

    feature_name_map = get_hmm_feature_output_name_map(feature_cols)
    scaled_feature_cols = tuple(feature_name_map[col] for col in feature_cols)

    X_scaled = _validate_feature_matrix(X_scaled, name="X_scaled")
    if X_scaled.shape != (len(out), len(feature_cols)):
        raise ValueError(
            f"X_scaled shape mismatch. Expected {(len(out), len(feature_cols))}, "
            f"got {X_scaled.shape}."
        )

    for i, scaled_col in enumerate(scaled_feature_cols):
        out[scaled_col] = X_scaled[:, i]

    return out, scaled_feature_cols


def scale_hmm_feature_panel(
    panel: HMMFeaturePanel,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    min_train_observations: int = DEFAULT_MIN_TRAIN_OBSERVATIONS,
    min_test_observations: int = DEFAULT_MIN_TEST_OBSERVATIONS,
    date_col: str = DEFAULT_DATE_COL,
) -> HMMScaledFeaturePanel:
    """
    Fit train-only scaler and transform full eligible panel.

    This is the main entry point for Chunk 5.
    """
    X_raw = get_feature_matrix(panel, as_numpy=True)
    X_raw = _validate_feature_matrix(X_raw, name="X_raw")

    train_indices, test_indices = chronological_train_test_indices(
        X_raw.shape[0],
        train_fraction=train_fraction,
        min_train_observations=min_train_observations,
        min_test_observations=min_test_observations,
    )

    scaler = fit_train_only_standard_scaler(
        X_raw,
        train_indices=train_indices,
    )
    X_scaled = transform_with_fitted_scaler(
        X_raw,
        scaler=scaler,
    )

    scaled_panel, scaled_feature_cols = append_scaled_columns(
        panel.eligible_panel,
        X_scaled=X_scaled,
        feature_cols=panel.feature_cols,
    )

    metadata = _scaler_metadata(
        panel=panel,
        scaler=scaler,
        X_raw=X_raw,
        train_indices=train_indices,
        test_indices=test_indices,
        train_fraction=train_fraction,
        date_col=date_col,
    )

    return HMMScaledFeaturePanel(
        market=panel.market,
        feature_set=panel.feature_set,
        feature_cols=tuple(panel.feature_cols),
        scaled_feature_cols=scaled_feature_cols,
        scaled_panel=scaled_panel,
        scaler=scaler,
        metadata=metadata,
        train_indices=train_indices,
        test_indices=test_indices,
        X_raw=X_raw,
        X_scaled=X_scaled,
    )


def transform_hmm_feature_panel_with_fitted_scaler(
    panel: HMMFeaturePanel,
    *,
    scaler: StandardScaler,
    source_scaler_metadata: Mapping[str, Any],
) -> HMMTransformedFeaturePanel:
    """
    Transform a panel with an already-fitted scaler.

    This is used when future rows are appended. Do not refit the scaler.
    """
    X_raw = get_feature_matrix(panel, as_numpy=True)
    X_raw = _validate_feature_matrix(X_raw, name="X_raw")

    X_scaled = transform_with_fitted_scaler(
        X_raw,
        scaler=scaler,
    )

    scaled_panel, scaled_feature_cols = append_scaled_columns(
        panel.eligible_panel,
        X_scaled=X_scaled,
        feature_cols=panel.feature_cols,
    )

    return HMMTransformedFeaturePanel(
        market=panel.market,
        feature_set=panel.feature_set,
        feature_cols=tuple(panel.feature_cols),
        scaled_feature_cols=scaled_feature_cols,
        scaled_panel=scaled_panel,
        X_raw=X_raw,
        X_scaled=X_scaled,
        source_scaler_metadata=dict(source_scaler_metadata),
    )


def assert_scaler_fit_uses_train_only(
    scaled: HMMScaledFeaturePanel,
    *,
    atol: float = 1.0e-12,
) -> None:
    """
    Assert scaler mean/scale equal train-window statistics, not full-sample stats.

    StandardScaler uses population standard deviation, i.e. ddof=0.
    """
    X_train = scaled.X_raw[scaled.train_indices]

    expected_mean = X_train.mean(axis=0)
    expected_scale = X_train.std(axis=0, ddof=0)
    expected_scale = np.where(expected_scale == 0.0, 1.0, expected_scale)

    # mypy/typing: scaled.scaler.mean_ / scale_ may be Optional in some stubs;
    # ensure we pass concrete ndarray to numpy.testing.assert_allclose
    np.testing.assert_allclose(np.asarray(scaled.scaler.mean_), expected_mean, atol=atol)
    np.testing.assert_allclose(np.asarray(scaled.scaler.scale_), expected_scale, atol=atol)


def max_abs_scaled_prefix_difference(
    prefix_scaled: pd.DataFrame,
    full_scaled: pd.DataFrame,
    *,
    scaled_feature_cols: Sequence[str],
    n_prefix_rows: int,
) -> float:
    """
    Compare scaled columns for a prefix panel versus a longer future-appended panel.

    Both panels must have been transformed using the same fitted scaler.
    """
    if n_prefix_rows <= 0:
        raise ValueError("n_prefix_rows must be positive.")

    missing_prefix = [col for col in scaled_feature_cols if col not in prefix_scaled.columns]
    missing_full = [col for col in scaled_feature_cols if col not in full_scaled.columns]

    if missing_prefix:
        raise ValueError(f"prefix_scaled missing columns: {missing_prefix}")

    if missing_full:
        raise ValueError(f"full_scaled missing columns: {missing_full}")

    prefix_values = prefix_scaled.loc[: n_prefix_rows - 1, list(scaled_feature_cols)].to_numpy(
        dtype=float
    )
    full_values = full_scaled.loc[: n_prefix_rows - 1, list(scaled_feature_cols)].to_numpy(
        dtype=float
    )

    return float(np.max(np.abs(prefix_values - full_values)))


def assert_scaled_prefix_invariance(
    prefix_scaled: pd.DataFrame,
    full_scaled: pd.DataFrame,
    *,
    scaled_feature_cols: Sequence[str],
    n_prefix_rows: int,
    atol: float = 1.0e-12,
) -> None:
    """
    Assert that appending future rows does not alter earlier scaled values.

    This assumes the same fitted scaler is reused.
    """
    diff = max_abs_scaled_prefix_difference(
        prefix_scaled,
        full_scaled,
        scaled_feature_cols=scaled_feature_cols,
        n_prefix_rows=n_prefix_rows,
    )

    if diff > atol:
        raise ValueError(f"Scaled prefix invariance failed: {diff} > {atol}.")


def scaler_metadata_to_json_dict(metadata: HMMScalerMetadata) -> dict[str, Any]:
    """Return scaler metadata as JSON-compatible dict."""
    return metadata.to_dict()


__all__ = [
    "HMMScalerMetadata",
    "HMMScaledFeaturePanel",
    "HMMTransformedFeaturePanel",
    "hash_numpy_array",
    "hash_dataframe_values",
    "chronological_train_test_indices",
    "fit_train_only_standard_scaler",
    "transform_with_fitted_scaler",
    "append_scaled_columns",
    "scale_hmm_feature_panel",
    "transform_hmm_feature_panel_with_fitted_scaler",
    "assert_scaler_fit_uses_train_only",
    "max_abs_scaled_prefix_difference",
    "assert_scaled_prefix_invariance",
    "scaler_metadata_to_json_dict",
]