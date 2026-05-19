"""
Point-in-time forward filtering for Phase 6 Gaussian HMM regimes.

This module implements custom HMM filtering:

    P(S_t | X_1:t)

It deliberately does not call hmmlearn.predict_proba for backtest-facing
probabilities, because predict_proba / posterior probabilities are full-sequence
diagnostics and can use future observations through smoothing.

The fitted HMM parameters come from hmmlearn:
- startprob_
- transmat_
- means_
- covars_
- covariance_type

The filter itself is implemented here in log-space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HMMFilterResult:
    """
    Output of the custom forward filter.

    Attributes
    ----------
    filtered_probs:
        Array with shape (n_observations, n_states). Row t is P(S_t | X_1:t).
    log_filtered_probs:
        Log version of filtered_probs.
    log_likelihood:
        Total log likelihood of the observed sequence under the fitted HMM.
    incremental_log_likelihood:
        Array with shape (n_observations,). Entry t is log p(X_t | X_1:t-1)
        with t=0 interpreted as log p(X_0).
    emission_log_likelihood:
        Array with shape (n_observations, n_states). Entry [t, k] is
        log p(X_t | S_t=k).
    """

    filtered_probs: np.ndarray
    log_filtered_probs: np.ndarray
    log_likelihood: float
    incremental_log_likelihood: np.ndarray
    emission_log_likelihood: np.ndarray


def stable_logsumexp(values: Any, axis: int | tuple[int, ...] | None = None) -> np.ndarray | float:
    """
    Numerically stable log(sum(exp(values))).

    Parameters
    ----------
    values:
        Array-like input.
    axis:
        Axis or axes over which to reduce. If None, reduce all values.

    Returns
    -------
    float or ndarray
        Log-sum-exp result.

    Notes
    -----
    Positive infinity is rejected. Negative infinity is allowed and useful for
    impossible HMM paths.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        raise ValueError("stable_logsumexp input cannot be empty.")

    if np.isnan(arr).any():
        raise ValueError("stable_logsumexp input contains NaN.")

    if np.isposinf(arr).any():
        raise ValueError("stable_logsumexp input contains positive infinity.")

    max_val = np.max(arr, axis=axis, keepdims=True)
    finite_max = np.isfinite(max_val)

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = np.where(finite_max, np.exp(arr - max_val), 0.0)
        sum_exp = np.sum(shifted, axis=axis, keepdims=True)
        out = np.where(finite_max, max_val + np.log(sum_exp), -np.inf)

    # Remove keepdims dimensions
    if axis is None:
        out = np.squeeze(out)
    else:
        out = np.squeeze(out, axis=axis)

    if np.ndim(out) == 0:
        return float(out)

    return out


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    """
    Convert input to 2D float array.

    A 1D input is treated as a single-feature time series and reshaped to
    (n_observations, 1).
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Got shape {arr.shape}.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation.")

    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def _validate_means(means: Any, *, n_features: int) -> np.ndarray:
    """Validate Gaussian HMM mean matrix."""
    arr = np.asarray(means, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"means must be 1D or 2D. Got shape {arr.shape}.")

    if arr.shape[0] == 0:
        raise ValueError("means must contain at least one state.")

    if arr.shape[1] != n_features:
        raise ValueError(
            f"means feature dimension mismatch. Expected {n_features}, got {arr.shape[1]}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("means contains non-finite values.")

    return arr


def _validate_diag_covars(covars: Any, *, n_states: int, n_features: int, min_covar: float) -> np.ndarray:
    """
    Validate diagonal covariance array and return shape (n_states, n_features).

    Supports hmmlearn-style diag covariances as either:
    - (n_states, n_features)
    - (n_states, n_features, n_features), from which the diagonal is extracted
    """
    arr = np.asarray(covars, dtype=float)

    if arr.ndim == 1:
        if n_states == 1 and arr.shape[0] == n_features:
            arr = arr.reshape(1, n_features)
        elif n_features == 1 and arr.shape[0] == n_states:
            arr = arr.reshape(n_states, 1)
        else:
            raise ValueError(
                f"Cannot interpret 1D diag covars with shape {arr.shape} for "
                f"n_states={n_states}, n_features={n_features}."
            )

    if arr.ndim == 3:
        if arr.shape != (n_states, n_features, n_features):
            raise ValueError(
                f"Diag covars 3D shape mismatch. Expected "
                f"{(n_states, n_features, n_features)}, got {arr.shape}."
            )
        arr = np.diagonal(arr, axis1=1, axis2=2)

    if arr.ndim != 2:
        raise ValueError(f"Diag covars must be 1D, 2D, or 3D. Got shape {arr.shape}.")

    if arr.shape != (n_states, n_features):
        raise ValueError(
            f"Diag covars shape mismatch. Expected {(n_states, n_features)}, got {arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("Diag covars contain non-finite values.")

    arr = np.maximum(arr, float(min_covar))

    if np.any(arr <= 0.0):
        raise ValueError("Diag covars must be strictly positive after regularization.")

    return arr


def _validate_full_covars(covars: Any, *, n_states: int, n_features: int) -> np.ndarray:
    """
    Validate full covariance array and return shape (n_states, n_features, n_features).
    """
    arr = np.asarray(covars, dtype=float)

    if arr.ndim == 2 and n_states == 1:
        arr = arr.reshape(1, n_features, n_features)

    expected_shape = (n_states, n_features, n_features)
    if arr.shape != expected_shape:
        raise ValueError(f"Full covars shape mismatch. Expected {expected_shape}, got {arr.shape}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Full covars contain non-finite values.")

    return arr


def gaussian_log_likelihood_diag(
    X: Any,
    means: Any,
    covars: Any,
    min_covar: float = 1.0e-8,
) -> np.ndarray:
    """
    Gaussian emission log likelihood for diagonal covariances.

    Parameters
    ----------
    X:
        Observations with shape (n_observations, n_features).
    means:
        State means with shape (n_states, n_features).
    covars:
        Diagonal covariances with shape (n_states, n_features), or full matrices
        from which diagonals can be extracted.
    min_covar:
        Minimum variance floor.

    Returns
    -------
    ndarray
        Matrix with shape (n_observations, n_states).
    """
    X_arr = _as_2d_float_array(X, name="X")
    means_arr = _validate_means(means, n_features=X_arr.shape[1])

    n_states, n_features = means_arr.shape
    covars_arr = _validate_diag_covars(
        covars,
        n_states=n_states,
        n_features=n_features,
        min_covar=min_covar,
    )

    diff = X_arr[:, None, :] - means_arr[None, :, :]
    log_det = np.sum(np.log(covars_arr), axis=1)
    quadratic = np.sum((diff * diff) / covars_arr[None, :, :], axis=2)

    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det[None, :] + quadratic)


def _regularized_cholesky(cov: np.ndarray, *, min_covar: float, max_attempts: int = 8) -> np.ndarray:
    """
    Compute Cholesky factor with increasing diagonal jitter if needed.
    """
    n_features = cov.shape[0]
    sym_cov = 0.5 * (cov + cov.T)

    jitter = float(min_covar)
    for _ in range(max_attempts):
        try:
            return np.linalg.cholesky(sym_cov + jitter * np.eye(n_features))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise ValueError("Full covariance is not positive definite after regularization.")


def gaussian_log_likelihood_full(
    X: Any,
    means: Any,
    covars: Any,
    min_covar: float = 1.0e-8,
) -> np.ndarray:
    """
    Gaussian emission log likelihood for full covariances.

    Parameters
    ----------
    X:
        Observations with shape (n_observations, n_features).
    means:
        State means with shape (n_states, n_features).
    covars:
        Full covariance matrices with shape (n_states, n_features, n_features).
    min_covar:
        Diagonal jitter used for numerical regularization.

    Returns
    -------
    ndarray
        Matrix with shape (n_observations, n_states).
    """
    X_arr = _as_2d_float_array(X, name="X")
    means_arr = _validate_means(means, n_features=X_arr.shape[1])

    n_states, n_features = means_arr.shape
    covars_arr = _validate_full_covars(
        covars,
        n_states=n_states,
        n_features=n_features,
    )

    out = np.empty((X_arr.shape[0], n_states), dtype=float)
    constant = n_features * np.log(2.0 * np.pi)

    for state in range(n_states):
        chol = _regularized_cholesky(
            covars_arr[state],
            min_covar=min_covar,
        )

        diff = X_arr - means_arr[state]
        solved = np.linalg.solve(chol, diff.T).T
        quadratic = np.sum(solved * solved, axis=1)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))

        out[:, state] = -0.5 * (constant + log_det + quadratic)

    return out


def gaussian_log_likelihood(
    X: Any,
    means: Any,
    covars: Any,
    covariance_type: str,
    min_covar: float = 1.0e-8,
) -> np.ndarray:
    """
    Dispatch Gaussian emission log likelihood by covariance type.

    Supported covariance types for Phase 6:
    - diag
    - full
    """
    if covariance_type == "diag":
        return gaussian_log_likelihood_diag(
            X=X,
            means=means,
            covars=covars,
            min_covar=min_covar,
        )

    if covariance_type == "full":
        return gaussian_log_likelihood_full(
            X=X,
            means=means,
            covars=covars,
            min_covar=min_covar,
        )

    raise ValueError(f"Unsupported covariance_type={covariance_type!r}. Use 'diag' or 'full'.")


def _validate_probability_vector(prob: Any, *, name: str, atol: float = 1.0e-10) -> np.ndarray:
    """Validate and normalize a probability vector within numerical tolerance."""
    arr = np.asarray(prob, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    if np.any(arr < -atol):
        raise ValueError(f"{name} contains negative probabilities.")

    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())

    if total <= 0.0:
        raise ValueError(f"{name} sums to zero.")

    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1. Got {total}.")

    return arr / total


def _validate_transition_matrix(transmat: Any, *, n_states: int, atol: float = 1.0e-10) -> np.ndarray:
    """Validate and normalize a row-stochastic transition matrix within tolerance."""
    arr = np.asarray(transmat, dtype=float)

    expected_shape = (n_states, n_states)
    if arr.shape != expected_shape:
        raise ValueError(f"transmat shape mismatch. Expected {expected_shape}, got {arr.shape}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("transmat contains non-finite values.")

    if np.any(arr < -atol):
        raise ValueError("transmat contains negative probabilities.")

    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1)

    if np.any(row_sums <= 0.0):
        raise ValueError("transmat contains a zero-sum row.")

    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"transmat rows must sum to 1. Got row sums {row_sums}.")

    return arr / row_sums[:, None]


def _log_prob(prob: np.ndarray) -> np.ndarray:
    """Convert probabilities to log probabilities, preserving zeros as -inf."""
    with np.errstate(divide="ignore"):
        return np.where(prob > 0.0, np.log(prob), -np.inf)


def forward_filter_from_log_likelihood(
    emission_log_likelihood: Any,
    startprob: Any,
    transmat: Any,
    *,
    atol: float = 1.0e-10,
) -> HMMFilterResult:
    """
    Run a custom log-space HMM forward filter from precomputed emissions.

    Parameters
    ----------
    emission_log_likelihood:
        Matrix with shape (n_observations, n_states). Entry [t, k] is
        log p(X_t | S_t=k).
    startprob:
        Initial state probability vector with shape (n_states,).
    transmat:
        Transition matrix with shape (n_states, n_states). Entry [i, j] is
        P(S_t=j | S_{t-1}=i).
    atol:
        Probability validation tolerance.

    Returns
    -------
    HMMFilterResult
        Point-in-time filtered probabilities and log-likelihood diagnostics.
    """
    log_b = np.asarray(emission_log_likelihood, dtype=float)

    if log_b.ndim != 2:
        raise ValueError(
            "emission_log_likelihood must be 2D with shape "
            "(n_observations, n_states)."
        )

    if log_b.shape[0] == 0 or log_b.shape[1] == 0:
        raise ValueError("emission_log_likelihood cannot have zero observations or states.")

    if np.isnan(log_b).any() or np.isposinf(log_b).any():
        raise ValueError("emission_log_likelihood contains NaN or positive infinity.")

    n_observations, n_states = log_b.shape

    start = _validate_probability_vector(startprob, name="startprob", atol=atol)
    if start.shape[0] != n_states:
        raise ValueError(
            f"startprob length mismatch. Expected {n_states}, got {start.shape[0]}."
        )

    transition = _validate_transition_matrix(transmat, n_states=n_states, atol=atol)

    log_start = _log_prob(start)
    log_transition = _log_prob(transition)

    log_filtered = np.empty((n_observations, n_states), dtype=float)
    incremental_loglik = np.empty(n_observations, dtype=float)

    raw_t0 = log_start + log_b[0]
    norm_t0 = stable_logsumexp(raw_t0)

    if not np.isfinite(norm_t0):
        raise ValueError("No valid HMM path for first observation.")

    log_filtered[0] = raw_t0 - norm_t0
    incremental_loglik[0] = norm_t0

    for t in range(1, n_observations):
        predicted = stable_logsumexp(
            log_filtered[t - 1][:, None] + log_transition,
            axis=0,
        )
        raw = predicted + log_b[t]
        norm = stable_logsumexp(raw)

        if not np.isfinite(norm):
            raise ValueError(f"No valid HMM path at observation index {t}.")

        log_filtered[t] = raw - norm
        incremental_loglik[t] = norm

    filtered = np.exp(log_filtered)

    row_sums = filtered.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(
            "Filtered probability row-sum check failed. "
            f"Min={row_sums.min()}, Max={row_sums.max()}."
        )

    return HMMFilterResult(
        filtered_probs=filtered,
        log_filtered_probs=log_filtered,
        log_likelihood=float(incremental_loglik.sum()),
        incremental_log_likelihood=incremental_loglik,
        emission_log_likelihood=log_b,
    )


def forward_filter_gaussian(
    X: Any,
    startprob: Any,
    transmat: Any,
    means: Any,
    covars: Any,
    covariance_type: str,
    *,
    min_covar: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> HMMFilterResult:
    """
    Run custom point-in-time filtering for a Gaussian HMM.

    This is the main entry point used by gaussian_hmm.py after a model has been
    fitted on the training window.
    """
    emission_log_likelihood = gaussian_log_likelihood(
        X=X,
        means=means,
        covars=covars,
        covariance_type=covariance_type,
        min_covar=min_covar,
    )

    return forward_filter_from_log_likelihood(
        emission_log_likelihood=emission_log_likelihood,
        startprob=startprob,
        transmat=transmat,
        atol=atol,
    )


def filtered_state_sequence(filtered_probs: Any) -> np.ndarray:
    """
    Convert filtered probabilities to the most likely filtered raw state sequence.
    """
    probs = np.asarray(filtered_probs, dtype=float)

    if probs.ndim != 2:
        raise ValueError(f"filtered_probs must be 2D. Got shape {probs.shape}.")

    if probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError("filtered_probs cannot have zero observations or states.")

    if not np.all(np.isfinite(probs)):
        raise ValueError("filtered_probs contains non-finite values.")

    return np.argmax(probs, axis=1).astype(int)


def max_abs_prefix_difference(full_probs: Any, prefix_probs: Any) -> float:
    """
    Compare a full-sequence filter with a prefix-only filter.

    For a valid point-in-time filter, filtering rows 0:k should be identical
    whether we run the filter on X[0:k] or on the full X[0:T].
    """
    full = np.asarray(full_probs, dtype=float)
    prefix = np.asarray(prefix_probs, dtype=float)

    if full.ndim != 2 or prefix.ndim != 2:
        raise ValueError("Both probability inputs must be 2D.")

    if full.shape[1] != prefix.shape[1]:
        raise ValueError(
            f"State dimension mismatch. Full={full.shape[1]}, prefix={prefix.shape[1]}."
        )

    n = min(full.shape[0], prefix.shape[0])
    if n == 0:
        raise ValueError("Cannot compare empty probability arrays.")

    return float(np.max(np.abs(full[:n] - prefix[:n])))


def assert_prefix_filter_invariance(
    full_probs: Any,
    prefix_probs: Any,
    *,
    atol: float = 1.0e-12,
) -> None:
    """
    Raise if prefix invariance fails.

    This is a direct no-lookahead test for the custom forward filter.
    """
    diff = max_abs_prefix_difference(full_probs, prefix_probs)
    if diff > atol:
        raise ValueError(f"Prefix filter invariance failed. max_abs_diff={diff} > {atol}.")


__all__ = [
    "HMMFilterResult",
    "stable_logsumexp",
    "gaussian_log_likelihood_diag",
    "gaussian_log_likelihood_full",
    "gaussian_log_likelihood",
    "forward_filter_from_log_likelihood",
    "forward_filter_gaussian",
    "filtered_state_sequence",
    "max_abs_prefix_difference",
    "assert_prefix_filter_invariance",
]