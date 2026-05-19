import math

import numpy as np
import pytest

from vrp.regimes.online_filter import (
    assert_prefix_filter_invariance,
    filtered_state_sequence,
    forward_filter_from_log_likelihood,
    forward_filter_gaussian,
    gaussian_log_likelihood,
    gaussian_log_likelihood_diag,
    gaussian_log_likelihood_full,
    max_abs_prefix_difference,
    stable_logsumexp,
)


def test_stable_logsumexp_matches_naive_on_moderate_values():
    values = np.array([-1.0, 0.0, 1.0, 2.0])
    expected = np.log(np.sum(np.exp(values)))

    actual = stable_logsumexp(values)

    assert np.isclose(actual, expected)


def test_stable_logsumexp_handles_large_negative_values():
    values = np.array([-1000.0, -1001.0, -1002.0])

    actual = stable_logsumexp(values)

    assert np.isfinite(actual)
    assert actual < -999.0


def test_stable_logsumexp_axis_reduction():
    values = np.array(
        [
            [-2.0, -1.0],
            [-4.0, -3.0],
        ]
    )

    actual = stable_logsumexp(values, axis=0)
    expected = np.log(np.sum(np.exp(values), axis=0))

    np.testing.assert_allclose(actual, expected)


def test_stable_logsumexp_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        stable_logsumexp(np.array([0.0, np.nan]))


def test_gaussian_log_likelihood_diag_1d_standard_normal():
    X = np.array([[0.0]])
    means = np.array([[0.0]])
    covars = np.array([[1.0]])

    loglik = gaussian_log_likelihood_diag(X, means, covars)

    expected = -0.5 * math.log(2.0 * math.pi)
    assert loglik.shape == (1, 1)
    assert np.isclose(loglik[0, 0], expected)


def test_gaussian_log_likelihood_diag_two_states():
    X = np.array([[0.0], [2.0]])
    means = np.array([[0.0], [2.0]])
    covars = np.array([[1.0], [1.0]])

    loglik = gaussian_log_likelihood_diag(X, means, covars)

    assert loglik.shape == (2, 2)
    assert loglik[0, 0] > loglik[0, 1]
    assert loglik[1, 1] > loglik[1, 0]


def test_gaussian_log_likelihood_full_matches_diag_for_identity_covariance():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
        ]
    )
    means = np.array([[0.0, 0.0]])
    diag_covars = np.array([[1.0, 1.0]])
    full_covars = np.array([np.eye(2)])

    diag_loglik = gaussian_log_likelihood_diag(X, means, diag_covars)
    full_loglik = gaussian_log_likelihood_full(X, means, full_covars)

    np.testing.assert_allclose(full_loglik, diag_loglik, atol=1.0e-8)


def test_gaussian_log_likelihood_dispatch_rejects_unknown_covariance_type():
    with pytest.raises(ValueError, match="Unsupported covariance_type"):
        gaussian_log_likelihood(
            X=np.array([[0.0]]),
            means=np.array([[0.0]]),
            covars=np.array([[1.0]]),
            covariance_type="tied",
        )


def test_forward_filter_from_log_likelihood_row_sums_are_one():
    emission_log_likelihood = np.log(
        np.array(
            [
                [0.80, 0.20],
                [0.70, 0.30],
                [0.30, 0.70],
                [0.20, 0.80],
            ]
        )
    )
    startprob = np.array([0.5, 0.5])
    transmat = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ]
    )

    result = forward_filter_from_log_likelihood(
        emission_log_likelihood=emission_log_likelihood,
        startprob=startprob,
        transmat=transmat,
    )

    assert result.filtered_probs.shape == (4, 2)
    np.testing.assert_allclose(result.filtered_probs.sum(axis=1), 1.0)
    assert np.isfinite(result.log_likelihood)
    assert result.incremental_log_likelihood.shape == (4,)


def test_forward_filter_first_step_matches_bayes_rule():
    emission = np.array([[0.80, 0.20]])
    emission_log_likelihood = np.log(emission)
    startprob = np.array([0.25, 0.75])
    transmat = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ]
    )

    result = forward_filter_from_log_likelihood(
        emission_log_likelihood=emission_log_likelihood,
        startprob=startprob,
        transmat=transmat,
    )

    unnormalized = startprob * emission[0]
    expected = unnormalized / unnormalized.sum()

    np.testing.assert_allclose(result.filtered_probs[0], expected)


def test_forward_filter_prefix_invariance_from_log_likelihood():
    emission_log_likelihood_full = np.log(
        np.array(
            [
                [0.80, 0.20],
                [0.75, 0.25],
                [0.60, 0.40],
                [0.40, 0.60],
                [0.30, 0.70],
                [0.20, 0.80],
            ]
        )
    )
    startprob = np.array([0.5, 0.5])
    transmat = np.array(
        [
            [0.85, 0.15],
            [0.10, 0.90],
        ]
    )

    full = forward_filter_from_log_likelihood(
        emission_log_likelihood=emission_log_likelihood_full,
        startprob=startprob,
        transmat=transmat,
    )
    prefix = forward_filter_from_log_likelihood(
        emission_log_likelihood=emission_log_likelihood_full[:4],
        startprob=startprob,
        transmat=transmat,
    )

    diff = max_abs_prefix_difference(full.filtered_probs, prefix.filtered_probs)

    assert diff < 1.0e-12
    assert_prefix_filter_invariance(full.filtered_probs, prefix.filtered_probs)


def test_forward_filter_gaussian_prefix_invariance():
    X_full = np.array(
        [
            [-2.0],
            [-1.5],
            [-1.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
        ]
    )
    startprob = np.array([0.5, 0.5])
    transmat = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
        ]
    )
    means = np.array([[-1.5], [1.5]])
    covars = np.array([[0.5], [0.5]])

    full = forward_filter_gaussian(
        X=X_full,
        startprob=startprob,
        transmat=transmat,
        means=means,
        covars=covars,
        covariance_type="diag",
    )
    prefix = forward_filter_gaussian(
        X=X_full[:5],
        startprob=startprob,
        transmat=transmat,
        means=means,
        covars=covars,
        covariance_type="diag",
    )

    assert_prefix_filter_invariance(full.filtered_probs, prefix.filtered_probs)


def test_filtered_state_sequence_returns_argmax_states():
    probs = np.array(
        [
            [0.90, 0.10],
            [0.40, 0.60],
            [0.49, 0.51],
        ]
    )

    states = filtered_state_sequence(probs)

    np.testing.assert_array_equal(states, np.array([0, 1, 1]))


def test_forward_filter_rejects_bad_startprob_sum():
    emission_log_likelihood = np.log(np.array([[0.5, 0.5]]))
    startprob = np.array([0.4, 0.4])
    transmat = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )

    with pytest.raises(ValueError, match="startprob must sum to 1"):
        forward_filter_from_log_likelihood(
            emission_log_likelihood=emission_log_likelihood,
            startprob=startprob,
            transmat=transmat,
        )


def test_forward_filter_rejects_bad_transition_row_sum():
    emission_log_likelihood = np.log(np.array([[0.5, 0.5]]))
    startprob = np.array([0.5, 0.5])
    transmat = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.7],
        ]
    )

    with pytest.raises(ValueError, match="transmat rows must sum to 1"):
        forward_filter_from_log_likelihood(
            emission_log_likelihood=emission_log_likelihood,
            startprob=startprob,
            transmat=transmat,
        )


def test_forward_filter_rejects_impossible_first_observation():
    emission_log_likelihood = np.array([[-np.inf, -np.inf]])
    startprob = np.array([0.5, 0.5])
    transmat = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )

    with pytest.raises(ValueError, match="No valid HMM path"):
        forward_filter_from_log_likelihood(
            emission_log_likelihood=emission_log_likelihood,
            startprob=startprob,
            transmat=transmat,
        )