"""
Gaussian HMM regime model for Phase 6.

This module implements:
- candidate model specification
- hmmlearn GaussianHMM construction
- multiple random initializations
- train/test log likelihood
- AIC/BIC extraction
- convergence metadata
- custom point-in-time filtered probabilities
- diagnostic-only smoothed probabilities
- raw/economic probability separation
- train-only economic state mapping
- t+1 signal aliases
- candidate validation and ranking rows

Out of scope:
- strategy exposure
- PnL backtest
- AR-HMM / Markov autoregression
- MSGARCH
- iBridgePy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import copy
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from vrp.regimes.hmm_registry import (
    HMM_CANDIDATE_RANKING_COLUMNS,
    HMM_DIAGNOSTIC_SMOOTHED_RAW_PROB_PREFIX,
    HMM_FILTERED_RAW_PROB_PREFIX,
    HMM_PRIMARY_MODEL,
    HMM_FALLBACK_MODEL,
    get_hmm_diagnostic_smoothed_probability_columns,
    get_hmm_filtered_raw_probability_columns,
    is_fallback_hmm_spec,
    is_primary_hmm_spec,
    validate_hmm_model_spec,
)
from vrp.regimes.hmm_scaling import HMMScaledFeaturePanel
from vrp.regimes.hmm_validation import (
    HMMValidationRules,
    CandidateValidationResult,
    extract_hmm_convergence_status,
    infer_hmm_model_failure_reason,
    validate_covariances,
    validate_filtered_probability_matrix,
    validate_hmm_candidate_result,
    validate_transition_matrix,
)
from vrp.regimes.online_filter import (
    HMMFilterResult,
    filtered_state_sequence,
    forward_filter_gaussian,
)
from vrp.regimes.state_labeling import (
    HMMStateLabelingResult,
    add_hmm_next_session_signal_columns,
    append_hmm_economic_probabilities,
    append_hmm_state_labels,
    label_hmm_states_from_train_properties,
    map_raw_probabilities_to_economic_probabilities,
    state_labeling_result_to_frame,
)


DEFAULT_RANDOM_SEED = 42
DEFAULT_N_ITER = 1000
DEFAULT_TOL = 1.0e-4
DEFAULT_ALGORITHM = "viterbi"
DEFAULT_IMPLEMENTATION = "log"
DEFAULT_MIN_COVAR = 1.0e-6
DEFAULT_N_INIT = 10
DEFAULT_COVARIANCE_REGULARIZATION = 1.0e-6


@dataclass(frozen=True)
class HMMFitConfig:
    """hmmlearn GaussianHMM fitting configuration."""

    random_seed: int = DEFAULT_RANDOM_SEED
    n_iter: int = DEFAULT_N_ITER
    tol: float = DEFAULT_TOL
    algorithm: str = DEFAULT_ALGORITHM
    implementation: str = DEFAULT_IMPLEMENTATION
    min_covar: float = DEFAULT_MIN_COVAR
    n_init: int = DEFAULT_N_INIT
    covariance_regularization: float = DEFAULT_COVARIANCE_REGULARIZATION
    select_best_initialization_by: str = "train_loglik"
    fail_on_non_convergence: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "HMMFitConfig":
        """Build config from a config dict, usually model_hmm.yaml['hmm_fit'].""" 
        if config is None:
            return cls()

        return cls(
            random_seed=int(config.get("random_seed", DEFAULT_RANDOM_SEED)),
            n_iter=int(config.get("n_iter", DEFAULT_N_ITER)),
            tol=float(config.get("tol", DEFAULT_TOL)),
            algorithm=str(config.get("algorithm", DEFAULT_ALGORITHM)),
            implementation=str(config.get("implementation", DEFAULT_IMPLEMENTATION)),
            min_covar=float(config.get("min_covar", DEFAULT_MIN_COVAR)),
            n_init=int(config.get("n_init", DEFAULT_N_INIT)),
            covariance_regularization=float(
                config.get("covariance_regularization", DEFAULT_COVARIANCE_REGULARIZATION)
            ),
            select_best_initialization_by=str(
                config.get("select_best_initialization_by", "train_loglik")
            ),
            fail_on_non_convergence=bool(config.get("fail_on_non_convergence", False)),
        )


@dataclass(frozen=True)
class HMMCandidateSpec:
    """One Phase 6 Gaussian HMM candidate specification."""

    feature_set: str
    n_states: int
    covariance_type: str

    def __post_init__(self) -> None:
        validate_hmm_model_spec(
            feature_set=self.feature_set,
            n_states=self.n_states,
            covariance_type=self.covariance_type,
        )

    @property
    def name(self) -> str:
        """Stable candidate name."""
        return f"{self.feature_set}_k{self.n_states}_{self.covariance_type}"

    @property
    def is_primary_default(self) -> bool:
        """Return whether this matches the default primary model."""
        return is_primary_hmm_spec(
            self.feature_set,
            self.n_states,
            self.covariance_type,
            primary_model=HMM_PRIMARY_MODEL,
        )

    @property
    def is_fallback_default(self) -> bool:
        """Return whether this matches the default fallback model."""
        return is_fallback_hmm_spec(
            self.feature_set,
            self.n_states,
            self.covariance_type,
            fallback_model=HMM_FALLBACK_MODEL,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "n_states": int(self.n_states),
            "covariance_type": self.covariance_type,
        }


@dataclass(frozen=True)
class HMMInitializationFit:
    """Result from one random initialization."""

    init_index: int
    random_state: int
    model: GaussianHMM | None
    train_loglik: float
    test_loglik: float
    converged: bool | None
    n_iter: int | None
    failed: bool
    failure_reason: str
    warning_messages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def score_key(self) -> float:
        """Selection score for this initialization."""
        if self.failed or self.model is None:
            return -np.inf
        return float(self.train_loglik)


@dataclass(frozen=True)
class HMMCandidateFitResult:
    """
    Fitted candidate result before probability/state-label output construction.
    """

    market: str
    spec: HMMCandidateSpec
    model: GaussianHMM | None
    scaled_panel: HMMScaledFeaturePanel
    fit_config: HMMFitConfig
    train_loglik: float
    test_loglik: float
    train_loglik_per_obs: float
    test_loglik_per_obs: float
    aic: float
    bic: float
    converged: bool | None
    n_iter: int | None
    best_init_index: int | None
    best_random_state: int | None
    initialization_results: tuple[HMMInitializationFit, ...]
    fit_failed: bool
    fit_failure_reason: str
    covariance_valid: bool | None
    covariance_rejection_reasons: tuple[str, ...]
    transition_matrix_valid: bool | None
    transition_matrix_rejection_reasons: tuple[str, ...]
    created_at_utc: str

    @property
    def n_features(self) -> int:
        return int(len(self.scaled_panel.feature_cols))

    @property
    def n_observations(self) -> int:
        return int(self.scaled_panel.X_scaled.shape[0])

    @property
    def n_train(self) -> int:
        return int(len(self.scaled_panel.train_indices))

    @property
    def n_test(self) -> int:
        return int(len(self.scaled_panel.test_indices))

    @property
    def has_fitted_model(self) -> bool:
        return self.model is not None and not self.fit_failed

    def base_ranking_row(self) -> dict[str, Any]:
        """
        Build the candidate-ranking row fields available after fitting.

        Output-stage validation adds occupancy, monotonicity, selected_primary,
        and final rejection_reason.
        """
        rejection_reason = str(self.fit_failure_reason).strip()

        row = {
            "market": self.market,
            "feature_set": self.spec.feature_set,
            "n_states": int(self.spec.n_states),
            "covariance_type": self.spec.covariance_type,
            "n_features": int(self.n_features),
            "n_observations": int(self.n_observations),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "train_loglik": float(self.train_loglik),
            "test_loglik": float(self.test_loglik),
            "train_loglik_per_obs": float(self.train_loglik_per_obs),
            "test_loglik_per_obs": float(self.test_loglik_per_obs),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "converged": self.converged,
            "n_iter": self.n_iter,
            "min_state_occupancy_train": np.nan,
            "min_state_occupancy_test": np.nan,
            "economic_monotonicity_passed": np.nan,
            "selected_primary": False,
            "rejection_reason": rejection_reason,
        }

        row = {col: row.get(col, np.nan) for col in HMM_CANDIDATE_RANKING_COLUMNS}

        return row


@dataclass(frozen=True)
class HMMCandidateOutput:
    """
    Full candidate output after filtering, labeling, and validation.
    """

    fit_result: HMMCandidateFitResult
    filter_result: HMMFilterResult | None
    diagnostic_smoothed_probs: np.ndarray | None
    raw_filtered_states: np.ndarray | None
    labeling: HMMStateLabelingResult | None
    validation: CandidateValidationResult
    output_panel: pd.DataFrame | None
    state_properties: pd.DataFrame | None
    ranking_row: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.validation.passed)

    @property
    def hmm_model_valid(self) -> bool:
        return bool(self.validation.hmm_model_valid)

    @property
    def hmm_model_failure_reason(self) -> str:
        return str(self.validation.hmm_model_failure_reason)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_hmm_candidate_spec(
    feature_set: str,
    n_states: int,
    covariance_type: str,
) -> HMMCandidateSpec:
    """Construct and validate one candidate spec."""
    return HMMCandidateSpec(
        feature_set=str(feature_set),
        n_states=int(n_states),
        covariance_type=str(covariance_type),
    )


def make_gaussian_hmm_model(
    spec: HMMCandidateSpec,
    fit_config: HMMFitConfig,
    *,
    random_state: int,
) -> GaussianHMM:
    """
    Construct hmmlearn GaussianHMM for one random initialization.
    """
    return GaussianHMM(
        n_components=spec.n_states,
        covariance_type=spec.covariance_type,
        min_covar=fit_config.min_covar,
        startprob_prior=1.0,
        transmat_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=fit_config.covariance_regularization,
        covars_weight=1,
        algorithm=fit_config.algorithm,
        random_state=random_state,
        n_iter=fit_config.n_iter,
        tol=fit_config.tol,
        verbose=False,
        params="stmc",
        init_params="stmc",
        implementation=fit_config.implementation,
    )


def _safe_model_score(model: GaussianHMM, X: np.ndarray) -> float:
    """Return model score or -inf on failure."""
    try:
        value = float(model.score(X))
    except Exception:
        return -np.inf

    if not np.isfinite(value):
        return -np.inf

    return value


def _safe_hmm_aic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    Return AIC.

    Uses hmmlearn's native aic() if available. Falls back to a conservative
    parameter-count approximation if not available.
    """
    if hasattr(model, "aic"):
        try:
            value = float(model.aic(X))
            if np.isfinite(value):
                return value
        except Exception:
            pass

    loglik = _safe_model_score(model, X)
    n_params = approximate_gaussian_hmm_n_parameters(
        n_states=int(model.n_components),
        n_features=int(X.shape[1]),
        covariance_type=str(model.covariance_type),
    )
    return float(2.0 * n_params - 2.0 * loglik)


def _safe_hmm_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    Return BIC.

    Uses hmmlearn's native bic() if available. Falls back to a conservative
    parameter-count approximation if not available.
    """
    if hasattr(model, "bic"):
        try:
            value = float(model.bic(X))
            if np.isfinite(value):
                return value
        except Exception:
            pass

    loglik = _safe_model_score(model, X)
    n_params = approximate_gaussian_hmm_n_parameters(
        n_states=int(model.n_components),
        n_features=int(X.shape[1]),
        covariance_type=str(model.covariance_type),
    )
    return float(np.log(X.shape[0]) * n_params - 2.0 * loglik)


def approximate_gaussian_hmm_n_parameters(
    *,
    n_states: int,
    n_features: int,
    covariance_type: str,
) -> int:
    """
    Approximate Gaussian HMM parameter count.

    Components:
    - start probabilities: K - 1
    - transition matrix: K * (K - 1)
    - means: K * D
    - covariances:
        diag: K * D
        full: K * D * (D + 1) / 2
    """
    if n_states not in {2, 3}:
        raise ValueError(f"n_states must be 2 or 3. Got {n_states}.")

    if covariance_type not in {"diag", "full"}:
        raise ValueError(f"Unsupported covariance_type={covariance_type!r}.")

    k = int(n_states)
    d = int(n_features)

    if d <= 0:
        raise ValueError(f"n_features must be positive. Got {n_features}.")

    n_start = k - 1
    n_transition = k * (k - 1)
    n_means = k * d

    if covariance_type == "diag":
        n_covars = k * d
    else:
        n_covars = int(k * d * (d + 1) / 2)

    return int(n_start + n_transition + n_means + n_covars)


def fit_one_hmm_initialization(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    spec: HMMCandidateSpec,
    fit_config: HMMFitConfig,
    init_index: int,
    random_state: int,
) -> HMMInitializationFit:
    """
    Fit one HMM initialization and score train/test likelihood.
    """
    model = make_gaussian_hmm_model(
        spec,
        fit_config,
        random_state=random_state,
    )

    warning_messages: list[str] = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train)

        warning_messages = [str(w.message) for w in caught]

        train_loglik = _safe_model_score(model, X_train)
        test_loglik = _safe_model_score(model, X_test)
        converged, n_iter = extract_hmm_convergence_status(model)

        failed = not np.isfinite(train_loglik)
        failure_reason = "non_finite_train_loglik" if failed else ""

        if fit_config.fail_on_non_convergence and converged is False:
            failed = True
            failure_reason = "non_convergence"

        return HMMInitializationFit(
            init_index=int(init_index),
            random_state=int(random_state),
            model=copy.deepcopy(model),
            train_loglik=float(train_loglik),
            test_loglik=float(test_loglik),
            converged=converged,
            n_iter=n_iter,
            failed=bool(failed),
            failure_reason=failure_reason,
            warning_messages=tuple(warning_messages),
        )

    except Exception as exc:
        return HMMInitializationFit(
            init_index=int(init_index),
            random_state=int(random_state),
            model=None,
            train_loglik=-np.inf,
            test_loglik=-np.inf,
            converged=False,
            n_iter=None,
            failed=True,
            failure_reason=f"fit_exception:{type(exc).__name__}:{exc}",
            warning_messages=tuple(warning_messages),
        )


def select_best_initialization(
    initialization_results: Sequence[HMMInitializationFit],
    *,
    select_by: str = "train_loglik",
) -> HMMInitializationFit | None:
    """
    Select best initialization.

    Phase 6 default: select by train_loglik only. OOS/test log likelihood is used
    for ranking diagnostics, not for fitting.
    """
    if select_by != "train_loglik":
        raise ValueError(
            f"Unsupported initialization selection metric {select_by!r}. "
            "Only 'train_loglik' is allowed."
        )

    usable = [
        result
        for result in initialization_results
        if not result.failed and result.model is not None and np.isfinite(result.train_loglik)
    ]

    if not usable:
        return None

    return max(usable, key=lambda result: result.train_loglik)


def validate_fitted_hmm_parameters(
    model: GaussianHMM,
    *,
    n_features: int,
    rules: HMMValidationRules | None = None,
) -> tuple[bool | None, tuple[str, ...], bool | None, tuple[str, ...]]:
    """
    Validate transition matrix and covariance parameters of a fitted HMM.

    Returns:
        covariance_valid,
        covariance_rejection_reasons,
        transition_matrix_valid,
        transition_matrix_rejection_reasons
    """
    rules = rules or HMMValidationRules()

    covariance_valid, covariance_reasons = validate_covariances(
        model.covars_,
        covariance_type=str(model.covariance_type),
        n_states=int(model.n_components),
        n_features=int(n_features),
        rules=rules,
    )

    transition_valid, transition_reasons = validate_transition_matrix(
        model.transmat_,
        n_states=int(model.n_components),
        rules=rules,
    )

    return (
        covariance_valid,
        tuple(covariance_reasons),
        transition_valid,
        tuple(transition_reasons),
    )


def fit_hmm_candidate(
    scaled_panel: HMMScaledFeaturePanel,
    *,
    spec: HMMCandidateSpec,
    fit_config: HMMFitConfig | None = None,
    validation_rules: HMMValidationRules | None = None,
) -> HMMCandidateFitResult:
    """
    Fit one HMM candidate using multiple random initializations.

    Inputs are already scaled using train-only scaling from Chunk 5.
    """
    fit_config = fit_config or HMMFitConfig()
    validation_rules = validation_rules or HMMValidationRules()

    if scaled_panel.feature_set != spec.feature_set:
        raise ValueError(
            f"Scaled panel feature_set={scaled_panel.feature_set!r} does not match "
            f"candidate spec feature_set={spec.feature_set!r}."
        )

    X = np.asarray(scaled_panel.X_scaled, dtype=float)
    X_train = np.asarray(scaled_panel.X_train_scaled, dtype=float)
    X_test = np.asarray(scaled_panel.X_test_scaled, dtype=float)

    if X.ndim != 2 or X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X, X_train, and X_test must all be 2D arrays.")

    if X_train.shape[1] != X.shape[1] or X_test.shape[1] != X.shape[1]:
        raise ValueError("Train/test feature dimensions do not match full matrix.")

    init_results: list[HMMInitializationFit] = []
    for init_index in range(fit_config.n_init):
        random_state = int(fit_config.random_seed + init_index)
        result = fit_one_hmm_initialization(
            X_train,
            X_test,
            spec=spec,
            fit_config=fit_config,
            init_index=init_index,
            random_state=random_state,
        )
        init_results.append(result)

    best = select_best_initialization(
        init_results,
        select_by=fit_config.select_best_initialization_by,
    )

    if best is None or best.model is None:
        failure_reason = "all_initializations_failed"
        return HMMCandidateFitResult(
            market=scaled_panel.market,
            spec=spec,
            model=None,
            scaled_panel=scaled_panel,
            fit_config=fit_config,
            train_loglik=-np.inf,
            test_loglik=-np.inf,
            train_loglik_per_obs=-np.inf,
            test_loglik_per_obs=-np.inf,
            aic=np.inf,
            bic=np.inf,
            converged=False,
            n_iter=None,
            best_init_index=None,
            best_random_state=None,
            initialization_results=tuple(init_results),
            fit_failed=True,
            fit_failure_reason=failure_reason,
            covariance_valid=None,
            covariance_rejection_reasons=(failure_reason,),
            transition_matrix_valid=None,
            transition_matrix_rejection_reasons=(failure_reason,),
            created_at_utc=_utc_now_iso(),
        )

    model = best.model

    train_loglik = float(best.train_loglik)
    test_loglik = float(best.test_loglik)

    train_loglik_per_obs = (
        float(train_loglik / X_train.shape[0]) if X_train.shape[0] > 0 else -np.inf
    )
    test_loglik_per_obs = (
        float(test_loglik / X_test.shape[0]) if X_test.shape[0] > 0 else -np.inf
    )

    aic = _safe_hmm_aic(model, X_train)
    bic = _safe_hmm_bic(model, X_train)

    (
        covariance_valid,
        covariance_reasons,
        transition_valid,
        transition_reasons,
    ) = validate_fitted_hmm_parameters(
        model,
        n_features=X.shape[1],
        rules=validation_rules,
    )

    parameter_reasons = list(covariance_reasons) + list(transition_reasons)
    fit_failed = len(parameter_reasons) > 0
    fit_failure_reason = "; ".join(parameter_reasons)

    return HMMCandidateFitResult(
        market=scaled_panel.market,
        spec=spec,
        model=model,
        scaled_panel=scaled_panel,
        fit_config=fit_config,
        train_loglik=train_loglik,
        test_loglik=test_loglik,
        train_loglik_per_obs=train_loglik_per_obs,
        test_loglik_per_obs=test_loglik_per_obs,
        aic=float(aic),
        bic=float(bic),
        converged=best.converged,
        n_iter=best.n_iter,
        best_init_index=best.init_index,
        best_random_state=best.random_state,
        initialization_results=tuple(init_results),
        fit_failed=bool(fit_failed),
        fit_failure_reason=fit_failure_reason,
        covariance_valid=covariance_valid,
        covariance_rejection_reasons=tuple(covariance_reasons),
        transition_matrix_valid=transition_valid,
        transition_matrix_rejection_reasons=tuple(transition_reasons),
        created_at_utc=_utc_now_iso(),
    )


def fit_hmm_candidates_for_feature_panel(
    scaled_panel: HMMScaledFeaturePanel,
    *,
    n_states_values: Sequence[int] = (2, 3),
    covariance_types: Sequence[str] = ("diag", "full"),
    fit_config: HMMFitConfig | None = None,
    validation_rules: HMMValidationRules | None = None,
) -> list[HMMCandidateFitResult]:
    """
    Fit all K/covariance candidates for one scaled feature panel.
    """
    results: list[HMMCandidateFitResult] = []

    for n_states in n_states_values:
        for covariance_type in covariance_types:
            spec = make_hmm_candidate_spec(
                feature_set=scaled_panel.feature_set,
                n_states=int(n_states),
                covariance_type=str(covariance_type),
            )
            result = fit_hmm_candidate(
                scaled_panel,
                spec=spec,
                fit_config=fit_config,
                validation_rules=validation_rules,
            )
            results.append(result)

    return results


def candidate_fit_results_to_ranking_frame(
    results: Sequence[HMMCandidateFitResult],
) -> pd.DataFrame:
    """
    Convert candidate fit results to candidate-ranking rows.
    """
    rows = [result.base_ranking_row() for result in results]
    frame = pd.DataFrame(rows)

    for col in HMM_CANDIDATE_RANKING_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    return frame.loc[:, list(HMM_CANDIDATE_RANKING_COLUMNS)]


def _empty_failed_output(
    fit_result: HMMCandidateFitResult,
    *,
    reason: str,
) -> HMMCandidateOutput:
    """Build failed output when filtering/labeling cannot run."""
    failure_reason = infer_hmm_model_failure_reason([reason]) or "unknown_failure"

    validation = CandidateValidationResult(
        market=fit_result.market,
        feature_set=fit_result.spec.feature_set,
        n_states=fit_result.spec.n_states,
        covariance_type=fit_result.spec.covariance_type,
        passed=False,
        rejection_reasons=(reason,),
        hmm_model_valid=False,
        hmm_model_failure_reason=failure_reason,
        converged=fit_result.converged,
    )

    ranking_row = fit_result.base_ranking_row()
    ranking_row["rejection_reason"] = reason
    ranking_row["selected_primary"] = False
    ranking_row["converged"] = fit_result.converged

    return HMMCandidateOutput(
        fit_result=fit_result,
        filter_result=None,
        diagnostic_smoothed_probs=None,
        raw_filtered_states=None,
        labeling=None,
        validation=validation,
        output_panel=None,
        state_properties=None,
        ranking_row=ranking_row,
    )


def compute_custom_filtered_probabilities(
    fit_result: HMMCandidateFitResult,
) -> HMMFilterResult:
    """
    Compute point-in-time filtered probabilities using fixed fitted parameters.

    Uses:
    - model.startprob_
    - model.transmat_
    - model.means_
    - model.covars_

    Does not call predict_proba.
    """
    if fit_result.model is None:
        raise ValueError("Cannot filter because fitted model is None.")

    model = fit_result.model

    return forward_filter_gaussian(
        X=fit_result.scaled_panel.X_scaled,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        covariance_type=str(model.covariance_type),
        min_covar=fit_result.fit_config.min_covar,
    )


def compute_diagnostic_smoothed_probabilities(
    fit_result: HMMCandidateFitResult,
) -> np.ndarray:
    """
    Compute diagnostic-only smoothed/posterior probabilities via hmmlearn.

    These columns must be named hmm_diagnostic_smoothed_prob_raw_state_* and must
    not be used for backtest-facing signals.
    """
    if fit_result.model is None:
        raise ValueError("Cannot compute smoothed diagnostics because fitted model is None.")

    smoothed = fit_result.model.predict_proba(fit_result.scaled_panel.X_scaled)
    smoothed = np.asarray(smoothed, dtype=float)

    expected_shape = (
        fit_result.scaled_panel.X_scaled.shape[0],
        fit_result.spec.n_states,
    )

    if smoothed.shape != expected_shape:
        raise ValueError(
            f"Smoothed probability shape mismatch. Expected {expected_shape}, got {smoothed.shape}."
        )

    if not np.all(np.isfinite(smoothed)):
        raise ValueError("Smoothed diagnostic probabilities contain non-finite values.")

    return smoothed


def append_raw_probability_columns(
    panel: pd.DataFrame,
    *,
    raw_filtered_probs: np.ndarray,
    diagnostic_smoothed_probs: np.ndarray | None,
    n_states: int,
) -> pd.DataFrame:
    """
    Append raw filtered probability and diagnostic smoothed probability columns.
    """
    out = panel.copy().reset_index(drop=True)

    raw_filtered_probs = np.asarray(raw_filtered_probs, dtype=float)
    expected_shape = (len(out), n_states)
    if raw_filtered_probs.shape != expected_shape:
        raise ValueError(
            f"raw_filtered_probs shape mismatch. Expected {expected_shape}, got {raw_filtered_probs.shape}."
        )

    raw_cols = get_hmm_filtered_raw_probability_columns(n_states)
    for state, col in enumerate(raw_cols):
        out[col] = raw_filtered_probs[:, state]

    if diagnostic_smoothed_probs is not None:
        smoothed = np.asarray(diagnostic_smoothed_probs, dtype=float)
        if smoothed.shape != expected_shape:
            raise ValueError(
                f"diagnostic_smoothed_probs shape mismatch. Expected {expected_shape}, got {smoothed.shape}."
            )

        smoothed_cols = get_hmm_diagnostic_smoothed_probability_columns(n_states)
        for state, col in enumerate(smoothed_cols):
            out[col] = smoothed[:, state]

    return out


def build_hmm_candidate_output_panel(
    fit_result: HMMCandidateFitResult,
    *,
    include_smoothed_probabilities_diagnostic: bool = True,
) -> tuple[
    HMMFilterResult,
    np.ndarray | None,
    np.ndarray,
    HMMStateLabelingResult,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build complete HMM output panel for one fitted candidate.

    Returns:
    - filter_result
    - diagnostic_smoothed_probs
    - raw_filtered_states
    - labeling
    - output_panel
    - state_properties
    """
    filter_result = compute_custom_filtered_probabilities(fit_result)
    raw_filtered_probs = filter_result.filtered_probs
    raw_filtered_states = filtered_state_sequence(raw_filtered_probs)

    diagnostic_smoothed_probs = (
        compute_diagnostic_smoothed_probabilities(fit_result)
        if include_smoothed_probabilities_diagnostic
        else None
    )

    labeling = label_hmm_states_from_train_properties(
        fit_result.scaled_panel.scaled_panel,
        raw_filtered_states,
        fit_result.scaled_panel.train_indices,
        n_states=fit_result.spec.n_states,
    )

    out = fit_result.scaled_panel.scaled_panel.copy().reset_index(drop=True)

    out = append_raw_probability_columns(
        out,
        raw_filtered_probs=raw_filtered_probs,
        diagnostic_smoothed_probs=diagnostic_smoothed_probs,
        n_states=fit_result.spec.n_states,
    )

    out = append_hmm_state_labels(
        out,
        raw_filtered_states,
        labeling,
    )

    out = append_hmm_economic_probabilities(
        out,
        raw_filtered_probs,
        labeling,
    )

    out = add_hmm_next_session_signal_columns(out)

    out["hmm_model_name"] = "gaussian_hmm_v1"
    out["hmm_feature_set"] = fit_result.spec.feature_set
    out["hmm_n_states"] = int(fit_result.spec.n_states)
    out["hmm_covariance_type"] = fit_result.spec.covariance_type
    out["hmm_model_converged"] = fit_result.converged
    out["hmm_train_loglik"] = fit_result.train_loglik
    out["hmm_test_loglik"] = fit_result.test_loglik
    out["hmm_aic"] = fit_result.aic
    out["hmm_bic"] = fit_result.bic

    state_properties = state_labeling_result_to_frame(labeling)

    return (
        filter_result,
        diagnostic_smoothed_probs,
        raw_filtered_states,
        labeling,
        out,
        state_properties,
    )


def _build_output_validation(
    fit_result: HMMCandidateFitResult,
    *,
    filter_result: HMMFilterResult,
    raw_filtered_states: np.ndarray,
    labeling: HMMStateLabelingResult,
    validation_rules: HMMValidationRules,
) -> CandidateValidationResult:
    """Run final candidate validation after filtering and state labeling."""
    train_idx = fit_result.scaled_panel.train_indices
    test_idx = fit_result.scaled_panel.test_indices

    train_states = raw_filtered_states[train_idx]
    test_states = raw_filtered_states[test_idx]

    feature_availability_passed = bool(
        fit_result.scaled_panel.scaled_panel.shape[0] == fit_result.scaled_panel.X_scaled.shape[0]
    )

    result = validate_hmm_candidate_result(
        market=fit_result.market,
        feature_set=fit_result.spec.feature_set,
        n_states=fit_result.spec.n_states,
        covariance_type=fit_result.spec.covariance_type,
        train_states=train_states,
        test_states=test_states,
        transmat=fit_result.model.transmat_ if fit_result.model is not None else None,
        covars=fit_result.model.covars_ if fit_result.model is not None else None,
        n_features=fit_result.n_features,
        filtered_probs=filter_result.filtered_probs,
        converged=fit_result.converged,
        economic_monotonicity_passed=labeling.economic_monotonicity_passed,
        feature_availability_passed=feature_availability_passed,
        rules=validation_rules,
    )

    return result


def _ranking_row_from_output(
    output: HMMCandidateOutput,
    *,
    selected_primary: bool = False,
) -> dict[str, Any]:
    """Create final candidate ranking row."""
    row = output.fit_result.base_ranking_row()

    validation = output.validation

    if validation.train_occupancy is not None:
        row["min_state_occupancy_train"] = validation.train_occupancy.min_occupancy

    if validation.test_occupancy is not None:
        row["min_state_occupancy_test"] = validation.test_occupancy.min_occupancy

    row["economic_monotonicity_passed"] = validation.economic_monotonicity_passed
    row["selected_primary"] = bool(selected_primary)
    row["rejection_reason"] = validation.rejection_reason_text
    row["converged"] = validation.converged

    return row


def build_hmm_candidate_output(
    fit_result: HMMCandidateFitResult,
    *,
    include_smoothed_probabilities_diagnostic: bool = True,
    validation_rules: HMMValidationRules | None = None,
) -> HMMCandidateOutput:
    """
    Build full candidate output after fitting.

    This is the main Chunk 8 entry point.
    """
    validation_rules = validation_rules or HMMValidationRules()

    if fit_result.model is None:
        return _empty_failed_output(fit_result, reason="missing_fitted_model")

    if fit_result.fit_failed:
        # Still try to build output if model exists, because diagnostic panels are useful.
        # Final validation will keep it rejected.
        pass

    try:
        (
            filter_result,
            diagnostic_smoothed_probs,
            raw_filtered_states,
            labeling,
            output_panel,
            state_properties,
        ) = build_hmm_candidate_output_panel(
            fit_result,
            include_smoothed_probabilities_diagnostic=include_smoothed_probabilities_diagnostic,
        )

        validation = _build_output_validation(
            fit_result,
            filter_result=filter_result,
            raw_filtered_states=raw_filtered_states,
            labeling=labeling,
            validation_rules=validation_rules,
        )

        if fit_result.fit_failure_reason:
            combined_reasons = tuple(
                reason for reason in (
                    fit_result.fit_failure_reason,
                    validation.rejection_reason_text,
                )
                if reason
            )
            final_reason_text = "; ".join(combined_reasons)
            failure_reason = (
                infer_hmm_model_failure_reason(combined_reasons)
                or validation.hmm_model_failure_reason
                or "unknown_failure"
            )
            validation = CandidateValidationResult(
                market=validation.market,
                feature_set=validation.feature_set,
                n_states=validation.n_states,
                covariance_type=validation.covariance_type,
                passed=False,
                rejection_reasons=combined_reasons,
                hmm_model_valid=False,
                hmm_model_failure_reason=failure_reason,
                train_occupancy=validation.train_occupancy,
                test_occupancy=validation.test_occupancy,
                probability_validation=validation.probability_validation,
                covariance_valid=validation.covariance_valid,
                transition_matrix_valid=validation.transition_matrix_valid,
                economic_monotonicity_passed=validation.economic_monotonicity_passed,
                feature_availability_passed=validation.feature_availability_passed,
                converged=validation.converged,
            )

        temp = HMMCandidateOutput(
            fit_result=fit_result,
            filter_result=filter_result,
            diagnostic_smoothed_probs=diagnostic_smoothed_probs,
            raw_filtered_states=raw_filtered_states,
            labeling=labeling,
            validation=validation,
            output_panel=output_panel,
            state_properties=state_properties,
            ranking_row={},
        )

        ranking_row = _ranking_row_from_output(temp)

        return HMMCandidateOutput(
            fit_result=fit_result,
            filter_result=filter_result,
            diagnostic_smoothed_probs=diagnostic_smoothed_probs,
            raw_filtered_states=raw_filtered_states,
            labeling=labeling,
            validation=validation,
            output_panel=output_panel,
            state_properties=state_properties,
            ranking_row=ranking_row,
        )

    except Exception as exc:
        return _empty_failed_output(
            fit_result,
            reason=f"candidate_output_exception:{type(exc).__name__}:{exc}",
        )


def build_hmm_candidate_outputs(
    fit_results: Sequence[HMMCandidateFitResult],
    *,
    include_smoothed_probabilities_diagnostic: bool = True,
    validation_rules: HMMValidationRules | None = None,
) -> list[HMMCandidateOutput]:
    """Build full output objects for several fitted candidates."""
    return [
        build_hmm_candidate_output(
            result,
            include_smoothed_probabilities_diagnostic=include_smoothed_probabilities_diagnostic,
            validation_rules=validation_rules,
        )
        for result in fit_results
    ]


def choose_primary_hmm_output(
    outputs: Sequence[HMMCandidateOutput],
    *,
    prefer_configured_primary: bool = True,
    allow_configured_fallback: bool = True,
) -> HMMCandidateOutput | None:
    """
    Choose selected primary candidate after final validation.

    Rules:
    - Do not use PnL.
    - Do not use crisis-window hit rate.
    - Prefer configured primary if valid.
    - Else use configured fallback if valid.
    - Else choose lowest BIC among valid outputs.
    """
    valid = [
        output
        for output in outputs
        if output.passed
        and output.fit_result.has_fitted_model
        and np.isfinite(output.fit_result.bic)
    ]

    if not valid:
        return None

    if prefer_configured_primary:
        primary = [
            output
            for output in valid
            if output.fit_result.spec.is_primary_default
        ]
        if primary:
            return min(primary, key=lambda output: output.fit_result.bic)

    if allow_configured_fallback:
        fallback = [
            output
            for output in valid
            if output.fit_result.spec.is_fallback_default
        ]
        if fallback:
            return min(fallback, key=lambda output: output.fit_result.bic)

    return min(valid, key=lambda output: output.fit_result.bic)


def candidate_outputs_to_ranking_frame(
    outputs: Sequence[HMMCandidateOutput],
    *,
    selected_output: HMMCandidateOutput | None = None,
) -> pd.DataFrame:
    """
    Convert final candidate outputs to hmm_candidate_model_ranking.csv schema.
    """
    selected_key = None
    if selected_output is not None:
        selected_key = (
            selected_output.fit_result.market,
            selected_output.fit_result.spec.feature_set,
            selected_output.fit_result.spec.n_states,
            selected_output.fit_result.spec.covariance_type,
        )

    rows: list[dict[str, Any]] = []
    for output in outputs:
        key = (
            output.fit_result.market,
            output.fit_result.spec.feature_set,
            output.fit_result.spec.n_states,
            output.fit_result.spec.covariance_type,
        )
        selected_primary = bool(selected_key is not None and key == selected_key)
        row = _ranking_row_from_output(output, selected_primary=selected_primary)
        rows.append(row)

    frame = pd.DataFrame(rows)

    for col in HMM_CANDIDATE_RANKING_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    return frame.loc[:, list(HMM_CANDIDATE_RANKING_COLUMNS)]


def fit_and_build_hmm_candidate_output(
    scaled_panel: HMMScaledFeaturePanel,
    *,
    spec: HMMCandidateSpec,
    fit_config: HMMFitConfig | None = None,
    validation_rules: HMMValidationRules | None = None,
    include_smoothed_probabilities_diagnostic: bool = True,
) -> HMMCandidateOutput:
    """
    Convenience wrapper for one candidate:
    fit -> filter -> label -> validate -> output.
    """
    fit_result = fit_hmm_candidate(
        scaled_panel,
        spec=spec,
        fit_config=fit_config,
        validation_rules=validation_rules,
    )

    return build_hmm_candidate_output(
        fit_result,
        include_smoothed_probabilities_diagnostic=include_smoothed_probabilities_diagnostic,
        validation_rules=validation_rules,
    )


def fit_and_build_hmm_outputs_for_feature_panel(
    scaled_panel: HMMScaledFeaturePanel,
    *,
    n_states_values: Sequence[int] = (2, 3),
    covariance_types: Sequence[str] = ("diag", "full"),
    fit_config: HMMFitConfig | None = None,
    validation_rules: HMMValidationRules | None = None,
    include_smoothed_probabilities_diagnostic: bool = True,
) -> list[HMMCandidateOutput]:
    """
    Fit and build final outputs for all K/covariance candidates for one feature set.
    """
    fit_results = fit_hmm_candidates_for_feature_panel(
        scaled_panel,
        n_states_values=n_states_values,
        covariance_types=covariance_types,
        fit_config=fit_config,
        validation_rules=validation_rules,
    )

    return build_hmm_candidate_outputs(
        fit_results,
        include_smoothed_probabilities_diagnostic=include_smoothed_probabilities_diagnostic,
        validation_rules=validation_rules,
    )


def choose_candidate_by_diagnostics(
    results: Sequence[HMMCandidateFitResult],
    *,
    prefer_primary: bool = True,
    allow_fallback: bool = True,
) -> HMMCandidateFitResult | None:
    """
    Preliminary candidate chooser using fitting diagnostics only.

    Final selection should use choose_primary_hmm_output() after filtering and
    validation.
    """
    usable = [
        result
        for result in results
        if result.has_fitted_model
        and result.converged is not False
        and np.isfinite(result.train_loglik)
        and np.isfinite(result.test_loglik)
        and np.isfinite(result.bic)
    ]

    if not usable:
        return None

    if prefer_primary:
        primary = [result for result in usable if result.spec.is_primary_default]
        if primary:
            return min(primary, key=lambda result: result.bic)

    if allow_fallback:
        fallback = [result for result in usable if result.spec.is_fallback_default]
        if fallback:
            return min(fallback, key=lambda result: result.bic)

    return min(usable, key=lambda result: result.bic)


__all__ = [
    "HMMFitConfig",
    "HMMCandidateSpec",
    "HMMInitializationFit",
    "HMMCandidateFitResult",
    "HMMCandidateOutput",
    "make_hmm_candidate_spec",
    "make_gaussian_hmm_model",
    "approximate_gaussian_hmm_n_parameters",
    "fit_one_hmm_initialization",
    "select_best_initialization",
    "validate_fitted_hmm_parameters",
    "fit_hmm_candidate",
    "fit_hmm_candidates_for_feature_panel",
    "candidate_fit_results_to_ranking_frame",
    "compute_custom_filtered_probabilities",
    "compute_diagnostic_smoothed_probabilities",
    "append_raw_probability_columns",
    "build_hmm_candidate_output_panel",
    "build_hmm_candidate_output",
    "build_hmm_candidate_outputs",
    "choose_primary_hmm_output",
    "candidate_outputs_to_ranking_frame",
    "fit_and_build_hmm_candidate_output",
    "fit_and_build_hmm_outputs_for_feature_panel",
    "choose_candidate_by_diagnostics",
]
