from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class MSVolError(RuntimeError):
    """Raised when Python MS-volatility model processing fails."""


@dataclass(frozen=True)
class AR1PrefilterResult:
    residuals: np.ndarray
    fitted: np.ndarray
    intercept: float
    phi: float
    method: str
    status: str


@dataclass(frozen=True)
class MSVolFitResult:
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray | None
    state_variance_estimates: np.ndarray
    conditional_variance: np.ndarray
    fit_status: str
    convergence_status: str
    selected_spec: str
    probability_extraction_method: str
    loglike: float | None
    aic: float | None
    bic: float | None
    nobs: int


@dataclass(frozen=True)
class RunMarketResult:
    market: str
    input_csv: Path
    raw_output_csv: Path
    preflight_json: Path
    skip_report_json: Path
    model_summary_json: Path
    status: str
    skip_reason: str
    n_observations: int


REQUIRED_INPUT_COLUMNS = [
    "date",
    "market",
    "log_return",
    "return_for_msgarch",
    "source_return_column",
    "input_available",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_project_path(path_like: str | Path, project_root: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    file_path = Path(path)

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return out


def load_msvol_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path, project_root)
    if not config_path.exists():
        raise MSVolError(f"MSVOL config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise MSVolError(f"Invalid MSVOL config: {config_path}")

    return config


def normalize_market_arg(market: str, config: dict[str, Any]) -> list[str]:
    market = market.upper().strip()
    configured = set(config.get("markets", {}).keys())

    if market == "ALL":
        return sorted(configured)

    if market not in configured:
        allowed = sorted(configured | {"ALL"})
        raise MSVolError(f"Unknown market '{market}'. Allowed values: {allowed}")

    return [market]


def get_market_paths(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    market = market.upper().strip()
    market_cfg = config.get("markets", {}).get(market)

    if not isinstance(market_cfg, dict):
        raise MSVolError(f"Missing market config for {market}")

    required = [
        "input_csv",
        "raw_output_csv",
        "preflight_json",
        "skip_report_json",
        "model_summary_json",
    ]
    missing = [key for key in required if key not in market_cfg]
    if missing:
        raise MSVolError(f"Market {market} config missing key(s): {missing}")

    return {
        key: resolve_project_path(market_cfg[key], project_root)
        for key in required
    }


def read_msvol_input_csv(path: str | Path, expected_market: str | None = None) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise MSVolError(f"MSVOL input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise MSVolError(f"MSVOL input CSV missing required column(s): {missing}")

    out = df[REQUIRED_INPUT_COLUMNS].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(out["date"].isna().sum())
    if bad_dates:
        raise MSVolError(f"MSVOL input CSV contains {bad_dates} invalid date value(s).")

    out["market"] = out["market"].astype(str).str.upper()

    if expected_market is not None:
        expected_market = expected_market.upper()
        bad_market = out["market"] != expected_market
        bad_count = int(bad_market.sum())
        if bad_count:
            raise MSVolError(
                f"MSVOL input CSV contains {bad_count} row(s) not matching market {expected_market}."
            )

    out["log_return"] = pd.to_numeric(out["log_return"], errors="coerce")
    out["return_for_msgarch"] = pd.to_numeric(out["return_for_msgarch"], errors="coerce")

    bad_ret = (
        out["log_return"].isna()
        | out["return_for_msgarch"].isna()
        | ~np.isfinite(out["log_return"])
        | ~np.isfinite(out["return_for_msgarch"])
    )
    bad_ret_count = int(bad_ret.sum())
    if bad_ret_count:
        raise MSVolError(f"MSVOL input CSV contains {bad_ret_count} invalid return row(s).")

    out["input_available"] = out["input_available"].astype(bool)

    unavailable = int((~out["input_available"]).sum())
    if unavailable:
        raise MSVolError(f"MSVOL input CSV contains {unavailable} unavailable input row(s).")

    out = out.sort_values("date").reset_index(drop=True)

    duplicate_dates = int(out["date"].duplicated().sum())
    if duplicate_dates:
        raise MSVolError(f"MSVOL input CSV contains {duplicate_dates} duplicate date row(s).")

    return out


def prefilter_ar1(values: pd.Series | np.ndarray | list[float]) -> AR1PrefilterResult:
    x = np.asarray(values, dtype=float)

    if x.ndim != 1:
        raise MSVolError("AR(1) prefilter input must be one-dimensional.")

    if len(x) < 10:
        raise MSVolError("Need at least 10 observations for AR(1) prefiltering.")

    if not np.all(np.isfinite(x)):
        raise MSVolError("AR(1) prefilter input contains non-finite values.")

    y = x[1:]
    x_lag = x[:-1]
    design = np.column_stack([np.ones_like(x_lag), x_lag])

    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(beta[0])
        phi = float(beta[1])

        fitted = np.empty_like(x)
        fitted[0] = float(np.mean(x))
        fitted[1:] = design @ beta

        residuals = x - fitted

        if not np.all(np.isfinite(residuals)):
            raise FloatingPointError("AR(1) residuals contain non-finite values.")

        return AR1PrefilterResult(
            residuals=residuals,
            fitted=fitted,
            intercept=intercept,
            phi=phi,
            method="AR1",
            status="ok",
        )

    except Exception:
        mean = float(np.mean(x))
        fitted = np.full_like(x, mean)
        residuals = x - fitted

        return AR1PrefilterResult(
            residuals=residuals,
            fitted=fitted,
            intercept=mean,
            phi=0.0,
            method="demean_fallback",
            status="ar1_failed_used_demean",
        )


def _as_probability_matrix(obj: Any, nobs: int, k_regimes: int = 2) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy(dtype=float)
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy(dtype=float).reshape(-1, 1)
    else:
        arr = np.asarray(obj, dtype=float)

    if arr.ndim != 2:
        raise MSVolError(f"Probability object must be 2D. Got shape {arr.shape}.")

    if arr.shape == (k_regimes, nobs):
        arr = arr.T

    if arr.shape != (nobs, k_regimes):
        raise MSVolError(
            f"Expected probability shape {(nobs, k_regimes)} or {(k_regimes, nobs)}, got {arr.shape}."
        )

    validate_probability_matrix(arr)
    return arr


def validate_probability_matrix(prob: np.ndarray, atol: float = 1e-5) -> None:
    arr = np.asarray(prob, dtype=float)

    if arr.ndim != 2:
        raise MSVolError(f"Probability matrix must be 2D. Got shape {arr.shape}.")

    if arr.shape[1] != 2:
        raise MSVolError(f"Expected 2 probability columns. Got {arr.shape[1]}.")

    if not np.all(np.isfinite(arr)):
        raise MSVolError("Probability matrix contains non-finite values.")

    if np.any(arr < -atol) or np.any(arr > 1.0 + atol):
        raise MSVolError("Probability matrix contains values outside [0, 1].")

    row_sums = arr.sum(axis=1)
    bad = np.abs(row_sums - 1.0) > atol
    if np.any(bad):
        raise MSVolError(f"Probability matrix has {int(bad.sum())} row(s) not summing to 1.")


def _extract_metric(result: Any, name: str) -> float | None:
    value = getattr(result, name, None)
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def _extract_statsmodels_probabilities(result: Any, nobs: int) -> tuple[np.ndarray, np.ndarray | None, str]:
    filtered_obj = getattr(result, "filtered_marginal_probabilities", None)
    if filtered_obj is None:
        raise MSVolError("statsmodels result does not expose filtered_marginal_probabilities.")

    filtered = _as_probability_matrix(filtered_obj, nobs=nobs, k_regimes=2)

    smoothed_obj = getattr(result, "smoothed_marginal_probabilities", None)
    smoothed = None
    if smoothed_obj is not None:
        smoothed = _as_probability_matrix(smoothed_obj, nobs=nobs, k_regimes=2)

    return filtered, smoothed, "statsmodels_filtered_marginal_probabilities"


def _extract_variances_from_params(result: Any) -> np.ndarray | None:
    params = getattr(result, "params", None)
    if params is None:
        return None

    model = getattr(result, "model", None)
    param_names = getattr(model, "param_names", None)

    if param_names is None:
        return None

    values = np.asarray(params, dtype=float)
    if len(param_names) != len(values):
        return None

    variances: dict[int, float] = {}

    for name, value in zip(param_names, values):
        clean = str(name).lower().replace(" ", "")

        if "sigma2" not in clean:
            continue

        if "[0]" in clean or ".0" in clean or "0" == clean[-1:]:
            variances[0] = float(value)
        elif "[1]" in clean or ".1" in clean or "1" == clean[-1:]:
            variances[1] = float(value)

    if 0 in variances and 1 in variances:
        arr = np.array([variances[0], variances[1]], dtype=float)
        if np.all(np.isfinite(arr)) and np.all(arr > 0.0):
            return arr

    return None


def estimate_state_variances_from_probabilities(
    residuals: np.ndarray,
    filtered_probabilities: np.ndarray,
    min_weight: float = 1e-8,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    probs = np.asarray(filtered_probabilities, dtype=float)

    validate_probability_matrix(probs)

    if len(residuals) != probs.shape[0]:
        raise MSVolError("Residual/probability length mismatch while estimating state variances.")

    squared = residuals**2
    variances = []

    for state_idx in range(probs.shape[1]):
        weights = probs[:, state_idx]
        denom = float(weights.sum())

        if denom <= min_weight:
            variances.append(float(np.var(residuals)))
        else:
            variances.append(float(np.sum(weights * squared) / denom))

    arr = np.asarray(variances, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise MSVolError("Estimated state variances contain non-finite values.")

    arr = np.maximum(arr, 1e-12)
    return arr


def fit_msvol_markov_regression(
    residuals: np.ndarray,
    config: dict[str, Any],
) -> MSVolFitResult:
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except Exception as exc:
        raise MSVolError(
            "statsmodels MarkovRegression is unavailable. Install/repair statsmodels."
        ) from exc

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1:
        raise MSVolError("MSVOL residual input must be one-dimensional.")
    if not np.all(np.isfinite(residuals)):
        raise MSVolError("MSVOL residual input contains non-finite values.")

    spec = config.get("model_spec", {})
    fit_policy = config.get("fit_policy", {})

    k_regimes = int(spec.get("k_regimes", 2))
    if k_regimes != 2:
        raise MSVolError("Phase 8 Python MSVOL supports exactly 2 regimes.")

    trend = str(spec.get("trend", "c"))
    switching_variance = bool(spec.get("switching_variance", True))
    switching_trend = bool(spec.get("switching_trend", False))

    model = MarkovRegression(
        endog=residuals,
        k_regimes=2,
        trend=trend,
        switching_trend=switching_trend,
        switching_variance=switching_variance,
    )

    result = model.fit(
        maxiter=int(fit_policy.get("maxiter", 300)),
        em_iter=int(fit_policy.get("em_iter", 20)),
        search_reps=int(fit_policy.get("search_reps", 20)),
        search_iter=int(fit_policy.get("search_iter", 20)),
        disp=bool(fit_policy.get("disp", False)),
    )

    nobs = len(residuals)
    filtered, smoothed, prob_method = _extract_statsmodels_probabilities(result, nobs=nobs)

    state_variances = _extract_variances_from_params(result)
    if state_variances is None:
        state_variances = estimate_state_variances_from_probabilities(residuals, filtered)

    conditional_variance = filtered @ state_variances
    conditional_variance = np.maximum(conditional_variance, 1e-12)

    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    converged = bool(mle_retvals.get("converged", False))
    warnflag = mle_retvals.get("warnflag", None)

    if converged:
        convergence_status = "converged"
    elif warnflag is not None:
        convergence_status = f"not_converged_warnflag_{warnflag}"
    else:
        convergence_status = "unknown"

    return MSVolFitResult(
        filtered_probabilities=filtered,
        smoothed_probabilities=smoothed,
        state_variance_estimates=state_variances,
        conditional_variance=conditional_variance,
        fit_status="ok",
        convergence_status=convergence_status,
        selected_spec=(
            f"MarkovRegression/k_regimes=2/trend={trend}/"
            f"switching_trend={switching_trend}/"
            f"switching_variance={switching_variance}"
        ),
        probability_extraction_method=prob_method,
        loglike=_extract_metric(result, "llf"),
        aic=_extract_metric(result, "aic"),
        bic=_extract_metric(result, "bic"),
        nobs=nobs,
    )


def build_raw_output_frame(
    input_df: pd.DataFrame,
    fit_result: MSVolFitResult,
) -> pd.DataFrame:
    n = len(input_df)

    filtered = _as_probability_matrix(
        fit_result.filtered_probabilities,
        nobs=n,
        k_regimes=2,
    )

    smoothed = None
    if fit_result.smoothed_probabilities is not None:
        smoothed = _as_probability_matrix(
            fit_result.smoothed_probabilities,
            nobs=n,
            k_regimes=2,
        )

    state_vars = np.asarray(fit_result.state_variance_estimates, dtype=float)
    if state_vars.shape != (2,):
        raise MSVolError(f"Expected 2 state variance estimates. Got shape {state_vars.shape}.")
    if not np.all(np.isfinite(state_vars)) or np.any(state_vars <= 0.0):
        raise MSVolError("State variance estimates must be positive finite values.")

    cond_var = np.asarray(fit_result.conditional_variance, dtype=float)
    if cond_var.shape != (n,):
        raise MSVolError(
            f"Conditional variance length mismatch. Expected {n}, got {cond_var.shape}."
        )
    if not np.all(np.isfinite(cond_var)) or np.any(cond_var <= 0.0):
        raise MSVolError("Conditional variance must be positive finite values.")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(input_df["date"]).dt.strftime("%Y-%m-%d"),
            "market": input_df["market"].astype(str).str.upper(),
            "msvol_raw_state_0_prob_filtered": filtered[:, 0],
            "msvol_raw_state_1_prob_filtered": filtered[:, 1],
            "msvol_raw_state_0_variance_estimate": state_vars[0],
            "msvol_raw_state_1_variance_estimate": state_vars[1],
            "msvol_conditional_variance": cond_var,
            "msvol_conditional_volatility": np.sqrt(cond_var),
            "msvol_model_valid": True,
            "msvol_fit_status": fit_result.fit_status,
            "msvol_skip_reason": "",
        }
    )

    if smoothed is not None:
        out["msvol_raw_state_0_prob_smoothed_diagnostic"] = smoothed[:, 0]
        out["msvol_raw_state_1_prob_smoothed_diagnostic"] = smoothed[:, 1]

    return out


def build_preflight_payload(
    market: str,
    input_csv: Path,
    input_exists: bool,
    n_observations: int,
    status: str,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "market": market.upper(),
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "true_msgarch": False,
        "python_available": True,
        "r_required": False,
        "input_csv": str(input_csv),
        "input_csv_exists": bool(input_exists),
        "n_observations": int(n_observations),
        "selected_spec": "MarkovRegression/k_regimes=2/trend=c/switching_variance=True",
        "status": status,
        "skip_reason": skip_reason,
        "created_at_utc": utc_now_iso(),
    }


def build_skip_payload(
    market: str,
    input_csv: Path,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "market": market.upper(),
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "true_msgarch": False,
        "skipped": True,
        "fit_status": "skipped",
        "input_csv": str(input_csv),
        "skip_reason": skip_reason,
        "created_at_utc": utc_now_iso(),
    }


def build_model_summary_payload(
    market: str,
    input_csv: Path,
    raw_output_csv: Path,
    input_hash: str,
    prefilter: AR1PrefilterResult,
    fit_result: MSVolFitResult,
) -> dict[str, Any]:
    state_vars = fit_result.state_variance_estimates

    return {
        "market": market.upper(),
        "implementation": "PYTHON_STATSMODELS_MARKOV_REGRESSION",
        "true_msgarch": False,
        "active_model_label": "Python-only Markov-switching volatility robustness model",
        "input_csv": str(input_csv),
        "raw_output_csv": str(raw_output_csv),
        "input_hash_sha256": input_hash,
        "selected_spec": fit_result.selected_spec,
        "n_observations": int(fit_result.nobs),
        "fit_status": fit_result.fit_status,
        "convergence_status": fit_result.convergence_status,
        "probability_extraction_method": fit_result.probability_extraction_method,
        "loglike": fit_result.loglike,
        "aic": fit_result.aic,
        "bic": fit_result.bic,
        "prefilter_method": prefilter.method,
        "prefilter_status": prefilter.status,
        "ar1_intercept": prefilter.intercept,
        "ar1_phi": prefilter.phi,
        "state_0_variance_estimate": float(state_vars[0]),
        "state_1_variance_estimate": float(state_vars[1]),
        "lower_variance_raw_state": int(np.argmin(state_vars)),
        "higher_variance_raw_state": int(np.argmax(state_vars)),
        "report_note": (
            "This is not true MSGARCH. True MSGARCH remains optional and requires "
            "the R MSGARCH package. This Python model is used only as a volatility-regime "
            "robustness proxy."
        ),
        "created_at_utc": utc_now_iso(),
    }


def run_msvol_for_market(
    market: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
    allow_skip: bool = False,
) -> RunMarketResult:
    market = market.upper().strip()
    paths = get_market_paths(market, config, project_root)

    input_csv = paths["input_csv"]
    raw_output_csv = paths["raw_output_csv"]
    preflight_json = paths["preflight_json"]
    skip_report_json = paths["skip_report_json"]
    model_summary_json = paths["model_summary_json"]

    min_observations = int(config.get("input_policy", {}).get("min_observations", 1000))

    input_exists = input_csv.exists()
    if not input_exists:
        reason = f"MSVOL input CSV not found: {input_csv}"
        write_json(
            preflight_json,
            build_preflight_payload(
                market=market,
                input_csv=input_csv,
                input_exists=False,
                n_observations=0,
                status="skipped" if allow_skip else "failed",
                skip_reason=reason,
            ),
        )

        if allow_skip:
            write_json(skip_report_json, build_skip_payload(market, input_csv, reason))
            return RunMarketResult(
                market=market,
                input_csv=input_csv,
                raw_output_csv=raw_output_csv,
                preflight_json=preflight_json,
                skip_report_json=skip_report_json,
                model_summary_json=model_summary_json,
                status="skipped",
                skip_reason=reason,
                n_observations=0,
            )

        raise MSVolError(reason)

    try:
        input_df = read_msvol_input_csv(input_csv, expected_market=market)
        n_observations = int(len(input_df))

        if n_observations < min_observations:
            raise MSVolError(
                f"{market} MSVOL input has {n_observations} observations, "
                f"below min_observations={min_observations}."
            )

        write_json(
            preflight_json,
            build_preflight_payload(
                market=market,
                input_csv=input_csv,
                input_exists=True,
                n_observations=n_observations,
                status="ready",
                skip_reason="",
            ),
        )

        prefilter = prefilter_ar1(input_df["return_for_msgarch"].to_numpy(dtype=float))
        fit_result = fit_msvol_markov_regression(prefilter.residuals, config)

        raw_df = build_raw_output_frame(input_df=input_df, fit_result=fit_result)

        raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_output_csv, index=False)

        input_hash = sha256_file(input_csv)

        write_json(
            model_summary_json,
            build_model_summary_payload(
                market=market,
                input_csv=input_csv,
                raw_output_csv=raw_output_csv,
                input_hash=input_hash,
                prefilter=prefilter,
                fit_result=fit_result,
            ),
        )

        if skip_report_json.exists():
            skip_report_json.unlink()

        return RunMarketResult(
            market=market,
            input_csv=input_csv,
            raw_output_csv=raw_output_csv,
            preflight_json=preflight_json,
            skip_report_json=skip_report_json,
            model_summary_json=model_summary_json,
            status="ok",
            skip_reason="",
            n_observations=n_observations,
        )

    except Exception as exc:
        reason = str(exc)

        n_observations = 0
        try:
            n_observations = int(len(read_msvol_input_csv(input_csv, expected_market=market)))
        except Exception:
            n_observations = 0

        write_json(
            preflight_json,
            build_preflight_payload(
                market=market,
                input_csv=input_csv,
                input_exists=True,
                n_observations=n_observations,
                status="skipped" if allow_skip else "failed",
                skip_reason=reason,
            ),
        )

        if allow_skip:
            write_json(skip_report_json, build_skip_payload(market, input_csv, reason))
            return RunMarketResult(
                market=market,
                input_csv=input_csv,
                raw_output_csv=raw_output_csv,
                preflight_json=preflight_json,
                skip_report_json=skip_report_json,
                model_summary_json=model_summary_json,
                status="skipped",
                skip_reason=reason,
                n_observations=n_observations,
            )

        raise MSVolError(reason) from exc