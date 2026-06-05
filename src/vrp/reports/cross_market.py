from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
import pandas as pd
import yaml


class CrossMarketError(RuntimeError):
    """Base exception for Phase 13 cross-market analysis failures."""


class CrossMarketInputError(CrossMarketError):
    """Raised when required Phase 13 inputs are missing or invalid."""


class CrossMarketLeakageError(CrossMarketError):
    """Raised when same-date US information leaks into India prediction."""


class CrossMarketMutationError(CrossMarketError):
    """Raised when locked Phase 9/10 artifacts are mutated."""


@dataclass(frozen=True)
class ArtifactHashes:
    """Container for locked artifact file hashes."""

    hashes: dict[str, str | None]

    def changed_paths(self, other: "ArtifactHashes") -> list[str]:
        changed: list[str] = []
        all_paths = sorted(set(self.hashes) | set(other.hashes))
        for path in all_paths:
            if self.hashes.get(path) != other.hashes.get(path):
                changed.append(path)
        return changed


def load_cross_market_config(path: str | Path) -> dict[str, Any]:
    """
    Load Phase 13 cross-market configuration.

    Parameters
    ----------
    path:
        Path to configs/cross_market.yaml.

    Returns
    -------
    dict
        Parsed YAML config.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise CrossMarketInputError(f"Cross-market config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise CrossMarketInputError(f"Invalid config format: {config_path}")

    if config.get("phase") != "phase_13":
        raise CrossMarketInputError(
            f"Expected phase='phase_13' in {config_path}, got {config.get('phase')!r}"
        )

    return config


def _repo_path(path: str | Path, root: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if root is None:
        return p
    return Path(root) / p


def _date64ns(values: Any, name: str | None = None) -> pd.Series:
    s = pd.Series(values)
    dt = pd.to_datetime(s, errors="coerce")

    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert(None)

    dt = dt.dt.normalize()
    return pd.Series(
        dt.to_numpy(dtype="datetime64[ns]"),
        index=s.index,
        name=name,
    )


def _assign_date64ns(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = _date64ns(out[col], name=col).to_numpy(dtype="datetime64[ns]")
    return out


def _iter_config_paths(obj: Any) -> Iterable[str]:
    """
    Recursively yield string values that look like file paths.

    This is used only for validation and forbidden-input scanning.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_config_paths(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_config_paths(value)
    elif isinstance(obj, str):
        normalized = obj.replace("\\", "/")
        if (
            normalized.endswith(".parquet")
            or normalized.endswith(".csv")
            or normalized.endswith(".json")
            or normalized.endswith(".yaml")
            or normalized.endswith(".yml")
        ):
            yield obj


def _normalized_path_str(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _forbidden_input_set(config: Mapping[str, Any]) -> set[str]:
    forbidden: set[str] = set()
    for group_paths in config.get("forbidden_inputs", {}).values():
        if isinstance(group_paths, list):
            forbidden.update(_normalized_path_str(p) for p in group_paths)
    return forbidden


def _configured_non_policy_paths(config: Mapping[str, Any]) -> set[str]:
    scan_config = {
        key: value
        for key, value in config.items()
        if key not in {"forbidden_inputs", "forbidden_keywords"}
    }
    return {_normalized_path_str(p) for p in _iter_config_paths(scan_config)}


def validate_no_forbidden_phase11_inputs(config: Mapping[str, Any]) -> None:
    """
    Fail if any configured Phase 13 input points to forbidden Phase 11 artifacts.

    This is a config-level check. Runtime read guards are implemented separately
    in guarded_read_parquet / guarded_read_csv.
    """
    forbidden = _forbidden_input_set(config)
    configured_paths = _configured_non_policy_paths(config)

    intersection = sorted(configured_paths & forbidden)
    if intersection:
        raise CrossMarketInputError(
            "Forbidden Phase 11 artifact configured as Phase 13 input: "
            + ", ".join(intersection)
        )


def validate_cross_market_inputs(
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    require_exists: bool = True,
) -> None:
    """
    Validate required Phase 13 input paths and model configuration.

    Parameters
    ----------
    config:
        Parsed cross-market config.
    root:
        Optional repo root used to resolve relative paths.
    require_exists:
        If True, fail when required input files do not exist.
    """
    validate_no_forbidden_phase11_inputs(config)

    input_files = config.get("input_files")
    if not isinstance(input_files, dict):
        raise CrossMarketInputError("Missing or invalid 'input_files' section.")

    for market in ("US", "INDIA"):
        if market not in input_files:
            raise CrossMarketInputError(f"Missing input_files.{market} section.")

        market_inputs = input_files[market]
        if not isinstance(market_inputs, dict):
            raise CrossMarketInputError(f"Invalid input_files.{market} section.")

        required_keys = {
            "vrp_har",
            "threshold",
            "gaussian_hmm",
            "markov_autoreg",
            "strategy_signals",
            "backtest",
        }
        missing_keys = sorted(required_keys - set(market_inputs))
        if missing_keys:
            raise CrossMarketInputError(
                f"Missing input_files.{market} keys: {missing_keys}"
            )

        if require_exists:
            for key in required_keys:
                path = _repo_path(market_inputs[key], root)
                if not path.exists():
                    raise CrossMarketInputError(
                        f"Required Phase 13 input missing: {market}.{key} -> {path}"
                    )

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise CrossMarketInputError("Config must define non-empty 'models' list.")

    supported = {"gaussian_hmm", "markov_autoreg"}
    unknown = sorted(set(models) - supported)
    if unknown:
        raise CrossMarketInputError(f"Unsupported Phase 13 model(s): {unknown}")

    alignment = config.get("alignment", {})
    if alignment.get("method") != "asof_backward_strict":
        raise CrossMarketInputError(
            "Phase 13 requires alignment.method='asof_backward_strict'."
        )
    if alignment.get("allow_exact_matches") is not False:
        raise CrossMarketInputError(
            "Phase 13 requires alignment.allow_exact_matches=false."
        )
    if alignment.get("require_us_lagged_date_lt_india_date") is not True:
        raise CrossMarketInputError(
            "Phase 13 requires us_lagged_date < india_date."
        )


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str | None:
    """
    Compute SHA256 hash for a file.

    Returns None if the file does not exist. This allows optional locked artifacts
    to be tracked without fabricating hashes.
    """
    p = Path(path)
    if not p.exists():
        return None
    if not p.is_file():
        raise CrossMarketInputError(f"Expected file path, got non-file: {p}")

    digest = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _locked_artifact_paths(config: Mapping[str, Any]) -> list[str]:
    locked = config.get("locked_artifacts", {})
    if not isinstance(locked, dict):
        return []

    paths: list[str] = []
    for group_name in ("phase9", "phase10"):
        group = locked.get(group_name, [])
        if isinstance(group, list):
            paths.extend(str(p) for p in group)
    return paths


def collect_locked_artifact_hashes(
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> ArtifactHashes:
    """
    Hash all locked Phase 9/10 artifacts.
    """
    hashes: dict[str, str | None] = {}
    for rel_path in _locked_artifact_paths(config):
        abs_path = _repo_path(rel_path, root)
        hashes[_normalized_path_str(rel_path)] = hash_file(abs_path)
    return ArtifactHashes(hashes=hashes)


def hash_locked_artifacts_before_after(
    before: ArtifactHashes,
    after: ArtifactHashes,
) -> None:
    """
    Fail if any locked Phase 9/10 artifact changed.
    """
    changed = before.changed_paths(after)
    if changed:
        raise CrossMarketMutationError(
            "Locked Phase 9/10 artifacts changed during Phase 13 run: "
            + ", ".join(changed)
        )


def _assert_not_forbidden_path(
    path: str | Path,
    config: Mapping[str, Any],
) -> None:
    normalized = _normalized_path_str(path)
    forbidden = _forbidden_input_set(config)

    if normalized in forbidden:
        raise CrossMarketInputError(
            f"Attempted to read forbidden Phase 11 artifact: {normalized}"
        )

    forbidden_keywords = config.get("forbidden_keywords", [])
    if isinstance(forbidden_keywords, list):
        lowered = normalized.lower()
        matched = [kw for kw in forbidden_keywords if str(kw).lower() in lowered]
        if matched:
            raise CrossMarketInputError(
                f"Attempted to read forbidden Phase 11/broker-like path: "
                f"{normalized}. Matched keyword(s): {matched}"
            )


def guarded_read_parquet(
    path: str | Path,
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read parquet while blocking forbidden Phase 11/broker artifacts.
    """
    _assert_not_forbidden_path(path, config)
    abs_path = _repo_path(path, root)
    if not abs_path.exists():
        raise CrossMarketInputError(f"Parquet input not found: {abs_path}")
    return pd.read_parquet(abs_path)


def guarded_read_csv(
    path: str | Path,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read CSV while blocking forbidden Phase 11/broker artifacts.
    """
    _assert_not_forbidden_path(path, config)
    abs_path = _repo_path(path, root)
    if not abs_path.exists():
        raise CrossMarketInputError(f"CSV input not found: {abs_path}")
    return pd.read_csv(abs_path, **kwargs)


def _coerce_date_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()

    if target_col not in out.columns:
        candidates = [
            "date",
            "Date",
            "datetime",
            "Datetime",
            "timestamp",
            "Timestamp",
            "trade_date",
            "session_date",
        ]
        matched = [c for c in candidates if c in out.columns]
        if not matched:
            if out.index.name in candidates or isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index()
                if out.columns[0] != target_col:
                    out = out.rename(columns={out.columns[0]: target_col})
            else:
                raise CrossMarketInputError(
                    f"Could not identify date column for target {target_col!r}. "
                    f"Available columns: {list(out.columns)}"
                )
        else:
            out = out.rename(columns={matched[0]: target_col})

    out[target_col] = _date64ns(
        out[target_col],
        name=target_col,
    ).to_numpy(dtype="datetime64[ns]")
    if out[target_col].isna().any():
        n_bad = int(out[target_col].isna().sum())
        raise CrossMarketInputError(
            f"{target_col} contains {n_bad} invalid or missing date values."
        )

    out = out.sort_values(target_col).drop_duplicates(target_col, keep="last")
    return out


def _first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool,
    logical_name: str,
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise CrossMarketInputError(
            f"Missing required column for {logical_name}. "
            f"Tried candidates={list(candidates)}. "
            f"Available columns={list(df.columns)}"
        )
    return None


def _numeric_series(df: pd.DataFrame, col: str, logical_name: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        raise CrossMarketInputError(
            f"Column {col!r} for {logical_name} has no numeric observations."
        )
    return values


def _infer_vrp_column(df: pd.DataFrame, market_lower: str) -> str:
    candidates = [
        f"{market_lower}_vrp_har_gk",
        "vrp_har_gk",
        "vrp_gk",
        "vrp",
        "har_vrp_gk",
        "forward_vrp_har_gk",
        "vrp_forecast_gk",
    ]
    return cast(
        str,
        _first_existing_column(
            df,
            candidates,
            required=True,
            logical_name=f"{market_lower} VRP",
        ),
    )


def _infer_iv_column(df: pd.DataFrame, market_lower: str) -> str:
    candidates = [
        f"{market_lower}_iv_ann",
        "iv_ann",
        "implied_vol_ann",
        "vix_ann",
        "india_vix_ann",
        "vix_close",
        "india_vix_close",
        "iv",
    ]
    return cast(
        str,
        _first_existing_column(
            df,
            candidates,
            required=True,
            logical_name=f"{market_lower} implied vol",
        ),
    )


def _infer_rv_column(df: pd.DataFrame, market_lower: str) -> str:
    candidates = [
        f"{market_lower}_rv_gk_22d_ann_lag1",
        "rv_gk_22d_ann_lag1",
        "rv_gk_22d_ann",
        "garman_klass_22d_ann_lag1",
        "garman_klass_22d_ann",
        "rv_gk",
        "realized_vol_ann",
    ]
    return cast(
        str,
        _first_existing_column(
            df,
            candidates,
            required=True,
            logical_name=f"{market_lower} realized vol",
        ),
    )


def _infer_model_columns(
    df: pd.DataFrame,
    market_lower: str,
    model: str,
    config: Mapping[str, Any],
) -> tuple[str, str | None]:
    model_sources = config.get("model_column_sources", {}).get(model, {})
    prob_candidate = model_sources.get("stress_probability")
    state_candidate = model_sources.get("state_name")

    probability_candidates = [
        c
        for c in [
            f"{market_lower}_stress_prob",
            prob_candidate,
            "stress_prob",
            "stress_probability",
            "filtered_prob_stress_for_next_session",
            "prob_stress_for_next_session",
            "hmm_filtered_prob_stress_for_next_session",
            "mar_filtered_prob_stress_for_next_session",
        ]
        if c
    ]

    state_candidates = [
        c
        for c in [
            f"{market_lower}_state_name",
            state_candidate,
            "state_name",
            "regime_state_name",
            "state_label",
            "hmm_state_name_for_next_session",
            "mar_state_name_for_next_session",
        ]
        if c
    ]

    prob_col = _first_existing_column(
        df,
        probability_candidates,
        required=True,
        logical_name=f"{market_lower} {model} stress probability",
    )
    state_col = _first_existing_column(
        df,
        state_candidates,
        required=False,
        logical_name=f"{market_lower} {model} state name",
    )
    return cast(str, prob_col), state_col


def _merge_on_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    left = _coerce_date_column(left, date_col)
    right = _coerce_date_column(right, date_col)

    overlap = [c for c in right.columns if c in left.columns and c != date_col]
    if overlap:
        right = right.drop(columns=overlap)

    return left.merge(right, on=date_col, how="inner")


def load_market_panels(
    config: Mapping[str, Any],
    model: str,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine VRP/HAR panel with requested regime panel for both markets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Raw merged US panel and raw merged India panel.
    """
    if model not in {"gaussian_hmm", "markov_autoreg"}:
        raise CrossMarketInputError(f"Unsupported model: {model}")

    inputs = config.get("input_files", {})

    us_vrp = guarded_read_parquet(inputs["US"]["vrp_har"], config, root)
    us_regime = guarded_read_parquet(inputs["US"][model], config, root)
    india_vrp = guarded_read_parquet(inputs["INDIA"]["vrp_har"], config, root)
    india_regime = guarded_read_parquet(inputs["INDIA"][model], config, root)

    us_raw = _merge_on_date(us_vrp, us_regime, "date")
    india_raw = _merge_on_date(india_vrp, india_regime, "date")

    us_raw["phase13_model"] = model
    india_raw["phase13_model"] = model

    return us_raw, india_raw


def _stress_from_state_name(
    state: pd.Series,
    config: Mapping[str, Any],
) -> pd.Series:
    patterns = config.get("stress_definition", {}).get("stress_state_name_patterns", [])
    if not patterns:
        patterns = ["stress", "high_vol", "high vol", "crisis", "volatile"]

    text = state.astype("string").str.lower()
    out = pd.Series(False, index=state.index)
    for pattern in patterns:
        out = out | text.str.contains(str(pattern).lower(), regex=False, na=False)
    return out.astype(float)


def normalize_market_columns(
    df: pd.DataFrame,
    market: str,
    model: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Normalize market/model panel columns for Phase 13.

    Normalized US columns:
        us_date, us_vrp_har_gk, us_iv_ann, us_rv_gk_22d_ann_lag1,
        us_stress_prob, us_state_name

    Normalized India columns:
        india_date, india_vrp_har_gk, india_iv_ann, india_rv_gk_22d_ann_lag1,
        india_stress_prob, india_state_name
    """
    market_upper = market.upper()
    if market_upper not in {"US", "INDIA"}:
        raise CrossMarketInputError(f"market must be US or INDIA, got {market!r}")

    prefix = "us" if market_upper == "US" else "india"
    date_col = f"{prefix}_date"

    src = _coerce_date_column(df, date_col)

    vrp_col = _infer_vrp_column(src, prefix)
    iv_col = _infer_iv_column(src, prefix)
    rv_col = _infer_rv_column(src, prefix)
    prob_col, state_col = _infer_model_columns(src, prefix, model, config)

    out = pd.DataFrame(
        {
            date_col: src[date_col],
            f"{prefix}_vrp_har_gk": _numeric_series(src, vrp_col, f"{prefix} VRP"),
            f"{prefix}_iv_ann": _numeric_series(src, iv_col, f"{prefix} IV"),
            f"{prefix}_rv_gk_22d_ann_lag1": _numeric_series(
                src, rv_col, f"{prefix} RV"
            ),
            f"{prefix}_stress_prob": _numeric_series(
                src, prob_col, f"{prefix} stress probability"
            ),
        }
    )

    if state_col is not None:
        out[f"{prefix}_state_name"] = src[state_col].astype("string")
    else:
        threshold = float(
            config.get("stress_definition", {}).get("probability_threshold", 0.50)
        )
        out[f"{prefix}_state_name"] = np.where(
            out[f"{prefix}_stress_prob"] >= threshold,
            "stress",
            "non_stress",
        )

    fallback_to_state_name = bool(
        config.get("stress_definition", {}).get("fallback_to_state_name", True)
    )
    if fallback_to_state_name:
        stress_prob = out[f"{prefix}_stress_prob"]
        missing_prob = stress_prob.isna()
        if missing_prob.any():
            inferred = _stress_from_state_name(out[f"{prefix}_state_name"], config)
            out.loc[missing_prob, f"{prefix}_stress_prob"] = inferred.loc[missing_prob]

    out["model"] = model
    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last")

    return out


def previous_us_trading_date_for_india(
    india_dates: Iterable[Any],
    us_dates: Iterable[Any],
) -> pd.DataFrame:
    """
    For each India date, select the latest US observation date strictly before it.

    This is the core Phase 13 timezone rule:
        us_lagged_date < india_date

    Returns
    -------
    pd.DataFrame
        Columns:
            india_date
            us_lagged_date
            lag_calendar_days
            lag_is_strictly_prior
    """
    india = pd.DataFrame(
        {"india_date": _date64ns(list(india_dates), name="india_date")}
    )
    us = pd.DataFrame(
        {"us_lagged_date": _date64ns(list(us_dates), name="us_lagged_date")}
    )

    india = india.dropna().drop_duplicates().sort_values("india_date")
    us = us.dropna().drop_duplicates().sort_values("us_lagged_date")

    india["india_date"] = india["india_date"].astype("datetime64[ns]")
    us["us_lagged_date"] = us["us_lagged_date"].astype("datetime64[ns]")

    if india.empty:
        raise CrossMarketInputError("No India dates supplied for alignment.")
    if us.empty:
        raise CrossMarketInputError("No US dates supplied for alignment.")

    aligned = pd.merge_asof(
        india,
        us,
        left_on="india_date",
        right_on="us_lagged_date",
        direction="backward",
        allow_exact_matches=False,
    )

    aligned["lag_calendar_days"] = (
        aligned["india_date"] - aligned["us_lagged_date"]
    ).dt.days
    aligned["lag_is_strictly_prior"] = aligned["us_lagged_date"] < aligned["india_date"]

    return aligned


def align_us_india_predictive_panel(
    us_df: pd.DataFrame,
    india_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build predictive US-India alignment.

    Every row is keyed by india_date. US data is joined from the latest US date
    strictly before the India date. Same-date matching is impossible by construction.
    """
    required_us = {"us_date"}
    required_india = {"india_date"}
    if not required_us.issubset(us_df.columns):
        raise CrossMarketInputError(f"US panel missing required columns: {required_us}")
    if not required_india.issubset(india_df.columns):
        raise CrossMarketInputError(
            f"India panel missing required columns: {required_india}"
        )

    us = us_df.copy()
    india = india_df.copy()

    us["us_date"] = _date64ns(
        us["us_date"],
        name="us_date",
    ).to_numpy(dtype="datetime64[ns]")
    india["india_date"] = _date64ns(
        india["india_date"],
        name="india_date",
    ).to_numpy(dtype="datetime64[ns]")

    us = us.sort_values("us_date").drop_duplicates("us_date", keep="last")
    india = india.sort_values("india_date").drop_duplicates("india_date", keep="last")

    alignment = previous_us_trading_date_for_india(
        india_dates=india["india_date"],
        us_dates=us["us_date"],
    )

    us_lagged = us.rename(columns={"us_date": "us_lagged_date"})
    us_lagged["us_lagged_date"] = us_lagged["us_lagged_date"].astype(
        "datetime64[ns]"
    )

    panel = india.merge(alignment, on="india_date", how="left")
    panel["india_date"] = panel["india_date"].astype("datetime64[ns]")
    panel["us_lagged_date"] = panel["us_lagged_date"].astype("datetime64[ns]")
    panel = panel.merge(us_lagged, on="us_lagged_date", how="left")

    panel["lag_calendar_days"] = (
        panel["india_date"] - panel["us_lagged_date"]
    ).dt.days
    panel["lag_is_strictly_prior"] = panel["us_lagged_date"] < panel["india_date"]

    assert_no_same_date_us_leakage(panel)

    max_lag = config.get("alignment", {}).get("max_lag_calendar_days")
    if max_lag is not None:
        stale = panel["lag_calendar_days"].dropna() > int(max_lag)
        if stale.any():
            n_stale = int(stale.sum())
            raise CrossMarketLeakageError(
                f"Predictive panel contains {n_stale} stale US lag(s) "
                f"beyond max_lag_calendar_days={max_lag}."
            )

    return panel.sort_values("india_date").reset_index(drop=True)


def assert_no_same_date_us_leakage(panel: pd.DataFrame) -> None:
    """
    Assert that predictive rows never use US same-date information.

    Required invariant:
        us_lagged_date < india_date
    """
    required = {"india_date", "us_lagged_date"}
    if not required.issubset(panel.columns):
        raise CrossMarketLeakageError(
            f"Predictive panel missing required leakage-audit columns: {required}"
        )

    audit = panel.copy()
    audit["india_date"] = _date64ns(
        audit["india_date"],
        name="india_date",
    ).to_numpy(dtype="datetime64[ns]")
    audit["us_lagged_date"] = _date64ns(
        audit["us_lagged_date"],
        name="us_lagged_date",
    ).to_numpy(dtype="datetime64[ns]")

    matched = audit["us_lagged_date"].notna()
    violations = matched & (audit["us_lagged_date"] >= audit["india_date"])

    if violations.any():
        sample = audit.loc[
            violations,
            ["india_date", "us_lagged_date"],
        ].head(10)
        raise CrossMarketLeakageError(
            "Same-date or future US leakage detected in predictive panel. "
            f"Sample violations: {sample.to_dict(orient='records')}"
        )


def build_cross_market_no_lookahead_audit(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    model: str | None = None,
) -> pd.DataFrame:
    """
    Build no-lookahead audit table for a predictive panel.
    """
    required_cols = {"india_date", "us_lagged_date", "lag_calendar_days"}
    missing = sorted(required_cols - set(panel.columns))
    if missing:
        raise CrossMarketLeakageError(
            f"Cannot build no-lookahead audit. Missing columns: {missing}"
        )

    work = panel.copy()
    work["india_date"] = _date64ns(
        work["india_date"],
        name="india_date",
    ).to_numpy(dtype="datetime64[ns]")
    work["us_lagged_date"] = _date64ns(
        work["us_lagged_date"],
        name="us_lagged_date",
    ).to_numpy(dtype="datetime64[ns]")

    matched = work["us_lagged_date"].notna()
    same_date_or_future = matched & (work["us_lagged_date"] >= work["india_date"])

    lag_days = pd.to_numeric(work["lag_calendar_days"], errors="coerce")
    stale_warning_days = int(
        config.get("alignment", {}).get("stale_lag_warning_calendar_days", 3)
    )
    max_lag_days = config.get("alignment", {}).get("max_lag_calendar_days", 7)
    max_lag_days_int = int(max_lag_days) if max_lag_days is not None else 7

    forbidden_configured = sorted(
        _configured_non_policy_paths(config) & _forbidden_input_set(config)
    )

    audit = pd.DataFrame(
        [
            {
                "model": model or (
                    str(work["model"].iloc[0])
                    if "model" in work.columns and len(work)
                    else ""
                ),
                "n_rows": int(len(work)),
                "n_missing_india_date": int(work["india_date"].isna().sum()),
                "n_missing_us_lagged_date": int(work["us_lagged_date"].isna().sum()),
                "n_same_date_or_future_us_violations": int(same_date_or_future.sum()),
                "min_lag_calendar_days": (
                    float(lag_days.min()) if lag_days.notna().any() else np.nan
                ),
                "median_lag_calendar_days": (
                    float(lag_days.median()) if lag_days.notna().any() else np.nan
                ),
                "max_lag_calendar_days": (
                    float(lag_days.max()) if lag_days.notna().any() else np.nan
                ),
                "n_lag_gt_stale_warning_days": int(
                    (lag_days > stale_warning_days).sum()
                ),
                "stale_warning_days": stale_warning_days,
                "n_lag_gt_max_lag_days": int((lag_days > max_lag_days_int).sum()),
                "max_lag_days": max_lag_days_int,
                "forbidden_phase11_inputs_configured": bool(forbidden_configured),
                "forbidden_phase11_input_paths": json.dumps(forbidden_configured),
                "passes_no_lookahead": bool(
                    int(same_date_or_future.sum()) == 0
                    and not bool(forbidden_configured)
                ),
            }
        ]
    )

    if int(same_date_or_future.sum()) > 0:
        raise CrossMarketLeakageError(
            "No-lookahead audit failed: same-date/future US observations found."
        )

    if forbidden_configured:
        raise CrossMarketInputError(
            "No-lookahead audit failed: forbidden Phase 11 inputs configured."
        )

    return audit

def _ensure_parent_dir(path: str | Path, root: str | Path | None = None) -> Path:
    out_path = _repo_path(path, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _safe_to_parquet(
    df: pd.DataFrame,
    path: str | Path,
    root: str | Path | None = None,
    *,
    index: bool = False,
) -> Path:
    out_path = _ensure_parent_dir(path, root)
    df.to_parquet(out_path, index=index)
    return out_path


def _safe_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    root: str | Path | None = None,
    *,
    index: bool = False,
) -> Path:
    out_path = _ensure_parent_dir(path, root)
    df.to_csv(out_path, index=index)
    return out_path


def _standardize_model_columns(panel: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Normalize accidental model_x/model_y columns after merges.
    """
    out = panel.copy()

    if "model" not in out.columns:
        if "model_x" in out.columns:
            out["model"] = out["model_x"]
        elif "model_y" in out.columns:
            out["model"] = out["model_y"]
        else:
            out["model"] = model

    for col in ("model_x", "model_y"):
        if col in out.columns:
            out = out.drop(columns=col)

    out["model"] = out["model"].fillna(model).astype(str)
    return out


def load_normalized_market_panels(
    config: Mapping[str, Any],
    model: str,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw Phase 13 inputs and return normalized US and India panels.
    """
    us_raw, india_raw = load_market_panels(config=config, model=model, root=root)

    us_df = normalize_market_columns(
        us_raw,
        market="US",
        model=model,
        config=config,
    )
    india_df = normalize_market_columns(
        india_raw,
        market="INDIA",
        model=model,
        config=config,
    )

    return us_df, india_df


def build_descriptive_same_date_panel(
    us_df: pd.DataFrame,
    india_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build same-date US-India panel for descriptive correlations only.

    This panel is not allowed for predictive India modelling or overlays.
    """
    required_us = {
        "us_date",
        "us_vrp_har_gk",
        "us_iv_ann",
        "us_rv_gk_22d_ann_lag1",
        "us_stress_prob",
        "us_state_name",
    }
    required_india = {
        "india_date",
        "india_vrp_har_gk",
        "india_iv_ann",
        "india_rv_gk_22d_ann_lag1",
        "india_stress_prob",
        "india_state_name",
    }

    missing_us = sorted(required_us - set(us_df.columns))
    missing_india = sorted(required_india - set(india_df.columns))
    if missing_us:
        raise CrossMarketInputError(
            f"US descriptive panel missing columns: {missing_us}"
        )
    if missing_india:
        raise CrossMarketInputError(
            f"India descriptive panel missing columns: {missing_india}"
        )

    us = us_df.copy()
    india = india_df.copy()

    us["date"] = _date64ns(
        us["us_date"],
        name="date",
    ).to_numpy(dtype="datetime64[ns]")
    india["date"] = _date64ns(
        india["india_date"],
        name="date",
    ).to_numpy(dtype="datetime64[ns]")

    us = us.sort_values("date").drop_duplicates("date", keep="last")
    india = india.sort_values("date").drop_duplicates("date", keep="last")

    us_model = us["model"].iloc[0] if "model" in us.columns and len(us) else ""
    india_model = (
        india["model"].iloc[0] if "model" in india.columns and len(india) else ""
    )
    model = str(us_model or india_model)

    if "model" in us.columns:
        us = us.drop(columns=["model"])
    if "model" in india.columns:
        india = india.drop(columns=["model"])

    panel = us.merge(india, on="date", how="inner")
    panel["model"] = model
    panel["panel_type"] = "descriptive_same_date"
    panel["same_date_descriptive_only"] = True
    panel["predictive_allowed"] = False

    ordered_front = [
        "model",
        "panel_type",
        "date",
        "us_date",
        "india_date",
        "same_date_descriptive_only",
        "predictive_allowed",
    ]
    ordered_front = [c for c in ordered_front if c in panel.columns]
    remaining = [c for c in panel.columns if c not in ordered_front]

    return panel[ordered_front + remaining].sort_values("date").reset_index(drop=True)


def create_lagged_us_features_for_india(
    panel: pd.DataFrame,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Create lagged-US feature aliases for India predictive modelling.

    Important:
    - The actual cross-market lag is carried by us_lagged_date < india_date.
    - This function adds explicit feature names expected by config.
    - The RV input is already named rv_gk_22d_ann_lag1 by earlier phases,
      so it is preserved under the same name.
    """
    if lag_days != 1:
        raise CrossMarketInputError(
            "Phase 13 currently supports lag_days=1 only because alignment is "
            "previous US trading date strictly before India date."
        )

    required = {
        "india_date",
        "us_lagged_date",
        "us_stress_prob",
        "us_vrp_har_gk",
        "us_iv_ann",
        "us_rv_gk_22d_ann_lag1",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise CrossMarketInputError(
            f"Cannot create US lag features. Missing: {missing}"
        )

    out = panel.copy()

    out["us_stress_prob_lag1"] = pd.to_numeric(out["us_stress_prob"], errors="coerce")
    out["us_vrp_har_gk_lag1"] = pd.to_numeric(out["us_vrp_har_gk"], errors="coerce")
    out["us_iv_ann_lag1"] = pd.to_numeric(out["us_iv_ann"], errors="coerce")

    # Preserve the Phase 13 config name. This variable is already lagged by
    # construction in the prior feature pipeline and by strict US<India alignment.
    out["us_rv_gk_22d_ann_lag1"] = pd.to_numeric(
        out["us_rv_gk_22d_ann_lag1"],
        errors="coerce",
    )

    return out


def create_lagged_india_local_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create India local lag features using only prior India rows.

    Current India stress is the dependent variable candidate.
    India VRP, IV, and stress probability lag1 are shifted by one India row.
    India RV column rv_gk_22d_ann_lag1 is already a lagged RV feature from
    earlier phases and is preserved without an additional shift.
    """
    required = {
        "india_date",
        "india_vrp_har_gk",
        "india_iv_ann",
        "india_rv_gk_22d_ann_lag1",
        "india_stress_prob",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise CrossMarketInputError(
            f"Cannot create India local lag features. Missing: {missing}"
        )

    out = panel.copy()
    out["india_date"] = _date64ns(
        out["india_date"],
        name="india_date",
    ).to_numpy(dtype="datetime64[ns]")
    out = out.sort_values("india_date").reset_index(drop=True)

    out["india_vrp_har_gk_lag1"] = pd.to_numeric(
        out["india_vrp_har_gk"],
        errors="coerce",
    ).shift(1)
    out["india_iv_ann_lag1"] = pd.to_numeric(
        out["india_iv_ann"],
        errors="coerce",
    ).shift(1)
    out["india_stress_prob_lag1"] = pd.to_numeric(
        out["india_stress_prob"],
        errors="coerce",
    ).shift(1)

    # Preserve exact config name.
    out["india_rv_gk_22d_ann_lag1"] = pd.to_numeric(
        out["india_rv_gk_22d_ann_lag1"],
        errors="coerce",
    )

    return out


def add_india_stress_indicator(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Add binary India stress indicator for logistic diagnostics.
    """
    if "india_stress_prob" not in panel.columns:
        raise CrossMarketInputError("Missing india_stress_prob for stress indicator.")

    threshold = float(
        config.get("stress_definition", {}).get("probability_threshold", 0.50)
    )

    out = panel.copy()
    out["india_stress_indicator"] = (
        pd.to_numeric(out["india_stress_prob"], errors="coerce") >= threshold
    ).astype(float)

    return out


def build_predictive_panel(
    us_df: pd.DataFrame,
    india_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Build the India predictive panel with strict lagged US information.

    Invariant:
        us_lagged_date < india_date

    Same-date US data is not allowed in this panel.
    """
    model = ""
    if "model" in india_df.columns and len(india_df):
        model = str(india_df["model"].iloc[0])
    elif "model" in us_df.columns and len(us_df):
        model = str(us_df["model"].iloc[0])

    panel = align_us_india_predictive_panel(
        us_df=us_df,
        india_df=india_df,
        config=config,
    )
    panel = _standardize_model_columns(panel, model=model)
    panel["panel_type"] = "predictive_lagged"
    panel["same_date_descriptive_only"] = False
    panel["predictive_allowed"] = True

    panel = create_lagged_us_features_for_india(panel, lag_days=lag_days)
    panel = create_lagged_india_local_features(panel)
    panel = add_india_stress_indicator(panel, config=config)

    assert_no_same_date_us_leakage(panel)

    ordered_front = [
        "model",
        "panel_type",
        "india_date",
        "us_lagged_date",
        "lag_calendar_days",
        "lag_is_strictly_prior",
        "same_date_descriptive_only",
        "predictive_allowed",
        "india_stress_indicator",
    ]
    ordered_front = [c for c in ordered_front if c in panel.columns]
    remaining = [c for c in panel.columns if c not in ordered_front]

    return panel[ordered_front + remaining].sort_values("india_date").reset_index(
        drop=True
    )


def build_combined_cross_market_panel(
    descriptive_panel: pd.DataFrame,
    predictive_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a convenience panel containing both descriptive and predictive rows.

    This is intentionally tagged by panel_type to avoid ambiguity.
    The separate descriptive and predictive parquet outputs remain the source
    of truth.
    """
    desc = descriptive_panel.copy()
    pred = predictive_panel.copy()

    desc["panel_type"] = "descriptive_same_date"
    desc["same_date_descriptive_only"] = True
    desc["predictive_allowed"] = False

    pred["panel_type"] = "predictive_lagged"
    pred["same_date_descriptive_only"] = False
    pred["predictive_allowed"] = True

    all_columns = sorted(set(desc.columns) | set(pred.columns))
    combined = pd.concat(
        [desc, pred],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    combined = combined.reindex(columns=all_columns)

    front = [
        "model",
        "panel_type",
        "date",
        "india_date",
        "us_date",
        "us_lagged_date",
        "lag_calendar_days",
        "lag_is_strictly_prior",
        "same_date_descriptive_only",
        "predictive_allowed",
    ]
    front = [c for c in front if c in combined.columns]
    remaining = [c for c in combined.columns if c not in front]

    return combined[front + remaining]


def build_alignment_audit(
    panel: pd.DataFrame,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Build required alignment coverage and stale-lag audit.

    Required failure condition:
        n_same_date_violations > 0
    """
    required = {"india_date", "us_lagged_date", "lag_calendar_days"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise CrossMarketInputError(f"Cannot build alignment audit. Missing: {missing}")

    work = panel.copy()
    work["india_date"] = _date64ns(
        work["india_date"],
        name="india_date",
    ).to_numpy(dtype="datetime64[ns]")
    work["us_lagged_date"] = _date64ns(
        work["us_lagged_date"],
        name="us_lagged_date",
    ).to_numpy(dtype="datetime64[ns]")
    work["lag_calendar_days"] = pd.to_numeric(
        work["lag_calendar_days"],
        errors="coerce",
    )

    matched = work["us_lagged_date"].notna()
    same_date_violations = matched & (work["us_lagged_date"] >= work["india_date"])

    lag = work.loc[matched, "lag_calendar_days"]

    audit = pd.DataFrame(
        [
            {
                "model": model or (
                    str(work["model"].iloc[0])
                    if "model" in work.columns and len(work)
                    else ""
                ),
                "n_india_dates": int(work["india_date"].nunique(dropna=True)),
                "n_matched_us_lagged_dates": int(matched.sum()),
                "n_missing_us_lagged_dates": int((~matched).sum()),
                "min_lag_calendar_days": (
                    float(lag.min()) if lag.notna().any() else np.nan
                ),
                "median_lag_calendar_days": (
                    float(lag.median()) if lag.notna().any() else np.nan
                ),
                "max_lag_calendar_days": (
                    float(lag.max()) if lag.notna().any() else np.nan
                ),
                "n_lag_gt_3_calendar_days": int((lag > 3).sum()),
                "n_lag_gt_7_calendar_days": int((lag > 7).sum()),
                "n_same_date_violations": int(same_date_violations.sum()),
            }
        ]
    )

    if int(audit["n_same_date_violations"].iloc[0]) > 0:
        raise CrossMarketLeakageError(
            "Alignment audit failed: n_same_date_violations > 0."
        )

    return audit


def _required_output_path(
    config: Mapping[str, Any],
    key: str,
) -> str:
    outputs = config.get("outputs", {})
    if key not in outputs:
        raise CrossMarketInputError(f"Missing outputs.{key} in config.")
    return str(outputs[key])


def write_phase13_panel_outputs(
    descriptive_panel: pd.DataFrame,
    predictive_panel: pd.DataFrame,
    combined_panel: pd.DataFrame,
    alignment_audit: pd.DataFrame,
    no_lookahead_audit: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, str]:
    """
    Write core Phase 13 panel outputs.

    Returns a mapping of logical output name to written path.
    """
    written: dict[str, str] = {}

    output_map = {
        "descriptive_same_date_panel": descriptive_panel,
        "predictive_panel": predictive_panel,
        "cross_market_panel": combined_panel,
    }

    for key, df in output_map.items():
        path = _required_output_path(config, key)
        written[key] = str(_safe_to_parquet(df, path, root=root))

    table_map = {
        "alignment_audit": alignment_audit,
        "no_lookahead_audit": no_lookahead_audit,
    }

    for key, df in table_map.items():
        path = _required_output_path(config, key)
        written[key] = str(_safe_to_csv(df, path, root=root))

    return written


def build_phase13_core_panels_for_model(
    config: Mapping[str, Any],
    model: str,
    root: str | Path | None = None,
    *,
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Build all core Phase 13 panels and audits for one regime model.

    This function does not write files.
    """
    us_df, india_df = load_normalized_market_panels(
        config=config,
        model=model,
        root=root,
    )

    descriptive_panel = build_descriptive_same_date_panel(us_df, india_df)
    predictive_panel = build_predictive_panel(
        us_df=us_df,
        india_df=india_df,
        config=config,
        lag_days=lag_days,
    )
    combined_panel = build_combined_cross_market_panel(
        descriptive_panel=descriptive_panel,
        predictive_panel=predictive_panel,
    )

    alignment_audit = build_alignment_audit(
        predictive_panel,
        model=model,
    )
    no_lookahead_audit = build_cross_market_no_lookahead_audit(
        predictive_panel,
        config=config,
        model=model,
    )

    return {
        "us_panel": us_df,
        "india_panel": india_df,
        "descriptive_panel": descriptive_panel,
        "predictive_panel": predictive_panel,
        "combined_panel": combined_panel,
        "alignment_audit": alignment_audit,
        "no_lookahead_audit": no_lookahead_audit,
    }


def build_phase13_core_panels_all_models(
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    models: Iterable[str] | None = None,
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Build stacked core Phase 13 panels for all requested models.

    This function does not write files.
    """
    requested_models = list(models or config.get("models", []))
    if not requested_models:
        raise CrossMarketInputError("No models requested for Phase 13 panel build.")

    descriptive_parts: list[pd.DataFrame] = []
    predictive_parts: list[pd.DataFrame] = []
    combined_parts: list[pd.DataFrame] = []
    alignment_audit_parts: list[pd.DataFrame] = []
    no_lookahead_audit_parts: list[pd.DataFrame] = []

    for model in requested_models:
        built = build_phase13_core_panels_for_model(
            config=config,
            model=model,
            root=root,
            lag_days=lag_days,
        )
        descriptive_parts.append(built["descriptive_panel"])
        predictive_parts.append(built["predictive_panel"])
        combined_parts.append(built["combined_panel"])
        alignment_audit_parts.append(built["alignment_audit"])
        no_lookahead_audit_parts.append(built["no_lookahead_audit"])

    return {
        "descriptive_panel": pd.concat(
            descriptive_parts,
            axis=0,
            ignore_index=True,
        ),
        "predictive_panel": pd.concat(
            predictive_parts,
            axis=0,
            ignore_index=True,
        ),
        "combined_panel": pd.concat(
            combined_parts,
            axis=0,
            ignore_index=True,
        ),
        "alignment_audit": pd.concat(
            alignment_audit_parts,
            axis=0,
            ignore_index=True,
        ),
        "no_lookahead_audit": pd.concat(
            no_lookahead_audit_parts,
            axis=0,
            ignore_index=True,
        ),
    }


def write_cross_market_metadata(
    config: Mapping[str, Any],
    outputs: Mapping[str, str | Path],
    root: str | Path | None = None,
) -> Path:
    """
    Write Phase 13 metadata JSON.

    Metadata is intentionally descriptive. It records the anti-leakage rules,
    locked artifacts, forbidden inputs, and generated outputs.
    """
    from datetime import datetime, timezone

    metadata_path = _required_output_path(config, "phase13_metadata")
    out_path = _ensure_parent_dir(metadata_path, root=root)

    metadata = {
        "phase": config.get("phase"),
        "name": config.get("name"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_policy": config.get("timezone_policy", {}),
        "alignment": config.get("alignment", {}),
        "models": config.get("models", []),
        "descriptive_diagnostics": config.get("descriptive_diagnostics", {}),
        "granger_diagnostics": {
            **config.get("granger_diagnostics", {}),
            "interpretation": (
                "Descriptive lead-lag predictive diagnostic only. "
                "Not causal proof."
            ),
        },
        "logistic_regression": config.get("logistic_regression", {}),
        "predictive_validation": config.get("predictive_validation", {}),
        "overlay": config.get("overlay", {}),
        "locked_artifacts": config.get("locked_artifacts", {}),
        "forbidden_inputs": config.get("forbidden_inputs", {}),
        "no_leakage_acceptance_rules": config.get(
            "no_leakage_acceptance_rules",
            [],
        ),
        "rejected_items": config.get("rejected_items", []),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    return out_path


def run_phase13_core_panel_build(
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    models: Iterable[str] | None = None,
    lag_days: int = 1,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build Phase 13 core panels for requested models and optionally write them.

    This is a convenience function for the later CLI runner.
    """
    built = build_phase13_core_panels_all_models(
        config=config,
        root=root,
        models=models,
        lag_days=lag_days,
    )

    if write_outputs:
        output_paths = write_phase13_panel_outputs(
            descriptive_panel=built["descriptive_panel"],
            predictive_panel=built["predictive_panel"],
            combined_panel=built["combined_panel"],
            alignment_audit=built["alignment_audit"],
            no_lookahead_audit=built["no_lookahead_audit"],
            config=config,
            root=root,
        )
        metadata_path = write_cross_market_metadata(
            config=config,
            outputs=output_paths,
            root=root,
        )
        built["written_outputs"] = pd.DataFrame(
            [
                {"output": key, "path": value}
                for key, value in {
                    **output_paths,
                    "phase13_metadata": str(metadata_path),
                }.items()
            ]
        )

    return built

def _correlation_settings(
    config: Mapping[str, Any] | None = None,
) -> tuple[list[str], int]:
    if config is None:
        return ["pearson", "spearman"], 100

    corr_cfg = config.get("correlations", {})
    methods = corr_cfg.get("methods", ["pearson", "spearman"])
    min_observations = int(corr_cfg.get("min_observations", 100))

    if not isinstance(methods, list) or not methods:
        methods = ["pearson", "spearman"]

    clean_methods = [str(m).lower() for m in methods]
    allowed = {"pearson", "spearman"}
    bad = sorted(set(clean_methods) - allowed)
    if bad:
        raise CrossMarketInputError(f"Unsupported correlation method(s): {bad}")

    return clean_methods, min_observations


def _date_sort_column(df: pd.DataFrame) -> str:
    for col in ("india_date", "date", "us_date"):
        if col in df.columns:
            return col
    raise CrossMarketInputError(
        "Could not identify date sort column. Expected one of: "
        "india_date, date, us_date."
    )


def _group_columns_for_stats(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if "model" in df.columns:
        cols.append("model")
    if "panel_type" in df.columns:
        cols.append("panel_type")
    return cols


def _iter_panel_groups(
    df: pd.DataFrame,
) -> Iterable[tuple[dict[str, Any], pd.DataFrame]]:
    group_cols = _group_columns_for_stats(df)

    if not group_cols:
        yield {"model": "", "panel_type": ""}, df.copy()
        return

    work = df.copy()
    for col in group_cols:
        work[col] = work[col].fillna("").astype(str)

    grouped = work.groupby(group_cols, dropna=False, sort=True)
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        meta.setdefault("model", "")
        meta.setdefault("panel_type", "")
        yield meta, part.copy()


def _as_numeric_pair(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        missing = [c for c in (x_col, y_col) if c not in df.columns]
        raise CrossMarketInputError(f"Missing columns for numeric pair: {missing}")

    out = pd.DataFrame(
        {
            x_col: pd.to_numeric(df[x_col], errors="coerce"),
            y_col: pd.to_numeric(df[y_col], errors="coerce"),
        }
    ).dropna()

    return out


def _is_constant(values: pd.Series) -> bool:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return True
    return bool(clean.nunique(dropna=True) <= 1)


def _correlation_one(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str,
    min_observations: int,
) -> dict[str, Any]:
    pair = _as_numeric_pair(df, x_col=x_col, y_col=y_col)
    n_obs = int(len(pair))

    row: dict[str, Any] = {
        "x": x_col,
        "y": y_col,
        "method": method,
        "n_obs": n_obs,
        "correlation": np.nan,
        "p_value": np.nan,
        "status": "ok",
        "reason": "",
    }

    if n_obs < min_observations:
        row["status"] = "skipped"
        row["reason"] = f"insufficient_observations_lt_{min_observations}"
        return row

    if _is_constant(pair[x_col]) or _is_constant(pair[y_col]):
        row["status"] = "skipped"
        row["reason"] = "constant_series"
        return row

    try:
        try:
            from scipy import stats

            if method == "pearson":
                corr, p_value = stats.pearsonr(pair[x_col], pair[y_col])
            elif method == "spearman":
                corr, p_value = stats.spearmanr(pair[x_col], pair[y_col])
            else:
                raise CrossMarketInputError(f"Unsupported method: {method}")

            row["correlation"] = float(cast(Any, corr))
            row["p_value"] = float(cast(Any, p_value))
        except ModuleNotFoundError as exc:
            if exc.name != "scipy":
                raise
            if method not in {"pearson", "spearman"}:
                raise CrossMarketInputError(f"Unsupported method: {method}")
            row["correlation"] = float(pair[x_col].corr(pair[y_col], method=method))
            row["p_value"] = np.nan
            row["reason"] = "scipy_unavailable_p_value_not_computed"
    except Exception as exc:
        row["status"] = "error"
        row["reason"] = str(exc)

    return row


def _compute_pair_correlations(
    panel: pd.DataFrame,
    pairs: Iterable[tuple[str, str, str]],
    *,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    methods, min_observations = _correlation_settings(config)
    rows: list[dict[str, Any]] = []

    for meta, part in _iter_panel_groups(panel):
        for pair_name, x_col, y_col in pairs:
            if x_col not in part.columns or y_col not in part.columns:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "x": x_col,
                        "y": y_col,
                        "method": "",
                        "n_obs": 0,
                        "correlation": np.nan,
                        "p_value": np.nan,
                        "status": "skipped",
                        "reason": "missing_required_column",
                    }
                )
                continue

            for method in methods:
                row = _correlation_one(
                    part,
                    x_col=x_col,
                    y_col=y_col,
                    method=method,
                    min_observations=min_observations,
                )
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        **row,
                    }
                )

    return pd.DataFrame(rows)


def compute_vrp_level_correlations(
    panel: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute US-India VRP level correlations.

    If the input panel is descriptive_same_date, this is same-date descriptive
    correlation. If the input panel is predictive_lagged, the US value is already
    strictly lagged by construction.
    """
    return _compute_pair_correlations(
        panel,
        pairs=[
            (
                "us_india_vrp_level",
                "us_vrp_har_gk",
                "india_vrp_har_gk",
            )
        ],
        config=config,
    )


def compute_vrp_change_correlations(
    panel: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute US-India VRP change correlations within each model/panel group.
    """
    parts: list[pd.DataFrame] = []

    for meta, part in _iter_panel_groups(panel):
        date_col = _date_sort_column(part)
        work = part.copy()
        work[date_col] = _date64ns(
            work[date_col],
            name=date_col,
        ).to_numpy(dtype="datetime64[ns]")
        work = work.sort_values(date_col).reset_index(drop=True)

        if (
            "us_vrp_har_gk" not in work.columns
            or "india_vrp_har_gk" not in work.columns
        ):
            parts.append(
                pd.DataFrame(
                    [
                        {
                            **meta,
                            "pair": "us_india_vrp_change",
                            "x": "us_vrp_har_gk_change",
                            "y": "india_vrp_har_gk_change",
                            "method": "",
                            "n_obs": 0,
                            "correlation": np.nan,
                            "p_value": np.nan,
                            "status": "skipped",
                            "reason": "missing_required_column",
                        }
                    ]
                )
            )
            continue

        work["us_vrp_har_gk_change"] = pd.to_numeric(
            work["us_vrp_har_gk"],
            errors="coerce",
        ).diff()
        work["india_vrp_har_gk_change"] = pd.to_numeric(
            work["india_vrp_har_gk"],
            errors="coerce",
        ).diff()

        corr = _compute_pair_correlations(
            work,
            pairs=[
                (
                    "us_india_vrp_change",
                    "us_vrp_har_gk_change",
                    "india_vrp_har_gk_change",
                )
            ],
            config=config,
        )

        for key, value in meta.items():
            corr[key] = value

        parts.append(corr)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=0, ignore_index=True)
    front = [
        "model",
        "panel_type",
        "pair",
        "x",
        "y",
        "method",
        "n_obs",
        "correlation",
        "p_value",
        "status",
        "reason",
    ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def compute_regime_probability_correlations(
    panel: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute US-India stress probability correlations.
    """
    x_col = (
        "us_stress_prob_lag1"
        if "us_stress_prob_lag1" in panel.columns
        else "us_stress_prob"
    )

    return _compute_pair_correlations(
        panel,
        pairs=[
            (
                "us_india_stress_probability",
                x_col,
                "india_stress_prob",
            )
        ],
        config=config,
    )


def compute_state_label_agreement(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive agreement between US and India regime state labels.

    This is a label-agreement diagnostic only. It is not used for predictive
    modelling.
    """
    rows: list[dict[str, Any]] = []

    for meta, part in _iter_panel_groups(panel):
        required = {"us_state_name", "india_state_name"}
        missing = sorted(required - set(part.columns))
        if missing:
            rows.append(
                {
                    **meta,
                    "table_type": "summary",
                    "us_state_name": "",
                    "india_state_name": "",
                    "n_obs": 0,
                    "fraction": np.nan,
                    "exact_label_agreement_rate": np.nan,
                    "status": "skipped",
                    "reason": f"missing_columns:{missing}",
                }
            )
            continue

        work = part[["us_state_name", "india_state_name"]].copy()
        work["us_state_name"] = work["us_state_name"].astype("string")
        work["india_state_name"] = work["india_state_name"].astype("string")
        work = work.dropna()

        n_obs = int(len(work))
        if n_obs == 0:
            rows.append(
                {
                    **meta,
                    "table_type": "summary",
                    "us_state_name": "",
                    "india_state_name": "",
                    "n_obs": 0,
                    "fraction": np.nan,
                    "exact_label_agreement_rate": np.nan,
                    "status": "skipped",
                    "reason": "no_nonmissing_state_pairs",
                }
            )
            continue

        us_norm = work["us_state_name"].str.lower().str.strip()
        india_norm = work["india_state_name"].str.lower().str.strip()
        agreement_rate = float((us_norm == india_norm).mean())

        rows.append(
            {
                **meta,
                "table_type": "summary",
                "us_state_name": "__all__",
                "india_state_name": "__all__",
                "n_obs": n_obs,
                "fraction": 1.0,
                "exact_label_agreement_rate": agreement_rate,
                "status": "ok",
                "reason": "",
            }
        )

        counts = (
            work.groupby(["us_state_name", "india_state_name"], dropna=False)
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
        )
        counts["fraction"] = counts["n_obs"] / float(n_obs)

        for _, count_row in counts.iterrows():
            rows.append(
                {
                    **meta,
                    "table_type": "state_pair",
                    "us_state_name": str(count_row["us_state_name"]),
                    "india_state_name": str(count_row["india_state_name"]),
                    "n_obs": int(count_row["n_obs"]),
                    "fraction": float(count_row["fraction"]),
                    "exact_label_agreement_rate": agreement_rate,
                    "status": "ok",
                    "reason": "",
                }
            )

    return pd.DataFrame(rows)


def _ols_hac_row(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    *,
    hac_maxlags: int = 5,
    min_observations: int = 250,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ols_beta": np.nan,
        "ols_t_stat": np.nan,
        "ols_p_value": np.nan,
        "ols_r_squared": np.nan,
        "ols_n_obs": 0,
        "ols_cov_type": "HAC",
        "ols_hac_maxlags": int(hac_maxlags),
        "ols_status": "ok",
        "ols_reason": "",
    }

    pair = _as_numeric_pair(df, x_col=x_col, y_col=y_col)
    row["ols_n_obs"] = int(len(pair))

    if len(pair) < min_observations:
        row["ols_status"] = "skipped"
        row["ols_reason"] = f"insufficient_observations_lt_{min_observations}"
        return row

    if _is_constant(pair[x_col]) or _is_constant(pair[y_col]):
        row["ols_status"] = "skipped"
        row["ols_reason"] = "constant_series"
        return row

    try:
        import statsmodels.api as sm

        y = pair[y_col].astype(float)
        x = sm.add_constant(pair[[x_col]].astype(float), has_constant="add")

        fitted = sm.OLS(y, x, missing="drop").fit()
        robust = fitted.get_robustcov_results(
            cov_type="HAC",
            maxlags=int(hac_maxlags),
        )

        names = list(robust.model.exog_names)
        x_idx = names.index(x_col)

        row["ols_beta"] = float(robust.params[x_idx])
        row["ols_t_stat"] = float(robust.tvalues[x_idx])
        row["ols_p_value"] = float(robust.pvalues[x_idx])
        row["ols_r_squared"] = float(fitted.rsquared)
    except Exception as exc:
        row["ols_status"] = "error"
        row["ols_reason"] = str(exc)

    return row


def build_lead_lag_table(
    panel: pd.DataFrame,
    max_lag: int = 5,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build US-leads-India lead-lag diagnostic table.

    For predictive panels, us_*_lag1 already means:
        latest US observation date strictly before India date.

    Additional lags are implemented as additional row shifts within the
    India-date sorted predictive panel.

    This table is descriptive/statistical. It is not causal proof.
    """
    if max_lag < 1:
        raise CrossMarketInputError("max_lag must be >= 1.")

    lead_lag_cfg = {}
    linear_cfg = {}
    if config is not None:
        lead_lag_cfg = config.get("lead_lag", {})
        linear_cfg = config.get("linear_regression", {})

    hac_maxlags = int(linear_cfg.get("hac_maxlags", 5))
    min_obs_ols = int(linear_cfg.get("min_observations", 250))
    corr_methods, min_obs_corr = _correlation_settings(config)

    pairs = [
        {
            "pair": "us_stress_prob_leads_india_stress_prob",
            "x_base": "us_stress_prob_lag1"
            if "us_stress_prob_lag1" in panel.columns
            else "us_stress_prob",
            "y": "india_stress_prob",
        },
        {
            "pair": "us_vrp_leads_india_vrp",
            "x_base": "us_vrp_har_gk_lag1"
            if "us_vrp_har_gk_lag1" in panel.columns
            else "us_vrp_har_gk",
            "y": "india_vrp_har_gk",
        },
    ]

    rows: list[dict[str, Any]] = []

    for meta, part in _iter_panel_groups(panel):
        date_col = _date_sort_column(part)
        work = part.copy()
        work[date_col] = _date64ns(
            work[date_col],
            name=date_col,
        ).to_numpy(dtype="datetime64[ns]")
        work = work.sort_values(date_col).reset_index(drop=True)

        for pair in pairs:
            pair_name = str(pair["pair"])
            x_base = str(pair["x_base"])
            y_col = str(pair["y"])

            if x_base not in work.columns or y_col not in work.columns:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "us_lag_trading_rows": np.nan,
                        "x": x_base,
                        "y": y_col,
                        "method": "",
                        "n_obs": 0,
                        "correlation": np.nan,
                        "p_value": np.nan,
                        "ols_beta": np.nan,
                        "ols_t_stat": np.nan,
                        "ols_p_value": np.nan,
                        "ols_r_squared": np.nan,
                        "ols_n_obs": 0,
                        "ols_cov_type": "HAC",
                        "ols_hac_maxlags": hac_maxlags,
                        "status": "skipped",
                        "reason": "missing_required_column",
                        "descriptive_only": True,
                        "causal_interpretation_allowed": False,
                    }
                )
                continue

            for lag in range(1, max_lag + 1):
                x_lag_col = f"{x_base}_extra_lag_{lag}"
                work[x_lag_col] = pd.to_numeric(work[x_base], errors="coerce").shift(
                    lag - 1
                )

                for method in corr_methods:
                    corr_row = _correlation_one(
                        work,
                        x_col=x_lag_col,
                        y_col=y_col,
                        method=method,
                        min_observations=min_obs_corr,
                    )
                    ols_row = _ols_hac_row(
                        work,
                        y_col=y_col,
                        x_col=x_lag_col,
                        hac_maxlags=hac_maxlags,
                        min_observations=min_obs_ols,
                    )

                    status = corr_row.get("status", "ok")
                    reason = corr_row.get("reason", "")
                    if status == "ok" and ols_row.get("ols_status") not in {"ok", ""}:
                        status = ols_row.get("ols_status", "skipped")
                        reason = ols_row.get("ols_reason", "")

                    rows.append(
                        {
                            **meta,
                            "pair": pair_name,
                            "us_lag_trading_rows": lag,
                            "x": x_base,
                            "x_lagged_for_test": x_lag_col,
                            "y": y_col,
                            "method": method,
                            "n_obs": int(corr_row.get("n_obs", 0)),
                            "correlation": corr_row.get("correlation", np.nan),
                            "p_value": corr_row.get("p_value", np.nan),
                            **ols_row,
                            "status": status,
                            "reason": reason,
                            "descriptive_only": bool(
                                lead_lag_cfg.get("descriptive_only", True)
                            ),
                            "causal_interpretation_allowed": False,
                        }
                    )

                if x_lag_col in work.columns:
                    work = work.drop(columns=[x_lag_col])

    return pd.DataFrame(rows)


def _validate_granger_series(
    panel: pd.DataFrame,
    y_col: str,
    x_col: str,
    *,
    min_required_ratio: float = 0.90,
) -> tuple[pd.DataFrame, str]:
    """
    Validate and coerce a two-series Granger input.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Clean two-column DataFrame and skip reason. If reason is empty,
        the data is usable.
    """
    if y_col not in panel.columns or x_col not in panel.columns:
        missing = [c for c in (y_col, x_col) if c not in panel.columns]
        return pd.DataFrame(), f"missing_columns:{missing}"

    raw = panel[[y_col, x_col]].copy()
    raw_nonmissing = raw.dropna()
    if raw_nonmissing.empty:
        return pd.DataFrame(), "no_nonmissing_rows"

    numeric = raw_nonmissing.apply(pd.to_numeric, errors="coerce")
    valid_ratio = float(numeric.notna().all(axis=1).mean())
    if valid_ratio < min_required_ratio:
        return pd.DataFrame(), "non_numeric_or_label_like_series"

    clean = numeric.dropna()
    if clean.empty:
        return pd.DataFrame(), "no_numeric_rows_after_coercion"

    if _is_constant(clean[y_col]) or _is_constant(clean[x_col]):
        return pd.DataFrame(), "constant_series"

    return clean[[y_col, x_col]].astype(float), ""


def compute_granger_diagnostics(
    panel: pd.DataFrame,
    max_lag: int = 5,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute Granger-style lead-lag diagnostics.

    This is descriptive only. It does not prove causality.

    Null hypothesis:
        the US series does not add predictive information for the India series,
        conditional on lags of the India series.

    Continuous series only. Regime labels are rejected.
    """
    if max_lag < 1:
        raise CrossMarketInputError("max_lag must be >= 1.")

    if config is not None:
        granger_cfg = config.get("granger_diagnostics", {})
        if granger_cfg.get("continuous_series_only", True) is not True:
            raise CrossMarketInputError(
                "Phase 13 requires Granger diagnostics on continuous series only."
            )

    candidate_panel = panel.copy()
    if "panel_type" in candidate_panel.columns:
        predictive = candidate_panel["panel_type"].astype(str) == "predictive_lagged"
        if predictive.any():
            candidate_panel = candidate_panel.loc[predictive].copy()

    pairs = [
        {
            "pair": "us_stress_prob_lagged_to_india_stress_prob",
            "y": "india_stress_prob",
            "x": "us_stress_prob_lag1"
            if "us_stress_prob_lag1" in candidate_panel.columns
            else "us_stress_prob",
        },
        {
            "pair": "us_vrp_lagged_to_india_vrp",
            "y": "india_vrp_har_gk",
            "x": "us_vrp_har_gk_lag1"
            if "us_vrp_har_gk_lag1" in candidate_panel.columns
            else "us_vrp_har_gk",
        },
    ]

    forbidden_label_cols = {
        "us_state_name",
        "india_state_name",
        "state_name",
        "regime_state_name",
        "state_label",
    }

    rows: list[dict[str, Any]] = []

    for meta, part in _iter_panel_groups(candidate_panel):
        date_col = _date_sort_column(part)
        work = part.copy()
        work[date_col] = _date64ns(
            work[date_col],
            name=date_col,
        ).to_numpy(dtype="datetime64[ns]")
        work = work.sort_values(date_col).reset_index(drop=True)

        for pair in pairs:
            pair_name = str(pair["pair"])
            y_col = str(pair["y"])
            x_col = str(pair["x"])

            if y_col in forbidden_label_cols or x_col in forbidden_label_cols:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "india_target": y_col,
                        "us_predictor": x_col,
                        "lag": np.nan,
                        "n_obs": 0,
                        "test": "",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "df_denom": np.nan,
                        "df_num": np.nan,
                        "status": "skipped",
                        "reason": "label_column_rejected",
                        "descriptive_only": True,
                        "causal_interpretation_allowed": False,
                    }
                )
                continue

            clean, reason = _validate_granger_series(work, y_col=y_col, x_col=x_col)

            min_obs = max(30, max_lag * 8 + 5)
            if reason:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "india_target": y_col,
                        "us_predictor": x_col,
                        "lag": np.nan,
                        "n_obs": 0,
                        "test": "",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "df_denom": np.nan,
                        "df_num": np.nan,
                        "status": "skipped",
                        "reason": reason,
                        "descriptive_only": True,
                        "causal_interpretation_allowed": False,
                    }
                )
                continue

            if len(clean) < min_obs:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "india_target": y_col,
                        "us_predictor": x_col,
                        "lag": np.nan,
                        "n_obs": int(len(clean)),
                        "test": "",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "df_denom": np.nan,
                        "df_num": np.nan,
                        "status": "skipped",
                        "reason": f"insufficient_observations_lt_{min_obs}",
                        "descriptive_only": True,
                        "causal_interpretation_allowed": False,
                    }
                )
                continue

            try:
                import contextlib
                import io
                import warnings

                from statsmodels.tsa.stattools import grangercausalitytests

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            result = grangercausalitytests(
                                clean[[y_col, x_col]],
                                maxlag=max_lag,
                                verbose=False,
                            )
                        except TypeError:
                            result = grangercausalitytests(
                                clean[[y_col, x_col]],
                                maxlag=max_lag,
                            )

                for lag, lag_result in result.items():
                    tests = lag_result[0]
                    for test_name in (
                        "ssr_ftest",
                        "ssr_chi2test",
                        "lrtest",
                        "params_ftest",
                    ):
                        if test_name not in tests:
                            continue

                        values = tests[test_name]
                        statistic = values[0] if len(values) > 0 else np.nan
                        p_value = values[1] if len(values) > 1 else np.nan
                        df_denom = values[2] if len(values) > 2 else np.nan
                        df_num = values[3] if len(values) > 3 else np.nan

                        rows.append(
                            {
                                **meta,
                                "pair": pair_name,
                                "india_target": y_col,
                                "us_predictor": x_col,
                                "lag": int(lag),
                                "n_obs": int(len(clean)),
                                "test": test_name,
                                "statistic": float(statistic)
                                if pd.notna(statistic)
                                else np.nan,
                                "p_value": float(p_value)
                                if pd.notna(p_value)
                                else np.nan,
                                "df_denom": float(df_denom)
                                if pd.notna(df_denom)
                                else np.nan,
                                "df_num": float(df_num)
                                if pd.notna(df_num)
                                else np.nan,
                                "status": "ok",
                                "reason": "",
                                "descriptive_only": True,
                                "causal_interpretation_allowed": False,
                            }
                        )
            except Exception as exc:
                rows.append(
                    {
                        **meta,
                        "pair": pair_name,
                        "india_target": y_col,
                        "us_predictor": x_col,
                        "lag": np.nan,
                        "n_obs": int(len(clean)),
                        "test": "",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "df_denom": np.nan,
                        "df_num": np.nan,
                        "status": "error",
                        "reason": str(exc),
                        "descriptive_only": True,
                        "causal_interpretation_allowed": False,
                    }
                )

    return pd.DataFrame(rows)


def compute_cross_market_stat_tables(
    descriptive_panel: pd.DataFrame,
    predictive_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Compute Phase 13 statistical diagnostic tables.

    Descriptive same-date panel is used for:
        - VRP level correlation
        - VRP change correlation
        - same-date regime probability correlation
        - state label agreement

    Predictive lagged panel is used for:
        - lead-lag table
        - Granger-style diagnostics
    """
    max_lag = int(config.get("lead_lag", {}).get("max_lag", 5))
    granger_max_lag = int(config.get("granger_diagnostics", {}).get("max_lag", max_lag))

    tables = {
        "vrp_level_correlations": compute_vrp_level_correlations(
            descriptive_panel,
            config=config,
        ),
        "vrp_change_correlations": compute_vrp_change_correlations(
            descriptive_panel,
            config=config,
        ),
        "regime_probability_correlations": compute_regime_probability_correlations(
            descriptive_panel,
            config=config,
        ),
        "state_label_agreement": compute_state_label_agreement(
            descriptive_panel,
        ),
        "lead_lag_table": build_lead_lag_table(
            predictive_panel,
            max_lag=max_lag,
            config=config,
        ),
        "granger_diagnostics": compute_granger_diagnostics(
            predictive_panel,
            max_lag=granger_max_lag,
            config=config,
        ),
    }

    return tables


def write_cross_market_stat_tables(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, str]:
    """
    Write Phase 13 statistical diagnostic CSV outputs.
    """
    written: dict[str, str] = {}

    output_keys = [
        "vrp_level_correlations",
        "vrp_change_correlations",
        "regime_probability_correlations",
        "state_label_agreement",
        "lead_lag_table",
        "granger_diagnostics",
    ]

    for key in output_keys:
        if key not in tables:
            continue
        path = _required_output_path(config, key)
        written[key] = str(_safe_to_csv(tables[key], path, root=root))

    return written

def _logistic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("logistic_regression", {}))
    cfg.setdefault("dependent_variable", "india_stress_indicator")
    cfg.setdefault("robust_standard_errors", "HC1")
    cfg.setdefault("min_observations", 250)
    cfg.setdefault("handle_perfect_separation", "skip_with_reason")
    return cfg


def _predictive_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("predictive_validation", {}))
    cfg.setdefault("enabled", True)
    cfg.setdefault("split_method", "chronological")
    cfg.setdefault("train_fraction", 0.70)
    cfg.setdefault("min_train_observations", 250)
    cfg.setdefault("min_test_observations", 100)
    cfg.setdefault("no_cutoff_tuning", True)
    return cfg


def _local_india_features(config: Mapping[str, Any]) -> list[str]:
    features = config.get("predictive_features", {}).get("local_india_lags", [])
    if not isinstance(features, list) or not features:
        features = [
            "india_vrp_har_gk_lag1",
            "india_iv_ann_lag1",
            "india_rv_gk_22d_ann_lag1",
            "india_stress_prob_lag1",
        ]
    return [str(x) for x in features]


def _lagged_us_features(config: Mapping[str, Any]) -> list[str]:
    features = config.get("predictive_features", {}).get("lagged_us_features", [])
    if not isinstance(features, list) or not features:
        features = [
            "us_stress_prob_lag1",
            "us_vrp_har_gk_lag1",
            "us_iv_ann_lag1",
            "us_rv_gk_22d_ann_lag1",
        ]
    return [str(x) for x in features]


def _drop_missing_feature_columns(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    return available, missing


def _prepare_logit_xy(
    panel: pd.DataFrame,
    y_col: str,
    features: list[str],
) -> tuple[pd.Series, pd.DataFrame, str]:
    if y_col not in panel.columns:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            f"missing_dependent_variable:{y_col}",
        )

    available_features, missing_features = _drop_missing_feature_columns(
        panel,
        features,
    )
    if missing_features:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            f"missing_feature_columns:{missing_features}",
        )

    if not available_features:
        return pd.Series(dtype=float), pd.DataFrame(), "no_available_features"

    work = panel[[y_col] + available_features].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

    for col in available_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if work.empty:
        return pd.Series(dtype=float), pd.DataFrame(), "no_complete_cases"

    y = work[y_col].astype(float)
    x = work[available_features].astype(float)

    unique_y = sorted(y.dropna().unique().tolist())
    if not set(unique_y).issubset({0.0, 1.0}):
        return y, x, f"dependent_variable_not_binary:{unique_y}"

    if y.nunique(dropna=True) < 2:
        return y, x, "dependent_variable_has_one_class"

    constant_features = [col for col in x.columns if _is_constant(x[col])]
    if constant_features:
        x = x.drop(columns=constant_features)

    if x.shape[1] == 0:
        return y, x, "all_features_constant"

    return y, x, ""


def _binary_auc_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    Compute binary ROC AUC without requiring sklearn.

    Uses average ranks, so ties are handled.
    """
    y = pd.Series(y_true).astype(float).reset_index(drop=True)
    score = pd.Series(y_score).astype(float).reset_index(drop=True)

    valid = y.notna() & score.notna()
    y = y.loc[valid]
    score = score.loc[valid]

    n_pos = int((y == 1.0).sum())
    n_neg = int((y == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = score.rank(method="average")
    sum_ranks_pos = float(ranks.loc[y == 1.0].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(y_prob).astype(float)
    valid = y.notna() & p.notna()
    if int(valid.sum()) == 0:
        return float("nan")
    return float(np.mean((p.loc[valid] - y.loc[valid]) ** 2))


def _classification_metric_row(
    y_true: pd.Series,
    y_prob: pd.Series,
) -> dict[str, float]:
    return {
        "auc": _binary_auc_score(y_true, y_prob),
        "brier_score": _brier_score(y_true, y_prob),
    }


def _fit_statsmodels_logit(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    robust_cov_type: str = "HC1",
    maxiter: int = 200,
) -> tuple[Any | None, str, str]:
    """
    Fit statsmodels Logit and return:
        fitted_result, status, reason
    """
    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except Exception as exc:
        return None, "error", str(exc)

    try:
        x_const = sm.add_constant(x.astype(float), has_constant="add")
        model = sm.Logit(y.astype(float), x_const, missing="drop")

        try:
            fitted = model.fit(
                disp=False,
                maxiter=maxiter,
                cov_type=robust_cov_type,
            )
        except TypeError:
            fitted = model.fit(
                disp=False,
                maxiter=maxiter,
            )

        return fitted, "ok", ""

    except PerfectSeparationError as exc:
        return None, "skipped", f"perfect_separation:{exc}"
    except np.linalg.LinAlgError as exc:
        return None, "skipped", f"linear_algebra_error:{exc}"
    except Exception as exc:
        reason = str(exc)
        if "perfect separation" in reason.lower():
            return None, "skipped", f"perfect_separation:{reason}"
        return None, "error", reason


def _result_predict_prob(fitted: Any, x: pd.DataFrame) -> pd.Series:
    import statsmodels.api as sm

    x_const = sm.add_constant(x.astype(float), has_constant="add")
    pred = fitted.predict(x_const)
    return pd.Series(pred, index=x.index).astype(float).clip(0.0, 1.0)


def _logit_summary_row(
    *,
    model: str,
    model_spec: str,
    y_col: str,
    features: list[str],
    y: pd.Series,
    x: pd.DataFrame,
    fitted: Any | None,
    status: str,
    reason: str,
    robust_cov_type: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "model_spec": model_spec,
        "dependent_variable": y_col,
        "features": json.dumps(features),
        "n_features_requested": len(features),
        "n_features_used": int(x.shape[1]) if isinstance(x, pd.DataFrame) else 0,
        "n_obs": int(len(y)),
        "n_positive": int((y == 1.0).sum()) if len(y) else 0,
        "n_negative": int((y == 0.0).sum()) if len(y) else 0,
        "positive_fraction": float((y == 1.0).mean()) if len(y) else np.nan,
        "pseudo_r2": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "log_likelihood": np.nan,
        "ll_null": np.nan,
        "auc": np.nan,
        "brier_score": np.nan,
        "robust_standard_errors": robust_cov_type,
        "status": status,
        "reason": reason,
    }

    if fitted is None or status != "ok":
        return row

    try:
        y_prob = _result_predict_prob(fitted, x)
        metrics = _classification_metric_row(y, y_prob)

        row.update(
            {
                "pseudo_r2": float(getattr(fitted, "prsquared", np.nan)),
                "aic": float(getattr(fitted, "aic", np.nan)),
                "bic": float(getattr(fitted, "bic", np.nan)),
                "log_likelihood": float(getattr(fitted, "llf", np.nan)),
                "ll_null": float(getattr(fitted, "llnull", np.nan)),
                "auc": metrics["auc"],
                "brier_score": metrics["brier_score"],
            }
        )
    except Exception as exc:
        row["status"] = "error"
        row["reason"] = f"metric_computation_failed:{exc}"

    return row


def _logit_parameter_table(
    *,
    model: str,
    model_spec: str,
    fitted: Any | None,
    status: str,
    reason: str,
) -> pd.DataFrame:
    if fitted is None or status != "ok":
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "model_spec": model_spec,
                    "parameter": "",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    "status": status,
                    "reason": reason,
                }
            ]
        )

    try:
        params = pd.Series(fitted.params)
        bse = pd.Series(fitted.bse, index=params.index)
        zvalues = pd.Series(fitted.tvalues, index=params.index)
        pvalues = pd.Series(fitted.pvalues, index=params.index)

        rows = []
        for param in params.index:
            rows.append(
                {
                    "model": model,
                    "model_spec": model_spec,
                    "parameter": str(param),
                    "estimate": float(params.loc[param]),
                    "std_error": float(bse.loc[param]),
                    "z_value": float(zvalues.loc[param]),
                    "p_value": float(pvalues.loc[param]),
                    "status": "ok",
                    "reason": "",
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "model_spec": model_spec,
                    "parameter": "",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    "status": "error",
                    "reason": f"parameter_table_failed:{exc}",
                }
            ]
        )


def _fit_logit_spec(
    panel: pd.DataFrame,
    *,
    model: str,
    model_spec: str,
    y_col: str,
    features: list[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    logit_cfg = _logistic_config(config)
    robust_cov_type = str(logit_cfg.get("robust_standard_errors", "HC1"))
    min_obs = int(logit_cfg.get("min_observations", 250))

    y, x, prep_reason = _prepare_logit_xy(panel, y_col=y_col, features=features)

    if prep_reason:
        status = "skipped"
        reason = prep_reason
        fitted = None
    elif len(y) < min_obs:
        status = "skipped"
        reason = f"insufficient_observations_lt_{min_obs}"
        fitted = None
    else:
        fitted, status, reason = _fit_statsmodels_logit(
            y,
            x,
            robust_cov_type=robust_cov_type,
        )

    summary = _logit_summary_row(
        model=model,
        model_spec=model_spec,
        y_col=y_col,
        features=features,
        y=y,
        x=x,
        fitted=fitted,
        status=status,
        reason=reason,
        robust_cov_type=robust_cov_type,
    )

    params = _logit_parameter_table(
        model=model,
        model_spec=model_spec,
        fitted=fitted,
        status=status,
        reason=reason,
    )

    return {
        "model": model,
        "model_spec": model_spec,
        "features": features,
        "y": y,
        "x": x,
        "fitted": fitted,
        "summary": summary,
        "params": params,
        "status": status,
        "reason": reason,
    }


def fit_logit_local_only(
    panel: pd.DataFrame,
    model_name: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    logit_cfg = _logistic_config(config)
    y_col = str(logit_cfg.get("dependent_variable", "india_stress_indicator"))
    features = _local_india_features(config)

    return _fit_logit_spec(
        panel,
        model=model_name,
        model_spec="local_only",
        y_col=y_col,
        features=features,
        config=config,
    )


def fit_logit_local_plus_us(
    panel: pd.DataFrame,
    model_name: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    logit_cfg = _logistic_config(config)
    y_col = str(logit_cfg.get("dependent_variable", "india_stress_indicator"))
    features = _local_india_features(config) + _lagged_us_features(config)

    return _fit_logit_spec(
        panel,
        model=model_name,
        model_spec="local_plus_us",
        y_col=y_col,
        features=features,
        config=config,
    )


def _likelihood_ratio_test(
    local_result: Any | None,
    plus_result: Any | None,
) -> dict[str, Any]:
    row = {
        "likelihood_ratio_stat": np.nan,
        "likelihood_ratio_df": np.nan,
        "likelihood_ratio_p_value": np.nan,
        "likelihood_ratio_status": "ok",
        "likelihood_ratio_reason": "",
    }

    if local_result is None or plus_result is None:
        row["likelihood_ratio_status"] = "skipped"
        row["likelihood_ratio_reason"] = "missing_fitted_result"
        return row

    try:
        from scipy import stats

        ll_local = float(local_result.llf)
        ll_plus = float(plus_result.llf)

        k_local = int(len(local_result.params))
        k_plus = int(len(plus_result.params))
        df = k_plus - k_local

        if df <= 0:
            row["likelihood_ratio_status"] = "skipped"
            row["likelihood_ratio_reason"] = "nonpositive_degrees_of_freedom"
            return row

        lr_stat = max(0.0, 2.0 * (ll_plus - ll_local))
        p_value = float(stats.chi2.sf(lr_stat, df))

        row["likelihood_ratio_stat"] = float(lr_stat)
        row["likelihood_ratio_df"] = int(df)
        row["likelihood_ratio_p_value"] = p_value
    except Exception as exc:
        row["likelihood_ratio_status"] = "error"
        row["likelihood_ratio_reason"] = str(exc)

    return row


def compare_logit_models(
    local_fit: Mapping[str, Any],
    plus_fit: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Compare local-only versus local-plus-US logistic models.

    Main Phase 13 question:
        Does lagged US information add incremental information beyond
        India's own lagged local features?
    """
    local_summary = dict(local_fit.get("summary", {}))
    plus_summary = dict(plus_fit.get("summary", {}))

    model_name = str(
        plus_summary.get("model")
        or local_summary.get("model")
        or plus_fit.get("model")
        or local_fit.get("model")
        or ""
    )

    lr = _likelihood_ratio_test(
        local_fit.get("fitted"),
        plus_fit.get("fitted"),
    )

    row = {
        "model": model_name,
        "comparison": "local_plus_us_vs_local_only",
        "local_status": local_summary.get("status", ""),
        "local_reason": local_summary.get("reason", ""),
        "plus_us_status": plus_summary.get("status", ""),
        "plus_us_reason": plus_summary.get("reason", ""),
        "local_n_obs": local_summary.get("n_obs", np.nan),
        "plus_us_n_obs": plus_summary.get("n_obs", np.nan),
        "local_pseudo_r2": local_summary.get("pseudo_r2", np.nan),
        "plus_us_pseudo_r2": plus_summary.get("pseudo_r2", np.nan),
        "delta_pseudo_r2": (
            plus_summary.get("pseudo_r2", np.nan)
            - local_summary.get("pseudo_r2", np.nan)
        ),
        "local_aic": local_summary.get("aic", np.nan),
        "plus_us_aic": plus_summary.get("aic", np.nan),
        "delta_aic_plus_minus_local": (
            plus_summary.get("aic", np.nan) - local_summary.get("aic", np.nan)
        ),
        "local_bic": local_summary.get("bic", np.nan),
        "plus_us_bic": plus_summary.get("bic", np.nan),
        "delta_bic_plus_minus_local": (
            plus_summary.get("bic", np.nan) - local_summary.get("bic", np.nan)
        ),
        "local_log_likelihood": local_summary.get("log_likelihood", np.nan),
        "plus_us_log_likelihood": plus_summary.get("log_likelihood", np.nan),
        "delta_log_likelihood": (
            plus_summary.get("log_likelihood", np.nan)
            - local_summary.get("log_likelihood", np.nan)
        ),
        "local_auc": local_summary.get("auc", np.nan),
        "plus_us_auc": plus_summary.get("auc", np.nan),
        "delta_auc": plus_summary.get("auc", np.nan) - local_summary.get("auc", np.nan),
        "local_brier_score": local_summary.get("brier_score", np.nan),
        "plus_us_brier_score": plus_summary.get("brier_score", np.nan),
        "delta_brier_score_plus_minus_local": (
            plus_summary.get("brier_score", np.nan)
            - local_summary.get("brier_score", np.nan)
        ),
        **lr,
        "incremental_us_features": json.dumps(list(plus_fit.get("features", []))),
        "local_features": json.dumps(list(local_fit.get("features", []))),
    }

    return pd.DataFrame([row])


def chronological_oos_logit_diagnostic(
    panel: pd.DataFrame,
    *,
    model_name: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Chronological OOS predictive diagnostic.

    This does not tune overlay cutoffs. It only evaluates whether local-plus-US
    has better held-out predictive diagnostics than local-only.
    """
    val_cfg = _predictive_validation_config(config)
    if not bool(val_cfg.get("enabled", True)):
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "model_spec": "",
                    "split_method": "chronological",
                    "status": "skipped",
                    "reason": "predictive_validation_disabled",
                }
            ]
        )

    if str(val_cfg.get("split_method", "chronological")) != "chronological":
        raise CrossMarketInputError(
            "Only chronological predictive validation is allowed."
        )

    logit_cfg = _logistic_config(config)
    y_col = str(logit_cfg.get("dependent_variable", "india_stress_indicator"))
    train_fraction = float(val_cfg.get("train_fraction", 0.70))
    min_train = int(val_cfg.get("min_train_observations", 250))
    min_test = int(val_cfg.get("min_test_observations", 100))

    date_col = _date_sort_column(panel)
    work = panel.copy()
    work[date_col] = _date64ns(
        work[date_col],
        name=date_col,
    ).to_numpy(dtype="datetime64[ns]")
    work = work.sort_values(date_col).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    specs = [
        ("local_only", _local_india_features(config)),
        ("local_plus_us", _local_india_features(config) + _lagged_us_features(config)),
    ]

    for model_spec, features in specs:
        y_all, x_all, prep_reason = _prepare_logit_xy(
            work,
            y_col=y_col,
            features=features,
        )

        base_row = {
            "model": model_name,
            "model_spec": model_spec,
            "split_method": "chronological",
            "train_fraction": train_fraction,
            "n_total_complete": int(len(y_all)),
            "n_train": 0,
            "n_test": 0,
            "train_start_date": "",
            "train_end_date": "",
            "test_start_date": "",
            "test_end_date": "",
            "auc": np.nan,
            "brier_score": np.nan,
            "test_positive_fraction": np.nan,
            "no_cutoff_tuning": bool(val_cfg.get("no_cutoff_tuning", True)),
            "status": "ok",
            "reason": "",
        }

        if prep_reason:
            base_row["status"] = "skipped"
            base_row["reason"] = prep_reason
            rows.append(base_row)
            continue

        n = int(len(y_all))
        split_idx = int(np.floor(n * train_fraction))

        if split_idx < min_train:
            base_row["status"] = "skipped"
            base_row["reason"] = f"train_observations_lt_{min_train}"
            rows.append(base_row)
            continue

        if n - split_idx < min_test:
            base_row["status"] = "skipped"
            base_row["reason"] = f"test_observations_lt_{min_test}"
            rows.append(base_row)
            continue

        y_train = y_all.iloc[:split_idx]
        x_train = x_all.iloc[:split_idx]
        y_test = y_all.iloc[split_idx:]
        x_test = x_all.iloc[split_idx:]

        if y_train.nunique(dropna=True) < 2:
            base_row["status"] = "skipped"
            base_row["reason"] = "train_dependent_variable_has_one_class"
            rows.append(base_row)
            continue

        if y_test.nunique(dropna=True) < 2:
            base_row["status"] = "skipped"
            base_row["reason"] = "test_dependent_variable_has_one_class"
            rows.append(base_row)
            continue

        fitted, status, reason = _fit_statsmodels_logit(
            y_train,
            x_train,
            robust_cov_type=str(logit_cfg.get("robust_standard_errors", "HC1")),
        )

        base_row["n_train"] = int(len(y_train))
        base_row["n_test"] = int(len(y_test))

        try:
            idx_train = y_train.index
            idx_test = y_test.index

            def _row_date(index_value: Any) -> str:
                return str(pd.Timestamp(work.at[index_value, date_col]).date())

            base_row["train_start_date"] = _row_date(idx_train.min())
            base_row["train_end_date"] = _row_date(idx_train.max())
            base_row["test_start_date"] = _row_date(idx_test.min())
            base_row["test_end_date"] = _row_date(idx_test.max())
        except Exception:
            pass

        if fitted is None or status != "ok":
            base_row["status"] = status
            base_row["reason"] = reason
            rows.append(base_row)
            continue

        try:
            y_prob = _result_predict_prob(fitted, x_test)
            metrics = _classification_metric_row(y_test, y_prob)

            base_row["auc"] = metrics["auc"]
            base_row["brier_score"] = metrics["brier_score"]
            base_row["test_positive_fraction"] = float((y_test == 1.0).mean())
            base_row["status"] = "ok"
            base_row["reason"] = ""
        except Exception as exc:
            base_row["status"] = "error"
            base_row["reason"] = f"oos_metric_failed:{exc}"

        rows.append(base_row)

    out = pd.DataFrame(rows)

    if len(out) == 2:
        local = out[out["model_spec"] == "local_only"]
        plus = out[out["model_spec"] == "local_plus_us"]
        if not local.empty and not plus.empty:
            if local["status"].iloc[0] == "ok" and plus["status"].iloc[0] == "ok":
                out["delta_auc_plus_minus_local"] = (
                    float(plus["auc"].iloc[0]) - float(local["auc"].iloc[0])
                )
                out["delta_brier_plus_minus_local"] = (
                    float(plus["brier_score"].iloc[0])
                    - float(local["brier_score"].iloc[0])
                )
            else:
                out["delta_auc_plus_minus_local"] = np.nan
                out["delta_brier_plus_minus_local"] = np.nan

    return out


def logistic_regression_india_stress(
    panel: pd.DataFrame,
    model_name: str,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Fit Phase 13 India stress logistic diagnostics.

    Returns
    -------
    dict[str, pd.DataFrame]
        logistic_model_summary
        logistic_parameter_summary
        logistic_model_comparison
        logistic_oos_diagnostics
    """
    if "panel_type" in panel.columns:
        work = panel[panel["panel_type"].astype(str) == "predictive_lagged"].copy()
        if work.empty:
            work = panel.copy()
    else:
        work = panel.copy()

    if "model" in work.columns:
        model_parts = []
        param_parts = []
        comparison_parts = []
        oos_parts = []

        for group_model, part in work.groupby("model", dropna=False, sort=True):
            group_model_name = str(group_model) if str(group_model) else model_name

            local_fit = fit_logit_local_only(
                part,
                model_name=group_model_name,
                config=config,
            )
            plus_fit = fit_logit_local_plus_us(
                part,
                model_name=group_model_name,
                config=config,
            )

            model_parts.extend([local_fit["summary"], plus_fit["summary"]])
            param_parts.extend([local_fit["params"], plus_fit["params"]])
            comparison_parts.append(compare_logit_models(local_fit, plus_fit))
            oos_parts.append(
                chronological_oos_logit_diagnostic(
                    part,
                    model_name=group_model_name,
                    config=config,
                )
            )

        return {
            "logistic_model_summary": pd.DataFrame(model_parts),
            "logistic_parameter_summary": pd.concat(
                param_parts,
                axis=0,
                ignore_index=True,
            )
            if param_parts
            else pd.DataFrame(),
            "logistic_model_comparison": pd.concat(
                comparison_parts,
                axis=0,
                ignore_index=True,
            )
            if comparison_parts
            else pd.DataFrame(),
            "logistic_oos_diagnostics": pd.concat(
                oos_parts,
                axis=0,
                ignore_index=True,
            )
            if oos_parts
            else pd.DataFrame(),
        }

    local_fit = fit_logit_local_only(
        work,
        model_name=model_name,
        config=config,
    )
    plus_fit = fit_logit_local_plus_us(
        work,
        model_name=model_name,
        config=config,
    )

    return {
        "logistic_model_summary": pd.DataFrame(
            [local_fit["summary"], plus_fit["summary"]]
        ),
        "logistic_parameter_summary": pd.concat(
            [local_fit["params"], plus_fit["params"]],
            axis=0,
            ignore_index=True,
        ),
        "logistic_model_comparison": compare_logit_models(local_fit, plus_fit),
        "logistic_oos_diagnostics": chronological_oos_logit_diagnostic(
            work,
            model_name=model_name,
            config=config,
        ),
    }


def compute_logistic_diagnostic_tables(
    predictive_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Compute all Phase 13 logistic diagnostic tables from the predictive panel.
    """
    if "model" in predictive_panel.columns:
        model_name = "all_models"
    else:
        model_name = "model"

    tables = logistic_regression_india_stress(
        predictive_panel,
        model_name=model_name,
        config=config,
    )

    return tables


def write_logistic_diagnostic_tables(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, str]:
    """
    Write Phase 13 logistic diagnostic CSV outputs.
    """
    written: dict[str, str] = {}

    key_map = {
        "logistic_model_summary": "logistic_model_summary",
        "logistic_model_comparison": "logistic_model_comparison",
        "logistic_oos_diagnostics": "logistic_oos_diagnostics",
    }

    for table_key, output_key in key_map.items():
        if table_key not in tables:
            continue
        path = _required_output_path(config, output_key)
        written[table_key] = str(_safe_to_csv(tables[table_key], path, root=root))

    # Parameter table is useful but not mandatory in the original output contract.
    # Write it only if tables_dir is configured.
    if "logistic_parameter_summary" in tables:
        tables_dir = config.get("outputs", {}).get("tables_dir")
        if tables_dir:
            param_path = Path(str(tables_dir)) / "logistic_parameter_summary.csv"
            written["logistic_parameter_summary"] = str(
                _safe_to_csv(
                    tables["logistic_parameter_summary"],
                    param_path,
                    root=root,
                )
            )

    return written
