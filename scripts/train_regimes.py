"""
Train regime models.

Phase 5:
- threshold regimes are preserved through a passthrough hook if your repo already
  exposes a threshold runner.

Phase 6:
- Gaussian HMM regime model
- train-only scaling
- custom filtered probabilities
- diagnostic-only smoothed probabilities
- candidate ranking
- feature availability
- state diagnostics
- probability/no-lookahead audits

Examples
--------
Primary + fallback only:

    python scripts/train_regimes.py --market US --model gaussian_hmm --primary

Full candidate grid:

    python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid

Custom candidate:

    python scripts/train_regimes.py --market INDIA --model gaussian_hmm --feature-set F3 --n-states 3 --covariance-type diag
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from vrp.regimes.hmm_features import (
    build_hmm_feature_panel,
    concatenate_feature_availability_tables,
)
from vrp.regimes.hmm_registry import (
    HMM_FALLBACK_MODEL,
    HMM_FEATURE_SETS,
    HMM_PRIMARY_MODEL,
    PHASE_6_TABLES_DIR,
    get_model_specific_pickle_path,
    get_model_specific_processed_path,
    get_primary_alias_model_path,
    get_primary_alias_processed_path,
    iter_hmm_candidate_specs,
    validate_hmm_model_spec,
)
from vrp.regimes.hmm_scaling import scale_hmm_feature_panel
from vrp.regimes.hmm_validation import (
    HMMValidationRules,
    assert_candidate_grid_values_are_valid,
)
from vrp.regimes.gaussian_hmm import (
    HMMCandidateOutput,
    HMMCandidateSpec,
    HMMFitConfig,
    build_hmm_candidate_output,
    choose_primary_hmm_output,
    fit_hmm_candidate,
)
from vrp.reports.hmm_diagnostics import (
    build_candidate_ranking_table,
    dataframe_content_hash,
    stable_json_hash,
    write_hmm_diagnostics_part1,
    write_hmm_diagnostics_part2,
    write_json,
    write_table_csv,
)


DEFAULT_CONFIG_PATH = Path("configs/model_hmm.yaml")
DEFAULT_PHASE_6_TABLES_DIR = Path("reports/tables/phase_6")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")


MARKET_TO_INPUT_CONFIG_KEY = {
    "US": "us_har_vrp",
    "INDIA": "india_har_vrp",
}

MARKET_TO_THRESHOLD_CONFIG_KEY = {
    "US": "us_threshold_regimes",
    "INDIA": "india_threshold_regimes",
}


def utc_now_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Train Phase 5 threshold regimes or Phase 6 Gaussian HMM regimes."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        default="ALL",
        help="Market to process.",
    )

    parser.add_argument(
        "--model",
        choices=["threshold", "gaussian_hmm"],
        default="gaussian_hmm",
        help="Regime model to train.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs/model_hmm.yaml.",
    )

    parser.add_argument(
        "--feature-set",
        choices=sorted(HMM_FEATURE_SETS.keys()),
        default=None,
        help="HMM feature set. If omitted with --primary, uses configured primary/fallback.",
    )

    parser.add_argument(
        "--n-states",
        type=int,
        choices=[2, 3],
        default=None,
        help="Number of HMM states.",
    )

    parser.add_argument(
        "--covariance-type",
        choices=["diag", "full"],
        default=None,
        help="GaussianHMM covariance type.",
    )

    parser.add_argument(
        "--primary",
        action="store_true",
        help="Run configured primary candidate and configured fallback candidate.",
    )

    parser.add_argument(
        "--run-grid",
        action="store_true",
        help="Run full candidate grid from config.",
    )

    parser.add_argument(
        "--include-smoothed-diagnostics",
        action="store_true",
        default=True,
        help="Write diagnostic-only smoothed probability columns.",
    )

    parser.add_argument(
        "--no-smoothed-diagnostics",
        action="store_true",
        help="Disable diagnostic-only smoothed probability columns.",
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override input Parquet path. Only valid when --market is US or INDIA.",
    )

    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=None,
        help="Override threshold regime Parquet path for comparison diagnostics.",
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Override reports/tables/phase_6 output directory.",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override data/processed output directory.",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override models output directory.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model/panel outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without fitting models.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first market/model failure.",
    )

    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config via safe_load."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must load as a dict. Got {type(data)!r}.")

    return data


def ensure_dir(path: Path) -> Path:
    """Create directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_commit() -> str:
    """Return current git commit if available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""

    if completed.returncode != 0:
        return ""

    return completed.stdout.strip()


def jsonable(value: Any) -> Any:
    """Convert common objects to JSON-compatible values."""
    if is_dataclass(value):
        return jsonable(asdict(value))

    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) else v

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return value


def file_hash(path: Path) -> str:
    """SHA256 hash of file bytes."""
    if not path.exists():
        return ""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_tables_dir(config: Mapping[str, Any], override: Path | None) -> Path:
    """Resolve Phase 6 tables directory."""
    if override is not None:
        return override

    return Path(
        config.get("paths", {})
        .get("output", {})
        .get("tables_dir", DEFAULT_PHASE_6_TABLES_DIR)
    )


def resolve_processed_dir(config: Mapping[str, Any], override: Path | None) -> Path:
    """Resolve processed output directory."""
    if override is not None:
        return override

    return Path(
        config.get("paths", {})
        .get("output", {})
        .get("processed_dir", DEFAULT_PROCESSED_DIR)
    )


def resolve_models_dir(config: Mapping[str, Any], override: Path | None) -> Path:
    """Resolve models output directory."""
    if override is not None:
        return override

    return Path(
        config.get("paths", {})
        .get("output", {})
        .get("model_dir", DEFAULT_MODELS_DIR)
    )


def markets_from_arg(market: str) -> list[str]:
    """Expand market arg."""
    if market == "ALL":
        return ["US", "INDIA"]
    return [market]


def resolve_input_path(
    market: str,
    config: Mapping[str, Any],
    *,
    input_override: Path | None,
) -> Path:
    """Resolve input HAR/VRP panel path for a market."""
    if input_override is not None:
        return input_override

    key = MARKET_TO_INPUT_CONFIG_KEY[market]
    path_text = config.get("paths", {}).get("input", {}).get(key)

    if not path_text:
        raise ValueError(f"Missing config paths.input.{key} for market={market}.")

    return Path(path_text)


def resolve_threshold_path(
    market: str,
    config: Mapping[str, Any],
    *,
    threshold_override: Path | None,
) -> Path | None:
    """Resolve optional threshold regime panel path for diagnostics."""
    if threshold_override is not None:
        return threshold_override

    key = MARKET_TO_THRESHOLD_CONFIG_KEY[market]
    path_text = config.get("paths", {}).get("input", {}).get(key)

    if not path_text:
        return None

    return Path(path_text)


def read_parquet_checked(path: Path) -> pd.DataFrame:
    """Read Parquet or raise clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {path}")

    df = pd.read_parquet(path)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pd.read_parquet did not return DataFrame for {path}")

    if df.empty:
        raise ValueError(f"Input Parquet is empty: {path}")

    return df


def maybe_read_threshold_panel(path: Path | None) -> pd.DataFrame | None:
    """Read optional threshold panel."""
    if path is None or not path.exists():
        return None

    return pd.read_parquet(path)


def get_hmm_fit_config(config: Mapping[str, Any]) -> HMMFitConfig:
    """Build HMMFitConfig from YAML."""
    hmm_fit = dict(config.get("hmm_fit", {}))
    if "random_seed" not in hmm_fit and "random_seed" in config:
        hmm_fit["random_seed"] = config["random_seed"]
    return HMMFitConfig.from_mapping(hmm_fit)


def get_validation_rules(config: Mapping[str, Any]) -> HMMValidationRules:
    """Build validation rules from YAML."""
    model_validation = config.get("model_validation", {})
    reject = model_validation.get("reject_candidate_if", {})
    probability = model_validation.get("probability_row_sum", {})
    availability = model_validation.get("feature_availability", {})

    return HMMValidationRules(
        min_train_state_occupancy=float(reject.get("min_train_state_occupancy_lt", 0.05)),
        min_test_state_occupancy=float(reject.get("min_test_state_occupancy_lt", 0.02)),
        near_absorbing_transition_threshold=float(
            reject.get("near_absorbing_transition_gt", 0.995)
        ),
        probability_row_sum_atol=float(probability.get("atol", 1.0e-8)),
        min_eligible_observations=int(availability.get("min_eligible_observations", 1000)),
        min_eligible_fraction=float(availability.get("min_eligible_fraction", 0.50)),
        reject_non_converged=bool(reject.get("model_does_not_converge", True)),
        reject_economic_monotonicity_failure=bool(
            reject.get("economic_monotonicity_failed", True)
        ),
        reject_feature_availability_failure=bool(
            reject.get("feature_availability_too_low", True)
        ),
    )


def get_train_split_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract chronological train/test split config."""
    split = config.get("train_test_split", {})

    return {
        "train_fraction": float(split.get("train_fraction", 0.70)),
        "min_train_observations": int(split.get("min_train_observations", 750)),
        "min_test_observations": int(split.get("min_test_observations", 250)),
    }


def candidate_specs_from_args(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> list[HMMCandidateSpec]:
    """Resolve candidate specs from CLI/config."""
    if args.run_grid:
        candidate_cfg = config.get("candidate_models", {})
        feature_sets = candidate_cfg.get("feature_sets", list(HMM_FEATURE_SETS))
        n_states_values = candidate_cfg.get("n_states", [2, 3])
        covariance_types = candidate_cfg.get("covariance_types", ["diag", "full"])

        assert_candidate_grid_values_are_valid(
            feature_sets=feature_sets,
            n_states_values=n_states_values,
            covariance_types=covariance_types,
        )

        specs = []
        for item in iter_hmm_candidate_specs(
            feature_sets=feature_sets,
            n_states_values=n_states_values,
            covariance_types=covariance_types,
        ):
            specs.append(
                HMMCandidateSpec(
                    feature_set=str(item["feature_set"]),
                    n_states=int(item["n_states"]),
                    covariance_type=str(item["covariance_type"]),
                )
            )
        return specs

    if args.primary or (
        args.feature_set is None and args.n_states is None and args.covariance_type is None
    ):
        primary = HMMCandidateSpec(
            feature_set=str(config.get("primary_model", HMM_PRIMARY_MODEL).get("feature_set", "F3")),
            n_states=int(config.get("primary_model", HMM_PRIMARY_MODEL).get("n_states", 3)),
            covariance_type=str(
                config.get("primary_model", HMM_PRIMARY_MODEL).get("covariance_type", "diag")
            ),
        )
        fallback = HMMCandidateSpec(
            feature_set=str(
                config.get("fallback_model", HMM_FALLBACK_MODEL).get("feature_set", "F3")
            ),
            n_states=int(config.get("fallback_model", HMM_FALLBACK_MODEL).get("n_states", 2)),
            covariance_type=str(
                config.get("fallback_model", HMM_FALLBACK_MODEL).get("covariance_type", "diag")
            ),
        )

        if primary == fallback:
            return [primary]

        return [primary, fallback]

    missing = []
    if args.feature_set is None:
        missing.append("--feature-set")
    if args.n_states is None:
        missing.append("--n-states")
    if args.covariance_type is None:
        missing.append("--covariance-type")

    if missing:
        raise ValueError(
            "For a custom HMM candidate, provide all of: "
            + ", ".join(missing)
            + ". Or use --primary / --run-grid."
        )

    validate_hmm_model_spec(args.feature_set, args.n_states, args.covariance_type)

    return [
        HMMCandidateSpec(
            feature_set=args.feature_set,
            n_states=args.n_states,
            covariance_type=args.covariance_type,
        )
    ]


def group_specs_by_feature_set(
    specs: Sequence[HMMCandidateSpec],
) -> dict[str, list[HMMCandidateSpec]]:
    """Group candidate specs by feature set."""
    grouped: dict[str, list[HMMCandidateSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.feature_set, []).append(spec)
    return grouped


def choose_diagnostic_output(outputs: Sequence[HMMCandidateOutput]) -> HMMCandidateOutput | None:
    """
    Choose a diagnostic output when no model passes validation.

    This does not mark selected_primary. It only provides an output panel so
    diagnostics and invalid-model artifacts can still be written.
    """
    with_panel = [output for output in outputs if output.output_panel is not None]

    if not with_panel:
        return None

    finite = [
        output
        for output in with_panel
        if np.isfinite(output.fit_result.bic)
    ]

    if finite:
        return min(finite, key=lambda output: output.fit_result.bic)

    return with_panel[0]


def save_candidate_artifacts(
    output: HMMCandidateOutput,
    *,
    processed_dir: Path,
    models_dir: Path,
    force: bool,
) -> dict[str, Path]:
    """Save model-specific panel and model bundle."""
    spec = output.fit_result.spec
    market = output.fit_result.market

    panel_path = get_model_specific_processed_path(
        market,
        spec.feature_set,
        spec.n_states,
        spec.covariance_type,
        processed_dir=processed_dir,
    )
    model_path = get_model_specific_pickle_path(
        market,
        spec.feature_set,
        spec.n_states,
        spec.covariance_type,
        model_dir=models_dir,
    )

    ensure_dir(panel_path.parent)
    ensure_dir(model_path.parent)

    if output.output_panel is not None:
        if panel_path.exists() and not force:
            raise FileExistsError(f"Output exists. Use --force to overwrite: {panel_path}")
        output.output_panel.to_parquet(panel_path, index=False)

    if model_path.exists() and not force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {model_path}")

    bundle = {
        "model_name": "gaussian_hmm_v1",
        "market": market,
        "spec": spec.to_dict(),
        "fit_config": asdict(output.fit_result.fit_config),
        "model": output.fit_result.model,
        "scaler": output.fit_result.scaled_panel.scaler,
        "scaler_metadata": output.fit_result.scaled_panel.metadata.to_dict(),
        "state_labeling": (
            output.labeling.to_metadata()
            if output.labeling is not None
            else {}
        ),
        "validation": {
            "passed": output.passed,
            "hmm_model_valid": output.hmm_model_valid,
            "hmm_model_failure_reason": output.hmm_model_failure_reason,
            "rejection_reason": output.validation.rejection_reason_text,
        },
        "created_at_utc": utc_now_iso(),
    }
    joblib.dump(bundle, model_path)

    key = f"{market.lower()}_{spec.feature_set}_k{spec.n_states}_{spec.covariance_type}"

    return {
        f"model_specific_panel_{key}": panel_path,
        f"model_specific_model_{key}": model_path,
    }


def save_primary_alias_artifacts(
    output: HMMCandidateOutput,
    *,
    processed_dir: Path,
    models_dir: Path,
    force: bool,
) -> dict[str, Path]:
    """Save primary alias panel/model for selected output."""
    market = output.fit_result.market

    panel_path = get_primary_alias_processed_path(
        market,
        processed_dir=processed_dir,
    )
    model_path = get_primary_alias_model_path(
        market,
        model_dir=models_dir,
    )

    ensure_dir(panel_path.parent)
    ensure_dir(model_path.parent)

    if output.output_panel is not None:
        if panel_path.exists() and not force:
            raise FileExistsError(f"Output exists. Use --force to overwrite: {panel_path}")
        output.output_panel.to_parquet(panel_path, index=False)

    if model_path.exists() and not force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {model_path}")

    bundle = {
        "model_name": "gaussian_hmm_v1",
        "market": market,
        "selected_primary": output.passed,
        "spec": output.fit_result.spec.to_dict(),
        "fit_config": asdict(output.fit_result.fit_config),
        "model": output.fit_result.model,
        "scaler": output.fit_result.scaled_panel.scaler,
        "scaler_metadata": output.fit_result.scaled_panel.metadata.to_dict(),
        "state_labeling": (
            output.labeling.to_metadata()
            if output.labeling is not None
            else {}
        ),
        "validation": {
            "passed": output.passed,
            "hmm_model_valid": output.hmm_model_valid,
            "hmm_model_failure_reason": output.hmm_model_failure_reason,
            "rejection_reason": output.validation.rejection_reason_text,
        },
        "created_at_utc": utc_now_iso(),
    }
    joblib.dump(bundle, model_path)

    return {
        "primary_alias_panel": panel_path,
        "primary_alias_model": model_path,
    }


def write_feature_availability_table(
    panels_by_feature_set: Mapping[str, Any],
    *,
    tables_dir: Path,
) -> Path:
    """Write reports/tables/phase_6/hmm_feature_availability.csv."""
    availability = concatenate_feature_availability_tables(
        panels_by_feature_set.values()
    )
    path = tables_dir / "hmm_feature_availability.csv"
    write_table_csv(availability, path)
    return path


def write_combined_metadata(
    *,
    outputs: Sequence[HMMCandidateOutput],
    selected_output: HMMCandidateOutput | None,
    input_path: Path,
    input_df: pd.DataFrame,
    config: Mapping[str, Any],
    config_path: Path,
    tables_dir: Path,
) -> Path:
    """Write a combined metadata JSON for all candidates in one market run."""
    selected = selected_output or choose_diagnostic_output(outputs)

    payload: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "input_data_hash": file_hash(input_path),
        "input_dataframe_hash": dataframe_content_hash(input_df),
        "config_path": str(config_path),
        "config_hash": stable_json_hash(config),
        "code_version_or_git_commit": get_git_commit(),
        "selected": (
            {
                "market": selected.fit_result.market,
                "feature_set": selected.fit_result.spec.feature_set,
                "n_states": selected.fit_result.spec.n_states,
                "covariance_type": selected.fit_result.spec.covariance_type,
                "hmm_model_valid": selected.hmm_model_valid,
                "hmm_model_failure_reason": selected.hmm_model_failure_reason,
            }
            if selected is not None
            else {}
        ),
        "candidates": [
            {
                "market": output.fit_result.market,
                "feature_set": output.fit_result.spec.feature_set,
                "n_states": output.fit_result.spec.n_states,
                "covariance_type": output.fit_result.spec.covariance_type,
                "passed": output.passed,
                "hmm_model_valid": output.hmm_model_valid,
                "hmm_model_failure_reason": output.hmm_model_failure_reason,
                "rejection_reason": output.validation.rejection_reason_text,
                "train_loglik": output.fit_result.train_loglik,
                "test_loglik": output.fit_result.test_loglik,
                "aic": output.fit_result.aic,
                "bic": output.fit_result.bic,
                "converged": output.fit_result.converged,
                "n_iter": output.fit_result.n_iter,
            }
            for output in outputs
        ],
    }

    path = tables_dir / "hmm_metadata.json"
    write_json(payload, path)
    return path


def write_market_hmm_diagnostics(
    *,
    outputs: Sequence[HMMCandidateOutput],
    selected_output: HMMCandidateOutput | None,
    diagnostic_output: HMMCandidateOutput,
    tables_dir: Path,
    threshold_panel: pd.DataFrame | None,
    input_path: Path,
    input_df: pd.DataFrame,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Path]:
    """Write all Phase 6 diagnostic tables for one market."""
    ensure_dir(tables_dir)

    paths: dict[str, Path] = {}

    paths["candidate_model_ranking"] = tables_dir / "hmm_candidate_model_ranking.csv"
    ranking = build_candidate_ranking_table(
        outputs,
        selected_output=selected_output,
    )
    write_table_csv(ranking, paths["candidate_model_ranking"])

    part1_paths = write_hmm_diagnostics_part1(
        diagnostic_output,
        selected_output=selected_output or diagnostic_output,
        all_outputs=outputs,
        tables_dir=tables_dir,
        input_data_hash=file_hash(input_path),
        feature_panel_hash=dataframe_content_hash(input_df),
        config_hash=stable_json_hash(config),
        code_version_or_git_commit=get_git_commit(),
    )
    paths.update(part1_paths)

    part2_paths = write_hmm_diagnostics_part2(
        diagnostic_output,
        threshold_panel=threshold_panel,
        crisis_windows=config.get("crisis_windows"),
        tables_dir=tables_dir,
    )
    paths.update(part2_paths)

    paths["metadata"] = write_combined_metadata(
        outputs=outputs,
        selected_output=selected_output,
        input_path=input_path,
        input_df=input_df,
        config=config,
        config_path=config_path,
        tables_dir=tables_dir,
    )

    return paths


def run_gaussian_hmm_for_market(
    market: str,
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run Phase 6 Gaussian HMM for one market."""
    input_path = resolve_input_path(
        market,
        config,
        input_override=args.input_path,
    )
    threshold_path = resolve_threshold_path(
        market,
        config,
        threshold_override=args.threshold_path,
    )
    tables_dir = resolve_tables_dir(config, args.tables_dir)
    processed_dir = resolve_processed_dir(config, args.processed_dir)
    models_dir = resolve_models_dir(config, args.models_dir)

    market_tables_dir = tables_dir / market.lower()
    ensure_dir(market_tables_dir)
    ensure_dir(processed_dir)
    ensure_dir(models_dir)

    specs = candidate_specs_from_args(args, config)

    if args.dry_run:
        return {
            "market": market,
            "dry_run": True,
            "input_path": str(input_path),
            "threshold_path": str(threshold_path) if threshold_path else "",
            "tables_dir": str(market_tables_dir),
            "processed_dir": str(processed_dir),
            "models_dir": str(models_dir),
            "candidate_specs": [spec.to_dict() for spec in specs],
        }

    input_df = read_parquet_checked(input_path)
    threshold_panel = maybe_read_threshold_panel(threshold_path)

    fit_config = get_hmm_fit_config(config)
    validation_rules = get_validation_rules(config)
    split_config = get_train_split_config(config)

    include_smoothed = bool(args.include_smoothed_diagnostics and not args.no_smoothed_diagnostics)

    grouped = group_specs_by_feature_set(specs)

    feature_panels = {}
    scaled_panels = {}

    all_outputs: list[HMMCandidateOutput] = []

    for feature_set, feature_specs in grouped.items():
        feature_panel = build_hmm_feature_panel(
            input_df,
            market=market,
            feature_set=feature_set,
            min_eligible_observations=validation_rules.min_eligible_observations,
            min_eligible_fraction=validation_rules.min_eligible_fraction,
        )
        feature_panels[feature_set] = feature_panel

        scaled_panel = scale_hmm_feature_panel(
            feature_panel,
            train_fraction=split_config["train_fraction"],
            min_train_observations=split_config["min_train_observations"],
            min_test_observations=split_config["min_test_observations"],
        )
        scaled_panels[feature_set] = scaled_panel

        for spec in feature_specs:
            fit_result = fit_hmm_candidate(
                scaled_panel,
                spec=spec,
                fit_config=fit_config,
                validation_rules=validation_rules,
            )
            output = build_hmm_candidate_output(
                fit_result,
                include_smoothed_probabilities_diagnostic=include_smoothed,
                validation_rules=validation_rules,
            )
            all_outputs.append(output)

    selected_output = choose_primary_hmm_output(all_outputs)
    diagnostic_output = selected_output or choose_diagnostic_output(all_outputs)

    if diagnostic_output is None:
        raise RuntimeError(f"No usable diagnostic HMM output could be built for market={market}.")

    artifact_paths: dict[str, Path] = {}

    feature_availability_path = write_feature_availability_table(
        feature_panels,
        tables_dir=market_tables_dir,
    )
    artifact_paths["feature_availability"] = feature_availability_path

    for output in all_outputs:
        saved = save_candidate_artifacts(
            output,
            processed_dir=processed_dir,
            models_dir=models_dir,
            force=args.force,
        )
        artifact_paths.update(saved)

    alias_source = selected_output or diagnostic_output
    alias_saved = save_primary_alias_artifacts(
        alias_source,
        processed_dir=processed_dir,
        models_dir=models_dir,
        force=args.force,
    )
    artifact_paths.update(alias_saved)

    diagnostics_paths = write_market_hmm_diagnostics(
        outputs=all_outputs,
        selected_output=selected_output,
        diagnostic_output=diagnostic_output,
        tables_dir=market_tables_dir,
        threshold_panel=threshold_panel,
        input_path=input_path,
        input_df=input_df,
        config=config,
        config_path=args.config,
    )
    artifact_paths.update(diagnostics_paths)

    return {
        "market": market,
        "dry_run": False,
        "input_path": str(input_path),
        "threshold_path": str(threshold_path) if threshold_path else "",
        "n_candidates": len(all_outputs),
        "selected": (
            selected_output.fit_result.spec.to_dict()
            if selected_output is not None
            else None
        ),
        "selected_valid": bool(selected_output is not None),
        "diagnostic_output": diagnostic_output.fit_result.spec.to_dict(),
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
        "candidate_status": [
            {
                "spec": output.fit_result.spec.to_dict(),
                "passed": output.passed,
                "hmm_model_valid": output.hmm_model_valid,
                "hmm_model_failure_reason": output.hmm_model_failure_reason,
                "rejection_reason": output.validation.rejection_reason_text,
            }
            for output in all_outputs
        ],
    }


def run_threshold_passthrough(args: argparse.Namespace) -> int:
    """
    Preserve Phase 5 threshold behavior through optional repo-specific passthrough.

    If your existing repo already had threshold code inside this script, keep that
    code and place the Gaussian HMM code above/below it. This passthrough tries
    common runner names; if none exist, it fails with an actionable message.
    """
    candidates = [
        ("vrp.regimes.threshold_regimes", "run_threshold_regimes_cli"),
        ("vrp.regimes.threshold_regimes", "main"),
        ("vrp.regimes.threshold", "run_threshold_regimes_cli"),
        ("vrp.regimes.threshold", "main"),
    ]

    last_error = ""

    for module_name, function_name in candidates:
        try:
            module = __import__(module_name, fromlist=[function_name])
            fn = getattr(module, function_name)
        except Exception as exc:
            last_error = f"{module_name}.{function_name}: {exc}"
            continue

        result = fn(args)
        return int(result or 0)

    raise RuntimeError(
        "Could not locate existing Phase 5 threshold runner. "
        "Keep your previous threshold branch in scripts/train_regimes.py, or expose "
        "one of: "
        + ", ".join(f"{m}.{f}" for m, f in candidates)
        + f". Last error: {last_error}"
    )


def print_run_summary(results: Sequence[Mapping[str, Any]]) -> None:
    """Print compact run summary."""
    print(json.dumps(jsonable(list(results)), indent=2, sort_keys=True))


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    if args.model == "threshold":
        return run_threshold_passthrough(args)

    config = load_yaml_config(args.config)
    markets = markets_from_arg(args.market)

    results: list[dict[str, Any]] = []

    for market in markets:
        try:
            result = run_gaussian_hmm_for_market(
                market,
                args=args,
                config=config,
            )
            results.append(result)
        except Exception as exc:
            error_result = {
                "market": market,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(error_result)

            if args.fail_fast:
                print_run_summary(results)
                raise

    print_run_summary(results)

    failed = [item for item in results if "error" in item]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())