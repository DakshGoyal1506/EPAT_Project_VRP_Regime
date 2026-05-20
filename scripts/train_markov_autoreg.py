"""
Dedicated Phase 7 runner for Markov autoregression regimes.

This script intentionally keeps MAR logic out of scripts/train_regimes.py.

Examples
--------
python scripts/train_markov_autoreg.py --market US --target vrp_har --order 1 --states 2 --primary --force

python scripts/train_markov_autoreg.py --market INDIA --target vrp_har --order 1 --states 2 --primary --force

python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force

python scripts/train_markov_autoreg.py --market ALL --run-grid --force
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vrp.regimes.markov_autoreg_registry import (
    MARConfig,
    MARModelSpec,
    ar_stability_path,
    candidate_ranking_path,
    ensure_output_directories,
    expand_candidate_specs,
    load_markov_autoreg_config,
    metadata_path,
    model_specific_model_path,
    model_specific_output_path,
    no_lookahead_audit_path,
    normalize_market,
    primary_alias_model_path,
    primary_alias_output_path,
    probability_audit_path,
    state_summary_path,
    transition_matrix_path,
)

from vrp.regimes.markov_autoreg import (
    MARCandidateFit,
    MARFullFilterResult,
    MARSignalOutput,
    build_mar_signal_output,
    filter_full_series_with_train_params,
    fit_markov_autoreg_candidate,
    fit_summary_dict,
    full_filter_summary_dict,
    mar_signal_summary_dict,
    prepare_mar_data_from_config,
    prepared_data_summary_dict,
)
from vrp.reports.markov_autoreg_diagnostics import write_phase7_diagnostics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 7 Markov autoregression regime model."
    )

    parser.add_argument(
        "--market",
        choices=["US", "INDIA", "ALL"],
        required=True,
        help="Market to run.",
    )

    parser.add_argument(
        "--config",
        default="configs/model_markov_autoreg.yaml",
        help="Path to Phase 7 MAR config.",
    )

    parser.add_argument(
        "--target",
        choices=["vrp_har", "rv", "iv", "returns"],
        default="vrp_har",
        help="Target alias.",
    )

    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Autoregressive order. Phase 7 approved default is 1.",
    )

    parser.add_argument(
        "--states",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of Markov regimes.",
    )

    parser.add_argument(
        "--switching-variance",
        dest="switching_variance",
        action="store_true",
        default=True,
        help="Enable regime-switching variance.",
    )

    parser.add_argument(
        "--no-switching-variance",
        dest="switching_variance",
        action="store_false",
        help="Disable regime-switching variance.",
    )

    parser.add_argument(
        "--switching-ar",
        dest="switching_ar",
        action="store_true",
        default=True,
        help="Enable regime-switching AR coefficient.",
    )

    parser.add_argument(
        "--no-switching-ar",
        dest="switching_ar",
        action="store_false",
        help="Disable regime-switching AR coefficient.",
    )

    parser.add_argument(
        "--switching-trend",
        dest="switching_trend",
        action="store_true",
        default=True,
        help="Enable regime-switching trend/intercept.",
    )

    parser.add_argument(
        "--no-switching-trend",
        dest="switching_trend",
        action="store_false",
        help="Disable regime-switching trend/intercept.",
    )

    parser.add_argument(
        "--primary",
        action="store_true",
        help="Also write primary alias outputs if the requested model succeeds.",
    )

    parser.add_argument(
        "--run-grid",
        action="store_true",
        help="Run approved candidate grid from config.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    parser.add_argument(
        "--strict-economic-check",
        action="store_true",
        help=(
            "Fail if economic state coherence check fails. "
            "Default is false because checks are diagnostic in Phase 7 development."
        ),
    )

    parser.add_argument(
        "--no-smoothed-diagnostic",
        action="store_true",
        help="Skip diagnostic smoothed probabilities.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_markov_autoreg_config(args.config)

    markets = ["US", "INDIA"] if args.market == "ALL" else [normalize_market(args.market)]
    specs = specs_from_args(args=args, cfg=cfg)

    all_market_summaries: list[dict[str, Any]] = []
    exit_code = 0

    for market in markets:
        print("=" * 100)
        print(f"Running Phase 7 MAR for market={market}")
        print(f"candidate_count={len(specs)}")

        try:
            market_summary = run_market(
                market=market,
                specs=specs,
                cfg=cfg,
                args=args,
            )
            all_market_summaries.append(market_summary)

        except Exception as exc:
            exit_code = 1
            print(f"[ERROR] market={market}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("=" * 100)
    print("Phase 7 MAR run complete.")
    print(json.dumps(json_safe(all_market_summaries), indent=2))

    return exit_code


def specs_from_args(args: argparse.Namespace, cfg: MARConfig) -> list[MARModelSpec]:
    if args.run_grid:
        return expand_candidate_specs(cfg)

    return [
        MARModelSpec(
            target=str(args.target),
            order=int(args.order),
            n_states=int(args.states),
            switching_ar=bool(args.switching_ar),
            switching_trend=bool(args.switching_trend),
            switching_variance=bool(args.switching_variance),
            primary=bool(args.primary),
        )
    ]


def run_market(
    market: str,
    specs: list[MARModelSpec],
    cfg: MARConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ranking_rows: list[dict[str, Any]] = []
    successful_outputs: list[dict[str, Any]] = []

    best_primary_signal: MARSignalOutput | None = None
    best_primary_spec: MARModelSpec | None = None

    for spec in specs:
        print("-" * 100)
        print(f"Candidate: market={market}, spec={spec}")

        try:
            result = run_one_candidate(
                market=market,
                spec=spec,
                cfg=cfg,
                args=args,
            )

            candidate = result["candidate"]
            signal = result.get("signal")

            ranking_row = fit_summary_dict(candidate)
            ranking_row["model_specific_output_path"] = str(
                model_specific_output_path(market, spec, cfg)
            )
            ranking_row["model_specific_model_path"] = str(
                model_specific_model_path(market, spec, cfg)
            )

            if signal is not None:
                ranking_row["signal_output_valid"] = True
                ranking_row["economic_check_passed"] = bool(
                    signal.state_mapping.economic_check.get("passed", False)
                )
                ranking_row["economic_check_invalid_reason"] = str(
                    signal.state_mapping.economic_check.get("invalid_reason", "")
                )
                successful_outputs.append(
                    {
                        "market": market,
                        "suffix": spec.suffix(),
                        "output_path": str(model_specific_output_path(market, spec, cfg)),
                        "model_path": str(model_specific_model_path(market, spec, cfg)),
                    }
                )

                if should_write_primary_alias(args=args, spec=spec, cfg=cfg):
                    best_primary_signal = signal
                    best_primary_spec = spec

            else:
                ranking_row["signal_output_valid"] = False
                ranking_row["economic_check_passed"] = False
                ranking_row["economic_check_invalid_reason"] = "no signal output"

            ranking_rows.append(flatten_for_csv(ranking_row))

        except Exception as exc:
            print(
                f"[CANDIDATE_FAILED] market={market}, spec={spec.suffix()}, "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            ranking_rows.append(
                flatten_for_csv(
                    {
                        "market": market,
                        "target": spec.target,
                        "order": spec.order,
                        "n_states": spec.n_states,
                        "switching_ar": spec.switching_ar,
                        "switching_trend": spec.switching_trend,
                        "switching_variance": spec.switching_variance,
                        "suffix": spec.suffix(),
                        "fit_converged": False,
                        "valid_candidate": False,
                        "invalid_reason": f"{type(exc).__name__}: {exc}",
                        "signal_output_valid": False,
                    }
                )
            )

    write_candidate_ranking(
        market=market,
        rows=ranking_rows,
        cfg=cfg,
        force=args.force,
    )

    if args.primary and best_primary_signal is not None and best_primary_spec is not None:
        write_primary_alias_outputs(
            market=market,
            signal=best_primary_signal,
            spec=best_primary_spec,
            cfg=cfg,
            force=args.force,
        )

    elif args.primary:
        print(
            f"[WARN] --primary requested for market={market}, but no primary-valid MAR output was produced.",
            file=sys.stderr,
        )

    return {
        "market": market,
        "n_candidates": len(specs),
        "n_successful_outputs": len(successful_outputs),
        "successful_outputs": successful_outputs,
        "candidate_ranking_path": str(candidate_ranking_path(market, cfg)),
        "primary_alias_written": bool(args.primary and best_primary_signal is not None),
    }


def run_one_candidate(
    market: str,
    spec: MARModelSpec,
    cfg: MARConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ensure_output_directories(market, spec, cfg)

    prepared = prepare_mar_data_from_config(
        market=market,
        spec=spec,
        cfg=cfg,
        enforce_min_observations=True,
    )

    candidate = fit_markov_autoreg_candidate(
        prepared=prepared,
        cfg=cfg,
        raise_on_invalid=False,
    )

    print(f"fit_valid={candidate.fit_summary.valid_candidate}")
    print(f"invalid_reason={candidate.fit_summary.invalid_reason}")

    if not candidate.fit_summary.valid_candidate:
        return {
            "prepared": prepared,
            "candidate": candidate,
            "full_filter": None,
            "signal": None,
        }

    full_filter = filter_full_series_with_train_params(
        candidate=candidate,
        cfg=cfg,
        include_smoothed_diagnostic=not bool(args.no_smoothed_diagnostic),
        raise_on_invalid=True,
    )

    signal = build_mar_signal_output(
        full_filter=full_filter,
        cfg=cfg,
        raise_on_incoherent=bool(args.strict_economic_check),
    )

    write_model_specific_outputs(
        market=market,
        spec=spec,
        prepared_summary=prepared_data_summary_dict(prepared),
        candidate=candidate,
        full_filter=full_filter,
        signal=signal,
        cfg=cfg,
        force=args.force,
    )

    print(f"wrote={model_specific_output_path(market, spec, cfg)}")
    print(f"economic_check={signal.state_mapping.economic_check}")

    return {
        "prepared": prepared,
        "candidate": candidate,
        "full_filter": full_filter,
        "signal": signal,
    }


def should_write_primary_alias(
    args: argparse.Namespace,
    spec: MARModelSpec,
    cfg: MARConfig,
) -> bool:
    if not bool(args.primary):
        return False

    if args.run_grid:
        primary = cfg.primary_model
        return (
            spec.target == primary.target
            and spec.order == primary.order
            and spec.n_states == primary.n_states
            and spec.switching_ar == primary.switching_ar
            and spec.switching_trend == primary.switching_trend
            and spec.switching_variance == primary.switching_variance
        )

    return True


def write_model_specific_outputs(
    market: str,
    spec: MARModelSpec,
    prepared_summary: dict[str, Any],
    candidate: MARCandidateFit,
    full_filter: MARFullFilterResult,
    signal: MARSignalOutput,
    cfg: MARConfig,
    *,
    force: bool,
) -> None:
    output_path = model_specific_output_path(market, spec, cfg)
    model_path = model_specific_model_path(market, spec, cfg)

    write_dataframe(signal.output_frame, output_path, force=force)
    write_pickle(
        build_model_payload(
            market=market,
            spec=spec,
            candidate=candidate,
            signal=signal,
        ),
        model_path,
        force=force,
    )

    write_state_summary(signal, state_summary_path(market, cfg), force=force)
    write_ar_stability(signal, ar_stability_path(market, cfg), force=force)
    write_transition_matrix(candidate, transition_matrix_path(market, cfg), force=force)
    write_probability_audit(full_filter, probability_audit_path(market, cfg), force=force)
    write_no_lookahead_audit(full_filter, no_lookahead_audit_path(market, cfg), force=force)

    write_metadata(
        market=market,
        spec=spec,
        prepared_summary=prepared_summary,
        candidate=candidate,
        full_filter=full_filter,
        signal=signal,
        cfg=cfg,
        path=metadata_path(market, cfg),
        force=force,
    )

    diagnostic_paths = write_phase7_diagnostics(
        signal=signal,
        cfg=cfg,
        force=force,
    )

    print(f"wrote diagnostics={diagnostic_paths}")


def write_primary_alias_outputs(
    market: str,
    signal: MARSignalOutput,
    spec: MARModelSpec,
    cfg: MARConfig,
    *,
    force: bool,
) -> None:
    output_path = primary_alias_output_path(market, cfg)
    model_path = primary_alias_model_path(market, cfg)

    write_dataframe(signal.output_frame, output_path, force=force)

    model_specific_path = model_specific_model_path(market, spec, cfg)
    if model_specific_path.exists():
        ensure_can_write(model_path, force=force)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_specific_path, model_path)
    else:
        write_pickle(
            build_model_payload(
                market=market,
                spec=spec,
                candidate=signal.full_filter.candidate,
                signal=signal,
            ),
            model_path,
            force=force,
        )

    print(f"wrote primary alias output={output_path}")
    print(f"wrote primary alias model={model_path}")


def build_model_payload(
    market: str,
    spec: MARModelSpec,
    candidate: MARCandidateFit,
    signal: MARSignalOutput,
) -> dict[str, Any]:
    params = None
    if candidate.result is not None and hasattr(candidate.result, "params"):
        params = np.asarray(candidate.result.params, dtype=float)

    return {
        "model_family": "markov_autoreg",
        "market": market,
        "spec": spec.to_dict(),
        "params": params,
        "fit_summary": fit_summary_dict(candidate),
        "state_mapping": mar_signal_summary_dict(signal)["state_mapping"],
        "target_transform": candidate.prepared.transform.__dict__,
        "train_test_split": candidate.prepared.split.__dict__,
        "note": (
            "Parameters estimated on train only. Full-series probabilities are produced "
            "by applying statsmodels filter() using these train-fitted parameters."
        ),
    }


def write_state_summary(
    signal: MARSignalOutput,
    path: Path,
    *,
    force: bool,
) -> None:
    df = signal.state_summary.copy()
    mapping = signal.state_mapping.raw_state_to_name
    df["economic_state_name"] = df["raw_state"].map(mapping)
    df["economic_state_code"] = df["economic_state_name"].map(
        {"calm": 0, "transition": 1, "stress": 2}
    )
    write_csv(df, path, force=force)


def write_ar_stability(
    signal: MARSignalOutput,
    path: Path,
    *,
    force: bool,
) -> None:
    df = signal.state_summary.copy()
    mapping = signal.state_mapping.raw_state_to_name
    df["economic_state_name"] = df["raw_state"].map(mapping)

    cols = [
        "market",
        "target",
        "n_states",
        "order",
        "raw_state",
        "economic_state_name",
        "intercept",
        "ar_lag1_phi",
        "sigma2",
        "target_mean_train",
        "target_std_train",
        "persistence_prob",
        "half_life_days",
        "ar_stable",
        "ar_warning",
    ]
    existing = [col for col in cols if col in df.columns]
    write_csv(df[existing], path, force=force)


def write_transition_matrix(
    candidate: MARCandidateFit,
    path: Path,
    *,
    force: bool,
) -> None:
    df = candidate.transition_matrix.copy()
    df.insert(0, "from_state", df.index)
    write_csv(df.reset_index(drop=True), path, force=force)


def write_probability_audit(
    full_filter: MARFullFilterResult,
    path: Path,
    *,
    force: bool,
) -> None:
    df = pd.DataFrame([full_filter.probability_audit.__dict__])
    write_csv(df, path, force=force)


def write_no_lookahead_audit(
    full_filter: MARFullFilterResult,
    path: Path,
    *,
    force: bool,
) -> None:
    df = pd.DataFrame([full_filter.lookahead_audit.__dict__])
    write_csv(df, path, force=force)


def write_candidate_ranking(
    market: str,
    rows: list[dict[str, Any]],
    cfg: MARConfig,
    *,
    force: bool,
) -> None:
    path = candidate_ranking_path(market, cfg)

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "market",
                "target",
                "order",
                "n_states",
                "switching_variance",
                "valid_candidate",
                "invalid_reason",
            ]
        )

    sort_cols = [col for col in ["valid_candidate", "bic", "aic"] if col in df.columns]
    if sort_cols:
        # valid True first, then lower BIC/AIC where available.
        if "valid_candidate" in df.columns:
            df["_valid_sort"] = df["valid_candidate"].astype(bool).astype(int)
            sort_by = ["_valid_sort"]
            ascending = [False]

            if "bic" in df.columns:
                sort_by.append("bic")
                ascending.append(True)
            elif "aic" in df.columns:
                sort_by.append("aic")
                ascending.append(True)

            df = df.sort_values(sort_by, ascending=ascending).drop(columns=["_valid_sort"])

    write_csv(df, path, force=force)


def write_metadata(
    market: str,
    spec: MARModelSpec,
    prepared_summary: dict[str, Any],
    candidate: MARCandidateFit,
    full_filter: MARFullFilterResult,
    signal: MARSignalOutput,
    cfg: MARConfig,
    path: Path,
    *,
    force: bool,
) -> None:
    metadata = {
        "phase": 7,
        "model_family": "markov_autoreg",
        "market": market,
        "spec": spec.to_dict(),
        "config_model_name": cfg.model_name,
        "implementation": cfg.implementation,
        "prepared_data": prepared_summary,
        "fit": fit_summary_dict(candidate),
        "full_filter": full_filter_summary_dict(full_filter),
        "signal": mar_signal_summary_dict(signal),
        "output_paths": {
            "model_specific_output": str(model_specific_output_path(market, spec, cfg)),
            "primary_alias_output": str(primary_alias_output_path(market, cfg)),
            "model_specific_model": str(model_specific_model_path(market, spec, cfg)),
            "primary_alias_model": str(primary_alias_model_path(market, cfg)),
            "candidate_ranking": str(candidate_ranking_path(market, cfg)),
            "state_summary": str(state_summary_path(market, cfg)),
            "transition_matrix": str(transition_matrix_path(market, cfg)),
            "probability_audit": str(probability_audit_path(market, cfg)),
            "no_lookahead_audit": str(no_lookahead_audit_path(market, cfg)),
            "ar_stability": str(ar_stability_path(market, cfg)),
        },
    }

    write_json(metadata, path, force=force)


def write_dataframe(df: pd.DataFrame, path: Path, *, force: bool) -> None:
    ensure_can_write(path, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path, *, force: bool) -> None:
    ensure_can_write(path, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path, *, force: bool) -> None:
    ensure_can_write(path, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=True)


def write_pickle(obj: Any, path: Path, *, force: bool) -> None:
    ensure_can_write(path, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(obj, f)


def ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --force to overwrite."
        )


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(json_safe(value), sort_keys=True)
        else:
            out[key] = json_safe(value)

    return out


def json_safe(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(asdict(obj))

    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if np.isfinite(value):
            return value
        return None

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        if np.isfinite(obj):
            return obj
        return None

    if isinstance(obj, Path):
        return str(obj)

    return obj


if __name__ == "__main__":
    raise SystemExit(main())