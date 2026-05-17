from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest



def _load_validate_args():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_features_cli.py"
    spec = importlib.util.spec_from_file_location("build_features_cli_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load build_features_cli.py from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_args


validate_args = _load_validate_args()


def test_validate_args_rejects_non_22_vrp_window() -> None:
    args = argparse.Namespace(
        feature="vrp",
        window=10,
        horizon=22,
        annualization_periods=252,
        max_vix_value=200.0,
    )

    with pytest.raises(ValueError, match="window 22"):
        validate_args(args)


def test_validate_args_rejects_non_22_vrp_horizon() -> None:
    args = argparse.Namespace(
        feature="vrp",
        window=22,
        horizon=10,
        annualization_periods=252,
        max_vix_value=200.0,
    )

    with pytest.raises(ValueError, match="horizon 22"):
        validate_args(args)