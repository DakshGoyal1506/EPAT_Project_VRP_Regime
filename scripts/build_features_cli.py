"""Lightweight CLI validation helpers for build_features.py."""

from __future__ import annotations

import argparse


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments beyond argparse choices.
    """
    if args.window < 2:
        raise ValueError("--window must be an integer >= 2.")

    if args.horizon < 1:
        raise ValueError("--horizon must be an integer >= 1.")

    if args.annualization_periods <= 0:
        raise ValueError("--annualization-periods must be positive.")

    if args.max_vix_value <= 0:
        raise ValueError("--max-vix-value must be positive.")

    if args.feature == "vrp":
        if args.window != 22:
            raise ValueError(
                "Phase 3 VRP currently supports only --window 22 because "
                "the feature registry and VRP column names are fixed to 22-day RV."
            )

        if args.horizon != 22:
            raise ValueError(
                "Phase 3 VRP currently supports only --horizon 22 because "
                "forward ex-post label columns are fixed to 22-day labels."
            )