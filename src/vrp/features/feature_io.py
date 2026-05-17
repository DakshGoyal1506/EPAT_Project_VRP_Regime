# src/vrp/features/feature_io.py

"""
Feature panel input/output helpers.

Scope:
- Save Phase 2 feature panels to Parquet.
- Load feature panels from Parquet.
- Validate required columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure parent directory exists for a file path.

    Parameters
    ----------
    path:
        Target file path.

    Returns
    -------
    Path
        Normalized Path object.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_feature_panel(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """
    Save a feature panel to Parquet.

    Parameters
    ----------
    df:
        Feature panel.
    path:
        Output Parquet path.
    index:
        Whether to write DataFrame index.

    Returns
    -------
    Path
        Written file path.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot save an empty feature panel.")

    out_path = ensure_parent_dir(path)
    df.to_parquet(out_path, index=index)
    return out_path


def load_feature_panel(
    path: str | Path,
    *,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    """
    Load a feature panel from Parquet.

    Parameters
    ----------
    path:
        Input Parquet path.
    sort_by_date:
        If True and a date column exists, sort by date and reset index.

    Returns
    -------
    pd.DataFrame
        Loaded feature panel.
    """
    in_path = Path(path)

    if not in_path.exists():
        raise FileNotFoundError(f"Feature panel not found: {in_path}")

    df = pd.read_parquet(in_path)

    if sort_by_date and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    return df


def assert_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    panel_name: str = "feature panel",
) -> None:
    """
    Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    required_columns:
        Required column names.
    panel_name:
        Name used in error message.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{panel_name} must be a pandas DataFrame.")

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{panel_name} is missing required column(s): {missing}"
        )