"""File IO helpers for data ingestion.

All Phase 1 cached datasets are written as Parquet. CSV is accepted only as a
source ingestion format for manual CBOE/NSE overrides.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def ensure_parent_dir(path: str | Path) -> Path:
    """Create a file path's parent directory if it does not exist."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def save_raw(df: pd.DataFrame, path: str | Path, *, force: bool = False) -> Path:
    """Save a source-specific raw DataFrame as Parquet.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination Parquet path.
    force:
        If False, refuse to overwrite existing files.
    """
    return _save_parquet(df=df, path=path, force=force)

def save_processed(df: pd.DataFrame, path: str | Path, *, force: bool = False) -> Path:
    """Save a canonical processed DataFrame as Parquet."""
    return _save_parquet(df=df, path=path, force=force)

def load_processed(path:str | Path) -> pd.DataFrame:
    """Load a processed Parquet file."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Processed file not found: {input_path}")
    return pd.read_parquet(input_path)

def _save_parquet(df: pd.DataFrame, path: str | Path, *, force: bool = False) -> Path:
    """Internal Parquet writer with explicit overwrite protection."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Refusing to save an empty DataFrame.")

    output_path = ensure_parent_dir(path)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"File already exists and force=False: {output_path}. "
            "Use --force to overwrite."
        )

    df.to_parquet(output_path, index=False)
    return output_path