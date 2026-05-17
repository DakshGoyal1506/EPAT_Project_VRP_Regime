"""
This module contains shared loader contracts and result containers.
Actual source-specific download logic belongs in:

- yahoo_loader.py
- fred_loader.py
- cboe_loader.py
- nse_loader.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class DataIngestionError(RuntimeError):
    """Raised when a data source cannot be loaded with actionable context."""


@dataclass(frozen=True)
class DataLoadResult:
    """Container returned by every source loader.

    Attributes
    ----------
    source_id:
        Unique source identifier from configs/data_sources.yaml.
    market:
        Market code, for example "US" or "INDIA".
    dataset:
        Canonical dataset name, for example "us_vix".
    role:
        Dataset role, for example "implied_volatility" or "underlying_ohlcv".
    symbol:
        Source symbol.
    frame:
        Canonical OHLCV DataFrame.
    raw_path:
        Source-specific raw Parquet path.
    processed_path:
        Canonical processed Parquet path. Multiple sources may point to the same
        processed path, but the orchestration layer must choose one explicitly.
    metadata:
        Extra source metadata useful for audit and debugging.
    """
    source_id: str
    market: str
    dataset: str
    role: str
    symbol: str | None
    frame: pd.DataFrame
    raw_path: Path
    processed_path: Path | None
    metadata: dict[str, Any]


class BaseDataLoader(ABC):
    """Abstract base class for source-specific data loaders."""

    def __init__(self, source_config: dict[str, Any]) -> None:
        self.source_config = source_config

    @property
    def source_id(self) -> str:
        return str(self.source_config["source_id"])

    @property
    def market(self) -> str:
        return str(self.source_config["market"])

    @property
    def dataset(self) -> str:
        return str(self.source_config["dataset"])

    @property
    def role(self) -> str:
        return str(self.source_config["role"])

    @property
    def symbol(self) -> str | None:
        symbol = self.source_config.get("symbol")
        return None if symbol is None else str(symbol)

    @property
    def raw_path(self) -> Path:
        raw_value = self.source_config.get("raw_path")
        if raw_value is None:
            raw_value = self.source_config.get("raw_output")
        if raw_value is None:
            raise DataIngestionError(
                f"Source {self.source_id} is missing required raw path config. "
                "Expected one of: raw_path, raw_output."
            )
        return Path(str(raw_value))

    @property
    def processed_path(self) -> Path | None:
        processed_value = self.source_config.get("processed_path")
        if processed_value is None:
            processed_value = self.source_config.get("interim_output")
        return None if processed_value is None else Path(str(processed_value))

    @abstractmethod
    def load(self, start: str | None = None, end: str | None = None) -> DataLoadResult:
        """Load and return one canonical OHLCV dataset.

        Implementations must return canonical columns only. They must not save files;
        saving belongs to the CLI orchestration layer.
        """
        pass