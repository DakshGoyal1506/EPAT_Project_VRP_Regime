"""Yahoo Finance data loader.

Uses yfinance for network downloads in production, but tests must mock the
network call and never depend on live internet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from vrp.data.base import BaseDataLoader, DataIngestionError, DataLoadResult
from vrp.data.cleaners import standardize_yahoo_ohlcv
from vrp.data.validators import validate_ohlcv_frame


def download_yahoo_ohlcv(
    symbol: str,
    start: str | None,
    end: str | None,
    market: str,
    source: str,
) -> pd.DataFrame:
    """Download Yahoo Finance OHLCV data and return canonical OHLCV columns."""
    try:
        raw = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(
            f"Yahoo Finance download failed for symbol={symbol}, "
            f"start={start}, end={end}. "
            "Check internet access, ticker validity, and yfinance availability."
        ) from exc

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataIngestionError(
            f"Yahoo Finance returned no rows for symbol={symbol}, "
            f"start={start}, end={end}. "
            "Try a later start date, verify the ticker, or use a fallback source."
        )

    prepared = _prepare_yfinance_frame(raw)
    canonical = standardize_yahoo_ohlcv(
        prepared,
        market=market,
        source=source,
        symbol=symbol,
    )
    validate_ohlcv_frame(canonical)
    return canonical


class YahooFinanceLoader(BaseDataLoader):
    """Config-driven Yahoo Finance loader."""

    def load(self, start: str | None = None, end: str | None = None) -> DataLoadResult:
        if self.symbol is None:
            raise DataIngestionError(f"Yahoo source {self.source_id} has no symbol.")

        resolved_start = start or self.source_config.get("start_date")
        resolved_end = end or self.source_config.get("end_date")

        frame = download_yahoo_ohlcv(
            symbol=self.symbol,
            start=resolved_start,
            end=resolved_end,
            market=self.market,
            source=self.source_id,
        )

        return DataLoadResult(
            source_id=self.source_id,
            market=self.market,
            dataset=self.dataset,
            role=self.role,
            symbol=self.symbol,
            frame=frame,
            raw_path=self.raw_path,
            processed_path=self.processed_path,
            metadata={
                "provider": self.source_config.get("provider"),
                "access_method": self.source_config.get("access_method"),
                "start": resolved_start,
                "end": resolved_end,
            },
        )


def _prepare_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output before passing it to generic cleaners.

    yfinance may return either flat columns:

    Date, Open, High, Low, Close, Adj Close, Volume

    or MultiIndex columns. For a single ticker, this function collapses the
    OHLCV level and resets the date index.
    """
    out = raw.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out = _collapse_yfinance_multiindex(out)

    if isinstance(out.index, pd.DatetimeIndex) or out.index.name is not None:
        out = out.reset_index()

    return out


def _collapse_yfinance_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance MultiIndex columns to simple OHLCV labels."""
    required = {"open", "high", "low", "close"}

    for level in range(raw.columns.nlevels):
        labels = [str(value) for value in raw.columns.get_level_values(level)]
        normalized = {label.strip().lower() for label in labels}
        if required.issubset(normalized):
            out = raw.copy()
            out.columns = labels
            return out

    raise DataIngestionError(
        "Could not collapse yfinance MultiIndex columns. "
        f"Received columns: {list(raw.columns)}"
    )