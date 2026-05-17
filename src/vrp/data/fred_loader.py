"""FRED data loader.

FRED VIXCLS is a close-only series, so it is explicitly converted into the
canonical OHLCV shape with open/high/low/adj_close equal to close and volume 0.
"""

from __future__ import annotations

import pandas as pd

try:
    from pandas_datareader import data as pdr
except Exception as exc:  # noqa: BLE001
    class _PdrFallback:
        @staticmethod
        def DataReader(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(
                "pandas-datareader import failed. Install a compatible pandas-datareader "
                "version for your pandas release."
            ) from exc

    pdr = _PdrFallback()

from vrp.data.base import BaseDataLoader, DataIngestionError, DataLoadResult
from vrp.data.cleaners import standardize_single_close_series
from vrp.data.validators import validate_ohlcv_frame


def download_fred_series(
    series_id: str,
    start: str | None,
    end: str | None,
    *,
    market: str = "US",
    source: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Download a FRED series and convert it to canonical OHLCV format."""
    resolved_source = source or f"fred_{series_id.lower()}"
    resolved_symbol = symbol or series_id

    try:
        raw = pdr.DataReader(series_id, "fred", start=start, end=end)
    except Exception:
        raw = _download_fred_csv(series_id=series_id, start=start, end=end)

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataIngestionError(
            f"FRED returned no rows for series_id={series_id}, "
            f"start={start}, end={end}."
        )

    prepared = _prepare_fred_frame(raw=raw, series_id=series_id)

    canonical = standardize_single_close_series(
        prepared,
        date_column="date",
        close_column=series_id,
        market=market,
        source=resolved_source,
        symbol=resolved_symbol,
    )
    validate_ohlcv_frame(canonical)
    return canonical


def _download_fred_csv(
    *,
    series_id: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Fallback FRED downloader using the public CSV export endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    try:
        raw = pd.read_csv(url)
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(
            f"FRED download failed for series_id={series_id}, start={start}, end={end}. "
            "Check internet access, pandas-datareader availability, or use a local/manual fallback."
        ) from exc

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataIngestionError(
            f"FRED returned no rows for series_id={series_id}, start={start}, end={end}."
        )

    date_candidates = ["DATE", "date", "observation_date", "Observation Date"]
    date_column = next((col for col in date_candidates if col in raw.columns), None)

    if date_column is None:
        raise DataIngestionError(
            f"FRED CSV response did not contain a recognized date column for "
            f"series_id={series_id}. Available columns: {list(raw.columns)}"
        )

    if series_id not in raw.columns:
        raise DataIngestionError(
            f"FRED CSV response did not contain expected series column '{series_id}'. "
            f"Available columns: {list(raw.columns)}"
        )

    prepared = raw.rename(columns={date_column: "date"}).copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared[series_id] = pd.to_numeric(prepared[series_id], errors="coerce")

    prepared = prepared.dropna(subset=["date", series_id])

    if start is not None:
        prepared = prepared[prepared["date"] >= pd.Timestamp(start)]

    if end is not None:
        prepared = prepared[prepared["date"] <= pd.Timestamp(end)]

    if prepared.empty:
        raise DataIngestionError(
            f"FRED CSV fallback produced no usable rows after date filtering. "
            f"series_id={series_id}, start={start}, end={end}."
        )

    return prepared[["date", series_id]].reset_index(drop=True)


def _prepare_fred_frame(raw: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Normalize FRED output from pandas-datareader or CSV fallback.

    pandas-datareader usually returns the date as the index.
    CSV fallback returns date as a normal column.
    This function handles both without double-inserting 'date'.
    """
    frame = raw.copy()

    if "date" in frame.columns:
        prepared = frame.copy()
    elif "DATE" in frame.columns:
        prepared = frame.rename(columns={"DATE": "date"}).copy()
    elif "observation_date" in frame.columns:
        prepared = frame.rename(columns={"observation_date": "date"}).copy()
    else:
        prepared = frame.rename_axis("date").reset_index()

    if series_id not in prepared.columns:
        raise DataIngestionError(
            f"FRED response did not contain expected series column '{series_id}'. "
            f"Available columns: {list(prepared.columns)}"
        )

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared[series_id] = pd.to_numeric(prepared[series_id], errors="coerce")
    prepared = prepared.dropna(subset=["date", series_id])

    if prepared.empty:
        raise DataIngestionError(
            f"FRED response produced no usable rows after parsing. series_id={series_id}."
        )

    return prepared[["date", series_id]].reset_index(drop=True)


class FredLoader(BaseDataLoader):
    """Config-driven FRED loader."""

    def load(self, start: str | None = None, end: str | None = None) -> DataLoadResult:
        if self.symbol is None:
            raise DataIngestionError(f"FRED source {self.source_id} has no series id.")

        resolved_start = start or self.source_config.get("start_date")
        resolved_end = end or self.source_config.get("end_date")

        frame = download_fred_series(
            series_id=self.symbol,
            start=resolved_start,
            end=resolved_end,
            market=self.market,
            source=self.source_id,
            symbol=self.symbol,
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