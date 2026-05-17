"""NSE local/manual CSV loader.

NSE scripted access can be brittle because of headers, cookies, and session
requirements. Phase 1 therefore implements a robust local/manual CSV path first.
No fragile scraping is added here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vrp.data.base import BaseDataLoader, DataIngestionError, DataLoadResult
from vrp.data.cleaners import standardize_ohlcv_columns
from vrp.data.validators import validate_ohlcv_frame


def load_nse_ohlcv_from_csv(
    csv_path: str | Path,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Load an NSE manual CSV and return canonical OHLCV columns."""
    input_path = Path(csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"NSE local CSV not found: {input_path}")

    try:
        raw = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(f"Could not read NSE CSV: {input_path}") from exc

    return standardize_nse_ohlcv_frame(
        raw,
        market=market,
        source=source,
        symbol=symbol,
        column_map=column_map,
    )


def standardize_nse_ohlcv_frame(
    raw: pd.DataFrame,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Standardize NSE India VIX or NIFTY CSV data into canonical OHLCV."""
    canonical = standardize_ohlcv_columns(
        raw,
        market=market,
        source=source,
        symbol=symbol,
        column_map=column_map
        or {
            "date": [
                "DATE",
                "Date",
                "date",
                "HistoricalDate",
                "TIMESTAMP",
                "timestamp",
            ],
            "open": ["OPEN", "Open", "open"],
            "high": ["HIGH", "High", "high"],
            "low": ["LOW", "Low", "low"],
            "close": ["CLOSE", "Close", "close", "India VIX", "INDIA VIX"],
            "adj_close": ["CLOSE", "Close", "close", "India VIX", "INDIA VIX"],
            "volume": ["Volume", "volume", "VOLUME"],
        },
    )
    validate_ohlcv_frame(canonical)
    return canonical


class NseLocalCsvLoader(BaseDataLoader):
    """Config-driven NSE local CSV loader."""

    def load(self, start: str | None = None, end: str | None = None) -> DataLoadResult:
        if self.symbol is None:
            raise DataIngestionError(f"NSE source {self.source_id} has no symbol.")

        local_csv_path = self.source_config.get("local_csv_path")
        if not local_csv_path:
            raise DataIngestionError(
                f"NSE source {self.source_id} requires local_csv_path in Phase 1. "
                "Manual fix: download the official NSE CSV in browser, save it under "
                "data/manual/nse/, and set local_csv_path in configs/data_sources.yaml. "
                "Do not block the project on brittle NSE scraping."
            )

        column_map = _column_map_from_config(self.source_config)

        frame = load_nse_ohlcv_from_csv(
            local_csv_path,
            market=self.market,
            source=self.source_id,
            symbol=self.symbol,
            column_map=column_map,
        )
        frame = _filter_dates(frame, start or self.source_config.get("start_date"), end)

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
                "access_used": "local_csv",
                "start": start or self.source_config.get("start_date"),
                "end": end or self.source_config.get("end_date"),
            },
        )


def _column_map_from_config(source_config: dict[str, Any]) -> dict[str, list[str]] | None:
    expected = source_config.get("expected_columns_any_of")
    if not isinstance(expected, dict):
        return None

    column_map: dict[str, list[str]] = {}
    for canonical_name, candidates in expected.items():
        if isinstance(candidates, list):
            column_map[str(canonical_name)] = [str(item) for item in candidates]
        elif isinstance(candidates, str):
            column_map[str(canonical_name)] = [candidates]

    return column_map


def _filter_dates(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    out = df.copy()

    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]

    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]

    if out.empty:
        raise DataIngestionError(
            f"NSE date filter produced no rows. start={start}, end={end}"
        )

    validate_ohlcv_frame(out)
    return out.reset_index(drop=True)