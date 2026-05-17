from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vrp.data.base import DataIngestionError, DataLoadResult
from vrp.data.cboe_loader import CboeVixLoader, load_cboe_vix_from_csv
from vrp.data.fred_loader import FredLoader, download_fred_series
from vrp.data.nse_loader import NseLocalCsvLoader, load_nse_ohlcv_from_csv
from vrp.data.schema import OHLCV_COLUMNS
from vrp.data.yahoo_loader import YahooFinanceLoader, download_yahoo_ohlcv
from vrp.data.validators import validate_ohlcv_frame


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_download_yahoo_ohlcv_uses_mocked_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.read_csv(
        FIXTURES_DIR / "yahoo_ohlcv_sample.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    def fake_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        return raw

    monkeypatch.setattr("vrp.data.yahoo_loader.yf.download", fake_download)

    df = download_yahoo_ohlcv(
        symbol="^GSPC",
        start="2024-01-01",
        end="2024-01-05",
        market="US",
        source="yahoo_spx",
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert df["source"].unique().tolist() == ["yahoo_spx"]
    assert df["market"].unique().tolist() == ["US"]
    assert df["symbol"].unique().tolist() == ["^GSPC"]
    validate_ohlcv_frame(df)


def test_yahoo_loader_returns_data_load_result(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.read_csv(
        FIXTURES_DIR / "yahoo_ohlcv_sample.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    def fake_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        return raw

    monkeypatch.setattr("vrp.data.yahoo_loader.yf.download", fake_download)

    loader = YahooFinanceLoader(
        {
            "source_id": "yahoo_spx",
            "market": "US",
            "dataset": "us_underlying",
            "role": "underlying_ohlcv",
            "symbol": "^GSPC",
            "provider": "yahoo_finance",
            "access_method": "yfinance",
            "start_date": "2024-01-01",
            "end_date": None,
            "raw_path": "data/raw/us_spx_yahoo.parquet",
            "processed_path": "data/processed/us_underlying.parquet",
        }
    )

    result = loader.load()

    assert isinstance(result, DataLoadResult)
    assert result.source_id == "yahoo_spx"
    assert result.market == "US"
    assert result.dataset == "us_underlying"
    assert list(result.frame.columns) == OHLCV_COLUMNS
    validate_ohlcv_frame(result.frame)


def test_yahoo_loader_raises_actionable_error_on_empty_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        return pd.DataFrame()

    monkeypatch.setattr("vrp.data.yahoo_loader.yf.download", fake_download)

    with pytest.raises(DataIngestionError, match="returned no rows"):
        download_yahoo_ohlcv(
            symbol="BAD",
            start="2024-01-01",
            end="2024-01-05",
            market="US",
            source="yahoo_bad",
        )


def test_download_fred_series_uses_mocked_pandas_datareader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {"VIXCLS": [13.5, 14.7, 14.1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    def fake_datareader(series_id, source, start=None, end=None):  # type: ignore[no-untyped-def]
        assert series_id == "VIXCLS"
        assert source == "fred"
        return raw

    monkeypatch.setattr("vrp.data.fred_loader.pdr.DataReader", fake_datareader)

    df = download_fred_series(
        series_id="VIXCLS",
        start="2024-01-01",
        end="2024-01-05",
        market="US",
        source="fred_vixcls",
        symbol="VIXCLS",
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert (df["open"] == df["close"]).all()
    assert (df["high"] == df["close"]).all()
    assert (df["low"] == df["close"]).all()
    assert (df["adj_close"] == df["close"]).all()
    assert (df["volume"] == 0).all()
    validate_ohlcv_frame(df)


def test_fred_loader_returns_data_load_result(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {"VIXCLS": [13.5, 14.7, 14.1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    def fake_datareader(series_id, source, start=None, end=None):  # type: ignore[no-untyped-def]
        return raw

    monkeypatch.setattr("vrp.data.fred_loader.pdr.DataReader", fake_datareader)

    loader = FredLoader(
        {
            "source_id": "fred_vixcls",
            "market": "US",
            "dataset": "us_vix",
            "role": "implied_volatility",
            "symbol": "VIXCLS",
            "provider": "fred",
            "access_method": "pandas_datareader",
            "start_date": "2024-01-01",
            "end_date": None,
            "raw_path": "data/raw/us_vix_fred.parquet",
            "processed_path": "data/processed/us_vix.parquet",
        }
    )

    result = loader.load()

    assert isinstance(result, DataLoadResult)
    assert result.source_id == "fred_vixcls"
    assert list(result.frame.columns) == OHLCV_COLUMNS
    validate_ohlcv_frame(result.frame)


def test_load_cboe_vix_from_csv_fixture() -> None:
    df = load_cboe_vix_from_csv(
        FIXTURES_DIR / "cboe_vix_sample.csv",
        market="US",
        source="cboe_vix",
        symbol="VIX",
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert df["source"].unique().tolist() == ["cboe_vix"]
    assert df["symbol"].unique().tolist() == ["VIX"]
    assert (df["volume"] == 0).all()
    validate_ohlcv_frame(df)


def test_cboe_loader_prefers_local_csv() -> None:
    loader = CboeVixLoader(
        {
            "source_id": "cboe_vix",
            "market": "US",
            "dataset": "us_vix",
            "role": "implied_volatility",
            "symbol": "VIX",
            "provider": "cboe",
            "access_method": "csv_url_or_local_csv",
            "start_date": "2024-01-01",
            "end_date": None,
            "official_url": "https://example.invalid/VIX_History.csv",
            "local_csv_path": str(FIXTURES_DIR / "cboe_vix_sample.csv"),
            "raw_path": "data/raw/us_vix_cboe.parquet",
            "processed_path": "data/processed/us_vix.parquet",
        }
    )

    result = loader.load()

    assert isinstance(result, DataLoadResult)
    assert result.metadata["access_used"] == "local_csv"
    assert list(result.frame.columns) == OHLCV_COLUMNS
    validate_ohlcv_frame(result.frame)


def test_load_nse_ohlcv_from_csv_fixture() -> None:
    df = load_nse_ohlcv_from_csv(
        FIXTURES_DIR / "nse_india_vix_sample.csv",
        market="INDIA",
        source="nse_india_vix",
        symbol="INDIA_VIX",
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert df["source"].unique().tolist() == ["nse_india_vix"]
    assert df["market"].unique().tolist() == ["INDIA"]
    assert df["symbol"].unique().tolist() == ["INDIA_VIX"]
    assert (df["volume"] == 0).all()
    validate_ohlcv_frame(df)


def test_nse_loader_requires_local_csv_path() -> None:
    loader = NseLocalCsvLoader(
        {
            "source_id": "nse_india_vix",
            "market": "INDIA",
            "dataset": "india_vix",
            "role": "implied_volatility",
            "symbol": "INDIA_VIX",
            "provider": "nse",
            "access_method": "local_csv_first",
            "start_date": "2024-01-01",
            "end_date": None,
            "local_csv_path": None,
            "raw_path": "data/raw/india_vix_nse.parquet",
            "processed_path": "data/processed/india_vix.parquet",
        }
    )

    with pytest.raises(DataIngestionError, match="requires local_csv_path"):
        loader.load()


def test_nse_loader_returns_data_load_result_with_local_csv() -> None:
    loader = NseLocalCsvLoader(
        {
            "source_id": "nse_india_vix",
            "market": "INDIA",
            "dataset": "india_vix",
            "role": "implied_volatility",
            "symbol": "INDIA_VIX",
            "provider": "nse",
            "access_method": "local_csv_first",
            "start_date": "2024-01-01",
            "end_date": None,
            "local_csv_path": str(FIXTURES_DIR / "nse_india_vix_sample.csv"),
            "raw_path": "data/raw/india_vix_nse.parquet",
            "processed_path": "data/processed/india_vix.parquet",
        }
    )

    result = loader.load()

    assert isinstance(result, DataLoadResult)
    assert result.source_id == "nse_india_vix"
    assert result.metadata["access_used"] == "local_csv"
    assert list(result.frame.columns) == OHLCV_COLUMNS
    validate_ohlcv_frame(result.frame)