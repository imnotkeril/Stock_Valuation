import pandas as pd
import pytest

from StockAnalysisSystem.src.utils.data_loader import DataLoader


def test_cache_dir_created(tmp_path):
    loader = DataLoader(cache_dir=str(tmp_path / "cache"))
    assert (tmp_path / "cache").is_dir()
    assert set(loader.providers) == {"yfinance", "alpha_vantage"}


def test_prices_served_from_provider_then_cache(tmp_path):
    loader = DataLoader(cache_dir=str(tmp_path))
    calls = []
    frame = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"]))

    def fake_fetch(ticker, start, end, interval):
        calls.append(ticker)
        return frame

    loader.providers["yfinance"] = fake_fetch
    first = loader.get_historical_prices("TEST", "2025-01-01", "2025-01-10")
    second = loader.get_historical_prices("TEST", "2025-01-01", "2025-01-10")
    assert len(calls) == 1  # second call hit the cache
    assert list(second["Close"]) == list(first["Close"]) == [1.0, 2.0]


def test_unknown_provider_rejected(tmp_path):
    loader = DataLoader(cache_dir=str(tmp_path))
    with pytest.raises(Exception):
        loader.get_historical_prices("TEST", provider="nope")
