"""Tests for the real DataLoader implementation with a fake yfinance backend."""

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from StockAnalysisSystem.src.utils import data_loader as dl_module
from StockAnalysisSystem.src.utils.data_loader import (
    DataLoader, display_symbol, normalize_price_frame, provider_symbol,
)
from StockAnalysisSystem.tests.offline import ORIGINAL_METHODS, load_real_statement


@pytest.fixture
def real_loader(monkeypatch, tmp_path):
    """A DataLoader with its real methods restored and a fake yfinance module."""
    for name, method in ORIGINAL_METHODS.items():
        monkeypatch.setattr(DataLoader, name, method)

    calls = {"history": 0, "info": 0, "statements": 0}
    index = pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York")
    history = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": [10, 11, 12, 13, 14],
                            "Adj Close": [10, 11, 12, 13, 14], "Volume": 100,
                            "Dividends": 0.0, "Stock Splits": 0.0}, index=index)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls["history"] += 1
            return history.copy()

        @property
        def info(self):
            calls["info"] += 1
            return {"longName": "Fake Corp", "sector": "Financial Services", "industry": "Banks - Diversified",
                    "marketCap": 1e9, "currentPrice": 50.0, "sharesOutstanding": 2e7, "dividendYield": 2.5,
                    "beta": 1.2, "quoteType": "EQUITY", "returnOnEquity": 0.12}

        @property
        def income_stmt(self):
            calls["statements"] += 1
            return load_real_statement("AAPL", "income")

    fake_yf = SimpleNamespace(Ticker=FakeTicker, download=lambda *a, **k: pd.DataFrame(),
                              Search=lambda *a, **k: SimpleNamespace(quotes=[
                                  {"symbol": "BRK-B", "longname": "Berkshire Hathaway", "quoteType": "EQUITY"}]))
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    loader = DataLoader(cache_dir=str(tmp_path / "cache"), offline=False)
    return loader, calls


def test_symbol_helpers():
    assert provider_symbol(" brk.b ") == "BRK-B"
    assert display_symbol("BRK-B") == "BRK.B"
    assert display_symbol("RDS-AB") == "RDS-AB"


def test_normalize_price_frame_multiindex_and_tz():
    index = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    columns = pd.MultiIndex.from_product([["Close", "Open", "Volume"], ["AAPL"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(9).reshape(3, 3), index=index, columns=columns)
    frame = normalize_price_frame(raw)
    assert frame.index.tz is None and frame.index.name == "Date"
    assert {"Open", "High", "Low", "Close", "Adj Close", "Volume"} <= set(frame.columns)
    assert (frame["Adj Close"] == frame["Close"]).all()


def test_normalize_price_frame_empty():
    assert normalize_price_frame(None).empty
    assert normalize_price_frame(pd.DataFrame({"Volume": [1]})).empty


def test_prices_are_cached(real_loader):
    loader, calls = real_loader
    first = loader.get_historical_prices("AAPL", "2024-01-01", "2024-01-31")
    second = loader.get_historical_prices("aapl", "2024-01-01", "2024-01-31")
    assert calls["history"] == 1
    assert first.index.tz is None and len(first) == 5
    pd.testing.assert_frame_equal(first, second)


def test_stale_cache_served_when_provider_fails(real_loader, monkeypatch):
    loader, calls = real_loader
    loader.get_historical_prices("AAPL", "2024-01-01", "2024-01-31")
    for path in loader.cache_dir.glob("AAPL_*"):
        old = time.time() - 30 * 86400
        os.utime(path, (old, old))
    monkeypatch.setattr(DataLoader, "_fetch_yfinance", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(DataLoader, "_fetch_alpha_vantage", lambda *a, **k: pd.DataFrame())
    data = loader.get_historical_prices("AAPL", "2024-01-01", "2024-01-31")
    assert len(data) == 5


def test_offline_mode_never_fetches(real_loader, tmp_path):
    _, calls = real_loader
    loader = DataLoader(cache_dir=str(tmp_path / "offline"), offline=True)
    assert loader.get_historical_prices("AAPL").empty
    assert loader.get_company_info("AAPL") == {}
    assert loader.get_financial_statements("AAPL").empty
    assert calls == {"history": 0, "info": 0, "statements": 0}


def test_company_info_normalisation(real_loader):
    loader, _ = real_loader
    info = loader.get_company_info("TEST")
    assert info["sector"] == "Financials" and info["sector_raw"] == "Financial Services"
    assert info["dividend_yield"] == pytest.approx(0.025)
    assert info["current_price"] == 50.0 and info["shares_outstanding"] == 2e7
    assert info["type"] == "Stock" and info["return_on_equity"] == 0.12


def test_cached_legacy_info_is_normalised(real_loader):
    loader, calls = real_loader
    legacy = {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "dividend_yield": 0.53}
    (loader.cache_dir / "AAPL_info_yfinance.json").write_text(json.dumps(legacy))
    info = loader.get_company_info("AAPL")
    assert calls["info"] == 0
    assert info["sector"] == "Technology" and info["dividend_yield"] == pytest.approx(0.0053)


def test_financial_statements(real_loader):
    loader, calls = real_loader
    income = loader.get_financial_statements("AAPL", "income", "annual")
    assert "Total Revenue" in income.index
    assert list(income.columns) == sorted(income.columns, reverse=True)
    loader.get_financial_statements("AAPL", "income", "annual")
    assert calls["statements"] == 1
    with pytest.raises(ValueError):
        loader.get_financial_statements("AAPL", "bogus")


def test_search_prefers_universe_then_provider(real_loader):
    loader, _ = real_loader
    results = loader.search_tickers("apple")
    assert results[0]["symbol"] == "AAPL"
    berkshire = loader.search_tickers("berkshire")
    assert berkshire[0]["symbol"] == "BRK.B"
    assert loader.search_tickers("  ") == []


def test_sector_data_uses_sector_etf(real_loader, monkeypatch):
    loader, _ = real_loader
    seen = {}
    monkeypatch.setattr(DataLoader, "get_historical_prices",
                        lambda self, ticker, **kwargs: seen.setdefault("ticker", ticker) and pd.DataFrame())
    loader.get_sector_data("Financial Services")
    assert seen["ticker"] == "XLF"
    assert loader.get_sector_data("Nonexistent").empty


def test_batch_data_dedupes(real_loader):
    loader, calls = real_loader
    batch = loader.get_batch_data(["AAPL", "aapl", "MSFT", ""], "2024-01-01", "2024-01-31")
    assert set(batch) == {"AAPL", "MSFT"}


def test_clear_cache(real_loader):
    loader, _ = real_loader
    loader.get_historical_prices("AAPL", "2024-01-01", "2024-01-31")
    assert loader.clear_cache(["AAPL"]) >= 1
    assert not list(loader.cache_dir.glob("AAPL_*"))


def test_module_has_no_hardcoded_keys():
    source = open(dl_module.__file__).read()
    assert "0VJE73FSCQIPH601" not in source
