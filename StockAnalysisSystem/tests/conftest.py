"""Shared pytest fixtures. All tests run fully offline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("STOCK_ANALYSIS_OFFLINE", "1")

from StockAnalysisSystem.src.utils.financial_data import build_financial_data  # noqa: E402
from StockAnalysisSystem.tests import synthetic  # noqa: E402
from StockAnalysisSystem.tests.offline import (  # noqa: E402
    OfflineDataLoader,
    install_offline_data_loader,
    load_real_info,
    load_real_prices,
    load_real_statement,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """Never hit the network and never write into the repository cache."""
    install_offline_data_loader(monkeypatch)
    from StockAnalysisSystem.src import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache", raising=False)
    (tmp_path / "cache").mkdir(exist_ok=True)

    def _no_network(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")

    try:
        import yfinance

        monkeypatch.setattr(yfinance, "Ticker", _no_network)
        monkeypatch.setattr(yfinance, "download", _no_network)
    except ImportError:  # pragma: no cover
        pass
    import requests

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests.Session, "request", _no_network)


@pytest.fixture
def data_loader(tmp_path) -> OfflineDataLoader:
    return OfflineDataLoader(cache_dir=str(tmp_path / "loader_cache"))


def _real_company(ticker: str) -> dict:
    info = load_real_info(ticker)
    prices = load_real_prices(ticker)
    statements = {
        "income_statement": load_real_statement(ticker, "income"),
        "balance_sheet": load_real_statement(ticker, "balance"),
        "cash_flow": load_real_statement(ticker, "cash"),
    }
    return {
        "ticker": ticker,
        "company_info": info,
        "price_data": prices,
        "financial_data": build_financial_data(company_info=info, price_data=prices, **statements),
    }


@pytest.fixture
def aapl() -> dict:
    """Real Apple data recorded from yfinance (FY2020-FY2024)."""
    return _real_company("AAPL")


@pytest.fixture
def msft() -> dict:
    """Real Microsoft data recorded from yfinance (FY2020-FY2024)."""
    return _real_company("MSFT")


@pytest.fixture
def make_company():
    """Factory: ``make_company("bank")`` -> synthetic company bundle (see tests/synthetic.py)."""
    return synthetic.make_company


@pytest.fixture(params=sorted(synthetic.PROFILES))
def any_synthetic_company(request) -> dict:
    """Parametrised over every synthetic profile."""
    return synthetic.make_company(request.param)
