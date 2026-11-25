"""
Offline data layer for tests.

``OfflineDataLoader`` serves the real AAPL/MSFT fixtures recorded from
yfinance plus synthetic companies from :mod:`synthetic`, and never touches
the network. ``install_offline_data_loader`` patches the public methods of
``DataLoader`` itself so every component (including Streamlit pages that
build their own loaders) uses the offline data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from StockAnalysisSystem.src.config import normalize_sector
from StockAnalysisSystem.src.industry.universe import SECTOR_ETFS, lookup_company, search_universe
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.tests import synthetic

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_TICKERS = ("AAPL", "MSFT")
STATEMENT_FILES = {"income": "income_annual.csv", "balance": "balance_annual.csv", "cash": "cash_annual.csv"}


def load_real_statement(ticker: str, statement_type: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / ticker / STATEMENT_FILES[statement_type], index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df


def load_real_info(ticker: str) -> Dict:
    return json.loads((FIXTURES / ticker / "info.json").read_text())


def load_real_prices(ticker: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / ticker / "prices.csv", index_col=0, parse_dates=True)


def _filter_dates(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    return df


def _synthetic_profile_for(ticker: str) -> Optional[synthetic.Profile]:
    """Synthetic profile for a synthetic symbol or a known universe company."""
    ticker = ticker.upper()
    if ticker in synthetic.SYMBOL_TO_KIND:
        return synthetic.PROFILES[synthetic.SYMBOL_TO_KIND[ticker]]
    record = lookup_company(ticker)
    if record is None and ticker not in SECTOR_ETFS.values() and ticker != "SPY":
        return None
    # Derive a deterministic profile from the closest synthetic sector archetype
    sector = record["sector"] if record else "Technology"
    archetype = {
        "Technology": "saas", "Healthcare": "pharma", "Financials": "bank",
        "Consumer Discretionary": "retail", "Consumer Staples": "staples", "Energy": "oil_gas",
        "Industrials": "industrial", "Materials": "materials", "Real Estate": "reit",
        "Communication Services": "telecom", "Utilities": "utility",
    }[sector]
    base = synthetic.PROFILES[archetype]
    seed = sum(map(ord, ticker))
    return synthetic.Profile(**{
        **base.__dict__,
        "symbol": ticker,
        "name": record["name"] if record else f"{ticker} ETF",
        "revenue": base.revenue * (0.5 + (seed % 10) / 10),
        "price": base.price * (0.6 + (seed % 7) / 10),
        "extra_income_rows": dict(base.extra_income_rows),
        "extra_balance_rows": dict(base.extra_balance_rows),
    })


class OfflineDataLoader(DataLoader):
    """DataLoader that only serves fixture and synthetic data."""

    def get_historical_prices(self, ticker: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None, provider: str = 'yfinance',
                              interval: str = '1d', force_refresh: bool = False) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in REAL_TICKERS:
            prices = load_real_prices(ticker)
        elif ticker == "XLK":
            prices = pd.read_csv(FIXTURES / "XLK_prices.csv", index_col=0, parse_dates=True)
        else:
            profile = _synthetic_profile_for(ticker)
            if profile is None:
                return pd.DataFrame()
            prices = synthetic.make_prices(profile, days=1300)
        filtered = _filter_dates(prices, start_date, end_date)
        # Fixtures end in April 2025: fall back to the most recent window of the
        # requested length so date-relative UI defaults still get data.
        if filtered.empty and start_date:
            span = (pd.Timestamp(end_date or pd.Timestamp.now()) - pd.Timestamp(start_date)).days
            filtered = prices[prices.index >= prices.index.max() - pd.Timedelta(days=max(span, 5))]
        return filtered.copy()

    def get_company_info(self, ticker: str, provider: str = 'yfinance') -> Dict:
        ticker = ticker.upper()
        if ticker in REAL_TICKERS:
            return load_real_info(ticker)
        profile = _synthetic_profile_for(ticker)
        return synthetic.make_company_info(profile) if profile else {}

    def get_financial_statements(self, ticker: str, statement_type: str = 'income',
                                 period: str = 'annual', force_refresh: bool = False) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in REAL_TICKERS:
            return load_real_statement(ticker, statement_type)
        profile = _synthetic_profile_for(ticker)
        if profile is None:
            return pd.DataFrame()
        key = {"income": "income_statement", "balance": "balance_sheet", "cash": "cash_flow"}[statement_type]
        return synthetic.make_statements(profile)[key]

    def get_sector_data(self, sector: str, force_refresh: bool = False) -> pd.DataFrame:
        etf = SECTOR_ETFS.get(normalize_sector(sector))
        return self.get_historical_prices(etf) if etf else pd.DataFrame()

    def search_tickers(self, query: str, limit: int = 10, provider: str = 'alpha_vantage') -> List[Dict]:
        return [{"symbol": r["symbol"], "name": r["name"], "type": "Equity", "region": "United States",
                 "currency": "USD", "sector": r["sector"]} for r in search_universe(query, limit)]

    def get_macro_indicators(self, indicators: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        return {}


PATCHED_METHODS = ("get_historical_prices", "get_company_info", "get_financial_statements",
                   "get_sector_data", "search_tickers", "get_macro_indicators")

#: The real implementations, captured before any patching (for DataLoader's own tests)
ORIGINAL_METHODS = {name: getattr(DataLoader, name) for name in PATCHED_METHODS}


def install_offline_data_loader(monkeypatch) -> None:
    """Patch DataLoader's network-facing methods with the offline versions."""
    for name in PATCHED_METHODS:
        monkeypatch.setattr(DataLoader, name, getattr(OfflineDataLoader, name))
