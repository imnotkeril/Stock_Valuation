"""Tests for shared helpers: financial_data utilities, sector normalisation, universe, factory."""

import math

import pandas as pd
import pytest

from StockAnalysisSystem.src.config import normalize_sector, SECTORS
from StockAnalysisSystem.src.industry.universe import (
    SECTOR_ETFS, UNIVERSE, get_sector_peers, get_universe, search_universe,
)
from StockAnalysisSystem.src.utils import financial_data as fd
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation, get_recommendation
from StockAnalysisSystem.src.valuation.sector_factor import MODEL_CLASSES, ValuationFactory


def test_get_latest_uses_aliases_and_most_recent_period(aapl):
    income = aapl["financial_data"]["income_statement"]
    assert fd.get_latest(income, "revenue") == pytest.approx(391_035_000_000)
    assert fd.get_value(income, "revenue", period=1) == pytest.approx(383_285_000_000)
    assert fd.find_row(income, "total revenue") == "Total Revenue"  # case-insensitive
    assert fd.get_latest(income, "does_not_exist") is None


def test_get_series_chronological(aapl):
    series = fd.get_series(aapl["financial_data"]["cash_flow"], "free_cash_flow", ascending=True)
    assert list(series.index) == sorted(series.index)
    assert len(series) == 4  # FY2020 is missing in the yfinance data and is dropped
    assert series.notna().all()


def test_find_row_skips_all_nan_rows():
    df = pd.DataFrame({"2024": [float("nan"), 5.0]}, index=["Total Revenue", "Operating Revenue"])
    assert fd.find_row(df, "revenue") == "Operating Revenue"


def test_safe_divide_and_cagr():
    assert fd.safe_divide(1, 0) is None
    assert fd.safe_divide(None, 2, default=0) == 0
    assert fd.cagr(pd.Series([100.0, 121.0])) == pytest.approx(0.21)
    assert fd.cagr(pd.Series([-1.0, 2.0])) is None


def test_build_financial_data_market_data(aapl):
    market = aapl["financial_data"]["market_data"]
    assert market["share_price"] > 0
    assert market["shares_outstanding"] > 1e10
    assert market["dividend_yield"] == pytest.approx(0.0053)


@pytest.mark.parametrize("raw, expected", [(0.53, 0.0053), (0.025, 0.025), (None, None), (3.4, 0.034)])
def test_normalize_dividend_yield(raw, expected):
    result = fd.normalize_dividend_yield(raw)
    assert (result is None and expected is None) or result == pytest.approx(expected)


@pytest.mark.parametrize("raw, canonical", [
    ("Financial Services", "Financials"), ("Consumer Cyclical", "Consumer Discretionary"),
    ("Consumer Defensive", "Consumer Staples"), ("Basic Materials", "Materials"),
    ("Information Technology", "Technology"), ("Health Care", "Healthcare"), (None, "Unknown"),
])
def test_normalize_sector(raw, canonical):
    assert normalize_sector(raw) == canonical


def test_universe_is_consistent():
    symbols = [c["symbol"] for c in UNIVERSE]
    assert len(symbols) == len(set(symbols)), "duplicate tickers in universe"
    assert {c["sector"] for c in UNIVERSE} == set(SECTORS) == set(SECTOR_ETFS)
    assert all(get_universe(sector) for sector in SECTORS)


def test_sector_peers_exclude_self_and_prefer_subsector():
    peers = get_sector_peers("JPM", "Financial Services", limit=3)
    assert [p["symbol"] for p in peers] == ["BAC", "WFC", "C"]
    assert all(p["symbol"] != "JPM" for p in get_sector_peers("JPM", "Financials", limit=50))


def test_search_universe():
    assert search_universe("aapl")[0]["symbol"] == "AAPL"
    assert any(r["symbol"] == "MSFT" for r in search_universe("micro"))


def test_summarize_valuations_weights_and_exclusions():
    summary = BaseValuation.summarize_valuations(
        {"a": {"value_per_share": 100}, "b": {"value_per_share": 200}, "c": {"value_per_share": -1},
         "d": {"value_per_share": None, "error": "boom"}, "e": {"value_per_share": float("nan")}},
        {"a": 3, "b": 1}, current_price=125)
    assert summary["weighted_value"] == pytest.approx(125)
    assert summary["potential_upside"] == pytest.approx(0)
    assert set(summary["excluded"]) == {"c", "d", "e"}
    assert math.isclose(sum(summary["weights"].values()), 1)


def test_summarize_valuations_empty():
    assert BaseValuation.summarize_valuations({}, None, 10)["weighted_value"] is None


@pytest.mark.parametrize("upside, label", [(30, "Strong Buy"), (15, "Buy"), (0, "Hold"),
                                           (-15, "Reduce"), (-40, "Sell"), (None, "N/A")])
def test_recommendation(upside, label):
    assert get_recommendation(upside)["label"] == label


@pytest.mark.parametrize("sector, model", [
    ("Financial Services", "FinancialSectorValuation"), ("Technology", "TechnologySectorValuation"),
    ("Energy", "EnergySectorValuation"), ("Utilities", "EnergySectorValuation"),
    ("Consumer Cyclical", "RetailSectorValuation"), ("Consumer Defensive", "RetailSectorValuation"),
    ("Industrials", "ManufacturingSectorValuation"), ("Basic Materials", "ManufacturingSectorValuation"),
    ("Real Estate", "RealEstateValuation"), ("Healthcare", "HealthcareValuation"),
    ("Communication Services", "CommunicationValuation"), ("Unknown", "BaseValuation"),
])
def test_factory_model_selection(data_loader, sector, model):
    assert ValuationFactory(data_loader).get_valuation_model(sector).__class__.__name__ == model


def test_factory_industry_override(data_loader):
    factory = ValuationFactory(data_loader)
    assert factory.get_model_key("Consumer Cyclical", "Auto Manufacturers") == "Manufacturing"


def test_all_models_instantiate(data_loader):
    for key, cls in MODEL_CLASSES.items():
        assert isinstance(cls(data_loader), BaseValuation), key
