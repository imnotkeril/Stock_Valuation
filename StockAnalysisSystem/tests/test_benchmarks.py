"""Live sector medians from cached universe data and the multiple relevance filter (review R8)."""

import json

import pytest

from StockAnalysisSystem.src.industry import benchmarks as bm
from StockAnalysisSystem.src.industry.universe import get_universe
from StockAnalysisSystem.src.utils.data_loader import DataLoader


@pytest.fixture
def loader(tmp_path):
    return DataLoader(cache_dir=str(tmp_path), offline=True)


def _cache(loader, symbol, **multiples):
    loader._cache_path(f"{symbol}_info_yfinance.json").write_text(
        json.dumps({"name": symbol, **multiples}), encoding="utf-8")


def test_medians_from_cached_companies_only(loader):
    tech = [r["symbol"] for r in get_universe("Technology")]
    for i, symbol in enumerate(tech[:7]):
        _cache(loader, symbol, pe_ratio=20.0 + i, pb_ratio=None, ev_ebitda=-5.0 if i == 0 else 15.0)
    live = bm.live_valuation_medians("Technology", loader, exclude=tech[0])
    assert live["companies"] == 6
    assert live["values"]["pe_ratio"] == pytest.approx(23.5)  # 21..26
    assert live["values"]["ev_ebitda"] == pytest.approx(15.0)
    assert "pb_ratio" not in live["values"] and live["counts"]["pe_ratio"] == 6
    assert live["as_of"] is not None


def test_too_few_companies_means_no_live_medians(loader):
    for symbol in [r["symbol"] for r in get_universe("Energy")][:3]:
        _cache(loader, symbol, pe_ratio=12.0)
    assert bm.live_valuation_medians("Energy", loader) is None


def test_implausible_values_ignored(loader):
    tech = [r["symbol"] for r in get_universe("Technology")]
    for symbol in tech[:5]:
        _cache(loader, symbol, pe_ratio=5000.0)
    assert bm.live_valuation_medians("Technology", loader) is None


def test_relevance_filter():
    assert "pb_ratio" in bm.irrelevant_multiples("Technology")
    assert bm.irrelevant_multiples("Financials").keys() == {"ps_ratio", "ev_ebitda"}
    assert bm.irrelevant_multiples("Real Estate") == {}


def test_relative_tab_falls_back_to_static(monkeypatch):
    from StockAnalysisSystem.src.pages.company_analysis import valuation

    monkeypatch.setattr(valuation, "live_valuation_medians", lambda sector, exclude=None: None)
    _, source = valuation.relative_benchmarks("Technology", "AAPL")
    assert source.startswith("static long-run")
    monkeypatch.setattr(valuation, "live_valuation_medians", lambda sector, exclude=None: {
        "values": {"pe_ratio": 30.0}, "counts": {"pe_ratio": 9}, "companies": 9, "as_of": "2026-11-26"})
    merged, source = valuation.relative_benchmarks("Technology", "AAPL")
    assert merged["pe_ratio"] == 30.0 and "P/E n=9" in source and "2026-11-26" in source
