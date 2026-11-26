"""Fixes for docs/REVIEW.md R-02 .. R-07 (P2 findings of the P1 review)."""

import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.pages.company_analysis import risk
from StockAnalysisSystem.src.pages.screening import valuation_filter as vf
from StockAnalysisSystem.src.valuation import capital

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_r02_discount_adjustment_starts_from_the_shared_rate(aapl, make_company):
    fd = aapl["financial_data"]
    expected = capital.cost_of_capital_for(fd, 0.12)
    assert vf.shared_discount_rate(fd, "Technology", 0.12) == pytest.approx(expected["wacc"])
    bank = make_company("bank")["financial_data"]
    assert vf.shared_discount_rate(bank, "Financials", 0.10) == pytest.approx(
        capital.cost_of_capital_for(bank, 0.10)["cost_of_equity"])
    assert vf.shared_discount_rate(None, "Technology", 0.12) == 0.12


def test_r02_filter_passes_shared_rate_plus_adjustment(data_loader, aapl):
    seen = {}

    class Factory:
        def get_company_valuation(self, symbol, sector=None, financial_data=None, **overrides):
            seen.update(overrides, data=financial_data)
            return {"value_per_share": None}

    from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

    vf.perform_valuation_analysis([{"symbol": "AAPL"}], data_loader, FinancialRatioAnalyzer(), Factory(),
                                  "Sector model", "All", discount_rate_adj=0.0025, workers=1)
    base = vf.shared_discount_rate(seen["data"], "Technology", 0.12)
    assert seen["discount_rate"] == pytest.approx(base + 0.0025)


def test_r03_zero_weight_methods_stay_out_of_the_range():
    result = {"valuation_methods": {"dcf": {"method": "DCF", "value_per_share": 100.0},
                                    "asset": {"method": "Asset", "value_per_share": 900.0}},
              "valuation_summary": {"weights": {"dcf": 1.0, "asset": 0.0}}}
    assert vf.method_values_of(result) == {"DCF": 100.0}


def test_r04_apv_cost_of_debt_slider_survives_a_high_risk_free_rate(monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RISK_FREE_RATE", "12")
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    kd = at.slider(key="val_apv_kd")
    assert kd.max >= 20.0 and kd.value >= 12.5


def test_r05_peer_helpers_are_defined_before_the_main_guard():
    from StockAnalysisSystem.src.pages.comparison import peer_comparison

    source = inspect.getsource(peer_comparison)
    guard = source.index('if __name__ == "__main__":')
    assert source.index("def valuation_rates") < guard and source.index("def valuation_method_label") < guard


def test_r06_only_banks_and_insurers_lose_the_cash_flow_component(make_company, aapl):
    bank = make_company("bank")
    assert risk.is_balance_sheet_lender("BANK", bank["financial_data"], bank["company_info"], "Financials")
    payments = {"industry": "Credit Services", "sector": "Financial Services"}
    assert not risk.is_balance_sheet_lender("V", aapl["financial_data"], payments, "Financials")
    assert not risk.is_balance_sheet_lender("AAPL", aapl["financial_data"], aapl["company_info"], "Technology")
    fee_based = risk.health_score(aapl["financial_data"], "Financials", lender=False)
    assert fee_based["components"]["cash_flow"] is not None


def test_r07_method_difference_text_is_not_absolute():
    from StockAnalysisSystem.src.pages.company_analysis import valuation

    source = inspect.getsource(valuation._render_method_differences)
    assert "not from the discount rate" not in source
    assert "cost of equity" in source
