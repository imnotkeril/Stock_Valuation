"""Peer Comparison relative-valuation summary (review R5 / U5): real method and rates, no N/A / 0.00 %."""

import pytest

from StockAnalysisSystem.src.pages.comparison.peer_comparison import (
    _pct_or_na,
    valuation_method_label,
    valuation_rates,
)
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory


@pytest.mark.parametrize("kind, model", [("bank", "Financial"), ("saas", "Technology"), ("oil_gas", "Energy")])
def test_method_and_rates_come_from_the_sector_model(make_company, kind, model):
    company = make_company(kind)
    result = ValuationFactory().get_company_valuation(company["ticker"], sector=company["company_info"]["sector"],
                                                      financial_data=company["financial_data"])
    assert valuation_method_label(result).startswith(model)
    rates = valuation_rates(result)
    assert rates["rate"] and 0.04 < rates["rate"] < 0.25
    assert rates["rf_label"].startswith("4.50% (fallback constant")


def test_missing_values_show_na_not_zero():
    rates = valuation_rates({"assumptions": {}})
    assert rates["rate"] is None and rates["terminal_growth"] is None and rates["rf_label"] is None
    assert _pct_or_na(None) == "n/a" and _pct_or_na(0.1) == "10.00%"
    assert valuation_method_label({}) == "N/A"
