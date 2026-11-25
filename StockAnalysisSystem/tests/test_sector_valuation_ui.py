"""Tests for the generic sector valuation UI (pages/company_analysis/sector_valuation.py)."""

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src.pages.company_analysis.sector_valuation import (
    _methods_table, _scalar_fields, _series_fields, format_value,
)


def _script(kind: str):
    from StockAnalysisSystem.src.pages.company_analysis.sector_valuation import render_sector_valuation
    from StockAnalysisSystem.tests import synthetic
    from StockAnalysisSystem.tests.offline import OfflineDataLoader
    from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

    company = synthetic.make_company(kind) if kind != "AAPL" else None
    if company is None:
        from StockAnalysisSystem.tests.conftest import _real_company
        company = _real_company("AAPL")
    factory = ValuationFactory(OfflineDataLoader())
    render_sector_valuation(company["ticker"], company["financial_data"], company["company_info"],
                            company["price_data"], key_prefix="t", factory=factory)


@pytest.mark.parametrize("kind", ["AAPL", "bank", "oil_gas", "utility", "reit", "retail", "industrial",
                                  "pharma", "biotech_loss", "telecom"])
def test_renders_for_every_sector(kind):
    at = AppTest.from_function(_script, args=(kind,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox, "subsector selector missing"
    assert at.multiselect, "method selector missing"


def test_changing_subsector_and_overriding_assumptions():
    at = AppTest.from_function(_script, args=("oil_gas",), default_timeout=120)
    at.run()
    subsector = at.selectbox[0]
    options = subsector.options
    assert len(options) > 1
    subsector.set_value(subsector.options[-1] if subsector.value != subsector.options[-1] else subsector.options[0])
    at.run()
    assert not at.exception
    if at.toggle:
        at.toggle[0].set_value(True).run()
        assert not at.exception
        if at.slider:
            slider = at.slider[0]
            slider.set_value(slider.max).run()
            assert not at.exception


def test_deselecting_all_methods_shows_hint():
    at = AppTest.from_function(_script, args=("AAPL",), default_timeout=120)
    at.run()
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("at least one valuation method" in i.value for i in at.info)


@pytest.mark.parametrize("key, value, expected", [
    ("value_per_share", 12.3456, "$12.35"), ("discount_rate", 0.1, "10.00%"), ("enterprise_value", 2.5e12, "$2.50T"),
    ("ev_ebitda_multiple", 14.2, "14.20x"), ("forecast_years", 7.0, "7"), ("label", "text", "text"),
])
def test_format_value(key, value, expected):
    assert format_value(key, value) == expected


def test_field_extraction():
    method = {"method": "DCF", "value_per_share": 10.0, "discount_rate": 0.09, "notes": "ok",
              "forecast_fcf": [1.0, 2.0, 3.0], "nested": {"a": {"b": 1}}, "flag": True}
    assert set(_scalar_fields(method)) == {"discount_rate", "notes"}
    assert _series_fields(method) == {"forecast_fcf": [1.0, 2.0, 3.0]}


def test_methods_table_marks_excluded_methods():
    result = {
        "current_price": 100.0,
        "valuation_summary": {"weights": {"dcf": 1.0}, "excluded": {"ddm": "no dividends"}},
        "valuation_methods": {"dcf": {"method": "DCF", "value_per_share": 120.0},
                              "ddm": {"method": "DDM", "value_per_share": None}},
    }
    table = _methods_table(result)
    assert list(table["Status"]) == ["Used", "no dividends"]
    assert table.loc[0, "Upside"] == "+20.0%"
