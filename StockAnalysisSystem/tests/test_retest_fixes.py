"""Regression tests for bugs found in the full-system re-test (2026-11-27, docs/QA.md)."""

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
from StockAnalysisSystem.tests import synthetic
from StockAnalysisSystem.tests.offline import OfflineDataLoader


def _value(kind: str, **kwargs):
    company = synthetic.make_company(kind)
    factory = ValuationFactory(OfflineDataLoader())
    return factory.get_company_valuation(company["ticker"], sector=company["company_info"].get("sector"),
                                         financial_data=company["financial_data"], **kwargs)


def test_utility_uses_subsector_terminal_growth():
    """RT-01: the utilities subsector default (2 %) beat by the generic sector table (1 %)."""
    result = _value("utility")
    assumptions = result["assumptions"]
    assert assumptions["terminal_growth"] == pytest.approx(0.02)
    assert assumptions["parameter_sources"]["terminal_growth"] == "default"


def test_company_specific_discount_rate_still_wins_over_subsector_default():
    """The shared WACC (ADR-007) stays the default discount rate; the 7 % subsector value is a slider hint only."""
    result = _value("utility")
    assumptions = result["assumptions"]
    assert assumptions["parameter_sources"]["discount_rate"] == "derived"
    assert assumptions["discount_rate"] == pytest.approx(result["assumptions"]["cost_of_capital"]["wacc"])


def _ui_script(kind: str):
    from StockAnalysisSystem.src.pages.company_analysis.sector_valuation import render_sector_valuation
    from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
    from StockAnalysisSystem.tests import synthetic
    from StockAnalysisSystem.tests.offline import OfflineDataLoader

    company = synthetic.make_company(kind)
    factory = ValuationFactory(OfflineDataLoader())
    render_sector_valuation(company["ticker"], company["financial_data"], company["company_info"],
                            company["price_data"], key_prefix="t", factory=factory)


def _fair_value_text(at) -> str:
    return next(m.value for m in at.metric if m.label == "Fair value / share")


@pytest.mark.parametrize("kind", ["utility", "industrial"])
def test_override_toggle_without_changes_keeps_the_result(kind):
    """RT-01: sliders started at static defaults (e.g. 9 % discount rate) instead of the values the model used,
    so switching "Override model assumptions" on, without touching anything, changed the fair value."""
    at = AppTest.from_function(_ui_script, args=(kind,), default_timeout=120)
    at.run()
    assert not at.exception
    before = _fair_value_text(at)
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert _fair_value_text(at) == before


def test_moved_slider_still_overrides():
    at = AppTest.from_function(_ui_script, args=("industrial",), default_timeout=120)
    at.run()
    before = _fair_value_text(at)
    at.toggle[0].set_value(True).run()
    discount = next(s for s in at.slider if "discount" in s.label.lower())
    discount.set_value(discount.max).run()
    assert not at.exception
    assert _fair_value_text(at) != before


def test_utility_terminal_growth_agrees_between_dcf_tab_and_sector_model():
    """RT-02: the DCF and APV tabs read the Utilities sector table (1 %) while the utility model uses 2 %."""
    import pandas as pd

    from StockAnalysisSystem.src.pages.company_analysis.valuation import _dcf_defaults

    tab = _dcf_defaults("Utilities", pd.Series([1.0, 1.1, 1.2]))
    assert tab["terminal_growth"] == pytest.approx(_value("utility")["assumptions"]["terminal_growth"])


def _market_risk_script():
    from StockAnalysisSystem.src.pages.company_analysis import risk
    from StockAnalysisSystem.tests import synthetic

    original = risk._benchmark_close
    risk._benchmark_close = lambda start, end: __import__("pandas").Series(dtype=float)  # offline
    try:
        company = synthetic.make_company("industrial")
        risk._render_market_risk(company["price_data"], company["financial_data"])
    finally:
        risk._benchmark_close = original


def test_provider_beta_is_labelled_apart_from_regression_beta():
    """RT-03: an unlabelled "Beta" (provider, 5y monthly) sat next to ratios that use the regression beta."""
    at = AppTest.from_function(_market_risk_script, default_timeout=120)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Beta (provider)" in labels and "Beta" not in labels


@pytest.mark.parametrize("subsector, expected", [("midstream", 0.015), ("renewables", 0.02), ("utilities", 0.02)])
def test_energy_subsector_terminal_growth_defaults_are_applied(subsector, expected):
    """RT-01 also activates the declared midstream / renewables defaults (were 1 % from the Energy table)."""
    result = _value("oil_gas", subsector=subsector)
    assert result["assumptions"]["terminal_growth"] == pytest.approx(expected)
