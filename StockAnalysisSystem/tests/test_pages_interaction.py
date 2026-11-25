"""Click every button on every top-level page and on the company analysis page (offline)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")
PAGES = [p for p in nav.PAGES if p != nav.COMPANY_ANALYSIS]


def _problems(at: AppTest):
    return [f"error: {e.value}" for e in at.error] + [f"exception: {e.value}" for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_every_button_works(page):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(page).run()
    keys = [b.key for b in at.button if b.key and not b.key.startswith("sidebar_")]
    problems = []
    for key in keys:
        at.button(key=key).click().run()
        problems += [f"[{key}] {p}" for p in _problems(at)]
        if at.sidebar.radio(key=nav.NAV_KEY).value != page:  # navigation buttons move to another page
            at.sidebar.radio(key=nav.NAV_KEY).set_value(page).run()
    assert not problems, problems


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JPM", "XOM", "NEE", "PLD", "JNJ", "CAT", "VZ", "AMZN"])
def test_company_tabs_have_no_errors(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not _problems(at), _problems(at)
    assert not at.warning, [w.value for w in at.warning]


def _open_company(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    return at


def test_dcf_and_apv_use_the_shared_cost_of_capital(aapl):
    from StockAnalysisSystem.src import config
    from StockAnalysisSystem.src.valuation import capital

    at = _open_company("AAPL")
    expected = capital.cost_of_capital_for(aapl["financial_data"], 0.10)
    discount = at.slider(key="val_dcf_discount").value
    assert discount == pytest.approx(round(expected["wacc"] * 100, 2), abs=0.3)
    assert discount != config.SECTOR_DCF_PARAMETERS["Technology"]["default_discount_rate"] * 100
    captions = [c.value for c in at.caption]
    assert any(f"Default WACC {discount:.2f}%" in c and "fallback constant" in c for c in captions), captions
    assert any(c.startswith("Same inputs as the WACC: rf 4.50 %") for c in captions), captions
    labels = [e.label for e in at.expander]
    assert labels.count("Why do DCF, APV and the Sector Model differ?") == 2, labels
