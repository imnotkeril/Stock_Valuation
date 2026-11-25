"""End-to-end smoke tests of the Streamlit application shell (offline)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")
PAGE_ERRORS = ("could not be displayed", "This page is unavailable")


def _errors(at: AppTest):
    return [e.value for e in at.error]


def _assert_clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    failures = [message for message in _errors(at) if any(marker in message for marker in PAGE_ERRORS)]
    assert not failures, failures


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.run()
    return at


def test_home_renders(app):
    _assert_clean(app)
    assert app.sidebar.radio(key=nav.NAV_KEY).value == nav.HOME


@pytest.mark.parametrize("page", [p for p in nav.PAGES if p != nav.COMPANY_ANALYSIS])
def test_every_page_renders(app, page):
    app.sidebar.radio(key=nav.NAV_KEY).set_value(page).run()
    _assert_clean(app)


def test_company_analysis_requires_selection(app):
    app.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    _assert_clean(app)
    assert any("Analyze company" in info.value for info in app.info)


@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "XOM", "NEE", "PLD"])
def test_company_analysis_flow(app, ticker):
    app.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    app.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    app.sidebar.button(key="ca_analyze").click().run()
    _assert_clean(app)
    headers = [h.value for h in app.header]
    assert any(f"({ticker})" in h for h in headers), headers
    assert len(app.tabs) >= 5

    # Any further interaction must keep the analysis on screen
    app.sidebar.radio(key="ca_period").set_value("quarterly").run()
    _assert_clean(app)
    assert any(f"({ticker})" in h.value for h in app.header)


def test_search_selects_company(app):
    app.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    app.sidebar.text_input(key="ca_search").input("micro").run()
    choice = app.sidebar.selectbox(key="ca_search_choice")
    assert choice.value.startswith("MSFT")
    app.sidebar.button(key="ca_analyze").click().run()
    _assert_clean(app)
    assert any("(MSFT)" in h.value for h in app.header)


def test_unknown_ticker_is_reported(app):
    app.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    app.sidebar.text_input(key="ca_search").input("ZZZZQ").run()
    app.sidebar.button(key="ca_analyze").click().run()
    assert not app.exception
    assert any("No market data found" in e for e in _errors(app))


def test_navigation_from_other_page_opens_company(app):
    app.session_state[nav.ACTIVE_TICKER_KEY] = "MSFT"
    app.session_state[nav.NAV_KEY] = nav.COMPANY_ANALYSIS
    app.run()
    _assert_clean(app)
    assert any("(MSFT)" in h.value for h in app.header)
    assert app.sidebar.selectbox(key="ca_popular").value == "MSFT"
