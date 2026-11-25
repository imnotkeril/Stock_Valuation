"""Per-company widget state is dropped when the analysed company changes (review R3 / U1)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def _analyze(at: AppTest, ticker: str) -> AppTest:
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    return at


@pytest.fixture
def app():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    return at


def test_reset_drops_only_company_widgets():
    from StockAnalysisSystem.src.main import reset_company_widgets

    state = {"val_dcf_discount": 15.0, "val_sector_AAPL_subsector": "x", "risk_contribution_margin": 0.4,
             "ca_popular": "AAPL", nav.NAV_KEY: nav.COMPANY_ANALYSIS, 3: "non-string key"}
    assert sorted(reset_company_widgets(state)) == ["risk_contribution_margin", "val_dcf_discount",
                                                     "val_sector_AAPL_subsector"]
    assert set(state) == {"ca_popular", nav.NAV_KEY, 3}


def test_next_company_gets_its_own_defaults(app):
    _analyze(app, "AAPL")
    aapl_default = app.slider(key="val_dcf_discount").value
    app.slider(key="val_dcf_discount").set_value(15.0).run()
    app.slider(key="val_dcf_years").set_value(12).run()
    assert app.slider(key="val_dcf_discount").value == 15.0

    _analyze(app, "XOM")
    assert not app.exception, [e.value for e in app.exception]
    xom_discount = app.slider(key="val_dcf_discount").value
    assert xom_discount != 15.0 and xom_discount != aapl_default
    assert app.slider(key="val_dcf_years").value != 12


def test_same_company_keeps_user_inputs(app):
    _analyze(app, "AAPL")
    app.slider(key="val_dcf_discount").set_value(15.0).run()
    _analyze(app, "AAPL")
    assert app.slider(key="val_dcf_discount").value == 15.0


def test_bank_after_tech_shows_bank_defaults(app):
    _analyze(app, "AAPL")
    app.slider(key="val_dcf_years").set_value(12).run()
    _analyze(app, "JPM")
    assert not app.exception, [e.value for e in app.exception]
    years = [s.value for s in app.slider if s.key == "val_dcf_years"]
    assert 12 not in years
    assert "val_dcf_years" not in app.session_state or app.session_state["val_dcf_years"] != 12
