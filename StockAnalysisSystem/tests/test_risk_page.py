"""Risk tab for banks (review R5 / U4): no FCF penalty, distress text once, explained stress test."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.pages.company_analysis.risk import health_score

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_bank_health_score_has_no_cash_flow_component(make_company):
    bank = make_company("bank")["financial_data"]
    as_bank = health_score(bank, "Financials")
    assert as_bank["components"]["cash_flow"] is None
    assert as_bank["overall_score"] is not None
    assert not [w for w in as_bank["weaknesses"] if "cash flow" in w.lower()]


def test_non_bank_keeps_cash_flow_component(aapl):
    assert health_score(aapl["financial_data"], "Technology")["components"]["cash_flow"] is not None


def _risk_texts(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return [c.value for c in at.caption] + [i.value for i in at.info] + [m.value for m in at.markdown]


def test_bank_risk_tab_texts():
    texts = _risk_texts("JPM")
    altman = [t for t in texts if "Altman" in t and len(t) > 40]
    assert len(altman) == len(set(" ".join(t.split()).lower() for t in altman)), altman
    assert any("does not apply to banks and insurers" in t and "operating leverage" in t for t in texts)
    assert not any("Revenue and operating income are required" in t for t in texts)
    assert any("cash-flow component is excluded" in t for t in texts)
    assert any("r = rf + ERP" in t for t in _risk_texts("AAPL"))
