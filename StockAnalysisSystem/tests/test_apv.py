"""Adjusted present value (offline, checked against hand calculation)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.valuation import apv

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_hamada_unlevered_beta():
    # beta_L 1.2, D/E 0.5, t 20% -> 1.2 / 1.4 = 0.857
    assert apv.unlevered_beta(1.2, 50, 100, 0.20) == pytest.approx(1.2 / 1.4)
    assert apv.unlevered_beta(1.2, 0, 100, 0.20) == pytest.approx(1.2)
    assert apv.unlevered_cost_of_equity(1.0, 0, 100, 0.2, risk_free=0.03, risk_premium=0.05) == pytest.approx(0.08)


def test_unlevered_fcf_adds_after_tax_interest():
    assert apv.unlevered_fcf(100, -10, 0.25) == pytest.approx(107.5)


def test_apv_matches_hand_calculation():
    # 2-year forecast, UFCF 100 growing 10%: 110, 121; ku 10%, g 2%
    # Tax shield = 25% * 5% * 200 = 2.5 per year at kd 5%, growing 2% after year 2
    r = apv.apv_valuation(100, 0.10, 2, 0.02, 0.10, debt=200, cost_of_debt=0.05, tax_rate=0.25,
                          cash=50, shares=10)
    pv_fcf = 110 / 1.1 + 121 / 1.1 ** 2
    pv_tv = 121 * 1.02 / 0.08 / 1.1 ** 2
    pv_ts = 2.5 / 1.05 + 2.5 / 1.05 ** 2 + 2.5 * 1.02 / 0.03 / 1.05 ** 2
    assert r["pv_forecast_fcf"] == pytest.approx(pv_fcf)
    assert r["pv_terminal_value"] == pytest.approx(pv_tv)
    assert r["tax_shield_value"] == pytest.approx(pv_ts)
    equity = pv_fcf + pv_tv + pv_ts - 200 + 50
    assert r["equity_value"] == pytest.approx(equity)
    assert r["value_per_share"] == pytest.approx(equity / 10)
    assert len(r["projections"]) == 2


def test_no_debt_means_no_tax_shield_and_equals_unlevered_dcf():
    r = apv.apv_valuation(100, 0.0, 5, 0.0, 0.10, debt=0, cost_of_debt=0.05, tax_rate=0.25)
    assert r["tax_shield_value"] == 0
    assert r["enterprise_value"] == pytest.approx(1000.0)  # perpetuity 100 / 10%
    assert r["value_per_share"] is None


@pytest.mark.parametrize("kwargs", [dict(unlevered_cost=0.02, terminal_growth=0.03), dict(years=0)])
def test_invalid_inputs(kwargs):
    base = dict(base_ufcf=100, growth=0.05, years=5, terminal_growth=0.02, unlevered_cost=0.09,
                debt=10, cost_of_debt=0.05, tax_rate=0.2)
    base.update(kwargs)
    with pytest.raises(ValueError):
        apv.apv_valuation(**base)


@pytest.mark.parametrize("ticker", ["AAPL", "JPM"])
def test_apv_tab_renders(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    if ticker == "AAPL":
        assert "APV per share" in labels
    else:
        assert any("APV is not meaningful" in i.value for i in at.info)
