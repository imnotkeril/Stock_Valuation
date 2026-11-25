"""Scenario analysis and historical crises (offline)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.models import stress_testing as stx

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_rate_multiple_factor():
    assert stx.rate_multiple_factor(0) == pytest.approx(1.0)
    assert stx.rate_multiple_factor(200) == pytest.approx(0.065 / 0.085)
    assert stx.rate_multiple_factor(-10_000) == 0.0


def test_scenarios_match_hand_calculation():
    # revenue 1000, EBITDA 200 (20 %), 10x, net debt 500, debt 600, interest 30, 100 shares
    table = stx.scenario_valuations(1000, 200, 10, 500, 100, debt=600, interest_expense=-30).set_index("scenario")
    assert table.loc["Base case", "value_per_share"] == pytest.approx((2000 - 500) / 100)
    rec = table.loc["Recession"]  # revenue 850, margin 17 %, multiple 8x
    assert rec["ebitda"] == pytest.approx(850 * 0.17)
    assert rec["value_per_share"] == pytest.approx((850 * 0.17 * 8 - 500) / 100)
    rates = table.loc["Rates +200 bp"]
    assert rates["ev_ebitda"] == pytest.approx(10 * 0.065 / 0.085)
    assert rates["interest_coverage"] == pytest.approx(200 / (30 + 600 * 0.5 * 0.02))
    assert table.loc["Multiple de-rating", "change"] == pytest.approx((1400 - 500) / 1500 - 1)
    assert (table.iloc[1:]["change"] < 0).all()


def test_equity_floored_at_zero_and_input_validation():
    table = stx.scenario_valuations(100, 10, 5, 1000, 10)
    assert (table["equity_value"] == 0).all()
    with pytest.raises(ValueError):
        stx.scenario_valuations(100, -5, 10, 0, 10)


def test_historical_scenarios_beta_and_actual():
    index = pd.bdate_range("2019-06-03", "2023-06-30")
    prices = pd.DataFrame({"Close": np.full(len(index), 100.0)}, index=index)
    covid = (index >= "2020-02-19") & (index <= "2020-03-23")
    prices.loc[covid, "Close"] = np.linspace(100, 60, covid.sum())
    result = stx.historical_scenarios(1.5, prices, current_price=200).set_index("crisis")
    assert result.loc["COVID-19 crash", "beta_implied"] == pytest.approx(1.5 * -0.339)
    assert result.loc["COVID-19 crash", "actual_drawdown"] == pytest.approx(-0.40)
    assert result.loc["2022 rate shock", "actual_drawdown"] == pytest.approx(0.0)
    assert pd.isna(result.loc["Global financial crisis", "actual_drawdown"])  # not in history
    assert result.loc["Global financial crisis", "implied_price"] == pytest.approx(200 * (1 + 1.5 * -0.568))
    assert stx.historical_scenarios(None)["beta_implied"].isna().all()


@pytest.mark.parametrize("ticker", ["AAPL", "JPM"])
def test_scenario_section_renders(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Historical crises" in m.value for m in at.markdown)
