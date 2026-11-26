"""DCF tab: growth fade to terminal growth and SBC deduction (review R9)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import dcf_projection

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_fade_hand_calculation():
    # Growth 10 % -> 2 % over 3 years: rates 10 %, 6 %, 2 %; FCF 110, 116.6, 118.932.
    dcf = dcf_projection(100.0, 0.10, 0.10, 0.02, 3, 0.0, 1.0, fade=True)
    assert dcf["growth_rates"] == pytest.approx([0.10, 0.06, 0.02])
    assert dcf["forecast_fcf"] == pytest.approx([110.0, 116.6, 118.932])
    assert dcf["terminal_value"] == pytest.approx(118.932 * 1.02 / 0.08)


def test_fade_lowers_value_when_growth_is_above_terminal():
    flat = dcf_projection(100.0, 0.15, 0.09, 0.025, 10, 0.0, 1.0)
    faded = dcf_projection(100.0, 0.15, 0.09, 0.025, 10, 0.0, 1.0, fade=True)
    assert faded["value_per_share"] < flat["value_per_share"]
    assert flat["growth_rates"] == [0.15] * 10


def test_dcf_tab_defaults_fade_and_sbc():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert at.checkbox(key="val_dcf_fade").value is True
    sbc = at.checkbox(key="val_dcf_sbc")
    assert sbc.value is True and "stock-based compensation" in sbc.label
    before = [m.value for m in at.metric if m.label == "Value per share"][0]
    sbc.uncheck().run()
    after = [m.value for m in at.metric if m.label == "Value per share"][0]
    assert before != after and not at.exception
