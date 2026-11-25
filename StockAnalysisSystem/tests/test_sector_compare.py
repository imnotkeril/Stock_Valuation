"""Multi-sector comparison view (offline)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.pages.comparison import sector_compare as sc

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def _series(start: float, end: float, n: int = 253) -> pd.Series:
    index = pd.bdate_range("2025-01-02", periods=n)
    return pd.Series(np.linspace(start, end, n), index=index)


def test_normalized_prices_start_at_100():
    prices = sc.normalized_prices({"A": _series(50, 75), "B": _series(200, 180)})
    assert prices.iloc[0].tolist() == [100.0, 100.0]
    assert prices["A"].iloc[-1] == pytest.approx(150.0)
    assert prices["B"].iloc[-1] == pytest.approx(90.0)


def test_performance_table_matches_hand_calculation():
    table = sc.performance_table({"Up": _series(100, 120), "Down": _series(100, 80)})
    assert table.loc["Up", "total_return"] == pytest.approx(0.20)
    assert table.loc["Down", "total_return"] == pytest.approx(-0.20)
    assert table.loc["Down", "max_drawdown"] == pytest.approx(-0.20)
    assert list(table.columns) == list(sc.PERFORMANCE_COLUMNS)


def test_fundamentals_are_medians_and_leaders_respect_direction():
    metrics = {
        "Technology": [{"pe_ratio": 30, "roe": 0.30}, {"pe_ratio": 40, "roe": 0.20}, {"pe_ratio": None, "roe": 0.25}],
        "Energy": [{"pe_ratio": 10, "roe": 0.15}, {"pe_ratio": -5, "roe": 0.10}],
        "Financials": [],
    }
    table = sc.fundamentals_table(metrics)
    assert table.loc["Technology", "pe_ratio"] == pytest.approx(35)
    assert table.loc["Technology", "roe"] == pytest.approx(0.25)
    assert table.loc["Energy", "companies"] == 2
    assert np.isnan(table.loc["Financials", "roe"])
    best = sc.best_sector_by_metric(table)
    assert best["roe"] == "Technology"           # higher is better
    assert best["pe_ratio"] == "Energy"          # lower positive is better (median 2.5 > 0)
    assert best["dividend_yield"] is None        # no data anywhere


def test_negative_multiples_are_not_leaders():
    table = sc.fundamentals_table({"A": [{"pe_ratio": -3}], "B": [{"pe_ratio": 12}]})
    assert sc.best_sector_by_metric(table)["pe_ratio"] == "B"


def test_compare_view_runs_in_the_app():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.SECTOR_ANALYSIS).run()
    at.segmented_control(key="sector_view").set_value("Compare sectors").run()
    assert not at.exception, [e.value for e in at.exception]
    at.multiselect(key=f"{sc.PREFIX}_sectors").set_value(["Technology", "Energy"]).run()
    at.number_input(key=f"{sc.PREFIX}_n").set_value(3).run()
    at.button(key=f"{sc.PREFIX}_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.tabs) >= 3
    assert at.session_state[sc.RESULTS_KEY]["sectors"] == ["Technology", "Energy"]

    at.selectbox(key=f"{sc.PREFIX}_chart_metric").set_value("roe").run()
    assert not at.exception


def test_compare_needs_two_sectors():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.SECTOR_ANALYSIS).run()
    at.segmented_control(key="sector_view").set_value("Compare sectors").run()
    at.multiselect(key=f"{sc.PREFIX}_sectors").set_value(["Energy"]).run()
    assert at.button(key=f"{sc.PREFIX}_run").disabled
    assert any("at least two sectors" in i.value for i in at.info)
