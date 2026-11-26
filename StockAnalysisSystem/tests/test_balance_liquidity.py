"""A1-A4 / P1-P4 balance liquidity grouping (T25, AC22.4): hand-computed groups, verdict, missing rows, render."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.models.balance_liquidity import balance_liquidity, liquidity_table

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def _balance(rows: dict) -> pd.DataFrame:
    return pd.DataFrame({pd.Timestamp("2024-12-31"): rows})


ILLIQUID = {
    "Total Assets": 1000.0, "Current Assets": 400.0,
    "Cash Cash Equivalents And Short Term Investments": 100.0, "Receivables": 120.0, "Inventory": 80.0,
    "Current Liabilities": 300.0, "Payables And Accrued Expenses": 150.0, "Accounts Payable": 90.0,
    "Total Liabilities Net Minority Interest": 700.0, "Total Equity Gross Minority Interest": 300.0,
}
LIQUID = {
    "Total Assets": 1000.0, "Current Assets": 800.0,
    "Cash Cash Equivalents And Short Term Investments": 300.0, "Receivables": 250.0,
    "Current Liabilities": 300.0, "Payables": 200.0,
    "Total Liabilities Net Minority Interest": 450.0, "Total Equity Gross Minority Interest": 550.0,
}


def test_grouping_golden_values_not_liquid():
    r = balance_liquidity(_balance(ILLIQUID))
    # A1 = 100; A2 = 120; A3 = CA 400 - 100 - 120 = 180 (inventory 80 + other current 100); A4 = 1000 - 400 = 600
    # P1 = payables and accrued 150 (beats trade AP 90); P2 = CL 300 - 150 = 150; P3 = 700 - 300 = 400; P4 = 300
    assert r.groups == {"A1": 100.0, "A2": 120.0, "A3": 180.0, "A4": 600.0,
                        "P1": 150.0, "P2": 150.0, "P3": 400.0, "P4": 300.0}
    assert sum(r.groups[k] for k in ("A1", "A2", "A3", "A4")) == 1000.0
    assert sum(r.groups[k] for k in ("P1", "P2", "P3", "P4")) == 1000.0
    # 100 >= 150 no; 120 >= 150 no; 180 >= 400 no; 600 <= 300 no
    assert r.conditions == {"A1 >= P1": False, "A2 >= P2": False, "A3 >= P3": False, "A4 <= P4": False}
    assert r.absolutely_liquid is False
    assert r.current_liquidity == pytest.approx(-80.0)       # (100 + 120) - (150 + 150)
    assert r.prospective_liquidity == pytest.approx(-220.0)  # 180 - 400
    assert r.assumed_zero == []


def test_grouping_golden_values_absolutely_liquid():
    r = balance_liquidity(_balance(LIQUID))
    # A: 300, 250, 800 - 550 = 250, 200; P: 200, 300 - 200 = 100, 450 - 300 = 150, 550
    assert r.groups == {"A1": 300.0, "A2": 250.0, "A3": 250.0, "A4": 200.0,
                        "P1": 200.0, "P2": 100.0, "P3": 150.0, "P4": 550.0}
    assert all(r.conditions.values()) and r.absolutely_liquid is True
    assert r.current_liquidity == pytest.approx(250.0)       # 550 - 300
    assert r.prospective_liquidity == pytest.approx(100.0)   # 250 - 150
    table = liquidity_table(r)
    assert list(table["Condition"]) == ["A1 >= P1", "A2 >= P2", "A3 >= P3", "A4 <= P4"]
    assert list(table["Surplus / deficit"]) == [100.0, 150.0, 100.0, -350.0]


def test_missing_rows():
    rows = {k: v for k, v in LIQUID.items() if k not in ("Receivables", "Payables")}
    r = balance_liquidity(_balance(rows))
    # receivables and payables -> 0; A3 = 800 - 300 = 500; P2 = 300
    assert r.groups["A2"] == 0.0 and r.groups["A3"] == 500.0 and r.groups["P1"] == 0.0 and r.groups["P2"] == 300.0
    assert set(r.assumed_zero) == {"receivables", "payables"}
    no_split = {k: v for k, v in LIQUID.items() if k not in ("Current Assets",)}
    assert balance_liquidity(_balance(no_split)) is None
    assert balance_liquidity(pd.DataFrame()) is None
    assert balance_liquidity(None) is None


def test_equity_fallback_to_stockholders_equity():
    rows = {k: v for k, v in LIQUID.items() if k != "Total Equity Gross Minority Interest"}
    rows["Stockholders Equity"] = 540.0
    assert balance_liquidity(_balance(rows)).groups["P4"] == 540.0


def test_real_aapl_groups_sum_to_total_assets(aapl):
    r = balance_liquidity(aapl["financial_data"]["balance_sheet"])
    # FY2024: A1 65.171B, A2 66.243B, A3 152.987 - 65.171 - 66.243 = 21.573B, A4 364.980 - 152.987 = 211.993B
    assert r.groups["A1"] == pytest.approx(65.171e9) and r.groups["A3"] == pytest.approx(21.573e9)
    assert r.groups["A4"] == pytest.approx(211.993e9)
    # P1 payables 95.561B, P2 176.392 - 95.561 = 80.831B, P3 308.030 - 176.392 = 131.638B, P4 56.950B
    assert r.groups["P2"] == pytest.approx(80.831e9) and r.groups["P3"] == pytest.approx(131.638e9)
    assert sum(r.groups[k] for k in ("A1", "A2", "A3", "A4")) == pytest.approx(364.980e9)
    assert sum(r.groups[k] for k in ("P1", "P2", "P3", "P4")) == pytest.approx(364.980e9)
    assert r.absolutely_liquid is False


def _open_company(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    return at


def test_ratios_tab_shows_grouping_and_verdict():
    at = _open_company("AAPL")
    texts = [m.value for m in at.markdown]
    assert "**Balance-sheet liquidity (A1-A4 / P1-P4)**" in texts
    assert any("Not absolutely liquid" in t and "prospective liquidity" in t for t in texts)
    assert any("A1 >= P1, A2 >= P2, A3 >= P3 and A4 <= P4" in c.value for c in at.caption)


def test_ratios_tab_bank_note():
    at = _open_company("JPM")
    assert any("Does not apply to banks and insurers" in i.value for i in at.info)
