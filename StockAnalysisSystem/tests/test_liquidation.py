"""Liquidation value (T22, AC22.1): golden values computed by hand, floor at zero, page render."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.valuation import liquidation as liq

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def _balance(rows: dict) -> pd.DataFrame:
    return pd.DataFrame({pd.Timestamp("2024-12-31"): rows})


BASE = {
    "Total Assets": 1000.0,
    "Cash Cash Equivalents And Short Term Investments": 100.0,
    "Receivables": 200.0,
    "Accounts Receivable": 120.0,          # trade only: the total "Receivables" row wins
    "Inventory": 150.0,
    "Investments And Advances": 50.0,
    "Net PPE": 300.0,
    "Goodwill And Other Intangible Assets": 100.0,
    "Total Liabilities Net Minority Interest": 400.0,
}


def test_asset_classes_residual_other_assets():
    books = liq.asset_classes(_balance(BASE))
    # other = 1000 - (100 + 200 + 150 + 50 + 300 + 100) = 100
    assert books == {"cash": 100.0, "receivables": 200.0, "inventory": 150.0, "investments": 50.0,
                     "ppe": 300.0, "intangibles": 100.0, "other": 100.0}
    assert sum(books.values()) == pytest.approx(1000.0)


def test_orderly_and_forced_golden_values():
    result = liq.liquidation_scenarios(_balance(BASE), shares=10)
    orderly, forced = result["orderly"], result["forced"]
    # Orderly: 100*1.00 + 200*0.85 + 150*0.60 + 50*0.90 + 300*0.50 + 100*0 + 100*0.30
    #        = 100 + 170 + 90 + 45 + 150 + 0 + 30 = 585; costs 5 % = 29.25
    #   equity = 585 - 29.25 - 400 = 155.75 -> 15.575 per share
    assert orderly.gross_recovery == pytest.approx(585.0)
    assert orderly.costs == pytest.approx(29.25)
    assert orderly.equity_value == pytest.approx(155.75)
    assert orderly.per_share == pytest.approx(15.575)
    assert orderly.recovered["intangibles"] == 0.0
    # Forced: 100 + 200*0.70 + 150*0.35 + 50*0.75 + 300*0.25 + 0 + 100*0.10
    #       = 100 + 140 + 52.5 + 37.5 + 75 + 0 + 10 = 415; costs 10 % = 41.5
    #   equity = 415 - 41.5 - 400 = -26.5 -> floored at 0
    assert forced.gross_recovery == pytest.approx(415.0)
    assert forced.costs == pytest.approx(41.5)
    assert forced.equity_value == 0.0
    assert forced.per_share == 0.0


def test_custom_rates_and_no_costs():
    books = liq.asset_classes(_balance(BASE))
    # all 100 %, no costs: 1000 - 400 = 600 -> 60 per share; intangibles at 100 % only because rates say so
    full = liq.liquidation_value(books, 400.0, 10, {k: 1.0 for k in books}, cost_pct=0.0)
    assert full.per_share == pytest.approx(60.0)
    # default orderly rates without costs: 585 - 400 = 185 -> 18.5
    assert liq.liquidation_value(books, 400.0, 10, liq.ORDERLY_RATES).per_share == pytest.approx(18.5)


def test_goodwill_fallback_and_missing_rows():
    rows = {k: v for k, v in BASE.items() if k not in ("Goodwill And Other Intangible Assets", "Inventory")}
    rows.update({"Goodwill": 60.0, "Other Intangible Assets": 40.0})
    books = liq.asset_classes(_balance(rows))
    # intangibles 60 + 40 = 100; inventory missing -> 0, so other = 1000 - 750 = 250
    assert books["intangibles"] == 100.0 and books["inventory"] == 0.0 and books["other"] == 250.0


def test_missing_inputs():
    assert liq.asset_classes(pd.DataFrame()) is None
    no_liab = {k: v for k, v in BASE.items() if not k.startswith("Total Liabilities")}
    assert liq.liquidation_scenarios(_balance(no_liab), 10) is None
    assert liq.liquidation_scenarios(_balance(BASE), None)["orderly"].per_share is None


def _open_company(ticker):
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value(ticker).run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_liquidation_section_renders(aapl):
    at = _open_company("AAPL")
    labels = [m.label for m in at.metric]
    assert "Orderly liquidation / share" in labels and "Forced liquidation / share" in labels
    assert "Liquidation value / share" not in labels  # the old single-haircut metric is gone
    assert any("a floor, not in the weighted fair value" in c.value for c in at.caption)
    from StockAnalysisSystem.src.pages.company_analysis.page_helpers import fmt_price

    data = aapl["financial_data"]
    expected = liq.liquidation_scenarios(data["balance_sheet"], data["market_data"]["shares_outstanding"])
    shown = {m.label: m.value for m in at.metric}
    assert shown["Orderly liquidation / share"] == fmt_price(expected["orderly"].per_share)
    assert shown["Forced liquidation / share"] == fmt_price(expected["forced"].per_share)
    assert not [e.value for e in at.error]


def test_liquidation_bank_note():
    at = _open_company("JPM")
    assert any("Not meaningful for banks and insurers" in i.value for i in at.info)
