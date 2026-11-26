"""
Methodology fixes from the DCF/WACC and sector audits (2026-11-26), each against a hand calculation.

D1 DCF tab on unlevered FCF · leases out of debt · common equity for banks · SBC in the generic FCFF ·
APV fixed-debt shield (test_apv.py) · telecom cash margin (test_sector_models.py) · Nareit FFO ·
one tax rule and terminal growth capped at the risk-free rate.
"""

import pandas as pd
import pytest

from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    dcf_projection,
    dcf_starting_cash_flow,
    unlevered_fcf_history,
)
from StockAnalysisSystem.src.utils.financial_data import build_financial_data, debt_excluding_leases
from StockAnalysisSystem.src.valuation import capital
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation, SectorSpecificDCF
from StockAnalysisSystem.src.valuation.sector_specific.common import capital_structure
from StockAnalysisSystem.src.valuation.sector_specific.financial_sector import FinancialSectorValuation
from StockAnalysisSystem.src.valuation.sector_specific.real_estate import RealEstateValuation

PERIODS = [pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2022-12-31")]


def _frame(rows):
    """yfinance layout: rows = line items, columns = periods (most recent first)."""
    return pd.DataFrame({k: [float(x) if x is not None else float("nan") for x in v] for k, v in rows.items()},
                        index=PERIODS).T


# ---------------------------------------------------------------------------
# D1: DCF tab discounts unlevered FCF at the WACC
# ---------------------------------------------------------------------------
def test_dcf_tab_starts_from_unlevered_fcf():
    # FCF 100 / 90 / 80 (2024 / 2023 / 2022), interest 20 / 10 / 10, tax 25 %:
    # UFCF = 100 + 20 x 0.75 = 115; 90 + 7.5 = 97.5; 80 + 7.5 = 87.5.
    cash_flow = _frame({"Free Cash Flow": [100, 90, 80], "Stock Based Compensation": [6, 5, 4]})
    income = _frame({"Interest Expense": [20, 10, 10]})
    ufcf = unlevered_fcf_history(cash_flow, income, 0.25)
    assert list(ufcf) == pytest.approx([87.5, 97.5, 115.0])
    sbc = cash_flow.loc["Stock Based Compensation"].iloc[::-1]
    # Latest year: 115 - SBC 6 = 109. Three-year average: (87.5 + 97.5 + 115) / 3 = 100 minus the same
    # years' SBC average (4 + 5 + 6) / 3 = 5 -> 95 (D9: not the latest-year SBC 6).
    assert dcf_starting_cash_flow(ufcf, sbc, 1) == pytest.approx((109.0, 6.0))
    assert dcf_starting_cash_flow(ufcf, sbc, 3) == pytest.approx((95.0, 5.0))
    assert dcf_starting_cash_flow(ufcf, sbc, 3, deduct_sbc=False) == pytest.approx((100.0, 0.0))
    # 109 growing 10 % for 2 years at WACC 10 %: 119.9, 131.89 -> PV 109 + 109 = 218.
    # TV = 131.89 x 1.02 / 0.08 = 1681.5975, PV / 1.21 = 1389.75. EV 1607.75 - net debt 50 = 1557.75; / 10 sh.
    dcf = dcf_projection(109.0, 0.10, 0.10, 0.02, 2, 50.0, 10.0)
    assert dcf["enterprise_value"] == pytest.approx(1607.75)
    assert dcf["value_per_share"] == pytest.approx(155.775)


# ---------------------------------------------------------------------------
# Leases: Yahoo Total Debt includes lease liabilities; cash flows are after rent
# ---------------------------------------------------------------------------
def test_debt_excludes_lease_liabilities():
    # VZ-like: LT debt 139.5 + current 18.6 = 158.1; leases 23.5; Total Debt = 181.6.
    balance = _frame({"Total Debt": [181.6, 168.4, 150.0], "Long Term Debt": [139.5, 121.4, 120.0],
                      "Current Debt": [18.6, 22.6, 10.0], "Capital Lease Obligations": [23.5, 24.4, 20.0],
                      "Cash And Cash Equivalents": [2.0, 2.0, 2.0]})
    assert debt_excluding_leases(balance) == pytest.approx(158.1)
    assert capital.total_debt(balance) == pytest.approx(158.1)
    assert capital_structure(balance)["total_debt"] == pytest.approx(158.1)
    # Only the total and the lease row: 181.6 - 23.5 = 158.1.
    only_total = balance.loc[["Total Debt", "Capital Lease Obligations"]]
    assert debt_excluding_leases(only_total) == pytest.approx(158.1)
    # Only the "and lease" rows: (163.0 - 23.5 long-term lease) + (23.1 - 4.5 current lease) = 158.1.
    combined = _frame({"Long Term Debt And Capital Lease Obligation": [163.0, 150, 140],
                       "Long Term Capital Lease Obligation": [23.5, 20, 20],
                       "Current Debt And Capital Lease Obligation": [23.1, 20, 20],
                       "Current Capital Lease Obligation": [4.5, 4, 4]})
    assert debt_excluding_leases(combined) == pytest.approx(158.1)


def test_debt_uses_one_period():
    # XOM-like: latest Total Debt missing. Borrowings 27.9 + 9.3 = 37.2 from the latest year, not the
    # previous year's Total Debt 41.7 (which also includes 4.0 of leases).
    balance = _frame({"Total Debt": [None, 41.7, 40.0], "Long Term Debt": [27.9, 32.8, 30.0],
                      "Current Debt": [9.3, 5.0, 6.0], "Capital Lease Obligations": [6.3, 4.0, 4.0]})
    assert debt_excluding_leases(balance) == pytest.approx(37.2)
    assert debt_excluding_leases(pd.DataFrame()) is None


def test_engine_bridge_excludes_leases():
    # EV 1000 - borrowings 150 + cash 50 = 900 equity (leases 40 not deducted); 10 shares -> 90.
    balance = _frame({"Total Debt": [190, 190, 190], "Long Term Debt": [120, 120, 120], "Current Debt": [30, 30, 30],
                      "Capital Lease Obligations": [40, 40, 40], "Cash And Cash Equivalents": [50, 50, 50]})
    data = build_financial_data(balance_sheet=balance, company_info={"shares_outstanding": 10.0})
    bridge = FinancialSectorValuation()._equity_bridge(1000.0, data)
    assert bridge["total_debt"] == pytest.approx(150.0)
    assert bridge["value_per_share"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Banks: residual income, P/B and ROE on common equity and net income to common
# ---------------------------------------------------------------------------
def test_financial_profile_uses_common_equity():
    # Stockholders equity 362.4 includes 20 of preferred: common 342.4. NI 57.0, to common 55.7.
    # ROE on common: 2024 55.7 / avg(342.4, 324.7) = 55.7 / 333.55; 2023 56.9 / avg(324.7, 300) = 56.9 / 312.35;
    # 2022 50 / 300 (no prior year) -> mean of the three.
    income = _frame({"Net Income": [57.0, 58.5, 51.0], "Net Income Common Stockholders": [55.7, 56.9, 50.0]})
    balance = _frame({"Stockholders Equity": [362.4, 344.8, 320.0], "Common Stock Equity": [342.4, 324.7, 300.0],
                      "Preferred Stock": [20.0, 20.1, 20.0]})
    data = build_financial_data(income_statement=income, balance_sheet=balance,
                                company_info={"shares_outstanding": 2.0, "current_price": 300.0})
    model = FinancialSectorValuation()
    profile = model._financial_profile(data, {"cost_of_equity": 0.10})
    assert profile["book_value"] == pytest.approx(342.4)
    assert profile["net_income"] == pytest.approx(55.7)
    assert profile["book_value_per_share"] == pytest.approx(171.2)
    expected_roe = (55.7 / 333.55 + 56.9 / 312.35 + 50.0 / 300.0) / 3
    assert model._return_on_equity(data) == pytest.approx(expected_roe)


# ---------------------------------------------------------------------------
# Generic FCFF deducts SBC; tax rule shared with the WACC
# ---------------------------------------------------------------------------
def _sbc_company():
    income = _frame({"Interest Expense": [10, 10, 10], "Tax Rate For Calcs": [0.21, 0.21, 0.21],
                     "Pretax Income": [200, 200, 200], "Tax Provision": [30, 30, 30], "Net Income": [170, 170, 170]})
    cash_flow = _frame({"Free Cash Flow": [100, 100, 100], "Stock Based Compensation": [5, 5, 5]})
    return build_financial_data(income_statement=income, cash_flow=cash_flow)


def test_generic_fcff_deducts_sbc_once():
    # Tax: capital rule reads Tax Rate For Calcs 21 % (the old 3-year average would give 30 / 200 = 15 %).
    # FCFF = 100 + 10 x 0.79 - SBC 5 = 102.9; without SBC deduction 107.9.
    model = AdvancedDCFValuation()
    data = _sbc_company()
    assert model._effective_tax_rate(data) == pytest.approx(0.21)
    assert model._effective_tax_rate(data) == capital.effective_tax_rate(data["income_statement"])
    assert list(model._unlevered_fcf_series(data)) == pytest.approx([102.9] * 3)
    assert list(model._unlevered_fcf_series(data, deduct_sbc=False)) == pytest.approx([107.9] * 3)
    # Technology model: SBC deducted once, not twice (still 102.9).
    assert list(SectorSpecificDCF()._sbc_adjusted_fcf(data)) == pytest.approx([102.9] * 3)


def test_net_income_fallback_does_not_deduct_sbc():
    # No FCF / OCF rows: FCF falls back to net income 170, already after SBC -> 170 + 7.9 = 177.9.
    data = _sbc_company()
    data["cash_flow"] = data["cash_flow"].drop(index="Free Cash Flow")
    assert list(AdvancedDCFValuation()._unlevered_fcf_series(data)) == pytest.approx([177.9] * 3)


def test_terminal_growth_capped_at_risk_free_rate():
    assert capital.cap_terminal_growth(0.02, 0.045) == {"terminal_growth": 0.02, "note": None}
    capped = capital.cap_terminal_growth(0.05, 0.045)
    assert capped["terminal_growth"] == pytest.approx(0.045) and "capped" in capped["note"]


def test_engine_dcf_caps_terminal_growth(make_company):
    company = make_company("retail")
    model = AdvancedDCFValuation()
    rf = model.risk_free().rate
    result = model.discounted_cash_flow_valuation(company["ticker"], company["financial_data"],
                                                  discount_rate=0.12, terminal_growth=rf + 0.01)
    assert result["terminal_growth"] == pytest.approx(rf)
    assert "risk-free" in result["note"]


# ---------------------------------------------------------------------------
# REIT: Nareit FFO
# ---------------------------------------------------------------------------
def test_reit_ffo_follows_nareit(make_company):
    # Synthetic REIT, 2024: revenue 8 bn, net income 33 % = 2.64 bn, D&A 12 % = 0.96 bn.
    # Gains on property sales 0.20 bn and impairments 0.10 bn:
    # FFO = 2.64 - 0.20 + 0.10 + 0.96 = 3.50 bn (old NI + D&A rule: 3.60 bn).
    company = make_company("reit")
    data = company["financial_data"]
    latest = data["income_statement"].columns[0]
    income, cash_flow = data["income_statement"].copy(), data["cash_flow"].copy()
    income.loc["Gain On Sale Of Ppe"] = 0.0
    income.loc["Gain On Sale Of Ppe", latest] = 0.20e9
    cash_flow.loc["Asset Impairment Charge"] = 0.0
    cash_flow.loc["Asset Impairment Charge", latest] = 0.10e9
    adjusted = {**data, "income_statement": income, "cash_flow": cash_flow}
    model = RealEstateValuation()
    result = model.get_valuation(company["ticker"], adjusted, methods=["ffo_multiple"])
    ffo = result["valuation_methods"]["ffo_multiple"]
    assert ffo["ffo"] == pytest.approx(3.50e9)
    assert ffo["gains_on_sale"] == pytest.approx(0.20e9) and ffo["impairments"] == pytest.approx(0.10e9)
    # Without the rows: NI + D&A = 3.60 bn, with a note that gains and impairments were not reported.
    plain = model.get_valuation(company["ticker"], data, methods=["ffo_multiple"])["valuation_methods"]["ffo_multiple"]
    assert plain["ffo"] == pytest.approx(3.60e9)
    assert "not reported" in plain["note"]
