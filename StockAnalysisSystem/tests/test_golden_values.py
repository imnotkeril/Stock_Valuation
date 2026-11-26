"""
Golden values (review R13): every formula against a hand calculation written out below.

Numbers are chosen so the arithmetic stays exact on paper; a change in any formula breaks these.
"""

import pytest

from StockAnalysisSystem.src.pages.company_analysis.page_helpers import dcf_projection
from StockAnalysisSystem.src.utils.rates import RiskFreeRate
from StockAnalysisSystem.src.valuation import apv, capital
from StockAnalysisSystem.src.valuation.sector_specific.common import run_revenue_dcf, two_stage_ddm
from StockAnalysisSystem.src.valuation.sector_specific.financial_sector import FinancialSectorValuation


def test_dcf_tab_projection():
    # FCF 100 growing 10 % for 2 years: 110, 121. Discount 10 %: PV 110/1.1 = 100, 121/1.21 = 100.
    # TV = 121 x 1.02 / (0.10 - 0.02) = 1542.75; PV = 1542.75 / 1.21 = 1275.
    # EV = 200 + 1275 = 1475; equity = 1475 - 50 = 1425; per share (10 shares) = 142.5.
    dcf = dcf_projection(100.0, 0.10, 0.10, 0.02, 2, 50.0, 10.0)
    assert dcf["forecast_fcf"] == pytest.approx([110.0, 121.0])
    assert dcf["present_values"] == pytest.approx([100.0, 100.0])
    assert dcf["terminal_value"] == pytest.approx(1542.75)
    assert dcf["pv_terminal"] == pytest.approx(1275.0)
    assert dcf["enterprise_value"] == pytest.approx(1475.0)
    assert dcf["value_per_share"] == pytest.approx(142.5)


def test_sector_revenue_dcf():
    # Revenue 1000 -> 1100 -> 1210, FCF margin 10 %: FCF 110, 121 -> same PVs as above.
    # Terminal value share = 1275 / 1475.
    dcf = run_revenue_dcf(1000.0, [0.10, 0.10], [0.10, 0.10], 0.10, 0.02)
    assert dcf["pv_forecast"] == pytest.approx(200.0)
    assert dcf["pv_terminal"] == pytest.approx(1275.0)
    assert dcf["enterprise_value"] == pytest.approx(1475.0)
    assert dcf["terminal_value_share"] == pytest.approx(1275.0 / 1475.0)


def test_wacc():
    # rf 4 %, raw beta 1.0 -> Blume 0.67 + 0.33 = 1.0; Ke = 4 % + 1.0 x 4.4 % = 8.4 %.
    # Kd = 12 / 200 = 6 % (inside [4.5 %, 12 %]); tax 25 %; E = 800, D = 200 -> weights 0.8 / 0.2.
    # WACC = 0.8 x 8.4 % + 0.2 x 6 % x 0.75 = 6.72 % + 0.90 % = 7.62 %.
    assert capital.EQUITY_RISK_PREMIUM == 0.044
    result = capital.cost_of_capital(raw_beta=1.0, debt=200.0, interest_expense=12.0, tax_rate=0.25,
                                     equity_value=800.0, fallback_wacc=0.09, rf=RiskFreeRate(0.04, "test"))
    assert result["cost_of_equity"] == pytest.approx(0.084)
    assert result["pre_tax_cost_of_debt"] == pytest.approx(0.06)
    assert result["after_tax_cost_of_debt"] == pytest.approx(0.045)
    assert result["wacc"] == pytest.approx(0.0762)


def test_apv_unlevered_cost_of_equity():
    # Blume(1.0) = 1.0; Hamada: 1.0 / (1 + 0.75 x 100 / 300) = 0.8; Ku = 4 % + 0.8 x 4.4 % = 7.52 %.
    assert apv.unlevered_beta(1.0, 100.0, 300.0, 0.25) == pytest.approx(0.8)
    assert apv.unlevered_cost_of_equity(1.0, 100.0, 300.0, 0.25, risk_free=0.04) == pytest.approx(0.0752)


def test_apv_valuation():
    # UFCF 100 flat, 1 year, Ku 10 %: PV 100 / 1.1 = 90.909; TV = 100 / 0.10 = 1000, PV = 909.091.
    # Unlevered value = 1000. Shield = 20 % x 5 % x 100 = 1 per year: PV 1 / 1.05 = 0.952;
    # terminal shield 1 / 0.05 = 20, PV = 19.048 -> shields = 20. EV = 1020; equity = 1020 - 100 + 10 = 930.
    result = apv.apv_valuation(100.0, 0.0, 1, 0.0, 0.10, 100.0, 0.05, 0.20, cash=10.0, shares=10.0)
    assert result["unlevered_value"] == pytest.approx(1000.0)
    assert result["tax_shield_value"] == pytest.approx(20.0)
    assert result["enterprise_value"] == pytest.approx(1020.0)
    assert result["value_per_share"] == pytest.approx(93.0)


def test_residual_income():
    # Book 100, ROE 15 % (= sustainable), Ke 10 %, payout 40 %, g 3 %, 1 year.
    # Year 1: NI 15, excess (15 % - 10 %) x 100 = 5, PV 5 / 1.1 = 4.5455; closing book 100 + 15 x 0.6 = 109.
    # Terminal excess = 5 % x 109 = 5.45; TV = 5.45 / 0.07 = 77.857; PV = 77.857 / 1.1 = 70.779.
    # Equity = 100 + 4.5455 + 70.779 = 175.325. Stable payout = 1 - 3 % / 15 % = 0.8.
    model = FinancialSectorValuation()._excess_return_model(100.0, 0.15, 0.15, 0.10, 0.40, 0.03, 1)
    assert model["projections"][0]["closing_book_value"] == pytest.approx(109.0)
    assert model["present_value_excess_returns"] == pytest.approx(5 / 1.1)
    assert model["terminal_value"] == pytest.approx(5.45 / 0.07)
    assert model["equity_value"] == pytest.approx(100 + 5 / 1.1 + 5.45 / 0.07 / 1.1)
    assert model["stable_payout_ratio"] == pytest.approx(0.8)


def test_dividend_discount_matches_gordon():
    # DPS 2 growing 5 % forever at Ke 10 %: Gordon value = 2 x 1.05 / 0.05 = 42, whatever the horizon.
    for years in (1, 2, 5):
        assert two_stage_ddm(2.0, 0.10, 0.05, 0.05, years)["value_per_share"] == pytest.approx(42.0)


def test_justified_price_to_book():
    # (ROE - g) / (Ke - g) = (15 % - 3 %) / (10 % - 3 %) = 12 / 7; x book value per share 70 = 120.
    profile = {"sustainable_roe": 0.15, "cost_of_equity": 0.10, "terminal_growth": 0.03,
               "book_value_per_share": 70.0, "price_to_book": 1.5}
    result = FinancialSectorValuation()._justified_pb_method("TEST", {}, profile)
    assert result["justified_pb"] == pytest.approx(12 / 7)
    assert result["value_per_share"] == pytest.approx(120.0)
