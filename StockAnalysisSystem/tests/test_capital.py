"""One cost of capital (ADR-007): hand-computed WACC, bands, rf metadata, generic == sector models."""

import pytest

from StockAnalysisSystem.src import config
from StockAnalysisSystem.src.models import stress_testing
from StockAnalysisSystem.src.utils.rates import RiskFreeRate
from StockAnalysisSystem.src.valuation import apv, capital
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
from StockAnalysisSystem.src.valuation.sector_specific.common import ValuationContext

RF = RiskFreeRate(0.04, "test", "2026-11-24")


def wacc(**overrides):
    args = dict(raw_beta=1.3, debt=100.0, interest_expense=6.0, tax_rate=0.21, equity_value=300.0,
                fallback_wacc=0.09, rf=RF)
    return capital.cost_of_capital(**{**args, **overrides})


def test_hand_computed_wacc():
    result = wacc()
    beta = 0.67 * 1.3 + 0.33
    ke = 0.04 + beta * capital.EQUITY_RISK_PREMIUM
    assert result["adjusted_beta"] == pytest.approx(beta)
    assert result["cost_of_equity"] == pytest.approx(ke)
    assert result["pre_tax_cost_of_debt"] == pytest.approx(0.06)
    assert result["wacc"] == pytest.approx(0.75 * ke + 0.25 * 0.06 * 0.79)
    assert result["equity_basis"] == "market" and result["debt_weight"] == pytest.approx(0.25)


@pytest.mark.parametrize("interest, expected", [(0.1, 0.045), (50.0, 0.12), (0.0, 0.06)])
def test_cost_of_debt_band_is_relative_to_rf(interest, expected):
    assert wacc(interest_expense=interest)["pre_tax_cost_of_debt"] == pytest.approx(expected)


def test_unknown_beta_and_no_equity():
    result = wacc(raw_beta=None, equity_value=None)
    assert result["beta"] == 1.0 and result["beta_assumed"]
    assert result["wacc"] == 0.09 and result["equity_basis"] == "sector default" and result["equity_weight"] is None


def test_wacc_band():
    assert wacc(raw_beta=0.0, debt=0.0)["wacc"] == capital.WACC_BOUNDS[0]
    assert wacc(equity_value=None, fallback_wacc=0.5)["wacc"] == capital.WACC_BOUNDS[1]
    assert wacc(raw_beta=6.0)["adjusted_beta"] == capital.BETA_BOUNDS[1]


def test_rate_metadata_reported():
    result = wacc()
    assert capital.rate_summary(result) == {"risk_free_rate": 0.04, "risk_free_rate_source": "test",
                                            "risk_free_rate_as_of": "2026-11-24",
                                            "equity_risk_premium": capital.EQUITY_RISK_PREMIUM}


def test_offline_default_rate_is_the_fallback(monkeypatch):
    monkeypatch.delenv(config.RISK_FREE_RATE_ENV, raising=False)
    assert capital.market_discount_rate() == pytest.approx(
        config.RISK_FREE_RATE_FALLBACK + capital.EQUITY_RISK_PREMIUM)


def test_env_override_flows_into_every_rate(monkeypatch, aapl):
    monkeypatch.setenv(config.RISK_FREE_RATE_ENV, "5")
    components = BaseValuation()._wacc_components(aapl["financial_data"], "Technology")
    assert components["risk_free_rate"] == pytest.approx(0.05) and components["risk_free_rate_source"] == "user override"
    assert capital.market_discount_rate() == pytest.approx(0.05 + capital.EQUITY_RISK_PREMIUM)
    assert apv.unlevered_cost_of_equity(1.0, 0.0, 100.0, 0.21) == pytest.approx(0.05 + capital.EQUITY_RISK_PREMIUM)


def _sector_context(company):
    data = company["financial_data"]
    sector = company["company_info"].get("sector")
    model = ValuationFactory().get_valuation_model(sector)
    return model, ValuationContext(ticker=company["ticker"], financial_data=data, subsector="x",
                                   sector=config.normalize_sector(sector))


@pytest.mark.parametrize("fixture", ["aapl", "msft"])
def test_generic_and_sector_wacc_are_equal(request, fixture):
    company = request.getfixturevalue(fixture)
    model, ctx = _sector_context(company)
    generic = BaseValuation()._wacc_components(company["financial_data"], ctx.sector)
    assert generic == ctx.cost_of_capital


def test_generic_and_sector_wacc_are_equal_for_every_synthetic_company(any_synthetic_company):
    _, ctx = _sector_context(any_synthetic_company)
    generic = BaseValuation()._wacc_components(any_synthetic_company["financial_data"], ctx.sector)
    assert generic["wacc"] == pytest.approx(ctx.cost_of_capital["wacc"])
    assert generic["cost_of_equity"] == pytest.approx(ctx.cost_of_capital["cost_of_equity"])


def test_sector_result_carries_rf(aapl):
    result = ValuationFactory().get_company_valuation("AAPL", sector="Technology",
                                                      financial_data=aapl["financial_data"])
    assumptions = result["assumptions"]
    assert assumptions["risk_free_rate"] == config.RISK_FREE_RATE_FALLBACK
    assert assumptions["risk_free_rate_source"] == "fallback constant"
    assert assumptions["equity_risk_premium"] == capital.EQUITY_RISK_PREMIUM


def test_stress_base_discount_is_market_rate():
    base = capital.market_discount_rate()
    growth = capital.LONG_RUN_GROWTH
    assert stress_testing.rate_multiple_factor(200) == pytest.approx((base - growth) / (base + 0.02 - growth))
