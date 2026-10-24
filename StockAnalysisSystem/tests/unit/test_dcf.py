import numpy as np
import pandas as pd
import pytest

from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation


@pytest.fixture
def dcf(fake_loader):
    return AdvancedDCFValuation(fake_loader)


def test_growth_rate_is_clipped_cagr(dcf):
    rising = pd.Series([200.0, 100.0], index=pd.to_datetime(["2025-09-30", "2024-09-30"]))
    assert dcf._estimate_initial_growth_rate(rising) == pytest.approx(0.25)  # +100% clipped to max
    flat = pd.Series([100.0, 100.0, 100.0])
    assert dcf._estimate_initial_growth_rate(flat) == pytest.approx(0.02)  # 0% clipped to min
    assert dcf._estimate_initial_growth_rate(pd.Series([-5.0, 10.0])) == pytest.approx(0.05)  # fallback


def test_multi_stage_dcf(dcf, financial_data):
    r = dcf.multi_stage_dcf_valuation("AAPL", financial_data, "Technology")
    assert "error" not in r
    assert r["value_per_share"] > 0
    assert r["conservative_value"] < r["value_per_share"]
    assert len(r["forecast_fcf"]) == 10
    parts = r["present_value_high_growth"] + r["present_value_transition"] + r["present_value_terminal"]
    assert parts == pytest.approx(r["enterprise_value"])


def test_forecast_driven_dcf(dcf, financial_data):
    r = dcf.forecast_driven_dcf("AAPL", financial_data, "Technology")
    assert "error" not in r
    assert r["value_per_share"] > 0
    assert len(r["forecast_fcf"]) == r["forecast_years"]
    assert list(r["forecast_financials"].columns) == ["revenue_growth", "revenue", "fcf_margin", "fcf"]


def test_forecast_growth_fades_to_terminal(dcf, financial_data):
    params = dcf._get_dcf_parameters("Technology")
    f = dcf._forecast_financial_statements(financial_data["income_statement"], financial_data["balance_sheet"],
                                           financial_data["cash_flow"], 7, "Technology")
    growth = f["summary"]["revenue_growth"]
    assert growth.iloc[-1] == pytest.approx(params["terminal_growth_rate"])


def test_missing_statements_return_error_not_exception(dcf):
    r = dcf.multi_stage_dcf_valuation("AAPL", {"income_statement": None, "balance_sheet": None, "cash_flow": None})
    assert r["value_per_share"] is None and "error" in r


def test_monte_carlo_and_scenarios(dcf, financial_data):
    mc = dcf.monte_carlo_dcf("AAPL", financial_data, "Technology")
    assert mc["simulations"] > 0 and np.isfinite(mc["median_value_per_share"])
    sc = dcf.scenario_analysis_dcf("AAPL", financial_data, "Technology")
    assert sc["conservative_value"] is not None
