import pytest

from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer


def test_ratio_categories_and_margin_math(financial_data):
    ratios = FinancialRatioAnalyzer().calculate_ratios(financial_data)
    assert set(ratios) == {"valuation", "profitability", "liquidity", "leverage", "efficiency", "growth"}
    income = financial_data["income_statement"]
    expected = income.loc["Net Income"].iloc[0] / income.loc["Total Revenue"].iloc[0]
    assert ratios["profitability"]["net_margin"] == pytest.approx(expected, rel=1e-6)
    assert ratios["liquidity"]["current_ratio"] > 0


def test_altman_z_score(financial_data):
    r = BankruptcyAnalyzer().calculate_altman_z_score(financial_data, "Technology")
    assert "error" not in r
    assert isinstance(r["score"], float)
    assert "components" in r


def test_comprehensive_risk_assessment(financial_data):
    r = BankruptcyAnalyzer().get_comprehensive_risk_assessment(financial_data, "Technology")
    assert r["overall_assessment"] in {"safe", "grey", "distress"}
    assert r["models"]
