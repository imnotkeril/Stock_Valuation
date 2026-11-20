"""Analytics models checked against hand calculations on real Apple data, plus the forecasting engine."""

import numpy as np
import pandas as pd
import pytest

from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.models.financial_statements import FinancialStatementAnalyzer
from StockAnalysisSystem.src.models.forecasting.forecast_engine import PRICE_MODELS, future_index
from StockAnalysisSystem.src.models.forecasting.time_series import TimeSeriesForecaster
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.utils.financial_data import get_latest, get_series


def test_ratios_match_hand_calculation(aapl):
    fd = aapl["financial_data"]
    income, balance = fd["income_statement"], fd["balance_sheet"]
    ratios = FinancialRatioAnalyzer().calculate_ratios(fd)
    assert ratios["profitability"]["gross_margin"] == pytest.approx(
        get_latest(income, "gross_profit") / get_latest(income, "revenue"))
    assert ratios["profitability"]["gross_margin"] == pytest.approx(0.462, abs=1e-3)
    assert ratios["liquidity"]["current_ratio"] == pytest.approx(0.867, abs=1e-3)
    assert ratios["leverage"]["debt_to_equity"] == pytest.approx(
        get_latest(balance, "total_debt") / get_latest(balance, "total_equity"))


def test_altman_z_matches_formula(aapl):
    fd = aapl["financial_data"]
    income, balance = fd["income_statement"], fd["balance_sheet"]
    assets = get_latest(balance, "total_assets")
    expected = (1.2 * (get_latest(balance, "current_assets") - get_latest(balance, "current_liabilities")) / assets
                + 1.4 * get_latest(balance, "retained_earnings") / assets
                + 3.3 * get_latest(income, "ebit") / assets
                + 0.6 * fd["market_data"]["market_cap"] / get_latest(balance, "total_liabilities")
                + 1.0 * get_latest(income, "revenue") / assets)
    result = BankruptcyAnalyzer().calculate_altman_z_score(fd, "Technology")
    assert result["score"] == pytest.approx(expected)
    assert result["assessment"] == "safe"


def test_distress_models_not_applied_to_banks(make_company):
    bank = make_company("bank")
    models = BankruptcyAnalyzer().get_comprehensive_risk_assessment(bank["financial_data"], "Financials")["models"]
    assert all(not model["applicable"] for model in models.values())


def test_health_score_range(aapl):
    analyzer = FinancialStatementAnalyzer()
    fd = aapl["financial_data"]
    health = analyzer.calculate_financial_health_score(analyzer.analyze_income_statement(fd["income_statement"]),
                                                       analyzer.analyze_balance_sheet(fd["balance_sheet"]),
                                                       analyzer.analyze_cash_flow(fd["cash_flow"]))
    assert 0 <= health["overall_score"] <= 100
    assert set(health["components"]) == {"profitability", "liquidity", "solvency", "cash_flow"}


@pytest.fixture
def forecaster(data_loader):
    return TimeSeriesForecaster(data_loader)


@pytest.mark.parametrize("model", PRICE_MODELS)
def test_price_forecast_schema(forecaster, aapl, model):
    result = forecaster.forecast_price(aapl["price_data"], horizon=15, model=model, confidence=0.9)
    assert "error" not in result, result.get("error")
    forecast, lower, upper = result["forecast"], result["lower"], result["upper"]
    assert len(forecast) == 15 and not forecast.isna().any() and (forecast > 0).all()
    assert (lower <= forecast).all() and (forecast <= upper).all()
    assert forecast.index[0] > aapl["price_data"].index[-1]
    assert forecast.index.day_of_week.max() < 5  # business days only
    assert set(result["metrics"]) == {"mae", "rmse", "mape"}


def test_price_forecast_failures(forecaster, aapl):
    assert forecaster.forecast_price(aapl["price_data"].head(10))["error"]
    assert forecaster.forecast_price(aapl["price_data"], model="nope")["error"]
    assert forecaster.forecast_price(pd.DataFrame())["error"]


def test_ensemble_and_evaluation(forecaster, aapl):
    ensemble = forecaster.create_ensemble_forecast(aapl["price_data"], horizon=10)
    assert sum(ensemble["weights"].values()) == pytest.approx(1)
    assert set(ensemble["components"]) == set(ensemble["weights"])
    evaluation = forecaster.evaluate_models(aapl["price_data"], test_size=20)
    assert list(evaluation.columns) == ["model", "mae", "rmse", "mape"]
    assert evaluation["rmse"].is_monotonic_increasing


def test_probabilities_are_reproducible_and_consistent(forecaster, aapl):
    prices = aapl["price_data"]
    p1 = forecaster.probability_of_price_increase(prices, horizon=30, seed=1)
    assert p1 == forecaster.probability_of_price_increase(prices, horizon=30, seed=1)
    assert 0 <= p1 <= 1
    change = forecaster.probability_of_price_change(prices, horizon=30, target_change_pct=10)
    assert change["prob_above"] + change["prob_below"] <= 1
    percentiles = list(change["percentiles"].values())
    assert percentiles == sorted(percentiles)


@pytest.mark.parametrize("model", ["auto", "linear", "cagr", "ets"])
def test_financial_metric_forecast(forecaster, aapl, model):
    revenue = get_series(aapl["financial_data"]["income_statement"], "revenue", ascending=True)
    result = forecaster.forecast_financial_metric(revenue, periods=3, model=model)
    assert "error" not in result, result.get("error")
    assert len(result["forecast"]) == 3
    assert str(result["forecast"].index[0].date()) == "2025-09-30"  # Apple's fiscal year end


def test_short_series_arima_is_refused(forecaster, aapl):
    revenue = get_series(aapl["financial_data"]["income_statement"], "revenue", ascending=True)
    assert "8 periods" in forecaster.forecast_financial_metric(revenue, model="arima")["error"]


def test_indicator_forecasts():
    forecaster = TimeSeriesForecaster.__new__(TimeSeriesForecaster)
    index = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": 100 + np.cumsum(rng.normal(0, 1, 60)), "b": 50 + np.cumsum(rng.normal(0, 1, 60))},
                         index=index)
    single = forecaster.forecast_economic_indicator_arima(frame["a"], horizon=6)
    assert len(single["forecast"]) == 6 and single["forecast"].index[0] == pd.Timestamp("2020-01-31")
    var = forecaster.forecast_economic_indicators_var(frame, horizon=6)
    assert "error" not in var and set(var["forecasts"]) == {"a", "b"}


def test_future_index_frequencies():
    daily = pd.Series(1.0, index=pd.bdate_range("2024-01-01", periods=10))
    assert future_index(daily, 3)[0] == pd.Timestamp("2024-01-15")
    annual = pd.Series(1.0, index=pd.to_datetime(["2021-09-30", "2022-09-30", "2023-09-30"]))
    assert list(future_index(annual, 2).year) == [2024, 2025]
