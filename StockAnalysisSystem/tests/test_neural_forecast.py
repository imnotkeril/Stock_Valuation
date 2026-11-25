"""Neural network (MLP) price forecast (offline, synthetic and recorded prices)."""

import numpy as np
import pandas as pd
import pytest

from StockAnalysisSystem.src.models.forecasting import forecast_engine as fe


def _prices(n=400, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    log = np.cumsum(drift + 0.01 * rng.standard_normal(n)) + np.log(100)
    return pd.Series(np.exp(log), index=pd.bdate_range("2023-01-02", periods=n), name="Close")


def test_fit_neural_shapes_and_band():
    values = np.log(_prices().values)
    mean, low, high = fe._fit_neural(values, 20, 0.95)
    assert len(mean) == len(low) == len(high) == 20
    assert np.all(low < mean) and np.all(mean < high)
    assert np.all(np.diff(high - low) > 0)  # band widens with the horizon
    # Deterministic for a fixed seed
    again, _, _ = fe._fit_neural(values, 20, 0.95)
    assert np.allclose(mean, again)


def test_neural_forecast_stays_plausible():
    close = _prices()
    values = np.log(close.values)
    mean, _, _ = fe._fit_neural(values, 30, 0.95)
    # 30 trading days of daily vol 1 %: recursion must not explode (clip at 3 sigma per step)
    assert abs(mean[-1] - values[-1]) < 30 * 3 * np.std(np.diff(values))


def test_needs_enough_history():
    with pytest.raises(ValueError):
        fe._fit_neural(np.log(_prices(30).values), 5, 0.95)


def test_registered_as_price_model():
    assert "neural" in fe.PRICE_MODELS and fe.PRICE_FITTERS["neural"] is fe._fit_neural


def test_forecast_price_api_with_neural(aapl):
    from StockAnalysisSystem.src.models.forecasting.time_series import TimeSeriesForecaster

    result = TimeSeriesForecaster().forecast_price(aapl["price_data"], horizon=15, model="neural")
    assert not result.get("error"), result.get("error")
    assert len(result["forecast"]) == 15 and result["model"] == "neural"
    assert (result["lower"] < result["forecast"]).all() and (result["forecast"] < result["upper"]).all()
    assert result["metrics"], "hold-out backtest metrics missing"
