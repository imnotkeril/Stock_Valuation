"""
Forecasting engine used by the Forecasting page.

Every public method returns a dict and never raises. The common forecast
schema is::

    {
        "model": str,
        "forecast": pd.Series,   # future dates
        "lower": pd.Series,      # lower confidence bound
        "upper": pd.Series,      # upper confidence bound
        "confidence": float,     # e.g. 0.95
        "metrics": {"mae", "rmse", "mape"} or {},   # hold-out backtest
        "history": pd.Series,    # data the model was fitted on
        "error": str,            # only on failure
    }

Price models work on log prices (forecasts stay positive and confidence
bands widen multiplicatively); statement metrics use short annual series.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("forecast_engine")

PRICE_MODELS = ("arima", "ets", "linear", "random_walk", "ml", "neural")
METRIC_MODELS = ("linear", "cagr", "ets", "arima")
INDICATOR_MODELS = ("arima", "ml")
#: Price history used for fitting (about two years of trading days)
MAX_PRICE_HISTORY = 504
LINEAR_WINDOW = 252
#: Cap on the annualised log drift used in Monte Carlo probabilities
MAX_ANNUAL_DRIFT = 0.30

try:  # pragma: no cover - optional dependency
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover
    Prophet = None
    PROPHET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _z(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + min(max(confidence, 0.5), 0.999) / 2))


def _close_series(price_data: Any) -> pd.Series:
    if isinstance(price_data, pd.Series):
        series = price_data
    elif isinstance(price_data, pd.DataFrame) and "Close" in price_data:
        series = price_data["Close"]
    else:
        return pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series.sort_index()


def _clean_series(series: Any) -> pd.Series:
    series = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    try:
        series.index = pd.to_datetime(series.index)
    except (TypeError, ValueError):
        pass
    return series.sort_index()


def future_index(history: pd.Series, horizon: int) -> pd.Index:
    """Dates following ``history``: business days for daily data, else the inferred frequency."""
    index = history.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return pd.RangeIndex(len(history), len(history) + horizon)
    median_gap = pd.Series(index).diff().median()
    if median_gap <= pd.Timedelta(days=4):
        freq = "B"
    elif median_gap <= pd.Timedelta(days=40):
        freq = "ME"
    elif median_gap <= pd.Timedelta(days=120):
        freq = pd.DateOffset(months=3)  # keeps the fiscal quarter-end alignment
    else:
        # Annual data: keep the company's fiscal year-end month (e.g. September for Apple)
        freq = pd.DateOffset(years=1)
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


def _metrics(actual: Sequence[float], predicted: Sequence[float]) -> Dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {}
    error = actual[mask] - predicted[mask]
    nonzero = actual[mask] != 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[mask][nonzero]))) if nonzero.any() else float("nan"),
    }


def _result(model: str, history: pd.Series, mean: Sequence[float], lower: Sequence[float],
            upper: Sequence[float], confidence: float, metrics: Optional[Dict[str, float]] = None,
            **extra: Any) -> Dict[str, Any]:
    index = future_index(history, len(mean))
    return {"model": model, "forecast": pd.Series(np.asarray(mean, dtype=float), index=index),
            "lower": pd.Series(np.asarray(lower, dtype=float), index=index),
            "upper": pd.Series(np.asarray(upper, dtype=float), index=index),
            "confidence": confidence, "metrics": metrics or {}, "history": history, **extra}


def _failure(model: str, error: Any, history: Optional[pd.Series] = None, **extra: Any) -> Dict[str, Any]:
    empty = pd.Series(dtype=float)
    return {"model": model, "forecast": empty, "lower": empty, "upper": empty, "confidence": None, "metrics": {},
            "history": history if history is not None else empty, "error": str(error), **extra}


# ---------------------------------------------------------------------------
# Core models on an arbitrary (log) series: return mean, std of the forecast
# ---------------------------------------------------------------------------
def _fit_arima(values: np.ndarray, horizon: int, confidence: float):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(values, order=(1, 1, 1)).fit()
        forecast = fit.get_forecast(horizon)
        mean = np.asarray(forecast.predicted_mean)
        low, high = np.asarray(forecast.conf_int(alpha=1 - confidence)).T
    return mean, low, high


def _fit_ets(values: np.ndarray, horizon: int, confidence: float):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(values, trend="add", damped_trend=True).fit()
    mean = np.asarray(fit.forecast(horizon))
    sigma = float(np.std(fit.resid[1:], ddof=1)) if len(values) > 3 else float(np.std(values))
    spread = _z(confidence) * sigma * np.sqrt(np.arange(1, horizon + 1))
    return mean, mean - spread, mean + spread


def _fit_linear(values: np.ndarray, horizon: int, confidence: float):
    n = len(values)
    x = np.arange(n, dtype=float)
    slope, intercept, *_ = stats.linregress(x, values)
    residual = values - (intercept + slope * x)
    sigma = float(np.std(residual, ddof=2)) if n > 2 else 0.0
    future = np.arange(n, n + horizon, dtype=float)
    mean = intercept + slope * future
    leverage = 1 + 1 / n + (future - x.mean()) ** 2 / max(((x - x.mean()) ** 2).sum(), 1e-12)
    spread = _z(confidence) * sigma * np.sqrt(leverage)
    return mean, mean - spread, mean + spread


def _fit_random_walk(values: np.ndarray, horizon: int, confidence: float):
    steps = np.diff(values)
    drift = float(np.mean(steps)) if len(steps) else 0.0
    sigma = float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0
    h = np.arange(1, horizon + 1)
    mean = values[-1] + drift * h
    spread = _z(confidence) * sigma * np.sqrt(h)
    return mean, mean - spread, mean + spread


def _fit_ml(values: np.ndarray, horizon: int, confidence: float, lags: int = 5):
    """Ridge regression on lagged changes, forecast recursively."""
    from sklearn.linear_model import Ridge

    steps = np.diff(values)
    if len(steps) <= lags + 5:
        raise ValueError("Not enough history for the ML model")
    X = np.column_stack([steps[i:len(steps) - lags + i] for i in range(lags)])
    y = steps[lags:]
    model = Ridge(alpha=1.0).fit(X, y)
    sigma = float(np.std(y - model.predict(X), ddof=1))
    window, level, mean = list(steps[-lags:]), values[-1], []
    for _ in range(horizon):
        step = float(model.predict(np.array(window[-lags:]).reshape(1, -1))[0])
        level += step
        mean.append(level)
        window.append(step)
    mean = np.array(mean)
    spread = _z(confidence) * sigma * np.sqrt(np.arange(1, horizon + 1))
    return mean, mean - spread, mean + spread


def _fit_neural(values: np.ndarray, horizon: int, confidence: float, lags: int = 10, seed: int = 0):
    """
    Multi-layer perceptron (two hidden layers) on standardised lagged changes, forecast
    recursively; the band comes from the in-sample residual spread growing with sqrt(h).
    """
    from sklearn.neural_network import MLPRegressor

    steps = np.diff(values)
    if len(steps) <= lags + 30:
        raise ValueError("Not enough history for the neural network model")
    scale = float(np.std(steps)) or 1.0
    scaled = steps / scale
    X = np.column_stack([scaled[i:len(scaled) - lags + i] for i in range(lags)])
    y = scaled[lags:]
    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                         random_state=seed).fit(X, y)
    sigma = float(np.std(y - model.predict(X), ddof=1)) * scale
    window, level, mean = list(scaled[-lags:]), values[-1], []
    for _ in range(horizon):
        step = float(model.predict(np.array(window[-lags:]).reshape(1, -1))[0])
        step = float(np.clip(step, -3.0, 3.0))  # keep the recursion from running away
        level += step * scale
        mean.append(level)
        window.append(step)
    mean = np.array(mean)
    spread = _z(confidence) * sigma * np.sqrt(np.arange(1, horizon + 1))
    return mean, mean - spread, mean + spread


def _fit_cagr(values: np.ndarray, horizon: int, confidence: float):
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        raise ValueError("CAGR requires positive first and last values")
    growth = (values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1
    yearly = np.diff(np.log(values[values > 0])) if (values > 0).all() else np.array([growth])
    sigma = float(np.std(yearly, ddof=1)) if len(yearly) > 1 else abs(growth) / 2
    h = np.arange(1, horizon + 1)
    mean = values[-1] * (1 + growth) ** h
    factor = np.exp(_z(confidence) * sigma * np.sqrt(h))
    return mean, mean / factor, mean * factor


def _fit_prophet(series: pd.Series, horizon: int, confidence: float):  # pragma: no cover - optional
    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(interval_width=confidence, daily_seasonality=False)
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq="B", include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].values, forecast["yhat_lower"].values, forecast["yhat_upper"].values


PRICE_FITTERS: Dict[str, Callable] = {"arima": _fit_arima, "ets": _fit_ets, "linear": _fit_linear,
                                      "random_walk": _fit_random_walk, "ml": _fit_ml,
                                      "neural": _fit_neural}
METRIC_FITTERS: Dict[str, Callable] = {"linear": _fit_linear, "cagr": _fit_cagr, "ets": _fit_ets,
                                       "arima": _fit_arima}


class ForecastEngineMixin:
    """Forecasting API used by the UI (see module docstring for the result schema)."""

    # ----- prices ---------------------------------------------------------------
    def _price_values(self, close: pd.Series, model: str) -> np.ndarray:
        values = np.log(close.values.astype(float))
        if model == "linear":
            values = values[-LINEAR_WINDOW:]
        return values

    def _run_price_model(self, close: pd.Series, horizon: int, model: str, confidence: float):
        if model == "prophet":
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet is not installed")
            mean, low, high = _fit_prophet(np.log(close), horizon, confidence)
        else:
            mean, low, high = PRICE_FITTERS[model](self._price_values(close, model), horizon, confidence)
        return np.exp(mean), np.exp(low), np.exp(high)

    def forecast_price(self, price_data: Any, horizon: int = 30, model: str = "arima",
                       confidence: float = 0.95) -> Dict[str, Any]:
        """Price forecast with confidence band and a hold-out backtest (see module docstring)."""
        close = _close_series(price_data).iloc[-MAX_PRICE_HISTORY:]
        model = str(model).lower()
        if model not in PRICE_FITTERS and model != "prophet":
            return _failure(model, f"Unknown model '{model}'", close)
        if len(close) < 30:
            return _failure(model, "At least 30 price observations are required", close)
        horizon = max(int(horizon), 1)
        try:
            mean, low, high = self._run_price_model(close, horizon, model, confidence)
            metrics = self._backtest(close, model, min(horizon, max(len(close) // 5, 5)))
            return _result(model, close, mean, low, high, confidence, metrics)
        except Exception as exc:
            logger.info(f"{model} price forecast failed: {exc}")
            return _failure(model, exc, close)

    def _backtest(self, close: pd.Series, model: str, test_size: int) -> Dict[str, float]:
        if len(close) <= test_size + 30:
            return {}
        try:
            train, test = close.iloc[:-test_size], close.iloc[-test_size:]
            mean, _, _ = self._run_price_model(train, test_size, model, 0.95)
            return _metrics(test.values, mean)
        except Exception as exc:
            logger.info(f"Backtest of {model} failed: {exc}")
            return {}

    def evaluate_models(self, price_data: Any, models: Optional[Sequence[str]] = None,
                        test_size: int = 30) -> pd.DataFrame:
        """Hold-out accuracy per model: columns model, mae, rmse, mape (sorted by rmse)."""
        close = _close_series(price_data).iloc[-MAX_PRICE_HISTORY:]
        rows = []
        for model in models or PRICE_MODELS:
            metrics = self._backtest(close, str(model).lower(), max(int(test_size), 5))
            if metrics:
                rows.append({"model": model, **metrics})
        frame = pd.DataFrame(rows, columns=["model", "mae", "rmse", "mape"])
        return frame.sort_values("rmse").reset_index(drop=True)

    @staticmethod
    def _blend(results: Dict[str, Dict[str, Any]], history: pd.Series, confidence: float,
               label: str = "ensemble") -> Dict[str, Any]:
        valid = {k: r for k, r in results.items() if not r.get("error") and len(r.get("forecast", [])) > 0}
        if not valid:
            return _failure(label, "No component model produced a forecast", history)
        inverse = {k: 1 / r["metrics"]["rmse"] for k, r in valid.items()
                   if r.get("metrics", {}).get("rmse") and r["metrics"]["rmse"] > 0}
        if len(inverse) != len(valid):
            inverse = {k: 1.0 for k in valid}
        total = sum(inverse.values())
        weights = {k: w / total for k, w in inverse.items()}
        index = next(iter(valid.values()))["forecast"].index
        combine = lambda key: sum(valid[k][key].values * w for k, w in weights.items())  # noqa: E731
        result = {"model": label, "forecast": pd.Series(combine("forecast"), index=index),
                  "lower": pd.Series(combine("lower"), index=index), "upper": pd.Series(combine("upper"), index=index),
                  "confidence": confidence, "history": history, "weights": weights,
                  "components": {k: r["forecast"] for k, r in valid.items()},
                  "metrics": {}}
        rmse = [valid[k]["metrics"].get("rmse") for k in weights if valid[k].get("metrics")]
        if rmse and all(rmse):
            result["metrics"] = {"rmse": float(sum(valid[k]["metrics"]["rmse"] * w for k, w in weights.items()))}
        return result

    def create_ensemble_forecast(self, price_data: Any, horizon: int = 30, models: Optional[Sequence[str]] = None,
                                 confidence: float = 0.95) -> Dict[str, Any]:
        """Inverse-RMSE weighted blend of price models (adds ``components`` and ``weights``)."""
        close = _close_series(price_data).iloc[-MAX_PRICE_HISTORY:]
        results = {m: self.forecast_price(close, horizon, m, confidence) for m in (models or PRICE_MODELS[:4])}
        return self._blend(results, close, confidence)

    # ----- probabilities (geometric Brownian motion Monte Carlo) ---------------
    @staticmethod
    def _simulate_terminal_prices(close: pd.Series, horizon: int, n_simulations: int, seed: int) -> np.ndarray:
        returns = np.diff(np.log(close.values.astype(float)))
        # A sample mean from a short window is a very noisy drift estimate; cap it at +/-30 % a year
        # so a recent crash or rally is not extrapolated without limit.
        mu = float(np.clip(np.mean(returns), -MAX_ANNUAL_DRIFT / 252, MAX_ANNUAL_DRIFT / 252))
        sigma = float(np.std(returns, ddof=1))
        rng = np.random.default_rng(seed)
        shocks = rng.normal(mu, sigma, size=(int(n_simulations), int(horizon))).sum(axis=1)
        return float(close.iloc[-1]) * np.exp(shocks)

    def probability_of_price_increase(self, price_data: Any, horizon: int = 30, threshold_pct: float = 0.0,
                                      n_simulations: int = 5000, seed: int = 42) -> Optional[float]:
        """P(price at the horizon > current price x (1 + threshold)); None without enough data."""
        close = _close_series(price_data).iloc[-MAX_PRICE_HISTORY:]
        if len(close) < 30:
            return None
        terminal = self._simulate_terminal_prices(close, horizon, n_simulations, seed)
        return float(np.mean(terminal > close.iloc[-1] * (1 + threshold_pct / 100)))

    def probability_of_price_change(self, price_data: Any, horizon: int = 30, target_change_pct: float = 10.0,
                                    n_simulations: int = 5000, seed: int = 42) -> Dict[str, Any]:
        """Probabilities of moving beyond +/- target, expected price and price percentiles."""
        close = _close_series(price_data).iloc[-MAX_PRICE_HISTORY:]
        if len(close) < 30:
            return {"error": "At least 30 price observations are required"}
        terminal = self._simulate_terminal_prices(close, horizon, n_simulations, seed)
        price = float(close.iloc[-1])
        target = abs(float(target_change_pct)) / 100
        return {"prob_above": float(np.mean(terminal >= price * (1 + target))),
                "prob_below": float(np.mean(terminal <= price * (1 - target))),
                "expected_price": float(np.mean(terminal)),
                "percentiles": {p: float(np.percentile(terminal, p)) for p in (5, 25, 50, 75, 95)},
                "current_price": price, "horizon": horizon, "target_change_pct": float(target_change_pct)}

    # ----- statement metrics ----------------------------------------------------
    def forecast_financial_metric(self, series: Any, periods: int = 3, model: str = "auto",
                                  confidence: float = 0.9) -> Dict[str, Any]:
        """Forecast a chronological statement metric (e.g. annual revenue)."""
        history = _clean_series(series)
        model = str(model).lower()
        if len(history) < 3:
            return _failure(model, "At least 3 periods are required", history)
        if model == "auto":
            model = "cagr" if (history > 0).all() else "linear"
        if model not in METRIC_FITTERS:
            return _failure(model, f"Unknown model '{model}'", history)
        if model == "arima" and len(history) < 8:
            return _failure(model, "ARIMA needs at least 8 periods; use linear, CAGR or ETS", history)
        try:
            mean, low, high = METRIC_FITTERS[model](history.values.astype(float), int(periods), confidence)
            metrics = {}
            if len(history) >= 4:
                try:
                    fit, _, _ = METRIC_FITTERS[model](history.values[:-1].astype(float), 1, confidence)
                    metrics = _metrics(history.values[-1:], fit)
                except Exception:
                    metrics = {}
            return _result(model, history, mean, low, high, confidence, metrics)
        except Exception as exc:
            return _failure(model, exc, history)

    def create_ensemble_financial_forecast(self, series: Any, periods: int = 3,
                                           models: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        history = _clean_series(series)
        results = {m: self.forecast_financial_metric(history, periods, m) for m in (models or ("linear", "cagr", "ets"))}
        return self._blend(results, history, 0.9)

    # ----- macro indicators -----------------------------------------------------
    def _indicator_forecast(self, series: Any, horizon: int, model: str, confidence: float) -> Dict[str, Any]:
        history = _clean_series(series)
        if len(history) < 8:
            return _failure(model, "At least 8 observations are required", history)
        try:
            fitter = _fit_arima if model == "arima" else _fit_ml
            mean, low, high = fitter(history.values.astype(float), int(horizon), confidence)
            return _result(model, history, mean, low, high, confidence)
        except Exception as exc:
            return _failure(model, exc, history)

    def forecast_economic_indicator_arima(self, series: Any, horizon: int = 12,
                                          confidence: float = 0.95) -> Dict[str, Any]:
        return self._indicator_forecast(series, horizon, "arima", confidence)

    def forecast_economic_indicator_ml(self, series: Any, horizon: int = 12,
                                       confidence: float = 0.95) -> Dict[str, Any]:
        return self._indicator_forecast(series, horizon, "ml", confidence)

    def create_ensemble_indicator_forecast(self, series: Any, horizon: int = 12,
                                           models: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        history = _clean_series(series)
        results = {m: self._indicator_forecast(history, horizon, m, 0.95) for m in (models or INDICATOR_MODELS)}
        return self._blend(results, history, 0.95)

    def forecast_economic_indicators_var(self, data: pd.DataFrame, horizon: int = 12) -> Dict[str, Any]:
        """Vector autoregression on aligned indicators (differenced for stationarity)."""
        from statsmodels.tsa.api import VAR

        try:
            frame = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").dropna()
            if frame.shape[1] < 2 or len(frame) < 12:
                raise ValueError("VAR needs at least two aligned series with 12 observations")
            diffed = frame.diff().dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = VAR(diffed.values).fit(maxlags=min(4, max(len(diffed) // 5, 1)), ic="aic")
            lag = max(fit.k_ar, 1)
            mean, low, high = fit.forecast_interval(diffed.values[-lag:], steps=int(horizon), alpha=0.05)
            last = frame.iloc[-1].values
            index = future_index(frame.iloc[:, 0], int(horizon))
            to_level = lambda steps: pd.DataFrame(np.cumsum(steps, axis=0) + last, index=index,  # noqa: E731
                                                  columns=frame.columns)
            levels, lower, upper = to_level(mean), to_level(low), to_level(high)
            return {"model": "var", "forecasts": {c: levels[c] for c in frame.columns},
                    "lower": {c: lower[c] for c in frame.columns}, "upper": {c: upper[c] for c in frame.columns},
                    "lag_order": int(fit.k_ar)}
        except Exception as exc:
            return {"model": "var", "forecasts": {}, "lower": {}, "upper": {}, "error": str(exc)}


__all__: List[str] = ["ForecastEngineMixin", "PRICE_MODELS", "METRIC_MODELS", "PROPHET_AVAILABLE", "future_index"]
