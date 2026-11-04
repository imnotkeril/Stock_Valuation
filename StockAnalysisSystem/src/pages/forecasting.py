"""
Forecasting page.

Entry point: :func:`run_forecasting_page` (no arguments).

Tabs:

* **Price Forecast** - ``TimeSeriesForecaster.forecast_price`` for each
  selected model (ARIMA, exponential smoothing, linear trend, random walk,
  machine learning and Prophet when installed), the weighted ensemble
  (``create_ensemble_forecast``), a backtest (``evaluate_models``) and
  Monte-Carlo probabilities (``probability_of_price_increase`` /
  ``probability_of_price_change``).
* **Financial Metrics** - statement line items and margins forecast with
  ``forecast_financial_metric`` / ``create_ensemble_financial_forecast``.
* **Scenarios** - bear/base/bull projections of revenue, earnings and the
  implied share price, with a probability-weighted target.
* **Economic Indicators** - FRED macro series (needs network access and
  ``pandas-datareader``) forecast with ARIMA/ML/ensemble and VAR.

Results are stored in ``st.session_state`` when the run buttons are pressed
and rendered from there, so they survive reruns triggered by other widgets.
Model failures are logged and reported as friendly messages.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import COLORS
from StockAnalysisSystem.src.navigation import get_active_ticker
from StockAnalysisSystem.src.pages import forecast_charts as charts
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.export import to_csv_bytes
from StockAnalysisSystem.src.utils.financial_data import (
    build_financial_data,
    cagr,
    format_large_number,
    get_latest,
    get_series,
    safe_divide,
    to_float,
)

try:
    from StockAnalysisSystem.src.models.forecasting.time_series import TimeSeriesForecaster
except Exception:  # pragma: no cover - the page degrades gracefully
    logging.getLogger(__name__).exception("TimeSeriesForecaster could not be imported")
    TimeSeriesForecaster = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

KEY_PREFIX = "fc"
PRICE_RESULT_KEY = "fc_price_result"
METRIC_RESULT_KEY = "fc_metric_result"
MACRO_RESULT_KEY = "fc_macro_result"
DEFAULT_TICKER = "AAPL"

PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None

#: Price models offered (key -> label); Prophet only when installed
PRICE_MODELS: Dict[str, str] = {
    "arima": "ARIMA",
    "ets": "Exponential Smoothing",
    "linear": "Linear Trend",
    "random_walk": "Random Walk (drift)",
    "ml": "Machine Learning",
}
if PROPHET_AVAILABLE:
    PRICE_MODELS["prophet"] = "Prophet"

METRIC_MODELS: Dict[str, str] = {
    "linear": "Linear Trend",
    "cagr": "CAGR",
    "ets": "Exponential Smoothing",
    "arima": "ARIMA",
}

MACRO_MODELS: Dict[str, str] = {"arima": "ARIMA", "ml": "Machine Learning", "ensemble": "Ensemble"}

MODEL_LABELS: Dict[str, str] = {**PRICE_MODELS, **METRIC_MODELS, "ensemble": "Ensemble", "auto": "Auto",
                                "prophet": "Prophet", "var": "VAR"}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "arima": "Autoregressive integrated moving average: captures short-term autocorrelation in price changes.",
    "ets": "Exponential smoothing: weights recent observations more heavily and extrapolates the trend.",
    "linear": "Linear trend: least-squares trend line extended into the future.",
    "random_walk": "Random walk with drift: the statistical baseline for efficient markets.",
    "ml": "Machine learning regression on lagged returns and technical features.",
    "prophet": "Prophet: additive trend + seasonality model (optional dependency).",
}

#: Label -> years of price history
HISTORY_OPTIONS: Dict[str, int] = {"1 year": 1, "2 years": 2, "3 years": 3, "5 years": 5}
#: Label -> trading days
HORIZON_OPTIONS: Dict[str, int] = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}
CONFIDENCE_OPTIONS = [80, 90, 95, 99]

#: Financial metrics: label -> (statement, canonical item) or ("ratio", numerator spec, denominator spec)
FINANCIAL_METRICS: Dict[str, Dict[str, Any]] = {
    "Revenue": {"statement": "income_statement", "item": "revenue", "kind": "money"},
    "Gross Profit": {"statement": "income_statement", "item": "gross_profit", "kind": "money"},
    "Operating Income": {"statement": "income_statement", "item": "operating_income", "kind": "money"},
    "Net Income": {"statement": "income_statement", "item": "net_income", "kind": "money"},
    "EBITDA": {"statement": "income_statement", "item": "ebitda", "kind": "money"},
    "Diluted EPS": {"statement": "income_statement", "item": "diluted_eps", "kind": "price"},
    "Operating Cash Flow": {"statement": "cash_flow", "item": "operating_cash_flow", "kind": "money"},
    "Free Cash Flow": {"statement": "cash_flow", "item": "free_cash_flow", "kind": "money"},
    "Gross Margin": {"ratio": (("income_statement", "gross_profit"), ("income_statement", "revenue")),
                     "kind": "pct"},
    "Operating Margin": {"ratio": (("income_statement", "operating_income"), ("income_statement", "revenue")),
                         "kind": "pct"},
    "Net Margin": {"ratio": (("income_statement", "net_income"), ("income_statement", "revenue")), "kind": "pct"},
    "Return on Equity": {"ratio": (("income_statement", "net_income"), ("balance_sheet", "total_equity")),
                         "kind": "pct"},
}

SCENARIOS = ("Bear", "Base", "Bull")
SCENARIO_COLORS = {"Bear": COLORS["accent"], "Base": COLORS["info"], "Bull": COLORS["primary"]}

#: Macro indicator key (DataLoader.INDICATOR_CODES) -> label
MACRO_INDICATORS: Dict[str, str] = {
    "INFLATION": "Consumer Price Index (CPI)",
    "UNEMPLOYMENT": "Unemployment Rate",
    "INTEREST_RATE": "Federal Funds Rate",
    "TREASURY_10Y": "10-Year Treasury Yield",
    "GDP": "Gross Domestic Product",
    "RETAIL_SALES": "Retail Sales",
    "INDUSTRIAL_PRODUCTION": "Industrial Production",
    "HOUSE_PRICE_INDEX": "Case-Shiller Home Price Index",
    "CONSUMER_SENTIMENT": "Consumer Sentiment",
}

_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


# ---------------------------------------------------------------------------
# Resources and data access
# ---------------------------------------------------------------------------
class _NoData(Exception):
    """Raised inside cached loaders so empty results are not cached."""


@st.cache_resource(show_spinner=False)
def _get_data_loader() -> DataLoader:
    return DataLoader()


def _create_forecaster() -> Optional[Any]:
    """TimeSeriesForecaster instance (``None`` when the module is unavailable)."""
    if TimeSeriesForecaster is None:
        return None
    try:
        return TimeSeriesForecaster(_get_data_loader())
    except TypeError:
        return TimeSeriesForecaster()
    except Exception:
        logger.exception("TimeSeriesForecaster could not be created")
        return None


def _get_forecaster() -> Optional[Any]:
    if "_fc_forecaster" not in st.session_state:
        st.session_state["_fc_forecaster"] = _create_forecaster()
    return st.session_state["_fc_forecaster"]


@st.cache_data(ttl=900, show_spinner=False)
def _cached_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = _get_data_loader().get_historical_prices(ticker, start, end)
    if prices is None or prices.empty or "Close" not in prices.columns:
        raise _NoData(ticker)
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_statement(ticker: str, statement_type: str, period: str) -> pd.DataFrame:
    frame = _get_data_loader().get_financial_statements(ticker, statement_type, period)
    if frame is None or frame.empty:
        raise _NoData(ticker)
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_info(ticker: str) -> Dict[str, Any]:
    info = _get_data_loader().get_company_info(ticker)
    if not info:
        raise _NoData(ticker)
    return dict(info)


def _guarded(loader: Callable[..., Any], empty: Any, *args: Any) -> Any:
    try:
        return loader(*args)
    except _NoData:
        return empty
    except Exception:
        logger.exception("Data loading failed: %s%s", getattr(loader, "__name__", loader), args)
        return empty


def load_prices(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * years))
    return _guarded(_cached_prices, pd.DataFrame(), ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def load_company_info(ticker: str) -> Dict[str, Any]:
    return _guarded(_cached_info, {}, ticker)


def load_financial_data(ticker: str, period: str = "annual") -> Dict[str, Any]:
    """Canonical financial data bundle (statements may be empty frames)."""
    statements = {key: _guarded(_cached_statement, pd.DataFrame(), ticker, stype, period)
                  for key, stype in (("income_statement", "income"), ("balance_sheet", "balance"),
                                     ("cash_flow", "cash"))}
    info = load_company_info(ticker)
    prices = load_prices(ticker, 1)
    return build_financial_data(company_info=info, price_data=prices, **statements)


def _safe_call(func: Optional[Callable[..., Any]], *args: Any, **kwargs: Any) -> Any:
    """Call a model method; log and return ``None`` on any failure."""
    if func is None:
        return None
    try:
        return func(*args, **kwargs)
    except Exception:
        logger.exception("Forecast call %s failed", getattr(func, "__name__", func))
        return None


def _method(obj: Any, name: str) -> Optional[Callable[..., Any]]:
    method = getattr(obj, name, None) if obj is not None else None
    return method if callable(method) else None


def _valid_forecast(result: Any) -> bool:
    return isinstance(result, Mapping) and not charts.as_series(result.get("forecast")).empty


def _error_text(result: Any, default: str = "no forecast produced") -> str:
    if isinstance(result, Mapping) and result.get("error"):
        return str(result["error"])
    return default


def _label(model: Any) -> str:
    return MODEL_LABELS.get(str(model), str(model).replace("_", " ").title())


def _clean_ticker(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _fmt(value: Any, kind: str = "number") -> str:
    number = to_float(value)
    if number is None:
        return "N/A"
    if kind == "money":
        return format_large_number(number)
    if kind == "price":
        return f"${number:,.2f}"
    if kind == "pct":
        return f"{number * 100:.1f}%"
    if kind == "pct_points":
        return f"{number:+.1f}%"
    if kind == "multiple":
        return f"{number:.1f}x"
    return f"{number:,.2f}"


# ---------------------------------------------------------------------------
# Price forecasting
# ---------------------------------------------------------------------------
def _close(prices: pd.DataFrame) -> pd.Series:
    if prices is None or prices.empty or "Close" not in prices.columns:
        return pd.Series(dtype=float)
    return charts.as_series(prices["Close"])


def _average_forecast(forecasts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Equal-weight average of model forecasts (used when the ensemble call is unavailable)."""
    frames = {key: charts.forecast_parts(result) for key, result in forecasts.items()}
    means = pd.DataFrame({k: v[0].reset_index(drop=True) for k, v in frames.items()})
    if means.empty:
        return {}
    index = next(iter(frames.values()))[0].index[: len(means)]
    result = {"model": "ensemble", "forecast": pd.Series(means.mean(axis=1).values, index=index),
              "weights": {k: 1 / len(frames) for k in frames}, "components": {k: v[0] for k, v in frames.items()}}
    lowers = pd.DataFrame({k: v[1].reset_index(drop=True) for k, v in frames.items() if not v[1].empty})
    uppers = pd.DataFrame({k: v[2].reset_index(drop=True) for k, v in frames.items() if not v[2].empty})
    if not lowers.empty and not uppers.empty:
        result["lower"] = pd.Series(lowers.mean(axis=1).values[: len(index)], index=index[: len(lowers)])
        result["upper"] = pd.Series(uppers.mean(axis=1).values[: len(index)], index=index[: len(uppers)])
    return result


def run_price_forecast(forecaster: Any, prices: pd.DataFrame, models: List[str], horizon: int,
                       confidence: float, target_change_pct: float = 10.0) -> Dict[str, Any]:
    """
    Run every selected price model plus ensemble, backtest and probabilities.

    Returns a result dict with ``forecasts`` ({model: F}), ``errors``
    ({model: message}), ``ensemble`` (F or ``{}``), ``evaluation``
    (DataFrame or ``None``), ``prob_up`` (float or ``None``) and
    ``prob_change`` (dict or ``None``). Never raises.
    """
    result: Dict[str, Any] = {"forecasts": {}, "errors": {}, "ensemble": {}, "evaluation": None,
                              "prob_up": None, "prob_change": None, "history": _close(prices)}
    forecast_price = _method(forecaster, "forecast_price")
    if forecast_price is None:
        result["error"] = ("The price forecasting engine (TimeSeriesForecaster.forecast_price) is not available "
                           "in this installation.")
        return result
    for model in models:
        output = _safe_call(forecast_price, prices, horizon=horizon, model=model, confidence=confidence)
        if _valid_forecast(output):
            result["forecasts"][model] = output
        else:
            result["errors"][model] = _error_text(output)
    if not result["forecasts"]:
        result["error"] = "None of the selected models produced a forecast."
        return result

    if len(result["forecasts"]) == 1:
        result["ensemble"] = next(iter(result["forecasts"].values()))
    else:
        ensemble = _safe_call(_method(forecaster, "create_ensemble_forecast"), prices, horizon=horizon,
                              models=list(result["forecasts"]), confidence=confidence)
        result["ensemble"] = ensemble if _valid_forecast(ensemble) else _average_forecast(result["forecasts"])

    evaluation = _safe_call(_method(forecaster, "evaluate_models"), prices, models=list(result["forecasts"]),
                            test_size=min(max(horizon, 10), 60))
    if isinstance(evaluation, pd.DataFrame) and not evaluation.empty:
        result["evaluation"] = evaluation
    prob_up = _safe_call(_method(forecaster, "probability_of_price_increase"), prices, horizon=horizon)
    result["prob_up"] = to_float(prob_up)
    prob_change = _safe_call(_method(forecaster, "probability_of_price_change"), prices, horizon=horizon,
                             target_change_pct=target_change_pct)
    result["prob_change"] = prob_change if isinstance(prob_change, Mapping) and not prob_change.get("error") else None
    hist = (result["ensemble"] or {}).get("history")
    if not charts.as_series(hist).empty:
        result["history"] = charts.as_series(hist)
    return result


def price_forecast_table(result: Mapping[str, Any]) -> pd.DataFrame:
    """Forecast values per model plus ensemble and bounds, indexed by date."""
    columns: Dict[str, pd.Series] = {}
    for model, output in (result.get("forecasts") or {}).items():
        columns[_label(model)] = charts.as_series(output.get("forecast"))
    forecast, lower, upper = charts.forecast_parts(result.get("ensemble") or {})
    if not forecast.empty and len(result.get("forecasts") or {}) > 1:
        columns["Ensemble"] = forecast
    if not lower.empty:
        columns["Lower"] = lower
    if not upper.empty:
        columns["Upper"] = upper
    if not columns:
        return pd.DataFrame()
    frame = pd.DataFrame(columns)
    frame.index.name = "Date"
    return frame


def _render_price_tab(ticker: str) -> None:
    st.subheader("Share price forecast")
    forecaster = _get_forecaster()
    if _method(forecaster, "forecast_price") is None:
        st.info("Price forecasting models are not available in this installation "
                "(TimeSeriesForecaster.forecast_price is missing).")
        _render_price_result()
        return

    col1, col2, col3 = st.columns(3)
    history_label = col1.selectbox("Price history", list(HISTORY_OPTIONS), index=1, key="fc_price_history")
    horizon_label = col2.selectbox("Forecast horizon", list(HORIZON_OPTIONS), index=1, key="fc_price_horizon")
    confidence = col3.select_slider("Confidence level (%)", options=CONFIDENCE_OPTIONS, value=95,
                                    key="fc_price_confidence")
    models = st.multiselect("Models", options=list(PRICE_MODELS), default=["arima", "ets", "linear"],
                            format_func=_label, key="fc_price_models",
                            help="Several models are blended into a weighted ensemble.")
    if not PROPHET_AVAILABLE:
        st.caption("Prophet is optional and not installed; install `prophet` to enable it.")
    target = st.number_input("Probability target move (%)", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
                             key="fc_price_target",
                             help="Probability of the price ending above +x% / below -x% at the horizon.")
    params = {"ticker": ticker, "history": history_label, "horizon": horizon_label, "confidence": confidence,
              "models": list(models), "target": float(target)}

    if st.button("Run price forecast", type="primary", key="fc_price_run", disabled=not models or not ticker):
        with st.spinner(f"Forecasting {ticker} ..."):
            prices = load_prices(ticker, HISTORY_OPTIONS[history_label])
            if prices.empty:
                st.session_state[PRICE_RESULT_KEY] = {"params": params, "ticker": ticker,
                                                      "error": f"No price history is available for {ticker}."}
            else:
                info = load_company_info(ticker)
                output = run_price_forecast(forecaster, prices, list(models), HORIZON_OPTIONS[horizon_label],
                                            confidence / 100.0, float(target))
                output.update({"params": params, "ticker": ticker, "company_name": info.get("name") or ticker})
                st.session_state[PRICE_RESULT_KEY] = output
    _render_price_result(params)


def _render_price_result(current_params: Optional[Mapping[str, Any]] = None) -> None:
    result = st.session_state.get(PRICE_RESULT_KEY)
    if not result:
        st.caption("Choose the settings above and click **Run price forecast**.")
        return
    params = result.get("params") or {}
    if current_params is not None and dict(current_params) != dict(params):
        st.info(f"Showing the last forecast ({params.get('ticker')}, {params.get('horizon')}). "
                "Click **Run price forecast** to update it with the current settings.")
    if result.get("error"):
        st.warning(result["error"])
        for model, message in (result.get("errors") or {}).items():
            st.caption(f"{_label(model)}: {message}")
        return

    ticker = result.get("ticker", "")
    history = charts.as_series(result.get("history"))
    ensemble = result.get("ensemble") or {}
    forecast, lower, upper = charts.forecast_parts(ensemble)
    current = float(history.iloc[-1]) if not history.empty else None
    target = float(forecast.iloc[-1]) if not forecast.empty else None
    change = (target / current - 1) if current and target is not None else None

    st.markdown(f"#### {result.get('company_name', ticker)} ({ticker})")
    if not history.empty:
        st.caption(f"{len(history)} closes up to {pd.Timestamp(history.index[-1]):%Y-%m-%d} | horizon "
                   f"{params.get('horizon')} | {params.get('confidence')}% confidence")
    cols = st.columns(4)
    cols[0].metric("Last close", _fmt(current, "price"))
    cols[1].metric(f"Forecast ({params.get('horizon')})", _fmt(target, "price"),
                   delta=f"{change * 100:+.1f}%" if change is not None else None)
    cols[2].metric("Forecast range", f"{_fmt(lower.iloc[-1], 'price')} - {_fmt(upper.iloc[-1], 'price')}"
                   if not lower.empty and not upper.empty else "N/A")
    cols[3].metric("P(price up)", _fmt(result.get("prob_up"), "pct"))
    if change is not None:
        message = (f"The {'ensemble' if len(result['forecasts']) > 1 else _label(next(iter(result['forecasts'])))} "
                   f"projects a {change * 100:+.1f}% move over {params.get('horizon')}. Statistical price forecasts "
                   "are highly uncertain - treat them as one input among many.")
        if change > 0.05:
            st.success(message)
        elif change < -0.05:
            st.error(message)
        else:
            st.info(message)
    if result.get("errors"):
        st.warning("Some models failed: " + "; ".join(f"{_label(m)} ({e})" for m, e in result["errors"].items()))

    view_forecast, view_models, view_prob, view_data = st.tabs(
        ["Forecast", "Model comparison", "Probabilities", "Data"])
    with view_forecast:
        components = result["forecasts"] if len(result["forecasts"]) > 1 else None
        st.plotly_chart(charts.forecast_chart(history, components, ensemble, title=f"{ticker} price forecast",
                                              history_points=504, labels=MODEL_LABELS),
                        key="fc_price_chart")
    with view_models:
        st.plotly_chart(charts.forecast_chart(history, result["forecasts"], None if len(result["forecasts"]) > 1
                                              else ensemble, title="Model forecasts", history_points=252,
                                              labels=MODEL_LABELS, band_label="Ensemble interval"),
                        key="fc_price_models_chart")
        evaluation = result.get("evaluation")
        if isinstance(evaluation, pd.DataFrame) and not evaluation.empty:
            st.markdown("**Backtest** (models refit on history excluding the last test window)")
            display = evaluation.copy()
            if "model" in display:
                display["model"] = display["model"].map(_label)
            st.dataframe(display, hide_index=True, key="fc_price_eval_table")
            metric = "mape" if "mape" in evaluation.columns else ("rmse" if "rmse" in evaluation.columns else "mae")
            st.plotly_chart(charts.evaluation_chart(evaluation, metric=metric, labels=MODEL_LABELS,
                                                    title=f"Backtest {metric.upper()} (lower is better)"),
                            key="fc_price_eval_chart")
        else:
            st.caption("Backtest results are not available.")
        weights = ensemble.get("weights") if isinstance(ensemble, Mapping) else None
        if weights and len(result["forecasts"]) > 1:
            st.plotly_chart(charts.weights_chart(weights, labels=MODEL_LABELS), key="fc_price_weights_chart")
        with st.expander("About the models"):
            for model in result["forecasts"]:
                st.markdown(f"**{_label(model)}** - {MODEL_DESCRIPTIONS.get(model, '')}")
    with view_prob:
        prob_change = result.get("prob_change") or {}
        cols = st.columns(3)
        cols[0].metric("P(price higher)", _fmt(result.get("prob_up"), "pct"))
        cols[1].metric(f"P(> +{params.get('target', 10):.0f}%)", _fmt(prob_change.get("prob_above"), "pct"))
        cols[2].metric(f"P(< -{params.get('target', 10):.0f}%)", _fmt(prob_change.get("prob_below"), "pct"))
        if prob_change:
            st.plotly_chart(charts.distribution_chart(prob_change.get("percentiles"), current_price=current,
                                                      expected_price=prob_change.get("expected_price")),
                            key="fc_price_dist_chart")
            st.caption("Monte-Carlo simulation of log returns calibrated on the selected history.")
        else:
            st.caption("Probability simulation is not available.")
    with view_data:
        table = price_forecast_table(result)
        if table.empty:
            st.caption("No forecast data.")
        else:
            st.dataframe(table.style.format("{:,.2f}"), key="fc_price_table")
            st.download_button("Download forecast (CSV)", data=to_csv_bytes(table),
                               file_name=f"{ticker}_price_forecast.csv", mime="text/csv",
                               key="fc_price_download", on_click="ignore")


# ---------------------------------------------------------------------------
# Financial metric forecasting
# ---------------------------------------------------------------------------
def metric_history(financial_data: Mapping[str, Any], metric: str) -> pd.Series:
    """Chronological (oldest first) series for a :data:`FINANCIAL_METRICS` entry."""
    spec = FINANCIAL_METRICS[metric]
    if "ratio" in spec:
        (num_stmt, num_item), (den_stmt, den_item) = spec["ratio"]
        numerator = get_series(financial_data.get(num_stmt), num_item, ascending=True)
        denominator = get_series(financial_data.get(den_stmt), den_item, ascending=True)
        common = numerator.index.intersection(denominator.index)
        if len(common) == 0:
            return pd.Series(dtype=float)
        denominator = denominator.loc[common].where(denominator.loc[common] != 0)
        series = (numerator.loc[common] / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        series = get_series(financial_data.get(spec["statement"]), spec["item"], ascending=True)
    try:
        series.index = pd.to_datetime(series.index)
    except (TypeError, ValueError):
        pass
    series.name = metric
    return series.astype(float)


def run_metric_forecast(forecaster: Any, financial_data: Mapping[str, Any], metrics: List[str], periods: int,
                        models: List[str]) -> Dict[str, Any]:
    """Forecast each metric with every model plus an ensemble. Never raises."""
    single = _method(forecaster, "forecast_financial_metric")
    ensemble_fn = _method(forecaster, "create_ensemble_financial_forecast")
    output: Dict[str, Any] = {"metrics": {}}
    if single is None and ensemble_fn is None:
        output["error"] = ("The financial forecasting engine (TimeSeriesForecaster.forecast_financial_metric) is "
                           "not available in this installation.")
        return output
    for metric in metrics:
        history = metric_history(financial_data, metric)
        entry: Dict[str, Any] = {"history": history, "forecasts": {}, "errors": {}, "ensemble": {}}
        if len(history) < 2:
            entry["error"] = "Not enough reported periods to forecast (at least two are needed)."
            output["metrics"][metric] = entry
            continue
        for model in models:
            result = _safe_call(single, history, periods=periods, model=model)
            if _valid_forecast(result):
                entry["forecasts"][model] = result
            else:
                entry["errors"][model] = _error_text(result)
        if len(entry["forecasts"]) > 1:
            ensemble = _safe_call(ensemble_fn, history, periods=periods, models=list(entry["forecasts"]))
            entry["ensemble"] = ensemble if _valid_forecast(ensemble) else _average_forecast(entry["forecasts"])
        elif entry["forecasts"]:
            entry["ensemble"] = next(iter(entry["forecasts"].values()))
        else:
            entry["error"] = "No model produced a forecast for this metric."
        output["metrics"][metric] = entry
    return output


def metric_summary_table(result: Mapping[str, Any]) -> pd.DataFrame:
    """One row per metric: last reported value, final forecast and implied growth."""
    rows = []
    for metric, entry in (result.get("metrics") or {}).items():
        history = charts.as_series(entry.get("history"))
        forecast, _, _ = charts.forecast_parts(entry.get("ensemble") or {})
        kind = FINANCIAL_METRICS.get(metric, {}).get("kind", "number")
        last = float(history.iloc[-1]) if not history.empty else None
        final = float(forecast.iloc[-1]) if not forecast.empty else None
        if kind == "pct":
            change = f"{(final - last) * 100:+.1f} pp" if last is not None and final is not None else "N/A"
        else:
            periods = len(forecast)
            annual = (final / last) ** (1 / periods) - 1 \
                if last is not None and final is not None and last > 0 and final > 0 and periods else None
            change = f"{annual * 100:+.1f}% / period" if annual is not None else "N/A"
        rows.append({"Metric": metric, "Last reported": _fmt(last, kind),
                     "Forecast (final period)": _fmt(final, kind), "Implied change": change,
                     "Status": entry.get("error") or "OK"})
    return pd.DataFrame(rows)


def metric_forecast_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Long table of history and forecasts for CSV export."""
    rows = []
    for metric, entry in (result.get("metrics") or {}).items():
        for date, value in charts.as_series(entry.get("history")).items():
            rows.append({"metric": metric, "date": date, "type": "reported", "model": "", "value": value})
        sources = dict(entry.get("forecasts") or {})
        if len(sources) > 1 and entry.get("ensemble"):
            sources["ensemble"] = entry["ensemble"]
        for model, forecast in sources.items():
            values, lower, upper = charts.forecast_parts(forecast)
            for date, value in values.items():
                rows.append({"metric": metric, "date": date, "type": "forecast", "model": model, "value": value,
                             "lower": lower.get(date), "upper": upper.get(date)})
    return pd.DataFrame(rows)


def _render_metrics_tab(ticker: str) -> None:
    st.subheader("Financial metric forecast")
    forecaster = _get_forecaster()
    if _method(forecaster, "forecast_financial_metric") is None:
        st.info("Financial metric forecasting is not available in this installation "
                "(TimeSeriesForecaster.forecast_financial_metric is missing).")
        _render_metric_result()
        return
    metrics = st.multiselect("Metrics", options=list(FINANCIAL_METRICS), default=["Revenue", "Net Income",
                                                                                 "Diluted EPS"],
                             key="fc_metric_items")
    col1, col2, col3 = st.columns(3)
    frequency = col1.selectbox("Statements", ["Annual", "Quarterly"], key="fc_metric_frequency")
    periods = col2.slider("Periods to forecast", min_value=1, max_value=8, value=3, key="fc_metric_periods")
    models = col3.multiselect("Models", options=list(METRIC_MODELS), default=["linear", "cagr", "ets"],
                              format_func=_label, key="fc_metric_models")
    params = {"ticker": ticker, "metrics": list(metrics), "frequency": frequency, "periods": periods,
              "models": list(models)}
    if st.button("Run metric forecast", type="primary", key="fc_metric_run",
                 disabled=not metrics or not models or not ticker):
        with st.spinner(f"Forecasting {ticker} fundamentals ..."):
            financial_data = load_financial_data(ticker, frequency.lower())
            if all(financial_data[k].empty for k in ("income_statement", "balance_sheet", "cash_flow")):
                output = {"error": f"No financial statements are available for {ticker}."}
            else:
                output = run_metric_forecast(forecaster, financial_data, list(metrics), int(periods), list(models))
            output.update({"params": params, "ticker": ticker})
            st.session_state[METRIC_RESULT_KEY] = output
    _render_metric_result(params)


def _render_metric_result(current_params: Optional[Mapping[str, Any]] = None) -> None:
    result = st.session_state.get(METRIC_RESULT_KEY)
    if not result:
        st.caption("Choose metrics and models, then click **Run metric forecast**.")
        return
    params = result.get("params") or {}
    if current_params is not None and dict(current_params) != dict(params):
        st.info(f"Showing the last metric forecast ({params.get('ticker')}). Click **Run metric forecast** to "
                "update it with the current settings.")
    if result.get("error"):
        st.warning(result["error"])
        return
    ticker = result.get("ticker", "")
    summary = metric_summary_table(result)
    if not summary.empty:
        st.dataframe(summary, hide_index=True, key="fc_metric_summary")
    entries = result.get("metrics") or {}
    if entries:
        tabs = st.tabs(list(entries))
        for tab, (metric, entry) in zip(tabs, entries.items()):
            with tab:
                if entry.get("error") and not entry.get("forecasts"):
                    st.info(entry["error"])
                kind = FINANCIAL_METRICS.get(metric, {}).get("kind", "number")
                scale = 100.0 if kind == "pct" else 1.0
                history = charts.as_series(entry.get("history")) * scale
                forecasts = {m: _scaled(f, scale) for m, f in (entry.get("forecasts") or {}).items()}
                ensemble = _scaled(entry.get("ensemble") or {}, scale) if len(forecasts) > 1 else None
                st.plotly_chart(charts.forecast_chart(history, forecasts, ensemble, title=f"{ticker} {metric}",
                                                      yaxis_title="%" if kind == "pct" else metric,
                                                      labels=MODEL_LABELS),
                                key=f"fc_metric_chart_{_slug(metric)}")
                for model, message in (entry.get("errors") or {}).items():
                    st.caption(f"{_label(model)}: {message}")
    frame = metric_forecast_frame(result)
    if not frame.empty:
        st.download_button("Download metric forecasts (CSV)", data=to_csv_bytes(frame),
                           file_name=f"{ticker}_metric_forecast.csv", mime="text/csv", key="fc_metric_download",
                           on_click="ignore")


def _scaled(result: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    if scale == 1.0 or not isinstance(result, Mapping):
        return dict(result or {})
    scaled = dict(result)
    for key in ("forecast", "lower", "upper"):
        series = charts.as_series(result.get(key))
        if not series.empty:
            scaled[key] = series * scale
    return scaled


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------
def scenario_base(financial_data: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Inputs for the scenario model derived from the latest annual statements."""
    income = financial_data.get("income_statement")
    market = financial_data.get("market_data") or {}
    revenue_history = get_series(income, "revenue", ascending=True)
    revenue = get_latest(income, "revenue")
    net_income = get_latest(income, "net_income")
    shares = get_latest(income, "diluted_shares") or to_float(market.get("shares_outstanding"))
    price = to_float(market.get("share_price"))
    eps = safe_divide(net_income, shares)
    growth = cagr(revenue_history) if len(revenue_history) >= 2 else None
    return {
        "revenue": revenue,
        "net_income": net_income,
        "net_margin": safe_divide(net_income, revenue),
        "shares": shares,
        "price": price,
        "eps": eps,
        "pe": safe_divide(price, eps) if eps and eps > 0 else None,
        "revenue_cagr": growth,
        "revenue_history": revenue_history,
    }


def default_assumptions(base: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Bear/base/bull assumptions (growth/margin in percent, P/E multiple, probability in percent)."""
    growth = to_float(base.get("revenue_cagr"))
    growth = 5.0 if growth is None else float(np.clip(growth * 100, -10.0, 30.0))
    margin = to_float(base.get("net_margin"))
    margin = 10.0 if margin is None else float(np.clip(margin * 100, -50.0, 60.0))
    pe = to_float(base.get("pe"))
    pe = 18.0 if pe is None else float(np.clip(pe, 5.0, 60.0))
    return {
        "Bear": {"growth": round(growth - 5.0, 1), "margin": round(margin - 2.0, 1), "pe": round(pe * 0.8, 1),
                 "probability": 25.0},
        "Base": {"growth": round(growth, 1), "margin": round(margin, 1), "pe": round(pe, 1), "probability": 50.0},
        "Bull": {"growth": round(growth + 5.0, 1), "margin": round(margin + 2.0, 1), "pe": round(pe * 1.2, 1),
                 "probability": 25.0},
    }


def compute_scenarios(base: Mapping[str, Any], assumptions: Mapping[str, Mapping[str, float]],
                      years: int) -> Dict[str, Any]:
    """
    Project revenue, earnings and the implied share price for each scenario.

    Args:
        base: Output of :func:`scenario_base`.
        assumptions: ``{scenario: {"growth": %, "margin": %, "pe": x, "probability": %}}``.
        years: Projection horizon in years.

    Returns:
        ``{"table": DataFrame (one row per scenario), "paths": {scenario: Series of revenue by year},
        "weighted_price": float | None, "weighted_upside": float | None, "probability_total": float}``.
    """
    revenue = to_float(base.get("revenue"))
    shares = to_float(base.get("shares"))
    price = to_float(base.get("price"))
    if revenue is None or revenue <= 0 or not shares:
        return {"table": pd.DataFrame(), "paths": {}, "weighted_price": None, "weighted_upside": None,
                "probability_total": 0.0, "error": "Revenue and share count are required for scenario analysis."}
    history = base.get("revenue_history")
    last_year = pd.Timestamp(history.index[-1]).year if isinstance(history, pd.Series) and not history.empty \
        else datetime.now().year - 1
    probability_total = sum(max(to_float(a.get("probability"), 0.0), 0.0) for a in assumptions.values())
    rows, paths = [], {}
    for name, assumption in assumptions.items():
        growth = to_float(assumption.get("growth"), 0.0) / 100
        margin = to_float(assumption.get("margin"), 0.0) / 100
        multiple = max(to_float(assumption.get("pe"), 0.0), 0.0)
        probability = max(to_float(assumption.get("probability"), 0.0), 0.0)
        growth = max(growth, -0.99)
        path = pd.Series([revenue * (1 + growth) ** year for year in range(0, years + 1)],
                         index=[last_year + year for year in range(0, years + 1)], name=name)
        final_revenue = float(path.iloc[-1])
        net_income = final_revenue * margin
        eps = net_income / shares
        implied = max(eps * multiple, 0.0)
        upside = safe_divide(implied, price) - 1 if price and safe_divide(implied, price) is not None else None
        annualised = ((implied / price) ** (1 / years) - 1) if price and implied > 0 and years > 0 else None
        rows.append({"Scenario": name, "Probability": probability / probability_total if probability_total else None,
                     "Revenue Growth": growth, "Net Margin": margin, "Exit P/E": multiple,
                     f"Revenue (Y{years})": final_revenue, "Net Income": net_income, "EPS": eps,
                     "Implied Price": implied, "Upside": upside, "Annualised Return": annualised})
        paths[name] = path
    table = pd.DataFrame(rows).set_index("Scenario")
    weighted_price = None
    if probability_total > 0:
        weighted_price = float((table["Implied Price"] * table["Probability"]).sum())
    weighted_upside = (weighted_price / price - 1) if weighted_price is not None and price else None
    return {"table": table, "paths": paths, "weighted_price": weighted_price, "weighted_upside": weighted_upside,
            "probability_total": probability_total}


def _render_scenario_tab(ticker: str) -> None:
    st.subheader("Scenario analysis")
    st.caption("Project revenue and earnings under bear, base and bull assumptions and translate them into an "
               "implied share price with an exit P/E multiple. Results update as you edit the assumptions.")
    if not ticker:
        st.info("Enter a ticker to build scenarios.")
        return
    with st.spinner("Loading financial statements ..."):
        financial_data = load_financial_data(ticker, "annual")
    base = scenario_base(financial_data)
    if base["revenue"] is None or not base["shares"]:
        st.info(f"Scenario analysis needs reported revenue and a share count, which are not available for {ticker}.")
        return

    cols = st.columns(4)
    cols[0].metric("Revenue (latest FY)", _fmt(base["revenue"], "money"))
    cols[1].metric("Net margin", _fmt(base["net_margin"], "pct"))
    cols[2].metric("EPS", _fmt(base["eps"], "price"))
    cols[3].metric("Share price / P/E", f"{_fmt(base['price'], 'price')} / {_fmt(base['pe'], 'multiple')}")
    reference = _metric_reference_growth(ticker)
    if reference is not None:
        st.caption(f"Model-implied revenue growth from the Financial Metrics tab: {reference * 100:+.1f}% per period.")

    years = st.slider("Projection horizon (years)", min_value=1, max_value=10, value=3, key="fc_scn_years")
    defaults = default_assumptions(base)
    assumptions: Dict[str, Dict[str, float]] = {}
    columns = st.columns(len(SCENARIOS))
    for column, scenario in zip(columns, SCENARIOS):
        values = defaults[scenario]
        prefix = f"fc_scn_{_slug(ticker)}_{scenario.lower()}"
        with column:
            st.markdown(f"**{scenario} case**")
            assumptions[scenario] = {
                "growth": st.number_input("Revenue growth (%/yr)", min_value=-50.0, max_value=100.0,
                                          value=values["growth"], step=0.5, key=f"{prefix}_growth"),
                "margin": st.number_input("Net margin (%)", min_value=-100.0, max_value=100.0,
                                          value=values["margin"], step=0.5, key=f"{prefix}_margin"),
                "pe": st.number_input("Exit P/E (x)", min_value=0.0, max_value=200.0, value=values["pe"],
                                      step=0.5, key=f"{prefix}_pe"),
                "probability": st.number_input("Probability (%)", min_value=0.0, max_value=100.0,
                                               value=values["probability"], step=5.0, key=f"{prefix}_prob"),
            }
    outcome = compute_scenarios(base, assumptions, int(years))
    if outcome.get("error"):
        st.info(outcome["error"])
        return
    if abs(outcome["probability_total"] - 100.0) > 0.01:
        if outcome["probability_total"] <= 0:
            st.warning("Assign a probability to at least one scenario to compute a weighted target.")
        else:
            st.caption(f"Probabilities sum to {outcome['probability_total']:.0f}% and were normalised to 100%.")

    cols = st.columns(3)
    cols[0].metric("Probability-weighted price", _fmt(outcome["weighted_price"], "price"),
                   delta=f"{outcome['weighted_upside'] * 100:+.1f}%" if outcome["weighted_upside"] is not None
                   else None)
    table = outcome["table"]
    cols[1].metric("Bear / Bull price", f"{_fmt(table.loc['Bear', 'Implied Price'], 'price')} / "
                                        f"{_fmt(table.loc['Bull', 'Implied Price'], 'price')}")
    cols[2].metric("Base-case annualised return", _fmt(table.loc["Base", "Annualised Return"], "pct"))

    display = pd.DataFrame(index=table.index)
    formats = {"Probability": "pct", "Revenue Growth": "pct", "Net Margin": "pct", "Exit P/E": "multiple",
               f"Revenue (Y{years})": "money", "Net Income": "money", "EPS": "price", "Implied Price": "price",
               "Upside": "pct", "Annualised Return": "pct"}
    for column, kind in formats.items():
        display[column] = [_fmt(v, kind) for v in table[column]]
    st.dataframe(display, key="fc_scn_table")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        history = base["revenue_history"].copy()
        history.index = [pd.Timestamp(i).year for i in history.index]
        st.plotly_chart(charts.scenario_paths_chart(outcome["paths"], history, title="Revenue projection",
                                                    yaxis_title="Revenue", colors=SCENARIO_COLORS),
                        key="fc_scn_paths_chart")
    with chart_right:
        st.plotly_chart(charts.scenario_price_chart(table["Implied Price"].to_dict(), base["price"],
                                                    outcome["weighted_price"], colors=SCENARIO_COLORS),
                        key="fc_scn_price_chart")
    st.download_button("Download scenarios (CSV)", data=to_csv_bytes(table), file_name=f"{ticker}_scenarios.csv",
                       mime="text/csv", key="fc_scn_download", on_click="ignore")


def _metric_reference_growth(ticker: str) -> Optional[float]:
    """Per-period revenue growth implied by a stored metric forecast for ``ticker`` (if any)."""
    result = st.session_state.get(METRIC_RESULT_KEY) or {}
    if result.get("ticker") != ticker:
        return None
    entry = (result.get("metrics") or {}).get("Revenue") or {}
    history = charts.as_series(entry.get("history"))
    forecast, _, _ = charts.forecast_parts(entry.get("ensemble") or {})
    if history.empty or forecast.empty or history.iloc[-1] <= 0 or forecast.iloc[-1] <= 0:
        return None
    return (forecast.iloc[-1] / history.iloc[-1]) ** (1 / len(forecast)) - 1


# ---------------------------------------------------------------------------
# Economic indicators
# ---------------------------------------------------------------------------
def _macro_series(frame: Any) -> pd.Series:
    if isinstance(frame, pd.Series):
        series = frame
    elif isinstance(frame, pd.DataFrame) and not frame.empty:
        series = frame.iloc[:, 0]
    else:
        return pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce").dropna()
    try:
        series.index = pd.to_datetime(series.index)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    return series.sort_index().astype(float)


def run_macro_forecast(forecaster: Any, data: Mapping[str, Any], horizon: int, models: List[str],
                       use_var: bool) -> Dict[str, Any]:
    """Forecast every non-empty macro series; optional VAR across them. Never raises."""
    output: Dict[str, Any] = {"indicators": {}, "var": None, "missing": []}
    model_methods = {"arima": "forecast_economic_indicator_arima", "ml": "forecast_economic_indicator_ml",
                     "ensemble": "create_ensemble_indicator_forecast"}
    for key, frame in data.items():
        series = _macro_series(frame)
        if len(series) < 8:
            output["missing"].append(key)
            continue
        entry: Dict[str, Any] = {"history": series, "forecasts": {}, "errors": {}}
        for model in models:
            method = _method(forecaster, model_methods.get(model, ""))
            if method is None:
                entry["errors"][model] = "not available"
                continue
            result = _safe_call(method, series, horizon=horizon)
            if _valid_forecast(result):
                entry["forecasts"][model] = result
            else:
                entry["errors"][model] = _error_text(result)
        output["indicators"][key] = entry
    if use_var and len(output["indicators"]) >= 2:
        frame = pd.DataFrame({k: v["history"] for k, v in output["indicators"].items()})
        frame = frame.resample("ME").last().ffill().dropna()
        var_fn = _method(forecaster, "forecast_economic_indicators_var")
        if var_fn is not None and len(frame) >= 12:
            result = _safe_call(var_fn, frame, horizon=horizon)
            if isinstance(result, Mapping) and result.get("forecasts") and not result.get("error"):
                output["var"] = {"result": result, "history": frame}
    return output


def _render_macro_tab() -> None:
    st.subheader("Economic indicators")
    datareader = importlib.util.find_spec("pandas_datareader") is not None
    loader = _get_data_loader()
    if not datareader or getattr(loader, "offline", False):
        st.info("Macro-economic data is downloaded from FRED and needs network access plus the optional "
                "`pandas-datareader` package. "
                + ("The package is not installed. " if not datareader else "")
                + ("Offline mode is enabled. " if getattr(loader, "offline", False) else "")
                + "You can still try to load the data below.")
    forecaster = _get_forecaster()
    available = [k for k in MACRO_INDICATORS if k in getattr(DataLoader, "INDICATOR_CODES", MACRO_INDICATORS)]
    indicators = st.multiselect("Indicators", options=available, default=available[:2],
                                format_func=lambda k: MACRO_INDICATORS.get(k, k), key="fc_macro_indicators")
    col1, col2, col3 = st.columns(3)
    years = col1.slider("History (years)", min_value=3, max_value=30, value=10, key="fc_macro_years")
    horizon = col2.slider("Horizon (periods)", min_value=3, max_value=24, value=12, key="fc_macro_horizon")
    models = col3.multiselect("Models", options=list(MACRO_MODELS), default=["arima", "ensemble"],
                              format_func=lambda k: MACRO_MODELS[k], key="fc_macro_models")
    use_var = st.checkbox("Joint VAR forecast (two or more indicators)", value=True, key="fc_macro_var")
    if st.button("Load data and forecast", key="fc_macro_run", disabled=not indicators or not models):
        with st.spinner("Loading macro-economic data ..."):
            start = (datetime.now() - timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
            try:
                data = loader.get_macro_indicators(list(indicators), start_date=start) or {}
            except Exception:
                logger.exception("Macro indicator loading failed")
                data = {}
            output = run_macro_forecast(forecaster, data, int(horizon), list(models), bool(use_var)) \
                if forecaster is not None else {"indicators": {}, "missing": list(indicators), "var": None}
            output["params"] = {"indicators": list(indicators), "years": years, "horizon": horizon,
                                "models": list(models)}
            output["requested"] = list(indicators)
            st.session_state[MACRO_RESULT_KEY] = output
    result = st.session_state.get(MACRO_RESULT_KEY)
    if not result:
        return
    if not result.get("indicators"):
        st.info("No macro-economic data could be loaded for the selected indicators. Check the network connection "
                "and that `pandas-datareader` is installed.")
        return
    if result.get("missing"):
        st.caption("No data for: " + ", ".join(MACRO_INDICATORS.get(k, k) for k in result["missing"]))
    labels = {**MODEL_LABELS, **MACRO_MODELS}
    for key, entry in result["indicators"].items():
        forecasts = entry.get("forecasts") or {}
        ensemble = forecasts.get("ensemble")
        components = {k: v for k, v in forecasts.items() if k != "ensemble"} if ensemble else forecasts
        st.plotly_chart(charts.forecast_chart(entry["history"], components, ensemble,
                                              title=MACRO_INDICATORS.get(key, key), yaxis_title="", labels=labels),
                        key=f"fc_macro_chart_{_slug(key)}")
        for model, message in (entry.get("errors") or {}).items():
            st.caption(f"{labels.get(model, model)}: {message}")
    if result.get("var"):
        st.plotly_chart(charts.var_forecast_chart(result["var"]["history"], result["var"]["result"],
                                                  labels=MACRO_INDICATORS, title="Vector autoregression (VAR)"),
                        key="fc_macro_var_chart")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _ticker_input() -> str:
    active = get_active_ticker()
    if "fc_ticker" not in st.session_state:
        st.session_state["fc_ticker"] = active or DEFAULT_TICKER
        st.session_state["_fc_last_active"] = active
    elif active and st.session_state.get("_fc_last_active") != active:
        # A different company was selected elsewhere in the app: follow it.
        st.session_state["fc_ticker"] = active
        st.session_state["_fc_last_active"] = active
    raw = st.text_input("Ticker symbol", key="fc_ticker", max_chars=15,
                        help="Defaults to the company selected in Company Analysis.")
    ticker = _clean_ticker(raw)
    if ticker and not _TICKER_PATTERN.match(ticker):
        st.warning(f"'{raw}' does not look like a valid ticker symbol.")
        return ""
    return ticker


def run_forecasting_page() -> None:
    """Render the forecasting page."""
    st.title("Forecasting")
    st.markdown("Statistical forecasts of share prices, fundamentals and macro-economic indicators, plus "
                "scenario analysis. Forecasts are model estimates, not predictions or investment advice.")
    ticker = _ticker_input()
    price_tab, metric_tab, scenario_tab, macro_tab = st.tabs(
        ["Price Forecast", "Financial Metrics", "Scenarios", "Economic Indicators"])
    sections = ((price_tab, lambda: _render_price_tab(ticker)), (metric_tab, lambda: _render_metrics_tab(ticker)),
                (scenario_tab, lambda: _render_scenario_tab(ticker)), (macro_tab, _render_macro_tab))
    for tab, render in sections:
        with tab:
            try:
                render()
            except Exception:  # never show a traceback to users
                logger.exception("Forecasting page section failed")
                st.error("This section could not be displayed because of an unexpected error. "
                         "Please try different settings.")


if __name__ == "__main__":  # pragma: no cover
    run_forecasting_page()
