"""
Plotly chart builders for the forecasting page.

Every builder accepts forecast results in the ``TimeSeriesForecaster``
schema (``{"forecast": Series, "lower": Series, "upper": Series, ...}``) or
plain Series, tolerates missing/NaN/empty inputs and returns a dark-themed
``go.Figure`` (an empty figure with an explanatory message when there is
nothing to draw). Nothing here raises for bad data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS

logger = logging.getLogger(__name__)

#: Line colours cycled for model series (ensemble/history have fixed colours)
SERIES_COLORS = [COLORS["secondary"], COLORS["accent"], COLORS["info"], COLORS["warning"], COLORS["success"],
                 COLORS["danger"], "#fcbe6e", "#91bea8"]
HISTORY_COLOR = "#e0e0e0"
ENSEMBLE_COLOR = COLORS["primary"]
DEFAULT_HEIGHT = 480


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(150,150,150,{alpha})"
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def apply_theme(fig: go.Figure, title: str = "", height: Optional[int] = None, yaxis_title: str = "",
                xaxis_title: str = "") -> go.Figure:
    """Apply the application's dark theme."""
    fig.update_layout(
        title={"text": title, "font": {"size": VIZ_SETTINGS["title_font_size"]}},
        template="plotly_dark",
        paper_bgcolor=VIZ_SETTINGS["background"],
        plot_bgcolor=VIZ_SETTINGS["background"],
        font={"family": VIZ_SETTINGS["font_family"], "color": VIZ_SETTINGS["text_color"],
              "size": VIZ_SETTINGS["axis_font_size"]},
        height=height or DEFAULT_HEIGHT,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=VIZ_SETTINGS["grid_color"], title_text=xaxis_title)
    fig.update_yaxes(gridcolor=VIZ_SETTINGS["grid_color"], title_text=yaxis_title)
    return fig


def empty_figure(message: str = "No data available", height: Optional[int] = None, title: str = "") -> go.Figure:
    """Blank themed figure showing ``message``."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 14, "color": VIZ_SETTINGS["text_color"]})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_theme(fig, title=title, height=height or 300)
    fig.update_layout(hovermode=False)
    return fig


def as_series(values: Any) -> pd.Series:
    """Numeric, finite, index-sorted Series from a Series/DataFrame/sequence (empty when invalid)."""
    try:
        if values is None:
            return pd.Series(dtype=float)
        if isinstance(values, pd.DataFrame):
            if values.empty:
                return pd.Series(dtype=float)
            column = next((c for c in ("forecast", "mean", "Close", "value") if c in values.columns), values.columns[0])
            values = values[column]
        series = values if isinstance(values, pd.Series) else pd.Series(list(values))
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return pd.Series(dtype=float)
        try:
            series = series.sort_index()
        except TypeError:
            pass
        return series.astype(float)
    except Exception as exc:  # defensive: never break a chart on odd input
        logger.debug("as_series failed: %s", exc)
        return pd.Series(dtype=float)


def forecast_parts(result: Any) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """``(forecast, lower, upper)`` Series from a forecast result dict, DataFrame or Series."""
    if isinstance(result, Mapping):
        return as_series(result.get("forecast")), as_series(result.get("lower")), as_series(result.get("upper"))
    if isinstance(result, pd.DataFrame):
        lower = result["lower"] if "lower" in result.columns else None
        upper = result["upper"] if "upper" in result.columns else None
        return as_series(result), as_series(lower), as_series(upper)
    return as_series(result), pd.Series(dtype=float), pd.Series(dtype=float)


def _band(fig: go.Figure, lower: pd.Series, upper: pd.Series, color: str, name: str) -> None:
    if lower.empty or upper.empty:
        return
    common = lower.index.intersection(upper.index)
    if len(common) == 0:
        return
    lower, upper = lower.loc[common], upper.loc[common]
    fig.add_trace(go.Scatter(x=list(upper.index), y=upper.values, mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip", name=f"{name} upper"))
    fig.add_trace(go.Scatter(x=list(lower.index), y=lower.values, mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor=_rgba(color, 0.18), name=name,
                             hovertemplate="%{y:,.2f}<extra>" + name + "</extra>"))


def forecast_chart(history: Any, forecasts: Optional[Mapping[str, Any]] = None, ensemble: Any = None,
                   title: str = "Forecast", height: Optional[int] = None, history_points: Optional[int] = None,
                   yaxis_title: str = "Price", band_label: str = "Confidence interval",
                   labels: Optional[Mapping[str, str]] = None) -> go.Figure:
    """
    History plus one or more forecasts.

    Args:
        history: Historical values (Series, DataFrame with ``Close``, or forecast dict with ``history``).
        forecasts: ``{model_key: forecast result}`` drawn as dashed lines.
        ensemble: Main forecast drawn solid with its confidence band. When
            omitted and exactly one forecast is given, that one gets the band.
        history_points: Only show the last N historical points.
        labels: Display names for model keys.
    """
    try:
        labels = labels or {}
        hist = as_series(history.get("history") if isinstance(history, Mapping) else history)
        if history_points and len(hist) > history_points:
            hist = hist.iloc[-history_points:]
        forecasts = {k: v for k, v in (forecasts or {}).items() if not forecast_parts(v)[0].empty}
        if ensemble is None and len(forecasts) == 1:
            ensemble = next(iter(forecasts.values()))
            forecasts = {}
        main, lower, upper = forecast_parts(ensemble)
        if hist.empty and main.empty and not forecasts:
            return empty_figure("No forecast data to display", height, title)

        fig = go.Figure()
        _band(fig, lower, upper, ENSEMBLE_COLOR, band_label)
        if not hist.empty:
            fig.add_trace(go.Scatter(x=list(hist.index), y=hist.values, mode="lines", name="History",
                                     line={"color": HISTORY_COLOR, "width": 1.6}))
        anchor = (hist.index[-1], hist.iloc[-1]) if not hist.empty else None
        for i, (key, result) in enumerate(forecasts.items()):
            series, _, _ = forecast_parts(result)
            x, y = list(series.index), list(series.values)
            if anchor is not None:
                x, y = [anchor[0]] + x, [anchor[1]] + y
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=labels.get(key, str(key)),
                                     line={"color": SERIES_COLORS[i % len(SERIES_COLORS)], "width": 1.4,
                                           "dash": "dash"}))
        if not main.empty:
            x, y = list(main.index), list(main.values)
            if anchor is not None:
                x, y = [anchor[0]] + x, [anchor[1]] + y
            if forecasts:
                name = "Ensemble"
            elif isinstance(ensemble, Mapping):
                name = labels.get(str(ensemble.get("model", "")), "Forecast")
            else:
                name = "Forecast"
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                     line={"color": ENSEMBLE_COLOR, "width": 2.6}))
        if anchor is not None:
            fig.add_vline(x=anchor[0], line={"color": VIZ_SETTINGS["grid_color"], "dash": "dot", "width": 1})
        return apply_theme(fig, title, height, yaxis_title=yaxis_title)
    except Exception as exc:
        logger.exception("forecast_chart failed")
        return empty_figure(f"Chart unavailable: {exc}", height, title)


def evaluation_chart(evaluation: Optional[pd.DataFrame], metric: str = "mape", title: str = "Backtest error",
                     labels: Optional[Mapping[str, str]] = None, height: Optional[int] = None) -> go.Figure:
    """Bar chart of one error metric (``mae``, ``rmse`` or ``mape``) per model."""
    try:
        labels = labels or {}
        if evaluation is None or not isinstance(evaluation, pd.DataFrame) or evaluation.empty \
                or metric not in evaluation.columns or "model" not in evaluation.columns:
            return empty_figure("No backtest results", height or 320, title)
        frame = evaluation[["model", metric]].copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna()
        if frame.empty:
            return empty_figure("No backtest results", height or 320, title)
        frame = frame.sort_values(metric)
        names = [labels.get(str(m), str(m)) for m in frame["model"]]
        colors = [ENSEMBLE_COLOR] + [COLORS["info"]] * (len(frame) - 1)
        suffix = "%" if metric == "mape" else ""
        fig = go.Figure(go.Bar(x=names, y=frame[metric].values, marker_color=colors,
                               text=[f"{v:,.2f}{suffix}" for v in frame[metric].values], textposition="outside"))
        apply_theme(fig, title, height or 340, yaxis_title=metric.upper())
        fig.update_layout(hovermode="closest", showlegend=False)
        return fig
    except Exception as exc:
        logger.exception("evaluation_chart failed")
        return empty_figure(f"Chart unavailable: {exc}", height or 320, title)


def distribution_chart(percentiles: Optional[Mapping[Any, Any]], current_price: Optional[float] = None,
                       expected_price: Optional[float] = None, title: str = "Simulated price distribution",
                       height: Optional[int] = None) -> go.Figure:
    """Percentile ladder (5/25/50/75/95) of simulated prices against the current price."""
    try:
        points: Dict[float, float] = {}
        for key, value in (percentiles or {}).items():
            try:
                pct, price = float(key), float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(pct) and np.isfinite(price):
                points[pct] = price
        if not points:
            return empty_figure("Probability simulation unavailable", height or 320, title)
        ordered = sorted(points.items())
        fig = go.Figure()
        prices = [p for _, p in ordered]
        fig.add_trace(go.Scatter(x=prices, y=[0] * len(prices), mode="markers+text",
                                 marker={"size": 14, "color": COLORS["info"]},
                                 text=[f"P{int(k) if float(k).is_integer() else k}<br>${v:,.2f}" for k, v in ordered],
                                 textposition="top center", name="Percentiles", hoverinfo="skip"))
        if len(prices) >= 2:
            fig.add_shape(type="line", x0=prices[0], x1=prices[-1], y0=0, y1=0,
                          line={"color": COLORS["info"], "width": 2})
        inner = [p for k, p in ordered if 25 <= k <= 75]
        if len(inner) >= 2:
            fig.add_shape(type="rect", x0=min(inner), x1=max(inner), y0=-0.15, y1=0.15,
                          fillcolor=_rgba(COLORS["info"], 0.25), line={"width": 0})
        for value, label, color in ((current_price, "Current", HISTORY_COLOR),
                                    (expected_price, "Expected", ENSEMBLE_COLOR)):
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and np.isfinite(value):
                fig.add_vline(x=value, line={"color": color, "dash": "dash", "width": 1.5},
                              annotation_text=f"{label} ${value:,.2f}", annotation_position="bottom")
        apply_theme(fig, title, height or 300, xaxis_title="Price")
        fig.update_yaxes(visible=False, range=[-1, 1])
        fig.update_layout(hovermode=False, showlegend=False)
        return fig
    except Exception as exc:
        logger.exception("distribution_chart failed")
        return empty_figure(f"Chart unavailable: {exc}", height or 320, title)


def weights_chart(weights: Optional[Mapping[str, Any]], labels: Optional[Mapping[str, str]] = None,
                  title: str = "Ensemble weights", height: Optional[int] = None) -> go.Figure:
    """Horizontal bar chart of ensemble weights (fractions)."""
    labels = labels or {}
    clean = {}
    for key, value in (weights or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            clean[labels.get(str(key), str(key))] = value
    if not clean:
        return empty_figure("No ensemble weights", height or 260, title)
    fig = go.Figure(go.Bar(x=[v * 100 for v in clean.values()], y=list(clean.keys()), orientation="h",
                           marker_color=ENSEMBLE_COLOR, text=[f"{v:.0%}" for v in clean.values()],
                           textposition="auto"))
    apply_theme(fig, title, height or max(220, 60 * len(clean)), xaxis_title="Weight (%)")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def scenario_paths_chart(paths: Mapping[str, Any], history: Any = None, title: str = "Scenario projections",
                         yaxis_title: str = "", height: Optional[int] = None,
                         colors: Optional[Mapping[str, str]] = None) -> go.Figure:
    """Historical values followed by one projected path per scenario."""
    try:
        colors = colors or {}
        hist = as_series(history)
        cleaned = {name: as_series(path) for name, path in (paths or {}).items()}
        cleaned = {k: v for k, v in cleaned.items() if not v.empty}
        if hist.empty and not cleaned:
            return empty_figure("No scenario data", height, title)
        fig = go.Figure()
        if not hist.empty:
            fig.add_trace(go.Scatter(x=list(hist.index), y=hist.values, mode="lines+markers", name="Historical",
                                     line={"color": HISTORY_COLOR, "width": 2}))
        for i, (name, path) in enumerate(cleaned.items()):
            fig.add_trace(go.Scatter(x=list(path.index), y=path.values, mode="lines+markers", name=name,
                                     line={"color": colors.get(name, SERIES_COLORS[i % len(SERIES_COLORS)]),
                                           "width": 2, "dash": "dash"}))
        return apply_theme(fig, title, height, yaxis_title=yaxis_title)
    except Exception as exc:
        logger.exception("scenario_paths_chart failed")
        return empty_figure(f"Chart unavailable: {exc}", height, title)


def scenario_price_chart(prices: Mapping[str, Any], current_price: Optional[float] = None,
                         weighted_target: Optional[float] = None, title: str = "Implied share price",
                         height: Optional[int] = None, colors: Optional[Mapping[str, str]] = None) -> go.Figure:
    """Bar chart of the implied price per scenario with current price and weighted target lines."""
    colors = colors or {}
    clean: Dict[str, float] = {}
    for name, value in (prices or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            clean[name] = value
    if not clean:
        return empty_figure("Implied prices unavailable", height or 340, title)
    fig = go.Figure(go.Bar(x=list(clean.keys()), y=list(clean.values()),
                           marker_color=[colors.get(k, SERIES_COLORS[i % len(SERIES_COLORS)])
                                         for i, k in enumerate(clean)],
                           text=[f"${v:,.2f}" for v in clean.values()], textposition="outside"))
    for value, label, color in ((current_price, "Current", HISTORY_COLOR),
                                (weighted_target, "Probability-weighted", ENSEMBLE_COLOR)):
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and np.isfinite(value):
            fig.add_hline(y=value, line={"color": color, "dash": "dash", "width": 1.5},
                          annotation_text=f"{label} ${value:,.2f}", annotation_position="top left")
    apply_theme(fig, title, height or 380, yaxis_title="Price per share")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def var_forecast_chart(history: Optional[pd.DataFrame], result: Optional[Mapping[str, Any]],
                       columns: Optional[Sequence[str]] = None, title: str = "VAR forecast",
                       height: Optional[int] = None, labels: Optional[Mapping[str, str]] = None) -> go.Figure:
    """Multi-series history + VAR forecasts (``{'forecasts': {col: Series}, 'lower', 'upper'}``)."""
    try:
        labels = labels or {}
        forecasts = (result or {}).get("forecasts") or {}
        if not forecasts:
            return empty_figure("VAR forecast unavailable", height, title)
        fig = go.Figure()
        for i, column in enumerate(columns or list(forecasts)):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            if isinstance(history, pd.DataFrame) and column in history.columns:
                hist = as_series(history[column])
                fig.add_trace(go.Scatter(x=list(hist.index), y=hist.values, mode="lines",
                                         name=f"{labels.get(column, column)} (history)",
                                         line={"color": color, "width": 1.4}))
            fc = as_series(forecasts.get(column))
            if not fc.empty:
                fig.add_trace(go.Scatter(x=list(fc.index), y=fc.values, mode="lines",
                                         name=f"{labels.get(column, column)} (VAR)",
                                         line={"color": color, "width": 2.2, "dash": "dash"}))
        return apply_theme(fig, title, height)
    except Exception as exc:
        logger.exception("var_forecast_chart failed")
        return empty_figure(f"Chart unavailable: {exc}", height, title)
