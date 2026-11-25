"""
Plotly chart builders for the company analysis pages.

Every builder returns a ``go.Figure`` styled with the application's dark theme
(``config.COLORS`` / ``config.VIZ_SETTINGS``). Builders never raise for missing
data: empty, all-NaN or otherwise unusable input yields an empty figure that
displays a short message instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    MUTED,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SERIES_COLORS,
    compare_to_benchmark,
    fmt_pct,
    format_ratio_value,
    is_number,
    ratio_meta,
    series_to_labels,
)

logger = logging.getLogger("company_analysis.charts")

DEFAULT_HEIGHT = 400


# ---------------------------------------------------------------------------
# Theme and rendering
# ---------------------------------------------------------------------------
def style_figure(fig: go.Figure, title: Optional[str] = None, height: int = DEFAULT_HEIGHT,
                 xaxis_title: Optional[str] = None, yaxis_title: Optional[str] = None,
                 yaxis_tickformat: Optional[str] = None, showlegend: bool = True,
                 hovermode: Optional[str] = "x unified") -> go.Figure:
    """Apply the application's dark theme to ``fig`` and return it."""
    layout: Dict[str, Any] = dict(
        template="plotly_dark",
        paper_bgcolor=VIZ_SETTINGS["background"],
        plot_bgcolor=VIZ_SETTINGS["background"],
        font=dict(family=VIZ_SETTINGS["font_family"], color=VIZ_SETTINGS["text_color"],
                  size=VIZ_SETTINGS["axis_font_size"]),
        height=height,
        margin=dict(l=60, r=30, t=70 if title else 40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=showlegend,
    )
    if hovermode:
        layout["hovermode"] = hovermode
    if title:
        layout["title"] = dict(text=title, font=dict(size=VIZ_SETTINGS["title_font_size"]))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=VIZ_SETTINGS["grid_color"], zerolinecolor=VIZ_SETTINGS["grid_color"],
                     title_text=xaxis_title)
    fig.update_yaxes(gridcolor=VIZ_SETTINGS["grid_color"], zerolinecolor=VIZ_SETTINGS["grid_color"],
                     title_text=yaxis_title)
    if yaxis_tickformat:
        fig.update_yaxes(tickformat=yaxis_tickformat)
    return fig


def empty_figure(message: str = "No data available", title: Optional[str] = None,
                 height: int = 300) -> go.Figure:
    """Placeholder figure showing ``message``."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=15, color=MUTED))
    style_figure(fig, title=title, height=height, showlegend=False, hovermode=None)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def is_empty_figure(fig: go.Figure) -> bool:
    """True for figures produced by :func:`empty_figure`."""
    return len(fig.data) == 0


def show_chart(fig: go.Figure, key: str) -> None:
    """Render a figure with the page theme (Streamlit's theme disabled) and a unique key."""
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def money_axis_scale(values: Sequence[Any]) -> Tuple[float, str]:
    """Divisor and unit suffix (e.g. ``(1e9, "B")``) that keep money axes readable."""
    finite = [abs(float(v)) for v in values if is_number(v)]
    peak = max(finite) if finite else 0.0
    for divisor, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if peak >= divisor:
            return divisor, unit
    return 1.0, ""


def _clean(series: Optional[pd.Series]) -> pd.Series:
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def _usable(series_map: Mapping[str, pd.Series]) -> Dict[str, pd.Series]:
    return {name: s for name, s in ((n, _clean(s)) for n, s in series_map.items()) if not s.empty}


def _money_label(unit: str, currency: str = "USD") -> str:
    return f"{currency} ({unit})" if unit else currency


# ---------------------------------------------------------------------------
# Price & performance
# ---------------------------------------------------------------------------
def price_chart(prices: pd.DataFrame, title: str, chart_type: str = "Candlestick",
                moving_averages: Optional[Mapping[int, pd.Series]] = None, show_volume: bool = True,
                currency: str = "USD") -> go.Figure:
    """Candlestick/line price chart with optional moving averages and a volume panel."""
    if not isinstance(prices, pd.DataFrame) or prices.empty or "Close" not in prices.columns \
            or prices["Close"].notna().sum() < 2:
        return empty_figure("No price history available", title)
    has_ohlc = all(c in prices.columns for c in ("Open", "High", "Low"))
    show_volume = bool(show_volume and "Volume" in prices.columns and prices["Volume"].notna().any())
    fig = make_subplots(rows=2 if show_volume else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if show_volume else [1.0])
    if chart_type == "Candlestick" and has_ohlc:
        fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"],
                                     close=prices["Close"], name="Price",
                                     increasing_line_color=COLORS["success"],
                                     decreasing_line_color=COLORS["danger"]), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close", mode="lines",
                                 line=dict(color=COLORS["info"], width=2)), row=1, col=1)
    ma_colors = [COLORS["warning"], COLORS["secondary"], COLORS["accent"], COLORS["primary"]]
    for i, (period, series) in enumerate((moving_averages or {}).items()):
        segment = series.reindex(prices.index) if isinstance(series, pd.Series) else None
        if segment is not None and segment.notna().any():
            fig.add_trace(go.Scatter(x=prices.index, y=segment, name=f"{period}-day MA", mode="lines",
                                     line=dict(width=1.4, color=ma_colors[i % len(ma_colors)])), row=1, col=1)
    if show_volume:
        opens = prices["Open"] if "Open" in prices.columns else prices["Close"]
        colors = np.where(prices["Close"] >= opens, COLORS["success"], COLORS["danger"])
        fig.add_trace(go.Bar(x=prices.index, y=prices["Volume"], name="Volume", opacity=0.7,
                             marker_color=colors, showlegend=False), row=2, col=1)
    style_figure(fig, title=title, height=560 if show_volume else 460)
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume (shares)", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def rebased_performance_chart(series_map: Mapping[str, pd.Series], title: str,
                              colors: Optional[Sequence[str]] = None) -> go.Figure:
    """Growth of 100 invested for each series (rebased at its first value)."""
    usable = _usable(series_map)
    usable = {name: s for name, s in usable.items() if s.iloc[0] > 0 and len(s) > 1}
    if not usable:
        return empty_figure("No overlapping price history", title)
    palette = list(colors or [COLORS["info"], COLORS["warning"], COLORS["secondary"], COLORS["primary"]])
    fig = go.Figure()
    for i, (name, series) in enumerate(usable.items()):
        fig.add_trace(go.Scatter(x=series.index, y=series / series.iloc[0] * 100, name=name, mode="lines",
                                 line=dict(color=palette[i % len(palette)], width=2),
                                 hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.add_hline(y=100, line_dash="dot", line_color="#666666")
    return style_figure(fig, title=title, height=420, yaxis_title="Value of 100 invested")


# ---------------------------------------------------------------------------
# Financial statement charts
# ---------------------------------------------------------------------------
def financial_bars_chart(bars: Mapping[str, pd.Series], title: str,
                         line: Optional[Tuple[str, pd.Series]] = None, line_tickformat: str = ".0%",
                         line_axis_title: str = "", colors: Optional[Sequence[str]] = None,
                         barmode: str = "group", currency: str = "USD", height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Grouped money bars per period, optionally with a line (e.g. a margin) on a secondary axis."""
    usable = _usable(bars)
    line_series = _clean(line[1]) if line else pd.Series(dtype=float)
    if not usable and line_series.empty:
        return empty_figure("No data available for this chart", title)
    divisor, unit = money_axis_scale([v for s in usable.values() for v in s.values])
    palette = list(colors or SERIES_COLORS)
    fig = make_subplots(specs=[[{"secondary_y": not line_series.empty}]])
    for i, (name, series) in enumerate(usable.items()):
        fig.add_trace(go.Bar(x=series_to_labels(series), y=series.values / divisor, name=name,
                             marker_color=palette[i % len(palette)],
                             hovertemplate=f"{name}: %{{y:,.2f}}{unit}<extra></extra>"),
                      secondary_y=False)
    if not line_series.empty:
        fig.add_trace(go.Scatter(x=series_to_labels(line_series), y=line_series.values, name=line[0],
                                 mode="lines+markers", line=dict(color=COLORS["warning"], width=2),
                                 hovertemplate=f"{line[0]}: %{{y:{line_tickformat}}}<extra></extra>"),
                      secondary_y=True)
        fig.update_yaxes(title_text=line_axis_title or line[0], tickformat=line_tickformat, secondary_y=True,
                         showgrid=False)
    style_figure(fig, title=title, height=height)
    fig.update_layout(barmode=barmode)
    fig.update_yaxes(title_text=_money_label(unit, currency), secondary_y=False)
    fig.update_xaxes(type="category")
    return fig


def line_trend_chart(series_map: Mapping[str, pd.Series], title: str, yaxis_title: Optional[str] = None,
                     tickformat: Optional[str] = None, reference_lines: Sequence[Tuple[float, str]] = (),
                     height: int = 380) -> go.Figure:
    """Lines per period for ratios / margins (``tickformat='.0%'`` for fractions)."""
    usable = _usable(series_map)
    if not usable:
        return empty_figure("No data available for this chart", title)
    fig = go.Figure()
    hover_fmt = tickformat or ",.2f"
    for i, (name, series) in enumerate(usable.items()):
        fig.add_trace(go.Scatter(x=series_to_labels(series), y=series.values, name=name, mode="lines+markers",
                                 line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
                                 hovertemplate=f"{name}: %{{y:{hover_fmt}}}<extra></extra>"))
    for value, label in reference_lines:
        fig.add_hline(y=value, line_dash="dot", line_color="#777777", annotation_text=label,
                      annotation_position="top left")
    style_figure(fig, title=title, height=height, yaxis_title=yaxis_title, yaxis_tickformat=tickformat)
    fig.update_xaxes(type="category")
    return fig


def balance_structure_chart(assets: Mapping[str, pd.Series], funding: Mapping[str, pd.Series], title: str,
                            currency: str = "USD") -> go.Figure:
    """Side-by-side stacked bars per period: asset composition vs liabilities & equity."""
    assets_u, funding_u = _usable(assets), _usable(funding)
    if not assets_u and not funding_u:
        return empty_figure("No balance sheet data available", title)
    divisor, unit = money_axis_scale([v for s in list(assets_u.values()) + list(funding_u.values())
                                      for v in s.values])
    fig = go.Figure()
    asset_colors = [COLORS["info"], COLORS["success"], "#5c8fd6"]
    funding_colors = [COLORS["accent"], COLORS["secondary"], COLORS["primary"]]
    for group, series_map, palette in (("Assets", assets_u, asset_colors),
                                       ("Liabilities & Equity", funding_u, funding_colors)):
        for i, (name, series) in enumerate(series_map.items()):
            fig.add_trace(go.Bar(x=series_to_labels(series), y=series.values / divisor, name=name,
                                 offsetgroup=group, legendgroup=group, marker_color=palette[i % len(palette)],
                                 hovertemplate=f"{group} — {name}: %{{y:,.2f}}{unit}<extra></extra>"))
    style_figure(fig, title=title, height=420, yaxis_title=_money_label(unit, currency))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(type="category")
    return fig


def waterfall_chart(steps: Sequence[Tuple[str, Optional[float], str]], title: str,
                    currency: str = "USD") -> go.Figure:
    """
    Waterfall from ``[(label, value, measure)]`` with measure ``absolute``,
    ``relative`` or ``total``. Steps with missing values are dropped.
    """
    steps = [(label, float(value), measure) for label, value, measure in steps if is_number(value)]
    if len(steps) < 2:
        return empty_figure("Not enough line items for a waterfall", title)
    divisor, unit = money_axis_scale([v for _, v, _ in steps])
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], y=[s[1] / divisor for s in steps], measure=[s[2] for s in steps],
        text=[f"{s[1] / divisor:,.1f}{unit}" for s in steps], textposition="outside",
        increasing=dict(marker=dict(color=COLORS["primary"])),
        decreasing=dict(marker=dict(color=COLORS["accent"])),
        totals=dict(marker=dict(color=COLORS["info"])),
        connector=dict(line=dict(color="#555555")),
        hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra></extra>",
    ))
    return style_figure(fig, title=title, height=420, yaxis_title=_money_label(unit, currency),
                        showlegend=False, hovermode="closest")


def growth_bar_chart(series_map: Mapping[str, pd.Series], title: str) -> go.Figure:
    """Grouped bars of period-over-period growth rates (fractions)."""
    usable = _usable(series_map)
    if not usable:
        return empty_figure("Not enough periods to compute growth", title)
    fig = go.Figure()
    for i, (name, series) in enumerate(usable.items()):
        fig.add_trace(go.Bar(x=series_to_labels(series), y=series.values, name=name,
                             marker_color=SERIES_COLORS[i % len(SERIES_COLORS)],
                             hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>"))
    style_figure(fig, title=title, height=400, yaxis_title="Growth", yaxis_tickformat=".0%")
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    return fig


def signed_bar_chart(labels: Sequence[str], values: Sequence[Optional[float]], title: str,
                     tickformat: str = ".0%", axis_title: str = "", height: Optional[int] = None) -> go.Figure:
    """Horizontal bars coloured by sign (e.g. CAGR by metric)."""
    pairs = [(label, float(v)) for label, v in zip(labels, values) if is_number(v)]
    if not pairs:
        return empty_figure("No data available for this chart", title)
    fig = go.Figure(go.Bar(
        x=[v for _, v in pairs], y=[label for label, _ in pairs], orientation="h",
        marker_color=[POSITIVE if v >= 0 else NEGATIVE for _, v in pairs],
        text=[f"{v:{tickformat}}" for _, v in pairs], textposition="outside",
        hovertemplate="%{y}: %{x:" + tickformat + "}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=VIZ_SETTINGS["text_color"], line_width=1)
    style_figure(fig, title=title, height=height or max(260, 42 * len(pairs) + 120), xaxis_title=axis_title,
                 showlegend=False, hovermode="closest")
    fig.update_xaxes(tickformat=tickformat)
    fig.update_yaxes(autorange="reversed")
    return fig


def benchmark_bar_chart(ratios: Mapping[str, Mapping[str, float]], benchmarks: Mapping[str, Mapping[str, float]],
                        keys: Sequence[str], title: str) -> go.Figure:
    """Company premium/discount to each sector benchmark, coloured by favourability."""
    labels, diffs, colors, hover = [], [], [], []
    for key in keys:
        meta = ratio_meta(key)
        value = ratios.get(meta["category"], {}).get(key)
        bench = benchmarks.get(meta["category"], {}).get(key)
        diff, assessment, color = compare_to_benchmark(value, bench, meta.get("higher_is_better"))
        if diff is None:
            continue
        labels.append(meta["label"])
        diffs.append(diff)
        colors.append(color)
        hover.append(f"{meta['label']}: {format_ratio_value(key, value)} vs sector "
                     f"{format_ratio_value(key, bench)} ({assessment})")
    if not labels:
        return empty_figure("No sector benchmarks available", title)
    fig = go.Figure(go.Bar(x=diffs, y=labels, orientation="h", marker_color=colors,
                           text=[fmt_pct(d, signed=True) for d in diffs], textposition="outside",
                           hovertext=hover, hoverinfo="text"))
    fig.add_vline(x=0, line_color=VIZ_SETTINGS["text_color"], line_width=1)
    style_figure(fig, title=title, height=max(260, 44 * len(labels) + 120),
                 xaxis_title="Difference vs sector benchmark", showlegend=False, hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return fig


def ratio_comparison_chart(labels: Sequence[str], company: Sequence[Optional[float]],
                           benchmark: Sequence[Optional[float]], title: str, tickformat: str = ",.2f",
                           company_name: str = "Company") -> go.Figure:
    """Grouped bars: company value vs sector benchmark for ratios of the same unit."""
    rows = [(lbl, c, b) for lbl, c, b in zip(labels, company, benchmark) if is_number(c) or is_number(b)]
    if not rows:
        return empty_figure("No ratios available", title)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1] if is_number(r[1]) else None for r in rows],
                         name=company_name, marker_color=COLORS["info"],
                         hovertemplate="%{x}: %{y:" + tickformat + "}<extra>" + company_name + "</extra>"))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[2] if is_number(r[2]) else None for r in rows],
                         name="Sector benchmark", marker_color=COLORS["warning"],
                         hovertemplate="%{x}: %{y:" + tickformat + "}<extra>Sector</extra>"))
    style_figure(fig, title=title, height=380, yaxis_tickformat=tickformat, hovermode="closest")
    fig.update_layout(barmode="group")
    return fig


# ---------------------------------------------------------------------------
# Valuation charts
# ---------------------------------------------------------------------------
def value_comparison_chart(labels: Sequence[str], values: Sequence[Optional[float]],
                           current_price: Optional[float], title: str,
                           colors: Optional[Sequence[str]] = None,
                           axis_title: str = "Value per share (USD)") -> go.Figure:
    """Per-share values by method as bars with the current price as a dashed reference line."""
    pairs = [(label, float(v), (colors[i] if colors else None))
             for i, (label, v) in enumerate(zip(labels, values)) if is_number(v)]
    if not pairs:
        return empty_figure("No per-share values available", title)
    price = float(current_price) if is_number(current_price) else None
    bar_colors = [c or (POSITIVE if price is None or v >= price else NEGATIVE) for _, v, c in pairs]
    fig = go.Figure(go.Bar(x=[p[0] for p in pairs], y=[p[1] for p in pairs], marker_color=bar_colors,
                           text=[f"${p[1]:,.2f}" for p in pairs], textposition="outside",
                           hovertemplate="%{x}: $%{y:,.2f}<extra></extra>"))
    if price is not None:
        fig.add_hline(y=price, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"Current price ${price:,.2f}", annotation_position="top left",
                      annotation_font_color=COLORS["warning"])
    style_figure(fig, title=title, height=420, yaxis_title=axis_title, showlegend=False, hovermode="closest")
    return fig


def dcf_cash_flow_chart(historical: pd.Series, forecast: Sequence[float], present_values: Sequence[float],
                        title: str, first_year: Optional[int] = None, currency: str = "USD") -> go.Figure:
    """Historical FCF, projected FCF and their present values."""
    hist = _clean(historical)
    forecast = [float(v) for v in forecast if is_number(v)]
    present_values = [float(v) for v in present_values if is_number(v)]
    if hist.empty and not forecast:
        return empty_figure("No free cash flow data available", title)
    divisor, unit = money_axis_scale(list(hist.values) + forecast)
    start = first_year or (pd.Timestamp(hist.index[-1]).year + 1 if not hist.empty and
                           isinstance(hist.index[-1], pd.Timestamp) else None)
    labels = [f"FY{start + i}E" if start else f"Year {i + 1}" for i in range(len(forecast))]
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Bar(x=series_to_labels(hist), y=hist.values / divisor, name="Historical FCF",
                             marker_color=COLORS["info"],
                             hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra>Historical</extra>"))
    if forecast:
        fig.add_trace(go.Bar(x=labels, y=np.array(forecast) / divisor, name="Projected FCF",
                             marker_color=COLORS["primary"],
                             hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra>Projected</extra>"))
    if present_values and len(present_values) == len(forecast):
        fig.add_trace(go.Scatter(x=labels, y=np.array(present_values) / divisor, name="Present value",
                                 mode="lines+markers", line=dict(color=COLORS["warning"], width=2),
                                 hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra>Present value</extra>"))
    style_figure(fig, title=title, height=400, yaxis_title=f"Free cash flow ({_money_label(unit, currency)})")
    fig.update_xaxes(type="category")
    return fig


def sensitivity_heatmap(z: Sequence[Sequence[Optional[float]]], x_values: Sequence[float],
                        y_values: Sequence[float], title: str, x_title: str, y_title: str,
                        current_price: Optional[float] = None, x_format: str = ".1%",
                        y_format: str = ".1%") -> go.Figure:
    """
    Value-per-share heatmap over two assumptions. Colours are centred on the
    current price (green above, red below) when it is known.
    """
    matrix = np.array([[float(v) if is_number(v) else np.nan for v in row] for row in z], dtype=float)
    if matrix.size == 0 or np.isnan(matrix).all():
        return empty_figure("Sensitivity grid could not be computed for these inputs", title)
    x_labels = [f"{x:{x_format}}" for x in x_values]
    y_labels = [f"{y:{y_format}}" for y in y_values]
    text = [[f"${v:,.2f}" if np.isfinite(v) else "n/a" for v in row] for row in matrix]
    price = float(current_price) if is_number(current_price) and current_price > 0 else None
    heat: Dict[str, Any] = dict(z=matrix, x=x_labels, y=y_labels, text=text, texttemplate="%{text}",
                                hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>Value: %{{text}}"
                                              "<extra></extra>",
                                colorbar=dict(title="USD / share"))
    if price is not None:
        spread = np.nanmax(np.abs(matrix - price)) or 1.0
        heat.update(zmin=price - spread, zmax=price + spread,
                    colorscale=[[0, COLORS["danger"]], [0.5, "#2b2b2b"], [1, COLORS["primary"]]])
    else:
        heat.update(colorscale="Viridis")
    fig = go.Figure(go.Heatmap(**heat))
    style_figure(fig, title=title, height=460, xaxis_title=x_title, yaxis_title=y_title, showlegend=False,
                 hovermode="closest")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def donut_chart(labels: Sequence[str], values: Sequence[Optional[float]], title: str,
                colors: Optional[Sequence[str]] = None) -> go.Figure:
    """Donut of positive shares (e.g. PV of forecast vs terminal value)."""
    pairs = [(label, float(v)) for label, v in zip(labels, values) if is_number(v) and float(v) > 0]
    if not pairs:
        return empty_figure("No positive values to display", title)
    fig = go.Figure(go.Pie(labels=[p[0] for p in pairs], values=[p[1] for p in pairs], hole=0.55,
                           marker=dict(colors=list(colors or SERIES_COLORS)), textinfo="label+percent",
                           hovertemplate="%{label}: %{percent}<extra></extra>"))
    return style_figure(fig, title=title, height=340, showlegend=False, hovermode=None)


# ---------------------------------------------------------------------------
# Risk charts
# ---------------------------------------------------------------------------
def gauge_chart(value: Optional[float], title: str, axis_range: Tuple[float, float],
                steps: Sequence[Tuple[float, float, str]], bar_color: Optional[str] = None,
                number_format: str = ".2f", height: int = 260, threshold: Optional[float] = None) -> go.Figure:
    """Gauge with coloured zones ``[(start, end, colour), ...]``."""
    if not is_number(value):
        return empty_figure("Not enough data", title, height=height)
    low, high = axis_range
    shown = float(np.clip(float(value), low, high))
    gauge: Dict[str, Any] = {
        "axis": {"range": [low, high], "tickcolor": VIZ_SETTINGS["text_color"]},
        "bar": {"color": bar_color or VIZ_SETTINGS["text_color"], "thickness": 0.3},
        "steps": [{"range": [s, e], "color": c} for s, e, c in steps],
        "bgcolor": "#1e1e1e",
    }
    if threshold is not None:
        gauge["threshold"] = {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": threshold}
    fig = go.Figure(go.Indicator(mode="gauge+number", value=shown, number={"valueformat": number_format},
                                 title={"text": title, "font": {"size": 15}}, gauge=gauge))
    style_figure(fig, height=height, showlegend=False, hovermode=None)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=20))
    if shown != float(value):
        fig.add_annotation(text=f"actual: {float(value):{number_format}}", x=0.5, y=0.0, xref="paper",
                           yref="paper", showarrow=False, font=dict(color=MUTED, size=12))
    return fig


def score_bar_chart(scores: Mapping[str, Optional[float]], title: str) -> go.Figure:
    """Horizontal 0-100 score bars coloured by strength."""
    pairs = [(name, float(v)) for name, v in scores.items() if is_number(v)]
    if not pairs:
        return empty_figure("No scores available", title)
    colors = [POSITIVE if v >= 65 else NEUTRAL if v >= 45 else NEGATIVE for _, v in pairs]
    fig = go.Figure(go.Bar(x=[v for _, v in pairs], y=[n for n, _ in pairs], orientation="h", marker_color=colors,
                           text=[f"{v:.0f}" for _, v in pairs], textposition="outside",
                           hovertemplate="%{y}: %{x:.0f}/100<extra></extra>"))
    style_figure(fig, title=title, height=max(240, 50 * len(pairs) + 110), xaxis_title="Score (0-100)",
                 showlegend=False, hovermode="closest")
    fig.update_xaxes(range=[0, 110])
    fig.update_yaxes(autorange="reversed")
    return fig


def returns_histogram(returns: pd.Series, title: str, markers: Optional[Mapping[str, float]] = None,
                      x_title: str = "Return") -> go.Figure:
    """Histogram of returns (fractions) with vertical VaR / CVaR markers."""
    data = _clean(returns)
    if len(data) < 10:
        return empty_figure("Not enough return observations", title)
    fig = go.Figure(go.Histogram(x=data.values, nbinsx=60, marker_color=COLORS["info"], opacity=0.8,
                                 name="Returns", hovertemplate="%{x:.2%}: %{y} obs<extra></extra>"))
    palette = [COLORS["warning"], COLORS["danger"], COLORS["secondary"], COLORS["accent"]]
    for i, (label, value) in enumerate((markers or {}).items()):
        if is_number(value):
            fig.add_vline(x=float(value), line_dash="dash", line_color=palette[i % len(palette)],
                          annotation_text=label, annotation_position="top left" if i % 2 else "top right",
                          annotation_font_color=palette[i % len(palette)])
    style_figure(fig, title=title, height=400, xaxis_title=x_title, yaxis_title="Frequency (observations)",
                 showlegend=False, hovermode="closest")
    fig.update_xaxes(tickformat=".1%")
    return fig


def drawdown_chart(drawdown: pd.Series, title: str) -> go.Figure:
    """Underwater chart: percentage below the running peak."""
    data = _clean(drawdown)
    if len(data) < 2:
        return empty_figure("Not enough price history", title)
    fig = go.Figure(go.Scatter(x=data.index, y=data.values, fill="tozeroy", mode="lines", name="Drawdown",
                               line=dict(color=COLORS["danger"], width=1.5),
                               hovertemplate="%{x|%Y-%m-%d}: %{y:.1%}<extra></extra>"))
    return style_figure(fig, title=title, height=320, yaxis_title="Drawdown from peak", yaxis_tickformat=".0%",
                        showlegend=False)


def time_series_chart(series_map: Mapping[str, pd.Series], title: str, yaxis_title: Optional[str] = None,
                      tickformat: Optional[str] = None, height: int = 340) -> go.Figure:
    """Lines over a DatetimeIndex (e.g. rolling volatility)."""
    usable = _usable(series_map)
    if not usable:
        return empty_figure("Not enough data", title)
    fig = go.Figure()
    for i, (name, series) in enumerate(usable.items()):
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines",
                                 line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=1.8)))
    return style_figure(fig, title=title, height=height, yaxis_title=yaxis_title, yaxis_tickformat=tickformat)


def scenario_chart(projections: Mapping[str, Tuple[Sequence[str], Sequence[float], str]], title: str,
                   yaxis_title: str, money: bool = True, tickformat: Optional[str] = None,
                   currency: str = "USD") -> go.Figure:
    """Lines per scenario: ``{name: (x labels, values, colour)}``."""
    usable = {name: (x, [float(v) for v in y], c) for name, (x, y, c) in projections.items()
              if len(y) and all(is_number(v) for v in y)}
    if not usable:
        return empty_figure("No projections available", title)
    divisor, unit = money_axis_scale([v for _, y, _ in usable.values() for v in y]) if money else (1.0, "")
    fig = go.Figure()
    for name, (x, y, color) in usable.items():
        fig.add_trace(go.Scatter(x=list(x), y=np.array(y) / divisor, name=name, mode="lines+markers",
                                 line=dict(color=color, width=2)))
    axis = f"{yaxis_title} ({_money_label(unit, currency)})" if money else yaxis_title
    style_figure(fig, title=title, height=360, yaxis_title=axis, yaxis_tickformat=tickformat)
    fig.update_xaxes(type="category")
    return fig


__all__: List[str] = [
    "style_figure", "empty_figure", "is_empty_figure", "show_chart", "money_axis_scale", "price_chart",
    "rebased_performance_chart", "financial_bars_chart", "line_trend_chart", "balance_structure_chart",
    "waterfall_chart", "growth_bar_chart", "signed_bar_chart", "benchmark_bar_chart", "ratio_comparison_chart",
    "value_comparison_chart", "dcf_cash_flow_chart", "sensitivity_heatmap", "donut_chart", "gauge_chart",
    "score_bar_chart", "returns_histogram", "drawdown_chart", "time_series_chart", "scenario_chart",
]
