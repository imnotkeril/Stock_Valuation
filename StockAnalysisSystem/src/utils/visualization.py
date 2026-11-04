"""
Plotly chart factory shared by every page.

All public ``plot_*`` methods of :class:`FinancialVisualizer`:

* return a :class:`plotly.graph_objects.Figure` and never raise: missing,
  empty or non-numeric input yields :meth:`FinancialVisualizer._empty_chart`
  with an explanatory message, and unexpected errors are logged and rendered
  the same way (no tracebacks reach the UI);
* share one theme derived from ``config.VIZ_SETTINGS`` / ``config.COLORS``;
* format axes consistently: percentages with ``%``, money with ``$`` and a
  K/M/B/T scale suffix, multiples with ``x``.

Figures do not set a fixed width unless ``width`` is passed, so they stretch
to the Streamlit container.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.utils.financial_data import find_row, format_large_number, to_float

logger = logging.getLogger(__name__)

#: Gains / favourable values and losses / unfavourable values
POSITIVE_COLOR = COLORS["primary"]
NEGATIVE_COLOR = COLORS["accent"]
NEUTRAL_COLOR = "#9e9e9e"

#: Display labels for ratio keys used across the code base
RATIO_LABELS: Dict[str, str] = {
    "pe_ratio": "P/E", "forward_pe": "Forward P/E", "peg_ratio": "PEG", "ps_ratio": "P/S", "pb_ratio": "P/B",
    "ev_ebitda": "EV/EBITDA", "ev_revenue": "EV/Revenue", "fcf_yield": "FCF Yield", "dividend_yield": "Dividend Yield",
    "payout_ratio": "Payout Ratio", "gross_margin": "Gross Margin", "operating_margin": "Operating Margin",
    "net_margin": "Net Margin", "ebitda_margin": "EBITDA Margin", "roe": "ROE", "roa": "ROA", "roic": "ROIC",
    "current_ratio": "Current Ratio", "quick_ratio": "Quick Ratio", "cash_ratio": "Cash Ratio",
    "debt_to_equity": "Debt/Equity", "debt_equity": "Debt/Equity", "debt_to_assets": "Debt/Assets",
    "interest_coverage": "Interest Coverage", "asset_turnover": "Asset Turnover",
    "inventory_turnover": "Inventory Turnover", "receivables_turnover": "Receivables Turnover",
    "revenue_growth": "Revenue Growth", "net_income_growth": "Net Income Growth", "eps_growth": "EPS Growth",
    "rnd_to_revenue": "R&D/Revenue",
}

_PERCENT_HINTS = ("margin", "growth", "yield", "return", "roe", "roa", "roic", "payout", "upside", "volatility")



# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
def _finite(value: Any) -> Optional[float]:
    """Finite float or None (handles numpy scalars, strings, NaN, inf)."""
    return to_float(value)


def _label(key: Any) -> str:
    """Human readable label for a ratio / metric key."""
    text = str(key)
    return RATIO_LABELS.get(text, text.replace("_", " ").title() if "_" in text or text.islower() else text)


def _is_percent_key(key: Any) -> bool:
    text = str(key).lower()
    return any(hint in text for hint in _PERCENT_HINTS)


def _money_scale(max_abs: Optional[float]) -> Tuple[float, str]:
    """Divisor and suffix (T/B/M/K) that keep axis numbers short."""
    max_abs = abs(max_abs or 0.0)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if max_abs >= threshold:
            return threshold, suffix
    return 1.0, ""


def _is_fraction_scale(values: Iterable[Any]) -> bool:
    """True when rate values look like fractions (0.08) rather than percent points (8)."""
    finite = [abs(v) for v in (_finite(x) for x in values) if v is not None]
    return bool(finite) and max(finite) <= 1.0


def _datetime_index(index: pd.Index) -> pd.Index:
    """Return a tz-naive DatetimeIndex when ``index`` holds dates, else ``index`` unchanged."""
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_localize(None) if index.tz is not None else index
    if len(index) == 0 or pd.api.types.is_numeric_dtype(index):
        return index
    try:
        converted = pd.to_datetime(index, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return index
    if converted.isna().mean() > 0.5:
        return index
    return pd.DatetimeIndex(converted).tz_localize(None)


def _looks_like_dates(labels: Iterable[Any]) -> bool:
    labels = list(labels)
    if not labels:
        return False
    if isinstance(labels[0], (pd.Timestamp, np.datetime64)) or hasattr(labels[0], "year"):
        return True
    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in labels):
        return False
    try:
        parsed = pd.to_datetime(pd.Index([str(x) for x in labels]), errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return False
    return parsed.notna().mean() > 0.5


def _period_labels(labels: Sequence[Any]) -> List[str]:
    """Readable period labels: years when they are unique, ISO dates otherwise."""
    parsed = pd.to_datetime(pd.Index([str(x) for x in labels]), errors="coerce", format="mixed")
    if parsed.notna().all():
        years = [str(ts.year) for ts in parsed]
        if len(set(years)) == len(years):
            return years
        return [ts.strftime("%Y-%m-%d") for ts in parsed]
    return [str(x) for x in labels]


def _numeric_series(values: Any) -> pd.Series:
    """Numeric Series without NaN/inf from a Series, list, dict or scalar."""
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0] if values.shape[1] else pd.Series(dtype=float)
    if isinstance(values, Mapping):
        values = pd.Series(values)
    if not isinstance(values, pd.Series):
        values = pd.Series(np.atleast_1d(np.asarray(values, dtype=object)))
    series = pd.to_numeric(values, errors="coerce").astype(float)
    return series[np.isfinite(series.to_numpy())]


def _price_close(data: Any) -> pd.Series:
    """Close price series (tz-naive, sorted) from a price DataFrame or Series."""
    if isinstance(data, pd.DataFrame):
        column = next((c for c in data.columns if str(c).lower() == "close"), None)
        if column is None:
            column = next((c for c in data.columns if str(c).lower() in ("adj close", "adj_close", "price")), None)
        if column is None:
            return pd.Series(dtype=float)
        data = data[column]
    if not isinstance(data, pd.Series):
        return pd.Series(dtype=float)
    series = pd.to_numeric(data, errors="coerce").astype(float)
    series.index = _datetime_index(series.index)
    series = series[np.isfinite(series.to_numpy())]
    if isinstance(series.index, pd.DatetimeIndex):
        series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _safe_int(value: Any, default: Optional[int]) -> Optional[int]:
    """Positive int or ``default`` (guards against e.g. a string passed positionally as height)."""
    if isinstance(value, bool):
        return default
    number = _finite(value)
    return int(number) if number and number > 0 else default


def _safe_chart(label: str) -> Callable:
    """Decorator: log unexpected errors and return an empty chart instead of raising."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: "FinancialVisualizer", *args, **kwargs) -> go.Figure:
            try:
                figure = func(self, *args, **kwargs)
            except Exception:  # never surface tracebacks in the UI
                logger.exception("Error creating %s chart", label)
                return self._empty_chart(f"Could not create the {label} chart")
            if not isinstance(figure, go.Figure):
                return self._empty_chart(f"No data available for the {label} chart")
            return figure

        return wrapper

    return decorator


class FinancialVisualizer:
    """
    Plotly visualisations for financial data (price charts, statement trends,
    ratio comparisons, valuation and screening charts).
    """

    def __init__(self, theme: str = "dark"):
        """
        Args:
            theme: ``"dark"`` (default, from ``config.VIZ_SETTINGS``) or ``"light"``.
        """
        self.theme = theme if theme in ("dark", "light") else "dark"
        self.colors = COLORS
        self.settings = VIZ_SETTINGS

        if self.theme == "dark":
            self.bg_color = VIZ_SETTINGS.get("background", "#121212")
            self.grid_color = VIZ_SETTINGS.get("grid_color", "#333333")
            self.text_color = VIZ_SETTINGS.get("text_color", "#e0e0e0")
            self.template = "plotly_dark"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#e0e0e0"
            self.text_color = "#333333"
            self.template = "plotly_white"

        self.palette: List[str] = list(dict.fromkeys(
            [COLORS["primary"], COLORS["info"], COLORS["secondary"], COLORS["warning"], COLORS["success"],
             COLORS["accent"], COLORS["danger"], *COLORS.get("sectors", {}).values()]
        ))

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------
    def _series_colors(self, keys: Sequence[Any], primary: Optional[Any] = None) -> Dict[Any, str]:
        """Distinct colour per key; ``primary`` gets the highlight colour."""
        others = [c for c in self.palette if c != COLORS["primary"]] if primary is not None else self.palette
        colors: Dict[Any, str] = {}
        index = 0
        for key in keys:
            if primary is not None and key == primary:
                colors[key] = COLORS["primary"]
            else:
                colors[key] = others[index % len(others)]
                index += 1
        return colors

    def _apply_layout(self, fig: go.Figure, title: Optional[str] = None, height: Any = None,
                      width: Any = None, default_height: Optional[int] = None, **layout: Any) -> go.Figure:
        """Apply the shared theme, title and size to ``fig``."""
        fig.update_layout(
            template=self.template,
            title=dict(text=title, font=dict(size=self.settings.get("title_font_size", 18))) if title else None,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color, family=self.settings.get("font_family", "Arial, sans-serif"),
                      size=self.settings.get("axis_font_size", 12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=30, t=80 if title else 40, b=50),
            height=_safe_int(height, default_height or self.settings.get("height", 600)),
            hoverlabel=dict(namelength=-1),
        )
        width = _safe_int(width, None)
        if width:
            fig.update_layout(width=width)
        if layout:
            fig.update_layout(**layout)
        fig.update_xaxes(gridcolor=self.grid_color, zerolinecolor=self.grid_color, linecolor=self.grid_color)
        fig.update_yaxes(gridcolor=self.grid_color, zerolinecolor=self.grid_color, linecolor=self.grid_color)
        return fig

    def _empty_chart(self, message: str, height: Optional[int] = None) -> go.Figure:
        """Placeholder figure with a centred ``message`` (``layout.meta['empty_chart']`` is True)."""
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=message, showarrow=False,
                           font=dict(size=14, color=self.text_color))
        fig.update_layout(
            template=self.template,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color, family=self.settings.get("font_family", "Arial, sans-serif")),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=_safe_int(height, 400),
            margin=dict(l=20, r=20, t=40, b=20),
            meta={"empty_chart": True, "message": message},
        )
        return fig

    def _get_health_score_color(self, score: float) -> str:
        """Colour for a 0-100 health score."""
        score = _finite(score) or 0.0
        if score >= 75:
            return self.colors["success"]
        if score >= 60:
            return "#9acd32"
        if score >= 45:
            return self.colors["warning"]
        if score >= 30:
            return "#ff8c69"
        return self.colors["danger"]

    def _money_axis(self, fig: go.Figure, values: Iterable[Any], axis: str = "y", prefix: str = "$",
                    **selector: Any) -> Tuple[float, str]:
        """Choose a K/M/B/T scale for ``values`` and label the axis; returns (divisor, suffix)."""
        finite = [abs(v) for v in (_finite(x) for x in values) if v is not None]
        divisor, suffix = _money_scale(max(finite) if finite else 0)
        update = fig.update_yaxes if axis == "y" else fig.update_xaxes
        update(tickprefix=prefix, ticksuffix=suffix, tickformat=",.2~f", **selector)
        return divisor, suffix

    # ------------------------------------------------------------------
    # Price charts
    # ------------------------------------------------------------------
    @_safe_chart("price")
    def plot_stock_price(self,
                         price_data: pd.DataFrame,
                         ticker: str,
                         company_name: Optional[str] = None,
                         ma_periods: Sequence[int] = (50, 200),
                         volume: bool = True,
                         height: Optional[int] = None,
                         width: Optional[int] = None,
                         chart_type: str = "candlestick",
                         title: Optional[str] = None) -> go.Figure:
        """
        Price chart with moving averages and an optional volume subplot.

        Args:
            price_data: OHLCV DataFrame indexed by date (tz-naive or tz-aware).
            ticker: Ticker symbol.
            company_name: Name used in the title.
            ma_periods: Moving-average windows (in rows / trading days).
            volume: Show a volume subplot when volume data is available.
            height: Figure height in pixels.
            width: Figure width in pixels (default: stretch to container).
            chart_type: ``"candlestick"``, ``"ohlc"``, ``"line"`` or ``"area"``.
                Falls back to a line chart when OHLC columns are missing.
            title: Custom title.
        """
        if not isinstance(price_data, pd.DataFrame) or price_data.empty:
            return self._empty_chart("No price data available", height)

        columns = {str(c).lower(): c for c in price_data.columns}
        df = pd.DataFrame(index=_datetime_index(price_data.index))
        for name in ("Open", "High", "Low", "Close", "Volume"):
            source = columns.get(name.lower())
            if source is None and name == "Close":
                source = columns.get("adj close")
            df[name] = pd.to_numeric(price_data[source], errors="coerce").to_numpy() if source is not None else np.nan
        df = df[np.isfinite(df["Close"].to_numpy())]
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.empty:
            return self._empty_chart("No valid closing prices available", height)

        has_ohlc = all(df[c].notna().any() for c in ("Open", "High", "Low"))
        kind = str(chart_type or "candlestick").lower()
        if kind in ("candlestick", "ohlc") and not has_ohlc:
            kind = "line"
        show_volume = bool(volume) and df["Volume"].notna().any() and float(df["Volume"].fillna(0).sum()) > 0

        fig = make_subplots(rows=2 if show_volume else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.75, 0.25] if show_volume else [1.0])

        if kind == "candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                         close=df["Close"], name="Price", increasing_line_color=POSITIVE_COLOR,
                                         decreasing_line_color=NEGATIVE_COLOR), row=1, col=1)
        elif kind == "ohlc":
            fig.add_trace(go.Ohlc(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                  name="Price", increasing_line_color=POSITIVE_COLOR,
                                  decreasing_line_color=NEGATIVE_COLOR), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines",
                                     line=dict(color=COLORS["info"], width=2),
                                     fill="tozeroy" if kind == "area" else None,
                                     hovertemplate="$%{y:,.2f}<extra>Close</extra>"), row=1, col=1)

        ma_colors = [COLORS["warning"], COLORS["secondary"], COLORS["success"], COLORS["danger"]]
        for i, period in enumerate(p for p in (ma_periods or []) if _safe_int(p, None)):
            period = int(period)
            if len(df) >= period:
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(window=period).mean(),
                                         name=f"{period}-day MA", mode="lines",
                                         line=dict(width=1.5, color=ma_colors[i % len(ma_colors)]),
                                         hovertemplate="$%{y:,.2f}<extra>" + f"{period}-day MA</extra>"),
                              row=1, col=1)

        if show_volume:
            reference = df["Open"].where(df["Open"].notna(), df["Close"].shift(1))
            bar_colors = np.where(df["Close"] >= reference.fillna(df["Close"]), POSITIVE_COLOR, NEGATIVE_COLOR)
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=bar_colors, opacity=0.6,
                                 showlegend=False, hovertemplate="%{y:,.0f}<extra>Volume</extra>"),
                          row=2, col=1)
            fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)

        display_name = company_name or ticker
        default_title = f"{display_name} ({ticker}) - Price" if company_name and company_name != ticker \
            else f"{ticker} - Price"
        self._apply_layout(fig, title or default_title, height, width, hovermode="x unified")
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text="Price", tickprefix="$", tickformat=",.2f", row=1, col=1)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1 and (df.index.day_of_week < 5).all() \
                and (df.index.normalize() == df.index).all():
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        return fig

    @_safe_chart("price comparison")
    def plot_price_comparison(self,
                              price_series: Union[Mapping[str, Any], pd.DataFrame],
                              names: Optional[Mapping[str, str]] = None,
                              primary_ticker: Optional[str] = None,
                              title: str = "Price Performance",
                              height: Optional[int] = None,
                              width: Optional[int] = None,
                              normalize: bool = False,
                              yaxis_title: Optional[str] = None) -> go.Figure:
        """
        Line chart comparing several price / return series.

        Args:
            price_series: ``{ticker: Series}`` (or a DataFrame with one column
                per ticker, or price DataFrames with a ``Close`` column).
            names: ``{ticker: display name}``.
            primary_ticker: Series drawn thicker in the highlight colour.
            normalize: When True the series are converted to percentage change
                from their first valid value; otherwise they are assumed to
                already be percentage returns.
        """
        if isinstance(price_series, pd.DataFrame):
            price_series = {column: price_series[column] for column in price_series.columns}
        if not isinstance(price_series, Mapping) or not price_series:
            return self._empty_chart("No price data available for comparison", height)

        names = dict(names or {})
        cleaned: Dict[str, pd.Series] = {}
        for key, data in price_series.items():
            series = _price_close(data) if isinstance(data, pd.DataFrame) else _price_close(
                data if isinstance(data, pd.Series) else pd.Series(data, dtype=float))
            if series.empty:
                continue
            if normalize:
                base = series.iloc[0]
                if not base:
                    continue
                series = (series / base - 1) * 100
            cleaned[str(key)] = series
        if not cleaned:
            return self._empty_chart("No price data available for comparison", height)

        colors = self._series_colors(list(cleaned), primary_ticker)
        fig = go.Figure()
        for key, series in cleaned.items():
            is_primary = primary_ticker is not None and key == primary_ticker
            label = names.get(key, key)
            label = f"{label} ({key})" if label != key else key
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=label,
                                     line=dict(color=colors[key], width=3 if is_primary else 1.6),
                                     hovertemplate="%{y:.2f}%<extra>" + label + "</extra>"))
        fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOR, width=1, dash="dot"))
        self._apply_layout(fig, title, height, width, default_height=500, hovermode="x unified")
        fig.update_yaxes(title_text=yaxis_title or "Return (%)", ticksuffix="%")
        return fig

    @_safe_chart("price vs. fair value")
    def plot_price_vs_value(self,
                            price_data: pd.DataFrame,
                            ticker: str,
                            fair_value: Optional[float] = None,
                            company_name: Optional[str] = None,
                            height: Optional[int] = None,
                            width: Optional[int] = None,
                            margin_of_safety: Optional[float] = None,
                            title: Optional[str] = None) -> go.Figure:
        """
        Closing price history with horizontal fair value (and buy-below) lines.

        Args:
            margin_of_safety: Fraction (0.25 == 25 %); draws ``fair_value * (1 - mos)``.
        """
        close = _price_close(price_data)
        if close.empty:
            return self._empty_chart("No price data available", height)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Price",
                                 line=dict(color=COLORS["info"], width=2),
                                 hovertemplate="$%{y:,.2f}<extra>Price</extra>"))
        fair = _finite(fair_value)
        if fair is not None and fair > 0:
            fig.add_hline(y=fair, line=dict(color=POSITIVE_COLOR, width=2, dash="dash"),
                          annotation_text=f"Fair value ${fair:,.2f}", annotation_position="top left")
            mos = _finite(margin_of_safety)
            if mos is not None and 0 < mos < 1:
                buy_below = fair * (1 - mos)
                fig.add_hline(y=buy_below, line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
                              annotation_text=f"Buy below ${buy_below:,.2f} ({mos:.0%} MoS)",
                              annotation_position="bottom left")
        name = company_name or ticker
        self._apply_layout(fig, title or f"{name} ({ticker}) - Price vs. Fair Value" if name != ticker
                           else title or f"{ticker} - Price vs. Fair Value", height, width, default_height=500,
                           hovermode="x unified")
        fig.update_yaxes(title_text="Price", tickprefix="$", tickformat=",.2f")
        return fig

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------
    @_safe_chart("financial statement trend")
    def plot_financial_statement_trend(self,
                                       statement_data: pd.DataFrame,
                                       statement_type: str,
                                       metrics: List[str],
                                       height: Optional[int] = None,
                                       width: Optional[int] = None,
                                       title: Optional[str] = None) -> go.Figure:
        """
        Line chart of statement line items over time.

        Args:
            statement_data: yfinance layout (line items x periods) or its transpose.
            statement_type: ``"income"``, ``"balance"`` or ``"cash_flow"`` (title only).
            metrics: Line items to plot; canonical keys / aliases are accepted
                (see ``utils.financial_data.LINE_ITEM_ALIASES``).
        """
        if not isinstance(statement_data, pd.DataFrame) or statement_data.empty:
            return self._empty_chart(f"No {statement_type} statement data available", height)
        df = statement_data
        if _looks_like_dates(df.index) and not _looks_like_dates(df.columns):
            df = df.T

        columns = list(df.columns)
        if _looks_like_dates(columns):
            dates = pd.to_datetime(pd.Index([str(c) for c in columns]), errors="coerce", format="mixed")
            order = np.argsort(dates.values)
            columns = [columns[i] for i in order]
        x_labels = _period_labels(columns)

        rows: Dict[str, pd.Series] = {}
        for metric in metrics or []:
            label = find_row(df, metric)
            if label is None:
                continue
            row = df.loc[label]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            values = pd.to_numeric(row[columns], errors="coerce").astype(float)
            if values.notna().any():
                rows[str(metric) if str(metric) in df.index else str(label)] = values
        if not rows:
            return self._empty_chart(f"Specified metrics not found in the {statement_type} statement", height)

        fig = go.Figure()
        all_values = np.concatenate([row.to_numpy() for row in rows.values()])
        divisor, suffix = _money_scale(np.nanmax(np.abs(all_values)))
        colors = self._series_colors(list(rows))
        for name, values in rows.items():
            fig.add_trace(go.Scatter(x=x_labels, y=(values / divisor).to_numpy(), mode="lines+markers", name=name,
                                     line=dict(color=colors[name], width=2), connectgaps=False,
                                     hovertemplate="$%{y:,.2f}" + suffix + "<extra>" + name + "</extra>"))
        title_map = {"income": "Income Statement Trends", "balance": "Balance Sheet Trends",
                     "cash_flow": "Cash Flow Statement Trends", "cash": "Cash Flow Statement Trends"}
        self._apply_layout(fig, title or title_map.get(statement_type, "Financial Statement Trends"), height, width,
                           hovermode="x unified")
        self._money_axis(fig, all_values)
        fig.update_xaxes(title_text="Period", type="category")
        return fig

    # ------------------------------------------------------------------
    # Ratios and metrics
    # ------------------------------------------------------------------
    @_safe_chart("financial ratios")
    def plot_financial_ratios(self,
                              ratio_data: Dict[str, Dict[str, Any]],
                              category: Optional[str] = None,
                              benchmark_data: Optional[Dict[str, Dict[str, float]]] = None,
                              height: Optional[int] = None,
                              width: Optional[int] = None) -> go.Figure:
        """
        Company ratios (optionally vs. benchmarks) as grouped bars.

        Args:
            ratio_data: ``{category: {ratio: value}}`` or
                ``{category: {ratio: {"value", "benchmark", "color", ...}}}``.
            category: Only plot this category (case-insensitive).
            benchmark_data: ``{category: {ratio: benchmark}}`` used when the
                ratio entries carry no ``benchmark``.
        """
        if not isinstance(ratio_data, Mapping) or not ratio_data:
            return self._empty_chart("No ratio data available", height)
        lookup = {str(k).lower(): k for k in ratio_data}
        if category:
            key = lookup.get(str(category).lower())
            categories = [key] if key is not None else []
        else:
            categories = list(ratio_data)
        bench_lookup = {str(k).lower(): v for k, v in (benchmark_data or {}).items() if isinstance(v, Mapping)}

        names, values, benchmarks, colors = [], [], [], []
        for cat in categories:
            entries = ratio_data.get(cat)
            if not isinstance(entries, Mapping):
                continue
            for ratio, info in entries.items():
                if isinstance(info, Mapping):
                    value = _finite(info.get("value"))
                    benchmark = _finite(info.get("benchmark"))
                    color = info.get("color") or COLORS["info"]
                else:
                    value, benchmark, color = _finite(info), None, COLORS["info"]
                if value is None:
                    continue
                if benchmark is None:
                    benchmark = _finite((bench_lookup.get(str(cat).lower()) or {}).get(ratio))
                names.append(_label(ratio) if len(categories) == 1 else f"{str(cat).title()}: {_label(ratio)}")
                values.append(value)
                benchmarks.append(benchmark)
                colors.append(color)
        if not names:
            return self._empty_chart("No ratios found for the specified category", height)

        fig = go.Figure(go.Bar(x=names, y=values, name="Company", marker_color=colors,
                               text=[f"{v:,.2f}" for v in values], textposition="outside",
                               hovertemplate="%{x}: %{y:,.2f}<extra>Company</extra>"))
        if any(b is not None for b in benchmarks):
            fig.add_trace(go.Bar(x=names, y=benchmarks, name="Sector benchmark", marker_color=COLORS["secondary"],
                                 opacity=0.75, hovertemplate="%{x}: %{y:,.2f}<extra>Benchmark</extra>"))
        title = f"Financial Ratios - {str(categories[0]).title()}" if category and categories else "Financial Ratios"
        self._apply_layout(fig, title, height, width, barmode="group")
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(margin=dict(b=120))
        return fig

    @_safe_chart("metrics")
    def plot_metrics_bar_chart(self,
                               metrics_df: Union[pd.DataFrame, Mapping[str, Any]],
                               title: str = "Metrics",
                               height: Optional[int] = None,
                               width: Optional[int] = None,
                               is_percent: bool = False) -> go.Figure:
        """
        Bar chart of ``Metric`` / ``Value`` rows (or ``{metric: value}``).

        Args:
            is_percent: Values are already in percent points (``12.5`` == 12.5 %).
        """
        if isinstance(metrics_df, Mapping):
            metrics_df = pd.DataFrame({"Metric": list(metrics_df), "Value": list(metrics_df.values())})
        if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
            return self._empty_chart("No metrics available", height)
        label_col = "Metric" if "Metric" in metrics_df.columns else metrics_df.columns[0]
        value_col = "Value" if "Value" in metrics_df.columns else metrics_df.columns[-1]
        df = pd.DataFrame({"Metric": metrics_df[label_col].astype(str),
                           "Value": pd.to_numeric(metrics_df[value_col], errors="coerce")})
        df = df[np.isfinite(df["Value"].to_numpy(dtype=float))]
        if df.empty:
            return self._empty_chart("No metrics available", height)
        suffix = "%" if is_percent else ""
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in df["Value"]]
        fig = go.Figure(go.Bar(x=df["Metric"], y=df["Value"], marker_color=colors,
                               text=[f"{v:,.2f}{suffix}" for v in df["Value"]], textposition="outside",
                               hovertemplate="%{x}: %{y:,.2f}" + suffix + "<extra></extra>"))
        self._apply_layout(fig, title, height, width, default_height=400, showlegend=False)
        fig.update_yaxes(ticksuffix=suffix)
        return fig

    @_safe_chart("comparison")
    def plot_bar_comparison(self,
                            bar_data: Mapping[str, Sequence[Any]],
                            companies: Sequence[str],
                            primary_index: Optional[int] = None,
                            title: str = "Comparison",
                            height: Optional[int] = None,
                            width: Optional[int] = None,
                            is_percent: Optional[bool] = None,
                            value_format: Optional[str] = None) -> go.Figure:
        """
        Grouped bars: one group per metric, one bar per company.

        Args:
            bar_data: ``{metric: [value per company]}`` (aligned with ``companies``).
            companies: Company labels.
            primary_index: Index of the highlighted company.
            is_percent: Values are fractions to show as percent; inferred from
                metric names when None.
            value_format: ``"currency"`` to format values as dollars.
        """
        companies = [str(c) for c in (companies or [])]
        if not isinstance(bar_data, Mapping) or not bar_data or not companies:
            return self._empty_chart("No comparison data available", height)
        metrics = [m for m, vals in bar_data.items()
                   if any(_finite(v) is not None for v in list(vals)[:len(companies)])]
        if not metrics:
            return self._empty_chart("No comparison data available", height)
        if is_percent is None:
            is_percent = all(_is_percent_key(m) for m in metrics)
        scale = 100.0 if is_percent else 1.0
        primary = companies[primary_index] if primary_index is not None and 0 <= primary_index < len(companies) \
            else None
        colors = self._series_colors(companies, primary)
        prefix = "$" if value_format == "currency" else ""
        suffix = "%" if is_percent else ""

        fig = go.Figure()
        for i, company in enumerate(companies):
            ys = []
            for metric in metrics:
                values = list(bar_data[metric])
                value = _finite(values[i]) if i < len(values) else None
                ys.append(value * scale if value is not None else None)
            fig.add_trace(go.Bar(x=[_label(m) for m in metrics], y=ys, name=company, marker_color=colors[company],
                                 marker_line=dict(width=2 if company == primary else 0, color="white"),
                                 hovertemplate=prefix + "%{y:,.2f}" + suffix + "<extra>" + company + "</extra>"))
        self._apply_layout(fig, title, height, width, default_height=400, barmode="group")
        fig.update_yaxes(tickprefix=prefix, ticksuffix=suffix)
        return fig

    @_safe_chart("radar comparison")
    def plot_radar_comparison(self,
                              radar_data: Mapping[str, Mapping[str, Any]],
                              names: Optional[Mapping[str, str]] = None,
                              primary_ticker: Optional[str] = None,
                              title: str = "Comparison",
                              height: Optional[int] = None,
                              width: Optional[int] = None,
                              metrics: Optional[Sequence[str]] = None,
                              radial_range: Optional[Tuple[float, float]] = None) -> go.Figure:
        """
        Radar chart of ``{company: {metric: value}}``.

        Metrics are the union of all companies' metrics unless ``metrics`` is
        given; missing values are drawn at 0. Use comparable scales (e.g.
        percentile ranks with ``radial_range=(0, 100)``).
        """
        if not isinstance(radar_data, Mapping) or not radar_data:
            return self._empty_chart("No comparison data available", height)
        companies = [c for c, d in radar_data.items() if isinstance(d, Mapping)]
        metric_list = list(metrics) if metrics else list(dict.fromkeys(m for c in companies for m in radar_data[c]))
        metric_list = [m for m in metric_list if any(_finite(radar_data[c].get(m)) is not None for c in companies)]
        if len(metric_list) < 2:
            return self._empty_chart("At least two metrics with data are needed for a radar chart", height)

        colors = self._series_colors(companies, primary_ticker)
        theta = [_label(m) for m in metric_list] + [_label(metric_list[0])]
        fig = go.Figure()
        for company in companies:
            values = [_finite(radar_data[company].get(m)) for m in metric_list]
            if all(v is None for v in values):
                continue
            r = [v if v is not None else 0.0 for v in values]
            label = (names or {}).get(company, company)
            is_primary = company == primary_ticker
            fig.add_trace(go.Scatterpolar(r=r + [r[0]], theta=theta, name=label, fill="toself",
                                          opacity=0.85 if is_primary else 0.55,
                                          line=dict(color=colors[company], width=3 if is_primary else 1.5),
                                          hovertemplate="%{theta}: %{r:.1f}<extra>" + label + "</extra>"))
        if not fig.data:
            return self._empty_chart("No comparison data available", height)
        radial = dict(visible=True, gridcolor=self.grid_color, linecolor=self.grid_color)
        if radial_range:
            radial["range"] = list(radial_range)
        self._apply_layout(fig, title, height, width, default_height=500,
                           polar=dict(bgcolor=self.bg_color, radialaxis=radial,
                                      angularaxis=dict(gridcolor=self.grid_color, linecolor=self.grid_color)))
        fig.update_layout(legend=dict(y=-0.15, yanchor="top", x=0.5, xanchor="center"))
        return fig

    @_safe_chart("company comparison")
    def plot_company_comparison(self,
                                comparison_data: Dict[str, Dict[str, float]],
                                metrics: List[str],
                                height: Optional[int] = None,
                                width: Optional[int] = None) -> go.Figure:
        """Radar chart comparing companies on ``metrics`` (``{company: {metric: value}}``)."""
        if not comparison_data or not metrics:
            return self._empty_chart("No comparison data available", height)
        return self.plot_radar_comparison(comparison_data, title="Company Comparison", height=height, width=width,
                                          metrics=metrics)

    @_safe_chart("metrics heatmap")
    def plot_metrics_heatmap(self,
                             data: pd.DataFrame,
                             title: str = "Metrics Heatmap",
                             height: Optional[int] = None,
                             width: Optional[int] = None,
                             is_percent: bool = False,
                             colorscale: Optional[Any] = None,
                             higher_is_better: Optional[Mapping[str, bool]] = None) -> go.Figure:
        """
        Heatmap of companies (rows) x metrics (columns).

        Colours are relative within each column (z-score), so metrics with
        different scales can share one chart; cells show the actual values.

        Args:
            is_percent: Values are percent points (``12.5`` == 12.5 %).
            colorscale: Plotly colour scale; default is red (worse) to green (better).
            higher_is_better: ``{metric: bool}``; metrics marked False are
                coloured inversely (e.g. P/E, Debt/Equity).
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            return self._empty_chart("No metrics data available", height)
        df = data.apply(pd.to_numeric, errors="coerce").astype(float)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.loc[df.notna().any(axis=1), df.notna().any(axis=0)]
        if df.empty:
            return self._empty_chart("No numeric metrics available", height)

        scores = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for column in df.columns:
            col = df[column]
            std = col.std(ddof=0)
            z = (col - col.mean()) / std if std and np.isfinite(std) else col * 0.0
            if higher_is_better and higher_is_better.get(column) is False:
                z = -z
            scores[column] = z.clip(-2.5, 2.5)
        suffix = "%" if is_percent else ""
        text = [[f"{v:,.2f}{suffix}" if np.isfinite(v) else "N/A" for v in row] for row in df.to_numpy()]
        fig = go.Figure(go.Heatmap(
            z=scores.to_numpy(), x=[_label(c) for c in df.columns], y=[str(i) for i in df.index],
            text=text, texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
            colorscale=colorscale or [[0, NEGATIVE_COLOR], [0.5, "#2b2b2b" if self.theme == "dark" else "#f5f5f5"],
                                      [1, POSITIVE_COLOR]],
            zmid=0, showscale=False, xgap=2, ygap=2,
        ))
        self._apply_layout(fig, title, height or max(300, 60 + 40 * len(df)), width)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(side="top", showgrid=False)
        fig.update_layout(margin=dict(t=110 if title else 60))
        return fig

    # ------------------------------------------------------------------
    # Sector / market charts
    # ------------------------------------------------------------------
    @_safe_chart("sector performance")
    def plot_sector_performance(self,
                                sector_data: Any,
                                metric: Any = "1Y",
                                height: Optional[int] = None,
                                width: Optional[int] = None,
                                title: Optional[str] = None) -> go.Figure:
        """
        Sector performance chart.

        Accepts:

        * a DataFrame with a ``Sector`` column and a return column ``metric``
          (percent points) -> horizontal bar ranking;
        * ``{ticker: return Series}`` (``metric`` may then be a ``{ticker: name}``
          mapping) -> line chart of the series;
        * a price DataFrame with a ``Close`` column (``metric`` = ticker label)
          -> normalised performance line.
        """
        if isinstance(sector_data, Mapping):
            names = metric if isinstance(metric, Mapping) else None
            return self.plot_price_comparison(sector_data, names, title=title or "Performance Comparison",
                                              height=height, width=width)
        if not isinstance(sector_data, pd.DataFrame) or sector_data.empty:
            return self._empty_chart("No sector performance data available", height)
        if "Sector" not in sector_data.columns:
            if any(str(c).lower() == "close" for c in sector_data.columns):
                label = str(metric) if isinstance(metric, str) else "Price"
                return self.plot_price_comparison({label: sector_data}, title=title or f"{label} Performance",
                                                  height=height, width=width, normalize=True)
            return self._empty_chart("No sector performance data available", height)
        if not isinstance(metric, str) or metric not in sector_data.columns:
            return self._empty_chart(f"No '{metric}' performance data available", height)

        df = pd.DataFrame({"Sector": sector_data["Sector"].astype(str),
                           "Value": pd.to_numeric(sector_data[metric], errors="coerce")})
        df = df[np.isfinite(df["Value"].to_numpy(dtype=float))].sort_values("Value")
        if df.empty:
            return self._empty_chart(f"No '{metric}' performance data available", height)
        colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in df["Value"]]
        fig = go.Figure(go.Bar(x=df["Value"], y=df["Sector"], orientation="h", marker_color=colors,
                               text=[f"{v:+.2f}%" for v in df["Value"]], textposition="auto",
                               hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
        self._apply_layout(fig, title or f"Sector Performance ({metric})", height, width,
                           default_height=200 + len(df) * 32, showlegend=False)
        fig.update_xaxes(title_text="Return", ticksuffix="%", zeroline=True)
        fig.update_yaxes(categoryorder="array", categoryarray=df["Sector"].tolist(), showgrid=False)
        fig.update_layout(margin=dict(l=170))
        return fig

    @_safe_chart("sector distribution")
    def plot_sector_distribution(self,
                                 sector_df: Union[pd.DataFrame, Mapping[str, Any]],
                                 title: str = "Sector Distribution",
                                 height: Optional[int] = None,
                                 width: Optional[int] = None) -> go.Figure:
        """Donut chart of ``Sector`` / ``Count`` rows (or ``{sector: count}``)."""
        if isinstance(sector_df, Mapping):
            sector_df = pd.DataFrame({"Sector": list(sector_df), "Count": list(sector_df.values())})
        if not isinstance(sector_df, pd.DataFrame) or sector_df.empty:
            return self._empty_chart("No sector data available", height)
        label_col = "Sector" if "Sector" in sector_df.columns else sector_df.columns[0]
        value_col = "Count" if "Count" in sector_df.columns else sector_df.columns[-1]
        df = pd.DataFrame({"label": sector_df[label_col].astype(str),
                           "value": pd.to_numeric(sector_df[value_col], errors="coerce")})
        df = df[df["value"] > 0]
        if df.empty:
            return self._empty_chart("No sector data available", height)
        sector_colors = COLORS.get("sectors", {})
        fallback = iter(self.palette * 3)
        colors = [sector_colors.get(label) or next(fallback) for label in df["label"]]
        fig = go.Figure(go.Pie(labels=df["label"], values=df["value"], hole=0.45, marker=dict(colors=colors),
                               hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
        self._apply_layout(fig, title, height, width, default_height=450)
        return fig

    def plot_method_distribution(self,
                                 method_df: Union[pd.DataFrame, Mapping[str, Any]],
                                 title: str = "Valuation Method Distribution",
                                 height: Optional[int] = None,
                                 width: Optional[int] = None) -> go.Figure:
        """Donut chart of ``Method`` / ``Count`` rows."""
        if isinstance(method_df, pd.DataFrame) and "Method" in method_df.columns:
            method_df = method_df.rename(columns={"Method": "Sector"})
        return self.plot_sector_distribution(method_df, title=title, height=height, width=width)

    @_safe_chart("market cap treemap")
    def plot_market_cap_treemap(self,
                                companies_df: pd.DataFrame,
                                title: str = "Market Cap Distribution",
                                height: Optional[int] = None,
                                width: Optional[int] = None,
                                color_column: Optional[str] = "Total Return",
                                group_column: Optional[str] = None) -> go.Figure:
        """
        Treemap sized by ``Market Cap`` (labels from ``Symbol``/``Name``).

        Tiles are coloured by ``color_column`` (percent points, red-green around
        zero) when present; ``group_column`` (e.g. ``Subsector``) adds a level.
        """
        if not isinstance(companies_df, pd.DataFrame) or companies_df.empty or "Market Cap" not in companies_df:
            return self._empty_chart("No market cap data available", height)
        df = companies_df.copy()
        df["_cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
        df = df[df["_cap"] > 0]
        if df.empty:
            return self._empty_chart("No market cap data available", height)
        label_col = "Symbol" if "Symbol" in df.columns else ("Name" if "Name" in df.columns else None)
        df["_label"] = df[label_col].astype(str) if label_col else [f"#{i + 1}" for i in range(len(df))]
        df["_name"] = df["Name"].astype(str) if "Name" in df.columns else df["_label"]

        ids, labels, parents, values, colors, hover = [], [], [], [], [], []
        has_color = bool(color_column) and color_column in df.columns
        color_values = pd.to_numeric(df[color_column], errors="coerce") if has_color else None
        if group_column and group_column in df.columns:
            for group, cap in df.groupby(group_column)["_cap"].sum().items():
                ids.append(f"group::{group}")
                labels.append(str(group))
                parents.append("")
                values.append(float(cap))
                colors.append(0.0)
                hover.append(format_large_number(cap))
        for idx, row in df.iterrows():
            ids.append(row["_label"])
            labels.append(row["_label"])
            parents.append(f"group::{row[group_column]}" if group_column and group_column in df.columns else "")
            values.append(float(row["_cap"]))
            change = _finite(color_values.loc[idx]) if has_color else None
            colors.append(change if change is not None else 0.0)
            extra = f"<br>{color_column}: {change:+.2f}%" if change is not None else ""
            hover.append(f"{row['_name']}<br>{format_large_number(row['_cap'])}{extra}")

        bound = max([abs(c) for c in colors] + [1.0])
        marker = dict(colors=colors, colorscale=[[0, NEGATIVE_COLOR], [0.5, "#3a3a3a"], [1, POSITIVE_COLOR]],
                      cmin=-bound, cmax=bound, cmid=0, showscale=has_color,
                      colorbar=dict(title=dict(text=f"{color_column} (%)", side="top")) if has_color else None)
        if not has_color:
            marker = dict(colors=[self.palette[i % len(self.palette)] for i in range(len(ids))])
        fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                   marker=marker, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                                   texttemplate="<b>%{label}</b><br>%{percentRoot:.1%}"))
        self._apply_layout(fig, title, height, width, default_height=500)
        fig.update_layout(margin=dict(l=10, r=10, t=60 if title else 20, b=10))
        return fig

    # ------------------------------------------------------------------
    # Valuation charts
    # ------------------------------------------------------------------
    @_safe_chart("valuation heatmap")
    def plot_valuation_heatmap(self,
                               valuation_data: Dict[str, Dict[str, float]],
                               height: Optional[int] = None,
                               width: Optional[int] = None) -> go.Figure:
        """
        Company multiples vs. sector averages as a one-row heatmap of the
        percentage premium (red) / discount (green).

        Args:
            valuation_data: ``{"company": {metric: value}, "sector_avg": {metric: value}}``.
        """
        if not isinstance(valuation_data, Mapping) or not isinstance(valuation_data.get("company"), Mapping) \
                or not isinstance(valuation_data.get("sector_avg"), Mapping):
            return self._empty_chart("No valuation comparison data available", height)
        company_vals, sector_vals = valuation_data["company"], valuation_data["sector_avg"]
        metrics, diffs, text = [], [], []
        for metric, raw in company_vals.items():
            company_value, sector_value = _finite(raw), _finite(sector_vals.get(metric))
            if company_value is None or not sector_value:
                continue
            diff = (company_value / sector_value - 1) * 100
            metrics.append(_label(metric))
            diffs.append(diff)
            text.append(f"{diff:+.1f}%<br>{company_value:,.2f} vs {sector_value:,.2f}")
        if not metrics:
            return self._empty_chart("No comparable valuation multiples available", height)
        bound = min(max(abs(d) for d in diffs), 100) or 1
        fig = go.Figure(go.Heatmap(
            z=[diffs], x=metrics, y=["vs. sector"], text=[text], texttemplate="%{text}",
            colorscale=[[0, POSITIVE_COLOR], [0.5, COLORS["info"]], [1, NEGATIVE_COLOR]],
            zmin=-bound, zmax=bound, zmid=0, xgap=3,
            colorbar=dict(title=dict(text="Premium vs.<br>sector (%)", side="top"), ticksuffix="%"),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))
        self._apply_layout(fig, "Valuation Multiples vs. Sector Averages", height, width, default_height=300)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(showgrid=False)
        return fig

    @_safe_chart("sector valuation heatmap")
    def plot_valuation_sector_heatmap(self,
                                      valuation_df: pd.DataFrame,
                                      title: str = "Sector Valuation Heatmap",
                                      height: Optional[int] = None,
                                      width: Optional[int] = None,
                                      metrics: Optional[Sequence[str]] = None) -> go.Figure:
        """
        Valuation multiples of sector companies relative to the sector median.

        Args:
            valuation_df: One row per company with ``Symbol`` (or ``Company``)
                and multiple columns (default: P/E, P/S, P/B, EV/EBITDA).
                Non-positive multiples are treated as not meaningful.
        """
        if not isinstance(valuation_df, pd.DataFrame) or valuation_df.empty:
            return self._empty_chart("No valuation data available", height)
        label_col = "Symbol" if "Symbol" in valuation_df.columns else (
            "Company" if "Company" in valuation_df.columns else None)
        candidates = list(metrics) if metrics else ["P/E", "P/S", "P/B", "EV/EBITDA", "EV/Revenue"]
        columns = [c for c in candidates if c in valuation_df.columns]
        if not columns:
            return self._empty_chart("No valuation multiples available", height)
        df = valuation_df[columns].apply(pd.to_numeric, errors="coerce").astype(float)
        df = df.where(df > 0)
        df.index = valuation_df[label_col].astype(str) if label_col else [str(i) for i in valuation_df.index]
        df = df.loc[df.notna().any(axis=1), df.notna().any(axis=0)]
        if df.empty:
            return self._empty_chart("No valuation multiples available", height)
        medians = df.median()
        premium = (df / medians - 1) * 100
        text = [[f"{v:,.1f}x" if np.isfinite(v) else "N/A" for v in row] for row in df.to_numpy()]
        hover = [[f"{v:,.1f}x ({p:+.0f}% vs median {m:,.1f}x)" if np.isfinite(v) else "N/A"
                  for v, p, m in zip(row, prow, medians.to_numpy())]
                 for row, prow in zip(df.to_numpy(), premium.to_numpy())]
        fig = go.Figure(go.Heatmap(
            z=premium.clip(-100, 100).to_numpy(), x=list(df.columns), y=list(df.index), text=text,
            texttemplate="%{text}", customdata=hover, hovertemplate="%{y} %{x}: %{customdata}<extra></extra>",
            colorscale=[[0, POSITIVE_COLOR], [0.5, "#2b2b2b" if self.theme == "dark" else "#f5f5f5"],
                        [1, NEGATIVE_COLOR]],
            zmid=0, zmin=-100, zmax=100, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="vs. sector<br>median", side="top"), ticksuffix="%"),
        ))
        self._apply_layout(fig, title, height or max(320, 80 + 32 * len(df)), width)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(side="top", showgrid=False)
        fig.update_layout(margin=dict(t=110 if title else 60))
        return fig

    @_safe_chart("valuation upside")
    def plot_valuation_upside(self,
                              upside_data: Mapping[str, Any],
                              names: Optional[Mapping[str, str]] = None,
                              primary_ticker: Optional[str] = None,
                              title: str = "Valuation Upside / Downside",
                              height: Optional[int] = None,
                              width: Optional[int] = None) -> go.Figure:
        """Horizontal bars of upside to fair value in percent points (``{ticker: pct}``)."""
        if not isinstance(upside_data, Mapping):
            return self._empty_chart("No valuation data available", height)
        items = [(str(k), _finite(v)) for k, v in upside_data.items()]
        items = sorted([(k, v) for k, v in items if v is not None], key=lambda kv: kv[1])
        if not items:
            return self._empty_chart("No valuation data available", height)
        labels = [f"{(names or {}).get(k, k)} ({k})" if (names or {}).get(k, k) != k else k for k, _ in items]
        values = [v for _, v in items]
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
        line_widths = [3 if k == primary_ticker else 0 for k, _ in items]
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                               marker_line=dict(width=line_widths, color="white"),
                               text=[f"{v:+.1f}%" for v in values], textposition="auto",
                               hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=NEUTRAL_COLOR, width=1))
        self._apply_layout(fig, title, height, width, default_height=max(300, 120 + 35 * len(items)),
                           showlegend=False)
        fig.update_xaxes(title_text="Upside to fair value", ticksuffix="%")
        fig.update_yaxes(showgrid=False)
        return fig

    @_safe_chart("fair value scatter")
    def plot_fair_value_scatter(self,
                                data: pd.DataFrame,
                                title: str = "Fair Value vs. Price",
                                height: Optional[int] = None,
                                width: Optional[int] = None,
                                margin_of_safety: Optional[float] = None) -> go.Figure:
        """
        Scatter of current price (x) against fair value (y), one point per stock.

        Args:
            data: Columns ``Symbol``, ``Price`` and ``Fair Value`` (optional
                ``Company`` for hover text).
            margin_of_safety: Fraction; adds the ``fair = price / (1 - mos)``
                line above which stocks qualify as undervalued.

        Points above the 45-degree line trade below their estimated fair value.
        """
        required = {"Price", "Fair Value"}
        if not isinstance(data, pd.DataFrame) or data.empty or not required.issubset(data.columns):
            return self._empty_chart("No valuation data available", height)
        df = pd.DataFrame({
            "label": data["Symbol"].astype(str) if "Symbol" in data.columns else [str(i) for i in data.index],
            "name": data["Company"].astype(str) if "Company" in data.columns else "",
            "price": pd.to_numeric(data["Price"], errors="coerce"),
            "fair": pd.to_numeric(data["Fair Value"], errors="coerce"),
        })
        df = df[(df["price"] > 0) & (df["fair"] > 0)]
        if df.empty:
            return self._empty_chart("No stocks with both a price and a fair value", height)
        upside = (df["fair"] / df["price"] - 1) * 100
        colors = [POSITIVE_COLOR if u >= 0 else NEGATIVE_COLOR for u in upside]
        hover = [f"{n} ({lbl})<br>" if n else f"{lbl}<br>" for n, lbl in zip(df["name"], df["label"])]
        hover = [f"{h}Price ${p:,.2f}<br>Fair value ${f:,.2f}<br>Upside {u:+.1f}%"
                 for h, p, f, u in zip(hover, df["price"], df["fair"], upside)]
        top = float(max(df["price"].max(), df["fair"].max())) * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Fair value = price",
                                 line=dict(color=NEUTRAL_COLOR, dash="dot"), hoverinfo="skip"))
        mos = _finite(margin_of_safety)
        if mos is not None and 0 < mos < 1:
            fig.add_trace(go.Scatter(x=[0, top * (1 - mos)], y=[0, top], mode="lines",
                                     name=f"{mos:.0%} margin of safety",
                                     line=dict(color=COLORS["warning"], dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["price"], y=df["fair"], mode="markers+text", text=df["label"],
                                 textposition="top center", name="Stocks", customdata=hover,
                                 marker=dict(color=colors, size=11, line=dict(width=1, color=self.bg_color)),
                                 hovertemplate="%{customdata}<extra></extra>"))
        self._apply_layout(fig, title, height, width, default_height=500)
        fig.update_xaxes(title_text="Current price", tickprefix="$", rangemode="tozero")
        fig.update_yaxes(title_text="Fair value", tickprefix="$", rangemode="tozero")
        return fig

    @_safe_chart("upside distribution")
    def plot_upside_histogram(self,
                              upside_data: Any,
                              title: str = "Distribution of Upside Potential",
                              height: Optional[int] = None,
                              width: Optional[int] = None) -> go.Figure:
        """Histogram of upside values in percent points."""
        values = _numeric_series(list(upside_data) if upside_data is not None else [])
        if values.empty:
            return self._empty_chart("No upside data available", height)
        fig = go.Figure(go.Histogram(x=values.values, nbinsx=max(5, min(30, len(values))),
                                     marker_color=COLORS["info"], opacity=0.85,
                                     hovertemplate="%{x}: %{y} stocks<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=NEUTRAL_COLOR, width=1, dash="dot"))
        median = float(values.median())
        fig.add_vline(x=median, line=dict(color=COLORS["warning"], width=2, dash="dash"),
                      annotation_text=f"Median {median:+.1f}%", annotation_position="top")
        self._apply_layout(fig, title, height, width, default_height=400, showlegend=False, bargap=0.05)
        fig.update_xaxes(title_text="Upside to fair value", ticksuffix="%")
        fig.update_yaxes(title_text="Number of stocks")
        return fig

    @_safe_chart("valuation breakdown")
    def plot_valuation_breakdown(self,
                                 components_df: Union[pd.DataFrame, Mapping[str, Any]],
                                 current_price: Optional[float] = None,
                                 title: str = "Valuation Breakdown",
                                 height: Optional[int] = None,
                                 width: Optional[int] = None) -> go.Figure:
        """Fair value per method (``Method`` / ``Value`` rows) with the current price as reference line."""
        if isinstance(components_df, Mapping):
            components_df = pd.DataFrame({"Method": list(components_df), "Value": list(components_df.values())})
        if not isinstance(components_df, pd.DataFrame) or components_df.empty:
            return self._empty_chart("No valuation methods available", height)
        label_col = "Method" if "Method" in components_df.columns else components_df.columns[0]
        value_col = "Value" if "Value" in components_df.columns else components_df.columns[-1]
        df = pd.DataFrame({"Method": components_df[label_col].astype(str),
                           "Value": pd.to_numeric(components_df[value_col], errors="coerce")})
        df = df[np.isfinite(df["Value"].to_numpy(dtype=float))]
        if df.empty:
            return self._empty_chart("No valuation methods produced a value", height)
        price = _finite(current_price)
        if price and price > 0:
            colors = [POSITIVE_COLOR if v >= price else NEGATIVE_COLOR for v in df["Value"]]
        else:
            colors = [COLORS["info"]] * len(df)
        fig = go.Figure(go.Bar(x=df["Method"], y=df["Value"], marker_color=colors,
                               text=[f"${v:,.2f}" for v in df["Value"]], textposition="outside",
                               hovertemplate="%{x}: $%{y:,.2f}<extra></extra>"))
        if price and price > 0:
            fig.add_hline(y=price, line=dict(color=COLORS["warning"], width=2, dash="dash"),
                          annotation_text=f"Current price ${price:,.2f}", annotation_position="top left")
        self._apply_layout(fig, title, height, width, default_height=400, showlegend=False)
        fig.update_yaxes(title_text="Value per share", tickprefix="$", tickformat=",.2f")
        return fig

    def _sensitivity_figure(self, values: np.ndarray, x_labels: List[str], y_labels: List[str],
                            x_title: str, y_title: str, current_price: Optional[float],
                            title: str, height: Any, width: Any,
                            marker: Optional[Tuple[int, int]] = None) -> go.Figure:
        """Heatmap of values per share (rows = y, columns = x)."""
        values = np.asarray(values, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        if values.size == 0 or np.isnan(values).all():
            return self._empty_chart("No valid sensitivity values", height)
        price = _finite(current_price)
        text = [[f"${v:,.2f}" if np.isfinite(v) else "N/A" for v in row] for row in values]
        if price and price > 0:
            z = (values / price - 1) * 100
            finite = np.abs(z[np.isfinite(z)])
            bound = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
            hover = [[f"{y_title}: {y_labels[i]}<br>{x_title}: {x_labels[j]}<br>Value: {text[i][j]}<br>"
                      f"vs. price: {z[i][j]:+.1f}%" if np.isfinite(z[i][j]) else "N/A"
                      for j in range(len(x_labels))] for i in range(len(y_labels))]
            heatmap = go.Heatmap(z=z, x=x_labels, y=y_labels, text=text, texttemplate="%{text}", customdata=hover,
                                 hovertemplate="%{customdata}<extra></extra>",
                                 colorscale=[[0, NEGATIVE_COLOR], [0.5, "#3a3a3a"], [1, POSITIVE_COLOR]],
                                 zmin=-bound, zmax=bound, zmid=0, xgap=1, ygap=1,
                                 colorbar=dict(title=dict(text="vs. current<br>price (%)", side="top"),
                                               ticksuffix="%"))
        else:
            hover = [[f"{y_title}: {y_labels[i]}<br>{x_title}: {x_labels[j]}<br>Value: {text[i][j]}"
                      for j in range(len(x_labels))] for i in range(len(y_labels))]
            heatmap = go.Heatmap(z=values, x=x_labels, y=y_labels, text=text, texttemplate="%{text}",
                                 customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                                 colorscale="Viridis", xgap=1, ygap=1,
                                 colorbar=dict(title=dict(text="Value per<br>share ($)", side="top"),
                                               tickprefix="$"))
        fig = go.Figure(heatmap)
        if marker is not None:
            row, col = marker
            fig.add_trace(go.Scatter(x=[x_labels[col]], y=[y_labels[row]], mode="markers", name="Base case",
                                     marker=dict(symbol="star", size=16, color="white",
                                                 line=dict(width=1.5, color="black")),
                                     hovertemplate="Base case<extra></extra>"))
        self._apply_layout(fig, title, height, width, default_height=450, showlegend=marker is not None)
        fig.update_xaxes(title_text=x_title, type="category", showgrid=False)
        fig.update_yaxes(title_text=y_title, type="category", showgrid=False)
        return fig

    @_safe_chart("DCF sensitivity")
    def plot_dcf_sensitivity(self,
                             dcf_data: Dict[str, Any],
                             height: Optional[int] = None,
                             width: Optional[int] = None) -> go.Figure:
        """
        DCF value sensitivity to growth (rows) and discount rate (columns).

        Args:
            dcf_data: ``{"sensitivity": {"growth_rates": [...], "discount_rates": [...],
                "values": [[value per share]]}, "current_price": float,
                "base_case": {"growth_rate", "discount_rate"}}``. ``values[i][j]``
                is the value for ``growth_rates[i]`` and ``discount_rates[j]``
                (a transposed matrix is detected). Rates may be fractions
                (0.08) or percent points (8).
        """
        if not isinstance(dcf_data, Mapping) or not isinstance(dcf_data.get("sensitivity"), Mapping):
            return self._empty_chart("No DCF sensitivity analysis data available", height)
        sensitivity = dcf_data["sensitivity"]
        growth = [_finite(g) for g in (sensitivity.get("growth_rates") or [])]
        discount = [_finite(d) for d in (sensitivity.get("discount_rates") or [])]
        raw_values = sensitivity.get("values")
        if not growth or not discount or raw_values is None or None in growth or None in discount:
            return self._empty_chart("Incomplete DCF sensitivity data", height)
        values = pd.DataFrame(raw_values).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if values.shape == (len(discount), len(growth)) and values.shape != (len(growth), len(discount)):
            values = values.T
        if values.shape != (len(growth), len(discount)):
            return self._empty_chart("Inconsistent DCF sensitivity data dimensions", height)

        growth_fraction = _is_fraction_scale(growth)
        discount_fraction = _is_fraction_scale(discount)

        def labels(rates: List[float], fraction: bool) -> List[str]:
            return [f"{r * 100:.1f}%" if fraction else f"{r:.1f}%" for r in rates]

        def nearest(rates: List[float], fraction: bool, base: Any) -> Optional[int]:
            base = _finite(base)
            if base is None:
                return None
            if fraction and abs(base) > 1:
                base /= 100
            elif not fraction and abs(base) <= 1:
                base *= 100
            return min(range(len(rates)), key=lambda i: abs(rates[i] - base))

        base_case = dcf_data.get("base_case") or {}
        growth_idx = nearest(growth, growth_fraction, base_case.get("growth_rate"))
        discount_idx = nearest(discount, discount_fraction, base_case.get("discount_rate"))
        marker = (growth_idx, discount_idx) if growth_idx is not None and discount_idx is not None else None
        return self._sensitivity_figure(values, labels(discount, discount_fraction), labels(growth, growth_fraction),
                                        "Discount Rate", "Growth Rate", dcf_data.get("current_price"),
                                        "DCF Sensitivity Analysis", height, width, marker=marker)

    # ------------------------------------------------------------------
    # Risk charts
    # ------------------------------------------------------------------
    @_safe_chart("bankruptcy risk")
    def plot_bankruptcy_risk(self,
                             risk_data: Dict[str, Any],
                             height: Optional[int] = None,
                             width: Optional[int] = None) -> go.Figure:
        """Gauge per bankruptcy model (``risk_data["models"]``) with the overall assessment."""
        if not isinstance(risk_data, Mapping) or not isinstance(risk_data.get("models"), Mapping):
            return self._empty_chart("No bankruptcy risk data available", height)

        gauges = []
        for model_name, model in risk_data["models"].items():
            if not isinstance(model, Mapping):
                continue
            score = _finite(model.get("score"))
            if model_name == "altman_z_score" and score is not None:
                thresholds = model.get("thresholds") or {}
                safe, grey = _finite(thresholds.get("safe")) or 3.0, _finite(thresholds.get("grey")) or 1.8
                max_val = max(safe * 1.5, score * 1.1)
                steps = [{"range": [min(0, score), grey], "color": NEGATIVE_COLOR},
                         {"range": [grey, safe], "color": COLORS["warning"]},
                         {"range": [safe, max_val], "color": POSITIVE_COLOR}]
                gauges.append((model.get("model") or "Altman Z-Score", score, [min(0, score), max_val], steps,
                               model.get("color")))
            elif model_name == "springate" and score is not None:
                threshold = _finite(model.get("threshold")) or 0.862
                max_val = max(threshold * 2, score * 1.1)
                steps = [{"range": [min(0, score), threshold], "color": NEGATIVE_COLOR},
                         {"range": [threshold, max_val], "color": POSITIVE_COLOR}]
                gauges.append((model.get("model") or "Springate S-Score", score, [min(0, score), max_val], steps,
                               model.get("color")))
            elif model_name == "zmijewski":
                probability = _finite(model.get("probability"))
                threshold = _finite(model.get("threshold")) or 0.5
                if probability is not None:
                    steps = [{"range": [0, threshold], "color": POSITIVE_COLOR},
                             {"range": [threshold, 1], "color": NEGATIVE_COLOR}]
                    gauges.append((model.get("model") or "Zmijewski Probability", probability, [0, 1], steps,
                                   model.get("color")))
                elif score is not None:
                    bound = max(10.0, abs(score) * 1.1)
                    steps = [{"range": [-bound, 0], "color": POSITIVE_COLOR},
                             {"range": [0, bound], "color": NEGATIVE_COLOR}]
                    gauges.append((model.get("model") or "Zmijewski Score", score, [-bound, bound], steps,
                                   model.get("color")))
            elif score is not None:
                max_val = max(10.0, score * 1.1)
                steps = [{"range": [0, max_val / 2], "color": NEGATIVE_COLOR},
                         {"range": [max_val / 2, max_val * 0.75], "color": COLORS["warning"]},
                         {"range": [max_val * 0.75, max_val], "color": POSITIVE_COLOR}]
                gauges.append((model.get("model") or _label(model_name), score, [min(0, score), max_val], steps,
                               model.get("color")))
        if not gauges:
            return self._empty_chart("No bankruptcy model produced a score", height)

        fig = make_subplots(rows=len(gauges), cols=1, specs=[[{"type": "indicator"}] for _ in gauges],
                            vertical_spacing=0.12)
        for row, (name, value, axis_range, steps, color) in enumerate(gauges, start=1):
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=value, title={"text": str(name), "font": {"size": 14}},
                number={"valueformat": ".2f"},
                gauge={"axis": {"range": axis_range, "tickcolor": self.text_color},
                       "bar": {"color": color or COLORS["info"], "thickness": 0.3}, "steps": steps,
                       "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": value}},
            ), row=row, col=1)
        self._apply_layout(fig, "Bankruptcy Risk Assessment", height, width, default_height=230 * len(gauges) + 60)
        if risk_data.get("overall_assessment"):
            fig.add_annotation(x=0.5, y=1.06, xref="paper", yref="paper", showarrow=False,
                               text=f"Overall assessment: {str(risk_data['overall_assessment']).capitalize()}",
                               font=dict(size=14, color=risk_data.get("overall_color") or self.text_color))
        fig.update_layout(margin=dict(t=110))
        return fig

    @_safe_chart("financial health")
    def plot_financial_health_score(self,
                                    health_data: Dict[str, Any],
                                    height: Optional[int] = None,
                                    width: Optional[int] = None) -> go.Figure:
        """Gauges for the overall health score and its profitability/liquidity/solvency components (0-100)."""
        if not isinstance(health_data, Mapping) or not health_data:
            return self._empty_chart("No financial health data available", height)
        overall = _finite(health_data.get("overall_score"))
        if overall is None:
            return self._empty_chart("Financial health score not available", height)
        components = health_data.get("components") or {}
        steps_overall = [{"range": [0, 30], "color": self.colors["danger"]}, {"range": [30, 45], "color": "#ff8c69"},
                         {"range": [45, 60], "color": self.colors["warning"]}, {"range": [60, 75], "color": "#9acd32"},
                         {"range": [75, 100], "color": self.colors["success"]}]
        steps_component = [{"range": [0, 30], "color": self.colors["danger"]},
                           {"range": [30, 70], "color": self.colors["warning"]},
                           {"range": [70, 100], "color": self.colors["success"]}]

        def gauge(value: float, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"axis": {"range": [0, 100], "tickcolor": self.text_color},
                    "bar": {"color": self._get_health_score_color(value), "thickness": 0.3}, "steps": steps,
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": value}}

        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "indicator"}] * 2] * 2,
                            subplot_titles=["Overall Financial Health", "Profitability", "Liquidity", "Solvency"])
        fig.add_trace(go.Indicator(mode="gauge+number", value=overall,
                                   title={"text": f"Rating: {health_data.get('rating', '')}"},
                                   gauge=gauge(overall, steps_overall)), row=1, col=1)
        for name, (row, col) in (("profitability", (1, 2)), ("liquidity", (2, 1)), ("solvency", (2, 2))):
            value = _finite(components.get(name)) if isinstance(components, Mapping) else None
            if value is not None:
                fig.add_trace(go.Indicator(mode="gauge+number", value=value, gauge=gauge(value, steps_component)),
                              row=row, col=col)
        self._apply_layout(fig, "Financial Health Assessment", height, width, default_height=600)
        fig.update_layout(margin=dict(t=110))
        return fig
