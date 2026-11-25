"""
Home page.

Entry point: :func:`render_home_page` (no arguments).

Landing page with a company search, quick-start buttons that jump to other
pages through :func:`navigation.go_to`, an overview of the features and a
sector ETF performance snapshot (shown only when market data is available).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.industry.universe import (
    MARKET_BENCHMARK,
    SECTOR_ETFS,
    lookup_company,
    search_universe,
)
from StockAnalysisSystem.src.navigation import (
    COMPANY_ANALYSIS,
    FORECASTING,
    PAGES,
    PEER_COMPARISON,
    REPORTS,
    SECTOR_ANALYSIS,
    STOCK_SCREENER,
    VALUATION_FILTER,
    get_active_ticker,
    go_to,
)
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.export import price_performance

logger = logging.getLogger(__name__)

#: Features shown on the landing page: (page, description)
FEATURES = [
    (COMPANY_ANALYSIS, "Financial statements, ratios, bankruptcy risk and a sector-specific valuation "
                       "(DCF, multiples, dividend, asset and sector models) for any company."),
    (PEER_COMPARISON, "Compare a company with its sector peers on growth, profitability, valuation and "
                      "price performance."),
    (SECTOR_ANALYSIS, "Sector ETF performance against the market and the fundamentals of each sector's "
                      "leading companies."),
    (STOCK_SCREENER, "Filter the company universe by sector, size, profitability and valuation metrics."),
    (VALUATION_FILTER, "Rank companies by the upside of their model fair value over the market price."),
    (FORECASTING, "Price forecasts (ARIMA, exponential smoothing, ML ensembles), fundamentals forecasts and "
                  "bull/base/bear scenarios."),
    (REPORTS, "Generate company, valuation, comparison and sector reports as PDF, Excel, CSV or JSON."),
]

#: Quick-start companies spanning the sector-specific valuation models
QUICK_START = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "AMZN", "NEE", "PLD"]

SNAPSHOT_PERIODS = {"1M": "return_1m", "3M": "return_3m", "6M": "return_6m", "1Y": "return_1y"}
_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


class _NoData(Exception):
    """Raised inside cached loaders so empty results are not cached."""


@st.cache_resource(show_spinner=False)
def _get_data_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(ttl=3600, show_spinner=False)
def _search(query: str) -> List[Dict[str, Any]]:
    try:
        results = _get_data_loader().search_tickers(query, limit=6)
    except Exception:
        logger.exception("Ticker search failed for %r", query)
        results = []
    return results or search_universe(query, limit=6)


@st.cache_data(ttl=3600, show_spinner=False)
def sector_snapshot() -> pd.DataFrame:
    """
    Trailing performance of every sector ETF and the market benchmark.

    Returns a DataFrame indexed by ETF ticker with ``Sector``, ``Last``,
    ``As of`` and the ``return_*`` columns of
    :func:`~StockAnalysisSystem.src.utils.export.price_performance`. Raises
    :class:`_NoData` (so nothing is cached) when no prices are available.
    """
    loader = _get_data_loader()
    end = datetime.now()
    start = (end - timedelta(days=400)).strftime("%Y-%m-%d")
    rows = {}
    for sector, etf in list(SECTOR_ETFS.items()) + [("Market (S&P 500)", MARKET_BENCHMARK)]:
        try:
            prices = loader.get_historical_prices(etf, start, end.strftime("%Y-%m-%d"))
        except Exception:
            logger.exception("Could not load %s prices", etf)
            continue
        perf = price_performance(prices)
        if perf.get("last_price") is None:
            continue
        rows[etf] = {"Sector": sector, "Last": perf["last_price"],
                     "As of": pd.Timestamp(prices.index.max()).strftime("%Y-%m-%d"),
                     **{key: perf.get(key) for key in SNAPSHOT_PERIODS.values()}}
    if not rows:
        raise _NoData("sector ETFs")
    return pd.DataFrame.from_dict(rows, orient="index")


def _load_snapshot() -> pd.DataFrame:
    try:
        return sector_snapshot()
    except _NoData:
        return pd.DataFrame()
    except Exception:
        logger.exception("Sector snapshot failed")
        return pd.DataFrame()


def _snapshot_chart(data: pd.DataFrame, column: str, label: str) -> go.Figure:
    sectors = data[data.index != MARKET_BENCHMARK].dropna(subset=[column]).sort_values(column)
    values = sectors[column] * 100
    fig = go.Figure(go.Bar(
        x=values.values, y=[f"{s} ({t})" for t, s in zip(sectors.index, sectors["Sector"])], orientation="h",
        marker_color=[COLORS["primary"] if v >= 0 else COLORS["accent"] for v in values.values],
        text=[f"{v:+.1f}%" for v in values.values], textposition="auto"))
    if MARKET_BENCHMARK in data.index and pd.notna(data.loc[MARKET_BENCHMARK, column]):
        market = float(data.loc[MARKET_BENCHMARK, column]) * 100
        fig.add_vline(x=market, line={"color": COLORS["warning"], "dash": "dash"},
                      annotation_text=f"{MARKET_BENCHMARK} {market:+.1f}%", annotation_position="top")
    fig.update_layout(template="plotly_dark", paper_bgcolor=VIZ_SETTINGS["background"],
                      plot_bgcolor=VIZ_SETTINGS["background"], height=420, showlegend=False,
                      title=f"Sector ETF returns - {label}", margin={"l": 10, "r": 10, "t": 50, "b": 30},
                      xaxis_title="Return (%)", font={"color": VIZ_SETTINGS["text_color"]})
    fig.update_xaxes(gridcolor=VIZ_SETTINGS["grid_color"])
    return fig


def _render_search() -> None:
    st.subheader("Find a company")
    query = st.text_input("Search by ticker or company name", key="home_search",
                          placeholder="e.g. AAPL, Microsoft, bank ...")
    query = str(query or "").strip()
    if not query:
        active = get_active_ticker()
        if active:
            st.button(f"Continue with {active}", key="home_continue", on_click=go_to,
                      args=(COMPANY_ANALYSIS, active), icon=":material/arrow_forward:")
        return
    results = _search(query)
    symbols = [str(r.get("symbol", "")).upper() for r in results]
    for result in results:
        symbol = str(result.get("symbol", "")).upper()
        if not symbol:
            continue
        name = result.get("name") or symbol
        sector = result.get("sector")
        st.button(f"{symbol} - {name}" + (f" ({sector})" if sector else ""), key=f"home_result_{symbol}",
                  on_click=go_to, args=(COMPANY_ANALYSIS, symbol))
    candidate = query.upper()
    if _TICKER_PATTERN.match(candidate) and candidate not in symbols:
        st.button(f"Analyze {candidate}", key="home_result_raw", on_click=go_to, args=(COMPANY_ANALYSIS, candidate))
    elif not results:
        st.caption("No matching companies found.")


def _render_quick_start() -> None:
    st.subheader("Quick start")
    st.caption("Each company uses a different sector-specific valuation model.")
    columns = st.columns(4)
    for i, symbol in enumerate(QUICK_START):
        record = lookup_company(symbol) or {"name": symbol, "sector": ""}
        with columns[i % 4]:
            st.button(f"{symbol}", key=f"home_pick_{symbol}", on_click=go_to, args=(COMPANY_ANALYSIS, symbol),
                      help=f"{record['name']} - {record['sector']}")


def _render_features() -> None:
    st.subheader("What you can do")
    columns = st.columns(2)
    for i, (page, description) in enumerate(FEATURES):
        with columns[i % 2].container(border=True):
            st.markdown(f"**{PAGES[page]} {page}**")
            st.caption(description)
            st.button("Open", key=f"home_open_{re.sub(r'[^a-z0-9]+', '_', page.lower())}", on_click=go_to,
                      args=(page,))


def _render_snapshot() -> None:
    st.subheader("Sector snapshot")
    with st.spinner("Loading sector ETF prices ..."):
        data = _load_snapshot()
    if data.empty:
        st.info("Sector ETF prices are not available right now (market data needs a network connection). "
                "All analysis pages remain usable with cached data.")
        return
    label = st.segmented_control("Period", list(SNAPSHOT_PERIODS), default="3M", key="home_snapshot_period") or "3M"
    column = SNAPSHOT_PERIODS[label]
    if data[column].isna().all():
        st.caption(f"Not enough history for {label} returns.")
        return
    st.plotly_chart(_snapshot_chart(data, column, label), key="home_snapshot_chart")
    st.caption(f"Prices as of {data['As of'].max()}. Sector ETFs: SPDR Select Sector funds; market: "
               f"{MARKET_BENCHMARK}.")
    table = data[["Sector", "Last"] + list(SNAPSHOT_PERIODS.values())].copy()
    table.columns = ["Sector", "Last"] + list(SNAPSHOT_PERIODS)
    st.dataframe(table.style.format({"Last": "${:,.2f}", **{p: "{:+.1%}" for p in SNAPSHOT_PERIODS}},
                                    na_rep="N/A"), key="home_snapshot_table")
    st.button("Open Sector Analysis", key="home_open_sector_snapshot", on_click=go_to, args=(SECTOR_ANALYSIS,))


def render_home_page() -> None:
    """Render the landing page."""
    st.title("Stock Analysis System")
    st.markdown("Sector-aware fundamental analysis, valuation, forecasting and reporting for US equities.")
    left, right = st.columns([3, 2], gap="large")
    sections = ((left, _render_features), (right, _render_search), (right, _render_quick_start))
    for container, render in sections:
        with container:
            _safe(render)
    st.divider()
    _safe(_render_snapshot)
    st.caption("Data: Yahoo Finance / Alpha Vantage (cached locally). For informational purposes only - "
               "not investment advice.")


def _safe(render) -> None:
    try:
        render()
    except Exception:  # never show a traceback to users
        logger.exception("Home page section %s failed", getattr(render, "__name__", render))
        st.warning("This section is temporarily unavailable.")


if __name__ == "__main__":  # pragma: no cover
    render_home_page()
