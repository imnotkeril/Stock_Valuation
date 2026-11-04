"""
Company overview tab.

Rendered by the app shell below its company header (name, ticker, sector,
price metrics, market cap), so this page starts directly with sections:
price chart, key statistics, business description, financial snapshot, key
ratios versus sector benchmarks, performance versus the sector ETF and
technical indicators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.industry.universe import SECTOR_ETFS
from StockAnalysisSystem.src.pages.company_analysis.charts import (
    benchmark_bar_chart,
    financial_bars_chart,
    is_empty_figure,
    price_chart,
    rebased_performance_chart,
    show_chart,
)
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    benchmark_table,
    canonical_sector,
    close_series,
    fmt_money,
    fmt_number,
    fmt_pct,
    fmt_price,
    fmt_ratio,
    fmt_shares,
    free_cash_flow_series,
    get_company_ratios,
    get_sector_benchmarks,
    get_statement,
    item_series,
    market_snapshot,
    normalized_prices,
)
from StockAnalysisSystem.src.utils.financial_data import get_latest, safe_divide, to_float

logger = logging.getLogger("company_overview")

KEY = "ov_"

#: Price chart ranges -> pandas offset (None = full history, "YTD" handled separately)
RANGES: Dict[str, Optional[pd.DateOffset]] = {
    "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
    "YTD": None, "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5),
    "Max": None,
}

#: Trailing return periods
RETURN_PERIODS: Dict[str, Optional[pd.DateOffset]] = {
    "1W": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6), "YTD": None, "1Y": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5),
}

OVERVIEW_RATIOS = ["pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "gross_margin", "operating_margin",
                   "net_margin", "roe", "roa", "current_ratio", "debt_to_equity", "interest_coverage"]


def render_company_overview(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                            price_data: pd.DataFrame, sector_data: Optional[pd.DataFrame] = None) -> None:
    """
    Render the company overview tab.

    Args:
        ticker: Company ticker symbol.
        financial_data: Canonical financial data bundle (see ``utils.financial_data``).
        company_info: Normalised company information dict.
        price_data: Daily OHLCV price history.
        sector_data: Daily price history of the sector ETF (optional).
    """
    company_info = company_info or {}
    financial_data = financial_data or {}
    sector = canonical_sector(company_info, financial_data)
    prices = normalized_prices(price_data)
    sector_prices = normalized_prices(sector_data) if isinstance(sector_data, pd.DataFrame) else pd.DataFrame()

    _render_price_chart(ticker, company_info, prices)
    _render_key_statistics(financial_data, company_info, prices, sector_prices, sector)
    _render_business_description(company_info)
    _render_financial_snapshot(financial_data)
    _render_key_ratios(financial_data, company_info, price_data, sector)
    _render_sector_performance(ticker, prices, sector_prices, sector)
    _render_technical_indicators(prices)


# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------
def _slice_range(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    if df.empty or choice == "Max":
        return df
    end = df.index[-1]
    if choice == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    else:
        start = end - RANGES[choice]
    sliced = df[df.index >= start]
    return sliced if len(sliced) >= 2 else df


def _render_price_chart(ticker: str, company_info: Dict[str, Any], prices: pd.DataFrame) -> None:
    st.subheader("Price History")
    if prices.empty:
        st.info("No price history is available for this company.")
        return

    currency = company_info.get("currency") or "USD"
    col1, col2, col3, col4 = st.columns([3, 2, 3, 1.4])
    with col1:
        range_choice = st.radio("Range", list(RANGES), index=4, horizontal=True, key=f"{KEY}range")
    with col2:
        has_ohlc = all(c in prices.columns for c in ("Open", "High", "Low"))
        chart_options = ["Candlestick", "Line"] if has_ohlc else ["Line"]
        chart_type = st.radio("Chart type", chart_options, horizontal=True, key=f"{KEY}chart_type")
    with col3:
        ma_periods = st.multiselect("Moving averages (days)", [20, 50, 100, 200], default=[50, 200],
                                    key=f"{KEY}ma")
    with col4:
        show_volume = st.toggle("Volume", value=True, key=f"{KEY}volume")

    # Moving averages use the full history so long windows are defined at the range start
    moving_averages = {p: prices["Close"].rolling(window=p).mean() for p in sorted(ma_periods)}
    view = _slice_range(prices, range_choice)
    name = company_info.get("name") or ticker
    fig = price_chart(view, title=f"{name} ({ticker}) — {range_choice} price", chart_type=chart_type,
                      moving_averages=moving_averages, show_volume=show_volume, currency=currency)
    show_chart(fig, key=f"{KEY}price_chart")


# ---------------------------------------------------------------------------
# Key statistics
# ---------------------------------------------------------------------------
def trailing_return(close: pd.Series, period: str) -> Optional[float]:
    """Total price return over ``period`` (key of RETURN_PERIODS) or None if history is too short."""
    if close.empty or len(close) < 2:
        return None
    end_date = close.index[-1]
    if period == "YTD":
        prior = close[close.index < pd.Timestamp(year=end_date.year, month=1, day=1)]
        start_value = prior.iloc[-1] if not prior.empty else None
    else:
        start_date = end_date - RETURN_PERIODS[period]
        if close.index[0] > start_date + pd.Timedelta(days=5):
            return None
        start_value = close[close.index <= start_date].iloc[-1] if (close.index <= start_date).any() \
            else close.iloc[0]
    return safe_divide(close.iloc[-1], start_value, None) - 1 if start_value else None


def annualized_volatility(close: pd.Series, window: int = 252) -> Optional[float]:
    returns = close.pct_change().dropna().tail(window)
    return float(returns.std() * np.sqrt(252)) if len(returns) > 20 else None


def _render_key_statistics(financial_data: Dict[str, Any], company_info: Dict[str, Any], prices: pd.DataFrame,
                           sector_prices: pd.DataFrame, sector: str) -> None:
    st.subheader("Key Statistics")
    close = close_series(prices)
    income = get_statement(financial_data, "income_statement")
    snap = market_snapshot(financial_data, company_info, prices)

    last_year = close[close.index >= close.index[-1] - pd.DateOffset(years=1)] if not close.empty else close
    high_52w = float(last_year.max()) if not last_year.empty else to_float(company_info.get("fifty_two_week_high"))
    low_52w = float(last_year.min()) if not last_year.empty else to_float(company_info.get("fifty_two_week_low"))
    last = float(close.iloc[-1]) if not close.empty else snap["price"]
    from_high = safe_divide(last, high_52w) - 1 if last and high_52w else None

    avg_volume = None
    if "Volume" in prices.columns and not prices.empty:
        avg_volume = to_float(prices["Volume"].tail(63).mean())
    avg_volume = avg_volume or to_float(company_info.get("average_volume"))

    eps = get_latest(income, "diluted_eps")
    if eps is None:
        eps = safe_divide(get_latest(income, "net_income"), snap["shares"])
    pe = to_float(company_info.get("pe_ratio"))
    if pe is None and eps and eps > 0 and last:
        pe = last / eps

    stats = [
        ("52-Week High", fmt_price(high_52w), None),
        ("52-Week Low", fmt_price(low_52w), None),
        ("From 52-Week High", fmt_pct(from_high, signed=True), "Distance of the last close from the 52-week high."),
        ("Avg. Volume (3M)", fmt_shares(avg_volume), "Average daily shares traded over ~63 sessions."),
        ("Beta", fmt_number(snap["beta"]), "Sensitivity of the stock to market moves (1.0 = market)."),
        ("Dividend Yield", fmt_pct(snap["dividend_yield"], decimals=2), None),
        ("P/E (TTM)", fmt_ratio(pe), "Price divided by trailing diluted EPS."),
        ("Forward P/E", fmt_ratio(company_info.get("forward_pe")), "Price divided by expected next-year EPS."),
        ("Diluted EPS", fmt_price(eps), "Latest fiscal-year diluted earnings per share."),
        ("Shares Outstanding", fmt_shares(snap["shares"]), None),
        ("Volatility (1Y, ann.)", fmt_pct(annualized_volatility(close)),
         "Annualised standard deviation of daily returns over the last year."),
        ("Employees", f"{int(company_info['employees']):,}" if to_float(company_info.get("employees")) else "N/A",
         None),
    ]
    for start in range(0, len(stats), 4):
        cols = st.columns(4)
        for col, (label, value, help_text) in zip(cols, stats[start:start + 4]):
            col.metric(label, value, help=help_text)

    if close.empty:
        return
    sector_close = close_series(sector_prices)
    etf = SECTOR_ETFS.get(sector)
    rows = []
    for period in RETURN_PERIODS:
        stock_ret = trailing_return(close, period)
        row = {"Period": period, "Stock": fmt_pct(stock_ret, signed=True)}
        if not sector_close.empty:
            etf_ret = trailing_return(sector_close, period)
            row[f"Sector ETF ({etf or 'ETF'})"] = fmt_pct(etf_ret, signed=True)
            diff = stock_ret - etf_ret if stock_ret is not None and etf_ret is not None else None
            row["Difference (pp)"] = f"{diff * 100:+.1f}" if diff is not None else "N/A"
        if stock_ret is not None:
            rows.append(row)
    if rows:
        st.markdown("**Trailing price returns**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------------------
# Business description
# ---------------------------------------------------------------------------
def _render_business_description(company_info: Dict[str, Any]) -> None:
    st.subheader("Business Description")
    description = (company_info.get("description") or company_info.get("longBusinessSummary") or "").strip()
    if description:
        if len(description) > 700:
            cut = description[:700].rsplit(" ", 1)[0]
            st.write(cut + " …")
            with st.expander("Read full description"):
                st.write(description)
        else:
            st.write(description)
    else:
        st.info("No business description is available for this company.")

    facts = [
        ("Country", company_info.get("country")),
        ("Exchange", company_info.get("exchange")),
        ("Currency", company_info.get("currency")),
        ("Website", company_info.get("website")),
    ]
    facts = [(label, value) for label, value in facts if value]
    if facts:
        st.markdown(" · ".join(f"**{label}:** {value}" for label, value in facts))


# ---------------------------------------------------------------------------
# Financial snapshot
# ---------------------------------------------------------------------------
def _latest_and_change(series: pd.Series) -> tuple:
    if series.empty:
        return None, None
    latest = float(series.iloc[-1])
    if len(series) < 2 or series.iloc[-2] == 0:
        return latest, None
    return latest, (latest - series.iloc[-2]) / abs(series.iloc[-2])


def _render_financial_snapshot(financial_data: Dict[str, Any]) -> None:
    st.subheader("Financial Snapshot")
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    cash_flow = get_statement(financial_data, "cash_flow")
    if income.empty and balance.empty and cash_flow.empty:
        st.info("Financial statements are not available for this company.")
        return

    revenue = item_series(income, "revenue")
    net_income = item_series(income, "net_income")
    fcf = free_cash_flow_series(cash_flow)
    cash = item_series(balance, "cash_and_short_term_investments")
    debt = item_series(balance, "total_debt")

    cols = st.columns(5)
    for col, (label, series) in zip(cols, [("Revenue", revenue), ("Net Income", net_income),
                                           ("Free Cash Flow", fcf), ("Cash & ST Investments", cash),
                                           ("Total Debt", debt)]):
        latest, change = _latest_and_change(series)
        col.metric(label, fmt_money(latest), delta=fmt_pct(change, signed=True) if change is not None else None,
                   delta_color="inverse" if label == "Total Debt" else "normal",
                   help="Latest fiscal year; delta is the change versus the prior year.")

    if revenue.empty and net_income.empty:
        st.info("Revenue and net income history are not available.")
        return
    margin = net_income.reindex(revenue.index) / revenue.replace(0, np.nan) if not revenue.empty \
        else pd.Series(dtype=float)
    fig = financial_bars_chart({"Revenue": revenue, "Net Income": net_income, "Free Cash Flow": fcf},
                               title="Revenue, earnings and free cash flow",
                               line=("Net margin", margin), line_axis_title="Net margin")
    show_chart(fig, key=f"{KEY}financial_snapshot")


# ---------------------------------------------------------------------------
# Key ratios vs sector benchmarks
# ---------------------------------------------------------------------------
def _render_key_ratios(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                       sector: str) -> None:
    st.subheader("Key Ratios vs Sector Benchmark")
    ratios = get_company_ratios(financial_data, company_info, price_data)
    if not any(ratios.values()):
        st.info("Not enough financial data to calculate key ratios.")
        return
    benchmarks = get_sector_benchmarks(sector)
    table = benchmark_table(ratios, benchmarks, OVERVIEW_RATIOS)
    if table.empty:
        st.info("Not enough financial data to calculate key ratios.")
        return
    label = sector if sector != "Unknown" else "general market"
    st.caption(f"Benchmarks: typical {label} values. Lower is better for valuation multiples and leverage; "
               "higher is better for margins, returns, liquidity and coverage.")
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.dataframe(table, hide_index=True)
    with col2:
        fig = benchmark_bar_chart(ratios, benchmarks, OVERVIEW_RATIOS, "Premium / discount to sector")
        if is_empty_figure(fig):
            st.info("No sector benchmarks are available for these ratios.")
        else:
            show_chart(fig, key=f"{KEY}ratio_vs_sector")


# ---------------------------------------------------------------------------
# Performance vs sector ETF
# ---------------------------------------------------------------------------
def _render_sector_performance(ticker: str, prices: pd.DataFrame, sector_prices: pd.DataFrame,
                               sector: str) -> None:
    etf = SECTOR_ETFS.get(sector)
    st.subheader(f"Performance vs Sector ETF{f' ({etf})' if etf else ''}")
    stock = close_series(prices)
    bench = close_series(sector_prices)
    if stock.empty:
        st.info("No price history is available for this company.")
        return
    if bench.empty:
        st.info("Sector ETF price data is not available, so the sector comparison is skipped.")
        return

    choice = st.selectbox("Comparison period", ["3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max"], index=3,
                          key=f"{KEY}perf_range")
    joined = pd.concat([stock.rename("stock"), bench.rename("etf")], axis=1, join="inner").dropna()
    joined = _slice_range(joined.rename(columns={"stock": "Close"}), choice).rename(columns={"Close": "stock"})
    if len(joined) < 5:
        st.info("The company and sector ETF price histories do not overlap enough for a comparison.")
        return

    fig = rebased_performance_chart({ticker: joined["stock"], etf or "Sector ETF": joined["etf"]},
                                    title=f"Growth of 100 — {ticker} vs {etf or 'sector ETF'} ({choice})")
    show_chart(fig, key=f"{KEY}sector_performance")

    stock_ret = joined["stock"].iloc[-1] / joined["stock"].iloc[0] - 1
    etf_ret = joined["etf"].iloc[-1] / joined["etf"].iloc[0] - 1
    daily = joined.pct_change().dropna()
    correlation = to_float(daily["stock"].corr(daily["etf"])) if len(daily) > 5 else None
    beta = safe_divide(daily["stock"].cov(daily["etf"]), daily["etf"].var()) if len(daily) > 5 else None
    cols = st.columns(5)
    cols[0].metric(f"{ticker} return", fmt_pct(stock_ret, signed=True))
    cols[1].metric(f"{etf or 'ETF'} return", fmt_pct(etf_ret, signed=True))
    excess = stock_ret - etf_ret
    cols[2].metric("Excess return", f"{excess * 100:+.1f} pp", delta=f"{excess * 100:+.1f} pp",
                   help="Stock return minus sector ETF return over the period (percentage points).")
    cols[3].metric("Correlation", fmt_number(correlation), help="Correlation of daily returns.")
    cols[4].metric("Beta vs sector", fmt_number(beta),
                   help="Sensitivity of the stock's daily returns to the sector ETF's daily returns.")


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def technical_indicators(close: pd.Series) -> Dict[str, Optional[float]]:
    """RSI(14), MACD(12, 26, 9), Bollinger %B(20, 2) and the 50/200-day moving averages."""
    result: Dict[str, Optional[float]] = {"rsi": None, "macd": None, "signal": None, "bb_pct": None,
                                          "ma50": None, "ma200": None}
    if len(close) < 30:
        return result
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if pd.notna(last_gain) and pd.notna(last_loss):
        result["rsi"] = 100.0 if last_loss == 0 else float(100 - 100 / (1 + last_gain / last_loss))
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    result["macd"] = float(macd.iloc[-1])
    result["signal"] = float(macd.ewm(span=9, adjust=False).mean().iloc[-1])
    sma20, std20 = close.rolling(20).mean().iloc[-1], close.rolling(20).std().iloc[-1]
    if pd.notna(std20) and std20 > 0:
        result["bb_pct"] = float((close.iloc[-1] - (sma20 - 2 * std20)) / (4 * std20))
    if len(close) >= 50:
        result["ma50"] = float(close.rolling(50).mean().iloc[-1])
    if len(close) >= 200:
        result["ma200"] = float(close.rolling(200).mean().iloc[-1])
    return result


def _render_technical_indicators(prices: pd.DataFrame) -> None:
    close = close_series(prices)
    if len(close) < 30:
        return
    st.subheader("Technical Indicators")
    ind = technical_indicators(close)
    cols = st.columns(4)

    rsi = ind["rsi"]
    rsi_status = "N/A" if rsi is None else ("Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
    cols[0].metric("RSI (14)", fmt_number(rsi, 1), help="Relative Strength Index: >70 overbought, <30 oversold.")
    _status(cols[0], rsi_status, NEGATIVE if rsi_status == "Overbought" else
            POSITIVE if rsi_status == "Oversold" else NEUTRAL)

    macd, signal = ind["macd"], ind["signal"]
    bullish = macd is not None and signal is not None and macd > signal
    cols[1].metric("MACD (12/26/9)", fmt_number(macd), help="MACD line minus its 9-day signal line "
                                                            "indicates momentum direction.")
    _status(cols[1], "Bullish momentum" if bullish else "Bearish momentum", POSITIVE if bullish else NEGATIVE)

    bb = ind["bb_pct"]
    bb_status = "N/A" if bb is None else ("Near upper band" if bb > 0.8 else "Near lower band" if bb < 0.2
                                          else "Mid-range")
    cols[2].metric("Bollinger %B (20, 2σ)", fmt_pct(bb, 0), help="Position of the price within the Bollinger "
                                                                   "bands (0% = lower band, 100% = upper band).")
    _status(cols[2], bb_status, NEGATIVE if bb_status == "Near upper band" else
            POSITIVE if bb_status == "Near lower band" else NEUTRAL)

    ma50, ma200 = ind["ma50"], ind["ma200"]
    if ma50 is not None and ma200 is not None:
        trend_up = ma50 > ma200
        cols[3].metric("50 / 200-day MA", f"{fmt_price(ma50)} / {fmt_price(ma200)}",
                       help="A 50-day average above the 200-day average (golden cross) signals an uptrend.")
        _status(cols[3], "Uptrend (golden cross)" if trend_up else "Downtrend (death cross)",
                POSITIVE if trend_up else NEGATIVE)
    elif ma50 is not None:
        above = close.iloc[-1] > ma50
        cols[3].metric("50-day MA", fmt_price(ma50))
        _status(cols[3], "Price above 50-day MA" if above else "Price below 50-day MA",
                POSITIVE if above else NEGATIVE)
    st.caption("Technical indicators describe recent price behaviour only and are not a valuation signal.")


def _status(container: Any, text: str, color: str) -> None:
    container.markdown(f"<span style='color:{color};font-size:0.9rem'>{text}</span>", unsafe_allow_html=True)


__all__: List[str] = ["render_company_overview", "trailing_return", "technical_indicators",
                      "annualized_volatility"]
