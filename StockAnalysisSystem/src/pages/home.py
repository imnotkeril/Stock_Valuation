from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from StockAnalysisSystem.src.pages.common import cached_financial_data, cached_prices
from StockAnalysisSystem.src.utils.helpers import format_money


def render(ticker: str) -> None:
    st.header(f"Overview: {ticker}")
    data = cached_financial_data(ticker)
    info = data.get("company_info") or {}
    if not info:
        st.error(f"No data found for {ticker}. Check the ticker symbol.")
        return

    st.subheader(f"{info.get('name', ticker)}  ({info.get('sector', 'n/a')} / {info.get('industry', 'n/a')})")
    cols = st.columns(4)
    cols[0].metric("Price", f"{info['current_price']:,.2f}" if info.get("current_price") else "n/a")
    cols[1].metric("Market cap", format_money(info.get("market_cap")))
    cols[2].metric("P/E", f"{info['pe_ratio']:.1f}" if info.get("pe_ratio") else "n/a")
    cols[3].metric("Beta", f"{info['beta']:.2f}" if info.get("beta") else "n/a")

    years = st.select_slider("Price history", options=[1, 2, 5, 10], value=2, format_func=lambda y: f"{y}y")
    end = date.today()
    start = end - timedelta(days=365 * years)
    prices = cached_prices(ticker, start.isoformat(), end.isoformat())
    if prices is None or prices.empty:
        st.warning("No price history available.")
    else:
        fig = go.Figure(go.Scatter(x=prices.index, y=prices["Close"], mode="lines", name=ticker))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Close")
        st.plotly_chart(fig)

    if info.get("description"):
        with st.expander("Business description"):
            st.write(info["description"])
