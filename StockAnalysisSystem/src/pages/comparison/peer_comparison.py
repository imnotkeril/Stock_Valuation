import pandas as pd
import plotly.express as px
import streamlit as st

from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.pages.common import cached_financial_data, clean_ticker
from StockAnalysisSystem.src.utils.helpers import flatten_ratios, has_statements

MAX_PEERS = 8


def render(ticker: str) -> None:
    st.header(f"Peer comparison: {ticker}")
    raw = st.text_input("Peers (comma separated)", value="MSFT, GOOGL")
    peers = [t for t in (clean_ticker(p) for p in raw.split(",")) if t and t != ticker][:MAX_PEERS]
    if not peers:
        st.info("Enter at least one peer ticker.")
        return

    analyzer = FinancialRatioAnalyzer()
    frames = []
    for symbol in [ticker] + peers:
        data = cached_financial_data(symbol)
        if not has_statements(data):
            st.warning(f"{symbol}: no statements, skipped")
            continue
        table = flatten_ratios(analyzer.calculate_ratios(data))
        table["ticker"] = symbol
        frames.append(table)

    if not frames:
        return
    long = pd.concat(frames)
    wide = long.pivot_table(index=["category", "ratio"], columns="ticker", values="value")
    st.dataframe(wide)

    ratio = st.selectbox("Chart ratio", sorted(long["ratio"].unique()))
    st.plotly_chart(px.bar(long[long["ratio"] == ratio], x="ticker", y="value", color="ticker"))
