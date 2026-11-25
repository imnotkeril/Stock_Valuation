"""Company Analysis - News tab: headlines and sentiment from Finnhub / Alpha Vantage (optional keys)."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.utils import news_sentiment as ns

DAYS = 30


def render_company_news(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                        price_data: pd.DataFrame, **_: Any) -> None:
    st.subheader("News and sentiment")
    providers = ns.available_providers()
    if not providers:
        st.info(f"Add `{ns.FINNHUB_ENV}` (free at finnhub.io) and/or `{ns.ALPHA_VANTAGE_ENV}` (news sentiment "
                "needs a premium Alpha Vantage plan) to `.env` to load company news with sentiment scores.",
                icon=":material/info:")
        return
    flag = f"news_loaded_{ticker}"
    if not st.session_state.get(flag):
        if st.button("Load news", key=f"news_load_{ticker}", icon=":material/newspaper:"):
            st.session_state[flag] = True
        else:
            st.caption(f"Sources: {', '.join(providers)}; last {DAYS} days.")
            return
    with st.spinner("Loading news..."):
        news = ns.get_news(ticker, DAYS)
    if news.empty:
        st.warning(f"No news found for {ticker} in the last {DAYS} days (or the provider limit was reached).")
        return

    weights = news["relevance"].fillna(1.0).clip(lower=0.05)
    average = float((news["sentiment"] * weights).sum() / weights.sum())
    cols = st.columns(4)
    cols[0].metric("Sentiment", ns.sentiment_label(average), f"{average:+.2f}", delta_color="normal")
    cols[1].metric("Articles", len(news))
    cols[2].metric("Positive", int((news["sentiment"] >= 0.15).sum()))
    cols[3].metric("Negative", int((news["sentiment"] <= -0.15).sum()))

    daily = ns.daily_sentiment(news)
    if len(daily) > 1:
        st.markdown("**Daily average sentiment**")
        st.bar_chart(daily["sentiment"])

    table = news.assign(sentiment_label=news["sentiment"].map(ns.sentiment_label))
    st.dataframe(table[["published", "source", "headline", "sentiment", "sentiment_label", "provider", "url"]],
                 width="stretch", hide_index=True,
                 column_config={"published": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
                                "sentiment": st.column_config.NumberColumn("Score", format="%.2f"),
                                "sentiment_label": "Tone",
                                "url": st.column_config.LinkColumn("Link", display_text="Open")})
    st.caption("Scores range from -1 (bearish) to +1 (bullish). Alpha Vantage provides per-ticker scores; Finnhub "
               "headlines are scored with a finance word list. Sentiment is context, not a trading signal.")

    peers = ns.finnhub_peers(ticker)
    if peers:
        st.caption(f"Finnhub peers: {', '.join(peers[:12])}")
