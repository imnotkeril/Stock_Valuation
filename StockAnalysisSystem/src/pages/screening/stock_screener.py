import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.pages.common import clean_ticker, get_loader
from StockAnalysisSystem.src.screening.screener_engine import ScreenerEngine

DEFAULT_UNIVERSE = "AAPL, MSFT, GOOGL, AMZN, JPM, XOM, JNJ, KO"
MAX_UNIVERSE = 30
METRICS = ["pe_ratio", "ps_ratio", "roe", "net_margin", "debt_to_equity", "current_ratio"]


def render() -> None:
    st.header("Stock screener")
    raw = st.text_area("Universe (comma separated, max %d)" % MAX_UNIVERSE, value=DEFAULT_UNIVERSE)
    universe = [t for t in (clean_ticker(x) for x in raw.split(",")) if t][:MAX_UNIVERSE]

    cols = st.columns(3)
    metric = cols[0].selectbox("Metric", METRICS)
    operator = cols[1].selectbox("Operator", ["<", ">", "<=", ">="])
    threshold = cols[2].number_input("Value", value=25.0)

    if st.button("Run screen"):
        engine = ScreenerEngine(get_loader())
        with st.spinner(f"Screening {len(universe)} tickers..."):
            results = engine.screen_stocks({metric: {"operator": operator, "value": threshold}},
                                           universe=universe, limit=MAX_UNIVERSE)
        if not results:
            st.info("No tickers matched (or data was unavailable).")
            return
        rows = []
        for item in results:
            flat = {k: v for k, v in item.items() if k != "ratios"}
            for category, values in (item.get("ratios") or {}).items():
                for name, value in values.items():
                    if name == metric:
                        flat[metric] = value
            rows.append(flat)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
