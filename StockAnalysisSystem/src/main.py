"""Streamlit entry point: `streamlit run StockAnalysisSystem/src/main.py` (or `python run.py`)."""
import sys
from pathlib import Path

# Package imports use the `StockAnalysisSystem.src...` prefix, so the project's parent must be importable.
_PARENT = str(Path(__file__).resolve().parents[2])
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import streamlit as st  # noqa: E402

from StockAnalysisSystem.src.pages import home  # noqa: E402
from StockAnalysisSystem.src.pages.common import clean_ticker  # noqa: E402
from StockAnalysisSystem.src.pages.company_analysis import financials, risk, valuation  # noqa: E402
from StockAnalysisSystem.src.pages.comparison import peer_comparison  # noqa: E402
from StockAnalysisSystem.src.pages.screening import stock_screener  # noqa: E402

PAGES = {
    "Overview": lambda t: home.render(t),
    "Financials": lambda t: financials.render(t),
    "Valuation": lambda t: valuation.render(t),
    "Bankruptcy risk": lambda t: risk.render(t),
    "Peer comparison": lambda t: peer_comparison.render(t),
    "Stock screener": lambda t: stock_screener.render(),
}


def main() -> None:
    st.set_page_config(page_title="Stock Analysis System", layout="wide")
    st.sidebar.title("Stock Analysis System")
    ticker = clean_ticker(st.sidebar.text_input("Ticker", value="AAPL"))
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption("Data: yfinance (no key needed). For research only; not investment advice.")

    if not ticker and page != "Stock screener":
        st.info("Enter a ticker in the sidebar.")
        return
    try:
        PAGES[page](ticker)
    except Exception as exc:
        st.error(f"Page failed: {exc}")
        st.exception(exc)


main()
