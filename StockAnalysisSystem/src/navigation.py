"""
Application navigation state shared by main.py and the pages.

main.py renders the sidebar page selector bound to ``st.session_state[NAV_KEY]``.
Pages can jump elsewhere with a button callback::

    st.button("Analyze", on_click=go_to, args=("Company Analysis", "AAPL"))

The currently selected company ticker (if any) is kept in
``st.session_state[ACTIVE_TICKER_KEY]`` so pages can use it as a default.
"""

from typing import Optional

import streamlit as st

NAV_KEY = "nav_page"
ACTIVE_TICKER_KEY = "active_ticker"

HOME = "Home"
COMPANY_ANALYSIS = "Company Analysis"
PEER_COMPARISON = "Peer Comparison"
SECTOR_ANALYSIS = "Sector Analysis"
STOCK_SCREENER = "Stock Screener"
VALUATION_FILTER = "Valuation Filter"
FORECASTING = "Forecasting"
REPORTS = "Reports"

#: Page name -> sidebar icon (Material Symbols shortcode)
PAGES = {
    HOME: ":material/home:",
    COMPANY_ANALYSIS: ":material/query_stats:",
    PEER_COMPARISON: ":material/compare_arrows:",
    SECTOR_ANALYSIS: ":material/donut_large:",
    STOCK_SCREENER: ":material/filter_alt:",
    VALUATION_FILTER: ":material/price_check:",
    FORECASTING: ":material/trending_up:",
    REPORTS: ":material/description:",
}


def go_to(page: str, ticker: Optional[str] = None) -> None:
    """Switch to ``page`` (optionally selecting ``ticker``). Use as an ``on_click`` callback."""
    if page not in PAGES:
        raise ValueError(f"Unknown page: {page}")
    st.session_state[NAV_KEY] = page
    if ticker:
        st.session_state[ACTIVE_TICKER_KEY] = ticker.upper()


def get_active_ticker(default: Optional[str] = None) -> Optional[str]:
    """Ticker currently selected in Company Analysis (or ``default``)."""
    return st.session_state.get(ACTIVE_TICKER_KEY) or default
