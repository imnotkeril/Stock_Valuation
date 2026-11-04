"""
Stock Analysis System — Streamlit entry point.

Run with ``python StockAnalysisSystem/run.py`` or
``streamlit run StockAnalysisSystem/src/main.py``.

This module is the application shell: page configuration, styling, sidebar
navigation, company selection and data loading. Every page lives in
``StockAnalysisSystem/src/pages`` and is imported lazily, so a failure in
one page never takes the whole application down.
"""

import importlib
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from StockAnalysisSystem.src import config, navigation as nav  # noqa: E402
from StockAnalysisSystem.src.config import COLORS  # noqa: E402
from StockAnalysisSystem.src.industry.universe import get_sector_peers, lookup_company  # noqa: E402
from StockAnalysisSystem.src.utils.data_loader import DataLoader  # noqa: E402
from StockAnalysisSystem.src.utils.financial_data import (  # noqa: E402
    build_financial_data, format_large_number, to_float,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')

APP_TITLE = "Stock Analysis System"

#: Quick-pick companies in the sidebar (ticker -> name)
POPULAR_TICKERS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.", "NVDA": "NVIDIA Corporation", "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.", "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil Corporation", "NEE": "NextEra Energy, Inc.", "PLD": "Prologis, Inc.",
    "PG": "The Procter & Gamble Company", "CAT": "Caterpillar Inc.", "VZ": "Verizon Communications Inc.",
}

#: Price history choices: label -> days (None = custom range)
DATE_RANGES: Dict[str, Optional[int]] = {
    "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365,
    "3 Years": 3 * 365, "5 Years": 5 * 365, "Custom": None,
}
DEFAULT_RANGE = "1 Year"

#: Page name -> (module, function) rendered for that page (Company Analysis is built in)
PAGE_RENDERERS = {
    nav.HOME: ("StockAnalysisSystem.src.pages.home", "render_home_page"),
    nav.PEER_COMPARISON: ("StockAnalysisSystem.src.pages.comparison.peer_comparison", "run_peer_comparison_page"),
    nav.SECTOR_ANALYSIS: ("StockAnalysisSystem.src.pages.comparison.sector_analysis", "run_sector_analysis_page"),
    nav.STOCK_SCREENER: ("StockAnalysisSystem.src.pages.screening.stock_screener", "run_stock_screener_page"),
    nav.VALUATION_FILTER: ("StockAnalysisSystem.src.pages.screening.valuation_filter", "run_valuation_filter_page"),
    nav.FORECASTING: ("StockAnalysisSystem.src.pages.forecasting", "run_forecasting_page"),
    nav.REPORTS: ("StockAnalysisSystem.src.pages.reports", "run_reports_page"),
}

#: Company analysis tabs: label -> (module, function)
COMPANY_TABS = {
    "Overview": ("StockAnalysisSystem.src.pages.company_analysis.overview", "render_company_overview"),
    "Financials": ("StockAnalysisSystem.src.pages.company_analysis.financials", "render_company_financials"),
    "Valuation": ("StockAnalysisSystem.src.pages.company_analysis.valuation", "render_company_valuation"),
    "Risk": ("StockAnalysisSystem.src.pages.company_analysis.risk", "render_company_risk"),
}

REQUEST_KEY = "company_request"

CUSTOM_CSS = f"""
<style>
    /* Buttons */
    .stButton button[kind="primary"] {{
        background-color: {COLORS['primary']};
        color: #121212;
        border: none;
        transition: all 0.2s ease;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: {COLORS['info']};
        color: #121212;
    }}

    /* Tables */
    .stDataFrame th {{
        background-color: #1f1f1f !important;
        color: {COLORS['primary']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px;
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 5px;
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        padding: 0 16px;
        background-color: #1f1f1f;
        color: #e0e0e0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: #121212;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {COLORS['info']};
        color: #121212;
    }}

    /* Metric cards */
    .metric-card {{
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: all 0.2s ease;
    }}
    .metric-card:hover {{
        box-shadow: 0 0 15px rgba(116, 241, 116, 0.3);
    }}
    .metric-value {{
        color: {COLORS['primary']};
        font-size: 24px;
        font-weight: bold;
    }}
    .metric-label {{
        color: {COLORS['info']};
        font-size: 14px;
    }}
    .company-subtitle {{
        color: {COLORS['info']};
        margin-top: -0.6rem;
    }}

    /* Indicator colours */
    .indicator-positive {{ color: {COLORS['primary']}; }}
    .indicator-neutral {{ color: {COLORS['warning']}; }}
    .indicator-negative {{ color: {COLORS['accent']}; }}
</style>
"""


# ---------------------------------------------------------------------------
# Cached resources and data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_data_loader() -> DataLoader:
    """Shared DataLoader instance."""
    return DataLoader()


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Cached ticker search."""
    return get_data_loader().search_tickers(query, limit)


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_bundle(ticker: str, start_date: str, end_date: str, period: str) -> Dict[str, Any]:
    """
    Load everything the company analysis needs.

    Returns:
        ``{ticker, company_info, price_data, financial_data, sector_data,
        period, start_date, end_date, loaded_at, warnings}``
    """
    loader = get_data_loader()
    warnings: List[str] = []

    company_info = loader.get_company_info(ticker) or {}
    price_data = loader.get_historical_prices(ticker, start_date, end_date)
    statements = {
        'income_statement': loader.get_financial_statements(ticker, 'income', period),
        'balance_sheet': loader.get_financial_statements(ticker, 'balance', period),
        'cash_flow': loader.get_financial_statements(ticker, 'cash', period),
    }
    if not company_info:
        record = lookup_company(ticker)
        company_info = {'symbol': ticker, 'name': record['name'] if record else ticker,
                        'sector': record['sector'] if record else 'Unknown', 'industry': 'N/A'}
        warnings.append("Company profile could not be loaded; showing limited information.")
    if all(df is None or df.empty for df in statements.values()):
        warnings.append("Financial statements are unavailable; fundamental analysis will be limited.")

    sector = company_info.get('sector', 'Unknown')
    sector_data = loader.get_sector_data(sector) if sector != 'Unknown' else pd.DataFrame()

    return {
        'ticker': ticker,
        'company_info': company_info,
        'price_data': price_data,
        'financial_data': build_financial_data(company_info=company_info, price_data=price_data, **statements),
        'sector_data': sector_data if sector_data is not None and not sector_data.empty else None,
        'period': period,
        'start_date': start_date,
        'end_date': end_date,
        'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_safely(label: str, renderer: Callable[..., Any], *args, **kwargs) -> Any:
    """Run a page renderer; show a contained error instead of crashing the app."""
    try:
        return renderer(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — isolate page failures
        logger.error(f"{label} failed: {traceback.format_exc()}")
        st.error(f"{label} could not be displayed: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None


def load_renderer(module_name: str, function_name: str) -> Optional[Callable[..., Any]]:
    """Import a page renderer lazily; report import failures in the UI."""
    try:
        return getattr(importlib.import_module(module_name), function_name)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Cannot load {module_name}.{function_name}: {traceback.format_exc()}")
        st.error(f"This page is unavailable: {e}")
        return None


def render_page(page: str) -> None:
    """Render a page by name."""
    if page == nav.COMPANY_ANALYSIS:
        render_safely("Company analysis", render_company_analysis)
        return
    module_name, function_name = PAGE_RENDERERS[page]
    renderer = load_renderer(module_name, function_name)
    if renderer is not None:
        render_safely(page, renderer)


def _format_price(value: Optional[float]) -> str:
    return f"${value:,.2f}" if value is not None else "N/A"


def _pct_change(close: pd.Series, periods: int) -> Optional[float]:
    if len(close) <= periods:
        return None
    previous = close.iloc[-1 - periods]
    return (close.iloc[-1] / previous - 1) * 100 if previous else None


def render_company_header(bundle: Dict[str, Any]) -> None:
    """Company name, classification and headline market metrics."""
    info = bundle['company_info']
    ticker = bundle['ticker']
    prices = bundle['price_data']
    market = bundle['financial_data']['market_data']

    st.header(f"{info.get('name', ticker)} ({ticker})")
    subtitle = " · ".join(str(x) for x in (info.get('sector'), info.get('industry'), info.get('exchange'),
                                            info.get('currency')) if x and x != 'N/A')
    if subtitle:
        st.markdown(f"<p class='company-subtitle'>{subtitle}</p>", unsafe_allow_html=True)

    close = prices['Close'].dropna() if prices is not None and not prices.empty else pd.Series(dtype=float)
    day_change = _pct_change(close, 1)
    month_change = _pct_change(close, 21)
    range_change = _pct_change(close, len(close) - 1) if len(close) > 1 else None

    cols = st.columns(5)
    cols[0].metric("Price", _format_price(to_float(market.get('share_price'))),
                   delta=f"{day_change:+.2f}% 1D" if day_change is not None else None)
    cols[1].metric("1-month change", f"{month_change:+.2f}%" if month_change is not None else "N/A")
    cols[2].metric(f"Change since {bundle['start_date']}", f"{range_change:+.2f}%" if range_change is not None else "N/A")
    cols[3].metric("Market cap", format_large_number(market.get('market_cap')))
    pe = to_float(info.get('pe_ratio'))
    cols[4].metric("P/E (TTM)", f"{pe:.1f}" if pe else "N/A")

    for warning in bundle.get('warnings', []):
        st.warning(warning)
    if not close.empty:
        last_date = close.index[-1]
        st.caption(f"Prices through {pd.Timestamp(last_date):%Y-%m-%d} · {bundle['period']} statements · "
                   f"loaded {bundle['loaded_at']}")


@st.cache_data(ttl=86400, show_spinner=False)
def _peer_snapshot(tickers: tuple) -> pd.DataFrame:
    loader = get_data_loader()
    rows = []
    for ticker in tickers:
        info = loader.get_company_info(ticker) or {}
        if not info:
            continue
        dividend = to_float(info.get('dividend_yield'))
        margin = to_float(info.get('profit_margins'))
        roe = to_float(info.get('return_on_equity'))
        rows.append({
            "Ticker": ticker,
            "Name": info.get('name', ticker),
            "Industry": info.get('industry', 'N/A'),
            "Market cap": format_large_number(info.get('market_cap')),
            "P/E": round(to_float(info.get('pe_ratio')), 1) if to_float(info.get('pe_ratio')) else None,
            "P/B": round(to_float(info.get('pb_ratio')), 1) if to_float(info.get('pb_ratio')) else None,
            "Dividend yield": f"{dividend * 100:.2f}%" if dividend is not None else "—",
            "Net margin": f"{margin * 100:.1f}%" if margin is not None else "—",
            "ROE": f"{roe * 100:.1f}%" if roe is not None else "—",
        })
    return pd.DataFrame(rows)


def render_peer_snapshot(bundle: Dict[str, Any]) -> None:
    """Compact peer table with a link to the full peer comparison page."""
    ticker = bundle['ticker']
    sector = bundle['company_info'].get('sector', 'Unknown')
    peers = get_sector_peers(ticker, sector, limit=6)
    if not peers:
        st.info("No peers are defined for this company's sector.")
        return
    st.subheader(f"{sector} peers")
    with st.spinner("Loading peer metrics..."):
        table = _peer_snapshot(tuple([ticker] + [p['symbol'] for p in peers]))
    if table.empty:
        st.info("Peer metrics are currently unavailable.")
    else:
        st.dataframe(table, hide_index=True, key="peers_snapshot")
    st.button("Open full peer comparison", key="peers_open_comparison", icon=":material/compare_arrows:",
              on_click=nav.go_to, args=(nav.PEER_COMPARISON, ticker))


# ---------------------------------------------------------------------------
# Company analysis page
# ---------------------------------------------------------------------------
SYNCED_TICKER_KEY = "_ca_synced_ticker"


def _company_selector() -> Optional[str]:
    """Sidebar company search / quick pick. Returns the chosen ticker."""
    active = nav.get_active_ticker()
    if active and st.session_state.get(SYNCED_TICKER_KEY) != active:
        # Another page selected a company: reflect it in the quick-pick box once
        st.session_state["ca_popular"] = active
        st.session_state["ca_search"] = ""
        st.session_state[SYNCED_TICKER_KEY] = active
    query = st.text_input("Search company or ticker", key="ca_search", placeholder="e.g. Apple or AAPL")
    if query.strip():
        results = search_tickers(query.strip())
        if not results:
            st.warning("No companies found. You can still type an exact ticker.")
            return query.strip().upper()
        options = [f"{r['symbol']} — {r.get('name', '')}" for r in results]
        choice = st.selectbox("Matches", options, key="ca_search_choice")
        return choice.split(" — ")[0] if choice else None

    options = list(POPULAR_TICKERS)
    for extra in (active, st.session_state.get("ca_popular")):
        if extra and extra not in options:
            options.insert(0, extra)
    st.session_state.setdefault("ca_popular", active or options[0])
    choice = st.selectbox("Popular companies", options,
                          format_func=lambda t: f"{t} — {POPULAR_TICKERS.get(t) or (lookup_company(t) or {}).get('name', '')}",
                          key="ca_popular")
    return choice


def _date_range_selector() -> tuple:
    today = datetime.now().date()
    label = st.selectbox("Price history", list(DATE_RANGES), index=list(DATE_RANGES).index(DEFAULT_RANGE),
                         key="ca_range")
    days = DATE_RANGES[label]
    if days is None:
        selected = st.date_input("Custom range", value=(today - timedelta(days=365), today), max_value=today,
                                 key="ca_custom_range")
        if isinstance(selected, (list, tuple)) and len(selected) == 2:
            return str(selected[0]), str(selected[1])
        return str(today - timedelta(days=365)), str(today)  # range still being picked
    return str(today - timedelta(days=days)), str(today)


def _set_request(ticker: str, start_date: str, end_date: str, period: str) -> None:
    st.session_state[REQUEST_KEY] = {'ticker': ticker.upper(), 'start_date': start_date, 'end_date': end_date,
                                     'period': period}
    st.session_state[nav.ACTIVE_TICKER_KEY] = ticker.upper()
    st.session_state[SYNCED_TICKER_KEY] = ticker.upper()


def render_company_analysis() -> None:
    with st.sidebar:
        st.subheader("Company")
        ticker = _company_selector()
        start_date, end_date = _date_range_selector()
        period = st.radio("Financial statements", ["annual", "quarterly"], horizontal=True, key="ca_period",
                          format_func=str.title)
        if st.button("Analyze company", type="primary", key="ca_analyze", icon=":material/query_stats:",
                     disabled=not ticker, width="stretch"):
            _set_request(ticker, start_date, end_date, period)

    request = st.session_state.get(REQUEST_KEY)
    active = nav.get_active_ticker()
    if active and (request is None or request['ticker'] != active):
        # Another page asked for this company: analyse it with the current settings
        _set_request(active, start_date, end_date, period)
        request = st.session_state[REQUEST_KEY]

    if not request:
        st.title("Company Analysis")
        st.info("Choose a company in the sidebar and click **Analyze company** to see its overview, "
                "financial statements, sector-specific valuation and risk profile.")
        return

    with st.spinner(f"Loading data for {request['ticker']}..."):
        bundle = load_company_bundle(request['ticker'], request['start_date'], request['end_date'],
                                     request['period'])

    if (bundle['price_data'] is None or bundle['price_data'].empty) and not bundle['company_info'].get('market_cap'):
        st.error(f"No market data found for **{request['ticker']}**. Check the ticker symbol or try again later.")
        return

    render_company_header(bundle)
    args = (bundle['ticker'], bundle['financial_data'], bundle['company_info'], bundle['price_data'])
    tab_labels = list(COMPANY_TABS) + ["Peers"]
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            if label == "Peers":
                render_safely("Peers", render_peer_snapshot, bundle)
                continue
            renderer = load_renderer(*COMPANY_TABS[label])
            if renderer is None:
                continue
            kwargs = {'sector_data': bundle['sector_data']} if label in {"Overview", "Valuation"} else {}
            render_safely(label, renderer, *args, **kwargs)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def render_sidebar_navigation() -> str:
    st.sidebar.title(f"📊 {APP_TITLE}")
    if nav.NAV_KEY not in st.session_state:
        st.session_state[nav.NAV_KEY] = nav.HOME
    page = st.sidebar.radio("Navigation", list(nav.PAGES), key=nav.NAV_KEY,
                            format_func=lambda name: f"{nav.PAGES[name]} {name}", label_visibility="collapsed")
    st.sidebar.divider()
    return page


def render_sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        with st.expander("Data & settings"):
            if config.OFFLINE_MODE:
                st.caption("Offline mode: only cached data is used.")
            providers = ["Yahoo Finance"] + (["Alpha Vantage"] if config.API_KEYS.get("alpha_vantage") else [])
            st.caption(f"Data providers: {', '.join(providers)}")
            if st.button("Clear cached data", key="sidebar_clear_cache", icon=":material/refresh:"):
                st.cache_data.clear()
                removed = get_data_loader().clear_cache()
                st.toast(f"Cleared {removed} cached files.")
        st.caption("For research and education only — not investment advice.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page = render_sidebar_navigation()
    render_page(page)
    render_sidebar_footer()


if __name__ == "__main__":
    main()
