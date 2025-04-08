import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import CACHE_DIR, CACHE_EXPIRY_DAYS, API_KEYS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.models.financial_statements import FinancialStatementAnalyzer
from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

# Page configuration
st.set_page_config(
    page_title="Stock Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Apply custom CSS for styling
def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
        <style>
        /* Основной фон и цвета */
        .main {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stSidebar {
            background-color: #1f1f1f;
        }

        /* Кнопки */
        .stButton button {
            background-color: #74f174;  /* Салатовый */
            color: #121212;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #90bff9;  /* Небесно-голубой при наведении */
            transform: scale(1.05);
        }

        /* Таблицы */
        .stDataFrame {
            width: 100%;
            max-width: 100%;
        }
        .stDataFrame th {
            background-color: #1f1f1f !important;
            color: #74f174 !important;
            text-align: center !important;
        }
        .stDataFrame td {
            text-align: center !important;
            vertical-align: middle !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background-color: #1f1f1f;
            border-radius: 10px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #1f1f1f;
            color: #e0e0e0;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #74f174;
            color: #121212;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #90bff9;
            color: #121212;
        }

        /* Графики */
        .stPlotlyChart {
            width: 100%;
            max-width: 100%;
        }

        /* Метрики */
        .metric-card {
            background-color: #1f1f1f;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(116, 241, 116, 0.3);
        }
        .metric-value {
            color: #74f174;
            font-size: 24px;
            font-weight: bold;
        }
        .metric-label {
            color: #90bff9;
            font-size: 14px;
        }

        /* Цвета индикаторов */
        .indicator-positive { color: #74f174; }
        .indicator-neutral { color: #fff59d; }
        .indicator-negative { color: #faa1a4; }
        </style>
    """, unsafe_allow_html=True)


# Initialize data loader and analyzers as session state objects
@st.cache_resource
def get_data_loader():
    """Initialize and cache the data loader"""
    return DataLoader()


@st.cache_resource
def get_ratio_analyzer():
    """Initialize and cache the ratio analyzer"""
    return FinancialRatioAnalyzer()


@st.cache_resource
def get_statement_analyzer():
    """Initialize and cache the financial statement analyzer"""
    return FinancialStatementAnalyzer()


@st.cache_resource
def get_bankruptcy_analyzer():
    """Initialize and cache the bankruptcy analyzer"""
    return BankruptcyAnalyzer()


@st.cache_resource
def get_visualizer():
    """Initialize and cache the financial visualizer"""
    return FinancialVisualizer(theme="dark")


# Cache functions for data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(ticker, start_date, end_date):
    """Load and cache stock price data"""
    loader = get_data_loader()
    return loader.get_historical_prices(ticker, start_date, end_date)


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_company_info(ticker):
    """Load and cache company information"""
    loader = get_data_loader()
    return loader.get_company_info(ticker)


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_financial_statements(ticker, statement_type, period):
    """Load and cache financial statements"""
    loader = get_data_loader()
    return loader.get_financial_statements(ticker, statement_type, period)


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_sector_data(sector):
    """Load and cache sector data"""
    loader = get_data_loader()
    return loader.get_sector_data(sector)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_tickers(query, limit=10):
    """Search and cache ticker results"""
    loader = get_data_loader()
    return loader.search_tickers(query, limit)


# Main application
def main():
    """Main application function"""
    # Apply custom CSS
    apply_custom_css()

    # Application title
    st.title("Stock Analysis System")
    st.markdown("### Comprehensive Financial Analysis & Valuation")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Company Selection")

        # Search box for tickers
        search_query = st.text_input("Search for a company:", "")

        if search_query:
            # Search for tickers
            search_results = search_tickers(search_query)

            if search_results:
                # Format search results
                options = [f"{result['symbol']} - {result['name']}" for result in search_results]
                selected_option = st.selectbox("Select a company:", options)

                # Extract ticker from selected option
                if selected_option:
                    ticker = selected_option.split(" - ")[0]
                else:
                    ticker = None
            else:
                st.warning("No companies found matching your search.")
                ticker = None
        else:
            # Default companies for quick selection
            common_tickers = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation",
                "GOOGL": "Alphabet Inc.",
                "AMZN": "Amazon.com, Inc.",
                "META": "Meta Platforms, Inc.",
                "TSLA": "Tesla, Inc.",
                "JPM": "JPMorgan Chase & Co.",
                "V": "Visa Inc."
            }

            ticker_options = [f"{t} - {n}" for t, n in common_tickers.items()]
            selected_option = st.selectbox("Select a common company:", ticker_options)

            # Extract ticker from selected option
            if selected_option:
                ticker = selected_option.split(" - ")[0]
            else:
                ticker = None

        # Date range selection
        st.header("Date Range")

        # Default to 1 year of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        date_options = {
            "1 Month": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            "3 Months": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            "6 Months": (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
            "1 Year": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            "3 Years": (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d'),
            "5 Years": (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d'),
            "Custom": "custom"
        }

        selected_range = st.selectbox("Select time period:", list(date_options.keys()), index=3)  # Default to 1 Year

        if selected_range == "Custom":
            start_date = st.date_input("Start date:", datetime.strptime(default_start_date, '%Y-%m-%d'))
            end_date = st.date_input("End date:", datetime.strptime(end_date, '%Y-%m-%d'))
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            start_date = date_options[selected_range]

        # Analysis options
        st.header("Analysis Options")

        # Financial statement period
        statement_period = st.radio("Financial statement period:", ["annual", "quarterly"], horizontal=True)

        # Expander for advanced options
        with st.expander("Advanced Options"):
            # Show moving averages option
            show_ma = st.checkbox("Show moving averages", value=True)

            # MA periods
            if show_ma:
                ma_periods_str = st.text_input("Moving average periods (comma-separated):", "50, 200")
                ma_periods = [int(p.strip()) for p in ma_periods_str.split(",") if p.strip().isdigit()]
            else:
                ma_periods = []

            # Show volume option
            show_volume = st.checkbox("Show volume", value=True)

        # Button to trigger analysis
        analyze_button = st.button("Analyze Company", type="primary")

    # Main content area
    if ticker and analyze_button:
        # Load data
        try:
            # Show loading message
            with st.spinner(f"Loading data for {ticker}..."):
                # Load price data
                price_data = load_stock_data(ticker, start_date, end_date)
                if price_data.empty:
                    st.error(f"Failed to load price data for {ticker}. Please check the ticker symbol and try again.")
                    return

                # Load company info
                company_info = load_company_info(ticker)
                if not company_info or 'name' not in company_info:
                    st.error(
                        f"Failed to load company information for {ticker}. Please check the ticker symbol and try again.")
                    return

                # Load financial statements
                income_statement = load_financial_statements(ticker, 'income', statement_period)
                balance_sheet = load_financial_statements(ticker, 'balance', statement_period)
                cash_flow = load_financial_statements(ticker, 'cash', statement_period)

                # Check if we have at least some financial data
                if income_statement.empty and balance_sheet.empty and cash_flow.empty:
                    st.warning(
                        f"Could not load financial statement data for {ticker}. Limited analysis will be available.")

                # Create financial data dict
                financial_data = {
                    'income_statement': income_statement,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'market_data': {
                        'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                        'market_cap': company_info.get('market_cap')
                    }
                }

                # Get company's sector
                sector = company_info.get('sector', 'Unknown')

                # Load sector data if available
                if sector != 'Unknown':
                    sector_data = load_sector_data(sector)
                    if sector_data is None:
                        st.warning(f"Could not load sector data for {sector}. Sector comparison will be limited.")
                else:
                    sector_data = None
                    st.info("Company sector information not available. Sector-specific analysis will be limited.")

        except Exception as e:
            st.error(f"An error occurred during data loading: {str(e)}")
            st.info("Please try again later or choose a different company to analyze.")
            return

            # Display company header
            st.header(f"{company_info.get('name', ticker)} ({ticker})")

            # Company metadata row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sector", sector)
            with col2:
                st.metric("Industry", company_info.get('industry', 'N/A'))
            with col3:
                st.metric("Exchange", company_info.get('exchange', 'N/A'))
            with col4:
                st.metric("Currency", company_info.get('currency', 'USD'))

            # Price metrics row
            latest_price = price_data['Close'].iloc[-1] if not price_data.empty else None
            price_change = price_data['Close'].pct_change().iloc[-1] * 100 if not price_data.empty else None
            price_change_30d = price_data['Close'].pct_change(30).iloc[-1] * 100 if not price_data.empty and len(
                price_data) > 30 else None

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${latest_price:.2f}" if latest_price else "N/A")
            with col2:
                st.metric("1-Day Change", f"{price_change:.2f}%" if price_change else "N/A",
                          delta=f"{price_change:.2f}%" if price_change else None)
            with col3:
                st.metric("30-Day Change", f"{price_change_30d:.2f}%" if price_change_30d else "N/A",
                          delta=f"{price_change_30d:.2f}%" if price_change_30d else None)
            with col4:
                st.metric("Market Cap",
                          f"${company_info.get('market_cap') / 1e9:.2f}B" if company_info.get('market_cap') else "N/A")

            # Create tabs for different analysis sections
            tabs = st.tabs([
                "Overview",
                "Financial Analysis",
                "Valuation",
                "Risk Analysis",
                "Peer Comparison"
            ])

            # Tab 1: Overview
            with tabs[0]:
                # Create visualizer
                visualizer = get_visualizer()

                # Stock price chart
                st.subheader("Historical Price Chart")
                fig = visualizer.plot_stock_price(
                    price_data,
                    ticker,
                    company_name=company_info.get('name'),
                    ma_periods=ma_periods,
                    volume=show_volume
                )
                st.plotly_chart(fig, use_container_width=True)

                # Company description
                st.subheader("Company Description")
                st.write(company_info.get('description', 'No description available.'))

                # Key metrics
                st.subheader("Key Financial Metrics")

                # Calculate financial ratios
                ratio_analyzer = get_ratio_analyzer()
                ratios = ratio_analyzer.calculate_ratios(financial_data)

                # Display key ratios in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="section-header"><h4>Valuation</h4></div>', unsafe_allow_html=True)

                    # P/E Ratio
                    pe_ratio = ratios.get('valuation', {}).get('pe_ratio')
                    if pe_ratio:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{pe_ratio:.2f}</div><div class="metric-label">P/E Ratio</div></div>',
                            unsafe_allow_html=True)

                    # P/S Ratio
                    ps_ratio = ratios.get('valuation', {}).get('ps_ratio')
                    if ps_ratio:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{ps_ratio:.2f}</div><div class="metric-label">P/S Ratio</div></div>',
                            unsafe_allow_html=True)

                    # P/B Ratio
                    pb_ratio = ratios.get('valuation', {}).get('pb_ratio')
                    if pb_ratio:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{pb_ratio:.2f}</div><div class="metric-label">P/B Ratio</div></div>',
                            unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="section-header"><h4>Profitability</h4></div>', unsafe_allow_html=True)

                    # Gross Margin
                    gross_margin = ratios.get('profitability', {}).get('gross_margin')
                    if gross_margin:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{gross_margin * 100:.2f}%</div><div class="metric-label">Gross Margin</div></div>',
                            unsafe_allow_html=True)

                    # Operating Margin
                    op_margin = ratios.get('profitability', {}).get('operating_margin')
                    if op_margin:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{op_margin * 100:.2f}%</div><div class="metric-label">Operating Margin</div></div>',
                            unsafe_allow_html=True)

                    # Net Margin
                    net_margin = ratios.get('profitability', {}).get('net_margin')
                    if net_margin:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{net_margin * 100:.2f}%</div><div class="metric-label">Net Margin</div></div>',
                            unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="section-header"><h4>Returns</h4></div>', unsafe_allow_html=True)

                    # ROE
                    roe = ratios.get('profitability', {}).get('roe')
                    if roe:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{roe * 100:.2f}%</div><div class="metric-label">Return on Equity</div></div>',
                            unsafe_allow_html=True)

                    # ROA
                    roa = ratios.get('profitability', {}).get('roa')
                    if roa:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{roa * 100:.2f}%</div><div class="metric-label">Return on Assets</div></div>',
                            unsafe_allow_html=True)

                    # Current Ratio
                    current_ratio = ratios.get('liquidity', {}).get('current_ratio')
                    if current_ratio:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{current_ratio:.2f}</div><div class="metric-label">Current Ratio</div></div>',
                            unsafe_allow_html=True)

            # Tab 2: Financial Analysis
            with tabs[1]:
                # Create statement analyzer
                statement_analyzer = get_statement_analyzer()

                # Analyze financial statements
                income_analysis = statement_analyzer.analyze_income_statement(income_statement)
                balance_analysis = statement_analyzer.analyze_balance_sheet(balance_sheet)
                cash_flow_analysis = statement_analyzer.analyze_cash_flow(cash_flow)

                # Create tabs for different financial statements
                finance_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios"])

                # Income Statement Tab
                with finance_tabs[0]:
                    st.subheader("Income Statement Analysis")

                    # Display income statement data
                    if not income_statement.empty:
                        # Show key metrics
                        col1, col2 = st.columns(2)

                        with col1:
                            # Revenue trend chart
                            st.subheader("Revenue Trend")
                            if 'key_metrics' in income_analysis and 'net_income' in income_analysis['key_metrics']:
                                net_income_trend = income_analysis['key_metrics']['net_income']
                                if net_income_trend and 'values' in net_income_trend:
                                    fig = visualizer.plot_financial_statement_trend(
                                        pd.DataFrame({k: [v] for k, v in net_income_trend['values'].items()}).T,
                                        'income',
                                        ['Net Income'],
                                        height=300
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Net Income trend data not available.")
                            else:
                                st.info("Net Income data not available.")

                        # Margins analysis
                        st.subheader("Margin Analysis")
                        if 'key_metrics' in income_analysis and 'margins' in income_analysis['key_metrics']:
                            margins = income_analysis['key_metrics']['margins']

                            # Create DataFrame for margin trends
                            margin_data = {}
                            margin_names = []

                            for margin_name, margin_trend in margins.items():
                                if margin_trend and 'values' in margin_trend:
                                    margin_data[margin_name] = margin_trend['values']
                                    margin_names.append(margin_name)

                            if margin_data:
                                # Convert to DataFrame for visualization
                                margin_df = pd.DataFrame(margin_data)

                                # Plot margins
                                fig = visualizer.plot_financial_statement_trend(
                                    margin_df,
                                    'income',
                                    margin_names,
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Margin trend data not available.")
                        else:
                            st.info("Margin data not available.")

                        # Display raw income statement
                        with st.expander("View Raw Income Statement"):
                            st.dataframe(income_statement)
                    else:
                        st.warning("Income statement data not available for this company.")

                # Balance Sheet Tab
                with finance_tabs[1]:
                    st.subheader("Balance Sheet Analysis")

                    # Display balance sheet data
                    if not balance_sheet.empty:
                        # Show key metrics
                        col1, col2 = st.columns(2)

                        with col1:
                            # Assets trend chart
                            st.subheader("Assets Trend")
                            if 'key_metrics' in balance_analysis and 'assets' in balance_analysis['key_metrics']:
                                assets = balance_analysis['key_metrics']['assets']

                                # Create DataFrame for assets trends
                                assets_data = {}
                                asset_names = []

                                for asset_name, asset_trend in assets.items():
                                    if asset_trend and 'values' in asset_trend:
                                        assets_data[asset_name] = asset_trend['values']
                                        asset_names.append(asset_name)

                                if assets_data:
                                    # Convert to DataFrame for visualization
                                    assets_df = pd.DataFrame(assets_data)

                                    # Plot assets
                                    fig = visualizer.plot_financial_statement_trend(
                                        assets_df,
                                        'balance',
                                        asset_names,
                                        height=300
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Asset trend data not available.")
                            else:
                                st.info("Asset data not available.")

                        with col2:
                            # Liabilities trend chart
                            st.subheader("Liabilities & Equity Trend")
                            if ('key_metrics' in balance_analysis and
                                    'liabilities' in balance_analysis['key_metrics'] and
                                    'equity' in balance_analysis['key_metrics']):

                                liabilities = balance_analysis['key_metrics']['liabilities']
                                equity = balance_analysis['key_metrics']['equity']

                                # Create DataFrame for trends
                                trend_data = {}
                                trend_names = []

                                # Add liabilities
                                for name, trend in liabilities.items():
                                    if trend and 'values' in trend:
                                        trend_data[name] = trend['values']
                                        trend_names.append(name)

                                # Add equity
                                for name, trend in equity.items():
                                    if trend and 'values' in trend:
                                        trend_data[name] = trend['values']
                                        trend_names.append(name)

                                if trend_data:
                                    # Convert to DataFrame for visualization
                                    trend_df = pd.DataFrame(trend_data)

                                    # Plot data
                                    fig = visualizer.plot_financial_statement_trend(
                                        trend_df,
                                        'balance',
                                        trend_names,
                                        height=300
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Liabilities and equity trend data not available.")
                            else:
                                st.info("Liabilities and equity data not available.")

                        # Financial position ratios
                        st.subheader("Financial Position Ratios")
                        if 'key_metrics' in balance_analysis and 'ratios' in balance_analysis['key_metrics']:
                            balance_ratios = balance_analysis['key_metrics']['ratios']

                            # Create DataFrame for ratio trends
                            ratio_data = {}
                            ratio_names = []

                            for ratio_name, ratio_trend in balance_ratios.items():
                                if ratio_trend and 'values' in ratio_trend:
                                    ratio_data[ratio_name] = ratio_trend['values']
                                    ratio_names.append(ratio_name)

                            if ratio_data:
                                # Convert to DataFrame for visualization
                                ratio_df = pd.DataFrame(ratio_data)

                                # Plot ratios
                                fig = visualizer.plot_financial_statement_trend(
                                    ratio_df,
                                    'balance',
                                    ratio_names,
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Financial position ratio trend data not available.")
                        else:
                            st.info("Financial position ratio data not available.")

                        # Display raw balance sheet
                        with st.expander("View Raw Balance Sheet"):
                            st.dataframe(balance_sheet)
                    else:
                        st.warning("Balance sheet data not available for this company.")

                # Cash Flow Tab
                with finance_tabs[2]:
                    st.subheader("Cash Flow Analysis")

                    # Display cash flow data
                    if not cash_flow.empty:
                        # Show key metrics
                        st.subheader("Cash Flow Components")
                        if 'key_metrics' in cash_flow_analysis:
                            # Create DataFrame for cash flow trends
                            cf_data = {}
                            cf_names = []

                            key_metrics = [
                                'operating_cash_flow',
                                'investing_cash_flow',
                                'financing_cash_flow',
                                'free_cash_flow'
                            ]

                            for metric in key_metrics:
                                if metric in cash_flow_analysis['key_metrics']:
                                    trend = cash_flow_analysis['key_metrics'][metric]
                                    if trend and 'values' in trend:
                                        cf_data[metric] = trend['values']
                                        cf_names.append(metric)

                            if cf_data:
                                # Convert to DataFrame for visualization
                                cf_df = pd.DataFrame(cf_data)

                                # Plot cash flows
                                fig = visualizer.plot_financial_statement_trend(
                                    cf_df,
                                    'cash_flow',
                                    cf_names,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Cash flow trend data not available.")
                        else:
                            st.info("Cash flow component data not available.")

                        # Display raw cash flow statement
                        with st.expander("View Raw Cash Flow Statement"):
                            st.dataframe(cash_flow)
                    else:
                        st.warning("Cash flow data not available for this company.")

                # Ratios Tab
                with finance_tabs[3]:
                    st.subheader("Financial Ratios Analysis")

                    # Show ratio analysis across categories
                    # Calculate key ratios
                    ratio_analyzer = get_ratio_analyzer()
                    ratios = ratio_analyzer.calculate_ratios(financial_data)

                    # Get sector benchmarks if available
                    if sector != 'Unknown':
                        sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector)
                    else:
                        sector_benchmarks = None

                    # Analyze ratios against benchmarks
                    ratio_analysis = ratio_analyzer.analyze_ratios(ratios, sector)

                    # Create tabs for ratio categories
                    ratio_categories = list(ratios.keys())
                    if ratio_categories:
                        ratio_tabs = st.tabs(ratio_categories)

                        # For each category, show ratios and comparison
                        for i, category in enumerate(ratio_categories):
                            with ratio_tabs[i]:
                                st.subheader(f"{category.capitalize()} Ratios")

                                # Get ratios for this category
                                category_ratios = ratios.get(category, {})
                                category_analysis = ratio_analysis.get(category, {})

                                if category_ratios:
                                    # Visualize ratios
                                    fig = visualizer.plot_financial_ratios(
                                        {category: category_analysis},
                                        category=category,
                                        benchmark_data=sector_benchmarks,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show ratio table with benchmarks
                                    st.subheader("Ratio Details")

                                    ratio_data = []
                                    for ratio_name, ratio_value in category_ratios.items():
                                        analysis = category_analysis.get(ratio_name, {})
                                        benchmark = sector_benchmarks.get(category, {}).get(
                                            ratio_name) if sector_benchmarks else None

                                        ratio_data.append({
                                            "Ratio": ratio_name.replace('_', ' ').title(),
                                            "Company Value": ratio_value,
                                            "Sector Average": benchmark,
                                            "Difference (%)": analysis.get(
                                                'percent_diff') if analysis and 'percent_diff' in analysis else None,
                                            "Assessment": analysis.get('assessment',
                                                                       '').capitalize() if analysis else None
                                        })

                                    # Create DataFrame and show
                                    if ratio_data:
                                        ratio_df = pd.DataFrame(ratio_data)
                                        st.dataframe(ratio_df, use_container_width=True)
                                    else:
                                        st.info("No ratio details available for display.")
                                else:
                                    st.info(f"No {category} ratios available for this company.")

                        # Get key ratios for the sector
                        if sector != 'Unknown':
                            st.subheader(f"Key Ratios for {sector} Sector")
                            key_ratios = ratio_analyzer.get_key_ratios_for_sector(sector)

                            if key_ratios:
                                # Display key ratios for this sector
                                key_ratio_data = []
                                for ratio_info in key_ratios:
                                    key_ratio_data.append({
                                        "Ratio": ratio_info['ratio'],
                                        "Category": ratio_info['category'],
                                        "Description": ratio_info['description']
                                    })

                                key_ratio_df = pd.DataFrame(key_ratio_data)
                                st.dataframe(key_ratio_df, use_container_width=True)
                            else:
                                st.info(f"No key ratios defined for the {sector} sector.")
                    else:
                        st.warning("No financial ratios available for analysis.")

            # Tab 3: Valuation
            with tabs[2]:
                st.subheader("Company Valuation Analysis")

                # Placeholder for DCF model implementation
                st.info(
                    "DCF valuation model implementation is pending. In a complete system, this would show detailed DCF analysis with sensitivity testing.")

                # Show valuation ratios
                valuation_ratios = ratios.get('valuation', {})
                sector_valuation = sector_benchmarks.get('valuation', {}) if sector_benchmarks else {}

                if valuation_ratios:
                    # Create comparison data
                    valuation_data = {
                        'company': valuation_ratios,
                        'sector_avg': sector_valuation
                    }

                    # Display valuation heatmap
                    st.subheader("Valuation Multiples vs. Sector")
                    fig = visualizer.plot_valuation_heatmap(
                        valuation_data,
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display valuation metrics
                    col1, col2, col3 = st.columns(3)

                    # Display key valuation metrics
                    with col1:
                        pe = valuation_ratios.get('pe_ratio')
                        sector_pe = sector_valuation.get('pe_ratio')
                        if pe is not None:
                            st.metric(
                                "P/E Ratio",
                                f"{pe:.2f}",
                                f"{((pe / sector_pe) - 1) * 100:.1f}% vs Sector" if sector_pe else None,
                                delta_color="inverse"
                            )

                    with col2:
                        pb = valuation_ratios.get('pb_ratio')
                        sector_pb = sector_valuation.get('pb_ratio')
                        if pb is not None:
                            st.metric(
                                "P/B Ratio",
                                f"{pb:.2f}",
                                f"{((pb / sector_pb) - 1) * 100:.1f}% vs Sector" if sector_pb else None,
                                delta_color="inverse"
                            )

                    with col3:
                        ps = valuation_ratios.get('ps_ratio')
                        sector_ps = sector_valuation.get('ps_ratio')
                        if ps is not None:
                            st.metric(
                                "P/S Ratio",
                                f"{ps:.2f}",
                                f"{((ps / sector_ps) - 1) * 100:.1f}% vs Sector" if sector_ps else None,
                                delta_color="inverse"
                            )
                else:
                    st.warning("Valuation ratios are not available for this company.")

            # Tab 4: Risk Analysis
            with tabs[3]:
                st.subheader("Risk Analysis")

                # Financial Health Score
                st.subheader("Financial Health Assessment")

                # Calculate financial health score
                statement_analyzer = get_statement_analyzer()
                health_score = statement_analyzer.calculate_financial_health_score(
                    income_analysis, balance_analysis, cash_flow_analysis
                )

                # Display financial health score gauge
                if health_score and health_score.get('overall_score') is not None:
                    fig = visualizer.plot_financial_health_score(
                        health_score,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display health score components as metrics
                    components = health_score.get('components', {})
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        profitability = components.get('profitability')
                        if profitability is not None:
                            st.metric("Profitability", f"{profitability}/100")

                    with col2:
                        liquidity = components.get('liquidity')
                        if liquidity is not None:
                            st.metric("Liquidity", f"{liquidity}/100")

                    with col3:
                        solvency = components.get('solvency')
                        if solvency is not None:
                            st.metric("Solvency", f"{solvency}/100")

                    with col4:
                        cash_flow = components.get('cash_flow')
                        if cash_flow is not None:
                            st.metric("Cash Flow", f"{cash_flow}/100")
                else:
                    st.warning("Financial health score could not be calculated due to insufficient data.")

                # Bankruptcy Risk Analysis
                st.subheader("Bankruptcy Risk Analysis")

                # Calculate bankruptcy risk
                bankruptcy_analyzer = get_bankruptcy_analyzer()
                risk_assessment = bankruptcy_analyzer.get_comprehensive_risk_assessment(
                    financial_data,
                    sector
                )

                # Display bankruptcy risk analysis
                if risk_assessment and 'models' in risk_assessment:
                    fig = visualizer.plot_bankruptcy_risk(
                        risk_assessment,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display risk assessment details
                    if risk_assessment.get('overall_assessment'):
                        assessment_color = risk_assessment.get('overall_color', '#ffffff')
                        st.markdown(
                            f"<div style='background-color: {assessment_color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                            f"<h4 style='color: #000000; text-align: center;'>{risk_assessment.get('overall_description', '')}</h4>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Bankruptcy risk analysis could not be performed due to insufficient data.")

                # Market Risk Indicators
                st.subheader("Market Risk Indicators")

                # Display beta and volatility if available
                col1, col2 = st.columns(2)

                with col1:
                    beta = company_info.get('beta')
                    if beta is not None:
                        beta_desc = ""
                        if beta < 0.8:
                            beta_desc = "Lower volatility than market"
                        elif beta < 1.2:
                            beta_desc = "Similar volatility to market"
                        else:
                            beta_desc = "Higher volatility than market"

                        st.metric("Beta", f"{beta:.2f}", beta_desc)
                    else:
                        st.metric("Beta", "N/A")

                with col2:
                    # Calculate volatility from price data
                    if not price_data.empty and len(price_data) > 20:
                        daily_returns = price_data['Close'].pct_change().dropna()
                        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                        st.metric("Annual Volatility", f"{volatility:.2f}%")
                    else:
                        st.metric("Annual Volatility", "N/A")

            # Tab 5: Peer Comparison
            with tabs[4]:
                st.subheader("Peer Comparison")

                # Placeholder for peer comparison implementation
                st.info(
                    "Peer comparison functionality is pending implementation. In a complete system, this would show detailed comparison with industry peers.")

                # Display sector performance if available
                if sector != 'Unknown':
                    st.subheader(f"{sector} Sector Performance")
                    st.info(
                        f"Sector performance analysis for {sector} would be displayed here, showing how {ticker} compares to peers.")
                else:
                    st.warning(
                        "Sector information is not available for this company, so peer comparison cannot be performed.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please try again later or choose a different company to analyze.")


if __name__ == "__main__":
    main()