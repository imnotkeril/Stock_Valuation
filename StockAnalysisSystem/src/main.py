import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS, UI_SETTINGS
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
        /* Main background and colors */
        .main {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stSidebar {
            background-color: #1f1f1f;
        }

        /* Buttons */
        .stButton button {
            background-color: #74f174;  /* Green */
            color: #121212;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #90bff9;  /* Blue on hover */
            transform: scale(1.05);
        }

        /* Tables */
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

        /* Charts */
        .stPlotlyChart {
            width: 100%;
            max-width: 100%;
        }

        /* Metrics */
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

        /* Indicator colors */
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
        try:
            # Show loading message
            with st.spinner(f"Loading data for {ticker}..."):
                # Load price data
                price_data = load_stock_data(ticker, start_date, end_date)
                if price_data is None or price_data.empty:
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
                if (income_statement is None or income_statement.empty) and \
                        (balance_sheet is None or balance_sheet.empty) and \
                        (cash_flow is None or cash_flow.empty):
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

                # Get company's sector (normalize "Technology" to "Information Technology")
                sector = company_info.get('sector', 'Unknown')
                if sector == 'Technology':
                    sector = 'Information Technology'

                # Load sector data if available
                sector_data = None
                if sector != 'Unknown':
                    try:
                        sector_data = load_sector_data(sector)
                        if sector_data is None or sector_data.empty:
                            st.warning(f"Could not load sector data for {sector}. Sector comparison will be limited.")
                    except Exception as e:
                        st.warning(f"Error loading sector data: {str(e)}. Sector comparison will be limited.")
                else:
                    st.info("Company sector information not available. Sector-specific analysis will be limited.")

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
                st.metric("1-Day Change", f"{price_change:.2f}%" if price_change is not None else "N/A",
                          delta=f"{price_change:.2f}%" if price_change is not None else None)
            with col3:
                st.metric("30-Day Change", f"{price_change_30d:.2f}%" if price_change_30d is not None else "N/A",
                          delta=f"{price_change_30d:.2f}%" if price_change_30d is not None else None)
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
                try:
                    fig = visualizer.plot_stock_price(
                        price_data,
                        ticker,
                        company_name=company_info.get('name'),
                        ma_periods=ma_periods,
                        volume=show_volume
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating price chart: {str(e)}")
                    logger.error(f"Error generating price chart: {str(e)}")

                # Company description
                st.subheader("Company Description")
                st.write(company_info.get('description', 'No description available.'))

                # Key metrics
                st.subheader("Key Financial Metrics")

                try:
                    # Calculate financial ratios
                    ratio_analyzer = get_ratio_analyzer()
                    ratios = ratio_analyzer.calculate_ratios(financial_data)

                    # Display key ratios in columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown('<div class="section-header"><h4>Valuation</h4></div>', unsafe_allow_html=True)

                        # P/E Ratio
                        pe_ratio = ratios.get('valuation', {}).get('pe_ratio')
                        if pe_ratio is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{pe_ratio:.2f}</div><div class="metric-label">P/E Ratio</div></div>',
                                unsafe_allow_html=True)

                        # P/S Ratio
                        ps_ratio = ratios.get('valuation', {}).get('ps_ratio')
                        if ps_ratio is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{ps_ratio:.2f}</div><div class="metric-label">P/S Ratio</div></div>',
                                unsafe_allow_html=True)

                        # P/B Ratio
                        pb_ratio = ratios.get('valuation', {}).get('pb_ratio')
                        if pb_ratio is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{pb_ratio:.2f}</div><div class="metric-label">P/B Ratio</div></div>',
                                unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header"><h4>Profitability</h4></div>', unsafe_allow_html=True)

                        # Gross Margin
                        gross_margin = ratios.get('profitability', {}).get('gross_margin')
                        if gross_margin is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{gross_margin * 100:.2f}%</div><div class="metric-label">Gross Margin</div></div>',
                                unsafe_allow_html=True)

                        # Operating Margin
                        op_margin = ratios.get('profitability', {}).get('operating_margin')
                        if op_margin is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{op_margin * 100:.2f}%</div><div class="metric-label">Operating Margin</div></div>',
                                unsafe_allow_html=True)

                        # Net Margin
                        net_margin = ratios.get('profitability', {}).get('net_margin')
                        if net_margin is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{net_margin * 100:.2f}%</div><div class="metric-label">Net Margin</div></div>',
                                unsafe_allow_html=True)

                    with col3:
                        st.markdown('<div class="section-header"><h4>Returns</h4></div>', unsafe_allow_html=True)

                        # ROE
                        roe = ratios.get('profitability', {}).get('roe')
                        if roe is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{roe * 100:.2f}%</div><div class="metric-label">Return on Equity</div></div>',
                                unsafe_allow_html=True)

                        # ROA
                        roa = ratios.get('profitability', {}).get('roa')
                        if roa is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{roa * 100:.2f}%</div><div class="metric-label">Return on Assets</div></div>',
                                unsafe_allow_html=True)

                        # Current Ratio
                        current_ratio = ratios.get('liquidity', {}).get('current_ratio')
                        if current_ratio is not None:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{current_ratio:.2f}</div><div class="metric-label">Current Ratio</div></div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error calculating financial ratios: {str(e)}")
                    logger.error(f"Error calculating financial ratios: {str(e)}")

            # Tab 2: Financial Analysis
            with tabs[1]:
                # Create statement analyzer
                statement_analyzer = get_statement_analyzer()

                # Create visualizer
                visualizer = get_visualizer()

                try:
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
                        if income_statement is not None and not income_statement.empty:
                            # Income statement summary
                            st.write("This analysis examines trends in revenue, profits, and margins over time.")

                            # Show key metrics
                            col1, col2 = st.columns(2)

                            with col1:
                                # Revenue chart
                                if 'key_metrics' in income_analysis and 'revenue' in income_analysis['key_metrics']:
                                    revenue_trend = income_analysis['key_metrics']['revenue']
                                    if revenue_trend and 'values' in revenue_trend:
                                        st.subheader("Revenue Trend")
                                        try:
                                            # Convert to DataFrame for visualization
                                            df_revenue = pd.DataFrame({
                                                'Period': list(revenue_trend['values'].keys()),
                                                'Revenue': list(revenue_trend['values'].values())
                                            })

                                            # Create chart
                                            fig = visualizer.plot_financial_statement_trend(
                                                pd.DataFrame(revenue_trend['values'], index=[0]).T,
                                                'income',
                                                ['Revenue'],
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)

                                            # Display growth rate
                                            if 'growth_rates' in income_analysis['key_metrics'] and 'revenue_growth' in \
                                                    income_analysis['key_metrics']['growth_rates']:
                                                growth = income_analysis['key_metrics']['growth_rates'][
                                                    'revenue_growth']
                                                if growth and 'latest' in growth:
                                                    growth_text = f"{growth['latest'] * 100:.2f}%" if growth[
                                                        'latest'] else "N/A"
                                                    growth_color = "#74f174" if growth['latest'] and growth[
                                                        'latest'] > 0 else "#faa1a4"
                                                    st.markdown(
                                                        f"<p>Latest Annual Growth: <span style='color:{growth_color};font-weight:bold;'>{growth_text}</span></p>",
                                                        unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"Error generating revenue chart: {str(e)}")
                                            logger.error(f"Error generating revenue chart: {str(e)}")
                                    else:
                                        st.info("Revenue trend data not available.")
                                else:
                                    st.info("Revenue data not available.")

                            with col2:
                                # Net Income chart
                                if 'key_metrics' in income_analysis and 'net_income' in income_analysis['key_metrics']:
                                    net_income_trend = income_analysis['key_metrics']['net_income']
                                    if net_income_trend and 'values' in net_income_trend:
                                        st.subheader("Net Income Trend")
                                        try:
                                            # Create chart
                                            fig = visualizer.plot_financial_statement_trend(
                                                pd.DataFrame(net_income_trend['values'], index=[0]).T,
                                                'income',
                                                ['Net Income'],
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)

                                            # Display growth rate
                                            if 'growth_rates' in income_analysis[
                                                'key_metrics'] and 'net_income_growth' in \
                                                    income_analysis['key_metrics']['growth_rates']:
                                                growth = income_analysis['key_metrics']['growth_rates'][
                                                    'net_income_growth']
                                                if growth and 'latest' in growth:
                                                    growth_text = f"{growth['latest'] * 100:.2f}%" if growth[
                                                        'latest'] else "N/A"
                                                    growth_color = "#74f174" if growth['latest'] and growth[
                                                        'latest'] > 0 else "#faa1a4"
                                                    st.markdown(
                                                        f"<p>Latest Annual Growth: <span style='color:{growth_color};font-weight:bold;'>{growth_text}</span></p>",
                                                        unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"Error generating net income chart: {str(e)}")
                                            logger.error(f"Error generating net income chart: {str(e)}")
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
                                    try:
                                        # Convert to DataFrame for visualization
                                        # The data needs to be transposed for plotting
                                        margin_values = {}
                                        for margin_name, values in margin_data.items():
                                            for period, value in values.items():
                                                if period not in margin_values:
                                                    margin_values[period] = {}
                                                margin_values[period][margin_name] = value

                                        margin_df = pd.DataFrame(margin_values).T

                                        # Plot margins
                                        fig = visualizer.plot_financial_statement_trend(
                                            margin_df,
                                            'income',
                                            margin_names,
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Display current margins
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            if 'gross_margin' in margins and margins['gross_margin'] and 'latest' in \
                                                    margins['gross_margin']:
                                                value = margins['gross_margin']['latest']
                                                st.metric("Gross Margin",
                                                          f"{value * 100:.2f}%" if value is not None else "N/A")

                                        with col2:
                                            if 'operating_margin' in margins and margins[
                                                'operating_margin'] and 'latest' in margins['operating_margin']:
                                                value = margins['operating_margin']['latest']
                                                st.metric("Operating Margin",
                                                          f"{value * 100:.2f}%" if value is not None else "N/A")

                                        with col3:
                                            if 'net_margin' in margins and margins['net_margin'] and 'latest' in \
                                                    margins['net_margin']:
                                                value = margins['net_margin']['latest']
                                                st.metric("Net Margin",
                                                          f"{value * 100:.2f}%" if value is not None else "N/A")
                                    except Exception as e:
                                        st.error(f"Error generating margins chart: {str(e)}")
                                        logger.error(f"Error generating margins chart: {str(e)}")
                                else:
                                    st.info("Margin trend data not available.")
                            else:
                                st.info("Margin data not available.")

                            # Display common-size income statement
                            st.subheader("Common-Size Income Statement")
                            st.write(
                                "Common-size analysis shows each item as a percentage of revenue, making it easier to understand proportions and trends.")

                            if 'common_size' in income_analysis:
                                try:
                                    common_size = income_analysis['common_size']
                                    if not isinstance(common_size, pd.DataFrame):
                                        st.info("Common-size data format is not as expected.")
                                    else:
                                        # Format as percentages
                                        formatted_df = common_size.applymap(
                                            lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")

                                        # Display as table
                                        st.dataframe(formatted_df)
                                except Exception as e:
                                    st.error(f"Error displaying common-size statement: {str(e)}")
                            else:
                                st.info("Common-size data not available.")

                            # Display raw income statement
                            with st.expander("View Raw Income Statement"):
                                st.dataframe(income_statement)
                        else:
                            st.warning("Income statement data not available for this company.")

                    # Balance Sheet Tab
                    with finance_tabs[1]:
                        st.subheader("Balance Sheet Analysis")

                        # Display balance sheet data
                        if balance_sheet is not None and not balance_sheet.empty:
                            # Balance sheet summary
                            st.write(
                                "This analysis examines the company's assets, liabilities, and equity positions over time.")

                            # Show key metrics
                            col1, col2 = st.columns(2)

                            with col1:
                                # Assets trend chart
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
                                        try:
                                            # Convert to DataFrame for visualization
                                            # The data needs to be transposed for plotting
                                            asset_values = {}
                                            for asset_name, values in assets_data.items():
                                                for period, value in values.items():
                                                    if period not in asset_values:
                                                        asset_values[period] = {}
                                                    asset_values[period][asset_name] = value

                                            assets_df = pd.DataFrame(asset_values).T

                                            # Plot assets
                                            st.subheader("Assets Trend")
                                            fig = visualizer.plot_financial_statement_trend(
                                                assets_df,
                                                'balance',
                                                asset_names,
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error generating assets chart: {str(e)}")
                                            logger.error(f"Error generating assets chart: {str(e)}")
                                    else:
                                        st.info("Asset trend data not available.")
                                else:
                                    st.info("Asset data not available.")

                            with col2:
                                # Liabilities & Equity trend chart
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
                                        try:
                                            # Convert to DataFrame for visualization
                                            # The data needs to be transposed for plotting
                                            values_dict = {}
                                            for name, values in trend_data.items():
                                                for period, value in values.items():
                                                    if period not in values_dict:
                                                        values_dict[period] = {}
                                                    values_dict[period][name] = value

                                            trend_df = pd.DataFrame(values_dict).T

                                            # Plot data
                                            st.subheader("Liabilities & Equity Trend")
                                            fig = visualizer.plot_financial_statement_trend(
                                                trend_df,
                                                'balance',
                                                trend_names,
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error generating liabilities chart: {str(e)}")
                                            logger.error(f"Error generating liabilities chart: {str(e)}")
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
                                    try:
                                        # Convert to DataFrame for visualization
                                        ratio_values = {}
                                        for ratio_name, values in ratio_data.items():
                                            for period, value in values.items():
                                                if period not in ratio_values:
                                                    ratio_values[period] = {}
                                                ratio_values[period][ratio_name] = value

                                        ratio_df = pd.DataFrame(ratio_values).T

                                        # Plot ratios
                                        fig = visualizer.plot_financial_statement_trend(
                                            ratio_df,
                                            'balance',
                                            ratio_names,
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Display current ratios
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            if 'current_ratio' in balance_ratios and balance_ratios[
                                                'current_ratio'] and 'latest' in balance_ratios['current_ratio']:
                                                value = balance_ratios['current_ratio']['latest']
                                                st.metric("Current Ratio",
                                                          f"{value:.2f}" if value is not None else "N/A")
                                                if value is not None:
                                                    if value >= 2:
                                                        st.markdown(
                                                            "<p style='color:#74f174'>Strong liquidity position</p>",
                                                            unsafe_allow_html=True)
                                                    elif value >= 1:
                                                        st.markdown("<p style='color:#fff59d'>Adequate liquidity</p>",
                                                                    unsafe_allow_html=True)
                                                    else:
                                                        st.markdown(
                                                            "<p style='color:#faa1a4'>Potential liquidity concerns</p>",
                                                            unsafe_allow_html=True)

                                        with col2:
                                            if 'debt_to_equity' in balance_ratios and balance_ratios[
                                                'debt_to_equity'] and 'latest' in balance_ratios['debt_to_equity']:
                                                value = balance_ratios['debt_to_equity']['latest']
                                                st.metric("Debt-to-Equity",
                                                          f"{value:.2f}" if value is not None else "N/A")
                                                if value is not None:
                                                    if value <= 0.5:
                                                        st.markdown("<p style='color:#74f174'>Low leverage</p>",
                                                                    unsafe_allow_html=True)
                                                    elif value <= 1.5:
                                                        st.markdown("<p style='color:#fff59d'>Moderate leverage</p>",
                                                                    unsafe_allow_html=True)
                                                    else:
                                                        st.markdown("<p style='color:#faa1a4'>High leverage</p>",
                                                                    unsafe_allow_html=True)

                                        with col3:
                                            if 'debt_to_assets' in balance_ratios and balance_ratios[
                                                'debt_to_assets'] and 'latest' in balance_ratios['debt_to_assets']:
                                                value = balance_ratios['debt_to_assets']['latest']
                                                st.metric("Debt-to-Assets",
                                                          f"{value:.2f}" if value is not None else "N/A")
                                                if value is not None:
                                                    if value <= 0.3:
                                                        st.markdown("<p style='color:#74f174'>Low debt burden</p>",
                                                                    unsafe_allow_html=True)
                                                    elif value <= 0.6:
                                                        st.markdown("<p style='color:#fff59d'>Moderate debt burden</p>",
                                                                    unsafe_allow_html=True)
                                                    else:
                                                        st.markdown("<p style='color:#faa1a4'>High debt burden</p>",
                                                                    unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error generating position ratios chart: {str(e)}")
                                        logger.error(f"Error generating position ratios chart: {str(e)}")
                                else:
                                    st.info("Financial position ratio trend data not available.")
                            else:
                                st.info("Financial position ratio data not available.")

                            # Display common-size balance sheet
                            st.subheader("Common-Size Balance Sheet")
                            st.write(
                                "Common-size analysis shows each item as a percentage of total assets, helping identify important proportions in the company's financial structure.")

                            if 'common_size' in balance_analysis:
                                try:
                                    common_size = balance_analysis['common_size']
                                    if not isinstance(common_size, pd.DataFrame):
                                        st.info("Common-size data format is not as expected.")
                                    else:
                                        # Format as percentages
                                        formatted_df = common_size.applymap(
                                            lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")

                                        # Display as table
                                        st.dataframe(formatted_df)
                                except Exception as e:
                                    st.error(f"Error displaying common-size statement: {str(e)}")
                            else:
                                st.info("Common-size data not available.")

                            # Display raw balance sheet
                            with st.expander("View Raw Balance Sheet"):
                                st.dataframe(balance_sheet)
                        else:
                            st.warning("Balance sheet data not available for this company.")

                    # Cash Flow Tab
                    with finance_tabs[2]:
                        st.subheader("Cash Flow Analysis")

                        # Display cash flow data
                        if cash_flow is not None and not cash_flow.empty:
                            # Cash flow summary
                            st.write("This analysis examines the company's ability to generate and utilize cash.")

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
                                    try:
                                        # Convert to DataFrame for visualization
                                        cf_values = {}
                                        for name, values in cf_data.items():
                                            for period, value in values.items():
                                                if period not in cf_values:
                                                    cf_values[period] = {}
                                                cf_values[period][name] = value

                                        cf_df = pd.DataFrame(cf_values).T

                                        # Plot cash flows
                                        fig = visualizer.plot_financial_statement_trend(
                                            cf_df,
                                            'cash_flow',
                                            cf_names,
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Display key cash flow metrics
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Operating Cash Flow
                                            if 'operating_cash_flow' in cash_flow_analysis['key_metrics'] and \
                                                    cash_flow_analysis['key_metrics'][
                                                        'operating_cash_flow'] and 'latest' in \
                                                    cash_flow_analysis['key_metrics']['operating_cash_flow']:
                                                ocf = cash_flow_analysis['key_metrics']['operating_cash_flow']['latest']
                                                st.metric("Operating Cash Flow (Latest)",
                                                          f"${ocf / 1e6:.2f}M" if ocf is not None else "N/A")

                                            # Investing Cash Flow
                                            if 'investing_cash_flow' in cash_flow_analysis['key_metrics'] and \
                                                    cash_flow_analysis['key_metrics'][
                                                        'investing_cash_flow'] and 'latest' in \
                                                    cash_flow_analysis['key_metrics']['investing_cash_flow']:
                                                icf = cash_flow_analysis['key_metrics']['investing_cash_flow']['latest']
                                                st.metric("Investing Cash Flow (Latest)",
                                                          f"${icf / 1e6:.2f}M" if icf is not None else "N/A")

                                        with col2:
                                            # Free Cash Flow
                                            if 'free_cash_flow' in cash_flow_analysis['key_metrics'] and \
                                                    cash_flow_analysis['key_metrics']['free_cash_flow'] and 'latest' in \
                                                    cash_flow_analysis['key_metrics']['free_cash_flow']:
                                                fcf = cash_flow_analysis['key_metrics']['free_cash_flow']['latest']
                                                st.metric("Free Cash Flow (Latest)",
                                                          f"${fcf / 1e6:.2f}M" if fcf is not None else "N/A")
                                                if fcf is not None:
                                                    if fcf > 0:
                                                        st.markdown(
                                                            "<p style='color:#74f174'>Positive free cash flow indicates the company is generating more cash than it's spending on capital investments.</p>",
                                                            unsafe_allow_html=True)
                                                    else:
                                                        st.markdown(
                                                            "<p style='color:#faa1a4'>Negative free cash flow may indicate heavy investment or operational challenges.</p>",
                                                            unsafe_allow_html=True)

                                            # Financing Cash Flow
                                            if 'financing_cash_flow' in cash_flow_analysis['key_metrics'] and \
                                                    cash_flow_analysis['key_metrics'][
                                                        'financing_cash_flow'] and 'latest' in \
                                                    cash_flow_analysis['key_metrics']['financing_cash_flow']:
                                                fin_cf = cash_flow_analysis['key_metrics']['financing_cash_flow'][
                                                    'latest']
                                                st.metric("Financing Cash Flow (Latest)",
                                                          f"${fin_cf / 1e6:.2f}M" if fin_cf is not None else "N/A")
                                    except Exception as e:
                                        st.error(f"Error generating cash flow chart: {str(e)}")
                                        logger.error(f"Error generating cash flow chart: {str(e)}")
                                else:
                                    st.info("Cash flow component data not available.")
                            else:
                                st.info("Cash flow component data not available.")

                            # Cash flow quality analysis
                            st.subheader("Cash Flow Quality")
                            if 'key_metrics' in cash_flow_analysis and 'cf_quality' in cash_flow_analysis[
                                'key_metrics']:
                                cf_quality = cash_flow_analysis['key_metrics']['cf_quality']

                                if 'ocf_to_net_income' in cf_quality and cf_quality['ocf_to_net_income'] and 'latest' in \
                                        cf_quality['ocf_to_net_income']:
                                    ocf_to_ni = cf_quality['ocf_to_net_income']['latest']
                                    st.metric("Operating Cash Flow to Net Income",
                                              f"{ocf_to_ni:.2f}" if ocf_to_ni is not None else "N/A")

                                    if ocf_to_ni is not None:
                                        if ocf_to_ni > 1.2:
                                            st.markdown(
                                                "<p style='color:#74f174'>Strong cash conversion (>1.2x) indicates high earnings quality.</p>",
                                                unsafe_allow_html=True)
                                        elif ocf_to_ni > 0.8:
                                            st.markdown(
                                                "<p style='color:#fff59d'>Adequate cash conversion (0.8-1.2x) suggests reasonable earnings quality.</p>",
                                                unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                "<p style='color:#faa1a4'>Low cash conversion (<0.8x) may indicate earnings quality concerns.</p>",
                                                unsafe_allow_html=True)
                            else:
                                st.info("Cash flow quality metrics not available.")

                            # Cash Flow Sustainability Analysis
                            if 'key_metrics' in cash_flow_analysis and 'free_cash_flow' in cash_flow_analysis[
                                'key_metrics'] and cash_flow_analysis['key_metrics']['free_cash_flow']:
                                st.subheader("Cash Flow Sustainability Analysis")
                                fcf_trend = cash_flow_analysis['key_metrics']['free_cash_flow']

                                # Check if we have FCF values
                                if 'values' in fcf_trend and len(fcf_trend['values']) > 0:
                                    # Calculate FCF growth and stability
                                    fcf_values = list(fcf_trend['values'].values())
                                    fcf_avg = sum(fcf_values) / len(fcf_values)

                                    # Count positive FCF periods
                                    positive_periods = sum(1 for v in fcf_values if v > 0)
                                    positive_percent = positive_periods / len(fcf_values) * 100

                                    # Calculate coefficient of variation (stability measure)
                                    if len(fcf_values) > 1:
                                        fcf_std = np.std(fcf_values)
                                        fcf_cv = fcf_std / abs(fcf_avg) if fcf_avg != 0 else float('inf')

                                        # Display metrics
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("FCF Positive Periods",
                                                      f"{positive_periods} of {len(fcf_values)} ({positive_percent:.1f}%)")

                                        with col2:
                                            if fcf_cv < float('inf'):
                                                st.metric("FCF Volatility", f"{fcf_cv:.2f}")
                                                if fcf_cv < 0.2:
                                                    st.markdown(
                                                        "<p style='color:#74f174'>Low volatility indicates stable and predictable cash flows.</p>",
                                                        unsafe_allow_html=True)
                                                elif fcf_cv < 0.5:
                                                    st.markdown(
                                                        "<p style='color:#fff59d'>Moderate volatility is typical for many businesses.</p>",
                                                        unsafe_allow_html=True)
                                                else:
                                                    st.markdown(
                                                        "<p style='color:#faa1a4'>High volatility suggests unpredictable cash flows.</p>",
                                                        unsafe_allow_html=True)

                                        # Overall FCF assessment
                                        st.subheader("Free Cash Flow Assessment")
                                        if positive_percent > 75 and fcf_avg > 0:
                                            if fcf_cv < 0.3:
                                                st.markdown(
                                                    "<p style='color:#74f174'>Strong and stable free cash flow generation, suggesting financial flexibility and potential for consistent shareholder returns.</p>",
                                                    unsafe_allow_html=True)
                                            else:
                                                st.markdown(
                                                    "<p style='color:#fff59d'>Generally positive but somewhat volatile cash flow generation. The company might benefit from strategies to stabilize cash flows.</p>",
                                                    unsafe_allow_html=True)
                                        elif positive_percent > 50 and fcf_avg > 0:
                                            st.markdown(
                                                "<p style='color:#fff59d'>Adequate but inconsistent free cash flow generation. This may limit financial flexibility during downturns.</p>",
                                                unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                "<p style='color:#faa1a4'>Weak or negative free cash flow generation over multiple periods suggests potential sustainability concerns. Monitor carefully.</p>",
                                                unsafe_allow_html=True)

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
                        ratio_categories = [cat for cat in ratios.keys() if ratios[cat]]
                        if ratio_categories:
                            ratio_tabs = st.tabs([cat.capitalize() for cat in ratio_categories])

                            # For each category, show ratios and comparison
                            for i, category in enumerate(ratio_categories):
                                with ratio_tabs[i]:
                                    st.subheader(f"{category.capitalize()} Ratios")

                                    # Get ratios for this category
                                    category_ratios = ratios.get(category, {})
                                    category_analysis = ratio_analysis.get(category, {})

                                    if category_ratios:
                                        try:
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

                                                # Format columns
                                                formatted_df = ratio_df.copy()
                                                formatted_df["Company Value"] = formatted_df["Company Value"].apply(
                                                    lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                                                formatted_df["Sector Average"] = formatted_df["Sector Average"].apply(
                                                    lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                                                formatted_df["Difference (%)"] = formatted_df["Difference (%)"].apply(
                                                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

                                                st.dataframe(formatted_df, use_container_width=True)

                                                # For valuation ratios, add interpretation
                                                if category == 'valuation':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Valuation Ratios Explanation:**

                                                    - **Lower values** generally indicate a **more attractive valuation**.
                                                    - **Higher values** suggest the market is pricing in significant **future growth** or other positive factors.
                                                    - **Comparison to sector average** helps determine if the stock is potentially undervalued or overvalued relative to peers.
                                                    """
                                                    st.markdown(interpretation)

                                                # For profitability ratios, add interpretation
                                                elif category == 'profitability':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Profitability Ratios Explanation:**

                                                    - **Higher values** are generally **better**, indicating more efficient use of capital and resources.
                                                    - **Gross margin** indicates pricing power and manufacturing efficiency.
                                                    - **Operating margin** reflects operational efficiency excluding financing and taxes.
                                                    - **Net margin** shows overall profitability after all expenses.
                                                    - **ROE and ROA** measure how efficiently the company generates profits from its equity and assets.
                                                    """
                                                    st.markdown(interpretation)

                                                # For liquidity ratios, add interpretation
                                                elif category == 'liquidity':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Liquidity Ratios Explanation:**

                                                    - **Higher values** generally indicate **stronger short-term financial health**.
                                                    - **Current ratio** > 2.0 suggests excellent liquidity.
                                                    - **Current ratio** between 1.0-2.0 indicates adequate liquidity.
                                                    - **Current ratio** < 1.0 may signal potential liquidity issues.
                                                    - **Quick ratio** is a more conservative measure excluding inventory.
                                                    - **Cash ratio** is the most stringent measure, considering only cash and equivalents.
                                                    """
                                                    st.markdown(interpretation)

                                                # For leverage ratios, add interpretation
                                                elif category == 'leverage':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Leverage Ratios Explanation:**

                                                    - **Lower values** generally indicate **lower financial risk**.
                                                    - **Debt-to-Equity** < 0.5 suggests conservative financing.
                                                    - **Debt-to-Equity** between 0.5-1.5 indicates moderate leverage.
                                                    - **Debt-to-Equity** > 1.5 may signal higher financial risk.
                                                    - **Debt-to-Assets** shows the percentage of assets financed by debt.
                                                    - **Interest Coverage** measures ability to pay interest expenses (higher is better).
                                                    """
                                                    st.markdown(interpretation)

                                                # For efficiency ratios, add interpretation
                                                elif category == 'efficiency':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Efficiency Ratios Explanation:**

                                                    - **Higher values** generally indicate **better operational efficiency**.
                                                    - **Asset Turnover** measures how efficiently assets generate revenue.
                                                    - **Inventory Turnover** indicates how quickly inventory is sold (higher is better).
                                                    - **Receivables Turnover** shows how quickly the company collects payments.
                                                    - **These ratios vary significantly by industry**, making sector comparison essential.
                                                    """
                                                    st.markdown(interpretation)

                                                # For growth ratios, add interpretation
                                                elif category == 'growth':
                                                    st.subheader("Interpretation")
                                                    interpretation = """
                                                    **Growth Ratios Explanation:**

                                                    - **Higher values** indicate **stronger growth**.
                                                    - **Revenue growth** shows top-line expansion.
                                                    - **Earnings growth** indicates bottom-line improvement.
                                                    - **Consistent positive growth** is generally preferable to volatile performance.
                                                    - **Growth should be analyzed in context** of the company's size, maturity, and industry.
                                                    """
                                                    st.markdown(interpretation)
                                                else:
                                                    st.info("No ratio details available for display.")
                                            else:
                                                st.info(f"No {category} ratios available for this company.")
                                        except Exception as e:
                                            st.error(f"Error processing {category} ratios: {str(e)}")

                                # Get sector key ratios
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

                                # Financial Health Score
                            st.subheader("Financial Health Assessment")

                            # Calculate financial health score
                            health_score = statement_analyzer.calculate_financial_health_score(
                                income_analysis, balance_analysis, cash_flow_analysis
                            )

                            # Display financial health score gauge
                            if health_score and health_score.get('overall_score') is not None:
                                try:
                                    fig = visualizer.plot_financial_health_score(
                                        health_score,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Display score interpretation
                                    rating = health_score.get('rating', '')
                                    if rating:
                                        rating_color = "#74f174" if rating in ["Excellent",
                                                                               "Strong"] else "#fff59d" if rating in [
                                            "Good", "Moderate"] else "#faa1a4"
                                        st.markdown(
                                            f"<h3 style='color:{rating_color};text-align:center;'>{rating} Financial Health</h3>",
                                            unsafe_allow_html=True)

                                    # Display health score components as metrics
                                    st.subheader("Financial Health Components")
                                    components = health_score.get('components', {})
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        profitability = components.get('profitability')
                                        if profitability is not None:
                                            st.metric("Profitability", f"{profitability}/100")
                                            if profitability >= 75:
                                                st.markdown("<p style='color:#74f174'>Strong</p>",
                                                            unsafe_allow_html=True)
                                            elif profitability >= 50:
                                                st.markdown("<p style='color:#fff59d'>Adequate</p>",
                                                            unsafe_allow_html=True)
                                            else:
                                                st.markdown("<p style='color:#faa1a4'>Weak</p>",
                                                            unsafe_allow_html=True)

                                    with col2:
                                        liquidity = components.get('liquidity')
                                        if liquidity is not None:
                                            st.metric("Liquidity", f"{liquidity}/100")
                                            if liquidity >= 75:
                                                st.markdown("<p style='color:#74f174'>Strong</p>",
                                                            unsafe_allow_html=True)
                                            elif liquidity >= 50:
                                                st.markdown("<p style='color:#fff59d'>Adequate</p>",
                                                            unsafe_allow_html=True)
                                            else:
                                                st.markdown("<p style='color:#faa1a4'>Weak</p>",
                                                            unsafe_allow_html=True)

                                    with col3:
                                        solvency = components.get('solvency')
                                        if solvency is not None:
                                            st.metric("Solvency", f"{solvency}/100")
                                            if solvency >= 75:
                                                st.markdown("<p style='color:#74f174'>Strong</p>",
                                                            unsafe_allow_html=True)
                                            elif solvency >= 50:
                                                st.markdown("<p style='color:#fff59d'>Adequate</p>",
                                                            unsafe_allow_html=True)
                                            else:
                                                st.markdown("<p style='color:#faa1a4'>Weak</p>",
                                                            unsafe_allow_html=True)

                                    with col4:
                                        cash_flow = components.get('cash_flow')
                                        if cash_flow is not None:
                                            st.metric("Cash Flow", f"{cash_flow}/100")
                                            if cash_flow >= 75:
                                                st.markdown("<p style='color:#74f174'>Strong</p>",
                                                            unsafe_allow_html=True)
                                            elif cash_flow >= 50:
                                                st.markdown("<p style='color:#fff59d'>Adequate</p>",
                                                            unsafe_allow_html=True)
                                            else:
                                                st.markdown("<p style='color:#faa1a4'>Weak</p>",
                                                            unsafe_allow_html=True)

                                    # Add interpretation
                                    st.subheader("Interpretation")
                                    if health_score.get('overall_score', 0) >= 75:
                                        st.markdown("""
                                                                **Strong Financial Health**: The company demonstrates excellent financial stability with strong profitability, adequate liquidity, manageable debt levels, and consistent cash generation. This suggests:

                                                                - Lower financial risk profile
                                                                - Greater ability to weather economic downturns
                                                                - Flexibility to invest in growth opportunities
                                                                - Capacity to return value to shareholders
                                                                """)
                                    elif health_score.get('overall_score', 0) >= 60:
                                        st.markdown("""
                                                                **Good Financial Health**: The company shows good overall financial condition with satisfactory metrics across most categories. This suggests:

                                                                - Reasonable financial stability
                                                                - Ability to manage routine financial obligations
                                                                - Some flexibility for strategic initiatives
                                                                - Consider monitoring weaker components for improvement
                                                                """)
                                    elif health_score.get('overall_score', 0) >= 45:
                                        st.markdown("""
                                                                **Moderate Financial Health**: The company shows adequate but mixed financial performance with some potential areas of concern. This suggests:

                                                                - Moderate financial risk profile
                                                                - May face challenges during economic downturns
                                                                - Limited financial flexibility
                                                                - Should focus on improving weaker components
                                                                """)
                                    else:
                                        st.markdown("""
                                                                **Concerning Financial Health**: The company shows significant weaknesses in multiple financial categories. This suggests:

                                                                - Elevated financial risk profile
                                                                - Vulnerability to economic challenges
                                                                - Limited ability to pursue growth opportunities
                                                                - May face challenges meeting financial obligations
                                                                - Consider strategies to improve profitability, reduce debt, or strengthen cash flows
                                                                """)
                                except Exception as e:
                                    st.error(
                                        f"Error generating financial health visualization: {str(e)}")
                                    logger.error(
                                        f"Error generating financial health visualization: {str(e)}")
                            else:
                                st.warning(
                                    "Financial health score could not be calculated due to insufficient data.")
                        else:
                            st.warning("No financial ratios available for analysis.")

                except Exception as e:
                    st.error(f"Error in financial analysis tab: {str(e)}")
                    logger.error(f"Error in financial analysis tab: {traceback.format_exc()}")

            # Tab 3: Valuation
            with tabs[2]:
                st.subheader("Company Valuation Analysis")

                try:
                    # Initialize valuation factory and models
                    from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
                    from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation

                    valuation_factory = ValuationFactory(get_data_loader())
                    dcf_model = AdvancedDCFValuation(get_data_loader())

                    # Prepare financial data for valuation
                    financial_data = {
                        'income_statement': income_statement,
                        'balance_sheet': balance_sheet,
                        'cash_flow': cash_flow,
                        'market_data': {
                            'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                            'market_cap': company_info.get('market_cap'),
                            'shares_outstanding': company_info.get('market_cap') / price_data['Close'].iloc[-1]
                            if company_info.get('market_cap') and not price_data.empty else None,
                            'beta': company_info.get('beta')
                        }
                    }

                    # Based on sector, show appropriate valuation interface
                    # Energy sector specific valuation tab
                    if sector == "Energy":
                        st.subheader("Energy Sector Valuation Analysis")

                        # Determine energy subsector and business model
                        try:
                            energy_subsector, business_model = st.columns(2)

                            with energy_subsector:
                                subsector = st.selectbox(
                                    "Energy subsector",
                                    ["oil_gas", "utilities", "renewables", "coal"],
                                    format_func=lambda x: {
                                        "oil_gas": "Oil & Gas",
                                        "utilities": "Utilities",
                                        "renewables": "Renewable Energy",
                                        "coal": "Coal"
                                    }.get(x, x.replace('_', ' ').title())
                                )

                            with business_model:
                                if subsector == "oil_gas":
                                    model = st.selectbox(
                                        "Business model",
                                        ["upstream", "integrated", "midstream", "downstream"],
                                        format_func=lambda x: x.title()
                                    )
                                elif subsector == "utilities":
                                    model = st.selectbox(
                                        "Utility type",
                                        ["regulated", "merchant", "integrated"],
                                        format_func=lambda x: x.title()
                                    )
                                elif subsector == "renewables":
                                    model = st.selectbox(
                                        "Renewable focus",
                                        ["solar", "wind", "hydro", "diversified"],
                                        format_func=lambda x: x.title()
                                    )
                                else:
                                    model = "mining"

                            # Initialize energy valuation model
                            st.info("Configuring energy sector-specific valuation parameters...")

                            # Parameters section
                            st.subheader("Key Valuation Parameters")

                            params_col1, params_col2 = st.columns(2)
                            with params_col1:
                                # Common parameters across all energy types
                                discount_rate = st.slider(
                                    "Discount Rate (%)",
                                    min_value=5.0,
                                    max_value=15.0,
                                    value=10.0,
                                    step=0.1,
                                    help="Higher for more volatile subsectors like upstream oil & gas, lower for utilities"
                                ) / 100

                                forecast_years = st.slider(
                                    "Forecast Years",
                                    min_value=5,
                                    max_value=20,
                                    value=10,
                                    step=1,
                                    help="Longer forecasts typical for utilities and midstream with long-term contracts"
                                )

                            with params_col2:
                                # Subsector-specific parameters
                                if subsector == "oil_gas":
                                    if model in ["upstream", "integrated"]:
                                        terminal_price = st.slider(
                                            "Long-term Oil Price ($/bbl)",
                                            min_value=40,
                                            max_value=100,
                                            value=65,
                                            step=5
                                        )

                                        reserve_life = st.slider(
                                            "Reserve Life (years)",
                                            min_value=5,
                                            max_value=20,
                                            value=12,
                                            step=1,
                                            help="Average number of years of production at current rates"
                                        )
                                    elif model == "midstream":
                                        contract_coverage = st.slider(
                                            "Contract Coverage (%)",
                                            min_value=50,
                                            max_value=100,
                                            value=80,
                                            step=5,
                                            help="Percentage of capacity under long-term contracts"
                                        ) / 100

                                        contract_duration = st.slider(
                                            "Average Contract Duration (years)",
                                            min_value=2,
                                            max_value=15,
                                            value=8,
                                            step=1
                                        )
                                elif subsector == "utilities":
                                    allowed_roe = st.slider(
                                        "Allowed Return on Equity (%)",
                                        min_value=7.0,
                                        max_value=12.0,
                                        value=9.5,
                                        step=0.1,
                                        help="Regulated return rate approved by utility commissions"
                                    ) / 100

                                    rate_base_growth = st.slider(
                                        "Rate Base Growth (%)",
                                        min_value=1.0,
                                        max_value=8.0,
                                        value=4.0,
                                        step=0.5,
                                        help="Annual growth rate of regulated asset base"
                                    ) / 100
                                elif subsector == "renewables":
                                    capacity_factor = st.slider(
                                        "Capacity Factor (%)",
                                        min_value=20,
                                        max_value=60,
                                        value=35,
                                        step=1,
                                        help="Percentage of theoretical maximum output actually produced annually"
                                    ) / 100

                                    ppa_coverage = st.slider(
                                        "PPA Coverage (%)",
                                        min_value=0,
                                        max_value=100,
                                        value=80,
                                        step=5,
                                        help="Percentage of production sold under Power Purchase Agreements"
                                    ) / 100

                            # Valuation methods selection
                            st.subheader("Valuation Methods")

                            valuation_methods = []

                            if subsector == "oil_gas":
                                if model == "upstream":
                                    valuation_methods = st.multiselect(
                                        "Select valuation methods",
                                        ["DCF", "NAV (Reserve-Based)", "Multiples"],
                                        default=["DCF", "NAV (Reserve-Based)"]
                                    )
                                elif model == "integrated":
                                    valuation_methods = st.multiselect(
                                        "Select valuation methods",
                                        ["DCF", "Sum-of-the-Parts", "Multiples"],
                                        default=["DCF", "Sum-of-the-Parts"]
                                    )
                                elif model == "midstream":
                                    valuation_methods = st.multiselect(
                                        "Select valuation methods",
                                        ["DCF", "Dividend Discount Model", "Multiples"],
                                        default=["DCF", "Dividend Discount Model"]
                                    )
                            elif subsector == "utilities":
                                valuation_methods = st.multiselect(
                                    "Select valuation methods",
                                    ["DCF", "Dividend Discount Model", "Regulated Asset Base"],
                                    default=["DCF", "Regulated Asset Base"]
                                )
                            elif subsector == "renewables":
                                valuation_methods = st.multiselect(
                                    "Select valuation methods",
                                    ["Project-Based DCF", "Multiples", "Real Options"],
                                    default=["Project-Based DCF", "Multiples"]
                                )
                            else:
                                valuation_methods = st.multiselect(
                                    "Select valuation methods",
                                    ["DCF", "Multiples"],
                                    default=["DCF"]
                                )

                            # Only run valuation if user selects methods
                            if not valuation_methods:
                                st.warning("Please select at least one valuation method to proceed.")
                            else:
                                with st.spinner("Running energy sector valuation..."):
                                    try:
                                        # Initialize energy sector valuation model
                                        from StockAnalysisSystem.src.valuation.sector_specific.energy_sector import \
                                            EnergySectorValuation
                                        energy_valuation = EnergySectorValuation(get_data_loader())

                                        # Set up the parameters based on user inputs
                                        energy_params = {
                                            'energy_type': subsector,
                                            'business_model': model
                                        }

                                        # Add custom parameters from user input
                                        if subsector == "oil_gas":
                                            if model in ["upstream", "integrated"]:
                                                energy_params['terminal_price'] = terminal_price
                                                energy_params['reserve_life'] = reserve_life
                                            elif model == "midstream":
                                                energy_params['contract_coverage'] = contract_coverage
                                                energy_params['contract_duration'] = contract_duration
                                        elif subsector == "utilities":
                                            energy_params['allowed_roe'] = allowed_roe
                                            energy_params['rate_base_growth'] = rate_base_growth
                                        elif subsector == "renewables":
                                            energy_params['capacity_factor'] = capacity_factor
                                            energy_params['ppa_coverage'] = ppa_coverage

                                        # Modify financial data with custom parameters
                                        financial_data_with_params = financial_data.copy() if financial_data else {}
                                        financial_data_with_params['custom_params'] = {
                                            'discount_rate': discount_rate,
                                            'forecast_years': forecast_years
                                        }

                                        # Special handling for various subsectors and models
                                        if subsector == "oil_gas" and model == "upstream":
                                            if "NAV (Reserve-Based)" in valuation_methods:
                                                # Get upstream metrics and calculate NAV
                                                upstream_metrics = energy_valuation._calculate_upstream_metrics(
                                                    financial_data_with_params)
                                                nav_result = energy_valuation._reserve_based_nav(ticker,
                                                                                                 financial_data_with_params,
                                                                                                 upstream_metrics)

                                            if "DCF" in valuation_methods:
                                                # Run upstream-specific DCF
                                                upstream_metrics = energy_valuation._calculate_upstream_metrics(
                                                    financial_data_with_params)
                                                dcf_result = energy_valuation._cyclical_upstream_dcf(ticker,
                                                                                                     financial_data_with_params,
                                                                                                     upstream_metrics)

                                            if "Multiples" in valuation_methods:
                                                # Run upstream-specific multiples valuation
                                                upstream_metrics = energy_valuation._calculate_upstream_metrics(
                                                    financial_data_with_params)
                                                multiples_result = energy_valuation._upstream_multiples_valuation(
                                                    ticker, financial_data_with_params, upstream_metrics)

                                            # Combine results
                                            valuation_result = energy_valuation.value_upstream_company(ticker,
                                                                                                       financial_data_with_params)

                                        elif subsector == "oil_gas" and model == "integrated":
                                            if "Sum-of-the-Parts" in valuation_methods:
                                                # Get integrated metrics and calculate SOTP
                                                integrated_metrics = energy_valuation._calculate_integrated_metrics(
                                                    financial_data_with_params)
                                                sotp_result = energy_valuation._sum_of_the_parts_valuation(ticker,
                                                                                                           financial_data_with_params,
                                                                                                           integrated_metrics)

                                            if "DCF" in valuation_methods:
                                                # Run integrated-specific DCF
                                                integrated_metrics = energy_valuation._calculate_integrated_metrics(
                                                    financial_data_with_params)
                                                dcf_result = energy_valuation._cyclical_integrated_dcf(ticker,
                                                                                                       financial_data_with_params,
                                                                                                       integrated_metrics)

                                            if "Multiples" in valuation_methods:
                                                # Run integrated-specific multiples valuation
                                                integrated_metrics = energy_valuation._calculate_integrated_metrics(
                                                    financial_data_with_params)
                                                multiples_result = energy_valuation._integrated_multiples_valuation(
                                                    ticker, financial_data_with_params, integrated_metrics)

                                            # Combine results
                                            valuation_result = energy_valuation.value_integrated_oil_gas(ticker,
                                                                                                         financial_data_with_params)

                                        elif subsector == "oil_gas" and model == "midstream":
                                            if "Dividend Discount Model" in valuation_methods:
                                                # Get midstream metrics and calculate DDM
                                                midstream_metrics = energy_valuation._calculate_midstream_metrics(
                                                    financial_data_with_params)
                                                ddm_result = energy_valuation._midstream_ddm(ticker,
                                                                                             financial_data_with_params,
                                                                                             midstream_metrics)

                                            if "DCF" in valuation_methods:
                                                # Run midstream-specific DCF
                                                midstream_metrics = energy_valuation._calculate_midstream_metrics(
                                                    financial_data_with_params)
                                                dcf_result = energy_valuation._midstream_dcf(ticker,
                                                                                             financial_data_with_params,
                                                                                             midstream_metrics)

                                            if "Multiples" in valuation_methods:
                                                # Run midstream-specific multiples valuation
                                                midstream_metrics = energy_valuation._calculate_midstream_metrics(
                                                    financial_data_with_params)
                                                multiples_result = energy_valuation._midstream_multiples_valuation(
                                                    ticker, financial_data_with_params, midstream_metrics)

                                            # Combine results
                                            valuation_result = energy_valuation.value_midstream_company(ticker,
                                                                                                        financial_data_with_params)

                                        elif subsector == "utilities":
                                            if "Dividend Discount Model" in valuation_methods:
                                                # Get utility metrics and calculate DDM
                                                utility_metrics = energy_valuation._calculate_utility_metrics(
                                                    financial_data_with_params)
                                                ddm_result = energy_valuation._utility_ddm(ticker,
                                                                                           financial_data_with_params,
                                                                                           utility_metrics)

                                            if "Regulated Asset Base" in valuation_methods:
                                                # Run RAB valuation
                                                utility_metrics = energy_valuation._calculate_utility_metrics(
                                                    financial_data_with_params)
                                                rab_result = energy_valuation._regulated_asset_base_valuation(ticker,
                                                                                                              financial_data_with_params,
                                                                                                              utility_metrics)

                                            if "DCF" in valuation_methods:
                                                # Run utility-specific DCF
                                                utility_metrics = energy_valuation._calculate_utility_metrics(
                                                    financial_data_with_params)
                                                dcf_result = energy_valuation._utility_dcf(ticker,
                                                                                           financial_data_with_params,
                                                                                           utility_metrics)

                                            # Combine results
                                            valuation_result = energy_valuation.value_utility_company(ticker,
                                                                                                      financial_data_with_params)

                                        elif subsector == "renewables":
                                            if "Project-Based DCF" in valuation_methods:
                                                # Get renewable metrics and calculate project DCF
                                                renewable_metrics = energy_valuation._calculate_renewable_metrics(
                                                    financial_data_with_params)
                                                dcf_result = energy_valuation._renewable_project_dcf(ticker,
                                                                                                     financial_data_with_params,
                                                                                                     renewable_metrics)

                                            if "Multiples" in valuation_methods:
                                                # Run renewable-specific multiples valuation
                                                renewable_metrics = energy_valuation._calculate_renewable_metrics(
                                                    financial_data_with_params)
                                                multiples_result = energy_valuation._renewable_multiples_valuation(
                                                    ticker, financial_data_with_params, renewable_metrics)

                                            if "Real Options" in valuation_methods:
                                                # Run real options valuation
                                                renewable_metrics = energy_valuation._calculate_renewable_metrics(
                                                    financial_data_with_params)
                                                options_result = energy_valuation._renewable_options_valuation(ticker,
                                                                                                               financial_data_with_params,
                                                                                                               renewable_metrics)

                                            # Combine results
                                            valuation_result = energy_valuation.value_renewable_company(ticker,
                                                                                                        financial_data_with_params)

                                        else:
                                            # Fall back to standard energy DCF
                                            valuation_result = energy_valuation.energy_sector_dcf(ticker,
                                                                                                  financial_data_with_params)

                                        # Display valuation results
                                        st.subheader("Valuation Results")

                                        # Get current market price for comparison
                                        current_price = price_data['Close'].iloc[-1] if not price_data.empty else None

                                        # Display blended value per share
                                        if valuation_result.get('blended_value_per_share') is not None:
                                            value_per_share = valuation_result.get('blended_value_per_share')
                                        else:
                                            value_per_share = valuation_result.get('value_per_share')

                                        conservative_value = valuation_result.get('conservative_value')

                                        # Create metrics display
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            if value_per_share and current_price:
                                                upside = ((value_per_share / current_price) - 1) * 100
                                                upside_text = f"{upside:.1f}% {'Upside' if upside > 0 else 'Downside'}"

                                                st.metric(
                                                    "Estimated Value",
                                                    f"${value_per_share:.2f}",
                                                    upside_text,
                                                    delta_color="normal" if upside > 0 else "inverse"
                                                )
                                            else:
                                                st.metric("Estimated Value",
                                                          f"${value_per_share:.2f}" if value_per_share else "N/A")

                                        with col2:
                                            if conservative_value and current_price:
                                                cons_upside = ((conservative_value / current_price) - 1) * 100
                                                cons_text = f"{cons_upside:.1f}% {'Upside' if cons_upside > 0 else 'Downside'}"

                                                st.metric(
                                                    "Conservative Value",
                                                    f"${conservative_value:.2f}",
                                                    cons_text,
                                                    delta_color="normal" if cons_upside > 0 else "inverse"
                                                )
                                            else:
                                                st.metric("Conservative Value",
                                                          f"${conservative_value:.2f}" if conservative_value else "N/A")

                                        with col3:
                                            enterprise_value = valuation_result.get('enterprise_value')
                                            equity_value = valuation_result.get('equity_value')

                                            if enterprise_value:
                                                st.metric("Enterprise Value", f"${enterprise_value / 1e9:.2f}B")
                                            else:
                                                st.metric("Enterprise Value", "N/A")

                                        # Create tabs for different valuation methods
                                        method_tabs = st.tabs([method for method in valuation_methods])

                                        # Display detailed results for each method
                                        for i, method in enumerate(valuation_methods):
                                            with method_tabs[i]:
                                                if method == "DCF" or method == "Project-Based DCF":
                                                    # DCF details
                                                    st.subheader("DCF Valuation Details")

                                                    dcf_params = {
                                                        "Discount Rate": f"{valuation_result.get('discount_rate', discount_rate) * 100:.1f}%",
                                                        "Initial Growth Rate": f"{valuation_result.get('initial_growth_rate', 'N/A') if isinstance(valuation_result.get('initial_growth_rate'), (int, float)) else 'N/A'}",
                                                        "Terminal Growth": f"{valuation_result.get('terminal_growth', 'N/A') if isinstance(valuation_result.get('terminal_growth'), (int, float)) else 'N/A'}"
                                                    }

                                                    st.json(dcf_params)

                                                    # Display cash flow forecast chart
                                                    if 'forecasted_cash_flows' in valuation_result or 'forecast_fcf' in valuation_result:
                                                        fcf_data = valuation_result.get('forecasted_cash_flows',
                                                                                        valuation_result.get(
                                                                                            'forecast_fcf', []))

                                                        if fcf_data and len(fcf_data) > 0:
                                                            st.subheader("Forecasted Cash Flows")

                                                            import plotly.graph_objects as go

                                                            fig = go.Figure()
                                                            fig.add_trace(
                                                                go.Bar(
                                                                    x=[f"Year {i + 1}" for i in range(len(fcf_data))],
                                                                    y=fcf_data,
                                                                    marker_color=COLORS['primary']
                                                                )
                                                            )

                                                            # Add terminal value
                                                            if 'terminal_value' in valuation_result:
                                                                # Calculate relative size for visualization
                                                                tv_display_value = max(fcf_data) * 2

                                                                fig.add_trace(
                                                                    go.Bar(
                                                                        x=["Terminal<br>Value"],
                                                                        y=[tv_display_value],
                                                                        marker_color=COLORS['secondary'],
                                                                        text=[
                                                                            f"${valuation_result['terminal_value'] / 1e9:.2f}B"],
                                                                        textposition="outside"
                                                                    )
                                                                )

                                                                fig.add_annotation(
                                                                    x=len(fcf_data),
                                                                    y=tv_display_value / 2,
                                                                    text=f"Full Terminal Value: ${valuation_result['terminal_value'] / 1e9:.2f}B<br>(Not to scale)",
                                                                    showarrow=True,
                                                                    arrowhead=1
                                                                )

                                                            fig.update_layout(
                                                                title="Forecasted Free Cash Flows",
                                                                xaxis_title="Forecast Period",
                                                                yaxis_title="Free Cash Flow ($)",
                                                                plot_bgcolor=VIZ_SETTINGS['background'],
                                                                paper_bgcolor=VIZ_SETTINGS['background'],
                                                                font=dict(color=VIZ_SETTINGS['text_color']),
                                                                height=400
                                                            )

                                                            st.plotly_chart(fig, use_container_width=True)

                                                elif method == "NAV (Reserve-Based)":
                                                    # NAV details for upstream
                                                    st.subheader("Reserve-Based NAV Details")

                                                    if 'reserves_value' in valuation_result:
                                                        reserves = valuation_result['reserves_value']

                                                        col1, col2 = st.columns(2)

                                                        with col1:
                                                            st.metric("Oil Value",
                                                                      f"${reserves.get('oil_value', 0) / 1e9:.2f}B")
                                                            st.metric("Oil Reserves",
                                                                      f"{reserves.get('oil_reserves_mmbbl', 0):.1f} MMbbl")
                                                            st.metric("Oil Price Used",
                                                                      f"${reserves.get('oil_price', 0):.2f}/bbl")

                                                        with col2:
                                                            st.metric("Gas Value",
                                                                      f"${reserves.get('gas_value', 0) / 1e9:.2f}B")
                                                            st.metric("Gas Reserves",
                                                                      f"{reserves.get('gas_reserves_bcf', 0):.1f} Bcf")
                                                            st.metric("Gas Price Used",
                                                                      f"${reserves.get('gas_price', 0):.2f}/Mcf")

                                                        # Display NAV bridge chart
                                                        st.subheader("NAV Value Bridge")

                                                        import plotly.graph_objects as go

                                                        # Create NAV bridge chart
                                                        nav_components = [
                                                            ('Oil Reserves', reserves.get('oil_value', 0) / 1e9),
                                                            ('Gas Reserves', reserves.get('gas_value', 0) / 1e9),
                                                            ('Undeveloped Acreage',
                                                             valuation_result.get('undeveloped_acreage_value',
                                                                                  0) / 1e9),
                                                            ('Other Assets',
                                                             valuation_result.get('other_assets_value', 0) / 1e9),
                                                            ('Total Liabilities',
                                                             -valuation_result.get('total_liabilities', 0) / 1e9),
                                                            ('Net Asset Value',
                                                             valuation_result.get('net_asset_value', 0) / 1e9)
                                                        ]

                                                        # Filter out zero values except NAV
                                                        nav_components = [c for c in nav_components if
                                                                          c[1] != 0 or c[0] == 'Net Asset Value']

                                                        fig = go.Figure()

                                                        fig.add_trace(
                                                            go.Waterfall(
                                                                name="NAV Bridge",
                                                                orientation="v",
                                                                measure=["relative"] * (len(nav_components) - 1) + [
                                                                    "total"],
                                                                x=[c[0] for c in nav_components],
                                                                y=[c[1] for c in nav_components],
                                                                connector={"line": {"color": "rgb(63, 63, 63)"}},
                                                                increasing={"marker": {"color": COLORS['success']}},
                                                                decreasing={"marker": {"color": COLORS['danger']}},
                                                                totals={"marker": {"color": COLORS['secondary']}}
                                                            )
                                                        )

                                                        fig.update_layout(
                                                            title="Net Asset Value Components ($ Billions)",
                                                            plot_bgcolor=VIZ_SETTINGS['background'],
                                                            paper_bgcolor=VIZ_SETTINGS['background'],
                                                            font=dict(color=VIZ_SETTINGS['text_color']),
                                                            height=400
                                                        )

                                                        st.plotly_chart(fig, use_container_width=True)

                                                elif method == "Sum-of-the-Parts":
                                                    # SOTP details for integrated
                                                    st.subheader("Sum-of-the-Parts Valuation")

                                                    if 'segment_values' in valuation_result:
                                                        segment_values = valuation_result['segment_values']

                                                        # Display segment values as a table
                                                        segment_data = []
                                                        for segment, data in segment_values.items():
                                                            segment_data.append({
                                                                "Segment": segment,
                                                                "EBITDA ($M)": data.get('EBITDA', 0) / 1e6,
                                                                "EV/EBITDA": data.get('EV_EBITDA_Multiple', 0),
                                                                "Value ($B)": data.get('Enterprise_Value', 0) / 1e9
                                                            })

                                                        segment_df = pd.DataFrame(segment_data)
                                                        st.dataframe(segment_df, use_container_width=True)

                                                        # Display SOTP pie chart
                                                        st.subheader("Segment Value Breakdown")

                                                        import plotly.express as px

                                                        # Create pie chart of segment values
                                                        fig = px.pie(
                                                            segment_df,
                                                            values="Value ($B)",
                                                            names="Segment",
                                                            title="Enterprise Value by Segment",
                                                            color_discrete_sequence=list(COLORS['sectors'].values())
                                                        )

                                                        fig.update_layout(
                                                            plot_bgcolor=VIZ_SETTINGS['background'],
                                                            paper_bgcolor=VIZ_SETTINGS['background'],
                                                            font=dict(color=VIZ_SETTINGS['text_color']),
                                                            legend=dict(font=dict(color=VIZ_SETTINGS['text_color'])),
                                                            height=400
                                                        )

                                                        st.plotly_chart(fig, use_container_width=True)

                                                        # Corporate overhead and debt adjustments
                                                        col1, col2, col3 = st.columns(3)

                                                        with col1:
                                                            st.metric(
                                                                "Sum of Segments",
                                                                f"${sum(data.get('Enterprise_Value', 0) for data in segment_values.values()) / 1e9:.2f}B"
                                                            )

                                                        with col2:
                                                            st.metric(
                                                                "Corporate Overhead",
                                                                f"${valuation_result.get('corporate_overhead', 0) / 1e9:.2f}B"
                                                            )

                                                        with col3:
                                                            st.metric(
                                                                "Net Debt",
                                                                f"${valuation_result.get('net_debt', 0) / 1e9:.2f}B"
                                                            )

                                                elif method == "Dividend Discount Model":
                                                    # DDM details
                                                    st.subheader("Dividend Discount Model Details")

                                                    if valuation_result.get(
                                                            'method') == 'utility_ddm' or valuation_result.get(
                                                            'method') == 'midstream_ddm':
                                                        # Get method-specific result
                                                        if valuation_result.get('method') == 'utility_ddm':
                                                            ddm_result = valuation_result
                                                        elif valuation_result.get('method') == 'midstream_ddm':
                                                            ddm_result = valuation_result
                                                        else:
                                                            # Try to get component from blended result
                                                            ddm_result = valuation_result.get('ddm_valuation', {})

                                                        # Display DDM parameters
                                                        col1, col2, col3 = st.columns(3)

                                                        with col1:
                                                            st.metric(
                                                                "Current Dividend",
                                                                f"${ddm_result.get('current_dividend', 0):.2f}"
                                                            )

                                                        with col2:
                                                            st.metric(
                                                                "Dividend Growth",
                                                                f"{ddm_result.get('dividend_growth_rate', 0) * 100:.1f}%"
                                                            )

                                                        with col3:
                                                            st.metric(
                                                                "Discount Rate",
                                                                f"{ddm_result.get('discount_rate', 0) * 100:.1f}%"
                                                            )

                                                        # Display dividend forecast chart if available
                                                        if 'future_distributions' in ddm_result:
                                                            distributions = ddm_result['future_distributions']

                                                            st.subheader("Projected Dividends/Distributions")

                                                            import plotly.graph_objects as go

                                                            fig = go.Figure()
                                                            fig.add_trace(
                                                                go.Scatter(
                                                                    x=[f"Year {i + 1}" for i in
                                                                       range(len(distributions))],
                                                                    y=distributions,
                                                                    mode='lines+markers',
                                                                    marker=dict(color=COLORS['primary'])
                                                                )
                                                            )

                                                            fig.update_layout(
                                                                title="Projected Dividends/Distributions",
                                                                xaxis_title="Year",
                                                                yaxis_title="Dividend/Distribution ($)",
                                                                plot_bgcolor=VIZ_SETTINGS['background'],
                                                                paper_bgcolor=VIZ_SETTINGS['background'],
                                                                font=dict(color=VIZ_SETTINGS['text_color']),
                                                                height=400
                                                            )

                                                            st.plotly_chart(fig, use_container_width=True)

                                                elif method == "Regulated Asset Base":
                                                    # RAB details for utilities
                                                    st.subheader("Regulated Asset Base Valuation")

                                                    # Get RAB results
                                                    if valuation_result.get('method') == 'regulated_asset_base':
                                                        rab_result = valuation_result
                                                    else:
                                                        # Try to get component from blended result
                                                        rab_result = valuation_result.get('rab_valuation', {})

                                                    if rab_result:
                                                        col1, col2 = st.columns(2)

                                                        with col1:
                                                            st.metric(
                                                                "Regulated Asset Base",
                                                                f"${rab_result.get('rab', 0) / 1e9:.2f}B"
                                                            )
                                                            st.metric(
                                                                "Equity in RAB",
                                                                f"${rab_result.get('equity_value_in_rab', 0) / 1e9:.2f}B"
                                                            )

                                                        with col2:
                                                            st.metric(
                                                                "RAB Multiple",
                                                                f"{rab_result.get('rab_multiple', 0):.2f}x"
                                                            )
                                                            st.metric(
                                                                "Adjusted RAB Value",
                                                                f"${rab_result.get('adjusted_rab_value', 0) / 1e9:.2f}B"
                                                            )

                                                        # Display RAB adjustment factors
                                                        st.subheader("RAB Valuation Adjustments")

                                                        adjustments = {
                                                            "Base RAB Multiple": 1.0,
                                                            "ROE Adjustment": rab_result.get('roe_adjustment', 0),
                                                            "Growth Adjustment": rab_result.get('growth_adjustment', 0),
                                                            "Final Multiple": rab_result.get('rab_multiple', 0)
                                                        }

                                                        import plotly.graph_objects as go

                                                        fig = go.Figure()

                                                        fig.add_trace(
                                                            go.Waterfall(
                                                                name="RAB Multiple",
                                                                orientation="v",
                                                                measure=["absolute", "relative", "relative", "total"],
                                                                x=list(adjustments.keys()),
                                                                y=list(adjustments.values()),
                                                                connector={"line": {"color": "rgb(63, 63, 63)"}},
                                                                increasing={"marker": {"color": COLORS['success']}},
                                                                decreasing={"marker": {"color": COLORS['danger']}},
                                                                totals={"marker": {"color": COLORS['secondary']}}
                                                            )
                                                        )

                                                        fig.update_layout(
                                                            title="RAB Multiple Adjustments",
                                                            plot_bgcolor=VIZ_SETTINGS['background'],
                                                            paper_bgcolor=VIZ_SETTINGS['background'],
                                                            font=dict(color=VIZ_SETTINGS['text_color']),
                                                            height=400
                                                        )

                                                        st.plotly_chart(fig, use_container_width=True)

                                                elif method == "Multiples":
                                                    # Multiples details
                                                    st.subheader("Multiples Valuation Details")

                                                    # Get multiples results based on subsector
                                                    if subsector == "oil_gas":
                                                        if model == "upstream":
                                                            multiples_result = valuation_result.get(
                                                                'multiples_valuation', {})
                                                        elif model == "integrated":
                                                            multiples_result = valuation_result.get(
                                                                'multiples_valuation', {})
                                                        elif model == "midstream":
                                                            multiples_result = valuation_result.get(
                                                                'multiples_valuation', {})
                                                    elif subsector == "renewables":
                                                        multiples_result = valuation_result.get('multiples_valuation',
                                                                                                {})
                                                    else:
                                                        # Try to get component from blended result
                                                        multiples_result = {}

                                                    if multiples_result and 'valuations' in multiples_result:
                                                        valuations = multiples_result['valuations']

                                                        # Display multiples as a table
                                                        multiples_data = []
                                                        for metric, data in valuations.items():
                                                            if 'enterprise_value' in data:
                                                                multiples_data.append({
                                                                    "Metric": metric.replace('_', ' ').replace('EV ',
                                                                                                               'EV/'),
                                                                    "Multiple": data.get('multiple', 0),
                                                                    "Value ($B)": data.get('enterprise_value', 0) / 1e9
                                                                })

                                                        if multiples_data:
                                                            multiples_df = pd.DataFrame(multiples_data)
                                                            st.dataframe(multiples_df, use_container_width=True)

                                                            # Display multiples bar chart
                                                            st.subheader("Value by Multiple")

                                                            import plotly.express as px

                                                            fig = px.bar(
                                                                multiples_df,
                                                                x="Metric",
                                                                y="Value ($B)",
                                                                title="Enterprise Value by Valuation Metric",
                                                                color="Metric",
                                                                text="Value ($B)",
                                                                color_discrete_sequence=list(COLORS['sectors'].values())
                                                            )

                                                            fig.update_layout(
                                                                xaxis_title="Valuation Metric",
                                                                yaxis_title="Enterprise Value ($B)",
                                                                plot_bgcolor=VIZ_SETTINGS['background'],
                                                                paper_bgcolor=VIZ_SETTINGS['background'],
                                                                font=dict(color=VIZ_SETTINGS['text_color']),
                                                                height=400
                                                            )

                                                            st.plotly_chart(fig, use_container_width=True)

                                                        # Display multiples adjustment factors if available
                                                        if 'base_multiples' in multiples_result and 'adjusted_multiples' in multiples_result:
                                                            st.subheader("Multiple Adjustments")

                                                            col1, col2 = st.columns(2)

                                                            with col1:
                                                                st.subheader("Base Multiples")
                                                                base_mult_df = pd.DataFrame({
                                                                    "Multiple": list(
                                                                        multiples_result['base_multiples'].keys()),
                                                                    "Value": list(
                                                                        multiples_result['base_multiples'].values())
                                                                })
                                                                st.dataframe(base_mult_df)

                                                            with col2:
                                                                st.subheader("Adjusted Multiples")
                                                                adj_mult_df = pd.DataFrame({
                                                                    "Multiple": list(
                                                                        multiples_result['adjusted_multiples'].keys()),
                                                                    "Value": list(
                                                                        multiples_result['adjusted_multiples'].values())
                                                                })
                                                                st.dataframe(adj_mult_df)

                                                elif method == "Real Options":
                                                    # Real options details for renewables
                                                    st.subheader("Real Options Valuation")

                                                    # Get options results
                                                    if valuation_result.get('method') == 'renewable_options':
                                                        options_result = valuation_result
                                                    else:
                                                        # Try to get component from blended result
                                                        options_result = valuation_result.get('options_valuation', {})

                                                    if options_result and 'components' in options_result:
                                                        components = options_result['components']

                                                        col1, col2, col3 = st.columns(3)

                                                        with col1:
                                                            st.metric(
                                                                "Operating Assets",
                                                                f"${components.get('operating_assets_value', 0) / 1e9:.2f}B"
                                                            )

                                                        with col2:
                                                            st.metric(
                                                                "Pipeline Value",
                                                                f"${components.get('pipeline_value', 0) / 1e9:.2f}B"
                                                            )

                                                        with col3:
                                                            st.metric(
                                                                "Growth Options",
                                                                f"${components.get('growth_options_value', 0) / 1e9:.2f}B"
                                                            )

                                                        # Display pipeline breakdown if available
                                                        if 'pipeline_details' in options_result:
                                                            pipeline = options_result['pipeline_details']

                                                            st.subheader("Pipeline Value Breakdown")

                                                            pipeline_data = []
                                                            for stage, data in pipeline.items():
                                                                pipeline_data.append({
                                                                    "Stage": stage.replace('_', ' ').title(),
                                                                    "Capacity (MW)": data.get('mw', 0),
                                                                    "Success Probability": f"{data.get('success_prob', 0) * 100:.0f}%",
                                                                    "Value per MW ($K)": data.get('value_per_mw',
                                                                                                  0) / 1e3,
                                                                    "Expected Value ($M)": data.get('expected_value',
                                                                                                    0) / 1e6
                                                                })

                                                            pipeline_df = pd.DataFrame(pipeline_data)
                                                            st.dataframe(pipeline_df, use_container_width=True)

                                                            # Create pie chart of pipeline values
                                                            import plotly.express as px

                                                            fig = px.pie(
                                                                pipeline_df,
                                                                values="Expected Value ($M)",
                                                                names="Stage",
                                                                title="Pipeline Value by Development Stage",
                                                                color_discrete_sequence=list(COLORS['sectors'].values())
                                                            )

                                                            fig.update_layout(
                                                                plot_bgcolor=VIZ_SETTINGS['background'],
                                                                paper_bgcolor=VIZ_SETTINGS['background'],
                                                                font=dict(color=VIZ_SETTINGS['text_color']),
                                                                legend=dict(
                                                                    font=dict(color=VIZ_SETTINGS['text_color'])),
                                                                height=400
                                                            )

                                                            st.plotly_chart(fig, use_container_width=True)

                                        # Add investment recommendation based on valuation
                                        st.subheader("Investment Recommendation")

                                        # Calculate metrics for recommendation
                                        if value_per_share and current_price:
                                            upside_pct = ((value_per_share / current_price) - 1) * 100

                                            # Set thresholds for recommendations
                                            if upside_pct > 25:
                                                recommendation = "Strong Buy"
                                                recommendation_color = "#74f174"  # Green
                                                rationale = "Significant undervaluation suggests strong upside potential."
                                            elif upside_pct > 10:
                                                recommendation = "Buy"
                                                recommendation_color = "#a5d6a7"  # Light green
                                                rationale = "Material undervaluation suggests attractive entry point."
                                            elif upside_pct > -10:
                                                recommendation = "Hold"
                                                recommendation_color = "#fff59d"  # Yellow
                                                rationale = "Current valuation appears reasonable relative to intrinsic value."
                                            elif upside_pct > -25:
                                                recommendation = "Reduce"
                                                recommendation_color = "#ffab91"  # Light red
                                                rationale = "Overvaluation suggests reducing position size."
                                            else:
                                                recommendation = "Sell"
                                                recommendation_color = "#faa1a4"  # Red
                                                rationale = "Significant overvaluation indicates potential for material downside."

                                            # Display recommendation
                                            st.markdown(
                                                f"<h3 style='text-align: center; color: {recommendation_color};'>{recommendation}</h3>",
                                                unsafe_allow_html=True)

                                            # Show rationale
                                            st.markdown(f"<p style='text-align: center;'>{rationale}</p>",
                                                        unsafe_allow_html=True)

                                            # Add sector-specific considerations
                                            if subsector == "oil_gas":
                                                st.markdown("""
                                                **Sector-Specific Considerations:**
                                                - Commodity price volatility creates both risks and opportunities
                                                - Capital intensity requires focus on capital allocation efficiency
                                                - Energy transition presents long-term structural challenges
                                                """)
                                            elif subsector == "utilities":
                                                st.markdown("""
                                                **Sector-Specific Considerations:**
                                                - Regulated returns provide stability but limit upside
                                                - Rate base growth is key driver of long-term value
                                                - Energy transition creates both risks and opportunities
                                                """)
                                            elif subsector == "renewables":
                                                st.markdown("""
                                                **Sector-Specific Considerations:**
                                                - Policy support provides tailwinds but creates regulatory risk
                                                - Project development execution is critical success factor
                                                - Technology innovation can drive both opportunities and obsolescence risk
                                                """)
                                        else:
                                            st.info("Insufficient data to generate investment recommendation.")

                                    except Exception as e:
                                        st.error(f"Error in energy sector valuation: {str(e)}")
                                        st.error(traceback.format_exc())

                        except Exception as e:
                            st.error(f"Error in energy sector valuation tab: {str(e)}")
                            st.error(traceback.format_exc())

                    elif sector == "Financials":
                    # Financial sector specific valuation
                    # Код для финансового сектора будет добавлен здесь

                    elif sector == "Information Technology" or sector == "Technology":
                    # Technology sector specific valuation
                    # Код для технологического сектора будет добавлен здесь

                    elif sector == "Healthcare":
                    # Healthcare sector specific valuation
                    # Код для сектора здравоохранения будет добавлен здесь

                    elif sector == "Real Estate":
                    # Real Estate sector specific valuation
                    # Код для сектора недвижимости будет добавлен здесь

                    elif sector == "Consumer Discretionary":
                    # Consumer Discretionary sector specific valuation
                    # Код для сектора потребительских товаров выборочного спроса будет добавлен здесь

                    elif sector == "Consumer Staples":
                    # Consumer Staples sector specific valuation
                    # Код для сектора потребительских товаров первой необходимости будет добавлен здесь

                    elif sector == "Industrials":
                    # Industrials sector specific valuation
                    # Код для промышленного сектора будет добавлен здесь

                    elif sector == "Materials":
                    # Materials sector specific valuation
                    # Код для сектора материалов будет добавлен здесь

                    elif sector == "Communication Services":
                    # Communication Services sector specific valuation
                    # Код для сектора коммуникационных услуг будет добавлен здесь

                    elif sector == "Utilities":
                    # Utilities sector specific valuation
                    # Код для сектора коммунальных услуг будет добавлен здесь

                    else:
                    # Generic valuation for unknown sectors
                    # Существующий код валюации с вкладками


                    # Create tabs for different valuation methods
                    valuation_tabs = st.tabs([
                        "Overview",
                        "DCF Valuation",
                        "Relative Valuation",
                        "Asset-Based Valuation",
                        "Sensitivity Analysis"
                    ])


                    # Overview tab
                    with valuation_tabs[0]:
                        st.write(
                            "This section provides various valuation methods to estimate the intrinsic value of the company.")

                        # Current market valuation
                        st.subheader("Current Market Valuation")
                        current_price = price_data['Close'].iloc[-1] if not price_data.empty else None
                        market_cap = company_info.get('market_cap')

                        # Display current market metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
                        with col2:
                            st.metric("Market Cap", f"${market_cap / 1e9:.2f}B" if market_cap else "N/A")
                        with col3:
                            shares_outstanding = market_cap / current_price if market_cap and current_price else None
                            st.metric("Shares Outstanding",
                                      f"{shares_outstanding / 1e6:.2f}M" if shares_outstanding else "N/A")

                        # Show valuation ratios
                        valuation_ratios = ratios.get('valuation', {})
                        sector_valuation = ratio_analyzer.get_sector_benchmarks(sector).get('valuation',
                                                                                            {}) if sector != 'Unknown' else {}

                        if valuation_ratios:
                            st.subheader("Valuation Metrics")

                            # Display valuation metrics in columns
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                pe_ratio = valuation_ratios.get('pe_ratio')
                                sector_pe = sector_valuation.get('pe_ratio')
                                st.metric("P/E Ratio",
                                          f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                                          f"{((pe_ratio / sector_pe) - 1) * 100:.1f}% vs Sector" if pe_ratio is not None and sector_pe is not None and sector_pe != 0 else None,
                                          delta_color="inverse")

                            with col2:
                                ps_ratio = valuation_ratios.get('ps_ratio')
                                sector_ps = sector_valuation.get('ps_ratio')
                                st.metric("P/S Ratio",
                                          f"{ps_ratio:.2f}" if ps_ratio else "N/A",
                                          f"{((ps_ratio / sector_ps) - 1) * 100:.1f}% vs Sector" if ps_ratio is not None and sector_ps is not None and sector_ps != 0 else None,
                                          delta_color="inverse")

                            with col3:
                                pb_ratio = valuation_ratios.get('pb_ratio')
                                sector_pb = sector_valuation.get('pb_ratio')
                                st.metric("P/B Ratio",
                                          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
                                          f"{((pb_ratio / sector_pb) - 1) * 100:.1f}% vs Sector" if pb_ratio is not None and sector_pb is not None and sector_pb != 0 else None,
                                          delta_color="inverse")

                            with col4:
                                ev_ebitda = valuation_ratios.get('ev_ebitda')
                                sector_ev_ebitda = sector_valuation.get('ev_ebitda')
                                st.metric("EV/EBITDA",
                                          f"{ev_ebitda:.2f}" if ev_ebitda else "N/A",
                                          f"{((ev_ebitda / sector_ev_ebitda) - 1) * 100:.1f}% vs Sector" if ev_ebitda is not None and sector_ev_ebitda is not None and sector_ev_ebitda != 0 else None,
                                          delta_color="inverse")

                            # Create valuation data for heatmap
                            valuation_data = {
                                'company': valuation_ratios,
                                'sector_avg': sector_valuation
                            }

                            # Display valuation heatmap
                            st.subheader("Valuation Multiples vs. Sector")
                            try:
                                fig = visualizer.plot_valuation_heatmap(
                                    valuation_data,
                                    height=250
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating valuation heatmap: {str(e)}")

                        # Summary of valuation methods
                        st.subheader("Valuation Methods")
                        st.write("""
                        ### Valuation approaches used:

                        1. **Discounted Cash Flow (DCF)** - Estimates intrinsic value based on projected future cash flows.
                        2. **Relative Valuation** - Compares the company to peers using various multiples.
                        3. **Asset-Based Valuation** - Determines value based on the company's assets and liabilities.
                        4. **Sensitivity Analysis** - Tests how changes in key assumptions affect valuation.

                        Each approach has strengths and limitations. A comprehensive valuation considers multiple methods.
                        """)

                    # DCF Valuation tab
                    with valuation_tabs[1]:
                        st.subheader("Discounted Cash Flow (DCF) Analysis")

                        # Prepare financial data
                        financial_data = {
                            'income_statement': income_statement,
                            'balance_sheet': balance_sheet,
                            'cash_flow': cash_flow,
                            'market_data': {
                                'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                                'market_cap': company_info.get('market_cap'),
                                'shares_outstanding': company_info.get('market_cap') / price_data['Close'].iloc[-1]
                                if company_info.get('market_cap') and not price_data.empty else None,
                                'beta': company_info.get('beta')
                            }
                        }

                        # DCF parameters input
                        st.subheader("DCF Parameters")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            forecast_years = st.slider("Forecast Years", min_value=5, max_value=10, value=5)

                        with col2:
                            growth_rate = st.slider("Growth Rate (%)", min_value=1, max_value=30, value=10) / 100

                        with col3:
                            discount_rate = st.slider("Discount Rate (%)", min_value=5, max_value=20, value=10) / 100

                        # Terminal value parameters
                        col1, col2 = st.columns(2)

                        with col1:
                            terminal_growth = st.slider("Terminal Growth Rate (%)", min_value=1, max_value=5,
                                                        value=2) / 100

                        with col2:
                            margin_of_safety = st.slider("Margin of Safety (%)", min_value=0, max_value=50,
                                                         value=25) / 100
                            # Add sector-specific parameters if sector is known
                            if sector != 'Unknown':
                                st.subheader(f"Sector-Specific Parameters ({sector})")

                                # Financial sector specific parameters
                                if sector == "Financials":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        bank_type = st.selectbox(
                                            "Bank/Financial Type",
                                            ["Commercial Bank", "Investment Bank", "Insurance", "Asset Management",
                                             "Diversified"],
                                            index=0
                                        )
                                    with col2:
                                        use_dividend_model = st.checkbox("Use Dividend Discount Model", value=True,
                                                                         help="Financial companies are often better valued using dividend models due to their capital structure")

                                    # Additional industry-specific metrics
                                    if bank_type in ["Commercial Bank", "Investment Bank"]:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            nim = st.slider("Net Interest Margin (%)",
                                                            min_value=1.0, max_value=5.0, value=2.5, step=0.1)
                                        with col2:
                                            capital_ratio = st.slider("Target Capital Ratio (%)",
                                                                      min_value=8.0, max_value=20.0, value=12.0,
                                                                      step=0.5)
                                    elif bank_type == "Insurance":
                                        combined_ratio = st.slider("Combined Ratio (%)",
                                                                   min_value=80.0, max_value=120.0, value=95.0,
                                                                   step=1.0,
                                                                   help="Combined ratio under 100% indicates underwriting profitability")

                                # Technology sector specific parameters
                                elif sector == "Information Technology" or sector == "Technology":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        tech_type = st.selectbox(
                                            "Technology Type",
                                            ["Software", "Hardware", "Semiconductors", "IT Services", "Internet"],
                                            index=0
                                        )
                                    with col2:
                                        rd_intensity = st.slider("R&D Intensity (%)",
                                                                 min_value=5.0, max_value=30.0, value=15.0, step=1.0,
                                                                 help="R&D spending as percentage of revenue")

                                    # Additional industry-specific metrics
                                    if tech_type == "Software" or tech_type == "Internet":
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            saas_metrics = st.checkbox("Consider SaaS Metrics", value=True)
                                        with col2:
                                            if saas_metrics:
                                                ltv_cac = st.slider("LTV/CAC Ratio",
                                                                    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                                                                    help="Lifetime Value to Customer Acquisition Cost ratio")

                                # Energy sector specific parameters
                                elif sector == "Energy":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        energy_type = st.selectbox(
                                            "Energy Type",
                                            ["Integrated Oil & Gas", "E&P", "Refining & Marketing", "Renewable Energy"],
                                            index=0
                                        )
                                    with col2:
                                        commodity_scenario = st.selectbox(
                                            "Commodity Price Scenario",
                                            ["Base Case", "Bull Case", "Bear Case"],
                                            index=0
                                        )

                                    if energy_type in ["Integrated Oil & Gas", "E&P"]:
                                        reserve_life = st.slider("Reserve Life (years)",
                                                                 min_value=5, max_value=30, value=15, step=1,
                                                                 help="Years of production at current rates")

                                # Healthcare sector specific parameters
                                elif sector == "Healthcare":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        healthcare_type = st.selectbox(
                                            "Healthcare Type",
                                            ["Pharmaceuticals", "Biotechnology", "Medical Devices",
                                             "Healthcare Services"],
                                            index=0
                                        )
                                    with col2:
                                        if healthcare_type in ["Pharmaceuticals", "Biotechnology"]:
                                            pipeline_value = st.slider("Pipeline Value Adjustment (%)",
                                                                       min_value=-20, max_value=100, value=30, step=5,
                                                                       help="Adjustment for R&D pipeline value")

                                # Real Estate sector specific parameters
                                elif sector == "Real Estate":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        property_type = st.selectbox(
                                            "Property Type",
                                            ["Residential", "Commercial", "Industrial", "Retail", "Office", "Mixed",
                                             "Specialized"],
                                            index=5
                                        )
                                    with col2:
                                        reit_structure = st.checkbox("REIT Structure", value=True,
                                                                     help="Real Estate Investment Trust tax structure")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        cap_rate = st.slider("Capitalization Rate (%)",
                                                             min_value=3.0, max_value=12.0, value=6.0, step=0.1,
                                                             help="Net operating income divided by property value")
                                    with col2:
                                        occupancy_rate = st.slider("Occupancy Rate (%)",
                                                                   min_value=70.0, max_value=100.0, value=95.0,
                                                                   step=1.0)

                                # Consumer Discretionary sector specific parameters
                                elif sector == "Consumer Discretionary":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        retail_type = st.selectbox(
                                            "Retail Type",
                                            ["Specialty Retail", "Department Stores", "Online Retail", "Apparel",
                                             "Automotive", "Restaurants"],
                                            index=0
                                        )
                                    with col2:
                                        omnichannel = st.checkbox("Omnichannel Strategy", value=True)

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        sss_growth = st.slider("Same-Store Sales Growth (%)",
                                                               min_value=-10.0, max_value=15.0, value=3.0, step=0.5,
                                                               help="Year-over-year sales growth for existing stores")
                                    with col2:
                                        gm_trend = st.slider("Gross Margin Trend (bps)",
                                                             min_value=-200, max_value=200, value=0, step=10,
                                                             help="Annual change in gross margin in basis points")

                                # Consumer Staples sector specific parameters
                                elif sector == "Consumer Staples":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        staples_type = st.selectbox(
                                            "Consumer Staples Type",
                                            ["Food & Beverage", "Household Products", "Personal Care", "Food Retail",
                                             "Staples Distribution"],
                                            index=0
                                        )
                                    with col2:
                                        brand_strength = st.slider("Brand Strength",
                                                                   min_value=1, max_value=5, value=3, step=1,
                                                                   help="1=Weak branding, 5=Strong brand portfolio with pricing power")

                                    pricing_power = st.slider("Pricing Power (%)",
                                                              min_value=0.0, max_value=10.0, value=2.0, step=0.5,
                                                              help="Ability to pass through input cost inflation")

                                # Industrials sector specific parameters
                                elif sector == "Industrials":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        industry_type = st.selectbox(
                                            "Industrial Type",
                                            ["Capital Goods", "Commercial Services", "Transportation",
                                             "Aerospace & Defense", "Machinery"],
                                            index=0
                                        )
                                    with col2:
                                        if industry_type == "Capital Goods" or industry_type == "Machinery":
                                            capex_cycle = st.selectbox(
                                                "Capex Cycle Position",
                                                ["Early Cycle", "Mid Cycle", "Late Cycle", "Downturn"],
                                                index=1
                                            )
                                        elif industry_type == "Transportation":
                                            transport_mode = st.selectbox(
                                                "Transportation Mode",
                                                ["Air", "Rail", "Marine", "Trucking", "Logistics"],
                                                index=0
                                            )

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        operating_margin = st.slider("Normalized Operating Margin (%)",
                                                                     min_value=5.0, max_value=25.0, value=15.0,
                                                                     step=0.5)
                                    with col2:
                                        asset_turnover = st.slider("Asset Turnover Ratio",
                                                                   min_value=0.5, max_value=2.5, value=1.2, step=0.1,
                                                                   help="Revenue divided by assets - efficiency metric")

                                # Materials sector specific parameters
                                elif sector == "Materials":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        material_type = st.selectbox(
                                            "Materials Type",
                                            ["Chemicals", "Metals & Mining", "Construction Materials",
                                             "Paper & Forest Products", "Containers & Packaging"],
                                            index=0
                                        )
                                    with col2:
                                        if material_type == "Metals & Mining":
                                            commodity_type = st.selectbox(
                                                "Commodity Type",
                                                ["Precious Metals", "Base Metals", "Iron Ore", "Coal", "Diversified"],
                                                index=4
                                            )

                                    commodity_price_outlook = st.slider("Commodity Price Outlook",
                                                                        min_value=-20, max_value=20, value=0, step=5,
                                                                        help="Expected change in commodity prices (%)")

                                # Communication Services sector specific parameters
                                elif sector == "Communication Services":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        comm_type = st.selectbox(
                                            "Communication Type",
                                            ["Telecommunications", "Media", "Entertainment", "Interactive Media"],
                                            index=0
                                        )
                                    with col2:
                                        if comm_type == "Telecommunications":
                                            telecom_type = st.selectbox(
                                                "Telecom Type",
                                                ["Wireless", "Wireline", "Integrated", "Infrastructure"],
                                                index=2
                                            )
                                        elif comm_type in ["Media", "Entertainment", "Interactive Media"]:
                                            content_investment = st.slider("Content Investment ($B)",
                                                                           min_value=0.1, max_value=20.0, value=2.0,
                                                                           step=0.1,
                                                                           help="Annual spending on content creation")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if comm_type == "Telecommunications":
                                            arpu = st.slider("ARPU ($)",
                                                             min_value=10, max_value=100, value=45, step=5,
                                                             help="Average Revenue Per User")
                                            churn = st.slider("Annual Churn Rate (%)",
                                                              min_value=0.5, max_value=5.0, value=1.2, step=0.1,
                                                              help="Percentage of customers who leave annually")
                                        else:
                                            dau_mau = st.slider("DAU/MAU Ratio (%)",
                                                                min_value=10, max_value=70, value=30, step=5,
                                                                help="Daily Active Users / Monthly Active Users")

                                # Utilities sector specific parameters
                                elif sector == "Utilities":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        utility_type = st.selectbox(
                                            "Utility Type",
                                            ["Electric", "Gas", "Water", "Multi-Utility", "Independent Power Producer"],
                                            index=0
                                        )
                                    with col2:
                                        regulatory_model = st.selectbox(
                                            "Regulatory Framework",
                                            ["Cost-of-Service", "Incentive-Based", "Hybrid", "Deregulated"],
                                            index=0
                                        )

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        allowed_roe = st.slider("Allowed Return on Equity (%)",
                                                                min_value=7.0, max_value=12.0, value=9.5, step=0.1,
                                                                help="Regulator-approved return rate")
                                    with col2:
                                        rate_base_growth = st.slider("Rate Base Growth (%)",
                                                                     min_value=1.0, max_value=8.0, value=4.0, step=0.5,
                                                                     help="Annual growth in regulated asset base")


                        # Run DCF valuation
                        if st.button("Calculate DCF Valuation", type="primary"):
                            with st.spinner("Calculating DCF valuation..."):
                                # Use sector-appropriate model if available
                                # Add sector-specific parameters to the kwargs
                                sector_kwargs = {}
                                if sector == "Financials":
                                    sector_kwargs["bank_type"] = bank_type
                                    if use_dividend_model:
                                        sector_kwargs["use_dividend_model"] = True
                                    if bank_type in ["Commercial Bank", "Investment Bank"]:
                                        sector_kwargs["net_interest_margin"] = nim / 100
                                        sector_kwargs["capital_ratio"] = capital_ratio / 100
                                    elif bank_type == "Insurance":
                                        sector_kwargs["combined_ratio"] = combined_ratio / 100

                                elif sector == "Information Technology" or sector == "Technology":
                                    sector_kwargs["tech_type"] = tech_type
                                    sector_kwargs["rd_intensity"] = rd_intensity / 100
                                    if tech_type in ["Software", "Internet"] and saas_metrics:
                                        sector_kwargs["ltv_cac"] = ltv_cac

                                elif sector == "Energy":
                                    sector_kwargs["energy_type"] = energy_type
                                    sector_kwargs["commodity_scenario"] = commodity_scenario
                                    if energy_type in ["Integrated Oil & Gas", "E&P"]:
                                        sector_kwargs["reserve_life"] = reserve_life

                                elif sector == "Healthcare":
                                    sector_kwargs["subsector"] = healthcare_type
                                    if healthcare_type in ["Pharmaceuticals", "Biotechnology"]:
                                        sector_kwargs["pipeline_adjustment"] = pipeline_value / 100

                                elif sector == "Real Estate":
                                    sector_kwargs["property_type"] = property_type
                                    sector_kwargs["reit_structure"] = reit_structure
                                    sector_kwargs["cap_rate"] = cap_rate / 100
                                    sector_kwargs["occupancy_rate"] = occupancy_rate / 100

                                elif sector == "Consumer Discretionary":
                                    sector_kwargs["retail_type"] = retail_type
                                    sector_kwargs["omnichannel"] = omnichannel
                                    sector_kwargs["sss_growth"] = sss_growth / 100
                                    sector_kwargs["gm_trend"] = gm_trend / 10000  # Convert bps to decimal

                                elif sector == "Consumer Staples":
                                    sector_kwargs["staples_type"] = staples_type
                                    sector_kwargs["brand_strength"] = brand_strength
                                    sector_kwargs["pricing_power"] = pricing_power / 100

                                elif sector == "Industrials":
                                    sector_kwargs["industry_type"] = industry_type
                                    sector_kwargs["operating_margin"] = operating_margin / 100
                                    sector_kwargs["asset_turnover"] = asset_turnover
                                    if industry_type == "Capital Goods" or industry_type == "Machinery":
                                        sector_kwargs["capex_cycle"] = capex_cycle
                                    elif industry_type == "Transportation":
                                        sector_kwargs["transport_mode"] = transport_mode

                                elif sector == "Materials":
                                    sector_kwargs["material_type"] = material_type
                                    sector_kwargs["commodity_price_outlook"] = commodity_price_outlook / 100
                                    if material_type == "Metals & Mining":
                                        sector_kwargs["commodity_type"] = commodity_type

                                elif sector == "Communication Services":
                                    sector_kwargs["comm_type"] = comm_type
                                    if comm_type == "Telecommunications":
                                        sector_kwargs["telecom_type"] = telecom_type
                                        sector_kwargs["arpu"] = arpu
                                        sector_kwargs["churn"] = churn / 100
                                    else:
                                        sector_kwargs["dau_mau"] = dau_mau / 100
                                        if comm_type in ["Media", "Entertainment", "Interactive Media"]:
                                            sector_kwargs[
                                                "content_investment"] = content_investment * 1e9  # Convert to dollars

                                elif sector == "Utilities":
                                    sector_kwargs["utility_type"] = utility_type
                                    sector_kwargs["regulatory_model"] = regulatory_model
                                    sector_kwargs["allowed_roe"] = allowed_roe / 100
                                    sector_kwargs["rate_base_growth"] = rate_base_growth / 100

                                # Use sector-appropriate model if available
                                if sector != 'Unknown':
                                    dcf_result = valuation_factory.get_company_valuation(ticker, sector,
                                                                                         **sector_kwargs)
                                else:
                                    # Use multi stage DCF with custom parameters
                                    # Modify financial_data to include the custom parameters
                                    financial_data['custom_params'] = {
                                        'forecast_years': forecast_years,
                                        'growth_rate': growth_rate,
                                        'discount_rate': discount_rate,
                                        'terminal_growth': terminal_growth,
                                        'margin_of_safety': margin_of_safety
                                    }
                                    dcf_result = dcf_model.multi_stage_dcf_valuation(ticker, financial_data)

                                # Display results
                                if dcf_result and 'error' not in dcf_result:
                                    # DCF Results
                                    st.subheader("DCF Valuation Results")

                                    # Value metrics
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        value_per_share = dcf_result.get('value_per_share')
                                        current_price = price_data['Close'].iloc[-1] if not price_data.empty else None

                                        if value_per_share and current_price:
                                            upside = ((value_per_share / current_price) - 1) * 100
                                            upside_text = f"{upside:.1f}% {'Upside' if upside > 0 else 'Downside'}"
                                            upside_color = "green" if upside > 0 else "red"

                                            st.metric(
                                                "Estimated Value",
                                                f"${value_per_share:.2f}",
                                                upside_text,
                                                delta_color="normal" if upside > 0 else "inverse"
                                            )
                                        else:
                                            st.metric("Estimated Value",
                                                      f"${value_per_share:.2f}" if value_per_share else "N/A")

                                    with col2:
                                        conservative_value = dcf_result.get('conservative_value')

                                        if conservative_value and current_price:
                                            cons_upside = ((conservative_value / current_price) - 1) * 100
                                            cons_text = f"{cons_upside:.1f}% {'Upside' if cons_upside > 0 else 'Downside'}"

                                            st.metric(
                                                "Conservative Value",
                                                f"${conservative_value:.2f}",
                                                cons_text,
                                                delta_color="normal" if cons_upside > 0 else "inverse"
                                            )
                                        else:
                                            st.metric("Conservative Value",
                                                      f"${conservative_value:.2f}" if conservative_value else "N/A")

                                    with col3:
                                        enterprise_value = dcf_result.get('enterprise_value')
                                        equity_value = dcf_result.get('equity_value')

                                        if enterprise_value:
                                            st.metric("Enterprise Value", f"${enterprise_value / 1e9:.2f}B")
                                        else:
                                            st.metric("Enterprise Value", "N/A")

                                    # Additional DCF metrics
                                    st.subheader("DCF Parameters & Metrics")
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        st.metric("Discount Rate",
                                                  f"{dcf_result.get('discount_rate') * 100:.2f}%" if 'discount_rate' in dcf_result else "N/A")

                                    with col2:
                                        if 'initial_growth_rate' in dcf_result:
                                            st.metric("Initial Growth Rate",
                                                      f"{dcf_result.get('initial_growth_rate') * 100:.2f}%")
                                        elif 'growth_rate' in dcf_result:
                                            st.metric("Growth Rate", f"{dcf_result.get('growth_rate') * 100:.2f}%")

                                    with col3:
                                        st.metric("Terminal Growth",
                                                  f"{dcf_result.get('terminal_growth') * 100:.2f}%" if 'terminal_growth' in dcf_result else "N/A")

                                    with col4:
                                        st.metric("Net Debt",
                                                  f"${dcf_result.get('net_debt') / 1e9:.2f}B" if 'net_debt' in dcf_result else "N/A")

                                    # Cash flow forecast chart
                                    st.subheader("Forecasted Cash Flows")

                                    forecast_fcf = dcf_result.get('forecast_fcf', [])
                                    if forecast_fcf:
                                        # Convert to DataFrame for visualization
                                        fcf_df = pd.DataFrame({
                                            'Year': [f"Year {i + 1}" for i in range(len(forecast_fcf))],
                                            'FCF': forecast_fcf
                                        })

                                        # Plot cash flows
                                        fig = go.Figure()
                                        fig.add_trace(
                                            go.Bar(
                                                x=fcf_df['Year'],
                                                y=fcf_df['FCF'],
                                                marker_color=COLORS['primary']
                                            )
                                        )

                                        # Add terminal value
                                        terminal_value = dcf_result.get('terminal_value')
                                        if terminal_value:
                                            # Calculate TV proportional to FCF scale
                                            # We'll show just a portion to keep the scale reasonable
                                            tv_display_value = max(forecast_fcf) * 2

                                            fig.add_trace(
                                                go.Bar(
                                                    x=["Terminal<br>Value"],
                                                    y=[tv_display_value],
                                                    marker_color=COLORS['secondary'],
                                                    text=[f"${terminal_value / 1e9:.2f}B"],
                                                    textposition="outside"
                                                )
                                            )

                                            # Add annotation explaining terminal value
                                            fig.add_annotation(
                                                x=len(forecast_fcf),
                                                y=tv_display_value / 2,
                                                text=f"Full Terminal Value: ${terminal_value / 1e9:.2f}B<br>(Not to scale)",
                                                showarrow=True,
                                                arrowhead=1
                                            )

                                        fig.update_layout(
                                            title="Forecasted Free Cash Flows",
                                            xaxis_title="Forecast Period",
                                            yaxis_title="Free Cash Flow ($)",
                                            plot_bgcolor=VIZ_SETTINGS['background'],
                                            paper_bgcolor=VIZ_SETTINGS['background'],
                                            font=dict(color=VIZ_SETTINGS['text_color']),
                                            height=400
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                        # Add explanation of DCF components
                                        present_value_fcf = dcf_result.get('present_value_fcf')
                                        present_value_terminal = dcf_result.get('present_value_terminal')

                                        if present_value_fcf is not None and present_value_terminal is not None:
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                st.metric("PV of Forecasted FCF", f"${present_value_fcf / 1e9:.2f}B")
                                                fcf_percent = present_value_fcf / (
                                                            present_value_fcf + present_value_terminal) * 100
                                                st.write(f"Represents **{fcf_percent:.1f}%** of total enterprise value")

                                            with col2:
                                                st.metric("PV of Terminal Value",
                                                          f"${present_value_terminal / 1e9:.2f}B")
                                                tv_percent = present_value_terminal / (
                                                            present_value_fcf + present_value_terminal) * 100
                                                st.write(f"Represents **{tv_percent:.1f}%** of total enterprise value")

                                    else:
                                        st.info("Forecasted cash flow data not available.")

                                else:
                                    st.error(f"DCF valuation failed: {dcf_result.get('error', 'Unknown error')}")

                        # DCF explanation
                        with st.expander("How DCF Valuation Works"):
                            st.markdown("""
                            **Discounted Cash Flow (DCF)** valuation estimates the intrinsic value of a company based on its expected future cash flows.

                            #### Key components:

                            1. **Forecast Period**: Typically 5-10 years of projected cash flows
                            2. **Growth Rate**: Expected rate of cash flow growth
                            3. **Discount Rate**: Usually the Weighted Average Cost of Capital (WACC)
                            4. **Terminal Value**: Represents all cash flows beyond the forecast period
                            5. **Margin of Safety**: Discount applied to the final valuation to account for uncertainty

                            DCF is considered one of the most theoretically sound valuation methods, but its accuracy depends heavily on the quality of inputs and assumptions.
                            """)

                    # Relative Valuation tab
                    with valuation_tabs[2]:
                        st.subheader("Relative Valuation Analysis")

                        # Run relative valuation
                        with st.spinner("Calculating relative valuation..."):
                            # Use base valuation model for relative analysis
                            relative_result = valuation_factory.valuation_models["Base"].relative_valuation(ticker,
                                                                                                            financial_data,
                                                                                                            sector)

                            if relative_result and 'valuations' in relative_result and relative_result['valuations']:
                                st.subheader("Valuation Based on Industry Multiples")

                                # Display average value
                                avg_value = relative_result.get('average_value')
                                value_per_share = relative_result.get('value_per_share')
                                current_price = price_data['Close'].iloc[-1] if not price_data.empty else None

                                col1, col2 = st.columns(2)

                                with col1:
                                    if value_per_share and current_price:
                                        upside = ((value_per_share / current_price) - 1) * 100
                                        upside_text = f"{upside:.1f}% {'Upside' if upside > 0 else 'Downside'}"

                                        st.metric(
                                            "Average Value per Share",
                                            f"${value_per_share:.2f}",
                                            upside_text,
                                            delta_color="normal" if upside > 0 else "inverse"
                                        )
                                    else:
                                        st.metric("Average Value per Share",
                                                  f"${value_per_share:.2f}" if value_per_share else "N/A")

                                with col2:
                                    if avg_value:
                                        st.metric("Total Equity Value", f"${avg_value / 1e9:.2f}B")

                                # Create data for multiple comparison chart
                                valuations = relative_result['valuations']

                                labels = []
                                values = []
                                multiples = []

                                for method, data in valuations.items():
                                    if method == 'pe':
                                        label = "P/E"
                                    elif method == 'ps':
                                        label = "P/S"
                                    elif method == 'pb':
                                        label = "P/B"
                                    elif method == 'ev_ebitda':
                                        label = "EV/EBITDA"
                                    else:
                                        label = method.upper()

                                    labels.append(label)
                                    values.append(data.get('value_per_share', 0))
                                    multiples.append(data.get('multiple', 0))

                                # Plot valuations from different multiples
                                st.subheader("Value Estimates by Multiple")

                                # Create a DataFrame for the chart
                                valuation_df = pd.DataFrame({
                                    'Method': labels,
                                    'Value': values
                                })

                                # Create figure
                                fig = go.Figure()

                                # Add bar chart
                                fig.add_trace(
                                    go.Bar(
                                        x=valuation_df['Method'],
                                        y=valuation_df['Value'],
                                        marker_color=COLORS['primary'],
                                        text=[f"${val:.2f}" for val in valuation_df['Value']],
                                        textposition="auto"
                                    )
                                )

                                # Add current price as a horizontal line
                                if current_price:
                                    fig.add_shape(
                                        type="line",
                                        x0=-0.5,
                                        x1=len(labels) - 0.5,
                                        y0=current_price,
                                        y1=current_price,
                                        line=dict(
                                            color=COLORS['warning'],
                                            width=2,
                                            dash="dash"
                                        )
                                    )

                                    fig.add_annotation(
                                        x=len(labels) / 2,
                                        y=current_price,
                                        text=f"Current Price: ${current_price:.2f}",
                                        showarrow=False,
                                        yshift=10,
                                        font=dict(color=COLORS['warning'])
                                    )

                                fig.update_layout(
                                    title="Value per Share by Valuation Method",
                                    xaxis_title="Valuation Method",
                                    yaxis_title="Value per Share ($)",
                                    plot_bgcolor=VIZ_SETTINGS['background'],
                                    paper_bgcolor=VIZ_SETTINGS['background'],
                                    font=dict(color=VIZ_SETTINGS['text_color']),
                                    height=400
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Display multiple details
                                st.subheader("Multiple Details")

                                multiple_data = []
                                for i, method in enumerate(labels):
                                    multiple_data.append({
                                        "Multiple": method,
                                        "Value Used": multiples[i],
                                        "Resulting Share Price": f"${values[i]:.2f}",
                                        "% Difference from Current": f"{((values[i] / current_price) - 1) * 100:.1f}%" if current_price else "N/A"
                                    })

                                st.dataframe(pd.DataFrame(multiple_data), use_container_width=True)

                                # Display metrics used for valuation
                                st.subheader("Financial Metrics Used")
                                metrics = relative_result.get('metrics', {})

                                if metrics:
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        if 'earnings' in metrics:
                                            st.metric("Net Income", f"${metrics['earnings'] / 1e9:.2f}B")

                                        if 'book_value' in metrics:
                                            st.metric("Book Value", f"${metrics['book_value'] / 1e9:.2f}B")

                                    with col2:
                                        if 'revenue' in metrics:
                                            st.metric("Revenue", f"${metrics['revenue'] / 1e9:.2f}B")

                                        if 'enterprise_value' in metrics:
                                            st.metric("Enterprise Value", f"${metrics['enterprise_value'] / 1e9:.2f}B")

                                    with col3:
                                        if 'ebitda' in metrics:
                                            st.metric("EBITDA", f"${metrics['ebitda'] / 1e9:.2f}B")
                            else:
                                st.warning("Could not perform relative valuation due to insufficient data.")

                        # Relative valuation explanation
                        with st.expander("How Relative Valuation Works"):
                            st.markdown("""
                            **Relative Valuation** estimates a company's value by comparing it to similar companies using standardized financial ratios or multiples.

                            #### Common multiples used:

                            1. **P/E (Price to Earnings)**: Based on earnings power
                            2. **P/S (Price to Sales)**: Useful for companies with no profits
                            3. **P/B (Price to Book)**: Compares market value to book value
                            4. **EV/EBITDA**: Enterprise Value to EBITDA, accounts for debt

                            Relative valuation is intuitive and reflects current market sentiment, but may perpetuate market mispricing and ignores company-specific growth trajectories.
                            """)

                    # Asset-Based Valuation tab
                    with valuation_tabs[3]:
                        st.subheader("Asset-Based Valuation Analysis")

                        # Run asset-based valuation
                        with st.spinner("Calculating asset-based valuation..."):
                            # Use base valuation model for asset-based analysis
                            asset_result = valuation_factory.valuation_models["Base"].asset_based_valuation(ticker,
                                                                                                            financial_data,
                                                                                                            sector)

                            if asset_result and 'book_value' in asset_result and asset_result['book_value'] is not None:
                                st.subheader("Asset-Based Value Estimates")

                                # Display different asset-based values
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Book Value per Share
                                    book_value_per_share = asset_result.get('book_value_per_share')
                                    current_price = price_data['Close'].iloc[-1] if not price_data.empty else None

                                    if book_value_per_share and current_price:
                                        pb_ratio = current_price / book_value_per_share
                                        book_vs_market = f"P/B: {pb_ratio:.2f}x"

                                        st.metric(
                                            "Book Value per Share",
                                            f"${book_value_per_share:.2f}",
                                            book_vs_market
                                        )
                                    else:
                                        st.metric("Book Value per Share",
                                                  f"${book_value_per_share:.2f}" if book_value_per_share else "N/A")

                                    # Adjusted Book Value per Share
                                    adjusted_book_value_per_share = asset_result.get('adjusted_book_value_per_share')

                                    if adjusted_book_value_per_share and book_value_per_share:
                                        adjustment = ((adjusted_book_value_per_share / book_value_per_share) - 1) * 100
                                        adj_text = f"{adjustment:.1f}% {'premium' if adjustment > 0 else 'discount'} to book"

                                        st.metric(
                                            "Adjusted Book Value per Share",
                                            f"${adjusted_book_value_per_share:.2f}",
                                            adj_text
                                        )
                                    else:
                                        st.metric("Adjusted Book Value per Share",
                                                  f"${adjusted_book_value_per_share:.2f}" if adjusted_book_value_per_share else "N/A")

                                with col2:
                                    # Liquidation Value per Share
                                    liquidation_value_per_share = asset_result.get('liquidation_value_per_share')

                                    if liquidation_value_per_share and book_value_per_share:
                                        liq_discount = ((liquidation_value_per_share / book_value_per_share) - 1) * 100
                                        liq_text = f"{liq_discount:.1f}% of book value"

                                        st.metric(
                                            "Liquidation Value per Share",
                                            f"${liquidation_value_per_share:.2f}",
                                            liq_text
                                        )
                                    else:
                                        st.metric("Liquidation Value per Share",
                                                  f"${liquidation_value_per_share:.2f}" if liquidation_value_per_share else "N/A")

                                    # Replacement Value per Share
                                    replacement_value_per_share = asset_result.get('replacement_value_per_share')

                                    if replacement_value_per_share and book_value_per_share:
                                        repl_premium = ((replacement_value_per_share / book_value_per_share) - 1) * 100
                                        repl_text = f"{repl_premium:.1f}% {'premium' if repl_premium > 0 else 'discount'} to book"

                                        st.metric(
                                            "Replacement Value per Share",
                                            f"${replacement_value_per_share:.2f}",
                                            repl_text
                                        )
                                    else:
                                        st.metric("Replacement Value per Share",
                                                  f"${replacement_value_per_share:.2f}" if replacement_value_per_share else "N/A")

                                # Create comparison chart of different asset-based values
                                values = []
                                labels = []

                                if book_value_per_share:
                                    values.append(book_value_per_share)
                                    labels.append("Book Value")

                                if adjusted_book_value_per_share:
                                    values.append(adjusted_book_value_per_share)
                                    labels.append("Adjusted Book Value")

                                if liquidation_value_per_share:
                                    values.append(liquidation_value_per_share)
                                    labels.append("Liquidation Value")

                                if replacement_value_per_share:
                                    values.append(replacement_value_per_share)
                                    labels.append("Replacement Value")

                                if current_price:
                                    values.append(current_price)
                                    labels.append("Current Market Price")

                                if values and labels:
                                    # Create data frame for chart
                                    asset_value_df = pd.DataFrame({
                                        'Value Type': labels,
                                        'Value per Share': values
                                    })

                                    # Plot asset-based values
                                    fig = go.Figure()

                                    # Add bar chart
                                    bar_colors = [COLORS['primary'] for _ in range(len(labels) - 1)]
                                    if current_price:
                                        bar_colors.append(COLORS['warning'])  # Different color for current price

                                    fig.add_trace(
                                        go.Bar(
                                            x=asset_value_df['Value Type'],
                                            y=asset_value_df['Value per Share'],
                                            marker_color=bar_colors,
                                            text=[f"${val:.2f}" for val in asset_value_df['Value per Share']],
                                            textposition="auto"
                                        )
                                    )

                                    fig.update_layout(
                                        title="Asset-Based Value Comparison",
                                        xaxis_title="Value Type",
                                        yaxis_title="Value per Share ($)",
                                        plot_bgcolor=VIZ_SETTINGS['background'],
                                        paper_bgcolor=VIZ_SETTINGS['background'],
                                        font=dict(color=VIZ_SETTINGS['text_color']),
                                        height=400
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                # Display total asset values
                                st.subheader("Total Asset Values")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Total Book Value",
                                              f"${asset_result.get('book_value') / 1e9:.2f}B" if asset_result.get(
                                                  'book_value') else "N/A")
                                    st.metric("Total Adjusted Book Value",
                                              f"${asset_result.get('adjusted_book_value') / 1e9:.2f}B" if asset_result.get(
                                                  'adjusted_book_value') else "N/A")

                                with col2:
                                    st.metric("Total Liquidation Value",
                                              f"${asset_result.get('liquidation_value') / 1e9:.2f}B" if asset_result.get(
                                                  'liquidation_value') else "N/A")
                                    st.metric("Total Replacement Value",
                                              f"${asset_result.get('replacement_value') / 1e9:.2f}B" if asset_result.get(
                                                  'replacement_value') else "N/A")

                            else:
                                st.warning("Could not perform asset-based valuation due to insufficient data.")

                        # Asset-based valuation explanation
                        with st.expander("How Asset-Based Valuation Works"):
                            st.markdown("""
                            **Asset-Based Valuation** determines a company's value based on the value of its underlying assets minus its liabilities.

                            #### Common approaches:

                            1. **Book Value**: Net assets at accounting value (assets - liabilities)
                            2. **Adjusted Book Value**: Book value with adjustments for asset value discrepancies
                            3. **Liquidation Value**: Expected proceeds if assets were sold in a liquidation
                            4. **Replacement Value**: Cost to replicate the company's assets at current prices

                            Asset-based valuation is particularly useful for capital-intensive businesses, real estate companies, and financial institutions. It provides a floor value but may undervalue companies with significant intangible assets or growth potential.
                            """)

                    # Sensitivity Analysis tab
                    with valuation_tabs[4]:
                        st.subheader("Sensitivity Analysis")
                        st.write("Analyze how changes in key assumptions affect the company's valuation.")

                        # Parameters to test
                        st.subheader("Select Parameters")

                        col1, col2 = st.columns(2)

                        with col1:
                            sensitivity_param1 = st.selectbox(
                                "Parameter 1",
                                ["Growth Rate", "Discount Rate", "Terminal Growth Rate", "Margin of Safety"],
                                index=0
                            )

                            range_p1 = 0.05  # Default range ±5%
                            if sensitivity_param1 == "Growth Rate":
                                base_p1 = 0.10  # 10%
                                min_p1 = 0.05
                                max_p1 = 0.15
                                step_p1 = 0.01
                            elif sensitivity_param1 == "Discount Rate":
                                base_p1 = 0.10  # 10%
                                min_p1 = 0.08
                                max_p1 = 0.12
                                step_p1 = 0.005
                            elif sensitivity_param1 == "Terminal Growth Rate":
                                base_p1 = 0.02  # 2%
                                min_p1 = 0.01
                                max_p1 = 0.03
                                step_p1 = 0.002
                            else:  # Margin of Safety
                                base_p1 = 0.25  # 25%
                                min_p1 = 0.15
                                max_p1 = 0.35
                                step_p1 = 0.05

                            p1_range = st.slider(
                                f"{sensitivity_param1} Range",
                                min_value=min_p1,
                                max_value=max_p1,
                                value=(min_p1, max_p1),
                                step=step_p1
                            )

                            p1_steps = st.number_input("Number of Steps", min_value=3, max_value=10, value=5)

                        with col2:
                            sensitivity_param2 = st.selectbox(
                                "Parameter 2",
                                ["Growth Rate", "Discount Rate", "Terminal Growth Rate", "Margin of Safety"],
                                index=1
                            )

                            range_p2 = 0.05  # Default range ±5%
                            if sensitivity_param2 == "Growth Rate":
                                base_p2 = 0.10  # 10%
                                min_p2 = 0.05
                                max_p2 = 0.15
                                step_p2 = 0.01
                            elif sensitivity_param2 == "Discount Rate":
                                base_p2 = 0.10  # 10%
                                min_p2 = 0.08
                                max_p2 = 0.12
                                step_p2 = 0.005
                            elif sensitivity_param2 == "Terminal Growth Rate":
                                base_p2 = 0.02  # 2%
                                min_p2 = 0.01
                                max_p2 = 0.03
                                step_p2 = 0.002
                            else:  # Margin of Safety
                                base_p2 = 0.25  # 25%
                                min_p2 = 0.15
                                max_p2 = 0.35
                                step_p2 = 0.05

                            p2_range = st.slider(
                                f"{sensitivity_param2} Range",
                                min_value=min_p2,
                                max_value=max_p2,
                                value=(min_p2, max_p2),
                                step=step_p2
                            )

                            p2_steps = st.number_input("Number of Steps ", min_value=3, max_value=10, value=5)

                        # Run sensitivity analysis
                        if st.button("Run Sensitivity Analysis", type="primary"):
                            with st.spinner("Running sensitivity analysis..."):
                                try:
                                    # Create parameter ranges
                                    p1_values = np.linspace(p1_range[0], p1_range[1], p1_steps)
                                    p2_values = np.linspace(p2_range[0], p2_range[1], p2_steps)

                                    # Initialize results matrix
                                    results = np.zeros((len(p1_values), len(p2_values)))

                                    # For each combination of parameters, calculate valuation
                                    for i, p1 in enumerate(p1_values):
                                        for j, p2 in enumerate(p2_values):
                                            # Clone financial data with modified parameters
                                            sensitivity_data = financial_data.copy()
                                            sensitivity_data['custom_params'] = {}

                                            # Set base parameters
                                            sensitivity_data['custom_params']['forecast_years'] = 5
                                            sensitivity_data['custom_params']['growth_rate'] = 0.10
                                            sensitivity_data['custom_params']['discount_rate'] = 0.10
                                            sensitivity_data['custom_params']['terminal_growth'] = 0.02
                                            sensitivity_data['custom_params']['margin_of_safety'] = 0.25

                                            # Modify the parameters being tested
                                            if sensitivity_param1 == "Growth Rate":
                                                sensitivity_data['custom_params']['growth_rate'] = p1
                                            elif sensitivity_param1 == "Discount Rate":
                                                sensitivity_data['custom_params']['discount_rate'] = p1
                                            elif sensitivity_param1 == "Terminal Growth Rate":
                                                sensitivity_data['custom_params']['terminal_growth'] = p1
                                            else:  # Margin of Safety
                                                sensitivity_data['custom_params']['margin_of_safety'] = p1

                                            if sensitivity_param2 == "Growth Rate":
                                                sensitivity_data['custom_params']['growth_rate'] = p2
                                            elif sensitivity_param2 == "Discount Rate":
                                                sensitivity_data['custom_params']['discount_rate'] = p2
                                            elif sensitivity_param2 == "Terminal Growth Rate":
                                                sensitivity_data['custom_params']['terminal_growth'] = p2
                                            else:  # Margin of Safety
                                                sensitivity_data['custom_params']['margin_of_safety'] = p2

                                            # Run valuation model with these parameters
                                            # This is a simplified model - in a real implementation,
                                            # we'd use a more sophisticated model
                                            try:
                                                # Use a simple DCF formula for speed
                                                # In real implementation, this would call the valuation model
                                                current_fcf = 1e9  # Placeholder starting FCF
                                                growth_rate = sensitivity_data['custom_params']['growth_rate']
                                                discount_rate = sensitivity_data['custom_params']['discount_rate']
                                                terminal_growth = sensitivity_data['custom_params']['terminal_growth']
                                                forecast_years = sensitivity_data['custom_params']['forecast_years']
                                                margin_of_safety = sensitivity_data['custom_params']['margin_of_safety']

                                                # Calculate future cash flows
                                                total_pv = 0
                                                for year in range(1, forecast_years + 1):
                                                    fcf = current_fcf * (1 + growth_rate) ** year
                                                    pv = fcf / (1 + discount_rate) ** year
                                                    total_pv += pv

                                                # Terminal value
                                                terminal_fcf = current_fcf * (1 + growth_rate) ** forecast_years * (1 + terminal_growth)
                                                terminal_value = terminal_fcf / (discount_rate - terminal_growth)
                                                pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

                                                # Total value
                                                total_value = total_pv + pv_terminal

                                                # Per share value with margin of safety
                                                shares = financial_data['market_data']['shares_outstanding']
                                                if shares:
                                                    value_per_share = (total_value / shares) * (1 - margin_of_safety)
                                                    results[i, j] = value_per_share
                                                else:
                                                    results[i, j] = 0
                                            except:
                                                # Handle valuation errors by setting value to zero
                                                results[i, j] = 0

                                    # Create heatmap of results
                                    fig = go.Figure(data=go.Heatmap(
                                        z=results,
                                        x=[f"{p:.2%}" for p in p2_values],
                                        y=[f"{p:.2%}" for p in p1_values],
                                        colorscale='RdBu_r',
                                        colorbar=dict(title="Value per Share ($)"),
                                        text=[[f"${val:.2f}" for val in row] for row in results],
                                        hoverinfo="text"
                                    ))

                                    # Add current price contour if available
                                    current_price = price_data['Close'].iloc[-1] if not price_data.empty else None
                                    if current_price:
                                        # Try to add contour for current price
                                        fig.add_contour(
                                            z=results,
                                            x=[f"{p:.2%}" for p in p2_values],
                                            y=[f"{p:.2%}" for p in p1_values],
                                            contours=dict(
                                                start=current_price,
                                                end=current_price,
                                                size=0.1,
                                                showlabels=True,
                                                labelfont=dict(color=COLORS['warning'])
                                            ),
                                            line=dict(color=COLORS['warning']),
                                            showscale=False
                                        )

                                    fig.update_layout(
                                        title=f"Sensitivity Analysis: {sensitivity_param1} vs {sensitivity_param2}",
                                        xaxis_title=sensitivity_param2,
                                        yaxis_title=sensitivity_param1,
                                        plot_bgcolor=VIZ_SETTINGS['background'],
                                        paper_bgcolor=VIZ_SETTINGS['background'],
                                        font=dict(color=VIZ_SETTINGS['text_color']),
                                        height=500
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Add interpretation
                                    st.subheader("Analysis Interpretation")

                                    # Find max and min values
                                    max_val = np.max(results)
                                    min_val = np.min(results[results > 0]) if np.any(
                                        results > 0) else 0
                                    range_val = max_val - min_val

                                    st.write(f"""
                                    - **Value Range**: ${min_val:.2f} to ${max_val:.2f} per share (variation of {range_val / min_val * 100:.1f}%)
                                    - **Key Insights**:
                                      - The highest value occurs at {sensitivity_param1.lower()} = {p1_values[np.unravel_index(results.argmax(), results.shape)[0]]:.2%} and {sensitivity_param2.lower()} = {p2_values[np.unravel_index(results.argmax(), results.shape)[1]]:.2%}
                                      - Valuation is most sensitive to changes in {sensitivity_param1 if np.std(results, axis=0).mean() > np.std(results, axis=1).mean() else sensitivity_param2}
                                    """)

                                except Exception as e:
                                    st.error(f"An error occurred during sensitivity analysis: {e}")
                                    st.info("Please check the parameters and try again.")
                except Exception as e:
                    st.error(f"An error occurred in the Valuation section: {str(e)}")
                    st.error(traceback.format_exc())

            # Tab 4: Risk Analysis
            with tabs[3]:
                st.info(
                    "Risk analysis tab is under development. This will include bankruptcy risk models and stress testing.")

            # Tab 5: Peer Comparison
            with tabs[4]:
                st.info(
                    "Peer comparison tab is under development. This will enable comparison with similar companies in the same sector.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            logger.error(f"Error in analysis: {traceback.format_exc()}")
            st.info("Please try again later or choose a different company to analyze.")


if __name__ == "__main__":
    main()