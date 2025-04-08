import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import project modules
from config import COLORS, SECTOR_SPECIFIC_RATIOS
from utils.data_loader import DataLoader
from models.ratio_analysis import FinancialRatioAnalyzer
from utils.visualization import FinancialVisualizer
from industry.sector_mapping import get_sector_peers
from valuation.sector_factor import ValuationFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('peer_comparison')


def run_peer_comparison_page():
    """Main function to run the peer comparison page"""
    st.title("Peer Comparison Analysis")
    st.markdown("Compare a company with its industry peers to evaluate relative performance and valuation.")

    # Initialize data loader and analyzers
    data_loader = DataLoader()
    ratio_analyzer = FinancialRatioAnalyzer()
    visualizer = FinancialVisualizer(theme="dark")
    valuation_factory = ValuationFactory(data_loader)

    # Sidebar for company selection and comparison settings
    with st.sidebar:
        st.header("Comparison Settings")

        # Primary company selection
        st.subheader("Primary Company")
        primary_ticker = st.text_input("Enter ticker symbol:", "AAPL").upper()

        # Load company info to get sector
        if primary_ticker:
            company_info = data_loader.get_company_info(primary_ticker)
            if not company_info:
                st.error(f"Could not find information for {primary_ticker}. Please check the ticker symbol.")
                return

            sector = company_info.get('sector', 'Unknown')
            if sector == 'Unknown':
                st.warning(f"Could not determine sector for {primary_ticker}. Some analysis may be limited.")

            st.info(f"Company: {company_info.get('name', primary_ticker)}")
            st.info(f"Sector: {sector}")

            # Peer companies selection
            st.subheader("Peer Companies")

            # Get peer companies
            peers = get_sector_peers(primary_ticker, sector, limit=10)

            if not peers:
                st.warning(f"No peers found for {primary_ticker} in sector {sector}.")
                peer_tickers = []
            else:
                # Allow selection of peers
                peer_options = [f"{peer['symbol']} - {peer['name']}" for peer in peers]
                selected_peers = st.multiselect(
                    "Select peer companies:",
                    options=peer_options,
                    default=peer_options[:min(5, len(peer_options))]  # Default select first 5
                )

                # Extract tickers from selected peers
                peer_tickers = [peer.split(" - ")[0] for peer in selected_peers]

            # Add option to manually add peers
            custom_peer = st.text_input("Add custom peer (ticker):")
            if custom_peer:
                custom_peer = custom_peer.upper()
                if custom_peer not in peer_tickers and custom_peer != primary_ticker:
                    peer_tickers.append(custom_peer)

            # Comparison period
            st.subheader("Comparison Period")
            period_options = {
                "1 Year": 365,
                "3 Years": 3 * 365,
                "5 Years": 5 * 365,
                "10 Years": 10 * 365,
                "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
            }
            selected_period = st.selectbox("Select time period:", list(period_options.keys()))
            days = period_options[selected_period]

            # Comparison metrics
            st.subheader("Comparison Metrics")

            # Get relevant metrics for this sector
            sector_metrics = SECTOR_SPECIFIC_RATIOS.get(sector, [])

            # Default metrics by category
            default_metrics = {
                "Valuation": ["P/E Ratio", "P/S Ratio", "P/B Ratio", "EV/EBITDA"],
                "Profitability": ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"],
                "Growth": ["Revenue Growth", "EPS Growth"],
                "Financial Health": ["Current Ratio", "Debt/Equity", "Interest Coverage"]
            }

            # Add sector-specific metrics
            if sector_metrics:
                default_metrics["Sector-Specific"] = sector_metrics

            # Allow selection of metrics by category
            selected_metrics = {}
            for category, metrics in default_metrics.items():
                with st.expander(f"{category} Metrics", expanded=(category == "Valuation")):
                    selected_metrics[category] = st.multiselect(
                        f"Select {category.lower()} metrics:",
                        options=metrics,
                        default=metrics[:min(3, len(metrics))]  # Default select first 3
                    )

            # Flatten selected metrics for later use
            flat_selected_metrics = [metric for metrics in selected_metrics.values() for metric in metrics]

            # Comparison button
            compare_button = st.button("Compare Companies", type="primary")

        else:
            compare_button = False
            peer_tickers = []

    # Main content area - show comparison results
    if primary_ticker and compare_button:
        # Display loading message
        with st.spinner(f"Analyzing {primary_ticker} and peer companies..."):
            # Combine primary ticker with peer tickers
            all_tickers = [primary_ticker] + peer_tickers

            try:
                # Load financial data for all companies
                company_data = {}
                company_names = {}
                company_sectors = {}
                company_ratios = {}

                # Calculate end date and start date
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

                # Load data for each company
                for ticker in all_tickers:
                    # Get company info
                    info = data_loader.get_company_info(ticker)
                    if not info:
                        st.warning(f"Could not find information for {ticker}. Skipping...")
                        continue

                    company_names[ticker] = info.get('name', ticker)
                    company_sectors[ticker] = info.get('sector', 'Unknown')

                    # Load financial statements
                    income_stmt = data_loader.get_financial_statements(ticker, 'income', 'annual')
                    balance_sheet = data_loader.get_financial_statements(ticker, 'balance', 'annual')
                    cash_flow = data_loader.get_financial_statements(ticker, 'cash', 'annual')

                    # Get market data
                    price_data = data_loader.get_historical_prices(ticker, start_date, end_date)

                    if price_data.empty:
                        st.warning(f"Could not load price data for {ticker}. Some analysis will be limited.")

                    # Create financial data dictionary
                    financial_data = {
                        'income_statement': income_stmt,
                        'balance_sheet': balance_sheet,
                        'cash_flow': cash_flow,
                        'market_data': {
                            'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                            'market_cap': info.get('market_cap'),
                            'shares_outstanding': info.get('shares_outstanding')
                        }
                    }

                    company_data[ticker] = financial_data

                    # Calculate financial ratios
                    company_ratios[ticker] = ratio_analyzer.calculate_ratios(financial_data)

                # Check if we have data for at least the primary company
                if primary_ticker not in company_data:
                    st.error(f"Could not load required data for {primary_ticker}.")
                    return

                # Display company header
                st.header(f"Comparison: {company_names.get(primary_ticker, primary_ticker)} vs. Peers")

                # Display key company metrics in a row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Primary Company", primary_ticker)
                with col2:
                    st.metric("Sector", company_sectors.get(primary_ticker, "N/A"))
                with col3:
                    num_peers = len([ticker for ticker in company_data if ticker != primary_ticker])
                    st.metric("Number of Peers", num_peers)
                with col4:
                    st.metric("Time Period", selected_period)

                # Create tabs for different analyses
                tabs = st.tabs([
                    "Performance Comparison",
                    "Valuation Metrics",
                    "Financial Metrics",
                    "Growth Metrics",
                    "Relative Valuation"
                ])

                # Tab 1: Performance Comparison
                with tabs[0]:
                    st.subheader("Stock Price Performance")

                    # Load price data for chart
                    price_series = {}
                    for ticker in all_tickers:
                        if ticker in company_data:
                            # Get price data
                            price_data = data_loader.get_historical_prices(ticker, start_date, end_date)
                            if not price_data.empty:
                                # Normalize to percentage change from start
                                first_price = price_data['Close'].iloc[0]
                                price_series[ticker] = (price_data['Close'] / first_price - 1) * 100

                    # Create chart
                    if price_series:
                        fig = visualizer.plot_price_comparison(
                            price_series,
                            company_names,
                            primary_ticker=primary_ticker,
                            title=f"Price Performance ({selected_period})",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No price data available for comparison.")

                    # Display key performance metrics
                    st.subheader("Performance Metrics")

                    # Calculate performance metrics
                    performance_data = []
                    for ticker in all_tickers:
                        if ticker in company_data:
                            # Get price data
                            price_data = data_loader.get_historical_prices(ticker, start_date, end_date)
                            if not price_data.empty:
                                # Calculate metrics
                                returns = price_data['Close'].pct_change().dropna()
                                total_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0] - 1) * 100
                                volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility

                                # Add to data
                                performance_data.append({
                                    "Company": company_names.get(ticker, ticker),
                                    "Ticker": ticker,
                                    f"Total Return ({selected_period})": f"{total_return:.2f}%",
                                    "Annualized Volatility": f"{volatility:.2f}%",
                                    "Sharpe Ratio": f"{(total_return / volatility):.2f}" if volatility != 0 else "N/A",
                                    "Current Price": f"${price_data['Close'].iloc[-1]:.2f}",
                                    "Market Cap": f"${company_data[ticker]['market_data'].get('market_cap', 0) / 1e9:.2f}B"
                                    if company_data[ticker]['market_data'].get('market_cap') else "N/A"
                                })

                    # Display performance metrics table
                    if performance_data:
                        performance_df = pd.DataFrame(performance_data)

                        # Highlight primary company
                        def highlight_primary(s):
                            return ['background-color: #1f1f1f' if s.Ticker == primary_ticker else '' for _ in s]

                        st.dataframe(
                            performance_df.style.apply(highlight_primary, axis=1),
                            use_container_width=True
                        )
                    else:
                        st.warning("No performance data available.")

                # Tab 2: Valuation Metrics
                with tabs[1]:
                    st.subheader("Valuation Metrics Comparison")

                    # Extract valuation metrics
                    valuation_metrics = ["pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "ev_revenue"]

                    # Prepare data for radar chart
                    radar_data = {}
                    for ticker in all_tickers:
                        if ticker in company_ratios:
                            radar_data[ticker] = {}
                            for metric in valuation_metrics:
                                if metric in company_ratios[ticker].get('valuation', {}):
                                    radar_data[ticker][metric] = company_ratios[ticker]['valuation'][metric]

                    # Create radar chart
                    if radar_data:
                        fig = visualizer.plot_radar_comparison(
                            radar_data,
                            company_names,
                            primary_ticker=primary_ticker,
                            title="Valuation Metrics Comparison",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valuation metrics available for comparison.")

                    # Display valuation metrics table
                    st.subheader("Valuation Metrics Details")

                    # Prepare data for table
                    valuation_data = []
                    metrics_display = {
                        "pe_ratio": "P/E Ratio",
                        "ps_ratio": "P/S Ratio",
                        "pb_ratio": "P/B Ratio",
                        "ev_ebitda": "EV/EBITDA",
                        "ev_revenue": "EV/Revenue"
                    }

                    for ticker in all_tickers:
                        if ticker in company_ratios:
                            row = {
                                "Company": company_names.get(ticker, ticker),
                                "Ticker": ticker
                            }

                            # Add metrics
                            for metric, display_name in metrics_display.items():
                                value = company_ratios[ticker].get('valuation', {}).get(metric)
                                row[display_name] = f"{value:.2f}" if value is not None else "N/A"

                            valuation_data.append(row)

                    # Display table
                    if valuation_data:
                        valuation_df = pd.DataFrame(valuation_data)

                        # Highlight primary company
                        def highlight_primary(s):
                            return ['background-color: #1f1f1f' if s.Ticker == primary_ticker else '' for _ in s]

                        st.dataframe(
                            valuation_df.style.apply(highlight_primary, axis=1),
                            use_container_width=True
                        )
                    else:
                        st.warning("No valuation data available.")

                # Tab 3: Financial Metrics
                with tabs[2]:
                    st.subheader("Financial Health Comparison")

                    # Collect financial metrics
                    categories = ['profitability', 'liquidity', 'leverage', 'efficiency']

                    # Create tabs for each metric category
                    financial_tabs = st.tabs([cat.capitalize() for cat in categories])

                    for i, category in enumerate(categories):
                        with financial_tabs[i]:
                            st.subheader(f"{category.capitalize()} Metrics")

                            # Get metrics for this category
                            metrics = set()
                            for ticker in company_ratios:
                                metrics.update(company_ratios[ticker].get(category, {}).keys())

                            metrics = sorted(list(metrics))

                            if not metrics:
                                st.warning(f"No {category} metrics available for comparison.")
                                continue

                            # Prepare data for bar chart
                            bar_data = {}
                            bar_companies = []

                            for ticker in all_tickers:
                                if ticker in company_ratios:
                                    company_name = company_names.get(ticker, ticker)
                                    bar_companies.append(company_name)

                                    for metric in metrics:
                                        if metric not in bar_data:
                                            bar_data[metric] = []

                                        value = company_ratios[ticker].get(category, {}).get(metric)
                                        bar_data[metric].append(value if value is not None else np.nan)

                            # Create bar chart
                            if bar_data and bar_companies:
                                fig = visualizer.plot_bar_comparison(
                                    bar_data,
                                    bar_companies,
                                    primary_index=0 if primary_ticker in all_tickers else None,
                                    title=f"{category.capitalize()} Metrics Comparison",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            # Display metrics table
                            st.subheader(f"{category.capitalize()} Metrics Details")

                            # Prepare data for table
                            metric_data = []

                            for ticker in all_tickers:
                                if ticker in company_ratios:
                                    row = {
                                        "Company": company_names.get(ticker, ticker),
                                        "Ticker": ticker
                                    }

                                    # Add metrics
                                    for metric in metrics:
                                        value = company_ratios[ticker].get(category, {}).get(metric)

                                        # Format based on metric type
                                        if "margin" in metric or "return" in metric or metric in ["roe", "roa"]:
                                            row[metric.replace('_',
                                                               ' ').title()] = f"{value * 100:.2f}%" if value is not None else "N/A"
                                        else:
                                            row[metric.replace('_',
                                                               ' ').title()] = f"{value:.2f}" if value is not None else "N/A"

                                    metric_data.append(row)

                            # Display table
                            if metric_data:
                                metric_df = pd.DataFrame(metric_data)

                                # Highlight primary company
                                def highlight_primary(s):
                                    return ['background-color: #1f1f1f' if s.Ticker == primary_ticker else '' for _ in
                                            s]

                                st.dataframe(
                                    metric_df.style.apply(highlight_primary, axis=1),
                                    use_container_width=True
                                )
                            else:
                                st.warning(f"No {category} data available.")

                # Tab 4: Growth Metrics
                with tabs[3]:
                    st.subheader("Growth Metrics Comparison")

                    # Check if we have growth data
                    growth_metrics = ["revenue_growth", "net_income_growth"]

                    # Prepare data for visualization
                    growth_data = {}
                    for ticker in all_tickers:
                        if ticker in company_ratios and 'growth' in company_ratios[ticker]:
                            growth_data[ticker] = company_ratios[ticker]['growth']

                    if not growth_data:
                        st.warning("No growth metrics available for comparison.")
                    else:
                        # Create bar chart for growth metrics
                        fig = visualizer.plot_growth_comparison(
                            growth_data,
                            company_names,
                            primary_ticker=primary_ticker,
                            title="Growth Metrics Comparison",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display growth metrics table
                        st.subheader("Growth Metrics Details")

                        # Prepare data for table
                        growth_table_data = []

                        for ticker in all_tickers:
                            if ticker in company_ratios:
                                row = {
                                    "Company": company_names.get(ticker, ticker),
                                    "Ticker": ticker
                                }

                                # Add growth metrics
                                for metric in growth_metrics:
                                    value = company_ratios[ticker].get('growth', {}).get(metric)
                                    row[metric.replace('_',
                                                       ' ').title()] = f"{value * 100:.2f}%" if value is not None else "N/A"

                                growth_table_data.append(row)

                        # Display table
                        if growth_table_data:
                            growth_df = pd.DataFrame(growth_table_data)

                            # Highlight primary company
                            def highlight_primary(s):
                                return ['background-color: #1f1f1f' if s.Ticker == primary_ticker else '' for _ in s]

                            st.dataframe(
                                growth_df.style.apply(highlight_primary, axis=1),
                                use_container_width=True
                            )
                        else:
                            st.warning("No growth data available.")

                # Tab 5: Relative Valuation
                with tabs[4]:
                    st.subheader("Relative Valuation Analysis")

                    # Calculate intrinsic value for each company
                    valuations = {}
                    for ticker in all_tickers:
                        if ticker in company_data:
                            # Get company sector
                            sector = company_sectors.get(ticker)

                            try:
                                # Get valuation using sector-specific model
                                valuation = valuation_factory.get_company_valuation(ticker, sector)
                                valuations[ticker] = valuation
                            except Exception as e:
                                st.warning(f"Could not calculate valuation for {ticker}: {str(e)}")

                    # Display valuation results
                    st.subheader("Fair Value Estimates")

                    # Prepare data for table
                    valuation_table_data = []

                    for ticker in all_tickers:
                        if ticker in valuations:
                            valuation = valuations[ticker]

                            current_price = company_data[ticker]['market_data'].get('share_price')
                            fair_value = valuation.get('value_per_share')

                            if current_price and fair_value:
                                upside = (fair_value / current_price - 1) * 100

                                valuation_table_data.append({
                                    "Company": company_names.get(ticker, ticker),
                                    "Ticker": ticker,
                                    "Current Price": f"${current_price:.2f}",
                                    "Fair Value": f"${fair_value:.2f}",
                                    "Upside/Downside": f"{upside:.2f}%",
                                    "Valuation Method": valuation.get('method', 'N/A').upper(),
                                })

                    # Display table
                    if valuation_table_data:
                        valuation_df = pd.DataFrame(valuation_table_data)

                        # Highlight primary company
                        def highlight_primary(s):
                            return ['background-color: #1f1f1f' if s.Ticker == primary_ticker else '' for _ in s]

                        # Color upside/downside
                        def color_upside(val):
                            try:
                                value = float(val.strip('%'))
                                if value > 10:
                                    return 'color: #74f174'  # Green for positive
                                elif value < -10:
                                    return 'color: #faa1a4'  # Red for negative
                                else:
                                    return 'color: #fff59d'  # Yellow for neutral
                            except:
                                return ''

                        # Apply styling
                        styled_df = valuation_df.style.apply(highlight_primary, axis=1)
                        styled_df = styled_df.applymap(color_upside, subset=['Upside/Downside'])

                        st.dataframe(styled_df, use_container_width=True)

                        # Create bar chart for upside/downside
                        upside_data = {}
                        for row in valuation_table_data:
                            ticker = row["Ticker"]
                            upside = float(row["Upside/Downside"].strip('%'))
                            upside_data[ticker] = upside

                        if upside_data:
                            fig = visualizer.plot_valuation_upside(
                                upside_data,
                                company_names,
                                primary_ticker=primary_ticker,
                                title="Valuation Upside/Downside",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valuation data available.")

                    # Display valuation summary
                    if primary_ticker in valuations:
                        st.subheader(f"Valuation Summary for {primary_ticker}")

                        primary_valuation = valuations[primary_ticker]
                        method = primary_valuation.get('method', 'dcf').upper()

                        # Display key valuation parameters
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Fair Value",
                                f"${primary_valuation.get('value_per_share', 0):.2f}",
                                f"{upside_data.get(primary_ticker, 0):.2f}% vs Current"
                            )

                        with col2:
                            if method == 'DCF':
                                st.metric(
                                    "Discount Rate",
                                    f"{primary_valuation.get('discount_rate', 0) * 100:.2f}%"
                                )
                            elif method == 'RELATIVE':
                                st.metric(
                                    "Avg Multiple",
                                    f"{primary_valuation.get('average_multiple', 0):.2f}x"
                                )

                        with col3:
                            if method == 'DCF':
                                st.metric(
                                    "Terminal Growth",
                                    f"{primary_valuation.get('terminal_growth', 0) * 100:.2f}%"
                                )
                            elif method == 'RELATIVE':
                                st.metric(
                                    "Sector Discount",
                                    f"{primary_valuation.get('sector_discount', 0) * 100:.2f}%"
                                )

                        # Display more details in expandable section
                        with st.expander("View Detailed Valuation Parameters"):
                            # Create a formatted display of valuation parameters
                            for key, value in primary_valuation.items():
                                if key not in ['company', 'error', 'method', 'value_per_share', 'ticker']:
                                    if isinstance(value, (int, float)):
                                        if 'rate' in key or 'growth' in key or 'margin' in key:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value * 100:.2f}%")
                                        else:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                                    elif isinstance(value, dict):
                                        st.write(f"**{key.replace('_', ' ').title()}:**")
                                        st.json(value)
                                    elif isinstance(value, list):
                                        st.write(f"**{key.replace('_', ' ').title()}:**")
                                        if all(isinstance(item, (int, float)) for item in value):
                                            st.write(", ".join(f"{item:.2f}" for item in value))
                                        else:
                                            st.write(str(value))
                                    else:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

            except Exception as e:
                st.error(f"An error occurred during peer comparison analysis: {str(e)}")
                logger.error(f"Error in peer comparison: {str(e)}")


# For direct execution
if __name__ == "__main__":
    run_peer_comparison_page()