import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import UI_SETTINGS, COLORS, VIZ_SETTINGS, SECTOR_MAPPING
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_screener')


def run_stock_screener_page():
    """Main function to run the stock screener page"""
    st.title("Stock Screener")
    st.markdown("Screen stocks based on financial metrics, performance, and sector-specific criteria.")

    # Initialize data loader and visualizer
    data_loader = DataLoader()
    visualizer = FinancialVisualizer(theme="dark")
    ratio_analyzer = FinancialRatioAnalyzer()

    # Sidebar for filter controls
    with st.sidebar:
        st.header("Screening Filters")

        # Predefined screening strategies
        st.subheader("Screening Strategies")
        screening_strategies = {
            "Custom": "Build your own custom filter",
            "Value Investing": "Low P/E, P/B ratios, high dividend yield",
            "Growth Investing": "High revenue growth, earnings growth",
            "Quality Investing": "Strong ROE, low debt, consistent profitability",
            "Dividend Investing": "High dividend yield, dividend growth",
            "Momentum": "Strong price performance, positive earnings surprises"
        }

        selected_strategy = st.selectbox(
            "Select a screening strategy:",
            options=list(screening_strategies.keys())
        )

        st.info(screening_strategies[selected_strategy])

        # Apply pre-defined filters based on strategy
        if selected_strategy != "Custom":
            st.info("Predefined filters will be applied based on your selected strategy.")

        # Basic filters
        st.subheader("Basic Filters")

        # Market cap filter
        market_cap_ranges = {
            "All": (0, float('inf')),
            "Mega Cap (>$200B)": (200e9, float('inf')),
            "Large Cap ($10B-$200B)": (10e9, 200e9),
            "Mid Cap ($2B-$10B)": (2e9, 10e9),
            "Small Cap ($300M-$2B)": (300e6, 2e9),
            "Micro Cap (<$300M)": (0, 300e6)
        }

        market_cap_filter = st.selectbox(
            "Market Capitalization:",
            options=list(market_cap_ranges.keys()),
            index=0
        )

        # Sector filter
        sectors = ["All"] + list(SECTOR_MAPPING.keys())
        selected_sector = st.selectbox(
            "Sector:",
            options=sectors,
            index=0
        )

        # If a sector is selected, show subsector filter
        if selected_sector != "All":
            subsectors = ["All"] + SECTOR_MAPPING.get(selected_sector, [])
            selected_subsector = st.selectbox(
                "Subsector:",
                options=subsectors,
                index=0
            )
        else:
            selected_subsector = "All"

        # Price filter
        st.subheader("Price Filter")
        min_price, max_price = st.slider(
            "Price Range ($):",
            min_value=0.0,
            max_value=5000.0,
            value=(0.0, 5000.0),
            step=1.0
        )

        # Advanced filters based on strategy
        st.subheader("Advanced Filters")

        # Create expandable sections for different filter categories
        with st.expander("Valuation Filters", expanded=(selected_strategy == "Value Investing")):
            # P/E Ratio
            pe_filter = st.checkbox("Filter by P/E Ratio", value=(selected_strategy == "Value Investing"))
            if pe_filter:
                min_pe, max_pe = st.slider(
                    "P/E Ratio Range:",
                    min_value=0.0,
                    max_value=200.0,
                    value=(0.0, 25.0 if selected_strategy == "Value Investing" else 200.0),
                    step=0.5
                )
            else:
                min_pe, max_pe = 0.0, float('inf')

            # P/B Ratio
            pb_filter = st.checkbox("Filter by P/B Ratio", value=(selected_strategy == "Value Investing"))
            if pb_filter:
                min_pb, max_pb = st.slider(
                    "P/B Ratio Range:",
                    min_value=0.0,
                    max_value=50.0,
                    value=(0.0, 3.0 if selected_strategy == "Value Investing" else 50.0),
                    step=0.1
                )
            else:
                min_pb, max_pb = 0.0, float('inf')

            # P/S Ratio
            ps_filter = st.checkbox("Filter by P/S Ratio", value=(selected_strategy == "Value Investing"))
            if ps_filter:
                min_ps, max_ps = st.slider(
                    "P/S Ratio Range:",
                    min_value=0.0,
                    max_value=50.0,
                    value=(0.0, 2.0 if selected_strategy == "Value Investing" else 50.0),
                    step=0.1
                )
            else:
                min_ps, max_ps = 0.0, float('inf')

            # EV/EBITDA
            ev_ebitda_filter = st.checkbox("Filter by EV/EBITDA", value=(selected_strategy == "Value Investing"))
            if ev_ebitda_filter:
                min_ev_ebitda, max_ev_ebitda = st.slider(
                    "EV/EBITDA Range:",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 10.0 if selected_strategy == "Value Investing" else 100.0),
                    step=0.5
                )
            else:
                min_ev_ebitda, max_ev_ebitda = 0.0, float('inf')

        with st.expander("Profitability Filters", expanded=(selected_strategy == "Quality Investing")):
            # ROE
            roe_filter = st.checkbox("Filter by ROE", value=(selected_strategy == "Quality Investing"))
            if roe_filter:
                min_roe, max_roe = st.slider(
                    "ROE Range (%):",
                    min_value=-50.0,
                    max_value=100.0,
                    value=(15.0 if selected_strategy == "Quality Investing" else 0.0, 100.0),
                    step=1.0
                )
            else:
                min_roe, max_roe = -float('inf'), float('inf')

            # Net Margin
            net_margin_filter = st.checkbox("Filter by Net Margin", value=(selected_strategy == "Quality Investing"))
            if net_margin_filter:
                min_net_margin, max_net_margin = st.slider(
                    "Net Margin Range (%):",
                    min_value=-50.0,
                    max_value=100.0,
                    value=(10.0 if selected_strategy == "Quality Investing" else 0.0, 100.0),
                    step=1.0
                )
            else:
                min_net_margin, max_net_margin = -float('inf'), float('inf')

        with st.expander("Growth Filters", expanded=(selected_strategy == "Growth Investing")):
            # Revenue Growth
            revenue_growth_filter = st.checkbox("Filter by Revenue Growth",
                                                value=(selected_strategy == "Growth Investing"))
            if revenue_growth_filter:
                min_revenue_growth, max_revenue_growth = st.slider(
                    "Revenue Growth Range (%):",
                    min_value=-50.0,
                    max_value=200.0,
                    value=(15.0 if selected_strategy == "Growth Investing" else 0.0, 200.0),
                    step=1.0
                )
            else:
                min_revenue_growth, max_revenue_growth = -float('inf'), float('inf')

            # EPS Growth
            eps_growth_filter = st.checkbox("Filter by EPS Growth", value=(selected_strategy == "Growth Investing"))
            if eps_growth_filter:
                min_eps_growth, max_eps_growth = st.slider(
                    "EPS Growth Range (%):",
                    min_value=-50.0,
                    max_value=200.0,
                    value=(15.0 if selected_strategy == "Growth Investing" else 0.0, 200.0),
                    step=1.0
                )
            else:
                min_eps_growth, max_eps_growth = -float('inf'), float('inf')

        with st.expander("Dividend Filters", expanded=(selected_strategy == "Dividend Investing")):
            # Dividend Yield
            dividend_filter = st.checkbox("Filter by Dividend Yield", value=(selected_strategy == "Dividend Investing"))
            if dividend_filter:
                min_dividend, max_dividend = st.slider(
                    "Dividend Yield Range (%):",
                    min_value=0.0,
                    max_value=20.0,
                    value=(3.0 if selected_strategy == "Dividend Investing" else 0.0, 20.0),
                    step=0.1
                )
            else:
                min_dividend, max_dividend = 0.0, float('inf')

            # Dividend Payout Ratio
            payout_filter = st.checkbox("Filter by Payout Ratio", value=(selected_strategy == "Dividend Investing"))
            if payout_filter:
                min_payout, max_payout = st.slider(
                    "Payout Ratio Range (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 75.0 if selected_strategy == "Dividend Investing" else 100.0),
                    step=1.0
                )
            else:
                min_payout, max_payout = 0.0, float('inf')

        with st.expander("Financial Health Filters", expanded=(selected_strategy == "Quality Investing")):
            # Debt-to-Equity Ratio
            de_filter = st.checkbox("Filter by Debt-to-Equity", value=(selected_strategy == "Quality Investing"))
            if de_filter:
                min_de, max_de = st.slider(
                    "Debt-to-Equity Range:",
                    min_value=0.0,
                    max_value=10.0,
                    value=(0.0, 1.0 if selected_strategy == "Quality Investing" else 10.0),
                    step=0.1
                )
            else:
                min_de, max_de = 0.0, float('inf')

            # Current Ratio
            current_ratio_filter = st.checkbox("Filter by Current Ratio",
                                               value=(selected_strategy == "Quality Investing"))
            if current_ratio_filter:
                min_current_ratio, max_current_ratio = st.slider(
                    "Current Ratio Range:",
                    min_value=0.0,
                    max_value=10.0,
                    value=(1.5 if selected_strategy == "Quality Investing" else 0.0, 10.0),
                    step=0.1
                )
            else:
                min_current_ratio, max_current_ratio = 0.0, float('inf')

        with st.expander("Performance Filters", expanded=(selected_strategy == "Momentum")):
            # 1-Year Performance
            perf_1y_filter = st.checkbox("Filter by 1-Year Performance", value=(selected_strategy == "Momentum"))
            if perf_1y_filter:
                min_perf_1y, max_perf_1y = st.slider(
                    "1-Year Performance Range (%):",
                    min_value=-100.0,
                    max_value=500.0,
                    value=(20.0 if selected_strategy == "Momentum" else -100.0, 500.0),
                    step=5.0
                )
            else:
                min_perf_1y, max_perf_1y = -float('inf'), float('inf')

        # Submit button
        screen_button = st.button("Screen Stocks", type="primary")

    # Main content area
    if screen_button:
        with st.spinner("Screening stocks based on your criteria..."):
            try:
                # Get universe of stocks based on sector filter
                if selected_sector != "All":
                    if selected_subsector != "All":
                        universe = get_stocks_by_sector_and_subsector(selected_sector, selected_subsector)
                    else:
                        universe = get_stocks_by_sector(selected_sector)
                else:
                    # Get a broad universe of stocks
                    universe = get_stock_universe()

                if not universe:
                    st.error("Could not load stock universe. Please try again later.")
                    return

                # Apply market cap filter
                if market_cap_filter != "All":
                    min_market_cap, max_market_cap = market_cap_ranges[market_cap_filter]
                else:
                    min_market_cap, max_market_cap = 0, float('inf')

                # Screen stocks based on criteria
                screened_stocks = screen_stocks(
                    universe,
                    data_loader,
                    ratio_analyzer,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    min_price=min_price,
                    max_price=max_price,
                    min_pe=min_pe,
                    max_pe=max_pe,
                    min_pb=min_pb,
                    max_pb=max_pb,
                    min_ps=min_ps,
                    max_ps=max_ps,
                    min_ev_ebitda=min_ev_ebitda,
                    max_ev_ebitda=max_ev_ebitda,
                    min_roe=min_roe / 100 if roe_filter else -float('inf'),  # Convert from % to decimal
                    max_roe=max_roe / 100 if roe_filter else float('inf'),
                    min_net_margin=min_net_margin / 100 if net_margin_filter else -float('inf'),
                    max_net_margin=max_net_margin / 100 if net_margin_filter else float('inf'),
                    min_revenue_growth=min_revenue_growth / 100 if revenue_growth_filter else -float('inf'),
                    max_revenue_growth=max_revenue_growth / 100 if revenue_growth_filter else float('inf'),
                    min_eps_growth=min_eps_growth / 100 if eps_growth_filter else -float('inf'),
                    max_eps_growth=max_eps_growth / 100 if eps_growth_filter else float('inf'),
                    min_dividend_yield=min_dividend / 100 if dividend_filter else 0,
                    max_dividend_yield=max_dividend / 100 if dividend_filter else float('inf'),
                    min_payout_ratio=min_payout / 100 if payout_filter else 0,
                    max_payout_ratio=max_payout / 100 if payout_filter else float('inf'),
                    min_debt_equity=min_de if de_filter else 0,
                    max_debt_equity=max_de if de_filter else float('inf'),
                    min_current_ratio=min_current_ratio if current_ratio_filter else 0,
                    max_current_ratio=max_current_ratio if current_ratio_filter else float('inf'),
                    min_perf_1y=min_perf_1y / 100 if perf_1y_filter else -float('inf'),
                    max_perf_1y=max_perf_1y / 100 if perf_1y_filter else float('inf')
                )

                # Display results
                st.header("Screening Results")

                if not screened_stocks:
                    st.warning("No stocks match your screening criteria. Try relaxing some filters.")
                    return

                # Display count of results
                st.success(f"Found {len(screened_stocks)} stocks matching your criteria.")

                # Create tabs for different views
                tabs = st.tabs(["Summary Table", "Detailed View", "Sector Breakdown", "Performance Comparison"])

                # Tab 1: Summary Table
                with tabs[0]:
                    st.subheader("Matching Stocks")

                    # Create a summary table
                    summary_table = []

                    for stock in screened_stocks:
                        summary_table.append({
                            "Symbol": stock["symbol"],
                            "Company": stock["name"],
                            "Sector": stock["sector"],
                            "Price": f"${stock.get('price', 0):.2f}",
                            "Market Cap": f"${stock.get('market_cap', 0) / 1e9:.2f}B" if stock.get('market_cap',
                                                                                                   0) >= 1e9 else f"${stock.get('market_cap', 0) / 1e6:.2f}M",
                            "P/E": f"{stock.get('pe_ratio', None):.2f}" if stock.get('pe_ratio') is not None else "N/A",
                            "P/B": f"{stock.get('pb_ratio', None):.2f}" if stock.get('pb_ratio') is not None else "N/A",
                            "ROE": f"{stock.get('roe', None) * 100:.2f}%" if stock.get('roe') is not None else "N/A",
                            "Dividend": f"{stock.get('dividend_yield', None) * 100:.2f}%" if stock.get(
                                'dividend_yield') is not None else "N/A",
                        })

                    # Convert to DataFrame and display
                    summary_df = pd.DataFrame(summary_table)
                    st.dataframe(summary_df, use_container_width=True)

                    # Add export option
                    st.download_button(
                        label="Download Results as CSV",
                        data=summary_df.to_csv(index=False).encode('utf-8'),
                        file_name="stock_screening_results.csv",
                        mime="text/csv"
                    )

                # Tab 2: Detailed View
                with tabs[1]:
                    st.subheader("Detailed Stock Information")

                    # Add a selector for individual stocks
                    stock_symbols = [stock["symbol"] for stock in screened_stocks]
                    stock_names = [f"{stock['symbol']} - {stock['name']}" for stock in screened_stocks]

                    selected_stock = st.selectbox(
                        "Select a stock to view details:",
                        options=stock_names
                    )

                    selected_symbol = selected_stock.split(" - ")[0]

                    # Get the selected stock data
                    stock_data = next((stock for stock in screened_stocks if stock["symbol"] == selected_symbol), None)

                    if stock_data:
                        # Create columns for key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Price", f"${stock_data.get('price', 0):.2f}")

                        with col2:
                            market_cap = stock_data.get('market_cap', 0)
                            if market_cap >= 1e9:
                                market_cap_str = f"${market_cap / 1e9:.2f}B"
                            else:
                                market_cap_str = f"${market_cap / 1e6:.2f}M"
                            st.metric("Market Cap", market_cap_str)

                        with col3:
                            st.metric("Sector", stock_data.get("sector", "N/A"))

                        with col4:
                            st.metric("Industry", stock_data.get("industry", "N/A"))

                        # Create tabs for different aspects of the stock
                        stock_tabs = st.tabs(["Valuation", "Profitability", "Growth", "Financial Health", "Dividends"])

                        # Tab for valuation metrics
                        with stock_tabs[0]:
                            st.subheader("Valuation Metrics")

                            # Create a dictionary of valuation metrics
                            valuation_metrics = {
                                "P/E Ratio": stock_data.get("pe_ratio"),
                                "Forward P/E": stock_data.get("forward_pe"),
                                "P/B Ratio": stock_data.get("pb_ratio"),
                                "P/S Ratio": stock_data.get("ps_ratio"),
                                "EV/EBITDA": stock_data.get("ev_ebitda"),
                                "EV/Revenue": stock_data.get("ev_revenue"),
                                "PEG Ratio": stock_data.get("peg_ratio")
                            }

                            # Display metrics as a bar chart
                            valuation_df = pd.DataFrame(
                                {"Metric": list(valuation_metrics.keys()), "Value": list(valuation_metrics.values())}
                            )
                            valuation_df = valuation_df.dropna()

                            if not valuation_df.empty:
                                fig = visualizer.plot_metrics_bar_chart(
                                    valuation_df,
                                    title="Valuation Metrics",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No valuation metrics available.")

                        # Tab for profitability metrics
                        with stock_tabs[1]:
                            st.subheader("Profitability Metrics")

                            # Create a dictionary of profitability metrics
                            profitability_metrics = {
                                "Gross Margin": stock_data.get("gross_margin"),
                                "Operating Margin": stock_data.get("operating_margin"),
                                "Net Margin": stock_data.get("net_margin"),
                                "ROE": stock_data.get("roe"),
                                "ROA": stock_data.get("roa"),
                                "ROIC": stock_data.get("roic")
                            }

                            # Convert to percentages for display
                            profitability_df = pd.DataFrame(
                                {"Metric": list(profitability_metrics.keys()),
                                 "Value": list(profitability_metrics.values())}
                            )
                            profitability_df = profitability_df.dropna()

                            if not profitability_df.empty:
                                # Convert to percentage
                                profitability_df["Value"] = profitability_df["Value"] * 100

                                fig = visualizer.plot_metrics_bar_chart(
                                    profitability_df,
                                    title="Profitability Metrics (%)",
                                    height=400,
                                    is_percent=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No profitability metrics available.")

                        # Tab for growth metrics
                        with stock_tabs[2]:
                            st.subheader("Growth Metrics")

                            # Create a dictionary of growth metrics
                            growth_metrics = {
                                "Revenue Growth": stock_data.get("revenue_growth"),
                                "EPS Growth": stock_data.get("eps_growth"),
                                "Dividend Growth": stock_data.get("dividend_growth"),
                                "EBITDA Growth": stock_data.get("ebitda_growth"),
                                "Free Cash Flow Growth": stock_data.get("fcf_growth")
                            }

                            # Convert to percentages for display
                            growth_df = pd.DataFrame(
                                {"Metric": list(growth_metrics.keys()), "Value": list(growth_metrics.values())}
                            )
                            growth_df = growth_df.dropna()

                            if not growth_df.empty:
                                # Convert to percentage
                                growth_df["Value"] = growth_df["Value"] * 100

                                fig = visualizer.plot_metrics_bar_chart(
                                    growth_df,
                                    title="Growth Metrics (%)",
                                    height=400,
                                    is_percent=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No growth metrics available.")

                        # Tab for financial health metrics
                        with stock_tabs[3]:
                            st.subheader("Financial Health Metrics")

                            # Create a dictionary of financial health metrics
                            financial_health_metrics = {
                                "Current Ratio": stock_data.get("current_ratio"),
                                "Quick Ratio": stock_data.get("quick_ratio"),
                                "Debt-to-Equity": stock_data.get("debt_equity"),
                                "Interest Coverage": stock_data.get("interest_coverage"),
                                "Cash-to-Debt": stock_data.get("cash_to_debt")
                            }

                            # Display metrics
                            financial_health_df = pd.DataFrame(
                                {"Metric": list(financial_health_metrics.keys()),
                                 "Value": list(financial_health_metrics.values())}
                            )
                            financial_health_df = financial_health_df.dropna()

                            if not financial_health_df.empty:
                                fig = visualizer.plot_metrics_bar_chart(
                                    financial_health_df,
                                    title="Financial Health Metrics",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No financial health metrics available.")

                        # Tab for dividend metrics
                        with stock_tabs[4]:
                            st.subheader("Dividend Metrics")

                            # Create a dictionary of dividend metrics
                            dividend_metrics = {
                                "Dividend Yield": stock_data.get("dividend_yield"),
                                "Payout Ratio": stock_data.get("payout_ratio"),
                                "Dividend Growth Rate": stock_data.get("dividend_growth"),
                                "Years of Dividend Growth": stock_data.get("dividend_years")
                            }

                            # Convert percentage metrics
                            dividend_df = pd.DataFrame(
                                {"Metric": list(dividend_metrics.keys()), "Value": list(dividend_metrics.values())}
                            )
                            dividend_df = dividend_df.dropna()

                            if not dividend_df.empty:
                                # Convert appropriate metrics to percentage
                                for i, metric in enumerate(dividend_df["Metric"]):
                                    if metric in ["Dividend Yield", "Payout Ratio", "Dividend Growth Rate"]:
                                        if pd.notnull(dividend_df.loc[i, "Value"]):
                                            dividend_df.loc[i, "Value"] = dividend_df.loc[i, "Value"] * 100

                                # Create visualization
                                fig = visualizer.plot_metrics_bar_chart(
                                    dividend_df,
                                    title="Dividend Metrics",
                                    height=400,
                                    is_percent=False  # We'll handle percentage formatting manually
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # If the stock pays dividends, show more information
                                if stock_data.get("dividend_yield") is not None and stock_data.get(
                                        "dividend_yield") > 0:
                                    st.info(
                                        f"This stock pays a dividend yield of {stock_data.get('dividend_yield', 0) * 100:.2f}%. "
                                        f"Based on the current price, this represents an annual dividend of approximately "
                                        f"${stock_data.get('price', 0) * stock_data.get('dividend_yield', 0):.2f} per share."
                                    )
                                else:
                                    st.info(
                                        "This stock does not pay dividends or dividend information is not available.")
                            else:
                                st.warning("No dividend metrics available.")

                        # Add stock price chart
                        st.subheader("Price History")

                        # Get historical price data
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

                        try:
                            price_data = data_loader.get_historical_prices(
                                selected_symbol,
                                start_date,
                                end_date
                            )

                            if not price_data.empty:
                                fig = visualizer.plot_stock_price(
                                    price_data,
                                    selected_symbol,
                                    company_name=stock_data.get("name", ""),
                                    ma_periods=[50, 200],
                                    volume=True,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No price data available for {selected_symbol}.")
                        except Exception as e:
                            st.error(f"Error loading price data: {str(e)}")
                    else:
                        st.error(f"Could not find data for {selected_symbol}.")

                # Tab 3: Sector Breakdown
                with tabs[2]:
                    st.subheader("Sector Distribution")

                    # Calculate sector distribution
                    sector_counts = {}
                    for stock in screened_stocks:
                        sector = stock.get("sector", "Unknown")
                        if sector in sector_counts:
                            sector_counts[sector] += 1
                        else:
                            sector_counts[sector] = 1

                    # Create a DataFrame for the sector distribution
                    sector_df = pd.DataFrame({
                        "Sector": list(sector_counts.keys()),
                        "Count": list(sector_counts.values())
                    })

                    # Sort by count (descending)
                    sector_df = sector_df.sort_values("Count", ascending=False)

                    # Create a pie chart
                    fig = visualizer.plot_sector_distribution(
                        sector_df,
                        title="Sector Distribution of Screened Stocks",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display sector breakdown table
                    st.dataframe(sector_df, use_container_width=True)

                    # If we have more than one sector, display average metrics by sector
                    if len(sector_counts) > 1:
                        st.subheader("Average Metrics by Sector")

                        # Calculate average metrics by sector
                        sector_metrics = {}

                        for sector in sector_counts.keys():
                            sector_stocks = [stock for stock in screened_stocks if stock.get("sector") == sector]

                            # Calculate averages
                            avg_pe = np.mean(
                                [stock.get("pe_ratio") for stock in sector_stocks if stock.get("pe_ratio") is not None])
                            avg_pb = np.mean(
                                [stock.get("pb_ratio") for stock in sector_stocks if stock.get("pb_ratio") is not None])
                            avg_roe = np.mean(
                                [stock.get("roe") for stock in sector_stocks if stock.get("roe") is not None])
                            avg_div = np.mean([stock.get("dividend_yield") for stock in sector_stocks if
                                               stock.get("dividend_yield") is not None])

                            sector_metrics[sector] = {
                                "Avg P/E": avg_pe if not np.isnan(avg_pe) else None,
                                "Avg P/B": avg_pb if not np.isnan(avg_pb) else None,
                                "Avg ROE (%)": avg_roe * 100 if not np.isnan(avg_roe) else None,
                                "Avg Dividend (%)": avg_div * 100 if not np.isnan(avg_div) else None,
                                "Count": sector_counts[sector]
                            }

                        # Convert to DataFrame
                        sector_metrics_df = pd.DataFrame.from_dict(sector_metrics, orient='index')

                        # Format the DataFrame
                        for col in ["Avg P/E", "Avg P/B"]:
                            sector_metrics_df[col] = sector_metrics_df[col].apply(
                                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

                        for col in ["Avg ROE (%)", "Avg Dividend (%)"]:
                            sector_metrics_df[col] = sector_metrics_df[col].apply(
                                lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

                        # Display the DataFrame
                        st.dataframe(sector_metrics_df, use_container_width=True)

                    # Display stocks by sector
                    st.subheader("Stocks by Sector")

                    # Create expandable sections for each sector
                    for sector in sector_df["Sector"]:
                        with st.expander(f"{sector} ({sector_counts[sector]} stocks)"):
                            # Get stocks in this sector
                            sector_stocks = [stock for stock in screened_stocks if stock.get("sector") == sector]

                            # Create a summary table
                            sector_table = []

                            for stock in sector_stocks:
                                sector_table.append({
                                    "Symbol": stock["symbol"],
                                    "Company": stock["name"],
                                    "Price": f"${stock.get('price', 0):.2f}",
                                    "Market Cap": f"${stock.get('market_cap', 0) / 1e9:.2f}B" if stock.get('market_cap',
                                                                                                           0) >= 1e9 else f"${stock.get('market_cap', 0) / 1e6:.2f}M",
                                    "P/E": f"{stock.get('pe_ratio', None):.2f}" if stock.get(
                                        'pe_ratio') is not None else "N/A",
                                    "P/B": f"{stock.get('pb_ratio', None):.2f}" if stock.get(
                                        'pb_ratio') is not None else "N/A",
                                    "ROE": f"{stock.get('roe', None) * 100:.2f}%" if stock.get(
                                        'roe') is not None else "N/A",
                                    "Dividend": f"{stock.get('dividend_yield', None) * 100:.2f}%" if stock.get(
                                        'dividend_yield') is not None else "N/A",
                                })

                            # Convert to DataFrame and display
                            sector_df = pd.DataFrame(sector_table)
                            st.dataframe(sector_df, use_container_width=True)

                # Tab 4: Performance Comparison
                with tabs[3]:
                    st.subheader("Performance Comparison")

                    # Select a subset of stocks for comparison (top 10 by market cap)
                    top_stocks = sorted(
                        screened_stocks,
                        key=lambda x: x.get("market_cap", 0),
                        reverse=True
                    )[:10]

                    # Get historical price data for comparison
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

                    performance_data = {}
                    company_names = {}

                    for stock in top_stocks:
                        symbol = stock["symbol"]
                        try:
                            price_data = data_loader.get_historical_prices(symbol, start_date, end_date)

                            if not price_data.empty:
                                # Calculate normalized returns
                                first_price = price_data['Close'].iloc[0]
                                performance_data[symbol] = (price_data['Close'] / first_price - 1) * 100
                                company_names[symbol] = stock["name"]
                        except Exception as e:
                            logger.error(f"Error loading price data for {symbol}: {str(e)}")

                    # Create performance comparison chart
                    if performance_data:
                        fig = visualizer.plot_price_comparison(
                            performance_data,
                            company_names,
                            title="1-Year Price Performance Comparison",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate performance metrics
                        performance_metrics = []

                        for symbol, returns in performance_data.items():
                            # Calculate total return
                            total_return = returns.iloc[-1]

                            # Calculate volatility
                            daily_returns = returns.pct_change().dropna()
                            volatility = daily_returns.std() * (252 ** 0.5)

                            # Calculate max drawdown
                            cumulative_returns = (1 + returns / 100)
                            running_max = cumulative_returns.cummax()
                            drawdown = (cumulative_returns / running_max - 1) * 100
                            max_drawdown = drawdown.min()

                            performance_metrics.append({
                                "Symbol": symbol,
                                "Company": company_names.get(symbol, symbol),
                                "Total Return": f"{total_return:.2f}%",
                                "Volatility": f"{volatility:.2f}%",
                                "Max Drawdown": f"{max_drawdown:.2f}%",
                                "Sharpe Ratio": f"{(total_return / volatility):.2f}" if volatility != 0 else "N/A"
                            })

                        # Display performance metrics table
                        st.subheader("Performance Metrics")
                        perf_df = pd.DataFrame(performance_metrics)
                        st.dataframe(perf_df, use_container_width=True)
                    else:
                        st.warning("No performance data available for comparison.")

            except Exception as e:
                st.error(f"An error occurred during stock screening: {str(e)}")
                logger.error(f"Error in stock screening: {str(e)}")


def get_stock_universe():
    """Get a broad universe of stocks for screening"""
    # In a real implementation, this would retrieve a comprehensive list of stocks
    # For now, we'll return a sample list from major indices

    # This is a placeholder. In a production environment, you would:
    # 1. Load from a database or API
    # 2. Include thousands of stocks from major exchanges
    # 3. Update regularly

    # Sample implementation with a few stocks from different sectors
    return [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services"},
        {"symbol": "GOOG", "name": "Alphabet Inc.", "sector": "Communication Services"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary"},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc.", "sector": "Financials"},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
        {"symbol": "V", "name": "Visa Inc.", "sector": "Financials"},
        {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Staples"},
        {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
        {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
        {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Discretionary"},
        {"symbol": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
        {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financials"},
        {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
        {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
        {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
        {"symbol": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Staples"},
        {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services"},
        {"symbol": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Staples"},
        {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Staples"},
        {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Staples"},
        {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Financials"},
        {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
        {"symbol": "TMO", "name": "Thermo Fisher Scientific Inc.", "sector": "Healthcare"},
        {"symbol": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
        {"symbol": "LLY", "name": "Eli Lilly and Co.", "sector": "Healthcare"},
        {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
        {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
        {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
        {"symbol": "DHR", "name": "Danaher Corp.", "sector": "Healthcare"},
        {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technology"},
        {"symbol": "INTC", "name": "Intel Corp.", "sector": "Technology"},
        {"symbol": "VZ", "name": "Verizon Communications Inc.", "sector": "Communication Services"},
        {"symbol": "T", "name": "AT&T Inc.", "sector": "Communication Services"},
        {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consumer Discretionary"}
    ]


def get_stocks_by_sector(sector):
    """Get stocks filtered by sector"""
    universe = get_stock_universe()
    return [stock for stock in universe if stock["sector"] == sector]


def get_stocks_by_sector_and_subsector(sector, subsector):
    """Get stocks filtered by sector and subsector"""
    # In a real implementation, you would retrieve stocks by subsector
    # For now, we'll just return stocks by sector
    return get_stocks_by_sector(sector)


def screen_stocks(universe, data_loader, ratio_analyzer, **filters):
    """
    Screen stocks based on specified criteria

    Args:
        universe: List of stocks to screen
        data_loader: DataLoader instance
        ratio_analyzer: FinancialRatioAnalyzer instance
        **filters: Various filtering criteria

    Returns:
        List of stocks that meet the criteria
    """
    # Extract filter criteria
    min_market_cap = filters.get("min_market_cap", 0)
    max_market_cap = filters.get("max_market_cap", float('inf'))
    min_price = filters.get("min_price", 0)
    max_price = filters.get("max_price", float('inf'))
    min_pe = filters.get("min_pe", 0)
    max_pe = filters.get("max_pe", float('inf'))
    min_pb = filters.get("min_pb", 0)
    max_pb = filters.get("max_pb", float('inf'))
    min_ps = filters.get("min_ps", 0)
    max_ps = filters.get("max_ps", float('inf'))
    min_ev_ebitda = filters.get("min_ev_ebitda", 0)
    max_ev_ebitda = filters.get("max_ev_ebitda", float('inf'))
    min_roe = filters.get("min_roe", -float('inf'))
    max_roe = filters.get("max_roe", float('inf'))
    min_net_margin = filters.get("min_net_margin", -float('inf'))
    max_net_margin = filters.get("max_net_margin", float('inf'))
    min_revenue_growth = filters.get("min_revenue_growth", -float('inf'))
    max_revenue_growth = filters.get("max_revenue_growth", float('inf'))
    min_eps_growth = filters.get("min_eps_growth", -float('inf'))
    max_eps_growth = filters.get("max_eps_growth", float('inf'))
    min_dividend_yield = filters.get("min_dividend_yield", 0)
    max_dividend_yield = filters.get("max_dividend_yield", float('inf'))
    min_payout_ratio = filters.get("min_payout_ratio", 0)
    max_payout_ratio = filters.get("max_payout_ratio", float('inf'))
    min_debt_equity = filters.get("min_debt_equity", 0)
    max_debt_equity = filters.get("max_debt_equity", float('inf'))
    min_current_ratio = filters.get("min_current_ratio", 0)
    max_current_ratio = filters.get("max_current_ratio", float('inf'))
    min_perf_1y = filters.get("min_perf_1y", -float('inf'))
    max_perf_1y = filters.get("max_perf_1y", float('inf'))

    # Results list
    screened_stocks = []

    # Process each stock in the universe
    for stock in universe:
        symbol = stock["symbol"]

        try:
            # Get company info
            company_info = data_loader.get_company_info(symbol)

            if not company_info:
                continue

            # Get latest price and performance data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date_1y = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            price_data = data_loader.get_historical_prices(symbol, start_date_1y, end_date)

            if price_data.empty:
                continue

            # Get current price
            current_price = price_data['Close'].iloc[-1]

            # Calculate 1-year performance
            if len(price_data) > 20:  # Ensure we have enough data
                perf_1y = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0] - 1)
            else:
                perf_1y = None

            # Get financial statements
            income_stmt = data_loader.get_financial_statements(symbol, 'income', 'annual')
            balance_sheet = data_loader.get_financial_statements(symbol, 'balance', 'annual')
            cash_flow = data_loader.get_financial_statements(symbol, 'cash', 'annual')

            # Combine into financial data dict
            financial_data = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'market_data': {
                    'share_price': current_price,
                    'market_cap': company_info.get('market_cap')
                }
            }

            # Calculate financial ratios
            ratios = ratio_analyzer.calculate_ratios(financial_data)

            # Prepare stock data with all metrics
            stock_data = {
                "symbol": symbol,
                "name": company_info.get('name', stock.get('name', symbol)),
                "sector": company_info.get('sector', stock.get('sector', 'Unknown')),
                "industry": company_info.get('industry', 'Unknown'),
                "price": current_price,
                "market_cap": company_info.get('market_cap'),
                "perf_1y": perf_1y,

                # Valuation ratios
                "pe_ratio": ratios.get('valuation', {}).get('pe_ratio'),
                "forward_pe": ratios.get('valuation', {}).get('forward_pe'),
                "pb_ratio": ratios.get('valuation', {}).get('pb_ratio'),
                "ps_ratio": ratios.get('valuation', {}).get('ps_ratio'),
                "ev_ebitda": ratios.get('valuation', {}).get('ev_ebitda'),
                "ev_revenue": ratios.get('valuation', {}).get('ev_revenue'),
                "peg_ratio": ratios.get('valuation', {}).get('peg_ratio'),

                # Profitability ratios
                "gross_margin": ratios.get('profitability', {}).get('gross_margin'),
                "operating_margin": ratios.get('profitability', {}).get('operating_margin'),
                "net_margin": ratios.get('profitability', {}).get('net_margin'),
                "roe": ratios.get('profitability', {}).get('roe'),
                "roa": ratios.get('profitability', {}).get('roa'),
                "roic": ratios.get('profitability', {}).get('roic'),

                # Growth ratios
                "revenue_growth": ratios.get('growth', {}).get('revenue_growth'),
                "eps_growth": ratios.get('growth', {}).get('net_income_growth'),  # Using net income growth as proxy
                "dividend_growth": ratios.get('growth', {}).get('dividend_growth'),
                "ebitda_growth": ratios.get('growth', {}).get('ebitda_growth'),
                "fcf_growth": ratios.get('growth', {}).get('fcf_growth'),

                # Liquidity ratios
                "current_ratio": ratios.get('liquidity', {}).get('current_ratio'),
                "quick_ratio": ratios.get('liquidity', {}).get('quick_ratio'),
                "cash_ratio": ratios.get('liquidity', {}).get('cash_ratio'),

                # Leverage ratios
                "debt_equity": ratios.get('leverage', {}).get('debt_to_equity'),
                "interest_coverage": ratios.get('leverage', {}).get('interest_coverage'),
                "debt_to_assets": ratios.get('leverage', {}).get('debt_to_assets'),
                "cash_to_debt": ratios.get('leverage', {}).get('cash_to_debt'),

                # Dividend metrics
                "dividend_yield": company_info.get('dividend_yield'),
                "payout_ratio": ratios.get('valuation', {}).get('payout_ratio'),
                "dividend_years": None  # Would need to be obtained from additional sources
            }

            # Apply filters
            if (
                    (stock_data.get("market_cap") is None or (
                            min_market_cap <= stock_data["market_cap"] <= max_market_cap)) and
                    (min_price <= stock_data["price"] <= max_price) and
                    (stock_data.get("pe_ratio") is None or np.isnan(stock_data.get("pe_ratio", np.nan)) or (
                            min_pe <= stock_data["pe_ratio"] <= max_pe)) and
                    (stock_data.get("pb_ratio") is None or np.isnan(stock_data.get("pb_ratio", np.nan)) or (
                            min_pb <= stock_data["pb_ratio"] <= max_pb)) and
                    (stock_data.get("ps_ratio") is None or np.isnan(stock_data.get("ps_ratio", np.nan)) or (
                            min_ps <= stock_data["ps_ratio"] <= max_ps)) and
                    (stock_data.get("ev_ebitda") is None or np.isnan(stock_data.get("ev_ebitda", np.nan)) or (
                            min_ev_ebitda <= stock_data["ev_ebitda"] <= max_ev_ebitda)) and
                    (stock_data.get("roe") is None or np.isnan(stock_data.get("roe", np.nan)) or (
                            min_roe <= stock_data["roe"] <= max_roe)) and
                    (stock_data.get("net_margin") is None or np.isnan(stock_data.get("net_margin", np.nan)) or (
                            min_net_margin <= stock_data["net_margin"] <= max_net_margin)) and
                    (stock_data.get("revenue_growth") is None or np.isnan(stock_data.get("revenue_growth", np.nan)) or (
                            min_revenue_growth <= stock_data["revenue_growth"] <= max_revenue_growth)) and
                    (stock_data.get("eps_growth") is None or np.isnan(stock_data.get("eps_growth", np.nan)) or (
                            min_eps_growth <= stock_data["eps_growth"] <= max_eps_growth)) and
                    (stock_data.get("dividend_yield") is None or np.isnan(stock_data.get("dividend_yield", np.nan)) or (
                            min_dividend_yield <= stock_data["dividend_yield"] <= max_dividend_yield)) and
                    (stock_data.get("payout_ratio") is None or np.isnan(stock_data.get("payout_ratio", np.nan)) or (
                            min_payout_ratio <= stock_data["payout_ratio"] <= max_payout_ratio)) and
                    (stock_data.get("debt_equity") is None or np.isnan(stock_data.get("debt_equity", np.nan)) or (
                            min_debt_equity <= stock_data["debt_equity"] <= max_debt_equity)) and
                    (stock_data.get("current_ratio") is None or np.isnan(stock_data.get("current_ratio", np.nan)) or (
                            min_current_ratio <= stock_data["current_ratio"] <= max_current_ratio)) and
                    (stock_data.get("perf_1y") is None or np.isnan(stock_data.get("perf_1y", np.nan)) or (
                            min_perf_1y <= stock_data["perf_1y"] <= max_perf_1y))
            ):
                # Stock meets all criteria, add to results
                screened_stocks.append(stock_data)

        except Exception as e:
            logger.error(f"Error processing stock {symbol}: {str(e)}")

    return screened_stocks


# For direct execution
if __name__ == "__main__":
    run_stock_screener_page()