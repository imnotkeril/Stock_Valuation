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
from StockAnalysisSystem.src.config import UI_SETTINGS, COLORS, VIZ_SETTINGS, SECTOR_MAPPING, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
from StockAnalysisSystem.src.pages.screening.stock_screener import get_stock_universe, get_stocks_by_sector, get_stocks_by_sector_and_subsector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('valuation_filter')


def run_valuation_filter_page():
    """Main function to run the valuation-based stock filter page"""
    st.title("Valuation-Based Stock Filter")
    st.markdown(
        "Screen stocks based on intrinsic value and relative valuation metrics to find potentially undervalued opportunities.")

    # Initialize data loader, analyzers and visualizer
    data_loader = DataLoader()
    visualizer = FinancialVisualizer(theme="dark")
    ratio_analyzer = FinancialRatioAnalyzer()
    valuation_factory = ValuationFactory(data_loader)

    # Sidebar for filter controls
    with st.sidebar:
        st.header("Valuation Filters")

        # Valuation approaches
        st.subheader("Valuation Approach")
        valuation_approaches = {
            "DCF": "Discounted Cash Flow - based on future cash flows",
            "Relative": "Relative Valuation - based on peer multiples",
            "Combined": "Combined approach using both DCF and multiples"
        }

        selected_approach = st.selectbox(
            "Select valuation approach:",
            options=list(valuation_approaches.keys()),
            index=2  # Default to Combined
        )

        st.info(valuation_approaches[selected_approach])

        # Universe selector
        st.subheader("Stock Universe")

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

        # Market cap filter
        st.subheader("Market Cap Filter")
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

        # DCF parameters expander (if DCF or Combined approach)
        if selected_approach in ["DCF", "Combined"]:
            with st.expander("DCF Parameters", expanded=True):
                # Discount rate adjustment
                discount_rate_adj = st.slider(
                    "Discount Rate Adjustment (%):",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.5,
                    help="Adjust the discount rate used in DCF calculations"
                )

                # Growth rate adjustment
                growth_rate_adj = st.slider(
                    "Growth Rate Adjustment (%):",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.5,
                    help="Adjust the growth rate used in DCF calculations"
                )

                # Terminal growth rate adjustment
                terminal_growth_adj = st.slider(
                    "Terminal Growth Adjustment (%):",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.2,
                    help="Adjust the terminal growth rate used in DCF calculations"
                )

                # Margin of safety
                margin_of_safety = st.slider(
                    "Margin of Safety (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=25.0,
                    step=5.0,
                    help="Required discount to fair value for considering a stock undervalued"
                )

        # Relative valuation parameters expander (if Relative or Combined approach)
        if selected_approach in ["Relative", "Combined"]:
            with st.expander("Relative Valuation Parameters", expanded=True):
                # P/E discount
                pe_discount = st.slider(
                    "P/E Discount (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=15.0,
                    step=5.0,
                    help="Required discount to sector average P/E"
                )

                # P/S discount
                ps_discount = st.slider(
                    "P/S Discount (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=15.0,
                    step=5.0,
                    help="Required discount to sector average P/S"
                )

                # P/B discount
                pb_discount = st.slider(
                    "P/B Discount (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=15.0,
                    step=5.0,
                    help="Required discount to sector average P/B"
                )

                # EV/EBITDA discount
                ev_ebitda_discount = st.slider(
                    "EV/EBITDA Discount (%):",
                    min_value=0.0,
                    max_value=50.0,
                    value=15.0,
                    step=5.0,
                    help="Required discount to sector average EV/EBITDA"
                )

        # Results sorting
        st.subheader("Results Sorting")
        sort_options = {
            "Upside Potential": "Sort by upside to fair value (descending)",
            "Market Cap": "Sort by market capitalization (descending)",
            "P/E Ratio": "Sort by P/E ratio (ascending)",
            "Dividend Yield": "Sort by dividend yield (descending)"
        }

        sort_by = st.selectbox(
            "Sort results by:",
            options=list(sort_options.keys()),
            index=0
        )

        st.info(sort_options[sort_by])

        # Max number of results
        max_results = st.slider(
            "Maximum number of results:",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

        # Run valuation button
        run_valuation_button = st.button("Run Valuation Analysis", type="primary")

    # Main content area
    if run_valuation_button:
        with st.spinner("Performing valuation analysis on selected stock universe..."):
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

                # Perform valuation analysis
                valuation_results = perform_valuation_analysis(
                    universe,
                    data_loader,
                    ratio_analyzer,
                    valuation_factory,
                    selected_approach,
                    selected_sector,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    discount_rate_adj=discount_rate_adj / 100 if selected_approach in ["DCF", "Combined"] else 0,
                    growth_rate_adj=growth_rate_adj / 100 if selected_approach in ["DCF", "Combined"] else 0,
                    terminal_growth_adj=terminal_growth_adj / 100 if selected_approach in ["DCF", "Combined"] else 0,
                    margin_of_safety=margin_of_safety / 100 if selected_approach in ["DCF", "Combined"] else 0.25,
                    pe_discount=pe_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    ps_discount=ps_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    pb_discount=pb_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    ev_ebitda_discount=ev_ebitda_discount / 100 if selected_approach in ["Relative",
                                                                                         "Combined"] else 0.15
                )

                # Sort the results
                if sort_by == "Upside Potential":
                    valuation_results = sorted(valuation_results, key=lambda x: x.get("upside", -float('inf')),
                                               reverse=True)
                elif sort_by == "Market Cap":
                    valuation_results = sorted(valuation_results, key=lambda x: x.get("market_cap", 0), reverse=True)
                elif sort_by == "P/E Ratio":
                    # Sort by P/E, but put None values at the end
                    valuation_results = sorted(
                        valuation_results,
                        key=lambda x: (x.get("pe_ratio") is None, x.get("pe_ratio", float('inf')))
                    )
                elif sort_by == "Dividend Yield":
                    # Sort by dividend yield, but put None values at the end
                    valuation_results = sorted(
                        valuation_results,
                        key=lambda x: (x.get("dividend_yield") is None, -x.get("dividend_yield", 0))
                    )

                # Limit results to max_results
                valuation_results = valuation_results[:max_results]

                # Display results
                st.header("Valuation Analysis Results")

                if not valuation_results:
                    st.warning("No stocks meet the valuation criteria. Try adjusting your parameters.")
                    return

                # Display count of results
                st.success(f"Found {len(valuation_results)} potentially undervalued stocks.")

                # Create tabs for different views
                tabs = st.tabs(["Summary Table", "Detailed Analysis", "Sector Breakdown", "Fair Value Distribution"])

                # Tab 1: Summary Table
                with tabs[0]:
                    st.subheader("Potentially Undervalued Stocks")

                    # Create a summary table
                    summary_table = []

                    for stock in valuation_results:
                        summary_table.append({
                            "Symbol": stock["symbol"],
                            "Company": stock["name"],
                            "Sector": stock["sector"],
                            "Current Price": f"${stock.get('current_price', 0):.2f}",
                            "Fair Value": f"${stock.get('fair_value', 0):.2f}",
                            "Upside": f"{stock.get('upside', 0) * 100:.2f}%",
                            "Method": stock.get("valuation_method", "N/A"),
                            "Market Cap": f"${stock.get('market_cap', 0) / 1e9:.2f}B" if stock.get('market_cap',
                                                                                                   0) >= 1e9 else f"${stock.get('market_cap', 0) / 1e6:.2f}M",
                            "P/E": f"{stock.get('pe_ratio', None):.2f}" if stock.get('pe_ratio') is not None else "N/A",
                            "P/B": f"{stock.get('pb_ratio', None):.2f}" if stock.get('pb_ratio') is not None else "N/A",
                        })

                    # Convert to DataFrame and display
                    summary_df = pd.DataFrame(summary_table)

                    # Define a function to color the upside column
                    def color_upside(val):
                        """Color positive upside green, negative red"""
                        try:
                            value = float(val.strip('%'))
                            if value >= 30:
                                return f'background-color: {COLORS["primary"]}; color: #121212'  # Green background with dark text
                            elif value >= 15:
                                return f'background-color: {COLORS["success"]}; color: #121212'  # Teal background with dark text
                            elif value >= 0:
                                return f'background-color: {COLORS["warning"]}; color: #121212'  # Yellow background with dark text
                            else:
                                return f'background-color: {COLORS["accent"]}; color: #121212'  # Red background with dark text
                        except:
                            return ''

                    # Apply styling
                    styled_df = summary_df.style.applymap(color_upside, subset=['Upside'])

                    st.dataframe(styled_df, use_container_width=True)

                    # Add export option
                    st.download_button(
                        label="Download Results as CSV",
                        data=summary_df.to_csv(index=False).encode('utf-8'),
                        file_name="valuation_analysis_results.csv",
                        mime="text/csv"
                    )

                # Tab 2: Detailed Analysis
                with tabs[1]:
                    st.subheader("Detailed Valuation Analysis")

                    # Add a selector for individual stocks
                    stock_symbols = [stock["symbol"] for stock in valuation_results]
                    stock_names = [f"{stock['symbol']} - {stock['name']}" for stock in valuation_results]

                    selected_stock = st.selectbox(
                        "Select a stock to view detailed analysis:",
                        options=stock_names
                    )

                    selected_symbol = selected_stock.split(" - ")[0]

                    # Get the selected stock data
                    stock_data = next((stock for stock in valuation_results if stock["symbol"] == selected_symbol),
                                      None)

                    if stock_data:
                        # Create columns for key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Current Price",
                                f"${stock_data.get('current_price', 0):.2f}"
                            )

                        with col2:
                            st.metric(
                                "Fair Value",
                                f"${stock_data.get('fair_value', 0):.2f}",
                                f"{stock_data.get('upside', 0) * 100:.2f}% Upside"
                            )

                        with col3:
                            st.metric(
                                "Valuation Method",
                                stock_data.get("valuation_method", "N/A")
                            )

                        with col4:
                            st.metric(
                                "Confidence",
                                stock_data.get("confidence_level", "Medium")
                            )

                        # Display valuation details
                        st.subheader("Valuation Breakdown")

                        # Create a chart showing fair value breakdown
                        if "valuation_components" in stock_data:
                            components = stock_data["valuation_components"]

                            # Convert to DataFrame
                            components_df = pd.DataFrame({
                                "Method": list(components.keys()),
                                "Value": list(components.values())
                            })

                            # Create a bar chart
                            fig = visualizer.plot_valuation_breakdown(
                                components_df,
                                current_price=stock_data.get('current_price', 0),
                                title=f"Valuation Breakdown for {selected_symbol}",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Display valuation parameters
                        if "valuation_parameters" in stock_data:
                            st.subheader("Valuation Parameters")

                            params = stock_data["valuation_parameters"]

                            # Create columns for key parameters
                            col1, col2, col3 = st.columns(3)

                            # DCF parameters
                            if "discount_rate" in params:
                                with col1:
                                    st.metric("Discount Rate", f"{params.get('discount_rate', 0) * 100:.2f}%")

                                with col2:
                                    st.metric("Growth Rate", f"{params.get('growth_rate', 0) * 100:.2f}%")

                                with col3:
                                    st.metric("Terminal Growth", f"{params.get('terminal_growth', 0) * 100:.2f}%")

                            # Relative valuation parameters
                            if "peer_pe" in params:
                                with col1:
                                    st.metric("Peer P/E Ratio", f"{params.get('peer_pe', 0):.2f}x")

                                with col2:
                                    st.metric("Peer P/B Ratio", f"{params.get('peer_pb', 0):.2f}x")

                                with col3:
                                    st.metric("Peer P/S Ratio", f"{params.get('peer_ps', 0):.2f}x")

                        # Display financial metrics
                        st.subheader("Key Financial Metrics")

                        # Create columns for financial metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("P/E Ratio", f"{stock_data.get('pe_ratio', None):.2f}x" if stock_data.get(
                                'pe_ratio') is not None else "N/A")

                        with col2:
                            st.metric("P/B Ratio", f"{stock_data.get('pb_ratio', None):.2f}x" if stock_data.get(
                                'pb_ratio') is not None else "N/A")

                        with col3:
                            st.metric("P/S Ratio", f"{stock_data.get('ps_ratio', None):.2f}x" if stock_data.get(
                                'ps_ratio') is not None else "N/A")

                        with col4:
                            st.metric("EV/EBITDA", f"{stock_data.get('ev_ebitda', None):.2f}x" if stock_data.get(
                                'ev_ebitda') is not None else "N/A")

                        # Add more rows of metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("ROE", f"{stock_data.get('roe', None) * 100:.2f}%" if stock_data.get(
                                'roe') is not None else "N/A")

                        with col2:
                            st.metric("Net Margin",
                                      f"{stock_data.get('net_margin', None) * 100:.2f}%" if stock_data.get(
                                          'net_margin') is not None else "N/A")

                        with col3:
                            st.metric("Debt/Equity", f"{stock_data.get('debt_equity', None):.2f}x" if stock_data.get(
                                'debt_equity') is not None else "N/A")

                        with col4:
                            st.metric("Dividend Yield",
                                      f"{stock_data.get('dividend_yield', None) * 100:.2f}%" if stock_data.get(
                                          'dividend_yield') is not None else "N/A")

                        # Display notes and investment thesis
                        if "notes" in stock_data:
                            st.subheader("Analysis Notes")
                            st.write(stock_data["notes"])

                        # Display price chart
                        st.subheader("Price History vs. Fair Value")

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
                                fig = visualizer.plot_price_vs_value(
                                    price_data,
                                    selected_symbol,
                                    fair_value=stock_data.get('fair_value', 0),
                                    company_name=stock_data.get("name", ""),
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
                    for stock in valuation_results:
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
                        title="Sector Distribution of Undervalued Stocks",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display sector breakdown with average upside
                    st.subheader("Average Upside by Sector")

                    # Calculate average upside by sector
                    sector_upside = {}
                    for stock in valuation_results:
                        sector = stock.get("sector", "Unknown")
                        upside = stock.get("upside", 0)

                        if sector in sector_upside:
                            sector_upside[sector]["total_upside"] += upside
                            sector_upside[sector]["count"] += 1
                        else:
                            sector_upside[sector] = {"total_upside": upside, "count": 1}

                    # Calculate averages
                    sector_avg_upside = []
                    for sector, data in sector_upside.items():
                        avg_upside = data["total_upside"] / data["count"]
                        sector_avg_upside.append({
                            "Sector": sector,
                            "Average Upside": f"{avg_upside * 100:.2f}%",
                            "Number of Stocks": data["count"]
                        })

                    # Convert to DataFrame and sort by average upside
                    upside_df = pd.DataFrame(sector_avg_upside)
                    upside_df = upside_df.sort_values("Average Upside", ascending=False)

                    # Display the DataFrame
                    st.dataframe(upside_df, use_container_width=True)

                    # Display stocks by sector
                    st.subheader("Undervalued Stocks by Sector")

                    # Create expandable sections for each sector
                    for sector in sector_df["Sector"]:
                        with st.expander(f"{sector} ({sector_counts[sector]} stocks)"):
                            # Get stocks in this sector
                            sector_stocks = [stock for stock in valuation_results if stock.get("sector") == sector]

                            # Create a summary table
                            sector_table = []

                            for stock in sector_stocks:
                                sector_table.append({
                                    "Symbol": stock["symbol"],
                                    "Company": stock["name"],
                                    "Current Price": f"${stock.get('current_price', 0):.2f}",
                                    "Fair Value": f"${stock.get('fair_value', 0):.2f}",
                                    "Upside": f"{stock.get('upside', 0) * 100:.2f}%",
                                    "Method": stock.get("valuation_method", "N/A"),
                                    "P/E": f"{stock.get('pe_ratio', None):.2f}" if stock.get(
                                        'pe_ratio') is not None else "N/A"
                                })

                            # Convert to DataFrame and display
                            sector_df = pd.DataFrame(sector_table)

                            # Apply upside coloring
                            styled_df = sector_df.style.applymap(color_upside, subset=['Upside'])

                            st.dataframe(styled_df, use_container_width=True)

                # Tab 4: Fair Value Distribution
                with tabs[3]:
                    st.subheader("Fair Value Distribution")

                    # Create a histogram of upside potential
                    upside_data = [stock.get("upside", 0) * 100 for stock in valuation_results]

                    # Create histogram figure
                    fig = visualizer.plot_upside_histogram(
                        upside_data,
                        title="Distribution of Upside Potential (%)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics
                    mean_upside = np.mean(upside_data)
                    median_upside = np.median(upside_data)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Mean Upside", f"{mean_upside:.2f}%")

                    with col2:
                        st.metric("Median Upside", f"{median_upside:.2f}%")

                    with col3:
                        st.metric(
                            "Stocks with >25% Upside",
                            f"{len([u for u in upside_data if u > 25])} ({len([u for u in upside_data if u > 25]) / len(upside_data) * 100:.2f}%)"
                        )

                    # Display valuation method distribution
                    st.subheader("Valuation Method Distribution")

                    # Count valuation methods
                    method_counts = {}
                    for stock in valuation_results:
                        method = stock.get("valuation_method", "Unknown")
                        if method in method_counts:
                            method_counts[method] += 1
                        else:
                            method_counts[method] = 1

                    # Create a DataFrame
                    method_df = pd.DataFrame({
                        "Method": list(method_counts.keys()),
                        "Count": list(method_counts.values())
                    })

                    # Create a pie chart
                    fig = visualizer.plot_method_distribution(
                        method_df,
                        title="Valuation Method Distribution",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during valuation analysis: {str(e)}")
                logger.error(f"Error in valuation analysis: {str(e)}")


def perform_valuation_analysis(universe, data_loader, ratio_analyzer, valuation_factory, approach, sector, **params):
    """
    Perform valuation analysis on a universe of stocks

    Args:
        universe: List of stocks to analyze
        data_loader: DataLoader instance
        ratio_analyzer: FinancialRatioAnalyzer instance
        valuation_factory: ValuationFactory instance
        approach: Valuation approach to use ("DCF", "Relative", or "Combined")
        sector: Selected sector (or "All")
        **params: Various valuation parameters

    Returns:
        List of stocks that meet the valuation criteria
    """
    # Extract parameters
    min_market_cap = params.get("min_market_cap", 0)
    max_market_cap = params.get("max_market_cap", float('inf'))
    discount_rate_adj = params.get("discount_rate_adj", 0)
    growth_rate_adj = params.get("growth_rate_adj", 0)
    terminal_growth_adj = params.get("terminal_growth_adj", 0)
    margin_of_safety = params.get("margin_of_safety", 0.25)
    pe_discount = params.get("pe_discount", 0.15)
    ps_discount = params.get("ps_discount", 0.15)
    pb_discount = params.get("pb_discount", 0.15)
    ev_ebitda_discount = params.get("ev_ebitda_discount", 0.15)

    # Results list
    valuation_results = []

    # Get sector benchmarks for relative valuation
    sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector if sector != "All" else None)

    # Adjust DCF parameters based on user inputs
    dcf_params = {}
    if sector != "All" and sector in SECTOR_DCF_PARAMETERS:
        dcf_params = SECTOR_DCF_PARAMETERS[sector].copy()

    if discount_rate_adj != 0:
        if "default_discount_rate" in dcf_params:
            dcf_params["default_discount_rate"] += discount_rate_adj
        else:
            dcf_params["default_discount_rate"] = 0.10 + discount_rate_adj

    if growth_rate_adj != 0:
        # Apply to terminal growth rate
        if "terminal_growth_rate" in dcf_params:
            dcf_params["terminal_growth_rate"] += terminal_growth_adj
        else:
            dcf_params["terminal_growth_rate"] = 0.02 + terminal_growth_adj

    # Process each stock in the universe
    for stock in universe:
        symbol = stock["symbol"]

        try:
            # Get company info
            company_info = data_loader.get_company_info(symbol)

            if not company_info:
                continue

            # Market cap filter
            market_cap = company_info.get('market_cap')
            if market_cap is None or not (min_market_cap <= market_cap <= max_market_cap):
                continue

            # Get latest price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date_1y = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            price_data = data_loader.get_historical_prices(symbol, start_date_1y, end_date)

            if price_data.empty:
                continue

            # Get current price
            current_price = price_data['Close'].iloc[-1]

            # Get financial statements
            income_stmt = data_loader.get_financial_statements(symbol, 'income', 'annual')
            balance_sheet = data_loader.get_financial_statements(symbol, 'balance', 'annual')
            cash_flow = data_loader.get_financial_statements(symbol, 'cash', 'annual')

            # Check if we have enough data for valuation
            if income_stmt.empty or balance_sheet.empty:
                continue

            # Combine into financial data dict
            financial_data = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'market_data': {
                    'share_price': current_price,
                    'market_cap': market_cap
                }
            }

            # Calculate ratios
            ratios = ratio_analyzer.calculate_ratios(financial_data)

            # Perform DCF valuation if needed
            dcf_result = None
            if approach in ["DCF", "Combined"]:
                try:
                    # Get company sector
                    stock_sector = company_info.get('sector', sector if sector != "All" else None)

                    # Perform DCF valuation with adjusted parameters
                    dcf_result = valuation_factory.get_company_valuation(
                        symbol,
                        sector=stock_sector,
                        **dcf_params
                    )
                except Exception as e:
                    logger.warning(f"DCF valuation failed for {symbol}: {str(e)}")

            # Perform relative valuation if needed
            relative_result = None
            if approach in ["Relative", "Combined"]:
                try:
                    # Get sector avg multiples
                    stock_sector = company_info.get('sector', sector if sector != "All" else None)
                    sector_multiples = sector_benchmarks.get(stock_sector, {}).get('valuation', {})

                    # Calculate relative valuation
                    relative_result = calculate_relative_valuation(
                        symbol,
                        company_info,
                        ratios,
                        sector_multiples,
                        current_price
                    )
                except Exception as e:
                    logger.warning(f"Relative valuation failed for {symbol}: {str(e)}")

            # Combine results based on approach
            fair_value = None
            valuation_method = None
            valuation_components = {}
            valuation_parameters = {}
            confidence_level = "Low"
            notes = ""

            if approach == "DCF" and dcf_result:
                fair_value = dcf_result.get('value_per_share')
                valuation_method = "DCF"
                valuation_components = {"DCF": fair_value}
                valuation_parameters = {
                    "discount_rate": dcf_result.get('discount_rate'),
                    "growth_rate": dcf_result.get('growth_rate'),
                    "terminal_growth": dcf_result.get('terminal_growth')
                }
                confidence_level = "Medium"
                notes = "Based on discounted cash flow analysis."

            elif approach == "Relative" and relative_result:
                fair_value = relative_result.get('fair_value')
                valuation_method = "Relative"
                valuation_components = relative_result.get('components', {})
                valuation_parameters = relative_result.get('parameters', {})
                confidence_level = "Medium"
                notes = "Based on sector average multiples."

            elif approach == "Combined":
                # Use both methods if available, otherwise fall back to the one that worked
                if dcf_result and relative_result:
                    # Weighted average (60% DCF, 40% relative)
                    dcf_value = dcf_result.get('value_per_share', 0)
                    rel_value = relative_result.get('fair_value', 0)

                    if dcf_value and rel_value:
                        fair_value = dcf_value * 0.6 + rel_value * 0.4
                        valuation_method = "Combined"
                        valuation_components = {
                            "DCF (60%)": dcf_value,
                            "Relative (40%)": rel_value
                        }
                        valuation_parameters = {
                            "discount_rate": dcf_result.get('discount_rate'),
                            "growth_rate": dcf_result.get('growth_rate'),
                            "terminal_growth": dcf_result.get('terminal_growth'),
                            "peer_pe": relative_result.get('parameters', {}).get('peer_pe'),
                            "peer_pb": relative_result.get('parameters', {}).get('peer_pb'),
                            "peer_ps": relative_result.get('parameters', {}).get('peer_ps')
                        }
                        confidence_level = "High"
                        notes = "Based on combined DCF and relative valuation methods."
                elif dcf_result:
                    fair_value = dcf_result.get('value_per_share')
                    valuation_method = "DCF"
                    valuation_components = {"DCF": fair_value}
                    valuation_parameters = {
                        "discount_rate": dcf_result.get('discount_rate'),
                        "growth_rate": dcf_result.get('growth_rate'),
                        "terminal_growth": dcf_result.get('terminal_growth')
                    }
                    confidence_level = "Medium"
                    notes = "Based on discounted cash flow analysis. Relative valuation not available."
                elif relative_result:
                    fair_value = relative_result.get('fair_value')
                    valuation_method = "Relative"
                    valuation_components = relative_result.get('components', {})
                    valuation_parameters = relative_result.get('parameters', {})
                    confidence_level = "Medium"
                    notes = "Based on sector average multiples. DCF valuation not available."

            # Calculate upside
            upside = (fair_value / current_price - 1) if fair_value and current_price > 0 else None

            # Apply margin of safety
            if upside is not None and upside > margin_of_safety:
                # Stock is potentially undervalued, add to results
                valuation_results.append({
                    "symbol": symbol,
                    "name": company_info.get('name', stock.get('name', symbol)),
                    "sector": company_info.get('sector', stock.get('sector', 'Unknown')),
                    "industry": company_info.get('industry', 'Unknown'),
                    "current_price": current_price,
                    "fair_value": fair_value,
                    "upside": upside,
                    "valuation_method": valuation_method,
                    "valuation_components": valuation_components,
                    "valuation_parameters": valuation_parameters,
                    "confidence_level": confidence_level,
                    "notes": notes,
                    "market_cap": market_cap,

                    # Add ratios
                    "pe_ratio": ratios.get('valuation', {}).get('pe_ratio'),
                    "pb_ratio": ratios.get('valuation', {}).get('pb_ratio'),
                    "ps_ratio": ratios.get('valuation', {}).get('ps_ratio'),
                    "ev_ebitda": ratios.get('valuation', {}).get('ev_ebitda'),
                    "roe": ratios.get('profitability', {}).get('roe'),
                    "net_margin": ratios.get('profitability', {}).get('net_margin'),
                    "debt_equity": ratios.get('leverage', {}).get('debt_to_equity'),
                    "dividend_yield": company_info.get('dividend_yield')
                })

        except Exception as e:
            logger.error(f"Error processing stock {symbol}: {str(e)}")

    return valuation_results


def calculate_relative_valuation(symbol, company_info, ratios, sector_multiples, current_price):
    """
    Calculate fair value using relative valuation methods

    Args:
        symbol: Stock symbol
        company_info: Company information
        ratios: Company financial ratios
        sector_multiples: Sector average multiples
        current_price: Current stock price

    Returns:
        Dictionary with relative valuation results
    """
    # Extract company ratios
    pe_ratio = ratios.get('valuation', {}).get('pe_ratio')
    pb_ratio = ratios.get('valuation', {}).get('pb_ratio')
    ps_ratio = ratios.get('valuation', {}).get('ps_ratio')
    ev_ebitda = ratios.get('valuation', {}).get('ev_ebitda')

    # Extract sector averages
    sector_pe = sector_multiples.get('pe_ratio')
    sector_pb = sector_multiples.get('pb_ratio')
    sector_ps = sector_multiples.get('ps_ratio')
    sector_ev_ebitda = sector_multiples.get('ev_ebitda')

    # Get earnings, book value, and sales per share
    market_cap = company_info.get('market_cap')
    if not market_cap or market_cap <= 0:
        return None

    # Estimate shares outstanding
    shares_outstanding = market_cap / current_price if current_price > 0 else None
    if not shares_outstanding:
        return None

    # Get latest financial data
    val_ratios = ratios.get('valuation', {})
    if not val_ratios:
        return None

    # Calculate value per share using each multiple
    value_components = {}

    # P/E based valuation
    if pe_ratio and sector_pe and pe_ratio > 0:
        # Get earnings per share
        eps = current_price / pe_ratio

        # Calculate fair value
        pe_value = eps * sector_pe
        value_components["P/E"] = pe_value

    # P/B based valuation
    if pb_ratio and sector_pb and pb_ratio > 0:
        # Get book value per share
        bvps = current_price / pb_ratio

        # Calculate fair value
        pb_value = bvps * sector_pb
        value_components["P/B"] = pb_value

    # P/S based valuation
    if ps_ratio and sector_ps and ps_ratio > 0:
        # Get sales per share
        sps = current_price / ps_ratio

        # Calculate fair value
        ps_value = sps * sector_ps
        value_components["P/S"] = ps_value

    # EV/EBITDA based valuation
    if ev_ebitda and sector_ev_ebitda and ev_ebitda > 0:
        # This is more complex and would need enterprise value and debt data
        # Simplified approach here
        ev_ebitda_value = current_price * (sector_ev_ebitda / ev_ebitda)
        value_components["EV/EBITDA"] = ev_ebitda_value

    # Calculate average fair value if we have components
    if value_components:
        # Weighted average based on reliability of metrics
        weights = {
            "P/E": 0.4,
            "EV/EBITDA": 0.3,
            "P/B": 0.2,
            "P/S": 0.1
        }

        weighted_sum = 0
        total_weight = 0

        for component, value in value_components.items():
            if component in weights:
                weighted_sum += value * weights[component]
                total_weight += weights[component]

        # Calculate weighted average
        fair_value = weighted_sum / total_weight if total_weight > 0 else None

        return {
            "fair_value": fair_value,
            "components": value_components,
            "parameters": {
                "peer_pe": sector_pe,
                "peer_pb": sector_pb,
                "peer_ps": sector_ps,
                "peer_ev_ebitda": sector_ev_ebitda
            }
        }

    return None


# For direct execution
if __name__ == "__main__":
    run_valuation_filter_page()