import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


# Import project modules
from StockAnalysisSystem.src.config import COLORS, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS, SECTOR_MAPPING, normalize_sector
from StockAnalysisSystem.src.utils.financial_data import build_financial_data, cagr, get_series
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.valuation import capital
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
from StockAnalysisSystem.src.pages.screening.stock_screener import (
    DEFAULT_WORKERS, get_stock_universe, get_stocks_by_sector, get_stocks_by_sector_and_subsector,
    parallel_map, progress_reporter,
)
from StockAnalysisSystem.src.utils.rates import risk_free_rate

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
            "Sector model": "The company's sector model (as on Company Analysis > Valuation): sector-specific "
                            "DCF, residual income, NAV or multiples, blended by weight",
            "Relative": "Relative Valuation - based on sector average multiples",
            "Combined": "60 % sector model + 40 % sector multiples"
        }

        selected_approach = st.selectbox(
            "Select valuation approach:",
            options=list(valuation_approaches.keys()),
            index=0  # Default: the sector model
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

        # Model parameters expander (sector model or Combined approach)
        if selected_approach in MODEL_APPROACHES:
            with st.expander("Model Parameters", expanded=True):
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

        include_outliers = st.checkbox(
            f"Include implausible results (|upside| > {OUTLIER_UPSIDE:.0%})", value=False, key="vf_outliers",
            help="Results this far from the price usually mean bad inputs (unusual capital structure, "
                 "one-off earnings, a model that does not fit the business). They are flagged 'check inputs'.")

        # Max number of results
        max_results = st.slider(
            "Maximum number of results:",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

        # Run valuation button
        if st.button("Run Valuation Analysis", type="primary", key="vf_run"):
            st.session_state["vf_run_active"] = True
        run_valuation_button = st.session_state.get("vf_run_active", False)

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
                    discount_rate_adj=discount_rate_adj / 100 if selected_approach in MODEL_APPROACHES else 0,
                    growth_rate_adj=growth_rate_adj / 100 if selected_approach in MODEL_APPROACHES else 0,
                    terminal_growth_adj=terminal_growth_adj / 100 if selected_approach in MODEL_APPROACHES else 0,
                    margin_of_safety=margin_of_safety / 100 if selected_approach in MODEL_APPROACHES else 0.25,
                    pe_discount=pe_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    ps_discount=ps_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    pb_discount=pb_discount / 100 if selected_approach in ["Relative", "Combined"] else 0.15,
                    ev_ebitda_discount=ev_ebitda_discount / 100 if selected_approach in ["Relative",
                                                                                         "Combined"] else 0.15,
                    include_outliers=include_outliers,
                    progress_callback=progress_reporter(st.progress(0.0, text="Starting..."), "Valuing"),
                )
                hidden_outliers = valuation_results.hidden_outliers

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
                if hidden_outliers:
                    st.caption(f"{len(hidden_outliers)} more with upside beyond {OUTLIER_UPSIDE:.0%} hidden as "
                               f"implausible ({', '.join(hidden_outliers[:10])}"
                               f"{', ...' if len(hidden_outliers) > 10 else ''}). Tick 'Include implausible "
                               "results' to list them.")

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
                            "Method range": _fmt_range(stock.get("method_range")),
                            "Check": stock.get("flag") or "",
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
                    styled_df = summary_df.style.map(color_upside, subset=['Upside'])

                    st.dataframe(styled_df, width="stretch")

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
                            st.plotly_chart(fig, width="stretch")

                        # Display valuation parameters
                        if "valuation_parameters" in stock_data:
                            st.subheader("Valuation Parameters")

                            params = stock_data["valuation_parameters"]

                            # Create columns for key parameters
                            col1, col2, col3 = st.columns(3)

                            # DCF parameters
                            if "discount_rate" in params:
                                with col1:
                                    st.metric("Discount Rate", _fmt_pct(params.get('discount_rate')))

                                with col2:
                                    st.metric("Growth Rate", _fmt_pct(params.get('growth_rate')))

                                with col3:
                                    st.metric("Terminal Growth", _fmt_pct(params.get('terminal_growth')))

                            # Relative valuation parameters
                            if "peer_pe" in params:
                                with col1:
                                    st.metric("Peer P/E Ratio", _fmt_multiple(params.get('peer_pe')))

                                with col2:
                                    st.metric("Peer P/B Ratio", _fmt_multiple(params.get('peer_pb')))

                                with col3:
                                    st.metric("Peer P/S Ratio", _fmt_multiple(params.get('peer_ps')))

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
                                st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")

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
                    st.dataframe(upside_df, width="stretch")

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
                            styled_df = sector_df.style.map(color_upside, subset=['Upside'])

                            st.dataframe(styled_df, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")

            except Exception as e:
                logger.exception("Valuation analysis failed")
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
        approach: "Sector model" (alias "DCF"), "Relative" or "Combined"
        sector: Selected sector (or "All")
        **params: Various valuation parameters

    Returns:
        ScreenResults (a list) of stocks above the margin of safety; implausible results
        (|upside| > OUTLIER_UPSIDE) only when ``include_outliers``, their symbols otherwise in
        ``.hidden_outliers``.
    """
    if approach == "DCF":
        approach = "Sector model"
    include_outliers = params.get("include_outliers", False)
    # Extract parameters
    min_market_cap = params.get("min_market_cap", 0)
    max_market_cap = params.get("max_market_cap", float('inf'))
    discount_rate_adj = params.get("discount_rate_adj", 0)
    growth_rate_adj = params.get("growth_rate_adj", 0)
    terminal_growth_adj = params.get("terminal_growth_adj", 0)
    margin_of_safety = params.get("margin_of_safety", 0.25)
    multiple_discounts = {
        "P/E": params.get("pe_discount", 0.15),
        "P/S": params.get("ps_discount", 0.15),
        "P/B": params.get("pb_discount", 0.15),
        "EV/EBITDA": params.get("ev_ebitda_discount", 0.15),
    }

    # Results list
    valuation_results = ScreenResults()

    # Get sector benchmarks for relative valuation
    sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector if sector != "All" else None)

    def dcf_overrides(stock_sector, income_stmt, model_data=None):
        """
        Model assumptions shifted by the user's adjustments. The discount-rate adjustment moves the shared
        cost of capital the model uses by default (WACC; cost of equity for financials), so 0 and +0.25 %
        are continuous (review R-02).
        """
        defaults = {**DCF_PARAMETERS, **SECTOR_DCF_PARAMETERS.get(normalize_sector(stock_sector), {})}
        overrides = {}
        if discount_rate_adj:
            overrides["discount_rate"] = shared_discount_rate(model_data, stock_sector,
                                                              defaults["default_discount_rate"]) + discount_rate_adj
        if terminal_growth_adj:
            overrides["terminal_growth"] = max(defaults["terminal_growth_rate"] + terminal_growth_adj, 0.0)
        if growth_rate_adj:
            historical = cagr(get_series(income_stmt, "revenue", ascending=True))
            base = historical if historical is not None else defaults["terminal_growth_rate"]
            overrides["revenue_growth"] = base + growth_rate_adj
        return overrides

    def analyse(stock):
        """('row', dict), ('outlier', symbol) or None for one stock."""
        symbol = stock["symbol"]

        try:
            # Get company info
            company_info = data_loader.get_company_info(symbol)

            if not company_info:
                return None

            # Market cap filter
            market_cap = company_info.get('market_cap')
            if market_cap is None or not (min_market_cap <= market_cap <= max_market_cap):
                return None

            # Get latest price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date_1y = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            price_data = data_loader.get_historical_prices(symbol, start_date_1y, end_date)

            if price_data.empty:
                return None

            # Get current price
            current_price = price_data['Close'].iloc[-1]

            # Get financial statements
            income_stmt = data_loader.get_financial_statements(symbol, 'income', 'annual')
            balance_sheet = data_loader.get_financial_statements(symbol, 'balance', 'annual')
            cash_flow = data_loader.get_financial_statements(symbol, 'cash', 'annual')

            # Check if we have enough data for valuation
            if income_stmt.empty or balance_sheet.empty:
                return None

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
            if approach in MODEL_APPROACHES:
                try:
                    # Get company sector
                    stock_sector = company_info.get('sector', sector if sector != "All" else None)

                    # Perform DCF valuation with adjusted parameters
                    model_data = build_financial_data(income_stmt, balance_sheet, cash_flow, company_info, price_data)
                    dcf_result = valuation_factory.get_company_valuation(
                        symbol,
                        sector=stock_sector,
                        financial_data=model_data,
                        **dcf_overrides(stock_sector, income_stmt, model_data)
                    )
                    if dcf_result.get('value_per_share') is None:
                        dcf_result = None
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
                        current_price,
                        discounts=multiple_discounts,
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

            model_label = method_label(dcf_result) if dcf_result else None
            method_values = method_values_of(dcf_result) if dcf_result else {}
            if approach == "Sector model" and dcf_result:
                fair_value = dcf_result.get('value_per_share')
                valuation_method = model_label
                valuation_components = method_values or {"Sector model": fair_value}
                valuation_parameters = {
                    **_dcf_parameters(dcf_result)
                }
                notes = "Based on the sector model."

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
                        valuation_method = f"Combined: {model_label} + multiples"
                        valuation_components = {
                            "Sector model (60%)": dcf_value,
                            "Relative (40%)": rel_value
                        }
                        valuation_parameters = {
                            **_dcf_parameters(dcf_result),
                            "peer_pe": relative_result.get('parameters', {}).get('peer_pe'),
                            "peer_pb": relative_result.get('parameters', {}).get('peer_pb'),
                            "peer_ps": relative_result.get('parameters', {}).get('peer_ps')
                        }
                        notes = "Based on the sector model and sector multiples."
                elif dcf_result:
                    fair_value = dcf_result.get('value_per_share')
                    valuation_method = model_label
                    valuation_components = method_values or {"Sector model": fair_value}
                    valuation_parameters = {
                        **_dcf_parameters(dcf_result)
                    }
                    notes = "Based on the sector model. Relative valuation not available."
                elif relative_result:
                    fair_value = relative_result.get('fair_value')
                    valuation_method = "Relative"
                    valuation_components = relative_result.get('components', {})
                    valuation_parameters = relative_result.get('parameters', {})
                    confidence_level = "Medium"
                    notes = "Based on sector average multiples. DCF valuation not available."

            # Calculate upside and sanity-check it. Combined: the range spans the model's methods and the
            # relative leg, because both feed the blended fair value (review R-03).
            range_values = method_values or valuation_components
            if approach == "Combined" and dcf_result and relative_result and relative_result.get('fair_value'):
                range_values = {**method_values, "Relative (multiples)": relative_result['fair_value']}
            sanity = sanity_check(fair_value, current_price, range_values)
            upside = sanity["upside"]
            if sanity["flag"]:
                notes = f"{notes} Check inputs: {sanity['reason']}".strip()

            # Apply margin of safety; implausible results only on request
            if upside is not None and upside > margin_of_safety and sanity["flag"] and not include_outliers:
                return ("outlier", symbol)
            elif upside is not None and upside > margin_of_safety:
                # Stock is potentially undervalued, add to results
                return ("row", {
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
                    "confidence_level": sanity["confidence"] if fair_value else confidence_level,
                    "flag": sanity["flag"],
                    "method_range": sanity["range"],
                    "method_spread": sanity["spread"],
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
            return None

        except Exception as e:
            logger.error(f"Error processing stock {symbol}: {str(e)}")
            return None

    risk_free_rate(data_loader)  # warm the shared rate once before the workers start
    outcomes = parallel_map(analyse, list(universe), workers=params.get("workers") or DEFAULT_WORKERS,
                            progress_callback=params.get("progress_callback"),
                            label=lambda stock: stock.get("symbol", "?"))
    for outcome in outcomes:
        if not outcome:
            continue
        kind, value = outcome
        if kind == "outlier":
            valuation_results.hidden_outliers.append(value)
        else:
            valuation_results.append(value)

    return valuation_results


#: Approaches that run the company's sector model.
MODEL_APPROACHES = ("Sector model", "Combined")
#: Upside beyond +-100 % is treated as implausible (review R4 / U2).
OUTLIER_UPSIDE = 1.0
#: (max - min) / fair value of the weighted methods above this = methods disagree.
WIDE_SPREAD = 0.6


class ScreenResults(list):
    """Screen rows plus the symbols left out as implausible."""

    def __init__(self, *args):
        super().__init__(*args)
        self.hidden_outliers = []


def _short_model(name):
    name = str(name or "Model")
    for suffix in ("SectorValuation", "Valuation"):
        if name.endswith(suffix) and name != suffix:
            name = name[: -len(suffix)]
    return name


def method_values_of(result):
    """Per-share values of the weighted methods of a valuation result, labelled."""
    methods = (result or {}).get("valuation_methods") or {}
    weights = ((result or {}).get("valuation_summary") or {}).get("weights") or {}
    values = {}
    for key, method in methods.items():
        value = method.get("value_per_share")
        if (weights.get(key) or 0) > 0 and isinstance(value, (int, float)) and value > 0:
            values[method.get("method") or key] = float(value)
    return values


def shared_discount_rate(financial_data, sector, fallback):
    """The model's default discount rate: shared WACC, or the cost of equity for financials."""
    if not financial_data:
        return fallback
    try:
        components = capital.cost_of_capital_for(financial_data, fallback)
    except Exception as e:  # noqa: BLE001 - incomplete statements: keep the sector default
        logger.warning(f"Shared cost of capital unavailable: {e}")
        return fallback
    key = "cost_of_equity" if normalize_sector(sector) == "Financials" else "wacc"
    value = components.get(key)
    return float(value) if isinstance(value, (int, float)) else fallback


def method_label(result):
    """The model actually used and its heaviest method, e.g. 'Financial: Residual income (excess return)'."""
    methods = (result or {}).get("valuation_methods") or {}
    weights = ((result or {}).get("valuation_summary") or {}).get("weights") or {}
    model = _short_model((result or {}).get("model"))
    usable = {k: w for k, w in weights.items() if k in methods}
    if not usable:
        return model
    top = max(usable, key=usable.get)
    return f"{model}: {methods[top].get('method') or top}"


def sanity_check(fair_value, current_price, components=None):
    """
    Upside, range and spread of the methods, a 'check inputs' flag and a confidence level.

    Flag when |upside| > OUTLIER_UPSIDE. Confidence: Low when flagged or the methods spread more
    than WIDE_SPREAD of the fair value, Medium above half of that, High otherwise.
    """
    upside = (fair_value / current_price - 1) if fair_value and current_price and current_price > 0 else None
    values = [v for v in (components or {}).values() if isinstance(v, (int, float)) and v > 0]
    value_range = (min(values), max(values)) if values else None
    spread = (value_range[1] - value_range[0]) / fair_value if value_range and fair_value else None
    flag, reason = None, ""
    if upside is not None and abs(upside) > OUTLIER_UPSIDE:
        flag = "check inputs"
        reason = f"fair value is {upside:+.0%} from the price."
    if flag or (spread is not None and spread > WIDE_SPREAD):
        confidence = "Low"
    elif spread is not None and spread > WIDE_SPREAD / 2:
        confidence = "Medium"
    else:
        confidence = "High" if spread is not None else "Medium"
    return {"upside": upside, "flag": flag, "reason": reason, "range": value_range, "spread": spread,
            "confidence": confidence}


def _fmt_range(value_range):
    if not value_range:
        return "N/A"
    return f"${value_range[0]:.0f}-${value_range[1]:.0f}"


def _dcf_parameters(result):
    """Key DCF assumptions from a standardised valuation result."""
    assumptions = (result or {}).get('assumptions') or {}
    return {
        "discount_rate": assumptions.get('discount_rate'),
        "growth_rate": assumptions.get('revenue_growth'),
        "terminal_growth": assumptions.get('terminal_growth'),
    }


def _fmt_pct(value):
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A"


def _fmt_multiple(value):
    return f"{value:.2f}x" if isinstance(value, (int, float)) else "N/A"


def calculate_relative_valuation(symbol, company_info, ratios, sector_multiples, current_price, discounts=None):
    """
    Calculate fair value using relative valuation methods

    Args:
        symbol: Stock symbol
        company_info: Company information
        ratios: Company financial ratios
        sector_multiples: Sector average multiples
        current_price: Current stock price
        discounts: Required discount to each sector multiple, e.g. {"P/E": 0.15}; the fair
            multiple is sector average x (1 - discount)

    Returns:
        Dictionary with relative valuation results
    """
    # Extract company ratios
    pe_ratio = ratios.get('valuation', {}).get('pe_ratio')
    pb_ratio = ratios.get('valuation', {}).get('pb_ratio')
    ps_ratio = ratios.get('valuation', {}).get('ps_ratio')
    ev_ebitda = ratios.get('valuation', {}).get('ev_ebitda')

    # Extract sector averages, reduced by the required discount
    discounts = discounts or {}

    def required_multiple(key, label):
        multiple = sector_multiples.get(key)
        if not multiple:
            return None
        return multiple * (1 - min(max(float(discounts.get(label, 0.0) or 0.0), 0.0), 0.95))

    sector_pe = required_multiple('pe_ratio', 'P/E')
    sector_pb = required_multiple('pb_ratio', 'P/B')
    sector_ps = required_multiple('ps_ratio', 'P/S')
    sector_ev_ebitda = required_multiple('ev_ebitda', 'EV/EBITDA')

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
        # Re-rate enterprise value, then bridge to equity through net debt (EV - market cap)
        enterprise_value = company_info.get('enterprise_value')
        if enterprise_value and enterprise_value > 0:
            net_debt = enterprise_value - market_cap
            implied_equity = enterprise_value * (sector_ev_ebitda / ev_ebitda) - net_debt
            ev_ebitda_value = implied_equity / shares_outstanding
        else:
            ev_ebitda_value = current_price * (sector_ev_ebitda / ev_ebitda)
        if ev_ebitda_value > 0:
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