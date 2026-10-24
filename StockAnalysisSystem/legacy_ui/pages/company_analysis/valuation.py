import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from config import COLORS, UI_SETTINGS
from utils.data_loader import DataLoader
from utils.visualization import FinancialVisualizer
from models.ratio_analysis import FinancialRatioAnalyzer
from valuation.sector_factor import ValuationFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('company_valuation')


def render_company_valuation(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                             price_data: pd.DataFrame, sector_data: Optional[Dict] = None):
    """
    Render the company valuation page

    Args:
        ticker: Company ticker symbol
        financial_data: Dictionary containing financial statement data
        company_info: Dictionary containing company information
        price_data: DataFrame with historical price data
        sector_data: Dictionary containing sector data (optional)
    """
    # Get company name and sector
    company_name = company_info.get('name', ticker)
    sector = company_info.get('sector', 'Unknown')

    # Page header
    st.header(f"Valuation Analysis: {company_name} ({ticker})")
    st.markdown(f"<p style='color: {COLORS['sectors'].get(sector, COLORS['secondary'])}'>Sector: {sector}</p>",
                unsafe_allow_html=True)

    # Create visualization helper
    visualizer = FinancialVisualizer()

    # Create valuation factory
    valuation_factory = ValuationFactory()

    # Get appropriate valuation model based on sector
    valuation_model = valuation_factory.get_valuation_model(sector)

    # Current market data
    current_price = price_data['Close'].iloc[-1] if not price_data.empty else company_info.get('current_price')
    market_cap = company_info.get('market_cap')

    # Show current market valuation
    st.subheader("Current Market Valuation")

    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${current_price:.2f}</div>
                <div class="metric-label">Current Price</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Format market cap
        if market_cap:
            if market_cap >= 1e12:
                formatted_market_cap = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                formatted_market_cap = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                formatted_market_cap = f"${market_cap / 1e6:.2f}M"
            else:
                formatted_market_cap = f"${market_cap:.2f}"
        else:
            formatted_market_cap = "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{formatted_market_cap}</div>
                <div class="metric-label">Market Cap</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        # Get P/E ratio
        pe_ratio = company_info.get('pe_ratio')
        if pe_ratio is None and current_price and financial_data.get('income_statement') is not None:
            # Try to calculate P/E from available data
            income_stmt = financial_data.get('income_statement')
            if not income_stmt.empty and 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                if net_income > 0 and market_cap:
                    pe_ratio = market_cap / net_income

        # Format P/E
        formatted_pe = f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{formatted_pe}</div>
                <div class="metric-label">P/E Ratio</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        # Get EV/EBITDA
        ev_ebitda = company_info.get('ev_ebitda')
        if ev_ebitda is None and market_cap and financial_data.get('income_statement') is not None:
            # Try to calculate EV/EBITDA
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if (not income_stmt.empty and not balance_sheet.empty and
                    'EBITDA' in income_stmt.index and
                    'Total Debt' in balance_sheet.index and
                    'Cash and Cash Equivalents' in balance_sheet.index):

                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]

                if ebitda > 0:
                    enterprise_value = market_cap + total_debt - cash
                    ev_ebitda = enterprise_value / ebitda

        # Format EV/EBITDA
        formatted_ev_ebitda = f"{ev_ebitda:.2f}" if ev_ebitda is not None else "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{formatted_ev_ebitda}</div>
                <div class="metric-label">EV/EBITDA</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create tabs for different valuation methods
    valuation_tabs = st.tabs(["Valuation Multiples", "DCF Valuation", "Sector-Specific", "Fair Value"])

    # Tab 1: Valuation Multiples
    with valuation_tabs[0]:
        st.subheader("Relative Valuation Analysis")

        # Calculate financial ratios
        ratio_analyzer = FinancialRatioAnalyzer()
        ratios = ratio_analyzer.calculate_ratios(financial_data)

        # Get sector benchmarks
        sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector)

        # Get valuation ratios
        valuation_ratios = ratios.get('valuation', {})
        sector_valuation = sector_benchmarks.get('valuation', {})

        # Key valuation multiples
        multiples_to_display = [
            {"name": "P/E Ratio", "key": "pe_ratio",
             "description": "Price to Earnings - Shows how much investors are willing to pay per dollar of earnings"},
            {"name": "P/S Ratio", "key": "ps_ratio",
             "description": "Price to Sales - Market value relative to annual revenue"},
            {"name": "P/B Ratio", "key": "pb_ratio",
             "description": "Price to Book - Market value relative to book value"},
            {"name": "EV/EBITDA", "key": "ev_ebitda",
             "description": "Enterprise Value to EBITDA - Company value relative to earnings before interest, taxes, depreciation, and amortization"},
            {"name": "EV/Revenue", "key": "ev_revenue",
             "description": "Enterprise Value to Revenue - Company value relative to revenue"}
        ]

        # Display multiples table
        multiples_data = []

        for multiple in multiples_to_display:
            company_value = valuation_ratios.get(multiple['key'])
            sector_value = sector_valuation.get(multiple['key'])

            # Calculate percentage difference
            if company_value is not None and sector_value is not None and sector_value != 0:
                perc_diff = (company_value / sector_value - 1) * 100
                # For valuation multiples, lower is generally better
                assessment = "Undervalued" if perc_diff < -10 else "Overvalued" if perc_diff > 10 else "Fair Valued"
                assessment_color = COLORS["primary"] if perc_diff < -10 else COLORS["accent"] if perc_diff > 10 else \
                COLORS["warning"]
            else:
                perc_diff = None
                assessment = "N/A"
                assessment_color = "#8888"

            # Format values
            formatted_company = f"{company_value:.2f}" if company_value is not None else "N/A"
            formatted_sector = f"{sector_value:.2f}" if sector_value is not None else "N/A"
            formatted_diff = f"{perc_diff:+.1f}%" if perc_diff is not None else "N/A"

            multiples_data.append({
                "Multiple": multiple['name'],
                "Company": formatted_company,
                "Sector Avg": formatted_sector,
                "Difference": formatted_diff,
                "Assessment": assessment,
                "Color": assessment_color,
                "Description": multiple['description']
            })

        # Convert to DataFrame
        multiples_df = pd.DataFrame(multiples_data)

        # Custom styling for the table
        styled_df = pd.DataFrame({
            "Multiple": multiples_df["Multiple"],
            "Company": multiples_df["Company"],
            "Sector Avg": multiples_df["Sector Avg"],
            "Difference": multiples_df["Difference"],
            "Assessment": [f"<span style='color:{row.Color}'>{row.Assessment}</span>" for _, row in
                           multiples_df.iterrows()]
        })

        # Display the styled dataframe
        st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Display descriptions for each multiple
        with st.expander("Understanding Valuation Multiples"):
            for _, row in multiples_df.iterrows():
                st.markdown(f"**{row['Multiple']}**: {row['Description']}")

        # Visualize multiples comparison
        if any(valuation_ratios.values()) and any(sector_valuation.values()):
            st.subheader("Comparative Valuation Analysis")

            # Prepare data for visualization
            valid_multiples = []
            company_values = []
            sector_values = []

            for multiple in multiples_to_display:
                company_value = valuation_ratios.get(multiple['key'])
                sector_value = sector_valuation.get(multiple['key'])

                if company_value is not None and sector_value is not None:
                    valid_multiples.append(multiple['name'])
                    company_values.append(company_value)
                    sector_values.append(sector_value)

            if valid_multiples:
                # Create DataFrame for visualization
                viz_data = pd.DataFrame({
                    'Multiple': valid_multiples,
                    f'{company_name}': company_values,
                    'Sector Average': sector_values
                })

                # Plot comparative bar chart
                fig = visualizer.plot_valuation_multiples_comparison(
                    viz_data,
                    company_name=company_name,
                    sector_name=f"{sector} Sector",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation
                st.markdown("""
                <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <h4>Interpretation</h4>
                    <p>🔹 <strong>Lower multiples</strong> generally indicate a company may be <span style='color:#74f174'>undervalued</span> relative to its peers.</p>
                    <p>🔹 <strong>Higher multiples</strong> may suggest the company is <span style='color:#faa1a4'>overvalued</span> or that the market expects higher growth.</p>
                    <p>🔹 Consider industry context and company growth rates when interpreting these multiples.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data to generate valuation multiples visualization.")

    # Tab 2: DCF Valuation
    with valuation_tabs[1]:
        st.subheader("Discounted Cash Flow (DCF) Valuation")

        # Create two columns - one for inputs, one for results
        dcf_col1, dcf_col2 = st.columns([3, 2])

        with dcf_col1:
            # DCF Inputs
            st.markdown('<div class="section-header"><h4>DCF Model Inputs</h4></div>', unsafe_allow_html=True)

            # Extract historical financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Extract historical free cash flow
            historical_fcf = None
            if cash_flow is not None and not cash_flow.empty:
                if 'Free Cash Flow' in cash_flow.index:
                    historical_fcf = cash_flow.loc['Free Cash Flow']
                elif all(x in cash_flow.index for x in ['Operating Cash Flow', 'Capital Expenditure']):
                    ocf = cash_flow.loc['Operating Cash Flow']
                    capex = cash_flow.loc['Capital Expenditure']
                    historical_fcf = ocf - abs(capex)

            # Display historical FCF if available
            if historical_fcf is not None and len(historical_fcf) > 0:
                # Create a DataFrame for display
                fcf_data = historical_fcf.to_frame(name='FCF')

                # Format for display (in millions or billions)
                fcf_display = fcf_data.copy()
                fcf_display['FCF'] = fcf_display['FCF'].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else
                    f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else
                    f"${x:.2f}"
                )

                # Calculate growth rates
                if len(historical_fcf) > 1:
                    fcf_series = historical_fcf.astype(float)
                    growth_rates = fcf_series.pct_change() * 100
                    # Reverse to get oldest to newest
                    growth_rates = growth_rates.iloc[::-1]
                    # Drop the first value which will be NaN
                    growth_rates = growth_rates.dropna()

                    # Calculate average growth rate
                    avg_growth_rate = growth_rates.mean()

                    # Store for later use
                    historical_growth_rate = avg_growth_rate / 100  # Convert percentage to decimal
                else:
                    historical_growth_rate = 0.03  # Default 3% if not enough data

                # Display historical FCF
                st.write("Historical Free Cash Flow (FCF):")
                st.write(fcf_display)

                # Display estimated growth rate
                if len(historical_fcf) > 1:
                    st.markdown(f"Estimated FCF Growth Rate: **{avg_growth_rate:.2f}%**")
            else:
                st.warning("Historical free cash flow data not available.")
                historical_growth_rate = 0.03  # Default 3% if no data

            # User inputs for DCF model
            st.markdown("### Model Parameters")

            # Get sector-specific defaults
            if sector in ["Technology", "Healthcare"]:
                default_growth = max(0.05, historical_growth_rate)
                default_terminal_growth = 0.03
                default_discount_rate = 0.10
            elif sector in ["Utilities", "Consumer Staples"]:
                default_growth = max(0.02, historical_growth_rate)
                default_terminal_growth = 0.02
                default_discount_rate = 0.07
            else:
                default_growth = max(0.03, historical_growth_rate)
                default_terminal_growth = 0.02
                default_discount_rate = 0.09

            # Growth rate
            growth_rate = st.slider(
                "Projected Annual Growth Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=default_growth * 100,
                step=0.5,
                help="Estimated annual growth rate for the company's free cash flow"
            ) / 100  # Convert to decimal

            # Terminal growth rate
            terminal_growth = st.slider(
                "Terminal Growth Rate (%)",
                min_value=0.0,
                max_value=5.0,
                value=default_terminal_growth * 100,
                step=0.1,
                help="Long-term growth rate that the company's free cash flow is expected to grow at indefinitely"
            ) / 100  # Convert to decimal

            # Discount rate
            discount_rate = st.slider(
                "Discount Rate / WACC (%)",
                min_value=4.0,
                max_value=20.0,
                value=default_discount_rate * 100,
                step=0.1,
                help="Weighted Average Cost of Capital - rate used to discount future cash flows"
            ) / 100  # Convert to decimal

            # Forecast period
            forecast_years = st.slider(
                "Forecast Period (Years)",
                min_value=5,
                max_value=10,
                value=5,
                step=1,
                help="Number of years to forecast future cash flows"
            )

        with dcf_col2:
            # DCF Results
            st.markdown('<div class="section-header"><h4>DCF Valuation Results</h4></div>', unsafe_allow_html=True)

            # Get most recent FCF as base
            if historical_fcf is not None and len(historical_fcf) > 0:
                base_fcf = historical_fcf.iloc[0]
            else:
                # Fallback: Estimate FCF from net income
                if income_stmt is not None and not income_stmt.empty and 'Net Income' in income_stmt.index:
                    # Simple estimate: Net Income * 0.8
                    base_fcf = income_stmt.loc['Net Income'].iloc[0] * 0.8
                else:
                    st.error("Insufficient data to perform DCF valuation.")
                    return

            # Perform DCF calculation
            try:
                # Forecast FCF
                forecast_fcf = []
                for year in range(1, forecast_years + 1):
                    fcf = base_fcf * (1 + growth_rate) ** year
                    forecast_fcf.append(fcf)

                # Calculate terminal value
                terminal_value = forecast_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

                # Calculate PV of forecast FCF
                pv_fcf = sum(fcf / (1 + discount_rate) ** year for year, fcf in enumerate(forecast_fcf, 1))

                # Calculate PV of terminal value
                pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

                # Enterprise value
                enterprise_value = pv_fcf + pv_terminal

                # Calculate equity value by subtracting net debt
                net_debt = 0
                if balance_sheet is not None and not balance_sheet.empty:
                    if all(x in balance_sheet.index for x in ['Total Debt', 'Cash and Cash Equivalents']):
                        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                        cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]
                        net_debt = total_debt - cash

                equity_value = enterprise_value - net_debt

                # Calculate per share value
                shares_outstanding = company_info.get('shares_outstanding')
                if not shares_outstanding and market_cap and current_price:
                    # Estimate from market cap and price
                    shares_outstanding = market_cap / current_price

                if shares_outstanding:
                    value_per_share = equity_value / shares_outstanding

                    # Compare to current price
                    if current_price:
                        upside_potential = (value_per_share / current_price - 1) * 100

                # Display results
                if enterprise_value >= 1e12:
                    ev_display = f"${enterprise_value / 1e12:.2f} trillion"
                elif enterprise_value >= 1e9:
                    ev_display = f"${enterprise_value / 1e9:.2f} billion"
                elif enterprise_value >= 1e6:
                    ev_display = f"${enterprise_value / 1e6:.2f} million"
                else:
                    ev_display = f"${enterprise_value:.2f}"

                if equity_value >= 1e12:
                    equity_display = f"${equity_value / 1e12:.2f} trillion"
                elif equity_value >= 1e9:
                    equity_display = f"${equity_value / 1e9:.2f} billion"
                elif equity_value >= 1e6:
                    equity_display = f"${equity_value / 1e6:.2f} million"
                else:
                    equity_display = f"${equity_value:.2f}"

                # Display color based on upside potential
                if current_price:
                    value_color = COLORS["primary"] if value_per_share > current_price else COLORS["accent"]
                else:
                    value_color = COLORS["secondary"]

                # Display results in cards
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {value_color};">${value_per_share:.2f}</div>
                        <div class="metric-label">Estimated Fair Value Per Share</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if current_price:
                    upside_color = COLORS["primary"] if upside_potential > 0 else COLORS["accent"]
                    upside_icon = "↗" if upside_potential > 0 else "↘"

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {upside_color};">{upside_icon} {upside_potential:+.2f}%</div>
                            <div class="metric-label">Upside Potential</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{ev_display}</div>
                        <div class="metric-label">Enterprise Value</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{equity_display}</div>
                        <div class="metric-label">Equity Value</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Add assessment
                if current_price:
                    if upside_potential > 20:
                        assessment = "Significantly Undervalued"
                        assessment_color = COLORS["primary"]
                    elif upside_potential > 5:
                        assessment = "Moderately Undervalued"
                        assessment_color = COLORS["primary"]
                    elif upside_potential > -5:
                        assessment = "Fairly Valued"
                        assessment_color = COLORS["warning"]
                    elif upside_potential > -20:
                        assessment = "Moderately Overvalued"
                        assessment_color = COLORS["accent"]
                    else:
                        assessment = "Significantly Overvalued"
                        assessment_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {assessment_color}20; color: {assessment_color}; 
                                    padding: 10px; border-radius: 5px; margin-top: 20px; text-align: center;">
                            <h3 style="margin: 0;">{assessment}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error in DCF calculation: {str(e)}")
                st.warning("Please check your inputs and try again.")

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        st.markdown(
            "This analysis shows how the estimated share price changes with different growth and discount rates.")

        # Create a matrix of values
        growth_rates = [growth_rate - 0.02, growth_rate - 0.01, growth_rate, growth_rate + 0.01, growth_rate + 0.02]
        discount_rates = [discount_rate - 0.02, discount_rate - 0.01, discount_rate, discount_rate + 0.01,
                          discount_rate + 0.02]

        # Create the sensitivity matrix
        sensitivity_matrix = []

        for g_rate in growth_rates:
            row = []
            for d_rate in discount_rates:
                try:
                    # Skip invalid combinations
                    if d_rate <= terminal_growth or g_rate < 0:
                        row.append(None)
                        continue

                    # Forecast FCF with this growth rate
                    sens_forecast_fcf = []
                    for year in range(1, forecast_years + 1):
                        fcf = base_fcf * (1 + g_rate) ** year
                        sens_forecast_fcf.append(fcf)

                    # Calculate terminal value with this discount rate
                    sens_terminal_value = sens_forecast_fcf[-1] * (1 + terminal_growth) / (d_rate - terminal_growth)

                    # Calculate PV of forecast FCF
                    sens_pv_fcf = sum(fcf / (1 + d_rate) ** year for year, fcf in enumerate(sens_forecast_fcf, 1))

                    # Calculate PV of terminal value
                    sens_pv_terminal = sens_terminal_value / (1 + d_rate) ** forecast_years

                    # Enterprise value
                    sens_enterprise_value = sens_pv_fcf + sens_pv_terminal

                    # Equity value
                    sens_equity_value = sens_enterprise_value - net_debt

                    # Value per share
                    sens_value_per_share = sens_equity_value / shares_outstanding

                    row.append(sens_value_per_share)
                except:
                    row.append(None)

            sensitivity_matrix.append(row)

        # Create a DataFrame for display
        growth_labels = [f"{r * 100:.1f}%" for r in growth_rates]
        discount_labels = [f"{r * 100:.1f}%" for r in discount_rates]

        sensitivity_df = pd.DataFrame(sensitivity_matrix, index=growth_labels, columns=discount_labels)

        # Plot heatmap
        fig = visualizer.plot_sensitivity_heatmap(
            sensitivity_df,
            x_title="Discount Rate",
            y_title="Growth Rate",
            current_value=value_per_share,
            current_price=current_price,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Sector-Specific Valuation
    with valuation_tabs[2]:
        st.subheader(f"{sector}-Specific Valuation")

        # Based on sector, display appropriate models and metrics
        if sector == "Financials":
            # Financials sector focus on P/B, ROE, and Dividend model
            st.markdown("""
            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h4>Financial Sector Valuation Approach</h4>
                <p>Financial companies like banks and insurance firms are typically valued differently from other sectors:</p>
                <ul>
                    <li><strong>P/B Ratio</strong> is more relevant than P/E</li>
                    <li><strong>Dividend Discount Model</strong> is commonly used</li>
                    <li><strong>ROE</strong> is a critical performance metric</li>
                    <li>Regulatory capital requirements affect valuation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Display key financial sector specific metrics
            fin_col1, fin_col2 = st.columns(2)

            with fin_col1:
                # P/B Ratio Analysis
                st.markdown('<div class="section-header"><h4>P/B Ratio Analysis</h4></div>', unsafe_allow_html=True)

                # Get P/B ratio
                pb_ratio = valuation_ratios.get('pb_ratio')
                sector_pb = sector_valuation.get('pb_ratio')

                # Display P/B comparison
                if pb_ratio is not None and sector_pb is not None:
                    pb_diff = (pb_ratio / sector_pb - 1) * 100
                    pb_color = COLORS["primary"] if pb_ratio < sector_pb else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{pb_ratio:.2f}</div>
                            <div class="metric-label">P/B Ratio</div>
                            <div style="color: {pb_color}; font-size: 14px;">{pb_diff:+.1f}% vs Sector Avg: {sector_pb:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # P/B Interpretation
                    if pb_ratio < sector_pb * 0.8:
                        interpretation = "Significantly below sector average - potentially undervalued if the bank maintains good asset quality."
                        int_color = COLORS["primary"]
                    elif pb_ratio < sector_pb * 0.95:
                        interpretation = "Moderately below sector average - may represent good value if fundamentals are sound."
                        int_color = COLORS["primary"]
                    elif pb_ratio < sector_pb * 1.05:
                        interpretation = "In line with sector average - fairly valued relative to peers."
                        int_color = COLORS["warning"]
                    elif pb_ratio < sector_pb * 1.2:
                        interpretation = "Above sector average - commands premium due to higher growth or better ROE."
                        int_color = COLORS["warning"]
                    else:
                        interpretation = "Significantly above sector average - may be overvalued unless growth prospects are exceptional."
                        int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("P/B ratio data not available.")

            with fin_col2:
                # ROE Analysis for Financials
                st.markdown('<div class="section-header"><h4>Return on Equity (ROE)</h4></div>', unsafe_allow_html=True)

                # Get ROE
                roe = ratios.get('profitability', {}).get('roe')
                sector_roe = sector_benchmarks.get('profitability', {}).get('roe')

                if roe is not None:
                    roe_percent = roe * 100
                    sector_roe_percent = sector_roe * 100 if sector_roe is not None else None

                    # Display ROE
                    roe_color = COLORS["primary"] if roe_percent > 10 else COLORS["warning"] if roe_percent > 5 else \
                    COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {roe_color};">{roe_percent:.2f}%</div>
                            <div class="metric-label">Return on Equity</div>
                            <div style="font-size: 14px;">
                                {f"Sector Avg: {sector_roe_percent:.2f}%" if sector_roe_percent is not None else ""}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ROE Interpretation for Financials
                    if roe_percent > 15:
                        interpretation = "Excellent ROE - indicates strong profitability and efficient use of capital."
                        int_color = COLORS["primary"]
                    elif roe_percent > 10:
                        interpretation = "Good ROE - above average performance for financial institutions."
                        int_color = COLORS["primary"]
                    elif roe_percent > 7:
                        interpretation = "Acceptable ROE - in line with typical returns for the financial sector."
                        int_color = COLORS["warning"]
                    elif roe_percent > 4:
                        interpretation = "Below average ROE - may indicate efficiency issues or lower leverage."
                        int_color = COLORS["warning"]
                    else:
                        interpretation = "Poor ROE - significantly underperforming the financial sector."
                        int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("ROE data not available.")

            # Dividend analysis for Financials
            st.markdown('<div class="section-header"><h4>Dividend Analysis</h4></div>', unsafe_allow_html=True)

            div_yield = company_info.get('dividend_yield')
            if div_yield is not None:
                div_yield_percent = div_yield * 100

                # Get sector average dividend yield
                sector_div_yield = sector_benchmarks.get('dividend_yield', 0.03) * 100

                # Display dividend yield
                div_color = COLORS["primary"] if div_yield_percent > sector_div_yield else COLORS["warning"]

                div_col1, div_col2 = st.columns(2)

                with div_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {div_color};">{div_yield_percent:.2f}%</div>
                            <div class="metric-label">Dividend Yield</div>
                            <div style="font-size: 14px;">Sector Avg: {sector_div_yield:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with div_col2:
                    # Payout ratio if available
                    payout_ratio = company_info.get('payout_ratio')
                    if payout_ratio is not None:
                        payout_percent = payout_ratio * 100
                        payout_color = COLORS["warning"] if payout_percent > 60 else COLORS["primary"]

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {payout_color};">{payout_percent:.2f}%</div>
                                <div class="metric-label">Payout Ratio</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Payout ratio not available.")
            else:
                st.info("Dividend data not available.")

        elif sector == "Technology":
            # Technology sector focuses on growth, R&D, and recurring revenue
            st.markdown("""
            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h4>Technology Sector Valuation Approach</h4>
                <p>Tech companies are typically valued with a focus on:</p>
                <ul>
                    <li><strong>Growth rates</strong> are more important than current profitability</li>
                    <li><strong>P/S Ratio</strong> often used for high-growth companies not yet profitable</li>
                    <li><strong>R&D investment</strong> as a driver of future growth</li>
                    <li><strong>Gross margin</strong> as an indicator of pricing power and scalability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Display key tech sector metrics
            tech_col1, tech_col2 = st.columns(2)

            with tech_col1:
                # P/S Ratio Analysis
                st.markdown('<div class="section-header"><h4>P/S Ratio Analysis</h4></div>', unsafe_allow_html=True)

                # Get P/S ratio
                ps_ratio = valuation_ratios.get('ps_ratio')
                sector_ps = sector_valuation.get('ps_ratio')

                # Display P/S comparison
                if ps_ratio is not None and sector_ps is not None:
                    ps_diff = (ps_ratio / sector_ps - 1) * 100
                    ps_color = COLORS["primary"] if ps_ratio < sector_ps else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{ps_ratio:.2f}</div>
                            <div class="metric-label">P/S Ratio</div>
                            <div style="color: {ps_color}; font-size: 14px;">{ps_diff:+.1f}% vs Sector Avg: {sector_ps:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Get gross margin for context
                    gross_margin = ratios.get('profitability', {}).get('gross_margin')

                    if gross_margin is not None:
                        gross_margin_percent = gross_margin * 100

                        # P/S Interpretation for Tech with Gross Margin context
                        if gross_margin_percent > 70:
                            gm_context = f"high gross margin of {gross_margin_percent:.1f}%"
                            margin_assessment = "excellent"
                        elif gross_margin_percent > 50:
                            gm_context = f"good gross margin of {gross_margin_percent:.1f}%"
                            margin_assessment = "good"
                        else:
                            gm_context = f"moderate gross margin of {gross_margin_percent:.1f}%"
                            margin_assessment = "moderate"

                        if ps_ratio < sector_ps * 0.7:
                            interpretation = f"Significantly below sector average - potentially undervalued given its {margin_assessment} {gm_context}."
                            int_color = COLORS["primary"]
                        elif ps_ratio < sector_ps * 0.9:
                            interpretation = f"Below sector average - may represent good value with {gm_context}."
                            int_color = COLORS["primary"]
                        elif ps_ratio < sector_ps * 1.1:
                            interpretation = f"In line with sector average - fairly valued with {gm_context}."
                            int_color = COLORS["warning"]
                        elif ps_ratio < sector_ps * 1.3:
                            interpretation = f"Above sector average - premium valuation potentially justified by {gm_context}."
                            int_color = COLORS["warning"]
                        else:
                            interpretation = f"Significantly above sector average - high growth expectations with {gm_context}."
                            int_color = COLORS["accent"]
                    else:
                        # P/S Interpretation without gross margin context
                        if ps_ratio < sector_ps * 0.7:
                            interpretation = "Significantly below sector average - potentially undervalued."
                            int_color = COLORS["primary"]
                        elif ps_ratio < sector_ps * 0.9:
                            interpretation = "Below sector average - may represent good value."
                            int_color = COLORS["primary"]
                        elif ps_ratio < sector_ps * 1.1:
                            interpretation = "In line with sector average - fairly valued."
                            int_color = COLORS["warning"]
                        elif ps_ratio < sector_ps * 1.3:
                            interpretation = "Above sector average - premium valuation."
                            int_color = COLORS["warning"]
                        else:
                            interpretation = "Significantly above sector average - high growth expectations."
                            int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("P/S ratio data not available.")

            with tech_col2:
                # Growth Analysis for Tech
                st.markdown('<div class="section-header"><h4>Growth Metrics</h4></div>', unsafe_allow_html=True)

                # Extract growth data
                revenue_growth = ratios.get('growth', {}).get('revenue_growth')

                if revenue_growth is not None:
                    growth_percent = revenue_growth * 100
                    growth_color = COLORS["primary"] if growth_percent > 15 else COLORS[
                        "warning"] if growth_percent > 5 else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {growth_color};">{growth_percent:.1f}%</div>
                            <div class="metric-label">Revenue Growth (YoY)</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Growth Interpretation for Tech
                    if growth_percent > 30:
                        interpretation = "Excellent growth rate - significantly outperforming tech sector averages."
                        int_color = COLORS["primary"]
                    elif growth_percent > 15:
                        interpretation = "Strong growth rate - above average for technology companies."
                        int_color = COLORS["primary"]
                    elif growth_percent > 8:
                        interpretation = "Good growth rate - in line with tech sector expectations."
                        int_color = COLORS["warning"]
                    elif growth_percent > 3:
                        interpretation = "Moderate growth - below average for technology sector."
                        int_color = COLORS["warning"]
                    else:
                        interpretation = "Low growth rate - significantly underperforming tech sector expectations."
                        int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Try to calculate from historical data
                    if income_stmt is not None and not income_stmt.empty and 'Total Revenue' in income_stmt.index and \
                            income_stmt.shape[1] >= 2:
                        current_rev = income_stmt.loc['Total Revenue'].iloc[0]
                        prev_rev = income_stmt.loc['Total Revenue'].iloc[1]

                        if prev_rev > 0:
                            revenue_growth = (current_rev / prev_rev) - 1
                            growth_percent = revenue_growth * 100
                            growth_color = COLORS["primary"] if growth_percent > 15 else COLORS[
                                "warning"] if growth_percent > 5 else COLORS["accent"]

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: {growth_color};">{growth_percent:.1f}%</div>
                                    <div class="metric-label">Revenue Growth (YoY)</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("Revenue growth data not available.")
                    else:
                        st.warning("Revenue growth data not available.")

            # R&D Analysis for Tech
            st.markdown('<div class="section-header"><h4>R&D Investment Analysis</h4></div>', unsafe_allow_html=True)

            # Look for R&D expenses in income statement
            rd_expense = None
            rd_to_revenue = None

            if income_stmt is not None and not income_stmt.empty:
                # Try different variations of R&D line items
                rd_candidates = ['Research and Development', 'Research & Development', 'R&D Expenses', 'R&D']

                for candidate in rd_candidates:
                    if candidate in income_stmt.index:
                        rd_expense = income_stmt.loc[candidate].iloc[0]

                        if 'Total Revenue' in income_stmt.index:
                            total_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                            if total_revenue > 0:
                                rd_to_revenue = rd_expense / total_revenue
                        break

            if rd_expense is not None and rd_to_revenue is not None:
                rd_percent = rd_to_revenue * 100
                rd_color = COLORS["primary"] if rd_percent > 10 else COLORS["warning"]

                # Format values
                if rd_expense >= 1e9:
                    rd_display = f"${rd_expense / 1e9:.2f}B"
                elif rd_expense >= 1e6:
                    rd_display = f"${rd_expense / 1e6:.2f}M"
                else:
                    rd_display = f"${rd_expense:.2f}"

                rd_col1, rd_col2 = st.columns(2)

                with rd_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{rd_display}</div>
                            <div class="metric-label">R&D Expenses</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with rd_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {rd_color};">{rd_percent:.2f}%</div>
                            <div class="metric-label">R&D / Revenue</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # R&D Interpretation
                if rd_percent > 20:
                    interpretation = "Very high R&D investment - indicating strong focus on innovation and future growth."
                    int_color = COLORS["primary"]
                elif rd_percent > 12:
                    interpretation = "High R&D investment - above average for technology companies."
                    int_color = COLORS["primary"]
                elif rd_percent > 8:
                    interpretation = "Moderate R&D investment - typical for established technology companies."
                    int_color = COLORS["warning"]
                elif rd_percent > 4:
                    interpretation = "Below average R&D investment - may indicate mature products or efficiency."
                    int_color = COLORS["warning"]
                else:
                    interpretation = "Low R&D investment - may limit future growth opportunities."
                    int_color = COLORS["accent"]

                st.markdown(
                    f"""
                    <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <p style="margin: 0; color: {int_color};">{interpretation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("R&D expense data not available.")

        elif sector == "Energy":
            # Energy sector focuses on reserves, production costs, and capital efficiency
            st.markdown("""
            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h4>Energy Sector Valuation Approach</h4>
                <p>Energy companies are typically valued with a focus on:</p>
                <ul>
                    <li><strong>EV/EBITDA</strong> is a primary valuation metric</li>
                    <li><strong>Reserve Life Index</strong> indicates future production potential</li>
                    <li><strong>Production costs</strong> determine profitability at various commodity prices</li>
                    <li><strong>Capital efficiency</strong> shows return on invested capital</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Display key energy sector metrics
            energy_col1, energy_col2 = st.columns(2)

            with energy_col1:
                # EV/EBITDA Analysis
                st.markdown('<div class="section-header"><h4>EV/EBITDA Analysis</h4></div>', unsafe_allow_html=True)

                # Get EV/EBITDA
                ev_ebitda = valuation_ratios.get('ev_ebitda')
                sector_ev_ebitda = sector_valuation.get('ev_ebitda')

                # Display EV/EBITDA comparison
                if ev_ebitda is not None and sector_ev_ebitda is not None:
                    ev_diff = (ev_ebitda / sector_ev_ebitda - 1) * 100
                    ev_color = COLORS["primary"] if ev_ebitda < sector_ev_ebitda else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{ev_ebitda:.2f}</div>
                            <div class="metric-label">EV/EBITDA</div>
                            <div style="color: {ev_color}; font-size: 14px;">{ev_diff:+.1f}% vs Sector Avg: {sector_ev_ebitda:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # EV/EBITDA Interpretation for Energy
                    if ev_ebitda < sector_ev_ebitda * 0.7:
                        interpretation = "Significantly below sector average - potentially undervalued but check for structural issues."
                        int_color = COLORS["primary"]
                    elif ev_ebitda < sector_ev_ebitda * 0.9:
                        interpretation = "Below sector average - may represent good value if fundamentals are sound."
                        int_color = COLORS["primary"]
                    elif ev_ebitda < sector_ev_ebitda * 1.1:
                        interpretation = "In line with sector average - fairly valued relative to peers."
                        int_color = COLORS["warning"]
                    elif ev_ebitda < sector_ev_ebitda * 1.3:
                        interpretation = "Above sector average - premium may be warranted if growth prospects are strong."
                        int_color = COLORS["warning"]
                    else:
                        interpretation = "Significantly above sector average - may be overvalued unless growth is exceptional."
                        int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("EV/EBITDA data not available.")

            with energy_col2:
                # FCF Yield Analysis for Energy
                st.markdown('<div class="section-header"><h4>FCF Yield Analysis</h4></div>', unsafe_allow_html=True)

                # Calculate FCF Yield
                fcf_yield = None

                if cash_flow is not None and not cash_flow.empty and market_cap:
                    if 'Free Cash Flow' in cash_flow.index:
                        fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
                        fcf_yield = fcf / market_cap
                    elif all(x in cash_flow.index for x in ['Operating Cash Flow', 'Capital Expenditure']):
                        ocf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                        capex = cash_flow.loc['Capital Expenditure'].iloc[0]
                        fcf = ocf - abs(capex)
                        fcf_yield = fcf / market_cap

                if fcf_yield is not None:
                    fcf_yield_percent = fcf_yield * 100
                    fcf_color = COLORS["primary"] if fcf_yield_percent > 5 else COLORS[
                        "warning"] if fcf_yield_percent > 0 else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {fcf_color};">{fcf_yield_percent:.2f}%</div>
                            <div class="metric-label">FCF Yield</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # FCF Yield Interpretation for Energy
                    if fcf_yield_percent > 10:
                        interpretation = "Excellent FCF yield - indicates strong cash generation relative to market value."
                        int_color = COLORS["primary"]
                    elif fcf_yield_percent > 6:
                        interpretation = "Strong FCF yield - above average for energy companies."
                        int_color = COLORS["primary"]
                    elif fcf_yield_percent > 3:
                        interpretation = "Moderate FCF yield - in line with energy sector averages."
                        int_color = COLORS["warning"]
                    elif fcf_yield_percent > 0:
                        interpretation = "Low FCF yield - below average for the energy sector."
                        int_color = COLORS["warning"]
                    else:
                        interpretation = "Negative FCF yield - company is not generating positive free cash flow."
                        int_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {int_color}20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="margin: 0; color: {int_color};">{interpretation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("FCF Yield data not available.")

        else:
            # Generic sector analysis
            st.info(f"Sector-specific valuation for {sector} will be shown here. This feature is under development.")

    # Tab 4: Fair Value Summary
    with valuation_tabs[3]:
        st.subheader("Fair Value Summary")

        # Create a valuation summary from all methods
        valuation_summary = []

        # 1. Relative valuation (multiples)
        # Use average of common multiples to estimate value
        relative_value = None
        if valuation_ratios and sector_valuation:
            pe_ratio = valuation_ratios.get('pe_ratio')
            sector_pe = sector_valuation.get('pe_ratio')

            pb_ratio = valuation_ratios.get('pb_ratio')
            sector_pb = sector_valuation.get('pb_ratio')

            ps_ratio = valuation_ratios.get('ps_ratio')
            sector_ps = sector_valuation.get('ps_ratio')

            ev_ebitda = valuation_ratios.get('ev_ebitda')
            sector_ev_ebitda = sector_valuation.get('ev_ebitda')

            # Initialize values
            pe_value = pb_value = ps_value = ev_value = None

            # Calculate values based on sector multiples
            if pe_ratio and sector_pe and income_stmt is not None and not income_stmt.empty and 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                eps = net_income / company_info.get('shares_outstanding') if company_info.get(
                    'shares_outstanding') else 0
                if eps > 0:
                    pe_value = eps * sector_pe

            if pb_ratio and sector_pb and balance_sheet is not None and not balance_sheet.empty and 'Total Stockholder Equity' in balance_sheet.index:
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                shares = company_info.get('shares_outstanding')
                if shares and shares > 0:
                    book_value_per_share = equity / shares
                    pb_value = book_value_per_share * sector_pb

            if ps_ratio and sector_ps and income_stmt is not None and not income_stmt.empty and 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                shares = company_info.get('shares_outstanding')
                if shares and shares > 0:
                    sales_per_share = revenue / shares
                    ps_value = sales_per_share * sector_ps

            if ev_ebitda and sector_ev_ebitda and income_stmt is not None and not income_stmt.empty and 'EBITDA' in income_stmt.index:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                shares = company_info.get('shares_outstanding')
                if shares and shares > 0:
                    # Calculate enterprise value
                    ev = ebitda * sector_ev_ebitda
                    # Subtract net debt to get equity value
                    net_debt = 0
                    if balance_sheet is not None and not balance_sheet.empty:
                        if all(x in balance_sheet.index for x in ['Total Debt', 'Cash and Cash Equivalents']):
                            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                            cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]
                            net_debt = total_debt - cash
                    equity_value = ev - net_debt
                    ev_value = equity_value / shares

            # Calculate average relative value from available methods
            values = [v for v in [pe_value, pb_value, ps_value, ev_value] if v is not None]
            if values:
                relative_value = sum(values) / len(values)

                # Add to summary
                valuation_summary.append({
                    "method": "Relative Valuation (Multiples)",
                    "value": relative_value,
                    "weight": 0.3,  # Weight for this method
                    "description": f"Based on {len(values)} industry multiples",
                    "color": COLORS["secondary"]
                })

        # 2. DCF valuation
        # Use the DCF value calculated earlier if available
        dcf_value = value_per_share if 'value_per_share' in locals() else None
        if dcf_value:
            valuation_summary.append({
                "method": "DCF Valuation",
                "value": dcf_value,
                "weight": 0.5,  # DCF gets higher weight
                "description": f"Growth: {growth_rate:.1%}, Discount: {discount_rate:.1%}",
                "color": COLORS["primary"]
            })

        # 3. Current market price
        if current_price:
            valuation_summary.append({
                "method": "Current Market Price",
                "value": current_price,
                "weight": 0.2,  # Current price gets lowest weight
                "description": "Current trading price",
                "color": COLORS["info"]
            })

        # Display valuation summary table
        if valuation_summary:
            # Create DataFrame
            summary_df = pd.DataFrame(valuation_summary)

            # Calculate weighted average
            weighted_sum = sum(row["value"] * row["weight"] for row in valuation_summary)
            total_weight = sum(row["weight"] for row in valuation_summary)
            weighted_average = weighted_sum / total_weight if total_weight > 0 else None

            # Display summary table
            st.markdown("### Valuation Methods Comparison")

            # Format values for display
            for i, row in summary_df.iterrows():
                # Create a colored badge for each method
                st.markdown(
                    f"""
                    <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 10px 0; 
                               border-left: 4px solid {row['color']};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0;">{row['method']}</h4>
                                <p style="margin: 5px 0; color: #a0a0a0; font-size: 14px;">{row['description']}</p>
                            </div>
                            <div style="text-align: right;">
                                <h3 style="margin: 0;">${row['value']:.2f}</h3>
                                <p style="margin: 0; color: #a0a0a0; font-size: 12px;">Weight: {row['weight']:.1f}</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Display weighted average
            if weighted_average:
                # Compare to current price
                if current_price:
                    upside = (weighted_average / current_price - 1) * 100
                    value_color = COLORS["primary"] if upside > 0 else COLORS["accent"]
                    upside_text = f"{upside:+.1f}% vs current price"
                else:
                    value_color = COLORS["secondary"]
                    upside_text = ""

                st.markdown("### Fair Value Estimate")
                st.markdown(
                    f"""
                    <div style="background-color: {value_color}20; padding: 20px; border-radius: 5px; margin: 20px 0; text-align: center;">
                        <h1 style="margin: 0; color: {value_color};">${weighted_average:.2f}</h1>
                        <p style="margin: 5px 0; font-size: 16px;">Weighted Average Fair Value</p>
                        <p style="color: {value_color}; font-size: 14px;">{upside_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Final assessment
                if current_price:
                    if upside > 20:
                        assessment = "Significantly Undervalued"
                        assessment_color = COLORS["primary"]
                    elif upside > 5:
                        assessment = "Moderately Undervalued"
                        assessment_color = COLORS["primary"]
                    elif upside > -5:
                        assessment = "Fairly Valued"
                        assessment_color = COLORS["warning"]
                    elif upside > -20:
                        assessment = "Moderately Overvalued"
                        assessment_color = COLORS["accent"]
                    else:
                        assessment = "Significantly Overvalued"
                        assessment_color = COLORS["accent"]

                    st.markdown(
                        f"""
                        <div style="background-color: {assessment_color}20; color: {assessment_color}; 
                                    padding: 15px; border-radius: 5px; margin: 10px 0; text-align: center;">
                            <h3 style="margin: 0;">{assessment}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Insufficient data to calculate fair value estimates.")


if __name__ == "__main__":
    # For direct testing
    # Dummy data for testing
    import yfinance as yf

    ticker = "AAPL"
    ticker_obj = yf.Ticker(ticker)

    company_info = {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'exchange': 'NASDAQ',
        'currency': 'USD',
        'market_cap': 2800000000000,
        'pe_ratio': 32.5,
        'description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'
    }

    price_data = ticker_obj.history(period="1y")
    income_stmt = ticker_obj.income_stmt
    balance_sheet = ticker_obj.balance_sheet
    cash_flow = ticker_obj.cashflow

    financial_data = {
        'income_statement': income_stmt,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow,
    }

    render_company_valuation(ticker, financial_data, company_info, price_data)