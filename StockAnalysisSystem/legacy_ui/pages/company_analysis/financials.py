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
from models.financial_statements import FinancialStatementAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('company_financials')


def render_company_financials(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                              price_data: pd.DataFrame, sector_data: Optional[Dict] = None):
    """
    Render the company financials page

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
    st.header(f"Financial Analysis: {company_name} ({ticker})")
    st.markdown(f"<p style='color: {COLORS['sectors'].get(sector, COLORS['secondary'])}'>Sector: {sector}</p>",
                unsafe_allow_html=True)

    # Create visualization helper
    visualizer = FinancialVisualizer()

    # Create statement analyzer
    statement_analyzer = FinancialStatementAnalyzer()

    # Extract financial statements
    income_stmt = financial_data.get('income_statement')
    balance_sheet = financial_data.get('balance_sheet')
    cash_flow = financial_data.get('cash_flow')

    # Check if we have financial data
    if (income_stmt is None or income_stmt.empty or
            balance_sheet is None or balance_sheet.empty or
            cash_flow is None or cash_flow.empty):
        st.warning("Financial statement data is not available for this company.")
        return

    # Create tabs for different financial statements
    financial_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios", "Growth Analysis"])

    # Tab 1: Income Statement
    with financial_tabs[0]:
        st.subheader("Income Statement Analysis")

        # Analyze income statement
        income_analysis = statement_analyzer.analyze_income_statement(income_stmt)

        # Create selector for view options
        view_options = ["Summary View", "Detailed View", "Visual Analysis", "Trend Analysis", "Common Size Analysis"]
        income_view = st.radio("Select View:", view_options, horizontal=True, key="income_view")

        if income_view == "Summary View":
            # Display income statement summary
            st.markdown('<div class="section-header"><h4>Income Statement Summary</h4></div>', unsafe_allow_html=True)

            # Extract key metrics
            last_periods = min(5, income_stmt.shape[1])
            periods = income_stmt.columns[:last_periods]

            # Get key income metrics
            metrics = [
                {"name": "Revenue", "row": "Total Revenue"},
                {"name": "Cost of Revenue", "row": "Cost of Revenue"},
                {"name": "Gross Profit", "row": "Gross Profit"},
                {"name": "Operating Expenses", "row": "Operating Expenses"},
                {"name": "Operating Income", "row": "Operating Income"},
                {"name": "Net Income", "row": "Net Income"}
            ]

            # Create dataframe for display
            income_df = pd.DataFrame(index=[m["name"] for m in metrics])

            # Fill data
            for period in periods:
                period_data = []
                for metric in metrics:
                    if metric["row"] in income_stmt.index:
                        period_data.append(income_stmt.loc[metric["row"], period])
                    else:
                        period_data.append(None)
                income_df[period] = period_data

            # Format for display
            income_display = income_df.copy()
            for col in income_display.columns:
                # Format values in millions/billions
                income_display[col] = income_display[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(
                        x) >= 1e6 else f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.table(income_display)

            # Display key ratios derived from income statement
            st.markdown('<div class="section-header"><h4>Key Profitability Metrics</h4></div>', unsafe_allow_html=True)

            # Calculate key profitability ratios
            if 'Total Revenue' in income_stmt.index and income_stmt.loc['Total Revenue'].iloc[0] != 0:
                total_revenue = income_stmt.loc['Total Revenue'].iloc[0]

                # Create columns for metrics
                profit_cols = st.columns(3)

                with profit_cols[0]:
                    # Gross Margin
                    if 'Gross Profit' in income_stmt.index:
                        gross_profit = income_stmt.loc['Gross Profit'].iloc[0]
                        gross_margin = gross_profit / total_revenue * 100

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value">{gross_margin:.2f}%</div>
                                <div class="metric-label">Gross Margin</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with profit_cols[1]:
                    # Operating Margin
                    if 'Operating Income' in income_stmt.index:
                        operating_income = income_stmt.loc['Operating Income'].iloc[0]
                        operating_margin = operating_income / total_revenue * 100

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value">{operating_margin:.2f}%</div>
                                <div class="metric-label">Operating Margin</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with profit_cols[2]:
                    # Net Margin
                    if 'Net Income' in income_stmt.index:
                        net_income = income_stmt.loc['Net Income'].iloc[0]
                        net_margin = net_income / total_revenue * 100

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value">{net_margin:.2f}%</div>
                                <div class="metric-label">Net Margin</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        elif income_view == "Detailed View":
            # Display full income statement
            st.markdown('<div class="section-header"><h4>Detailed Income Statement</h4></div>', unsafe_allow_html=True)

            # Format income statement for display
            display_income = income_stmt.copy()

            # Format each column
            for col in display_income.columns:
                display_income[col] = display_income[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else
                    f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else
                    f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.dataframe(display_income, use_container_width=True)

            # Option to download as CSV
            csv = income_stmt.to_csv().encode('utf-8')
            st.download_button(
                "Download Income Statement as CSV",
                csv,
                f"{ticker}_income_statement.csv",
                "text/csv",
                key='download-income-csv'
            )

        elif income_view == "Visual Analysis":
            # Visual representation of income statement components
            st.markdown('<div class="section-header"><h4>Revenue & Earnings Visualization</h4></div>',
                        unsafe_allow_html=True)

            # Revenue and earnings trend
            if 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                # Create DataFrame for visualization
                viz_data = pd.DataFrame({
                    'Revenue': income_stmt.loc['Total Revenue'],
                    'Net Income': income_stmt.loc['Net Income']
                })

                # Plot revenue and income
                fig = visualizer.plot_income_statement_components(
                    viz_data,
                    title="Revenue & Net Income",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Income breakdown
            st.markdown('<div class="section-header"><h4>Profit Components Breakdown</h4></div>',
                        unsafe_allow_html=True)

            # Check if we have the necessary components
            key_components = ['Gross Profit', 'Operating Income', 'Net Income']
            if all(comp in income_stmt.index for comp in key_components):
                # Get most recent data
                latest_data = income_stmt.iloc[:, 0]

                # Create data for waterfall chart
                waterfall_data = [
                    {'category': 'Revenue', 'value': latest_data['Total Revenue']},
                    {'category': 'Cost of Revenue',
                     'value': -1 * (latest_data['Total Revenue'] - latest_data['Gross Profit'])},
                    {'category': 'Gross Profit', 'value': latest_data['Gross Profit'], 'cumulative': True},
                    {'category': 'Operating Expenses',
                     'value': -1 * (latest_data['Gross Profit'] - latest_data['Operating Income'])},
                    {'category': 'Operating Income', 'value': latest_data['Operating Income'], 'cumulative': True},
                    {'category': 'Other Income/Expenses',
                     'value': -1 * (latest_data['Operating Income'] - latest_data['Net Income'])},
                    {'category': 'Net Income', 'value': latest_data['Net Income'], 'cumulative': True}
                ]

                # Plot waterfall chart
                fig = visualizer.plot_profit_waterfall(
                    waterfall_data,
                    title="Profit Breakdown",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to create profit breakdown visualization.")

            # Expense breakdown
            st.markdown('<div class="section-header"><h4>Expense Breakdown</h4></div>', unsafe_allow_html=True)

            # Find expense items
            expense_items = [item for item in income_stmt.index if 'expense' in item.lower() or 'cost' in item.lower()]
            if expense_items:
                # Get most recent data
                latest_data = income_stmt.iloc[:, 0]

                # Create data for pie chart
                expense_data = {}
                for item in expense_items:
                    value = latest_data[item]
                    if value < 0:  # Make sure it's a positive number for the pie chart
                        value = abs(value)
                    expense_data[item] = value

                # Plot pie chart
                fig = visualizer.plot_expense_breakdown(
                    expense_data,
                    title="Expense Composition",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not identify expense items for breakdown visualization.")

        elif income_view == "Trend Analysis":
            # Trend analysis of income statement items
            st.markdown('<div class="section-header"><h4>Revenue & Income Growth Trends</h4></div>',
                        unsafe_allow_html=True)

            # Key metrics to track
            trend_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']

            # Check if we have enough periods
            if income_stmt.shape[1] > 1:
                # Calculate growth rates
                growth_data = {}

                for metric in trend_metrics:
                    if metric in income_stmt.index:
                        series = income_stmt.loc[metric]
                        growth = series.pct_change(-1) * 100  # Calculate YoY growth
                        growth_data[f"{metric} Growth"] = growth

                if growth_data:
                    # Create DataFrame
                    growth_df = pd.DataFrame(growth_data)

                    # Plot growth trends
                    fig = visualizer.plot_growth_trends(
                        growth_df,
                        title="YoY Growth Rates",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display growth table
                    st.markdown('<div class="section-header"><h4>Growth Rates (%)</h4></div>', unsafe_allow_html=True)

                    # Format growth rates for display
                    growth_display = growth_df.copy()
                    for col in growth_display.columns:
                        growth_display[col] = growth_display[col].apply(
                            lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                        )

                    st.table(growth_display)
                else:
                    st.info("Could not calculate growth trends from the available data.")
            else:
                st.info("Need at least two periods of data to calculate growth trends.")

            # Margin trends
            st.markdown('<div class="section-header"><h4>Margin Trends</h4></div>', unsafe_allow_html=True)

            # Calculate margins over time
            if 'Total Revenue' in income_stmt.index and income_stmt.shape[1] > 0:
                margin_data = {}

                # Calculate each margin if possible
                if 'Gross Profit' in income_stmt.index:
                    margin_data['Gross Margin'] = income_stmt.loc['Gross Profit'] / income_stmt.loc[
                        'Total Revenue'] * 100

                if 'Operating Income' in income_stmt.index:
                    margin_data['Operating Margin'] = income_stmt.loc['Operating Income'] / income_stmt.loc[
                        'Total Revenue'] * 100

                if 'Net Income' in income_stmt.index:
                    margin_data['Net Margin'] = income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue'] * 100

                if margin_data:
                    # Create DataFrame
                    margin_df = pd.DataFrame(margin_data)

                    # Plot margin trends
                    fig = visualizer.plot_margin_trends(
                        margin_df,
                        title="Margin Trends",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display margin table
                    st.markdown('<div class="section-header"><h4>Profit Margins (%)</h4></div>', unsafe_allow_html=True)

                    # Format margins for display
                    margin_display = margin_df.copy()
                    for col in margin_display.columns:
                        margin_display[col] = margin_display[col].apply(
                            lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                        )

                    st.table(margin_display)
                else:
                    st.info("Could not calculate margin trends from the available data.")
            else:
                st.info("Insufficient revenue data to calculate margin trends.")

        elif income_view == "Common Size Analysis":
            # Common size analysis (everything as % of revenue)
            st.markdown('<div class="section-header"><h4>Common Size Income Statement</h4></div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Common size analysis shows each line item as a percentage of total revenue, helping to identify trends in cost structure and profitability.")

            if 'Total Revenue' in income_stmt.index:
                # Create common size dataframe
                common_size = pd.DataFrame(index=income_stmt.index)

                # Calculate percentages for each period
                for col in income_stmt.columns:
                    revenue = income_stmt.loc['Total Revenue', col]
                    if revenue != 0:
                        common_size[col] = income_stmt[col] / revenue * 100
                    else:
                        common_size[col] = np.nan

                # Format for display
                common_size_display = common_size.copy()
                for col in common_size_display.columns:
                    common_size_display[col] = common_size_display[col].apply(
                        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                    )

                # Display the table
                st.dataframe(common_size_display, use_container_width=True)

                # Visualize key components
                st.markdown('<div class="section-header"><h4>Key Components as % of Revenue</h4></div>',
                            unsafe_allow_html=True)

                # Select components to visualize
                key_components = [
                    'Cost of Revenue',
                    'Gross Profit',
                    'Research and Development' if 'Research and Development' in income_stmt.index else None,
                    'Selling General and Administrative' if 'Selling General and Administrative' in income_stmt.index else None,
                    'Operating Income',
                    'Net Income'
                ]

                # Filter out None values
                key_components = [comp for comp in key_components if comp is not None and comp in common_size.index]

                if key_components:
                    # Create DataFrame for selected components
                    comp_df = common_size.loc[key_components]

                    # Transpose for better visualization
                    comp_df = comp_df.T

                    # Plot stacked area chart
                    fig = visualizer.plot_common_size_trends(
                        comp_df,
                        title="Key Components as % of Revenue",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not identify key components for visualization.")
            else:
                st.warning("Total Revenue data not available for common size analysis.")

    # Tab 2: Balance Sheet
    with financial_tabs[1]:
        st.subheader("Balance Sheet Analysis")

        # Analyze balance sheet
        balance_analysis = statement_analyzer.analyze_balance_sheet(balance_sheet)

        # Create selector for view options
        view_options = ["Summary View", "Detailed View", "Visual Analysis", "Trend Analysis", "Common Size Analysis"]
        balance_view = st.radio("Select View:", view_options, horizontal=True, key="balance_view")

        if balance_view == "Summary View":
            # Display balance sheet summary
            st.markdown('<div class="section-header"><h4>Balance Sheet Summary</h4></div>', unsafe_allow_html=True)

            # Extract key metrics
            last_periods = min(5, balance_sheet.shape[1])
            periods = balance_sheet.columns[:last_periods]

            # Get key balance sheet metrics
            metrics = [
                {"name": "Total Current Assets", "row": "Total Current Assets"},
                {"name": "Total Assets", "row": "Total Assets"},
                {"name": "Total Current Liabilities", "row": "Total Current Liabilities"},
                {"name": "Total Liabilities", "row": "Total Liabilities"},
                {"name": "Total Stockholder Equity", "row": "Total Stockholder Equity"}
            ]

            # Create dataframe for display
            balance_df = pd.DataFrame(index=[m["name"] for m in metrics])

            # Fill data
            for period in periods:
                period_data = []
                for metric in metrics:
                    if metric["row"] in balance_sheet.index:
                        period_data.append(balance_sheet.loc[metric["row"], period])
                    else:
                        period_data.append(None)
                balance_df[period] = period_data

            # Format for display
            balance_display = balance_df.copy()
            for col in balance_display.columns:
                # Format values in millions/billions
                balance_display[col] = balance_display[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(
                        x) >= 1e6 else f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.table(balance_display)

            # Display key ratios derived from balance sheet
            st.markdown('<div class="section-header"><h4>Key Balance Sheet Metrics</h4></div>', unsafe_allow_html=True)

            # Create columns for metrics
            balance_cols = st.columns(3)

            with balance_cols[0]:
                # Current Ratio
                if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                    current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                    current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]

                    if current_liabilities != 0:
                        current_ratio = current_assets / current_liabilities

                        # Determine color based on ratio
                        if current_ratio >= 2:
                            ratio_color = COLORS["primary"]  # Good
                        elif current_ratio >= 1:
                            ratio_color = COLORS["warning"]  # Adequate
                        else:
                            ratio_color = COLORS["accent"]  # Poor

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {ratio_color};">{current_ratio:.2f}</div>
                                <div class="metric-label">Current Ratio</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            with balance_cols[1]:
                # Debt-to-Equity Ratio
                if 'Total Liabilities' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                    total_liabilities = balance_sheet.loc['Total Liabilities'].iloc[0]
                    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]

                    if total_equity != 0:
                        debt_equity_ratio = total_liabilities / total_equity

                        # Determine color based on ratio (lower is better)
                        if debt_equity_ratio <= 1:
                            ratio_color = COLORS["primary"]  # Good
                        elif debt_equity_ratio <= 2:
                            ratio_color = COLORS["warning"]  # Adequate
                        else:
                            ratio_color = COLORS["accent"]  # Poor

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {ratio_color};">{debt_equity_ratio:.2f}</div>
                                <div class="metric-label">Debt-to-Equity</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            with balance_cols[2]:
                # Working Capital
                if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                    current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                    current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]

                    working_capital = current_assets - current_liabilities

                    # Format working capital
                    if abs(working_capital) >= 1e9:
                        wc_display = f"${working_capital / 1e9:.2f}B"
                    elif abs(working_capital) >= 1e6:
                        wc_display = f"${working_capital / 1e6:.2f}M"
                    else:
                        wc_display = f"${working_capital:.2f}"

                    # Determine color based on value
                    ratio_color = COLORS["primary"] if working_capital > 0 else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {ratio_color};">{wc_display}</div>
                            <div class="metric-label">Working Capital</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        elif balance_view == "Detailed View":
            # Display full balance sheet
            st.markdown('<div class="section-header"><h4>Detailed Balance Sheet</h4></div>', unsafe_allow_html=True)

            # Format balance sheet for display
            display_balance = balance_sheet.copy()

            # Format each column
            for col in display_balance.columns:
                display_balance[col] = display_balance[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else
                    f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else
                    f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.dataframe(display_balance, use_container_width=True)

            # Option to download as CSV
            csv = balance_sheet.to_csv().encode('utf-8')
            st.download_button(
                "Download Balance Sheet as CSV",
                csv,
                f"{ticker}_balance_sheet.csv",
                "text/csv",
                key='download-balance-csv'
            )

        elif balance_view == "Visual Analysis":
            # Visual representation of balance sheet components
            st.markdown('<div class="section-header"><h4>Assets & Liabilities Visualization</h4></div>',
                        unsafe_allow_html=True)

            # Get most recent balance sheet data
            latest_data = balance_sheet.iloc[:, 0]

            # Check if we have the necessary components
            if 'Total Current Assets' in latest_data and 'Total Current Liabilities' in latest_data:
                # Create data for asset breakdown
                asset_components = {}
                liability_components = {}

                # Extract asset components
                if 'Total Current Assets' in latest_data:
                    asset_components['Current Assets'] = latest_data['Total Current Assets']

                if 'Total Assets' in latest_data and 'Total Current Assets' in latest_data:
                    asset_components['Non-Current Assets'] = latest_data['Total Assets'] - latest_data[
                        'Total Current Assets']

                # Extract liability components
                if 'Total Current Liabilities' in latest_data:
                    liability_components['Current Liabilities'] = latest_data['Total Current Liabilities']

                if 'Total Liabilities' in latest_data and 'Total Current Liabilities' in latest_data:
                    liability_components['Non-Current Liabilities'] = latest_data['Total Liabilities'] - latest_data[
                        'Total Current Liabilities']

                if 'Total Stockholder Equity' in latest_data:
                    liability_components['Stockholder Equity'] = latest_data['Total Stockholder Equity']

                # Create column layout
                visual_col1, visual_col2 = st.columns(2)

                with visual_col1:
                    # Asset composition
                    st.markdown('<div class="section-header"><h5>Asset Composition</h5></div>', unsafe_allow_html=True)

                    if asset_components:
                        fig = visualizer.plot_balance_sheet_composition(
                            asset_components,
                            title="Asset Breakdown",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not extract asset components for visualization.")

                with visual_col2:
                    # Liability and Equity composition
                    st.markdown('<div class="section-header"><h5>Liabilities & Equity</h5></div>',
                                unsafe_allow_html=True)

                    if liability_components:
                        fig = visualizer.plot_balance_sheet_composition(
                            liability_components,
                            title="Liabilities & Equity Breakdown",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not extract liability components for visualization.")

                # Balance Sheet Structure
                st.markdown('<div class="section-header"><h4>Balance Sheet Structure</h4></div>',
                            unsafe_allow_html=True)

                # Prepare data for waterfall chart
                if 'Total Assets' in latest_data and 'Total Liabilities' in latest_data and 'Total Stockholder Equity' in latest_data:
                    waterfall_data = [
                        {'category': 'Total Assets', 'value': latest_data['Total Assets']},
                        {'category': 'Total Liabilities', 'value': -latest_data['Total Liabilities']},
                        {'category': 'Equity', 'value': latest_data['Total Stockholder Equity'], 'isSum': True}
                    ]

                    fig = visualizer.plot_balance_sheet_waterfall(
                        waterfall_data,
                        title="Balance Sheet Structure",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not extract balance sheet structure for visualization.")

                # Working Capital Analysis
                st.markdown('<div class="section-header"><h4>Working Capital Analysis</h4></div>',
                            unsafe_allow_html=True)

                if 'Total Current Assets' in latest_data and 'Total Current Liabilities' in latest_data:
                    current_assets = latest_data['Total Current Assets']
                    current_liabilities = latest_data['Total Current Liabilities']
                    working_capital = current_assets - current_liabilities

                    # Create data for gauge chart
                    wc_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0

                    fig = visualizer.plot_working_capital_gauge(
                        wc_ratio,
                        title="Current Ratio",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Working Capital Components
                    wc_components = {}

                    # Try to find specific current asset components
                    for asset_type in ['Cash and Cash Equivalents', 'Short Term Investments',
                                       'Net Receivables', 'Inventory', 'Other Current Assets']:
                        if asset_type in latest_data:
                            wc_components[asset_type] = latest_data[asset_type]

                    # Try to find specific current liability components
                    for liability_type in ['Accounts Payable', 'Short Term Debt',
                                           'Accrued Liabilities', 'Other Current Liabilities']:
                        if liability_type in latest_data:
                            wc_components[liability_type] = -latest_data[liability_type]  # Negative for liabilities

                    if wc_components:
                        fig = visualizer.plot_working_capital_components(
                            wc_components,
                            title="Working Capital Components",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Current assets and liabilities data not available for working capital analysis.")

        elif balance_view == "Trend Analysis":
            # Trend analysis of balance sheet items
            st.markdown('<div class="section-header"><h4>Balance Sheet Trends</h4></div>', unsafe_allow_html=True)

            # Key metrics to track
            trend_metrics = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity',
                             'Total Current Assets', 'Total Current Liabilities']

            # Check if we have enough periods
            if balance_sheet.shape[1] > 1:
                # Create DataFrames for visualization
                metrics_data = {}

                for metric in trend_metrics:
                    if metric in balance_sheet.index:
                        metrics_data[metric] = balance_sheet.loc[metric]

                if metrics_data:
                    # Create DataFrame
                    metrics_df = pd.DataFrame(metrics_data)

                    # Plot trends
                    fig = visualizer.plot_balance_sheet_trends(
                        metrics_df,
                        title="Balance Sheet Trends",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not extract key metrics for trend visualization.")

                # Ratio trends
                st.markdown('<div class="section-header"><h4>Financial Ratio Trends</h4></div>',
                            unsafe_allow_html=True)

                # Calculate ratios over time
                ratio_data = {}

                # Current Ratio
                if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                    current_assets = balance_sheet.loc['Total Current Assets']
                    current_liabilities = balance_sheet.loc['Total Current Liabilities']
                    ratio_data['Current Ratio'] = current_assets / current_liabilities

                # Debt-to-Equity Ratio
                if 'Total Liabilities' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                    total_liabilities = balance_sheet.loc['Total Liabilities']
                    total_equity = balance_sheet.loc['Total Stockholder Equity']
                    ratio_data['Debt-to-Equity'] = total_liabilities / total_equity

                # Debt-to-Assets Ratio
                if 'Total Liabilities' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                    total_liabilities = balance_sheet.loc['Total Liabilities']
                    total_assets = balance_sheet.loc['Total Assets']
                    ratio_data['Debt-to-Assets'] = total_liabilities / total_assets

                if ratio_data:
                    # Create DataFrame
                    ratio_df = pd.DataFrame(ratio_data)

                    # Plot ratio trends
                    fig = visualizer.plot_balance_sheet_ratios(
                        ratio_df,
                        title="Balance Sheet Ratio Trends",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display ratio table
                    st.markdown('<div class="section-header"><h4>Financial Ratios</h4></div>',
                                unsafe_allow_html=True)

                    # Format ratios for display
                    ratio_display = ratio_df.copy()
                    for col in ratio_display.columns:
                        ratio_display[col] = ratio_display[col].apply(
                            lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
                        )

                    st.table(ratio_display)
                else:
                    st.info("Could not calculate ratio trends from the available data.")
            else:
                st.info("Need at least two periods of data to perform trend analysis.")

        elif balance_view == "Common Size Analysis":
            # Common size analysis (everything as % of total assets)
            st.markdown('<div class="section-header"><h4>Common Size Balance Sheet</h4></div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Common size analysis shows each line item as a percentage of total assets, helping to identify changes in the company's financial structure over time.")

            if 'Total Assets' in balance_sheet.index:
                # Create common size dataframe
                common_size = pd.DataFrame(index=balance_sheet.index)

                # Calculate percentages for each period
                for col in balance_sheet.columns:
                    total_assets = balance_sheet.loc['Total Assets', col]
                    if total_assets != 0:
                        common_size[col] = balance_sheet[col] / total_assets * 100
                    else:
                        common_size[col] = np.nan

                # Format for display
                common_size_display = common_size.copy()
                for col in common_size_display.columns:
                    common_size_display[col] = common_size_display[col].apply(
                        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                    )

                # Display the table
                st.dataframe(common_size_display, use_container_width=True)

                # Visualize key components
                st.markdown('<div class="section-header"><h4>Key Components as % of Total Assets</h4></div>',
                            unsafe_allow_html=True)

                # Select components to visualize
                key_components = [
                    'Total Current Assets',
                    'Property Plant and Equipment' if 'Property Plant and Equipment' in balance_sheet.index else None,
                    'Intangible Assets' if 'Intangible Assets' in balance_sheet.index else None,
                    'Total Current Liabilities',
                    'Long Term Debt' if 'Long Term Debt' in balance_sheet.index else None,
                    'Total Stockholder Equity'
                ]

                # Filter out None values
                key_components = [comp for comp in key_components if comp is not None and comp in common_size.index]

                if key_components:
                    # Create DataFrame for selected components
                    comp_df = common_size.loc[key_components]

                    # Transpose for better visualization
                    comp_df = comp_df.T

                    # Plot stacked area chart
                    fig = visualizer.plot_common_size_trends(
                        comp_df,
                        title="Key Components as % of Total Assets",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not identify key components for visualization.")
            else:
                st.warning("Total Assets data not available for common size analysis.")

    # Tab 3: Cash Flow
    with financial_tabs[2]:
        st.subheader("Cash Flow Analysis")

        # Analyze cash flow statement
        cash_flow_analysis = statement_analyzer.analyze_cash_flow(cash_flow)

        # Create selector for view options
        view_options = ["Summary View", "Detailed View", "Visual Analysis", "Trend Analysis"]
        cash_flow_view = st.radio("Select View:", view_options, horizontal=True, key="cash_flow_view")

        if cash_flow_view == "Summary View":
            # Display cash flow summary
            st.markdown('<div class="section-header"><h4>Cash Flow Summary</h4></div>', unsafe_allow_html=True)

            # Extract key metrics
            last_periods = min(5, cash_flow.shape[1])
            periods = cash_flow.columns[:last_periods]

            # Get key cash flow metrics
            metrics = [
                {"name": "Operating Cash Flow", "row": "Operating Cash Flow"},
                {"name": "Capital Expenditure", "row": "Capital Expenditure"},
                {"name": "Free Cash Flow", "row": "Free Cash Flow", "calculated": True},
                {"name": "Investing Cash Flow", "row": "Investing Cash Flow"},
                {"name": "Financing Cash Flow", "row": "Financing Cash Flow"},
                {"name": "Net Change in Cash", "row": "Net Change in Cash"}
            ]

            # Create dataframe for display
            cf_df = pd.DataFrame(index=[m["name"] for m in metrics])

            # Fill data
            for period in periods:
                period_data = []
                for metric in metrics:
                    if metric.get("calculated") and metric["name"] == "Free Cash Flow":
                        # Calculate FCF if needed
                        if "Operating Cash Flow" in cash_flow.index and "Capital Expenditure" in cash_flow.index:
                            ocf = cash_flow.loc["Operating Cash Flow", period]
                            capex = cash_flow.loc["Capital Expenditure", period]
                            # Ensure capex is negative (for subtraction)
                            capex_abs = abs(capex) if capex < 0 else -capex
                            fcf = ocf - capex_abs
                            period_data.append(fcf)
                        else:
                            period_data.append(None)
                    elif metric["row"] in cash_flow.index:
                        period_data.append(cash_flow.loc[metric["row"], period])
                    else:
                        period_data.append(None)
                cf_df[period] = period_data

            # Format for display
            cf_display = cf_df.copy()
            for col in cf_display.columns:
                # Format values in millions/billions
                cf_display[col] = cf_display[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(
                        x) >= 1e6 else f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.table(cf_display)

            # Display key ratios derived from cash flow
            st.markdown('<div class="section-header"><h4>Key Cash Flow Metrics</h4></div>',
                        unsafe_allow_html=True)

            # Create columns for metrics
            cf_cols = st.columns(3)

            with cf_cols[0]:
                # Free Cash Flow
                if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                    ocf = cash_flow.loc["Operating Cash Flow"].iloc[0]
                    capex = cash_flow.loc["Capital Expenditure"].iloc[0]

                    # Ensure capex is negative (for subtraction)
                    capex_abs = abs(capex) if capex < 0 else -capex
                    fcf = ocf - capex_abs

                    # Format FCF
                    if abs(fcf) >= 1e9:
                        fcf_display = f"${fcf / 1e9:.2f}B"
                    elif abs(fcf) >= 1e6:
                        fcf_display = f"${fcf / 1e6:.2f}M"
                    else:
                        fcf_display = f"${fcf:.2f}"

                    # Determine color based on value
                    fcf_color = COLORS["primary"] if fcf > 0 else COLORS["accent"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {fcf_color};">{fcf_display}</div>
                            <div class="metric-label">Free Cash Flow</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with cf_cols[1]:
                # Cash Flow Conversion
                if all(x in cash_flow.index for x in ["Operating Cash Flow"]) and "Net Income" in income_stmt.index:
                    ocf = cash_flow.loc["Operating Cash Flow"].iloc[0]
                    net_income = income_stmt.loc["Net Income"].iloc[0]

                    if net_income != 0:
                        cf_conversion = ocf / net_income * 100

                        # Determine color based on ratio
                        if cf_conversion >= 100:
                            conv_color = COLORS["primary"]  # Good
                        elif cf_conversion >= 80:
                            conv_color = COLORS["warning"]  # Adequate
                        else:
                            conv_color = COLORS["accent"]  # Poor

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {conv_color};">{cf_conversion:.2f}%</div>
                                <div class="metric-label">OCF to Net Income</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            with cf_cols[2]:
                # FCF Yield (if market cap available)
                if "market_cap" in company_info:
                    market_cap = company_info["market_cap"]

                    if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                        ocf = cash_flow.loc["Operating Cash Flow"].iloc[0]
                        capex = cash_flow.loc["Capital Expenditure"].iloc[0]

                        # Ensure capex is negative (for subtraction)
                        capex_abs = abs(capex) if capex < 0 else -capex
                        fcf = ocf - capex_abs

                        if market_cap > 0:
                            fcf_yield = fcf / market_cap * 100

                            # Determine color based on yield
                            if fcf_yield >= 5:
                                yield_color = COLORS["primary"]  # Good
                            elif fcf_yield >= 2:
                                yield_color = COLORS["warning"]  # Adequate
                            else:
                                yield_color = COLORS["accent"]  # Poor

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: {yield_color};">{fcf_yield:.2f}%</div>
                                    <div class="metric-label">FCF Yield</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        elif cash_flow_view == "Detailed View":
            # Display full cash flow statement
            st.markdown('<div class="section-header"><h4>Detailed Cash Flow Statement</h4></div>',
                        unsafe_allow_html=True)

            # Format cash flow for display
            display_cf = cash_flow.copy()

            # Format each column
            for col in display_cf.columns:
                display_cf[col] = display_cf[col].apply(
                    lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else
                    f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else
                    f"${x:.2f}" if x is not None else "N/A"
                )

            # Display the table
            st.dataframe(display_cf, use_container_width=True)

            # Option to download as CSV
            csv = cash_flow.to_csv().encode('utf-8')
            st.download_button(
                "Download Cash Flow Statement as CSV",
                csv,
                f"{ticker}_cash_flow.csv",
                "text/csv",
                key='download-cash-flow-csv'
            )

        elif cash_flow_view == "Visual Analysis":
            # Visual representation of cash flow components
            st.markdown('<div class="section-header"><h4>Cash Flow Components</h4></div>',
                        unsafe_allow_html=True)

            # Check if we have the necessary components
            if all(x in cash_flow.index for x in ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]):
                # Get most recent data
                latest_data = cash_flow.iloc[:, 0]

                # Create data for waterfall chart
                waterfall_data = [
                    {'category': 'Operating CF', 'value': latest_data['Operating Cash Flow']},
                    {'category': 'Investing CF', 'value': latest_data['Investing Cash Flow']},
                    {'category': 'Financing CF', 'value': latest_data['Financing Cash Flow']},
                    {'category': 'Net Change', 'value': latest_data.get('Net Change in Cash',
                                                                        latest_data['Operating Cash Flow'] +
                                                                        latest_data['Investing Cash Flow'] +
                                                                        latest_data['Financing Cash Flow']),
                     'isSum': True}
                ]

                fig = visualizer.plot_cash_flow_waterfall(
                    waterfall_data,
                    title="Cash Flow Components",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not extract cash flow components for visualization.")

            # Free Cash Flow Analysis
            st.markdown('<div class="section-header"><h4>Free Cash Flow Analysis</h4></div>',
                        unsafe_allow_html=True)

            if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                # Calculate FCF
                ocf = cash_flow.loc["Operating Cash Flow"].iloc[0]
                capex = cash_flow.loc["Capital Expenditure"].iloc[0]

                # Ensure capex is negative (for subtraction)
                capex_abs = abs(capex) if capex < 0 else -capex
                fcf = ocf - capex_abs

                # Create data for FCF breakdown
                fcf_data = [
                    {'category': 'Operating Cash Flow', 'value': ocf},
                    {'category': 'Capital Expenditure', 'value': -capex_abs},
                    {'category': 'Free Cash Flow', 'value': fcf, 'isSum': True}
                ]

                fig = visualizer.plot_free_cash_flow_breakdown(
                    fcf_data,
                    title="Free Cash Flow Breakdown",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not extract components for Free Cash Flow analysis.")

            # Cash Flow to Net Income Comparison
            st.markdown('<div class="section-header"><h4>Cash Flow vs. Net Income</h4></div>',
                        unsafe_allow_html=True)

            if "Operating Cash Flow" in cash_flow.index and "Net Income" in income_stmt.index:
                # Get historical data for both metrics
                ocf_series = cash_flow.loc["Operating Cash Flow"]
                ni_series = income_stmt.loc["Net Income"]

                # Create comparison DataFrame
                comp_periods = sorted(list(set(ocf_series.index).intersection(set(ni_series.index))))

                if comp_periods:
                    comp_data = pd.DataFrame({
                        'Operating Cash Flow': [ocf_series.get(p) for p in comp_periods],
                        'Net Income': [ni_series.get(p) for p in comp_periods]
                    }, index=comp_periods)

                    fig = visualizer.plot_cash_flow_vs_income(
                        comp_data,
                        title="Operating Cash Flow vs. Net Income",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No matching periods found for cash flow and income comparison.")
            else:
                st.info("Cash flow or net income data not available for comparison.")

        elif cash_flow_view == "Trend Analysis":
            # Trend analysis of cash flow items
            st.markdown('<div class="section-header"><h4>Cash Flow Trends</h4></div>', unsafe_allow_html=True)

            # Key metrics to track
            trend_metrics = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow',
                             'Free Cash Flow', 'Net Change in Cash']

            # Check if we have enough periods
            if cash_flow.shape[1] > 1:
                # Create DataFrames for visualization
                metrics_data = {}

                for metric in trend_metrics:
                    if metric in cash_flow.index:
                        metrics_data[metric] = cash_flow.loc[metric]
                    elif metric == 'Free Cash Flow':
                        # Calculate FCF if not directly available
                        if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                            ocf = cash_flow.loc["Operating Cash Flow"]
                            capex = cash_flow.loc["Capital Expenditure"]

                            # Calculate FCF (OCF - abs(CapEx))
                            fcf = ocf - capex.abs()
                            metrics_data[metric] = fcf

                if metrics_data:
                    # Create DataFrame
                    metrics_df = pd.DataFrame(metrics_data)

                    # Plot trends
                    fig = visualizer.plot_cash_flow_trends(
                        metrics_df,
                        title="Cash Flow Trends",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not extract key metrics for trend visualization.")

                # Cash Flow Ratios Trends
                st.markdown('<div class="section-header"><h4>Cash Flow Ratio Trends</h4></div>',
                            unsafe_allow_html=True)

                # Calculate cash flow ratios over time
                ratio_data = {}

                # OCF to Net Income
                if "Operating Cash Flow" in cash_flow.index and "Net Income" in income_stmt.index:
                    # Get matching periods
                    common_periods = sorted(list(set(cash_flow.columns).intersection(set(income_stmt.columns))))

                    if common_periods:
                        ocf = cash_flow.loc["Operating Cash Flow", common_periods]
                        ni = income_stmt.loc["Net Income", common_periods]

                        # Calculate ratio (avoid division by zero)
                        ocf_ni_ratio = pd.Series(index=common_periods)
                        for period in common_periods:
                            if ni[period] != 0:
                                ocf_ni_ratio[period] = ocf[period] / ni[period] * 100
                            else:
                                ocf_ni_ratio[period] = np.nan

                        ratio_data['OCF to Net Income (%)'] = ocf_ni_ratio

                # CapEx to OCF
                if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                    ocf = cash_flow.loc["Operating Cash Flow"]
                    capex = cash_flow.loc["Capital Expenditure"]

                    # Calculate ratio (avoid division by zero)
                    capex_ocf_ratio = pd.Series(index=cash_flow.columns)
                    for period in cash_flow.columns:
                        if ocf[period] != 0:
                            # Use absolute value of CapEx for ratio
                            capex_abs = abs(capex[period]) if capex[period] < 0 else capex[period]
                            capex_ocf_ratio[period] = capex_abs / ocf[period] * 100
                        else:
                            capex_ocf_ratio[period] = np.nan

                    ratio_data['CapEx to OCF (%)'] = capex_ocf_ratio

                # FCF to OCF
                if "Operating Cash Flow" in cash_flow.index:
                    ocf = cash_flow.loc["Operating Cash Flow"]

                    # Calculate FCF
                    if "Capital Expenditure" in cash_flow.index:
                        capex = cash_flow.loc["Capital Expenditure"]
                        fcf = ocf - capex.abs()

                        # Calculate ratio (avoid division by zero)
                        fcf_ocf_ratio = pd.Series(index=cash_flow.columns)
                        for period in cash_flow.columns:
                            if ocf[period] != 0:
                                fcf_ocf_ratio[period] = fcf[period] / ocf[period] * 100
                            else:
                                fcf_ocf_ratio[period] = np.nan

                        ratio_data['FCF to OCF (%)'] = fcf_ocf_ratio

                if ratio_data:
                    # Create DataFrame
                    ratio_df = pd.DataFrame(ratio_data)

                    # Plot ratio trends
                    fig = visualizer.plot_cash_flow_ratios(
                        ratio_df,
                        title="Cash Flow Ratio Trends",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display ratio table
                    st.markdown('<div class="section-header"><h4>Cash Flow Ratios</h4></div>',
                                unsafe_allow_html=True)

                    # Format ratios for display
                    ratio_display = ratio_df.copy()
                    for col in ratio_display.columns:
                        ratio_display[col] = ratio_display[col].apply(
                            lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                        )

                    st.table(ratio_display)
                else:
                    st.info("Could not calculate cash flow ratio trends from the available data.")
            else:
                st.info("Need at least two periods of data to perform trend analysis.")

    # Tab 4: Ratios
    with financial_tabs[3]:
        st.subheader("Financial Ratios Analysis")

        # Create financial ratio analyzer
        from models.ratio_analysis import FinancialRatioAnalyzer
        ratio_analyzer = FinancialRatioAnalyzer()

        # Calculate ratios
        ratios = ratio_analyzer.calculate_ratios(financial_data)

        # Get sector benchmarks if sector is known
        if sector != 'Unknown':
            sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector)
        else:
            sector_benchmarks = None

        # Create ratio categories and descriptions
        ratio_categories = [
            {
                "name": "Valuation",
                "description": "Valuation ratios compare a company's stock price to its financial performance.",
                "ratios": [
                    {"name": "P/E Ratio", "key": "pe_ratio",
                     "description": "Price to Earnings - Shows how much investors are willing to pay per dollar of earnings."},
                    {"name": "P/S Ratio", "key": "ps_ratio",
                     "description": "Price to Sales - Market value relative to annual revenue."},
                    {"name": "P/B Ratio", "key": "pb_ratio",
                     "description": "Price to Book - Market value relative to book value."},
                    {"name": "EV/EBITDA", "key": "ev_ebitda",
                     "description": "Enterprise Value to EBITDA - Company value relative to earnings before interest, taxes, depreciation, and amortization."},
                    {"name": "EV/Revenue", "key": "ev_revenue",
                     "description": "Enterprise Value to Revenue - Company value relative to revenue."}
                ]
            },
            {
                "name": "Profitability",
                "description": "Profitability ratios measure a company's ability to generate earnings relative to its expenses.",
                "ratios": [
                    {"name": "Gross Margin", "key": "gross_margin",
                     "description": "Gross Profit / Revenue - Measures the efficiency of production and pricing."},
                    {"name": "Operating Margin", "key": "operating_margin",
                     "description": "Operating Income / Revenue - Measures operational efficiency."},
                    {"name": "Net Margin", "key": "net_margin",
                     "description": "Net Income / Revenue - Measures overall profitability."},
                    {"name": "ROE", "key": "roe",
                     "description": "Return on Equity - Net Income / Shareholder Equity - Measures profitability relative to shareholder investment."},
                    {"name": "ROA", "key": "roa",
                     "description": "Return on Assets - Net Income / Total Assets - Measures how efficiently assets are used to generate profit."},
                    {"name": "ROIC", "key": "roic",
                     "description": "Return on Invested Capital - (Net Income - Dividends) / Invested Capital - Measures efficiency at allocating capital to profitable investments."}
                ]
            },
            {
                "name": "Liquidity",
                "description": "Liquidity ratios measure a company's ability to meet its short-term obligations.",
                "ratios": [
                    {"name": "Current Ratio", "key": "current_ratio",
                     "description": "Current Assets / Current Liabilities - Measures short-term solvency."},
                    {"name": "Quick Ratio", "key": "quick_ratio",
                     "description": "(Current Assets - Inventory) / Current Liabilities - A more stringent measure of liquidity."},
                    {"name": "Cash Ratio", "key": "cash_ratio",
                     "description": "Cash & Equivalents / Current Liabilities - Most conservative liquidity measure."}
                ]
            },
            {
                "name": "Leverage",
                "description": "Leverage ratios measure a company's debt levels and ability to meet debt obligations.",
                "ratios": [
                    {"name": "Debt-to-Equity", "key": "debt_to_equity",
                     "description": "Total Debt / Shareholder Equity - Measures financial leverage."},
                    {"name": "Debt-to-Assets", "key": "debt_to_assets",
                     "description": "Total Debt / Total Assets - Measures portion of assets financed with debt."},
                    {"name": "Interest Coverage", "key": "interest_coverage",
                     "description": "EBIT / Interest Expense - Measures ability to pay interest on outstanding debt."}
                ]
            },
            {
                "name": "Efficiency",
                "description": "Efficiency ratios measure how effectively a company is using its assets and managing its operations.",
                "ratios": [
                    {"name": "Asset Turnover", "key": "asset_turnover",
                     "description": "Revenue / Average Total Assets - Measures efficiency of asset utilization."},
                    {"name": "Inventory Turnover", "key": "inventory_turnover",
                     "description": "COGS / Average Inventory - Measures efficiency of inventory management."},
                    {"name": "Receivables Turnover", "key": "receivables_turnover",
                     "description": "Revenue / Average Accounts Receivable - Measures efficiency in collecting receivables."}
                ]
            },
            {
                "name": "Growth",
                "description": "Growth ratios measure a company's rate of growth in key financial metrics.",
                "ratios": [
                    {"name": "Revenue Growth", "key": "revenue_growth",
                     "description": "(Current Revenue / Previous Revenue) - 1 - Measures growth in sales."},
                    {"name": "Net Income Growth", "key": "net_income_growth",
                     "description": "(Current Net Income / Previous Net Income) - 1 - Measures growth in bottom line."}
                ]
            }
        ]

        # Create tabs for each ratio category
        ratio_tabs = st.tabs([category["name"] for category in ratio_categories])

        # Render each ratio category in its tab
        for i, category in enumerate(ratio_categories):
            with ratio_tabs[i]:
                st.markdown(f'<div class="section-header"><h4>{category["name"]} Ratios</h4></div>',
                            unsafe_allow_html=True)
                st.markdown(category["description"])

                # Check if we have data for this category
                category_ratios = ratios.get(category["name"].lower(), {})

                if category_ratios:
                    # Create columns for metric cards
                    num_columns = 3
                    ratio_metrics = []

                    # Prepare data for display
                    for ratio_info in category["ratios"]:
                        ratio_key = ratio_info["key"]
                        if ratio_key in category_ratios:
                            ratio_value = category_ratios[ratio_key]

                            # Get benchmark if available
                            benchmark = None
                            if sector_benchmarks:
                                sector_category = sector_benchmarks.get(category["name"].lower(), {})
                                benchmark = sector_category.get(ratio_key)

                            ratio_metrics.append({
                                "name": ratio_info["name"],
                                "value": ratio_value,
                                "benchmark": benchmark,
                                "description": ratio_info["description"]
                            })

                    # Check if we have any metrics to display
                    if ratio_metrics:
                        # Create metric cards in columns
                        cols = st.columns(num_columns)

                        for j, metric in enumerate(ratio_metrics):
                            col_idx = j % num_columns

                            with cols[col_idx]:
                                # Format value based on ratio type
                                if category["name"] == "Profitability" or category["name"] == "Growth":
                                    # Display as percentage
                                    formatted_value = f"{metric['value'] * 100:.2f}%"
                                else:
                                    # Display as decimal
                                    formatted_value = f"{metric['value']:.2f}"

                                # Determine color based on benchmark comparison
                                value_color = COLORS["secondary"]  # Default color

                                if metric["benchmark"] is not None:
                                    # Calculate percentage difference
                                    pct_diff = (metric["value"] / metric["benchmark"] - 1) * 100

                                    # Determine if higher is better for this category
                                    higher_is_better = True
                                    if category["name"] == "Valuation" or category["name"] == "Leverage":
                                        higher_is_better = False

                                    # Set color based on comparison
                                    if higher_is_better:
                                        if pct_diff > 10:
                                            value_color = COLORS["primary"]  # Good
                                        elif pct_diff < -10:
                                            value_color = COLORS["accent"]  # Poor
                                        else:
                                            value_color = COLORS["warning"]  # Neutral
                                    else:
                                        if pct_diff < -10:
                                            value_color = COLORS["primary"]  # Good
                                        elif pct_diff > 10:
                                            value_color = COLORS["accent"]  # Poor
                                        else:
                                            value_color = COLORS["warning"]  # Neutral

                                # Create the metric card
                                benchmark_text = ""
                                if metric["benchmark"] is not None:
                                    pct_diff = (metric["value"] / metric["benchmark"] - 1) * 100
                                    benchmark_text = f"<div style='font-size: 12px;'>Sector: {metric['benchmark']:.2f} ({pct_diff:+.1f}%)</div>"

                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-value" style="color: {value_color};">{formatted_value}</div>
                                        <div class="metric-label">{metric["name"]}</div>
                                        {benchmark_text}
                                        <div style='font-size: 11px; color: #a0a0a0; margin-top: 5px;'>{metric["description"]}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # Visualization of ratios
                        st.markdown('<div class="section-header"><h5>Visualization</h5></div>', unsafe_allow_html=True)

                        # Prepare data for visualization
                        vis_data = {metric["name"]: metric["value"] for metric in ratio_metrics}

                        # Get benchmark data if available
                        benchmark_data = None
                        if sector_benchmarks:
                            sector_category = sector_benchmarks.get(category["name"].lower(), {})
                            benchmark_data = {info["name"]: sector_category.get(info["key"])
                                              for info in category["ratios"]
                                              if info["key"] in sector_category}

                        # Create visualization based on category
                        if category["name"] == "Valuation":
                            fig = visualizer.plot_valuation_ratios(
                                vis_data,
                                benchmark_data=benchmark_data,
                                title=f"{category['name']} Ratios Comparison",
                                height=400
                            )
                        elif category["name"] == "Profitability":
                            fig = visualizer.plot_profitability_ratios(
                                vis_data,
                                benchmark_data=benchmark_data,
                                title=f"{category['name']} Ratios (%)",
                                height=400
                            )
                        else:
                            # Generic visualization for other categories
                            fig = visualizer.plot_financial_ratios(
                                vis_data,
                                benchmark_data=benchmark_data,
                                title=f"{category['name']} Ratios",
                                height=400
                            )

                        st.plotly_chart(fig, use_container_width=True)

                        # Show industry comparison if benchmarks are available
                        if sector_benchmarks and benchmark_data:
                            st.markdown('<div class="section-header"><h5>Industry Comparison</h5></div>',
                                        unsafe_allow_html=True)

                            # Create comparison table
                            comp_data = []
                            for metric in ratio_metrics:
                                if metric["benchmark"] is not None:
                                    pct_diff = (metric["value"] / metric["benchmark"] - 1) * 100
                                    comp_data.append({
                                        "Ratio": metric["name"],
                                        "Company": metric["value"],
                                        "Sector Average": metric["benchmark"],
                                        "Difference": f"{pct_diff:+.1f}%"
                                    })

                            if comp_data:
                                # Create DataFrame
                                comp_df = pd.DataFrame(comp_data)

                                # Format values
                                if category["name"] == "Profitability" or category["name"] == "Growth":
                                    comp_df["Company"] = comp_df["Company"].apply(lambda x: f"{x * 100:.2f}%")
                                    comp_df["Sector Average"] = comp_df["Sector Average"].apply(
                                        lambda x: f"{x * 100:.2f}%" if x else "N/A")
                                else:
                                    comp_df["Company"] = comp_df["Company"].apply(lambda x: f"{x:.2f}")
                                    comp_df["Sector Average"] = comp_df["Sector Average"].apply(
                                        lambda x: f"{x:.2f}" if x else "N/A")

                                # Display table
                                st.table(comp_df)
                    else:
                        st.info(f"No {category['name'].lower()} ratios available for this company.")
                else:
                    st.info(f"No {category['name'].lower()} ratios available for this company.")
        # Tab 5: Growth Analysis
        with financial_tabs[4]:
            st.subheader("Growth Analysis")

            # Check if we have enough historical data
            if (income_stmt is not None and income_stmt.shape[1] > 1 and
                    balance_sheet is not None and balance_sheet.shape[1] > 1):

                # Create selector for view options
                view_options = ["Revenue & Profit Growth", "Balance Sheet Growth", "Per Share Metrics",
                                "Compound Growth"]
                growth_view = st.radio("Select View:", view_options, horizontal=True, key="growth_view")

                if growth_view == "Revenue & Profit Growth":
                    # Revenue and profit growth analysis
                    st.markdown('<div class="section-header"><h4>Revenue & Profit Growth Trends</h4></div>',
                                unsafe_allow_html=True)

                    # Key metrics to track growth
                    growth_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']

                    # Calculate growth rates
                    growth_data = {}
                    for metric in growth_metrics:
                        if metric in income_stmt.index:
                            # Get values for all periods
                            values = income_stmt.loc[metric]

                            # Calculate year-over-year growth
                            growth = []
                            growth_pct = []
                            periods = list(values.index)

                            for i in range(1, len(periods)):
                                current = values[periods[i - 1]]
                                previous = values[periods[i]]

                                if previous != 0:
                                    growth_rate = (current / previous) - 1
                                    growth_pct.append(growth_rate * 100)
                                else:
                                    growth_pct.append(None)

                                growth.append({
                                    'period': periods[i - 1],
                                    'value': current,
                                    'previous': previous,
                                    'growth': growth_rate if previous != 0 else None
                                })

                            growth_data[metric] = {
                                'values': values.to_dict(),
                                'growth': growth,
                                'growth_pct': growth_pct,
                                'periods': periods[:-1]  # Remove last period as it has no growth rate
                            }

                    if growth_data:
                        # Create visualization
                        fig = visualizer.plot_revenue_profit_growth(
                            growth_data,
                            title="Revenue & Profit Growth",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display growth rates in table
                        st.markdown('<div class="section-header"><h4>Growth Rates (%)</h4></div>',
                                    unsafe_allow_html=True)

                        # Create DataFrame for growth rates
                        growth_df = pd.DataFrame()

                        for metric, data in growth_data.items():
                            if len(data['growth_pct']) > 0:
                                growth_df[metric] = pd.Series(data['growth_pct'], index=data['periods'])

                        if not growth_df.empty:
                            # Format for display
                            growth_display = growth_df.copy()
                            for col in growth_display.columns:
                                growth_display[col] = growth_display[col].apply(
                                    lambda x: f"{x:.2f}%" if x is not None else "N/A"
                                )

                            st.table(growth_display)

                            # Calculate CAGR for each metric
                            st.markdown('<div class="section-header"><h4>Compound Annual Growth Rate (CAGR)</h4></div>',
                                        unsafe_allow_html=True)
                            st.markdown(
                                "CAGR measures the annual growth rate over a period, smoothing out year-to-year volatility.")

                            cagr_data = []
                            for metric, data in growth_data.items():
                                values = data['values']
                                periods = list(values.keys())

                                if len(periods) >= 2:
                                    start_value = values[periods[-1]]  # Oldest period
                                    end_value = values[periods[0]]  # Most recent period
                                    years = len(periods) - 1

                                    if start_value > 0:
                                        cagr = (end_value / start_value) ** (1 / years) - 1
                                        cagr_data.append({
                                            "Metric": metric,
                                            "CAGR": f"{cagr * 100:.2f}%",
                                            "Start Period": periods[-1],
                                            "End Period": periods[0],
                                            "Start Value": f"${start_value / 1e9:.2f}B" if start_value >= 1e9 else f"${start_value / 1e6:.2f}M",
                                            "End Value": f"${end_value / 1e9:.2f}B" if end_value >= 1e9 else f"${end_value / 1e6:.2f}M"
                                        })

                            if cagr_data:
                                cagr_df = pd.DataFrame(cagr_data)
                                st.table(cagr_df)
                        else:
                            st.info("Could not calculate growth rates from the available data.")
                    else:
                        st.info("Insufficient data to analyze revenue and profit growth.")

                elif growth_view == "Balance Sheet Growth":
                    # Balance sheet growth analysis
                    st.markdown('<div class="section-header"><h4>Balance Sheet Growth Trends</h4></div>',
                                unsafe_allow_html=True)

                    # Key metrics to track
                    bs_metrics = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity',
                                  'Total Current Assets', 'Total Current Liabilities']

                    # Calculate growth rates
                    growth_data = {}
                    for metric in bs_metrics:
                        if metric in balance_sheet.index:
                            # Get values for all periods
                            values = balance_sheet.loc[metric]

                            # Calculate year-over-year growth
                            growth_pct = []
                            periods = list(values.index)

                            for i in range(1, len(periods)):
                                current = values[periods[i - 1]]
                                previous = values[periods[i]]

                                if previous != 0:
                                    growth_rate = (current / previous) - 1
                                    growth_pct.append(growth_rate * 100)
                                else:
                                    growth_pct.append(None)

                            growth_data[metric] = {
                                'values': values.to_dict(),
                                'growth_pct': growth_pct,
                                'periods': periods[:-1]  # Remove last period as it has no growth rate
                            }

                    if growth_data:
                        # Create visualization
                        fig = visualizer.plot_balance_sheet_growth(
                            growth_data,
                            title="Balance Sheet Growth",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display growth rates in table
                        st.markdown('<div class="section-header"><h4>Growth Rates (%)</h4></div>',
                                    unsafe_allow_html=True)

                        # Create DataFrame for growth rates
                        growth_df = pd.DataFrame()

                        for metric, data in growth_data.items():
                            if len(data['growth_pct']) > 0:
                                growth_df[metric] = pd.Series(data['growth_pct'], index=data['periods'])

                        if not growth_df.empty:
                            # Format for display
                            growth_display = growth_df.copy()
                            for col in growth_display.columns:
                                growth_display[col] = growth_display[col].apply(
                                    lambda x: f"{x:.2f}%" if x is not None else "N/A"
                                )

                            st.table(growth_display)
                        else:
                            st.info("Could not calculate growth rates from the available data.")
                    else:
                        st.info("Insufficient data to analyze balance sheet growth.")

                elif growth_view == "Per Share Metrics":
                    # Per share metrics growth
                    st.markdown('<div class="section-header"><h4>Per Share Metrics Growth</h4></div>',
                                unsafe_allow_html=True)

                    # Check if we have share count data
                    shares_outstanding = company_info.get('shares_outstanding')

                    if shares_outstanding:
                        # Calculate historical share count if possible
                        historical_shares = {}

                        # Try to get from income statement
                        if 'Diluted Weighted Average Shares' in income_stmt.index:
                            for period in income_stmt.columns:
                                historical_shares[period] = income_stmt.loc['Diluted Weighted Average Shares', period]
                        else:
                            # Use current share count for all periods
                            for period in income_stmt.columns:
                                historical_shares[period] = shares_outstanding

                        # Calculate per share metrics
                        per_share_data = {}

                        # EPS calculation
                        if 'Net Income' in income_stmt.index:
                            eps_values = {}
                            eps_growth = []
                            periods = list(income_stmt.columns)

                            for period in periods:
                                net_income = income_stmt.loc['Net Income', period]
                                shares = historical_shares[period]
                                if shares > 0:
                                    eps = net_income / shares
                                    eps_values[period] = eps

                            # Calculate EPS growth
                            for i in range(1, len(periods)):
                                current = eps_values.get(periods[i - 1])
                                previous = eps_values.get(periods[i])

                                if current is not None and previous is not None and previous != 0:
                                    growth_rate = (current / previous) - 1
                                    eps_growth.append({
                                        'period': periods[i - 1],
                                        'value': current,
                                        'previous': previous,
                                        'growth': growth_rate
                                    })

                            per_share_data['EPS'] = {
                                'values': eps_values,
                                'growth': eps_growth
                            }

                        # BVPS (Book Value Per Share) calculation
                        if 'Total Stockholder Equity' in balance_sheet.index:
                            bvps_values = {}
                            bvps_growth = []
                            periods = list(balance_sheet.columns)

                            for period in periods:
                                equity = balance_sheet.loc['Total Stockholder Equity', period]
                                shares = historical_shares.get(period)
                                if shares and shares > 0:
                                    bvps = equity / shares
                                    bvps_values[period] = bvps

                            # Calculate BVPS growth
                            for i in range(1, len(periods)):
                                current = bvps_values.get(periods[i - 1])
                                previous = bvps_values.get(periods[i])

                                if current is not None and previous is not None and previous != 0:
                                    growth_rate = (current / previous) - 1
                                    bvps_growth.append({
                                        'period': periods[i - 1],
                                        'value': current,
                                        'previous': previous,
                                        'growth': growth_rate
                                    })

                            per_share_data['BVPS'] = {
                                'values': bvps_values,
                                'growth': bvps_growth
                            }

                        # FCF Per Share calculation
                        if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                            fcfps_values = {}
                            fcfps_growth = []
                            periods = list(cash_flow.columns)

                            for period in periods:
                                ocf = cash_flow.loc["Operating Cash Flow", period]
                                capex = cash_flow.loc["Capital Expenditure", period]
                                capex_abs = abs(capex) if capex < 0 else capex
                                fcf = ocf - capex_abs

                                shares = historical_shares.get(period)
                                if shares and shares > 0:
                                    fcfps = fcf / shares
                                    fcfps_values[period] = fcfps

                            # Calculate FCFPS growth
                            for i in range(1, len(periods)):
                                current = fcfps_values.get(periods[i - 1])
                                previous = fcfps_values.get(periods[i])

                                if current is not None and previous is not None and previous != 0:
                                    growth_rate = (current / previous) - 1
                                    fcfps_growth.append({
                                        'period': periods[i - 1],
                                        'value': current,
                                        'previous': previous,
                                        'growth': growth_rate
                                    })

                            per_share_data['FCF/Share'] = {
                                'values': fcfps_values,
                                'growth': fcfps_growth
                            }

                        if per_share_data:
                            # Create visualization
                            fig = visualizer.plot_per_share_metrics(
                                per_share_data,
                                title="Per Share Metrics Growth",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Display per share metrics in table
                            st.markdown('<div class="section-header"><h4>Per Share Metrics</h4></div>',
                                        unsafe_allow_html=True)

                            # Create DataFrame for per share values
                            metrics_df = pd.DataFrame()

                            for metric, data in per_share_data.items():
                                metrics_df[metric] = pd.Series(data['values'])

                            if not metrics_df.empty:
                                # Format for display
                                metrics_display = metrics_df.copy()
                                for col in metrics_display.columns:
                                    metrics_display[col] = metrics_display[col].apply(
                                        lambda x: f"${x:.2f}" if x is not None else "N/A"
                                    )

                                st.table(metrics_display)

                                # Display growth rates
                                st.markdown('<div class="section-header"><h4>Per Share Growth Rates (%)</h4></div>',
                                            unsafe_allow_html=True)

                                # Create DataFrame for growth rates
                                growth_df = pd.DataFrame()

                                for metric, data in per_share_data.items():
                                    if 'growth' in data and data['growth']:
                                        growth_rates = {}
                                        for item in data['growth']:
                                            growth_rates[item['period']] = item['growth'] * 100

                                        growth_df[metric] = pd.Series(growth_rates)

                                if not growth_df.empty:
                                    # Format for display
                                    growth_display = growth_df.copy()
                                    for col in growth_display.columns:
                                        growth_display[col] = growth_display[col].apply(
                                            lambda x: f"{x:.2f}%" if x is not None else "N/A"
                                        )

                                    st.table(growth_display)
                            else:
                                st.info("Could not calculate per share metrics from the available data.")
                        else:
                            st.info("Could not calculate per share metrics from the available data.")
                    else:
                        st.info("Share count data is not available for per share metrics calculation.")

                elif growth_view == "Compound Growth":
                    # Compound growth analysis
                    st.markdown('<div class="section-header"><h4>Compound Annual Growth Rate (CAGR)</h4></div>',
                                unsafe_allow_html=True)
                    st.markdown("""
                        Compound Annual Growth Rate (CAGR) measures the mean annual growth rate of an investment over a specified time period.
                        It represents one of the most accurate ways to calculate and determine returns for anything that can rise or fall in value over time.
                    """)

                    # Select metrics for CAGR analysis
                    financial_metrics = {
                        "Income Statement": {
                            "Total Revenue": income_stmt.loc[
                                "Total Revenue"] if "Total Revenue" in income_stmt.index else None,
                            "Gross Profit": income_stmt.loc[
                                "Gross Profit"] if "Gross Profit" in income_stmt.index else None,
                            "Operating Income": income_stmt.loc[
                                "Operating Income"] if "Operating Income" in income_stmt.index else None,
                            "Net Income": income_stmt.loc["Net Income"] if "Net Income" in income_stmt.index else None
                        },
                        "Balance Sheet": {
                            "Total Assets": balance_sheet.loc[
                                "Total Assets"] if "Total Assets" in balance_sheet.index else None,
                            "Total Stockholder Equity": balance_sheet.loc[
                                "Total Stockholder Equity"] if "Total Stockholder Equity" in balance_sheet.index else None,
                            "Total Current Assets": balance_sheet.loc[
                                "Total Current Assets"] if "Total Current Assets" in balance_sheet.index else None
                        },
                        "Cash Flow": {
                            "Operating Cash Flow": cash_flow.loc[
                                "Operating Cash Flow"] if "Operating Cash Flow" in cash_flow.index else None,
                            "Free Cash Flow": None  # Calculate below
                        }
                    }

                    # Calculate Free Cash Flow if possible
                    if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                        ocf = cash_flow.loc["Operating Cash Flow"]
                        capex = cash_flow.loc["Capital Expenditure"]
                        capex_abs = capex.abs() if (capex < 0).any() else capex
                        fcf = ocf - capex_abs
                        financial_metrics["Cash Flow"]["Free Cash Flow"] = fcf

                    # Calculate CAGR for each metric
                    cagr_data = []

                    for category, metrics in financial_metrics.items():
                        for metric_name, metric_values in metrics.items():
                            if metric_values is not None and not metric_values.empty:
                                periods = list(metric_values.index)

                                if len(periods) >= 2:
                                    start_value = metric_values[periods[-1]]  # Oldest period
                                    end_value = metric_values[periods[0]]  # Most recent period
                                    years = len(periods) - 1

                                    if start_value > 0:
                                        cagr = (end_value / start_value) ** (1 / years) - 1
                                        cagr_data.append({
                                            "Category": category,
                                            "Metric": metric_name,
                                            "CAGR": cagr * 100,
                                            "Start Period": periods[-1],
                                            "End Period": periods[0],
                                            "Years": years,
                                            "Start Value": start_value,
                                            "End Value": end_value
                                        })

                    if cagr_data:
                        # Create DataFrame
                        cagr_df = pd.DataFrame(cagr_data)

                        # Sort by CAGR
                        cagr_df = cagr_df.sort_values("CAGR", ascending=False)

                        # Create visualization
                        fig = visualizer.plot_cagr_comparison(
                            cagr_df,
                            title="Compound Annual Growth Rate (CAGR)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display CAGR table
                        st.markdown('<div class="section-header"><h4>CAGR by Metric</h4></div>', unsafe_allow_html=True)

                        # Format for display
                        display_df = cagr_df.copy()
                        display_df["CAGR"] = display_df["CAGR"].apply(lambda x: f"{x:.2f}%")
                        display_df["Start Value"] = display_df["Start Value"].apply(
                            lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(
                                x) >= 1e6 else f"${x:.2f}"
                        )
                        display_df["End Value"] = display_df["End Value"].apply(
                            lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(
                                x) >= 1e6 else f"${x:.2f}"
                        )

                        # Display only relevant columns
                        display_columns = ["Category", "Metric", "CAGR", "Start Period", "End Period", "Start Value",
                                           "End Value"]

                        st.table(display_df[display_columns])

                        # Analysis of growth sustainability
                        st.markdown('<div class="section-header"><h4>Growth Sustainability Analysis</h4></div>',
                                    unsafe_allow_html=True)

                        # Compare growth rates of key metrics
                        if all(x in cagr_df["Metric"].values for x in ["Total Revenue", "Net Income"]):
                            revenue_cagr = cagr_df[cagr_df["Metric"] == "Total Revenue"]["CAGR"].values[0]
                            income_cagr = cagr_df[cagr_df["Metric"] == "Net Income"]["CAGR"].values[0]

                            if income_cagr > revenue_cagr:
                                st.markdown("""
                                    <div class="metric-card" style="padding: 20px;">
                                        <div style="font-weight: bold; color: #74f174;">Net Income is growing faster than Revenue</div>
                                        <div style="margin-top: 10px;">This suggests improving profit margins and operational efficiency, which is a positive sign for sustainable growth.</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="metric-card" style="padding: 20px;">
                                        <div style="font-weight: bold; color: #fff59d;">Revenue is growing faster than Net Income</div>
                                        <div style="margin-top: 10px;">This suggests declining profit margins or increased costs, which may indicate challenges in scaling profitably.</div>
                                    </div>
                                """, unsafe_allow_html=True)

                        # Compare revenue growth to asset growth
                        if all(x in cagr_df["Metric"].values for x in ["Total Revenue", "Total Assets"]):
                            revenue_cagr = cagr_df[cagr_df["Metric"] == "Total Revenue"]["CAGR"].values[0]
                            assets_cagr = cagr_df[cagr_df["Metric"] == "Total Assets"]["CAGR"].values[0]

                            if revenue_cagr > assets_cagr:
                                st.markdown("""
                                    <div class="metric-card" style="padding: 20px;">
                                        <div style="font-weight: bold; color: #74f174;">Revenue is growing faster than Assets</div>
                                        <div style="margin-top: 10px;">This indicates improving asset utilization and efficiency, suggesting the company is generating more revenue from its asset base.</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="metric-card" style="padding: 20px;">
                                        <div style="font-weight: bold; color: #fff59d;">Assets are growing faster than Revenue</div>
                                        <div style="margin-top: 10px;">This might indicate declining asset efficiency or significant investments for future growth.</div>
                                    </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Insufficient data to calculate compound growth rates.")
            else:
                st.info("Need at least two periods of financial data to perform growth analysis.")


def render_financial_analysis_page():
    """
    Render the financial analysis page
    """
    st.title("Financial Analysis")
    st.write("Analyze a company's financial statements, ratios, and growth metrics.")

    # Let the user input a ticker
    ticker = st.text_input("Enter a ticker symbol:", "AAPL")

    if ticker:
        # Initialize DataLoader
        data_loader = DataLoader()

        # Load data for the company
        try:
            with st.spinner(f"Loading financial data for {ticker}..."):
                # Get financial statements
                income_stmt = data_loader.get_financial_statements(ticker, 'income', 'annual')
                balance_sheet = data_loader.get_financial_statements(ticker, 'balance', 'annual')
                cash_flow = data_loader.get_financial_statements(ticker, 'cash', 'annual')

                # Get company info
                company_info = data_loader.get_company_info(ticker)

                # Get historical price data (last 1 year)
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                price_data = data_loader.get_historical_prices(ticker, start_date, end_date)

                # Prepare financial data
                financial_data = {
                    'income_statement': income_stmt,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'market_data': {
                        'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                        'market_cap': company_info.get('market_cap')
                    }
                }

                # Render the financial analysis
                render_company_financials(ticker, financial_data, company_info, price_data)

        except Exception as e:
            st.error(f"An error occurred while loading data for {ticker}: {e}")
            st.info("Please check the ticker symbol and try again.")


def render_compound_growth_analysis(income_stmt, balance_sheet, cash_flow):
    """
    Render the compound growth analysis section
    """
    # Select metrics for CAGR analysis
    financial_metrics = {
        "Income Statement": {
            "Total Revenue": income_stmt.loc["Total Revenue"] if "Total Revenue" in income_stmt.index else None,
            "Gross Profit": income_stmt.loc["Gross Profit"] if "Gross Profit" in income_stmt.index else None,
            "Operating Income": income_stmt.loc[
                "Operating Income"] if "Operating Income" in income_stmt.index else None,
            "Net Income": income_stmt.loc["Net Income"] if "Net Income" in income_stmt.index else None
        },
        "Balance Sheet": {
            "Total Assets": balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else None,
            "Total Stockholder Equity": balance_sheet.loc[
                "Total Stockholder Equity"] if "Total Stockholder Equity" in balance_sheet.index else None,
            "Total Current Assets": balance_sheet.loc[
                "Total Current Assets"] if "Total Current Assets" in balance_sheet.index else None
        },
        "Cash Flow": {
            "Operating Cash Flow": cash_flow.loc[
                "Operating Cash Flow"] if "Operating Cash Flow" in cash_flow.index else None,
            "Free Cash Flow": None  # Calculate below
        }
    }

    # Calculate Free Cash Flow if possible
    if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
        ocf = cash_flow.loc["Operating Cash Flow"]
        capex = cash_flow.loc["Capital Expenditure"]
        capex_abs = capex.abs() if (capex < 0).any() else capex
        fcf = ocf - capex_abs
        financial_metrics["Cash Flow"]["Free Cash Flow"] = fcf

    # Calculate CAGR for each metric
    cagr_data = []

    for category, metrics in financial_metrics.items():
        for metric_name, metric_values in metrics.items():
            if metric_values is not None and not metric_values.empty:
                periods = list(metric_values.index)

                if len(periods) >= 2:
                    start_value = metric_values[periods[-1]]  # Oldest period
                    end_value = metric_values[periods[0]]  # Most recent period
                    years = len(periods) - 1

                    if start_value > 0:
                        cagr = (end_value / start_value) ** (1 / years) - 1
                        cagr_data.append({
                            "Category": category,
                            "Metric": metric_name,
                            "CAGR": cagr * 100,
                            "Start Period": periods[-1],
                            "End Period": periods[0],
                            "Years": years,
                            "Start Value": start_value,
                            "End Value": end_value
                        })

    if cagr_data:
        # Create DataFrame
        cagr_df = pd.DataFrame(cagr_data)

        # Sort by CAGR
        cagr_df = cagr_df.sort_values("CAGR", ascending=False)

        # Create visualization
        fig = visualizer.plot_cagr_comparison(
            cagr_df,
            title="Compound Annual Growth Rate (CAGR)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display CAGR table
        st.markdown('<div class="section-header"><h4>CAGR by Metric</h4></div>', unsafe_allow_html=True)

        # Format for display
        display_df = cagr_df.copy()
        display_df["CAGR"] = display_df["CAGR"].apply(lambda x: f"{x:.2f}%")
        display_df["Start Value"] = display_df["Start Value"].apply(
            lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else f"${x:.2f}"
        )
        display_df["End Value"] = display_df["End Value"].apply(
            lambda x: f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else f"${x:.2f}"
        )

        # Display only relevant columns
        display_columns = ["Category", "Metric", "CAGR", "Start Period", "End Period", "Start Value", "End Value"]

        st.table(display_df[display_columns])

        # Analysis of growth sustainability
        st.markdown('<div class="section-header"><h4>Growth Sustainability Analysis</h4></div>', unsafe_allow_html=True)

        # Compare growth rates of key metrics
        if all(x in cagr_df["Metric"].values for x in ["Total Revenue", "Net Income"]):
            revenue_cagr = cagr_df[cagr_df["Metric"] == "Total Revenue"]["CAGR"].values[0]
            income_cagr = cagr_df[cagr_df["Metric"] == "Net Income"]["CAGR"].values[0]

            if income_cagr > revenue_cagr:
                st.markdown("""
                    <div class="metric-card" style="padding: 20px;">
                        <div style="font-weight: bold; color: #74f174;">Net Income is growing faster than Revenue</div>
                        <div style="margin-top: 10px;">This suggests improving profit margins and operational efficiency, which is a positive sign for sustainable growth.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="metric-card" style="padding: 20px;">
                        <div style="font-weight: bold; color: #fff59d;">Revenue is growing faster than Net Income</div>
                        <div style="margin-top: 10px;">This suggests declining profit margins or increased costs, which may indicate challenges in scaling profitably.</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data to calculate compound growth rates.")


def render_growth_analysis(income_stmt, balance_sheet, cash_flow, company_info):
    """
    Render the growth analysis tab
    """
    st.subheader("Growth Analysis")

    # Check if we have enough historical data
    if (income_stmt is not None and income_stmt.shape[1] > 1 and
            balance_sheet is not None and balance_sheet.shape[1] > 1):

        # Create selector for view options
        view_options = ["Revenue & Profit Growth", "Balance Sheet Growth", "Per Share Metrics", "Compound Growth"]
        growth_view = st.radio("Select View:", view_options, horizontal=True, key="growth_view")

        if growth_view == "Revenue & Profit Growth":
            # Revenue and profit growth analysis
            st.markdown('<div class="section-header"><h4>Revenue & Profit Growth Trends</h4></div>',
                        unsafe_allow_html=True)

            # Key metrics to track growth
            growth_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']

            # Calculate growth rates
            growth_data = {}
            for metric in growth_metrics:
                if metric in income_stmt.index:
                    # Get values for all periods
                    values = income_stmt.loc[metric]

                    # Calculate year-over-year growth
                    growth = []
                    growth_pct = []
                    periods = list(values.index)

                    for i in range(1, len(periods)):
                        current = values[periods[i - 1]]
                        previous = values[periods[i]]

                        if previous != 0:
                            growth_rate = (current / previous) - 1
                            growth_pct.append(growth_rate * 100)
                        else:
                            growth_pct.append(None)

                        growth.append({
                            'period': periods[i - 1],
                            'value': current,
                            'previous': previous,
                            'growth': growth_rate if previous != 0 else None
                        })

                    growth_data[metric] = {
                        'values': values.to_dict(),
                        'growth': growth,
                        'growth_pct': growth_pct,
                        'periods': periods[:-1]  # Remove last period as it has no growth rate
                    }

            if growth_data:
                # Create visualization
                fig = visualizer.plot_revenue_profit_growth(
                    growth_data,
                    title="Revenue & Profit Growth",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display growth rates in table
                st.markdown('<div class="section-header"><h4>Growth Rates (%)</h4></div>', unsafe_allow_html=True)

                # Create DataFrame for growth rates
                growth_df = pd.DataFrame()

                for metric, data in growth_data.items():
                    if len(data['growth_pct']) > 0:
                        growth_df[metric] = pd.Series(data['growth_pct'], index=data['periods'])

                if not growth_df.empty:
                    # Format for display
                    growth_display = growth_df.copy()
                    for col in growth_display.columns:
                        growth_display[col] = growth_display[col].apply(
                            lambda x: f"{x:.2f}%" if x is not None else "N/A"
                        )

                    st.table(growth_display)

                # Calculate CAGR for each metric
                st.markdown('<div class="section-header"><h4>Compound Annual Growth Rate (CAGR)</h4></div>',
                            unsafe_allow_html=True)
                st.markdown(
                    "CAGR measures the annual growth rate over a period, smoothing out year-to-year volatility.")

                cagr_data = []
                for metric, data in growth_data.items():
                    values = data['values']
                    periods = list(values.keys())

                    if len(periods) >= 2:
                        start_value = values[periods[-1]]  # Oldest period
                        end_value = values[periods[0]]  # Most recent period
                        years = len(periods) - 1

                        if start_value > 0:
                            cagr = (end_value / start_value) ** (1 / years) - 1
                            cagr_data.append({
                                "Metric": metric,
                                "CAGR": f"{cagr * 100:.2f}%",
                                "Start Period": periods[-1],
                                "End Period": periods[0],
                                "Start Value": f"${start_value / 1e9:.2f}B" if start_value >= 1e9 else f"${start_value / 1e6:.2f}M",
                                "End Value": f"${end_value / 1e9:.2f}B" if end_value >= 1e9 else f"${end_value / 1e6:.2f}M"
                            })

                if cagr_data:
                    cagr_df = pd.DataFrame(cagr_data)
                    st.table(cagr_df)
            else:
                st.info("Could not calculate growth rates from the available data.")

        elif growth_view == "Balance Sheet Growth":
            # Balance sheet growth analysis
            st.markdown('<div class="section-header"><h4>Balance Sheet Growth Trends</h4></div>',
                        unsafe_allow_html=True)

            # Key metrics to track
            bs_metrics = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity',
                          'Total Current Assets', 'Total Current Liabilities']

            # Calculate growth rates
            growth_data = {}
            for metric in bs_metrics:
                if metric in balance_sheet.index:
                    # Get values for all periods
                    values = balance_sheet.loc[metric]

                    # Calculate year-over-year growth
                    growth_pct = []
                    periods = list(values.index)

                    for i in range(1, len(periods)):
                        current = values[periods[i - 1]]
                        previous = values[periods[i]]

                        if previous != 0:
                            growth_rate = (current / previous) - 1
                            growth_pct.append(growth_rate * 100)
                        else:
                            growth_pct.append(None)

                    growth_data[metric] = {
                        'values': values.to_dict(),
                        'growth_pct': growth_pct,
                        'periods': periods[:-1]  # Remove last period as it has no growth rate
                    }

            if growth_data:
                # Create visualization
                fig = visualizer.plot_balance_sheet_growth(
                    growth_data,
                    title="Balance Sheet Growth",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display growth rates in table
                st.markdown('<div class="section-header"><h4>Growth Rates (%)</h4></div>', unsafe_allow_html=True)

                # Create DataFrame for growth rates
                growth_df = pd.DataFrame()

                for metric, data in growth_data.items():
                    if len(data['growth_pct']) > 0:
                        growth_df[metric] = pd.Series(data['growth_pct'], index=data['periods'])

                if not growth_df.empty:
                    # Format for display
                    growth_display = growth_df.copy()
                    for col in growth_display.columns:
                        growth_display[col] = growth_display[col].apply(
                            lambda x: f"{x:.2f}%" if x is not None else "N/A"
                        )

                    st.table(growth_display)
            else:
                st.info("Could not calculate growth rates from the available data.")

        elif growth_view == "Per Share Metrics":
            # Per share metrics growth
            st.markdown('<div class="section-header"><h4>Per Share Metrics Growth</h4></div>', unsafe_allow_html=True)

            # Check if we have share count data
            shares_outstanding = company_info.get('shares_outstanding')

            if shares_outstanding:
                # Calculate historical share count if possible
                historical_shares = {}

                # Try to get from income statement
                if 'Diluted Weighted Average Shares' in income_stmt.index:
                    for period in income_stmt.columns:
                        historical_shares[period] = income_stmt.loc['Diluted Weighted Average Shares', period]
                else:
                    # Use current share count for all periods
                    for period in income_stmt.columns:
                        historical_shares[period] = shares_outstanding

                # Calculate per share metrics
                per_share_data = {}

                # EPS calculation
                if 'Net Income' in income_stmt.index:
                    eps_values = {}
                    eps_growth = []
                    periods = list(income_stmt.columns)

                    for period in periods:
                        net_income = income_stmt.loc['Net Income', period]
                        shares = historical_shares[period]
                        if shares > 0:
                            eps = net_income / shares
                            eps_values[period] = eps

                    # Calculate EPS growth
                    for i in range(1, len(periods)):
                        current = eps_values.get(periods[i - 1])
                        previous = eps_values.get(periods[i])

                        if current is not None and previous is not None and previous != 0:
                            growth_rate = (current / previous) - 1
                            eps_growth.append({
                                'period': periods[i - 1],
                                'value': current,
                                'previous': previous,
                                'growth': growth_rate
                            })

                    per_share_data['EPS'] = {
                        'values': eps_values,
                        'growth': eps_growth
                    }

                # BVPS (Book Value Per Share) calculation
                if 'Total Stockholder Equity' in balance_sheet.index:
                    bvps_values = {}
                    bvps_growth = []
                    periods = list(balance_sheet.columns)

                    for period in periods:
                        equity = balance_sheet.loc['Total Stockholder Equity', period]
                        shares = historical_shares.get(period)
                        if shares and shares > 0:
                            bvps = equity / shares
                            bvps_values[period] = bvps

                    # Calculate BVPS growth
                    for i in range(1, len(periods)):
                        current = bvps_values.get(periods[i - 1])
                        previous = bvps_values.get(periods[i])

                        if current is not None and previous is not None and previous != 0:
                            growth_rate = (current / previous) - 1
                            bvps_growth.append({
                                'period': periods[i - 1],
                                'value': current,
                                'previous': previous,
                                'growth': growth_rate
                            })

                    per_share_data['BVPS'] = {
                        'values': bvps_values,
                        'growth': bvps_growth
                    }

                # FCF Per Share calculation
                if all(x in cash_flow.index for x in ["Operating Cash Flow", "Capital Expenditure"]):
                    fcfps_values = {}
                    fcfps_growth = []
                    periods = list(cash_flow.columns)

                    for period in periods:
                        ocf = cash_flow.loc["Operating Cash Flow", period]
                        capex = cash_flow.loc["Capital Expenditure", period]
                        capex_abs = abs(capex) if capex < 0 else capex
                        fcf = ocf - capex_abs

                        shares = historical_shares.get(period)
                        if shares and shares > 0:
                            fcfps = fcf / shares
                            fcfps_values[period] = fcfps

                    # Calculate FCFPS growth
                    for i in range(1, len(periods)):
                        current = fcfps_values.get(periods[i - 1])
                        previous = fcfps_values.get(periods[i])

                        if current is not None and previous is not None and previous != 0:
                            growth_rate = (current / previous) - 1
                            fcfps_growth.append({
                                'period': periods[i - 1],
                                'value': current,
                                'previous': previous,
                                'growth': growth_rate
                            })

                    per_share_data['FCF/Share'] = {
                        'values': fcfps_values,
                        'growth': fcfps_growth
                    }

                if per_share_data:
                    # Create visualization
                    fig = visualizer.plot_per_share_metrics(
                        per_share_data,
                        title="Per Share Metrics Growth",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display per share metrics in table
                    st.markdown('<div class="section-header"><h4>Per Share Metrics</h4></div>', unsafe_allow_html=True)

                    # Create DataFrame for per share values
                    metrics_df = pd.DataFrame()

                    for metric, data in per_share_data.items():
                        metrics_df[metric] = pd.Series(data['values'])

                    if not metrics_df.empty:
                        # Format for display
                        metrics_display = metrics_df.copy()
                        for col in metrics_display.columns:
                            metrics_display[col] = metrics_display[col].apply(
                                lambda x: f"${x:.2f}" if x is not None else "N/A"
                            )

                        st.table(metrics_display)

                    # Display growth rates
                    st.markdown('<div class="section-header"><h4>Per Share Growth Rates (%)</h4></div>',
                                unsafe_allow_html=True)

                    # Create DataFrame for growth rates
                    growth_df = pd.DataFrame()

                    for metric, data in per_share_data.items():
                        if 'growth' in data and data['growth']:
                            growth_rates = {}
                            for item in data['growth']:
                                growth_rates[item['period']] = item['growth'] * 100

                            growth_df[metric] = pd.Series(growth_rates)

                    if not growth_df.empty:
                        # Format for display
                        growth_display = growth_df.copy()
                        for col in growth_display.columns:
                            growth_display[col] = growth_display[col].apply(
                                lambda x: f"{x:.2f}%" if x is not None else "N/A"
                            )

                        st.table(growth_display)
                else:
                    st.info("Could not calculate per share metrics from the available data.")
            else:
                st.info("Share count data is not available for per share metrics calculation.")

        elif growth_view == "Compound Growth":
            # Compound growth analysis
            st.markdown('<div class="section-header"><h4>Compound Annual Growth Rate (CAGR)</h4></div>',
                        unsafe_allow_html=True)
            st.markdown("""
                Compound Annual Growth Rate (CAGR) measures the mean annual growth rate of an investment over a specified time period.
                It represents one of the most accurate ways to calculate and determine returns for anything that can rise or fall in value over time.
            """)

            # Call the compound growth analysis function
            render_compound_growth_analysis(income_stmt, balance_sheet, cash_flow)
    else:
        st.info("Insufficient historical data to perform growth analysis.")


def render_company_financials(ticker, financial_data, company_info, price_data):
    """
    Render the financial analysis for a specific company
    """
    income_stmt = financial_data.get('income_statement')
    balance_sheet = financial_data.get('balance_sheet')
    cash_flow = financial_data.get('cash_flow')

    # Create tabs for different financial statements
    financial_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios", "Growth Analysis"])

    # Tab 5: Growth Analysis
    with financial_tabs[4]:
        render_growth_analysis(income_stmt, balance_sheet, cash_flow, company_info)

    # Other tabs implementation would go here...


# Execute this when the script is run directly
if __name__ == "__main__":
    render_financial_analysis_page()