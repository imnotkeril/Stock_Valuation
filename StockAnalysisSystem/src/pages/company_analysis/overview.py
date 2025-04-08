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
from models.financial_statements import FinancialStatementAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('company_overview')


def render_company_overview(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                            price_data: pd.DataFrame, sector_data: Optional[Dict] = None):
    """
    Render the company overview page

    Args:
        ticker: Company ticker symbol
        financial_data: Dictionary containing financial statement data
        company_info: Dictionary containing company information
        price_data: DataFrame with historical price data
        sector_data: Dictionary containing sector data (optional)
    """
    # Display company header
    st.header(f"{company_info.get('name', ticker)} ({ticker})")

    # Get sector and industry
    sector = company_info.get('sector', 'Unknown')
    industry = company_info.get('industry', 'Unknown')

    # Company metadata row with custom styling
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sector</div>
                <div class="metric-value" style="color: {COLORS['sectors'].get(sector, COLORS['secondary'])}">
                    {sector}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Industry</div>
                <div class="metric-value">{industry}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        exchange = company_info.get('exchange', 'N/A')
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Exchange</div>
                <div class="metric-value">{exchange}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        currency = company_info.get('currency', 'USD')
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Currency</div>
                <div class="metric-value">{currency}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Price metrics row
    latest_price = price_data['Close'].iloc[-1] if not price_data.empty else None
    price_change = price_data['Close'].pct_change().iloc[-1] * 100 if not price_data.empty else None
    price_change_30d = price_data['Close'].pct_change(30).iloc[-1] * 100 if not price_data.empty and len(
        price_data) > 30 else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price_color = COLORS['primary']
        price_display = f"${latest_price:.2f}" if latest_price else "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value" style="color: {price_color}">{price_display}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if price_change is not None:
            change_color = COLORS['primary'] if price_change >= 0 else COLORS['accent']
            change_icon = "↗" if price_change >= 0 else "↘"
            change_display = f"{change_icon} {price_change:.2f}%"
        else:
            change_color = COLORS['info']
            change_display = "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">1-Day Change</div>
                <div class="metric-value" style="color: {change_color}">{change_display}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        if price_change_30d is not None:
            change_color = COLORS['primary'] if price_change_30d >= 0 else COLORS['accent']
            change_icon = "↗" if price_change_30d >= 0 else "↘"
            change_display = f"{change_icon} {price_change_30d:.2f}%"
        else:
            change_color = COLORS['info']
            change_display = "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">30-Day Change</div>
                <div class="metric-value" style="color: {change_color}">{change_display}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        market_cap = company_info.get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                cap_display = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_display = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                cap_display = f"${market_cap / 1e6:.2f}M"
            else:
                cap_display = f"${market_cap:.2f}"
        else:
            cap_display = "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{cap_display}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create tabs for different sections
    tabs = st.tabs(["Overview", "Price Chart", "Key Financials", "About"])

    # Tab: Overview
    with tabs[0]:
        # Two columns layout
        col1, col2 = st.columns([3, 2])

        with col1:
            # Overview card with quick stats
            st.subheader("Quick Overview")

            # Create financial ratio analyzer
            ratio_analyzer = FinancialRatioAnalyzer()

            # Calculate key financial ratios
            ratios = ratio_analyzer.calculate_ratios(financial_data)

            # Get sector benchmarks if sector is known
            if sector != 'Unknown':
                sector_benchmarks = ratio_analyzer.get_sector_benchmarks(sector)
            else:
                sector_benchmarks = None

            # Display key ratios in a dashboard
            key_metrics = [
                {
                    "category": "Valuation",
                    "metrics": [
                        {"name": "P/E Ratio", "key": "pe_ratio", "category": "valuation", "format": ".2f"},
                        {"name": "P/S Ratio", "key": "ps_ratio", "category": "valuation", "format": ".2f"},
                        {"name": "P/B Ratio", "key": "pb_ratio", "category": "valuation", "format": ".2f"}
                    ]
                },
                {
                    "category": "Profitability",
                    "metrics": [
                        {"name": "Gross Margin", "key": "gross_margin", "category": "profitability", "format": ".1%"},
                        {"name": "Operating Margin", "key": "operating_margin", "category": "profitability",
                         "format": ".1%"},
                        {"name": "Net Margin", "key": "net_margin", "category": "profitability", "format": ".1%"}
                    ]
                },
                {
                    "category": "Returns",
                    "metrics": [
                        {"name": "ROE", "key": "roe", "category": "profitability", "format": ".1%"},
                        {"name": "ROA", "key": "roa", "category": "profitability", "format": ".1%"}
                    ]
                },
                {
                    "category": "Financial Health",
                    "metrics": [
                        {"name": "Current Ratio", "key": "current_ratio", "category": "liquidity", "format": ".2f"},
                        {"name": "Debt-to-Equity", "key": "debt_to_equity", "category": "leverage", "format": ".2f"}
                    ]
                }
            ]

            # Display metric categories
            for metric_group in key_metrics:
                # Display category header
                st.markdown(f"<div class='section-header'><h4>{metric_group['category']}</h4></div>",
                            unsafe_allow_html=True)

                # Create columns for metrics
                metric_cols = st.columns(len(metric_group["metrics"]))

                # Display metrics in columns
                for i, metric in enumerate(metric_group["metrics"]):
                    with metric_cols[i]:
                        # Get metric value
                        value = ratios.get(metric["category"], {}).get(metric["key"])
                        benchmark = sector_benchmarks.get(metric["category"], {}).get(
                            metric["key"]) if sector_benchmarks else None

                        # Format value
                        if value is not None:
                            if "%" in metric["format"]:
                                # Percentage format
                                formatted_value = f"{value:{metric['format']}}"
                            else:
                                # Regular format
                                formatted_value = f"{value:{metric['format']}}"
                        else:
                            formatted_value = "N/A"

                        # Compare to benchmark
                        if benchmark is not None and value is not None:
                            # Determine if higher or lower is better for this metric
                            higher_is_better = True
                            # Valuation metrics where lower is better
                            if metric["category"] == "valuation" or (
                                    metric["category"] == "leverage" and metric["key"] == "debt_to_equity"):
                                higher_is_better = False

                            # Calculate percentage difference
                            perc_diff = (value / benchmark - 1) * 100

                            # Determine color based on comparison
                            if (higher_is_better and perc_diff > 10) or (not higher_is_better and perc_diff < -10):
                                color = COLORS["primary"]  # positive
                            elif (higher_is_better and perc_diff < -10) or (not higher_is_better and perc_diff > 10):
                                color = COLORS["accent"]  # negative
                            else:
                                color = COLORS["warning"]  # neutral

                            # Create comparison indicator
                            comparison = f"<span style='color:{color};font-size:12px;'>{perc_diff:+.1f}% vs sector</span>"
                        else:
                            comparison = ""

                        # Display metric card
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-value">{formatted_value}</div>
                                <div class="metric-label">{metric["name"]}</div>
                                {comparison}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Display company description (truncated)
            st.subheader("Business Description")
            description = company_info.get('description', "No description available.")
            if len(description) > 500:
                st.markdown(f"{description[:500]}...")
                st.markdown(f"<span style='color:{COLORS['info']};cursor:pointer;'>Read more</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(description)

        with col2:
            # Recent performance card
            st.subheader("Performance")

            # Create performance table
            if not price_data.empty:
                # Calculate performance metrics
                perf_data = {
                    "Period": ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD"],
                    "Return": [
                        price_data['Close'].pct_change(5).iloc[-1] * 100 if len(price_data) > 5 else None,
                        price_data['Close'].pct_change(20).iloc[-1] * 100 if len(price_data) > 20 else None,
                        price_data['Close'].pct_change(60).iloc[-1] * 100 if len(price_data) > 60 else None,
                        price_data['Close'].pct_change(125).iloc[-1] * 100 if len(price_data) > 125 else None,
                        price_data['Close'].pct_change(252).iloc[-1] * 100 if len(price_data) > 252 else None,
                        price_data['Close'].iloc[-1] /
                        price_data[price_data.index.year == price_data.index[-1].year].iloc[0]['Close'] * 100 - 100
                    ]
                }

                # Create DataFrame
                perf_df = pd.DataFrame(perf_data)

                # Remove None values
                perf_df = perf_df.dropna()

                # Format return column
                perf_df["Color"] = perf_df["Return"].apply(lambda x: COLORS["primary"] if x >= 0 else COLORS["accent"])
                perf_df["Formatted Return"] = perf_df.apply(
                    lambda
                        row: f"<span style='color:{row['Color']}'>{'+' if row['Return'] >= 0 else ''}{row['Return']:.2f}%</span>",
                    axis=1
                )

                # Display formatted table
                for i, row in perf_df.iterrows():
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid #333;">
                            <span>{row['Period']}</span>
                            {row['Formatted Return']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No price data available to calculate performance.")

            # Trading information
            st.subheader("Trading Information")

            trading_info = [
                {"label": "52-Week High", "value": company_info.get('fifty_two_week_high')},
                {"label": "52-Week Low", "value": company_info.get('fifty_two_week_low')},
                {"label": "Average Volume", "value": company_info.get('average_volume')},
                {"label": "Beta", "value": company_info.get('beta')},
                {"label": "Dividend Yield", "value": company_info.get('dividend_yield')}
            ]

            # Display trading information
            for info in trading_info:
                if info["value"] is not None:
                    # Format value based on type
                    if isinstance(info["value"], float):
                        if info["label"] == "Dividend Yield" and info["value"] > 0:
                            formatted_value = f"{info['value']:.2%}"
                        elif info["label"] in ["52-Week High", "52-Week Low"]:
                            formatted_value = f"${info['value']:.2f}"
                        elif info["label"] == "Average Volume":
                            formatted_value = f"{info['value']:,.0f}"
                        else:
                            formatted_value = f"{info['value']:.2f}"
                    else:
                        formatted_value = str(info["value"])

                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid #333;">
                            <span>{info['label']}</span>
                            <span>{formatted_value}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Tab: Price Chart
    with tabs[1]:
        # Create visualizer
        visualizer = FinancialVisualizer()

        # Configure chart options
        st.subheader("Stock Price Chart")

        # Chart controls
        col1, col2, col3 = st.columns(3)

        with col1:
            chart_type = st.selectbox("Chart Type:", ["Line", "Candlestick", "OHLC"], index=0)

        with col2:
            show_volume = st.checkbox("Show Volume", value=True)

        with col3:
            show_ma = st.checkbox("Show Moving Averages", value=True)

        # MA periods if selected
        if show_ma:
            ma_periods = st.multiselect("Moving Average Periods:", [5, 10, 20, 50, 100, 200], default=[50, 200])
        else:
            ma_periods = []

        # Plot stock price chart
        if not price_data.empty:
            fig = visualizer.plot_stock_price(
                price_data,
                ticker,
                company_name=company_info.get('name'),
                chart_type=chart_type.lower(),
                ma_periods=ma_periods,
                volume=show_volume,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price data available to display chart.")

        # Technical indicators (simplified)
        if not price_data.empty and len(price_data) > 14:
            st.subheader("Technical Indicators")

            # Calculate RSI
            delta = price_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate MACD
            ema12 = price_data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = price_data['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()

            # Calculate Bollinger Bands
            sma20 = price_data['Close'].rolling(window=20).mean()
            std20 = price_data['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)

            # Get last values
            last_rsi = rsi.iloc[-1]
            last_macd = macd.iloc[-1]
            last_signal = signal.iloc[-1]
            last_upper = upper_band.iloc[-1]
            last_lower = lower_band.iloc[-1]
            last_close = price_data['Close'].iloc[-1]

            # Display indicators
            indicators_col1, indicators_col2, indicators_col3 = st.columns(3)

            with indicators_col1:
                # RSI
                rsi_color = COLORS["accent"] if last_rsi > 70 else COLORS["primary"] if last_rsi < 30 else COLORS[
                    "warning"]
                rsi_status = "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {rsi_color}">{last_rsi:.1f}</div>
                        <div class="metric-label">RSI (14) - {rsi_status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with indicators_col2:
                # MACD
                macd_color = COLORS["primary"] if last_macd > last_signal else COLORS["accent"]
                macd_status = "Bullish" if last_macd > last_signal else "Bearish"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {macd_color}">{last_macd:.2f}</div>
                        <div class="metric-label">MACD - {macd_status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with indicators_col3:
                # Bollinger Bands
                bb_position = (last_close - last_lower) / (last_upper - last_lower) * 100
                bb_color = COLORS["accent"] if bb_position > 80 else COLORS["primary"] if bb_position < 20 else COLORS[
                    "warning"]
                bb_status = "Upper Band" if bb_position > 80 else "Lower Band" if bb_position < 20 else "Middle"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {bb_color}">{bb_position:.1f}%</div>
                        <div class="metric-label">BB Position - {bb_status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Tab: Key Financials
    with tabs[2]:
        # Financials overview
        st.subheader("Financial Summary")

        # Create statement analyzer
        statement_analyzer = FinancialStatementAnalyzer()

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Check if we have financial data
        if income_stmt is not None and not income_stmt.empty:
            # Analyze income statement
            income_analysis = statement_analyzer.analyze_income_statement(income_stmt)

            # Display income statement summary
            st.markdown('<div class="section-header"><h4>Revenue & Earnings</h4></div>', unsafe_allow_html=True)

            # Extract key metrics
            last_periods = min(4, income_stmt.shape[1])
            periods = income_stmt.columns[:last_periods]

            # Get key income metrics
            metrics = [
                {"name": "Revenue", "row": "Total Revenue"},
                {"name": "Gross Profit", "row": "Gross Profit"},
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

            # Calculate growth rate
            if len(periods) >= 2:
                growth_data = []
                for metric in metrics:
                    if metric["row"] in income_stmt.index:
                        current = income_stmt.loc[metric["row"], periods[0]]
                        previous = income_stmt.loc[metric["row"], periods[1]]
                        if current is not None and previous is not None and previous != 0:
                            growth_rate = (current / previous - 1) * 100
                            growth_data.append(f"{growth_rate:+.1f}%")
                        else:
                            growth_data.append("N/A")
                    else:
                        growth_data.append("N/A")

                # Add growth column
                income_display["YoY Growth"] = growth_data

            # Display the table
            st.dataframe(income_display, use_container_width=True)

            # Plot revenue trend
            revenue_data = income_stmt.loc["Total Revenue"] if "Total Revenue" in income_stmt.index else None
            net_income_data = income_stmt.loc["Net Income"] if "Net Income" in income_stmt.index else None

            if revenue_data is not None and net_income_data is not None:
                # Create visualizer
                visualizer = FinancialVisualizer()

                # Create DataFrame for plotting
                plot_data = pd.DataFrame({
                    "Revenue": revenue_data,
                    "Net Income": net_income_data
                })

                # Plot financial trends
                fig = visualizer.plot_financial_trends(
                    plot_data,
                    title="Revenue & Net Income Trend",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No income statement data available.")

        # Balance Sheet Summary
        if balance_sheet is not None and not balance_sheet.empty:
            # Balance sheet analysis
            balance_analysis = statement_analyzer.analyze_balance_sheet(balance_sheet)

            # Display balance sheet summary
            st.markdown('<div class="section-header"><h4>Balance Sheet Summary</h4></div>', unsafe_allow_html=True)

            # Key balance sheet metrics
            bs_metrics = [
                {"name": "Total Assets", "row": "Total Assets"},
                {"name": "Total Liabilities", "row": "Total Liabilities"},
                {"name": "Shareholders' Equity", "row": "Total Stockholder Equity"},
                {"name": "Debt-to-Equity", "type": "calculated"}
            ]

            # Last period for display
            last_period = balance_sheet.columns[0]

            # Create columns for display
            bs_col1, bs_col2 = st.columns(2)

            # Display metrics in columns
            with bs_col1:
                for i in range(0, len(bs_metrics) // 2 + len(bs_metrics) % 2):
                    metric = bs_metrics[i]
                    if metric["type"] == "calculated" and metric["name"] == "Debt-to-Equity":
                        # Calculate D/E ratio
                        if "Total Liabilities" in balance_sheet.index and "Total Stockholder Equity" in balance_sheet.index:
                            liabilities = balance_sheet.loc["Total Liabilities", last_period]
                            equity = balance_sheet.loc["Total Stockholder Equity", last_period]
                            if equity != 0:
                                value = liabilities / equity
                                formatted_value = f"{value:.2f}"
                            else:
                                formatted_value = "N/A"
                        else:
                            formatted_value = "N/A"
                    else:
                        # Regular metric
                        if metric["row"] in balance_sheet.index:
                            value = balance_sheet.loc[metric["row"], last_period]
                            if abs(value) >= 1e9:
                                formatted_value = f"${value / 1e9:.2f}B"
                            elif abs(value) >= 1e6:
                                formatted_value = f"${value / 1e6:.2f}M"
                            else:
                                formatted_value = f"${value:.2f}"
                        else:
                            formatted_value = "N/A"

                    # Display metric
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{formatted_value}</div>
                            <div class="metric-label">{metric["name"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Display metrics in second column
            with bs_col2:
                for i in range(len(bs_metrics) // 2 + len(bs_metrics) % 2, len(bs_metrics)):
                    metric = bs_metrics[i]
                    if metric["type"] == "calculated":
                        # Custom calculation
                        formatted_value = "N/A"
                    else:
                        # Regular metric
                        if metric["row"] in balance_sheet.index:
                            value = balance_sheet.loc[metric["row"], last_period]
                            if abs(value) >= 1e9:
                                formatted_value = f"${value / 1e9:.2f}B"
                            elif abs(value) >= 1e6:
                                formatted_value = f"${value / 1e6:.2f}M"
                            else:
                                formatted_value = f"${value:.2f}"
                        else:
                            formatted_value = "N/A"

                    # Display metric
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{formatted_value}</div>
                            <div class="metric-label">{metric["name"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Add asset composition chart
            if "Total Assets" in balance_sheet.index and balance_sheet.loc["Total Assets", last_period] > 0:
                # Find key asset components
                asset_components = {
                    "Current Assets": balance_sheet.loc[
                        "Total Current Assets", last_period] if "Total Current Assets" in balance_sheet.index else 0,
                    "Property & Equipment": balance_sheet.loc[
                        "Property Plant and Equipment", last_period] if "Property Plant and Equipment" in balance_sheet.index else 0,
                    "Intangible Assets": balance_sheet.loc[
                        "Intangible Assets", last_period] if "Intangible Assets" in balance_sheet.index else 0,
                    "Investments": balance_sheet.loc[
                        "Investments", last_period] if "Investments" in balance_sheet.index else 0,
                    "Other Assets": 0  # Calculate as remainder
                }

                # Calculate Other Assets
                total_assets = balance_sheet.loc["Total Assets", last_period]
                specified_assets = sum(asset_components.values())
                asset_components["Other Assets"] = max(0, total_assets - specified_assets)

                # Create pie chart
                fig = visualizer.plot_balance_sheet_composition(
                    asset_components,
                    "Asset Composition",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No balance sheet data available.")

        # Cash Flow Summary
        if cash_flow is not None and not cash_flow.empty:
            # Cash flow analysis
            cf_analysis = statement_analyzer.analyze_cash_flow(cash_flow)

            # Display cash flow summary
            st.markdown('<div class="section-header"><h4>Cash Flow Summary</h4></div>', unsafe_allow_html=True)

            # Key cash flow metrics
            cf_metrics = [
                {"name": "Operating Cash Flow", "row": "Operating Cash Flow"},
                {"name": "Capital Expenditure", "row": "Capital Expenditure"},
                {"name": "Free Cash Flow", "type": "calculated"},
                {"name": "Dividends Paid", "row": "Dividends Paid"}
            ]

            # Last periods for display
            last_periods = min(2, cash_flow.shape[1])
            cf_periods = cash_flow.columns[:last_periods]

            # Create dataframe for display
            cf_df = pd.DataFrame(index=[m["name"] for m in cf_metrics])

            # Fill data
            for period in cf_periods:
                period_data = []
                for metric in cf_metrics:
                    if metric["type"] == "calculated" and metric["name"] == "Free Cash Flow":
                        # Calculate FCF
                        if "Operating Cash Flow" in cash_flow.index and "Capital Expenditure" in cash_flow.index:
                            ocf = cash_flow.loc["Operating Cash Flow", period]
                            capex = cash_flow.loc["Capital Expenditure", period]
                            fcf = ocf - abs(capex)
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

            # Calculate growth rate
            if len(cf_periods) >= 2:
                growth_data = []
                for metric in cf_metrics:
                    if metric["type"] == "calculated" and metric["name"] == "Free Cash Flow":
                        # Calculate FCF growth
                        if "Operating Cash Flow" in cash_flow.index and "Capital Expenditure" in cash_flow.index:
                            ocf_current = cash_flow.loc["Operating Cash Flow", cf_periods[0]]
                            capex_current = cash_flow.loc["Capital Expenditure", cf_periods[0]]
                            fcf_current = ocf_current - abs(capex_current)

                            ocf_previous = cash_flow.loc["Operating Cash Flow", cf_periods[1]]
                            capex_previous = cash_flow.loc["Capital Expenditure", cf_periods[1]]
                            fcf_previous = ocf_previous - abs(capex_previous)

                            if fcf_previous != 0:
                                growth_rate = (fcf_current / fcf_previous - 1) * 100
                                growth_data.append(f"{growth_rate:+.1f}%")
                            else:
                                growth_data.append("N/A")
                        else:
                            growth_data.append("N/A")
                    elif metric["row"] in cash_flow.index:
                        current = cash_flow.loc[metric["row"], cf_periods[0]]
                        previous = cash_flow.loc[metric["row"], cf_periods[1]]
                        if current is not None and previous is not None and previous != 0:
                            growth_rate = (current / previous - 1) * 100
                            growth_data.append(f"{growth_rate:+.1f}%")
                        else:
                            growth_data.append("N/A")
                    else:
                        growth_data.append("N/A")

                # Add growth column
                cf_display["YoY Growth"] = growth_data

            # Display the table
            st.dataframe(cf_display, use_container_width=True)

            # Plot cash flow components
            ocf_data = cash_flow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cash_flow.index else None
            capex_data = cash_flow.loc["Capital Expenditure"] if "Capital Expenditure" in cash_flow.index else None

            if ocf_data is not None and capex_data is not None:
                # Create DataFrame for plotting
                fcf_data = ocf_data - abs(capex_data)
                plot_data = pd.DataFrame({
                    "Operating Cash Flow": ocf_data,
                    "Capital Expenditure": abs(capex_data),
                    "Free Cash Flow": fcf_data
                })

                # Plot cash flow trends
                fig = visualizer.plot_cash_flow_trends(
                    plot_data,
                    title="Cash Flow Components",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cash flow data available.")

    # Tab: About Company
    with tabs[3]:
        # Company description
        st.subheader("Company Profile")
        description = company_info.get('description', "No description available.")
        st.markdown(description)

        # Company metadata
        st.subheader("Company Information")

        # Create columns for info display
        info_col1, info_col2 = st.columns(2)

        # Define company info details
        company_details = [
            {"label": "Company Name", "value": company_info.get('name')},
            {"label": "Ticker", "value": ticker},
            {"label": "Sector", "value": sector},
            {"label": "Industry", "value": industry},
            {"label": "Exchange", "value": company_info.get('exchange')},
            {"label": "Currency", "value": company_info.get('currency')},
            {"label": "Country", "value": company_info.get('country')},
            {"label": "Website", "value": company_info.get('website')},
            {"label": "Employees", "value": company_info.get('employees')},
            {"label": "CEO", "value": company_info.get('ceo')},
            {"label": "Founded", "value": company_info.get('founded')}
        ]

        # Display company details in two columns
        for i, detail in enumerate(company_details):
            # Choose column based on index
            col = info_col1 if i < len(company_details) // 2 + len(company_details) % 2 else info_col2

            # Format value
            value = detail["value"] if detail["value"] is not None else "N/A"
            if detail["label"] == "Employees" and isinstance(value, (int, float)):
                value = f"{value:,.0f}"

            # Display as table row
            with col:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid #333;">
                        <span style="color:#a0a0a0;">{detail["label"]}</span>
                        <span>{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Add company address if available
        address_parts = [
            company_info.get('address'),
            company_info.get('city'),
            company_info.get('state'),
            company_info.get('zip'),
            company_info.get('country')
        ]

        # Filter out None values and join with comma
        address = ', '.join([part for part in address_parts if part])

        if address:
            st.subheader("Address")
            st.markdown(address)


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

    render_company_overview(ticker, financial_data, company_info, price_data)