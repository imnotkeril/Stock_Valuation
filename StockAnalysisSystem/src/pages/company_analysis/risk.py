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
from models.bankruptcy_models import BankruptcyAnalyzer
from models.financial_statements import FinancialStatementAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('company_risk')


def render_company_risk(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                        price_data: pd.DataFrame, sector_data: Optional[Dict] = None):
    """
    Render the company risk analysis page

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
    st.header(f"Risk Analysis: {company_name} ({ticker})")
    st.markdown(f"<p style='color: {COLORS['sectors'].get(sector, COLORS['secondary'])}'>Sector: {sector}</p>",
                unsafe_allow_html=True)

    # Create visualization helper
    visualizer = FinancialVisualizer()

    # Create tabs for different risk analyses
    risk_tabs = st.tabs(["Financial Health", "Bankruptcy Risk", "Market Risk", "Scenario Analysis"])

    # Tab 1: Financial Health
    with risk_tabs[0]:
        st.subheader("Financial Health Assessment")

        # Create financial statement analyzer
        statement_analyzer = FinancialStatementAnalyzer()

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Check if we have financial data
        if (income_stmt is not None and not income_stmt.empty and
                balance_sheet is not None and not balance_sheet.empty and
                cash_flow is not None and not cash_flow.empty):

            # Perform financial health analysis
            income_analysis = statement_analyzer.analyze_income_statement(income_stmt)
            balance_analysis = statement_analyzer.analyze_balance_sheet(balance_sheet)
            cash_flow_analysis = statement_analyzer.analyze_cash_flow(cash_flow)

            # Calculate financial health score
            health_score = statement_analyzer.calculate_financial_health_score(
                income_analysis, balance_analysis, cash_flow_analysis
            )

            # Display overall financial health score gauge
            if health_score and health_score.get('overall_score') is not None:
                # Create columns for summary and gauge
                health_col1, health_col2 = st.columns([2, 3])

                with health_col1:
                    # Display overall score
                    overall_score = health_score.get('overall_score')

                    # Determine color based on score
                    if overall_score >= 80:
                        score_color = COLORS["primary"]
                        assessment = "Excellent"
                    elif overall_score >= 65:
                        score_color = COLORS["primary"]
                        assessment = "Good"
                    elif overall_score >= 50:
                        score_color = COLORS["warning"]
                        assessment = "Moderate"
                    elif overall_score >= 35:
                        score_color = COLORS["accent"]
                        assessment = "Weak"
                    else:
                        score_color = COLORS["accent"]
                        assessment = "Poor"

                    st.markdown(
                        f"""
                                <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                                    <h2 style="margin: 0; color: {score_color};">{overall_score}/100</h2>
                                    <p style="margin: 5px 0; font-size: 18px;">Financial Health Score</p>
                                    <p style="margin: 0; color: {score_color}; font-size: 16px;">{assessment}</p>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                    # Add assessment summary
                    st.markdown("### Assessment Summary")
                    st.markdown(health_score.get('assessment_summary', 'No assessment available.'))

                with health_col2:
                    # Display gauge chart
                    fig = visualizer.plot_financial_health_score(
                        health_score,
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display health score components
                st.markdown("### Financial Health Components")

                # Get score components
                components = health_score.get('components', {})

                # Create columns for components
                comp_cols = st.columns(4)

                # Define component info
                components_info = [
                    {
                        "name": "Profitability",
                        "value": components.get('profitability'),
                        "description": "Ability to generate earnings relative to expenses"
                    },
                    {
                        "name": "Liquidity",
                        "value": components.get('liquidity'),
                        "description": "Ability to meet short-term obligations"
                    },
                    {
                        "name": "Solvency",
                        "value": components.get('solvency'),
                        "description": "Long-term financial stability and debt management"
                    },
                    {
                        "name": "Cash Flow",
                        "value": components.get('cash_flow'),
                        "description": "Generation and quality of cash flow from operations"
                    }
                ]

                # Display component cards
                for i, component in enumerate(components_info):
                    with comp_cols[i]:
                        value = component['value']
                        if value is not None:
                            # Determine color based on score
                            if value >= 80:
                                comp_color = COLORS["primary"]
                                comp_assessment = "Excellent"
                            elif value >= 65:
                                comp_color = COLORS["primary"]
                                comp_assessment = "Good"
                            elif value >= 50:
                                comp_color = COLORS["warning"]
                                comp_assessment = "Moderate"
                            elif value >= 35:
                                comp_color = COLORS["accent"]
                                comp_assessment = "Weak"
                            else:
                                comp_color = COLORS["accent"]
                                comp_assessment = "Poor"

                            st.markdown(
                                f"""
                                        <div class="metric-card">
                                            <div class="metric-value" style="color: {comp_color};">{value}/100</div>
                                            <div class="metric-label">{component['name']}</div>
                                            <div style="font-size: 12px; color: {comp_color};">{comp_assessment}</div>
                                        </div>
                                        """,
                                unsafe_allow_html=True
                            )

                            # Add tooltip/expandable description
                            with st.expander(f"About {component['name']} Score"):
                                st.markdown(component['description'])
                        else:
                            st.markdown(
                                f"""
                                        <div class="metric-card">
                                            <div class="metric-value">N/A</div>
                                            <div class="metric-label">{component['name']}</div>
                                        </div>
                                        """,
                                unsafe_allow_html=True
                            )

                # Display detailed analysis and recommendations
                st.markdown("### Key Strengths and Weaknesses")

                # Get strengths and weaknesses
                strengths = health_score.get('strengths', [])
                weaknesses = health_score.get('weaknesses', [])

                # Create columns for strengths and weaknesses
                strength_col, weakness_col = st.columns(2)

                with strength_col:
                    st.markdown(
                        f"""
                                <div style="background-color: {COLORS['primary']}10; padding: 15px; border-radius: 10px; 
                                           border-left: 4px solid {COLORS['primary']};">
                                    <h4 style="margin: 0; color: {COLORS['primary']};">Strengths</h4>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                    if strengths:
                        for strength in strengths:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("No significant strengths identified.")

                with weakness_col:
                    st.markdown(
                        f"""
                                <div style="background-color: {COLORS['accent']}10; padding: 15px; border-radius: 10px; 
                                           border-left: 4px solid {COLORS['accent']};">
                                    <h4 style="margin: 0; color: {COLORS['accent']};">Weaknesses</h4>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                    if weaknesses:
                        for weakness in weaknesses:
                            st.markdown(f"- {weakness}")
                    else:
                        st.markdown("No significant weaknesses identified.")

                # Recommendations
                st.markdown("### Recommendations")
                recommendations = health_score.get('recommendations', [])

                if recommendations:
                    for recommendation in recommendations:
                        st.markdown(
                            f"""
                                    <div style="background-color: #1f1f1f; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                        <p style="margin: 0;">📌 {recommendation}</p>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No specific recommendations available.")
            else:
                st.warning("Financial health score could not be calculated due to insufficient financial data.")
        else:
            st.warning("Insufficient financial data to perform health assessment.")

        # Tab 2: Bankruptcy Risk
    with risk_tabs[1]:
        st.subheader("Bankruptcy Risk Analysis")

        # Create bankruptcy analyzer
        bankruptcy_analyzer = BankruptcyAnalyzer()

        # Check if we have financial data for bankruptcy analysis
        if (income_stmt is not None and not income_stmt.empty and
                balance_sheet is not None and not balance_sheet.empty):

            # Perform bankruptcy risk assessment
            risk_assessment = bankruptcy_analyzer.get_comprehensive_risk_assessment(
                financial_data,
                sector
            )

            if risk_assessment and 'models' in risk_assessment:
                # Show overall risk assessment
                overall_risk = risk_assessment.get('overall_assessment')
                overall_color = risk_assessment.get('overall_color', '#ffffff')
                overall_desc = risk_assessment.get('overall_description', '')

                st.markdown(
                    f"""
                            <div style="background-color: {overall_color}20; padding: 20px; border-radius: 10px; 
                                      text-align: center; margin-bottom: 20px; border: 1px solid {overall_color}50;">
                                <h2 style="margin: 0; color: {overall_color};">{overall_risk}</h2>
                                <p style="margin: 10px 0;">{overall_desc}</p>
                            </div>
                            """,
                    unsafe_allow_html=True
                )

                # Display model scores
                st.markdown("### Bankruptcy Risk Models")

                # Get model results
                models = risk_assessment.get('models', {})

                # Plot models as gauge charts
                fig = visualizer.plot_bankruptcy_risk(
                    risk_assessment,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display each model's result with explanation
                for model_name, model_data in models.items():
                    score = model_data.get('score')
                    interpretation = model_data.get('interpretation', '')
                    risk_level = model_data.get('risk_level', '')
                    color = model_data.get('color', '#ffffff')
                    description = model_data.get('description', '')

                    # Create expandable section for each model
                    with st.expander(f"{model_name} - {risk_level}"):
                        st.markdown(
                            f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <h4 style="margin: 0;">{model_name}</h4>
                                            <p style="margin: 5px 0; color: #a0a0a0;">{description}</p>
                                        </div>
                                        <div style="text-align: right;">
                                            <h3 style="margin: 0; color: {color};">{score:.2f}</h3>
                                            <p style="margin: 0; color: {color};">{risk_level}</p>
                                        </div>
                                    </div>
                                    <hr style="margin: 10px 0; border-color: #333;">
                                    <p>{interpretation}</p>
                                    """,
                            unsafe_allow_html=True
                        )

                # Show financial red flags
                st.markdown("### Financial Red Flags")
                red_flags = risk_assessment.get('red_flags', [])

                if red_flags:
                    for flag in red_flags:
                        st.markdown(
                            f"""
                                    <div style="background-color: {COLORS['accent']}10; padding: 10px; border-radius: 5px; 
                                              margin: 5px 0; border-left: 4px solid {COLORS['accent']};">
                                        <p style="margin: 0;">🚩 {flag}</p>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        f"""
                                <div style="background-color: {COLORS['primary']}10; padding: 10px; border-radius: 5px; 
                                          margin: 5px 0; border-left: 4px solid {COLORS['primary']};">
                                    <p style="margin: 0;">✓ No significant financial red flags detected</p>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                # Add limitations note
                st.markdown("""
                        <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-top: 20px;">
                            <h4 style="margin: 0;">Limitations of Bankruptcy Models</h4>
                            <p style="margin: 10px 0; font-size: 14px; color: #a0a0a0;">
                                Bankruptcy prediction models provide an estimate of financial distress risk, but have limitations:
                                <ul style="margin: 5px 0; color: #a0a0a0;">
                                    <li>Models are based on historical data and may not capture new market dynamics</li>
                                    <li>Industry-specific factors may not be fully accounted for</li>
                                    <li>Non-financial factors like management quality are not considered</li>
                                    <li>These assessments should be part of a broader investment analysis</li>
                                </ul>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Bankruptcy risk assessment could not be performed due to insufficient data.")
        else:
            st.warning("Insufficient financial data to perform bankruptcy risk analysis.")

        # Tab 3: Market Risk
    with risk_tabs[2]:
        st.subheader("Market Risk Analysis")

        # Check if we have price data
        if price_data is not None and not price_data.empty and len(price_data) > 30:
            # Calculate market risk metrics

            # Calculate daily returns
            price_data['Daily Return'] = price_data['Close'].pct_change()

            # Create columns for volatility and beta
            market_col1, market_col2 = st.columns(2)

            with market_col1:
                # Volatility Analysis
                st.markdown('<div class="section-header"><h4>Volatility Analysis</h4></div>', unsafe_allow_html=True)

                # Calculate volatility
                daily_volatility = price_data['Daily Return'].std()
                annualized_volatility = daily_volatility * (252 ** 0.5) * 100  # Annualized, convert to percentage

                # Get sector volatility if available
                sector_volatility = None
                if sector_data is not None and not sector_data.empty:
                    sector_data['Daily Return'] = sector_data['Close'].pct_change()
                    sector_daily_volatility = sector_data['Daily Return'].std()
                    sector_volatility = sector_daily_volatility * (252 ** 0.5) * 100

                # Display volatility
                vol_color = COLORS["accent"] if annualized_volatility > 30 else COLORS[
                    "warning"] if annualized_volatility > 20 else COLORS["primary"]

                st.markdown(
                    f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {vol_color};">{annualized_volatility:.2f}%</div>
                                <div class="metric-label">Annualized Volatility</div>
                                <div style="font-size: 12px; color: #a0a0a0;">
                                    {f"Sector Volatility: {sector_volatility:.2f}%" if sector_volatility else ""}
                                </div>
                            </div>
                            """,
                    unsafe_allow_html=True
                )

                # Volatility explanation
                st.markdown("#### Understanding Volatility")

                # Determine volatility level relative to market
                if annualized_volatility > 40:
                    vol_description = "extremely high"
                    vol_text = "This indicates a very risky stock with potential for large price swings."
                elif annualized_volatility > 30:
                    vol_description = "very high"
                    vol_text = "This indicates a high-risk stock with potential for significant price movements."
                elif annualized_volatility > 20:
                    vol_description = "high"
                    vol_text = "This indicates an above-average risk level with substantial price fluctuations."
                elif annualized_volatility > 15:
                    vol_description = "moderate"
                    vol_text = "This indicates an average risk level, typical for many stocks."
                elif annualized_volatility > 10:
                    vol_description = "low"
                    vol_text = "This indicates a relatively stable stock with below-average price movements."
                else:
                    vol_description = "very low"
                    vol_text = "This indicates a highly stable stock with minimal price fluctuations."

                st.markdown(
                    f"""
                            <div style="background-color: {vol_color}10; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <p style="margin: 0;">This stock has <strong>{vol_description} volatility</strong>. {vol_text}</p>
                            </div>
                            """,
                    unsafe_allow_html=True
                )

                # Display historical volatility chart
                st.markdown("#### Historical Volatility")

                # Calculate rolling volatility (21-day window, approximately 1 month)
                price_data['Rolling Volatility'] = price_data['Daily Return'].rolling(window=21).std() * (
                            252 ** 0.5) * 100

                # Plot rolling volatility
                fig = visualizer.plot_volatility(
                    price_data,
                    ticker,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            with market_col2:
                # Beta Analysis
                st.markdown('<div class="section-header"><h4>Beta Analysis</h4></div>', unsafe_allow_html=True)

                # Get beta from company info
                beta = company_info.get('beta')

                if beta is not None:
                    # Determine beta characteristics
                    if beta > 1.5:
                        beta_description = "very high"
                        beta_color = COLORS["accent"]
                        beta_text = "This indicates a stock that tends to move significantly more than the market."
                    elif beta > 1.1:
                        beta_description = "high"
                        beta_color = COLORS["warning"]
                        beta_text = "This indicates a stock that tends to move more than the market."
                    elif beta > 0.9:
                        beta_description = "market-like"
                        beta_color = COLORS["warning"]
                        beta_text = "This indicates a stock that tends to move in line with the market."
                    elif beta > 0.6:
                        beta_description = "low"
                        beta_color = COLORS["primary"]
                        beta_text = "This indicates a stock that tends to move less than the market."
                    elif beta > 0:
                        beta_description = "very low"
                        beta_color = COLORS["primary"]
                        beta_text = "This indicates a stock that has very little correlation with market movements."
                    elif beta < 0:
                        beta_description = "negative"
                        beta_color = COLORS["secondary"]
                        beta_text = "This indicates a stock that tends to move in the opposite direction of the market."

                    # Display beta
                    st.markdown(
                        f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: {beta_color};">{beta:.2f}</div>
                                    <div class="metric-label">Beta (vs. Market)</div>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                    # Beta explanation
                    st.markdown("#### Understanding Beta")
                    st.markdown(
                        f"""
                                <div style="background-color: {beta_color}10; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                    <p style="margin: 0;">This stock has <strong>{beta_description} beta</strong>. {beta_text}</p>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                    # Beta interpretation table
                    st.markdown("#### Beta Interpretation")
                    st.markdown("""
                            | Beta Range | Meaning | Market Relation |
                            | --- | --- | --- |
                            | β > 1.5 | Very High | Moves much more than the market |
                            | 1.1 < β < 1.5 | High | Moves more than the market |
                            | 0.9 < β < 1.1 | Medium | Moves with the market |
                            | 0.6 < β < 0.9 | Low | Moves less than the market |
                            | 0 < β < 0.6 | Very Low | Moves much less than the market |
                            | β < 0 | Negative | Moves opposite to the market |
                            """)
                else:
                    # If beta is not available, try to calculate it using price data and S&P 500
                    st.warning("Beta information not available from company data.")

            # Value at Risk (VaR) Analysis
            st.markdown('<div class="section-header"><h4>Value at Risk (VaR) Analysis</h4></div>',
                        unsafe_allow_html=True)

            # Calculate VaR
            returns = price_data['Daily Return'].dropna()

            # Calculate parametric VaR
            mean_return = returns.mean()
            std_return = returns.std()

            # 95% and 99% VaR
            var_95 = (mean_return - 1.645 * std_return) * 100
            var_99 = (mean_return - 2.326 * std_return) * 100

            # Historical VaR
            hist_var_95 = np.percentile(returns, 5) * 100
            hist_var_99 = np.percentile(returns, 1) * 100

            # Display VaR
            var_col1, var_col2 = st.columns(2)

            with var_col1:
                st.markdown("#### Parametric VaR")

                st.markdown(
                    f"""
                            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 5px 0;">
                                <p style="margin: 0; font-size: 14px;">1-Day 95% VaR: <span style="color: {COLORS['accent']};">{abs(var_95):.2f}%</span></p>
                                <p style="margin: 5px 0; color: #a0a0a0; font-size: 12px;">
                                    95% confidence the daily loss will not exceed {abs(var_95):.2f}%
                                </p>
                            </div>

                            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 5px 0;">
                                <p style="margin: 0; font-size: 14px;">1-Day 99% VaR: <span style="color: {COLORS['accent']};">{abs(var_99):.2f}%</span></p>
                                <p style="margin: 5px 0; color: #a0a0a0; font-size: 12px;">
                                    99% confidence the daily loss will not exceed {abs(var_99):.2f}%
                                </p>
                            </div>
                            """,
                    unsafe_allow_html=True
                )

            with var_col2:
                st.markdown("#### Historical VaR")

                st.markdown(
                    f"""
                            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 5px 0;">
                                <p style="margin: 0; font-size: 14px;">1-Day 95% VaR: <span style="color: {COLORS['accent']};">{abs(hist_var_95):.2f}%</span></p>
                                <p style="margin: 5px 0; color: #a0a0a0; font-size: 12px;">
                                    Based on actual historical returns distribution
                                </p>
                            </div>

                            <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 5px 0;">
                                <p style="margin: 0; font-size: 14px;">1-Day 99% VaR: <span style="color: {COLORS['accent']};">{abs(hist_var_99):.2f}%</span></p>
                                <p style="margin: 5px 0; color: #a0a0a0; font-size: 12px;">
                                    Based on actual historical returns distribution
                                </p>
                            </div>
                            """,
                    unsafe_allow_html=True
                )

            # Visualize return distribution
            st.markdown("#### Returns Distribution")

            # Plot returns distribution
            fig = visualizer.plot_returns_distribution(
                returns * 100,  # Convert to percentage
                ticker,
                var_95,
                var_99,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # VaR interpretation
            st.markdown("#### Value at Risk Interpretation")

            # Calculate dollar VaR based on a sample investment
            investment_amount = 10000  # $10,000 investment
            dollar_var_95 = investment_amount * abs(var_95) / 100

            st.markdown(
                f"""
                        <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin: 10px 0;">
                            <p style="margin: 0;">For a <strong>${investment_amount:,}</strong> investment in {ticker}:</p>
                            <ul style="margin: 10px 0;">
                                <li>There is a 5% probability of losing more than <strong>${dollar_var_95:.2f}</strong> in a single day.</li>
                                <li>This stock's daily Value at Risk is <strong>{abs(var_95):.2f}%</strong> with 95% confidence.</li>
                            </ul>
                            <p style="margin: 5px 0; color: #a0a0a0; font-size: 12px;">
                                Value at Risk (VaR) estimates the maximum loss that might be experienced in a given time period with a specified confidence level.
                            </p>
                        </div>
                        """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Insufficient price data to perform market risk analysis.")

        # Tab 4: Scenario Analysis
    with risk_tabs[3]:
        st.subheader("Scenario Analysis")

        # Check if we have financial data for scenario analysis
        if (income_stmt is not None and not income_stmt.empty and
                balance_sheet is not None and not balance_sheet.empty and
                cash_flow is not None and not cash_flow.empty):

            st.markdown("""
                    Scenario analysis examines how a company's financials and valuation might change under different economic and business conditions.
                    The scenarios below model potential outcomes under various assumptions.
                    """)

            # Define scenarios
            scenarios = [
                {
                    "name": "Base Case",
                    "description": "Current projections, moderate growth, stable margins",
                    "revenue_growth": 0.10,  # 10%
                    "margin_change": 0.00,  # 0%
                    "discount_rate": 0.10,  # 10%
                    "color": COLORS["warning"]
                },
                {
                    "name": "Optimistic Case",
                    "description": "Higher growth, margin expansion, favorable economic conditions",
                    "revenue_growth": 0.20,  # 20%
                    "margin_change": 0.02,  # +2%
                    "discount_rate": 0.09,  # 9%
                    "color": COLORS["primary"]
                },
                {
                    "name": "Pessimistic Case",
                    "description": "Slowing growth, margin compression, challenging conditions",
                    "revenue_growth": 0.05,  # 5%
                    "margin_change": -0.03,  # -3%
                    "discount_rate": 0.12,  # 12%
                    "color": COLORS["accent"]
                },
                {
                    "name": "Recession Case",
                    "description": "Negative growth, significant margin compression, high discount rates",
                    "revenue_growth": -0.10,  # -10%
                    "margin_change": -0.05,  # -5%
                    "discount_rate": 0.15,  # 15%
                    "color": COLORS["accent"]
                }
            ]

            # Create tabs for editing scenarios and viewing results
            scenario_tabs = st.tabs(["Scenario Parameters", "Financial Impact", "Valuation Impact"])

            # Tab: Scenario Parameters
            with scenario_tabs[0]:
                st.markdown("### Customize Scenario Parameters")
                st.markdown("Adjust the parameters below to model different economic and business scenarios.")

                # Initialize scenario data
                for i, scenario in enumerate(scenarios):
                    # Create columns for editing parameters
                    param_cols = st.columns([3, 2, 2, 2])

                    with param_cols[0]:
                        st.markdown(
                            f"""
                                    <div style="background-color: {scenario['color']}20; padding: 10px; border-radius: 5px; 
                                               border-left: 4px solid {scenario['color']};">
                                        <h4 style="margin: 0; color: {scenario['color']};">{scenario['name']}</h4>
                                        <p style="margin: 5px 0; font-size: 12px;">{scenario['description']}</p>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                        )

                    with param_cols[1]:
                        scenarios[i]["revenue_growth"] = st.slider(
                            f"Revenue Growth ({scenario['name']})",
                            min_value=-0.20,
                            max_value=0.30,
                            value=scenario['revenue_growth'],
                            step=0.01,
                            format="%.0f%%",
                            key=f"rev_growth_{i}"
                        )

                    with param_cols[2]:
                        scenarios[i]["margin_change"] = st.slider(
                            f"Margin Change ({scenario['name']})",
                            min_value=-0.10,
                            max_value=0.10,
                            value=scenario['margin_change'],
                            step=0.01,
                            format="%.0f%%",
                            key=f"margin_{i}"
                        )

                    with param_cols[3]:
                        scenarios[i]["discount_rate"] = st.slider(
                            f"Discount Rate ({scenario['name']})",
                            min_value=0.05,
                            max_value=0.20,
                            value=scenario['discount_rate'],
                            step=0.01,
                            format="%.0f%%",
                            key=f"discount_{i}"
                        )

                # Add note about scenarios
                st.markdown("""
                        <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-top: 20px;">
                            <h4 style="margin: 0;">About Scenario Analysis</h4>
                            <p style="margin: 10px 0; font-size: 14px; color: #a0a0a0;">
                                Scenario analysis helps investors understand the range of potential outcomes for a company under different conditions:
                                <ul style="margin: 5px 0; color: #a0a0a0;">
                                    <li><strong>Revenue Growth</strong>: How sales might change year-over-year</li>
                                    <li><strong>Margin Change</strong>: How profit margins might expand or contract</li>
                                    <li><strong>Discount Rate</strong>: How the required return might change with risk perception</li>
                                </ul>
                                These scenarios provide a sensitivity analysis rather than precise forecasts.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            # Tab: Financial Impact
            with scenario_tabs[1]:
                st.markdown("### Financial Impact by Scenario")

                # Get baseline financial metrics
                if income_stmt is not None and not income_stmt.empty:
                    # Most recent financial data
                    current_revenue = income_stmt.loc['Total Revenue'].iloc[
                        0] if 'Total Revenue' in income_stmt.index else None
                    current_operating_income = income_stmt.loc['Operating Income'].iloc[
                        0] if 'Operating Income' in income_stmt.index else None
                    current_net_income = income_stmt.loc['Net Income'].iloc[
                        0] if 'Net Income' in income_stmt.index else None
                    current_operating_margin = current_operating_income / current_revenue if current_operating_income and current_revenue else None

                    if current_revenue and current_operating_margin:
                        # Calculate scenario projections (simplified 3-year projection)
                        projection_years = 3
                        scenario_projections = []

                        for scenario in scenarios:
                            projection = {
                                "name": scenario["name"],
                                "color": scenario["color"],
                                "years": [],
                                "revenue": [],
                                "operating_income": [],
                                "operating_margin": []
                            }

                            for year in range(projection_years + 1):  # +1 for current year (year 0)
                                projection["years"].append(year)

                                # Year 0 is current
                                if year == 0:
                                    projection["revenue"].append(current_revenue)
                                    projection["operating_margin"].append(current_operating_margin)
                                    projection["operating_income"].append(current_operating_income)
                                else:
                                    # Calculate revenue with compounded growth
                                    revenue = current_revenue * (1 + scenario["revenue_growth"]) ** year
                                    projection["revenue"].append(revenue)

                                    # Calculate margins with changes
                                    margin = current_operating_margin + (scenario["margin_change"] * year)
                                    # Ensure margin stays within reasonable bounds
                                    margin = max(0.01, min(0.5, margin))
                                    projection["operating_margin"].append(margin)

                                    # Calculate operating income
                                    operating_income = revenue * margin
                                    projection["operating_income"].append(operating_income)

                            scenario_projections.append(projection)

                        # Display financial projections
                        st.markdown("#### Revenue Projections")
                        revenue_fig = visualizer.plot_scenario_projections(
                            scenario_projections,
                            "revenue",
                            "Revenue",
                            format_billions=True,
                            height=350
                        )
                        st.plotly_chart(revenue_fig, use_container_width=True)

                        st.markdown("#### Operating Margin Projections")
                        margin_fig = visualizer.plot_scenario_projections(
                            scenario_projections,
                            "operating_margin",
                            "Operating Margin",
                            format_percentage=True,
                            height=350
                        )
                        st.plotly_chart(margin_fig, use_container_width=True)

                        st.markdown("#### Operating Income Projections")
                        income_fig = visualizer.plot_scenario_projections(
                            scenario_projections,
                            "operating_income",
                            "Operating Income",
                            format_billions=True,
                            height=350
                        )
                        st.plotly_chart(income_fig, use_container_width=True)

                        # Display year 3 projections in a table
                        st.markdown("#### Financial Metrics - Year 3 Projections")

                        # Create DataFrame for comparison
                        year3_data = []
                        for projection in scenario_projections:
                            year3_data.append({
                                "Scenario": projection["name"],
                                "Revenue": projection["revenue"][3],  # Year 3
                                "Op. Margin": projection["operating_margin"][3],  # Year 3
                                "Op. Income": projection["operating_income"][3],  # Year 3
                                "Color": projection["color"]
                            })

                        year3_df = pd.DataFrame(year3_data)

                        # Format for display
                        display_df = pd.DataFrame()
                        display_df["Scenario"] = year3_df["Scenario"]
                        display_df["Revenue"] = year3_df["Revenue"].apply(
                            lambda x: f"${x / 1e9:.2f}B" if x >= 1e9 else f"${x / 1e6:.2f}M"
                        )
                        display_df["Operating Margin"] = year3_df["Op. Margin"].apply(
                            lambda x: f"{x * 100:.1f}%"
                        )
                        display_df["Operating Income"] = year3_df["Op. Income"].apply(
                            lambda x: f"${x / 1e9:.2f}B" if x >= 1e9 else f"${x / 1e6:.2f}M"
                        )

                        # Display table
                        st.table(display_df)
                    else:
                        st.warning("Insufficient financial data to perform scenario projections.")
                else:
                    st.warning("Income statement data not available for scenario analysis.")

            # Tab: Valuation Impact
            with scenario_tabs[2]:
                st.markdown("### Valuation Impact by Scenario")

                # Get baseline financial metrics for valuation
                if income_stmt is not None and not income_stmt.empty and cash_flow is not None and not cash_flow.empty:
                    # Try to get Free Cash Flow
                    base_fcf = None

                    # Try to get FCF from cash flow statement
                    if 'Free Cash Flow' in cash_flow.index:
                        base_fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
                    # Or calculate from Operating Cash Flow and Capital Expenditure
                    elif all(x in cash_flow.index for x in ['Operating Cash Flow', 'Capital Expenditure']):
                        ocf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                        capex = cash_flow.loc['Capital Expenditure'].iloc[0]
                        base_fcf = ocf - abs(capex)
                    # Or estimate from Net Income
                    elif 'Net Income' in income_stmt.index:
                        net_income = income_stmt.loc['Net Income'].iloc[0]
                        # Simple approximation: FCF = Net Income * 0.8
                        base_fcf = net_income * 0.8

                    if base_fcf:
                        # Calculate valuation impacts
                        current_price = price_data['Close'].iloc[-1] if not price_data.empty else company_info.get(
                            'current_price')

                        # Get shares outstanding
                        shares_outstanding = company_info.get('shares_outstanding')
                        if not shares_outstanding and 'market_cap' in company_info and current_price:
                            # Estimate from market cap and price
                            shares_outstanding = company_info['market_cap'] / current_price

                        if shares_outstanding:
                            # Calculate net debt
                            net_debt = 0
                            if balance_sheet is not None and not balance_sheet.empty:
                                if all(x in balance_sheet.index for x in ['Total Debt', 'Cash and Cash Equivalents']):
                                    total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                                    cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]
                                    net_debt = total_debt - cash

                            # Calculate DCF valuations for different scenarios
                            terminal_growth = 0.02  # Default terminal growth rate
                            forecast_years = 5  # Default forecast period

                            scenario_valuations = []

                            for scenario in scenarios:
                                # Apply scenario growth rate and discount rate
                                growth_rate = scenario["revenue_growth"]
                                discount_rate = scenario["discount_rate"]

                                # Simplified valuation calculation
                                # Forecast FCF with growth rate
                                forecast_fcf = []
                                for year in range(1, forecast_years + 1):
                                    fcf = base_fcf * (1 + growth_rate) ** year
                                    # Apply margin impact
                                    margin_impact = 1 + (scenario["margin_change"] * year)
                                    fcf *= margin_impact
                                    forecast_fcf.append(fcf)

                                # Terminal value
                                terminal_value = forecast_fcf[-1] * (1 + terminal_growth) / (
                                            discount_rate - terminal_growth)

                                # Discount cash flows
                                pv_fcf = sum(
                                    fcf / (1 + discount_rate) ** year for year, fcf in enumerate(forecast_fcf, 1))
                                pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

                                # Enterprise value
                                enterprise_value = pv_fcf + pv_terminal

                                # Equity value
                                equity_value = enterprise_value - net_debt

                                # Value per share
                                value_per_share = equity_value / shares_outstanding

                                # Upside potential
                                if current_price:
                                    upside = (value_per_share / current_price - 1) * 100
                                else:
                                    upside = None

                                # Add to valuations
                                scenario_valuations.append({
                                    "name": scenario["name"],
                                    "color": scenario["color"],
                                    "growth_rate": growth_rate,
                                    "discount_rate": discount_rate,
                                    "enterprise_value": enterprise_value,
                                    "equity_value": equity_value,
                                    "value_per_share": value_per_share,
                                    "upside": upside
                                })

                            # Display valuations
                            st.markdown("#### Valuation by Scenario")

                            for valuation in scenario_valuations:
                                # Format enterprise value
                                if valuation["enterprise_value"] >= 1e12:
                                    ev_display = f"${valuation['enterprise_value'] / 1e12:.2f}T"
                                elif valuation["enterprise_value"] >= 1e9:
                                    ev_display = f"${valuation['enterprise_value'] / 1e9:.2f}B"
                                else:
                                    ev_display = f"${valuation['enterprise_value'] / 1e6:.2f}M"

                                # Format upside
                                upside_display = f"{valuation['upside']:+.1f}%" if valuation[
                                                                                       'upside'] is not None else "N/A"
                                upside_color = COLORS["primary"] if valuation['upside'] > 0 else COLORS["accent"] if \
                                valuation['upside'] < 0 else "#ffffff"

                                st.markdown(
                                    f"""
                                            <div style="background-color: {valuation['color']}20; padding: 15px; border-radius: 10px; 
                                                      margin: 10px 0; border-left: 4px solid {valuation['color']};">
                                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                                    <div>
                                                        <h4 style="margin: 0; color: {valuation['color']};">{valuation['name']}</h4>
                                                        <p style="margin: 5px 0; font-size: 12px;">
                                                            Growth: {valuation['growth_rate'] * 100:.1f}% | 
                                                            Discount Rate: {valuation['discount_rate'] * 100:.1f}%
                                                        </p>
                                                    </div>
                                                    <div style="text-align: right;">
                                                        <h3 style="margin: 0;">${valuation['value_per_share']:.2f}</h3>
                                                        <p style="margin: 0; color: {upside_color};">{upside_display}</p>
                                                    </div>
                                                </div>
                                                <div style="margin-top: 10px;">
                                                    <p style="margin: 0; font-size: 14px;">Enterprise Value: {ev_display}</p>
                                                </div>
                                            </div>
                                            """,
                                    unsafe_allow_html=True
                                )

                            # Create comparative bar chart for share prices
                            st.markdown("#### Share Price Comparison")

                            # Prepare data for visualization
                            scenarios_list = [v["name"] for v in scenario_valuations]
                            prices_list = [v["value_per_share"] for v in scenario_valuations]
                            colors_list = [v["color"] for v in scenario_valuations]

                            # Add current price if available
                            if current_price:
                                scenarios_list.append("Current Price")
                                prices_list.append(current_price)
                                colors_list.append("#888888")

                            # Create DataFrame
                            price_df = pd.DataFrame({
                                'Scenario': scenarios_list,
                                'Price': prices_list,
                                'Color': colors_list
                            })

                            # Plot bar chart
                            fig = visualizer.plot_scenario_prices(
                                price_df,
                                current_price,
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Add interpretation
                            st.markdown("""
                                    <div style="background-color: #1f1f1f; padding: 15px; border-radius: 5px; margin-top: 20px;">
                                        <h4 style="margin: 0;">Valuation Sensitivity Analysis</h4>
                                        <p style="margin: 10px 0; color: #a0a0a0;">
                                            This analysis shows how the company's estimated value changes under different scenarios.
                                            The wider the range between optimistic and pessimistic cases, the more sensitive the 
                                            valuation is to changing conditions, indicating higher investment uncertainty.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("Shares outstanding data not available for valuation calculation.")
                    else:
                        st.warning("Cash flow data not available for scenario analysis.")
                else:
                    st.warning("Financial data not available for valuation impact analysis.")
        else:
            st.info("Scenario analysis requires financial statement data which is not available for this company.")

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
            'beta': 1.25,
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

        render_company_risk(ticker, financial_data, company_info, price_data)