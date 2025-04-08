import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import UI_SETTINGS, COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('home_page')


def render_home_page():
    """Render the home page of the application"""

    # Set page title and header
    st.title("Stock Analysis System")
    st.markdown("### Comprehensive Financial Analysis & Company Valuation")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Main features section
        st.subheader("Main Features")

        # Create feature cards
        feature_cards = [
            {
                "title": "Company Analysis",
                "description": "Comprehensive financial analysis with industry-specific metrics.",
                "icon": "📊",
                "color": COLORS["primary"]
            },
            {
                "title": "Sector-Specific Valuation",
                "description": "Valuation models tailored to different industry sectors.",
                "icon": "💰",
                "color": COLORS["secondary"]
            },
            {
                "title": "Peer Comparison",
                "description": "Compare companies with sector peers and industry benchmarks.",
                "icon": "📈",
                "color": COLORS["info"]
            },
            {
                "title": "Risk Assessment",
                "description": "Evaluate financial health and bankruptcy risk.",
                "icon": "⚠️",
                "color": COLORS["warning"]
            }
        ]

        # Display feature cards
        for feature in feature_cards:
            st.markdown(
                f"""
                <div style="background-color: {feature['color']}20; padding: 20px; 
                            border-radius: 10px; margin: 10px 0; border-left: 5px solid {feature['color']}">
                    <h3 style="margin:0;">{feature['icon']} {feature['title']}</h3>
                    <p style="margin-top:5px;">{feature['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        # Quick company search
        st.subheader("Quick Search")
        st.markdown("Find and analyze a company:")

        # Search box for companies
        search_query = st.text_input("Search by company name or ticker:", placeholder="e.g. AAPL, Apple, MSFT...")

        # Search button
        search_button = st.button("Search", type="primary", key="search_home")

        # Market overview section
        st.markdown("---")
        st.subheader("Market Overview")

        # Market metrics
        market_metrics = [
            {"name": "S&P 500", "value": "4,783.25", "change": "+0.59%", "color": COLORS["primary"]},
            {"name": "NASDAQ", "value": "15,132.83", "change": "+0.97%", "color": COLORS["primary"]},
            {"name": "DOW", "value": "37,430.52", "change": "+0.23%", "color": COLORS["primary"]},
            {"name": "10Y Treasury", "value": "3.915%", "change": "-0.018", "color": COLORS["success"]}
        ]

        # Display market metrics
        for metric in market_metrics:
            st.markdown(
                f"""
                <div style="background-color: #1f1f1f; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{metric['name']}</span>
                        <span>{metric['value']}</span>
                    </div>
                    <div style="color: {metric['color']}; text-align: right;">
                        {metric['change']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Popular sectors section
    st.markdown("---")
    st.subheader("Industry Sectors")

    # Create columns for sectors
    sector_cols = st.columns(3)

    # Define sectors
    sectors = [
        {"name": "Technology", "icon": "💻", "color": COLORS["sectors"]["Technology"]},
        {"name": "Healthcare", "icon": "🏥", "color": COLORS["sectors"]["Healthcare"]},
        {"name": "Financials", "icon": "🏦", "color": COLORS["sectors"]["Financials"]},
        {"name": "Consumer", "icon": "🛒", "color": COLORS["sectors"]["Consumer Discretionary"]},
        {"name": "Energy", "icon": "⚡", "color": COLORS["sectors"]["Energy"]},
        {"name": "Real Estate", "icon": "🏢", "color": COLORS["sectors"]["Real Estate"]}
    ]

    # Display sector buttons
    for i, sector in enumerate(sectors):
        with sector_cols[i % 3]:
            st.markdown(
                f"""
                <div style="background-color: {sector['color']}20; padding: 15px; 
                            border-radius: 10px; margin: 10px 0; text-align: center; 
                            cursor: pointer; border: 1px solid {sector['color']}40;">
                    <h3 style="margin:0;">{sector['icon']}</h3>
                    <p style="margin:5px 0;">{sector['name']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Recent companies section
    st.markdown("---")
    st.subheader("Popular Companies")

    # Create columns for popular companies
    company_cols = st.columns(4)

    # Define popular companies
    companies = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "color": COLORS["sectors"]["Technology"]},
        {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "color": COLORS["sectors"]["Technology"]},
        {"ticker": "AMZN", "name": "Amazon", "sector": "Consumer",
         "color": COLORS["sectors"]["Consumer Discretionary"]},
        {"ticker": "GOOGL", "name": "Alphabet", "sector": "Technology", "color": COLORS["sectors"]["Technology"]},
        {"ticker": "META", "name": "Meta Platforms", "sector": "Technology", "color": COLORS["sectors"]["Technology"]},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer",
         "color": COLORS["sectors"]["Consumer Discretionary"]},
        {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financials", "color": COLORS["sectors"]["Financials"]},
        {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "color": COLORS["sectors"]["Healthcare"]}
    ]

    # Display company cards
    for i, company in enumerate(companies):
        with company_cols[i % 4]:
            st.markdown(
                f"""
                <div style="background-color: {company['color']}20; padding: 10px; 
                            border-radius: 10px; margin: 8px 0; text-align: center;
                            cursor: pointer; border: 1px solid {company['color']}40;">
                    <h4 style="margin:0;">{company['ticker']}</h4>
                    <p style="margin:5px 0; font-size: 12px;">{company['name']}</p>
                    <p style="margin:0; font-size: 11px; opacity: 0.7;">{company['sector']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #a0a0a0; font-size: 12px;">
            <p>Stock Analysis System | Sector-Specific Fundamental Analysis</p>
            <p>Powered by AI & Financial Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # For direct testing
    render_home_page()