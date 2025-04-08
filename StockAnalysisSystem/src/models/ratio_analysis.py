import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.config import SECTOR_SPECIFIC_RATIOS, RATIO_CATEGORIES, SECTOR_MAPPING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ratio_analysis')


class FinancialRatioAnalyzer:
    """
    Class for analyzing financial ratios of companies and comparing them with industry benchmarks.
    Provides functionality for calculating various financial ratios based on financial statements
    and analyzing them in the context of sector and industry standards.
    """

    def __init__(self):
        """Initialize the financial ratio analyzer"""
        # Define sector-specific important ratios
        self.sector_specific_ratios = SECTOR_SPECIFIC_RATIOS

        # Define ratio categories for organization
        self.ratio_categories = RATIO_CATEGORIES

    def calculate_ratios(self, financial_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate all financial ratios from financial statements

        Args:
            financial_data: Dictionary containing income_statement, balance_sheet, and cash_flow dataframes

        Returns:
            Dictionary of ratio categories with calculated ratios
        """
        ratios = {
            "valuation": {},
            "profitability": {},
            "liquidity": {},
            "leverage": {},
            "efficiency": {},
            "growth": {}
        }

        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            market_data = financial_data.get('market_data', {})

            # Check if we have the necessary data
            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements to calculate ratios")
                return ratios

            # Get the most recent data (first column)
            if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                income = income_stmt.iloc[:, 0]
            else:
                logger.warning("Income statement is empty or not a DataFrame")
                income = pd.Series()

            if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                balance = balance_sheet.iloc[:, 0]
            else:
                logger.warning("Balance sheet is empty or not a DataFrame")
                balance = pd.Series()

            if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                cf = cash_flow.iloc[:, 0]
            else:
                logger.warning("Cash flow statement is empty or not a DataFrame")
                cf = pd.Series()

            # Calculate valuation ratios
            if market_data and 'market_cap' in market_data and 'share_price' in market_data:
                price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                shares_outstanding = market_cap / price if price else None

                # P/E Ratio
                if shares_outstanding and 'Net Income' in income:
                    eps = income.get('Net Income') / shares_outstanding
                    ratios['valuation']['pe_ratio'] = price / eps if eps and eps > 0 else None

                # P/S Ratio
                if market_cap and 'Total Revenue' in income:
                    ratios['valuation']['ps_ratio'] = market_cap / income.get('Total Revenue') if income.get(
                        'Total Revenue') else None

                # P/B Ratio
                if market_cap and 'Total Stockholder Equity' in balance:
                    ratios['valuation']['pb_ratio'] = market_cap / balance.get(
                        'Total Stockholder Equity') if balance.get('Total Stockholder Equity') else None

                # EV/EBITDA
                if market_cap and 'Total Debt' in balance and 'Cash and Cash Equivalents' in balance and 'EBITDA' in income:
                    enterprise_value = market_cap + balance.get('Total Debt', 0) - balance.get(
                        'Cash and Cash Equivalents', 0)
                    ratios['valuation']['ev_ebitda'] = enterprise_value / income.get('EBITDA') if income.get(
                        'EBITDA') and income.get('EBITDA') > 0 else None

                # EV/Revenue
                if market_cap and 'Total Debt' in balance and 'Cash and Cash Equivalents' in balance and 'Total Revenue' in income:
                    enterprise_value = market_cap + balance.get('Total Debt', 0) - balance.get(
                        'Cash and Cash Equivalents', 0)
                    ratios['valuation']['ev_revenue'] = enterprise_value / income.get('Total Revenue') if income.get(
                        'Total Revenue') else None

            # Calculate profitability ratios
            # ROE (Return on Equity)
            if 'Net Income' in income and 'Total Stockholder Equity' in balance:
                ratios['profitability']['roe'] = income.get('Net Income') / balance.get(
                    'Total Stockholder Equity') if balance.get('Total Stockholder Equity') else None

            # ROA (Return on Assets)
            if 'Net Income' in income and 'Total Assets' in balance:
                ratios['profitability']['roa'] = income.get('Net Income') / balance.get('Total Assets') if balance.get(
                    'Total Assets') else None

            # Gross Margin
            if 'Gross Profit' in income and 'Total Revenue' in income:
                ratios['profitability']['gross_margin'] = income.get('Gross Profit') / income.get(
                    'Total Revenue') if income.get('Total Revenue') else None

            # Operating Margin
            if 'Operating Income' in income and 'Total Revenue' in income:
                ratios['profitability']['operating_margin'] = income.get('Operating Income') / income.get(
                    'Total Revenue') if income.get('Total Revenue') else None

            # Net Margin
            if 'Net Income' in income and 'Total Revenue' in income:
                ratios['profitability']['net_margin'] = income.get('Net Income') / income.get(
                    'Total Revenue') if income.get('Total Revenue') else None

            # Calculate liquidity ratios
            # Current Ratio
            if 'Total Current Assets' in balance and 'Total Current Liabilities' in balance:
                ratios['liquidity']['current_ratio'] = balance.get('Total Current Assets') / balance.get(
                    'Total Current Liabilities') if balance.get('Total Current Liabilities') else None

            # Quick Ratio
            if 'Total Current Assets' in balance and 'Inventory' in balance and 'Total Current Liabilities' in balance:
                quick_assets = balance.get('Total Current Assets', 0) - balance.get('Inventory', 0)
                ratios['liquidity']['quick_ratio'] = quick_assets / balance.get(
                    'Total Current Liabilities') if balance.get('Total Current Liabilities') else None

            # Cash Ratio
            if 'Cash and Cash Equivalents' in balance and 'Total Current Liabilities' in balance:
                ratios['liquidity']['cash_ratio'] = balance.get('Cash and Cash Equivalents') / balance.get(
                    'Total Current Liabilities') if balance.get('Total Current Liabilities') else None

            # Calculate leverage ratios
            # Debt-to-Equity
            if 'Total Debt' in balance and 'Total Stockholder Equity' in balance:
                ratios['leverage']['debt_to_equity'] = balance.get('Total Debt') / balance.get(
                    'Total Stockholder Equity') if balance.get('Total Stockholder Equity') else None
            elif 'Total Liabilities' in balance and 'Total Stockholder Equity' in balance:
                ratios['leverage']['debt_to_equity'] = balance.get('Total Liabilities') / balance.get(
                    'Total Stockholder Equity') if balance.get('Total Stockholder Equity') else None

            # Debt-to-Assets
            if 'Total Debt' in balance and 'Total Assets' in balance:
                ratios['leverage']['debt_to_assets'] = balance.get('Total Debt') / balance.get(
                    'Total Assets') if balance.get('Total Assets') else None
            elif 'Total Liabilities' in balance and 'Total Assets' in balance:
                ratios['leverage']['debt_to_assets'] = balance.get('Total Liabilities') / balance.get(
                    'Total Assets') if balance.get('Total Assets') else None

            # Interest Coverage Ratio
            if 'Operating Income' in income and 'Interest Expense' in income:
                ratios['leverage']['interest_coverage'] = income.get('Operating Income') / abs(
                    income.get('Interest Expense')) if income.get('Interest Expense') and income.get(
                    'Interest Expense') != 0 else None

            # Calculate efficiency ratios
            # Asset Turnover
            if 'Total Revenue' in income and 'Total Assets' in balance:
                ratios['efficiency']['asset_turnover'] = income.get('Total Revenue') / balance.get(
                    'Total Assets') if balance.get('Total Assets') else None

            # Inventory Turnover
            if 'Cost of Revenue' in income and 'Inventory' in balance:
                ratios['efficiency']['inventory_turnover'] = income.get('Cost of Revenue') / balance.get(
                    'Inventory') if balance.get('Inventory') and balance.get('Inventory') > 0 else None

            # Receivables Turnover
            if 'Total Revenue' in income and 'Net Receivables' in balance:
                ratios['efficiency']['receivables_turnover'] = income.get('Total Revenue') / balance.get(
                    'Net Receivables') if balance.get('Net Receivables') and balance.get(
                    'Net Receivables') > 0 else None

            # Calculate growth ratios if we have historical data
            if isinstance(income_stmt, pd.DataFrame) and income_stmt.shape[1] >= 2:
                current_year = income_stmt.iloc[:, 0]
                prev_year = income_stmt.iloc[:, 1]

                # Revenue Growth
                if 'Total Revenue' in current_year and 'Total Revenue' in prev_year:
                    if prev_year['Total Revenue'] and prev_year['Total Revenue'] != 0:
                        ratios['growth']['revenue_growth'] = (
                                    current_year['Total Revenue'] / prev_year['Total Revenue'] - 1)

                # Net Income Growth
                if 'Net Income' in current_year and 'Net Income' in prev_year:
                    if prev_year['Net Income'] and prev_year['Net Income'] != 0:
                        ratios['growth']['net_income_growth'] = (
                                    current_year['Net Income'] / prev_year['Net Income'] - 1)

            # Remove any NaN values
            for category in ratios:
                ratios[category] = {k: v for k, v in ratios[category].items() if v is not None and not pd.isna(v)}

            return ratios

        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return ratios

    def get_sector_benchmarks(self, sector: str) -> Dict[str, Dict[str, float]]:
        """
        Get benchmark ratios for a specific sector

        Args:
            sector: Market sector name

        Returns:
            Dictionary of benchmark ratios by category
        """
        # In a real implementation, these would be fetched from a database
        # For now, we'll use hardcoded sample values for demonstration
        benchmarks = {
            "Technology": {
                "valuation": {"pe_ratio": 25.0, "ps_ratio": 5.0, "pb_ratio": 6.0, "ev_ebitda": 18.0},
                "profitability": {"gross_margin": 0.60, "operating_margin": 0.25, "net_margin": 0.20, "roe": 0.22,
                                  "roa": 0.15},
                "liquidity": {"current_ratio": 2.5, "quick_ratio": 2.0},
                "leverage": {"debt_to_equity": 0.5, "interest_coverage": 15.0}
            },
            "Healthcare": {
                "valuation": {"pe_ratio": 22.0, "ps_ratio": 4.0, "pb_ratio": 4.0, "ev_ebitda": 15.0},
                "profitability": {"gross_margin": 0.65, "operating_margin": 0.18, "net_margin": 0.15, "roe": 0.18,
                                  "roa": 0.10},
                "liquidity": {"current_ratio": 2.0, "quick_ratio": 1.7},
                "leverage": {"debt_to_equity": 0.6, "interest_coverage": 12.0}
            },
            "Financials": {
                "valuation": {"pe_ratio": 14.0, "pb_ratio": 1.2, "ps_ratio": 3.0},
                "profitability": {"net_margin": 0.25, "roe": 0.12, "roa": 0.01},
                "liquidity": {"current_ratio": 1.2},
                "leverage": {"debt_to_equity": 1.5}
            },
            "Consumer Discretionary": {
                "valuation": {"pe_ratio": 18.0, "ps_ratio": 1.5, "pb_ratio": 3.5, "ev_ebitda": 12.0},
                "profitability": {"gross_margin": 0.35, "operating_margin": 0.12, "net_margin": 0.08, "roe": 0.15,
                                  "roa": 0.08},
                "liquidity": {"current_ratio": 1.8, "quick_ratio": 0.9},
                "leverage": {"debt_to_equity": 0.8, "interest_coverage": 8.0}
            },
            "Energy": {
                "valuation": {"pe_ratio": 15.0, "ps_ratio": 1.0, "pb_ratio": 1.5, "ev_ebitda": 6.0},
                "profitability": {"gross_margin": 0.30, "operating_margin": 0.15, "net_margin": 0.10, "roe": 0.10,
                                  "roa": 0.06},
                "liquidity": {"current_ratio": 1.5, "quick_ratio": 1.2},
                "leverage": {"debt_to_equity": 0.4, "interest_coverage": 10.0}
            },
            "Utilities": {
                "valuation": {"pe_ratio": 16.0, "ps_ratio": 1.8, "pb_ratio": 1.7, "ev_ebitda": 9.0},
                "profitability": {"gross_margin": 0.40, "operating_margin": 0.22, "net_margin": 0.12, "roe": 0.10,
                                  "roa": 0.04},
                "liquidity": {"current_ratio": 1.1, "quick_ratio": 1.0},
                "leverage": {"debt_to_equity": 1.2, "interest_coverage": 5.0}
            }
        }

        # Return benchmarks for the requested sector or a default if not found
        return benchmarks.get(sector, {
            "valuation": {"pe_ratio": 20.0, "ps_ratio": 2.5, "pb_ratio": 3.0, "ev_ebitda": 12.0},
            "profitability": {"gross_margin": 0.40, "operating_margin": 0.15, "net_margin": 0.10, "roe": 0.15,
                              "roa": 0.08},
            "liquidity": {"current_ratio": 1.8, "quick_ratio": 1.5},
            "leverage": {"debt_to_equity": 0.7, "interest_coverage": 8.0}
        })

    def analyze_ratios(self, company_ratios: Dict[str, Dict[str, float]],
                       sector: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze company ratios against sector benchmarks

        Args:
            company_ratios: Dictionary of company's financial ratios
            sector: Market sector of the company

        Returns:
            Dictionary with analysis results including benchmark comparison and assessments
        """
        benchmarks = self.get_sector_benchmarks(sector)
        analysis = {}

        # Color thresholds for visualization (positive/neutral/negative)
        assessment_colors = {
            "positive": "#74f174",  # Green
            "neutral": "#fff59d",  # Yellow
            "negative": "#faa1a4"  # Red
        }

        # Define ratio interpretations (higher is better or lower is better)
        higher_is_better = {
            "valuation": ["pb_ratio", "ps_ratio", "pe_ratio", "ev_ebitda", "ev_revenue"],
            # Lower values are better for valuation
            "profitability": [],  # Higher values are better for all profitability metrics
            "liquidity": [],  # Higher values are better for liquidity
            "leverage": ["debt_to_equity", "debt_to_assets"],  # Lower values are better for these leverage metrics
            "efficiency": [],  # Higher values are generally better for efficiency
            "growth": []  # Higher values are better for growth
        }

        # Process each ratio category
        for category in company_ratios:
            if category not in analysis:
                analysis[category] = {}

            benchmark_category = benchmarks.get(category, {})

            for ratio_name, ratio_value in company_ratios[category].items():
                # Skip if the ratio value is None or NaN
                if ratio_value is None or pd.isna(ratio_value):
                    continue

                benchmark_value = benchmark_category.get(ratio_name)

                # Calculate performance relative to benchmark if available
                if benchmark_value is not None and benchmark_value != 0:
                    # For valuation metrics, lower is better, so invert the percentage
                    is_valuation_inverse = ratio_name in higher_is_better.get(category, [])

                    # Calculate percent difference from benchmark
                    percent_diff = (ratio_value / benchmark_value - 1) * 100

                    # For metrics where lower is better, invert the interpretation
                    if is_valuation_inverse:
                        assessment = "positive" if percent_diff < -10 else "negative" if percent_diff > 10 else "neutral"
                    else:
                        assessment = "positive" if percent_diff > 10 else "negative" if percent_diff < -10 else "neutral"

                    analysis[category][ratio_name] = {
                        "value": ratio_value,
                        "benchmark": benchmark_value,
                        "percent_diff": percent_diff,
                        "assessment": assessment,
                        "color": assessment_colors[assessment]
                    }
                else:
                    # If no benchmark available, only store the value
                    analysis[category][ratio_name] = {
                        "value": ratio_value,
                        "benchmark": None,
                        "percent_diff": None,
                        "assessment": "neutral",
                        "color": assessment_colors["neutral"]
                    }

        return analysis

    def get_key_ratios_for_sector(self, sector: str) -> List[Dict[str, str]]:
        """
        Get the most important ratios for a specific sector

        Args:
            sector: Market sector name

        Returns:
            List of dictionaries with ratio information
        """
        # Get the specific ratios for this sector
        specific_ratios = self.sector_specific_ratios.get(sector, [])

        # Find the categories for each ratio
        ratios_with_categories = []

        for ratio in specific_ratios:
            # Find which category this ratio belongs to
            category = None
            for cat, ratios in self.ratio_categories.items():
                if ratio in ratios:
                    category = cat
                    break

            # If category not found, try to infer from ratio name
            if category is None:
                if "margin" in ratio.lower() or "return" in ratio.lower() or "roe" in ratio.lower():
                    category = "Profitability"
                elif "p/e" in ratio.lower() or "p/s" in ratio.lower() or "ev/" in ratio.lower():
                    category = "Valuation"
                elif "debt" in ratio.lower() or "coverage" in ratio.lower():
                    category = "Leverage"
                elif "ratio" in ratio.lower() and any(term in ratio.lower() for term in ["current", "quick", "cash"]):
                    category = "Liquidity"
                elif "turnover" in ratio.lower() or "efficiency" in ratio.lower():
                    category = "Efficiency"
                else:
                    category = "Other"

            ratios_with_categories.append({
                "ratio": ratio,
                "category": category or "Other",
                "description": self._get_ratio_description(ratio)
            })

        return ratios_with_categories

    def _get_ratio_description(self, ratio: str) -> str:
        """
        Get description for a specific ratio

        Args:
            ratio: Ratio name

        Returns:
            Description of the ratio
        """
        descriptions = {
            "P/E": "Price to Earnings - Shows how much investors are willing to pay per dollar of earnings",
            "Forward P/E": "Forward Price to Earnings - P/E calculated using projected future earnings",
            "PEG": "Price/Earnings to Growth - P/E ratio divided by earnings growth rate",
            "P/S": "Price to Sales - Market value relative to annual revenue",
            "P/B": "Price to Book - Market value relative to book value",
            "EV/EBITDA": "Enterprise Value to EBITDA - Company value relative to earnings before interest, taxes, depreciation, and amortization",
            "EV/Revenue": "Enterprise Value to Revenue - Company value relative to revenue",
            "ROE": "Return on Equity - Net income relative to shareholders' equity",
            "ROA": "Return on Assets - Net income relative to total assets",
            "ROIC": "Return on Invested Capital - Operating profit relative to invested capital",
            "Gross Margin": "Gross Profit divided by Revenue - Indicates product profitability",
            "Operating Margin": "Operating Income divided by Revenue - Shows operational efficiency",
            "Net Margin": "Net Income divided by Revenue - Overall profit margin",
            "Current Ratio": "Current Assets divided by Current Liabilities - Measures short-term liquidity",
            "Quick Ratio": "Liquid Assets divided by Current Liabilities - Stricter measure of liquidity",
            "Cash Ratio": "Cash divided by Current Liabilities - Most conservative liquidity measure",
            "Debt/Equity": "Total Debt divided by Shareholders' Equity - Indicates financial leverage",
            "Debt/EBITDA": "Total Debt divided by EBITDA - Shows debt repayment capacity",
            "Interest Coverage": "EBIT divided by Interest Expense - Measures ability to pay interest expenses",
            "Asset Turnover": "Revenue divided by Average Total Assets - Shows asset efficiency",
            "Inventory Turnover": "Cost of Goods Sold divided by Average Inventory - Indicates inventory management efficiency",
            "Receivables Turnover": "Revenue divided by Average Accounts Receivable - Shows efficiency in collecting debts",
            "Net Interest Margin": "Net Interest Income divided by Average Earning Assets - Key metric for banks",
            "R&D/Revenue": "R&D Expenses divided by Revenue - Measures innovation intensity",
            "Same-Store Sales Growth": "Growth in revenue from existing stores - Important for retail",
            "Reserve Replacement Ratio": "New reserves added relative to production - Critical for energy companies",
            "Funds From Operations (FFO)": "Net income plus depreciation, amortization, deferred taxes - Key for REITs",
            "NAV": "Net Asset Value - Assets minus liabilities, critical for REITs and financial companies",
            "ARPU": "Average Revenue Per User - Important for subscription-based businesses"
        }

        return descriptions.get(ratio, "No description available")