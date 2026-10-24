import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.config import SECTOR_SPECIFIC_RATIOS, RATIO_CATEGORIES, SECTOR_MAPPING, SECTOR_METRICS

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

        # Use sector metrics for benchmarks
        self.sector_metrics = SECTOR_METRICS

        # Define ratio categories for organization
        self.ratio_categories = RATIO_CATEGORIES

        # Initialize ratio descriptions
        self._init_ratio_descriptions()

    def _init_ratio_descriptions(self):
        """Initialize detailed descriptions for financial ratios"""
        self.ratio_descriptions = {
            # Valuation Ratios
            "P/E": "Price to Earnings - Shows how much investors are willing to pay per dollar of earnings. Lower values may indicate undervaluation, while higher values can suggest high growth expectations or overvaluation.",
            "Forward P/E": "Forward Price to Earnings - P/E calculated using projected future earnings. Useful for companies with expected earnings growth or recovery.",
            "PEG": "Price/Earnings to Growth - P/E ratio divided by earnings growth rate. A PEG ratio below 1 may indicate undervaluation relative to growth prospects.",
            "P/S": "Price to Sales - Market value relative to annual revenue. Useful for evaluating companies without positive earnings. Lower is generally better, but varies by industry.",
            "P/B": "Price to Book - Market value relative to book value. Values below 1 may indicate undervaluation, but can also signal underlying problems.",
            "P/C": "Price to Cash Flow - Share price divided by operating cash flow per share. Shows how much investors pay for each dollar of cash flow.",
            "P/FCF": "Price to Free Cash Flow - Share price divided by free cash flow per share. Lower values may indicate better valuation.",
            "EV/EBITDA": "Enterprise Value to EBITDA - Company value (including debt) relative to earnings before interest, taxes, depreciation, and amortization. Lower values suggest potential undervaluation.",
            "EV/Revenue": "Enterprise Value to Revenue - Company value relative to revenue. Particularly useful for companies without positive earnings.",

            # Profitability Ratios
            "ROE": "Return on Equity - Net income relative to shareholders' equity. Measures how efficiently a company uses equity to generate profits. Higher is better.",
            "ROA": "Return on Assets - Net income relative to total assets. Shows how efficiently a company uses its assets. Higher is better.",
            "ROI": "Return on Investment - Gain from investment relative to investment cost. Measures efficiency of investments.",
            "ROIC": "Return on Invested Capital - Net operating profit after taxes relative to invested capital. Shows how efficiently a company uses all its capital. Higher is better.",
            "Gross Margin": "Gross Profit divided by Revenue - Indicates product profitability and pricing power. Higher margins indicate stronger competitive position.",
            "Operating Margin": "Operating Income divided by Revenue - Shows operational efficiency excluding financing decisions. Higher is better.",
            "Net Margin": "Net Income divided by Revenue - Overall profit margin after all expenses. Higher is better.",

            # Liquidity Ratios
            "Current Ratio": "Current Assets divided by Current Liabilities - Measures ability to pay short-term obligations. Values below 1 can indicate liquidity problems.",
            "Quick Ratio": "Liquid Assets (excluding inventory) divided by Current Liabilities - Stricter measure of liquidity. Values below 1 may be concerning for non-retail companies.",
            "Cash Ratio": "Cash & Equivalents divided by Current Liabilities - Most conservative liquidity measure. Shows ability to pay short-term obligations with only cash.",
            "Cash/sh": "Cash per Share - Cash & Equivalents divided by outstanding shares. Indicates cash backing per share.",

            # Leverage Ratios
            "Debt/Equity": "Total Debt divided by Shareholders' Equity - Indicates financial leverage and risk. Higher values indicate greater leverage and potentially higher risk.",
            "Debt/EBITDA": "Total Debt divided by EBITDA - Shows debt repayment capacity. Values above 4-5 may indicate excessive debt for many industries.",
            "Interest Coverage": "EBIT divided by Interest Expense - Measures ability to pay interest expenses. Values below 2 may indicate potential financial distress.",
            "LT Debt/Equity": "Long-term Debt divided by Equity - Focuses on long-term leverage without short-term obligations.",

            # Efficiency Ratios
            "Asset Turnover": "Revenue divided by Average Total Assets - Shows how efficiently assets generate revenue. Higher is better.",
            "Inventory Turnover": "Cost of Goods Sold divided by Average Inventory - Indicates inventory management efficiency. Higher values suggest better inventory management.",
            "Days Inventory Outstanding": "365 divided by Inventory Turnover - Average days inventory is held before sale. Lower is generally better.",
            "Receivables Turnover": "Revenue divided by Average Accounts Receivable - Shows efficiency in collecting debts. Higher is better.",
            "Days Sales Outstanding": "365 divided by Receivables Turnover - Average days to collect payment after sale. Lower is better.",
            "Payables Turnover": "Cost of Goods Sold divided by Average Accounts Payable - Measures how quickly a company pays suppliers.",
            "Cash Conversion Cycle": "DIO + DSO - DPO - Days to convert resources into cash flows. Shorter cycles are generally better.",

            # Industry-Specific Ratios
            "Net Interest Margin": "Net Interest Income divided by Average Earning Assets - Key metric for banks. Higher is better.",
            "Efficiency Ratio": "Non-interest Expense divided by Revenue - Key for financial institutions. Lower is better, indicates cost efficiency.",
            "NPL Ratio": "Non-Performing Loans divided by Total Loans - For banks, lower is better. Indicates loan quality.",
            "R&D/Revenue": "R&D Expenses divided by Revenue - Measures innovation intensity. Higher values common in technology and healthcare.",
            "Same-Store Sales Growth": "Growth in revenue from existing stores - Important for retail companies. Positive values indicate organic growth.",
            "Revenue per Square Foot": "Revenue divided by Retail Space - For retail, higher values indicate better space utilization.",
            "Reserve Replacement Ratio": "New reserves added relative to production - Critical for energy companies. Values above 100% mean growing reserves.",
            "Production Cost per Barrel": "Production costs divided by barrels produced - For energy companies, lower is better.",
            "FFO": "Funds From Operations - Net income plus depreciation, amortization, minus gains on sales. Key REIT metric.",
            "AFFO": "Adjusted Funds From Operations - FFO minus recurring capital expenditures. More conservative REIT metric.",
            "Cap Rate": "Net Operating Income divided by Property Value - For real estate, higher rates suggest better returns but possibly higher risk.",
            "Occupancy Rate": "Occupied space divided by total leasable space - For REITs, higher is better.",
            "ARPU": "Average Revenue Per User - Important for subscription and telecom businesses. Higher is better.",
            "Churn Rate": "Percentage of customers who discontinue service - For subscription businesses, lower is better.",
            "Customer Acquisition Cost": "Marketing expenses divided by new customers - Lower is better, especially relative to customer lifetime value."
        }

        # Add industry-specific ratios from sector metrics
        for sector, metrics in self.sector_metrics.items():
            if 'key_ratios' in metrics:
                for ratio in metrics['key_ratios']:
                    if ratio not in self.ratio_descriptions:
                        self.ratio_descriptions[ratio] = f"Important metric for the {sector} sector."

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

            # Get previous year's data for growth calculations
            if isinstance(income_stmt, pd.DataFrame) and income_stmt.shape[1] >= 2:
                prev_income = income_stmt.iloc[:, 1]
            else:
                prev_income = None

            if isinstance(balance_sheet, pd.DataFrame) and balance_sheet.shape[1] >= 2:
                prev_balance = balance_sheet.iloc[:, 1]
            else:
                prev_balance = None

            # Extract key values using flexible mapping to handle different statement formats
            self._extract_financial_values(income, balance, cf, market_data, ratios, prev_income, prev_balance)

            # Remove any NaN values
            for category in ratios:
                ratios[category] = {k: v for k, v in ratios[category].items() if v is not None and not pd.isna(v)}

            return ratios

        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return ratios

    def _extract_financial_values(self, income, balance, cf, market_data, ratios, prev_income=None, prev_balance=None):
        """
        Extract key financial values from statements with flexible mapping to handle different formats

        Args:
            income: Income statement series
            balance: Balance sheet series
            cf: Cash flow series
            market_data: Market data dictionary
            ratios: Ratios dictionary to populate
            prev_income: Previous year's income statement
            prev_balance: Previous year's balance sheet
        """
        # Market data extraction
        price = market_data.get('share_price')
        market_cap = market_data.get('market_cap')
        shares_outstanding = market_data.get('shares_outstanding')

        if market_cap is None and price is not None and shares_outstanding is not None:
            market_cap = price * shares_outstanding
        elif shares_outstanding is None and price is not None and market_cap is not None:
            shares_outstanding = market_cap / price

        # Income statement metrics with flexible mapping
        revenue = self._get_value_with_alternatives(income, [
            'Total Revenue', 'Revenue', 'Net Revenue', 'Sales', 'Net Sales'
        ])

        cost_of_revenue = self._get_value_with_alternatives(income, [
            'Cost of Revenue', 'Cost of Goods Sold', 'COGS', 'Cost of Sales'
        ])

        gross_profit = self._get_value_with_alternatives(income, [
            'Gross Profit', 'Gross Income'
        ])
        if gross_profit is None and revenue is not None and cost_of_revenue is not None:
            gross_profit = revenue - cost_of_revenue

        operating_income = self._get_value_with_alternatives(income, [
            'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
        ])

        net_income = self._get_value_with_alternatives(income, [
            'Net Income', 'Net Profit', 'Net Earnings'
        ])

        interest_expense = self._get_value_with_alternatives(income, [
            'Interest Expense', 'Interest and Debt Expense', 'Interest Paid'
        ])

        tax_expense = self._get_value_with_alternatives(income, [
            'Tax Expense', 'Income Tax Expense', 'Provision for Income Taxes'
        ])

        ebitda = self._get_value_with_alternatives(income, [
            'EBITDA', 'Earnings Before Interest Taxes Depreciation and Amortization'
        ])
        if ebitda is None and operating_income is not None:
            # Try to calculate EBITDA
            depreciation = self._get_value_with_alternatives(income, [
                'Depreciation and Amortization', 'Depreciation', 'Amortization'
            ])
            if depreciation is not None:
                ebitda = operating_income + depreciation

        eps = self._get_value_with_alternatives(income, [
            'EPS (Basic)', 'Basic EPS', 'Earnings Per Share', 'EPS'
        ])
        if eps is None and net_income is not None and shares_outstanding is not None:
            eps = net_income / shares_outstanding

        # Balance sheet metrics with flexible mapping
        total_assets = self._get_value_with_alternatives(balance, [
            'Total Assets', 'Assets'
        ])

        current_assets = self._get_value_with_alternatives(balance, [
            'Total Current Assets', 'Current Assets'
        ])

        cash_and_equivalents = self._get_value_with_alternatives(balance, [
            'Cash and Cash Equivalents', 'Cash and Equivalents', 'Cash'
        ])

        inventory = self._get_value_with_alternatives(balance, [
            'Inventory', 'Inventories'
        ])

        accounts_receivable = self._get_value_with_alternatives(balance, [
            'Net Receivables', 'Accounts Receivable', 'Trade Receivables'
        ])

        current_liabilities = self._get_value_with_alternatives(balance, [
            'Total Current Liabilities', 'Current Liabilities'
        ])

        accounts_payable = self._get_value_with_alternatives(balance, [
            'Accounts Payable', 'Trade Payables'
        ])

        total_debt = self._get_value_with_alternatives(balance, [
            'Total Debt', 'Short-term Debt and Long-term Debt', 'Debt'
        ])
        if total_debt is None:
            short_term_debt = self._get_value_with_alternatives(balance, [
                'Short-term Debt', 'Current Debt'
            ])
            long_term_debt = self._get_value_with_alternatives(balance, [
                'Long-term Debt', 'Non-current Debt'
            ])
            if short_term_debt is not None and long_term_debt is not None:
                total_debt = short_term_debt + long_term_debt

        total_liabilities = self._get_value_with_alternatives(balance, [
            'Total Liabilities', 'Total Liabilities Net Minority Interest'
        ])

        stockholder_equity = self._get_value_with_alternatives(balance, [
            'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
        ])

        # Cash flow metrics with flexible mapping
        operating_cash_flow = self._get_value_with_alternatives(cf, [
            'Operating Cash Flow', 'Cash Flow from Operations', 'Net Cash Provided by Operating Activities'
        ])

        capital_expenditure = self._get_value_with_alternatives(cf, [
            'Capital Expenditure', 'Capital Expenditures', 'Purchase of Property Plant and Equipment'
        ])

        free_cash_flow = self._get_value_with_alternatives(cf, [
            'Free Cash Flow', 'FCF'
        ])
        if free_cash_flow is None and operating_cash_flow is not None and capital_expenditure is not None:
            free_cash_flow = operating_cash_flow + capital_expenditure  # CapEx is usually negative

        # Calculate historical growth rates if previous year data is available
        revenue_growth = None
        net_income_growth = None

        if prev_income is not None:
            prev_revenue = self._get_value_with_alternatives(prev_income, [
                'Total Revenue', 'Revenue', 'Net Revenue', 'Sales', 'Net Sales'
            ])
            if revenue is not None and prev_revenue is not None and prev_revenue != 0:
                revenue_growth = (revenue / prev_revenue) - 1

            prev_net_income = self._get_value_with_alternatives(prev_income, [
                'Net Income', 'Net Profit', 'Net Earnings'
            ])
            if net_income is not None and prev_net_income is not None and prev_net_income != 0:
                net_income_growth = (net_income / prev_net_income) - 1

        # Calculate Valuation Ratios
        self._calculate_valuation_ratios(ratios, market_cap, price, shares_outstanding, revenue, eps,
                                         stockholder_equity, ebitda, total_debt, cash_and_equivalents,
                                         free_cash_flow, operating_cash_flow)

        # Calculate Profitability Ratios
        self._calculate_profitability_ratios(ratios, net_income, stockholder_equity, total_assets,
                                             revenue, gross_profit, operating_income, ebitda,
                                             total_debt, interest_expense)

        # Calculate Liquidity Ratios
        self._calculate_liquidity_ratios(ratios, current_assets, current_liabilities, inventory,
                                         cash_and_equivalents, shares_outstanding)

        # Calculate Leverage Ratios
        self._calculate_leverage_ratios(ratios, total_debt, stockholder_equity, total_assets,
                                        total_liabilities, operating_income, interest_expense, ebitda)

        # Calculate Efficiency Ratios
        self._calculate_efficiency_ratios(ratios, revenue, total_assets, cost_of_revenue, inventory,
                                          accounts_receivable, accounts_payable)

        # Calculate Growth Ratios
        if revenue_growth is not None:
            ratios['growth']['revenue_growth'] = revenue_growth
        if net_income_growth is not None:
            ratios['growth']['net_income_growth'] = net_income_growth

    def _get_value_with_alternatives(self, data_series, field_alternatives):
        """
        Try to get a value from a series using multiple alternative field names

        Args:
            data_series: Pandas Series containing the financial data
            field_alternatives: List of possible field names to try

        Returns:
            The value if found, None otherwise
        """
        if data_series is None:
            return None

        for field in field_alternatives:
            if field in data_series:
                return data_series[field]
        return None

    def _calculate_valuation_ratios(self, ratios, market_cap, price, shares_outstanding, revenue, eps,
                                    stockholder_equity, ebitda, total_debt, cash_and_equivalents,
                                    free_cash_flow, operating_cash_flow):
        """Calculate and populate valuation ratios"""
        # P/E Ratio
        if price is not None and eps is not None and eps > 0:
            ratios['valuation']['pe_ratio'] = price / eps

        # P/S Ratio
        if market_cap is not None and revenue is not None and revenue > 0:
            ratios['valuation']['ps_ratio'] = market_cap / revenue

        # P/B Ratio
        if market_cap is not None and stockholder_equity is not None and stockholder_equity > 0:
            ratios['valuation']['pb_ratio'] = market_cap / stockholder_equity

        # EV/EBITDA
        if market_cap is not None and ebitda is not None and ebitda > 0:
            enterprise_value = market_cap + (total_debt or 0) - (cash_and_equivalents or 0)
            ratios['valuation']['ev_ebitda'] = enterprise_value / ebitda

        # EV/Revenue
        if market_cap is not None and revenue is not None and revenue > 0:
            enterprise_value = market_cap + (total_debt or 0) - (cash_and_equivalents or 0)
            ratios['valuation']['ev_revenue'] = enterprise_value / revenue

        # P/FCF - Price to Free Cash Flow
        if market_cap is not None and free_cash_flow is not None and free_cash_flow > 0:
            ratios['valuation']['p_fcf'] = market_cap / free_cash_flow

        # P/CF - Price to Cash Flow
        if market_cap is not None and operating_cash_flow is not None and operating_cash_flow > 0:
            ratios['valuation']['p_cf'] = market_cap / operating_cash_flow

    def _calculate_profitability_ratios(self, ratios, net_income, stockholder_equity, total_assets,
                                        revenue, gross_profit, operating_income, ebitda,
                                        total_debt, interest_expense):
        """Calculate and populate profitability ratios"""
        # ROE (Return on Equity)
        if net_income is not None and stockholder_equity is not None and stockholder_equity > 0:
            ratios['profitability']['roe'] = net_income / stockholder_equity

        # ROA (Return on Assets)
        if net_income is not None and total_assets is not None and total_assets > 0:
            ratios['profitability']['roa'] = net_income / total_assets

        # Gross Margin
        if gross_profit is not None and revenue is not None and revenue > 0:
            ratios['profitability']['gross_margin'] = gross_profit / revenue

        # Operating Margin
        if operating_income is not None and revenue is not None and revenue > 0:
            ratios['profitability']['operating_margin'] = operating_income / revenue

        # Net Margin
        if net_income is not None and revenue is not None and revenue > 0:
            ratios['profitability']['net_margin'] = net_income / revenue

        # EBITDA Margin
        if ebitda is not None and revenue is not None and revenue > 0:
            ratios['profitability']['ebitda_margin'] = ebitda / revenue

        # Return on Invested Capital (ROIC)
        if operating_income is not None and stockholder_equity is not None and total_debt is not None:
            invested_capital = stockholder_equity + total_debt
            if invested_capital > 0:
                ratios['profitability']['roic'] = operating_income / invested_capital

    def _calculate_liquidity_ratios(self, ratios, current_assets, current_liabilities, inventory,
                                    cash_and_equivalents, shares_outstanding):
        """Calculate and populate liquidity ratios"""
        # Current Ratio
        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            ratios['liquidity']['current_ratio'] = current_assets / current_liabilities

        # Quick Ratio
        if current_assets is not None and inventory is not None and current_liabilities is not None and current_liabilities > 0:
            quick_assets = current_assets - inventory
            ratios['liquidity']['quick_ratio'] = quick_assets / current_liabilities

        # Cash Ratio
        if cash_and_equivalents is not None and current_liabilities is not None and current_liabilities > 0:
            ratios['liquidity']['cash_ratio'] = cash_and_equivalents / current_liabilities

        # Cash per Share
        if cash_and_equivalents is not None and shares_outstanding is not None and shares_outstanding > 0:
            ratios['liquidity']['cash_per_share'] = cash_and_equivalents / shares_outstanding

    def _calculate_leverage_ratios(self, ratios, total_debt, stockholder_equity, total_assets,
                                   total_liabilities, operating_income, interest_expense, ebitda):
        """Calculate and populate leverage ratios"""
        # Debt-to-Equity
        if total_debt is not None and stockholder_equity is not None and stockholder_equity > 0:
            ratios['leverage']['debt_to_equity'] = total_debt / stockholder_equity
        elif total_liabilities is not None and stockholder_equity is not None and stockholder_equity > 0:
            ratios['leverage']['debt_to_equity'] = total_liabilities / stockholder_equity

        # Debt-to-Assets
        if total_debt is not None and total_assets is not None and total_assets > 0:
            ratios['leverage']['debt_to_assets'] = total_debt / total_assets
        elif total_liabilities is not None and total_assets is not None and total_assets > 0:
            ratios['leverage']['debt_to_assets'] = total_liabilities / total_assets

        # Interest Coverage Ratio
        if operating_income is not None and interest_expense is not None and interest_expense != 0:
            ratios['leverage']['interest_coverage'] = operating_income / abs(interest_expense)

        # Debt/EBITDA
        if total_debt is not None and ebitda is not None and ebitda > 0:
            ratios['leverage']['debt_to_ebitda'] = total_debt / ebitda

    def _calculate_efficiency_ratios(self, ratios, revenue, total_assets, cost_of_revenue, inventory,
                                     accounts_receivable, accounts_payable):
        """Calculate and populate efficiency ratios"""
        # Asset Turnover
        if revenue is not None and total_assets is not None and total_assets > 0:
            ratios['efficiency']['asset_turnover'] = revenue / total_assets

        # Inventory Turnover
        if cost_of_revenue is not None and inventory is not None and inventory > 0:
            ratios['efficiency']['inventory_turnover'] = cost_of_revenue / inventory

            # Days Inventory Outstanding (DIO)
            ratios['efficiency']['days_inventory_outstanding'] = 365 / ratios['efficiency']['inventory_turnover']

        # Receivables Turnover
        if revenue is not None and accounts_receivable is not None and accounts_receivable > 0:
            ratios['efficiency']['receivables_turnover'] = revenue / accounts_receivable

            # Days Sales Outstanding (DSO)
            ratios['efficiency']['days_sales_outstanding'] = 365 / ratios['efficiency']['receivables_turnover']

        # Payables Turnover
        if cost_of_revenue is not None and accounts_payable is not None and accounts_payable > 0:
            ratios['efficiency']['payables_turnover'] = cost_of_revenue / accounts_payable

            # Days Payable Outstanding (DPO)
            ratios['efficiency']['days_payable_outstanding'] = 365 / ratios['efficiency']['payables_turnover']

        # Cash Conversion Cycle
        if ('days_inventory_outstanding' in ratios['efficiency'] and
                'days_sales_outstanding' in ratios['efficiency'] and
                'days_payable_outstanding' in ratios['efficiency']):
            ratios['efficiency']['cash_conversion_cycle'] = (
                    ratios['efficiency']['days_inventory_outstanding'] +
                    ratios['efficiency']['days_sales_outstanding'] -
                    ratios['efficiency']['days_payable_outstanding']
            )

    def get_sector_benchmarks(self, sector: str) -> Dict[str, Dict[str, float]]:
        """
        Get benchmark ratios for a specific sector

        Args:
            sector: Market sector name

        Returns:
            Dictionary of benchmark ratios by category
        """
        # Try to get sector metrics from config first
        if sector in self.sector_metrics and 'benchmarks' in self.sector_metrics[sector]:
            sector_benchmarks = self.sector_metrics[sector]['benchmarks']

            # Organize by categories
            benchmarks = {
                "valuation": {},
                "profitability": {},
                "liquidity": {},
                "leverage": {},
                "efficiency": {},
                "growth": {}
            }

            # Map each benchmark to its appropriate category
            for ratio, value in sector_benchmarks.items():
                category = self._determine_ratio_category(ratio)
                if category:
                    benchmarks[category][ratio] = value

            return benchmarks

        # Fallback to hardcoded benchmarks
        benchmarks = {
            "Technology": {
                "valuation": {"pe_ratio": 25.0, "ps_ratio": 5.0, "pb_ratio": 6.0, "ev_ebitda": 18.0},
                "profitability": {"gross_margin": 0.60, "operating_margin": 0.25, "net_margin": 0.20, "roe": 0.22,
                                  "roa": 0.15, "roic": 0.18},
                "liquidity": {"current_ratio": 2.5, "quick_ratio": 2.0, "cash_ratio": 1.2},
                "leverage": {"debt_to_equity": 0.5, "interest_coverage": 15.0, "debt_to_ebitda": 1.8},
                "efficiency": {"asset_turnover": 0.8, "inventory_turnover": 8.0},
                "growth": {"revenue_growth": 0.15, "net_income_growth": 0.18}
            },
            "Healthcare": {
                "valuation": {"pe_ratio": 22.0, "ps_ratio": 4.0, "pb_ratio": 4.0, "ev_ebitda": 15.0}, "profitability": {"gross_margin": 0.65, "operating_margin": 0.18, "net_margin": 0.15, "roe": 0.18,
                                  "roa": 0.10, "roic": 0.14},
                "liquidity": {"current_ratio": 2.0, "quick_ratio": 1.7, "cash_ratio": 1.0},
                "leverage": {"debt_to_equity": 0.6, "interest_coverage": 12.0, "debt_to_ebitda": 2.0},
                "efficiency": {"asset_turnover": 0.6, "inventory_turnover": 5.0},
                "growth": {"revenue_growth": 0.12, "net_income_growth": 0.14}
            },
            "Financials": {
                "valuation": {"pe_ratio": 14.0, "pb_ratio": 1.2, "ps_ratio": 3.0, "ev_ebitda": 12.0},
                "profitability": {"net_margin": 0.25, "roe": 0.12, "roa": 0.01, "roic": 0.08},
                "liquidity": {"current_ratio": 1.2, "quick_ratio": 1.0},
                "leverage": {"debt_to_equity": 1.5, "interest_coverage": 4.0, "debt_to_ebitda": 3.5},
                "efficiency": {"asset_turnover": 0.3},
                "growth": {"revenue_growth": 0.06, "net_income_growth": 0.08}
            },
            "Consumer Discretionary": {
                "valuation": {"pe_ratio": 18.0, "ps_ratio": 1.5, "pb_ratio": 3.5, "ev_ebitda": 12.0},
                "profitability": {"gross_margin": 0.35, "operating_margin": 0.12, "net_margin": 0.08, "roe": 0.15,
                                  "roa": 0.08, "roic": 0.12},
                "liquidity": {"current_ratio": 1.8, "quick_ratio": 0.9, "cash_ratio": 0.5},
                "leverage": {"debt_to_equity": 0.8, "interest_coverage": 8.0, "debt_to_ebitda": 2.2},
                "efficiency": {"asset_turnover": 1.5, "inventory_turnover": 6.0},
                "growth": {"revenue_growth": 0.08, "net_income_growth": 0.10}
            },
            "Consumer Staples": {
                "valuation": {"pe_ratio": 20.0, "ps_ratio": 1.8, "pb_ratio": 4.0, "ev_ebitda": 13.0},
                "profitability": {"gross_margin": 0.40, "operating_margin": 0.15, "net_margin": 0.10, "roe": 0.18,
                                  "roa": 0.09, "roic": 0.14},
                "liquidity": {"current_ratio": 1.6, "quick_ratio": 0.8, "cash_ratio": 0.4},
                "leverage": {"debt_to_equity": 0.7, "interest_coverage": 10.0, "debt_to_ebitda": 2.0},
                "efficiency": {"asset_turnover": 1.2, "inventory_turnover": 7.0},
                "growth": {"revenue_growth": 0.05, "net_income_growth": 0.07}
            },
            "Energy": {
                "valuation": {"pe_ratio": 15.0, "ps_ratio": 1.0, "pb_ratio": 1.5, "ev_ebitda": 6.0},
                "profitability": {"gross_margin": 0.30, "operating_margin": 0.15, "net_margin": 0.10, "roe": 0.10,
                                  "roa": 0.06, "roic": 0.08},
                "liquidity": {"current_ratio": 1.5, "quick_ratio": 1.2, "cash_ratio": 0.6},
                "leverage": {"debt_to_equity": 0.4, "interest_coverage": 10.0, "debt_to_ebitda": 2.5},
                "efficiency": {"asset_turnover": 0.5, "inventory_turnover": 10.0},
                "growth": {"revenue_growth": 0.04, "net_income_growth": 0.05}
            },
            "Industrials": {
                "valuation": {"pe_ratio": 18.0, "ps_ratio": 1.2, "pb_ratio": 3.0, "ev_ebitda": 10.0},
                "profitability": {"gross_margin": 0.30, "operating_margin": 0.12, "net_margin": 0.09, "roe": 0.14,
                                  "roa": 0.07, "roic": 0.10},
                "liquidity": {"current_ratio": 1.7, "quick_ratio": 1.3, "cash_ratio": 0.5},
                "leverage": {"debt_to_equity": 0.6, "interest_coverage": 9.0, "debt_to_ebitda": 2.2},
                "efficiency": {"asset_turnover": 0.8, "inventory_turnover": 5.0},
                "growth": {"revenue_growth": 0.06, "net_income_growth": 0.08}
            },
            "Materials": {
                "valuation": {"pe_ratio": 16.0, "ps_ratio": 1.3, "pb_ratio": 2.0, "ev_ebitda": 8.0},
                "profitability": {"gross_margin": 0.25, "operating_margin": 0.12, "net_margin": 0.08, "roe": 0.12,
                                  "roa": 0.06, "roic": 0.09},
                "liquidity": {"current_ratio": 1.8, "quick_ratio": 1.2, "cash_ratio": 0.5},
                "leverage": {"debt_to_equity": 0.5, "interest_coverage": 8.0, "debt_to_ebitda": 2.5},
                "efficiency": {"asset_turnover": 0.7, "inventory_turnover": 6.0},
                "growth": {"revenue_growth": 0.05, "net_income_growth": 0.06}
            },
            "Real Estate": {
                "valuation": {"pe_ratio": 20.0, "ps_ratio": 6.0, "pb_ratio": 1.5, "ev_ebitda": 15.0},
                "profitability": {"gross_margin": 0.55, "operating_margin": 0.40, "net_margin": 0.25, "roe": 0.08,
                                  "roa": 0.04, "roic": 0.06},
                "liquidity": {"current_ratio": 1.3, "quick_ratio": 1.1},
                "leverage": {"debt_to_equity": 1.1, "interest_coverage": 3.0, "debt_to_ebitda": 6.0},
                "efficiency": {"asset_turnover": 0.2},
                "growth": {"revenue_growth": 0.04, "net_income_growth": 0.05}
            },
            "Communication Services": {
                "valuation": {"pe_ratio": 20.0, "ps_ratio": 2.5, "pb_ratio": 3.5, "ev_ebitda": 9.0},
                "profitability": {"gross_margin": 0.55, "operating_margin": 0.25, "net_margin": 0.15, "roe": 0.16,
                                  "roa": 0.08, "roic": 0.12},
                "liquidity": {"current_ratio": 1.5, "quick_ratio": 1.3},
                "leverage": {"debt_to_equity": 0.9, "interest_coverage": 7.0, "debt_to_ebitda": 2.5},
                "efficiency": {"asset_turnover": 0.6},
                "growth": {"revenue_growth": 0.08, "net_income_growth": 0.10}
            },
            "Utilities": {
                "valuation": {"pe_ratio": 16.0, "ps_ratio": 1.8, "pb_ratio": 1.7, "ev_ebitda": 9.0},
                "profitability": {"gross_margin": 0.40, "operating_margin": 0.22, "net_margin": 0.12, "roe": 0.10,
                                  "roa": 0.04, "roic": 0.07},
                "liquidity": {"current_ratio": 1.1, "quick_ratio": 1.0},
                "leverage": {"debt_to_equity": 1.2, "interest_coverage": 5.0, "debt_to_ebitda": 4.0},
                "efficiency": {"asset_turnover": 0.3},
                "growth": {"revenue_growth": 0.03, "net_income_growth": 0.04}
            }
        }

        # Return benchmarks for the requested sector or a default if not found
        return benchmarks.get(sector, {
            "valuation": {"pe_ratio": 20.0, "ps_ratio": 2.5, "pb_ratio": 3.0, "ev_ebitda": 12.0},
            "profitability": {"gross_margin": 0.40, "operating_margin": 0.15, "net_margin": 0.10, "roe": 0.15,
                              "roa": 0.08, "roic": 0.12},
            "liquidity": {"current_ratio": 1.8, "quick_ratio": 1.5, "cash_ratio": 0.6},
            "leverage": {"debt_to_equity": 0.7, "interest_coverage": 8.0, "debt_to_ebitda": 2.5},
            "efficiency": {"asset_turnover": 0.8, "inventory_turnover": 6.0},
            "growth": {"revenue_growth": 0.08, "net_income_growth": 0.10}
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

        # Define ratio interpretations (whether higher or lower values are better)
        # For valuation metrics (like P/E, P/S), lower values are generally better
        # For most other metrics, higher values are generally better
        lower_is_better = {
            "valuation": ["pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda", "ev_revenue", "p_fcf", "p_cf"],
            "profitability": [],  # Higher values are better for all profitability metrics
            "liquidity": [],  # Higher values are better for liquidity
            "leverage": ["debt_to_equity", "debt_to_assets", "debt_to_ebitda"], # Lower values are better for these
            "efficiency": ["days_inventory_outstanding", "days_sales_outstanding", "days_payable_outstanding",
                         "cash_conversion_cycle"],  # Lower is better for these
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
                    # For metrics where lower is better, invert the interpretation
                    is_lower_better = ratio_name in lower_is_better.get(category, [])

                    # Calculate percent difference from benchmark
                    percent_diff = (ratio_value / benchmark_value - 1) * 100

                    # For metrics where lower is better, invert the interpretation
                    if is_lower_better:
                        assessment = "positive" if percent_diff < -10 else "negative" if percent_diff > 10 else "neutral"
                    else:
                        assessment = "positive" if percent_diff > 10 else "negative" if percent_diff < -10 else "neutral"

                    analysis[category][ratio_name] = {
                        "value": ratio_value,
                        "benchmark": benchmark_value,
                        "percent_diff": percent_diff,
                        "assessment": assessment,
                        "color": assessment_colors[assessment],
                        "description": self._get_ratio_description(ratio_name),
                        "interpretation": self._interpret_ratio(ratio_name, ratio_value, benchmark_value, is_lower_better)
                    }
                else:
                    # If no benchmark available, only store the value
                    analysis[category][ratio_name] = {
                        "value": ratio_value,
                        "benchmark": None,
                        "percent_diff": None,
                        "assessment": "neutral",
                        "color": assessment_colors["neutral"],
                        "description": self._get_ratio_description(ratio_name),
                        "interpretation": self._interpret_ratio_no_benchmark(ratio_name, ratio_value)
                    }

        return analysis

    def _interpret_ratio(self, ratio_name: str, value: float, benchmark: float, is_lower_better: bool) -> str:
        """
        Provide interpretation for a ratio value compared to benchmark

        Args:
            ratio_name: Name of the ratio
            value: Ratio value
            benchmark: Benchmark value
            is_lower_better: Whether lower values are better for this ratio

        Returns:
            Interpretation string
        """
        # Calculate percent difference
        percent_diff = (value / benchmark - 1) * 100

        # Specific interpretations based on ratio type
        if is_lower_better:
            if percent_diff < -25:
                return f"Significantly better than sector average ({abs(percent_diff):.1f}% lower)"
            elif percent_diff < -10:
                return f"Better than sector average ({abs(percent_diff):.1f}% lower)"
            elif percent_diff > 25:
                return f"Significantly worse than sector average ({percent_diff:.1f}% higher)"
            elif percent_diff > 10:
                return f"Worse than sector average ({percent_diff:.1f}% higher)"
            else:
                return f"Around sector average ({benchmark:.2f})"
        else:
            if percent_diff > 25:
                return f"Significantly better than sector average ({percent_diff:.1f}% higher)"
            elif percent_diff > 10:
                return f"Better than sector average ({percent_diff:.1f}% higher)"
            elif percent_diff < -25:
                return f"Significantly worse than sector average ({abs(percent_diff):.1f}% lower)"
            elif percent_diff < -10:
                return f"Worse than sector average ({abs(percent_diff):.1f}% lower)"
            else:
                return f"Around sector average ({benchmark:.2f})"

    def _interpret_ratio_no_benchmark(self, ratio_name: str, value: float) -> str:
        """
        Provide interpretation for a ratio value without benchmark

        Args:
            ratio_name: Name of the ratio
            value: Ratio value

        Returns:
            Interpretation string
        """
        # General interpretations based on ratio type and common industry knowledge
        if ratio_name == "pe_ratio":
            if value < 10:
                return "Low P/E ratio may indicate undervaluation or concerns about future growth"
            elif value < 20:
                return "Moderate P/E ratio within typical market range"
            else:
                return "High P/E ratio indicates high growth expectations or potential overvaluation"

        elif ratio_name == "pb_ratio":
            if value < 1:
                return "Trading below book value, potential value opportunity or deteriorating business"
            elif value < 3:
                return "Moderate P/B ratio, reasonable valuation relative to assets"
            else:
                return "High P/B ratio, premium valuation indicating strong expected returns on equity"

        elif ratio_name == "current_ratio":
            if value < 1:
                return "May face liquidity issues, as current assets don't cover current liabilities"
            elif value < 1.5:
                return "Adequate but not strong liquidity position"
            else:
                return "Strong liquidity position with good coverage of short-term obligations"

        elif ratio_name == "debt_to_equity":
            if value < 0.5:
                return "Conservative capital structure with low leverage"
            elif value < 1.5:
                return "Moderate leverage, typical for many industries"
            else:
                return "High leverage may indicate financial risk if cash flows deteriorate"

        elif ratio_name == "roe":
            if value < 0.08:
                return "Below average return on equity, possible efficiency concerns"
            elif value < 0.15:
                return "Moderate return on equity, reasonable efficiency"
            else:
                return "Strong return on equity indicating efficient use of shareholder capital"

        elif ratio_name == "operating_margin":
            if value < 0.08:
                return "Low operating margin may indicate competitive pressure or inefficiencies"
            elif value < 0.15:
                return "Moderate operating margin, typical for many industries"
            else:
                return "Strong operating margin indicating pricing power or operational efficiency"

        # Default interpretation if no specific rule
        return f"Value: {value:.2f}"

    def get_key_ratios_for_sector(self, sector: str) -> List[Dict[str, str]]:
        """
        Get the most important ratios for a specific sector

        Args:
            sector: Market sector name

        Returns:
            List of dictionaries with ratio information
        """
        # Try new config format first
        if sector in self.sector_metrics and 'key_ratios' in self.sector_metrics[sector]:
            specific_ratios = self.sector_metrics[sector]['key_ratios']
        else:
            # Fall back to old config format
            specific_ratios = self.sector_specific_ratios.get(sector, [])

        # Find the categories for each ratio
        ratios_with_categories = []

        for ratio in specific_ratios:
            # Find which category this ratio belongs to
            category = self._determine_ratio_category(ratio)

            ratios_with_categories.append({
                "ratio": ratio,
                "category": category,
                "description": self._get_ratio_description(ratio)
            })

        return ratios_with_categories

    def _determine_ratio_category(self, ratio: str) -> str:
        """
        Determine the category for a given ratio name

        Args:
            ratio: Ratio name

        Returns:
            Category name
        """
        # First check in defined categories
        for category, ratios in self.ratio_categories.items():
            if ratio in ratios:
                return category

        # If not found, try to infer from ratio name
        ratio_lower = ratio.lower()

        if any(term in ratio_lower for term in ["p/e", "p/s", "p/b", "ev/", "price", "valuation"]):
            return "valuation"
        elif any(term in ratio_lower for term in ["margin", "return", "roe", "roa", "roic", "profitability"]):
            return "profitability"
        elif any(term in ratio_lower for term in ["debt", "leverage", "coverage", "equity", "interest"]):
            return "leverage"
        elif any(term in ratio_lower for term in ["current", "quick", "cash", "liquidity"]):
            return "liquidity"
        elif any(term in ratio_lower for term in ["turnover", "efficiency", "cycle", "days", "asset"]):
            return "efficiency"
        elif any(term in ratio_lower for term in ["growth", "increase"]):
            return "growth"

        # Default
        return "other"

    def _get_ratio_description(self, ratio: str) -> str:
        """
        Get description for a specific ratio

        Args:
            ratio: Ratio name

        Returns:
            Description of the ratio
        """
        # Normalize ratio name to match descriptions dictionary
        normalized_ratio = ratio

        # Try exact match first
        if ratio in self.ratio_descriptions:
            return self.ratio_descriptions[ratio]

        # Try case-insensitive match
        ratio_lower = ratio.lower()
        for key in self.ratio_descriptions:
            if key.lower() == ratio_lower:
                return self.ratio_descriptions[key]

        # Try variations
        variations = {
            "pe_ratio": "P/E",
            "ps_ratio": "P/S",
            "pb_ratio": "P/B",
            "p_fcf": "P/FCF",
            "p_cf": "P/C",
            "ev_ebitda": "EV/EBITDA",
            "ev_revenue": "EV/Revenue",
            "debt_to_equity": "Debt/Equity",
            "debt_to_ebitda": "Debt/EBITDA",
            "cash_per_share": "Cash/sh"
        }

        if ratio in variations and variations[ratio] in self.ratio_descriptions:
            return self.ratio_descriptions[variations[ratio]]

        # Return generic description if not found
        return f"Financial ratio measuring {ratio.replace('_', ' ').title()}"

    def calculate_financial_strength(self, financial_data: Dict) -> Dict[str, float]:
        """
        Calculate financial strength score based on liquidity, solvency, and profitability

        Args:
            financial_data: Dictionary containing financial statement data

        Returns:
            Dictionary with financial strength scores
        """
        # Calculate ratios first
        ratios = self.calculate_ratios(financial_data)

        # Initialize scores
        strength_scores = {
            "liquidity_score": 0.0,
            "solvency_score": 0.0,
            "profitability_score": 0.0,
            "overall_score": 0.0
        }

        # Analyze liquidity
        if 'liquidity' in ratios:
            liquidity_factors = []

            # Current ratio
            if 'current_ratio' in ratios['liquidity']:
                cur_ratio = ratios['liquidity']['current_ratio']
                if cur_ratio >= 2.0:
                    liquidity_factors.append(1.0)
                elif cur_ratio >= 1.5:
                    liquidity_factors.append(0.8)
                elif cur_ratio >= 1.0:
                    liquidity_factors.append(0.6)
                elif cur_ratio >= 0.8:
                    liquidity_factors.append(0.3)
                else:
                    liquidity_factors.append(0.0)

            # Quick ratio
            if 'quick_ratio' in ratios['liquidity']:
                quick_ratio = ratios['liquidity']['quick_ratio']
                if quick_ratio >= 1.5:
                    liquidity_factors.append(1.0)
                elif quick_ratio >= 1.0:
                    liquidity_factors.append(0.8)
                elif quick_ratio >= 0.7:
                    liquidity_factors.append(0.5)
                elif quick_ratio >= 0.5:
                    liquidity_factors.append(0.3)
                else:
                    liquidity_factors.append(0.0)

            # Cash ratio
            if 'cash_ratio' in ratios['liquidity']:
                cash_ratio = ratios['liquidity']['cash_ratio']
                if cash_ratio >= 0.5:
                    liquidity_factors.append(1.0)
                elif cash_ratio >= 0.3:
                    liquidity_factors.append(0.8)
                elif cash_ratio >= 0.2:
                    liquidity_factors.append(0.5)
                elif cash_ratio >= 0.1:
                    liquidity_factors.append(0.3)
                else:
                    liquidity_factors.append(0.0)

            # Calculate average liquidity score if we have any factors
            if liquidity_factors:
                strength_scores["liquidity_score"] = sum(liquidity_factors) / len(liquidity_factors)

        # Analyze solvency
        if 'leverage' in ratios:
            solvency_factors = []

            # Debt to equity
            if 'debt_to_equity' in ratios['leverage']:
                de_ratio = ratios['leverage']['debt_to_equity']
                if de_ratio <= 0.5:
                    solvency_factors.append(1.0)
                elif de_ratio <= 1.0:
                    solvency_factors.append(0.8)
                elif de_ratio <= 1.5:
                    solvency_factors.append(0.6)
                elif de_ratio <= 2.0:
                    solvency_factors.append(0.3)
                else:
                    solvency_factors.append(0.0)

            # Interest coverage
            if 'interest_coverage' in ratios['leverage']:
                int_cov = ratios['leverage']['interest_coverage']
                if int_cov >= 6.0:
                    solvency_factors.append(1.0)
                elif int_cov >= 4.0:
                    solvency_factors.append(0.8)
                elif int_cov >= 2.0:
                    solvency_factors.append(0.5)
                elif int_cov >= 1.0:
                    solvency_factors.append(0.2)
                else:
                    solvency_factors.append(0.0)

            # Debt to EBITDA
            if 'debt_to_ebitda' in ratios['leverage']:
                debt_ebitda = ratios['leverage']['debt_to_ebitda']
                if debt_ebitda <= 2.0:
                    solvency_factors.append(1.0)
                elif debt_ebitda <= 3.0:
                    solvency_factors.append(0.8)
                elif debt_ebitda <= 4.0:
                    solvency_factors.append(0.6)
                elif debt_ebitda <= 5.0:
                    solvency_factors.append(0.3)
                else:
                    solvency_factors.append(0.0)

            # Calculate average solvency score if we have any factors
            if solvency_factors:
                strength_scores["solvency_score"] = sum(solvency_factors) / len(solvency_factors)

        # Analyze profitability
        if 'profitability' in ratios:
            profit_factors = []

            # ROE
            if 'roe' in ratios['profitability']:
                roe = ratios['profitability']['roe']
                if roe >= 0.20:
                    profit_factors.append(1.0)
                elif roe >= 0.15:
                    profit_factors.append(0.8)
                elif roe >= 0.10:
                    profit_factors.append(0.6)
                elif roe >= 0.05:
                    profit_factors.append(0.3)
                else:
                    profit_factors.append(0.0)

            # Operating margin
            if 'operating_margin' in ratios['profitability']:
                op_margin = ratios['profitability']['operating_margin']
                if op_margin >= 0.20:
                    profit_factors.append(1.0)
                elif op_margin >= 0.15:
                    profit_factors.append(0.8)
                elif op_margin >= 0.10:
                    profit_factors.append(0.6)
                elif op_margin >= 0.05:
                    profit_factors.append(0.3)
                else:
                    profit_factors.append(0.0)

            # Net margin
            if 'net_margin' in ratios['profitability']:
                net_margin = ratios['profitability']['net_margin']
                if net_margin >= 0.15:
                    profit_factors.append(1.0)
                elif net_margin >= 0.10:
                    profit_factors.append(0.8)
                elif net_margin >= 0.05:
                    profit_factors.append(0.6)
                elif net_margin >= 0.02:
                    profit_factors.append(0.3)
                else:
                    profit_factors.append(0.0)

            # Calculate average profitability score if we have any factors
            if profit_factors:
                strength_scores["profitability_score"] = sum(profit_factors) / len(profit_factors)

        # Calculate overall financial strength score
        scores_to_average = []
        for score_name in ["liquidity_score", "solvency_score", "profitability_score"]:
            if strength_scores[score_name] > 0:
                scores_to_average.append(strength_scores[score_name])

        if scores_to_average:
            strength_scores["overall_score"] = sum(scores_to_average) / len(scores_to_average)

        return strength_scores

    def analyze_financial_sustainability(self, financial_data: Dict, sector: str) -> Dict[str, Any]:
        """
        Analyze financial sustainability and future outlook based on historical data

        Args:
            financial_data: Dictionary containing financial statement data
            sector: Market sector of the company

        Returns:
            Dictionary with sustainability analysis
        """
        # Calculate ratios
        ratios = self.calculate_ratios(financial_data)

        # Initialize results
        sustainability = {
            "financial_strength": {},
            "growth_analysis": {},
            "efficiency_trends": {},
            "risk_factors": [],
            "overall_assessment": "",
            "sustainability_score": 0.0
        }

        # Calculate financial strength scores
        sustainability["financial_strength"] = self.calculate_financial_strength(financial_data)

        # Analyze growth if we have growth ratios
        if 'growth' in ratios and ratios['growth']:
            revenue_growth = ratios['growth'].get('revenue_growth')
            net_income_growth = ratios['growth'].get('net_income_growth')

            # Get sector benchmarks
            benchmarks = self.get_sector_benchmarks(sector)
            benchmark_growth = benchmarks.get('growth', {})
            benchmark_rev_growth = benchmark_growth.get('revenue_growth')
            benchmark_ni_growth = benchmark_growth.get('net_income_growth')

            growth_assessment = {}

            # Revenue growth assessment
            if revenue_growth is not None:
                if benchmark_rev_growth is not None:
                    relative_growth = "above average" if revenue_growth > benchmark_rev_growth else "below average" if revenue_growth < benchmark_rev_growth else "average"
                    growth_assessment["revenue_growth"] = {
                        "value": revenue_growth,
                        "benchmark": benchmark_rev_growth,
                        "assessment": relative_growth,
                        "sustainability": "high" if revenue_growth > 0.15 else "moderate" if revenue_growth > 0.05 else "low"
                    }
                else:
                    growth_assessment["revenue_growth"] = {
                        "value": revenue_growth,
                        "assessment": "high" if revenue_growth > 0.15 else "moderate" if revenue_growth > 0.05 else "low",
                        "sustainability": "high" if revenue_growth > 0.15 else "moderate" if revenue_growth > 0.05 else "low"
                    }

            # Net income growth assessment
            if net_income_growth is not None:
                if benchmark_ni_growth is not None:
                    relative_growth = "above average" if net_income_growth > benchmark_ni_growth else "below average" if net_income_growth < benchmark_ni_growth else "average"
                    growth_assessment["net_income_growth"] = {
                        "value": net_income_growth,
                        "benchmark": benchmark_ni_growth,
                        "assessment": relative_growth,
                        "sustainability": "high" if net_income_growth > 0.15 else "moderate" if net_income_growth > 0.05 else "low"
                    }
                else:
                    growth_assessment["net_income_growth"] = {
                        "value": net_income_growth,
                        "assessment": "high" if net_income_growth > 0.15 else "moderate" if net_income_growth > 0.05 else "low",
                        "sustainability": "high" if net_income_growth > 0.15 else "moderate" if net_income_growth > 0.05 else "low"
                    }

            sustainability["growth_analysis"] = growth_assessment

            # Analyze efficiency trends
        if 'efficiency' in ratios and ratios['efficiency']:
            efficiency_trends = {}

            # Asset turnover assessment
            if 'asset_turnover' in ratios['efficiency']:
                asset_turnover = ratios['efficiency']['asset_turnover']
                efficiency_trends["asset_turnover"] = {
                    "value": asset_turnover,
                    "assessment": "high" if asset_turnover > 1.0 else "moderate" if asset_turnover > 0.5 else "low"
                }

            # Inventory turnover assessment
            if 'inventory_turnover' in ratios['efficiency']:
                inventory_turnover = ratios['efficiency']['inventory_turnover']
                efficiency_trends["inventory_turnover"] = {
                    "value": inventory_turnover,
                    "assessment": "high" if inventory_turnover > 8.0 else "moderate" if inventory_turnover > 4.0 else "low"
                }

            # Cash conversion cycle assessment
            if 'cash_conversion_cycle' in ratios['efficiency']:
                ccc = ratios['efficiency']['cash_conversion_cycle']
                efficiency_trends["cash_conversion_cycle"] = {
                    "value": ccc,
                    "assessment": "good" if ccc < 30 else "moderate" if ccc < 60 else "poor"
                }

            sustainability["efficiency_trends"] = efficiency_trends

            # Identify risk factors
        risk_factors = []

        # Liquidity risks
        if 'liquidity' in ratios:
            if ratios['liquidity'].get('current_ratio', 999) < 1.0:
                risk_factors.append("Low current ratio indicates potential short-term liquidity issues")
            if ratios['liquidity'].get('quick_ratio', 999) < 0.8:
                risk_factors.append("Low quick ratio suggests potential liquidity challenges if sales slow down")

        # Leverage risks
        if 'leverage' in ratios:
            if ratios['leverage'].get('debt_to_equity', 0) > 2.0:
                risk_factors.append("High debt-to-equity ratio indicates significant financial leverage risk")
            if ratios['leverage'].get('interest_coverage', 999) < 2.0:
                risk_factors.append("Low interest coverage ratio suggests vulnerability to interest rate increases")
            if ratios['leverage'].get('debt_to_ebitda', 0) > 5.0:
                risk_factors.append("High debt-to-EBITDA ratio may indicate debt repayment challenges")

        # Profitability risks
        if 'profitability' in ratios:
            if ratios['profitability'].get('operating_margin', 999) < 0.05:
                risk_factors.append("Low operating margin provides limited buffer against cost increases")
            if ratios['profitability'].get('net_margin', 999) < 0.02:
                risk_factors.append("Very low net margin indicates minimal profitability")

        # Efficiency risks
        if 'efficiency' in ratios:
            if ratios['efficiency'].get('asset_turnover', 999) < 0.3:
                risk_factors.append("Low asset turnover suggests inefficient use of assets")
            if 'inventory_turnover' in ratios['efficiency'] and ratios['efficiency']['inventory_turnover'] < 3.0:
                risk_factors.append(
                    "Low inventory turnover may indicate inventory management issues or obsolescence risk")

        sustainability["risk_factors"] = risk_factors

        # Calculate overall sustainability score
        sustainability_factors = []

        if "overall_score" in sustainability["financial_strength"]:
            sustainability_factors.append(sustainability["financial_strength"]["overall_score"])

        # Growth factor
        growth_factor = 0.0
        if 'growth' in ratios:
            growth_scores = []
            if 'revenue_growth' in ratios['growth'] and ratios['growth']['revenue_growth'] > 0:
                rev_growth_score = min(1.0, ratios['growth']['revenue_growth'] / 0.15)  # Max score at 15% growth
                growth_scores.append(rev_growth_score)
            if 'net_income_growth' in ratios['growth'] and ratios['growth']['net_income_growth'] > 0:
                ni_growth_score = min(1.0, ratios['growth']['net_income_growth'] / 0.15)  # Max score at 15% growth
                growth_scores.append(ni_growth_score)

            if growth_scores:
                growth_factor = sum(growth_scores) / len(growth_scores)
                sustainability_factors.append(growth_factor)

        # Efficiency factor
        efficiency_factor = 0.0
        if 'efficiency' in ratios:
            efficiency_scores = []
            if 'asset_turnover' in ratios['efficiency']:
                asset_turn_score = min(1.0, ratios['efficiency']['asset_turnover'])  # Max score at 1.0
                efficiency_scores.append(asset_turn_score)
            if 'inventory_turnover' in ratios['efficiency']:
                inv_turn_score = min(1.0, ratios['efficiency']['inventory_turnover'] / 8.0)  # Max score at 8.0
                efficiency_scores.append(inv_turn_score)

            if efficiency_scores:
                efficiency_factor = sum(efficiency_scores) / len(efficiency_scores)
                sustainability_factors.append(efficiency_factor)

        # Risk penalty based on number of risk factors
        risk_penalty = min(0.5, len(risk_factors) * 0.1)  # Max penalty of 0.5

        # Calculate overall sustainability score
        if sustainability_factors:
            raw_score = sum(sustainability_factors) / len(sustainability_factors)
            sustainability["sustainability_score"] = max(0.0, raw_score - risk_penalty)

            # Overall assessment
            if sustainability["sustainability_score"] >= 0.8:
                sustainability["overall_assessment"] = "Strong financial sustainability with solid fundamentals"
            elif sustainability["sustainability_score"] >= 0.6:
                sustainability["overall_assessment"] = "Good financial sustainability with some minor concerns"
            elif sustainability["sustainability_score"] >= 0.4:
                sustainability[
                    "overall_assessment"] = "Moderate financial sustainability with notable areas for improvement"
            elif sustainability["sustainability_score"] >= 0.2:
                sustainability["overall_assessment"] = "Weak financial sustainability with significant concerns"
            else:
                sustainability["overall_assessment"] = "Poor financial sustainability with major structural issues"
        else:
            sustainability["sustainability_score"] = 0.0
            sustainability["overall_assessment"] = "Insufficient data to assess financial sustainability"

        return sustainability

    def analyze_business_lifecycle(self, financial_data: Dict, years: int = 3) -> Dict[str, Any]:
        """
        Analyze where a company is in its business lifecycle based on financial metrics

        Args:
            financial_data: Dictionary containing financial statement data over multiple years
            years: Number of years of data to analyze

        Returns:
            Dictionary with business lifecycle analysis
        """
        # We need multiple years of data for proper lifecycle analysis
        income_stmt = financial_data.get('income_statement')
        if not isinstance(income_stmt, pd.DataFrame) or income_stmt.shape[1] < 2:
            return {
                "lifecycle_stage": "Unknown",
                "confidence": 0.0,
                "characteristics": [],
                "description": "Insufficient historical data to determine business lifecycle stage"
            }

        # Get relevant metrics for years available (up to the requested number)
        available_years = min(years, income_stmt.shape[1])

        # Extract revenue for each year
        revenue = []
        for i in range(available_years):
            if 'Total Revenue' in income_stmt.iloc[:, i]:
                revenue.append(income_stmt.iloc[:, i]['Total Revenue'])
            else:
                # Try alternate field names
                for field in ['Revenue', 'Net Revenue', 'Sales', 'Net Sales']:
                    if field in income_stmt.iloc[:, i]:
                        revenue.append(income_stmt.iloc[:, i][field])
                        break

        # Ensure we have enough revenue data
        if len(revenue) < 2:
            return {
                "lifecycle_stage": "Unknown",
                "confidence": 0.0,
                "characteristics": [],
                "description": "Insufficient revenue data to determine business lifecycle stage"
            }

        # Calculate revenue growth rates
        revenue_growth = [(revenue[i] / revenue[i + 1] - 1) for i in range(len(revenue) - 1)]
        avg_revenue_growth = sum(revenue_growth) / len(revenue_growth)

        # Extract profits for each year
        net_income = []
        for i in range(available_years):
            if 'Net Income' in income_stmt.iloc[:, i]:
                net_income.append(income_stmt.iloc[:, i]['Net Income'])
            else:
                # Try alternate field names
                for field in ['Net Profit', 'Net Earnings', 'Profit After Tax']:
                    if field in income_stmt.iloc[:, i]:
                        net_income.append(income_stmt.iloc[:, i][field])
                        break

        # Calculate profit margin if we have both revenue and net income
        profit_margin = []
        if len(net_income) > 0:
            for i in range(min(len(revenue), len(net_income))):
                if revenue[i] > 0:
                    profit_margin.append(net_income[i] / revenue[i])

        avg_profit_margin = sum(profit_margin) / len(profit_margin) if profit_margin else None

        # Determine lifecycle stage based on financial characteristics
        # Define threshold values for classification
        startup_growth = 0.50  # 50%+ annual growth for startup
        growth_stage = 0.15  # 15%+ annual growth for growth stage
        maturity_growth = 0.05  # 5%+ annual growth for maturity

        characteristics = []

        # Lifecycle classification logic
        if avg_revenue_growth >= startup_growth:
            stage = "Startup/Early Stage"
            confidence = min(1.0, avg_revenue_growth / startup_growth)
            characteristics = [
                f"Very high revenue growth ({avg_revenue_growth:.1%} average)",
                "Rapidly increasing market penetration",
                "Revenue growing much faster than industry average"
            ]
            if avg_profit_margin is not None:
                if avg_profit_margin < 0:
                    characteristics.append("Negative profit margins as company invests in growth")
                    confidence += 0.1
                else:
                    characteristics.append("Positive profit margins unusual for early stage")
                    confidence -= 0.1

            description = "Company shows characteristics of early-stage rapid growth, with high revenue increases typical of new market entrants or disruptive innovation. Focus on market share acquisition and scaling operations."

        elif avg_revenue_growth >= growth_stage:
            stage = "Growth Stage"
            confidence = min(1.0, avg_revenue_growth / growth_stage)
            characteristics = [
                f"Strong revenue growth ({avg_revenue_growth:.1%} average)",
                "Established market position",
                "Consistent expansion"
            ]
            if avg_profit_margin is not None:
                if avg_profit_margin > 0:
                    characteristics.append("Positive and improving profit margins")
                    confidence += 0.1
                else:
                    characteristics.append("Negative profit margins unusual for growth stage")
                    confidence -= 0.1

            description = "Company demonstrates solid growth stage characteristics with consistent revenue increases and improving profitability. Typically, this stage involves reinvesting profits to capture additional market share while establishing operational efficiency."

        elif avg_revenue_growth >= maturity_growth:
            stage = "Maturity Stage"
            confidence = min(1.0, (avg_revenue_growth - 0.01) / (maturity_growth - 0.01))
            characteristics = [
                f"Moderate revenue growth ({avg_revenue_growth:.1%} average)",
                "Stable market position",
                "Growth near industry average"
            ]
            if avg_profit_margin is not None:
                if avg_profit_margin > 0.10:
                    characteristics.append("Strong profit margins typical of mature companies")
                    confidence += 0.1
                elif avg_profit_margin > 0:
                    characteristics.append("Positive profit margins")
                else:
                    characteristics.append("Negative profit margins unusual for maturity stage")
                    confidence -= 0.2

            description = "Company shows signs of business maturity with slower but steady growth, stable market position, and focus on operational efficiency. Mature companies typically prioritize profitability and shareholder returns over aggressive expansion."

        elif avg_revenue_growth >= 0:
            stage = "Late Maturity/Early Decline"
            confidence = min(1.0, 1 - (avg_revenue_growth / maturity_growth))
            characteristics = [
                f"Slow revenue growth ({avg_revenue_growth:.1%} average)",
                "Limited organic growth opportunities",
                "Growth below industry average"
            ]
            if avg_profit_margin is not None:
                if avg_profit_margin > 0.15:
                    characteristics.append("High profit margins from cost optimization")
                elif avg_profit_margin > 0:
                    characteristics.append("Stable profit margins")
                else:
                    characteristics.append("Declining profit margins")

            description = "Company exhibits characteristics of late maturity or early decline, with minimal revenue growth and focus on cost control. At this stage, companies often emphasize efficiency, dividends, and may seek growth through acquisitions rather than organic expansion."

        else:  # Negative growth
            stage = "Decline Stage"
            confidence = min(1.0, -avg_revenue_growth / 0.10)  # 10% decline as reference
            characteristics = [
                f"Negative revenue growth ({avg_revenue_growth:.1%} average)",
                "Contracting market presence",
                "Below-industry performance"
            ]
            if avg_profit_margin is not None:
                if avg_profit_margin > 0:
                    characteristics.append("Still maintaining positive profit margins despite decline")
                else:
                    characteristics.append("Negative and deteriorating profit margins")
                    confidence += 0.1

            description = "Company shows clear signs of business decline with contracting revenues and diminishing market relevance. This phase typically requires significant business model transformation, cost-cutting, or consideration of exit strategies if decline is structural rather than cyclical."

        return {
            "lifecycle_stage": stage,
            "confidence": min(1.0, confidence),
            "characteristics": characteristics,
            "description": description,
            "revenue_growth": avg_revenue_growth,
            "profit_margin": avg_profit_margin
        }

    def calculate_du_pont_analysis(self, financial_data: Dict) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Perform DuPont analysis to break down ROE into component factors

        Args:
            financial_data: Dictionary containing financial statement data

        Returns:
            Dictionary with DuPont analysis results
        """
        # Calculate ratios first
        ratios = self.calculate_ratios(financial_data)

        # Get the relevant data from financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if not isinstance(income_stmt, pd.DataFrame) or not isinstance(balance_sheet, pd.DataFrame):
            return {"error": "Missing required financial statements for DuPont analysis"}

        # Extract required values
        income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
        balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()

        # Get net income
        net_income = None
        for field in ['Net Income', 'Net Profit', 'Net Earnings']:
            if field in income:
                net_income = income[field]
                break

        # Get revenue
        revenue = None
        for field in ['Total Revenue', 'Revenue', 'Net Revenue', 'Sales', 'Net Sales']:
            if field in income:
                revenue = income[field]
                break

        # Get total assets
        total_assets = None
        for field in ['Total Assets', 'Assets']:
            if field in balance:
                total_assets = balance[field]
                break

        # Get stockholder equity
        equity = None
        for field in ['Total Stockholder Equity', 'Total Equity', 'Shareholders Equity']:
            if field in balance:
                equity = balance[field]
                break

        # If we're missing any required data, return error
        if None in [net_income, revenue, total_assets, equity] or 0 in [revenue, total_assets, equity]:
            return {"error": "Missing required financial data for DuPont analysis"}

        # Calculate DuPont components
        net_profit_margin = net_income / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / equity

        # Calculate ROE using DuPont formula
        roe_dupont = net_profit_margin * asset_turnover * equity_multiplier

        # Compare with directly calculated ROE
        direct_roe = ratios.get('profitability', {}).get('roe')

        # Calculate contribution of each factor
        total_contribution = abs(net_profit_margin) + abs(asset_turnover) + abs(equity_multiplier)

        profit_margin_contribution = abs(net_profit_margin) / total_contribution
        asset_turnover_contribution = abs(asset_turnover) / total_contribution
        equity_multiplier_contribution = abs(equity_multiplier) / total_contribution

        return {
            "roe_dupont": roe_dupont,
            "roe_direct": direct_roe,
            "components": {
                "net_profit_margin": net_profit_margin,
                "asset_turnover": asset_turnover,
                "equity_multiplier": equity_multiplier
            },
            "contributions": {
                "net_profit_margin": profit_margin_contribution,
                "asset_turnover": asset_turnover_contribution,
                "equity_multiplier": equity_multiplier_contribution
            },
            "interpretation": self._interpret_dupont_analysis(net_profit_margin, asset_turnover, equity_multiplier)
        }

    def _interpret_dupont_analysis(self, profit_margin: float, asset_turnover: float, equity_multiplier: float) -> str:
        """
        Provide interpretation of DuPont analysis results

        Args:
            profit_margin: Net profit margin
            asset_turnover: Asset turnover
            equity_multiplier: Equity multiplier

        Returns:
            Interpretation string
        """
        interpretations = []

        # Interpret profit margin
        if profit_margin > 0.15:
            interpretations.append(
                "The company has excellent profit margins, indicating strong pricing power, cost control, or premium positioning.")
        elif profit_margin > 0.10:
            interpretations.append("The company has good profit margins, better than many competitors.")
        elif profit_margin > 0.05:
            interpretations.append("The company has moderate profit margins, typical for its industry.")
        elif profit_margin > 0:
            interpretations.append(
                "The company has relatively low profit margins, which may limit its ability to generate returns.")
        else:
            interpretations.append(
                "The company is unprofitable, which is a fundamental challenge to creating shareholder value.")

        # Interpret asset turnover
        if asset_turnover > 1.5:
            interpretations.append(
                "Asset utilization is excellent, generating significant revenue from its asset base.")
        elif asset_turnover > 1.0:
            interpretations.append("Asset utilization is good, with efficient use of assets to generate sales.")
        elif asset_turnover > 0.5:
            interpretations.append("Asset utilization is moderate, with room for improvement in asset efficiency.")
        else:
            interpretations.append(
                "Asset utilization is low, indicating potential inefficiency in using assets to generate revenue.")

        # Interpret equity multiplier
        if equity_multiplier > 4.0:
            interpretations.append(
                "The company has very high financial leverage, which amplifies both returns and risks.")
        elif equity_multiplier > 2.5:
            interpretations.append(
                "The company has high financial leverage, increasing potential returns but also financial risk.")
        elif equity_multiplier > 1.5:
            interpretations.append("The company has moderate financial leverage, typical for its industry.")
        else:
            interpretations.append(
                "The company has conservative financial leverage, reducing financial risk but potentially limiting returns.")

        # Overall interpretation
        if profit_margin > 0:
            if profit_margin > 0.10 and asset_turnover > 1.0:
                overall = "The company generates strong returns through both high profitability and efficient asset utilization."
            elif profit_margin > 0.10:
                overall = "The company's returns are primarily driven by strong profitability rather than asset efficiency."
            elif asset_turnover > 1.0:
                overall = "The company's returns are primarily driven by efficient asset utilization rather than high margins."
            elif equity_multiplier > 2.5:
                overall = "The company relies significantly on financial leverage to generate returns for shareholders."
            else:
                overall = "The company shows balanced but moderate performance across profitability, efficiency, and leverage."
        else:
            overall = "The company's negative profitability indicates fundamental business challenges that need to be addressed."

        return overall + " " + " ".join(interpretations)