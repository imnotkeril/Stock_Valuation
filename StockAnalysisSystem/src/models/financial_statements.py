import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('financial_statements')


class FinancialStatementsAnalyzer:
    """
    Class for analyzing financial statements of companies.
    Provides functionality for horizontal, vertical, and common-size analysis
    of income statements, balance sheets, and cash flow statements.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize financial statements analyzer with data loader

        Args:
            data_loader: Optional DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        logger.info("Initialized FinancialStatementsAnalyzer")

    def get_financial_statements(self, ticker: str, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get all financial statements for a company

        Args:
            ticker: Company ticker symbol
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with income statement, balance sheet, and cash flow
        """
        try:
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income', 'annual', force_refresh)
            balance_sheet = self.data_loader.get_financial_statements(ticker, 'balance', 'annual', force_refresh)
            cash_flow = self.data_loader.get_financial_statements(ticker, 'cash', 'annual', force_refresh)

            # Also get quarterly data for more recent analysis
            income_stmt_quarterly = self.data_loader.get_financial_statements(ticker, 'income', 'quarterly',
                                                                              force_refresh)
            balance_sheet_quarterly = self.data_loader.get_financial_statements(ticker, 'balance', 'quarterly',
                                                                                force_refresh)
            cash_flow_quarterly = self.data_loader.get_financial_statements(ticker, 'cash', 'quarterly', force_refresh)

            statements = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'income_statement_quarterly': income_stmt_quarterly,
                'balance_sheet_quarterly': balance_sheet_quarterly,
                'cash_flow_quarterly': cash_flow_quarterly
            }

            return statements
        except Exception as e:
            logger.error(f"Error fetching financial statements for {ticker}: {e}")
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'income_statement_quarterly': pd.DataFrame(),
                'balance_sheet_quarterly': pd.DataFrame(),
                'cash_flow_quarterly': pd.DataFrame()
            }

    def perform_horizontal_analysis(self, statement: pd.DataFrame) -> pd.DataFrame:
        """
        Perform horizontal analysis (year-over-year changes) on a financial statement

        Args:
            statement: Financial statement DataFrame

        Returns:
            DataFrame with horizontal analysis results
        """
        if statement.empty or statement.shape[1] < 2:
            logger.warning("Not enough data for horizontal analysis")
            return pd.DataFrame()

        try:
            # Make a copy of the original statement
            horizontal_analysis = statement.copy()

            # Calculate year-over-year percentage changes
            for i in range(statement.shape[1] - 1):
                current_year = statement.iloc[:, i]
                prev_year = statement.iloc[:, i + 1]

                # Calculate percentage change
                pct_change = (current_year - prev_year) / prev_year.abs()

                # Create a column name for the change
                col_name = f"{statement.columns[i]}_vs_{statement.columns[i + 1]}"

                # Add the percentage change as a new column
                horizontal_analysis[col_name] = pct_change

            return horizontal_analysis
        except Exception as e:
            logger.error(f"Error performing horizontal analysis: {e}")
            return pd.DataFrame()

    def perform_vertical_analysis(self, statement: pd.DataFrame, statement_type: str) -> pd.DataFrame:
        """
        Perform vertical analysis (common-size analysis) on a financial statement

        Args:
            statement: Financial statement DataFrame
            statement_type: Type of statement ('income', 'balance', 'cash')

        Returns:
            DataFrame with vertical analysis results
        """
        if statement.empty:
            logger.warning("Empty statement provided for vertical analysis")
            return pd.DataFrame()

        try:
            # Make a copy of the original statement
            vertical_analysis = statement.copy()

            # Determine the base value for each statement type
            base_items = {
                'income': ['Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'],
                'balance': ['Total Assets', 'Assets'],
                'cash': ['Operating Cash Flow', 'Net Cash Provided by Operating Activities']
            }

            # Get appropriate base items for this statement type
            base_item_options = base_items.get(statement_type, base_items['income'])

            # Find the first matching base item in the statement
            base_item = None
            for item in base_item_options:
                if item in statement.index:
                    base_item = item
                    break

            if base_item is None:
                logger.warning(f"Could not find base item for {statement_type} statement")
                return vertical_analysis

            # Perform vertical analysis for each column (year)
            for col in statement.columns:
                base_value = statement.loc[base_item, col]

                if base_value != 0:  # Avoid division by zero
                    # Calculate each item as a percentage of the base value
                    vertical_analysis[f"{col}_pct"] = statement[col] / base_value
                else:
                    vertical_analysis[f"{col}_pct"] = np.nan

            return vertical_analysis
        except Exception as e:
            logger.error(f"Error performing vertical analysis: {e}")
            return pd.DataFrame()

    def calculate_financial_statement_metrics(self, statements: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate various financial metrics from financial statements

        Args:
            statements: Dictionary containing financial statements

        Returns:
            Dictionary with calculated financial metrics
        """
        metrics = {
            'profitability': {},
            'liquidity': {},
            'efficiency': {},
            'solvency': {},
            'growth': {},
            'cash_flow': {}
        }

        try:
            income_stmt = statements.get('income_statement')
            balance_sheet = statements.get('balance_sheet')
            cash_flow = statements.get('cash_flow')

            # Check if we have the necessary data
            if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
                logger.warning("Missing required financial statements for metrics calculation")
                return metrics

            # Use the most recent year's data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

            # Previous year's data for growth calculations
            prev_income = income_stmt.iloc[:, 1] if income_stmt.shape[1] > 1 else None
            prev_balance = balance_sheet.iloc[:, 1] if balance_sheet.shape[1] > 1 else None

            # Calculate profitability metrics
            self._calculate_profitability_metrics(metrics, latest_income, latest_balance)

            # Calculate liquidity metrics
            self._calculate_liquidity_metrics(metrics, latest_balance)

            # Calculate efficiency metrics
            self._calculate_efficiency_metrics(metrics, latest_income, latest_balance)

            # Calculate solvency metrics
            self._calculate_solvency_metrics(metrics, latest_income, latest_balance, latest_cash_flow)

            # Calculate growth metrics
            if prev_income is not None and prev_balance is not None:
                self._calculate_growth_metrics(metrics, latest_income, prev_income, latest_balance, prev_balance)

            # Calculate cash flow metrics
            if latest_cash_flow is not None:
                self._calculate_cash_flow_metrics(metrics, latest_cash_flow, latest_income, latest_balance)

            return metrics
        except Exception as e:
            logger.error(f"Error calculating financial statement metrics: {e}")
            return metrics

    def _calculate_profitability_metrics(self, metrics: Dict, income: pd.Series, balance: pd.Series):
        """Calculate profitability metrics and add them to the metrics dictionary"""
        # Flexibly try different field names from financial statements
        revenue = self._get_value_from_alternatives(income, [
            'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
        ])

        gross_profit = self._get_value_from_alternatives(income, [
            'Gross Profit', 'Gross Income'
        ])

        operating_income = self._get_value_from_alternatives(income, [
            'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
        ])

        net_income = self._get_value_from_alternatives(income, [
            'Net Income', 'Net Profit', 'Net Earnings'
        ])

        total_assets = self._get_value_from_alternatives(balance, [
            'Total Assets', 'Assets'
        ])

        equity = self._get_value_from_alternatives(balance, [
            'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
        ])

        # Calculate metrics if we have the required values
        if revenue is not None:
            if gross_profit is not None:
                # Gross Margin = Gross Profit / Revenue
                metrics['profitability']['gross_margin'] = gross_profit / revenue

            if operating_income is not None:
                # Operating Margin = Operating Income / Revenue
                metrics['profitability']['operating_margin'] = operating_income / revenue

            if net_income is not None:
                # Net Profit Margin = Net Income / Revenue
                metrics['profitability']['net_margin'] = net_income / revenue

        if net_income is not None:
            if total_assets is not None and total_assets != 0:
                # Return on Assets (ROA) = Net Income / Total Assets
                metrics['profitability']['roa'] = net_income / total_assets

            if equity is not None and equity != 0:
                # Return on Equity (ROE) = Net Income / Shareholders' Equity
                metrics['profitability']['roe'] = net_income / equity

        # Calculate EBITDA if possible
        ebitda = operating_income
        if operating_income is not None:
            depreciation_amortization = self._get_value_from_alternatives(income, [
                'Depreciation and Amortization', 'Depreciation', 'Amortization'
            ])

            if depreciation_amortization is not None:
                ebitda = operating_income + depreciation_amortization

                if revenue is not None and revenue != 0:
                    # EBITDA Margin = EBITDA / Revenue
                    metrics['profitability']['ebitda_margin'] = ebitda / revenue

    def _calculate_liquidity_metrics(self, metrics: Dict, balance: pd.Series):
        """Calculate liquidity metrics and add them to the metrics dictionary"""
        current_assets = self._get_value_from_alternatives(balance, [
            'Total Current Assets', 'Current Assets'
        ])

        current_liabilities = self._get_value_from_alternatives(balance, [
            'Total Current Liabilities', 'Current Liabilities'
        ])

        cash_and_equivalents = self._get_value_from_alternatives(balance, [
            'Cash and Cash Equivalents', 'Cash and Equivalents', 'Cash'
        ])

        short_term_investments = self._get_value_from_alternatives(balance, [
            'Short Term Investments', 'Marketable Securities'
        ])

        inventory = self._get_value_from_alternatives(balance, [
            'Inventory', 'Inventories'
        ])

        # Calculate metrics if we have the required values
        if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
            # Current Ratio = Current Assets / Current Liabilities
            metrics['liquidity']['current_ratio'] = current_assets / current_liabilities

            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            if inventory is not None:
                metrics['liquidity']['quick_ratio'] = (current_assets - inventory) / current_liabilities

            # Cash Ratio = (Cash + Short-term Investments) / Current Liabilities
            cash_plus_investments = cash_and_equivalents
            if short_term_investments is not None:
                cash_plus_investments = cash_and_equivalents + short_term_investments

            if cash_plus_investments is not None:
                metrics['liquidity']['cash_ratio'] = cash_plus_investments / current_liabilities

        # Working Capital
        if current_assets is not None and current_liabilities is not None:
            metrics['liquidity']['working_capital'] = current_assets - current_liabilities

    def _calculate_efficiency_metrics(self, metrics: Dict, income: pd.Series, balance: pd.Series):
        """Calculate efficiency metrics and add them to the metrics dictionary"""
        revenue = self._get_value_from_alternatives(income, [
            'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
        ])

        total_assets = self._get_value_from_alternatives(balance, [
            'Total Assets', 'Assets'
        ])

        inventory = self._get_value_from_alternatives(balance, [
            'Inventory', 'Inventories'
        ])

        accounts_receivable = self._get_value_from_alternatives(balance, [
            'Net Receivables', 'Accounts Receivable', 'Trade Receivables'
        ])

        accounts_payable = self._get_value_from_alternatives(balance, [
            'Accounts Payable', 'Trade Payables'
        ])

        cost_of_goods_sold = self._get_value_from_alternatives(income, [
            'Cost of Revenue', 'Cost of Goods Sold', 'COGS', 'Cost of Sales'
        ])

        # Calculate metrics if we have the required values
        if revenue is not None:
            if total_assets is not None and total_assets != 0:
                # Asset Turnover = Revenue / Total Assets
                metrics['efficiency']['asset_turnover'] = revenue / total_assets

            if accounts_receivable is not None and accounts_receivable != 0:
                # Receivables Turnover = Revenue / Accounts Receivable
                metrics['efficiency']['receivables_turnover'] = revenue / accounts_receivable

                # Days Sales Outstanding (DSO) = 365 / Receivables Turnover
                metrics['efficiency']['days_sales_outstanding'] = 365 / metrics['efficiency']['receivables_turnover']

        if cost_of_goods_sold is not None:
            if inventory is not None and inventory != 0:
                # Inventory Turnover = Cost of Goods Sold / Inventory
                metrics['efficiency']['inventory_turnover'] = cost_of_goods_sold / inventory

                # Days Inventory Outstanding (DIO) = 365 / Inventory Turnover
                metrics['efficiency']['days_inventory_outstanding'] = 365 / metrics['efficiency']['inventory_turnover']

            if accounts_payable is not None and accounts_payable != 0:
                # Payables Turnover = Cost of Goods Sold / Accounts Payable
                metrics['efficiency']['payables_turnover'] = cost_of_goods_sold / accounts_payable

                # Days Payables Outstanding (DPO) = 365 / Payables Turnover
                metrics['efficiency']['days_payables_outstanding'] = 365 / metrics['efficiency']['payables_turnover']

        # Calculate Cash Conversion Cycle
        if ('days_inventory_outstanding' in metrics['efficiency'] and
                'days_sales_outstanding' in metrics['efficiency'] and
                'days_payables_outstanding' in metrics['efficiency']):
            # Cash Conversion Cycle = DIO + DSO - DPO
            metrics['efficiency']['cash_conversion_cycle'] = (
                    metrics['efficiency']['days_inventory_outstanding'] +
                    metrics['efficiency']['days_sales_outstanding'] -
                    metrics['efficiency']['days_payables_outstanding']
            )

    def _calculate_solvency_metrics(self, metrics: Dict, income: pd.Series, balance: pd.Series,
                                    cash_flow: pd.Series = None):
        """Calculate solvency metrics and add them to the metrics dictionary"""
        total_assets = self._get_value_from_alternatives(balance, [
            'Total Assets', 'Assets'
        ])

        total_liabilities = self._get_value_from_alternatives(balance, [
            'Total Liabilities', 'Total Liabilities Net Minority Interest'
        ])

        equity = self._get_value_from_alternatives(balance, [
            'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
        ])

        total_debt = self._get_value_from_alternatives(balance, [
            'Total Debt', 'Short-term Debt and Long-term Debt', 'Debt'
        ])

        if total_debt is None:
            # Try to calculate from components
            short_term_debt = self._get_value_from_alternatives(balance, [
                'Short-term Debt', 'Current Debt'
            ])

            long_term_debt = self._get_value_from_alternatives(balance, [
                'Long-term Debt', 'Non-current Debt'
            ])

            if short_term_debt is not None and long_term_debt is not None:
                total_debt = short_term_debt + long_term_debt

        operating_income = self._get_value_from_alternatives(income, [
            'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
        ])

        interest_expense = self._get_value_from_alternatives(income, [
            'Interest Expense', 'Interest and Debt Expense', 'Interest Paid'
        ])

        ebitda = None
        if operating_income is not None:
            depreciation_amortization = self._get_value_from_alternatives(income, [
                'Depreciation and Amortization', 'Depreciation', 'Amortization'
            ])

            if depreciation_amortization is not None:
                ebitda = operating_income + depreciation_amortization

        # Calculate metrics if we have the required values
        if total_assets is not None and total_liabilities is not None:
            # Debt Ratio = Total Liabilities / Total Assets
            if total_assets != 0:
                metrics['solvency']['debt_ratio'] = total_liabilities / total_assets

        if equity is not None:
            if total_assets is not None and total_assets != 0:
                # Equity Ratio = Total Equity / Total Assets
                metrics['solvency']['equity_ratio'] = equity / total_assets

            if total_liabilities is not None:
                # Debt-to-Equity Ratio = Total Liabilities / Total Equity
                if equity != 0:
                    metrics['solvency']['debt_to_equity'] = total_liabilities / equity

        if total_debt is not None:
            if equity is not None and equity != 0:
                # Debt-to-Equity Ratio (using Total Debt) = Total Debt / Total Equity
                metrics['solvency']['debt_to_equity_with_total_debt'] = total_debt / equity

            if ebitda is not None and ebitda != 0:
                # Debt-to-EBITDA Ratio = Total Debt / EBITDA
                metrics['solvency']['debt_to_ebitda'] = total_debt / ebitda

        if operating_income is not None and interest_expense is not None and interest_expense != 0:
            # Interest Coverage Ratio = Operating Income / Interest Expense
            metrics['solvency']['interest_coverage'] = operating_income / abs(interest_expense)

        # Calculate cash flow solvency metrics if cash flow data is available
        if cash_flow is not None:
            operating_cash_flow = self._get_value_from_alternatives(cash_flow, [
                'Operating Cash Flow', 'Cash Flow from Operations', 'Net Cash Provided by Operating Activities'
            ])

            if operating_cash_flow is not None:
                if total_liabilities is not None and total_liabilities != 0:
                    # Cash Flow to Debt Ratio = Operating Cash Flow / Total Liabilities
                    metrics['solvency']['cash_flow_to_debt'] = operating_cash_flow / total_liabilities

                if total_debt is not None and total_debt != 0:
                    # Cash Flow to Total Debt = Operating Cash Flow / Total Debt
                    metrics['solvency']['cash_flow_to_total_debt'] = operating_cash_flow / total_debt

    def _calculate_growth_metrics(self, metrics: Dict, current_income: pd.Series, prev_income: pd.Series,
                                  current_balance: pd.Series, prev_balance: pd.Series):
        """Calculate growth metrics and add them to the metrics dictionary"""
        # Get current and previous values for key metrics
        current_revenue = self._get_value_from_alternatives(current_income, [
            'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
        ])

        prev_revenue = self._get_value_from_alternatives(prev_income, [
            'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
        ])

        current_net_income = self._get_value_from_alternatives(current_income, [
            'Net Income', 'Net Profit', 'Net Earnings'
        ])

        prev_net_income = self._get_value_from_alternatives(prev_income, [
            'Net Income', 'Net Profit', 'Net Earnings'
        ])

        current_operating_income = self._get_value_from_alternatives(current_income, [
            'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
        ])

        prev_operating_income = self._get_value_from_alternatives(prev_income, [
            'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
        ])

        current_assets = self._get_value_from_alternatives(current_balance, [
            'Total Assets', 'Assets'
        ])

        prev_assets = self._get_value_from_alternatives(prev_balance, [
            'Total Assets', 'Assets'
        ])

        current_equity = self._get_value_from_alternatives(current_balance, [
            'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
        ])

        prev_equity = self._get_value_from_alternatives(prev_balance, [
            'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
        ])

        # Calculate growth rates
        if current_revenue is not None and prev_revenue is not None and prev_revenue != 0:
            # Revenue Growth Rate = (Current Revenue - Previous Revenue) / Previous Revenue
            metrics['growth']['revenue_growth'] = (current_revenue - prev_revenue) / prev_revenue

        if current_net_income is not None and prev_net_income is not None and prev_net_income != 0:
            # Net Income Growth Rate = (Current Net Income - Previous Net Income) / Previous Net Income
            metrics['growth']['net_income_growth'] = (current_net_income - prev_net_income) / prev_net_income

        if current_operating_income is not None and prev_operating_income is not None and prev_operating_income != 0:
            # Operating Income Growth Rate = (Current Operating Income - Previous Operating Income) / Previous Operating Income
            metrics['growth']['operating_income_growth'] = (
                                                                       current_operating_income - prev_operating_income) / prev_operating_income

        if current_assets is not None and prev_assets is not None and prev_assets != 0:
            # Total Assets Growth Rate = (Current Assets - Previous Assets) / Previous Assets
            metrics['growth']['assets_growth'] = (current_assets - prev_assets) / prev_assets

        if current_equity is not None and prev_equity is not None and prev_equity != 0:
            # Shareholders' Equity Growth Rate = (Current Equity - Previous Equity) / Previous Equity
            metrics['growth']['equity_growth'] = (current_equity - prev_equity) / prev_equity

    def _calculate_cash_flow_metrics(self, metrics: Dict, cash_flow: pd.Series, income: pd.Series, balance: pd.Series):
        """Calculate cash flow metrics and add them to the metrics dictionary"""
        operating_cash_flow = self._get_value_from_alternatives(cash_flow, [
            'Operating Cash Flow', 'Cash Flow from Operations', 'Net Cash Provided by Operating Activities'
        ])

        capital_expenditures = self._get_value_from_alternatives(cash_flow, [
            'Capital Expenditure', 'Capital Expenditures', 'Purchase of Property Plant and Equipment'
        ])

        free_cash_flow = self._get_value_from_alternatives(cash_flow, [
            'Free Cash Flow', 'FCF'
        ])

        # Calculate Free Cash Flow if not directly available
        if free_cash_flow is None and operating_cash_flow is not None and capital_expenditures is not None:
            free_cash_flow = operating_cash_flow + capital_expenditures  # Note: CapEx is usually negative

        net_income = self._get_value_from_alternatives(income, [
            'Net Income', 'Net Profit', 'Net Earnings'
        ])

        revenue = self._get_value_from_alternatives(income, [
            'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
        ])

        total_assets = self._get_value_from_alternatives(balance, [
            'Total Assets', 'Assets'
        ])

        market_cap = None  # Need to get this from market data, not from financial statements

        # Calculate metrics if we have the required values
        if operating_cash_flow is not None:
            if net_income is not None and net_income != 0:
                # Operating Cash Flow to Net Income Ratio = Operating Cash Flow / Net Income
                metrics['cash_flow']['ocf_to_net_income'] = operating_cash_flow / net_income

            if revenue is not None and revenue != 0:
                # Operating Cash Flow Ratio = Operating Cash Flow / Revenue
                metrics['cash_flow']['ocf_to_revenue'] = operating_cash_flow / revenue

            if total_assets is not None and total_assets != 0:
                # Operating Cash Flow Return on Assets = Operating Cash Flow / Total Assets
                metrics['cash_flow']['ocf_to_assets'] = operating_cash_flow / total_assets

        if free_cash_flow is not None:
            if revenue is not None and revenue != 0:
                # Free Cash Flow to Revenue = Free Cash Flow / Revenue
                metrics['cash_flow']['fcf_to_revenue'] = free_cash_flow / revenue

            if net_income is not None and net_income != 0:
                # Free Cash Flow to Net Income = Free Cash Flow / Net Income
                metrics['cash_flow']['fcf_to_net_income'] = free_cash_flow / net_income

            if market_cap is not None and market_cap != 0:
                # Free Cash Flow Yield = Free Cash Flow / Market Cap
                metrics['cash_flow']['fcf_yield'] = free_cash_flow / market_cap

        if capital_expenditures is not None:
            if revenue is not None and revenue != 0:
                # Capital Expenditure to Revenue Ratio = Capital Expenditure / Revenue
                metrics['cash_flow']['capex_to_revenue'] = abs(capital_expenditures) / revenue

            if operating_cash_flow is not None and operating_cash_flow != 0:
                # Capital Expenditure to Operating Cash Flow Ratio = Capital Expenditure / Operating Cash Flow
                metrics['cash_flow']['capex_to_ocf'] = abs(capital_expenditures) / operating_cash_flow

    def _get_value_from_alternatives(self, data_series: pd.Series, field_alternatives: List[str]) -> Optional[float]:
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
            if field in data_series.index:
                return data_series[field]
        return None

    def analyze_balance_sheet_structure(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure of a balance sheet (assets and liabilities)

        Args:
            balance_sheet: Balance sheet DataFrame

        Returns:
            Dictionary with balance sheet structure analysis
        """
        if balance_sheet.empty:
            logger.warning("Empty balance sheet provided for structure analysis")
            return {}

        try:
            # Get the most recent data
            latest_balance = balance_sheet.iloc[:, 0]

            # Extract key balance sheet components
            total_assets = self._get_value_from_alternatives(latest_balance, [
                'Total Assets', 'Assets'
            ])

            current_assets = self._get_value_from_alternatives(latest_balance, [
                'Total Current Assets', 'Current Assets'
            ])

            non_current_assets = self._get_value_from_alternatives(latest_balance, [
                'Total Non-current Assets', 'Non-current Assets', 'Long-term Assets'
            ])

            if non_current_assets is None and total_assets is not None and current_assets is not None:
                non_current_assets = total_assets - current_assets

            total_liabilities = self._get_value_from_alternatives(latest_balance, [
                'Total Liabilities', 'Total Liabilities Net Minority Interest'
            ])

            current_liabilities = self._get_value_from_alternatives(latest_balance, [
                'Total Current Liabilities', 'Current Liabilities'
            ])

            non_current_liabilities = self._get_value_from_alternatives(latest_balance, [
                'Total Non-current Liabilities', 'Non-current Liabilities', 'Long-term Liabilities'
            ])

            if non_current_liabilities is None and total_liabilities is not None and current_liabilities is not None:
                non_current_liabilities = total_liabilities - current_liabilities

            equity = self._get_value_from_alternatives(latest_balance, [
                'Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'
            ])

            # Get key asset components
            cash_and_equivalents = self._get_value_from_alternatives(latest_balance, [
                'Cash and Cash Equivalents', 'Cash and Equivalents', 'Cash'
            ])

            short_term_investments = self._get_value_from_alternatives(latest_balance, [
                'Short Term Investments', 'Marketable Securities'
            ])

            accounts_receivable = self._get_value_from_alternatives(latest_balance, [
                'Net Receivables', 'Accounts Receivable', 'Trade Receivables'
            ])

            inventory = self._get_value_from_alternatives(latest_balance, [
                'Inventory', 'Inventories'
            ])

            property_plant_equipment = self._get_value_from_alternatives(latest_balance, [
                'Property Plant and Equipment', 'Fixed Assets', 'PP&E'
            ])

            intangible_assets = self._get_value_from_alternatives(latest_balance, [
                'Intangible Assets', 'Goodwill and Intangible Assets'
            ])

            goodwill = self._get_value_from_alternatives(latest_balance, [
                'Goodwill', 'Goodwill and Other Intangible Assets'
            ])

            # Get key liability components
            accounts_payable = self._get_value_from_alternatives(latest_balance, [
                'Accounts Payable', 'Trade Payables'
            ])

            short_term_debt = self._get_value_from_alternatives(latest_balance, [
                'Short-term Debt', 'Current Debt', 'Short-term Borrowings'
            ])

            long_term_debt = self._get_value_from_alternatives(latest_balance, [
                'Long-term Debt', 'Non-current Debt'
            ])

            # Calculate structure percentages
            structure = {'assets': {}, 'liabilities_and_equity': {}}

            # Assets structure
            if total_assets and total_assets != 0:
                if current_assets is not None:
                    structure['assets']['current_assets_pct'] = current_assets / total_assets

                if non_current_assets is not None:
                    structure['assets']['non_current_assets_pct'] = non_current_assets / total_assets

                if cash_and_equivalents is not None:
                    structure['assets']['cash_and_equivalents_pct'] = cash_and_equivalents / total_assets

                if short_term_investments is not None:
                    structure['assets']['short_term_investments_pct'] = short_term_investments / total_assets

                if accounts_receivable is not None:
                    structure['assets']['accounts_receivable_pct'] = accounts_receivable / total_assets

                if inventory is not None:
                    structure['assets']['inventory_pct'] = inventory / total_assets

                if property_plant_equipment is not None:
                    structure['assets']['property_plant_equipment_pct'] = property_plant_equipment / total_assets

                if intangible_assets is not None:
                    structure['assets']['intangible_assets_pct'] = intangible_assets / total_assets

                if goodwill is not None:
                    structure['assets']['goodwill_pct'] = goodwill / total_assets

            # Liabilities and equity structure
            total_liab_and_equity = total_assets  # By accounting equation, these should be equal

            if total_liab_and_equity and total_liab_and_equity != 0:
                if total_liabilities is not None:
                    structure['liabilities_and_equity'][
                        'total_liabilities_pct'] = total_liabilities / total_liab_and_equity

                if current_liabilities is not None:
                    structure['liabilities_and_equity'][
                        'current_liabilities_pct'] = current_liabilities / total_liab_and_equity

                if non_current_liabilities is not None:
                    structure['liabilities_and_equity'][
                        'non_current_liabilities_pct'] = non_current_liabilities / total_liab_and_equity

                if equity is not None:
                    structure['liabilities_and_equity']['equity_pct'] = equity / total_liab_and_equity

                if accounts_payable is not None:
                    structure['liabilities_and_equity'][
                        'accounts_payable_pct'] = accounts_payable / total_liab_and_equity

                if short_term_debt is not None:
                    structure['liabilities_and_equity']['short_term_debt_pct'] = short_term_debt / total_liab_and_equity

                if long_term_debt is not None:
                    structure['liabilities_and_equity']['long_term_debt_pct'] = long_term_debt / total_liab_and_equity

            # Add absolute values for reference
            structure['absolute_values'] = {
                'total_assets': total_assets,
                'current_assets': current_assets,
                'non_current_assets': non_current_assets,
                'total_liabilities': total_liabilities,
                'current_liabilities': current_liabilities,
                'non_current_liabilities': non_current_liabilities,
                'equity': equity,
                'cash_and_equivalents': cash_and_equivalents,
                'accounts_receivable': accounts_receivable,
                'inventory': inventory,
                'property_plant_equipment': property_plant_equipment,
                'intangible_assets': intangible_assets,
                'goodwill': goodwill,
                'accounts_payable': accounts_payable,
                'short_term_debt': short_term_debt,
                'long_term_debt': long_term_debt
            }

            return structure

        except Exception as e:
            logger.error(f"Error analyzing balance sheet structure: {e}")
            return {}

    def analyze_income_statement_structure(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure of an income statement

        Args:
            income_stmt: Income statement DataFrame

        Returns:
            Dictionary with income statement structure analysis
        """
        if income_stmt.empty:
            logger.warning("Empty income statement provided for structure analysis")
            return {}

        try:
            # Get the most recent data
            latest_income = income_stmt.iloc[:, 0]

            # Extract key income statement components
            revenue = self._get_value_from_alternatives(latest_income, [
                'Total Revenue', 'Revenue', 'Sales', 'Net Revenue', 'Net Sales'
            ])

            cost_of_revenue = self._get_value_from_alternatives(latest_income, [
                'Cost of Revenue', 'Cost of Goods Sold', 'COGS', 'Cost of Sales'
            ])

            gross_profit = self._get_value_from_alternatives(latest_income, [
                'Gross Profit', 'Gross Income'
            ])

            if gross_profit is None and revenue is not None and cost_of_revenue is not None:
                gross_profit = revenue - cost_of_revenue

            operating_expenses = self._get_value_from_alternatives(latest_income, [
                'Operating Expenses', 'Total Operating Expenses'
            ])

            operating_income = self._get_value_from_alternatives(latest_income, [
                'Operating Income', 'Operating Profit', 'EBIT', 'Income From Operations'
            ])

            interest_expense = self._get_value_from_alternatives(latest_income, [
                'Interest Expense', 'Interest and Debt Expense', 'Interest Paid'
            ])

            income_before_tax = self._get_value_from_alternatives(latest_income, [
                'Income Before Tax', 'Pretax Income', 'Earnings Before Tax'
            ])

            income_tax_expense = self._get_value_from_alternatives(latest_income, [
                'Income Tax Expense', 'Tax Provision', 'Tax Expense'
            ])

            net_income = self._get_value_from_alternatives(latest_income, [
                'Net Income', 'Net Profit', 'Net Earnings'
            ])

            # Additional important components
            research_development = self._get_value_from_alternatives(latest_income, [
                'Research and Development', 'R&D Expense', 'Research Expense'
            ])

            sga_expense = self._get_value_from_alternatives(latest_income, [
                'Selling General and Administrative', 'SG&A Expense', 'Selling and Administrative'
            ])

            depreciation_amortization = self._get_value_from_alternatives(latest_income, [
                'Depreciation and Amortization', 'Depreciation', 'Amortization'
            ])

            # Calculate EBITDA if not directly available
            ebitda = self._get_value_from_alternatives(latest_income, [
                'EBITDA', 'Earnings Before Interest Taxes Depreciation and Amortization'
            ])

            if ebitda is None and operating_income is not None and depreciation_amortization is not None:
                ebitda = operating_income + depreciation_amortization

            # Calculate structure percentages
            structure = {'components': {}, 'margins': {}}

            # Structure as percentage of revenue
            if revenue and revenue != 0:
                if cost_of_revenue is not None:
                    structure['components']['cost_of_revenue_pct'] = cost_of_revenue / revenue

                if gross_profit is not None:
                    structure['margins']['gross_margin'] = gross_profit / revenue

                if operating_expenses is not None:
                    structure['components']['operating_expenses_pct'] = operating_expenses / revenue

                if operating_income is not None:
                    structure['margins']['operating_margin'] = operating_income / revenue

                if ebitda is not None:
                    structure['margins']['ebitda_margin'] = ebitda / revenue

                if net_income is not None:
                    structure['margins']['net_margin'] = net_income / revenue

                if research_development is not None:
                    structure['components']['research_development_pct'] = research_development / revenue

                if sga_expense is not None:
                    structure['components']['sga_expense_pct'] = sga_expense / revenue

                if depreciation_amortization is not None:
                    structure['components']['depreciation_amortization_pct'] = depreciation_amortization / revenue

                if interest_expense is not None:
                    structure['components']['interest_expense_pct'] = abs(interest_expense) / revenue

                if income_tax_expense is not None:
                    structure['components']['income_tax_expense_pct'] = income_tax_expense / revenue

            # Calculate effective tax rate
            if income_tax_expense is not None and income_before_tax is not None and income_before_tax != 0:
                structure['metrics'] = {
                    'effective_tax_rate': income_tax_expense / income_before_tax
                }

            # Add absolute values for reference
            structure['absolute_values'] = {
                'revenue': revenue,
                'cost_of_revenue': cost_of_revenue,
                'gross_profit': gross_profit,
                'operating_expenses': operating_expenses,
                'operating_income': operating_income,
                'ebitda': ebitda,
                'interest_expense': interest_expense,
                'income_before_tax': income_before_tax,
                'income_tax_expense': income_tax_expense,
                'net_income': net_income,
                'research_development': research_development,
                'sga_expense': sga_expense,
                'depreciation_amortization': depreciation_amortization
            }

            return structure

        except Exception as e:
            logger.error(f"Error analyzing income statement structure: {e}")
            return {}

    def analyze_cash_flow_statement_structure(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure of a cash flow statement

        Args:
            cash_flow: Cash flow statement DataFrame

        Returns:
            Dictionary with cash flow statement structure analysis
        """
        if cash_flow.empty:
            logger.warning("Empty cash flow statement provided for structure analysis")
            return {}

        try:
            # Get the most recent data
            latest_cash_flow = cash_flow.iloc[:, 0]

            # Extract key cash flow components
            operating_cash_flow = self._get_value_from_alternatives(latest_cash_flow, [
                'Operating Cash Flow', 'Cash Flow from Operations', 'Net Cash Provided by Operating Activities'
            ])

            investing_cash_flow = self._get_value_from_alternatives(latest_cash_flow, [
                'Investing Cash Flow', 'Cash Flow from Investing', 'Net Cash Used in Investing Activities'
            ])

            financing_cash_flow = self._get_value_from_alternatives(latest_cash_flow, [
                'Financing Cash Flow', 'Cash Flow from Financing', 'Net Cash Used in Financing Activities'
            ])

            capital_expenditures = self._get_value_from_alternatives(latest_cash_flow, [
                'Capital Expenditure', 'Capital Expenditures', 'Purchase of Property Plant and Equipment'
            ])

            dividends_paid = self._get_value_from_alternatives(latest_cash_flow, [
                'Dividends Paid', 'Common Stock Dividends Paid', 'Cash Dividends Paid'
            ])

            stock_repurchase = self._get_value_from_alternatives(latest_cash_flow, [
                'Stock Repurchase', 'Repurchase of Common Stock', 'Share Repurchase'
            ])

            free_cash_flow = self._get_value_from_alternatives(latest_cash_flow, [
                'Free Cash Flow', 'FCF'
            ])

            # Calculate free cash flow if not directly available
            if free_cash_flow is None and operating_cash_flow is not None and capital_expenditures is not None:
                free_cash_flow = operating_cash_flow + capital_expenditures  # CapEx is usually negative

            net_change_in_cash = self._get_value_from_alternatives(latest_cash_flow, [
                'Net Change in Cash', 'Net Increase in Cash', 'Net Decrease in Cash'
            ])

            # If net change in cash not directly available, try to calculate
            if net_change_in_cash is None:
                components = [operating_cash_flow, investing_cash_flow, financing_cash_flow]
                if all(x is not None for x in components):
                    net_change_in_cash = sum(components)

            # Calculate structure
            structure = {'composition': {}, 'metrics': {}}

            # Calculate total absolute cash flow
            absolute_cash_flows = []
            for flow in [operating_cash_flow, investing_cash_flow, financing_cash_flow]:
                if flow is not None:
                    absolute_cash_flows.append(abs(flow))

            total_cash_flow_absolute = sum(absolute_cash_flows) if absolute_cash_flows else None

            # Calculate percentages of total cash flow
            if total_cash_flow_absolute and total_cash_flow_absolute != 0:
                if operating_cash_flow is not None:
                    structure['composition']['operating_cash_flow_pct'] = abs(
                        operating_cash_flow) / total_cash_flow_absolute
                    structure['composition'][
                        'operating_cash_flow_direction'] = 'inflow' if operating_cash_flow > 0 else 'outflow'

                if investing_cash_flow is not None:
                    structure['composition']['investing_cash_flow_pct'] = abs(
                        investing_cash_flow) / total_cash_flow_absolute
                    structure['composition'][
                        'investing_cash_flow_direction'] = 'inflow' if investing_cash_flow > 0 else 'outflow'

                if financing_cash_flow is not None:
                    structure['composition']['financing_cash_flow_pct'] = abs(
                        financing_cash_flow) / total_cash_flow_absolute
                    structure['composition'][
                        'financing_cash_flow_direction'] = 'inflow' if financing_cash_flow > 0 else 'outflow'

            # Calculate free cash flow metrics
            if free_cash_flow is not None and operating_cash_flow is not None and operating_cash_flow != 0:
                structure['metrics']['free_cash_flow_to_operating_cash_flow'] = free_cash_flow / operating_cash_flow

            if capital_expenditures is not None and operating_cash_flow is not None and operating_cash_flow != 0:
                structure['metrics']['capex_to_operating_cash_flow'] = abs(capital_expenditures) / operating_cash_flow

            # Shareholder returns metrics
            if dividends_paid is not None and operating_cash_flow is not None and operating_cash_flow > 0:
                structure['metrics']['dividend_payout_from_ocf'] = abs(dividends_paid) / operating_cash_flow

            if stock_repurchase is not None and operating_cash_flow is not None and operating_cash_flow > 0:
                structure['metrics']['repurchase_payout_from_ocf'] = abs(stock_repurchase) / operating_cash_flow

            if dividends_paid is not None and stock_repurchase is not None and free_cash_flow is not None and free_cash_flow > 0:
                total_shareholder_return = abs(dividends_paid) + abs(stock_repurchase) if stock_repurchase else abs(
                    dividends_paid)
                structure['metrics']['shareholder_return_from_fcf'] = total_shareholder_return / free_cash_flow

            # Add absolute values
            structure['absolute_values'] = {
                'operating_cash_flow': operating_cash_flow,
                'investing_cash_flow': investing_cash_flow,
                'financing_cash_flow': financing_cash_flow,
                'capital_expenditures': capital_expenditures,
                'free_cash_flow': free_cash_flow,
                'dividends_paid': dividends_paid,
                'stock_repurchase': stock_repurchase,
                'net_change_in_cash': net_change_in_cash
            }

            return structure

        except Exception as e:
            logger.error(f"Error analyzing cash flow statement structure: {e}")
            return {}

    def analyze_financial_health(self, statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform a comprehensive financial health analysis

        Args:
            statements: Dictionary containing financial statements

        Returns:
            Dictionary with financial health analysis
        """
        financial_health = {
            'liquidity': {'score': None, 'assessment': None, 'metrics': {}},
            'solvency': {'score': None, 'assessment': None, 'metrics': {}},
            'profitability': {'score': None, 'assessment': None, 'metrics': {}},
            'efficiency': {'score': None, 'assessment': None, 'metrics': {}},
            'growth': {'score': None, 'assessment': None, 'metrics': {}},
            'overall': {'score': None, 'assessment': None}
        }

        try:
            # Calculate all financial metrics
            metrics = self.calculate_financial_statement_metrics(statements)

            # Assess liquidity
            liquidity_metrics = metrics.get('liquidity', {})
            if liquidity_metrics:
                liquidity_score = self._assess_liquidity(liquidity_metrics)
                financial_health['liquidity']['score'] = liquidity_score
                financial_health['liquidity']['metrics'] = liquidity_metrics
                financial_health['liquidity']['assessment'] = self._get_liquidity_assessment(liquidity_score)

            # Assess solvency
            solvency_metrics = metrics.get('solvency', {})
            if solvency_metrics:
                solvency_score = self._assess_solvency(solvency_metrics)
                financial_health['solvency']['score'] = solvency_score
                financial_health['solvency']['metrics'] = solvency_metrics
                financial_health['solvency']['assessment'] = self._get_solvency_assessment(solvency_score)

            # Assess profitability
            profitability_metrics = metrics.get('profitability', {})
            if profitability_metrics:
                profitability_score = self._assess_profitability(profitability_metrics)
                financial_health['profitability']['score'] = profitability_score
                financial_health['profitability']['metrics'] = profitability_metrics
                financial_health['profitability']['assessment'] = self._get_profitability_assessment(
                    profitability_score)

            # Assess efficiency
            efficiency_metrics = metrics.get('efficiency', {})
            if efficiency_metrics:
                efficiency_score = self._assess_efficiency(efficiency_metrics)
                financial_health['efficiency']['score'] = efficiency_score
                financial_health['efficiency']['metrics'] = efficiency_metrics
                financial_health['efficiency']['assessment'] = self._get_efficiency_assessment(efficiency_score)

            # Assess growth
            growth_metrics = metrics.get('growth', {})
            if growth_metrics:
                growth_score = self._assess_growth(growth_metrics)
                financial_health['growth']['score'] = growth_score
                financial_health['growth']['metrics'] = growth_metrics
                financial_health['growth']['assessment'] = self._get_growth_assessment(growth_score)

            # Calculate overall financial health score
            category_scores = [
                financial_health['liquidity']['score'],
                financial_health['solvency']['score'],
                financial_health['profitability']['score'],
                financial_health['efficiency']['score'],
                financial_health['growth']['score']
            ]

            valid_scores = [score for score in category_scores if score is not None]

            if valid_scores:
                overall_score = sum(valid_scores) / len(valid_scores)
                financial_health['overall']['score'] = overall_score
                financial_health['overall']['assessment'] = self._get_overall_assessment(overall_score)

            return financial_health

        except Exception as e:
            logger.error(f"Error performing financial health analysis: {e}")
            return financial_health

    def _assess_liquidity(self, liquidity_metrics: Dict[str, float]) -> float:
        """Assess liquidity and return a score between 0 and 1"""
        score_components = []

        # Current Ratio assessment
        if 'current_ratio' in liquidity_metrics:
            current_ratio = liquidity_metrics['current_ratio']
            if current_ratio >= 2.0:
                score_components.append(1.0)
            elif current_ratio >= 1.5:
                score_components.append(0.8)
            elif current_ratio >= 1.0:
                score_components.append(0.6)
            elif current_ratio >= 0.8:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Quick Ratio assessment
        if 'quick_ratio' in liquidity_metrics:
            quick_ratio = liquidity_metrics['quick_ratio']
            if quick_ratio >= 1.5:
                score_components.append(1.0)
            elif quick_ratio >= 1.0:
                score_components.append(0.8)
            elif quick_ratio >= 0.8:
                score_components.append(0.6)
            elif quick_ratio >= 0.5:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Cash Ratio assessment
        if 'cash_ratio' in liquidity_metrics:
            cash_ratio = liquidity_metrics['cash_ratio']
            if cash_ratio >= 0.5:
                score_components.append(1.0)
            elif cash_ratio >= 0.3:
                score_components.append(0.8)
            elif cash_ratio >= 0.2:
                score_components.append(0.6)
            elif cash_ratio >= 0.1:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Calculate average score if we have components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return None

    def _assess_solvency(self, solvency_metrics: Dict[str, float]) -> float:
        """Assess solvency and return a score between 0 and 1"""
        score_components = []

        # Debt-to-Equity assessment
        if 'debt_to_equity' in solvency_metrics:
            debt_to_equity = solvency_metrics['debt_to_equity']
            if debt_to_equity <= 0.5:
                score_components.append(1.0)
            elif debt_to_equity <= 1.0:
                score_components.append(0.8)
            elif debt_to_equity <= 1.5:
                score_components.append(0.6)
            elif debt_to_equity <= 2.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Interest Coverage assessment
        if 'interest_coverage' in solvency_metrics:
            interest_coverage = solvency_metrics['interest_coverage']
            if interest_coverage >= 5.0:
                score_components.append(1.0)
            elif interest_coverage >= 3.0:
                score_components.append(0.8)
            elif interest_coverage >= 2.0:
                score_components.append(0.6)
            elif interest_coverage >= 1.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Debt-to-EBITDA assessment
        if 'debt_to_ebitda' in solvency_metrics:
            debt_to_ebitda = solvency_metrics['debt_to_ebitda']
            if debt_to_ebitda <= 2.0:
                score_components.append(1.0)
            elif debt_to_ebitda <= 3.0:
                score_components.append(0.8)
            elif debt_to_ebitda <= 4.0:
                score_components.append(0.6)
            elif debt_to_ebitda <= 5.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Calculate average score if we have components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return None

    def _assess_profitability(self, profitability_metrics: Dict[str, float]) -> float:
        """Assess profitability and return a score between 0 and 1"""
        score_components = []

        # ROE assessment
        if 'roe' in profitability_metrics:
            roe = profitability_metrics['roe']
            if roe >= 0.25:
                score_components.append(1.0)
            elif roe >= 0.15:
                score_components.append(0.8)
            elif roe >= 0.10:
                score_components.append(0.6)
            elif roe >= 0.05:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Operating Margin assessment
        if 'operating_margin' in profitability_metrics:
            operating_margin = profitability_metrics['operating_margin']
            if operating_margin >= 0.25:
                score_components.append(1.0)
            elif operating_margin >= 0.15:
                score_components.append(0.8)
            elif operating_margin >= 0.10:
                score_components.append(0.6)
            elif operating_margin >= 0.05:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Net Margin assessment
        if 'net_margin' in profitability_metrics:
            net_margin = profitability_metrics['net_margin']
            if net_margin >= 0.20:
                score_components.append(1.0)
            elif net_margin >= 0.12:
                score_components.append(0.8)
            elif net_margin >= 0.08:
                score_components.append(0.6)
            elif net_margin >= 0.04:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Calculate average score if we have components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return None

    def _assess_efficiency(self, efficiency_metrics: Dict[str, float]) -> float:
        """Assess efficiency and return a score between 0 and 1"""
        score_components = []

        # Asset Turnover assessment
        if 'asset_turnover' in efficiency_metrics:
            asset_turnover = efficiency_metrics['asset_turnover']
            if asset_turnover >= 1.5:
                score_components.append(1.0)
            elif asset_turnover >= 1.0:
                score_components.append(0.8)
            elif asset_turnover >= 0.7:
                score_components.append(0.6)
            elif asset_turnover >= 0.4:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Inventory Turnover assessment
        if 'inventory_turnover' in efficiency_metrics:
            inventory_turnover = efficiency_metrics['inventory_turnover']
            if inventory_turnover >= 10.0:
                score_components.append(1.0)
            elif inventory_turnover >= 8.0:
                score_components.append(0.8)
            elif inventory_turnover >= 6.0:
                score_components.append(0.6)
            elif inventory_turnover >= 4.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Receivables Turnover assessment
        if 'receivables_turnover' in efficiency_metrics:
            receivables_turnover = efficiency_metrics['receivables_turnover']
            if receivables_turnover >= 12.0:
                score_components.append(1.0)
            elif receivables_turnover >= 8.0:
                score_components.append(0.8)
            elif receivables_turnover >= 6.0:
                score_components.append(0.6)
            elif receivables_turnover >= 4.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Cash Conversion Cycle assessment
        if 'cash_conversion_cycle' in efficiency_metrics:
            ccc = efficiency_metrics['cash_conversion_cycle']
            if ccc <= 30:
                score_components.append(1.0)
            elif ccc <= 45:
                score_components.append(0.8)
            elif ccc <= 60:
                score_components.append(0.6)
            elif ccc <= 75:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Calculate average score if we have components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return None

    def _assess_growth(self, growth_metrics: Dict[str, float]) -> float:
        """Assess growth and return a score between 0 and 1"""
        score_components = []

        # Revenue Growth assessment
        if 'revenue_growth' in growth_metrics:
            revenue_growth = growth_metrics['revenue_growth']
            if revenue_growth >= 0.20:
                score_components.append(1.0)
            elif revenue_growth >= 0.10:
                score_components.append(0.8)
            elif revenue_growth >= 0.05:
                score_components.append(0.6)
            elif revenue_growth >= 0.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Net Income Growth assessment
        if 'net_income_growth' in growth_metrics:
            net_income_growth = growth_metrics['net_income_growth']
            if net_income_growth >= 0.25:
                score_components.append(1.0)
            elif net_income_growth >= 0.15:
                score_components.append(0.8)
            elif net_income_growth >= 0.08:
                score_components.append(0.6)
            elif net_income_growth >= 0.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Assets Growth assessment
        if 'assets_growth' in growth_metrics:
            assets_growth = growth_metrics['assets_growth']
            if assets_growth >= 0.15:
                score_components.append(1.0)
            elif assets_growth >= 0.08:
                score_components.append(0.8)
            elif assets_growth >= 0.04:
                score_components.append(0.6)
            elif assets_growth >= 0.0:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Calculate average score if we have components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return None

    def _get_liquidity_assessment(self, score: float) -> str:
        """Get liquidity assessment based on score"""
        if score is None:
            return "Not enough data to assess liquidity"
        elif score >= 0.8:
            return "Excellent liquidity position with strong ability to meet short-term obligations"
        elif score >= 0.6:
            return "Good liquidity position with adequate ability to meet short-term obligations"
        elif score >= 0.4:
            return "Moderate liquidity position with some concerns about short-term obligations"
        elif score >= 0.2:
            return "Poor liquidity position with significant concerns about short-term obligations"
        else:
            return "Critical liquidity position with severe concerns about meeting short-term obligations"

    def _get_solvency_assessment(self, score: float) -> str:
        """Get solvency assessment based on score"""
        if score is None:
            return "Not enough data to assess solvency"
        elif score >= 0.8:
            return "Excellent solvency position with very low financial risk"
        elif score >= 0.6:
            return "Good solvency position with manageable financial risk"
        elif score >= 0.4:
            return "Moderate solvency position with potential financial risk concerns"
        elif score >= 0.2:
            return "Poor solvency position with significant financial risk"
        else:
            return "Critical solvency position with severe financial risk"

    def _get_profitability_assessment(self, score: float) -> str:
        """Get profitability assessment based on score"""
        if score is None:
            return "Not enough data to assess profitability"
        elif score >= 0.8:
            return "Excellent profitability with strong margins and return on investment"
        elif score >= 0.6:
            return "Good profitability with solid margins and return on investment"
        elif score >= 0.4:
            return "Moderate profitability with average margins and return on investment"
        elif score >= 0.2:
            return "Poor profitability with below-average margins and return on investment"
        else:
            return "Critical profitability issues with very low or negative returns"

    def _get_efficiency_assessment(self, score: float) -> str:
        """Get efficiency assessment based on score"""
        if score is None:
            return "Not enough data to assess operational efficiency"
        elif score >= 0.8:
            return "Excellent operational efficiency with optimal asset utilization"
        elif score >= 0.6:
            return "Good operational efficiency with effective asset utilization"
        elif score >= 0.4:
            return "Moderate operational efficiency with adequate asset utilization"
        elif score >= 0.2:
            return "Poor operational efficiency with suboptimal asset utilization"
        else:
            return "Critical operational efficiency issues with ineffective asset utilization"

    def _get_growth_assessment(self, score: float) -> str:
        """Get growth assessment based on score"""
        if score is None:
            return "Not enough data to assess growth"
        elif score >= 0.8:
            return "Excellent growth with strong increases in revenue, profit, and assets"
        elif score >= 0.6:
            return "Good growth with solid increases in revenue, profit, and assets"
        elif score >= 0.4:
            return "Moderate growth with average increases in revenue, profit, and assets"
        elif score >= 0.2:
            return "Low growth with minimal increases in revenue, profit, and assets"
        else:
            return "Negative growth with decreases in revenue, profit, and assets"

    def _get_overall_assessment(self, score: float) -> str:
        """Get overall financial health assessment based on score"""
        if score is None:
            return "Not enough data to assess overall financial health"
        elif score >= 0.8:
            return "Excellent financial health with strong performance across liquidity, solvency, profitability, efficiency, and growth"
        elif score >= 0.6:
            return "Good financial health with solid performance in most areas and no critical weaknesses"
        elif score >= 0.4:
            return "Moderate financial health with satisfactory performance in some areas but room for improvement in others"
        elif score >= 0.2:
            return "Weak financial health with significant concerns in multiple areas that require attention"
        else:
            return "Poor financial health with critical issues across multiple areas requiring immediate intervention"

    def analyze_financial_statements(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of all financial statements for a company

        Args:
            ticker: Company ticker symbol
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with complete financial statement analysis
        """
        # Get all financial statements
        statements = self.get_financial_statements(ticker, force_refresh)

        # Initialize results
        analysis = {
            'ticker': ticker,
            'date_analyzed': datetime.now().strftime('%Y-%m-%d'),
            'statements': {},
            'horizontal_analysis': {},
            'vertical_analysis': {},
            'structure_analysis': {},
            'metrics': {},
            'financial_health': {},
            'summary': {}
        }

        # Check if we have data to analyze
        if not statements['income_statement'].empty and not statements['balance_sheet'].empty:
            # Store the statement data
            analysis['statements'] = statements

            # Perform horizontal analysis
            analysis['horizontal_analysis']['income_statement'] = self.perform_horizontal_analysis(
                statements['income_statement'])
            analysis['horizontal_analysis']['balance_sheet'] = self.perform_horizontal_analysis(
                statements['balance_sheet'])
            if not statements['cash_flow'].empty:
                analysis['horizontal_analysis']['cash_flow'] = self.perform_horizontal_analysis(statements['cash_flow'])

            # Perform vertical analysis
            analysis['vertical_analysis']['income_statement'] = self.perform_vertical_analysis(
                statements['income_statement'], 'income')
            analysis['vertical_analysis']['balance_sheet'] = self.perform_vertical_analysis(statements['balance_sheet'],
                                                                                            'balance')
            if not statements['cash_flow'].empty:
                analysis['vertical_analysis']['cash_flow'] = self.perform_vertical_analysis(statements['cash_flow'],
                                                                                            'cash')

            # Analyze structure
            analysis['structure_analysis']['balance_sheet'] = self.analyze_balance_sheet_structure(
                statements['balance_sheet'])
            analysis['structure_analysis']['income_statement'] = self.analyze_income_statement_structure(
                statements['income_statement'])
            if not statements['cash_flow'].empty:
                analysis['structure_analysis']['cash_flow'] = self.analyze_cash_flow_statement_structure(
                    statements['cash_flow'])

            # Calculate metrics
            analysis['metrics'] = self.calculate_financial_statement_metrics(statements)

            # Analyze financial health
            analysis['financial_health'] = self.analyze_financial_health(statements)

            # Generate summary insights
            analysis['summary'] = self._generate_summary_insights(analysis)

        return analysis

    def _generate_summary_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate key insights from the financial statement analysis

        Args:
            analysis: Complete financial statement analysis dictionary

        Returns:
            Dictionary with summary insights
        """
        insights = {
            'key_strengths': [],
            'key_weaknesses': [],
            'key_trends': [],
            'key_ratios': {}
        }

        try:
            # Extract financial health assessments
            financial_health = analysis.get('financial_health', {})

            # Key strengths (any category with score >= 0.7)
            for category in ['liquidity', 'solvency', 'profitability', 'efficiency', 'growth']:
                category_data = financial_health.get(category, {})
                category_score = category_data.get('score')

                if category_score is not None and category_score >= 0.7:
                    insights['key_strengths'].append(f"Strong {category}: {category_data.get('assessment')}")

            # Key weaknesses (any category with score <= 0.4)
            for category in ['liquidity', 'solvency', 'profitability', 'efficiency', 'growth']:
                category_data = financial_health.get(category, {})
                category_score = category_data.get('score')

                if category_score is not None and category_score <= 0.4:
                    insights['key_weaknesses'].append(f"Weak {category}: {category_data.get('assessment')}")

            # Extract metrics
            metrics = analysis.get('metrics', {})

            # Extract key trends from growth metrics
            growth_metrics = metrics.get('growth', {})

            for metric, value in growth_metrics.items():
                if value is not None:
                    trend_description = f"{metric.replace('_', ' ').title()}: "
                    if value > 0.15:
                        trend_description += f"Strong growth ({value:.1%})"
                    elif value > 0.05:
                        trend_description += f"Moderate growth ({value:.1%})"
                    elif value > 0:
                        trend_description += f"Slight growth ({value:.1%})"
                    elif value > -0.05:
                        trend_description += f"Slight decline ({value:.1%})"
                    else:
                        trend_description += f"Significant decline ({value:.1%})"

                    insights['key_trends'].append(trend_description)

            # Extract key ratios
            for category, category_metrics in metrics.items():
                # Select the most important metrics from each category
                key_metrics_by_category = {
                    'profitability': ['roe', 'operating_margin', 'net_margin'],
                    'liquidity': ['current_ratio', 'quick_ratio'],
                    'solvency': ['debt_to_equity', 'interest_coverage'],
                    'efficiency': ['asset_turnover', 'inventory_turnover'],
                    'growth': ['revenue_growth', 'net_income_growth']
                }

                if category in key_metrics_by_category:
                    for metric in key_metrics_by_category[category]:
                        if metric in category_metrics:
                            insights['key_ratios'][metric] = category_metrics[metric]

            # Add overall assessment
            overall_health = financial_health.get('overall', {})
            if 'assessment' in overall_health:
                insights['overall_assessment'] = overall_health['assessment']

            return insights

        except Exception as e:
            logger.error(f"Error generating summary insights: {e}")
            return insights

    def benchmark_against_industry(self, financial_metrics: Dict[str, Dict[str, float]],
                                   sector: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare company financial metrics with industry benchmarks

        Args:
            financial_metrics: Dictionary with calculated financial metrics
            sector: Industry sector for comparison

        Returns:
            Dictionary with benchmark comparison results
        """
        from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

        benchmark_results = {
            'profitability': {},
            'liquidity': {},
            'solvency': {},
            'efficiency': {},
            'growth': {},
            'overall': {
                'above_average_metrics': [],
                'below_average_metrics': []
            }
        }

        try:
            # Initialize ratio analyzer to get industry benchmarks
            ratio_analyzer = FinancialRatioAnalyzer()

            # Get industry benchmarks
            industry_benchmarks = ratio_analyzer.get_sector_benchmarks(sector)

            # Compare each category of metrics
            for category in financial_metrics:
                if category in industry_benchmarks:
                    category_benchmarks = industry_benchmarks.get(category, {})
                    company_metrics = financial_metrics.get(category, {})

                    # Initialize category results if not exists
                    if category not in benchmark_results:
                        benchmark_results[category] = {}

                    # Compare each metric with benchmark
                    for metric, value in company_metrics.items():
                        benchmark_value = None

                        # Try to find corresponding benchmark
                        if metric in category_benchmarks:
                            benchmark_value = category_benchmarks[metric]
                        else:
                            # Try similar names
                            similar_metrics = [
                                k for k in category_benchmarks.keys()
                                if metric.lower().replace('_', ' ') in k.lower().replace('_', ' ')
                            ]
                            if similar_metrics:
                                benchmark_value = category_benchmarks[similar_metrics[0]]

                        if benchmark_value is not None:
                            # Calculate performance vs benchmark
                            performance_ratio = value / benchmark_value if benchmark_value != 0 else 1.0

                            # Determine if higher or lower is better for this metric
                            higher_is_better = True
                            if metric in ['debt_to_equity', 'debt_to_ebitda', 'days_inventory_outstanding',
                                          'days_sales_outstanding']:
                                higher_is_better = False
                                # Invert the ratio for metrics where lower is better
                                performance_ratio = 1 / performance_ratio if performance_ratio != 0 else 1.0

                            # Store benchmark comparison
                            benchmark_results[category][metric] = {
                                'company_value': value,
                                'industry_benchmark': benchmark_value,
                                'performance_ratio': performance_ratio,
                                'performance_assessment': self._assess_benchmark_performance(
                                    performance_ratio, higher_is_better
                                )
                            }

                            # Add to overall assessment lists
                            if performance_ratio >= 1.1:  # At least 10% better than benchmark
                                benchmark_results['overall']['above_average_metrics'].append(
                                    f"{metric.replace('_', ' ').title()}"
                                )
                            elif performance_ratio <= 0.9:  # At least 10% worse than benchmark
                                benchmark_results['overall']['below_average_metrics'].append(
                                    f"{metric.replace('_', ' ').title()}"
                                )

            return benchmark_results

        except Exception as e:
            logger.error(f"Error comparing with industry benchmarks: {e}")
            return benchmark_results

    def _assess_benchmark_performance(self, performance_ratio: float, higher_is_better: bool) -> str:
        """
        Assess performance compared to industry benchmark

        Args:
            performance_ratio: Ratio of company metric to industry benchmark
            higher_is_better: Whether higher values are better for this metric

        Returns:
            String assessment of performance
        """
        # Adjust logic based on whether higher is better
        if higher_is_better:
            if performance_ratio >= 1.5:
                return "Significantly outperforms industry"
            elif performance_ratio >= 1.2:
                return "Outperforms industry"
            elif performance_ratio >= 0.95:
                return "In line with industry"
            elif performance_ratio >= 0.8:
                return "Slightly underperforms industry"
            else:
                return "Significantly underperforms industry"
        else:
            # For metrics where lower is better, we already inverted the ratio,
            # so we can use the same logic
            if performance_ratio >= 1.5:
                return "Significantly outperforms industry"
            elif performance_ratio >= 1.2:
                return "Outperforms industry"
            elif performance_ratio >= 0.95:
                return "In line with industry"
            elif performance_ratio >= 0.8:
                return "Slightly underperforms industry"
            else:
                return "Significantly underperforms industry"

    def forecast_financial_statements(self, statements: Dict[str, pd.DataFrame],
                                      years: int = 5, growth_method: str = 'average') -> Dict[str, pd.DataFrame]:
        """
        Generate simple forecasts for financial statements

        Args:
            statements: Dictionary with financial statements
            years: Number of years to forecast
            growth_method: Method for calculating growth ('average', 'cagr', 'recent')

        Returns:
            Dictionary with forecasted financial statements
        """
        forecasted_statements = {}

        try:
            # Check if we have historical data
            income_stmt = statements.get('income_statement')
            balance_sheet = statements.get('balance_sheet')
            cash_flow = statements.get('cash_flow')

            if income_stmt is None or income_stmt.empty or income_stmt.shape[1] < 2:
                logger.warning("Not enough historical income statement data for forecasting")
                return forecasted_statements

            # Forecast income statement
            forecasted_statements['income_statement'] = self._forecast_statement(
                income_stmt, years, growth_method, 'income'
            )

            # Forecast balance sheet
            if balance_sheet is not None and not balance_sheet.empty and balance_sheet.shape[1] >= 2:
                forecasted_statements['balance_sheet'] = self._forecast_statement(
                    balance_sheet, years, growth_method, 'balance'
                )

            # Forecast cash flow
            if cash_flow is not None and not cash_flow.empty and cash_flow.shape[1] >= 2:
                forecasted_statements['cash_flow'] = self._forecast_statement(
                    cash_flow, years, growth_method, 'cash'
                )

            return forecasted_statements

        except Exception as e:
            logger.error(f"Error forecasting financial statements: {e}")
            return forecasted_statements

    def _forecast_statement(self, statement: pd.DataFrame, years: int,
                            growth_method: str, statement_type: str) -> pd.DataFrame:
        """
        Forecast a single financial statement

        Args:
            statement: Financial statement DataFrame
            years: Number of years to forecast
            growth_method: Method for calculating growth
            statement_type: Type of statement ('income', 'balance', 'cash')

        Returns:
            DataFrame with forecasted statement
        """
        try:
            # Make a copy of the original statement
            forecasted = statement.copy()

            # Get latest year
            latest_year = statement.columns[0]

            # Determine base year format and generate forecasted years
            try:
                if isinstance(latest_year, pd.Timestamp):
                    latest_year_date = latest_year
                else:
                    latest_year_date = pd.to_datetime(latest_year)

                # Generate future years with the same month and day
                forecast_years = [
                    latest_year_date.replace(year=latest_year_date.year + i + 1)
                    for i in range(years)
                ]
            except:
                # If date parsing fails, use simple incrementing
                if isinstance(latest_year, int):
                    forecast_years = [latest_year + i + 1 for i in range(years)]
                else:
                    # Try to extract year as integer
                    try:
                        year_int = int(str(latest_year)[:4])
                        forecast_years = [f"{year_int + i + 1}" for i in range(years)]
                    except:
                        # Fallback to generic naming
                        forecast_years = [f"Forecast Year {i + 1}" for i in range(years)]

            # Add future years as new columns
            for year in forecast_years:
                forecasted[year] = None

            # Calculate growth rates for each line item
            growth_rates = {}
            for idx in statement.index:
                # Skip non-numeric items
                if not pd.api.types.is_numeric_dtype(statement.loc[idx]):
                    continue

                # Get historical values for this item
                historical_values = statement.loc[idx].values

                # Calculate growth rate based on method
                if growth_method == 'cagr' and len(historical_values) >= 2:
                    # Compound Annual Growth Rate
                    start_value = historical_values[-1]  # Oldest value
                    end_value = historical_values[0]  # Latest value

                    # Skip if values are invalid for CAGR calculation
                    if start_value <= 0 or end_value <= 0:
                        growth_rates[idx] = 0
                        continue

                    years_count = len(historical_values) - 1
                    growth_rates[idx] = (end_value / start_value) ** (1 / years_count) - 1

                elif growth_method == 'recent' and len(historical_values) >= 2:
                    # Most recent year-over-year growth
                    if historical_values[1] != 0:
                        growth_rates[idx] = (historical_values[0] - historical_values[1]) / abs(historical_values[1])
                    else:
                        growth_rates[idx] = 0

                else:
                    # Average year-over-year growth (default)
                    yearly_growth_rates = []
                    for i in range(len(historical_values) - 1):
                        if historical_values[i + 1] != 0:
                            yearly_growth_rates.append(
                                (historical_values[i] - historical_values[i + 1]) / abs(historical_values[i + 1])
                            )

                    if yearly_growth_rates:
                        growth_rates[idx] = sum(yearly_growth_rates) / len(yearly_growth_rates)
                    else:
                        growth_rates[idx] = 0

                # Apply caps to growth rates based on item type and statement type
                growth_rates[idx] = self._apply_growth_rate_caps(idx, growth_rates[idx], statement_type)

            # Apply growth rates to forecast future years
            latest_values = statement[latest_year]

            for i, year in enumerate(forecast_years):
                for idx in statement.index:
                    if idx in growth_rates:
                        if i == 0:
                            # First forecast year based on latest actual
                            forecasted.loc[idx, year] = latest_values[idx] * (1 + growth_rates[idx])
                        else:
                            # Subsequent years based on previous forecast
                            prev_year = forecast_years[i - 1]
                            forecasted.loc[idx, year] = forecasted.loc[idx, prev_year] * (1 + growth_rates[idx])
                    else:
                        # For non-numeric items, just copy the latest value
                        forecasted.loc[idx, year] = latest_values[idx]

            # Ensure certain accounting relationships are maintained
            self._ensure_accounting_relationships(forecasted, forecast_years, statement_type)

            return forecasted

        except Exception as e:
            logger.error(f"Error forecasting statement: {e}")
            return statement

    def _apply_growth_rate_caps(self, item_name: str, growth_rate: float, statement_type: str) -> float:
        """
        Apply reasonable caps to growth rates based on item type and statement

        Args:
            item_name: Name of the financial item
            growth_rate: Calculated growth rate
            statement_type: Type of statement

        Returns:
            Capped growth rate
        """
        # Convert item name to lowercase for easier matching
        item_lower = str(item_name).lower()

        # Default caps
        max_cap = 0.30  # 30% maximum growth
        min_cap = -0.30  # -30% minimum growth (maximum decline)

        # Revenue-related items typically have moderate growth
        if 'revenue' in item_lower or 'sales' in item_lower:
            max_cap = 0.25  # 25% max growth for revenue
            min_cap = -0.15  # -15% min growth for revenue

        # Cost items often grow with revenue but can be more volatile
        elif ('cost' in item_lower or 'expense' in item_lower
              or 'cogs' in item_lower or 'expenditure' in item_lower):
            max_cap = 0.20  # 20% max growth for costs
            min_cap = -0.10  # -10% min growth for costs

        # Assets typically grow more slowly
        elif statement_type == 'balance' and ('asset' in item_lower or 'property' in item_lower):
            max_cap = 0.20  # 20% max growth for assets
            min_cap = -0.10  # -10% min growth for assets

        # Liabilities also typically grow more slowly
        elif statement_type == 'balance' and 'liabilit' in item_lower:
            max_cap = 0.15  # 15% max growth for liabilities
            min_cap = -0.15  # -15% min growth for liabilities

        # Apply caps
        return max(min(growth_rate, max_cap), min_cap)

    def _ensure_accounting_relationships(self, forecast_df: pd.DataFrame,
                                         forecast_years: List, statement_type: str) -> None:
        """
        Ensure that basic accounting relationships are maintained in forecasts

        Args:
            forecast_df: DataFrame with forecasted statement
            forecast_years: List of forecast year columns
            statement_type: Type of financial statement
        """
        try:
            # For income statement, ensure basic relationships
            if statement_type == 'income':
                for year in forecast_years:
                    # Revenue - COGS = Gross Profit
                    if ('Total Revenue' in forecast_df.index
                            and 'Cost of Revenue' in forecast_df.index
                            and 'Gross Profit' in forecast_df.index):
                        forecast_df.loc['Gross Profit', year] = (
                                forecast_df.loc['Total Revenue', year] - forecast_df.loc['Cost of Revenue', year]
                        )

                    # Gross Profit - Operating Expenses = Operating Income
                    if ('Gross Profit' in forecast_df.index
                            and 'Operating Expenses' in forecast_df.index
                            and 'Operating Income' in forecast_df.index):
                        forecast_df.loc['Operating Income', year] = (
                                forecast_df.loc['Gross Profit', year] - forecast_df.loc['Operating Expenses', year]
                        )

                    # Operating Income - Interest Expense + Interest Income - Taxes = Net Income (simplified)
                    if 'Operating Income' in forecast_df.index and 'Net Income' in forecast_df.index:
                        interest_expense = 0
                        if 'Interest Expense' in forecast_df.index:
                            interest_expense = forecast_df.loc['Interest Expense', year]

                        interest_income = 0
                        if 'Interest Income' in forecast_df.index:
                            interest_income = forecast_df.loc['Interest Income', year]

                        income_tax = 0
                        if 'Income Tax Expense' in forecast_df.index:
                            income_tax = forecast_df.loc['Income Tax Expense', year]

                        forecast_df.loc['Net Income', year] = (
                                forecast_df.loc[
                                    'Operating Income', year] - interest_expense + interest_income - income_tax
                        )

            # For balance sheet, ensure Assets = Liabilities + Equity
            elif statement_type == 'balance':
                for year in forecast_years:
                    if ('Total Assets' in forecast_df.index
                            and 'Total Liabilities' in forecast_df.index
                            and 'Total Stockholder Equity' in forecast_df.index):
                        # Calculate implied equity
                        implied_equity = (
                                forecast_df.loc['Total Assets', year] - forecast_df.loc['Total Liabilities', year]
                        )

                        # Update equity to maintain balance
                        forecast_df.loc['Total Stockholder Equity', year] = implied_equity

                    # Ensure current assets sum correctly if components are present
                    current_asset_components = [
                        'Cash and Cash Equivalents',
                        'Short Term Investments',
                        'Net Receivables',
                        'Inventory'
                    ]

                    if ('Total Current Assets' in forecast_df.index
                            and all(item in forecast_df.index for item in current_asset_components)):
                        # Calculate sum of current asset components
                        component_sum = sum(forecast_df.loc[item, year] for item in current_asset_components)

                        # Check if there's a large discrepancy
                        current_assets = forecast_df.loc['Total Current Assets', year]
                        if component_sum > 0 and abs(current_assets - component_sum) / component_sum > 0.2:
                            # Adjust total current assets
                            forecast_df.loc['Total Current Assets', year] = component_sum

            # For cash flow, ensure consistency with income and balance sheet if possible
            elif statement_type == 'cash':
                for year in forecast_years:
                    # Basic relationship: OCF + ICF + FCF = Net Change in Cash
                    if ('Operating Cash Flow' in forecast_df.index
                            and 'Investing Cash Flow' in forecast_df.index
                            and 'Financing Cash Flow' in forecast_df.index
                            and 'Net Change in Cash' in forecast_df.index):
                        forecast_df.loc['Net Change in Cash', year] = (
                                forecast_df.loc['Operating Cash Flow', year] +
                                forecast_df.loc['Investing Cash Flow', year] +
                                forecast_df.loc['Financing Cash Flow', year]
                        )

                    # Free Cash Flow = Operating Cash Flow + Capital Expenditure
                    if ('Free Cash Flow' in forecast_df.index
                            and 'Operating Cash Flow' in forecast_df.index
                            and 'Capital Expenditure' in forecast_df.index):
                        forecast_df.loc['Free Cash Flow', year] = (
                                forecast_df.loc['Operating Cash Flow', year] +
                                forecast_df.loc['Capital Expenditure', year]  # CapEx is usually negative
                        )

        except Exception as e:
            logger.error(f"Error ensuring accounting relationships: {e}")
            # Continue without adjustment if error occurs