import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('base_valuation')


class BaseValuation:
    """
    Base class for company valuation models.
    Provides foundational methods for calculating company value
    using different valuation approaches.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize base valuation class

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self.risk_free_rate = RISK_FREE_RATE
        self.dcf_parameters = DCF_PARAMETERS
        self.sector_dcf_parameters = SECTOR_DCF_PARAMETERS

    def calculate_intrinsic_value(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  sector: str = None, method: str = 'dcf') -> Dict[str, Any]:
        """
        Calculate the intrinsic value of a company using specified valuation method

        Args:
            ticker: Company ticker symbol
            financial_data: Pre-loaded financial data (if available)
            sector: Company sector for sector-specific adjustments
            method: Valuation method to use ('dcf', 'relative', 'asset-based', 'dividend')

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Select valuation method
            if method == 'dcf':
                return self.discounted_cash_flow_valuation(ticker, financial_data, sector)
            elif method == 'relative':
                return self.relative_valuation(ticker, financial_data, sector)
            elif method == 'asset-based':
                return self.asset_based_valuation(ticker, financial_data, sector)
            elif method == 'dividend':
                return self.dividend_discount_valuation(ticker, financial_data, sector)
            else:
                raise ValueError(f"Unsupported valuation method: {method}")

        except Exception as e:
            logger.error(f"Error calculating intrinsic value for {ticker}: {e}")
            return {
                'company': ticker,
                'value_per_share': None,
                'total_value': None,
                'error': str(e),
                'method': method
            }

    def discounted_cash_flow_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                       sector: str = None) -> Dict[str, Any]:
        """
        Calculate company value using Discounted Cash Flow (DCF) method

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with DCF valuation results
        """
        # This is a base implementation - specific DCF models will be implemented
        # in dcf_models.py with more sophisticated forecasting and adjustments

        try:
            # Get parameters based on sector if available
            params = self._get_dcf_parameters(sector)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for DCF valuation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Simple linear growth forecast for demonstration
            # Real implementation would use more sophisticated forecasting methods
            forecast_years = params['forecast_years']
            growth_rate = self._estimate_growth_rate(historical_fcf)
            terminal_growth = params['terminal_growth_rate']
            discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Forecast future free cash flows
            future_fcf = self._forecast_fcf(historical_fcf.iloc[0], forecast_years, growth_rate)

            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(future_fcf[-1], terminal_growth, discount_rate)

            # Discount all future cash flows to present value
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(future_fcf))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Calculate equity value
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding is not None and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                # If shares outstanding not available, estimate from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    value_per_share = equity_value / estimated_shares
                else:
                    value_per_share = None

            # Apply margin of safety
            if value_per_share is not None:
                safety_margin = params['default_margin_of_safety']
                conservative_value = value_per_share * (1 - safety_margin)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'dcf',
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': discount_rate,
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'forecast_fcf': future_fcf,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'historical_fcf': historical_fcf.to_dict(),
                'net_debt': net_debt,
                'safety_margin': params['default_margin_of_safety']
            }

        except Exception as e:
            logger.error(f"Error in DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'dcf',
                'enterprise_value': None,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def relative_valuation(self, ticker: str, financial_data: Dict[str, Any],
                           sector: str = None) -> Dict[str, Any]:
        """
        Calculate company value using relative valuation (multiples)

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with relative valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            market_cap = market_data.get('market_cap')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for relative valuation")

            # Get most recent data
            income = income_stmt.iloc[:, 0]
            balance = balance_sheet.iloc[:, 0]

            # Get sector-appropriate multiples
            multiples = self._get_sector_multiples(sector)

            # Calculate key metrics for valuation
            metrics = {}
            if 'Net Income' in income.index:
                metrics['earnings'] = income.loc['Net Income']
            if 'EBITDA' in income.index:
                metrics['ebitda'] = income.loc['EBITDA']
            elif 'Operating Income' in income.index and 'Depreciation & Amortization' in income.index:
                metrics['ebitda'] = income.loc['Operating Income'] + income.loc['Depreciation & Amortization']
            if 'Total Revenue' in income.index:
                metrics['revenue'] = income.loc['Total Revenue']
            if 'Total Stockholder Equity' in balance.index:
                metrics['book_value'] = balance.loc['Total Stockholder Equity']
            if 'Total Assets' in balance.index and 'Total Liabilities' in balance.index:
                metrics['enterprise_value'] = market_cap + balance.loc['Total Liabilities'] - balance.loc[
                    'Cash and Cash Equivalents'] if 'Cash and Cash Equivalents' in balance.index else market_cap + \
                                                                                                      balance.loc[
                                                                                                          'Total Liabilities']

            # Calculate estimated values based on different multiples
            valuations = {}

            if 'earnings' in metrics and metrics['earnings'] > 0:
                valuations['pe'] = {
                    'multiple': multiples.get('pe', 15),
                    'value': metrics['earnings'] * multiples.get('pe', 15),
                    'description': 'Price to Earnings'
                }

            if 'ebitda' in metrics and metrics['ebitda'] > 0:
                valuations['ev_ebitda'] = {
                    'multiple': multiples.get('ev_ebitda', 10),
                    'value': metrics['ebitda'] * multiples.get('ev_ebitda', 10),
                    'description': 'Enterprise Value to EBITDA'
                }

            if 'revenue' in metrics and metrics['revenue'] > 0:
                valuations['ps'] = {
                    'multiple': multiples.get('ps', 2),
                    'value': metrics['revenue'] * multiples.get('ps', 2),
                    'description': 'Price to Sales'
                }

            if 'book_value' in metrics and metrics['book_value'] > 0:
                valuations['pb'] = {
                    'multiple': multiples.get('pb', 2),
                    'value': metrics['book_value'] * multiples.get('pb', 2),
                    'description': 'Price to Book'
                }

            # Calculate average valuation
            if valuations:
                total_value = sum(v['value'] for v in valuations.values())
                average_value = total_value / len(valuations)

                # Calculate per share value
                shares_outstanding = market_data.get('shares_outstanding')
                if shares_outstanding and shares_outstanding > 0:
                    value_per_share = average_value / shares_outstanding
                else:
                    value_per_share = None
            else:
                average_value = None
                value_per_share = None

            return {
                'company': ticker,
                'method': 'relative',
                'valuations': valuations,
                'average_value': average_value,
                'value_per_share': value_per_share,
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Error in relative valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'relative',
                'average_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def asset_based_valuation(self, ticker: str, financial_data: Dict[str, Any],
                              sector: str = None) -> Dict[str, Any]:
        """
        Calculate company value using asset-based approach

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with asset-based valuation results
        """
        try:
            # Extract balance sheet
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})

            if balance_sheet is None:
                raise ValueError("Missing balance sheet for asset-based valuation")

            # Get most recent data
            balance = balance_sheet.iloc[:, 0]

            # Calculate book value (net asset value)
            if 'Total Assets' in balance.index and 'Total Liabilities' in balance.index:
                book_value = balance.loc['Total Assets'] - balance.loc['Total Liabilities']
            elif 'Total Stockholder Equity' in balance.index:
                book_value = balance.loc['Total Stockholder Equity']
            else:
                raise ValueError("Unable to calculate book value from balance sheet")

            # Adjust book value based on sector-specific factors
            adjusted_book_value = self._adjust_book_value(book_value, balance, sector)

            # Calculate liquidation value (conservative estimate)
            liquidation_value = self._calculate_liquidation_value(balance)

            # Calculate replacement value (cost to rebuild business)
            replacement_value = self._estimate_replacement_value(balance, sector)

            # Calculate per share values
            shares_outstanding = market_data.get('shares_outstanding')
            if shares_outstanding and shares_outstanding > 0:
                book_value_per_share = book_value / shares_outstanding
                adjusted_book_value_per_share = adjusted_book_value / shares_outstanding
                liquidation_value_per_share = liquidation_value / shares_outstanding
                replacement_value_per_share = replacement_value / shares_outstanding
            else:
                book_value_per_share = None
                adjusted_book_value_per_share = None
                liquidation_value_per_share = None
                replacement_value_per_share = None

            return {
                'company': ticker,
                'method': 'asset-based',
                'book_value': book_value,
                'adjusted_book_value': adjusted_book_value,
                'liquidation_value': liquidation_value,
                'replacement_value': replacement_value,
                'book_value_per_share': book_value_per_share,
                'adjusted_book_value_per_share': adjusted_book_value_per_share,
                'liquidation_value_per_share': liquidation_value_per_share,
                'replacement_value_per_share': replacement_value_per_share
            }

        except Exception as e:
            logger.error(f"Error in asset-based valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'asset-based',
                'book_value': None,
                'adjusted_book_value': None,
                'book_value_per_share': None,
                'error': str(e)
            }

    def dividend_discount_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                    sector: str = None) -> Dict[str, Any]:
        """
        Calculate company value using dividend discount model

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with dividend discount valuation results
        """
        try:
            # Extract income statement
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            dividend_yield = market_data.get('dividend_yield')
            share_price = market_data.get('share_price')

            if income_stmt is None:
                raise ValueError("Missing income statement for dividend discount valuation")

            # Get most recent financial data
            income = income_stmt.iloc[:, 0]

            # Determine current dividend
            current_dividend = None

            # Try to get dividend from cash flow statement
            if cash_flow is not None and not cash_flow.empty:
                cf = cash_flow.iloc[:, 0]
                if 'Dividends Paid' in cf.index:
                    shares_outstanding = market_data.get('shares_outstanding')
                    if shares_outstanding and shares_outstanding > 0:
                        current_dividend = abs(cf.loc['Dividends Paid']) / shares_outstanding

            # If not available, calculate from yield and price
            if current_dividend is None and dividend_yield and share_price:
                current_dividend = share_price * dividend_yield

            # If still not available, check if we can derive from income statement
            if current_dividend is None and 'Dividends per Share' in income.index:
                current_dividend = income.loc['Dividends per Share']

            if current_dividend is None or current_dividend <= 0:
                raise ValueError("Company does not pay dividends or dividend data is unavailable")

            # Get parameters based on sector
            params = self._get_dcf_parameters(sector)

            # Estimate dividend growth rate
            dividend_growth = self._estimate_dividend_growth(financial_data)

            # Calculate discount rate (required rate of return)
            discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Apply Gordon Growth Model if growth < discount rate
            if dividend_growth < discount_rate:
                # Gordon Growth Model (for stable dividend growth)
                value_per_share = current_dividend * (1 + dividend_growth) / (discount_rate - dividend_growth)
            else:
                # Multi-stage dividend discount model (simplified)
                high_growth_years = 5
                high_growth_dividends = []

                for year in range(1, high_growth_years + 1):
                    dividend = current_dividend * (1 + dividend_growth) ** year
                    high_growth_dividends.append(dividend)

                # Terminal value with more sustainable growth rate
                terminal_growth = min(dividend_growth * 0.5, params['terminal_growth_rate'])
                terminal_dividend = high_growth_dividends[-1] * (1 + terminal_growth)
                terminal_value = terminal_dividend / (discount_rate - terminal_growth)

                # Calculate present value
                present_value_dividends = sum(div / (1 + discount_rate) ** (i + 1)
                                              for i, div in enumerate(high_growth_dividends))
                present_value_terminal = terminal_value / (1 + discount_rate) ** high_growth_years

                value_per_share = present_value_dividends + present_value_terminal

            # Total value
            shares_outstanding = market_data.get('shares_outstanding')
            if shares_outstanding and shares_outstanding > 0:
                total_value = value_per_share * shares_outstanding
            else:
                total_value = None

            return {
                'company': ticker,
                'method': 'dividend-discount',
                'value_per_share': value_per_share,
                'total_value': total_value,
                'current_dividend': current_dividend,
                'dividend_growth': dividend_growth,
                'discount_rate': discount_rate
            }

        except Exception as e:
            logger.error(f"Error in dividend discount valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'dividend-discount',
                'value_per_share': None,
                'total_value': None,
                'error': str(e)
            }

    # Helper methods for calculations

    def _load_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Load all required financial data for valuation"""
        try:
            # Get financial statements
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income', 'annual')
            balance_sheet = self.data_loader.get_financial_statements(ticker, 'balance', 'annual')
            cash_flow = self.data_loader.get_financial_statements(ticker, 'cash', 'annual')

            # Get company info and market data
            company_info = self.data_loader.get_company_info(ticker)

            # Extract market data from company info
            market_data = {
                'market_cap': company_info.get('market_cap'),
                'share_price': company_info.get('current_price', company_info.get('regular_market_price')),
                'shares_outstanding': company_info.get('shares_outstanding'),
                'dividend_yield': company_info.get('dividend_yield'),
                'beta': company_info.get('beta')
            }

            return {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'company_info': company_info,
                'market_data': market_data
            }

        except Exception as e:
            logger.error(f"Error loading financial data for {ticker}: {e}")
            raise ValueError(f"Failed to load required financial data for {ticker}: {e}")

    def _get_dcf_parameters(self, sector: str = None) -> Dict[str, Any]:
        """Get DCF parameters with sector-specific adjustments if applicable"""
        # Use base parameters as default
        params = self.dcf_parameters.copy()

        # Apply sector-specific parameters if available
        if sector and sector in self.sector_dcf_parameters:
            for key, value in self.sector_dcf_parameters[sector].items():
                params[key] = value

        return params

    def _calculate_historical_fcf(self, income_stmt: pd.DataFrame, cash_flow: pd.DataFrame) -> pd.Series:
        """Calculate historical free cash flow from financial statements"""
        try:
            if income_stmt.empty or cash_flow.empty:
                return pd.Series()

            # Get the periods that are available in both statements
            common_periods = income_stmt.columns.intersection(cash_flow.columns)

            if len(common_periods) == 0:
                raise ValueError("No common periods in income statement and cash flow")

            # Create a series for historical free cash flow
            historical_fcf = pd.Series(index=common_periods)

            for period in common_periods:
                # Method 1: Direct from cash flow statement if available
                if 'Free Cash Flow' in cash_flow.index:
                    historical_fcf[period] = cash_flow.loc['Free Cash Flow', period]

                # Method 2: Operating Cash Flow - Capital Expenditures
                elif 'Operating Cash Flow' in cash_flow.index and any(capex in cash_flow.index for capex in
                                                                      ['Capital Expenditure', 'Capital Expenditures',
                                                                       'Purchase of Property Plant and Equipment']):
                    operating_cf = cash_flow.loc['Operating Cash Flow', period]

                    # Find capital expenditure entry (different statements use different names)
                    for capex_name in ['Capital Expenditure', 'Capital Expenditures',
                                       'Purchase of Property Plant and Equipment']:
                        if capex_name in cash_flow.index:
                            capex = cash_flow.loc[capex_name, period]
                            historical_fcf[period] = operating_cf - abs(capex)
                            break

                # Method 3: Net Income + D&A - Changes in WC - CAPEX (simplified)
                else:
                    if 'Net Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                        net_income = income_stmt.loc['Net Income', period]
                        depreciation = income_stmt.loc['Depreciation & Amortization', period]

                        # Estimate capex (might not be accurate)
                        capex_estimate = depreciation * 1.5  # Simple approximation

                        historical_fcf[period] = net_income + depreciation - capex_estimate

            return historical_fcf

        except Exception as e:
            logger.error(f"Error calculating historical FCF: {e}")
            return pd.Series()

    def _estimate_growth_rate(self, historical_fcf: pd.Series) -> float:
        """Estimate future growth rate based on historical cash flow"""
        try:
            if len(historical_fcf) < 2:
                # Not enough data, use conservative estimate
                return 0.03  # 3% default growth

            # Calculate year-over-year growth rates
            growth_rates = []
            periods = list(historical_fcf.index)

            for i in range(len(periods) - 1):
                if historical_fcf[periods[i + 1]] > 0 and historical_fcf[periods[i]] > 0:
                    annual_growth = (historical_fcf[periods[i]] / historical_fcf[periods[i + 1]]) - 1
                    growth_rates.append(annual_growth)

            if not growth_rates:
                return 0.03  # Default growth if can't calculate

            # Use weighted average, giving more weight to recent years
            weights = list(range(1, len(growth_rates) + 1))
            weighted_growth = sum(r * w for r, w in zip(growth_rates, weights)) / sum(weights)

            # Cap growth rate to reasonable range
            growth_rate = max(0.01, min(0.20, weighted_growth))

            return growth_rate

        except Exception as e:
            logger.error(f"Error estimating growth rate: {e}")
            return 0.03  # Default conservative growth rate

    def _forecast_fcf(self, last_fcf: float, forecast_years: int, growth_rate: float) -> List[float]:
        """Forecast future free cash flows based on growth rate"""
        future_fcf = []

        for year in range(1, forecast_years + 1):
            fcf = last_fcf * (1 + growth_rate) ** year
            future_fcf.append(fcf)

        return future_fcf

    def _calculate_terminal_value(self, final_fcf: float, terminal_growth: float, discount_rate: float) -> float:
        """Calculate terminal value using perpetuity growth model"""
        # Gordon Growth Model: TV = FCF * (1 + g) / (r - g)
        if discount_rate <= terminal_growth:
            # Avoid division by zero or negative values
            # Ensure discount rate is at least 3% higher than growth
            discount_rate = terminal_growth + 0.03

        terminal_value = final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        return terminal_value

    def _calculate_net_debt(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate net debt (total debt - cash)"""
        try:
            if balance_sheet.empty:
                return 0

            # Get the most recent balance sheet
            balance = balance_sheet.iloc[:, 0]

            # Calculate total debt
            total_debt = 0
            for debt_item in ['Total Debt', 'Long Term Debt', 'Short Term Debt', 'Current Portion of Long Term Debt']:
                if debt_item in balance.index:
                    total_debt += balance.loc[debt_item]

            # Calculate cash and equivalents
            cash = 0
            for cash_item in ['Cash and Cash Equivalents', 'Short Term Investments', 'Cash and Short Term Investments']:
                if cash_item in balance.index:
                    cash += balance.loc[cash_item]

            # Calculate net debt
            net_debt = total_debt - cash

            return net_debt

        except Exception as e:
            logger.error(f"Error calculating net debt: {e}")
            return 0

    def _calculate_discount_rate(self, ticker: str, financial_data: Dict[str, Any], sector: str = None) -> Optional[
        float]:
        """
        Calculate discount rate (WACC) based on financial data

        This is a simplified WACC calculation. For a more accurate model,
        we would need details about debt costs, tax rates, and capital structure.
        """
        try:
            # Extract data needed for WACC calculation
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            beta = market_data.get('beta')

            if balance_sheet is None or beta is None:
                # Fall back to sector-based discount rate
                if sector and sector in self.sector_dcf_parameters:
                    return self.sector_dcf_parameters[sector].get('default_discount_rate')
                else:
                    return self.dcf_parameters.get('default_discount_rate')

            # Get the most recent balance sheet
            balance = balance_sheet.iloc[:, 0]

            # Calculate cost of equity using CAPM
            # CAPM: Cost of Equity = Risk-Free Rate + Beta * Market Risk Premium
            market_risk_premium = 0.05  # Standard assumption: ~5%
            cost_of_equity = self.risk_free_rate + beta * market_risk_premium

            # Calculate cost of debt (simplified)
            cost_of_debt = 0.04  # Assume ~4% debt cost
            tax_rate = 0.25  # Assume ~25% tax rate
            after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

            # Calculate weights for equity and debt
            market_cap = market_data.get('market_cap')
            total_debt = 0

            for debt_item in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_item in balance.index:
                    total_debt += balance.loc[debt_item]

            if market_cap and market_cap > 0 and total_debt >= 0:
                total_value = market_cap + total_debt
                weight_equity = market_cap / total_value
                weight_debt = total_debt / total_value
            else:
                # Default to 70% equity, 30% debt if data not available
                weight_equity = 0.7
                weight_debt = 0.3

            # Calculate WACC
            wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

            # Ensure WACC is in a reasonable range
            wacc = max(0.05, min(0.20, wacc))

            return wacc

        except Exception as e:
            logger.error(f"Error calculating discount rate: {e}")
            # Fall back to sector-based discount rate
            if sector and sector in self.sector_dcf_parameters:
                return self.sector_dcf_parameters[sector].get('default_discount_rate')
            else:
                return self.dcf_parameters.get('default_discount_rate')

    def _get_sector_multiples(self, sector: str = None) -> Dict[str, float]:
        """
        Get appropriate valuation multiples for a given sector

        Args:
            sector: Industry sector

        Returns:
            Dictionary of valuation multiples
        """
        # Default multiples (for general use)
        default_multiples = {
            'pe': 15.0,  # P/E ratio
            'forward_pe': 12.0,  # Forward P/E ratio
            'peg': 1.0,  # Price/Earnings to Growth
            'ps': 2.0,  # Price to Sales
            'pb': 2.0,  # Price to Book
            'ev_ebitda': 10.0,  # Enterprise Value to EBITDA
            'ev_revenue': 3.0,  # Enterprise Value to Revenue
            'ev_fcf': 15.0  # Enterprise Value to Free Cash Flow
        }

        # Sector-specific multiples based on typical values
        sector_multiples = {
            'Technology': {
                'pe': 25.0, 'forward_pe': 20.0, 'ps': 5.0, 'pb': 5.0, 'ev_ebitda': 15.0
            },
            'Healthcare': {
                'pe': 20.0, 'forward_pe': 16.0, 'ps': 4.0, 'pb': 4.0, 'ev_ebitda': 12.0
            },
            'Financials': {
                'pe': 12.0, 'forward_pe': 10.0, 'ps': 3.0, 'pb': 1.5, 'ev_ebitda': 8.0
            },
            'Energy': {
                'pe': 12.0, 'forward_pe': 10.0, 'ps': 1.5, 'pb': 1.5, 'ev_ebitda': 6.0
            },
            'Consumer Discretionary': {
                'pe': 18.0, 'forward_pe': 15.0, 'ps': 1.5, 'pb': 3.0, 'ev_ebitda': 9.0
            },
            'Consumer Staples': {
                'pe': 20.0, 'forward_pe': 18.0, 'ps': 2.0, 'pb': 4.0, 'ev_ebitda': 12.0
            },
            'Industrials': {
                'pe': 18.0, 'forward_pe': 15.0, 'ps': 1.5, 'pb': 3.0, 'ev_ebitda': 9.0
            },
            'Materials': {
                'pe': 15.0, 'forward_pe': 12.0, 'ps': 1.5, 'pb': 2.0, 'ev_ebitda': 8.0
            },
            'Real Estate': {
                'pe': 18.0, 'forward_pe': 15.0, 'ps': 3.0, 'pb': 2.0, 'ev_ebitda': 15.0
            },
            'Communication Services': {
                'pe': 20.0, 'forward_pe': 16.0, 'ps': 3.0, 'pb': 3.0, 'ev_ebitda': 10.0
            },
            'Utilities': {
                'pe': 18.0, 'forward_pe': 16.0, 'ps': 2.0, 'pb': 1.5, 'ev_ebitda': 10.0
            }
        }

        # Return sector-specific multiples if available, else default
        if sector and sector in sector_multiples:
            return sector_multiples[sector]
        else:
            # Try to match sector partially (e.g., "Tech" should match "Technology")
            if sector:
                for sector_key in sector_multiples:
                    if sector.lower() in sector_key.lower() or sector_key.lower() in sector.lower():
                        return sector_multiples[sector_key]

            return default_multiples

    def _adjust_book_value(self, book_value: float, balance_sheet: pd.Series, sector: str = None) -> float:
        """
        Adjust book value based on sector-specific factors

        Args:
            book_value: Raw book value
            balance_sheet: Balance sheet data
            sector: Industry sector

        Returns:
            Adjusted book value
        """
        try:
            # Start with raw book value
            adjusted_value = book_value

            # Adjustments for technology and R&D-heavy sectors
            if sector in ['Technology', 'Healthcare', 'Communication Services']:
                # Increase value to account for undervalued intellectual property
                # and R&D (which is typically expensed rather than capitalized)
                if 'Intangible Assets' in balance_sheet.index:
                    intangibles = balance_sheet.loc['Intangible Assets']
                    adjusted_value += intangibles * 0.5  # Add back 50% of intangibles

                # If R&D or similar costs can be found, adjust for them
                rd_indicators = ['Research and Development', 'R&D Expense', 'Research Expense']
                for rd_item in rd_indicators:
                    if rd_item in balance_sheet.index:
                        rd_expense = balance_sheet.loc[rd_item]
                        adjusted_value += rd_expense * 3  # Capitalize 3 years of R&D

            # Adjustments for real estate and asset-heavy sectors
            elif sector in ['Real Estate', 'Utilities', 'Energy', 'Materials', 'Industrials']:
                # Adjust for potential undervaluation of property and equipment
                if 'Property Plant and Equipment' in balance_sheet.index:
                    ppe = balance_sheet.loc['Property Plant and Equipment']
                    if sector == 'Real Estate':
                        adjusted_value += ppe * 0.2  # Add 20% to property values
                    else:
                        adjusted_value += ppe * 0.1  # Add 10% to property values

            # Adjustments for financial sector
            elif sector == 'Financials':
                # Financial firms may have undervalued loan portfolios or investments
                if 'Total Investments' in balance_sheet.index:
                    investments = balance_sheet.loc['Total Investments']
                    adjusted_value += investments * 0.05  # Add 5% to investment values

            # General adjustment for all sectors
            # Goodwill is often overvalued after acquisitions
            if 'Goodwill' in balance_sheet.index:
                goodwill = balance_sheet.loc['Goodwill']
                adjusted_value -= goodwill * 0.2  # Subtract 20% of goodwill

            return max(adjusted_value, book_value * 0.8)  # Don't reduce by more than 20%

        except Exception as e:
            logger.error(f"Error adjusting book value: {e}")
            return book_value  # Return original value if adjustment fails

    def _calculate_liquidation_value(self, balance_sheet: pd.Series) -> float:
        """
        Calculate liquidation value by applying discount to assets

        Args:
            balance_sheet: Balance sheet data

        Returns:
            Liquidation value
        """
        try:
            liquidation_value = 0

            # Apply different recovery rates to different asset classes

            # Cash and equivalents (100% recovery)
            for cash_item in ['Cash and Cash Equivalents', 'Short Term Investments', 'Cash and Short Term Investments']:
                if cash_item in balance_sheet.index:
                    liquidation_value += balance_sheet.loc[cash_item]

            # Accounts receivable (80% recovery)
            for receivable_item in ['Accounts Receivable', 'Net Receivables']:
                if receivable_item in balance_sheet.index:
                    liquidation_value += balance_sheet.loc[receivable_item] * 0.8

            # Inventory (50% recovery)
            if 'Inventory' in balance_sheet.index:
                liquidation_value += balance_sheet.loc['Inventory'] * 0.5

            # Property, plant and equipment (40% recovery)
            for ppe_item in ['Property Plant and Equipment', 'Gross Property Plant and Equipment']:
                if ppe_item in balance_sheet.index:
                    liquidation_value += balance_sheet.loc[ppe_item] * 0.4

            # Intangibles (10% recovery, often worthless in liquidation)
            for intangible_item in ['Intangible Assets', 'Goodwill']:
                if intangible_item in balance_sheet.index:
                    liquidation_value += balance_sheet.loc[intangible_item] * 0.1

            # Other assets (30% recovery)
            if 'Other Assets' in balance_sheet.index:
                liquidation_value += balance_sheet.loc['Other Assets'] * 0.3

            # Subtract liabilities
            for liability_item in ['Total Liabilities', 'Total Current Liabilities', 'Long Term Debt']:
                if liability_item in balance_sheet.index:
                    liquidation_value -= balance_sheet.loc[liability_item]

            # Ensure value isn't negative
            return max(0, liquidation_value)

        except Exception as e:
            logger.error(f"Error calculating liquidation value: {e}")
            # Fallback: assume liquidation value is 50% of book value
            if 'Total Stockholder Equity' in balance_sheet.index:
                return balance_sheet.loc['Total Stockholder Equity'] * 0.5
            elif 'Total Assets' in balance_sheet.index and 'Total Liabilities' in balance_sheet.index:
                return (balance_sheet.loc['Total Assets'] - balance_sheet.loc['Total Liabilities']) * 0.5
            else:
                return 0

    def _estimate_replacement_value(self, balance_sheet: pd.Series, sector: str = None) -> float:
        """
        Estimate replacement value (cost to rebuild business)

        Args:
            balance_sheet: Balance sheet data
            sector: Industry sector

        Returns:
            Replacement value estimate
        """
        try:
            replacement_value = 0

            # Physical assets (replacement cost higher than book value due to inflation)
            if 'Property Plant and Equipment' in balance_sheet.index:
                ppe = balance_sheet.loc['Property Plant and Equipment']
                # Different sectors have different PPE intensity and replacement costs
                if sector in ['Energy', 'Materials', 'Utilities', 'Industrials', 'Real Estate']:
                    replacement_value += ppe * 1.3  # 30% premium for asset-heavy sectors
                else:
                    replacement_value += ppe * 1.2  # 20% premium for other sectors

            # Inventory (replacement at market value)
            if 'Inventory' in balance_sheet.index:
                replacement_value += balance_sheet.loc['Inventory'] * 1.1

            # Intangibles and R&D (cost to develop intellectual property)
            if 'Intangible Assets' in balance_sheet.index:
                intangibles = balance_sheet.loc['Intangible Assets']
                if sector in ['Technology', 'Healthcare', 'Communication Services']:
                    replacement_value += intangibles * 2.0  # High cost to redevelop IP
                else:
                    replacement_value += intangibles * 1.5

            # Brand value and other intangibles (cost to build brand recognition)
            if 'Goodwill' in balance_sheet.index:
                goodwill = balance_sheet.loc['Goodwill']
                if sector in ['Consumer Discretionary', 'Consumer Staples']:
                    replacement_value += goodwill * 0.8  # Brand-heavy sectors
                else:
                    replacement_value += goodwill * 0.5

            # Working capital requirements
            current_assets = 0
            current_liabilities = 0

            if 'Total Current Assets' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets']

            if 'Total Current Liabilities' in balance_sheet.index:
                current_liabilities = balance_sheet.loc['Total Current Liabilities']

            working_capital = current_assets - current_liabilities
            replacement_value += working_capital

            # Add additional costs for customer acquisition, employee hiring, etc.
            # This is a rough estimate based on sector
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets']

                if sector in ['Technology', 'Healthcare']:
                    replacement_value += total_assets * 0.15  # 15% premium
                elif sector in ['Consumer Discretionary', 'Consumer Staples']:
                    replacement_value += total_assets * 0.12  # 12% premium
                else:
                    replacement_value += total_assets * 0.10  # 10% premium

            return replacement_value

        except Exception as e:
            logger.error(f"Error estimating replacement value: {e}")
            # Fallback: estimate replacement value as 120% of book value
            if 'Total Stockholder Equity' in balance_sheet.index:
                return balance_sheet.loc['Total Stockholder Equity'] * 1.2
            elif 'Total Assets' in balance_sheet.index and 'Total Liabilities' in balance_sheet.index:
                return (balance_sheet.loc['Total Assets'] - balance_sheet.loc['Total Liabilities']) * 1.2
            else:
                return 0

    def _estimate_dividend_growth(self, financial_data: Dict[str, Any]) -> float:
        """
        Estimate dividend growth rate based on historical data and financial metrics

        Args:
            financial_data: Dictionary with financial statements and market data

        Returns:
            Estimated dividend growth rate
        """
        try:
            # Extract required data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None or income_stmt.empty or balance_sheet.empty:
                return 0.03  # Default to 3% if data not available

            # Get historical dividend data if available
            historical_dividends = None
            try:
                # Try to get historical dividends from data_loader
                ticker = financial_data.get('company_info', {}).get('symbol')
                if ticker:
                    historical_dividends = self.data_loader.get_historical_dividends(ticker)
            except:
                pass

            # Method 1: Calculate from historical dividend growth if available
            if historical_dividends is not None and not historical_dividends.empty and len(historical_dividends) >= 2:
                # Resample to get annual dividends
                annual_dividends = historical_dividends.resample('Y').sum()

                if len(annual_dividends) >= 2:
                    # Calculate growth rates
                    growth_rates = []
                    annual_dividends_values = annual_dividends.values

                    for i in range(1, len(annual_dividends_values)):
                        if annual_dividends_values[i - 1] > 0:
                            growth_rate = (annual_dividends_values[i] / annual_dividends_values[i - 1]) - 1
                            growth_rates.append(growth_rate)

                    if growth_rates:
                        # Weight recent years more heavily
                        weights = list(range(1, len(growth_rates) + 1))
                        weighted_growth = sum(g * w for g, w in zip(growth_rates, weights)) / sum(weights)
                        return max(0.01, min(0.15, weighted_growth))  # Cap between 1% and 15%

            # Method 2: Estimate from fundamentals (sustainable growth rate)
            # Sustainable Growth Rate = Retention Ratio * Return on Equity

            # Get the most recent data
            income = income_stmt.iloc[:, 0]
            balance = balance_sheet.iloc[:, 0]

            # Calculate retention ratio (1 - Dividend Payout Ratio)
            retention_ratio = 0.7  # Default assumption: 70% retention

            if 'Net Income' in income.index:
                net_income = income.loc['Net Income']

                # Try to find dividend info
                dividend_payout = None

                for div_item in ['Dividends Paid', 'Common Stock Dividends Paid']:
                    if div_item in income.index:
                        dividend_payout = abs(income.loc[div_item])
                        break

                if dividend_payout is not None and net_income > 0:
                    payout_ratio = dividend_payout / net_income
                    retention_ratio = 1 - payout_ratio

            # Calculate ROE
            roe = 0.10  # Default assumption: 10% ROE

            if 'Net Income' in income.index and 'Total Stockholder Equity' in balance.index:
                net_income = income.loc['Net Income']
                equity = balance.loc['Total Stockholder Equity']

                if equity > 0:
                    roe = net_income / equity

            # Calculate sustainable growth rate
            sustainable_growth = retention_ratio * roe

            # Ensure growth is within reasonable bounds
            growth_rate = max(0.01, min(0.15, sustainable_growth))

            return growth_rate

        except Exception as e:
            logger.error(f"Error estimating dividend growth: {e}")
            return 0.03  # Default to 3% growth

    def _calculate_wacc_components(self, financial_data: Dict[str, Any], sector: str = None) -> Dict[str, float]:
        """
        Calculate detailed components of Weighted Average Cost of Capital

        Args:
            financial_data: Dictionary with financial statements and market data
            sector: Industry sector for sector-specific adjustments

        Returns:
            Dictionary with WACC components
        """
        try:
            # Extract required data
            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')
            market_data = financial_data.get('market_data', {})

            beta = market_data.get('beta', 1.0)  # Default to market beta (1.0) if not available

            # Get the most recent data
            balance = balance_sheet.iloc[:, 0] if balance_sheet is not None and not balance_sheet.empty else None
            income = income_stmt.iloc[:, 0] if income_stmt is not None and not income_stmt.empty else None

            # Calculate Cost of Equity using CAPM
            # CAPM: Cost of Equity = Risk-Free Rate + Beta * Market Risk Premium
            market_risk_premium = 0.05  # Standard assumption: ~5%
            cost_of_equity = self.risk_free_rate + beta * market_risk_premium

            # Calculate Cost of Debt
            # Try to extract interest expense and total debt from financial statements
            interest_expense = None
            total_debt = 0

            if income is not None and 'Interest Expense' in income.index:
                interest_expense = abs(income.loc['Interest Expense'])

            if balance is not None:
                for debt_item in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                    if debt_item in balance.index:
                        total_debt += balance.loc[debt_item]

            # Calculate effective interest rate if possible
            if interest_expense is not None and total_debt > 0:
                cost_of_debt = interest_expense / total_debt
                # Ensure cost of debt is reasonable
                cost_of_debt = max(0.02, min(0.10, cost_of_debt))
            else:
                # Estimate based on sector and risk-free rate
                if sector in ['Utilities', 'Consumer Staples', 'Real Estate']:
                    # Lower risk sectors
                    cost_of_debt = self.risk_free_rate + 0.02
                elif sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
                    # Higher risk sectors
                    cost_of_debt = self.risk_free_rate + 0.04
                else:
                    # Medium risk sectors
                    cost_of_debt = self.risk_free_rate + 0.03

            # Estimate tax rate
            tax_rate = 0.25  # Default corporate tax rate

            if income is not None and 'Income Tax Expense' in income.index and 'Income Before Tax' in income.index:
                pretax_income = income.loc['Income Before Tax']
                tax_expense = income.loc['Income Tax Expense']

                if pretax_income > 0:
                    effective_tax_rate = tax_expense / pretax_income
                    # Ensure tax rate is reasonable
                    tax_rate = max(0.10, min(0.40, effective_tax_rate))

            # After-tax cost of debt
            after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

            # Calculate capital structure weights
            market_cap = market_data.get('market_cap')

            if market_cap and market_cap > 0 and total_debt >= 0:
                total_value = market_cap + total_debt
                weight_equity = market_cap / total_value
                weight_debt = total_debt / total_value
            else:
                # Use sector averages if market data not available
                if sector in ['Financials', 'Utilities', 'Real Estate']:
                    # Higher debt sectors
                    weight_equity = 0.6
                    weight_debt = 0.4
                elif sector in ['Technology', 'Healthcare']:
                    # Lower debt sectors
                    weight_equity = 0.8
                    weight_debt = 0.2
                else:
                    # Medium debt sectors
                    weight_equity = 0.7
                    weight_debt = 0.3

            # Calculate WACC
            wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

            # Ensure WACC is reasonable
            wacc = max(0.05, min(0.20, wacc))

            return {
                'wacc': wacc,
                'cost_of_equity': cost_of_equity,
                'cost_of_debt': cost_of_debt,
                'after_tax_cost_of_debt': after_tax_cost_of_debt,
                'weight_equity': weight_equity,
                'weight_debt': weight_debt,
                'tax_rate': tax_rate,
                'beta': beta,
                'risk_free_rate': self.risk_free_rate,
                'market_risk_premium': market_risk_premium
            }

        except Exception as e:
            logger.error(f"Error calculating WACC components: {e}")
            # Return default values
            return {
                'wacc': 0.10,
                'cost_of_equity': 0.12,
                'cost_of_debt': 0.05,
                'after_tax_cost_of_debt': 0.0375,
                'weight_equity': 0.7,
                'weight_debt': 0.3,
                'tax_rate': 0.25,
                'beta': 1.0,
                'risk_free_rate': self.risk_free_rate,
                'market_risk_premium': 0.05
            }