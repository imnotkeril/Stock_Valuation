import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from valuation.base_valuation import BaseValuation
from config import SECTOR_DCF_PARAMETERS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_estate_valuation')


class RealEstateValuation(BaseValuation):
    """
    Real Estate sector valuation model.

    Specialized for Real Estate Investment Trusts (REITs), property developers,
    and real estate services companies with unique valuation approaches
    including:
    - Net Asset Value (NAV)
    - Funds From Operations (FFO) multiple
    - Adjusted Funds From Operations (AFFO) multiple
    - Cap Rate analysis
    - DCF adapted for real estate cash flows
    """

    def __init__(self, data_loader=None):
        """
        Initialize real estate valuation model with sector-specific parameters.

        Args:
            data_loader: Optional data loader instance
        """
        super().__init__(data_loader)

        # Sector-specific parameters
        self.sector_params = SECTOR_DCF_PARAMETERS.get('Real Estate', {})

        # Default capitalization rates by property type
        self.default_cap_rates = {
            'Office': 0.07,  # 7.0%
            'Retail': 0.065,  # 6.5%
            'Industrial': 0.06,  # 6.0%
            'Multifamily': 0.055,  # 5.5%
            'Hotel': 0.085,  # 8.5%
            'Healthcare': 0.075,  # 7.5%
            'Self-storage': 0.065,  # 6.5%
            'Data Center': 0.06,  # 6.0%
            'Mixed': 0.065  # 6.5%
        }

        # Default assumptions
        self.default_expense_ratio = 0.40  # 40% expense ratio if not available
        self.noi_growth_rate = 0.02  # 2% NOI growth rate

    def get_valuation(self, ticker: str, financial_data: Optional[Dict] = None,
                      property_type: str = 'Mixed') -> Dict:
        """
        Perform comprehensive real estate valuation using multiple methods.

        Args:
            ticker: Company ticker symbol
            financial_data: Pre-loaded financial data if available
            property_type: Type of property (Office, Retail, Industrial, etc.)

        Returns:
            Dictionary with valuation results from multiple approaches
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get market data
            market_data = financial_data.get('market_data', {})

            # Calculate valuations using different methods
            nav_valuation = self.nav_valuation(ticker, financial_data)
            ffo_valuation = self.ffo_valuation(ticker, financial_data)
            cap_rate_valuation = self.cap_rate_valuation(ticker, financial_data, property_type)
            dcf_valuation = self.real_estate_dcf(ticker, financial_data)

            # Combine all valuations
            all_valuations = {
                'nav': nav_valuation,
                'ffo': ffo_valuation,
                'cap_rate': cap_rate_valuation,
                'dcf': dcf_valuation
            }

            # Create valuation summary with all methods
            valuation_values = [v.get('value_per_share') for v in all_valuations.values()
                                if v.get('value_per_share') is not None]

            if not valuation_values:
                raise ValueError(f"Could not calculate any valuation for {ticker}")

            # Calculate average valuation and range
            avg_valuation = sum(valuation_values) / len(valuation_values)
            min_valuation = min(valuation_values)
            max_valuation = max(valuation_values)

            # Current price for comparison
            current_price = market_data.get('share_price')

            # Calculate potential upside/downside
            if current_price:
                potential_upside = (avg_valuation / current_price - 1) * 100
            else:
                potential_upside = None

            return {
                'company': ticker,
                'sector': 'Real Estate',
                'property_type': property_type,
                'current_price': current_price,
                'valuation_summary': {
                    'average': avg_valuation,
                    'min': min_valuation,
                    'max': max_valuation,
                    'range': max_valuation - min_valuation,
                    'potential_upside': potential_upside
                },
                'valuation_methods': all_valuations,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"Error in real estate valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'sector': 'Real Estate',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def nav_valuation(self, ticker: str, financial_data: Dict) -> Dict:
        """
        Calculate Net Asset Value (NAV) - a key metric for real estate companies.

        NAV = Market Value of Assets - Total Liabilities

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary

        Returns:
            Dictionary with NAV valuation results
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if balance_sheet is None or income_stmt is None:
                raise ValueError("Missing required financial statements for NAV calculation")

            # Get latest balance sheet
            latest_balance = balance_sheet.iloc[:, 0]
            latest_income = income_stmt.iloc[:, 0]

            # Get total assets and liabilities
            if 'Total Assets' in latest_balance.index and 'Total Liabilities' in latest_balance.index:
                total_assets = latest_balance.loc['Total Assets']
                total_liabilities = latest_balance.loc['Total Liabilities']
                book_nav = total_assets - total_liabilities
            else:
                raise ValueError("Balance sheet missing critical asset or liability data")

            # For REITs, adjust book value of real estate assets based on market conditions
            # We'll use a simplified approach with an adjustment factor
            real_estate_assets = 0

            # Try to find real estate assets
            for asset_name in ['Real Estate Investments', 'Property, Plant, and Equipment',
                               'Land', 'Buildings', 'Investment Properties']:
                if asset_name in latest_balance.index:
                    real_estate_assets += latest_balance.loc[asset_name]

            # If we couldn't identify specific real estate assets, estimate from total assets
            # Most REITs have 70-90% of their assets in real estate
            if real_estate_assets == 0:
                real_estate_assets = total_assets * 0.8

            # Apply market adjustment (premium or discount to book value)
            # The factor could be derived from market conditions, cap rates, etc.
            # For simplicity, we'll use a modest premium of 10%
            market_adjustment_factor = 1.1

            # Calculate market-adjusted NAV
            adjusted_real_estate_value = real_estate_assets * market_adjustment_factor
            adjustment_amount = adjusted_real_estate_value - real_estate_assets
            market_nav = book_nav + adjustment_amount

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                nav_per_share = market_nav / shares_outstanding
            else:
                nav_per_share = None

            return {
                'method': 'NAV',
                'book_nav': book_nav,
                'market_nav': market_nav,
                'total_assets': total_assets,
                'real_estate_assets': real_estate_assets,
                'total_liabilities': total_liabilities,
                'adjustment_factor': market_adjustment_factor,
                'adjustment_amount': adjustment_amount,
                'value_per_share': nav_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating NAV for {ticker}: {e}")
            return {
                'method': 'NAV',
                'error': str(e),
                'value_per_share': None
            }

    def ffo_valuation(self, ticker: str, financial_data: Dict) -> Dict:
        """
        Calculate valuation based on Funds From Operations (FFO) and
        Adjusted Funds From Operations (AFFO) - key metrics for REITs.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary

        Returns:
            Dictionary with FFO/AFFO valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None:
                raise ValueError("Missing income statement for FFO calculation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Calculate FFO (Funds From Operations)
            # FFO = Net Income + Depreciation & Amortization + (Losses - Gains on Property Sales)
            if 'Net Income' not in latest_income.index:
                raise ValueError("Net Income data missing from income statement")

            net_income = latest_income.loc['Net Income']

            # Get depreciation & amortization
            depreciation = 0
            for d_name in ['Depreciation & Amortization', 'Depreciation', 'Depreciation and Amortization']:
                if d_name in latest_income.index:
                    depreciation += latest_income.loc[d_name]
                    break

            # Try to get gains/losses from property sales (if available)
            # This information might be hard to extract automatically from standard reports
            property_gains_losses = 0
            if latest_cash_flow is not None:
                for gl_name in ['Gain/Loss on Sale of Assets', 'Gain/Loss on Sale of Property']:
                    if gl_name in latest_cash_flow.index:
                        property_gains_losses -= latest_cash_flow.loc[gl_name]  # Subtract gains, add losses
                        break

            # Calculate FFO
            ffo = net_income + depreciation + property_gains_losses

            # Calculate AFFO (Adjusted FFO)
            # AFFO = FFO - Recurring Capital Expenditures
            # We'll estimate recurring capex as a percentage of depreciation (typically 40-60%)
            recurring_capex = depreciation * 0.5
            affo = ffo - recurring_capex

            # Calculate per share values
            if shares_outstanding and shares_outstanding > 0:
                ffo_per_share = ffo / shares_outstanding
                affo_per_share = affo / shares_outstanding
            else:
                ffo_per_share = None
                affo_per_share = None

            # Apply valuation multiples
            # FFO multiples typically range from 12-20x depending on quality, growth, etc.
            ffo_multiple = 15  # Median FFO multiple
            affo_multiple = 17  # AFFO multiples are typically higher

            # Calculate values using multiples
            if ffo_per_share is not None:
                ffo_value = ffo_per_share * ffo_multiple
                affo_value = affo_per_share * affo_multiple

                # Average of FFO and AFFO valuations
                average_value = (ffo_value + affo_value) / 2
            else:
                ffo_value = None
                affo_value = None
                average_value = None

            return {
                'method': 'FFO/AFFO',
                'ffo': ffo,
                'affo': affo,
                'ffo_per_share': ffo_per_share,
                'affo_per_share': affo_per_share,
                'ffo_multiple': ffo_multiple,
                'affo_multiple': affo_multiple,
                'ffo_value': ffo_value,
                'affo_value': affo_value,
                'value_per_share': average_value
            }

        except Exception as e:
            logger.error(f"Error calculating FFO valuation for {ticker}: {e}")
            return {
                'method': 'FFO/AFFO',
                'error': str(e),
                'value_per_share': None
            }

    def cap_rate_valuation(self, ticker: str, financial_data: Dict, property_type: str = 'Mixed') -> Dict:
        """
        Calculate valuation based on Capitalization Rate (Cap Rate) - 
        the ratio of Net Operating Income (NOI) to property value.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            property_type: Type of property portfolio

        Returns:
            Dictionary with Cap Rate valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for Cap Rate valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate NOI (Net Operating Income)
            # NOI = Operating Revenue - Operating Expenses
            operating_revenue = None

            # Try to find operating revenue
            for rev_name in ['Total Revenue', 'Revenue', 'Rental Income', 'Property Revenue']:
                if rev_name in latest_income.index:
                    operating_revenue = latest_income.loc[rev_name]
                    break

            if operating_revenue is None:
                raise ValueError("Could not determine operating revenue")

            # Try to find operating expenses or estimate them
            operating_expenses = None

            for exp_name in ['Operating Expenses', 'Property Operating Expenses']:
                if exp_name in latest_income.index:
                    operating_expenses = latest_income.loc[exp_name]
                    break

            # If operating expenses not explicitly stated, estimate using typical expense ratio
            if operating_expenses is None:
                operating_expenses = operating_revenue * self.default_expense_ratio

            # Calculate NOI
            noi = operating_revenue - operating_expenses

            # Get property value from balance sheet
            property_value = None

            for prop_name in ['Real Estate Investments', 'Property, Plant, and Equipment',
                              'Investment Properties']:
                if prop_name in latest_balance.index:
                    property_value = latest_balance.loc[prop_name]
                    break

            if property_value is None:
                # If we can't find property value, estimate using Total Assets
                # Most REITs have 70-90% of their assets in real estate
                property_value = latest_balance.loc[
                                     'Total Assets'] * 0.8 if 'Total Assets' in latest_balance.index else None

            if property_value is None:
                raise ValueError("Could not determine property value")

            # Calculate implied cap rate based on financials
            implied_cap_rate = noi / property_value

            # Get market cap rate for the property type
            market_cap_rate = self.default_cap_rates.get(property_type, self.default_cap_rates['Mixed'])

            # Calculate property market value using market cap rate
            # Value = NOI / Cap Rate
            market_value = noi / market_cap_rate

            # Calculate equity value (subtracting debt)
            total_debt = 0

            for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_name in latest_balance.index:
                    total_debt += latest_balance.loc[debt_name]

            equity_value = market_value - total_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'Cap Rate',
                'noi': noi,
                'book_property_value': property_value,
                'implied_cap_rate': implied_cap_rate,
                'market_cap_rate': market_cap_rate,
                'market_property_value': market_value,
                'total_debt': total_debt,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating Cap Rate valuation for {ticker}: {e}")
            return {
                'method': 'Cap Rate',
                'error': str(e),
                'value_per_share': None
            }

    def real_estate_dcf(self, ticker: str, financial_data: Dict) -> Dict:
        """
        Perform DCF valuation adapted for real estate companies.

        This DCF model is modified to focus on:
        - NOI (Net Operating Income) growth rather than general FCF
        - Terminal value based on exit cap rate
        - REIT-specific adjustments

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary

        Returns:
            Dictionary with DCF valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for DCF valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate NOI (Net Operating Income)
            operating_revenue = None

            # Try to find operating revenue
            for rev_name in ['Total Revenue', 'Revenue', 'Rental Income', 'Property Revenue']:
                if rev_name in latest_income.index:
                    operating_revenue = latest_income.loc[rev_name]
                    break

            if operating_revenue is None:
                raise ValueError("Could not determine operating revenue")

            # Try to find operating expenses or estimate them
            operating_expenses = None

            for exp_name in ['Operating Expenses', 'Property Operating Expenses']:
                if exp_name in latest_income.index:
                    operating_expenses = latest_income.loc[exp_name]
                    break

            # If operating expenses not explicitly stated, estimate using typical expense ratio
            if operating_expenses is None:
                operating_expenses = operating_revenue * self.default_expense_ratio

            # Calculate initial NOI
            initial_noi = operating_revenue - operating_expenses

            # Get parameters
            forecast_years = self.sector_params.get('forecast_years', 5)
            terminal_growth = self.sector_params.get('terminal_growth_rate', 0.02)
            discount_rate = self.sector_params.get('default_discount_rate', 0.08)
            exit_cap_rate = 0.065  # Typical exit cap rate for stabilized properties

            # Estimate NOI growth (could be refined with historical analysis)
            # We'll use a simplified approach with a constant growth rate
            noi_growth_rate = self.noi_growth_rate

            # Project NOI for forecast period
            projected_noi = []
            current_noi = initial_noi

            for year in range(1, forecast_years + 1):
                current_noi *= (1 + noi_growth_rate)
                projected_noi.append(current_noi)

            # Calculate terminal value using exit cap rate
            terminal_noi = projected_noi[-1] * (1 + terminal_growth)
            terminal_value = terminal_noi / exit_cap_rate

            # Discount cash flows
            present_value_noi = sum(noi / (1 + discount_rate) ** year
                                    for year, noi in enumerate(projected_noi, 1))

            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_noi + present_value_terminal

            # Subtract debt to get equity value
            total_debt = 0

            for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_name in latest_balance.index:
                    total_debt += latest_balance.loc[debt_name]

            # Add cash
            cash = 0

            for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                if cash_name in latest_balance.index:
                    cash += latest_balance.loc[cash_name]

            equity_value = enterprise_value - total_debt + cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'DCF',
                'initial_noi': initial_noi,
                'noi_growth_rate': noi_growth_rate,
                'projected_noi': projected_noi,
                'discount_rate': discount_rate,
                'exit_cap_rate': exit_cap_rate,
                'terminal_value': terminal_value,
                'present_value_noi': present_value_noi,
                'present_value_terminal': present_value_terminal,
                'enterprise_value': enterprise_value,
                'total_debt': total_debt,
                'cash': cash,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating real estate DCF for {ticker}: {e}")
            return {
                'method': 'DCF',
                'error': str(e),
                'value_per_share': None
            }

    def calculate_premium_discount_to_nav(self, ticker: str, financial_data: Optional[Dict] = None) -> Dict:
        """
        Calculate premium or discount to NAV - a key metric for REITs.

        Args:
            ticker: Company ticker symbol
            financial_data: Pre-loaded financial data if available

        Returns:
            Dictionary with premium/discount analysis
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get NAV valuation
            nav_result = self.nav_valuation(ticker, financial_data)

            # Get market data
            market_data = financial_data.get('market_data', {})
            current_price = market_data.get('share_price')

            if current_price is None or nav_result.get('value_per_share') is None:
                raise ValueError("Missing required data for premium/discount calculation")

            # Calculate premium/discount
            nav_per_share = nav_result.get('value_per_share')
            premium_discount_pct = ((current_price / nav_per_share) - 1) * 100

            # Interpret the result
            if premium_discount_pct > 5:
                interpretation = "Trading at a significant premium to NAV"
            elif premium_discount_pct > 0:
                interpretation = "Trading at a slight premium to NAV"
            elif premium_discount_pct > -5:
                interpretation = "Trading close to NAV"
            elif premium_discount_pct > -15:
                interpretation = "Trading at a discount to NAV"
            else:
                interpretation = "Trading at a significant discount to NAV"

            return {
                'nav_per_share': nav_per_share,
                'current_price': current_price,
                'premium_discount_pct': premium_discount_pct,
                'interpretation': interpretation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"Error calculating NAV premium/discount for {ticker}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }