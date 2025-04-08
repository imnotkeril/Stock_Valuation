import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.config import SECTOR_DCF_PARAMETERS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('communication_valuation')


class CommunicationValuation(BaseValuation):
    """
    Communication sector valuation model.

    Specialized for telecommunication companies, media corporations,
    social networks, and entertainment companies. Includes:

    - Subscriber-based valuation for telecoms and streaming services
    - Digital advertising revenue models for media and social platforms
    - Content library valuation for media companies
    - User metrics integration (ARPU, churn, LTV)
    """

    def __init__(self, data_loader=None):
        """
        Initialize communication sector valuation model with sector-specific parameters.

        Args:
            data_loader: Optional data loader instance
        """
        super().__init__(data_loader)

        # Sector-specific parameters
        self.sector_params = SECTOR_DCF_PARAMETERS.get('Communication Services', {})

        # Default subsector multiples for valuation
        self.subsector_multiples = {
            'Telecommunications': {
                'p_e': 16,
                'ev_ebitda': 7.5,
                'p_s': 2.5,
                'ev_revenue': 3.0,
                'p_fcf': 15
            },
            'Media': {
                'p_e': 18,
                'ev_ebitda': 9,
                'p_s': 3.5,
                'ev_revenue': 4.0,
                'p_fcf': 17
            },
            'Social Media': {
                'p_e': 30,
                'ev_ebitda': 20,
                'p_s': 7,
                'ev_revenue': 7.5,
                'p_fcf': 25
            },
            'Entertainment': {
                'p_e': 22,
                'ev_ebitda': 12,
                'p_s': 4,
                'ev_revenue': 4.5,
                'p_fcf': 20
            }
        }

        # Default parameters for subscriber-based models
        self.subscriber_model_params = {
            'Telecommunications': {
                'churn_rate': 0.15,  # 15% annual churn
                'arpu_growth': 0.02,  # 2% annual ARPU growth
                'subscriber_growth': 0.03,  # 3% annual subscriber growth
                'margin': 0.40,  # 40% margin
                'customer_acquisition_cost': 300  # $300 per new customer
            },
            'Media Streaming': {
                'churn_rate': 0.25,  # 25% annual churn
                'arpu_growth': 0.05,  # 5% annual ARPU growth
                'subscriber_growth': 0.15,  # 15% annual subscriber growth
                'margin': 0.35,  # 35% margin
                'customer_acquisition_cost': 100  # $100 per new customer
            }
        }

        # Content valuation parameters
        self.content_valuation_params = {
            'content_multiple': 5,  # 5x annual content revenue
            'useful_life': 7,  # 7 years average content life
            'decay_rate': 0.20  # 20% annual value decay for content
        }

    def get_valuation(self, ticker: str, financial_data: Optional[Dict] = None,
                      subsector: str = 'Telecommunications',
                      user_metrics: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive communication company valuation.

        Args:
            ticker: Company ticker symbol
            financial_data: Pre-loaded financial data if available
            subsector: Communication subsector (Telecom, Media, Social Media, Entertainment)
            user_metrics: Optional subscriber/user metrics data

        Returns:
            Dictionary with comprehensive valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get market data
            market_data = financial_data.get('market_data', {})

            # Calculate valuations using different methods
            # Traditional DCF, always applicable
            dcf_valuation = self.standard_dcf_valuation(ticker, financial_data, subsector)

            # Multiples valuation, always applicable
            multiples_valuation = self.communication_multiples_valuation(ticker, financial_data, subsector)

            # Subscriber-based valuation for telecom and subscription businesses
            subscriber_valuation = None
            if subsector in ['Telecommunications', 'Media'] and user_metrics and 'subscribers' in user_metrics:
                subscriber_valuation = self.subscriber_based_valuation(ticker, financial_data, user_metrics, subsector)

            # Social media valuation for platforms with users/advertising
            social_valuation = None
            if subsector == 'Social Media' and user_metrics and 'users' in user_metrics:
                social_valuation = self.user_based_valuation(ticker, financial_data, user_metrics)

            # Content library valuation for media and entertainment
            content_valuation = None
            if subsector in ['Media', 'Entertainment'] and financial_data:
                content_valuation = self.content_library_valuation(ticker, financial_data)

            # Combine all valuations
            all_valuations = {
                'dcf': dcf_valuation,
                'multiples': multiples_valuation
            }

            if subscriber_valuation:
                all_valuations['subscriber'] = subscriber_valuation

            if social_valuation:
                all_valuations['social'] = social_valuation

            if content_valuation:
                all_valuations['content'] = content_valuation

            # Create valuation summary with weighted approach
            valuation_values = []

            # Add DCF valuation (weighted based on subsector)
            if dcf_valuation.get('value_per_share') is not None:
                # Higher weight for telecom (more stable cash flows)
                dcf_weight = 0.6 if subsector == 'Telecommunications' else 0.4
                valuation_values.extend([dcf_valuation.get('value_per_share')] * int(dcf_weight * 10))

            # Add multiples valuation
            if multiples_valuation.get('value_per_share') is not None:
                multiples_weight = 0.4
                valuation_values.extend([multiples_valuation.get('value_per_share')] * int(multiples_weight * 10))

            # Add subscriber valuation if available
            if subscriber_valuation and subscriber_valuation.get('value_per_share') is not None:
                # Higher weight for telecom and subscription businesses
                sub_weight = 0.3
                valuation_values.extend([subscriber_valuation.get('value_per_share')] * int(sub_weight * 10))

            # Add social media valuation if available
            if social_valuation and social_valuation.get('value_per_share') is not None:
                # Higher weight for social media companies
                social_weight = 0.4 if subsector == 'Social Media' else 0.2
                valuation_values.extend([social_valuation.get('value_per_share')] * int(social_weight * 10))

            # Add content valuation if available
            if content_valuation and content_valuation.get('value_per_share') is not None:
                # Higher weight for media and entertainment
                content_weight = 0.3 if subsector in ['Media', 'Entertainment'] else 0.1
                valuation_values.extend([content_valuation.get('value_per_share')] * int(content_weight * 10))

            if not valuation_values:
                raise ValueError(f"Could not calculate any valuation for {ticker}")

            # Calculate average, min, max valuation
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
                'sector': 'Communication Services',
                'subsector': subsector,
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
            logger.error(f"Error in communication valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'sector': 'Communication Services',
                'subsector': subsector,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def standard_dcf_valuation(self, ticker: str, financial_data: Dict,
                               subsector: str = 'Telecommunications') -> Dict:
        """
        Perform standard DCF valuation for communication sector companies.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Communication subsector

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

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for DCF calculation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Get DCF parameters with subsector adjustments
            params = self._get_dcf_parameters(subsector)

            # Forecast period and terminal growth
            forecast_years = params.get('forecast_years', 5)
            terminal_growth = params.get('terminal_growth_rate', 0.02)

            # Calculate appropriate discount rate
            discount_rate = self._calculate_communication_discount_rate(ticker, financial_data, subsector)

            # Estimate growth rate based on company stage and subsector
            growth_rate = self._estimate_communication_growth_rate(ticker, financial_data, subsector)

            # Get latest FCF as starting point
            latest_fcf = historical_fcf.iloc[0]

            # Forecast future free cash flows with subsector-specific patterns
            forecast_fcf = []

            for year in range(1, forecast_years + 1):
                # Apply declining growth for telecom, higher sustained growth for digital
                if subsector == 'Telecommunications':
                    # Telecom: Stable growth declining toward terminal rate
                    year_growth = max(growth_rate * (1 - 0.1 * (year - 1)), terminal_growth)
                elif subsector in ['Social Media', 'Media']:
                    # Digital media: Higher initial growth with faster convergence
                    # to terminal rate in later years
                    if year <= 2:
                        year_growth = growth_rate
                    else:
                        year_growth = growth_rate * (1 - 0.2 * (year - 2))
                        year_growth = max(year_growth, terminal_growth)
                else:
                    # Standard declining growth pattern
                    year_growth = max(growth_rate * (1 - 0.15 * (year - 1)), terminal_growth)

                # Apply growth to previous year's FCF
                if year == 1:
                    fcf = latest_fcf * (1 + year_growth)
                else:
                    fcf = forecast_fcf[-1] * (1 + year_growth)

                forecast_fcf.append(fcf)

            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(forecast_fcf[-1], terminal_growth, discount_rate)

            # Discount all future cash flows to present value
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecast_fcf))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Calculate equity value
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'DCF',
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
                'forecast_years': forecast_years,
                'forecast_fcf': forecast_fcf,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating DCF for {ticker}: {e}")
            return {
                'method': 'DCF',
                'error': str(e),
                'value_per_share': None
            }

    def communication_multiples_valuation(self, ticker: str, financial_data: Dict,
                                          subsector: str = 'Telecommunications') -> Dict:
        """
        Perform multiples-based valuation adapted for communication companies.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Communication subsector

        Returns:
            Dictionary with multiples valuation results
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
                raise ValueError("Missing required financial statements for multiples valuation")

            # Get most recent data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Get relevant financials
            metrics = {}

            # Earnings
            if 'Net Income' in latest_income.index:
                metrics['earnings'] = latest_income.loc['Net Income']

            # Revenue
            if 'Total Revenue' in latest_income.index:
                metrics['revenue'] = latest_income.loc['Total Revenue']

            # EBITDA
            if 'EBITDA' in latest_income.index:
                metrics['ebitda'] = latest_income.loc['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                metrics['ebitda'] = latest_income.loc['Operating Income'] + latest_income.loc[
                    'Depreciation & Amortization']

            # Free Cash Flow
            if latest_cash_flow is not None:
                if 'Free Cash Flow' in latest_cash_flow.index:
                    metrics['fcf'] = latest_cash_flow.loc['Free Cash Flow']
                elif 'Operating Cash Flow' in latest_cash_flow.index and 'Capital Expenditure' in latest_cash_flow.index:
                    metrics['fcf'] = latest_cash_flow.loc['Operating Cash Flow'] - abs(
                        latest_cash_flow.loc['Capital Expenditure'])

            # Get appropriate multiples for the subsector
            multiples = self.subsector_multiples.get(subsector, self.subsector_multiples['Telecommunications'])

            # Calculate valuations using different multiples
            valuations = {}

            # P/E valuation
            if 'earnings' in metrics and metrics['earnings'] > 0:
                valuations['pe'] = {
                    'multiple': multiples['p_e'],
                    'value': metrics['earnings'] * multiples['p_e'],
                    'description': 'Price to Earnings'
                }

            # EV/EBITDA valuation
            if 'ebitda' in metrics and metrics['ebitda'] > 0:
                valuations['ev_ebitda'] = {
                    'multiple': multiples['ev_ebitda'],
                    'value': metrics['ebitda'] * multiples['ev_ebitda'],
                    'description': 'Enterprise Value to EBITDA'
                }

                # Adjust for debt and cash to get equity value
                total_debt = 0
                for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                    if debt_name in latest_balance.index:
                        total_debt += latest_balance.loc[debt_name]

                cash = 0
                for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                    if cash_name in latest_balance.index:
                        cash += latest_balance.loc[cash_name]

                valuations['ev_ebitda']['equity_value'] = valuations['ev_ebitda']['value'] - total_debt + cash

            # Revenue-based valuation (important for growth businesses)
            if 'revenue' in metrics and metrics['revenue'] > 0:
                valuations['ps'] = {
                    'multiple': multiples['p_s'],
                    'value': metrics['revenue'] * multiples['p_s'],
                    'description': 'Price to Sales'
                }

                valuations['ev_revenue'] = {
                    'multiple': multiples['ev_revenue'],
                    'value': metrics['revenue'] * multiples['ev_revenue'],
                    'description': 'Enterprise Value to Revenue'
                }

                # Adjust EV/Revenue for debt and cash
                if 'ev_revenue' in valuations:
                    total_debt = 0
                    for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                        if debt_name in latest_balance.index:
                            total_debt += latest_balance.loc[debt_name]

                    cash = 0
                    for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                        if cash_name in latest_balance.index:
                            cash += latest_balance.loc[cash_name]

                    valuations['ev_revenue']['equity_value'] = valuations['ev_revenue']['value'] - total_debt + cash

            # FCF-based valuation
            if 'fcf' in metrics and metrics['fcf'] > 0:
                valuations['p_fcf'] = {
                    'multiple': multiples['p_fcf'],
                    'value': metrics['fcf'] * multiples['p_fcf'],
                    'description': 'Price to Free Cash Flow'
                }

            # Calculate final equity value
            equity_values = []

            # PE and P/FCF already give equity value
            if 'pe' in valuations:
                equity_values.append(valuations['pe']['value'])

            if 'p_fcf' in valuations:
                equity_values.append(valuations['p_fcf']['value'])

            # EV-based metrics need to be adjusted to equity value
            if 'ev_ebitda' in valuations and 'equity_value' in valuations['ev_ebitda']:
                equity_values.append(valuations['ev_ebitda']['equity_value'])

            if 'ev_revenue' in valuations and 'equity_value' in valuations['ev_revenue']:
                equity_values.append(valuations['ev_revenue']['equity_value'])

            # Use PS directly if not enough equity values
            if len(equity_values) < 2 and 'ps' in valuations:
                equity_values.append(valuations['ps']['value'])

            # Calculate average equity value
            if equity_values:
                avg_equity_value = sum(equity_values) / len(equity_values)

                # Calculate per share value
                if shares_outstanding and shares_outstanding > 0:
                    value_per_share = avg_equity_value / shares_outstanding
                else:
                    value_per_share = None
            else:
                avg_equity_value = None
                value_per_share = None

            return {
                'method': 'Communication Multiples',
                'subsector': subsector,
                'metrics': metrics,
                'valuations': valuations,
                'equity_value': avg_equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating communication multiples valuation for {ticker}: {e}")
            return {
                'method': 'Communication Multiples',
                'subsector': subsector,
                'error': str(e),
                'value_per_share': None
            }

    def subscriber_based_valuation(self, ticker: str, financial_data: Dict,
                                   user_metrics: Dict,
                                   subsector: str = 'Telecommunications') -> Dict:
        """
        Perform valuation based on subscriber metrics and lifetime value.

        Useful for telecom companies, streaming services, and subscription businesses.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            user_metrics: Dictionary with subscriber metrics
            subsector: Communication subsector

        Returns:
            Dictionary with subscriber-based valuation results
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Extract subscriber metrics
            if 'subscribers' not in user_metrics:
                raise ValueError("Subscriber count required for subscriber-based valuation")

            subscribers = user_metrics.get('subscribers')
            arpu = user_metrics.get('arpu')  # Average Revenue Per User (monthly)
            churn_rate = user_metrics.get('churn_rate')  # Annual churn rate

            # Get income statement data for ARPU calculation if not provided
            if arpu is None and financial_data and 'income_statement' is not None:
                income_stmt = financial_data.get('income_statement')
                if income_stmt is not None and 'Total Revenue' in income_stmt.index:
                    latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    # Estimate monthly ARPU from annual revenue
                    arpu = (latest_revenue / subscribers) / 12

            # Use default parameters if metrics not provided
            if arpu is None:
                # Estimate based on subsector averages
                arpu = 50 if subsector == 'Telecommunications' else 12  # Monthly ARPU

            if churn_rate is None:
                # Use subsector default
                params = self.subscriber_model_params.get(subsector, self.subscriber_model_params['Telecommunications'])
                churn_rate = params.get('churn_rate')

            # Get other parameters from defaults
            params = self.subscriber_model_params.get(subsector, self.subscriber_model_params['Telecommunications'])
            arpu_growth = params.get('arpu_growth')
            subscriber_growth = params.get('subscriber_growth')
            margin = params.get('margin')
            cac = params.get('customer_acquisition_cost')

            # Calculate average subscriber lifetime (1/churn_rate)
            avg_lifetime = 1 / churn_rate

            # Calculate lifetime value (LTV) using DCF approach
            discount_rate = self._calculate_communication_discount_rate(ticker, financial_data, subsector)

            # Calculate LTV
            ltv = 0
            current_arpu = arpu

            # Project 10 years of cash flows (or less if churn is high)
            projection_years = min(10, int(avg_lifetime * 2))

            for year in range(1, projection_years + 1):
                # Probability of customer still being active
                survival_prob = (1 - churn_rate) ** (year - 1)

                # Annual revenue for the year (12 * monthly ARPU)
                annual_revenue = current_arpu * 12

                # Contribution margin
                contribution = annual_revenue * margin

                # Present value of year's contribution
                pv_contribution = contribution * survival_prob / (1 + discount_rate) ** year

                # Add to LTV
                ltv += pv_contribution

                # Grow ARPU for next year
                current_arpu *= (1 + arpu_growth)

            # Subtract customer acquisition cost
            ltv -= cac

            # Ensure LTV is positive
            ltv = max(0, ltv)

            # Calculate total subscriber value
            total_subscriber_value = ltv * subscribers

            # Project future subscriber growth (5-year horizon)
            future_value = 0
            current_subscribers = subscribers

            for year in range(1, 6):
                # New subscribers in the year
                new_subscribers = current_subscribers * subscriber_growth

                # Value of new subscribers (LTV discounted for acquisition year)
                new_subscriber_value = new_subscribers * ltv / (1 + discount_rate) ** year

                # Add to future value
                future_value += new_subscriber_value

                # Update subscriber count for next year
                current_subscribers += new_subscribers

            # Calculate enterprise value
            enterprise_value = total_subscriber_value + future_value

            # Calculate equity value by adjusting for debt and cash
            if financial_data and 'balance_sheet' in financial_data:
                balance_sheet = financial_data.get('balance_sheet')

                # Get debt and cash
                total_debt = 0
                for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                    if debt_name in balance_sheet.index:
                        total_debt += balance_sheet.loc[debt_name].iloc[0]

                cash = 0
                for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                    if cash_name in balance_sheet.index:
                        cash += balance_sheet.loc[cash_name].iloc[0]

                equity_value = enterprise_value - total_debt + cash
            else:
                equity_value = enterprise_value

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'Subscriber-Based',
                'subscribers': subscribers,
                'arpu': arpu,
                'churn_rate': churn_rate,
                'customer_lifetime': avg_lifetime,
                'ltv': ltv,
                'cac': cac,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating subscriber-based valuation for {ticker}: {e}")
            return {
                'method': 'Subscriber-Based',
                'error': str(e),
                'value_per_share': None
            }

    def user_based_valuation(self, ticker: str, financial_data: Dict,
                             user_metrics: Dict) -> Dict:
        """
        Perform valuation based on user metrics for social media platforms.

        Uses metrics like DAU/MAU, revenue per user, and engagement.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            user_metrics: Dictionary with user metrics

        Returns:
            Dictionary with user-based valuation results
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Extract user metrics
            if 'users' not in user_metrics:
                raise ValueError("User count required for user-based valuation")

            users = user_metrics.get('users')  # Total users
            mau = user_metrics.get('mau', users)  # Monthly active users
            dau = user_metrics.get('dau')  # Daily active users
            arpu = user_metrics.get('arpu')  # Annual revenue per user

            # Extract income statement data
            income_stmt = financial_data.get('income_statement')

            # Calculate ARPU if not provided but revenue is available
            if arpu is None and income_stmt is not None and 'Total Revenue' in income_stmt.index:
                latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                arpu = latest_revenue / mau
            elif arpu is None:
                # Use industry average if no data
                arpu = 25  # $25 per user annually

            # Calculate engagement ratio (DAU/MAU) if both metrics available
            engagement_ratio = dau / mau if dau and mau else 0.5  # Default 50% engagement

            # Calculate user value based on ARPU, engagement, and growth potential
            # Higher engagement = higher value per user
            base_user_value = arpu * 5  # Base multiplier

            # Adjust for engagement
            if engagement_ratio > 0.7:
                # High engagement premium
                user_value = base_user_value * 1.5
            elif engagement_ratio > 0.5:
                # Moderate engagement
                user_value = base_user_value * 1.2
            else:
                # Lower engagement
                user_value = base_user_value

            # Calculate base platform value
            platform_value = mau * user_value

            # Calculate growth potential value
            # If user growth rate is provided, use it
            user_growth_rate = user_metrics.get('growth_rate', 0.1)  # Default 10% annual growth

            # Project 5 years of user growth
            future_value = 0
            current_users = mau
            discount_rate = self._calculate_communication_discount_rate(ticker, financial_data, 'Social Media')

            for year in range(1, 6):
                # New users in the year
                new_users = current_users * user_growth_rate

                # Value of new users (accounting for time value)
                new_user_value = new_users * user_value / (1 + discount_rate) ** year

                # Add to future value
                future_value += new_user_value

                # Update user count for next year (compounding growth)
                current_users += new_users

                # Gradually decrease growth rate for realism
                user_growth_rate = max(user_growth_rate * 0.8, 0.05)

            # Calculate enterprise value
            enterprise_value = platform_value + future_value

            # Calculate equity value by adjusting for debt and cash
            if financial_data and 'balance_sheet' in financial_data:
                balance_sheet = financial_data.get('balance_sheet')

                # Get debt and cash
                total_debt = 0
                for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                    if debt_name in balance_sheet.index:
                        total_debt += balance_sheet.loc[debt_name].iloc[0]

                cash = 0
                for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                    if cash_name in balance_sheet.index:
                        cash += balance_sheet.loc[cash_name].iloc[0]

                equity_value = enterprise_value - total_debt + cash
            else:
                equity_value = enterprise_value

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'User-Based (Social Media)',
                'monthly_active_users': mau,
                'daily_active_users': dau if dau else None,
                'engagement_ratio': engagement_ratio,
                'arpu': arpu,
                'user_value': user_value,
                'platform_value': platform_value,
                'future_value': future_value,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating user-based valuation for {ticker}: {e}")
            return {
                'method': 'User-Based (Social Media)',
                'error': str(e),
                'value_per_share': None
            }

    def content_library_valuation(self, ticker: str, financial_data: Dict) -> Dict:
        """
        Value content libraries for media and entertainment companies.

        This method estimates the value of a company's content catalog
        (shows, movies, music, etc.) based on revenue generation and longevity.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary

        Returns:
            Dictionary with content library valuation results
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Extract income statement data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None:
                raise ValueError("Income statement required for content library valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]

            # Estimate content revenue
            # We need to identify revenue from content licensing, streaming, distribution
            content_revenue = None

            # Try to find content-specific revenue in segmented data
            # This is a simplified approach - actual companies would have detailed breakdowns
            for revenue_name in ['Content Revenue', 'Media Revenue', 'Entertainment Revenue',
                                 'Licensing Revenue', 'Subscription Revenue']:
                if revenue_name in latest_income.index:
                    content_revenue = latest_income.loc[revenue_name]
                    break

            # If no specific content revenue, estimate from total revenue
            if content_revenue is None:
                if 'Total Revenue' in latest_income.index:
                    # Estimate content revenue as percentage of total
                    total_revenue = latest_income.loc['Total Revenue']
                    # Default: 60% of revenue for media companies comes from content
                    content_revenue = total_revenue * 0.6
                else:
                    raise ValueError("Could not estimate content revenue")

            # Get content production/acquisition costs
            content_costs = None

            for cost_name in ['Content Costs', 'Programming Costs', 'Production Costs',
                              'Content Acquisition']:
                if cost_name in latest_income.index:
                    content_costs = latest_income.loc[cost_name]
                    break

            # If content costs not found, estimate from intangible assets or COGS
            if content_costs is None:
                if balance_sheet is not None:
                    latest_balance = balance_sheet.iloc[:, 0]

                    for asset_name in ['Content Assets', 'Film Library', 'Intangible Assets',
                                       'Content Rights']:
                        if asset_name in latest_balance.index:
                            # Use asset value as proxy for accumulated content investment
                            content_costs = latest_balance.loc[asset_name] * 0.2  # Annual amortization
                            break

            # If still no content costs, estimate from revenue
            if content_costs is None:
                # Content costs typically 60-70% of content revenue for production companies
                content_costs = content_revenue * 0.65

            # Calculate content profit margin
            content_margin = (content_revenue - content_costs) / content_revenue

            # Apply valuation multiple based on catalog quality and longevity
            # Base multiple from parameters
            base_multiple = self.content_valuation_params.get('content_multiple', 5)

            # Adjust multiple based on margin
            if content_margin > 0.4:
                # Premium content with high margins
                adjusted_multiple = base_multiple * 1.3
            elif content_margin > 0.2:
                # Standard content
                adjusted_multiple = base_multiple
            else:
                # Lower margin content
                adjusted_multiple = base_multiple * 0.7

            # Calculate base content library value
            content_library_value = content_revenue * adjusted_multiple

            # Account for future content production and catalog expansion
            # Assuming content investment grows and generates returns over time
            future_content_value = 0
            growth_rate = 0.05  # 5% annual growth in content value
            discount_rate = self._calculate_communication_discount_rate(ticker, financial_data, 'Media')

            for year in range(1, 6):
                # Future year's content value growth
                year_value = content_library_value * growth_rate

                # Discount to present value
                pv_value = year_value / (1 + discount_rate) ** year

                # Add to future content value
                future_content_value += pv_value

            # Calculate total content enterprise value
            content_enterprise_value = content_library_value + future_content_value

            # Adjust for company's other assets and liabilities to get content-based equity value
            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                # Get debt and cash
                total_debt = 0
                for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                    if debt_name in latest_balance.index:
                        total_debt += latest_balance.loc[debt_name]

                cash = 0
                for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                    if cash_name in latest_balance.index:
                        cash += latest_balance.loc[cash_name]

                # Adjust for non-content assets (we don't want to double-count everything)
                # Simplified approach: add 30% of non-content assets
                total_assets = latest_balance.loc['Total Assets'] if 'Total Assets' in latest_balance.index else 0
                content_assets = total_assets * 0.7  # Assume 70% of assets are content-related
                non_content_assets = total_assets - content_assets

                equity_value = content_enterprise_value + (non_content_assets * 0.3) - total_debt + cash
            else:
                equity_value = content_enterprise_value

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'Content Library Valuation',
                'content_revenue': content_revenue,
                'content_costs': content_costs,
                'content_margin': content_margin,
                'adjusted_multiple': adjusted_multiple,
                'content_library_value': content_library_value,
                'future_content_value': future_content_value,
                'content_enterprise_value': content_enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating content library valuation for {ticker}: {e}")
            return {
                'method': 'Content Library Valuation',
                'error': str(e),
                'value_per_share': None
            }

    def _calculate_communication_discount_rate(self, ticker: str, financial_data: Dict,
                                               subsector: str) -> float:
        """
        Calculate appropriate discount rate for communication company based on
        risk profile and subsector.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Communication subsector

        Returns:
            Discount rate as float
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            beta = market_data.get('beta')

            # Base discount rate from sector parameters
            base_rate = self.sector_params.get('default_discount_rate', 0.10)

            # Subsector risk premium
            subsector_premium = {
                'Telecommunications': -0.01,  # Lower risk due to utility-like nature
                'Media': 0.00,  # Base case
                'Social Media': 0.03,  # Higher risk due to disruption/competition
                'Entertainment': 0.01  # Slightly higher risk
            }.get(subsector, 0.00)

            # Company-specific adjustments
            company_premium = 0.00

            # Beta-based adjustment if available
            if beta is not None:
                # Higher beta = higher risk = higher discount rate
                if beta > 1.3:
                    company_premium += 0.01
                elif beta < 0.8:
                    company_premium -= 0.01

            # Adjust based on capital intensity (telecom has high capex)
            if subsector == 'Telecommunications':
                # Check capex intensity if cash flow available
                cash_flow = financial_data.get('cash_flow')
                income_stmt = financial_data.get('income_statement')

                if cash_flow is not None and income_stmt is not None:
                    latest_cf = cash_flow.iloc[:, 0]
                    latest_income = income_stmt.iloc[:, 0]

                    if 'Capital Expenditure' in latest_cf.index and 'Total Revenue' in latest_income.index:
                        capex = abs(latest_cf.loc['Capital Expenditure'])
                        revenue = latest_income.loc['Total Revenue']

                        capex_intensity = capex / revenue

                        if capex_intensity > 0.20:
                            # Very high capex intensity
                            company_premium += 0.01

            # Adjust based on revenue stability and customer base
            # More stable revenue = lower discount rate
            income_stmt = financial_data.get('income_statement')

            if income_stmt is not None and len(income_stmt.columns) >= 3:
                # Check revenue volatility over time
                if 'Total Revenue' in income_stmt.index:
                    revenues = income_stmt.loc['Total Revenue']

                    # Calculate year-over-year changes
                    yoy_changes = []

                    for i in range(len(revenues) - 1):
                        if revenues[i + 1] > 0:  # Avoid division by zero
                            change = abs((revenues[i] / revenues[i + 1]) - 1)
                            yoy_changes.append(change)

                    if yoy_changes:
                        avg_volatility = sum(yoy_changes) / len(yoy_changes)

                        if avg_volatility > 0.15:
                            # High volatility
                            company_premium += 0.01
                        elif avg_volatility < 0.05:
                            # Low volatility
                            company_premium -= 0.01

            # Calculate final discount rate
            discount_rate = base_rate + subsector_premium + company_premium

            # Ensure discount rate is in reasonable range
            discount_rate = max(0.07, min(0.20, discount_rate))

            return discount_rate

        except Exception as e:
            logger.warning(f"Error calculating communication discount rate for {ticker}: {e}")
            # Return default rate from sector parameters
            return self.sector_params.get('default_discount_rate', 0.10)

    def _estimate_communication_growth_rate(self, ticker: str, financial_data: Dict,
                                            subsector: str) -> float:
        """
        Estimate appropriate growth rate for communication company based on
        historical data and subsector characteristics.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Communication subsector

        Returns:
            Estimated growth rate as float
        """
        try:
            # Get income statement for historical growth analysis
            income_stmt = financial_data.get('income_statement')

            if income_stmt is None or income_stmt.empty:
                # Use subsector default if no data available
                return {
                    'Telecommunications': 0.02,  # 2% for telecom (mature)
                    'Media': 0.04,  # 4% for traditional media
                    'Social Media': 0.15,  # 15% for social platforms
                    'Entertainment': 0.05  # 5% for entertainment
                }.get(subsector, 0.04)

            # Check revenue growth history
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue']

                # Calculate annual growth rates if we have at least 2 years of data
                growth_rates = []

                if len(revenue) >= 2:
                    for i in range(len(revenue) - 1):
                        if revenue[i + 1] > 0:  # Avoid division by zero
                            annual_growth = (revenue[i] / revenue[i + 1]) - 1
                            growth_rates.append(annual_growth)

                if growth_rates:
                    # Calculate weighted average (more recent years have higher weight)
                    weights = list(range(1, len(growth_rates) + 1))
                    weighted_growth = sum(r * w for r, w in zip(growth_rates, weights)) / sum(weights)

                    # Adjust extreme values
                    if weighted_growth > 0.25:
                        # Cap very high growth rate and adjust down
                        weighted_growth = 0.25 - (weighted_growth - 0.25) * 0.5
                    elif weighted_growth < 0:
                        # Adjust negative growth less severely as it may be temporary
                        weighted_growth = weighted_growth * 0.5

                    # Blend with subsector average for stability
                    subsector_growth = {
                        'Telecommunications': 0.02,
                        'Media': 0.04,
                        'Social Media': 0.15,
                        'Entertainment': 0.05
                    }.get(subsector, 0.04)

                    # For telecom, history is more predictive; for social media, industry trends matter more
                    if subsector == 'Telecommunications':
                        blend_factor = 0.7  # 70% weight to historical performance
                    elif subsector == 'Social Media':
                        blend_factor = 0.4  # 40% weight to historical, 60% to industry trends
                    else:
                        blend_factor = 0.6  # 60% weight to historical

                    blended_growth = (weighted_growth * blend_factor) + (subsector_growth * (1 - blend_factor))

                    return max(0.01, min(0.25, blended_growth))  # Ensure reasonable range

            # If we can't calculate from history, use subsector default
            return {
                'Telecommunications': 0.02,
                'Media': 0.04,
                'Social Media': 0.15,
                'Entertainment': 0.05
            }.get(subsector, 0.04)

        except Exception as e:
            logger.warning(f"Error estimating communication growth rate for {ticker}: {e}")
            # Return default rate based on subsector
            return {
                'Telecommunications': 0.02,
                'Media': 0.04,
                'Social Media': 0.15,
                'Entertainment': 0.05
            }.get(subsector, 0.04)