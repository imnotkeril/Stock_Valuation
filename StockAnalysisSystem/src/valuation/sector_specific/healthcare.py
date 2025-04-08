import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.config import SECTOR_DCF_PARAMETERS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('healthcare_valuation')


class HealthcareValuation(BaseValuation):
    """
    Healthcare sector valuation model.

    Specialized for pharmaceutical companies, biotechnology firms,
    medical device manufacturers, and healthcare service providers.
    Features specialized valuation methodologies including:

    - DCF with R&D adjustment
    - Pipeline/Drug value analysis
    - Patent portfolio valuation
    - Multiples analysis adapted for healthcare subsectors
    """

    def __init__(self, data_loader=None):
        """
        Initialize healthcare valuation model with sector-specific parameters.

        Args:
            data_loader: Optional data loader instance
        """
        super().__init__(data_loader)

        # Sector-specific parameters
        self.sector_params = SECTOR_DCF_PARAMETERS.get('Healthcare', {})

        # Default subsector multiples for valuation
        self.subsector_multiples = {
            'Pharmaceuticals': {
                'p_e': 20,
                'ev_ebitda': 13,
                'p_s': 4.5,
                'ev_revenue': 5,
                'p_b': 4.2
            },
            'Biotechnology': {
                'p_e': 25,
                'ev_ebitda': 15,
                'p_s': 8,
                'ev_revenue': 9,
                'p_b': 5
            },
            'Medical Devices': {
                'p_e': 22,
                'ev_ebitda': 16,
                'p_s': 5,
                'ev_revenue': 6,
                'p_b': 4
            },
            'Healthcare Services': {
                'p_e': 18,
                'ev_ebitda': 12,
                'p_s': 2,
                'ev_revenue': 2.5,
                'p_b': 3.5
            },
            'Healthcare Technology': {
                'p_e': 28,
                'ev_ebitda': 18,
                'p_s': 7,
                'ev_revenue': 7.5,
                'p_b': 6
            }
        }

        # Default R&D success rates by phase
        self.rd_success_rates = {
            'Preclinical': 0.05,  # 5% success from preclinical to approval
            'Phase1': 0.10,  # 10% success from Phase 1 to approval
            'Phase2': 0.30,  # 30% success from Phase 2 to approval
            'Phase3': 0.60,  # 60% success from Phase 3 to approval
            'Filed': 0.90  # 90% success from filing to approval
        }

        # Default R&D capitalization factor
        # What portion of R&D is viewed as investment vs. expense
        self.rd_capitalization_factor = 0.70  # 70% of R&D treated as investment

    def rd_adjusted_dcf(self, ticker: str, financial_data: Dict, subsector: str = 'Pharmaceuticals') -> Dict:
        """
        Perform DCF valuation adjusted for R&D investments in healthcare companies.

        Healthcare companies, especially in pharma and biotech, invest heavily in R&D.
        This method treats a portion of R&D as capital investment rather than expense,
        which gives a more accurate picture of true economic value.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Healthcare subsector

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
                raise ValueError("Missing required financial statements for DCF calculation")

            # Get historical data for analysis
            income_hist = income_stmt.copy()

            # Find R&D expenses in income statement
            rd_expenses = None
            rd_colname = None

            for rd_name in ['Research & Development', 'R&D Expenses', 'Research and Development']:
                if rd_name in income_hist.index:
                    rd_expenses = income_hist.loc[rd_name]
                    rd_colname = rd_name
                    break

            # If we couldn't find R&D, use industry average as percentage of revenue
            if rd_expenses is None:
                if 'Total Revenue' in income_hist.index:
                    # Industry average R&D as % of revenue
                    rd_ratio = {
                        'Pharmaceuticals': 0.15,  # 15% of revenue
                        'Biotechnology': 0.30,  # 30% of revenue
                        'Medical Devices': 0.10,  # 10% of revenue
                        'Healthcare Services': 0.03,  # 3% of revenue
                        'Healthcare Technology': 0.20  # 20% of revenue
                    }.get(subsector, 0.15)

                    # Estimate R&D based on revenue
                    rd_expenses = income_hist.loc['Total Revenue'] * rd_ratio
                else:
                    raise ValueError("Could not determine R&D expenses or revenue")

            # Normalize and capitalize R&D
            num_years = min(5, len(income_hist.columns))
            rd_to_capitalize = pd.Series(0, index=income_hist.columns[:num_years])

            # For the past few years, capitalize a portion of R&D
            for i in range(num_years):
                # Apply capitalization factor
                rd_to_capitalize[i] = rd_expenses[i] * self.rd_capitalization_factor

            # Adjust EBIT and EBITDA for R&D capitalization
            # Get base EBIT
            ebit = None

            for ebit_name in ['EBIT', 'Operating Income']:
                if ebit_name in income_hist.index:
                    ebit = income_hist.loc[ebit_name].copy()
                    break

            if ebit is None:
                # Calculate EBIT from components
                if 'Net Income' in income_hist.index and 'Interest Expense' in income_hist.index:
                    net_income = income_hist.loc['Net Income']
                    interest_expense = income_hist.loc['Interest Expense']

                    # Estimate taxes based on effective tax rate
                    if 'Income Tax Expense' in income_hist.index:
                        tax_expense = income_hist.loc['Income Tax Expense']
                    else:
                        # Assume 25% tax rate
                        tax_expense = net_income * 0.25 / 0.75

                    ebit = net_income + interest_expense + tax_expense
                else:
                    raise ValueError("Could not determine or calculate EBIT")

            # Adjust EBIT for R&D capitalization
            adjusted_ebit = ebit.copy()

            for i in range(num_years):
                adjusted_ebit[i] += rd_to_capitalize[i]

            # Get most recent data
            latest_adjusted_ebit = adjusted_ebit[0]

            # Get DCF parameters
            forecast_years = self.sector_params.get('forecast_years', 7)
            terminal_growth = self.sector_params.get('terminal_growth_rate', 0.03)

            # Calculate discount rate based on company risk profile
            discount_rate = self._calculate_healthcare_discount_rate(ticker, financial_data, subsector)

            # Estimate growth rate based on historical data and subsector
            growth_rate = self._estimate_healthcare_growth_rate(ticker, financial_data, subsector)

            # Forecast adjusted EBIT
            forecast_ebit = []
            current_ebit = latest_adjusted_ebit

            for year in range(1, forecast_years + 1):
                # Apply declining growth rate for later years
                year_growth = max(growth_rate * (1 - 0.05 * (year - 1)), terminal_growth)
                current_ebit *= (1 + year_growth)
                forecast_ebit.append(current_ebit)

            # Adjust for tax
            tax_rate = 0.21  # Corporate tax rate
            forecast_nopat = [ebit * (1 - tax_rate) for ebit in forecast_ebit]

            # Estimate capital expenditures (including ongoing R&D)
            # In healthcare, significant portion of capex is actually R&D
            forecast_capex = []

            # Use latest R&D as base
            latest_rd = rd_expenses[0]

            for year in range(1, forecast_years + 1):
                # Assume R&D grows with revenue but at a slightly slower pace
                rd_growth = max(growth_rate * 0.9, terminal_growth)
                rd_expense = latest_rd * (1 + rd_growth) ** year

                # Traditional capex is typically lower in healthcare than other sectors
                traditional_capex_ratio = {
                    'Pharmaceuticals': 0.07,  # 7% of revenue
                    'Biotechnology': 0.05,  # 5% of revenue
                    'Medical Devices': 0.08,  # 8% of revenue
                    'Healthcare Services': 0.10,  # 10% of revenue
                    'Healthcare Technology': 0.06  # 6% of revenue
                }.get(subsector, 0.07)

                # Estimate revenue
                if 'Total Revenue' in income_hist.index:
                    latest_revenue = income_hist.loc['Total Revenue'][0]
                    forecast_revenue = latest_revenue * (1 + growth_rate) ** year
                    traditional_capex = forecast_revenue * traditional_capex_ratio
                else:
                    # Estimate based on EBIT
                    traditional_capex = forecast_ebit[year - 1] * 0.3

                # Total capex is sum of capitalized R&D and traditional capex
                total_capex = (rd_expense * self.rd_capitalization_factor) + traditional_capex
                forecast_capex.append(total_capex)

            # Estimate depreciation (including amortized R&D)
            forecast_depreciation = []

            # Get current depreciation
            depreciation = None

            for dep_name in ['Depreciation & Amortization', 'Depreciation', 'Depreciation and Amortization']:
                if dep_name in income_hist.index:
                    depreciation = income_hist.loc[dep_name][0]
                    break

            if depreciation is None:
                # Estimate based on assets
                if 'Total Assets' in balance_sheet.index:
                    total_assets = balance_sheet.loc['Total Assets'][0]
                    depreciation = total_assets * 0.05  # Assume 5% depreciation rate
                else:
                    # Rough estimate based on EBIT
                    depreciation = latest_adjusted_ebit * 0.15

            for year in range(1, forecast_years + 1):
                # Depreciation grows with capex
                year_depreciation = depreciation * (1 + 0.05 * year)
                forecast_depreciation.append(year_depreciation)

            # Estimate changes in working capital
            forecast_wc_change = []

            # Working capital requirement as percentage of revenue
            wc_ratio = {
                'Pharmaceuticals': 0.15,  # 15% of revenue
                'Biotechnology': 0.10,  # 10% of revenue
                'Medical Devices': 0.18,  # 18% of revenue
                'Healthcare Services': 0.08,  # 8% of revenue
                'Healthcare Technology': 0.12  # 12% of revenue
            }.get(subsector, 0.15)

            if 'Total Revenue' in income_hist.index:
                latest_revenue = income_hist.loc['Total Revenue'][0]
                prev_revenue = latest_revenue

                for year in range(1, forecast_years + 1):
                    forecast_revenue = latest_revenue * (1 + growth_rate) ** year
                    wc_change = (forecast_revenue - prev_revenue) * wc_ratio
                    forecast_wc_change.append(wc_change)
                    prev_revenue = forecast_revenue
            else:
                # Rough estimate based on EBIT growth
                for year in range(1, forecast_years + 1):
                    wc_change = forecast_ebit[year - 1] * 0.1
                    forecast_wc_change.append(wc_change)

            # Calculate free cash flow
            forecast_fcf = []

            for year in range(forecast_years):
                fcf = forecast_nopat[year] + forecast_depreciation[year] - forecast_capex[year] - forecast_wc_change[
                    year]
                forecast_fcf.append(fcf)

            # Calculate terminal value
            terminal_fcf = forecast_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Discount cash flows
            pv_fcf = sum(fcf / (1 + discount_rate) ** (year + 1) for year, fcf in enumerate(forecast_fcf))
            pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = pv_fcf + pv_terminal

            # Calculate equity value
            total_debt = 0

            for debt_name in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_name in balance_sheet.index:
                    total_debt += balance_sheet.loc[debt_name][0]

            cash = 0

            for cash_name in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                if cash_name in balance_sheet.index:
                    cash += balance_sheet.loc[cash_name][0]

            equity_value = enterprise_value - total_debt + cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'R&D Adjusted DCF',
                'rd_expenses': rd_expenses[0] if isinstance(rd_expenses, pd.Series) else rd_expenses,
                'rd_capitalization_factor': self.rd_capitalization_factor,
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
                'forecast_years': forecast_years,
                'forecast_fcf': forecast_fcf,
                'terminal_value': terminal_value,
                'pv_fcf': pv_fcf,
                'pv_terminal': pv_terminal,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating R&D adjusted DCF for {ticker}: {e}")
            return {
                'method': 'R&D Adjusted DCF',
                'error': str(e),
                'value_per_share': None
            }

    def healthcare_multiples_valuation(self, ticker: str, financial_data: Dict,
                                       subsector: str = 'Pharmaceuticals') -> Dict:
        """
        Perform multiples-based valuation adapted for healthcare companies.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Healthcare subsector

        Returns:
            Dictionary with multiples valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for multiples valuation")

            # Get most recent data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

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

            # Book value
            if 'Total Stockholder Equity' in latest_balance.index:
                metrics['book_value'] = latest_balance.loc['Total Stockholder Equity']

            # R&D expenses (important for healthcare companies)
            for rd_name in ['Research & Development', 'R&D Expenses', 'Research and Development']:
                if rd_name in latest_income.index:
                    metrics['rd_expenses'] = latest_income.loc[rd_name]
                    break

            # Get appropriate multiples for the subsector
            multiples = self.subsector_multiples.get(subsector, self.subsector_multiples['Pharmaceuticals'])

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

            # Revenue-based valuation (important for biotech and early stage companies)
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

            # Book value-based valuation
            if 'book_value' in metrics and metrics['book_value'] > 0:
                valuations['pb'] = {
                    'multiple': multiples['p_b'],
                    'value': metrics['book_value'] * multiples['p_b'],
                    'description': 'Price to Book'
                }

            # R&D-based valuation (for research-intensive companies)
            if 'rd_expenses' in metrics and metrics['rd_expenses'] > 0 and 'revenue' in metrics:
                # Higher multiple for companies with higher R&D intensity
                rd_intensity = metrics['rd_expenses'] / metrics['revenue']
                rd_multiple = multiples['p_s'] * (1 + rd_intensity)

                valuations['rd_based'] = {
                    'multiple': rd_multiple,
                    'value': metrics['revenue'] * rd_multiple,
                    'description': 'R&D Intensity Adjusted Revenue Multiple'
                }

            # Calculate final equity value
            equity_values = []

            # PE and PB already give equity value
            if 'pe' in valuations:
                equity_values.append(valuations['pe']['value'])

            if 'pb' in valuations:
                equity_values.append(valuations['pb']['value'])

            # EV-based metrics need to be adjusted to equity value
            if 'ev_ebitda' in valuations and 'equity_value' in valuations['ev_ebitda']:
                equity_values.append(valuations['ev_ebitda']['equity_value'])

            if 'ev_revenue' in valuations and 'equity_value' in valuations['ev_revenue']:
                equity_values.append(valuations['ev_revenue']['equity_value'])

            if 'rd_based' in valuations:
                equity_values.append(valuations['rd_based']['value'])

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
                'method': 'Healthcare Multiples',
                'subsector': subsector,
                'metrics': metrics,
                'valuations': valuations,
                'equity_value': avg_equity_value,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating healthcare multiples valuation for {ticker}: {e}")
            return {
                'method': 'Healthcare Multiples',
                'subsector': subsector,
                'error': str(e),
                'value_per_share': None
            }

    def pipeline_valuation(self, ticker: str, financial_data: Dict,
                           pipeline_data: Dict) -> Dict:
        """
        Value pharmaceutical/biotech pipeline using risk-adjusted NPV approach.

        This method evaluates drugs/products in development pipeline by:
        1. Estimating peak sales for each product
        2. Applying success probabilities based on development stage
        3. Calculating NPV of each product's future cash flows
        4. Summing risk-adjusted values across the pipeline

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            pipeline_data: Drug/product pipeline data dictionary

        Returns:
            Dictionary with pipeline valuation results
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Pipeline data should be a dictionary of products/candidates
            # Each with details on development stage, target market, estimated launch, etc.
            if not pipeline_data or not isinstance(pipeline_data, dict):
                raise ValueError("Valid pipeline data required for pipeline valuation")

            # Calculate value for each pipeline product
            product_values = {}
            total_pipeline_value = 0

            for product_name, product_info in pipeline_data.items():
                # Get product details
                stage = product_info.get('stage', 'Phase1')  # Development stage
                indication = product_info.get('indication', '')  # Treatment/disease area
                peak_sales = product_info.get('peak_sales')  # Estimated peak annual sales
                launch_year = product_info.get('launch_year', datetime.now().year + 5)  # Estimated launch year
                patent_expiry = product_info.get('patent_expiry', launch_year + 12)  # Patent expiry

                # If peak sales not provided, use market size estimate
                if not peak_sales:
                    market_size = product_info.get('market_size', 1000)  # Market size in millions
                    market_share = product_info.get('market_share', 0.1)  # Expected market share (10%)
                    peak_sales = market_size * market_share

                # Get success probability based on stage
                success_probability = self.rd_success_rates.get(stage, 0.1)

                # Apply therapeutic area adjustment
                # Some areas have higher/lower success rates
                if 'oncology' in indication.lower():
                    success_probability *= 0.8  # Lower for oncology
                elif 'rare' in indication.lower() or 'orphan' in indication.lower():
                    success_probability *= 1.2  # Higher for rare diseases

                # Cap probability at 95%
                success_probability = min(success_probability, 0.95)

                # Calculate cash flow projection
                current_year = datetime.now().year
                years_to_launch = max(0, launch_year - current_year)
                patent_life = max(0, patent_expiry - launch_year)

                # Discount rate (higher for earlier stage products)
                base_discount_rate = 0.12  # 12% base
                stage_adjustment = {
                    'Preclinical': 0.08,  # +8% for preclinical
                    'Phase1': 0.06,  # +6% for Phase 1
                    'Phase2': 0.04,  # +4% for Phase 2
                    'Phase3': 0.02,  # +2% for Phase 3
                    'Filed': 0.01  # +1% for Filed
                }.get(stage, 0.04)

                discount_rate = base_discount_rate + stage_adjustment

                # Ramp up period to peak sales (years)
                ramp_up = min(4, patent_life // 3)

                # Operating margin for pharmaceutical products
                operating_margin = product_info.get('operating_margin', 0.65)  # 65% margin

                # Cash flow projection (simplified model)
                cash_flows = []

                # Pre-launch: R&D investments (negative cash flow)
                for year in range(years_to_launch):
                    # R&D expense higher in later stages
                    rd_expense = peak_sales * {
                        'Preclinical': -0.02,
                        'Phase1': -0.05,
                        'Phase2': -0.10,
                        'Phase3': -0.15,
                        'Filed': -0.05
                    }.get(stage, -0.10)

                    cash_flows.append(rd_expense)

                # Post-launch: Revenue and profit
                for year in range(patent_life):
                    if year < ramp_up:
                        # Ramp up period
                        sales = peak_sales * (year + 1) / ramp_up
                    elif year < patent_life - 3:
                        # Peak sales period
                        sales = peak_sales
                    else:
                        # Decline due to patent expiry / competition
                        years_from_expiry = patent_life - year
                        sales = peak_sales * years_from_expiry / 3

                    # Calculate operating profit
                    profit = sales * operating_margin
                    cash_flows.append(profit)

                # Calculate NPV
                npv = 0
                for i, cf in enumerate(cash_flows):
                    npv += cf / ((1 + discount_rate) ** (i + 1))

                # Apply success probability to get risk-adjusted NPV
                risk_adjusted_npv = npv * success_probability

                # Add to total pipeline value
                total_pipeline_value += risk_adjusted_npv

                # Store product valuation
                product_values[product_name] = {
                    'stage': stage,
                    'indication': indication,
                    'peak_sales': peak_sales,
                    'success_probability': success_probability,
                    'npv': npv,
                    'risk_adjusted_npv': risk_adjusted_npv
                }

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = total_pipeline_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'Pipeline Valuation',
                'total_pipeline_value': total_pipeline_value,
                'product_values': product_values,
                'value_per_share': value_per_share
            }

        except Exception as e:
            logger.error(f"Error calculating pipeline valuation for {ticker}: {e}")
            return {
                'method': 'Pipeline Valuation',
                'error': str(e),
                'value_per_share': None
            }

    def _calculate_healthcare_discount_rate(self, ticker: str, financial_data: Dict,
                                            subsector: str) -> float:
        """
        Calculate appropriate discount rate for healthcare company based on
        risk profile and subsector.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Healthcare subsector

        Returns:
            Discount rate as float
        """
        try:
            # Get market data
            market_data = financial_data.get('market_data', {})
            beta = market_data.get('beta')

            # Base discount rate from sector parameters
            base_rate = self.sector_params.get('default_discount_rate', 0.11)

            # Subsector risk premium
            subsector_premium = {
                'Pharmaceuticals': 0.00,  # Base case
                'Biotechnology': 0.04,  # Higher risk
                'Medical Devices': 0.01,  # Slightly higher risk
                'Healthcare Services': -0.01,  # Lower risk
                'Healthcare Technology': 0.02  # Moderate risk
            }.get(subsector, 0.00)

            # Company-specific adjustments
            company_premium = 0.00

            # Beta-based adjustment if available
            if beta is not None:
                # Higher beta = higher risk = higher discount rate
                if beta > 1.5:
                    company_premium += 0.02
                elif beta < 0.8:
                    company_premium -= 0.01

            # Adjust based on profitability
            income_stmt = financial_data.get('income_statement')
            if income_stmt is not None:
                latest_income = income_stmt.iloc[:, 0]

                # Check profitability
                if 'Net Income' in latest_income.index:
                    net_income = latest_income.loc['Net Income']

                    if net_income <= 0:
                        # Unprofitable companies are higher risk
                        company_premium += 0.03

                    # Check profit margin if revenue available
                    if 'Total Revenue' in latest_income.index and latest_income.loc['Total Revenue'] > 0:
                        margin = net_income / latest_income.loc['Total Revenue']

                        if margin < 0.05:
                            company_premium += 0.01
                        elif margin > 0.20:
                            company_premium -= 0.01

            # Adjust based on R&D intensity (higher R&D = higher risk but also potential)
            if income_stmt is not None:
                latest_income = income_stmt.iloc[:, 0]

                rd_expense = None
                for rd_name in ['Research & Development', 'R&D Expenses', 'Research and Development']:
                    if rd_name in latest_income.index:
                        rd_expense = latest_income.loc[rd_name]
                        break

                if rd_expense is not None and 'Total Revenue' in latest_income.index and latest_income.loc[
                    'Total Revenue'] > 0:
                    rd_intensity = rd_expense / latest_income.loc['Total Revenue']

                    if rd_intensity > 0.20:
                        # Very high R&D intensity
                        company_premium += 0.02
                    elif rd_intensity > 0.10:
                        # High R&D intensity
                        company_premium += 0.01

            # Calculate final discount rate
            discount_rate = base_rate + subsector_premium + company_premium

            # Ensure discount rate is in reasonable range
            discount_rate = max(0.08, min(0.20, discount_rate))

            return discount_rate

        except Exception as e:
            logger.warning(f"Error calculating healthcare discount rate for {ticker}: {e}")
            # Return default rate from sector parameters
            return self.sector_params.get('default_discount_rate', 0.11)

    def _estimate_healthcare_growth_rate(self, ticker: str, financial_data: Dict,
                                         subsector: str) -> float:
        """
        Estimate appropriate growth rate for healthcare company based on
        historical data and subsector characteristics.

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Healthcare subsector

        Returns:
            Estimated growth rate as float
        """
        try:
            # Get income statement for historical growth analysis
            income_stmt = financial_data.get('income_statement')

            if income_stmt is None or income_stmt.empty:
                # Use subsector default if no data available
                return {
                    'Pharmaceuticals': 0.05,  # 5% for pharma
                    'Biotechnology': 0.15,  # 15% for biotech
                    'Medical Devices': 0.07,  # 7% for devices
                    'Healthcare Services': 0.05,  # 5% for services
                    'Healthcare Technology': 0.12  # 12% for health tech
                }.get(subsector, 0.06)

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
                        'Pharmaceuticals': 0.05,
                        'Biotechnology': 0.15,
                        'Medical Devices': 0.07,
                        'Healthcare Services': 0.05,
                        'Healthcare Technology': 0.12
                    }.get(subsector, 0.06)

                    blended_growth = (weighted_growth * 0.7) + (subsector_growth * 0.3)

                    return max(0.02, min(0.25, blended_growth))  # Ensure reasonable range

            # If we can't calculate from history, use subsector default
            return {
                'Pharmaceuticals': 0.05,
                'Biotechnology': 0.15,
                'Medical Devices': 0.07,
                'Healthcare Services': 0.05,
                'Healthcare Technology': 0.12
            }.get(subsector, 0.06)

        except Exception as e:
            logger.warning(f"Error estimating healthcare growth rate for {ticker}: {e}")
            # Return default rate based on subsector
            return {
                'Pharmaceuticals': 0.05,
                'Biotechnology': 0.15,
                'Medical Devices': 0.07,
                'Healthcare Services': 0.05,
                'Healthcare Technology': 0.12
            }.get(subsector, 0.06)