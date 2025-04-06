import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from pathlib import Path
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from utils.data_loader import DataLoader
from valuation.base_valuation import BaseValuation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dcf_models')

from config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from utils.data_loader import DataLoader
from valuation.base_valuation import BaseValuation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dcf_models')


class AdvancedDCFValuation(BaseValuation):
    """
    Advanced DCF valuation models with sophisticated forecasting techniques,
    scenario analysis, and sector-specific adjustments.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize the advanced DCF valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized AdvancedDCFValuation")

    def multi_stage_dcf_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  sector: str = None) -> Dict[str, Any]:
        """Perform a multi-stage DCF valuation with different growth phases"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

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

            # Determine growth phases based on company profile
            initial_growth_rate = self._estimate_initial_growth_rate(historical_fcf, ticker, sector)
            steady_growth_rate = min(initial_growth_rate * 0.5, params['terminal_growth_rate'] * 1.5)
            terminal_growth = params['terminal_growth_rate']

            # Define growth phases
            high_growth_years = 5
            transition_years = 5

            # Calculate discount rate (WACC)
            discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Starting FCF (most recent)
            last_fcf = historical_fcf.iloc[0]

            # Forecast cash flows for high growth phase
            high_growth_fcf = []
            for year in range(1, high_growth_years + 1):
                fcf = last_fcf * (1 + initial_growth_rate) ** year
                high_growth_fcf.append(fcf)

            # Forecast cash flows for transition phase (declining growth rate)
            transition_fcf = []
            for year in range(1, transition_years + 1):
                # Linear decline in growth rate from initial to steady
                growth_rate = initial_growth_rate - ((initial_growth_rate - steady_growth_rate) *
                                                     year / transition_years)
                fcf = high_growth_fcf[-1] * (1 + growth_rate) ** year
                transition_fcf.append(fcf)

            # Calculate terminal value
            terminal_fcf = transition_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_high_growth = sum(fcf / (1 + discount_rate) ** year
                                            for year, fcf in enumerate(high_growth_fcf, 1))

            present_value_transition = sum(fcf / (1 + discount_rate) ** (year + high_growth_years)
                                           for year, fcf in enumerate(transition_fcf, 1))

            present_value_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

            # Calculate enterprise and equity values
            enterprise_value = present_value_high_growth + present_value_transition + present_value_terminal
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

            # Combine all cash flows for output
            forecast_fcf = high_growth_fcf + transition_fcf

            return {
                'company': ticker,
                'method': 'multi_stage_dcf',
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': discount_rate,
                'initial_growth_rate': initial_growth_rate,
                'steady_growth_rate': steady_growth_rate,
                'terminal_growth': terminal_growth,
                'high_growth_years': high_growth_years,
                'transition_years': transition_years,
                'forecast_fcf': forecast_fcf,
                'terminal_value': terminal_value,
                'present_value_high_growth': present_value_high_growth,
                'present_value_transition': present_value_transition,
                'present_value_terminal': present_value_terminal,
                'historical_fcf': historical_fcf.to_dict(),
                'net_debt': net_debt,
                'safety_margin': params['default_margin_of_safety']
            }

        except Exception as e:
            logger.error(f"Error in multi-stage DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'multi_stage_dcf',
                'enterprise_value': None,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def forecast_driven_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                            sector: str = None) -> Dict[str, Any]:
        """Perform a DCF valuation with detailed forecasting of financial statements"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

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

            # Forecast years
            forecast_years = params['forecast_years']

            # Create forecasts for key financial statement items
            forecast = self._forecast_financial_statements(income_stmt, balance_sheet, cash_flow,
                                                           forecast_years, sector)

            # Extract forecasted FCF
            forecasted_fcf = forecast['fcf']

            # Calculate discount rate (WACC)
            discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Terminal growth rate
            terminal_growth = params['terminal_growth_rate']

            # Calculate terminal value
            terminal_fcf = forecasted_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (year + 1)
                                    for year, fcf in enumerate(forecasted_fcf))

            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise and equity values
            enterprise_value = present_value_fcf + present_value_terminal
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
                'method': 'forecast_driven_dcf',
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'forecast_financials': forecast['summary'],
                'forecast_fcf': forecasted_fcf,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'net_debt': net_debt,
                'safety_margin': params['default_margin_of_safety']
            }

        except Exception as e:
            logger.error(f"Error in forecast-driven DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'forecast_driven_dcf',
                'enterprise_value': None,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def monte_carlo_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                        sector: str = None, simulations: int = 1000) -> Dict[str, Any]:
        """Perform a Monte Carlo simulation for DCF valuation to account for uncertainty"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

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

            # Base parameters
            forecast_years = params['forecast_years']
            base_growth_rate = self._estimate_growth_rate(historical_fcf)
            base_terminal_growth = params['terminal_growth_rate']
            base_discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Starting FCF (most recent)
            last_fcf = historical_fcf.iloc[0]

            # Define parameter ranges for Monte Carlo simulation
            growth_std = base_growth_rate * 0.3  # 30% standard deviation around the base growth rate
            terminal_growth_std = base_terminal_growth * 0.2  # 20% standard deviation
            discount_rate_std = base_discount_rate * 0.1  # 10% standard deviation

            # Run simulations
            values_per_share = []
            enterprise_values = []

            for _ in range(simulations):
                # Randomize parameters for this simulation
                sim_growth_rate = np.random.normal(base_growth_rate, growth_std)
                sim_terminal_growth = np.random.normal(base_terminal_growth, terminal_growth_std)
                sim_discount_rate = np.random.normal(base_discount_rate, discount_rate_std)

                # Ensure parameters are within reasonable bounds
                sim_growth_rate = max(0.01, min(0.30, sim_growth_rate))
                sim_terminal_growth = max(0.01, min(0.05, sim_terminal_growth))
                sim_discount_rate = max(0.05, min(0.20, sim_discount_rate))

                # Ensure discount rate > terminal growth
                if sim_discount_rate <= sim_terminal_growth:
                    sim_discount_rate = sim_terminal_growth + 0.03

                # Forecast future free cash flows
                sim_fcf = []
                for year in range(1, forecast_years + 1):
                    fcf = last_fcf * (1 + sim_growth_rate) ** year
                    sim_fcf.append(fcf)

                # Calculate terminal value
                terminal_fcf = sim_fcf[-1] * (1 + sim_terminal_growth)
                terminal_value = terminal_fcf / (sim_discount_rate - sim_terminal_growth)

                # Calculate present values
                present_value_fcf = sum(fcf / (1 + sim_discount_rate) ** (year + 1)
                                        for year, fcf in enumerate(sim_fcf))

                present_value_terminal = terminal_value / (1 + sim_discount_rate) ** forecast_years

                # Calculate enterprise and equity values
                enterprise_value = present_value_fcf + present_value_terminal
                net_debt = self._calculate_net_debt(balance_sheet)
                equity_value = enterprise_value - net_debt

                enterprise_values.append(enterprise_value)

                # Calculate per share value
                if shares_outstanding is not None and shares_outstanding > 0:
                    value_per_share = equity_value / shares_outstanding
                    values_per_share.append(value_per_share)

            # Calculate statistics from simulations
            if values_per_share:
                mean_value = np.mean(values_per_share)
                median_value = np.median(values_per_share)
                std_dev = np.std(values_per_share)

                # Calculate percentiles for confidence intervals
                percentile_5 = np.percentile(values_per_share, 5)
                percentile_25 = np.percentile(values_per_share, 25)
                percentile_75 = np.percentile(values_per_share, 75)
                percentile_95 = np.percentile(values_per_share, 95)
            else:
                mean_value = None
                median_value = None
                std_dev = None
                percentile_5 = None
                percentile_25 = None
                percentile_75 = None
                percentile_95 = None

            # Calculate enterprise value statistics
            mean_ev = np.mean(enterprise_values) if enterprise_values else None
            median_ev = np.median(enterprise_values) if enterprise_values else None

            return {
                'company': ticker,
                'method': 'monte_carlo_dcf',
                'mean_enterprise_value': mean_ev,
                'median_enterprise_value': median_ev,
                'mean_value_per_share': mean_value,
                'median_value_per_share': median_value,
                'std_deviation': std_dev,
                'percentile_5': percentile_5,
                'percentile_25': percentile_25,
                'percentile_75': percentile_75,
                'percentile_95': percentile_95,
                'base_parameters': {
                    'discount_rate': base_discount_rate,
                    'growth_rate': base_growth_rate,
                    'terminal_growth': base_terminal_growth
                },
                'simulations': simulations,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in Monte Carlo DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'monte_carlo_dcf',
                'mean_value_per_share': None,
                'median_value_per_share': None,
                'error': str(e)
            }

    def scenario_analysis_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                              sector: str = None) -> Dict[str, Any]:
        """Perform DCF valuation with different scenarios (bullish, base, bearish)"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

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

            # Base parameters
            forecast_years = params['forecast_years']
            base_growth_rate = self._estimate_growth_rate(historical_fcf)
            base_terminal_growth = params['terminal_growth_rate']
            base_discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Starting FCF (most recent)
            last_fcf = historical_fcf.iloc[0]

            # Define scenario parameters
            scenarios = {
                'bullish': {
                    'growth_rate': min(base_growth_rate * 1.5, 0.30),  # 50% higher than base, capped at 30%
                    'terminal_growth': min(base_terminal_growth * 1.3, 0.04),  # 30% higher, capped at 4%
                    'discount_rate': max(base_discount_rate * 0.9, 0.06)  # 10% lower, floor at 6%
                },
                'base': {
                    'growth_rate': base_growth_rate,
                    'terminal_growth': base_terminal_growth,
                    'discount_rate': base_discount_rate
                },
                'bearish': {
                    'growth_rate': max(base_growth_rate * 0.6, 0.01),  # 40% lower, floor at 1%
                    'terminal_growth': max(base_terminal_growth * 0.7, 0.01),  # 30% lower, floor at 1%
                    'discount_rate': min(base_discount_rate * 1.1, 0.18)  # 10% higher, cap at 18%
                }
            }

            # Run valuation for each scenario
            scenario_results = {}

            for scenario_name, scenario_params in scenarios.items():
                # Forecast future free cash flows
                scenario_fcf = []
                for year in range(1, forecast_years + 1):
                    fcf = last_fcf * (1 + scenario_params['growth_rate']) ** year
                    scenario_fcf.append(fcf)

                # Calculate terminal value
                terminal_fcf = scenario_fcf[-1] * (1 + scenario_params['terminal_growth'])
                terminal_value = terminal_fcf / (scenario_params['discount_rate'] - scenario_params['terminal_growth'])

                # Calculate present values
                present_value_fcf = sum(fcf / (1 + scenario_params['discount_rate']) ** (year + 1)
                                        for year, fcf in enumerate(scenario_fcf))

                present_value_terminal = terminal_value / (1 + scenario_params['discount_rate']) ** forecast_years

                # Calculate enterprise and equity values
                enterprise_value = present_value_fcf + present_value_terminal
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

                # Store scenario results
                scenario_results[scenario_name] = {
                    'parameters': scenario_params,
                    'enterprise_value': enterprise_value,
                    'equity_value': equity_value,
                    'value_per_share': value_per_share,
                    'forecast_fcf': scenario_fcf,
                    'terminal_value': terminal_value,
                    'present_value_fcf': present_value_fcf,
                    'present_value_terminal': present_value_terminal
                }

            # Apply margin of safety to base case
            if scenario_results['base']['value_per_share'] is not None:
                safety_margin = params['default_margin_of_safety']
                conservative_value = scenario_results['base']['value_per_share'] * (1 - safety_margin)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'scenario_analysis_dcf',
                'scenarios': scenario_results,
                'conservative_value': conservative_value,
                'historical_fcf': historical_fcf.to_dict(),
                'net_debt': self._calculate_net_debt(balance_sheet),
                'safety_margin': params['default_margin_of_safety']
            }

        except Exception as e:
            logger.error(f"Error in scenario analysis DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'scenario_analysis_dcf',
                'scenarios': {},
                'conservative_value': None,
                'error': str(e)
            }

    # Helper methods for DCF calculations

    def _estimate_initial_growth_rate(self, historical_fcf: pd.Series, ticker: str,
                                      sector: str = None) -> float:
        """Estimate initial growth rate based on historical data and sector trends"""
        try:
            # Start with the base growth rate estimate
            base_growth = self._estimate_growth_rate(historical_fcf)

            # Sector-specific growth adjustments
            sector_growth_factors = {
                "Technology": 1.2,  # Technology companies often grow faster
                "Healthcare": 1.1,  # Healthcare has stable growth
                "Financials": 0.9,  # Financial sector typically grows slower
                "Consumer Discretionary": 1.0,  # Average growth
                "Consumer Staples": 0.8,  # Slower but steady growth
                "Energy": 0.9,  # Cyclical growth
                "Industrials": 0.9,  # Moderate growth
                "Materials": 0.9,  # Cyclical growth
                "Real Estate": 0.8,  # Slower growth
                "Communication Services": 1.1,  # Good growth potential
                "Utilities": 0.7  # Slow but very stable growth
            }

            # Adjust growth rate based on sector
            if sector and sector in sector_growth_factors:
                adjusted_growth = base_growth * sector_growth_factors[sector]
            else:
                adjusted_growth = base_growth

            # Cap growth rate to reasonable range
            return max(0.01, min(0.30, adjusted_growth))

        except Exception as e:
            logger.error(f"Error estimating initial growth rate: {e}")
            return 0.05  # Default 5% growth

    def _get_growth_assumption(self, financial_data: pd.DataFrame, metric: str,
                               sector: str = None) -> float:
        """Calculate growth assumptions for financial forecasting"""
        try:
            if metric not in financial_data.index:
                # Default growth assumptions if data not available
                default_growth = 0.05  # 5% general growth

                # Sector-specific default growth rates
                sector_growth = {
                    "Technology": 0.10,
                    "Healthcare": 0.07,
                    "Financials": 0.04,
                    "Consumer Discretionary": 0.05,
                    "Consumer Staples": 0.03,
                    "Energy": 0.03,
                    "Industrials": 0.04,
                    "Materials": 0.03,
                    "Real Estate": 0.03,
                    "Communication Services": 0.06,
                    "Utilities": 0.02
                }

                return sector_growth.get(sector, default_growth)

            # Get the number of periods for analysis
            periods = min(financial_data.shape[1], 5)  # Use up to 5 years of data

            if periods < 2:
                # Not enough data for historical calculation
                default_growth = 0.05
                return default_growth

            # Calculate year-over-year growth rates
            growth_rates = []

            for i in range(periods - 1):
                current_value = financial_data.iloc[:, i][metric]
                prev_value = financial_data.iloc[:, i + 1][metric]

                if prev_value > 0 and current_value > 0:
                    yoy_growth = (current_value / prev_value) - 1
                    growth_rates.append(yoy_growth)

            if not growth_rates:
                # Default if cannot calculate from history
                default_growth = 0.05
                return default_growth

            # Use weighted average, giving more weight to recent periods
            weights = list(range(1, len(growth_rates) + 1))
            weighted_growth = sum(r * w for r, w in zip(growth_rates, weights)) / sum(weights)

            # Apply sector adjustments if provided
            if sector:
                sector_modifiers = {
                    "Technology": 1.2,
                    "Healthcare": 1.1,
                    "Financials": 0.9,
                    "Consumer Discretionary": 1.0,
                    "Consumer Staples": 0.8,
                    "Energy": 0.9,
                    "Industrials": 0.9,
                    "Materials": 0.9,
                    "Real Estate": 0.8,
                    "Communication Services": 1.1,
                    "Utilities": 0.7
                }

                modifier = sector_modifiers.get(sector, 1.0)
                weighted_growth *= modifier

            # Cap growth to reasonable range
            max_growth = 0.30  # 30% cap on growth
            min_growth = -0.10  # -10% floor on growth

            return max(min_growth, min(max_growth, weighted_growth))

        except Exception as e:
            logger.error(f"Error calculating growth assumption for {metric}: {e}")
            return 0.03  # Conservative default on error

    def _calculate_financial_ratios(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame,
                                    cash_flow: pd.DataFrame) -> Dict[str, float]:
        """Calculate key financial ratios from historical statements for forecasting"""
        ratios = {}

        try:
            # Use historical data to calculate ratios (use averages for stability)
            # Get how many periods we have data for
            income_periods = min(income_stmt.shape[1], 3)  # Use up to 3 years of data
            balance_periods = min(balance_sheet.shape[1], 3)
            cash_flow_periods = min(cash_flow.shape[1], 3)

            # Profitability ratios
            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gross_margins = []
                for i in range(income_periods):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        gross_margins.append(income_stmt.iloc[:, i]['Gross Profit'] / income_stmt.iloc[:, i]['Total Revenue'])

                if gross_margins:
                    ratios['Gross Margin'] = sum(gross_margins) / len(gross_margins)

            # R&D to Revenue ratio
            if 'Research and Development' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                rd_ratios = []
                for i in range(income_periods):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        rd_ratios.append(income_stmt.iloc[:, i]['Research and Development'] / income_stmt.iloc[:, i]['Total Revenue'])

                if rd_ratios:
                    ratios['R&D Ratio'] = sum(rd_ratios) / len(rd_ratios)

            # SG&A to Revenue ratio
            if 'SG&A Expense' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                sga_ratios = []
                for i in range(income_periods):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        sga_ratios.append(income_stmt.iloc[:, i]['SG&A Expense'] / income_stmt.iloc[:, i]['Total Revenue'])

                if sga_ratios:
                    ratios['SG&A Ratio'] = sum(sga_ratios) / len(sga_ratios)

            # Other OpEx to Revenue ratio
            if 'Other Operating Expenses' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                other_opex_ratios = []
                for i in range(income_periods):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        other_opex_ratios.append(income_stmt.iloc[:, i]['Other Operating Expenses'] / income_stmt.iloc[:, i]['Total Revenue'])

                if other_opex_ratios:
                    ratios['Other OpEx Ratio'] = sum(other_opex_ratios) / len(other_opex_ratios)

            # Effective Tax Rate
            if 'Income Tax Expense' in income_stmt.index and 'Income Before Tax' in income_stmt.index:
                tax_rates = []
                for i in range(income_periods):
                    if income_stmt.iloc[:, i]['Income Before Tax'] > 0:
                        tax_rates.append(income_stmt.iloc[:, i]['Income Tax Expense'] / income_stmt.iloc[:, i]['Income Before Tax'])

                if tax_rates:
                    avg_tax_rate = sum(tax_rates) / len(tax_rates)
                    ratios['Effective Tax Rate'] = min(0.4, max(0.15, avg_tax_rate))  # Cap between 15% and 40%

            # Asset-based ratios

            # Cash to Revenue
            if 'Cash and Cash Equivalents' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                cash_ratios = []
                for i in range(min(balance_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        cash_ratios.append(balance_sheet.iloc[:, i]['Cash and Cash Equivalents'] / income_stmt.iloc[:, i]['Total Revenue'])

                if cash_ratios:
                    ratios['Cash to Revenue'] = sum(cash_ratios) / len(cash_ratios)

            # Receivables to Revenue
            if 'Net Receivables' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                receivables_ratios = []
                for i in range(min(balance_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        receivables_ratios.append(balance_sheet.iloc[:, i]['Net Receivables'] / income_stmt.iloc[:, i]['Total Revenue'])

                if receivables_ratios:
                    ratios['Receivables to Revenue'] = sum(receivables_ratios) / len(receivables_ratios)

            # Inventory to COGS
            if 'Inventory' in balance_sheet.index and 'Cost of Revenue' in income_stmt.index:
                inventory_ratios = []
                for i in range(min(balance_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Cost of Revenue'] > 0:
                        inventory_ratios.append(balance_sheet.iloc[:, i]['Inventory'] / income_stmt.iloc[:, i]['Cost of Revenue'])

                if inventory_ratios:
                    ratios['Inventory to COGS'] = sum(inventory_ratios) / len(inventory_ratios)

            # Payables to COGS
            if 'Accounts Payable' in balance_sheet.index and 'Cost of Revenue' in income_stmt.index:
                payables_ratios = []
                for i in range(min(balance_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Cost of Revenue'] > 0:
                        payables_ratios.append(balance_sheet.iloc[:, i]['Accounts Payable'] / income_stmt.iloc[:, i]['Cost of Revenue'])

                if payables_ratios:
                    ratios['Payables to COGS'] = sum(payables_ratios) / len(payables_ratios)

            # CapEx to Revenue
            if 'Capital Expenditure' in cash_flow.index and 'Total Revenue' in income_stmt.index:
                capex_ratios = []
                for i in range(min(cash_flow_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Total Revenue'] > 0:
                        # Capital Expenditure is typically negative in cash flow statements
                        capex_ratios.append(abs(cash_flow.iloc[:, i]['Capital Expenditure']) / income_stmt.iloc[:, i]['Total Revenue'])

                if capex_ratios:
                    ratios['CapEx to Revenue'] = sum(capex_ratios) / len(capex_ratios)

            # Depreciation Rate
            if 'Depreciation & Amortization' in income_stmt.index and 'Property Plant and Equipment' in balance_sheet.index:
                depreciation_rates = []
                for i in range(min(income_periods, balance_periods)):
                    if balance_sheet.iloc[:, i]['Property Plant and Equipment'] > 0:
                        depreciation_rates.append(income_stmt.iloc[:, i]['Depreciation & Amortization'] /
                                                balance_sheet.iloc[:, i]['Property Plant and Equipment'])

                if depreciation_rates:
                    avg_rate = sum(depreciation_rates) / len(depreciation_rates)
                    ratios['Depreciation Rate'] = min(0.2, max(0.05, avg_rate))  # Cap between 5% and 20%

            # Interest Rate
            if 'Interest Expense' in income_stmt.index and 'Total Debt' in balance_sheet.index:
                interest_rates = []
                for i in range(min(income_periods, balance_periods)):
                    if balance_sheet.iloc[:, i]['Total Debt'] > 0:
                        interest_rates.append(abs(income_stmt.iloc[:, i]['Interest Expense']) /
                                            balance_sheet.iloc[:, i]['Total Debt'])

                if interest_rates:
                    avg_rate = sum(interest_rates) / len(interest_rates)
                    ratios['Interest Rate'] = min(0.1, max(0.02, avg_rate))  # Cap between 2% and 10%

            # Debt Growth Rate
            if 'Total Debt' in balance_sheet.index and balance_periods > 1:
                debt_growth_rates = []
                for i in range(balance_periods - 1):
                    if balance_sheet.iloc[:, i + 1]['Total Debt'] > 0:
                        growth_rate = (balance_sheet.iloc[:, i]['Total Debt'] /
                                    balance_sheet.iloc[:, i + 1]['Total Debt']) - 1
                        debt_growth_rates.append(growth_rate)

                if debt_growth_rates:
                    avg_rate = sum(debt_growth_rates) / len(debt_growth_rates)
                    ratios['Debt Growth Rate'] = min(0.2, max(-0.1, avg_rate))  # Cap between -10% and 20%

            # Dividend Payout Ratio
            if 'Dividends Paid' in cash_flow.index and 'Net Income' in income_stmt.index:
                payout_ratios = []
                for i in range(min(cash_flow_periods, income_periods)):
                    if income_stmt.iloc[:, i]['Net Income'] > 0:
                        payout_ratios.append(abs(cash_flow.iloc[:, i]['Dividends Paid']) /
                                           income_stmt.iloc[:, i]['Net Income'])

                if payout_ratios:
                    avg_ratio = sum(payout_ratios) / len(payout_ratios)
                    ratios['Dividend Payout Ratio'] = min(1.0, max(0.0, avg_ratio))  # Cap between 0% and 100%

            return ratios

        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return ratios

    def _forecast_financial_statements(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame,
                                       cash_flow: pd.DataFrame, forecast_years: int,
                                       sector: str = None) -> Dict[str, Any]:
        """Create detailed forecasts of financial statements"""
        try:
            # Extract the most recent financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0]

            # Calculate historical financial ratios and trends
            financial_ratios = self._calculate_financial_ratios(income_stmt, balance_sheet, cash_flow)

            # Get growth assumptions
            revenue_growth = self._get_growth_assumption(income_stmt, 'Total Revenue', sector)

            # Create forecast dataframes
            forecast_income = pd.DataFrame(index=income_stmt.index,
                                           columns=[f'Year+{i + 1}' for i in range(forecast_years)])

            forecast_balance = pd.DataFrame(index=balance_sheet.index,
                                            columns=[f'Year+{i + 1}' for i in range(forecast_years)])

            forecast_cash_flow = pd.DataFrame(index=cash_flow.index,
                                              columns=[f'Year+{i + 1}' for i in range(forecast_years)])

            # Forecast income statement
            for year in range(forecast_years):
                year_col = f'Year+{year + 1}'

                # Revenue forecast
                if 'Total Revenue' in latest_income.index:
                    if year == 0:
                        # First year based on historical growth
                        forecast_income.loc['Total Revenue', year_col] = latest_income['Total Revenue'] * (
                                    1 + revenue_growth)
                    else:
                        # Subsequent years with potentially declining growth
                        growth_factor = revenue_growth * (1 - year * 0.1)  # Gradual decline in growth
                        growth_factor = max(revenue_growth * 0.5, growth_factor)  # Floor at 50% of initial
                        prev_revenue = forecast_income.loc['Total Revenue', f'Year+{year}']
                        forecast_income.loc['Total Revenue', year_col] = prev_revenue * (1 + growth_factor)

                # Cost of Revenue based on gross margin
                if 'Total Revenue' in forecast_income.index and 'Cost of Revenue' in latest_income.index:
                    if 'Gross Margin' in financial_ratios:
                        gross_margin = financial_ratios['Gross Margin']
                        forecast_revenue = forecast_income.loc['Total Revenue', year_col]
                        forecast_income.loc['Cost of Revenue', year_col] = forecast_revenue * (1 - gross_margin)

                # Operating expenses based on historical ratios
                if 'Total Revenue' in forecast_income.index:
                    forecast_revenue = forecast_income.loc['Total Revenue', year_col]

                    # Research and Development
                    if 'R&D Ratio' in financial_ratios and 'Research and Development' in latest_income.index:
                        rd_ratio = financial_ratios['R&D Ratio']
                        forecast_income.loc['Research and Development', year_col] = forecast_revenue * rd_ratio

                    # SG&A (Sales, General & Administrative)
                    if 'SG&A Ratio' in financial_ratios and 'SG&A Expense' in latest_income.index:
                        sga_ratio = financial_ratios['SG&A Ratio']
                        forecast_income.loc['SG&A Expense', year_col] = forecast_revenue * sga_ratio

                    # Other Operating Expenses
                    if 'Other OpEx Ratio' in financial_ratios and 'Other Operating Expenses' in latest_income.index:
                        other_opex_ratio = financial_ratios['Other OpEx Ratio']
                        forecast_income.loc['Other Operating Expenses', year_col] = forecast_revenue * other_opex_ratio

                # Calculate operating income
                operating_expense_items = [
                    'Cost of Revenue', 'Research and Development',
                    'SG&A Expense', 'Other Operating Expenses'
                ]

                if 'Total Revenue' in forecast_income.index:
                    forecast_revenue = forecast_income.loc['Total Revenue', year_col]
                    total_expenses = 0

                    for expense_item in operating_expense_items:
                        if expense_item in forecast_income.index and not pd.isna(
                                forecast_income.loc[expense_item, year_col]):
                            total_expenses += forecast_income.loc[expense_item, year_col]

                    forecast_income.loc['Operating Income', year_col] = forecast_revenue - total_expenses

                # Interest Expense based on debt levels
                if 'Interest Expense' in latest_income.index and 'Total Debt' in latest_balance.index:
                    # Assume debt grows with revenue but at a slower rate
                    if 'Debt Growth Rate' in financial_ratios:
                        debt_growth = financial_ratios['Debt Growth Rate']
                    else:
                        debt_growth = revenue_growth * 0.7  # 70% of revenue growth

                    if 'Interest Rate' in financial_ratios:
                        interest_rate = financial_ratios['Interest Rate']
                    else:
                        # Estimate interest rate from historical data
                        if latest_balance['Total Debt'] > 0:
                            interest_rate = abs(latest_income['Interest Expense']) / latest_balance['Total Debt']
                        else:
                            interest_rate = 0.05  # Default 5%

                    if year == 0:
                        forecast_debt = latest_balance['Total Debt'] * (1 + debt_growth)
                    else:
                        prev_debt = forecast_balance.loc[
                            'Total Debt', f'Year+{year}'] if 'Total Debt' in forecast_balance.index else latest_balance[
                                                                                                             'Total Debt'] * (
                                                                                                                     1 + debt_growth) ** year
                        forecast_debt = prev_debt * (1 + debt_growth)

                    # Store debt for balance sheet forecast
                    forecast_balance.loc['Total Debt', year_col] = forecast_debt

                    # Calculate interest expense
                    forecast_income.loc['Interest Expense', year_col] = -forecast_debt * interest_rate

                # Income before tax
                if 'Operating Income' in forecast_income.index and 'Interest Expense' in forecast_income.index:
                    op_income = forecast_income.loc['Operating Income', year_col]
                    int_expense = forecast_income.loc['Interest Expense', year_col]
                    forecast_income.loc[
                        'Income Before Tax', year_col] = op_income + int_expense  # int_expense is negative

                # Income Tax
                if 'Income Before Tax' in forecast_income.index:
                    if 'Effective Tax Rate' in financial_ratios:
                        tax_rate = financial_ratios['Effective Tax Rate']
                    else:
                        tax_rate = 0.25  # Default 25% tax rate

                    pre_tax_income = forecast_income.loc['Income Before Tax', year_col]
                    forecast_income.loc[
                        'Income Tax Expense', year_col] = pre_tax_income * tax_rate if pre_tax_income > 0 else 0

                # Net Income
                if 'Income Before Tax' in forecast_income.index and 'Income Tax Expense' in forecast_income.index:
                    pre_tax = forecast_income.loc['Income Before Tax', year_col]
                    tax = forecast_income.loc['Income Tax Expense', year_col]
                    forecast_income.loc['Net Income', year_col] = pre_tax - tax

            # Forecast balance sheet
            for year in range(forecast_years):
                year_col = f'Year+{year + 1}'

                # Assets
                # Cash and Equivalents
                if 'Cash and Cash Equivalents' in latest_balance.index:
                    if 'Cash to Revenue' in financial_ratios:
                        cash_ratio = financial_ratios['Cash to Revenue']
                    else:
                        cash_ratio = latest_balance['Cash and Cash Equivalents'] / latest_income[
                            'Total Revenue'] if 'Total Revenue' in latest_income.index and latest_income[
                            'Total Revenue'] > 0 else 0.1

                    forecast_revenue = forecast_income.loc[
                        'Total Revenue', year_col] if 'Total Revenue' in forecast_income.index else 0
                    forecast_balance.loc['Cash and Cash Equivalents', year_col] = forecast_revenue * cash_ratio

                # Accounts Receivable
                if 'Net Receivables' in latest_balance.index and 'Total Revenue' in forecast_income.index:
                    if 'Receivables to Revenue' in financial_ratios:
                        receivables_ratio = financial_ratios['Receivables to Revenue']
                    else:
                        receivables_ratio = latest_balance['Net Receivables'] / latest_income[
                            'Total Revenue'] if 'Total Revenue' in latest_income.index and latest_income[
                            'Total Revenue'] > 0 else 0.15

                    forecast_revenue = forecast_income.loc['Total Revenue', year_col]
                    forecast_balance.loc['Net Receivables', year_col] = forecast_revenue * receivables_ratio

                # Inventory
                if 'Inventory' in latest_balance.index and 'Cost of Revenue' in forecast_income.index:
                    if 'Inventory to COGS' in financial_ratios:
                        inventory_ratio = financial_ratios['Inventory to COGS']
                    else:
                        inventory_ratio = latest_balance['Inventory'] / latest_income[
                            'Cost of Revenue'] if 'Cost of Revenue' in latest_income.index and latest_income[
                            'Cost of Revenue'] > 0 else 0.2

                    forecast_cogs = forecast_income.loc['Cost of Revenue', year_col]
                    forecast_balance.loc['Inventory', year_col] = forecast_cogs * inventory_ratio

                # Property, Plant & Equipment (PP&E)
                if 'Property Plant and Equipment' in latest_balance.index:
                    if 'CapEx to Revenue' in financial_ratios:
                        capex_ratio = financial_ratios['CapEx to Revenue']
                    else:
                        # Estimate from historical data
                        if 'Capital Expenditure' in latest_cash_flow.index and 'Total Revenue' in latest_income.index:
                            capex_ratio = abs(latest_cash_flow['Capital Expenditure']) / latest_income['Total Revenue']
                        else:
                            capex_ratio = 0.05  # Default 5% of revenue

                    if 'Depreciation Rate' in financial_ratios:
                        depreciation_rate = financial_ratios['Depreciation Rate']
                    else:
                        if 'Depreciation & Amortization' in latest_income.index and 'Property Plant and Equipment' in latest_balance.index:
                            depreciation_rate = latest_income['Depreciation & Amortization'] / latest_balance[
                                'Property Plant and Equipment']
                        else:
                            depreciation_rate = 0.1  # Default 10% depreciation

                    forecast_revenue = forecast_income.loc[
                        'Total Revenue', year_col] if 'Total Revenue' in forecast_income.index else 0
                    capex = forecast_revenue * capex_ratio

                    if year == 0:
                        prev_ppe = latest_balance['Property Plant and Equipment']
                    else:
                        prev_ppe = forecast_balance.loc['Property Plant and Equipment', f'Year+{year}']

                    depreciation = prev_ppe * depreciation_rate
                    forecast_balance.loc['Property Plant and Equipment', year_col] = prev_ppe + capex - depreciation

                    # Store depreciation for cash flow forecast
                    forecast_income.loc['Depreciation & Amortization', year_col] = depreciation

                # Total Assets
                asset_items = [
                    'Cash and Cash Equivalents', 'Net Receivables', 'Inventory',
                    'Property Plant and Equipment', 'Intangible Assets', 'Goodwill',
                    'Other Assets'
                ]

                total_assets = 0
                for asset_item in asset_items:
                    if asset_item in forecast_balance.index and not pd.isna(forecast_balance.loc[asset_item, year_col]):
                        total_assets += forecast_balance.loc[asset_item, year_col]
                    elif asset_item in latest_balance.index:
                        # If not forecasted, assume it grows with revenue
                        if 'Total Revenue' in forecast_income.index and 'Total Revenue' in latest_income.index:
                            growth_factor = forecast_income.loc['Total Revenue', year_col] / latest_income[
                                'Total Revenue']
                            forecast_balance.loc[asset_item, year_col] = latest_balance[asset_item] * growth_factor
                            total_assets += forecast_balance.loc[asset_item, year_col]

                forecast_balance.loc['Total Assets', year_col] = total_assets

                # Liabilities
                # Accounts Payable
                if 'Accounts Payable' in latest_balance.index and 'Cost of Revenue' in forecast_income.index:
                    if 'Payables to COGS' in financial_ratios:
                        payables_ratio = financial_ratios['Payables to COGS']
                    else:
                        payables_ratio = latest_balance['Accounts Payable'] / latest_income[
                            'Cost of Revenue'] if 'Cost of Revenue' in latest_income.index and latest_income[
                            'Cost of Revenue'] > 0 else 0.15

                    forecast_cogs = forecast_income.loc['Cost of Revenue', year_col]
                    forecast_balance.loc['Accounts Payable', year_col] = forecast_cogs * payables_ratio

                # Total Liabilities
                liability_items = [
                    'Accounts Payable', 'Short Term Debt', 'Total Debt',
                    'Other Current Liabilities', 'Other Liabilities'
                ]

                total_liabilities = 0
                for liability_item in liability_items:
                    if liability_item in forecast_balance.index and not pd.isna(
                            forecast_balance.loc[liability_item, year_col]):
                        total_liabilities += forecast_balance.loc[liability_item, year_col]
                    elif liability_item in latest_balance.index:
                        # If not forecasted, assume it grows with revenue
                        if 'Total Revenue' in forecast_income.index and 'Total Revenue' in latest_income.index:
                            growth_factor = forecast_income.loc['Total Revenue', year_col] / latest_income[
                                'Total Revenue']
                            forecast_balance.loc[liability_item, year_col] = latest_balance[
                                                                                 liability_item] * growth_factor
                            total_liabilities += forecast_balance.loc[liability_item, year_col]

                forecast_balance.loc['Total Liabilities', year_col] = total_liabilities

                # Stockholder Equity
                if 'Total Stockholder Equity' in latest_balance.index:
                    if year == 0:
                        prev_equity = latest_balance['Total Stockholder Equity']
                    else:
                        prev_equity = forecast_balance.loc['Total Stockholder Equity', f'Year+{year}']

                    net_income = forecast_income.loc[
                        'Net Income', year_col] if 'Net Income' in forecast_income.index else 0

                    # Estimate dividends paid
                    if 'Dividend Payout Ratio' in financial_ratios:
                        payout_ratio = financial_ratios['Dividend Payout Ratio']
                    else:
                        if 'Dividends Paid' in latest_cash_flow.index and 'Net Income' in latest_income.index:
                            payout_ratio = abs(latest_cash_flow['Dividends Paid']) / latest_income['Net Income'] if \
                            latest_income['Net Income'] > 0 else 0
                        else:
                            payout_ratio = 0.3  # Default 30% payout

                    dividends = net_income * payout_ratio if net_income > 0 else 0

                    # Assume no share issuances/buybacks for simplicity
                    forecast_balance.loc['Total Stockholder Equity', year_col] = prev_equity + net_income - dividends

                    # Store dividends for cash flow forecast
                    forecast_cash_flow.loc['Dividends Paid', year_col] = -dividends

            # Forecast cash flow statement
            for year in range(forecast_years):
                year_col = f'Year+{year + 1}'

                # Net Income
                if 'Net Income' in forecast_income.index:
                    forecast_cash_flow.loc['Net Income', year_col] = forecast_income.loc['Net Income', year_col]

                # Depreciation & Amortization
                if 'Depreciation & Amortization' in forecast_income.index:
                    forecast_cash_flow.loc['Depreciation & Amortization', year_col] = forecast_income.loc[
                        'Depreciation & Amortization', year_col]

                # Changes in Working Capital
                working_capital_items = {
                    'Net Receivables': -1,  # Increase in receivables = negative cash flow
                    'Inventory': -1,  # Increase in inventory = negative cash flow
                    'Accounts Payable': 1  # Increase in payables = positive cash flow
                }

                change_in_wc = 0
                for item, direction in working_capital_items.items():
                    if item in forecast_balance.index:
                        if year == 0:
                            prev_value = latest_balance[item] if item in latest_balance.index else 0
                        else:
                            prev_value = forecast_balance.loc[item, f'Year+{year}']

                        current_value = forecast_balance.loc[item, year_col]
                        change = current_value - prev_value
                        change_in_wc += change * direction

                forecast_cash_flow.loc['Change in Working Capital', year_col] = change_in_wc

                # Operating Cash Flow
                ocf_items = ['Net Income', 'Depreciation & Amortization', 'Change in Working Capital']
                operating_cf = 0

                for item in ocf_items:
                    if item in forecast_cash_flow.index and not pd.isna(forecast_cash_flow.loc[item, year_col]):
                        operating_cf += forecast_cash_flow.loc[item, year_col]

                forecast_cash_flow.loc['Operating Cash Flow', year_col] = operating_cf

                # Capital Expenditure
                if 'Property Plant and Equipment' in forecast_balance.index:
                    if year == 0:
                        prev_ppe = latest_balance[
                            'Property Plant and Equipment'] if 'Property Plant and Equipment' in latest_balance.index else 0
                    else:
                        prev_ppe = forecast_balance.loc['Property Plant and Equipment', f'Year+{year}']

                    current_ppe = forecast_balance.loc['Property Plant and Equipment', year_col]
                    depreciation = forecast_income.loc[
                        'Depreciation & Amortization', year_col] if 'Depreciation & Amortization' in forecast_income.index else 0

                    capex = -(current_ppe - prev_ppe + depreciation)  # Negative for cash outflow
                    forecast_cash_flow.loc['Capital Expenditure', year_col] = capex

                # Free Cash Flow
                forecast_cash_flow.loc['Free Cash Flow', year_col] = operating_cf + forecast_cash_flow.loc[
                    'Capital Expenditure', year_col] if 'Capital Expenditure' in forecast_cash_flow.index else operating_cf

            # Extract forecasted FCF
            forecasted_fcf = []
            for year in range(forecast_years):
                year_col = f'Year+{year + 1}'
                if 'Free Cash Flow' in forecast_cash_flow.index:
                    fcf = forecast_cash_flow.loc['Free Cash Flow', year_col]
                    forecasted_fcf.append(fcf)

            # Create a summary of forecasts
            summary = {
                'Income Statement': {
                    'Revenue': [
                        forecast_income.loc['Total Revenue', col] if 'Total Revenue' in forecast_income.index else None
                        for col in forecast_income.columns],
                    'Operating Income': [forecast_income.loc[
                                             'Operating Income', col] if 'Operating Income' in forecast_income.index else None
                                         for col in forecast_income.columns],
                    'Net Income': [
                        forecast_income.loc['Net Income', col] if 'Net Income' in forecast_income.index else None for
                        col in forecast_income.columns]
                },
                'Balance Sheet': {
                    'Total Assets': [
                        forecast_balance.loc['Total Assets', col] if 'Total Assets' in forecast_balance.index else None
                        for col in forecast_balance.columns],
                    'Total Liabilities': [forecast_balance.loc[
                                              'Total Liabilities', col] if 'Total Liabilities' in forecast_balance.index else None
                                          for col in forecast_balance.columns],
                    'Stockholder Equity': [forecast_balance.loc[
                                               'Total Stockholder Equity', col] if 'Total Stockholder Equity' in forecast_balance.index else None
                                           for col in forecast_balance.columns]
                },
                'Cash Flow': {
                    'Operating Cash Flow': [forecast_cash_flow.loc[
                                                'Operating Cash Flow', col] if 'Operating Cash Flow' in forecast_cash_flow.index else None
                                            for col in forecast_cash_flow.columns],
                    'CapEx': [forecast_cash_flow.loc[
                                  'Capital Expenditure', col] if 'Capital Expenditure' in forecast_cash_flow.index else None
                              for col in forecast_cash_flow.columns],
                    'Free Cash Flow': [forecast_cash_flow.loc[
                                           'Free Cash Flow', col] if 'Free Cash Flow' in forecast_cash_flow.index else None
                                       for col in forecast_cash_flow.columns]
                }
            }

            return {
                'income_statement': forecast_income,
                'balance_sheet': forecast_balance,
                'cash_flow': forecast_cash_flow,
                'fcf': forecasted_fcf,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Error in financial forecasting: {e}")
            # Return a minimal forecast with only FCF if full forecast fails
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'fcf': [],
                'summary': {}
            }


class SectorSpecificDCF(AdvancedDCFValuation):
    """
    DCF valuation models with sector-specific adjustments and techniques.
    Each sector may require different approaches to forecasting, risk assessment, and valuation.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize the sector-specific DCF valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized SectorSpecificDCF")

    def get_sector_dcf_model(self, sector: str) -> str:
        """Get the appropriate DCF model type for a given sector"""
        sector_model_mapping = {
            "Technology": "multi_stage_dcf",
            "Healthcare": "multi_stage_dcf",
            "Financials": "dividend_discount",
            "Consumer Discretionary": "forecast_driven_dcf",
            "Consumer Staples": "forecast_driven_dcf",
            "Energy": "scenario_analysis_dcf",
            "Industrials": "forecast_driven_dcf",
            "Materials": "scenario_analysis_dcf",
            "Real Estate": "asset_based_nav",
            "Communication Services": "multi_stage_dcf",
            "Utilities": "dividend_discount"
        }

        return sector_model_mapping.get(sector, "multi_stage_dcf")

    def apply_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                         sector: str = None) -> Dict[str, Any]:
        """Apply the appropriate sector-specific DCF valuation model"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get company info to determine sector if not provided
            if sector is None:
                company_info = financial_data.get('company_info', {})
                sector = company_info.get('sector')

            # If still no sector, use a default
            if not sector:
                logger.warning(f"No sector information available for {ticker}, using default DCF model")
                return self.multi_stage_dcf_valuation(ticker, financial_data)

            # Get the appropriate model for this sector
            model_type = self.get_sector_dcf_model(sector)

            # Apply the selected model
            if model_type == "multi_stage_dcf":
                return self.multi_stage_dcf_valuation(ticker, financial_data, sector)
            elif model_type == "forecast_driven_dcf":
                return self.forecast_driven_dcf(ticker, financial_data, sector)
            elif model_type == "scenario_analysis_dcf":
                return self.scenario_analysis_dcf(ticker, financial_data, sector)
            elif model_type == "dividend_discount":
                # For financials and utilities, dividend discount model works better
                dividend_result = self.dividend_discount_valuation(ticker, financial_data, sector)

                # If the company doesn't pay dividends, fall back to multi-stage DCF
                if 'error' in dividend_result and 'dividend' in dividend_result['error'].lower():
                    logger.info(f"Falling back to multi-stage DCF for {ticker} as dividend model failed")
                    return self.multi_stage_dcf_valuation(ticker, financial_data, sector)

                return dividend_result
            elif model_type == "asset_based_nav":
                # For real estate, use asset-based approach with DCF as supplement
                asset_result = self.asset_based_valuation(ticker, financial_data, sector)
                dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, sector)

                # Blend the two approaches (60% asset-based, 40% DCF for real estate)
                if 'adjusted_book_value_per_share' in asset_result and asset_result['adjusted_book_value_per_share'] and \
                        'value_per_share' in dcf_result and dcf_result['value_per_share']:
                    blended_value = asset_result['adjusted_book_value_per_share'] * 0.6 + dcf_result[
                        'value_per_share'] * 0.4

                    asset_result['dcf_value_per_share'] = dcf_result['value_per_share']
                    asset_result['blended_value_per_share'] = blended_value

                    # Add key DCF parameters to result
                    asset_result['dcf_parameters'] = {
                        'discount_rate': dcf_result.get('discount_rate'),
                        'initial_growth_rate': dcf_result.get('initial_growth_rate'),
                        'terminal_growth': dcf_result.get('terminal_growth')
                    }

                return asset_result
            else:
                # Default to multi-stage DCF
                return self.multi_stage_dcf_valuation(ticker, financial_data, sector)

        except Exception as e:
            logger.error(f"Error applying sector-specific DCF for {ticker}: {e}")
            # Fall back to base DCF
            return self.multi_stage_dcf_valuation(ticker, financial_data, sector)

    def financial_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Specialized DCF for financial sector companies (banks, insurance, etc.)"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For financial companies, a dividend discount model is often more appropriate
            # But we'll make some modifications for companies that retain more earnings

            # First try standard dividend discount model
            dividend_result = self.dividend_discount_valuation(ticker, financial_data, "Financials")

            # If the company doesn't pay dividends, use a modified approach based on ROE
            if 'error' in dividend_result and 'dividend' in dividend_result['error'].lower():
                # Extract financial statements
                income_stmt = financial_data.get('income_statement')
                balance_sheet = financial_data.get('balance_sheet')

                # Get market data
                market_data = financial_data.get('market_data', {})
                shares_outstanding = market_data.get('shares_outstanding')

                if income_stmt is None or balance_sheet is None:
                    raise ValueError("Missing required financial statements for financial sector valuation")

                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate ROE
                if 'Net Income' in income.index and 'Total Stockholder Equity' in balance.index:
                    net_income = income.loc['Net Income']
                    equity = balance.loc['Total Stockholder Equity']

                    if equity > 0:
                        roe = net_income / equity
                    else:
                        roe = 0.1  # Default assumption
                else:
                    roe = 0.1  # Default assumption

                # Calculate discount rate
                discount_rate = self._calculate_discount_rate(ticker, financial_data, "Financials") or 0.1

                # Calculate sustainable growth rate (g = ROE * Retention Rate)
                if 'Dividends Paid' in financial_data.get('cash_flow', pd.DataFrame()).iloc[:, 0].index:
                    dividends_paid = abs(financial_data['cash_flow'].iloc[:, 0]['Dividends Paid'])
                    retention_rate = 1 - (dividends_paid / net_income) if net_income > 0 else 0.5
                else:
                    retention_rate = 0.5  # Default retention rate

                growth_rate = roe * retention_rate

                # Cap growth rate to reasonable bounds
                growth_rate = min(max(growth_rate, 0.01), 0.15)

                # For financials, use a three-stage model:
                # 1) High growth phase (using calculated growth rate)
                # 2) Transition to sustainable growth
                # 3) Terminal phase with lower growth

                # Parameters
                high_growth_years = 5
                transition_years = 5
                terminal_growth = min(growth_rate * 0.4, 0.03)  # 40% of calculated growth, capped at 3%

                # Starting point: Latest earnings per share
                if shares_outstanding and shares_outstanding > 0:
                    latest_eps = net_income / shares_outstanding
                else:
                    # Estimate shares from market data
                    current_price = market_data.get('share_price')
                    market_cap = market_data.get('market_cap')
                    if current_price and market_cap and current_price > 0:
                        estimated_shares = market_cap / current_price
                        latest_eps = net_income / estimated_shares
                    else:
                        # Cannot calculate value per share without shares data
                        raise ValueError("Cannot determine shares outstanding for per-share valuation")

                # Forecast EPS growth
                future_eps = []

                # High growth phase
                for year in range(1, high_growth_years + 1):
                    eps = latest_eps * (1 + growth_rate) ** year
                    future_eps.append(eps)

                # Transition phase
                for year in range(1, transition_years + 1):
                    # Linear decline in growth rate
                    transition_growth = growth_rate - ((growth_rate - terminal_growth) * year / transition_years)
                    eps = future_eps[-1] * (1 + transition_growth)
                    future_eps.append(eps)

                # Present value calculations
                present_values = []

                # Discount each future EPS
                for i, eps in enumerate(future_eps):
                    present_values.append(eps / (1 + discount_rate) ** (i + 1))

                # Terminal value
                terminal_eps = future_eps[-1] * (1 + terminal_growth)
                terminal_value = terminal_eps / (discount_rate - terminal_growth)
                present_value_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

                # Sum all present values
                value_per_share = sum(present_values) + present_value_terminal

                # Apply margin of safety
                conservative_value = value_per_share * 0.85  # 15% margin of safety

                return {
                    'company': ticker,
                    'method': 'financial_sector_dcf',
                    'value_per_share': value_per_share,
                    'conservative_value': conservative_value,
                    'roe': roe,
                    'retention_rate': retention_rate,
                    'growth_rate': growth_rate,
                    'discount_rate': discount_rate,
                    'terminal_growth': terminal_growth,
                    'high_growth_years': high_growth_years,
                    'transition_years': transition_years,
                    'latest_eps': latest_eps,
                    'forecast_eps': future_eps,
                    'sector': 'Financials'
                }

            return dividend_result

        except Exception as e:
            logger.error(f"Error in financial sector DCF for {ticker}: {e}")
            # Fall back to standard dividend discount model
            return self.dividend_discount_valuation(ticker, financial_data, "Financials")

    def tech_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Specialized DCF for technology sector companies"""
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For tech companies:
            # 1. Longer high-growth phase
            # 2. Focus on revenue growth rather than earnings
            # 3. R&D treated as investment rather than expense for valuation

            # Extract statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for technology sector valuation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Tech-specific adjustments to FCF
            adjusted_fcf = self._adjust_tech_fcf(income_stmt, cash_flow, historical_fcf)

            # Get parameters for tech sector
            params = self._get_dcf_parameters("Technology")

            # Tech companies often have longer growth runways
            high_growth_years = 7  # Longer than standard
            transition_years = 5

            # Growth rates for tech typically start higher but eventually converge to market
            initial_growth_rate = self._estimate_tech_growth_rate(adjusted_fcf, income_stmt, cash_flow)
            steady_growth_rate = min(initial_growth_rate * 0.4, 0.08)  # Cap at 8%
            terminal_growth = params['terminal_growth_rate']

            # Discount rate often higher for tech due to higher risk
            discount_rate = self._calculate_discount_rate(ticker, financial_data, "Technology") or params[
                'default_discount_rate']

            # Starting FCF (most recent adjusted)
            last_fcf = adjusted_fcf.iloc[0]

            # Forecast cash flows for high growth phase
            high_growth_fcf = []
            for year in range(1, high_growth_years + 1):
                fcf = last_fcf * (1 + initial_growth_rate) ** year
                high_growth_fcf.append(fcf)

            # Forecast cash flows for transition phase
            transition_fcf = []
            for year in range(1, transition_years + 1):
                # Linear decline in growth rate
                growth_rate = initial_growth_rate - ((initial_growth_rate - steady_growth_rate) *
                                                     year / transition_years)
                fcf = high_growth_fcf[-1] * (1 + growth_rate) ** year
                transition_fcf.append(fcf)

            # Calculate terminal value
            terminal_fcf = transition_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_high_growth = sum(fcf / (1 + discount_rate) ** year
                                            for year, fcf in enumerate(high_growth_fcf, 1))

            present_value_transition = sum(fcf / (1 + discount_rate) ** (year + high_growth_years)
                                           for year, fcf in enumerate(transition_fcf, 1))

            present_value_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

            # Calculate enterprise and equity values
            enterprise_value = present_value_high_growth + present_value_transition + present_value_terminal
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

            # Apply tech-specific margin of safety (often higher due to higher uncertainty)
            safety_margin = params['default_margin_of_safety'] * 1.2  # 20% higher than standard
            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            return {
                'company': ticker,
                'method': 'tech_sector_dcf',
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': discount_rate,
                'initial_growth_rate': initial_growth_rate,
                'steady_growth_rate': steady_growth_rate,
                'terminal_growth': terminal_growth,
                'high_growth_years': high_growth_years,
                'transition_years': transition_years,
                'adjusted_fcf': adjusted_fcf.to_dict(),
                'forecast_fcf': high_growth_fcf + transition_fcf,
                'terminal_value': terminal_value,
                'present_value_high_growth': present_value_high_growth,
                'present_value_transition': present_value_transition,
                'present_value_terminal': present_value_terminal,
                'net_debt': net_debt,
                'safety_margin': safety_margin,
                'sector': 'Technology'
            }

        except Exception as e:
            logger.error(f"Error in technology sector DCF for {ticker}: {e}")
            # Fall back to multi-stage DCF
            return self.multi_stage_dcf_valuation(ticker, financial_data, "Technology")

    def _adjust_tech_fcf(self, income_stmt: pd.DataFrame, cash_flow: pd.DataFrame,
                         historical_fcf: pd.Series) -> pd.Series:
        """
        Tech-specific FCF adjustments
        For tech companies, R&D is often treated as an investment rather than an expense
        for valuation purposes since it creates long-term value
        """
        try:
            adjusted_fcf = historical_fcf.copy()

            # If Research and Development is available in income statement
            if 'Research and Development' in income_stmt.index:
                # Calculate the adjustment for each period
                for period in adjusted_fcf.index:
                    if period in income_stmt.columns:
                        # Add back a portion of R&D (treating it as investment)
                        rd_expense = income_stmt.loc['Research and Development', period]
                        # Typically add back 70% of R&D for tech companies
                        adjustment = rd_expense * 0.7
                        adjusted_fcf[period] += adjustment

            return adjusted_fcf

        except Exception as e:
            logger.error(f"Error adjusting tech FCF: {e}")
            return historical_fcf  # Return original if adjustment fails

    def _estimate_tech_growth_rate(self, adjusted_fcf: pd.Series,
                                   income_stmt: pd.DataFrame,
                                   cash_flow: pd.DataFrame) -> float:
        """Estimate growth rate for technology companies"""
        try:
            # Base growth from FCF
            base_growth = self._estimate_growth_rate(adjusted_fcf)

            # For tech companies, we also look at revenue growth
            if 'Total Revenue' in income_stmt.index and income_stmt.shape[1] >= 2:
                revenue_growth_rates = []

                for i in range(min(income_stmt.shape[1] - 1, 3)):  # Use up to 3 years
                    if income_stmt.iloc[:, i + 1]['Total Revenue'] > 0:
                        growth = (income_stmt.iloc[:, i]['Total Revenue'] /
                                  income_stmt.iloc[:, i + 1]['Total Revenue']) - 1
                        revenue_growth_rates.append(growth)

                if revenue_growth_rates:
                    # Weight more recent years more heavily
                    weights = list(range(1, len(revenue_growth_rates) + 1))
                    revenue_growth = sum(r * w for r, w in zip(revenue_growth_rates, weights)) / sum(weights)

                    # Blend FCF growth and revenue growth (60/40)
                    blended_growth = (base_growth * 0.6) + (revenue_growth * 0.4)

                    # Apply tech sector premium (typically higher growth)
                    tech_growth = blended_growth * 1.2

                    # Cap to reasonable range
                    return max(0.05, min(0.25, tech_growth))

            # If revenue data insufficient, use base growth with premium
            return max(0.05, min(0.25, base_growth * 1.3))

        except Exception as e:
            logger.error(f"Error estimating tech growth rate: {e}")
            return 0.10  # Default 10% growth for tech

    def energy_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Specialized DCF for energy sector companies"""
        try:
            # For energy companies:
            # 1. Highly cyclical cash flows
            # 2. Commodity price sensitivity
            # 3. High capital intensity

            # Use scenario analysis DCF which is better for cyclical industries
            scenario_analysis = self.scenario_analysis_dcf(ticker, financial_data, "Energy")

            # Enhance with industry-specific considerations
            # Note: In a full implementation, we would adjust for oil price scenarios,
            # reserves, and regulatory changes

            return scenario_analysis

        except Exception as e:
            logger.error(f"Error in energy sector DCF for {ticker}: {e}")
            return self.scenario_analysis_dcf(ticker, financial_data, "Energy")

    def real_estate_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Specialized DCF for real estate companies and REITs"""
        try:
            # For real estate:
            # 1. NAV (Net Asset Value) is critical
            # 2. FFO (Funds From Operations) instead of EPS
            # 3. Cap rates are important

            # Use asset based valuation
            asset_based = self.asset_based_valuation(ticker, financial_data, "Real Estate")

            # Also calculate DCF
            dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Real Estate")

            # Blend the approaches - real estate often valued via blend of asset value and income value
            if 'adjusted_book_value_per_share' in asset_based and asset_based['adjusted_book_value_per_share'] and \
                    'value_per_share' in dcf_result and dcf_result['value_per_share']:
                # 60% asset-based, 40% DCF for real estate
                blended_value = asset_based['adjusted_book_value_per_share'] * 0.6 + dcf_result['value_per_share'] * 0.4

                asset_based['dcf_value_per_share'] = dcf_result['value_per_share']
                asset_based['blended_value_per_share'] = blended_value

            return asset_based

        except Exception as e:
            logger.error(f"Error in real estate DCF for {ticker}: {e}")
            # Return asset-based valuation as fallback
            return self.asset_based_valuation(ticker, financial_data, "Real Estate")