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

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS, SECTOR_METRICS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation

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

    def _estimate_initial_growth_rate(self, historical_fcf: pd.Series, ticker: str = None,
                                      sector: str = None) -> float:
        """
        Estimate initial FCF growth as the CAGR of the historical FCF series
        (index is ordered newest first, as returned by the data loader),
        clipped to a sane range. Falls back to a conservative default.
        """
        default_growth = 0.05
        min_growth, max_growth = 0.02, 0.25
        try:
            values = pd.to_numeric(historical_fcf, errors='coerce').dropna().values
            if len(values) >= 2:
                newest, oldest = float(values[0]), float(values[-1])
                if newest > 0 and oldest > 0:
                    cagr = (newest / oldest) ** (1.0 / (len(values) - 1)) - 1
                    return float(min(max(cagr, min_growth), max_growth))
        except Exception as e:
            logger.warning(f"Growth estimation failed for {ticker}: {e}")
        return default_growth

    def _forecast_financial_statements(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame,
                                       cash_flow: pd.DataFrame, forecast_years: int,
                                       sector: str = None) -> Dict[str, Any]:
        """
        Forecast revenue and free cash flow: revenue growth fades linearly from the
        historical CAGR to the terminal growth rate, FCF = revenue x average historical FCF margin.

        Returns:
            {'fcf': [float, ...], 'summary': pd.DataFrame(revenue, fcf_margin, fcf) per forecast year}
        """
        params = self._get_dcf_parameters(sector)
        historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)
        if historical_fcf.empty:
            raise ValueError("Unable to calculate historical free cash flow")

        revenue_row = next((r for r in ('Total Revenue', 'Revenue', 'Operating Revenue')
                            if r in income_stmt.index), None)
        if revenue_row is None:
            raise ValueError("Revenue not found in income statement")

        revenue = pd.to_numeric(income_stmt.loc[revenue_row], errors='coerce').dropna()
        common = revenue.index.intersection(historical_fcf.index)
        if len(common) == 0 or revenue.iloc[0] <= 0:
            raise ValueError("Cannot align revenue and free cash flow history")

        margin = float((historical_fcf[common] / revenue[common]).replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not np.isfinite(margin):
            raise ValueError("Cannot estimate free cash flow margin")

        start_growth = self._estimate_initial_growth_rate(revenue, None, sector)
        end_growth = params['terminal_growth_rate']

        rows, fcfs = [], []
        current_revenue = float(revenue.iloc[0])
        for year in range(1, forecast_years + 1):
            growth = start_growth + (end_growth - start_growth) * (year - 1) / max(forecast_years - 1, 1)
            current_revenue *= 1 + growth
            fcf = current_revenue * margin
            fcfs.append(fcf)
            rows.append({'year': year, 'revenue_growth': growth, 'revenue': current_revenue,
                         'fcf_margin': margin, 'fcf': fcf})

        return {'fcf': fcfs, 'summary': pd.DataFrame(rows).set_index('year')}

    def multi_stage_dcf_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  sector: str = None) -> Dict[str, Any]:
        """
        Perform a multi-stage DCF valuation with different growth phases

        This model recognizes that companies typically go through different phases of growth:
        1. High growth phase - rapid expansion
        2. Transition phase - growth gradually slows
        3. Mature phase - stable, terminal growth

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with DCF valuation results
        """
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
                previous = transition_fcf[-1] if transition_fcf else high_growth_fcf[-1]
                fcf = previous * (1 + growth_rate)
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

    def dividend_discount_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                    sector: str = None) -> Dict[str, Any]:
        """
        Perform a Dividend Discount Model (DDM) valuation

        The DDM values a stock based on the present value of its expected future dividends.
        This method is especially useful for mature companies with stable dividend payments.

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with DDM valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get parameters based on sector if available
            params = self._get_dcf_parameters(sector)

            # Extract dividend history
            if self.data_loader:
                dividend_history = self.data_loader.get_historical_dividends(ticker)
            else:
                dividend_history = pd.Series()

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Extract income statement and cash flow
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')

            # Check if company pays dividends
            if dividend_history.empty or dividend_history.sum() == 0:
                # Try to get dividend data from cash flow statement
                if cash_flow is not None and 'Dividends Paid' in cash_flow.index:
                    has_dividends = abs(cash_flow.iloc[0]['Dividends Paid']) > 0
                    if not has_dividends:
                        raise ValueError(f"Company {ticker} doesn't appear to pay dividends")
                else:
                    raise ValueError(f"No dividend data available for {ticker}")

            # Calculate or estimate latest annual dividend
            if not dividend_history.empty:
                # Resample to annual frequency if we have multiple dividend payments
                annual_dividends = dividend_history.resample('Y').sum()
                latest_annual_dividend = annual_dividends.iloc[-1] if not annual_dividends.empty else 0
            elif cash_flow is not None and 'Dividends Paid' in cash_flow.index:
                # Estimate from cash flow
                latest_annual_dividend = abs(
                    cash_flow.iloc[0]['Dividends Paid']) / shares_outstanding if shares_outstanding else 0
            else:
                raise ValueError(f"Unable to determine dividend payments for {ticker}")

            # If dividend is zero, this model is not appropriate
            if latest_annual_dividend == 0:
                raise ValueError(f"Cannot apply DDM to company {ticker} with no dividend payments")

            # Estimate dividend growth rate
            if not dividend_history.empty and len(dividend_history) >= 3:
                # Calculate growth rate from historical dividends
                annual_dividends = dividend_history.resample('Y').sum()

                if len(annual_dividends) >= 2:
                    # Calculate year-over-year growth rates
                    growth_rates = []
                    for i in range(1, len(annual_dividends)):
                        if annual_dividends.iloc[i - 1] > 0:
                            growth_rate = (annual_dividends.iloc[i] / annual_dividends.iloc[i - 1]) - 1
                            growth_rates.append(growth_rate)

                    if growth_rates:
                        # Use average growth rate, potentially with more weight to recent years
                        dividend_growth_rate = sum(growth_rates) / len(growth_rates)
                    else:
                        # Default growth if calculation fails
                        dividend_growth_rate = 0.03  # 3% default growth
                else:
                    dividend_growth_rate = 0.03  # Default growth
            else:
                # Estimate from earnings growth if available
                if income_stmt is not None and income_stmt.shape[1] >= 2:
                    if 'Net Income' in income_stmt.index:
                        # Calculate earnings growth rate
                        latest_income = income_stmt.iloc[0]['Net Income']
                        prev_income = income_stmt.iloc[1]['Net Income']

                        if prev_income > 0:
                            earnings_growth = (latest_income / prev_income) - 1
                            # Dividend growth is typically lower than earnings growth
                            dividend_growth_rate = min(earnings_growth * 0.7, 0.15)  # Cap at 15%
                        else:
                            dividend_growth_rate = 0.03  # Default

            # Cap dividend growth rate to reasonable bounds
            dividend_growth_rate = max(0.01, min(0.15, dividend_growth_rate))

            # Calculate discount rate
            discount_rate = self._calculate_discount_rate(ticker, financial_data, sector) or params[
                'default_discount_rate']

            # Ensure discount rate > dividend growth rate for model validity
            if discount_rate <= dividend_growth_rate:
                discount_rate = dividend_growth_rate + 0.03  # Add 3% minimum spread

            # Check if we are using single-stage or multi-stage model
            # For high-growth companies, a multi-stage model is more appropriate

            if dividend_growth_rate > discount_rate * 0.66:  # If growth rate > 2/3 of discount rate
                # Use multi-stage model (high growth followed by stable growth)
                high_growth_years = 5
                terminal_growth_rate = min(0.03, params['terminal_growth_rate'])  # Long-term sustainable growth

                # Calculate dividend payments during high growth phase
                high_growth_dividends = []
                for year in range(1, high_growth_years + 1):
                    dividend = latest_annual_dividend * (1 + dividend_growth_rate) ** year
                    high_growth_dividends.append(dividend)

                # Calculate present value of high growth phase dividends
                pv_high_growth = sum(
                    dividend / (1 + discount_rate) ** year
                    for year, dividend in enumerate(high_growth_dividends, 1)
                )

                # Calculate terminal value using Gordon Growth Model
                # The dividend in year after high growth phase
                terminal_dividend = high_growth_dividends[-1] * (1 + terminal_growth_rate)
                terminal_value = terminal_dividend / (discount_rate - terminal_growth_rate)

                # Discount terminal value to present
                pv_terminal = terminal_value / (1 + discount_rate) ** high_growth_years

                # Final valuation is sum of high growth dividends + terminal value
                value_per_share = pv_high_growth + pv_terminal

                # Results dictionary with details
                result = {
                    'company': ticker,
                    'method': 'multi_stage_ddm',
                    'value_per_share': value_per_share,
                    'latest_annual_dividend': latest_annual_dividend,
                    'high_growth_rate': dividend_growth_rate,
                    'terminal_growth_rate': terminal_growth_rate,
                    'discount_rate': discount_rate,
                    'high_growth_years': high_growth_years,
                    'high_growth_dividends': high_growth_dividends,
                    'pv_high_growth': pv_high_growth,
                    'pv_terminal': pv_terminal,
                    'sector': sector
                }
            else:
                # Use single-stage Gordon Growth Model
                # V = D1 / (r - g) where D1 is next year's dividend
                next_dividend = latest_annual_dividend * (1 + dividend_growth_rate)
                value_per_share = next_dividend / (discount_rate - dividend_growth_rate)

                # Results dictionary with details
                result = {
                    'company': ticker,
                    'method': 'gordon_growth_ddm',
                    'value_per_share': value_per_share,
                    'latest_annual_dividend': latest_annual_dividend,
                    'next_dividend': next_dividend,
                    'dividend_growth_rate': dividend_growth_rate,
                    'discount_rate': discount_rate,
                    'sector': sector
                }

            # Apply margin of safety
            safety_margin = params['default_margin_of_safety']
            result['conservative_value'] = value_per_share * (1 - safety_margin)
            result['safety_margin'] = safety_margin

            return result

        except Exception as e:
            logger.error(f"Error in dividend discount valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'dividend_discount_model',
                'value_per_share': None,
                'error': str(e)
            }

    def asset_based_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                              sector: str = None) -> Dict[str, Any]:
        """
        Perform an asset-based valuation (Net Asset Value)

        This model values a company based on the fair market value of its assets
        minus liabilities. It is most appropriate for capital-intensive businesses,
        real estate companies, holding companies, and investment funds.

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments

        Returns:
            Dictionary with asset-based valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Extract balance sheet
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if balance_sheet is None:
                raise ValueError("Missing required balance sheet for asset-based valuation")

            # Get latest balance sheet data
            latest_balance = balance_sheet.iloc[:, 0]

            # Extract key asset and liability categories
            total_assets = latest_balance.get('Total Assets')
            total_liabilities = latest_balance.get('Total Liabilities Net Minority Interest')

            if total_assets is None or total_liabilities is None:
                raise ValueError("Missing total assets or liabilities data")

            # Calculate book value (net asset value)
            book_value = total_assets - total_liabilities

            # Calculate book value per share
            if shares_outstanding is not None and shares_outstanding > 0:
                book_value_per_share = book_value / shares_outstanding
            else:
                # Estimate shares outstanding from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')

                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    book_value_per_share = book_value / estimated_shares
                else:
                    raise ValueError("Cannot determine shares outstanding for per-share valuation")

            # Apply sector-specific adjustments to book value
            adjusted_book_value, adjustment_factors = self._apply_asset_adjustments(latest_balance, sector)

            # Calculate adjusted book value per share
            adjusted_book_value_per_share = adjusted_book_value / shares_outstanding if shares_outstanding else None

            # Calculate price-to-book ratio
            current_price = market_data.get('share_price')
            if current_price and book_value_per_share:
                price_to_book = current_price / book_value_per_share
            else:
                price_to_book = None

            # Calculate adjusted price-to-book ratio
            if current_price and adjusted_book_value_per_share:
                adjusted_price_to_book = current_price / adjusted_book_value_per_share
            else:
                adjusted_price_to_book = None

            # Apply margin of safety
            # For asset-based valuations, often use adjusted book value as the base
            params = self._get_dcf_parameters(sector)
            safety_margin = params['default_margin_of_safety'] * 0.8  # Typically lower for asset-based valuations
            conservative_value = adjusted_book_value_per_share * (
                        1 - safety_margin) if adjusted_book_value_per_share else None

            return {
                'company': ticker,
                'method': 'asset_based_valuation',
                'book_value': book_value,
                'book_value_per_share': book_value_per_share,
                'adjusted_book_value': adjusted_book_value,
                'adjusted_book_value_per_share': adjusted_book_value_per_share,
                'price_to_book': price_to_book,
                'adjusted_price_to_book': adjusted_price_to_book,
                'current_price': current_price,
                'conservative_value': conservative_value,
                'safety_margin': safety_margin,
                'adjustment_factors': adjustment_factors,
                'sector': sector
            }

        except Exception as e:
            logger.error(f"Error in asset-based valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'asset_based_valuation',
                'book_value_per_share': None,
                'adjusted_book_value_per_share': None,
                'error': str(e)
            }

    def _apply_asset_adjustments(self, balance: pd.Series, sector: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Apply adjustments to balance sheet items to get a more accurate asset value

        Different assets may need to be marked up or down from their book value:
        - Cash and cash equivalents: typically no adjustment needed
        - Inventory: may need discounting due to obsolescence risk
        - Real estate: often undervalued on balance sheets (historical cost)
        - Intangible assets: may be overvalued or undervalued

        Args:
            balance: Balance sheet data as pandas Series
            sector: Company sector for sector-specific adjustments

        Returns:
            Tuple of (adjusted book value, dictionary of adjustment factors)
        """
        # Start with unadjusted total assets and liabilities
        total_assets = balance.get('Total Assets', 0)
        total_liabilities = balance.get('Total Liabilities Net Minority Interest', 0)

        # Initialize adjustments dictionary
        adjustments = {}
        adjustment_factors = {}

        # Cash and short-term investments (typically no adjustment)
        cash = balance.get('Cash and Cash Equivalents', 0)
        short_term_investments = balance.get('Short Term Investments', 0)
        adjustments['Cash and Investments'] = cash + short_term_investments
        adjustment_factors['Cash and Investments'] = 1.0

        # Accounts Receivable (typically discounted)
        receivables = balance.get('Net Receivables', 0)
        ar_factor = 0.9  # 10% discount for potentially uncollectable accounts
        adjustments['Receivables'] = receivables * ar_factor
        adjustment_factors['Receivables'] = ar_factor

        # Inventory (discount varies by sector)
        inventory = balance.get('Inventory', 0)
        if sector == 'Technology':
            inv_factor = 0.7  # Higher obsolescence risk
        elif sector == 'Consumer Discretionary':
            inv_factor = 0.75  # Fashion changes, trends
        elif sector == 'Consumer Staples':
            inv_factor = 0.9  # Lower obsolescence
        else:
            inv_factor = 0.8  # Default
        adjustments['Inventory'] = inventory * inv_factor
        adjustment_factors['Inventory'] = inv_factor

        # Property, Plant & Equipment (often undervalued, especially land)
        ppe = balance.get('Property Plant and Equipment', 0)
        if sector == 'Real Estate':
            ppe_factor = 1.5  # Significantly undervalued often
        elif sector in ['Industrials', 'Energy', 'Materials']:
            ppe_factor = 1.2  # Typically undervalued
        else:
            ppe_factor = 1.0  # Default - no adjustment
        adjustments['PPE'] = ppe * ppe_factor
        adjustment_factors['PPE'] = ppe_factor

        # Intangible Assets (highly variable)
        intangibles = balance.get('Intangible Assets', 0)
        goodwill = balance.get('Goodwill', 0)

        if sector in ['Technology', 'Healthcare', 'Communication Services']:
            # These sectors often have valuable IP and brand value
            intangibles_factor = 1.2
        elif sector in ['Consumer Staples', 'Consumer Discretionary']:
            # These can have valuable brands
            intangibles_factor = 1.1
        else:
            # More conservative for other sectors
            intangibles_factor = 0.7

        # Often goodwill is written off in conservative analysis
        goodwill_factor = 0.0

        adjustments['Intangibles'] = intangibles * intangibles_factor
        adjustment_factors['Intangibles'] = intangibles_factor

        adjustments['Goodwill'] = goodwill * goodwill_factor
        adjustment_factors['Goodwill'] = goodwill_factor

        # Other assets (miscellaneous, often discounted)
        other_assets = balance.get('Other Assets', 0)
        other_factor = 0.8
        adjustments['Other Assets'] = other_assets * other_factor
        adjustment_factors['Other Assets'] = other_factor

        # Liabilities (typically taken at face value)
        # But could adjust for off-balance sheet liabilities if data available
        liability_factor = 1.0
        adjustments['Liabilities'] = total_liabilities * liability_factor
        adjustment_factors['Liabilities'] = liability_factor

        # Calculate total adjusted assets
        adjusted_assets = sum([adj for key, adj in adjustments.items() if key != 'Liabilities'])

        # Calculate adjusted book value
        adjusted_book_value = adjusted_assets - adjustments['Liabilities']

        return adjusted_book_value, adjustment_factors


class SectorSpecificDCF(AdvancedDCFValuation):
    """
    DCF valuation models with sector-specific adjustments and techniques.
    Each sector may require different approaches to forecasting, risk assessment, and valuation.

    This class provides specialized DCF implementations for different industry sectors,
    as different sectors have unique financial characteristics, growth patterns,
    and value drivers that require customized valuation approaches.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize the sector-specific DCF valuation class"""
        super().__init__(data_loader)

        # Load sector-specific metrics and benchmarks from config
        self.sector_metrics = SECTOR_METRICS

        logger.info("Initialized SectorSpecificDCF with sector-specific parameters")

    def get_sector_dcf_model(self, sector: str) -> str:
        """
        Get the appropriate DCF model type for a given sector

        Different sectors require different valuation approaches due to varying
        business models, growth patterns, and financial characteristics.

        Args:
            sector: Industry sector name

        Returns:
            String identifying the most appropriate DCF model type
        """
        # Map sectors to the most appropriate valuation models
        # This mapping is based on industry best practices as documented in the PDF materials
        sector_model_mapping = {
            "Technology": "multi_stage_dcf",  # Tech needs multi-stage due to high initial growth
            "Healthcare": "multi_stage_dcf",  # Healthcare has longer growth runways
            "Financials": "dividend_discount",  # Banks and financials are better valued with DDM
            "Consumer Discretionary": "forecast_driven_dcf",  # Retail benefits from detailed forecasting
            "Consumer Staples": "forecast_driven_dcf",  # Stable companies with predictable metrics
            "Energy": "scenario_analysis_dcf",  # Energy needs scenarios due to commodity exposure
            "Industrials": "forecast_driven_dcf",  # Manufacturing benefits from detailed forecasts
            "Materials": "scenario_analysis_dcf",  # Materials are cyclical and need scenario analysis
            "Real Estate": "asset_based_nav",  # REITs are primarily valued on asset value
            "Communication Services": "multi_stage_dcf",  # Telecom/media has long-term contracts
            "Utilities": "dividend_discount"  # Utilities are mature, dividend-focused companies
        }

        # Try exact match
        model = sector_model_mapping.get(sector)
        if model:
            return model

        # Try partial match
        if sector:
            for sector_key, model_type in sector_model_mapping.items():
                if sector.lower() in sector_key.lower() or sector_key.lower() in sector.lower():
                    return model_type

        # Default to multi-stage DCF as the most versatile approach
        return "multi_stage_dcf"

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
                # For real estate, use specialized real estate DCF model
                return self.real_estate_dcf(ticker, financial_data)
            else:
                # Default to multi-stage DCF
                return self.multi_stage_dcf_valuation(ticker, financial_data, sector)

        except Exception as e:
            logger.error(f"Error applying sector-specific DCF for {ticker}: {e}")
            # Fall back to base DCF
            return self.multi_stage_dcf_valuation(ticker, financial_data, sector)

    def financial_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for financial sector companies (banks, insurance, etc.)

        Financial companies have unique characteristics requiring adjustments to standard DCF:
        1. FCF is not as relevant due to regulated capital requirements
        2. Dividend Discount Model often works better for banks
        3. Profitability is tied to interest rates and credit quality
        4. Return on Equity (ROE) is a critical driver of value
        5. Balance sheet growth is a key factor unlike other sectors

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with financial sector DCF valuation results
        """
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
        """
        Specialized DCF for technology sector companies

        Technology companies have unique characteristics that require adjustments to standard DCF:
        1. Higher focus on growth and reinvestment
        2. R&D is treated more as an investment than an expense
        3. Longer growth runway but higher uncertainty
        4. Often valued using alternative metrics like ARR, CAC, LTV

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with technology sector DCF valuation results
        """
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
                previous = transition_fcf[-1] if transition_fcf else high_growth_fcf[-1]
                fcf = previous * (1 + growth_rate)
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
        """
        Specialized DCF for energy sector companies

        Energy companies have unique characteristics requiring adjustments to standard DCF:
        1. Highly cyclical cash flows tied to commodity prices
        2. Reserve replacement is critical to long-term valuation
        3. High capital intensity and maintenance capex requirements
        4. Increasing regulatory and environmental pressures
        5. Different valuation metrics like EV/EBITDA, P/CF, and reserve valuations

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with energy sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For energy companies, scenario analysis works better due to cyclicality
            scenario_analysis = self.scenario_analysis_dcf(ticker, financial_data, "Energy")

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Try to enhance with energy-specific metrics
            energy_metrics = {}

            # Calculate reserve life index if data available
            # Note: In a real implementation, this would come from company reports
            # As we don't have reserve data, we estimate based on available info
            energy_metrics['reserve_life_index'] = None
            energy_metrics['reserve_replacement_ratio'] = None

            # Calculate P/CF (Price to Cash Flow) - important for energy companies
            try:
                market_data = financial_data.get('market_data', {})
                market_cap = market_data.get('market_cap')

                if market_cap and 'Operating Cash Flow' in cash_flow.index:
                    operating_cf = cash_flow.iloc[:, 0]['Operating Cash Flow']
                    if operating_cf > 0:
                        energy_metrics['price_to_cash_flow'] = market_cap / operating_cf
            except:
                energy_metrics['price_to_cash_flow'] = None

            # Enhance production costs if available
            energy_metrics['production_cost_per_boe'] = None

            # Add supplementary metrics to the result
            scenario_analysis['energy_specific_metrics'] = energy_metrics

            # Add oil price sensitivity analysis
            # Simplified approach: estimate how valuation changes with oil price changes
            oil_price_sensitivity = {
                'base_case_oil_price': 70,  # Assumed base case oil price
                'upside_case': scenario_analysis['scenarios']['bullish']['value_per_share'],
                'base_case': scenario_analysis['scenarios']['base']['value_per_share'],
                'downside_case': scenario_analysis['scenarios']['bearish']['value_per_share'],
            }

            scenario_analysis['oil_price_sensitivity'] = oil_price_sensitivity

            return scenario_analysis

        except Exception as e:
            logger.error(f"Error in energy sector DCF for {ticker}: {e}")
            return self.scenario_analysis_dcf(ticker, financial_data, "Energy")

    def real_estate_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for real estate companies and REITs

        Real estate companies have unique characteristics requiring adjustments to standard DCF:
        1. NAV (Net Asset Value) is a critical valuation method
        2. FFO (Funds From Operations) is preferred over EPS
        3. Cap rates determine property values and vary by property type
        4. Recurring and predictable cash flows from leases
        5. Different accounting treatment with large depreciation charges

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with real estate DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For real estate, we need both NAV and income approaches
            asset_based = self.asset_based_valuation(ticker, financial_data, "Real Estate")
            dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Real Estate")

            # Extract key financial statements for REIT-specific metrics
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Calculate FFO (Funds From Operations) - key REIT metric
            # FFO = Net Income + Depreciation - Gains on Sale of Properties
            reit_metrics = {}

            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                net_income = latest_income.get('Net Income', 0)
                depreciation = latest_income.get('Depreciation & Amortization', 0)

                # We don't typically have gains data, so it's a simplified FFO calculation
                ffo = net_income + depreciation

                # Calculate FFO per share
                shares_outstanding = financial_data.get('market_data', {}).get('shares_outstanding')
                if shares_outstanding and shares_outstanding > 0:
                    ffo_per_share = ffo / shares_outstanding
                    reit_metrics['ffo_per_share'] = ffo_per_share

                    # Calculate P/FFO (Price to FFO) - important REIT valuation metric
                    current_price = financial_data.get('market_data', {}).get('share_price')
                    if current_price and ffo_per_share > 0:
                        p_ffo = current_price / ffo_per_share
                        reit_metrics['p_ffo'] = p_ffo

                # Calculate FFO yield
                market_cap = financial_data.get('market_data', {}).get('market_cap')
                if market_cap and market_cap > 0 and ffo > 0:
                    ffo_yield = ffo / market_cap
                    reit_metrics['ffo_yield'] = ffo_yield

            # Add occupancy rate and lease metrics if available
            # (These would normally come from company reports)
            reit_metrics['occupancy_rate'] = None
            reit_metrics['weighted_avg_lease_term'] = None

            # Estimate cap rate (simplified)
            if 'ffo_yield' in reit_metrics:
                # Cap rate is often related to FFO yield, but typically higher
                estimated_cap_rate = min(reit_metrics['ffo_yield'] * 1.2, 0.08)
                reit_metrics['estimated_cap_rate'] = estimated_cap_rate

            # Add REIT-specific metrics to asset-based result
            asset_based['reit_metrics'] = reit_metrics

            # Blend the approaches - real estate often valued via blend of asset value and income value
            if 'adjusted_book_value_per_share' in asset_based and asset_based['adjusted_book_value_per_share'] and \
                    'value_per_share' in dcf_result and dcf_result['value_per_share']:
                # 60% asset-based, 40% DCF for real estate
                blended_value = asset_based['adjusted_book_value_per_share'] * 0.6 + dcf_result['value_per_share'] * 0.4

                asset_based['dcf_value_per_share'] = dcf_result['value_per_share']
                asset_based['blended_value_per_share'] = blended_value

                # If we have P/FFO, also calculate FFO-based value
                if 'p_ffo' in reit_metrics and 'ffo_per_share' in reit_metrics:
                    # Use sector average P/FFO (typically 12-18)
                    sector_avg_p_ffo = 15
                    ffo_based_value = reit_metrics['ffo_per_share'] * sector_avg_p_ffo
                    asset_based['ffo_based_value'] = ffo_based_value

                    # Include FFO-based value in the blend (50% NAV, 30% DCF, 20% FFO)
                    enhanced_blend = (
                            asset_based['adjusted_book_value_per_share'] * 0.5 +
                            dcf_result['value_per_share'] * 0.3 +
                            ffo_based_value * 0.2
                    )

                    asset_based['enhanced_blended_value'] = enhanced_blend

            return asset_based

        except Exception as e:
            logger.error(f"Error in real estate DCF for {ticker}: {e}")
            # Return asset-based valuation as fallback
            return self.asset_based_valuation(ticker, financial_data, "Real Estate")

    def consumer_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                            discretionary: bool = True) -> Dict[str, Any]:
        """
        Specialized DCF for consumer sector companies (discretionary vs. staples)

        Consumer companies have unique characteristics requiring adjustments to standard DCF:
        1. Consumer Discretionary: More cyclical, higher growth potential, but more sensitive to economic cycles
        2. Consumer Staples: More defensive, lower growth, stable margins and cash flows

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            discretionary: True for Consumer Discretionary, False for Consumer Staples

        Returns:
            Dictionary with consumer sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Set sector based on discretionary flag
            sector = "Consumer Discretionary" if discretionary else "Consumer Staples"

            # For consumer companies, forecast-driven DCF works best
            return self.forecast_driven_dcf(ticker, financial_data, sector)

        except Exception as e:
            logger.error(f"Error in consumer sector DCF for {ticker}: {e}")
            return self.forecast_driven_dcf(ticker, financial_data, sector)

    def utilities_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for utilities sector companies

        Utilities companies have unique characteristics requiring adjustments to standard DCF:
        1. Highly regulated with stable and predictable cash flows
        2. High capital expenditures with long asset lifespans
        3. Dividend-focused with high payout ratios
        4. Growth tied to regulatory approvals and rate cases
        5. Sensitivity to interest rates due to capital structure

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with utilities sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For utilities, dividend discount model is often most appropriate
            dividend_result = self.dividend_discount_valuation(ticker, financial_data, "Utilities")

            # If the company doesn't pay dividends, fall back to standard DCF
            if 'error' in dividend_result and 'dividend' in dividend_result['error'].lower():
                logger.info(f"Falling back to multi-stage DCF for {ticker} as dividend model failed")
                dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Utilities")
            else:
                # Use both DDM and DCF for comparison
                dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Utilities")

                # Combine results - for utilities, DDM is often more reliable
                if 'value_per_share' in dividend_result and 'value_per_share' in dcf_result:
                    # Weighted average (60% DDM, 40% DCF for utilities)
                    if dividend_result['value_per_share'] is not None and dcf_result['value_per_share'] is not None:
                        combined_value = (
                                dividend_result['value_per_share'] * 0.6 +
                                dcf_result['value_per_share'] * 0.4
                        )
                        dcf_result['combined_value_per_share'] = combined_value
                        dcf_result['ddm_value_per_share'] = dividend_result['value_per_share']

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Calculate utilities-specific metrics
            utilities_metrics = {}

            # Dividend metrics (critical for utilities)
            if cash_flow is not None and income_stmt is not None and not cash_flow.empty and not income_stmt.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Dividends Paid' in latest_cash_flow.index and 'Net Income' in latest_income.index:
                    dividend_paid = abs(latest_cash_flow['Dividends Paid'])
                    net_income = latest_income['Net Income']

                    if net_income > 0:
                        payout_ratio = dividend_paid / net_income
                        utilities_metrics['dividend_payout_ratio'] = payout_ratio

                        # Get market data
                        market_data = financial_data.get('market_data', {})
                        market_cap = market_data.get('market_cap')

                        if market_cap and market_cap > 0:
                            dividend_yield = dividend_paid / market_cap
                            utilities_metrics['dividend_yield'] = dividend_yield

            # Regulatory metrics - utilities are heavily regulated
            # Allowed return on equity (ROE) is often set by regulators
            if income_stmt is not None and balance_sheet is not None and not income_stmt.empty and not balance_sheet.empty:
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]

                if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                    net_income = latest_income['Net Income']
                    equity = latest_balance['Total Stockholder Equity']

                    if equity > 0:
                        actual_roe = net_income / equity
                        utilities_metrics['actual_roe'] = actual_roe

                        # Typical allowed ROE for utilities is 9-11%
                        allowed_roe = 0.1  # 10% is a common regulatory approved ROE
                        utilities_metrics['estimated_allowed_roe'] = allowed_roe

                        # ROE gap (difference between actual and allowed)
                        roe_gap = actual_roe - allowed_roe
                        utilities_metrics['roe_gap'] = roe_gap

            # Interest rate sensitivity
            # Utilities typically have high debt levels
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]

                if 'Total Debt' in latest_balance.index and 'Total Assets' in latest_balance.index:
                    debt_to_assets = latest_balance['Total Debt'] / latest_balance['Total Assets']
                    utilities_metrics['debt_to_assets'] = debt_to_assets

                    # Interest rate sensitivity (higher debt = higher sensitivity)
                    # Simplified measure - actual models would be more complex
                    if debt_to_assets > 0.6:
                        interest_rate_sensitivity = "High"
                    elif debt_to_assets > 0.4:
                        interest_rate_sensitivity = "Medium"
                    else:
                        interest_rate_sensitivity = "Low"

                    utilities_metrics['interest_rate_sensitivity'] = interest_rate_sensitivity

                    # Impact of 1% interest rate increase
                    if 'Total Debt' in latest_balance.index and income_stmt is not None and 'Net Income' in income_stmt.iloc[
                                                                                                            :,
                                                                                                            0].index:
                        debt = latest_balance['Total Debt']
                        net_income = income_stmt.iloc[:, 0]['Net Income']

                        # Simplified calculation - assumes all debt reprices
                        interest_impact = debt * 0.01  # 1% of debt
                        after_tax_impact = interest_impact * 0.75  # Assuming 25% tax rate

                        if net_income > 0:
                            percent_impact = after_tax_impact / net_income
                            utilities_metrics['interest_rate_impact'] = percent_impact

            # Rate base growth (key driver for utilities)
            if balance_sheet is not None and balance_sheet.shape[1] >= 2:
                if 'Property Plant and Equipment' in balance_sheet.index:
                    current_ppe = balance_sheet.iloc[:, 0]['Property Plant and Equipment']
                    prev_ppe = balance_sheet.iloc[:, 1]['Property Plant and Equipment']

                    if prev_ppe > 0:
                        rate_base_growth = (current_ppe / prev_ppe) - 1
                        utilities_metrics['rate_base_growth'] = rate_base_growth

            # Capex intensity
            if cash_flow is not None and income_stmt is not None and not cash_flow.empty and not income_stmt.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Capital Expenditure' in latest_cash_flow.index and 'Total Revenue' in latest_income.index:
                    capex = abs(latest_cash_flow['Capital Expenditure'])
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        capex_to_revenue = capex / revenue
                        utilities_metrics['capex_to_revenue'] = capex_to_revenue

                        # Capex to Depreciation ratio
                        if 'Depreciation & Amortization' in latest_income.index:
                            depreciation = latest_income['Depreciation & Amortization']

                            if depreciation > 0:
                                capex_to_depreciation = capex / depreciation
                                utilities_metrics['capex_to_depreciation'] = capex_to_depreciation

                                # Classify capex strategy
                                if capex_to_depreciation > 1.5:
                                    utilities_metrics['capex_strategy'] = "Heavy investment/growth"
                                elif capex_to_depreciation > 1.1:
                                    utilities_metrics['capex_strategy'] = "Moderate investment"
                                elif capex_to_depreciation > 0.9:
                                    utilities_metrics['capex_strategy'] = "Maintenance capex"
                                else:
                                    utilities_metrics['capex_strategy'] = "Underinvestment"

            # Add utilities metrics to the result
            dcf_result['utilities_metrics'] = utilities_metrics

            return dcf_result

        except Exception as e:
            logger.error(f"Error in utilities sector DCF for {ticker}: {e}")
            return self.multi_stage_dcf_valuation(ticker, financial_data, "Utilities")

    def relative_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                           sector: str = None, peers: List[str] = None) -> Dict[str, Any]:
        """
        Perform a relative valuation based on market multiples

        This model values a company based on how similar companies are valued
        in the market using various multiples like P/E, EV/EBITDA, etc.

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data
            sector: Company sector for sector-specific adjustments
            peers: List of peer company tickers (if None, will be determined automatically)

        Returns:
            Dictionary with relative valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sector if not provided
            if sector is None:
                company_info = financial_data.get('company_info', {})
                sector = company_info.get('sector')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Extract key financial metrics
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for relative valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate key metrics for this company
            company_metrics = {}

            # Get basic metrics
            if 'Total Revenue' in latest_income.index:
                company_metrics['revenue'] = latest_income['Total Revenue']

            if 'Net Income' in latest_income.index:
                company_metrics['net_income'] = latest_income['Net Income']

                # Calculate EPS if shares outstanding available
                if shares_outstanding and shares_outstanding > 0:
                    company_metrics['eps'] = company_metrics['net_income'] / shares_outstanding

            if 'EBITDA' in latest_income.index:
                company_metrics['ebitda'] = latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                # Estimate EBITDA if not directly available
                company_metrics['ebitda'] = latest_income['Operating Income'] + latest_income[
                    'Depreciation & Amortization']

            if 'Total Assets' in latest_balance.index:
                company_metrics['total_assets'] = latest_balance['Total Assets']

            if 'Total Stockholder Equity' in latest_balance.index:
                company_metrics['book_value'] = latest_balance['Total Stockholder Equity']

                # Calculate BVPS if shares outstanding available
                if shares_outstanding and shares_outstanding > 0:
                    company_metrics['bvps'] = company_metrics['book_value'] / shares_outstanding

            # Calculate Enterprise Value
            if market_data.get('market_cap') is not None:
                market_cap = market_data['market_cap']
                company_metrics['market_cap'] = market_cap

                # Calculate EV
                total_debt = latest_balance.get('Total Debt', 0)
                cash = latest_balance.get('Cash and Cash Equivalents', 0)
                enterprise_value = market_cap + total_debt - cash
                company_metrics['enterprise_value'] = enterprise_value

                # Calculate multiples
                if 'revenue' in company_metrics and company_metrics['revenue'] > 0:
                    company_metrics['ps_ratio'] = market_cap / company_metrics['revenue']
                    company_metrics['ev_to_revenue'] = enterprise_value / company_metrics['revenue']

                if 'net_income' in company_metrics and company_metrics['net_income'] > 0:
                    company_metrics['pe_ratio'] = market_cap / company_metrics['net_income']

                if 'ebitda' in company_metrics and company_metrics['ebitda'] > 0:
                    company_metrics['ev_to_ebitda'] = enterprise_value / company_metrics['ebitda']

                if 'book_value' in company_metrics and company_metrics['book_value'] > 0:
                    company_metrics['pb_ratio'] = market_cap / company_metrics['book_value']

            # Get sector multiples from benchmark data
            sector_multiples = self._get_sector_multiples(sector)

            # If peers are provided, get their multiples
            peer_multiples = {}
            if peers and self.data_loader:
                peer_data = {}
                for peer in peers:
                    try:
                        # Get peer data
                        peer_financial_data = self._load_financial_data(peer)
                        peer_data[peer] = peer_financial_data
                    except Exception as e:
                        logger.warning(f"Error loading data for peer {peer}: {e}")
                        continue

                # Calculate multiples for peers
                for peer, data in peer_data.items():
                    peer_multiples[peer] = self._calculate_multiples(data)

            # Calculate valuations based on different multiples
            valuations = {}

            # P/E based valuation
            if 'net_income' in company_metrics and company_metrics['net_income'] > 0:
                if sector_multiples and 'median_pe' in sector_multiples:
                    pe_value = company_metrics['net_income'] * sector_multiples['median_pe']
                    valuations['pe_sector_median'] = pe_value / shares_outstanding if shares_outstanding else None

                # Peer average P/E
                peer_pe_values = [p.get('pe_ratio') for p in peer_multiples.values() if p.get('pe_ratio')]
                if peer_pe_values:
                    peer_avg_pe = sum(peer_pe_values) / len(peer_pe_values)
                    pe_value = company_metrics['net_income'] * peer_avg_pe
                    valuations['pe_peer_average'] = pe_value / shares_outstanding if shares_outstanding else None

            # EV/EBITDA based valuation
            if 'ebitda' in company_metrics and company_metrics['ebitda'] > 0:
                if sector_multiples and 'median_ev_to_ebitda' in sector_multiples:
                    ev_value = company_metrics['ebitda'] * sector_multiples['median_ev_to_ebitda']

                    # Adjust for debt and cash
                    total_debt = latest_balance.get('Total Debt', 0)
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)
                    equity_value = ev_value - total_debt + cash

                    valuations[
                        'ev_ebitda_sector_median'] = equity_value / shares_outstanding if shares_outstanding else None

                # Peer average EV/EBITDA
                peer_ev_ebitda_values = [p.get('ev_to_ebitda') for p in peer_multiples.values() if
                                         p.get('ev_to_ebitda')]
                if peer_ev_ebitda_values:
                    peer_avg_ev_ebitda = sum(peer_ev_ebitda_values) / len(peer_ev_ebitda_values)
                    ev_value = company_metrics['ebitda'] * peer_avg_ev_ebitda

                    # Adjust for debt and cash
                    total_debt = latest_balance.get('Total Debt', 0)
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)
                    equity_value = ev_value - total_debt + cash

                    valuations[
                        'ev_ebitda_peer_average'] = equity_value / shares_outstanding if shares_outstanding else None

            # P/S based valuation
            if 'revenue' in company_metrics and company_metrics['revenue'] > 0:
                if sector_multiples and 'median_ps' in sector_multiples:
                    ps_value = company_metrics['revenue'] * sector_multiples['median_ps']
                    valuations['ps_sector_median'] = ps_value / shares_outstanding if shares_outstanding else None

                # Peer average P/S
                peer_ps_values = [p.get('ps_ratio') for p in peer_multiples.values() if p.get('ps_ratio')]
                if peer_ps_values:
                    peer_avg_ps = sum(peer_ps_values) / len(peer_ps_values)
                    ps_value = company_metrics['revenue'] * peer_avg_ps
                    valuations['ps_peer_average'] = ps_value / shares_outstanding if shares_outstanding else None

            # P/B based valuation
            if 'book_value' in company_metrics and company_metrics['book_value'] > 0:
                if sector_multiples and 'median_pb' in sector_multiples:
                    pb_value = company_metrics['book_value'] * sector_multiples['median_pb']
                    valuations['pb_sector_median'] = pb_value / shares_outstanding if shares_outstanding else None

                # Peer average P/B
                peer_pb_values = [p.get('pb_ratio') for p in peer_multiples.values() if p.get('pb_ratio')]
                if peer_pb_values:
                    peer_avg_pb = sum(peer_pb_values) / len(peer_pb_values)
                    pb_value = company_metrics['book_value'] * peer_avg_pb
                    valuations['pb_peer_average'] = pb_value / shares_outstanding if shares_outstanding else None

            # Calculate weighted average valuation
            # Weights depend on sector
            if sector == "Financials":
                weights = {
                    'pe_sector_median': 0.2,
                    'pe_peer_average': 0.2,
                    'pb_sector_median': 0.3,
                    'pb_peer_average': 0.3,
                    'ev_ebitda_sector_median': 0.0,
                    'ev_ebitda_peer_average': 0.0,
                    'ps_sector_median': 0.0,
                    'ps_peer_average': 0.0
                }
            elif sector == "Technology":
                weights = {
                    'pe_sector_median': 0.15,
                    'pe_peer_average': 0.15,
                    'ev_ebitda_sector_median': 0.2,
                    'ev_ebitda_peer_average': 0.2,
                    'ps_sector_median': 0.15,
                    'ps_peer_average': 0.15,
                    'pb_sector_median': 0.0,
                    'pb_peer_average': 0.0
                }
            elif sector == "Real Estate":
                weights = {
                    'pe_sector_median': 0.0,
                    'pe_peer_average': 0.0,
                    'ev_ebitda_sector_median': 0.3,
                    'ev_ebitda_peer_average': 0.3,
                    'ps_sector_median': 0.0,
                    'ps_peer_average': 0.0,
                    'pb_sector_median': 0.2,
                    'pb_peer_average': 0.2
                }
            else:
                # Default weights
                weights = {
                    'pe_sector_median': 0.2,
                    'pe_peer_average': 0.2,
                    'ev_ebitda_sector_median': 0.2,
                    'ev_ebitda_peer_average': 0.2,
                    'ps_sector_median': 0.1,
                    'ps_peer_average': 0.1,
                    'pb_sector_median': 0.0,
                    'pb_peer_average': 0.0
                }

            # Calculate weighted average
            weighted_sum = 0
            weight_sum = 0

            for method, weight in weights.items():
                if method in valuations and valuations[method] is not None:
                    weighted_sum += valuations[method] * weight
                    weight_sum += weight

            if weight_sum > 0:
                weighted_average = weighted_sum / weight_sum
            else:
                weighted_average = None

            # Apply margin of safety
            params = self._get_dcf_parameters(sector)
            safety_margin = params['default_margin_of_safety']
            conservative_value = weighted_average * (1 - safety_margin) if weighted_average else None

            return {
                'company': ticker,
                'method': 'relative_valuation',
                'company_metrics': company_metrics,
                'sector_multiples': sector_multiples,
                'peer_multiples': peer_multiples,
                'valuations': valuations,
                'weighted_average': weighted_average,
                'conservative_value': conservative_value,
                'safety_margin': safety_margin,
                'sector': sector,
                'peers': peers
            }

        except Exception as e:
            logger.error(f"Error in relative valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'relative_valuation',
                'weighted_average': None,
                'conservative_value': None,
                'error': str(e)
            }

    def _get_sector_multiples(self, sector: str) -> Dict[str, float]:
        """Get median multiples for a given sector from benchmark data"""
        try:
            # This would typically come from a database or external source
            # For simplicity, using hardcoded values based on typical ranges

            sector_multiples = {
                "Technology": {
                    'median_pe': 25.0,
                    'median_ps': 5.0,
                    'median_pb': 5.0,
                    'median_ev_to_ebitda': 15.0
                },
                "Financials": {
                    'median_pe': 12.0,
                    'median_ps': 3.0,
                    'median_pb': 1.2,
                    'median_ev_to_ebitda': 10.0
                },
                "Healthcare": {
                    'median_pe': 20.0,
                    'median_ps': 4.0,
                    'median_pb': 4.0,
                    'median_ev_to_ebitda': 12.0
                },
                "Consumer Discretionary": {
                    'median_pe': 18.0,
                    'median_ps': 1.5,
                    'median_pb': 3.0,
                    'median_ev_to_ebitda': 10.0
                },
                "Consumer Staples": {
                    'median_pe': 20.0,
                    'median_ps': 2.0,
                    'median_pb': 4.5,
                    'median_ev_to_ebitda': 12.0
                },
                "Energy": {
                    'median_pe': 10.0,
                    'median_ps': 1.0,
                    'median_pb': 1.5,
                    'median_ev_to_ebitda': 6.0
                },
                "Industrials": {
                    'median_pe': 16.0,
                    'median_ps': 1.5,
                    'median_pb': 3.0,
                    'median_ev_to_ebitda': 9.0
                },
                "Materials": {
                    'median_pe': 14.0,
                    'median_ps': 1.3,
                    'median_pb': 2.0,
                    'median_ev_to_ebitda': 8.0
                },
                "Real Estate": {
                    'median_pe': 15.0,
                    'median_ps': 6.0,
                    'median_pb': 1.8,
                    'median_ev_to_ebitda': 18.0
                },
                "Communication Services": {
                    'median_pe': 18.0,
                    'median_ps': 3.0,
                    'median_pb': 3.5,
                    'median_ev_to_ebitda': 8.0
                },
                "Utilities": {
                    'median_pe': 16.0,
                    'median_ps': 2.5,
                    'median_pb': 1.8,
                    'median_ev_to_ebitda': 10.0
                }
            }

            # Return multiples for the specified sector, or default if not found
            return sector_multiples.get(sector, {
                'median_pe': 15.0,
                'median_ps': 2.0,
                'median_pb': 2.5,
                'median_ev_to_ebitda': 10.0
            })

        except Exception as e:
            logger.error(f"Error getting sector multiples for {sector}: {e}")
            return {}

    def _calculate_multiples(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key multiples for a company based on its financial data"""
        multiples = {}

        try:
            # Extract statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            market_cap = market_data.get('market_cap')

            if not income_stmt or not balance_sheet or not market_cap:
                return multiples

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate enterprise value
            total_debt = latest_balance.get('Total Debt', 0)
            cash = latest_balance.get('Cash and Cash Equivalents', 0)
            enterprise_value = market_cap + total_debt - cash

            # Calculate multiples
            if 'Total Revenue' in latest_income.index and latest_income['Total Revenue'] > 0:
                multiples['ps_ratio'] = market_cap / latest_income['Total Revenue']
                multiples['ev_to_revenue'] = enterprise_value / latest_income['Total Revenue']

            if 'Net Income' in latest_income.index and latest_income['Net Income'] > 0:
                multiples['pe_ratio'] = market_cap / latest_income['Net Income']

            if 'EBITDA' in latest_income.index and latest_income['EBITDA'] > 0:
                multiples['ev_to_ebitda'] = enterprise_value / latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                # Estimate EBITDA if not directly available
                ebitda = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']
                if ebitda > 0:
                    multiples['ev_to_ebitda'] = enterprise_value / ebitda

            if 'Total Stockholder Equity' in latest_balance.index and latest_balance['Total Stockholder Equity'] > 0:
                multiples['pb_ratio'] = market_cap / latest_balance['Total Stockholder Equity']

            return multiples

        except Exception as e:
            logger.error(f"Error calculating multiples: {e}")
            return multiples

    def industrial_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for industrial sector companies

        Industrial companies have unique characteristics requiring adjustments to standard DCF:
        1. Capital intensive with high fixed costs and operational leverage
        2. Cyclical business aligned with economic cycles
        3. High capex requirements for maintenance and growth
        4. Focus on operational efficiency and capacity utilization
        5. Often have significant debt levels

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with industrial sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For industrial companies, forecast-driven DCF works well
            dcf_result = self.forecast_driven_dcf(ticker, financial_data, "Industrials")

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Calculate industrial-specific metrics
            industrial_metrics = {}

            # Asset utilization metrics - critical for industrial companies
            if income_stmt is not None and balance_sheet is not None and not income_stmt.empty and not balance_sheet.empty:
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]

                # Asset turnover
                if 'Total Revenue' in latest_income.index and 'Total Assets' in latest_balance.index:
                    revenue = latest_income['Total Revenue']
                    assets = latest_balance['Total Assets']
                    if assets > 0:
                        asset_turnover = revenue / assets
                        industrial_metrics['asset_turnover'] = asset_turnover

                # ROIC (Return on Invested Capital) - important for industrial companies
                if 'Operating Income' in latest_income.index:
                    ebit = latest_income['Operating Income']
                    # Calculate invested capital (equity + debt - cash)
                    equity = latest_balance.get('Total Stockholder Equity', 0)
                    debt = latest_balance.get('Total Debt', 0)
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)

                    invested_capital = equity + debt - cash

                    if invested_capital > 0:
                        # Adjust for taxes (assume 25% tax rate if not available)
                        tax_rate = 0.25
                        if 'Income Tax Expense' in latest_income.index and 'Income Before Tax' in latest_income.index:
                            if latest_income['Income Before Tax'] > 0:
                                tax_rate = latest_income['Income Tax Expense'] / latest_income['Income Before Tax']

                        nopat = ebit * (1 - tax_rate)
                        roic = nopat / invested_capital
                        industrial_metrics['roic'] = roic

                # Capex efficiency - industrial companies are capital intensive
            if cash_flow is not None and income_stmt is not None and not cash_flow.empty and not income_stmt.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Capital Expenditure' in latest_cash_flow.index and 'Total Revenue' in latest_income.index:
                    capex = abs(latest_cash_flow['Capital Expenditure'])
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        capex_to_revenue = capex / revenue
                        industrial_metrics['capex_to_revenue'] = capex_to_revenue

                    # Capex to Depreciation ratio (maintenance vs. growth)
                    if 'Depreciation & Amortization' in latest_income.index:
                        depreciation = latest_income['Depreciation & Amortization']

                        if depreciation > 0:
                            capex_to_depreciation = capex / depreciation
                            industrial_metrics['capex_to_depreciation'] = capex_to_depreciation

                            # Classify capex strategy
                            if capex_to_depreciation > 1.5:
                                industrial_metrics['capex_strategy'] = "Growth/expansion"
                            elif capex_to_depreciation > 1.1:
                                industrial_metrics['capex_strategy'] = "Balanced growth & maintenance"
                            elif capex_to_depreciation > 0.9:
                                industrial_metrics['capex_strategy'] = "Primarily maintenance"
                            else:
                                industrial_metrics['capex_strategy'] = "Underinvestment"

                # Debt metrics - industrial companies often have significant debt
            if balance_sheet is not None and income_stmt is not None and not balance_sheet.empty and not income_stmt.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Total Debt' in latest_balance.index and 'EBITDA' in latest_income.index:
                    debt = latest_balance['Total Debt']
                    ebitda = latest_income['EBITDA']

                    if ebitda > 0:
                        debt_to_ebitda = debt / ebitda
                        industrial_metrics['debt_to_ebitda'] = debt_to_ebitda

                        # Classify debt level
                        if debt_to_ebitda > 4.0:
                            industrial_metrics['debt_level'] = "High"
                        elif debt_to_ebitda > 2.5:
                            industrial_metrics['debt_level'] = "Moderate"
                        else:
                            industrial_metrics['debt_level'] = "Low"

                # Interest coverage ratio
                if 'Operating Income' in latest_income.index and 'Interest Expense' in latest_income.index:
                    ebit = latest_income['Operating Income']
                    interest_expense = abs(latest_income['Interest Expense'])

                    if interest_expense > 0:
                        interest_coverage = ebit / interest_expense
                        industrial_metrics['interest_coverage'] = interest_coverage

                # Operational efficiency - critical for industrial companies
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                if 'Gross Profit' in latest_income.index and 'Total Revenue' in latest_income.index:
                    gross_profit = latest_income['Gross Profit']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        gross_margin = gross_profit / revenue
                        industrial_metrics['gross_margin'] = gross_margin

                if 'Operating Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                    operating_income = latest_income['Operating Income']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        operating_margin = operating_income / revenue
                        industrial_metrics['operating_margin'] = operating_margin

                # Add industrial metrics to the result
            dcf_result['industrial_metrics'] = industrial_metrics

            return dcf_result

        except Exception as e:
            logger.error(f"Error in industrial sector DCF for {ticker}: {e}")
            return self.forecast_driven_dcf(ticker, financial_data, "Industrials")

    def healthcare_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for healthcare sector companies

        Healthcare companies have unique characteristics requiring adjustments to standard DCF:
        1. High R&D expenditures with uncertain returns and long development timelines
        2. Patent protection creates temporary monopolies but with defined lifespans
        3. Regulatory approvals create binary outcomes for clinical-stage companies
        4. Different business models within healthcare (pharma, biotech, medical devices, services)
        5. Demographic trends and public policy heavily influence growth

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with healthcare sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For healthcare companies, multi-stage DCF works well due to patent cycles
            dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Healthcare")

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get company info
            company_info = financial_data.get('company_info', {})

            # Calculate healthcare-specific metrics
            healthcare_metrics = {}

            # R&D intensity - critical for healthcare, especially pharma/biotech
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                if 'Research and Development' in latest_income.index and 'Total Revenue' in latest_income.index:
                    rd_expense = latest_income['Research and Development']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        rd_to_revenue = rd_expense / revenue
                        healthcare_metrics['rd_to_revenue'] = rd_to_revenue

                        # Classify R&D intensity
                        if rd_to_revenue > 0.20:
                            healthcare_metrics['rd_intensity'] = "Very high (biotech/early-stage)"
                        elif rd_to_revenue > 0.15:
                            healthcare_metrics['rd_intensity'] = "High (pharmaceutical/research focused)"
                        elif rd_to_revenue > 0.10:
                            healthcare_metrics['rd_intensity'] = "Moderate (established pharma)"
                        elif rd_to_revenue > 0.05:
                            healthcare_metrics['rd_intensity'] = "Low (medical devices/services)"
                        else:
                            healthcare_metrics['rd_intensity'] = "Very low (healthcare services)"

            # Growth metrics
            if income_stmt is not None and income_stmt.shape[1] >= 3:
                # Calculate revenue CAGR over last 3 years if available
                if 'Total Revenue' in income_stmt.index:
                    latest_revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                    oldest_revenue = income_stmt.iloc[
                        income_stmt.index.get_loc('Total Revenue'), min(2, income_stmt.shape[1] - 1)]

                    if oldest_revenue > 0:
                        # Calculate compound annual growth rate
                        years = min(3, income_stmt.shape[1])
                        revenue_cagr = (latest_revenue / oldest_revenue) ** (1 / years) - 1
                        healthcare_metrics['revenue_cagr_3yr'] = revenue_cagr

            # Profitability metrics - vary widely in healthcare
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                if 'Gross Profit' in latest_income.index and 'Total Revenue' in latest_income.index:
                    gross_profit = latest_income['Gross Profit']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        gross_margin = gross_profit / revenue
                        healthcare_metrics['gross_margin'] = gross_margin

                if 'Operating Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                    operating_income = latest_income['Operating Income']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        operating_margin = operating_income / revenue
                        healthcare_metrics['operating_margin'] = operating_margin

                        # Early-stage biotech often has negative operating margins
                        if operating_margin < 0:
                            healthcare_metrics['development_stage'] = "Pre-profit (R&D phase)"
                        elif operating_margin < 0.1:
                            healthcare_metrics['development_stage'] = "Early commercialization"
                        elif operating_margin < 0.2:
                            healthcare_metrics['development_stage'] = "Established commercial"
                        else:
                            healthcare_metrics['development_stage'] = "Mature franchise"

            # Cash runway - important for pre-profit healthcare companies
            if cash_flow is not None and balance_sheet is not None and not cash_flow.empty and not balance_sheet.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]

                if 'Operating Cash Flow' in latest_cash_flow.index and 'Cash and Cash Equivalents' in latest_balance.index:
                    operating_cf = latest_cash_flow['Operating Cash Flow']
                    cash = latest_balance['Cash and Cash Equivalents']

                    # For companies with negative cash flow, calculate runway
                    if operating_cf < 0:
                        cash_burn_rate = abs(operating_cf)
                        cash_runway_years = cash / cash_burn_rate
                        healthcare_metrics['cash_runway_years'] = cash_runway_years

                        if cash_runway_years < 1:
                            healthcare_metrics['funding_risk'] = "Very high - needs immediate financing"
                        elif cash_runway_years < 2:
                            healthcare_metrics['funding_risk'] = "High - near-term financing likely"
                        elif cash_runway_years < 3:
                            healthcare_metrics['funding_risk'] = "Moderate - has time to achieve milestones"
                        else:
                            healthcare_metrics['funding_risk'] = "Low - well-funded"

            # Add healthcare metrics to the result
            dcf_result['healthcare_metrics'] = healthcare_metrics

            return dcf_result

        except Exception as e:
            logger.error(f"Error in healthcare sector DCF for {ticker}: {e}")
            return self.multi_stage_dcf_valuation(ticker, financial_data, "Healthcare")

    def communications_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for communications sector companies

        Communications companies have unique characteristics requiring adjustments to standard DCF:
        1. High fixed costs with infrastructure and network investments
        2. Recurring revenue models with subscriber bases
        3. High customer acquisition costs but low marginal costs
        4. Regulatory environment that can affect pricing and competition
        5. Technology shifts creating both opportunities and risks

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with communications sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For communications companies, multi-stage DCF works well
            dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Communication Services")

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Calculate communications-specific metrics
            comm_metrics = {}

            # Infrastructure intensity
            if balance_sheet is not None and income_stmt is not None and not balance_sheet.empty and not income_stmt.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Property Plant and Equipment' in latest_balance.index and 'Total Revenue' in latest_income.index:
                    ppe = latest_balance['Property Plant and Equipment']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        infrastructure_intensity = ppe / revenue
                        comm_metrics['infrastructure_intensity'] = infrastructure_intensity

                        # Classify infrastructure model
                        if infrastructure_intensity > 1.5:
                            comm_metrics['infrastructure_model'] = "Heavy infrastructure (telecom/cable)"
                        elif infrastructure_intensity > 0.8:
                            comm_metrics['infrastructure_model'] = "Moderate infrastructure"
                        elif infrastructure_intensity > 0.4:
                            comm_metrics['infrastructure_model'] = "Light infrastructure"
                        else:
                            comm_metrics['infrastructure_model'] = "Asset-light (digital/content)"

            # Capex intensity
            if cash_flow is not None and income_stmt is not None and not cash_flow.empty and not income_stmt.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Capital Expenditure' in latest_cash_flow.index and 'Total Revenue' in latest_income.index:
                    capex = abs(latest_cash_flow['Capital Expenditure'])
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        capex_to_revenue = capex / revenue
                        comm_metrics['capex_to_revenue'] = capex_to_revenue

            # Margin profile
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                if 'EBITDA' in latest_income.index and 'Total Revenue' in latest_income.index:
                    ebitda = latest_income['EBITDA']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        ebitda_margin = ebitda / revenue
                        comm_metrics['ebitda_margin'] = ebitda_margin

                # If EBITDA not available, calculate it
                elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                    operating_income = latest_income['Operating Income']
                    depreciation = latest_income['Depreciation & Amortization']
                    ebitda = operating_income + depreciation

                    if 'Total Revenue' in latest_income.index:
                        revenue = latest_income['Total Revenue']
                        if revenue > 0:
                            ebitda_margin = ebitda / revenue
                            comm_metrics['ebitda_margin'] = ebitda_margin

            # Return on invested capital
            if income_stmt is not None and balance_sheet is not None and not income_stmt.empty and not balance_sheet.empty:
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]

                if 'Operating Income' in latest_income.index:
                    ebit = latest_income['Operating Income']

                    # Calculate invested capital
                    equity = latest_balance.get('Total Stockholder Equity', 0)
                    debt = latest_balance.get('Total Debt', 0)
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)

                    invested_capital = equity + debt - cash

                    if invested_capital > 0:
                        # Adjust for taxes
                        tax_rate = 0.25  # Assume 25% tax rate if not available
                        if 'Income Tax Expense' in latest_income.index and 'Income Before Tax' in latest_income.index:
                            if latest_income['Income Before Tax'] > 0:
                                tax_rate = latest_income['Income Tax Expense'] / latest_income['Income Before Tax']

                        nopat = ebit * (1 - tax_rate)
                        roic = nopat / invested_capital
                        comm_metrics['roic'] = roic

            # Add communications metrics to the result
            dcf_result['communications_metrics'] = comm_metrics

            return dcf_result

        except Exception as e:
            logger.error(f"Error in communications sector DCF for {ticker}: {e}")
            return self.multi_stage_dcf_valuation(ticker, financial_data, "Communication Services")

    def materials_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized DCF for materials sector companies

        Materials companies have unique characteristics requiring adjustments to standard DCF:
        1. Highly cyclical with commodity price exposure
        2. Capital intensive with high fixed costs
        3. Dependent on global economic activity and industrial production
        4. Environmental regulations create challenges and opportunities
        5. Consolidation often drives industry structure

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with materials sector DCF valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # For materials companies, scenario analysis works well due to cyclicality
            dcf_result = self.scenario_analysis_dcf(ticker, financial_data, "Materials")

            # Extract key financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Calculate materials-specific metrics
            materials_metrics = {}

            # Margin profiles through the cycle
            if income_stmt is not None and not income_stmt.empty:
                years = min(income_stmt.shape[1], 5)  # Look at up to 5 years of data

                if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                    gross_margins = []
                    for i in range(years):
                        if i < income_stmt.shape[1]:
                            revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), i]
                            gross_profit = income_stmt.iloc[income_stmt.index.get_loc('Gross Profit'), i]

                            if revenue > 0:
                                gross_margin = gross_profit / revenue
                                gross_margins.append(gross_margin)

                    if gross_margins:
                        materials_metrics['avg_gross_margin'] = sum(gross_margins) / len(gross_margins)
                        materials_metrics['min_gross_margin'] = min(gross_margins)
                        materials_metrics['max_gross_margin'] = max(gross_margins)
                        materials_metrics['gross_margin_volatility'] = max(gross_margins) - min(gross_margins)

                if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                    operating_margins = []
                    for i in range(years):
                        if i < income_stmt.shape[1]:
                            revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), i]
                            operating_income = income_stmt.iloc[income_stmt.index.get_loc('Operating Income'), i]

                            if revenue > 0:
                                operating_margin = operating_income / revenue
                                operating_margins.append(operating_margin)

                    if operating_margins:
                        materials_metrics['avg_operating_margin'] = sum(operating_margins) / len(operating_margins)
                        materials_metrics['min_operating_margin'] = min(operating_margins)
                        materials_metrics['max_operating_margin'] = max(operating_margins)
                        materials_metrics['operating_margin_volatility'] = max(operating_margins) - min(
                            operating_margins)

                        # Determine cyclicality based on margin volatility
                        if materials_metrics.get('operating_margin_volatility', 0) > 0.15:
                            materials_metrics['cyclicality'] = "Highly cyclical"
                        elif materials_metrics.get('operating_margin_volatility', 0) > 0.10:
                            materials_metrics['cyclicality'] = "Moderately cyclical"
                        elif materials_metrics.get('operating_margin_volatility', 0) > 0.05:
                            materials_metrics['cyclicality'] = "Somewhat cyclical"
                        else:
                            materials_metrics['cyclicality'] = "Relatively stable"

            # Capital intensity
            if balance_sheet is not None and income_stmt is not None and not balance_sheet.empty and not income_stmt.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Property Plant and Equipment' in latest_balance.index and 'Total Revenue' in latest_income.index:
                    ppe = latest_balance['Property Plant and Equipment']
                    revenue = latest_income['Total Revenue']

                    if revenue > 0:
                        capital_intensity = ppe / revenue
                        materials_metrics['capital_intensity'] = capital_intensity

                        # Classify capital intensity
                        if capital_intensity > 1.0:
                            materials_metrics['capital_intensity_level'] = "Very high"
                        elif capital_intensity > 0.6:
                            materials_metrics['capital_intensity_level'] = "High"
                        elif capital_intensity > 0.3:
                            materials_metrics['capital_intensity_level'] = "Moderate"
                        else:
                            materials_metrics['capital_intensity_level'] = "Low"

            # Return on invested capital (ROIC)
            if income_stmt is not None and balance_sheet is not None and not income_stmt.empty and not balance_sheet.empty:
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]

                if 'Operating Income' in latest_income.index:
                    ebit = latest_income['Operating Income']

                    # Calculate invested capital
                    equity = latest_balance.get('Total Stockholder Equity', 0)
                    debt = latest_balance.get('Total Debt', 0)
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)

                    invested_capital = equity + debt - cash

                    if invested_capital > 0:
                        # Adjust for taxes
                        tax_rate = 0.25  # Assume 25% tax rate if not available
                        if 'Income Tax Expense' in latest_income.index and 'Income Before Tax' in latest_income.index:
                            if latest_income['Income Before Tax'] > 0:
                                tax_rate = latest_income['Income Tax Expense'] / latest_income['Income Before Tax']

                        nopat = ebit * (1 - tax_rate)
                        roic = nopat / invested_capital
                        materials_metrics['roic'] = roic

                        # Calculate historical ROIC if available
                        if income_stmt.shape[1] >= 3 and balance_sheet.shape[1] >= 3:
                            historical_roic = []
                            for i in range(min(3, income_stmt.shape[1])):
                                if i < income_stmt.shape[1] and i < balance_sheet.shape[1]:
                                    historical_ebit = income_stmt.iloc[
                                        income_stmt.index.get_loc('Operating Income'), i]
                                    historical_equity = balance_sheet.iloc[
                                        balance_sheet.index.get_loc('Total Stockholder Equity'), i]
                                    historical_debt = balance_sheet.iloc[balance_sheet.index.get_loc(
                                        'Total Debt'), i] if 'Total Debt' in balance_sheet.index else 0
                                    historical_cash = balance_sheet.iloc[balance_sheet.index.get_loc(
                                        'Cash and Cash Equivalents'), i] if 'Cash and Cash Equivalents' in balance_sheet.index else 0

                                    historical_invested_capital = historical_equity + historical_debt - historical_cash

                                    if historical_invested_capital > 0:
                                        historical_nopat = historical_ebit * (1 - tax_rate)
                                        historical_roic.append(historical_nopat / historical_invested_capital)

                            if historical_roic:
                                materials_metrics['avg_roic'] = sum(historical_roic) / len(historical_roic)
                                materials_metrics['roic_volatility'] = max(historical_roic) - min(historical_roic)

            # Debt metrics
            if balance_sheet is not None and income_stmt is not None and not balance_sheet.empty and not income_stmt.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                latest_income = income_stmt.iloc[:, 0]

                if 'Total Debt' in latest_balance.index:
                    debt = latest_balance['Total Debt']

                    if 'EBITDA' in latest_income.index:
                        ebitda = latest_income['EBITDA']

                        if ebitda > 0:
                            debt_to_ebitda = debt / ebitda
                            materials_metrics['debt_to_ebitda'] = debt_to_ebitda

                    elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                        # Calculate EBITDA if not available directly
                        operating_income = latest_income['Operating Income']
                        depreciation = latest_income['Depreciation & Amortization']

                        ebitda = operating_income + depreciation

                        if ebitda > 0:
                            debt_to_ebitda = debt / ebitda
                            materials_metrics['debt_to_ebitda'] = debt_to_ebitda

            # Add materials metrics to the result
            dcf_result['materials_metrics'] = materials_metrics

            return dcf_result

        except Exception as e:
            logger.error(f"Error in materials sector DCF for {ticker}: {e}")
            return self.scenario_analysis_dcf(ticker, financial_data, "Materials")