import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from utils.data_loader import DataLoader
from valuation.base_valuation import BaseValuation
from valuation.dcf_models import AdvancedDCFValuation, SectorSpecificDCF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('energy_sector')


class EnergySectorValuation(SectorSpecificDCF):
    """
    Specialized valuation models for energy sector companies (oil & gas, utilities, renewables)

    Energy sector valuation requires different approaches due to:
    1. High capital intensity and long project lifecycles
    2. Commodity price sensitivity and cyclical nature
    3. Reserve-based valuation for E&P companies
    4. Regulatory environments for utilities
    5. Transition risks related to climate change and renewables
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize energy sector valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized EnergySectorValuation")

        # Energy sector specific parameters
        self.commodity_price_cycles = {
            'oil': {
                'current_price': 75.0,  # USD per barrel
                'long_term_price': 65.0,  # USD per barrel
                'volatility': 0.25,  # Annual price volatility
                'mean_reversion': 0.15,  # Speed of mean reversion
            },
            'natural_gas': {
                'current_price': 3.5,  # USD per MMBtu
                'long_term_price': 3.0,  # USD per MMBtu
                'volatility': 0.30,  # Annual price volatility
                'mean_reversion': 0.20,  # Speed of mean reversion
            },
            'electricity': {
                'current_price': 0.12,  # USD per kWh
                'long_term_price': 0.11,  # USD per kWh
                'volatility': 0.10,  # Annual price volatility
                'mean_reversion': 0.10,  # Speed of mean reversion
            }
        }

        # Industry-specific parameters
        self.reserve_replacement_cost = {
            'oil': 15.0,  # USD per barrel
            'natural_gas': 1.2,  # USD per MCF
            'coal': 20.0,  # USD per ton
        }

        # Capital expenditure parameters
        self.maintenance_capex_percent = 0.10  # 10% of assets for maintenance
        self.transition_capex_factor = 0.02  # Additional 2% for energy transition

    def value_energy_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master method to value an energy sector company, selecting the most appropriate model
        based on company sub-sector and business model
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine energy sub-sector and business model
            sub_sector, business_model = self._determine_energy_profile(ticker, financial_data)

            # Select valuation method based on sub-sector and business model
            if sub_sector == "oil_gas" and business_model == "upstream":
                result = self.value_upstream_company(ticker, financial_data)
            elif sub_sector == "oil_gas" and business_model == "integrated":
                result = self.value_integrated_oil_gas(ticker, financial_data)
            elif sub_sector == "oil_gas" and business_model == "midstream":
                result = self.value_midstream_company(ticker, financial_data)
            elif sub_sector == "utilities":
                result = self.value_utility_company(ticker, financial_data)
            elif sub_sector == "renewables":
                result = self.value_renewable_company(ticker, financial_data)
            else:
                # Use general approach for energy companies
                result = self.energy_sector_dcf(ticker, financial_data)

            # Add sub-sector and business model information to result
            result['sub_sector'] = sub_sector
            result['business_model'] = business_model

            return result

        except Exception as e:
            logger.error(f"Error valuing energy company {ticker}: {e}")
            # Fall back to standard approach
            return self.energy_sector_dcf(ticker, financial_data)

    def value_upstream_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an upstream oil & gas company (E&P) using specialized models including:
        1. Reserve-based NAV (Net Asset Value)
        2. Cyclical-adjusted DCF
        3. EV/Reserves and EV/Production multiples
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key upstream metrics
            upstream_metrics = self._calculate_upstream_metrics(financial_data)

            # 1. Perform reserve-based NAV valuation
            nav_result = self._reserve_based_nav(ticker, financial_data, upstream_metrics)

            # 2. Perform cyclical-adjusted DCF
            dcf_result = self._cyclical_upstream_dcf(ticker, financial_data, upstream_metrics)

            # 3. Perform multiples valuation
            multiples_result = self._upstream_multiples_valuation(ticker, financial_data, upstream_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'nav': 0.4,  # Base weight for NAV
                'dcf': 0.3,  # Base weight for DCF
                'multiples': 0.3  # Base weight for multiples
            }

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.8  # 20% margin of safety for cyclical industry
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'upstream_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'nav_valuation': nav_result,
                'dcf_valuation': dcf_result,
                'multiples_valuation': multiples_result,
                'upstream_metrics': upstream_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in upstream company valuation for {ticker}: {e}")
            # Fall back to general energy sector DCF
            return self.energy_sector_dcf(ticker, financial_data)

    def value_integrated_oil_gas(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an integrated oil & gas company using specialized models including:
        1. Sum-of-the-parts valuation (SOTP)
        2. Cyclical-adjusted DCF
        3. Multiples valuation with segment breakdown
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key integrated metrics
            integrated_metrics = self._calculate_integrated_metrics(financial_data)

            # 1. Perform sum-of-the-parts valuation
            sotp_result = self._sum_of_the_parts_valuation(ticker, financial_data, integrated_metrics)

            # 2. Perform cyclical-adjusted DCF
            dcf_result = self._cyclical_integrated_dcf(ticker, financial_data, integrated_metrics)

            # 3. Perform multiples valuation
            multiples_result = self._integrated_multiples_valuation(ticker, financial_data, integrated_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'sotp': 0.4,  # Base weight for SOTP
                'dcf': 0.3,  # Base weight for DCF
                'multiples': 0.3  # Base weight for multiples
            }

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety for integrated companies
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'integrated_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'sotp_valuation': sotp_result,
                'dcf_valuation': dcf_result,
                'multiples_valuation': multiples_result,
                'integrated_metrics': integrated_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in integrated oil & gas valuation for {ticker}: {e}")
            # Fall back to general energy sector DCF
            return self.energy_sector_dcf(ticker, financial_data)

    def value_midstream_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a midstream energy company (pipelines, storage, processing) using specialized models including:
        1. Dividend Discount Model (DDM)
        2. DCF with long-term contracts
        3. EV/EBITDA multiples
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key midstream metrics
            midstream_metrics = self._calculate_midstream_metrics(financial_data)

            # 1. Perform dividend discount model
            ddm_result = self._midstream_ddm(ticker, financial_data, midstream_metrics)

            # 2. Perform DCF with long-term contracts
            dcf_result = self._midstream_dcf(ticker, financial_data, midstream_metrics)

            # 3. Perform multiples valuation
            multiples_result = self._midstream_multiples_valuation(ticker, financial_data, midstream_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'ddm': 0.4,  # Base weight for DDM
                'dcf': 0.3,  # Base weight for DCF
                'multiples': 0.3  # Base weight for multiples
            }

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'midstream_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'ddm_valuation': ddm_result,
                'dcf_valuation': dcf_result,
                'multiples_valuation': multiples_result,
                'midstream_metrics': midstream_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in midstream company valuation for {ticker}: {e}")
            # Fall back to general energy sector DCF
            return self.energy_sector_dcf(ticker, financial_data)

    def value_utility_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a utility company using specialized models including:
        1. Dividend Discount Model (DDM)
        2. Regulated asset base (RAB) valuation
        3. DCF with regulatory considerations
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key utility metrics
            utility_metrics = self._calculate_utility_metrics(financial_data)

            # 1. Perform dividend discount model
            ddm_result = self._utility_ddm(ticker, financial_data, utility_metrics)

            # 2. Perform regulated asset base valuation
            rab_result = self._regulated_asset_base_valuation(ticker, financial_data, utility_metrics)

            # 3. Perform DCF with regulatory considerations
            dcf_result = self._utility_dcf(ticker, financial_data, utility_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'ddm': 0.4,  # Base weight for DDM
                'rab': 0.3,  # Base weight for RAB
                'dcf': 0.3  # Base weight for DCF
            }

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.9  # 10% margin of safety for utilities (lower than other energy)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'utility_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'ddm_valuation': ddm_result,
                'rab_valuation': rab_result,
                'dcf_valuation': dcf_result,
                'utility_metrics': utility_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in utility company valuation for {ticker}: {e}")
            # Fall back to general energy sector DCF
            return self.energy_sector_dcf(ticker, financial_data)

    def value_renewable_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a renewable energy company using specialized models including:
        1. Project-based DCF
        2. Growth-adjusted multiples
        3. Real options valuation
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key renewable metrics
            renewable_metrics = self._calculate_renewable_metrics(financial_data)

            # 1. Perform project-based DCF
            dcf_result = self._renewable_project_dcf(ticker, financial_data, renewable_metrics)

            # 2. Perform growth-adjusted multiples valuation
            multiples_result = self._renewable_multiples_valuation(ticker, financial_data, renewable_metrics)

            # 3. Perform real options valuation
            options_result = self._renewable_options_valuation(ticker, financial_data, renewable_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'dcf': 0.4,  # Base weight for DCF
                'multiples': 0.4,  # Base weight for multiples
                'options': 0.2  # Base weight for real options
            }

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.8  # 20% margin of safety for growth-focused renewables
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'renewable_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'dcf_valuation': dcf_result,
                'multiples_valuation': multiples_result,
                'options_valuation': options_result,
                'renewable_metrics': renewable_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in renewable company valuation for {ticker}: {e}")
            # Fall back to general energy sector DCF
            return self.energy_sector_dcf(ticker, financial_data)

    def energy_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        General DCF model for energy companies with commodity price cycle adjustments
        and capital intensity considerations
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get parameters based on sector if available
            params = self._get_dcf_parameters("Energy")

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for energy DCF valuation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Determine commodity exposure for price cycle modeling
            commodity_exposure = self._determine_commodity_exposure(ticker, financial_data)

            # Get latest revenue, earnings, and assets
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            revenue = latest_income['Total Revenue'] if 'Total Revenue' in latest_income.index else None
            ebitda = None

            # Try to find EBITDA directly
            if 'EBITDA' in latest_income.index:
                ebitda = latest_income['EBITDA']
            else:
                # Calculate from components
                if 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                    ebitda = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']

            # Determine growth phases based on commodity cycle
            long_term_growth = params['terminal_growth_rate']
            initial_growth_rate = self._estimate_energy_growth_rate(historical_fcf, commodity_exposure)

            # Define growth phases with commodity cycle consideration
            forecast_years = params['forecast_years']

            # Price cycle modeling
            oil_price_cycle = [1.05, 1.10, 0.95, 0.85, 0.90, 1.05, 1.10, 0.95]

            # Calculate discount rate (WACC) with energy specifics
            discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, commodity_exposure)

            # Starting FCF (most recent)
            last_fcf = historical_fcf.iloc[0]

            # Forecast cash flows with commodity cycle adjustments
            forecasted_cash_flows = []
            forecasted_revenues = []

            current_fcf = last_fcf
            current_revenue = revenue

            for year in range(1, forecast_years + 1):
                # Apply commodity cycle
                cycle_index = (year - 1) % len(oil_price_cycle)
                commodity_factor = oil_price_cycle[cycle_index]

                # Commodity-adjusted growth rate
                adjusted_growth = initial_growth_rate * commodity_factor

                # Revenue growth
                if current_revenue is not None:
                    current_revenue = current_revenue * (1 + adjusted_growth)
                    forecasted_revenues.append(current_revenue)

                # Cash flow forecast
                # Energy companies tend to have higher capex in upcycles
                if commodity_factor > 1:
                    # During upcycle, FCF growth is dampened by higher capex
                    fcf_growth = adjusted_growth * 0.8
                else:
                    # During downcycle, capex is reduced, so FCF doesn't fall as much
                    fcf_growth = adjusted_growth * 1.2

                current_fcf = current_fcf * (1 + fcf_growth)
                forecasted_cash_flows.append(current_fcf)

            # Calculate terminal value with conservative long-term growth
            final_fcf = forecasted_cash_flows[-1]
            # Use average of last 3 years to smooth out cycle effects
            if len(forecasted_cash_flows) >= 3:
                final_fcf = sum(forecasted_cash_flows[-3:]) / 3

            terminal_value = final_fcf * (1 + long_term_growth) / (discount_rate - long_term_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding is not None and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            # Apply margin of safety
            safety_margin = params['default_margin_of_safety'] * 1.2  # Higher margin for cyclical industry
            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            return {
                'company': ticker,
                'method': 'energy_sector_dcf',
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': discount_rate,
                'initial_growth_rate': initial_growth_rate,
                'terminal_growth': long_term_growth,
                'forecast_years': forecast_years,
                'commodity_exposure': commodity_exposure,
                'forecasted_cash_flows': forecasted_cash_flows,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'historical_fcf': historical_fcf.to_dict(),
                'net_debt': net_debt,
                'safety_margin': safety_margin
            }

        except Exception as e:
            logger.error(f"Error in energy sector DCF for {ticker}: {e}")
            # Fall back to standard DCF
            return self.discounted_cash_flow_valuation(ticker, financial_data, "Energy")

    # Helper methods for energy sector valuation

    def _determine_energy_profile(self, ticker: str, financial_data: Dict[str, Any]) -> Tuple[str, str]:
        """Determine the sub-sector and business model of an energy company"""
        try:
            # Extract company info and financial statements
            company_info = financial_data.get('company_info', {})
            income_stmt = financial_data.get('income_statement')

            # Try to get industry from company info
            industry = company_info.get('industry', '').lower()
            sector = company_info.get('sector', '').lower()

            # Default classifications
            sub_sector = "oil_gas"  # Default to oil & gas
            business_model = "integrated"  # Default to integrated

            # Determine sub-sector based on industry description
            if any(term in industry for term in ['utility', 'electric', 'gas distribution', 'water']):
                sub_sector = "utilities"
                business_model = "regulated"
            elif any(term in industry for term in ['oil', 'gas', 'petroleum', 'drilling']):
                sub_sector = "oil_gas"

                # Determine business model for oil & gas
                if any(term in industry for term in ['exploration', 'production', 'e&p', 'drilling']):
                    business_model = "upstream"
                elif any(term in industry for term in ['pipeline', 'midstream', 'storage', 'transportation']):
                    business_model = "midstream"
                elif any(term in industry for term in ['refining', 'marketing']):
                    business_model = "downstream"
                else:
                    business_model = "integrated"
            elif any(term in industry for term in ['renewable', 'solar', 'wind', 'alternative']):
                sub_sector = "renewables"
                business_model = "independent"
            elif any(term in industry for term in ['coal', 'mining']):
                sub_sector = "coal"
                business_model = "mining"

            # Further refine based on financial metrics if available
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                # Check for utilities characteristics (high depreciation, stable margins)
                if 'Depreciation & Amortization' in latest_income.index and 'Total Revenue' in latest_income.index:
                    depreciation_ratio = latest_income['Depreciation & Amortization'] / latest_income['Total Revenue']

                    if depreciation_ratio > 0.15 and 'Operating Margin' in latest_income.index:
                        op_margin = latest_income['Operating Margin']
                        if 0.10 <= op_margin <= 0.25:  # Typical utility margins
                            sub_sector = "utilities"
                            business_model = "regulated"

            return sub_sector, business_model

        except Exception as e:
            logger.warning(f"Error determining energy profile for {ticker}: {e}")
            return "oil_gas", "integrated"  # Default values

    def _calculate_upstream_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key upstream oil & gas metrics for valuation"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get data for multiple periods
            periods = min(income_stmt.shape[1], balance_sheet.shape[1], 3)  # Use up to 3 years

            # Estimate reserves and production data
            # Note: In practice, this would come from company filings, but for this example we'll estimate
            reserves_data = {
                'Oil_Reserves_MMBbl': company_info.get('oil_reserves', 1000),  # Million barrels
                'Gas_Reserves_BCF': company_info.get('gas_reserves', 5000),  # Billion cubic feet
                'Oil_Production_MBblpd': company_info.get('oil_production', 100),  # Thousand barrels per day
                'Gas_Production_MMcfpd': company_info.get('gas_production', 500),  # Million cubic feet per day
                'Reserve_Life_Index': company_info.get('reserve_life', 10),  # Years
                'Reserve_Replacement_Ratio': company_info.get('reserve_replacement', 1.0)  # Ratio
            }

            metrics['Reserves'] = reserves_data

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]
                balance = balance_sheet.iloc[:, i]
                cf = cash_flow.iloc[:, i] if cash_flow is not None and cash_flow.shape[1] > i else None

                period_metrics = {}

                # Revenue
                if 'Total Revenue' in income.index:
                    revenue = income['Total Revenue']
                    period_metrics['Revenue'] = revenue

                # EBITDA and margins
                if 'EBITDA' in income.index:
                    ebitda = income['EBITDA']
                    period_metrics['EBITDA'] = ebitda
                    if 'Revenue' in period_metrics and period_metrics['Revenue'] > 0:
                        period_metrics['EBITDA_Margin'] = ebitda / period_metrics['Revenue']
                elif 'Operating Income' in income.index and 'Depreciation & Amortization' in income.index:
                    ebitda = income['Operating Income'] + income['Depreciation & Amortization']
                    period_metrics['EBITDA'] = ebitda
                    if 'Revenue' in period_metrics and period_metrics['Revenue'] > 0:
                        period_metrics['EBITDA_Margin'] = ebitda / period_metrics['Revenue']

                # Operating Income and Margin
                if 'Operating Income' in income.index and 'Revenue' in period_metrics:
                    operating_income = income['Operating Income']
                    period_metrics['Operating_Income'] = operating_income
                    period_metrics['Operating_Margin'] = operating_income / period_metrics['Revenue']

                # Finding and Production Costs
                if cf is not None and 'Capital Expenditure' in cf.index:
                    capex = abs(cf['Capital Expenditure'])
                    period_metrics['CapEx'] = capex
                    if 'Revenue' in period_metrics:
                        period_metrics['CapEx_to_Revenue'] = capex / period_metrics['Revenue']

                # Reserve metrics calculated from production and reserves
                # In a real-world scenario, these would be calculated from detailed production data
                daily_oil_equivalent = reserves_data['Oil_Production_MBblpd'] + (
                            reserves_data['Gas_Production_MMcfpd'] / 6)  # Convert gas to oil equivalent
                annual_production = daily_oil_equivalent * 365 / 1000  # Annual production in million boe

                # Calculate finding & development costs
                if 'CapEx' in period_metrics and annual_production > 0:
                    period_metrics['Finding_Costs_per_BOE'] = period_metrics['CapEx'] / annual_production

                # Calculate operating costs
                if 'Operating Income' in period_metrics and 'Revenue' in period_metrics and annual_production > 0:
                    operating_costs = period_metrics['Revenue'] - period_metrics['Operating_Income']
                    period_metrics['Operating_Cost_per_BOE'] = operating_costs / annual_production

                # Debt metrics
                if 'Total Debt' in balance.index:
                    debt = balance['Total Debt']
                    period_metrics['Debt'] = debt
                    if 'EBITDA' in period_metrics and period_metrics['EBITDA'] > 0:
                        period_metrics['Debt_to_EBITDA'] = debt / period_metrics['EBITDA']

                # Net Asset Value (NAV) components
                if 'Property Plant and Equipment' in balance.index:
                    ppe = balance['Property Plant and Equipment']
                    period_metrics['PPE'] = ppe
                    if 'Revenue' in period_metrics:
                        period_metrics['PPE_to_Revenue'] = ppe / period_metrics['Revenue']

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                # Revenue growth
                if 'Revenue' in metrics["Latest"] and 'Revenue' in metrics["Year-1"] and metrics["Year-1"]['Revenue'] > 0:
                    revenue_growth = (metrics["Latest"]['Revenue'] / metrics["Year-1"]['Revenue']) - 1
                    growth_metrics['Revenue_Growth'] = revenue_growth

                # EBITDA growth
                if 'EBITDA' in metrics["Latest"] and 'EBITDA' in metrics["Year-1"] and metrics["Year-1"]['EBITDA'] > 0:
                    ebitda_growth = (metrics["Latest"]['EBITDA'] / metrics["Year-1"]['EBITDA']) - 1
                    growth_metrics['EBITDA_Growth'] = ebitda_growth

                # Margin trends
                for margin in ['EBITDA_Margin', 'Operating_Margin']:
                    if margin in metrics["Latest"] and margin in metrics["Year-1"]:
                        margin_change = metrics["Latest"][margin] - metrics["Year-1"][margin]
                        growth_metrics[f'{margin}_Change'] = margin_change

                metrics["Growth"] = growth_metrics

            return metrics

            except Exception as e:
            logger.error(f"Error calculating upstream metrics: {e}")
            return {}

    def _reserve_based_nav(self, ticker: str, financial_data: Dict[str, Any],
                           upstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate net asset value based on proven reserves for an upstream oil & gas company
        This is a key valuation method for E&P companies
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get reserves data
            reserves_data = upstream_metrics.get('Reserves', {})

            # Get latest financials
            latest_balance = balance_sheet.iloc[:, 0] if balance_sheet is not None else None

            # 1. Calculate value of oil and gas reserves

            # Set commodity prices (could be retrieved from market data)
            oil_price = self.commodity_price_cycles['oil']['current_price']  # USD per barrel
            gas_price = self.commodity_price_cycles['natural_gas']['current_price']  # USD per MMBtu

            # Convert gas price from MMBtu to MCF (thousand cubic feet)
            gas_price_mcf = gas_price * 1.037  # Approximate conversion factor

            # Get reserves
            oil_reserves_mmbbl = reserves_data.get('Oil_Reserves_MMBbl', 0)  # Million barrels
            gas_reserves_bcf = reserves_data.get('Gas_Reserves_BCF', 0)  # Billion cubic feet

            # Apply discount to reserves (PV-10 concept)
            oil_reserves_discount = 0.7  # 30% discount for PV-10 and risk adjustment
            gas_reserves_discount = 0.6  # 40% discount for PV-10 and risk adjustment

            # Calculate per-unit production costs
            oil_production_cost = upstream_metrics.get('Latest', {}).get('Operating_Cost_per_BOE', 25)  # Default $25/boe
            gas_production_cost = oil_production_cost / 6  # Approximate conversion (6 MCF = 1 BOE)

            # Calculate net value of reserves
            oil_value = oil_reserves_mmbbl * (oil_price - oil_production_cost) * oil_reserves_discount
            gas_value = gas_reserves_bcf * 1000 * (
                        gas_price_mcf - gas_production_cost) * gas_reserves_discount  # BCF to MMCF

            total_reserves_value = oil_value + gas_value

            # 2. Add value of undeveloped acreage (estimate)
            # In a real valuation, this would come from detailed land holdings data
            undeveloped_acreage_value = total_reserves_value * 0.15  # Rough estimate

            # 3. Add other assets value
            other_assets_value = 0

            if latest_balance is not None:
                # Cash and equivalents
                if 'Cash and Cash Equivalents' in latest_balance.index:
                    other_assets_value += latest_balance['Cash and Cash Equivalents']

                # Other current assets (excluding inventory)
                if 'Total Current Assets' in latest_balance.index:
                    current_assets = latest_balance['Total Current Assets']
                    cash = latest_balance.get('Cash and Cash Equivalents', 0)
                    inventory = latest_balance.get('Inventory', 0)

                    other_current_assets = current_assets - cash - inventory
                    other_assets_value += other_current_assets * 0.9  # 10% discount for uncertain realization

                # Other non-current assets (excluding PPE which is captured in reserves)
                if 'Total Assets' in latest_balance.index and 'Property Plant and Equipment' in latest_balance.index:
                    total_assets = latest_balance['Total Assets']
                    ppe = latest_balance['Property Plant and Equipment']
                    current_assets = latest_balance.get('Total Current Assets', 0)

                    other_non_current_assets = total_assets - ppe - current_assets
                    other_assets_value += other_non_current_assets * 0.7  # 30% discount

            # 4. Subtract debt and other liabilities
            total_liabilities = 0

            if latest_balance is not None:
                if 'Total Liabilities' in latest_balance.index:
                    total_liabilities = latest_balance['Total Liabilities']

            # 5. Calculate net asset value
            net_asset_value = total_reserves_value + undeveloped_acreage_value + other_assets_value - total_liabilities

            # 6. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = net_asset_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'reserve_based_nav',
                'value_per_share': value_per_share,
                'net_asset_value': net_asset_value,
                'reserves_value': {
                    'oil_value': oil_value,
                    'gas_value': gas_value,
                    'total_reserves_value': total_reserves_value,
                    'oil_price': oil_price,
                    'gas_price': gas_price,
                    'oil_reserves_mmbbl': oil_reserves_mmbbl,
                    'gas_reserves_bcf': gas_reserves_bcf
                },
                'undeveloped_acreage_value': undeveloped_acreage_value,
                'other_assets_value': other_assets_value,
                'total_liabilities': total_liabilities
            }

        except Exception as e:
            logger.error(f"Error in reserve-based NAV valuation for {ticker}: {e}")
            return {
                'method': 'reserve_based_nav',
                'value_per_share': None,
                'error': str(e)
            }


    def _cyclical_upstream_dcf(self, ticker: str, financial_data: Dict[str, Any],
                               upstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a DCF valuation specifically for upstream oil & gas companies
        with adjustments for commodity price cycles and reserve depletion
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get reserves data
            reserves_data = upstream_metrics.get('Reserves', {})
            reserve_life = reserves_data.get('Reserve_Life_Index', 10)

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0] if income_stmt is not None else None

            # Get parameters specific to energy sector
            params = self._get_dcf_parameters("Energy")

            # 1. Set up oil price forecast
            # Use commodity price scenarios
            oil_price_scenarios = {
                'base': [75, 70, 68, 65, 65, 65, 65, 65, 65, 65],  # USD per barrel
                'upside': [85, 90, 85, 80, 75, 70, 70, 70, 70, 70],
                'downside': [65, 60, 55, 50, 50, 50, 50, 50, 50, 50]
            }

            # Use base scenario for main valuation
            oil_price_forecast = oil_price_scenarios['base']
            forecast_years = min(params['forecast_years'], len(oil_price_forecast))

            # 2. Get starting metrics
            if latest_income is not None and 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                # Estimate from production and prices
                daily_oil_production = reserves_data.get('Oil_Production_MBblpd', 100)  # thousand barrels per day
                daily_gas_production = reserves_data.get('Gas_Production_MMcfpd', 500)  # million cubic feet per day

                annual_oil_production = daily_oil_production * 365 / 1000  # million barrels per year
                annual_gas_production = daily_gas_production * 365 / 1000  # billion cubic feet per year

                current_oil_price = self.commodity_price_cycles['oil']['current_price']
                current_gas_price = self.commodity_price_cycles['natural_gas']['current_price'] * 1.037  # MMBtu to MCF

                revenue = (annual_oil_production * current_oil_price * 1000000) + \
                          (annual_gas_production * current_gas_price * 1000000)

            # Get EBITDA margin
            ebitda_margin = upstream_metrics.get('Latest', {}).get('EBITDA_Margin', 0.4)  # Default 40%

            # 3. Project production decline curve
            # Typical upstream companies face natural production decline without investment
            base_decline_rate = 0.10  # 10% annual production decline without investment
            reserve_replacement_ratio = reserves_data.get('Reserve_Replacement_Ratio', 1.0)

            # Adjust decline rate based on reserve replacement
            effective_decline_rate = base_decline_rate * (2 - reserve_replacement_ratio)
            effective_decline_rate = max(0, min(0.2, effective_decline_rate))  # Cap between 0-20%

            # 4. Project future cash flows
            forecasted_cash_flows = []
            forecasted_revenues = []
            forecasted_ebitda = []

            current_revenue = revenue
            current_ebitda_margin = ebitda_margin

            for year in range(forecast_years):
                # Project production change
                production_change = -effective_decline_rate

                # Adjust revenue for production change and oil price
                price_change = oil_price_forecast[year] / oil_price_forecast[max(0, year - 1)] if year > 0 else 1

                # Revenue changes with both production and price
                revenue_change = (1 + production_change) * price_change - 1
                current_revenue = current_revenue * (1 + revenue_change)
                forecasted_revenues.append(current_revenue)

                # EBITDA margin typically expands in high price environments and contracts in low price
                if price_change > 1.05:  # Price increasing
                    margin_change = 0.02  # Margin expansion
                elif price_change < 0.95:  # Price decreasing
                    margin_change = -0.03  # Margin contraction
                else:
                    margin_change = 0

                current_ebitda_margin = min(0.6, max(0.2, current_ebitda_margin + margin_change))
                current_ebitda = current_revenue * current_ebitda_margin
                forecasted_ebitda.append(current_ebitda)

                # Calculate free cash flow
                # For upstream companies, capital expenditure is critical to model
                # Higher capex in price upswings, lower in downswings
                if price_change > 1.05:
                    capex_ratio = 0.3  # 30% of revenue in growth periods
                elif price_change < 0.95:
                    capex_ratio = 0.15  # 15% of revenue in declining periods
                else:
                    capex_ratio = 0.22  # 22% in stable periods

                capex = current_revenue * capex_ratio

                # Cash taxes (simplified)
                tax_rate = 0.25  # 25% tax rate
                taxable_income = current_ebitda * 0.7  # Assuming 30% of EBITDA goes to D&A
                taxes = max(0, taxable_income * tax_rate)

                # Working capital changes (typically minimal for E&P)
                working_capital_change = current_revenue * 0.01 * revenue_change

                # Free cash flow
                fcf = current_ebitda - capex - taxes - working_capital_change
                forecasted_cash_flows.append(fcf)

            # 5. Calculate terminal value
            # For E&P companies, terminal value should consider reserve life
            remaining_reserve_life = max(0, reserve_life - forecast_years)

            if remaining_reserve_life > 20:
                # Standard perpetuity approach if plenty of reserves remain
                terminal_growth = params['terminal_growth_rate']
                terminal_fcf = forecasted_cash_flows[-1] * (1 + terminal_growth)
                discount_rate = self._calculate_energy_discount_rate(ticker, financial_data)
                terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            else:
                # Limited life terminal value for shorter reserve life
                terminal_fcf = forecasted_cash_flows[-1]
                discount_rate = self._calculate_energy_discount_rate(ticker, financial_data)

                # Sum of discounted cash flows for remaining life
                terminal_value = 0
                for i in range(remaining_reserve_life):
                    # Assume declining production and cash flow
                    year_fcf = terminal_fcf * (1 - effective_decline_rate) ** i
                    terminal_value += year_fcf / (1 + discount_rate) ** i

            # 6. Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # 7. Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # 8. Adjust for net debt and other items
            net_debt = self._calculate_net_debt(balance_sheet)

            # 9. Calculate equity value
            equity_value = enterprise_value - net_debt

            # 10. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'cyclical_upstream_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'discount_rate': discount_rate,
                'forecasted_revenues': forecasted_revenues,
                'forecasted_ebitda': forecasted_ebitda,
                'forecasted_cash_flows': forecasted_cash_flows,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'oil_price_forecast': oil_price_forecast[:forecast_years],
                'effective_decline_rate': effective_decline_rate,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in cyclical upstream DCF for {ticker}: {e}")
            return {
                'method': 'cyclical_upstream_dcf',
                'value_per_share': None,
                'error': str(e)
            }


    def _upstream_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                      upstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an upstream oil & gas company using sector-specific multiples
        like EV/Reserves, EV/Production, and EV/EBITDA
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get reserves data
            reserves_data = upstream_metrics.get('Reserves', {})

            # Get latest metrics
            latest_metrics = upstream_metrics.get('Latest', {})

            # Get latest financials
            latest_income = income_stmt.iloc[:, 0] if income_stmt is not None else None

            # 1. Collect key metrics for valuation

            # EBITDA
            ebitda = latest_metrics.get('EBITDA')
            if ebitda is None and latest_income is not None:
                if 'EBITDA' in latest_income.index:
                    ebitda = latest_income['EBITDA']
                elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                    ebitda = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']

            # Revenue
            revenue = latest_metrics.get('Revenue')
            if revenue is None and latest_income is not None and 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']

            # Get reserve and production figures in BOE (barrels of oil equivalent)
            oil_reserves = reserves_data.get('Oil_Reserves_MMBbl', 0)  # Million barrels
            gas_reserves = reserves_data.get('Gas_Reserves_BCF', 0)  # Billion cubic feet

            # Convert gas to BOE (roughly 6 MCF = 1 BOE)
            gas_reserves_boe = gas_reserves / 6
            total_reserves_mmboe = oil_reserves + gas_reserves_boe

            # Daily production in BOE
            oil_production = reserves_data.get('Oil_Production_MBblpd', 0) * 1000  # Barrels per day
            gas_production = reserves_data.get('Gas_Production_MMcfpd', 0)  # Thousand cubic feet per day

            # Convert gas to BOE
            gas_production_boe = gas_production / 6
            total_production_boepd = oil_production + gas_production_boe

            # 2. Define upstream-specific multiples
            # These would ideally come from industry databases of comparable companies
            base_multiples = {
                'EV_Reserves': 12.0,  # EV per BOE of proved reserves
                'EV_Production': 35000,  # EV per daily BOE produced
                'EV_EBITDA': 5.0,  # EV/EBITDA
                'EV_Revenue': 2.5  # EV/Revenue
            }

            # 3. Adjust multiples based on company-specific factors

            # Reserve life adjustment
            reserve_life = reserves_data.get('Reserve_Life_Index', 10)
            if reserve_life > 12:
                reserves_multiple_adj = 1.2  # Premium for long-lived reserves
            elif reserve_life < 8:
                reserves_multiple_adj = 0.8  # Discount for short-lived reserves
            else:
                reserves_multiple_adj = 1.0

            # Production growth or decline adjustment
            if 'Growth' in upstream_metrics and 'Revenue_Growth' in upstream_metrics['Growth']:
                growth_rate = upstream_metrics['Growth']['Revenue_Growth']
                if growth_rate > 0.1:
                    growth_multiple_adj = 1.2  # Premium for growing production
                elif growth_rate < -0.05:
                    growth_multiple_adj = 0.8  # Discount for declining production
                else:
                    growth_multiple_adj = 1.0
            else:
                growth_multiple_adj = 1.0

            # Margin quality adjustment
            ebitda_margin = latest_metrics.get('EBITDA_Margin')
            if ebitda_margin and ebitda_margin > 0.45:
                margin_multiple_adj = 1.2  # Premium for high margins
            elif ebitda_margin and ebitda_margin < 0.30:
                margin_multiple_adj = 0.8  # Discount for low margins
            else:
                margin_multiple_adj = 1.0

            # Apply adjustments
            adjusted_multiples = {
                'EV_Reserves': base_multiples['EV_Reserves'] * reserves_multiple_adj,
                'EV_Production': base_multiples['EV_Production'] * growth_multiple_adj,
                'EV_EBITDA': base_multiples['EV_EBITDA'] * margin_multiple_adj,
                'EV_Revenue': base_multiples['EV_Revenue'] * growth_multiple_adj
            }

            # 4. Calculate valuations using different multiples
            valuations = {}

            # EV/Reserves valuation
            if total_reserves_mmboe > 0:
                valuations['EV_Reserves'] = {
                    'multiple': adjusted_multiples['EV_Reserves'],
                    'metric_value': total_reserves_mmboe,
                    'enterprise_value': total_reserves_mmboe * adjusted_multiples['EV_Reserves'] * 1000000
                    # Convert to dollars
                }

            # EV/Production valuation
            if total_production_boepd > 0:
                valuations['EV_Production'] = {
                    'multiple': adjusted_multiples['EV_Production'],
                    'metric_value': total_production_boepd,
                    'enterprise_value': total_production_boepd * adjusted_multiples['EV_Production']
                }

            # EV/EBITDA valuation
            if ebitda and ebitda > 0:
                valuations['EV_EBITDA'] = {
                    'multiple': adjusted_multiples['EV_EBITDA'],
                    'metric_value': ebitda,
                    'enterprise_value': ebitda * adjusted_multiples['EV_EBITDA']
                }

            # EV/Revenue valuation
            if revenue and revenue > 0:
                valuations['EV_Revenue'] = {
                    'multiple': adjusted_multiples['EV_Revenue'],
                    'metric_value': revenue,
                    'enterprise_value': revenue * adjusted_multiples['EV_Revenue']
                }

            # 5. Calculate weighted average enterprise value
            # For upstream, reserves and production metrics are most important
            weights = {
                'EV_Reserves': 0.4,
                'EV_Production': 0.3,
                'EV_EBITDA': 0.2,
                'EV_Revenue': 0.1
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                return {
                    'method': 'upstream_multiples',
                    'value_per_share': None,
                    'error': 'Insufficient metrics for valuation'
                }

            # 6. Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # 7. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'upstream_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'valuations': valuations,
                'reserves_mmboe': total_reserves_mmboe,
                'production_boepd': total_production_boepd,
                'reserves_multiple_adj': reserves_multiple_adj,
                'growth_multiple_adj': growth_multiple_adj,
                'margin_multiple_adj': margin_multiple_adj,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in upstream multiples valuation for {ticker}: {e}")
            return {
                'method': 'upstream_multiples',
                'value_per_share': None,
                'error': str(e)
            }


    def _calculate_integrated_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics for integrated oil & gas companies"""
        try:
            # For integrated companies, we're interested in segment data
            # This would typically come from company filings with segment breakdowns
            # For this example, we'll estimate the breakdown

            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Estimate segment breakdown (would come from filings in practice)
            # Typical integrated company breakdown
            segment_breakdown = {
                'Upstream': {
                    'Revenue_Percent': 0.40,  # 40% of revenue
                    'EBITDA_Percent': 0.60,  # 60% of EBITDA (higher margin business)
                    'Assets_Percent': 0.55   # 55% of assets
                },
                'Downstream': {
                    'Revenue_Percent': 0.45,  # 45% of revenue
                    'EBITDA_Percent': 0.25,  # 25% of EBITDA (lower margin business)
                    'Assets_Percent': 0.30    # 30% of assets
                },
                'Midstream': {
                    'Revenue_Percent': 0.10,  # 10% of revenue
                    'EBITDA_Percent': 0.12,  # 12% of EBITDA
                    'Assets_Percent': 0.12    # 12% of assets
                },
                'Chemicals': {
                    'Revenue_Percent': 0.05,  # 5% of revenue
                    'EBITDA_Percent': 0.03,  # 3% of EBITDA
                    'Assets_Percent': 0.03    # 3% of assets
                }
            }

            # Calculate total revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
                metrics['Total_Revenue'] = revenue

                # Calculate segment revenues
                metrics['Segment_Revenue'] = {}
                for segment, data in segment_breakdown.items():
                    metrics['Segment_Revenue'][segment] = revenue * data['Revenue_Percent']

            # Calculate EBITDA
            if 'EBITDA' in latest_income.index:
                ebitda = latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                ebitda = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']
            else:
                ebitda = None

            if ebitda:
                metrics['EBITDA'] = ebitda

                # Calculate segment EBITDA
                metrics['Segment_EBITDA'] = {}
                for segment, data in segment_breakdown.items():
                    metrics['Segment_EBITDA'][segment] = ebitda * data['EBITDA_Percent']

            # Calculate total assets
            if 'Total Assets' in latest_balance.index:
                assets = latest_balance['Total Assets']
                metrics['Total_Assets'] = assets

                # Calculate segment assets
                metrics['Segment_Assets'] = {}
                for segment, data in segment_breakdown.items():
                    metrics['Segment_Assets'][segment] = assets * data['Assets_Percent']

            # Calculate margin metrics
            if 'Total_Revenue' in metrics and 'EBITDA' in metrics and metrics['Total_Revenue'] > 0:
                metrics['EBITDA_Margin'] = metrics['EBITDA'] / metrics['Total_Revenue']

                # Calculate segment margins
                metrics['Segment_Margins'] = {}
                for segment in segment_breakdown.keys():
                    if segment in metrics['Segment_Revenue'] and segment in metrics['Segment_EBITDA'] and metrics['Segment_Revenue'][segment] > 0:
                        metrics['Segment_Margins'][segment] = metrics['Segment_EBITDA'][segment] / metrics['Segment_Revenue'][segment]

            # Calculate ROA metrics
            if 'Total_Assets' in metrics and 'EBITDA' in metrics and metrics['Total_Assets'] > 0:
                metrics['EBITDA_ROA'] = metrics['EBITDA'] / metrics['Total_Assets']

                # Calculate segment ROA
                metrics['Segment_ROA'] = {}
                for segment in segment_breakdown.keys():
                    if segment in metrics['Segment_Assets'] and segment in metrics['Segment_EBITDA'] and metrics['Segment_Assets'][segment] > 0:
                        metrics['Segment_ROA'][segment] = metrics['Segment_EBITDA'][segment] / metrics['Segment_Assets'][segment]

            # Calculate debt metrics
            if 'Total Debt' in latest_balance.index:
                debt = latest_balance['Total Debt']
                metrics['Total_Debt'] = debt

                if 'EBITDA' in metrics and metrics['EBITDA'] > 0:
                    metrics['Debt_to_EBITDA'] = debt / metrics['EBITDA']

            # Get refining metrics (would typically come from company filings)
            if 'Downstream' in segment_breakdown:
                refining_metrics = {
                    'Refining_Capacity': company_info.get('refining_capacity', 1000),  # Thousand barrels per day
                    'Refining_Utilization': company_info.get('refining_utilization', 0.85),  # 85% utilization
                    'Refining_Margin': company_info.get('refining_margin', 10)  # USD per barrel
                }
                metrics['Refining'] = refining_metrics

            # Get production metrics (upstream segment)
            if 'Upstream' in segment_breakdown:
                production_metrics = {
                    'Oil_Production': company_info.get('oil_production', 500),  # Thousand barrels per day
                    'Gas_Production': company_info.get('gas_production', 2000),  # Million cubic feet per day
                    'Reserves_Life': company_info.get('reserves_life', 12)  # Years
                }
                metrics['Production'] = production_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating integrated metrics: {e}")
            return {}

    def _sum_of_the_parts_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                   integrated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a Sum-of-the-Parts (SOTP) valuation for an integrated oil & gas company
        by valuing each business segment separately
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get segment data
            segment_ebitda = integrated_metrics.get('Segment_EBITDA', {})
            segment_revenue = integrated_metrics.get('Segment_Revenue', {})
            segment_assets = integrated_metrics.get('Segment_Assets', {})

            # Define segment-specific EV/EBITDA multiples
            # These would ideally come from comparable companies in each segment
            segment_multiples = {
                'Upstream': {
                    'EV_EBITDA': 5.0,  # Upstream tends to have lower multiples due to cyclicality
                    'EV_Revenue': 1.5
                },
                'Downstream': {
                    'EV_EBITDA': 6.0,  # Downstream can be more stable
                    'EV_Revenue': 0.5
                },
                'Midstream': {
                    'EV_EBITDA': 9.0,  # Midstream has stable cash flows, higher multiples
                    'EV_Revenue': 3.0
                },
                'Chemicals': {
                    'EV_EBITDA': 7.0,
                    'EV_Revenue': 1.0
                }
            }

            # Calculate segment values
            segment_values = {}
            total_enterprise_value = 0

            for segment, ebitda in segment_ebitda.items():
                if segment in segment_multiples and ebitda > 0:
                    # Value based on EBITDA
                    ev_ebitda = ebitda * segment_multiples[segment]['EV_EBITDA']

                    # Add revenue-based valuation as a check
                    ev_revenue = segment_revenue.get(segment, 0) * segment_multiples[segment]['EV_Revenue']

                    # Use EBITDA-based valuation as primary
                    segment_ev = ev_ebitda

                    segment_values[segment] = {
                        'EBITDA': ebitda,
                        'EV_EBITDA_Multiple': segment_multiples[segment]['EV_EBITDA'],
                        'EV_EBITDA_Value': ev_ebitda,
                        'Revenue': segment_revenue.get(segment, 0),
                        'EV_Revenue_Multiple': segment_multiples[segment]['EV_Revenue'],
                        'EV_Revenue_Value': ev_revenue,
                        'Enterprise_Value': segment_ev
                    }

                    total_enterprise_value += segment_ev

            # Add any additional value not captured in segment analysis
            # For example, exploration assets, corporate value, etc.
            corporate_overhead = -0.05 * total_enterprise_value  # Corporate overhead typically reduces value
            total_enterprise_value += corporate_overhead

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = total_enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'sum_of_the_parts',
                'value_per_share': value_per_share,
                'enterprise_value': total_enterprise_value,
                'equity_value': equity_value,
                'segment_values': segment_values,
                'corporate_overhead': corporate_overhead,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in sum-of-the-parts valuation for {ticker}: {e}")
            return {
                'method': 'sum_of_the_parts',
                'value_per_share': None,
                'error': str(e)
            }

    def _cyclical_integrated_dcf(self, ticker: str, financial_data: Dict[str, Any],
                                integrated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a DCF valuation for an integrated oil & gas company
        with cyclical adjustments for each business segment
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get parameters specific to energy sector
            params = self._get_dcf_parameters("Energy")

            # Get segment data
            total_revenue = integrated_metrics.get('Total_Revenue', 0)
            total_ebitda = integrated_metrics.get('EBITDA', 0)
            segment_ebitda = integrated_metrics.get('Segment_EBITDA', {})
            segment_revenue = integrated_metrics.get('Segment_Revenue', {})
            segment_margins = integrated_metrics.get('Segment_Margins', {})

            # Define segment-specific growth assumptions
            segment_growth = {
                'Upstream': {
                    'initial_growth': 0.03,  # 3% initial growth
                    'long_term_growth': 0.01,  # 1% long-term growth
                    'volatility': 0.15  # 15% annual volatility
                },
                'Downstream': {
                    'initial_growth': 0.02,  # 2% initial growth
                    'long_term_growth': 0.01,  # 1% long-term growth
                    'volatility': 0.10  # 10% annual volatility
                },
                'Midstream': {
                    'initial_growth': 0.04,  # 4% initial growth
                    'long_term_growth': 0.02,  # 2% long-term growth
                    'volatility': 0.05  # 5% annual volatility (more stable)
                },
                'Chemicals': {
                    'initial_growth': 0.03,  # 3% initial growth
                    'long_term_growth': 0.02,  # 2% long-term growth
                    'volatility': 0.08  # 8% annual volatility
                }
            }

            # Commodity price cycles (simplified)
            oil_price_cycle = [1.05, 1.10, 0.95, 0.85, 0.90, 1.05, 1.10, 0.95]
            refining_margin_cycle = [1.10, 1.05, 0.90, 0.85, 0.95, 1.15, 1.05, 0.90]

            # Calculate discount rate
            discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})

            # Forecast period
            forecast_years = params['forecast_years']

            # Create segment-specific forecasts
            segment_forecasts = {}
            total_forecasted_cash_flows = [0] * forecast_years

            for segment, segment_data in segment_growth.items():
                # Get segment starting values
                segment_starting_revenue = segment_revenue.get(segment, 0)
                segment_starting_ebitda = segment_ebitda.get(segment, 0)
                segment_starting_margin = segment_margins.get(segment, 0.2)  # Default to 20% margin

                # Skip if no meaningful segment data
                if segment_starting_revenue <= 0 or segment_starting_ebitda <= 0:
                    continue

                # Create forecast arrays
                forecasted_revenue = []
                forecasted_ebitda = []
                forecasted_cash_flows = []

                current_revenue = segment_starting_revenue
                current_margin = segment_starting_margin

                for year in range(forecast_years):
                    # Apply appropriate price cycle for each segment
                    cycle_index = (year) % len(oil_price_cycle)

                    if segment == 'Upstream':
                        price_factor = oil_price_cycle[cycle_index]
                    elif segment == 'Downstream':
                        price_factor = refining_margin_cycle[cycle_index]
                    else:
                        # Less cyclical segments
                        price_factor = 1.0

                    # Calculate growth adjusted for price cycle
                    base_growth = segment_data['initial_growth'] * (
                        1 - year / (forecast_years * 2)
                    )  # Linear decline in growth rate

                    adjusted_growth = base_growth * price_factor

                    # Add some random volatility
                    volatility_adjustment = np.random.normal(0, segment_data['volatility'] / 3)
                    adjusted_growth += volatility_adjustment

                    # Update revenue
                    current_revenue *= (1 + adjusted_growth)
                    forecasted_revenue.append(current_revenue)

                    # Update margin - margins typically expand in high price environments
                    if price_factor > 1.05:
                        margin_change = 0.01  # Margin improvement in good times
                    elif price_factor < 0.95:
                        margin_change = -0.015  # Margin compression in bad times
                    else:
                        margin_change = 0

                    current_margin = max(0.05, min(0.4, current_margin + margin_change))  # Keep margins reasonable

                    # Calculate EBITDA
                    current_ebitda = current_revenue * current_margin
                    forecasted_ebitda.append(current_ebitda)

                    # Estimate capex (higher in good times, lower in bad)
                    if segment == 'Upstream':
                        # Upstream requires higher capex
                        capex_ratio = 0.25 if price_factor > 1 else 0.15
                    elif segment == 'Midstream':
                        # Midstream has steady capex
                        capex_ratio = 0.18
                    else:
                        # Other segments more moderate
                        capex_ratio = 0.12 if price_factor > 1 else 0.08

                    capex = current_revenue * capex_ratio

                    # Estimate taxes (simplified)
                    tax_rate = 0.25  # 25% tax rate
                    taxable_income = current_ebitda * 0.7  # Assuming 30% of EBITDA goes to D&A
                    taxes = max(0, taxable_income * tax_rate)

                    # Working capital changes
                    wc_change = current_revenue * 0.01 * adjusted_growth

                    # Calculate free cash flow
                    fcf = current_ebitda - capex - taxes - wc_change
                    forecasted_cash_flows.append(fcf)

                    # Add to total cash flows
                    total_forecasted_cash_flows[year] += fcf

                # Store segment forecast
                segment_forecasts[segment] = {
                    'revenue': forecasted_revenue,
                    'ebitda': forecasted_ebitda,
                    'cash_flows': forecasted_cash_flows,
                    'margin': current_margin
                }

            # Calculate terminal value
            terminal_growth = params['terminal_growth_rate']
            terminal_fcf = total_forecasted_cash_flows[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(total_forecasted_cash_flows))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'cyclical_integrated_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'segment_forecasts': segment_forecasts,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'total_forecasted_cash_flows': total_forecasted_cash_flows,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in cyclical integrated DCF for {ticker}: {e}")
            return {
                'method': 'cyclical_integrated_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _integrated_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                      integrated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an integrated oil & gas company using sector-specific multiples
        with segment-based adjustments
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get segment data
            total_revenue = integrated_metrics.get('Total_Revenue', 0)
            total_ebitda = integrated_metrics.get('EBITDA', 0)
            total_assets = integrated_metrics.get('Total_Assets', 0)
            segment_ebitda = integrated_metrics.get('Segment_EBITDA', {})

            # Define multiples (these would typically come from comparable companies)
            base_multiples = {
                'EV_EBITDA': 6.0,  # Base EV/EBITDA for integrated companies
                'EV_Revenue': 1.0,  # Base EV/Revenue
                'EV_Assets': 0.8   # Base EV/Total Assets
            }

            # Calculate base valuations
            valuations = {}

            # EV/EBITDA valuation
            if total_ebitda > 0:
                valuations['EV_EBITDA'] = {
                    'multiple': base_multiples['EV_EBITDA'],
                    'metric_value': total_ebitda,
                    'enterprise_value': total_ebitda * base_multiples['EV_EBITDA']
                }

            # EV/Revenue valuation
            if total_revenue > 0:
                valuations['EV_Revenue'] = {
                    'multiple': base_multiples['EV_Revenue'],
                    'metric_value': total_revenue,
                    'enterprise_value': total_revenue * base_multiples['EV_Revenue']
                }

            # EV/Assets valuation
            if total_assets > 0:
                valuations['EV_Assets'] = {
                    'multiple': base_multiples['EV_Assets'],
                    'metric_value': total_assets,
                    'enterprise_value': total_assets * base_multiples['EV_Assets']
                }

            # Segment-based adjustments
            # EV/EBITDA can be adjusted based on segment mix
            if 'EV_EBITDA' in valuations and segment_ebitda:
                # If company has more upstream exposure, adjust multiple down (more volatile)
                # If more midstream/downstream, adjust multiple up (more stable)
                upstream_pct = segment_ebitda.get('Upstream', 0) / total_ebitda if total_ebitda > 0 else 0
                midstream_pct = segment_ebitda.get('Midstream', 0) / total_ebitda if total_ebitda > 0 else 0

                # Adjust multiple: upstream exposure decreases multiple, midstream increases it
                multiple_adjustment = -0.5 * upstream_pct + 1.0 * midstream_pct

                adjusted_multiple = base_multiples['EV_EBITDA'] * (1 + multiple_adjustment)
                adjusted_ev = total_ebitda * adjusted_multiple

                valuations['EV_EBITDA_Adjusted'] = {
                    'multiple': adjusted_multiple,
                    'metric_value': total_ebitda,
                    'enterprise_value': adjusted_ev,
                    'upstream_pct': upstream_pct,
                    'midstream_pct': midstream_pct,
                    'multiple_adjustment': multiple_adjustment
                }

            # Calculate weighted average enterprise value
            weights = {
                'EV_EBITDA': 0.4,
                'EV_EBITDA_Adjusted': 0.4,
                'EV_Revenue': 0.1,
                'EV_Assets': 0.1
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                return {
                    'method': 'integrated_multiples',
                    'value_per_share': None,
                    'error': 'Insufficient metrics for valuation'
                }

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'integrated_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'base_multiples': base_multiples,
                'valuations': valuations,
                'weights': weights,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in integrated multiples valuation for {ticker}: {e}")
            return {
                'method': 'integrated_multiples',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_midstream_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics for midstream energy companies (pipelines, storage, etc.)"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get most recent financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Revenue and EBITDA
            if 'Total Revenue' in latest_income.index:
                metrics['Revenue'] = latest_income['Total Revenue']

            if 'EBITDA' in latest_income.index:
                metrics['EBITDA'] = latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                metrics['EBITDA'] = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']

            # Calculate EBITDA margin
            if 'Revenue' in metrics and 'EBITDA' in metrics and metrics['Revenue'] > 0:
                metrics['EBITDA_Margin'] = metrics['EBITDA'] / metrics['Revenue']

            # Debt metrics
            if 'Total Debt' in latest_balance.index:
                metrics['Debt'] = latest_balance['Total Debt']

                if 'EBITDA' in metrics and metrics['EBITDA'] > 0:
                    metrics['Debt_to_EBITDA'] = metrics['Debt'] / metrics['EBITDA']

            # Distribution metrics (for MLPs and similar structures)
            if latest_cash_flow is not None and 'Dividends Paid' in latest_cash_flow.index:
                metrics['Distributions'] = abs(latest_cash_flow['Dividends Paid'])

                if 'EBITDA' in metrics and metrics['EBITDA'] > 0:
                    metrics['Distribution_Coverage_Ratio'] = metrics['EBITDA'] / metrics['Distributions']

            # Asset metrics
            if 'Property Plant and Equipment' in latest_balance.index:
                metrics['PPE'] = latest_balance['Property Plant and Equipment']

                if 'Revenue' in metrics and metrics['Revenue'] > 0:
                    metrics['Asset_Intensity'] = metrics['PPE'] / metrics['Revenue']

            # Contract metrics (typical for midstream)
            contract_metrics = {
                'Contracted_Capacity_Pct': company_info.get('contracted_capacity_pct', 0.85),  # % capacity contracted
                'Average_Contract_Duration': company_info.get('avg_contract_duration', 8),  # years
                'Take_or_Pay_Pct': company_info.get('take_or_pay_pct', 0.70)  # % of contracts that are take-or-pay
            }
            metrics['Contracts'] = contract_metrics

            # Operational metrics
            operational_metrics = {
                'Pipeline_Miles': company_info.get('pipeline_miles', 5000),
                'Storage_Capacity': company_info.get('storage_capacity', 50),  # million barrels
                'Throughput_Volume': company_info.get('throughput_volume', 1.5)  # million barrels per day
            }
            metrics['Operations'] = operational_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating midstream metrics: {e}")
            return {}

    def _midstream_ddm(self, ticker: str, financial_data: Dict[str, Any],
                      midstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dividend Discount Model specifically adapted for midstream companies,
        which typically have high dividend/distribution yields
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            share_price = market_data.get('share_price')
            shares_outstanding = market_data.get('shares_outstanding')
            dividend_yield = market_data.get('dividend_yield')

            # Check if company pays meaningful dividends
            if not dividend_yield or dividend_yield < 0.01:  # Less than 1%
                return {
                    'method': 'midstream_ddm',
                    'value_per_share': None,
                    'error': 'Insufficient dividend yield for DDM analysis'
                }

            # Get current dividend per share
            current_dividend = None

            # Try to get from cash flow
            if cash_flow is not None and 'Dividends Paid' in cash_flow.iloc[:, 0].index:
                total_dividends = abs(cash_flow.iloc[:, 0]['Dividends Paid'])
                if shares_outstanding and shares_outstanding > 0:
                    current_dividend = total_dividends / shares_outstanding

            # If not available, calculate from yield and share price
            if current_dividend is None and dividend_yield and share_price:
                current_dividend = share_price * dividend_yield

            if current_dividend is None or current_dividend <= 0:
                return {
                    'method': 'midstream_ddm',
                    'value_per_share': None,
                    'error': 'Unable to determine current dividend'
                }

            # Contract metrics affect distribution growth stability
            contract_metrics = midstream_metrics.get('Contracts', {})
            contracted_capacity_pct = contract_metrics.get('Contracted_Capacity_Pct', 0.85)
            avg_contract_duration = contract_metrics.get('Average_Contract_Duration', 8)
            take_or_pay_pct = contract_metrics.get('Take_or_Pay_Pct', 0.70)

            # Calculate contract quality score (0-1)
            contract_quality = (contracted_capacity_pct * 0.4) + (min(1, avg_contract_duration / 10) * 0.3) + (
                        take_or_pay_pct * 0.3)

            # Define growth phases for distributions
            # Midstream companies typically have a period of higher growth followed by stable growth
            high_growth_years = 5
            transition_years = 5

            # Determine initial growth rate
            # For MLPs/midstream, distribution growth is tied to:
            # 1. Existing asset cash flow growth
            # 2. New projects coming online
            # 3. Acquisitions

            # Base initial growth rate
            initial_growth_rate = 0.05  # 5% base case

            # Adjust for contract quality
            initial_growth_rate *= (0.8 + contract_quality * 0.4)  # Range: 4-6% based on contract quality

            # Check if distribution coverage ratio is available
            # Higher coverage enables faster growth
            if 'Distribution_Coverage_Ratio' in midstream_metrics:
                coverage_ratio = midstream_metrics['Distribution_Coverage_Ratio']
                if coverage_ratio > 1.2:
                    initial_growth_rate *= min(coverage_ratio / 1.2, 1.5)  # Bonus for high coverage
                elif coverage_ratio < 1.0:
                    initial_growth_rate *= max(coverage_ratio, 0.5)  # Penalty for low coverage

            # Terminal growth rate (long-term)
            terminal_growth = 0.02  # 2% long-term growth (inflation-like)

            # Calculate discount rate
            # Midstream discount rates are typically lower than E&P due to more stable cash flows
            base_discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})
            # Adjust discount rate based on contract quality and leverage
            discount_rate = base_discount_rate * (1.1 - contract_quality * 0.2)  # Better contracts = lower risk

            # Check debt levels
            if 'Debt_to_EBITDA' in midstream_metrics:
                debt_to_ebitda = midstream_metrics['Debt_to_EBITDA']
                if debt_to_ebitda > 4.5:
                    discount_rate *= (1 + (debt_to_ebitda - 4.5) * 0.05)  # Higher leverage = higher risk
                elif debt_to_ebitda < 3.5:
                    discount_rate *= (1 - (3.5 - debt_to_ebitda) * 0.02)  # Lower leverage = lower risk

            # Calculate future distributions
            future_distributions = []

            # High growth phase
            for year in range(1, high_growth_years + 1):
                future_distributions.append(current_dividend * (1 + initial_growth_rate) ** year)

            # Transition phase (linear decline in growth to terminal rate)
            last_distribution = future_distributions[-1]
            for year in range(1, transition_years + 1):
                transition_growth = initial_growth_rate - (
                            (initial_growth_rate - terminal_growth) * year / transition_years)
                next_distribution = last_distribution * (1 + transition_growth)
                future_distributions.append(next_distribution)
                last_distribution = next_distribution

            # Calculate present value of explicit forecast
            present_value = sum(dividend / (1 + discount_rate) ** year
                                for year, dividend in enumerate(future_distributions, 1))

            # Calculate terminal value
            terminal_distribution = future_distributions[-1] * (1 + terminal_growth)
            terminal_value = terminal_distribution / (discount_rate - terminal_growth)
            present_value_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

            # Total value per share
            value_per_share = present_value + present_value_terminal

            return {
                'method': 'midstream_ddm',
                'value_per_share': value_per_share,
                'current_dividend': current_dividend,
                'initial_growth_rate': initial_growth_rate,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
                'contract_quality': contract_quality,
                'future_distributions': future_distributions,
                'present_value': present_value,
                'present_value_terminal': present_value_terminal
            }

        except Exception as e:
            logger.error(f"Error in midstream DDM valuation for {ticker}: {e}")
            return {
                'method': 'midstream_ddm',
                'value_per_share': None,
                'error': str(e)
            }

    def _midstream_dcf(self, ticker: str, financial_data: Dict[str, Any],
                       midstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        DCF valuation for midstream companies with adjustment for long-term contracts
        and infrastructure assets
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get most recent financial data
            latest_income = income_stmt.iloc[:, 0] if income_stmt is not None else None
            latest_balance = balance_sheet.iloc[:, 0] if balance_sheet is not None else None
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Get key metrics
            revenue = midstream_metrics.get('Revenue')
            ebitda = midstream_metrics.get('EBITDA')
            ebitda_margin = midstream_metrics.get('EBITDA_Margin', 0.6)  # Default 60% if not available

            # Get contract data
            contract_metrics = midstream_metrics.get('Contracts', {})
            contracted_capacity_pct = contract_metrics.get('Contracted_Capacity_Pct', 0.85)
            avg_contract_duration = contract_metrics.get('Average_Contract_Duration', 8)

            # Get operational data
            operational_metrics = midstream_metrics.get('Operations', {})

            # Define DCF parameters
            forecast_years = 10  # Midstream typically uses longer forecast period

            # Initial growth rate based on contract profile
            base_growth_rate = 0.04  # 4% base case

            # Adjust growth rate based on contract profile
            contract_adjusted_growth = base_growth_rate * (0.8 + contracted_capacity_pct * 0.4)

            # Consider average contract duration for growth stability
            contract_duration_factor = min(1.0, avg_contract_duration / 10)  # Scale to max of 1.0
            growth_stability = contract_duration_factor

            # Calculate growth by year
            growth_rates = []
            for year in range(forecast_years):
                if year < avg_contract_duration:
                    # During contract period, growth is more stable
                    year_growth = contract_adjusted_growth * (1 - year / (forecast_years * 1.5))
                else:
                    # After contract period, more uncertainty
                    year_growth = contract_adjusted_growth * (1 - year / forecast_years) * 0.7
                growth_rates.append(max(0.01, year_growth))  # Minimum growth of 1%

            # Terminal growth rate
            terminal_growth = 0.02  # Infrastructure assets typically grow with inflation

            # Calculate discount rate with midstream adjustment
            base_discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})
            discount_rate = base_discount_rate * (1 - growth_stability * 0.1)  # More stable contracts = lower discount

            # Ensure terminal growth < discount rate
            if terminal_growth >= discount_rate - 0.03:
                terminal_growth = discount_rate - 0.03

            # Project EBITDA
            forecasted_ebitda = []
            current_ebitda = ebitda

            for i, growth_rate in enumerate(growth_rates):
                if i == 0:
                    forecasted_ebitda.append(current_ebitda * (1 + growth_rate))
                else:
                    forecasted_ebitda.append(forecasted_ebitda[-1] * (1 + growth_rate))

            # Project capex
            forecasted_capex = []

            # Midstream typically has higher maintenance capex (% of revenue)
            maintenance_capex_pct = 0.08  # 8% of revenue

            # Project capex as % of revenue
            for i, ebitda_value in enumerate(forecasted_ebitda):
                # Estimate revenue from EBITDA using margin
                estimated_revenue = ebitda_value / ebitda_margin

                # Calculate capex (maintenance and growth)
                if i < 3:
                    # Higher growth capex in early years
                    growth_capex_pct = 0.12  # 12% of revenue
                elif i < 6:
                    # Moderate growth capex in middle years
                    growth_capex_pct = 0.08  # 8% of revenue
                else:
                    # Lower growth capex in later years
                    growth_capex_pct = 0.04  # 4% of revenue

                total_capex_pct = maintenance_capex_pct + growth_capex_pct
                total_capex = estimated_revenue * total_capex_pct

                forecasted_capex.append(total_capex)

            # Calculate free cash flow
            forecasted_fcf = []

            for i, ebitda_value in enumerate(forecasted_ebitda):
                # Estimate D&A (typically 3-4% of revenue for midstream)
                estimated_revenue = ebitda_value / ebitda_margin
                depreciation = estimated_revenue * 0.035  # 3.5% of revenue

                # Calculate income taxes
                taxable_income = ebitda_value - depreciation

                # MLPs may have different tax structures
                is_mlp = True  # Simplified assumption - would come from company data
                if is_mlp:
                    effective_tax_rate = 0.10  # Lower for MLPs
                else:
                    effective_tax_rate = 0.25  # Standard corporate rate

                income_taxes = taxable_income * effective_tax_rate

                # Calculate FCF
                fcf = ebitda_value - forecasted_capex[i] - income_taxes
                forecasted_fcf.append(fcf)

            # Calculate terminal value
            terminal_fcf = forecasted_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecasted_fcf))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'midstream_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'forecasted_ebitda': forecasted_ebitda,
                'forecasted_capex': forecasted_capex,
                'forecasted_fcf': forecasted_fcf,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'growth_rates': growth_rates,
                'contract_profile': {
                    'contracted_capacity_pct': contracted_capacity_pct,
                    'avg_contract_duration': avg_contract_duration,
                    'growth_stability': growth_stability
                },
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in midstream DCF valuation for {ticker}: {e}")
            return {
                'method': 'midstream_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _midstream_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                       midstream_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a midstream energy company using sector-specific multiples
        including EV/EBITDA and rate-based metrics
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get key metrics
            ebitda = midstream_metrics.get('EBITDA')
            revenue = midstream_metrics.get('Revenue')
            distribution = midstream_metrics.get('Distributions')
            ppe = midstream_metrics.get('PPE')  # Property, Plant, Equipment

            # Get contract data
            contract_metrics = midstream_metrics.get('Contracts', {})
            contracted_capacity_pct = contract_metrics.get('Contracted_Capacity_Pct', 0.85)
            avg_contract_duration = contract_metrics.get('Average_Contract_Duration', 8)

            # Get operational data
            operational_metrics = midstream_metrics.get('Operations', {})
            pipeline_miles = operational_metrics.get('Pipeline_Miles')
            storage_capacity = operational_metrics.get('Storage_Capacity')
            throughput_volume = operational_metrics.get('Throughput_Volume')

            # Define base multiples for midstream
            base_multiples = {
                'EV_EBITDA': 10.0,  # Midstream typically trades at higher multiples due to stable cash flows
                'EV_Revenue': 3.5,
                'Distribution_Yield': 0.07,  # 7% typical yield
                'EV_PPE': 1.2,  # Enterprise value to Property, Plant & Equipment
                'Value_per_Mile': 1.5  # $ million per mile of pipeline
            }

            # Adjust multiples based on contract quality
            contract_quality = (contracted_capacity_pct * 0.4) + (min(1, avg_contract_duration / 10) * 0.6)
            multiple_adjustment = (contract_quality - 0.8) * 2  # Normalize around 0

            adjusted_multiples = {
                'EV_EBITDA': base_multiples['EV_EBITDA'] * (1 + multiple_adjustment * 0.15),
                'EV_Revenue': base_multiples['EV_Revenue'] * (1 + multiple_adjustment * 0.1),
                'Distribution_Yield': base_multiples['Distribution_Yield'] * (1 - multiple_adjustment * 0.1),
                'EV_PPE': base_multiples['EV_PPE'] * (1 + multiple_adjustment * 0.1),
                'Value_per_Mile': base_multiples['Value_per_Mile'] * (1 + multiple_adjustment * 0.1)
            }

            # Calculate valuations using different metrics
            valuations = {}

            # EV/EBITDA valuation
            if ebitda and ebitda > 0:
                valuations['EV_EBITDA'] = {
                    'multiple': adjusted_multiples['EV_EBITDA'],
                    'metric_value': ebitda,
                    'enterprise_value': ebitda * adjusted_multiples['EV_EBITDA']
                }

            # EV/Revenue valuation
            if revenue and revenue > 0:
                valuations['EV_Revenue'] = {
                    'multiple': adjusted_multiples['EV_Revenue'],
                    'metric_value': revenue,
                    'enterprise_value': revenue * adjusted_multiples['EV_Revenue']
                }

            # Distribution yield valuation (reverse - higher yield means lower price)
            if distribution and distribution > 0 and shares_outstanding and shares_outstanding > 0:
                distribution_per_share = distribution / shares_outstanding
                price_based_on_yield = distribution_per_share / adjusted_multiples['Distribution_Yield']
                valuations['Distribution_Yield'] = {
                    'multiple': adjusted_multiples['Distribution_Yield'],
                    'metric_value': distribution_per_share,
                    'value_per_share': price_based_on_yield,
                    'enterprise_value': price_based_on_yield * shares_outstanding
                }

            # EV/PPE valuation
            if ppe and ppe > 0:
                valuations['EV_PPE'] = {
                    'multiple': adjusted_multiples['EV_PPE'],
                    'metric_value': ppe,
                    'enterprise_value': ppe * adjusted_multiples['EV_PPE']
                }

            # Value per mile (asset-based valuation)
            if pipeline_miles and pipeline_miles > 0:
                pipeline_value = pipeline_miles * adjusted_multiples['Value_per_Mile'] * 1000000  # convert to dollars
                valuations['Value_per_Mile'] = {
                    'multiple': adjusted_multiples['Value_per_Mile'],
                    'metric_value': pipeline_miles,
                    'enterprise_value': pipeline_value
                }

            # Calculate weighted average enterprise value
            weights = {
                'EV_EBITDA': 0.45,
                'EV_Revenue': 0.15,
                'Distribution_Yield': 0.15,
                'EV_PPE': 0.15,
                'Value_per_Mile': 0.10
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                return {
                    'method': 'midstream_multiples',
                    'value_per_share': None,
                    'error': 'Insufficient metrics for valuation'
                }

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'midstream_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'contract_quality': contract_quality,
                'multiple_adjustment': multiple_adjustment,
                'valuations': valuations,
                'weights': weights,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in midstream multiples valuation for {ticker}: {e}")
            return {
                'method': 'midstream_multiples',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_utility_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics specific to utility companies"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get most recent financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Basic financial metrics
            if 'Total Revenue' in latest_income.index:
                metrics['Revenue'] = latest_income['Total Revenue']

            if 'EBITDA' in latest_income.index:
                metrics['EBITDA'] = latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                metrics['EBITDA'] = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']

            # Calculate EBITDA margin
            if 'Revenue' in metrics and 'EBITDA' in metrics and metrics['Revenue'] > 0:
                metrics['EBITDA_Margin'] = metrics['EBITDA'] / metrics['Revenue']

            # Debt metrics
            if 'Total Debt' in latest_balance.index:
                metrics['Debt'] = latest_balance['Total Debt']

                if 'EBITDA' in metrics and metrics['EBITDA'] > 0:
                    metrics['Debt_to_EBITDA'] = metrics['Debt'] / metrics['EBITDA']

            # Dividend metrics
            if latest_cash_flow is not None and 'Dividends Paid' in latest_cash_flow.index:
                metrics['Dividends'] = abs(latest_cash_flow['Dividends Paid'])

                if 'Net Income' in latest_income.index and latest_income['Net Income'] > 0:
                    metrics['Payout_Ratio'] = metrics['Dividends'] / latest_income['Net Income']

            # Regulated asset metrics
            if 'Property Plant and Equipment' in latest_balance.index:
                metrics['PPE'] = latest_balance['Property Plant and Equipment']

                # Estimate Regulated Asset Base (RAB)
                # In a real implementation, this would come from company filings
                metrics['Regulated_Asset_Base'] = metrics['PPE'] * 0.85  # Assuming 85% of PPE is regulated

            # Utility-specific operational metrics
            utility_metrics = {
                'Allowed_ROE': company_info.get('allowed_roe', 0.095),  # Allowed return on equity
                'Regulatory_Lag': company_info.get('regulatory_lag', 0.5),  # Years of lag in rate recovery
                'Rate_Base_Growth': company_info.get('rate_base_growth', 0.04),  # Annual growth in rate base
                'Customer_Growth': company_info.get('customer_growth', 0.01),  # Annual customer growth
                'Energy_Sales_Growth': company_info.get('energy_sales_growth', 0.005)  # Annual energy sales growth
            }
            metrics['Utility_Metrics'] = utility_metrics

            # Generation mix (for electric utilities)
            generation_mix = {
                'Natural_Gas': company_info.get('natural_gas_pct', 0.30),
                'Coal': company_info.get('coal_pct', 0.20),
                'Nuclear': company_info.get('nuclear_pct', 0.20),
                'Renewables': company_info.get('renewables_pct', 0.25),
                'Other': company_info.get('other_gen_pct', 0.05)
            }
            metrics['Generation_Mix'] = generation_mix

            return metrics

        except Exception as e:
            logger.error(f"Error calculating utility metrics: {e}")
            return {}

    def _utility_ddm(self, ticker: str, financial_data: Dict[str, Any],
                     utility_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dividend Discount Model adapted for utility companies, which typically
        pay stable dividends with steady growth
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            share_price = market_data.get('share_price')
            shares_outstanding = market_data.get('shares_outstanding')
            dividend_yield = market_data.get('dividend_yield')

            # Check if company pays meaningful dividends
            if not dividend_yield or dividend_yield < 0.01:  # Less than 1%
                return {
                    'method': 'utility_ddm',
                    'value_per_share': None,
                    'error': 'Insufficient dividend yield for DDM analysis'
                }

            # Get current dividend per share
            current_dividend = None

            # Try to get from cash flow
            if cash_flow is not None and 'Dividends Paid' in cash_flow.iloc[:, 0].index:
                total_dividends = abs(cash_flow.iloc[:, 0]['Dividends Paid'])
                if shares_outstanding and shares_outstanding > 0:
                    current_dividend = total_dividends / shares_outstanding

            # If not available, calculate from yield and share price
            if current_dividend is None and dividend_yield and share_price:
                current_dividend = share_price * dividend_yield

            if current_dividend is None or current_dividend <= 0:
                return {
                    'method': 'utility_ddm',
                    'value_per_share': None,
                    'error': 'Unable to determine current dividend'
                }

            # Get utility-specific metrics
            utility_specific = utility_metrics.get('Utility_Metrics', {})
            allowed_roe = utility_specific.get('Allowed_ROE', 0.095)
            rate_base_growth = utility_specific.get('Rate_Base_Growth', 0.04)

            # Get payout ratio
            payout_ratio = utility_metrics.get('Payout_Ratio', 0.65)  # Default 65% if not available

            # Determine dividend growth rate based on:
            # 1. Allowed ROE
            # 2. Rate base growth
            # 3. Payout ratio

            # Basic growth formula for utilities: g = ROE * (1 - Payout Ratio)
            retention_rate = 1 - payout_ratio
            base_growth_rate = allowed_roe * retention_rate

            # Adjust for rate base growth (enables higher dividend growth)
            dividend_growth_rate = (base_growth_rate * 0.7) + (rate_base_growth * 0.3)

            # Cap growth rate at reasonable levels for utilities
            dividend_growth_rate = min(0.06, max(0.02, dividend_growth_rate))

            # Terminal growth rate (long-term)
            terminal_growth = 0.025  # 2.5% long-term growth (slightly above inflation)

            # Calculate discount rate
            # Utilities typically have lower discount rates due to stable cash flows
            base_discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})
            discount_rate = base_discount_rate * 0.9  # Utilities typically 10% discount vs other energy

            # Ensure minimum spread between growth and discount rate
            if discount_rate - terminal_growth < 0.03:
                terminal_growth = discount_rate - 0.03

            # Apply Gordon Growth Model for utility (stable growth justifies this approach)
            value_per_share = current_dividend * (1 + dividend_growth_rate) / (discount_rate - terminal_growth)

            return {
                'method': 'utility_ddm',
                'value_per_share': value_per_share,
                'current_dividend': current_dividend,
                'dividend_growth_rate': dividend_growth_rate,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
                'allowed_roe': allowed_roe,
                'rate_base_growth': rate_base_growth,
                'payout_ratio': payout_ratio
            }

        except Exception as e:
            logger.error(f"Error in utility DDM valuation for {ticker}: {e}")
            return {
                'method': 'utility_ddm',
                'value_per_share': None,
                'error': str(e)
            }

    def _regulated_asset_base_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                        utility_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a utility company based on its Regulated Asset Base (RAB),
        which is a key valuation methodology for regulated utilities
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get key metrics
            rab = utility_metrics.get('Regulated_Asset_Base')
            if not rab:
                # If RAB not directly available, estimate from PPE
                ppe = utility_metrics.get('PPE')
                if ppe:
                    rab = ppe * 0.85  # Assume 85% of PPE is regulated
                else:
                    return {
                        'method': 'regulated_asset_base',
                        'value_per_share': None,
                        'error': 'Unable to determine Regulated Asset Base'
                    }

            # Get utility-specific metrics
            utility_specific = utility_metrics.get('Utility_Metrics', {})
            allowed_roe = utility_specific.get('Allowed_ROE', 0.095)  # Allowed return on equity
            rate_base_growth = utility_specific.get('Rate_Base_Growth', 0.04)  # Rate base growth

            # Calculate debt
            debt = utility_metrics.get('Debt', 0)

            # Typical utility capital structure
            equity_ratio = 0.5  # 50% equity in rate base
            debt_ratio = 0.5  # 50% debt in rate base

            # Calculate company value
            equity_value_in_rab = rab * equity_ratio

            # Apply a premium or discount to RAB based on:
            # 1. Allowed ROE vs. cost of equity
            # 2. Rate base growth potential
            # 3. Regulatory environment quality

            # Base case: RAB at book value (1.0x multiple)
            rab_multiple = 1.0

            # Adjust for allowed ROE
            # Higher allowed ROE = premium to RAB
            if allowed_roe > 0.10:
                roe_adjustment = 0.10  # 10% premium for higher allowed ROE
            elif allowed_roe < 0.09:
                roe_adjustment = -0.10  # 10% discount for lower allowed ROE
            else:
                roe_adjustment = 0

            # Adjust for rate base growth
            # Higher growth = premium to RAB
            if rate_base_growth > 0.05:
                growth_adjustment = 0.15  # 15% premium for high growth
            elif rate_base_growth < 0.03:
                growth_adjustment = -0.05  # 5% discount for low growth
            else:
                growth_adjustment = 0

            # Apply adjustments
            rab_multiple += roe_adjustment + growth_adjustment

            # Calculate adjusted RAB value
            adjusted_rab_value = equity_value_in_rab * rab_multiple

            # Calculate equity value (RAB equity portion - net debt)
            equity_value = adjusted_rab_value

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'regulated_asset_base',
                'value_per_share': value_per_share,
                'rab': rab,
                'equity_value_in_rab': equity_value_in_rab,
                'rab_multiple': rab_multiple,
                'roe_adjustment': roe_adjustment,
                'growth_adjustment': growth_adjustment,
                'adjusted_rab_value': adjusted_rab_value,
                'equity_value': equity_value,
                'allowed_roe': allowed_roe,
                'rate_base_growth': rate_base_growth
            }

        except Exception as e:
            logger.error(f"Error in regulated asset base valuation for {ticker}: {e}")
            return {
                'method': 'regulated_asset_base',
                'value_per_share': None,
                'error': str(e)
            }

    def _utility_dcf(self, ticker: str, financial_data: Dict[str, Any],
                     utility_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        DCF valuation for utility companies with adjustments for regulated returns
        and stable cash flows
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get utility-specific metrics
            utility_specific = utility_metrics.get('Utility_Metrics', {})
            allowed_roe = utility_specific.get('Allowed_ROE', 0.095)
            rate_base_growth = utility_specific.get('Rate_Base_Growth', 0.04)
            regulatory_lag = utility_specific.get('Regulatory_Lag', 0.5)
            customer_growth = utility_specific.get('Customer_Growth', 0.01)

            # Get generation mix (for electric utilities)
            generation_mix = utility_metrics.get('Generation_Mix', {})
            renewables_pct = generation_mix.get('Renewables', 0.25)

            # Get key financial metrics
            revenue = utility_metrics.get('Revenue')
            ebitda = utility_metrics.get('EBITDA')
            ebitda_margin = utility_metrics.get('EBITDA_Margin', 0.3)  # Default 30% if not available
            rab = utility_metrics.get('Regulated_Asset_Base')

            if not revenue or not ebitda:
                return {
                    'method': 'utility_dcf',
                    'value_per_share': None,
                    'error': 'Insufficient financial data for DCF valuation'
                }

            # Define DCF parameters
            forecast_years = 10  # Utilities typically use longer forecast period due to stable cash flows

            # Determine growth trajectory
            # Utility growth is typically tied to:
            # 1. Rate base growth (capital investments)
            # 2. Customer growth
            # 3. Regulatory approvals for rate increases

            # Calculate base revenue growth
            base_revenue_growth = (rate_base_growth * 0.7) + (customer_growth * 0.3)

            # Adjust for renewables trend
            # Higher renewable mix typically enables faster rate base growth
            renewables_adjustment = (renewables_pct - 0.25) * 0.1  # 0.1% higher growth per 10% above baseline
            adjusted_revenue_growth = base_revenue_growth + renewables_adjustment

            # Project revenue and EBITDA
            forecasted_revenue = []
            forecasted_ebitda = []

            current_revenue = revenue
            current_ebitda_margin = ebitda_margin

            for year in range(forecast_years):
                # Calculate year's growth rate (typically declining over time)
                year_growth = adjusted_revenue_growth * (1 - year / (forecast_years * 2))

                # Update revenue
                current_revenue *= (1 + year_growth)
                forecasted_revenue.append(current_revenue)

                # Calculate EBITDA (utilities have stable margins)
                # Allow for slight margin expansion in early years if investing in renewables
                if year < 3 and renewables_pct > 0.3:
                    margin_expansion = 0.002  # 0.2% per year
                else:
                    margin_expansion = 0

                current_ebitda_margin = min(0.4, current_ebitda_margin + margin_expansion)
                current_ebitda = current_revenue * current_ebitda_margin
                forecasted_ebitda.append(current_ebitda)

            # Project capex (critical for utilities)
            forecasted_capex = []

            # For utilities, capex is closely tied to rate base growth
            for i, revenue_value in enumerate(forecasted_revenue):
                # Base capex as percentage of revenue
                base_capex_pct = 0.25  # 25% of revenue

                # Additional capex for renewables transition
                if renewables_pct > 0.3 and i < 5:
                    renewables_capex = 0.05  # Additional 5% of revenue
                else:
                    renewables_capex = 0

                total_capex_pct = base_capex_pct + renewables_capex
                total_capex = revenue_value * total_capex_pct

                forecasted_capex.append(total_capex)

            # Calculate free cash flow
            forecasted_fcf = []

            for i, ebitda_value in enumerate(forecasted_ebitda):
                # Estimate D&A (typically high for utilities, ~10-15% of revenue)
                revenue_value = forecasted_revenue[i]
                depreciation = revenue_value * 0.12  # 12% of revenue

                # Calculate income taxes
                taxable_income = ebitda_value - depreciation
                effective_tax_rate = 0.25  # Standard corporate rate
                income_taxes = taxable_income * effective_tax_rate

                # Working capital changes (typically small for utilities)
                if i > 0:
                    revenue_change = (forecasted_revenue[i] - forecasted_revenue[i - 1]) / forecasted_revenue[i - 1]
                    wc_change = forecasted_revenue[i] * 0.01 * revenue_change
                else:
                    wc_change = 0

                # Calculate FCF
                fcf = ebitda_value - forecasted_capex[i] - income_taxes - wc_change
                forecasted_fcf.append(fcf)

            # Calculate discount rate (typically lower for utilities)
            base_discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})
            discount_rate = base_discount_rate * 0.85  # Utilities typically 15% discount vs other energy

            # Terminal growth rate
            terminal_growth = 0.02  # 2% long-term growth (infrastructure assets typically grow with inflation)

            # Calculate terminal value
            terminal_fcf = forecasted_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecasted_fcf))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'utility_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'base_revenue_growth': base_revenue_growth,
                'adjusted_revenue_growth': adjusted_revenue_growth,
                'renewables_adjustment': renewables_adjustment,
                'forecasted_revenue': forecasted_revenue,
                'forecasted_ebitda': forecasted_ebitda,
                'forecasted_capex': forecasted_capex,
                'forecasted_fcf': forecasted_fcf,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in utility DCF valuation for {ticker}: {e}")
            return {
                'method': 'utility_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_renewable_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics for renewable energy companies"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get most recent financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None else None

            # Basic financial metrics
            if 'Total Revenue' in latest_income.index:
                metrics['Revenue'] = latest_income['Total Revenue']

            if 'EBITDA' in latest_income.index:
                metrics['EBITDA'] = latest_income['EBITDA']
            elif 'Operating Income' in latest_income.index and 'Depreciation & Amortization' in latest_income.index:
                metrics['EBITDA'] = latest_income['Operating Income'] + latest_income['Depreciation & Amortization']

            # Calculate EBITDA margin
            if 'Revenue' in metrics and 'EBITDA' in metrics and metrics['Revenue'] > 0:
                metrics['EBITDA_Margin'] = metrics['EBITDA'] / metrics['Revenue']

            # Debt metrics
            if 'Total Debt' in latest_balance.index:
                metrics['Debt'] = latest_balance['Total Debt']

                if 'EBITDA' in metrics and metrics['EBITDA'] > 0:
                    metrics['Debt_to_EBITDA'] = metrics['Debt'] / metrics['EBITDA']

            # Asset metrics
            if 'Property Plant and Equipment' in latest_balance.index:
                metrics['PPE'] = latest_balance['Property Plant and Equipment']

                if 'Revenue' in metrics and metrics['Revenue'] > 0:
                    metrics['Asset_Intensity'] = metrics['PPE'] / metrics['Revenue']

            # Renewable-specific operational metrics
            renewable_metrics = {
                'Installed_Capacity_MW': company_info.get('installed_capacity_mw', 1000),
                'Capacity_Factor': company_info.get('capacity_factor', 0.35),
                'PPA_Coverage': company_info.get('ppa_coverage', 0.80),  # % of output under Power Purchase Agreements
                'Average_PPA_Duration': company_info.get('avg_ppa_duration', 15),  # years
                'Average_PPA_Price': company_info.get('avg_ppa_price', 40),  # $/MWh
                'Development_Pipeline_MW': company_info.get('development_pipeline_mw', 2000)
            }
            metrics['Renewable_Metrics'] = renewable_metrics

            # Generation mix
            generation_mix = {
                'Solar': company_info.get('solar_pct', 0.40),
                'Wind': company_info.get('wind_pct', 0.40),
                'Hydro': company_info.get('hydro_pct', 0.10),
                'Storage': company_info.get('storage_pct', 0.05),
                'Other': company_info.get('other_renewable_pct', 0.05)
            }
            metrics['Generation_Mix'] = generation_mix

            # Calculate key performance metrics
            if 'Installed_Capacity_MW' in renewable_metrics and 'Revenue' in metrics:
                metrics['Revenue_per_MW'] = metrics['Revenue'] / renewable_metrics['Installed_Capacity_MW']

            if 'EBITDA' in metrics and 'Installed_Capacity_MW' in renewable_metrics:
                metrics['EBITDA_per_MW'] = metrics['EBITDA'] / renewable_metrics['Installed_Capacity_MW']

            return metrics

        except Exception as e:
            logger.error(f"Error calculating renewable metrics: {e}")
            return {}

    def _renewable_project_dcf(self, ticker: str, financial_data: Dict[str, Any],
                               renewable_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        DCF valuation specifically for renewable energy companies based on
        project-level economics
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get renewable-specific metrics
            renewable_specific = renewable_metrics.get('Renewable_Metrics', {})
            installed_capacity = renewable_specific.get('Installed_Capacity_MW', 0)
            capacity_factor = renewable_specific.get('Capacity_Factor', 0.35)
            ppa_coverage = renewable_specific.get('PPA_Coverage', 0.80)
            avg_ppa_duration = renewable_specific.get('Average_PPA_Duration', 15)
            avg_ppa_price = renewable_specific.get('Average_PPA_Price', 40)
            pipeline_mw = renewable_specific.get('Development_Pipeline_MW', 0)

            # Get generation mix
            generation_mix = renewable_metrics.get('Generation_Mix', {})

            # Get basic financials
            ebitda = renewable_metrics.get('EBITDA')
            ebitda_per_mw = renewable_metrics.get('EBITDA_per_MW')

            if not installed_capacity or not ebitda:
                return {
                    'method': 'renewable_project_dcf',
                    'value_per_share': None,
                    'error': 'Insufficient data for project-based valuation'
                }

            # Calculate key project economics
            # 1. Annual energy production (MWh)
            hours_per_year = 8760  # hours in a year
            annual_mwh = installed_capacity * capacity_factor * hours_per_year

            # 2. Revenue based on PPA and market exposure
            ppa_revenue = annual_mwh * ppa_coverage * avg_ppa_price
            market_exposure = 1 - ppa_coverage
            market_price = 35  # Assumed market price ($/MWh)
            market_revenue = annual_mwh * market_exposure * market_price
            total_revenue = ppa_revenue + market_revenue

            # 3. Operating costs
            # Typical costs per MWh
            if 'Solar' in generation_mix and generation_mix['Solar'] > 0.5:
                opex_per_mwh = 10  # $10/MWh for solar (lower O&M)
            elif 'Wind' in generation_mix and generation_mix['Wind'] > 0.5:
                opex_per_mwh = 15  # $15/MWh for wind
            else:
                opex_per_mwh = 12  # Mixed portfolio

            annual_opex = annual_mwh * opex_per_mwh

            # 4. EBITDA calculation
            calculated_ebitda = total_revenue - annual_opex

            # Reconcile with reported EBITDA if available
            if ebitda and abs(calculated_ebitda - ebitda) / ebitda > 0.2:
                # If calculated is significantly different from reported, adjust opex
                opex_adjustment = (calculated_ebitda - ebitda) / annual_mwh
                opex_per_mwh += opex_adjustment
                annual_opex = annual_mwh * opex_per_mwh
                calculated_ebitda = total_revenue - annual_opex

            # 5. Project DCF parameters
            project_life = 25  # years for typical renewable assets
            forecast_years = min(project_life, 20)  # Cap forecast at 20 years

            # Asset aging factors
            # Energy production typically declines with age
            aging_factors = [1.0] * forecast_years
            for i in range(forecast_years):
                if i > 0:
                    # Simple linear degradation
                    aging_factors[i] = aging_factors[i - 1] - 0.005  # 0.5% annual degradation

            # Calculate discount rate
            base_discount_rate = self._calculate_energy_discount_rate(ticker, financial_data, {})

            # Adjust discount rate based on PPA coverage (lower risk with higher PPA)
            discount_rate = base_discount_rate * (1 - (ppa_coverage - 0.5) * 0.1)

            # Project cashflows
            forecasted_revenue = []
            forecasted_ebitda = []
            forecasted_fcf = []

            for year in range(forecast_years):
                # Calculate year's production with aging
                year_production = annual_mwh * aging_factors[year]

                # Calculate revenue
                if year < avg_ppa_duration:
                    # PPA still in effect
                    year_ppa_revenue = year_production * ppa_coverage * avg_ppa_price
                else:
                    # After PPA expiry, assume merchant pricing with slight premium
                    year_ppa_revenue = year_production * ppa_coverage * market_price * 1.1

                year_market_revenue = year_production * market_exposure * market_price
                year_revenue = year_ppa_revenue + year_market_revenue
                forecasted_revenue.append(year_revenue)

                # Calculate EBITDA
                year_opex = year_production * opex_per_mwh
                year_ebitda = year_revenue - year_opex
                forecasted_ebitda.append(year_ebitda)

                # Capital expenditures
                # Renewable projects have high upfront capex but low ongoing capex
                if year < 3:
                    maintenance_capex = year_revenue * 0.05  # 5% of revenue
                else:
                    maintenance_capex = year_revenue * 0.03  # 3% of revenue

                # Calculate taxes (simplified)
                depreciation = (installed_capacity * 1.5e6) / 20  # $1.5M per MW over 20 years
                taxable_income = year_ebitda - depreciation
                tax_rate = 0.25
                income_taxes = max(0, taxable_income * tax_rate)

                # Free cash flow
                fcf = year_ebitda - maintenance_capex - income_taxes
                forecasted_fcf.append(fcf)

            # Terminal value calculation
            # For renewables, terminal value is lower due to asset degradation
            terminal_fcf = forecasted_fcf[-1] * 0.9  # 10% reduction to account for aging

            # Lower terminal growth for aging assets
            terminal_growth = 0.01  # 1% long-term growth

            terminal_value = terminal_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(forecasted_fcf))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Operating assets value
            operating_assets_value = present_value_fcf + present_value_terminal

            # Add value of development pipeline
            # Typically valued at $100K-$300K per MW depending on stage and quality
            # Use differentiated values based on technology mix
            if 'Solar' in generation_mix and generation_mix['Solar'] > 0.5:
                pipeline_value_per_mw = 0.1e6  # $100K per MW for solar
            elif 'Wind' in generation_mix and generation_mix['Wind'] > 0.5:
                pipeline_value_per_mw = 0.15e6  # $150K per MW for wind
            else:
                pipeline_value_per_mw = 0.12e6  # Mixed portfolio

            pipeline_value = pipeline_mw * pipeline_value_per_mw

            # Calculate enterprise value
            enterprise_value = operating_assets_value + pipeline_value

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'renewable_project_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'operating_assets_value': operating_assets_value,
                'pipeline_value': pipeline_value,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'project_metrics': {
                    'installed_capacity_mw': installed_capacity,
                    'annual_mwh': annual_mwh,
                    'capacity_factor': capacity_factor,
                    'ppa_coverage': ppa_coverage,
                    'avg_ppa_duration': avg_ppa_duration,
                    'avg_ppa_price': avg_ppa_price
                },
                'forecasted_revenue': forecasted_revenue,
                'forecasted_ebitda': forecasted_ebitda,
                'forecasted_fcf': forecasted_fcf,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in renewable project DCF valuation for {ticker}: {e}")
            return {
                'method': 'renewable_project_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _renewable_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                       renewable_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a renewable energy company using sector-specific multiples,
        particularly focused on capacity-based metrics
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get renewable-specific metrics
            renewable_specific = renewable_metrics.get('Renewable_Metrics', {})
            installed_capacity = renewable_specific.get('Installed_Capacity_MW', 0)
            capacity_factor = renewable_specific.get('Capacity_Factor', 0.35)
            ppa_coverage = renewable_specific.get('PPA_Coverage', 0.80)
            pipeline_mw = renewable_specific.get('Development_Pipeline_MW', 0)

            # Get generation mix
            generation_mix = renewable_metrics.get('Generation_Mix', {})

            # Get key financial metrics
            ebitda = renewable_metrics.get('EBITDA')
            revenue = renewable_metrics.get('Revenue')
            ebitda_per_mw = renewable_metrics.get('EBITDA_per_MW')

            if not installed_capacity or not ebitda:
                return {
                    'method': 'renewable_multiples',
                    'value_per_share': None,
                    'error': 'Insufficient data for multiples-based valuation'
                }

            # Define base multiples for renewable companies
            base_multiples = {
                'EV_EBITDA': 11.0,  # Renewable companies typically trade at premium multiples
                'EV_per_MW': 1.5e6,  # $1.5 million per MW of installed capacity
                'Pipeline_per_MW': 0.15e6,  # $150K per MW of pipeline
                'EV_Revenue': 3.0
            }

            # Adjust multiples based on generation mix and contract quality
            mix_adjustment = 0

            # Solar typically trades at higher multiples than wind
            if 'Solar' in generation_mix:
                mix_adjustment += generation_mix['Solar'] * 0.1  # 10% premium for solar content

            # Storage adds premium
            if 'Storage' in generation_mix:
                mix_adjustment += generation_mix['Storage'] * 0.2  # 20% premium for storage content

            # PPA coverage adds premium (lower risk)
            contract_adjustment = (ppa_coverage - 0.7) * 0.15  # Premium for above-average PPA coverage

            # Capacity factor adjustment
            # Higher capacity factor = better asset quality
            cf_adjustment = (capacity_factor - 0.35) * 0.3  # Adjustment based on deviation from average

            # Apply all adjustments
            total_adjustment = mix_adjustment + contract_adjustment + cf_adjustment

            adjusted_multiples = {
                'EV_EBITDA': base_multiples['EV_EBITDA'] * (1 + total_adjustment),
                'EV_per_MW': base_multiples['EV_per_MW'] * (1 + total_adjustment),
                'Pipeline_per_MW': base_multiples['Pipeline_per_MW'] * (1 + total_adjustment),
                'EV_Revenue': base_multiples['EV_Revenue'] * (1 + total_adjustment)
            }

            # Calculate valuations using different metrics
            valuations = {}

            # EV/EBITDA valuation
            if ebitda > 0:
                valuations['EV_EBITDA'] = {
                    'multiple': adjusted_multiples['EV_EBITDA'],
                    'metric_value': ebitda,
                    'enterprise_value': ebitda * adjusted_multiples['EV_EBITDA']
                }

            # EV per MW valuation
            if installed_capacity > 0:
                valuations['EV_per_MW'] = {
                    'multiple': adjusted_multiples['EV_per_MW'],
                    'metric_value': installed_capacity,
                    'enterprise_value': installed_capacity * adjusted_multiples['EV_per_MW']
                }

            # Pipeline valuation
            if pipeline_mw > 0:
                pipeline_value = pipeline_mw * adjusted_multiples['Pipeline_per_MW']
                valuations['Pipeline_Value'] = {
                    'multiple': adjusted_multiples['Pipeline_per_MW'],
                    'metric_value': pipeline_mw,
                    'value': pipeline_value
                }
            else:
                pipeline_value = 0

            # EV/Revenue valuation
            if revenue > 0:
                valuations['EV_Revenue'] = {
                    'multiple': adjusted_multiples['EV_Revenue'],
                    'metric_value': revenue,
                    'enterprise_value': revenue * adjusted_multiples['EV_Revenue']
                }

            # Calculate weighted average enterprise value
            weights = {
                'EV_EBITDA': 0.4,
                'EV_per_MW': 0.4,
                'EV_Revenue': 0.2
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                return {
                    'method': 'renewable_multiples',
                    'value_per_share': None,
                    'error': 'Insufficient metrics for valuation'
                }

            # Add pipeline value
            enterprise_value += pipeline_value

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'renewable_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'pipeline_value': pipeline_value,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'total_adjustment': total_adjustment,
                'mix_adjustment': mix_adjustment,
                'contract_adjustment': contract_adjustment,
                'cf_adjustment': cf_adjustment,
                'valuations': valuations,
                'weights': weights,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in renewable multiples valuation for {ticker}: {e}")
            return {
                'method': 'renewable_multiples',
                'value_per_share': None,
                'error': str(e)
            }

    def _renewable_options_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                     renewable_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a renewable energy company using real options approach,
        particularly useful for development-stage projects
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get renewable-specific metrics
            renewable_specific = renewable_metrics.get('Renewable_Metrics', {})
            installed_capacity = renewable_specific.get('Installed_Capacity_MW', 0)
            pipeline_mw = renewable_specific.get('Development_Pipeline_MW', 0)

            # Get generation mix
            generation_mix = renewable_metrics.get('Generation_Mix', {})

            # 1. Base case DCF value for operating assets
            # This is simplified - in practice, would call _renewable_project_dcf
            # and extract the operating assets value

            # Simplified calculation of operating assets value
            operating_ebitda = renewable_metrics.get('EBITDA', 0)
            ebitda_multiple = 9.0  # Conservative multiple for base calculation
            operating_assets_value = operating_ebitda * ebitda_multiple

            # 2. Development pipeline stages (simplified estimate)
            # In practice, would come from detailed project data
            early_stage_mw = pipeline_mw * 0.4  # 40% early stage
            mid_stage_mw = pipeline_mw * 0.4  # 40% mid stage
            late_stage_mw = pipeline_mw * 0.2  # 20% late stage

            # 3. Success probabilities by stage
            early_success_prob = 0.3  # 30% probability of completion
            mid_success_prob = 0.6  # 60% probability of completion
            late_success_prob = 0.85  # 85% probability of completion

            # 4. Expected value per MW by development stage
            # Value increases as projects advance through development
            early_stage_value_per_mw = 0.05e6  # $50K per MW
            mid_stage_value_per_mw = 0.2e6  # $200K per MW
            late_stage_value_per_mw = 0.5e6  # $500K per MW

            # 5. Option values (accounting for strategic flexibility)
            # Real options premium factors
            # Market volatility enhances option value
            volatility_premium = 1.2  # 20% premium for options value due to market volatility

            # Calculate expected values with options premium
            early_stage_value = early_stage_mw * early_stage_value_per_mw * early_success_prob * volatility_premium
            mid_stage_value = mid_stage_mw * mid_stage_value_per_mw * mid_success_prob * volatility_premium
            late_stage_value = late_stage_mw * late_stage_value_per_mw * late_success_prob * volatility_premium

            # Total development pipeline value
            pipeline_value = early_stage_value + mid_stage_value + late_stage_value

            # 6. Growth options (future expansion potential)
            # Value of potential to expand beyond current pipeline
            growth_options_value = operating_assets_value * 0.1  # Simplified: 10% of operating assets

            # Total enterprise value
            enterprise_value = operating_assets_value + pipeline_value + growth_options_value

            # Adjust for net debt
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'renewable_options',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'components': {
                    'operating_assets_value': operating_assets_value,
                    'pipeline_value': pipeline_value,
                    'growth_options_value': growth_options_value
                },
                'pipeline_details': {
                    'early_stage': {
                        'mw': early_stage_mw,
                        'success_prob': early_success_prob,
                        'value_per_mw': early_stage_value_per_mw,
                        'expected_value': early_stage_value
                    },
                    'mid_stage': {
                        'mw': mid_stage_mw,
                        'success_prob': mid_success_prob,
                        'value_per_mw': mid_stage_value_per_mw,
                        'expected_value': mid_stage_value
                    },
                    'late_stage': {
                        'mw': late_stage_mw,
                        'success_prob': late_success_prob,
                        'value_per_mw': late_stage_value_per_mw,
                        'expected_value': late_stage_value
                    }
                },
                'volatility_premium': volatility_premium,
                'net_debt': net_debt
            }

        except Exception as e:
            logger.error(f"Error in renewable options valuation for {ticker}: {e}")
            return {
                'method': 'renewable_options',
                'value_per_share': None,
                'error': str(e)
            }

    # Helper methods for energy sector

    def _determine_commodity_exposure(self, ticker: str, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """Determine the company's exposure to different energy commodities"""
        try:
            company_info = financial_data.get('company_info', {})

            # Default exposure levels
            exposure = {
                'oil': 0.5,
                'natural_gas': 0.3,
                'electricity': 0.1,
                'coal': 0.1
            }

            # Extract industry info
            industry = company_info.get('industry', '').lower()

            # Try to determine from industry description
            if 'oil' in industry and 'gas' not in industry:
                exposure = {'oil': 0.8, 'natural_gas': 0.1, 'electricity': 0.05, 'coal': 0.05}
            elif 'gas' in industry and 'oil' not in industry:
                exposure = {'oil': 0.2, 'natural_gas': 0.7, 'electricity': 0.05, 'coal': 0.05}
            elif 'coal' in industry:
                exposure = {'oil': 0.1, 'natural_gas': 0.2, 'electricity': 0.1, 'coal': 0.6}
            elif 'utility' in industry or 'electric' in industry:
                exposure = {'oil': 0.05, 'natural_gas': 0.3, 'electricity': 0.6, 'coal': 0.05}
            elif 'renewable' in industry or 'solar' in industry or 'wind' in industry:
                exposure = {'oil': 0.05, 'natural_gas': 0.1, 'electricity': 0.8, 'coal': 0.05}
            elif 'refin' in industry:
                exposure = {'oil': 0.7, 'natural_gas': 0.2, 'electricity': 0.05, 'coal': 0.05}

            # Normalize exposure
            total = sum(exposure.values())
            for commodity in exposure:
                exposure[commodity] /= total

            return exposure

        except Exception as e:
            logger.error(f"Error determining commodity exposure for {ticker}: {e}")
            return {'oil': 0.5, 'natural_gas': 0.3, 'electricity': 0.1, 'coal': 0.1}  # Default values

    def _estimate_energy_growth_rate(self, historical_fcf: pd.Series,
                                     commodity_exposure: Dict[str, float]) -> float:
        """Estimate growth rate for energy companies based on commodity exposure"""
        try:
            # Start with base growth rate from historical data
            base_growth = self._estimate_growth_rate(historical_fcf)

            # Adjust based on commodity exposure and price outlook
            # These growth modifiers would ideally come from commodity price forecasts
            commodity_growth_modifiers = {
                'oil': 0.2,  # Oil-focused more cyclical
                'natural_gas': 0.5,  # Natural gas more stable
                'electricity': 0.7,  # Electricity most stable
                'coal': -0.3  # Coal facing structural decline
            }

            # Calculate weighted growth modifier
            weighted_modifier = 0
            for commodity, exposure in commodity_exposure.items():
                weighted_modifier += exposure * commodity_growth_modifiers[commodity]

            # Apply modifier to base growth
            adjusted_growth = base_growth * (1 + weighted_modifier)

            # Cap growth to reasonable range
            return max(0.01, min(0.15, adjusted_growth))

        except Exception as e:
            logger.error(f"Error estimating energy growth rate: {e}")
            return 0.03  # Default 3% growth

    def _calculate_energy_discount_rate(self, ticker: str, financial_data: Dict[str, Any],
                                        commodity_exposure: Dict[str, float]) -> float:
        """Calculate discount rate for energy companies with sector-specific risk adjustments"""
        try:
            # Start with standard discount rate
            base_discount_rate = self._calculate_discount_rate(ticker, financial_data)

            # If not available, use default
            if not base_discount_rate:
                base_discount_rate = 0.10  # 10% base

            # Risk premium adjustments based on commodity exposure
            # Higher exposure to volatile commodities = higher discount rate
            commodity_risk_premiums = {
                'oil': 0.02,  # +2% for oil exposure
                'natural_gas': 0.01,  # +1% for gas exposure
                'electricity': 0.005,  # +0.5% for electricity exposure
                'coal': 0.03  # +3% for coal exposure (higher risk due to transition)
            }

            # Calculate weighted risk premium
            weighted_premium = 0
            if commodity_exposure:
                for commodity, exposure in commodity_exposure.items():
                    weighted_premium += exposure * commodity_risk_premiums.get(commodity, 0)

            # Apply risk premium to base rate
            adjusted_discount_rate = base_discount_rate + weighted_premium

            # Cap to reasonable range
            return max(0.07, min(0.18, adjusted_discount_rate))

        except Exception as e:
            logger.error(f"Error calculating energy discount rate for {ticker}: {e}")
            return 0.12  # Default 12% discount rate