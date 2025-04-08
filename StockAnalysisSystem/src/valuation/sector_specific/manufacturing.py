import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directories to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('manufacturing_sector')


class ManufacturingSectorValuation(AdvancedDCFValuation):
    """
    Specialized valuation models for manufacturing sector companies.

    This class provides valuation methods tailored to different manufacturing sub-sectors:
    - Heavy Manufacturing (Industrial Equipment, Machinery)
    - Automotive Manufacturing
    - Aerospace & Defense
    - Electronics Manufacturing
    - Consumer Goods Manufacturing
    - Chemical/Materials Manufacturing

    Each sub-sector has specific metrics and value drivers that are incorporated
    into the valuation methods.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize manufacturing sector valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized ManufacturingSectorValuation")

        # Manufacturing sub-sector specific parameters
        self.manufacturing_subsector_parameters = {
            "Heavy_Manufacturing": {
                "revenue_growth": 0.03,
                "terminal_growth": 0.02,
                "discount_rate_premium": 0.01,
                "capex_to_revenue": 0.08,
                "roic_target": 0.12,
                "important_ratios": ["fixed_asset_turnover", "operating_margin", "roic"]
            },
            "Automotive": {
                "revenue_growth": 0.04,
                "terminal_growth": 0.02,
                "discount_rate_premium": 0.015,
                "capex_to_revenue": 0.07,
                "roic_target": 0.10,
                "important_ratios": ["inventory_turnover", "gross_margin", "r_and_d_to_revenue"]
            },
            "Aerospace_Defense": {
                "revenue_growth": 0.05,
                "terminal_growth": 0.025,
                "discount_rate_premium": 0.01,
                "capex_to_revenue": 0.06,
                "roic_target": 0.15,
                "important_ratios": ["backlog_to_revenue", "r_and_d_to_revenue", "operating_margin"]
            },
            "Electronics": {
                "revenue_growth": 0.06,
                "terminal_growth": 0.03,
                "discount_rate_premium": 0.02,
                "capex_to_revenue": 0.09,
                "roic_target": 0.18,
                "important_ratios": ["r_and_d_to_revenue", "inventory_turnover", "gross_margin"]
            },
            "Consumer_Goods": {
                "revenue_growth": 0.04,
                "terminal_growth": 0.02,
                "discount_rate_premium": 0.005,
                "capex_to_revenue": 0.05,
                "roic_target": 0.14,
                "important_ratios": ["inventory_turnover", "gross_margin", "working_capital_ratio"]
            },
            "Chemical_Materials": {
                "revenue_growth": 0.03,
                "terminal_growth": 0.02,
                "discount_rate_premium": 0.015,
                "capex_to_revenue": 0.10,
                "roic_target": 0.11,
                "important_ratios": ["capacity_utilization", "fixed_asset_turnover", "operating_margin"]
            }
        }

        # Cyclical industry flags (affects valuation)
        self.cyclical_subsectors = ["Automotive", "Chemical_Materials", "Heavy_Manufacturing"]

        # Asset-intensive subsectors
        self.asset_intensive_subsectors = ["Heavy_Manufacturing", "Automotive", "Chemical_Materials"]

    def detect_manufacturing_subsector(self, ticker: str, financial_data: Dict[str, Any]) -> str:
        """
        Determine the manufacturing sub-sector of a company based on available data

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and other data

        Returns:
            String indicating the detected manufacturing sub-sector
        """
        try:
            # Extract company info from financial data
            company_info = financial_data.get('company_info', {})
            industry = company_info.get('industry', '').lower()
            business_summary = company_info.get('description', '').lower()

            # Check for keywords in industry and business description
            auto_keywords = ['automotive', 'auto parts', 'car manufacturer', 'vehicle']
            aero_defense_keywords = ['aerospace', 'defense', 'aircraft', 'aviation', 'space', 'military']
            electronics_keywords = ['electronics', 'semiconductor', 'electronic components', 'circuit',
                                    'electronics manufacturing']
            consumer_goods_keywords = ['consumer goods', 'appliance', 'household', 'packaged goods',
                                       'consumer products']
            chemical_keywords = ['chemical', 'materials', 'specialty material', 'plastics', 'polymers', 'composites']
            heavy_keywords = ['industrial equipment', 'machinery', 'heavy equipment', 'industrial machinery']

            # Determine sub-sector based on keywords in industry and description
            if any(keyword in industry or keyword in business_summary for keyword in auto_keywords):
                return "Automotive"

            if any(keyword in industry or keyword in business_summary for keyword in aero_defense_keywords):
                return "Aerospace_Defense"

            if any(keyword in industry or keyword in business_summary for keyword in electronics_keywords):
                return "Electronics"

            if any(keyword in industry or keyword in business_summary for keyword in consumer_goods_keywords):
                return "Consumer_Goods"

            if any(keyword in industry or keyword in business_summary for keyword in chemical_keywords):
                return "Chemical_Materials"

            if any(keyword in industry or keyword in business_summary for keyword in heavy_keywords):
                return "Heavy_Manufacturing"

            # If still not determined, look at financials for clues
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is not None and balance_sheet is not None:
                # High R&D typically indicates Electronics or Aerospace
                if 'Research and Development' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    r_and_d = income_stmt.iloc[0]['Research and Development']
                    r_and_d_ratio = r_and_d / revenue if revenue > 0 else 0

                    if r_and_d_ratio > 0.10:
                        return "Electronics"  # Very high R&D
                    elif r_and_d_ratio > 0.06:
                        return "Aerospace_Defense"  # High R&D

                # High fixed assets typically indicates Heavy Manufacturing
                if 'Property Plant and Equipment' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                    ppe = balance_sheet.iloc[0]['Property Plant and Equipment']
                    total_assets = balance_sheet.iloc[0]['Total Assets']
                    ppe_ratio = ppe / total_assets if total_assets > 0 else 0

                    if ppe_ratio > 0.40:
                        return "Heavy_Manufacturing"

                # High inventory typically indicates Consumer Goods
                if 'Inventory' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                    inventory = balance_sheet.iloc[0]['Inventory']
                    total_assets = balance_sheet.iloc[0]['Total Assets']
                    inventory_ratio = inventory / total_assets if total_assets > 0 else 0

                    if inventory_ratio > 0.25:
                        return "Consumer_Goods"

            # Default to Heavy Manufacturing if unable to determine
            logger.info(
                f"Unable to precisely determine manufacturing sub-sector for {ticker}, defaulting to Heavy Manufacturing")
            return "Heavy_Manufacturing"

        except Exception as e:
            logger.error(f"Error detecting manufacturing sub-sector for {ticker}: {e}")
            # Default to Heavy Manufacturing
            return "Heavy_Manufacturing"

    def manufacturing_dcf_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                    subsector: str = None) -> Dict[str, Any]:
        """
        Perform specialized DCF valuation for manufacturing companies

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Performing manufacturing DCF valuation for {ticker} as {subsector}")

            # Get sub-sector specific parameters
            if subsector in self.manufacturing_subsector_parameters:
                subsector_params = self.manufacturing_subsector_parameters[subsector]
            else:
                # Default to Heavy Manufacturing if sub-sector not recognized
                subsector_params = self.manufacturing_subsector_parameters["Heavy_Manufacturing"]
                logger.warning(
                    f"Unrecognized manufacturing sub-sector {subsector}, using Heavy Manufacturing parameters")

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Check required data
            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for manufacturing DCF valuation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Get base parameters
            params = self._get_dcf_parameters("Industrials")  # Use Industrials as base

            # Adjust parameters based on sub-sector
            adjusted_params = params.copy()
            adjusted_params['terminal_growth_rate'] = subsector_params['terminal_growth']

            # Modify discount rate based on sub-sector risk
            base_discount_rate = self._calculate_discount_rate(ticker, financial_data, "Industrials")
            if base_discount_rate:
                adjusted_discount_rate = base_discount_rate + subsector_params['discount_rate_premium']
            else:
                adjusted_discount_rate = params['default_discount_rate'] + subsector_params['discount_rate_premium']

            # Calculate manufacturing-specific metrics
            manufacturing_metrics = self._calculate_manufacturing_metrics(ticker, financial_data, subsector)

            # Forecast parameters
            forecast_years = params['forecast_years']

            # For cyclical industries, use normalized earnings/cash flows
            is_cyclical = subsector in self.cyclical_subsectors

            # Calculate initial growth rate with sector-specific adjustments
            if is_cyclical:
                # For cyclical industries, use a more conservative approach
                # Look at longer-term average growth rates rather than recent performance
                # which might be at a peak or trough of the cycle
                base_growth_rate = self._estimate_normalized_growth_rate(historical_fcf, income_stmt)
            else:
                base_growth_rate = self._estimate_growth_rate(historical_fcf)

            # Adjust growth based on manufacturing metrics and industry trends
            growth_adjustment = 0

            # Adjust for ROIC (Return on Invested Capital)
            if 'roic' in manufacturing_metrics:
                # Companies with higher ROIC tend to grow faster
                roic = manufacturing_metrics['roic']
                target_roic = subsector_params['roic_target']

                if roic > target_roic:
                    # Above average ROIC suggests sustainable growth
                    growth_adjustment += min((roic - target_roic) * 2, 0.02)
                else:
                    # Below average ROIC suggests challenges
                    growth_adjustment += max((roic - target_roic) * 2, -0.02)

            # Adjust for capacity utilization
            if 'capacity_utilization' in manufacturing_metrics:
                capacity_util = manufacturing_metrics['capacity_utilization']
                # High utilization suggests potential for growth through expansion
                if capacity_util > 0.85:
                    growth_adjustment += 0.01  # Room for growth through expansion
                elif capacity_util < 0.70:
                    growth_adjustment -= 0.01  # Struggling with utilization

            # Adjust for R&D investment (indicator of future growth)
            if 'r_and_d_to_revenue' in manufacturing_metrics:
                r_and_d_ratio = manufacturing_metrics['r_and_d_to_revenue']
                # Compare to industry average
                if subsector in ["Electronics", "Aerospace_Defense"]:
                    # These sectors rely heavily on R&D
                    benchmark = 0.08  # 8% of revenue
                else:
                    benchmark = 0.04  # 4% of revenue

                if r_and_d_ratio > benchmark:
                    growth_adjustment += min((r_and_d_ratio - benchmark) * 0.5, 0.015)

            # Final growth rate (base + metrics adjustment + subsector default)
            # Weight: 40% calculated growth, 40% metrics adjustment, 20% subsector default
            initial_growth_rate = (base_growth_rate * 0.4) + (growth_adjustment) + (
                        subsector_params['revenue_growth'] * 0.2)

            # Cap growth to reasonable range
            initial_growth_rate = max(0.01, min(0.20, initial_growth_rate))

            logger.info(f"Calculated initial growth rate for {ticker}: {initial_growth_rate:.2%}")

            # Starting FCF (most recent or normalized for cyclical industries)
            if is_cyclical and len(historical_fcf) >= 3:
                # For cyclical industries, use average of last 3 years for starting point
                last_fcf = historical_fcf.iloc[:3].mean()
            else:
                last_fcf = historical_fcf.iloc[0]

            # Forecast cash flows with manufacturing-specific adjustments
            forecasted_fcf = []

            for year in range(1, forecast_years + 1):
                # Apply declining growth over time
                if year <= 2:
                    growth = initial_growth_rate
                elif year <= 5:
                    # Linear decline from initial to steady state
                    growth = initial_growth_rate - ((initial_growth_rate - adjusted_params['terminal_growth_rate'])
                                                    * (year - 2) / 3)
                else:
                    growth = adjusted_params['terminal_growth_rate']

                # For asset-intensive subsectors, adjust FCF to account for higher reinvestment needs
                if subsector in self.asset_intensive_subsectors:
                    # Apply higher capital intensity in early years (growth phase)
                    if year <= 3:
                        capital_intensity_factor = 1.2  # 20% higher capital needs in growth phase
                    else:
                        capital_intensity_factor = 1.1  # 10% higher in later years
                else:
                    capital_intensity_factor = 1.0

                # Calculate FCF with growth and capital intensity adjustment
                fcf = last_fcf * (1 + growth) ** year / capital_intensity_factor

                # For cyclical industries, add cyclicality pattern
                if is_cyclical:
                    # Simplified cyclical pattern - adjust based on year in cycle
                    cycle_position = year % 5  # Assume 5-year industry cycle
                    if cycle_position == 1:  # Early upswing
                        cycle_factor = 1.10
                    elif cycle_position == 2:  # Peak
                        cycle_factor = 1.15
                    elif cycle_position == 3:  # Early downswing
                        cycle_factor = 0.95
                    elif cycle_position == 4:  # Trough
                        cycle_factor = 0.85
                    else:  # Recovery
                        cycle_factor = 1.00

                    fcf *= cycle_factor

                forecasted_fcf.append(fcf)

            # Calculate terminal value
            terminal_growth = adjusted_params['terminal_growth_rate']

            # For cyclical industries, use normalized terminal value
            if is_cyclical:
                # Use average of last 3 forecast years to normalize for cyclicality
                terminal_base = sum(forecasted_fcf[-3:]) / 3
                terminal_fcf = terminal_base * (1 + terminal_growth)
            else:
                terminal_fcf = forecasted_fcf[-1] * (1 + terminal_growth)

            terminal_value = terminal_fcf / (adjusted_discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + adjusted_discount_rate) ** year
                                    for year, fcf in enumerate(forecasted_fcf, 1))

            present_value_terminal = terminal_value / (1 + adjusted_discount_rate) ** forecast_years

            # Calculate enterprise and equity values
            enterprise_value = present_value_fcf + present_value_terminal
            net_debt = self._calculate_net_debt(balance_sheet)
            equity_value = enterprise_value - net_debt

            # For manufacturing companies, check for pension obligations (common off-balance sheet items)
            if 'pension_obligation' in manufacturing_metrics:
                equity_value -= manufacturing_metrics['pension_obligation']

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

            # Apply margin of safety (higher for more cyclical sub-sectors)
            safety_margin = adjusted_params['default_margin_of_safety']
            if is_cyclical:
                safety_margin *= 1.2  # 20% higher safety margin for cyclical industries

            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            # Prepare results
            result = {
                'company': ticker,
                'method': 'manufacturing_dcf',
                'subsector': subsector,
                'is_cyclical': is_cyclical,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'discount_rate': adjusted_discount_rate,
                'initial_growth_rate': initial_growth_rate,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'forecast_fcf': forecasted_fcf,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'historical_fcf': historical_fcf.to_dict(),
                'net_debt': net_debt,
                'safety_margin': safety_margin,
                'manufacturing_metrics': manufacturing_metrics
            }

            return result

        except Exception as e:
            logger.error(f"Error in manufacturing DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'manufacturing_dcf',
                'subsector': subsector if subsector else 'Unknown',
                'enterprise_value': None,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_manufacturing_metrics(self, ticker: str, financial_data: Dict[str, Any],
                                         subsector: str) -> Dict[str, Any]:
        """
        Calculate manufacturing-specific metrics for valuation adjustments

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Manufacturing sub-sector

        Returns:
            Dictionary with manufacturing-specific metrics
        """
        metrics = {}

        try:
            # Extract data from financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Use most recent financials
            income = income_stmt.iloc[:, 0]
            balance = balance_sheet.iloc[:, 0]

            # 1. Return on Invested Capital (ROIC) - critical for manufacturing
            if 'Operating Income' in income.index:
                operating_income = income['Operating Income']

                # Adjust for taxes
                tax_rate = 0.25  # Approximate tax rate if not available
                if 'Income Tax Expense' in income.index and 'Income Before Tax' in income.index:
                    if income['Income Before Tax'] > 0:
                        tax_rate = income['Income Tax Expense'] / income['Income Before Tax']

                nopat = operating_income * (1 - tax_rate)

                # Calculate invested capital
                invested_capital = 0

                # Total stockholder equity
                if 'Total Stockholder Equity' in balance.index:
                    invested_capital += balance['Total Stockholder Equity']

                # Add long-term debt
                if 'Long Term Debt' in balance.index:
                    invested_capital += balance['Long Term Debt']
                elif 'Total Debt' in balance.index:
                    # Approximate if only total debt is available
                    invested_capital += balance['Total Debt'] * 0.7  # Assume 70% is long-term

                if invested_capital > 0:
                    metrics['roic'] = nopat / invested_capital

            # 2. Fixed Asset Turnover - important for asset-intensive manufacturing
            if 'Total Revenue' in income.index and 'Property Plant and Equipment' in balance.index:
                revenue = income['Total Revenue']
                ppe = balance['Property Plant and Equipment']

                if ppe > 0:
                    metrics['fixed_asset_turnover'] = revenue / ppe

            # 3. Capital Expenditure to Revenue Ratio
            if 'Total Revenue' in income.index and cash_flow is not None:
                revenue = income['Total Revenue']

                if 'Capital Expenditure' in cash_flow.iloc[:, 0].index:
                    capex = abs(cash_flow.iloc[0]['Capital Expenditure'])
                    metrics['capex_to_revenue'] = capex / revenue

            # 4. R&D to Revenue Ratio - important for innovation-driven manufacturing
            if 'Total Revenue' in income.index and 'Research and Development' in income.index:
                revenue = income['Total Revenue']
                r_and_d = income['Research and Development']

                metrics['r_and_d_to_revenue'] = r_and_d / revenue

            # 5. Gross Margin and Operating Margin
            if 'Total Revenue' in income.index:
                revenue = income['Total Revenue']

                if 'Gross Profit' in income.index:
                    gross_profit = income['Gross Profit']
                    metrics['gross_margin'] = gross_profit / revenue
                elif 'Cost of Revenue' in income.index:
                    gross_profit = revenue - income['Cost of Revenue']
                    metrics['gross_margin'] = gross_profit / revenue

                if 'Operating Income' in income.index:
                    operating_income = income['Operating Income']
                    metrics['operating_margin'] = operating_income / revenue

            # 6. Inventory Turnover - important for manufacturing efficiency
            if 'Cost of Revenue' in income.index and 'Inventory' in balance.index:
                cogs = income['Cost of Revenue']
                inventory = balance['Inventory']

                if inventory > 0:
                    metrics['inventory_turnover'] = cogs / inventory
                    # Add days inventory outstanding
                    metrics['days_inventory_outstanding'] = 365 / metrics['inventory_turnover']

            # 7. Working Capital Ratio
            current_assets = 0
            current_liabilities = 0

            if 'Total Current Assets' in balance.index:
                current_assets = balance['Total Current Assets']

            if 'Total Current Liabilities' in balance.index:
                current_liabilities = balance['Total Current Liabilities']

            if current_liabilities > 0:
                metrics['working_capital_ratio'] = current_assets / current_liabilities

            # 8. Capacity Utilization - difficult to get directly, often in company reports
            # For now, use an estimate based on asset turnover trend
            # In a real implementation, this would come from company reports
            if 'fixed_asset_turnover' in metrics:
                # This is a simplification - actual capacity utilization would be better
                fixed_asset_turnover = metrics['fixed_asset_turnover']

                # Roughly estimate capacity utilization
                if subsector == "Heavy_Manufacturing":
                    # Benchmark for heavy manufacturing
                    metrics['capacity_utilization'] = min(0.95, fixed_asset_turnover / 2.5)
                elif subsector == "Automotive":
                    metrics['capacity_utilization'] = min(0.95, fixed_asset_turnover / 3.0)
                elif subsector == "Electronics":
                    metrics['capacity_utilization'] = min(0.95, fixed_asset_turnover / 4.0)
                else:
                    metrics['capacity_utilization'] = min(0.95, fixed_asset_turnover / 3.5)

            # 9. Order Backlog to Revenue - important for certain manufacturing sectors
            # This would typically come from company reports
            # For now, use industry averages as placeholders
            if subsector == "Aerospace_Defense":
                # Aerospace typically has large backlogs
                metrics['backlog_to_revenue'] = 2.5  # 2.5 years of revenue in backlog (placeholder)
            elif subsector == "Heavy_Manufacturing":
                metrics['backlog_to_revenue'] = 0.8  # 0.8 years (placeholder)

            # 10. Pension and retirement obligations (common in older manufacturing firms)
            # Check if mentioned in the balance sheet or notes
            pension_obligation = 0
            for item in ['Pension Obligation', 'Retirement Benefit Obligation', 'Post-Retirement Benefits']:
                if item in balance.index:
                    pension_obligation += balance[item]

            if pension_obligation > 0:
                metrics['pension_obligation'] = pension_obligation

            return metrics

        except Exception as e:
            logger.error(f"Error calculating manufacturing metrics for {ticker}: {e}")
            return metrics

    def _estimate_normalized_growth_rate(self, historical_fcf: pd.Series, income_stmt: pd.DataFrame) -> float:
        """
        Estimate normalized growth rate for cyclical industries by looking at full-cycle performance

        Args:
            historical_fcf: Series with historical free cash flow
            income_stmt: Income statement DataFrame

        Returns:
            Normalized growth rate
        """
        try:
            # For cyclical industries, look at longer timeframes
            if len(historical_fcf) >= 5:
                # Calculate compound annual growth rate from oldest to newest
                start_fcf = historical_fcf.iloc[-1]  # Oldest
                end_fcf = historical_fcf.iloc[0]  # Newest

                if start_fcf > 0:
                    years = len(historical_fcf) - 1
                    cagr = (end_fcf / start_fcf) ** (1 / years) - 1

                    # Cap to reasonable range
                    cagr = max(0.01, min(0.15, cagr))
                    return cagr

            # If FCF data is insufficient, look at revenue trends
            if income_stmt is not None and 'Total Revenue' in income_stmt.index and income_stmt.shape[1] >= 5:
                # Calculate revenue CAGR over 5 years
                start_revenue = income_stmt.iloc[0, -1]  # Oldest
                end_revenue = income_stmt.iloc[0, 0]  # Newest

                if start_revenue > 0:
                    years = income_stmt.shape[1] - 1
                    revenue_cagr = (end_revenue / start_revenue) ** (1 / years) - 1

                    # Adjust downward for FCF (typically grows slower than revenue)
                    fcf_cagr = revenue_cagr * 0.8

                    # Cap to reasonable range
                    fcf_cagr = max(0.01, min(0.15, fcf_cagr))
                    return fcf_cagr

            # Default to conservative growth if insufficient data
            return 0.03  # 3% default normalized growth

        except Exception as e:
            logger.error(f"Error estimating normalized growth rate: {e}")
            return 0.03  # Default to 3% on error

    def manufacturing_replacement_value(self, ticker: str, financial_data: Dict[str, Any] = None,
                                        subsector: str = None) -> Dict[str, Any]:
        """
        Perform replacement value based valuation for manufacturing companies
        (Important for asset-intensive manufacturing companies)

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Performing replacement value valuation for {ticker} as {subsector}")

            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if balance_sheet is None:
                raise ValueError("Missing balance sheet for replacement value valuation")

            # Get the most recent balance sheet data
            balance = balance_sheet.iloc[:, 0]

            # For manufacturing, replacement value focuses on physical assets, specialized equipment,
            # and intangibles like technology and patents

            # 1. Calculate replacement value of physical assets

            # Start with property, plant, and equipment (PPE)
            replacement_ppe = 0
            if 'Property Plant and Equipment' in balance.index:
                book_ppe = balance['Property Plant and Equipment']

                # Apply replacement factor based on subsector
                # Newer equipment/industries typically have lower factor
                # Older industries often need more to replace due to inflation, tech changes
                if subsector == "Electronics":
                    ppe_factor = 1.2  # 20% premium to book value
                elif subsector in ["Heavy_Manufacturing", "Chemical_Materials"]:
                    ppe_factor = 1.5  # 50% premium to book value
                else:
                    ppe_factor = 1.3  # 30% premium

                replacement_ppe = book_ppe * ppe_factor

            # 2. Replacement value of inventory
            replacement_inventory = 0
            if 'Inventory' in balance.index:
                book_inventory = balance['Inventory']

                # Apply replacement factor based on subsector
                if subsector == "Electronics":
                    # Electronic components may need to be replaced with newer versions
                    inventory_factor = 1.1
                elif subsector == "Automotive":
                    inventory_factor = 1.15
                else:
                    inventory_factor = 1.05

                replacement_inventory = book_inventory * inventory_factor

            # 3. Intangible assets replacement (R&D, patents, technology)
            replacement_intangibles = 0

            # Start with book intangibles but adjust significantly
            if 'Intangible Assets' in balance.index:
                book_intangibles = balance['Intangible Assets']

                # For R&D intensive sectors, intangibles often undervalued
                if subsector in ["Electronics", "Aerospace_Defense"]:
                    intangibles_factor = 2.0  # 100% premium
                elif subsector == "Automotive":
                    intangibles_factor = 1.5  # 50% premium
                else:
                    intangibles_factor = 1.3  # 30% premium

                replacement_intangibles = book_intangibles * intangibles_factor

            # 4. Additional intangible value from R&D history
            # This captures value of accumulated knowledge and tech not on balance sheet
            r_and_d_replacement = 0

            if income_stmt is not None and 'Research and Development' in income_stmt.index:
                # Look at multiple years of R&D if available
                r_and_d_years = min(income_stmt.shape[1], 5)  # Up to 5 years

                # Sum R&D with depreciation factor
                # More recent R&D is more valuable (less obsolete)
                for i in range(r_and_d_years):
                    r_and_d = income_stmt.iloc[:, i]['Research and Development']
                    year_factor = 0.7 ** i  # Roughly 30% annual depreciation
                    r_and_d_replacement += r_and_d * year_factor

            # 5. Other assets at book value (cash, receivables, etc.)
            other_assets = 0

            if 'Cash and Cash Equivalents' in balance.index:
                other_assets += balance['Cash and Cash Equivalents']

            if 'Net Receivables' in balance.index:
                other_assets += balance['Net Receivables']

            # Additional items can be added

            # 6. Sum all replacement values
            total_asset_replacement = replacement_ppe + replacement_inventory + replacement_intangibles + r_and_d_replacement + other_assets

            # 7. Subtract liabilities
            total_liabilities = 0

            if 'Total Liabilities' in balance.index:
                total_liabilities = balance['Total Liabilities']
            else:
                # Estimate from components
                if 'Long Term Debt' in balance.index:
                    total_liabilities += balance['Long Term Debt']
                if 'Short Term Debt' in balance.index:
                    total_liabilities += balance['Short Term Debt']
                if 'Accounts Payable' in balance.index:
                    total_liabilities += balance['Accounts Payable']
                # Add other significant liabilities
                for item in ['Pension Obligation', 'Deferred Long Term Liabilities']:
                    if item in balance.index:
                        total_liabilities += balance[item]

            # Calculate net replacement value
            net_replacement_value = total_asset_replacement - total_liabilities

            # Apply an economic obsolescence factor
            # This accounts for whether the business's current performance
            # justifies rebuilding the assets at full replacement cost

            economic_obsolescence = 0

            if income_stmt is not None and 'Operating Income' in income_stmt.iloc[:, 0].index:
                operating_income = income_stmt.iloc[0]['Operating Income']

                # Calculate normalized return on replacement
                if total_asset_replacement > 0:
                    return_on_replacement = operating_income / total_asset_replacement

                    # Compare to industry required return
                    if subsector in self.manufacturing_subsector_parameters:
                        required_return = self.manufacturing_subsector_parameters[subsector]['roic_target']
                    else:
                        required_return = 0.12  # Default

                    # If actual return < required, assets are economically obsolete
                    if return_on_replacement < required_return:
                        obsolescence_factor = max(0.3, return_on_replacement / required_return)
                        economic_obsolescence = total_asset_replacement * (1 - obsolescence_factor)

            # Final adjusted replacement value
            adjusted_replacement_value = net_replacement_value - economic_obsolescence

            # Calculate per share value
            if shares_outstanding is not None and shares_outstanding > 0:
                value_per_share = adjusted_replacement_value / shares_outstanding
            else:
                # If shares outstanding not available, estimate from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    value_per_share = adjusted_replacement_value / estimated_shares
                else:
                    value_per_share = None

            # Apply margin of safety
            safety_margin = 0.25  # 25% for replacement value approach
            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            # Compile results with detailed breakdown
            result = {
                'company': ticker,
                'method': 'manufacturing_replacement_value',
                'subsector': subsector,
                'replacement_values': {
                    'ppe': replacement_ppe,
                    'inventory': replacement_inventory,
                    'intangibles': replacement_intangibles,
                    'r_and_d_capitalized': r_and_d_replacement,
                    'other_assets': other_assets
                },
                'total_asset_replacement': total_asset_replacement,
                'total_liabilities': total_liabilities,
                'net_replacement_value': net_replacement_value,
                'economic_obsolescence': economic_obsolescence,
                'adjusted_replacement_value': adjusted_replacement_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'safety_margin': safety_margin
            }

            return result

        except Exception as e:
            logger.error(f"Error in manufacturing replacement value for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'manufacturing_replacement_value',
                'subsector': subsector if subsector else 'Unknown',
                'adjusted_replacement_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def manufacturing_relative_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                         subsector: str = None) -> Dict[str, Any]:
        """
        Perform relative valuation for manufacturing companies using sector-specific multiples

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Performing manufacturing relative valuation for {ticker} as {subsector}")

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for relative valuation")

            # Get most recent data
            income = income_stmt.iloc[:, 0]
            balance = balance_sheet.iloc[:, 0]

            # Calculate manufacturing metrics (for adjustments to multiples)
            manufacturing_metrics = self._calculate_manufacturing_metrics(ticker, financial_data, subsector)

            # Define multiple ranges by subsector
            subsector_multiples = {
                "Heavy_Manufacturing": {
                    "EV/EBITDA": 7.5,
                    "EV/EBIT": 10.0,
                    "P/E": 14.0,
                    "P/B": 1.8,
                    "EV/Sales": 1.2
                },
                "Automotive": {
                    "EV/EBITDA": 6.0,
                    "EV/EBIT": 8.0,
                    "P/E": 10.0,
                    "P/B": 1.5,
                    "EV/Sales": 0.8
                },
                "Aerospace_Defense": {
                    "EV/EBITDA": 9.0,
                    "EV/EBIT": 12.0,
                    "P/E": 16.0,
                    "P/B": 2.5,
                    "EV/Sales": 1.5
                },
                "Electronics": {
                    "EV/EBITDA": 11.0,
                    "EV/EBIT": 15.0,
                    "P/E": 20.0,
                    "P/B": 3.0,
                    "EV/Sales": 2.0
                },
                "Consumer_Goods": {
                    "EV/EBITDA": 9.0,
                    "EV/EBIT": 12.0,
                    "P/E": 18.0,
                    "P/B": 2.5,
                    "EV/Sales": 1.5
                },
                "Chemical_Materials": {
                    "EV/EBITDA": 7.0,
                    "EV/EBIT": 9.0,
                    "P/E": 12.0,
                    "P/B": 1.6,
                    "EV/Sales": 1.0
                }
            }

            # Get multiples for the identified subsector
            multiples = subsector_multiples.get(subsector, subsector_multiples["Heavy_Manufacturing"])

            # Adjust multiples based on company-specific metrics
            adjusted_multiples = multiples.copy()

            # Adjust based on ROIC
            if 'roic' in manufacturing_metrics:
                roic = manufacturing_metrics['roic']
                # Compare to industry target
                target_roic = self.manufacturing_subsector_parameters[subsector]['roic_target']

                roic_adjustment = (roic - target_roic) / target_roic
                # Cap the adjustment
                roic_impact = min(max(roic_adjustment * 0.5, -0.2), 0.3)  # -20% to +30%

                # Apply to EV/EBITDA and P/E which most affected by profitability
                adjusted_multiples["EV/EBITDA"] *= (1 + roic_impact)
                adjusted_multiples["P/E"] *= (1 + roic_impact)

            # Adjust based on operating margin
            if 'operating_margin' in manufacturing_metrics:
                operating_margin = manufacturing_metrics['operating_margin']

                # Benchmarks vary by subsector
                if subsector == "Electronics":
                    benchmark = 0.15  # 15% for Electronics
                elif subsector == "Chemical_Materials":
                    benchmark = 0.12  # 12% for Chemicals
                else:
                    benchmark = 0.10  # 10% for others

                margin_adjustment = (operating_margin - benchmark) / benchmark
                margin_impact = min(max(margin_adjustment * 0.4, -0.15), 0.25)  # Cap at -15% to +25%

                # Apply to multiples
                adjusted_multiples["EV/EBITDA"] *= (1 + margin_impact)
                adjusted_multiples["EV/EBIT"] *= (1 + margin_impact)

            # Adjust for R&D investment (higher is better)
            if 'r_and_d_to_revenue' in manufacturing_metrics and subsector in ["Electronics", "Aerospace_Defense",
                                                                               "Automotive"]:
                r_and_d_ratio = manufacturing_metrics['r_and_d_to_revenue']

                # Benchmark depends on sector
                if subsector == "Electronics":
                    benchmark = 0.08  # 8% for Electronics
                elif subsector == "Aerospace_Defense":
                    benchmark = 0.06  # 6% for Aerospace
                else:
                    benchmark = 0.04  # 4% for Automotive

                rd_adjustment = (r_and_d_ratio - benchmark) / benchmark
                rd_impact = min(max(rd_adjustment * 0.3, -0.1), 0.2)  # Cap at -10% to +20%

                # Apply to multiples (primarily affects long-term valuation)
                adjusted_multiples["P/E"] *= (1 + rd_impact)

            # Calculate key metrics for valuation
            metrics = {}

            # Extract key financials
            if 'Total Revenue' in income.index:
                metrics['revenue'] = income.loc['Total Revenue']

            if 'Net Income' in income.index:
                metrics['earnings'] = income.loc['Net Income']

            if 'EBITDA' in income.index:
                metrics['ebitda'] = income.loc['EBITDA']
            elif 'Operating Income' in income.index and 'Depreciation & Amortization' in income.index:
                # Calculate EBITDA if not directly available
                metrics['ebitda'] = income.loc['Operating Income'] + income.loc['Depreciation & Amortization']

            if 'Operating Income' in income.index:
                metrics['ebit'] = income.loc['Operating Income']

            if 'Total Stockholder Equity' in balance.index:
                metrics['book_value'] = balance.loc['Total Stockholder Equity']

            # Calculate enterprise value
            market_cap = market_data.get('market_cap', 0)

            if 'Total Debt' in balance.index:
                total_debt = balance.loc['Total Debt']
            else:
                # Estimate from long-term and short-term debt
                total_debt = 0
                if 'Long Term Debt' in balance.index:
                    total_debt += balance.loc['Long Term Debt']
                if 'Short Term Debt' in balance.index:
                    total_debt += balance.loc['Short Term Debt']

            # Cash and equivalents
            cash = 0
            if 'Cash and Cash Equivalents' in balance.index:
                cash = balance.loc['Cash and Cash Equivalents']
            elif 'Cash and Short Term Investments' in balance.index:
                cash = balance.loc['Cash and Short Term Investments']

            # Calculate enterprise value
            metrics['enterprise_value'] = market_cap + total_debt - cash

            # Calculate estimated values based on different multiples
            valuations = {}

            # EV/EBITDA valuation
            if 'ebitda' in metrics and metrics['ebitda'] > 0:
                ev_ebitda_multiple = adjusted_multiples.get("EV/EBITDA")
                ev_from_ebitda = metrics['ebitda'] * ev_ebitda_multiple
                equity_value_from_ebitda = ev_from_ebitda - total_debt + cash

                valuations['ev_ebitda'] = {
                    'multiple': ev_ebitda_multiple,
                    'enterprise_value': ev_from_ebitda,
                    'equity_value': equity_value_from_ebitda,
                    'description': 'Enterprise Value to EBITDA'
                }

            # EV/EBIT valuation
            if 'ebit' in metrics and metrics['ebit'] > 0:
                ev_ebit_multiple = adjusted_multiples.get("EV/EBIT")
                ev_from_ebit = metrics['ebit'] * ev_ebit_multiple
                equity_value_from_ebit = ev_from_ebit - total_debt + cash

                valuations['ev_ebit'] = {
                    'multiple': ev_ebit_multiple,
                    'enterprise_value': ev_from_ebit,
                    'equity_value': equity_value_from_ebit,
                    'description': 'Enterprise Value to EBIT'
                }

            # EV/Sales valuation
            if 'revenue' in metrics and metrics['revenue'] > 0:
                ev_sales_multiple = adjusted_multiples.get("EV/Sales")
                ev_from_sales = metrics['revenue'] * ev_sales_multiple
                equity_value_from_sales = ev_from_sales - total_debt + cash

                valuations['ev_sales'] = {
                    'multiple': ev_sales_multiple,
                    'enterprise_value': ev_from_sales,
                    'equity_value': equity_value_from_sales,
                    'description': 'Enterprise Value to Sales'
                }

            # P/E valuation
            if 'earnings' in metrics and metrics['earnings'] > 0:
                pe_multiple = adjusted_multiples.get("P/E")
                equity_value_from_earnings = metrics['earnings'] * pe_multiple

                valuations['pe'] = {
                    'multiple': pe_multiple,
                    'equity_value': equity_value_from_earnings,
                    'description': 'Price to Earnings'
                }

            # P/B valuation
            if 'book_value' in metrics and metrics['book_value'] > 0:
                pb_multiple = adjusted_multiples.get("P/B")
                equity_value_from_book = metrics['book_value'] * pb_multiple

                valuations['pb'] = {
                    'multiple': pb_multiple,
                    'equity_value': equity_value_from_book,
                    'description': 'Price to Book'
                }

            # Calculate average equity value
            equity_values = []

            # For cyclical industries, weight EV/EBITDA and P/B more heavily
            is_cyclical = subsector in self.cyclical_subsectors

            if is_cyclical:
                # Weight different methods
                weights = {
                    'ev_ebitda': 0.35,
                    'ev_ebit': 0.15,
                    'ev_sales': 0.15,
                    'pe': 0.15,
                    'pb': 0.20
                }

                weighted_sum = 0
                total_weight = 0

                for method, valuation in valuations.items():
                    if 'equity_value' in valuation:
                        weight = weights.get(method, 0.20)  # Default weight
                        weighted_sum += valuation['equity_value'] * weight
                        total_weight += weight

                if total_weight > 0:
                    avg_equity_value = weighted_sum / total_weight
                else:
                    # Simple average as fallback
                    for method, valuation in valuations.items():
                        if 'equity_value' in valuation:
                            equity_values.append(valuation['equity_value'])

                    avg_equity_value = sum(equity_values) / len(equity_values) if equity_values else None
            else:
                # Simple average for non-cyclical
                for method, valuation in valuations.items():
                    if 'equity_value' in valuation:
                        equity_values.append(valuation['equity_value'])

                avg_equity_value = sum(equity_values) / len(equity_values) if equity_values else None

            # Calculate per share values
            shares_outstanding = market_data.get('shares_outstanding')
            per_share_values = {}

            if shares_outstanding and shares_outstanding > 0:
                for method, valuation in valuations.items():
                    if 'equity_value' in valuation:
                        per_share_values[method] = valuation['equity_value'] / shares_outstanding

                if avg_equity_value:
                    value_per_share = avg_equity_value / shares_outstanding
                else:
                    value_per_share = None
            else:
                value_per_share = None

            # Apply margin of safety (higher for cyclical industries)
            safety_margin = 0.25  # Default 25%
            if is_cyclical:
                safety_margin = 0.30  # 30% for cyclical industries

            if value_per_share:
                conservative_value = value_per_share * (1 - safety_margin)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'manufacturing_relative_valuation',
                'subsector': subsector,
                'is_cyclical': is_cyclical,
                'valuations': valuations,
                'average_equity_value': avg_equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'per_share_values': per_share_values,
                'metrics': metrics,
                'multiples_used': adjusted_multiples,
                'manufacturing_metrics': manufacturing_metrics,
                'safety_margin': safety_margin
            }

        except Exception as e:
            logger.error(f"Error in manufacturing relative valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'manufacturing_relative_valuation',
                'subsector': subsector if subsector else 'Unknown',
                'average_equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def manufacturing_earnings_power_value(self, ticker: str, financial_data: Dict[str, Any] = None,
                                           subsector: str = None) -> Dict[str, Any]:
        """
        Perform valuation based on Earnings Power Value (EPV) methodology
        Often used for manufacturing companies with stable operations

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Performing earnings power valuation for {ticker} as {subsector}")

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for earnings power valuation")

            # Get manufacturing metrics
            manufacturing_metrics = self._calculate_manufacturing_metrics(ticker, financial_data, subsector)

            # Is this a cyclical industry?
            is_cyclical = subsector in self.cyclical_subsectors

            # 1. Calculate normalized earnings (looking through cycles for cyclical companies)
            normalized_earnings = self._calculate_normalized_earnings(income_stmt, is_cyclical)

            # 2. Adjust for one-time items and normalization

            # Start with operating income as base
            if 'Operating Income' in income_stmt.iloc[:, 0].index:
                latest_operating_income = income_stmt.iloc[0]['Operating Income']
            else:
                raise ValueError("Operating Income not found in income statement")

            # Adjustments to normalize earnings
            adjustments = {}

            # For cyclical industries, normalize to mid-cycle earnings
            if is_cyclical:
                # Adjust to normalized level
                cycle_adjustment = normalized_earnings - latest_operating_income
                adjustments['cyclical_normalization'] = cycle_adjustment

            # Adjust for R&D (considered investment rather than expense)
            if 'r_and_d_to_revenue' in manufacturing_metrics and 'Research and Development' in income_stmt.iloc[:,
                                                                                               0].index:
                r_and_d = income_stmt.iloc[0]['Research and Development']

                # Capitalize portion of R&D (varies by sector)
                if subsector in ["Electronics", "Aerospace_Defense"]:
                    r_and_d_factor = 0.7  # 70% considered investment
                else:
                    r_and_d_factor = 0.5  # 50% for others

                r_and_d_adjustment = r_and_d * r_and_d_factor
                adjustments['r_and_d_capitalization'] = r_and_d_adjustment

            # Normalize depreciation if abnormally high or low
            if 'Depreciation & Amortization' in income_stmt.iloc[:,
                                                0].index and 'Property Plant and Equipment' in balance_sheet.iloc[:,
                                                                                               0].index:
                depreciation = income_stmt.iloc[0]['Depreciation & Amortization']
                ppe = balance_sheet.iloc[0]['Property Plant and Equipment']

                if ppe > 0:
                    depreciation_rate = depreciation / ppe

                    # Check if depreciation rate is abnormal
                    normal_rate = 0.08  # 8% typical for manufacturing

                    if abs(depreciation_rate - normal_rate) > 0.03:  # More than 3% difference
                        normalized_depreciation = ppe * normal_rate
                        depreciation_adjustment = normalized_depreciation - depreciation
                        adjustments['depreciation_normalization'] = depreciation_adjustment

            # Apply all adjustments to get normalized operating income
            adjusted_operating_income = latest_operating_income
            for adjustment, value in adjustments.items():
                adjusted_operating_income += value

            # 3. Apply tax rate to get normalized after-tax operating income

            # Estimate effective tax rate
            effective_tax_rate = 0.25  # Default corporate tax rate

            if 'Income Tax Expense' in income_stmt.iloc[:, 0].index and 'Income Before Tax' in income_stmt.iloc[:,
                                                                                               0].index:
                income_before_tax = income_stmt.iloc[0]['Income Before Tax']
                tax_expense = income_stmt.iloc[0]['Income Tax Expense']

                if income_before_tax > 0:
                    effective_tax_rate = min(0.35, max(0.15, tax_expense / income_before_tax))

            # Calculate normalized after-tax operating income
            normalized_after_tax_income = adjusted_operating_income * (1 - effective_tax_rate)

            # 4. Determine appropriate capitalization rate (inverse of multiple)

            # Base capitalization rate by subsector
            if subsector in self.manufacturing_subsector_parameters:
                base_cap_rate = self.manufacturing_subsector_parameters[subsector]['roic_target']
            else:
                base_cap_rate = 0.12  # Default 12%

            # Adjust for company-specific factors
            cap_rate_adjustments = 0

            # Adjustment for margin stability
            if len(income_stmt.columns) >= 3:
                # Calculate margin stability over time
                margins = []
                for i in range(min(3, len(income_stmt.columns))):
                    if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                        if income_stmt.iloc[0, i]['Total Revenue'] > 0:
                            margin = income_stmt.iloc[0, i]['Operating Income'] / income_stmt.iloc[0, i][
                                'Total Revenue']
                            margins.append(margin)

                if len(margins) >= 2:
                    # Calculate margin volatility
                    import numpy as np
                    margin_std = np.std(margins)
                    average_margin = np.mean(margins)

                    if average_margin > 0:
                        volatility = margin_std / average_margin

                        # Adjust cap rate based on volatility
                        if volatility > 0.3:  # High volatility
                            cap_rate_adjustments += 0.02  # 200 basis points premium
                        elif volatility < 0.1:  # Low volatility
                            cap_rate_adjustments -= 0.01  # 100 basis points discount

            # Adjustment for debt level (higher debt = higher risk)
            if 'Total Debt' in balance_sheet.iloc[:, 0].index and 'Total Assets' in balance_sheet.iloc[:, 0].index:
                debt = balance_sheet.iloc[0]['Total Debt']
                assets = balance_sheet.iloc[0]['Total Assets']

                if assets > 0:
                    debt_to_assets = debt / assets

                    # Adjust cap rate based on leverage
                    if debt_to_assets > 0.5:  # Highly leveraged
                        cap_rate_adjustments += 0.02
                    elif debt_to_assets < 0.2:  # Low leverage
                        cap_rate_adjustments -= 0.01

            # Final capitalization rate
            capitalization_rate = base_cap_rate + cap_rate_adjustments

            # 5. Calculate Earnings Power Value (EPV)
            epv = normalized_after_tax_income / capitalization_rate

            # 6. Adjust for excess or deficit assets/liabilities

            # Excess cash (cash beyond operating needs)
            excess_cash = 0
            if 'Cash and Cash Equivalents' in balance_sheet.iloc[:, 0].index and 'Total Revenue' in income_stmt.iloc[:,
                                                                                                    0].index:
                cash = balance_sheet.iloc[0]['Cash and Cash Equivalents']
                revenue = income_stmt.iloc[0]['Total Revenue']

                # Typical operating cash is 2-5% of revenue for manufacturing
                operating_cash_needs = revenue * 0.05

                if cash > operating_cash_needs:
                    excess_cash = cash - operating_cash_needs

            # Debt adjustment
            total_debt = 0
            if 'Total Debt' in balance_sheet.iloc[:, 0].index:
                total_debt = balance_sheet.iloc[0]['Total Debt']
            else:
                # Sum components
                if 'Long Term Debt' in balance_sheet.iloc[:, 0].index:
                    total_debt += balance_sheet.iloc[0]['Long Term Debt']
                if 'Short Term Debt' in balance_sheet.iloc[:, 0].index:
                    total_debt += balance_sheet.iloc[0]['Short Term Debt']

            # Non-operating assets
            non_operating_assets = 0
            # This would include investments, land held for sale, etc.
            # In a full implementation, we would identify these from financial statements

            # 7. Final EPV calculation
            adjusted_epv = epv + excess_cash + non_operating_assets - total_debt

            # Calculate per share value
            if shares_outstanding is not None and shares_outstanding > 0:
                value_per_share = adjusted_epv / shares_outstanding
            else:
                # If shares outstanding not available, estimate from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    value_per_share = adjusted_epv / estimated_shares
                else:
                    value_per_share = None

            # Apply margin of safety
            safety_margin = 0.25  # 25% for EPV approach
            if is_cyclical:
                safety_margin = 0.30  # 30% for cyclical industries

            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            return {
                'company': ticker,
                'method': 'manufacturing_earnings_power_value',
                'subsector': subsector,
                'is_cyclical': is_cyclical,
                'normalized_operating_income': adjusted_operating_income,
                'normalized_after_tax_income': normalized_after_tax_income,
                'adjustments': adjustments,
                'capitalization_rate': capitalization_rate,
                'epv': epv,
                'excess_cash': excess_cash,
                'total_debt': total_debt,
                'adjusted_epv': adjusted_epv,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'safety_margin': safety_margin
            }

        except Exception as e:
            logger.error(f"Error in manufacturing earnings power valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'manufacturing_earnings_power_value',
                'subsector': subsector if subsector else 'Unknown',
                'epv': None,
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_normalized_earnings(self, income_stmt: pd.DataFrame, is_cyclical: bool) -> float:
        """
        Calculate normalized earnings for a manufacturing company,
        smoothing out cyclical effects for cyclical industries

        Args:
            income_stmt: Income statement DataFrame
            is_cyclical: Flag indicating if the company is in a cyclical industry

        Returns:
            Normalized operating income
        """
        try:
            # If insufficient data, return latest operating income
            if income_stmt is None or income_stmt.empty or 'Operating Income' not in income_stmt.index:
                raise ValueError("Income statement data insufficient for normalization")

            # For non-cyclical companies or limited data, use current operating income
            if not is_cyclical or income_stmt.shape[1] < 3:
                return income_stmt.iloc[0]['Operating Income']

            # For cyclical industries, average over multiple years to smooth the cycle
            # Use up to 5 years (or as many as available)
            years_available = min(5, income_stmt.shape[1])
            operating_incomes = []

            for i in range(years_available):
                if 'Operating Income' in income_stmt.iloc[:, i].index:
                    operating_incomes.append(income_stmt.iloc[0, i]['Operating Income'])

            if not operating_incomes:
                raise ValueError("Operating Income not found in income statement")

            # Calculate weighted average, giving more weight to recent years
            # but still accounting for cyclicality
            weights = [0.35, 0.25, 0.20, 0.15, 0.05][:len(operating_incomes)]

            # Normalize weights to sum to 1
            weights = [w / sum(weights) for w in weights]

            # Calculate weighted average
            normalized_income = sum(inc * w for inc, w in zip(operating_incomes, weights))

            return normalized_income

        except Exception as e:
            logger.error(f"Error calculating normalized earnings: {e}")
            # If normalization fails, return the latest operating income if available
            if income_stmt is not None and not income_stmt.empty and 'Operating Income' in income_stmt.iloc[:, 0].index:
                return income_stmt.iloc[0]['Operating Income']
            else:
                return 0

    def apply_manufacturing_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                      subsector: str = None) -> Dict[str, Any]:
        """
        Apply the most appropriate manufacturing valuation methods based on sub-sector and available data

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with comprehensive valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Applying comprehensive manufacturing valuation for {ticker} as {subsector}")

            # Determine if company is in a cyclical industry
            is_cyclical = subsector in self.cyclical_subsectors

            # Determine if company is asset-intensive
            is_asset_intensive = subsector in self.asset_intensive_subsectors

            # Apply different valuation methods
            dcf_result = self.manufacturing_dcf_valuation(ticker, financial_data, subsector)
            relative_result = self.manufacturing_relative_valuation(ticker, financial_data, subsector)

            # For asset-intensive industries, add replacement value approach
            replacement_result = None
            if is_asset_intensive:
                replacement_result = self.manufacturing_replacement_value(ticker, financial_data, subsector)

            # For stable companies, add earnings power value approach
            epv_result = None
            if not is_cyclical:  # More useful for non-cyclical companies
                epv_result = self.manufacturing_earnings_power_value(ticker, financial_data, subsector)

            # Determine weights for different methods based on sub-sector
            weights = {}

            if is_asset_intensive and is_cyclical:
                # For cyclical, asset-intensive industries like Heavy Manufacturing
                weights = {
                    'dcf': 0.25,
                    'relative': 0.35,
                    'replacement': 0.40
                }
            elif is_asset_intensive and not is_cyclical:
                # For non-cyclical, asset-intensive industries
                weights = {
                    'dcf': 0.30,
                    'relative': 0.25,
                    'replacement': 0.30,
                    'epv': 0.15
                }
            elif is_cyclical and not is_asset_intensive:
                # For cyclical but not asset-intensive (like Automotive suppliers)
                weights = {
                    'dcf': 0.30,
                    'relative': 0.50,
                    'epv': 0.20
                }
            elif subsector in ["Electronics", "Aerospace_Defense"]:
                # For technology-driven manufacturing
                weights = {
                    'dcf': 0.45,
                    'relative': 0.35,
                    'epv': 0.20
                }
            else:
                # Default weights
                weights = {
                    'dcf': 0.40,
                    'relative': 0.40,
                    'epv': 0.20
                }

            # Collect per-share values from different methods
            values_per_share = {}

            if dcf_result and 'value_per_share' in dcf_result and dcf_result['value_per_share']:
                values_per_share['dcf'] = dcf_result['value_per_share']

            if relative_result and 'value_per_share' in relative_result and relative_result['value_per_share']:
                values_per_share['relative'] = relative_result['value_per_share']

            if replacement_result and 'value_per_share' in replacement_result and replacement_result['value_per_share']:
                values_per_share['replacement'] = replacement_result['value_per_share']

            if epv_result and 'value_per_share' in epv_result and epv_result['value_per_share']:
                values_per_share['epv'] = epv_result['value_per_share']

            # Calculate weighted average value per share
            weighted_value = 0
            total_weight = 0

            for method, weight in weights.items():
                if method in values_per_share:
                    weighted_value += values_per_share[method] * weight
                    total_weight += weight

            if total_weight > 0:
                final_value_per_share = weighted_value / total_weight
            else:
                final_value_per_share = None

            # Apply margin of safety (higher for cyclical industries)
            safety_margin = 0.25  # Base margin of safety
            if is_cyclical:
                safety_margin = 0.30  # Higher for cyclical industries

            conservative_value = final_value_per_share * (1 - safety_margin) if final_value_per_share else None

            # Compile all results
            return {
                'company': ticker,
                'method': 'comprehensive_manufacturing_valuation',
                'subsector': subsector,
                'is_cyclical': is_cyclical,
                'is_asset_intensive': is_asset_intensive,
                'value_per_share': final_value_per_share,
                'conservative_value': conservative_value,
                'values_by_method': values_per_share,
                'weights': weights,
                'methods': {
                    'dcf': dcf_result,
                    'relative': relative_result,
                    'replacement': replacement_result,
                    'epv': epv_result
                },
                'safety_margin': safety_margin
            }

        except Exception as e:
            logger.error(f"Error in comprehensive manufacturing valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'comprehensive_manufacturing_valuation',
                'subsector': subsector if subsector else 'Unknown',
                'value_per_share': None,
                'error': str(e)
            }

    def get_manufacturing_specific_ratios(self, ticker: str, financial_data: Dict[str, Any] = None,
                                          subsector: str = None) -> Dict[str, Any]:
        """
        Calculate manufacturing-specific financial ratios important for analyzing manufacturing companies

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Manufacturing sub-sector (will be detected if None)

        Returns:
            Dictionary with manufacturing-specific ratios
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_manufacturing_subsector(ticker, financial_data)

            logger.info(f"Calculating manufacturing-specific ratios for {ticker} as {subsector}")

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get manufacturing metrics (many are already ratios)
            manufacturing_metrics = self._calculate_manufacturing_metrics(ticker, financial_data, subsector)

            # Initialize ratios dictionary
            ratios = {
                'profitability': {},
                'efficiency': {},
                'liquidity': {},
                'leverage': {},
                'operating': {},
                'valuation': {}
            }

            # 1. Profitability Ratios (manufacturing-specific)

            # Copy from metrics if already calculated
            for key in ['gross_margin', 'operating_margin', 'roic']:
                if key in manufacturing_metrics:
                    ratios['profitability'][key] = manufacturing_metrics[key]

            # Add EBITDA margin if not present
            if income_stmt is not None and 'Total Revenue' in income_stmt.iloc[:, 0].index:
                revenue = income_stmt.iloc[0]['Total Revenue']

                if 'EBITDA' in income_stmt.iloc[:, 0].index and revenue > 0:
                    ebitda = income_stmt.iloc[0]['EBITDA']
                    ratios['profitability']['ebitda_margin'] = ebitda / revenue
                elif 'Operating Income' in income_stmt.iloc[:,
                                           0].index and 'Depreciation & Amortization' in income_stmt.iloc[:,
                                                                                         0].index and revenue > 0:
                    operating_income = income_stmt.iloc[0]['Operating Income']
                    depreciation = income_stmt.iloc[0]['Depreciation & Amortization']
                    ebitda = operating_income + depreciation
                    ratios['profitability']['ebitda_margin'] = ebitda / revenue

            # 2. Efficiency Ratios (critical for manufacturing)

            # Copy from metrics if already calculated
            for key in ['fixed_asset_turnover', 'inventory_turnover', 'days_inventory_outstanding']:
                if key in manufacturing_metrics:
                    ratios['efficiency'][key] = manufacturing_metrics[key]

            # Add total asset turnover if not present
            if income_stmt is not None and balance_sheet is not None:
                if 'Total Revenue' in income_stmt.iloc[:, 0].index and 'Total Assets' in balance_sheet.iloc[:, 0].index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    assets = balance_sheet.iloc[0]['Total Assets']

                    if assets > 0:
                        ratios['efficiency']['asset_turnover'] = revenue / assets

            # Calculate Cash Conversion Cycle
            if 'days_inventory_outstanding' in ratios['efficiency']:
                dio = ratios['efficiency']['days_inventory_outstanding']

                # Days Sales Outstanding (if we can calculate it)
                if balance_sheet is not None and income_stmt is not None:
                    if 'Net Receivables' in balance_sheet.iloc[:, 0].index and 'Total Revenue' in income_stmt.iloc[:,
                                                                                                  0].index:
                        receivables = balance_sheet.iloc[0]['Net Receivables']
                        revenue = income_stmt.iloc[0]['Total Revenue']

                        if revenue > 0:
                            dso = (receivables / revenue) * 365
                            ratios['efficiency']['days_sales_outstanding'] = dso

                    # Days Payable Outstanding
                    if 'Accounts Payable' in balance_sheet.iloc[:, 0].index and 'Cost of Revenue' in income_stmt.iloc[:,
                                                                                                     0].index:
                        payables = balance_sheet.iloc[0]['Accounts Payable']
                        cogs = income_stmt.iloc[0]['Cost of Revenue']

                        if cogs > 0:
                            dpo = (payables / cogs) * 365
                            ratios['efficiency']['days_payable_outstanding'] = dpo

                # Calculate Cash Conversion Cycle if we have the components
                if 'days_sales_outstanding' in ratios['efficiency'] and 'days_payable_outstanding' in ratios[
                    'efficiency']:
                    dio = ratios['efficiency']['days_inventory_outstanding']
                    dso = ratios['efficiency']['days_sales_outstanding']
                    dpo = ratios['efficiency']['days_payable_outstanding']

                    ccc = dio + dso - dpo
                    ratios['efficiency']['cash_conversion_cycle'] = ccc

            # 3. Liquidity Ratios

            if balance_sheet is not None:
                # Current Ratio
                if 'Total Current Assets' in balance_sheet.iloc[:,
                                             0].index and 'Total Current Liabilities' in balance_sheet.iloc[:, 0].index:
                    current_assets = balance_sheet.iloc[0]['Total Current Assets']
                    current_liabilities = balance_sheet.iloc[0]['Total Current Liabilities']

                    if current_liabilities > 0:
                        ratios['liquidity']['current_ratio'] = current_assets / current_liabilities

                # Quick Ratio (more stringent for manufacturing due to inventory)
                if 'Total Current Assets' in balance_sheet.iloc[:, 0].index and 'Inventory' in balance_sheet.iloc[:,
                                                                                               0].index and 'Total Current Liabilities' in balance_sheet.iloc[
                                                                                                                                           :,
                                                                                                                                           0].index:
                    current_assets = balance_sheet.iloc[0]['Total Current Assets']
                    inventory = balance_sheet.iloc[0]['Inventory']
                    current_liabilities = balance_sheet.iloc[0]['Total Current Liabilities']

                    if current_liabilities > 0:
                        ratios['liquidity']['quick_ratio'] = (current_assets - inventory) / current_liabilities

            # 4. Leverage Ratios (important for asset-intensive manufacturing)

            if balance_sheet is not None:
                # Debt to Equity
                if 'Total Debt' in balance_sheet.iloc[:, 0].index and 'Total Stockholder Equity' in balance_sheet.iloc[
                                                                                                    :, 0].index:
                    debt = balance_sheet.iloc[0]['Total Debt']
                    equity = balance_sheet.iloc[0]['Total Stockholder Equity']

                    if equity > 0:
                        ratios['leverage']['debt_to_equity'] = debt / equity

                # Debt to EBITDA
                if 'Total Debt' in balance_sheet.iloc[:, 0].index and income_stmt is not None:
                    debt = balance_sheet.iloc[0]['Total Debt']

                    if 'EBITDA' in income_stmt.iloc[:, 0].index:
                        ebitda = income_stmt.iloc[0]['EBITDA']

                        if ebitda > 0:
                            ratios['leverage']['debt_to_ebitda'] = debt / ebitda
                    elif 'Operating Income' in income_stmt.iloc[:,
                                               0].index and 'Depreciation & Amortization' in income_stmt.iloc[:,
                                                                                             0].index:
                        operating_income = income_stmt.iloc[0]['Operating Income']
                        depreciation = income_stmt.iloc[0]['Depreciation & Amortization']
                        ebitda = operating_income + depreciation

                        if ebitda > 0:
                            ratios['leverage']['debt_to_ebitda'] = debt / ebitda

                # Interest Coverage Ratio
                if income_stmt is not None and 'Operating Income' in income_stmt.iloc[:,
                                                                     0].index and 'Interest Expense' in income_stmt.iloc[
                                                                                                        :, 0].index:
                    operating_income = income_stmt.iloc[0]['Operating Income']
                    interest_expense = abs(income_stmt.iloc[0]['Interest Expense'])

                    if interest_expense > 0:
                        ratios['leverage']['interest_coverage'] = operating_income / interest_expense

            # 5. Manufacturing-Specific Operating Ratios

            # Copy from metrics if already calculated
            for key in ['r_and_d_to_revenue', 'capex_to_revenue', 'capacity_utilization', 'backlog_to_revenue']:
                if key in manufacturing_metrics:
                    ratios['operating'][key] = manufacturing_metrics[key]

            # CapEx to Depreciation ratio (indicates investment vs. maintenance)
            if cash_flow is not None and income_stmt is not None:
                if 'Capital Expenditure' in cash_flow.iloc[:,
                                            0].index and 'Depreciation & Amortization' in income_stmt.iloc[:, 0].index:
                    capex = abs(cash_flow.iloc[0]['Capital Expenditure'])
                    depreciation = income_stmt.iloc[0]['Depreciation & Amortization']

                    if depreciation > 0:
                        ratios['operating']['capex_to_depreciation'] = capex / depreciation

            # 6. Manufacturing Sector Benchmarks

            # Add sector benchmarks based on subsector
            benchmarks = {
                "Heavy_Manufacturing": {
                    "gross_margin": 0.25,
                    "operating_margin": 0.08,
                    "roic": 0.12,
                    "fixed_asset_turnover": 2.5,
                    "inventory_turnover": 5.0,
                    "cash_conversion_cycle": 60,
                    "debt_to_equity": 0.6,
                    "capex_to_depreciation": 1.2
                },
                "Automotive": {
                    "gross_margin": 0.20,
                    "operating_margin": 0.06,
                    "roic": 0.10,
                    "fixed_asset_turnover": 3.0,
                    "inventory_turnover": 8.0,
                    "cash_conversion_cycle": 40,
                    "debt_to_equity": 0.7,
                    "capex_to_depreciation": 1.3
                },
                "Aerospace_Defense": {
                    "gross_margin": 0.30,
                    "operating_margin": 0.12,
                    "roic": 0.15,
                    "fixed_asset_turnover": 3.5,
                    "inventory_turnover": 4.0,
                    "cash_conversion_cycle": 80,
                    "debt_to_equity": 0.5,
                    "r_and_d_to_revenue": 0.07
                },
                "Electronics": {
                    "gross_margin": 0.35,
                    "operating_margin": 0.13,
                    "roic": 0.18,
                    "fixed_asset_turnover": 4.0,
                    "inventory_turnover": 6.0,
                    "cash_conversion_cycle": 50,
                    "debt_to_equity": 0.4,
                    "r_and_d_to_revenue": 0.08
                },
                "Consumer_Goods": {
                    "gross_margin": 0.32,
                    "operating_margin": 0.10,
                    "roic": 0.14,
                    "fixed_asset_turnover": 3.5,
                    "inventory_turnover": 7.0,
                    "cash_conversion_cycle": 45,
                    "debt_to_equity": 0.5,
                    "capex_to_depreciation": 1.1
                },
                "Chemical_Materials": {
                    "gross_margin": 0.25,
                    "operating_margin": 0.09,
                    "roic": 0.11,
                    "fixed_asset_turnover": 2.0,
                    "inventory_turnover": 6.0,
                    "cash_conversion_cycle": 55,
                    "debt_to_equity": 0.6,
                    "capex_to_depreciation": 1.3
                }
            }

            # Add benchmarks to return value
            if subsector in benchmarks:
                ratios['benchmarks'] = benchmarks[subsector]
            else:
                ratios['benchmarks'] = benchmarks["Heavy_Manufacturing"]  # Default

            # 7. Calculate Score Relative to Benchmarks

            # For each key ratio, calculate a score (1-10) relative to benchmark
            if 'benchmarks' in ratios:
                scores = {}

                for category in ['profitability', 'efficiency', 'operating', 'leverage']:
                    for ratio_name, ratio_value in ratios[category].items():
                        if ratio_name in ratios['benchmarks'] and ratios['benchmarks'][ratio_name] is not None:
                            benchmark = ratios['benchmarks'][ratio_name]

                            # For ratios where higher is better
                            if ratio_name in ['gross_margin', 'operating_margin', 'roic', 'fixed_asset_turnover',
                                              'inventory_turnover', 'r_and_d_to_revenue', 'interest_coverage']:
                                if benchmark > 0:
                                    score = min(10, max(1, (ratio_value / benchmark) * 5))
                                else:
                                    score = 5  # Default if benchmark is zero

                            # For ratios where lower is better
                            elif ratio_name in ['cash_conversion_cycle', 'days_inventory_outstanding',
                                                'debt_to_equity', 'debt_to_ebitda']:
                                if ratio_value > 0:
                                    score = min(10, max(1, (benchmark / ratio_value) * 5))
                                else:
                                    score = 10  # Perfect score if ratio is zero (unlikely)

                            # For ratios where being close to benchmark is better
                            elif ratio_name in ['capex_to_depreciation']:
                                # Score highest when within 20% of benchmark
                                ratio = ratio_value / benchmark
                                if 0.8 <= ratio <= 1.2:
                                    score = 8 + (1 - abs(ratio - 1)) * 10  # 8-10 range
                                else:
                                    score = max(1, 8 - abs(ratio - 1) * 5)  # Lower score as we deviate

                            # Default case
                            else:
                                score = 5

                            scores[ratio_name] = score

                # Calculate average score
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    scores['overall'] = avg_score

                ratios['scores'] = scores

            return ratios

        except Exception as e:
            logger.error(f"Error calculating manufacturing-specific ratios for {ticker}: {e}")
            return {
                'profitability': {},
                'efficiency': {},
                'liquidity': {},
                'leverage': {},
                'operating': {},
                'valuation': {},
                'error': str(e)
            }