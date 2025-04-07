import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from utils.data_loader import DataLoader
from valuation.base_valuation import BaseValuation
from valuation.dcf_models import AdvancedDCFValuation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('retail_sector')


class RetailSectorValuation(AdvancedDCFValuation):
    """
    Specialized valuation models for retail sector companies.

    This class provides valuation methods tailored to different retail sub-sectors:
    - Traditional Brick and Mortar Retail
    - E-commerce
    - Omnichannel Retail
    - Discount Retailers
    - Luxury Retail
    - Specialty Retail

    Each sub-sector has specific metrics and value drivers that are incorporated
    into the valuation methods.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize retail sector valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized RetailSectorValuation")

        # Retail sub-sector specific parameters
        self.retail_subsector_parameters = {
            "Traditional_Retail": {
                "revenue_growth": 0.02,  # Lower growth expectations
                "terminal_growth": 0.015,
                "margin_improvement": 0.001,  # Limited margin improvement
                "discount_rate_premium": 0.01,  # Higher risk premium
                "store_metrics": True,
                "important_ratios": ["same_store_sales", "sales_per_sqft", "inventory_turnover"]
            },
            "E_Commerce": {
                "revenue_growth": 0.15,  # Higher growth
                "terminal_growth": 0.03,
                "margin_improvement": 0.005,  # Greater potential for margin improvement
                "discount_rate_premium": 0.015,  # Higher risk but potential rewards
                "store_metrics": False,
                "important_ratios": ["customer_acquisition_cost", "conversion_rate", "average_order_value"]
            },
            "Omnichannel": {
                "revenue_growth": 0.08,
                "terminal_growth": 0.025,
                "margin_improvement": 0.003,
                "discount_rate_premium": 0.005,
                "store_metrics": True,
                "important_ratios": ["online_sales_percentage", "sales_per_sqft", "inventory_turnover"]
            },
            "Discount": {
                "revenue_growth": 0.05,
                "terminal_growth": 0.02,
                "margin_improvement": 0.001,  # Focus on volume over margin
                "discount_rate_premium": 0.005,
                "store_metrics": True,
                "important_ratios": ["same_store_sales", "inventory_turnover", "sales_per_sqft"]
            },
            "Luxury": {
                "revenue_growth": 0.06,
                "terminal_growth": 0.025,
                "margin_improvement": 0.002,  # Higher margins but slower improvement
                "discount_rate_premium": 0.01,
                "store_metrics": True,
                "important_ratios": ["sales_per_sqft", "gross_margin", "brand_value_ratio"]
            },
            "Specialty": {
                "revenue_growth": 0.04,
                "terminal_growth": 0.02,
                "margin_improvement": 0.002,
                "discount_rate_premium": 0.01,
                "store_metrics": True,
                "important_ratios": ["same_store_sales", "sales_per_sqft", "category_dominance"]
            }
        }

    def detect_retail_subsector(self, ticker: str, financial_data: Dict[str, Any]) -> str:
        """
        Determine the retail sub-sector of a company based on available data

        Args:
            ticker: Company ticker symbol
            financial_data: Dictionary with financial statements and other data

        Returns:
            String indicating the detected retail sub-sector
        """
        try:
            # Extract company info from financial data
            company_info = financial_data.get('company_info', {})
            industry = company_info.get('industry', '').lower()
            business_summary = company_info.get('description', '').lower()

            # Try to extract store count and online presence from data
            income_stmt = financial_data.get('income_statement')
            revenue = income_stmt.iloc[0][
                'Total Revenue'] if income_stmt is not None and 'Total Revenue' in income_stmt.index else 0

            # Check for keywords in business description
            e_commerce_keywords = ['e-commerce', 'ecommerce', 'online retail', 'internet retail', 'digital platform']
            traditional_retail_keywords = ['department store', 'brick and mortar', 'physical store', 'retail store']
            discount_keywords = ['discount', 'low price', 'value retailer', 'budget', 'mass market']
            luxury_keywords = ['luxury', 'premium', 'high-end', 'upscale']
            specialty_keywords = ['specialty', 'niche', 'specialized', 'category killer']

            # Determine the sub-sector based on keywords
            if any(keyword in business_summary for keyword in e_commerce_keywords):
                if any(keyword in business_summary for keyword in traditional_retail_keywords):
                    return "Omnichannel"
                return "E_Commerce"

            if any(keyword in business_summary for keyword in discount_keywords):
                return "Discount"

            if any(keyword in business_summary for keyword in luxury_keywords):
                return "Luxury"

            if any(keyword in business_summary for keyword in specialty_keywords):
                return "Specialty"

            if any(keyword in business_summary for keyword in traditional_retail_keywords):
                return "Traditional_Retail"

            # Default to traditional retail if unable to determine
            logger.info(f"Unable to precisely determine retail sub-sector for {ticker}, defaulting to Omnichannel")
            return "Omnichannel"

        except Exception as e:
            logger.error(f"Error detecting retail sub-sector for {ticker}: {e}")
            # Default to omnichannel as it's a middle ground
            return "Omnichannel"

    def retail_dcf_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                             subsector: str = None) -> Dict[str, Any]:
        """
        Perform specialized DCF valuation for retail companies

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Retail sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_retail_subsector(ticker, financial_data)

            logger.info(f"Performing retail DCF valuation for {ticker} as {subsector}")

            # Get sub-sector specific parameters
            if subsector in self.retail_subsector_parameters:
                subsector_params = self.retail_subsector_parameters[subsector]
            else:
                # Default to omnichannel if sub-sector not recognized
                subsector_params = self.retail_subsector_parameters["Omnichannel"]
                logger.warning(f"Unrecognized retail sub-sector {subsector}, using Omnichannel parameters")

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Check required data
            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for retail DCF valuation")

            # Get historical free cash flow data
            historical_fcf = self._calculate_historical_fcf(income_stmt, cash_flow)

            if historical_fcf.empty:
                raise ValueError("Unable to calculate historical free cash flow")

            # Get base parameters
            params = self._get_dcf_parameters("Consumer Discretionary")  # Use Consumer Discretionary as base

            # Adjust parameters based on sub-sector
            adjusted_params = params.copy()
            adjusted_params['terminal_growth_rate'] = subsector_params['terminal_growth']

            # Modify discount rate based on sub-sector risk
            base_discount_rate = self._calculate_discount_rate(ticker, financial_data, "Consumer Discretionary")
            if base_discount_rate:
                adjusted_discount_rate = base_discount_rate + subsector_params['discount_rate_premium']
            else:
                adjusted_discount_rate = params['default_discount_rate'] + subsector_params['discount_rate_premium']

            # Calculate retail-specific metrics
            retail_metrics = self._calculate_retail_metrics(ticker, financial_data, subsector)

            # Forecast parameters
            forecast_years = params['forecast_years']

            # Calculate initial growth rate (using base and adjustment for retail metrics)
            base_growth_rate = self._estimate_growth_rate(historical_fcf)

            # Adjust growth based on retail metrics
            growth_adjustment = 0

            # Adjust for same store sales growth if available
            if 'same_store_sales_growth' in retail_metrics:
                # Weight same store sales growth in the growth rate calculation
                if retail_metrics['same_store_sales_growth'] > 0:
                    growth_adjustment += min(retail_metrics['same_store_sales_growth'] * 0.5, 0.02)
                else:
                    growth_adjustment += max(retail_metrics['same_store_sales_growth'] * 0.5, -0.02)

            # Adjust for store expansion
            if 'store_growth' in retail_metrics and retail_metrics['store_growth'] > 0:
                growth_adjustment += min(retail_metrics['store_growth'] * 0.3, 0.01)

            # Adjust for e-commerce growth
            if 'e_commerce_growth' in retail_metrics:
                growth_adjustment += min(retail_metrics['e_commerce_growth'] * 0.2, 0.03)

            # Final growth rate (base + metrics adjustment + subsector default)
            # Weight: 50% calculated growth, 30% metrics adjustment, 20% subsector default
            initial_growth_rate = (base_growth_rate * 0.5) + (growth_adjustment) + (
                        subsector_params['revenue_growth'] * 0.2)

            # Cap growth to reasonable range
            initial_growth_rate = max(0.01, min(0.25, initial_growth_rate))

            logger.info(f"Calculated initial growth rate for {ticker}: {initial_growth_rate:.2%}")

            # Starting FCF (most recent)
            last_fcf = historical_fcf.iloc[0]

            # Forecast cash flows with retail-specific adjustments
            forecasted_fcf = []

            for year in range(1, forecast_years + 1):
                # Apply declining growth over time
                if year <= 2:
                    growth = initial_growth_rate
                elif year <= 5:
                    # Linear decline from initial to steady state
                    growth = initial_growth_rate - ((initial_growth_rate - subsector_params['terminal_growth'])
                                                    * (year - 2) / 3)
                else:
                    growth = subsector_params['terminal_growth']

                # Apply margin improvement if applicable
                if year > 1:
                    margin_improvement = min(year * subsector_params['margin_improvement'], 0.05)
                    margin_factor = 1 + margin_improvement
                else:
                    margin_factor = 1

                # Calculate FCF with growth and margin improvements
                fcf = last_fcf * (1 + growth) ** year * margin_factor

                # Adjust for store openings/closings for traditional retailers
                if subsector_params['store_metrics'] and 'planned_store_changes' in retail_metrics:
                    if year <= len(retail_metrics['planned_store_changes']):
                        store_change_impact = retail_metrics['planned_store_changes'][year - 1] * 0.01
                        fcf *= (1 + store_change_impact)

                forecasted_fcf.append(fcf)

            # Calculate terminal value
            terminal_growth = adjusted_params['terminal_growth_rate']
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

            # Add operating lease adjustment for retailers (if applicable)
            if 'operating_lease_adjustment' in retail_metrics:
                equity_value -= retail_metrics['operating_lease_adjustment']

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

            # Apply margin of safety (higher for more volatile retail sub-sectors)
            safety_margin = adjusted_params['default_margin_of_safety']
            if subsector in ["E_Commerce", "Specialty"]:
                safety_margin *= 1.2  # 20% higher safety margin

            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            # Prepare results
            result = {
                'company': ticker,
                'method': 'retail_dcf',
                'subsector': subsector,
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
                'retail_metrics': retail_metrics
            }

            return result

        except Exception as e:
            logger.error(f"Error in retail DCF valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'retail_dcf',
                'subsector': subsector if subsector else 'Unknown',
                'enterprise_value': None,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_retail_metrics(self, ticker: str, financial_data: Dict[str, Any],
                                  subsector: str) -> Dict[str, Any]:
        """
        Calculate retail-specific metrics for valuation adjustments

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary
            subsector: Retail sub-sector

        Returns:
            Dictionary with retail-specific metrics
        """
        metrics = {}

        try:
            # Extract data from financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            # 1. Extract same-store sales growth (comp sales) from descriptions if available
            business_summary = company_info.get('description', '').lower()

            # Try to find same-store sales growth in description
            sss_patterns = [
                r'same[\s-]store sales growth of (\d+\.?\d*)%',
                r'comparable sales grew by (\d+\.?\d*)%',
                r'comp sales of (\d+\.?\d*)%',
            ]

            for pattern in sss_patterns:
                import re
                match = re.search(pattern, business_summary)
                if match:
                    metrics['same_store_sales_growth'] = float(match.group(1)) / 100
                    break

            # 2. Estimate inventory metrics
            if income_stmt is not None and balance_sheet is not None:
                if 'Cost of Revenue' in income_stmt.index and 'Inventory' in balance_sheet.index:
                    # Get the most recent data
                    cogs = income_stmt.iloc[0]['Cost of Revenue']
                    inventory = balance_sheet.iloc[0]['Inventory']

                    # Calculate inventory turnover
                    if inventory > 0:
                        inventory_turnover = cogs / inventory
                        metrics['inventory_turnover'] = inventory_turnover

                        # Calculate days inventory outstanding (DIO)
                        metrics['days_inventory_outstanding'] = 365 / inventory_turnover

            # 3. Estimate e-commerce percentage if available
            # This would typically come from company reports, but we'll use industry averages
            # based on the subsector if not available
            if subsector == "E_Commerce":
                metrics['e_commerce_percentage'] = 0.95  # 95% e-commerce
                metrics['e_commerce_growth'] = 0.20  # 20% growth
            elif subsector == "Omnichannel":
                metrics['e_commerce_percentage'] = 0.30  # 30% e-commerce
                metrics['e_commerce_growth'] = 0.15  # 15% growth
            elif subsector == "Traditional_Retail":
                metrics['e_commerce_percentage'] = 0.10  # 10% e-commerce
                metrics['e_commerce_growth'] = 0.10  # 10% growth
            else:
                # Default for other subsectors
                metrics['e_commerce_percentage'] = 0.20
                metrics['e_commerce_growth'] = 0.12

            # 4. Estimate store count and changes
            # Ideally this would come from company reports
            # For now, use industry averages or placeholder values
            if subsector_params := self.retail_subsector_parameters.get(subsector):
                if subsector_params['store_metrics']:
                    # Placeholder for store count (would come from company data)
                    metrics['estimated_store_count'] = 500

                    # Estimate store growth based on capex and industry patterns
                    if cash_flow is not None and 'Capital Expenditure' in cash_flow.index:
                        capex = abs(cash_flow.iloc[0]['Capital Expenditure'])

                        # Very rough estimation of store growth from capex
                        if income_stmt is not None and 'Total Revenue' in income_stmt.index:
                            revenue = income_stmt.iloc[0]['Total Revenue']
                            capex_to_revenue = capex / revenue

                            # Estimate store growth based on capex to revenue ratio
                            if capex_to_revenue > 0.10:
                                store_growth = 0.08  # High capex = high growth
                            elif capex_to_revenue > 0.05:
                                store_growth = 0.05  # Moderate capex
                            else:
                                store_growth = 0.02  # Low capex

                            metrics['store_growth'] = store_growth

                            # Placeholder for planned store changes
                            # In a real implementation, this would come from company guidance
                            metrics['planned_store_changes'] = [
                                store_growth,
                                store_growth * 0.9,
                                store_growth * 0.8,
                                store_growth * 0.7,
                                store_growth * 0.6,
                            ]

            # 5. Operating lease adjustment (important for retailers)
            # ASC 842 requires operating leases on balance sheet, but older data might not have this
            if balance_sheet is not None:
                # Check if operating leases are already on balance sheet
                lease_liabilities = 0
                for item in ['Operating Lease Liability', 'Lease Liability']:
                    if item in balance_sheet.index:
                        lease_liabilities = balance_sheet.iloc[0][item]
                        break

                # If not on balance sheet, estimate from rental expense
                if lease_liabilities == 0 and income_stmt is not None:
                    # Look for rental expense in income statement
                    rental_expense = 0
                    for item in ['Rental Expense', 'Lease Expense']:
                        if item in income_stmt.index:
                            rental_expense = income_stmt.iloc[0][item]
                            break

                    # If found, estimate lease liability using multiple of annual rent
                    if rental_expense > 0:
                        # Typical multiple is 5-8x annual rent
                        lease_multiple = 6
                        operating_lease_adjustment = rental_expense * lease_multiple
                        metrics['operating_lease_adjustment'] = operating_lease_adjustment

            # 6. Gross margin and operating margin (key for retailers)
            if income_stmt is not None:
                if 'Total Revenue' in income_stmt.index and 'Cost of Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    cogs = income_stmt.iloc[0]['Cost of Revenue']

                    gross_profit = revenue - cogs
                    metrics['gross_margin'] = gross_profit / revenue

                if 'Operating Income' in income_stmt.index:
                    operating_income = income_stmt.iloc[0]['Operating Income']
                    metrics['operating_margin'] = operating_income / revenue

            # 7. Estimate sales per square foot (key retail metric)
            # This would typically come from company reports
            # For now, use industry benchmarks based on subsector
            if subsector == "Luxury":
                metrics['sales_per_sqft'] = 1500  # $1,500 per square foot (high end)
            elif subsector == "Specialty":
                metrics['sales_per_sqft'] = 600  # $600 per square foot
            elif subsector == "Traditional_Retail":
                metrics['sales_per_sqft'] = 300  # $300 per square foot
            elif subsector == "Discount":
                metrics['sales_per_sqft'] = 250  # $250 per square foot

            return metrics

        except Exception as e:
            logger.error(f"Error calculating retail metrics for {ticker}: {e}")
            return metrics

    def retail_relative_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  subsector: str = None) -> Dict[str, Any]:
        """
        Perform relative valuation for retail companies using sector-specific multiples

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Retail sub-sector (will be detected if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_retail_subsector(ticker, financial_data)

            logger.info(f"Performing retail relative valuation for {ticker} as {subsector}")

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

            # Define multiple ranges by subsector
            subsector_multiples = {
                "Traditional_Retail": {
                    "EV/Sales": 0.5,
                    "EV/EBITDA": 7.0,
                    "P/E": 12.0,
                    "P/B": 1.5,
                },
                "E_Commerce": {
                    "EV/Sales": 2.0,
                    "EV/EBITDA": 15.0,
                    "P/E": 30.0,
                    "P/B": 4.0,
                },
                "Omnichannel": {
                    "EV/Sales": 0.8,
                    "EV/EBITDA": 10.0,
                    "P/E": 18.0,
                    "P/B": 2.5,
                },
                "Discount": {
                    "EV/Sales": 0.7,
                    "EV/EBITDA": 9.0,
                    "P/E": 16.0,
                    "P/B": 2.0,
                },
                "Luxury": {
                    "EV/Sales": 2.5,
                    "EV/EBITDA": 12.0,
                    "P/E": 20.0,
                    "P/B": 3.0,
                },
                "Specialty": {
                    "EV/Sales": 1.2,
                    "EV/EBITDA": 10.0,
                    "P/E": 18.0,
                    "P/B": 2.5,
                }
            }

            # Get multiples for the identified subsector
            multiples = subsector_multiples.get(subsector, subsector_multiples["Omnichannel"])

            # Retail-specific metrics (if available)
            retail_metrics = self._calculate_retail_metrics(ticker, financial_data, subsector)

            # Adjust multiples based on company-specific metrics
            adjusted_multiples = multiples.copy()

            # Adjust based on inventory turnover (higher is better)
            if 'inventory_turnover' in retail_metrics:
                inventory_turnover = retail_metrics['inventory_turnover']
                # Benchmark: 4 is average for retail
                if inventory_turnover > 6:
                    # Premium for high inventory turnover
                    multiple_adjustment = 0.15  # 15% premium
                elif inventory_turnover < 3:
                    # Discount for low inventory turnover
                    multiple_adjustment = -0.15  # 15% discount
                else:
                    multiple_adjustment = (inventory_turnover - 4) / 10  # Smaller adjustment

                # Apply adjustment to all multiples
                for key in adjusted_multiples:
                    adjusted_multiples[key] *= (1 + multiple_adjustment)

            # Adjust based on gross margin (higher is better)
            if 'gross_margin' in retail_metrics:
                gross_margin = retail_metrics['gross_margin']
                # Benchmarks vary by subsector
                if subsector == "Luxury":
                    benchmark = 0.50  # 50% benchmark for luxury
                elif subsector == "Discount":
                    benchmark = 0.25  # 25% benchmark for discount
                else:
                    benchmark = 0.35  # 35% for others

                margin_adjustment = (gross_margin - benchmark) / benchmark
                margin_impact = min(max(margin_adjustment * 0.5, -0.2), 0.2)  # Cap at ±20%

                # Apply to EV/EBITDA and P/E which are most affected by margins
                adjusted_multiples["EV/EBITDA"] *= (1 + margin_impact)
                adjusted_multiples["P/E"] *= (1 + margin_impact)

            # Adjust for e-commerce presence (higher is better)
            if 'e_commerce_percentage' in retail_metrics:
                e_commerce_pct = retail_metrics['e_commerce_percentage']
                # Higher e-commerce warrants higher multiples (except for pure e-commerce)
                if subsector != "E_Commerce" and e_commerce_pct > 0.15:
                    ecom_premium = min((e_commerce_pct - 0.15) * 0.5, 0.15)  # Cap at 15% premium

                    # Apply to EV/Sales which benefits most from e-commerce
                    adjusted_multiples["EV/Sales"] *= (1 + ecom_premium)
                    adjusted_multiples["P/E"] *= (1 + ecom_premium * 0.5)

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

            # EV/Sales valuation
            if 'revenue' in metrics and metrics['revenue'] > 0:
                ev_sales_multiple = adjusted_multiples.get("EV/Sales", 0.8)
                ev_from_sales = metrics['revenue'] * ev_sales_multiple
                equity_value_from_sales = ev_from_sales - total_debt + cash

                valuations['ev_sales'] = {
                    'multiple': ev_sales_multiple,
                    'enterprise_value': ev_from_sales,
                    'equity_value': equity_value_from_sales,
                    'description': 'Enterprise Value to Sales'
                }

            # EV/EBITDA valuation
            if 'ebitda' in metrics and metrics['ebitda'] > 0:
                ev_ebitda_multiple = adjusted_multiples.get("EV/EBITDA", 10.0)
                ev_from_ebitda = metrics['ebitda'] * ev_ebitda_multiple
                equity_value_from_ebitda = ev_from_ebitda - total_debt + cash

                valuations['ev_ebitda'] = {
                    'multiple': ev_ebitda_multiple,
                    'enterprise_value': ev_from_ebitda,
                    'equity_value': equity_value_from_ebitda,
                    'description': 'Enterprise Value to EBITDA'
                }

            # P/E valuation
            if 'earnings' in metrics and metrics['earnings'] > 0:
                pe_multiple = adjusted_multiples.get("P/E", 15.0)
                equity_value_from_earnings = metrics['earnings'] * pe_multiple

                valuations['pe'] = {
                    'multiple': pe_multiple,
                    'equity_value': equity_value_from_earnings,
                    'description': 'Price to Earnings'
                }

            # P/B valuation
            if 'book_value' in metrics and metrics['book_value'] > 0:
                pb_multiple = adjusted_multiples.get("P/B", 2.0)
                equity_value_from_book = metrics['book_value'] * pb_multiple

                valuations['pb'] = {
                    'multiple': pb_multiple,
                    'equity_value': equity_value_from_book,
                    'description': 'Price to Book'
                }

            # Calculate average equity value
            equity_values = []
            for method, valuation in valuations.items():
                if 'equity_value' in valuation:
                    equity_values.append(valuation['equity_value'])

            if equity_values:
                avg_equity_value = sum(equity_values) / len(equity_values)
            else:
                avg_equity_value = None

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

            # Apply margin of safety
            safety_margin = 0.25  # 25% discount for retail companies due to volatility
            if value_per_share:
                conservative_value = value_per_share * (1 - safety_margin)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'retail_relative_valuation',
                'subsector': subsector,
                'valuations': valuations,
                'average_equity_value': avg_equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'per_share_values': per_share_values,
                'metrics': metrics,
                'multiples_used': adjusted_multiples,
                'retail_metrics': retail_metrics
            }

        except Exception as e:
            logger.error(f"Error in retail relative valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'retail_relative_valuation',
                'average_equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def store_based_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                              store_count: int = None, avg_store_value: float = None) -> Dict[str, Any]:
        """
        Perform valuation based on store count and average store value
        (Useful for traditional retailers with substantial physical presence)

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            store_count: Number of stores (if None, will be estimated)
            avg_store_value: Average value per store (if None, will be estimated)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Get company info
            company_info = financial_data.get('company_info', {})
            business_summary = company_info.get('description', '')

            # Determine subsector
            subsector = self.detect_retail_subsector(ticker, financial_data)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Calculate retail metrics
            retail_metrics = self._calculate_retail_metrics(ticker, financial_data, subsector)

            # Try to determine store count from metrics or business description
            if store_count is None:
                # First check if we already have it in retail metrics
                if 'estimated_store_count' in retail_metrics:
                    store_count = retail_metrics['estimated_store_count']
                else:
                    # Try to extract from business summary using regex
                    import re
                    store_patterns = [
                        r'(\d{1,3}(?:,\d{3})*) stores',
                        r'operates (\d{1,3}(?:,\d{3})*) locations',
                        r'(\d{1,3}(?:,\d{3})*) retail outlets'
                    ]

                    for pattern in store_patterns:
                        match = re.search(pattern, business_summary)
                        if match:
                            # Convert to integer (removing commas)
                            store_count = int(match.group(1).replace(',', ''))
                            break

                    # If still not found, use estimated value based on revenue
                    if store_count is None and income_stmt is not None and 'Total Revenue' in income_stmt.index:
                        revenue = income_stmt.iloc[0]['Total Revenue']

                        # Rough estimation based on subsector
                        if subsector == "Traditional_Retail":
                            # Assume $5M revenue per store
                            store_count = int(revenue / 5_000_000)
                        elif subsector == "Discount":
                            # Assume $8M revenue per store
                            store_count = int(revenue / 8_000_000)
                        elif subsector == "Luxury":
                            # Assume $3M revenue per store
                            store_count = int(revenue / 3_000_000)
                        elif subsector == "Specialty":
                            # Assume $4M revenue per store
                            store_count = int(revenue / 4_000_000)
                        else:
                            # Default
                            store_count = int(revenue / 5_000_000)

                        # Cap at reasonable range
                        store_count = max(10, min(5000, store_count))

            # Determine average store value if not provided
            if avg_store_value is None:
                # Base values by subsector
                if subsector == "Luxury":
                    base_store_value = 5_000_000  # $5M per luxury store
                elif subsector == "Traditional_Retail":
                    base_store_value = 2_500_000  # $2.5M per traditional store
                elif subsector == "Discount":
                    base_store_value = 3_000_000  # $3M per discount store
                elif subsector == "Specialty":
                    base_store_value = 2_000_000  # $2M per specialty store
                else:
                    base_store_value = 2_500_000  # Default

                # Adjust based on metrics (if available)
                if 'sales_per_sqft' in retail_metrics:
                    # Higher sales per square foot = higher store value
                    # Assume baseline is $300 per sqft
                    sales_per_sqft = retail_metrics['sales_per_sqft']
                    sqft_factor = sales_per_sqft / 300

                    # Apply factor with diminishing returns
                    import math
                    value_factor = math.sqrt(sqft_factor)

                    # Apply to base value
                    avg_store_value = base_store_value * value_factor
                else:
                    avg_store_value = base_store_value

                # If we have profitability data, further adjust store value
                if 'operating_margin' in retail_metrics:
                    # Baseline margin of 8%
                    margin = retail_metrics['operating_margin']
                    margin_factor = margin / 0.08

                    # Apply with 50% weight to avoid extreme values
                    avg_store_value *= (1 + (margin_factor - 1) * 0.5)

            # Calculate total store value
            total_store_value = store_count * avg_store_value

            # Adjust for corporate overhead, debt, and other factors
            # Get balance sheet items
            net_debt = 0
            if balance_sheet is not None:
                # Calculate debt
                total_debt = 0
                if 'Total Debt' in balance_sheet.index:
                    total_debt = balance_sheet.iloc[0]['Total Debt']
                else:
                    if 'Long Term Debt' in balance_sheet.index:
                        total_debt += balance_sheet.iloc[0]['Long Term Debt']
                    if 'Short Term Debt' in balance_sheet.index:
                        total_debt += balance_sheet.iloc[0]['Short Term Debt']

                # Calculate cash
                cash = 0
                if 'Cash and Cash Equivalents' in balance_sheet.index:
                    cash = balance_sheet.iloc[0]['Cash and Cash Equivalents']
                elif 'Cash and Short Term Investments' in balance_sheet.index:
                    cash = balance_sheet.iloc[0]['Cash and Short Term Investments']

                net_debt = total_debt - cash

            # Adjust for e-commerce value
            ecommerce_value = 0
            if 'e_commerce_percentage' in retail_metrics and income_stmt is not None and 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.iloc[0]['Total Revenue']
                e_commerce_pct = retail_metrics['e_commerce_percentage']

                # E-commerce revenue
                ecommerce_revenue = revenue * e_commerce_pct

                # Apply higher multiple to e-commerce revenue
                ecommerce_multiple = 1.5  # 1.5x sales for e-commerce component
                ecommerce_value = ecommerce_revenue * ecommerce_multiple

            # Corporate overhead adjustment (negative value)
            corporate_overhead = -total_store_value * 0.15  # Assume 15% overhead drag

            # Calculate final equity value
            equity_value = total_store_value + ecommerce_value + corporate_overhead - net_debt

            # Per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                # Try to estimate from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    value_per_share = equity_value / estimated_shares
                else:
                    value_per_share = None

            # Apply margin of safety
            safety_margin = 0.25  # 25% for store-based valuation due to uncertainties
            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            return {
                'company': ticker,
                'method': 'store_based_valuation',
                'subsector': subsector,
                'store_count': store_count,
                'avg_store_value': avg_store_value,
                'total_store_value': total_store_value,
                'ecommerce_value': ecommerce_value,
                'corporate_overhead': corporate_overhead,
                'net_debt': net_debt,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'retail_metrics': retail_metrics
            }

        except Exception as e:
            logger.error(f"Error in store-based valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'store_based_valuation',
                'store_count': store_count,
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def retail_sum_of_parts(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform sum-of-parts valuation for retailers with multiple brands or segments

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine subsector
            subsector = self.detect_retail_subsector(ticker, financial_data)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Start with store-based valuation as one component
            store_valuation = self.store_based_valuation(ticker, financial_data)

            # Then value e-commerce operations separately
            retail_metrics = self._calculate_retail_metrics(ticker, financial_data, subsector)

            # Parts valuation
            parts = {}

            # 1. Physical Stores (from store valuation)
            if 'total_store_value' in store_valuation:
                parts['physical_stores'] = {
                    'value': store_valuation['total_store_value'],
                    'method': 'store_based_valuation',
                    'description': f"{store_valuation.get('store_count', 'Unknown')} physical retail locations"
                }

            # 2. E-commerce Operations
            if 'e_commerce_percentage' in retail_metrics and income_stmt is not None and 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.iloc[0]['Total Revenue']
                e_commerce_pct = retail_metrics['e_commerce_percentage']

                # E-commerce revenue
                ecommerce_revenue = revenue * e_commerce_pct

                # Higher multiple for e-commerce operations
                ecommerce_multiple = 2.0 if e_commerce_pct > 0.5 else 1.5
                ecommerce_value = ecommerce_revenue * ecommerce_multiple

                parts['ecommerce'] = {
                    'value': ecommerce_value,
                    'method': 'revenue_multiple',
                    'multiple': ecommerce_multiple,
                    'revenue': ecommerce_revenue,
                    'description': f"E-commerce operations ({e_commerce_pct:.1%} of revenue)"
                }

            # 3. Private Label Brands (if applicable)
            # Estimate from gross margin
            if 'gross_margin' in retail_metrics and retail_metrics['gross_margin'] > 0.35:
                # Assume high margin indicates valuable private label brands
                private_label_value = 0

                if income_stmt is not None and 'Gross Profit' in income_stmt.index:
                    gross_profit = income_stmt.iloc[0]['Gross Profit']

                    # Estimate private label premium
                    private_label_premium = (retail_metrics['gross_margin'] - 0.35) * 2
                    private_label_value = gross_profit * private_label_premium

                    parts['private_labels'] = {
                        'value': private_label_value,
                        'method': 'gross_profit_multiple',
                        'description': "Value of private label brands"
                    }

            # 4. Real Estate (if owned)
            real_estate_value = 0
            if balance_sheet is not None and 'Property Plant and Equipment' in balance.index:
                ppe = balance.loc['Property Plant and Equipment']

                # Assume 60% of PPE is real estate for traditional retailers
                if subsector in ["Traditional_Retail", "Discount"]:
                    real_estate_pct = 0.6
                else:
                    real_estate_pct = 0.4

                real_estate_value = ppe * real_estate_pct

                # Apply premium to book value of real estate (often undervalued)
                real_estate_premium = 1.3  # 30% premium to book value
                real_estate_value *= real_estate_premium

                parts['real_estate'] = {
                    'value': real_estate_value,
                    'method': 'adjusted_book_value',
                    'description': "Owned real estate assets"
                }

            # 5. Loyalty Program (if significant)
            loyalty_value = 0

            # Rough estimate based on revenue
            if income_stmt is not None and 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.iloc[0]['Total Revenue']

                # Assume loyalty program value is 3-5% of revenue for retailers with strong programs
                loyalty_value = revenue * 0.04

                parts['loyalty_program'] = {
                    'value': loyalty_value,
                    'method': 'revenue_percentage',
                    'description': "Customer loyalty program value"
                }

            # Sum the parts
            sum_of_parts = sum(part['value'] for part in parts.values())

            # Adjust for corporate overhead and synergies
            corporate_overhead = -sum_of_parts * 0.15  # 15% overhead drag

            # Add corporate adjustments to parts
            parts['corporate_overhead'] = {
                'value': corporate_overhead,
                'method': 'percentage_of_sum',
                'description': "Corporate overhead and admin costs"
            }

            # Get debt and cash
            net_debt = 0
            if balance_sheet is not None:
                # Calculate debt
                total_debt = 0
                if 'Total Debt' in balance_sheet.index:
                    total_debt = balance_sheet.iloc[0]['Total Debt']
                else:
                    if 'Long Term Debt' in balance_sheet.index:
                        total_debt += balance_sheet.iloc[0]['Long Term Debt']
                    if 'Short Term Debt' in balance_sheet.index:
                        total_debt += balance_sheet.iloc[0]['Short Term Debt']

                # Calculate cash
                cash = 0
                if 'Cash and Cash Equivalents' in balance_sheet.index:
                    cash = balance_sheet.iloc[0]['Cash and Cash Equivalents']
                elif 'Cash and Short Term Investments' in balance_sheet.index:
                    cash = balance_sheet.iloc[0]['Cash and Short Term Investments']

                net_debt = total_debt - cash

            # Calculate adjusted sum of parts
            adjusted_sum = sum_of_parts + corporate_overhead

            # Final equity value
            equity_value = adjusted_sum - net_debt

            # Per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                # Try to estimate from market data
                current_price = market_data.get('share_price')
                market_cap = market_data.get('market_cap')
                if current_price and market_cap and current_price > 0:
                    estimated_shares = market_cap / current_price
                    value_per_share = equity_value / estimated_shares
                else:
                    value_per_share = None

            # Apply margin of safety
            safety_margin = 0.25  # 25% for sum-of-parts valuation
            conservative_value = value_per_share * (1 - safety_margin) if value_per_share else None

            return {
                'company': ticker,
                'method': 'retail_sum_of_parts',
                'subsector': subsector,
                'parts': parts,
                'sum_of_parts': sum_of_parts,
                'corporate_overhead': corporate_overhead,
                'adjusted_sum': adjusted_sum,
                'net_debt': net_debt,
                'equity_value': equity_value,
                'value_per_share': value_per_share,
                'conservative_value': conservative_value,
                'retail_metrics': retail_metrics
            }

        except Exception as e:
            logger.error(f"Error in retail sum-of-parts valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'retail_sum_of_parts',
                'equity_value': None,
                'value_per_share': None,
                'error': str(e)
            }

    def apply_retail_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                               subsector: str = None) -> Dict[str, Any]:
        """
        Apply the most appropriate retail valuation method based on sub-sector and available data

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Retail sub-sector (will be detected if None)

        Returns:
            Dictionary with comprehensive valuation results
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_retail_subsector(ticker, financial_data)

            logger.info(f"Applying comprehensive retail valuation for {ticker} as {subsector}")

            # Apply different valuation methods
            dcf_result = self.retail_dcf_valuation(ticker, financial_data, subsector)
            relative_result = self.retail_relative_valuation(ticker, financial_data, subsector)

            # Apply store-based valuation for traditional retailers with physical stores
            store_based_result = None
            if subsector in ["Traditional_Retail", "Discount", "Luxury", "Specialty"]:
                store_based_result = self.store_based_valuation(ticker, financial_data)

            # Apply sum-of-parts for retailers with diverse operations
            sotp_result = None
            if subsector == "Omnichannel":
                sotp_result = self.retail_sum_of_parts(ticker, financial_data)

            # Determine weights for different methods based on sub-sector
            weights = {}

            if subsector == "E_Commerce":
                weights = {
                    'dcf': 0.50,
                    'relative': 0.50
                }
            elif subsector == "Traditional_Retail":
                weights = {
                    'dcf': 0.30,
                    'relative': 0.40,
                    'store': 0.30
                }
            elif subsector == "Omnichannel":
                weights = {
                    'dcf': 0.30,
                    'relative': 0.30,
                    'sotp': 0.40
                }
            elif subsector == "Discount":
                weights = {
                    'dcf': 0.25,
                    'relative': 0.40,
                    'store': 0.35
                }
            elif subsector == "Luxury":
                weights = {
                    'dcf': 0.30,
                    'relative': 0.35,
                    'store': 0.35
                }
            elif subsector == "Specialty":
                weights = {
                    'dcf': 0.35,
                    'relative': 0.40,
                    'store': 0.25
                }
            else:
                # Default weights
                weights = {
                    'dcf': 0.40,
                    'relative': 0.60
                }

            # Collect per-share values from different methods
            values_per_share = {}

            if dcf_result and 'value_per_share' in dcf_result and dcf_result['value_per_share']:
                values_per_share['dcf'] = dcf_result['value_per_share']

            if relative_result and 'value_per_share' in relative_result and relative_result['value_per_share']:
                values_per_share['relative'] = relative_result['value_per_share']

            if store_based_result and 'value_per_share' in store_based_result and store_based_result['value_per_share']:
                values_per_share['store'] = store_based_result['value_per_share']

            if sotp_result and 'value_per_share' in sotp_result and sotp_result['value_per_share']:
                values_per_share['sotp'] = sotp_result['value_per_share']

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

            # Apply margin of safety (using a standard 25% for retail)
            safety_margin = 0.25
            conservative_value = final_value_per_share * (1 - safety_margin) if final_value_per_share else None

            # Compile all results
            return {
                'company': ticker,
                'method': 'comprehensive_retail_valuation',
                'subsector': subsector,
                'value_per_share': final_value_per_share,
                'conservative_value': conservative_value,
                'values_by_method': values_per_share,
                'weights': weights,
                'methods': {
                    'dcf': dcf_result,
                    'relative': relative_result,
                    'store_based': store_based_result,
                    'sum_of_parts': sotp_result
                },
                'safety_margin': safety_margin
            }

        except Exception as e:
            logger.error(f"Error in comprehensive retail valuation for {ticker}: {e}")
            return {
                'company': ticker,
                'method': 'comprehensive_retail_valuation',
                'subsector': subsector if subsector else 'Unknown',
                'value_per_share': None,
                'error': str(e)
            }

    def get_retail_specific_ratios(self, ticker: str, financial_data: Dict[str, Any] = None,
                                   subsector: str = None) -> Dict[str, Any]:
        """
        Calculate retail-specific financial ratios that are important for analyzing retail companies

        Args:
            ticker: Company ticker symbol
            financial_data: Financial data dictionary (will be loaded if None)
            subsector: Retail sub-sector (will be detected if None)

        Returns:
            Dictionary with retail-specific ratios
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine sub-sector if not provided
            if subsector is None:
                subsector = self.detect_retail_subsector(ticker, financial_data)

            logger.info(f"Calculating retail-specific ratios for {ticker} as {subsector}")

            # Extract financial statements and metrics
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get base retail metrics
            retail_metrics = self._calculate_retail_metrics(ticker, financial_data, subsector)

            # Initialize ratios dictionary
            ratios = {
                'general': {},
                'efficiency': {},
                'profitability': {},
                'growth': {},
                'operational': {},
                'valuation': {}
            }

            # 1. General Financial Ratios (that are particularly important for retail)
            if income_stmt is not None and balance_sheet is not None:
                # Get required data points
                if 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']

                    # Historical revenue growth (if we have multiple periods)
                    if income_stmt.shape[1] >= 2:
                        prev_revenue = income_stmt.iloc[1]['Total Revenue']
                        revenue_growth = (revenue / prev_revenue) - 1
                        ratios['growth']['revenue_growth'] = revenue_growth

                # Gross Margin
                if 'Total Revenue' in income_stmt.index and 'Cost of Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    cogs = income_stmt.iloc[0]['Cost of Revenue']
                    gross_profit = revenue - cogs

                    gross_margin = gross_profit / revenue
                    ratios['profitability']['gross_margin'] = gross_margin

                # Operating Margin
                if 'Total Revenue' in income_stmt.index and 'Operating Income' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    operating_income = income_stmt.iloc[0]['Operating Income']

                    operating_margin = operating_income / revenue
                    ratios['profitability']['operating_margin'] = operating_margin

                # Net Margin
                if 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']
                    net_income = income_stmt.iloc[0]['Net Income']

                    net_margin = net_income / revenue
                    ratios['profitability']['net_margin'] = net_margin

            # 2. Retail-Specific Efficiency Ratios

            # Inventory Turnover Ratio
            if 'inventory_turnover' in retail_metrics:
                ratios['efficiency']['inventory_turnover'] = retail_metrics['inventory_turnover']
            elif balance_sheet is not None and income_stmt is not None:
                if 'Inventory' in balance_sheet.index and 'Cost of Revenue' in income_stmt.index:
                    inventory = balance_sheet.iloc[0]['Inventory']
                    cogs = income_stmt.iloc[0]['Cost of Revenue']

                    if inventory > 0:
                        inventory_turnover = cogs / inventory
                        ratios['efficiency']['inventory_turnover'] = inventory_turnover

            # Days Inventory Outstanding (DIO)
            if 'days_inventory_outstanding' in retail_metrics:
                ratios['efficiency']['days_inventory_outstanding'] = retail_metrics['days_inventory_outstanding']
            elif 'inventory_turnover' in ratios['efficiency']:
                days_inventory = 365 / ratios['efficiency']['inventory_turnover']
                ratios['efficiency']['days_inventory_outstanding'] = days_inventory

            # Days Sales Outstanding (DSO) - Relevant for retailers with credit sales
            if balance_sheet is not None and income_stmt is not None:
                if 'Net Receivables' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                    receivables = balance_sheet.iloc[0]['Net Receivables']
                    revenue = income_stmt.iloc[0]['Total Revenue']

                    if revenue > 0:
                        dso = (receivables / revenue) * 365
                        ratios['efficiency']['days_sales_outstanding'] = dso

            # Days Payable Outstanding (DPO)
            if balance_sheet is not None and income_stmt is not None:
                if 'Accounts Payable' in balance_sheet.index and 'Cost of Revenue' in income_stmt.index:
                    payables = balance_sheet.iloc[0]['Accounts Payable']
                    cogs = income_stmt.iloc[0]['Cost of Revenue']

                    if cogs > 0:
                        dpo = (payables / cogs) * 365
                        ratios['efficiency']['days_payable_outstanding'] = dpo

            # Cash Conversion Cycle (CCC)
            if ('days_inventory_outstanding' in ratios['efficiency'] and
                    'days_sales_outstanding' in ratios['efficiency'] and
                    'days_payable_outstanding' in ratios['efficiency']):
                dio = ratios['efficiency']['days_inventory_outstanding']
                dso = ratios['efficiency']['days_sales_outstanding']
                dpo = ratios['efficiency']['days_payable_outstanding']

                ccc = dio + dso - dpo
                ratios['efficiency']['cash_conversion_cycle'] = ccc

            # 3. Retail-Specific Operational Metrics

            # Same-Store Sales Growth (Comp Sales)
            if 'same_store_sales_growth' in retail_metrics:
                ratios['operational']['same_store_sales_growth'] = retail_metrics['same_store_sales_growth']

            # Sales per Square Foot
            if 'sales_per_sqft' in retail_metrics:
                ratios['operational']['sales_per_sqft'] = retail_metrics['sales_per_sqft']

            # E-commerce Percentage
            if 'e_commerce_percentage' in retail_metrics:
                ratios['operational']['e_commerce_percentage'] = retail_metrics['e_commerce_percentage']

            # E-commerce Growth
            if 'e_commerce_growth' in retail_metrics:
                ratios['operational']['e_commerce_growth'] = retail_metrics['e_commerce_growth']

            # Store Count and Growth
            if 'estimated_store_count' in retail_metrics:
                ratios['operational']['store_count'] = retail_metrics['estimated_store_count']

            if 'store_growth' in retail_metrics:
                ratios['operational']['store_growth'] = retail_metrics['store_growth']

            # 4. Retail-Specific Valuation Ratios

            # EV/Sales (particularly important for retail)
            market_data = financial_data.get('market_data', {})
            market_cap = market_data.get('market_cap')

            if market_cap and income_stmt is not None and balance_sheet is not None:
                if 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']

                    # Calculate Enterprise Value
                    total_debt = 0
                    if 'Total Debt' in balance_sheet.index:
                        total_debt = balance_sheet.iloc[0]['Total Debt']
                    else:
                        if 'Long Term Debt' in balance_sheet.index:
                            total_debt += balance_sheet.iloc[0]['Long Term Debt']
                        if 'Short Term Debt' in balance_sheet.index:
                            total_debt += balance_sheet.iloc[0]['Short Term Debt']

                    cash = 0
                    if 'Cash and Cash Equivalents' in balance_sheet.index:
                        cash = balance_sheet.iloc[0]['Cash and Cash Equivalents']
                    elif 'Cash and Short Term Investments' in balance_sheet.index:
                        cash = balance_sheet.iloc[0]['Cash and Short Term Investments']

                    enterprise_value = market_cap + total_debt - cash

                    if revenue > 0:
                        ev_sales = enterprise_value / revenue
                        ratios['valuation']['ev_sales'] = ev_sales

            # P/S (Price-to-Sales)
            if market_cap and income_stmt is not None:
                if 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']

                    if revenue > 0:
                        price_sales = market_cap / revenue
                        ratios['valuation']['price_sales'] = price_sales

            # EV/EBITDA
            if market_cap and income_stmt is not None and balance_sheet is not None:
                ebitda = 0
                if 'EBITDA' in income_stmt.index:
                    ebitda = income_stmt.iloc[0]['EBITDA']
                elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                    ebitda = income_stmt.iloc[0]['Operating Income'] + income_stmt.iloc[0][
                        'Depreciation & Amortization']

                if ebitda > 0:
                    # Use previously calculated enterprise_value
                    if 'enterprise_value' in locals():
                        ev_ebitda = enterprise_value / ebitda
                        ratios['valuation']['ev_ebitda'] = ev_ebitda

            # 5. Retail Sector Benchmarks

            # Add sector benchmarks based on subsector
            benchmarks = {
                "Traditional_Retail": {
                    "inventory_turnover": 4.0,
                    "gross_margin": 0.35,
                    "operating_margin": 0.05,
                    "same_store_sales_growth": 0.02,
                    "sales_per_sqft": 300,
                    "e_commerce_percentage": 0.10,
                    "cash_conversion_cycle": 60
                },
                "E_Commerce": {
                    "inventory_turnover": 8.0,
                    "gross_margin": 0.40,
                    "operating_margin": 0.07,
                    "same_store_sales_growth": None,  # Not applicable
                    "sales_per_sqft": None,  # Not applicable
                    "e_commerce_percentage": 0.95,
                    "cash_conversion_cycle": 30
                },
                "Omnichannel": {
                    "inventory_turnover": 6.0,
                    "gross_margin": 0.38,
                    "operating_margin": 0.06,
                    "same_store_sales_growth": 0.03,
                    "sales_per_sqft": 400,
                    "e_commerce_percentage": 0.30,
                    "cash_conversion_cycle": 45
                },
                "Discount": {
                    "inventory_turnover": 7.0,
                    "gross_margin": 0.25,
                    "operating_margin": 0.04,
                    "same_store_sales_growth": 0.03,
                    "sales_per_sqft": 250,
                    "e_commerce_percentage": 0.10,
                    "cash_conversion_cycle": 35
                },
                "Luxury": {
                    "inventory_turnover": 3.0,
                    "gross_margin": 0.60,
                    "operating_margin": 0.15,
                    "same_store_sales_growth": 0.04,
                    "sales_per_sqft": 1500,
                    "e_commerce_percentage": 0.15,
                    "cash_conversion_cycle": 90
                },
                "Specialty": {
                    "inventory_turnover": 5.0,
                    "gross_margin": 0.45,
                    "operating_margin": 0.08,
                    "same_store_sales_growth": 0.03,
                    "sales_per_sqft": 600,
                    "e_commerce_percentage": 0.20,
                    "cash_conversion_cycle": 55
                }
            }

            # Add benchmarks to return value
            if subsector in benchmarks:
                ratios['benchmarks'] = benchmarks[subsector]
            else:
                ratios['benchmarks'] = benchmarks["Omnichannel"]  # Default

            # 6. Calculate Score Relative to Benchmarks

            # For each key ratio, calculate a score (1-10) relative to benchmark
            if 'benchmarks' in ratios:
                scores = {}

                for category in ['efficiency', 'profitability', 'operational']:
                    for ratio_name, ratio_value in ratios[category].items():
                        if ratio_name in ratios['benchmarks'] and ratios['benchmarks'][ratio_name] is not None:
                            benchmark = ratios['benchmarks'][ratio_name]

                            # For ratios where higher is better
                            if ratio_name in ['inventory_turnover', 'gross_margin', 'operating_margin',
                                              'same_store_sales_growth', 'sales_per_sqft', 'e_commerce_percentage']:
                                if benchmark > 0:
                                    score = min(10, max(1, (ratio_value / benchmark) * 5))
                                else:
                                    score = 5  # Default if benchmark is zero

                            # For ratios where lower is better
                            elif ratio_name in ['cash_conversion_cycle', 'days_inventory_outstanding']:
                                if ratio_value > 0:
                                    score = min(10, max(1, (benchmark / ratio_value) * 5))
                                else:
                                    score = 10  # Perfect score if ratio is zero (unlikely)

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
            logger.error(f"Error calculating retail-specific ratios for {ticker}: {e}")
            return {
                'general': {},
                'efficiency': {},
                'profitability': {},
                'growth': {},
                'operational': {},
                'valuation': {},
                'error': str(e)
            }