import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SECTOR_METRICS, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.valuation.sector_factor import SectorAnalyzer, SectorFactory
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('energy_sector')


class EnergySectorAnalyzer(SectorAnalyzer):
    """
    Energy sector-specific analyzer.
    Implements analysis methods tailored for energy companies (oil, gas, renewable energy).
    """

    def __init__(self, sector_name: str = "Energy"):
        """Initialize with Energy sector parameters"""
        super().__init__(sector_name)
        self.data_loader = DataLoader()
        self.commodity_price_sensitivity = {
            "oil": 0.7,  # 70% of value for oil companies is tied to oil prices
            "gas": 0.6,  # 60% for gas companies
            "coal": 0.5,  # 50% for coal companies
            "renewable": 0.2  # Renewable energy companies are less commodity sensitive
        }
        logger.info("Energy sector analyzer initialized")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an energy company using sector-specific metrics

        Args:
            financial_data: Dictionary with company financial statements and information

        Returns:
            Dictionary with energy sector-specific analysis results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            # Basic validation
            if income_stmt is None or balance_sheet is None or cash_flow is None:
                logger.error("Missing required financial statements for energy sector analysis")
                return {"error": "Insufficient financial data for analysis"}

            # Get company subsector (oil, gas, renewable)
            subsector = self._determine_energy_subsector(company_info)

            # Calculate energy-specific metrics
            metrics = {}

            # 1. Reserve Replacement Ratio (RRR) - critical for oil & gas
            # Note: This is typically available in annual reports, estimated here
            metrics['reserve_replacement_ratio'] = self._estimate_reserve_replacement_ratio(
                financial_data, subsector)

            # 2. Production Cost per Barrel/Unit
            metrics['production_cost_per_unit'] = self._estimate_production_cost(
                financial_data, subsector)

            # 3. Reserve Life Index (RLI)
            metrics['reserve_life_index'] = self._estimate_reserve_life(
                financial_data, subsector)

            # 4. Price to Cash Flow (critical for energy companies)
            metrics['price_to_cash_flow'] = self._calculate_price_to_cash_flow(financial_data)

            # 5. EV/EBITDA - widely used in energy sector
            metrics['ev_to_ebitda'] = self._calculate_ev_to_ebitda(financial_data)

            # 6. EV/2P Reserves (for oil & gas companies)
            if subsector in ['oil', 'gas']:
                metrics['ev_to_reserves'] = self._estimate_ev_to_reserves(financial_data)

            # 7. CAPEX Intensity
            metrics['capex_intensity'] = self._calculate_capex_intensity(financial_data)

            # 8. Price to Tangible Book Value
            metrics['price_to_tangible_book'] = self._calculate_price_to_tangible_book(financial_data)

            # 9. Debt-adjusted cash flow multiple
            metrics['debt_adjusted_cash_flow_multiple'] = self._calculate_debt_adjusted_cf_multiple(financial_data)

            # 10. Commodity Price Sensitivity
            metrics['commodity_price_sensitivity'] = self.commodity_price_sensitivity.get(subsector, 0.5)

            # 11. Free Cash Flow Yield
            metrics['fcf_yield'] = self._calculate_fcf_yield(financial_data)

            # Analyze operational efficiency
            operational_metrics = self._analyze_operational_efficiency(financial_data, subsector)
            metrics.update(operational_metrics)

            # Risk assessment specific to energy sector
            risk_assessment = self._assess_energy_risks(financial_data, subsector)

            return {
                "sector": "Energy",
                "subsector": subsector,
                "metrics": metrics,
                "risk_assessment": risk_assessment,
                "summary": self._generate_analysis_summary(metrics, risk_assessment, subsector)
            }

        except Exception as e:
            logger.error(f"Error analyzing energy company: {e}")
            return {"error": str(e)}

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate valuation for an energy company using sector-specific methods

        Args:
            financial_data: Dictionary with company financial information
            valuation_method: Specific valuation method to use (if None, use most appropriate)

        Returns:
            Dictionary with valuation results
        """
        try:
            company_info = financial_data.get('company_info', {})
            subsector = self._determine_energy_subsector(company_info)

            # If method not specified, determine best method based on subsector
            if valuation_method is None:
                if subsector in ['oil', 'gas']:
                    # Resource companies often valued using NAV or EV/Reserve methods
                    valuation_method = 'nav'
                elif subsector == 'renewable':
                    # Renewable companies often valued using DCF
                    valuation_method = 'dcf'
                else:
                    # Default for integrated energy companies
                    valuation_method = 'multiple'

            valuation_result = {}

            # Implement different valuation methods
            if valuation_method.lower() == 'nav':
                valuation_result = self._calculate_nav_valuation(financial_data, subsector)
            elif valuation_method.lower() == 'dcf':
                valuation_result = self._calculate_dcf_valuation(financial_data, subsector)
            elif valuation_method.lower() == 'multiple':
                valuation_result = self._calculate_multiple_valuation(financial_data, subsector)
            elif valuation_method.lower() == 'reserve':
                valuation_result = self._calculate_reserve_valuation(financial_data, subsector)
            else:
                logger.warning(f"Unknown valuation method {valuation_method}, using DCF")
                valuation_result = self._calculate_dcf_valuation(financial_data, subsector)

            # Add commodity price sensitivity scenario analysis
            price_scenarios = self._run_commodity_price_scenarios(financial_data, subsector)
            valuation_result['price_sensitivity'] = price_scenarios

            return valuation_result

        except Exception as e:
            logger.error(f"Error calculating energy sector valuation: {e}")
            return {"error": str(e)}

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with energy sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results and analysis
        """
        try:
            benchmarks = self.get_sector_benchmarks()

            if not benchmarks:
                logger.warning("No energy sector benchmarks available")
                return {"warning": "No sector benchmarks available for comparison"}

            comparison_results = {}

            # Compare each metric with benchmark
            for metric, benchmark in benchmarks.items():
                if metric in company_metrics:
                    actual_value = company_metrics[metric]

                    # Determine if higher or lower is better
                    higher_is_better = True
                    if metric in ['Debt/EBITDA', 'production_cost_per_unit']:
                        higher_is_better = False

                    # Calculate performance against benchmark
                    if higher_is_better:
                        performance = (actual_value / benchmark) * 100 if benchmark else 0
                    else:
                        performance = (benchmark / actual_value) * 100 if actual_value else 0

                    # Interpret result
                    if higher_is_better:
                        if performance >= 120:
                            interpretation = "Significantly outperforms sector"
                        elif performance >= 100:
                            interpretation = "Outperforms sector"
                        elif performance >= 80:
                            interpretation = "Slightly below sector average"
                        else:
                            interpretation = "Significantly underperforms sector"
                    else:
                        if performance >= 120:
                            interpretation = "Significantly better than sector (lower is better)"
                        elif performance >= 100:
                            interpretation = "Better than sector average (lower is better)"
                        elif performance >= 80:
                            interpretation = "Slightly worse than sector average (lower is better)"
                        else:
                            interpretation = "Significantly worse than sector average (lower is better)"

                    comparison_results[metric] = {
                        "actual": actual_value,
                        "benchmark": benchmark,
                        "performance": performance,
                        "interpretation": interpretation
                    }

            # Calculate overall score
            overall_score = self.calculate_ratio_score(company_metrics)

            return {
                "benchmark_comparison": comparison_results,
                "overall_score": overall_score,
                "summary": self._generate_benchmark_summary(comparison_results, overall_score)
            }

        except Exception as e:
            logger.error(f"Error benchmarking energy company: {e}")
            return {"error": str(e)}

    # Helper methods specific to Energy sector

    def _determine_energy_subsector(self, company_info: Dict[str, Any]) -> str:
        """Determine energy subsector (oil, gas, renewable, etc.)"""
        industry = company_info.get('industry', '').lower()
        description = company_info.get('description', '').lower()

        if 'renewable' in industry or 'renewable' in description or 'solar' in industry or 'wind' in industry:
            return 'renewable'
        elif 'gas' in industry and 'oil' not in industry:
            return 'gas'
        elif 'coal' in industry:
            return 'coal'
        else:
            return 'oil'  # Default to oil for integrated companies

    def _estimate_reserve_replacement_ratio(self, financial_data: Dict[str, Any], subsector: str) -> float:
        """
        Estimate Reserve Replacement Ratio (RRR)
        RRR = New Reserves Added / Production

        Note: This is usually found in annual reports, here we estimate from financials
        """
        if subsector == 'renewable':
            return None  # Not applicable for renewable energy

        # In real implementation, this would use actual reserve data
        # For estimation purposes, we use exploration costs as a proxy for new reserves
        try:
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')

            if income_stmt is None or cash_flow is None:
                return None

            # Exploration costs (proxy for reserve additions)
            exploration_costs = 0

            # Look for exploration or development costs
            if 'Exploration Expense' in income_stmt.index:
                exploration_costs = income_stmt.iloc[0]['Exploration Expense']
            elif 'Capital Expenditure' in cash_flow.index:
                # Assume 40% of CAPEX goes to exploration for oil & gas companies
                exploration_costs = cash_flow.iloc[0]['Capital Expenditure'] * 0.4

            # Try to find production revenue as proxy for production volume
            production_revenue = 0
            if 'Total Revenue' in income_stmt.index:
                production_revenue = income_stmt.iloc[0]['Total Revenue']

            if production_revenue > 0 and exploration_costs != 0:
                # Crude estimate - real RRR would use reserve and production volumes
                estimated_rrr = (exploration_costs / production_revenue) * 3.5
                return max(0, min(2.0, estimated_rrr))  # Cap between 0 and 200%

            return None
        except Exception as e:
            logger.error(f"Error estimating reserve replacement ratio: {e}")
            return None

    def _estimate_production_cost(self, financial_data: Dict[str, Any], subsector: str) -> float:
        """
        Estimate production cost per barrel/unit

        Note: This would ideally come from company reports with actual production volumes
        """
        try:
            income_stmt = financial_data.get('income_statement')

            if income_stmt is None:
                return None

            # Cost of goods sold as proxy for production costs
            if 'Cost of Revenue' in income_stmt.index:
                production_costs = income_stmt.iloc[0]['Cost of Revenue']

                # Revenue as proxy for production volume
                if 'Total Revenue' in income_stmt.index:
                    production_revenue = income_stmt.iloc[0]['Total Revenue']

                    if production_revenue > 0:
                        # Crude estimate - real metric would use actual production volumes
                        cost_ratio = production_costs / production_revenue

                        # Convert ratio to estimated $ per barrel/unit based on subsector
                        if subsector == 'oil':
                            # Oil prices historically ~$60/barrel, so scale based on this
                            return cost_ratio * 60
                        elif subsector == 'gas':
                            # Natural gas prices vary, use approximate scaling
                            return cost_ratio * 3.5
                        elif subsector == 'coal':
                            return cost_ratio * 50
                        elif subsector == 'renewable':
                            # For renewable, return cost per MWh estimate
                            return cost_ratio * 100

            return None
        except Exception as e:
            logger.error(f"Error estimating production cost: {e}")
            return None

    def _estimate_reserve_life(self, financial_data: Dict[str, Any], subsector: str) -> float:
        """
        Estimate Reserve Life Index (RLI) - years of reserves at current production rate

        Note: This would ideally come from company reports with proven reserves data
        """
        if subsector == 'renewable':
            return None  # Not applicable for renewable energy

        try:
            # For a rough estimate, we can use the relationship between
            # property/plant/equipment (PPE) assets and annual production
            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')

            if balance_sheet is None or income_stmt is None:
                return None

            # Use PPE as a proxy for reserves
            ppe = None
            if 'Property Plant and Equipment' in balance_sheet.index:
                ppe = balance_sheet.iloc[0]['Property Plant and Equipment']

            # Use Cost of Revenue as a proxy for annual production value
            annual_production = None
            if 'Cost of Revenue' in income_stmt.index:
                annual_production = income_stmt.iloc[0]['Cost of Revenue']

            if ppe is not None and annual_production and annual_production > 0:
                # Crude estimate
                estimated_rli = (ppe / annual_production) * 0.8

                # Typical reserve life varies by subsector
                if subsector == 'oil':
                    # Oil companies typically have 8-15 years of reserves
                    return max(5, min(20, estimated_rli))
                elif subsector == 'gas':
                    # Gas companies often have longer reserve lives
                    return max(8, min(25, estimated_rli * 1.2))
                elif subsector == 'coal':
                    # Coal companies often have very long reserve lives
                    return max(15, min(40, estimated_rli * 1.5))

                return estimated_rli

            return None
        except Exception as e:
            logger.error(f"Error estimating reserve life: {e}")
            return None

    def _calculate_price_to_cash_flow(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Price to Cash Flow ratio"""
        try:
            cash_flow = financial_data.get('cash_flow')
            market_data = financial_data.get('market_data', {})

            if cash_flow is None or not market_data:
                return None

            # Get market cap
            market_cap = market_data.get('market_cap')

            # Get operating cash flow
            operating_cf = None
            if 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.iloc[0]['Operating Cash Flow']

            if market_cap and operating_cf and operating_cf > 0:
                return market_cap / operating_cf

            return None
        except Exception as e:
            logger.error(f"Error calculating price to cash flow: {e}")
            return None

    def _calculate_ev_to_ebitda(self, financial_data: Dict[str, Any]) -> float:
        """Calculate EV/EBITDA ratio"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None or not market_data:
                return None

            # Calculate Enterprise Value
            market_cap = market_data.get('market_cap')

            # Get debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            enterprise_value = market_cap + total_debt - cash_and_equivalents

            # Calculate EBITDA
            ebitda = None
            if 'EBITDA' in income_stmt.index:
                ebitda = income_stmt.iloc[0]['EBITDA']
            elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                operating_income = income_stmt.iloc[0]['Operating Income']
                depreciation = income_stmt.iloc[0]['Depreciation & Amortization']
                ebitda = operating_income + depreciation

            if enterprise_value > 0 and ebitda and ebitda > 0:
                return enterprise_value / ebitda

            return None
        except Exception as e:
            logger.error(f"Error calculating EV/EBITDA: {e}")
            return None

    def _estimate_ev_to_reserves(self, financial_data: Dict[str, Any]) -> float:
        """
        Estimate EV to proven reserves ratio (EV/2P)

        Note: This would ideally use actual reserve data from company reports
        """
        try:
            # Since we don't have actual reserve data, we'll use PPE as a proxy
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if balance_sheet is None or not market_data:
                return None

            # Calculate Enterprise Value
            market_cap = market_data.get('market_cap')

            # Get debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            enterprise_value = market_cap + total_debt - cash_and_equivalents

            # Use PPE as a proxy for reserves
            ppe = None
            if 'Property Plant and Equipment' in balance_sheet.index:
                ppe = balance_sheet.iloc[0]['Property Plant and Equipment']

            if enterprise_value > 0 and ppe and ppe > 0:
                # Apply a factor to convert PPE to an estimate of proven reserves value
                estimated_reserve_value = ppe * 1.2
                return enterprise_value / estimated_reserve_value

            return None
        except Exception as e:
            logger.error(f"Error estimating EV to reserves: {e}")
            return None

    def _calculate_capex_intensity(self, financial_data: Dict[str, Any]) -> float:
        """Calculate CAPEX intensity (CAPEX / Revenue)"""
        try:
            cash_flow = financial_data.get('cash_flow')
            income_stmt = financial_data.get('income_statement')

            if cash_flow is None or income_stmt is None:
                return None

            # Get CAPEX
            capex = None
            if 'Capital Expenditure' in cash_flow.index:
                capex = abs(cash_flow.iloc[0]['Capital Expenditure'])

            # Get Revenue
            revenue = None
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.iloc[0]['Total Revenue']

            if capex is not None and revenue and revenue > 0:
                return capex / revenue

            return None
        except Exception as e:
            logger.error(f"Error calculating CAPEX intensity: {e}")
            return None

    def _calculate_price_to_tangible_book(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Price to Tangible Book Value"""
        try:
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if balance_sheet is None or not market_data:
                return None

            # Get market cap
            market_cap = market_data.get('market_cap')

            # Calculate tangible book value (equity - intangibles)
            tangible_book_value = None

            if 'Total Stockholder Equity' in balance_sheet.index:
                equity = balance_sheet.iloc[0]['Total Stockholder Equity']

                # Subtract intangibles
                intangibles = 0
                if 'Goodwill' in balance_sheet.index:
                    intangibles += balance_sheet.iloc[0]['Goodwill']

                if 'Intangible Assets' in balance_sheet.index:
                    intangibles += balance_sheet.iloc[0]['Intangible Assets']

                tangible_book_value = equity - intangibles

            if market_cap and tangible_book_value and tangible_book_value > 0:
                return market_cap / tangible_book_value

            return None
        except Exception as e:
            logger.error(f"Error calculating price to tangible book: {e}")
            return None

    def _calculate_debt_adjusted_cash_flow_multiple(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate Debt-Adjusted Cash Flow Multiple
        (Enterprise Value / (Operating Cash Flow + Interest Expense))
        """
        try:
            cash_flow = financial_data.get('cash_flow')
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if cash_flow is None or income_stmt is None or balance_sheet is None or not market_data:
                return None

            # Calculate Enterprise Value
            market_cap = market_data.get('market_cap')

            # Get debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            enterprise_value = market_cap + total_debt - cash_and_equivalents

            # Get operating cash flow
            operating_cf = None
            if 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.iloc[0]['Operating Cash Flow']

            # Get interest expense
            interest_expense = 0
            if 'Interest Expense' in income_stmt.index:
                interest_expense = abs(income_stmt.iloc[0]['Interest Expense'])

            if enterprise_value > 0 and operating_cf and (operating_cf + interest_expense) > 0:
                return enterprise_value / (operating_cf + interest_expense)

            return None
        except Exception as e:
            logger.error(f"Error calculating debt-adjusted cash flow multiple: {e}")
            return None

    def _calculate_debt_adjusted_cf_multiple(self, financial_data: Dict[str, Any]) -> float:
        """Calculate (Market Cap + Total Debt) / Operating Cash Flow"""
        try:
            cash_flow = financial_data.get('cash_flow')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if cash_flow is None or balance_sheet is None or not market_data:
                return None

            market_cap = market_data.get('market_cap')
            if not market_cap or 'Operating Cash Flow' not in cash_flow.index:
                return None

            operating_cf = cash_flow.iloc[0]['Operating Cash Flow']
            if not operating_cf or operating_cf <= 0:
                return None

            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            return (market_cap + total_debt) / operating_cf
        except Exception as e:
            logger.error(f"Error calculating debt-adjusted cash flow multiple: {e}")
            return None

    def _calculate_fcf_yield(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Free Cash Flow Yield (FCF / Enterprise Value)"""
        try:
            cash_flow = financial_data.get('cash_flow')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if cash_flow is None or balance_sheet is None or not market_data:
                return None

            # Calculate Enterprise Value
            market_cap = market_data.get('market_cap')

            # Get debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            enterprise_value = market_cap + total_debt - cash_and_equivalents

            # Get Free Cash Flow
            fcf = None
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.iloc[0]['Free Cash Flow']
            elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                operating_cf = cash_flow.iloc[0]['Operating Cash Flow']
                capex = cash_flow.iloc[0]['Capital Expenditure']
                fcf = operating_cf + capex  # CAPEX is negative in cash flow statements

            if enterprise_value > 0 and fcf:
                return fcf / enterprise_value

            return None
        except Exception as e:
            logger.error(f"Error calculating FCF yield: {e}")
            return None

    def _analyze_operational_efficiency(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, float]:
        """Analyze operational efficiency metrics specific to energy companies"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None:
                return {}

            metrics = {}

            # EBITDA Margin
            if 'EBITDA' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                ebitda = income_stmt.iloc[0]['EBITDA']
                revenue = income_stmt.iloc[0]['Total Revenue']

                if revenue > 0:
                    metrics['ebitda_margin'] = ebitda / revenue

            # Return on Capital Employed (ROCE) - important for capital-intensive industries
            if 'Operating Income' in income_stmt.index:
                operating_income = income_stmt.iloc[0]['Operating Income']

                # Capital Employed = Total Assets - Current Liabilities
                capital_employed = None
                if 'Total Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                    total_assets = balance_sheet.iloc[0]['Total Assets']
                    current_liabilities = balance_sheet.iloc[0]['Total Current Liabilities']
                    capital_employed = total_assets - current_liabilities

                    if capital_employed > 0:
                        metrics['roce'] = operating_income / capital_employed

            # Asset Turnover
            if 'Total Revenue' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                revenue = income_stmt.iloc[0]['Total Revenue']
                assets = balance_sheet.iloc[0]['Total Assets']

                if assets > 0:
                    metrics['asset_turnover'] = revenue / assets

            # Operating Margin
            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.iloc[0]['Operating Income']
                revenue = income_stmt.iloc[0]['Total Revenue']

                if revenue > 0:
                    metrics['operating_margin'] = operating_income / revenue

            return metrics
        except Exception as e:
            logger.error(f"Error analyzing operational efficiency: {e}")
            return {}

    def _assess_energy_risks(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Assess energy sector-specific risks"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None:
                return {}

            risks = {}

            # 1. Commodity Price Risk
            risks['commodity_price_risk'] = {
                'level': self.commodity_price_sensitivity.get(subsector, 0.5),
                'description': f"Sensitivity to commodity prices: {self._interpret_risk_level(self.commodity_price_sensitivity.get(subsector, 0.5))}"
            }

            # 2. Debt Risk
            debt_to_ebitda = None
            if 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
                debt = balance_sheet.iloc[0]['Total Debt']
                ebitda = income_stmt.iloc[0]['EBITDA']

                if ebitda > 0:
                    debt_to_ebitda = debt / ebitda
                    debt_risk_level = 0.3  # Low
                    if debt_to_ebitda > 3.0:
                        debt_risk_level = 0.6  # Medium
                    if debt_to_ebitda > 4.0:
                        debt_risk_level = 0.8  # High

                    risks['debt_risk'] = {
                        'level': debt_risk_level,
                        'value': debt_to_ebitda,
                        'description': f"Debt/EBITDA ratio: {debt_to_ebitda:.2f}. {self._interpret_risk_level(debt_risk_level)}"
                    }

            # 3. Operational Risk (based on operating margin)
            op_risk_level = 0.4  # Default medium-low
            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.iloc[0]['Operating Income']
                revenue = income_stmt.iloc[0]['Total Revenue']

                if revenue > 0:
                    operating_margin = operating_income / revenue
                    if operating_margin < 0.05:
                        op_risk_level = 0.8  # High
                    elif operating_margin < 0.15:
                        op_risk_level = 0.6  # Medium
                    elif operating_margin > 0.25:
                        op_risk_level = 0.2  # Low

                    risks['operational_risk'] = {
                        'level': op_risk_level,
                        'value': operating_margin,
                        'description': f"Operating margin: {operating_margin:.2%}. {self._interpret_risk_level(op_risk_level)}"
                    }

            # 4. Reserve Risk (specific to oil & gas)
            if subsector in ['oil', 'gas']:
                reserve_risk_level = 0.5  # Default medium

                # If we have RRR, adjust risk level
                rrr = self._estimate_reserve_replacement_ratio(financial_data, subsector)
                if rrr:
                    if rrr < 0.7:
                        reserve_risk_level = 0.9  # Very high
                    elif rrr < 1.0:
                        reserve_risk_level = 0.7  # High
                    elif rrr > 1.2:
                        reserve_risk_level = 0.3  # Low

                # If we have reserve life, adjust risk level
                rli = self._estimate_reserve_life(financial_data, subsector)
                if rli:
                    if rli < 5:
                        reserve_risk_level = max(reserve_risk_level, 0.8)  # High
                    elif rli > 15:
                        reserve_risk_level = min(reserve_risk_level, 0.4)  # Low

                risks['reserve_risk'] = {
                    'level': reserve_risk_level,
                    'description': f"Reserve risk: {self._interpret_risk_level(reserve_risk_level)}"
                }

            # 5. Regulatory Risk (especially high for coal, medium for oil & gas, lower for renewables)
            regulatory_risk_level = 0.5  # Default medium
            if subsector == 'coal':
                regulatory_risk_level = 0.9  # Very high
            elif subsector == 'oil':
                regulatory_risk_level = 0.7  # High
            elif subsector == 'gas':
                regulatory_risk_level = 0.6  # Medium-high
            elif subsector == 'renewable':
                regulatory_risk_level = 0.3  # Low

            risks['regulatory_risk'] = {
                'level': regulatory_risk_level,
                'description': f"Regulatory risk: {self._interpret_risk_level(regulatory_risk_level)}"
            }

            # Calculate overall risk level
            risk_factors = [risk_info['level'] for risk_info in risks.values() if 'level' in risk_info]
            overall_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.5

            risks['overall_risk'] = {
                'level': overall_risk,
                'description': f"Overall risk: {self._interpret_risk_level(overall_risk)}"
            }

            return risks
        except Exception as e:
            logger.error(f"Error assessing energy risks: {e}")
            return {}

    def _interpret_risk_level(self, risk_level: float) -> str:
        """Interpret a risk level value"""
        if risk_level >= 0.8:
            return "Very High Risk"
        elif risk_level >= 0.6:
            return "High Risk"
        elif risk_level >= 0.4:
            return "Medium Risk"
        elif risk_level >= 0.2:
            return "Low Risk"
        else:
            return "Very Low Risk"

    def _calculate_nav_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """
        Calculate Net Asset Value (NAV) valuation for energy company
        This method is particularly relevant for resource companies
        """
        try:
            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')
            market_data = financial_data.get('market_data', {})

            if balance_sheet is None or income_stmt is None:
                return {"error": "Missing required data for NAV valuation"}

            # Get basic accounting values
            total_assets = None
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.iloc[0]['Total Assets']

            total_liabilities = None
            if 'Total Liabilities' in balance_sheet.index:
                total_liabilities = balance_sheet.iloc[0]['Total Liabilities']

            book_value = None
            if total_assets is not None and total_liabilities is not None:
                book_value = total_assets - total_liabilities

            # For resource companies, adjust book value based on reserves
            adjusted_nav = book_value

            # Apply sector-specific adjustments
            if subsector == 'oil' or subsector == 'gas':
                # Oil & gas reserves are often undervalued on balance sheet
                # Adjust based on reserve life and production costs
                reserve_life = self._estimate_reserve_life(financial_data, subsector)
                production_cost = self._estimate_production_cost(financial_data, subsector)

                if 'Property Plant and Equipment' in balance_sheet.index:
                    ppe = balance_sheet.iloc[0]['Property Plant and Equipment']

                    # Apply a premium to PPE based on reserve quality
                    ppe_adjustment = 1.0  # Default no adjustment

                    if reserve_life and production_cost:
                        # Higher reserve life and lower production cost = better quality reserves
                        if reserve_life > 15 and production_cost < 40:
                            ppe_adjustment = 1.5  # 50% premium
                        elif reserve_life > 10 and production_cost < 50:
                            ppe_adjustment = 1.3  # 30% premium
                        elif reserve_life < 5 or production_cost > 60:
                            ppe_adjustment = 0.8  # 20% discount

                    # Apply the adjustment
                    adjusted_ppe = ppe * ppe_adjustment

                    # Adjust NAV with the PPE difference
                    adjusted_nav = book_value + (adjusted_ppe - ppe)

            elif subsector == 'renewable':
                # Renewable assets often have long-term contracts that aren't fully reflected
                # in book value, so apply a premium
                adjusted_nav = book_value * 1.2  # 20% premium for renewable

            elif subsector == 'coal':
                # Coal assets often face regulatory risks and declining demand
                # so apply a discount
                adjusted_nav = book_value * 0.8  # 20% discount for coal

            # Calculate per share values
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding and shares_outstanding > 0:
                book_value_per_share = book_value / shares_outstanding if book_value else None
                adjusted_nav_per_share = adjusted_nav / shares_outstanding if adjusted_nav else None
            else:
                book_value_per_share = None
                adjusted_nav_per_share = None

            # Get current market price for comparison
            current_price = market_data.get('share_price')

            # Calculate discount/premium to NAV
            nav_discount = None
            if current_price and adjusted_nav_per_share and adjusted_nav_per_share > 0:
                nav_discount = (current_price / adjusted_nav_per_share - 1) * 100

            return {
                "valuation_method": "Net Asset Value (NAV)",
                "book_value": book_value,
                "book_value_per_share": book_value_per_share,
                "adjusted_nav": adjusted_nav,
                "adjusted_nav_per_share": adjusted_nav_per_share,
                "current_price": current_price,
                "nav_discount_premium": nav_discount,
                "interpretation": self._interpret_nav_valuation(nav_discount, subsector)
            }

        except Exception as e:
            logger.error(f"Error calculating NAV valuation: {e}")
            return {"error": str(e)}

    def _interpret_nav_valuation(self, nav_discount: Optional[float], subsector: str) -> str:
        """Interpret NAV valuation results"""
        if nav_discount is None:
            return "Unable to determine NAV discount/premium."

        if subsector == 'oil' or subsector == 'gas':
            # Oil & gas companies typically trade at a discount to NAV
            if nav_discount < -30:
                return "Trading at a significant discount to NAV, potentially undervalued if reserve estimates are accurate."
            elif nav_discount < -10:
                return "Trading at a moderate discount to NAV, potentially fairly valued given typical sector discounts."
            elif nav_discount < 10:
                return "Trading near NAV, suggesting fair valuation."
            else:
                return "Trading at a premium to NAV, potentially overvalued unless growth prospects are exceptional."

        elif subsector == 'renewable':
            # Renewable companies often trade at a premium due to growth
            if nav_discount < 0:
                return "Trading below NAV, potentially undervalued for a renewable energy company."
            elif nav_discount < 30:
                return "Trading at a modest premium to NAV, potentially fairly valued given growth prospects."
            else:
                return "Trading at a significant premium to NAV, valuation depends on long-term growth realization."

        elif subsector == 'coal':
            # Coal companies often trade at larger discounts due to regulatory risks
            if nav_discount < -50:
                return "Trading at a very large discount to NAV, reflecting significant regulatory and transition risks."
            elif nav_discount < -30:
                return "Trading at a typical discount to NAV for coal companies, potentially fairly valued."
            else:
                return "Trading at a smaller discount than typical for coal companies, potentially overvalued unless company has unique advantages."

        else:
            # Generic interpretation
            if nav_discount < -20:
                return "Trading at a significant discount to NAV, potentially undervalued."
            elif nav_discount < 0:
                return "Trading at a modest discount to NAV, potentially fairly valued."
            elif nav_discount < 20:
                return "Trading at a premium to NAV, valuation depends on growth prospects."
            else:
                return "Trading at a significant premium to NAV, potentially overvalued unless growth prospects are exceptional."

    def _calculate_dcf_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """
        Calculate DCF valuation for energy company
        Incorporating sector-specific adjustments to the DCF model
        """
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                return {"error": "Missing required data for DCF valuation"}

            # Get DCF parameters based on subsector
            if subsector == 'oil' or subsector == 'gas':
                forecast_years = 7
                terminal_growth = 0.01  # 1% terminal growth for oil & gas (limited long-term growth)
                discount_rate = 0.10  # 10% discount rate
            elif subsector == 'coal':
                forecast_years = 5
                terminal_growth = 0.00  # 0% terminal growth for coal (declining industry)
                discount_rate = 0.12  # 12% discount rate due to higher risk
            elif subsector == 'renewable':
                forecast_years = 10
                terminal_growth = 0.03  # 3% terminal growth for renewable (higher long-term growth)
                discount_rate = 0.09  # 9% discount rate
            else:
                forecast_years = 7
                terminal_growth = 0.02  # 2% default terminal growth
                discount_rate = 0.10  # 10% default discount rate

            # Get initial free cash flow
            initial_fcf = None
            if 'Free Cash Flow' in cash_flow.index:
                initial_fcf = cash_flow.iloc[0]['Free Cash Flow']
            elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                operating_cf = cash_flow.iloc[0]['Operating Cash Flow']
                capex = cash_flow.iloc[0]['Capital Expenditure']
                initial_fcf = operating_cf + capex  # CAPEX is negative in cash flow statements

            if initial_fcf is None:
                return {"error": "Unable to determine initial free cash flow"}

            # Energy companies often have cyclical FCF, so look at historical average if available
            if cash_flow.shape[1] > 1:
                historical_fcf = []
                for year in range(min(3, cash_flow.shape[1])):
                    if 'Free Cash Flow' in cash_flow.index:
                        fcf = cash_flow.iloc[:, year]['Free Cash Flow']
                        historical_fcf.append(fcf)
                    elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                        ocf = cash_flow.iloc[:, year]['Operating Cash Flow']
                        capex = cash_flow.iloc[:, year]['Capital Expenditure']
                        fcf = ocf + capex
                        historical_fcf.append(fcf)

                if historical_fcf:
                    # Use average FCF as a base for cyclical industries
                    initial_fcf = sum(historical_fcf) / len(historical_fcf)

            # Estimate growth rate based on historical performance and subsector
            growth_rate = 0.03  # Default growth

            # Adjust based on subsector outlook
            if subsector == 'oil' or subsector == 'gas':
                growth_rate = 0.02  # Lower growth for oil & gas
            elif subsector == 'coal':
                growth_rate = 0.00  # No growth or decline for coal
            elif subsector == 'renewable':
                growth_rate = 0.10  # Higher growth for renewable

            # Try to refine growth rate from historical data if available
            if income_stmt.shape[1] > 1:
                revenue_growth = []
                for year in range(income_stmt.shape[1] - 1):
                    if 'Total Revenue' in income_stmt.index:
                        current_rev = income_stmt.iloc[:, year]['Total Revenue']
                        prev_rev = income_stmt.iloc[:, year + 1]['Total Revenue']
                        if prev_rev > 0:
                            annual_growth = (current_rev / prev_rev) - 1
                            revenue_growth.append(annual_growth)

                if revenue_growth:
                    # Use weighted average of historical growth and subsector assumption
                    historical_growth = sum(revenue_growth) / len(revenue_growth)
                    growth_rate = (historical_growth * 0.6) + (growth_rate * 0.4)

            # Cap growth rate based on subsector
            if subsector == 'oil' or subsector == 'gas':
                growth_rate = min(growth_rate, 0.05)  # Max 5% for oil & gas
            elif subsector == 'coal':
                growth_rate = min(growth_rate, 0.02)  # Max 2% for coal
            elif subsector == 'renewable':
                growth_rate = min(growth_rate, 0.15)  # Max 15% for renewable

            # Project future cash flows
            projected_fcf = []
            for year in range(1, forecast_years + 1):
                if year <= 3:
                    # Use initial growth rate for first 3 years
                    fcf = initial_fcf * (1 + growth_rate) ** year
                else:
                    # Gradually transition to terminal growth for later years
                    transition_factor = (year - 3) / (forecast_years - 3)
                    effective_growth = growth_rate * (1 - transition_factor) + terminal_growth * transition_factor
                    fcf = initial_fcf * (1 + growth_rate) ** 3 * (1 + effective_growth) ** (year - 3)
                projected_fcf.append(fcf)

            # Calculate terminal value
            final_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = final_fcf / (discount_rate - terminal_growth)

            # Calculate present values
            present_value_fcf = sum(fcf / (1 + discount_rate) ** year
                                    for year, fcf in enumerate(projected_fcf, 1))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_fcf + present_value_terminal

            # Calculate equity value by adjusting for debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            equity_value = enterprise_value - total_debt + cash_and_equivalents

            # Calculate per share value
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            # Get current market price for comparison
            current_price = market_data.get('share_price')

            # Calculate upside/downside potential
            upside_potential = None
            if current_price and value_per_share and current_price > 0:
                upside_potential = (value_per_share / current_price - 1) * 100

            return {
                "valuation_method": "Discounted Cash Flow (DCF)",
                "initial_fcf": initial_fcf,
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "forecast_years": forecast_years,
                "projected_fcf": projected_fcf,
                "terminal_value": terminal_value,
                "present_value_fcf": present_value_fcf,
                "present_value_terminal": present_value_terminal,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "value_per_share": value_per_share,
                "current_price": current_price,
                "upside_potential": upside_potential,
                "interpretation": self._interpret_dcf_valuation(upside_potential, subsector)
            }

        except Exception as e:
            logger.error(f"Error calculating DCF valuation: {e}")
            return {"error": str(e)}

    def _interpret_dcf_valuation(self, upside_potential: Optional[float], subsector: str) -> str:
        """Interpret DCF valuation results"""
        if upside_potential is None:
            return "Unable to determine upside potential."

        if upside_potential > 50:
            return f"Significant upside potential of {upside_potential:.1f}%. Stock appears substantially undervalued based on DCF analysis."
        elif upside_potential > 20:
            return f"Moderate upside potential of {upside_potential:.1f}%. Stock appears undervalued based on DCF analysis."
        elif upside_potential > -10:
            return f"Limited upside/downside of {upside_potential:.1f}%. Stock appears fairly valued based on DCF analysis."
        elif upside_potential > -30:
            return f"Moderate downside risk of {-upside_potential:.1f}%. Stock appears overvalued based on DCF analysis."
        else:
            return f"Significant downside risk of {-upside_potential:.1f}%. Stock appears substantially overvalued based on DCF analysis."

    def _calculate_multiple_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """
        Calculate valuation based on sector-specific multiples
        Different multiples are important for different energy subsectors
        """
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                return {"error": "Missing required data for multiple valuation"}

            valuations = {}
            current_price = market_data.get('share_price')
            shares_outstanding = market_data.get('shares_outstanding')

            # Determine which multiples to use based on subsector
            multiples_to_use = []

            if subsector == 'oil' or subsector == 'gas':
                # Oil & gas companies are commonly valued on EV/EBITDA and P/CF
                multiples_to_use = ['ev_ebitda', 'p_cf', 'p_bv']
            elif subsector == 'coal':
                # Coal companies often valued on P/B and EV/EBITDA
                multiples_to_use = ['p_bv', 'ev_ebitda']
            elif subsector == 'renewable':
                # Renewable often valued on EV/EBITDA and revenue multiples
                multiples_to_use = ['ev_ebitda', 'ev_revenue']
            else:
                # Default for integrated energy
                multiples_to_use = ['ev_ebitda', 'p_cf', 'p_bv']

            # Get sector benchmark multiples
            benchmarks = self.get_sector_benchmarks()

            # EV/EBITDA Valuation
            if 'ev_ebitda' in multiples_to_use:
                ebitda = None
                if 'EBITDA' in income_stmt.index:
                    ebitda = income_stmt.iloc[0]['EBITDA']
                elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                    operating_income = income_stmt.iloc[0]['Operating Income']
                    depreciation = income_stmt.iloc[0]['Depreciation & Amortization']
                    ebitda = operating_income + depreciation

                if ebitda and ebitda > 0:
                    # Get total debt and cash for EV calculation
                    total_debt = 0
                    if 'Total Debt' in balance_sheet.index:
                        total_debt = balance_sheet.iloc[0]['Total Debt']

                    cash_and_equivalents = 0
                    if 'Cash and Cash Equivalents' in balance_sheet.index:
                        cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

                    # Get benchmark multiple
                    benchmark_multiple = benchmarks.get('EV/EBITDA', 8.0)

                    # Calculate enterprise value
                    enterprise_value = ebitda * benchmark_multiple

                    # Calculate equity value
                    equity_value = enterprise_value - total_debt + cash_and_equivalents

                    # Calculate per share value
                    if shares_outstanding and shares_outstanding > 0:
                        value_per_share = equity_value / shares_outstanding

                        valuations['ev_ebitda'] = {
                            'value_per_share': value_per_share,
                            'multiple_used': benchmark_multiple,
                            'equity_value': equity_value
                        }

            # P/CF Valuation
            if 'p_cf' in multiples_to_use:
                operating_cf = None
                if 'Operating Cash Flow' in cash_flow.index:
                    operating_cf = cash_flow.iloc[0]['Operating Cash Flow']

                if operating_cf and operating_cf > 0:
                    # Get benchmark multiple
                    benchmark_multiple = benchmarks.get('P/CF', 6.0)

                    # Calculate equity value
                    equity_value = operating_cf * benchmark_multiple

                    # Calculate per share value
                    if shares_outstanding and shares_outstanding > 0:
                        value_per_share = equity_value / shares_outstanding

                        valuations['p_cf'] = {
                            'value_per_share': value_per_share,
                            'multiple_used': benchmark_multiple,
                            'equity_value': equity_value
                        }

            # P/BV Valuation
            if 'p_bv' in multiples_to_use:
                book_value = None
                if 'Total Stockholder Equity' in balance_sheet.index:
                    book_value = balance_sheet.iloc[0]['Total Stockholder Equity']

                if book_value and book_value > 0:
                    # Get benchmark multiple
                    benchmark_multiple = benchmarks.get('P/B', 1.5)

                    # Calculate equity value
                    equity_value = book_value * benchmark_multiple

                    # Calculate per share value
                    if shares_outstanding and shares_outstanding > 0:
                        value_per_share = equity_value / shares_outstanding

                        valuations['p_bv'] = {
                            'value_per_share': value_per_share,
                            'multiple_used': benchmark_multiple,
                            'equity_value': equity_value
                        }

            # EV/Revenue Valuation (especially for renewable growth companies)
            if 'ev_revenue' in multiples_to_use:
                revenue = None
                if 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[0]['Total Revenue']

                if revenue and revenue > 0:
                    # Get total debt and cash for EV calculation
                    total_debt = 0
                    if 'Total Debt' in balance_sheet.index:
                        total_debt = balance_sheet.iloc[0]['Total Debt']

                    cash_and_equivalents = 0
                    if 'Cash and Cash Equivalents' in balance_sheet.index:
                        cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

                    # Get benchmark multiple
                    benchmark_multiple = benchmarks.get('EV/Revenue', 3.0)

                    # For renewables, use higher multiple
                    if subsector == 'renewable':
                        benchmark_multiple = max(benchmark_multiple, 4.0)

                    # Calculate enterprise value
                    enterprise_value = revenue * benchmark_multiple

                    # Calculate equity value
                    equity_value = enterprise_value - total_debt + cash_and_equivalents

                    # Calculate per share value
                    if shares_outstanding and shares_outstanding > 0:
                        value_per_share = equity_value / shares_outstanding

                        valuations['ev_revenue'] = {
                            'value_per_share': value_per_share,
                            'multiple_used': benchmark_multiple,
                            'equity_value': equity_value
                        }

            # Calculate weighted average valuation
            if valuations:
                # Define weights based on subsector
                weights = {}

                if subsector == 'oil' or subsector == 'gas':
                    weights = {
                        'ev_ebitda': 0.5,
                        'p_cf': 0.3,
                        'p_bv': 0.2,
                        'ev_revenue': 0.0
                    }
                elif subsector == 'coal':
                    weights = {
                        'ev_ebitda': 0.4,
                        'p_cf': 0.2,
                        'p_bv': 0.4,
                        'ev_revenue': 0.0
                    }
                elif subsector == 'renewable':
                    weights = {
                        'ev_ebitda': 0.4,
                        'p_cf': 0.2,
                        'p_bv': 0.1,
                        'ev_revenue': 0.3
                    }
                else:
                    weights = {
                        'ev_ebitda': 0.4,
                        'p_cf': 0.3,
                        'p_bv': 0.2,
                        'ev_revenue': 0.1
                    }

                # Calculate weighted average
                weighted_sum = 0
                total_weight = 0

                for method, valuation in valuations.items():
                    weight = weights.get(method, 0.0)
                    if weight > 0 and 'value_per_share' in valuation:
                        weighted_sum += valuation['value_per_share'] * weight
                        total_weight += weight

                if total_weight > 0:
                    weighted_average = weighted_sum / total_weight
                else:
                    weighted_average = None

                # Calculate upside/downside potential
                upside_potential = None
                if current_price and weighted_average and current_price > 0:
                    upside_potential = (weighted_average / current_price - 1) * 100

                return {
                    "valuation_method": "Multiple-based Valuation",
                    "valuations": valuations,
                    "weighted_average": weighted_average,
                    "current_price": current_price,
                    "upside_potential": upside_potential,
                    "interpretation": self._interpret_multiple_valuation(upside_potential, subsector)
                }
            else:
                return {"error": "Unable to calculate multiple-based valuation"}

        except Exception as e:
            logger.error(f"Error calculating multiple-based valuation: {e}")
            return {"error": str(e)}

    def _interpret_multiple_valuation(self, upside_potential: Optional[float], subsector: str) -> str:
        """Interpret multiple-based valuation results"""
        if upside_potential is None:
            return "Unable to determine valuation upside/downside."

        # Sector-specific interpretation
        threshold_high = 30 if subsector == 'renewable' else 20
        threshold_low = -20 if subsector == 'coal' else -15

        if upside_potential > threshold_high:
            return f"Significant upside potential of {upside_potential:.1f}%. Stock appears undervalued based on sector multiples."
        elif upside_potential > 5:
            return f"Moderate upside potential of {upside_potential:.1f}%. Stock appears somewhat undervalued based on sector multiples."
        elif upside_potential > threshold_low:
            return f"Limited upside/downside of {upside_potential:.1f}%. Stock appears fairly valued based on sector multiples."
        else:
            return f"Potential downside of {-upside_potential:.1f}%. Stock appears overvalued based on sector multiples."

    def _calculate_reserve_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """
        Calculate valuation based on reserves (primarily for oil & gas companies)
        This approach values a company based on its proven reserves
        """
        try:
            if subsector not in ['oil', 'gas']:
                return {"error": "Reserve-based valuation is only applicable for oil & gas companies"}

            balance_sheet = financial_data.get('balance_sheet')
            income_stmt = financial_data.get('income_statement')
            market_data = financial_data.get('market_data', {})

            if balance_sheet is None or income_stmt is None:
                return {"error": "Missing required data for reserve-based valuation"}

            # In a real implementation, actual reserve data would be used
            # For this simplified example, we'll estimate based on balance sheet items

            # Use Property Plant and Equipment as a proxy for production assets
            ppe = None
            if 'Property Plant and Equipment' in balance_sheet.index:
                ppe = balance_sheet.iloc[0]['Property Plant and Equipment']

            if ppe is None:
                return {"error": "Unable to estimate reserves from financial data"}

            # Estimate reserves value by applying a sector-specific multiplier to PPE
            # This is a crude approximation - real analysis would use actual reserves
            reserve_multiple = 1.8 if subsector == 'oil' else 2.2  # Gas assets often valued higher per BOE
            estimated_reserve_value = ppe * reserve_multiple

            # Determine per-barrel/MCF valuation benchmark
            # Typical valuation ranges by subsector
            if subsector == 'oil':
                # Oil typically valued at $10-20 per barrel of 2P reserves
                per_unit_value = 15.0  # $ per barrel
                unit_name = "barrel"
            else:  # gas
                # Gas typically valued at $1-3 per MCF of 2P reserves
                per_unit_value = 2.0  # $ per MCF
                unit_name = "MCF"

            # Estimate reserve quantity (very approximate)
            estimated_reserves = estimated_reserve_value / per_unit_value

            # Calculate enterprise value based on reserves
            enterprise_value = estimated_reserves * per_unit_value

            # Calculate equity value by adjusting for debt and cash
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.iloc[0]['Total Debt']

            cash_and_equivalents = 0
            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash_and_equivalents = balance_sheet.iloc[0]['Cash and Cash Equivalents']

            equity_value = enterprise_value - total_debt + cash_and_equivalents

            # Calculate per share value
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            # Get current market price for comparison
            current_price = market_data.get('share_price')

            # Calculate upside/downside potential
            upside_potential = None
            if current_price and value_per_share and current_price > 0:
                upside_potential = (value_per_share / current_price - 1) * 100

            return {
                "valuation_method": "Reserve-Based Valuation",
                "estimated_reserves": estimated_reserves,
                "reserve_unit": unit_name,
                "per_unit_value": per_unit_value,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "value_per_share": value_per_share,
                "current_price": current_price,
                "upside_potential": upside_potential,
                "interpretation": self._interpret_reserve_valuation(upside_potential, subsector)
            }

        except Exception as e:
            logger.error(f"Error calculating reserve-based valuation: {e}")
            return {"error": str(e)}

    def _interpret_reserve_valuation(self, upside_potential: Optional[float], subsector: str) -> str:
        """Interpret reserve-based valuation results"""
        if upside_potential is None:
            return "Unable to determine valuation based on reserves."

        if upside_potential > 30:
            return f"Significant upside potential of {upside_potential:.1f}%. Company appears undervalued on a reserves basis."
        elif upside_potential > 10:
            return f"Moderate upside potential of {upside_potential:.1f}%. Company appears somewhat undervalued on a reserves basis."
        elif upside_potential > -10:
            return f"Limited upside/downside. Company appears fairly valued on a reserves basis."
        else:
            return f"Potential downside of {-upside_potential:.1f}%. Company appears overvalued on a reserves basis."

    def _run_commodity_price_scenarios(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """
        Run commodity price sensitivity scenarios to assess valuation impact
        Different energy subsectors have different sensitivity to commodity prices
        """
        try:
            income_stmt = financial_data.get('income_statement')

            if income_stmt is None:
                return {"error": "Missing required data for commodity price scenarios"}

            # Get sensitivity factor based on subsector
            sensitivity = self.commodity_price_sensitivity.get(subsector, 0.5)

            # Set up price scenarios
            scenarios = {
                "bearish": -0.30,  # 30% price decline
                "base_case": 0.00,  # Current prices
                "bullish": 0.30  # 30% price increase
            }

            scenario_impacts = {}

            # Get base metrics
            base_revenue = None
            base_ebitda = None

            if 'Total Revenue' in income_stmt.index:
                base_revenue = income_stmt.iloc[0]['Total Revenue']

            if 'EBITDA' in income_stmt.index:
                base_ebitda = income_stmt.iloc[0]['EBITDA']
            elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                base_ebitda = income_stmt.iloc[0]['Operating Income'] + income_stmt.iloc[0][
                    'Depreciation & Amortization']

            # Calculate scenario impacts
            for scenario, price_change in scenarios.items():
                # Impact is based on price change and sensitivity factor
                impact = price_change * sensitivity

                # Calculate projected metrics under this scenario
                projected_metrics = {}

                if base_revenue is not None:
                    # Revenue impact - direct relation to price changes
                    revenue_impact = base_revenue * (1 + impact)
                    projected_metrics['revenue'] = revenue_impact

                if base_ebitda is not None:
                    # EBITDA impact - typically more sensitive than revenue
                    # Due to operating leverage
                    ebitda_leverage = 1.5  # EBITDA typically moves more than revenue
                    ebitda_impact = base_ebitda * (1 + (impact * ebitda_leverage))
                    projected_metrics['ebitda'] = ebitda_impact

                # Store scenario results
                scenario_impacts[scenario] = {
                    "price_change": price_change * 100,  # Convert to percentage
                    "metrics_impact": impact * 100,  # Convert to percentage
                    "projected_metrics": projected_metrics
                }

            return {
                "sensitivity_factor": sensitivity,
                "scenarios": scenario_impacts,
                "interpretation": self._interpret_price_scenarios(sensitivity, subsector)
            }

        except Exception as e:
            logger.error(f"Error running commodity price scenarios: {e}")
            return {"error": str(e)}

    def _interpret_price_scenarios(self, sensitivity: float, subsector: str) -> str:
        """Interpret commodity price sensitivity results"""
        if sensitivity > 0.7:
            return f"High sensitivity to commodity prices. Profitability and valuation highly dependent on market prices."
        elif sensitivity > 0.4:
            return f"Moderate sensitivity to commodity prices. Significant but manageable exposure to price fluctuations."
        else:
            return f"Lower sensitivity to commodity prices. More stable financial performance across price cycles."

    def _generate_analysis_summary(self, metrics: Dict[str, Any], risk_assessment: Dict[str, Any],
                                   subsector: str) -> str:
        """Generate summary analysis for energy company"""
        try:
            summary_points = []

            # Add subsector-specific introduction
            if subsector == 'oil':
                summary_points.append(
                    "This oil sector company demonstrates characteristics typical of the upstream/integrated petroleum industry.")
            elif subsector == 'gas':
                summary_points.append(
                    "This natural gas sector company operates in a market with distinctive supply-demand dynamics from the broader energy sector.")
            elif subsector == 'coal':
                summary_points.append(
                    "This coal sector company faces specific structural challenges related to energy transition and environmental regulations.")
            elif subsector == 'renewable':
                summary_points.append(
                    "This renewable energy company operates in a high-growth sector with evolving market dynamics and policy support mechanisms.")

            # Analyze operational metrics
            if 'ebitda_margin' in metrics:
                margin = metrics['ebitda_margin']
                if margin > 0.30:
                    summary_points.append(
                        f"Strong EBITDA margin of {margin:.1%} demonstrates excellent operational efficiency.")
                elif margin > 0.20:
                    summary_points.append(
                        f"Solid EBITDA margin of {margin:.1%} indicates good operational performance.")
                else:
                    summary_points.append(
                        f"EBITDA margin of {margin:.1%} suggests challenges in operational efficiency.")

            # Analyze capital efficiency
            if 'roce' in metrics:
                roce = metrics['roce']
                if roce > 0.15:
                    summary_points.append(
                        f"Strong return on capital employed ({roce:.1%}) indicates effective use of capital.")
                elif roce > 0.08:
                    summary_points.append(
                        f"Moderate return on capital employed ({roce:.1%}) suggests reasonable capital efficiency.")
                else:
                    summary_points.append(
                        f"Low return on capital employed ({roce:.1%}) may indicate challenges in capital allocation.")

            # Analyze subsector-specific metrics
            if subsector in ['oil', 'gas']:
                if 'reserve_replacement_ratio' in metrics and metrics['reserve_replacement_ratio']:
                    rrr = metrics['reserve_replacement_ratio']
                    if rrr > 1.2:
                        summary_points.append(
                            f"Strong reserve replacement ratio of {rrr:.2f} indicates growth in resource base.")
                    elif rrr > 1.0:
                        summary_points.append(
                            f"Reserve replacement ratio of {rrr:.2f} indicates maintenance of resource base.")
                    else:
                        summary_points.append(
                            f"Reserve replacement ratio of {rrr:.2f} suggests declining resource base.")

                if 'reserve_life_index' in metrics and metrics['reserve_life_index']:
                    rli = metrics['reserve_life_index']
                    if rli > 15:
                        summary_points.append(
                            f"Long reserve life of approximately {rli:.1f} years provides production visibility.")
                    elif rli > 8:
                        summary_points.append(
                            f"Moderate reserve life of approximately {rli:.1f} years is in line with industry norms.")
                    else:
                        summary_points.append(
                            f"Relatively short reserve life of approximately {rli:.1f} years may require focus on reserve additions.")

            # Add risk assessment summary
            if 'overall_risk' in risk_assessment:
                risk_level = risk_assessment['overall_risk']['level']
                if risk_level > 0.7:
                    summary_points.append(
                        "The company faces significant risks that could impact financial performance and valuation.")
                elif risk_level > 0.4:
                    summary_points.append(
                        "The company faces moderate risks that require monitoring but appear manageable.")
                else:
                    summary_points.append(
                        "The company demonstrates a relatively low risk profile within its sector.")

            # Add valuation indicators
            if 'price_to_cash_flow' in metrics:
                pcf = metrics['price_to_cash_flow']
                if pcf:
                    if pcf < 5:
                        summary_points.append(
                            f"Low price-to-cash-flow ratio of {pcf:.1f} may indicate potential undervaluation.")
                    elif pcf > 10:
                        summary_points.append(
                            f"Elevated price-to-cash-flow ratio of {pcf:.1f} suggests premium valuation.")
                    else:
                        summary_points.append(
                            f"Price-to-cash-flow ratio of {pcf:.1f} is within typical industry range.")

            if 'ev_to_ebitda' in metrics:
                ev_ebitda = metrics['ev_to_ebitda']
                if ev_ebitda:
                    if ev_ebitda < 5:
                        summary_points.append(
                            f"Low EV/EBITDA multiple of {ev_ebitda:.1f} may indicate potential undervaluation.")
                    elif ev_ebitda > 10:
                        summary_points.append(
                            f"Elevated EV/EBITDA multiple of {ev_ebitda:.1f} suggests premium valuation.")
                    else:
                        summary_points.append(
                            f"EV/EBITDA multiple of {ev_ebitda:.1f} is within typical industry range.")

            # Generate overall conclusion
            if subsector in ['oil', 'gas', 'coal']:
                summary_points.append(
                    f"As a {subsector} sector company, its valuation is closely tied to commodity prices, operational efficiency, and reserve management.")
            elif subsector == 'renewable':
                summary_points.append(
                    "As a renewable energy company, its valuation is primarily driven by growth expectations, policy support, and operational execution.")

            # Join all points into a cohesive summary
            return " ".join(summary_points)

        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return "Unable to generate detailed analysis due to data limitations."

    def _generate_benchmark_summary(self, comparison_results: Dict[str, Dict[str, Any]],
                                    overall_score: float) -> str:
        """Generate summary of benchmark comparison results"""
        try:
            summary_points = []

            # Overall assessment based on score
            if overall_score >= 80:
                summary_points.append(
                    f"The company outperforms sector benchmarks on key metrics with an overall score of {overall_score:.1f}/100.")
            elif overall_score >= 60:
                summary_points.append(
                    f"The company performs above average relative to sector benchmarks with a score of {overall_score:.1f}/100.")
            elif overall_score >= 40:
                summary_points.append(
                    f"The company performs near sector average with a score of {overall_score:.1f}/100.")
            else:
                summary_points.append(
                    f"The company underperforms sector benchmarks on key metrics with a score of {overall_score:.1f}/100.")

            # Identify strongest metrics
            strengths = []
            weaknesses = []

            for metric, result in comparison_results.items():
                if result.get('performance', 0) >= 120:
                    strengths.append(metric)
                elif result.get('performance', 0) <= 80:
                    weaknesses.append(metric)

            # Add summary of strengths
            if strengths:
                if len(strengths) > 2:
                    strength_text = f"Notable strengths include {', '.join(strengths[:-1])} and {strengths[-1]}."
                elif len(strengths) == 2:
                    strength_text = f"Notable strengths include {strengths[0]} and {strengths[1]}."
                else:
                    strength_text = f"A notable strength is {strengths[0]}."
                summary_points.append(strength_text)

            # Add summary of weaknesses
            if weaknesses:
                if len(weaknesses) > 2:
                    weakness_text = f"Areas for improvement include {', '.join(weaknesses[:-1])} and {weaknesses[-1]}."
                elif len(weaknesses) == 2:
                    weakness_text = f"Areas for improvement include {weaknesses[0]} and {weaknesses[1]}."
                else:
                    weakness_text = f"An area for improvement is {weaknesses[0]}."
                summary_points.append(weakness_text)

            # Add sector-specific conclusion
            if overall_score >= 60:
                summary_points.append(
                    "Within the energy sector context, the company demonstrates competitive operational and financial metrics.")
            else:
                summary_points.append(
                    "Within the energy sector context, the company faces challenges in matching industry benchmarks.")

            # Join all points into a cohesive summary
            return " ".join(summary_points)

        except Exception as e:
            logger.error(f"Error generating benchmark summary: {e}")
            return "Unable to generate detailed benchmark comparison due to data limitations."

# Register the EnergySectorAnalyzer with the SectorFactory
SectorFactory.register_analyzer("Energy", EnergySectorAnalyzer)