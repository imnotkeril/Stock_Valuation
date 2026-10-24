import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Add the project root to Python path
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


from StockAnalysisSystem.src.config import SECTOR_METRICS, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.valuation.sector_factor import SectorAnalyzer, SectorFactory
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('retail_sector')


class RetailSectorAnalyzer(SectorAnalyzer):
    """
    Retail sector-specific analyzer.
    Implements analysis methods tailored for retail and consumer companies.
    """

    def __init__(self, sector_name: str = "Consumer Discretionary"):
        """Initialize with Retail sector parameters"""
        super().__init__(sector_name)
        self.data_loader = DataLoader()

        # Define retail sub-sectors
        self.retail_subsectors = {
            "apparel": ["apparel", "clothing", "fashion", "footwear", "accessories"],
            "department_store": ["department store", "multi-line retail"],
            "specialty": ["specialty", "home improvement", "electronics", "furniture", "auto parts"],
            "e_commerce": ["e-commerce", "online", "internet retail"],
            "grocery": ["grocery", "food retail", "supermarket"],
            "discount": ["discount", "dollar store", "off-price"]
        }

        logger.info(f"{sector_name} sector analyzer initialized")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a retail company using sector-specific metrics

        Args:
            financial_data: Dictionary with company financial statements and information

        Returns:
            Dictionary with retail sector-specific analysis results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            # Basic validation
            if income_stmt is None or balance_sheet is None:
                logger.error("Missing required financial statements for retail sector analysis")
                return {"error": "Insufficient financial data for analysis"}

            # Get company subsector
            subsector = self._determine_retail_subsector(company_info)

            # Calculate retail-specific metrics
            retail_metrics = {}

            # 1. Inventory Turnover
            inventory_turnover, days_inventory = self._calculate_inventory_metrics(income_stmt, balance_sheet)
            retail_metrics['inventory_turnover'] = inventory_turnover
            retail_metrics['days_inventory'] = days_inventory

            # 2. Same-Store Sales Growth (estimate)
            sss_growth = self._estimate_same_store_sales_growth(income_stmt, company_info)
            retail_metrics['same_store_sales_growth'] = sss_growth

            # 3. Gross Margin
            gross_margin = self._calculate_gross_margin(income_stmt)
            retail_metrics['gross_margin'] = gross_margin

            # 4. Operating Margin
            operating_margin = self._calculate_operating_margin(income_stmt)
            retail_metrics['operating_margin'] = operating_margin

            # 5. Revenue per Square Foot (estimate)
            revenue_per_sqft = self._estimate_revenue_per_sqft(income_stmt, company_info)
            retail_metrics['revenue_per_sqft'] = revenue_per_sqft

            # 6. SG&A as percentage of revenue
            sg_and_a_percentage = self._calculate_sga_percentage(income_stmt)
            retail_metrics['sg_and_a_percentage'] = sg_and_a_percentage

            # 7. E-commerce sales percentage (estimate)
            ecommerce_percentage = self._estimate_ecommerce_percentage(company_info, subsector)
            retail_metrics['ecommerce_percentage'] = ecommerce_percentage

            # 8. Loyalty program metrics (if available)
            loyalty_metrics = self._estimate_loyalty_metrics(company_info)
            if loyalty_metrics:
                retail_metrics.update(loyalty_metrics)

            # 9. Comparable Store Count Growth
            store_count_growth = self._estimate_store_count_growth(company_info)
            retail_metrics['store_count_growth'] = store_count_growth

            # 10. Cash Conversion Cycle
            cash_conversion_cycle = self._calculate_cash_conversion_cycle(
                income_stmt, balance_sheet)
            retail_metrics['cash_conversion_cycle'] = cash_conversion_cycle

            # Perform deeper analysis based on subsector
            subsector_analysis = self._analyze_retail_subsector(
                financial_data, subsector, retail_metrics)

            # Assess retail-specific risks
            risk_assessment = self._assess_retail_risks(financial_data, subsector, retail_metrics)

            # Generate SWOT analysis
            swot_analysis = self._generate_retail_swot(financial_data, subsector, retail_metrics)

            return {
                "sector": "Retail",
                "subsector": subsector,
                "metrics": retail_metrics,
                "subsector_analysis": subsector_analysis,
                "risk_assessment": risk_assessment,
                "swot_analysis": swot_analysis,
                "summary": self._generate_retail_summary(retail_metrics, risk_assessment, subsector)
            }

        except Exception as e:
            logger.error(f"Error analyzing retail company: {e}")
            return {"error": str(e)}

    def _calculate_sum_of_parts_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Calculate sum-of-parts valuation, particularly useful for multi-format retailers"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for sum-of-parts valuation")

            # This method is most useful for retailers with multiple business segments
            # We'll create segment estimates based on company description or subsector

            segments = []
            segment_revenues = {}
            segment_ebitda = {}

            # Try to identify segments from company description
            description = company_info.get('description', '').lower()

            # For department stores, we might have apparel, home, beauty segments
            if subsector == 'department_store':
                segments = ['apparel', 'home', 'beauty', 'services']
                # Rough estimate of segment split
                segment_revenues = {
                    'apparel': 0.50,  # 50% of revenue
                    'home': 0.25,  # 25% of revenue
                    'beauty': 0.20,  # 20% of revenue
                    'services': 0.05  # 5% of revenue
                }
                segment_ebitda = {
                    'apparel': 0.45,  # 45% of EBITDA
                    'home': 0.30,  # 30% of EBITDA
                    'beauty': 0.22,  # 22% of EBITDA
                    'services': 0.03  # 3% of EBITDA
                }

            # For specialty retailers with physical and online channels
            elif subsector == 'specialty' or subsector == 'apparel':
                segments = ['stores', 'e-commerce']
                # Check for e-commerce estimates
                ecommerce_percentage = self._estimate_ecommerce_percentage(company_info, subsector)

                if ecommerce_percentage:
                    segment_revenues = {
                        'stores': 1 - ecommerce_percentage,
                        'e-commerce': ecommerce_percentage
                    }
                    # E-commerce often has different profitability
                    if ecommerce_percentage > 0.3:  # High e-commerce penetration
                        segment_ebitda = {
                            'stores': (1 - ecommerce_percentage) * 0.9,  # Stores slightly less profitable
                            'e-commerce': ecommerce_percentage * 1.1  # E-commerce slightly more profitable
                        }
                    else:  # Lower e-commerce penetration
                        segment_ebitda = {
                            'stores': (1 - ecommerce_percentage) * 1.1,  # Stores more profitable at lower e-comm mix
                            'e-commerce': ecommerce_percentage * 0.9  # E-commerce less profitable at lower scale
                        }
                else:
                    # Default split if no e-commerce estimate
                    segment_revenues = {
                        'stores': 0.85,
                        'e-commerce': 0.15
                    }
                    segment_ebitda = {
                        'stores': 0.90,
                        'e-commerce': 0.10
                    }

            # For grocery retailers with different formats
            elif subsector == 'grocery':
                segments = ['traditional', 'organic/premium', 'e-commerce']

                # Default segment splits
                segment_revenues = {
                    'traditional': 0.75,
                    'organic/premium': 0.15,
                    'e-commerce': 0.10
                }
                segment_ebitda = {
                    'traditional': 0.70,
                    'organic/premium': 0.25,
                    'e-commerce': 0.05
                }

            # If we haven't set segments based on subsector, use a generic approach
            if not segments:
                segments = ['main_business', 'secondary_business']
                segment_revenues = {
                    'main_business': 0.80,
                    'secondary_business': 0.20
                }
                segment_ebitda = {
                    'main_business': 0.85,
                    'secondary_business': 0.15
                }

            # Get overall financial metrics
            revenue = None

            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

            if 'EBITDA' in income_stmt.index:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
            elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].iloc[0]
                depreciation = income_stmt.loc['Depreciation & Amortization'].iloc[0]
                ebitda = operating_income + depreciation

            if revenue is None or ebitda is None:
                raise ValueError("Missing revenue or EBITDA data required for sum-of-parts valuation")

            # Calculate segment values
            segment_values = {}
            total_enterprise_value = 0

            for segment in segments:
                # Calculate segment revenue and EBITDA
                segment_revenue = revenue * segment_revenues.get(segment, 0)
                segment_ebitda_value = ebitda * segment_ebitda.get(segment, 0)

                # Determine appropriate multiple for each segment
                if segment == 'e-commerce':
                    ev_ebitda_multiple = 15.0
                    ev_sales_multiple = 2.0
                elif segment == 'stores' or segment == 'traditional':
                    ev_ebitda_multiple = 7.0
                    ev_sales_multiple = 0.6
                elif segment == 'organic/premium':
                    ev_ebitda_multiple = 10.0
                    ev_sales_multiple = 1.0
                elif segment == 'beauty':
                    ev_ebitda_multiple = 12.0
                    ev_sales_multiple = 1.5
                elif segment == 'home':
                    ev_ebitda_multiple = 8.0
                    ev_sales_multiple = 0.8
                elif segment == 'apparel':
                    ev_ebitda_multiple = 9.0
                    ev_sales_multiple = 1.0
                else:
                    ev_ebitda_multiple = 8.0
                    ev_sales_multiple = 0.8

                # Calculate segment enterprise value
                if segment_ebitda_value > 0:
                    # Prefer EBITDA-based valuation if available
                    segment_ev = segment_ebitda_value * ev_ebitda_multiple
                else:
                    # Fall back to revenue-based valuation
                    segment_ev = segment_revenue * ev_sales_multiple

                segment_values[segment] = {
                    'revenue': segment_revenue,
                    'ebitda': segment_ebitda_value,
                    'ev_ebitda_multiple': ev_ebitda_multiple,
                    'ev_sales_multiple': ev_sales_multiple,
                    'enterprise_value': segment_ev
                }

                total_enterprise_value += segment_ev

            # Calculate equity value
            total_debt = 0
            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]

            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]

            equity_value = total_enterprise_value - total_debt + cash

            # Calculate per share value
            shares_outstanding = company_info.get('shares_outstanding')

            if shares_outstanding:
                per_share_value = equity_value / shares_outstanding
            else:
                # Try to estimate shares from market cap and share price
                market_cap = company_info.get('market_cap')
                share_price = company_info.get('share_price')

                if market_cap and share_price and share_price > 0:
                    shares_outstanding = market_cap / share_price
                    per_share_value = equity_value / shares_outstanding
                else:
                    per_share_value = None

            # Return sum-of-parts valuation results
            return {
                "valuation_method": "sum_of_parts",
                "total_enterprise_value": total_enterprise_value,
                "equity_value": equity_value,
                "per_share_value": per_share_value,
                "segment_values": segment_values,
                "segments": segments,
                "segment_revenue_splits": segment_revenues,
                "segment_ebitda_splits": segment_ebitda
            }

        except Exception as e:
            logger.error(f"Error in retail sum-of-parts valuation: {e}")
            return {
                "valuation_method": "sum_of_parts",
                "error": str(e),
                "total_enterprise_value": None,
                "equity_value": None,
                "per_share_value": None
            }

    def _run_retail_scenarios(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Run retail-specific scenarios to analyze valuation sensitivity"""
        try:
            # Define scenarios relevant to retail
            scenarios = {
                "base_case": {
                    "description": "Current trajectory continues",
                    "same_store_sales_growth": 0.02,  # 2% SSS growth
                    "gross_margin_change": 0.0,  # No change in margins
                    "store_count_growth": 0.02,  # 2% store growth
                    "ecommerce_growth": 0.10,  # 10% e-commerce growth
                    "probability": 0.60  # 60% probability
                },
                "optimistic": {
                    "description": "Strong consumer spending and operational improvements",
                    "same_store_sales_growth": 0.04,  # 4% SSS growth
                    "gross_margin_change": 0.01,  # 1% margin expansion
                    "store_count_growth": 0.04,  # 4% store growth
                    "ecommerce_growth": 0.20,  # 20% e-commerce growth
                    "probability": 0.20  # 20% probability
                },
                "pessimistic": {
                    "description": "Consumer spending slowdown and margin pressure",
                    "same_store_sales_growth": -0.01,  # -1% SSS decline
                    "gross_margin_change": -0.01,  # 1% margin contraction
                    "store_count_growth": 0.0,  # Flat store count
                    "ecommerce_growth": 0.05,  # 5% e-commerce growth
                    "probability": 0.20  # 20% probability
                }
            }

            # Customize scenarios based on subsector
            if subsector == 'department_store':
                # Department stores facing structural challenges
                scenarios["base_case"]["store_count_growth"] = -0.01  # Store closures
                scenarios["base_case"]["same_store_sales_growth"] = 0.0
                scenarios["optimistic"]["store_count_growth"] = 0.0  # Flat store count at best
                scenarios["pessimistic"]["store_count_growth"] = -0.03  # Larger store closures
                scenarios["pessimistic"]["same_store_sales_growth"] = -0.03

            elif subsector == 'e_commerce':
                # E-commerce with higher growth expectations
                scenarios["base_case"]["ecommerce_growth"] = 0.20
                scenarios["optimistic"]["ecommerce_growth"] = 0.30
                scenarios["pessimistic"]["ecommerce_growth"] = 0.10
                # No store metrics for e-commerce
                for scenario in scenarios.values():
                    scenario.pop("store_count_growth", None)
                    scenario.pop("same_store_sales_growth", None)

            elif subsector == 'discount':
                # Discount retailers often more resilient in downturns
                scenarios["base_case"]["store_count_growth"] = 0.03
                scenarios["pessimistic"]["same_store_sales_growth"] = 0.01  # Still positive in downturn

            # Execute the base valuation method
            base_valuation = self._calculate_retail_dcf(financial_data, subsector)

            # Calculate valuations under each scenario
            scenario_valuations = {}

            for scenario_name, scenario_params in scenarios.items():
                # Create a modified copy of financial data with scenario adjustments
                scenario_financials = self._adjust_financials_for_scenario(
                    financial_data.copy(), scenario_params, subsector)

                # Run DCF with adjusted financials
                scenario_valuation = self._calculate_retail_dcf(scenario_financials, subsector)

                # Store results
                scenario_valuations[scenario_name] = {
                    "description": scenario_params["description"],
                    "parameters": scenario_params,
                    "enterprise_value": scenario_valuation.get("enterprise_value"),
                    "equity_value": scenario_valuation.get("equity_value"),
                    "per_share_value": scenario_valuation.get("per_share_value"),
                    "probability": scenario_params.get("probability", 0)
                }

            # Calculate expected value
            expected_per_share_value = 0
            total_probability = 0

            for scenario, valuation in scenario_valuations.items():
                if valuation.get("per_share_value") and valuation.get("probability"):
                    expected_per_share_value += valuation["per_share_value"] * valuation["probability"]
                    total_probability += valuation["probability"]

            # Normalize if probabilities don't sum to 1
            if total_probability > 0 and total_probability != 1.0:
                expected_per_share_value = expected_per_share_value / total_probability

            return {
                "scenarios": scenario_valuations,
                "expected_per_share_value": expected_per_share_value
            }

        except Exception as e:
            logger.error(f"Error running retail scenarios: {e}")
            return {
                "error": str(e),
                "scenarios": {},
                "expected_per_share_value": None
            }

    def _adjust_financials_for_scenario(self, financial_data: Dict[str, Any],
                                        scenario: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Adjust financial data based on retail scenario parameters"""
        try:
            # Make a deep copy of income statement to avoid modifying original
            if 'income_statement' in financial_data and financial_data['income_statement'] is not None:
                income_stmt = financial_data['income_statement'].copy()

                # Adjust revenue based on same-store sales and store count growth
                if 'Total Revenue' in income_stmt.index:
                    base_revenue = income_stmt.iloc[0]['Total Revenue']

                    revenue_multiplier = 1.0

                    # Apply same-store sales growth if applicable
                    if 'same_store_sales_growth' in scenario and subsector != 'e_commerce':
                        revenue_multiplier += scenario['same_store_sales_growth']

                    # Apply store count growth if applicable
                    if 'store_count_growth' in scenario and subsector != 'e_commerce':
                        # Store growth affects only a portion of revenue (new stores ramp up)
                        revenue_multiplier += (scenario['store_count_growth'] * 0.5)

                    # Apply e-commerce growth
                    if 'ecommerce_growth' in scenario:
                        # Estimate e-commerce percentage
                        ecommerce_pct = self._estimate_ecommerce_percentage(
                            financial_data.get('company_info', {}), subsector) or 0.15

                        # Adjust revenue based on e-commerce growth
                        revenue_multiplier += (scenario['ecommerce_growth'] * ecommerce_pct)

                    # Apply the revenue multiplier to first year
                    income_stmt.iloc[0]['Total Revenue'] = base_revenue * revenue_multiplier

                # Adjust gross profit based on margin changes
                if 'gross_margin_change' in scenario and 'Gross Profit' in income_stmt.index:
                    base_margin = income_stmt.iloc[0]['Gross Profit'] / income_stmt.iloc[0]['Total Revenue']
                    new_margin = base_margin + scenario['gross_margin_change']
                    income_stmt.iloc[0]['Gross Profit'] = income_stmt.iloc[0]['Total Revenue'] * new_margin

                # Adjust operating expenses (simplified approach)
                if 'Gross Profit' in income_stmt.index and 'Operating Income' in income_stmt.index:
                    # Keep the same operating expense to revenue ratio
                    base_op_expense = income_stmt.iloc[0]['Gross Profit'] - income_stmt.iloc[0]['Operating Income']
                    op_expense_to_revenue = base_op_expense / income_stmt.iloc[0]['Total Revenue']

                    # Apply the same ratio to calculate new operating expense
                    new_op_expense = income_stmt.iloc[0]['Total Revenue'] * op_expense_to_revenue
                    income_stmt.iloc[0]['Operating Income'] = income_stmt.iloc[0]['Gross Profit'] - new_op_expense

                # Update the financial data with modified income statement
                financial_data['income_statement'] = income_stmt

            return financial_data

        except Exception as e:
            logger.error(f"Error adjusting financials for scenario: {e}")
            return financial_data  # Return original data if adjustments fail

    def _benchmark_against_subsectors(self, company_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Compare company metrics against retail subsector benchmarks"""
        try:
            # Define benchmarks for each retail subsector
            subsector_benchmarks = {
                "apparel": {
                    "gross_margin": 0.40,
                    "operating_margin": 0.10,
                    "inventory_turnover": 5.0,
                    "same_store_sales_growth": 0.03,
                    "sg_and_a_percentage": 0.28
                },
                "department_store": {
                    "gross_margin": 0.35,
                    "operating_margin": 0.06,
                    "inventory_turnover": 4.0,
                    "same_store_sales_growth": 0.01,
                    "sg_and_a_percentage": 0.30
                },
                "specialty": {
                    "gross_margin": 0.42,
                    "operating_margin": 0.09,
                    "inventory_turnover": 4.5,
                    "same_store_sales_growth": 0.03,
                    "sg_and_a_percentage": 0.26
                },
                "e_commerce": {
                    "gross_margin": 0.45,
                    "operating_margin": 0.08,
                    "inventory_turnover": 8.0,
                    "revenue_growth": 0.20,
                    "sg_and_a_percentage": 0.35
                },
                "grocery": {
                    "gross_margin": 0.25,
                    "operating_margin": 0.04,
                    "inventory_turnover": 12.0,
                    "same_store_sales_growth": 0.02,
                    "sg_and_a_percentage": 0.18
                },
                "discount": {
                    "gross_margin": 0.30,
                    "operating_margin": 0.07,
                    "inventory_turnover": 6.0,
                    "same_store_sales_growth": 0.03,
                    "sg_and_a_percentage": 0.20
                }
            }

            # Compare company metrics against each subsector
            subsector_comparisons = {}

            for subsector, benchmarks in subsector_benchmarks.items():
                subsector_score = 0
                comparison_count = 0

                metrics_comparison = {}

                for metric, benchmark in benchmarks.items():
                    if metric in company_metrics and company_metrics[metric] is not None:
                        # Determine if higher or lower is better
                        higher_is_better = True
                        if metric in ['sg_and_a_percentage']:
                            higher_is_better = False

                        # Calculate performance
                        if higher_is_better:
                            ratio = company_metrics[metric] / benchmark if benchmark else 0
                        else:
                            ratio = benchmark / company_metrics[metric] if company_metrics[metric] else 0

                        # Calculate score (0-100)
                        score = min(100, max(0, ratio * 100 - 20))  # Scale to make 80% = neutral
                        subsector_score += score
                        comparison_count += 1

                        # Store comparison
                        metrics_comparison[metric] = {
                            "company_value": company_metrics[metric],
                            "benchmark": benchmark,
                            "ratio": ratio,
                            "score": score
                        }

                # Calculate average score for this subsector
                average_score = subsector_score / comparison_count if comparison_count > 0 else 0

                # Determine best fit
                fit_assessment = "Poor fit"
                if average_score >= 90:
                    fit_assessment = "Excellent fit"
                elif average_score >= 75:
                    fit_assessment = "Good fit"
                elif average_score >= 60:
                    fit_assessment = "Moderate fit"
                elif average_score >= 40:
                    fit_assessment = "Fair fit"

                subsector_comparisons[subsector] = {
                    "metrics_comparison": metrics_comparison,
                    "average_score": average_score,
                    "fit_assessment": fit_assessment
                }

            # Determine best matching subsector
            best_subsector = max(subsector_comparisons.items(),
                                 key=lambda x: x[1]["average_score"])

            return {
                "subsector_comparisons": subsector_comparisons,
                "best_matching_subsector": best_subsector[0],
                "best_matching_score": best_subsector[1]["average_score"]
            }

        except Exception as e:
            logger.error(f"Error benchmarking against subsectors: {e}")
            return {"error": str(e)}

    def _estimate_store_count(self, company_info: Dict[str, Any], revenue: float, subsector: str) -> int:
        """Estimate store count based on revenue and subsector"""
        try:
            # Check company description for store count information
            description = company_info.get('description', '').lower()

            # Simple pattern matching for store count in description
            import re
            store_count_patterns = [
                r'(\d+,\d+|\d+)\s+stores',
                r'(\d+,\d+|\d+)\s+locations',
                r'operates\s+(\d+,\d+|\d+)',
                r'operates\s+approximately\s+(\d+,\d+|\d+)'
            ]

            for pattern in store_count_patterns:
                match = re.search(pattern, description)
                if match:
                    # Found a number that might be store count
                    store_count_str = match.group(1).replace(',', '')
                    return int(store_count_str)

            # If no match in description, estimate based on revenue and subsector
            if revenue:
                # Average revenue per store by subsector
                revenue_per_store = {
                    'apparel': 2000000,  # $2M per store
                    'department_store': 30000000,  # $30M per store
                    'specialty': 4000000,  # $4M per store
                    'grocery': 20000000,  # $20M per store
                    'discount': 10000000  # $10M per store
                }

                avg_revenue = revenue_per_store.get(subsector, 5000000)  # Default: $5M per store

                # Estimate store count
                estimated_stores = int(revenue / avg_revenue)

                # Sanity check
                if estimated_stores < 1:
                    estimated_stores = 1

                return estimated_stores

            return None

        except Exception as e:
            logger.error(f"Error estimating store count: {e}")
            return None

    def _generate_retail_benchmark_summary(self, comparison_results: Dict[str, Dict[str, Any]],
                                           subsector_benchmarking: Dict[str, Dict[str, Any]],
                                           overall_score: float) -> str:
        """Generate summary of retail benchmarking results"""
        try:
            # Start with overall assessment
            summary = f"Overall, the company scores {overall_score:.1f}/100 against retail sector benchmarks. "

            # Add assessment based on score
            if overall_score >= 80:
                summary += "The company significantly outperforms retail sector averages across key metrics. "
            elif overall_score >= 65:
                summary += "The company performs above average compared to retail sector benchmarks. "
            elif overall_score >= 50:
                summary += "The company's performance is in line with retail sector benchmarks. "
            elif overall_score >= 35:
                summary += "The company underperforms compared to retail sector benchmarks. "
            else:
                summary += "The company significantly underperforms compared to retail sector averages. "

            # Add best matching subsector if available
            if subsector_benchmarking and 'best_matching_subsector' in subsector_benchmarking:
                best_subsector = subsector_benchmarking['best_matching_subsector']
                best_score = subsector_benchmarking.get('best_matching_score', 0)

                summary += f"The company's metrics most closely align with the {best_subsector} retail subsector "
                summary += f"(similarity score: {best_score:.1f}/100). "

            # Highlight strong metrics
            strong_metrics = []
            for metric, data in comparison_results.items():
                if data.get('performance', 0) >= 110:  # Significantly outperforming
                    metric_name = metric.replace('_', ' ').title()
                    strong_metrics.append(metric_name)

            if strong_metrics:
                summary += f"Notable strengths include {', '.join(strong_metrics[:3])}. "

            # Highlight weak metrics
            weak_metrics = []
            for metric, data in comparison_results.items():
                if data.get('performance', 0) <= 75:  # Significantly underperforming
                    metric_name = metric.replace('_', ' ').title()
                    weak_metrics.append(metric_name)

            if weak_metrics:
                summary += f"Areas for improvement include {', '.join(weak_metrics[:3])}. "

            return summary

        except Exception as e:
            logger.error(f"Error generating retail benchmark summary: {e}")
            return "Error generating benchmark summary."

    def _generate_retail_summary(self, retail_metrics: Dict[str, float],
                                 risk_assessment: Dict[str, Any], subsector: str) -> str:
        """Generate overall summary of retail company analysis"""
        try:
            # Start with subsector identification
            summary = f"The company operates in the {subsector.replace('_', ' ')} retail subsector. "

            # Add key metrics summary
            if retail_metrics.get('gross_margin'):
                summary += f"Gross margin of {retail_metrics['gross_margin'] * 100:.1f}% "

                # Comparative assessment by subsector
                if subsector == 'grocery' and retail_metrics['gross_margin'] > 0.25:
                    summary += "is above average for grocery retailers, suggesting premium positioning. "
                elif subsector == 'apparel' and retail_metrics['gross_margin'] > 0.40:
                    summary += "indicates strong brand value and pricing power. "
                elif subsector == 'discount' and retail_metrics['gross_margin'] < 0.25:
                    summary += "aligns with discount retail strategy focusing on volume over margin. "
                else:
                    summary += "is in line with sector averages. "

            # Comment on operational efficiency
            if retail_metrics.get('inventory_turnover'):
                summary += f"Inventory turnover of {retail_metrics['inventory_turnover']:.1f}x "

                if retail_metrics['inventory_turnover'] > 8:
                    summary += "demonstrates excellent inventory management. "
                elif retail_metrics['inventory_turnover'] > 5:
                    summary += "shows good inventory management practices. "
                elif retail_metrics['inventory_turnover'] < 3:
                    summary += "suggests potential inventory management challenges. "
                else:
                    summary += "is within normal range for the sector. "

            # E-commerce presence
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                summary += f"E-commerce represents approximately {retail_metrics['ecommerce_percentage'] * 100:.1f}% of sales, "

                if retail_metrics['ecommerce_percentage'] > 0.30:
                    summary += "indicating strong digital channel penetration. "
                elif retail_metrics['ecommerce_percentage'] < 0.10:
                    summary += "suggesting opportunity for digital channel expansion. "
                else:
                    summary += "which is in line with sector averages. "

            # Store productivity
            if retail_metrics.get('revenue_per_sqft') and subsector != 'e_commerce':
                summary += f"Estimated revenue per square foot of ${retail_metrics['revenue_per_sqft']:.0f} "

                if subsector == 'apparel' and retail_metrics['revenue_per_sqft'] > 500:
                    summary += "demonstrates exceptional store productivity for apparel retail. "
                elif subsector == 'grocery' and retail_metrics['revenue_per_sqft'] > 700:
                    summary += "indicates top-tier store productivity for grocery retail. "
                elif retail_metrics['revenue_per_sqft'] < 200:
                    summary += "suggests potential opportunity to improve store productivity. "
                else:
                    summary += "is consistent with industry standards. "

            # Growth metrics
            if retail_metrics.get('same_store_sales_growth'):
                summary += f"Same-store sales growth of {retail_metrics['same_store_sales_growth'] * 100:.1f}% "

                if retail_metrics['same_store_sales_growth'] > 0.04:
                    summary += "shows strong customer traction and sales momentum. "
                elif retail_metrics['same_store_sales_growth'] < 0:
                    summary += "indicates challenges in maintaining sales at existing locations. "
                else:
                    summary += "is in line with sector performance. "

            # Overall risk assessment
            if risk_assessment and 'overall_risk' in risk_assessment:
                overall_risk = risk_assessment['overall_risk']
                summary += f"Overall risk assessment: {overall_risk.get('description', '')}. "

                # Add highest individual risk factor
                highest_risk = None
                highest_risk_level = 0

                for risk_type, risk_data in risk_assessment.items():
                    if risk_type != 'overall_risk' and isinstance(risk_data, dict):
                        risk_level = risk_data.get('level', 0)
                        if risk_level > highest_risk_level:
                            highest_risk_level = risk_level
                            highest_risk = risk_data.get('description', '')

                if highest_risk and highest_risk_level > 0.6:
                    summary += f"The most significant risk factor is {highest_risk}. "

            return summary

        except Exception as e:
            logger.error(f"Error generating retail summary: {e}")
            return f"The company operates in the {subsector.replace('_', ' ')} retail subsector. Detailed analysis unavailable."

    def _generate_retail_swot(self, financial_data: Dict[str, Any],
                              subsector: str, retail_metrics: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate a SWOT analysis for a retail company"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            swot = {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }

            # Strengths analysis
            # Financial strength
            if retail_metrics.get('operating_margin'):
                if retail_metrics['operating_margin'] > 0.08:
                    swot["strengths"].append("Strong operating margins compared to retail sector peers")

            if retail_metrics.get('gross_margin'):
                if retail_metrics['gross_margin'] > 0.40:
                    swot["strengths"].append("Premium pricing power and brand strength reflected in high gross margins")

            # Inventory management
            if retail_metrics.get('inventory_turnover'):
                if retail_metrics['inventory_turnover'] > 5:
                    swot["strengths"].append("Efficient inventory management with above-average turnover")

            # Store productivity
            if retail_metrics.get('revenue_per_sqft'):
                if retail_metrics['revenue_per_sqft'] > 500:
                    swot["strengths"].append("Exceptional store productivity with high revenue per square foot")
                elif retail_metrics['revenue_per_sqft'] > 300:
                    swot["strengths"].append("Strong store productivity")

            # E-commerce presence
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] > 0.25:
                    swot["strengths"].append("Well-developed e-commerce capabilities compared to sector peers")

            # Growth metrics
            if retail_metrics.get('same_store_sales_growth'):
                if retail_metrics['same_store_sales_growth'] > 0.04:
                    swot["strengths"].append("Superior same-store sales growth indicating strong customer traction")

            # Weaknesses analysis
            # Financial weakness
            if retail_metrics.get('operating_margin'):
                if retail_metrics['operating_margin'] < 0.03:
                    swot["weaknesses"].append(
                        "Below-average operating margins suggesting potential cost control issues")

            # Inventory issues
            if retail_metrics.get('inventory_turnover'):
                if retail_metrics['inventory_turnover'] < 3:
                    swot["weaknesses"].append("Slow inventory turnover indicating potential merchandising challenges")

            # Store productivity concerns
            if retail_metrics.get('revenue_per_sqft') and subsector != 'e_commerce':
                if retail_metrics['revenue_per_sqft'] < 200:
                    swot["weaknesses"].append("Low store productivity compared to industry benchmarks")

            # SG&A efficiency
            if retail_metrics.get('sg_and_a_percentage'):
                if retail_metrics['sg_and_a_percentage'] > 0.30:
                    swot["weaknesses"].append(
                        "High SG&A expenses as percentage of revenue suggesting operational inefficiency")

            # E-commerce presence
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] < 0.10:
                    swot["weaknesses"].append("Limited e-commerce penetration in increasingly digital retail landscape")

            # Negative comp growth
            if retail_metrics.get('same_store_sales_growth'):
                if retail_metrics['same_store_sales_growth'] < 0:
                    swot["weaknesses"].append(
                        "Declining same-store sales indicate potential market share loss or demand issues")

            # Cash conversion cycle
            if retail_metrics.get('cash_conversion_cycle'):
                if retail_metrics['cash_conversion_cycle'] > 60:
                    swot["weaknesses"].append("Extended cash conversion cycle tying up working capital")

            # Opportunities analysis
            # General retail opportunities
            swot["opportunities"].append("Expansion of e-commerce capabilities and omnichannel integration")

            # Subsector-specific opportunities
            if subsector == 'apparel':
                swot["opportunities"].append(
                    "Development of sustainable/ethical clothing lines to capture growing consumer preference")
                swot["opportunities"].append("Optimization of supply chain for faster fashion cycles")

            elif subsector == 'department_store':
                swot["opportunities"].append("Reimagining store formats to create experiential retail destinations")
                swot["opportunities"].append("Strategic space optimization through partnerships and concessions")

            elif subsector == 'specialty':
                swot["opportunities"].append("Expansion of private label offerings to improve margins")
                swot["opportunities"].append(
                    "Development of service-focused revenue streams to complement product sales")

            elif subsector == 'e_commerce':
                swot["opportunities"].append("Integration of AI/ML to improve personalization and conversion")
                swot["opportunities"].append("Potential for international expansion to tap new markets")

            elif subsector == 'grocery':
                swot["opportunities"].append(
                    "Expansion of prepared foods and meal solutions to capture convenience trend")
                swot["opportunities"].append("Development of automated/contactless checkout technologies")

            elif subsector == 'discount':
                swot["opportunities"].append("Expansion into underserved urban or rural markets")
                swot["opportunities"].append("Addition of digital-enabled treasure hunt experiences to drive traffic")

            # Store growth opportunities
            if retail_metrics.get('revenue_per_sqft') and retail_metrics.get('operating_margin'):
                if retail_metrics['revenue_per_sqft'] > 300 and retail_metrics['operating_margin'] > 0.05:
                    swot["opportunities"].append(
                        "Potential for profitable store network expansion given strong unit economics")

            # E-commerce growth
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] < 0.20:
                    swot["opportunities"].append(
                        "Significant headroom for e-commerce sales growth and digital customer acquisition")

            # Threats analysis
            # Common retail threats
            swot["threats"].append("Intensifying competition from pure-play e-commerce retailers")
            swot["threats"].append("Shifting consumer preferences and spending patterns")

            # Subsector-specific threats
            if subsector == 'apparel':
                swot["threats"].append("Fast-changing fashion trends increasing inventory obsolescence risk")
                swot["threats"].append("Growth of clothing rental and second-hand markets")

            elif subsector == 'department_store':
                swot["threats"].append("Continued mall traffic declines and anchor store challenges")
                swot["threats"].append("Category-killer specialty retailers drawing away market share")

            elif subsector == 'specialty':
                swot["threats"].append("Increased competition from mass merchants expanding into specialty categories")
                swot["threats"].append("Margin pressure from comparison shopping and price transparency")

            elif subsector == 'e_commerce':
                swot["threats"].append("Rising customer acquisition costs across digital channels")
                swot["threats"].append("Increasing shipping/fulfillment costs impacting profitability")

            elif subsector == 'grocery':
                swot["threats"].append("Expansion of non-traditional food retailers (e.g., Amazon, dollar stores)")
                swot["threats"].append("Margin pressure from discounters and wholesale clubs")

            elif subsector == 'discount':
                swot["threats"].append("Potential market saturation in core geographies")
                swot["threats"].append("Rising costs for imported goods due to trade policies")

            # Financial threats
            if balance_sheet is not None:
                if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                    debt = balance_sheet.iloc[0]['Total Debt']
                    equity = balance_sheet.iloc[0]['Total Stockholder Equity']

                    if equity > 0 and debt / equity > 1.2:
                        swot["threats"].append(
                            "Higher than average debt levels may limit flexibility in economic downturns")

            # Inventory threats
            if retail_metrics.get('inventory_turnover') and retail_metrics.get('days_inventory'):
                if retail_metrics['inventory_turnover'] < 3 and retail_metrics['days_inventory'] > 120:
                    swot["threats"].append(
                        "Slow-moving inventory could lead to significant markdowns and margin erosion")

            return swot

        except Exception as e:
            logger.error(f"Error generating retail SWOT analysis: {e}")
            return {
                "strengths": ["Error generating strengths"],
                "weaknesses": ["Error generating weaknesses"],
                "opportunities": ["Error generating opportunities"],
                "threats": ["Error generating threats"],
                "error": str(e)
            }

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate valuation for a retail company using sector-specific methods

        Args:
            financial_data: Dictionary with company financial information
            valuation_method: Specific valuation method to use (if None, use most appropriate)

        Returns:
            Dictionary with valuation results
        """
        try:
            company_info = financial_data.get('company_info', {})
            subsector = self._determine_retail_subsector(company_info)

            # If method not specified, determine best method based on subsector
            if valuation_method is None:
                if subsector in ['e_commerce', 'specialty']:
                    # Growth-oriented retailers often valued using DCF
                    valuation_method = 'dcf'
                elif subsector in ['department_store', 'grocery']:
                    # Mature retailers often valued using multiples
                    valuation_method = 'multiple'
                else:
                    # Default for retail
                    valuation_method = 'dcf'

            valuation_result = {}

            # Implement different valuation methods
            if valuation_method.lower() == 'dcf':
                valuation_result = self._calculate_retail_dcf(financial_data, subsector)
            elif valuation_method.lower() == 'multiple':
                valuation_result = self._calculate_retail_multiple_valuation(financial_data, subsector)
            elif valuation_method.lower() == 'sum_of_parts':
                valuation_result = self._calculate_sum_of_parts_valuation(financial_data, subsector)
            else:
                logger.warning(f"Unknown valuation method {valuation_method}, using DCF")
                valuation_result = self._calculate_retail_dcf(financial_data, subsector)

            # Add retail-specific scenario analysis
            retail_scenarios = self._run_retail_scenarios(financial_data, subsector)
            valuation_result['scenario_analysis'] = retail_scenarios

            return valuation_result

        except Exception as e:
            logger.error(f"Error calculating retail sector valuation: {e}")
            return {"error": str(e)}


    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with retail sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results and analysis
        """
        try:
            benchmarks = self.get_sector_benchmarks()

            if not benchmarks:
                logger.warning("No retail sector benchmarks available")
                return {"warning": "No sector benchmarks available for comparison"}

            comparison_results = {}

            # Compare each metric with benchmark
            for metric, benchmark in benchmarks.items():
                if metric in company_metrics:
                    actual_value = company_metrics[metric]

                    # Determine if higher or lower is better
                    higher_is_better = True
                    if metric in ['days_inventory', 'cash_conversion_cycle', 'sg_and_a_percentage']:
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

            # Benchmarking against specific retail subsectors
            subsector_benchmarking = self._benchmark_against_subsectors(company_metrics)

            return {
                "benchmark_comparison": comparison_results,
                "subsector_benchmarking": subsector_benchmarking,
                "overall_score": overall_score,
                "summary": self._generate_retail_benchmark_summary(comparison_results, subsector_benchmarking,
                                                                   overall_score)
            }

        except Exception as e:
            logger.error(f"Error benchmarking retail company: {e}")
            return {"error": str(e)}

        # Helper methods specific to Retail sector

    def get_sector_benchmarks(self) -> Dict[str, float]:
        """
        Get benchmark values for retail sector metrics

        Returns:
            Dictionary of metric names and their benchmark values
        """
        # Use benchmarks from configuration if available
        if self.sector_name in self.sector_config and 'benchmarks' in self.sector_config[self.sector_name]:
            return self.sector_config[self.sector_name]['benchmarks']

        # Otherwise, use retail-specific defaults
        return {
            "gross_margin": 0.35,
            "operating_margin": 0.08,
            "inventory_turnover": 6.0,
            "same_store_sales_growth": 0.03,
            "sg_and_a_percentage": 0.25,
            "ecommerce_percentage": 0.15,
            "revenue_per_sqft": 400,
            "cash_conversion_cycle": 40
        }

    def _determine_retail_subsector(self, company_info: Dict[str, Any]) -> str:
        """Determine retail subsector from company information"""
        industry = company_info.get('industry', '').lower()
        description = company_info.get('description', '').lower()

        # Check each subsector's keywords
        for subsector, keywords in self.retail_subsectors.items():
            for keyword in keywords:
                if keyword in industry or keyword in description:
                    return subsector

        # Default to general retail
        return "general_retail"


    def _calculate_inventory_metrics(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> tuple:
        """Calculate inventory turnover and days inventory"""
        try:
            if income_stmt is None or balance_sheet is None:
                return None, None

            # Get cost of goods sold
            cogs = None
            if 'Cost of Revenue' in income_stmt.index:
                cogs = income_stmt.loc['Cost of Revenue'].iloc[0]

            # Get inventory
            inventory = None
            if 'Inventory' in balance_sheet.index:
                inventory = balance_sheet.loc['Inventory'].iloc[0]

            # Calculate inventory turnover
            if cogs and inventory and inventory > 0:
                inventory_turnover = cogs / inventory
                days_inventory = 365 / inventory_turnover
                return inventory_turnover, days_inventory

            return None, None
        except Exception as e:
            logger.error(f"Error calculating inventory metrics: {e}")
            return None, None


    def _estimate_same_store_sales_growth(self, income_stmt: pd.DataFrame, company_info: Dict[str, Any]) -> float:
        """
        Estimate same-store sales growth

        Note: This would typically come from company reports rather than financials
        """
        try:
            # This is a rough estimate as SSS is usually only available in company presentations
            # We'll make a rough estimate based on revenue growth
            if income_stmt is None or income_stmt.shape[1] < 2:
                return None

            # Calculate revenue growth
            if 'Total Revenue' in income_stmt.index:
                current_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                previous_revenue = income_stmt.loc['Total Revenue'].iloc[1]

                if previous_revenue > 0:
                    revenue_growth = (current_revenue / previous_revenue) - 1

                    # Estimate SSS by adjusting for new store growth if available
                    store_count_growth = self._estimate_store_count_growth(company_info)

                    if store_count_growth is not None:
                        # Roughly estimate: SSS = Total Growth - (Store Count Growth * Weight)
                        # This is a simplification; actual SSS calculation is more complex
                        estimated_sss = revenue_growth - (store_count_growth * 0.7)
                        return estimated_sss
                    else:
                        # If no store count growth data, estimate SSS as 70% of total growth
                        return revenue_growth * 0.7

            return None
        except Exception as e:
            logger.error(f"Error estimating same-store sales growth: {e}")
            return None


    def _calculate_gross_margin(self, income_stmt: pd.DataFrame) -> float:
        """Calculate gross margin"""
        try:
            if income_stmt is None:
                return None

            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gross_profit = income_stmt.loc['Gross Profit'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

                if revenue > 0:
                    return gross_profit / revenue

            return None
        except Exception as e:
            logger.error(f"Error calculating gross margin: {e}")
            return None


    def _calculate_operating_margin(self, income_stmt: pd.DataFrame) -> float:
        """Calculate operating margin"""
        try:
            if income_stmt is None:
                return None

            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

                if revenue > 0:
                    return operating_income / revenue

            return None
        except Exception as e:
            logger.error(f"Error calculating operating margin: {e}")
            return None


    def _estimate_revenue_per_sqft(self, income_stmt: pd.DataFrame, company_info: Dict[str, Any]) -> float:
        """
        Estimate revenue per square foot

        Note: Square footage is typically available in company reports, not financials
        """
        # This is a rough estimate as square footage is not typically in financial statements
        # In real implementation, this would come from company investor presentations
        try:
            # Extract revenue
            if income_stmt is None:
                return None

            revenue = None
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

            if not revenue:
                return None

            # Try to extract square footage from company description (in real implementation,
            # this would come from other sources or manual input)
            description = company_info.get('description', '').lower()
            subsector = self._determine_retail_subsector(company_info)

            # Since we don't have actual square footage, create rough estimate based on subsector
            if subsector == 'apparel':
                estimated_avg_store_size = 3500  # sq ft
            elif subsector == 'department_store':
                estimated_avg_store_size = 100000  # sq ft
            elif subsector == 'specialty':
                estimated_avg_store_size = 20000  # sq ft
            elif subsector == 'grocery':
                estimated_avg_store_size = 45000  # sq ft
            elif subsector == 'discount':
                estimated_avg_store_size = 30000  # sq ft
            else:
                estimated_avg_store_size = 25000  # sq ft

            # Rough estimate of store count based on revenue (very rough approximation)
            if revenue < 1000000000:  # < $1 billion
                estimated_store_count = 50
            elif revenue < 5000000000:  # $1-5 billion
                estimated_store_count = 200
            elif revenue < 20000000000:  # $5-20 billion
                estimated_store_count = 500
            else:  # > $20 billion
                estimated_store_count = 1000

            # Adjust for e-commerce
            if subsector == 'e_commerce':
                # For e-commerce, use employees instead of square footage
                if company_info.get('employees'):
                    return revenue / company_info.get('employees')
                return None

            estimated_total_sqft = estimated_avg_store_size * estimated_store_count

            # Calculate revenue per square foot
            if estimated_total_sqft > 0:
                return revenue / estimated_total_sqft

            return None
        except Exception as e:
            logger.error(f"Error estimating revenue per square foot: {e}")
            return None


    def _calculate_sga_percentage(self, income_stmt: pd.DataFrame) -> float:
        """Calculate SG&A as percentage of revenue"""
        try:
            if income_stmt is None:
                return None

            sga = None
            for field in ['SG&A Expense', 'Selling General and Administrative', 'Operating Expenses']:
                if field in income_stmt.index:
                    sga = income_stmt.loc[field].iloc[0]
                    break

            revenue = None
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

            if sga is not None and revenue and revenue > 0:
                return sga / revenue

            return None
        except Exception as e:
            logger.error(f"Error calculating SG&A percentage: {e}")
            return None


    def _estimate_ecommerce_percentage(self, company_info: Dict[str, Any], subsector: str) -> float:
        """
        Estimate e-commerce percentage of sales

        Note: This typically comes from company reports rather than financials
        """
        try:
            # This is a rough estimate as e-commerce % is usually only available in company presentations
            description = company_info.get('description', '').lower()

            # If it's an e-commerce company, assume high percentage
            if subsector == 'e_commerce':
                return 0.95

            # Check if description mentions e-commerce focus
            if 'online' in description and 'sales' in description:
                ecomm_indicators = ['growing online', 'e-commerce growth', 'digital sales', 'omnichannel']

                # Count how many indicators are present
                indicator_count = sum(1 for indicator in ecomm_indicators if indicator in description)

                # Use indicator count to estimate e-commerce percentage
                if indicator_count >= 3:
                    return 0.45  # High e-commerce focus
                elif indicator_count >= 2:
                    return 0.30  # Medium e-commerce focus
                elif indicator_count >= 1:
                    return 0.15  # Some e-commerce focus

            # Default estimates by subsector if no other information
            if subsector == 'apparel':
                return 0.25  # Apparel retailers typically have higher online sales
            elif subsector == 'department_store':
                return 0.20  # Department stores usually have meaningful online presence
            elif subsector == 'specialty':
                return 0.15  # Specialty retailers vary widely in online presence
            elif subsector == 'grocery':
                return 0.08  # Grocery typically has lower but growing online sales
            elif subsector == 'discount':
                return 0.10  # Discount retailers often have lower online penetration

            # Default for general retail
            return 0.15

        except Exception as e:
            logger.error(f"Error estimating e-commerce percentage: {e}")
            return None


    def _estimate_loyalty_metrics(self, company_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate loyalty program metrics

        Note: These metrics typically come from company reports rather than financials
        """
        # This would normally come from company reports, not financial statements
        # For this implementation, we'll return None as these metrics aren't typically available
        return None


    def _estimate_store_count_growth(self, company_info: Dict[str, Any]) -> float:
        """
        Estimate store count growth

        Note: This would typically come from company reports rather than financials
        """
        # This would normally come from company reports, not financial statements
        # For this implementation, we'll use a rough estimate based on company description
        try:
            description = company_info.get('description', '').lower()

            # Look for indicators of store growth in the description
            expansion_indicators = ['expansion', 'opening new stores', 'growing footprint', 'new locations']
            contraction_indicators = ['closing stores', 'reducing footprint', 'downsizing', 'store closures']

            # Count indicators
            expansion_count = sum(1 for indicator in expansion_indicators if indicator in description)
            contraction_count = sum(1 for indicator in contraction_indicators if indicator in description)

            # Determine growth direction and magnitude
            if expansion_count > contraction_count:
                if expansion_count >= 3:
                    return 0.10  # Strong expansion
                elif expansion_count >= 2:
                    return 0.05  # Moderate expansion
                else:
                    return 0.02  # Slight expansion
            elif contraction_count > expansion_count:
                if contraction_count >= 3:
                    return -0.10  # Strong contraction
                elif contraction_count >= 2:
                    return -0.05  # Moderate contraction
                else:
                    return -0.02  # Slight contraction
            else:
                return 0.0  # Stable store count

        except Exception as e:
            logger.error(f"Error estimating store count growth: {e}")
            return None


    def _calculate_cash_conversion_cycle(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate Cash Conversion Cycle (DIO + DSO - DPO)"""
        try:
            if income_stmt is None or balance_sheet is None:
                return None

            # Calculate Days Inventory Outstanding (DIO)
            inventory_turnover, days_inventory = self._calculate_inventory_metrics(income_stmt, balance_sheet)

            # Calculate Days Sales Outstanding (DSO)
            days_sales_outstanding = None
            if 'Net Receivables' in balance_sheet.index and 'Total Revenue' in income_stmt.index:
                receivables = balance_sheet.loc['Net Receivables'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

            if 'Accounts Payable' in balance_sheet.index and 'Cost of Revenue' in income_stmt.index:
                payables = balance_sheet.loc['Accounts Payable'].iloc[0]
                cogs = income_stmt.loc['Cost of Revenue'].iloc[0]

                if cogs > 0:
                    payables_turnover = cogs / payables
                    days_payable_outstanding = 365 / payables_turnover

            # Calculate Cash Conversion Cycle
            if days_inventory is not None and days_sales_outstanding is not None and days_payable_outstanding is not None:
                return days_inventory + days_sales_outstanding - days_payable_outstanding

            return None
        except Exception as e:
            logger.error(f"Error calculating cash conversion cycle: {e}")
            return None

    def _analyze_retail_subsector(self, financial_data: Dict[str, Any],
                                  subsector: str, retail_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform subsector-specific analysis"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            subsector_analysis = {
                "subsector": subsector,
                "key_metrics": [],
                "competitive_position": {},
                "growth_profile": {}
            }

            # Define key metrics by subsector
            if subsector == 'apparel':
                subsector_analysis["key_metrics"] = ["gross_margin", "inventory_turnover", "same_store_sales_growth"]

                # Analyze competitive position
                if retail_metrics.get('gross_margin'):
                    if retail_metrics['gross_margin'] > 0.45:
                        subsector_analysis["competitive_position"]["pricing_power"] = "strong"
                    elif retail_metrics['gross_margin'] > 0.35:
                        subsector_analysis["competitive_position"]["pricing_power"] = "average"
                    else:
                        subsector_analysis["competitive_position"]["pricing_power"] = "weak"

                if retail_metrics.get('inventory_turnover'):
                    if retail_metrics['inventory_turnover'] > 6:
                        subsector_analysis["competitive_position"]["inventory_management"] = "excellent"
                    elif retail_metrics['inventory_turnover'] > 4:
                        subsector_analysis["competitive_position"]["inventory_management"] = "good"
                    else:
                        subsector_analysis["competitive_position"]["inventory_management"] = "needs improvement"

                # Analyze growth profile
                if retail_metrics.get('same_store_sales_growth'):
                    if retail_metrics['same_store_sales_growth'] > 0.03:
                        subsector_analysis["growth_profile"]["sss_trajectory"] = "above sector average"
                    elif retail_metrics['same_store_sales_growth'] > 0:
                        subsector_analysis["growth_profile"]["sss_trajectory"] = "in line with sector"
                    else:
                        subsector_analysis["growth_profile"]["sss_trajectory"] = "below sector average"

            elif subsector == 'department_store':
                subsector_analysis["key_metrics"] = ["operating_margin", "inventory_turnover", "sg_and_a_percentage"]

                # Analyze competitive position
                if retail_metrics.get('sg_and_a_percentage'):
                    if retail_metrics['sg_and_a_percentage'] < 0.20:
                        subsector_analysis["competitive_position"]["cost_structure"] = "very efficient"
                    elif retail_metrics['sg_and_a_percentage'] < 0.25:
                        subsector_analysis["competitive_position"]["cost_structure"] = "efficient"
                    else:
                        subsector_analysis["competitive_position"]["cost_structure"] = "less efficient than peers"

            elif subsector == 'specialty':
                subsector_analysis["key_metrics"] = ["gross_margin", "operating_margin", "revenue_per_sqft"]

                # Analyze competitive position
                if retail_metrics.get('revenue_per_sqft'):
                    if retail_metrics['revenue_per_sqft'] > 500:
                        subsector_analysis["competitive_position"]["store_productivity"] = "exceptional"
                    elif retail_metrics['revenue_per_sqft'] > 300:
                        subsector_analysis["competitive_position"]["store_productivity"] = "strong"
                    else:
                        subsector_analysis["competitive_position"]["store_productivity"] = "average or below"

            elif subsector == 'e_commerce':
                subsector_analysis["key_metrics"] = ["gross_margin", "sg_and_a_percentage", "operating_margin"]

                # Analyze growth profile
                if income_stmt is not None and income_stmt.shape[1] >= 3:
                    revenue_growth = []
                    for i in range(min(income_stmt.shape[1] - 1, 2)):
                        if 'Total Revenue' in income_stmt.index:
                            current_rev = income_stmt.loc['Total Revenue'].iloc[i]
                            prev_rev = income_stmt.loc['Total Revenue'].iloc[i + 1]
                            if prev_rev > 0:
                                growth = (current_rev / prev_rev) - 1
                                revenue_growth.append(growth)

                    if revenue_growth:
                        avg_growth = sum(revenue_growth) / len(revenue_growth)
                        if avg_growth > 0.25:
                            subsector_analysis["growth_profile"]["revenue_trajectory"] = "high growth"
                        elif avg_growth > 0.15:
                            subsector_analysis["growth_profile"]["revenue_trajectory"] = "strong growth"
                        elif avg_growth > 0.05:
                            subsector_analysis["growth_profile"]["revenue_trajectory"] = "moderate growth"
                        else:
                            subsector_analysis["growth_profile"]["revenue_trajectory"] = "slow growth"

            elif subsector == 'grocery':
                subsector_analysis["key_metrics"] = ["gross_margin", "operating_margin", "inventory_turnover"]

                # Analyze competitive position
                if retail_metrics.get('gross_margin') and retail_metrics.get('operating_margin'):
                    gross_margin = retail_metrics['gross_margin']
                    operating_margin = retail_metrics['operating_margin']

                    if gross_margin > 0.30 and operating_margin > 0.05:
                        subsector_analysis["competitive_position"]["market_position"] = "premium/specialty grocer"
                    elif gross_margin > 0.25 and operating_margin > 0.03:
                        subsector_analysis["competitive_position"][
                            "market_position"] = "traditional grocer with strong positioning"
                    elif gross_margin < 0.20:
                        subsector_analysis["competitive_position"]["market_position"] = "discount/warehouse focus"
                    else:
                        subsector_analysis["competitive_position"]["market_position"] = "traditional grocer"

            elif subsector == 'discount':
                subsector_analysis["key_metrics"] = ["operating_margin", "revenue_per_sqft", "same_store_sales_growth"]

            # General retail analysis regardless of subsector
            # Store productivity analysis
            if retail_metrics.get('revenue_per_sqft') and retail_metrics.get('operating_margin'):
                rev_per_sqft = retail_metrics['revenue_per_sqft']
                op_margin = retail_metrics['operating_margin']

                productivity_score = (rev_per_sqft * op_margin * 100) if rev_per_sqft < 100 else (
                            rev_per_sqft * op_margin)

                if productivity_score > 30:
                    subsector_analysis["competitive_position"]["overall_productivity"] = "excellent"
                elif productivity_score > 15:
                    subsector_analysis["competitive_position"]["overall_productivity"] = "good"
                elif productivity_score > 5:
                    subsector_analysis["competitive_position"]["overall_productivity"] = "fair"
                else:
                    subsector_analysis["competitive_position"]["overall_productivity"] = "poor"

            # Growth outlook
            if retail_metrics.get('same_store_sales_growth') and retail_metrics.get('store_count_growth'):
                sss_growth = retail_metrics['same_store_sales_growth']
                store_growth = retail_metrics['store_count_growth']

                total_growth_potential = sss_growth + (store_growth if store_growth > 0 else 0)

                if total_growth_potential > 0.10:
                    subsector_analysis["growth_profile"]["overall_outlook"] = "excellent"
                elif total_growth_potential > 0.05:
                    subsector_analysis["growth_profile"]["overall_outlook"] = "good"
                elif total_growth_potential > 0.02:
                    subsector_analysis["growth_profile"]["overall_outlook"] = "moderate"
                elif total_growth_potential > 0:
                    subsector_analysis["growth_profile"]["overall_outlook"] = "slow"
                else:
                    subsector_analysis["growth_profile"]["overall_outlook"] = "declining"

            return subsector_analysis

        except Exception as e:
            logger.error(f"Error analyzing retail subsector: {e}")
            return {"error": str(e)}


    def _assess_retail_risks(self, financial_data: Dict[str, Any],
                             subsector: str, retail_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess retail-specific risks"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            risks = {}

            # 1. Inventory Risk
            inventory_risk_level = 0.5  # Default medium
            if retail_metrics.get('inventory_turnover'):
                inventory_turnover = retail_metrics['inventory_turnover']

                if inventory_turnover < 3:
                    inventory_risk_level = 0.8  # High risk
                elif inventory_turnover < 4:
                    inventory_risk_level = 0.6  # Medium-high risk
                elif inventory_turnover > 6:
                    inventory_risk_level = 0.3  # Low risk

            risks['inventory_risk'] = {
                'level': inventory_risk_level,
                'description': f"Inventory risk: {self._interpret_risk_level(inventory_risk_level)}"
            }

            # 2. Competition Risk
            competition_risk_level = 0.5  # Default medium

            # Use margins to assess competitive pressure
            if retail_metrics.get('gross_margin') and retail_metrics.get('operating_margin'):
                gross_margin = retail_metrics['gross_margin']
                operating_margin = retail_metrics['operating_margin']

                # Low margins often indicate high competition
                if operating_margin < 0.03:
                    competition_risk_level = 0.8  # High risk
                elif gross_margin < 0.25 and operating_margin < 0.05:
                    competition_risk_level = 0.7  # Medium-high risk
                elif gross_margin > 0.40 and operating_margin > 0.10:
                    competition_risk_level = 0.3  # Low risk

            # Adjust based on subsector
            if subsector == 'department_store':
                competition_risk_level = min(0.8, competition_risk_level + 0.2)  # Increase risk (challenged sector)
            elif subsector == 'grocery':
                competition_risk_level = min(0.7, competition_risk_level + 0.1)  # Slightly increase risk

            risks['competition_risk'] = {
                'level': competition_risk_level,
                'description': f"Competition risk: {self._interpret_risk_level(competition_risk_level)}"
            }

            # 3. E-commerce Disruption Risk
            ecommerce_risk_level = 0.5  # Default medium

            # Based on estimated e-commerce percentage and subsector
            if retail_metrics.get('ecommerce_percentage'):
                ecommerce_percentage = retail_metrics['ecommerce_percentage']

                if subsector != 'e_commerce':  # If not an e-commerce company
                    if ecommerce_percentage < 0.10:
                        ecommerce_risk_level = 0.8  # High risk - low online presence
                    elif ecommerce_percentage < 0.20:
                        ecommerce_risk_level = 0.6  # Medium-high risk
                    elif ecommerce_percentage > 0.30:
                        ecommerce_risk_level = 0.3  # Low risk - strong online presence
                else:
                    ecommerce_risk_level = 0.2  # Low risk for e-commerce companies

            # Adjust based on subsector vulnerability
            if subsector == 'apparel' or subsector == 'specialty':
                ecommerce_risk_level = min(0.9, ecommerce_risk_level + 0.1)  # Increase risk (vulnerable sectors)
            elif subsector == 'grocery':
                ecommerce_risk_level = max(0.2, ecommerce_risk_level - 0.1)  # Decrease risk (less vulnerable)

            risks['ecommerce_disruption_risk'] = {
                'level': ecommerce_risk_level,
                'description': f"E-commerce disruption risk: {self._interpret_risk_level(ecommerce_risk_level)}"
            }

            # 4. Store Productivity Risk
            store_risk_level = 0.5  # Default medium

            if retail_metrics.get('revenue_per_sqft') and subsector != 'e_commerce':
                rev_per_sqft = retail_metrics['revenue_per_sqft']

                if subsector == 'apparel':
                    if rev_per_sqft < 300:
                        store_risk_level = 0.7  # High risk
                    elif rev_per_sqft > 600:
                        store_risk_level = 0.3  # Low risk
                elif subsector == 'department_store':
                    if rev_per_sqft < 200:
                        store_risk_level = 0.7  # High risk
                    elif rev_per_sqft > 400:
                        store_risk_level = 0.3  # Low risk
                elif subsector == 'specialty':
                    if rev_per_sqft < 250:
                        store_risk_level = 0.7  # High risk
                    elif rev_per_sqft > 500:
                        store_risk_level = 0.3  # Low risk
                elif subsector == 'grocery':
                    if rev_per_sqft < 400:
                        store_risk_level = 0.7  # High risk
                    elif rev_per_sqft > 600:
                        store_risk_level = 0.3  # Low risk

            risks['store_productivity_risk'] = {
                'level': store_risk_level,
                'description': f"Store productivity risk: {self._interpret_risk_level(store_risk_level)}"
            }

            # 5. Debt Risk
            debt_risk_level = 0.4  # Default medium-low

            if balance_sheet is not None:
                if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                    debt = balance_sheet.loc['Total Debt'].iloc[0]
                    equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]

                    if equity > 0:
                        debt_to_equity = debt / equity

                        if debt_to_equity > 1.5:
                            debt_risk_level = 0.8  # High risk
                        elif debt_to_equity > 1.0:
                            debt_risk_level = 0.6  # Medium-high risk
                        elif debt_to_equity < 0.5:
                            debt_risk_level = 0.2  # Low risk

            risks['debt_risk'] = {
                'level': debt_risk_level,
                'description': f"Debt risk: {self._interpret_risk_level(debt_risk_level)}"
            }

            # Calculate overall risk score
            risk_weights = {
                'inventory_risk': 0.20,
                'competition_risk': 0.25,
                'ecommerce_disruption_risk': 0.25,
                'store_productivity_risk': 0.15,
                'debt_risk': 0.15
            }

            weighted_risk = 0
            for risk_type, weight in risk_weights.items():
                if risk_type in risks:
                    weighted_risk += risks[risk_type]['level'] * weight

            risks['overall_risk'] = {
                'level': weighted_risk,
                'description': f"Overall risk: {self._interpret_risk_level(weighted_risk)}"
            }

            return risks

        except Exception as e:
            logger.error(f"Error assessing retail risks: {e}")
            return {"error": str(e)}


    def _interpret_risk_level(self, risk_level: float) -> str:
        """Convert numerical risk level to description"""
        if risk_level > 0.7:
            return "High"
        elif risk_level > 0.5:
            return "Medium-High"
        elif risk_level > 0.3:
            return "Medium"
        elif risk_level > 0.1:
            return "Medium-Low"
        else:
            return "Low"


    def _generate_retail_swot(self, financial_data: Dict[str, Any],
                              subsector: str, retail_metrics: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate a SWOT analysis for a retail company"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            swot = {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }

            # Strengths analysis
            # Financial strength
            if retail_metrics.get('operating_margin'):
                if retail_metrics['operating_margin'] > 0.08:
                    swot["strengths"].append("Strong operating margins compared to retail sector peers")

            if retail_metrics.get('gross_margin'):
                if retail_metrics['gross_margin'] > 0.40:
                    swot["strengths"].append("Premium pricing power and brand strength reflected in high gross margins")

            # Inventory management
            if retail_metrics.get('inventory_turnover'):
                if retail_metrics['inventory_turnover'] > 5:
                    swot["strengths"].append("Efficient inventory management with above-average turnover")

            # Store productivity
            if retail_metrics.get('revenue_per_sqft'):
                if retail_metrics['revenue_per_sqft'] > 500:
                    swot["strengths"].append("Exceptional store productivity with high revenue per square foot")
                elif retail_metrics['revenue_per_sqft'] > 300:
                    swot["strengths"].append("Strong store productivity")

            # E-commerce presence
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] > 0.25:
                    swot["strengths"].append("Well-developed e-commerce capabilities compared to sector peers")

            # Growth metrics
            if retail_metrics.get('same_store_sales_growth'):
                if retail_metrics['same_store_sales_growth'] > 0.04:
                    swot["strengths"].append("Superior same-store sales growth indicating strong customer traction")

            # Weaknesses analysis
            # Financial weakness
            if retail_metrics.get('operating_margin'):
                if retail_metrics['operating_margin'] < 0.03:
                    swot["weaknesses"].append("Below-average operating margins suggesting potential cost control issues")

            # Inventory issues
            if retail_metrics.get('inventory_turnover'):
                if retail_metrics['inventory_turnover'] < 3:
                    swot["weaknesses"].append("Slow inventory turnover indicating potential merchandising challenges")

            # Store productivity concerns
            if retail_metrics.get('revenue_per_sqft') and subsector != 'e_commerce':
                if retail_metrics['revenue_per_sqft'] < 200:
                    swot["weaknesses"].append("Low store productivity compared to industry benchmarks")

            # SG&A efficiency
            if retail_metrics.get('sg_and_a_percentage'):
                if retail_metrics['sg_and_a_percentage'] > 0.30:
                    swot["weaknesses"].append(
                        "High SG&A expenses as percentage of revenue suggesting operational inefficiency")

            # E-commerce presence
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] < 0.10:
                    swot["weaknesses"].append("Limited e-commerce penetration in increasingly digital retail landscape")

            # Negative comp growth
            if retail_metrics.get('same_store_sales_growth'):
                if retail_metrics['same_store_sales_growth'] < 0:
                    swot["weaknesses"].append(
                        "Declining same-store sales indicate potential market share loss or demand issues")

            # Cash conversion cycle
            if retail_metrics.get('cash_conversion_cycle'):
                if retail_metrics['cash_conversion_cycle'] > 60:
                    swot["weaknesses"].append("Extended cash conversion cycle tying up working capital")

            # Opportunities analysis
            # General retail opportunities
            swot["opportunities"].append("Expansion of e-commerce capabilities and omnichannel integration")

            # Subsector-specific opportunities
            if subsector == 'apparel':
                swot["opportunities"].append(
                    "Development of sustainable/ethical clothing lines to capture growing consumer preference")
                swot["opportunities"].append("Optimization of supply chain for faster fashion cycles")

            elif subsector == 'department_store':
                swot["opportunities"].append("Reimagining store formats to create experiential retail destinations")
                swot["opportunities"].append("Strategic space optimization through partnerships and concessions")

            elif subsector == 'specialty':
                swot["opportunities"].append("Expansion of private label offerings to improve margins")
                swot["opportunities"].append("Development of service-focused revenue streams to complement product sales")

            elif subsector == 'e_commerce':
                swot["opportunities"].append("Integration of AI/ML to improve personalization and conversion")
                swot["opportunities"].append("Potential for international expansion to tap new markets")

            elif subsector == 'grocery':
                swot["opportunities"].append("Expansion of prepared foods and meal solutions to capture convenience trend")
                swot["opportunities"].append("Development of automated/contactless checkout technologies")

            elif subsector == 'discount':
                swot["opportunities"].append("Expansion into underserved urban or rural markets")
                swot["opportunities"].append("Addition of digital-enabled treasure hunt experiences to drive traffic")

            # Store growth opportunities
            if retail_metrics.get('revenue_per_sqft') and retail_metrics.get('operating_margin'):
                if retail_metrics['revenue_per_sqft'] > 300 and retail_metrics['operating_margin'] > 0.05:
                    swot["opportunities"].append(
                        "Potential for profitable store network expansion given strong unit economics")

            # E-commerce growth
            if retail_metrics.get('ecommerce_percentage') and subsector != 'e_commerce':
                if retail_metrics['ecommerce_percentage'] < 0.20:
                    swot["opportunities"].append(
                        "Significant headroom for e-commerce sales growth and digital customer acquisition")

            # Threats analysis
            # Common retail threats
            swot["threats"].append("Intensifying competition from pure-play e-commerce retailers")
            swot["threats"].append("Shifting consumer preferences and spending patterns")

            # Subsector-specific threats
            if subsector == 'apparel':
                swot["threats"].append("Fast-changing fashion trends increasing inventory obsolescence risk")
                swot["threats"].append("Growth of clothing rental and second-hand markets")

            elif subsector == 'department_store':
                swot["threats"].append("Continued mall traffic declines and anchor store challenges")
                swot["threats"].append("Category-killer specialty retailers drawing away market share")

            elif subsector == 'specialty':
                swot["threats"].append("Increased competition from mass merchants expanding into specialty categories")
                swot["threats"].append("Margin pressure from comparison shopping and price transparency")

            elif subsector == 'e_commerce':
                swot["threats"].append("Rising customer acquisition costs across digital channels")
                swot["threats"].append("Increasing shipping/fulfillment costs impacting profitability")

            elif subsector == 'grocery':
                swot["threats"].append("Expansion of non-traditional food retailers (e.g., Amazon, dollar stores)")
                swot["threats"].append("Margin pressure from discounters and wholesale clubs")

            elif subsector == 'discount':
                swot["threats"].append("Potential market saturation in core geographies")
                swot["threats"].append("Rising costs for imported goods due to trade policies")

            # Financial threats
            if balance_sheet is not None:
                if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                    debt = balance_sheet.loc['Total Debt'].iloc[0]
                    equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]

                    if equity > 0 and debt / equity > 1.2:
                        swot["threats"].append(
                            "Higher than average debt levels may limit flexibility in economic downturns")

            # Inventory threats
            if retail_metrics.get('inventory_turnover') and retail_metrics.get('days_inventory'):
                if retail_metrics['inventory_turnover'] < 3 and retail_metrics['days_inventory'] > 120:
                    swot["threats"].append("Slow-moving inventory could lead to significant markdowns and margin erosion")

            return swot

        except Exception as e:
            logger.error(f"Error generating retail SWOT analysis: {e}")
            return {
                "strengths": ["Error generating strengths"],
                "weaknesses": ["Error generating weaknesses"],
                "opportunities": ["Error generating opportunities"],
                "threats": ["Error generating threats"],
                "error": str(e)
            }


    def _calculate_retail_dcf(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Calculate DCF valuation specifically tailored for retail companies"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or balance_sheet is None or cash_flow is None:
                raise ValueError("Missing required financial statements for DCF valuation")

            # Determine appropriate DCF parameters based on retail subsector
            forecast_years = 5  # Default
            terminal_growth = 0.02  # Default

            if subsector == 'e_commerce':
                forecast_years = 7  # Longer forecast for high-growth e-commerce
                terminal_growth = 0.03  # Higher long-term growth
            elif subsector == 'department_store':
                forecast_years = 4  # Shorter forecast for challenged subsector
                terminal_growth = 0.01  # Lower long-term growth
            elif subsector == 'discount':
                forecast_years = 6  # Longer forecast for expanding discounters
                terminal_growth = 0.025  # Slightly higher growth

            # Calculate WACC - Weighted Average Cost of Capital
            # This is a simplified version; a real implementation would be more sophisticated
            risk_free_rate = self.dcf_parameters.get('risk_free_rate', 0.03)

            # Adjust equity risk premium based on subsector
            if subsector == 'department_store':
                equity_risk_premium = 0.06  # Higher risk premium
            elif subsector == 'e_commerce':
                equity_risk_premium = 0.055  # Slightly higher premium
            else:
                equity_risk_premium = 0.05  # Standard premium

            # Get beta from company info or estimate
            beta = company_info.get('beta', None)
            if beta is None:
                # Estimate beta based on subsector
                if subsector == 'department_store':
                    beta = 1.3  # Higher volatility
                elif subsector == 'grocery':
                    beta = 0.8  # Lower volatility
                elif subsector == 'e_commerce':
                    beta = 1.4  # Higher volatility
                else:
                    beta = 1.1  # Average retail beta

            # Cost of equity using CAPM
            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # Cost of debt (simplified)
            pretax_cost_of_debt = 0.045  # Assumed based on average retail borrowing costs
            tax_rate = 0.25  # Assumed tax rate
            after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)

            # Calculate weights for equity and debt
            if 'Total Debt' in balance_sheet.index and company_info.get('market_cap'):
                debt = balance_sheet.loc['Total Debt'].iloc[0]
                equity = company_info['market_cap']

                total_capital = debt + equity
                weight_debt = debt / total_capital
                weight_equity = equity / total_capital
            else:
                # Estimate capital structure based on subsector
                if subsector == 'department_store':
                    weight_debt = 0.4
                    weight_equity = 0.6
                elif subsector == 'e_commerce':
                    weight_debt = 0.2
                    weight_equity = 0.8
                else:
                    weight_debt = 0.3
                    weight_equity = 0.7

            # Calculate WACC
            wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

            # Estimate free cash flow
            historical_fcf = []
            historical_years = min(3, cash_flow.shape[1])

            for i in range(historical_years):
                if 'Free Cash Flow' in cash_flow.index:
                    historical_fcf.append(cash_flow.loc['Free Cash Flow'].iloc[i])
                elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                    operating_cf = cash_flow.loc['Operating Cash Flow'].iloc[i]
                    capex = cash_flow.loc['Capital Expenditure'].iloc[i]  # Note: capex is usually negative

            # Estimate FCF growth rate
            if len(historical_fcf) >= 2:
                growth_rates = []
                for i in range(len(historical_fcf) - 1):
                    if historical_fcf[i + 1] != 0:
                        growth_rate = (historical_fcf[i] / historical_fcf[i + 1]) - 1
                        growth_rates.append(growth_rate)

                if growth_rates:
                    initial_growth = sum(growth_rates) / len(growth_rates)
                    # Cap growth rate
                    initial_growth = max(0.01, min(0.15, initial_growth))
                else:
                    initial_growth = 0.03  # Default growth rate
            else:
                initial_growth = 0.03  # Default growth rate

            # For retail, adjust growth rate based on same-store sales and store growth
            retail_metrics = {}
            inventory_turnover, days_inventory = self._calculate_inventory_metrics(income_stmt, balance_sheet)
            retail_metrics['inventory_turnover'] = inventory_turnover
            retail_metrics['days_inventory'] = days_inventory
            sss_growth = self._estimate_same_store_sales_growth(income_stmt, company_info)
            retail_metrics['same_store_sales_growth'] = sss_growth
            store_count_growth = self._estimate_store_count_growth(company_info)
            retail_metrics['store_count_growth'] = store_count_growth

            # Adjust initial growth based on retail-specific metrics
            if sss_growth is not None:
                # Weight SSS growth more heavily in the growth calculation
                initial_growth = (initial_growth * 0.4) + (sss_growth * 0.6)

            if store_count_growth is not None and store_count_growth > 0:
                # Add a portion of store growth to the initial growth rate
                initial_growth += (store_count_growth * 0.3)

            # Forecast future FCF
            latest_fcf = historical_fcf[0] if historical_fcf else None

            if latest_fcf is None:
                # Try to estimate FCF from income statement and balance sheet
                if 'Net Income' in income_stmt.index:
                    net_income = income_stmt.loc['Net Income'].iloc[0]
                    # Rough estimation of FCF as a percentage of net income
                    latest_fcf = net_income * 0.8  # Simplistic assumption

            if latest_fcf is None:
                raise ValueError("Unable to determine base free cash flow for forecasting")

            # Generate FCF forecast with declining growth
            forecasted_fcf = []
            terminal_year_fcf = 0

            for year in range(1, forecast_years + 1):
                # Calculate year-specific growth rate (declining over time)
                year_growth = initial_growth * (1 - (year - 1) / forecast_years)
                year_fcf = latest_fcf * (1 + year_growth) ** year
                forecasted_fcf.append(year_fcf)

                if year == forecast_years:
                    terminal_year_fcf = year_fcf

            # Calculate terminal value
            terminal_value = (terminal_year_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)

            # Discount FCF to present value
            present_value_fcf = sum([fcf / ((1 + wacc) ** year) for year, fcf in enumerate(forecasted_fcf, 1)])
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            # Calculate enterprise value and equity value
            enterprise_value = present_value_fcf + present_value_terminal

            # Calculate net debt
            if 'Total Debt' in balance_sheet.index and 'Cash and Cash Equivalents' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]
                net_debt = total_debt - cash
            else:
                net_debt = 0

            equity_value = enterprise_value - net_debt

            # Calculate per share value
            shares_outstanding = company_info.get('shares_outstanding')

            if shares_outstanding:
                per_share_value = equity_value / shares_outstanding
            else:
                # Try to estimate shares from market cap and share price
                market_cap = company_info.get('market_cap')
                share_price = company_info.get('share_price')

                if market_cap and share_price and share_price > 0:
                    shares_outstanding = market_cap / share_price
                    per_share_value = equity_value / shares_outstanding
                else:
                    per_share_value = None

            # Return detailed DCF results
            return {
                "valuation_method": "dcf",
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "per_share_value": per_share_value,
                "wacc": wacc,
                "cost_of_equity": cost_of_equity,
                "cost_of_debt": after_tax_cost_of_debt,
                "beta": beta,
                "risk_free_rate": risk_free_rate,
                "equity_risk_premium": equity_risk_premium,
                "initial_growth_rate": initial_growth,
                "terminal_growth_rate": terminal_growth,
                "forecast_years": forecast_years,
                "forecasted_fcf": forecasted_fcf,
                "terminal_value": terminal_value,
                "present_value_fcf": present_value_fcf,
                "present_value_terminal": present_value_terminal,
                "net_debt": net_debt,
                "retail_adjustments": {
                    "same_store_sales_growth": sss_growth,
                    "store_count_growth": store_count_growth
                }
            }

        except Exception as e:
            logger.error(f"Error in retail DCF calculation: {e}")
            return {
                "valuation_method": "dcf",
                "error": str(e),
                "enterprise_value": None,
                "equity_value": None,
                "per_share_value": None
            }

    def _calculate_retail_multiple_valuation(self, financial_data: Dict[str, Any], subsector: str) -> Dict[str, Any]:
        """Calculate valuation using retail-specific multiples"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for multiple valuation")

            # Define appropriate multiples based on retail subsector
            pe_multiple = None
            ev_ebitda_multiple = None
            ev_sales_multiple = None

            if subsector == 'apparel':
                pe_multiple = 18.0
                ev_ebitda_multiple = 9.0
                ev_sales_multiple = 1.2
            elif subsector == 'department_store':
                pe_multiple = 12.0
                ev_ebitda_multiple = 6.5
                ev_sales_multiple = 0.5
            elif subsector == 'specialty':
                pe_multiple = 16.0
                ev_ebitda_multiple = 8.5
                ev_sales_multiple = 1.0
            elif subsector == 'e_commerce':
                pe_multiple = 30.0
                ev_ebitda_multiple = 15.0
                ev_sales_multiple = 2.0
            elif subsector == 'grocery':
                pe_multiple = 14.0
                ev_ebitda_multiple = 7.0
                ev_sales_multiple = 0.5
            elif subsector == 'discount':
                pe_multiple = 20.0
                ev_ebitda_multiple = 10.0
                ev_sales_multiple = 0.8
            else:  # default for general retail
                pe_multiple = 16.0
                ev_ebitda_multiple = 8.0
                ev_sales_multiple = 1.0

            # Get required financial metrics
            earnings = None
            if 'Net Income' in income_stmt.index:
                earnings = income_stmt.loc['Net Income'].iloc[0]

            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]

            if 'EBITDA' in income_stmt.index:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
            elif 'Operating Income' in income_stmt.index and 'Depreciation & Amortization' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].iloc[0]
                depreciation = income_stmt.loc['Depreciation & Amortization'].iloc[0]
                ebitda = operating_income + depreciation

            if 'Total Debt' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]

            if 'Cash and Cash Equivalents' in balance_sheet.index:
                cash = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]

            # Calculate valuations based on different multiples
            valuations = {}
            per_share_valuations = {}

            shares_outstanding = company_info.get('shares_outstanding')

            # P/E-based valuation
            if earnings is not None and pe_multiple is not None:
                pe_equity_value = earnings * pe_multiple
                valuations['pe_based'] = pe_equity_value

                if shares_outstanding:
                    per_share_valuations['pe_based'] = pe_equity_value / shares_outstanding

            # EV/EBITDA-based valuation
            if ebitda is not None and ev_ebitda_multiple is not None:
                ev_ebitda_enterprise_value = ebitda * ev_ebitda_multiple
                ev_ebitda_equity_value = ev_ebitda_enterprise_value - total_debt + cash
                valuations['ev_ebitda_based'] = ev_ebitda_equity_value

                if shares_outstanding:
                    per_share_valuations['ev_ebitda_based'] = ev_ebitda_equity_value / shares_outstanding

            # EV/Sales-based valuation
            if revenue is not None and ev_sales_multiple is not None:
                ev_sales_enterprise_value = revenue * ev_sales_multiple
                ev_sales_equity_value = ev_sales_enterprise_value - total_debt + cash
                valuations['ev_sales_based'] = ev_sales_equity_value

                if shares_outstanding:
                    per_share_valuations['ev_sales_based'] = ev_sales_equity_value / shares_outstanding

            # For retail companies, add specific metrics
            retail_metrics = {}
            inventory_turnover, days_inventory = self._calculate_inventory_metrics(income_stmt, balance_sheet)
            retail_metrics['inventory_turnover'] = inventory_turnover
            retail_metrics['days_inventory'] = days_inventory

            # Calculate EV/Locations for store-based retailers
            store_count = self._estimate_store_count(company_info, revenue, subsector)
            if store_count and store_count > 0 and subsector != 'e_commerce':
                ev_store_enterprise_value = None

                # Use the EV/EBITDA-based enterprise value if available
                if 'ev_ebitda_based' in valuations:
                    ev_store_enterprise_value = ev_ebitda_enterprise_value
                # Otherwise use the EV/Sales-based enterprise value
                elif 'ev_sales_based' in valuations:
                    ev_store_enterprise_value = ev_sales_enterprise_value

                if ev_store_enterprise_value:
                    valuations['ev_per_store'] = ev_store_enterprise_value / store_count
                    retail_metrics['ev_per_store'] = ev_store_enterprise_value / store_count

            # Calculate weighted average valuation
            # Weight each method based on subsector characteristics
            if subsector == 'apparel' or subsector == 'specialty':
                weights = {
                    'pe_based': 0.35,
                    'ev_ebitda_based': 0.40,
                    'ev_sales_based': 0.25
                }
            elif subsector == 'department_store' or subsector == 'grocery':
                weights = {
                    'pe_based': 0.25,
                    'ev_ebitda_based': 0.45,
                    'ev_sales_based': 0.30
                }
            elif subsector == 'e_commerce':
                weights = {
                    'pe_based': 0.20,
                    'ev_ebitda_based': 0.30,
                    'ev_sales_based': 0.50
                }
            elif subsector == 'discount':
                weights = {
                    'pe_based': 0.30,
                    'ev_ebitda_based': 0.40,
                    'ev_sales_based': 0.30
                }
            else:  # default for general retail
                weights = {
                    'pe_based': 0.33,
                    'ev_ebitda_based': 0.34,
                    'ev_sales_based': 0.33
                }

            # Calculate weighted average equity value
            weighted_equity_value = 0
            total_weight = 0

            for method, value in valuations.items():
                if method in weights and value is not None:
                    weighted_equity_value += value * weights[method]
                    total_weight += weights[method]

            # Normalize if not all methods were available
            if total_weight > 0:
                weighted_equity_value = weighted_equity_value / total_weight
                if shares_outstanding:
                    weighted_per_share_value = weighted_equity_value / shares_outstanding
                else:
                    weighted_per_share_value = None
            else:
                weighted_equity_value = None
                weighted_per_share_value = None

            # Return detailed multiple valuation results
            return {
                "valuation_method": "multiple",
                "multiples_used": {
                    "pe_multiple": pe_multiple,
                    "ev_ebitda_multiple": ev_ebitda_multiple,
                    "ev_sales_multiple": ev_sales_multiple
                },
                "valuations": valuations,
                "per_share_valuations": per_share_valuations,
                "weighted_equity_value": weighted_equity_value,
                "weighted_per_share_value": weighted_per_share_value,
                "retail_metrics": retail_metrics,
                "estimated_store_count": store_count
            }

        except Exception as e:
            logger.error(f"Error in retail multiple valuation: {e}")
            return {
                "valuation_method": "multiple",
                "error": str(e),
                "weighted_equity_value": None,
                "weighted_per_share_value": None
            }

# Register the RetailSectorAnalyzer with the factory
SectorFactory.register_analyzer("Consumer Discretionary", RetailSectorAnalyzer)