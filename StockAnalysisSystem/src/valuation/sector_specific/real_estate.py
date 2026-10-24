import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple


# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.valuation.sector_factor import SectorAnalyzer, SectorFactory
from StockAnalysisSystem.src.config import SECTOR_METRICS, SECTOR_DCF_PARAMETERS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_estate_sector')


class RealEstateSectorAnalyzer(SectorAnalyzer):
    """
    Real Estate Sector Analyzer for REITs, real estate developers, and property management companies.

    Real estate companies have unique characteristics requiring special analysis:
    1. Net Asset Value (NAV) is a primary valuation metric, especially for REITs
    2. Funds From Operations (FFO) is used instead of earnings (EPS)
    3. Focus on recurring income streams and lease terms
    4. High depreciation expenses which distort traditional earnings
    5. Capital-intensive with high debt levels but stable cash flows
    6. Dividend yields are typically higher than other sectors
    """

    def __init__(self, sector_name: str = "Real Estate"):
        """Initialize the real estate sector analyzer"""
        super().__init__(sector_name)
        logger.info("Initialized Real Estate Sector Analyzer")

        # Additional real estate sector specific benchmarks if not in config
        self.real_estate_benchmarks = {
            "reit": {
                "ffo_yield": 0.08,  # 8% FFO yield is typical
                "affo_yield": 0.07,  # 7% AFFO yield
                "dividend_yield": 0.05,  # 5% dividend yield
                "p_nav": 0.95,  # Price to NAV ratio (0.95 = 5% discount to NAV)
                "debt_to_assets": 0.50,  # 50% debt to total assets
                "interest_coverage": 3.0,  # 3x interest coverage ratio
                "occupancy_rate": 0.95  # 95% occupancy rate
            },
            "developer": {
                "p_nav": 0.85,  # Developers often trade at larger discount to NAV
                "roe": 0.12,  # 12% return on equity
                "debt_to_equity": 1.2,  # Higher leverage
                "pre_sale_rate": 0.60,  # 60% pre-sale rate for developments
                "margin_on_cost": 0.25  # 25% profit margin on development cost
            },
            "property_manager": {
                "aum_growth": 0.08,  # 8% growth in assets under management
                "management_fee_margin": 0.65,  # 65% margin on management fees
                "retention_rate": 0.85,  # 85% client retention rate
                "recurring_revenue": 0.80  # 80% of revenue should be recurring
            }
        }

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a real estate company using sector-specific metrics

        Args:
            financial_data: Dictionary containing financial statements and other data

        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for analysis")

            # Company info for determining subsector
            company_info = financial_data.get('company_info', {})
            company_name = company_info.get('name', 'Unknown')
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # Initialize analysis results
            analysis = {
                'company_name': company_name,
                'sector': 'Real Estate',
                'subsector': subsector,
                'metrics': {},
                'performance_assessment': {},
                'key_strengths': [],
                'key_weaknesses': [],
                'risk_factors': []
            }

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            latest_cash_flow = cash_flow.iloc[:, 0] if (cash_flow is not None and not cash_flow.empty) else pd.Series()

            # Calculate general financial ratios and real estate specific metrics
            analysis['metrics'] = self._calculate_real_estate_metrics(
                latest_income, latest_balance, latest_cash_flow, subsector
            )

            # Evaluate performance against benchmarks
            analysis['performance_assessment'] = self._assess_performance(analysis['metrics'], subsector)

            # Identify strengths and weaknesses
            analysis['key_strengths'], analysis['key_weaknesses'] = self._identify_strengths_weaknesses(
                analysis['performance_assessment']
            )

            # Analyze risk factors
            analysis['risk_factors'] = self._analyze_risk_factors(
                financial_data, analysis['metrics'], subsector
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing real estate company: {e}")
            return {'error': str(e)}

    def calculate_valuation(self, financial_data: Dict[str, Any],
                            valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate real estate company valuation using sector-appropriate methods

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_method: Specific valuation method to use (if None, uses the most appropriate)

        Returns:
            Dictionary with valuation results
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for valuation")

            # Determine subsector
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # Determine valuation method if not specified
            if valuation_method is None:
                # For real estate firms, different methods are preferred for different subsectors
                if subsector == 'reit':
                    valuation_method = 'nav'  # Net Asset Value is primary for REITs
                elif subsector == 'developer':
                    valuation_method = 'adjusted_nav'  # Adjusted NAV for developers with pipeline projects
                elif subsector == 'property_manager':
                    valuation_method = 'discounted_cash_flow'  # DCF more appropriate for service businesses
                else:
                    valuation_method = 'nav'  # Default for real estate

            # Initialize valuation result
            valuation_result = {
                'company_name': company_info.get('name', 'Unknown'),
                'ticker': company_info.get('symbol', 'Unknown'),
                'valuation_method': valuation_method,
                'estimated_value_per_share': None,
                'current_price': market_data.get('share_price'),
                'price_to_value_ratio': None,
                'valuation_assessment': None,
                'valuation_details': {},
                'assumptions': {},
                'sensitivity': {}
            }

            # Calculate valuation based on selected method
            if valuation_method == 'nav':
                valuation_result = self._nav_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'ffo_multiple':
                valuation_result = self._ffo_multiple_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'adjusted_nav':
                valuation_result = self._adjusted_nav_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'discounted_cash_flow':
                valuation_result = self._real_estate_dcf_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'dividend_discount':
                valuation_result = self._dividend_discount_valuation(
                    financial_data, valuation_result
                )
            else:
                # Default to a combination of methods for real estate firms
                return self._combined_valuation_approach(financial_data)

            # Add price-to-value assessment
            if (valuation_result['estimated_value_per_share'] is not None and
                    valuation_result['current_price'] is not None):

                valuation_result['price_to_value_ratio'] = (
                        valuation_result['current_price'] / valuation_result['estimated_value_per_share']
                )

                # Add assessment
                if valuation_result['price_to_value_ratio'] < 0.8:
                    valuation_result['valuation_assessment'] = "Significantly undervalued"
                elif valuation_result['price_to_value_ratio'] < 0.95:
                    valuation_result['valuation_assessment'] = "Moderately undervalued"
                elif valuation_result['price_to_value_ratio'] < 1.05:
                    valuation_result['valuation_assessment'] = "Fairly valued"
                elif valuation_result['price_to_value_ratio'] < 1.2:
                    valuation_result['valuation_assessment'] = "Moderately overvalued"
                else:
                    valuation_result['valuation_assessment'] = "Significantly overvalued"

            return valuation_result

        except Exception as e:
            logger.error(f"Error calculating real estate company valuation: {e}")
            return {
                'error': str(e),
                'valuation_method': valuation_method or 'unknown'
            }

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with real estate sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        try:
            # Determine subsector if available in metrics
            subsector = company_metrics.get('subsector', 'reit')  # Default to REIT

            # Get appropriate benchmarks
            sector_benchmarks = self.get_sector_benchmarks()
            subsector_benchmarks = self.real_estate_benchmarks.get(subsector, {})

            # Combine benchmarks
            benchmarks = {**sector_benchmarks, **subsector_benchmarks}

            # Initialize benchmark comparison
            benchmark_comparison = {
                'metrics_vs_benchmarks': {},
                'overall_score': 0,
                'percentile_rank': {},
                'quartile': {},
                'assessment': {}
            }

            # Compare each metric with benchmark
            total_score = 0
            metrics_count = 0

            for metric, value in company_metrics.items():
                if metric in benchmarks and isinstance(value, (int, float)):
                    benchmark_value = benchmarks[metric]

                    # Calculate percentage difference
                    if benchmark_value != 0:
                        percent_diff = ((value - benchmark_value) / benchmark_value) * 100
                    else:
                        percent_diff = 0

                    # Determine if higher or lower is better for this metric
                    higher_is_better = True
                    if metric in ['debt_to_assets', 'debt_to_equity', 'cap_rate', 'vacancy_rate']:
                        higher_is_better = False

                    # Calculate score (0-100)
                    if higher_is_better:
                        if value >= benchmark_value:
                            score = min(100, 50 + (percent_diff / 2))  # Cap at 100
                        else:
                            score = max(0, 50 - (abs(percent_diff) / 2))  # Floor at 0
                    else:
                        if value <= benchmark_value:
                            score = min(100, 50 + (abs(percent_diff) / 2))  # Cap at 100
                        else:
                            score = max(0, 50 - (percent_diff / 2))  # Floor at 0

                    # Determine assessment
                    if score >= 80:
                        assessment = "Excellent"
                        quartile = 1
                    elif score >= 60:
                        assessment = "Good"
                        quartile = 2
                    elif score >= 40:
                        assessment = "Average"
                        quartile = 3
                    else:
                        assessment = "Below Average"
                        quartile = 4

                    # Add to comparison
                    benchmark_comparison['metrics_vs_benchmarks'][metric] = {
                        'company_value': value,
                        'benchmark_value': benchmark_value,
                        'percent_diff': percent_diff,
                        'score': score,
                        'higher_is_better': higher_is_better,
                        'assessment': assessment
                    }

                    benchmark_comparison['quartile'][metric] = quartile
                    benchmark_comparison['assessment'][metric] = assessment

                    # Add to overall score
                    total_score += score
                    metrics_count += 1

            # Calculate overall score
            if metrics_count > 0:
                benchmark_comparison['overall_score'] = total_score / metrics_count

            # Add overall assessment
            if benchmark_comparison['overall_score'] >= 80:
                benchmark_comparison['overall_assessment'] = "Excellent performance compared to peers"
            elif benchmark_comparison['overall_score'] >= 60:
                benchmark_comparison['overall_assessment'] = "Good performance compared to peers"
            elif benchmark_comparison['overall_score'] >= 40:
                benchmark_comparison['overall_assessment'] = "Average performance compared to peers"
            else:
                benchmark_comparison['overall_assessment'] = "Below average performance compared to peers"

            return benchmark_comparison

        except Exception as e:
            logger.error(f"Error benchmarking real estate company: {e}")
            return {'error': str(e)}

    def _combined_valuation_approach(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combined valuation approach using multiple methodologies

        Provides a more robust valuation by combining results of different methods
        """
        try:
            # Determine subsector
            company_info = financial_data.get('company_info', {})
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # Initialize combined result
            combined_result = {
                'company_name': company_info.get('name', 'Unknown'),
                'ticker': company_info.get('symbol', 'Unknown'),
                'valuation_method': 'combined',
                'estimated_value_per_share': None,
                'valuation_approaches': {},
                'weighted_values': {},
                'current_price': financial_data.get('market_data', {}).get('share_price'),
                'price_to_value_ratio': None,
                'valuation_assessment': None
            }

            # Apply different valuation methods based on subsector
            methods_to_use = []

            if subsector == 'reit':
                methods_to_use = ['nav', 'ffo_multiple', 'dividend_discount', 'real_estate_dcf']
                weights = {'nav': 0.35, 'ffo_multiple': 0.35, 'dividend_discount': 0.20, 'real_estate_dcf': 0.10}
            elif subsector == 'developer':
                methods_to_use = ['adjusted_nav', 'nav', 'real_estate_dcf', 'ffo_multiple']
                weights = {'adjusted_nav': 0.45, 'nav': 0.25, 'real_estate_dcf': 0.20, 'ffo_multiple': 0.10}
            elif subsector == 'property_manager':
                methods_to_use = ['real_estate_dcf', 'nav', 'ffo_multiple', 'dividend_discount']
                weights = {'real_estate_dcf': 0.5, 'nav': 0.2, 'ffo_multiple': 0.2, 'dividend_discount': 0.1}
            else:
                methods_to_use = ['nav', 'ffo_multiple', 'real_estate_dcf', 'dividend_discount']
                weights = {'nav': 0.4, 'ffo_multiple': 0.3, 'real_estate_dcf': 0.2, 'dividend_discount': 0.1}

            # Apply each method
            for method in methods_to_use:
                # Create a temporary result template
                temp_result = {
                    'company_name': company_info.get('name', 'Unknown'),
                    'ticker': company_info.get('symbol', 'Unknown'),
                    'valuation_method': method,
                    'estimated_value_per_share': None,
                    'current_price': financial_data.get('market_data', {}).get('share_price')
                }

                # Apply valuation method
                try:
                    if method == 'nav':
                        method_result = self._nav_valuation(financial_data, temp_result)
                    elif method == 'ffo_multiple':
                        method_result = self._ffo_multiple_valuation(financial_data, temp_result)
                    elif method == 'adjusted_nav':
                        method_result = self._adjusted_nav_valuation(financial_data, temp_result)
                    elif method == 'real_estate_dcf':
                        method_result = self._real_estate_dcf_valuation(financial_data, temp_result)
                    elif method == 'dividend_discount':
                        method_result = self._dividend_discount_valuation(financial_data, temp_result)
                    else:
                        continue

                    # Store the result
                    if 'error' not in method_result:
                        combined_result['valuation_approaches'][method] = method_result
                except Exception as method_error:
                    logger.warning(f"Error applying {method} method: {method_error}")
                    continue

            # Calculate weighted average of all successful methods
            weighted_sum = 0
            weights_used_sum = 0

            for method, result in combined_result['valuation_approaches'].items():
                value = result.get('estimated_value_per_share')
                if value is not None and value > 0:
                    weight = weights.get(method, 0)
                    weighted_sum += value * weight
                    weights_used_sum += weight

                    # Store individual weighted values
                    combined_result['weighted_values'][method] = {
                        'value': value,
                        'weight': weight,
                        'weighted_value': value * weight
                    }

            # Calculate final estimated value
            if weights_used_sum > 0:
                combined_result['estimated_value_per_share'] = weighted_sum / weights_used_sum

                # Add price-to-value assessment
                if combined_result['current_price'] is not None:
                    combined_result['price_to_value_ratio'] = (
                            combined_result['current_price'] / combined_result['estimated_value_per_share']
                    )

                    # Add assessment
                    if combined_result['price_to_value_ratio'] < 0.8:
                        combined_result['valuation_assessment'] = "Significantly undervalued"
                    elif combined_result['price_to_value_ratio'] < 0.95:
                        combined_result['valuation_assessment'] = "Moderately undervalued"
                    elif combined_result['price_to_value_ratio'] < 1.05:
                        combined_result['valuation_assessment'] = "Fairly valued"
                    elif combined_result['price_to_value_ratio'] < 1.2:
                        combined_result['valuation_assessment'] = "Moderately overvalued"
                    else:
                        combined_result['valuation_assessment'] = "Significantly overvalued"

            return combined_result

        except Exception as e:
            logger.error(f"Error performing combined valuation approach: {e}")
            return {
                'error': str(e),
                'valuation_method': 'combined'
            }

    def _determine_real_estate_subsector(self, industry: str) -> str:
        """
        Determine the specific real estate subsector based on industry description

        Args:
            industry: Industry description from company info

        Returns:
            Subsector classification: 'reit', 'developer', or 'property_manager'
        """
        industry_lower = industry.lower() if industry else ""

        if "reit" in industry_lower or "real estate investment trust" in industry_lower:
            return "reit"
        elif "developer" in industry_lower or "development" in industry_lower:
            return "developer"
        elif "property management" in industry_lower or "property services" in industry_lower:
            return "property_manager"
        else:
            # Default classification based on more specific terms
            if any(term in industry_lower for term in ["rental", "lease", "income properties"]):
                return "reit"
            elif any(term in industry_lower for term in ["construction", "building", "project"]):
                return "developer"
            elif any(term in industry_lower for term in ["manage", "service", "consult"]):
                return "property_manager"

        # Default to REIT if we can't determine
        return "reit"

    def _calculate_real_estate_metrics(self, income: pd.Series, balance: pd.Series,
                                       cash_flow: pd.Series, subsector: str) -> Dict[str, float]:
        """
        Calculate real estate specific metrics and financial ratios

        Args:
            income: Income statement data
            balance: Balance sheet data
            cash_flow: Cash flow statement data
            subsector: Real estate subsector classification

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}

        # Extract revenue
        revenue = None
        for field in ['Total Revenue', 'Revenue', 'Net Revenue', 'Sales', 'Net Sales']:
            if field in income:
                revenue = income[field]
                break

        # Extract net income
        net_income = None
        for field in ['Net Income', 'Net Profit', 'Net Earnings']:
            if field in income:
                net_income = income[field]
                break

        # Basic financial metrics
        try:
            if 'Total Assets' in balance and 'Total Liabilities' in balance:
                metrics['total_assets'] = balance['Total Assets']
                metrics['total_liabilities'] = balance['Total Liabilities']
                metrics['equity'] = balance['Total Assets'] - balance['Total Liabilities']
                metrics['debt_to_assets'] = balance['Total Liabilities'] / balance['Total Assets']

            if 'Total Stockholder Equity' in balance and balance['Total Stockholder Equity'] > 0:
                metrics['equity'] = balance['Total Stockholder Equity']

                if net_income is not None:
                    metrics['roe'] = net_income / balance['Total Stockholder Equity']

            if 'Interest Expense' in income and net_income is not None:
                interest_expense = abs(income['Interest Expense'])
                ebit = net_income + interest_expense
                metrics['interest_coverage'] = ebit / interest_expense if interest_expense > 0 else float('inf')

            # Calculate real estate specific metrics based on subsector
            if subsector == 'reit':
                # Funds From Operations (FFO)
                # FFO = Net Income + Depreciation - Gains From Property Sales
                depreciation = income.get('Depreciation & Amortization', 0)
                # Gains from sales normally not directly available, would need to be estimated
                gains_from_sales = 0

                ffo = net_income + depreciation - gains_from_sales if net_income is not None else None

                if ffo is not None:
                    metrics['ffo'] = ffo
                    if revenue is not None and revenue > 0:
                        metrics['ffo_margin'] = ffo / revenue

                    # Adjusted FFO (AFFO): FFO - Recurring Capital Expenditures
                    # Note: Recurring CapEx is often estimated
                    if 'Capital Expenditure' in cash_flow:
                        estimated_recurring_capex = cash_flow['Capital Expenditure'] * 0.6  # Estimate 60% as recurring
                        metrics['affo'] = ffo - abs(estimated_recurring_capex)

                    # Calculate yield metrics if market cap is available
                    if 'market_cap' in metrics and metrics['market_cap'] > 0:
                        metrics['ffo_yield'] = ffo / metrics['market_cap']
                        if 'affo' in metrics:
                            metrics['affo_yield'] = metrics['affo'] / metrics['market_cap']

                # Net Asset Value (NAV) components
                if 'Property Plant and Equipment' in balance:
                    metrics['property_value'] = balance['Property Plant and Equipment']

                # Occupancy rate - typically from company reports, not financial statements
                # This would require additional data not typically in standard financials
                metrics['occupancy_rate'] = 0.95  # Default assumption for illustration

            elif subsector == 'developer':
                # Development-specific metrics
                # Land bank and development pipeline usually from company reports
                # Pre-sales and margins also from reports

                # Simple metrics from financials
                if revenue is not None and net_income is not None:
                    metrics['profit_margin'] = net_income / revenue

                # Inventory turnover for developers
                if 'Inventory' in balance and revenue is not None:
                    metrics['inventory_turnover'] = revenue / balance['Inventory']

            elif subsector == 'property_manager':
                # Property management metrics
                # AUM (Assets Under Management) from company reports
                # Management fee revenue
                if revenue is not None:
                    # Extract operating expenses
                    operating_expenses = 0
                    for expense_item in ['Operating Expenses', 'SG&A', 'Selling General and Administrative']:
                        if expense_item in income:
                            operating_expenses = income[expense_item]
                            break

                    # Fee-based margin
                    if operating_expenses > 0:
                        metrics['fee_margin'] = (revenue - operating_expenses) / revenue

        except Exception as e:
            logger.warning(f"Error calculating real estate metrics: {e}")

        return metrics

    def _assess_performance(self, metrics: Dict[str, float], subsector: str) -> Dict[str, Dict[str, Any]]:
        """
        Assess company performance based on calculated metrics

        Args:
            metrics: Dictionary of financial metrics
            subsector: Real estate subsector classification

        Returns:
            Dictionary with performance assessments
        """
        assessment = {}

        # Get benchmarks
        sector_benchmarks = self.get_sector_benchmarks()
        subsector_benchmarks = self.real_estate_benchmarks.get(subsector, {})

        # Combine benchmarks
        benchmarks = {**sector_benchmarks, **subsector_benchmarks}

        # Assess each metric
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or pd.isna(value):
                continue

            # Check if we have a benchmark for this metric
            if metric in benchmarks:
                benchmark = benchmarks[metric]

                # Calculate percentage difference
                pct_diff = ((value - benchmark) / benchmark) * 100 if benchmark != 0 else 0

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ['debt_to_assets', 'debt_to_equity', 'vacancy_rate']:
                    higher_is_better = False

                # Assess performance
                if higher_is_better:
                    if value > benchmark * 1.2:
                        rating = "Excellent"
                    elif value > benchmark * 1.0:
                        rating = "Good"
                    elif value > benchmark * 0.8:
                        rating = "Average"
                    else:
                        rating = "Below Average"
                else:
                    if value < benchmark * 0.8:
                        rating = "Excellent"
                    elif value < benchmark * 1.0:
                        rating = "Good"
                    elif value < benchmark * 1.2:
                        rating = "Average"
                    else:
                        rating = "Below Average"

                assessment[metric] = {
                    'value': value,
                    'benchmark': benchmark,
                    'pct_diff': pct_diff,
                    'rating': rating,
                    'higher_is_better': higher_is_better
                }
            else:
                # For metrics without benchmarks, provide relative assessment
                if metric == 'ffo_margin':
                    if value > 0.6:
                        rating = "Excellent"
                    elif value > 0.5:
                        rating = "Good"
                    elif value > 0.4:
                        rating = "Average"
                    else:
                        rating = "Below Average"

                    assessment[metric] = {
                        'value': value,
                        'rating': rating,
                        'higher_is_better': True
                    }
                elif metric == 'occupancy_rate':
                    if value > 0.95:
                        rating = "Excellent"
                    elif value > 0.90:
                        rating = "Good"
                    elif value > 0.85:
                        rating = "Average"
                    else:
                        rating = "Below Average"

                    assessment[metric] = {
                        'value': value,
                        'rating': rating,
                        'higher_is_better': True
                    }

        return assessment

    def _identify_strengths_weaknesses(self, performance_assessment: Dict[str, Dict[str, Any]]) -> Tuple[
        List[str], List[str]]:
        """
        Identify key strengths and weaknesses based on performance assessment

        Args:
            performance_assessment: Dictionary with performance assessments

        Returns:
            Tuple containing lists of strengths and weaknesses
        """
        strengths = []
        weaknesses = []

        for metric, assessment in performance_assessment.items():
            # Check if there's a rating
            if 'rating' not in assessment:
                continue

            rating = assessment['rating']
            value = assessment['value']

            # Format the metric name for readability
            readable_metric = metric.replace('_', ' ').title()

            # Add strengths
            if rating == "Excellent":
                if 'benchmark' in assessment:
                    benchmark = assessment['benchmark']
                    if assessment.get('higher_is_better', True):
                        strengths.append(f"Strong {readable_metric} of {value:.2f} (vs. benchmark {benchmark:.2f})")
                    else:
                        strengths.append(
                            f"Excellent {readable_metric} of {value:.2f}, well below benchmark of {benchmark:.2f}")
                else:
                    strengths.append(f"Excellent {readable_metric} of {value:.2f}")
            elif rating == "Good":
                if 'benchmark' in assessment:
                    strengths.append(f"Good {readable_metric} of {value:.2f}")

            # Add weaknesses
            if rating == "Below Average":
                if 'benchmark' in assessment:
                    benchmark = assessment['benchmark']
                    if assessment.get('higher_is_better', True):
                        weaknesses.append(f"Weak {readable_metric} of {value:.2f} (vs. benchmark {benchmark:.2f})")
                    else:
                        weaknesses.append(f"High {readable_metric} of {value:.2f}, above benchmark of {benchmark:.2f}")
                else:
                    weaknesses.append(f"Below average {readable_metric} of {value:.2f}")
            elif rating == "Average" and 'benchmark' in assessment:
                # Include some average metrics as mild weaknesses if they're significantly below benchmark
                if assessment.get('higher_is_better', True) and value < assessment['benchmark'] * 0.9:
                    weaknesses.append(
                        f"Average {readable_metric} of {value:.2f}, below benchmark of {assessment['benchmark']:.2f}")

        return strengths, weaknesses

    def _analyze_risk_factors(self, financial_data: Dict[str, Any],
                              metrics: Dict[str, float], subsector: str) -> List[str]:
        """
        Analyze and identify key risk factors for a real estate company

        Args:
            financial_data: Dictionary containing financial statements and other data
            metrics: Dictionary of calculated metrics
            subsector: Real estate subsector classification

        Returns:
            List of identified risk factors
        """
        risk_factors = []

        # Extract financial data
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return ["Insufficient financial data to analyze risks"]

        # Get latest financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()

        # Check debt levels
        if metrics.get('debt_to_assets', 0) > 0.6:
            risk_factors.append("High leverage with debt-to-assets ratio exceeding 60%")

        # Check interest coverage
        if metrics.get('interest_coverage', float('inf')) < 2.0:
            risk_factors.append("Low interest coverage ratio indicates vulnerability to interest rate increases")

        # Check occupancy for REITs
        if subsector == 'reit' and metrics.get('occupancy_rate', 1.0) < 0.85:
            risk_factors.append(
                "Low occupancy rate below 85% indicates potential issues with property quality or market demand")

        # Check FFO margin for REITs
        if subsector == 'reit' and metrics.get('ffo_margin', 1.0) < 0.3:
            risk_factors.append("Low FFO margin indicates inefficient property operations")

        # Check development risks for developers
        if subsector == 'developer':
            if metrics.get('inventory_turnover', float('inf')) < 0.5:
                risk_factors.append("Low inventory turnover suggests slow project sales or development delays")
            if metrics.get('profit_margin', 1.0) < 0.15:
                risk_factors.append(
                    "Low profit margin indicates potential development cost overruns or pricing pressure")

        # Check property management risks
        if subsector == 'property_manager' and metrics.get('fee_margin', 1.0) < 0.4:
            risk_factors.append("Low fee margin indicates inefficient operations or pricing pressure")

        # Check company size and concentration risks
        if 'total_assets' in metrics:
            property_concentration = False  # Would need additional data

            if property_concentration:
                risk_factors.append("High property concentration increases vulnerability to local market downturns")

        # Check liquidity
        current_assets = None
        current_liabilities = None

        for field in ['Current Assets', 'Total Current Assets']:
            if field in latest_balance:
                current_assets = latest_balance[field]
                break

        for field in ['Current Liabilities', 'Total Current Liabilities']:
            if field in latest_balance:
                current_liabilities = latest_balance[field]
                break

        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio < 1.0:
                risk_factors.append("Current ratio below 1.0 indicates potential liquidity issues")

        # Market risks - these would normally come from external market data
        interest_rate_risk = True  # Real estate is generally sensitive to interest rate changes
        if interest_rate_risk:
            risk_factors.append(
                "Sensitivity to interest rate changes could impact property valuations and financing costs")

        return risk_factors

    def _nav_valuation(self, financial_data: Dict[str, Any], valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Net Asset Value (NAV) valuation for real estate companies

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_result: Dictionary to store valuation results

        Returns:
            Updated valuation result dictionary
        """
        try:
            # Extract financial statements
            balance_sheet = financial_data.get('balance_sheet')

            if balance_sheet is None or balance_sheet.empty:
                raise ValueError("Missing balance sheet data required for NAV valuation")

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding is None or shares_outstanding <= 0:
                raise ValueError("Missing or invalid shares outstanding data")

            # Get latest balance sheet data
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate basic book value
            if 'Total Assets' in latest_balance and 'Total Liabilities' in latest_balance:
                book_value = latest_balance['Total Assets'] - latest_balance['Total Liabilities']
            elif 'Total Stockholder Equity' in latest_balance:
                book_value = latest_balance['Total Stockholder Equity']
            else:
                raise ValueError("Cannot calculate book value from balance sheet")

            # For real estate, we make adjustments to book value:
            # 1. Real estate assets are often carried at historical cost minus depreciation
            # 2. Market value is often higher than book value

            # Get property value from balance sheet
            property_value_book = 0
            for field in ['Property Plant and Equipment', 'Investment Properties', 'Real Estate Assets']:
                if field in latest_balance:
                    property_value_book += latest_balance[field]

            # Apply adjustment factors based on property types
            # In a real implementation, this would use more granular data on property types and locations
            property_value_adjustment = 1.15  # Assuming 15% markup to fair market value

            # Adjust property value
            property_value_adjusted = property_value_book * property_value_adjustment

            # Calculate NAV by replacing book value of properties with adjusted value
            nav = book_value + (property_value_adjusted - property_value_book)

            # Calculate NAV per share
            nav_per_share = nav / shares_outstanding

            # Update valuation result
            valuation_result['estimated_value_per_share'] = nav_per_share
            valuation_result['valuation_details'] = {
                'book_value': book_value,
                'book_value_per_share': book_value / shares_outstanding,
                'property_value_book': property_value_book,
                'property_value_adjustment_factor': property_value_adjustment,
                'property_value_adjusted': property_value_adjusted,
                'nav': nav,
                'nav_per_share': nav_per_share
            }

            valuation_result['assumptions'] = {
                'property_value_adjustment': property_value_adjustment,
                'footnote': "Property values adjusted based on estimated market value premium to book value"
            }

            # Add sensitivity analysis
            sensitivity = {}
            adjustment_scenarios = [1.05, 1.10, 1.15, 1.20, 1.25]
            for adj in adjustment_scenarios:
                prop_val_adj = property_value_book * adj
                sens_nav = book_value + (prop_val_adj - property_value_book)
                sens_nav_per_share = sens_nav / shares_outstanding
                sensitivity[f"{(adj - 1) * 100:.0f}%_property_adjustment"] = sens_nav_per_share

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error in NAV valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _ffo_multiple_valuation(self, financial_data: Dict[str, Any], valuation_result: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Funds From Operations (FFO) multiple valuation for REITs and real estate companies

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_result: Dictionary to store valuation results

        Returns:
            Updated valuation result dictionary
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or income_stmt.empty:
                raise ValueError("Missing income statement data required for FFO multiple valuation")

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding is None or shares_outstanding <= 0:
                raise ValueError("Missing or invalid shares outstanding data")

            # Get latest income statement data
            latest_income = income_stmt.iloc[:, 0]

            # Calculate FFO
            # FFO = Net Income + Depreciation - Gains From Property Sales
            net_income = None
            for field in ['Net Income', 'Net Profit', 'Net Earnings']:
                if field in latest_income:
                    net_income = latest_income[field]
                    break

            if net_income is None:
                raise ValueError("Cannot find net income in income statement")

            # Get depreciation
            depreciation = latest_income.get('Depreciation & Amortization', 0)

            # Gains from property sales - typically not directly available
            # In practice, this would be extracted from financial statement notes
            gains_from_sales = 0  # Default assumption

            ffo = net_income + depreciation - gains_from_sales

            # Calculate FFO per share
            ffo_per_share = ffo / shares_outstanding

            # Determine appropriate FFO multiple based on subsector
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # Default multiples by subsector
            ffo_multiples = {
                'reit': 16.0,  # Standard multiple for REITs
                'developer': 12.0,  # Lower for developers due to development risk
                'property_manager': 14.0  # Service businesses often have different multiples
            }

            # Get appropriate multiple
            ffo_multiple = ffo_multiples.get(subsector, 15.0)

            # Calculate value based on FFO multiple
            value_per_share = ffo_per_share * ffo_multiple

            # Update valuation result
            valuation_result['estimated_value_per_share'] = value_per_share
            valuation_result['valuation_details'] = {
                'net_income': net_income,
                'depreciation': depreciation,
                'gains_from_sales': gains_from_sales,
                'ffo': ffo,
                'ffo_per_share': ffo_per_share,
                'ffo_multiple': ffo_multiple,
                'value_per_share': value_per_share
            }

            valuation_result['assumptions'] = {
                'ffo_multiple': ffo_multiple,
                'footnote': f"FFO multiple of {ffo_multiple}x based on typical multiples for {subsector} companies"
            }

            # Add sensitivity analysis
            sensitivity = {}
            multiple_scenarios = [ffo_multiple * 0.8, ffo_multiple * 0.9, ffo_multiple,
                                  ffo_multiple * 1.1, ffo_multiple * 1.2]
            for mult in multiple_scenarios:
                sens_value = ffo_per_share * mult
                sensitivity[f"FFO_multiple_{mult:.1f}x"] = sens_value

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error in FFO multiple valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _adjusted_nav_valuation(self, financial_data: Dict[str, Any], valuation_result: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Adjusted Net Asset Value valuation for real estate developers with development pipeline

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_result: Dictionary to store valuation results

        Returns:
            Updated valuation result dictionary
        """
        try:
            # This is similar to NAV valuation but includes adjustments for development pipeline
            # First get the basic NAV
            nav_result = self._nav_valuation(financial_data, valuation_result.copy())

            if 'error' in nav_result:
                raise ValueError(nav_result['error'])

            # Extract NAV details
            nav = nav_result['valuation_details']['nav']
            nav_per_share = nav_result['valuation_details']['nav_per_share']
            shares_outstanding = financial_data.get('market_data', {}).get('shares_outstanding')

            # For development companies, we need to add value of:
            # 1. Development pipeline (projects under construction)
            # 2. Land bank (future development opportunities)

            # These would normally come from company reports
            # Using default assumptions for illustration
            development_pipeline_value = nav * 0.2  # Assuming pipeline worth 20% of NAV
            land_bank_value = nav * 0.1  # Assuming land bank worth 10% of NAV

            # Calculate adjusted NAV
            adjusted_nav = nav + development_pipeline_value + land_bank_value
            adjusted_nav_per_share = adjusted_nav / shares_outstanding

            # Update valuation result
            valuation_result['estimated_value_per_share'] = adjusted_nav_per_share
            valuation_result['valuation_details'] = {
                'base_nav': nav,
                'base_nav_per_share': nav_per_share,
                'development_pipeline_value': development_pipeline_value,
                'land_bank_value': land_bank_value,
                'adjusted_nav': adjusted_nav,
                'adjusted_nav_per_share': adjusted_nav_per_share
            }

            valuation_result['assumptions'] = {
                'development_pipeline_adjustment': "20% of NAV",
                'land_bank_adjustment': "10% of NAV",
                'footnote': "Development pipeline and land bank values estimated based on typical percentages for real estate developers"
            }

            # Add sensitivity analysis
            sensitivity = {}
            pipeline_scenarios = [0.1, 0.15, 0.2, 0.25, 0.3]
            for pipeline_pct in pipeline_scenarios:
                dev_val = nav * pipeline_pct
                land_val = nav * (pipeline_pct / 2)  # Land bank as half of pipeline %
                sens_nav = nav + dev_val + land_val
                sens_nav_per_share = sens_nav / shares_outstanding
                sensitivity[f"pipeline_{pipeline_pct * 100:.0f}%_of_nav"] = sens_nav_per_share

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error in adjusted NAV valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _real_estate_dcf_valuation(self, financial_data: Dict[str, Any], valuation_result: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Discounted Cash Flow valuation adjusted for real estate companies

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_result: Dictionary to store valuation results

        Returns:
            Updated valuation result dictionary
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or income_stmt.empty or cash_flow is None or cash_flow.empty:
                raise ValueError("Missing income statement or cash flow data required for DCF valuation")

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding is None or shares_outstanding <= 0:
                raise ValueError("Missing or invalid shares outstanding data")

            # Determine subsector
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # Get parameters based on subsector
            forecast_years = 5  # Standard forecast period
            terminal_growth_rates = {
                'reit': 0.02,  # Stable growth for REITs
                'developer': 0.03,  # Higher growth for developers
                'property_manager': 0.025  # Moderate growth for service businesses
            }
            terminal_growth = terminal_growth_rates.get(subsector, 0.02)

            # Discount rates (WACC) based on subsector
            discount_rates = {
                'reit': 0.07,  # Lower risk for REITs
                'developer': 0.09,  # Higher risk for developers
                'property_manager': 0.08  # Moderate risk for service businesses
            }
            discount_rate = discount_rates.get(subsector, 0.08)

            # Get historical cash flow data
            # For real estate, we use operating cash flow as the base
            operating_cash_flows = []
            for year in range(min(3, cash_flow.shape[1])):
                if 'Operating Cash Flow' in cash_flow.iloc[:, year]:
                    operating_cash_flows.append(cash_flow.iloc[:, year]['Operating Cash Flow'])

            if not operating_cash_flows:
                raise ValueError("Cannot find operating cash flow data")

            # Calculate average growth rate from historical data
            if len(operating_cash_flows) >= 2:
                growth_rates = [(operating_cash_flows[i] / operating_cash_flows[i + 1] - 1)
                                for i in range(len(operating_cash_flows) - 1)]
                historical_growth = sum(growth_rates) / len(growth_rates)
                # Cap growth rate to reasonable range
                growth_rate = max(0.01, min(0.15, historical_growth))
            else:
                # Default growth rates by subsector
                default_growth_rates = {
                    'reit': 0.03,  # Stable growth for REITs
                    'developer': 0.06,  # Higher growth for developers
                    'property_manager': 0.05  # Moderate growth for service businesses
                }
                growth_rate = default_growth_rates.get(subsector, 0.04)

                # Starting cash flow for projection
            starting_cash_flow = operating_cash_flows[0]

            # Project future cash flows
            projected_cash_flows = []
            for year in range(1, forecast_years + 1):
                cf = starting_cash_flow * (1 + growth_rate) ** year
                projected_cash_flows.append(cf)

            # Calculate terminal value
            terminal_value = projected_cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

            # Discount cash flows to present value
            present_value_cf = sum(cf / (1 + discount_rate) ** year
                                   for year, cf in enumerate(projected_cash_flows, 1))
            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

            # Calculate enterprise value
            enterprise_value = present_value_cf + present_value_terminal

            # Get debt and cash to calculate equity value
            balance_sheet = financial_data.get('balance_sheet')
            if balance_sheet is None or balance_sheet.empty:
                raise ValueError("Missing balance sheet data")

            latest_balance = balance_sheet.iloc[:, 0]

            # Get debt
            total_debt = 0
            for debt_field in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_field in latest_balance:
                    total_debt += latest_balance[debt_field]

            # Get cash
            cash = 0
            for cash_field in ['Cash and Cash Equivalents', 'Cash and Short Term Investments']:
                if cash_field in latest_balance:
                    cash += latest_balance[cash_field]

            # Calculate equity value
            equity_value = enterprise_value - total_debt + cash

            # Calculate per share value
            value_per_share = equity_value / shares_outstanding

            # Update valuation result
            valuation_result['estimated_value_per_share'] = value_per_share
            valuation_result['valuation_details'] = {
                'starting_cash_flow': starting_cash_flow,
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'projected_cash_flows': projected_cash_flows,
                'terminal_value': terminal_value,
                'present_value_cf': present_value_cf,
                'present_value_terminal': present_value_terminal,
                'enterprise_value': enterprise_value,
                'total_debt': total_debt,
                'cash': cash,
                'equity_value': equity_value,
                'value_per_share': value_per_share
            }

            valuation_result['assumptions'] = {
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'footnote': f"DCF parameters adjusted for {subsector} real estate company profile"
            }

            # Add sensitivity analysis
            sensitivity = {}
            # Growth rate sensitivity
            for alt_growth in [growth_rate - 0.02, growth_rate - 0.01, growth_rate,
                               growth_rate + 0.01, growth_rate + 0.02]:
                alt_projected_cf = []
                for year in range(1, forecast_years + 1):
                    cf = starting_cash_flow * (1 + alt_growth) ** year
                    alt_projected_cf.append(cf)

                alt_terminal = alt_projected_cf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                alt_pv_cf = sum(cf / (1 + discount_rate) ** year
                                for year, cf in enumerate(alt_projected_cf, 1))
                alt_pv_terminal = alt_terminal / (1 + discount_rate) ** forecast_years
                alt_ev = alt_pv_cf + alt_pv_terminal
                alt_equity = alt_ev - total_debt + cash
                alt_value_per_share = alt_equity / shares_outstanding

                sensitivity[f"growth_{alt_growth:.1%}"] = alt_value_per_share

            # Discount rate sensitivity
            for alt_discount in [discount_rate - 0.01, discount_rate - 0.005, discount_rate,
                                 discount_rate + 0.005, discount_rate + 0.01]:
                alt_pv_cf = sum(cf / (1 + alt_discount) ** year
                                for year, cf in enumerate(projected_cash_flows, 1))
                alt_terminal = projected_cash_flows[-1] * (1 + terminal_growth) / (alt_discount - terminal_growth)
                alt_pv_terminal = alt_terminal / (1 + alt_discount) ** forecast_years
                alt_ev = alt_pv_cf + alt_pv_terminal
                alt_equity = alt_ev - total_debt + cash
                alt_value_per_share = alt_equity / shares_outstanding

                sensitivity[f"discount_{alt_discount:.1%}"] = alt_value_per_share

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error in real estate DCF valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _dividend_discount_valuation(self, financial_data: Dict[str, Any], valuation_result: Dict[str, Any]) -> \
    Dict[str, Any]:
        """
        Dividend Discount Model valuation for real estate companies

        Args:
            financial_data: Dictionary containing financial statements and other data
            valuation_result: Dictionary to store valuation results

        Returns:
            Updated valuation result dictionary
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            cash_flow = financial_data.get('cash_flow')
            company_info = financial_data.get('company_info', {})

            if income_stmt is None or income_stmt.empty or cash_flow is None or cash_flow.empty:
                raise ValueError(
                    "Missing income statement or cash flow data required for dividend discount valuation")

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if shares_outstanding is None or shares_outstanding <= 0:
                raise ValueError("Missing or invalid shares outstanding data")

            # Determine subsector
            industry = company_info.get('industry', 'Real Estate')
            subsector = self._determine_real_estate_subsector(industry)

            # REITs are required to pay out at least 90% of taxable income as dividends
            # which makes them good candidates for DDM

            # Get historical dividend data
            dividends_paid = []
            for year in range(min(3, cash_flow.shape[1])):
                cf_column = cash_flow.iloc[:, year]
                for div_field in ['Dividends Paid', 'Common Stock Dividends Paid', 'Payments of Dividends']:
                    if div_field in cf_column:
                        div_amount = abs(cf_column[div_field])  # Usually reported as negative in cash flow
                        if shares_outstanding > 0:
                            dividends_paid.append(div_amount / shares_outstanding)  # Per share
                        break

            if not dividends_paid:
                raise ValueError("Cannot find dividend payment data")

            # Calculate dividend growth rate from historical data
            if len(dividends_paid) >= 2:
                dividend_growth_rates = [(dividends_paid[i] / dividends_paid[i + 1] - 1)
                                         for i in range(len(dividends_paid) - 1)]
                historical_growth = sum(dividend_growth_rates) / len(dividend_growth_rates)
                # Cap growth rate to reasonable range
                dividend_growth = max(0.01, min(0.10, historical_growth))
            else:
                # Default dividend growth rates by subsector
                default_div_growth = {
                    'reit': 0.04,  # Stable dividend growth for REITs
                    'developer': 0.05,  # Variable for developers
                    'property_manager': 0.06  # Higher potential for fee-based businesses
                }
                dividend_growth = default_div_growth.get(subsector, 0.04)

            # Current dividend per share
            current_dividend = dividends_paid[0]

            # Discount rates based on subsector (required rate of return)
            discount_rates = {
                'reit': 0.08,  # Lower for stable REITs
                'developer': 0.10,  # Higher for developers
                'property_manager': 0.09  # Medium for service businesses
            }
            discount_rate = discount_rates.get(subsector, 0.09)

            # Apply Gordon Growth Model for dividend valuation
            # Value = Dividend * (1 + Dividend Growth) / (Discount Rate - Dividend Growth)
            # Only valid when Discount Rate > Dividend Growth

            if discount_rate <= dividend_growth:
                # If discount rate <= growth rate, we need to use multi-stage model
                # But for simplicity, we'll adjust growth rate down
                dividend_growth = discount_rate - 0.02  # Ensure a 2% spread

            next_dividend = current_dividend * (1 + dividend_growth)
            value_per_share = next_dividend / (discount_rate - dividend_growth)

            # Update valuation result
            valuation_result['estimated_value_per_share'] = value_per_share
            valuation_result['valuation_details'] = {
                'current_dividend_per_share': current_dividend,
                'dividend_growth_rate': dividend_growth,
                'discount_rate': discount_rate,
                'next_dividend': next_dividend,
                'value_per_share': value_per_share
            }

            valuation_result['assumptions'] = {
                'dividend_growth_rate': dividend_growth,
                'discount_rate': discount_rate,
                'model': "Gordon Growth Model",
                'footnote': "Dividend discount model particularly relevant for REITs with stable dividend policies"
            }

            # Add sensitivity analysis
            sensitivity = {}
            # Growth rate sensitivity
            for alt_growth in [dividend_growth - 0.01, dividend_growth - 0.005, dividend_growth,
                               dividend_growth + 0.005, dividend_growth + 0.01]:
                # Ensure discount rate > growth rate
                if discount_rate <= alt_growth:
                    continue

                alt_next_div = current_dividend * (1 + alt_growth)
                alt_value = alt_next_div / (discount_rate - alt_growth)
                sensitivity[f"growth_{alt_growth:.1%}"] = alt_value

            # Discount rate sensitivity
            for alt_discount in [discount_rate - 0.01, discount_rate - 0.005, discount_rate,
                                 discount_rate + 0.005, discount_rate + 0.01]:
                # Ensure discount rate > growth rate
                if alt_discount <= dividend_growth:
                    continue

                alt_value = next_dividend / (alt_discount - dividend_growth)
                sensitivity[f"discount_{alt_discount:.1%}"] = alt_value

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error in dividend discount valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

# Register the analyzer with the factory
SectorFactory.register_analyzer("Real Estate", RealEstateSectorAnalyzer)