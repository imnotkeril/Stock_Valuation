import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.valuation.sector_factor import SectorAnalyzer
from StockAnalysisSystem.src.config import SECTOR_METRICS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('utilities_sector')


class UtilitiesSectorAnalyzer(SectorAnalyzer):
    """
    Specialized analyzer for utilities sector companies.

    This sector includes electric, gas, water, and renewable utilities,
    along with independent power producers.
    """

    def __init__(self, sector_name: str = "Utilities"):
        """Initialize utilities sector analyzer"""
        super().__init__(sector_name)
        logger.info("Initialized Utilities sector analyzer")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform utilities-specific analysis on company data

        Args:
            financial_data: Dictionary containing financial statements and other company data

        Returns:
            Dictionary with analysis results specific to utilities companies
        """
        result = {
            "sector": "Utilities",
            "industry_metrics": {},
            "sector_specific_analysis": {},
            "risk_factors": [],
            "competitive_advantages": [],
            "regulatory_analysis": {},
            "power_generation_mix": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Skip analysis if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for utilities sector analysis")
            return result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Calculate utilities-specific metrics
        self._calculate_utilities_metrics(result, latest_income, latest_balance, latest_cash_flow)

        # Analyze regulatory environment
        self._analyze_regulatory_environment(result, financial_data)

        # Analyze power generation mix
        self._analyze_power_generation_mix(result, financial_data)

        # Identify key industry risk factors
        self._identify_risk_factors(result, financial_data)

        # Evaluate competitive advantages
        self._evaluate_competitive_advantages(result, financial_data)

        return result

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using utilities-specific methods

        Args:
            financial_data: Dictionary containing financial information
            valuation_method: Specific valuation method to use (if None, use the most appropriate)

        Returns:
            Dictionary with valuation results
        """
        # Default to DDM if no method specified (dividend discount model is common for utilities)
        if valuation_method is None:
            valuation_method = "DDM"

        valuation_result = {
            "sector": "Utilities",
            "methods": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for utilities sector valuation")
            return valuation_result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Perform Dividend Discount Model valuation
        if valuation_method in ["DDM", "all"]:
            self._calculate_ddm_valuation(valuation_result, financial_data)

        # Perform Regulated Asset Base valuation
        if valuation_method in ["RAB", "all"]:
            self._calculate_rab_valuation(valuation_result, financial_data)

        # Perform P/E valuation (common for utilities)
        if valuation_method in ["P/E", "all"]:
            self._calculate_pe_valuation(valuation_result, latest_income, market_data)

        # Perform DCF with utilities-specific adjustments
        if valuation_method in ["DCF", "all"]:
            self._calculate_dcf_valuation(valuation_result, financial_data)

        return valuation_result

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with utilities sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks()

        benchmark_results = {
            "sector": "Utilities",
            "comparisons": {},
            "score": 0.0,
            "strengths": [],
            "weaknesses": []
        }

        # Compare key utilities metrics
        key_metrics = [
            "Dividend Yield",
            "Interest Coverage",
            "Net Margin",
            "ROE",
            "Debt/EBITDA",
            "Regulatory Asset Base Growth",
            "Payout Ratio"
        ]

        comparison_scores = []

        for metric in key_metrics:
            if metric in company_metrics and metric in benchmarks:
                company_value = company_metrics[metric]
                benchmark_value = benchmarks[metric]

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ["Debt/EBITDA", "Payout Ratio"]:
                    higher_is_better = False

                # Calculate percent difference from benchmark
                if benchmark_value != 0:
                    percent_diff = (company_value / benchmark_value - 1) * 100
                else:
                    percent_diff = 0

                # Evaluate performance
                if higher_is_better:
                    if percent_diff >= 20:
                        performance = "excellent"
                        score = 1.0
                        benchmark_results["strengths"].append(
                            f"Strong {metric} ({percent_diff:.1f}% above sector median)")
                    elif percent_diff >= 5:
                        performance = "good"
                        score = 0.75
                        benchmark_results["strengths"].append(f"Above-average {metric}")
                    elif percent_diff >= -5:
                        performance = "average"
                        score = 0.5
                    elif percent_diff >= -20:
                        performance = "below average"
                        score = 0.25
                        benchmark_results["weaknesses"].append(f"Below-average {metric}")
                    else:
                        performance = "poor"
                        score = 0.0
                        benchmark_results["weaknesses"].append(
                            f"Weak {metric} ({-percent_diff:.1f}% below sector median)")
                else:
                    # For metrics where lower is better
                    if percent_diff <= -20:
                        performance = "excellent"
                        score = 1.0
                        benchmark_results["strengths"].append(
                            f"Strong {metric} ({-percent_diff:.1f}% below sector median)")
                    elif percent_diff <= -5:
                        performance = "good"
                        score = 0.75
                        benchmark_results["strengths"].append(f"Above-average {metric}")
                    elif percent_diff <= 5:
                        performance = "average"
                        score = 0.5
                    elif percent_diff <= 20:
                        performance = "below average"
                        score = 0.25
                        benchmark_results["weaknesses"].append(f"Below-average {metric}")
                    else:
                        performance = "poor"
                        score = 0.0
                        benchmark_results["weaknesses"].append(
                            f"Weak {metric} ({percent_diff:.1f}% above sector median)")

                benchmark_results["comparisons"][metric] = {
                    "company_value": company_value,
                    "benchmark_value": benchmark_value,
                    "percent_difference": percent_diff,
                    "performance": performance
                }

                comparison_scores.append(score)

        # Calculate overall score
        if comparison_scores:
            benchmark_results["score"] = sum(comparison_scores) / len(comparison_scores)

        return benchmark_results

    def _calculate_utilities_metrics(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                     cash_flow: pd.Series) -> None:
        """Calculate utilities-specific metrics"""
        metrics = {}

        # Margin Metrics
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            metrics['Operating Margin'] = operating_margin

        if 'Net Income' in income and 'Total Revenue' in income:
            net_margin = income['Net Income'] / income['Total Revenue']
            metrics['Net Margin'] = net_margin

        # EBITDA and Coverage Ratios - key for regulated utilities
        if 'EBITDA' in income:
            ebitda = income['EBITDA']
        elif 'Operating Income' in income and 'Depreciation & Amortization' in income:
            ebitda = income['Operating Income'] + income['Depreciation & Amortization']
        else:
            ebitda = None

        if ebitda is not None and 'Total Revenue' in income:
            ebitda_margin = ebitda / income['Total Revenue']
            metrics['EBITDA Margin'] = ebitda_margin

        # Interest Coverage Ratio
        if 'Operating Income' in income and 'Interest Expense' in income and income['Interest Expense'] != 0:
            interest_coverage = income['Operating Income'] / abs(income['Interest Expense'])
            metrics['Interest Coverage'] = interest_coverage

        # Debt Metrics
        if 'Total Debt' in balance and ebitda is not None and ebitda > 0:
            debt_to_ebitda = balance['Total Debt'] / ebitda
            metrics['Debt/EBITDA'] = debt_to_ebitda

        # ROE and ROA
        if 'Net Income' in income:
            if 'Total Stockholder Equity' in balance and balance['Total Stockholder Equity'] > 0:
                roe = income['Net Income'] / balance['Total Stockholder Equity']
                metrics['ROE'] = roe

            if 'Total Assets' in balance and balance['Total Assets'] > 0:
                roa = income['Net Income'] / balance['Total Assets']
                metrics['ROA'] = roa

        # Capital Intensity - utilities are very capital-intensive
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            metrics['Capital Intensity'] = capital_intensity

        # Regulatory Asset Base (Regulated Asset Base) - key for regulated utilities
        # Note: This typically would come from company reports, not financial statements
        metrics['Regulatory Asset Base'] = None
        metrics['Allowed ROE'] = None  # Allowed return set by regulators
        metrics['Actual ROE'] = metrics.get('ROE')  # For comparison with allowed ROE

        # Dividend Metrics - utilities are often dividend-focused
        if 'Net Income' in income:
            # Calculate payout ratio
            if cash_flow is not None and 'Dividends Paid' in cash_flow:
                payout_ratio = abs(cash_flow['Dividends Paid']) / income['Net Income']
                metrics['Payout Ratio'] = payout_ratio

        # Dividend Yield - market data would typically provide this
        metrics['Dividend Yield'] = None  # Placeholder

        # CAPEX Intensity
        if cash_flow is not None and 'Capital Expenditure' in cash_flow and 'Total Revenue' in income:
            capex_to_revenue = abs(cash_flow['Capital Expenditure']) / income['Total Revenue']
            metrics['CAPEX/Revenue'] = capex_to_revenue

            # CAPEX to Depreciation ratio
            if 'Depreciation & Amortization' in income and income['Depreciation & Amortization'] > 0:
                capex_to_depreciation = abs(cash_flow['Capital Expenditure']) / income['Depreciation & Amortization']
                metrics['CAPEX/Depreciation'] = capex_to_depreciation

        result["industry_metrics"] = metrics

    def _analyze_regulatory_environment(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze regulatory environment for utilities company"""
        # Note: In a production system, this would extract data from company reports
        # or incorporate information about regulatory jurisdictions

        regulatory_analysis = {
            "regulatory_framework": None,  # e.g., "rate of return", "incentive-based", "performance-based"
            "regulatory_jurisdictions": None,  # e.g., ["California", "Nevada"]
            "allowed_return_on_equity": None,  # e.g., 9.5%
            "regulatory_lag": None,  # e.g., "low", "medium", "high"
            "regulatory_environment_assessment": None,  # e.g., "favorable", "neutral", "challenging"
            "rate_case_frequency": None,  # e.g., "2-3 years"
            "recent_rate_decisions": None  # e.g., "approved $X million increase in Dec 2023"
        }

        result["regulatory_analysis"] = regulatory_analysis

    def _analyze_power_generation_mix(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze power generation mix for electric utilities"""
        # Note: In a production system, this would extract data from company reports
        # about generation capacity by fuel type

        power_generation_mix = {
            "total_capacity_mw": None,
            "generation_by_fuel_type": {
                "natural_gas": None,  # percentage
                "coal": None,  # percentage
                "nuclear": None,  # percentage
                "hydro": None,  # percentage
                "wind": None,  # percentage
                "solar": None,  # percentage
                "other_renewables": None  # percentage
            },
            "carbon_intensity": None,  # e.g., "kg CO2/MWh"
            "renewable_capacity_growth_rate": None,  # e.g., "15% per year"
            "generation_fleet_avg_age": None  # e.g., "15 years"
        }

        result["power_generation_mix"] = power_generation_mix

    def _identify_risk_factors(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Identify key industry risk factors for utilities companies"""
        risks = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Regulatory risk - inherent in utilities sector
        risks.append({
            "factor": "Regulatory Risk",
            "description": "Utilities are subject to regulatory oversight that can impact allowed returns, capital recovery, and operational decisions",
            "severity": "High"
        })

        # Interest rate risk
        risks.append({
            "factor": "Interest Rate Sensitivity",
            "description": "Utilities' high capital needs and dividend-focused valuation makes them particularly sensitive to interest rate changes",
            "severity": "Medium"
        })

        # Check for high debt levels
        if 'Total Debt' in balance and 'EBITDA' in income:
            if income['EBITDA'] > 0:
                debt_to_ebitda = balance['Total Debt'] / income['EBITDA']
                if debt_to_ebitda > 4.0:
                    risks.append({
                        "factor": "High Debt Burden",
                        "description": "Elevated debt levels increase financial risk and interest rate sensitivity",
                        "severity": "High" if debt_to_ebitda > 5.0 else "Medium"
                    })

        # Check for weak interest coverage
        if 'Operating Income' in income and 'Interest Expense' in income and income['Interest Expense'] != 0:
            interest_coverage = income['Operating Income'] / abs(income['Interest Expense'])
            if interest_coverage < 3.0:
                risks.append({
                    "factor": "Weak Interest Coverage",
                    "description": "Low interest coverage ratio indicates heightened financial risk",
                    "severity": "High" if interest_coverage < 2.0 else "Medium"
                })

        # Environmental risk (particularly for electric utilities)
        risks.append({
            "factor": "Environmental Regulatory Pressure",
            "description": "Utilities face increasing environmental regulations and transition costs related to decarbonization",
            "severity": "Medium"
        })

        # Weather and climate risk
        risks.append({
            "factor": "Weather and Climate Risk",
            "description": "Extreme weather events and changing climate patterns can impact infrastructure and operational costs",
            "severity": "Medium"
        })

        # Capital expenditure requirements
        risks.append({
            "factor": "High Capital Expenditure Requirements",
            "description": "Utilities face ongoing capital needs for infrastructure maintenance and expansion",
            "severity": "Medium"
        })

        result["risk_factors"] = risks

    def _evaluate_competitive_advantages(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Evaluate potential competitive advantages for utilities companies"""
        advantages = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Regulatory relationship - key advantage for utilities
        advantages.append({
            "advantage": "Regulatory Framework",
            "description": "Utilities operate as regulated monopolies with protected service territories and predictable returns",
            "strength": "High"
        })

        # Check for low-cost power generation (for electric utilities)
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            if operating_margin > 0.22:  # Above average for utilities
                advantages.append({
                    "advantage": "Low-Cost Operations",
                    "description": "Above-average operating margins suggest cost advantages in generation or distribution",
                    "strength": "Medium"
                })

        # Check for strong dividend yield
        if 'industry_metrics' in result and 'Dividend Yield' in result['industry_metrics']:
            dividend_yield = result['industry_metrics']['Dividend Yield']
            if dividend_yield and dividend_yield > 0.04:  # Above average yield
                advantages.append({
                    "advantage": "Strong Dividend Yield",
                    "description": "Above-average dividend yield indicates strong cash flow generation and shareholder returns",
                    "strength": "Medium"
                })

        # Scale advantages
        if 'Total Revenue' in income and income['Total Revenue'] > 5000000000:  # $5B+ revenue
            advantages.append({
                "advantage": "Scale and Geographic Diversification",
                "description": "Large scale operations can provide cost advantages and regulatory diversification",
                "strength": "Medium"
            })

        # Rate base growth
        advantages.append({
            "advantage": "Rate Base Growth",
            "description": "Growing regulatory asset base provides visible path to future earnings growth",
            "strength": "Medium"
        })

        result["competitive_advantages"] = advantages

    def _calculate_ddm_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate valuation using Dividend Discount Model"""
        valuation = {}

        # Extract required data
        income_stmt = financial_data.get('income_statement')
        cash_flow = financial_data.get('cash_flow')
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if income_stmt is None or cash_flow is None:
            valuation["error"] = "Missing income statement or cash flow for DDM valuation"
            result["methods"]["DDM"] = valuation
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        cf = cash_flow.iloc[:, 0]

        # Get current dividend
        current_dividend = None
        shares_outstanding = market_data.get('shares_outstanding')

        if 'Dividends Paid' in cf and shares_outstanding:
            annual_dividend_total = abs(cf['Dividends Paid'])
            current_dividend = annual_dividend_total / shares_outstanding

        # If we can't calculate dividend, skip this valuation method
        if current_dividend is None or current_dividend <= 0:
            valuation["error"] = "Unable to calculate dividend per share or no dividend paid"
            result["methods"]["DDM"] = valuation
            return

        # Estimate dividend growth rate
        # For utilities, this is often linked to EPS growth or regulated asset base growth
        if income_stmt.shape[1] >= 2 and 'Net Income' in income_stmt.index:
            # Calculate EPS growth rate
            current_net_income = income_stmt.iloc[0]['Net Income']
            previous_net_income = income_stmt.iloc[1]['Net Income']

            if previous_net_income > 0:
                eps_growth = (current_net_income / previous_net_income) - 1
                # Utilities typically have modest growth rates
                dividend_growth = min(eps_growth, 0.05)  # Cap at 5%
            else:
                dividend_growth = 0.03  # Default modest growth
        else:
            dividend_growth = 0.03  # Default modest growth for utilities

        # Calculate required rate of return (discount rate)
        # For utilities, this is typically their cost of equity
        required_return = 0.08  # Default 8% for utilities

        # DDM Stage 1: Calculate terminal value using Gordon Growth Model
        # V = D * (1 + g) / (r - g)
        terminal_value = current_dividend * (1 + dividend_growth) / (required_return - dividend_growth)

        # Calculate per share value (DDM directly gives per share value)
        value_per_share = terminal_value

        # Calculate total equity value
        if shares_outstanding:
            equity_value = value_per_share * shares_outstanding
        else:
            equity_value = None

        # Store valuation result
        valuation["value_per_share"] = value_per_share
        valuation["equity_value"] = equity_value
        valuation["current_dividend"] = current_dividend
        valuation["dividend_growth_rate"] = dividend_growth
        valuation["required_return"] = required_return

        result["methods"]["DDM"] = valuation

    def _calculate_rab_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate valuation based on Regulated Asset Base"""
        valuation = {}

        # Note: In a real implementation, this would extract RAB data from company reports
        # as this information is not typically directly available in financial statements

        # For this example, we'll initialize with a note
        valuation["note"] = "Regulated Asset Base valuation requires data typically found in company reports"
        valuation["key_metrics_needed"] = [
            "Regulatory Asset Base (RAB)",
            "Allowed Return on RAB",
            "Expected RAB growth rate"
        ]
        valuation["typical_valuation_approaches"] = [
            "Premium/Discount to RAB: Typically 1.1-1.3x RAB for US utilities",
            "Allowed Return x (1 + Growth premium)",
            "Discounted RAB growth over regulatory cycle"
        ]

        result["methods"]["RAB"] = valuation

    def _calculate_pe_valuation(self, result: Dict[str, Any], income: pd.Series, market_data: Dict[str, Any]) -> None:
        """Calculate P/E-based valuation (common for utilities)"""
        valuation = {}

        # Get earnings
        net_income = None
        if 'Net Income' in income:
            net_income = income['Net Income']

        # If we can't get net income, skip this valuation method
        if net_income is None or net_income <= 0:
            valuation["error"] = "Unable to calculate P/E because net income is negative or unavailable"
            result["methods"]["P/E"] = valuation
            return

        # Get sector P/E multiple
        sector_pe = 16.0  # Typical for utilities
        if 'benchmarks' in self.sector_metrics:
            if 'P/E' in self.sector_metrics['benchmarks']:
                sector_pe = self.sector_metrics['benchmarks']['P/E']

        # Calculate equity value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            eps = net_income / shares_outstanding
            value_per_share = eps * sector_pe
            equity_value = value_per_share * shares_outstanding
        else:
            equity_value = net_income * sector_pe
            value_per_share = None

        # Store valuation result
        valuation["equity_value"] = equity_value
        valuation["value_per_share"] = value_per_share
        valuation["multiple_used"] = sector_pe
        valuation["net_income"] = net_income

        result["methods"]["P/E"] = valuation

    def _calculate_dcf_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate DCF with utilities-specific adjustments"""
        valuation = {}

        # This would typically call a DCF model implementation with sector-specific parameters
        # For brevity, we'll note what would be different for utilities companies

        valuation["note"] = "Utilities DCF would typically include:"
        valuation["adjustments"] = [
            "Lower discount rates (reflecting lower risk)",
            "Lower terminal growth rates (due to regulatory constraints)",
            "Focus on regulated vs. unregulated business segments",
            "Incorporation of regulatory asset base growth",
            "Consideration of allowed return on equity set by regulators",
            "Adjustment for environmental compliance capital expenditures"
        ]

        result["methods"]["DCF"] = valuation

    def get_key_ratios(self) -> List[str]:
        """
        Get the list of key ratios for utilities sector

        Returns:
            List of ratio names that are most relevant for utilities sector
        """
        # Override parent method to return utilities-specific ratios
        return [
            "Dividend Yield",
            "P/E",
            "Debt/EBITDA",
            "Interest Coverage",
            "Payout Ratio",
            "ROE",
            "EBITDA Margin",
            "CAPEX/Depreciation"
        ]


# Register with factory
if __name__ != "__main__":  # Avoid registration during imports
    try:
        from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory
        SectorFactory.register_analyzer("Utilities", UtilitiesSectorAnalyzer)
        logger.info("Registered Utilities sector analyzer with factory")
    except ImportError:
        logger.warning("Could not register Utilities sector analyzer with factory")
