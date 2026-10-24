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
logger = logging.getLogger('communications_sector')


class CommunicationsSectorAnalyzer(SectorAnalyzer):
    """
    Specialized analyzer for communications sector companies.

    This sector includes telecom providers, media companies, social networks,
    entertainment companies, and other communication services businesses.
    """

    def __init__(self, sector_name: str):
        """Initialize communications sector analyzer"""
        super().__init__(sector_name)
        logger.info(f"Initialized {sector_name} analyzer")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform communications-specific analysis on company data

        Args:
            financial_data: Dictionary containing financial statements and other company data

        Returns:
            Dictionary with analysis results specific to communications companies
        """
        result = {
            "sector": "Communication Services",
            "industry_metrics": {},
            "sector_specific_analysis": {},
            "risk_factors": [],
            "competitive_advantages": [],
            "subscription_metrics": {},
            "content_analysis": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Skip analysis if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for communications sector analysis")
            return result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Calculate communications-specific metrics
        self._calculate_communications_metrics(result, latest_income, latest_balance, latest_cash_flow)

        # Analyze subscription metrics if applicable
        self._analyze_subscription_metrics(result, financial_data)

        # Analyze content investments if applicable
        self._analyze_content_investments(result, financial_data)

        # Identify key industry risk factors
        self._identify_risk_factors(result, financial_data)

        # Evaluate competitive advantages
        self._evaluate_competitive_advantages(result, financial_data)

        return result

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using communications-specific methods

        Args:
            financial_data: Dictionary containing financial information
            valuation_method: Specific valuation method to use (if None, use the most appropriate)

        Returns:
            Dictionary with valuation results
        """
        # Default to EV/EBITDA if no method specified
        if valuation_method is None:
            valuation_method = "EV/EBITDA"

        valuation_result = {
            "sector": "Communication Services",
            "methods": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for communications sector valuation")
            return valuation_result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None

        # Perform EV/EBITDA valuation
        if valuation_method in ["EV/EBITDA", "all"]:
            self._calculate_ev_ebitda_valuation(valuation_result, latest_income, latest_balance, market_data)

        # Perform subscriber-based valuation
        if valuation_method in ["Subscriber", "all"]:
            self._calculate_subscriber_valuation(valuation_result, financial_data)

        # Perform content-based valuation
        if valuation_method in ["Content", "all"]:
            self._calculate_content_valuation(valuation_result, financial_data)

        # Perform P/E valuation (common for media companies)
        if valuation_method in ["P/E", "all"]:
            self._calculate_pe_valuation(valuation_result, latest_income, market_data)

        return valuation_result

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with communications sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks()
        benchmark_results = {
            "sector": "Communication Services",
            "comparisons": {},
            "score": 0.0,
            "strengths": [],
            "weaknesses": []
        }

        # Compare key communication metrics
        key_metrics = [
            "EBITDA Margin",
            "ARPU",
            "Churn Rate",
            "Subscriber Growth",
            "Content Cost Ratio",
            "Customer Acquisition Cost",
            "Debt/EBITDA"
        ]

        comparison_scores = []

        for metric in key_metrics:
            if metric in company_metrics and metric in benchmarks:
                company_value = company_metrics[metric]
                benchmark_value = benchmarks[metric]

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ["Churn Rate", "Customer Acquisition Cost", "Debt/EBITDA"]:
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

    def _calculate_communications_metrics(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                          cash_flow: Optional[pd.Series]) -> None:
        """Calculate communications-specific metrics"""
        metrics = {}

        # EBITDA Margin - key metric for communications sector
        if 'EBITDA' in income and 'Total Revenue' in income:
            ebitda_margin = income['EBITDA'] / income['Total Revenue']
            metrics['EBITDA Margin'] = ebitda_margin
        elif 'Operating Income' in income and 'Depreciation & Amortization' in income and 'Total Revenue' in income:
            ebitda = income['Operating Income'] + income['Depreciation & Amortization']
            ebitda_margin = ebitda / income['Total Revenue']
            metrics['EBITDA Margin'] = ebitda_margin

        # Capital Intensity
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            metrics['Capital Intensity'] = capital_intensity

            # Classify network infrastructure intensity
            if capital_intensity > 0.7:
                metrics['Infrastructure Model'] = "Heavy infrastructure (network-focused telecom)"
            elif capital_intensity > 0.4:
                metrics['Infrastructure Model'] = "Medium infrastructure (hybrid telecom/media)"
            else:
                metrics['Infrastructure Model'] = "Light infrastructure (media/digital content)"

        # Content Cost Ratio (for media companies)
        # Note: This would typically come from footnotes or management discussion
        # Placeholder for when that data is available
        metrics['Content Cost Ratio'] = None

        # R&D to Revenue (for tech-oriented communications companies)
        if 'Research and Development' in income and 'Total Revenue' in income:
            rd_to_revenue = income['Research and Development'] / income['Total Revenue']
            metrics['R&D to Revenue'] = rd_to_revenue

        # CAPEX Intensity
        if cash_flow is not None and 'Capital Expenditure' in cash_flow and 'Total Revenue' in income:
            capex_to_revenue = abs(cash_flow['Capital Expenditure']) / income['Total Revenue']
            metrics['CAPEX to Revenue'] = capex_to_revenue

        # Debt metrics
        if 'Total Debt' in balance:
            # Debt to EBITDA
            if 'EBITDA' in income and income['EBITDA'] > 0:
                debt_to_ebitda = balance['Total Debt'] / income['EBITDA']
                metrics['Debt/EBITDA'] = debt_to_ebitda
            elif 'Operating Income' in income and 'Depreciation & Amortization' in income:
                ebitda = income['Operating Income'] + income['Depreciation & Amortization']
                if ebitda > 0:
                    debt_to_ebitda = balance['Total Debt'] / ebitda
                    metrics['Debt/EBITDA'] = debt_to_ebitda

        # User metrics (placeholders - these typically come from company reports, not financial statements)
        metrics['ARPU'] = None  # Average Revenue Per User
        metrics['Subscriber Growth'] = None
        metrics['Churn Rate'] = None
        metrics['Customer Acquisition Cost'] = None

        result["industry_metrics"] = metrics

    def _analyze_subscription_metrics(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze subscription metrics for communications companies"""
        # Note: In a production system, this would extract data from company reports
        # or management discussions, which often contain user metrics

        # For this example, we'll initialize with placeholder values
        subscription_metrics = {
            "Total Subscribers": None,
            "Subscriber Growth (YoY)": None,
            "ARPU (Average Revenue Per User)": None,
            "Churn Rate": None,
            "Customer Lifetime Value": None,
            "Customer Acquisition Cost": None,
            "LTV/CAC Ratio": None
        }

        # If we had user metrics, we could calculate:
        # LTV = ARPU * (1 / Churn Rate)
        # LTV/CAC Ratio = LTV / CAC

        result["subscription_metrics"] = subscription_metrics

    def _analyze_content_investments(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze content investments for media companies"""
        # Note: In a production system, this would extract data from company reports
        # or management discussions about content investments and ROI

        content_analysis = {
            "Content Investment ($M)": None,
            "Content as % of Revenue": None,
            "Content Investment Growth (YoY)": None,
            "Content ROI Estimate": None,
            "Content Amortization Period": None,
            "Content Strategy Assessment": None
        }

        result["content_analysis"] = content_analysis

    def _identify_risk_factors(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Identify key industry risk factors for communications companies"""
        risks = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Check for high debt levels (common in telecom)
        if 'Total Debt' in balance and 'EBITDA' in income:
            if income['EBITDA'] > 0:
                debt_to_ebitda = balance['Total Debt'] / income['EBITDA']
                if debt_to_ebitda > 3.5:
                    risks.append({
                        "factor": "High Debt Burden",
                        "description": "Elevated debt to EBITDA levels increase financial risk and interest expense burden",
                        "severity": "High" if debt_to_ebitda > 4.5 else "Medium"
                    })

        # Check for high capital intensity (telecom risk)
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            if capital_intensity > 0.6:
                risks.append({
                    "factor": "High Capital Intensity",
                    "description": "Significant investment in infrastructure requires ongoing capital expenditure and creates high operational leverage",
                    "severity": "Medium"
                })

        # Check for low EBITDA margin
        if 'industry_metrics' in result and 'EBITDA Margin' in result['industry_metrics']:
            ebitda_margin = result['industry_metrics']['EBITDA Margin']
            if ebitda_margin < 0.25:  # Communications typically need high margins to cover infrastructure costs
                risks.append({
                    "factor": "Low EBITDA Margin",
                    "description": "Below-average EBITDA margin may not support necessary capital investments",
                    "severity": "High" if ebitda_margin < 0.15 else "Medium"
                })

        # Regulatory risk (common in communications)
        risks.append({
            "factor": "Regulatory Risk",
            "description": "Communications sector faces ongoing regulatory challenges including pricing controls, access requirements, and privacy regulations",
            "severity": "Medium"
        })

        # Technology disruption risk
        risks.append({
            "factor": "Technology Disruption",
            "description": "Rapid technological changes can render existing infrastructure or content distribution models obsolete",
            "severity": "Medium"
        })

        result["risk_factors"] = risks

    def _evaluate_competitive_advantages(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Evaluate potential competitive advantages for communications companies"""
        advantages = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Check for strong EBITDA margin (network effect/scale advantage)
        if 'industry_metrics' in result and 'EBITDA Margin' in result['industry_metrics']:
            ebitda_margin = result['industry_metrics']['EBITDA Margin']
            if ebitda_margin > 0.35:  # Above average for communications
                advantages.append({
                    "advantage": "Strong Operating Profitability",
                    "description": "Above-average EBITDA margins indicate pricing power or cost advantages",
                    "strength": "High" if ebitda_margin > 0.45 else "Medium"
                })

        # Network effects (placeholder - would come from qualitative assessment)
        advantages.append({
            "advantage": "Network Effects",
            "description": "Communications businesses often benefit from network effects as user base grows",
            "strength": "Medium"
        })

        # Spectrum assets (for telecom)
        # This would typically come from footnotes or management discussion

        # Content library (for media)
        # This would typically come from footnotes or management discussion

        result["competitive_advantages"] = advantages

    def _calculate_ev_ebitda_valuation(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                       market_data: Dict[str, Any]) -> None:
        """Calculate EV/EBITDA-based valuation (common for communications)"""
        valuation = {}

        # Get EBITDA
        ebitda = None
        if 'EBITDA' in income:
            ebitda = income['EBITDA']
        elif 'Operating Income' in income and 'Depreciation & Amortization' in income:
            ebitda = income['Operating Income'] + income['Depreciation & Amortization']

        # If we can't calculate EBITDA, skip this valuation method
        if ebitda is None or ebitda <= 0:
            valuation["error"] = "Unable to calculate EBITDA or EBITDA is negative"
            result["methods"]["EV/EBITDA"] = valuation
            return

        # Get sector EV/EBITDA multiple
        sector_multiple = 8.0  # Typical for communications
        if 'sector_metrics' in self.sector_metrics and 'benchmarks' in self.sector_metrics['sector_metrics']:
            if 'EV/EBITDA' in self.sector_metrics['sector_metrics']['benchmarks']:
                sector_multiple = self.sector_metrics['sector_metrics']['benchmarks']['EV/EBITDA']

        # Calculate enterprise value
        enterprise_value = ebitda * sector_multiple

        # Calculate equity value
        total_debt = balance.get('Total Debt', 0)
        cash = balance.get('Cash and Cash Equivalents', 0)
        equity_value = enterprise_value - total_debt + cash

        # Calculate per share value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            value_per_share = equity_value / shares_outstanding
        else:
            value_per_share = None

        # Store valuation result
        valuation["enterprise_value"] = enterprise_value
        valuation["equity_value"] = equity_value
        valuation["value_per_share"] = value_per_share
        valuation["multiple_used"] = sector_multiple
        valuation["ebitda"] = ebitda

        result["methods"]["EV/EBITDA"] = valuation

    def _calculate_subscriber_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate valuation based on subscriber metrics"""
        valuation = {}

        # In a real implementation, this would extract subscriber data from company reports
        # and apply appropriate multiples based on industry benchmarks

        # For this example, we'll initialize with a note
        valuation["note"] = "Subscriber-based valuation requires user metrics data typically found in company reports"
        valuation["key_metrics_needed"] = [
            "Total subscribers",
            "ARPU (Average Revenue Per User)",
            "Churn rate",
            "Subscriber growth rate"
        ]
        valuation["typical_valuation_approaches"] = [
            "EV/Subscriber: $500-1500 per subscriber for telecom",
            "EV/Subscriber: $150-600 per subscriber for media streaming",
            "DCF based on subscriber lifetime value calculations"
        ]

        result["methods"]["Subscriber"] = valuation

    def _calculate_content_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate valuation based on content assets"""
        valuation = {}

        # In a real implementation, this would analyze content libraries and their value
        # based on data from company reports and industry benchmarks

        # For this example, we'll initialize with a note
        valuation["note"] = "Content-based valuation requires data on content assets typically found in company reports"
        valuation["key_metrics_needed"] = [
            "Content library size and quality",
            "Content investment amounts",
            "Content engagement metrics",
            "Content monetization efficiency"
        ]
        valuation["typical_valuation_approaches"] = [
            "Content library replacement cost analysis",
            "Content ROI-based valuation",
            "Comparable content acquisition transactions"
        ]

        result["methods"]["Content"] = valuation

    def _calculate_pe_valuation(self, result: Dict[str, Any], income: pd.Series, market_data: Dict[str, Any]) -> None:
        """Calculate P/E-based valuation (common for media companies)"""
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
        sector_pe = 20.0  # Typical for communications, especially media
        if 'sector_metrics' in self.sector_metrics and 'benchmarks' in self.sector_metrics['sector_metrics']:
            if 'P/E' in self.sector_metrics['sector_metrics']['benchmarks']:
                sector_pe = self.sector_metrics['sector_metrics']['benchmarks']['P/E']

        # Calculate equity value
        equity_value = net_income * sector_pe

        # Calculate per share value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            value_per_share = equity_value / shares_outstanding
        else:
            value_per_share = None

        # Store valuation result
        valuation["equity_value"] = equity_value
        valuation["value_per_share"] = value_per_share
        valuation["multiple_used"] = sector_pe
        valuation["net_income"] = net_income

        result["methods"]["P/E"] = valuation


# Register with factory
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory

SectorFactory.register_analyzer("Communication Services", CommunicationsSectorAnalyzer)