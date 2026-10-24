import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.valuation.sector_factor import SectorAnalyzer, SectorFactory
from StockAnalysisSystem.src.config import SECTOR_METRICS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tech_sector')


class TechnologySectorAnalyzer(SectorAnalyzer):
    """
    Analyzer for Technology sector companies.

    Technology companies typically have:
    - High growth rates
    - High gross margins
    - Significant R&D expenses
    - Low or negative earnings in early stages
    - Subscription-based or recurring revenue models
    """

    def __init__(self, sector_name: str = "Technology"):
        """Initialize with Technology sector parameters"""
        super().__init__(sector_name)

        # Specific tech subsectors
        self.subsectors = ["Software", "Hardware", "Semiconductors", "Internet", "IT Services"]

        # Specific metrics important for tech sector
        self.specific_metrics = {
            "r&d_intensity": "R&D expenses as percentage of revenue",
            "gross_margin": "Gross profit as percentage of revenue",
            "recurring_revenue": "Percentage of revenue that is recurring",
            "customer_acquisition_cost": "Cost to acquire a new customer",
            "customer_lifetime_value": "Expected revenue from a customer over time",
            "churn_rate": "Rate at which customers stop using the product",
            "arpu": "Average Revenue Per User"
        }

        logger.info("Initialized Technology sector analyzer")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:

        metrics = financial_data.get("metrics", {})

        """
        Analyze a technology company using sector-specific metrics

        Args:
            financial_data: Dictionary containing financial statements and metrics

        Returns:
            Dictionary with analysis results
        """
        results = {
            "sector": self.sector_name,
            "key_metrics": {},
            "analysis": {},
            "growth_assessment": {},
            "efficiency_assessment": {},
            "risk_factors": {},
            "recommendations": []
        }

        # Extract relevant financial information
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')
        market_data = financial_data.get('market_data', {})

        # Check if we have the necessary data
        if income_stmt is None or income_stmt.empty:
            logger.warning("Missing income statement data for technology analysis")
            return results

        if balance_sheet is None or balance_sheet.empty:
            logger.warning("Missing balance sheet data for technology analysis")
            return results

        # Get the most recent financial data (first column)
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else pd.Series()

        # Calculate key tech metrics
        revenue = self._get_value_with_alternatives(latest_income, ['Total Revenue', 'Revenue', 'Net Revenue'])
        gross_profit = self._get_value_with_alternatives(latest_income, ['Gross Profit', 'Gross Income'])
        r_and_d = self._get_value_with_alternatives(latest_income, ['Research and Development', 'R&D Expense'])

        if revenue > 0:
            # Calculate tech-specific ratios
            results["key_metrics"]["gross_margin"] = gross_profit / revenue if revenue else 0
            results["key_metrics"]["r&d_intensity"] = r_and_d / revenue if revenue else 0

            # Extract or calculate other key metrics
            results["key_metrics"]["ps_ratio"] = metrics.get('ps_ratio', 0)
            results["key_metrics"]["revenue_growth"] = metrics.get('revenue_growth', 0)
            results["key_metrics"]["operating_margin"] = latest_income.get('Operating Income',
                                                                           0) / revenue if revenue else 0

            # Special consideration for tech: revenue per employee
            employees = financial_data.get('company_info', {}).get('employees', 0)
            if employees > 0:
                results["key_metrics"]["revenue_per_employee"] = revenue / employees

        # Analyze growth potential
        revenue_growth = metrics.get('revenue_growth', 0)
        industry_avg_growth = self.sector_config.get('benchmarks', {}).get('Revenue Growth', 0.15)

        if revenue_growth > industry_avg_growth * 1.5:
            growth_assessment = "Exceptional growth"
        elif revenue_growth > industry_avg_growth:
            growth_assessment = "Above average growth"
        elif revenue_growth > industry_avg_growth * 0.5:
            growth_assessment = "Average growth"
        else:
            growth_assessment = "Below average growth"

        results["growth_assessment"] = {
            "rating": growth_assessment,
            "company_growth": revenue_growth,
            "industry_avg_growth": industry_avg_growth
        }

        # Analyze operational efficiency
        gross_margin = results["key_metrics"].get("gross_margin", 0)
        industry_avg_margin = self.sector_config.get('benchmarks', {}).get('Gross Margin', 0.50)

        if gross_margin > industry_avg_margin * 1.2:
            efficiency_assessment = "Superior efficiency"
        elif gross_margin > industry_avg_margin:
            efficiency_assessment = "Above average efficiency"
        elif gross_margin > industry_avg_margin * 0.8:
            efficiency_assessment = "Average efficiency"
        else:
            efficiency_assessment = "Below average efficiency"

        results["efficiency_assessment"] = {
            "rating": efficiency_assessment,
            "company_margin": gross_margin,
            "industry_avg_margin": industry_avg_margin
        }

        # Identify key risk factors for tech companies
        results["risk_factors"] = {
            "technological_disruption": self._assess_disruption_risk(financial_data),
            "competition": self._assess_competition_risk(financial_data),
            "regulation": self._assess_regulation_risk(financial_data)
        }

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:

        metrics = financial_data.get("metrics", {})

        """
        Calculate valuation for a technology company using appropriate methods

        Args:
            financial_data: Dictionary containing financial information
            valuation_method: Optional specific method to use

        Returns:
            Dictionary with valuation results
        """
        # If no method specified, use the first preferred method for this sector
        if not valuation_method:
            preferred_methods = self.get_preferred_valuation_methods()
            valuation_method = preferred_methods[0] if preferred_methods else "DCF"

        valuation_results = {
            "method": valuation_method,
            "value": None,
            "metrics_used": [],
            "assumptions": {},
            "sensitivity": {}
        }

        # Extract necessary data for valuation
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')
        market_data = financial_data.get('market_data', {})

        if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
            logger.warning("Insufficient financial data for valuation")
            return valuation_results

        # Get historical years for growth calculation
        years = list(income_stmt.keys())
        years.sort(reverse=True)  # Most recent first

        if len(years) < 2:
            logger.warning("Insufficient historical data for growth calculation")
            # Use sector average growth if not enough data
            revenue_growth = self.sector_config.get('benchmarks', {}).get('Revenue Growth', 0.15)
        else:
            # Calculate historical revenue growth
            revenue_values = [income_stmt[year].get('Total Revenue', 0) for year in years if year in income_stmt]
            if len(revenue_values) >= 2 and revenue_values[1] > 0:  # Check if we have at least 2 values
                revenue_growth = (revenue_values[0] / revenue_values[1]) - 1
            else:
                revenue_growth = self.sector_config.get('benchmarks', {}).get('Revenue Growth', 0.15)

        # Different valuation methods
        if valuation_method == "DCF":
            # For tech companies, we use a multi-stage DCF model
            # with higher initial growth rate that tapers down

            # Get latest year data
            latest_year = years[0] if years else None
            latest_income = income_stmt.get(latest_year, {}) if latest_year else {}
            latest_cash_flow = cash_flow.get(latest_year, {}) if latest_year else {}

            # Get or calculate FCF
            fcf = latest_cash_flow.get('Free Cash Flow', 0)
            if fcf == 0:
                # Calculate FCF if not available
                operating_cash_flow = latest_cash_flow.get('Operating Cash Flow', 0)
                capital_expenditure = latest_cash_flow.get('Capital Expenditure', 0)
                fcf = operating_cash_flow + capital_expenditure  # CapEx is negative

            # For tech companies, we often use a 2-stage growth model
            # High growth stage followed by terminal growth
            forecast_years = self.dcf_parameters.get('forecast_years', 7)  # Tech typically uses longer forecast period
            terminal_growth = self.dcf_parameters.get('terminal_growth_rate', 0.03)
            discount_rate = self.dcf_parameters.get('default_discount_rate', 0.12)  # Higher for tech due to risk

            # Initial growth rate based on historical data or sector benchmark, capped at reasonable range
            initial_growth = min(max(revenue_growth, 0.05), 0.30)  # Cap between 5% and 30%

            # Growth transition - linear decrease from initial to terminal growth
            growth_rates = []
            for year in range(forecast_years):
                # Linear transition from initial to terminal growth
                growth_rate = initial_growth - (initial_growth - terminal_growth) * (year / (forecast_years - 1))
                growth_rates.append(growth_rate)

            # Calculate future FCF
            future_fcf = []
            current_fcf = fcf
            for growth_rate in growth_rates:
                current_fcf = current_fcf * (1 + growth_rate)
                future_fcf.append(current_fcf)

            # Calculate terminal value
            terminal_value = future_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

            # Calculate present value of FCFs and terminal value
            total_pv = 0
            for i, cf in enumerate(future_fcf):
                total_pv += cf / ((1 + discount_rate) ** (i + 1))

            # Add present value of terminal value
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** forecast_years)
            enterprise_value = total_pv + terminal_value_pv

            # Calculate equity value
            latest_balance = balance_sheet.get(latest_year, {}) if latest_year else {}
            total_debt = latest_balance.get('Total Debt', 0)
            cash_and_equivalents = latest_balance.get('Cash and Cash Equivalents', 0)
            equity_value = enterprise_value - total_debt + cash_and_equivalents

            # Calculate per share value if shares outstanding available
            shares_outstanding = financial_data.get('company_info', {}).get('shares_outstanding', None)
            per_share_value = equity_value / shares_outstanding if shares_outstanding else None

            # Save results
            valuation_results["value"] = per_share_value if per_share_value else equity_value
            valuation_results["enterprise_value"] = enterprise_value
            valuation_results["equity_value"] = equity_value
            valuation_results["metrics_used"] = ["FCF", "Revenue Growth", "WACC"]
            valuation_results["assumptions"] = {
                "initial_growth_rate": initial_growth,
                "terminal_growth_rate": terminal_growth,
                "discount_rate": discount_rate,
                "forecast_years": forecast_years
            }

            # Add sensitivity analysis
            valuation_results["sensitivity"] = self._calculate_dcf_sensitivity(
                fcf,
                initial_growth,
                terminal_growth,
                discount_rate,
                forecast_years,
                total_debt,
                cash_and_equivalents,
                shares_outstanding
            )

        elif valuation_method == "EV/Revenue":
            # Revenue multiple valuation - often used for high-growth tech companies
            latest_year = years[0] if years else None
            latest_income = income_stmt.get(latest_year, {}) if latest_year else {}

            revenue = latest_income.get('Total Revenue', 0)
            if revenue <= 0:
                logger.warning("Revenue is zero or negative, cannot apply EV/Revenue multiple")
                return valuation_results

            # Determine appropriate EV/Revenue multiple based on growth and margins
            revenue_growth = metrics.get('revenue_growth', 0.15)  # Default to 15% if not available
            gross_margin = metrics.get('gross_margin', 0.5)  # Default to 50% if not available

            # Base multiple on the characteristics of the company
            # High growth + high margin = higher multiple
            base_multiple = 5.0  # Average tech EV/Revenue multiple

            # Adjust for growth
            if revenue_growth > 0.3:  # >30% growth
                growth_factor = 1.5
            elif revenue_growth > 0.2:  # >20% growth
                growth_factor = 1.25
            elif revenue_growth > 0.1:  # >10% growth
                growth_factor = 1.0
            else:
                growth_factor = 0.75

            # Adjust for margins
            if gross_margin > 0.7:  # >70% margin
                margin_factor = 1.5
            elif gross_margin > 0.5:  # >50% margin
                margin_factor = 1.25
            elif gross_margin > 0.3:  # >30% margin
                margin_factor = 1.0
            else:
                margin_factor = 0.75

            # Calculate adjusted multiple
            adjusted_multiple = base_multiple * growth_factor * margin_factor

            # Calculate enterprise value
            enterprise_value = revenue * adjusted_multiple

            # Calculate equity value
            latest_balance = balance_sheet.get(latest_year, {}) if latest_year else {}
            total_debt = latest_balance.get('Total Debt', 0)
            cash_and_equivalents = latest_balance.get('Cash and Cash Equivalents', 0)
            equity_value = enterprise_value - total_debt + cash_and_equivalents

            # Calculate per share value if shares outstanding available
            shares_outstanding = financial_data.get('company_info', {}).get('shares_outstanding', None)
            per_share_value = equity_value / shares_outstanding if shares_outstanding else None

            # Save results
            valuation_results["value"] = per_share_value if per_share_value else equity_value
            valuation_results["enterprise_value"] = enterprise_value
            valuation_results["equity_value"] = equity_value
            valuation_results["ev_revenue_multiple"] = adjusted_multiple
            valuation_results["metrics_used"] = ["Revenue", "Revenue Growth", "Gross Margin"]
            valuation_results["assumptions"] = {
                "base_multiple": base_multiple,
                "growth_factor": growth_factor,
                "margin_factor": margin_factor,
                "adjusted_multiple": adjusted_multiple
            }

            # Add sensitivity analysis
            valuation_results["sensitivity"] = {
                "low_multiple": revenue * (adjusted_multiple * 0.8),
                "base_multiple": revenue * adjusted_multiple,
                "high_multiple": revenue * (adjusted_multiple * 1.2)
            }

        elif valuation_method == "PEG":
            # Price/Earnings to Growth ratio - useful for growth tech companies
            latest_year = years[0] if years else None
            latest_income = income_stmt.get(latest_year, {}) if latest_year else {}

            earnings = latest_income.get('Net Income', 0)
            if earnings <= 0:
                logger.warning("Earnings are zero or negative, cannot apply PEG ratio")
                return valuation_results

            # Get growth rate and P/E
            growth_rate = metrics.get('revenue_growth', 0.15)  # Default to 15% if not available
            pe_ratio = metrics.get('pe_ratio', 20)  # Default to 20 if not available

            # Calculate PEG
            peg_ratio = pe_ratio / (growth_rate * 100)  # Convert growth to percentage

            # For fair valuation, PEG should be around 1.0
            fair_pe = growth_rate * 100

            # Calculate fair value
            shares_outstanding = financial_data.get('company_info', {}).get('shares_outstanding', None)
            if not shares_outstanding:
                logger.warning("Shares outstanding not available, cannot calculate per share value")
                return valuation_results

            earnings_per_share = earnings / shares_outstanding
            fair_value = earnings_per_share * fair_pe

            # Save results
            valuation_results["value"] = fair_value
            valuation_results["peg_ratio"] = peg_ratio
            valuation_results["fair_pe"] = fair_pe
            valuation_results["metrics_used"] = ["Earnings", "Growth Rate", "P/E Ratio"]
            valuation_results["assumptions"] = {
                "growth_rate": growth_rate,
                "current_pe": pe_ratio,
                "fair_peg_ratio": 1.0
            }

            # Add sensitivity analysis
            valuation_results["sensitivity"] = {
                "low_growth": earnings_per_share * (growth_rate * 80),  # 20% lower growth
                "base_growth": earnings_per_share * (growth_rate * 100),
                "high_growth": earnings_per_share * (growth_rate * 120)  # 20% higher growth
            }

        return valuation_results

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with technology sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks()
        key_ratios = self.get_key_ratios()

        results = {
            "overall_score": 0,
            "metrics_comparison": {},
            "strengths": [],
            "weaknesses": [],
            "peer_percentile": {}
        }

        # Compare each key metric with benchmark
        for ratio in key_ratios:
            if ratio in company_metrics and ratio in benchmarks:
                company_value = company_metrics[ratio]
                benchmark_value = benchmarks[ratio]

                # Determine if higher is better (default) or lower is better
                higher_is_better = True

                # Specific cases where lower is better for tech sector
                if ratio in ['Debt/Equity', 'Customer Acquisition Cost']:
                    higher_is_better = False

                # Calculate percentage difference
                if benchmark_value > 0:
                    if higher_is_better:
                        perc_diff = (company_value / benchmark_value - 1) * 100
                    else:
                        perc_diff = (benchmark_value / company_value - 1) * 100
                else:
                    perc_diff = 0

                # Determine rating
                if perc_diff > 20:
                    rating = "Excellent"
                elif perc_diff > 0:
                    rating = "Good"
                elif perc_diff > -20:
                    rating = "Average"
                else:
                    rating = "Poor"

                # Add to results
                results["metrics_comparison"][ratio] = {
                    "company_value": company_value,
                    "benchmark_value": benchmark_value,
                    "percentage_difference": perc_diff,
                    "rating": rating
                }

                # Add to strengths or weaknesses
                if rating in ["Excellent", "Good"]:
                    results["strengths"].append(
                        f"{ratio}: {rating} ({perc_diff:.1f}% {'above' if perc_diff > 0 else 'below'} benchmark)")
                elif rating == "Poor":
                    results["weaknesses"].append(
                        f"{ratio}: {rating} ({abs(perc_diff):.1f}% {'below' if perc_diff < 0 else 'above'} benchmark)")

        # Calculate overall score based on comparison with benchmarks
        if results["metrics_comparison"]:
            score = 0
            for metric_data in results["metrics_comparison"].values():
                if metric_data["rating"] == "Excellent":
                    score += 4
                elif metric_data["rating"] == "Good":
                    score += 3
                elif metric_data["rating"] == "Average":
                    score += 2
                else:  # Poor
                    score += 1

            # Convert to 0-100 scale
            max_possible = 4 * len(results["metrics_comparison"])
            results["overall_score"] = (score / max_possible) * 100

            # Add percentile ranking
            if results["overall_score"] >= 80:
                results["peer_percentile"] = "Top 20%"
            elif results["overall_score"] >= 60:
                results["peer_percentile"] = "Top 40%"
            elif results["overall_score"] >= 40:
                results["peer_percentile"] = "Middle 20%"
            elif results["overall_score"] >= 20:
                results["peer_percentile"] = "Bottom 40%"
            else:
                results["peer_percentile"] = "Bottom 20%"

        return results

    def _assess_disruption_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk of technological disruption"""
        # Logic to assess disruption risk based on R&D intensity, patents, etc.
        metrics = financial_data.get('metrics', {})
        income_stmt = financial_data.get('income_statement', {})

        # Get the latest year data
        latest_year = list(income_stmt.keys())[0] if income_stmt else None
        latest_income = income_stmt.get(latest_year, {}) if latest_year else {}

        # Calculate R&D intensity
        revenue = latest_income.get('Total Revenue', 0)
        r_and_d = latest_income.get('Research & Development', 0)
        r_and_d_intensity = r_and_d / revenue if revenue > 0 else 0

        # Industry average R&D intensity
        avg_r_and_d_intensity = 0.15  # Typical for tech sector

        # Determine risk level
        if r_and_d_intensity > avg_r_and_d_intensity * 1.5:
            risk_level = "Low"
            explanation = "High R&D investment suggests the company is actively innovating"
        elif r_and_d_intensity > avg_r_and_d_intensity:
            risk_level = "Moderate-Low"
            explanation = "Above average R&D investment provides some protection against disruption"
        elif r_and_d_intensity > avg_r_and_d_intensity * 0.5:
            risk_level = "Moderate"
            explanation = "Average R&D investment may not be sufficient in rapidly changing tech landscape"
        else:
            risk_level = "High"
            explanation = "Low R&D investment increases vulnerability to technological disruption"

        return {
            "level": risk_level,
            "explanation": explanation,
            "r&d_intensity": r_and_d_intensity,
            "industry_avg": avg_r_and_d_intensity
        }

    def _assess_competition_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk from competition"""
        # Assess market share trend, margin compression, etc.
        metrics = financial_data.get('metrics', {})
        income_stmt = financial_data.get('income_statement', {})

        # Get multiple years of data for trend analysis
        years = list(income_stmt.keys())
        years.sort(reverse=True)  # Most recent first

        # Look at gross margin trend - declining margins often indicate increasing competition
        gross_margins = []
        for year in years[:3]:  # Look at last 3 years if available
            if year in income_stmt:
                revenue = income_stmt[year].get('Total Revenue', 0)
                gross_profit = income_stmt[year].get('Gross Profit', 0)
                if revenue > 0:
                    gross_margins.append(gross_profit / revenue)

        # Determine margin trend
        margin_trend = "stable"
        if len(gross_margins) >= 2:
            if gross_margins[0] < gross_margins[-1] * 0.9:  # 10% or more decline
                margin_trend = "declining"
            elif gross_margins[0] > gross_margins[-1] * 1.1:  # 10% or more increase
                margin_trend = "improving"

        # Check market share data if available
        market_share = metrics.get('market_share', None)
        market_share_trend = metrics.get('market_share_trend', "stable")

        # Determine risk level
        if margin_trend == "declining" and market_share_trend == "declining":
            risk_level = "High"
            explanation = "Declining margins and market share suggest intense competition"
        elif margin_trend == "declining" or market_share_trend == "declining":
            risk_level = "Moderate-High"
            explanation = "Some indicators suggest increasing competitive pressure"
        elif margin_trend == "stable" and market_share_trend == "stable":
            risk_level = "Moderate"
            explanation = "Stable margins and market share, but tech markets tend to be competitive"
        else:
            risk_level = "Low"
            explanation = "Improving margins and/or market share indicate strong competitive position"

        return {
            "level": risk_level,
            "explanation": explanation,
            "margin_trend": margin_trend,
            "market_share_trend": market_share_trend
        }

    def _assess_regulation_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk from regulatory changes"""
        # Assess exposure to privacy regulations, antitrust, data protection, etc.
        company_info = financial_data.get('company_info', {})

        # For tech companies, key regulatory areas are:
        # 1. Data privacy/protection (GDPR, CCPA, etc.)
        # 2. Antitrust/competition concerns
        # 3. Content moderation (for platforms)
        # 4. AI and algorithm regulation

        # Determine which sub-sector the company is in
        industry = company_info.get('industry', '')
        sub_sector = None

        if 'software' in industry.lower() or 'saas' in industry.lower():
            sub_sector = "Software"
        elif 'hardware' in industry.lower() or 'device' in industry.lower():
            sub_sector = "Hardware"
        elif 'semiconductor' in industry.lower() or 'chip' in industry.lower():
            sub_sector = "Semiconductors"
        elif 'internet' in industry.lower() or 'platform' in industry.lower() or 'social media' in industry.lower():
            sub_sector = "Internet/Platforms"
        elif 'service' in industry.lower() or 'consulting' in industry.lower():
            sub_sector = "IT Services"
        else:
            sub_sector = "Other Tech"

        # Different sub-sectors have different regulatory risks
        if sub_sector == "Internet/Platforms":
            risk_level = "High"
            explanation = "Internet platforms face increasing scrutiny on privacy, content moderation, and antitrust issues"
        elif sub_sector == "Software":
            risk_level = "Moderate"
            explanation = "Software companies face moderate regulatory risks around data protection and security"
        elif sub_sector == "Semiconductors":
            risk_level = "Moderate-High"
            explanation = "Semiconductor companies face export controls, geopolitical tensions, and supply chain regulations"
        elif sub_sector == "Hardware":
            risk_level = "Moderate"
            explanation = "Hardware manufacturers face environmental regulations and right-to-repair legislation"
        else:
            risk_level = "Moderate-Low"
            explanation = "IT services face fewer direct regulatory challenges than other tech sub-sectors"

        return {
            "level": risk_level,
            "explanation": explanation,
            "sub_sector": sub_sector
        }

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []

        # Extract key metrics from analysis
        key_metrics = analysis_results.get('key_metrics', {})
        growth_assessment = analysis_results.get('growth_assessment', {})
        efficiency_assessment = analysis_results.get('efficiency_assessment', {})
        risk_factors = analysis_results.get('risk_factors', {})

        # Recommendations based on growth
        growth_rating = growth_assessment.get('rating', '')
        if growth_rating in ["Below average growth", "Average growth"]:
            recommendations.append(
                "Consider initiatives to accelerate revenue growth, potentially through new product development or market expansion")

        # Recommendations based on efficiency
        efficiency_rating = efficiency_assessment.get('rating', '')
        if efficiency_rating in ["Below average efficiency", "Average efficiency"]:
            recommendations.append(
                "Focus on improving operational efficiency to bring gross margins closer to industry benchmarks")

        # Recommendations based on R&D
        r_and_d_intensity = key_metrics.get('r&d_intensity', 0)
        if r_and_d_intensity < 0.1:  # Less than 10% of revenue
            recommendations.append(
                "Evaluate R&D investment strategy as current levels may be insufficient to maintain competitive position in tech sector")

        # Recommendations based on risk factors
        disruption_risk = risk_factors.get('technological_disruption', {}).get('level', '')
        competition_risk = risk_factors.get('competition', {}).get('level', '')
        regulation_risk = risk_factors.get('regulation', {}).get('level', '')

        if disruption_risk in ["High", "Moderate"]:
            recommendations.append(
                "Increase focus on innovation and emerging technologies to mitigate risk of technological disruption")

        if competition_risk in ["High", "Moderate-High"]:
            recommendations.append(
                "Strengthen competitive differentiation through product features, service quality, or pricing strategy")

        if regulation_risk in ["High", "Moderate-High"]:
            recommendations.append(
                "Develop robust compliance frameworks and engage proactively with regulatory developments")

        # If we don't have enough recommendations, add general ones
        if len(recommendations) < 3:
            recommendations.append(
                "Consider strategic partnerships or acquisitions to enhance technological capabilities or market access")
            recommendations.append("Evaluate subscription-based revenue models to increase recurring revenue streams")

        return recommendations

    def _calculate_dcf_sensitivity(self, base_fcf, initial_growth, terminal_growth, discount_rate,
                                   forecast_years, total_debt, cash_and_equivalents, shares_outstanding):
        """Calculate sensitivity analysis for DCF model"""
        sensitivity = {}

        # Sensitivity to growth rate changes
        growth_scenarios = {
            "low_growth": initial_growth * 0.8,  # 20% lower
            "base_growth": initial_growth,
            "high_growth": initial_growth * 1.2  # 20% higher
        }

        # Sensitivity to discount rate changes
        discount_scenarios = {
            "low_discount": discount_rate * 0.9,  # 10% lower
            "base_discount": discount_rate,
            "high_discount": discount_rate * 1.1  # 10% higher
        }

        # Calculate enterprise value for each scenario
        enterprise_values = {}
        for growth_key, growth_val in growth_scenarios.items():
            for discount_key, discount_val in discount_scenarios.items():
                scenario_key = f"{growth_key}_{discount_key}"

                # Calculate with new parameters
                future_fcf = []
                current_fcf = base_fcf

                # Linear transition from initial to terminal growth
                for year in range(forecast_years):
                    growth_rate = growth_val - (growth_val - terminal_growth) * (year / (forecast_years - 1))
                    current_fcf = current_fcf * (1 + growth_rate)
                    future_fcf.append(current_fcf)

                # Terminal value
                terminal_value = future_fcf[-1] * (1 + terminal_growth) / (discount_val - terminal_growth)

                # Present value
                total_pv = 0
                for i, cf in enumerate(future_fcf):
                    total_pv += cf / ((1 + discount_val) ** (i + 1))

                # Add terminal value
                terminal_value_pv = terminal_value / ((1 + discount_val) ** forecast_years)
                ev = total_pv + terminal_value_pv

                # Calculate equity value
                equity_value = ev - total_debt + cash_and_equivalents

                # Calculate per share value if possible
                per_share_value = equity_value / shares_outstanding if shares_outstanding else equity_value

                enterprise_values[scenario_key] = per_share_value if shares_outstanding else equity_value

        # Add to sensitivity results
        sensitivity["scenarios"] = enterprise_values

        # Extract key scenarios for quick reference
        sensitivity["base_case"] = enterprise_values.get("base_growth_base_discount")
        sensitivity["worst_case"] = enterprise_values.get("low_growth_high_discount")
        sensitivity["best_case"] = enterprise_values.get("high_growth_low_discount")

        return sensitivity

    def _get_value_with_alternatives(self, data_series, field_alternatives):
        """
        Try to get a value from a series using multiple alternative field names

        Args:
            data_series: Pandas Series containing the financial data
            field_alternatives: List of possible field names to try

        Returns:
            The value if found, None otherwise
        """
        if data_series is None:
            return None

        for field in field_alternatives:
            if field in data_series:
                return data_series[field]
        return None

# Register the Technology sector analyzer with the factory
SectorFactory.register_analyzer("Technology", TechnologySectorAnalyzer)

# Also register common aliases
SectorFactory.register_analyzer("Information Technology", TechnologySectorAnalyzer)
SectorFactory.register_analyzer("IT", TechnologySectorAnalyzer)
SectorFactory.register_analyzer("Tech", TechnologySectorAnalyzer)