import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add project root to path
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
logger = logging.getLogger('healthcare_sector')


class HealthcareSectorAnalyzer(SectorAnalyzer):
    """
    Specialized analyzer for Healthcare sector companies.

    Healthcare companies have unique characteristics that require specific analysis:
    1. High R&D expenditures with uncertain returns and long development timelines
    2. Patent protection creates temporary monopolies but with defined lifespans
    3. Regulatory approvals create binary outcomes for clinical-stage companies
    4. Different business models within healthcare (pharma, biotech, medical devices, services)
    5. Demographic trends and public policy heavily influence growth
    """

    def __init__(self, sector_name: str = "Healthcare"):
        """Initialize the Healthcare sector analyzer"""
        super().__init__(sector_name)
        self.data_loader = DataLoader()
        logger.info("Healthcare Sector Analyzer initialized")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a healthcare company using sector-specific metrics and methods

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with analysis results
        """
        results = {
            'sector': self.sector_name,
            'subsector_analysis': self._determine_healthcare_subsector(financial_data),
            'key_metrics': self._calculate_healthcare_metrics(financial_data),
            'risk_assessment': self._assess_healthcare_risks(financial_data),
            'growth_analysis': self._analyze_growth_potential(financial_data),
            'innovation_assessment': self._assess_innovation_metrics(financial_data),
            'regulatory_exposure': self._assess_regulatory_exposure(financial_data)
        }

        return results

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using sector-specific methods

        Args:
            financial_data: Dictionary containing company financial information
            valuation_method: Specific valuation method to use (if None, use the most appropriate)

        Returns:
            Dictionary with valuation results
        """
        # Get company info
        company_info = financial_data.get('company_info', {})
        ticker = company_info.get('symbol', 'Unknown')

        # Get subsector for more tailored analysis
        subsector = self._determine_healthcare_subsector(financial_data)['subsector']

        # Determine valuation method if not specified
        if valuation_method is None:
            if subsector == 'Biotech' or subsector == 'Early-Stage Pharma':
                valuation_method = 'pipeline_based_dcf'
            elif subsector == 'Medical Devices':
                valuation_method = 'multi_stage_dcf'
            elif subsector == 'Healthcare Services':
                valuation_method = 'standard_dcf'
            else:
                valuation_method = 'multi_stage_dcf'

        # Apply appropriate valuation method
        if valuation_method == 'pipeline_based_dcf':
            return self._pipeline_based_valuation(financial_data)
        elif valuation_method == 'multi_stage_dcf':
            return self._multi_stage_dcf_valuation(financial_data)
        elif valuation_method == 'standard_dcf':
            return self._standard_dcf_valuation(financial_data)
        elif valuation_method == 'ebitda_multiple':
            return self._ebitda_multiple_valuation(financial_data)
        else:
            logger.warning(f"Unsupported valuation method {valuation_method} for {ticker}. Using multi-stage DCF.")
            return self._multi_stage_dcf_valuation(financial_data)

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with healthcare sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks()

        benchmark_results = {
            'sector': self.sector_name,
            'metrics_comparison': {},
            'overall_score': 0.0,
            'strengths': [],
            'weaknesses': []
        }

        # Define key metrics to analyze
        key_metrics = {
            'R&D/Revenue': {'benchmark': 0.15, 'higher_is_better': True, 'importance': 'high'},
            'Gross Margin': {'benchmark': 0.65, 'higher_is_better': True, 'importance': 'high'},
            'Operating Margin': {'benchmark': 0.18, 'higher_is_better': True, 'importance': 'medium'},
            'ROE': {'benchmark': 0.18, 'higher_is_better': True, 'importance': 'medium'},
            'Revenue Growth': {'benchmark': 0.12, 'higher_is_better': True, 'importance': 'high'},
            'Debt/Equity': {'benchmark': 0.60, 'higher_is_better': False, 'importance': 'medium'},
            'Current Ratio': {'benchmark': 2.0, 'higher_is_better': True, 'importance': 'low'},
        }

        # Compare each metric with benchmark
        score_sum = 0
        score_weights = 0

        for metric, details in key_metrics.items():
            if metric in company_metrics:
                company_value = company_metrics[metric]
                benchmark_value = details['benchmark']

                # Calculate percentage difference
                if benchmark_value != 0:
                    pct_diff = (company_value - benchmark_value) / benchmark_value * 100
                else:
                    pct_diff = 0

                # Determine if the comparison is favorable
                if details['higher_is_better']:
                    is_favorable = company_value >= benchmark_value
                else:
                    is_favorable = company_value <= benchmark_value

                # Calculate score (0-100)
                if details['higher_is_better']:
                    if benchmark_value > 0:
                        score = min(100, (company_value / benchmark_value) * 100)
                    else:
                        score = 100 if company_value > 0 else 0
                else:
                    if company_value > 0:
                        score = min(100, (benchmark_value / company_value) * 100)
                    else:
                        score = 100  # Perfect score for zero debt, etc.

                # Assign weight based on importance
                weight = 3 if details['importance'] == 'high' else 2 if details['importance'] == 'medium' else 1

                # Add to running weighted score
                score_sum += score * weight
                score_weights += weight

                # Add to results
                benchmark_results['metrics_comparison'][metric] = {
                    'company_value': company_value,
                    'benchmark': benchmark_value,
                    'pct_difference': pct_diff,
                    'favorable': is_favorable,
                    'score': score,
                    'importance': details['importance']
                }

                # Add to strengths or weaknesses
                if is_favorable and abs(pct_diff) > 10:
                    benchmark_results['strengths'].append(f"{metric}: {pct_diff:.1f}% better than sector average")
                elif not is_favorable and abs(pct_diff) > 10:
                    benchmark_results['weaknesses'].append(f"{metric}: {pct_diff:.1f}% worse than sector average")

        # Calculate overall score
        if score_weights > 0:
            benchmark_results['overall_score'] = score_sum / score_weights

        return benchmark_results

    def _determine_healthcare_subsector(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the specific healthcare subsector based on company characteristics

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with subsector classification and confidence
        """
        # Extract key financial data for classification
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        company_info = financial_data.get('company_info', {})

        # Default result
        result = {
            'subsector': 'General Healthcare',
            'confidence': 0.5,
            'characteristics': []
        }

        # Check if we have necessary data
        if income_stmt is None or income_stmt.empty:
            return result

        # Get industry description if available
        industry = company_info.get('industry', '').lower()
        business_description = company_info.get('description', '').lower()

        # Extract key metrics for classification
        metrics = {}

        # Try to get R&D expense
        try:
            if 'Research and Development' in income_stmt.index:
                metrics['rd_expense'] = income_stmt.iloc[income_stmt.index.get_loc('Research and Development'), 0]

                # Calculate R&D to Revenue ratio
                if 'Total Revenue' in income_stmt.index:
                    revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                    if revenue > 0:
                        metrics['rd_to_revenue'] = metrics['rd_expense'] / revenue
        except:
            pass

        # Try to get Gross Margin
        try:
            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gross_profit = income_stmt.iloc[income_stmt.index.get_loc('Gross Profit'), 0]
                revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                if revenue > 0:
                    metrics['gross_margin'] = gross_profit / revenue
        except:
            pass

        # Try to get Operating Margin
        try:
            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.iloc[income_stmt.index.get_loc('Operating Income'), 0]
                revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                if revenue > 0:
                    metrics['operating_margin'] = operating_income / revenue
        except:
            pass

        # Check Net Income
        try:
            if 'Net Income' in income_stmt.index:
                metrics['net_income'] = income_stmt.iloc[income_stmt.index.get_loc('Net Income'), 0]
        except:
            pass

        # Classification logic
        subsector = 'General Healthcare'
        confidence = 0.5
        characteristics = []

        # Check for biotech/pharma keywords
        biotech_keywords = ['biotech', 'biologics', 'gene therapy', 'cell therapy', 'immunotherapy']
        pharma_keywords = ['pharmaceutical', 'pharma', 'drug', 'medicine', 'therapeutics']
        medical_device_keywords = ['medical device', 'diagnostic', 'imaging', 'equipment', 'implant']
        healthcare_service_keywords = ['hospital', 'provider', 'service', 'clinic', 'care management']

        # Check business description for keywords
        if any(keyword in business_description for keyword in biotech_keywords):
            biotech_score = 3
        elif any(keyword in industry for keyword in biotech_keywords):
            biotech_score = 2
        else:
            biotech_score = 0

        if any(keyword in business_description for keyword in pharma_keywords):
            pharma_score = 3
        elif any(keyword in industry for keyword in pharma_keywords):
            pharma_score = 2
        else:
            pharma_score = 0

        if any(keyword in business_description for keyword in medical_device_keywords):
            device_score = 3
        elif any(keyword in industry for keyword in medical_device_keywords):
            device_score = 2
        else:
            device_score = 0

        if any(keyword in business_description for keyword in healthcare_service_keywords):
            service_score = 3
        elif any(keyword in industry for keyword in healthcare_service_keywords):
            service_score = 2
        else:
            service_score = 0

        # Add points based on financial metrics
        if metrics.get('rd_to_revenue', 0) > 0.2:
            biotech_score += 2
            pharma_score += 1
            characteristics.append('Very high R&D intensity (>20% of revenue)')
        elif metrics.get('rd_to_revenue', 0) > 0.1:
            pharma_score += 2
            biotech_score += 1
            characteristics.append('High R&D intensity (>10% of revenue)')

        if metrics.get('gross_margin', 0) > 0.8:
            biotech_score += 1
            pharma_score += 1
            characteristics.append('Very high gross margins (>80%)')
        elif metrics.get('gross_margin', 0) > 0.6:
            pharma_score += 1
            device_score += 1
            characteristics.append('High gross margins (>60%)')

        if metrics.get('operating_margin', 0) < 0:
            if metrics.get('rd_to_revenue', 0) > 0.2:
                biotech_score += 2
                characteristics.append(
                    'Negative operating margins with high R&D spend (typical of early-stage biotech)')

        if metrics.get('net_income', 0) is not None:
            if metrics.get('net_income', 0) < 0:
                biotech_score += 1
                characteristics.append('Negative net income')
            else:
                pharma_score += 1
                service_score += 1
                characteristics.append('Positive net income')

        # Determine the highest scoring subsector
        scores = {
            'Biotech': biotech_score,
            'Pharmaceutical': pharma_score,
            'Medical Devices': device_score,
            'Healthcare Services': service_score
        }

        max_score = max(scores.values())
        max_subsectors = [sub for sub, score in scores.items() if score == max_score]

        if len(max_subsectors) == 1:
            subsector = max_subsectors[0]
            confidence = min(0.9, 0.5 + (max_score / 10))
        else:
            # If tie, prefer the more specific classification
            if 'Biotech' in max_subsectors:
                subsector = 'Biotech'
            elif 'Pharmaceutical' in max_subsectors:
                subsector = 'Pharmaceutical'
            elif 'Medical Devices' in max_subsectors:
                subsector = 'Medical Devices'
            else:
                subsector = 'Healthcare Services'
            confidence = min(0.7, 0.4 + (max_score / 10))

        # Further refinement for pharmaceutical/biotech companies
        if subsector in ['Biotech', 'Pharmaceutical']:
            if metrics.get('net_income', 0) is not None:
                if metrics.get('net_income', 0) < 0 and metrics.get('rd_to_revenue', 0) > 0.2:
                    if subsector == 'Pharmaceutical':
                        subsector = 'Early-Stage Pharma'
                    else:
                        subsector = 'Development-Stage Biotech'
                    characteristics.append('Pre-profit R&D-intensive company')
                elif metrics.get('net_income', 0) > 0:
                    if subsector == 'Pharmaceutical':
                        subsector = 'Commercial-Stage Pharma'
                    else:
                        subsector = 'Commercial-Stage Biotech'
                    characteristics.append('Profitable with commercialized products')

        # Add subsector-specific characteristics
        if subsector == 'Biotech' or subsector == 'Development-Stage Biotech':
            characteristics.append('Focus on biological drugs and therapies')
            characteristics.append('Typically higher risk/reward profile than traditional pharma')
        elif subsector == 'Pharmaceutical' or subsector == 'Commercial-Stage Pharma':
            characteristics.append('Develops and markets pharmaceutical drugs')
            characteristics.append('More diversified product portfolio than biotech')
        elif subsector == 'Medical Devices':
            characteristics.append('Produces medical equipment and supplies')
            characteristics.append('Different regulatory pathway than drugs (FDA 510(k)/PMA vs. NDA)')
        elif subsector == 'Healthcare Services':
            characteristics.append('Provides healthcare services directly to patients or other providers')
            characteristics.append('Less R&D intensive, more focus on operational efficiency')

        return {
            'subsector': subsector,
            'confidence': confidence,
            'characteristics': characteristics,
            'metrics': metrics
        }

    def _calculate_healthcare_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate healthcare sector-specific financial metrics

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with healthcare-specific metrics
        """
        metrics = {}

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        if income_stmt is None or income_stmt.empty:
            return metrics

        # Get the latest financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0] if balance_sheet is not None and not balance_sheet.empty else None
        cf = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Calculate R&D intensity (key metric for healthcare)
        if 'Research and Development' in income.index and 'Total Revenue' in income.index:
            rd_expense = income['Research and Development']
            revenue = income['Total Revenue']

            if revenue > 0:
                metrics['rd_to_revenue'] = rd_expense / revenue

            # Also calculate absolute R&D spend
            metrics['rd_expense'] = rd_expense

            # Calculate 3-year R&D CAGR if we have historical data
            if income_stmt.shape[1] >= 3 and 'Research and Development' in income_stmt.index:
                rd_3yr_ago = income_stmt.iloc[income_stmt.index.get_loc('Research and Development'),
                min(2, income_stmt.shape[1] - 1)]

                if rd_3yr_ago > 0:
                    # Calculate compound annual growth rate
                    years = min(3, income_stmt.shape[1])
                    metrics['rd_growth_cagr'] = (rd_expense / rd_3yr_ago) ** (1 / years) - 1

        # Calculate gross margin
        if 'Gross Profit' in income.index and 'Total Revenue' in income.index:
            gross_profit = income['Gross Profit']
            revenue = income['Total Revenue']

            if revenue > 0:
                metrics['gross_margin'] = gross_profit / revenue

        # Calculate operating margin
        if 'Operating Income' in income.index and 'Total Revenue' in income.index:
            operating_income = income['Operating Income']
            revenue = income['Total Revenue']

            if revenue > 0:
                metrics['operating_margin'] = operating_income / revenue

        # Calculate cash runway (critical for pre-profit healthcare companies)
        if balance is not None and cf is not None:
            if 'Cash and Cash Equivalents' in balance.index and 'Operating Cash Flow' in cf.index:
                cash = balance['Cash and Cash Equivalents']
                operating_cf = cf['Operating Cash Flow']

                # For companies with negative operating cash flow
                if operating_cf < 0:
                    cash_burn_rate = abs(operating_cf)
                    metrics['cash_runway_years'] = cash / cash_burn_rate

                    if metrics['cash_runway_years'] < 1:
                        metrics['cash_runway_assessment'] = "Critical - less than 1 year"
                    elif metrics['cash_runway_years'] < 2:
                        metrics['cash_runway_assessment'] = "Concerning - less than 2 years"
                    elif metrics['cash_runway_years'] < 3:
                        metrics['cash_runway_assessment'] = "Adequate - 2-3 years"
                    else:
                        metrics['cash_runway_assessment'] = "Strong - more than 3 years"

        # Calculate efficiency of R&D spending
        if 'rd_expense' in metrics and 'Total Revenue' in income.index:
            revenue = income['Total Revenue']
            metrics['revenue_per_rd_dollar'] = revenue / metrics['rd_expense'] if metrics['rd_expense'] > 0 else 0

        # Get revenue growth
        if income_stmt.shape[1] >= 2 and 'Total Revenue' in income_stmt.index:
            current_revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
            prev_revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 1]

            if prev_revenue > 0:
                metrics['revenue_growth_yoy'] = (current_revenue / prev_revenue) - 1

                # Calculate 3-year revenue CAGR if we have enough data
                if income_stmt.shape[1] >= 3:
                    revenue_3yr_ago = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'),
                    min(2, income_stmt.shape[1] - 1)]

                    if revenue_3yr_ago > 0:
                        years = min(3, income_stmt.shape[1])
                        metrics['revenue_cagr_3yr'] = (current_revenue / revenue_3yr_ago) ** (1 / years) - 1

        return metrics

    def _assess_healthcare_risks(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess specific risks related to healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with risk assessment
        """
        # Get subsector and metrics
        subsector_info = self._determine_healthcare_subsector(financial_data)
        subsector = subsector_info['subsector']
        metrics = self._calculate_healthcare_metrics(financial_data)

        risks = {
            'overall_risk': 'Moderate',
            'financial_risk': 'Moderate',
            'regulatory_risk': 'High',  # Healthcare is highly regulated by default
            'development_risk': 'Moderate',
            'competition_risk': 'Moderate',
            'specific_risks': [],
            'risk_score': 5.0  # Scale 1-10
        }

        # Financial risk assessment
        if 'cash_runway_years' in metrics:
            if metrics['cash_runway_years'] < 1:
                risks['financial_risk'] = 'Very High'
                risks['specific_risks'].append('Critical cash runway under 1 year')
                risks['risk_score'] += 2.0
            elif metrics['cash_runway_years'] < 2:
                risks['financial_risk'] = 'High'
                risks['specific_risks'].append('Short cash runway under 2 years')
                risks['risk_score'] += 1.0
            elif metrics['cash_runway_years'] > 4:
                risks['financial_risk'] = 'Low'

        # Development risk assessment based on subsector
        if subsector in ['Biotech', 'Development-Stage Biotech', 'Early-Stage Pharma']:
            risks['development_risk'] = 'Very High'
            risks['specific_risks'].append('High clinical development failure rates')
            risks['risk_score'] += 1.5
        elif subsector in ['Commercial-Stage Pharma', 'Commercial-Stage Biotech']:
            risks['development_risk'] = 'High'
            risks['specific_risks'].append('Pipeline development risks for future growth')
            risks['risk_score'] += 0.5
        elif subsector == 'Medical Devices':
            risks['development_risk'] = 'Moderate'
        elif subsector == 'Healthcare Services':
            risks['development_risk'] = 'Low'
            risks['risk_score'] -= 0.5

        # Regulatory risk assessment based on subsector
        if subsector in ['Biotech', 'Development-Stage Biotech', 'Early-Stage Pharma']:
            risks['regulatory_risk'] = 'Very High'
            risks['specific_risks'].append('FDA approval uncertainties')
            risks['risk_score'] += 1.0
        elif subsector in ['Commercial-Stage Pharma', 'Commercial-Stage Biotech']:
            risks['regulatory_risk'] = 'High'
        elif subsector == 'Medical Devices':
            risks['regulatory_risk'] = 'High'
            risks['specific_risks'].append('FDA 510(k) or PMA pathway risks')
        elif subsector == 'Healthcare Services':
            risks['regulatory_risk'] = 'Moderate'
            risks['specific_risks'].append('Healthcare policy and reimbursement risks')
            risks['risk_score'] -= 0.5

        # Patent/IP risk for research-based companies
        if 'rd_to_revenue' in metrics and metrics['rd_to_revenue'] > 0.1:
            risks['specific_risks'].append('Patent cliff and intellectual property risks')
            risks['risk_score'] += 0.5

        # Competition risk assessment
        if subsector == 'Healthcare Services':
            risks['competition_risk'] = 'Moderate'
            risks['specific_risks'].append('Local market competition')
        elif subsector == 'Medical Devices':
            risks['competition_risk'] = 'High'
            risks['specific_risks'].append('Rapid technological obsolescence risk')
            risks['risk_score'] += 0.5

        # Additional risks for loss-making companies
        income_stmt = financial_data.get('income_statement')
        if income_stmt is not None and not income_stmt.empty:
            income = income_stmt.iloc[:, 0]
            if 'Net Income' in income.index and income['Net Income'] < 0:
                # Check number of consecutive loss-making years
                loss_years = 0
                for i in range(min(income_stmt.shape[1], 5)):
                    if i < income_stmt.shape[1] and 'Net Income' in income_stmt.index:
                        if income_stmt.iloc[income_stmt.index.get_loc('Net Income'), i] < 0:
                            loss_years += 1

                if loss_years >= 3:
                    risks['specific_risks'].append(f'Persistent losses for {loss_years} consecutive years')
                    risks['financial_risk'] = 'High'
                    risks['risk_score'] += 1.0

        # Cap the risk score
        risks['risk_score'] = min(10.0, max(1.0, risks['risk_score']))

        # Determine overall risk based on risk score
        if risks['risk_score'] >= 8.0:
            risks['overall_risk'] = 'Very High'
        elif risks['risk_score'] >= 6.0:
            risks['overall_risk'] = 'High'
        elif risks['risk_score'] >= 4.0:
            risks['overall_risk'] = 'Moderate'
        elif risks['risk_score'] >= 2.0:
            risks['overall_risk'] = 'Low'
        else:
            risks['overall_risk'] = 'Very Low'

        return risks

    def _analyze_growth_potential(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze growth potential specific to healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with growth analysis
        """
        # Get subsector information
        subsector_info = self._determine_healthcare_subsector(financial_data)
        subsector = subsector_info['subsector']

        # Get financial metrics
        metrics = self._calculate_healthcare_metrics(financial_data)

        growth_analysis = {
            'growth_potential': 'Moderate',
            'growth_drivers': [],
            'growth_barriers': [],
            'revenue_growth': metrics.get('revenue_growth_yoy'),
            'forward_growth_estimate': None,
            'growth_sustainability_score': 5.0  # Scale 1-10
        }

        # Extract income statement for historical analysis
        income_stmt = financial_data.get('income_statement')

        if income_stmt is not None and not income_stmt.empty:
            # Calculate historical growth metrics if we have enough data
            if income_stmt.shape[1] >= 3 and 'Total Revenue' in income_stmt.index:
                # Calculate revenue growth stability
                growth_rates = []
                for i in range(min(income_stmt.shape[1], 5) - 1):
                    if i + 1 < income_stmt.shape[1]:
                        current = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), i]
                        previous = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), i + 1]

                        if previous > 0:
                            growth_rate = (current / previous) - 1
                            growth_rates.append(growth_rate)

                if growth_rates:
                    growth_analysis['historical_growth_rates'] = growth_rates
                    growth_analysis['mean_growth_rate'] = sum(growth_rates) / len(growth_rates)
                    growth_analysis['growth_volatility'] = np.std(growth_rates) if len(growth_rates) > 1 else 0

                    # Assess growth stability
                    if growth_analysis['growth_volatility'] < 0.05:
                        growth_analysis['growth_stability'] = 'Very Stable'
                        growth_analysis['growth_sustainability_score'] += 1.0
                    elif growth_analysis['growth_volatility'] < 0.1:
                        growth_analysis['growth_stability'] = 'Stable'
                        growth_analysis['growth_sustainability_score'] += 0.5
                    elif growth_analysis['growth_volatility'] > 0.25:
                        growth_analysis['growth_stability'] = 'Highly Volatile'
                        growth_analysis['growth_sustainability_score'] -= 1.0
                        growth_analysis['growth_barriers'].append('Inconsistent historical growth')
                    else:
                        growth_analysis['growth_stability'] = 'Moderately Volatile'

        # Subsector-specific growth analysis
        if subsector in ['Biotech', 'Development-Stage Biotech']:
            # Early-stage companies with high growth potential but high risk
            growth_analysis['growth_potential'] = 'Very High'
            growth_analysis['growth_drivers'].append('Novel biological therapies with large market potential')
            growth_analysis['growth_drivers'].append('High unmet medical need in target indications')
            growth_analysis['growth_barriers'].append('Clinical development and regulatory hurdles')
            growth_analysis['growth_barriers'].append('Funding requirements for R&D progression')

            # Adjust growth score based on R&D investment
            if metrics.get('rd_to_revenue', 0) > 0.2:
                growth_analysis['growth_sustainability_score'] += 1.0

            # Estimate forward growth
            growth_analysis['forward_growth_estimate'] = 0.30  # Biotech can have explosive growth upon approval

        elif subsector in ['Pharmaceutical', 'Commercial-Stage Pharma', 'Commercial-Stage Biotech']:
            growth_analysis['growth_potential'] = 'Moderate to High'
            growth_analysis['growth_drivers'].append('Existing commercial products with established revenue')
            growth_analysis['growth_drivers'].append('Pipeline products for future growth')
            growth_analysis['growth_barriers'].append('Patent expirations and generic competition')

            # Check for R&D investment as growth driver
            if metrics.get('rd_to_revenue', 0) > 0.15:
                growth_analysis['growth_drivers'].append('Strong R&D investment supporting future pipeline')
                growth_analysis['growth_sustainability_score'] += 1.0
            elif metrics.get('rd_to_revenue', 0) < 0.1:
                growth_analysis['growth_barriers'].append('Below-average R&D investment may limit future growth')
                growth_analysis['growth_sustainability_score'] -= 1.0

            # Estimate forward growth based on historical performance
            if 'mean_growth_rate' in growth_analysis:
                growth_analysis['forward_growth_estimate'] = growth_analysis[
                                                                 'mean_growth_rate'] * 0.9  # Slight discount
            else:
                growth_analysis['forward_growth_estimate'] = 0.08  # Industry average

        elif subsector == 'Medical Devices':
            growth_analysis['growth_potential'] = 'Moderate'
            growth_analysis['growth_drivers'].append('Innovation in medical technology')
            growth_analysis['growth_drivers'].append('Aging population increasing device demand')
            growth_analysis['growth_barriers'].append('Reimbursement challenges for new devices')
            growth_analysis['growth_barriers'].append('Competitive landscape with rapid technological change')

            # Estimate forward growth
            if 'mean_growth_rate' in growth_analysis:
                growth_analysis['forward_growth_estimate'] = growth_analysis['mean_growth_rate']
            else:
                growth_analysis['forward_growth_estimate'] = 0.07  # Industry average

        elif subsector == 'Healthcare Services':
            growth_analysis['growth_potential'] = 'Moderate'
            growth_analysis['growth_drivers'].append('Increasing healthcare demand from aging population')
            growth_analysis['growth_drivers'].append('Potential for operational efficiency improvements')
            growth_analysis['growth_barriers'].append('Reimbursement pressure from insurers and government')
            growth_analysis['growth_barriers'].append('Regulatory changes affecting business model')

            # Estimate forward growth
            if 'mean_growth_rate' in growth_analysis:
                growth_analysis['forward_growth_estimate'] = growth_analysis['mean_growth_rate']
            else:
                growth_analysis['forward_growth_estimate'] = 0.06  # Industry average

        # Adjust growth potential based on historical performance
        if 'mean_growth_rate' in growth_analysis:
            if growth_analysis['mean_growth_rate'] > 0.25:
                growth_analysis['growth_potential'] = 'Very High'
                growth_analysis['growth_sustainability_score'] += 1.0
            elif growth_analysis['mean_growth_rate'] > 0.15:
                growth_analysis['growth_potential'] = 'High'
                growth_analysis['growth_sustainability_score'] += 0.5
            elif growth_analysis['mean_growth_rate'] < 0.05:
                if growth_analysis['growth_potential'] != 'Very High':  # Don't downgrade early biotech
                    growth_analysis['growth_potential'] = 'Low'
                    growth_analysis['growth_sustainability_score'] -= 0.5

        # Cap the sustainability score
        growth_analysis['growth_sustainability_score'] = min(10.0,
                                                             max(1.0, growth_analysis['growth_sustainability_score']))

        return growth_analysis

    def _assess_innovation_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess innovation capabilities specific to healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with innovation assessment
        """
        # Get subsector and metrics
        subsector_info = self._determine_healthcare_subsector(financial_data)
        subsector = subsector_info['subsector']
        metrics = self._calculate_healthcare_metrics(financial_data)

        innovation_assessment = {
            'innovation_strength': 'Moderate',
            'key_innovation_metrics': {},
            'innovation_drivers': [],
            'innovation_barriers': [],
            'innovation_score': 5.0  # Scale 1-10
        }

        # R&D intensity is the primary innovation metric in healthcare
        if 'rd_to_revenue' in metrics:
            rd_intensity = metrics['rd_to_revenue']
            innovation_assessment['key_innovation_metrics']['rd_intensity'] = rd_intensity

            if rd_intensity > 0.25:
                innovation_assessment['innovation_strength'] = 'Very Strong'
                innovation_assessment['innovation_score'] += 2.0
                innovation_assessment['innovation_drivers'].append(
                    'Exceptionally high R&D investment (>25% of revenue)')
            elif rd_intensity > 0.15:
                innovation_assessment['innovation_strength'] = 'Strong'
                innovation_assessment['innovation_score'] += 1.0
                innovation_assessment['innovation_drivers'].append('Above-average R&D investment (>15% of revenue)')
            elif rd_intensity < 0.05:
                innovation_assessment['innovation_strength'] = 'Weak'
                innovation_assessment['innovation_score'] -= 1.0
                innovation_assessment['innovation_barriers'].append('Below-average R&D investment (<5% of revenue)')

        # R&D growth rate
        if 'rd_growth_cagr' in metrics:
            rd_growth = metrics['rd_growth_cagr']
            innovation_assessment['key_innovation_metrics']['rd_growth'] = rd_growth

            if rd_growth > 0.2:
                innovation_assessment['innovation_score'] += 1.0
                innovation_assessment['innovation_drivers'].append('Rapidly increasing R&D investment (>20% CAGR)')
            elif rd_growth > 0.1:
                innovation_assessment['innovation_score'] += 0.5
                innovation_assessment['innovation_drivers'].append('Growing R&D investment (>10% CAGR)')
            elif rd_growth < 0:
                innovation_assessment['innovation_score'] -= 1.0
                innovation_assessment['innovation_barriers'].append('Decreasing R&D investment')

        # R&D efficiency
        if 'revenue_per_rd_dollar' in metrics:
            rd_efficiency = metrics['revenue_per_rd_dollar']
            innovation_assessment['key_innovation_metrics']['rd_efficiency'] = rd_efficiency

            # Different expectations based on subsector
            if subsector in ['Commercial-Stage Pharma', 'Commercial-Stage Biotech']:
                if rd_efficiency > 5:
                    innovation_assessment['innovation_score'] += 1.0
                    innovation_assessment['innovation_drivers'].append('High R&D efficiency (>$5 revenue per $1 R&D)')
                elif rd_efficiency < 2:
                    innovation_assessment['innovation_score'] -= 0.5
                    innovation_assessment['innovation_barriers'].append('Low R&D efficiency (<$2 revenue per $1 R&D)')
            elif subsector == 'Medical Devices' and rd_efficiency < 3:
                innovation_assessment['innovation_score'] -= 0.5
                innovation_assessment['innovation_barriers'].append('Below-average R&D efficiency for device company')

        # Subsector-specific assessments
        if subsector in ['Biotech', 'Development-Stage Biotech']:
            innovation_assessment['innovation_drivers'].append('Focus on novel biological therapies')

            # Pre-revenue biotech is expected to have no R&D efficiency yet
            if 'revenue_per_rd_dollar' in innovation_assessment['key_innovation_metrics']:
                del innovation_assessment['key_innovation_metrics']['rd_efficiency']

            # High rd_intensity is expected for biotech
            if metrics.get('rd_to_revenue', 0) > 0.5:
                innovation_assessment['innovation_drivers'].append('R&D-focused business model (>50% of revenue)')
                innovation_assessment['innovation_score'] += 0.5

        elif subsector in ['Pharmaceutical', 'Commercial-Stage Pharma']:
            if metrics.get('rd_to_revenue', 0) > 0.15:
                innovation_assessment['innovation_drivers'].append('Above-average pharmaceutical R&D investment')
                innovation_assessment['innovation_score'] += 0.5

        elif subsector == 'Medical Devices':
            if metrics.get('rd_to_revenue', 0) > 0.1:
                innovation_assessment['innovation_drivers'].append('Strong R&D for a medical device company')
                innovation_assessment['innovation_score'] += 0.5

        elif subsector == 'Healthcare Services':
            # Services companies typically have minimal R&D
            innovation_assessment['innovation_barriers'].append('Service-focused business with limited product R&D')
            innovation_assessment['innovation_score'] -= 1.0

        # Cap the innovation score
        innovation_assessment['innovation_score'] = min(10.0, max(1.0, innovation_assessment['innovation_score']))

        # Final innovation strength assessment based on score
        if innovation_assessment['innovation_score'] >= 8.0:
            innovation_assessment['innovation_strength'] = 'Very Strong'
        elif innovation_assessment['innovation_score'] >= 6.0:
            innovation_assessment['innovation_strength'] = 'Strong'
        elif innovation_assessment['innovation_score'] >= 4.0:
            innovation_assessment['innovation_strength'] = 'Moderate'
        elif innovation_assessment['innovation_score'] >= 2.0:
            innovation_assessment['innovation_strength'] = 'Weak'
        else:
            innovation_assessment['innovation_strength'] = 'Very Weak'

        return innovation_assessment

    def _assess_regulatory_exposure(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess regulatory exposure specific to healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with regulatory exposure assessment
        """
        # Get subsector
        subsector_info = self._determine_healthcare_subsector(financial_data)
        subsector = subsector_info['subsector']

        regulatory_assessment = {
            'regulatory_exposure': 'High',  # Healthcare default
            'regulatory_risk_factors': [],
            'regulatory_opportunities': [],
            'regulatory_risk_score': 7.0,  # Scale 1-10, healthcare starts high
            'key_regulatory_considerations': []
        }

        # Subsector-specific regulatory assessment
        if subsector in ['Biotech', 'Development-Stage Biotech', 'Early-Stage Pharma']:
            regulatory_assessment['regulatory_exposure'] = 'Very High'
            regulatory_assessment['regulatory_risk_score'] += 1.0
            regulatory_assessment['regulatory_risk_factors'].append('FDA approval process for novel therapies')
            regulatory_assessment['regulatory_risk_factors'].append('Clinical trial design and endpoint requirements')
            regulatory_assessment['key_regulatory_considerations'].append('Phase I/II/III clinical trial progression')
            regulatory_assessment['key_regulatory_considerations'].append('FDA meeting outcomes and guidance')

            # Opportunity in breakthrough designations
            regulatory_assessment['regulatory_opportunities'].append('Potential for breakthrough therapy designation')
            regulatory_assessment['regulatory_opportunities'].append(
                'Orphan drug designation potential for rare diseases')

        elif subsector in ['Pharmaceutical', 'Commercial-Stage Pharma', 'Commercial-Stage Biotech']:
            regulatory_assessment['regulatory_risk_factors'].append('Post-marketing surveillance requirements')
            regulatory_assessment['regulatory_risk_factors'].append('Patent challenges and generic competition')
            regulatory_assessment['regulatory_risk_factors'].append('Pricing pressure from government programs')
            regulatory_assessment['key_regulatory_considerations'].append(
                'REMS (Risk Evaluation and Mitigation Strategies) requirements')
            regulatory_assessment['key_regulatory_considerations'].append('Manufacturing facility compliance (GMP)')

            # Commercial stage typically has some approved products reducing overall risk
            regulatory_assessment['regulatory_risk_score'] -= 0.5

            # Opportunity in label expansions
            regulatory_assessment['regulatory_opportunities'].append('Label expansion opportunities for existing drugs')
            regulatory_assessment['regulatory_opportunities'].append('New indication approvals expanding market')

        elif subsector == 'Medical Devices':
            regulatory_assessment['regulatory_risk_factors'].append('510(k) clearance or PMA approval pathways')
            regulatory_assessment['regulatory_risk_factors'].append('Post-market surveillance requirements')
            regulatory_assessment['regulatory_risk_factors'].append('Reimbursement code assignments')
            regulatory_assessment['key_regulatory_considerations'].append('Device classification (Class I, II, or III)')
            regulatory_assessment['key_regulatory_considerations'].append('Quality System Regulation compliance')

            # 510(k) pathway is slightly less rigorous than drug approval
            regulatory_assessment['regulatory_risk_score'] -= 1.0

            # Opportunity in breakthrough device designation
            regulatory_assessment['regulatory_opportunities'].append('Breakthrough device designation potential')
            regulatory_assessment['regulatory_opportunities'].append('De Novo classification process for novel devices')

        elif subsector == 'Healthcare Services':
            regulatory_assessment['regulatory_exposure'] = 'Moderate'
            regulatory_assessment['regulatory_risk_score'] -= 2.0
            regulatory_assessment['regulatory_risk_factors'].append('Medicare/Medicaid reimbursement changes')
            regulatory_assessment['regulatory_risk_factors'].append('HIPAA compliance requirements')
            regulatory_assessment['regulatory_risk_factors'].append('Affordable Care Act provisions')
            regulatory_assessment['key_regulatory_considerations'].append('State-specific healthcare regulations')
            regulatory_assessment['key_regulatory_considerations'].append('Data privacy and security requirements')

            # Opportunity in value-based care models
            regulatory_assessment['regulatory_opportunities'].append('Value-based care model incentives')
            regulatory_assessment['regulatory_opportunities'].append('Telehealth expansion and reimbursement')

        # Adjust regulatory exposure based on risk score
        if regulatory_assessment['regulatory_risk_score'] >= 8.0:
            regulatory_assessment['regulatory_exposure'] = 'Very High'
        elif regulatory_assessment['regulatory_risk_score'] >= 6.0:
            regulatory_assessment['regulatory_exposure'] = 'High'
        elif regulatory_assessment['regulatory_risk_score'] >= 4.0:
            regulatory_assessment['regulatory_exposure'] = 'Moderate'
        else:
            regulatory_assessment['regulatory_exposure'] = 'Low'

        return regulatory_assessment

    def _pipeline_based_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline-based DCF valuation method, particularly suited for biotech and early-stage pharma

        This method values companies based on the probability-adjusted net present value
        of their drug/therapy pipeline, particularly useful for companies without current revenue.

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with valuation results
        """
        # Get company info
        company_info = financial_data.get('company_info', {})
        ticker = company_info.get('symbol', 'Unknown')

        # Get market data
        market_data = financial_data.get('market_data', {})
        current_price = market_data.get('share_price')
        shares_outstanding = market_data.get('shares_outstanding')

        # Get subsector for more tailored analysis
        subsector = self._determine_healthcare_subsector(financial_data)['subsector']

        # Since we don't have direct pipeline data in financial statements,
        # we'll create a simplified model based on R&D spending and other metrics

        # Extract balance sheet and cash flow to estimate cash available
        balance_sheet = financial_data.get('balance_sheet')
        income_stmt = financial_data.get('income_statement')
        cash_flow = financial_data.get('cash_flow')

        # Initialize valuation results
        valuation_result = {
            'company': ticker,
            'method': 'pipeline_based_dcf',
            'subsector': subsector,
            'enterprise_value': None,
            'equity_value': None,
            'value_per_share': None,
            'current_price': current_price,
            'upside_potential': None,
            'key_assumptions': {}
        }

        # Cannot perform valuation without basic financial data
        if income_stmt is None or income_stmt.empty:
            valuation_result['error'] = "Insufficient financial data for pipeline-based valuation"
            return valuation_result

        # Extract necessary data
        if balance_sheet is not None and not balance_sheet.empty:
            balance = balance_sheet.iloc[:, 0]

            # Get cash position
            if 'Cash and Cash Equivalents' in balance.index:
                cash = balance['Cash and Cash Equivalents']
                valuation_result['key_assumptions']['cash_position'] = float(cash)

            # Get total debt
            total_debt = 0
            for debt_field in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_field in balance.index:
                    total_debt += balance[debt_field]

            valuation_result['key_assumptions']['total_debt'] = float(total_debt)

        # Get R&D spending (key for pipeline valuation)
        income = income_stmt.iloc[:, 0]
        rd_expense = 0
        if 'Research and Development' in income.index:
            rd_expense = income['Research and Development']
            valuation_result['key_assumptions']['annual_rd_expense'] = float(rd_expense)

        # Get revenue to determine if pre-commercial or commercial stage
        revenue = 0
        if 'Total Revenue' in income.index:
            revenue = income['Total Revenue']
            valuation_result['key_assumptions']['annual_revenue'] = float(revenue)

        # Determine if pre-commercial or commercial
        is_pre_commercial = revenue < rd_expense * 0.5  # Significantly less revenue than R&D spend

        # Set assumptions based on subsector and commercial status
        if subsector in ['Biotech', 'Development-Stage Biotech', 'Early-Stage Pharma'] and is_pre_commercial:
            # Pre-commercial biotech assumptions
            discount_rate = 0.15  # Higher discount rate for pre-commercial
            value_multiplier = 4.5  # Approximate multiple of R&D spend for valuation
            success_probability = 0.15  # Probability of successful commercialization
            time_to_market = 5  # Years until potential commercialization
            peak_sales_to_rd_ratio = 12  # Peak annual sales as multiple of current R&D

            # Pipeline value calculation - simplified approach for development-stage biotech
            peak_sales_potential = rd_expense * peak_sales_to_rd_ratio

            # Adjust based on R&D intensity
            if rd_expense > 100000000:  # >$100M annual R&D
                value_multiplier = 5.0
                success_probability = 0.18
            elif rd_expense < 20000000:  # <$20M annual R&D
                value_multiplier = 3.5
                success_probability = 0.12

            # Record assumptions
            valuation_result['key_assumptions']['discount_rate'] = discount_rate
            valuation_result['key_assumptions']['value_multiplier'] = value_multiplier
            valuation_result['key_assumptions']['success_probability'] = success_probability
            valuation_result['key_assumptions']['time_to_market'] = time_to_market
            valuation_result['key_assumptions']['peak_sales_potential'] = float(peak_sales_potential)

            # Calculate risk-adjusted NPV
            risk_adjusted_npv = 0

            # Simple projection for peak sales achieved over time_to_market + 5 years with 8-year patent life
            projected_sales = []
            patent_life = 8  # years of exclusivity after approval
            ramp_up_years = 3  # years to reach peak sales after launch

            for year in range(1, time_to_market + patent_life + 1):
                if year <= time_to_market:
                    # Pre-launch: no revenue
                    annual_sales = 0
                elif year <= time_to_market + ramp_up_years:
                    # Ramp-up phase to peak sales
                    ramp_factor = (year - time_to_market) / ramp_up_years
                    annual_sales = peak_sales_potential * ramp_factor
                elif year <= time_to_market + patent_life:
                    # Peak sales during exclusivity
                    annual_sales = peak_sales_potential
                else:
                    # Decline after patent expiry
                    annual_sales = peak_sales_potential * 0.3  # 70% decline

                # Discount to present value
                present_value = annual_sales / ((1 + discount_rate) ** year)

                # Apply success probability
                risk_adjusted_present_value = present_value * success_probability
                risk_adjusted_npv += risk_adjusted_present_value

                projected_sales.append({
                    'year': year,
                    'phase': 'Pre-launch' if year <= time_to_market else 'Commercial',
                    'annual_sales': float(annual_sales),
                    'present_value': float(present_value),
                    'risk_adjusted_pv': float(risk_adjusted_present_value)
                })

            # Approximate costs and margins
            cost_of_revenue_pct = 0.25  # 25% COGS for biotech
            opex_pct = 0.40  # 40% operating expenses excluding R&D
            operating_margin = 1 - cost_of_revenue_pct - opex_pct
            tax_rate = 0.21  # US corporate tax rate

            # Adjust NPV for operating margin and taxes
            operating_profit_npv = risk_adjusted_npv * operating_margin
            after_tax_npv = operating_profit_npv * (1 - tax_rate)

            # Add these assumptions
            valuation_result['key_assumptions']['operating_margin'] = operating_margin
            valuation_result['key_assumptions']['tax_rate'] = tax_rate

            # Alternatively, use R&D multiple method as a crosscheck
            rd_multiple_valuation = rd_expense * value_multiplier

            # Calculate enterprise value (weighted average of methods)
            enterprise_value = (after_tax_npv * 0.7) + (rd_multiple_valuation * 0.3)

            # Adjust for cash and debt
            equity_value = enterprise_value + valuation_result['key_assumptions'].get('cash_position', 0) - \
                           valuation_result['key_assumptions'].get('total_debt', 0)

            # Calculate per share value
            if shares_outstanding:
                value_per_share = equity_value / shares_outstanding

                # Calculate upside/downside vs current price
                if current_price:
                    upside_potential = (value_per_share / current_price) - 1
                    valuation_result['upside_potential'] = float(upside_potential)

                valuation_result['value_per_share'] = float(value_per_share)

            valuation_result['enterprise_value'] = float(enterprise_value)
            valuation_result['equity_value'] = float(equity_value)
            valuation_result['projected_sales'] = projected_sales

            # Add probabilistic range
            valuation_result['bull_case'] = float(value_per_share * 1.5)  # 50% higher success probability
            valuation_result['bear_case'] = float(value_per_share * 0.5)  # 50% lower success probability

        else:
            # For commercial-stage companies, use standard multi-stage DCF
            return self._multi_stage_dcf_valuation(financial_data)

        return valuation_result

    def _multi_stage_dcf_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-stage DCF valuation method suitable for established healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with valuation results
        """
        # Get company info
        company_info = financial_data.get('company_info', {})
        ticker = company_info.get('symbol', 'Unknown')

        # Get subsector for more tailored analysis
        subsector = self._determine_healthcare_subsector(financial_data)['subsector']

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Get market data
        market_data = financial_data.get('market_data', {})
        current_price = market_data.get('share_price')
        shares_outstanding = market_data.get('shares_outstanding')

        # Initialize valuation results
        valuation_result = {
            'company': ticker,
            'method': 'multi_stage_dcf',
            'subsector': subsector,
            'enterprise_value': None,
            'equity_value': None,
            'value_per_share': None,
            'current_price': current_price,
            'upside_potential': None,
            'key_assumptions': {},
            'forecast': {}
        }

        # Cannot perform valuation without basic financial data
        if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
            valuation_result['error'] = "Insufficient financial data for DCF valuation"
            return valuation_result

        # Get historical free cash flow data
        historical_fcf = []
        if cash_flow is not None and not cash_flow.empty:
            for i in range(min(5, cash_flow.shape[1])):
                if i < cash_flow.shape[1]:
                    cf_data = cash_flow.iloc[:, i]

                    # Try to get FCF directly
                    if 'Free Cash Flow' in cf_data.index:
                        fcf = cf_data['Free Cash Flow']
                    # Otherwise calculate from components
                    elif 'Operating Cash Flow' in cf_data.index and 'Capital Expenditure' in cf_data.index:
                        fcf = cf_data['Operating Cash Flow'] + cf_data['Capital Expenditure']  # CapEx is negative
                    else:
                        continue

                    historical_fcf.append(float(fcf))

        # If we don't have FCF data, try to estimate from income statement
        if not historical_fcf and income_stmt is not None and not income_stmt.empty:
            for i in range(min(5, income_stmt.shape[1])):
                if i < income_stmt.shape[1]:
                    inc_data = income_stmt.iloc[:, i]

                    if 'Net Income' in inc_data.index:
                        net_income = inc_data['Net Income']

                        # Rough FCF approximation (net income × 0.9 assuming 10% of income goes to capex/wc)
                        estimated_fcf = net_income * 0.9
                        historical_fcf.append(float(estimated_fcf))

        if not historical_fcf:
            valuation_result['error'] = "Unable to determine cash flows for DCF valuation"
            return valuation_result

        # Get the most recent FCF as starting point
        starting_fcf = historical_fcf[0]

        # Calculate historical growth rate
        if len(historical_fcf) > 1:
            growth_rates = []
            for i in range(len(historical_fcf) - 1):
                if historical_fcf[i + 1] > 0:  # Avoid division by zero or negative
                    annual_growth = (historical_fcf[i] / historical_fcf[i + 1]) - 1
                    growth_rates.append(annual_growth)

            if growth_rates:
                # Use weighted average favoring recent growth
                weights = list(range(1, len(growth_rates) + 1))
                historical_growth = sum(rate * weight for rate, weight in zip(growth_rates, weights)) / sum(weights)
            else:
                historical_growth = 0.08  # Default healthcare sector growth
        else:
            historical_growth = 0.08  # Default growth rate

        # Set DCF parameters based on subsector
        if subsector in ['Biotech', 'Commercial-Stage Biotech']:
            # High growth potential but higher risk
            high_growth_years = 5
            high_growth_rate = max(historical_growth, 0.12)  # At least 12% for biotech
            transition_years = 5
            terminal_growth = 0.03
            discount_rate = 0.11

        elif subsector in ['Pharmaceutical', 'Commercial-Stage Pharma']:
            # Moderate growth with patent protection
            high_growth_years = 5
            high_growth_rate = max(historical_growth, 0.08)  # At least 8% for pharma
            transition_years = 5
            terminal_growth = 0.025
            discount_rate = 0.09

        elif subsector == 'Medical Devices':
            # Moderate growth with technological innovation
            high_growth_years = 5
            high_growth_rate = max(historical_growth, 0.07)  # At least 7% for medical devices
            transition_years = 5
            terminal_growth = 0.025
            discount_rate = 0.095

        elif subsector == 'Healthcare Services':
            # Steady growth with demographic tailwinds
            high_growth_years = 4
            high_growth_rate = max(historical_growth, 0.06)  # At least 6% for services
            transition_years = 4
            terminal_growth = 0.02
            discount_rate = 0.08

        else:
            # Default healthcare parameters
            high_growth_years = 5
            high_growth_rate = max(historical_growth, 0.07)
            transition_years = 5
            terminal_growth = 0.025
            discount_rate = 0.09

        # Record the assumptions
        valuation_result['key_assumptions']['high_growth_years'] = high_growth_years
        valuation_result['key_assumptions']['high_growth_rate'] = float(high_growth_rate)
        valuation_result['key_assumptions']['transition_years'] = transition_years
        valuation_result['key_assumptions']['terminal_growth_rate'] = float(terminal_growth)
        valuation_result['key_assumptions']['discount_rate'] = float(discount_rate)
        valuation_result['key_assumptions']['starting_fcf'] = float(starting_fcf)

        # Forecast cash flows for high growth phase
        high_growth_fcf = []
        cumulative_fcf = starting_fcf

        for year in range(1, high_growth_years + 1):
            fcf = cumulative_fcf * (1 + high_growth_rate)
            high_growth_fcf.append(float(fcf))
            cumulative_fcf = fcf

        # Forecast cash flows for transition phase
        transition_fcf = []

        for year in range(1, transition_years + 1):
            # Linear decline in growth rate from high_growth_rate to terminal_growth
            growth_rate = high_growth_rate - ((high_growth_rate - terminal_growth) * year / transition_years)
            fcf = cumulative_fcf * (1 + growth_rate)
            transition_fcf.append(float(fcf))
            cumulative_fcf = fcf

        # Calculate terminal value
        terminal_fcf = transition_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)

        # Discount all cash flows to present value
        present_value_high_growth = sum(fcf / (1 + discount_rate) ** year
                                        for year, fcf in enumerate(high_growth_fcf, 1))

        present_value_transition = sum(fcf / (1 + discount_rate) ** (year + high_growth_years)
                                       for year, fcf in enumerate(transition_fcf, 1))

        present_value_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

        # Calculate enterprise value
        enterprise_value = present_value_high_growth + present_value_transition + present_value_terminal

        # Calculate net debt
        net_debt = 0
        if balance_sheet is not None and not balance_sheet.empty:
            balance = balance_sheet.iloc[:, 0]

            # Calculate total debt
            total_debt = 0
            for debt_field in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_field in balance.index:
                    total_debt += balance[debt_field]

            # Get cash and equivalents
            cash = 0
            if 'Cash and Cash Equivalents' in balance.index:
                cash = balance['Cash and Cash Equivalents']

            net_debt = total_debt - cash

        # Calculate equity value
        equity_value = enterprise_value - net_debt

        # Calculate per share value
        if shares_outstanding:
            value_per_share = equity_value / shares_outstanding

            # Calculate upside/downside vs current price
            if current_price:
                upside_potential = (value_per_share / current_price) - 1
                valuation_result['upside_potential'] = float(upside_potential)

            valuation_result['value_per_share'] = float(value_per_share)

        valuation_result['enterprise_value'] = float(enterprise_value)
        valuation_result['equity_value'] = float(equity_value)
        valuation_result['net_debt'] = float(net_debt)

        # Save the forecast
        valuation_result['forecast'] = {
            'high_growth_fcf': high_growth_fcf,
            'transition_fcf': transition_fcf,
            'terminal_value': float(terminal_value),
            'present_value_high_growth': float(present_value_high_growth),
            'present_value_transition': float(present_value_transition),
            'present_value_terminal': float(present_value_terminal)
        }

        # Add sensitivity analysis
        valuation_result['sensitivity'] = {
            'discount_rate_plus_1pct': float(self._recalculate_dcf_value(
                high_growth_fcf, transition_fcf, terminal_fcf, discount_rate + 0.01,
                terminal_growth, high_growth_years, transition_years, net_debt, shares_outstanding
            )),
            'discount_rate_minus_1pct': float(self._recalculate_dcf_value(
                high_growth_fcf, transition_fcf, terminal_fcf, discount_rate - 0.01,
                terminal_growth, high_growth_years, transition_years, net_debt, shares_outstanding
            )),
            'growth_rate_plus_1pct': float(self._recalculate_dcf_value(
                high_growth_fcf, transition_fcf, terminal_fcf, discount_rate,
                terminal_growth + 0.01, high_growth_years, transition_years, net_debt, shares_outstanding
            )),
            'growth_rate_minus_1pct': float(self._recalculate_dcf_value(
                high_growth_fcf, transition_fcf, terminal_fcf, discount_rate,
                terminal_growth - 0.01, high_growth_years, transition_years, net_debt, shares_outstanding
            ))
        }

        return valuation_result

    def _recalculate_dcf_value(self, high_growth_fcf, transition_fcf, terminal_fcf,
                               discount_rate, terminal_growth, high_growth_years,
                               transition_years, net_debt, shares_outstanding):
        """Helper function to recalculate DCF value with different parameters"""
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)

        pv_high_growth = sum(fcf / (1 + discount_rate) ** year
                             for year, fcf in enumerate(high_growth_fcf, 1))

        pv_transition = sum(fcf / (1 + discount_rate) ** (year + high_growth_years)
                            for year, fcf in enumerate(transition_fcf, 1))

        pv_terminal = terminal_value / (1 + discount_rate) ** (high_growth_years + transition_years)

        enterprise_value = pv_high_growth + pv_transition + pv_terminal
        equity_value = enterprise_value - net_debt

        if shares_outstanding:
            return equity_value / shares_outstanding
        else:
            return None

    def _standard_dcf_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard DCF valuation method for established healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with valuation results
        """
        # This is a simpler DCF model compared to multi-stage, useful for stable companies
        # For healthcare, the multi-stage model is generally more appropriate, so we'll redirect
        return self._multi_stage_dcf_valuation(financial_data)

    def _ebitda_multiple_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        EBITDA multiple valuation method, common for healthcare companies

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with valuation results
        """
        # Get company info
        company_info = financial_data.get('company_info', {})
        ticker = company_info.get('symbol', 'Unknown')

        # Get subsector for more tailored analysis
        subsector = self._determine_healthcare_subsector(financial_data)['subsector']

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Get market data
        market_data = financial_data.get('market_data', {})
        current_price = market_data.get('share_price')
        shares_outstanding = market_data.get('shares_outstanding')

        # Initialize valuation results
        valuation_result = {
            'company': ticker,
            'method': 'ebitda_multiple',
            'subsector': subsector,
            'enterprise_value': None,
            'equity_value': None,
            'value_per_share': None,
            'current_price': current_price,
            'upside_potential': None,
            'key_assumptions': {}
        }

        # Cannot perform valuation without basic financial data
        if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
            valuation_result['error'] = "Insufficient financial data for EBITDA multiple valuation"
            return valuation_result

        # Get EBITDA from income statement
        income = income_stmt.iloc[:, 0]
        ebitda = None

        if 'EBITDA' in income.index:
            ebitda = income['EBITDA']
        elif 'Operating Income' in income.index and 'Depreciation & Amortization' in income.index:
            # Calculate EBITDA from components
            ebitda = income['Operating Income'] + income['Depreciation & Amortization']

        if ebitda is None or ebitda <= 0:
            valuation_result['error'] = "Unable to determine positive EBITDA for multiple valuation"
            return valuation_result

        # Set appropriate EBITDA multiple based on subsector
        if subsector in ['Biotech', 'Commercial-Stage Biotech', 'Development-Stage Biotech']:
            # Higher multiples for high-growth biotech
            ebitda_multiple = 15.0
        elif subsector in ['Pharmaceutical', 'Commercial-Stage Pharma', 'Early-Stage Pharma']:
            # Strong, stable pharma companies
            ebitda_multiple = 13.0
        elif subsector == 'Medical Devices':
            # Medical device companies
            ebitda_multiple = 12.0
        elif subsector == 'Healthcare Services':
            # Service companies typically lower multiple
            ebitda_multiple = 10.0
        else:
            # Default healthcare multiple
            ebitda_multiple = 12.0

        # Record the assumptions
        valuation_result['key_assumptions']['ebitda'] = float(ebitda)
        valuation_result['key_assumptions']['ebitda_multiple'] = float(ebitda_multiple)

        # Calculate enterprise value
        enterprise_value = ebitda * ebitda_multiple

        # Calculate net debt
        net_debt = 0
        if balance_sheet is not None and not balance_sheet.empty:
            balance = balance_sheet.iloc[:, 0]

            # Calculate total debt
            total_debt = 0
            for debt_field in ['Total Debt', 'Long Term Debt', 'Short Term Debt']:
                if debt_field in balance.index:
                    total_debt += balance[debt_field]

            # Get cash and equivalents
            cash = 0
            if 'Cash and Cash Equivalents' in balance.index:
                cash = balance['Cash and Cash Equivalents']

            net_debt = total_debt - cash

        # Calculate equity value
        equity_value = enterprise_value - net_debt

        # Calculate per share value
        if shares_outstanding:
            value_per_share = equity_value / shares_outstanding

            # Calculate upside/downside vs current price
            if current_price:
                upside_potential = (value_per_share / current_price) - 1
                valuation_result['upside_potential'] = float(upside_potential)

            valuation_result['value_per_share'] = float(value_per_share)

        valuation_result['enterprise_value'] = float(enterprise_value)
        valuation_result['equity_value'] = float(equity_value)
        valuation_result['net_debt'] = float(net_debt)

        # Add sensitivity analysis with different multiples
        multiples_sensitivity = {}
        for multiple_delta in [-3, -2, -1, 1, 2, 3]:
            test_multiple = ebitda_multiple + multiple_delta
            test_ev = ebitda * test_multiple
            test_equity = test_ev - net_debt

            if shares_outstanding:
                test_price = test_equity / shares_outstanding
                multiples_sensitivity[f'multiple_{test_multiple:.1f}x'] = float(test_price)

        valuation_result['multiples_sensitivity'] = multiples_sensitivity

        return valuation_result


# Register this analyzer with the factory
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory

SectorFactory.register_analyzer("Healthcare", HealthcareSectorAnalyzer)