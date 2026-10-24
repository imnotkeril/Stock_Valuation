import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

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
logger = logging.getLogger('financial_sector')


class FinancialSectorAnalyzer(SectorAnalyzer):
    """
    Financial Sector Analyzer for banks, insurance companies, and other financial institutions.

    Financial companies have unique characteristics requiring special analysis:
    1. Heavy regulation affects capital requirements and growth potential
    2. Net interest margin and loan quality are critical performance indicators
    3. Traditional DCF models often don't work well due to difficulty forecasting cash flows
    4. Book value and return on equity are more important than for other sectors
    5. Different subsectors (banking, insurance, asset management) have distinct metrics
    """

    def __init__(self, sector_name: str = "Financials"):
        """Initialize the financial sector analyzer"""
        super().__init__(sector_name)
        logger.info("Initialized Financial Sector Analyzer")

        # Additional financial sector specific benchmarks if not in config
        self.financial_benchmarks = {
            "bank": {
                "net_interest_margin": 0.035,  # 3.5% is typical for healthy banks
                "efficiency_ratio": 0.60,  # Below 60% is generally good
                "loan_loss_provision": 0.01,  # 1% of total loans
                "tier_1_capital_ratio": 0.12,  # 12% or higher is strong
                "non_performing_loans_ratio": 0.01  # Below 1% is excellent
            },
            "insurance": {
                "combined_ratio": 0.95,  # Below 100% means underwriting profit
                "loss_ratio": 0.65,  # 65% is typically good
                "expense_ratio": 0.30,  # 30% is standard target
                "return_on_equity": 0.15,  # 15% or higher is strong
                "investment_yield": 0.04  # 4% is a solid investment return
            },
            "asset_management": {
                "fee_revenue_to_aum": 0.01,  # 1% of assets under management
                "aum_growth": 0.10,  # 10% annual growth in AUM
                "operating_margin": 0.30,  # 30% is strong
                "revenue_per_employee": 500000,  # $500K per employee
                "client_retention_rate": 0.90  # 90% retention is good
            }
        }

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a financial company using sector-specific metrics

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
            industry = company_info.get('industry', 'Financials')
            subsector = self._determine_financial_subsector(industry)

            # Initialize analysis results
            analysis = {
                'company_name': company_name,
                'sector': 'Financials',
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

            # Calculate general financial ratios
            analysis['metrics'] = self._calculate_financial_metrics(
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
            logger.error(f"Error analyzing financial company: {e}")
            return {'error': str(e)}

    def calculate_valuation(self, financial_data: Dict[str, Any],
                            valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate financial institution valuation using sector-appropriate methods

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
            industry = company_info.get('industry', 'Financials')
            subsector = self._determine_financial_subsector(industry)

            # Determine valuation method if not specified
            if valuation_method is None:
                # For financial firms, different methods are preferred for different subsectors
                if subsector == 'bank':
                    valuation_method = 'dividend_discount'
                elif subsector == 'insurance':
                    valuation_method = 'adjusted_book_value'
                elif subsector == 'asset_management':
                    valuation_method = 'aum_multiple'
                else:
                    valuation_method = 'excess_return'  # Default for financials

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()

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
            if valuation_method == 'dividend_discount':
                valuation_result = self._dividend_discount_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'adjusted_book_value':
                valuation_result = self._adjusted_book_value_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'excess_return':
                valuation_result = self._excess_return_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'aum_multiple':
                valuation_result = self._aum_multiple_valuation(
                    financial_data, valuation_result
                )
            elif valuation_method == 'relative_valuation':
                valuation_result = self._relative_valuation(
                    financial_data, valuation_result
                )
            else:
                # Default to a combination of methods for financial firms
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
            logger.error(f"Error calculating financial company valuation: {e}")
            return {
                'error': str(e),
                'valuation_method': valuation_method or 'unknown'
            }

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with financial sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        try:
            # Determine subsector if available in metrics
            subsector = company_metrics.get('subsector', 'bank')  # Default to bank

            # Get appropriate benchmarks
            sector_benchmarks = self.get_sector_benchmarks()
            subsector_benchmarks = self.financial_benchmarks.get(subsector, {})

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
                    if metric in ['efficiency_ratio', 'expense_ratio', 'combined_ratio',
                                  'non_performing_loans_ratio', 'cost_of_risk', 'debt_to_equity']:
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
            logger.error(f"Error benchmarking financial company: {e}")
            return {'error': str(e)}

    def _determine_financial_subsector(self, industry: str) -> str:
        """Determine the specific financial subsector"""
        industry = industry.lower() if industry else ""

        if any(term in industry for term in ['bank', 'credit', 'deposit', 'lending']):
            return 'bank'
        elif any(term in industry for term in ['insurance', 'reinsurance', 'assurance']):
            return 'insurance'
        elif any(term in industry for term in ['asset management', 'investment management',
                                               'wealth', 'fund', 'capital management']):
            return 'asset_management'
        else:
            return 'bank'  # Default to bank

    def _calculate_financial_metrics(self, income: pd.Series, balance: pd.Series,
                                     cash_flow: pd.Series, subsector: str) -> Dict[str, float]:
        """Calculate financial sector specific metrics"""
        metrics = {
            'subsector': subsector
        }

        # General financial metrics
        if 'Total Revenue' in income.index and 'Net Income' in income.index:
            revenue = income['Total Revenue']
            net_income = income['Net Income']

            if revenue > 0:
                metrics['net_margin'] = net_income / revenue

        if 'Total Stockholder Equity' in balance.index and 'Net Income' in income.index:
            equity = balance['Total Stockholder Equity']
            net_income = income['Net Income']

            if equity > 0:
                metrics['return_on_equity'] = net_income / equity

        if 'Total Assets' in balance.index and 'Net Income' in income.index:
            assets = balance['Total Assets']
            net_income = income['Net Income']

            if assets > 0:
                metrics['return_on_assets'] = net_income / assets

        if 'Total Stockholder Equity' in balance.index and 'Total Assets' in balance.index:
            equity = balance['Total Stockholder Equity']
            assets = balance['Total Assets']

            if equity > 0:
                metrics['equity_to_assets'] = equity / assets

        # Bank-specific metrics
        if subsector == 'bank':
            # Net Interest Margin - key banking metric
            if 'Interest Income' in income.index and 'Interest Expense' in income.index and 'Total Assets' in balance.index:
                interest_income = income['Interest Income']
                interest_expense = income['Interest Expense']
                assets = balance['Total Assets']

                if assets > 0:
                    net_interest_income = interest_income - interest_expense
                    metrics['net_interest_margin'] = net_interest_income / assets

            # Efficiency Ratio - operating expense / revenue
            if 'Total Revenue' in income.index and 'Non Interest Expense' in income.index:
                revenue = income['Total Revenue']
                non_interest_expense = income['Non Interest Expense']

                if revenue > 0:
                    metrics['efficiency_ratio'] = non_interest_expense / revenue

            # Loan to Deposit Ratio
            if 'Net Loans' in balance.index and 'Total Deposits' in balance.index:
                loans = balance['Net Loans']
                deposits = balance['Total Deposits']

                if deposits > 0:
                    metrics['loan_to_deposit_ratio'] = loans / deposits

            # Non-Performing Loans Ratio
            if 'Non Performing Loans' in balance.index and 'Net Loans' in balance.index:
                npl = balance['Non Performing Loans']
                loans = balance['Net Loans']

                if loans > 0:
                    metrics['non_performing_loans_ratio'] = npl / loans

        # Insurance-specific metrics
        elif subsector == 'insurance':
            # Combined Ratio - key insurance profitability metric
            if 'Loss Expense' in income.index and 'Underwriting Expense' in income.index and 'Premium Income' in income.index:
                loss_expense = income['Loss Expense']
                underwriting_expense = income['Underwriting Expense']
                premium_income = income['Premium Income']

                if premium_income > 0:
                    metrics['combined_ratio'] = (loss_expense + underwriting_expense) / premium_income

            # Loss Ratio
            if 'Loss Expense' in income.index and 'Premium Income' in income.index:
                loss_expense = income['Loss Expense']
                premium_income = income['Premium Income']

                if premium_income > 0:
                    metrics['loss_ratio'] = loss_expense / premium_income

            # Expense Ratio
            if 'Underwriting Expense' in income.index and 'Premium Income' in income.index:
                underwriting_expense = income['Underwriting Expense']
                premium_income = income['Premium Income']

                if premium_income > 0:
                    metrics['expense_ratio'] = underwriting_expense / premium_income

            # Investment Yield
            if 'Investment Income' in income.index and 'Total Investments' in balance.index:
                investment_income = income['Investment Income']
                investments = balance['Total Investments']

                if investments > 0:
                    metrics['investment_yield'] = investment_income / investments

        # Asset Management specific metrics
        elif subsector == 'asset_management':
            # Assets Under Management metrics (if available)
            if 'Assets Under Management' in balance.index:
                aum = balance['Assets Under Management']
                metrics['assets_under_management'] = aum

                # Fee Revenue to AUM
                if 'Fee Revenue' in income.index:
                    fee_revenue = income['Fee Revenue']

                    if aum > 0:
                        metrics['fee_revenue_to_aum'] = fee_revenue / aum

            # Operating Margin
            if 'Operating Income' in income.index and 'Total Revenue' in income.index:
                operating_income = income['Operating Income']
                revenue = income['Total Revenue']

                if revenue > 0:
                    metrics['operating_margin'] = operating_income / revenue

        return metrics

    def _assess_performance(self, metrics: Dict[str, float], subsector: str) -> Dict[str, str]:
        """Assess performance of financial metrics against benchmarks"""
        assessment = {}

        # Get appropriate benchmarks
        sector_benchmarks = self.get_sector_benchmarks()
        subsector_benchmarks = self.financial_benchmarks.get(subsector, {})

        # Combine benchmarks
        benchmarks = {**sector_benchmarks, **subsector_benchmarks}

        for metric, value in metrics.items():
            if metric in benchmarks and isinstance(value, (int, float)):
                benchmark = benchmarks[metric]

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ['efficiency_ratio', 'expense_ratio', 'combined_ratio',
                              'non_performing_loans_ratio', 'cost_of_risk', 'debt_to_equity']:
                    higher_is_better = False

                # Assess performance
                if higher_is_better:
                    if value >= benchmark * 1.2:
                        assessment[metric] = "Excellent"
                    elif value >= benchmark:
                        assessment[metric] = "Good"
                    elif value >= benchmark * 0.8:
                        assessment[metric] = "Average"
                    else:
                        assessment[metric] = "Below Average"
                else:  # Lower is better
                    if value <= benchmark * 0.8:
                        assessment[metric] = "Excellent"
                    elif value <= benchmark:
                        assessment[metric] = "Good"
                    elif value <= benchmark * 1.2:
                        assessment[metric] = "Average"
                    else:
                        assessment[metric] = "Below Average"

        return assessment

    def _identify_strengths_weaknesses(self, performance_assessment: Dict[str, str]) -> tuple:
        """Identify key strengths and weaknesses based on performance assessment"""
        strengths = []
        weaknesses = []

        for metric, assessment in performance_assessment.items():
            # Create a readable metric name
            readable_metric = metric.replace('_', ' ').title()

            if assessment in ["Excellent", "Good"]:
                strengths.append(f"{readable_metric}: {assessment}")
            elif assessment == "Below Average":
                weaknesses.append(f"{readable_metric}: {assessment}")

        return strengths, weaknesses

    def _analyze_risk_factors(self, financial_data: Dict[str, Any],
                              metrics: Dict[str, float], subsector: str) -> List[str]:
        """Identify key risk factors for the financial institution"""
        risk_factors = []

        # Extract financial statements for historical analysis
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Analyze capital adequacy (key for all financial institutions)
        if 'equity_to_assets' in metrics:
            equity_to_assets = metrics['equity_to_assets']

            if equity_to_assets < 0.05:
                risk_factors.append(
                    "Very low capital buffer: equity-to-assets ratio below 5% indicates high leverage and vulnerability to economic shocks")
            elif equity_to_assets < 0.08:
                risk_factors.append(
                    "Low capital buffer: equity-to-assets ratio below 8% may indicate insufficient capital cushion")

        # Subsector specific risks
        if subsector == 'bank':
            # Asset quality risks
            if 'non_performing_loans_ratio' in metrics:
                npl_ratio = metrics['non_performing_loans_ratio']

                if npl_ratio > 0.05:
                    risk_factors.append(
                        "High non-performing loans: NPL ratio above 5% indicates significant credit quality concerns")
                elif npl_ratio > 0.02:
                    risk_factors.append(
                        "Elevated non-performing loans: NPL ratio above 2% suggests moderate credit quality issues")

            # Funding risks
            if 'loan_to_deposit_ratio' in metrics:
                ltd_ratio = metrics['loan_to_deposit_ratio']

                if ltd_ratio > 1.0:
                    risk_factors.append(
                        "Funding risk: loan-to-deposit ratio above 100% indicates reliance on wholesale funding sources")
                elif ltd_ratio > 0.9:
                    risk_factors.append(
                        "Potential funding pressure: loan-to-deposit ratio above 90% suggests limited deposit funding cushion")

            # Profitability sustainability
            if 'net_interest_margin' in metrics:
                nim = metrics['net_interest_margin']

                if nim < 0.02:
                    risk_factors.append(
                        "Low net interest margin: NIM below 2% may indicate challenges in core banking profitability")

        elif subsector == 'insurance':
            # Underwriting risks
            if 'combined_ratio' in metrics:
                combined_ratio = metrics['combined_ratio']

                if combined_ratio > 1.05:
                    risk_factors.append(
                        "Underwriting losses: combined ratio above 105% indicates significant underwriting losses")
                elif combined_ratio > 1.0:
                    risk_factors.append(
                        "Underwriting concerns: combined ratio above 100% indicates underwriting losses")

            # Investment risk
            if 'investment_yield' in metrics:
                investment_yield = metrics['investment_yield']

                if investment_yield < 0.02:
                    risk_factors.append(
                        "Low investment returns: investment yield below 2% may indicate conservative portfolio or challenging market conditions")

        elif subsector == 'asset_management':
            # Fee pressure
            if 'fee_revenue_to_aum' in metrics:
                fee_to_aum = metrics['fee_revenue_to_aum']

                if fee_to_aum < 0.005:
                    risk_factors.append(
                        "Fee pressure: fee revenue to AUM ratio below 0.5% suggests significant fee pressure")

            # Operating leverage
            if 'operating_margin' in metrics:
                operating_margin = metrics['operating_margin']

                if operating_margin < 0.15:
                    risk_factors.append(
                        "Low operating margin: margin below 15% may indicate insufficient operating leverage or high cost structure")

        # Analyze historical trends if data available
        if income_stmt is not None and income_stmt.shape[1] >= 3:
            # Look for declining revenues
            if 'Total Revenue' in income_stmt.index:
                recent_revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                previous_revenue = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 2]

                if recent_revenue < previous_revenue:
                    risk_factors.append("Declining revenue trend: revenues have decreased over the past few years")

            # Look for declining profitability
            if 'Net Income' in income_stmt.index:
                recent_income = income_stmt.iloc[income_stmt.index.get_loc('Net Income'), 0]
                previous_income = income_stmt.iloc[income_stmt.index.get_loc('Net Income'), 2]

                if recent_income < previous_income:
                    risk_factors.append("Declining profitability: net income has decreased over the past few years")

        # Add general financial sector risks
        risk_factors.append(
            "Regulatory risk: financial institutions face evolving regulatory requirements that may impact business models and profitability")
        risk_factors.append(
            "Interest rate risk: changes in interest rates can significantly impact profitability and asset values")

        return risk_factors

    def _dividend_discount_valuation(self, financial_data: Dict[str, Any],
                                     valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Dividend Discount Model valuation for financial companies

        For banks and financial institutions with stable dividends, DDM is often the most appropriate valuation method
        """
        try:
            # Extract required data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None or shares_outstanding is None:
                raise ValueError("Missing required data for dividend discount valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]

            # Check if the company pays dividends
            dividend_data = {}

            # Try different approaches to find dividend information
            if 'Dividends Paid' in latest_income.index:
                dividends_paid = abs(latest_income['Dividends Paid'])
                dividend_per_share = dividends_paid / shares_outstanding
                dividend_data['dividend_per_share'] = dividend_per_share
            elif 'Dividend Per Share' in latest_income.index:
                dividend_per_share = latest_income['Dividend Per Share']
                dividend_data['dividend_per_share'] = dividend_per_share
            elif 'dividend_yield' in market_data and market_data['dividend_yield'] > 0 and market_data.get(
                    'share_price'):
                # Calculate from yield and price
                dividend_per_share = market_data['share_price'] * market_data['dividend_yield']
                dividend_data['dividend_per_share'] = dividend_per_share
            else:
                raise ValueError("No dividend information available for DDM valuation")

            if dividend_data['dividend_per_share'] <= 0:
                raise ValueError("Company does not pay dividends, DDM valuation not applicable")

            # Determine dividend growth rate
            dividend_growth_rate = self._estimate_sustainable_growth_rate(financial_data)
            dividend_data['growth_rate'] = dividend_growth_rate

            # Determine discount rate (required rate of return)
            required_return = self._calculate_required_return(financial_data)
            dividend_data['required_return'] = required_return

            # Make sure required return > growth rate for model validity
            if required_return <= dividend_growth_rate:
                # Adjust growth rate down or required return up to ensure model validity
                adjusted_growth_rate = required_return * 0.7  # 70% of required return
                dividend_data['original_growth_rate'] = dividend_growth_rate
                dividend_data['growth_rate'] = adjusted_growth_rate
                dividend_growth_rate = adjusted_growth_rate
                dividend_data['growth_adjustment_note'] = "Growth rate was adjusted down to maintain model validity"

            # Determine if multi-stage or single-stage model is appropriate
            if dividend_growth_rate > required_return * 0.7:  # If growth is high relative to discount rate
                # Use a multi-stage model with transitioning growth rates
                high_growth_years = 5

                # Calculate present value of near-term dividends
                pv_dividends = 0
                next_dividend = dividend_data['dividend_per_share']

                for year in range(1, high_growth_years + 1):
                    next_dividend *= (1 + dividend_growth_rate)
                    pv_dividends += next_dividend / ((1 + required_return) ** year)

                # Calculate terminal value with lower long-term growth
                terminal_growth_rate = min(0.04, required_return * 0.5)  # Lower of 4% or 50% of required return
                terminal_dividend = next_dividend * (1 + terminal_growth_rate)
                terminal_value = terminal_dividend / (required_return - terminal_growth_rate)

                # Discount terminal value to present
                pv_terminal = terminal_value / ((1 + required_return) ** high_growth_years)

                # Sum for total value
                estimated_value = pv_dividends + pv_terminal

                # Add details to result
                dividend_data['high_growth_years'] = high_growth_years
                dividend_data['terminal_growth_rate'] = terminal_growth_rate
                dividend_data['pv_dividends'] = pv_dividends
                dividend_data['pv_terminal'] = pv_terminal
                dividend_data['model_type'] = 'multi-stage'
            else:
                # Use single-stage (Gordon Growth) model
                next_dividend = dividend_data['dividend_per_share'] * (1 + dividend_growth_rate)
                estimated_value = next_dividend / (required_return - dividend_growth_rate)

                dividend_data['next_dividend'] = next_dividend
                dividend_data['model_type'] = 'single-stage'

            # Update valuation result
            valuation_result['estimated_value_per_share'] = estimated_value
            valuation_result['valuation_details']['dividend_discount'] = dividend_data
            valuation_result['assumptions'] = {
                'dividend_per_share': dividend_data['dividend_per_share'],
                'dividend_growth_rate': dividend_growth_rate,
                'required_return': required_return,
                'terminal_growth_rate': dividend_data.get('terminal_growth_rate', dividend_growth_rate)
            }

            # Add sensitivity analysis
            sensitivity = {}

            # Sensitivity to growth rate
            growth_scenarios = {
                'low_growth': max(0.01, dividend_growth_rate - 0.02),
                'base_growth': dividend_growth_rate,
                'high_growth': dividend_growth_rate + 0.02
            }

            growth_values = {}
            for scenario, growth in growth_scenarios.items():
                if growth >= required_return:
                    growth = required_return - 0.01  # Ensure model validity

                if dividend_data['model_type'] == 'single-stage':
                    next_div = dividend_data['dividend_per_share'] * (1 + growth)
                    growth_values[scenario] = next_div / (required_return - growth)
                else:
                    # Simplified sensitivity for multi-stage
                    growth_impact = (growth / dividend_growth_rate) * estimated_value
                    growth_values[scenario] = growth_impact

            sensitivity['growth_rate'] = growth_values

            # Sensitivity to required return
            return_scenarios = {
                'low_return': max(0.05, required_return - 0.02),
                'base_return': required_return,
                'high_return': required_return + 0.02
            }

            return_values = {}
            for scenario, ret in return_scenarios.items():
                if ret <= dividend_growth_rate:
                    ret = dividend_growth_rate + 0.01  # Ensure model validity

                if dividend_data['model_type'] == 'single-stage':
                    next_div = dividend_data['dividend_per_share'] * (1 + dividend_growth_rate)
                    return_values[scenario] = next_div / (ret - dividend_growth_rate)
                else:
                    # Simplified sensitivity for multi-stage
                    return_impact = (required_return / ret) * estimated_value
                    return_values[scenario] = return_impact

            sensitivity['required_return'] = return_values

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error performing dividend discount valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _estimate_sustainable_growth_rate(self, financial_data: Dict[str, Any]) -> float:
        """
        Estimate sustainable growth rate based on ROE and retention ratio

        Sustainable Growth Rate = Return on Equity * (1 - Dividend Payout Ratio)
        """
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None:
                return 0.03  # Default growth rate if data not available

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate ROE
            if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                net_income = latest_income['Net Income']
                equity = latest_balance['Total Stockholder Equity']

                if equity > 0:
                    roe = net_income / equity
                else:
                    roe = 0.10  # Default ROE if data issue
            else:
                roe = 0.10  # Default ROE if metrics not available

            # Calculate retention ratio (1 - payout ratio)
            retention_ratio = 0.6  # Default retention ratio (60%)

            if 'Dividends Paid' in latest_income.index and 'Net Income' in latest_income.index:
                dividends_paid = abs(latest_income['Dividends Paid'])
                net_income = latest_income['Net Income']

                if net_income > 0:
                    payout_ratio = dividends_paid / net_income
                    retention_ratio = 1 - payout_ratio

            # Calculate sustainable growth rate
            growth_rate = roe * retention_ratio

            # Cap growth rate to reasonable range for financial institutions
            return min(max(growth_rate, 0.01), 0.15)  # Between 1% and 15%

        except Exception as e:
            logger.warning(f"Error estimating sustainable growth rate: {e}")
            return 0.03  # Default growth rate if calculation fails

    def _calculate_required_return(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate required rate of return using CAPM

        Required Return = Risk-Free Rate + Beta * Market Risk Premium
        """
        try:
            market_data = financial_data.get('market_data', {})
            company_info = financial_data.get('company_info', {})

            # Get risk-free rate (use 10-year Treasury yield as proxy)
            risk_free_rate = 0.03  # Default 3%

            # Get beta
            beta = market_data.get('beta', 1.0)  # Default to market beta

            # Market risk premium
            market_risk_premium = 0.05  # Standard assumption: 5%

            # Financial industry may have higher risk premium
            subsector = self._determine_financial_subsector(company_info.get('industry', ''))
            if subsector == 'bank':
                industry_premium = 0.01  # Additional 1% for banking risk
            elif subsector == 'insurance':
                industry_premium = 0.015  # Additional 1.5% for insurance risk
            else:
                industry_premium = 0.01  # Default additional premium

            # Calculate CAPM required return
            required_return = risk_free_rate + (beta * market_risk_premium) + industry_premium

            # Ensure required return is within reasonable bounds
            return min(max(required_return, 0.06), 0.15)  # Between 6% and 15%

        except Exception as e:
            logger.warning(f"Error calculating required return: {e}")
            return 0.09  # Default 9% if calculation fails

    def _adjusted_book_value_valuation(self, financial_data: Dict[str, Any],
                                       valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjusted Book Value valuation - especially useful for insurance companies

        Adjusts the book value for factors like undervalued investments,
        potential claim reserve inadequacy, and intangible assets.
        """
        try:
            # Extract required data
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})
            company_info = financial_data.get('company_info', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if balance_sheet is None or shares_outstanding is None:
                raise ValueError("Missing required data for adjusted book value valuation")

            # Get latest balance sheet data
            latest_balance = balance_sheet.iloc[:, 0]

            # Get book value (equity)
            if 'Total Stockholder Equity' in latest_balance.index:
                book_value = latest_balance['Total Stockholder Equity']
            else:
                raise ValueError("Total Stockholder Equity not found in balance sheet")

            # Initialize adjustments dictionary
            adjustments = {}

            # Determine subsector for specific adjustments
            subsector = self._determine_financial_subsector(company_info.get('industry', ''))

            # Start with base book value
            adjusted_book_value = book_value

            # Make subsector-specific adjustments
            if subsector == 'insurance':
                # For insurance companies, adjust for reserve adequacy
                if 'Loss Reserves' in latest_balance.index:
                    loss_reserves = latest_balance['Loss Reserves']
                    # Adjust for potential reserve inadequacy (conservative assumption)
                    reserve_adjustment = -loss_reserves * 0.05  # 5% potential shortfall
                    adjusted_book_value += reserve_adjustment
                    adjustments['reserve_adjustment'] = reserve_adjustment

                # Adjust for investment portfolio valuation
                if 'Total Investments' in latest_balance.index:
                    investments = latest_balance['Total Investments']
                    # Adjust for potential unrealized gains/losses (simplified)
                    investment_adjustment = investments * 0.03  # 3% potential undervaluation
                    adjusted_book_value += investment_adjustment
                    adjustments['investment_adjustment'] = investment_adjustment

            elif subsector == 'bank':
                # For banks, adjust for loan loss reserve adequacy
                if 'Loan Loss Reserves' in latest_balance.index and 'Net Loans' in latest_balance.index:
                    loan_loss_reserves = latest_balance['Loan Loss Reserves']
                    net_loans = latest_balance['Net Loans']

                    # Calculate current reserve ratio
                    current_reserve_ratio = loan_loss_reserves / net_loans if net_loans > 0 else 0

                    # Assume adequate ratio based on economic conditions (simplified)
                    adequate_ratio = 0.015  # 1.5% in normal conditions

                    if current_reserve_ratio < adequate_ratio:
                        # Adjust for potential reserve shortfall
                        reserve_shortfall = (adequate_ratio - current_reserve_ratio) * net_loans
                        adjusted_book_value -= reserve_shortfall
                        adjustments['loan_loss_reserve_adjustment'] = -reserve_shortfall

            # Adjust for goodwill and intangibles (common for all financial institutions)
            if 'Goodwill' in latest_balance.index:
                goodwill = latest_balance['Goodwill']
                # Typically discount goodwill in financial valuation (conservative)
                goodwill_adjustment = -goodwill * 0.5  # 50% haircut to goodwill
                adjusted_book_value += goodwill_adjustment
                adjustments['goodwill_adjustment'] = goodwill_adjustment

            if 'Intangible Assets' in latest_balance.index:
                intangibles = latest_balance['Intangible Assets']
                # Typically discount intangibles in financial valuation (conservative)
                intangible_adjustment = -intangibles * 0.3  # 30% haircut to intangibles
                adjusted_book_value += intangible_adjustment
                adjustments['intangible_adjustment'] = intangible_adjustment

            # Calculate book value per share
            book_value_per_share = book_value / shares_outstanding if shares_outstanding > 0 else 0

            # Calculate adjusted book value per share
            adjusted_book_value_per_share = adjusted_book_value / shares_outstanding if shares_outstanding > 0 else 0

            # Apply valuation multiple based on ROE
            income_stmt = financial_data.get('income_statement')
            if income_stmt is not None and 'Net Income' in income_stmt.iloc[:, 0].index:
                net_income = income_stmt.iloc[:, 0]['Net Income']
                if book_value > 0:
                    roe = net_income / book_value

                    # Determine P/B multiple based on ROE
                    if roe >= 0.15:  # High ROE
                        pb_multiple = 1.5
                    elif roe >= 0.10:  # Good ROE
                        pb_multiple = 1.2
                    elif roe >= 0.05:  # Average ROE
                        pb_multiple = 1.0
                    else:  # Below average ROE
                        pb_multiple = 0.8

                    # Apply multiple to adjusted book value
                    final_value_per_share = adjusted_book_value_per_share * pb_multiple

                    adjustments['pb_multiple'] = pb_multiple
                    adjustments['roe'] = roe
                else:
                    final_value_per_share = adjusted_book_value_per_share
            else:
                # If ROE calculation not possible, use standard multiple
                final_value_per_share = adjusted_book_value_per_share * 1.0
                adjustments['pb_multiple'] = 1.0

            # Update valuation result
            valuation_result['estimated_value_per_share'] = final_value_per_share
            valuation_result['valuation_details']['adjusted_book_value'] = {
                'book_value': book_value,
                'book_value_per_share': book_value_per_share,
                'adjusted_book_value': adjusted_book_value,
                'adjusted_book_value_per_share': adjusted_book_value_per_share,
                'final_value_per_share': final_value_per_share,
                'adjustments': adjustments
            }

            valuation_result['assumptions'] = {
                'book_value': book_value,
                'adjusted_book_value': adjusted_book_value,
                'pb_multiple': adjustments.get('pb_multiple', 1.0)
            }

            # Add sensitivity analysis
            sensitivity = {}

            # Sensitivity to P/B multiple
            pb_scenarios = {
                'conservative': adjustments.get('pb_multiple', 1.0) * 0.8,
                'base_case': adjustments.get('pb_multiple', 1.0),
                'optimistic': adjustments.get('pb_multiple', 1.0) * 1.2
            }

            pb_values = {}
            for scenario, multiple in pb_scenarios.items():
                pb_values[scenario] = adjusted_book_value_per_share * multiple

            sensitivity['pb_multiple'] = pb_values

            # Sensitivity to adjustment severity
            adjustment_scenarios = {
                'more_conservative': 1.5,  # 50% more severe adjustments
                'base_case': 1.0,
                'less_conservative': 0.5  # 50% less severe adjustments
            }

            adjustment_values = {}
            total_adjustments = sum([adj for adj in adjustments.values() if isinstance(adj, (int, float))])

            for scenario, factor in adjustment_scenarios.items():
                modified_book_value = book_value + (total_adjustments * factor)
                modified_per_share = modified_book_value / shares_outstanding if shares_outstanding > 0 else 0
                adjustment_values[scenario] = modified_per_share * adjustments.get('pb_multiple', 1.0)

            sensitivity['adjustment_severity'] = adjustment_values

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error performing adjusted book value valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _excess_return_valuation(self, financial_data: Dict[str, Any],
                                 valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Excess Return Model valuation, also known as the Residual Income Model

        Combines book value with the present value of expected excess returns
        (returns above the cost of equity) to value financial institutions.
        """
        try:
            # Extract required data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None or shares_outstanding is None:
                raise ValueError("Missing required data for excess return valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get key financial metrics
            if 'Net Income' not in latest_income.index or 'Total Stockholder Equity' not in latest_balance.index:
                raise ValueError("Required metrics not found in financial statements")

            net_income = latest_income['Net Income']
            equity = latest_balance['Total Stockholder Equity']

            # Calculate ROE
            roe = net_income / equity if equity > 0 else 0

            # Calculate required return (cost of equity)
            required_return = self._calculate_required_return(financial_data)

            # Calculate excess return (ROE - Required Return)
            excess_return = roe - required_return

            # Calculate book value per share
            book_value_per_share = equity / shares_outstanding

            # Initialize valuation details
            excess_return_details = {
                'book_value': equity,
                'book_value_per_share': book_value_per_share,
                'roe': roe,
                'required_return': required_return,
                'excess_return': excess_return
            }

            # If excess return is negative, value may be below book value
            if excess_return <= 0:
                # Company destroying value, apply discount to book value
                discount_factor = max(0.7, 1 + (excess_return * 5))  # Limit discount to 30%
                estimated_value = book_value_per_share * discount_factor

                excess_return_details['discount_factor'] = discount_factor
                excess_return_details['estimated_value'] = estimated_value
                excess_return_details[
                    'note'] = "Negative excess return indicates value destruction, applying discount to book value"
            else:
                # Forecast future excess returns
                forecast_years = 5
                terminal_growth_rate = min(0.03, required_return / 2)  # Conservative long-term growth

                # Initialize present value calculation
                present_value_excess_returns = 0

                # Current book value and excess return
                current_book_value = equity
                current_excess_return = net_income - (required_return * current_book_value)

                forecast_details = []

                # Project and discount future excess returns
                for year in range(1, forecast_years + 1):
                    # Project book value growth (assuming some earnings retention)
                    earnings_retention = 0.5  # Assume 50% retention ratio
                    projected_book_value = current_book_value * (1 + (roe * earnings_retention))

                    # Project excess return with declining trend (competitive forces)
                    persistence_factor = 0.9 ** (year - 1)
                    projected_excess_return = current_excess_return * persistence_factor

                    # Calculate present value of this year's excess return
                    pv_factor = 1 / ((1 + required_return) ** year)
                    pv_excess_return = projected_excess_return * pv_factor

                    # Add to total
                    present_value_excess_returns += pv_excess_return

                    # Store forecast details
                    forecast_details.append({
                        'year': year,
                        'projected_book_value': projected_book_value,
                        'projected_excess_return': projected_excess_return,
                        'pv_excess_return': pv_excess_return
                    })

                    # Update for next iteration
                    current_book_value = projected_book_value
                    current_excess_return = projected_excess_return

                # Calculate terminal value of excess returns
                if excess_return > 0:
                    terminal_excess_return = current_excess_return * (1 + terminal_growth_rate)
                    terminal_value = terminal_excess_return / (required_return - terminal_growth_rate)
                    pv_terminal_value = terminal_value / ((1 + required_return) ** forecast_years)
                else:
                    pv_terminal_value = 0

                # Calculate total estimated value
                estimated_value = book_value_per_share + (present_value_excess_returns / shares_outstanding) + (
                            pv_terminal_value / shares_outstanding)

                # Add forecast and terminal value details
                excess_return_details['forecast_years'] = forecast_years
                excess_return_details['forecast_details'] = forecast_details
                excess_return_details['terminal_growth_rate'] = terminal_growth_rate
                excess_return_details['pv_excess_returns'] = present_value_excess_returns
                excess_return_details['pv_terminal_value'] = pv_terminal_value
                excess_return_details['estimated_value'] = estimated_value

            # Update valuation result
            valuation_result['estimated_value_per_share'] = excess_return_details['estimated_value']
            valuation_result['valuation_details']['excess_return'] = excess_return_details

            valuation_result['assumptions'] = {
                'book_value_per_share': book_value_per_share,
                'roe': roe,
                'required_return': required_return,
                'terminal_growth_rate': excess_return_details.get('terminal_growth_rate', 0)
            }

            # Add sensitivity analysis
            sensitivity = {}

            # Sensitivity to required return
            return_scenarios = {
                'low_required_return': max(0.06, required_return - 0.02),
                'base_case': required_return,
                'high_required_return': required_return + 0.02
            }

            return_values = {}
            for scenario, ret in return_scenarios.items():
                # Simple sensitivity approximation
                if excess_return <= 0:
                    # For negative excess return cases
                    discount_factor = max(0.7, 1 + ((roe - ret) * 5))
                    return_values[scenario] = book_value_per_share * discount_factor
                else:
                    # For positive excess return cases
                    sensitivity_factor = required_return / ret
                    return_values[scenario] = book_value_per_share + (
                                sensitivity_factor * (estimated_value - book_value_per_share))

            sensitivity['required_return'] = return_values

            # Sensitivity to ROE
            roe_scenarios = {
                'declining_roe': max(0.03, roe - 0.03),
                'base_case': roe,
                'improving_roe': roe + 0.03
            }

            roe_values = {}
            for scenario, return_on_equity in roe_scenarios.items():
                # Simple sensitivity approximation
                excess_ret = return_on_equity - required_return

                if excess_ret <= 0:
                    discount_factor = max(0.7, 1 + (excess_ret * 5))
                    roe_values[scenario] = book_value_per_share * discount_factor
                else:
                    # Scale the excess return portion based on ROE difference
                    sensitivity_factor = return_on_equity / roe
                    roe_values[scenario] = book_value_per_share + (
                                sensitivity_factor * (estimated_value - book_value_per_share))

            sensitivity['roe'] = roe_values

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error performing excess return valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _aum_multiple_valuation(self, financial_data: Dict[str, Any],
                                valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assets Under Management multiple valuation for asset management firms

        Values asset managers based on a percentage of AUM plus a multiple of earnings
        """
        try:
            # Extract required data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})
            company_info = financial_data.get('company_info', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or shares_outstanding is None:
                raise ValueError("Missing required data for AUM multiple valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]

            # Check if this is indeed an asset management firm
            subsector = self._determine_financial_subsector(company_info.get('industry', ''))
            if subsector != 'asset_management':
                raise ValueError("AUM multiple valuation is only applicable to asset management firms")

            # Look for AUM data
            aum = None

            # Try various field names where AUM might be reported
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                for field in ['Assets Under Management', 'AUM', 'Client Assets']:
                    if field in latest_balance.index:
                        aum = latest_balance[field]
                        break

            # If AUM not found, try estimating from company info or revenue
            if aum is None:
                if 'aum' in company_info:
                    aum = company_info['aum']
                elif 'Total Revenue' in latest_income.index:
                    # Rough estimation: assuming fee revenue is about 0.7% of AUM
                    revenue = latest_income['Total Revenue']
                    aum = revenue / 0.007

            if aum is None:
                raise ValueError("Assets Under Management (AUM) data not available")

            # Get earnings
            if 'Net Income' in latest_income.index:
                earnings = latest_income['Net Income']
            else:
                raise ValueError("Net Income data not available")

            # Calculate operating margin if possible
            operating_margin = None
            if 'Operating Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                operating_income = latest_income['Operating Income']
                revenue = latest_income['Total Revenue']

                if revenue > 0:
                    operating_margin = operating_income / revenue

            # Determine AUM multiple based on profitability
            if operating_margin is not None:
                if operating_margin >= 0.3:  # High margin
                    aum_percentage = 0.03  # 3% of AUM
                elif operating_margin >= 0.2:  # Good margin
                    aum_percentage = 0.025  # 2.5% of AUM
                else:  # Lower margin
                    aum_percentage = 0.02  # 2% of AUM
            else:
                aum_percentage = 0.025  # Default if margin unknown

            # Determine earnings multiple based on growth prospects
            # (would ideally use AUM growth rate if available)
            if 'aum_growth' in company_info:
                aum_growth = company_info['aum_growth']

                if aum_growth >= 0.15:  # High growth
                    earnings_multiple = 15
                elif aum_growth >= 0.08:  # Moderate growth
                    earnings_multiple = 12
                else:  # Low growth
                    earnings_multiple = 10
            else:
                earnings_multiple = 12  # Default multiple

            # Calculate valuation components
            aum_based_value = aum * aum_percentage
            earnings_based_value = earnings * earnings_multiple

            # Combine for total firm value (typical industry approach is a blend)
            firm_value = (aum_based_value * 0.7) + (earnings_based_value * 0.3)

            # Calculate per share value
            value_per_share = firm_value / shares_outstanding

            # Initialize valuation details
            aum_details = {
                'aum': aum,
                'earnings': earnings,
                'operating_margin': operating_margin,
                'aum_percentage': aum_percentage,
                'earnings_multiple': earnings_multiple,
                'aum_based_value': aum_based_value,
                'earnings_based_value': earnings_based_value,
                'firm_value': firm_value,
                'value_per_share': value_per_share
            }

            # Update valuation result
            valuation_result['estimated_value_per_share'] = value_per_share
            valuation_result['valuation_details']['aum_multiple'] = aum_details

            valuation_result['assumptions'] = {
                'aum': aum,
                'aum_percentage': aum_percentage,
                'earnings_multiple': earnings_multiple
            }

            # Add sensitivity analysis
            sensitivity = {}

            # Sensitivity to AUM percentage
            aum_percentage_scenarios = {
                'low_percentage': max(0.01, aum_percentage - 0.01),
                'base_case': aum_percentage,
                'high_percentage': aum_percentage + 0.01
            }

            aum_percentage_values = {}
            for scenario, percentage in aum_percentage_scenarios.items():
                new_aum_value = aum * percentage
                new_firm_value = (new_aum_value * 0.7) + (earnings_based_value * 0.3)
                aum_percentage_values[scenario] = new_firm_value / shares_outstanding

            sensitivity['aum_percentage'] = aum_percentage_values

            # Sensitivity to earnings multiple
            earnings_multiple_scenarios = {
                'conservative_multiple': max(8, earnings_multiple - 3),
                'base_case': earnings_multiple,
                'premium_multiple': earnings_multiple + 3
            }

            earnings_multiple_values = {}
            for scenario, multiple in earnings_multiple_scenarios.items():
                new_earnings_value = earnings * multiple
                new_firm_value = (aum_based_value * 0.7) + (new_earnings_value * 0.3)
                earnings_multiple_values[scenario] = new_firm_value / shares_outstanding

            sensitivity['earnings_multiple'] = earnings_multiple_values

            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error performing AUM multiple valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _relative_valuation(self, financial_data: Dict[str, Any],
                            valuation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Relative valuation using industry-specific multiples
        For financials, typically uses P/B and P/E multiples adjusted for ROE
        """
        try:
            # Extract required data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})
            company_info = financial_data.get('company_info', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None or shares_outstanding is None:
                raise ValueError("Missing required data for relative valuation")

            # Get latest financial data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get required metrics
            if 'Net Income' not in latest_income.index:
                raise ValueError("Net Income not found in income statement")

            if 'Total Stockholder Equity' not in latest_balance.index:
                raise ValueError("Total Stockholder Equity not found in balance sheet")

            net_income = latest_income['Net Income']
            equity = latest_balance['Total Stockholder Equity']

            # Calculate EPS and BVPS
            eps = net_income / shares_outstanding
            bvps = equity / shares_outstanding

            # Calculate ROE
            roe = net_income / equity if equity > 0 else 0

            # Determine subsector
            subsector = self._determine_financial_subsector(company_info.get('industry', ''))

            # Get appropriate industry multiples based on subsector
            if subsector == 'bank':

                # For banks, P/B is key and varies with ROE
                if roe >= 0.15:  # High performing bank
                    pb_multiple = 1.8
                    pe_multiple = 14

                elif roe >= 0.10:  # Average performing bank
                    pb_multiple = 1.3
                    pe_multiple = 12

                else:  # Below average performing bank
                    pb_multiple = 0.9
                    pe_multiple = 10

                # Weight more heavily towards P/B for banks
                pb_weight = 0.7
                pe_weight = 0.3

            elif subsector == 'insurance':

                # For insurance, balance P/B and P/E
                if roe >= 0.15:  # High performing insurer
                    pb_multiple = 1.6
                    pe_multiple = 13

                elif roe >= 0.10:  # Average performing insurer
                    pb_multiple = 1.2
                    pe_multiple = 11

                else:  # Below average performing insurer
                    pb_multiple = 0.8
                    pe_multiple = 9

                # More balanced weights for insurance
                pb_weight = 0.5
                pe_weight = 0.5


            elif subsector == 'asset_management':

                # For asset managers, P/E is typically more important
                if roe >= 0.2:  # High performing asset manager
                    pb_multiple = 3.0
                    pe_multiple = 18

                elif roe >= 0.15:  # Average performing asset manager
                    pb_multiple = 2.2
                    pe_multiple = 15

                else:  # Below average performing asset manager
                    pb_multiple = 1.5
                    pe_multiple = 12

                # Weight more heavily towards P/E for asset managers
                pb_weight = 0.3
                pe_weight = 0.7
            else:

                # Default financial company multiples
                if roe >= 0.12:  # High performing financial
                    pb_multiple = 1.5
                    pe_multiple = 13

                elif roe >= 0.08:  # Average performing financial
                    pb_multiple = 1.2
                    pe_multiple = 11

                else:  # Below average performing financial
                    pb_multiple = 0.9
                    pe_multiple = 9

                # Default weights
                pb_weight = 0.5
                pe_weight = 0.5

            # Calculate valuation based on each multiple
            pb_valuation = bvps * pb_multiple
            pe_valuation = eps * pe_multiple

            # Combined weighted valuation
            weighted_valuation = (pb_valuation * pb_weight) + (pe_valuation * pe_weight)

            # Initialize relative valuation details
            relative_details = {
                'subsector': subsector,
                'eps': eps,
                'bvps': bvps,
                'roe': roe,
                'pb_multiple': pb_multiple,
                'pe_multiple': pe_multiple,
                'pb_valuation': pb_valuation,
                'pe_valuation': pe_valuation,
                'pb_weight': pb_weight,
                'pe_weight': pe_weight,
                'weighted_valuation': weighted_valuation
            }

            # Update valuation result
            valuation_result['estimated_value_per_share'] = weighted_valuation
            valuation_result['valuation_details']['relative_valuation'] = relative_details
            valuation_result['assumptions'] = {
                'eps': eps,
                'bvps': bvps,
                'pb_multiple': pb_multiple,
                'pe_multiple': pe_multiple,
                'subsector': subsector
            }

            # Add sensitivity analysis
            sensitivity = {}

            # Sensitivity to P/B multiple
            pb_multiple_scenarios = {
                'low_pb': pb_multiple * 0.8,
                'base_case': pb_multiple,
                'high_pb': pb_multiple * 1.2
            }

            pb_multiple_values = {}

            for scenario, multiple in pb_multiple_scenarios.items():
                new_pb_valuation = bvps * multiple
                new_weighted_valuation = (new_pb_valuation * pb_weight) + (pe_valuation * pe_weight)
                pb_multiple_values[scenario] = new_weighted_valuation
            sensitivity['pb_multiple'] = pb_multiple_values

            # Sensitivity to P/E multiple
            pe_multiple_scenarios = {
                'low_pe': pe_multiple * 0.8,
                'base_case': pe_multiple,
                'high_pe': pe_multiple * 1.2
            }
            pe_multiple_values = {}

            for scenario, multiple in pe_multiple_scenarios.items():
                new_pe_valuation = eps * multiple
                new_weighted_valuation = (pb_valuation * pb_weight) + (new_pe_valuation * pe_weight)
                pe_multiple_values[scenario] = new_weighted_valuation
            sensitivity['pe_multiple'] = pe_multiple_values
            valuation_result['sensitivity'] = sensitivity

            return valuation_result

        except Exception as e:
            logger.error(f"Error performing relative valuation: {e}")
            valuation_result['error'] = str(e)
            return valuation_result

    def _combined_valuation_approach(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:

        """
        Combined valuation approach using multiple methods
        For financial companies, often a blend of methods provides the most reliable valuation
        """

        try:

            # Initialize an empty result
            combined_result = {
                'company_name': financial_data.get('company_info', {}).get('name', 'Unknown'),
                'ticker': financial_data.get('company_info', {}).get('symbol', 'Unknown'),
                'valuation_method': 'combined_approach',
                'estimated_value_per_share': None,
                'current_price': financial_data.get('market_data', {}).get('share_price'),
                'price_to_value_ratio': None,
                'valuation_assessment': None,
                'valuation_details': {},
                'method_weights': {},
                'individual_valuations': {},
                'assumptions': {}
            }

            # Determine the subsector to choose appropriate methods
            company_info = financial_data.get('company_info', {})
            subsector = self._determine_financial_subsector(company_info.get('industry', ''))

            # Initialize methods to use based on subsector
            methods_to_use = []
            method_weights = {}

            if subsector == 'bank':

                methods_to_use = ['relative_valuation', 'excess_return', 'dividend_discount']
                method_weights = {
                    'relative_valuation': 0.4,
                    'excess_return': 0.4,
                    'dividend_discount': 0.2
                }

            elif subsector == 'insurance':

                methods_to_use = ['adjusted_book_value', 'relative_valuation', 'dividend_discount']
                method_weights = {
                    'adjusted_book_value': 0.4,
                    'relative_valuation': 0.4,
                    'dividend_discount': 0.2
                }

            elif subsector == 'asset_management':

                methods_to_use = ['aum_multiple', 'relative_valuation', 'excess_return']
                method_weights = {
                    'aum_multiple': 0.5,
                    'relative_valuation': 0.3,
                    'excess_return': 0.2
                }

            else:

                methods_to_use = ['relative_valuation', 'adjusted_book_value', 'excess_return']
                method_weights = {
                    'relative_valuation': 0.4,
                    'adjusted_book_value': 0.3,
                    'excess_return': 0.3
                }

            # Store the weights
            combined_result['method_weights'] = method_weights

            # Calculate each valuation method
            for method in methods_to_use:

                # Initialize base result for this method
                method_result = {
                    'valuation_method': method,
                    'estimated_value_per_share': None,
                    'valuation_details': {},
                    'assumptions': {}
                }

                # Calculate valuation using the appropriate method
                if method == 'relative_valuation':
                    method_result = self._relative_valuation(financial_data, method_result)

                elif method == 'excess_return':
                    method_result = self._excess_return_valuation(financial_data, method_result)

                elif method == 'dividend_discount':
                    method_result = self._dividend_discount_valuation(financial_data, method_result)

                elif method == 'adjusted_book_value':
                    method_result = self._adjusted_book_value_valuation(financial_data, method_result)

                elif method == 'aum_multiple':
                    method_result = self._aum_multiple_valuation(financial_data, method_result)

                # Store the individual valuation if successful
                if 'error' not in method_result and method_result['estimated_value_per_share'] is not None:

                    combined_result['individual_valuations'][method] = {
                        'value': method_result['estimated_value_per_share'],
                        'details': method_result['valuation_details'].get(method, {})
                    }

                    # Add the detailed valuation to the combined result
                    combined_result['valuation_details'][method] = method_result['valuation_details'].get(method,
                                                                                                          {})

                    # Combine assumptions
                    for key, value in method_result['assumptions'].items():
                        if key not in combined_result['assumptions']:
                            combined_result['assumptions'][key] = value

            # Calculate weighted average if we have valuations
            if combined_result['individual_valuations']:

                weighted_sum = 0
                total_weight = 0

                for method, weight in method_weights.items():

                    if method in combined_result['individual_valuations']:
                        weighted_sum += combined_result['individual_valuations'][method]['value'] * weight
                        total_weight += weight

                if total_weight > 0:

                    combined_result['estimated_value_per_share'] = weighted_sum / total_weight

                    # Calculate price-to-value ratio
                    if combined_result['current_price'] is not None and combined_result[
                        'estimated_value_per_share'] > 0:
                        combined_result['price_to_value_ratio'] = (
                                combined_result['current_price'] / combined_result['estimated_value_per_share']
                        )

                        # Add assessment based on price-to-value ratio
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

            # Add a summary of the valuation approach
            combined_result['valuation_summary'] = (
                    f"Combined valuation approach used for {subsector} company, with weightings: "
                    + ", ".join([f"{m}: {w * 100:.0f}%" for m, w in method_weights.items() if
                                 m in combined_result['individual_valuations']])
            )

            return combined_result

        except Exception as e:
            logger.error(f"Error in combined valuation approach: {e}")

            return {
                'error': str(e),
                'valuation_method': 'combined_approach'
            }