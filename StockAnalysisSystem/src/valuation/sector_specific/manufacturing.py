import os
import sys
import pandas as pd
import logging
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
logger = logging.getLogger('manufacturing_sector')


class ManufacturingSectorAnalyzer(SectorAnalyzer):
    """
    Specialized analyzer for manufacturing and industrial sector companies.

    This sector includes industrial machinery, equipment manufacturing,
    aerospace & defense, industrial products, and other manufacturing businesses.
    """

    def __init__(self, sector_name: str = "Industrials"):
        """Initialize manufacturing sector analyzer"""
        super().__init__(sector_name=sector_name)
        logger.info("Initialized Manufacturing/Industrial sector analyzer")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform manufacturing-specific analysis on company data

        Args:
            financial_data: Dictionary containing financial statements and other company data

        Returns:
            Dictionary with analysis results specific to manufacturing companies
        """
        result = {
            "sector": "Industrials",
            "industry_metrics": {},
            "sector_specific_analysis": {},
            "risk_factors": [],
            "competitive_advantages": [],
            "operational_efficiency": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Skip analysis if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for manufacturing sector analysis")
            return result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Calculate manufacturing-specific metrics
        self._calculate_manufacturing_metrics(result, latest_income, latest_balance, latest_cash_flow)

        # Analyze operational efficiency
        self._analyze_operational_efficiency(result, latest_income, latest_balance)

        # Identify key industry risk factors
        self._identify_risk_factors(result, financial_data)

        # Evaluate competitive advantages
        self._evaluate_competitive_advantages(result, financial_data)

        return result

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using manufacturing-specific methods

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
            "sector": "Industrials",
            "methods": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for manufacturing sector valuation")
            return valuation_result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None

        # Perform EV/EBITDA valuation
        if valuation_method in ["EV/EBITDA", "all"]:
            self._calculate_ev_ebitda_valuation(valuation_result, latest_income, latest_balance, market_data)

        # Perform ROIC-based valuation
        if valuation_method in ["ROIC", "all"]:
            self._calculate_roic_valuation(valuation_result, financial_data)

        # Perform Asset-based valuation
        if valuation_method in ["Asset", "all"]:
            self._calculate_asset_based_valuation(valuation_result, latest_balance, market_data)

        # Perform DCF with manufacturing-specific adjustments
        if valuation_method in ["DCF", "all"]:
            self._calculate_dcf_valuation(valuation_result, financial_data)

        return valuation_result

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with manufacturing sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks().get("benchmarks", {})

        benchmark_results = {
            "sector": "Industrials",
            "comparisons": {},
            "score": 0.0,
            "strengths": [],
            "weaknesses": []
        }

        # Compare key manufacturing metrics
        key_metrics = [
            "ROIC",
            "Operating Margin",
            "Asset Turnover",
            "Inventory Turnover",
            "CAPEX/Sales",
            "Cash Conversion Cycle",
            "Fixed Asset Turnover",
            "Debt/EBITDA"
        ]

        comparison_scores = []

        for metric in key_metrics:
            if metric in company_metrics and metric in benchmarks:
                company_value = company_metrics[metric]
                benchmark_value = benchmarks[metric]

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ["Debt/EBITDA", "Cash Conversion Cycle"]:
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

    def _calculate_manufacturing_metrics(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                         cash_flow: pd.Series) -> None:
        """"
        Calculate manufacturing-specific metrics

        Args:
            result: Dictionary to store calculation results
            income: Income statement data
            balance: Balance sheet data
            cash_flow: Cash flow statement data
        """
        metrics = {}

        # Asset Utilization Metrics
        if 'Total Revenue' in income and 'Property Plant and Equipment' in balance:
            fixed_asset_turnover = income['Total Revenue'] / balance['Property Plant and Equipment']
            metrics['Fixed Asset Turnover'] = fixed_asset_turnover

        # Operating Efficiency Metrics
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            metrics['Operating Margin'] = operating_margin

        # Capital Intensity
        if 'Total Revenue' in income and 'Property Plant and Equipment' in balance:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            metrics['Capital Intensity'] = capital_intensity

        # CAPEX Ratio
        if cash_flow is not None and 'Capital Expenditure' in cash_flow and 'Total Revenue' in income:
            capex_to_sales = abs(cash_flow['Capital Expenditure']) / income['Total Revenue']
            metrics['CAPEX/Sales'] = capex_to_sales

        # Inventory Management
        if 'Cost of Revenue' in income and 'Inventory' in balance:
            inventory_turnover = income['Cost of Revenue'] / balance['Inventory']
            metrics['Inventory Turnover'] = inventory_turnover

            # Days Inventory Outstanding
            if inventory_turnover > 0:
                metrics['Days Inventory Outstanding'] = 365 / inventory_turnover

        # Labor Productivity (if available)
        if 'Total Revenue' in income:
            # Note: Number of employees typically comes from company info, not financial statements
            # This is a placeholder for when that data is available
            metrics['Revenue per Employee'] = None

        # ROIC (Return on Invested Capital)
        if 'Operating Income' in income:
            # Calculate invested capital (equity + debt - cash)
            invested_capital = balance.get('Total Stockholder Equity', 0)
            if 'Total Debt' in balance:
                invested_capital += balance['Total Debt']
            if 'Cash and Cash Equivalents' in balance:
                invested_capital -= balance['Cash and Cash Equivalents']

            if invested_capital > 0:
                # Assume 25% tax rate if not available
                tax_rate = 0.25
                if 'Income Tax Expense' in income and 'Income Before Tax' in income and income['Income Before Tax'] > 0:
                    tax_rate = income['Income Tax Expense'] / income['Income Before Tax']

                nopat = income['Operating Income'] * (1 - tax_rate)
                roic = nopat / invested_capital
                metrics['ROIC'] = roic

        result["industry_metrics"] = metrics

    def _analyze_operational_efficiency(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series) -> None:
        """Analyze operational efficiency specific to manufacturing companies"""
        operational = {}

        if income is None or balance is None:
            result["operational_efficiency"] = operational
            return

        # Calculate efficiency ratios
        if 'Gross Profit' in income and 'Total Revenue' in income:
            operational['Gross Margin'] = income['Gross Profit'] / income['Total Revenue']

            # Classify manufacturing efficiency
            if operational['Gross Margin'] >= 0.40:
                operational['Efficiency Assessment'] = "High efficiency manufacturing operation"
            elif operational['Gross Margin'] >= 0.25:
                operational['Efficiency Assessment'] = "Average efficiency manufacturing operation"
            else:
                operational['Efficiency Assessment'] = "Low efficiency manufacturing operation"

        # Asset efficiency
        if 'Total Revenue' in income and 'Total Assets' in balance:
            operational['Asset Turnover'] = income['Total Revenue'] / balance['Total Assets']

            # Plant utilization efficiency (if data available)
            if 'Property Plant and Equipment' in balance:
                operational['PP&E Turnover'] = income['Total Revenue'] / balance['Property Plant and Equipment']

        # Working capital efficiency
        if 'Inventory' in balance and 'Net Receivables' in balance and 'Accounts Payable' in balance:
            current_assets = balance.get('Inventory', 0) + balance.get('Net Receivables', 0)
            current_liabilities = balance.get('Accounts Payable', 0)

            if current_liabilities > 0:
                operational['Working Capital Ratio'] = current_assets / current_liabilities

        result["operational_efficiency"] = operational

    def _identify_risk_factors(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Identify key industry risk factors for manufacturing companies"""
        risks = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Check for high capital intensity
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            if capital_intensity > 0.5:
                risks.append({
                    "factor": "High Capital Intensity",
                    "description": "Significant investment in fixed assets relative to revenue indicates high operational leverage and potential risk during downturns",
                    "severity": "Medium"
                })

        # Check for low inventory turnover
        if 'industry_metrics' in result and 'Inventory Turnover' in result['industry_metrics']:
            inventory_turnover = result['industry_metrics']['Inventory Turnover']
            if inventory_turnover < 4:
                risks.append({
                    "factor": "Low Inventory Turnover",
                    "description": "Slow-moving inventory may indicate operational inefficiencies or product obsolescence risk",
                    "severity": "Medium"
                })

        # Check for high debt levels
        if 'Total Debt' in balance and 'Total Stockholder Equity' in balance:
            debt_to_equity = balance['Total Debt'] / balance['Total Stockholder Equity']
            if debt_to_equity > 1.0:
                risks.append({
                    "factor": "High Debt Leverage",
                    "description": "Elevated debt levels increase financial risk and interest expense burden",
                    "severity": "High" if debt_to_equity > 2.0 else "Medium"
                })

        # Check for thin operating margins
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            if operating_margin < 0.08:
                risks.append({
                    "factor": "Thin Operating Margins",
                    "description": "Low operating margins provide little buffer against cost increases or pricing pressure",
                    "severity": "High" if operating_margin < 0.05 else "Medium"
                })

        # Check for high fixed costs (if data available)
        if 'Gross Profit' in income and 'Operating Income' in income and 'Total Revenue' in income:
            gross_margin = income['Gross Profit'] / income['Total Revenue']
            operating_margin = income['Operating Income'] / income['Total Revenue']

            # High difference between gross and operating margin suggests high fixed costs
            if (gross_margin - operating_margin) > 0.25:
                risks.append({
                    "factor": "High Fixed Cost Structure",
                    "description": "Large gap between gross and operating margins indicates high fixed costs and operational leverage",
                    "severity": "Medium"
                })

        result["risk_factors"] = risks

    def _evaluate_competitive_advantages(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Evaluate potential competitive advantages for manufacturing companies"""
        advantages = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Check for superior operational efficiency
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            if operating_margin > 0.15:  # Above average for manufacturing
                advantages.append({
                    "advantage": "Superior Operational Efficiency",
                    "description": "Above-average operating margins indicate efficient operations and potential cost advantages",
                    "strength": "High" if operating_margin > 0.20 else "Medium"
                })

        # Check for efficient asset utilization
        if 'industry_metrics' in result and 'Fixed Asset Turnover' in result['industry_metrics']:
            fixed_asset_turnover = result['industry_metrics']['Fixed Asset Turnover']
            if fixed_asset_turnover > 3.0:  # Above average for manufacturing
                advantages.append({
                    "advantage": "Efficient Asset Utilization",
                    "description": "High fixed asset turnover indicates efficient use of manufacturing facilities and equipment",
                    "strength": "High" if fixed_asset_turnover > 5.0 else "Medium"
                })

        # Check for strong ROIC
        if 'industry_metrics' in result and 'ROIC' in result['industry_metrics']:
            roic = result['industry_metrics']['ROIC']
            if roic > 0.12:  # Above average for manufacturing
                advantages.append({
                    "advantage": "Superior Capital Returns",
                    "description": "Above-average return on invested capital suggests competitive advantages in capital deployment",
                    "strength": "High" if roic > 0.18 else "Medium"
                })

        # Check for low capital intensity (asset-light model)
        if 'industry_metrics' in result and 'Capital Intensity' in result['industry_metrics']:
            capital_intensity = result['industry_metrics']['Capital Intensity']
            if capital_intensity < 0.3:  # Below average for manufacturing
                advantages.append({
                    "advantage": "Asset-Light Model",
                    "description": "Lower capital intensity relative to industry suggests efficient operations with less capital requirements",
                    "strength": "Medium"
                })

        result["competitive_advantages"] = advantages

    def _calculate_ev_ebitda_valuation(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                       market_data: Dict[str, Any]) -> None:
        """Calculate EV/EBITDA-based valuation (common for manufacturing)"""
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
        sector_multiple = 9.0  # Typical for industrials
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

    def _calculate_roic_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate ROIC-based valuation"""
        valuation = {}

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Skip if data is missing
        if income_stmt is None or balance_sheet is None:
            valuation["error"] = "Missing required financial statements"
            result["methods"]["ROIC"] = valuation
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Calculate ROIC
        roic = None
        if 'Operating Income' in income:
            # Calculate invested capital
            invested_capital = balance.get('Total Stockholder Equity', 0)
            if 'Total Debt' in balance:
                invested_capital += balance['Total Debt']
            if 'Cash and Cash Equivalents' in balance:
                invested_capital -= balance['Cash and Cash Equivalents']

            if invested_capital > 0:
                # Assume 25% tax rate if not available
                tax_rate = 0.25
                if 'Income Tax Expense' in income and 'Income Before Tax' in income and income['Income Before Tax'] > 0:
                    tax_rate = income['Income Tax Expense'] / income['Income Before Tax']

                nopat = income['Operating Income'] * (1 - tax_rate)
                roic = nopat / invested_capital

        # If we can't calculate ROIC, skip this valuation method
        if roic is None or roic <= 0:
            valuation["error"] = "Unable to calculate ROIC or ROIC is negative"
            result["methods"]["ROIC"] = valuation
            return

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Get WACC (weighted average cost of capital)
        wacc = 0.09  # Default for industrials sector if not provided

        # Calculate value using ROIC-WACC spread
        if roic > wacc:
            # Company is creating value
            value_premium = (roic - wacc) / wacc

            # Apply premium to book value
            if 'Total Stockholder Equity' in balance:
                book_value = balance['Total Stockholder Equity']
                adjusted_equity_value = book_value * (1 + value_premium)

                # Calculate per share value
                shares_outstanding = market_data.get('shares_outstanding')
                if shares_outstanding and shares_outstanding > 0:
                    value_per_share = adjusted_equity_value / shares_outstanding
                else:
                    value_per_share = None

                # Store valuation result
                valuation["book_value"] = book_value
                valuation["value_premium"] = value_premium
                valuation["adjusted_equity_value"] = adjusted_equity_value
                valuation["value_per_share"] = value_per_share
                valuation["roic"] = roic
                valuation["wacc"] = wacc
            else:
                valuation["error"] = "Missing stockholder equity data"
        else:
            # Company is destroying value
            value_discount = (wacc - roic) / wacc

            # Apply discount to book value
            if 'Total Stockholder Equity' in balance:
                book_value = balance['Total Stockholder Equity']
                adjusted_equity_value = book_value * (1 - value_discount)

                # Calculate per share value
                shares_outstanding = market_data.get('shares_outstanding')
                if shares_outstanding and shares_outstanding > 0:
                    value_per_share = adjusted_equity_value / shares_outstanding
                else:
                    value_per_share = None

                # Store valuation result
                valuation["book_value"] = book_value
                valuation["value_discount"] = value_discount
                valuation["adjusted_equity_value"] = adjusted_equity_value
                valuation["value_per_share"] = value_per_share
                valuation["roic"] = roic
                valuation["wacc"] = wacc
            else:
                valuation["error"] = "Missing stockholder equity data"

        result["methods"]["ROIC"] = valuation

    def _calculate_asset_based_valuation(self, result: Dict[str, Any], balance: pd.Series,
                                         market_data: Dict[str, Any]) -> None:
        """Calculate asset-based valuation (useful for asset-heavy manufacturing)"""
        valuation = {}

        # Check if we have balance sheet data
        if balance is None:
            valuation["error"] = "Missing balance sheet data"
            result["methods"]["Asset"] = valuation
            return

        # Calculate adjusted asset values
        # For manufacturing companies, certain assets may need to be revalued

        # 1. Property, Plant & Equipment (often understated at historical cost)
        if 'Property Plant and Equipment' in balance:
            ppe = balance['Property Plant and Equipment']
            # Adjust for potential undervaluation of real property assets (typically 15-30% higher than book)
            adjusted_ppe = ppe * 1.2  # 20% premium to book value
        else:
            ppe = 0
            adjusted_ppe = 0

        # 2. Inventory (may be understated or overstated)
        if 'Inventory' in balance:
            inventory = balance['Inventory']
            # For manufacturing, typically discount inventory (risk of obsolescence)
            adjusted_inventory = inventory * 0.9  # 10% discount to book value
        else:
            inventory = 0
            adjusted_inventory = 0

        # 3. Intangibles (often understated in manufacturing)
        if 'Intangible Assets' in balance:
            intangibles = balance['Intangible Assets']
            # No adjustment for intangibles
            adjusted_intangibles = intangibles
        else:
            intangibles = 0
            adjusted_intangibles = 0

        # 4. Get other assets
        other_assets = balance.get('Total Assets', 0) - ppe - inventory - intangibles

        # Calculate total adjusted assets
        adjusted_assets = adjusted_ppe + adjusted_inventory + adjusted_intangibles + other_assets

        # Calculate total liabilities
        total_liabilities = balance.get('Total Liabilities Net Minority Interest', 0)

        # Calculate adjusted book value
        adjusted_book_value = adjusted_assets - total_liabilities

        # Calculate per share value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            value_per_share = adjusted_book_value / shares_outstanding
        else:
            value_per_share = None

        # Store valuation result
        valuation["book_value"] = balance.get('Total Stockholder Equity', 0)
        valuation["adjusted_assets"] = adjusted_assets
        valuation["total_liabilities"] = total_liabilities
        valuation["adjusted_book_value"] = adjusted_book_value
        valuation["value_per_share"] = value_per_share
        valuation["asset_adjustments"] = {
            "ppe": {
                "original": ppe,
                "adjusted": adjusted_ppe,
                "adjustment_factor": 1.2
            },
            "inventory": {
                "original": inventory,
                "adjusted": adjusted_inventory,
                "adjustment_factor": 0.9
            },
            "intangibles": {
                "original": intangibles,
                "adjusted": adjusted_intangibles,
                "adjustment_factor": 1.0
            }
        }

        result["methods"]["Asset"] = valuation

    def _calculate_dcf_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate DCF with manufacturing-specific adjustments"""
        valuation = {}

        # This would typically call a DCF model implementation with sector-specific parameters
        # For brevity, we'll note what would be different for manufacturing companies

        valuation["note"] = "Manufacturing DCF would typically include:"
        valuation["adjustments"] = [
            "Higher capital expenditure requirements",
            "Adjustments for cyclicality in cash flows",
            "Emphasis on maintenance vs. growth capex",
            "Sector-specific discount rates (typically 9-11% for industrials)",
            "Focus on operational metrics and asset utilization"
        ]

        result["methods"]["DCF"] = valuation


# Register with factory
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory

SectorFactory.register_analyzer("Industrials", ManufacturingSectorAnalyzer)