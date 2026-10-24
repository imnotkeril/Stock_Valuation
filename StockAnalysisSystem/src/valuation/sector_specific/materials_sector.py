import os
import sys
import logging
import pandas as pd
import numpy as np
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
logger = logging.getLogger('materials_sector')


class MaterialsSectorAnalyzer(SectorAnalyzer):
    """
    Specialized analyzer for materials sector companies.

    This sector includes metals and mining, chemicals, forestry products,
    construction materials, packaging, and other materials businesses.
    """

    def __init__(self):
        """Initialize materials sector analyzer"""
        super().__init__("Materials")
        self.key_ratios = self.get_key_ratios()
        self.preferred_valuation_methods = self.get_preferred_valuation_methods()
        logger.info(f"Initialized Materials sector analyzer with {len(self.key_ratios)} key ratios")

    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform materials-specific analysis on company data

        Args:
            financial_data: Dictionary containing financial statements and other company data

        Returns:
            Dictionary with analysis results specific to materials companies
        """
        result = {
            "sector": "Materials",
            "industry_metrics": {},
            "sector_specific_analysis": {},
            "risk_factors": [],
            "competitive_advantages": [],
            "cyclicality_analysis": {},
            "commodity_exposure": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Skip analysis if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for materials sector analysis")
            return result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None
        latest_cash_flow = cash_flow.iloc[:, 0] if cash_flow is not None and not cash_flow.empty else None

        # Calculate materials-specific metrics
        self._calculate_materials_metrics(result, latest_income, latest_balance, latest_cash_flow)

        # Analyze cyclicality
        self._analyze_cyclicality(result, financial_data)

        # Analyze commodity exposure
        self._analyze_commodity_exposure(result, financial_data)

        # Identify key industry risk factors
        self._identify_risk_factors(result, financial_data)

        # Evaluate competitive advantages
        self._evaluate_competitive_advantages(result, financial_data)

        return result

    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using materials-specific methods

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
            "sector": "Materials",
            "methods": {}
        }

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if income_stmt is None or balance_sheet is None:
            logger.warning("Missing essential financial statements for materials sector valuation")
            return valuation_result

        # Get most recent financial data
        latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else None
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else None

        # Perform EV/EBITDA valuation
        if valuation_method in ["EV/EBITDA", "all"]:
            self._calculate_ev_ebitda_valuation(valuation_result, latest_income, latest_balance, market_data)

        # Perform P/B valuation (common for materials companies)
        if valuation_method in ["P/B", "all"]:
            self._calculate_pb_valuation(valuation_result, latest_balance, market_data)

        # Perform NAV (Net Asset Value) valuation
        if valuation_method in ["NAV", "all"]:
            self._calculate_nav_valuation(valuation_result, financial_data)

        # Perform cyclical-adjusted valuation (for materials companies)
        if valuation_method in ["Cyclical", "all"]:
            self._calculate_cyclical_valuation(valuation_result, financial_data)

        return valuation_result

    def _calculate_materials_metrics(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                     cash_flow: pd.Series) -> None:
        """Calculate materials-specific metrics"""
        metrics = {}

        # Margin Metrics
        if 'Gross Profit' in income and 'Total Revenue' in income:
            gross_margin = income['Gross Profit'] / income['Total Revenue']
            metrics['Gross Margin'] = gross_margin

        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            metrics['Operating Margin'] = operating_margin

        # EBITDA Margin - key for materials sector
        if 'EBITDA' in income and 'Total Revenue' in income:
            ebitda_margin = income['EBITDA'] / income['Total Revenue']
            metrics['EBITDA Margin'] = ebitda_margin
        elif 'Operating Income' in income and 'Depreciation & Amortization' in income and 'Total Revenue' in income:
            ebitda = income['Operating Income'] + income['Depreciation & Amortization']
            ebitda_margin = ebitda / income['Total Revenue']
            metrics['EBITDA Margin'] = ebitda_margin

        # Asset Utilization
        if 'Total Revenue' in income and 'Total Assets' in balance:
            asset_turnover = income['Total Revenue'] / balance['Total Assets']
            metrics['Asset Turnover'] = asset_turnover

        # Capital Intensity
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            metrics['Capital Intensity'] = capital_intensity

        # Production Metrics
        # Note: Typically would come from company reports, not financial statements
        metrics['Production Volume'] = None
        metrics['Production Cost per Unit'] = None

        # Inventory Management
        if 'Cost of Revenue' in income and 'Inventory' in balance:
            inventory_turnover = income['Cost of Revenue'] / balance['Inventory']
            metrics['Inventory Turnover'] = inventory_turnover

            if inventory_turnover > 0:
                metrics['Days Inventory Outstanding'] = 365 / inventory_turnover

        # CAPEX Intensity
        if cash_flow is not None and 'Capital Expenditure' in cash_flow and 'Total Revenue' in income:
            capex_to_revenue = abs(cash_flow['Capital Expenditure']) / income['Total Revenue']
            metrics['CAPEX/Revenue'] = capex_to_revenue

            # CAPEX to Depreciation ratio
            if 'Depreciation & Amortization' in income and income['Depreciation & Amortization'] > 0:
                capex_to_depreciation = abs(cash_flow['Capital Expenditure']) / income['Depreciation & Amortization']
                metrics['CAPEX/Depreciation'] = capex_to_depreciation

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

        result["industry_metrics"] = metrics

    def _analyze_cyclicality(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze cyclicality of materials company performance"""
        cyclicality = {}

        # Extract financial statements for historical analysis
        income_stmt = financial_data.get('income_statement')

        # Need at least 3 years of data for cyclicality analysis
        if income_stmt is None or income_stmt.shape[1] < 3:
            cyclicality["note"] = "Insufficient historical data for cyclicality analysis"
            result["cyclicality_analysis"] = cyclicality
            return

        # Calculate margin volatility
        margin_series = []

        # Check for various margin metrics across years
        for i in range(min(5, income_stmt.shape[1])):
            year_data = income_stmt.iloc[:, i]
            if 'Operating Income' in year_data.index and 'Total Revenue' in year_data.index:
                if year_data['Total Revenue'] > 0:
                    margin = year_data['Operating Income'] / year_data['Total Revenue']
                    margin_series.append(margin)

        # Calculate volatility if we have sufficient data
        if len(margin_series) >= 3:
            margin_std = pd.Series(margin_series).std()
            margin_mean = pd.Series(margin_series).mean()

            if margin_mean > 0:
                # Coefficient of variation as a measure of volatility
                cyclicality["margin_volatility"] = margin_std / margin_mean

                # Classify cyclicality
                if cyclicality["margin_volatility"] > 0.5:
                    cyclicality["cyclicality_assessment"] = "Highly cyclical business"
                elif cyclicality["margin_volatility"] > 0.3:
                    cyclicality["cyclicality_assessment"] = "Moderately cyclical business"
                else:
                    cyclicality["cyclicality_assessment"] = "Relatively stable business within materials sector"

        # Calculate revenue volatility if available
        revenue_series = []

        for i in range(min(5, income_stmt.shape[1]) - 1):
            current_revenue = income_stmt.iloc[:, i]['Total Revenue']
            prev_revenue = income_stmt.iloc[:, i + 1]['Total Revenue']

            if prev_revenue > 0:
                annual_growth = (current_revenue / prev_revenue) - 1
                revenue_series.append(annual_growth)

        if len(revenue_series) >= 2:
            cyclicality["revenue_growth_volatility"] = pd.Series(revenue_series).std()

        result["cyclicality_analysis"] = cyclicality

    def _analyze_commodity_exposure(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Analyze commodity price exposure and impact"""
        # Note: In a production system, this would extract data from company reports
        # or management discussions about commodity exposure

        commodity_exposure = {
            "primary_commodities": None,  # e.g., "iron ore", "copper", "chemicals"
            "commodity_price_sensitivity": None,  # e.g., "high", "medium", "low"
            "margin_impact_per_10pct_price_move": None,  # e.g., "150 bps"
            "hedging_strategy": None,  # e.g., "partial", "none", "extensive"
            "vertical_integration": None  # e.g., "high", "moderate", "low"
        }

        result["commodity_exposure"] = commodity_exposure

    def _identify_risk_factors(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Identify key industry risk factors for materials companies"""
        risks = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Cyclicality risk - inherent in materials sector
        risks.append({
            "factor": "Commodity Price Volatility",
            "description": "Materials companies are exposed to fluctuations in commodity prices that impact margins",
            "severity": "High"
        })

        # Check for high capital intensity
        if 'Property Plant and Equipment' in balance and 'Total Revenue' in income:
            capital_intensity = balance['Property Plant and Equipment'] / income['Total Revenue']
            if capital_intensity > 0.7:
                risks.append({
                    "factor": "High Capital Intensity",
                    "description": "Significant investment in fixed assets creates high operational leverage and maintenance capex requirements",
                    "severity": "Medium"
                })

        # Check for high debt levels
        if 'Total Debt' in balance and 'EBITDA' in income:
            if income['EBITDA'] > 0:
                debt_to_ebitda = balance['Total Debt'] / income['EBITDA']
                if debt_to_ebitda > 3.0:
                    risks.append({
                        "factor": "High Debt Burden",
                        "description": "Elevated debt levels increase financial risk during industry downturns",
                        "severity": "High" if debt_to_ebitda > 4.0 else "Medium"
                    })

        # Check for low operating margin
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            if operating_margin < 0.10:  # Materials typically need higher margins to cover high capex
                risks.append({
                    "factor": "Thin Operating Margins",
                    "description": "Low operating margins provide limited buffer against commodity price decreases",
                    "severity": "High" if operating_margin < 0.05 else "Medium"
                })

        # Environmental regulatory risk
        risks.append({
            "factor": "Environmental Regulatory Pressure",
            "description": "Materials sector faces increasing environmental regulations, compliance costs, and potential liabilities",
            "severity": "Medium"
        })

        # Check for high CAPEX requirements
        if cash_flow is not None and 'Capital Expenditure' in cash_flow and 'Depreciation & Amortization' in income:
            if income['Depreciation & Amortization'] > 0:
                capex_to_depreciation = abs(cash_flow['Capital Expenditure']) / income['Depreciation & Amortization']
                if capex_to_depreciation > 1.3:
                    risks.append({
                        "factor": "High Capital Expenditure Requirements",
                        "description": "Maintenance and growth capital expenditures significantly exceed depreciation",
                        "severity": "Medium"
                    })

        result["risk_factors"] = risks

    def _evaluate_competitive_advantages(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Evaluate potential competitive advantages for materials companies"""
        advantages = []

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if income_stmt is None or balance_sheet is None:
            return

        # Get most recent financial data
        income = income_stmt.iloc[:, 0]
        balance = balance_sheet.iloc[:, 0]

        # Check for cost leadership (indicated by strong margins)
        if 'Operating Income' in income and 'Total Revenue' in income:
            operating_margin = income['Operating Income'] / income['Total Revenue']
            if operating_margin > 0.15:  # Above average for materials
                advantages.append({
                    "advantage": "Cost Leadership",
                    "description": "Above-average operating margins suggest cost advantages in production or operations",
                    "strength": "High" if operating_margin > 0.25 else "Medium"
                })

        # Check for efficient asset utilization
        if 'industry_metrics' in result and 'Asset Turnover' in result['industry_metrics']:
            asset_turnover = result['industry_metrics']['Asset Turnover']
            if asset_turnover > 0.8:  # Above average for materials sector
                advantages.append({
                    "advantage": "Efficient Asset Utilization",
                    "description": "Above-average asset turnover indicates efficient use of capital-intensive assets",
                    "strength": "High" if asset_turnover > 1.0 else "Medium"
                })

        # Check for strong ROIC
        if 'industry_metrics' in result and 'ROIC' in result['industry_metrics']:
            roic = result['industry_metrics']['ROIC']
            if roic > 0.10:  # Above average for materials
                advantages.append({
                    "advantage": "Superior Capital Returns",
                    "description": "Above-average return on invested capital suggests competitive advantages",
                    "strength": "High" if roic > 0.15 else "Medium"
                })

        # Resource advantages (would typically come from company reports)
        advantages.append({
            "advantage": "Resource Position",
            "description": "Materials companies often derive advantage from resource quality, quantity, and access",
            "strength": "Medium"
        })

        # Scale advantages
        if 'Total Revenue' in income and income['Total Revenue'] > 5000000000:  # $5B+ revenue
            advantages.append({
                "advantage": "Scale Economies",
                "description": "Large scale operations can provide cost advantages in materials processing",
                "strength": "Medium"
            })

        result["competitive_advantages"] = advantages

    def _calculate_ev_ebitda_valuation(self, result: Dict[str, Any], income: pd.Series, balance: pd.Series,
                                       market_data: Dict[str, Any]) -> None:
        """Calculate EV/EBITDA-based valuation (common for materials)"""
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
        sector_multiple = 8.0  # Typical for materials
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

    def _calculate_pb_valuation(self, result: Dict[str, Any], balance: pd.Series, market_data: Dict[str, Any]) -> None:
        """Calculate P/B-based valuation (common for materials)"""
        valuation = {}

        # Get book value
        book_value = None
        if 'Total Stockholder Equity' in balance:
            book_value = balance['Total Stockholder Equity']

        # If we can't calculate book value, skip this valuation method
        if book_value is None or book_value <= 0:
            valuation["error"] = "Unable to calculate book value or book value is negative"
            result["methods"]["P/B"] = valuation
            return

        # Get sector P/B multiple
        sector_multiple = 2.0  # Typical for materials
        if 'sector_metrics' in self.sector_metrics and 'benchmarks' in self.sector_metrics['sector_metrics']:
            if 'P/B' in self.sector_metrics['sector_metrics']['benchmarks']:
                sector_multiple = self.sector_metrics['sector_metrics']['benchmarks']['P/B']

        # Calculate equity value
        equity_value = book_value * sector_multiple

        # Calculate per share value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            value_per_share = equity_value / shares_outstanding
        else:
            value_per_share = None

        # Store valuation result
        valuation["equity_value"] = equity_value
        valuation["value_per_share"] = value_per_share
        valuation["multiple_used"] = sector_multiple
        valuation["book_value"] = book_value

        result["methods"]["P/B"] = valuation


    def _calculate_nav_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """Calculate NAV-based valuation (Net Asset Value)"""
        valuation = {}

        # Extract financial statements
        balance_sheet = financial_data.get('balance_sheet')

        # Get market data
        market_data = financial_data.get('market_data', {})

        # Skip valuation if data is missing
        if balance_sheet is None:
            valuation["error"] = "Missing balance sheet for NAV valuation"
            result["methods"]["NAV"] = valuation
            return

        # Get most recent financial data
        balance = balance_sheet.iloc[:, 0]

        # Asset adjustments based on materials sector characteristics
        # These would typically be derived from detailed company disclosures

        # 1. Property, Plant & Equipment
        if 'Property Plant and Equipment' in balance:
            ppe = balance['Property Plant and Equipment']
            # For materials companies, PPE might be understated (especially resource reserves)
            adjusted_ppe = ppe * 1.2  # 20% premium
        else:
            ppe = 0
            adjusted_ppe = 0

        # 2. Inventory
        if 'Inventory' in balance:
            inventory = balance['Inventory']
            # For materials, may need to adjust based on commodity price cycles
            adjusted_inventory = inventory * 1.0  # No adjustment in base case
        else:
            inventory = 0
            adjusted_inventory = 0

        # 3. Resource reserves (not typically on balance sheet)
        # Would typically come from company reports and third-party assessments
        resource_value = 0  # Placeholder

        # 4. Get other assets
        other_assets = balance.get('Total Assets', 0) - ppe - inventory

        # Calculate total adjusted assets
        adjusted_assets = adjusted_ppe + adjusted_inventory + resource_value + other_assets

        # Calculate total liabilities
        total_liabilities = balance.get('Total Liabilities Net Minority Interest', 0)

        # Calculate adjusted net asset value
        adjusted_nav = adjusted_assets - total_liabilities

        # Calculate per share value
        shares_outstanding = market_data.get('shares_outstanding')
        if shares_outstanding and shares_outstanding > 0:
            value_per_share = adjusted_nav / shares_outstanding
        else:
            value_per_share = None

        # Store valuation result
        valuation["book_value"] = balance.get('Total Stockholder Equity', 0)
        valuation["adjusted_assets"] = adjusted_assets
        valuation["resource_value"] = resource_value
        valuation["total_liabilities"] = total_liabilities
        valuation["adjusted_nav"] = adjusted_nav
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
                "adjustment_factor": 1.0
            }
        }

        result["methods"]["NAV"] = valuation

    def _calculate_cyclical_valuation(self, result: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """
        Specialized cyclical valuation for materials sector companies

        Args:
            result: Dictionary to store valuation results
            financial_data: Dictionary with company financial data
        """
        valuation = {}

        # Extract financial statements
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')
        cash_flow = financial_data.get('cash_flow')

        # Validate data availability
        if income_stmt is None or income_stmt.shape[1] < 3:
            valuation["error"] = "Insufficient data for cyclical valuation"
            result["methods"]["Cyclical"] = valuation
            return

        # Calculate EBITDA values across cycle
        ebitda_values = []
        for i in range(min(5, income_stmt.shape[1])):
            year_income = income_stmt.iloc[:, i]

            # Calculate EBITDA
            if 'EBITDA' in year_income:
                ebitda_values.append(year_income['EBITDA'])
            elif 'Operating Income' in year_income and 'Depreciation & Amortization' in year_income:
                ebitda = year_income['Operating Income'] + year_income['Depreciation & Amortization']
                ebitda_values.append(ebitda)

        # Validate EBITDA calculation
        if not ebitda_values:
            valuation["error"] = "Unable to calculate EBITDA"
            result["methods"]["Cyclical"] = valuation
            return

        # Calculate mid-cycle EBITDA
        cycle_ebitda = np.mean(ebitda_values)
        ebitda_volatility = np.std(ebitda_values) / cycle_ebitda if cycle_ebitda > 0 else 0

        # Determine multiple with cyclicality adjustment
        base_multiple = 8.0  # Base multiple for materials sector
        volatility_adjustment = 1.0
        if ebitda_volatility > 0:
            volatility_adjustment = 1.0 + (ebitda_volatility * 2)  # Adjustment for volatility
        sector_multiple = base_multiple / volatility_adjustment  # Higher volatility should reduce multiple

        # Calculate enterprise value
        enterprise_value = cycle_ebitda * sector_multiple

        # Adjust value based on balance sheet
        balance = balance_sheet.iloc[:, 0]
        total_debt = balance.get('Total Debt', 0)
        cash = balance.get('Cash and Cash Equivalents', 0)
        equity_value = enterprise_value - total_debt + cash

        # Calculate share value
        market_data = financial_data.get('market_data', {})
        shares_outstanding = market_data.get('shares_outstanding')
        value_per_share = equity_value / shares_outstanding if shares_outstanding and shares_outstanding > 0 else None

        # Prepare valuation results
        valuation = {
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "value_per_share": value_per_share,
            "mid_cycle_ebitda": cycle_ebitda,
            "ebitda_volatility": ebitda_volatility,
            "multiple_used": sector_multiple,
            "historical_ebitda": ebitda_values,
            "cyclicality_factors": {
                "base_multiple": base_multiple,
                "volatility_adjustment": volatility_adjustment
            }
        }

        # Scenario analysis
        valuation["scenario_analysis"] = {
            "base_scenario": value_per_share,
            "optimistic_scenario": value_per_share * 1.2 if value_per_share else None,
            "pessimistic_scenario": value_per_share * 0.8 if value_per_share else None
        }

        # Add results to overall result
        result["methods"]["Cyclical"] = valuation

    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        benchmarks = self.get_sector_benchmarks()
        benchmark_results = {
            "sector": "Materials",
            "comparisons": {},
            "score": 0.0,
            "strengths": [],
            "weaknesses": []
        }

        # Compare key materials metrics
        key_metrics = [
            "ROIC",
            "Operating Margin",
            "EBITDA Margin",
            "Cash Conversion Cycle",
            "Gross Margin",
            "Asset Turnover",
            "Debt/EBITDA",
            "EV/EBITDA"
        ]

        comparison_scores = []

        for metric in key_metrics:
            if metric in company_metrics and metric in benchmarks:
                company_value = company_metrics[metric]
                benchmark_value = benchmarks[metric]

                # Determine if higher is better for this metric
                higher_is_better = True
                if metric in ["Debt/EBITDA", "Cash Conversion Cycle", "EV/EBITDA"]:
                    higher_is_better = False

                # Calculate percent difference from benchmark
                if benchmark_value is not None and benchmark_value != 0:
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

# Register with factory
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory
SectorFactory.register_analyzer("Materials", MaterialsSectorAnalyzer)