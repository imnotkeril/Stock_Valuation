import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from utils.data_loader import DataLoader
from valuation.base_valuation import BaseValuation
from valuation.dcf_models import AdvancedDCFValuation, SectorSpecificDCF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tech_sector')


class TechnologySectorValuation(SectorSpecificDCF):
    """
    Specialized valuation models for technology sector companies

    Technology sector valuation requires different approaches due to:
    1. High growth rates with longer runway
    2. Higher R&D spending that should be capitalized rather than expensed
    3. Different metrics of success (users, subscribers, etc.)
    4. Network effects and platform economics
    5. Intangible asset-heavy business models
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize technology sector valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized TechnologySectorValuation")

        # Specific parameters for technology sector
        self.rd_capitalization_rate = 0.7  # Percentage of R&D to capitalize (vs expense)
        self.rd_amortization_period = 5  # Years to amortize capitalized R&D
        self.min_terminal_growth = 0.03  # Higher minimum terminal growth for tech
        self.default_revenue_growth_fadeout = 15  # Years for growth to normalize

    def value_tech_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master method to value a technology sector company, selecting the most appropriate model
        based on company sub-sector and maturity
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine tech sub-sector and maturity
            sub_sector, maturity = self._determine_tech_profile(ticker, financial_data)

            # Select valuation method based on sub-sector and maturity
            if sub_sector == "software" and maturity == "growth":
                result = self.value_saas_company(ticker, financial_data)
            elif sub_sector == "hardware":
                result = self.value_hardware_company(ticker, financial_data)
            elif sub_sector == "semiconductor":
                result = self.value_semiconductor_company(ticker, financial_data)
            elif sub_sector == "internet" and maturity == "growth":
                result = self.value_internet_company(ticker, financial_data)
            elif maturity == "startup":
                result = self.value_tech_startup(ticker, financial_data)
            else:
                # Use general approach for technology companies
                result = self.tech_sector_dcf(ticker, financial_data)

            # Add sub-sector and maturity information to result
            result['sub_sector'] = sub_sector
            result['maturity'] = maturity

            return result

        except Exception as e:
            logger.error(f"Error valuing technology company {ticker}: {e}")
            # Fall back to standard approach
            return self.tech_sector_dcf(ticker, financial_data)

    def value_saas_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a SaaS (Software as a Service) company using specialized models including:
        1. Revenue-based DCF with growth fade model
        2. Rule of 40 valuation and other SaaS metrics
        3. Customer cohort analysis (if data available)
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate key SaaS metrics
            saas_metrics = self._calculate_saas_metrics(financial_data)

            # 1. Perform revenue-based DCF with growth fade model
            dcf_result = self._saas_revenue_dcf(ticker, financial_data, saas_metrics)

            # 2. Perform rule of 40 valuation
            rule40_result = self._rule_of_40_valuation(ticker, financial_data, saas_metrics)

            # 3. Attempt cohort analysis if data available
            cohort_result = self._cohort_based_valuation(ticker, financial_data, saas_metrics)

            # 4. Calculate valuation using SaaS multiples
            multiples_result = self._saas_multiples_valuation(ticker, financial_data, saas_metrics)

            # Blend the valuation approaches based on available data and confidence
            valuation_weights = {
                'dcf': 0.4,  # Base weight for DCF
                'rule40': 0.2,  # Base weight for Rule of 40
                'cohort': 0.2,  # Base weight for cohort analysis
                'multiples': 0.2  # Base weight for multiples
            }

            # Adjust weights if any method failed or has low confidence
            if cohort_result.get('value_per_share') is None:
                # Redistribute cohort weight to other methods
                valuation_weights['dcf'] += valuation_weights['cohort'] * 0.5
                valuation_weights['rule40'] += valuation_weights['cohort'] * 0.25
                valuation_weights['multiples'] += valuation_weights['cohort'] * 0.25
                valuation_weights['cohort'] = 0

            # Calculate weighted average if possible
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'saas_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'dcf_valuation': dcf_result,
                'rule_of_40_valuation': rule40_result,
                'cohort_valuation': cohort_result,
                'multiples_valuation': multiples_result,
                'saas_metrics': saas_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in SaaS company valuation for {ticker}: {e}")
            # Fall back to general technology sector DCF
            return self.tech_sector_dcf(ticker, financial_data)

    def value_hardware_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a hardware technology company using specialized models including:
        1. Traditional DCF with cyclical adjustments
        2. EV/EBITDA valuation with hardware-specific multiples
        3. Adjusted ROA/ROIC analysis
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate hardware metrics
            hardware_metrics = self._calculate_hardware_metrics(financial_data)

            # 1. Perform DCF with cyclical adjustments
            dcf_result = self._cyclical_hardware_dcf(ticker, financial_data, hardware_metrics)

            # 2. Perform EV/EBITDA valuation with hardware-specific multiples
            evebitda_result = self._hardware_evebitda_valuation(ticker, financial_data, hardware_metrics)

            # 3. Perform ROA/ROIC analysis
            roic_result = self._hardware_roic_valuation(ticker, financial_data, hardware_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'dcf': 0.4,
                'evebitda': 0.4,
                'roic': 0.2
            }

            # Calculate weighted average
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'hardware_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'dcf_valuation': dcf_result,
                'evebitda_valuation': evebitda_result,
                'roic_valuation': roic_result,
                'hardware_metrics': hardware_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in hardware company valuation for {ticker}: {e}")
            # Fall back to general technology sector DCF
            return self.tech_sector_dcf(ticker, financial_data)

    def value_semiconductor_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a semiconductor company using specialized models including:
        1. Cyclical-adjusted DCF
        2. Industry-specific multiples
        3. IP/R&D based valuation
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate semiconductor metrics
            semi_metrics = self._calculate_semiconductor_metrics(financial_data)

            # 1. Perform cyclical-adjusted DCF
            dcf_result = self._cyclical_semiconductor_dcf(ticker, financial_data, semi_metrics)

            # 2. Perform industry-specific multiples valuation
            multiples_result = self._semiconductor_multiples_valuation(ticker, financial_data, semi_metrics)

            # 3. Perform IP/R&D based valuation
            ip_result = self._semiconductor_ip_valuation(ticker, financial_data, semi_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'dcf': 0.4,
                'multiples': 0.4,
                'ip': 0.2
            }

            # Calculate weighted average
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'semiconductor_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'dcf_valuation': dcf_result,
                'multiples_valuation': multiples_result,
                'ip_valuation': ip_result,
                'semiconductor_metrics': semi_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in semiconductor company valuation for {ticker}: {e}")
            # Fall back to general technology sector DCF
            return self.tech_sector_dcf(ticker, financial_data)

    def value_internet_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an internet company (platforms, marketplaces, social media) using specialized models including:
        1. User-based valuation (ARPU, CAC, LTV)
        2. Platform economics DCF
        3. Networking effects premium
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate internet metrics
            internet_metrics = self._calculate_internet_metrics(financial_data)

            # 1. Perform user-based valuation
            user_result = self._user_based_valuation(ticker, financial_data, internet_metrics)

            # 2. Perform platform economics DCF
            platform_result = self._platform_economics_dcf(ticker, financial_data, internet_metrics)

            # 3. Perform multiples valuation with network effects premium
            multiples_result = self._internet_multiples_valuation(ticker, financial_data, internet_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'user': 0.3,
                'platform': 0.4,
                'multiples': 0.3
            }

            # Adjust weights if user metrics aren't available
            if user_result.get('value_per_share') is None:
                valuation_weights['platform'] += valuation_weights['user'] * 0.6
                valuation_weights['multiples'] += valuation_weights['user'] * 0.4
                valuation_weights['user'] = 0

            # Calculate weighted average
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.8  # 20% margin of safety (higher for internet companies)
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'internet_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'user_valuation': user_result,
                'platform_valuation': platform_result,
                'multiples_valuation': multiples_result,
                'internet_metrics': internet_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in internet company valuation for {ticker}: {e}")
            # Fall back to general technology sector DCF
            return self.tech_sector_dcf(ticker, financial_data)

    def value_tech_startup(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an early-stage tech company using specialized models including:
        1. Venture capital method (exit multiple)
        2. Comparable transactions
        3. Growth-adjusted multiples
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Calculate startup metrics
            startup_metrics = self._calculate_startup_metrics(financial_data)

            # 1. Perform venture capital method valuation
            vc_result = self._venture_capital_valuation(ticker, financial_data, startup_metrics)

            # 2. Perform comparable transactions valuation
            transactions_result = self._comparable_transactions_valuation(ticker, financial_data, startup_metrics)

            # 3. Perform growth-adjusted multiples valuation
            multiples_result = self._startup_multiples_valuation(ticker, financial_data, startup_metrics)

            # Blend the valuation approaches
            valuation_weights = {
                'vc': 0.4,
                'transactions': 0.3,
                'multiples': 0.3
            }

            # Calculate weighted average
            total_value = 0
            total_weight = 0

            for method, weight in valuation_weights.items():
                result_var = locals()[f"{method}_result"]
                if result_var.get('value_per_share') is not None and weight > 0:
                    total_value += result_var['value_per_share'] * weight
                    total_weight += weight

            if total_weight > 0:
                blended_value = total_value / total_weight
            else:
                blended_value = None

            # Apply margin of safety (higher for startups due to uncertainty)
            if blended_value:
                conservative_value = blended_value * 0.7  # 30% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'tech_startup_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'vc_valuation': vc_result,
                'transactions_valuation': transactions_result,
                'multiples_valuation': multiples_result,
                'startup_metrics': startup_metrics,
                'weights': {k: v for k, v in valuation_weights.items() if v > 0}
            }

        except Exception as e:
            logger.error(f"Error in tech startup valuation for {ticker}: {e}")
            # Fall back to revenue multiple approach for startups
            return self._startup_multiples_valuation(ticker, financial_data)

    # Helper methods for tech sector valuation

    def _determine_tech_profile(self, ticker: str, financial_data: Dict[str, Any]) -> Tuple[str, str]:
        """Determine the sub-sector and maturity stage of a technology company"""
        try:
            # Extract company info and financial statements
            company_info = financial_data.get('company_info', {})
            income_stmt = financial_data.get('income_statement')

            # Try to get industry from company info
            industry = company_info.get('industry', '').lower()
            sector = company_info.get('sector', '').lower()

            # Default classifications
            sub_sector = "software"  # Default to software
            maturity = "growth"  # Default to growth stage

            # Determine sub-sector
            if any(term in industry for term in ['software', 'saas', 'cloud']):
                sub_sector = "software"
            elif any(term in industry for term in ['hardware', 'computer', 'device', 'electronics']):
                sub_sector = "hardware"
            elif any(term in industry for term in ['semiconductor', 'chip', 'integrated circuit']):
                sub_sector = "semiconductor"
            elif any(term in industry for term in ['internet', 'e-commerce', 'social media', 'online']):
                sub_sector = "internet"

            # Determine maturity based on financial metrics
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]

                # Check revenue
                if 'Total Revenue' in latest_income.index:
                    revenue = latest_income['Total Revenue']

                    # Check profitability
                    net_income = None
                    if 'Net Income' in latest_income.index:
                        net_income = latest_income['Net Income']

                    # Classify based on revenue and profitability
                    if revenue < 100000000:  # Less than $100M
                        maturity = "startup"
                    elif revenue < 1000000000:  # Less than $1B
                        if net_income is not None and net_income < 0:
                            maturity = "growth"
                        else:
                            maturity = "expansion"
                    else:  # $1B+
                        if net_income is not None and net_income > 0:
                            maturity = "mature"
                        else:
                            maturity = "expansion"

            return sub_sector, maturity

        except Exception as e:
            logger.warning(f"Error determining tech profile for {ticker}: {e}")
            return "software", "growth"  # Default values

    def _calculate_saas_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key SaaS metrics for valuation"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            metrics = {}

            if income_stmt is None:
                return metrics

            # Get data for multiple periods to calculate trends
            periods = min(income_stmt.shape[1], 3)  # Use up to 3 years of data

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]

                period_metrics = {}

                # Revenue
                revenue = None
                if 'Total Revenue' in income.index:
                    revenue = income['Total Revenue']
                    period_metrics['Revenue'] = revenue

                # Gross Profit and Margin
                if 'Gross Profit' in income.index and revenue:
                    gross_profit = income['Gross Profit']
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / revenue if revenue > 0 else None
                elif 'Cost of Revenue' in income.index and revenue:
                    cost_of_revenue = income['Cost of Revenue']
                    gross_profit = revenue - cost_of_revenue
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / revenue if revenue > 0 else None

                # Operating Income/Loss and Margin
                if 'Operating Income' in income.index and revenue:
                    operating_income = income['Operating Income']
                    period_metrics['Operating_Income'] = operating_income
                    period_metrics['Operating_Margin'] = operating_income / revenue if revenue > 0 else None

                # R&D Expense and as % of Revenue
                if 'Research and Development' in income.index and revenue:
                    rd_expense = income['Research and Development']
                    period_metrics['RD_Expense'] = rd_expense
                    period_metrics['RD_to_Revenue'] = rd_expense / revenue if revenue > 0 else None

                # Sales & Marketing Expense and as % of Revenue
                if 'Sales and Marketing' in income.index and revenue:
                    sm_expense = income['Sales and Marketing']
                    period_metrics['SM_Expense'] = sm_expense
                    period_metrics['SM_to_Revenue'] = sm_expense / revenue if revenue > 0 else None

                # Calculate Rule of 40 score (Growth + Profitability)
                # For the current period, we can only calculate profitability
                if revenue and 'Operating_Margin' in period_metrics:
                    # We'll calculate growth when comparing periods
                    rule_of_40 = period_metrics['Operating_Margin'] * 100  # Convert to percentage
                    period_metrics['Rule_of_40'] = rule_of_40

                metrics[year] = period_metrics

            # Calculate growth rates and update Rule of 40
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                if 'Revenue' in metrics["Latest"] and 'Revenue' in metrics["Year-1"] and metrics["Year-1"][
                    'Revenue'] > 0:
                    revenue_growth = (metrics["Latest"]['Revenue'] / metrics["Year-1"]['Revenue']) - 1
                    metrics["Latest"]['Revenue_Growth'] = revenue_growth

                    # Update Rule of 40 with growth component
                    if 'Rule_of_40' in metrics["Latest"]:
                        metrics["Latest"]['Rule_of_40'] = (revenue_growth * 100) + metrics["Latest"]['Rule_of_40']

            # Calculate additional SaaS-specific metrics if cash flow data is available
            if cash_flow is not None and not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]

                # Calculate Customer Acquisition Cost (CAC) if we have marketing expense and new customers
                # This is a rough approximation as we don't have customer count data
                if 'Latest' in metrics and 'SM_Expense' in metrics['Latest']:
                    # We can only estimate CAC - in a real scenario, you'd need customer acquisition data
                    # Assume marketing expense is primarily for customer acquisition
                    metrics['CAC_Estimate'] = metrics['Latest']['SM_Expense']

                # Calculate Burn Rate
                if 'Operating Cash Flow' in latest_cash_flow.index:
                    operating_cash_flow = latest_cash_flow['Operating Cash Flow']
                    if operating_cash_flow < 0:
                        metrics['Monthly_Burn_Rate'] = abs(operating_cash_flow) / 12

                        # Calculate Runway
                        if balance_sheet is not None and not balance_sheet.empty:
                            latest_balance = balance_sheet.iloc[:, 0]
                            if 'Cash and Cash Equivalents' in latest_balance.index:
                                cash = latest_balance['Cash and Cash Equivalents']
                                if metrics['Monthly_Burn_Rate'] > 0:
                                    metrics['Runway_Months'] = cash / metrics['Monthly_Burn_Rate']

            return metrics

        except Exception as e:
            logger.error(f"Error calculating SaaS metrics: {e}")
            return {}

    def _saas_revenue_dcf(self, ticker: str, financial_data: Dict[str, Any],
                          saas_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a revenue-based DCF valuation specifically for SaaS companies
        with growth fade model and margin expansion
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest income data
            latest_income = income_stmt.iloc[:, 0]

            # Get latest saas metrics
            latest_metrics = saas_metrics.get('Latest', {})

            # 1. Determine starting revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'saas_revenue_dcf',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # 2. Determine revenue growth rate
            # Initial growth rate from historical data or use sector average
            initial_growth_rate = latest_metrics.get('Revenue_Growth')

            if initial_growth_rate is None:
                # Try to calculate from historical data
                if income_stmt.shape[1] >= 2:
                    prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                 1].index else None
                    if prev_revenue and prev_revenue > 0:
                        initial_growth_rate = (revenue / prev_revenue) - 1
                    else:
                        # Use default based on SaaS sector
                        initial_growth_rate = 0.25  # 25% default growth for SaaS
                else:
                    initial_growth_rate = 0.25  # Default if no historical data

            # Cap growth rate to reasonable bounds
            initial_growth_rate = min(max(initial_growth_rate, 0.10), 0.50)  # Between 10% and 50%

            # 3. Determine gross margin and operating margin
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin is None:
                if 'Gross Profit' in latest_income.index and revenue > 0:
                    gross_margin = latest_income['Gross Profit'] / revenue
                else:
                    # Default gross margin for SaaS
                    gross_margin = 0.70  # 70% is typical for SaaS

            operating_margin = latest_metrics.get('Operating_Margin')
            if operating_margin is None:
                if 'Operating Income' in latest_income.index and revenue > 0:
                    operating_margin = latest_income['Operating Income'] / revenue
                else:
                    # Default operating margin for SaaS (often negative for growth stage)
                    operating_margin = -0.10  # -10% is common for growth SaaS

            # 4. Set up DCF parameters
            forecast_years = 10  # Longer forecast for SaaS
            terminal_growth = 0.03  # Long-term growth rate
            target_operating_margin = 0.25  # Target mature SaaS operating margin
            margin_expansion_years = 7  # Years to reach target margin

            # 5. Calculate discount rate (WACC)
            beta = market_data.get('beta', 1.4)  # Default beta for SaaS
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            # Add size premium and company-specific risk
            size_premium = 0.02  # SaaS companies often have higher risk
            company_risk = 0.01  # Additional company-specific risk

            cost_of_equity = risk_free_rate + beta * equity_risk_premium + size_premium + company_risk

            # Simplified WACC calculation
            wacc = cost_of_equity  # For most
            # SaaS companies equity heavy
            # Ensure WACC is reasonable
            wacc = max(0.10, min(0.20, wacc))  # Between 10% and 20%

            # 6. Forecast revenue and cash flows
            forecasted_revenues = []
            forecasted_cash_flows = []

            current_revenue = revenue
            current_growth_rate = initial_growth_rate
            current_operating_margin = operating_margin

            for year in range(1, forecast_years + 1):
                # Revenue growth fades over time
                growth_fade = (initial_growth_rate - terminal_growth) * (year / forecast_years)
                year_growth_rate = initial_growth_rate - growth_fade

                # Revenue for the year
                current_revenue = current_revenue * (1 + year_growth_rate)
                forecasted_revenues.append(current_revenue)

                # Operating margin expands over time (if starting below target)
                if operating_margin < target_operating_margin:
                    if year <= margin_expansion_years:
                        margin_improvement = (target_operating_margin - operating_margin) * (
                                    year / margin_expansion_years)
                        current_operating_margin = operating_margin + margin_improvement
                    else:
                        current_operating_margin = target_operating_margin

                # Operating income
                operating_income = current_revenue * current_operating_margin

                # Taxes (effective rate)
                effective_tax_rate = 0.25  # Typical corporate tax rate
                taxes = max(0, operating_income * effective_tax_rate)  # Only tax if profitable

                # Capital expenditures (typically low for SaaS)
                capex = current_revenue * 0.05  # 5% of revenue

                # Changes in working capital
                working_capital_change = current_revenue * 0.02 * year_growth_rate  # Tied to growth

                # Free cash flow
                fcf = operating_income - taxes - capex - working_capital_change
                forecasted_cash_flows.append(fcf)

            # 7. Calculate terminal value
            final_fcf = forecasted_cash_flows[-1]
            terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

            # 8. Calculate present value of cash flows and terminal value
            present_value_fcf = sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            enterprise_value = present_value_fcf + present_value_terminal

            # 9. Adjust for net cash/debt
            net_cash = 0
            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # 10. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'saas_revenue_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'initial_growth_rate': initial_growth_rate,
                'terminal_growth': terminal_growth,
                'gross_margin': gross_margin,
                'initial_operating_margin': operating_margin,
                'target_operating_margin': target_operating_margin,
                'wacc': wacc,
                'forecast_years': forecast_years,
                'forecasted_revenues': forecasted_revenues,
                'forecasted_cash_flows': forecasted_cash_flows,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'terminal_value': terminal_value,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in SaaS revenue DCF for {ticker}: {e}")
            return {
                'method': 'saas_revenue_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _rule_of_40_valuation(self, ticker: str, financial_data: Dict[str, Any],
                              saas_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a SaaS company using the Rule of 40 framework,
        which states that a healthy SaaS company's growth rate plus profitability
        should exceed 40%
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]

            # Get Rule of 40 score
            latest_metrics = saas_metrics.get('Latest', {})
            rule_of_40_score = latest_metrics.get('Rule_of_40')

            # If Rule of 40 score not available, calculate it
            if rule_of_40_score is None:
                # Get growth rate
                revenue_growth = latest_metrics.get('Revenue_Growth')
                if revenue_growth is None:
                    if income_stmt.shape[1] >= 2:
                        current_revenue = latest_income[
                            'Total Revenue'] if 'Total Revenue' in latest_income.index else None
                        prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                     1].index else None
                        if current_revenue and prev_revenue and prev_revenue > 0:
                            revenue_growth = (current_revenue / prev_revenue) - 1
                        else:
                            revenue_growth = 0.25  # Default 25% growth
                    else:
                        revenue_growth = 0.25  # Default if no historical data

                # Get operating margin
                operating_margin = latest_metrics.get('Operating_Margin')
                if operating_margin is None:
                    if 'Operating Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                        if latest_income['Total Revenue'] > 0:
                            operating_margin = latest_income['Operating Income'] / latest_income['Total Revenue']
                        else:
                            operating_margin = -0.10  # Default -10%
                    else:
                        operating_margin = -0.10  # Default if not available

                # Calculate Rule of 40 score
                rule_of_40_score = (revenue_growth * 100) + (operating_margin * 100)  # As percentages

            # Get current revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'rule_of_40',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Define EV/Revenue multiples based on Rule of 40 score
            # These multiples are based on observed market data for SaaS companies
            if rule_of_40_score >= 70:
                ev_revenue_multiple = 15.0  # Premium multiple for exceptional performance
            elif rule_of_40_score >= 40:
                ev_revenue_multiple = 10.0  # Solid multiple for companies meeting Rule of 40
            elif rule_of_40_score >= 20:
                ev_revenue_multiple = 5.0  # Moderate multiple for decent performers
            else:
                ev_revenue_multiple = 3.0  # Lower multiple for underperformers

            # Calculate enterprise value
            enterprise_value = revenue * ev_revenue_multiple

            # Adjust for net cash/debt
            net_cash = 0
            balance_sheet = financial_data.get('balance_sheet')

            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'rule_of_40',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'rule_of_40_score': rule_of_40_score,
                'ev_revenue_multiple': ev_revenue_multiple,
                'revenue': revenue,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in Rule of 40 valuation for {ticker}: {e}")
            return {
                'method': 'rule_of_40',
                'value_per_share': None,
                'error': str(e)
            }

    def _cohort_based_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                saas_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a SaaS company using cohort analysis focusing on customer lifetime value (LTV)
        relative to customer acquisition cost (CAC)

        Note: This is a simplified approach since detailed cohort data is typically not
        available in standard financial statements
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]

            # Get latest metrics
            latest_metrics = saas_metrics.get('Latest', {})

            # For this analysis, we need:
            # 1. Revenue per customer (ARPU)
            # 2. Gross margin
            # 3. Customer retention/churn rate
            # 4. Customer acquisition cost (CAC)

            # In a real implementation, this data would come from company's disclosures
            # Here we'll make estimates based on available financial data

            # Get revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'cohort_based',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Estimate number of customers
            # This would typically come from company disclosures
            # For this example, we'll make a rough estimate
            estimated_customers = 10000  # Default assumption

            # Calculate ARPU (Average Revenue Per User)
            arpu = revenue / estimated_customers

            # Get gross margin
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin is None:
                if 'Gross Profit' in latest_income.index and revenue > 0:
                    gross_margin = latest_income['Gross Profit'] / revenue
                else:
                    gross_margin = 0.70  # Default for SaaS

            # Estimate churn rate
            # Again, this would come from company disclosures
            annual_churn_rate = 0.15  # Default assumption (15% annual churn)

            # Calculate customer lifetime (in years)
            customer_lifetime = 1 / annual_churn_rate

            # Calculate LTV (Lifetime Value)
            ltv = (arpu * gross_margin * customer_lifetime)

            # Get CAC (Customer Acquisition Cost)
            # Either from metrics or estimate
            cac = saas_metrics.get('CAC_Estimate')
            if cac is None:
                # Estimate CAC from Sales & Marketing expense
                if 'SM_Expense' in latest_metrics:
                    # Assume 70% of S&M is for acquisition and rest for retention
                    acquisition_expense = latest_metrics['SM_Expense'] * 0.7
                    # Assume 20% of estimated customers are new
                    new_customers = estimated_customers * 0.2
                    if new_customers > 0:
                        cac = acquisition_expense / new_customers
                    else:
                        cac = arpu * 1.2  # Default: 1.2x ARPU
                else:
                    cac = arpu * 1.2  # Default: 1.2x ARPU

            # Calculate LTV:CAC ratio
            ltv_cac_ratio = ltv / cac if cac > 0 else 0

            # Determine enterprise value based on LTV and future customer acquisition
            # Estimate future customer growth
            customer_growth_rate = 0.20  # Default assumption

            # Forecast customer acquisition for 5 years
            total_customers = estimated_customers
            cumulative_value = 0

            for year in range(1, 6):
                new_customers = total_customers * customer_growth_rate
                customer_value = (ltv - cac) * new_customers

                # Discount to present value
                discount_rate = 0.15  # Higher discount for cohort model
                present_value = customer_value / ((1 + discount_rate) ** year)

                cumulative_value += present_value
                total_customers += new_customers

            # Add value of existing customers
            existing_customer_value = ltv * estimated_customers * 0.7  # Discount existing base

            enterprise_value = cumulative_value + existing_customer_value

            # Adjust for net cash/debt
            net_cash = 0
            balance_sheet = financial_data.get('balance_sheet')

            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'cohort_based',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'ltv': ltv,
                'cac': cac,
                'ltv_cac_ratio': ltv_cac_ratio,
                'arpu': arpu,
                'gross_margin': gross_margin,
                'annual_churn_rate': annual_churn_rate,
                'customer_lifetime': customer_lifetime,
                'estimated_customers': estimated_customers,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in cohort-based valuation for {ticker}: {e}")
            return {
                'method': 'cohort_based',
                'value_per_share': None,
                'error': str(e)
            }

    def _saas_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                  saas_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a SaaS company using sector-specific multiples
        like EV/Revenue, EV/ARR, etc.
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]

            # Get latest metrics if available
            latest_metrics = {} if saas_metrics is None else saas_metrics.get('Latest', {})

            # Get revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'saas_multiples',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Define base multiples for SaaS companies
            # These values typically come from market comparables
            base_multiples = {
                'EV_Revenue': 8.0,  # Enterprise Value / Revenue
                'EV_Growth_Adjusted': 0.7  # EV/Revenue divided by growth rate
            }

            # Determine growth rate
            growth_rate = latest_metrics.get('Revenue_Growth')
            if growth_rate is None:
                if income_stmt.shape[1] >= 2:
                    prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                 1].index else None
                    if prev_revenue and prev_revenue > 0:
                        growth_rate = (revenue / prev_revenue) - 1
                    else:
                        growth_rate = 0.25  # Default growth rate
                else:
                    growth_rate = 0.25  # Default if no historical data

            # Determine gross margin
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin is None:
                if 'Gross Profit' in latest_income.index and revenue > 0:
                    gross_margin = latest_income['Gross Profit'] / revenue
                else:
                    gross_margin = 0.70  # Default for SaaS

            # Adjust multiples based on growth and margins
            # Higher growth and margins justify higher multiples

            # Growth adjustment
            if growth_rate >= 0.40:  # 40%+ growth
                growth_multiple_adjustment = 1.5
            elif growth_rate >= 0.20:  # 20-40% growth
                growth_multiple_adjustment = 1.0
            else:  # <20% growth
                growth_multiple_adjustment = 0.7

            # Margin adjustment
            if gross_margin >= 0.80:  # 80%+ gross margin
                margin_multiple_adjustment = 1.3
            elif gross_margin >= 0.70:  # 70-80% gross margin
                margin_multiple_adjustment = 1.0
            else:  # <70% gross margin
                margin_multiple_adjustment = 0.8

            # Apply adjustments to base multiples
            adjusted_multiples = {
                'EV_Revenue': base_multiples['EV_Revenue'] * growth_multiple_adjustment * margin_multiple_adjustment
            }

            # Calculate growth-adjusted multiple
            if growth_rate > 0:
                adjusted_multiples['EV_Growth_Adjusted'] = adjusted_multiples['EV_Revenue'] / growth_rate
            else:
                adjusted_multiples['EV_Growth_Adjusted'] = adjusted_multiples['EV_Revenue']

            # Calculate valuations using different multiples
            valuations = {}

            # EV/Revenue valuation
            ev_revenue = revenue * adjusted_multiples['EV_Revenue']
            valuations['EV_Revenue'] = {
                'multiple': adjusted_multiples['EV_Revenue'],
                'enterprise_value': ev_revenue
            }

            # Determine final enterprise value (using EV/Revenue)
            enterprise_value = ev_revenue

            # Adjust for net cash/debt
            net_cash = 0
            balance_sheet = financial_data.get('balance_sheet')

            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'saas_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'growth_rate': growth_rate,
                'gross_margin': gross_margin,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'valuations': valuations,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in SaaS multiples valuation for {ticker}: {e}")
            return {
                'method': 'saas_multiples',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_hardware_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key hardware technology metrics for valuation"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get data for multiple periods to calculate trends
            periods = min(income_stmt.shape[1], balance_sheet.shape[1], 3)  # Use up to 3 years

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]
                balance = balance_sheet.iloc[:, i]

                period_metrics = {}

                # Revenue and growth
                if 'Total Revenue' in income.index:
                    revenue = income['Total Revenue']
                    period_metrics['Revenue'] = revenue

                # Gross Profit and Margin
                if 'Gross Profit' in income.index and 'Revenue' in period_metrics:
                    gross_profit = income['Gross Profit']
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']
                elif 'Cost of Revenue' in income.index and 'Revenue' in period_metrics:
                    cost_of_revenue = income['Cost of Revenue']
                    gross_profit = period_metrics['Revenue'] - cost_of_revenue
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']

                # Operating Income and Margin
                if 'Operating Income' in income.index and 'Revenue' in period_metrics:
                    operating_income = income['Operating Income']
                    period_metrics['Operating_Income'] = operating_income
                    period_metrics['Operating_Margin'] = operating_income / period_metrics['Revenue']

                # R&D as % of Revenue
                if 'Research and Development' in income.index and 'Revenue' in period_metrics:
                    rd_expense = income['Research and Development']
                    period_metrics['RD_Expense'] = rd_expense
                    period_metrics['RD_to_Revenue'] = rd_expense / period_metrics['Revenue']

                # Asset utilization
                if 'Total Assets' in balance.index and 'Revenue' in period_metrics:
                    total_assets = balance['Total Assets']
                    period_metrics['Asset_Turnover'] = period_metrics['Revenue'] / total_assets

                # Fixed asset utilization
                if 'Property Plant and Equipment' in balance.index and 'Revenue' in period_metrics:
                    ppe = balance['Property Plant and Equipment']
                    period_metrics['Fixed_Asset_Turnover'] = period_metrics['Revenue'] / ppe if ppe > 0 else None

                # Inventory metrics
                if 'Inventory' in balance.index:
                    inventory = balance['Inventory']
                    period_metrics['Inventory'] = inventory

                    if 'Cost of Revenue' in income.index:
                        cogs = income['Cost of Revenue']
                        period_metrics['Inventory_Turnover'] = cogs / inventory if inventory > 0 else None
                        if period_metrics['Inventory_Turnover']:
                            period_metrics['Days_Inventory'] = 365 / period_metrics['Inventory_Turnover']

                # Return on assets
                if 'Net Income' in income.index and 'Total Assets' in balance.index:
                    net_income = income['Net Income']
                    total_assets = balance['Total Assets']
                    period_metrics['ROA'] = net_income / total_assets

                # Capital intensity
                if 'Capital Expenditure' in income.index and 'Revenue' in period_metrics:
                    capex = abs(income['Capital Expenditure'])  # Usually negative in cash flow
                    period_metrics['CapEx_to_Revenue'] = capex / period_metrics['Revenue']

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                # Revenue growth
                if 'Revenue' in metrics["Latest"] and 'Revenue' in metrics["Year-1"] and metrics["Year-1"][
                    'Revenue'] > 0:
                    revenue_growth = (metrics["Latest"]['Revenue'] / metrics["Year-1"]['Revenue']) - 1
                    growth_metrics['Revenue_Growth'] = revenue_growth

                # Margin trends
                for margin in ['Gross_Margin', 'Operating_Margin']:
                    if margin in metrics["Latest"] and margin in metrics["Year-1"]:
                        margin_change = metrics["Latest"][margin] - metrics["Year-1"][margin]
                        growth_metrics[f'{margin}_Change'] = margin_change

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating hardware metrics: {e}")
            return {}

    def _cyclical_hardware_dcf(self, ticker: str, financial_data: Dict[str, Any],
                               hardware_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform DCF valuation for hardware technology companies
        adjusting for cyclical nature of hardware business
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = hardware_metrics.get('Latest', {})
            growth_metrics = hardware_metrics.get('Growth', {})

            # 1. Determine starting revenue and growth rate
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'cyclical_hardware_dcf',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Determine revenue growth rate
            growth_rate = growth_metrics.get('Revenue_Growth')
            if growth_rate is None:
                # Try to calculate from historical data
                if income_stmt.shape[1] >= 2:
                    prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                 1].index else None
                    if prev_revenue and prev_revenue > 0:
                        growth_rate = (revenue / prev_revenue) - 1
                    else:
                        # Use default based on hardware sector
                        growth_rate = 0.08  # 8% default growth for hardware
                else:
                    growth_rate = 0.08  # Default if no historical data

            # 2. Determine margins
            operating_margin = latest_metrics.get('Operating_Margin')
            if operating_margin is None:
                if 'Operating Income' in latest_income.index and revenue > 0:
                    operating_margin = latest_income['Operating Income'] / revenue
                else:
                    # Default operating margin for hardware
                    operating_margin = 0.12  # 12% is typical for hardware

            # 3. Set up DCF parameters
            forecast_years = 8  # Standard forecast for hardware

            # Hardware companies often have cyclical patterns
            # Here we model a basic cycle with 3 years up, 2 years down
            cycle_pattern = [1.1, 1.05, 1.02, 0.98, 0.95]  # Multipliers for cyclical adjustment

            # Terminal growth rate - lower for hardware than software
            terminal_growth = 0.02

            # 4. Calculate discount rate (WACC)
            beta = market_data.get('beta', 1.2)  # Default beta for hardware tech
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # Hardware companies often have more debt than software
            debt_ratio = 0
            if 'Total Debt' in latest_balance.index and 'Total Assets' in latest_balance.index:
                debt_ratio = latest_balance['Total Debt'] / latest_balance['Total Assets']

            cost_of_debt = 0.04  # Typical cost of debt
            tax_rate = 0.25  # Typical tax rate

            # WACC calculation
            if debt_ratio > 0:
                equity_ratio = 1 - debt_ratio
                wacc = (cost_of_equity * equity_ratio) + (cost_of_debt * (1 - tax_rate) * debt_ratio)
            else:
                wacc = cost_of_equity

            # Ensure WACC is reasonable
            wacc = max(0.08, min(0.15, wacc))  # Between 8% and 15%

            # 5. Forecast cash flows with cyclical pattern
            forecasted_cash_flows = []
            current_revenue = revenue

            for year in range(1, forecast_years + 1):
                # Apply cyclical adjustment
                cycle_index = (year - 1) % len(cycle_pattern)
                cyclical_factor = cycle_pattern[cycle_index]

                # Base growth rate adjusted for cycle
                year_growth = growth_rate * cyclical_factor

                # Revenue for the year
                current_revenue = current_revenue * (1 + year_growth)

                # Operating income with cyclical margin
                # Hardware margins compress in down cycles
                if cyclical_factor < 1:
                    year_margin = operating_margin * 0.9  # 10% margin compression in down cycle
                else:
                    year_margin = operating_margin * 1.05  # 5% margin expansion in up cycle

                operating_income = current_revenue * year_margin

                # Taxes
                taxes = operating_income * tax_rate

                # Capital expenditures (higher for hardware companies)
                capex_ratio = latest_metrics.get('CapEx_to_Revenue', 0.08)  # Default 8% of revenue
                capex = current_revenue * capex_ratio

                # Depreciation (estimate based on PPE)
                depreciation_rate = 0.1  # 10% annual depreciation
                if 'Property Plant and Equipment' in latest_balance.index:
                    ppe = latest_balance['Property Plant and Equipment']
                    depreciation = ppe * depreciation_rate
                else:
                    # Estimate depreciation as a percentage of capex
                    depreciation = capex * 0.8

                # Changes in working capital
                # Hardware companies have higher working capital needs
                working_capital_change = current_revenue * 0.03 * year_growth

                # Free cash flow
                fcf = operating_income - taxes + depreciation - capex - working_capital_change
                forecasted_cash_flows.append(fcf)

            # 6. Calculate terminal value
            final_fcf = forecasted_cash_flows[-1]
            terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

            # 7. Calculate present value of cash flows and terminal value
            present_value_fcf = sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            enterprise_value = present_value_fcf + present_value_terminal

            # 8. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 9. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'cyclical_hardware_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'growth_rate': growth_rate,
                'operating_margin': operating_margin,
                'terminal_growth': terminal_growth,
                'wacc': wacc,
                'cycle_pattern': cycle_pattern,
                'forecast_years': forecast_years,
                'forecasted_cash_flows': forecasted_cash_flows,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'terminal_value': terminal_value,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in cyclical hardware DCF for {ticker}: {e}")
            return {
                'method': 'cyclical_hardware_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _hardware_evebitda_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                     hardware_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a hardware technology company using EV/EBITDA multiples
        adjusted for hardware sector specifics
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = hardware_metrics.get('Latest', {})

            # 1. Calculate EBITDA
            ebitda = None

            # Try to find EBITDA directly
            if 'EBITDA' in latest_income.index:
                ebitda = latest_income['EBITDA']
            else:
                # Calculate from components
                operating_income = None
                if 'Operating Income' in latest_income.index:
                    operating_income = latest_income['Operating Income']
                elif 'EBIT' in latest_income.index:
                    operating_income = latest_income['EBIT']

                depreciation_amortization = None
                if 'Depreciation & Amortization' in latest_income.index:
                    depreciation_amortization = latest_income['Depreciation & Amortization']
                elif 'Depreciation and Amortization' in latest_income.index:
                    depreciation_amortization = latest_income['Depreciation and Amortization']

                # Calculate EBITDA if we have both components
                if operating_income is not None and depreciation_amortization is not None:
                    ebitda = operating_income + depreciation_amortization
                elif operating_income is not None:
                    # Estimate D&A if not available
                    if 'Property Plant and Equipment' in latest_balance.index:
                        ppe = latest_balance['Property Plant and Equipment']
                        estimated_da = ppe * 0.1  # Assume 10% depreciation rate
                        ebitda = operating_income + estimated_da
                    else:
                        # Just use operating income as fallback
                        ebitda = operating_income

            if ebitda is None:
                return {
                    'method': 'hardware_evebitda',
                    'value_per_share': None,
                    'error': 'Cannot determine EBITDA'
                }

            # 2. Determine appropriate EV/EBITDA multiple
            # Base multiple for hardware technology sector
            base_multiple = 10.0  # Starting point

            # Adjust for margins
            if 'Operating_Margin' in latest_metrics:
                op_margin = latest_metrics['Operating_Margin']
                if op_margin > 0.15:  # Above average margin
                    base_multiple += 2.0
                elif op_margin < 0.08:  # Below average margin
                    base_multiple -= 2.0

            # Adjust for growth
            if 'Revenue_Growth' in hardware_metrics.get('Growth', {}):
                growth = hardware_metrics['Growth']['Revenue_Growth']
                if growth > 0.15:  # High growth
                    base_multiple += 1.5
                elif growth < 0.05:  # Low growth
                    base_multiple -= 1.5

            # Adjust for asset utilization
            if 'Asset_Turnover' in latest_metrics:
                asset_turnover = latest_metrics['Asset_Turnover']
                if asset_turnover > 0.8:  # Efficient asset utilization
                    base_multiple += 1.0
                elif asset_turnover < 0.5:  # Inefficient asset utilization
                    base_multiple -= 1.0

            # Ensure multiple is within reasonable range
            evebitda_multiple = max(6.0, min(15.0, base_multiple))

            # 3. Calculate enterprise value
            enterprise_value = ebitda * evebitda_multiple

            # 4. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 5. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'hardware_evebitda',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'ebitda': ebitda,
                'evebitda_multiple': evebitda_multiple,
                'base_multiple': base_multiple,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in hardware EV/EBITDA valuation for {ticker}: {e}")
            return {
                'method': 'hardware_evebitda',
                'value_per_share': None,
                'error': str(e)
            }

    def _hardware_roic_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                 hardware_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a hardware technology company based on ROIC (Return on Invested Capital)
        and capital efficiency
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = hardware_metrics.get('Latest', {})

            # 1. Calculate NOPAT (Net Operating Profit After Tax)
            if 'Operating Income' in latest_income.index:
                operating_income = latest_income['Operating Income']
            elif 'EBIT' in latest_income.index:
                operating_income = latest_income['EBIT']
            else:
                return {
                    'method': 'hardware_roic',
                    'value_per_share': None,
                    'error': 'Cannot determine operating income'
                }

            # Apply tax rate
            tax_rate = 0.25  # Standard assumption
            if 'Income Tax Expense' in latest_income.index and 'Income Before Tax' in latest_income.index:
                if latest_income['Income Before Tax'] > 0:
                    tax_rate = latest_income['Income Tax Expense'] / latest_income['Income Before Tax']
                    tax_rate = max(0.15, min(0.35, tax_rate))  # Sanity check

            nopat = operating_income * (1 - tax_rate)

            # 2. Calculate Invested Capital
            invested_capital = 0

            # Fixed assets (PPE)
            if 'Property Plant and Equipment' in latest_balance.index:
                invested_capital += latest_balance['Property Plant and Equipment']

            # Working capital (excluding cash and short-term debt)
            current_assets = 0
            current_liabilities = 0

            if 'Total Current Assets' in latest_balance.index:
                current_assets = latest_balance['Total Current Assets']
                # Exclude cash
                if 'Cash and Cash Equivalents' in latest_balance.index:
                    current_assets -= latest_balance['Cash and Cash Equivalents']

            if 'Total Current Liabilities' in latest_balance.index:
                current_liabilities = latest_balance['Total Current Liabilities']
                # Exclude short-term debt
                if 'Short Term Debt' in latest_balance.index:
                    current_liabilities -= latest_balance['Short Term Debt']

            # Add working capital to invested capital
            working_capital = current_assets - current_liabilities
            invested_capital += working_capital

            # Add goodwill and intangibles
            if 'Goodwill' in latest_balance.index:
                invested_capital += latest_balance['Goodwill']

            if 'Intangible Assets' in latest_balance.index:
                invested_capital += latest_balance['Intangible Assets']

            # 3. Calculate ROIC
            if invested_capital > 0:
                roic = nopat / invested_capital
            else:
                return {
                    'method': 'hardware_roic',
                    'value_per_share': None,
                    'error': 'Cannot calculate invested capital'
                }

            # 4. Calculate value based on Economic Value Added (EVA) approach
            # First, we need the cost of capital (WACC)
            beta = market_data.get('beta', 1.2)  # Default beta for hardware tech
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # For hardware companies, include debt in WACC
            debt_ratio = 0
            if 'Total Debt' in latest_balance.index and 'Total Assets' in latest_balance.index:
                debt_ratio = latest_balance['Total Debt'] / latest_balance['Total Assets']

            cost_of_debt = 0.04  # Typical cost of debt

            # WACC calculation
            if debt_ratio > 0:
                equity_ratio = 1 - debt_ratio
                wacc = (cost_of_equity * equity_ratio) + (cost_of_debt * (1 - tax_rate) * debt_ratio)
            else:
                wacc = cost_of_equity

            # Ensure WACC is reasonable
            wacc = max(0.08, min(0.15, wacc))  # Between 8% and 15%

            # Calculate EVA (Economic Value Added)
            eva = nopat - (invested_capital * wacc)

            # Calculate sustainable growth rate
            if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                roe = latest_income['Net Income'] / latest_balance['Total Stockholder Equity']

                # Estimate retention rate
                retention_rate = 0.7  # Default assumption (70% retention)

                sustainable_growth = roe * retention_rate
                sustainable_growth = max(0.02, min(0.15, sustainable_growth))  # Sanity check
            else:
                sustainable_growth = 0.05  # Default if can't calculate

            # 5. Calculate terminal value using EVA method
            forecast_years = 8
            terminal_growth = 0.02  # Lower than growth rate

            # Forecast EVA for each year
            present_value_eva = 0
            current_eva = eva

            for year in range(1, forecast_years + 1):
                # EVA growth fades over time
                eva_growth = max(sustainable_growth - ((sustainable_growth - terminal_growth) * year / forecast_years),
                                 terminal_growth)

                current_eva *= (1 + eva_growth)
                present_value_eva += current_eva / ((1 + wacc) ** year)

            # Terminal value of EVA
            terminal_eva = current_eva * (1 + terminal_growth)
            terminal_value = terminal_eva / (wacc - terminal_growth)
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            # 6. Calculate total company value
            # Current invested capital + PV of future EVA
            enterprise_value = invested_capital + present_value_eva + present_value_terminal

            # Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'hardware_roic',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'nopat': nopat,
                'invested_capital': invested_capital,
                'roic': roic,
                'wacc': wacc,
                'eva': eva,
                'present_value_eva': present_value_eva,
                'present_value_terminal': present_value_terminal,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in hardware ROIC valuation for {ticker}: {e}")
            return {
                'method': 'hardware_roic',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_semiconductor_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key semiconductor metrics for valuation"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            metrics = {}

            if income_stmt is None or balance_sheet is None:
                return metrics

            # Get data for multiple periods to calculate trends
            periods = min(income_stmt.shape[1], balance_sheet.shape[1], 3)  # Use up to 3 years

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]
                balance = balance_sheet.iloc[:, i]

                period_metrics = {}

                # Revenue
                if 'Total Revenue' in income.index:
                    revenue = income['Total Revenue']
                    period_metrics['Revenue'] = revenue

                # Gross Profit and Margin
                if 'Gross Profit' in income.index and 'Revenue' in period_metrics:
                    gross_profit = income['Gross Profit']
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']
                elif 'Cost of Revenue' in income.index and 'Revenue' in period_metrics:
                    cost_of_revenue = income['Cost of Revenue']
                    gross_profit = period_metrics['Revenue'] - cost_of_revenue
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']

                # Operating Income and Margin
                if 'Operating Income' in income.index and 'Revenue' in period_metrics:
                    operating_income = income['Operating Income']
                    period_metrics['Operating_Income'] = operating_income
                    period_metrics['Operating_Margin'] = operating_income / period_metrics['Revenue']

                # R&D as % of Revenue - critical for semiconductor companies
                if 'Research and Development' in income.index and 'Revenue' in period_metrics:
                    rd_expense = income['Research and Development']
                    period_metrics['RD_Expense'] = rd_expense
                    period_metrics['RD_to_Revenue'] = rd_expense / period_metrics['Revenue']

                # Capital intensity - semiconductor is very capital intensive
                if 'Capital Expenditure' in income.index and 'Revenue' in period_metrics:
                    capex = abs(income['Capital Expenditure'])  # Usually negative in cash flow
                    period_metrics['CapEx'] = capex
                    period_metrics['CapEx_to_Revenue'] = capex / period_metrics['Revenue']

                # Inventory metrics - important for cyclical businesses
                if 'Inventory' in balance.index:
                    inventory = balance['Inventory']
                    period_metrics['Inventory'] = inventory

                    if 'Cost of Revenue' in income.index:
                        cogs = income['Cost of Revenue']
                        period_metrics['Inventory_Turnover'] = cogs / inventory if inventory > 0 else None
                        if period_metrics['Inventory_Turnover']:
                            period_metrics['Days_Inventory'] = 365 / period_metrics['Inventory_Turnover']

                # Intellectual property metrics - very important for semiconductors
                if 'Intangible Assets' in balance.index:
                    intangibles = balance['Intangible Assets']
                    period_metrics['Intangible_Assets'] = intangibles
                    if 'Revenue' in period_metrics:
                        period_metrics['Intangibles_to_Revenue'] = intangibles / period_metrics['Revenue']

                # Patents are not directly visible in financials but can be estimated
                # from intangibles and R&D
                if 'Intangible_Assets' in period_metrics and 'RD_Expense' in period_metrics:
                    estimated_ip_value = period_metrics['Intangible_Assets'] + (period_metrics['RD_Expense'] * 3)
                    period_metrics['Estimated_IP_Value'] = estimated_ip_value

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                # Revenue growth
                if 'Revenue' in metrics["Latest"] and 'Revenue' in metrics["Year-1"] and metrics["Year-1"][
                    'Revenue'] > 0:
                    revenue_growth = (metrics["Latest"]['Revenue'] / metrics["Year-1"]['Revenue']) - 1
                    growth_metrics['Revenue_Growth'] = revenue_growth

                # Margin trends
                for margin in ['Gross_Margin', 'Operating_Margin']:
                    if margin in metrics["Latest"] and margin in metrics["Year-1"]:
                        margin_change = metrics["Latest"][margin] - metrics["Year-1"][margin]
                        growth_metrics[f'{margin}_Change'] = margin_change

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating semiconductor metrics: {e}")
            return {}

    def _cyclical_semiconductor_dcf(self, ticker: str, financial_data: Dict[str, Any],
                                    semi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cyclical-adjusted DCF valuation for semiconductor companies
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = semi_metrics.get('Latest', {})
            growth_metrics = semi_metrics.get('Growth', {})

            # 1. Determine starting revenue and growth rate
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'cyclical_semiconductor_dcf',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Determine revenue growth rate
            growth_rate = growth_metrics.get('Revenue_Growth')
            if growth_rate is None:
                # Try to calculate from historical data
                if income_stmt.shape[1] >= 2:
                    prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                 1].index else None
                    if prev_revenue and prev_revenue > 0:
                        growth_rate = (revenue / prev_revenue) - 1
                    else:
                        # Use default based on semiconductor sector
                        growth_rate = 0.10  # 10% default growth for semiconductors
                else:
                    growth_rate = 0.10  # Default if no historical data

            # 2. Determine operating margin
            operating_margin = latest_metrics.get('Operating_Margin')
            if operating_margin is None:
                if 'Operating Income' in latest_income.index and revenue > 0:
                    operating_margin = latest_income['Operating Income'] / revenue
                else:
                    # Default operating margin for semiconductors
                    operating_margin = 0.20  # 20% is typical for semiconductors

            # 3. Set up DCF parameters
            forecast_years = 8  # Longer forecast for cyclical industry

            # Semiconductor industry has more pronounced cycles
            # Typically 3-4 year cycles in demand
            cycle_pattern = [1.15, 1.10, 0.95, 0.90, 1.05, 1.15, 1.10, 0.95]  # Represents a semiconductor cycle

            # Terminal growth rate
            terminal_growth = 0.025  # 2.5% long-term growth

            # 4. Calculate discount rate (WACC)
            beta = market_data.get('beta', 1.3)  # Default beta for semiconductors
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            # Add size and industry premium
            industry_premium = 0.01  # Additional risk factor for semiconductor volatility

            cost_of_equity = risk_free_rate + beta * equity_risk_premium + industry_premium

            # Semiconductor companies often have significant debt for fab investments
            if 'Total Debt' in latest_balance.index and 'Total Assets' in latest_balance.index:
                debt_ratio = latest_balance['Total Debt'] / latest_balance['Total Assets']
            else:
                debt_ratio = 0.2  # Default debt ratio for semiconductors

            cost_of_debt = 0.04  # Typical cost of debt
            tax_rate = 0.20  # Many semiconductor companies have
            # favorable tax schemes due to R&D credits

            # WACC calculation
            equity_ratio = 1 - debt_ratio
            wacc = (cost_of_equity * equity_ratio) + (cost_of_debt * (1 - tax_rate) * debt_ratio)

            # Ensure WACC is reasonable
            wacc = max(0.09, min(0.15, wacc))  # Between 9% and 15%

            # 5. Forecast cash flows with cyclical pattern
            forecasted_cash_flows = []
            forecasted_revenues = []
            current_revenue = revenue

            for year in range(1, forecast_years + 1):
                # Apply cyclical adjustment
                cycle_index = (year - 1) % len(cycle_pattern)
                cyclical_factor = cycle_pattern[cycle_index]

                # Base growth rate adjusted for cycle
                year_growth = growth_rate * cyclical_factor

                # Revenue for the year
                current_revenue = current_revenue * (1 + year_growth)
                forecasted_revenues.append(current_revenue)

                # Operating income with cyclical margin
                # Semiconductor margins are highly cyclical
                if cyclical_factor < 1:
                    year_margin = operating_margin * 0.8  # 20% margin compression in down cycle
                elif cyclical_factor > 1.1:
                    year_margin = operating_margin * 1.2  # 20% margin expansion in up cycle
                else:
                    year_margin = operating_margin

                operating_income = current_revenue * year_margin

                # Taxes
                taxes = operating_income * tax_rate

                # Capital expenditures (high and cyclical for semiconductor companies)
                # In semiconductor, capex is highest at the beginning of an upcycle
                if cyclical_factor > 1.1 and cycle_pattern[(cycle_index - 1) % len(cycle_pattern)] < 1:
                    # Beginning of upcycle, heavy investment
                    capex_ratio = 0.20  # 20% of revenue
                else:
                    capex_ratio = latest_metrics.get('CapEx_to_Revenue', 0.12)  # Default 12% of revenue

                capex = current_revenue * capex_ratio

                # Depreciation (semiconductor equipment depreciates rapidly)
                depreciation_rate = 0.15  # 15% annual depreciation
                if 'Property Plant and Equipment' in latest_balance.index:
                    ppe = latest_balance['Property Plant and Equipment']
                    depreciation = ppe * depreciation_rate * (1 + (year - 1) * 0.05)  # Increasing as PPE grows
                else:
                    # Estimate depreciation as a percentage of capex
                    depreciation = capex * 0.7

                # Changes in working capital
                # Working capital needs increase in upcycles
                if cyclical_factor > 1:
                    working_capital_change = current_revenue * 0.05
                else:
                    working_capital_change = current_revenue * 0.02

                # Free cash flow
                fcf = operating_income - taxes + depreciation - capex - working_capital_change
                forecasted_cash_flows.append(fcf)

            # 6. Calculate terminal value
            # For cyclical industries, use normalized earnings for terminal value
            normalized_fcf = sum(forecasted_cash_flows[-4:]) / 4  # Average of last 4 years (full cycle)
            terminal_value = normalized_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

            # 7. Calculate present value of cash flows and terminal value
            present_value_fcf = sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            enterprise_value = present_value_fcf + present_value_terminal

            # 8. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 9. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'cyclical_semiconductor_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'growth_rate': growth_rate,
                'operating_margin': operating_margin,
                'terminal_growth': terminal_growth,
                'wacc': wacc,
                'cycle_pattern': cycle_pattern,
                'forecast_years': forecast_years,
                'forecasted_revenues': forecasted_revenues,
                'forecasted_cash_flows': forecasted_cash_flows,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'terminal_value': terminal_value,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in cyclical semiconductor DCF for {ticker}: {e}")
            return {
                'method': 'cyclical_semiconductor_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _semiconductor_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                           semi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a semiconductor company using industry-specific multiples
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = semi_metrics.get('Latest', {})
            growth_metrics = semi_metrics.get('Growth', {})

            # 1. Get financial metrics for valuation
            # Revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'semiconductor_multiples',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # EBITDA
            ebitda = None
            if 'EBITDA' in latest_income.index:
                ebitda = latest_income['EBITDA']
            else:
                # Calculate from components
                operating_income = None
                if 'Operating Income' in latest_income.index:
                    operating_income = latest_income['Operating Income']

                depreciation_amortization = None
                if 'Depreciation & Amortization' in latest_income.index:
                    depreciation_amortization = latest_income['Depreciation & Amortization']

                if operating_income is not None and depreciation_amortization is not None:
                    ebitda = operating_income + depreciation_amortization

            # Gross Profit
            gross_profit = None
            if 'Gross Profit' in latest_income.index:
                gross_profit = latest_income['Gross Profit']
            elif 'Total Revenue' in latest_income.index and 'Cost of Revenue' in latest_income.index:
                gross_profit = latest_income['Total Revenue'] - latest_income['Cost of Revenue']

            # R&D Expense
            rd_expense = None
            if 'Research and Development' in latest_income.index:
                rd_expense = latest_income['Research and Development']

            # 2. Determine appropriate multiples based on semiconductor industry
            # Base multiples - these would ideally come from industry databases
            base_multiples = {
                'EV_Revenue': 4.0,
                'EV_EBITDA': 12.0,
                'EV_GrossProfit': 8.0,
                'EV_RD': 10.0  # Semiconductor-specific multiple
            }

            # Adjust multiples based on company metrics

            # Growth adjustment
            revenue_growth = growth_metrics.get('Revenue_Growth')
            if revenue_growth:
                if revenue_growth > 0.20:  # High growth
                    growth_adjustment = 1.3
                elif revenue_growth > 0.10:  # Moderate growth
                    growth_adjustment = 1.1
                elif revenue_growth < 0:  # Declining revenue
                    growth_adjustment = 0.7
                else:
                    growth_adjustment = 1.0
            else:
                growth_adjustment = 1.0

            # Margin adjustment
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin:
                if gross_margin > 0.55:  # Premium margin (fabless or specialized)
                    margin_adjustment = 1.4
                elif gross_margin > 0.45:  # Good margin
                    margin_adjustment = 1.2
                elif gross_margin < 0.35:  # Low margin (commoditized)
                    margin_adjustment = 0.8
                else:
                    margin_adjustment = 1.0
            else:
                margin_adjustment = 1.0

            # IP/R&D adjustment
            rd_to_revenue = latest_metrics.get('RD_to_Revenue')
            if rd_to_revenue:
                if rd_to_revenue > 0.20:  # Research intensive
                    rd_adjustment = 1.3
                elif rd_to_revenue > 0.12:  # Average R&D
                    rd_adjustment = 1.1
                elif rd_to_revenue < 0.08:  # Low R&D
                    rd_adjustment = 0.8
                else:
                    rd_adjustment = 1.0
            else:
                rd_adjustment = 1.0

            # Apply adjustments to base multiples
            adjusted_multiples = {
                'EV_Revenue': base_multiples['EV_Revenue'] * growth_adjustment * margin_adjustment,
                'EV_EBITDA': base_multiples['EV_EBITDA'] * growth_adjustment,
                'EV_GrossProfit': base_multiples['EV_GrossProfit'] * margin_adjustment,
                'EV_RD': base_multiples['EV_RD'] * rd_adjustment
            }

            # 3. Calculate valuations using different multiples
            valuations = {}

            if revenue:
                valuations['EV_Revenue'] = {
                    'multiple': adjusted_multiples['EV_Revenue'],
                    'metric_value': revenue,
                    'enterprise_value': revenue * adjusted_multiples['EV_Revenue']
                }

            if ebitda:
                valuations['EV_EBITDA'] = {
                    'multiple': adjusted_multiples['EV_EBITDA'],
                    'metric_value': ebitda,
                    'enterprise_value': ebitda * adjusted_multiples['EV_EBITDA']
                }

            if gross_profit:
                valuations['EV_GrossProfit'] = {
                    'multiple': adjusted_multiples['EV_GrossProfit'],
                    'metric_value': gross_profit,
                    'enterprise_value': gross_profit * adjusted_multiples['EV_GrossProfit']
                }

            if rd_expense:
                valuations['EV_RD'] = {
                    'multiple': adjusted_multiples['EV_RD'],
                    'metric_value': rd_expense,
                    'enterprise_value': rd_expense * adjusted_multiples['EV_RD']
                }

            # 4. Calculate weighted average enterprise value
            # Weights reflect importance of each metric for semiconductor valuation
            weights = {
                'EV_Revenue': 0.2,
                'EV_EBITDA': 0.3,
                'EV_GrossProfit': 0.3,
                'EV_RD': 0.2
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                # Fallback to EV/Revenue if other metrics not available
                if 'EV_Revenue' in valuations:
                    enterprise_value = valuations['EV_Revenue']['enterprise_value']
                else:
                    return {
                        'method': 'semiconductor_multiples',
                        'value_per_share': None,
                        'error': 'Insufficient metrics for valuation'
                    }

            # 5. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 6. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'semiconductor_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'ebitda': ebitda,
                'gross_profit': gross_profit,
                'rd_expense': rd_expense,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'valuations': valuations,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in semiconductor multiples valuation for {ticker}: {e}")
            return {
                'method': 'semiconductor_multiples',
                'value_per_share': None,
                'error': str(e)
            }

    def _semiconductor_ip_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                    semi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value a semiconductor company based on intellectual property (IP) and R&D
        which are critical value drivers in this industry
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = semi_metrics.get('Latest', {})

            # 1. Estimate value of intellectual property
            # This includes patents, trade secrets, proprietary processes, etc.

            # Start with reported intangible assets
            intangible_assets = 0
            if 'Intangible Assets' in latest_balance.index:
                intangible_assets = latest_balance['Intangible Assets']

            if 'Goodwill' in latest_balance.index:
                # Add a portion of goodwill - some represents IP from acquisitions
                goodwill = latest_balance['Goodwill']
                ip_portion_of_goodwill = goodwill * 0.3  # Assume 30% of goodwill is IP
                intangible_assets += ip_portion_of_goodwill

            # 2. Estimate value of R&D investments
            # Semiconductor companies often have significant R&D that gets expensed
            # but creates long-term value. We capitalize a portion of historical R&D.

            rd_asset_value = 0
            rd_amortization_years = 5  # Typical semiconductor R&D lifecycle

            # Get historical R&D if available
            historical_rd = []

            for i in range(min(rd_amortization_years, income_stmt.shape[1])):
                if 'Research and Development' in income_stmt.iloc[:, i].index:
                    historical_rd.append(income_stmt.iloc[:, i]['Research and Development'])
                else:
                    # If no data, estimate from recent years or use zero
                    if historical_rd:
                        # Use average of available data
                        historical_rd.append(sum(historical_rd) / len(historical_rd))
                    else:
                        historical_rd.append(0)

            # Capitalize and amortize R&D
            for i, rd_expense in enumerate(historical_rd):
                # More recent R&D has more remaining value
                remaining_value_percent = (rd_amortization_years - i) / rd_amortization_years
                rd_asset_value += rd_expense * 0.7 * remaining_value_percent  # Capitalize 70% of R&D

            # 3. Calculate total IP value
            total_ip_value = intangible_assets + rd_asset_value

            # 4. Determine IP value multiple
            # Different semiconductor segments have different IP value
            # (e.g., fabless design companies vs foundries)

            # Base multiple for semiconductor IP
            base_ip_multiple = 2.0  # Starting point

            # Adjust based on gross margin (proxy for IP value)
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin:
                if gross_margin > 0.60:  # High IP-value business (e.g., fabless designers)
                    ip_multiple_adjustment = 1.5
                elif gross_margin > 0.50:  # Strong IP
                    ip_multiple_adjustment = 1.3
                elif gross_margin < 0.40:  # Lower IP value (e.g., pure foundry)
                    ip_multiple_adjustment = 0.8
                else:
                    ip_multiple_adjustment = 1.0
            else:
                ip_multiple_adjustment = 1.0

            # Adjust based on R&D intensity
            rd_to_revenue = latest_metrics.get('RD_to_Revenue')
            if rd_to_revenue:
                if rd_to_revenue > 0.20:  # Research intensive
                    rd_multiple_adjustment = 1.4
                elif rd_to_revenue > 0.15:  # Above average R&D
                    rd_multiple_adjustment = 1.2
                elif rd_to_revenue < 0.10:  # Below average R&D
                    rd_multiple_adjustment = 0.8
                else:
                    rd_multiple_adjustment = 1.0
            else:
                rd_multiple_adjustment = 1.0

            # Apply adjustments
            ip_multiple = base_ip_multiple * ip_multiple_adjustment * rd_multiple_adjustment

            # 5. Calculate IP-based enterprise value
            ip_based_value = total_ip_value * ip_multiple

            # 6. Add value of tangible assets (excluding cash, which will be handled separately)
            tangible_asset_value = 0

            if 'Property Plant and Equipment' in latest_balance.index:
                tangible_asset_value += latest_balance['Property Plant and Equipment']

            if 'Inventory' in latest_balance.index:
                tangible_asset_value += latest_balance['Inventory'] * 0.8  # Discount inventory slightly

            # 7. Calculate total enterprise value
            enterprise_value = ip_based_value + tangible_asset_value

            # 8. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 9. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'semiconductor_ip',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'intangible_assets': intangible_assets,
                'rd_asset_value': rd_asset_value,
                'total_ip_value': total_ip_value,
                'ip_multiple': ip_multiple,
                'tangible_asset_value': tangible_asset_value,
                'ip_based_value': ip_based_value,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in semiconductor IP valuation for {ticker}: {e}")
            return {
                'method': 'semiconductor_ip',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_internet_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics for internet/platform companies"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None:
                return metrics

            # Get data for multiple periods
            periods = min(income_stmt.shape[1], 3)  # Use up to 3 years

            # User metrics - these would typically come from company filings/presentations
            # Since we don't have direct access, we'll make estimations or placeholders
            user_data = {
                'MAU': company_info.get('MAU'),  # Monthly Active Users
                'DAU': company_info.get('DAU'),  # Daily Active Users
                'Paid_Users': company_info.get('Paid_Users'),  # Number of paying users
                'ARPU': company_info.get('ARPU')  # Average Revenue Per User
            }

            metrics['User_Data'] = user_data

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]

                period_metrics = {}

                # Revenue
                if 'Total Revenue' in income.index:
                    revenue = income['Total Revenue']
                    period_metrics['Revenue'] = revenue

                # Gross Profit and Margin
                if 'Gross Profit' in income.index and 'Revenue' in period_metrics:
                    gross_profit = income['Gross Profit']
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']
                elif 'Cost of Revenue' in income.index and 'Revenue' in period_metrics:
                    cost_of_revenue = income['Cost of Revenue']
                    gross_profit = period_metrics['Revenue'] - cost_of_revenue
                    period_metrics['Gross_Profit'] = gross_profit
                    period_metrics['Gross_Margin'] = gross_profit / period_metrics['Revenue']

                # Operating Income and Margin
                if 'Operating Income' in income.index and 'Revenue' in period_metrics:
                    operating_income = income['Operating Income']
                    period_metrics['Operating_Income'] = operating_income
                    period_metrics['Operating_Margin'] = operating_income / period_metrics['Revenue']

                # Sales & Marketing Expense and % of Revenue
                if 'Sales and Marketing' in income.index and 'Revenue' in period_metrics:
                    sm_expense = income['Sales and Marketing']
                    period_metrics['SM_Expense'] = sm_expense
                    period_metrics['SM_to_Revenue'] = sm_expense / period_metrics['Revenue']

                # R&D as % of Revenue
                if 'Research and Development' in income.index and 'Revenue' in period_metrics:
                    rd_expense = income['Research and Development']
                    period_metrics['RD_Expense'] = rd_expense
                    period_metrics['RD_to_Revenue'] = rd_expense / period_metrics['Revenue']

                # Calculate ARPU if we have user data but don't have direct ARPU
                if user_data.get('MAU') and 'ARPU' not in user_data and 'Revenue' in period_metrics:
                    estimated_arpu = period_metrics['Revenue'] / user_data['MAU']
                    period_metrics['Estimated_ARPU'] = estimated_arpu

                # Estimate Customer Acquisition Cost (CAC) if we have S&M and user growth
                # This is a very rough estimation
                if 'SM_Expense' in period_metrics and user_data.get('MAU'):
                    # Assume 5% monthly user growth
                    estimated_new_users = user_data['MAU'] * 0.05 * 12  # Annual new users
                    if estimated_new_users > 0:
                        estimated_cac = period_metrics['SM_Expense'] / estimated_new_users
                        period_metrics['Estimated_CAC'] = estimated_cac

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                # Revenue growth
                if 'Revenue' in metrics["Latest"] and 'Revenue' in metrics["Year-1"] and metrics["Year-1"][
                    'Revenue'] > 0:
                    revenue_growth = (metrics["Latest"]['Revenue'] / metrics["Year-1"]['Revenue']) - 1
                    growth_metrics['Revenue_Growth'] = revenue_growth

                # Margin trends
                for margin in ['Gross_Margin', 'Operating_Margin']:
                    if margin in metrics["Latest"] and margin in metrics["Year-1"]:
                        margin_change = metrics["Latest"][margin] - metrics["Year-1"][margin]
                        growth_metrics[f'{margin}_Change'] = margin_change

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating internet metrics: {e}")
            return {}

    def _user_based_valuation(self, ticker: str, financial_data: Dict[str, Any],
                              internet_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an internet company based on user metrics (MAU, ARPU, LTV, etc.)
        This is particularly relevant for social media, marketplaces, and platforms
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_metrics = internet_metrics.get('Latest', {})
            user_data = internet_metrics.get('User_Data', {})

            # 1. Get user base metrics
            mau = user_data.get('MAU')  # Monthly Active Users

            # If no direct MAU data, need to estimate from other sources
            if mau is None:
                # Try to derive from revenue and estimated ARPU
                if 'Estimated_ARPU' in latest_metrics and 'Revenue' in latest_metrics:
                    estimated_arpu = latest_metrics['Estimated_ARPU']
                    if estimated_arpu > 0:
                        mau = latest_metrics['Revenue'] / estimated_arpu
                else:
                    # Unable to estimate MAU, cannot use this method
                    return {
                        'method': 'user_based',
                        'value_per_share': None,
                        'error': 'Cannot determine MAU'
                    }

            # 2. Get or estimate ARPU
            arpu = user_data.get('ARPU')
            if arpu is None and 'Estimated_ARPU' in latest_metrics:
                arpu = latest_metrics['Estimated_ARPU']
            elif arpu is None and 'Revenue' in latest_metrics and mau > 0:
                arpu = latest_metrics['Revenue'] / mau

            if arpu is None:
                # Unable to estimate ARPU, cannot use this method
                return {
                    'method': 'user_based',
                    'value_per_share': None,
                    'error': 'Cannot determine ARPU'
                }

            # 3. Get or estimate retention rate
            # This would typically come from company disclosures
            # For this example, we'll use a default based on industry averages
            retention_rate = 0.8  # Default assumption (80% annual retention)

            # 4. Get or estimate gross margin
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin is None:
                if 'Gross_Profit' in latest_metrics and 'Revenue' in latest_metrics and latest_metrics['Revenue'] > 0:
                    gross_margin = latest_metrics['Gross_Profit'] / latest_metrics['Revenue']
                else:
                    # Default for internet platforms
                    gross_margin = 0.7  # 70% is typical for internet platforms

            # 5. Calculate Lifetime Value (LTV)
            # LTV = ARPU * Gross Margin * Customer Lifetime
            # Customer Lifetime = 1 / (1 - Retention Rate)
            customer_lifetime = 1 / (1 - retention_rate)
            ltv = arpu * gross_margin * customer_lifetime

            # 6. Get or estimate Customer Acquisition Cost (CAC)
            cac = None
            if 'Estimated_CAC' in latest_metrics:
                cac = latest_metrics['Estimated_CAC']
            elif 'SM_Expense' in latest_metrics:
                # Rough estimation based on S&M expense
                # Assume 80% of S&M is for acquisition and 20% for retention
                acquisition_expense = latest_metrics['SM_Expense'] * 0.8

                # Assume 10% annual user growth to estimate new users
                estimated_new_users = mau * 0.1

                if estimated_new_users > 0:
                    cac = acquisition_expense / estimated_new_users

            if cac is None:
                # Default based on industry (varies widely)
                cac = arpu * 2  # Default: CAC is 2x ARPU

            # 7. Calculate LTV/CAC ratio
            ltv_cac_ratio = ltv / cac if cac > 0 else 0

            # 8. Project user growth and future value
            # Estimate future growth rates
            current_users = mau
            growth_rates = []

            # Determine user growth trajectory based on company maturity
            # This would ideally come from historical data or company guidance
            base_growth_rate = 0.15  # Initial annual growth rate

            # Project growth for 5 years with declining growth rate
            for i in range(5):
                growth_rates.append(max(base_growth_rate * (1 - i * 0.2), 0.03))

            # Calculate future user base and values
            future_users = []
            future_values = []
            cumulative_value = 0

            for i, growth_rate in enumerate(growth_rates):
                if i == 0:
                    new_users = current_users * growth_rate
                else:
                    new_users = future_users[i - 1] * growth_rate

                future_users.append(current_users + new_users)
                current_users = future_users[i]

                # Calculate value created from new users
                year_value = (ltv - cac) * new_users

                # Discount to present value
                discount_rate = 0.12  # Higher discount rate for user-based model
                present_value = year_value / ((1 + discount_rate) ** (i + 1))

                future_values.append(present_value)
                cumulative_value += present_value

            # 9. Add value of existing users
            existing_user_value = ltv * mau

            # 10. Calculate enterprise value
            enterprise_value = cumulative_value + existing_user_value

            # 11. Adjust for net cash/debt
            net_cash = 0
            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # 12. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'user_based',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'mau': mau,
                'arpu': arpu,
                'retention_rate': retention_rate,
                'gross_margin': gross_margin,
                'customer_lifetime': customer_lifetime,
                'ltv': ltv,
                'cac': cac,
                'ltv_cac_ratio': ltv_cac_ratio,
                'existing_user_value': existing_user_value,
                'new_user_value': cumulative_value,
                'future_users': future_users,
                'future_values': future_values,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in user-based valuation for {ticker}: {e}")
            return {
                'method': 'user_based',
                'value_per_share': None,
                'error': str(e)
            }

    def _platform_economics_dcf(self, ticker: str, financial_data: Dict[str, Any],
                                internet_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an internet company using a DCF model that incorporates
        platform economics (network effects, scaling properties, etc.)
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Get latest metrics
            latest_metrics = internet_metrics.get('Latest', {})
            growth_metrics = internet_metrics.get('Growth', {})

            # 1. Determine starting revenue and growth rate
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'platform_economics_dcf',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Determine revenue growth rate
            growth_rate = growth_metrics.get('Revenue_Growth')
            if growth_rate is None:
                # Try to calculate from historical data
                if income_stmt.shape[1] >= 2:
                    prev_revenue = income_stmt.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_stmt.iloc[:,
                                                                                                 1].index else None
                    if prev_revenue and prev_revenue > 0:
                        growth_rate = (revenue / prev_revenue) - 1
                    else:
                        # Use default based on internet companies
                        growth_rate = 0.25  # 25% default growth
                else:
                    growth_rate = 0.25  # Default if no historical data

            # 2. Determine margins
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin is None:
                if 'Gross Profit' in latest_income.index and revenue > 0:
                    gross_margin = latest_income['Gross Profit'] / revenue
                else:
                    # Default gross margin for internet platforms
                    gross_margin = 0.70  # 70% is typical

            operating_margin = latest_metrics.get('Operating_Margin')
            if operating_margin is None:
                if 'Operating Income' in latest_income.index and revenue > 0:
                    operating_margin = latest_income['Operating Income'] / revenue
                else:
                    # Default operating margin for internet platforms
                    operating_margin = 0.15  # 15% is typical for mature platforms

            # 3. Set up DCF parameters specific to platform economics
            forecast_years = 10  # Longer forecast for platform businesses due to network effects

            # Platform businesses often show strong network effects
            # This manifests as increasing returns to scale
            # Here we model margin expansion and decreasing growth decay

            # Initial target margin (platforms often have high margins at scale)
            target_operating_margin = 0.30  # 30% target margin

            # Terminal growth rate
            terminal_growth = 0.03  # 3% long-term growth

            # 4. Calculate discount rate (WACC)
            beta = market_data.get('beta', 1.3)  # Default beta for internet platforms
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            # Add size premium and specific risk
            size_premium = 0.01
            specific_risk = 0.01  # Platform competition, regulation risks

            cost_of_equity = risk_free_rate + beta * equity_risk_premium + size_premium + specific_risk

            # Most internet platforms are primarily equity financed
            wacc = cost_of_equity

            # Ensure WACC is reasonable
            wacc = max(0.09, min(0.16, wacc))  # Between 9% and 16%

            # 5. Forecast cash flows with platform economics
            forecasted_cash_flows = []
            forecasted_revenues = []
            forecasted_margins = []
            current_revenue = revenue
            current_margin = operating_margin

            for year in range(1, forecast_years + 1):
                # Platform growth often follows an S-curve
                # Initial high growth, followed by moderation
                growth_decay = (1 / (1 + 0.5 * year)) * (growth_rate - terminal_growth)
                year_growth_rate = growth_rate - growth_decay

                # Platform margin expansion with scale
                margin_improvement = min(0.02, (target_operating_margin - current_margin) / (forecast_years - year + 1))
                year_margin = current_margin + margin_improvement

                # Revenue for the year
                current_revenue = current_revenue * (1 + year_growth_rate)
                forecasted_revenues.append(current_revenue)

                # Update margin for next year
                current_margin = year_margin
                forecasted_margins.append(current_margin)

                # Operating income
                operating_income = current_revenue * current_margin

                # Taxes
                tax_rate = 0.25  # Standard tax rate
                taxes = max(0, operating_income * tax_rate)  # Only tax if profitable

                # Capital expenditures
                # Internet platforms typically have lower capex than traditional businesses
                capex = current_revenue * 0.07  # 7% of revenue

                # Changes in working capital
                # Platforms often have favorable working capital dynamics
                working_capital_change = current_revenue * 0.01  # 1% of revenue

                # Free cash flow
                fcf = operating_income - taxes - capex - working_capital_change
                forecasted_cash_flows.append(fcf)

            # 6. Calculate terminal value
            final_fcf = forecasted_cash_flows[-1]
            terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

            # 7. Calculate present value of cash flows and terminal value
            present_value_fcf = sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(forecasted_cash_flows))
            present_value_terminal = terminal_value / ((1 + wacc) ** forecast_years)

            enterprise_value = present_value_fcf + present_value_terminal

            # 8. Adjust for net cash/debt
            if 'Cash and Cash Equivalents' in latest_balance.index and 'Total Debt' in latest_balance.index:
                cash = latest_balance['Cash and Cash Equivalents']
                debt = latest_balance['Total Debt']
                net_cash = cash - debt
            else:
                net_cash = 0

            equity_value = enterprise_value + net_cash

            # 9. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'platform_economics_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'growth_rate': growth_rate,
                'operating_margin': operating_margin,
                'target_operating_margin': target_operating_margin,
                'terminal_growth': terminal_growth,
                'wacc': wacc,
                'forecast_years': forecast_years,
                'forecasted_revenues': forecasted_revenues,
                'forecasted_margins': forecasted_margins,
                'forecasted_cash_flows': forecasted_cash_flows,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'terminal_value': terminal_value,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in platform economics DCF for {ticker}: {e}")
            return {
                'method': 'platform_economics_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _internet_multiples_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                      internet_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an internet company using sector-specific multiples
        adjusted for network effects premium
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest data
            latest_income = income_stmt.iloc[:, 0]

            # Get metrics
            latest_metrics = internet_metrics.get('Latest', {})
            growth_metrics = internet_metrics.get('Growth', {})
            user_data = internet_metrics.get('User_Data', {})

            # 1. Get financial metrics for valuation

            # Revenue
            if 'Total Revenue' in latest_income.index:
                revenue = latest_income['Total Revenue']
            else:
                return {
                    'method': 'internet_multiples',
                    'value_per_share': None,
                    'error': 'Cannot determine revenue'
                }

            # Gross Profit
            gross_profit = None
            if 'Gross Profit' in latest_income.index:
                gross_profit = latest_income['Gross Profit']
            elif 'Cost of Revenue' in latest_income.index:
                gross_profit = revenue - latest_income['Cost of Revenue']

            # EBITDA
            ebitda = None
            if 'EBITDA' in latest_income.index:
                ebitda = latest_income['EBITDA']
            else:
                # Try to calculate from components
                operating_income = None
                if 'Operating Income' in latest_income.index:
                    operating_income = latest_income['Operating Income']

                depreciation_amortization = None
                if 'Depreciation & Amortization' in latest_income.index:
                    depreciation_amortization = latest_income['Depreciation & Amortization']

                if operating_income is not None and depreciation_amortization is not None:
                    ebitda = operating_income + depreciation_amortization

            # 2. Determine appropriate multiples for internet companies
            # Base multiples - these would ideally come from industry databases
            base_multiples = {
                'EV_Revenue': 6.0,
                'EV_EBITDA': 18.0,
                'EV_GrossProfit': 8.0,
                'EV_MAU': 100.0  # Value per monthly active user ($)
            }

            # 3. Adjust multiples based on growth and platform strength

            # Growth adjustment
            revenue_growth = growth_metrics.get('Revenue_Growth')
            if revenue_growth:
                if revenue_growth > 0.40:  # Hypergrowth
                    growth_adjustment = 1.6
                elif revenue_growth > 0.25:  # High growth
                    growth_adjustment = 1.3
                elif revenue_growth > 0.15:  # Good growth
                    growth_adjustment = 1.1
                elif revenue_growth < 0.05:  # Low growth
                    growth_adjustment = 0.8
                else:
                    growth_adjustment = 1.0
            else:
                growth_adjustment = 1.0

            # Margin adjustment
            gross_margin = latest_metrics.get('Gross_Margin')
            if gross_margin:
                if gross_margin > 0.75:  # Exceptional margins
                    margin_adjustment = 1.4
                elif gross_margin > 0.65:  # Very good margins
                    margin_adjustment = 1.2
                elif gross_margin < 0.55:  # Below average margins
                    margin_adjustment = 0.8
                else:
                    margin_adjustment = 1.0
            else:
                margin_adjustment = 1.0

            # Network effects premium
            # This would ideally be determined by user growth, engagement metrics, etc.
            network_premium = 1.0

            # If we have MAU data, check for network effects
            mau = user_data.get('MAU')
            if mau:
                # Large user bases command higher premiums due to network effects
                if mau > 1000000000:  # >1B users (major platforms)
                    network_premium = 1.5
                elif mau > 100000000:  # >100M users (established platforms)
                    network_premium = 1.3
                elif mau > 10000000:  # >10M users (growing platforms)
                    network_premium = 1.1

            # Apply adjustments to base multiples
            adjusted_multiples = {
                'EV_Revenue': base_multiples['EV_Revenue'] * growth_adjustment * network_premium,
                'EV_EBITDA': base_multiples['EV_EBITDA'] * margin_adjustment * network_premium,
                'EV_GrossProfit': base_multiples['EV_GrossProfit'] * margin_adjustment * growth_adjustment,
                'EV_MAU': base_multiples['EV_MAU'] * network_premium * growth_adjustment
            }

            # 4. Calculate valuations using different multiples
            valuations = {}

            if revenue:
                valuations['EV_Revenue'] = {
                    'multiple': adjusted_multiples['EV_Revenue'],
                    'metric_value': revenue,
                    'enterprise_value': revenue * adjusted_multiples['EV_Revenue']
                }

            if ebitda and ebitda > 0:
                valuations['EV_EBITDA'] = {
                    'multiple': adjusted_multiples['EV_EBITDA'],
                    'metric_value': ebitda,
                    'enterprise_value': ebitda * adjusted_multiples['EV_EBITDA']
                }

            if gross_profit:
                valuations['EV_GrossProfit'] = {
                    'multiple': adjusted_multiples['EV_GrossProfit'],
                    'metric_value': gross_profit,
                    'enterprise_value': gross_profit * adjusted_multiples['EV_GrossProfit']
                }

            if mau:
                valuations['EV_MAU'] = {
                    'multiple': adjusted_multiples['EV_MAU'],
                    'metric_value': mau,
                    'enterprise_value': mau * adjusted_multiples['EV_MAU']
                }

            # 5. Calculate weighted average enterprise value
            # Weights based on reliability for internet platforms
            weights = {
                'EV_Revenue': 0.4,
                'EV_EBITDA': 0.2,
                'EV_GrossProfit': 0.2,
                'EV_MAU': 0.2
            }

            total_weight = 0
            weighted_ev = 0

            for metric, weight in weights.items():
                if metric in valuations:
                    weighted_ev += valuations[metric]['enterprise_value'] * weight
                    total_weight += weight

            if total_weight > 0:
                enterprise_value = weighted_ev / total_weight
            else:
                # Fallback to EV/Revenue if other metrics not available
                if 'EV_Revenue' in valuations:
                    enterprise_value = valuations['EV_Revenue']['enterprise_value']
                else:
                    return {
                        'method': 'internet_multiples',
                        'value_per_share': None,
                        'error': 'Insufficient metrics for valuation'
                    }

            # 6. Adjust for net cash/debt
            net_cash = 0
            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            equity_value = enterprise_value + net_cash

            # 7. Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'internet_multiples',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'revenue': revenue,
                'ebitda': ebitda,
                'gross_profit': gross_profit,
                'mau': mau,
                'base_multiples': base_multiples,
                'adjusted_multiples': adjusted_multiples,
                'growth_adjustment': growth_adjustment,
                'margin_adjustment': margin_adjustment,
                'network_premium': network_premium,
                'valuations': valuations,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in internet multiples valuation for {ticker}: {e}")
            return {
                'method': 'internet_multiples',
                'value_per_share': None,
                'error': str(e)
            }