import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import RISK_FREE_RATE, DCF_PARAMETERS, SECTOR_DCF_PARAMETERS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation, SectorSpecificDCF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('financial_sector')


class FinancialSectorValuation(SectorSpecificDCF):
    """
    Specialized valuation models for financial sector companies 
    (banks, insurance companies, asset management, etc.)

    Financial sector valuation requires different approaches due to:
    1. Regulatory requirements and capital constraints
    2. Different financial reporting (interest income vs. revenue)
    3. Balance sheet focus instead of income statement
    4. Different risk factors (credit risk, interest rate risk)
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize financial sector valuation class"""
        super().__init__(data_loader)
        logger.info("Initialized FinancialSectorValuation")

        # Specific parameters for financial sector
        self.deposit_growth_adjustment = 0.95  # Adjustment factor for deposit growth
        self.loan_to_deposit_target = 0.8  # Target loan-to-deposit ratio
        self.min_capital_adequacy = 0.10  # Minimum capital adequacy ratio

    def value_financial_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master method to value a financial sector company, selecting the most appropriate model
        based on company sub-sector (banking, insurance, asset management)
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Determine financial sub-sector
            sub_sector = self._determine_financial_subsector(ticker, financial_data)

            # Select valuation method based on sub-sector
            if sub_sector == "banking":
                result = self.value_bank(ticker, financial_data)
            elif sub_sector == "insurance":
                result = self.value_insurance_company(ticker, financial_data)
            elif sub_sector == "asset_management":
                result = self.value_asset_manager(ticker, financial_data)
            else:
                # Use general approach for financial companies
                result = self.financial_sector_dcf(ticker, financial_data)

            # Add sub-sector information to result
            result['sub_sector'] = sub_sector

            return result

        except Exception as e:
            logger.error(f"Error valuing financial company {ticker}: {e}")
            # Fall back to standard approach
            return self.financial_sector_dcf(ticker, financial_data)

    def value_bank(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value a bank using specialized models including:
        1. Adjusted Book Value approach
        2. Dividend Discount Model with sustainable ROE
        3. Residual Income / Excess Return Model
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for bank valuation")

            # Get the most recent data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]

            # Calculate key banking metrics
            banking_metrics = self._calculate_banking_metrics(financial_data)

            # 1. Calculate adjusted book value
            adjusted_book_value = self._calculate_adjusted_book_value(latest_balance, banking_metrics)

            # 2. Perform DDM valuation with sustainable ROE
            ddm_result = self._bank_dividend_model(ticker, financial_data, banking_metrics)

            # 3. Calculate residual income valuation
            ri_result = self._residual_income_valuation(ticker, financial_data, banking_metrics)

            # Calculate per share values
            if shares_outstanding and shares_outstanding > 0:
                adjusted_book_value_per_share = adjusted_book_value / shares_outstanding
            else:
                adjusted_book_value_per_share = None

            # Blend the valuation approaches (40% Book Value, 30% DDM, 30% Residual Income)
            if adjusted_book_value_per_share and ddm_result.get('value_per_share') and ri_result.get('value_per_share'):
                blended_value = (
                        adjusted_book_value_per_share * 0.4 +
                        ddm_result['value_per_share'] * 0.3 +
                        ri_result['value_per_share'] * 0.3
                )
            elif adjusted_book_value_per_share:
                # If other methods failed, rely more on book value
                blended_value = adjusted_book_value_per_share
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'bank_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'adjusted_book_value': adjusted_book_value,
                'adjusted_book_value_per_share': adjusted_book_value_per_share,
                'ddm_valuation': ddm_result,
                'residual_income_valuation': ri_result,
                'banking_metrics': banking_metrics
            }

        except Exception as e:
            logger.error(f"Error in bank valuation for {ticker}: {e}")
            # Fall back to general financial sector DCF
            return self.financial_sector_dcf(ticker, financial_data)

    def value_insurance_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an insurance company using specialized models including:
        1. Adjusted Book Value with focus on reserves
        2. Dividend Discount Model with combined ratio adjustment
        3. Appraisal Value (Embedded Value + New Business Value)
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for insurance company valuation")

            # Calculate insurance metrics
            insurance_metrics = self._calculate_insurance_metrics(financial_data)

            # 1. Calculate adjusted book value with focus on reserves
            adjusted_book_value = self._calculate_insurance_adjusted_book_value(
                balance_sheet.iloc[:, 0], insurance_metrics)

            # 2. Perform DDM valuation with combined ratio adjustment
            ddm_result = self._insurance_dividend_model(ticker, financial_data, insurance_metrics)

            # 3. Calculate appraisal value (embedded value + new business value)
            # Note: This is a simplified approach as true embedded value requires detailed actuarial data
            appraisal_result = self._insurance_appraisal_value(ticker, financial_data, insurance_metrics)

            # Calculate per share values
            if shares_outstanding and shares_outstanding > 0:
                adjusted_book_value_per_share = adjusted_book_value / shares_outstanding
            else:
                adjusted_book_value_per_share = None

            # Blend the valuation approaches (35% Book Value, 35% DDM, 30% Appraisal)
            if adjusted_book_value_per_share and ddm_result.get('value_per_share') and appraisal_result.get(
                    'value_per_share'):
                blended_value = (
                        adjusted_book_value_per_share * 0.35 +
                        ddm_result['value_per_share'] * 0.35 +
                        appraisal_result['value_per_share'] * 0.30
                )
            elif adjusted_book_value_per_share:
                # If other methods failed, rely more on book value
                blended_value = adjusted_book_value_per_share
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'insurance_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'adjusted_book_value': adjusted_book_value,
                'adjusted_book_value_per_share': adjusted_book_value_per_share,
                'ddm_valuation': ddm_result,
                'appraisal_valuation': appraisal_result,
                'insurance_metrics': insurance_metrics
            }

        except Exception as e:
            logger.error(f"Error in insurance company valuation for {ticker}: {e}")
            # Fall back to general financial sector DCF
            return self.financial_sector_dcf(ticker, financial_data)

    def value_asset_manager(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Value an asset management company using specialized models including:
        1. AUM-based valuation
        2. Fee-based DCF model
        3. P/E or EV/EBITDA with AUM growth adjustment
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)

            # Extract financial statements
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            if income_stmt is None or balance_sheet is None:
                raise ValueError("Missing required financial statements for asset manager valuation")

            # Calculate asset management metrics
            am_metrics = self._calculate_asset_manager_metrics(financial_data)

            # 1. Calculate AUM-based valuation
            aum_valuation = self._aum_based_valuation(ticker, financial_data, am_metrics)

            # 2. Perform fee-based DCF model
            fee_dcf_result = self._fee_based_dcf(ticker, financial_data, am_metrics)

            # 3. Calculate relative valuation with AUM growth adjustment
            relative_result = self._asset_manager_relative_valuation(ticker, financial_data, am_metrics)

            # Blend the valuation approaches (30% AUM, 40% Fee DCF, 30% Relative)
            if (aum_valuation.get('value_per_share') and
                    fee_dcf_result.get('value_per_share') and
                    relative_result.get('value_per_share')):
                blended_value = (
                        aum_valuation['value_per_share'] * 0.30 +
                        fee_dcf_result['value_per_share'] * 0.40 +
                        relative_result['value_per_share'] * 0.30
                )
            elif fee_dcf_result.get('value_per_share'):
                # If other methods failed, rely more on DCF
                blended_value = fee_dcf_result['value_per_share']
            else:
                blended_value = None

            # Apply margin of safety
            if blended_value:
                conservative_value = blended_value * 0.85  # 15% margin of safety
            else:
                conservative_value = None

            return {
                'company': ticker,
                'method': 'asset_manager_valuation',
                'blended_value_per_share': blended_value,
                'conservative_value': conservative_value,
                'aum_valuation': aum_valuation,
                'fee_dcf_valuation': fee_dcf_result,
                'relative_valuation': relative_result,
                'asset_manager_metrics': am_metrics
            }

        except Exception as e:
            logger.error(f"Error in asset manager valuation for {ticker}: {e}")
            # Fall back to general financial sector DCF
            return self.financial_sector_dcf(ticker, financial_data)

    # Helper methods for financial sector valuation

    def _determine_financial_subsector(self, ticker: str, financial_data: Dict[str, Any]) -> str:
        """Determine the subsector of a financial company"""
        try:
            # Try to get industry from company info
            company_info = financial_data.get('company_info', {})
            industry = company_info.get('industry', '')

            # Check income statement structure for clues
            income_stmt = financial_data.get('income_statement')

            if income_stmt is not None:
                # Banking indicators
                banking_indicators = ['Net Interest Income', 'Loan Loss Provision', 'Net Interest Margin']

                # Insurance indicators
                insurance_indicators = ['Premium Income', 'Claims', 'Underwriting Income', 'Combined Ratio']

                # Asset management indicators
                am_indicators = ['Asset Management Fees', 'Assets Under Management', 'Advisory Fees']

                # Count matches for each sub-sector
                banking_count = sum(
                    1 for indicator in banking_indicators if any(indicator in index for index in income_stmt.index))
                insurance_count = sum(
                    1 for indicator in insurance_indicators if any(indicator in index for index in income_stmt.index))
                am_count = sum(
                    1 for indicator in am_indicators if any(indicator in index for index in income_stmt.index))

                # Determine based on highest count
                if banking_count > insurance_count and banking_count > am_count:
                    return "banking"
                elif insurance_count > banking_count and insurance_count > am_count:
                    return "insurance"
                elif am_count > banking_count and am_count > insurance_count:
                    return "asset_management"

            # If can't determine from income statement, use industry description
            if any(term in industry.lower() for term in ['bank', 'credit', 'loan', 'deposit']):
                return "banking"
            elif any(term in industry.lower() for term in ['insurance', 'underwriting', 'reinsurance']):
                return "insurance"
            elif any(term in industry.lower() for term in ['asset management', 'investment management', 'wealth']):
                return "asset_management"

            # Default to generic financial
            return "financial"

        except Exception as e:
            logger.warning(f"Error determining financial subsector for {ticker}: {e}")
            return "financial"  # Generic financial as fallback

    def _calculate_banking_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key banking metrics for valuation"""
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

                # Find net interest income
                net_interest_income = None
                for item in ['Net Interest Income', 'Interest Income - Interest Expense']:
                    if item in income.index:
                        net_interest_income = income[item]
                        break

                # If not directly available, calculate it
                if net_interest_income is None:
                    interest_income = next((income[item] for item in ['Interest Income', 'Total Interest Income']
                                            if item in income.index), None)
                    interest_expense = next((income[item] for item in ['Interest Expense', 'Total Interest Expense']
                                             if item in income.index), None)

                    if interest_income is not None and interest_expense is not None:
                        net_interest_income = interest_income - interest_expense

                # Find assets, loans, deposits, equity
                total_assets = next((balance[item] for item in ['Total Assets'] if item in balance.index), None)

                loans = next((balance[item] for item in ['Net Loans', 'Loans', 'Loans Receivable']
                              if item in balance.index), None)

                deposits = next((balance[item] for item in ['Total Deposits', 'Deposits']
                                 if item in balance.index), None)

                equity = next((balance[item] for item in ['Total Equity', 'Total Stockholder Equity']
                               if item in balance.index), None)

                # Calculate net income
                net_income = next((income[item] for item in ['Net Income', 'Net Income Common Stockholders']
                                   if item in income.index), None)

                # Calculate provision for loan losses
                loan_loss_provision = next(
                    (income[item] for item in ['Provision for Loan Losses', 'Credit Loss Provision']
                     if item in income.index), 0)

                # Calculate key ratios
                period_metrics = {}

                # Net Interest Margin (NIM)
                if net_interest_income is not None and total_assets is not None and total_assets > 0:
                    period_metrics['NIM'] = net_interest_income / total_assets

                # Return on Assets (ROA)
                if net_income is not None and total_assets is not None and total_assets > 0:
                    period_metrics['ROA'] = net_income / total_assets

                # Return on Equity (ROE)
                if net_income is not None and equity is not None and equity > 0:
                    period_metrics['ROE'] = net_income / equity

                # Loan-to-Deposit Ratio
                if loans is not None and deposits is not None and deposits > 0:
                    period_metrics['Loan_to_Deposit'] = loans / deposits

                # Capital Adequacy (Equity/Assets)
                if equity is not None and total_assets is not None and total_assets > 0:
                    period_metrics['Capital_Adequacy'] = equity / total_assets

                # Provision Rate
                if loan_loss_provision is not None and loans is not None and loans > 0:
                    period_metrics['Provision_Rate'] = loan_loss_provision / loans

                # Efficiency Ratio (if we can find non-interest expense)
                non_interest_expense = next((income[item] for item in ['Non-Interest Expense', 'Operating Expense']
                                             if item in income.index), None)

                if non_interest_expense is not None and net_interest_income is not None:
                    # Try to find non-interest income
                    non_interest_income = next((income[item] for item in ['Non-Interest Income', 'Fee Income']
                                                if item in income.index), 0)

                    total_income = net_interest_income + non_interest_income

                    if total_income > 0:
                        period_metrics['Efficiency_Ratio'] = non_interest_expense / total_income

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                for key in metrics["Latest"]:
                    if key in metrics["Year-1"] and metrics["Year-1"][key] > 0:
                        growth = (metrics["Latest"][key] / metrics["Year-1"][key]) - 1
                        growth_metrics[f"{key}_Growth"] = growth

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating banking metrics: {e}")
            return {}

    def _calculate_insurance_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key insurance metrics for valuation"""
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

                # Find premium income
                premium_income = next(
                    (income[item] for item in ['Premium Income', 'Net Premiums Earned', 'Insurance Revenue']
                     if item in income.index), None)

                # Find claims and expenses
                claims = next((income[item] for item in ['Claims', 'Insurance Claims', 'Benefits and Claims']
                               if item in income.index), None)

                underwriting_expenses = next(
                    (income[item] for item in ['Underwriting Expenses', 'Policy Acquisition Costs']
                     if item in income.index), None)

                # Find investment income
                investment_income = next((income[item] for item in ['Investment Income', 'Net Investment Income']
                                          if item in income.index), None)

                # Find total assets, reserves, and equity
                total_assets = next((balance[item] for item in ['Total Assets'] if item in balance.index), None)

                reserves = next(
                    (balance[item] for item in ['Insurance Reserves', 'Policy Reserves', 'Technical Provisions']
                     if item in balance.index), None)

                equity = next((balance[item] for item in ['Total Equity', 'Total Stockholder Equity']
                               if item in balance.index), None)

                # Calculate net income
                net_income = next((income[item] for item in ['Net Income', 'Net Income Common Stockholders']
                                   if item in income.index), None)

                # Calculate key ratios
                period_metrics = {}

                # Combined Ratio
                if premium_income is not None and premium_income > 0:
                    # If we have both claims and expenses
                    if claims is not None and underwriting_expenses is not None:
                        period_metrics['Combined_Ratio'] = (claims + underwriting_expenses) / premium_income
                    # If we only have claims
                    elif claims is not None:
                        # Estimate underwriting expenses as a percentage of claims
                        period_metrics['Combined_Ratio'] = claims / premium_income * 1.3  # Rough estimate

                # Investment Yield
                if investment_income is not None and total_assets is not None and total_assets > 0:
                    period_metrics['Investment_Yield'] = investment_income / total_assets

                # Return on Equity (ROE)
                if net_income is not None and equity is not None and equity > 0:
                    period_metrics['ROE'] = net_income / equity

                # Reserve to Equity Ratio
                if reserves is not None and equity is not None and equity > 0:
                    period_metrics['Reserve_to_Equity'] = reserves / equity

                # Premium to Equity (Insurance Leverage)
                if premium_income is not None and equity is not None and equity > 0:
                    period_metrics['Premium_to_Equity'] = premium_income / equity

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                for key in metrics["Latest"]:
                    if key in metrics["Year-1"] and metrics["Year-1"][key] > 0:
                        growth = (metrics["Latest"][key] / metrics["Year-1"][key]) - 1
                        growth_metrics[f"{key}_Growth"] = growth

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating insurance metrics: {e}")
            return {}

    def _calculate_asset_manager_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key asset management metrics for valuation"""
        try:
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            company_info = financial_data.get('company_info', {})

            metrics = {}

            if income_stmt is None:
                return metrics

            # Estimate AUM if possible from company info or notes
            aum = company_info.get('AUM')  # This would need to be extracted from additional sources

            # Get data for multiple periods
            periods = min(income_stmt.shape[1], 3)  # Use up to 3 years

            # Calculate metrics for each period
            for i in range(periods):
                year = f"Year-{i}" if i > 0 else "Latest"
                income = income_stmt.iloc[:, i]

                # Find fee revenue
                fee_revenue = next((income[item] for item in ['Asset Management Fees', 'Management Fees', 'Fee Revenue']
                                    if item in income.index), None)

                # If not directly available, estimate from total revenue
                if fee_revenue is None and 'Total Revenue' in income.index:
                    fee_revenue = income['Total Revenue']

                # Find operating income and net income
                operating_income = next((income[item] for item in ['Operating Income', 'Income Before Tax']
                                         if item in income.index), None)

                net_income = next((income[item] for item in ['Net Income', 'Net Income Common Stockholders']
                                   if item in income.index), None)

                # Calculate key metrics
                period_metrics = {}

                # If we have AUM data
                if aum is not None:
                    # Fee Rate (Fee Revenue / AUM)
                    if fee_revenue is not None:
                        period_metrics['Fee_Rate'] = fee_revenue / aum

                # Profit Margin
                if net_income is not None and fee_revenue is not None and fee_revenue > 0:
                    period_metrics['Profit_Margin'] = net_income / fee_revenue

                # Operating Margin
                if operating_income is not None and fee_revenue is not None and fee_revenue > 0:
                    period_metrics['Operating_Margin'] = operating_income / fee_revenue

                metrics[year] = period_metrics

            # Calculate growth rates and trends
            if periods > 1 and "Latest" in metrics and "Year-1" in metrics:
                growth_metrics = {}

                for key in metrics["Latest"]:
                    if key in metrics["Year-1"] and metrics["Year-1"][key] > 0:
                        growth = (metrics["Latest"][key] / metrics["Year-1"][key]) - 1
                        growth_metrics[f"{key}_Growth"] = growth

                metrics["Growth"] = growth_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating asset manager metrics: {e}")
            return {}

    def _calculate_adjusted_book_value(self, latest_balance: pd.Series, banking_metrics: Dict[str, Any]) -> float:
        """Calculate adjusted book value for a bank with adjustments for loan quality and intangibles"""
        try:
            # Start with reported book value
            if 'Total Stockholder Equity' in latest_balance.index:
                book_value = latest_balance['Total Stockholder Equity']
            elif 'Total Equity' in latest_balance.index:
                book_value = latest_balance['Total Equity']
            else:
                # If equity is not directly available, calculate as assets minus liabilities
                if 'Total Assets' in latest_balance.index and 'Total Liabilities' in latest_balance.index:
                    book_value = latest_balance['Total Assets'] - latest_balance['Total Liabilities']
                else:
                    return 0  # Can't calculate book value

            # Start with reported book value
            adjusted_book_value = book_value

            # Adjust for loan quality
            loans = next((latest_balance[item] for item in ['Net Loans', 'Loans', 'Loans Receivable']
                          if item in latest_balance.index), None)

            if loans is not None:
                # Get provision rate or use default
                latest_metrics = banking_metrics.get('Latest', {})
                provision_rate = latest_metrics.get('Provision_Rate', 0.01)  # Default 1%

                # Adjust for potential additional loan losses
                # Higher provision rates indicate higher risk, so add more buffer
                additional_loan_loss = loans * provision_rate * 0.5
                adjusted_book_value -= additional_loan_loss

            # Adjust for goodwill and intangibles
            goodwill = next((latest_balance[item] for item in ['Goodwill', 'Goodwill and Intangible Assets']
                             if item in latest_balance.index), 0)

            intangibles = next((latest_balance[item] for item in ['Intangible Assets', 'Other Intangible Assets']
                                if item in latest_balance.index), 0)
            # For banks, we often discount goodwill and intangibles
            # Because in financial stress, these assets may have less value
            goodwill_adjustment = goodwill * 0.5  # Discount goodwill by 50%
            intangibles_adjustment = intangibles * 0.7  # Discount intangibles by 30%
            adjusted_book_value = adjusted_book_value - goodwill_adjustment - intangibles_adjustment

            # Adjust for off-balance sheet items
            # Since we don't have detailed data, use a conservative estimate
            off_balance_sheet_adjustment = 0
            if 'Total Assets' in latest_balance.index:
                # Apply small adjustment based on total assets
                off_balance_sheet_adjustment = latest_balance['Total Assets'] * 0.01  # 1% of assets
                adjusted_book_value -= off_balance_sheet_adjustment

            return max(adjusted_book_value, 0)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error calculating adjusted book value: {e}")
            # Return unadjusted book value as fallback
            if 'Total Stockholder Equity' in latest_balance.index:
                return latest_balance['Total Stockholder Equity']
            elif 'Total Equity' in latest_balance.index:
                return latest_balance['Total Equity']
            return 0

    def _bank_dividend_model(self, ticker: str, financial_data: Dict[str, Any],
                             banking_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dividend discount model valuation for banks with sustainable ROE approach"""
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

            # Get bank metrics
            latest_metrics = banking_metrics.get('Latest', {})
            growth_metrics = banking_metrics.get('Growth', {})

            # Calculate key inputs for DDM
            # 1. Current book value per share
            if 'Total Stockholder Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Stockholder Equity'] / shares_outstanding
            elif 'Total Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Equity'] / shares_outstanding
            else:
                book_value_per_share = None

            # 2. Sustainable ROE
            # Use reported ROE or calculate it
            roe = latest_metrics.get('ROE')
            if roe is None:
                # Calculate ROE if not provided
                if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                    if latest_balance['Total Stockholder Equity'] > 0:
                        roe = latest_income['Net Income'] / latest_balance['Total Stockholder Equity']
                    else:
                        roe = 0.10  # Default assumption
                else:
                    roe = 0.10  # Default assumption

            # 3. Dividend payout ratio
            # Try to find from cash flow statement
            payout_ratio = None
            if cash_flow is not None and not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                if 'Dividends Paid' in latest_cash_flow.index and 'Net Income' in latest_income.index:
                    if latest_income['Net Income'] > 0:
                        payout_ratio = abs(latest_cash_flow['Dividends Paid']) / latest_income['Net Income']

            # If not found, use default or estimate from industry average
            if payout_ratio is None:
                payout_ratio = 0.40  # Banks typically pay out 30-50% of earnings

            # 4. Retention rate and growth rate
            retention_rate = 1 - payout_ratio
            sustainable_growth_rate = roe * retention_rate

            # Cap growth rate to reasonable bounds
            sustainable_growth_rate = min(max(sustainable_growth_rate, 0.01), 0.15)

            # 5. Calculate discount rate (cost of equity)
            # Use CAPM or Banking sector specific rate
            beta = market_data.get('beta', 1.2)  # Default beta for banks
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            discount_rate = risk_free_rate + beta * equity_risk_premium

            # Ensure discount rate is reasonable
            discount_rate = max(0.08, min(0.18, discount_rate))  # Between 8% and 18%

            # Ensure discount rate > growth rate
            if discount_rate <= sustainable_growth_rate:
                discount_rate = sustainable_growth_rate + 0.04  # Minimum spread of 4%

            # 6. Current EPS
            if 'Net Income' in latest_income.index and shares_outstanding and shares_outstanding > 0:
                eps = latest_income['Net Income'] / shares_outstanding
            else:
                # If can't calculate EPS, use book value and ROE
                if book_value_per_share:
                    eps = book_value_per_share * roe
                else:
                    # Can't calculate value
                    return {
                        'method': 'bank_ddm',
                        'value_per_share': None,
                        'error': 'Cannot calculate EPS'
                    }

            # 7. Current dividend per share
            dps = eps * payout_ratio

            # 8. Build DDM model
            # For banks, typically use a 3-stage model:
            # - Initial growth phase (based on sustainable growth)
            # - Transition phase (declining to terminal growth)
            # - Terminal phase

            # DDM parameters
            initial_phase_years = 5
            transition_phase_years = 5
            terminal_growth = 0.03  # Long-term growth rate

            # Calculate present value of dividends
            present_value = 0

            # Initial growth phase
            current_dps = dps
            for year in range(1, initial_phase_years + 1):
                projected_dps = current_dps * (1 + sustainable_growth_rate) ** year
                present_value += projected_dps / ((1 + discount_rate) ** year)

            # Last DPS from initial phase
            last_initial_dps = current_dps * (1 + sustainable_growth_rate) ** initial_phase_years

            # Transition phase with declining growth
            for year in range(1, transition_phase_years + 1):
                # Linear decline in growth rate
                growth_rate = sustainable_growth_rate - ((sustainable_growth_rate - terminal_growth) *
                                                         year / transition_phase_years)
                projected_dps = last_initial_dps * (1 + growth_rate) ** year
                present_value += projected_dps / ((1 + discount_rate) ** (year + initial_phase_years))

            # Last DPS from transition phase
            last_transition_dps = last_initial_dps * (1 + terminal_growth)

            # Terminal value using Gordon Growth model
            terminal_value = last_transition_dps * (1 + terminal_growth) / (discount_rate - terminal_growth)
            present_value_terminal = terminal_value / (
                        (1 + discount_rate) ** (initial_phase_years + transition_phase_years))

            # Add terminal value to present value
            total_present_value = present_value + present_value_terminal

            return {
                'method': 'bank_ddm',
                'value_per_share': total_present_value,
                'current_eps': eps,
                'current_dps': dps,
                'book_value_per_share': book_value_per_share,
                'sustainable_roe': roe,
                'payout_ratio': payout_ratio,
                'retention_rate': retention_rate,
                'sustainable_growth_rate': sustainable_growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'initial_phase_years': initial_phase_years,
                'transition_phase_years': transition_phase_years,
                'present_value_dividends': present_value,
                'present_value_terminal': present_value_terminal
            }

        except Exception as e:
            logger.error(f"Error in bank dividend model for {ticker}: {e}")
            return {
                'method': 'bank_ddm',
                'value_per_share': None,
                'error': str(e)
            }

    def _residual_income_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                   banking_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform residual income (excess return) valuation for banks
        This model is particularly suitable for financial institutions
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

            # Get bank metrics
            latest_metrics = banking_metrics.get('Latest', {})

            # 1. Calculate current book value per share
            if 'Total Stockholder Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Stockholder Equity'] / shares_outstanding
            elif 'Total Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Equity'] / shares_outstanding
            else:
                return {
                    'method': 'residual_income',
                    'value_per_share': None,
                    'error': 'Cannot calculate book value per share'
                }

            # 2. Calculate cost of equity (discount rate)
            beta = market_data.get('beta', 1.2)  # Default beta for banks
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # Ensure cost of equity is reasonable
            cost_of_equity = max(0.08, min(0.18, cost_of_equity))  # Between 8% and 18%

            # 3. Calculate ROE
            roe = latest_metrics.get('ROE')
            if roe is None:
                # Calculate ROE if not provided
                if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                    if latest_balance['Total Stockholder Equity'] > 0:
                        roe = latest_income['Net Income'] / latest_balance['Total Stockholder Equity']
                    else:
                        roe = 0.10  # Default assumption
                else:
                    roe = 0.10  # Default assumption

            # 4. Calculate excess ROE (ROE - Cost of Equity)
            excess_roe = roe - cost_of_equity

            # 5. Forecast future excess returns
            forecast_years = 10  # Typically longer for RI model

            # Determine growth rate in book value
            # This depends on retention ratio
            if 'Dividends Paid' in financial_data.get('cash_flow', pd.DataFrame()).iloc[:, 0].index:
                dividends_paid = abs(financial_data['cash_flow'].iloc[:, 0]['Dividends Paid'])
                if 'Net Income' in latest_income.index and latest_income['Net Income'] > 0:
                    payout_ratio = dividends_paid / latest_income['Net Income']
                    retention_rate = 1 - payout_ratio
                else:
                    retention_rate = 0.6  # Default assumption
            else:
                retention_rate = 0.6  # Default assumption for banks

            # Growth in book value = ROE * Retention Rate
            book_value_growth = roe * retention_rate

            # Cap growth rate to reasonable bounds
            book_value_growth = min(max(book_value_growth, 0.01), 0.15)

            # Initial excess returns fade over time to a sustainable level
            # This is normal as competition reduces excess returns
            initial_excess_roe = excess_roe
            terminal_excess_roe = excess_roe * 0.3  # Long-term excess returns are typically lower

            # Calculate present value of all future excess returns
            present_value_excess_returns = 0
            current_book_value = book_value_per_share

            for year in range(1, forecast_years + 1):
                # Book value grows each year by retained earnings
                current_book_value *= (1 + book_value_growth)

                # Excess ROE declines over time (linear fade)
                current_excess_roe = initial_excess_roe - ((initial_excess_roe - terminal_excess_roe) *
                                                           year / forecast_years)

                # Excess return for the year
                excess_return = current_book_value * current_excess_roe

                # Discount to present value
                present_value_excess_returns += excess_return / ((1 + cost_of_equity) ** year)

            # Terminal value of excess returns (if any persist beyond forecast period)
            if terminal_excess_roe > 0:
                final_book_value = current_book_value
                final_excess_return = final_book_value * terminal_excess_roe

                terminal_value = final_excess_return / (cost_of_equity - book_value_growth)
                present_value_terminal = terminal_value / ((1 + cost_of_equity) ** forecast_years)

                present_value_excess_returns += present_value_terminal

            # 6. Final value = Current Book Value + PV of Future Excess Returns
            value_per_share = book_value_per_share + present_value_excess_returns

            return {
                'method': 'residual_income',
                'value_per_share': value_per_share,
                'book_value_per_share': book_value_per_share,
                'roe': roe,
                'cost_of_equity': cost_of_equity,
                'excess_roe': excess_roe,
                'retention_rate': retention_rate,
                'book_value_growth': book_value_growth,
                'present_value_excess_returns': present_value_excess_returns,
                'forecast_years': forecast_years
            }

        except Exception as e:
            logger.error(f"Error in residual income valuation for {ticker}: {e}")
            return {
                'method': 'residual_income',
                'value_per_share': None,
                'error': str(e)
            }

    def _calculate_insurance_adjusted_book_value(self, latest_balance: pd.Series,
                                                 insurance_metrics: Dict[str, Any]) -> float:
        """Calculate adjusted book value for an insurance company with reserve adjustments"""
        try:
            # Start with reported book value
            if 'Total Stockholder Equity' in latest_balance.index:
                book_value = latest_balance['Total Stockholder Equity']
            elif 'Total Equity' in latest_balance.index:
                book_value = latest_balance['Total Equity']
            else:
                # If equity is not directly available, calculate as assets minus liabilities
                if 'Total Assets' in latest_balance.index and 'Total Liabilities' in latest_balance.index:
                    book_value = latest_balance['Total Assets'] - latest_balance['Total Liabilities']
                else:
                    return 0  # Can't calculate book value

            # Insurance companies' value is highly sensitive to reserve adequacy
            adjusted_book_value = book_value

            # Get insurance reserves
            reserves = next(
                (latest_balance[item] for item in ['Insurance Reserves', 'Policy Reserves', 'Technical Provisions']
                 if item in latest_balance.index), None)

            # If we have reserve data, adjust for potential reserve adequacy issues
            if reserves is not None:
                # Latest metrics
                latest_metrics = insurance_metrics.get('Latest', {})

                # Combined ratio is a key metric for reserve adequacy
                combined_ratio = latest_metrics.get('Combined_Ratio')

                if combined_ratio:
                    # Adjust reserves based on combined ratio
                    # Higher combined ratio indicates potential underreserving
                    if combined_ratio > 1.0:
                        # Potential reserve deficiency, add additional buffer
                        reserve_adjustment = reserves * (combined_ratio - 1.0) * 0.5
                        adjusted_book_value -= reserve_adjustment
                    elif combined_ratio < 0.95:
                        # Potential reserve redundancy, but be conservative
                        reserve_adjustment = reserves * (0.95 - combined_ratio) * 0.2
                        adjusted_book_value += reserve_adjustment
                else:
                    # If combined ratio not available, apply a small conservative adjustment
                    reserve_adjustment = reserves * 0.03  # 3% safety margin
                    adjusted_book_value -= reserve_adjustment

            # Adjust for goodwill and intangibles
            goodwill = next((latest_balance[item] for item in ['Goodwill', 'Goodwill and Intangible Assets']
                             if item in latest_balance.index), 0)

            intangibles = next((latest_balance[item] for item in ['Intangible Assets', 'Other Intangible Assets']
                                if item in latest_balance.index), 0)

            # For insurance companies, discount intangibles
            goodwill_adjustment = goodwill * 0.6  # Discount goodwill by 60%
            intangibles_adjustment = intangibles * 0.7  # Discount intangibles by 30%
            adjusted_book_value = adjusted_book_value - goodwill_adjustment - intangibles_adjustment

            # Adjust for deferred acquisition costs (DAC)
            dac = next((latest_balance[item] for item in ['Deferred Acquisition Costs', 'DAC']
                        if item in latest_balance.index), 0)

            dac_adjustment = dac * 0.2  # Discount DAC by 20%
            adjusted_book_value -= dac_adjustment

            return max(adjusted_book_value, 0)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error calculating insurance adjusted book value: {e}")
            # Return unadjusted book value as fallback
            if 'Total Stockholder Equity' in latest_balance.index:
                return latest_balance['Total Stockholder Equity']
            elif 'Total Equity' in latest_balance.index:
                return latest_balance['Total Equity']
            return 0

    def _insurance_dividend_model(self, ticker: str, financial_data: Dict[str, Any],
                                  insurance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dividend discount model valuation for insurance companies"""
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

            # Get insurance metrics
            latest_metrics = insurance_metrics.get('Latest', {})

            # Calculate key inputs for DDM
            # 1. Current book value per share
            if 'Total Stockholder Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Stockholder Equity'] / shares_outstanding
            elif 'Total Equity' in latest_balance.index and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = latest_balance['Total Equity'] / shares_outstanding
            else:
                book_value_per_share = None

            # 2. Sustainable ROE
            # Use reported ROE or calculate it
            roe = latest_metrics.get('ROE')
            if roe is None:
                # Calculate ROE if not provided
                if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                    if latest_balance['Total Stockholder Equity'] > 0:
                        roe = latest_income['Net Income'] / latest_balance['Total Stockholder Equity']
                    else:
                        roe = 0.10  # Default assumption
                else:
                    roe = 0.10  # Default assumption

            # Adjust ROE based on combined ratio
            combined_ratio = latest_metrics.get('Combined_Ratio')
            if combined_ratio:
                # Higher combined ratio reduces sustainable ROE
                if combined_ratio > 1.0:
                    roe_adjustment = (combined_ratio - 1.0) * 0.5
                    roe = max(roe - roe_adjustment, 0.05)  # Minimum 5% ROE
                elif combined_ratio < 0.95:
                    roe_adjustment = (0.95 - combined_ratio) * 0.3
                    roe = min(roe + roe_adjustment, 0.20)  # Maximum 20% ROE

            # 3. Dividend payout ratio
            # Try to find from cash flow statement
            payout_ratio = None
            if cash_flow is not None and not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[:, 0]
                if 'Dividends Paid' in latest_cash_flow.index and 'Net Income' in latest_income.index:
                    if latest_income['Net Income'] > 0:
                        payout_ratio = abs(latest_cash_flow['Dividends Paid']) / latest_income['Net Income']

            # If not found, use default or estimate from industry average
            if payout_ratio is None:
                payout_ratio = 0.45  # Insurance companies typically have higher payouts

            # 4. Retention rate and growth rate
            retention_rate = 1 - payout_ratio
            sustainable_growth_rate = roe * retention_rate

            # Cap growth rate to reasonable bounds
            sustainable_growth_rate = min(max(sustainable_growth_rate, 0.01), 0.12)

            # 5. Calculate discount rate (cost of equity)
            beta = market_data.get('beta', 1.0)  # Default beta for insurance
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            discount_rate = risk_free_rate + beta * equity_risk_premium

            # Ensure discount rate is reasonable
            discount_rate = max(0.07, min(0.15, discount_rate))  # Between 7% and 15%

            # Ensure discount rate > growth rate
            if discount_rate <= sustainable_growth_rate:
                discount_rate = sustainable_growth_rate + 0.03  # Minimum spread of 3%

            # 6. Current EPS
            if 'Net Income' in latest_income.index and shares_outstanding and shares_outstanding > 0:
                eps = latest_income['Net Income'] / shares_outstanding
            else:
                # If can't calculate EPS, use book value and ROE
                if book_value_per_share:
                    eps = book_value_per_share * roe
                else:
                    # Can't calculate value
                    return {
                        'method': 'insurance_ddm',
                        'value_per_share': None,
                        'error': 'Cannot calculate EPS'
                    }

            # 7. Current dividend per share
            dps = eps * payout_ratio

            # 8. Build DDM model
            # For insurance, typically use a 2-stage model:
            # - Growth phase (based on sustainable growth)
            # - Terminal phase

            # DDM parameters
            growth_phase_years = 10
            terminal_growth = 0.03  # Long-term growth rate

            # Calculate present value of dividends
            present_value = 0

            # Growth phase
            current_dps = dps
            for year in range(1, growth_phase_years + 1):
                projected_dps = current_dps * (1 + sustainable_growth_rate) ** year
                present_value += projected_dps / ((1 + discount_rate) ** year)

            # Terminal phase
            terminal_dps = current_dps * (1 + sustainable_growth_rate) ** growth_phase_years * (1 + terminal_growth)
            terminal_value = terminal_dps / (discount_rate - terminal_growth)
            present_value_terminal = terminal_value / ((1 + discount_rate) ** growth_phase_years)

            # Total present value
            total_present_value = present_value + present_value_terminal

            return {
                'method': 'insurance_ddm',
                'value_per_share': total_present_value,
                'current_eps': eps,
                'current_dps': dps,
                'book_value_per_share': book_value_per_share,
                'sustainable_roe': roe,
                'combined_ratio': combined_ratio,
                'payout_ratio': payout_ratio,
                'retention_rate': retention_rate,
                'sustainable_growth_rate': sustainable_growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'growth_phase_years': growth_phase_years,
                'present_value_dividends': present_value,
                'present_value_terminal': present_value_terminal
            }

        except Exception as e:
            logger.error(f"Error in insurance dividend model for {ticker}: {e}")
            return {
                'method': 'insurance_ddm',
                'value_per_share': None,
                'error': str(e)
            }

    def _insurance_appraisal_value(self, ticker: str, financial_data: Dict[str, Any],
                                   insurance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate appraisal value (embedded value + new business value) for insurance companies
        Note: This is a simplified approach since true embedded value requires actuarial data
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

            # Get insurance metrics
            latest_metrics = insurance_metrics.get('Latest', {})

            # Step 1: Calculate adjusted net asset value (ANAV)
            # This is similar to adjusted book value
            if 'Total Stockholder Equity' in latest_balance.index:
                anav = latest_balance['Total Stockholder Equity']
            elif 'Total Equity' in latest_balance.index:
                anav = latest_balance['Total Equity']
            else:
                # If equity is not directly available, calculate as assets minus liabilities
                if 'Total Assets' in latest_balance.index and 'Total Liabilities' in latest_balance.index:
                    anav = latest_balance['Total Assets'] - latest_balance['Total Liabilities']
                else:
                    return {
                        'method': 'insurance_appraisal',
                        'value_per_share': None,
                        'error': 'Cannot calculate net asset value'
                    }

            # Adjust for intangibles
            goodwill = next((latest_balance[item] for item in ['Goodwill', 'Goodwill and Intangible Assets']
                             if item in latest_balance.index), 0)

            intangibles = next((latest_balance[item] for item in ['Intangible Assets', 'Other Intangible Assets']
                                if item in latest_balance.index), 0)

            anav = anav - goodwill - intangibles

            # Step 2: Estimate present value of future profits (PVFP) from in-force business
            # Simplified approach using ROE and retention rate

            # Get ROE
            roe = latest_metrics.get('ROE')
            if roe is None:
                # Calculate ROE if not provided
                if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                    if latest_balance['Total Stockholder Equity'] > 0:
                        roe = latest_income['Net Income'] / latest_balance['Total Stockholder Equity']
                    else:
                        roe = 0.10  # Default assumption
                else:
                    roe = 0.10  # Default assumption

            # Find premium income as a base for in-force business
            premium_income = next(
                (latest_income[item] for item in ['Premium Income', 'Net Premiums Earned', 'Insurance Revenue']
                 if item in latest_income.index), None)

            if premium_income is None:
                # Estimate from total revenue
                if 'Total Revenue' in latest_income.index:
                    premium_income = latest_income['Total Revenue'] * 0.8  # Assume 80% is premium
                else:
                    premium_income = anav * 0.5  # Rough estimate based on equity

            # Use combined ratio to estimate profit margin on in-force business
            combined_ratio = latest_metrics.get('Combined_Ratio', 0.95)  # Default 95%

            # Profit margin from underwriting
            underwriting_margin = 1 - combined_ratio

            # Add investment income (simplified estimate)
            investment_yield = latest_metrics.get('Investment_Yield', 0.03)  # Default 3%

            # Estimate investment assets
            investment_assets = next((latest_balance[item] for item in ['Invested Assets', 'Total Investments']
                                      if item in latest_balance.index), None)

            if investment_assets is None:
                # Estimate from total assets
                if 'Total Assets' in latest_balance.index:
                    investment_assets = latest_balance['Total Assets'] * 0.7  # Assume 70% is invested
                else:
                    investment_assets = anav * 2  # Rough estimate
            investment_income = investment_assets * investment_yield

            # Total profit from in-force business
            in_force_profit = (premium_income * underwriting_margin) + investment_income

            # Apply cost of capital charge
            capital_charge_rate = 0.06  # 6% cost of capital
            capital_charge = anav * capital_charge_rate

            adjusted_in_force_profit = in_force_profit - capital_charge

            # Calculate present value of future profits
            # For simplicity, assume declining profits over 10 years
            forecast_years = 10
            discount_rate = 0.09  # Typical discount rate for insurance PVs

            pvfp = 0
            current_profit = adjusted_in_force_profit
            profit_decline_rate = 0.1  # 10% annual decline in profit from in-force business

            for year in range(1, forecast_years + 1):
                current_profit *= (1 - profit_decline_rate)
                pvfp += current_profit / ((1 + discount_rate) ** year)

            # Step 3: Estimate value of new business (VNB)
            # Typically calculated as a multiple of one year's new business contribution

            # Estimate new business premium (NBP)
            premium_growth = 0.05  # Assume 5% growth in premiums
            new_business_premium = premium_income * premium_growth

            # Assume new business margin
            new_business_margin = 0.02  # 2% margin on new business

            # One year's new business contribution
            new_business_contribution = new_business_premium * new_business_margin

            # Apply multiple to get value of new business
            new_business_multiple = 10  # Typical range: 8-12x
            vnb = new_business_contribution * new_business_multiple

            # Step 4: Calculate total appraisal value
            appraisal_value = anav + pvfp + vnb

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                appraisal_value_per_share = appraisal_value / shares_outstanding
            else:
                appraisal_value_per_share = None

            return {
                'method': 'insurance_appraisal',
                'value_per_share': appraisal_value_per_share,
                'total_appraisal_value': appraisal_value,
                'adjusted_net_asset_value': anav,
                'present_value_future_profits': pvfp,
                'value_new_business': vnb,
                'combined_ratio': combined_ratio,
                'investment_yield': investment_yield,
                'premium_income': premium_income,
                'forecast_years': forecast_years
            }

        except Exception as e:
            logger.error(f"Error in insurance appraisal valuation for {ticker}: {e}")
            return {
                'method': 'insurance_appraisal',
                'value_per_share': None,
                'error': str(e)
            }

    def _aum_based_valuation(self, ticker: str, financial_data: Dict[str, Any],
                             am_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Value an asset management company based on AUM (Assets Under Management)
        This approach is widely used for asset managers and wealth management firms
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest income data
            latest_income = income_stmt.iloc[:, 0]

            # Get company info to find AUM if available
            company_info = financial_data.get('company_info', {})

            # Try to find AUM (Assets Under Management)
            # This would typically need to be extracted from company reports or other sources
            aum = company_info.get('AUM')

            # If AUM not directly available, estimate it from revenue
            if aum is None:
                # Find fee revenue
                fee_revenue = next((latest_income[item] for item in
                                    ['Asset Management Fees', 'Management Fees', 'Fee Revenue']
                                    if item in latest_income.index), None)

                # If fee revenue not available, use total revenue
                if fee_revenue is None and 'Total Revenue' in latest_income.index:
                    fee_revenue = latest_income['Total Revenue']
                elif fee_revenue is None:
                    return {
                        'method': 'aum_based',
                        'value_per_share': None,
                        'error': 'Cannot determine fee revenue or AUM'
                    }

                # Estimate AUM based on typical fee rates
                estimated_fee_rate = am_metrics.get('Latest', {}).get('Fee_Rate', 0.007)  # Default 0.7%

                if estimated_fee_rate > 0:
                    aum = fee_revenue / estimated_fee_rate
                else:
                    # Typical industry revenue-to-AUM ratios
                    aum = fee_revenue / 0.007  # Typical average fee rate

            # Calculate different valuation multiples based on AUM
            # These multiples vary by type of assets, client mix, and profitability

            # Get operating margins if available
            operating_margin = am_metrics.get('Latest', {}).get('Operating_Margin')

            # Determine appropriate AUM multiple based on margins and asset mix
            if operating_margin:
                if operating_margin > 0.35:  # High margin (alternatives, active management)
                    aum_multiple = 0.04  # 4% of AUM
                elif operating_margin > 0.25:  # Good margin (mixed active/passive)
                    aum_multiple = 0.03  # 3% of AUM
                elif operating_margin > 0.15:  # Average margin (retail focused)
                    aum_multiple = 0.02  # 2% of AUM
                else:  # Low margin (passive, institutional)
                    aum_multiple = 0.015  # 1.5% of AUM
            else:
                # Default to average multiple
                aum_multiple = 0.025  # 2.5% of AUM

            # Calculate enterprise value
            enterprise_value = aum * aum_multiple

            # Adjust for net cash/debt
            net_cash = 0
            balance_sheet = financial_data.get('balance_sheet')

            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                # Find cash and debt items
                cash = next((latest_balance[item] for item in
                             ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                             if item in latest_balance.index), 0)

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                             if item in latest_balance.index), 0)

                net_cash = cash - debt

            # Calculate equity value
            equity_value = enterprise_value + net_cash

            # Calculate per share value
            if shares_outstanding and shares_outstanding > 0:
                value_per_share = equity_value / shares_outstanding
            else:
                value_per_share = None

            return {
                'method': 'aum_based',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'aum': aum,
                'aum_multiple': aum_multiple,
                'operating_margin': operating_margin,
                'net_cash': net_cash
            }

        except Exception as e:
            logger.error(f"Error in AUM-based valuation for {ticker}: {e}")
            return {
                'method': 'aum_based',
                'value_per_share': None,
                'error': str(e)
            }

    def _fee_based_dcf(self, ticker: str, financial_data: Dict[str, Any],
                       am_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform DCF valuation specifically adapted for asset managers
        with focus on fee revenue and operating margins
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

            # Parameters for DCF
            forecast_years = 10
            terminal_growth = 0.03  # Long-term growth

            # 1. Determine starting fee revenue
            fee_revenue = next((latest_income[item] for item in
                                ['Asset Management Fees', 'Management Fees', 'Fee Revenue']
                                if item in latest_income.index), None)

            # If fee revenue not directly available, use total revenue
            if fee_revenue is None and 'Total Revenue' in latest_income.index:
                fee_revenue = latest_income['Total Revenue']
            elif fee_revenue is None:
                return {
                    'method': 'fee_based_dcf',
                    'value_per_share': None,
                    'error': 'Cannot determine fee revenue'
                }

            # 2. Determine operating margin
            operating_margin = am_metrics.get('Latest', {}).get('Operating_Margin')

            if operating_margin is None:
                # Calculate operating margin if not provided
                if 'Operating Income' in latest_income.index:
                    operating_margin = latest_income['Operating Income'] / fee_revenue
                else:
                    # Default assumption
                    operating_margin = 0.25  # 25% is typical for asset managers

            # 3. Determine tax rate
            effective_tax_rate = 0.25  # Default 25%
            if 'Income Tax Expense' in latest_income.index and 'Income Before Tax' in latest_income.index:
                if latest_income['Income Before Tax'] > 0:
                    effective_tax_rate = latest_income['Income Tax Expense'] / latest_income['Income Before Tax']
                    effective_tax_rate = min(0.40, max(0.15, effective_tax_rate))  # Sanity check

            # 4. Determine growth rates
            # Asset management growth is driven by:
            # a) Market appreciation
            # b) Net flows (new assets minus redemptions)
            # c) Fee compression (typical in the industry)

            # Initial growth rate
            initial_growth_rate = 0.06  # Default 6%

            # If we have historical data, calculate growth trend
            if income_stmt.shape[1] >= 2:
                previous_revenue = None
                for col in range(1, min(income_stmt.shape[1], 3)):
                    previous_period = income_stmt.iloc[:, col]
                    prev_fee_revenue = next((previous_period[item] for item in
                                             ['Asset Management Fees', 'Management Fees', 'Fee Revenue']
                                             if item in previous_period.index), None)

                    if prev_fee_revenue is None and 'Total Revenue' in previous_period.index:
                        prev_fee_revenue = previous_period['Total Revenue']

                    if prev_fee_revenue and prev_fee_revenue > 0:
                        previous_revenue = prev_fee_revenue
                        break

                if previous_revenue:
                    historical_growth = (fee_revenue / previous_revenue) - 1
                    initial_growth_rate = min(0.15, max(0.0, historical_growth))  # Cap between 0% and 15%

            # 5. Calculate discount rate (WACC)
            beta = market_data.get('beta', 1.2)  # Default beta for asset managers
            risk_free_rate = RISK_FREE_RATE
            equity_risk_premium = 0.05  # Standard assumption

            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # Adjust for industry-specific risks
            cost_of_equity += 0.01  # Additional 1% for sector volatility

            # Calculate WACC
            # Asset managers typically have low debt levels
            if balance_sheet is not None:
                latest_balance = balance_sheet.iloc[:, 0]

                debt = next((latest_balance[item] for item in
                             ['Total Debt', 'Long Term Debt']
                             if item in latest_balance.index), 0)

                equity = next((latest_balance[item] for item in
                               ['Total Stockholder Equity', 'Total Equity']
                               if item in latest_balance.index), None)

                if equity and equity > 0:
                    debt_to_capital = debt / (debt + equity)
                    equity_to_capital = equity / (debt + equity)

                    cost_of_debt = 0.05  # Typical cost of debt
                    after_tax_cost_of_debt = cost_of_debt * (1 - effective_tax_rate)

                    wacc = (equity_to_capital * cost_of_equity) + (debt_to_capital * after_tax_cost_of_debt)
                else:
                    wacc = cost_of_equity
            else:
                wacc = cost_of_equity

            # Ensure WACC is reasonable
            wacc = max(0.08, min(0.15, wacc))  # Between 8% and 15%

            # 6. Forecast future cash flows
            forecasted_cash_flows = []
            current_revenue = fee_revenue
            current_growth_rate = initial_growth_rate

            for year in range(1, forecast_years + 1):
                # Revenue decreases over time
                growth_decay = (initial_growth_rate - terminal_growth) * (year / forecast_years)
                year_growth_rate = initial_growth_rate - growth_decay

                # Revenue for the year
                current_revenue = current_revenue * (1 + year_growth_rate)

                # Operating income (EBIT)
                # Include slight margin compression over time (industry trend)
                year_margin = operating_margin * (1 - 0.01 * year)  # 1% annual compression
                year_margin = max(0.15, year_margin)  # Floor at 15%

                operating_income = current_revenue * year_margin

                # Tax
                taxes = operating_income * effective_tax_rate

                # Depreciation (typically small for asset managers)
                depreciation = current_revenue * 0.01  # Assume 1% of revenue

                # Capital expenditures
                capex = depreciation * 1.2  # Slightly higher than depreciation

                # Changes in working capital
                working_capital_change = current_revenue * 0.01 * year_growth_rate  # Tied to growth

                # Free cash flow
                fcf = operating_income - taxes + depreciation - capex - working_capital_change
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
                'method': 'fee_based_dcf',
                'value_per_share': value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'fee_revenue': fee_revenue,
                'operating_margin': operating_margin,
                'initial_growth_rate': initial_growth_rate,
                'terminal_growth': terminal_growth,
                'wacc': wacc,
                'forecast_years': forecast_years,
                'forecasted_cash_flows': forecasted_cash_flows,
                'present_value_fcf': present_value_fcf,
                'present_value_terminal': present_value_terminal,
                'terminal_value': terminal_value
            }

        except Exception as e:
            logger.error(f"Error in fee-based DCF for {ticker}: {e}")
            return {
                'method': 'fee_based_dcf',
                'value_per_share': None,
                'error': str(e)
            }

    def _asset_manager_relative_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                          am_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform relative valuation for asset managers using industry-specific multiples
        """
        try:
            # Extract financial statements
            income_stmt = financial_data.get('income_statement')

            # Get market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding')

            # Get latest income data
            latest_income = income_stmt.iloc[:, 0]

            # Define industry multiples (these would ideally come from sector database)
            # These are typical multiples for asset management firms
            industry_multiples = {
                'PE': 15.0,  # Price to Earnings
                'PB': 2.5,  # Price to Book
                'PS': 3.0,  # Price to Sales
                'EV_EBITDA': 10.0  # EV to EBITDA
            }

            # Adjust multiples based on operating margin and growth
            operating_margin = am_metrics.get('Latest', {}).get('Operating_Margin')
            growth_metrics = am_metrics.get('Growth', {})

            # Apply adjustments to multiples
            if operating_margin:
                # Premium for high margin firms
                if operating_margin > 0.35:
                    multiple_adjustment = 1.3  # 30% premium
                elif operating_margin > 0.25:
                    multiple_adjustment = 1.15  # 15% premium
                elif operating_margin > 0.15:
                    multiple_adjustment = 1.0  # No adjustment
                else:
                    multiple_adjustment = 0.85  # 15% discount

                # Apply adjustment
                for key in industry_multiples:
                    industry_multiples[key] *= multiple_adjustment

            # Calculate valuation using different multiples
            valuations = {}

            # 1. P/E Valuation
            if 'Net Income' in latest_income.index and latest_income['Net Income'] > 0:
                earnings = latest_income['Net Income']
                pe_valuation = earnings * industry_multiples['PE']

                if shares_outstanding and shares_outstanding > 0:
                    pe_value_per_share = pe_valuation / shares_outstanding
                    valuations['PE'] = {
                        'value': pe_valuation,
                        'value_per_share': pe_value_per_share,
                        'multiple': industry_multiples['PE']
                    }

            # 2. P/S Valuation
            revenue = next((latest_income[item] for item in
                            ['Asset Management Fees', 'Management Fees', 'Fee Revenue', 'Total Revenue']
                            if item in latest_income.index), None)

            if revenue and revenue > 0:
                ps_valuation = revenue * industry_multiples['PS']

                if shares_outstanding and shares_outstanding > 0:
                    ps_value_per_share = ps_valuation / shares_outstanding
                    valuations['PS'] = {
                        'value': ps_valuation,
                        'value_per_share': ps_value_per_share,
                        'multiple': industry_multiples['PS']
                    }

            # 3. EV/EBITDA Valuation
            ebitda = None

            # Try to find EBITDA directly
            for item in ['EBITDA', 'Adjusted EBITDA']:
                if item in latest_income.index:
                    ebitda = latest_income[item]
                    break

            # If not found, calculate from components
            if ebitda is None:
                operating_income = next((latest_income[item] for item in
                                         ['Operating Income', 'EBIT']
                                         if item in latest_income.index), None)

                depreciation = next((latest_income[item] for item in
                                     ['Depreciation & Amortization', 'Depreciation and Amortization']
                                     if item in latest_income.index), None)

                if operating_income is not None:
                    if depreciation is not None:
                        ebitda = operating_income + depreciation
                    else:
                        # Estimate depreciation if not available
                        ebitda = operating_income * 1.1  # Assume D&A is 10% of operating income

            if ebitda and ebitda > 0:
                ev_ebitda_valuation = ebitda * industry_multiples['EV_EBITDA']

                # Adjust for net debt to get equity value
                net_debt = 0
                balance_sheet = financial_data.get('balance_sheet')

                if balance_sheet is not None:
                    latest_balance = balance_sheet.iloc[:, 0]

                    debt = next((latest_balance[item] for item in
                                 ['Total Debt', 'Long Term Debt', 'Short Term Debt']
                                 if item in latest_balance.index), 0)

                    cash = next((latest_balance[item] for item in
                                 ['Cash and Cash Equivalents', 'Cash and Short Term Investments']
                                 if item in latest_balance.index), 0)

                    net_debt = debt - cash

                equity_value_ebitda = ev_ebitda_valuation - net_debt

                if shares_outstanding and shares_outstanding > 0:
                    ebitda_value_per_share = equity_value_ebitda / shares_outstanding
                    valuations['EV_EBITDA'] = {
                        'value': equity_value_ebitda,
                        'value_per_share': ebitda_value_per_share,
                        'multiple': industry_multiples['EV_EBITDA'],
                        'net_debt': net_debt
                    }

            # 4. Calculate blended valuation
            if valuations:
                # Calculate average value per share
                total_value = 0
                total_weight = 0
                weights = {'PE': 0.4, 'PS': 0.3, 'EV_EBITDA': 0.3}

                for key, weight in weights.items():
                    if key in valuations and valuations[key].get('value_per_share') is not None:
                        total_value += valuations[key]['value_per_share'] * weight
                        total_weight += weight

                if total_weight > 0:
                    blended_value_per_share = total_value / total_weight
                else:
                    blended_value_per_share = None

                return {
                    'method': 'relative_valuation',
                    'value_per_share': blended_value_per_share,
                    'valuations': valuations,
                    'industry_multiples': industry_multiples
                }
            else:
                return {
                    'method': 'relative_valuation',
                    'value_per_share': None,
                    'error': 'Could not calculate valuations with available data'
                }

        except Exception as e:
            logger.error(f"Error in asset manager relative valuation for {ticker}: {e}")
            return {
                'method': 'relative_valuation',
                'value_per_share': None,
                'error': str(e)
            }