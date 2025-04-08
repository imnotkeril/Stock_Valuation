import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.config import SECTOR_MAPPING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('financial_statements')


class FinancialStatementAnalyzer:
    """
    Class for analyzing financial statements of companies.
    Provides functionality for analyzing balance sheets, income statements, and cash flow statements,
    including trend analysis, common-size analysis, and growth calculations.
    """

    def __init__(self):
        """Initialize the financial statement analyzer"""
        pass

    def analyze_income_statement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze income statement to identify trends and key metrics

        Args:
            df: DataFrame containing income statement data with columns as periods

        Returns:
            Dictionary with analysis results
        """
        if df is None or df.empty:
            logger.warning("Empty income statement provided for analysis")
            return {}

        try:
            # Convert to numeric data where possible
            df = df.apply(pd.to_numeric, errors='ignore')

            # Get the number of periods
            num_periods = df.shape[1]

            # Ensure we have at least 2 periods for trend analysis
            if num_periods < 2:
                logger.warning("Need at least 2 periods for income statement trend analysis")
                return self._analyze_single_period_income(df)

            # Calculate growth rates
            growth_rates = pd.DataFrame(index=df.index)
            for i in range(num_periods - 1):
                period_current = df.columns[i]
                period_prev = df.columns[i + 1]

                # Calculate percentage change
                pct_change = (df[period_current] - df[period_prev]) / df[period_prev].abs()
                growth_rates[f'{period_current} vs {period_prev}'] = pct_change

            # Calculate compound annual growth rate (CAGR) if we have at least 3 periods
            cagr = pd.Series(index=df.index)
            if num_periods >= 3:
                years = num_periods - 1  # Assume annual data
                first_period = df.columns[-1]
                last_period = df.columns[0]

                # CAGR = (End Value / Start Value)^(1/Years) - 1
                cagr = (df[last_period] / df[first_period]) ** (1 / years) - 1

            # Calculate common-size analysis (as percentage of revenue)
            common_size = pd.DataFrame(index=df.index)
            for col in df.columns:
                if 'Total Revenue' in df.index and df.loc['Total Revenue', col] != 0:
                    common_size[col] = df[col] / df.loc['Total Revenue', col]

            # Extract key metrics and trends
            key_metrics = {
                'revenue': self._extract_metric_trend(df, 'Total Revenue'),
                'gross_profit': self._extract_metric_trend(df, 'Gross Profit'),
                'operating_income': self._extract_metric_trend(df, 'Operating Income'),
                'net_income': self._extract_metric_trend(df, 'Net Income'),
                'margins': {
                    'gross_margin': self._calculate_margin_trend(df, 'Gross Profit', 'Total Revenue'),
                    'operating_margin': self._calculate_margin_trend(df, 'Operating Income', 'Total Revenue'),
                    'net_margin': self._calculate_margin_trend(df, 'Net Income', 'Total Revenue')
                },
                'growth_rates': {
                    'revenue_growth': self._extract_growth_rate(growth_rates, 'Total Revenue'),
                    'net_income_growth': self._extract_growth_rate(growth_rates, 'Net Income')
                },
                'cagr': {
                    'revenue_cagr': cagr.get('Total Revenue', None),
                    'net_income_cagr': cagr.get('Net Income', None)
                }
            }

            # Calculate and add efficiency metrics
            if 'Research and Development' in df.index and 'Total Revenue' in df.index:
                key_metrics['efficiency'] = {
                    'rnd_to_revenue': self._calculate_ratio_trend(df, 'Research and Development', 'Total Revenue')
                }

            return {
                'key_metrics': key_metrics,
                'growth_rates': growth_rates,
                'common_size': common_size,
                'cagr': cagr,
                'periods': list(df.columns)
            }
        except Exception as e:
            logger.error(f"Error analyzing income statement: {e}")
            return {}

    def analyze_balance_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze balance sheet to identify trends and key metrics

        Args:
            df: DataFrame containing balance sheet data with columns as periods

        Returns:
            Dictionary with analysis results
        """
        if df is None or df.empty:
            logger.warning("Empty balance sheet provided for analysis")
            return {}

        try:
            # Convert to numeric data where possible
            df = df.apply(pd.to_numeric, errors='ignore')

            # Get the number of periods
            num_periods = df.shape[1]

            # Ensure we have at least 2 periods for trend analysis
            if num_periods < 2:
                logger.warning("Need at least 2 periods for balance sheet trend analysis")
                return self._analyze_single_period_balance(df)

            # Calculate growth rates
            growth_rates = pd.DataFrame(index=df.index)
            for i in range(num_periods - 1):
                period_current = df.columns[i]
                period_prev = df.columns[i + 1]

                # Calculate percentage change
                pct_change = (df[period_current] - df[period_prev]) / df[period_prev].abs()
                growth_rates[f'{period_current} vs {period_prev}'] = pct_change

            # Calculate common-size analysis (as percentage of total assets)
            common_size = pd.DataFrame(index=df.index)
            for col in df.columns:
                if 'Total Assets' in df.index and df.loc['Total Assets', col] != 0:
                    common_size[col] = df[col] / df.loc['Total Assets', col]

            # Extract key metrics and trends
            key_metrics = {
                'assets': {
                    'total_assets': self._extract_metric_trend(df, 'Total Assets'),
                    'current_assets': self._extract_metric_trend(df, 'Total Current Assets'),
                    'cash_equivalents': self._extract_metric_trend(df, 'Cash and Cash Equivalents')
                },
                'liabilities': {
                    'total_liabilities': self._extract_metric_trend(df, 'Total Liabilities'),
                    'current_liabilities': self._extract_metric_trend(df, 'Total Current Liabilities'),
                    'long_term_debt': self._extract_metric_trend(df, 'Long Term Debt')
                },
                'equity': {
                    'total_equity': self._extract_metric_trend(df, 'Total Stockholder Equity')
                },
                'ratios': {
                    'current_ratio': self._calculate_ratio_trend(df, 'Total Current Assets',
                                                                 'Total Current Liabilities'),
                    'debt_to_equity': self._calculate_ratio_trend(df, 'Total Liabilities', 'Total Stockholder Equity'),
                    'debt_to_assets': self._calculate_ratio_trend(df, 'Total Liabilities', 'Total Assets')
                }
            }

            return {
                'key_metrics': key_metrics,
                'growth_rates': growth_rates,
                'common_size': common_size,
                'periods': list(df.columns)
            }
        except Exception as e:
            logger.error(f"Error analyzing balance sheet: {e}")
            return {}

    def analyze_cash_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cash flow statement to identify trends and key metrics

        Args:
            df: DataFrame containing cash flow data with columns as periods

        Returns:
            Dictionary with analysis results
        """
        if df is None or df.empty:
            logger.warning("Empty cash flow statement provided for analysis")
            return {}

        try:
            # Convert to numeric data where possible
            df = df.apply(pd.to_numeric, errors='ignore')

            # Get the number of periods
            num_periods = df.shape[1]

            # Ensure we have at least 2 periods for trend analysis
            if num_periods < 2:
                logger.warning("Need at least 2 periods for cash flow trend analysis")
                return self._analyze_single_period_cash_flow(df)

            # Calculate growth rates
            growth_rates = pd.DataFrame(index=df.index)
            for i in range(num_periods - 1):
                period_current = df.columns[i]
                period_prev = df.columns[i + 1]

                # Calculate percentage change (handle zeros and sign changes carefully)
                pct_change = (df[period_current] - df[period_prev]) / df[period_prev].abs()
                growth_rates[f'{period_current} vs {period_prev}'] = pct_change

            # Identify key metrics for cash flow analysis
            operating_cf_key = self._find_key_by_pattern(df, ['Operating Cash Flow', 'Net Cash from Operations'])
            investing_cf_key = self._find_key_by_pattern(df, ['Investing Cash Flow', 'Net Cash from Investing'])
            financing_cf_key = self._find_key_by_pattern(df, ['Financing Cash Flow', 'Net Cash from Financing'])
            capex_key = self._find_key_by_pattern(df, ['Capital Expenditure', 'CAPEX',
                                                       'Purchase of Property Plant and Equipment'])
            fcf_key = 'Free Cash Flow'  # May need to calculate this

            # Extract key metrics and trends
            key_metrics = {
                'operating_cash_flow': self._extract_metric_trend(df, operating_cf_key) if operating_cf_key else None,
                'investing_cash_flow': self._extract_metric_trend(df, investing_cf_key) if investing_cf_key else None,
                'financing_cash_flow': self._extract_metric_trend(df, financing_cf_key) if financing_cf_key else None,
                'capital_expenditure': self._extract_metric_trend(df, capex_key) if capex_key else None
            }

            # Calculate Free Cash Flow if not directly available
            if fcf_key not in df.index and operating_cf_key and capex_key:
                fcf = df.loc[operating_cf_key] - df.loc[capex_key]
                key_metrics['free_cash_flow'] = {
                    'values': fcf.to_dict(),
                    'latest': fcf.iloc[0],
                    'trend': self._calculate_trend(fcf)
                }
            else:
                key_metrics['free_cash_flow'] = self._extract_metric_trend(df, fcf_key)

            # Calculate cash flow quality metrics if possible
            if 'Net Income' in df.index and operating_cf_key:
                key_metrics['cf_quality'] = {
                    'ocf_to_net_income': self._calculate_ratio_trend(df, operating_cf_key, 'Net Income')
                }

            return {
                'key_metrics': key_metrics,
                'growth_rates': growth_rates,
                'periods': list(df.columns)
            }
        except Exception as e:
            logger.error(f"Error analyzing cash flow statement: {e}")
            return {}

    def calculate_financial_health_score(self, income_data: Dict, balance_data: Dict, cash_flow_data: Dict) -> Dict[
        str, Any]:
        """
        Calculate overall financial health score based on all statements

        Args:
            income_data: Results from income statement analysis
            balance_data: Results from balance sheet analysis
            cash_flow_data: Results from cash flow analysis

        Returns:
            Dictionary with financial health components and overall score
        """
        score_components = {}
        try:
            # 1. Profitability (0-100 score)
            profitability_score = 0
            profitability_factors = 0

            # Check net margin trend
            if income_data and 'key_metrics' in income_data and 'margins' in income_data['key_metrics']:
                net_margin = income_data['key_metrics']['margins'].get('net_margin', {})
                if net_margin and 'latest' in net_margin:
                    margin_value = net_margin['latest']
                    if margin_value > 0.20:  # Over 20% margin
                        profitability_score += 100
                    elif margin_value > 0.15:  # 15-20% margin
                        profitability_score += 90
                    elif margin_value > 0.10:  # 10-15% margin
                        profitability_score += 80
                    elif margin_value > 0.05:  # 5-10% margin
                        profitability_score += 60
                    elif margin_value > 0:  # 0-5% margin
                        profitability_score += 40
                    else:  # Negative margin
                        profitability_score += 0

                    profitability_factors += 1

                    # Add points for improving trend
                    if 'trend' in net_margin and net_margin['trend'] == 'improving':
                        profitability_score += 20
                        profitability_factors += 0.2
                    elif 'trend' in net_margin and net_margin['trend'] == 'deteriorating':
                        profitability_score -= 20
                        profitability_factors += 0.2

            # Check revenue growth
            if income_data and 'key_metrics' in income_data and 'growth_rates' in income_data['key_metrics']:
                revenue_growth = income_data['key_metrics']['growth_rates'].get('revenue_growth', {})
                if revenue_growth and 'latest' in revenue_growth:
                    growth_value = revenue_growth['latest']
                    if growth_value > 0.20:  # Over 20% growth
                        profitability_score += 100
                    elif growth_value > 0.10:  # 10-20% growth
                        profitability_score += 80
                    elif growth_value > 0.05:  # 5-10% growth
                        profitability_score += 60
                    elif growth_value > 0:  # 0-5% growth
                        profitability_score += 40
                    else:  # Negative growth
                        profitability_score += 0

                    profitability_factors += 1

            # Calculate profitability subscore
            if profitability_factors > 0:
                score_components['profitability'] = round(profitability_score / profitability_factors)
            else:
                score_components['profitability'] = None

            # 2. Liquidity (0-100 score)
            liquidity_score = 0
            liquidity_factors = 0

            # Check current ratio
            if balance_data and 'key_metrics' in balance_data and 'ratios' in balance_data['key_metrics']:
                current_ratio = balance_data['key_metrics']['ratios'].get('current_ratio', {})
                if current_ratio and 'latest' in current_ratio:
                    ratio_value = current_ratio['latest']
                    if ratio_value > 2.0:  # Excellent liquidity
                        liquidity_score += 100
                    elif ratio_value > 1.5:  # Good liquidity
                        liquidity_score += 80
                    elif ratio_value > 1.0:  # Adequate liquidity
                        liquidity_score += 60
                    elif ratio_value > 0.8:  # Concerning liquidity
                        liquidity_score += 30
                    else:  # Poor liquidity
                        liquidity_score += 0

                    liquidity_factors += 1

            # Check cash ratio or cash to total assets
            if balance_data and 'key_metrics' in balance_data and 'assets' in balance_data['key_metrics']:
                cash = balance_data['key_metrics']['assets'].get('cash_equivalents', {})
                total_assets = balance_data['key_metrics']['assets'].get('total_assets', {})

                if cash and total_assets and 'latest' in cash and 'latest' in total_assets:
                    if total_assets['latest'] > 0:
                        cash_ratio = cash['latest'] / total_assets['latest']

                        if cash_ratio > 0.25:  # Excellent cash position
                            liquidity_score += 100
                        elif cash_ratio > 0.15:  # Good cash position
                            liquidity_score += 80
                        elif cash_ratio > 0.10:  # Adequate cash position
                            liquidity_score += 60
                        elif cash_ratio > 0.05:  # Concerning cash position
                            liquidity_score += 30
                        else:  # Low cash position
                            liquidity_score += 0

                        liquidity_factors += 1

            # Calculate liquidity subscore
            if liquidity_factors > 0:
                score_components['liquidity'] = round(liquidity_score / liquidity_factors)
            else:
                score_components['liquidity'] = None

            # 3. Solvency (0-100 score)
            solvency_score = 0
            solvency_factors = 0

            # Check debt to equity ratio
            if balance_data and 'key_metrics' in balance_data and 'ratios' in balance_data['key_metrics']:
                debt_to_equity = balance_data['key_metrics']['ratios'].get('debt_to_equity', {})
                if debt_to_equity and 'latest' in debt_to_equity:
                    ratio_value = debt_to_equity['latest']
                    if ratio_value < 0.3:  # Very low leverage
                        solvency_score += 100
                    elif ratio_value < 0.5:  # Low leverage
                        solvency_score += 90
                    elif ratio_value < 1.0:  # Moderate leverage
                        solvency_score += 70
                    elif ratio_value < 1.5:  # High leverage
                        solvency_score += 40
                    elif ratio_value < 2.0:  # Very high leverage
                        solvency_score += 20
                    else:  # Excessive leverage
                        solvency_score += 0

                    solvency_factors += 1

            # Check debt to assets ratio
            if balance_data and 'key_metrics' in balance_data and 'ratios' in balance_data['key_metrics']:
                debt_to_assets = balance_data['key_metrics']['ratios'].get('debt_to_assets', {})
                if debt_to_assets and 'latest' in debt_to_assets:
                    ratio_value = debt_to_assets['latest']
                    if ratio_value < 0.2:  # Very low debt
                        solvency_score += 100
                    elif ratio_value < 0.3:  # Low debt
                        solvency_score += 90
                    elif ratio_value < 0.4:  # Moderate debt
                        solvency_score += 70
                    elif ratio_value < 0.5:  # High debt
                        solvency_score += 50
                    elif ratio_value < 0.7:  # Very high debt
                        solvency_score += 20
                    else:  # Excessive debt
                        solvency_score += 0

                    solvency_factors += 1

            # Calculate solvency subscore
            if solvency_factors > 0:
                score_components['solvency'] = round(solvency_score / solvency_factors)
            else:
                score_components['solvency'] = None

            # 4. Cash Flow Quality (0-100 score)
            cash_flow_score = 0
            cash_flow_factors = 0

            # Check free cash flow
            if cash_flow_data and 'key_metrics' in cash_flow_data:
                free_cash_flow = cash_flow_data['key_metrics'].get('free_cash_flow', {})
                if free_cash_flow and 'latest' in free_cash_flow:
                    if free_cash_flow['latest'] > 0:
                        cash_flow_score += 80

                        # Add points for improving trend
                        if 'trend' in free_cash_flow and free_cash_flow['trend'] == 'improving':
                            cash_flow_score += 20
                        elif 'trend' in free_cash_flow and free_cash_flow['trend'] == 'deteriorating':
                            cash_flow_score -= 20
                    else:
                        cash_flow_score += 20  # Negative FCF

                    cash_flow_factors += 1

            # Check operating cash flow to net income ratio (cash flow quality)
            if cash_flow_data and 'key_metrics' in cash_flow_data and 'cf_quality' in cash_flow_data['key_metrics']:
                ocf_to_ni = cash_flow_data['key_metrics']['cf_quality'].get('ocf_to_net_income', {})
                if ocf_to_ni and 'latest' in ocf_to_ni:
                    ratio_value = ocf_to_ni['latest']
                    if ratio_value > 1.2:  # Excellent cash conversion
                        cash_flow_score += 100
                    elif ratio_value > 1.0:  # Good cash conversion
                        cash_flow_score += 90
                    elif ratio_value > 0.8:  # Adequate cash conversion
                        cash_flow_score += 70
                    elif ratio_value > 0.6:  # Concerning cash conversion
                        cash_flow_score += 40
                    else:  # Poor cash conversion
                        cash_flow_score += 0

                    cash_flow_factors += 1

            # Calculate cash flow subscore
            if cash_flow_factors > 0:
                score_components['cash_flow'] = round(cash_flow_score / cash_flow_factors)
            else:
                score_components['cash_flow'] = None

            # Calculate overall financial health score (weighted average)
            valid_scores = [s for s in score_components.values() if s is not None]
            if valid_scores:
                overall_score = sum(valid_scores) / len(valid_scores)
            else:
                overall_score = None

            # Determine financial health rating
            rating = None
            if overall_score is not None:
                if overall_score >= 90:
                    rating = "Excellent"
                elif overall_score >= 75:
                    rating = "Strong"
                elif overall_score >= 60:
                    rating = "Good"
                elif overall_score >= 45:
                    rating = "Moderate"
                elif overall_score >= 30:
                    rating = "Weak"
                else:
                    rating = "Poor"

            return {
                'components': score_components,
                'overall_score': overall_score,
                'rating': rating
            }

        except Exception as e:
            logger.error(f"Error calculating financial health score: {e}")
            return {
                'components': score_components,
                'overall_score': None,
                'rating': None
            }

    # Helper methods

    def _analyze_single_period_income(self, df: pd.DataFrame) -> Dict:
        """Analyze a single period income statement"""
        try:
            # Get the current period (first column)
            current_period = df.columns[0]

            # Calculate common-size analysis (as percentage of revenue)
            common_size = pd.Series(index=df.index)
            if 'Total Revenue' in df.index and df.loc['Total Revenue', current_period] != 0:
                common_size = df[current_period] / df.loc['Total Revenue', current_period]

            # Extract key metrics
            key_metrics = {
                'revenue': {'latest': df.loc['Total Revenue', current_period] if 'Total Revenue' in df.index and current_period in df.columns else None},
                'gross_profit': {
                    'latest': df.loc['Gross Profit', current_period] if 'Gross Profit' in df.index else None},
                'operating_income': {
                    'latest': df.loc['Operating Income', current_period] if 'Operating Income' in df.index else None},
                'net_income': {'latest': df.loc['Net Income', current_period] if 'Net Income' in df.index else None},
                'margins': {
                    'gross_margin': {
                        'latest': df.loc['Gross Profit', current_period] / df.loc['Total Revenue', current_period]
                        if 'Gross Profit' in df.index and 'Total Revenue' in df.index and df.loc[
                            'Total Revenue', current_period] != 0 else None},
                    'operating_margin': {
                        'latest': df.loc['Operating Income', current_period] / df.loc['Total Revenue', current_period]
                        if 'Operating Income' in df.index and 'Total Revenue' in df.index and df.loc[
                            'Total Revenue', current_period] != 0 else None},
                    'net_margin': {
                        'latest': df.loc['Net Income', current_period] / df.loc['Total Revenue', current_period]
                        if 'Net Income' in df.index and 'Total Revenue' in df.index and df.loc[
                            'Total Revenue', current_period] != 0 else None}
                }
            }

            return {
                'key_metrics': key_metrics,
                'common_size': common_size,
                'periods': [current_period]
            }
        except Exception as e:
            logger.error(f"Error analyzing single period income statement: {e}")
            return {}

    def _analyze_single_period_balance(self, df: pd.DataFrame) -> Dict:
        """Analyze a single period balance sheet"""
        try:
            # Get the current period (first column)
            current_period = df.columns[0]

            # Calculate common-size analysis (as percentage of total assets)
            common_size = pd.Series(index=df.index)
            if 'Total Assets' in df.index and df.loc['Total Assets', current_period] != 0:
                common_size = df[current_period] / df.loc['Total Assets', current_period]

            # Extract key metrics
            key_metrics = {
                'assets': {
                    'total_assets': {
                        'latest': df.loc['Total Assets', current_period] if 'Total Assets' in df.index else None},
                    'current_assets': {'latest': df.loc[
                        'Total Current Assets', current_period] if 'Total Current Assets' in df.index else None},
                    'cash_equivalents': {'latest': df.loc[
                        'Cash and Cash Equivalents', current_period] if 'Cash and Cash Equivalents' in df.index else None}
                },
                'liabilities': {
                    'total_liabilities': {'latest': df.loc[
                        'Total Liabilities', current_period] if 'Total Liabilities' in df.index else None},
                    'current_liabilities': {'latest': df.loc[
                        'Total Current Liabilities', current_period] if 'Total Current Liabilities' in df.index else None},
                    'long_term_debt': {
                        'latest': df.loc['Long Term Debt', current_period] if 'Long Term Debt' in df.index else None}
                },
                'equity': {
                    'total_equity': {'latest': df.loc[
                        'Total Stockholder Equity', current_period] if 'Total Stockholder Equity' in df.index else None}
                },
                'ratios': {
                    'current_ratio': {'latest': df.loc['Total Current Assets', current_period] / df.loc[
                        'Total Current Liabilities', current_period]
                    if 'Total Current Assets' in df.index and 'Total Current Liabilities' in df.index and df.loc[
                        'Total Current Liabilities', current_period] != 0 else None},
                    'debt_to_equity': {'latest': df.loc['Total Liabilities', current_period] / df.loc[
                        'Total Stockholder Equity', current_period]
                    if 'Total Liabilities' in df.index and 'Total Stockholder Equity' in df.index and df.loc[
                        'Total Stockholder Equity', current_period] != 0 else None},
                    'debt_to_assets': {
                        'latest': df.loc['Total Liabilities', current_period] / df.loc['Total Assets', current_period]
                        if 'Total Liabilities' in df.index and 'Total Assets' in df.index and df.loc[
                            'Total Assets', current_period] != 0 else None}
                }
            }

            return {
                'key_metrics': key_metrics,
                'common_size': common_size,
                'periods': [current_period]
            }
        except Exception as e:
            logger.error(f"Error analyzing single period balance sheet: {e}")
            return {}

    def _analyze_single_period_cash_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze a single period cash flow statement"""
        try:
            # Get the current period (first column)
            current_period = df.columns[0]

            # Identify key metrics for cash flow analysis
            operating_cf_key = self._find_key_by_pattern(df, ['Operating Cash Flow', 'Net Cash from Operations'])
            investing_cf_key = self._find_key_by_pattern(df, ['Investing Cash Flow', 'Net Cash from Investing'])
            financing_cf_key = self._find_key_by_pattern(df, ['Financing Cash Flow', 'Net Cash from Financing'])
            capex_key = self._find_key_by_pattern(df, ['Capital Expenditure', 'CAPEX',
                                                       'Purchase of Property Plant and Equipment'])
            fcf_key = 'Free Cash Flow'  # May need to calculate this

            # Extract key metrics
            key_metrics = {
                'operating_cash_flow': {'latest': df.loc[
                    operating_cf_key, current_period] if operating_cf_key and operating_cf_key in df.index else None},
                'investing_cash_flow': {'latest': df.loc[
                    investing_cf_key, current_period] if investing_cf_key and investing_cf_key in df.index else None},
                'financing_cash_flow': {'latest': df.loc[
                    financing_cf_key, current_period] if financing_cf_key and financing_cf_key in df.index else None},
                'capital_expenditure': {
                    'latest': df.loc[capex_key, current_period] if capex_key and capex_key in df.index else None}
            }

            # Calculate Free Cash Flow if not directly available
            if fcf_key not in df.index and operating_cf_key and capex_key:
                free_cash_flow = df.loc[operating_cf_key, current_period] - df.loc[capex_key, current_period]
                key_metrics['free_cash_flow'] = {'latest': free_cash_flow}
            else:
                key_metrics['free_cash_flow'] = {
                    'latest': df.loc[fcf_key, current_period] if fcf_key in df.index else None}

            return {
                'key_metrics': key_metrics,
                'periods': [current_period]
            }
        except Exception as e:
            logger.error(f"Error analyzing single period cash flow statement: {e}")
            return {}

    def _extract_metric_trend(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Extract a metric's values and determine its trend"""
        if metric_name not in df.index:
            return None

        values = df.loc[metric_name]
        latest = values.iloc[0]

        return {
            'values': values.to_dict(),
            'latest': latest,
            'trend': self._calculate_trend(values)
        }

    def _calculate_margin_trend(self, df: pd.DataFrame, numerator: str, denominator: str) -> Dict[str, Any]:
        """Calculate and analyze margin trend"""
        if numerator not in df.index or denominator not in df.index:
            return None

        margins = pd.Series(index=df.columns)

        for col in df.columns:
            if df.loc[denominator, col] != 0:
                margins[col] = df.loc[numerator, col] / df.loc[denominator, col]
            else:
                margins[col] = None

        latest = margins.iloc[0]

        return {
            'values': margins.to_dict(),
            'latest': latest,
            'trend': self._calculate_trend(margins)
        }

    def _calculate_ratio_trend(self, df: pd.DataFrame, numerator: str, denominator: str) -> Dict[str, Any]:
        """Calculate and analyze ratio trend"""
        # Similar to margin trend calculation but with more generic naming
        return self._calculate_margin_trend(df, numerator, denominator)

    def _extract_growth_rate(self, growth_df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Extract growth rate for a specific metric"""
        if metric_name not in growth_df.index:
            return None

        growth_rates = growth_df.loc[metric_name]
        latest = growth_rates.iloc[0]

        return {
            'values': growth_rates.to_dict(),
            'latest': latest,
            'trend': self._calculate_trend(growth_rates)
        }

    def _calculate_trend(self, series: pd.Series) -> str:
        """Determine the trend direction from a time series"""
        if len(series) < 2:
            return "stable"

        try:
            # Use simple linear regression to determine trend
            import numpy as np
            from scipy import stats

            x = np.arange(len(series))
            y = series.values

            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return "stable"

            slope, _, _, p_value, _ = stats.linregress(x[mask], y[mask])

            # Determine significance and direction of trend
            if p_value > 0.05:  # Not statistically significant
                return "stable"
            elif slope > 0:
                return "improving"
            else:
                return "deteriorating"
        except Exception:
            return "stable"

    def _find_key_by_pattern(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Find a row in the dataframe that matches one of the given patterns"""
        for pattern in patterns:
            for key in df.index:
                if pattern.lower() in key.lower():
                    return key
        return None