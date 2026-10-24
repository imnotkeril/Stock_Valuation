import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.config import COLORS, BANKRUPTCY_MODELS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bankruptcy_models')


class BankruptcyAnalyzer:
    """
    Class for analyzing bankruptcy risk using various models including:
    - Altman Z-Score (Original, Modified, and Financial variants)
    - Springate Model
    - Zmijewski Model
    - Taffler Z-Score
    - Ohlson O-Score

    The class provides both individual model calculations and comprehensive
    risk assessments combining multiple models with sector-specific adaptations.
    """

    def __init__(self):
        """Initialize the bankruptcy analyzer"""
        # Map sectors to appropriate Z-score models
        self.sector_model_mapping = {
            "Financials": "financial",
            "Banks": "financial",
            "Insurance": "financial",
            "Real Estate": "modified",
            "Technology": "modified",
            "Healthcare": "modified",
            "Industrials": "original",
            "Materials": "original",
            "Energy": "original",
            "Consumer Discretionary": "modified",
            "Consumer Staples": "modified",
            "Communication Services": "modified",
            "Utilities": "modified"
        }

        # Define risk assessment thresholds
        self.risk_thresholds = {
            "z_score": {
                "original": {
                    "safe": 2.99,
                    "grey": 1.81
                },
                "modified": {
                    "safe": 2.60,
                    "grey": 1.10
                },
                "financial": {
                    "safe": 7.0,
                    "grey": 4.5
                }
            },
            "springate": {
                "threshold": 0.862
            },
            "zmijewski": {
                "threshold": 0.5
            },
            "taffler": {
                "threshold": 0.3,
                "grey": 0.2
            },
            "ohlson": {
                "threshold": 0.38
            },
            "fulmer": {
                "threshold": 0
            }
        }

        # Define colors for risk categories
        self.risk_colors = {
            "safe": COLORS["success"],
            "grey": COLORS["warning"],
            "distress": COLORS["danger"]
        }

        # Define weighted importance of different models for overall assessment
        self.model_weights = {
            "altman_z_score": 0.35,
            "springate": 0.20,
            "zmijewski": 0.20,
            "taffler": 0.15,
            "ohlson": 0.10
        }

        # Model descriptions for educational purposes
        self.model_descriptions = {
            "altman_z_score": "The Altman Z-Score is a financial formula that predicts the probability of a company going bankrupt. Developed by Edward I. Altman in 1968, it uses five financial ratios to determine a company's financial health.",
            "springate": "The Springate model is a bankruptcy prediction model developed by Gordon L.V. Springate in 1978. It uses four financial ratios to predict if a company is headed for bankruptcy.",
            "zmijewski": "The Zmijewski model is a bankruptcy prediction model developed by Mark E. Zmijewski in 1984. It uses a probit analysis that weights different financial ratios including return on assets, financial leverage, and liquidity.",
            "taffler": "The Taffler Z-Score is a bankruptcy prediction model developed by Richard Taffler in 1983 for UK companies. It focuses on liquidity and uses four financial ratios.",
            "ohlson": "The Ohlson O-Score is a bankruptcy prediction model developed by James Ohlson in 1980. It uses nine financial ratios and applies a logistic regression approach.",
            "fulmer": "The Fulmer H-Factor model is a bankruptcy prediction model that uses nine financial variables. It was designed to be effective with smaller companies."
        }

    def calculate_altman_z_score(self, financial_data: Dict, sector: str = None) -> Dict[str, Any]:
        """
        Calculate the Altman Z-Score for bankruptcy prediction

        Args:
            financial_data: Dict with income_statement, balance_sheet data
            sector: Company's sector to determine appropriate model variant

        Returns:
            Dictionary with Z-Score, components, and risk assessment
        """
        try:
            # Get appropriate model variant based on sector
            model_variant = "original"
            if sector:
                model_variant = self.sector_model_mapping.get(sector, "original")

            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Z-Score calculation")
                return self._get_empty_z_score_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate Z-Score components based on model variant
                if model_variant == "original" or model_variant == "modified":
                    return self._calculate_standard_z_score(income, balance, market_data, model_variant)
                elif model_variant == "financial":
                    return self._calculate_financial_z_score(income, balance)

            logger.warning("Invalid or empty financial data for Z-Score calculation")
            return self._get_empty_z_score_result()

        except Exception as e:
            logger.error(f"Error calculating Altman Z-Score: {e}")
            return self._get_empty_z_score_result()

    def calculate_springate_score(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Calculate the Springate model score for bankruptcy prediction

        Args:
            financial_data: Dict with income_statement, balance_sheet data

        Returns:
            Dictionary with Springate score and risk assessment
        """
        try:
            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Springate score calculation")
                return self._get_empty_springate_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate components
                try:
                    # A = Working Capital / Total Assets
                    working_capital = self._get_value_with_alternatives(balance,
                                                                        ['Total Current Assets',
                                                                         'Current Assets']) - self._get_value_with_alternatives(
                        balance,
                        ['Total Current Liabilities', 'Current Liabilities'])
                    total_assets = self._get_value_with_alternatives(balance, ['Total Assets', 'Assets'])
                    component_a = working_capital / total_assets if total_assets != 0 else 0

                    # B = EBIT / Total Assets
                    ebit = self._get_value_with_alternatives(income,
                                                             ['Operating Income', 'EBIT', 'Income From Operations'])
                    component_b = ebit / total_assets if total_assets != 0 else 0

                    # C = EBT / Current Liabilities
                    ebt = self._get_value_with_alternatives(income,
                                                            ['Income Before Tax', 'Earnings Before Tax',
                                                             'Profit Before Tax'])
                    current_liabilities = self._get_value_with_alternatives(balance,
                                                                            ['Total Current Liabilities',
                                                                             'Current Liabilities'])
                    component_c = ebt / current_liabilities if current_liabilities != 0 else 0

                    # D = Sales / Total Assets
                    sales = self._get_value_with_alternatives(income,
                                                              ['Total Revenue', 'Revenue', 'Sales', 'Net Sales'])
                    component_d = sales / total_assets if total_assets != 0 else 0

                    # Calculate Springate Score: 1.03*A + 3.07*B + 0.66*C + 0.4*D
                    springate_score = 1.03 * component_a + 3.07 * component_b + 0.66 * component_c + 0.4 * component_d

                    # Assess bankruptcy risk
                    threshold = self.risk_thresholds["springate"]["threshold"]
                    is_safe = springate_score > threshold

                    return {
                        'score': springate_score,
                        'components': {
                            'working_capital_to_total_assets': component_a,
                            'ebit_to_total_assets': component_b,
                            'ebt_to_current_liabilities': component_c,
                            'sales_to_total_assets': component_d
                        },
                        'assessment': 'safe' if is_safe else 'distress',
                        'description': 'Low bankruptcy risk' if is_safe else 'High bankruptcy risk',
                        'color': self.risk_colors['safe'] if is_safe else self.risk_colors['distress'],
                        'threshold': threshold,
                        'model': 'Springate',
                        'explanation': self._get_springate_explanation(springate_score, component_a, component_b,
                                                                       component_c, component_d)
                    }
                except Exception as e:
                    logger.error(f"Error in Springate score calculation: {e}")
                    return self._get_empty_springate_result()

            logger.warning("Invalid or empty financial data for Springate score calculation")
            return self._get_empty_springate_result()

        except Exception as e:
            logger.error(f"Error calculating Springate score: {e}")
            return self._get_empty_springate_result()

    def calculate_zmijewski_score(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Calculate the Zmijewski model score for bankruptcy prediction

        Args:
            financial_data: Dict with income_statement, balance_sheet data

        Returns:
            Dictionary with Zmijewski score and risk assessment
        """
        try:
            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Zmijewski score calculation")
                return self._get_empty_zmijewski_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate components
                try:
                    # ROA = Net Income / Total Assets
                    net_income = self._get_value_with_alternatives(income,
                                                                   ['Net Income', 'Net Profit', 'Net Earnings'])
                    total_assets = self._get_value_with_alternatives(balance,
                                                                     ['Total Assets', 'Assets'])
                    roa = net_income / total_assets if total_assets != 0 else 0

                    # Financial Leverage = Total Liabilities / Total Assets
                    total_liabilities = self._get_value_with_alternatives(balance,
                                                                          ['Total Liabilities',
                                                                           'Total Liabilities Net Minority Interest'])
                    leverage = total_liabilities / total_assets if total_assets != 0 else 0

                    # Liquidity = Current Assets / Current Liabilities
                    current_assets = self._get_value_with_alternatives(balance,
                                                                       ['Total Current Assets', 'Current Assets'])
                    current_liabilities = self._get_value_with_alternatives(balance,
                                                                            ['Total Current Liabilities',
                                                                             'Current Liabilities'])
                    liquidity = current_assets / current_liabilities if current_liabilities != 0 else 0

                    # Calculate Zmijewski's X-score: -4.3 - 4.5*ROA + 5.7*Leverage - 0.004*Liquidity
                    x_score = -4.3 - 4.5 * roa + 5.7 * leverage - 0.004 * liquidity

                    # Convert to probability: 1 / (1 + e^(-x_score))
                    probability = 1 / (1 + np.exp(-x_score))

                    # Assess bankruptcy risk
                    threshold = self.risk_thresholds["zmijewski"]["threshold"]
                    is_safe = probability < threshold

                    return {
                        'score': x_score,
                        'probability': probability,
                        'components': {
                            'roa': roa,
                            'leverage': leverage,
                            'liquidity': liquidity
                        },
                        'assessment': 'safe' if is_safe else 'distress',
                        'description': 'Low bankruptcy risk' if is_safe else 'High bankruptcy risk',
                        'color': self.risk_colors['safe'] if is_safe else self.risk_colors['distress'],
                        'threshold': threshold,
                        'model': 'Zmijewski',
                        'explanation': self._get_zmijewski_explanation(probability, roa, leverage, liquidity)
                    }
                except Exception as e:
                    logger.error(f"Error in Zmijewski score calculation: {e}")
                    return self._get_empty_zmijewski_result()

            logger.warning("Invalid or empty financial data for Zmijewski score calculation")
            return self._get_empty_zmijewski_result()

        except Exception as e:
            logger.error(f"Error calculating Zmijewski score: {e}")
            return self._get_empty_zmijewski_result()

    def calculate_taffler_score(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Calculate the Taffler Z-Score for bankruptcy prediction (UK model)

        Args:
            financial_data: Dict with income_statement, balance_sheet data

        Returns:
            Dictionary with Taffler score and risk assessment
        """
        try:
            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Taffler score calculation")
                return self._get_empty_taffler_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate components
                try:
                    # R1 = Profit Before Tax / Current Liabilities
                    profit_before_tax = self._get_value_with_alternatives(income,
                                                                          ['Income Before Tax', 'Earnings Before Tax',
                                                                           'Profit Before Tax'])
                    current_liabilities = self._get_value_with_alternatives(balance,
                                                                            ['Total Current Liabilities',
                                                                             'Current Liabilities'])
                    r1 = profit_before_tax / current_liabilities if current_liabilities != 0 else 0

                    # R2 = Current Assets / Total Liabilities
                    current_assets = self._get_value_with_alternatives(balance,
                                                                       ['Total Current Assets', 'Current Assets'])
                    total_liabilities = self._get_value_with_alternatives(balance,
                                                                          ['Total Liabilities',
                                                                           'Total Liabilities Net Minority Interest'])
                    r2 = current_assets / total_liabilities if total_liabilities != 0 else 0

                    # R3 = Current Liabilities / Total Assets
                    total_assets = self._get_value_with_alternatives(balance,
                                                                     ['Total Assets', 'Assets'])
                    r3 = current_liabilities / total_assets if total_assets != 0 else 0

                    # R4 = (Quick Assets - Current Liabilities) / (Operating Costs - Depreciation)
                    quick_assets = current_assets - self._get_value_with_alternatives(balance,
                                                                                      ['Inventory', 'Inventories'])
                    operating_costs = self._get_value_with_alternatives(income,
                                                                        ['Total Operating Expenses',
                                                                         'Operating Expenses'])
                    if operating_costs is None or operating_costs == 0:
                        # Try alternate calculation
                        revenue = self._get_value_with_alternatives(income,
                                                                    ['Total Revenue', 'Revenue', 'Sales'])
                        operating_income = self._get_value_with_alternatives(income,
                                                                             ['Operating Income', 'EBIT',
                                                                              'Income From Operations'])
                        if revenue is not None and operating_income is not None:
                            operating_costs = revenue - operating_income
                        else:
                            operating_costs = 1  # Fallback to avoid division by zero

                    depreciation = self._get_value_with_alternatives(income,
                                                                     ['Depreciation and Amortization', 'Depreciation',
                                                                      'Amortization'])
                    if depreciation is None:
                        depreciation = 0

                    r4_denominator = operating_costs - depreciation
                    r4 = 0 if r4_denominator == 0 else (quick_assets - current_liabilities) / r4_denominator

                    # Calculate Taffler Z-Score: 0.53*R1 + 0.13*R2 + 0.18*R3 + 0.16*R4
                    taffler_score = 0.53 * r1 + 0.13 * r2 + 0.18 * r3 + 0.16 * r4

                    # Assess bankruptcy risk
                    safe_threshold = self.risk_thresholds["taffler"]["threshold"]
                    grey_threshold = self.risk_thresholds["taffler"]["grey"]

                    if taffler_score > safe_threshold:
                        assessment = "safe"
                        description = "Low bankruptcy risk (Safe Zone)"
                        color = self.risk_colors["safe"]
                    elif taffler_score > grey_threshold:
                        assessment = "grey"
                        description = "Moderate bankruptcy risk (Grey Zone)"
                        color = self.risk_colors["grey"]
                    else:
                        assessment = "distress"
                        description = "High bankruptcy risk (Distress Zone)"
                        color = self.risk_colors["distress"]

                    return {
                        'score': taffler_score,
                        'components': {
                            'profit_to_current_liabilities': r1,
                            'current_assets_to_total_liabilities': r2,
                            'current_liabilities_to_total_assets': r3,
                            'no_credit_interval': r4
                        },
                        'assessment': assessment,
                        'description': description,
                        'color': color,
                        'thresholds': {
                            'safe': safe_threshold,
                            'grey': grey_threshold
                        },
                        'model': 'Taffler',
                        'explanation': self._get_taffler_explanation(taffler_score, r1, r2, r3, r4)
                    }
                except Exception as e:
                    logger.error(f"Error in Taffler score calculation: {e}")
                    return self._get_empty_taffler_result()

            logger.warning("Invalid or empty financial data for Taffler score calculation")
            return self._get_empty_taffler_result()

        except Exception as e:
            logger.error(f"Error calculating Taffler score: {e}")
            return self._get_empty_taffler_result()

    def calculate_ohlson_score(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Calculate the Ohlson O-Score for bankruptcy prediction

        Args:
            financial_data: Dict with income_statement, balance_sheet data

        Returns:
            Dictionary with Ohlson score and risk assessment
        """
        try:
            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            market_data = financial_data.get('market_data', {})

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Ohlson score calculation")
                return self._get_empty_ohlson_result()

            # Check if we have current and previous year data
            if (isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and
                    not income_stmt.empty and not balance_sheet.empty):

                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Get previous year data if available
                prev_income = None
                prev_balance = None
                if income_stmt.shape[1] > 1:
                    prev_income = income_stmt.iloc[:, 1]
                if balance_sheet.shape[1] > 1:
                    prev_balance = balance_sheet.iloc[:, 1]

                try:
                    # Calculate Ohlson O-Score components

                    # SIZE = log(Total Assets / GNP Price Index)
                    # Since GNP Price Index is external data, we'll just use log of total assets
                    total_assets = self._get_value_with_alternatives(balance,
                                                                     ['Total Assets', 'Assets'])
                    size = np.log(total_assets / 100) if total_assets > 0 else 0  # Dividing by arbitrary value as proxy

                    # TLTA = Total Liabilities / Total Assets
                    total_liabilities = self._get_value_with_alternatives(balance,
                                                                          ['Total Liabilities',
                                                                           'Total Liabilities Net Minority Interest'])
                    tlta = total_liabilities / total_assets if total_assets != 0 else 0

                    # WCTA = Working Capital / Total Assets
                    current_assets = self._get_value_with_alternatives(balance,
                                                                       ['Total Current Assets', 'Current Assets'])
                    current_liabilities = self._get_value_with_alternatives(balance,
                                                                            ['Total Current Liabilities',
                                                                             'Current Liabilities'])
                    working_capital = current_assets - current_liabilities
                    wcta = working_capital / total_assets if total_assets != 0 else 0

                    # CLCA = Current Liabilities / Current Assets
                    clca = current_liabilities / current_assets if current_assets != 0 else 0

                    # OENEG = 1 if Total Liabilities > Total Assets, else 0
                    oeneg = 1 if total_liabilities > total_assets else 0

                    # NITA = Net Income / Total Assets
                    net_income = self._get_value_with_alternatives(income,
                                                                   ['Net Income', 'Net Profit', 'Net Earnings'])
                    nita = net_income / total_assets if total_assets != 0 else 0

                    # FUTL = Funds from Operations / Total Liabilities
                    # For simplicity, we use operating cash flow
                    cash_flow = financial_data.get('cash_flow')
                    funds_from_operations = 0
                    if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                        cf = cash_flow.iloc[:, 0]
                        funds_from_operations = self._get_value_with_alternatives(cf,
                                                                                  ['Operating Cash Flow',
                                                                                   'Cash Flow from Operations',
                                                                                   'Net Cash Provided by Operating Activities'])
                    futl = funds_from_operations / total_liabilities if total_liabilities != 0 else 0

                    # INTWO = 1 if Net Income < 0 for last two years, else 0
                    intwo = 0
                    if net_income < 0:
                        if prev_income is not None:
                            prev_net_income = self._get_value_with_alternatives(prev_income,
                                                                                ['Net Income', 'Net Profit',
                                                                                 'Net Earnings'])
                            if prev_net_income < 0:
                                intwo = 1
                        else:
                            # If we don't have previous year data, just use current
                            intwo = 1

                    # CHIN = (Net Income_t - Net Income_{t-1}) / (|Net Income_t| + |Net Income_{t-1}|)
                    chin = 0
                    if prev_income is not None:
                        prev_net_income = self._get_value_with_alternatives(prev_income,
                                                                            ['Net Income', 'Net Profit',
                                                                             'Net Earnings'])
                        denominator = abs(net_income) + abs(prev_net_income)
                        if denominator > 0:
                            chin = (net_income - prev_net_income) / denominator

                    # Calculate O-Score using Ohlson's model
                    o_score = -1.32 - 0.407 * size + 6.03 * tlta - 1.43 * wcta + 0.0757 * clca + \
                              -2.37 * nita - 1.83 * futl + 0.285 * intwo - 0.521 * chin

                    # Convert to probability (logistic model): 1 / (1 + e^(-O-Score))
                    probability = 1 / (1 + np.exp(-o_score))

                    # Assess bankruptcy risk
                    threshold = self.risk_thresholds["ohlson"]["threshold"]
                    is_safe = probability < threshold

                    return {
                        'score': o_score,
                        'probability': probability,
                        'components': {
                            'size': size,
                            'total_liabilities_to_total_assets': tlta,
                            'working_capital_to_total_assets': wcta,
                            'current_liabilities_to_current_assets': clca,
                            'negative_equity_dummy': oeneg,
                            'net_income_to_total_assets': nita,
                            'funds_from_operations_to_total_liabilities': futl,
                            'negative_earnings_last_two_years': intwo,
                            'change_in_net_income': chin
                        },
                        'assessment': 'safe' if is_safe else 'distress',
                        'description': 'Low bankruptcy risk' if is_safe else 'High bankruptcy risk',
                        'color': self.risk_colors['safe'] if is_safe else self.risk_colors['distress'],
                        'threshold': threshold,
                        'model': 'Ohlson',
                        'explanation': self._get_ohlson_explanation(probability, size, tlta, wcta, nita)
                    }

                except Exception as e:
                    logger.error(f"Error in Ohlson score calculation: {e}")
                    return self._get_empty_ohlson_result()

            logger.warning("Invalid or empty financial data for Ohlson score calculation")
            return self._get_empty_ohlson_result()

        except Exception as e:
            logger.error(f"Error calculating Ohlson score: {e}")
            return self._get_empty_ohlson_result()

    def calculate_fulmer_score(self, financial_data: Dict) -> Dict[str, Any]:
        """
        Calculate the Fulmer H-Factor model score for bankruptcy prediction

        Args:
            financial_data: Dict with income_statement, balance_sheet data

        Returns:
            Dictionary with Fulmer score and risk assessment
        """
        try:
            # Extract required financial data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')
            cash_flow = financial_data.get('cash_flow')

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Fulmer score calculation")
                return self._get_empty_fulmer_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Get cash flow data if available
                cf = None
                if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                    cf = cash_flow.iloc[:, 0]

                try:
                    # Calculate Fulmer H-Factor components

                    # V1 = Retained Earnings / Total Assets
                    retained_earnings = self._get_value_with_alternatives(balance,
                                                                          ['Retained Earnings',
                                                                           'Accumulated Retained Earnings'])
                    total_assets = self._get_value_with_alternatives(balance,
                                                                     ['Total Assets', 'Assets'])
                    v1 = retained_earnings / total_assets if total_assets != 0 else 0

                    # V2 = Sales / Total Assets
                    sales = self._get_value_with_alternatives(income,
                                                              ['Total Revenue', 'Revenue', 'Sales', 'Net Sales'])
                    v2 = sales / total_assets if total_assets != 0 else 0

                    # V3 = EBT / Equity
                    ebt = self._get_value_with_alternatives(income,
                                                            ['Income Before Tax', 'Earnings Before Tax',
                                                             'Profit Before Tax'])
                    equity = self._get_value_with_alternatives(balance,
                                                               ['Total Stockholder Equity', 'Total Equity',
                                                                'Shareholders Equity'])
                    v3 = ebt / equity if equity != 0 else 0

                    # V4 = Cash Flow / Total Debt
                    cash_flow_value = 0
                    if cf is not None:
                        cash_flow_value = self._get_value_with_alternatives(cf,
                                                                            ['Operating Cash Flow',
                                                                             'Cash Flow from Operations',
                                                                             'Net Cash Provided by Operating Activities'])
                    total_debt = self._get_value_with_alternatives(balance,
                                                                   ['Total Debt', 'Short-term Debt and Long-term Debt',
                                                                    'Debt'])
                    if total_debt is None or total_debt == 0:
                        total_debt = self._get_value_with_alternatives(balance,
                                                                       ['Total Liabilities',
                                                                        'Total Liabilities Net Minority Interest'])
                    v4 = cash_flow_value / total_debt if total_debt != 0 else 0

                    # V5 = Total Debt / Total Assets
                    v5 = total_debt / total_assets if total_assets != 0 else 0

                    # V6 = Current Liabilities / Total Assets
                    current_liabilities = self._get_value_with_alternatives(balance,
                                                                            ['Total Current Liabilities',
                                                                             'Current Liabilities'])
                    v6 = current_liabilities / total_assets if total_assets != 0 else 0

                    # V7 = Log(Tangible Assets)
                    tangible_assets = total_assets - self._get_value_with_alternatives(balance,
                                                                                       [
                                                                                           'Goodwill and Intangible Assets',
                                                                                           'Intangible Assets',
                                                                                           'Goodwill'])
                    v7 = np.log(tangible_assets) if tangible_assets > 0 else 0

                    # V8 = Working Capital / Total Debt
                    current_assets = self._get_value_with_alternatives(balance,
                                                                       ['Total Current Assets', 'Current Assets'])
                    working_capital = current_assets - current_liabilities
                    v8 = working_capital / total_debt if total_debt != 0 else 0

                    # V9 = Log(EBIT / Interest)
                    ebit = self._get_value_with_alternatives(income,
                                                             ['Operating Income', 'EBIT', 'Income From Operations'])
                    interest_expense = abs(self._get_value_with_alternatives(income,
                                                                             ['Interest Expense',
                                                                              'Interest and Debt Expense',
                                                                              'Interest Paid']))
                    interest_ratio = ebit / interest_expense if interest_expense != 0 else 1
                    v9 = np.log(interest_ratio) if interest_ratio > 0 else 0

                    # Calculate Fulmer H-Factor
                    h_score = 5.528 * v1 + 0.212 * v2 + 0.073 * v3 + 1.270 * v4 - 0.120 * v5 + \
                              2.335 * v6 + 0.575 * v7 + 1.083 * v8 + 0.894 * v9 - 6.075

                    # Assess bankruptcy risk
                    threshold = self.risk_thresholds["fulmer"]["threshold"]
                    is_safe = h_score > threshold

                    return {
                        'score': h_score,
                        'components': {
                            'retained_earnings_to_total_assets': v1,
                            'sales_to_total_assets': v2,
                            'ebt_to_equity': v3,
                            'cash_flow_to_total_debt': v4,
                            'debt_to_total_assets': v5,
                            'current_liabilities_to_total_assets': v6,
                            'log_tangible_assets': v7,
                            'working_capital_to_total_debt': v8,
                            'log_ebit_to_interest': v9
                        },
                        'assessment': 'safe' if is_safe else 'distress',
                        'description': 'Low bankruptcy risk' if is_safe else 'High bankruptcy risk',
                        'color': self.risk_colors['safe'] if is_safe else self.risk_colors['distress'],
                        'threshold': threshold,
                        'model': 'Fulmer H-Factor',
                        'explanation': self._get_fulmer_explanation(h_score, v1, v2, v4, v8)
                    }

                except Exception as e:
                    logger.error(f"Error in Fulmer score calculation: {e}")
                    return self._get_empty_fulmer_result()

                logger.warning("Invalid or empty financial data for Fulmer score calculation")
                return self._get_empty_fulmer_result()

        except Exception as e:
            logger.error(f"Error calculating Fulmer score: {e}")
            return self._get_empty_fulmer_result()

    def get_comprehensive_risk_assessment(self, financial_data: Dict, sector: str = None) -> Dict[str, Any]:
        """
        Calculate comprehensive bankruptcy risk assessment using multiple models

        Args:
            financial_data: Dict with income_statement, balance_sheet data
            sector: Company's sector

        Returns:
            Dictionary with all model results and overall assessment
        """
        try:
            # Calculate all supported bankruptcy models
            z_score_results = self.calculate_altman_z_score(financial_data, sector)
            springate_results = self.calculate_springate_score(financial_data)
            zmijewski_results = self.calculate_zmijewski_score(financial_data)
            taffler_results = self.calculate_taffler_score(financial_data)
            ohlson_results = self.calculate_ohlson_score(financial_data)
            fulmer_results = self.calculate_fulmer_score(financial_data)

            # Get available models that returned valid scores
            available_models = []
            model_assessments = {}

            # Check Z-Score results
            if z_score_results.get('score') is not None:
                available_models.append('altman_z_score')
                model_assessments['altman_z_score'] = z_score_results.get('assessment')

            # Check Springate results
            if springate_results.get('score') is not None:
                available_models.append('springate')
                model_assessments['springate'] = springate_results.get('assessment')

            # Check Zmijewski results
            if zmijewski_results.get('probability') is not None:
                available_models.append('zmijewski')
                model_assessments['zmijewski'] = zmijewski_results.get('assessment')

            # Check Taffler results
            if taffler_results.get('score') is not None:
                available_models.append('taffler')
                model_assessments['taffler'] = taffler_results.get('assessment')

            # Check Ohlson results
            if ohlson_results.get('probability') is not None:
                available_models.append('ohlson')
                model_assessments['ohlson'] = ohlson_results.get('assessment')

            # Check Fulmer results
            if fulmer_results.get('score') is not None:
                available_models.append('fulmer')
                model_assessments['fulmer'] = fulmer_results.get('assessment')

            # Determine overall risk level based on weighted model assessments
            overall_assessment = None
            overall_description = None
            overall_color = None

            if available_models:
                # Initialize score dictionary
                assessment_scores = {
                    'safe': 2.0,
                    'grey': 1.0,
                    'distress': 0.0
                }

                # Calculate weighted score
                weighted_sum = 0
                total_weight = 0

                for model in available_models:
                    assessment = model_assessments[model]
                    weight = self.model_weights.get(model, 0.1)  # Default weight if not specified

                    weighted_sum += assessment_scores.get(assessment, 1.0) * weight
                    total_weight += weight

                average_score = weighted_sum / total_weight if total_weight > 0 else 1.0

                # Determine overall assessment based on average score
                if average_score >= 1.5:
                    overall_assessment = 'safe'
                    overall_description = 'Company shows strong financial health with low bankruptcy risk'
                    overall_color = self.risk_colors['safe']
                elif average_score >= 0.75:
                    overall_assessment = 'grey'
                    overall_description = 'Company shows moderate financial health with some bankruptcy risk indicators'
                    overall_color = self.risk_colors['grey']
                else:
                    overall_assessment = 'distress'
                    overall_description = 'Company shows significant financial distress with high bankruptcy risk indicators'
                    overall_color = self.risk_colors['distress']

                # Add risk factor analysis
                risk_factors = self._analyze_risk_factors(financial_data, {
                    'altman_z_score': z_score_results,
                    'springate': springate_results,
                    'zmijewski': zmijewski_results,
                    'taffler': taffler_results,
                    'ohlson': ohlson_results,
                    'fulmer': fulmer_results
                })

                # Add recommended actions based on risk level
                recommended_actions = self._get_recommended_actions(overall_assessment, risk_factors)

                # Format the results with the most comprehensive view
                return {
                    'models': {
                        'altman_z_score': z_score_results,
                        'springate': springate_results,
                        'zmijewski': zmijewski_results,
                        'taffler': taffler_results,
                        'ohlson': ohlson_results,
                        'fulmer': fulmer_results
                    },
                    'overall_assessment': overall_assessment,
                    'overall_description': overall_description,
                    'overall_color': overall_color,
                    'risk_factors': risk_factors,
                    'recommended_actions': recommended_actions,
                    'weighted_score': average_score,
                    'model_consensus': self._calculate_model_consensus(model_assessments)
                }

            return {
                'models': {
                    'altman_z_score': z_score_results,
                    'springate': springate_results,
                    'zmijewski': zmijewski_results,
                    'taffler': taffler_results,
                    'ohlson': ohlson_results,
                    'fulmer': fulmer_results
                },
                'overall_assessment': None,
                'overall_description': "Unable to determine risk level due to insufficient data",
                'overall_color': None
            }

        except Exception as e:
            logger.error(f"Error calculating comprehensive risk assessment: {e}")
            return {
                'models': {
                    'altman_z_score': self._get_empty_z_score_result(),
                    'springate': self._get_empty_springate_result(),
                    'zmijewski': self._get_empty_zmijewski_result(),
                    'taffler': self._get_empty_taffler_result(),
                    'ohlson': self._get_empty_ohlson_result(),
                    'fulmer': self._get_empty_fulmer_result()
                },
                'overall_assessment': None,
                'overall_description': 'Error in risk assessment calculation',
                'overall_color': None
            }

    def analyze_bankruptcy_risk_trends(self, financial_data: Dict, years: int = 3, sector: str = None) -> Dict[
        str, Any]:
        """
        Analyze trends in bankruptcy risk over time

        Args:
            financial_data: Dict with historical financial statements
            years: Number of years to analyze
            sector: Company sector

        Returns:
            Dictionary with bankruptcy risk trends and analysis
        """
        try:
            # Extract historical data
            income_stmt = financial_data.get('income_statement')
            balance_sheet = financial_data.get('balance_sheet')

            if not isinstance(income_stmt, pd.DataFrame) or not isinstance(balance_sheet, pd.DataFrame):
                return {"error": "Insufficient data for trend analysis"}

            # Determine how many years of data we have available
            available_years = min(years, min(income_stmt.shape[1], balance_sheet.shape[1]))

            if available_years < 2:
                return {"error": "Need at least 2 years of data for trend analysis"}

            # Store scores for each year
            yearly_scores = []

            # Calculate scores for each available year
            for year in range(available_years):
                # Create a financial data subset for this year
                year_data = {
                    'income_statement': income_stmt.iloc[:, year:year + 1],
                    'balance_sheet': balance_sheet.iloc[:, year:year + 1]
                }

                # If we have cash flow statement
                if 'cash_flow' in financial_data and isinstance(financial_data['cash_flow'], pd.DataFrame):
                    cash_flow = financial_data['cash_flow']
                    if cash_flow.shape[1] > year:
                        year_data['cash_flow'] = cash_flow.iloc[:, year:year + 1]

                # Calculate z-score for this year
                z_score = self.calculate_altman_z_score(year_data, sector)

                # Get year label (from column name if available, otherwise just index)
                year_label = income_stmt.columns[year]
                if isinstance(year_label, pd.Timestamp):
                    year_label = year_label.strftime('%Y')
                else:
                    year_label = str(year_label)

                yearly_scores.append({
                    'year': year_label,
                    'z_score': z_score.get('score'),
                    'assessment': z_score.get('assessment')
                })

            # Reverse to get chronological order
            yearly_scores.reverse()

            # Analyze trend
            trend_analysis = self._analyze_score_trend(yearly_scores)

            return {
                'yearly_scores': yearly_scores,
                'trend': trend_analysis['trend'],
                'trend_description': trend_analysis['description'],
                'risk_trajectory': trend_analysis['trajectory']
            }

        except Exception as e:
            logger.error(f"Error analyzing bankruptcy risk trends: {e}")
            return {"error": f"Error in trend analysis: {e}"}

    def compare_with_industry_benchmarks(self, financial_data: Dict, sector: str) -> Dict[str, Any]:
        """
        Compare company's bankruptcy risk with industry benchmarks

        Args:
            financial_data: Dict with financial statement data
            sector: Company's sector

        Returns:
            Dictionary with industry comparison results
        """
        try:
            # Calculate company's bankruptcy risk
            risk_assessment = self.get_comprehensive_risk_assessment(financial_data, sector)

            # Industry benchmarks (ideally should come from database or API)
            # This is a simplified version with hardcoded values
            industry_benchmarks = {
                "Technology": {"z_score": 3.2, "zmijewski_probability": 0.15},
                "Healthcare": {"z_score": 3.0, "zmijewski_probability": 0.18},
                "Financials": {"z_score": 6.5, "zmijewski_probability": 0.22},
                "Consumer Discretionary": {"z_score": 2.7, "zmijewski_probability": 0.25},
                "Consumer Staples": {"z_score": 2.9, "zmijewski_probability": 0.20},
                "Energy": {"z_score": 2.3, "zmijewski_probability": 0.30},
                "Industrials": {"z_score": 2.8, "zmijewski_probability": 0.22},
                "Materials": {"z_score": 2.5, "zmijewski_probability": 0.28},
                "Real Estate": {"z_score": 2.1, "zmijewski_probability": 0.25},
                "Communication Services": {"z_score": 2.8, "zmijewski_probability": 0.20},
                "Utilities": {"z_score": 2.4, "zmijewski_probability": 0.22}
            }

            # Default benchmarks if sector not found
            default_benchmark = {"z_score": 2.7, "zmijewski_probability": 0.22}

            # Get benchmark for the specified sector
            benchmark = industry_benchmarks.get(sector, default_benchmark)

            # Get company scores
            company_z_score = risk_assessment.get('models', {}).get('altman_z_score', {}).get('score')
            company_zmijewski = risk_assessment.get('models', {}).get('zmijewski', {}).get('probability')

            # Compare with benchmarks
            z_score_vs_industry = None
            zmijewski_vs_industry = None

            if company_z_score is not None and benchmark.get('z_score') is not None:
                z_score_vs_industry = company_z_score / benchmark['z_score'] if benchmark['z_score'] != 0 else 1

            if company_zmijewski is not None and benchmark.get('zmijewski_probability') is not None:
                # For Zmijewski, lower probability is better
                zmijewski_vs_industry = benchmark[
                                            'zmijewski_probability'] / company_zmijewski if company_zmijewski != 0 else 1

            # Create comparison result
            comparison = {
                'company_metrics': {
                    'z_score': company_z_score,
                    'zmijewski_probability': company_zmijewski
                },
                'industry_benchmarks': benchmark,
                'relative_performance': {
                    'z_score': z_score_vs_industry,
                    'zmijewski': zmijewski_vs_industry
                }
            }

            # Add interpretation
            if z_score_vs_industry is not None:
                if z_score_vs_industry > 1.2:
                    comparison[
                        'z_score_interpretation'] = "Company's financial health is significantly stronger than industry average"
                elif z_score_vs_industry > 1.0:
                    comparison[
                        'z_score_interpretation'] = "Company's financial health is better than industry average"
                elif z_score_vs_industry > 0.8:
                    comparison[
                        'z_score_interpretation'] = "Company's financial health is slightly below industry average"
                else:
                    comparison[
                        'z_score_interpretation'] = "Company's financial health is significantly weaker than industry average"

            if zmijewski_vs_industry is not None:
                if zmijewski_vs_industry > 1.2:
                    comparison[
                        'zmijewski_interpretation'] = "Company's bankruptcy risk is significantly lower than industry average"
                elif zmijewski_vs_industry > 1.0:
                    comparison[
                        'zmijewski_interpretation'] = "Company's bankruptcy risk is lower than industry average"
                elif zmijewski_vs_industry > 0.8:
                    comparison[
                        'zmijewski_interpretation'] = "Company's bankruptcy risk is slightly higher than industry average"
                else:
                    comparison[
                        'zmijewski_interpretation'] = "Company's bankruptcy risk is significantly higher than industry average"

            return comparison

        except Exception as e:
            logger.error(f"Error comparing with industry benchmarks: {e}")
            return {"error": f"Error in industry comparison: {e}"}

        # Private helper methods

    def _calculate_standard_z_score(self, income: pd.Series, balance: pd.Series, market_data: Dict,
                                    model_variant: str) -> Dict[str, Any]:
        """Calculate standard Z-Score (original or modified version)"""
        try:
            # Calculate components
            # A = Working Capital / Total Assets
            working_capital = self._get_value_with_alternatives(balance,
                                                                ['Total Current Assets',
                                                                 'Current Assets']) - self._get_value_with_alternatives(
                balance,
                ['Total Current Liabilities', 'Current Liabilities'])
            total_assets = self._get_value_with_alternatives(balance, ['Total Assets', 'Assets'])
            component_a = working_capital / total_assets if total_assets != 0 else 0

            # B = Retained Earnings / Total Assets
            retained_earnings = self._get_value_with_alternatives(balance, ['Retained Earnings',
                                                                            'Accumulated Retained Earnings'])
            component_b = retained_earnings / total_assets if total_assets != 0 else 0

            # C = EBIT / Total Assets
            ebit = self._get_value_with_alternatives(income, ['Operating Income', 'EBIT', 'Income From Operations'])
            component_c = ebit / total_assets if total_assets != 0 else 0

            # D = Market Value of Equity / Total Liabilities
            if model_variant == "original":
                # For original model, try to use market cap if available
                market_cap = market_data.get('market_cap')
                if market_cap is None or market_cap == 0:
                    # If market cap not available, use book value of equity
                    market_cap = self._get_value_with_alternatives(balance,
                                                                   ['Total Stockholder Equity', 'Total Equity',
                                                                    'Shareholders Equity'])
            else:
                # For modified model, use book value of equity
                market_cap = self._get_value_with_alternatives(balance,
                                                               ['Total Stockholder Equity', 'Total Equity',
                                                                'Shareholders Equity'])

            total_liabilities = self._get_value_with_alternatives(balance,
                                                                  ['Total Liabilities',
                                                                   'Total Liabilities Net Minority Interest'])
            component_d = market_cap / total_liabilities if total_liabilities != 0 else 0

            # E = Sales / Total Assets
            sales = self._get_value_with_alternatives(income, ['Total Revenue', 'Revenue', 'Sales', 'Net Sales'])
            component_e = sales / total_assets if total_assets != 0 else 0

            # Calculate Z-Score based on model variant
            if model_variant == "original":
                # Original Z-Score: 1.2*A + 1.4*B + 3.3*C + 0.6*D + 0.999*E
                z_score = 1.2 * component_a + 1.4 * component_b + 3.3 * component_c + 0.6 * component_d + 0.999 * component_e
            else:
                # Modified Z-Score: 6.56*A + 3.26*B + 6.72*C + 1.05*D
                z_score = 6.56 * component_a + 3.26 * component_b + 6.72 * component_c + 1.05 * component_d

            # Assess bankruptcy risk
            thresholds = self.risk_thresholds["z_score"][model_variant]
            safe_threshold = thresholds["safe"]
            grey_threshold = thresholds["grey"]

            assessment = None
            description = None
            color = None

            if z_score > safe_threshold:
                assessment = "safe"
                description = "Low bankruptcy risk (Safe Zone)"
                color = self.risk_colors["safe"]
            elif z_score > grey_threshold:
                assessment = "grey"
                description = "Moderate bankruptcy risk (Grey Zone)"
                color = self.risk_colors["grey"]
            else:
                assessment = "distress"
                description = "High bankruptcy risk (Distress Zone)"
                color = self.risk_colors["distress"]

            return {
                'score': z_score,
                'components': {
                    'working_capital_to_total_assets': component_a,
                    'retained_earnings_to_total_assets': component_b,
                    'ebit_to_total_assets': component_c,
                    'equity_to_total_liabilities': component_d,
                    'sales_to_total_assets': component_e
                },
                'assessment': assessment,
                'description': description,
                'color': color,
                'thresholds': {
                    'safe': safe_threshold,
                    'grey': grey_threshold
                },
                'model': f"Altman Z-Score ({model_variant})",
                'explanation': self._get_z_score_explanation(z_score, model_variant, component_a, component_b,
                                                             component_c, component_d, component_e)
            }
        except Exception as e:
            logger.error(f"Error in standard Z-Score calculation: {e}")
            return self._get_empty_z_score_result()

    def _calculate_financial_z_score(self, income: pd.Series, balance: pd.Series) -> Dict[str, Any]:
        """Calculate Z-Score variant for financial institutions"""
        try:
            # Calculate components for financial sector model
            # Modified components for banks and financial institutions
            # A = Equity / Total Assets
            equity = self._get_value_with_alternatives(balance,
                                                       ['Total Stockholder Equity', 'Total Equity',
                                                        'Shareholders Equity'])
            total_assets = self._get_value_with_alternatives(balance, ['Total Assets', 'Assets'])
            component_a = equity / total_assets if total_assets != 0 else 0

            # B = EBIT / Total Assets
            ebit = self._get_value_with_alternatives(income, ['Operating Income', 'EBIT', 'Income From Operations'])
            component_b = ebit / total_assets if total_assets != 0 else 0

            # C = Retained Earnings / Total Assets
            retained_earnings = self._get_value_with_alternatives(balance,
                                                                  ['Retained Earnings',
                                                                   'Accumulated Retained Earnings'])
            component_c = retained_earnings / total_assets if total_assets != 0 else 0

            # D = Net Interest Income / Total Assets
            interest_income = self._get_value_with_alternatives(income,
                                                                ['Interest Income', 'Interest Revenue']) or 0
            interest_expense = abs(self._get_value_with_alternatives(income,
                                                                     ['Interest Expense',
                                                                      'Interest and Debt Expense',
                                                                      'Interest Paid']) or 0)
            net_interest_income = interest_income - interest_expense
            component_d = net_interest_income / total_assets if total_assets != 0 else 0

            # Financial Z-Score: 3.25 + 6.56*A + 3.26*B + 6.72*C + 1.05*D
            z_score = 3.25 + 6.56 * component_a + 3.26 * component_b + 6.72 * component_c + 1.05 * component_d

            # Assess bankruptcy risk
            thresholds = self.risk_thresholds["z_score"]["financial"]
            safe_threshold = thresholds["safe"]
            grey_threshold = thresholds["grey"]

            assessment = None
            description = None
            color = None

            if z_score > safe_threshold:
                assessment = "safe"
                description = "Low bankruptcy risk (Safe Zone)"
                color = self.risk_colors["safe"]
            elif z_score > grey_threshold:
                assessment = "grey"
                description = "Moderate bankruptcy risk (Grey Zone)"
                color = self.risk_colors["grey"]
            else:
                assessment = "distress"
                description = "High bankruptcy risk (Distress Zone)"
                color = self.risk_colors["distress"]

            return {
                'score': z_score,
                'components': {
                    'equity_to_total_assets': component_a,
                    'ebit_to_total_assets': component_b,
                    'retained_earnings_to_total_assets': component_c,
                    'net_interest_income_to_total_assets': component_d
                },
                'assessment': assessment,
                'description': description,
                'color': color,
                'thresholds': {
                    'safe': safe_threshold,
                    'grey': grey_threshold
                },
                'model': "Altman Z-Score (financial)",
                'explanation': self._get_financial_z_score_explanation(z_score, component_a, component_b,
                                                                       component_c, component_d)
            }
        except Exception as e:
            logger.error(f"Error in financial Z-Score calculation: {e}")
            return self._get_empty_z_score_result()

    def _get_value_with_alternatives(self, series: pd.Series, keys: List[str]) -> float:
        """
        Try to get a value from a series using a list of alternative keys

        Args:
            series: The pandas Series to search
            keys: List of alternative keys to try

        Returns:
            Value if found, None otherwise
        """
        if series is None:
            return None

        # Try direct lookup for each key
        for key in keys:
            if key in series:
                value = series[key]
                return float(value) if pd.notna(value) else None

        # Try case-insensitive lookup
        keys_lower = [k.lower() for k in keys]
        for idx in series.index:
            if isinstance(idx, str) and idx.lower() in keys_lower:
                value = series[idx]
                return float(value) if pd.notna(value) else None

        # Try partial match
        for idx in series.index:
            if isinstance(idx, str):
                for key in keys:
                    if key.lower() in idx.lower():
                        value = series[idx]
                        return float(value) if pd.notna(value) else None

        return None

    def _get_z_score_explanation(self, z_score: float, model_variant: str,
                                 component_a: float, component_b: float, component_c: float,
                                 component_d: float, component_e: float = None) -> str:
        """
        Generate an explanation of the Z-Score result

        Args:
            z_score: Calculated Z-Score
            model_variant: Which Z-Score variant was used
            component_a: Working Capital / Total Assets
            component_b: Retained Earnings / Total Assets
            component_c: EBIT / Total Assets
            component_d: Market Value of Equity / Total Liabilities
            component_e: Sales / Total Assets (only used in original model)

        Returns:
            Explanation text
        """
        # Get thresholds based on model variant
        thresholds = self.risk_thresholds["z_score"][model_variant]
        safe_threshold = thresholds["safe"]
        grey_threshold = thresholds["grey"]

        # Base explanation of the model and score
        if model_variant == "original":
            base_explanation = (
                f"The Altman Z-Score of {z_score:.2f} is calculated using the original model "
                f"created for manufacturing companies. "
            )
        elif model_variant == "modified":
            base_explanation = (
                f"The Altman Z-Score of {z_score:.2f} is calculated using the modified model "
                f"which is suitable for private companies and non-manufacturing firms. "
            )
        else:  # financial
            base_explanation = (
                f"The Altman Z-Score of {z_score:.2f} is calculated using the specialized model "
                f"adapted for financial institutions. "
            )

        # Risk assessment explanation
        if z_score > safe_threshold:
            risk_explanation = (
                f"This score is above {safe_threshold:.2f}, indicating that the company is in the 'Safe Zone' "
                f"with a low probability of financial distress in the next two years. "
            )
        elif z_score > grey_threshold:
            risk_explanation = (
                f"This score is between {grey_threshold:.2f} and {safe_threshold:.2f}, placing the company in the 'Grey Zone'. "
                f"The model cannot make a clear prediction, suggesting a moderate risk of financial distress. "
                f"Further analysis is recommended. "
            )
        else:
            risk_explanation = (
                f"This score is below {grey_threshold:.2f}, placing the company in the 'Distress Zone' "
                f"with a higher likelihood of financial distress or bankruptcy within two years. "
                f"Immediate attention to financial health is warranted. "
            )

        # Component analysis to identify strengths/weaknesses
        component_analysis = "Analysis of individual components: "

        # Analyze component A: Working Capital / Total Assets (liquidity)
        if component_a < 0:
            component_analysis += (
                "The company has negative working capital, indicating potential immediate liquidity problems. "
            )
        elif component_a < 0.1:
            component_analysis += (
                "The company has low liquidity with minimal working capital relative to assets. "
            )
        else:
            component_analysis += (
                "The company's liquidity position appears adequate. "
            )

        # Analyze component B: Retained Earnings / Total Assets (profitability over time)
        if component_b < 0:
            component_analysis += (
                "The company has accumulated losses, which may indicate a history of unprofitability. "
            )
        elif component_b < 0.1:
            component_analysis += (
                "The company has a limited history of retained profits. "
            )
        else:
            component_analysis += (
                "The company has demonstrated ability to generate and retain earnings. "
            )

        # Analyze component C: EBIT / Total Assets (operating performance)
        if component_c < 0:
            component_analysis += (
                "The company is generating operating losses, a significant concern. "
            )
        elif component_c < 0.05:
            component_analysis += (
                "The company's operating profitability is quite low relative to assets. "
            )
        else:
            component_analysis += (
                "The company is generating good operating returns from its assets. "
            )

        # Analyze component D: Market Value of Equity / Total Liabilities (solvency)
        if component_d < 0.5:
            component_analysis += (
                "The company has high debt relative to its equity value, a potential solvency concern. "
            )
        elif component_d < 1:
            component_analysis += (
                "The company's debt levels are moderate relative to equity. "
            )
        else:
            component_analysis += (
                "The company has strong equity value relative to its debt obligations. "
            )

        # For original model, also analyze component E: Sales / Total Assets (efficiency)
        if model_variant == "original" and component_e is not None:
            if component_e < 0.5:
                component_analysis += (
                    "The company is generating low sales relative to assets, indicating poor efficiency. "
                )
            elif component_e < 1:
                component_analysis += (
                    "The company's asset efficiency is moderate. "
                )
            else:
                component_analysis += (
                    "The company is generating strong sales relative to its asset base. "
                )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_financial_z_score_explanation(self, z_score: float, component_a: float,
                                           component_b: float, component_c: float, component_d: float) -> str:
        """
        Generate an explanation of the Financial Z-Score result

        Args:
            z_score: Calculated Financial Z-Score
            component_a: Equity / Total Assets
            component_b: EBIT / Total Assets
            component_c: Retained Earnings / Total Assets
            component_d: Net Interest Income / Total Assets

        Returns:
            Explanation text
        """
        # Get thresholds
        thresholds = self.risk_thresholds["z_score"]["financial"]
        safe_threshold = thresholds["safe"]
        grey_threshold = thresholds["grey"]

        # Base explanation
        base_explanation = (
            f"The Altman Z-Score of {z_score:.2f} is calculated using the specialized model "
            f"adapted for financial institutions. "
        )

        # Risk assessment explanation
        if z_score > safe_threshold:
            risk_explanation = (
                f"This score is above {safe_threshold:.2f}, indicating that the financial institution is in the 'Safe Zone' "
                f"with a low probability of financial distress. "
            )
        elif z_score > grey_threshold:
            risk_explanation = (
                f"This score is between {grey_threshold:.2f} and {safe_threshold:.2f}, placing the institution in the 'Grey Zone'. "
                f"The model suggests a moderate risk of financial distress. Further analysis is recommended. "
            )
        else:
            risk_explanation = (
                f"This score is below {grey_threshold:.2f}, placing the financial institution in the 'Distress Zone' "
                f"with a higher likelihood of financial difficulties. Immediate attention to financial health is warranted. "
            )

        # Component analysis specific to financial institutions
        component_analysis = "Analysis of individual components: "

        # Analyze component A: Equity / Total Assets (capital adequacy)
        if component_a < 0.08:
            component_analysis += (
                "The institution's capital adequacy is concerning, with low equity relative to assets. "
            )
        elif component_a < 0.12:
            component_analysis += (
                "The institution's capital buffer is moderate, potentially limiting its ability to absorb losses. "
            )
        else:
            component_analysis += (
                "The institution has a solid capital base relative to its assets. "
            )

        # Analyze component B: EBIT / Total Assets (operating performance)
        if component_b < 0.005:
            component_analysis += (
                "The institution's operating profitability is very low, a potential concern. "
            )
        elif component_b < 0.01:
            component_analysis += (
                "The institution's operating profitability is modest. "
            )
        else:
            component_analysis += (
                "The institution is generating good operating returns. "
            )

        # Analyze component C: Retained Earnings / Total Assets (stability)
        if component_c < 0.02:
            component_analysis += (
                "The institution has limited accumulated earnings, indicating potential long-term instability. "
            )
        elif component_c < 0.05:
            component_analysis += (
                "The institution has a moderate level of accumulated earnings. "
            )
        else:
            component_analysis += (
                "The institution has strong accumulated earnings, suggesting historical stability. "
            )

        # Analyze component D: Net Interest Income / Total Assets (core profitability)
        if component_d < 0.02:
            component_analysis += (
                "The institution's net interest margins are very thin, which may impact profitability. "
            )
        elif component_d < 0.03:
            component_analysis += (
                "The institution has moderate net interest margins. "
            )
        else:
            component_analysis += (
                "The institution has strong net interest margins relative to assets. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_springate_explanation(self, springate_score: float, component_a: float,
                                   component_b: float, component_c: float, component_d: float) -> str:
        """
        Generate an explanation of the Springate model result

        Args:
            springate_score: Calculated Springate score
            component_a: Working Capital / Total Assets
            component_b: EBIT / Total Assets
            component_c: EBT / Current Liabilities
            component_d: Sales / Total Assets

        Returns:
            Explanation text
        """
        threshold = self.risk_thresholds["springate"]["threshold"]

        # Base explanation
        base_explanation = (
            f"The Springate score of {springate_score:.2f} is derived from a model developed in 1978 by Gordon Springate. "
            f"This model uses multiple financial ratios to assess bankruptcy risk. "
        )

        # Risk assessment explanation
        if springate_score > threshold:
            risk_explanation = (
                f"This score is above the threshold of {threshold:.3f}, suggesting that the company "
                f"is financially healthy and has a low probability of business failure. "
            )
        else:
            risk_explanation = (
                f"This score is below the threshold of {threshold:.3f}, indicating that the company "
                f"may face financial distress and has characteristics similar to companies that have failed. "
            )

        # Component analysis
        component_analysis = "Analysis of individual components: "

        # Analyze component A: Working Capital / Total Assets (liquidity)
        if component_a < 0:
            component_analysis += (
                "The company has negative working capital, indicating immediate liquidity challenges. "
            )
        elif component_a < 0.1:
            component_analysis += (
                "The company's working capital is low relative to assets, suggesting limited liquidity. "
            )
        else:
            component_analysis += (
                "The company has adequate working capital relative to assets. "
            )

        # Analyze component B: EBIT / Total Assets (profitability)
        if component_b < 0:
            component_analysis += (
                "The company is generating operating losses, which significantly impacts its score. "
            )
        elif component_b < 0.05:
            component_analysis += (
                "The company's operating profitability is weak relative to assets. "
            )
        else:
            component_analysis += (
                "The company is achieving good operating returns on its assets. "
            )

        # Analyze component C: EBT / Current Liabilities (short-term debt servicing)
        if component_c < 0:
            component_analysis += (
                "The company's pre-tax losses suggest potential issues in servicing short-term obligations. "
            )
        elif component_c < 0.1:
            component_analysis += (
                "The company's ability to service short-term debt from earnings appears limited. "
            )
        else:
            component_analysis += (
                "The company has good capacity to cover short-term obligations from earnings. "
            )

        # Analyze component D: Sales / Total Assets (efficiency)
        if component_d < 0.7:
            component_analysis += (
                "The company is generating relatively low sales from its asset base. "
            )
        elif component_d < 1.5:
            component_analysis += (
                "The company's asset efficiency is moderate. "
            )
        else:
            component_analysis += (
                "The company is efficient at generating sales from its assets. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_zmijewski_explanation(self, probability: float, roa: float, leverage: float, liquidity: float) -> str:
        """
        Generate an explanation of the Zmijewski model result

        Args:
            probability: Calculated bankruptcy probability
            roa: Return on Assets
            leverage: Total Liabilities / Total Assets
            liquidity: Current Ratio

        Returns:
            Explanation text
        """
        threshold = self.risk_thresholds["zmijewski"]["threshold"]

        # Base explanation
        base_explanation = (
            f"The Zmijewski model calculates a bankruptcy probability of {probability:.2%}. "
            f"This model, developed in 1984, uses a statistical technique called probit analysis "
            f"to predict bankruptcy based on financial ratios. "
        )

        # Risk assessment explanation
        if probability < threshold:
            risk_explanation = (
                f"This probability is below the typical threshold of {threshold:.0%}, suggesting "
                f"that the company has a lower risk of financial failure. "
            )
        else:
            risk_explanation = (
                f"This probability is above the typical threshold of {threshold:.0%}, indicating "
                f"that the company has characteristics associated with higher bankruptcy risk. "
            )

        # Component analysis
        component_analysis = "Analysis of individual components: "

        # Analyze ROA: Net Income / Total Assets
        if roa < 0:
            component_analysis += (
                "The company is unprofitable with a negative return on assets, which significantly increases risk. "
            )
        elif roa < 0.02:
            component_analysis += (
                "The company's profitability is low, with return on assets below typical sustainable levels. "
            )
        else:
            component_analysis += (
                "The company's return on assets is positive, which reduces bankruptcy risk. "
            )

        # Analyze leverage: Total Liabilities / Total Assets
        if leverage > 0.9:
            component_analysis += (
                "The company has very high financial leverage with liabilities covering most of assets, a major risk factor. "
            )
        elif leverage > 0.7:
            component_analysis += (
                "The company has significant financial leverage, which increases risk. "
            )
        else:
            component_analysis += (
                "The company's financial leverage is at manageable levels. "
            )

        # Analyze liquidity: Current Ratio
        if liquidity < 1:
            component_analysis += (
                "The company's liquidity is concerning, with current assets insufficient to cover current liabilities. "
            )
        elif liquidity < 1.5:
            component_analysis += (
                "The company's liquidity is moderate, providing some buffer for short-term obligations. "
            )
        else:
            component_analysis += (
                "The company has good liquidity with strong coverage of short-term obligations. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_taffler_explanation(self, taffler_score: float, r1: float, r2: float, r3: float, r4: float) -> str:
        """
        Generate an explanation of the Taffler model result

        Args:
            taffler_score: Calculated Taffler Z-Score
            r1: Profit Before Tax / Current Liabilities
            r2: Current Assets / Total Liabilities
            r3: Current Liabilities / Total Assets
            r4: No-Credit Interval

        Returns:
            Explanation text
        """
        safe_threshold = self.risk_thresholds["taffler"]["threshold"]
        grey_threshold = self.risk_thresholds["taffler"]["grey"]

        # Base explanation
        base_explanation = (
            f"The Taffler Z-Score of {taffler_score:.2f} is calculated using a model developed by Richard Taffler "
            f"in 1983 specifically for UK companies. The model focuses on liquidity and financial risk. "
        )

        # Risk assessment explanation
        if taffler_score > safe_threshold:
            risk_explanation = (
                f"This score is above {safe_threshold:.2f}, indicating a low probability of financial failure. "
                f"Companies with scores in this range typically have solid financial health. "
            )
        elif taffler_score > grey_threshold:
            risk_explanation = (
                f"This score is between {grey_threshold:.2f} and {safe_threshold:.2f}, placing the company in a gray area. "
                f"Further analysis is recommended to assess financial risk. "
            )
        else:
            risk_explanation = (
                f"This score is below {grey_threshold:.2f}, suggesting that the company exhibits characteristics "
                f"of firms that have experienced financial distress. Close monitoring is advisable. "
            )

        # Component analysis
        component_analysis = "Analysis of individual components: "

        # Analyze R1: Profit Before Tax / Current Liabilities (profitability)
        if r1 < 0:
            component_analysis += (
                "The company's pre-tax losses relative to current liabilities are a significant concern. "
            )
        elif r1 < 0.1:
            component_analysis += (
                "The company's profits are relatively low compared to its short-term obligations. "
            )
        else:
            component_analysis += (
                "The company has good profitability relative to short-term obligations. "
            )

        # Analyze R2: Current Assets / Total Liabilities (financial risk)
        if r2 < 0.5:
            component_analysis += (
                "The company has low current assets relative to total liabilities, indicating higher financial risk. "
            )
        elif r2 < 1:
            component_analysis += (
                "The company's current assets provide moderate coverage of total liabilities. "
            )
        else:
            component_analysis += (
                "The company has good asset coverage of total liabilities. "
            )

        # Analyze R3: Current Liabilities / Total Assets (liquidity position)
        if r3 > 0.5:
            component_analysis += (
                "A large portion of the company's assets is financed by short-term obligations, which may create liquidity pressure. "
            )
        elif r3 > 0.3:
            component_analysis += (
                "The company has a moderate level of short-term obligations relative to assets. "
            )
        else:
            component_analysis += (
                "The company has a comfortable level of short-term obligations relative to total assets. "
            )

        # Analyze R4: No-Credit Interval (liquidity under pressure)
        if r4 < -0.2:
            component_analysis += (
                "The company may face challenges meeting operational expenses if credit becomes unavailable. "
            )
        elif r4 < 0:
            component_analysis += (
                "The company has limited ability to continue operations without credit access. "
            )
        else:
            component_analysis += (
                "The company appears to have sufficient quick assets to cover operations even if credit becomes restricted. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_ohlson_explanation(self, probability: float, size: float, tlta: float, wcta: float, nita: float) -> str:
        """
        Generate an explanation of the Ohlson O-Score model result

        Args:
            probability: Calculated bankruptcy probability
            size: Log of Total Assets
            tlta: Total Liabilities / Total Assets
            wcta: Working Capital / Total Assets
            nita: Net Income / Total Assets

        Returns:
            Explanation text
        """
        threshold = self.risk_thresholds["ohlson"]["threshold"]

        # Base explanation
        base_explanation = (
            f"The Ohlson model estimates a bankruptcy probability of {probability:.2%}. "
            f"Developed in 1980 by James Ohlson, this model uses logistic regression on multiple financial ratios "
            f"to predict business failure. "
        )

        # Risk assessment explanation
        if probability < threshold:
            risk_explanation = (
                f"This probability is below the typical threshold of {threshold:.0%}, suggesting "
                f"that the company has a lower risk of financial failure within one year. "
            )
        else:
            risk_explanation = (
                f"This probability is above the typical threshold of {threshold:.0%}, indicating "
                f"that the company has an elevated risk of financial distress. "
            )

        # Component analysis (focusing on the most significant components)
        component_analysis = "Analysis of key components: "

        # Analyze company size
        if size < 8:  # Corresponds to assets of approximately $3M
            component_analysis += (
                "The company's small size may make it more vulnerable to economic shocks. "
            )
        elif size < 10:  # Corresponds to assets of approximately $22M
            component_analysis += (
                "The company's moderate size provides some stability, but size-related vulnerabilities may exist. "
            )
        else:
            component_analysis += (
                "The company's larger size generally reduces bankruptcy risk. "
            )

        # Analyze total liabilities to total assets (leverage)
        if tlta > 0.9:
            component_analysis += (
                "The company has very high leverage with liabilities covering most of its assets, a significant risk factor. "
            )
        elif tlta > 0.7:
            component_analysis += (
                "The company has substantial leverage, which increases financial risk. "
            )
        else:
            component_analysis += (
                "The company's leverage is at manageable levels. "
            )

        # Analyze working capital to total assets (liquidity)
        if wcta < 0:
            component_analysis += (
                "The company has negative working capital, indicating potential short-term liquidity challenges. "
            )
        elif wcta < 0.1:
            component_analysis += (
                "The company's working capital buffer is limited relative to assets. "
            )
        else:
            component_analysis += (
                "The company has adequate working capital, which helps reduce bankruptcy risk. "
            )

        # Analyze net income to total assets (profitability)
        if nita < 0:
            component_analysis += (
                "The company's negative return on assets significantly increases bankruptcy risk. "
            )
        elif nita < 0.03:
            component_analysis += (
                "The company's profitability is relatively low, which can increase vulnerability. "
            )
        else:
            component_analysis += (
                "The company's profitability is at healthy levels, which helps mitigate bankruptcy risk. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _get_fulmer_explanation(self, h_score: float, v1: float, v2: float, v4: float, v8: float) -> str:
        """
        Generate an explanation of the Fulmer H-Factor model result

        Args:
            h_score: Calculated Fulmer H-Factor score
            v1: Retained Earnings / Total Assets
            v2: Sales / Total Assets
            v4: Cash Flow / Total Debt
            v8: Working Capital / Total Debt

        Returns:
            Explanation text
        """
        threshold = self.risk_thresholds["fulmer"]["threshold"]

        # Base explanation
        base_explanation = (
            f"The Fulmer H-Factor score of {h_score:.2f} is based on a model that uses "
            f"nine financial variables to predict business failure. "
            f"It was designed to be effective with smaller companies and has shown good accuracy in some studies. "
        )

        # Risk assessment explanation
        if h_score > threshold:
            risk_explanation = (
                f"This score is above the threshold of {threshold:.1f}, suggesting "
                f"that the company has a lower probability of business failure. "
            )
        else:
            risk_explanation = (
                f"This score is below the threshold of {threshold:.1f}, indicating "
                f"that the company has characteristics associated with higher failure risk. "
            )

        # Component analysis (focusing on the most significant components)
        component_analysis = "Analysis of key components: "

        # Analyze retained earnings to total assets (cumulative profitability)
        if v1 < 0:
            component_analysis += (
                "The company has accumulated losses rather than retained earnings, suggesting a history of unprofitability. "
            )
        elif v1 < 0.1:
            component_analysis += (
                "The company has limited retained earnings relative to assets, indicating modest historical profitability. "
            )
        else:
            component_analysis += (
                "The company has strong retained earnings relative to assets, demonstrating good historical profitability. "
            )

        # Analyze sales to total assets (efficiency)
        if v2 < 0.7:
            component_analysis += (
                "The company generates relatively low sales from its asset base, suggesting efficiency concerns. "
            )
        elif v2 < 1.5:
            component_analysis += (
                "The company's asset efficiency is moderate. "
            )
        else:
            component_analysis += (
                "The company generates strong sales from its asset base. "
            )

        # Analyze cash flow to total debt (debt serviceability)
        if v4 < 0:
            component_analysis += (
                "The company's negative cash flow relative to debt is a significant concern for debt serviceability. "
            )
        elif v4 < 0.2:
            component_analysis += (
                "The company's cash flow provides limited coverage of debt obligations. "
            )
        else:
            component_analysis += (
                "The company's cash flow provides good coverage of debt obligations. "
            )

        # Analyze working capital to total debt (liquidity)
        if v8 < 0:
            component_analysis += (
                "The company's negative working capital relative to debt raises liquidity concerns. "
            )
        elif v8 < 0.3:
            component_analysis += (
                "The company's working capital provides limited buffer relative to debt obligations. "
            )
        else:
            component_analysis += (
                "The company has good working capital relative to debt, enhancing financial flexibility. "
            )

        # Combine all parts
        full_explanation = base_explanation + risk_explanation + component_analysis
        return full_explanation

    def _analyze_risk_factors(self, financial_data: Dict, model_results: Dict) -> List[Dict[str, str]]:
        """
        Analyze specific risk factors based on financial data and model results

        Args:
            financial_data: Dictionary with financial statements
            model_results: Dictionary with bankruptcy model results

        Returns:
            List of risk factors with descriptions and severities
        """
        risk_factors = []

        # Extract data needed for analysis
        income_stmt = financial_data.get('income_statement')
        balance_sheet = financial_data.get('balance_sheet')

        if not isinstance(income_stmt, pd.DataFrame) or not isinstance(balance_sheet, pd.DataFrame):
            return risk_factors

        # Get most recent financial data
        income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
        balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()

        # Extract key financial metrics
        net_income = self._get_value_with_alternatives(income, ['Net Income', 'Net Profit', 'Net Earnings'])
        ebit = self._get_value_with_alternatives(income, ['Operating Income', 'EBIT', 'Income From Operations'])
        total_assets = self._get_value_with_alternatives(balance, ['Total Assets', 'Assets'])
        total_liabilities = self._get_value_with_alternatives(balance,
                                                              ['Total Liabilities',
                                                               'Total Liabilities Net Minority Interest'])
        current_assets = self._get_value_with_alternatives(balance, ['Total Current Assets', 'Current Assets'])
        current_liabilities = self._get_value_with_alternatives(balance,
                                                                ['Total Current Liabilities', 'Current Liabilities'])
        cash = self._get_value_with_alternatives(balance, ['Cash and Cash Equivalents', 'Cash'])
        equity = self._get_value_with_alternatives(balance,
                                                   ['Total Stockholder Equity', 'Total Equity', 'Shareholders Equity'])
        retained_earnings = self._get_value_with_alternatives(balance,
                                                              ['Retained Earnings', 'Accumulated Retained Earnings'])

        # Check for negative net income (losses)
        if net_income is not None and net_income < 0:
            severity = "high" if abs(net_income) > 0.05 * total_assets else "medium"
            risk_factors.append({
                "factor": "Operating Losses",
                "description": "The company is currently unprofitable, which can deplete resources over time.",
                "severity": severity
            })

        # Check for negative equity (technical insolvency)
        if equity is not None and equity < 0:
            risk_factors.append({
                "factor": "Negative Equity",
                "description": "The company's liabilities exceed its assets, indicating technical insolvency.",
                "severity": "critical"
            })

        # Check for negative working capital
        if current_assets is not None and current_liabilities is not None:
            working_capital = current_assets - current_liabilities
            if working_capital < 0:
                severity = "high" if abs(working_capital) > 0.1 * total_assets else "medium"
                risk_factors.append({
                    "factor": "Negative Working Capital",
                    "description": "Current liabilities exceed current assets, indicating potential liquidity problems.",
                    "severity": severity
                })

        # Check for accumulated losses
        if retained_earnings is not None and retained_earnings < 0:
            severity = "high" if abs(retained_earnings) > 0.2 * total_assets else "medium"
            risk_factors.append({
                "factor": "Accumulated Losses",
                "description": "The company has accumulated losses over time rather than retained earnings.",
                "severity": severity
            })

        # Check for high debt relative to assets
        if total_liabilities is not None and total_assets is not None and total_assets > 0:
            debt_to_assets = total_liabilities / total_assets
            if debt_to_assets > 0.8:
                severity = "critical" if debt_to_assets > 0.9 else "high"
                risk_factors.append({
                    "factor": "High Leverage",
                    "description": f"Debt represents {debt_to_assets:.0%} of total assets, indicating high financial risk.",
                    "severity": severity
                })

                # Check for low cash relative to current liabilities
            if cash is not None and current_liabilities is not None and current_liabilities > 0:
                cash_ratio = cash / current_liabilities
                if cash_ratio < 0.1:
                    severity = "high" if cash_ratio < 0.05 else "medium"
                    risk_factors.append({
                        "factor": "Low Cash Reserves",
                        "description": f"Cash covers only {cash_ratio:.0%} of current liabilities, suggesting limited liquidity buffer.",
                        "severity": severity
                    })

                # Check for inability to service debt from operations
            if ebit is not None and total_liabilities is not None and total_liabilities > 0:
                debt_service_capability = ebit / total_liabilities
                if debt_service_capability < 0.05:
                    severity = "high" if debt_service_capability < 0 else "medium"
                    risk_factors.append({
                        "factor": "Poor Debt Service Capability",
                        "description": "Operating income is insufficient to adequately service debt obligations.",
                        "severity": severity
                    })

                # Add risk factors based on model results
                # Check Altman Z-Score
            z_score_result = model_results.get('altman_z_score', {})
            if z_score_result.get('assessment') == 'distress':
                risk_factors.append({
                    "factor": "Altman Z-Score Distress",
                    "description": "The Altman Z-Score indicates high bankruptcy risk based on overall financial structure.",
                    "severity": "high"
                })

            # Check Zmijewski model
            zmijewski_result = model_results.get('zmijewski', {})
            if zmijewski_result.get('probability') is not None and zmijewski_result.get('probability') > 0.7:
                risk_factors.append({
                    "factor": "High Bankruptcy Probability (Zmijewski)",
                    "description": f"The Zmijewski model indicates a {zmijewski_result.get('probability'):.0%} probability of bankruptcy.",
                    "severity": "high"
                })

            # Add model consistency factor if multiple models show distress
            distress_count = sum(1 for model, result in model_results.items()
                                 if result.get('assessment') == 'distress')
            total_models = sum(1 for model, result in model_results.items()
                               if result.get('assessment') is not None)

            if distress_count >= 2 and total_models >= 3:
                risk_factors.append({
                    "factor": "Multiple Model Warning",
                    "description": f"{distress_count} out of {total_models} bankruptcy models indicate financial distress.",
                    "severity": "high"
                })

        return risk_factors

    def _get_recommended_actions(self, risk_level: str, risk_factors: List[Dict[str, str]]) -> List[str]:
        """
        Generate recommended actions based on risk level and specific risk factors

        Args:
            risk_level: Overall risk assessment (safe, grey, distress)
            risk_factors: List of identified risk factors

        Returns:
            List of recommended actions
        """
        recommendations = []

        # Base recommendations based on overall risk level
        if risk_level == "safe":
            recommendations.append(
                "Continue regular financial monitoring and maintain current financial discipline."
            )
            recommendations.append(
                "Consider proactive measures to further strengthen financial position."
            )
        elif risk_level == "grey":
            recommendations.append(
                "Implement more frequent financial monitoring and establish early warning indicators."
            )
            recommendations.append(
                "Develop contingency plans to address potential liquidity or operational challenges."
            )
        else:  # distress
            recommendations.append(
                "Consider engaging financial restructuring advisors to evaluate strategic options."
            )
            recommendations.append(
                "Implement immediate action plan to address critical financial vulnerabilities."
            )

        # Add specific recommendations based on risk factors
        factor_types = [factor["factor"] for factor in risk_factors]

        if "Operating Losses" in factor_types:
            recommendations.append(
                "Conduct detailed cost structure analysis to identify cost reduction opportunities."
            )
            recommendations.append(
                "Evaluate product/service profitability and consider eliminating underperforming offerings."
            )

        if "Negative Working Capital" in factor_types or "Low Cash Reserves" in factor_types:
            recommendations.append(
                "Improve working capital management through tighter inventory controls, accelerated collections, and optimized payment terms."
            )
            recommendations.append(
                "Secure additional liquidity through credit lines or other financing options before urgent need arises."
            )

        if "High Leverage" in factor_types or "Poor Debt Service Capability" in factor_types:
            recommendations.append(
                "Explore debt restructuring options to extend maturities or adjust terms."
            )
            recommendations.append(
                "Consider asset sales or equity issuance to strengthen balance sheet and reduce debt levels."
            )

        if "Negative Equity" in factor_types:
            recommendations.append(
                "Evaluate recapitalization options to address balance sheet insolvency."
            )
            recommendations.append(
                "Consider strategic alternatives including merger, acquisition, or significant restructuring."
            )

        # Return unique recommendations (remove duplicates)
        return list(dict.fromkeys(recommendations))

    def _analyze_score_trend(self, yearly_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trend in bankruptcy scores over time

        Args:
            yearly_scores: List of yearly scores with assessment

        Returns:
            Dictionary with trend analysis
        """
        if not yearly_scores or len(yearly_scores) < 2:
            return {
                "trend": "insufficient_data",
                "description": "Insufficient historical data to determine trend",
                "trajectory": None
            }

        # Extract scores and calculate changes
        scores = [item.get('z_score') for item in yearly_scores if item.get('z_score') is not None]
        years = [item.get('year') for item in yearly_scores if item.get('z_score') is not None]

        if len(scores) < 2:
            return {
                "trend": "insufficient_data",
                "description": "Insufficient historical data to determine trend",
                "trajectory": None
            }

        # Calculate year-over-year changes
        changes = [scores[i] - scores[i - 1] for i in range(1, len(scores))]

        # Calculate average change and determine trend direction
        avg_change = sum(changes) / len(changes)

        # Get the first and last assessment
        first_assessment = next((item.get('assessment') for item in yearly_scores if item.get('assessment')), None)
        last_assessment = next(
            (item.get('assessment') for item in reversed(yearly_scores) if item.get('assessment')), None)

        # Determine if there was a zone change
        zone_change = first_assessment != last_assessment

        # Analyze trend
        if avg_change > 0.2:
            trend = "strong_improvement"
            description = "Financial health has shown strong improvement over the analyzed period."
            trajectory = "positive"
        elif avg_change > 0.05:
            trend = "moderate_improvement"
            description = "Financial health has shown moderate improvement over time."
            trajectory = "positive"
        elif avg_change > -0.05:
            trend = "stable"
            description = "Financial health has remained relatively stable over the analyzed period."
            trajectory = "neutral"
        elif avg_change > -0.2:
            trend = "moderate_deterioration"
            description = "Financial health has shown moderate deterioration over time."
            trajectory = "negative"
        else:
            trend = "significant_deterioration"
            description = "Financial health has deteriorated significantly over the analyzed period."
            trajectory = "negative"

        # Add information about zone changes
        if zone_change:
            if last_assessment == "safe" and first_assessment in ["grey", "distress"]:
                description += " The company has moved from a higher risk zone to the safe zone."
            elif last_assessment == "grey" and first_assessment == "distress":
                description += " The company has moved from the distress zone to the grey zone."
            elif last_assessment == "distress" and first_assessment in ["safe", "grey"]:
                description += " The company has moved from a lower risk zone to the distress zone."
            elif last_assessment == "grey" and first_assessment == "safe":
                description += " The company has moved from the safe zone to the grey zone."

        # Include specific years in the description
        if years and len(years) >= 2:
            description += f" (Period: {years[0]}-{years[-1]})"

        return {
            "trend": trend,
            "description": description,
            "trajectory": trajectory,
            "average_annual_change": avg_change,
            "years_analyzed": years if years else None
        }

    def _calculate_model_consensus(self, model_assessments: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate the consensus among different bankruptcy models

        Args:
            model_assessments: Dictionary of model names and their assessments

        Returns:
            Dictionary with consensus information
        """
        if not model_assessments:
            return {
                "strength": None,
                "description": "No model data available for consensus analysis"
            }

        # Count occurrences of each assessment
        assessment_counts = {
            "safe": 0,
            "grey": 0,
            "distress": 0
        }

        for assessment in model_assessments.values():
            if assessment in assessment_counts:
                assessment_counts[assessment] += 1

        total_models = sum(assessment_counts.values())

        # Determine the dominant assessment
        dominant_assessment = max(assessment_counts.items(), key=lambda x: x[1])[0]
        dominant_percentage = (assessment_counts[
                                   dominant_assessment] / total_models) * 100 if total_models > 0 else 0

        # Determine consensus strength
        if dominant_percentage >= 80:
            consensus_strength = "strong"
        elif dominant_percentage >= 60:
            consensus_strength = "moderate"
        else:
            consensus_strength = "weak"

        # Generate description
        description = f"{assessment_counts[dominant_assessment]} out of {total_models} models indicate {dominant_assessment} zone "

        if consensus_strength == "strong":
            description += "(strong consensus)"
        elif consensus_strength == "moderate":
            description += "(moderate consensus)"
        else:
            description += "(weak consensus, mixed signals)"

        return {
            "dominant_assessment": dominant_assessment,
            "strength": consensus_strength,
            "description": description,
            "model_count": total_models,
            "assessment_distribution": assessment_counts
        }

    def _get_empty_z_score_result(self) -> Dict[str, Any]:
        """Return empty Z-Score result structure"""
        return {
            'score': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'thresholds': None,
            'model': "Altman Z-Score"
        }

    def _get_empty_springate_result(self) -> Dict[str, Any]:
        """Return empty Springate result structure"""
        return {
            'score': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'threshold': None,
            'model': "Springate"
        }

    def _get_empty_zmijewski_result(self) -> Dict[str, Any]:
        """Return empty Zmijewski result structure"""
        return {
            'score': None,
            'probability': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'threshold': None,
            'model': "Zmijewski"
        }

    def _get_empty_taffler_result(self) -> Dict[str, Any]:
        """Return empty Taffler result structure"""
        return {
            'score': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'thresholds': None,
            'model': "Taffler"
        }

    def _get_empty_ohlson_result(self) -> Dict[str, Any]:
        """Return empty Ohlson result structure"""
        return {
            'score': None,
            'probability': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'threshold': None,
            'model': "Ohlson"
        }

    def _get_empty_fulmer_result(self) -> Dict[str, Any]:
        """Return empty Fulmer result structure"""
        return {
            'score': None,
            'components': {},
            'assessment': None,
            'description': None,
            'color': None,
            'threshold': None,
            'model': "Fulmer H-Factor"
        }