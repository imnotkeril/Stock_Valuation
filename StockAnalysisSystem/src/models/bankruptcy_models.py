import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.config import COLORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bankruptcy_models')


class BankruptcyAnalyzer:
    """
    Class for analyzing bankruptcy risk using various models including:
    - Altman Z-Score
    - Springate Model
    - Zmijewski Model
    - and more sector-specific models
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
            }
        }

        # Define colors for risk categories
        self.risk_colors = {
            "safe": COLORS["success"],
            "grey": COLORS["warning"],
            "distress": COLORS["danger"]
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

            if income_stmt is None or balance_sheet is None:
                logger.warning("Missing financial statements for Z-Score calculation")
                return self._get_empty_z_score_result()

            # Check if DataFrames are not empty
            if isinstance(income_stmt, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame) and \
                    not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                income = income_stmt.iloc[:, 0]
                balance = balance_sheet.iloc[:, 0]

                # Calculate Z-Score components
                if model_variant == "original" or model_variant == "modified":
                    return self._calculate_standard_z_score(income, balance, model_variant)
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
                    working_capital = self._get_value(balance, 'Total Current Assets') - self._get_value(balance,
                                                                                                         'Total Current Liabilities')
                    total_assets = self._get_value(balance, 'Total Assets')
                    component_a = working_capital / total_assets if total_assets != 0 else 0

                    # B = EBIT / Total Assets
                    ebit = self._get_value(income, 'Operating Income')
                    component_b = ebit / total_assets if total_assets != 0 else 0

                    # C = EBT / Current Liabilities
                    ebt = self._get_value(income, 'Income Before Tax')
                    current_liabilities = self._get_value(balance, 'Total Current Liabilities')
                    component_c = ebt / current_liabilities if current_liabilities != 0 else 0

                    # D = Sales / Total Assets
                    sales = self._get_value(income, 'Total Revenue')
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
                        'model': 'Springate'
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
                    net_income = self._get_value(income, 'Net Income')
                    total_assets = self._get_value(balance, 'Total Assets')
                    roa = net_income / total_assets if total_assets != 0 else 0

                    # Financial Leverage = Total Liabilities / Total Assets
                    total_liabilities = self._get_value(balance, 'Total Liabilities')
                    leverage = total_liabilities / total_assets if total_assets != 0 else 0

                    # Liquidity = Current Assets / Current Liabilities
                    current_assets = self._get_value(balance, 'Total Current Assets')
                    current_liabilities = self._get_value(balance, 'Total Current Liabilities')
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
                        'model': 'Zmijewski'
                    }
                except Exception as e:
                    logger.error(f"Error in Zmijewski score calculation: {e}")
                    return self._get_empty_zmijewski_result()

            logger.warning("Invalid or empty financial data for Zmijewski score calculation")
            return self._get_empty_zmijewski_result()

        except Exception as e:
            logger.error(f"Error calculating Zmijewski score: {e}")
            return self._get_empty_zmijewski_result()

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

            # Determine overall risk level
            models_scores = []

            # Add Z-Score assessment (0 for distress, 1 for grey zone, 2 for safe)
            if z_score_results.get('assessment') == 'safe':
                models_scores.append(2)
            elif z_score_results.get('assessment') == 'grey':
                models_scores.append(1)
            elif z_score_results.get('assessment') == 'distress':
                models_scores.append(0)

            # Add Springate assessment (0 for distress, 2 for safe)
            if springate_results.get('assessment') == 'safe':
                models_scores.append(2)
            elif springate_results.get('assessment') == 'distress':
                models_scores.append(0)

            # Add Zmijewski assessment (0 for distress, 2 for safe)
            if zmijewski_results.get('assessment') == 'safe':
                models_scores.append(2)
            elif zmijewski_results.get('assessment') == 'distress':
                models_scores.append(0)

            # Calculate average score if we have valid assessments
            overall_assessment = None
            overall_description = None
            overall_color = None

            if models_scores:
                average_score = sum(models_scores) / len(models_scores)

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

            return {
                'models': {
                    'altman_z_score': z_score_results,
                    'springate': springate_results,
                    'zmijewski': zmijewski_results
                },
                'overall_assessment': overall_assessment,
                'overall_description': overall_description,
                'overall_color': overall_color
            }

        except Exception as e:
            logger.error(f"Error calculating comprehensive risk assessment: {e}")
            return {
                'models': {
                    'altman_z_score': self._get_empty_z_score_result(),
                    'springate': self._get_empty_springate_result(),
                    'zmijewski': self._get_empty_zmijewski_result()
                },
                'overall_assessment': None,
                'overall_description': None,
                'overall_color': None
            }

    # Private helper methods

    def _calculate_standard_z_score(self, income: pd.Series, balance: pd.Series, model_variant: str) -> Dict[str, Any]:
        """Calculate standard Z-Score (original or modified version)"""
        try:
            # Calculate components
            # A = Working Capital / Total Assets
            working_capital = self._get_value(balance, 'Total Current Assets') - self._get_value(balance,
                                                                                                 'Total Current Liabilities')
            total_assets = self._get_value(balance, 'Total Assets')
            component_a = working_capital / total_assets if total_assets != 0 else 0

            # B = Retained Earnings / Total Assets
            retained_earnings = self._get_value(balance, 'Retained Earnings')
            component_b = retained_earnings / total_assets if total_assets != 0 else 0

            # C = EBIT / Total Assets
            ebit = self._get_value(income, 'Operating Income')
            component_c = ebit / total_assets if total_assets != 0 else 0

            # D = Market Value of Equity / Total Liabilities
            if model_variant == "original":
                # For original model, try to use market cap if available
                market_cap = self._get_value(balance, 'Market Cap')
                if market_cap is None or market_cap == 0:
                    # If market cap not available, use book value of equity
                    market_cap = self._get_value(balance, 'Total Stockholder Equity')
            else:
                # For modified model, use book value of equity
                market_cap = self._get_value(balance, 'Total Stockholder Equity')

            total_liabilities = self._get_value(balance, 'Total Liabilities')
            component_d = market_cap / total_liabilities if total_liabilities != 0 else 0

            # E = Sales / Total Assets
            sales = self._get_value(income, 'Total Revenue')
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
                'model': f"Altman Z-Score ({model_variant})"
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
            equity = self._get_value(balance, 'Total Stockholder Equity')
            total_assets = self._get_value(balance, 'Total Assets')
            component_a = equity / total_assets if total_assets != 0 else 0

            # B = EBIT / Total Assets
            ebit = self._get_value(income, 'Operating Income')
            component_b = ebit / total_assets if total_assets != 0 else 0

            # C = Retained Earnings / Total Assets
            retained_earnings = self._get_value(balance, 'Retained Earnings')
            component_c = retained_earnings / total_assets if total_assets != 0 else 0

            # D = Net Interest Income / Total Assets
            interest_income = self._get_value(income, 'Interest Income') or 0
            interest_expense = abs(self._get_value(income, 'Interest Expense') or 0)
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
                'model': "Altman Z-Score (financial)"
            }
        except Exception as e:
            logger.error(f"Error in financial Z-Score calculation: {e}")
            return self._get_empty_z_score_result()

    def _get_value(self, series: pd.Series, key: str) -> float:
        """
        Safely get a value from a series, handling different key names and formats

        Args:
            series: The pandas Series to search
            key: The key to find

        Returns:
            Value if found, 0 otherwise
        """
        # Try direct key lookup
        if key in series:
            return series[key]

        # Try case-insensitive lookup
        for idx in series.index:
            if isinstance(idx, str) and idx.lower() == key.lower():
                return series[idx]

        # Try partial match
        for idx in series.index:
            if isinstance(idx, str) and key.lower() in idx.lower():
                return series[idx]

        return 0

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