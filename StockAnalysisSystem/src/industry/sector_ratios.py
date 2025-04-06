import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SECTOR_DATA_DIR
from industry.benchmarks import IndustryBenchmarks
from industry.sector_mapping import SectorMapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_ratios')


class SectorRatioAnalyzer:
    """
    Class for analyzing sector-specific financial ratios.
    Provides methods for calculating specialized ratios based on sector,
    comparing companies within sectors, and evaluating performance
    relative to sector standards.
    """

    def __init__(self,
                 benchmarks: Optional[IndustryBenchmarks] = None,
                 sector_mapper: Optional[SectorMapper] = None,
                 data_dir: str = SECTOR_DATA_DIR):
        """
        Initialize sector ratio analyzer

        Args:
            benchmarks: IndustryBenchmarks instance
            sector_mapper: SectorMapper instance
            data_dir: Directory for sector data files
        """
        self.data_dir = Path(data_dir)
        self.benchmarks = benchmarks or IndustryBenchmarks()
        self.sector_mapper = sector_mapper or SectorMapper()

        # Load specialized ratio definitions
        self.sector_ratio_definitions = self._load_sector_ratio_definitions()
        logger.info("Initialized SectorRatioAnalyzer")

    def get_sector_specific_ratios(self, ticker: str, sector: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate sector-specific ratios for a company

        Args:
            ticker: Company ticker symbol
            sector: Company sector
            financial_data: Dictionary with financial statements and market data

        Returns:
            Dictionary with sector-specific ratios and analysis
        """
        try:
            # Normalize sector name
            normalized_sector = self.sector_mapper._normalize_sector_name(sector)

            # Get ratio definitions for this sector
            ratio_definitions = self.sector_ratio_definitions.get(normalized_sector, {})

            if not ratio_definitions:
                logger.warning(f"No specialized ratio definitions found for sector: {normalized_sector}")
                return {}

            # Calculate sector-specific ratios based on definitions
            sector_ratios = {}

            for ratio_name, ratio_info in ratio_definitions.items():
                # Skip if we're missing data for this ratio
                if not self._check_data_available(ratio_info['required_data'], financial_data):
                    continue

                # Calculate the ratio
                ratio_value = self._calculate_ratio(ratio_info, financial_data)

                if ratio_value is not None:
                    sector_ratios[ratio_name] = {
                        'value': ratio_value,
                        'description': ratio_info.get('description', ''),
                        'benchmark': ratio_info.get('benchmark'),
                        'is_good_high': ratio_info.get('is_good_high', True)
                    }

            # Get benchmarks for comparison
            benchmarks = self.benchmarks.get_sector_benchmarks(normalized_sector)
            sector_benchmarks = benchmarks.get('sector_specific', {})

            # Add benchmark comparison
            for ratio_name, ratio_data in sector_ratios.items():
                if ratio_name in sector_benchmarks:
                    ratio_data['benchmark'] = sector_benchmarks[ratio_name]

                # Add assessment based on benchmark
                if 'benchmark' in ratio_data and ratio_data['benchmark'] is not None:
                    ratio_data['assessment'] = self._assess_ratio(
                        ratio_data['value'],
                        ratio_data['benchmark'],
                        ratio_data.get('is_good_high', True)
                    )

            return {
                'sector': normalized_sector,
                'ratios': sector_ratios
            }
        except Exception as e:
            logger.error(f"Error calculating sector-specific ratios for {ticker}: {e}")
            return {'error': str(e)}

    def compare_with_sector_averages(self, ticker: str, sector: str, ratios: Dict[str, Dict[str, Any]]) -> Dict[
        str, Any]:
        """
        Compare company ratios with sector averages

        Args:
            ticker: Company ticker symbol
            sector: Company sector
            ratios: Dictionary with calculated ratios for the company

        Returns:
            Dictionary with comparison analysis
        """
        try:
            # Normalize sector name
            normalized_sector = self.sector_mapper._normalize_sector_name(sector)

            # Get benchmarks for the sector
            benchmarks = self.benchmarks.get_sector_benchmarks(normalized_sector)

            if not benchmarks:
                logger.warning(f"No benchmarks found for sector: {normalized_sector}")
                return {'error': 'No benchmarks available for sector'}

            # Initialize result dictionary
            comparison = {
                'sector': normalized_sector,
                'company': ticker,
                'categories': {}
            }

            # Compare ratios by category
            for category, category_ratios in ratios.items():
                # Skip non-ratio categories
                if category in ['sector', 'error']:
                    continue

                category_benchmarks = benchmarks.get(category, {})
                category_comparison = {}

                for ratio_name, ratio_data in category_ratios.items():
                    if 'value' not in ratio_data:
                        continue

                    ratio_value = ratio_data['value']
                    benchmark_value = category_benchmarks.get(ratio_name)

                    # Skip if no benchmark available
                    if benchmark_value is None:
                        continue

                    # Calculate percentage difference
                    if benchmark_value != 0:
                        pct_diff = (ratio_value / benchmark_value - 1) * 100
                    else:
                        pct_diff = np.nan

                    is_good_high = ratio_data.get('is_good_high', True)

                    category_comparison[ratio_name] = {
                        'company_value': ratio_value,
                        'sector_value': benchmark_value,
                        'pct_diff': pct_diff,
                        'assessment': self._assess_difference(pct_diff, is_good_high)
                    }

                if category_comparison:
                    comparison['categories'][category] = category_comparison

            # Add overall summary
            comparison['summary'] = self._create_comparison_summary(comparison['categories'])

            return comparison
        except Exception as e:
            logger.error(f"Error comparing {ticker} with sector averages: {e}")
            return {'error': str(e)}

    def get_sector_performance_metrics(self, sector: str) -> Dict[str, Any]:
        """
        Get key performance metrics for a sector

        Args:
            sector: Sector name

        Returns:
            Dictionary with sector performance metrics
        """
        try:
            # Normalize sector name
            normalized_sector = self.sector_mapper._normalize_sector_name(sector)

            # Get performance data
            sector_performance = self.benchmarks.get_sector_performance()

            if sector_performance.empty:
                logger.warning(f"No performance data available for sectors")
                return {'error': 'No sector performance data available'}

            # Filter for the specific sector
            sector_data = sector_performance[sector_performance['Sector'] == normalized_sector]

            if sector_data.empty:
                logger.warning(f"No performance data found for sector: {normalized_sector}")
                return {'error': f'No performance data for {normalized_sector}'}

            # Extract performance metrics
            metrics = sector_data.iloc[0].to_dict()

            # Remove sector name from metrics
            if 'Sector' in metrics:
                del metrics['Sector']

            return {
                'sector': normalized_sector,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics for sector {sector}: {e}")
            return {'error': str(e)}

    # Private helper methods

    def _load_sector_ratio_definitions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load sector-specific ratio definitions"""
        # Default definitions
        definitions = {
            "Technology": {
                "rnd_to_revenue": {
                    "description": "R&D Expenses as a percentage of Revenue",
                    "required_data": ["income_statement"],
                    "formula": "R&D Expenses / Revenue",
                    "benchmark": 0.15,
                    "is_good_high": True
                },
                "software_amortization_ratio": {
                    "description": "Software Amortization as a percentage of Revenue",
                    "required_data": ["income_statement", "cash_flow"],
                    "formula": "Amortization / Revenue",
                    "benchmark": 0.05,
                    "is_good_high": False
                },
                "recurring_revenue_pct": {
                    "description": "Recurring Revenue as a percentage of Total Revenue",
                    "required_data": ["income_statement"],
                    "formula": "Subscription Revenue / Revenue",
                    "benchmark": 0.60,
                    "is_good_high": True
                }
            },
            "Healthcare": {
                "rnd_to_revenue": {
                    "description": "R&D Expenses as a percentage of Revenue",
                    "required_data": ["income_statement"],
                    "formula": "R&D Expenses / Revenue",
                    "benchmark": 0.12,
                    "is_good_high": True
                },
                "pipeline_value_ratio": {
                    "description": "Pipeline Value to Market Cap ratio",
                    "required_data": ["company_profile", "market_data"],
                    "formula": "Pipeline Value / Market Cap",
                    "benchmark": 1.5,
                    "is_good_high": True
                }
            },
            "Financials": {
                "net_interest_margin": {
                    "description": "Net Interest Income as a percentage of Average Earning Assets",
                    "required_data": ["income_statement", "balance_sheet"],
                    "formula": "Net Interest Income / Average Earning Assets",
                    "benchmark": 0.035,
                    "is_good_high": True
                },
                "efficiency_ratio": {
                    "description": "Non-Interest Expense as a percentage of Revenue",
                    "required_data": ["income_statement"],
                    "formula": "Non-Interest Expense / Revenue",
                    "benchmark": 0.60,
                    "is_good_high": False
                },
                "loan_to_deposit": {
                    "description": "Total Loans as a percentage of Total Deposits",
                    "required_data": ["balance_sheet"],
                    "formula": "Total Loans / Total Deposits",
                    "benchmark": 0.80,
                    "is_good_high": None  # Optimal range, not simply higher or lower
                },
                "non_performing_loans": {
                    "description": "Non-Performing Loans as a percentage of Total Loans",
                    "required_data": ["balance_sheet"],
                    "formula": "Non-Performing Loans / Total Loans",
                    "benchmark": 0.01,
                    "is_good_high": False
                }
            },
            "Energy": {
                "reserve_replacement_ratio": {
                    "description": "New Reserves Added to Reserves Produced",
                    "required_data": ["company_profile"],
                    "formula": "New Reserves / Production",
                    "benchmark": 1.0,
                    "is_good_high": True
                },
                "production_cost_per_boe": {
                    "description": "Production Cost per Barrel of Oil Equivalent",
                    "required_data": ["income_statement", "company_profile"],
                    "formula": "Production Costs / Production Volume",
                    "benchmark": 15.0,  # dollars per barrel
                    "is_good_high": False
                },
                "reserve_life_index": {
                    "description": "Reserves to Production Ratio (in years)",
                    "required_data": ["company_profile"],
                    "formula": "Total Reserves / Annual Production",
                    "benchmark": 10.0,  # years
                    "is_good_high": True
                }
            },
            "Consumer Discretionary": {
                "same_store_sales_growth": {
                    "description": "Year-over-Year Sales Growth for Existing Stores",
                    "required_data": ["company_profile"],
                    "formula": "Current SSS / Previous SSS - 1",
                    "benchmark": 0.03,  # 3% growth
                    "is_good_high": True
                },
                "inventory_to_sales": {
                    "description": "Inventory as a percentage of Sales",
                    "required_data": ["income_statement", "balance_sheet"],
                    "formula": "Inventory / Revenue",
                    "benchmark": 0.15,
                    "is_good_high": False
                },
                "online_sales_pct": {
                    "description": "Online Sales as a percentage of Total Sales",
                    "required_data": ["income_statement", "company_profile"],
                    "formula": "Online Sales / Revenue",
                    "benchmark": 0.20,
                    "is_good_high": True
                }
            },
            "Real Estate": {
                "funds_from_operations": {
                    "description": "Funds From Operations per Share",
                    "required_data": ["income_statement", "cash_flow", "market_data"],
                    "formula": "(Net Income + Depreciation + Amortization - Gains on Sale) / Shares Outstanding",
                    "benchmark": None,  # Varies by property type
                    "is_good_high": True
                },
                "occupancy_rate": {
                    "description": "Percentage of Properties Occupied",
                    "required_data": ["company_profile"],
                    "formula": "Occupied Space / Total Space",
                    "benchmark": 0.95,
                    "is_good_high": True
                },
                "net_asset_value_premium": {
                    "description": "Market Price Premium to Net Asset Value",
                    "required_data": ["balance_sheet", "market_data"],
                    "formula": "Market Cap / Net Asset Value - 1",
                    "benchmark": 0.05,
                    "is_good_high": None  # Can be either depending on context
                }
            }
        }

        try:
            # Check if we have a definitions file
            definitions_file = self.data_dir / "sector_ratio_definitions.json"

            if definitions_file.exists():
                # Load definitions
                with open(definitions_file, 'r') as f:
                    file_definitions = json.load(f)

                # Merge with default definitions
                for sector, sector_defs in file_definitions.items():
                    if sector in definitions:
                        # Update existing sector definitions
                        definitions[sector].update(sector_defs)
                    else:
                        # Add new sector
                        definitions[sector] = sector_defs

                logger.info(f"Loaded ratio definitions for {len(file_definitions)} sectors")
        except Exception as e:
            logger.error(f"Error loading sector ratio definitions: {e}")

        return definitions

    def _check_data_available(self, required_data: List[str], financial_data: Dict[str, Any]) -> bool:
        """Check if all required data is available for a ratio"""
        return all(data_type in financial_data and not (
                isinstance(financial_data[data_type], pd.DataFrame) and financial_data[data_type].empty
        ) for data_type in required_data)

    def _calculate_ratio(self, ratio_info: Dict[str, Any], financial_data: Dict[str, Any]) -> Optional[float]:
        """Calculate a ratio based on its definition and available financial data"""
        # This is a simplified implementation
        # In a real implementation, we would parse the formula and calculate it
        # dynamically based on the data

        # For now, just return a random value for demo purposes
        # In production, this would be replaced with actual calculation logic
        return np.random.uniform(0.5, 1.5) * ratio_info.get('benchmark', 1.0)

    def _assess_ratio(self, value: float, benchmark: float, is_good_high: bool) -> str:
        """Assess a ratio based on its value compared to benchmark"""
        if np.isnan(value) or np.isnan(benchmark):
            return "neutral"

        # Calculate percentage difference
        if benchmark != 0:
            pct_diff = (value / benchmark - 1) * 100
        else:
            return "neutral"

        # Assess based on direction
        if is_good_high:
            if pct_diff > 20:
                return "excellent"
            elif pct_diff > 5:
                return "good"
            elif pct_diff >= -5:
                return "neutral"
            elif pct_diff >= -20:
                return "warning"
            else:
                return "poor"
        else:
            if pct_diff < -20:
                return "excellent"
            elif pct_diff < -5:
                return "good"
            elif pct_diff <= 5:
                return "neutral"
            elif pct_diff <= 20:
                return "warning"
            else:
                return "poor"

    def _assess_difference(self, pct_diff: float, is_good_high: bool) -> str:
        """Assess the difference between company ratio and benchmark"""
        if np.isnan(pct_diff):
            return "neutral"

        # Assess based on direction
        if is_good_high:
            if pct_diff > 20:
                return "excellent"
            elif pct_diff > 5:
                return "good"
            elif pct_diff >= -5:
                return "neutral"
            elif pct_diff >= -20:
                return "warning"
            else:
                return "poor"
        else:
            if pct_diff < -20:
                return "excellent"
            elif pct_diff < -5:
                return "good"
            elif pct_diff <= 5:
                return "neutral"
            elif pct_diff <= 20:
                return "warning"
            else:
                return "poor"

    def _create_comparison_summary(self, categories: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create a summary of the comparison results"""
        # Count assessments across all categories
        assessment_counts = {
            "excellent": 0,
            "good": 0,
            "neutral": 0,
            "warning": 0,
            "poor": 0
        }

        total_ratios = 0

        for category, ratios in categories.items():
            for ratio_name, ratio_data in ratios.items():
                if 'assessment' in ratio_data:
                    assessment = ratio_data['assessment']
                    if assessment in assessment_counts:
                        assessment_counts[assessment] += 1
                        total_ratios += 1

        # Calculate percentages
        assessment_percentages = {}
        for assessment, count in assessment_counts.items():
            if total_ratios > 0:
                assessment_percentages[assessment] = (count / total_ratios) * 100
            else:
                assessment_percentages[assessment] = 0

        # Determine overall assessment
        if total_ratios > 0:
            positive_pct = assessment_percentages['excellent'] + assessment_percentages['good']
            negative_pct = assessment_percentages['warning'] + assessment_percentages['poor']

            if positive_pct > 60:
                overall = "Strong performer relative to sector"
            elif positive_pct > 40:
                overall = "Above average performer in sector"
            elif positive_pct > negative_pct:
                overall = "Average performer in sector"
            elif positive_pct == negative_pct:
                overall = "Mixed performance relative to sector"
            elif negative_pct > 60:
                overall = "Underperformer relative to sector"
            else:
                overall = "Below average performer in sector"
        else:
            overall = "Insufficient data for comparison"

        return {
            "assessment_counts": assessment_counts,
            "assessment_percentages": assessment_percentages,
            "total_ratios_compared": total_ratios,
            "overall_assessment": overall
        }