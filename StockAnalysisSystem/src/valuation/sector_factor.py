import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SECTOR_METRICS, SECTOR_DCF_PARAMETERS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_factory')


class SectorAnalyzer(ABC):
    """
    Abstract base class for sector-specific analyzers.
    Each sector should implement its own analyzer with specific metrics and methods.
    """

    def __init__(self, sector_name: str):
        """
        Initialize sector analyzer with sector-specific parameters

        Args:
            sector_name: The name of the sector
        """
        self.sector_name = sector_name
        self.sector_config = SECTOR_METRICS.get(sector_name, {})
        self.dcf_parameters = SECTOR_DCF_PARAMETERS.get(sector_name, {})

        # Default parameters if not found in config
        if not self.dcf_parameters:
            self.dcf_parameters = {
                "forecast_years": 5,
                "terminal_growth_rate": 0.02,
                "default_discount_rate": 0.10
            }

        logger.info(f"Initialized {sector_name} analyzer")

    @abstractmethod
    def analyze_company(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a company using sector-specific metrics and methods

        Args:
            financial_data: Dictionary containing company financial information

        Returns:
            Dictionary with analysis results
        """
        pass

    @abstractmethod
    def calculate_valuation(self, financial_data: Dict[str, Any], valuation_method: str = None) -> Dict[str, Any]:
        """
        Calculate company valuation using sector-specific methods

        Args:
            financial_data: Dictionary containing company financial information
            valuation_method: Specific valuation method to use (if None, use the most appropriate)

        Returns:
            Dictionary with valuation results
        """
        pass

    @abstractmethod
    def benchmark_company(self, company_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics with sector benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Dictionary with benchmarking results
        """
        pass

    def get_key_ratios(self) -> List[str]:
        """
        Get the list of key ratios for this sector

        Returns:
            List of ratio names that are most relevant for this sector
        """
        return self.sector_config.get('key_ratios', [])

    def get_preferred_valuation_methods(self) -> List[str]:
        """
        Get the preferred valuation methods for this sector

        Returns:
            List of valuation method names
        """
        return self.sector_config.get('valuation_methods', ['DCF'])

    def get_sector_benchmarks(self) -> Dict[str, float]:
        """
        Get benchmark values for key metrics in this sector

        Returns:
            Dictionary of metric names and their benchmark values
        """
        return self.sector_config.get('benchmarks', {})

    def calculate_ratio_score(self, company_metrics: Dict[str, float]) -> float:
        """
        Calculate an overall score based on how company's ratios compare to benchmarks

        Args:
            company_metrics: Dictionary with company financial metrics

        Returns:
            Score between 0 and 100
        """
        benchmarks = self.get_sector_benchmarks()
        key_ratios = self.get_key_ratios()

        if not benchmarks or not key_ratios:
            return 50.0  # Default middle score

        score = 0.0
        count = 0

        for ratio in key_ratios:
            if ratio in company_metrics and ratio in benchmarks:
                ratio_value = company_metrics[ratio]
                benchmark_value = benchmarks[ratio]

                # Determine if higher is better (default) or lower is better
                higher_is_better = True

                # Specific cases where lower is better
                if ratio in ['Debt/Equity', 'Debt/EBITDA', 'Churn Rate', 'NPL Ratio']:
                    higher_is_better = False

                # Calculate ratio score (0-100)
                if higher_is_better:
                    if benchmark_value > 0:
                        ratio_score = min(100, (ratio_value / benchmark_value) * 100)
                    else:
                        ratio_score = 100 if ratio_value > 0 else 0
                else:
                    if ratio_value > 0:
                        ratio_score = min(100, (benchmark_value / ratio_value) * 100)
                    else:
                        ratio_score = 100  # No debt/ratio is good for ratios where lower is better

                score += ratio_score
                count += 1

        if count > 0:
            return score / count
        else:
            return 50.0  # Default score


class SectorFactory:
    """
    Factory class for creating sector-specific analyzers
    """
    _analyzers = {}  # Class variable to store registered analyzers

    @classmethod
    def register_analyzer(cls, sector_name: str, analyzer_class):
        """
        Register a sector analyzer class

        Args:
            sector_name: Name of the sector
            analyzer_class: Class for the sector analyzer
        """
        cls._analyzers[sector_name] = analyzer_class
        logger.info(f"Registered analyzer for {sector_name}")

    @classmethod
    def get_analyzer(cls, sector_name: str) -> Optional[SectorAnalyzer]:
        """
        Get the appropriate analyzer for a given sector

        Args:
            sector_name: Name of the sector

        Returns:
            Instance of the appropriate SectorAnalyzer or None if not found
        """
        analyzer_class = cls._analyzers.get(sector_name)
        if analyzer_class:
            return analyzer_class(sector_name)

        logger.warning(f"No analyzer found for sector: {sector_name}")
        return None