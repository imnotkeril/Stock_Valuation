import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.screening.filters import (
    Filter, ValueFilter, RangeFilter, GrowthFilter, SectorFilter,
    MarketCapFilter, DividendFilter, FilterSet, FilterFactory
)
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategies')


class ScreeningStrategy:
    """Base class for stock screening strategies"""

    def __init__(self, name: str, description: str):
        """
        Initialize the screening strategy

        Args:
            name: Name of the strategy
            description: Description of the strategy
        """
        self.name = name
        self.description = description
        self.filter_set = FilterSet()

    def get_filters(self) -> FilterSet:
        """Get the filter set for this strategy"""
        return self.filter_set

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply the strategy to company data"""
        return self.filter_set.apply(company_data)

    def get_description(self) -> str:
        """Get a human-readable description of the strategy"""
        return self.description

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "filter_set": self.filter_set.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScreeningStrategy':
        """Create strategy from dictionary representation"""
        strategy = cls(
            name=data["name"],
            description=data["description"]
        )

        if "filter_set" in data:
            strategy.filter_set = FilterSet.from_dict(data["filter_set"])

        return strategy