import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('filters')


class Filter(ABC):
    """Abstract base class for all stock filters"""

    @abstractmethod
    def apply(self, company_data: Dict[str, Any]) -> bool:
        """
        Apply the filter to company data

        Args:
            company_data: Dictionary containing company information

        Returns:
            True if company passes the filter, False otherwise
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a human-readable description of the filter

        Returns:
            String description of the filter
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert filter to dictionary representation

        Returns:
            Dictionary representation of the filter
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Filter':
        """
        Create filter from dictionary representation

        Args:
            data: Dictionary representation of the filter

        Returns:
            Filter instance
        """
        pass


class ValueFilter(Filter):
    """Filter that compares a numeric metric to a threshold value"""

    def __init__(self, metric: str, operator: str, value: Union[float, int],
                 metric_display_name: Optional[str] = None):
        """
        Initialize the value filter

        Args:
            metric: Name of the metric to filter on
            operator: Comparison operator (>, <, >=, <=, =, !=)
            value: Threshold value for comparison
            metric_display_name: Human-readable name for the metric (for display purposes)
        """
        self.metric = metric
        self.operator = operator
        self.value = value
        self.metric_display_name = metric_display_name or metric

        # Define operator functions
        self.operator_funcs = {
            ">": lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "<": lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
            "=": lambda x, y: x == y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y
        }

        # Operator display formats
        self.operator_formats = {
            ">": "greater than",
            ">=": "greater than or equal to",
            "<": "less than",
            "<=": "less than or equal to",
            "=": "equal to",
            "==": "equal to",
            "!=": "not equal to"
        }

        if operator not in self.operator_funcs:
            raise ValueError(f"Unsupported operator: {operator}")

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        # Get the metric value
        metric_value = self._get_metric_value(company_data, self.metric)

        # If metric not found, filter fails
        if metric_value is None:
            return False

        # Apply the comparison operator
        return self.operator_funcs[self.operator](metric_value, self.value)

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        operator_text = self.operator_formats.get(self.operator, self.operator)
        return f"{self.metric_display_name} {operator_text} {self.value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "value",
            "metric": self.metric,
            "operator": self.operator,
            "value": self.value,
            "metric_display_name": self.metric_display_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValueFilter':
        """Create filter from dictionary representation"""
        return cls(
            metric=data["metric"],
            operator=data["operator"],
            value=data["value"],
            metric_display_name=data.get("metric_display_name")
        )

    def _get_metric_value(self, company_data: Dict[str, Any], metric: str) -> Optional[Union[float, int]]:
        """Extract metric value from company data"""
        # Check direct company properties
        if metric in company_data:
            return company_data[metric]

        # Check growth rates
        if metric in ['revenue_growth', 'earnings_growth', 'ebitda_growth'] and metric in company_data:
            return company_data[metric]

        # Check ratios
        ratios = company_data.get('ratios', {})
        for category in ratios:
            if metric in ratios[category]:
                return ratios[category][metric]

        # Try to navigate nested dictionaries using dot notation
        if '.' in metric:
            parts = metric.split('.')
            current = company_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None


class GrowthFilter(Filter):
    """Filter that checks for positive or negative growth over a specified period"""

    def __init__(self, metric: str, periods: int = 1, threshold: Optional[float] = None,
                 direction: str = "positive", metric_display_name: Optional[str] = None):
        """
        Initialize the growth filter

        Args:
            metric: Name of the metric to filter on growth
            periods: Number of periods to check growth over
            threshold: Growth threshold (optional)
            direction: "positive" or "negative" to filter for growth or decline
            metric_display_name: Human-readable name for the metric (for display purposes)
        """
        self.metric = metric
        self.periods = periods
        self.threshold = threshold
        self.direction = direction.lower()
        self.metric_display_name = metric_display_name or metric

        if self.direction not in ["positive", "negative"]:
            raise ValueError("Direction must be either 'positive' or 'negative'")

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        # Check if company has historical data
        if 'historical_data' not in company_data or not company_data['historical_data']:
            return False

        historical_data = company_data['historical_data']

        # Make sure we have enough periods
        if len(historical_data) <= self.periods:
            return False

        # Calculate growth
        try:
            current_value = self._get_metric_value(historical_data[0], self.metric)
            past_value = self._get_metric_value(historical_data[self.periods], self.metric)

            if current_value is None or past_value is None or past_value == 0:
                return False

            growth_rate = (current_value / past_value) - 1

            # Apply direction filter
            if self.direction == "positive":
                if growth_rate <= 0:
                    return False

                # Apply threshold if specified
                if self.threshold is not None and growth_rate < self.threshold:
                    return False
            else:  # negative
                if growth_rate >= 0:
                    return False

                # Apply threshold if specified (for negative growth, threshold is typically negative)
                if self.threshold is not None and growth_rate > self.threshold:
                    return False

            return True

        except Exception:
            return False

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        if self.threshold is not None:
            if self.direction == "positive":
                return f"{self.metric_display_name} growth of at least {self.threshold:.1%} over {self.periods} periods"
            else:
                return f"{self.metric_display_name} decline of at least {abs(self.threshold):.1%} over {self.periods} periods"
        else:
            if self.direction == "positive":
                return f"{self.metric_display_name} showing positive growth over {self.periods} periods"
            else:
                return f"{self.metric_display_name} showing negative growth over {self.periods} periods"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "growth",
            "metric": self.metric,
            "periods": self.periods,
            "threshold": self.threshold,
            "direction": self.direction,
            "metric_display_name": self.metric_display_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrowthFilter':
        """Create filter from dictionary representation"""
        return cls(
            metric=data["metric"],
            periods=data.get("periods", 1),
            threshold=data.get("threshold"),
            direction=data.get("direction", "positive"),
            metric_display_name=data.get("metric_display_name")
        )

    def _get_metric_value(self, period_data: Dict[str, Any], metric: str) -> Optional[Union[float, int]]:
        """Extract metric value from period data"""
        if metric in period_data:
            return period_data[metric]

        # Try to navigate nested dictionaries using dot notation
        if '.' in metric:
            parts = metric.split('.')
            current = period_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            if isinstance(current, (int, float)):
                return current

        return None


class SectorFilter(Filter):
    """Filter that selects companies from specific sectors or industries"""

    def __init__(self, sectors: Optional[List[str]] = None,
                 industries: Optional[List[str]] = None,
                 exclude: bool = False):
        """
        Initialize the sector filter

        Args:
            sectors: List of sectors to include (optional)
            industries: List of industries to include (optional)
            exclude: If True, filter will exclude the specified sectors/industries
        """
        self.sectors = sectors or []
        self.industries = industries or []
        self.exclude = exclude

        if not self.sectors and not self.industries:
            raise ValueError("At least one sector or industry must be specified")

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        company_sector = company_data.get('sector')
        company_industry = company_data.get('industry')

        # Check sector match
        sector_match = False
        if self.sectors and company_sector:
            sector_match = any(company_sector.lower() == s.lower() for s in self.sectors)

        # Check industry match
        industry_match = False
        if self.industries and company_industry:
            industry_match = any(company_industry.lower() == i.lower() for i in self.industries)

        # Determine if company matches criteria
        matches = sector_match or industry_match

        # Apply exclude logic if needed
        if self.exclude:
            return not matches
        else:
            return matches

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        parts = []

        if self.sectors:
            if len(self.sectors) == 1:
                parts.append(f"sector is {self.sectors[0]}")
            else:
                sectors_str = ", ".join(self.sectors)
                parts.append(f"sector is one of: {sectors_str}")

        if self.industries:
            if len(self.industries) == 1:
                parts.append(f"industry is {self.industries[0]}")
            else:
                industries_str = ", ".join(self.industries)
                parts.append(f"industry is one of: {industries_str}")

        description = " or ".join(parts)

        if self.exclude:
            return f"Exclude companies where {description}"
        else:
            return f"Include only companies where {description}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "sector",
            "sectors": self.sectors,
            "industries": self.industries,
            "exclude": self.exclude
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectorFilter':
        """Create filter from dictionary representation"""
        return cls(
            sectors=data.get("sectors"),
            industries=data.get("industries"),
            exclude=data.get("exclude", False)
        )


class MarketCapFilter(Filter):
    """Filter that selects companies within specific market capitalization ranges"""

    # Market cap categories (in USD)
    CATEGORIES = {
        "mega": (200e9, None),  # > $200 billion
        "large": (10e9, 200e9),  # $10 billion - $200 billion
        "mid": (2e9, 10e9),  # $2 billion - $10 billion
        "small": (300e6, 2e9),  # $300 million - $2 billion
        "micro": (50e6, 300e6),  # $50 million - $300 million
        "nano": (0, 50e6)  # < $50 million
    }

    def __init__(self, categories: Optional[List[str]] = None,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None):
        """
        Initialize the market cap filter

        Args:
            categories: List of market cap categories to include
            min_value: Minimum market cap value (optional)
            max_value: Maximum market cap value (optional)
        """
        self.categories = categories or []
        self.min_value = min_value
        self.max_value = max_value

        # Validate the categories
        for category in self.categories:
            if category.lower() not in self.CATEGORIES:
                raise ValueError(f"Invalid market cap category: {category}")

        if not self.categories and self.min_value is None and self.max_value is None:
            raise ValueError("Either categories or min/max values must be specified")

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        market_cap = company_data.get('market_cap')

        # If market cap is missing, filter fails
        if market_cap is None:
            return False

        # Check category matches if categories specified
        if self.categories:
            for category in self.categories:
                cat_lower = category.lower()
                if cat_lower in self.CATEGORIES:
                    min_cap, max_cap = self.CATEGORIES[cat_lower]

                    if min_cap is not None and market_cap < min_cap:
                        continue

                    if max_cap is not None and market_cap > max_cap:
                        continue

                    return True

            # If we get here, no categories matched
            return False

        # Check custom range if specified
        if self.min_value is not None and market_cap < self.min_value:
            return False

        if self.max_value is not None and market_cap > self.max_value:
            return False

        # If we get here, market cap is within specified range
        return True

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        if self.categories:
            if len(self.categories) == 1:
                return f"Market cap is {self.categories[0]} cap"
            else:
                categories_str = ", ".join(self.categories)
                return f"Market cap is one of: {categories_str} cap"
        else:
            # Custom range
            if self.min_value is not None and self.max_value is not None:
                return f"Market cap between ${self._format_currency(self.min_value)} and ${self._format_currency(self.max_value)}"
            elif self.min_value is not None:
                return f"Market cap greater than ${self._format_currency(self.min_value)}"
            else:  # self.max_value is not None
                return f"Market cap less than ${self._format_currency(self.max_value)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "market_cap",
            "categories": self.categories,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketCapFilter':
        """Create filter from dictionary representation"""
        return cls(
            categories=data.get("categories"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value")
        )

    def _format_currency(self, value: float) -> str:
        """Format currency value with appropriate suffix"""
        if value >= 1e12:
            return f"{value / 1e12:.1f}T"
        elif value >= 1e9:
            return f"{value / 1e9:.1f}B"
        elif value >= 1e6:
            return f"{value / 1e6:.1f}M"
        else:
            return f"{value:.0f}"


class DividendFilter(Filter):
    """Filter that selects companies based on dividend characteristics"""

    def __init__(self, min_yield: Optional[float] = None,
                 max_yield: Optional[float] = None,
                 pays_dividend: bool = True,
                 min_growth_rate: Optional[float] = None,
                 years_of_growth: Optional[int] = None):
        """
        Initialize the dividend filter

        Args:
            min_yield: Minimum dividend yield (optional)
            max_yield: Maximum dividend yield (optional)
            pays_dividend: Whether company must pay dividends
            min_growth_rate: Minimum dividend growth rate (optional)
            years_of_growth: Minimum consecutive years of dividend growth (optional)
        """
        self.min_yield = min_yield
        self.max_yield = max_yield
        self.pays_dividend = pays_dividend
        self.min_growth_rate = min_growth_rate
        self.years_of_growth = years_of_growth

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        # Check if company pays dividends
        dividend_yield = company_data.get('dividend_yield')

        if self.pays_dividend:
            if dividend_yield is None or dividend_yield <= 0:
                return False
        else:
            # If we're looking for companies that don't pay dividends
            if dividend_yield is not None and dividend_yield > 0:
                return False
            # Skip the rest of the checks as they only apply to dividend-paying companies
            return True

        # Check yield range if specified
        if self.min_yield is not None and dividend_yield < self.min_yield:
            return False

        if self.max_yield is not None and dividend_yield > self.max_yield:
            return False

        # Check dividend growth if required
        if self.min_growth_rate is not None or self.years_of_growth is not None:
            # Need historical dividend data
            if 'dividend_history' not in company_data or not company_data['dividend_history']:
                return False

            dividend_history = company_data['dividend_history']

            # Check if we have enough years of data
            if self.years_of_growth is not None and len(dividend_history) < self.years_of_growth:
                return False

            # Check dividend growth rate
            if self.min_growth_rate is not None:
                years_to_check = self.years_of_growth or min(5, len(dividend_history))

                if years_to_check > 1:
                    # Calculate average growth rate
                    growth_rates = []
                    for i in range(min(years_to_check, len(dividend_history) - 1)):
                        current = dividend_history[i]
                        previous = dividend_history[i + 1]

                        if previous == 0:
                            continue

                        growth_rate = (current / previous) - 1
                        growth_rates.append(growth_rate)

                    if not growth_rates:
                        return False

                    avg_growth_rate = sum(growth_rates) / len(growth_rates)

                    if avg_growth_rate < self.min_growth_rate:
                        return False

            # Check consecutive years of growth
            if self.years_of_growth is not None:
                consecutive_growth_years = 0
                for i in range(len(dividend_history) - 1):
                    if dividend_history[i] > dividend_history[i + 1]:
                        consecutive_growth_years += 1
                    else:
                        break

                if consecutive_growth_years < self.years_of_growth:
                    return False

        # If we get here, company meets all dividend criteria
        return True

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        if not self.pays_dividend:
            return "Companies that do not pay dividends"

        parts = ["Dividend-paying companies"]

        if self.min_yield is not None and self.max_yield is not None:
            parts.append(f"yield between {self.min_yield:.1%} and {self.max_yield:.1%}")
        elif self.min_yield is not None:
            parts.append(f"yield at least {self.min_yield:.1%}")
        elif self.max_yield is not None:
            parts.append(f"yield at most {self.max_yield:.1%}")

        if self.min_growth_rate is not None:
            parts.append(f"dividend growth rate of at least {self.min_growth_rate:.1%}")

        if self.years_of_growth is not None:
            parts.append(f"at least {self.years_of_growth} consecutive years of dividend growth")

        return " with ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "dividend",
            "min_yield": self.min_yield,
            "max_yield": self.max_yield,
            "pays_dividend": self.pays_dividend,
            "min_growth_rate": self.min_growth_rate,
            "years_of_growth": self.years_of_growth
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DividendFilter':
        """Create filter from dictionary representation"""
        return cls(
            min_yield=data.get("min_yield"),
            max_yield=data.get("max_yield"),
            pays_dividend=data.get("pays_dividend", True),
            min_growth_rate=data.get("min_growth_rate"),
            years_of_growth=data.get("years_of_growth")
        )


class FilterFactory:
    """Factory class for creating filters from dictionaries"""

    @staticmethod
    def create_filter(filter_dict: Dict[str, Any]) -> Filter:
        """
        Create a filter from a dictionary representation

        Args:
            filter_dict: Dictionary representation of a filter

        Returns:
            Filter instance
        """
        filter_type = filter_dict.get("type", "").lower()

        if filter_type == "value":
            return ValueFilter.from_dict(filter_dict)
        elif filter_type == "range":
            return RangeFilter.from_dict(filter_dict)
        elif filter_type == "growth":
            return GrowthFilter.from_dict(filter_dict)
        elif filter_type == "sector":
            return SectorFilter.from_dict(filter_dict)
        elif filter_type == "market_cap":
            return MarketCapFilter.from_dict(filter_dict)
        elif filter_type == "dividend":
            return DividendFilter.from_dict(filter_dict)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    @staticmethod
    def create_filters_from_list(filter_list: List[Dict[str, Any]]) -> List[Filter]:
        """
        Create multiple filters from a list of dictionary representations

        Args:
            filter_list: List of filter dictionaries

        Returns:
            List of Filter instances
        """
        return [FilterFactory.create_filter(f) for f in filter_list]

    @staticmethod
    def create_filter_from_criteria(metric: str, operator: str,
                                    value: Union[float, int, str, List]) -> Filter:
        """
        Create a filter from simple criteria

        Args:
            metric: Name of the metric to filter on
            operator: Comparison operator
            value: Threshold value for comparison

        Returns:
            Filter instance
        """
        # Handle range operators
        if operator == "between" and isinstance(value, list) and len(value) == 2:
            return RangeFilter(metric=metric, min_value=value[0], max_value=value[1])

        # Handle basic comparison operators
        if operator in [">", ">=", "<", "<=", "=", "==", "!="]:
            return ValueFilter(metric=metric, operator=operator, value=value)

        # Handle growth operators
        if operator == "increase":
            return GrowthFilter(metric=metric, direction="positive",
                                threshold=value if isinstance(value, (int, float)) else None)

        if operator == "decrease":
            return GrowthFilter(metric=metric, direction="negative",
                                threshold=value if isinstance(value, (int, float)) else None)

        # Handle list operators
        if operator == "in" and isinstance(value, list):
            # For sector filters
            if metric.lower() in ["sector", "industry"]:
                return SectorFilter(sectors=[value] if metric.lower() == "sector" else None,
                                    industries=[value] if metric.lower() == "industry" else None)

        raise ValueError(f"Unsupported operator: {operator} for metric: {metric}")


class FilterSet:
    """Class for managing and applying multiple filters"""

    def __init__(self, filters: Optional[List[Filter]] = None, combine_with_and: bool = True):
        """
        Initialize the filter set

        Args:
            filters: List of filters to apply
            combine_with_and: If True, all filters must pass; if False, any filter can pass
        """
        self.filters = filters or []
        self.combine_with_and = combine_with_and

    def add_filter(self, filter_obj: Filter) -> None:
        """Add a filter to the set"""
        self.filters.append(filter_obj)

    def add_filters(self, filters: List[Filter]) -> None:
        """Add multiple filters to the set"""
        self.filters.extend(filters)

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """
        Apply all filters to company data

        Args:
            company_data: Dictionary containing company information

        Returns:
            True if company passes filters, False otherwise
        """
        if not self.filters:
            return True

        if self.combine_with_and:
            # All filters must pass
            for filter_obj in self.filters:
                if not filter_obj.apply(company_data):
                    return False
            return True
        else:
            # Any filter can pass
            for filter_obj in self.filters:
                if filter_obj.apply(company_data):
                    return True
            return False

    def get_description(self) -> str:
        """Get a human-readable description of the filter set"""
        if not self.filters:
            return "No filters applied"

        filter_descriptions = [f.get_description() for f in self.filters]

        if self.combine_with_and:
            return " AND ".join(filter_descriptions)
        else:
            return " OR ".join(filter_descriptions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter set to dictionary representation"""
        return {
            "filters": [f.to_dict() for f in self.filters],
            "combine_with_and": self.combine_with_and
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterSet':
        """Create filter set from dictionary representation"""
        filter_dicts = data.get("filters", [])
        filters = [FilterFactory.create_filter(f) for f in filter_dicts]

        return cls(
            filters=filters,
            combine_with_and=data.get("combine_with_and", True)
        )
        if isinstance(current, (int, float)):
            return current

        return None


class RangeFilter(Filter):
    """Filter that checks if a numeric metric is within a specified range"""

    def __init__(self, metric: str, min_value: Optional[Union[float, int]] = None,
                 max_value: Optional[Union[float, int]] = None,
                 inclusive: bool = True, metric_display_name: Optional[str] = None):
        """
        Initialize the range filter

        Args:
            metric: Name of the metric to filter on
            min_value: Minimum value for the range (optional)
            max_value: Maximum value for the range (optional)
            inclusive: Whether range boundaries are inclusive
            metric_display_name: Human-readable name for the metric (for display purposes)
        """
        self.metric = metric
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
        self.metric_display_name = metric_display_name or metric

        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

    def apply(self, company_data: Dict[str, Any]) -> bool:
        """Apply filter to company data"""
        # Get the metric value
        metric_value = self._get_metric_value(company_data, self.metric)

        # If metric not found, filter fails
        if metric_value is None:
            return False

        # Check min boundary if specified
        if self.min_value is not None:
            if self.inclusive:
                if metric_value < self.min_value:
                    return False
            else:
                if metric_value <= self.min_value:
                    return False

        # Check max boundary if specified
        if self.max_value is not None:
            if self.inclusive:
                if metric_value > self.max_value:
                    return False
            else:
                if metric_value >= self.max_value:
                    return False

        # If we get here, the value is within range
        return True

    def get_description(self) -> str:
        """Get a human-readable description of the filter"""
        if self.min_value is not None and self.max_value is not None:
            if self.inclusive:
                return f"{self.metric_display_name} between {self.min_value} and {self.max_value} (inclusive)"
            else:
                return f"{self.metric_display_name} between {self.min_value} and {self.max_value} (exclusive)"
        elif self.min_value is not None:
            if self.inclusive:
                return f"{self.metric_display_name} greater than or equal to {self.min_value}"
            else:
                return f"{self.metric_display_name} greater than {self.min_value}"
        else:  # self.max_value is not None
            if self.inclusive:
                return f"{self.metric_display_name} less than or equal to {self.max_value}"
            else:
                return f"{self.metric_display_name} less than {self.max_value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation"""
        return {
            "type": "range",
            "metric": self.metric,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive,
            "metric_display_name": self.metric_display_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RangeFilter':
        """Create filter from dictionary representation"""
        return cls(
            metric=data["metric"],
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            inclusive=data.get("inclusive", True),
            metric_display_name=data.get("metric_display_name")
        )

    def _get_metric_value(self, company_data: Dict[str, Any], metric: str) -> Optional[Union[float, int]]:
        """Extract metric value from company data"""
        # Check direct company properties
        if metric in company_data:
            return company_data[metric]

        # Check growth rates
        if metric in ['revenue_growth', 'earnings_growth', 'ebitda_growth'] and metric in company_data:
            return company_data[metric]

        # Check ratios
        ratios = company_data.get('ratios', {})
        for category in ratios:
            if metric in ratios[category]:
                return ratios[category][metric]

        # Try to navigate nested dictionaries using dot notation
        if '.' in metric:
            parts = metric.split('.')
            current = company_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            if isinstance(current, (int, float)):
                return current

        return None