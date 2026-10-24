import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SCREENING_TEMPLATES
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('screener_engine')


class FilterOperator:
    """Enumeration of filter operators for screening criteria"""
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "="
    NOT_EQUAL = "!="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    INCREASE = "increase"
    DECREASE = "decrease"


class ScreenerEngine:
    """
    Engine for screening stocks based on financial and market criteria.
    Provides functionality to filter companies using various financial ratios,
    valuation metrics, growth rates, and other criteria.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the stock screener engine

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self.ratio_analyzer = FinancialRatioAnalyzer()
        self.screening_templates = SCREENING_TEMPLATES
        logger.info("ScreenerEngine initialized")

    def screen_stocks(self, filters: Dict[str, Dict[str, Union[str, float, int]]],
                      universe: Optional[List[str]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Screen stocks based on specified filters

        Args:
            filters: Dictionary of filters {metric: {operator: <, >, =, etc.>, value: <threshold>}}
            universe: List of stock tickers to screen (if None, uses default universe)
            limit: Maximum number of results to return

        Returns:
            List of stocks that match the filter criteria with their data
        """
        if universe is None:
            # If no universe specified, use default from data_loader
            universe = self._get_default_universe()

        if not universe:
            logger.warning("No stocks in universe to screen")
            return []

        logger.info(f"Screening {len(universe)} stocks with {len(filters)} filters")

        # Track stocks that pass all filters
        passing_stocks = []

        # Process each stock in the universe
        for ticker in universe:
            try:
                # Get company data
                company_data = self._get_company_data(ticker)

                # Skip stocks with insufficient data
                if not company_data:
                    continue

                # Apply filters
                if self._apply_filters(company_data, filters):
                    passing_stocks.append(company_data)

                    # Break early if we've reached the limit
                    if len(passing_stocks) >= limit:
                        break

            except Exception as e:
                logger.warning(f"Error screening {ticker}: {e}")
                continue

        logger.info(f"Found {len(passing_stocks)} stocks matching criteria")
        return passing_stocks

    def get_template(self, template_name: str) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Get a predefined screening template

        Args:
            template_name: Name of the template

        Returns:
            Dictionary of filters for the template
        """
        if template_name in self.screening_templates:
            return self.screening_templates[template_name]
        else:
            logger.warning(f"Template {template_name} not found, returning empty filter set")
            return {}

    def get_all_templates(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        """
        Get all available screening templates

        Returns:
            Dictionary of all templates
        """
        return self.screening_templates

    def create_custom_template(self, name: str, filters: Dict[str, Dict[str, Union[str, float, int]]]) -> bool:
        """
        Create a custom screening template

        Args:
            name: Name for the new template
            filters: Dictionary of filters for the template

        Returns:
            True if template was created successfully, False otherwise
        """
        if name in self.screening_templates:
            logger.warning(f"Template {name} already exists, use update_template instead")
            return False

        self.screening_templates[name] = filters
        logger.info(f"Created new template: {name}")
        return True

    def update_template(self, name: str, filters: Dict[str, Dict[str, Union[str, float, int]]]) -> bool:
        """
        Update an existing screening template

        Args:
            name: Name of the template to update
            filters: New filters for the template

        Returns:
            True if template was updated successfully, False otherwise
        """
        if name not in self.screening_templates:
            logger.warning(f"Template {name} does not exist, use create_custom_template instead")
            return False

        self.screening_templates[name] = filters
        logger.info(f"Updated template: {name}")
        return True

    def delete_template(self, name: str) -> bool:
        """
        Delete a screening template

        Args:
            name: Name of the template to delete

        Returns:
            True if template was deleted successfully, False otherwise
        """
        if name not in self.screening_templates:
            logger.warning(f"Template {name} does not exist")
            return False

        del self.screening_templates[name]
        logger.info(f"Deleted template: {name}")
        return True

    def screen_by_template(self, template_name: str, universe: Optional[List[str]] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Screen stocks using a predefined template

        Args:
            template_name: Name of the template to use
            universe: List of stock tickers to screen (if None, uses default universe)
            limit: Maximum number of results to return

        Returns:
            List of stocks that match the template criteria with their data
        """
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"Could not find template {template_name}")
            return []

        return self.screen_stocks(template, universe, limit)

    def create_sector_screener(self, sector: str) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Create a screening template tailored for a specific sector

        Args:
            sector: Market sector name

        Returns:
            Dictionary of filters appropriate for the sector
        """
        # Get sector-specific benchmarks
        sector_benchmarks = self.ratio_analyzer.get_sector_benchmarks(sector)

        # Create filters based on sector benchmarks
        filters = {}

        # Valuation metrics
        if 'valuation' in sector_benchmarks:
            for metric, value in sector_benchmarks['valuation'].items():
                # For valuation metrics, we usually want values below the benchmark
                filters[metric] = {"operator": "<", "value": value * 1.2}  # 20% buffer

        # Profitability metrics
        if 'profitability' in sector_benchmarks:
            for metric, value in sector_benchmarks['profitability'].items():
                # For profitability metrics, we usually want values above the benchmark
                filters[metric] = {"operator": ">", "value": value * 0.8}  # 20% buffer

        # Liquidity metrics
        if 'liquidity' in sector_benchmarks:
            for metric, value in sector_benchmarks['liquidity'].items():
                # For liquidity metrics, we usually want values above the benchmark
                filters[metric] = {"operator": ">", "value": value * 0.8}  # 20% buffer

        # Leverage metrics
        if 'leverage' in sector_benchmarks:
            for metric, value in sector_benchmarks['leverage'].items():
                if metric in ['debt_to_equity', 'debt_to_ebitda']:
                    # For debt metrics, we usually want values below the benchmark
                    filters[metric] = {"operator": "<", "value": value * 1.2}  # 20% buffer
                else:
                    # For coverage ratios, we want values above the benchmark
                    filters[metric] = {"operator": ">", "value": value * 0.8}  # 20% buffer

        # Efficiency metrics
        if 'efficiency' in sector_benchmarks:
            for metric, value in sector_benchmarks['efficiency'].items():
                if metric in ['inventory_turnover', 'asset_turnover']:
                    # For turnover metrics, we usually want values above the benchmark
                    filters[metric] = {"operator": ">", "value": value * 0.8}  # 20% buffer
                elif metric in ['days_inventory_outstanding', 'days_sales_outstanding']:
                    # For days outstanding, we usually want values below the benchmark
                    filters[metric] = {"operator": "<", "value": value * 1.2}  # 20% buffer

        # Growth metrics
        if 'growth' in sector_benchmarks:
            for metric, value in sector_benchmarks['growth'].items():
                # For growth metrics, we usually want values above the benchmark
                filters[metric] = {"operator": ">", "value": value * 0.7}  # 30% buffer

        logger.info(f"Created sector-specific screener for {sector} with {len(filters)} criteria")
        return filters

    def _get_default_universe(self) -> List[str]:
        """Get default universe of stocks to screen"""
        try:
            # Try to get S&P 500 components as default universe
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()

            logger.info(f"Loaded default universe with {len(tickers)} stocks")
            return tickers
        except Exception as e:
            logger.error(f"Error loading default universe: {e}")
            # Return a small set of major tickers as fallback
            fallback_tickers = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK.B', 'JNJ',
                'UNH', 'XOM', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
                'PFE', 'KO', 'AVGO', 'PEP', 'WMT', 'LLY', 'COST', 'TMO', 'MRK', 'CSCO'
            ]
            logger.info(f"Using fallback universe with {len(fallback_tickers)} stocks")
            return fallback_tickers

    def _get_company_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company data for screening"""
        try:
            # Get company info
            company_info = self.data_loader.get_company_info(ticker)

            # Get financial statements
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income', 'annual')
            balance_sheet = self.data_loader.get_financial_statements(ticker, 'balance', 'annual')
            cash_flow = self.data_loader.get_financial_statements(ticker, 'cash', 'annual')

            # Get historical prices
            historical_prices = self.data_loader.get_historical_prices(ticker)

            # Calculate ratios
            financial_data = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'market_data': {
                    'market_cap': company_info.get('market_cap'),
                    'share_price': company_info.get('current_price', company_info.get('regular_market_price')),
                    'shares_outstanding': company_info.get('shares_outstanding'),
                    'dividend_yield': company_info.get('dividend_yield'),
                    'beta': company_info.get('beta')
                }
            }

            ratios = self.ratio_analyzer.calculate_ratios(financial_data)

            # Build comprehensive company dataset
            company_data = {
                'ticker': ticker,
                'name': company_info.get('name', ''),
                'sector': company_info.get('sector', ''),
                'industry': company_info.get('industry', ''),
                'country': company_info.get('country', ''),
                'exchange': company_info.get('exchange', ''),
                'market_cap': company_info.get('market_cap'),
                'beta': company_info.get('beta'),
                'price': company_info.get('current_price', company_info.get('regular_market_price')),
                'ratios': ratios
            }

            # Calculate growth rates if we have at least 2 years of data
            if income_stmt is not None and income_stmt.shape[1] >= 2:
                growth_rates = self._calculate_growth_rates(income_stmt)
                company_data.update(growth_rates)

            return company_data
        except Exception as e:
            logger.warning(f"Error getting data for {ticker}: {e}")
            return {}

    def _calculate_growth_rates(self, income_stmt: pd.DataFrame) -> Dict[str, float]:
        """Calculate key growth rates from income statement"""
        growth_rates = {}

        try:
            # Get the two most recent years
            latest = income_stmt.iloc[:, 0]
            previous = income_stmt.iloc[:, 1]

            # Calculate revenue growth
            if 'Total Revenue' in latest.index and 'Total Revenue' in previous.index:
                latest_revenue = latest['Total Revenue']
                previous_revenue = previous['Total Revenue']

                if previous_revenue > 0:
                    revenue_growth = (latest_revenue / previous_revenue) - 1
                    growth_rates['revenue_growth'] = revenue_growth

            # Calculate earnings growth
            if 'Net Income' in latest.index and 'Net Income' in previous.index:
                latest_income = latest['Net Income']
                previous_income = previous['Net Income']

                if previous_income > 0:
                    earnings_growth = (latest_income / previous_income) - 1
                    growth_rates['earnings_growth'] = earnings_growth

            # Calculate EBITDA growth
            if 'EBITDA' in latest.index and 'EBITDA' in previous.index:
                latest_ebitda = latest['EBITDA']
                previous_ebitda = previous['EBITDA']

                if previous_ebitda > 0:
                    ebitda_growth = (latest_ebitda / previous_ebitda) - 1
                    growth_rates['ebitda_growth'] = ebitda_growth
        except Exception as e:
            logger.warning(f"Error calculating growth rates: {e}")

        return growth_rates

    def _apply_filters(self, company_data: Dict[str, Any],
                       filters: Dict[str, Dict[str, Union[str, float, int]]]) -> bool:
        """
        Apply filters to company data to determine if it passes the screening criteria

        Args:
            company_data: Dictionary with company data
            filters: Dictionary of filters to apply

        Returns:
            True if company passes all filters, False otherwise
        """
        for metric, filter_info in filters.items():
            operator = filter_info.get("operator")
            filter_value = filter_info.get("value")

            # Skip if missing operator or value
            if not operator or filter_value is None:
                continue

            # Get company value for the metric
            company_value = self._get_metric_value(company_data, metric)

            # Skip if company doesn't have this metric
            if company_value is None:
                return False

            # Apply the appropriate comparison based on operator
            if not self._apply_comparison(company_value, operator, filter_value):
                return False

        # If we made it here, company passed all filters
        return True

    def _get_metric_value(self, company_data: Dict[str, Any], metric: str) -> Optional[Union[float, int, str]]:
        """
        Get the value of a specific metric from company data

        Args:
            company_data: Dictionary with company data
            metric: Name of the metric to retrieve

        Returns:
            Value of the metric if found, None otherwise
        """
        # Check direct company properties first
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

        # Check for alternate names
        metric_alternatives = {
            "P/E": ["pe_ratio", "price_to_earnings"],
            "P/S": ["ps_ratio", "price_to_sales"],
            "P/B": ["pb_ratio", "price_to_book"],
            "EV/EBITDA": ["ev_ebitda", "enterprise_value_to_ebitda"],
            "ROE": ["roe", "return_on_equity"],
            "ROA": ["roa", "return_on_assets"],
            "Operating Margin": ["operating_margin"],
            "Net Margin": ["net_margin", "profit_margin"],
            "Current Ratio": ["current_ratio"],
            "Debt/Equity": ["debt_to_equity"],
            "Dividend Yield": ["dividend_yield"]
        }

        if metric in metric_alternatives:
            for alt_name in metric_alternatives[metric]:
                # Check direct properties
                if alt_name in company_data:
                    return company_data[alt_name]

                # Check ratios
                for category in ratios:
                    if alt_name in ratios[category]:
                        return ratios[category][alt_name]

        # Metric not found
        return None

    def _apply_comparison(self, value: Union[float, int, str],
                          operator: str,
                          threshold: Union[float, int, str, List]) -> bool:
        """
        Apply a comparison operation between a value and a threshold

        Args:
            value: The value to compare
            operator: The comparison operator
            threshold: The threshold value(s) to compare against

        Returns:
            True if the comparison passes, False otherwise
        """
        try:
            if operator == FilterOperator.GREATER_THAN:
                return value > threshold
            elif operator == FilterOperator.GREATER_EQUAL:
                return value >= threshold
            elif operator == FilterOperator.LESS_THAN:
                return value < threshold
            elif operator == FilterOperator.LESS_EQUAL:
                return value <= threshold
            elif operator == FilterOperator.EQUAL:
                return value == threshold
            elif operator == FilterOperator.NOT_EQUAL:
                return value != threshold
            elif operator == FilterOperator.BETWEEN:
                if isinstance(threshold, list) and len(threshold) == 2:
                    return threshold[0] <= value <= threshold[1]
                return False
            elif operator == FilterOperator.IN:
                if isinstance(threshold, list):
                    return value in threshold
                return False
            elif operator == FilterOperator.NOT_IN:
                if isinstance(threshold, list):
                    return value not in threshold
                return False
            elif operator == FilterOperator.CONTAINS:
                if isinstance(value, str) and isinstance(threshold, str):
                    return threshold in value
                return False
            elif operator == FilterOperator.NOT_CONTAINS:
                if isinstance(value, str) and isinstance(threshold, str):
                    return threshold not in value
                return False
            elif operator == FilterOperator.INCREASE:
                return value > 0
            elif operator == FilterOperator.DECREASE:
                return value < 0
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            logger.warning(f"Error applying comparison: {e}")
            return False

    def combine_filters(self, *filter_sets: Dict[str, Dict[str, Union[str, float, int]]]) -> Dict[
        str, Dict[str, Union[str, float, int]]]:
        """
        Combine multiple filter sets into one

        Args:
            *filter_sets: Variable number of filter sets to combine

        Returns:
            Combined filter set
        """
        combined_filters = {}

        for filters in filter_sets:
            for metric, filter_info in filters.items():
                combined_filters[metric] = filter_info

        return combined_filters

    def rank_results(self, results: List[Dict[str, Any]],
                     metrics: List[str],
                     ascending: List[bool] = None) -> List[Dict[str, Any]]:
        """
        Rank screening results by specified metrics

        Args:
            results: List of company data dictionaries
            metrics: List of metrics to rank by (in priority order)
            ascending: List of booleans indicating sort direction for each metric

        Returns:
            Sorted list of company data dictionaries
        """
        if not results or not metrics:
            return results

        # Set default ascending values if not provided
        if ascending is None:
            # By default, sort profitability/growth metrics descending and valuation metrics ascending
            ascending = []
            for metric in metrics:
                if metric.lower() in ['pe_ratio', 'p/e', 'ps_ratio', 'p/s', 'pb_ratio', 'p/b', 'ev_ebitda',
                                      'ev/ebitda']:
                    ascending.append(True)  # Lower is better for valuation ratios
                else:
                    ascending.append(False)  # Higher is better for other metrics

        # Make sure ascending list matches metrics list in length
        while len(ascending) < len(metrics):
            ascending.append(False)

        # Convert to DataFrame for easier sorting
        df = pd.DataFrame(results)

        # Create columns for sorting based on nested metrics
        for i, metric in enumerate(metrics):
            column_name = f"sort_{i}"

            # Extract values for the metric
            df[column_name] = df.apply(lambda row: self._get_metric_value(row, metric), axis=1)

            # Replace NaN with appropriate extreme values based on sort direction
            if ascending[i]:
                df[column_name] = df[column_name].fillna(float('inf'))  # For ascending sorts, NaN should be last
            else:
                df[column_name] = df[column_name].fillna(float('-inf'))  # For descending sorts, NaN should be last

        # Create sort columns list
        sort_columns = [f"sort_{i}" for i in range(len(metrics))]

        # Sort the DataFrame
        df_sorted = df.sort_values(sort_columns, ascending=ascending)

        # Drop the sorting columns
        df_sorted = df_sorted.drop(columns=sort_columns)

        # Convert back to list of dictionaries
        return df_sorted.to_dict('records')