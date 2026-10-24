import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SECTOR_METRICS, SECTOR_MAPPING
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('industry_benchmarks')


class IndustryBenchmarks:
    """
    Class for managing industry benchmarks and sector-specific metrics.
    Provides methods to retrieve, calculate and compare company metrics against industry standards.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the industry benchmarks manager

        Args:
            data_loader: Optional DataLoader instance for fetching market data
        """
        self.data_loader = data_loader if data_loader else DataLoader()

        # Load sector metrics configuration
        self.sector_metrics = SECTOR_METRICS

        # Load industry mapping
        self.sector_mapping = SECTOR_MAPPING

        # Pre-load static benchmarks for faster access
        self._static_benchmarks = self._load_static_benchmarks()

        # Cache for dynamically calculated benchmarks
        self._dynamic_benchmarks_cache = {}
        self._cache_expiry = datetime.now() - timedelta(days=1)  # Force initial refresh

        logger.info("Industry benchmarks system initialized")

    def get_benchmarks(self, sector: str, metric_type: Optional[str] = None,
                       refresh: bool = False) -> Dict[str, Any]:
        """
        Get benchmark metrics for a specific industry sector

        Args:
            sector: The market sector (e.g., "Technology", "Healthcare")
            metric_type: Optional type of metrics to retrieve (e.g., "valuation", "profitability")
            refresh: Force refresh of dynamic benchmarks

        Returns:
            Dictionary containing benchmark metrics for the sector
        """
        # Normalize sector name to match our configuration
        normalized_sector = self._normalize_sector_name(sector)

        if not normalized_sector:
            logger.warning(f"Unrecognized sector: {sector}")
            return {}

        # Get static benchmarks
        static_benchmarks = self._static_benchmarks.get(normalized_sector, {})

        # Check if we need dynamic benchmarks
        current_time = datetime.now()
        cache_expired = (current_time - self._cache_expiry).days >= 1

        if refresh or cache_expired or normalized_sector not in self._dynamic_benchmarks_cache:
            # Refresh dynamic benchmarks
            self._refresh_dynamic_benchmarks()
            self._cache_expiry = current_time

        # Get dynamic benchmarks
        dynamic_benchmarks = self._dynamic_benchmarks_cache.get(normalized_sector, {})

        # Combine static and dynamic benchmarks
        combined_benchmarks = {**static_benchmarks, **dynamic_benchmarks}

        # Filter by metric type if specified
        if metric_type:
            filtered_benchmarks = {}
            for key, value in combined_benchmarks.items():
                if isinstance(value, dict) and metric_type in key.lower():
                    filtered_benchmarks[key] = value

            return filtered_benchmarks

        return combined_benchmarks

    def compare_with_benchmarks(self, company_metrics: Dict[str, Any],
                                sector: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare company metrics with sector benchmarks

        Args:
            company_metrics: Dictionary of company metrics
            sector: The market sector of the company

        Returns:
            Dictionary with comparison results including percentile rankings
        """
        # Get sector benchmarks
        benchmarks = self.get_benchmarks(sector)

        if not benchmarks:
            return {"error": f"No benchmarks available for sector: {sector}"}

        comparison_results = {}

        # Process each category of metrics
        for category, metrics in company_metrics.items():
            category_benchmarks = {}

            for metric_name, metric_value in metrics.items():
                # Skip if value is None
                if metric_value is None:
                    continue

                # Find the corresponding benchmark
                benchmark_found = False

                # Check if metric exists directly in benchmarks
                if metric_name in benchmarks:
                    benchmark_value = benchmarks[metric_name]
                    benchmark_found = True

                # Check if metric exists in a category in benchmarks
                if not benchmark_found:
                    for bench_category, bench_metrics in benchmarks.items():
                        if isinstance(bench_metrics, dict) and metric_name in bench_metrics:
                            benchmark_value = bench_metrics[metric_name]
                            benchmark_found = True
                            break

                if benchmark_found:
                    # Create comparison data
                    comparison = self._create_comparison(metric_name, metric_value, benchmark_value)
                    category_benchmarks[metric_name] = comparison

            if category_benchmarks:
                comparison_results[category] = category_benchmarks

        return comparison_results

    def get_sector_percentiles(self, sector: str, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get percentile distributions for specific metrics in a sector

        Args:
            sector: Market sector name
            metrics: List of metrics to get percentiles for

        Returns:
            Dictionary with percentile distributions for each metric
        """
        normalized_sector = self._normalize_sector_name(sector)

        if not normalized_sector:
            return {}

        # Try to get sector data from data_loader
        try:
            # Get a list of companies in this sector
            sector_companies = self._get_sector_companies(normalized_sector)

            if not sector_companies:
                logger.warning(f"No companies found for sector: {normalized_sector}")
                return {}

            # Collect metrics for all companies
            metrics_data = {}
            for metric in metrics:
                metrics_data[metric] = []

            for ticker in sector_companies:
                company_data = self._get_company_metrics(ticker)
                for metric in metrics:
                    if metric in company_data and company_data[metric] is not None:
                        metrics_data[metric].append(company_data[metric])

            # Calculate percentiles
            percentiles = {}
            percentile_values = [10, 25, 50, 75, 90]

            for metric, values in metrics_data.items():
                if values:
                    metric_percentiles = {}
                    for p in percentile_values:
                        try:
                            metric_percentiles[f"p{p}"] = np.percentile(values, p)
                        except Exception as e:
                            logger.error(f"Error calculating percentile {p} for {metric}: {e}")
                            metric_percentiles[f"p{p}"] = None

                    # Also include min, max, mean
                    metric_percentiles["min"] = min(values)
                    metric_percentiles["max"] = max(values)
                    metric_percentiles["mean"] = sum(values) / len(values)
                    metric_percentiles["count"] = len(values)

                    percentiles[metric] = metric_percentiles

            return percentiles

        except Exception as e:
            logger.error(f"Error getting sector percentiles for {sector}: {e}")
            return {}

    def get_metric_ranking(self, ticker: str, metric: str) -> Dict[str, Any]:
        """
        Get ranking of a company for a specific metric within its sector

        Args:
            ticker: Company ticker symbol
            metric: Metric name to rank

        Returns:
            Dictionary with ranking information
        """
        try:
            # Get company metrics
            company_metrics = self._get_company_metrics(ticker)

            if not company_metrics or metric not in company_metrics:
                return {"error": f"Metric {metric} not available for {ticker}"}

            company_value = company_metrics[metric]

            # Get company sector
            company_info = {}
            if self.data_loader:
                company_info = self.data_loader.get_company_info(ticker)

            if not company_info or 'sector' not in company_info:
                return {"error": f"Sector information not available for {ticker}"}

            sector = company_info['sector']
            normalized_sector = self._normalize_sector_name(sector)

            # Get sector companies
            sector_companies = self._get_sector_companies(normalized_sector)

            if not sector_companies:
                return {"error": f"No peer companies found in sector {sector}"}

            # Collect metric values for all sector companies
            metric_values = []
            for peer_ticker in sector_companies:
                peer_metrics = self._get_company_metrics(peer_ticker)
                if peer_metrics and metric in peer_metrics and peer_metrics[metric] is not None:
                    metric_values.append((peer_ticker, peer_metrics[metric]))

            if not metric_values:
                return {"error": f"No comparable data found for metric {metric} in sector {sector}"}

            # Sort by metric value (determine if higher is better or lower is better)
            higher_is_better = self._is_higher_better(metric)
            metric_values.sort(key=lambda x: x[1], reverse=higher_is_better)

            # Find company's position
            company_position = next((i for i, (ticker_val, _) in enumerate(metric_values)
                                     if ticker_val == ticker), -1)

            if company_position == -1:
                return {"error": f"Company {ticker} not found in sorted metrics list"}

            # Calculate percentile
            percentile = (len(metric_values) - company_position) / len(metric_values) * 100 if higher_is_better else \
                company_position / len(metric_values) * 100

            return {
                "ticker": ticker,
                "metric": metric,
                "value": company_value,
                "sector": sector,
                "rank": company_position + 1,
                "total_companies": len(metric_values),
                "percentile": percentile,
                "top_companies": metric_values[:5] if higher_is_better else metric_values[-5:],
                "bottom_companies": metric_values[-5:] if higher_is_better else metric_values[:5]
            }

        except Exception as e:
            logger.error(f"Error getting metric ranking for {ticker}, {metric}: {e}")
            return {"error": f"Failed to calculate ranking: {str(e)}"}

    def get_sector_matrix(self, sector: str, x_metric: str, y_metric: str) -> Dict[str, Any]:
        """
        Create a positioning matrix for companies in a sector based on two metrics

        Args:
            sector: Market sector name
            x_metric: Metric for X-axis
            y_metric: Metric for Y-axis

        Returns:
            Dictionary with matrix data
        """
        normalized_sector = self._normalize_sector_name(sector)

        if not normalized_sector:
            return {"error": f"Unrecognized sector: {sector}"}

        try:
            # Get sector companies
            sector_companies = self._get_sector_companies(normalized_sector)

            if not sector_companies:
                return {"error": f"No companies found for sector: {normalized_sector}"}

            # Collect metrics for all companies
            matrix_data = []

            for ticker in sector_companies:
                company_metrics = self._get_company_metrics(ticker)
                company_info = self.data_loader.get_company_info(ticker) if self.data_loader else {}

                if company_metrics and x_metric in company_metrics and y_metric in company_metrics:
                    x_value = company_metrics[x_metric]
                    y_value = company_metrics[y_metric]

                    if x_value is not None and y_value is not None:
                        market_cap = company_info.get('market_cap', None)

                        matrix_data.append({
                            "ticker": ticker,
                            "name": company_info.get('name', ticker),
                            "x_value": x_value,
                            "y_value": y_value,
                            "market_cap": market_cap
                        })

            if not matrix_data:
                return {"error": f"No companies with both {x_metric} and {y_metric} metrics in {sector}"}

            # Calculate quadrant boundaries (medians)
            x_values = [item["x_value"] for item in matrix_data]
            y_values = [item["y_value"] for item in matrix_data]

            x_median = np.median(x_values)
            y_median = np.median(y_values)

            # Assign quadrants
            for item in matrix_data:
                if item["x_value"] >= x_median and item["y_value"] >= y_median:
                    item["quadrant"] = 1  # Top right
                elif item["x_value"] < x_median and item["y_value"] >= y_median:
                    item["quadrant"] = 2  # Top left
                elif item["x_value"] < x_median and item["y_value"] < y_median:
                    item["quadrant"] = 3  # Bottom left
                else:
                    item["quadrant"] = 4  # Bottom right

            return {
                "sector": sector,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "x_median": x_median,
                "y_median": y_median,
                "companies": matrix_data
            }

        except Exception as e:
            logger.error(f"Error creating sector matrix for {sector}: {e}")
            return {"error": f"Failed to create sector matrix: {str(e)}"}

    def calculate_sector_average(self, sector: str, metrics: List[str]) -> Dict[str, float]:
        """
        Calculate average values for specified metrics across a sector

        Args:
            sector: Market sector name
            metrics: List of metrics to calculate averages for

        Returns:
            Dictionary with average values for each metric
        """
        normalized_sector = self._normalize_sector_name(sector)

        if not normalized_sector:
            return {}

        try:
            # Get sector companies
            sector_companies = self._get_sector_companies(normalized_sector)

            if not sector_companies:
                logger.warning(f"No companies found for sector: {normalized_sector}")
                return {}

            # Collect metrics for all companies
            metrics_data = {metric: [] for metric in metrics}

            for ticker in sector_companies:
                company_metrics = self._get_company_metrics(ticker)

                for metric in metrics:
                    if metric in company_metrics and company_metrics[metric] is not None:
                        metrics_data[metric].append(company_metrics[metric])

            # Calculate averages, filtering out extreme values
            averages = {}

            for metric, values in metrics_data.items():
                if values:
                    # Remove outliers (values beyond 3 standard deviations)
                    if len(values) > 5:  # Only apply outlier removal if we have enough data
                        mean = np.mean(values)
                        std = np.std(values)
                        filtered_values = [v for v in values if abs(v - mean) <= 3 * std]
                    else:
                        filtered_values = values

                    averages[metric] = sum(filtered_values) / len(filtered_values)
                else:
                    averages[metric] = None

            return averages

        except Exception as e:
            logger.error(f"Error calculating sector averages for {sector}: {e}")
            return {}

    def _load_static_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load static benchmarks from configuration"""
        static_benchmarks = {}

        for sector, metrics in self.sector_metrics.items():
            if 'benchmarks' in metrics:
                static_benchmarks[sector] = metrics['benchmarks']

        return static_benchmarks

    def _refresh_dynamic_benchmarks(self):
        """Refresh dynamically calculated benchmarks from current market data"""
        dynamic_benchmarks = {}

        try:
            # Define key metrics to calculate for each sector
            # These are metrics that change over time and should be calculated dynamically
            key_metrics = {
                "valuation": ["pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda"],
                "profitability": ["gross_margin", "operating_margin", "net_margin", "roe", "roa"],
                "leverage": ["debt_to_equity", "debt_to_ebitda"],
                "growth": ["revenue_growth", "eps_growth"]
            }

            # Process each sector
            for sector in self.sector_metrics.keys():
                # Flatten metrics list
                all_metrics = []
                for category in key_metrics.values():
                    all_metrics.extend(category)

                # Calculate averages
                sector_averages = self.calculate_sector_average(sector, all_metrics)

                # Organize by category
                categorized_benchmarks = {}
                for category, metrics in key_metrics.items():
                    category_benchmarks = {}
                    for metric in metrics:
                        if metric in sector_averages:
                            category_benchmarks[metric] = sector_averages[metric]

                    if category_benchmarks:
                        categorized_benchmarks[category] = category_benchmarks

                dynamic_benchmarks[sector] = categorized_benchmarks

        except Exception as e:
            logger.error(f"Error refreshing dynamic benchmarks: {e}")

        self._dynamic_benchmarks_cache = dynamic_benchmarks

    def _normalize_sector_name(self, sector: str) -> str:
        """Normalize sector name to match our configuration"""
        # Check exact match
        if sector in self.sector_metrics:
            return sector

        # Check case-insensitive match
        sector_lower = sector.lower()
        for config_sector in self.sector_metrics.keys():
            if config_sector.lower() == sector_lower:
                return config_sector

        # Check partial match
        for config_sector in self.sector_metrics.keys():
            if sector_lower in config_sector.lower() or config_sector.lower() in sector_lower:
                return config_sector

        # Check sector mapping
        for config_sector, industries in self.sector_mapping.items():
            for industry in industries:
                if sector_lower == industry.lower() or sector_lower in industry.lower():
                    return config_sector

        return ""

    def _get_sector_companies(self, sector: str) -> List[str]:
        """Get list of companies in a specific sector"""
        if not self.data_loader:
            return []

        try:
            # For a real implementation, you would use a proper API or database
            # to get all companies in a sector. Here we use a simplified approach.

            # Option 1: Use sector ETFs holdings
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
                "Industrials": "XLI",
                "Communication Services": "XLC"
            }

            etf_ticker = sector_etfs.get(sector)
            if etf_ticker:
                # This is a placeholder for actual implementation
                # In a real system, you would fetch ETF holdings
                # Here we just return some sample tickers
                if sector == "Technology":
                    return ["AAPL", "MSFT", "NVDA", "ADBE", "ORCL", "CRM", "IBM", "AMD", "INTC", "CSCO"]
                elif sector == "Healthcare":
                    return ["JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "TMO", "ABT", "MDT", "AMGN"]
                # Add more sectors as needed

            # Option 2: Use screening API to find companies by sector
            # This would be implemented in a real system

            return []

        except Exception as e:
            logger.error(f"Error getting companies for sector {sector}: {e}")
            return []

    def _get_company_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get key metrics for a company"""
        if not self.data_loader:
            return {}

        try:
            # Get financial ratios from data loader
            financial_ratios = self.data_loader.get_financial_ratios(ticker)

            # Flatten the ratios dictionary for easier access
            metrics = {}
            for category, ratios in financial_ratios.items():
                if isinstance(ratios, dict):
                    for ratio_name, ratio_value in ratios.items():
                        metrics[ratio_name] = ratio_value

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics for {ticker}: {e}")
            return {}

    def _create_comparison(self, metric_name: str, company_value: float,
                           benchmark_value: float) -> Dict[str, Any]:
        """Create comparison data between company metric and benchmark"""
        # Determine if higher values are better for this metric
        higher_is_better = self._is_higher_better(metric_name)

        # Calculate relative difference
        if benchmark_value != 0:
            relative_diff = (company_value - benchmark_value) / abs(benchmark_value)
        else:
            relative_diff = 0 if company_value == 0 else 1 if company_value > 0 else -1

        # Determine assessment
        if higher_is_better:
            if relative_diff > 0.2:
                assessment = "excellent"
            elif relative_diff > 0.05:
                assessment = "good"
            elif relative_diff > -0.05:
                assessment = "average"
            elif relative_diff > -0.2:
                assessment = "below_average"
            else:
                assessment = "poor"
        else:
            if relative_diff < -0.2:
                assessment = "excellent"
            elif relative_diff < -0.05:
                assessment = "good"
            elif relative_diff < 0.05:
                assessment = "average"
            elif relative_diff < 0.2:
                assessment = "below_average"
            else:
                assessment = "poor"

        # Define color based on assessment
        colors = {
            "excellent": "#74f174",  # Green
            "good": "#a5e1ad",  # Light green
            "average": "#fff59d",  # Yellow
            "below_average": "#ffb74d",  # Orange
            "poor": "#faa1a4"  # Red
        }

        return {
            "company_value": company_value,
            "benchmark_value": benchmark_value,
            "relative_diff": relative_diff,
            "absolute_diff": company_value - benchmark_value,
            "percent_diff": relative_diff * 100,
            "assessment": assessment,
            "color": colors[assessment],
            "higher_is_better": higher_is_better
        }

    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for this metric"""
        # Metrics where lower values are better
        lower_is_better_metrics = [
            "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda", "ev_revenue",
            "debt_to_equity", "debt_to_assets", "debt_to_ebitda",
            "days_inventory_outstanding", "days_sales_outstanding",
            "cash_conversion_cycle", "churn_rate", "expense_ratio",
            "beta", "volatility"
        ]

        metric_lower = metric_name.lower()

        # Check for exact matches
        if metric_lower in lower_is_better_metrics:
            return False

        # Check for partial matches
        for lower_metric in lower_is_better_metrics:
            if lower_metric in metric_lower:
                return False

        # Default: higher is better
        return True