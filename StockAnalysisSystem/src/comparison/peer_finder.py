import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SECTOR_MAPPING, PEER_ANALYSIS
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('peer_finder')


class PeerFinder:
    """
    Class for identifying and analyzing peer companies for comparison.
    Provides methods to find similar companies based on various criteria
    such as sector, size, business model, and financial metrics.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the peer finder

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()

        # Load configuration
        self.sector_mapping = SECTOR_MAPPING
        self.peer_config = PEER_ANALYSIS

        # Dictionary to cache peer data for faster access
        self._peer_cache = {}

        logger.info("Peer finder initialized")

    def find_peers(self, ticker: str, max_peers: Optional[int] = None,
                   similarity_metrics: Optional[List[str]] = None,
                   industry_only: bool = True,
                   size_range: float = 0.5,
                   manual_peers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find peers for a given company based on specified criteria

        Args:
            ticker: Ticker symbol of the company
            max_peers: Maximum number of peers to return
            similarity_metrics: List of metrics to use for similarity calculation
            industry_only: If True, only return peers from the same industry
            size_range: Size similarity range as a factor (e.g., 0.5 means 50% to 150% of the company's size)
            manual_peers: List of manually specified peer tickers to include

        Returns:
            List of peer companies with similarity scores
        """
        # Set defaults from configuration if not provided
        if max_peers is None:
            max_peers = self.peer_config.get('max_peers', 10)

        if similarity_metrics is None:
            similarity_metrics = self.peer_config.get('similarity_metrics',
                                                      ["Market Cap", "Revenue Growth", "Profit Margin", "Beta"])

        # Check cache first
        cache_key = f"{ticker}_{max_peers}_{'-'.join(similarity_metrics)}_{industry_only}_{size_range}"
        if manual_peers:
            cache_key += f"_{'_'.join(manual_peers)}"

        if cache_key in self._peer_cache:
            return self._peer_cache[cache_key]

        try:
            # Get company info
            company_info = self.data_loader.get_company_info(ticker)

            if not company_info:
                logger.error(f"Failed to get information for company {ticker}")
                return []

            # Get company financial data
            financial_ratios = self.data_loader.get_financial_ratios(ticker)
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income')
            balance_sheet = self.data_loader.get_financial_statements(ticker, 'balance')

            # Extract company sector and industry
            company_sector = company_info.get('sector', '')
            company_industry = company_info.get('industry', '')
            company_market_cap = company_info.get('market_cap', 0)

            if not company_sector:
                logger.warning(f"No sector information for {ticker}, peer finding may be less accurate")

            # Find potential peers
            potential_peers = []

            # First, try to get peers from data_loader if it has this capability
            api_peers = []
            try:
                api_peers = self.data_loader.get_industry_peers(ticker, max_peers=max_peers * 2)
            except Exception as e:
                logger.info(f"Failed to get peers from API for {ticker}: {e}")

            if api_peers:
                for peer_info in api_peers:
                    peer_ticker = peer_info.get('symbol', '')
                    if peer_ticker and peer_ticker != ticker:
                        potential_peers.append(peer_ticker)

            # If we don't have enough peers from API, try other methods
            if len(potential_peers) < max_peers:
                # Find more peers by sector/industry
                additional_peers = self._find_peers_by_sector(company_sector, company_industry)

                # Remove duplicates and the original company
                for peer in additional_peers:
                    if peer not in potential_peers and peer != ticker:
                        potential_peers.append(peer)

            # Add manual peers if specified
            if manual_peers:
                for peer in manual_peers:
                    if peer not in potential_peers and peer != ticker:
                        potential_peers.append(peer)

            # Calculate similarity scores for potential peers
            peer_similarities = []

            for peer_ticker in potential_peers:
                try:
                    # Get peer information
                    peer_info = self.data_loader.get_company_info(peer_ticker)
                    peer_ratios = self.data_loader.get_financial_ratios(peer_ticker)

                    if not peer_info:
                        continue

                    # Check industry match if required
                    if industry_only:
                        peer_industry = peer_info.get('industry', '')
                        peer_sector = peer_info.get('sector', '')

                        if not (peer_industry == company_industry or peer_sector == company_sector):
                            continue

                    # Check size similarity based on market cap
                    peer_market_cap = peer_info.get('market_cap', 0)

                    if company_market_cap > 0 and peer_market_cap > 0:
                        size_ratio = peer_market_cap / company_market_cap

                        if size_ratio < (1 - size_range) or size_ratio > (1 + size_range):
                            # Company size is too different
                            continue

                    # Calculate similarity score based on selected metrics
                    similarity_score = self._calculate_similarity(
                        ticker, peer_ticker, similarity_metrics,
                        company_info, peer_info,
                        financial_ratios, peer_ratios
                    )

                    # Store peer with similarity information
                    peer_data = {
                        "ticker": peer_ticker,
                        "name": peer_info.get('name', peer_ticker),
                        "industry": peer_info.get('industry', 'N/A'),
                        "sector": peer_info.get('sector', 'N/A'),
                        "market_cap": peer_market_cap,
                        "similarity_score": similarity_score
                    }

                    peer_similarities.append(peer_data)

                except Exception as e:
                    logger.warning(f"Error processing peer {peer_ticker}: {e}")
                    continue

            # Sort peers by similarity score (higher is better)
            peer_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Limit to maximum number of peers
            result = peer_similarities[:max_peers]

            # Cache the result
            self._peer_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error finding peers for {ticker}: {e}")
            return []

    def get_peer_comparison_metrics(self, ticker: str, peers: List[str],
                                    metric_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comparative metrics for a company and its peers

        Args:
            ticker: Ticker symbol of the main company
            peers: List of peer company tickers
            metric_categories: Categories of metrics to include

        Returns:
            Dictionary with comparative metric data
        """
        if metric_categories is None:
            # Get default metric categories from config
            metric_categories = list(self.peer_config.get('comparison_metrics', {}).keys())

        try:
            # Get metric definitions for requested categories
            metrics_to_fetch = []
            metrics_by_category = {}

            for category in metric_categories:
                category_metrics = self.peer_config.get('comparison_metrics', {}).get(category, [])
                metrics_by_category[category] = category_metrics
                metrics_to_fetch.extend(category_metrics)

            # Prepare result structure
            comparison_data = {
                "main_company": ticker,
                "peers": peers,
                "metrics_by_category": metrics_by_category,
                "company_data": {},
                "peer_data": {}
            }

            # Get main company data
            company_data = self._get_company_metrics(ticker, metrics_to_fetch)
            comparison_data["company_data"] = company_data

            # Get peer data
            peer_data = {}
            for peer in peers:
                peer_metrics = self._get_company_metrics(peer, metrics_to_fetch)
                peer_data[peer] = peer_metrics

            comparison_data["peer_data"] = peer_data

            # Calculate averages, min, max for each metric
            metric_statistics = {}

            for metric in metrics_to_fetch:
                values = [company_data.get(metric)]
                values.extend([peer_data[peer].get(metric) for peer in peers if peer in peer_data])
                values = [v for v in values if v is not None]

                if values:
                    metric_statistics[metric] = {
                        "average": sum(values) / len(values),
                        "median": np.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std": np.std(values) if len(values) > 1 else 0
                    }

            comparison_data["metric_statistics"] = metric_statistics

            return comparison_data

        except Exception as e:
            logger.error(f"Error getting peer comparison metrics for {ticker}: {e}")
            return {"error": str(e)}

    def get_peer_performance_comparison(self, ticker: str, peers: List[str],
                                        period: str = '1y') -> Dict[str, Any]:
        """
        Compare stock price performance between a company and its peers

        Args:
            ticker: Ticker symbol of the main company
            peers: List of peer company tickers
            period: Time period for comparison ('1m', '3m', '6m', '1y', '3y', '5y')

        Returns:
            Dictionary with performance comparison data
        """
        try:
            # Determine date range
            end_date = datetime.now().strftime('%Y-%m-%d')

            if period == '1m':
                days = 30
            elif period == '3m':
                days = 90
            elif period == '6m':
                days = 180
            elif period == '3y':
                days = 3 * 365
            elif period == '5y':
                days = 5 * 365
            else:  # Default to 1y
                days = 365

            start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

            # Get price data for main company
            company_prices = self.data_loader.get_historical_prices(ticker, start_date, end_date)

            if company_prices.empty:
                return {"error": f"No price data available for {ticker}"}

            # Get price data for peers
            peer_prices = {}

            for peer in peers:
                prices = self.data_loader.get_historical_prices(peer, start_date, end_date)
                if not prices.empty:
                    peer_prices[peer] = prices

            # Calculate performance metrics
            performance_data = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "tickers": [ticker] + peers,
                "price_change": {},
                "relative_performance": {},
                "correlation": {},
                "volatility": {}
            }

            # Calculate metrics for main company
            company_returns = self._calculate_performance_metrics(company_prices)
            performance_data["price_change"][ticker] = company_returns
            performance_data["volatility"][ticker] = company_returns["volatility"]

            # Calculate metrics for peers and relative performance
            for peer, prices in peer_prices.items():
                peer_returns = self._calculate_performance_metrics(prices)
                performance_data["price_change"][peer] = peer_returns
                performance_data["volatility"][peer] = peer_returns["volatility"]

                # Calculate correlation with main company
                if len(company_prices) > 1 and len(prices) > 1:
                    # Align dates
                    combined = pd.DataFrame({
                        'company': company_prices['Close'],
                        'peer': prices['Close']
                    })
                    combined = combined.dropna()

                    if len(combined) > 1:
                        correlation = combined['company'].corr(combined['peer'])
                        performance_data["correlation"][peer] = correlation

            # Calculate relative performance to main company
            main_change = performance_data["price_change"][ticker]["total_change"]

            for peer in peers:
                if peer in performance_data["price_change"]:
                    peer_change = performance_data["price_change"][peer]["total_change"]
                    relative_perf = peer_change - main_change
                    performance_data["relative_performance"][peer] = relative_perf

            return performance_data

        except Exception as e:
            logger.error(f"Error comparing performance for {ticker} and peers: {e}")
            return {"error": str(e)}

    def _find_peers_by_sector(self, sector: str, industry: str) -> List[str]:
        """Find potential peers based on sector and industry"""
        # This would typically use a database or API to find companies in the same sector/industry
        # For simplicity, we'll return a predefined list based on sector

        sector_peers = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "NFLX", "NVDA", "ADBE", "CRM", "INTC",
                           "CSCO", "ORCL", "IBM", "AMD", "PYPL", "QCOM"],
            "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "TMO", "ABT", "MDT", "AMGN",
                           "BMY", "CVS", "CI", "ZTS", "ISRG", "VRTX"],
            "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
                           "SCHW", "CB", "PNC", "TFC", "USB", "COF"],
            "Consumer Discretionary": ["AMZN", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "TJX", "BKNG", "EBAY",
                                       "BBY", "DG", "DLTR", "GPS", "F", "GM"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "EL", "GIS", "K", "KHC",
                                 "KMB", "MO", "PM", "STZ", "SYY", "TSN"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "PSX", "VLO", "SLB", "OXY", "MPC", "KMI",
                       "WMB", "HES", "HAL", "BKR", "DVN", "MRO"],
            "Industrials": ["HON", "UNP", "UPS", "BA", "CAT", "GE", "MMM", "LMT", "RTX", "DE",
                            "FDX", "GD", "CSX", "EMR", "ETN", "WM"],
            "Materials": ["LIN", "APD", "ECL", "SHW", "DD", "NEM", "NUE", "FCX", "DOW", "PPG",
                          "IP", "BLL", "VMC", "MLM", "CF", "MOS"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "AVB", "EQR", "DLR", "O", "SPG",
                            "WELL", "VTR", "ESS", "BXP", "UDR", "HST"],
            "Communication Services": ["GOOGL", "FB", "NFLX", "TMUS", "CMCSA", "VZ", "T", "CHTR", "ATVI", "EA",
                                       "TTWO", "DIS", "MTCH", "TWTR", "OMC", "LYV"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "ED", "PEG",
                          "ES", "WEC", "AEE", "CMS", "FE", "ETR"]
        }

        # First try to find peers by industry within the sector mapping
        industry_peers = []
        for sector_key, industries in self.sector_mapping.items():
            if sector.lower() in sector_key.lower() or sector_key.lower() in sector.lower():
                for industry_name in industries:
                    if industry.lower() in industry_name.lower() or industry_name.lower() in industry.lower():
                        # Found a match - would query database here in a real implementation
                        # For now, return peers from the parent sector
                        return sector_peers.get(sector_key, [])

        # Fallback to sector-based peers
        for sector_key, peers in sector_peers.items():
            if sector.lower() in sector_key.lower() or sector_key.lower() in sector.lower():
                return peers

        # Return empty list if no matches
        return []

    def _calculate_similarity(self, main_ticker: str, peer_ticker: str,
                              metrics: List[str], main_info: Dict, peer_info: Dict,
                              main_ratios: Dict, peer_ratios: Dict) -> float:
        """Calculate similarity score between two companies based on multiple metrics"""
        similarity_scores = []

        for metric in metrics:
            metric_score = self._calculate_metric_similarity(
                metric, main_ticker, peer_ticker, main_info, peer_info, main_ratios, peer_ratios
            )

            if metric_score is not None:
                similarity_scores.append(metric_score)

        # Return average similarity score if we have any scores
        if similarity_scores:
            return sum(similarity_scores) / len(similarity_scores)

        # If no metrics could be compared, return a low similarity
        return 0.2

    def _calculate_metric_similarity(self, metric: str, main_ticker: str, peer_ticker: str,
                                     main_info: Dict, peer_info: Dict,
                                     main_ratios: Dict, peer_ratios: Dict) -> Optional[float]:
        """Calculate similarity score for a specific metric"""
        # Different calculation methods based on metric type
        if metric == "Market Cap":
            main_value = main_info.get('market_cap')
            peer_value = peer_info.get('market_cap')

            if main_value and peer_value and main_value > 0:
                ratio = min(peer_value / main_value, main_value / peer_value)
                # Transform to a similarity score between 0 and 1
                return ratio

        elif metric == "Revenue":
            # Get revenue from income statement
            main_income = self.data_loader.get_financial_statements(main_ticker, 'income')
            peer_income = self.data_loader.get_financial_statements(peer_ticker, 'income')

            if not main_income.empty and not peer_income.empty:
                main_revenue = main_income.iloc[
                    main_income.index.get_loc('Total Revenue'), 0] if 'Total Revenue' in main_income.index else None
                peer_revenue = peer_income.iloc[
                    peer_income.index.get_loc('Total Revenue'), 0] if 'Total Revenue' in peer_income.index else None

                if main_revenue and peer_revenue and main_revenue > 0:
                    ratio = min(peer_revenue / main_revenue, main_revenue / peer_revenue)
                    return ratio

        elif "Margin" in metric:
            # Check in financial ratios
            metric_key = metric.lower().replace(' ', '_')

            main_value = None
            peer_value = None

            # Try to find in different categories
            for category in main_ratios:
                if isinstance(main_ratios[category], dict) and metric_key in main_ratios[category]:
                    main_value = main_ratios[category][metric_key]

            for category in peer_ratios:
                if isinstance(peer_ratios[category], dict) and metric_key in peer_ratios[category]:
                    peer_value = peer_ratios[category][metric_key]

            if main_value is not None and peer_value is not None:
                # Calculate absolute difference and convert to similarity
                diff = abs(main_value - peer_value)
                # Scale difference to similarity (0 = identical, 1 = very different)
                # Margins typically range from 0 to 1, so scale accordingly
                similarity = max(0, 1 - diff)
                return similarity

        elif metric == "Beta":
            main_beta = main_info.get('beta')
            peer_beta = peer_info.get('beta')

            if main_beta is not None and peer_beta is not None:
                diff = abs(main_beta - peer_beta)
                # Beta typically ranges from 0 to 2, so scale accordingly
                similarity = max(0, 1 - diff / 2)
                return similarity

        elif "Growth" in metric:
            # Try to calculate growth rates if not readily available
            growth_metric = metric.lower().replace(' ', '_')

            # For growth metrics, check multiple periods if available
            main_growth = self._calculate_growth_rate(main_ticker, metric)
            peer_growth = self._calculate_growth_rate(peer_ticker, metric)

            if main_growth is not None and peer_growth is not None:
                # Growth rates can vary widely, so use a logarithmic scale
                # to measure similarity
                ratio = (min(main_growth, peer_growth) + 0.1) / (max(main_growth, peer_growth) + 0.1)
                return ratio

        # Default case or if metric not found
        return None

    def _calculate_growth_rate(self, ticker: str, growth_metric: str) -> Optional[float]:
        """Calculate growth rate for a specific metric"""
        # Get financial statements
        income_stmt = self.data_loader.get_financial_statements(ticker, 'income')

        if income_stmt.empty or income_stmt.shape[1] < 2:
            return None

        # Determine what value to use based on growth metric
        if "Revenue" in growth_metric:
            if 'Total Revenue' in income_stmt.index:
                current = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                previous = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 1]

                if previous and previous > 0:
                    return (current / previous) - 1

        elif "EPS" in growth_metric:
            if 'EPS (Basic)' in income_stmt.index:
                current = income_stmt.iloc[income_stmt.index.get_loc('EPS (Basic)'), 0]
                previous = income_stmt.iloc[income_stmt.index.get_loc('EPS (Basic)'), 1]

                if previous and previous > 0:
                    return (current / previous) - 1

        elif "Net Income" in growth_metric or "Earnings" in growth_metric:
            if 'Net Income' in income_stmt.index:
                current = income_stmt.iloc[income_stmt.index.get_loc('Net Income'), 0]
                previous = income_stmt.iloc[income_stmt.index.get_loc('Net Income'), 1]

                if previous and previous > 0:
                    return (current / previous) - 1

        return None

    def _get_company_metrics(self, ticker: str, metrics: List[str]) -> Dict[str, Any]:
        """Get specific metrics for a company"""
        company_metrics = {}

        try:
            # Get company info and financial ratios
            company_info = self.data_loader.get_company_info(ticker)
            financial_ratios = self.data_loader.get_financial_ratios(ticker)

            # Get financial statements
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income')
            balance_sheet = self.data_loader.get_financial_statements(ticker, 'balance')

            # Map metrics to data sources and extract values
            for metric in metrics:
                # Check if metric is in company info
                if metric.lower().replace(' ', '_') in company_info:
                    company_metrics[metric] = company_info[metric.lower().replace(' ', '_')]
                    continue

                # Check in financial ratios
                metric_found = False
                for category, ratios in financial_ratios.items():
                    if isinstance(ratios, dict):
                        for ratio_name, ratio_value in ratios.items():
                            if ratio_name.lower().replace('_', ' ') == metric.lower() or \
                                    ratio_name.lower() == metric.lower().replace(' ', '_'):
                                company_metrics[metric] = ratio_value
                                metric_found = True
                                break

                    if metric_found:
                        break

                if metric_found:
                    continue

                # Check in income statement
                if not income_stmt.empty:
                    if metric in income_stmt.index:
                        company_metrics[metric] = income_stmt.iloc[income_stmt.index.get_loc(metric), 0]
                        continue

                    # Common metrics with different naming
                    if metric == "Revenue" and "Total Revenue" in income_stmt.index:
                        company_metrics[metric] = income_stmt.iloc[income_stmt.index.get_loc("Total Revenue"), 0]
                        continue

                    if metric == "EPS" and "EPS (Basic)" in income_stmt.index:
                        company_metrics[metric] = income_stmt.iloc[income_stmt.index.get_loc("EPS (Basic)"), 0]
                        continue

                # Check in balance sheet
                if not balance_sheet.empty:
                    if metric in balance_sheet.index:
                        company_metrics[metric] = balance_sheet.iloc[balance_sheet.index.get_loc(metric), 0]
                        continue

                # Calculate metrics that might not be directly available
                if metric == "ROE" and not balance_sheet.empty and not income_stmt.empty:
                    if "Net Income" in income_stmt.index and "Total Stockholder Equity" in balance_sheet.index:
                        net_income = income_stmt.iloc[income_stmt.index.get_loc("Net Income"), 0]
                        equity = balance_sheet.iloc[balance_sheet.index.get_loc("Total Stockholder Equity"), 0]
                        if equity > 0:
                            company_metrics[metric] = net_income / equity

                if metric == "ROA" and not balance_sheet.empty and not income_stmt.empty:
                    if "Net Income" in income_stmt.index and "Total Assets" in balance_sheet.index:
                        net_income = income_stmt.iloc[income_stmt.index.get_loc("Net Income"), 0]
                        assets = balance_sheet.iloc[balance_sheet.index.get_loc("Total Assets"), 0]
                        if assets > 0:
                            company_metrics[metric] = net_income / assets

                # Growth metrics
                if "Growth" in metric and not income_stmt.empty and income_stmt.shape[1] >= 2:
                    growth_rate = self._calculate_growth_rate(ticker, metric)
                    if growth_rate is not None:
                        company_metrics[metric] = growth_rate

            return company_metrics

        except Exception as e:
            logger.error(f"Error getting metrics for {ticker}: {e}")
            return company_metrics

    def _calculate_performance_metrics(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from price data"""
        if price_data.empty or len(price_data) < 2:
            return {
                "total_change": 0,
                "annualized_return": 0,
                "volatility": 0
            }

        try:
            # Calculate daily returns
            returns = price_data['Close'].pct_change().dropna()

            # Calculate total price change
            start_price = price_data['Close'].iloc[0]
            end_price = price_data['Close'].iloc[-1]
            total_change = (end_price / start_price) - 1

            # Calculate volatility (annualized)
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)

            # Calculate annualized return
            days = (price_data.index[-1] - price_data.index[0]).days
            if days > 0:
                annualized_return = ((1 + total_change) ** (365 / days)) - 1
            else:
                annualized_return = 0

            return {
                "total_change": total_change,
                "annualized_return": annualized_return,
                "volatility": annualized_volatility
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "total_change": 0,
                "annualized_return": 0,
                "volatility": 0
            }