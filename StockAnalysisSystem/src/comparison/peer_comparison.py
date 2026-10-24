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

from StockAnalysisSystem.src.config import PEER_ANALYSIS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.comparison.peer_finder import PeerFinder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('peer_comparison')


class PeerComparison:
    """
    Class for performing comprehensive peer comparison analysis.
    Compares a company with its peers across various financial metrics,
    performance indicators, and valuation multiples.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the peer comparison system

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self.peer_finder = PeerFinder(data_loader)

        # Load configuration
        self.peer_config = PEER_ANALYSIS

        logger.info("Peer comparison system initialized")

    def analyze_against_peers(self, ticker: str, peer_tickers: Optional[List[str]] = None,
                              find_peers: bool = True, max_peers: int = 5,
                              category_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive peer comparison analysis

        Args:
            ticker: Main company ticker symbol
            peer_tickers: Optional list of peer tickers to compare against
            find_peers: Whether to automatically find additional peers if needed
            max_peers: Maximum number of peers to include in analysis
            category_filter: Optional list of metric categories to include

        Returns:
            Dictionary with peer comparison analysis results
        """
        try:
            # Get company information
            company_info = self.data_loader.get_company_info(ticker)

            if not company_info:
                return {"error": f"Failed to retrieve information for {ticker}"}

            # Get or find peer companies
            peers = []

            if peer_tickers:
                # Get information for specified peers
                for peer in peer_tickers:
                    peer_info = self.data_loader.get_company_info(peer)
                    if peer_info:
                        peers.append({
                            "ticker": peer,
                            "name": peer_info.get('name', peer),
                            "sector": peer_info.get('sector', 'N/A'),
                            "industry": peer_info.get('industry', 'N/A'),
                            "market_cap": peer_info.get('market_cap', None)
                        })

                        # Get company information
                        company_info = self.data_loader.get_company_info(ticker)

                        if not company_info:
                            return {"error": f"Failed to retrieve information for {ticker}"}

                        # Get or find peer companies
                        peers = []

                        if peer_tickers:
                            # Get information for specified peers
                            for peer in peer_tickers:
                                peer_info = self.data_loader.get_company_info(peer)
                                if peer_info:
                                    peers.append({
                                        "ticker": peer,
                                        "name": peer_info.get('name', peer),
                                        "sector": peer_info.get('sector', 'N/A'),
                                        "industry": peer_info.get('industry', 'N/A'),
                                        "market_cap": peer_info.get('market_cap', None)
                                    })

                        # Find additional peers if needed
                        if find_peers and (not peers or len(peers) < max_peers):
                            needed_peers = max_peers - len(peers)
                            found_peers = self.peer_finder.find_peers(ticker, max_peers=needed_peers)

                            # Add found peers if they're not already in the list
                            existing_tickers = [p["ticker"] for p in peers]
                            for peer in found_peers:
                                if peer["ticker"] not in existing_tickers and peer["ticker"] != ticker:
                                    peers.append(peer)

                        # Prepare the result structure
                        result = {
                            "company": {
                                "ticker": ticker,
                                "name": company_info.get('name', ticker),
                                "sector": company_info.get('sector', 'N/A'),
                                "industry": company_info.get('industry', 'N/A'),
                                "market_cap": company_info.get('market_cap', None)
                            },
                            "peers": peers,
                            "comparison": {},
                            "performance": {},
                            "valuation": {},
                            "efficiency": {},
                            "financial_health": {},
                            "growth": {},
                            "percentile_rankings": {},
                            "advanced_analysis": {}
                        }

                        # Get metric categories to include
                        if category_filter:
                            categories = category_filter
                        else:
                            categories = self.peer_config.get('comparison_metrics', {}).keys()

                        # Perform comparison analysis for selected categories
                        if peers:
                            peer_tickers = [p["ticker"] for p in peers]
                            comparison_data = self.peer_finder.get_peer_comparison_metrics(ticker, peer_tickers,
                                                                                           categories)

                            # Process comparison data by category
                            for category in categories:
                                category_metrics = self.peer_config.get('comparison_metrics', {}).get(category, [])

                                if category_metrics:
                                    category_comparison = self._analyze_category_comparison(
                                        comparison_data, category, category_metrics
                                    )

                                    result["comparison"][category] = category_comparison

                            # Performance comparison
                            performance_comparison = self.peer_finder.get_peer_performance_comparison(ticker,
                                                                                                      peer_tickers)
                            result["performance"] = self._process_performance_comparison(performance_comparison)

                            # Percentile rankings
                            result["percentile_rankings"] = self._calculate_percentile_rankings(ticker, peer_tickers)

                            # Advanced analysis
                            result["advanced_analysis"] = self._perform_advanced_analysis(ticker, peer_tickers)

                        return result

        except Exception as e:
            logger.error(f"Error in peer comparison analysis for {ticker}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def compare_specific_metrics(self, ticker: str, peer_tickers: List[str],
                                 metrics: List[str]) -> Dict[str, Any]:
        """
        Compare specific metrics between a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers to compare against
            metrics: List of specific metrics to compare

        Returns:
            Dictionary with metric comparison results
        """
        try:
            # Get company information
            company_info = self.data_loader.get_company_info(ticker)

            if not company_info:
                return {"error": f"Failed to retrieve information for {ticker}"}

            # Get peer information
            peers = []

            for peer in peer_tickers:
                peer_info = self.data_loader.get_company_info(peer)
                if peer_info:
                    peers.append({
                        "ticker": peer,
                        "name": peer_info.get('name', peer)
                    })

            # Prepare result structure
            result = {
                "company": {
                    "ticker": ticker,
                    "name": company_info.get('name', ticker)
                },
                "peers": peers,
                "metrics": {}
            }

            # Get data for each metric
            for metric in metrics:
                metric_data = {
                    "name": metric,
                    "company_value": None,
                    "peer_values": {},
                    "average": None,
                    "median": None,
                    "min": None,
                    "max": None
                }

                # Get company value
                company_value = self._get_metric_value(ticker, metric)
                metric_data["company_value"] = company_value

                # Get peer values
                values = []

                for peer in peer_tickers:
                    peer_value = self._get_metric_value(peer, metric)
                    if peer_value is not None:
                        metric_data["peer_values"][peer] = peer_value
                        values.append(peer_value)

                # Calculate statistics
                if values:
                    metric_data["average"] = sum(values) / len(values)
                    metric_data["median"] = np.median(values)
                    metric_data["min"] = min(values)
                    metric_data["max"] = max(values)

                    # Calculate percentile
                    if company_value is not None:
                        # Determine if higher is better for this metric
                        higher_is_better = not self._is_lower_better(metric)

                        # Sort values based on whether higher or lower is better
                        sorted_values = sorted(values, reverse=higher_is_better)

                        # Find position
                        try:
                            position = next(i for i, v in enumerate(sorted_values) if
                                            v >= company_value) if higher_is_better else next(
                                i for i, v in enumerate(sorted_values) if v <= company_value)
                            percentile = (len(sorted_values) - position) / len(sorted_values) * 100
                            metric_data["percentile"] = percentile
                        except:
                            metric_data["percentile"] = None

                result["metrics"][metric] = metric_data

            return result

        except Exception as e:
            logger.error(f"Error comparing specific metrics for {ticker}: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def compare_valuation_multiples(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Compare valuation multiples between a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers to compare against

        Returns:
            Dictionary with valuation multiple comparison results
        """
        try:
            # Define valuation multiples to compare
            valuation_multiples = [
                "pe_ratio", "forward_pe", "peg_ratio", "ps_ratio", "pb_ratio",
                "ev_ebitda", "ev_revenue", "ev_fcf"
            ]

            # Get comparison results
            comparison = self.compare_specific_metrics(ticker, peer_tickers, valuation_multiples)

            # Add interpretation
            if "metrics" in comparison:
                for metric, data in comparison["metrics"].items():
                    if data["company_value"] is not None and data["average"] is not None:
                        # Determine if lower is better (true for all valuation multiples)
                        relative_value = data["company_value"] / data["average"] if data["average"] > 0 else 1

                        if relative_value < 0.7:
                            data["interpretation"] = "Significantly undervalued compared to peers"
                        elif relative_value < 0.9:
                            data["interpretation"] = "Moderately undervalued compared to peers"
                        elif relative_value < 1.1:
                            data["interpretation"] = "Valued in line with peers"
                        elif relative_value < 1.3:
                            data["interpretation"] = "Moderately overvalued compared to peers"
                        else:
                            data["interpretation"] = "Significantly overvalued compared to peers"

            return comparison

        except Exception as e:
            logger.error(f"Error comparing valuation multiples for {ticker}: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def compare_financial_strength(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Compare financial strength metrics between a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers to compare against

        Returns:
            Dictionary with financial strength comparison results
        """
        try:
            # Define financial strength metrics to compare
            financial_metrics = [
                "current_ratio", "quick_ratio", "debt_to_equity", "debt_to_ebitda",
                "interest_coverage", "roe", "roa", "roic"
            ]

            # Get comparison results
            comparison = self.compare_specific_metrics(ticker, peer_tickers, financial_metrics)

            # Calculate overall financial strength
            if "metrics" in comparison:
                strength_scores = {}

                # Get company's values
                company_values = {}
                for metric, data in comparison["metrics"].items():
                    company_values[metric] = data["company_value"]

                # Calculate liquidity score
                liquidity_score = self._calculate_liquidity_score(company_values)
                strength_scores["liquidity"] = liquidity_score

                # Calculate solvency score
                solvency_score = self._calculate_solvency_score(company_values)
                strength_scores["solvency"] = solvency_score

                # Calculate profitability score
                profitability_score = self._calculate_profitability_score(company_values)
                strength_scores["profitability"] = profitability_score

                # Calculate overall score
                overall_score = (liquidity_score + solvency_score + profitability_score) / 3
                strength_scores["overall"] = overall_score

                comparison["financial_strength"] = strength_scores

                # Get peer financial strength scores
                peer_strength = {}

                for peer in peer_tickers:
                    peer_values = {}
                    for metric, data in comparison["metrics"].items():
                        if peer in data["peer_values"]:
                            peer_values[metric] = data["peer_values"][peer]

                    peer_liquidity = self._calculate_liquidity_score(peer_values)
                    peer_solvency = self._calculate_solvency_score(peer_values)
                    peer_profitability = self._calculate_profitability_score(peer_values)

                    peer_overall = (peer_liquidity + peer_solvency + peer_profitability) / 3

                    peer_strength[peer] = {
                        "liquidity": peer_liquidity,
                        "solvency": peer_solvency,
                        "profitability": peer_profitability,
                        "overall": peer_overall
                    }

                comparison["peer_financial_strength"] = peer_strength

            return comparison

        except Exception as e:
            logger.error(f"Error comparing financial strength for {ticker}: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def compare_growth_metrics(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Compare growth metrics between a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers to compare against

        Returns:
            Dictionary with growth metrics comparison results
        """
        try:
            # Define growth metrics to compare
            growth_metrics = [
                "revenue_growth", "eps_growth", "ebitda_growth", "fcf_growth"
            ]

            # Get comparison results
            comparison = self.compare_specific_metrics(ticker, peer_tickers, growth_metrics)

            # Add growth premium/discount analysis
            if "metrics" in comparison and "company" in comparison:
                company_ticker = comparison["company"]["ticker"]

                # Get P/E ratios
                company_pe = self._get_metric_value(company_ticker, "pe_ratio")

                peer_pes = {}
                for peer in peer_tickers:
                    peer_pe = self._get_metric_value(peer, "pe_ratio")
                    if peer_pe is not None:
                        peer_pes[peer] = peer_pe

                peer_avg_pe = sum(peer_pes.values()) / len(peer_pes) if peer_pes else None

                # Get EPS growth rates
                company_eps_growth = comparison["metrics"].get("eps_growth", {}).get("company_value")

                peer_eps_growth = {}
                if "eps_growth" in comparison["metrics"]:
                    peer_eps_growth = comparison["metrics"]["eps_growth"].get("peer_values", {})

                peer_avg_eps_growth = sum(peer_eps_growth.values()) / len(
                    peer_eps_growth) if peer_eps_growth else None

                # Calculate PEG ratios
                company_peg = company_pe / company_eps_growth if company_pe and company_eps_growth and company_eps_growth > 0 else None

                peer_pegs = {}
                for peer, pe in peer_pes.items():
                    if peer in peer_eps_growth and peer_eps_growth[peer] and peer_eps_growth[peer] > 0:
                        peer_pegs[peer] = pe / peer_eps_growth[peer]

                peer_avg_peg = sum(peer_pegs.values()) / len(peer_pegs) if peer_pegs else None

                # Add growth premium analysis
                growth_premium = {
                    "company_pe": company_pe,
                    "peer_avg_pe": peer_avg_pe,
                    "company_eps_growth": company_eps_growth,
                    "peer_avg_eps_growth": peer_avg_eps_growth,
                    "company_peg": company_peg,
                    "peer_avg_peg": peer_avg_peg
                }

                # Add interpretation
                if company_peg and peer_avg_peg:
                    peg_ratio = company_peg / peer_avg_peg

                    if peg_ratio < 0.7:
                        growth_premium["interpretation"] = "Significant growth discount relative to peers"
                    elif peg_ratio < 0.9:
                        growth_premium["interpretation"] = "Moderate growth discount relative to peers"
                    elif peg_ratio < 1.1:
                        growth_premium["interpretation"] = "Valued appropriately for growth relative to peers"
                    elif peg_ratio < 1.3:
                        growth_premium["interpretation"] = "Moderate growth premium relative to peers"
                    else:
                        growth_premium["interpretation"] = "Significant growth premium relative to peers"

                comparison["growth_premium"] = growth_premium

            return comparison

        except Exception as e:
            logger.error(f"Error comparing growth metrics for {ticker}: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def _analyze_category_comparison(self, comparison_data: Dict[str, Any],
                                     category: str, metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze peer comparison data for a specific category of metrics

        Args:
            comparison_data: Raw comparison data from peer_finder
            category: Metric category (e.g., "Valuation", "Profitability")
            metrics: List of metrics in this category

        Returns:
            Dictionary with analysis results for the category
        """
        result = {}

        # Get company data
        company_data = comparison_data.get("company_data", {})

        # Get peer data
        peer_data = comparison_data.get("peer_data", {})

        # Get statistics
        stats = comparison_data.get("metric_statistics", {})

        # Analyze each metric
        for metric in metrics:
            if metric in stats:
                # Get company value
                company_value = company_data.get(metric)

                # Get statistics
                metric_stats = stats[metric]

                # Create metric analysis
                metric_analysis = {
                    "name": metric,
                    "company_value": company_value,
                    "peer_average": metric_stats.get("average"),
                    "peer_median": metric_stats.get("median"),
                    "peer_min": metric_stats.get("min"),
                    "peer_max": metric_stats.get("max"),
                    "peer_values": {peer: data.get(metric) for peer, data in peer_data.items() if
                                    metric in data}
                }

                # Calculate percent difference from peer average
                if company_value is not None and metric_stats.get("average") is not None and metric_stats.get(
                        "average") != 0:
                    percent_diff = ((company_value / metric_stats["average"]) - 1) * 100
                    metric_analysis["percent_diff_from_average"] = percent_diff

                    # Determine if the difference is favorable
                    higher_is_better = not self._is_lower_better(metric)
                    favorable = (percent_diff > 0 and higher_is_better) or (
                                percent_diff < 0 and not higher_is_better)
                    metric_analysis["favorable"] = favorable

                    # Add interpretation
                    metric_analysis["interpretation"] = self._interpret_metric_comparison(
                        metric, company_value, metric_stats["average"], percent_diff, favorable
                    )

                # Add to results
                result[metric] = metric_analysis

        return result

    def _process_performance_comparison(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and analyze performance comparison data

        Args:
            performance_data: Raw performance comparison data from peer_finder

        Returns:
            Dictionary with processed performance comparison
        """
        result = {}

        # Extract price change data
        price_change = performance_data.get("price_change", {})
        result["price_change"] = price_change

        # Extract relative performance
        relative_performance = performance_data.get("relative_performance", {})
        result["relative_performance"] = relative_performance

        # Extract correlation data
        correlation = performance_data.get("correlation", {})
        result["correlation"] = correlation

        # Extract volatility data
        volatility = performance_data.get("volatility", {})
        result["volatility"] = volatility

        # Add period information
        result["period"] = performance_data.get("period")
        result["start_date"] = performance_data.get("start_date")
        result["end_date"] = performance_data.get("end_date")

        # Add tickers
        result["tickers"] = performance_data.get("tickers", [])

        # Calculate performance ranking
        company_ticker = result["tickers"][0] if result["tickers"] else None

        if company_ticker:
            # Rank by total price change
            tickers_by_return = sorted(
                result["tickers"],
                key=lambda t: price_change.get(t, {}).get("total_change", -999),
                reverse=True
            )

            result["return_ranking"] = {
                "tickers": tickers_by_return,
                "company_rank": tickers_by_return.index(
                    company_ticker) + 1 if company_ticker in tickers_by_return else None
            }

            # Rank by volatility (lower is better)
            tickers_by_volatility = sorted(
                result["tickers"],
                key=lambda t: volatility.get(t, 999)
            )

            result["volatility_ranking"] = {
                "tickers": tickers_by_volatility,
                "company_rank": tickers_by_volatility.index(
                    company_ticker) + 1 if company_ticker in tickers_by_volatility else None
            }

        return result

    def _calculate_percentile_rankings(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate percentile rankings for a company within its peer group

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers

        Returns:
            Dictionary with percentile rankings for various metrics
        """
        rankings = {}

        # Define metrics to rank by category
        metrics_by_category = {
            "Valuation": ["pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda"],
            "Profitability": ["operating_margin", "net_margin", "roe", "roa"],
            "Growth": ["revenue_growth", "eps_growth"],
            "Financial Health": ["current_ratio", "debt_to_equity", "interest_coverage"]
        }

        # Calculate rankings for each category
        for category, metrics in metrics_by_category.items():
            category_rankings = {}

            for metric in metrics:
                # Get company value
                company_value = self._get_metric_value(ticker, metric)

                if company_value is not None:
                    # Get peer values
                    peer_values = []

                    for peer in peer_tickers:
                        peer_value = self._get_metric_value(peer, metric)
                        if peer_value is not None:
                            peer_values.append(peer_value)

                    if peer_values:
                        # Determine if higher is better
                        higher_is_better = not self._is_lower_better(metric)

                        # Calculate percentile rank
                        percentile = self._calculate_percentile(company_value, peer_values, higher_is_better)

                        category_rankings[metric] = {
                            "value": company_value,
                            "percentile": percentile,
                            "higher_is_better": higher_is_better
                        }

            if category_rankings:
                rankings[category] = category_rankings

        return rankings

    def _perform_advanced_analysis(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Perform advanced comparative analysis between a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers

        Returns:
            Dictionary with advanced analysis results
        """
        results = {}

        # Calculate returns vs. risk (Sharpe ratio proxy)
        try:
            returns_risk = self._analyze_returns_vs_risk(ticker, peer_tickers)
            results["returns_vs_risk"] = returns_risk
        except Exception as e:
            logger.warning(f"Error calculating returns vs. risk: {e}")

        # Calculate efficiency analysis
        try:
            efficiency = self._analyze_efficiency_metrics(ticker, peer_tickers)
            results["efficiency"] = efficiency
        except Exception as e:
            logger.warning(f"Error calculating efficiency metrics: {e}")

        # Calculate valuation premium/discount
        try:
            valuation = self._analyze_valuation_premium(ticker, peer_tickers)
            results["valuation_premium"] = valuation
        except Exception as e:
            logger.warning(f"Error calculating valuation premium: {e}")

        return results

    def _analyze_returns_vs_risk(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Analyze returns vs. risk metrics for a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers

        Returns:
            Dictionary with returns vs. risk analysis
        """
        # Get performance data for 1 year
        performance = self.peer_finder.get_peer_performance_comparison(ticker, peer_tickers, period='1y')

        price_change = performance.get("price_change", {})
        volatility = performance.get("volatility", {})

        # Calculate Sharpe ratio proxy (using total return / volatility)
        sharpe_ratios = {}

        # Add main company
        if ticker in price_change and ticker in volatility:
            company_return = price_change[ticker].get("total_change", 0)
            company_vol = volatility[ticker]

            if company_vol > 0:
                sharpe_ratios[ticker] = company_return / company_vol

        # Add peers
        for peer in peer_tickers:
            if peer in price_change and peer in volatility:
                peer_return = price_change[peer].get("total_change", 0)
                peer_vol = volatility[peer]

                if peer_vol > 0:
                    sharpe_ratios[peer] = peer_return / peer_vol

        # Rank companies by Sharpe ratio
        ranked_tickers = sorted(sharpe_ratios.keys(), key=lambda t: sharpe_ratios[t], reverse=True)

        # Calculate company's rank
        company_rank = ranked_tickers.index(ticker) + 1 if ticker in ranked_tickers else None

        return {
            "sharpe_ratios": sharpe_ratios,
            "ranking": ranked_tickers,
            "company_rank": company_rank
        }

    def _analyze_efficiency_metrics(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Analyze operational efficiency metrics for a company and its peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers

        Returns:
            Dictionary with efficiency metrics analysis
        """
        # Define efficiency metrics
        efficiency_metrics = ["asset_turnover", "inventory_turnover", "receivables_turnover"]

        # Get metric values
        company_values = {}
        peer_values = {peer: {} for peer in peer_tickers}

        for metric in efficiency_metrics:
            # Get company value
            company_value = self._get_metric_value(ticker, metric)
            if company_value is not None:
                company_values[metric] = company_value

            # Get peer values
            for peer in peer_tickers:
                peer_value = self._get_metric_value(peer, metric)
                if peer_value is not None:
                    peer_values[peer][metric] = peer_value

        # Calculate average efficiency scores
        efficiency_scores = {}

        # Add main company
        if company_values:
            # Normalize and average values
            normalized_values = []

            for metric, value in company_values.items():
                # Get all peer values for this metric
                all_values = [value]
                for peer in peer_tickers:
                    if metric in peer_values[peer]:
                        all_values.append(peer_values[peer][metric])

                # Skip if not enough data
                if len(all_values) < 2:
                    continue

                # Calculate min and max
                min_val = min(all_values)
                max_val = max(all_values)

                # Normalize (higher is better for efficiency metrics)
                if max_val > min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                    normalized_values.append(normalized)

            # Calculate average efficiency score
            if normalized_values:
                efficiency_scores[ticker] = sum(normalized_values) / len(normalized_values)

        # Add peers
        for peer in peer_tickers:
            if peer_values[peer]:
                # Normalize and average values
                normalized_values = []

                for metric, value in peer_values[peer].items():
                    # Get all values for this metric
                    all_values = []
                    if metric in company_values:
                        all_values.append(company_values[metric])

                    for p in peer_tickers:
                        if metric in peer_values[p]:
                            all_values.append(peer_values[p][metric])

                    # Skip if not enough data
                    if len(all_values) < 2:
                        continue

                    # Calculate min and max
                    min_val = min(all_values)
                    max_val = max(all_values)

                    # Normalize (higher is better for efficiency metrics)
                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                        normalized_values.append(normalized)

                # Calculate average efficiency score
                if normalized_values:
                    efficiency_scores[peer] = sum(normalized_values) / len(normalized_values)

        # Rank companies by efficiency score
        ranked_tickers = sorted(efficiency_scores.keys(), key=lambda t: efficiency_scores[t], reverse=True)

        # Calculate company's rank
        company_rank = ranked_tickers.index(ticker) + 1 if ticker in ranked_tickers else None

        return {
            "efficiency_scores": efficiency_scores,
            "ranking": ranked_tickers,
            "company_rank": company_rank
        }

    def _analyze_valuation_premium(self, ticker: str, peer_tickers: List[str]) -> Dict[str, Any]:
        """
        Analyze valuation premium/discount relative to peers

        Args:
            ticker: Main company ticker symbol
            peer_tickers: List of peer tickers

        Returns:
            Dictionary with valuation premium analysis
        """
        # Define valuation metrics
        valuation_metrics = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"]

        # Get metric values
        company_values = {}
        all_values = {metric: [] for metric in valuation_metrics}

        for metric in valuation_metrics:
            # Get company value
            company_value = self._get_metric_value(ticker, metric)
            if company_value is not None:
                company_values[metric] = company_value
                all_values[metric].append(company_value)

            # Get peer values
            for peer in peer_tickers:
                peer_value = self._get_metric_value(peer, metric)
                if peer_value is not None:
                    all_values[metric].append(peer_value)

        # Calculate premium/discount for each metric
        premium_discount = {}

        for metric in valuation_metrics:
            if metric in company_values and len(all_values[metric]) > 1:
                # Calculate peer average (excluding company)
                peer_values = [v for v in all_values[metric] if v != company_values[metric]]
                peer_avg = sum(peer_values) / len(peer_values)

                # Calculate premium/discount (negative means discount)
                if peer_avg > 0:
                    premium = ((company_values[metric] / peer_avg) - 1) * 100
                    premium_discount[metric] = premium

        # Calculate composite premium/discount
        if premium_discount:
            composite_premium = sum(premium_discount.values()) / len(premium_discount)
        else:
            composite_premium = None

        # Add interpretation
        interpretation = None
        if composite_premium is not None:
            if composite_premium < -30:
                interpretation = "Significant discount to peers (potentially undervalued)"
            elif composite_premium < -10:
                interpretation = "Moderate discount to peers"
            elif composite_premium < 10:
                interpretation = "Valued in line with peers"
            elif composite_premium < 30:
                interpretation = "Moderate premium to peers"
            else:
                interpretation = "Significant premium to peers (potentially overvalued)"

        return {
            "metrics": premium_discount,
            "composite_premium": composite_premium,
            "interpretation": interpretation
        }

    def _get_metric_value(self, ticker: str, metric: str) -> Optional[float]:
        """
        Get value for a specific metric for a company

        Args:
            ticker: Company ticker symbol
            metric: Metric name

        Returns:
            Metric value or None if not available
        """
        try:
            # Get financial ratios
            financial_ratios = self.data_loader.get_financial_ratios(ticker)

            # Check in each category
            for category, ratios in financial_ratios.items():
                if isinstance(ratios, dict):
                    # Direct match
                    if metric in ratios:
                        return ratios[metric]

                    # Check various forms of the metric name
                    normalized_metric = metric.lower().replace('-', '_').replace(' ', '_')
                    for key, value in ratios.items():
                        if key.lower().replace('-', '_').replace(' ', '_') == normalized_metric:
                            return value

            # If not found in ratios, try to calculate from financial statements
            if metric == "revenue_growth" or metric == "eps_growth":
                return self._calculate_growth_metric(ticker, metric)

            return None

        except Exception as e:
            logger.warning(f"Error getting metric {metric} for {ticker}: {e}")
            return None

    def _calculate_growth_metric(self, ticker: str, metric: str) -> Optional[float]:
        """
        Calculate growth metric from financial statements

        Args:
            ticker: Company ticker symbol
            metric: Growth metric name

        Returns:
            Growth rate or None if not available
        """
        try:
            # Get income statement for the last few years
            income_stmt = self.data_loader.get_financial_statements(ticker, 'income')

            if income_stmt is None or income_stmt.empty or income_stmt.shape[1] < 2:
                return None

            if metric == "revenue_growth":
                if "Total Revenue" in income_stmt.index:
                    current = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 0]
                    previous = income_stmt.iloc[income_stmt.index.get_loc('Total Revenue'), 1]

                    if previous and previous > 0:
                        return (current / previous) - 1

            elif metric == "eps_growth":
                # Try different EPS field names
                for eps_field in ["EPS (Basic)", "EPS Basic", "Basic EPS", "EPS"]:
                    if eps_field in income_stmt.index:
                        current = income_stmt.iloc[income_stmt.index.get_loc(eps_field), 0]
                        previous = income_stmt.iloc[income_stmt.index.get_loc(eps_field), 1]

                        if previous and previous > 0:
                            return (current / previous) - 1

            return None

        except Exception as e:
            logger.warning(f"Error calculating growth metric {metric} for {ticker}: {e}")
            return None

    def _is_lower_better(self, metric: str) -> bool:
        """
        Determine if lower values are better for a metric

        Args:
            metric: Metric name

        Returns:
            True if lower values are better, False otherwise
        """
        # Valuation multiples - lower is better
        valuation_multiples = [
            "pe_ratio", "forward_pe", "peg_ratio", "ps_ratio", "pb_ratio",
            "ev_ebitda", "ev_revenue", "ev_fcf", "ev_sales"
        ]

        # Leverage/debt metrics - lower is better
        leverage_metrics = [
            "debt_to_equity", "debt_to_assets", "debt_to_ebitda",
            "net_debt_to_ebitda", "financial_leverage"
        ]

        # Efficiency metrics where lower is better
        efficiency_lower_better = [
            "days_inventory_outstanding", "days_sales_outstanding",
            "days_payable_outstanding", "cash_conversion_cycle",
            "operating_cycle"
        ]

        # Normalize metric name for comparison
        normalized = metric.lower().replace('-', '_').replace(' ', '_')

        # Check various categories
        if any(normalized.startswith(m.lower()) or normalized == m.lower() for m in valuation_multiples):
            return True
        elif any(normalized.startswith(m.lower()) or normalized == m.lower() for m in leverage_metrics):
            return True
        elif any(normalized.startswith(m.lower()) or normalized == m.lower() for m in efficiency_lower_better):
            return True

        # Default: higher is better for most metrics
        return False

    def _calculate_percentile(self, value: float, comparable_values: List[float],
                              higher_is_better: bool) -> float:
        """
        Calculate percentile rank of a value within a set of comparable values

        Args:
            value: Value to rank
            comparable_values: List of comparable values
            higher_is_better: Whether higher values are better

        Returns:
            Percentile rank (0-100)
        """
        if not comparable_values:
            return 50.0  # Default to median if no comparables

        # Add the value itself to the comparables
        all_values = comparable_values + [value]

        # Sort values based on whether higher is better
        sorted_values = sorted(all_values, reverse=higher_is_better)

        # Find position
        position = sorted_values.index(value)

        # Calculate percentile
        percentile = (len(sorted_values) - position - 1) / (len(sorted_values) - 1) * 100

        return percentile

    def _interpret_metric_comparison(self, metric: str, company_value: float,
                                     peer_avg: float, percent_diff: float,
                                     favorable: bool) -> str:
        """
        Generate interpretation for metric comparison

        Args:
            metric: Metric name
            company_value: Company's metric value
            peer_avg: Peer average value
            percent_diff: Percent difference from peer average
            favorable: Whether the difference is favorable

        Returns:
            Interpretation string
        """
        # Determine metric category
        metric_lower = metric.lower().replace('-', '_').replace(' ', '_')

        if any(term in metric_lower for term in ["pe_", "pb_", "ps_", "ev_"]):
            category = "valuation"
        elif any(term in metric_lower for term in ["margin", "roe", "roa", "roic"]):
            category = "profitability"
        elif any(term in metric_lower for term in ["debt", "leverage", "coverage"]):
            category = "leverage"
        elif any(term in metric_lower for term in ["current", "quick", "cash"]):
            category = "liquidity"
        elif any(term in metric_lower for term in ["turnover", "cycle", "days"]):
            category = "efficiency"
        elif any(term in metric_lower for term in ["growth", "increase"]):
            category = "growth"
        else:
            category = "general"

        # Generate interpretation based on category and magnitude
        abs_diff = abs(percent_diff)

        if abs_diff < 5:
            return f"In line with peer average ({peer_avg:.2f})"

        # Determine magnitude description
        if abs_diff > 50:
            magnitude = "Significantly"
        elif abs_diff > 25:
            magnitude = "Substantially"
        elif abs_diff > 10:
            magnitude = "Moderately"
        else:
            magnitude = "Slightly"

        # Valuation multiples interpretation
        if category == "valuation":
            if favorable:
                return f"{magnitude} undervalued compared to peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} overvalued compared to peers (peer avg: {peer_avg:.2f})"

        # Profitability interpretation
        elif category == "profitability":
            if favorable:
                return f"{magnitude} more profitable than peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} less profitable than peers (peer avg: {peer_avg:.2f})"

        # Leverage interpretation
        elif category == "leverage":
            if favorable:
                return f"{magnitude} lower leverage than peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} higher leverage than peers (peer avg: {peer_avg:.2f})"

        # Liquidity interpretation
        elif category == "liquidity":
            if favorable:
                return f"{magnitude} more liquid than peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} less liquid than peers (peer avg: {peer_avg:.2f})"

        # Efficiency interpretation
        elif category == "efficiency":
            if favorable:
                return f"{magnitude} more efficient than peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} less efficient than peers (peer avg: {peer_avg:.2f})"

        # Growth interpretation
        elif category == "growth":
            if favorable:
                return f"{magnitude} faster growth than peers (peer avg: {peer_avg:.2f})"
            else:
                return f"{magnitude} slower growth than peers (peer avg: {peer_avg:.2f})"

        # General interpretation
        else:
            if favorable:
                return f"{magnitude} better than peer average ({peer_avg:.2f})"
            else:
                return f"{magnitude} worse than peer average ({peer_avg:.2f})"

    def _calculate_liquidity_score(self, company_values: Dict[str, float]) -> float:
        """
        Calculate liquidity score based on liquidity ratios

        Args:
            company_values: Dictionary of company metric values

        Returns:
            Liquidity score (0-1)
        """
        score_components = []

        # Current ratio
        if "current_ratio" in company_values:
            current_ratio = company_values["current_ratio"]
            if current_ratio >= 2.0:
                score_components.append(1.0)
            elif current_ratio >= 1.5:
                score_components.append(0.8)
            elif current_ratio >= 1.0:
                score_components.append(0.6)
            elif current_ratio >= 0.8:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Quick ratio
        if "quick_ratio" in company_values:
            quick_ratio = company_values["quick_ratio"]
            if quick_ratio >= 1.5:
                score_components.append(1.0)
            elif quick_ratio >= 1.0:
                score_components.append(0.8)
            elif quick_ratio >= 0.7:
                score_components.append(0.6)
            elif quick_ratio >= 0.5:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Cash ratio
        if "cash_ratio" in company_values:
            cash_ratio = company_values["cash_ratio"]
            if cash_ratio >= 0.5:
                score_components.append(1.0)
            elif cash_ratio >= 0.3:
                score_components.append(0.8)
            elif cash_ratio >= 0.2:
                score_components.append(0.6)
            elif cash_ratio >= 0.1:
                score_components.append(0.3)
            else:
                score_components.append(0.1)

        # Return average score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.5  # Default to middle if no data

    def _calculate_solvency_score(self, company_values: Dict[str, float]) -> float:
        """
        Calculate solvency score based on leverage ratios

        Args:
            company_values: Dictionary of company metric values

        Returns:
            Solvency score (0-1)
        """
        score_components = []

        # Debt to equity
        if "debt_to_equity" in company_values:
            debt_to_equity = company_values["debt_to_equity"]
            if debt_to_equity <= 0.3:
                score_components.append(1.0)
            elif debt_to_equity <= 0.6:
                score_components.append(0.8)
            elif debt_to_equity <= 1.0:
                score_components.append(0.6)
            elif debt_to_equity <= 1.5:
                score_components.append(0.4)
            elif debt_to_equity <= 2.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Debt to EBITDA
        if "debt_to_ebitda" in company_values:
            debt_to_ebitda = company_values["debt_to_ebitda"]
            if debt_to_ebitda <= 1.0:
                score_components.append(1.0)
            elif debt_to_ebitda <= 2.0:
                score_components.append(0.8)
            elif debt_to_ebitda <= 3.0:
                score_components.append(0.6)
            elif debt_to_ebitda <= 4.0:
                score_components.append(0.4)
            elif debt_to_ebitda <= 5.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Interest coverage
        if "interest_coverage" in company_values:
            interest_coverage = company_values["interest_coverage"]
            if interest_coverage >= 8.0:
                score_components.append(1.0)
            elif interest_coverage >= 5.0:
                score_components.append(0.8)
            elif interest_coverage >= 3.0:
                score_components.append(0.6)
            elif interest_coverage >= 2.0:
                score_components.append(0.4)
            elif interest_coverage >= 1.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Return average score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.5  # Default to middle if no data

    def _calculate_profitability_score(self, company_values: Dict[str, float]) -> float:
        """
        Calculate profitability score based on profitability ratios

        Args:
            company_values: Dictionary of company metric values

        Returns:
            Profitability score (0-1)
        """
        score_components = []

        # Return on Equity (ROE)
        if "roe" in company_values:
            roe = company_values["roe"]
            if roe >= 0.20:
                score_components.append(1.0)
            elif roe >= 0.15:
                score_components.append(0.8)
            elif roe >= 0.10:
                score_components.append(0.6)
            elif roe >= 0.05:
                score_components.append(0.4)
            elif roe >= 0.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Return on Assets (ROA)
        if "roa" in company_values:
            roa = company_values["roa"]
            if roa >= 0.10:
                score_components.append(1.0)
            elif roa >= 0.08:
                score_components.append(0.8)
            elif roa >= 0.05:
                score_components.append(0.6)
            elif roa >= 0.03:
                score_components.append(0.4)
            elif roa >= 0.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Return on Invested Capital (ROIC)
        if "roic" in company_values:
            roic = company_values["roic"]
            if roic >= 0.15:
                score_components.append(1.0)
            elif roic >= 0.12:
                score_components.append(0.8)
            elif roic >= 0.09:
                score_components.append(0.6)
            elif roic >= 0.06:
                score_components.append(0.4)
            elif roic >= 0.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Operating Margin
        if "operating_margin" in company_values:
            operating_margin = company_values["operating_margin"]
            if operating_margin >= 0.25:
                score_components.append(1.0)
            elif operating_margin >= 0.15:
                score_components.append(0.8)
            elif operating_margin >= 0.10:
                score_components.append(0.6)
            elif operating_margin >= 0.05:
                score_components.append(0.4)
            elif operating_margin >= 0.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)

        # Return average score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.5  # Default to middle if no data