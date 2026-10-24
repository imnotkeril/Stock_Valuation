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
from StockAnalysisSystem.src.industry.benchmarks import IndustryBenchmarks
from StockAnalysisSystem.src.comparison.peer_finder import PeerFinder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('industry_benchmarking')


class IndustryBenchmarking:
    """
    Class for industry benchmarking and comparative analysis.
    Provides methods to compare companies against industry standards,
    sector averages, and identify industry trends.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the industry benchmarking system

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader if data_loader else DataLoader()

        # Initialize dependency components
        self.benchmarks = IndustryBenchmarks(data_loader)
        self.peer_finder = PeerFinder(data_loader)

        # Load sector configuration
        self.sector_metrics = SECTOR_METRICS
        self.sector_mapping = SECTOR_MAPPING

        logger.info("Industry benchmarking system initialized")

    def analyze_company_in_sector(self, ticker: str, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a company's position within its industry sector

        Args:
            ticker: Company ticker symbol
            sector: Optional sector override (uses company's reported sector if not provided)

        Returns:
            Dictionary with industry benchmarking analysis results
        """
        try:
            # Get company information
            company_info = self.data_loader.get_company_info(ticker)

            if not company_info:
                return {"error": f"Failed to retrieve information for {ticker}"}

            # Determine sector
            if not sector:
                sector = company_info.get('sector', '')

            if not sector:
                return {"error": f"Sector information not available for {ticker}"}

            # Normalize sector name
            sector = self._normalize_sector_name(sector)

            # Get financial ratios
            financial_ratios = self.data_loader.get_financial_ratios(ticker)

            if not financial_ratios:
                return {"error": f"Financial ratios not available for {ticker}"}

            # Get key sector metrics
            key_metrics = self.sector_metrics.get(sector, {}).get('key_ratios', [])

            # Prepare result structure
            result = {
                "ticker": ticker,
                "name": company_info.get('name', ticker),
                "sector": sector,
                "industry": company_info.get('industry', ''),
                "market_cap": company_info.get('market_cap', None),
                "sector_metrics": {},
                "benchmarking_results": {},
                "percentile_rankings": {},
                "peer_comparison": {},
                "sector_performance": {},
                "strengths_weaknesses": {
                    "strengths": [],
                    "weaknesses": [],
                    "neutral": []
                }
            }

            # Get sector benchmarks
            sector_benchmarks = self.benchmarks.get_benchmarks(sector)

            # Compare company metrics with sector benchmarks
            flat_metrics = self._flatten_metrics(financial_ratios)

            # Get benchmark comparison
            comparison_results = self.benchmarks.compare_with_benchmarks(financial_ratios, sector)
            result["benchmarking_results"] = comparison_results

            # Calculate percentile rankings for key metrics
            if key_metrics:
                percentile_rankings = {}

                for metric in key_metrics:
                    ranking = self.benchmarks.get_metric_ranking(ticker, metric)
                    if 'error' not in ranking:
                        percentile_rankings[metric] = ranking

                result["percentile_rankings"] = percentile_rankings

            # Find and compare with peers
            peers = self.peer_finder.find_peers(ticker, max_peers=5)
            peer_tickers = [peer_data['ticker'] for peer_data in peers]

            if peer_tickers:
                # Get comparison metrics for peers
                peer_comparison = self.peer_finder.get_peer_comparison_metrics(
                    ticker, peer_tickers, metric_categories=["Valuation", "Profitability", "Growth"]
                )

                # Get performance comparison with peers
                peer_performance = self.peer_finder.get_peer_performance_comparison(ticker, peer_tickers)

                result["peer_comparison"] = {
                    "metrics": peer_comparison,
                    "performance": peer_performance,
                    "peers": peers
                }

            # Get sector performance
            sector_performance = self._analyze_sector_performance(sector)
            result["sector_performance"] = sector_performance

            # Perform SWOT analysis based on benchmarking
            result["strengths_weaknesses"] = self._identify_strengths_weaknesses(comparison_results)

            # Get sector positioning matrix
            result["sector_positioning"] = self._create_sector_positioning(ticker, sector)

            return result

        except Exception as e:
            logger.error(f"Error in industry benchmarking analysis for {ticker}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def get_sector_report(self, sector: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report on a sector including trends, leaders, and metrics

        Args:
            sector: Market sector name

        Returns:
            Dictionary with sector analysis report
        """
        try:
            # Normalize sector name
            normalized_sector = self._normalize_sector_name(sector)

            if not normalized_sector:
                return {"error": f"Unrecognized sector: {sector}"}

            # Prepare report structure
            report = {
                "sector": normalized_sector,
                "as_of_date": datetime.now().strftime('%Y-%m-%d'),
                "summary": {},
                "key_metrics": {},
                "top_companies": {},
                "performance": {},
                "trends": {},
                "sub_industries": []
            }

            # Get key metrics for the sector
            key_metrics = self.sector_metrics.get(normalized_sector, {}).get('key_ratios', [])
            report["key_metrics"]["metrics"] = key_metrics

            # Get sector benchmarks
            sector_benchmarks = self.benchmarks.get_benchmarks(normalized_sector)
            report["key_metrics"]["benchmarks"] = sector_benchmarks

            # Get sector performance
            sector_performance = self._analyze_sector_performance(normalized_sector)
            report["performance"] = sector_performance

            # Get top companies in the sector
            top_companies = self._get_sector_leaders(normalized_sector)
            report["top_companies"] = top_companies

            # Get sub-industries in this sector
            sub_industries = self.sector_mapping.get(normalized_sector, [])

            if sub_industries:
                industry_data = []

                for industry in sub_industries:
                    # Get companies in this sub-industry
                    industry_companies = self._get_companies_in_industry(normalized_sector, industry)

                    if industry_companies:
                        # Calculate industry averages
                        industry_metrics = self._calculate_industry_averages(industry_companies, key_metrics)

                        industry_data.append({
                            "name": industry,
                            "companies_count": len(industry_companies),
                            "metrics": industry_metrics,
                            "top_companies": industry_companies[:3] if len(
                                industry_companies) > 3 else industry_companies
                        })

                report["sub_industries"] = industry_data

            # Generate sector summary
            report["summary"] = self._generate_sector_summary(normalized_sector, sector_performance, top_companies)

            # Identify sector trends
            report["trends"] = self._identify_sector_trends(normalized_sector)

            return report

        except Exception as e:
            logger.error(f"Error generating sector report for {sector}: {e}")
            return {"error": f"Failed to generate sector report: {str(e)}"}

    def compare_sectors(self, sectors: List[str], metrics: List[str]) -> Dict[str, Any]:
        """
        Compare different sectors based on specified metrics

        Args:
            sectors: List of sector names to compare
            metrics: List of metrics for comparison

        Returns:
            Dictionary with sector comparison results
        """
        try:
            # Normalize sector names
            normalized_sectors = [self._normalize_sector_name(sector) for sector in sectors]
            normalized_sectors = [s for s in normalized_sectors if s]

            if not normalized_sectors:
                return {"error": "No valid sectors provided for comparison"}

            # Prepare result structure
            result = {
                "sectors": normalized_sectors,
                "metrics": metrics,
                "comparison_data": {},
                "performance": {},
                "valuations": {}
            }

            # Compare metrics across sectors
            metric_comparison = {}

            for metric in metrics:
                metric_values = {}

                for sector in normalized_sectors:
                    # Get sector average for this metric
                    sector_average = self._get_sector_metric_average(sector, metric)
                    metric_values[sector] = sector_average

                metric_comparison[metric] = metric_values

            result["comparison_data"] = metric_comparison

            # Compare sector performance
            sector_performance = {}

            for period in ['1m', '3m', '6m', '1y']:
                period_perf = {}

                for sector in normalized_sectors:
                    # Get sector performance for this period
                    performance = self._get_sector_performance(sector, period)
                    period_perf[sector] = performance

                sector_performance[period] = period_perf

            result["performance"] = sector_performance

            # Compare sector valuations
            valuation_metrics = [m for m in metrics if m in ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda']]

            if valuation_metrics:
                historical_valuations = {}

                for metric in valuation_metrics:
                    metric_history = {}

                    for sector in normalized_sectors:
                        # Get historical valuation data for this sector and metric
                        history = self._get_sector_valuation_history(sector, metric)
                        metric_history[sector] = history

                    historical_valuations[metric] = metric_history

                result["valuations"] = historical_valuations

            return result

        except Exception as e:
            logger.error(f"Error comparing sectors: {e}")
            return {"error": f"Sector comparison failed: {str(e)}"}

    def analyze_industry_trends(self, sector: str, period: str = '5y') -> Dict[str, Any]:
        """
        Analyze trends within an industry sector over time

        Args:
            sector: Market sector name
            period: Time period for analysis ('1y', '3y', '5y', '10y')

        Returns:
            Dictionary with industry trend analysis
        """
        try:
            # Normalize sector name
            normalized_sector = self._normalize_sector_name(sector)

            if not normalized_sector:
                return {"error": f"Unrecognized sector: {sector}"}

            # Prepare result structure
            result = {
                "sector": normalized_sector,
                "period": period,
                "performance_trend": {},
                "valuation_trend": {},
                "growth_trend": {},
                "profitability_trend": {},
                "market_share_trend": {}
            }

            # Calculate end date (current date)
            end_date = datetime.now()

            # Calculate start date based on period
            if period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '3y':
                start_date = end_date - timedelta(days=3 * 365)
            elif period == '10y':
                start_date = end_date - timedelta(days=10 * 365)
            else:  # Default to 5y
                start_date = end_date - timedelta(days=5 * 365)

            # Get quarterly data points for trends
            quarters = []
            current_date = start_date
            while current_date <= end_date:
                quarters.append(current_date)
                # Add approximately 3 months
                current_date = current_date + timedelta(days=91)

            # Analyze sector performance trend
            performance_trend = self._analyze_sector_performance_trend(normalized_sector, quarters)
            result["performance_trend"] = performance_trend

            # Analyze valuation trends
            valuation_metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda']
            valuation_trend = self._analyze_sector_valuation_trend(normalized_sector, valuation_metrics, quarters)
            result["valuation_trend"] = valuation_trend

            # Analyze growth trends
            growth_trend = self._analyze_sector_growth_trend(normalized_sector, quarters)
            result["growth_trend"] = growth_trend

            # Analyze profitability trends
            profitability_metrics = ['operating_margin', 'net_margin', 'roe', 'roa']
            profitability_trend = self._analyze_sector_metric_trend(
                normalized_sector, profitability_metrics, quarters
            )
            result["profitability_trend"] = profitability_trend

            # Analyze market share trends among top companies
            market_share_trend = self._analyze_market_share_trend(normalized_sector, quarters)
            result["market_share_trend"] = market_share_trend

            return result

        except Exception as e:
            logger.error(f"Error analyzing industry trends for {sector}: {e}")
            return {"error": f"Industry trend analysis failed: {str(e)}"}

    def _normalize_sector_name(self, sector: str) -> str:
        """Normalize sector name to match configuration"""
        # Delegate to benchmarks component
        return self.benchmarks._normalize_sector_name(sector)

    def _flatten_metrics(self, categorized_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Flatten categorized metrics into a single dictionary"""
        flat_metrics = {}

        for category, metrics in categorized_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    flat_metrics[metric_name] = metric_value

        return flat_metrics

    def _analyze_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Analyze sector performance over various time periods"""
        try:
            # Get sector performance data
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
            if not etf_ticker:
                return {}

            # Get historical prices for the sector ETF
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date_5y = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            sector_data = self.data_loader.get_historical_prices(etf_ticker, start_date_5y, end_date)

            if sector_data.empty:
                return {}

            # Calculate performance for different periods
            performance = {}

            # Periods to calculate
            periods = {
                "1m": 30,
                "3m": 90,
                "6m": 180,
                "1y": 365,
                "3y": 3 * 365,
                "5y": 5 * 365
            }

            for period_name, days in periods.items():
                if len(sector_data) > days / 2:  # Ensure we have enough data
                    # Get the date that's approximately 'days' days ago
                    target_date = (datetime.now() - timedelta(days=days))

                    # Find the closest data point
                    closest_index = sector_data.index[sector_data.index <= pd.Timestamp(target_date)]

                    if len(closest_index) > 0:
                        start_price = sector_data.loc[closest_index[-1], 'Close']
                        end_price = sector_data['Close'].iloc[-1]

                        period_return = (end_price / start_price) - 1
                        performance[period_name] = period_return

            # Calculate rolling volatility (annualized)
            returns = sector_data['Close'].pct_change().dropna()
            volatility_annual = returns.std() * np.sqrt(252)

            # Get S&P 500 for comparison
            sp500_data = self.data_loader.get_historical_prices("SPY", start_date_5y, end_date)

            relative_performance = {}
            sp500_volatility = None

            if not sp500_data.empty:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                sp500_volatility = sp500_returns.std() * np.sqrt(252)

                # Calculate beta
                if len(returns) > 30 and len(sp500_returns) > 30:
                    # Align dates
                    combined = pd.DataFrame({
                        'sector': returns,
                        'market': sp500_returns
                    }).dropna()

                    if len(combined) > 30:
                        # Calculate beta
                        beta = combined.cov().iloc[0, 1] / combined['market'].var()

                        # Calculate relative performance vs S&P 500
                        for period_name, sector_return in performance.items():
                            if period_name in periods:
                                days = periods[period_name]

                                # Find corresponding market return
                                target_date = (datetime.now() - timedelta(days=days))
                                closest_index = sp500_data.index[sp500_data.index <= pd.Timestamp(target_date)]

                                if len(closest_index) > 0:
                                    market_start_price = sp500_data.loc[closest_index[-1], 'Close']
                                    market_end_price = sp500_data['Close'].iloc[-1]

                                    market_return = (market_end_price / market_start_price) - 1
                                    relative_return = sector_return - market_return

                                    relative_performance[period_name] = relative_return

            return {
                "etf_ticker": etf_ticker,
                "performance": performance,
                "volatility": float(volatility_annual) if not pd.isna(volatility_annual) else None,
                "beta": float(beta) if 'beta' in locals() and not pd.isna(beta) else None,
                "relative_to_sp500": relative_performance,
                "sp500_volatility": float(sp500_volatility) if sp500_volatility is not None and not pd.isna(
                    sp500_volatility) else None
            }

        except Exception as e:
            logger.error(f"Error analyzing sector performance for {sector}: {e}")
            return {}

    def _identify_strengths_weaknesses(self, comparison_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[
        str, List[Dict[str, Any]]]:
        """Identify company strengths and weaknesses based on benchmark comparisons"""
        strengths = []
        weaknesses = []
        neutral = []

        for category, metrics in comparison_results.items():
            for metric_name, comparison in metrics.items():
                if 'assessment' in comparison:
                    metric_info = {
                        "metric": metric_name,
                        "category": category,
                        "value": comparison['company_value'],
                        "benchmark": comparison['benchmark_value'],
                        "percent_diff": comparison['percent_diff']
                    }

                    if comparison['assessment'] in ['excellent', 'good']:
                        strengths.append(metric_info)
                    elif comparison['assessment'] in ['below_average', 'poor']:
                        weaknesses.append(metric_info)
                    else:
                        neutral.append(metric_info)

        # Sort strengths and weaknesses by magnitude of difference
        strengths.sort(key=lambda x: abs(x['percent_diff']) if 'percent_diff' in x else 0, reverse=True)
        weaknesses.sort(key=lambda x: abs(x['percent_diff']) if 'percent_diff' in x else 0, reverse=True)

        return {
            "strengths": strengths[:5],  # Top 5 strengths
            "weaknesses": weaknesses[:5],  # Top 5 weaknesses
            "neutral": neutral
        }

    def _create_sector_positioning(self, ticker: str, sector: str) -> Dict[str, Any]:
        """Create sector positioning matrix for the company"""
        # Get company financial data
        financial_ratios = self.data_loader.get_financial_ratios(ticker)
        flat_ratios = self._flatten_metrics(financial_ratios)

        # Select metrics based on sector
        if sector in ["Technology", "Healthcare", "Communication Services"]:
            x_metric = "operating_margin"
            y_metric = "revenue_growth"
        elif sector in ["Financials", "Real Estate"]:
            x_metric = "roe"
            y_metric = "debt_to_equity"
        elif sector in ["Energy", "Materials", "Utilities"]:
            x_metric = "roa"
            y_metric = "dividend_yield"
        else:  # Consumer, Industrials
            x_metric = "net_margin"
            y_metric = "asset_turnover"

        # If the selected metrics aren't available, fall back to basic metrics
        if x_metric not in flat_ratios:
            x_metric = "roe"
        if y_metric not in flat_ratios:
            y_metric = "operating_margin"

        # Create positioning matrix
        matrix = self.benchmarks.get_sector_matrix(sector, x_metric, y_metric)

        return matrix

    def _get_sector_leaders(self, sector: str) -> Dict[str, List[Dict[str, Any]]]:
        """Identify leading companies in the sector by various metrics"""
        try:
            # Get companies in the sector
            sector_companies = self._get_companies_in_sector(sector)

            if not sector_companies:
                return {}

            # Define categories and metrics
            categories = {
                "market_cap": "Largest Companies",
                "revenue_growth": "Fastest Growing",
                "operating_margin": "Most Profitable",
                "roe": "Highest ROE",
                "dividend_yield": "Highest Dividend Yield"
            }

            # Collect and sort companies by each metric
            leaders = {}

            for metric, title in categories.items():
                companies_with_metric = []

                for ticker in sector_companies:
                    # Get company data
                    company_info = self.data_loader.get_company_info(ticker)
                    financial_ratios = self.data_loader.get_financial_ratios(ticker)
                    flat_ratios = self._flatten_metrics(financial_ratios)

                    if company_info:
                        metric_value = None

                        # Get metric value
                        if metric == "market_cap":
                            metric_value = company_info.get('market_cap')
                        else:
                            metric_value = flat_ratios.get(metric)

                        if metric_value is not None:
                            companies_with_metric.append({
                                "ticker": ticker,
                                "name": company_info.get('name', ticker),
                                "value": metric_value
                            })

                # Sort by metric value (descending)
                companies_with_metric.sort(key=lambda x: x['value'], reverse=True)

                # Keep top 5
                leaders[title] = companies_with_metric[:5] if len(companies_with_metric) > 5 else companies_with_metric

            return leaders

        except Exception as e:
            logger.error(f"Error getting sector leaders for {sector}: {e}")
            return {}

    def _get_companies_in_sector(self, sector: str) -> List[str]:
        """Get list of companies in a specific sector"""
        # This implementation is simplified for the example
        # In a real system, you would use a database or API
        sector_companies = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "NVDA", "ADBE", "CRM", "INTC", "CSCO", "ORCL", "IBM",
                           "AMD"],
            "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "TMO", "ABT", "MDT", "AMGN", "BMY", "CVS", "CI"],
            "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA", "SCHW", "CB", "PNC"],
            "Consumer Discretionary": ["AMZN", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "TJX", "BKNG", "EBAY", "BBY",
                                       "DG"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "EL", "GIS", "K", "KHC", "KMB", "MO", "PM"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "PSX", "VLO", "SLB", "OXY", "MPC", "KMI", "WMB", "HES", "HAL"],
            "Industrials": ["HON", "UNP", "UPS", "BA", "CAT", "GE", "MMM", "LMT", "RTX", "DE", "FDX", "GD", "CSX"],
            "Materials": ["LIN", "APD", "ECL", "SHW", "DD", "NEM", "NUE", "FCX", "DOW", "PPG", "IP", "BLL", "VMC"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "AVB", "EQR", "DLR", "O", "SPG", "WELL", "VTR", "ESS"],
            "Communication Services": ["GOOGL", "FB", "NFLX", "TMUS", "CMCSA", "VZ", "T", "CHTR", "ATVI", "EA", "TTWO",
                                       "DIS"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "ED", "PEG", "ES", "WEC", "AEE"]
        }

        return sector_companies.get(sector, [])

    def _get_companies_in_industry(self, sector: str, industry: str) -> List[str]:
        """Get list of companies in a specific industry within a sector"""
        # This is a simplified implementation
        # In a real system, you would use a database or API

        # Get all companies in the sector
        sector_companies = self._get_companies_in_sector(sector)
        industry_companies = []

        # Filter by industry
        for ticker in sector_companies:
            company_info = self.data_loader.get_company_info(ticker)
            if company_info and company_info.get('industry', '') == industry:
                industry_companies.append(ticker)

        return industry_companies

    def _calculate_industry_averages(self, tickers: List[str], metrics: List[str]) -> Dict[str, float]:
        """Calculate average metrics for a list of companies"""
        if not tickers or not metrics:
            return {}

        # Collect metric values
        metric_values = {metric: [] for metric in metrics}

        for ticker in tickers:
            financial_ratios = self.data_loader.get_financial_ratios(ticker)
            flat_ratios = self._flatten_metrics(financial_ratios)

            for metric in metrics:
                if metric in flat_ratios and flat_ratios[metric] is not None:
                    metric_values[metric].append(flat_ratios[metric])

        # Calculate averages
        averages = {}

        for metric, values in metric_values.items():
            if values:
                averages[metric] = sum(values) / len(values)

        return averages

    def _generate_sector_summary(self, sector: str, performance: Dict[str, Any],
                                 leaders: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of sector performance and characteristics"""
        summary = {
            "sector": sector,
            "performance_summary": {},
            "leaders_summary": {},
            "characteristics": {}
        }

        # Performance summary
        if performance:
            perf_data = performance.get('performance', {})

            # Current year performance
            ytd_return = perf_data.get('1y')

            if ytd_return is not None:
                summary['performance_summary']['ytd_return'] = ytd_return

                # Classify performance
                if ytd_return > 0.2:
                    performance_classification = "Strong growth"
                elif ytd_return > 0.05:
                    performance_classification = "Moderate growth"
                elif ytd_return > -0.05:
                    performance_classification = "Stable"
                elif ytd_return > -0.2:
                    performance_classification = "Moderate decline"
                else:
                    performance_classification = "Significant decline"

                summary['performance_summary']['classification'] = performance_classification

            # Volatility
            volatility = performance.get('volatility')
            if volatility is not None:
                summary['performance_summary']['volatility'] = volatility

                # Classify volatility
                if volatility < 0.15:
                    volatility_classification = "Low volatility"
                elif volatility < 0.25:
                    volatility_classification = "Moderate volatility"
                else:
                    volatility_classification = "High volatility"

                summary['performance_summary']['volatility_classification'] = volatility_classification

            # Beta
            beta = performance.get('beta')
            if beta is not None:
                summary['performance_summary']['beta'] = beta

        # Leaders summary
        if leaders:
            # Get largest companies
            largest_companies = leaders.get('Largest Companies', [])
            if largest_companies:
                summary['leaders_summary']['largest_companies'] = [
                    company['ticker'] for company in largest_companies[:3]
                ]

            # Get fastest growing
            fastest_growing = leaders.get('Fastest Growing', [])
            if fastest_growing:
                summary['leaders_summary']['fastest_growing'] = [
                    company['ticker'] for company in fastest_growing[:3]
                ]

            # Get most profitable
            most_profitable = leaders.get('Most Profitable', [])
            if most_profitable:
                summary['leaders_summary']['most_profitable'] = [
                    company['ticker'] for company in most_profitable[:3]
                ]

        # Sector characteristics
        # These would normally come from a database or configuration
        sector_characteristics = {
            "Technology": {
                "growth_rate": "High",
                "cyclicality": "Moderate",
                "capital_intensity": "Low",
                "key_success_factors": ["Innovation", "Intellectual property", "Network effects"],
                "disruption_risk": "High"
            },
            "Healthcare": {
                "growth_rate": "Moderate",
                "cyclicality": "Low",
                "capital_intensity": "Moderate",
                "key_success_factors": ["Research & development", "Regulatory approval", "Patent protection"],
                "disruption_risk": "Moderate"
            },
            "Financials": {
                "growth_rate": "Low",
                "cyclicality": "High",
                "capital_intensity": "Low",
                "key_success_factors": ["Risk management", "Customer relationships", "Regulatory compliance"],
                "disruption_risk": "Moderate"
            },
            "Consumer Discretionary": {
                "growth_rate": "Moderate",
                "cyclicality": "High",
                "capital_intensity": "Moderate",
                "key_success_factors": ["Brand strength", "Customer experience", "Supply chain efficiency"],
                "disruption_risk": "High"
            },
            "Consumer Staples": {
                "growth_rate": "Low",
                "cyclicality": "Low",
                "capital_intensity": "Moderate",
                "key_success_factors": ["Brand loyalty", "Distribution network", "Cost control"],
                "disruption_risk": "Low"
            },
            "Energy": {
                "growth_rate": "Low",
                "cyclicality": "High",
                "capital_intensity": "High",
                "key_success_factors": ["Resource access", "Operational efficiency", "Scale"],
                "disruption_risk": "High"
            },
            "Industrials": {
                "growth_rate": "Moderate",
                "cyclicality": "High",
                "capital_intensity": "High",
                "key_success_factors": ["Operational efficiency", "Global presence", "Innovation"],
                "disruption_risk": "Moderate"
            },
            "Materials": {
                "growth_rate": "Low",
                "cyclicality": "High",
                "capital_intensity": "High",
                "key_success_factors": ["Cost leadership", "Vertical integration", "Scale"],
                "disruption_risk": "Low"
            },
            "Real Estate": {
                "growth_rate": "Low",
                "cyclicality": "Moderate",
                "capital_intensity": "High",
                "key_success_factors": ["Location", "Property management", "Capital allocation"],
                "disruption_risk": "Moderate"
            },
            "Communication Services": {
                "growth_rate": "Moderate",
                "cyclicality": "Moderate",
                "capital_intensity": "High",
                "key_success_factors": ["Network infrastructure", "Content", "Customer acquisition"],
                "disruption_risk": "High"
            },
            "Utilities": {
                "growth_rate": "Low",
                "cyclicality": "Low",
                "capital_intensity": "Very high",
                "key_success_factors": ["Regulatory relationships", "Operational efficiency", "Capital allocation"],
                "disruption_risk": "Moderate"
            }
        }

        if sector in sector_characteristics:
            summary['characteristics'] = sector_characteristics[sector]

        return summary

    def _identify_sector_trends(self, sector: str) -> Dict[str, Any]:
        """Identify trends within a sector based on historical data"""
        # This would normally use historical data to identify trends
        # For this example, we'll use predefined trends

        sector_trends = {
            "Technology": {
                "emerging_themes": ["AI/Machine Learning", "Cloud Computing", "Cybersecurity", "Remote Work Solutions"],
                "consolidation_trend": "High",
                "innovation_pace": "Very High",
                "regulatory_pressure": "Increasing"
            },
            "Healthcare": {
                "emerging_themes": ["Telemedicine", "Precision Medicine", "Digital Health", "Biosimilars"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "High",
                "regulatory_pressure": "High"
            },
            "Financials": {
                "emerging_themes": ["Fintech Integration", "Digital Banking", "ESG Investing", "Blockchain"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "Moderate",
                "regulatory_pressure": "High"
            },
            "Consumer Discretionary": {
                "emerging_themes": ["E-commerce", "Direct-to-Consumer", "Experiential Retail", "Subscription Models"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "High",
                "regulatory_pressure": "Low"
            },
            "Consumer Staples": {
                "emerging_themes": ["Organic/Natural Products", "Private Label Growth", "Direct-to-Consumer",
                                    "Sustainable Packaging"],
                "consolidation_trend": "High",
                "innovation_pace": "Moderate",
                "regulatory_pressure": "Moderate"
            },
            "Energy": {
                "emerging_themes": ["Renewable Energy", "Battery Storage", "Carbon Capture", "ESG Integration"],
                "consolidation_trend": "High",
                "innovation_pace": "Moderate",
                "regulatory_pressure": "Increasing"
            },
            "Industrials": {
                "emerging_themes": ["Automation", "IoT/Smart Manufacturing", "Electrification",
                                    "Supply Chain Resilience"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "Moderate",
                "regulatory_pressure": "Moderate"
            },
            "Materials": {
                "emerging_themes": ["Sustainable Materials", "Recycling Technologies", "Lightweight Materials",
                                    "Specialty Chemicals"],
                "consolidation_trend": "High",
                "innovation_pace": "Moderate",
                "regulatory_pressure": "Increasing"
            },
            "Real Estate": {
                "emerging_themes": ["Flexible Workspaces", "Industrial/Logistics Growth", "PropTech",
                                    "ESG Building Standards"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "Low",
                "regulatory_pressure": "Moderate"
            },
            "Communication Services": {
                "emerging_themes": ["5G Networks", "Streaming Content", "Digital Advertising", "Gaming/eSports"],
                "consolidation_trend": "High",
                "innovation_pace": "High",
                "regulatory_pressure": "Increasing"
            },
            "Utilities": {
                "emerging_themes": ["Renewable Energy Integration", "Smart Grid", "Energy Storage",
                                    "Electric Vehicle Infrastructure"],
                "consolidation_trend": "Moderate",
                "innovation_pace": "Low",
                "regulatory_pressure": "High"
            }
        }

        return sector_trends.get(sector, {})

    def _get_sector_metric_average(self, sector: str, metric: str) -> Optional[float]:
        """Get average value for a specific metric across a sector"""
        try:
            # Get sector companies
            sector_companies = self._get_companies_in_sector(sector)

            if not sector_companies:
                return None

            # Collect metric values
            values = []

            for ticker in sector_companies:
                # Get financial ratios
                financial_ratios = self.data_loader.get_financial_ratios(ticker)
                flat_ratios = self._flatten_metrics(financial_ratios)

                # Check if metric exists
                if metric in flat_ratios and flat_ratios[metric] is not None:
                    values.append(flat_ratios[metric])

            # Calculate average
            if values:
                # Remove outliers (values beyond 3 standard deviations)
                if len(values) > 5:
                    mean = np.mean(values)
                    std = np.std(values)
                    filtered_values = [v for v in values if abs(v - mean) <= 3 * std]
                else:
                    filtered_values = values

                return sum(filtered_values) / len(filtered_values)

            return None

        except Exception as e:
            logger.error(f"Error calculating sector average for {sector}, {metric}: {e}")
            return None

    def _get_sector_performance(self, sector: str, period: str = '1y') -> Dict[str, float]:
        """Get sector performance data for a specific period"""
        try:
            # Get sector ETF
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
            if not etf_ticker:
                return {}

            # Get historical prices
            end_date = datetime.now().strftime('%Y-%m-%d')

            if period == '1m':
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == '3m':
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif period == '6m':
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            else:  # Default to 1y
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            sector_data = self.data_loader.get_historical_prices(etf_ticker, start_date, end_date)

            if sector_data.empty:
                return {}

            # Calculate performance metrics
            start_price = sector_data['Close'].iloc[0]
            end_price = sector_data['Close'].iloc[-1]

            total_return = (end_price / start_price) - 1

            # Calculate volatility
            returns = sector_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Get S&P 500 for comparison
            sp500_data = self.data_loader.get_historical_prices("SPY", start_date, end_date)

            relative_return = None

            if not sp500_data.empty:
                sp500_start = sp500_data['Close'].iloc[0]
                sp500_end = sp500_data['Close'].iloc[-1]

                sp500_return = (sp500_end / sp500_start) - 1
                relative_return = total_return - sp500_return

            return {
                "total_return": total_return,
                "volatility": volatility,
                "relative_to_sp500": relative_return
            }

        except Exception as e:
            logger.error(f"Error getting sector performance for {sector}, {period}: {e}")
            return {}

    def _get_sector_valuation_history(self, sector: str, metric: str) -> Dict[str, float]:
        """Get historical valuation data for a sector"""
        # This would normally fetch historical valuation data from a database
        # For simplicity, we'll return representative values

        # Sample data structure: {date: value}
        if metric == 'pe_ratio':
            return {
                "2015-01-01": 18.5,
                "2016-01-01": 20.2,
                "2017-01-01": 22.1,
                "2018-01-01": 24.3,
                "2019-01-01": 21.7,
                "2020-01-01": 25.8,
                "2021-01-01": 28.5,
                "2022-01-01": 26.9,
                "2023-01-01": 24.2,
                "2024-01-01": 25.1
            }
        elif metric == 'pb_ratio':
            return {
                "2015-01-01": 2.8,
                "2016-01-01": 3.1,
                "2017-01-01": 3.4,
                "2018-01-01": 3.6,
                "2019-01-01": 3.2,
                "2020-01-01": 3.7,
                "2021-01-01": 4.2,
                "2022-01-01": 3.9,
                "2023-01-01": 3.5,
                "2024-01-01": 3.6
            }
        elif metric == 'ps_ratio':
            return {
                "2015-01-01": 1.6,
                "2016-01-01": 1.8,
                "2017-01-01": 2.0,
                "2018-01-01": 2.1,
                "2019-01-01": 1.9,
                "2020-01-01": 2.3,
                "2021-01-01": 2.6,
                "2022-01-01": 2.4,
                "2023-01-01": 2.2,
                "2024-01-01": 2.3
            }
        elif metric == 'ev_ebitda':
            return {
                "2015-01-01": 9.2,
                "2016-01-01": 10.1,
                "2017-01-01": 11.3,
                "2018-01-01": 12.1,
                "2019-01-01": 10.8,
                "2020-01-01": 12.9,
                "2021-01-01": 14.2,
                "2022-01-01": 13.4,
                "2023-01-01": 12.1,
                "2024-01-01": 12.5
            }
        else:
            return {}

    def _analyze_sector_performance_trend(self, sector: str, quarters: List[datetime]) -> Dict[
        str, List[Dict[str, Any]]]:
        """Analyze sector performance trend over specified quarters"""
        try:
            # Get sector ETF
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
            if not etf_ticker:
                return {}

            # Get full historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = quarters[0].strftime('%Y-%m-%d')

            sector_data = self.data_loader.get_historical_prices(etf_ticker, start_date, end_date)
            sp500_data = self.data_loader.get_historical_prices("SPY", start_date, end_date)

            if sector_data.empty:
                return {}

            # Calculate performance for each quarter
            performance_trend = []

            for i in range(len(quarters) - 1):
                start_quarter = quarters[i]
                end_quarter = quarters[i + 1]

                # Find closest data points
                start_idx = sector_data.index[sector_data.index >= pd.Timestamp(start_quarter)]
                end_idx = sector_data.index[sector_data.index >= pd.Timestamp(end_quarter)]

                if len(start_idx) > 0 and len(end_idx) > 0:
                    sector_start_price = sector_data.loc[start_idx[0], 'Close']
                    sector_end_price = sector_data.loc[end_idx[0], 'Close']

                    sector_return = (sector_end_price / sector_start_price) - 1

                    # Get S&P 500 return for the same period
                    sp500_return = None
                    if not sp500_data.empty:
                        sp500_start_idx = sp500_data.index[sp500_data.index >= pd.Timestamp(start_quarter)]
                        sp500_end_idx = sp500_data.index[sp500_data.index >= pd.Timestamp(end_quarter)]

                        if len(sp500_start_idx) > 0 and len(sp500_end_idx) > 0:
                            sp500_start_price = sp500_data.loc[sp500_start_idx[0], 'Close']
                            sp500_end_price = sp500_data.loc[sp500_end_idx[0], 'Close']

                            sp500_return = (sp500_end_price / sp500_start_price) - 1

                    quarter_data = {
                        "period_start": start_quarter.strftime('%Y-%m-%d'),
                        "period_end": end_quarter.strftime('%Y-%m-%d'),
                        "return": sector_return,
                        "sp500_return": sp500_return,
                        "relative_return": sector_return - sp500_return if sp500_return is not None else None
                    }

                    performance_trend.append(quarter_data)

            return {"quarterly_performance": performance_trend}

        except Exception as e:
            logger.error(f"Error analyzing sector performance trend for {sector}: {e}")
            return {}

    def _analyze_sector_valuation_trend(self, sector: str, metrics: List[str],
                                        quarters: List[datetime]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Analyze sector valuation metrics trend over specified quarters"""
        # This would normally query a database for historical valuation data
        # For this example, we'll generate representative data

        result = {}

        for metric in metrics:
            metric_trend = []

            # Get annual history (simplified)
            annual_history = self._get_sector_valuation_history(sector, metric)

            # Convert to quarterly data (interpolation)
            for quarter in quarters:
                year = quarter.year
                quarter_num = (quarter.month - 1) // 3 + 1

                # Get closest annual data points
                year_key = f"{year}-01-01"
                prev_year_key = f"{year - 1}-01-01"

                if year_key in annual_history:
                    current_value = annual_history[year_key]
                    prev_value = annual_history.get(prev_year_key, current_value)

                    # Simple linear interpolation
                    interpolated_value = prev_value + (current_value - prev_value) * (quarter_num / 4)

                    metric_trend.append({
                        "date": quarter.strftime('%Y-%m-%d'),
                        "value": interpolated_value
                    })

            if metric_trend:
                result[metric] = metric_trend

        return result

    def _analyze_sector_growth_trend(self, sector: str, quarters: List[datetime]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze growth trends for a sector over time"""
        # This would normally use historical financial data
        # For simplicity, we'll use representative data

        # Define trend patterns based on sector
        if sector in ["Technology", "Healthcare", "Communication Services"]:
            base_growth = 0.15
            volatility = 0.05
        elif sector in ["Consumer Discretionary", "Industrials"]:
            base_growth = 0.08
            volatility = 0.07
        elif sector in ["Financials", "Materials", "Real Estate"]:
            base_growth = 0.06
            volatility = 0.08
        else:  # Utilities, Energy, Consumer Staples
            base_growth = 0.04
            volatility = 0.03

        # Generate trend data
        revenue_growth = []
        earnings_growth = []

        for i, quarter in enumerate(quarters):
            # Add cyclicality and trend
            cycle_effect = 0.03 * np.sin(i * np.pi / 4)  # ~2-year cycle
            trend_effect = -0.0015 * i  # Gradually slowing growth

            # Random noise
            revenue_noise = np.random.normal(0, volatility * 0.5)
            earnings_noise = np.random.normal(0, volatility)

            # Calculate growth rates
            rev_growth = base_growth + cycle_effect + trend_effect + revenue_noise
            earn_growth = base_growth * 1.2 + cycle_effect * 1.5 + trend_effect + earnings_noise

            revenue_growth.append({
                "date": quarter.strftime('%Y-%m-%d'),
                "value": rev_growth
            })

            earnings_growth.append({
                "date": quarter.strftime('%Y-%m-%d'),
                "value": earn_growth
            })

        return {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth
        }

    def _analyze_sector_metric_trend(self, sector: str, metrics: List[str],
                                     quarters: List[datetime]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze trends for specified metrics over time"""
        # This would normally use historical financial data
        # For simplicity, we'll generate representative data

        result = {}

        # Define baseline values by sector
        if sector == "Technology":
            baselines = {
                "operating_margin": 0.25,
                "net_margin": 0.18,
                "roe": 0.22,
                "roa": 0.12
            }
        elif sector == "Healthcare":
            baselines = {
                "operating_margin": 0.20,
                "net_margin": 0.15,
                "roe": 0.18,
                "roa": 0.10
            }
        elif sector == "Financials":
            baselines = {
                "operating_margin": 0.35,
                "net_margin": 0.25,
                "roe": 0.12,
                "roa": 0.01
            }
        elif sector == "Consumer Discretionary":
            baselines = {
                "operating_margin": 0.12,
                "net_margin": 0.08,
                "roe": 0.15,
                "roa": 0.08
            }
        elif sector == "Consumer Staples":
            baselines = {
                "operating_margin": 0.15,
                "net_margin": 0.10,
                "roe": 0.18,
                "roa": 0.09
            }
        elif sector == "Energy":
            baselines = {
                "operating_margin": 0.15,
                "net_margin": 0.10,
                "roe": 0.10,
                "roa": 0.06
            }
        elif sector == "Industrials":
            baselines = {
                "operating_margin": 0.12,
                "net_margin": 0.09,
                "roe": 0.14,
                "roa": 0.07
            }
        elif sector == "Materials":
            baselines = {
                "operating_margin": 0.12,
                "net_margin": 0.08,
                "roe": 0.12,
                "roa": 0.06
            }
        elif sector == "Real Estate":
            baselines = {
                "operating_margin": 0.40,
                "net_margin": 0.25,
                "roe": 0.08,
                "roa": 0.04
            }
        elif sector == "Communication Services":
            baselines = {
                "operating_margin": 0.25,
                "net_margin": 0.15,
                "roe": 0.16,
                "roa": 0.08
            }
        elif sector == "Utilities":
            baselines = {
                "operating_margin": 0.22,
                "net_margin": 0.12,
                "roe": 0.10,
                "roa": 0.04
            }
        else:
            baselines = {
                "operating_margin": 0.15,
                "net_margin": 0.10,
                "roe": 0.15,
                "roa": 0.08
            }

        # Generate trend data for each metric
        for metric in metrics:
            if metric in baselines:
                baseline = baselines[metric]
                volatility = baseline * 0.15  # Volatility proportional to baseline

                metric_trend = []

                for i, quarter in enumerate(quarters):
                    # Add cyclicality and trend
                    cycle_effect = baseline * 0.1 * np.sin(i * np.pi / 4)  # ~2-year cycle

                    # Sector-specific trends
                    if sector in ["Technology", "Healthcare"] and metric in ["operating_margin", "roe"]:
                        trend_effect = baseline * 0.005 * i  # Improving margins over time
                    elif sector in ["Energy", "Materials"] and i > len(quarters) / 2:
                        trend_effect = baseline * 0.004 * (i - len(quarters) / 2)  # Improving in later periods
                    else:
                        trend_effect = baseline * 0.001 * i  # Slight improvement over time

                    # Random noise
                    noise = np.random.normal(0, volatility)

                    # Calculate metric value
                    value = baseline + cycle_effect + trend_effect + noise

                    # Ensure reasonable values
                    value = max(0, value)  # No negative margins/returns

                    metric_trend.append({
                        "date": quarter.strftime('%Y-%m-%d'),
                        "value": value
                    })

                result[metric] = metric_trend

        return result

    def _analyze_market_share_trend(self, sector: str, quarters: List[datetime]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze market share trends among top companies in a sector"""
        # This would normally use historical revenue data
        # For simplicity, we'll create representative data

        # Get top companies in sector
        sector_companies = self._get_companies_in_sector(sector)
        if not sector_companies:
            return {}

        # Select top 5 companies by market cap
        top_companies = []
        for ticker in sector_companies[:10]:  # Check first 10
            company_info = self.data_loader.get_company_info(ticker)
            if company_info and company_info.get('market_cap'):
                top_companies.append({
                    "ticker": ticker,
                    "name": company_info.get('name', ticker),
                    "market_cap": company_info.get('market_cap')
                })

        top_companies.sort(key=lambda x: x['market_cap'], reverse=True)
        top_companies = top_companies[:5]  # Keep top 5

        if not top_companies:
            return {}

        # Generate market share data (percentages)
        market_share_trend = {}

        # Initial market shares (based on current size)
        total_cap = sum(company['market_cap'] for company in top_companies)
        initial_shares = {}

        for company in top_companies:
            initial_shares[company['ticker']] = company['market_cap'] / total_cap if total_cap > 0 else 0.2

        # Adjust to ensure sum is 100%
        share_sum = sum(initial_shares.values())
        if share_sum > 0:
            for ticker in initial_shares:
                initial_shares[ticker] /= share_sum

        # Generate trend
        for ticker in initial_shares:
            company_trend = []

            # Market share evolution parameters vary by ticker
            # This simulates how different companies gain/lose share
            if ticker == top_companies[0]['ticker']:  # Market leader
                trend = 0.001  # Slight increase over time
                volatility = 0.005
            elif ticker == top_companies[-1]['ticker']:  # Smallest company
                trend = 0.002  # Faster growth (small base)
                volatility = 0.01
            else:  # Middle companies
                trend = -0.0005  # Slight decrease
                volatility = 0.007

            share = initial_shares[ticker]

            for i, quarter in enumerate(quarters):
                # Update share with trend and random noise
                share += trend + np.random.normal(0, volatility)

                # Ensure share stays between 0% and 100%
                share = max(0.01, min(0.7, share))

                company_trend.append({
                    "date": quarter.strftime('%Y-%m-%d'),
                    "value": share
                })

            market_share_trend[ticker] = company_trend

        # Normalize to ensure shares sum to 100% for each quarter
        for i in range(len(quarters)):
            quarter_sum = sum(market_share_trend[ticker][i]["value"] for ticker in market_share_trend)

            if quarter_sum > 0:
                for ticker in market_share_trend:
                    market_share_trend[ticker][i]["value"] /= quarter_sum

        return {"market_shares": market_share_trend}