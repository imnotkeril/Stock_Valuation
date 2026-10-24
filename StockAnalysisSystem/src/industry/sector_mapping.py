import os
import sys
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import SECTOR_MAPPING, SECTOR_METRICS
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_mapping')


class SectorMapper:
    """
    Class responsible for mapping companies to sectors and industries.
    Provides functionalities for sector-based classification, finding companies
    within the same sector/industry, and standardizing sector/industry names.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the sector mapper with configuration data.

        Args:
            data_loader: DataLoader instance for fetching company data
        """
        self.data_loader = data_loader if data_loader else DataLoader()

        # Load sector mapping from config
        self.sector_mapping = SECTOR_MAPPING

        # Load sector metrics for additional context
        self.sector_metrics = SECTOR_METRICS

        # Create inverse mapping (industry -> sector)
        self.industry_to_sector = self._create_industry_to_sector_mapping()

        # Cache for company sector data
        self.company_sector_cache = {}

        logger.info("SectorMapper initialized")

    def _create_industry_to_sector_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from industry to sector.

        Returns:
            Dictionary with industry names as keys and sector names as values
        """
        industry_to_sector = {}

        for sector, industries in self.sector_mapping.items():
            for industry in industries:
                industry_to_sector[industry.lower()] = sector

        return industry_to_sector

    def get_company_sector(self, ticker: str, refresh: bool = False) -> Dict[str, str]:
        """
        Get the sector and industry information for a company.

        Args:
            ticker: Company ticker symbol
            refresh: Whether to refresh cached data

        Returns:
            Dictionary with sector and industry information
        """
        # Check cache first unless refresh is requested
        if ticker in self.company_sector_cache and not refresh:
            return self.company_sector_cache[ticker]

        try:
            # Get company information from data loader
            company_info = self.data_loader.get_company_info(ticker)

            if not company_info:
                logger.warning(f"Unable to fetch company info for {ticker}")
                return {
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "subsector": "Unknown",
                    "confidence": 0.0
                }

            # Extract sector and industry
            sector = company_info.get('sector', 'Unknown')
            industry = company_info.get('industry', 'Unknown')

            # Standardize sector name
            standardized_sector = self.standardize_sector_name(sector)

            # Determine subsector (more specific classification within industry)
            subsector = self._determine_subsector(industry, company_info)

            # Calculate confidence in the classification
            confidence = self._calculate_classification_confidence(standardized_sector, industry, company_info)

            result = {
                "sector": standardized_sector,
                "industry": industry,
                "subsector": subsector,
                "confidence": confidence
            }

            # Cache the result
            self.company_sector_cache[ticker] = result

            return result

        except Exception as e:
            logger.error(f"Error determining sector for {ticker}: {e}")
            return {
                "sector": "Unknown",
                "industry": "Unknown",
                "subsector": "Unknown",
                "confidence": 0.0
            }

    def standardize_sector_name(self, sector: str) -> str:
        """
        Standardize sector name to match one of the defined sectors.

        Args:
            sector: Original sector name

        Returns:
            Standardized sector name
        """
        if not sector or sector == 'Unknown':
            return 'Unknown'

        # Direct match
        if sector in self.sector_mapping:
            return sector

        # Try case-insensitive match
        sector_lower = sector.lower()
        for std_sector in self.sector_mapping:
            if std_sector.lower() == sector_lower:
                return std_sector

        # Try partial match
        for std_sector in self.sector_mapping:
            if sector_lower in std_sector.lower() or std_sector.lower() in sector_lower:
                return std_sector

        # Special cases and aliases
        sector_aliases = {
            "technology": "Technology",
            "tech": "Technology",
            "it": "Technology",
            "software": "Technology",
            "hardware": "Technology",
            "semiconductor": "Technology",

            "healthcare": "Healthcare",
            "health": "Healthcare",
            "medical": "Healthcare",
            "pharma": "Healthcare",
            "biotech": "Healthcare",

            "financial": "Financials",
            "finance": "Financials",
            "bank": "Financials",
            "insurance": "Financials",

            "consumer discretionary": "Consumer Discretionary",
            "retail": "Consumer Discretionary",
            "apparel": "Consumer Discretionary",
            "luxury": "Consumer Discretionary",
            "automobile": "Consumer Discretionary",
            "auto": "Consumer Discretionary",

            "consumer staples": "Consumer Staples",
            "food": "Consumer Staples",
            "beverage": "Consumer Staples",
            "household": "Consumer Staples",

            "energy": "Energy",
            "oil": "Energy",
            "gas": "Energy",
            "renewable": "Energy",

            "industrial": "Industrials",
            "manufacturing": "Industrials",
            "aerospace": "Industrials",
            "defense": "Industrials",
            "machinery": "Industrials",
            "transportation": "Industrials",

            "materials": "Materials",
            "chemical": "Materials",
            "mining": "Materials",
            "metals": "Materials",

            "real estate": "Real Estate",
            "reit": "Real Estate",
            "property": "Real Estate",

            "communication": "Communication Services",
            "telecom": "Communication Services",
            "media": "Communication Services",
            "entertainment": "Communication Services",

            "utilities": "Utilities",
            "electric": "Utilities",
            "water": "Utilities",
            "gas utility": "Utilities"
        }

        for alias, std_sector in sector_aliases.items():
            if alias in sector_lower:
                return std_sector

        # If no match found, return original
        return sector

    def _determine_subsector(self, industry: str, company_info: Dict[str, Any]) -> str:
        """
        Determine a more specific subsector classification.

        Args:
            industry: Company industry
            company_info: Full company information dict

        Returns:
            Subsector classification
        """
        if not industry or industry == 'Unknown':
            return 'Unknown'

        # Some data providers already include a sub-industry field
        if 'subIndustry' in company_info:
            return company_info['subIndustry']

        # Try to determine from business description if available
        description = company_info.get('description', '')

        # Technology subsector determination
        if self.standardize_sector_name(industry) == 'Technology':
            tech_keywords = {
                'Software Development': ['software', 'application', 'saas', 'cloud', 'platform'],
                'Hardware': ['hardware', 'computer', 'device', 'electronics', 'semiconductor manufacturing'],
                'Semiconductors': ['semiconductor', 'chip', 'integrated circuit', 'processor'],
                'IT Services': ['consulting', 'outsourcing', 'IT service', 'system integration'],
                'Internet Services': ['internet', 'e-commerce', 'online', 'search engine', 'social media']
            }

            return self._keyword_based_classification(description, tech_keywords, industry)

        # Financial subsector determination
        elif self.standardize_sector_name(industry) == 'Financials':
            finance_keywords = {
                'Commercial Banking': ['bank', 'commercial bank', 'retail bank', 'banking services'],
                'Investment Banking': ['investment bank', 'capital market', 'securities', 'trading'],
                'Insurance': ['insurance', 'insurer', 'risk management', 'underwriting'],
                'Asset Management': ['asset management', 'fund', 'investment management', 'portfolio'],
                'Consumer Finance': ['consumer loan', 'credit card', 'personal finance', 'lending']
            }

            return self._keyword_based_classification(description, finance_keywords, industry)

        # For other sectors, just return the industry
        return industry

    def _keyword_based_classification(self, text: str, keyword_dict: Dict[str, List[str]], default: str) -> str:
        """
        Classify text based on keywords.

        Args:
            text: Text to classify
            keyword_dict: Dictionary of {classification: [keywords]}
            default: Default classification if no match

        Returns:
            Classification based on keyword matches
        """
        if not text:
            return default

        text = text.lower()

        # Count matches for each classification
        matches = {classification: 0 for classification in keyword_dict}

        for classification, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    matches[classification] += 1

        # Find classification with most matches
        best_match = max(matches.items(), key=lambda x: x[1])

        # If no matches, return default
        if best_match[1] == 0:
            return default

        return best_match[0]

    def _calculate_classification_confidence(self, sector: str, industry: str, company_info: Dict[str, Any]) -> float:
        """
        Calculate confidence level in sector/industry classification.

        Args:
            sector: Classified sector
            industry: Classified industry
            company_info: Company information dictionary

        Returns:
            Confidence level from 0.0 to 1.0
        """
        # Start with base confidence
        confidence = 0.5

        # If sector is unknown, low confidence
        if sector == 'Unknown':
            return 0.1

        # If we have a direct match with known sectors, higher confidence
        if sector in self.sector_mapping:
            confidence += 0.2

        # If industry is in the known industries for this sector, higher confidence
        if industry in self.sector_mapping.get(sector, []):
            confidence += 0.2

        # If multiple data sources agree, higher confidence
        if company_info.get('industry_alternate') and industry == company_info.get('industry_alternate'):
            confidence += 0.1

        # If business description supports classification, higher confidence
        description = company_info.get('description', '')
        if description:
            if sector.lower() in description.lower() or industry.lower() in description.lower():
                confidence += 0.1

        # Cap at 1.0
        return min(1.0, confidence)

    def find_peer_companies(self, ticker: str, same_industry: bool = True,
                            max_peers: int = 10) -> List[Dict[str, Any]]:
        """
        Find peer companies in the same sector/industry.

        Args:
            ticker: Company ticker to find peers for
            same_industry: Whether to require peers to be in the same industry (stricter)
            max_peers: Maximum number of peers to return

        Returns:
            List of peer company dictionaries with ticker and metadata
        """
        try:
            # First get the company's sector info
            company_sector_info = self.get_company_sector(ticker)

            if company_sector_info["sector"] == "Unknown":
                logger.warning(f"Cannot find peers for {ticker} - unknown sector")
                return []

            # Try to use the data loader's get_industry_peers method if available
            if hasattr(self.data_loader, 'get_industry_peers'):
                peers = self.data_loader.get_industry_peers(ticker, max_peers)
                if peers:
                    return peers

            # Fallback implementation if data loader doesn't have the method
            # or if it returns no results

            # First try to get peers from a sector ETF
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
                "Utilities": "XLU"
            }

            peers = []
            sector = company_sector_info["sector"]
            industry = company_sector_info["industry"]

            # Look up the appropriate sector ETF
            etf_ticker = sector_etfs.get(sector)

            if etf_ticker:
                try:
                    # Try to load the ETF holdings
                    etf_data = self.data_loader.get_company_info(etf_ticker)

                    if 'holdings' in etf_data:
                        holdings = etf_data['holdings']

                        # Filter by industry if requested
                        potential_peers = []

                        for holding_ticker in holdings[:30]:  # Check top 30 holdings
                            if holding_ticker == ticker:
                                continue

                            holding_info = self.data_loader.get_company_info(holding_ticker)

                            if not holding_info:
                                continue

                            # If same_industry is True, check that industries match
                            if same_industry:
                                if holding_info.get('industry') == industry:
                                    potential_peers.append({
                                        'symbol': holding_ticker,
                                        'name': holding_info.get('name', holding_ticker),
                                        'sector': holding_info.get('sector', 'Unknown'),
                                        'industry': holding_info.get('industry', 'Unknown')
                                    })
                            else:
                                # Just require same sector
                                if self.standardize_sector_name(holding_info.get('sector', '')) == sector:
                                    potential_peers.append({
                                        'symbol': holding_ticker,
                                        'name': holding_info.get('name', holding_ticker),
                                        'sector': holding_info.get('sector', 'Unknown'),
                                        'industry': holding_info.get('industry', 'Unknown')
                                    })

                        # Sort by market cap similarity if available
                        company_market_cap = self.data_loader.get_company_info(ticker).get('market_cap')

                        if company_market_cap:
                            for peer in potential_peers:
                                peer_market_cap = self.data_loader.get_company_info(peer['symbol']).get('market_cap')
                                if peer_market_cap:
                                    # Calculate size similarity (0 to 1, higher is more similar)
                                    ratio = min(company_market_cap, peer_market_cap) / max(company_market_cap,
                                                                                           peer_market_cap)
                                    peer['size_similarity'] = ratio
                                else:
                                    peer['size_similarity'] = 0.0

                            # Sort by size similarity (most similar first)
                            potential_peers.sort(key=lambda x: x.get('size_similarity', 0.0), reverse=True)

                        # Return the top peers
                        peers = potential_peers[:max_peers]
                except Exception as e:
                    logger.warning(f"Error getting peers from ETF {etf_ticker}: {e}")

            # If we couldn't get peers from ETF, just look up some companies in the same sector
            if not peers:
                # This would require a database of companies or API that lets us search by sector
                # Just return a placeholder for now
                logger.warning(f"Fallback peer selection for {ticker} - limited functionality")

                # Here you would normally query a database or API for companies in sector
                # Example fallback logic:
                if sector == "Technology":
                    peers = [
                        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology",
                         "industry": "Consumer Electronics"},
                        {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology",
                         "industry": "Software—Infrastructure"},
                        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology",
                         "industry": "Internet Content & Information"}
                    ]
                # Add similar fallbacks for other sectors as needed

            return peers

        except Exception as e:
            logger.error(f"Error finding peers for {ticker}: {e}")
            return []

    def get_all_companies_in_sector(self, sector: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all companies in a specific sector.

        Args:
            sector: Sector to query
            limit: Maximum number of companies to return

        Returns:
            List of company dictionaries
        """
        # Standardize the sector name
        standardized_sector = self.standardize_sector_name(sector)

        if standardized_sector == "Unknown":
            logger.warning(f"Unknown sector: {sector}")
            return []

        try:
            # This would typically come from a database or API query
            # For now, we'll use a screening approach via data_loader

            # Define sector-specific screening filters
            screening_filters = {
                "sector": {"operator": "=", "value": standardized_sector}
            }

            # Use the screener if available in data_loader
            if hasattr(self.data_loader, 'get_stock_screener'):
                return self.data_loader.get_stock_screener(screening_filters, limit)

            # Fallback implementation if screener not available
            # Use sector ETFs as a proxy for sector companies
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
                "Utilities": "XLU"
            }

            etf_ticker = sector_etfs.get(standardized_sector)

            if not etf_ticker:
                logger.warning(f"No ETF found for sector {standardized_sector}")
                return []

            # Try to get ETF holdings
            etf_data = self.data_loader.get_company_info(etf_ticker)

            if 'holdings' not in etf_data:
                logger.warning(f"No holdings data for ETF {etf_ticker}")
                return []

            companies = []

            for holding_ticker in etf_data['holdings'][:limit]:
                try:
                    holding_info = self.data_loader.get_company_info(holding_ticker)

                    if holding_info:
                        companies.append({
                            'symbol': holding_ticker,
                            'name': holding_info.get('name', holding_ticker),
                            'sector': holding_info.get('sector', 'Unknown'),
                            'industry': holding_info.get('industry', 'Unknown'),
                            'market_cap': holding_info.get('market_cap')
                        })
                except Exception as e:
                    logger.warning(f"Error getting info for {holding_ticker}: {e}")
                    continue

            return companies

        except Exception as e:
            logger.error(f"Error getting companies in sector {sector}: {e}")
            return []

    def get_sector_classification_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the full sector classification hierarchy.

        Returns:
            Nested dictionary of the sector -> industry -> subsector hierarchy
        """
        hierarchy = {}

        # Start with base sector mapping
        for sector, industries in self.sector_mapping.items():
            hierarchy[sector] = {}

            for industry in industries:
                # For each industry, determine common subsectors
                # This would typically come from a predefined classification system
                # but for now we'll use a simplified approach

                hierarchy[sector][industry] = self._get_subsectors_for_industry(industry)

        return hierarchy

    def _get_subsectors_for_industry(self, industry: str) -> List[str]:
        """
        Get common subsectors for a given industry.

        Args:
            industry: Industry name

        Returns:
            List of common subsectors
        """
        # This is a simplified implementation
        # Ideally this would come from a standard classification system like GICS

        industry_subsectors = {
            # Technology
            "Software": ["Application Software", "System Software", "Development Tools", "Security Software"],
            "Hardware": ["Computer Hardware", "Storage Devices", "Networking Equipment", "Consumer Electronics"],
            "Semiconductors": ["Semiconductor Design", "Semiconductor Equipment", "Integrated Circuits",
                               "Specialized Chips"],
            "IT Services": ["IT Consulting", "Data Processing", "System Integration", "Outsourcing"],
            "Internet": ["E-Commerce", "Social Media", "Search Engines", "Online Content"],

            # Healthcare
            "Biotechnology": ["Genetic Research", "Drug Discovery", "Vaccines", "Medical Research"],
            "Pharmaceuticals": ["Generic Drugs", "Branded Drugs", "OTC Products", "Drug Manufacturing"],
            "Medical Devices": ["Diagnostic Equipment", "Therapeutic Devices", "Monitoring Equipment",
                                "Surgical Tools"],
            "Healthcare Services": ["Hospitals", "Clinics", "Managed Care", "Specialized Health Services"],

            # Financials
            "Banks": ["Commercial Banks", "Investment Banks", "Regional Banks", "Credit Unions"],
            "Insurance": ["Life Insurance", "Property Insurance", "Health Insurance", "Reinsurance"],
            "Asset Management": ["Mutual Funds", "ETFs", "Wealth Management", "Private Equity"],
            "Diversified Financial Services": ["Consumer Finance", "Business Finance", "Financial Exchanges",
                                               "Transaction Processing"],

            # Add more industries and subsectors as needed
        }

        # Return subsectors if available, otherwise empty list
        return industry_subsectors.get(industry, [])

    def get_sector_size_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get relative size data for each sector (by market cap, companies, etc.).

        Returns:
            Dictionary with sector size metrics
        """
        # This would typically come from a database of current market data
        # For now, we'll return approximate data

        # These are approximate percentages of the total market
        sector_market_cap_weights = {
            "Technology": 28.0,
            "Healthcare": 13.0,
            "Financials": 11.0,
            "Consumer Discretionary": 10.0,
            "Communication Services": 9.0,
            "Industrials": 8.0,
            "Consumer Staples": 7.0,
            "Energy": 5.0,
            "Utilities": 3.0,
            "Real Estate": 3.0,
            "Materials": 3.0
        }

        # Approximate number of companies in each sector
        sector_company_counts = {
            "Technology": 150,
            "Healthcare": 130,
            "Financials": 120,
            "Consumer Discretionary": 110,
            "Industrials": 100,
            "Communication Services": 60,
            "Consumer Staples": 50,
            "Energy": 40,
            "Materials": 30,
            "Real Estate": 30,
            "Utilities": 30
        }

        return {
            "market_cap_weights": sector_market_cap_weights,
            "company_counts": sector_company_counts
        }

    def get_sector_performance(self, sector: str,
                               period: str = '1y') -> Dict[str, float]:
        """
        Get performance metrics for a sector over a specified period.

        Args:
            sector: Sector name
            period: Time period ('1m', '3m', '6m', '1y', '3y', '5y')

        Returns:
            Dictionary with performance metrics
        """
        standardized_sector = self.standardize_sector_name(sector)

        if standardized_sector == "Unknown":
            logger.warning(f"Unknown sector: {sector}")
            return {}

        try:
            # Use sector ETFs as proxies for sector performance
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
                "Utilities": "XLU"
            }

            etf_ticker = sector_etfs.get(standardized_sector)

            if not etf_ticker:
                logger.warning(f"No ETF found for sector {standardized_sector}")
                return {}

            # Get historical prices for the ETF
            # Map period string to date range
            date_map = {
                '1m': 30,
                '3m': 90,
                '6m': 180,
                '1y': 365,
                '3y': 3 * 365,
                '5y': 5 * 365
            }

            days = date_map.get(period, 365)  # Default to 1 year

            # Get sector ETF price data
            etf_data = self.data_loader.get_sector_data(standardized_sector, True)

            if etf_data.empty:
                logger.warning(f"No price data for sector ETF {etf_ticker}")
                return {}

            # Calculate performance metrics
            performance = {}

            # Total return
            start_price = etf_data['Close'].iloc[0]
            end_price = etf_data['Close'].iloc[-1]
            performance['total_return'] = (end_price / start_price - 1) * 100

            # Calculate annualized return for periods longer than 1 year
            years = days / 365
            if years > 1:
                performance['annualized_return'] = ((end_price / start_price) ** (1 / years) - 1) * 100

            # Volatility (standard deviation of daily returns)
            daily_returns = etf_data['Close'].pct_change().dropna()
            performance['volatility'] = daily_returns.std() * (252 ** 0.5) * 100  # Annualized

            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            performance['max_drawdown'] = drawdown.min() * 100

            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_return = (performance.get('annualized_return', performance['total_return']) / 100) - risk_free_rate
            if performance['volatility'] > 0:
                performance['sharpe_ratio'] = excess_return / (performance['volatility'] / 100)

            return performance

        except Exception as e:
            logger.error(f"Error getting performance for sector {sector}: {e}")
            return {}


# Helper functions
def get_sectors_list() -> List[str]:
    """
    Get a list of all standard sectors.

    Returns:
        List of sector names
    """
    return list(SECTOR_MAPPING.keys())


def get_industries_for_sector(sector: str) -> List[str]:
    """
    Get a list of industries within a sector.

    Args:
        sector: Sector name

    Returns:
        List of industry names
    """
    mapper = SectorMapper()
    standardized_sector = mapper.standardize_sector_name(sector)

    return SECTOR_MAPPING.get(standardized_sector, [])


def map_ticker_to_sector(ticker: str) -> str:
    """
    Map a ticker to its sector.

    Args:
        ticker: Company ticker symbol

    Returns:
        Sector name
    """
    mapper = SectorMapper()
    sector_info = mapper.get_company_sector(ticker)

    return sector_info["sector"]