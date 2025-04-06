import os
import logging
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SECTOR_MAPPING, SECTOR_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_mapping')


class SectorMapper:
    """
    Class for mapping companies to sectors and industries.
    Provides functionality to identify the sector and industry
    for a given company based on its profile data.
    """

    def __init__(self, data_dir: str = SECTOR_DATA_DIR):
        """
        Initialize the sector mapper with default mappings

        Args:
            data_dir: Directory for sector data files
        """
        self.data_dir = Path(data_dir)
        self.sector_mapping = SECTOR_MAPPING

        # Additional mappings loaded from data files
        self.company_to_sector = self._load_company_mappings()
        logger.info("Initialized SectorMapper")

    def get_sector(self, ticker: str, company_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the sector for a company

        Args:
            ticker: Company ticker symbol
            company_info: Optional company profile information

        Returns:
            Sector name (standardized)
        """
        # First check direct mapping
        if ticker in self.company_to_sector:
            return self.company_to_sector[ticker]['sector']

        # If company info is provided, use that
        if company_info:
            # Try to get sector directly from company info
            if 'sector' in company_info and company_info['sector']:
                sector = company_info['sector']
                return self._normalize_sector_name(sector)

            # Try to get industry and map to sector
            if 'industry' in company_info and company_info['industry']:
                industry = company_info['industry']
                sector = self._map_industry_to_sector(industry)
                if sector != "Unknown":
                    return sector

        # Default to "Unknown" if we can't determine sector
        return "Unknown"

    def get_industry(self, ticker: str, company_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the industry for a company

        Args:
            ticker: Company ticker symbol
            company_info: Optional company profile information

        Returns:
            Industry name
        """
        # First check direct mapping
        if ticker in self.company_to_sector:
            return self.company_to_sector[ticker]['industry']

        # If company info is provided, use that
        if company_info and 'industry' in company_info:
            return company_info['industry']

        # Default to "Unknown" if we can't determine industry
        return "Unknown"

    def get_peer_companies(self, ticker: str, sector: str, max_peers: int = 5) -> List[str]:
        """
        Get list of peer companies in the same sector

        Args:
            ticker: Company ticker symbol
            sector: Company sector
            max_peers: Maximum number of peers to return

        Returns:
            List of peer tickers
        """
        # In a real implementation, this would query a database or API
        # Here we provide sample data
        sector_peers = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "INTC", "CSCO", "ADBE", "CRM"],
            "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "ABBV", "DHR", "BMY", "LLY"],
            "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA"],
            "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "DG"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "EL", "CL", "GIS"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI"],
            "Industrials": ["HON", "UNP", "UPS", "BA", "CAT", "GE", "LMT", "RTX", "MMM", "FDX"],
            "Materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "DOW", "DD", "NUE", "BHP"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "DLR", "AVB", "EQR"],
            "Communication Services": ["GOOGL", "META", "VZ", "T", "CMCSA", "NFLX", "DIS", "TMUS", "ATVI", "EA"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ES", "WEC"]
        }

        # Get normalized sector name
        normalized_sector = self._normalize_sector_name(sector)

        # Get peers for the sector
        peers = sector_peers.get(normalized_sector, [])

        # Remove the ticker itself from the list
        peers = [p for p in peers if p != ticker]

        # Return up to max_peers
        return peers[:max_peers]

    def add_company_mapping(self, ticker: str, sector: str, industry: str) -> bool:
        """
        Add or update a company to sector/industry mapping

        Args:
            ticker: Company ticker symbol
            sector: Sector name
            industry: Industry name

        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize sector name
            normalized_sector = self._normalize_sector_name(sector)

            # Add to mapping
            self.company_to_sector[ticker] = {
                'sector': normalized_sector,
                'industry': industry
            }

            # Save mappings
            self._save_company_mappings()

            logger.info(f"Added mapping for {ticker}: Sector={normalized_sector}, Industry={industry}")
            return True
        except Exception as e:
            logger.error(f"Error adding company mapping for {ticker}: {e}")
            return False

    def get_sectors(self) -> List[str]:
        """
        Get list of all sectors

        Returns:
            List of sector names
        """
        return list(self.sector_mapping.keys())

    def get_industries_for_sector(self, sector: str) -> List[str]:
        """
        Get list of industries for a sector

        Args:
            sector: Sector name

        Returns:
            List of industry names
        """
        # Normalize sector name
        normalized_sector = self._normalize_sector_name(sector)

        # Return industries for the sector
        if normalized_sector in self.sector_mapping:
            return self.sector_mapping[normalized_sector]
        else:
            return []

    # Private helper methods

    def _load_company_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load company to sector mappings from file"""
        mappings = {}

        try:
            # Check if we have a mappings file
            mapping_file = self.data_dir / "company_sector_mappings.json"

            if mapping_file.exists():
                # Load mappings
                with open(mapping_file, 'r') as f:
                    mappings = json.load(f)

                logger.info(f"Loaded mappings for {len(mappings)} companies")
        except Exception as e:
            logger.error(f"Error loading company mappings: {e}")

        return mappings

    def _save_company_mappings(self) -> bool:
        """Save company to sector mappings to file"""
        try:
            # Ensure data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Save mappings
            mapping_file = self.data_dir / "company_sector_mappings.json"

            with open(mapping_file, 'w') as f:
                json.dump(self.company_to_sector, f, indent=2)

            logger.info(f"Saved mappings for {len(self.company_to_sector)} companies")
            return True
        except Exception as e:
            logger.error(f"Error saving company mappings: {e}")
            return False

    def _normalize_sector_name(self, sector: str) -> str:
        """Normalize sector name to match our standard sector names"""
        # Check if it's already a valid sector
        if sector in self.sector_mapping:
            return sector

        # Try common variants
        sector_variants = {
            "Technology": ["Information Technology", "Tech", "IT", "Information Tech", "Software"],
            "Financials": ["Financial", "Finance", "Banking", "Banks", "Insurance"],
            "Healthcare": ["Health Care", "Health", "Medical", "Pharmaceutical", "Biotech"],
            "Consumer Discretionary": ["Consumer Cyclical", "Discretionary", "Retail", "Apparel"],
            "Consumer Staples": ["Staples", "Consumer Defensive", "Food", "Beverage"],
            "Energy": ["Oil & Gas", "Energy Equipment", "Energy Services", "Petroleum"],
            "Industrials": ["Industrial", "Manufacturing", "Aerospace", "Defense", "Transportation"],
            "Materials": ["Basic Materials", "Chemicals", "Mining", "Metals"],
            "Real Estate": ["Realty", "REIT", "Property", "Real Estate Investment"],
            "Communication Services": ["Communication", "Telecom", "Media", "Entertainment"],
            "Utilities": ["Utility", "Electric Utility", "Water Utility", "Gas Utility", "Power"]
        }

        # Check for common variants
        sector_lower = sector.lower()
        for standard_sector, variants in sector_variants.items():
            if any(variant.lower() in sector_lower for variant in variants):
                return standard_sector

        # If not found, return original sector
        return sector

    def _map_industry_to_sector(self, industry: str) -> str:
        """Map an industry to its parent sector"""
        industry_to_sector = {}

        # Build industry to sector mapping from the sector mapping
        for sector, industries in self.sector_mapping.items():
            for ind in industries:
                industry_to_sector[ind.lower()] = sector

        # Normalize industry name
        industry_lower = industry.lower()

        # Direct lookup
        if industry_lower in industry_to_sector:
            return industry_to_sector[industry_lower]

        # Try partial match
        for ind, sector in industry_to_sector.items():
            if ind in industry_lower or industry_lower in ind:
                return sector

        # If no match found, try to guess based on keywords
        sector_keywords = {
            "Technology": ["software", "hardware", "semiconductor", "tech", "internet", "computer", "electronics"],
            "Healthcare": ["health", "medical", "pharma", "biotech", "drug", "hospital"],
            "Financials": ["bank", "insurance", "asset", "financial", "invest", "capital", "credit"],
            "Consumer Discretionary": ["retail", "auto", "luxury", "apparel", "restaurant", "hotel", "entertainment"],
            "Consumer Staples": ["food", "beverage", "household", "personal", "grocery"],
            "Energy": ["oil", "gas", "energy", "fuel", "power", "renewable"],
            "Industrials": ["industrial", "machinery", "defense", "aerospace", "transport"],
            "Materials": ["material", "chemical", "metal", "mining", "steel", "paper"],
            "Real Estate": ["real estate", "property", "reit", "development"],
            "Communication Services": ["communication", "telecom", "media", "entertainment", "social"],
            "Utilities": ["utility", "electric", "water", "gas"]
        }

        # Count keyword matches for each sector
        sector_matches = {sector: 0 for sector in sector_keywords}
        for sector, keywords in sector_keywords.items():
            for keyword in keywords:
                if keyword in industry_lower:
                    sector_matches[sector] += 1

        # Get sector with most matches
        best_match = max(sector_matches.items(), key=lambda x: x[1])

        # If we have at least one match, return the sector
        if best_match[1] > 0:
            return best_match[0]

        # If still no match, return Unknown
        return "Unknown"