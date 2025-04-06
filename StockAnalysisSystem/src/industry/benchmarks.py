import os
import logging
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SECTOR_DATA_DIR, SECTOR_MAPPING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('industry_benchmarks')


class IndustryBenchmarks:
    """
    Class for managing industry benchmarks and sector-specific data.
    Provides data for comparing companies with industry averages.
    """

    def __init__(self, data_dir: str = SECTOR_DATA_DIR):
        """
        Initialize industry benchmarks manager

        Args:
            data_dir: Directory for sector data files
        """
        self.data_dir = Path(data_dir)
        self.sector_mapping = SECTOR_MAPPING

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load sector benchmarks if available, otherwise use defaults
        self.benchmarks = self._load_benchmarks()

        logger.info(f"Initialized IndustryBenchmarks with data directory: {self.data_dir}")

    def get_sector_benchmarks(self, sector: str) -> Dict[str, Dict[str, float]]:
        """
        Get benchmark financial ratios for a specific sector

        Args:
            sector: The market sector name

        Returns:
            Dictionary of benchmark ratios by category
        """
        # Normalize sector name
        normalized_sector = self._normalize_sector_name(sector)

        # Return benchmarks for the sector if available, otherwise return defaults
        return self.benchmarks.get(normalized_sector, self._get_default_benchmarks())

    def get_industry_benchmarks(self, industry: str) -> Dict[str, Dict[str, float]]:
        """
        Get benchmark financial ratios for a specific industry

        Args:
            industry: The industry name

        Returns:
            Dictionary of benchmark ratios by category
        """
        # Map industry to sector first
        sector = self._map_industry_to_sector(industry)

        # Get sector benchmarks as a base
        sector_benchmarks = self.get_sector_benchmarks(sector)

        # Look for industry-specific benchmarks
        industry_benchmarks = self._get_industry_specific_benchmarks(industry)

        # Merge sector and industry benchmarks, with industry taking precedence
        merged_benchmarks = {}
        for category in sector_benchmarks:
            if category in industry_benchmarks:
                merged_benchmarks[category] = {**sector_benchmarks[category], **industry_benchmarks[category]}
            else:
                merged_benchmarks[category] = sector_benchmarks[category]

        return merged_benchmarks

    def get_sector_performance(self) -> pd.DataFrame:
        """
        Get recent performance metrics for all sectors

        Returns:
            DataFrame with sector performance data
        """
        try:
            # Check if we have recent data file
            performance_file = self.data_dir / "sector_performance.csv"

            if performance_file.exists():
                # Load data
                df = pd.read_csv(performance_file)

                # Check if data is recent (within 7 days)
                if 'date' in df.columns:
                    latest_date = pd.to_datetime(df['date'].max())
                    now = pd.Timestamp.now()
                    if (now - latest_date).days <= 7:
                        return df

            # If no file or data is old, return sample data
            return self._get_sample_sector_performance()
        except Exception as e:
            logger.error(f"Error loading sector performance data: {e}")
            return self._get_sample_sector_performance()

    def update_benchmarks(self, sector: str, benchmarks: Dict[str, Dict[str, float]]) -> bool:
        """
        Update benchmarks for a specific sector

        Args:
            sector: The market sector name
            benchmarks: New benchmark data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize sector name
            normalized_sector = self._normalize_sector_name(sector)

            # Update internal benchmarks dictionary
            self.benchmarks[normalized_sector] = benchmarks

            # Save to file
            self._save_benchmarks()

            logger.info(f"Updated benchmarks for sector: {normalized_sector}")
            return True
        except Exception as e:
            logger.error(f"Error updating benchmarks for sector {sector}: {e}")
            return False

    def get_peer_companies(self, ticker: str, sector: str, max_peers: int = 5) -> List[str]:
        """
        Get list of peer companies in the same sector

        Args:
            ticker: Company ticker
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

    # Private helper methods

    def _load_benchmarks(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load benchmark data from files or return defaults"""
        benchmarks = {}

        try:
            # Check if we have a benchmarks file
            benchmark_file = self.data_dir / "sector_benchmarks.json"

            if benchmark_file.exists():
                # Load benchmarks
                with open(benchmark_file, 'r') as f:
                    benchmarks = json.load(f)

                logger.info(f"Loaded benchmarks for {len(benchmarks)} sectors")
            else:
                # Use default benchmarks
                benchmarks = self._get_default_sectors_benchmarks()

                # Save defaults for future use
                self._save_benchmarks(benchmarks)

                logger.info("Created default sector benchmarks")
        except Exception as e:
            logger.error(f"Error loading benchmarks: {e}")
            benchmarks = self._get_default_sectors_benchmarks()

        return benchmarks

    def _save_benchmarks(self, benchmarks=None) -> bool:
        """Save benchmarks to file"""
        if benchmarks is None:
            benchmarks = self.benchmarks

        try:
            # Save to file
            benchmark_file = self.data_dir / "sector_benchmarks.json"

            with open(benchmark_file, 'w') as f:
                json.dump(benchmarks, f, indent=2)

            logger.info(f"Saved benchmarks to {benchmark_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving benchmarks: {e}")
            return False

    def _normalize_sector_name(self, sector: str) -> str:
        """Normalize sector name to match our benchmark keys"""
        # Check if it's already a valid sector
        if sector in self.benchmarks:
            return sector

        # Try common variants
        sector_variants = {
            "Technology": ["Information Technology", "Tech", "IT", "Information Tech"],
            "Financials": ["Financial", "Finance", "Banking", "Banks"],
            "Healthcare": ["Health Care", "Health", "Medical"],
            "Consumer Discretionary": ["Consumer Cyclical", "Discretionary", "Retail"],
            "Consumer Staples": ["Staples", "Consumer Defensive"],
            "Energy": ["Oil & Gas", "Energy Equipment", "Energy Services"],
            "Industrials": ["Industrial", "Manufacturing", "Aerospace", "Defense", "Transportation"],
            "Materials": ["Basic Materials", "Chemicals", "Mining"],
            "Real Estate": ["Realty", "REIT", "Property"],
            "Communication Services": ["Communication", "Telecom", "Media"],
            "Utilities": ["Utility", "Electric Utility", "Water Utility", "Gas Utility"]
        }

        # Check for common variants
        for standard_sector, variants in sector_variants.items():
            if any(variant.lower() in sector.lower() for variant in variants):
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

        # If still no match, return a default
        return "Unknown"


    def _get_industry_specific_benchmarks(self, industry: str) -> Dict[str, Dict[str, float]]:
        """Get industry-specific benchmark data"""
        try:
            # Check if we have industry benchmark file
            industry_file = self.data_dir / f"industry_{industry.lower().replace(' ', '_')}.json"

            if industry_file.exists():
                # Load benchmarks
                with open(industry_file, 'r') as f:
                    return json.load(f)

            # If no file, return empty dict
            return {}
        except Exception as e:
            logger.error(f"Error loading industry benchmarks for {industry}: {e}")
            return {}


    def _get_default_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get default benchmarks for generic companies"""
        return {
            "valuation": {
                "pe_ratio": 20.0,
                "ps_ratio": 2.5,
                "pb_ratio": 3.0,
                "ev_ebitda": 12.0
            },
            "profitability": {
                "gross_margin": 0.40,
                "operating_margin": 0.15,
                "net_margin": 0.10,
                "roe": 0.15,
                "roa": 0.08
            },
            "liquidity": {
                "current_ratio": 1.8,
                "quick_ratio": 1.5,
                "cash_ratio": 0.8
            },
            "leverage": {
                "debt_to_equity": 0.7,
                "interest_coverage": 8.0,
                "debt_to_assets": 0.3
            },
            "efficiency": {
                "asset_turnover": 0.9,
                "inventory_turnover": 5.0,
                "receivables_turnover": 7.0
            },
            "growth": {
                "revenue_growth": 0.05,
                "net_income_growth": 0.05
            }
        }


    def _get_default_sectors_benchmarks(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get default benchmarks for all sectors"""
        return {
            "Technology": {
                "valuation": {
                    "pe_ratio": 30.0,
                    "ps_ratio": 5.0,
                    "pb_ratio": 6.0,
                    "ev_ebitda": 18.0,
                    "ev_revenue": 5.0
                },
                "profitability": {
                    "gross_margin": 0.60,
                    "operating_margin": 0.25,
                    "net_margin": 0.20,
                    "roe": 0.22,
                    "roa": 0.15
                },
                "liquidity": {
                    "current_ratio": 2.5,
                    "quick_ratio": 2.0,
                    "cash_ratio": 1.5
                },
                "leverage": {
                    "debt_to_equity": 0.5,
                    "interest_coverage": 15.0,
                    "debt_to_assets": 0.2
                },
                "efficiency": {
                    "asset_turnover": 0.7,
                    "rnd_to_revenue": 0.15
                },
                "growth": {
                    "revenue_growth": 0.15,
                    "net_income_growth": 0.18
                }
            },
            "Healthcare": {
                "valuation": {
                    "pe_ratio": 25.0,
                    "ps_ratio": 4.0,
                    "pb_ratio": 4.0,
                    "ev_ebitda": 15.0
                },
                "profitability": {
                    "gross_margin": 0.65,
                    "operating_margin": 0.18,
                    "net_margin": 0.15,
                    "roe": 0.18,
                    "roa": 0.10
                },
                "liquidity": {
                    "current_ratio": 2.0,
                    "quick_ratio": 1.7,
                    "cash_ratio": 1.2
                },
                "leverage": {
                    "debt_to_equity": 0.6,
                    "interest_coverage": 12.0,
                    "debt_to_assets": 0.25
                },
                "efficiency": {
                    "asset_turnover": 0.5,
                    "rnd_to_revenue": 0.12
                },
                "growth": {
                    "revenue_growth": 0.08,
                    "net_income_growth": 0.10
                }
            },
            "Financials": {
                "valuation": {
                    "pe_ratio": 14.0,
                    "pb_ratio": 1.2,
                    "ps_ratio": 3.0
                },
                "profitability": {
                    "net_margin": 0.25,
                    "roe": 0.12,
                    "roa": 0.01
                },
                "liquidity": {
                    "current_ratio": 1.2
                },
                "leverage": {
                    "debt_to_equity": 1.5
                },
                "efficiency": {
                    "asset_turnover": 0.05
                },
                "growth": {
                    "revenue_growth": 0.05,
                    "net_income_growth": 0.06
                }
            },
            "Consumer Discretionary": {
                "valuation": {
                    "pe_ratio": 22.0,
                    "ps_ratio": 1.5,
                    "pb_ratio": 3.5,
                    "ev_ebitda": 12.0
                },
                "profitability": {
                    "gross_margin": 0.35,
                    "operating_margin": 0.12,
                    "net_margin": 0.08,
                    "roe": 0.15,
                    "roa": 0.08
                },
                "liquidity": {
                    "current_ratio": 1.8,
                    "quick_ratio": 0.9,
                    "cash_ratio": 0.5
                },
                "leverage": {
                    "debt_to_equity": 0.8,
                    "interest_coverage": 8.0,
                    "debt_to_assets": 0.35
                },
                "efficiency": {
                    "asset_turnover": 1.2,
                    "inventory_turnover": 6.0
                },
                "growth": {
                    "revenue_growth": 0.06,
                    "net_income_growth": 0.07
                }
            },
            "Consumer Staples": {
                "valuation": {
                    "pe_ratio": 22.0,
                    "ps_ratio": 1.8,
                    "pb_ratio": 5.0,
                    "ev_ebitda": 14.0
                },
                "profitability": {
                    "gross_margin": 0.40,
                    "operating_margin": 0.15,
                    "net_margin": 0.12,
                    "roe": 0.20,
                    "roa": 0.09
                },
                "liquidity": {
                    "current_ratio": 1.5,
                    "quick_ratio": 0.8,
                    "cash_ratio": 0.4
                },
                "leverage": {
                    "debt_to_equity": 1.0,
                    "interest_coverage": 10.0,
                    "debt_to_assets": 0.4
                },
                "efficiency": {
                    "asset_turnover": 1.0,
                    "inventory_turnover": 7.0
                },
                "growth": {
                    "revenue_growth": 0.04,
                    "net_income_growth": 0.05
                }
            },
            "Energy": {
                "valuation": {
                    "pe_ratio": 15.0,
                    "ps_ratio": 1.0,
                    "pb_ratio": 1.5,
                    "ev_ebitda": 6.0
                },
                "profitability": {
                    "gross_margin": 0.30,
                    "operating_margin": 0.15,
                    "net_margin": 0.10,
                    "roe": 0.10,
                    "roa": 0.06
                },
                "liquidity": {
                    "current_ratio": 1.5,
                    "quick_ratio": 1.2,
                    "cash_ratio": 0.6
                },
                "leverage": {
                    "debt_to_equity": 0.4,
                    "interest_coverage": 10.0,
                    "debt_to_assets": 0.2
                },
                "efficiency": {
                    "asset_turnover": 0.4
                },
                "growth": {
                    "revenue_growth": 0.03,
                    "net_income_growth": 0.03
                }
            },
            "Industrials": {
                "valuation": {
                    "pe_ratio": 20.0,
                    "ps_ratio": 1.5,
                    "pb_ratio": 3.0,
                    "ev_ebitda": 10.0
                },
                "profitability": {
                    "gross_margin": 0.30,
                    "operating_margin": 0.12,
                    "net_margin": 0.08,
                    "roe": 0.14,
                    "roa": 0.07
                },
                "liquidity": {
                    "current_ratio": 1.7,
                    "quick_ratio": 1.2,
                    "cash_ratio": 0.5
                },
                "leverage": {
                    "debt_to_equity": 0.7,
                    "interest_coverage": 12.0,
                    "debt_to_assets": 0.3
                },
                "efficiency": {
                    "asset_turnover": 0.8,
                    "inventory_turnover": 5.0
                },
                "growth": {
                    "revenue_growth": 0.05,
                    "net_income_growth": 0.06
                }
            },
            "Materials": {
                "valuation": {
                    "pe_ratio": 18.0,
                    "ps_ratio": 1.3,
                    "pb_ratio": 2.0,
                    "ev_ebitda": 8.0
                },
                "profitability": {
                    "gross_margin": 0.25,
                    "operating_margin": 0.10,
                    "net_margin": 0.08,
                    "roe": 0.12,
                    "roa": 0.06
                },
                "liquidity": {
                    "current_ratio": 1.8,
                    "quick_ratio": 1.3,
                    "cash_ratio": 0.6
                },
                "leverage": {
                    "debt_to_equity": 0.6,
                    "interest_coverage": 9.0,
                    "debt_to_assets": 0.25
                },
                "efficiency": {
                    "asset_turnover": 0.7,
                    "inventory_turnover": 6.0
                },
                "growth": {
                    "revenue_growth": 0.04,
                    "net_income_growth": 0.04
                }
            },
            "Real Estate": {
                "valuation": {
                    "pe_ratio": 18.0,
                    "ps_ratio": 5.0,
                    "pb_ratio": 1.8,
                    "ev_ebitda": 15.0
                },
                "profitability": {
                    "gross_margin": 0.60,
                    "operating_margin": 0.30,
                    "net_margin": 0.20,
                    "roe": 0.10,
                    "roa": 0.04
                },
                "liquidity": {
                    "current_ratio": 2.0,
                    "quick_ratio": 1.8,
                    "cash_ratio": 0.7
                },
                "leverage": {
                    "debt_to_equity": 1.2,
                    "interest_coverage": 3.0,
                    "debt_to_assets": 0.5
                },
                "efficiency": {
                    "asset_turnover": 0.15
                },
                "growth": {
                    "revenue_growth": 0.04,
                    "net_income_growth": 0.05
                }
            },
            "Communication Services": {
                "valuation": {
                    "pe_ratio": 20.0,
                    "ps_ratio": 2.5,
                    "pb_ratio": 3.0,
                    "ev_ebitda": 9.0
                },
                "profitability": {
                    "gross_margin": 0.50,
                    "operating_margin": 0.20,
                    "net_margin": 0.15,
                    "roe": 0.15,
                    "roa": 0.08
                },
                "liquidity": {
                    "current_ratio": 1.6,
                    "quick_ratio": 1.4,
                    "cash_ratio": 0.8
                },
                "leverage": {
                    "debt_to_equity": 0.9,
                    "interest_coverage": 8.0,
                    "debt_to_assets": 0.4
                },
                "efficiency": {
                    "asset_turnover": 0.6
                },
                "growth": {
                    "revenue_growth": 0.07,
                    "net_income_growth": 0.08
                }
            },
            "Utilities": {
                "valuation": {
                    "pe_ratio": 18.0,
                    "ps_ratio": 2.0,
                    "pb_ratio": 1.8,
                    "ev_ebitda": 12.0
                },
                "profitability": {
                    "gross_margin": 0.45,
                    "operating_margin": 0.20,
                    "net_margin": 0.12,
                    "roe": 0.10,
                    "roa": 0.04
                },
                "liquidity": {
                    "current_ratio": 1.0,
                    "quick_ratio": 0.8,
                    "cash_ratio": 0.3
                },
                "leverage": {
                    "debt_to_equity": 1.5,
                    "interest_coverage": 3.5,
                    "debt_to_assets": 0.6
                },
                "efficiency": {
                    "asset_turnover": 0.3
                },
                "growth": {
                    "revenue_growth": 0.03,
                    "net_income_growth": 0.04
                }
            }
        }

    def _get_sample_sector_performance(self) -> pd.DataFrame:
        """Get sample sector performance data for UI when real data is not available"""
        # Sample sector performance data
        data = {
            'Sector': [
                'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
                'Consumer Staples', 'Energy', 'Industrials', 'Materials',
                'Real Estate', 'Communication Services', 'Utilities'
            ],
            '1D': [0.5, -0.2, 0.3, -0.1, 0.2, 1.2, 0.4, 0.8, -0.5, 0.1, -0.3],
            '1W': [2.1, 1.5, -0.8, 1.2, 0.5, 3.2, 1.1, 2.3, -1.2, 0.7, -0.9],
            '1M': [5.3, 3.2, 2.1, 4.5, 1.8, 7.5, 3.5, 4.8, -3.5, 2.2, 1.5],
            '3M': [12.5, 7.8, 5.2, 10.3, 4.5, 15.2, 8.2, 9.5, -5.2, 6.5, 3.8],
            '1Y': [25.3, 15.2, 10.5, 18.7, 8.2, -12.5, 14.5, 17.2, -8.5, 12.8, 7.5],
            'YTD': [15.2, 8.5, 6.2, 11.2, 5.3, 9.8, 7.5, 10.2, -4.8, 7.2, 4.5]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add date column
        df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

        return df