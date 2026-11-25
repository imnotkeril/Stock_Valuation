"""
Company -> sector / industry mapping.

Sector names are always the canonical names of ``config.SECTOR_MAPPING``
(see :func:`config.normalize_sector`). The company universe itself lives in
:mod:`StockAnalysisSystem.src.industry.universe`; this module only adds
lookups on top of it plus optional user overrides persisted as JSON in
``SECTOR_DATA_DIR`` (written lazily, only by :meth:`SectorMapper.add_company_mapping`).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from StockAnalysisSystem.src.config import SECTOR_DATA_DIR, SECTOR_MAPPING, normalize_sector
from StockAnalysisSystem.src.industry import universe

logger = logging.getLogger(__name__)

MAPPINGS_FILE = "company_sector_mappings.json"

# Fallback keywords for sector names that ``normalize_sector`` does not know
# (checked in order, so more specific sectors come first).
_SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "Real Estate": ["real estate", "reit", "realty", "property"],
    "Communication Services": ["communication", "telecom", "media", "entertainment", "interactive"],
    "Consumer Staples": ["staples", "consumer defensive", "food", "beverage", "household", "tobacco"],
    "Consumer Discretionary": ["discretionary", "consumer cyclical", "retail", "apparel", "automobile", "leisure"],
    "Financials": ["financ", "bank", "insurance", "capital markets", "asset management"],
    "Healthcare": ["health", "medical", "pharma", "biotech", "life science"],
    "Technology": ["technology", "information tech", "software", "semiconductor", "hardware", "it services"],
    "Utilities": ["utilit", "electric", "power", "water"],
    "Energy": ["energy", "oil", "gas", "petroleum", "coal"],
    "Materials": ["material", "chemical", "mining", "metal", "steel", "paper", "packaging"],
    "Industrials": ["industrial", "manufactur", "aerospace", "defense", "transport", "machinery", "construction"],
}

# Keywords used to map a free-text industry (e.g. yfinance "Software - Infrastructure")
# to a sector. Multi-word keywords are matched before single words.
_INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "Technology": ["software", "hardware", "semiconductor", "computer", "electronic", "it services",
                   "information technology", "internet software", "communication equipment", "consumer electronics"],
    "Healthcare": ["health", "medical", "pharma", "biotech", "drug", "hospital", "diagnostic", "life science"],
    "Financials": ["bank", "insurance", "asset management", "capital markets", "credit services", "financial",
                   "brokerage", "mortgage"],
    "Consumer Discretionary": ["retail", "auto", "luxury", "apparel", "restaurant", "hotel", "lodging", "leisure",
                               "travel", "home improvement", "footwear", "gambling", "resorts"],
    "Consumer Staples": ["food", "beverage", "household", "personal products", "grocery", "tobacco",
                         "discount stores", "farm products"],
    "Energy": ["oil", "gas", "petroleum", "coal", "drilling", "refining", "pipeline", "solar"],
    "Industrials": ["industrial", "machinery", "defense", "aerospace", "transport", "railroad", "trucking",
                    "airline", "construction", "engineering", "freight", "electrical equipment"],
    "Materials": ["material", "chemical", "metal", "mining", "steel", "paper", "gold", "silver", "copper",
                  "packaging", "aluminum", "lumber"],
    "Real Estate": ["real estate", "reit", "property", "development"],
    "Communication Services": ["communication services", "telecom", "media", "entertainment", "publishing",
                               "broadcasting", "advertising", "internet content", "electronic gaming"],
    "Utilities": ["utilit", "electric", "water", "regulated gas", "renewable", "independent power"],
}


def normalize_sector_name(sector: Optional[str]) -> str:
    """
    Canonical sector name for any provider spelling.

    Uses :func:`config.normalize_sector` first and falls back to keyword
    matching (e.g. ``"Banks"`` -> ``"Financials"``). Unmatched names are
    returned stripped; missing names become ``"Unknown"``.
    """
    canonical = normalize_sector(sector)
    if canonical == "Unknown" or canonical in SECTOR_MAPPING:
        return canonical
    lowered = canonical.lower()
    for name, subsectors in SECTOR_MAPPING.items():
        if any(lowered == s.lower() for s in subsectors):
            return name
    for name, keywords in _SECTOR_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return name
    return canonical


def map_industry_to_sector(industry: Optional[str]) -> str:
    """Map a free-text industry name to a canonical sector (``"Unknown"`` if no match)."""
    if not industry:
        return "Unknown"
    lowered = str(industry).strip().lower()
    for sector, subsectors in SECTOR_MAPPING.items():
        if any(lowered == s.lower() for s in subsectors):
            return sector
    best, best_len = "Unknown", 0
    for sector, keywords in _INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered and len(keyword) > best_len:
                best, best_len = sector, len(keyword)
    return best


class SectorMapper:
    """Resolve the sector and industry of a company."""

    def __init__(self, data_dir: str = SECTOR_DATA_DIR):
        """
        Args:
            data_dir: Directory holding optional user overrides
                (``company_sector_mappings.json``). Nothing is written on init.
        """
        self.data_dir = Path(data_dir)
        self.sector_mapping = SECTOR_MAPPING
        self._company_to_sector: Optional[Dict[str, Dict[str, str]]] = None

    @property
    def company_to_sector(self) -> Dict[str, Dict[str, str]]:
        """User overrides ``{ticker: {"sector", "industry"}}``, loaded lazily."""
        if self._company_to_sector is None:
            self._company_to_sector = self._load_company_mappings()
        return self._company_to_sector

    def get_sector(self, ticker: str, company_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Canonical sector of a company.

        Resolution order: user override, ``company_info["sector"]``, curated
        universe, ``company_info["industry"]``. Returns ``"Unknown"`` otherwise.
        """
        ticker = (ticker or "").upper()
        override = self.company_to_sector.get(ticker)
        if override and override.get("sector"):
            return normalize_sector_name(override["sector"])

        info = company_info or {}
        sector = normalize_sector_name(info.get("sector"))
        if sector != "Unknown":
            return sector

        record = universe.lookup_company(ticker)
        if record:
            return record["sector"]

        return map_industry_to_sector(info.get("industry"))

    def get_industry(self, ticker: str, company_info: Optional[Dict[str, Any]] = None) -> str:
        """Industry of a company: override, ``company_info["industry"]``, universe subsector or ``"Unknown"``."""
        ticker = (ticker or "").upper()
        override = self.company_to_sector.get(ticker)
        if override and override.get("industry"):
            return override["industry"]
        if company_info and company_info.get("industry"):
            return str(company_info["industry"])
        record = universe.lookup_company(ticker)
        return record["subsector"] if record else "Unknown"

    def get_peer_companies(self, ticker: str, sector: str, max_peers: int = 5) -> List[str]:
        """Peer tickers from the curated universe (same subsector first, excluding ``ticker``)."""
        peers = universe.get_sector_peers(ticker, normalize_sector_name(sector), limit=max_peers)
        return [peer["symbol"] for peer in peers]

    def add_company_mapping(self, ticker: str, sector: str, industry: str) -> bool:
        """Add or update a user override and persist it (best effort). Returns True on success."""
        ticker = (ticker or "").upper()
        if not ticker:
            return False
        self.company_to_sector[ticker] = {"sector": normalize_sector_name(sector), "industry": industry}
        return self._save_company_mappings()

    def get_sectors(self) -> List[str]:
        """Canonical sector names."""
        return list(self.sector_mapping.keys())

    def get_industries_for_sector(self, sector: str) -> List[str]:
        """Industries (``config.SECTOR_MAPPING`` subsectors) of a sector; empty list if unknown."""
        return list(self.sector_mapping.get(normalize_sector_name(sector), []))

    # ------------------------------------------------------------------ helpers
    def _normalize_sector_name(self, sector: Optional[str]) -> str:
        return normalize_sector_name(sector)

    def _map_industry_to_sector(self, industry: Optional[str]) -> str:
        return map_industry_to_sector(industry)

    def _load_company_mappings(self) -> Dict[str, Dict[str, str]]:
        mapping_file = self.data_dir / MAPPINGS_FILE
        if not mapping_file.exists():
            return {}
        try:
            data = json.loads(mapping_file.read_text())
            return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {mapping_file}: {e}")
            return {}

    def _save_company_mappings(self) -> bool:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / MAPPINGS_FILE).write_text(json.dumps(self.company_to_sector, indent=2))
            return True
        except OSError as e:
            logger.warning(f"Could not save company mappings: {e}")
            return False

# ---------------------------------------------------------------------------
# Module-level helpers (stable public API used by the UI pages)
# ---------------------------------------------------------------------------
from StockAnalysisSystem.src.industry.universe import (  # noqa: E402
    get_sector_companies,
    get_sector_peers,
    get_subsectors,
)

__all__ = ["SectorMapper", "get_sector_companies", "get_sector_peers", "get_subsectors",
           "normalize_sector_name", "map_industry_to_sector"]
