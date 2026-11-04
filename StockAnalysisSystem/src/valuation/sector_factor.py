"""
Valuation model factory.

Maps a company's sector (and, where it matters, industry) to the specialised
valuation model and exposes a single ``get_company_valuation`` entry point
that returns the standardised result documented in
``BaseValuation.build_valuation_result``.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type

# Add the repository root to the Python path when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import normalize_sector
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.sector_specific.communication_sector import CommunicationValuation
from StockAnalysisSystem.src.valuation.sector_specific.energy_sector import EnergySectorValuation
from StockAnalysisSystem.src.valuation.sector_specific.financial_sector import FinancialSectorValuation
from StockAnalysisSystem.src.valuation.sector_specific.healthcare import HealthcareValuation
from StockAnalysisSystem.src.valuation.sector_specific.manufacturing import ManufacturingSectorValuation
from StockAnalysisSystem.src.valuation.sector_specific.real_estate import RealEstateValuation
from StockAnalysisSystem.src.valuation.sector_specific.retail_sector import RetailSectorValuation
from StockAnalysisSystem.src.valuation.sector_specific.tech_sector import TechnologySectorValuation

logger = logging.getLogger('sector_factory')

# Backward-compatible aliases for the names used by older code.
FinancialValuation = FinancialSectorValuation
TechValuation = TechnologySectorValuation
EnergyValuation = EnergySectorValuation
RetailValuation = RetailSectorValuation
ManufacturingValuation = ManufacturingSectorValuation

#: Model registry: key -> class
MODEL_CLASSES: Dict[str, Type[BaseValuation]] = {
    "Base": BaseValuation,
    "Financial": FinancialSectorValuation,
    "Technology": TechnologySectorValuation,
    "Energy": EnergySectorValuation,
    "Retail": RetailSectorValuation,
    "Manufacturing": ManufacturingSectorValuation,
    "RealEstate": RealEstateValuation,
    "Healthcare": HealthcareValuation,
    "Communication": CommunicationValuation,
}

#: Canonical sector (config.SECTOR_MAPPING key) -> model key
SECTOR_MODEL_MAP: Dict[str, str] = {
    "Financials": "Financial",
    "Technology": "Technology",
    "Energy": "Energy",
    "Utilities": "Energy",
    "Consumer Discretionary": "Retail",
    "Consumer Staples": "Retail",
    "Industrials": "Manufacturing",
    "Materials": "Manufacturing",
    "Real Estate": "RealEstate",
    "Healthcare": "Healthcare",
    "Communication Services": "Communication",
}

#: Industry keywords that override the sector mapping (checked case-insensitively)
INDUSTRY_MODEL_OVERRIDES: Dict[str, str] = {
    "auto manufacturers": "Manufacturing",
    "auto parts": "Manufacturing",
    "automobiles": "Manufacturing",
    "reit": "RealEstate",
}

#: Default subsector for sectors that share a model with another sector
SECTOR_DEFAULT_SUBSECTOR: Dict[str, str] = {
    "Utilities": "utilities",
}


class ValuationFactory:
    """
    Factory class for creating the appropriate valuation model for a company.

    Model instances are created lazily and cached, so constructing the factory
    is cheap.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader if data_loader else DataLoader()
        self._instances: Dict[str, BaseValuation] = {}

        # Kept for backward compatibility with code that reads these attributes
        self.sector_mapping = dict(SECTOR_MODEL_MAP)

    # ------------------------------------------------------------------
    # Model lookup
    # ------------------------------------------------------------------
    @property
    def valuation_models(self) -> Dict[str, BaseValuation]:
        """All model instances keyed by model key (instantiates them)."""
        return {key: self._get_instance(key) for key in MODEL_CLASSES}

    def _get_instance(self, model_key: str) -> BaseValuation:
        if model_key not in self._instances:
            self._instances[model_key] = MODEL_CLASSES[model_key](self.data_loader)
        return self._instances[model_key]

    def get_model_key(self, sector: Optional[str], industry: Optional[str] = None) -> str:
        """Return the registry key of the model for a sector / industry."""
        if industry:
            industry_lower = str(industry).lower()
            for keyword, key in INDUSTRY_MODEL_OVERRIDES.items():
                if keyword in industry_lower:
                    return key
        return SECTOR_MODEL_MAP.get(normalize_sector(sector), "Base")

    def get_valuation_model(self, sector: Optional[str], industry: Optional[str] = None) -> BaseValuation:
        """Get the specialised valuation model for a sector (Base if unknown)."""
        return self._get_instance(self.get_model_key(sector, industry))

    def get_subsectors(self, sector: Optional[str], industry: Optional[str] = None) -> Dict[str, str]:
        """Subsector options ({key: label}) of the model used for ``sector``."""
        return dict(self.get_valuation_model(sector, industry).SUBSECTORS)

    def get_default_subsector(self, ticker: str, sector: Optional[str],
                              financial_data: Optional[Dict[str, Any]] = None,
                              company_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Best guess of the subsector key for a company."""
        canonical = normalize_sector(sector)
        if canonical in SECTOR_DEFAULT_SUBSECTOR:
            return SECTOR_DEFAULT_SUBSECTOR[canonical]
        industry = (company_info or {}).get('industry')
        model = self.get_valuation_model(sector, industry)
        try:
            return model.detect_subsector(ticker, financial_data, company_info)
        except Exception as e:  # detection is best effort only
            logger.warning(f"Subsector detection failed for {ticker}: {e}")
            return next(iter(model.SUBSECTORS), None)

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def get_company_valuation(self, ticker: str, sector: Optional[str] = None,
                              subsector: Optional[str] = None,
                              financial_data: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Value a company with the model appropriate for its sector.

        Args:
            ticker: Company ticker symbol.
            sector: Sector name (any provider spelling). Looked up when omitted.
            subsector: Subsector key of the selected model (see get_subsectors).
            financial_data: Canonical financial data bundle; loaded when omitted.
            **kwargs: Passed through to the model's ``get_valuation``.

        Returns:
            Standardised valuation result (see BaseValuation.build_valuation_result).
        """
        company_info = (financial_data or {}).get('company_info') or {}
        if sector is None:
            sector = company_info.get('sector') or self._determine_company_sector(ticker)
        canonical = normalize_sector(sector)
        model = self.get_valuation_model(canonical, company_info.get('industry'))

        if subsector is None and canonical in SECTOR_DEFAULT_SUBSECTOR:
            subsector = SECTOR_DEFAULT_SUBSECTOR[canonical]

        logger.info(f"Using {model.__class__.__name__} for {ticker} ({canonical})")
        try:
            result = model.get_valuation(ticker, financial_data=financial_data, subsector=subsector,
                                         sector=canonical, **kwargs)
        except Exception as e:
            logger.error(f"Error in valuation for {ticker}: {e}")
            result = model.build_error_result(ticker, e, sector=canonical, subsector=subsector)

        result.setdefault('sector', canonical)
        result.setdefault('model', model.__class__.__name__)
        return result

    def _normalize_sector(self, sector: Optional[str]) -> str:
        """Backward-compatible wrapper around config.normalize_sector."""
        return normalize_sector(sector)

    def _determine_company_sector(self, ticker: str) -> str:
        """Determine company sector from the data loader (``Unknown`` on failure)."""
        try:
            company_info = self.data_loader.get_company_info(ticker) or {}
            return normalize_sector(company_info.get("sector"))
        except Exception as e:
            logger.error(f"Error determining sector for {ticker}: {e}")
            return "Unknown"

    def get_comparable_companies(self, ticker: str, sector: Optional[str] = None,
                                 count: int = 5) -> Dict[str, Any]:
        """
        Find comparable companies in the same sector.

        Returns:
            Dict with ``company``, ``sector`` and ``comparable_companies`` (list of
            records with ``symbol``, ``name``, ``sector``, ``subsector``).
        """
        try:
            from StockAnalysisSystem.src.industry.sector_mapping import get_sector_peers

            if sector is None:
                sector = self._determine_company_sector(ticker)
            peers: List[Dict[str, str]] = get_sector_peers(ticker, sector, limit=count)
            return {"company": ticker, "sector": normalize_sector(sector), "comparable_companies": peers}
        except Exception as e:
            logger.error(f"Error finding comparable companies for {ticker}: {e}")
            return {"company": ticker, "sector": normalize_sector(sector), "comparable_companies": [],
                    "error": str(e)}
