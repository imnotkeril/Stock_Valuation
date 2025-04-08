import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation

# Import sector-specific valuation models
from StockAnalysisSystem.src.valuation.sector_specific.financial_sector import FinancialValuation
from StockAnalysisSystem.src.valuation.sector_specific.tech_sector import TechValuation
from StockAnalysisSystem.src.valuation.sector_specific.energy_sector import EnergyValuation
from StockAnalysisSystem.src.valuation.sector_specific.retail_sector import RetailValuation
from StockAnalysisSystem.src.valuation.sector_specific.manufacturing import ManufacturingValuation
from StockAnalysisSystem.src.valuation.sector_specific.real_estate import RealEstateValuation
from StockAnalysisSystem.src.valuation.sector_specific.healthcare import HealthcareValuation
from StockAnalysisSystem.src.valuation.sector_specific.communication_sector import CommunicationValuation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_factory')


class ValuationFactory:
    """
    Factory class for creating appropriate valuation models based on
    company sector and characteristics.

    This class serves as a central hub for determining which specialized
    valuation model should be used for a given company.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the valuation factory.

        Args:
            data_loader: Optional data loader to use for all valuation models
        """
        self.data_loader = data_loader if data_loader else DataLoader()

        # Initialize sector mapping
        self.sector_mapping = {
            # Map GICS sectors and subsectors to our valuation models
            "Financials": "Financial",
            "Banks": "Financial",
            "Insurance": "Financial",
            "Diversified Financials": "Financial",
            "Financial Services": "Financial",

            "Information Technology": "Technology",
            "Software": "Technology",
            "IT Services": "Technology",
            "Technology Hardware": "Technology",
            "Semiconductors": "Technology",

            "Energy": "Energy",
            "Oil & Gas": "Energy",
            "Oil, Gas & Consumable Fuels": "Energy",
            "Energy Equipment & Services": "Energy",

            "Consumer Discretionary": "Retail",
            "Retailing": "Retail",
            "Consumer Services": "Retail",
            "Consumer Durables & Apparel": "Retail",
            "Automobiles & Components": "Manufacturing",

            "Consumer Staples": "Retail",
            "Food & Staples Retailing": "Retail",
            "Food, Beverage & Tobacco": "Retail",
            "Household & Personal Products": "Retail",

            "Industrials": "Manufacturing",
            "Capital Goods": "Manufacturing",
            "Commercial & Professional Services": "Manufacturing",
            "Transportation": "Manufacturing",

            "Materials": "Manufacturing",
            "Chemicals": "Manufacturing",
            "Construction Materials": "Manufacturing",
            "Metals & Mining": "Manufacturing",

            "Real Estate": "RealEstate",
            "REIT": "RealEstate",
            "Real Estate Management & Development": "RealEstate",

            "Health Care": "Healthcare",
            "Healthcare": "Healthcare",
            "Pharmaceuticals": "Healthcare",
            "Biotechnology": "Healthcare",
            "Healthcare Equipment & Services": "Healthcare",

            "Communication Services": "Communication",
            "Telecommunication Services": "Communication",
            "Media & Entertainment": "Communication",
            "Interactive Media & Services": "Communication"
        }

        # Initialize valuation model instances
        self.valuation_models = {
            "Base": BaseValuation(self.data_loader),
            "Financial": FinancialValuation(self.data_loader),
            "Technology": TechValuation(self.data_loader),
            "Energy": EnergyValuation(self.data_loader),
            "Retail": RetailValuation(self.data_loader),
            "Manufacturing": ManufacturingValuation(self.data_loader),
            "RealEstate": RealEstateValuation(self.data_loader),
            "Healthcare": HealthcareValuation(self.data_loader),
            "Communication": CommunicationValuation(self.data_loader)
        }

    def get_valuation_model(self, sector: str) -> BaseValuation:
        """
        Get the appropriate valuation model for a given sector.

        Args:
            sector: Company sector name

        Returns:
            Specialized valuation model instance
        """
        # Normalize sector name
        normalized_sector = self._normalize_sector(sector)

        # Map sector to valuation model
        model_key = self.sector_mapping.get(normalized_sector, "Base")

        # Get the model or fall back to base valuation
        return self.valuation_models.get(model_key, self.valuation_models["Base"])

    def get_company_valuation(self, ticker: str, sector: Optional[str] = None,
                              subsector: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive valuation for a company using the appropriate model.

        Args:
            ticker: Company ticker symbol
            sector: Company sector (optional, will be determined if not provided)
            subsector: Company subsector for more specialized valuation
            **kwargs: Additional parameters for specific valuation models

        Returns:
            Dictionary with valuation results
        """
        try:
            # If sector not provided, try to determine it
            if sector is None:
                sector = self._determine_company_sector(ticker)

            # Get appropriate valuation model
            valuation_model = self.get_valuation_model(sector)

            # Log which model is being used
            logger.info(f"Using {valuation_model.__class__.__name__} for {ticker} ({sector})")

            # Perform valuation with model-specific parameters
            if isinstance(valuation_model, FinancialValuation):
                # Special handling for financial sector
                bank_type = kwargs.get("bank_type", "Commercial")
                return valuation_model.get_valuation(ticker, bank_type=bank_type)

            elif isinstance(valuation_model, TechValuation):
                # Special handling for tech sector
                tech_type = kwargs.get("tech_type", "SaaS")
                return valuation_model.get_valuation(ticker, tech_type=tech_type)

            elif isinstance(valuation_model, EnergyValuation):
                # Special handling for energy sector
                energy_type = kwargs.get("energy_type", "Integrated")
                return valuation_model.get_valuation(ticker, energy_type=energy_type)

            elif isinstance(valuation_model, RetailValuation):
                # Special handling for retail sector
                retail_type = kwargs.get("retail_type", "General")
                return valuation_model.get_valuation(ticker, retail_type=retail_type)

            elif isinstance(valuation_model, ManufacturingValuation):
                # Special handling for manufacturing sector
                industry = kwargs.get("industry", "General")
                return valuation_model.get_valuation(ticker, industry=industry)

            elif isinstance(valuation_model, RealEstateValuation):
                # Special handling for real estate sector
                property_type = kwargs.get("property_type", "Mixed")
                return valuation_model.get_valuation(ticker, property_type=property_type)

            elif isinstance(valuation_model, HealthcareValuation):
                # Special handling for healthcare sector
                healthcare_subsector = subsector or kwargs.get("subsector", "Pharmaceuticals")
                pipeline_data = kwargs.get("pipeline_data")
                return valuation_model.get_valuation(ticker, subsector=healthcare_subsector,
                                                     pipeline_data=pipeline_data)

            elif isinstance(valuation_model, CommunicationValuation):
                # Special handling for communication sector
                comm_subsector = subsector or kwargs.get("subsector", "Telecommunications")
                user_metrics = kwargs.get("user_metrics")
                return valuation_model.get_valuation(ticker, subsector=comm_subsector, user_metrics=user_metrics)

            else:
                # Default valuation
                return valuation_model.calculate_intrinsic_value(ticker)

        except Exception as e:
            logger.error(f"Error in valuation for {ticker}: {e}")
            return {
                "company": ticker,
                "error": str(e),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def _normalize_sector(self, sector: str) -> str:
        """
        Normalize sector name for consistent mapping.

        Args:
            sector: Original sector name

        Returns:
            Normalized sector name
        """
        if sector is None:
            return ""

        # Convert to string, strip whitespace, and capitalize
        normalized = str(sector).strip()

        # Return original if it's a direct match
        if normalized in self.sector_mapping:
            return normalized

        # Try case-insensitive matching
        for key in self.sector_mapping:
            if key.lower() == normalized.lower():
                return key

        # Return original if no match found
        return normalized

    def _determine_company_sector(self, ticker: str) -> str:
        """
        Determine company sector from ticker symbol.

        Args:
            ticker: Company ticker symbol

        Returns:
            Determined sector name
        """
        try:
            # Try to get company info from data loader
            company_info = self.data_loader.get_company_info(ticker)

            # Extract sector information
            sector = company_info.get("sector")

            if sector:
                return sector

            # If no sector found, fall back to default
            logger.warning(f"Could not determine sector for {ticker}, using default")
            return "Information Technology"  # Default sector

        except Exception as e:
            logger.error(f"Error determining sector for {ticker}: {e}")
            return "Information Technology"  # Default sector

    def get_comparable_companies(self, ticker: str, sector: Optional[str] = None,
                                 count: int = 5) -> Dict[str, Any]:
        """
        Find comparable companies in the same sector.

        Args:
            ticker: Company ticker symbol
            sector: Company sector (optional, will be determined if not provided)
            count: Number of comparable companies to return

        Returns:
            Dictionary with comparable companies and their key metrics
        """
        try:
            # If sector not provided, try to determine it
            if sector is None:
                sector = self._determine_company_sector(ticker)

            # Get company info
            company_info = self.data_loader.get_company_info(ticker)

            # TODO: Implement peer company search logic
            # This would involve searching for companies in the same sector
            # with similar market cap, revenue, etc.

            # Placeholder for now
            return {
                "company": ticker,
                "sector": sector,
                "comparable_companies": [],
                "message": "Comparable companies search not yet implemented"
            }

        except Exception as e:
            logger.error(f"Error finding comparable companies for {ticker}: {e}")
            return {
                "company": ticker,
                "error": str(e)
            }