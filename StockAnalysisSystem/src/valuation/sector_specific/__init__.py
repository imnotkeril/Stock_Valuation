"""Importing the package registers every sector analyzer with SectorFactory."""
from StockAnalysisSystem.src.valuation.sector_specific import (  # noqa: F401
    communication_sector, energy_sector, financial_sector, healthcare, manufacturing,
    materials_sector, real_estate, retail_sector, tech_sector, utilities_sector,
)
