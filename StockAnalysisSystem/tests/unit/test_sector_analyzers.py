import pytest

from StockAnalysisSystem.src.valuation import sector_specific  # noqa: F401  (registers analyzers)
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory

SECTORS = ["Technology", "Energy", "Healthcare", "Utilities", "Materials", "Real Estate",
           "Communication Services", "Financials", "Industrials", "Consumer Discretionary"]


@pytest.mark.parametrize("sector", SECTORS)
def test_analyzer_registered_and_runs(sector, financial_data):
    analyzer = SectorFactory.get_analyzer(sector)
    assert analyzer is not None
    result = analyzer.analyze_company(financial_data)
    assert isinstance(result, dict)
    assert not result.get("error")
