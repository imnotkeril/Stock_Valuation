import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

# Package imports use the `StockAnalysisSystem.src...` prefix, so the project's parent must be importable.
PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

logging.disable(logging.CRITICAL)

FIXTURES = Path(__file__).parent / "fixtures"


def _read_statement(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / f"aapl_{name}.csv", index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df


class FakeLoader:
    """Offline stand-in for DataLoader serving recorded AAPL annual statements."""

    def __init__(self):
        self._statements = {k: _read_statement(k) for k in ("income", "balance", "cash")}
        self._info = json.loads((FIXTURES / "aapl_info.json").read_text(encoding="utf-8"))

    def get_financial_statements(self, ticker, statement_type="income", period="annual", force_refresh=False):
        return self._statements[statement_type].copy()

    def get_company_info(self, ticker, provider="yfinance"):
        return dict(self._info)


@pytest.fixture
def fake_loader():
    return FakeLoader()


@pytest.fixture
def financial_data(fake_loader):
    info = fake_loader.get_company_info("AAPL")
    return {
        "income_statement": fake_loader.get_financial_statements("AAPL", "income"),
        "balance_sheet": fake_loader.get_financial_statements("AAPL", "balance"),
        "cash_flow": fake_loader.get_financial_statements("AAPL", "cash"),
        "company_info": info,
        "market_data": {
            "market_cap": info["market_cap"],
            "share_price": info["current_price"],
            "shares_outstanding": info["shares_outstanding"],
            "dividend_yield": info.get("dividend_yield"),
            "beta": info.get("beta"),
        },
    }
