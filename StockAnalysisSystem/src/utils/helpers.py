"""UI-independent helpers shared by the Streamlit pages."""
from typing import Any, Dict, Optional

import pandas as pd

# yfinance sector names -> names used by SECTOR_DCF_PARAMETERS and the SectorFactory
YF_SECTOR_TO_MODEL_SECTOR = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


def normalize_sector(sector: Optional[str]) -> Optional[str]:
    """Map a provider sector name to the model sector name."""
    if not sector:
        return None
    return YF_SECTOR_TO_MODEL_SECTOR.get(sector, sector)


def load_financial_data(loader, ticker: str) -> Dict[str, Any]:
    """Load annual statements and market data in the shape the analyzers expect."""
    info = loader.get_company_info(ticker) or {}
    return {
        "income_statement": loader.get_financial_statements(ticker, "income", "annual"),
        "balance_sheet": loader.get_financial_statements(ticker, "balance", "annual"),
        "cash_flow": loader.get_financial_statements(ticker, "cash", "annual"),
        "company_info": info,
        "market_data": {
            "market_cap": info.get("market_cap"),
            "share_price": info.get("current_price"),
            "shares_outstanding": info.get("shares_outstanding"),
            "dividend_yield": info.get("dividend_yield"),
            "beta": info.get("beta"),
        },
    }


def has_statements(financial_data: Dict[str, Any]) -> bool:
    """True when all three statements are non-empty DataFrames."""
    return all(
        isinstance(financial_data.get(k), pd.DataFrame) and not financial_data[k].empty
        for k in ("income_statement", "balance_sheet", "cash_flow")
    )


def flatten_ratios(ratios: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """{'category': {'ratio': value}} -> DataFrame(category, ratio, value)."""
    rows = [
        {"category": category, "ratio": name, "value": float(value)}
        for category, items in ratios.items()
        for name, value in items.items()
        if isinstance(value, (int, float)) and value == value
    ]
    return pd.DataFrame(rows, columns=["category", "ratio", "value"])


def format_money(value: Optional[float]) -> str:
    """Human-readable currency amount: 1.23T / 4.5B / 6.7M."""
    if value is None or value != value:
        return "n/a"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(value) >= size:
            return f"{value / size:,.2f}{unit}"
    return f"{value:,.2f}"
