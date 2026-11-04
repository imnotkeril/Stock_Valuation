import os
from pathlib import Path
from typing import Dict, List, Optional

# Application paths
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to the root directory
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"
CACHE_DIR = DATA_DIR / "cache"
SECTOR_DATA_DIR = DATA_DIR / "sector_data"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, SECTOR_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Load variables from a local .env file when python-dotenv is available.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:  # pragma: no cover
    pass

# API keys are read from the environment only (see .env.example).
API_KEYS = {
    "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "financial_modeling_prep": os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", ""),
}

# Offline mode: never call external APIs, serve only cached / bundled data.
OFFLINE_MODE = os.environ.get("STOCK_ANALYSIS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}

# Cache settings
CACHE_EXPIRY_DAYS = 1  # Default cache expiry in days

# Sector mappings
SECTOR_MAPPING = {
    "Technology": ["Software", "Hardware", "Semiconductors", "Internet", "IT Services"],
    "Healthcare": ["Biotechnology", "Pharmaceuticals", "Medical Devices", "Healthcare Services"],
    "Financials": ["Banks", "Insurance", "Asset Management", "Diversified Financial Services"],
    "Consumer Discretionary": ["Retail", "Automotive", "Leisure", "Apparel", "Entertainment"],
    "Consumer Staples": ["Food & Beverage", "Household Products", "Personal Products"],
    "Energy": ["Oil & Gas", "Coal", "Renewable Energy"],
    "Industrials": ["Aerospace & Defense", "Construction", "Machinery", "Transportation"],
    "Materials": ["Chemicals", "Metals & Mining", "Paper & Forest Products"],
    "Real Estate": ["REIT", "Real Estate Development", "Real Estate Services"],
    "Communication Services": ["Telecom", "Media", "Entertainment"],
    "Utilities": ["Electric Utilities", "Gas Utilities", "Water Utilities", "Multi-Utilities"]
}

# Canonical sector names used across the application are the keys of
# SECTOR_MAPPING. Data providers use other spellings (yfinance: "Financial
# Services", "Consumer Cyclical", ...; GICS: "Information Technology", ...).
SECTORS: List[str] = list(SECTOR_MAPPING.keys())

SECTOR_ALIASES: Dict[str, str] = {
    # yfinance / Morningstar
    "technology": "Technology",
    "financial services": "Financials",
    "healthcare": "Healthcare",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
    "industrials": "Industrials",
    "energy": "Energy",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
    "utilities": "Utilities",
    # GICS and common variants
    "information technology": "Technology",
    "tech": "Technology",
    "health care": "Healthcare",
    "financials": "Financials",
    "financial": "Financials",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "materials": "Materials",
    "telecommunication services": "Communication Services",
    "telecommunications": "Communication Services",
    "communication": "Communication Services",
}


def normalize_sector(sector: Optional[str]) -> str:
    """
    Map any provider-specific sector name to the canonical name used in this
    application (a key of ``SECTOR_MAPPING``). Unknown values are returned
    stripped; missing values become ``"Unknown"``.
    """
    if sector is None:
        return "Unknown"
    name = str(sector).strip()
    if not name or name.upper() in {"N/A", "NA", "NONE", "UNKNOWN"}:
        return "Unknown"
    if name in SECTOR_MAPPING:
        return name
    return SECTOR_ALIASES.get(name.lower(), name)


# Financial ratios settings
RATIO_CATEGORIES = {
    "Valuation": ["P/E", "Forward P/E", "PEG", "P/S", "P/B", "EV/EBITDA", "EV/Revenue"],
    "Profitability": ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA", "ROIC"],
    "Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio"],
    "Leverage": ["Debt/Equity", "Debt/EBITDA", "Interest Coverage"],
    "Efficiency": ["Asset Turnover", "Inventory Turnover", "Receivables Turnover"]
}

# Sector-specific important ratios
SECTOR_SPECIFIC_RATIOS = {
    "Technology": ["P/S", "R&D/Revenue", "Gross Margin", "Operating Margin"],
    "Healthcare": ["P/E", "R&D/Revenue", "Operating Margin", "ROE"],
    "Financials": ["P/B", "ROE", "Net Interest Margin", "Efficiency Ratio"],
    "Energy": ["EV/EBITDA", "Reserve Replacement Ratio", "P/CF", "Debt/EBITDA"],
    "Consumer Discretionary": ["P/E", "Same-Store Sales Growth", "Inventory Turnover"],
    "Consumer Staples": ["EV/EBITDA", "Dividend Yield", "Operating Margin"],
    "Industrials": ["EV/EBITDA", "ROA", "ROIC", "Operating Margin"],
    "Materials": ["P/B", "EV/EBITDA", "ROE", "ROIC"],
    "Real Estate": ["FFO", "NAV", "Occupancy Rate", "Debt/Assets"],
    "Communication Services": ["P/E", "ARPU", "Churn Rate", "EBITDA Margin"],
    "Utilities": ["Dividend Yield", "P/E", "Debt/EBITDA", "Interest Coverage"]
}

# Color scheme for visualization (dark mode pastel colors)
COLORS = {
    "primary": "#74f174",  # Green
    "secondary": "#bf9ffb",  # Purple
    "accent": "#faa1a4",  # Pink/Red
    "warning": "#fff59d",  # Yellow
    "info": "#90bff9",  # Blue
    "danger": "#f48fb1",  # Pink
    "success": "#70ccbd",  # Teal

    # Sequential color palette for heatmaps
    "sequential": ["#081d58", "#253494", "#225ea8", "#1d91c0", "#41b6c4", "#7fcdbb", "#c7e9b4", "#edf8b1"],

    # Sector colors
    "sectors": {
        "Technology": "#74f174",
        "Healthcare": "#bf9ffb",
        "Financials": "#90bff9",
        "Consumer Discretionary": "#faa1a4",
        "Consumer Staples": "#f48fb1",
        "Energy": "#fcbe6e",
        "Industrials": "#91bea8",
        "Materials": "#70ccbd",
        "Real Estate": "#fff59d",
        "Communication Services": "#c582ff",
        "Utilities": "#b3df8a"
    }
}

# Default visualization settings
VIZ_SETTINGS = {
    "height": 600,
    "width": 800,
    "background": "#121212",
    "font_family": "Arial, sans-serif",
    "title_font_size": 18,
    "label_font_size": 14,
    "axis_font_size": 12,
    "text_color": "#e0e0e0",
    "grid_color": "#333333",
    "theme": "dark"
}

# Default risk-free rate for financial calculations
RISK_FREE_RATE = 0.03  # 3% annual risk-free rate

# Financial modeling parameters
DCF_PARAMETERS = {
    "forecast_years": 5,
    "terminal_growth_rate": 0.02,  # 2% long-term growth
    "default_discount_rate": 0.10,  # 10% discount rate
    "default_margin_of_safety": 0.25  # 25% margin of safety
}

# Sector-specific DCF parameters
SECTOR_DCF_PARAMETERS = {
    "Technology": {"forecast_years": 7, "terminal_growth_rate": 0.03, "default_discount_rate": 0.12},
    "Healthcare": {"forecast_years": 7, "terminal_growth_rate": 0.03, "default_discount_rate": 0.11},
    "Financials": {"forecast_years": 5, "terminal_growth_rate": 0.02, "default_discount_rate": 0.10},
    "Energy": {"forecast_years": 5, "terminal_growth_rate": 0.01, "default_discount_rate": 0.09},
    "Consumer Discretionary": {"forecast_years": 5, "terminal_growth_rate": 0.02, "default_discount_rate": 0.10},
    "Consumer Staples": {"forecast_years": 5, "terminal_growth_rate": 0.02, "default_discount_rate": 0.08},
    "Industrials": {"forecast_years": 5, "terminal_growth_rate": 0.02, "default_discount_rate": 0.09},
    "Materials": {"forecast_years": 5, "terminal_growth_rate": 0.01, "default_discount_rate": 0.09},
    "Real Estate": {"forecast_years": 7, "terminal_growth_rate": 0.02, "default_discount_rate": 0.08},
    "Communication Services": {"forecast_years": 6, "terminal_growth_rate": 0.02, "default_discount_rate": 0.10},
    "Utilities": {"forecast_years": 5, "terminal_growth_rate": 0.01, "default_discount_rate": 0.07}
}

# Streamlit UI settings
UI_SETTINGS = {
    "sidebar_width": 300,
    "content_width": 1000,
    "show_logo": True,
    "logo_path": str(SRC_DIR / "assets" / "logo.png") if (SRC_DIR / "assets" / "logo.png").exists() else None,
    "default_tabs": ["Overview", "Financial Analysis", "Valuation", "Comparison", "Risk Analysis"]
}