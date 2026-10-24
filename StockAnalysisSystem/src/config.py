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

# API keys: read from environment / StockAnalysisSystem/.env (never hard-code).
# Canonical names are *_API_KEY; the legacy *_KEY names are still accepted.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:  # python-dotenv is optional at import time
    pass


def _env_key(name: str) -> Optional[str]:
    return os.environ.get(f"{name}_API_KEY") or os.environ.get(f"{name}_KEY") or None


API_KEYS = {
    "alpha_vantage": _env_key("ALPHA_VANTAGE"),
    "finnhub": _env_key("FINNHUB"),
    "financial_modeling_prep": _env_key("FINANCIAL_MODELING_PREP"),
}

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
# Add to existing config.py file

# Valuation models configuration
VALUATION_MODELS = {
    "DCF": "Discounted Cash Flow",
    "DDM": "Dividend Discount Model",
    "NAV": "Net Asset Value",
    "Multiple": "Multiple Analysis",
    "SOTP": "Sum of the Parts"
}

# Forecasting models configuration
FORECASTING_MODELS = {
    "ARIMA": "AutoRegressive Integrated Moving Average",
    "ETS": "Error-Trend-Seasonality",
    "Prophet": "Facebook Prophet",
    "LSTM": "Long Short-Term Memory Networks",
    "RF": "Random Forest",
    "XGBOOST": "XGBoost Regression",
}

# Bankruptcy prediction models
BANKRUPTCY_MODELS = {
    "Altman": "Altman Z-Score",
    "Springate": "Springate Model",
    "Taffler": "Taffler Z-Score",
    "Zmijewski": "Zmijewski X-Score"
}

# Updated sector-specific important metrics to match new TZ requirements
SECTOR_METRICS = {
    "Technology": {
        "key_ratios": ["P/S", "R&D/Revenue", "Gross Margin", "Operating Margin"],
        "valuation_methods": ["DCF", "EV/Revenue", "PEG"],
        "benchmarks": {
            "ROE": 0.15,
            "Gross Margin": 0.50,
            "Revenue Growth": 0.15
        }
    },
    "Healthcare": {
        "key_ratios": ["P/E", "R&D/Revenue", "Operating Margin", "ROE"],
        "valuation_methods": ["DCF", "P/E", "EV/EBITDA"],
        "benchmarks": {
            "R&D/Revenue": 0.12,
            "Net Margin": 0.15,
            "ROE": 0.18
        }
    },
    "Financials": {
        "key_ratios": ["P/B", "ROE", "Net Interest Margin", "Efficiency Ratio"],
        "valuation_methods": ["DDM", "P/B", "Residual Income"],
        "benchmarks": {
            "ROE": 0.12,
            "Efficiency Ratio": 0.60,
            "NPL Ratio": 0.01
        }
    },
    "Energy": {
        "key_ratios": ["EV/EBITDA", "Reserve Replacement Ratio", "P/CF", "Debt/EBITDA"],
        "valuation_methods": ["NAV", "EV/EBITDA", "DCF"],
        "benchmarks": {
            "Debt/EBITDA": 2.5,
            "FCF Yield": 0.06
        }
    },
    "Consumer Discretionary": {
        "key_ratios": ["P/E", "Same-Store Sales Growth", "Inventory Turnover"],
        "valuation_methods": ["DCF", "EV/EBITDA", "P/S"],
        "benchmarks": {
            "Gross Margin": 0.35,
            "Inventory Turnover": 6
        }
    },
    "Consumer Staples": {
        "key_ratios": ["EV/EBITDA", "Dividend Yield", "Operating Margin"],
        "valuation_methods": ["DCF", "Dividend Yield", "P/E"],
        "benchmarks": {
            "Dividend Yield": 0.02,
            "Operating Margin": 0.15,
            "Debt/EBITDA": 2.0
        }
    },
    "Industrials": {
        "key_ratios": ["EV/EBITDA", "ROA", "ROIC", "Operating Margin"],
        "valuation_methods": ["DCF", "EV/EBITDA", "ROIC"],
        "benchmarks": {
            "ROIC": 0.10,
            "Operating Margin": 0.12
        }
    },
    "Materials": {
        "key_ratios": ["P/B", "EV/EBITDA", "ROE", "ROIC"],
        "valuation_methods": ["NAV", "EV/EBITDA", "P/CF"],
        "benchmarks": {
            "ROIC": 0.09,
            "Operating Margin": 0.12,
            "Debt/EBITDA": 2.5
        }
    },
    "Real Estate": {
        "key_ratios": ["FFO", "NAV", "Occupancy Rate", "Debt/Assets"],
        "valuation_methods": ["NAV", "Cap Rate", "FFO Multiple"],
        "benchmarks": {
            "FFO Yield": 0.08,
            "Occupancy Rate": 0.90,
            "Debt/Assets": 0.50
        }
    },
    "Communication Services": {
        "key_ratios": ["P/E", "ARPU", "Churn Rate", "EBITDA Margin"],
        "valuation_methods": ["DCF", "EV/EBITDA", "EV/Subscriber"],
        "benchmarks": {
            "EBITDA Margin": 0.35,
            "Churn Rate": 0.02
        }
    },
    "Utilities": {
        "key_ratios": ["Dividend Yield", "P/E", "Debt/EBITDA", "Interest Coverage"],
        "valuation_methods": ["DDM", "Regulated Asset Base", "P/E"],
        "benchmarks": {
            "Dividend Yield": 0.04,
            "Debt/EBITDA": 4.0,
            "Interest Coverage": 3.0
        }
    }
}

# Peer analysis configuration
PEER_ANALYSIS = {
    "max_peers": 10,
    "similarity_metrics": ["Market Cap", "Revenue Growth", "Profit Margin", "Beta"],
    "comparison_metrics": {
        "Valuation": ["P/E", "EV/EBITDA", "P/S", "P/B"],
        "Profitability": ["ROE", "ROA", "Net Margin", "Operating Margin"],
        "Growth": ["Revenue Growth", "EPS Growth", "EBITDA Growth"],
        "Efficiency": ["Asset Turnover", "Inventory Turnover"],
        "Financial Health": ["Debt/Equity", "Current Ratio", "Interest Coverage"]
    }
}

# Risk analysis settings
RISK_ANALYSIS = {
    "var_confidence_levels": [0.90, 0.95, 0.99],
    "sensitivity_factors": ["Interest Rates", "GDP Growth", "Inflation", "Currency"],
    "scenario_templates": {
        "Base Case": {"gdp_growth": 0.025, "inflation": 0.02, "interest_rate": 0.03},
        "Optimistic": {"gdp_growth": 0.04, "inflation": 0.015, "interest_rate": 0.025},
        "Pessimistic": {"gdp_growth": 0.005, "inflation": 0.04, "interest_rate": 0.05},
        "Stagflation": {"gdp_growth": -0.01, "inflation": 0.07, "interest_rate": 0.06},
        "Recovery": {"gdp_growth": 0.035, "inflation": 0.025, "interest_rate": 0.035}
    }
}

# Screening templates
SCREENING_TEMPLATES = {
    "Value Investing": {
        "P/E": {"operator": "<", "value": 15},
        "P/B": {"operator": "<", "value": 1.5},
        "Dividend Yield": {"operator": ">", "value": 0.03},
        "ROE": {"operator": ">", "value": 0.15},
        "Debt/Equity": {"operator": "<", "value": 0.5}
    },
    "Growth Investing": {
        "Revenue Growth": {"operator": ">", "value": 0.15},
        "EPS Growth": {"operator": ">", "value": 0.15},
        "ROE": {"operator": ">", "value": 0.15},
        "PEG": {"operator": "<", "value": 1.5}
    },
    "Dividend Investing": {
        "Dividend Yield": {"operator": ">", "value": 0.04},
        "Dividend Payout Ratio": {"operator": "<", "value": 0.6},
        "Dividend Growth": {"operator": ">", "value": 0.05},
        "Interest Coverage": {"operator": ">", "value": 3}
    },
    "Quality Investing": {
        "ROE": {"operator": ">", "value": 0.15},
        "Debt/Equity": {"operator": "<", "value": 0.5},
        "Operating Margin": {"operator": ">", "value": 0.15},
        "Free Cash Flow Margin": {"operator": ">", "value": 0.1}
    }
}

# Report templates
REPORT_TEMPLATES = {
    "Company Overview": ["company_info", "price_history", "key_metrics", "peer_comparison"],
    "Valuation Report": ["valuation_summary", "dcf_analysis", "multiple_analysis", "price_targets"],
    "Financial Analysis": ["income_statement", "balance_sheet", "cash_flow", "ratio_analysis"],
    "Risk Assessment": ["risk_metrics", "scenario_analysis", "stress_testing", "bankruptcy_risk"],
    "Industry Analysis": ["industry_overview", "competitive_positioning", "market_trends", "regulatory_environment"]
}