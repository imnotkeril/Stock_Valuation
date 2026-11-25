"""
Synthetic, internally consistent financial statements in yfinance layout.

Real fixtures (AAPL, MSFT) only cover technology companies. The profiles
below let tests exercise every sector-specific model with realistic line
item names (index) and period-end dates (columns, most recent first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

PERIODS = [pd.Timestamp(f"{year}-12-31") for year in (2024, 2023, 2022, 2021)]


@dataclass
class Profile:
    symbol: str
    name: str
    sector: str            # yfinance spelling, as returned by the provider
    industry: str
    revenue: float         # latest annual revenue
    growth: float          # annual revenue growth
    gross_margin: float
    operating_margin: float
    net_margin: float
    assets_to_revenue: float
    equity_ratio: float    # equity / assets
    debt_ratio: float      # debt / assets
    cash_ratio: float      # cash / assets
    capex_ratio: float     # capex / revenue
    rd_ratio: float = 0.0
    payout_ratio: float = 0.3
    shares: float = 1e9
    price: float = 100.0
    beta: float = 1.0
    dividend_yield: Optional[float] = None
    inventory_ratio: float = 0.05   # inventory / revenue
    ppe_ratio: float = 0.3          # net PPE / assets
    extra_income_rows: Dict[str, float] = field(default_factory=dict)   # row -> fraction of revenue
    extra_balance_rows: Dict[str, float] = field(default_factory=dict)  # row -> fraction of assets


PROFILES: Dict[str, Profile] = {
    "bank": Profile("TBNK", "Test Bank Corp.", "Financial Services", "Banks - Diversified",
                    revenue=120e9, growth=0.05, gross_margin=1.0, operating_margin=0.38, net_margin=0.30,
                    assets_to_revenue=30.0, equity_ratio=0.09, debt_ratio=0.12, cash_ratio=0.10,
                    capex_ratio=0.0, payout_ratio=0.30, shares=2.9e9, price=200.0, beta=1.1,
                    dividend_yield=2.3, inventory_ratio=0.0, ppe_ratio=0.01,
                    extra_income_rows={"Net Interest Income": 0.55, "Interest Income": 1.1,
                                       "Interest Expense": 0.55, "Non Interest Income": 0.45,
                                       "Credit Losses Provision": -0.05}),
    "insurer": Profile("TINS", "Test Insurance Co.", "Financial Services", "Insurance - Property & Casualty",
                       revenue=60e9, growth=0.07, gross_margin=1.0, operating_margin=0.15, net_margin=0.12,
                       assets_to_revenue=1.8, equity_ratio=0.22, debt_ratio=0.06, cash_ratio=0.04,
                       capex_ratio=0.0, shares=0.6e9, price=250.0, beta=0.7, dividend_yield=1.5,
                       inventory_ratio=0.0, ppe_ratio=0.01),
    "reit": Profile("TREIT", "Test Realty Trust", "Real Estate", "REIT - Industrial",
                    revenue=8e9, growth=0.08, gross_margin=0.74, operating_margin=0.40, net_margin=0.33,
                    assets_to_revenue=11.0, equity_ratio=0.55, debt_ratio=0.35, cash_ratio=0.01,
                    capex_ratio=0.15, payout_ratio=0.90, shares=0.92e9, price=110.0, beta=1.0,
                    dividend_yield=3.5, inventory_ratio=0.0, ppe_ratio=0.85),
    "oil_gas": Profile("TOIL", "Test Petroleum Corp.", "Energy", "Oil & Gas Integrated",
                       revenue=340e9, growth=0.02, gross_margin=0.32, operating_margin=0.14, net_margin=0.10,
                       assets_to_revenue=1.3, equity_ratio=0.60, debt_ratio=0.09, cash_ratio=0.05,
                       capex_ratio=0.07, payout_ratio=0.45, shares=4.3e9, price=115.0, beta=0.9,
                       dividend_yield=3.3, inventory_ratio=0.06, ppe_ratio=0.65),
    "utility": Profile("TUTL", "Test Electric Co.", "Utilities", "Utilities - Regulated Electric",
                       revenue=25e9, growth=0.04, gross_margin=0.55, operating_margin=0.27, net_margin=0.16,
                       assets_to_revenue=7.0, equity_ratio=0.30, debt_ratio=0.45, cash_ratio=0.01,
                       capex_ratio=0.35, payout_ratio=0.65, shares=2.0e9, price=75.0, beta=0.5,
                       dividend_yield=2.9, inventory_ratio=0.02, ppe_ratio=0.80),
    "retail": Profile("TRTL", "Test Stores Inc.", "Consumer Cyclical", "Home Improvement Retail",
                      revenue=155e9, growth=0.03, gross_margin=0.33, operating_margin=0.14, net_margin=0.10,
                      assets_to_revenue=0.5, equity_ratio=0.05, debt_ratio=0.55, cash_ratio=0.04,
                      capex_ratio=0.02, payout_ratio=0.55, shares=1.0e9, price=380.0, beta=1.0,
                      dividend_yield=2.4, inventory_ratio=0.15, ppe_ratio=0.30),
    "staples": Profile("TSTP", "Test Beverages Co.", "Consumer Defensive", "Beverages - Non-Alcoholic",
                       revenue=46e9, growth=0.04, gross_margin=0.60, operating_margin=0.29, net_margin=0.23,
                       assets_to_revenue=2.1, equity_ratio=0.26, debt_ratio=0.42, cash_ratio=0.12,
                       capex_ratio=0.04, payout_ratio=0.75, shares=4.3e9, price=62.0, beta=0.6,
                       dividend_yield=3.1, inventory_ratio=0.09, ppe_ratio=0.10),
    "industrial": Profile("TIND", "Test Machinery Corp.", "Industrials", "Farm & Heavy Construction Machinery",
                          revenue=65e9, growth=0.05, gross_margin=0.35, operating_margin=0.20, net_margin=0.16,
                          assets_to_revenue=1.3, equity_ratio=0.22, debt_ratio=0.43, cash_ratio=0.08,
                          capex_ratio=0.03, rd_ratio=0.03, payout_ratio=0.25, shares=0.49e9, price=350.0,
                          beta=1.1, dividend_yield=1.6, inventory_ratio=0.25, ppe_ratio=0.15),
    "materials": Profile("TMAT", "Test Chemicals plc", "Basic Materials", "Specialty Chemicals",
                         revenue=33e9, growth=0.02, gross_margin=0.47, operating_margin=0.27, net_margin=0.19,
                         assets_to_revenue=2.4, equity_ratio=0.50, debt_ratio=0.25, cash_ratio=0.06,
                         capex_ratio=0.10, payout_ratio=0.40, shares=0.48e9, price=460.0, beta=0.9,
                         dividend_yield=1.2, inventory_ratio=0.15, ppe_ratio=0.30),
    "pharma": Profile("TPHR", "Test Pharma Inc.", "Healthcare", "Drug Manufacturers - General",
                      revenue=60e9, growth=0.06, gross_margin=0.73, operating_margin=0.30, net_margin=0.22,
                      assets_to_revenue=1.8, equity_ratio=0.35, debt_ratio=0.30, cash_ratio=0.10,
                      capex_ratio=0.05, rd_ratio=0.20, payout_ratio=0.45, shares=2.5e9, price=110.0,
                      beta=0.5, dividend_yield=2.8, inventory_ratio=0.10, ppe_ratio=0.15),
    "biotech_loss": Profile("TBIO", "Test Biotech Inc.", "Healthcare", "Biotechnology",
                            revenue=0.4e9, growth=0.40, gross_margin=0.85, operating_margin=-0.90,
                            net_margin=-0.85, assets_to_revenue=6.0, equity_ratio=0.70, debt_ratio=0.05,
                            cash_ratio=0.60, capex_ratio=0.05, rd_ratio=1.2, payout_ratio=0.0,
                            shares=0.12e9, price=45.0, beta=1.6, inventory_ratio=0.02, ppe_ratio=0.05),
    "telecom": Profile("TTEL", "Test Telecom Inc.", "Communication Services", "Telecom Services",
                       revenue=134e9, growth=0.01, gross_margin=0.58, operating_margin=0.22, net_margin=0.09,
                       assets_to_revenue=2.9, equity_ratio=0.25, debt_ratio=0.45, cash_ratio=0.01,
                       capex_ratio=0.13, payout_ratio=0.60, shares=4.2e9, price=42.0, beta=0.4,
                       dividend_yield=6.4, inventory_ratio=0.02, ppe_ratio=0.35),
    "saas": Profile("TSAS", "Test Cloud Software", "Technology", "Software - Application",
                    revenue=35e9, growth=0.11, gross_margin=0.76, operating_margin=0.18, net_margin=0.15,
                    assets_to_revenue=2.9, equity_ratio=0.60, debt_ratio=0.09, cash_ratio=0.15,
                    capex_ratio=0.02, rd_ratio=0.14, payout_ratio=0.10, shares=0.96e9, price=300.0,
                    beta=1.3, dividend_yield=0.5, inventory_ratio=0.0, ppe_ratio=0.05),
}


def _scaled(latest: float, growth: float) -> pd.Series:
    """Values for PERIODS (most recent first) growing at ``growth`` per year."""
    return pd.Series([latest / (1 + growth) ** i for i in range(len(PERIODS))], index=PERIODS, dtype=float)


def make_statements(profile: Profile) -> Dict[str, pd.DataFrame]:
    """Build income statement, balance sheet and cash flow for a profile."""
    p = profile
    rev = _scaled(p.revenue, p.growth)
    cogs = rev * (1 - p.gross_margin)
    gross = rev - cogs
    op_income = rev * p.operating_margin
    rd = rev * p.rd_ratio
    sga = gross - op_income - rd
    dep = rev * max(p.capex_ratio * 0.8, 0.01)
    interest = rev * 0.01
    pretax = op_income - interest
    net_income = rev * p.net_margin
    tax = (pretax - net_income).clip(lower=0)
    ebitda = op_income + dep
    shares = pd.Series([p.shares * (1 + 0.01 * i) for i in range(len(PERIODS))], index=PERIODS)

    income = {
        "Total Revenue": rev, "Operating Revenue": rev, "Cost Of Revenue": cogs, "Gross Profit": gross,
        "Research And Development": rd, "Selling General And Administration": sga,
        "Operating Expense": sga + rd, "Operating Income": op_income, "EBIT": op_income + 0 * rev,
        "EBITDA": ebitda, "Normalized EBITDA": ebitda, "Reconciled Depreciation": dep,
        "Interest Expense": interest, "Pretax Income": pretax, "Tax Provision": tax,
        "Net Income": net_income, "Net Income Common Stockholders": net_income,
        "Diluted EPS": net_income / shares, "Basic EPS": net_income / shares,
        "Diluted Average Shares": shares, "Basic Average Shares": shares,
        "Tax Rate For Calcs": pd.Series(0.21, index=PERIODS),
    }
    for row, frac in p.extra_income_rows.items():
        income[row] = rev * frac
    if p.rd_ratio == 0:
        income.pop("Research And Development")

    assets = rev * p.assets_to_revenue
    equity = assets * p.equity_ratio
    debt = assets * p.debt_ratio
    cash = assets * p.cash_ratio
    liabilities = assets - equity
    current_assets = cash + rev * (p.inventory_ratio + 0.10)
    current_liabilities = rev * 0.15 + debt * 0.1
    balance = {
        "Total Assets": assets, "Total Liabilities Net Minority Interest": liabilities,
        "Stockholders Equity": equity, "Common Stock Equity": equity,
        "Total Equity Gross Minority Interest": equity, "Total Debt": debt,
        "Long Term Debt": debt * 0.9, "Current Debt": debt * 0.1, "Net Debt": (debt - cash).clip(lower=0),
        "Cash And Cash Equivalents": cash, "Cash Cash Equivalents And Short Term Investments": cash * 1.2,
        "Current Assets": current_assets, "Current Liabilities": current_liabilities,
        "Working Capital": current_assets - current_liabilities, "Inventory": rev * p.inventory_ratio,
        "Accounts Receivable": rev * 0.08, "Accounts Payable": rev * 0.07,
        "Retained Earnings": equity * 0.8, "Net PPE": assets * p.ppe_ratio, "Gross PPE": assets * p.ppe_ratio * 1.6,
        "Goodwill": assets * 0.05, "Goodwill And Other Intangible Assets": assets * 0.08,
        "Tangible Book Value": equity - assets * 0.08, "Invested Capital": equity + debt,
        "Ordinary Shares Number": shares, "Share Issued": shares,
    }
    for row, frac in p.extra_balance_rows.items():
        balance[row] = assets * frac
    if p.inventory_ratio == 0:
        balance.pop("Inventory")

    capex = -(rev * p.capex_ratio)
    ocf = net_income + dep + rev * 0.01
    dividends = -(net_income.clip(lower=0) * p.payout_ratio)
    cash_flow = {
        "Operating Cash Flow": ocf, "Capital Expenditure": capex, "Free Cash Flow": ocf + capex,
        "Depreciation And Amortization": dep, "Stock Based Compensation": rev * 0.01,
        "Change In Working Capital": rev * -0.005, "Cash Dividends Paid": dividends,
        "Repurchase Of Capital Stock": -(net_income.clip(lower=0) * 0.2),
        "Investing Cash Flow": capex * 1.1, "Financing Cash Flow": dividends * 1.5,
        "Net Income From Continuing Operations": net_income,
    }
    if p.payout_ratio == 0:
        cash_flow.pop("Cash Dividends Paid")

    def frame(rows: Dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame({k: v.astype(float) for k, v in rows.items()}).T[PERIODS]

    return {"income_statement": frame(income), "balance_sheet": frame(balance), "cash_flow": frame(cash_flow)}


def make_company_info(profile: Profile) -> Dict:
    p = profile
    return {
        "symbol": p.symbol, "name": p.name, "sector": p.sector, "industry": p.industry,
        "country": "United States", "exchange": "NYQ", "currency": "USD",
        "market_cap": p.shares * p.price, "pe_ratio": p.price / (p.revenue * p.net_margin / p.shares)
        if p.net_margin > 0 else None,
        "pb_ratio": p.shares * p.price / (p.revenue * p.assets_to_revenue * p.equity_ratio),
        "dividend_yield": p.dividend_yield, "beta": p.beta, "current_price": p.price,
        "shares_outstanding": p.shares, "description": f"{p.name} is a synthetic test company.",
        "website": "https://example.com", "employees": 10000, "type": "Stock",
    }


def make_prices(profile: Profile, days: int = 800, seed: int = 7) -> pd.DataFrame:
    """Geometric random walk price history ending at ``profile.price``."""
    rng = np.random.default_rng(seed + sum(map(ord, profile.symbol)))
    index = pd.bdate_range(end=pd.Timestamp("2025-04-08"), periods=days, name="Date")
    returns = rng.normal(0.0003, 0.015 * profile.beta + 0.005, size=days)
    close = profile.price * np.exp(np.cumsum(returns) - np.sum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, days)))
    open_ = (high + low) / 2
    volume = rng.integers(1_000_000, 20_000_000, size=days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
                         "Dividends": 0.0, "Stock Splits": 0.0, "Adj Close": close}, index=index)


def make_company(kind: str) -> Dict:
    """
    Return everything needed to analyse a synthetic company::

        {"ticker", "profile", "company_info", "price_data", "financial_data"}
    """
    from StockAnalysisSystem.src.utils.financial_data import build_financial_data

    profile = PROFILES[kind]
    statements = make_statements(profile)
    info = make_company_info(profile)
    prices = make_prices(profile)
    return {
        "ticker": profile.symbol,
        "profile": profile,
        "company_info": info,
        "price_data": prices,
        "financial_data": build_financial_data(company_info=info, price_data=prices, **statements),
    }


SYMBOL_TO_KIND = {profile.symbol: kind for kind, profile in PROFILES.items()}
