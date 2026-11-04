"""
Helpers for working with financial statement DataFrames.

Financial statements in this project follow the yfinance layout:

* index   -> line items (e.g. ``"Total Revenue"``, ``"Net Income"``)
* columns -> reporting period end dates (usually most recent first)

Line item names differ between data providers and even between companies
reported by the same provider, so every lookup should go through the helpers
in this module instead of indexing ``df.loc["Some Name"]`` directly.

The canonical ``financial_data`` bundle passed between the data layer,
analysis models, valuation models and UI pages is::

    {
        "income_statement": pd.DataFrame,
        "balance_sheet": pd.DataFrame,
        "cash_flow": pd.DataFrame,
        "company_info": dict,              # optional
        "market_data": {                   # optional
            "share_price": float,
            "market_cap": float,
            "shares_outstanding": float,
            "beta": float,
            "dividend_yield": float,       # fraction, e.g. 0.005 == 0.5 %
        },
    }

Use :func:`build_financial_data` to assemble it consistently.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Line item aliases
# ---------------------------------------------------------------------------
# Keys are canonical names used throughout the code base. Values are the
# candidate row labels, in priority order. Lookups are case-, space- and
# punctuation-insensitive, so only genuinely different labels need listing.

LINE_ITEM_ALIASES: Dict[str, List[str]] = {
    # ----- Income statement -------------------------------------------------
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue", "Revenues", "Net Sales", "Sales"],
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue", "Cost Of Goods Sold", "COGS"],
    "gross_profit": ["Gross Profit"],
    "operating_expense": ["Operating Expense", "Total Operating Expenses"],
    "rd_expense": ["Research And Development", "Research Development", "R&D Expense"],
    "sga_expense": ["Selling General And Administration", "Selling General Administrative",
                    "General And Administrative Expense"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "interest_income": ["Interest Income", "Interest Income Non Operating"],
    "net_interest_income": ["Net Interest Income"],
    "pretax_income": ["Pretax Income", "Income Before Tax"],
    "tax_provision": ["Tax Provision", "Income Tax Expense"],
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest",
                   "Net Income Including Noncontrolling Interests"],
    "diluted_eps": ["Diluted EPS"],
    "basic_eps": ["Basic EPS"],
    "diluted_shares": ["Diluted Average Shares", "Basic Average Shares"],
    "basic_shares": ["Basic Average Shares", "Diluted Average Shares"],
    "depreciation": ["Reconciled Depreciation", "Depreciation And Amortization",
                     "Depreciation Amortization Depletion", "Depreciation"],
    "tax_rate": ["Tax Rate For Calcs"],

    # ----- Balance sheet -----------------------------------------------------
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity",
                     "Total Equity Gross Minority Interest"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "cash": ["Cash And Cash Equivalents", "Cash Financial", "Cash"],
    "cash_and_short_term_investments": ["Cash Cash Equivalents And Short Term Investments",
                                        "Cash And Short Term Investments", "Cash And Cash Equivalents"],
    "short_term_investments": ["Other Short Term Investments", "Short Term Investments"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt"],
    "net_debt": ["Net Debt"],
    "inventory": ["Inventory"],
    "receivables": ["Accounts Receivable", "Receivables", "Net Receivables"],
    "payables": ["Accounts Payable", "Payables"],
    "retained_earnings": ["Retained Earnings"],
    "working_capital": ["Working Capital"],
    "net_ppe": ["Net PPE", "Property Plant Equipment Net"],
    "gross_ppe": ["Gross PPE"],
    "goodwill": ["Goodwill"],
    "goodwill_and_intangibles": ["Goodwill And Other Intangible Assets"],
    "intangible_assets": ["Other Intangible Assets", "Intangible Assets"],
    "tangible_book_value": ["Tangible Book Value"],
    "invested_capital": ["Invested Capital"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    "minority_interest": ["Minority Interest"],
    "preferred_stock": ["Preferred Stock"],

    # ----- Cash flow ---------------------------------------------------------
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                            "Total Cash From Operating Activities"],
    "capital_expenditure": ["Capital Expenditure", "Purchase Of PPE", "Capital Expenditures",
                            "Net PPE Purchase And Sale"],
    "free_cash_flow": ["Free Cash Flow"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid", "Dividends Paid"],
    "share_repurchases": ["Repurchase Of Capital Stock", "Common Stock Payments"],
    "stock_based_compensation": ["Stock Based Compensation"],
    "change_in_working_capital": ["Change In Working Capital"],
    "cash_flow_depreciation": ["Depreciation And Amortization", "Depreciation Amortization Depletion",
                               "Reconciled Depreciation"],
    "investing_cash_flow": ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"],
    "financing_cash_flow": ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"],
}

ItemSpec = Union[str, Sequence[str]]


def _normalize_label(label: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def is_empty(df: Optional[pd.DataFrame]) -> bool:
    """Return True when ``df`` is None, not a DataFrame or has no data."""
    return df is None or not isinstance(df, pd.DataFrame) or df.empty


def _candidates(item: ItemSpec) -> List[str]:
    if isinstance(item, str):
        return list(LINE_ITEM_ALIASES.get(item, [])) + [item]
    result: List[str] = []
    for entry in item:
        result.extend(LINE_ITEM_ALIASES.get(entry, []))
        result.append(entry)
    return result


def find_row(df: Optional[pd.DataFrame], item: ItemSpec) -> Optional[Any]:
    """
    Find the index label in ``df`` that matches ``item``.

    ``item`` may be a canonical key from :data:`LINE_ITEM_ALIASES`, a literal
    row label, or a sequence of either. Matching is exact first, then
    case/space/punctuation-insensitive. Rows that are entirely NaN are skipped
    so a later alias with real data wins.
    """
    if is_empty(df):
        return None

    candidates = _candidates(item)
    index = list(df.index)
    normalized = {}
    for label in index:
        normalized.setdefault(_normalize_label(label), label)

    for candidate in candidates:
        label = candidate if candidate in df.index else normalized.get(_normalize_label(candidate))
        if label is None:
            continue
        row = df.loc[label]
        if isinstance(row, pd.DataFrame):  # duplicated labels
            row = row.iloc[0]
        if pd.to_numeric(row, errors="coerce").notna().any():
            return label
    return None


def has_item(df: Optional[pd.DataFrame], item: ItemSpec) -> bool:
    """Return True if ``df`` contains a non-empty row for ``item``."""
    return find_row(df, item) is not None


def _sorted_columns(df: pd.DataFrame, ascending: bool) -> List[Any]:
    columns = list(df.columns)
    try:
        keys = pd.to_datetime(pd.Index(columns), errors="raise")
        order = np.argsort(keys.values)
        ordered = [columns[i] for i in order]
    except (ValueError, TypeError):
        ordered = columns[::-1]  # assume provider order: most recent first
    return ordered if ascending else ordered[::-1]


def get_series(df: Optional[pd.DataFrame], item: ItemSpec, ascending: bool = False,
               dropna: bool = True) -> pd.Series:
    """
    Return the numeric time series for ``item``.

    Args:
        df: Financial statement DataFrame.
        item: Canonical key, row label or list of candidates.
        ascending: Chronological order (oldest first) when True, otherwise most
            recent first (provider order).
        dropna: Drop missing periods.

    Returns:
        A float Series indexed by period. Empty Series when not found.
    """
    label = find_row(df, item)
    if label is None:
        return pd.Series(dtype=float)
    row = df.loc[label]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    row = pd.to_numeric(row, errors="coerce").astype(float)
    row = row[_sorted_columns(df, ascending=ascending)]
    if dropna:
        row = row.dropna()
    row.name = str(label)
    return row


def get_value(df: Optional[pd.DataFrame], item: ItemSpec, period: int = 0,
              default: Optional[float] = None) -> Optional[float]:
    """
    Return the value of ``item`` for a given period.

    ``period=0`` is the most recent period with data, ``1`` the one before, etc.
    """
    series = get_series(df, item, ascending=False)
    if len(series) <= period:
        return default
    value = series.iloc[period]
    return default if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def get_latest(df: Optional[pd.DataFrame], item: ItemSpec, default: Optional[float] = None) -> Optional[float]:
    """Return the most recent non-missing value of ``item``."""
    return get_value(df, item, 0, default)


def first_available(df: Optional[pd.DataFrame], items: Iterable[ItemSpec],
                    default: Optional[float] = None) -> Optional[float]:
    """Return the latest value of the first item in ``items`` that exists."""
    for item in items:
        value = get_latest(df, item)
        if value is not None:
            return value
    return default


def latest_period(df: Optional[pd.DataFrame]) -> Optional[Any]:
    """Return the label of the most recent column."""
    if is_empty(df):
        return None
    return _sorted_columns(df, ascending=False)[0]


def chronological(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a copy of ``df`` with columns sorted oldest -> newest."""
    if is_empty(df):
        return pd.DataFrame()
    return df[_sorted_columns(df, ascending=True)].copy()


def safe_divide(numerator: Optional[float], denominator: Optional[float],
                default: Optional[float] = None) -> Optional[float]:
    """Divide, returning ``default`` for missing values or a zero denominator."""
    try:
        if numerator is None or denominator is None:
            return default
        numerator = float(numerator)
        denominator = float(denominator)
        if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
            return default
        result = numerator / denominator
        return default if math.isinf(result) else result
    except (TypeError, ValueError):
        return default


def cagr(series: pd.Series) -> Optional[float]:
    """
    Compound annual growth rate between the first and last value of a
    chronological series. Returns None when it is not meaningful
    (fewer than two points or non-positive endpoints).
    """
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return None
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    years = len(series) - 1
    return (end / start) ** (1 / years) - 1


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert ``value`` to float, returning ``default`` for missing/invalid input."""
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def normalize_dividend_yield(value: Any) -> Optional[float]:
    """
    Return a dividend yield as a fraction (0.005 == 0.5 %).

    Recent yfinance versions report ``dividendYield`` in percentage points
    (``0.53`` meaning 0.53 %) while older ones used fractions. A yield above
    20 % as a fraction is implausible for a listed equity, so larger values
    are treated as percentage points.
    """
    value = to_float(value)
    if value is None or value < 0:
        return None
    return value / 100 if value > 0.2 else value


def build_financial_data(income_statement: Optional[pd.DataFrame] = None,
                         balance_sheet: Optional[pd.DataFrame] = None,
                         cash_flow: Optional[pd.DataFrame] = None,
                         company_info: Optional[Dict[str, Any]] = None,
                         price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Assemble the canonical ``financial_data`` bundle.

    Market data is derived from the latest close in ``price_data`` when
    available, falling back to ``company_info`` fields.
    """
    income_statement = income_statement if isinstance(income_statement, pd.DataFrame) else pd.DataFrame()
    balance_sheet = balance_sheet if isinstance(balance_sheet, pd.DataFrame) else pd.DataFrame()
    cash_flow = cash_flow if isinstance(cash_flow, pd.DataFrame) else pd.DataFrame()
    company_info = dict(company_info or {})

    share_price = None
    if isinstance(price_data, pd.DataFrame) and not price_data.empty and "Close" in price_data.columns:
        share_price = to_float(price_data["Close"].dropna().iloc[-1]) if price_data["Close"].notna().any() else None
    if share_price is None:
        share_price = to_float(company_info.get("current_price") or company_info.get("regular_market_price"))

    market_cap = to_float(company_info.get("market_cap"))
    shares_outstanding = to_float(company_info.get("shares_outstanding"))
    if shares_outstanding is None:
        shares_outstanding = get_latest(balance_sheet, "shares_outstanding")
    if shares_outstanding is None:
        shares_outstanding = get_latest(income_statement, "diluted_shares")
    if shares_outstanding is None and market_cap and share_price:
        shares_outstanding = market_cap / share_price
    if market_cap is None and shares_outstanding and share_price:
        market_cap = shares_outstanding * share_price

    dividend_yield = normalize_dividend_yield(company_info.get("dividend_yield"))

    return {
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "company_info": company_info,
        "market_data": {
            "share_price": share_price,
            "market_cap": market_cap,
            "shares_outstanding": shares_outstanding,
            "beta": to_float(company_info.get("beta")),
            "dividend_yield": dividend_yield,
        },
    }


def format_large_number(value: Optional[float], prefix: str = "$", decimals: int = 2) -> str:
    """Format a number with T/B/M/K suffix, e.g. ``$2.76T``."""
    value = to_float(value)
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    value = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{sign}{prefix}{value / threshold:.{decimals}f}{suffix}"
    return f"{sign}{prefix}{value:.{decimals}f}"
