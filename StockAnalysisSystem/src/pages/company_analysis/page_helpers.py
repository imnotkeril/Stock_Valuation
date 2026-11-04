"""
Shared helpers for the company analysis pages (overview, financials, valuation, risk).

Everything here is presentation-level glue on top of the shared data helpers in
``utils.financial_data``: number formatting, line-item tables,
ratio calculation with sector benchmarks, market snapshot and a small DCF
engine used for interactive what-if analysis.

All functions are defensive: missing statements or line items yield ``None`` /
empty results instead of exceptions, so pages can show a friendly message.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import COLORS, normalize_sector
from StockAnalysisSystem.src.utils.financial_data import (
    find_row,
    format_large_number,
    get_latest,
    get_series,
    is_empty,
    normalize_dividend_yield,
    safe_divide,
    to_float,
)

logger = logging.getLogger("company_analysis")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
POSITIVE = COLORS["primary"]
NEGATIVE = COLORS["accent"]
NEUTRAL = COLORS["warning"]
MUTED = "#9e9e9e"
SERIES_COLORS = [COLORS["info"], COLORS["primary"], COLORS["secondary"], COLORS["warning"],
                 COLORS["accent"], COLORS["success"], COLORS["danger"], "#fcbe6e"]


# ---------------------------------------------------------------------------
# Generic safety helpers
# ---------------------------------------------------------------------------
def safe_call(fn: Callable[..., Any], *args, default: Any = None, label: str = "", **kwargs) -> Any:
    """Call ``fn`` and return ``default`` (logging the error) if it raises."""
    try:
        result = fn(*args, **kwargs)
        return default if result is None else result
    except Exception as exc:  # models are owned elsewhere; never crash the page
        logger.warning("%s failed: %s", label or getattr(fn, "__name__", "call"), exc)
        return default


def is_number(value: Any) -> bool:
    return to_float(value) is not None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_money(value: Any, decimals: int = 2) -> str:
    """``$2.76T`` style formatting of large monetary amounts."""
    return format_large_number(value, "$", decimals)


def fmt_price(value: Any) -> str:
    value = to_float(value)
    return "N/A" if value is None else f"${value:,.2f}"


def fmt_pct(value: Any, decimals: int = 1, signed: bool = False) -> str:
    """Format a fraction (0.123) as a percentage string (12.3%)."""
    value = to_float(value)
    if value is None:
        return "N/A"
    return f"{value * 100:+.{decimals}f}%" if signed else f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: Any, decimals: int = 2, suffix: str = "x") -> str:
    value = to_float(value)
    return "N/A" if value is None else f"{value:,.{decimals}f}{suffix}"


def fmt_number(value: Any, decimals: int = 2) -> str:
    value = to_float(value)
    return "N/A" if value is None else f"{value:,.{decimals}f}"


def fmt_shares(value: Any) -> str:
    return format_large_number(value, "", 2)


def colored_text(text: str, color: str, bold: bool = True) -> str:
    weight = "font-weight:600;" if bold else ""
    return f"<span style='color:{color};{weight}'>{text}</span>"


def show_colored(text: str, color: str) -> None:
    st.markdown(colored_text(text, color), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Statement access
# ---------------------------------------------------------------------------
def get_statement(financial_data: Optional[Dict[str, Any]], key: str) -> pd.DataFrame:
    """Return a statement from the canonical bundle (empty DataFrame when missing)."""
    df = (financial_data or {}).get(key)
    return df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()


def period_label(column: Any) -> str:
    """Readable label for a statement column (``FY2024`` for annual period ends)."""
    try:
        ts = pd.Timestamp(column)
        return f"FY{ts.year}"
    except (ValueError, TypeError):
        return str(column)


def _unique_labels(columns: Sequence[Any]) -> List[str]:
    labels = [period_label(c) for c in columns]
    if len(set(labels)) == len(labels):
        return labels
    out = []
    for column in columns:
        try:
            out.append(pd.Timestamp(column).strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            out.append(str(column))
    return out


def item_series(df: pd.DataFrame, item: Any) -> pd.Series:
    """Chronological (oldest first) numeric series for a line item."""
    return get_series(df, item, ascending=True)


def line_item_frame(df: pd.DataFrame, items: Sequence[Tuple[str, Any]],
                    most_recent_first: bool = True) -> pd.DataFrame:
    """
    Build a numeric table of selected line items.

    Args:
        df: Statement in yfinance layout.
        items: ``[(display label, item spec), ...]`` where the spec is anything
            accepted by ``find_row`` (canonical key, row label or list).
        most_recent_first: Column order.

    Returns:
        DataFrame indexed by display label with period labels as columns.
        Items that are not reported are omitted.
    """
    rows: Dict[str, pd.Series] = {}
    for label, spec in items:
        series = get_series(df, spec, ascending=not most_recent_first, dropna=False)
        if not series.empty and series.notna().any():
            rows[label] = series
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).T
    # Drop stub periods where most line items are missing (e.g. a 5th year with only a few rows)
    coverage = frame.notna().mean()
    if (coverage >= 0.3).any():
        frame = frame.loc[:, coverage >= 0.3]
    try:
        order = sorted(frame.columns, key=pd.Timestamp, reverse=most_recent_first)
        frame = frame[order]
    except (ValueError, TypeError):
        pass
    frame.columns = _unique_labels(list(frame.columns))
    return frame


def format_frame(frame: pd.DataFrame, formatter: Callable[[Any], str],
                 row_formatters: Optional[Dict[str, Callable[[Any], str]]] = None) -> pd.DataFrame:
    """String-format a numeric frame (per-row formatter overrides allowed)."""
    if frame.empty:
        return frame
    row_formatters = row_formatters or {}
    out = {}
    for label, row in frame.iterrows():
        fn = row_formatters.get(label, formatter)
        out[label] = [fn(v) if is_number(v) else "—" for v in row.values]
    return pd.DataFrame.from_dict(out, orient="index", columns=list(frame.columns))


def free_cash_flow_series(cash_flow: pd.DataFrame) -> pd.Series:
    """Chronological free cash flow: reported FCF, else OCF - |capex|, else empty."""
    fcf = item_series(cash_flow, "free_cash_flow")
    if not fcf.empty:
        return fcf
    ocf = item_series(cash_flow, "operating_cash_flow")
    capex = item_series(cash_flow, "capital_expenditure")
    if ocf.empty:
        return pd.Series(dtype=float)
    if capex.empty:
        return pd.Series(dtype=float)
    aligned = pd.concat([ocf, capex], axis=1, join="inner")
    result = aligned.iloc[:, 0] - aligned.iloc[:, 1].abs()
    result.name = "Free Cash Flow"
    return result


def ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio aligned on periods; NaN where the denominator is 0."""
    if numerator.empty or denominator.empty:
        return pd.Series(dtype=float)
    aligned = pd.concat([numerator, denominator], axis=1, join="inner")
    den = aligned.iloc[:, 1].replace(0, np.nan)
    return (aligned.iloc[:, 0] / den).astype(float)


def yoy_growth(series: pd.Series) -> pd.Series:
    """Year-over-year growth of a chronological series (handles negative bases)."""
    if len(series) < 2:
        return pd.Series(dtype=float)
    prev = series.shift(1)
    growth = (series - prev) / prev.abs().replace(0, np.nan)
    return growth.iloc[1:].astype(float)


def series_to_labels(series: pd.Series) -> List[str]:
    return _unique_labels(list(series.index))


def is_quarterly(columns: Sequence[Any]) -> bool:
    """True when period ends are spaced roughly a quarter apart."""
    try:
        dates = sorted(pd.Timestamp(c) for c in columns)
    except (ValueError, TypeError):
        return False
    if len(dates) < 2:
        return False
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    return float(np.median(gaps)) < 200


def growth_label(columns: Sequence[Any]) -> str:
    """``QoQ`` for quarterly statements, ``YoY`` otherwise."""
    return "QoQ" if is_quarterly(columns) else "YoY"


def annualized_growth(series: pd.Series) -> Optional[float]:
    """
    CAGR between the first and last value of a chronological series, using the
    actual time span when the index holds dates (works for quarterly data).
    None when not meaningful (fewer than two points or non-positive endpoints).
    """
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return None
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    try:
        years = (pd.Timestamp(series.index[-1]) - pd.Timestamp(series.index[0])).days / 365.25
    except (ValueError, TypeError):
        years = float(len(series) - 1)
    if years <= 0.2:
        return None
    return (end / start) ** (1 / years) - 1


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def close_series(price_data: Optional[pd.DataFrame]) -> pd.Series:
    """Clean, chronologically sorted closing prices with a tz-naive DatetimeIndex."""
    if is_empty(price_data):
        return pd.Series(dtype=float)
    column = "Close" if "Close" in price_data.columns else ("Adj Close" if "Adj Close" in price_data.columns
                                                              else None)
    if column is None:
        return pd.Series(dtype=float)
    close = pd.to_numeric(price_data[column], errors="coerce").astype(float)
    try:
        index = pd.DatetimeIndex(pd.to_datetime(close.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        close.index = index
    except (ValueError, TypeError):
        return pd.Series(dtype=float)
    close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
    return close[close > 0]


def normalized_prices(price_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """OHLCV frame with tz-naive sorted DatetimeIndex (empty when unusable)."""
    if is_empty(price_data) or "Close" not in price_data.columns:
        return pd.DataFrame()
    df = price_data.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(df.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        df.index = index
    except (ValueError, TypeError):
        return pd.DataFrame()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[df["Close"].notna()]


def latest_close(price_data: Optional[pd.DataFrame]) -> Optional[float]:
    close = close_series(price_data)
    return float(close.iloc[-1]) if not close.empty else None


def canonical_sector(company_info: Optional[Dict[str, Any]], financial_data: Optional[Dict[str, Any]] = None) -> str:
    info = company_info or {}
    sector = info.get("sector") or ((financial_data or {}).get("company_info") or {}).get("sector")
    return normalize_sector(sector)


def market_snapshot(financial_data: Optional[Dict[str, Any]], company_info: Optional[Dict[str, Any]],
                    price_data: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """Price, share count, market cap, debt, cash and enterprise value."""
    info = company_info or {}
    market = (financial_data or {}).get("market_data") or {}
    balance = get_statement(financial_data, "balance_sheet")
    income = get_statement(financial_data, "income_statement")

    price = latest_close(price_data) or to_float(market.get("share_price")) or to_float(info.get("current_price"))
    shares = (to_float(market.get("shares_outstanding")) or to_float(info.get("shares_outstanding"))
              or get_latest(balance, "shares_outstanding") or get_latest(income, "diluted_shares"))
    market_cap = to_float(market.get("market_cap")) or to_float(info.get("market_cap"))
    if market_cap is None and price and shares:
        market_cap = price * shares
    if shares is None and market_cap and price:
        shares = market_cap / price

    total_debt = get_latest(balance, "total_debt")
    if total_debt is None:
        parts = [get_latest(balance, "long_term_debt"), get_latest(balance, "short_term_debt")]
        total_debt = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    cash = get_latest(balance, "cash_and_short_term_investments")
    if cash is None:
        cash = get_latest(balance, "cash")
    net_debt = None
    if total_debt is not None or cash is not None:
        net_debt = (total_debt or 0.0) - (cash or 0.0)
    enterprise_value = market_cap + net_debt if market_cap is not None and net_debt is not None else market_cap

    beta = to_float(market.get("beta")) if market.get("beta") is not None else to_float(info.get("beta"))
    dividend_yield = to_float(market.get("dividend_yield"))
    if dividend_yield is None:
        dividend_yield = normalize_dividend_yield(info.get("dividend_yield"))

    return {
        "price": price, "shares": shares, "market_cap": market_cap, "total_debt": total_debt,
        "cash": cash, "net_debt": net_debt, "enterprise_value": enterprise_value,
        "beta": beta, "dividend_yield": dividend_yield,
    }


# ---------------------------------------------------------------------------
# Ratios and sector benchmarks
# ---------------------------------------------------------------------------
#: Ratio metadata. Keys and categories match FinancialRatioAnalyzer.calculate_ratios.
RATIO_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "pe_ratio": {"label": "P/E", "category": "valuation", "kind": "x", "higher_is_better": False,
                 "description": "Share price divided by earnings per share."},
    "ps_ratio": {"label": "P/S", "category": "valuation", "kind": "x", "higher_is_better": False,
                 "description": "Market capitalisation divided by revenue."},
    "pb_ratio": {"label": "P/B", "category": "valuation", "kind": "x", "higher_is_better": False,
                 "description": "Market capitalisation divided by shareholders' equity."},
    "ev_ebitda": {"label": "EV/EBITDA", "category": "valuation", "kind": "x", "higher_is_better": False,
                  "description": "Enterprise value divided by EBITDA; neutral to capital structure."},
    "ev_revenue": {"label": "EV/Revenue", "category": "valuation", "kind": "x", "higher_is_better": False,
                   "description": "Enterprise value divided by revenue."},
    "gross_margin": {"label": "Gross Margin", "category": "profitability", "kind": "pct", "higher_is_better": True,
                     "description": "Gross profit / revenue - pricing power and production efficiency."},
    "operating_margin": {"label": "Operating Margin", "category": "profitability", "kind": "pct",
                         "higher_is_better": True,
                         "description": "Operating income / revenue - operational efficiency."},
    "net_margin": {"label": "Net Margin", "category": "profitability", "kind": "pct", "higher_is_better": True,
                   "description": "Net income / revenue - overall profitability after all expenses."},
    "roe": {"label": "Return on Equity", "category": "profitability", "kind": "pct", "higher_is_better": True,
            "description": "Net income / shareholders' equity."},
    "roa": {"label": "Return on Assets", "category": "profitability", "kind": "pct", "higher_is_better": True,
            "description": "Net income / total assets."},
    "current_ratio": {"label": "Current Ratio", "category": "liquidity", "kind": "x", "higher_is_better": True,
                      "description": "Current assets / current liabilities - short-term solvency."},
    "quick_ratio": {"label": "Quick Ratio", "category": "liquidity", "kind": "x", "higher_is_better": True,
                    "description": "(Current assets - inventory) / current liabilities."},
    "cash_ratio": {"label": "Cash Ratio", "category": "liquidity", "kind": "x", "higher_is_better": True,
                   "description": "Cash & equivalents / current liabilities - the strictest liquidity test."},
    "debt_to_equity": {"label": "Debt/Equity", "category": "leverage", "kind": "x", "higher_is_better": False,
                       "description": "Total debt / shareholders' equity - financial leverage."},
    "debt_to_assets": {"label": "Debt/Assets", "category": "leverage", "kind": "x", "higher_is_better": False,
                       "description": "Total debt / total assets - share of assets financed with debt."},
    "interest_coverage": {"label": "Interest Coverage", "category": "leverage", "kind": "x",
                          "higher_is_better": True,
                          "description": "Operating income / interest expense - ability to service debt."},
    "asset_turnover": {"label": "Asset Turnover", "category": "efficiency", "kind": "x", "higher_is_better": True,
                       "description": "Revenue / total assets - how efficiently assets generate sales."},
    "inventory_turnover": {"label": "Inventory Turnover", "category": "efficiency", "kind": "x",
                           "higher_is_better": True,
                           "description": "Cost of revenue / inventory - how quickly inventory is sold."},
    "receivables_turnover": {"label": "Receivables Turnover", "category": "efficiency", "kind": "x",
                             "higher_is_better": True,
                             "description": "Revenue / receivables - how quickly customers pay."},
    "revenue_growth": {"label": "Revenue Growth", "category": "growth", "kind": "pct", "higher_is_better": True,
                       "description": "Year-over-year change in revenue."},
    "net_income_growth": {"label": "Net Income Growth", "category": "growth", "kind": "pct",
                          "higher_is_better": True,
                          "description": "Year-over-year change in net income."},
}

#: Additional ratios reported by FinancialRatioAnalyzer: key -> (label, category, kind, higher_is_better, description)
_EXTRA_RATIOS = {
    "forward_pe": ("Forward P/E", "valuation", "x", False, "Price divided by analysts' next-year EPS estimate."),
    "peg_ratio": ("PEG", "valuation", "x", False, "P/E divided by EPS growth (in %)."),
    "price_to_fcf": ("P/FCF", "valuation", "x", False, "Market capitalisation divided by free cash flow."),
    "price_to_cash_flow": ("P/CF", "valuation", "x", False, "Market capitalisation divided by operating cash flow."),
    "price_to_ffo": ("P/FFO", "valuation", "x", False, "Market capitalisation divided by funds from operations."),
    "dividend_yield": ("Dividend Yield", "valuation", "pct", True, "Annual dividends divided by market value."),
    "payout_ratio": ("Payout Ratio", "valuation", "pct", None, "Dividends paid divided by net income."),
    "ebitda_margin": ("EBITDA Margin", "profitability", "pct", True, "EBITDA / revenue - cash operating profit."),
    "fcf_margin": ("FCF Margin", "profitability", "pct", True, "Free cash flow / revenue."),
    "roic": ("ROIC", "profitability", "pct", True, "After-tax operating income / invested capital."),
    "net_interest_margin": ("Net Interest Margin", "profitability", "pct", True,
                            "Net interest income / average assets - key for banks."),
    "operating_cash_flow_ratio": ("Operating CF Ratio", "liquidity", "x", True,
                                  "Operating cash flow / current liabilities."),
    "debt_to_ebitda": ("Debt/EBITDA", "leverage", "x", False, "Years of EBITDA needed to repay total debt."),
    "net_debt_to_ebitda": ("Net Debt/EBITDA", "leverage", "x", False, "Debt net of cash relative to EBITDA."),
    "cash_to_debt": ("Cash/Debt", "leverage", "x", True, "Cash and short-term investments relative to debt."),
    "days_inventory_outstanding": ("Days Inventory", "efficiency", "days", False,
                                   "Average days inventory is held before sale."),
    "days_sales_outstanding": ("Days Sales Outstanding", "efficiency", "days", False,
                               "Average days to collect receivables."),
    "rnd_to_revenue": ("R&D / Revenue", "efficiency", "pct", None, "Research & development intensity."),
    "eps_growth": ("EPS Growth", "growth", "pct", True, "Change in diluted EPS versus the prior period."),
    "operating_income_growth": ("Operating Income Growth", "growth", "pct", True,
                                "Change in operating income versus the prior period."),
    "ebitda_growth": ("EBITDA Growth", "growth", "pct", True, "Change in EBITDA versus the prior period."),
    "fcf_growth": ("FCF Growth", "growth", "pct", True, "Change in free cash flow versus the prior period."),
    "dividend_growth": ("Dividend Growth", "growth", "pct", True, "Change in dividends paid versus the prior period."),
    "revenue_cagr": ("Revenue CAGR", "growth", "pct", True, "Compound annual revenue growth over the history."),
    "net_income_cagr": ("Net Income CAGR", "growth", "pct", True, "Compound annual net income growth."),
    "eps_cagr": ("EPS CAGR", "growth", "pct", True, "Compound annual diluted EPS growth."),
}
for _key, (_label, _category, _kind, _better, _description) in _EXTRA_RATIOS.items():
    RATIO_DEFINITIONS.setdefault(_key, {"label": _label, "category": _category, "kind": _kind,
                                        "higher_is_better": _better, "description": _description})

RATIO_CATEGORIES: Dict[str, str] = {
    "valuation": "Valuation", "profitability": "Profitability", "liquidity": "Liquidity",
    "leverage": "Leverage", "efficiency": "Efficiency", "growth": "Growth",
}


def ratio_meta(key: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Metadata for a ratio key; unknown keys get a readable generic description."""
    if key in RATIO_DEFINITIONS:
        return RATIO_DEFINITIONS[key]
    pct_hints = ("margin", "growth", "cagr", "yield", "return", "ratio_pct")
    return {"label": key.replace("_", " ").title(), "category": category or "", "description": "",
            "kind": "pct" if any(h in key for h in pct_hints) else "x", "higher_is_better": None}


def format_ratio_value(key: str, value: Any) -> str:
    kind = ratio_meta(key).get("kind", "x")
    if kind == "pct":
        return fmt_pct(value)
    if kind == "days":
        value = to_float(value)
        return "N/A" if value is None else f"{value:,.0f} days"
    return fmt_ratio(value)


def compute_ratios(financial_data: Optional[Dict[str, Any]],
                   company_info: Optional[Dict[str, Any]] = None,
                   price_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
    """
    Latest-period ratios computed with the shared line-item helpers.

    Fallback for ``FinancialRatioAnalyzer.calculate_ratios`` (same keys) when the
    analyzer is unavailable.
    """
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    snap = market_snapshot(financial_data, company_info, price_data)

    revenue = get_latest(income, "revenue")
    net_income = get_latest(income, "net_income")
    gross_profit = get_latest(income, "gross_profit")
    if gross_profit is None and revenue is not None and get_latest(income, "cost_of_revenue") is not None:
        gross_profit = revenue - get_latest(income, "cost_of_revenue")
    operating_income = get_latest(income, "operating_income")
    ebitda = get_latest(income, "ebitda")
    interest = get_latest(income, "interest_expense")
    cost_of_revenue = get_latest(income, "cost_of_revenue")
    equity = get_latest(balance, "total_equity")
    assets = get_latest(balance, "total_assets")
    current_assets = get_latest(balance, "current_assets")
    current_liabilities = get_latest(balance, "current_liabilities")
    inventory = get_latest(balance, "inventory")
    receivables = get_latest(balance, "receivables")
    cash = get_latest(balance, "cash")
    debt = snap["total_debt"]
    market_cap = snap["market_cap"]
    ev = snap["enterprise_value"]

    def positive(value: Optional[float]) -> Optional[float]:
        return value if value is not None and value > 0 else None

    ratios: Dict[str, Dict[str, Optional[float]]] = {cat: {} for cat in RATIO_CATEGORIES}
    ratios["valuation"] = {
        "pe_ratio": safe_divide(market_cap, positive(net_income)),
        "ps_ratio": safe_divide(market_cap, positive(revenue)),
        "pb_ratio": safe_divide(market_cap, positive(equity)),
        "ev_ebitda": safe_divide(ev, positive(ebitda)) if ev is not None and ev > 0 else None,
        "ev_revenue": safe_divide(ev, positive(revenue)) if ev is not None and ev > 0 else None,
    }
    ratios["profitability"] = {
        "gross_margin": safe_divide(gross_profit, positive(revenue)),
        "operating_margin": safe_divide(operating_income, positive(revenue)),
        "net_margin": safe_divide(net_income, positive(revenue)),
        "roe": safe_divide(net_income, positive(equity)),
        "roa": safe_divide(net_income, positive(assets)),
    }
    quick_assets = current_assets - (inventory or 0.0) if current_assets is not None else None
    ratios["liquidity"] = {
        "current_ratio": safe_divide(current_assets, positive(current_liabilities)),
        "quick_ratio": safe_divide(quick_assets, positive(current_liabilities)),
        "cash_ratio": safe_divide(cash, positive(current_liabilities)),
    }
    ratios["leverage"] = {
        "debt_to_equity": safe_divide(debt, positive(equity)),
        "debt_to_assets": safe_divide(debt, positive(assets)),
        "interest_coverage": safe_divide(operating_income, abs(interest)) if interest else None,
    }
    ratios["efficiency"] = {
        "asset_turnover": safe_divide(revenue, positive(assets)),
        "inventory_turnover": safe_divide(cost_of_revenue, positive(inventory)),
        "receivables_turnover": safe_divide(revenue, positive(receivables)),
    }
    revenue_series = item_series(income, "revenue")
    income_series = item_series(income, "net_income")
    rev_growth = yoy_growth(revenue_series)
    ni_growth = yoy_growth(income_series)
    ratios["growth"] = {
        "revenue_growth": float(rev_growth.iloc[-1]) if not rev_growth.empty else None,
        "net_income_growth": float(ni_growth.iloc[-1]) if not ni_growth.empty else None,
    }
    return {cat: {k: float(v) for k, v in values.items() if is_number(v)} for cat, values in ratios.items()}


def get_company_ratios(financial_data: Optional[Dict[str, Any]],
                       company_info: Optional[Dict[str, Any]] = None,
                       price_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
    """
    Latest ratios from ``FinancialRatioAnalyzer.calculate_ratios``.

    The analyzer is the source of truth (it deliberately omits ratios that are
    not meaningful for a sector). :func:`compute_ratios` is used only when the
    analyzer fails or returns nothing.
    """
    from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

    bundle = dict(financial_data or {})
    if company_info and not bundle.get("company_info"):
        bundle["company_info"] = company_info
    analyzer_ratios = safe_call(lambda: FinancialRatioAnalyzer().calculate_ratios(bundle),
                                default={}, label="FinancialRatioAnalyzer.calculate_ratios")
    clean: Dict[str, Dict[str, float]] = {}
    if isinstance(analyzer_ratios, dict):
        for category, values in analyzer_ratios.items():
            if isinstance(values, dict):
                clean[category] = {k: float(v) for k, v in values.items() if is_number(v)}
    if any(clean.values()):
        return clean
    return compute_ratios(financial_data, company_info, price_data)


def get_sector_benchmarks(sector: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Sector benchmark ratios (FinancialRatioAnalyzer.get_sector_benchmarks) or {}."""
    from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

    canonical = normalize_sector(sector)
    benchmarks = safe_call(lambda: FinancialRatioAnalyzer().get_sector_benchmarks(canonical), default={},
                           label="FinancialRatioAnalyzer.get_sector_benchmarks")
    if not isinstance(benchmarks, dict):
        return {}
    clean: Dict[str, Dict[str, float]] = {}
    for category, values in benchmarks.items():
        if isinstance(values, dict):
            clean[category] = {k: float(v) for k, v in values.items() if is_number(v)}
    return clean


def compare_to_benchmark(value: Optional[float], benchmark: Optional[float],
                         higher_is_better: Optional[bool] = True, band: float = 0.10) -> Tuple[Optional[float], str, str]:
    """
    Compare a company value with a benchmark.

    Returns:
        ``(difference as fraction of the benchmark, assessment label, colour)``.
    """
    value, benchmark = to_float(value), to_float(benchmark)
    if value is None or benchmark is None or benchmark == 0:
        return None, "No benchmark", MUTED
    diff = (value - benchmark) / abs(benchmark)
    if abs(diff) <= band:
        return diff, "In line", NEUTRAL
    if higher_is_better is None:
        return diff, ("Above sector" if diff > 0 else "Below sector"), MUTED
    better = diff > 0 if higher_is_better else diff < 0
    return diff, ("Better than sector" if better else "Worse than sector"), (POSITIVE if better else NEGATIVE)


def benchmark_table(ratios: Dict[str, Dict[str, float]], benchmarks: Dict[str, Dict[str, float]],
                    keys: Sequence[str]) -> pd.DataFrame:
    """Display table comparing company ratios with sector benchmarks."""
    rows = []
    for key in keys:
        meta = ratio_meta(key)
        category = meta["category"]
        value = ratios.get(category, {}).get(key)
        bench = benchmarks.get(category, {}).get(key)
        if value is None:
            continue
        diff, label, _ = compare_to_benchmark(value, bench, meta.get("higher_is_better"))
        rows.append({
            "Metric": meta["label"],
            "Company": format_ratio_value(key, value),
            "Sector Benchmark": format_ratio_value(key, bench) if bench is not None else "N/A",
            "Difference": fmt_pct(diff, signed=True) if diff is not None else "N/A",
            "Assessment": label,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DCF engine (interactive what-if analysis)
# ---------------------------------------------------------------------------
def dcf_projection(base_fcf: Optional[float], growth: float, discount_rate: float, terminal_growth: float,
                   years: int, net_debt: Optional[float], shares: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    Two-stage DCF: ``years`` of FCF growing at ``growth`` then a Gordon terminal value.

    Returns None when the inputs are not meaningful (no FCF, discount rate not
    above terminal growth, no share count).
    """
    base_fcf, shares = to_float(base_fcf), to_float(shares)
    if base_fcf is None or shares is None or shares <= 0 or years < 1:
        return None
    if discount_rate <= terminal_growth or discount_rate <= -1:
        return None
    forecast = [base_fcf * (1 + growth) ** year for year in range(1, years + 1)]
    discount_factors = [(1 + discount_rate) ** year for year in range(1, years + 1)]
    present_values = [fcf / factor for fcf, factor in zip(forecast, discount_factors)]
    terminal_value = forecast[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / discount_factors[-1]
    enterprise_value = sum(present_values) + pv_terminal
    equity_value = enterprise_value - (to_float(net_debt) or 0.0)
    return {
        "forecast_fcf": forecast,
        "present_values": present_values,
        "pv_fcf": float(sum(present_values)),
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": equity_value / shares,
    }


def finite_or_none(value: Any) -> Optional[float]:
    value = to_float(value)
    return None if value is None or math.isnan(value) else value


def has_any_statement(financial_data: Optional[Dict[str, Any]]) -> bool:
    return any(not get_statement(financial_data, key).empty
               for key in ("income_statement", "balance_sheet", "cash_flow"))


def row_exists(df: pd.DataFrame, item: Any) -> bool:
    return find_row(df, item) is not None


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------
def render_section(label: str, renderer: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """
    Run one page section; an unexpected failure is reported in that section
    only (short message, details in the log) so the rest of the page renders.
    """
    try:
        renderer(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — isolate unexpected data shapes / model failures
        logger.exception("%s section failed", label)
        st.error(f"The {label} section could not be displayed: {exc}")


def reset_widgets_on_change(scope: str, signature: Any, widget_keys: Sequence[str]) -> None:
    """
    Forget the values of ``widget_keys`` when ``signature`` (e.g. the ticker)
    changes, so data-dependent defaults apply to a newly selected company.
    """
    marker = f"_{scope}_signature"
    if st.session_state.get(marker) != signature:
        for key in widget_keys:
            st.session_state.pop(key, None)
        st.session_state[marker] = signature
