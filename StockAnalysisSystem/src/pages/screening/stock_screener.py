"""
Stock screener page and the per-ticker data layer shared by the screening and
comparison pages.

Page entry point: :func:`run_stock_screener_page`.

Shared helpers (used by ``valuation_filter``, ``peer_comparison`` and
``sector_analysis``):

* :data:`METRICS` - display metadata of every screening metric;
* :func:`get_stock_snapshot` / :func:`load_stock_snapshot` - cached company
  info, prices, statements (``financial_data`` bundle) and key metrics;
* :func:`compute_key_metrics` - key ratios from a ``financial_data`` bundle;
* :func:`screen_stocks` - filter a universe on metric ranges;
* :func:`get_stock_universe`, :func:`get_stocks_by_sector`,
  :func:`get_stocks_by_sector_and_subsector` - thin wrappers around
  ``industry.universe``;
* table / formatting helpers and :func:`render_analyze_jump`.

Metric values are fractions for percentages (``0.25`` == 25 %), plain numbers
for multiples and ratios, and dollars for prices / market caps.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import RISK_FREE_RATE, SECTORS, normalize_sector
from StockAnalysisSystem.src.industry.universe import get_subsectors, get_universe, lookup_company
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.navigation import COMPANY_ANALYSIS, go_to
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.financial_data import (
    build_financial_data, format_large_number, get_latest, get_value, safe_divide, to_float,
)
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer

logger = logging.getLogger(__name__)

PREFIX = "screener"
CACHE_TTL_SECONDS = 3600
DEFAULT_LOOKBACK_DAYS = 365

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------
#: key -> {label, kind, category, higher_is_better}
#: kinds: percent (fraction), multiple (x), ratio, currency ($), large_currency ($, shown in $B)
METRICS: Dict[str, Dict[str, Any]] = {
    "price": {"label": "Price", "kind": "currency", "category": "Market", "higher_is_better": None},
    "market_cap": {"label": "Market Cap", "kind": "large_currency", "category": "Market", "higher_is_better": None},
    "perf_1y": {"label": "1Y Return", "kind": "percent", "category": "Performance", "higher_is_better": True},
    "volatility": {"label": "Volatility", "kind": "percent", "category": "Performance", "higher_is_better": False},
    "beta": {"label": "Beta", "kind": "ratio", "category": "Performance", "higher_is_better": None},
    "pe_ratio": {"label": "P/E", "kind": "multiple", "category": "Valuation", "higher_is_better": False},
    "pb_ratio": {"label": "P/B", "kind": "multiple", "category": "Valuation", "higher_is_better": False},
    "ps_ratio": {"label": "P/S", "kind": "multiple", "category": "Valuation", "higher_is_better": False},
    "ev_ebitda": {"label": "EV/EBITDA", "kind": "multiple", "category": "Valuation", "higher_is_better": False},
    "fcf_yield": {"label": "FCF Yield", "kind": "percent", "category": "Valuation", "higher_is_better": True},
    "gross_margin": {"label": "Gross Margin", "kind": "percent", "category": "Profitability",
                     "higher_is_better": True},
    "operating_margin": {"label": "Operating Margin", "kind": "percent", "category": "Profitability",
                         "higher_is_better": True},
    "net_margin": {"label": "Net Margin", "kind": "percent", "category": "Profitability", "higher_is_better": True},
    "roe": {"label": "ROE", "kind": "percent", "category": "Profitability", "higher_is_better": True},
    "roa": {"label": "ROA", "kind": "percent", "category": "Profitability", "higher_is_better": True},
    "revenue_growth": {"label": "Revenue Growth", "kind": "percent", "category": "Growth", "higher_is_better": True},
    "eps_growth": {"label": "EPS Growth", "kind": "percent", "category": "Growth", "higher_is_better": True},
    "current_ratio": {"label": "Current Ratio", "kind": "ratio", "category": "Financial Health",
                      "higher_is_better": True},
    "quick_ratio": {"label": "Quick Ratio", "kind": "ratio", "category": "Financial Health",
                    "higher_is_better": True},
    "debt_equity": {"label": "Debt/Equity", "kind": "ratio", "category": "Financial Health",
                    "higher_is_better": False},
    "interest_coverage": {"label": "Interest Coverage", "kind": "ratio", "category": "Financial Health",
                          "higher_is_better": True},
    "dividend_yield": {"label": "Dividend Yield", "kind": "percent", "category": "Dividends",
                       "higher_is_better": True},
    "payout_ratio": {"label": "Payout Ratio", "kind": "percent", "category": "Dividends", "higher_is_better": None},
}

METRIC_CATEGORIES: Dict[str, List[str]] = {}
for _key, _spec in METRICS.items():
    METRIC_CATEGORIES.setdefault(_spec["category"], []).append(_key)

_UNIT_SUFFIX = {"percent": " (%)", "multiple": " (x)", "currency": " ($)", "large_currency": " ($B)", "ratio": ""}

#: FinancialRatioAnalyzer.calculate_ratios (category, key) -> metric key
_ANALYZER_KEYS: Dict[Tuple[str, str], str] = {
    ("valuation", "pe_ratio"): "pe_ratio", ("valuation", "pb_ratio"): "pb_ratio",
    ("valuation", "ps_ratio"): "ps_ratio", ("valuation", "ev_ebitda"): "ev_ebitda",
    ("profitability", "gross_margin"): "gross_margin", ("profitability", "operating_margin"): "operating_margin",
    ("profitability", "net_margin"): "net_margin", ("profitability", "roe"): "roe", ("profitability", "roa"): "roa",
    ("liquidity", "current_ratio"): "current_ratio", ("liquidity", "quick_ratio"): "quick_ratio",
    ("leverage", "debt_to_equity"): "debt_equity", ("leverage", "interest_coverage"): "interest_coverage",
    ("growth", "revenue_growth"): "revenue_growth",
}

#: metric key -> (category, key) in ``industry.benchmarks.get_sector_benchmarks(...)["ratios"]``
BENCHMARK_KEYS: Dict[str, Tuple[str, str]] = {
    "pe_ratio": ("valuation", "pe_ratio"), "pb_ratio": ("valuation", "pb_ratio"),
    "ps_ratio": ("valuation", "ps_ratio"), "ev_ebitda": ("valuation", "ev_ebitda"),
    "gross_margin": ("profitability", "gross_margin"), "operating_margin": ("profitability", "operating_margin"),
    "net_margin": ("profitability", "net_margin"), "roe": ("profitability", "roe"), "roa": ("profitability", "roa"),
    "current_ratio": ("liquidity", "current_ratio"), "quick_ratio": ("liquidity", "quick_ratio"),
    "debt_equity": ("leverage", "debt_to_equity"), "interest_coverage": ("leverage", "interest_coverage"),
    "revenue_growth": ("growth", "revenue_growth"),
}

MARKET_CAP_BUCKETS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "All": (None, None),
    "Mega cap (> $200B)": (200e9, None),
    "Large cap ($10B - $200B)": (10e9, 200e9),
    "Mid cap ($2B - $10B)": (2e9, 10e9),
    "Small cap ($300M - $2B)": (300e6, 2e9),
    "Micro cap (< $300M)": (None, 300e6),
}


def metric_label(key: str, with_unit: bool = False) -> str:
    """Display label of a metric (optionally with its unit, e.g. ``"Net Margin (%)"``)."""
    spec = METRICS.get(key)
    if spec is None:
        return key.replace("_", " ").title()
    return spec["label"] + (_UNIT_SUFFIX[spec["kind"]] if with_unit else "")


def to_display(value: Any, kind: str) -> float:
    """Metric value in display units (percent points, $B for market cap); NaN when missing."""
    value = to_float(value)
    if value is None:
        return float("nan")
    if kind == "percent":
        return value * 100
    if kind == "large_currency":
        return value / 1e9
    return value


def format_metric(value: Any, key_or_kind: str) -> str:
    """Human readable metric value (``"24.0%"``, ``"29.4x"``, ``"$2.76T"``, ``"N/A"``)."""
    kind = METRICS.get(key_or_kind, {}).get("kind", key_or_kind)
    value = to_float(value)
    if value is None:
        return "N/A"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "multiple":
        return f"{value:.1f}x"
    if kind == "currency":
        return f"${value:,.2f}"
    if kind == "large_currency":
        return format_large_number(value)
    return f"{value:.2f}"


def metrics_table(rows: Iterable[Mapping[str, Any]], metric_keys: Sequence[str],
                  id_columns: Sequence[Tuple[str, str]] = (("symbol", "Symbol"), ("name", "Company"),
                                                           ("sector", "Sector"))) -> pd.DataFrame:
    """
    Numeric display table: identifier columns plus one column per metric in
    display units with the unit in the header (e.g. ``"ROE (%)"``).
    """
    records = []
    for row in rows:
        record = {label: row.get(key) for key, label in id_columns}
        for key in metric_keys:
            record[metric_label(key, with_unit=True)] = to_display(row.get(key), METRICS[key]["kind"])
        records.append(record)
    columns = [label for _, label in id_columns] + [metric_label(k, with_unit=True) for k in metric_keys]
    return pd.DataFrame.from_records(records, columns=columns)


def numeric_column_config(df: pd.DataFrame, decimals: int = 2) -> Dict[str, Any]:
    """``st.dataframe`` column config formatting every numeric column with ``decimals``."""
    return {column: st.column_config.NumberColumn(format=f"%.{decimals}f")
            for column in df.columns if pd.api.types.is_numeric_dtype(df[column])}


def benchmark_value(benchmarks: Optional[Mapping[str, Any]], metric: str) -> Optional[float]:
    """Benchmark of ``metric`` from ``get_sector_benchmarks(...)`` output (or its ``ratios`` dict)."""
    if not benchmarks or metric not in BENCHMARK_KEYS:
        return None
    ratios = benchmarks.get("ratios", benchmarks)
    category, key = BENCHMARK_KEYS[metric]
    section = ratios.get(category) if isinstance(ratios, Mapping) else None
    return to_float(section.get(key)) if isinstance(section, Mapping) else None


# ---------------------------------------------------------------------------
# Universe (thin wrappers around industry.universe)
# ---------------------------------------------------------------------------
def get_stock_universe() -> List[Dict[str, str]]:
    """Every company of the curated universe as ``{symbol, name, sector, subsector}`` records."""
    return get_universe()


def get_stocks_by_sector(sector: str) -> List[Dict[str, str]]:
    """Universe companies of ``sector`` (any provider spelling)."""
    return get_universe(sector)


def get_stocks_by_sector_and_subsector(sector: str, subsector: Optional[str]) -> List[Dict[str, str]]:
    """Universe companies of ``sector`` and ``subsector`` (``"All"`` disables the subsector filter)."""
    return get_universe(sector, subsector)


# ---------------------------------------------------------------------------
# Metrics and per-ticker data
# ---------------------------------------------------------------------------
def date_window(days: int, end: Optional[date] = None) -> Tuple[str, str]:
    """``(start, end)`` ISO dates covering the last ``days`` calendar days."""
    end = end or date.today()
    return (end - timedelta(days=int(days))).isoformat(), end.isoformat()


def close_prices(price_data: Optional[pd.DataFrame]) -> pd.Series:
    """Clean, sorted closing price series (empty when unavailable)."""
    if not isinstance(price_data, pd.DataFrame) or price_data.empty or "Close" not in price_data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(price_data["Close"], errors="coerce").astype(float)
    close = close[np.isfinite(close.to_numpy())]
    if isinstance(close.index, pd.DatetimeIndex):
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def trailing_window(close: pd.Series, days: int) -> pd.Series:
    """Last ``days`` calendar days of a date-indexed series."""
    if close.empty or not isinstance(close.index, pd.DatetimeIndex):
        return close
    return close[close.index >= close.index[-1] - pd.Timedelta(days=int(days))]


def performance_stats(close: pd.Series) -> Dict[str, Optional[float]]:
    """
    Total return, annualised return / volatility, max drawdown (fractions) and
    Sharpe ratio (risk-free rate from config) of a closing price series.
    """
    empty = {"total_return": None, "annual_return": None, "volatility": None, "max_drawdown": None, "sharpe": None}
    close = close.dropna() if isinstance(close, pd.Series) else pd.Series(dtype=float)
    if len(close) < 2 or close.iloc[0] <= 0:
        return empty
    total = float(close.iloc[-1] / close.iloc[0] - 1)
    returns = close.pct_change().dropna()
    volatility = float(returns.std() * math.sqrt(252)) if len(returns) > 1 else None
    if isinstance(close.index, pd.DatetimeIndex):
        years = max((close.index[-1] - close.index[0]).days / 365.25, 1 / 365.25)
    else:
        years = max(len(close) / 252, 1 / 252)
    annual = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    drawdown = float((close / close.cummax() - 1).min())
    sharpe = safe_divide(annual - RISK_FREE_RATE, volatility)
    return {"total_return": total, "annual_return": annual, "volatility": volatility, "max_drawdown": drawdown,
            "sharpe": sharpe}


def _growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1


def _positive(value: Optional[float]) -> Optional[float]:
    return value if value is not None and value > 0 else None


def _local_metrics(financial_data: Mapping[str, Any], company_info: Mapping[str, Any],
                   price_data: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """Key metrics computed directly from the statements with the shared line-item helpers."""
    income = financial_data.get("income_statement")
    balance = financial_data.get("balance_sheet")
    cash_flow = financial_data.get("cash_flow")
    market = financial_data.get("market_data") or {}

    close = close_prices(price_data)
    price = to_float(close.iloc[-1]) if not close.empty else to_float(market.get("share_price"))
    if price is None:
        price = to_float(company_info.get("current_price"))
    market_cap = to_float(market.get("market_cap")) or to_float(company_info.get("market_cap"))

    revenue = get_latest(income, "revenue")
    net_income = get_latest(income, "net_income")
    gross_profit = get_latest(income, "gross_profit")
    operating_income = get_latest(income, "operating_income")
    ebitda = get_latest(income, "ebitda")
    if ebitda is None and operating_income is not None:
        ebitda = operating_income + (get_latest(income, "depreciation") or get_latest(cash_flow, "depreciation")
                                     or 0.0)
    interest = get_latest(income, "interest_expense")

    equity = get_latest(balance, "total_equity")
    assets = get_latest(balance, "total_assets")
    debt = get_latest(balance, "total_debt")
    cash = get_latest(balance, "cash_and_short_term_investments")
    current_assets = get_latest(balance, "current_assets")
    current_liabilities = get_latest(balance, "current_liabilities")
    inventory = get_latest(balance, "inventory") or 0.0

    free_cash_flow = get_latest(cash_flow, "free_cash_flow")
    if free_cash_flow is None:
        ocf, capex = get_latest(cash_flow, "operating_cash_flow"), get_latest(cash_flow, "capital_expenditure")
        free_cash_flow = ocf - abs(capex) if ocf is not None and capex is not None else None
    dividends = get_latest(cash_flow, "dividends_paid")

    enterprise_value = market_cap + (debt or 0.0) - (cash or 0.0) if market_cap else None

    eps_growth = _growth(get_value(income, "diluted_eps", 0), get_value(income, "diluted_eps", 1))
    if eps_growth is None:
        eps_growth = _growth(get_value(income, "net_income", 0), get_value(income, "net_income", 1))

    year = trailing_window(close, 365)
    perf = performance_stats(year) if len(year) >= 20 else {}

    dividend_yield = to_float(market.get("dividend_yield"))
    return {
        "price": price,
        "market_cap": market_cap,
        "perf_1y": perf.get("total_return"),
        "volatility": perf.get("volatility"),
        "beta": to_float(market.get("beta")) or to_float(company_info.get("beta")),
        "pe_ratio": safe_divide(market_cap, _positive(net_income)),
        "pb_ratio": safe_divide(market_cap, _positive(equity)),
        "ps_ratio": safe_divide(market_cap, _positive(revenue)),
        "ev_ebitda": safe_divide(enterprise_value, _positive(ebitda)) if enterprise_value and enterprise_value > 0
        else None,
        "fcf_yield": safe_divide(free_cash_flow, _positive(market_cap)),
        "gross_margin": safe_divide(gross_profit, _positive(revenue)),
        "operating_margin": safe_divide(operating_income, _positive(revenue)),
        "net_margin": safe_divide(net_income, _positive(revenue)),
        "roe": safe_divide(net_income, _positive(equity)),
        "roa": safe_divide(net_income, _positive(assets)),
        "revenue_growth": _growth(get_value(income, "revenue", 0), get_value(income, "revenue", 1)),
        "eps_growth": eps_growth,
        "current_ratio": safe_divide(current_assets, _positive(current_liabilities)),
        "quick_ratio": safe_divide(current_assets - inventory, _positive(current_liabilities))
        if current_assets is not None else None,
        "debt_equity": safe_divide(debt, _positive(equity)),
        "interest_coverage": safe_divide(operating_income, abs(interest)) if interest else None,
        "dividend_yield": dividend_yield if dividend_yield is not None else 0.0 if dividends == 0 else None,
        "payout_ratio": safe_divide(abs(dividends), _positive(net_income)) if dividends is not None else None,
    }


def compute_key_metrics(financial_data: Optional[Mapping[str, Any]],
                        company_info: Optional[Mapping[str, Any]] = None,
                        price_data: Optional[pd.DataFrame] = None,
                        ratio_analyzer: Optional[FinancialRatioAnalyzer] = None) -> Dict[str, Optional[float]]:
    """
    Key screening metrics (see :data:`METRICS`) for one company.

    Ratios from ``FinancialRatioAnalyzer.calculate_ratios`` take precedence so
    the numbers match the Company Analysis page; gaps are filled from the
    statements via the shared line-item helpers. Missing values are None.
    """
    financial_data = financial_data or {}
    company_info = company_info or financial_data.get("company_info") or {}
    metrics = _local_metrics(financial_data, company_info, price_data)
    try:
        ratios = (ratio_analyzer or FinancialRatioAnalyzer()).calculate_ratios(dict(financial_data)) or {}
    except Exception:  # analyzer problems must not break screening
        logger.exception("Ratio analysis failed; using statement-based metrics only")
        ratios = {}
    for (category, key), metric in _ANALYZER_KEYS.items():
        value = to_float((ratios.get(category) or {}).get(key)) if isinstance(ratios, Mapping) else None
        if value is not None:
            metrics[metric] = value
    return {key: to_float(metrics.get(key)) for key in METRICS}


def build_stock_snapshot(symbol: str, data_loader: DataLoader, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                         ratio_analyzer: Optional[FinancialRatioAnalyzer] = None) -> Dict[str, Any]:
    """
    Load everything the screening / comparison pages need for one ticker.

    Returns:
        ``{symbol, name, sector, subsector, industry, company_info, price_data,
        financial_data, metrics}``.

    Raises:
        LookupError: when neither company information nor prices are available.
    """
    symbol = symbol.strip().upper()
    record = lookup_company(symbol) or {}
    info = data_loader.get_company_info(symbol) or {}
    start, end = date_window(lookback_days)
    try:
        prices = data_loader.get_historical_prices(symbol, start, end)
    except Exception:
        logger.exception("Price download failed for %s", symbol)
        prices = pd.DataFrame()
    if not info and (prices is None or prices.empty):
        raise LookupError(f"No data found for {symbol}")

    statements = {}
    for key, statement_type in (("income_statement", "income"), ("balance_sheet", "balance"),
                                ("cash_flow", "cash")):
        try:
            statements[key] = data_loader.get_financial_statements(symbol, statement_type, "annual")
        except Exception:
            logger.exception("Could not load %s statement for %s", statement_type, symbol)
            statements[key] = pd.DataFrame()

    financial_data = build_financial_data(company_info=info, price_data=prices, **statements)
    sector = normalize_sector(info.get("sector") or record.get("sector"))
    industry = info.get("industry") if info.get("industry") not in (None, "", "N/A") else None
    return {
        "symbol": symbol,
        "name": info.get("name") or record.get("name") or symbol,
        "sector": sector,
        "subsector": record.get("subsector") or industry or "N/A",
        "industry": industry or record.get("subsector") or "N/A",
        "company_info": info,
        "price_data": prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(),
        "financial_data": financial_data,
        "metrics": compute_key_metrics(financial_data, info, prices, ratio_analyzer),
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=2000)
def load_stock_snapshot(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, Any]:
    """Cached :func:`build_stock_snapshot` using a default ``DataLoader`` (raises on failure; not cached)."""
    return build_stock_snapshot(symbol, DataLoader(), lookback_days)


def get_stock_snapshot(symbol: str, data_loader: Optional[DataLoader] = None,
                       lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                       ratio_analyzer: Optional[FinancialRatioAnalyzer] = None) -> Optional[Dict[str, Any]]:
    """
    Snapshot of one ticker or None when its data cannot be loaded (logged).

    Uses the Streamlit cache unless a specific ``data_loader`` is passed.
    """
    try:
        if data_loader is not None:
            return build_stock_snapshot(symbol, data_loader, lookback_days, ratio_analyzer)
        return load_stock_snapshot(symbol.strip().upper(), int(lookback_days))
    except LookupError as e:
        logger.info(str(e))
    except Exception:
        logger.exception("Could not load data for %s", symbol)
    return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=500)
def load_price_history(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Cached daily price history of ``symbol`` (empty DataFrame when unavailable)."""
    start, end = date_window(lookback_days)
    try:
        prices = DataLoader().get_historical_prices(symbol.upper(), start, end)
    except Exception:
        logger.exception("Could not load prices for %s", symbol)
        return pd.DataFrame()
    return prices if isinstance(prices, pd.DataFrame) else pd.DataFrame()


def snapshot_row(snapshot: Mapping[str, Any], record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Flat, lightweight result row (identifiers + metrics, no DataFrames)."""
    record = record or {}
    row = {
        "symbol": snapshot["symbol"],
        "name": snapshot.get("name") or record.get("name") or snapshot["symbol"],
        "sector": snapshot.get("sector") or record.get("sector") or "Unknown",
        "subsector": record.get("subsector") or snapshot.get("subsector") or "N/A",
        "industry": snapshot.get("industry") or "N/A",
    }
    row.update(snapshot.get("metrics") or {})
    return row


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------
#: legacy ``min_<name>`` / ``max_<name>`` keyword names -> metric key
LEGACY_FILTERS = {
    "pe": "pe_ratio", "pb": "pb_ratio", "ps": "ps_ratio", "dividend": "dividend_yield", "payout": "payout_ratio",
    "de": "debt_equity", "perf_1y": "perf_1y", "market_cap": "market_cap", "price": "price",
}

Criteria = Dict[str, Tuple[Optional[float], Optional[float]]]


def _bound(value: Any) -> Optional[float]:
    value = to_float(value)
    return value if value is not None and math.isfinite(value) else None


def build_criteria(criteria: Optional[Mapping[str, Sequence[Any]]] = None, **filters: Any) -> Criteria:
    """
    Normalise screening criteria to ``{metric: (min, max)}``.

    ``criteria`` maps metric keys to ``(min, max)``; legacy keyword filters
    such as ``min_pe=0, max_pe=25`` or ``min_roe=0.15`` are merged in.
    None and infinite bounds mean "no bound"; criteria without any bound are dropped.
    """
    merged: Dict[str, List[Optional[float]]] = {}
    for metric, bounds in (criteria or {}).items():
        lo, hi = (list(bounds) + [None, None])[:2]
        merged[metric] = [_bound(lo), _bound(hi)]
    for name, value in filters.items():
        side, _, base = name.partition("_")
        if side not in ("min", "max") or not base:
            raise TypeError(f"Unknown screening filter: {name}")
        metric = LEGACY_FILTERS.get(base, base)
        if metric not in METRICS:
            raise TypeError(f"Unknown screening metric: {base}")
        merged.setdefault(metric, [None, None])[0 if side == "min" else 1] = _bound(value)
    return {metric: (lo, hi) for metric, (lo, hi) in merged.items() if lo is not None or hi is not None}


def passes_criteria(row: Mapping[str, Any], criteria: Criteria, include_missing: bool = False) -> bool:
    """True when every criterion holds (missing values fail unless ``include_missing``)."""
    for metric, (lo, hi) in criteria.items():
        value = to_float(row.get(metric))
        if value is None:
            if include_missing:
                continue
            return False
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return False
    return True


def screen_stocks(universe: Iterable[Any], data_loader: Optional[DataLoader] = None,
                  ratio_analyzer: Optional[FinancialRatioAnalyzer] = None, *,
                  criteria: Optional[Mapping[str, Sequence[Any]]] = None, include_missing: bool = False,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None,
                  failures: Optional[List[str]] = None, **filters: Any) -> List[Dict[str, Any]]:
    """
    Screen ``universe`` (records with ``symbol`` or plain tickers) on metric ranges.

    Args:
        data_loader: Optional loader; the cached default loader is used when None.
        ratio_analyzer: Optional analyzer (only used with an explicit ``data_loader``).
        criteria: ``{metric: (min, max)}`` in metric units (fractions for percentages).
        include_missing: Keep stocks whose value for a filtered metric is unknown.
        progress_callback: ``callback(done, total, symbol)`` after each ticker.
        failures: List receiving tickers whose data could not be loaded.
        **filters: Legacy ``min_<metric>`` / ``max_<metric>`` bounds.

    Returns:
        Matching rows (see :func:`snapshot_row`) in universe order. Tickers that
        fail to load are skipped, never fatal.
    """
    rules = build_criteria(criteria, **filters)
    records = [u if isinstance(u, Mapping) else {"symbol": str(u)} for u in universe]
    results: List[Dict[str, Any]] = []
    for done, record in enumerate(records, start=1):
        symbol = str(record.get("symbol", "")).strip().upper()
        snapshot = get_stock_snapshot(symbol, data_loader, ratio_analyzer=ratio_analyzer) if symbol else None
        if snapshot is None:
            if failures is not None and symbol:
                failures.append(symbol)
        else:
            row = snapshot_row(snapshot, record)
            if passes_criteria(row, rules, include_missing):
                results.append(row)
        if progress_callback:
            progress_callback(done, len(records), symbol)
    return results


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
def render_analyze_jump(symbols: Sequence[str], key_prefix: str, names: Optional[Mapping[str, str]] = None,
                        label: str = "Open in Company Analysis") -> None:
    """Selectbox + button that jumps to Company Analysis for the chosen ticker."""
    symbols = [s for s in dict.fromkeys(symbols) if s]
    if not symbols:
        return
    names = names or {}
    col_pick, col_button = st.columns([3, 1], vertical_alignment="bottom")
    with col_pick:
        choice = st.selectbox(label, symbols, key=f"{key_prefix}_analyze_pick",
                              format_func=lambda s: f"{s} - {names[s]}" if names.get(s) else s)
    with col_button:
        st.button("Analyze", key=f"{key_prefix}_analyze", icon=":material/query_stats:", on_click=go_to,
                  args=(COMPANY_ANALYSIS, choice), type="secondary")


def progress_reporter(bar: Any, verb: str = "Loading") -> Callable[[int, int, str], None]:
    """Progress callback updating an ``st.progress`` bar."""

    def report(done: int, total: int, symbol: str) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"{verb} {symbol} ({done}/{total})")

    return report


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
#: Filter slider ranges in display units: metric -> (min, max, step). A bound
#: at the end of the slider means "no limit".
FILTER_RANGES: Dict[str, Tuple[float, float, float]] = {
    "price": (0.0, 2000.0, 5.0), "perf_1y": (-100.0, 300.0, 1.0), "volatility": (0.0, 150.0, 1.0),
    "beta": (0.0, 3.0, 0.05), "pe_ratio": (0.0, 100.0, 0.5), "pb_ratio": (0.0, 50.0, 0.1),
    "ps_ratio": (0.0, 30.0, 0.1), "ev_ebitda": (0.0, 60.0, 0.5), "fcf_yield": (-20.0, 30.0, 0.5),
    "gross_margin": (-50.0, 100.0, 1.0), "operating_margin": (-100.0, 100.0, 1.0),
    "net_margin": (-100.0, 100.0, 1.0), "roe": (-100.0, 200.0, 1.0), "roa": (-50.0, 50.0, 0.5),
    "revenue_growth": (-50.0, 200.0, 1.0), "eps_growth": (-100.0, 300.0, 1.0),
    "current_ratio": (0.0, 10.0, 0.1), "quick_ratio": (0.0, 10.0, 0.1), "debt_equity": (0.0, 10.0, 0.1),
    "interest_coverage": (0.0, 100.0, 1.0), "dividend_yield": (0.0, 15.0, 0.1), "payout_ratio": (0.0, 150.0, 1.0),
}

#: Strategy presets: name -> (description, {metric: (min, max)} in display units)
STRATEGIES: Dict[str, Tuple[str, Dict[str, Tuple[float, float]]]] = {
    "Custom": ("Build your own filters.", {}),
    "Value": ("Low earnings and book multiples with a dividend.",
              {"pe_ratio": (0.0, 20.0), "pb_ratio": (0.0, 3.0), "dividend_yield": (1.0, 15.0)}),
    "Growth": ("Double-digit revenue and earnings growth.",
               {"revenue_growth": (10.0, 200.0), "eps_growth": (10.0, 300.0)}),
    "Quality": ("High returns on equity, solid margins and moderate leverage.",
                {"roe": (15.0, 200.0), "net_margin": (10.0, 100.0), "debt_equity": (0.0, 1.5)}),
    "Dividend": ("High, sustainable dividends.", {"dividend_yield": (3.0, 15.0), "payout_ratio": (0.0, 80.0)}),
    "Momentum": ("Strong 12-month price performance.", {"perf_1y": (15.0, 300.0)}),
}

TABLE_METRICS = ["price", "market_cap", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "net_margin", "roe",
                 "revenue_growth", "debt_equity", "dividend_yield", "perf_1y"]

RESULTS_KEY = f"{PREFIX}_results"
FILTERS_KEY = f"{PREFIX}_active_filters"


def _range_key(metric: str) -> str:
    return f"{PREFIX}_range_{metric}"


def _apply_strategy() -> None:
    """``on_change`` callback: load the preset filters of the selected strategy."""
    _, preset = STRATEGIES.get(st.session_state.get(f"{PREFIX}_strategy"), ("", {}))
    if not preset:
        return
    st.session_state[FILTERS_KEY] = list(preset)
    for metric, bounds in preset.items():
        st.session_state[_range_key(metric)] = tuple(float(b) for b in bounds)


def _criteria_from_widgets(active: Sequence[str]) -> Criteria:
    """Slider ranges (display units) -> metric criteria (metric units); slider ends mean no limit."""
    criteria: Criteria = {}
    for metric in active:
        lo_limit, hi_limit, _ = FILTER_RANGES[metric]
        lo, hi = st.session_state.get(_range_key(metric), (lo_limit, hi_limit))
        scale = 100.0 if METRICS[metric]["kind"] == "percent" else 1.0
        criteria[metric] = (None if lo <= lo_limit else lo / scale, None if hi >= hi_limit else hi / scale)
    return criteria


def _render_controls() -> Dict[str, Any]:
    """Sidebar filter controls; returns the current screening parameters."""
    with st.sidebar:
        st.subheader("Screening filters")
        st.selectbox("Strategy", list(STRATEGIES), key=f"{PREFIX}_strategy", on_change=_apply_strategy,
                     help="Presets fill in the ratio filters below; adjust them freely.")
        st.caption(STRATEGIES[st.session_state.get(f"{PREFIX}_strategy", "Custom")][0])

        sector = st.selectbox("Sector", ["All"] + SECTORS, key=f"{PREFIX}_sector")
        subsector = "All"
        if sector != "All":
            subsector = st.selectbox("Subsector", ["All"] + get_subsectors(sector), key=f"{PREFIX}_subsector_{sector}")
        cap_bucket = st.selectbox("Market capitalisation", list(MARKET_CAP_BUCKETS), key=f"{PREFIX}_market_cap")

        st.session_state.setdefault(FILTERS_KEY, [])
        active = st.multiselect("Ratio filters", list(FILTER_RANGES), key=FILTERS_KEY, format_func=metric_label,
                                placeholder="Add a filter")
        for metric in active:
            lo_limit, hi_limit, step = FILTER_RANGES[metric]
            st.session_state.setdefault(_range_key(metric), (lo_limit, hi_limit))
            unit = {"percent": "%", "multiple": "x", "currency": "$"}.get(METRICS[metric]["kind"], "")
            st.slider(f"{metric_label(metric)} ({unit})" if unit else metric_label(metric), min_value=lo_limit,
                      max_value=hi_limit, step=step, key=_range_key(metric),
                      help="Moving a handle to the end of the slider removes that limit.")
        include_missing = st.checkbox("Keep stocks with missing data", key=f"{PREFIX}_include_missing",
                                      help="Otherwise stocks without a value for an active filter are excluded.")
        run = st.button("Run screen", type="primary", key=f"{PREFIX}_run", icon=":material/filter_alt:",
                        width="stretch")

    criteria = _criteria_from_widgets(active)
    lo_cap, hi_cap = MARKET_CAP_BUCKETS[cap_bucket]
    if lo_cap is not None or hi_cap is not None:
        criteria["market_cap"] = (lo_cap, hi_cap)
    return {"sector": sector, "subsector": subsector, "market_cap": cap_bucket, "criteria": criteria,
            "include_missing": include_missing, "run": run}


def _run_screen(params: Mapping[str, Any]) -> None:
    universe = get_stocks_by_sector_and_subsector(params["sector"], params["subsector"]) \
        if params["sector"] != "All" else get_stock_universe()
    failures: List[str] = []
    bar = st.progress(0.0, text="Loading company data...")
    try:
        rows = screen_stocks(universe, criteria=params["criteria"], include_missing=params["include_missing"],
                             progress_callback=progress_reporter(bar, "Screening"), failures=failures)
    finally:
        bar.empty()
    st.session_state[RESULTS_KEY] = {
        "rows": rows, "failures": failures, "universe_size": len(universe),
        "signature": _signature(params), "sector": params["sector"], "subsector": params["subsector"],
        "criteria": dict(params["criteria"]), "include_missing": params["include_missing"],
    }


def _signature(params: Mapping[str, Any]) -> Tuple:
    return (params["sector"], params["subsector"], params["include_missing"],
            tuple(sorted((k, v) for k, v in params["criteria"].items())))


def _describe_criteria(criteria: Mapping[str, Tuple[Optional[float], Optional[float]]]) -> str:
    parts = []
    for metric, (lo, hi) in criteria.items():
        label = metric_label(metric)
        if lo is not None and hi is not None:
            parts.append(f"{format_metric(lo, metric)} <= {label} <= {format_metric(hi, metric)}")
        elif lo is not None:
            parts.append(f"{label} >= {format_metric(lo, metric)}")
        else:
            parts.append(f"{label} <= {format_metric(hi, metric)}")
    return "; ".join(parts) if parts else "no ratio filters"


def _render_results(results: Mapping[str, Any], params: Mapping[str, Any], visualizer: FinancialVisualizer) -> None:
    rows: List[Dict[str, Any]] = results["rows"]
    scope = results["sector"] if results["sector"] != "All" else "all sectors"
    if results["subsector"] != "All":
        scope += f" / {results['subsector']}"
    st.subheader("Screening results")
    st.caption(f"Universe: {scope} ({results['universe_size']} stocks). "
               f"Criteria: {_describe_criteria(results['criteria'])}.")
    if results["signature"] != _signature(params):
        st.info("Filters changed since the last run - click **Run screen** to refresh the results.",
                icon=":material/info:")
    if results["failures"]:
        st.warning(f"Data unavailable for {len(results['failures'])} stocks (skipped): "
                   f"{', '.join(results['failures'][:15])}{'...' if len(results['failures']) > 15 else ''}")
    if not rows:
        st.warning("No stocks match your criteria. Try relaxing some filters or enabling "
                   "**Keep stocks with missing data**.")
        return
    st.success(f"{len(rows)} of {results['universe_size']} stocks match.")

    metric_columns = list(dict.fromkeys(TABLE_METRICS + [m for m in results["criteria"] if m in METRICS]))
    tab_table, tab_charts, tab_performance, tab_details = st.tabs(
        ["Results", "Charts", "Price performance", "Stock details"])

    with tab_table:
        col_sort, col_order = st.columns([3, 1], vertical_alignment="bottom")
        with col_sort:
            sort_metric = st.selectbox("Sort by", metric_columns, key=f"{PREFIX}_sort_by", format_func=metric_label,
                                       index=metric_columns.index("market_cap"))
        with col_order:
            descending = st.toggle("Descending", value=True, key=f"{PREFIX}_sort_desc")
        ordered = sorted(rows, key=lambda r: (to_float(r.get(sort_metric)) is None,
                                              -(to_float(r.get(sort_metric)) or 0) if descending
                                              else (to_float(r.get(sort_metric)) or 0)))
        table = metrics_table(ordered, metric_columns, id_columns=(("symbol", "Symbol"), ("name", "Company"),
                                                                   ("sector", "Sector"), ("subsector", "Subsector")))
        st.dataframe(table, hide_index=True, column_config=numeric_column_config(table), key=f"{PREFIX}_table")
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name="stock_screener_results.csv", mime="text/csv", key=f"{PREFIX}_download",
                           icon=":material/download:")
        render_analyze_jump([r["symbol"] for r in ordered], PREFIX, {r["symbol"]: r["name"] for r in rows})

    with tab_charts:
        group = "sector" if len({r["sector"] for r in rows}) > 1 else "subsector"
        counts = pd.Series([r[group] for r in rows]).value_counts()
        st.plotly_chart(visualizer.plot_sector_distribution(
            pd.DataFrame({"Sector": counts.index, "Count": counts.values}),
            title=f"Matches by {group}", height=400), key=f"{PREFIX}_chart_distribution")
        chart_metric = st.selectbox("Compare metric", metric_columns, key=f"{PREFIX}_chart_metric",
                                    format_func=metric_label, index=metric_columns.index("pe_ratio"))
        kind = METRICS[chart_metric]["kind"]
        values = [(r["symbol"], to_display(r.get(chart_metric), kind)) for r in rows]
        values = sorted([v for v in values if np.isfinite(v[1])], key=lambda v: v[1], reverse=True)[:30]
        st.plotly_chart(visualizer.plot_metrics_bar_chart(
            pd.DataFrame(values, columns=["Metric", "Value"]), title=metric_label(chart_metric, with_unit=True),
            height=420, is_percent=kind == "percent"), key=f"{PREFIX}_chart_metric_bars")

    with tab_performance:
        top = sorted(rows, key=lambda r: to_float(r.get("market_cap")) or 0, reverse=True)[:10]
        series, names = {}, {}
        for row in top:
            snapshot = get_stock_snapshot(row["symbol"])
            close = trailing_window(close_prices(snapshot["price_data"] if snapshot else None), 365)
            if len(close) > 1:
                series[row["symbol"]] = (close / close.iloc[0] - 1) * 100
                names[row["symbol"]] = row["name"]
        if series:
            st.caption("Largest matches by market capitalisation (up to 10).")
            st.plotly_chart(visualizer.plot_price_comparison(series, names, title="1-Year Price Performance",
                                                             height=500), key=f"{PREFIX}_chart_performance")
        else:
            st.info("No price history available for the matching stocks.")

    with tab_details:
        symbols = [r["symbol"] for r in rows]
        pick = st.selectbox("Stock", symbols, key=f"{PREFIX}_detail_pick",
                            format_func=lambda s: f"{s} - {next(r['name'] for r in rows if r['symbol'] == s)}")
        row = next(r for r in rows if r["symbol"] == pick)
        cols = st.columns(4)
        for col, metric in zip(cols, ("price", "market_cap", "pe_ratio", "dividend_yield")):
            col.metric(metric_label(metric), format_metric(row.get(metric), metric))
        category = st.segmented_control("Metric group", [c for c in METRIC_CATEGORIES if c != "Market"],
                                        default="Profitability", key=f"{PREFIX}_detail_group") or "Profitability"
        keys = METRIC_CATEGORIES[category]
        percent = all(METRICS[k]["kind"] == "percent" for k in keys)
        frame = pd.DataFrame({"Metric": [metric_label(k) for k in keys],
                              "Value": [to_display(row.get(k), METRICS[k]["kind"]) for k in keys]})
        st.plotly_chart(visualizer.plot_metrics_bar_chart(frame, title=f"{pick} - {category}", height=380,
                                                          is_percent=percent), key=f"{PREFIX}_detail_metrics")
        snapshot = get_stock_snapshot(pick)
        if snapshot is not None and not snapshot["price_data"].empty:
            st.plotly_chart(visualizer.plot_stock_price(snapshot["price_data"], pick, company_name=row["name"],
                                                        height=500), key=f"{PREFIX}_detail_price")
        else:
            st.info(f"No price history available for {pick}.")


def run_stock_screener_page() -> None:
    """Render the Stock Screener page."""
    st.title("Stock Screener")
    st.markdown("Screen the curated universe on valuation, profitability, growth, financial health, dividends "
                "and momentum. Results stay on screen until you run a new screen.")
    visualizer = FinancialVisualizer(theme="dark")
    params = _render_controls()
    if params["run"]:
        _run_screen(params)
    results = st.session_state.get(RESULTS_KEY)
    if not results:
        st.info("Choose a universe and filters in the sidebar, then click **Run screen**.", icon=":material/tune:")
        return
    _render_results(results, params, visualizer)


if __name__ == "__main__":  # pragma: no cover - run with ``streamlit run``
    run_stock_screener_page()
