"""
Shared building blocks for the sector-specific valuation models.

Used by the retail, manufacturing, real estate, healthcare and communication
models. Provides data preparation, CAPM / WACC, the enterprise-to-equity value
bridge, a revenue-driven FCFF DCF engine, multiples and dividend discount
helpers, result sanitising and :class:`SectorValuationMixin`, which implements
the unified ``get_valuation`` flow documented in ``BaseValuation``.

All monetary amounts are in the reporting currency of the statements; rates
are fractions (0.08 == 8 %).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from StockAnalysisSystem.src.config import (
    DCF_PARAMETERS,
    RISK_FREE_RATE,
    SECTOR_DCF_PARAMETERS,
    normalize_sector,
)
from StockAnalysisSystem.src.utils.financial_data import (
    build_financial_data,
    cagr,
    first_available,
    get_latest,
    get_series,
    is_empty,
    normalize_dividend_yield,
    safe_divide,
    to_float,
)

logger = logging.getLogger(__name__)

#: Mature-market equity risk premium used in CAPM.
EQUITY_RISK_PREMIUM = 0.055
#: Bounds applied to a CAPM-derived WACC.
MIN_DISCOUNT_RATE = 0.06
MAX_DISCOUNT_RATE = 0.20
DEFAULT_TAX_RATE = 0.21
MAX_TAX_RATE = 0.35
#: Per-share values above this multiple of the share price are treated as model failures.
MAX_VALUE_TO_PRICE = 20.0

STATEMENT_KEYS = ("income_statement", "balance_sheet", "cash_flow")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def clean(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to Python types; NaN and inf become None."""
    if isinstance(value, dict):
        return {(k.strftime("%Y-%m-%d") if isinstance(k, pd.Timestamp) else k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def clip(value: Optional[float], low: float, high: float) -> Optional[float]:
    """Clamp ``value`` to [low, high]; None stays None."""
    value = to_float(value)
    return None if value is None else float(min(max(value, low), high))


def growth_path(start: float, end: float, years: int) -> List[float]:
    """Growth rates fading linearly from ``start`` (year 1) to ``end`` (final year)."""
    years = max(int(years), 1)
    if years == 1:
        return [float(start)]
    return [float(start + (end - start) * i / (years - 1)) for i in range(years)]


def keyword_match(text: str, rules: Sequence[Tuple[str, Sequence[str]]]) -> Optional[str]:
    """Return the first key whose keywords occur in ``text`` (lower-case)."""
    text = (text or "").lower()
    if not text:
        return None
    for key, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return key
    return None


def not_applicable(reason: str, **details: Any) -> Dict[str, Any]:
    """Method result for a method that cannot be applied to this company / data."""
    return {"value_per_share": None, "applicable": False, "note": reason, **details}


def finalize_method(result: Dict[str, Any], price: Optional[float]) -> Dict[str, Any]:
    """
    Sanitise a method result: finite numbers only, and ``value_per_share`` set to
    None (with a note) when it is non-positive or implausibly large.
    """
    result = dict(result)
    value = to_float(result.get("value_per_share"))
    if value is None:
        result["value_per_share"] = None
    elif value <= 0:
        result["value_per_share"] = None
        result["implied_value_per_share"] = value
        result.setdefault("note", "Implied equity value is not positive; the method is not meaningful here.")
    elif price and value / price > MAX_VALUE_TO_PRICE:
        result["value_per_share"] = None
        result["implied_value_per_share"] = value
        result.setdefault("note", f"Implausible result ({value / price:.0f}x the share price) excluded.")
    else:
        result["value_per_share"] = value
    return clean(result)


def _datetime_index(series: pd.Series) -> pd.Series:
    try:
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    except (TypeError, ValueError):
        pass
    return series


def series_by_period(df: Optional[pd.DataFrame], item: Any) -> pd.Series:
    """Chronological series of ``item`` indexed by period end date."""
    return _datetime_index(get_series(df, item, ascending=True))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def get_statements(financial_data: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(income statement, balance sheet, cash flow) as DataFrames (empty when missing)."""
    financial_data = financial_data or {}
    frames = []
    for key in STATEMENT_KEYS:
        df = financial_data.get(key)
        frames.append(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    return frames[0], frames[1], frames[2]


def load_financial_data(data_loader: Any, ticker: str) -> Dict[str, Any]:
    """Load the canonical financial data bundle for ``ticker`` through ``data_loader``."""
    if data_loader is None:
        raise ValueError(f"No financial data supplied for {ticker} and no data loader available")
    statements = {
        "income_statement": data_loader.get_financial_statements(ticker, "income", "annual"),
        "balance_sheet": data_loader.get_financial_statements(ticker, "balance", "annual"),
        "cash_flow": data_loader.get_financial_statements(ticker, "cash", "annual"),
    }
    company_info = data_loader.get_company_info(ticker) or {}
    prices = None
    try:
        start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = data_loader.get_historical_prices(ticker, start_date=start)
    except Exception as exc:  # price history is optional
        logger.info(f"No recent prices for {ticker}: {exc}")
    return build_financial_data(company_info=company_info, price_data=prices, **statements)


def prepare_financial_data(financial_data: Optional[Dict[str, Any]], ticker: str,
                           data_loader: Any = None) -> Dict[str, Any]:
    """
    Return a complete canonical bundle: statements as DataFrames and market data
    filled from company info / statements where the caller left gaps.
    """
    if financial_data is None:
        financial_data = load_financial_data(data_loader, ticker)
    income, balance, cash_flow = get_statements(financial_data)
    company_info = dict(financial_data.get("company_info") or {})
    market = dict(financial_data.get("market_data") or {})
    derived = build_financial_data(income, balance, cash_flow, company_info)["market_data"]
    for key, value in derived.items():
        if to_float(market.get(key)) is None and value is not None:
            market[key] = value
    for key in ("share_price", "market_cap", "shares_outstanding", "beta"):
        market[key] = to_float(market.get(key))
    if market["share_price"] is None and market["market_cap"] and market["shares_outstanding"]:
        market["share_price"] = market["market_cap"] / market["shares_outstanding"]
    if market["market_cap"] is None and market["share_price"] and market["shares_outstanding"]:
        market["market_cap"] = market["share_price"] * market["shares_outstanding"]
    market["dividend_yield"] = normalize_dividend_yield(market.get("dividend_yield"))
    return {**financial_data, "income_statement": income, "balance_sheet": balance, "cash_flow": cash_flow,
            "company_info": company_info, "market_data": market}


# ---------------------------------------------------------------------------
# Corporate finance building blocks
# ---------------------------------------------------------------------------
def capital_structure(balance: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Latest debt, cash (incl. short-term investments), minority interest and preferred stock."""
    debt = get_latest(balance, "total_debt")
    if debt is None:
        parts = [get_latest(balance, "long_term_debt"), get_latest(balance, "short_term_debt")]
        debt = sum(p for p in parts if p is not None)
    cash = first_available(balance, ["cash_and_short_term_investments", "cash"], 0.0)
    return {
        "total_debt": max(float(debt or 0.0), 0.0),
        "cash": max(float(cash or 0.0), 0.0),
        "minority_interest": max(float(get_latest(balance, "minority_interest", 0.0)), 0.0),
        "preferred_stock": max(float(get_latest(balance, "preferred_stock", 0.0)), 0.0),
    }


def equity_bridge(enterprise_value: float, capital: Dict[str, float]) -> Dict[str, float]:
    """Equity value = EV - debt + cash - minority interest - preferred stock."""
    net_debt = capital["total_debt"] - capital["cash"]
    equity = enterprise_value - net_debt - capital["minority_interest"] - capital["preferred_stock"]
    return {"enterprise_value": float(enterprise_value), "total_debt": capital["total_debt"],
            "cash": capital["cash"], "net_debt": float(net_debt),
            "minority_interest": capital["minority_interest"], "preferred_stock": capital["preferred_stock"],
            "equity_value": float(equity)}


def effective_tax_rate(income: Optional[pd.DataFrame]) -> float:
    """Effective tax rate from the statements, bounded to [0, 35 %]; 21 % when not meaningful."""
    rate = get_latest(income, "tax_rate")
    if rate is None:
        tax, pretax = series_by_period(income, "tax_provision"), series_by_period(income, "pretax_income")
        common = tax.index.intersection(pretax.index)
        total_pretax = float(pretax[common].sum()) if len(common) else 0.0
        rate = safe_divide(float(tax[common].sum()), total_pretax) if total_pretax > 0 else None
    if rate is None or not 0 <= rate <= 0.5:
        return DEFAULT_TAX_RATE
    return float(min(rate, MAX_TAX_RATE))


def cost_of_capital(income: pd.DataFrame, balance: pd.DataFrame, market: Dict[str, Any],
                    fallback_rate: float) -> Dict[str, Any]:
    """
    CAPM cost of equity (Blume-adjusted beta) and market-value WACC.

    Cost of debt is interest expense / total debt, bounded to
    [rf + 1 %, rf + 8 %]. The WACC is bounded to [6 %, 20 %]; ``fallback_rate``
    is used when neither market nor book equity is available.
    """
    raw_beta = to_float(market.get("beta"))
    beta = clip(0.67 * raw_beta + 0.33, 0.3, 3.0) if raw_beta is not None else 1.0
    cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM
    capital = capital_structure(balance)
    debt = capital["total_debt"]
    interest = abs(get_latest(income, "interest_expense", 0.0) or 0.0)
    pre_tax_kd = clip(safe_divide(interest, debt, RISK_FREE_RATE + 0.02), RISK_FREE_RATE + 0.01, RISK_FREE_RATE + 0.08)
    tax_rate = effective_tax_rate(income)
    equity = to_float(market.get("market_cap"))
    equity_basis = "market"
    if not equity or equity <= 0:
        equity, equity_basis = get_latest(balance, "total_equity"), "book"
    if equity is None or equity <= 0:
        wacc, equity_weight = fallback_rate, None
        equity_basis = "sector default"
    else:
        equity_weight = equity / (equity + debt)
        wacc = equity_weight * cost_of_equity + (1 - equity_weight) * pre_tax_kd * (1 - tax_rate)
    return {
        "wacc": float(clip(wacc, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)),
        "unbounded_wacc": float(wacc),
        "cost_of_equity": float(cost_of_equity),
        "pre_tax_cost_of_debt": float(pre_tax_kd),
        "after_tax_cost_of_debt": float(pre_tax_kd * (1 - tax_rate)),
        "tax_rate": tax_rate,
        "raw_beta": raw_beta,
        "adjusted_beta": float(beta),
        "risk_free_rate": RISK_FREE_RATE,
        "equity_risk_premium": EQUITY_RISK_PREMIUM,
        "equity_weight": equity_weight,
        "equity_basis": equity_basis,
    }


def operating_history(income: pd.DataFrame, cash_flow: pd.DataFrame, tax_rate: float) -> pd.DataFrame:
    """
    Chronological operating history with free cash flow to the firm:
    FCFF = EBIT (1 - t) + D&A - capex + change in working capital (cash-flow sign).

    Falls back to operating cash flow - capex when EBIT is missing. Columns:
    revenue, ebit, ebitda, da, capex, delta_wc, ocf, net_income, fcff and margins.
    """
    revenue = series_by_period(income, "revenue")
    if revenue.empty:
        return pd.DataFrame()
    da = series_by_period(cash_flow, "cash_flow_depreciation")
    if da.empty:
        da = series_by_period(income, "depreciation")
    frame = pd.DataFrame({
        "revenue": revenue,
        "ebit": series_by_period(income, ["operating_income", "ebit"]),
        "ebitda": series_by_period(income, "ebitda"),
        "da": da,
        "capex": series_by_period(cash_flow, "capital_expenditure").abs(),
        "delta_wc": series_by_period(cash_flow, "change_in_working_capital"),
        "ocf": series_by_period(cash_flow, "operating_cash_flow"),
        "net_income": series_by_period(income, "net_income"),
        "rd": series_by_period(income, "rd_expense"),
    })
    frame = frame[frame["revenue"].notna()].sort_index()
    if frame.empty:
        return frame
    frame["ebitda"] = frame["ebitda"].fillna(frame["ebit"] + frame["da"])
    capex = frame["capex"].fillna(frame["da"])
    fcff = frame["ebit"] * (1 - tax_rate) + frame["da"].fillna(0.0) - capex + frame["delta_wc"].fillna(0.0)
    has_cash_flow = frame["capex"].notna() | frame["ocf"].notna()
    frame["fcff"] = fcff.where(has_cash_flow).fillna(frame["ocf"] - capex)
    positive_revenue = frame["revenue"].where(frame["revenue"] > 0)
    for column in ("fcff", "ebit", "ebitda", "net_income", "capex", "da", "rd"):
        frame[f"{column}_margin"] = frame[column] / positive_revenue
    return frame


def run_revenue_dcf(base_revenue: float, growth_rates: Sequence[float], margins: Sequence[float],
                    discount_rate: float, terminal_growth: float) -> Dict[str, Any]:
    """
    Revenue-driven DCF: revenue grows along ``growth_rates``; FCF = revenue x margin.
    Terminal value uses the Gordon growth model on year n+1 cash flow.

    Raises:
        ValueError: when the discount rate does not exceed the terminal growth rate.
    """
    if discount_rate <= terminal_growth:
        raise ValueError(f"Discount rate ({discount_rate:.2%}) must exceed terminal growth "
                         f"({terminal_growth:.2%}) for a Gordon growth terminal value")
    revenue, projections, pv_total = float(base_revenue), [], 0.0
    for year, (growth, margin) in enumerate(zip(growth_rates, margins), start=1):
        revenue *= 1 + growth
        fcf = revenue * margin
        factor = (1 + discount_rate) ** -year
        pv_total += fcf * factor
        projections.append({"year": year, "revenue": revenue, "growth": growth, "fcf_margin": margin,
                            "fcf": fcf, "discount_factor": factor, "present_value": fcf * factor})
    last = projections[-1]
    terminal_value = last["fcf"] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value * last["discount_factor"]
    enterprise_value = pv_total + pv_terminal
    return {
        "projections": projections,
        "forecast_fcf": [row["fcf"] for row in projections],
        "pv_forecast": pv_total,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "terminal_value_share": safe_divide(pv_terminal, enterprise_value),
        "enterprise_value": enterprise_value,
    }


def two_stage_ddm(dps: float, cost_of_equity: float, initial_growth: float, terminal_growth: float,
                  years: int) -> Dict[str, Any]:
    """Two-stage dividend discount model with growth fading to a Gordon terminal stage."""
    if cost_of_equity <= terminal_growth:
        raise ValueError(f"Cost of equity ({cost_of_equity:.2%}) must exceed terminal growth "
                         f"({terminal_growth:.2%})")
    dividend, projections, pv_total = float(dps), [], 0.0
    for year, growth in enumerate(growth_path(initial_growth, terminal_growth, years), start=1):
        dividend *= 1 + growth
        factor = (1 + cost_of_equity) ** -year
        pv_total += dividend * factor
        projections.append({"year": year, "dividend": dividend, "growth": growth, "present_value": dividend * factor})
    terminal_value = dividend * (1 + terminal_growth) / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value * (1 + cost_of_equity) ** -len(projections)
    return {"projections": projections, "pv_dividends": pv_total, "terminal_value": terminal_value,
            "pv_terminal": pv_terminal, "value_per_share": pv_total + pv_terminal}


# ---------------------------------------------------------------------------
# Valuation context and shared methods
# ---------------------------------------------------------------------------
@dataclass
class ValuationContext:
    """Everything a valuation method needs for one company."""

    ticker: str
    financial_data: Dict[str, Any]
    subsector: str
    sector: str
    params: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def income(self) -> pd.DataFrame:
        return self.financial_data["income_statement"]

    @property
    def balance(self) -> pd.DataFrame:
        return self.financial_data["balance_sheet"]

    @property
    def cash_flow(self) -> pd.DataFrame:
        return self.financial_data["cash_flow"]

    @property
    def company_info(self) -> Dict[str, Any]:
        return self.financial_data.get("company_info") or {}

    @property
    def market(self) -> Dict[str, Any]:
        return self.financial_data["market_data"]

    @property
    def price(self) -> Optional[float]:
        price = to_float(self.market.get("share_price"))
        return price if price and price > 0 else None

    @property
    def shares(self) -> Optional[float]:
        shares = to_float(self.market.get("shares_outstanding"))
        return shares if shares and shares > 0 else None

    @property
    def sector_defaults(self) -> Dict[str, Any]:
        return {**DCF_PARAMETERS, **SECTOR_DCF_PARAMETERS.get(self.sector, {})}

    @cached_property
    def capital(self) -> Dict[str, float]:
        return capital_structure(self.balance)

    @cached_property
    def tax_rate(self) -> float:
        return effective_tax_rate(self.income)

    @cached_property
    def history(self) -> pd.DataFrame:
        return operating_history(self.income, self.cash_flow, self.tax_rate)

    @cached_property
    def cost_of_capital(self) -> Dict[str, Any]:
        return cost_of_capital(self.income, self.balance, self.market, self.sector_defaults["default_discount_rate"])

    def latest(self, statement: str, item: Any, default: Optional[float] = None) -> Optional[float]:
        return get_latest(self.financial_data.get(statement), item, default)

    def per_share(self, equity_value: Optional[float]) -> Optional[float]:
        return safe_divide(equity_value, self.shares)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


def latest_metric(ctx: ValuationContext, column: str) -> Optional[float]:
    """Most recent non-missing value of an ``operating_history`` column."""
    history = ctx.history
    if history.empty or column not in history:
        return None
    values = history[column].dropna()
    return float(values.iloc[-1]) if len(values) else None


def average_margin(ctx: ValuationContext, column: str, years: Optional[int] = None) -> Optional[float]:
    """Average of ``<column>_margin`` over the last ``years`` periods (all when None)."""
    history = ctx.history
    name = f"{column}_margin"
    if history.empty or name not in history:
        return None
    values = history[name].dropna()
    if years:
        values = values.tail(int(years))
    return float(values.mean()) if len(values) else None


def revenue_growth_estimate(ctx: ValuationContext, low: float = -0.05, high: float = 0.15) -> float:
    """Historical revenue CAGR bounded to [low, high]; terminal growth when unavailable."""
    growth = cagr(series_by_period(ctx.income, "revenue"))
    if growth is None:
        return float(ctx.sector_defaults["terminal_growth_rate"])
    return float(min(max(growth, low), high))


def fcff_dcf(ctx: ValuationContext, margin_years: Optional[int] = 3, margin_label: str = "normalised") -> Dict[str, Any]:
    """
    Revenue-driven FCFF DCF: latest revenue grown at ``revenue_growth`` fading
    to ``terminal_growth``, FCF margin = average FCFF margin of the last
    ``margin_years`` periods (all periods when None), discounted at
    ``discount_rate``; equity via the EV bridge.
    """
    history = ctx.history.dropna(subset=["fcff"]) if not ctx.history.empty else ctx.history
    if history.empty:
        raise ValueError("Revenue and free cash flow history are required for a DCF")
    window = history.tail(int(margin_years)) if margin_years else history
    margin = float(window["fcff_margin"].dropna().mean()) if window["fcff_margin"].notna().any() else None
    base_revenue = float(ctx.history["revenue"].dropna().iloc[-1])
    if margin is None or base_revenue <= 0:
        return not_applicable("Positive revenue is required for a revenue-driven DCF.")
    historical_fcf = {period: value for period, value in history["fcff"].items()}
    if margin <= 0:
        return not_applicable(f"The {margin_label} free cash flow margin is negative ({margin:.1%}); "
                              "a DCF would not be meaningful.", fcf_margin=margin, historical_fcf=historical_fcf)
    rate, terminal, years = ctx.params["discount_rate"], ctx.params["terminal_growth"], ctx.params["forecast_years"]
    growth = ctx.params["revenue_growth"]
    dcf = run_revenue_dcf(base_revenue, growth_path(growth, terminal, years), [margin] * years, rate, terminal)
    bridge = equity_bridge(dcf["enterprise_value"], ctx.capital)
    return {
        **dcf, **bridge,
        "value_per_share": ctx.per_share(bridge["equity_value"]),
        "discount_rate": rate, "terminal_growth": terminal, "revenue_growth": growth, "forecast_years": years,
        "base_revenue": base_revenue, "fcf_margin": margin, "margin_years": int(len(window)),
        "historical_fcf": historical_fcf,
    }


def current_multiples(ctx: ValuationContext) -> Dict[str, Optional[float]]:
    """Market-implied multiples at the current price (for context)."""
    market_cap = to_float(ctx.market.get("market_cap"))
    if not market_cap:
        return {}
    capital = ctx.capital
    ev = market_cap + capital["total_debt"] - capital["cash"] + capital["minority_interest"] + capital["preferred_stock"]
    ebitda, revenue = latest_metric(ctx, "ebitda"), latest_metric(ctx, "revenue")
    earnings, book = ctx.latest("income_statement", "net_income"), ctx.latest("balance_sheet", "total_equity")
    return {
        "enterprise_value": ev,
        "ev_ebitda": safe_divide(ev, ebitda) if ebitda and ebitda > 0 else None,
        "pe": safe_divide(market_cap, earnings) if earnings and earnings > 0 else None,
        "ev_sales": safe_divide(ev, revenue) if revenue and revenue > 0 else None,
        "pb": safe_divide(market_cap, book) if book and book > 0 else None,
    }


def multiples_valuation(ctx: ValuationContext, entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Value the company on benchmark multiples.

    Each entry: ``key``, ``label``, ``metric`` (name), ``metric_value``,
    ``multiple``, ``basis`` (``"ev"`` -> EV bridge to equity, ``"equity"``)
    and ``weight``. The result blends the per-share values of the applicable
    multiples by weight.
    """
    capital = ctx.capital
    claims = capital["total_debt"] - capital["cash"] + capital["minority_interest"] + capital["preferred_stock"]
    applied, excluded = {}, {}
    for entry in entries:
        key, metric = entry["key"], entry["metric"]
        value, multiple = to_float(entry.get("metric_value")), to_float(entry.get("multiple"))
        if value is None:
            excluded[key] = f"{metric} not available"
            continue
        if value <= 0:
            excluded[key] = f"{metric} is not positive"
            continue
        if not multiple or multiple <= 0:
            excluded[key] = "no benchmark multiple"
            continue
        ev = value * multiple if entry["basis"] == "ev" else None
        equity = ev - claims if ev is not None else value * multiple
        per_share = ctx.per_share(equity)
        if per_share is None or per_share <= 0:
            excluded[key] = "implied equity value is not positive" if ctx.shares else "shares outstanding unavailable"
            continue
        applied[key] = {"label": entry["label"], "metric": metric, "metric_value": value, "multiple": multiple,
                        "implied_enterprise_value": ev, "implied_equity_value": equity,
                        "value_per_share": per_share, "weight": float(entry.get("weight", 1.0))}
    result = {"multiples": applied, "excluded": excluded, "current_multiples": current_multiples(ctx),
              "net_debt": capital["total_debt"] - capital["cash"]}
    total_weight = sum(item["weight"] for item in applied.values())
    if not applied or total_weight <= 0:
        reasons = "; ".join(f"{k}: {v}" for k, v in excluded.items()) or "no multiples configured"
        return {**result, **not_applicable(f"No multiple could be applied ({reasons}).")}
    result["value_per_share"] = sum(i["value_per_share"] * i["weight"] for i in applied.values()) / total_weight
    result["equity_value"] = sum(i["implied_equity_value"] * i["weight"] for i in applied.values()) / total_weight
    return result


def dividends_per_share(ctx: ValuationContext) -> Optional[float]:
    """Latest dividends paid per share (cash flow), falling back to dividend yield x price."""
    paid = ctx.latest("cash_flow", "dividends_paid")
    if paid and ctx.shares:
        return abs(paid) / ctx.shares
    dividend_yield = to_float(ctx.market.get("dividend_yield"))
    if dividend_yield and ctx.price:
        return dividend_yield * ctx.price
    return None


def dividend_discount(ctx: ValuationContext, max_growth: float = 0.08) -> Dict[str, Any]:
    """Two-stage DDM at the CAPM cost of equity (``cost_of_equity`` parameter)."""
    dps = dividends_per_share(ctx)
    if not dps or dps <= 0:
        return not_applicable("The company does not pay a dividend.")
    ke = ctx.params.get("cost_of_equity") or ctx.cost_of_capital["cost_of_equity"]
    terminal = ctx.params["terminal_growth"]
    growth = cagr(series_by_period(ctx.cash_flow, "dividends_paid").abs())
    source = "historical dividend growth"
    if growth is None:
        roe = safe_divide(ctx.latest("income_statement", "net_income"), ctx.latest("balance_sheet", "total_equity"))
        payout = safe_divide(dps * (ctx.shares or 0), ctx.latest("income_statement", "net_income"))
        growth = roe * (1 - min(max(payout or 0, 0), 1)) if roe and roe > 0 else terminal
        source = "sustainable growth (ROE x retention)"
    growth = float(min(max(growth, 0.0), max_growth))
    ddm = two_stage_ddm(dps, ke, growth, terminal, ctx.params["forecast_years"])
    return {**ddm, "dividend_per_share": dps, "cost_of_equity": ke, "initial_growth": growth,
            "growth_source": source, "terminal_growth": terminal}


# ---------------------------------------------------------------------------
# Unified get_valuation flow
# ---------------------------------------------------------------------------
Runner = Callable[[ValuationContext], Dict[str, Any]]


class SectorValuationMixin:
    """
    Implements the unified valuation contract on top of ``BaseValuation``.

    Subclasses define the contract class attributes plus:

    * ``SUBSECTOR_KEYWORDS``: ordered (subsector, industry keywords) rules.
    * ``SUBSECTOR_ALIASES``: legacy / alternative subsector names -> key.
    * ``SUBSECTOR_PARAMETER_DEFAULTS``: per-subsector PARAMETERS defaults.
    * ``SUBSECTOR_WEIGHTS``: per-subsector method weights.
    * ``EXTRA_INPUTS``: accepted non-slider kwargs (e.g. ``pipeline_data``).
    * ``_method_runners()`` and ``_parameter_derivations(ctx)``.
    """

    SUBSECTOR_KEYWORDS: Sequence[Tuple[str, Sequence[str]]] = ()
    SUBSECTOR_ALIASES: Dict[str, str] = {}
    SUBSECTOR_PARAMETER_DEFAULTS: Dict[str, Dict[str, float]] = {}
    SUBSECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {}
    EXTRA_INPUTS: Tuple[str, ...] = ()

    # ----- subsectors -------------------------------------------------------
    def normalize_subsector(self, subsector: Optional[str]) -> Optional[str]:
        """Map a subsector key, label or legacy name to a ``SUBSECTORS`` key (None if unknown)."""
        if subsector is None:
            return None
        name = str(subsector).strip()
        if name in self.SUBSECTORS:
            return name
        lowered = name.lower().replace("-", "_").replace(" ", "_")
        for key, label in self.SUBSECTORS.items():
            if lowered == key or name.lower() == str(label).lower():
                return key
        aliases = {k.lower(): v for k, v in self.SUBSECTOR_ALIASES.items()}
        return aliases.get(name.lower()) or aliases.get(lowered)

    def detect_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                         company_info: Optional[Dict[str, Any]] = None) -> str:
        """Subsector key from the company's industry, falling back to sector and financials."""
        info = company_info or (financial_data or {}).get("company_info") or {}
        match = keyword_match(str(info.get("industry") or ""), self.SUBSECTOR_KEYWORDS)
        if match:
            return match
        try:
            from StockAnalysisSystem.src.industry.universe import lookup_company

            record = lookup_company(ticker or info.get("symbol") or "")
            if record:
                match = keyword_match(record.get("subsector", ""), self.SUBSECTOR_KEYWORDS)
                if match:
                    return match
        except ImportError:  # pragma: no cover
            pass
        return self._fallback_subsector(info, financial_data or {})

    def _fallback_subsector(self, company_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        return next(iter(self.SUBSECTORS))

    def _sector_for_subsector(self, subsector: str) -> str:
        return self.SECTOR

    # ----- contract overrides ----------------------------------------------
    def get_available_methods(self, subsector: Optional[str] = None) -> Dict[str, str]:
        keys = self.SUBSECTOR_METHODS.get(self.normalize_subsector(subsector)) if subsector else None
        return {k: self.METHODS[k] for k in (keys or self.METHODS) if k in self.METHODS}

    def get_parameters(self, subsector: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        key = self.normalize_subsector(subsector)
        overrides = self.SUBSECTOR_PARAMETER_DEFAULTS.get(key, {})
        params = {}
        for name, spec in self.PARAMETERS.items():
            if spec.get("subsectors") and key not in spec["subsectors"]:
                continue
            params[name] = {**spec, **({"default": overrides[name]} if name in overrides else {})}
        return params

    def get_default_weights(self, subsector: Optional[str] = None) -> Dict[str, float]:
        return dict(self.SUBSECTOR_WEIGHTS.get(self.normalize_subsector(subsector), self.DEFAULT_METHOD_WEIGHTS))

    def get_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                      subsector: Optional[str] = None, sector: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Run the selected methods (``methods``; default: all available for the
        subsector) and blend them with ``weights`` (default: subsector weights).
        PARAMETERS may be passed as keyword arguments; unknown kwargs are ignored.
        Never raises: total failures return ``build_error_result``.
        """
        try:
            kwargs = self._translate_legacy_kwargs(dict(kwargs))
            ctx = self._make_context(ticker, financial_data, subsector, sector, kwargs)
            methods = {key: self._execute(key, ctx) for key in self._select_methods(ctx, kwargs.get("methods"))}
            weights = self.get_default_weights(ctx.subsector)
            if isinstance(kwargs.get("weights"), dict):
                weights.update({k: v for k, v in kwargs["weights"].items() if to_float(v) is not None})
            return self.build_valuation_result(
                ticker, ctx.financial_data, methods, weights=weights, sector=ctx.sector,
                subsector=ctx.subsector, assumptions=self._assumptions(ctx), notes=list(ctx.notes))
        except Exception as exc:
            logger.warning(f"{self.__class__.__name__} valuation failed for {ticker}: {exc}")
            return self.build_error_result(ticker, exc, sector=normalize_sector(sector) if sector else self.SECTOR,
                                           subsector=self.normalize_subsector(subsector))

    def run_method(self, method_key: str, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                   subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run a single method and return its result dict (used by the legacy entry points)."""
        try:
            kwargs = self._translate_legacy_kwargs(dict(kwargs))
            ctx = self._make_context(ticker, financial_data, subsector, kwargs.get("sector"), kwargs)
        except Exception as exc:
            return {"company": ticker, "method": self.METHODS.get(method_key, method_key),
                    "method_key": method_key, "value_per_share": None, "error": str(exc)}
        return {"company": ticker, "subsector": ctx.subsector, **self._execute(method_key, ctx)}

    # ----- hooks ------------------------------------------------------------
    def _method_runners(self) -> Dict[str, Runner]:
        raise NotImplementedError

    def _parameter_derivations(self, ctx: ValuationContext) -> Dict[str, Callable[[], Any]]:
        """{parameter: zero-arg callable} deriving each PARAMETER from data (resolved in order)."""
        return common_derivations(ctx)

    def _translate_legacy_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return kwargs

    def _enrich_context(self, ctx: ValuationContext) -> None:
        """Hook for model-specific notes / assumptions once parameters are resolved."""

    # ----- internals --------------------------------------------------------
    def _make_context(self, ticker: str, financial_data: Optional[Dict[str, Any]], subsector: Optional[str],
                      sector: Optional[str], kwargs: Dict[str, Any]) -> ValuationContext:
        data = prepare_financial_data(financial_data, ticker, getattr(self, "data_loader", None))
        if all(is_empty(data[key]) for key in STATEMENT_KEYS):
            raise ValueError(f"No financial statements available for {ticker}")
        info = data["company_info"]
        key = self.normalize_subsector(subsector)
        notes = []
        if subsector is not None and key is None:
            notes.append(f"Unknown subsector '{subsector}'; detected from company data instead.")
        key = key or self.detect_subsector(ticker, data, info)
        canonical = normalize_sector(sector) if sector else "Unknown"
        if canonical == "Unknown" or canonical not in SECTOR_DCF_PARAMETERS:
            reported = normalize_sector(info.get("sector"))
            canonical = reported if reported in SECTOR_DCF_PARAMETERS else self._sector_for_subsector(key)
        ctx = ValuationContext(ticker=ticker, financial_data=data, subsector=key, sector=canonical, notes=notes,
                               inputs={name: kwargs[name] for name in self.EXTRA_INPUTS if name in kwargs})
        if not ctx.shares:
            ctx.note("Shares outstanding unavailable; per-share values cannot be computed.")
        self._resolve_parameters(ctx, kwargs)
        self._enrich_context(ctx)
        return ctx

    def _resolve_parameters(self, ctx: ValuationContext, kwargs: Dict[str, Any]) -> None:
        derivations = self._parameter_derivations(ctx)
        overrides = self.SUBSECTOR_PARAMETER_DEFAULTS.get(ctx.subsector, {})
        for name, spec in self.PARAMETERS.items():
            spec = {**spec, "default": overrides.get(name, spec.get("default"))}
            value, source = to_float(kwargs.get(name)), "user"
            if value is not None and spec.get("optional") and value <= 0:
                value = None
            if value is not None:
                low, high = spec.get("min"), spec.get("max")
                bounded = min(max(value, low if low is not None else value), high if high is not None else value)
                if bounded != value:
                    ctx.note(f"{spec.get('label', name)} limited to the allowed range ({bounded:g}).")
                value = bounded
            else:
                source = "derived"
                derive = derivations.get(name)
                try:
                    value = to_float(derive()) if derive else None
                except Exception as exc:  # derivation is best effort
                    logger.info(f"Could not derive {name} for {ctx.ticker}: {exc}")
                    value = None
                if value is None and not spec.get("optional"):
                    value, source = to_float(spec.get("default")), "default"
                if value is None:
                    source = "not provided"
            if value is not None and spec.get("kind") == "int":
                value = int(round(value))
            ctx.params[name], ctx.sources[name] = value, source

    def _select_methods(self, ctx: ValuationContext, requested: Any) -> List[str]:
        available = list(self.get_available_methods(ctx.subsector))
        if not requested:
            return available
        requested = [requested] if isinstance(requested, str) else list(requested)
        labels = {str(label).lower(): key for key, label in self.METHODS.items()}
        selected = []
        for item in requested:
            key = item if item in self.METHODS else labels.get(str(item).lower())
            if key and key not in selected:
                selected.append(key)
        if not selected:
            ctx.note(f"None of the requested methods {requested} exist; all available methods were run.")
            return available
        return selected

    def _execute(self, key: str, ctx: ValuationContext) -> Dict[str, Any]:
        label = self.METHODS.get(key, key)
        runner = self._method_runners().get(key)
        if runner is None:
            return {"method": label, "method_key": key, "value_per_share": None, "error": f"Unknown method '{key}'"}
        try:
            result = runner(ctx) or {}
        except Exception as exc:
            logger.info(f"{label} failed for {ctx.ticker}: {exc}")
            result = {"value_per_share": None, "error": str(exc)}
        result.update(method=label, method_key=key)
        return finalize_method(result, ctx.price)

    def _assumptions(self, ctx: ValuationContext) -> Dict[str, Any]:
        applicable = self.get_parameters(ctx.subsector)
        values = {name: ctx.params.get(name) for name in ctx.params
                  if name in applicable or ctx.sources.get(name) == "user"}
        sources = {name: ctx.sources[name] for name in values}
        return clean({**values, **ctx.assumptions, "parameter_sources": sources,
                      "cost_of_capital": ctx.cost_of_capital, "tax_rate": ctx.tax_rate})


def common_derivations(ctx: ValuationContext) -> Dict[str, Callable[[], Any]]:
    """Derivations shared by every model: WACC, terminal growth, horizon and revenue growth."""
    return {
        "discount_rate": lambda: ctx.cost_of_capital["wacc"],
        "terminal_growth": lambda: ctx.sector_defaults["terminal_growth_rate"],
        "forecast_years": lambda: ctx.sector_defaults["forecast_years"],
        "revenue_growth": lambda: revenue_growth_estimate(ctx),
        "cost_of_equity": lambda: ctx.cost_of_capital["cost_of_equity"],
    }


# Shared PARAMETERS specs -----------------------------------------------------
def percent_param(label: str, low: float, high: float, default: float, help_text: str,
                  subsectors: Optional[List[str]] = None, step: float = 0.0025, **extra: Any) -> Dict[str, Any]:
    return {"label": label, "min": low, "max": high, "default": default, "step": step, "kind": "percent",
            "help": help_text, "subsectors": subsectors, **extra}


def number_param(label: str, low: float, high: float, default: float, help_text: str,
                 subsectors: Optional[List[str]] = None, step: float = 0.5, kind: str = "number",
                 **extra: Any) -> Dict[str, Any]:
    return {"label": label, "min": low, "max": high, "default": default, "step": step, "kind": kind,
            "help": help_text, "subsectors": subsectors, **extra}


def core_parameters(discount_default: float = 0.09, terminal_default: float = 0.02,
                    years_default: int = 5, growth_low: float = -0.10,
                    growth_high: float = 0.30) -> Dict[str, Dict[str, Any]]:
    """Discount rate, terminal growth, forecast horizon and near-term revenue growth."""
    return {
        "discount_rate": percent_param(
            "Discount rate (WACC)", 0.04, 0.20, discount_default,
            "Defaults to a CAPM-based WACC (Blume-adjusted beta, market-value weights)."),
        "terminal_growth": percent_param(
            "Terminal growth rate", 0.0, 0.05, terminal_default,
            "Perpetual growth after the forecast period; must stay below the discount rate."),
        "forecast_years": number_param("Forecast horizon (years)", 3, 15, years_default,
                                       "Number of explicitly forecast years.", step=1, kind="int"),
        "revenue_growth": percent_param(
            "Near-term revenue growth", growth_low, growth_high, 0.04,
            "Year-1 growth, fading linearly to the terminal rate. Defaults to the historical revenue CAGR."),
    }


def multiple_parameters(subsectors: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Benchmark EV/EBITDA, P/E and EV/Sales multiples for the relative valuation."""
    return {
        "ev_ebitda_multiple": number_param("Benchmark EV/EBITDA", 2.0, 40.0, 10.0,
                                           "Peer EV/EBITDA multiple applied to EBITDA.", subsectors),
        "pe_multiple": number_param("Benchmark P/E", 4.0, 60.0, 18.0,
                                    "Peer price/earnings multiple applied to net income.", subsectors),
        "ev_sales_multiple": number_param("Benchmark EV/Sales", 0.1, 15.0, 1.5,
                                          "Peer EV/revenue multiple applied to revenue.", subsectors, step=0.1),
    }


def subsector_multiple_defaults(table: Dict[str, Tuple[float, float, float]]) -> Dict[str, Dict[str, float]]:
    """{subsector: (EV/EBITDA, P/E, EV/Sales)} -> SUBSECTOR_PARAMETER_DEFAULTS entries."""
    return {key: {"ev_ebitda_multiple": ev_ebitda, "pe_multiple": pe, "ev_sales_multiple": ev_sales}
            for key, (ev_ebitda, pe, ev_sales) in table.items()}


def merge_defaults(*tables: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Merge several {subsector: {param: default}} tables."""
    merged: Dict[str, Dict[str, float]] = {}
    for table in tables:
        for key, values in table.items():
            merged.setdefault(key, {}).update(values)
    return merged


def standard_multiples(ctx: ValuationContext, ebitda: Optional[float] = None, earnings: Optional[float] = None,
                       weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Dict[str, Any]:
    """EV/EBITDA, P/E and EV/Sales relative valuation using the benchmark-multiple parameters."""
    ebitda = latest_metric(ctx, "ebitda") if ebitda is None else ebitda
    earnings = ctx.latest("income_statement", "net_income") if earnings is None else earnings
    entries = [
        {"key": "ev_ebitda", "label": "EV/EBITDA", "metric": "EBITDA", "metric_value": ebitda,
         "multiple": ctx.params.get("ev_ebitda_multiple"), "basis": "ev", "weight": weights[0]},
        {"key": "pe", "label": "P/E", "metric": "Net income", "metric_value": earnings,
         "multiple": ctx.params.get("pe_multiple"), "basis": "equity", "weight": weights[1]},
        {"key": "ev_sales", "label": "EV/Sales", "metric": "Revenue", "metric_value": latest_metric(ctx, "revenue"),
         "multiple": ctx.params.get("ev_sales_multiple"), "basis": "ev", "weight": weights[2]},
    ]
    return multiples_valuation(ctx, entries)


SEGMENT_METRICS = (("ebitda", "ev_ebitda", "EV/EBITDA"), ("ebit", "ev_ebit", "EV/EBIT"), ("revenue", "ev_sales", "EV/Sales"))


def sum_of_parts(ctx: ValuationContext, segments: Any, corporate_costs: Any = None) -> Dict[str, Any]:
    """
    Sum-of-the-parts from segment data.

    ``segments``: list of dicts (or {name: dict}) with either ``value``
    (segment enterprise value) or a metric and multiple: ``ebitda`` +
    ``ev_ebitda``, ``ebit`` + ``ev_ebit`` or ``revenue`` + ``ev_sales``
    (``multiple`` is accepted for any of them). ``corporate_costs`` is the
    annual unallocated cost, capitalised at the blended EV/EBITDA multiple.
    """
    if isinstance(segments, dict):
        segments = [{"name": name, **spec} for name, spec in segments.items() if isinstance(spec, dict)]
    parts, skipped = [], {}
    for index, segment in enumerate(segments or [], start=1):
        if not isinstance(segment, dict):
            skipped[f"Segment {index}"] = "segment must be a dict"
            continue
        name = str(segment.get("name") or f"Segment {index}")
        ev, basis, metric_value, multiple = to_float(segment.get("value")), "stated value", None, None
        if ev is None:
            for metric, multiple_key, label in SEGMENT_METRICS:
                metric_value = to_float(segment.get(metric))
                multiple = to_float(segment.get(multiple_key) or segment.get("multiple"))
                if metric_value is not None and multiple:
                    ev, basis = metric_value * multiple, label
                    break
        if ev is None or ev < 0:
            skipped[name] = "needs 'value', or ebitda / ebit / revenue with a positive multiple"
            continue
        parts.append({"name": name, "basis": basis, "metric_value": metric_value if basis != "stated value" else None,
                      "multiple": multiple if basis != "stated value" else None, "enterprise_value": ev})
    if not parts:
        raise ValueError(f"No valid segments supplied ({skipped})")
    ebitda_parts = [p for p in parts if p["basis"] == "EV/EBITDA"]
    blended = (sum(p["enterprise_value"] for p in ebitda_parts) / sum(p["metric_value"] for p in ebitda_parts)
               if ebitda_parts else ctx.params.get("ev_ebitda_multiple") or 10.0)
    corporate_value = -abs(to_float(corporate_costs, 0.0)) * blended
    enterprise_value = sum(p["enterprise_value"] for p in parts) + corporate_value
    bridge = equity_bridge(enterprise_value, ctx.capital)
    return {**bridge, "parts": parts, "skipped_segments": skipped, "corporate_costs_value": corporate_value,
            "blended_ev_ebitda": blended, "value_per_share": ctx.per_share(bridge["equity_value"])}


def operating_ratios(ctx: ValuationContext) -> Dict[str, Dict[str, Optional[float]]]:
    """Profitability, efficiency, liquidity, leverage, growth, investment and valuation ratios."""
    def latest(statement: str, item: Any) -> Optional[float]:
        return ctx.latest(statement, item)

    revenue = latest_metric(ctx, "revenue")
    cogs = latest("income_statement", "cost_of_revenue")
    gross = latest("income_statement", "gross_profit")
    if gross is None and revenue is not None and cogs is not None:
        gross = revenue - cogs
    ebit, ebitda = latest_metric(ctx, "ebit"), latest_metric(ctx, "ebitda")
    net_income = latest("income_statement", "net_income")
    assets, equity = latest("balance_sheet", "total_assets"), latest("balance_sheet", "total_equity")
    inventory, receivables = latest("balance_sheet", "inventory"), latest("balance_sheet", "receivables")
    payables, net_ppe = latest("balance_sheet", "payables"), latest("balance_sheet", "net_ppe")
    current_assets, current_liabilities = latest("balance_sheet", "current_assets"), latest("balance_sheet",
                                                                                           "current_liabilities")
    capital = ctx.capital
    debt, cash = capital["total_debt"], capital["cash"]
    interest = abs(latest("income_statement", "interest_expense") or 0.0) or None
    invested = (equity or 0.0) + debt - cash
    inventory_turnover = safe_divide(cogs, inventory) if inventory and inventory > 0 else None
    dio = safe_divide(365.0, inventory_turnover)
    dso = safe_divide(receivables * 365.0, revenue) if receivables is not None else None
    dpo = safe_divide(payables * 365.0, cogs) if payables is not None else None
    revenue_series = series_by_period(ctx.income, "revenue")
    income_series = series_by_period(ctx.income, "net_income")

    def yoy(series: pd.Series) -> Optional[float]:
        if len(series) < 2 or series.iloc[-2] <= 0:
            return None
        return float(series.iloc[-1] / series.iloc[-2] - 1)

    return clean({
        "profitability": {
            "gross_margin": safe_divide(gross, revenue), "operating_margin": safe_divide(ebit, revenue),
            "ebitda_margin": safe_divide(ebitda, revenue), "net_margin": safe_divide(net_income, revenue),
            "roe": safe_divide(net_income, equity) if equity and equity > 0 else None,
            "roa": safe_divide(net_income, assets),
            "roic": safe_divide(ebit * (1 - ctx.tax_rate), invested) if ebit is not None and invested > 0 else None,
        },
        "efficiency": {
            "asset_turnover": safe_divide(revenue, assets), "fixed_asset_turnover": safe_divide(revenue, net_ppe),
            "inventory_turnover": inventory_turnover, "days_inventory_outstanding": dio,
            "days_sales_outstanding": dso, "days_payable_outstanding": dpo,
            "cash_conversion_cycle": dio + dso - dpo if None not in (dio, dso, dpo) else None,
        },
        "liquidity": {
            "current_ratio": safe_divide(current_assets, current_liabilities),
            "quick_ratio": safe_divide((current_assets or 0.0) - (inventory or 0.0), current_liabilities)
            if current_assets is not None else None,
        },
        "leverage": {
            "debt_to_equity": safe_divide(debt, equity) if equity and equity > 0 else None,
            "debt_to_ebitda": safe_divide(debt, ebitda) if ebitda and ebitda > 0 else None,
            "net_debt_to_ebitda": safe_divide(debt - cash, ebitda) if ebitda and ebitda > 0 else None,
            "interest_coverage": safe_divide(ebit, interest),
        },
        "growth": {"revenue_growth": yoy(revenue_series), "revenue_cagr": cagr(revenue_series),
                   "net_income_growth": yoy(income_series)},
        "investment": {
            "capex_to_revenue": safe_divide(latest_metric(ctx, "capex"), revenue),
            "capex_to_depreciation": safe_divide(latest_metric(ctx, "capex"), latest_metric(ctx, "da")),
            "rd_to_revenue": safe_divide(latest("income_statement", "rd_expense"), revenue),
        },
        "valuation": current_multiples(ctx),
    })


def legacy_summary(result: Dict[str, Any], method_name: str, safety_margin: float = 0.25,
                   **extra: Any) -> Dict[str, Any]:
    """Add the keys returned by the pre-contract ``apply_*_valuation`` entry points."""
    result = dict(result)
    methods = result.get("valuation_methods") or {}
    value = result.get("value_per_share")
    result.update({
        "method": method_name,
        "methods": methods,
        "values_by_method": {k: v["value_per_share"] for k, v in methods.items() if v.get("value_per_share")},
        "weights": (result.get("valuation_summary") or {}).get("weights", {}),
        "safety_margin": safety_margin,
        "conservative_value": value * (1 - safety_margin) if value else None,
        **extra,
    })
    return result
