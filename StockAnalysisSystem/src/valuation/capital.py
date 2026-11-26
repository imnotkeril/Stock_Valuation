"""
One cost of capital for every valuation model (ADR-007).

    rf    = utils.rates.risk_free_rate()  (user override -> FRED DGS10 -> fallback constant)
    beta  = Blume-adjusted: 0.67 * raw + 0.33, clamped to BETA_BOUNDS; 1.0 when unknown
    Ke    = rf + beta * EQUITY_RISK_PREMIUM, clamped to COST_OF_EQUITY_BOUNDS
    Kd    = interest expense / total debt, clamped to [rf + 0.5 %, rf + 8 %]; rf + 2 % when not derivable
    WACC  = E/(D+E) * Ke + D/(D+E) * Kd * (1 - t), clamped to WACC_BOUNDS

Equity is the market value (market cap, else shares x price), book equity as a fallback; with neither,
the sector default discount rate is used. The generic DCF, every sector model, APV and the stress tests
read their rates from here, so the same company always gets the same discount rate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from StockAnalysisSystem.src.utils.financial_data import (
    debt_excluding_leases,
    get_latest,
    get_series,
    safe_divide,
    to_float,
)
from StockAnalysisSystem.src.utils.rates import RiskFreeRate, risk_free_rate

#: US equity risk premium: Damodaran's implied ERP for the US, July 2026 update (4.42 %), rounded. ADR-007.
EQUITY_RISK_PREMIUM = 0.044
BETA_BOUNDS = (0.3, 3.0)
COST_OF_EQUITY_BOUNDS = (0.045, 0.25)
#: Pre-tax cost of debt as a spread over the risk-free rate.
DEBT_SPREAD_BOUNDS = (0.005, 0.08)
DEFAULT_DEBT_SPREAD = 0.02
WACC_BOUNDS = (0.06, 0.20)
DEFAULT_TAX_RATE = 0.21
MAX_TAX_RATE = 0.35
#: Long-run nominal growth used for the stress-test multiple sensitivity.
LONG_RUN_GROWTH = 0.025


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def adjusted_beta(raw_beta: Optional[float]) -> float:
    """Blume adjustment toward 1.0 (raw betas mean-revert); 1.0 when the raw beta is unknown."""
    if raw_beta is None:
        return 1.0
    return _clip(0.67 * raw_beta + 0.33, *BETA_BOUNDS)


def cost_of_equity(beta: float, rf: float, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """CAPM cost of equity for an already adjusted beta."""
    return _clip(rf + beta * erp, *COST_OF_EQUITY_BOUNDS)


def market_discount_rate(rf: Optional[RiskFreeRate] = None) -> float:
    """Discount rate of an average (beta 1, unlevered) company: rf + ERP."""
    rf = rf or risk_free_rate()
    return cost_of_equity(1.0, rf.rate)


def total_debt(balance: Optional[pd.DataFrame]) -> float:
    """
    Borrowings used by every valuation bridge and the WACC: debt without lease liabilities
    (utils.financial_data.debt_excluding_leases); 0 when unavailable. Cash flows and EBITDA are after
    rent, so leases stay an operating cost and are not counted as debt as well.
    """
    return float(debt_excluding_leases(balance) or 0.0)


def effective_tax_rate(income: Optional[pd.DataFrame]) -> float:
    """Effective tax rate from the statements, bounded to [0, 35 %]; 21 % when not meaningful."""
    rate = get_latest(income, "tax_rate")
    if rate is None:
        tax, pretax = get_series(income, "tax_provision"), get_series(income, "pretax_income")
        common = tax.index.intersection(pretax.index)
        total_pretax = float(pretax[common].sum()) if len(common) else 0.0
        rate = safe_divide(float(tax[common].sum()), total_pretax) if total_pretax > 0 else None
    if rate is None or not 0 <= rate <= 0.5:
        return DEFAULT_TAX_RATE
    return float(min(rate, MAX_TAX_RATE))


def cap_terminal_growth(terminal_growth: float, rf: Optional[float]) -> Dict[str, Any]:
    """
    Stable growth cannot exceed the nominal growth of the economy, proxied by the risk-free rate
    (Damodaran, "Growth Rates and Terminal Value"). Returns the capped rate and a note when capped.
    """
    growth = float(terminal_growth)
    if rf is not None and growth > rf:
        return {"terminal_growth": float(rf),
                "note": f"Terminal growth {growth:.2%} capped at the risk-free rate {rf:.2%}: stable growth "
                        "cannot exceed nominal economic growth."}
    return {"terminal_growth": growth, "note": None}


def _positive(value: Any) -> Optional[float]:
    number = to_float(value)
    return number if number is not None and number > 0 else None


def _equity_value(financial_data: Dict[str, Any]) -> tuple[Optional[float], str]:
    market = financial_data.get("market_data") or {}
    market_cap = _positive(market.get("market_cap"))
    if market_cap:
        return market_cap, "market"
    shares, price = _positive(market.get("shares_outstanding")), _positive(market.get("share_price"))
    if shares and price:
        return shares * price, "market"
    book = _positive(get_latest(financial_data.get("balance_sheet"), "total_equity"))
    return (book, "book") if book else (None, "sector default")


def cost_of_capital(*, raw_beta: Optional[float], debt: float, interest_expense: float, tax_rate: float,
                    equity_value: Optional[float], fallback_wacc: float, equity_basis: str = "market",
                    rf: Optional[RiskFreeRate] = None) -> Dict[str, Any]:
    """CAPM cost of equity, cost of debt and market-value WACC with every input that produced them."""
    rf = rf or risk_free_rate()
    beta = adjusted_beta(raw_beta)
    ke = cost_of_equity(beta, rf.rate)
    debt = max(float(debt or 0.0), 0.0)
    implied_kd = safe_divide(abs(interest_expense or 0.0), debt)
    if implied_kd:
        kd = _clip(implied_kd, rf.rate + DEBT_SPREAD_BOUNDS[0], rf.rate + DEBT_SPREAD_BOUNDS[1])
    else:
        kd = rf.rate + DEFAULT_DEBT_SPREAD
    if equity_value and equity_value > 0:
        equity_weight = equity_value / (equity_value + debt)
        raw_wacc = equity_weight * ke + (1 - equity_weight) * kd * (1 - tax_rate)
    else:
        equity_weight, raw_wacc, equity_basis = None, fallback_wacc, "sector default"
    return {
        "wacc": _clip(raw_wacc, *WACC_BOUNDS),
        "unbounded_wacc": float(raw_wacc),
        "cost_of_equity": ke,
        "pre_tax_cost_of_debt": float(kd),
        "after_tax_cost_of_debt": float(kd * (1 - tax_rate)),
        "tax_rate": float(tax_rate),
        "raw_beta": raw_beta,
        "beta": beta,
        "adjusted_beta": beta,
        "beta_assumed": raw_beta is None,
        "risk_free_rate": rf.rate,
        "risk_free_rate_source": rf.source,
        "risk_free_rate_as_of": rf.as_of,
        "risk_free_rate_label": rf.label,
        "equity_risk_premium": EQUITY_RISK_PREMIUM,
        "equity_weight": equity_weight,
        "debt_weight": None if equity_weight is None else 1 - equity_weight,
        "equity_basis": equity_basis,
    }


def cost_of_capital_for(financial_data: Dict[str, Any], fallback_wacc: float,
                        rf: Optional[RiskFreeRate] = None, loader=None) -> Dict[str, Any]:
    """Cost of capital from a canonical financial-data bundle (see utils.financial_data)."""
    market = financial_data.get("market_data") or {}
    raw_beta = to_float(market.get("beta"))
    if raw_beta is None:
        raw_beta = to_float((financial_data.get("company_info") or {}).get("beta"))
    income = financial_data.get("income_statement")
    equity, basis = _equity_value(financial_data)
    return cost_of_capital(
        raw_beta=raw_beta,
        debt=total_debt(financial_data.get("balance_sheet")),
        interest_expense=abs(get_latest(income, "interest_expense") or 0.0),
        tax_rate=effective_tax_rate(income),
        equity_value=equity, equity_basis=basis, fallback_wacc=fallback_wacc,
        rf=rf or risk_free_rate(loader),
    )


def rate_summary(components: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level result assumptions: the risk-free rate actually used and where it came from."""
    return {key: components.get(key) for key in (
        "risk_free_rate", "risk_free_rate_source", "risk_free_rate_as_of", "equity_risk_premium")}
