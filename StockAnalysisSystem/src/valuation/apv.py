"""
Adjusted present value (APV).

APV = value of the firm as if all-equity financed (unlevered free cash flow discounted
at the unlevered cost of equity) + present value of the interest tax shield.
Equity value = APV - debt + cash. Pure functions, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from StockAnalysisSystem.src.config import RISK_FREE_RATE

EQUITY_RISK_PREMIUM = 0.055


def unlevered_beta(levered_beta: float, debt: float, equity_value: float, tax_rate: float) -> float:
    """Hamada: beta_u = beta_L / (1 + (1 - t) * D / E)."""
    if equity_value <= 0:
        return levered_beta
    return levered_beta / (1 + (1 - tax_rate) * max(debt, 0.0) / equity_value)


def unlevered_cost_of_equity(levered_beta: float, debt: float, equity_value: float, tax_rate: float,
                             risk_free: float = RISK_FREE_RATE, risk_premium: float = EQUITY_RISK_PREMIUM) -> float:
    return risk_free + unlevered_beta(levered_beta, debt, equity_value, tax_rate) * risk_premium


def unlevered_fcf(fcf: float, interest_expense: float, tax_rate: float) -> float:
    """Add back after-tax interest: free cash flow to the firm before financing effects."""
    return fcf + abs(interest_expense or 0.0) * (1 - tax_rate)


def apv_valuation(base_ufcf: float, growth: float, years: int, terminal_growth: float, unlevered_cost: float,
                  debt: float, cost_of_debt: float, tax_rate: float, cash: float = 0.0,
                  shares: Optional[float] = None) -> Dict[str, Any]:
    """
    APV with constant-growth forecast and debt growing with the business after the forecast.

    Tax shield each year = tax_rate * cost_of_debt * debt, discounted at the cost of debt
    (debt-like risk); after the horizon it grows at ``terminal_growth``.
    """
    if unlevered_cost <= terminal_growth:
        raise ValueError("Unlevered cost of equity must exceed terminal growth.")
    if years < 1:
        raise ValueError("Forecast years must be at least 1.")
    debt = max(debt or 0.0, 0.0)
    rows: List[Dict[str, float]] = []
    pv_fcf = 0.0
    pv_shield = 0.0
    fcf = base_ufcf
    shield = tax_rate * cost_of_debt * debt
    for year in range(1, years + 1):
        fcf *= 1 + growth
        fcf_discount = (1 + unlevered_cost) ** year
        shield_discount = (1 + cost_of_debt) ** year if cost_of_debt > -1 else 1.0
        pv_fcf += fcf / fcf_discount
        pv_shield += shield / shield_discount
        rows.append({"year": year, "ufcf": fcf, "pv_ufcf": fcf / fcf_discount,
                     "tax_shield": shield, "pv_tax_shield": shield / shield_discount})
    terminal_value = fcf * (1 + terminal_growth) / (unlevered_cost - terminal_growth)
    pv_terminal = terminal_value / (1 + unlevered_cost) ** years
    if shield > 0 and cost_of_debt > terminal_growth:
        terminal_shield = shield * (1 + terminal_growth) / (cost_of_debt - terminal_growth)
        pv_terminal_shield = terminal_shield / (1 + cost_of_debt) ** years
    else:
        pv_terminal_shield = 0.0
    unlevered_value = pv_fcf + pv_terminal
    tax_shield_value = pv_shield + pv_terminal_shield
    enterprise_value = unlevered_value + tax_shield_value
    equity_value = enterprise_value - debt + (cash or 0.0)
    return {
        "method": "Adjusted present value (APV)",
        "unlevered_value": unlevered_value,
        "pv_forecast_fcf": pv_fcf,
        "pv_terminal_value": pv_terminal,
        "tax_shield_value": tax_shield_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": equity_value / shares if shares else None,
        "terminal_share": pv_terminal / unlevered_value if unlevered_value else None,
        "projections": rows,
    }
