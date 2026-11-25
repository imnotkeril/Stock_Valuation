"""
Stress testing and scenario analysis (pure functions).

* :func:`scenario_valuations` - macro shocks (rates, recession, margin squeeze, de-rating,
  stagflation) applied to an EV/EBITDA valuation with a net-debt bridge, plus the effect on
  interest coverage.
* :func:`historical_scenarios` - what the 2008, 2020 and 2022 market drawdowns imply for the
  stock (beta x market drawdown) and, when the price history covers the window, the stock's
  actual peak-to-trough move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from StockAnalysisSystem.src.valuation.capital import LONG_RUN_GROWTH, market_discount_rate


@dataclass(frozen=True)
class Shock:
    name: str
    description: str
    revenue: float = 0.0        # relative change, -0.15 = -15 %
    margin_pts: float = 0.0     # EBITDA margin change in percentage points (-0.03 = -3 pts)
    multiple: float = 0.0       # relative change of the EV/EBITDA multiple
    rate_bp: float = 0.0        # change of interest rates in basis points
    floating_debt: float = 0.5  # share of debt repricing with rates


SCENARIOS: List[Shock] = [
    Shock("Rates +200 bp", "Higher discount rates compress multiples; floating debt reprices.", rate_bp=200),
    Shock("Recession", "Revenue -15 %, EBITDA margin -3 pts, multiples -20 %.", revenue=-0.15, margin_pts=-0.03,
          multiple=-0.20),
    Shock("Margin squeeze", "Input-cost inflation: EBITDA margin -5 pts.", margin_pts=-0.05),
    Shock("Multiple de-rating", "Sentiment shift: EV/EBITDA -30 %.", multiple=-0.30),
    Shock("Stagflation", "Rates +300 bp, revenue -5 %, EBITDA margin -2 pts.", revenue=-0.05, margin_pts=-0.02,
          rate_bp=300),
]

#: S&P 500 peak-to-trough drawdowns and windows
HISTORICAL_CRISES = [
    {"name": "Global financial crisis", "start": "2007-10-09", "end": "2009-03-09", "market_drawdown": -0.568},
    {"name": "COVID-19 crash", "start": "2020-02-19", "end": "2020-03-23", "market_drawdown": -0.339},
    {"name": "2022 rate shock", "start": "2022-01-03", "end": "2022-10-12", "market_drawdown": -0.254},
]


def rate_multiple_factor(rate_bp: float, base_discount: Optional[float] = None,
                         growth: float = LONG_RUN_GROWTH) -> float:
    """
    Multiple scaling when discount rates move by ``rate_bp``: (r - g) / (r + d - g), multiple ~ 1 / (r - g).

    ``base_discount`` defaults to the market discount rate rf + ERP (valuation/capital.py).
    """
    if base_discount is None:
        base_discount = market_discount_rate()
    shifted = base_discount + rate_bp / 10_000 - growth
    return (base_discount - growth) / shifted if shifted > 0 else 0.0


def scenario_valuations(revenue: float, ebitda: float, ev_ebitda: float, net_debt: float, shares: float,
                        debt: float = 0.0, interest_expense: float = 0.0,
                        scenarios: Optional[List[Shock]] = None,
                        base_discount: Optional[float] = None) -> pd.DataFrame:
    """
    Equity value per share under each shock (first row = base case).

    EV = revenue x EBITDA margin x EV/EBITDA; equity = EV - net debt (floored at 0).
    Interest cost rises by ``rate_bp`` on the floating share of debt.
    """
    if not (revenue and revenue > 0 and ebitda and ebitda > 0 and ev_ebitda and ev_ebitda > 0 and shares):
        raise ValueError("Positive revenue, EBITDA, EV/EBITDA and shares are required.")
    margin = ebitda / revenue
    interest = abs(interest_expense or 0.0)
    base_discount = market_discount_rate() if base_discount is None else base_discount

    def row(name: str, description: str, rev: float, mgn: float, mult: float, int_cost: float) -> Dict[str, Any]:
        stressed_ebitda = rev * max(mgn, 0.0)
        ev = stressed_ebitda * mult
        equity = max(ev - net_debt, 0.0)
        return {"scenario": name, "description": description, "revenue": rev, "ebitda": stressed_ebitda,
                "ebitda_margin": mgn, "ev_ebitda": mult, "enterprise_value": ev, "equity_value": equity,
                "value_per_share": equity / shares,
                "interest_coverage": stressed_ebitda / int_cost if int_cost > 0 else None}

    rows = [row("Base case", "Latest EBITDA at the current multiple.", revenue, margin, ev_ebitda, interest)]
    for shock in scenarios or SCENARIOS:
        mult = ev_ebitda * (1 + shock.multiple) * rate_multiple_factor(shock.rate_bp, base_discount)
        int_cost = interest + max(debt or 0.0, 0.0) * shock.floating_debt * shock.rate_bp / 10_000
        rows.append(row(shock.name, shock.description, revenue * (1 + shock.revenue), margin + shock.margin_pts,
                        mult, int_cost))
    frame = pd.DataFrame(rows)
    base = frame.loc[0, "value_per_share"]
    frame["change"] = frame["value_per_share"] / base - 1 if base else None
    return frame


_EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _close(prices: Optional[pd.DataFrame]) -> pd.Series:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return _EMPTY
    column = next((c for c in ("Adj Close", "Close", "close", "adj_close") if c in prices.columns), None)
    if column is None:
        return _EMPTY
    series = pd.to_numeric(prices[column], errors="coerce").dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()]
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series.sort_index()


def historical_scenarios(beta: Optional[float], prices: Optional[pd.DataFrame] = None,
                         current_price: Optional[float] = None) -> pd.DataFrame:
    """Beta-implied and (when covered by ``prices``) actual drawdowns for each historical crisis."""
    close = _close(prices)
    rows = []
    for crisis in HISTORICAL_CRISES:
        implied = max(beta * crisis["market_drawdown"], -0.95) if beta is not None else None
        window = close[(close.index >= pd.Timestamp(crisis["start"])) & (close.index <= pd.Timestamp(crisis["end"]))]
        actual = float((window / window.cummax() - 1).min()) if len(window) > 5 else None
        rows.append({"crisis": crisis["name"], "window": f"{crisis['start']} - {crisis['end']}",
                     "market_drawdown": crisis["market_drawdown"], "beta_implied": implied,
                     "actual_drawdown": actual,
                     "implied_price": current_price * (1 + implied) if current_price and implied is not None else None})
    return pd.DataFrame(rows)
