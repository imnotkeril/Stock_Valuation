"""
Balance-sheet liquidity grouping A1-A4 / P1-P4 (Russian textbook method).

Sources: Sheremet A.D., Negashev E.V. "Metodika finansovogo analiza"; Savitskaya G.V.
"Analiz khozyaistvennoi deyatelnosti". Assets are grouped by how fast they turn into cash,
liabilities by how soon they fall due. Mapping of the yfinance (US GAAP) balance sheet, latest period:

====  =========================  ============================================================
A1    most liquid assets         cash and short-term investments (RAS 1250 + 1240)
A2    quickly realisable         receivables (RAS 1230; total receivables, incl. other receivables)
A3    slowly realisable          the rest of current assets: inventory + prepaid / other current
                                 assets (RAS 1210 + 1220 + 1260, Savitskaya / Sheremet edition)
A4    hard to realise            non-current assets = total assets - current assets (RAS 1100)
P1    most urgent liabilities    payables (RAS 1520): "Payables And Accrued Expenses", else
                                 "Payables", else "Accounts Payable"
P2    short-term liabilities     the rest of current liabilities: short-term debt / current portion,
                                 deferred revenue, other current liabilities (RAS 1510 + 1550)
P3    long-term liabilities      total liabilities - current liabilities (RAS 1400)
P4    permanent liabilities      total equity incl. minority interest (RAS 1300)
====  =========================  ============================================================

A3 and P2 are residuals, so A1+A2+A3+A4 = total assets and P1+P2+P3+P4 = total liabilities +
equity (= total assets when the statement balances). Long-term investments stay in A4 (they
are non-current assets). US deferred revenue stays in P2 (an obligation to deliver within a year;
Sheremet's move of RAS 1530 deferred income into P4 concerns grants, not customer prepayments).

The balance is **absolutely liquid** when A1 >= P1, A2 >= P2, A3 >= P3 and A4 <= P4 all hold.
Current liquidity = (A1 + A2) - (P1 + P2); prospective liquidity = A3 - P3.

Needs a current / non-current split: banks and insurers do not report one, and their liquidity
is regulated (LCR / NSFR), so the method does not apply to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from StockAnalysisSystem.src.utils.financial_data import get_latest

ASSET_GROUPS = {"A1": "Most liquid (cash + short-term investments)", "A2": "Quickly realisable (receivables)",
                "A3": "Slowly realisable (inventory, other current)", "A4": "Hard to realise (non-current assets)"}
LIABILITY_GROUPS = {"P1": "Most urgent (payables)", "P2": "Short-term (debt, other current)",
                    "P3": "Long-term liabilities", "P4": "Permanent (equity)"}
#: (asset, liability, operator) of the four conditions
CONDITIONS = (("A1", "P1", ">="), ("A2", "P2", ">="), ("A3", "P3", ">="), ("A4", "P4", "<="))


@dataclass
class BalanceLiquidity:
    groups: Dict[str, float]
    conditions: Dict[str, bool]            # "A1 >= P1" -> met
    absolutely_liquid: bool
    current_liquidity: float               # (A1 + A2) - (P1 + P2)
    prospective_liquidity: float           # A3 - P3
    assumed_zero: List[str] = field(default_factory=list)  # rows missing and taken as 0


def _first(balance: pd.DataFrame, rows: List[str]) -> Optional[float]:
    for row in rows:
        value = get_latest(balance, row)
        if value is not None:
            return value
    return None


def balance_liquidity(balance: Optional[pd.DataFrame]) -> Optional[BalanceLiquidity]:
    """A1-A4 / P1-P4 grouping of the latest balance sheet; None without the totals it needs."""
    totals = {key: get_latest(balance, key) for key in
              ("total_assets", "current_assets", "current_liabilities", "total_liabilities")}
    equity = _first(balance, ["Total Equity Gross Minority Interest", "total_equity"])
    if any(v is None for v in totals.values()) or equity is None:
        return None
    assumed: List[str] = []

    def part(label: str, rows: List[str]) -> float:
        value = _first(balance, rows)
        if value is None:
            assumed.append(label)
            return 0.0
        return value

    cash = _first(balance, ["cash_and_short_term_investments"])
    if cash is None:
        cash = part("cash", ["cash"]) + (get_latest(balance, "short_term_investments") or 0.0)
    a2 = part("receivables", ["Receivables", "Accounts Receivable", "Net Receivables"])
    p1 = part("payables", ["Payables And Accrued Expenses", "Payables", "Accounts Payable"])
    ca, cl = totals["current_assets"], totals["current_liabilities"]
    groups = {
        "A1": cash, "A2": a2, "A3": ca - cash - a2, "A4": totals["total_assets"] - ca,
        "P1": p1, "P2": cl - p1, "P3": totals["total_liabilities"] - cl, "P4": equity,
    }
    conditions = {f"{a} {op} {p}": (groups[a] >= groups[p] if op == ">=" else groups[a] <= groups[p])
                  for a, p, op in CONDITIONS}
    return BalanceLiquidity(
        groups=groups, conditions=conditions, absolutely_liquid=all(conditions.values()),
        current_liquidity=(groups["A1"] + groups["A2"]) - (groups["P1"] + groups["P2"]),
        prospective_liquidity=groups["A3"] - groups["P3"], assumed_zero=assumed)


def liquidity_table(result: BalanceLiquidity) -> pd.DataFrame:
    """One row per pair: asset group, liability group, surplus (+) / deficit (-), condition met."""
    rows = []
    for a, p, op in CONDITIONS:
        rows.append({"Assets": f"{a} {ASSET_GROUPS[a]}", "Asset value": result.groups[a],
                     "Liabilities": f"{p} {LIABILITY_GROUPS[p]}", "Liability value": result.groups[p],
                     "Surplus / deficit": result.groups[a] - result.groups[p],
                     "Condition": f"{a} {op} {p}", "Met": result.conditions[f"{a} {op} {p}"]})
    return pd.DataFrame(rows)
