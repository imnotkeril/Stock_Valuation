"""
Liquidation value: what shareholders would receive if the company sold its assets and paid all liabilities.

Method (Damodaran, "The Dark Side of Valuation", ch. on distressed firms; McKinsey "Valuation",
asset-based approaches; standard credit-analysis recovery tables)::

    gross recovery   = sum(book value of asset class x recovery rate)
    liquidation cost = cost % x gross recovery          (fees, severance, wind-down; optional)
    equity value     = gross recovery - liquidation cost - total liabilities
    per share        = max(equity value, 0) / shares outstanding

Liabilities are taken at 100 % of book: in a wind-down every creditor is paid before equity.
The value is floored at zero because shareholders have limited liability.

Asset classes come from the latest balance sheet. "Other assets" is the residual
``total assets - every named class`` so the classes always add up to total assets and nothing
is counted twice (deferred tax assets, right-of-use assets, prepaid items land there).

Default recovery rates (% of book). Textbook ranges: cash 100; receivables 70-90; inventory
30-70 (finished goods high, work in progress low); marketable long-term securities 75-95;
PP&E 20-60 (specialised plant low, real estate high); goodwill and intangibles 0 (they do not
survive a break-up); other assets 10-40.

=========================== ========= =======
Asset class                  Orderly   Forced
=========================== ========= =======
Cash & short-term invest.      100       100
Receivables                     85        70
Inventory                       60        35
Long-term investments           90        75
PP&E                            50        25
Goodwill & intangibles           0         0
Other assets                    30        10
Liquidation costs (% gross)      5        10
=========================== ========= =======

*Orderly* liquidation sells over 6-12 months to strategic buyers; *forced* (fire-sale)
liquidation sells within weeks, typically at auction.

Not meaningful for banks and insurers: their assets are loans and securities marked under
regulatory rules and their liabilities are deposits / policy reserves; a resolution regime,
not an asset sale, applies. The result is a floor for distressed or asset-heavy companies and is
never part of the weighted fair value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from StockAnalysisSystem.src.utils.financial_data import get_latest, safe_divide

#: Asset class key -> display label (order = display order)
ASSET_CLASSES: Dict[str, str] = {
    "cash": "Cash & short-term investments",
    "receivables": "Receivables",
    "inventory": "Inventory",
    "investments": "Long-term investments",
    "ppe": "PP&E (net)",
    "intangibles": "Goodwill & intangibles",
    "other": "Other assets",
}

#: Default recovery rates as fractions of book value (see the module docstring for the ranges).
ORDERLY_RATES: Dict[str, float] = {"cash": 1.00, "receivables": 0.85, "inventory": 0.60, "investments": 0.90,
                                   "ppe": 0.50, "intangibles": 0.0, "other": 0.30}
FORCED_RATES: Dict[str, float] = {"cash": 1.00, "receivables": 0.70, "inventory": 0.35, "investments": 0.75,
                                  "ppe": 0.25, "intangibles": 0.0, "other": 0.10}
#: Default liquidation costs as a fraction of gross recovery.
ORDERLY_COST = 0.05
FORCED_COST = 0.10

#: Balance-sheet rows per class, tried in order (total receivables before trade receivables only).
_ROWS: Dict[str, List[str]] = {
    "cash": ["cash_and_short_term_investments", "cash"],
    "receivables": ["Receivables", "Accounts Receivable", "Net Receivables"],
    "inventory": ["inventory"],
    "investments": ["Investments And Advances", "Long Term Equity Investment", "Investmentin Financial Assets",
                    "Long Term Investments"],
    "ppe": ["net_ppe"],
}


@dataclass
class LiquidationResult:
    """One liquidation scenario."""

    gross_recovery: float
    costs: float
    total_liabilities: float
    equity_value: float          # floored at 0
    per_share: Optional[float]
    recovered: Dict[str, float] = field(default_factory=dict)  # class -> recovered amount


def asset_classes(balance_sheet: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    """
    Book value of each asset class from the latest balance sheet; None without total assets.

    Missing rows count as 0. "other" is the residual so the classes sum to total assets
    (floored at 0 if the named rows overlap and exceed the total).
    """
    total_assets = get_latest(balance_sheet, "total_assets")
    if total_assets is None:
        return None
    books: Dict[str, float] = {}
    for key, rows in _ROWS.items():
        value = None
        for row in rows:
            value = get_latest(balance_sheet, row)
            if value is not None:
                break
        books[key] = max(value or 0.0, 0.0)
    intangibles = get_latest(balance_sheet, "goodwill_and_intangibles")
    if intangibles is None:
        intangibles = (get_latest(balance_sheet, "goodwill") or 0.0) + (
            get_latest(balance_sheet, "intangible_assets") or 0.0)
    books["intangibles"] = max(intangibles, 0.0)
    books["other"] = max(total_assets - sum(books.values()), 0.0)
    return {key: books[key] for key in ASSET_CLASSES}


def liquidation_value(books: Dict[str, float], total_liabilities: float, shares: Optional[float],
                      rates: Dict[str, float], cost_pct: float = 0.0) -> LiquidationResult:
    """
    Liquidation equity value for one set of recovery rates.

    Args:
        books: Book value per asset class (:func:`asset_classes`).
        total_liabilities: All liabilities, paid at 100 %.
        shares: Shares outstanding (None/0 -> ``per_share`` None).
        rates: Recovery rate per class as a fraction (missing class -> 0).
        cost_pct: Liquidation costs as a fraction of gross recovery.
    """
    recovered = {key: books.get(key, 0.0) * max(rates.get(key, 0.0), 0.0) for key in books}
    gross = sum(recovered.values())
    costs = gross * max(cost_pct, 0.0)
    equity = max(gross - costs - total_liabilities, 0.0)
    per_share = safe_divide(equity, shares) if shares and shares > 0 else None
    return LiquidationResult(gross_recovery=gross, costs=costs, total_liabilities=total_liabilities,
                             equity_value=equity, per_share=per_share, recovered=recovered)


def liquidation_scenarios(balance_sheet: Optional[pd.DataFrame], shares: Optional[float],
                          orderly_rates: Optional[Dict[str, float]] = None,
                          forced_rates: Optional[Dict[str, float]] = None,
                          orderly_cost: float = ORDERLY_COST,
                          forced_cost: float = FORCED_COST) -> Optional[Dict[str, LiquidationResult]]:
    """Orderly and forced liquidation from a balance sheet; None without total assets or liabilities."""
    books = asset_classes(balance_sheet)
    liabilities = get_latest(balance_sheet, "total_liabilities")
    if books is None or liabilities is None:
        return None
    return {
        "orderly": liquidation_value(books, liabilities, shares, orderly_rates or ORDERLY_RATES, orderly_cost),
        "forced": liquidation_value(books, liabilities, shares, forced_rates or FORCED_RATES, forced_cost),
    }
