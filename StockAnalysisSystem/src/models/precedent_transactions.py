"""
Precedent transactions: value a company at the multiples paid in comparable M&A deals.

Deals are maintained by the user (no bundled figures: deal multiples must come from the
user's own sources - press releases, merger proxies, fairness opinions). They are stored as
CSV in ``DATA_DIR/user/precedent_deals.csv``. With ``FINANCIAL_MODELING_PREP_API_KEY`` a feed
of recent M&A announcements is available to help find deals (it has no multiples).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from StockAnalysisSystem.src.config import DATA_DIR

logger = logging.getLogger(__name__)

DEALS_FILE = DATA_DIR / "user" / "precedent_deals.csv"
COLUMNS = ["date", "acquirer", "target", "sector", "ev_billions", "ev_revenue", "ev_ebitda", "source"]
FMP_ENV = "FINANCIAL_MODELING_PREP_API_KEY"
FMP_MA_URL = "https://financialmodelingprep.com/stable/mergers-acquisitions-latest"
FMP_FREE_LIMIT = 5


def empty_deals() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c in {"date", "acquirer", "target", "sector", "source"}
                                      else "float") for c in COLUMNS})


def clean_deals(deals: pd.DataFrame) -> pd.DataFrame:
    """Keep known columns, coerce numbers, drop rows without a target or any multiple."""
    frame = deals.copy() if isinstance(deals, pd.DataFrame) else empty_deals()
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[COLUMNS]
    for column in ("ev_billions", "ev_revenue", "ev_ebitda"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] <= 0, column] = float("nan")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("acquirer", "target", "sector", "source"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    keep = (frame["target"] != "") & (frame["ev_revenue"].notna() | frame["ev_ebitda"].notna())
    return frame[keep].reset_index(drop=True)


def load_deals(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DEALS_FILE
    try:
        return clean_deals(pd.read_csv(path))
    except (OSError, ValueError, pd.errors.ParserError):
        return empty_deals()


def save_deals(deals: pd.DataFrame, path: Optional[Path] = None) -> int:
    path = path or DEALS_FILE
    frame = clean_deals(deals)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(frame)


def deal_multiples(deals: pd.DataFrame, sector: Optional[str] = None, since_year: Optional[int] = None) -> Dict[str, Any]:
    """Median / quartile EV/Revenue and EV/EBITDA of the matching deals."""
    frame = clean_deals(deals)
    if sector:
        frame = frame[frame["sector"].str.lower() == sector.lower()]
    if since_year:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce").dt.year >= since_year]
    out: Dict[str, Any] = {"deals": len(frame)}
    for column in ("ev_revenue", "ev_ebitda"):
        values = frame[column].dropna()
        out[column] = {"median": float(values.median()) if len(values) else None,
                       "q1": float(values.quantile(0.25)) if len(values) else None,
                       "q3": float(values.quantile(0.75)) if len(values) else None, "count": int(len(values))}
    return out


def implied_values(multiples: Dict[str, Any], revenue: Optional[float], ebitda: Optional[float],
                   net_debt: float, shares: Optional[float]) -> List[Dict[str, Any]]:
    """Implied equity value per share at the median (and quartile range) deal multiples."""
    rows = []
    for column, metric, label in (("ev_revenue", revenue, "EV/Revenue"), ("ev_ebitda", ebitda, "EV/EBITDA")):
        stats = multiples.get(column) or {}
        if not stats.get("median") or not metric or metric <= 0 or not shares:
            continue

        def per_share(multiple: Optional[float]) -> Optional[float]:
            return max(metric * multiple - (net_debt or 0.0), 0.0) / shares if multiple else None

        rows.append({"method": f"Deal {label}", "multiple": stats["median"], "deals": stats["count"],
                     "value_per_share": per_share(stats["median"]), "low": per_share(stats["q1"]),
                     "high": per_share(stats["q3"])})
    return rows


def fmp_recent_deals(limit: int = 50) -> pd.DataFrame:
    """Latest M&A announcements from Financial Modeling Prep (empty without key or on error)."""
    key = os.environ.get(FMP_ENV, "").strip()
    columns = ["date", "acquirer", "target", "link"]
    if not key:
        return pd.DataFrame(columns=columns)
    try:
        response = requests.get(FMP_MA_URL, params={"page": 0, "limit": limit, "apikey": key}, timeout=20)
        if response.status_code == 402 and limit > FMP_FREE_LIMIT:  # free plan caps limit at 5
            response = requests.get(FMP_MA_URL, params={"page": 0, "limit": FMP_FREE_LIMIT, "apikey": key},
                                    timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # network / plan limits
        logger.info("FMP M&A feed failed: %s", type(exc).__name__)
        return pd.DataFrame(columns=columns)
    rows = [{"date": d.get("transactionDate") or d.get("acceptedDate", "")[:10],
             "acquirer": d.get("companyName") or d.get("symbol"), "target": d.get("targetedCompanyName"),
             "link": d.get("link")} for d in data if isinstance(d, dict)] if isinstance(data, list) else []
    return pd.DataFrame(rows, columns=columns)
