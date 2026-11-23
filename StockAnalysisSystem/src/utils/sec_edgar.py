"""
SEC EDGAR client (free, no API key).

* :func:`get_cik` - ticker -> 10-digit CIK (``company_tickers.json``)
* :func:`get_recent_filings` - latest 10-K / 10-Q / 8-K ... with links (``submissions`` API)
* :func:`get_annual_history` - 10+ years of annual line items from XBRL company facts,
  longer than the 4-5 years Yahoo Finance returns

SEC requires automated clients to send a User-Agent with the requester's own contact
address and rejects anonymous ones (HTTP 403). Set ``SEC_EDGAR_USER_AGENT`` (e.g.
``"Your Name you@yourdomain.com"``) in ``.env``; without it the feature is disabled and
the UI explains what to add. Responses are cached as JSON in ``CACHE_DIR/sec`` for a day;
in offline mode only the cache is read. All public functions return empty results
instead of raising when EDGAR is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

from StockAnalysisSystem.src import config

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
CACHE_TTL_SECONDS = 24 * 3600
TIMEOUT_SECONDS = 20

#: Line item -> US-GAAP concepts tried in order (companies report revenue under different tags)
ANNUAL_CONCEPTS: Dict[str, Sequence[str]] = {
    "Revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense"),
    "Operating income": ("OperatingIncomeLoss",),
    "Net income": ("NetIncomeLoss", "ProfitLoss"),
    "Diluted EPS": ("EarningsPerShareDiluted",),
    "Total assets": ("Assets",),
    "Total liabilities": ("Liabilities",),
    "Shareholders' equity": ("StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "Operating cash flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "Capital expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "Dividends paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}


USER_AGENT_ENV = "SEC_EDGAR_USER_AGENT"


def _user_agent() -> str:
    return os.environ.get(USER_AGENT_ENV, "").strip()


def is_configured() -> bool:
    """True when a contact User-Agent is set (required by SEC for network access)."""
    return bool(_user_agent())


def _cache_path(name: str) -> Path:
    return Path(config.CACHE_DIR) / "sec" / f"{name}.json"


def _fetch_json(url: str) -> Any:
    """HTTP GET -> JSON (network only; patched in tests)."""
    response = requests.get(url, headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"},
                            timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _cached_json(name: str, url: str) -> Optional[Any]:
    """Fresh cache -> network -> stale cache -> None."""
    path = _cache_path(name)
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if config.OFFLINE_MODE or time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS:
                return cached
        except (OSError, ValueError):
            cached = None
    if config.OFFLINE_MODE or not is_configured():
        return cached
    try:
        data = _fetch_json(url)
    except Exception as exc:  # network, HTTP 403/404, bad JSON
        logger.info("SEC EDGAR request failed for %s: %s", url, exc)
        return cached
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning("Could not cache SEC EDGAR response %s", name)
    return data


def get_cik(ticker: str) -> Optional[str]:
    """10-digit, zero-padded CIK of a ticker, or None (unknown ticker or EDGAR unavailable)."""
    data = _cached_json("company_tickers", TICKERS_URL)
    if not isinstance(data, dict):
        return None
    wanted = ticker.strip().upper().replace(".", "-")
    for row in data.values():
        if isinstance(row, dict) and str(row.get("ticker", "")).upper() == wanted:
            return f"{int(row['cik_str']):010d}"
    return None


def get_recent_filings(ticker: str, forms: Optional[Sequence[str]] = ("10-K", "10-Q", "8-K", "20-F", "DEF 14A"),
                       limit: int = 15) -> pd.DataFrame:
    """Most recent filings (newest first): form, filed, report_date, description, url."""
    columns = ["form", "filed", "report_date", "description", "url"]
    cik = get_cik(ticker)
    data = _cached_json(f"submissions_{cik}", SUBMISSIONS_URL.format(cik=cik)) if cik else None
    recent = (data or {}).get("filings", {}).get("recent", {}) if isinstance(data, dict) else {}
    if not recent or not recent.get("form"):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({
        "form": recent.get("form", []),
        "filed": recent.get("filingDate", []),
        "report_date": recent.get("reportDate", []),
        "description": recent.get("primaryDocDescription", []),
        "accession": recent.get("accessionNumber", []),
        "document": recent.get("primaryDocument", []),
    })
    if forms:
        frame = frame[frame["form"].isin(set(forms))]
    cik_int = int(cik)
    frame["url"] = [ARCHIVE_URL.format(cik_int=cik_int, accession=a.replace("-", ""), document=d)
                    for a, d in zip(frame["accession"], frame["document"])]
    frame["description"] = frame["description"].replace("", None).fillna(frame["form"])
    return frame.sort_values("filed", ascending=False).head(limit)[columns].reset_index(drop=True)


def _annual_series(facts: Dict[str, Any], concepts: Sequence[str]) -> pd.Series:
    """Annual values by fiscal year end (from the first concept with data); duplicates keep the latest filing."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        units = gaap.get(concept, {}).get("units", {})
        entries = units.get("USD") or units.get("USD/shares") or []
        rows = [e for e in entries if e.get("form") in ANNUAL_FORMS and e.get("fp") == "FY" and e.get("end")]
        # Flows (income/cash flow) have a start date: keep full-year periods only
        rows = [e for e in rows if "start" not in e
                or 330 <= (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days <= 400]
        if not rows:
            continue
        frame = pd.DataFrame(rows).sort_values("filed")
        series = frame.drop_duplicates("end", keep="last").set_index("end")["val"].astype(float)
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    return pd.Series(dtype=float)


def get_annual_history(ticker: str, years: int = 10) -> pd.DataFrame:
    """Annual line items (rows) by fiscal year end (columns, oldest first) for the last ``years`` years."""
    cik = get_cik(ticker)
    facts = _cached_json(f"facts_{cik}", FACTS_URL.format(cik=cik)) if cik else None
    if not isinstance(facts, dict):
        return pd.DataFrame()
    rows = {item: _annual_series(facts, concepts) for item, concepts in ANNUAL_CONCEPTS.items()}
    rows = {k: v for k, v in rows.items() if not v.empty}
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).T
    table = table[sorted(table.columns)[-years:]]
    if "Operating cash flow" in table.index and "Capital expenditure" in table.index:
        table.loc["Free cash flow"] = table.loc["Operating cash flow"] - table.loc["Capital expenditure"]
    return table


def cagr(series: pd.Series) -> Optional[float]:
    """Compound annual growth rate between the first and last positive values of a yearly series."""
    values = series.dropna()
    values = values[values > 0]
    if len(values) < 2:
        return None
    years = (values.index[-1] - values.index[0]).days / 365.25 if isinstance(values.index, pd.DatetimeIndex) \
        else len(values) - 1
    if years <= 0:
        return None
    return float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)


def history_summary(table: pd.DataFrame) -> List[Dict[str, Any]]:
    """CAGR per line item over the available history (for display)."""
    out = []
    for item in table.index:
        out.append({"item": item, "years": int(table.loc[item].notna().sum()), "cagr": cagr(table.loc[item])})
    return out
