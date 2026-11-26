"""
Insider transactions (SEC Form 4) from Finnhub ``stock/insider-transactions``.

Optional: needs ``FINNHUB_API_KEY`` (free plan). Without the key nothing is requested and the
caller shows how to enable it. Network or provider errors give an empty table, never an exception.
Responses are cached like the news feed (``utils.news_sentiment._cached``; the key never reaches
the cache file name).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from StockAnalysisSystem.src.utils import news_sentiment as ns

logger = logging.getLogger(__name__)

FINNHUB_ENV = ns.FINNHUB_ENV
ENDPOINT = "stock/insider-transactions"
COLUMNS = ["date", "name", "code", "type", "shares", "price", "value", "holding_after"]
#: Form 4 transaction codes (SEC). Only P and S are open-market trades that carry a signal.
CODES = {
    "P": "Buy (open market)", "S": "Sell (open market)", "A": "Grant / award", "M": "Option exercise",
    "F": "Tax withholding", "G": "Gift", "D": "Disposition to issuer", "C": "Conversion", "X": "Option exercise",
    "J": "Other",
}
MARKET_CODES = ("P", "S")


def is_configured() -> bool:
    return bool(ns._key(FINNHUB_ENV))


def parse_transactions(payload: Any) -> pd.DataFrame:
    """Finnhub payload -> one row per transaction, newest first (empty frame for anything unexpected)."""
    rows = []
    items = payload.get("data") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        when = pd.to_datetime(item.get("transactionDate") or item.get("filingDate"), errors="coerce")
        change = pd.to_numeric(item.get("change"), errors="coerce")
        if pd.isna(when) or pd.isna(change) or change == 0:
            continue
        price = pd.to_numeric(item.get("transactionPrice"), errors="coerce")
        code = str(item.get("transactionCode") or "").strip().upper()
        price = float(price) if pd.notna(price) and price > 0 else None
        rows.append({
            "date": when.normalize(), "name": str(item.get("name") or "").strip() or "n/a", "code": code,
            "type": CODES.get(code, code or "n/a"), "shares": float(change), "price": price,
            "value": float(change) * price if price is not None else None,
            "holding_after": pd.to_numeric(item.get("share"), errors="coerce"),
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values("date", ascending=False, kind="stable").reset_index(drop=True) if not frame.empty \
        else frame


def get_insider_transactions(ticker: str, days: int = 365, today: Optional[date] = None) -> pd.DataFrame:
    """Transactions of the last ``days`` days (empty without a key or on any provider failure)."""
    key = ns._key(FINNHUB_ENV)
    if not key or not ticker:
        return pd.DataFrame(columns=COLUMNS)
    end = today or date.today()
    params = {"symbol": ticker.upper(), "from": (end - timedelta(days=days)).isoformat(), "to": end.isoformat(),
              "token": key}
    try:
        payload = ns._cached("finnhub_insider", ns.FINNHUB_URL.format(endpoint=ENDPOINT), params)
    except Exception as exc:  # noqa: BLE001 - optional provider must never break the page
        logger.info("Insider transactions unavailable: %s", type(exc).__name__)
        return pd.DataFrame(columns=COLUMNS)
    return parse_transactions(payload)


def summarize(transactions: pd.DataFrame, days: int = 90, today: Optional[date] = None) -> Dict[str, Any]:
    """Open-market buys vs sells over the last ``days`` days (codes P / S only)."""
    end = pd.Timestamp(today or date.today())
    recent = transactions[(transactions["date"] >= end - pd.Timedelta(days=days))] if not transactions.empty \
        else transactions
    market = recent[recent["code"].isin(MARKET_CODES)] if not recent.empty else recent
    buys = market[market["code"] == "P"] if not market.empty else market
    sells = market[market["code"] == "S"] if not market.empty else market

    def total(frame, column):
        return float(frame[column].abs().sum()) if not frame.empty and frame[column].notna().any() else 0.0

    buy_value, sell_value = total(buys, "value"), total(sells, "value")
    return {
        "days": days, "transactions": int(len(recent)), "buys": int(len(buys)), "sells": int(len(sells)),
        "buy_shares": total(buys, "shares"), "sell_shares": total(sells, "shares"),
        "buy_value": buy_value, "sell_value": sell_value, "net_value": buy_value - sell_value,
        "insiders_buying": int(buys["name"].nunique()) if not buys.empty else 0,
        "insiders_selling": int(sells["name"].nunique()) if not sells.empty else 0,
    }
