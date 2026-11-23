"""
Company news and sentiment from Finnhub and Alpha Vantage (both optional, free keys).

* Finnhub (``FINNHUB_API_KEY``): company news, basic financial metrics, peers. Headlines
  are scored with a small finance word list (:func:`lexicon_score`).
* Alpha Vantage (``ALPHA_VANTAGE_API_KEY``): ``NEWS_SENTIMENT`` with a per-ticker score.

Responses are cached as JSON in ``CACHE_DIR/news`` (6 hours); offline mode reads the cache
only. Missing keys or provider errors give empty results, never exceptions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from StockAnalysisSystem.src import config

logger = logging.getLogger(__name__)

FINNHUB_ENV = "FINNHUB_API_KEY"
ALPHA_VANTAGE_ENV = "ALPHA_VANTAGE_API_KEY"
FINNHUB_URL = "https://finnhub.io/api/v1/{endpoint}"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CACHE_TTL_SECONDS = 6 * 3600
TIMEOUT_SECONDS = 20
NEWS_COLUMNS = ["published", "provider", "source", "headline", "url", "sentiment", "relevance"]

#: Compact finance sentiment lexicon (inspired by Loughran-McDonald categories)
POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "record", "growth", "upgrade",
    "upgrades", "upgraded", "outperform", "strong", "stronger", "gain", "gains", "profit", "profits", "raise",
    "raises", "raised", "bullish", "boost", "boosts", "expands", "expansion", "wins", "win", "approval",
    "approved", "breakthrough", "jumps", "jump", "rises", "rise", "higher", "buyback", "dividend", "exceeds",
    "tops", "optimistic", "rebound", "recovers", "accelerates", "partnership",
}
NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunges", "slump", "slumps", "fall", "falls", "drop", "drops",
    "downgrade", "downgrades", "downgraded", "underperform", "weak", "weaker", "loss", "losses", "cut", "cuts",
    "bearish", "lawsuit", "sues", "probe", "investigation", "recall", "layoffs", "layoff", "warning", "warns",
    "decline", "declines", "lower", "fraud", "default", "bankruptcy", "slowdown", "delay", "delays", "fine",
    "fined", "penalty", "tumbles", "tumble", "sinks", "sink", "concerns", "risk", "risks", "halt", "halts",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed"}
_WORD = re.compile(r"[a-z']+")


def _key(env: str) -> str:
    return os.environ.get(env, "").strip()


def available_providers() -> List[str]:
    return [name for name, env in (("Finnhub", FINNHUB_ENV), ("Alpha Vantage", ALPHA_VANTAGE_ENV)) if _key(env)]


def lexicon_score(text: str) -> float:
    """Sentiment in [-1, 1]: (positive - negative) / matched words; a negator flips the next word."""
    words = _WORD.findall((text or "").lower())
    score, hits, flip = 0, 0, False
    for word in words:
        if word in _NEGATORS:
            flip = True
            continue
        sign = 1 if word in POSITIVE else -1 if word in NEGATIVE else 0
        if sign:
            score += -sign if flip else sign
            hits += 1
        flip = False
    return score / hits if hits else 0.0


def _get_json(url: str, params: Dict[str, Any]) -> Any:
    """HTTP GET -> JSON (network only; patched in tests)."""
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _cached(name: str, url: str, params: Dict[str, Any]) -> Optional[Any]:
    """Fresh cache -> network -> stale cache. The cache file name never contains the API key."""
    public = {k: v for k, v in params.items() if k not in {"token", "apikey"}}
    digest = hashlib.sha1(json.dumps([url, public], sort_keys=True).encode()).hexdigest()[:16]
    path = Path(config.CACHE_DIR) / "news" / f"{name}_{digest}.json"
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if config.OFFLINE_MODE or time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS:
                return cached
        except (OSError, ValueError):
            cached = None
    if config.OFFLINE_MODE:
        return cached
    try:
        data = _get_json(url, params)
    except Exception as exc:  # network, HTTP errors, bad JSON
        logger.info("%s request failed: %s", name, type(exc).__name__)
        return cached
    if isinstance(data, dict) and ({"Note", "Information", "Error Message", "error"} & set(data)):
        logger.info("%s returned a limit or error message", name)  # rate limit: keep the old copy
        return cached
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------
def finnhub_news(ticker: str, days: int = 30, today: Optional[date] = None) -> pd.DataFrame:
    key = _key(FINNHUB_ENV)
    if not key:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    end = today or date.today()
    params = {"symbol": ticker.upper(), "from": (end - timedelta(days=days)).isoformat(), "to": end.isoformat(),
              "token": key}
    data = _cached("finnhub_news", FINNHUB_URL.format(endpoint="company-news"), params)
    rows = []
    for item in data if isinstance(data, list) else []:
        headline = str(item.get("headline") or "").strip()
        if not headline or not item.get("datetime"):
            continue
        rows.append({"published": datetime.fromtimestamp(int(item["datetime"]), tz=timezone.utc).replace(tzinfo=None),
                     "provider": "Finnhub", "source": item.get("source") or "", "headline": headline,
                     "url": item.get("url") or "",
                     "sentiment": lexicon_score(f"{headline}. {item.get('summary') or ''}"), "relevance": None})
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def finnhub_metrics(ticker: str) -> Dict[str, Any]:
    """Finnhub ``stock/metric`` (all): 52-week range, beta, margins, growth ... ({} without key)."""
    key = _key(FINNHUB_ENV)
    if not key:
        return {}
    data = _cached("finnhub_metric", FINNHUB_URL.format(endpoint="stock/metric"),
                   {"symbol": ticker.upper(), "metric": "all", "token": key})
    return dict(data.get("metric") or {}) if isinstance(data, dict) else {}


def finnhub_peers(ticker: str) -> List[str]:
    key = _key(FINNHUB_ENV)
    if not key:
        return []
    data = _cached("finnhub_peers", FINNHUB_URL.format(endpoint="stock/peers"), {"symbol": ticker.upper(), "token": key})
    return [str(p) for p in data if str(p).upper() != ticker.upper()] if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------
def alpha_vantage_news(ticker: str, limit: int = 50) -> pd.DataFrame:
    key = _key(ALPHA_VANTAGE_ENV)
    if not key:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    data = _cached("av_news", ALPHA_VANTAGE_URL,
                   {"function": "NEWS_SENTIMENT", "tickers": ticker.upper(), "limit": limit, "apikey": key})
    rows = []
    for item in (data or {}).get("feed", []) if isinstance(data, dict) else []:
        ticker_score = next((t for t in item.get("ticker_sentiment", [])
                             if str(t.get("ticker", "")).upper() == ticker.upper()), None)
        score = ticker_score.get("ticker_sentiment_score") if ticker_score else item.get("overall_sentiment_score")
        try:
            published = datetime.strptime(str(item.get("time_published", ""))[:15], "%Y%m%dT%H%M%S")
            score = float(score)
        except (TypeError, ValueError):
            continue
        rows.append({"published": published, "provider": "Alpha Vantage", "source": item.get("source") or "",
                     "headline": str(item.get("title") or "").strip(), "url": item.get("url") or "",
                     "sentiment": max(-1.0, min(1.0, score)),
                     "relevance": float(ticker_score["relevance_score"]) if ticker_score else None})
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def get_news(ticker: str, days: int = 30) -> pd.DataFrame:
    """News from all configured providers, newest first, duplicates (same headline) removed."""
    frames = [f for f in (finnhub_news(ticker, days), alpha_vantage_news(ticker)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    news = pd.concat(frames, ignore_index=True)
    news["key"] = news["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    news = news.sort_values("published", ascending=False).drop_duplicates("key").drop(columns="key")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return news[news["published"] >= cutoff].reset_index(drop=True)


def sentiment_label(score: Optional[float]) -> str:
    if score is None or score != score:
        return "n/a"
    if score >= 0.15:
        return "Bullish"
    if score <= -0.15:
        return "Bearish"
    return "Neutral"


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Mean sentiment and article count per day."""
    if news.empty:
        return pd.DataFrame(columns=["sentiment", "articles"])
    by_day = news.assign(day=pd.to_datetime(news["published"]).dt.normalize()).groupby("day")
    return pd.DataFrame({"sentiment": by_day["sentiment"].mean(), "articles": by_day.size()})
