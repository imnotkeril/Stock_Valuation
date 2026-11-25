"""
Risk-free rate used by every cost-of-capital calculation.

Source order:
    1. env ``STOCK_ANALYSIS_RISK_FREE_RATE`` (user override; fraction 0.045 or percent 4.5)
    2. FRED ``DGS10`` (10-year Treasury constant-maturity yield, percent), latest valid
       observation; DataLoader caches it for a day and serves a stale copy when FRED fails
    3. ``config.RISK_FREE_RATE_FALLBACK``

In offline mode step 2 only reads the cache, it never calls the network.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from StockAnalysisSystem.src import config

logger = logging.getLogger(__name__)

SERIES = "DGS10"
#: Plausible band for a US 10-year yield; values outside are treated as bad data.
PLAUSIBLE_RANGE = (0.0, 0.20)


@dataclass(frozen=True)
class RiskFreeRate:
    """A risk-free rate with where it came from."""

    rate: float
    source: str
    as_of: Optional[str] = None

    @property
    def label(self) -> str:
        """Human-readable form, e.g. ``4.12 % (FRED DGS10, as of 2026-11-24)``."""
        suffix = f", as of {self.as_of}" if self.as_of else ""
        return f"{self.rate * 100:.2f} % ({self.source}{suffix})"


def parse_rate(text: str) -> Optional[float]:
    """Parse a user-supplied rate: ``0.045`` and ``4.5`` both mean 4.5 %."""
    try:
        value = float(str(text).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if value > 1:
        value /= 100
    return value if PLAUSIBLE_RANGE[0] <= value <= PLAUSIBLE_RANGE[1] else None


def _from_env() -> Optional[RiskFreeRate]:
    raw = os.environ.get(config.RISK_FREE_RATE_ENV, "").strip()
    if not raw:
        return None
    value = parse_rate(raw)
    if value is None:
        logger.warning(f"Ignoring invalid {config.RISK_FREE_RATE_ENV} value")
        return None
    return RiskFreeRate(value, "user override")


def _from_fred(loader) -> Optional[RiskFreeRate]:
    frame = loader.get_fred_series(SERIES)
    if frame is None or frame.empty or SERIES not in frame.columns:
        return None
    series = frame[SERIES].dropna()
    if series.empty:
        return None
    value = float(series.iloc[-1]) / 100
    if not PLAUSIBLE_RANGE[0] <= value <= PLAUSIBLE_RANGE[1]:
        logger.warning(f"Implausible {SERIES} value {value:.4f}; using fallback")
        return None
    return RiskFreeRate(value, f"FRED {SERIES}", series.index[-1].strftime("%Y-%m-%d"))


def risk_free_rate(loader=None) -> RiskFreeRate:
    """Current risk-free rate (never raises)."""
    override = _from_env()
    if override:
        return override
    try:
        if loader is None:
            from StockAnalysisSystem.src.utils.data_loader import DataLoader
            loader = DataLoader()
        live = _from_fred(loader)
        if live:
            return live
    except Exception as e:  # provider/caching problems must never break a valuation
        logger.warning(f"Risk-free rate lookup failed: {e}")
    return RiskFreeRate(config.RISK_FREE_RATE_FALLBACK, "fallback constant")
