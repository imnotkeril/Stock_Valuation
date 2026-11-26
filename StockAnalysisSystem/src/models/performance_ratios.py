"""
Risk-adjusted performance ratios from daily closing prices (252 trading days a year).

All inputs at date t use only prices up to t (trailing history); nothing is forecast.
Returns are simple daily returns ``P_t / P_{t-1} - 1``. ``rf`` is an annual rate; its daily
equivalent is ``rf / 252``. Annualised means are arithmetic (daily mean x 252) except Calmar,
which uses the compound annual growth rate by definition.

* **Sortino** (Sortino & Price 1994): ``mean(r - rf_d) x 252 / (DD x sqrt(252))`` with downside
  deviation ``DD = sqrt(mean over ALL N days of min(r - rf_d, 0)^2)`` (divide by N, not by the
  number of down days; target = daily rf).
* **Beta**: OLS slope of the stock's daily excess returns on the benchmark's,
  ``cov(r - rf_d, m - rf_d) / var(m - rf_d)`` on the overlapping dates.
* **Treynor**: ``(Rp - rf) / beta``; N/A when beta <= 0 (the ratio flips sign meaninglessly).
* **Jensen's alpha** (Jensen 1968): ``Rp - [rf + beta (Rm - rf)]``.
* **Information ratio**: ``mean(r - m) x 252 / (std(r - m) x sqrt(252))``; the tracking error is
  vs the benchmark, never vs rf.
* **Calmar** (Young 1991): ``CAGR / |max drawdown|`` over the trailing 36 months
  (756 trading days) when the history is that long, otherwise over the whole history
  (``calmar_window`` says which). N/A without a drawdown.

Rp, Rm, beta, Treynor, Jensen and IR use the same overlapping window of stock and benchmark
returns. No benchmark (or fewer than ``min_obs`` common days) -> beta, Treynor, Jensen and IR are N/A.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
#: Calmar's trailing window: 36 months of trading days.
CALMAR_DAYS = 36 * 21


def _returns(close: Optional[pd.Series]) -> pd.Series:
    if close is None:
        return pd.Series(dtype=float)
    close = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    close = close[close > 0]
    return close.pct_change().dropna()


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> Optional[float]:
    """Annualised Sortino ratio with target = daily risk-free rate; None without downside."""
    if len(returns) == 0:
        return None
    excess = returns - rf / TRADING_DAYS
    downside = math.sqrt(float((np.minimum(excess, 0.0) ** 2).mean()))
    if downside == 0:
        return None
    return float(excess.mean()) * TRADING_DAYS / (downside * math.sqrt(TRADING_DAYS))


def calmar_ratio(close: pd.Series, window: int = CALMAR_DAYS) -> Dict[str, Any]:
    """CAGR / |max drawdown| over the last ``window`` returns (whole history if shorter)."""
    close = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    close = close[close > 0]
    trailing = len(close) > window
    if trailing:
        close = close.iloc[-(window + 1):]
    out: Dict[str, Any] = {"calmar": None, "calmar_cagr": None, "calmar_max_drawdown": None,
                                       "calmar_window": "36 months" if trailing else "whole history"}
    if len(close) < 2:
        return out
    years = (len(close) - 1) / TRADING_DAYS
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)
    max_dd = float((close / close.cummax() - 1).min())
    out.update(calmar_cagr=cagr, calmar_max_drawdown=max_dd)
    if max_dd < 0:
        out["calmar"] = cagr / abs(max_dd)
    return out


def performance_ratios(close: pd.Series, benchmark_close: Optional[pd.Series] = None, rf: float = 0.0,
                       min_obs: int = 20) -> Dict[str, Any]:
    """
    Sortino, Calmar, beta, Treynor, Jensen's alpha and Information ratio (see module docstring).

    Args:
        close: Daily closing prices of the stock (chronological).
        benchmark_close: Daily closing prices of the benchmark (S&P 500); None/empty -> benchmark ratios N/A.
        rf: Annual risk-free rate as a fraction.
        min_obs: Minimum number of daily returns (stock alone, and stock/benchmark overlap).

    Returns:
        Dict of floats or None (N/A). Empty dict when the stock has fewer than ``min_obs`` returns.
    """
    returns = _returns(close)
    if len(returns) < min_obs:
        return {}
    result: Dict[str, Any] = {"sortino": sortino_ratio(returns, rf)}
    result.update(calmar_ratio(close))
    result.update(beta=None, treynor=None, jensens_alpha=None, information_ratio=None, tracking_error=None,
                  benchmark_days=0)

    bench = _returns(benchmark_close)
    joined = pd.concat([returns.rename("p"), bench.rename("m")], axis=1, join="inner").dropna()
    result["benchmark_days"] = len(joined)
    if len(joined) < min_obs:
        return result
    rf_d = rf / TRADING_DAYS
    p_ex, m_ex = joined["p"] - rf_d, joined["m"] - rf_d
    var_m = float(m_ex.var(ddof=1))
    if var_m > 0:
        beta = float(np.cov(p_ex, m_ex, ddof=1)[0, 1] / var_m)
        rp, rm = float(joined["p"].mean()) * TRADING_DAYS, float(joined["m"].mean()) * TRADING_DAYS
        result["beta"] = beta
        result["jensens_alpha"] = rp - (rf + beta * (rm - rf))
        result["treynor"] = (rp - rf) / beta if beta > 0 else None
    active = joined["p"] - joined["m"]
    te = float(active.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    if te > 0:
        result["tracking_error"] = te
        result["information_ratio"] = float(active.mean()) * TRADING_DAYS / te
    return result
