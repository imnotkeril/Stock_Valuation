"""
Multi-sector comparison view of the Sector Analysis page.

Compares several sectors side by side: sector ETF price performance (vs. SPY) and
the median fundamentals of each sector's universe companies. The calculations are
pure functions so they can be tested offline; :func:`render_sector_comparison`
draws the Streamlit view.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import SECTORS
from StockAnalysisSystem.src.industry.universe import get_sector_etf, get_universe
from StockAnalysisSystem.src.pages.screening.stock_screener import (
    METRICS, close_prices, format_metric, get_stock_snapshot, load_price_history, metric_label,
    performance_stats, to_display,
)

PREFIX = "sector_cmp"
RESULTS_KEY = f"{PREFIX}_results"
MARKET_ETF = "SPY"
DEFAULT_SECTORS = ["Technology", "Financials", "Energy", "Healthcare"]
PERIODS = {"6 Months": 182, "1 Year": 365, "3 Years": 3 * 365, "5 Years": 5 * 365}
FUNDAMENTAL_METRICS = ["pe_ratio", "pb_ratio", "ev_ebitda", "net_margin", "roe", "revenue_growth",
                       "debt_equity", "dividend_yield"]
#: Metrics where a lower median is "better" when ranking sectors
LOWER_IS_BETTER = {"pe_ratio", "pb_ratio", "ev_ebitda", "debt_equity"}
PERFORMANCE_COLUMNS = {"total_return": "Total return", "annual_return": "Annual return",
                       "volatility": "Volatility", "max_drawdown": "Max drawdown", "sharpe": "Sharpe ratio"}


def normalized_prices(closes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Closing prices rebased to 100 at each series' first common date."""
    frame = pd.DataFrame({name: s for name, s in closes.items() if isinstance(s, pd.Series) and not s.empty})
    frame = frame.dropna()
    if frame.empty:
        return frame
    return frame / frame.iloc[0] * 100


def performance_table(closes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """One row per series with :func:`performance_stats` (fractions; Sharpe as a ratio)."""
    rows = {name: performance_stats(series) for name, series in closes.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[list(PERFORMANCE_COLUMNS)]


def fundamentals_table(metrics_by_sector: Mapping[str, Sequence[Mapping[str, Any]]],
                       metric_keys: Sequence[str] = FUNDAMENTAL_METRICS) -> pd.DataFrame:
    """Median of each metric over the companies of each sector (metric units) plus a ``companies`` count."""
    rows = {}
    for sector, company_metrics in metrics_by_sector.items():
        frame = pd.DataFrame(list(company_metrics), columns=list(metric_keys))
        frame = frame.apply(pd.to_numeric, errors="coerce")
        row = frame.median(skipna=True).to_dict()
        row["companies"] = len(company_metrics)
        rows[sector] = row
    return pd.DataFrame.from_dict(rows, orient="index")[list(metric_keys) + ["companies"]]


def best_sector_by_metric(fundamentals: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Sector with the most attractive median per metric (lowest for multiples and leverage)."""
    best: Dict[str, Optional[str]] = {}
    for metric in fundamentals.columns:
        if metric == "companies":
            continue
        column = fundamentals[metric].dropna()
        if metric in LOWER_IS_BETTER:
            column = column[column > 0]  # negative multiples are not "cheap"
        if column.empty:
            best[metric] = None
        else:
            best[metric] = column.idxmin() if metric in LOWER_IS_BETTER else column.idxmax()
    return best


def load_comparison(sectors: Sequence[str], days: int, max_companies: int,
                    progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Download ETF prices and company snapshots for ``sectors`` (cached by the screener helpers)."""
    closes: Dict[str, pd.Series] = {}
    etfs: Dict[str, Optional[str]] = {}
    metrics_by_sector: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[str] = []
    total = len(sectors) * (max_companies + 1) + 1
    done = 0

    def tick(text: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(min(done / total, 1.0), text)

    market = close_prices(load_price_history(MARKET_ETF, days))
    tick(MARKET_ETF)
    for sector in sectors:
        etf = get_sector_etf(sector)
        etfs[sector] = etf
        if etf:
            series = close_prices(load_price_history(etf, days))
            if series.empty:
                failures.append(etf)
            else:
                closes[sector] = series
        tick(f"{sector} ETF")
        metrics_by_sector[sector] = []
        for record in get_universe(sector)[:max_companies]:
            snapshot = get_stock_snapshot(record["symbol"], lookback_days=days)
            if snapshot is None:
                failures.append(record["symbol"])
            else:
                metrics_by_sector[sector].append(snapshot["metrics"])
            tick(f"{sector}: {record['symbol']}")
    if not market.empty:
        closes[f"Market ({MARKET_ETF})"] = market
    return {"closes": closes, "etfs": etfs, "metrics": metrics_by_sector, "failures": failures,
            "sectors": list(sectors), "days": days}


def _display_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(index=fundamentals.index)
    for metric in FUNDAMENTAL_METRICS:
        table[metric_label(metric, with_unit=True)] = [
            to_display(v, METRICS[metric]["kind"]) for v in fundamentals[metric]]
    table["Companies"] = fundamentals["companies"].astype(int)
    return table


def _display_performance(performance: pd.DataFrame) -> pd.DataFrame:
    table = performance.rename(columns=PERFORMANCE_COLUMNS).copy()
    for column in ["Total return", "Annual return", "Volatility", "Max drawdown"]:
        table[column] = table[column] * 100
    return table.rename(columns={c: f"{c} (%)" for c in ["Total return", "Annual return", "Volatility",
                                                            "Max drawdown"]})


def render_sector_comparison() -> None:
    st.header("Compare sectors")
    st.caption("Sector ETF performance against the market and the median fundamentals of each sector's "
               "largest companies in the universe.")
    col_sectors, col_period, col_n = st.columns([3, 1, 1], vertical_alignment="bottom")
    with col_sectors:
        sectors = st.multiselect("Sectors", SECTORS, default=[s for s in DEFAULT_SECTORS if s in SECTORS],
                                 key=f"{PREFIX}_sectors", max_selections=6, placeholder="Pick 2-6 sectors")
    with col_period:
        period = st.selectbox("Period", list(PERIODS), index=1, key=f"{PREFIX}_period")
    with col_n:
        max_companies = st.number_input("Companies per sector", 3, 20, 8, key=f"{PREFIX}_n",
                                        help="Largest companies of each sector in the universe.")
    run = st.button("Compare sectors", type="primary", key=f"{PREFIX}_run", icon=":material/compare_arrows:",
                    disabled=len(sectors) < 2)
    if len(sectors) < 2:
        st.info("Select at least two sectors to compare.")

    if run:
        bar = st.progress(0.0, text="Loading sector data...")
        try:
            st.session_state[RESULTS_KEY] = load_comparison(
                sectors, PERIODS[period], int(max_companies),
                progress=lambda value, text: bar.progress(value, text=f"Loading {text}..."))
        finally:
            bar.empty()

    results = st.session_state.get(RESULTS_KEY)
    if not results:
        return
    if results["sectors"] != list(sectors) or results["days"] != PERIODS[period]:
        st.info("Selection changed since the last run - click **Compare sectors** to refresh.",
                icon=":material/info:")
    if results["failures"]:
        st.warning(f"Data unavailable for {len(results['failures'])} tickers (skipped): "
                   f"{', '.join(results['failures'][:15])}")

    tab_perf, tab_fund, tab_best = st.tabs(["Performance", "Fundamentals", "Leaders by metric"])
    with tab_perf:
        prices = normalized_prices(results["closes"])
        if prices.empty:
            st.info("No price history available for the selected sectors.")
        else:
            st.markdown("**Growth of 100** (sector ETFs vs. market)")
            st.line_chart(prices)
            etf_note = ", ".join(f"{s}: {e or 'n/a'}" for s, e in results["etfs"].items())
            st.caption(f"ETFs - {etf_note}.")
            st.dataframe(_display_performance(performance_table(results["closes"])), width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f")
                                        for c in _display_performance(performance_table(results["closes"]))})
    fundamentals = fundamentals_table(results["metrics"])
    with tab_fund:
        st.dataframe(_display_fundamentals(fundamentals), width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in _display_fundamentals(fundamentals).columns if c != "Companies"})
        metric = st.selectbox("Chart metric", FUNDAMENTAL_METRICS, format_func=metric_label,
                              key=f"{PREFIX}_chart_metric")
        values = pd.Series([to_display(v, METRICS[metric]["kind"]) for v in fundamentals[metric]],
                           index=fundamentals.index, name=metric_label(metric, with_unit=True)).dropna()
        if values.empty:
            st.info("No data for this metric.")
        else:
            st.bar_chart(values)
        st.caption("Medians over the companies with data; financial-sector multiples are not directly comparable "
                   "with industrial ones.")
    with tab_best:
        best = best_sector_by_metric(fundamentals)
        rows = [{"Metric": metric_label(m), "Leader": s or "n/a",
                 "Median": format_metric(fundamentals.at[s, m], m) if s else "n/a",
                 "Rule": "lowest positive" if m in LOWER_IS_BETTER else "highest"}
                for m, s in best.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
