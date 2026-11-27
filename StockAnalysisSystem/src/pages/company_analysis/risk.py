"""
Company risk tab.

Sections: financial health score, bankruptcy / distress models (Altman Z,
Springate, Zmijewski — chosen and interpreted by sector), red flags,
liquidity and solvency versus the sector, and market risk from the price
history (volatility, drawdown, Value at Risk, Sharpe ratio; Sortino, Treynor,
Calmar, Information ratio and Jensen's alpha vs the S&P 500), plus a simple
revenue stress test.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.utils.rates import risk_free_rate as current_risk_free_rate
from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.models.financial_statements import FinancialStatementAnalyzer
from StockAnalysisSystem.src.models.performance_ratios import performance_ratios
from StockAnalysisSystem.src.pages.company_analysis.charts import (
    drawdown_chart,
    gauge_chart,
    returns_histogram,
    score_bar_chart,
    show_chart,
    time_series_chart,
)
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    benchmark_table,
    canonical_sector,
    close_series,
    fmt_money,
    fmt_number,
    fmt_pct,
    get_company_ratios,
    get_sector_benchmarks,
    get_statement,
    has_any_statement,
    render_section,
    safe_call,
)
from StockAnalysisSystem.src.utils.financial_data import get_latest, to_float

logger = logging.getLogger("company_risk")

KEY = "risk_"
TRADING_DAYS = 252

LIQUIDITY_KEYS = ["current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity", "debt_to_assets",
                  "debt_to_ebitda", "net_debt_to_ebitda", "cash_to_debt", "interest_coverage"]

#: Gauge zones per distress model: (axis range, [(start, end, colour)])
GAUGE_ZONES = {
    "springate": ((-1.0, 3.0), [(-1.0, 0.862, NEGATIVE), (0.862, 3.0, POSITIVE)]),
    "zmijewski": ((0.0, 1.0), [(0.0, 0.5, POSITIVE), (0.5, 1.0, NEGATIVE)]),
}


def render_company_risk(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                        price_data: pd.DataFrame) -> None:
    """
    Render the risk tab.

    Args:
        ticker: Company ticker symbol.
        financial_data: Canonical financial data bundle.
        company_info: Normalised company information.
        price_data: Daily OHLCV price history.
    """
    sector = canonical_sector(company_info, financial_data)
    lender = is_balance_sheet_lender(ticker, financial_data, company_info, sector)
    if has_any_statement(financial_data):
        render_section("Financial health", _render_health_score, financial_data, sector, lender)
        render_section("Distress models", _render_distress_models, financial_data, sector)
        render_section("Liquidity and solvency", _render_liquidity, financial_data, company_info, price_data,
                       sector, lender)
        render_section("Stress test", _render_stress_test, financial_data, sector, lender)
        render_section("Scenario analysis", _render_scenarios, financial_data, company_info, price_data, sector,
                       lender)
    else:
        st.info("Financial statements are unavailable, so only market risk can be assessed.")
    render_section("Market risk", _render_market_risk, price_data, financial_data)


# ---------------------------------------------------------------------------
# Fundamental risk
# ---------------------------------------------------------------------------
#: Financial subsectors whose balance sheet is the business (deposits, policy reserves): FCF, operating
#: leverage and EBITDA scenarios do not apply. Payment networks, asset managers and exchanges keep them.
LENDER_SUBSECTORS = ("bank", "insurance")


def is_balance_sheet_lender(ticker: str, financial_data: Dict[str, Any], company_info: Optional[Dict[str, Any]],
                            sector: Optional[str]) -> bool:
    """True for banks and insurers (the financial sector model's subsector), False for V/MA/BLK-type firms."""
    if sector != "Financials":
        return False
    try:
        from StockAnalysisSystem.src.valuation.sector_specific.financial_sector import FinancialSectorValuation

        return FinancialSectorValuation().detect_subsector(ticker, financial_data, company_info) in LENDER_SUBSECTORS
    except Exception:  # noqa: BLE001 - unknown shape: keep the conservative bank treatment
        logger.exception("Financial subsector detection failed for %s", ticker)
        return True


def health_score(financial_data: Dict[str, Any], sector: Optional[str] = None,
                 lender: Optional[bool] = None) -> Dict[str, Any]:
    """
    Financial health score. For banks and insurers (``lender``; default: any Financials company) the
    cash-flow component is left out (free cash flow is not meaningful when deposits and reserves are
    operating liabilities) and the weights rebalance.
    """
    lender = (sector == "Financials") if lender is None else lender
    analyzer = FinancialStatementAnalyzer()
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    cash_flow = get_statement(financial_data, "cash_flow")
    cash_analysis = {} if lender else analyzer.analyze_cash_flow(cash_flow)
    return analyzer.calculate_financial_health_score(
        analyzer.analyze_income_statement(income), analyzer.analyze_balance_sheet(balance), cash_analysis)


def _render_health_score(financial_data: Dict[str, Any], sector: Optional[str] = None,
                         lender: bool = False) -> None:
    st.subheader("Financial health score")
    health = safe_call(lambda: health_score(financial_data, sector, lender), default={},
                       label="financial health score") or {}
    score = to_float(health.get("overall_score"))
    if score is None:
        st.info("Not enough data to compute a financial health score.")
        return

    left, right = st.columns([1, 2])
    with left:
        show_chart(gauge_chart(score, f"Overall: {health.get('rating', '')}", (0, 100),
                               [(0, 45, NEGATIVE), (45, 65, NEUTRAL), (65, 100, POSITIVE)],
                               number_format=".0f"), key=f"{KEY}health_gauge")
    with right:
        components = {name.replace("_", " ").title(): (value.get("score") if isinstance(value, dict) else value)
                      for name, value in (health.get("components") or {}).items()}
        show_chart(score_bar_chart(components, "Score by component"), key=f"{KEY}health_components")
    if health.get("assessment_summary"):
        st.markdown(health["assessment_summary"])
    if lender:
        st.caption("Bank / insurer: the cash-flow component is excluded, because free cash flow is not "
                   "meaningful when deposits and policy reserves are operating liabilities.")
    strengths, weaknesses = health.get("strengths") or [], health.get("weaknesses") or []
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths**")
        st.markdown("\n".join(f"- {s}" for s in strengths) if strengths else "_None identified_")
    with col2:
        st.markdown("**Weaknesses**")
        st.markdown("\n".join(f"- {w}" for w in weaknesses) if weaknesses else "_None identified_")


def _altman_gauge(model: Dict[str, Any]) -> Any:
    thresholds = model.get("thresholds") or {}
    distress = to_float(thresholds.get("grey", thresholds.get("distress")))
    safe = to_float(thresholds.get("safe"))
    if distress is None or safe is None:
        distress, safe = 1.81, 2.99
    high = max(safe * 2, (to_float(model.get("score")) or 0) * 1.1, safe + 1)
    return gauge_chart(model.get("score"), model.get("model", "Altman Z-Score"), (min(0.0, distress - 1), high),
                       [(min(0.0, distress - 1), distress, NEGATIVE), (distress, safe, NEUTRAL),
                        (safe, high, POSITIVE)], bar_color=model.get("color"))


def _render_distress_models(financial_data: Dict[str, Any], sector: str) -> None:
    st.subheader("Bankruptcy and distress models")
    assessment = safe_call(lambda: BankruptcyAnalyzer().get_comprehensive_risk_assessment(financial_data, sector),
                           default={}, label="bankruptcy assessment") or {}
    models = assessment.get("models") or {}
    if not models:
        st.info("Distress models could not be computed from the available statements.")
        return

    level = assessment.get("overall_risk_level", "N/A")
    st.markdown(f"Overall distress risk: <span style='color:{assessment.get('overall_color', '#e0e0e0')};"
                f"font-weight:bold'>{level}</span>", unsafe_allow_html=True)
    shown = set()  # each explanation is printed once (the overall text often repeats a model's text)
    if assessment.get("overall_description"):
        st.caption(assessment["overall_description"])
        shown.add(_normalized(assessment["overall_description"]))

    applicable = {k: v for k, v in models.items() if isinstance(v, dict) and v.get("applicable", True)
                  and to_float(v.get("score")) is not None}
    if not applicable:
        reasons = unique_sentences(v.get("interpretation") for v in models.values()
                                   if isinstance(v, dict) and v.get("interpretation")
                                   and _normalized(v.get("interpretation")) not in shown)
        if reasons or not shown:
            st.info(reasons or "The distress models are not applicable to this company.")
    else:
        columns = st.columns(len(applicable))
        for column, (key, model) in zip(columns, applicable.items()):
            with column:
                if key == "altman_z_score":
                    fig = _altman_gauge(model)
                elif key == "zmijewski":
                    axis, zones = GAUGE_ZONES["zmijewski"]
                    fig = gauge_chart(model.get("probability"), "Zmijewski P(distress)", axis, zones,
                                      bar_color=model.get("color"), number_format=".0%")
                else:
                    axis, zones = GAUGE_ZONES.get(key, ((-1.0, 3.0), []))
                    fig = gauge_chart(model.get("score"), model.get("model", key), axis, zones,
                                      bar_color=model.get("color"))
                show_chart(fig, key=f"{KEY}gauge_{key}")
                text = model.get("interpretation", "")
                if text and _normalized(text) not in shown:
                    st.caption(text)
                    shown.add(_normalized(text))

        rows = [{"Model": m.get("model", k), "Score": fmt_number(m.get("score")),
                 "Zone": str(m.get("assessment", "")).title(), "Formula": m.get("formula", "")}
                for k, m in applicable.items()]
        with st.expander("Model details"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, key=f"{KEY}models_table")
            st.caption("Zones: Altman Z — distress below 1.81, grey 1.81–2.99, safe above 2.99 (original model; "
                       "the Z'' variant used for non-manufacturers has its own cut-offs). Springate — distress "
                       "below 0.862. Zmijewski — distress when the estimated probability exceeds 50 %.")

    flags = assessment.get("red_flags") or []
    if flags:
        st.markdown("**Red flags**")
        st.markdown("\n".join(f"- {flag}" for flag in flags))


def _normalized(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def unique_sentences(texts) -> str:
    """Join texts, keeping each sentence once (model interpretations often repeat each other)."""
    import re

    seen, kept = set(), []
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
            key = _normalized(sentence)
            if key and key not in seen:
                seen.add(key)
                kept.append(sentence.strip())
    return " ".join(kept)


def _render_liquidity(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                      sector: str, lender: bool = False) -> None:
    st.subheader("Liquidity and solvency")
    ratios = get_company_ratios(financial_data, company_info, price_data)
    table = benchmark_table(ratios, get_sector_benchmarks(sector), LIQUIDITY_KEYS)
    if table.empty:
        st.info("Liquidity and leverage ratios are unavailable for this company.")
        return
    st.dataframe(table, hide_index=True, key=f"{KEY}liquidity_table")
    if lender:
        st.caption("For banks and insurers, current ratio and debt-based leverage are not meaningful: "
                   "deposits and policy reserves are operating liabilities, not financing.")


def _render_scenarios(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                      sector: str, lender: bool = False) -> None:
    """Macro shock scenarios on an EV/EBITDA valuation and historical crisis drawdowns."""
    from StockAnalysisSystem.src.models.stress_testing import historical_scenarios, scenario_valuations
    from StockAnalysisSystem.src.pages.company_analysis.page_helpers import market_snapshot
    from StockAnalysisSystem.src.valuation.capital import LONG_RUN_GROWTH, market_discount_rate

    st.subheader("Scenario analysis")
    snapshot = market_snapshot(financial_data, company_info, price_data)
    price = snapshot.get("price")

    st.markdown("**Macro shocks** - equity value per share from latest EBITDA at the current EV/EBITDA, "
                "after each shock")
    income = get_statement(financial_data, "income_statement")
    revenue, ebitda = get_latest(income, "revenue"), get_latest(income, "ebitda")
    ev = snapshot.get("enterprise_value")
    if lender:
        st.info("EBITDA-based scenarios do not apply to banks and insurers; see the historical crises below.")
    elif not (revenue and ebitda and ebitda > 0 and ev and ev > 0 and snapshot.get("shares")):
        st.info("Revenue, positive EBITDA, enterprise value and shares are required for the macro scenarios.")
    else:
        table = scenario_valuations(revenue, ebitda, ev / ebitda, snapshot.get("net_debt") or 0.0,
                                    snapshot["shares"], snapshot.get("total_debt") or 0.0,
                                    get_latest(income, "interest_expense") or 0.0)
        display = pd.DataFrame({
            "Scenario": table["scenario"],
            "Value / share": table["value_per_share"].map(lambda v: f"${v:,.2f}"),
            "Change": table["change"].map(lambda v: fmt_pct(v, signed=True)),
            "EBITDA margin": table["ebitda_margin"].map(fmt_pct),
            "EV/EBITDA": table["ev_ebitda"].map(lambda v: f"{v:.1f}x"),
            "Interest cover": table["interest_coverage"].map(lambda v: f"{v:.1f}x" if v is not None and v == v else "n/a"),
            "Assumptions": table["description"],
        })
        st.dataframe(display, width="stretch", hide_index=True)
        worst = table.iloc[1:].sort_values("change").iloc[0]
        st.caption(f"Worst case: **{worst['scenario']}** ({fmt_pct(worst['change'], signed=True)}). The base case "
                   "values the company at its current multiple, so it equals the market price up to the EV bridge. "
                   f"Rate shocks scale the multiple by (r - g) / (r + shock - g) with r = rf + ERP = "
                   f"{market_discount_rate():.1%}, g = {LONG_RUN_GROWTH:.1%}.")

    st.markdown("**Historical crises** - S&P 500 peak-to-trough drawdowns")
    beta = to_float(snapshot.get("beta"))
    crises = historical_scenarios(beta, price_data, price)
    display = pd.DataFrame({
        "Crisis": crises["crisis"], "Window": crises["window"],
        "S&P 500": crises["market_drawdown"].map(fmt_pct),
        "Beta-implied": crises["beta_implied"].map(lambda v: fmt_pct(v) if v is not None and v == v else "n/a"),
        "Implied price": crises["implied_price"].map(lambda v: f"${v:,.2f}" if v is not None and v == v else "n/a"),
        "Actual drawdown": crises["actual_drawdown"].map(
            lambda v: fmt_pct(v) if v is not None and v == v else "not in loaded history"),
    })
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(f"Beta {fmt_number(beta)}. Beta-implied moves assume the stock follows its average market "
               "sensitivity; actual drawdowns come from the loaded price history.")


def _render_stress_test(financial_data: Dict[str, Any], sector: Optional[str] = None,
                        lender: bool = False) -> None:
    """Effect of a revenue decline on operating income, given fixed vs variable costs."""
    st.subheader("Revenue stress test")
    if lender:
        st.info("The revenue / operating-income stress test does not apply to banks and insurers: their "
                "risk sits in credit losses, funding costs and capital ratios, not in operating leverage. "
                "See the historical crises in Scenario analysis below.")
        return
    income = get_statement(financial_data, "income_statement")
    revenue = get_latest(income, "revenue")
    operating_income = get_latest(income, "operating_income")
    gross_profit = get_latest(income, "gross_profit")
    interest = abs(get_latest(income, "interest_expense") or 0.0)
    if not revenue or revenue <= 0 or operating_income is None:
        st.info("Revenue and operating income are required for the stress test.")
        return
    contribution_margin = (gross_profit / revenue) if gross_profit and gross_profit > 0 else 0.4
    contribution_margin = st.slider(
        "Contribution margin (share of each revenue dollar that is variable profit)", 0.05, 1.0,
        float(round(min(max(contribution_margin, 0.05), 1.0), 2)), 0.01, key=f"{KEY}contribution_margin",
        help="Defaults to the gross margin: operating expenses are treated as fixed in the short run.")
    rows = []
    for shock in (0.0, -0.05, -0.10, -0.20, -0.30):
        stressed_ebit = operating_income + revenue * shock * contribution_margin
        coverage = stressed_ebit / interest if interest else None
        rows.append({
            "Revenue change": fmt_pct(shock, 0, signed=True),
            "Revenue": fmt_money(revenue * (1 + shock)),
            "Operating income": fmt_money(stressed_ebit),
            "Operating margin": fmt_pct(stressed_ebit / (revenue * (1 + shock))),
            "Interest coverage": f"{coverage:.1f}x" if coverage is not None else "No interest",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, key=f"{KEY}stress_table")
    st.caption("Operating leverage: a revenue decline reduces operating income by revenue change × contribution "
               "margin. Interest coverage below ~1.5x signals refinancing risk.")


# ---------------------------------------------------------------------------
# Market risk
# ---------------------------------------------------------------------------
def market_risk_metrics(close: pd.Series, risk_free_rate: Optional[float] = None) -> Dict[str, Optional[float]]:
    """
    Annualised return/volatility, Sharpe, max drawdown and 1-day historical VaR/CVaR (95 %, 99 %).

    ``risk_free_rate`` defaults to the current rate (utils.rates).
    """
    close = pd.to_numeric(close, errors="coerce").dropna()
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return {}
    volatility = float(returns.std() * np.sqrt(TRADING_DAYS))
    years = len(returns) / TRADING_DAYS
    annual_return = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) if years > 0 else None
    drawdown = close / close.cummax() - 1
    var95, var99 = float(returns.quantile(0.05)), float(returns.quantile(0.01))
    if risk_free_rate is None:
        risk_free_rate = current_risk_free_rate().rate
    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": (annual_return - risk_free_rate) / volatility if annual_return is not None and volatility
        else None,
        "max_drawdown": float(drawdown.min()),
        "var_95": var95,
        "cvar_95": float(returns[returns <= var95].mean()),
        "var_99": var99,
        "worst_day": float(returns.min()),
        "best_day": float(returns.max()),
    }


#: Market benchmark for beta, Treynor, Jensen's alpha and the Information ratio (S&P 500 ETF).
MARKET_BENCHMARK = "SPY"


@st.cache_data(ttl=3600, show_spinner=False)
def _benchmark_close(start: str, end: str) -> pd.Series:
    """S&P 500 (SPY) closes for the stock's date range; empty when unavailable (offline, provider error)."""
    from StockAnalysisSystem.src.utils.data_loader import DataLoader

    try:
        return close_series(DataLoader().get_historical_prices(MARKET_BENCHMARK, start, end))
    except Exception:  # noqa: BLE001 - a missing benchmark only turns its ratios into N/A
        logger.info("Benchmark %s unavailable", MARKET_BENCHMARK, exc_info=True)
        return pd.Series(dtype=float)


def _ratio(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return "N/A"
    return fmt_pct(value) if pct else f"{value:.2f}"


def _render_performance_ratios(close: pd.Series, rf: float) -> None:
    """Sortino, Treynor, Calmar, Information ratio and Jensen's alpha (models/performance_ratios.py)."""
    bench = pd.Series(dtype=float)
    if isinstance(close.index, pd.DatetimeIndex) and len(close):
        bench = _benchmark_close(close.index.min().date().isoformat(), close.index.max().date().isoformat())
    ratios = performance_ratios(close, bench, rf)
    if not ratios:
        return
    cols = st.columns(5)
    cols[0].metric("Sortino ratio", _ratio(ratios["sortino"]),
                   help="Excess return over rf per unit of downside deviation (only days below rf count as risk).")
    cols[1].metric("Treynor ratio", _ratio(ratios["treynor"], pct=True),
                   help="Excess return over rf per unit of beta (systematic risk). N/A without a benchmark or "
                        "when beta <= 0.")
    cols[2].metric("Calmar ratio", _ratio(ratios["calmar"]),
                   help=f"CAGR / |max drawdown| over the {ratios['calmar_window']}. N/A without a drawdown.")
    cols[3].metric("Information ratio", _ratio(ratios["information_ratio"]),
                   help=f"Annualised return above {MARKET_BENCHMARK} / tracking error (std of the return "
                        "difference). N/A without a benchmark.")
    cols[4].metric("Jensen's alpha", _ratio(ratios["jensens_alpha"], pct=True),
                   help="Annual return above what CAPM predicts: Rp - [rf + beta (Rm - rf)]. N/A without a "
                        "benchmark.")
    if ratios["beta"] is not None:
        bench_text = (f"Benchmark: S&P 500 ({MARKET_BENCHMARK}), {ratios['benchmark_days']} common days, "
                      f"regression beta {ratios['beta']:.2f}.")
    else:
        bench_text = f"S&P 500 ({MARKET_BENCHMARK}) history unavailable, so benchmark-based ratios are N/A."
    st.caption("Sortino = (Rp - rf) / downside deviation; Treynor = (Rp - rf) / beta; Calmar = CAGR / |max "
               f"drawdown| ({ratios['calmar_window']}); Information ratio = active return / tracking error; "
               "Jensen's alpha = Rp - [rf + beta (Rm - rf)]. Daily returns, annualised with 252 days. "
               + bench_text)


def _render_market_risk(price_data: pd.DataFrame, financial_data: Dict[str, Any]) -> None:
    st.subheader("Market risk")
    close = close_series(price_data)
    rf = current_risk_free_rate()
    metrics = market_risk_metrics(close, rf.rate)
    if not metrics:
        st.info("At least a month of price history is needed for market risk metrics.")
        return
    beta = to_float(((financial_data or {}).get("market_data") or {}).get("beta"))
    cols = st.columns(5)
    cols[0].metric("Annualised volatility", fmt_pct(metrics["annual_volatility"]))
    cols[1].metric("Beta (provider)", f"{beta:.2f}" if beta is not None else "N/A",
                   help="Beta reported by the data provider (Yahoo: 5 years of monthly returns). Treynor and "
                        "Jensen's alpha below use the regression beta over the selected price history instead; "
                        "the valuation models use the Blume-adjusted beta.")
    cols[2].metric("Max drawdown", fmt_pct(metrics["max_drawdown"]))
    cols[3].metric("1-day VaR (95 %)", fmt_pct(metrics["var_95"], 2))
    cols[4].metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics["sharpe_ratio"] is not None
                   else "N/A")
    st.caption(f"Over the selected price history. Sharpe uses a {rf.label} risk-free rate. "
               f"VaR: on 95 % of days the loss was smaller than {fmt_pct(-metrics['var_95'], 2)}; "
               f"the average loss on the worst 5 % of days (CVaR) was {fmt_pct(-metrics['cvar_95'], 2)}.")
    _render_performance_ratios(close, rf.rate)

    returns = close.pct_change().dropna()
    left, right = st.columns(2)
    with left:
        show_chart(returns_histogram(returns, "Daily return distribution",
                                     {"VaR 95%": metrics["var_95"], "VaR 99%": metrics["var_99"]}),
                   key=f"{KEY}returns_hist")
    with right:
        show_chart(drawdown_chart(close / close.cummax() - 1, "Drawdown from peak"), key=f"{KEY}drawdown")
    rolling = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
    show_chart(time_series_chart({"63-day rolling volatility": rolling.dropna()}, "Rolling volatility (annualised)",
                                 tickformat=".0%"), key=f"{KEY}rolling_vol")
