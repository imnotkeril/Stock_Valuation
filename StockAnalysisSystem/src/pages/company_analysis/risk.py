"""
Company risk tab.

Sections: financial health score, bankruptcy / distress models (Altman Z,
Springate, Zmijewski — chosen and interpreted by sector), red flags,
liquidity and solvency versus the sector, and market risk from the price
history (volatility, drawdown, Value at Risk, Sharpe ratio), plus a simple
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
    if has_any_statement(financial_data):
        render_section("Financial health", _render_health_score, financial_data)
        render_section("Distress models", _render_distress_models, financial_data, sector)
        render_section("Liquidity and solvency", _render_liquidity, financial_data, company_info, price_data,
                       sector)
        render_section("Stress test", _render_stress_test, financial_data)
        render_section("Scenario analysis", _render_scenarios, financial_data, company_info, price_data, sector)
    else:
        st.info("Financial statements are unavailable, so only market risk can be assessed.")
    render_section("Market risk", _render_market_risk, price_data, financial_data)


# ---------------------------------------------------------------------------
# Fundamental risk
# ---------------------------------------------------------------------------
def _render_health_score(financial_data: Dict[str, Any]) -> None:
    st.subheader("Financial health score")
    analyzer = FinancialStatementAnalyzer()
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    cash_flow = get_statement(financial_data, "cash_flow")
    health = safe_call(lambda: analyzer.calculate_financial_health_score(
        analyzer.analyze_income_statement(income), analyzer.analyze_balance_sheet(balance),
        analyzer.analyze_cash_flow(cash_flow)), default={}, label="financial health score") or {}
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
    if assessment.get("overall_description"):
        st.caption(assessment["overall_description"])

    applicable = {k: v for k, v in models.items() if isinstance(v, dict) and v.get("applicable", True)
                  and to_float(v.get("score")) is not None}
    if not applicable:
        reasons = {v.get("interpretation") for v in models.values() if isinstance(v, dict) and v.get("interpretation")}
        st.info(" ".join(sorted(reasons)) or "The distress models are not applicable to this company.")
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
                st.caption(model.get("interpretation", ""))

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


def _render_liquidity(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                      sector: str) -> None:
    st.subheader("Liquidity and solvency")
    ratios = get_company_ratios(financial_data, company_info, price_data)
    table = benchmark_table(ratios, get_sector_benchmarks(sector), LIQUIDITY_KEYS)
    if table.empty:
        st.info("Liquidity and leverage ratios are unavailable for this company.")
        return
    st.dataframe(table, hide_index=True, key=f"{KEY}liquidity_table")
    if sector == "Financials":
        st.caption("For banks and insurers, current ratio and debt-based leverage are not meaningful: "
                   "deposits and policy reserves are operating liabilities, not financing.")


def _render_scenarios(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                      sector: str) -> None:
    """Macro shock scenarios on an EV/EBITDA valuation and historical crisis drawdowns."""
    from StockAnalysisSystem.src.models.stress_testing import historical_scenarios, scenario_valuations
    from StockAnalysisSystem.src.pages.company_analysis.page_helpers import market_snapshot

    st.subheader("Scenario analysis")
    snapshot = market_snapshot(financial_data, company_info, price_data)
    price = snapshot.get("price")

    st.markdown("**Macro shocks** - equity value per share from latest EBITDA at the current EV/EBITDA, "
                "after each shock")
    income = get_statement(financial_data, "income_statement")
    revenue, ebitda = get_latest(income, "revenue"), get_latest(income, "ebitda")
    ev = snapshot.get("enterprise_value")
    if sector == "Financials":
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
                   "Rate shocks scale the multiple by (r - g) / (r + shock - g) with r = 9 %, g = 2.5 %.")

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


def _render_stress_test(financial_data: Dict[str, Any]) -> None:
    """Effect of a revenue decline on operating income, given fixed vs variable costs."""
    st.subheader("Revenue stress test")
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
    cols[1].metric("Beta", f"{beta:.2f}" if beta is not None else "N/A")
    cols[2].metric("Max drawdown", fmt_pct(metrics["max_drawdown"]))
    cols[3].metric("1-day VaR (95 %)", fmt_pct(metrics["var_95"], 2))
    cols[4].metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics["sharpe_ratio"] is not None
                   else "N/A")
    st.caption(f"Over the selected price history. Sharpe uses a {rf.label} risk-free rate. "
               f"VaR: on 95 % of days the loss was smaller than {fmt_pct(-metrics['var_95'], 2)}; "
               f"the average loss on the worst 5 % of days (CVaR) was {fmt_pct(-metrics['cvar_95'], 2)}.")

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
