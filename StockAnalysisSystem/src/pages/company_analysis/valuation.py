"""
Company valuation tab.

Sub-tabs:

* Fair Value Summary - the sector model's blended fair value, per-method
  values versus the current price and the recommendation.
* DCF - interactive two-stage DCF on free cash flow with a discount-rate x
  terminal-growth sensitivity grid.
* Relative - current multiples versus sector benchmarks and the prices they imply.
* Asset-Based - book, tangible book and adjusted net asset value per share.
* Sector Model - the sector-specific model with its subsector methods and
  adjustable assumptions (pages/company_analysis/sector_valuation.py).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import SECTOR_DCF_PARAMETERS, DCF_PARAMETERS
from StockAnalysisSystem.src.industry.benchmarks import irrelevant_multiples, live_valuation_medians
from StockAnalysisSystem.src.pages.company_analysis.charts import (
    dcf_cash_flow_chart,
    donut_chart,
    ratio_comparison_chart,
    sensitivity_heatmap,
    show_chart,
    value_comparison_chart,
)
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    canonical_sector,
    dcf_projection,
    fmt_money,
    fmt_pct,
    fmt_price,
    free_cash_flow_series,
    get_company_ratios,
    get_sector_benchmarks,
    get_statement,
    market_snapshot,
    render_section,
)
from StockAnalysisSystem.src.pages.company_analysis.sector_valuation import (
    get_valuation_factory,
    render_sector_valuation,
)
from StockAnalysisSystem.src.utils.financial_data import cagr, get_latest, safe_divide, to_float
from StockAnalysisSystem.src.valuation import capital

logger = logging.getLogger("company_valuation")

KEY = "val_"

#: Multiples compared in the relative valuation: key -> (label, metric description)
MULTIPLES = {
    "pe_ratio": ("P/E", "earnings"),
    "ps_ratio": ("P/S", "sales"),
    "pb_ratio": ("P/B", "book value"),
    "ev_ebitda": ("EV/EBITDA", "EBITDA"),
}


def render_company_valuation(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                             price_data: pd.DataFrame, sector_data: Optional[pd.DataFrame] = None) -> None:
    """
    Render the valuation tab.

    Args:
        ticker: Company ticker symbol.
        financial_data: Canonical financial data bundle.
        company_info: Normalised company information.
        price_data: Daily OHLCV price history.
        sector_data: Sector ETF prices (unused; accepted for a uniform signature).
    """
    snapshot = market_snapshot(financial_data, company_info, price_data)
    tabs = st.tabs(["Fair Value Summary", "DCF", "APV", "Relative", "Asset-Based", "Sector Model", "Precedents"])
    with tabs[6]:
        render_section("Precedent transactions", _render_precedents, financial_data, company_info, snapshot)
    with tabs[0]:
        render_section("Fair value summary", _render_summary, ticker, financial_data, company_info)
    with tabs[1]:
        render_section("DCF", _render_dcf, ticker, financial_data, company_info, snapshot)
    with tabs[2]:
        render_section("APV", _render_apv, ticker, financial_data, company_info, snapshot)
    with tabs[3]:
        render_section("Relative valuation", _render_relative, financial_data, company_info, price_data, snapshot)
    with tabs[4]:
        render_section("Asset-based valuation", _render_asset_based, financial_data, snapshot)
    with tabs[5]:
        render_section("Sector model", render_sector_valuation, ticker, financial_data, company_info, price_data,
                       key_prefix=f"{KEY}sector")


# ---------------------------------------------------------------------------
# Fair value summary
# ---------------------------------------------------------------------------
def _render_summary(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any]) -> None:
    sector = canonical_sector(company_info, financial_data)
    with st.spinner("Valuing the company..."):
        result = get_valuation_factory().get_company_valuation(ticker, sector=sector, financial_data=financial_data)
    if result.get("error") and not result.get("valuation_methods"):
        st.warning(f"The valuation could not be completed: {result['error']}")
        return

    fair, price = to_float(result.get("value_per_share")), to_float(result.get("current_price"))
    upside = to_float(result.get("potential_upside"))
    recommendation = result.get("recommendation") or {}
    cols = st.columns(4)
    cols[0].metric("Weighted fair value", fmt_price(fair))
    cols[1].metric("Current price", fmt_price(price))
    cols[2].metric("Upside / downside", fmt_pct(upside / 100 if upside is not None else None, signed=True))
    cols[3].markdown(
        f"<div class='metric-card'><div class='metric-label'>Recommendation</div>"
        f"<div class='metric-value' style='color:{recommendation.get('color', '#9e9e9e')}'>"
        f"{recommendation.get('label', 'N/A')}</div>"
        f"{_confidence_line(recommendation)}</div>", unsafe_allow_html=True)
    if recommendation.get("rationale"):
        st.caption(recommendation["rationale"])
    if recommendation.get("confidence_reasons") and recommendation.get("confidence") != "Low":
        st.caption(f"Confidence {recommendation['confidence'].lower()}: "
                   f"{'; '.join(recommendation['confidence_reasons'])}.")

    methods = result.get("valuation_methods") or {}
    weights = (result.get("valuation_summary") or {}).get("weights") or {}
    labels = [m.get("method", k) for k, m in methods.items() if k in weights]
    values = [m.get("value_per_share") for k, m in methods.items() if k in weights]
    show_chart(value_comparison_chart(labels + ["Weighted fair value"], values + [fair], price,
                                      "Fair value by method vs current price"), key=f"{KEY}summary_chart")

    rows = []
    for key, method in methods.items():
        value = to_float(method.get("value_per_share"))
        rows.append({
            "Method": method.get("method", key),
            "Value / share": fmt_price(value),
            "Weight": fmt_pct(weights.get(key), 0) if key in weights else "—",
            "Upside": fmt_pct(value / price - 1, signed=True) if value and price else "—",
            "Note": method.get("error") or ("" if key in weights else method.get("note", "Not applicable")),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, key=f"{KEY}summary_table")
    st.caption(f"Model: {result.get('model')} · subsector: {result.get('subsector') or 'n/a'}. "
               "Adjust subsector, methods and assumptions in the Sector Model tab.")


def _confidence_line(recommendation: Dict[str, Any]) -> str:
    """Small 'Confidence: X' line under the recommendation badge (empty when unknown)."""
    level = recommendation.get("confidence")
    if not level:
        return ""
    return f"<div class='metric-label'>Confidence: {level}</div>"


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------
def _sector_params(sector: str) -> Dict[str, Any]:
    return {**DCF_PARAMETERS, **SECTOR_DCF_PARAMETERS.get(sector, {})}


def _shared_capital(financial_data: Dict[str, Any], sector: str) -> Dict[str, Any]:
    """The cost of capital every model uses (valuation/capital.py, ADR-007)."""
    return capital.cost_of_capital_for(financial_data or {}, float(_sector_params(sector)["default_discount_rate"]))


def _rate_caption(components: Dict[str, Any]) -> str:
    return (f"rf {components['risk_free_rate_label']} · ERP {components['equity_risk_premium']:.1%} · "
            f"beta {components['adjusted_beta']:.2f} (Blume-adjusted) · cost of equity "
            f"{components['cost_of_equity']:.2%} · pre-tax cost of debt {components['pre_tax_cost_of_debt']:.2%}")


def _dcf_defaults(sector: str, fcf: pd.Series, components: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """DCF defaults: the shared WACC when ``components`` is given, else the sector table rate."""
    params = _sector_params(sector)
    growth = cagr(fcf.tail(4)) if len(fcf) >= 2 else None
    rate = components["wacc"] if components else params["default_discount_rate"]
    return {
        "discount_rate": float(rate),
        "terminal_growth": float(params["terminal_growth_rate"]),
        "forecast_years": int(params["forecast_years"]),
        "growth": float(min(max(growth if growth is not None else 0.05, -0.05), 0.25)),
    }


def _render_dcf(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                snapshot: Dict[str, Optional[float]]) -> None:
    st.subheader("Discounted cash flow")
    sector = canonical_sector(company_info, financial_data)
    if sector == "Financials":
        st.info("Free-cash-flow DCF is not meaningful for banks and insurers (debt is raw material, not "
                "financing). Use the Sector Model tab: excess-return and justified P/B methods.")
    fcf = free_cash_flow_series(get_statement(financial_data, "cash_flow"))
    if fcf.empty:
        st.info("Free cash flow history is unavailable, so a DCF cannot be built.")
        return
    components = _shared_capital(financial_data, sector)
    defaults = _dcf_defaults(sector, fcf, components)
    base_options = {"Latest year": float(fcf.iloc[-1]), f"Average of last {min(len(fcf), 3)} years":
                    float(fcf.tail(3).mean())}
    sbc = abs(get_latest(get_statement(financial_data, "cash_flow"), "stock_based_compensation") or 0.0)

    col1, col2 = st.columns(2)
    with col1:
        base_label = st.radio("Starting free cash flow", list(base_options), horizontal=True, key=f"{KEY}dcf_base")
        growth = st.slider("FCF growth (forecast period)", -10.0, 30.0, round(defaults["growth"] * 100, 1), 0.5,
                           format="%.1f%%", key=f"{KEY}dcf_growth") / 100
        years = st.slider("Forecast years", 3, 15, defaults["forecast_years"], key=f"{KEY}dcf_years")
        fade = st.checkbox("Fade growth to terminal growth", value=True, key=f"{KEY}dcf_fade",
                           help="Growth declines linearly from the forecast rate in year 1 to the terminal rate "
                                "in the final year, instead of jumping at the end of the forecast.")
        deduct_sbc = st.checkbox(f"Deduct stock-based compensation ({fmt_money(sbc)} last year)", value=sbc > 0,
                                 disabled=not sbc, key=f"{KEY}dcf_sbc",
                                 help="SBC is a real cost to shareholders (dilution) but is added back in "
                                      "operating cash flow. The Technology sector model deducts it too.")
    with col2:
        discount = st.slider("Discount rate (WACC)", 4.0, 20.0, round(defaults["discount_rate"] * 100, 2), 0.25,
                             format="%.2f%%", key=f"{KEY}dcf_discount",
                             help="Default: the WACC shared by every model (CAPM cost of equity, after-tax cost "
                                  "of debt, market-value weights).") / 100
        terminal = st.slider("Terminal growth", 0.0, 5.0, round(defaults["terminal_growth"] * 100, 2), 0.25,
                             format="%.2f%%", key=f"{KEY}dcf_terminal") / 100
    st.caption(f"Default WACC {components['wacc']:.2%}: {_rate_caption(components)}.")
    _render_method_differences(components)

    base_fcf = base_options[base_label] - (sbc if deduct_sbc else 0.0)
    net_debt = snapshot.get("net_debt")
    shares = snapshot.get("shares")
    if base_fcf <= 0:
        if sector == "Financials":  # already explained above; bank FCF is routinely negative
            return
        st.warning(f"Starting free cash flow is negative ({fmt_money(base_fcf)}): a DCF on current cash flow "
                   "would imply no value. Try the multi-year average or the Sector Model tab.")
        return
    if discount <= terminal:
        st.warning("The discount rate must be higher than the terminal growth rate.")
        return
    dcf = dcf_projection(base_fcf, growth, discount, terminal, years, net_debt, shares, fade=fade)
    if dcf is None:
        st.info("Shares outstanding are unavailable, so per-share value cannot be computed.")
        return

    price = snapshot.get("price")
    value = dcf["value_per_share"]
    cols = st.columns(4)
    cols[0].metric("Value per share", fmt_price(value),
                   delta=fmt_pct(value / price - 1, signed=True) if price and value else None)
    cols[1].metric("Enterprise value", fmt_money(dcf["enterprise_value"]))
    cols[2].metric("Equity value", fmt_money(dcf["equity_value"]))
    terminal_share = safe_divide(dcf["pv_terminal"], dcf["enterprise_value"])
    cols[3].metric("Terminal value share", fmt_pct(terminal_share))
    if terminal_share and terminal_share > 0.8:
        st.caption("More than 80 % of the value comes from the terminal value: the result is very sensitive "
                   "to long-term assumptions.")

    left, right = st.columns([2, 1])
    with left:
        show_chart(dcf_cash_flow_chart(fcf, dcf["forecast_fcf"], dcf["present_values"], "Free cash flow forecast"),
                   key=f"{KEY}dcf_chart")
    with right:
        show_chart(donut_chart(["PV of forecast FCF", "PV of terminal value"], [dcf["pv_fcf"], dcf["pv_terminal"]],
                               "Enterprise value composition"), key=f"{KEY}dcf_donut")

    discount_grid = sorted({round(discount + step, 4) for step in np.arange(-0.02, 0.0201, 0.01)})
    terminal_grid = sorted({round(max(terminal + step, 0.0), 4) for step in np.arange(-0.01, 0.0101, 0.005)})
    z: List[List[Optional[float]]] = []
    for g in terminal_grid:
        row = []
        for r in discount_grid:
            point = dcf_projection(base_fcf, growth, r, g, years, net_debt, shares, fade=fade) if r > g else None
            row.append(point["value_per_share"] if point else None)
        z.append(row)
    show_chart(sensitivity_heatmap(z, discount_grid, terminal_grid, "Sensitivity: value per share",
                                   "Discount rate", "Terminal growth", current_price=price),
               key=f"{KEY}dcf_sensitivity")


# ---------------------------------------------------------------------------
# Adjusted present value
# ---------------------------------------------------------------------------
def _apv_inputs(financial_data: Dict[str, Any], snapshot: Dict[str, Optional[float]],
                components: Dict[str, Any]) -> Dict[str, Any]:
    """Default APV inputs: tax rate, cost of debt and beta from the shared cost of capital; debt and cash."""
    income = get_statement(financial_data, "income_statement")
    return {"tax_rate": components["tax_rate"], "interest": abs(get_latest(income, "interest_expense") or 0.0),
            "debt": snapshot.get("total_debt") or 0.0, "cash": snapshot.get("cash") or 0.0,
            "cost_of_debt": components["pre_tax_cost_of_debt"], "beta": components["raw_beta"],
            "risk_free_rate": components["risk_free_rate"]}


def _render_apv(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                snapshot: Dict[str, Optional[float]]) -> None:
    from StockAnalysisSystem.src.valuation.apv import apv_valuation, unlevered_cost_of_equity, unlevered_fcf

    st.subheader("Adjusted present value (APV)")
    st.caption("Value as if the company had no debt (unlevered free cash flow at the unlevered cost of equity) "
               "plus the present value of interest tax shields. Useful when leverage is changing, e.g. LBOs and "
               "recapitalisations.")
    sector = canonical_sector(company_info, financial_data)
    if sector == "Financials":
        st.info("APV is not meaningful for banks and insurers: debt is part of operations. Use the Sector Model tab.")
        return
    fcf = free_cash_flow_series(get_statement(financial_data, "cash_flow"))
    shares, market_cap = snapshot.get("shares"), snapshot.get("market_cap")
    if fcf.empty or not shares:
        st.info("Free cash flow history or shares outstanding are unavailable, so APV cannot be computed.")
        return
    components = _shared_capital(financial_data, sector)
    inputs = _apv_inputs(financial_data, snapshot, components)
    defaults = _dcf_defaults(sector, fcf, components)
    ku_default = unlevered_cost_of_equity(inputs["beta"], inputs["debt"], market_cap or 0.0, inputs["tax_rate"],
                                          risk_free=inputs["risk_free_rate"])
    # Very low betas (e.g. integrated oil in some periods) give implausibly cheap equity: same band as the WACC
    ku_default = min(max(ku_default, capital.WACC_BOUNDS[0]), capital.WACC_BOUNDS[1])

    col1, col2, col3 = st.columns(3)
    with col1:
        growth = st.slider("UFCF growth", -10.0, 30.0, round(defaults["growth"] * 100, 1), 0.5, format="%.1f%%",
                           key=f"{KEY}apv_growth") / 100
        years = st.slider("Forecast years", 3, 15, defaults["forecast_years"], key=f"{KEY}apv_years")
    with col2:
        ku = st.slider("Unlevered cost of equity", 4.0, 20.0, round(ku_default * 100, 2), 0.25, format="%.2f%%",
                       key=f"{KEY}apv_ku", help="Risk-free rate + unlevered beta (Blume-adjusted, then Hamada) x "
                                                "equity risk premium.") / 100
        terminal = st.slider("Terminal growth", 0.0, 5.0, round(defaults["terminal_growth"] * 100, 2), 0.25,
                             format="%.2f%%", key=f"{KEY}apv_terminal") / 100
    with col3:
        kd = st.slider("Cost of debt", 1.0, 15.0, round(inputs["cost_of_debt"] * 100, 2), 0.25, format="%.2f%%",
                       key=f"{KEY}apv_kd", help="Default: interest expense / total debt.") / 100
        tax_rate = st.slider("Tax rate", 0.0, 35.0, round(inputs["tax_rate"] * 100, 1), 0.5, format="%.1f%%",
                             key=f"{KEY}apv_tax", help="Default: effective tax rate (same as the WACC).") / 100
    st.caption(f"Same inputs as the WACC: {_rate_caption(components)}.")
    _render_method_differences(components)

    base = unlevered_fcf(float(fcf.tail(3).mean()), inputs["interest"], tax_rate)
    if base <= 0:
        st.warning(f"Average unlevered free cash flow is not positive ({fmt_money(base)}), so APV implies no value.")
        return
    if ku <= terminal:
        st.warning("The unlevered cost of equity must be higher than terminal growth.")
        return
    result = apv_valuation(base, growth, years, terminal, ku, inputs["debt"], kd, tax_rate, inputs["cash"], shares)
    price = snapshot.get("price")
    value = result["value_per_share"]
    cols = st.columns(4)
    cols[0].metric("APV per share", fmt_price(value),
                   delta=fmt_pct(value / price - 1, signed=True) if price and value else None)
    cols[1].metric("Unlevered firm value", fmt_money(result["unlevered_value"]))
    cols[2].metric("PV of tax shields", fmt_money(result["tax_shield_value"]))
    cols[3].metric("Equity value", fmt_money(result["equity_value"]))
    st.caption(f"Base unlevered FCF {fmt_money(base)} (3-year average FCF + after-tax interest) · debt "
               f"{fmt_money(inputs['debt'])} · cash {fmt_money(inputs['cash'])}.")
    show_chart(donut_chart(["PV of forecast UFCF", "PV of terminal value", "PV of tax shields"],
                           [result["pv_forecast_fcf"], result["pv_terminal_value"], result["tax_shield_value"]],
                           "APV composition"), key=f"{KEY}apv_donut")


def _render_method_differences(components: Dict[str, Any]) -> None:
    """Why DCF, APV and the Sector Model give different values for one company (review R2)."""
    with st.expander("Why do DCF, APV and the Sector Model differ?"):
        st.markdown(
            f"All three use the **same cost of capital**: risk-free rate {components['risk_free_rate_label']}, "
            f"equity risk premium {components['equity_risk_premium']:.1%}, Blume-adjusted beta "
            f"{components['adjusted_beta']:.2f}, cost of equity {components['cost_of_equity']:.2%}, "
            f"WACC {components['wacc']:.2%}. They differ in what they discount:\n\n"
            "- **DCF tab**: reported free cash flow (latest year or a multi-year average) grown at one rate, "
            "then a terminal value, discounted at the WACC. No stock-based compensation deduction.\n"
            "- **APV tab**: unlevered free cash flow (3-year average FCF + after-tax interest) at the "
            "*unlevered* cost of equity, plus the present value of interest tax shields at the cost of debt.\n"
            "- **Sector Model tab**: the sector's own methods (for example SBC-adjusted DCF for technology, "
            "cycle-normalised DCF for industrials, residual income for banks) blended with multiples. Its "
            "weighted value is the Fair Value Summary.\n\n"
            "So the gap between them comes from the cash-flow definition, the starting year, growth and "
            "the method mix, not from the discount rate.")


# ---------------------------------------------------------------------------
# Precedent transactions
# ---------------------------------------------------------------------------
def _render_precedents(financial_data: Dict[str, Any], company_info: Dict[str, Any],
                       snapshot: Dict[str, Optional[float]]) -> None:
    from StockAnalysisSystem.src.models import precedent_transactions as pt

    st.subheader("Precedent transactions")
    st.caption("Value the company at the multiples acquirers paid in comparable deals (these include a control "
               "premium, so they usually exceed trading multiples). Enter deals from press releases, merger proxies "
               "or fairness opinions; the table is saved locally and shared by all companies.")
    sector = canonical_sector(company_info, financial_data)
    deals = pt.load_deals()
    edited = st.data_editor(
        deals if not deals.empty else pt.empty_deals(), num_rows="dynamic", width="stretch", hide_index=True,
        key=f"{KEY}deals_editor",
        column_config={"date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                       "sector": st.column_config.TextColumn("Sector", help=f"e.g. {sector}"),
                       "ev_billions": st.column_config.NumberColumn("EV ($B)", format="%.2f"),
                       "ev_revenue": st.column_config.NumberColumn("EV/Revenue", format="%.2f"),
                       "ev_ebitda": st.column_config.NumberColumn("EV/EBITDA", format="%.2f"),
                       "source": st.column_config.TextColumn("Source")})
    col_save, col_upload = st.columns([1, 3], vertical_alignment="bottom")
    with col_save:
        if st.button("Save deals", key=f"{KEY}deals_save", icon=":material/save:"):
            st.success(f"Saved {pt.save_deals(edited)} deals.")
            deals = pt.load_deals()
    with col_upload:
        upload = st.file_uploader("Or import a CSV with the same columns", type="csv", key=f"{KEY}deals_upload")
        if upload is not None and st.button("Import CSV", key=f"{KEY}deals_import"):
            imported = pt.clean_deals(pd.read_csv(upload))
            st.success(f"Imported {pt.save_deals(pd.concat([deals, imported], ignore_index=True))} deals.")
            deals = pt.load_deals()

    scope = st.radio("Deals to use", [f"{sector} only", "All sectors"], horizontal=True, key=f"{KEY}deals_scope")
    multiples = pt.deal_multiples(deals, sector if scope.endswith("only") else None)
    if not multiples["deals"]:
        st.info("No matching deals with multiples yet - add some above and click **Save deals**.")
    else:
        income = get_statement(financial_data, "income_statement")
        rows = pt.implied_values(multiples, get_latest(income, "revenue"), get_latest(income, "ebitda"),
                                 snapshot.get("net_debt") or 0.0, snapshot.get("shares"))
        if not rows:
            st.info("Revenue / EBITDA or shares are unavailable for this company.")
        else:
            price = snapshot.get("price")
            cols = st.columns(len(rows))
            for col, row in zip(cols, rows):
                value = row["value_per_share"]
                col.metric(f"{row['method']} ({row['multiple']:.1f}x, {row['deals']} deals)", fmt_price(value),
                           delta=fmt_pct(value / price - 1, signed=True) if price and value else None)
                if row["low"] is not None and row["high"] is not None:
                    col.caption(f"Interquartile range {fmt_price(row['low'])} - {fmt_price(row['high'])}")

    feed = pt.fmp_recent_deals()
    if not feed.empty:
        with st.expander("Recent M&A announcements (Financial Modeling Prep)"):
            st.dataframe(feed, width="stretch", hide_index=True,
                         column_config={"link": st.column_config.LinkColumn("Filing", display_text="Open")})
    elif not os.environ.get(pt.FMP_ENV):
        st.caption(f"Tip: add `{pt.FMP_ENV}` to `.env` to list recent M&A announcements here.")


# ---------------------------------------------------------------------------
# Relative valuation
# ---------------------------------------------------------------------------
def _render_relative(financial_data: Dict[str, Any], company_info: Dict[str, Any], price_data: pd.DataFrame,
                     snapshot: Dict[str, Optional[float]]) -> None:
    st.subheader("Relative valuation")
    sector = canonical_sector(company_info, financial_data)
    ratios = (get_company_ratios(financial_data, company_info, price_data) or {}).get("valuation", {})
    benchmarks, source = relative_benchmarks(sector, company_info.get("symbol"))
    excluded = irrelevant_multiples(sector)
    price = snapshot.get("price")
    keys = [k for k in MULTIPLES if to_float(ratios.get(k)) or to_float(benchmarks.get(k))]
    if not keys:
        st.info("Valuation multiples are unavailable for this company.")
        return

    show_chart(ratio_comparison_chart([MULTIPLES[k][0] for k in keys], [ratios.get(k) for k in keys],
                                      [benchmarks.get(k) for k in keys], f"Multiples vs {sector} benchmark",
                                      company_name=company_info.get("symbol", "Company")),
               key=f"{KEY}relative_chart")

    rows, implied_labels, implied_values = [], [], []
    for key in keys:
        label, metric = MULTIPLES[key]
        current, bench = to_float(ratios.get(key)), to_float(benchmarks.get(key))
        implied = None
        if price and current and current > 0 and bench:
            if key == "ev_ebitda":
                ev = snapshot.get("enterprise_value")
                shares, net_debt = snapshot.get("shares"), snapshot.get("net_debt") or 0.0
                if ev and shares:
                    implied = (ev * bench / current - net_debt) / shares
            else:
                implied = price * bench / current
        if implied is not None and implied > 0 and key not in excluded:
            implied_labels.append(f"At sector {label}")
            implied_values.append(implied)
        rows.append({"Multiple": label, "Company": f"{current:.1f}x" if current else "N/A",
                     "Sector": f"{bench:.1f}x" if bench else "N/A",
                     "Premium / discount": fmt_pct(current / bench - 1, signed=True) if current and bench else "—",
                     "Implied price": fmt_price(implied) if implied and implied > 0 else "—",
                     "Used": f"No: {excluded[key]}" if key in excluded else "Yes"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, key=f"{KEY}relative_table")
    st.caption(f"Sector multiples: {source}.")
    if implied_values:
        show_chart(value_comparison_chart(implied_labels, implied_values, price,
                                          "Price implied by sector multiples"), key=f"{KEY}relative_implied")
        st.caption(f"Average implied price (multiples marked Used): {fmt_price(float(np.mean(implied_values)))}. "
                   "A premium can be justified by faster growth, higher returns on capital or lower risk.")


def relative_benchmarks(sector: str, symbol: Optional[str] = None) -> tuple:
    """
    Sector multiples for the Relative tab: live medians of cached universe companies when at least
    LIVE_MIN_COMPANIES have a value (per multiple), else the static long-run medians. Returns
    ``(benchmarks, source description)``.
    """
    static = (get_sector_benchmarks(sector) or {}).get("valuation", {})
    live = live_valuation_medians(sector, exclude=symbol)
    if not live:
        return static, "static long-run sector medians (no cached universe data yet)"
    merged = {**static, **live["values"]}
    used = ", ".join(f"{MULTIPLES[k][0]} n={live['counts'][k]}" for k in MULTIPLES if k in live["values"])
    return merged, (f"live medians of {live['companies']} cached {sector} universe companies "
                    f"({used}; data as of {live['as_of']}); other multiples use static long-run medians")


# ---------------------------------------------------------------------------
# Asset-based valuation
# ---------------------------------------------------------------------------
def _render_asset_based(financial_data: Dict[str, Any], snapshot: Dict[str, Optional[float]]) -> None:
    st.subheader("Asset-based valuation")
    balance = get_statement(financial_data, "balance_sheet")
    equity = get_latest(balance, "total_equity")
    shares, price = snapshot.get("shares"), snapshot.get("price")
    if equity is None or not shares:
        st.info("Book equity or share count is unavailable.")
        return
    tangible = get_latest(balance, "tangible_book_value")
    if tangible is None:
        intangibles = get_latest(balance, "goodwill_and_intangibles") or get_latest(balance, "goodwill") or 0.0
        tangible = equity - intangibles
    receivables = get_latest(balance, "receivables") or 0.0
    inventory = get_latest(balance, "inventory") or 0.0
    ppe = get_latest(balance, "net_ppe") or 0.0
    haircut = st.slider("Liquidation haircut on receivables / inventory / PP&E", 0, 80, 30, 5, format="%d%%",
                        key=f"{KEY}asset_haircut") / 100
    liquidation = tangible - haircut * (receivables + inventory + ppe)

    values = {"Book value": equity / shares, "Tangible book value": tangible / shares,
              "Liquidation value": liquidation / shares}
    cols = st.columns(len(values))
    for col, (label, value) in zip(cols, values.items()):
        col.metric(f"{label} / share", fmt_price(value),
                   delta=f"P/x {price / value:.2f}" if price and value and value > 0 else None,
                   delta_color="off")
    show_chart(value_comparison_chart(list(values), list(values.values()), price, "Asset values per share"),
               key=f"{KEY}asset_chart")
    st.caption("Asset-based values are a floor for asset-heavy or distressed companies; for asset-light "
               "businesses most value lies in intangibles not recorded on the balance sheet.")
