"""
Company financials tab: income statement, balance sheet, cash flow, ratios and growth.

All line items are looked up through ``utils.financial_data`` aliases, so the
page works with any provider spelling and degrades gracefully (friendly
messages) when statements or individual line items are missing — e.g. banks
without a current/non-current split or companies with negative earnings.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.pages.company_analysis.charts import (
    balance_structure_chart,
    benchmark_bar_chart,
    financial_bars_chart,
    growth_bar_chart,
    is_empty_figure,
    line_trend_chart,
    show_chart,
    signed_bar_chart,
    waterfall_chart,
)
from StockAnalysisSystem.src.pages.company_analysis.page_helpers import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    RATIO_CATEGORIES,
    annualized_growth,
    benchmark_table,
    canonical_sector,
    fmt_money,
    fmt_number,
    fmt_pct,
    fmt_price,
    fmt_ratio,
    fmt_shares,
    format_frame,
    format_ratio_value,
    free_cash_flow_series,
    get_company_ratios,
    get_sector_benchmarks,
    get_statement,
    growth_label,
    has_any_statement,
    is_number,
    item_series,
    line_item_frame,
    market_snapshot,
    ratio_meta,
    ratio_series,
    render_section,
    safe_call,
    series_to_labels,
    show_colored,
    yoy_growth,
)
from StockAnalysisSystem.src.utils.financial_data import get_series, latest_period, safe_divide, to_float

logger = logging.getLogger("company_financials")

KEY = "fin_"

INCOME_ITEMS: List[Tuple[str, Any]] = [
    ("Revenue", "revenue"), ("Cost of Revenue", "cost_of_revenue"), ("Gross Profit", "gross_profit"),
    ("Research & Development", "rd_expense"), ("SG&A", "sga_expense"), ("Operating Expenses", "operating_expense"),
    ("Operating Income", "operating_income"), ("EBITDA", "ebitda"), ("Net Interest Income", "net_interest_income"),
    ("Interest Expense", "interest_expense"), ("Pretax Income", "pretax_income"), ("Income Tax", "tax_provision"),
    ("Net Income", "net_income"), ("Diluted EPS", "diluted_eps"), ("Diluted Shares", "diluted_shares"),
]
BALANCE_ITEMS: List[Tuple[str, Any]] = [
    ("Cash & Equivalents", "cash"),
    ("Cash & Short-Term Investments", ["Cash Cash Equivalents And Short Term Investments",
                                      "Cash And Short Term Investments"]),
    ("Receivables", "receivables"), ("Inventory", "inventory"), ("Current Assets", "current_assets"),
    ("Net PP&E", "net_ppe"), ("Goodwill & Intangibles", "goodwill_and_intangibles"),
    ("Total Assets", "total_assets"), ("Accounts Payable", "payables"),
    ("Current Liabilities", "current_liabilities"), ("Short-Term Debt", "short_term_debt"),
    ("Long-Term Debt", "long_term_debt"), ("Total Debt", "total_debt"),
    ("Total Liabilities", "total_liabilities"), ("Retained Earnings", "retained_earnings"),
    ("Shareholders' Equity", "total_equity"), ("Working Capital", "working_capital"),
    ("Tangible Book Value", "tangible_book_value"), ("Shares Outstanding", "shares_outstanding"),
]
CASH_FLOW_ITEMS: List[Tuple[str, Any]] = [
    ("Operating Cash Flow", "operating_cash_flow"), ("Capital Expenditure", "capital_expenditure"),
    ("Free Cash Flow", "free_cash_flow"), ("Investing Cash Flow", "investing_cash_flow"),
    ("Financing Cash Flow", "financing_cash_flow"), ("Dividends Paid", "dividends_paid"),
    ("Share Repurchases", "share_repurchases"), ("Stock-Based Compensation", "stock_based_compensation"),
    ("Depreciation & Amortization", "cash_flow_depreciation"),
    ("Change in Working Capital", "change_in_working_capital"),
]

RATIO_INTERPRETATION: Dict[str, str] = {
    "valuation": "Lower multiples generally indicate a cheaper valuation; higher multiples mean the market prices "
                 "in more growth or quality. Compare with the sector to judge relative value.",
    "profitability": "Higher is better. Gross margin reflects pricing power, operating margin operating "
                     "efficiency, net margin overall profitability; ROE / ROA / ROIC measure returns on capital.",
    "liquidity": "Higher values mean stronger short-term health: a current ratio above 2 is comfortable, 1-2 "
                 "adequate and below 1 may signal liquidity pressure. Quick and cash ratios are stricter tests.",
    "leverage": "Lower debt ratios mean lower financial risk: D/E below 0.5 is conservative, 0.5-1.5 moderate and "
                "above 1.5 high. Interest coverage above ~5x indicates comfortable debt service.",
    "efficiency": "Higher turnover means assets, inventory and receivables are used more productively; fewer days "
                  "outstanding is better. Values vary strongly by industry.",
    "growth": "Consistent positive growth is preferable to volatile performance; judge growth in the context of "
              "company size, maturity and industry.",
}


def render_company_financials(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                              price_data: pd.DataFrame, **_: Any) -> None:
    """
    Render the financial statement analysis tab.

    Args:
        ticker: Company ticker symbol.
        financial_data: Canonical financial data bundle (see ``utils.financial_data``).
        company_info: Normalised company information dict.
        price_data: Daily OHLCV price history (used for market-based ratios).
    """
    financial_data = financial_data or {}
    company_info = company_info or {}
    if not has_any_statement(financial_data):
        st.warning(f"Financial statements are not available for {ticker}.")
        return

    income = get_statement(financial_data, "income_statement")
    columns = list(income.columns) or list(get_statement(financial_data, "balance_sheet").columns)
    currency = company_info.get("currency") or "USD"
    st.caption(f"Figures in {currency} as reported, {'quarterly' if growth_label(columns) == 'QoQ' else 'annual'} "
               "periods, most recent first.")

    tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios", "Growth"])
    sections: List[Tuple[str, Callable[..., None]]] = [
        ("Income statement", _render_income_statement), ("Balance sheet", _render_balance_sheet),
        ("Cash flow", _render_cash_flow), ("Ratios", _render_ratios), ("Growth", _render_growth),
    ]
    for tab, (label, renderer) in zip(tabs, sections):
        with tab:
            render_section(label.lower(), renderer, ticker, financial_data, company_info, price_data)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _value_at(df: pd.DataFrame, item: Any, column: Any) -> Optional[float]:
    """Value of ``item`` in a specific period column (None when missing)."""
    series = get_series(df, item, dropna=False)
    if series.empty or column not in series.index:
        return None
    return to_float(series[column])


def _format_raw_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Format every row of an as-reported statement with a unit-appropriate formatter."""
    frame = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.empty:
        return frame
    coverage = frame.notna().mean()
    if (coverage >= 0.3).any():
        frame = frame.loc[:, coverage >= 0.3]
    try:
        frame = frame[sorted(frame.columns, key=pd.Timestamp, reverse=True)]
    except (ValueError, TypeError):
        pass
    frame.columns = series_to_labels(pd.Series(index=frame.columns, dtype=float))
    formatters: Dict[str, Callable[[Any], str]] = {}
    for label in frame.index:
        name = str(label).lower()
        if "eps" in name or "per share" in name:
            formatters[label] = fmt_price
        elif "shares" in name or name.startswith("share ") or "number" in name:
            formatters[label] = fmt_shares
        elif "rate" in name:
            formatters[label] = fmt_pct
        else:
            formatters[label] = fmt_money
    return format_frame(frame, fmt_money, formatters)


def _raw_statement_expander(ticker: str, df: pd.DataFrame, title: str, file_stub: str) -> None:
    with st.expander(f"Full {title} (as reported)"):
        formatted = _format_raw_statement(df)
        if formatted.empty:
            st.info(f"The {title} contains no numeric data.")
            return
        st.dataframe(formatted)
        st.download_button(f"Download {title} (CSV)", df.to_csv().encode("utf-8"),
                           file_name=f"{ticker}_{file_stub}.csv", mime="text/csv", key=f"{KEY}{file_stub}_csv")


def _common_size(frame: pd.DataFrame, base_label: str, exclude: Sequence[str] = ()) -> pd.DataFrame:
    if frame.empty or base_label not in frame.index:
        return pd.DataFrame()
    base = frame.loc[base_label].replace(0, np.nan)
    rows = frame.drop(index=[r for r in exclude if r in frame.index])
    return rows.div(base, axis=1)


def _growth_frame(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Period-over-period growth per row of a most-recent-first frame."""
    if frame.empty or frame.shape[1] < 2:
        return pd.DataFrame()
    rows = frame.drop(index=[r for r in exclude if r in frame.index])
    chronological = rows[rows.columns[::-1]]
    previous = chronological.shift(1, axis=1)
    growth = (chronological - previous) / previous.abs().replace(0, np.nan)
    growth = growth.iloc[:, 1:]
    return growth[growth.columns[::-1]].dropna(how="all")


def _assess(value: Optional[float], good: float, ok: float, higher_is_better: bool = True,
            labels: Tuple[str, str, str] = ("Strong", "Adequate", "Weak")) -> Tuple[str, str]:
    """Three-level assessment of ``value`` against two thresholds."""
    if value is None:
        return "N/A", NEUTRAL
    if higher_is_better:
        if value >= good:
            return labels[0], POSITIVE
        return (labels[1], NEUTRAL) if value >= ok else (labels[2], NEGATIVE)
    if value <= good:
        return labels[0], POSITIVE
    return (labels[1], NEUTRAL) if value <= ok else (labels[2], NEGATIVE)


def _metric_with_status(container: Any, label: str, value: str, status: Tuple[str, str],
                        delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
    container.metric(label, value, delta=delta, help=help_text)
    text, color = status
    if text != "N/A":
        container.markdown(f"<span style='color:{color};font-size:0.9rem'>{text}</span>", unsafe_allow_html=True)


def _pp_delta(series: pd.Series) -> Optional[str]:
    """Change between the last two values of a fraction series, in percentage points."""
    clean = series.dropna()
    if len(clean) < 2:
        return None
    return f"{(clean.iloc[-1] - clean.iloc[-2]) * 100:+.1f} pp"


# ---------------------------------------------------------------------------
# Income statement
# ---------------------------------------------------------------------------
def _render_income_statement(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                             price_data: pd.DataFrame) -> None:
    st.subheader("Income Statement")
    income = get_statement(financial_data, "income_statement")
    if income.empty:
        st.info("The income statement is not available for this company.")
        return
    currency = company_info.get("currency") or "USD"
    frame = line_item_frame(income, INCOME_ITEMS)
    label = growth_label(income.columns)
    view = st.radio("Show", ["Values", "Common-size (% of revenue)", f"Growth ({label})"], horizontal=True,
                    key=f"{KEY}is_view")
    _render_statement_table(frame, view, "Revenue", per_share_rows=("Diluted EPS",), share_rows=("Diluted Shares",))

    revenue = item_series(income, "revenue")
    gross = item_series(income, "gross_profit")
    operating = item_series(income, "operating_income")
    ebitda = item_series(income, "ebitda")
    net_income = item_series(income, "net_income")
    margins = {
        "Gross margin": ratio_series(gross, revenue),
        "Operating margin": ratio_series(operating, revenue),
        "EBITDA margin": ratio_series(ebitda, revenue),
        "Net margin": ratio_series(net_income, revenue),
    }
    margins = {name: s for name, s in margins.items() if s.notna().any()}

    if margins:
        cols = st.columns(len(margins))
        for col, (name, series) in zip(cols, margins.items()):
            col.metric(name, fmt_pct(series.dropna().iloc[-1]), delta=_pp_delta(series),
                       help=f"Latest period; delta versus the prior period ({label}).")

    col1, col2 = st.columns(2)
    with col1:
        fig = financial_bars_chart({"Revenue": revenue, "Operating Income": operating, "Net Income": net_income},
                                   title="Revenue and earnings", line=("Net margin", margins.get("Net margin")),
                                   currency=currency)
        show_chart(fig, key=f"{KEY}is_revenue_chart")
    with col2:
        fig = line_trend_chart(margins, title="Margin trends", yaxis_title="Margin (% of revenue)",
                               tickformat=".0%")
        show_chart(fig, key=f"{KEY}is_margin_chart")

    period = latest_period(income)
    steps = _profit_waterfall_steps(income, period)
    fig = waterfall_chart(steps, title=f"From revenue to net income ({_period_text(period)})", currency=currency)
    if not is_empty_figure(fig):
        show_chart(fig, key=f"{KEY}is_waterfall")

    _raw_statement_expander(ticker, income, "income statement", "income_statement")


def _period_text(period: Any) -> str:
    try:
        return pd.Timestamp(period).strftime("%b %Y")
    except (ValueError, TypeError):
        return str(period)


def _profit_waterfall_steps(income: pd.DataFrame, period: Any) -> List[Tuple[str, Optional[float], str]]:
    revenue = _value_at(income, "revenue", period)
    if revenue is None:
        return []
    steps: List[Tuple[str, Optional[float], str]] = [("Revenue", revenue, "absolute")]
    base = revenue
    gross = _value_at(income, "gross_profit", period)
    if gross is not None and abs(gross - revenue) > 1e-9:
        steps += [("Cost of revenue", gross - base, "relative"), ("Gross profit", gross, "total")]
        base = gross
    for label, item, total_label in (("Operating expenses", "operating_income", "Operating income"),
                                     ("Interest & other", "pretax_income", "Pretax income"),
                                     ("Taxes & other", "net_income", "Net income")):
        value = _value_at(income, item, period)
        if value is not None:
            steps += [(label, value - base, "relative"), (total_label, value, "total")]
            base = value
    return steps


def _render_statement_table(frame: pd.DataFrame, view: str, base_label: str,
                            per_share_rows: Sequence[str] = (), share_rows: Sequence[str] = ()) -> None:
    if frame.empty:
        st.info("No standard line items were found in this statement.")
        return
    non_money = list(per_share_rows) + list(share_rows)
    if view.startswith("Common-size"):
        table = _common_size(frame, base_label, exclude=non_money)
        if table.empty:
            st.info(f"{base_label} is not reported, so a common-size view is not available.")
            return
        st.dataframe(format_frame(table, fmt_pct))
    elif view.startswith("Growth"):
        table = _growth_frame(frame, exclude=list(share_rows))
        if table.empty:
            st.info("At least two periods are needed to compute growth.")
            return
        st.dataframe(format_frame(table, lambda v: fmt_pct(v, signed=True)))
    else:
        formatters = {row: fmt_price for row in per_share_rows}
        formatters.update({row: fmt_shares for row in share_rows})
        st.dataframe(format_frame(frame, fmt_money, formatters))


# ---------------------------------------------------------------------------
# Balance sheet
# ---------------------------------------------------------------------------
def _render_balance_sheet(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                          price_data: pd.DataFrame) -> None:
    st.subheader("Balance Sheet")
    balance = get_statement(financial_data, "balance_sheet")
    if balance.empty:
        st.info("The balance sheet is not available for this company.")
        return
    currency = company_info.get("currency") or "USD"
    frame = line_item_frame(balance, BALANCE_ITEMS)
    view = st.radio("Show", ["Values", "Common-size (% of total assets)", f"Growth ({growth_label(balance.columns)})"],
                    horizontal=True, key=f"{KEY}bs_view")
    _render_statement_table(frame, view, "Total Assets", share_rows=("Shares Outstanding",))

    total_assets = item_series(balance, "total_assets")
    current_assets = item_series(balance, "current_assets")
    current_liabilities = item_series(balance, "current_liabilities")
    total_liabilities = item_series(balance, "total_liabilities")
    equity = item_series(balance, "total_equity")
    debt = item_series(balance, "total_debt")
    inventory = item_series(balance, "inventory")
    cash = item_series(balance, "cash")

    if not current_assets.empty and not total_assets.empty:
        assets = {"Current assets": current_assets, "Non-current assets": total_assets.sub(current_assets).dropna()}
    else:
        assets = {"Cash": cash, "Other assets": total_assets.sub(cash, fill_value=0).dropna()
                  if not cash.empty else total_assets}
    if not current_liabilities.empty and not total_liabilities.empty:
        funding = {"Current liabilities": current_liabilities,
                   "Non-current liabilities": total_liabilities.sub(current_liabilities).dropna(),
                   "Equity": equity}
    else:
        funding = {"Liabilities": total_liabilities, "Equity": equity}
    col1, col2 = st.columns(2)
    with col1:
        fig = balance_structure_chart(assets, funding, title="Balance sheet structure", currency=currency)
        show_chart(fig, key=f"{KEY}bs_structure")

    quick_assets = current_assets.sub(inventory, fill_value=0) if not current_assets.empty else current_assets
    position = {
        "Current ratio": ratio_series(current_assets, current_liabilities),
        "Quick ratio": ratio_series(quick_assets, current_liabilities),
        "Debt / equity": ratio_series(debt, equity.where(equity > 0)),
        "Debt / assets": ratio_series(debt, total_assets),
    }
    with col2:
        fig = line_trend_chart(position, title="Financial position ratios", yaxis_title="Ratio (x)",
                               tickformat=".2f", reference_lines=[(1.0, "1.0x")])
        show_chart(fig, key=f"{KEY}bs_ratios")

    latest = {name: (series.dropna().iloc[-1] if series.notna().any() else None) for name, series in position.items()}
    cols = st.columns(4)
    _metric_with_status(cols[0], "Current Ratio", fmt_ratio(latest["Current ratio"]),
                        _assess(latest["Current ratio"], 2.0, 1.0, labels=("Strong liquidity", "Adequate liquidity",
                                                                           "Potential liquidity pressure")),
                        help_text="Current assets / current liabilities.")
    _metric_with_status(cols[1], "Debt / Equity", fmt_ratio(latest["Debt / equity"]),
                        _assess(latest["Debt / equity"], 0.5, 1.5, higher_is_better=False,
                                labels=("Low leverage", "Moderate leverage", "High leverage")),
                        help_text="Total debt / shareholders' equity.")
    _metric_with_status(cols[2], "Debt / Assets", fmt_ratio(latest["Debt / assets"]),
                        _assess(latest["Debt / assets"], 0.3, 0.6, higher_is_better=False,
                                labels=("Low debt burden", "Moderate debt burden", "High debt burden")),
                        help_text="Total debt / total assets.")
    snap = market_snapshot(financial_data, company_info, price_data)
    bvps = safe_divide(equity.iloc[-1] if not equity.empty else None, snap["shares"])
    cols[3].metric("Book Value / Share", fmt_price(bvps),
                   delta=f"P/B {snap['price'] / bvps:.2f}x" if bvps and bvps > 0 and snap["price"] else None,
                   delta_color="off", help="Shareholders' equity divided by shares outstanding.")
    if current_assets.empty:
        st.caption("This company does not report a current / non-current split (typical for banks and insurers), "
                   "so liquidity ratios are not shown.")

    _raw_statement_expander(ticker, balance, "balance sheet", "balance_sheet")


# ---------------------------------------------------------------------------
# Cash flow
# ---------------------------------------------------------------------------
def _render_cash_flow(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                      price_data: pd.DataFrame) -> None:
    st.subheader("Cash Flow")
    cash_flow = get_statement(financial_data, "cash_flow")
    if cash_flow.empty:
        st.info("The cash flow statement is not available for this company.")
        return
    income = get_statement(financial_data, "income_statement")
    currency = company_info.get("currency") or "USD"

    frame = line_item_frame(cash_flow, CASH_FLOW_ITEMS)
    fcf = free_cash_flow_series(cash_flow)
    if "Free Cash Flow" not in frame.index and not fcf.empty and not frame.empty:
        computed = pd.Series(fcf.values, index=series_to_labels(fcf))
        frame.loc["Free Cash Flow (OCF - CapEx)"] = computed.reindex(frame.columns)
    if frame.empty:
        st.info("No standard line items were found in the cash flow statement.")
    else:
        st.dataframe(format_frame(frame, fmt_money))

    ocf = item_series(cash_flow, "operating_cash_flow")
    capex = item_series(cash_flow, "capital_expenditure")
    net_income = item_series(income, "net_income")
    revenue = item_series(income, "revenue")
    col1, col2 = st.columns(2)
    with col1:
        fig = financial_bars_chart({"Operating cash flow": ocf, "Capital expenditure": -capex.abs(),
                                    "Free cash flow": fcf}, title="Cash generation",
                                   line=("FCF margin", ratio_series(fcf, revenue)), currency=currency)
        show_chart(fig, key=f"{KEY}cf_generation")
    with col2:
        fig = financial_bars_chart({"Net income": net_income, "Operating cash flow": ocf, "Free cash flow": fcf},
                                   title="Earnings vs cash flow", currency=currency)
        show_chart(fig, key=f"{KEY}cf_vs_income")

    st.markdown("**Cash flow quality**")
    conversion = ratio_series(ocf, net_income.where(net_income > 0))
    fcf_margin = ratio_series(fcf, revenue)
    capex_intensity = ratio_series(capex.abs(), revenue)
    latest_conv = conversion.dropna().iloc[-1] if conversion.notna().any() else None
    cols = st.columns(4)
    _metric_with_status(cols[0], "OCF / Net Income", fmt_ratio(latest_conv),
                        _assess(latest_conv, 1.2, 0.8, labels=("High earnings quality", "Reasonable conversion",
                                                                "Weak cash conversion")),
                        help_text="Operating cash flow relative to net income; >1.2x indicates strong cash "
                                  "conversion, <0.8x may signal earnings-quality concerns.")
    latest_margin = fcf_margin.dropna().iloc[-1] if fcf_margin.notna().any() else None
    _metric_with_status(cols[1], "FCF Margin", fmt_pct(latest_margin),
                        _assess(latest_margin, 0.10, 0.0, labels=("Strong", "Positive", "Negative")),
                        delta=_pp_delta(fcf_margin), help_text="Free cash flow / revenue.")
    latest_capex = capex_intensity.dropna().iloc[-1] if capex_intensity.notna().any() else None
    cols[2].metric("CapEx / Revenue", fmt_pct(latest_capex), help="Capital intensity of the business.")
    snap = market_snapshot(financial_data, company_info, price_data)
    fcf_yield = safe_divide(fcf.iloc[-1] if not fcf.empty else None, snap["market_cap"])
    _metric_with_status(cols[3], "FCF Yield", fmt_pct(fcf_yield),
                        _assess(fcf_yield, 0.05, 0.02, labels=("Attractive", "Moderate", "Low")),
                        help_text="Latest free cash flow / market capitalisation.")

    _render_fcf_sustainability(fcf)
    _render_capital_allocation(cash_flow, ocf)
    _raw_statement_expander(ticker, cash_flow, "cash flow statement", "cash_flow")


def _render_fcf_sustainability(fcf: pd.Series) -> None:
    values = fcf.dropna()
    if len(values) < 2:
        return
    st.markdown("**Free cash flow sustainability**")
    positive_periods = int((values > 0).sum())
    share_positive = positive_periods / len(values)
    mean = float(values.mean())
    cv = float(values.std() / abs(mean)) if mean else None
    cols = st.columns(3)
    cols[0].metric("Positive FCF periods", f"{positive_periods} of {len(values)}")
    cols[1].metric("FCF variability (CV)", fmt_number(cv),
                   help="Coefficient of variation (std / |mean|): below 0.2 is stable, above 0.5 volatile.")
    cols[2].metric("Average FCF", fmt_money(mean))
    if share_positive > 0.75 and mean > 0:
        if cv is not None and cv < 0.3:
            st.success("Strong and stable free cash flow generation supports financial flexibility and "
                       "consistent shareholder returns.")
        else:
            st.info("Free cash flow is generally positive but somewhat volatile.")
    elif share_positive > 0.5 and mean > 0:
        st.info("Free cash flow is adequate but inconsistent, which may limit flexibility in downturns.")
    else:
        st.warning("Weak or negative free cash flow over several periods raises sustainability questions — "
                   "the company may depend on external financing.")


def _render_capital_allocation(cash_flow: pd.DataFrame, ocf: pd.Series) -> None:
    period = latest_period(cash_flow)
    ocf_value = _value_at(cash_flow, "operating_cash_flow", period)
    uses = [("Capital expenditure", "capital_expenditure"), ("Dividends", "dividends_paid"),
            ("Share repurchases", "share_repurchases")]
    rows = []
    for label, item in uses:
        value = _value_at(cash_flow, item, period)
        if value is None:
            continue
        rows.append({"Use of cash": label, "Amount": fmt_money(abs(value)),
                     "% of operating cash flow": fmt_pct(safe_divide(abs(value), ocf_value))
                     if ocf_value and ocf_value > 0 else "N/A"})
    if rows:
        st.markdown(f"**Capital allocation ({_period_text(period)})**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------
def ratio_history(financial_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Key ratios per period (chronological series) computed from period-end values.

    Returns:
        Numeric DataFrame indexed by ratio label with period labels as columns
        (most recent first). Empty when no ratio can be computed.
    """
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    revenue = item_series(income, "revenue")
    net_income = item_series(income, "net_income")
    equity = item_series(balance, "total_equity")
    assets = item_series(balance, "total_assets")
    interest = item_series(income, "interest_expense").abs()
    series = {
        "Gross margin": ratio_series(item_series(income, "gross_profit"), revenue),
        "Operating margin": ratio_series(item_series(income, "operating_income"), revenue),
        "Net margin": ratio_series(net_income, revenue),
        "Return on equity": ratio_series(net_income, equity.where(equity > 0)),
        "Return on assets": ratio_series(net_income, assets),
        "Asset turnover": ratio_series(revenue, assets),
        "Current ratio": ratio_series(item_series(balance, "current_assets"),
                                      item_series(balance, "current_liabilities")),
        "Debt / equity": ratio_series(item_series(balance, "total_debt"), equity.where(equity > 0)),
        "Interest coverage": ratio_series(item_series(income, "operating_income"), interest.where(interest > 0)),
    }
    series = {k: v for k, v in series.items() if v.notna().any()}
    if not series:
        return pd.DataFrame()
    frame = pd.DataFrame(series).T
    frame = frame[sorted(frame.columns, reverse=True)]
    frame.columns = series_to_labels(pd.Series(index=frame.columns, dtype=float))
    return frame.dropna(how="all")


_HISTORY_PCT_ROWS = {"Gross margin", "Operating margin", "Net margin", "Return on equity", "Return on assets"}


def _render_ratios(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                   price_data: pd.DataFrame) -> None:
    st.subheader("Financial Ratios")
    sector = canonical_sector(company_info, financial_data)
    ratios = get_company_ratios(financial_data, company_info, price_data)
    benchmarks = get_sector_benchmarks(sector)
    categories = [cat for cat in RATIO_CATEGORIES if ratios.get(cat)]
    if not categories:
        st.info("Not enough financial data to calculate ratios.")
        return

    category = st.radio("Category", categories, format_func=lambda c: RATIO_CATEGORIES[c], horizontal=True,
                        key=f"{KEY}ratio_category")
    keys = list(ratios.get(category, {}))
    st.caption(RATIO_INTERPRETATION.get(category, ""))
    table = benchmark_table(ratios, benchmarks, keys)
    if not table.empty:
        descriptions = {ratio_meta(k, category)["label"]: ratio_meta(k, category).get("description", "")
                        for k in keys}
        table["Description"] = table["Metric"].map(descriptions).fillna("")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.dataframe(table, hide_index=True)
    with col2:
        fig = benchmark_bar_chart(ratios, benchmarks, keys,
                                  f"{RATIO_CATEGORIES[category]} vs {sector if sector != 'Unknown' else 'market'}")
        if is_empty_figure(fig):
            st.info("No sector benchmarks are available for this category.")
        else:
            show_chart(fig, key=f"{KEY}ratio_benchmark_chart")

    history = ratio_history(financial_data)
    if not history.empty:
        st.markdown("**Ratio history**")
        formatters = {row: fmt_pct for row in history.index if row in _HISTORY_PCT_ROWS}
        st.dataframe(format_frame(history, fmt_ratio, formatters))
        options = list(history.index)
        default = [r for r in ("Operating margin", "Net margin", "Return on equity") if r in options] or options[:2]
        chosen = st.multiselect("Plot ratios", options, default=default, key=f"{KEY}ratio_trend")
        pct = [r for r in chosen if r in _HISTORY_PCT_ROWS]
        other = [r for r in chosen if r not in _HISTORY_PCT_ROWS]
        chronological = history[history.columns[::-1]]
        cols = st.columns(2)
        if pct:
            with cols[0]:
                fig = line_trend_chart({r: chronological.loc[r] for r in pct}, title="Margins & returns",
                                       tickformat=".0%")
                show_chart(fig, key=f"{KEY}ratio_trend_pct")
        if other:
            with cols[1]:
                fig = line_trend_chart({r: chronological.loc[r] for r in other}, title="Multiples & coverage",
                                       yaxis_title="Ratio (x)", tickformat=".2f")
                show_chart(fig, key=f"{KEY}ratio_trend_x")

    _render_sector_key_ratios(sector, ratios)


def _render_sector_key_ratios(sector: str, ratios: Dict[str, Dict[str, float]]) -> None:
    if sector == "Unknown":
        return
    from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

    key_ratios = safe_call(lambda: FinancialRatioAnalyzer().get_key_ratios_for_sector(sector), default=[],
                           label="get_key_ratios_for_sector")
    rows = []
    for entry in key_ratios if isinstance(key_ratios, list) else []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        category = str(entry.get("category") or "").lower()
        value = ratios.get(category, {}).get(key) if key else None
        rows.append({"Ratio": entry.get("ratio", ""), "Category": entry.get("category", ""),
                     "Company": format_ratio_value(key, value) if value is not None else
                     ("Company-reported KPI" if key is None else "N/A"),
                     "Why it matters": entry.get("description", "")})
    if rows:
        st.markdown(f"**Key ratios for the {sector} sector**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------
def growth_metric_series(financial_data: Dict[str, Any], company_info: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, pd.Series]:
    """Chronological series of the metrics used in the growth analysis."""
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    cash_flow = get_statement(financial_data, "cash_flow")
    metrics = {
        "Revenue": item_series(income, "revenue"),
        "Gross Profit": item_series(income, "gross_profit"),
        "Operating Income": item_series(income, "operating_income"),
        "EBITDA": item_series(income, "ebitda"),
        "Net Income": item_series(income, "net_income"),
        "Diluted EPS": item_series(income, "diluted_eps"),
        "Operating Cash Flow": item_series(cash_flow, "operating_cash_flow"),
        "Free Cash Flow": free_cash_flow_series(cash_flow),
        "Total Assets": item_series(balance, "total_assets"),
        "Shareholders' Equity": item_series(balance, "total_equity"),
    }
    return {name: s for name, s in metrics.items() if len(s.dropna()) >= 1}


def per_share_series(financial_data: Dict[str, Any], company_info: Optional[Dict[str, Any]] = None,
                     price_data: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, pd.Series], bool]:
    """
    Per-share metrics per period.

    Returns:
        ``({name: chronological series}, uses_current_share_count)``.
    """
    income = get_statement(financial_data, "income_statement")
    balance = get_statement(financial_data, "balance_sheet")
    cash_flow = get_statement(financial_data, "cash_flow")
    shares = item_series(income, "diluted_shares")
    if shares.empty:
        shares = item_series(balance, "shares_outstanding")
    constant = False
    if shares.empty:
        current = market_snapshot(financial_data, company_info, price_data)["shares"]
        index = list(income.columns) or list(balance.columns) or list(cash_flow.columns)
        if not current or not index:
            return {}, False
        shares = pd.Series(current, index=sorted(index), dtype=float)
        constant = True
    shares = shares.where(shares > 0)
    eps = item_series(income, "diluted_eps")
    if eps.empty:
        eps = ratio_series(item_series(income, "net_income"), shares)
    result = {
        "EPS (diluted)": eps,
        "Revenue / share": ratio_series(item_series(income, "revenue"), shares),
        "Book value / share": ratio_series(item_series(balance, "total_equity"),
                                           item_series(balance, "shares_outstanding")
                                           if not item_series(balance, "shares_outstanding").empty else shares),
        "FCF / share": ratio_series(free_cash_flow_series(cash_flow), shares),
    }
    return {k: v.dropna() for k, v in result.items() if v.notna().any()}, constant


def _render_growth(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                   price_data: pd.DataFrame) -> None:
    st.subheader("Growth Analysis")
    metrics = growth_metric_series(financial_data, company_info)
    multi_period = {name: s for name, s in metrics.items() if len(s) >= 2}
    if not multi_period:
        st.info("At least two reporting periods are needed for growth analysis.")
        return
    any_columns = next(iter(multi_period.values())).index
    label = growth_label(any_columns)

    options = list(multi_period)
    default = [m for m in ("Revenue", "Net Income", "Free Cash Flow") if m in options] or options[:2]
    chosen = st.multiselect("Metrics", options, default=default, key=f"{KEY}growth_metrics")
    growth = {name: yoy_growth(multi_period[name]) for name in chosen}
    fig = growth_bar_chart(growth, title=f"{label} growth")
    show_chart(fig, key=f"{KEY}growth_chart")

    table = pd.DataFrame({name: yoy_growth(series) for name, series in multi_period.items()}).T
    table = table.dropna(how="all")
    if not table.empty:
        table = table[sorted(table.columns, reverse=True)]
        table.columns = series_to_labels(pd.Series(index=table.columns, dtype=float))
        st.markdown(f"**{label} growth by metric**")
        st.dataframe(format_frame(table.dropna(axis=1, how="all"), lambda v: fmt_pct(v, signed=True)))

    cagr_rows, cagr_values = [], {}
    for name, series in multi_period.items():
        value = annualized_growth(series)
        cagr_values[name] = value
        cagr_rows.append({
            "Metric": name,
            "CAGR": fmt_pct(value, signed=True) if value is not None else "n/m",
            "From": f"{series_to_labels(series)[0]}: {_format_metric(name, series.iloc[0])}",
            "To": f"{series_to_labels(series)[-1]}: {_format_metric(name, series.iloc[-1])}",
        })
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("**Compound annual growth rate (CAGR)**")
        st.dataframe(pd.DataFrame(cagr_rows), hide_index=True)
        st.caption("CAGR smooths year-to-year volatility; 'n/m' = not meaningful (negative or zero endpoints).")
    with col2:
        fig = signed_bar_chart(list(cagr_values), list(cagr_values.values()), title="CAGR by metric",
                               tickformat=".1%", axis_title="Compound annual growth")
        show_chart(fig, key=f"{KEY}cagr_chart")

    _render_growth_insights(cagr_values)
    _render_per_share(financial_data, company_info, price_data)


def _format_metric(name: str, value: Any) -> str:
    return fmt_price(value) if "EPS" in name else fmt_money(value)


def _render_growth_insights(cagr: Dict[str, Optional[float]]) -> None:
    insights: List[Tuple[str, bool]] = []
    revenue, net_income, assets = cagr.get("Revenue"), cagr.get("Net Income"), cagr.get("Total Assets")
    if is_number(revenue) and is_number(net_income):
        if net_income > revenue:
            insights.append(("Net income is growing faster than revenue — margins are expanding, a positive sign "
                             "for sustainable growth.", True))
        else:
            insights.append(("Revenue is growing faster than net income — margins are compressing or costs are "
                             "rising, which may make profitable scaling harder.", False))
    if is_number(revenue) and is_number(assets):
        if revenue > assets:
            insights.append(("Revenue is growing faster than assets — asset utilisation is improving.", True))
        else:
            insights.append(("Assets are growing faster than revenue — asset efficiency is declining or the "
                             "company is investing ahead of growth.", False))
    fcf, ni = cagr.get("Free Cash Flow"), cagr.get("Net Income")
    if is_number(fcf) and is_number(ni):
        insights.append((("Free cash flow is keeping pace with earnings." if fcf >= ni else
                          "Free cash flow is growing more slowly than earnings — check working capital and "
                          "capital spending."), fcf >= ni))
    if insights:
        st.markdown("**Growth sustainability**")
        for text, positive in insights:
            show_colored(("▲ " if positive else "▼ ") + text, POSITIVE if positive else NEUTRAL)


def _render_per_share(financial_data: Dict[str, Any], company_info: Dict[str, Any],
                      price_data: pd.DataFrame) -> None:
    per_share, constant = per_share_series(financial_data, company_info, price_data)
    if not per_share:
        return
    st.markdown("**Per-share metrics**")
    frame = pd.DataFrame(per_share).T
    frame = frame[sorted(frame.columns, reverse=True)]
    frame.columns = series_to_labels(pd.Series(index=frame.columns, dtype=float))
    st.dataframe(format_frame(frame, fmt_price))
    if constant:
        st.caption("Historical share counts are not reported; the current share count is used for all periods.")
    fig = line_trend_chart(per_share, title="Per-share metrics", yaxis_title="USD per share", tickformat="$,.2f")
    show_chart(fig, key=f"{KEY}per_share_chart")


__all__ = ["render_company_financials", "ratio_history", "growth_metric_series", "per_share_series"]
