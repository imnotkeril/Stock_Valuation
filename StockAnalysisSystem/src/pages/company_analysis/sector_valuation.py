"""
Sector-specific valuation UI.

The controls are generated from what the selected valuation model declares
(``SUBSECTORS``, ``get_available_methods``, ``get_parameters``), so every
sector model gets a complete interface without sector-specific UI code.

Public API::

    render_sector_valuation(ticker, financial_data, company_info, price_data,
                            key_prefix="sector_val") -> Optional[Dict]
"""

import logging
import numbers
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from StockAnalysisSystem.src.config import COLORS, VIZ_SETTINGS, normalize_sector
from StockAnalysisSystem.src.utils.financial_data import format_large_number, to_float
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

logger = logging.getLogger('sector_valuation')

#: Qualitative considerations shown next to the recommendation.
SECTOR_CONSIDERATIONS: Dict[str, List[str]] = {
    "Technology": [
        "Growth durability and reinvestment needs drive most of the value — small changes in long-term growth move the result a lot.",
        "Stock-based compensation is a real cost; check that it is reflected in free cash flow.",
        "Product cycles and platform shifts can make historical margins a poor guide.",
    ],
    "Healthcare": [
        "Patent cliffs and pipeline success probabilities dominate long-term value.",
        "Regulatory decisions (approvals, pricing reform) create binary outcomes.",
        "R&D is an investment — capitalising it gives a truer picture of returns on capital.",
    ],
    "Financials": [
        "Book value and sustainable ROE versus cost of equity are the core value drivers; EV/FCF is not meaningful for banks.",
        "Credit cycles and interest-rate moves can change earnings power quickly.",
        "Capital requirements cap payouts and growth.",
    ],
    "Consumer Discretionary": [
        "Demand is cyclical — normalise margins across the cycle.",
        "Same-store sales, inventory turnover and e-commerce mix are key operating signals.",
        "Brand strength and pricing power determine margin resilience.",
    ],
    "Consumer Staples": [
        "Stable demand supports lower discount rates and dividend-based methods.",
        "Input-cost inflation and private-label competition pressure margins.",
        "Volume vs. price growth mix indicates the quality of revenue growth.",
    ],
    "Energy": [
        "Commodity price volatility creates both risks and opportunities.",
        "Capital intensity requires focus on capital allocation efficiency.",
        "The energy transition presents long-term structural challenges.",
    ],
    "Utilities": [
        "Regulated returns provide stability but limit upside.",
        "Rate base growth is the key driver of long-term value.",
        "The energy transition creates both risks and opportunities.",
    ],
    "Industrials": [
        "Earnings are cyclical — value on normalised, mid-cycle margins.",
        "Order backlog and book-to-bill ratios lead revenue.",
        "Replacement cost of assets is a useful floor for asset-heavy businesses.",
    ],
    "Materials": [
        "Commodity prices drive earnings; avoid extrapolating peak or trough margins.",
        "Asset quality (cost curve position) determines resilience in downturns.",
        "High capital intensity makes balance-sheet strength critical.",
    ],
    "Real Estate": [
        "NAV and cap rates anchor value; FFO/AFFO replace EPS for REITs.",
        "Interest rates move cap rates and therefore asset values.",
        "Occupancy, lease duration and tenant quality drive cash-flow stability.",
    ],
    "Communication Services": [
        "Subscriber economics (ARPU, churn, acquisition cost) drive telecom and streaming value.",
        "Advertising revenue is cyclical; engagement metrics lead monetisation.",
        "Content and network investments are large, recurring capital needs.",
    ],
}

#: Subsector-specific overrides of the considerations above.
SUBSECTOR_CONSIDERATIONS: Dict[str, List[str]] = {
    "utilities": SECTOR_CONSIDERATIONS["Utilities"],
    "renewables": [
        "Policy support provides tailwinds but creates regulatory risk.",
        "Project development execution is a critical success factor.",
        "Technology innovation can drive both opportunities and obsolescence risk.",
    ],
    "midstream": [
        "Long-term, fee-based contracts reduce commodity exposure.",
        "Distribution coverage and leverage determine payout sustainability.",
        "Volume risk depends on the producers behind the contracts.",
    ],
}

#: Model class -> user-facing description
MODEL_LABELS: Dict[str, str] = {
    "BaseValuation": "General multi-method model",
    "FinancialSectorValuation": "Financial institutions model (banks, insurers, asset managers)",
    "TechnologySectorValuation": "Technology model (software, hardware, semiconductors, internet)",
    "EnergySectorValuation": "Energy & utilities model (oil & gas, utilities, renewables)",
    "RetailSectorValuation": "Consumer & retail model",
    "ManufacturingSectorValuation": "Industrials & materials model",
    "RealEstateValuation": "Real estate model (REITs, developers, services)",
    "HealthcareValuation": "Healthcare model (pharma, biotech, devices, services)",
    "CommunicationValuation": "Communication services model (telecom, media, entertainment)",
}

_METHOD_COLORS = [COLORS["info"], COLORS["secondary"], COLORS["success"], COLORS["warning"],
                  COLORS["accent"], COLORS["primary"], COLORS["danger"]]

_PERCENT_HINTS = ("rate", "growth", "margin", "yield", "roe", "roic", "roa", "wacc", "probability", "coverage",
                  "payout", "churn", "upside", "premium", "discount", "factor", "cost_of", "weight")
_MONEY_HINTS = ("value", "price", "cash", "debt", "revenue", "ebitda", "income", "nav", "ffo", "noi", "fcf",
                "equity", "assets", "book", "earnings", "dividend", "capex", "sales", "liabilities")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Number) and not isinstance(value, bool) and to_float(value) is not None


def _label(key: str) -> str:
    text = key.replace('_', ' ').strip()
    for short in ("dcf", "ev", "nav", "ffo", "affo", "roe", "roic", "wacc", "fcf", "ebitda", "pe", "pb", "ps",
                  "ddm", "arpu", "noi", "rnpv", "sotp", "rab", "ppa", "irr", "npv", "sbc", "eps"):
        text = " ".join(word.upper() if word.lower() == short else word for word in text.split())
    return text[:1].upper() + text[1:]


def format_value(key: str, value: Any) -> str:
    """Human-friendly formatting of a result field based on its name."""
    if not _is_number(value):
        return str(value)
    number = float(value)
    name = key.lower()
    if "per_share" in name or name.endswith("price") or name in {"value", "fair_value"}:
        return f"${number:,.2f}"
    if "multiple" in name or name.endswith("_ratio") or name.startswith(("pe", "pb", "ps", "ev_")):
        return f"{number:,.2f}x"
    if any(hint in name for hint in _PERCENT_HINTS) and abs(number) <= 2:
        return f"{number * 100:.2f}%"
    if any(hint in name for hint in _MONEY_HINTS) and abs(number) >= 1e4:
        return format_large_number(number)
    if name.endswith(("years", "_count", "life", "duration")):
        return f"{number:,.0f}"
    return f"{number:,.4g}" if abs(number) < 1e4 else f"{number:,.0f}"


def _scalar_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"method", "value_per_share", "error", "company", "ticker", "timestamp"}
    fields = {}
    for key, value in result.items():
        if key in skip:
            continue
        if _is_number(value) or (isinstance(value, str) and len(value) < 120):
            fields[key] = value
    return fields


def _series_fields(result: Dict[str, Any]) -> Dict[str, List[float]]:
    series = {}
    for key, value in result.items():
        if isinstance(value, (list, tuple)) and 1 < len(value) <= 40 and all(_is_number(v) for v in value):
            series[key] = [float(v) for v in value]
    return series


def _table_fields(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tables = {}
    for key, value in result.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            tables[key] = pd.DataFrame(value)
        elif isinstance(value, dict) and value and all(_is_number(v) for v in value.values()):
            tables[key] = pd.DataFrame({"Item": [_label(str(k)) for k in value],
                                        "Value": [format_value(str(k), v) for k, v in value.items()]})
    return tables


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _render_parameter(name: str, spec: Dict[str, Any], key: str, disabled: bool) -> Any:
    kind = spec.get("kind", "number")
    low, high = to_float(spec.get("min"), 0.0), to_float(spec.get("max"), 1.0)
    default = to_float(spec.get("default"))
    default = (low + high) / 2 if default is None else min(max(default, low), high)
    step = to_float(spec.get("step"))
    label, help_text = spec.get("label", _label(name)), spec.get("help")

    if kind == "percent":
        value = st.slider(label, min_value=round(low * 100, 4), max_value=round(high * 100, 4),
                          value=round(default * 100, 4), step=round((step or 0.0025) * 100, 4), format="%.2f%%",
                          help=help_text, key=key, disabled=disabled)
        return value / 100
    if kind == "int":
        return int(st.slider(label, min_value=int(low), max_value=int(high), value=int(round(default)),
                             step=int(step or 1), help=help_text, key=key, disabled=disabled))
    fmt = "$%.2f" if kind == "currency" else "%.2f"
    return float(st.slider(label, min_value=float(low), max_value=float(high), value=float(default),
                           step=float(step or (high - low) / 100 or 0.01), format=fmt, help=help_text, key=key,
                           disabled=disabled))


@st.cache_resource(show_spinner=False)
def _get_factory() -> ValuationFactory:
    return ValuationFactory()


def _fingerprint(financial_data: Dict[str, Any]) -> Tuple:
    """Cheap, stable identity of a financial data bundle (cached copies differ by id)."""
    parts: List[Any] = []
    for key in ("income_statement", "balance_sheet", "cash_flow"):
        df = (financial_data or {}).get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            parts.append((key, df.shape, str(df.columns[0]), float(pd.to_numeric(df.iloc[:, 0], errors="coerce").sum())))
        else:
            parts.append((key, None))
    market = (financial_data or {}).get("market_data") or {}
    parts.append(tuple(sorted((k, to_float(v)) for k, v in market.items())))
    return tuple(parts)


def _run_valuation(factory: ValuationFactory, ticker: str, sector: str, subsector: Optional[str],
                   financial_data: Dict[str, Any], methods: Tuple[str, ...],
                   overrides: Tuple[Tuple[str, Any], ...], memo_key: str) -> Dict[str, Any]:
    """Run (and memoise per session) a valuation for a given set of inputs."""
    memo = st.session_state.setdefault(memo_key, {})
    signature = (ticker, sector, subsector, methods, overrides, _fingerprint(financial_data))
    if signature not in memo:
        with st.spinner("Running sector valuation..."):
            memo.clear()
            memo[signature] = factory.get_company_valuation(
                ticker, sector=sector, subsector=subsector, financial_data=financial_data,
                methods=list(methods), **dict(overrides))
    return memo[signature]


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_headline(result: Dict[str, Any]) -> None:
    fair_value = to_float(result.get("value_per_share"))
    price = to_float(result.get("current_price"))
    upside = to_float(result.get("potential_upside"))
    recommendation = result.get("recommendation") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fair value / share", f"${fair_value:,.2f}" if fair_value else "N/A")
    col2.metric("Current price", f"${price:,.2f}" if price else "N/A")
    col3.metric("Upside / downside", f"{upside:+.1f}%" if upside is not None else "N/A",
                delta=f"{upside:+.1f}%" if upside is not None else None)
    with col4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Recommendation</div>"
            f"<div class='metric-value' style='color:{recommendation.get('color', '#9e9e9e')}'>"
            f"{recommendation.get('label', 'N/A')}</div></div>", unsafe_allow_html=True)
    if recommendation.get("rationale"):
        st.caption(recommendation["rationale"])


def _methods_table(result: Dict[str, Any]) -> pd.DataFrame:
    summary = result.get("valuation_summary") or {}
    weights = summary.get("weights") or {}
    excluded = summary.get("excluded") or {}
    price = to_float(result.get("current_price"))
    rows = []
    for key, method in (result.get("valuation_methods") or {}).items():
        value = to_float(method.get("value_per_share"))
        usable = key in weights
        rows.append({
            "Method": method.get("method") or _label(key),
            "Fair value / share": f"${value:,.2f}" if value is not None else "—",
            "Weight": f"{weights[key] * 100:.0f}%" if usable else "—",
            "Upside": f"{(value / price - 1) * 100:+.1f}%" if usable and price else "—",
            "Status": "Used" if usable else str(excluded.get(key) or method.get("error") or "Not applicable"),
        })
    return pd.DataFrame(rows)


def _methods_chart(result: Dict[str, Any]) -> Optional[go.Figure]:
    summary = result.get("valuation_summary") or {}
    weights = summary.get("weights") or {}
    methods = result.get("valuation_methods") or {}
    usable = [(methods[k].get("method") or _label(k), float(methods[k]["value_per_share"])) for k in weights]
    if not usable:
        return None
    labels, values = zip(*usable)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation="h",
        marker_color=[_METHOD_COLORS[i % len(_METHOD_COLORS)] for i in range(len(values))],
        text=[f"${v:,.2f}" for v in values], textposition="auto",
        hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
    ))
    price = to_float(result.get("current_price"))
    fair = to_float(result.get("value_per_share"))
    if price:
        fig.add_vline(x=price, line_dash="dash", line_color=COLORS["accent"],
                      annotation_text=f"Price ${price:,.2f}", annotation_position="top")
    if fair:
        fig.add_vline(x=fair, line_color=COLORS["primary"],
                      annotation_text=f"Fair value ${fair:,.2f}", annotation_position="bottom")
    fig.update_layout(
        title="Fair value by method", height=max(260, 70 * len(values) + 120),
        plot_bgcolor=VIZ_SETTINGS["background"], paper_bgcolor=VIZ_SETTINGS["background"],
        font=dict(color=VIZ_SETTINGS["text_color"]), margin=dict(l=10, r=10, t=60, b=40),
        xaxis=dict(title="Value per share ($)", gridcolor=VIZ_SETTINGS["grid_color"], tickprefix="$"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _series_chart(name: str, values: List[float]) -> go.Figure:
    money = any(h in name.lower() for h in _MONEY_HINTS + ("fcf", "flow", "forecast", "projection"))
    fig = go.Figure(go.Bar(x=[f"Y{i + 1}" for i in range(len(values))], y=values,
                           marker_color=COLORS["info"],
                           hovertemplate="%{x}: %{y:,.3s}<extra></extra>" if money else None))
    fig.update_layout(title=_label(name), height=260, margin=dict(l=10, r=10, t=40, b=30),
                      plot_bgcolor=VIZ_SETTINGS["background"], paper_bgcolor=VIZ_SETTINGS["background"],
                      font=dict(color=VIZ_SETTINGS["text_color"]),
                      yaxis=dict(gridcolor=VIZ_SETTINGS["grid_color"], tickformat="~s" if money else None))
    return fig


def _render_method_details(result: Dict[str, Any], key_prefix: str) -> None:
    for index, (key, method) in enumerate((result.get("valuation_methods") or {}).items()):
        title = method.get("method") or _label(key)
        value = to_float(method.get("value_per_share"))
        header = f"{title} — ${value:,.2f}/share" if value is not None else f"{title} — n/a"
        with st.expander(header, expanded=False):
            if method.get("error"):
                st.warning(str(method["error"]))
            fields = _scalar_fields(method)
            if fields:
                st.dataframe(pd.DataFrame({"Item": [_label(k) for k in fields],
                                           "Value": [format_value(k, v) for k, v in fields.items()]}),
                             hide_index=True, key=f"{key_prefix}_fields_{index}")
            for series_index, (name, values) in enumerate(_series_fields(method).items()):
                st.plotly_chart(_series_chart(name, values), key=f"{key_prefix}_series_{index}_{series_index}")
            for table_index, (name, table) in enumerate(_table_fields(method).items()):
                st.markdown(f"**{_label(name)}**")
                st.dataframe(table, hide_index=True, key=f"{key_prefix}_table_{index}_{table_index}")
            if not (fields or method.get("error")):
                st.caption("No additional details reported by this method.")


def _render_result(result: Dict[str, Any], sector: str, subsector: Optional[str], key_prefix: str) -> None:
    if result.get("error") and not result.get("valuation_methods"):
        st.error(f"Sector valuation could not be completed: {result['error']}")
        return

    _render_headline(result)

    fig = _methods_chart(result)
    if fig is not None:
        st.plotly_chart(fig, key=f"{key_prefix}_methods_chart")
    else:
        st.warning("None of the selected methods produced a usable value for this company. "
                   "Try other methods or adjust the assumptions.")

    st.markdown("##### Methods")
    st.dataframe(_methods_table(result), hide_index=True, key=f"{key_prefix}_methods_table")

    summary = result.get("valuation_summary") or {}
    if summary.get("min") is not None and len(summary.get("methods_used") or []) > 1:
        st.caption(f"Range across methods: ${summary['min']:,.2f} – ${summary['max']:,.2f} "
                   f"(simple average ${summary['simple_average']:,.2f}, median ${summary['median']:,.2f}).")

    st.markdown("##### Method details")
    _render_method_details(result, key_prefix)

    assumptions = {k: v for k, v in (result.get("assumptions") or {}).items() if _is_number(v) or isinstance(v, str)}
    if assumptions:
        with st.expander("Assumptions used", expanded=False):
            st.dataframe(pd.DataFrame({"Assumption": [_label(k) for k in assumptions],
                                       "Value": [format_value(k, v) for k, v in assumptions.items()]}),
                         hide_index=True, key=f"{key_prefix}_assumptions")

    notes = [str(n) for n in result.get("notes") or [] if n]
    considerations = SUBSECTOR_CONSIDERATIONS.get(subsector or "") or SECTOR_CONSIDERATIONS.get(sector, [])
    if notes or considerations:
        st.markdown("##### Sector-specific considerations")
        st.markdown("\n".join(f"- {line}" for line in considerations + notes))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_sector_valuation(ticker: str, financial_data: Dict[str, Any], company_info: Dict[str, Any],
                            price_data: pd.DataFrame, key_prefix: str = "sector_val",
                            factory: Optional[ValuationFactory] = None) -> Optional[Dict[str, Any]]:
    """
    Render the sector-specific valuation section and return the valuation result.

    Args:
        ticker: Company ticker.
        financial_data: Canonical financial data bundle.
        company_info: Normalised company info (sector in any spelling).
        price_data: Daily price history (used for context only).
        key_prefix: Prefix for widget keys (must be unique per placement).
        factory: Optional ValuationFactory (a cached one is used otherwise).
    """
    company_info = company_info or {}
    sector = normalize_sector(company_info.get("sector"))
    factory = factory or _get_factory()
    model = factory.get_valuation_model(sector, company_info.get("industry"))
    prefix = f"{key_prefix}_{ticker}"

    sector_label = sector if sector != "Unknown" else "an unclassified sector"
    model_name = model.__class__.__name__
    st.markdown(f"Valuation model: **{MODEL_LABELS.get(model_name, model_name)}** — applied to {sector_label}.")

    subsectors = dict(model.SUBSECTORS or {})
    subsector: Optional[str] = None
    controls_left, controls_right = st.columns(2)
    if subsectors:
        detected = factory.get_default_subsector(ticker, sector, financial_data, company_info)
        options = list(subsectors)
        with controls_left:
            subsector = st.selectbox(
                "Business model / subsector", options, index=options.index(detected) if detected in options else 0,
                format_func=lambda key: subsectors.get(key, _label(key)), key=f"{prefix}_subsector",
                help="Pre-selected from the company's industry and financial profile.")

    available = model.get_available_methods(subsector)
    with controls_right:
        selected = st.multiselect("Valuation methods", list(available), default=list(available),
                                  format_func=lambda key: available.get(key, _label(key)),
                                  key=f"{prefix}_{subsector}_methods")
    if not selected:
        st.info("Select at least one valuation method.")
        return None

    overrides: Dict[str, Any] = {}
    parameters = model.get_parameters(subsector)
    if parameters:
        with st.expander("Valuation assumptions", expanded=False):
            customize = st.toggle("Override model assumptions", value=False, key=f"{prefix}_{subsector}_customize",
                                  help="When off, the model derives assumptions from the company's own data.")
            columns = st.columns(2)
            for index, (name, spec) in enumerate(parameters.items()):
                with columns[index % 2]:
                    value = _render_parameter(name, spec, f"{prefix}_{subsector}_param_{name}", not customize)
                if customize:
                    overrides[name] = value

    try:
        result = _run_valuation(factory, ticker, sector, subsector, financial_data, tuple(selected),
                                tuple(sorted(overrides.items())), memo_key=f"{prefix}_memo")
    except Exception as e:  # the factory should not raise, but never break the page
        logger.exception(f"Sector valuation failed for {ticker}")
        st.error(f"Sector valuation failed: {e}")
        return None

    _render_result(result, sector, subsector, prefix)
    return result
