import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from StockAnalysisSystem.src.pages.common import cached_financial_data, get_loader
from StockAnalysisSystem.src.utils.helpers import has_statements, normalize_sector
from StockAnalysisSystem.src.valuation import sector_specific  # noqa: F401  (registers sector analyzers)
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation
from StockAnalysisSystem.src.valuation.sector_factor import SectorFactory

METHODS = {
    "Multi-stage DCF": "multi_stage_dcf_valuation",
    "Forecast-driven DCF": "forecast_driven_dcf",
    "Monte Carlo DCF": "monte_carlo_dcf",
    "Scenario analysis DCF": "scenario_analysis_dcf",
    "Dividend discount model": "dividend_discount_valuation",
}


def _per_share(result: dict):
    return (result.get("value_per_share") or result.get("median_value_per_share")
            or result.get("mean_value_per_share") or result.get("conservative_value"))


def render(ticker: str) -> None:
    st.header(f"Valuation: {ticker}")
    data = cached_financial_data(ticker)
    if not has_statements(data):
        st.error("Financial statements are not available for this ticker.")
        return

    info = data["company_info"]
    sector = normalize_sector(info.get("sector"))
    price = info.get("current_price")
    st.caption(f"Sector: {sector or 'n/a'}  |  current price: {price if price else 'n/a'}")

    chosen = st.multiselect("Methods", list(METHODS), default=["Multi-stage DCF", "Forecast-driven DCF"])
    if not chosen:
        st.info("Select at least one method.")
        return

    dcf = AdvancedDCFValuation(get_loader())
    rows, details = [], {}
    for label in chosen:
        result = getattr(dcf, METHODS[label])(ticker, data, sector)
        details[label] = result
        value = _per_share(result)
        rows.append({
            "method": label,
            "value / share": round(float(value), 2) if value else None,
            "upside vs price": f"{(value / price - 1) * 100:+.1f}%" if value and price else "n/a",
            "note": result.get("error", ""),
        })

    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True)

    valid = table.dropna(subset=["value / share"])
    if not valid.empty:
        fig = go.Figure(go.Bar(x=valid["method"], y=valid["value / share"], name="Intrinsic value"))
        if price:
            fig.add_hline(y=price, line_dash="dash", annotation_text="current price")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)

    for label, result in details.items():
        if result.get("error"):
            continue
        with st.expander(f"{label}: details"):
            scalars = {k: v for k, v in result.items() if isinstance(v, (int, float, str)) and v is not None}
            st.dataframe(pd.DataFrame({"value": pd.Series(scalars).astype(str)}))

    analyzer = SectorFactory.get_analyzer(sector) if sector else None
    if analyzer is not None:
        with st.expander(f"Sector analysis ({sector})"):
            try:
                st.json(analyzer.analyze_company(data), expanded=False)
            except Exception as exc:  # sector models are heuristic; never break the page
                st.warning(f"Sector analysis unavailable: {exc}")

    st.caption("Model output for research only. Not investment advice.")
