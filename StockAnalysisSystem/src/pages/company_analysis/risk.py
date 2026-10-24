import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.pages.common import cached_financial_data
from StockAnalysisSystem.src.utils.helpers import has_statements, normalize_sector


def render(ticker: str) -> None:
    st.header(f"Bankruptcy risk: {ticker}")
    data = cached_financial_data(ticker)
    if not has_statements(data):
        st.error("Financial statements are not available for this ticker.")
        return

    sector = normalize_sector(data["company_info"].get("sector"))
    assessment = BankruptcyAnalyzer().get_comprehensive_risk_assessment(data, sector)
    if assessment.get("error"):
        st.error(assessment["error"])
        return

    st.metric("Overall assessment", str(assessment.get("overall_assessment", "n/a")).upper())
    st.write(assessment.get("overall_description", ""))

    models = assessment.get("models", {})
    rows = [{"model": name, "score": m.get("score"), "assessment": m.get("assessment", m.get("zone", ""))}
            for name, m in models.items() if isinstance(m, dict)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    factors = assessment.get("risk_factors") or []
    if factors:
        st.subheader("Risk factors")
        for factor in factors:
            st.write(f"- **{factor.get('factor', '')}**: {factor.get('description', '')}")

    actions = assessment.get("recommended_actions") or []
    if actions:
        st.subheader("Suggested follow-ups")
        for action in actions:
            st.write(f"- {action}")
