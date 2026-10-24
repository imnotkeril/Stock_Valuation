import plotly.express as px
import streamlit as st

from StockAnalysisSystem.src.models.financial_statements import FinancialStatementsAnalyzer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.pages.common import cached_financial_data, get_loader
from StockAnalysisSystem.src.utils.helpers import flatten_ratios, has_statements


def render(ticker: str) -> None:
    st.header(f"Financials: {ticker}")
    data = cached_financial_data(ticker)
    if not has_statements(data):
        st.error("Financial statements are not available for this ticker.")
        return

    tab_stmt, tab_ratio, tab_trend = st.tabs(["Statements", "Ratios", "Trends (YoY)"])

    with tab_stmt:
        choice = st.radio("Statement", ["Income", "Balance sheet", "Cash flow"], horizontal=True)
        key = {"Income": "income_statement", "Balance sheet": "balance_sheet", "Cash flow": "cash_flow"}[choice]
        st.dataframe(data[key])

    with tab_ratio:
        ratios = FinancialRatioAnalyzer().calculate_ratios(data)
        table = flatten_ratios(ratios)
        if table.empty:
            st.info("No ratios could be calculated.")
        else:
            st.dataframe(table, hide_index=True)
            category = st.selectbox("Chart category", sorted(table["category"].unique()))
            subset = table[table["category"] == category]
            st.plotly_chart(px.bar(subset, x="ratio", y="value"))

    with tab_trend:
        analyzer = FinancialStatementsAnalyzer(get_loader())
        which = st.radio("Statement ", ["Income", "Balance sheet", "Cash flow"], horizontal=True, key="trend_stmt")
        key = {"Income": "income_statement", "Balance sheet": "balance_sheet", "Cash flow": "cash_flow"}[which]
        trend = analyzer.perform_horizontal_analysis(data[key])
        st.caption("Year-over-year change per line item")
        st.dataframe(trend)
