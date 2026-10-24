"""Shared Streamlit plumbing: cached loader and cached data fetches."""
import streamlit as st

from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.helpers import load_financial_data


@st.cache_resource
def get_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_financial_data(ticker: str):
    return load_financial_data(get_loader(), ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str, start: str, end: str):
    return get_loader().get_historical_prices(ticker, start, end)


def clean_ticker(raw: str) -> str:
    return "".join(ch for ch in raw.strip().upper() if ch.isalnum() or ch in ".-^")[:12]
