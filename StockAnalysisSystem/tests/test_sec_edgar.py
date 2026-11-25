"""SEC EDGAR client and Overview section (offline, synthetic responses in the SEC JSON format)."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src.utils import sec_edgar as sec

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"}}


def _fy(end, val, filed, start=None, form="10-K", fp="FY"):
    entry = {"end": end, "val": val, "filed": filed, "form": form, "fp": fp}
    if start:
        entry["start"] = start
    return entry


FACTS = {"facts": {"us-gaap": {
    # Older years under one revenue tag, newer under another: first tag with data wins
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        _fy("2022-09-24", 394.0e9, "2022-10-28", start="2021-09-26"),
        _fy("2023-09-30", 383.0e9, "2023-11-03", start="2022-09-25"),
        _fy("2023-09-30", 383.3e9, "2024-11-01", start="2022-09-25"),  # restated later: keep latest filing
        _fy("2024-09-28", 391.0e9, "2024-11-01", start="2023-10-01"),
        _fy("2024-06-29", 85.8e9, "2024-08-02", start="2024-03-31", form="10-Q", fp="Q3"),  # quarterly: ignored
        _fy("2024-09-28", 94.9e9, "2024-11-01", start="2024-06-30"),  # 3-month period in a 10-K: ignored
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        _fy("2022-09-24", 99.8e9, "2022-10-28", start="2021-09-26"),
        _fy("2024-09-28", 93.7e9, "2024-11-01", start="2023-10-01"),
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [_fy("2024-09-28", 6.08, "2024-11-01", start="2023-10-01")]}},
    "Assets": {"units": {"USD": [_fy("2024-09-28", 365.0e9, "2024-11-01")]}},  # balance: no start date
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        _fy("2024-09-28", 118.3e9, "2024-11-01", start="2023-10-01")]}},
    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
        _fy("2024-09-28", 9.4e9, "2024-11-01", start="2023-10-01")]}},
}}}

SUBMISSIONS = {"filings": {"recent": {
    "form": ["8-K", "10-K", "4", "10-Q"],
    "filingDate": ["2024-11-15", "2024-11-01", "2024-11-02", "2024-08-02"],
    "reportDate": ["2024-11-15", "2024-09-28", "", "2024-06-29"],
    "primaryDocDescription": ["", "10-K", "", "10-Q"],
    "accessionNumber": ["0000320193-24-000130", "0000320193-24-000123", "0000320193-24-000125",
                        "0000320193-24-000081"],
    "primaryDocument": ["a8k.htm", "aapl-20240928.htm", "x.xml", "aapl-20240629.htm"],
}}}


@pytest.fixture
def edgar(monkeypatch):
    """Configured client whose network returns the synthetic responses; counts calls."""
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if url == sec.TICKERS_URL:
            return TICKERS
        if "submissions" in url:
            return SUBMISSIONS
        if "companyfacts" in url:
            return FACTS
        raise AssertionError(url)

    monkeypatch.setenv(sec.USER_AGENT_ENV, "Test Runner test@example.org")
    monkeypatch.setattr(sec.config, "OFFLINE_MODE", False)
    monkeypatch.setattr(sec, "_fetch_json", fake_fetch)
    return calls


def test_cik_lookup(edgar):
    assert sec.get_cik("aapl") == "0000320193"
    assert sec.get_cik("BRK.B") == "0001067983"  # dot share classes map to SEC's dash
    assert sec.get_cik("ZZZZQ") is None


def test_responses_are_cached(edgar):
    sec.get_cik("AAPL")
    sec.get_cik("AAPL")
    assert edgar.count(sec.TICKERS_URL) == 1


def test_recent_filings(edgar):
    filings = sec.get_recent_filings("AAPL")
    assert filings["form"].tolist() == ["8-K", "10-K", "10-Q"]  # form 4 filtered out, newest first
    assert filings.loc[1, "url"] == ("https://www.sec.gov/Archives/edgar/data/320193/"
                                     "000032019324000123/aapl-20240928.htm")
    assert filings.loc[0, "description"] == "8-K"  # empty description falls back to the form


def test_annual_history(edgar):
    history = sec.get_annual_history("AAPL")
    years = [c.year for c in history.columns]
    assert years == [2022, 2023, 2024]
    revenue = history.loc["Revenue"]
    assert revenue.iloc[1] == pytest.approx(383.3e9)  # restatement wins
    assert revenue.iloc[2] == pytest.approx(391.0e9)  # 3-month period ignored
    assert history.loc["Diluted EPS"].iloc[-1] == pytest.approx(6.08)
    assert history.loc["Free cash flow"].iloc[-1] == pytest.approx(108.9e9)
    assert "Total liabilities" not in history.index  # not reported -> no empty row


def test_cagr():
    series = pd.Series([100.0, 121.0], index=pd.to_datetime(["2022-12-31", "2024-12-31"]))
    assert sec.cagr(series) == pytest.approx(0.10, abs=1e-3)
    assert sec.cagr(pd.Series([-5.0, 10.0])) is None


def test_disabled_without_user_agent(monkeypatch):
    monkeypatch.delenv(sec.USER_AGENT_ENV, raising=False)
    monkeypatch.setattr(sec.config, "OFFLINE_MODE", False)
    monkeypatch.setattr(sec, "_fetch_json", lambda url: pytest.fail("network used without a User-Agent"))
    assert not sec.is_configured()
    assert sec.get_cik("AAPL") is None
    assert sec.get_recent_filings("AAPL").empty


def test_network_error_returns_empty(monkeypatch):
    monkeypatch.setenv(sec.USER_AGENT_ENV, "Test Runner test@example.org")
    monkeypatch.setattr(sec.config, "OFFLINE_MODE", False)

    def boom(url):
        raise ConnectionError("down")

    monkeypatch.setattr(sec, "_fetch_json", boom)
    assert sec.get_annual_history("AAPL").empty


def _overview_script():
    from StockAnalysisSystem.src.pages.company_analysis.overview import _render_sec_edgar
    _render_sec_edgar("AAPL")


def test_overview_section_hint_without_user_agent(monkeypatch):
    monkeypatch.delenv(sec.USER_AGENT_ENV, raising=False)
    at = AppTest.from_function(_overview_script).run()
    assert not at.exception
    assert any(sec.USER_AGENT_ENV in i.value for i in at.info)


def test_overview_section_loads_on_click(edgar):
    at = AppTest.from_function(_overview_script).run()
    assert not edgar  # nothing fetched before the click
    at.button(key="ov_edgar_AAPL").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 2
    assert any("CAGR" in m.value for m in at.markdown)
