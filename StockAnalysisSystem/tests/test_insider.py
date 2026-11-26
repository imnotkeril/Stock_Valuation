"""Insider transactions from Finnhub (SPEC AC22.3): optional key, never raises."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.utils import insider
from StockAnalysisSystem.src.utils import news_sentiment as ns

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")
TODAY = date(2026, 11, 26)
# Shape of a Finnhub stock/insider-transactions response (fields as documented by Finnhub).
PAYLOAD = {"symbol": "AAPL", "data": [
    {"name": "COOK TIMOTHY D", "share": 3280000, "change": -50000, "filingDate": "2026-10-03",
     "transactionDate": "2026-10-01", "transactionCode": "S", "transactionPrice": 250.0},
    {"name": "LEVINSON ARTHUR D", "share": 4200000, "change": 1000, "filingDate": "2026-11-12",
     "transactionDate": "2026-11-10", "transactionCode": "P", "transactionPrice": 240.0},
    {"name": "O'BRIEN DEIRDRE", "share": 140000, "change": 20000, "filingDate": "2026-10-16",
     "transactionDate": "2026-10-15", "transactionCode": "M", "transactionPrice": 0},
    {"name": "WILLIAMS JEFFREY E", "share": 490000, "change": -8000, "filingDate": "2026-04-03",
     "transactionDate": "2026-04-01", "transactionCode": "S", "transactionPrice": 170.0},
    {"name": "BROKEN ROW", "change": None, "transactionDate": "2026-11-01"},
]}


def test_parse_orders_newest_first_and_labels_codes():
    trades = insider.parse_transactions(PAYLOAD)
    assert trades["name"].tolist() == ["LEVINSON ARTHUR D", "O'BRIEN DEIRDRE", "COOK TIMOTHY D", "WILLIAMS JEFFREY E"]
    assert trades.loc[0, "type"] == "Buy (open market)" and trades.loc[0, "value"] == pytest.approx(240_000)
    assert trades.loc[1, "price"] is None or pd.isna(trades.loc[1, "price"])  # option exercise at $0
    assert trades.loc[2, "value"] == pytest.approx(-12_500_000)


def test_summary_counts_open_market_trades_in_window():
    s = insider.summarize(insider.parse_transactions(PAYLOAD), days=90, today=TODAY)
    # window 2026-08-28 .. 2026-11-26: P 1000 x 240 = 240,000; S 50,000 x 250 = 12,500,000; M ignored; April S outside
    assert (s["buys"], s["sells"], s["transactions"]) == (1, 1, 3)
    assert s["buy_value"] == pytest.approx(240_000) and s["sell_value"] == pytest.approx(12_500_000)
    assert s["net_value"] == pytest.approx(240_000 - 12_500_000)


@pytest.mark.parametrize("payload", [None, {}, {"data": "oops"}, {"error": "limit"}, []])
def test_unexpected_payloads_give_an_empty_table(payload):
    assert insider.parse_transactions(payload).empty
    assert insider.summarize(insider.parse_transactions(payload))["transactions"] == 0


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.delenv(insider.FINNHUB_ENV, raising=False)
    monkeypatch.setattr(ns, "_cached", lambda *a, **k: pytest.fail("must not request without a key"))
    assert not insider.is_configured()
    assert insider.get_insider_transactions("AAPL").empty


def test_key_never_in_params_cache_name_and_errors_are_swallowed(monkeypatch):
    monkeypatch.setenv(insider.FINNHUB_ENV, "test-key")
    seen = {}

    def fake_cached(name, url, params):
        seen.update(name=name, url=url, params=params)
        return PAYLOAD

    monkeypatch.setattr(ns, "_cached", fake_cached)
    assert len(insider.get_insider_transactions("aapl", today=TODAY)) == 4
    assert seen["url"].endswith("stock/insider-transactions") and seen["params"]["symbol"] == "AAPL"

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(ns, "_cached", boom)
    assert insider.get_insider_transactions("AAPL").empty


@pytest.mark.parametrize("with_key", [False, True])
def test_overview_section_renders(monkeypatch, with_key):
    if with_key:
        monkeypatch.setenv(insider.FINNHUB_ENV, "test-key")
        monkeypatch.setattr(ns, "_cached", lambda *a, **k: PAYLOAD)
    else:
        monkeypatch.delenv(insider.FINNHUB_ENV, raising=False)
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    if with_key:
        assert any(m.label == "Open-market sells (90 days)" for m in at.metric)
    else:
        assert any(insider.FINNHUB_ENV in i.value for i in at.info)
