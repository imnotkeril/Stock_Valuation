"""Precedent transactions (offline, user-entered illustrative deals)."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.models import precedent_transactions as pt

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")
DEALS = pd.DataFrame([
    {"date": "2021-05-01", "acquirer": "A", "target": "T1", "sector": "Technology", "ev_revenue": 6.0, "ev_ebitda": 20.0},
    {"date": "2022-07-01", "acquirer": "B", "target": "T2", "sector": "Technology", "ev_revenue": 8.0, "ev_ebitda": 24.0},
    {"date": "2023-03-01", "acquirer": "C", "target": "T3", "sector": "Technology", "ev_revenue": 10.0, "ev_ebitda": None},
    {"date": "2023-09-01", "acquirer": "D", "target": "T4", "sector": "Energy", "ev_revenue": 1.5, "ev_ebitda": 6.0},
    {"date": "2024-01-01", "acquirer": "E", "target": "", "sector": "Energy", "ev_revenue": 2.0},   # no target
    {"date": "2024-02-01", "acquirer": "F", "target": "T6", "sector": "Energy", "ev_revenue": -1},   # bad multiple
])


def test_clean_save_load_roundtrip():
    assert pt.load_deals().empty
    assert pt.save_deals(DEALS) == 4
    loaded = pt.load_deals()
    assert loaded["target"].tolist() == ["T1", "T2", "T3", "T4"]
    assert list(loaded.columns) == pt.COLUMNS


def test_deal_multiples_by_sector_and_year():
    tech = pt.deal_multiples(DEALS, "technology")
    assert tech["deals"] == 3
    assert tech["ev_revenue"]["median"] == pytest.approx(8.0)
    assert tech["ev_ebitda"]["median"] == pytest.approx(22.0) and tech["ev_ebitda"]["count"] == 2
    assert pt.deal_multiples(DEALS, "Technology", since_year=2022)["deals"] == 2
    assert pt.deal_multiples(DEALS)["deals"] == 4


def test_implied_values():
    rows = pt.implied_values(pt.deal_multiples(DEALS, "Technology"), revenue=100, ebitda=30, net_debt=50, shares=10)
    by = {r["method"]: r for r in rows}
    assert by["Deal EV/Revenue"]["value_per_share"] == pytest.approx((800 - 50) / 10)
    assert by["Deal EV/EBITDA"]["value_per_share"] == pytest.approx((30 * 22 - 50) / 10)
    assert by["Deal EV/Revenue"]["low"] < by["Deal EV/Revenue"]["high"]
    assert pt.implied_values(pt.deal_multiples(DEALS, "Technology"), None, -5, 0, 10) == []


def test_fmp_feed_needs_key(monkeypatch):
    monkeypatch.delenv(pt.FMP_ENV, raising=False)
    assert pt.fmp_recent_deals().empty


def test_fmp_feed_retries_with_free_plan_limit(monkeypatch):
    monkeypatch.setenv(pt.FMP_ENV, "test-key")
    limits = []

    class Resp:
        def __init__(self, status, data):
            self.status_code, self._data = status, data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self):
            return self._data

    def fake_get(url, params, timeout):
        limits.append(params["limit"])
        if params["limit"] > pt.FMP_FREE_LIMIT:
            return Resp(402, "Premium Query Parameter")
        return Resp(200, [{"transactionDate": "2026-11-20", "companyName": "Buyer Inc",
                           "targetedCompanyName": "Target Inc", "link": "https://example.com"}])

    monkeypatch.setattr(pt.requests, "get", fake_get)
    feed = pt.fmp_recent_deals()
    assert limits == [50, pt.FMP_FREE_LIMIT]
    assert feed.iloc[0]["target"] == "Target Inc"


def test_precedents_tab_renders():
    pt.save_deals(DEALS)
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Deal EV/Revenue (8.0x, 3 deals)") for label in labels), labels


def test_sample_deals_are_labelled_and_sector_specific():
    sample = pt.sample_deals("Technology")
    assert len(sample) == 5 and pt.is_sample(sample)
    assert (sample["source"] == pt.SAMPLE_SOURCE).all()
    assert sample["target"].str.startswith("Example Target").all()
    # median of the spread (0.75, 0.9, 1.0, 1.1, 1.3) is the sector level itself
    stats = pt.deal_multiples(sample, "Technology")
    assert stats["ev_revenue"]["median"] == pytest.approx(6.0)
    assert stats["ev_ebitda"]["median"] == pytest.approx(20.0)
    assert not pt.is_sample(DEALS)


def test_empty_tab_offers_sample_and_states_weight_status():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert any("Not part of the weighted fair value" in i.value for i in at.info)
    at.button(key="val_deals_sample_load").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("illustrative sample" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Deal EV/Revenue (6.0x, 5 deals)") for label in labels), labels
    assert pt.load_deals().empty, "sample must not be saved without the Save button"
