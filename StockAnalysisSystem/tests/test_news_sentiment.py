"""News and sentiment (offline, synthetic Finnhub / Alpha Vantage responses)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src.utils import news_sentiment as ns

NOW = datetime.now(timezone.utc)
FINNHUB_NEWS = [
    {"datetime": int((NOW - timedelta(days=1)).timestamp()), "headline": "Apple beats estimates, raises dividend",
     "summary": "", "source": "Reuters", "url": "https://example.org/1"},
    {"datetime": int((NOW - timedelta(days=2)).timestamp()), "headline": "Apple faces EU probe over App Store fees",
     "summary": "Regulators warn of a fine.", "source": "FT", "url": "https://example.org/2"},
    {"datetime": int((NOW - timedelta(days=2)).timestamp()), "headline": "Apple shares rally to record high",
     "summary": "", "source": "CNBC", "url": "https://example.org/3"},
]
AV_NEWS = {"feed": [
    {"title": "Apple shares rally to record high", "url": "https://example.org/3b",  # duplicate headline
     "time_published": (NOW - timedelta(days=2)).strftime("%Y%m%dT%H%M%S"), "source": "Benzinga",
     "overall_sentiment_score": 0.1,
     "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.45", "relevance_score": "0.9"}]},
    {"title": "Supplier warns on iPhone demand", "url": "https://example.org/4",
     "time_published": (NOW - timedelta(days=3)).strftime("%Y%m%dT%H%M%S"), "source": "Bloomberg",
     "overall_sentiment_score": -0.2,
     "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "-0.35", "relevance_score": "0.6"}]},
    {"title": "Old story", "url": "https://example.org/5", "time_published": "20200101T000000",
     "overall_sentiment_score": 0.0, "ticker_sentiment": []},
]}


@pytest.fixture
def providers(monkeypatch):
    calls = []

    def fake_get(url, params):
        calls.append((url, dict(params)))
        if "company-news" in url:
            return FINNHUB_NEWS
        if "stock/peers" in url:
            return ["AAPL", "MSFT", "DELL"]
        if "stock/metric" in url:
            return {"metric": {"beta": 1.2, "52WeekHigh": 260.1}}
        if "alphavantage" in url:
            return AV_NEWS
        raise AssertionError(url)

    monkeypatch.setenv(ns.FINNHUB_ENV, "test-finnhub")
    monkeypatch.setenv(ns.ALPHA_VANTAGE_ENV, "test-av")
    monkeypatch.setattr(ns.config, "OFFLINE_MODE", False)
    monkeypatch.setattr(ns, "_get_json", fake_get)
    return calls


@pytest.mark.parametrize("text, sign", [
    ("Apple beats estimates and raises guidance", 1), ("Shares plunge after profit warning", -1),
    ("Company did not miss estimates", 1), ("Apple unveils new colours", 0)])
def test_lexicon_score(text, sign):
    score = ns.lexicon_score(text)
    assert (score > 0) - (score < 0) == sign
    assert -1 <= score <= 1


def test_combined_news_dedup_and_scores(providers):
    news = ns.get_news("AAPL", days=30)
    assert len(news) == 4  # 3 Finnhub + 2 AV - 1 duplicate; 2020 story outside the window
    assert news["published"].is_monotonic_decreasing
    warn = news[news["headline"].str.startswith("Supplier")].iloc[0]
    assert warn["sentiment"] == pytest.approx(-0.35) and warn["relevance"] == pytest.approx(0.9 - 0.3)
    assert news[news["headline"].str.startswith("Apple beats")]["sentiment"].iloc[0] > 0
    assert news[news["headline"].str.contains("probe")]["sentiment"].iloc[0] < 0


def test_api_keys_never_in_cache_file_names(providers, tmp_path):
    ns.get_news("AAPL")
    from StockAnalysisSystem.src import config
    names = " ".join(p.name for p in (tmp_path / "cache" / "news").glob("*")) if (tmp_path / "cache").exists() else ""
    names += " ".join(p.name for p in config.CACHE_DIR.glob("news/*"))
    assert "test-finnhub" not in names and "test-av" not in names
    ns.get_news("AAPL")
    assert len(providers) == 2  # second call served from cache


def test_peers_and_metrics(providers):
    assert ns.finnhub_peers("AAPL") == ["MSFT", "DELL"]
    assert ns.finnhub_metrics("AAPL")["beta"] == 1.2


def test_rate_limit_message_keeps_cache_empty(monkeypatch):
    monkeypatch.setenv(ns.ALPHA_VANTAGE_ENV, "k")
    monkeypatch.delenv(ns.FINNHUB_ENV, raising=False)
    monkeypatch.setattr(ns.config, "OFFLINE_MODE", False)
    monkeypatch.setattr(ns, "_get_json", lambda url, params: {"Information": "rate limit"})
    assert ns.get_news("AAPL").empty


def test_no_keys_no_network(monkeypatch):
    monkeypatch.delenv(ns.FINNHUB_ENV, raising=False)
    monkeypatch.delenv(ns.ALPHA_VANTAGE_ENV, raising=False)
    monkeypatch.setattr(ns, "_get_json", lambda *a: pytest.fail("network without key"))
    assert ns.available_providers() == []
    assert ns.get_news("AAPL").empty and ns.finnhub_peers("AAPL") == [] and ns.finnhub_metrics("AAPL") == {}


def test_daily_sentiment_and_labels():
    news = pd.DataFrame({"published": pd.to_datetime(["2025-01-02 10:00", "2025-01-02 15:00", "2025-01-03 09:00"]),
                         "sentiment": [0.5, -0.1, 0.2]})
    daily = ns.daily_sentiment(news)
    assert daily["sentiment"].tolist() == pytest.approx([0.2, 0.2])
    assert daily["articles"].tolist() == [2, 1]
    assert [ns.sentiment_label(v) for v in (0.3, 0.0, -0.3, None)] == ["Bullish", "Neutral", "Bearish", "n/a"]


def _news_script():
    import pandas as pd
    from StockAnalysisSystem.src.pages.company_analysis.news import render_company_news
    render_company_news("AAPL", {}, {}, pd.DataFrame())


def test_news_tab_hint_without_keys(monkeypatch):
    monkeypatch.delenv(ns.FINNHUB_ENV, raising=False)
    monkeypatch.delenv(ns.ALPHA_VANTAGE_ENV, raising=False)
    at = AppTest.from_function(_news_script).run()
    assert not at.exception
    assert any(ns.FINNHUB_ENV in i.value for i in at.info)


def test_news_tab_loads_on_click(providers):
    at = AppTest.from_function(_news_script).run()
    assert not providers
    at.button(key="news_load_AAPL").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.label for m in at.metric][:2] == ["Sentiment", "Articles"]
    assert at.metric[1].value == "4"
