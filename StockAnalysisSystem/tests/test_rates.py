"""Risk-free rate service: FRED DGS10 -> stale cache -> constant, env override, offline safety."""

from types import SimpleNamespace

import pytest

from StockAnalysisSystem.src import config
from StockAnalysisSystem.src.utils import data_loader as dl_module
from StockAnalysisSystem.src.utils import rates
from StockAnalysisSystem.src.utils.data_loader import DataLoader

CSV = "observation_date,DGS10\n2026-11-20,4.40\n2026-11-21,.\n2026-11-24,4.12\n"


@pytest.fixture
def loader(tmp_path, monkeypatch):
    monkeypatch.delenv(config.RISK_FREE_RATE_ENV, raising=False)
    return DataLoader(cache_dir=str(tmp_path / "cache"), offline=False)


@pytest.fixture
def fred(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["id"])
        return SimpleNamespace(text=CSV, raise_for_status=lambda: None)

    monkeypatch.setattr(dl_module.requests, "get", fake_get)
    return calls


def test_latest_valid_observation_as_fraction(loader, fred):
    rf = rates.risk_free_rate(loader)
    assert rf.rate == pytest.approx(0.0412)
    assert rf.as_of == "2026-11-24" and rf.source == "FRED DGS10"
    assert rf.label == "4.12 % (FRED DGS10, as of 2026-11-24)"
    assert fred == ["DGS10"]


def test_cached_for_a_day_and_offline_reads_cache_only(loader, fred):
    rates.risk_free_rate(loader)
    rates.risk_free_rate(loader)
    assert fred == ["DGS10"]
    loader.offline = True
    assert rates.risk_free_rate(loader).rate == pytest.approx(0.0412)
    assert fred == ["DGS10"]


def test_stale_cache_served_when_fred_fails(loader, fred, monkeypatch):
    rates.risk_free_rate(loader)
    monkeypatch.setattr(dl_module.DataLoader, "_is_fresh", staticmethod(lambda path, days: False))

    def broken(url, params, timeout):
        raise ConnectionError("down")

    monkeypatch.setattr(dl_module.requests, "get", broken)
    assert rates.risk_free_rate(loader).rate == pytest.approx(0.0412)


def test_fallback_constant_without_data(tmp_path, monkeypatch):
    monkeypatch.delenv(config.RISK_FREE_RATE_ENV, raising=False)
    offline = DataLoader(cache_dir=str(tmp_path / "empty"), offline=True)
    rf = rates.risk_free_rate(offline)
    assert rf.rate == config.RISK_FREE_RATE_FALLBACK and rf.source == "fallback constant" and rf.as_of is None


def test_implausible_value_ignored(loader, monkeypatch):
    monkeypatch.setattr(dl_module.requests, "get", lambda url, params, timeout: SimpleNamespace(
        text="observation_date,DGS10\n2026-11-24,45\n", raise_for_status=lambda: None))
    assert rates.risk_free_rate(loader).source == "fallback constant"


@pytest.mark.parametrize("raw, expected", [("0.045", 0.045), ("4.5", 0.045), ("4.5%", 0.045)])
def test_env_override(loader, fred, monkeypatch, raw, expected):
    monkeypatch.setenv(config.RISK_FREE_RATE_ENV, raw)
    rf = rates.risk_free_rate(loader)
    assert rf.rate == pytest.approx(expected) and rf.source == "user override"
    assert fred == []


def test_invalid_env_override_ignored(loader, fred, monkeypatch):
    monkeypatch.setenv(config.RISK_FREE_RATE_ENV, "abc")
    assert rates.risk_free_rate(loader).source == "FRED DGS10"


def test_never_raises(monkeypatch):
    monkeypatch.delenv(config.RISK_FREE_RATE_ENV, raising=False)
    broken = SimpleNamespace(get_fred_series=lambda code: 1 / 0)
    assert rates.risk_free_rate(broken).source == "fallback constant"
