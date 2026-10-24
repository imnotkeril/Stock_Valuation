import importlib
from pathlib import Path

import StockAnalysisSystem.src.config as config


def test_no_hardcoded_keys(monkeypatch):
    for name in ("ALPHA_VANTAGE", "FINNHUB", "FINANCIAL_MODELING_PREP"):
        monkeypatch.delenv(f"{name}_API_KEY", raising=False)
        monkeypatch.delenv(f"{name}_KEY", raising=False)
    cfg = importlib.reload(config)
    source = Path(cfg.__file__).read_text(encoding="utf-8")
    assert 'environ.get("ALPHA_VANTAGE_API_KEY", "' not in source
    assert set(cfg.API_KEYS) == {"alpha_vantage", "finnhub", "financial_modeling_prep"}


def test_keys_read_from_env_with_legacy_alias(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "canonical")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("FINNHUB_KEY", "legacy")
    cfg = importlib.reload(config)
    assert cfg.API_KEYS["alpha_vantage"] == "canonical"
    assert cfg.API_KEYS["finnhub"] == "legacy"
