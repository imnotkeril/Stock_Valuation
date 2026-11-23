"""Guard against API keys hard-coded in the source (ported from branch chore/audit)."""

import importlib
import re
from pathlib import Path

import StockAnalysisSystem.src.config as config

SRC = Path(__file__).resolve().parents[1] / "src"
# os.environ.get("SOMETHING_KEY", "<non-empty default>") or os.getenv(...) with a default
HARDCODED_DEFAULT = re.compile(r"""(?:environ\.get|getenv)\(\s*["'][A-Z_]*KEY["']\s*,\s*["'][^"']+["']""")


def test_no_hardcoded_key_defaults_in_source():
    offenders = [
        f"{path.relative_to(SRC)}:{lineno}"
        for path in SRC.rglob("*.py")
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if HARDCODED_DEFAULT.search(line)
    ]
    assert not offenders, offenders


def test_keys_come_from_environment(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "from-env")
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    try:
        cfg = importlib.reload(config)
        assert cfg.API_KEYS["alpha_vantage"] == "from-env"
        assert cfg.API_KEYS["financial_modeling_prep"] == ""
    finally:
        monkeypatch.undo()
        importlib.reload(config)
