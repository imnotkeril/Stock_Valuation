"""Saved custom screener templates (offline)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.pages.screening import stock_screener as ss

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")


def test_save_load_delete_roundtrip():
    assert ss.load_user_templates() == {}
    name = ss.save_user_template("  Cheap   quality ", {"filters": {"pe_ratio": [0.0, 15.0]}, "sector": "All"})
    assert name == "Cheap quality"
    assert ss.load_user_templates()[name]["filters"] == {"pe_ratio": [0.0, 15.0]}
    ss.save_user_template(name, {"filters": {}, "sector": "Energy"})  # same name overwrites
    assert ss.load_user_templates()[name]["sector"] == "Energy"
    assert ss.delete_user_template(name)
    assert not ss.delete_user_template(name)
    assert ss.load_user_templates() == {}


@pytest.mark.parametrize("bad", ["", "   ", "Value"])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError):
        ss.save_user_template(bad, {})


def test_corrupt_file_means_no_templates():
    ss.TEMPLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ss.TEMPLATES_FILE.write_text("{not json", encoding="utf-8")
    assert ss.load_user_templates() == {}


def _screener() -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.STOCK_SCREENER).run()
    return at


def test_save_and_reapply_template_in_ui():
    at = _screener()
    at.sidebar.selectbox(key=ss.STRATEGY_KEY).set_value("Quality").run()
    at.sidebar.selectbox(key=f"{ss.PREFIX}_sector").set_value("Energy").run()
    at.sidebar.slider(key=ss._range_key("roe")).set_value((20.0, 200.0)).run()
    at.sidebar.text_input(key=ss.TEMPLATE_NAME_KEY).input("Energy quality").run()
    at.sidebar.button(key=f"{ss.PREFIX}_template_save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Saved template 'Energy quality'" in s.value for s in at.sidebar.success)
    assert at.sidebar.selectbox(key=ss.STRATEGY_KEY).value == "Energy quality"

    saved = ss.load_user_templates()["Energy quality"]
    assert saved["sector"] == "Energy"
    assert saved["filters"]["roe"] == [20.0, 200.0]
    assert set(saved["filters"]) == {"roe", "net_margin", "debt_equity"}

    # Fresh session: picking the saved template restores every control
    at = _screener()
    assert "Energy quality (saved)" in at.sidebar.selectbox(key=ss.STRATEGY_KEY).options
    at.sidebar.selectbox(key=ss.STRATEGY_KEY).set_value("Energy quality").run()
    assert at.sidebar.selectbox(key=f"{ss.PREFIX}_sector").value == "Energy"
    assert sorted(at.sidebar.multiselect(key=ss.FILTERS_KEY).value) == ["debt_equity", "net_margin", "roe"]
    assert at.sidebar.slider(key=ss._range_key("roe")).value == (20.0, 200.0)

    at.sidebar.button(key=f"{ss.PREFIX}_template_delete").click().run()
    assert not at.exception
    assert at.sidebar.selectbox(key=ss.STRATEGY_KEY).value == "Custom"
    assert "Energy quality" not in ss.load_user_templates()


def test_saving_without_name_warns():
    at = _screener()
    at.sidebar.button(key=f"{ss.PREFIX}_template_save").click().run()
    assert not at.exception
    assert not at.error
    assert any("Enter a template name" in w.value for w in at.sidebar.warning)
