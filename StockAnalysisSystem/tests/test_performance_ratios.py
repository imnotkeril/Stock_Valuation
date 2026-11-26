"""Risk-adjusted ratios (T23, AC22.2): hand-computed golden values, N/A rules, Risk tab render."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from StockAnalysisSystem.src import navigation as nav
from StockAnalysisSystem.src.models.performance_ratios import performance_ratios, sortino_ratio

MAIN = str(Path(__file__).resolve().parents[1] / "src" / "main.py")

R = [0.02, -0.01, 0.03, -0.02]     # stock daily returns
M = [0.01, -0.005, 0.02, -0.01]    # benchmark daily returns


def _prices(returns, start=100.0):
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    return pd.Series(start * np.cumprod([1.0] + [1 + r for r in returns]), index=dates)


def test_golden_values_rf_zero():
    out = performance_ratios(_prices(R), _prices(M), rf=0.0, min_obs=2)
    # Sortino: downside^2 = [0, 0.0001, 0, 0.0004] -> mean over ALL 4 days = 0.000125, DD = 0.0111803
    #   mean r = 0.005 -> 0.005 * 252 / (0.0111803 * sqrt 252) = 0.005 * sqrt 252 / 0.0111803 = 7.0993
    assert out["sortino"] == pytest.approx(0.005 * math.sqrt(252) / math.sqrt(0.000125))
    assert out["sortino"] == pytest.approx(7.0993, abs=1e-4)
    # Beta: dr = [.015, -.015, .025, -.025], dm = [.00625, -.00875, .01625, -.01375]
    #   sum(dr*dm) = 0.000975, sum(dm^2) = 0.00056875 -> beta = 12/7 = 1.714286
    assert out["beta"] == pytest.approx(12 / 7)
    # Rp = 0.005 * 252 = 1.26, Rm = 0.00375 * 252 = 0.945
    # Treynor = 1.26 / (12/7) = 0.735; Jensen = 1.26 - 12/7 * 0.945 = 1.26 - 1.62 = -0.36
    assert out["treynor"] == pytest.approx(0.735)
    assert out["jensens_alpha"] == pytest.approx(-0.36)
    # IR: active = [.01, -.005, .01, -.01], mean 0.00125; squared deviations sum 3.1875e-4, var (n-1) 1.0625e-4
    #   IR = 0.00125 * 252 / (sqrt(1.0625e-4) * sqrt 252) = 1.92505
    assert out["information_ratio"] == pytest.approx(0.00125 * 252 / (math.sqrt(1.0625e-4) * math.sqrt(252)))
    assert out["information_ratio"] == pytest.approx(1.92505, abs=1e-4)
    # Calmar: prices 100, 102, 100.98, 104.0094, 101.929212; max DD = 101.929212/104.0094 - 1 = -0.02
    #   CAGR over 4/252 years = 1.01929212^63 - 1 = 2.3328 -> Calmar = 2.3328 / 0.02 = 116.6
    assert out["calmar_max_drawdown"] == pytest.approx(-0.02)
    assert out["calmar"] == pytest.approx((1.01929212 ** 63 - 1) / 0.02, rel=1e-6)
    assert out["calmar_window"] == "whole history"


def test_golden_values_with_risk_free_rate():
    rf = 0.0252  # daily 0.0001
    out = performance_ratios(_prices(R), _prices(M), rf=rf, min_obs=2)
    # Sortino: excess = [.0199, -.0101, .0299, -.0201]; downside^2 mean = (0.00010201 + 0.00040401) / 4
    #   = 0.000126505; mean excess 0.0049 -> 0.0049 * sqrt 252 / sqrt(0.000126505) = 6.9158
    assert out["sortino"] == pytest.approx(0.0049 * math.sqrt(252) / math.sqrt(0.000126505))
    assert out["sortino"] == pytest.approx(6.9158, abs=1e-4)
    # Beta unchanged by a constant rf: 12/7. Treynor = (1.26 - 0.0252) / (12/7) = 0.7203
    assert out["beta"] == pytest.approx(12 / 7)
    assert out["treynor"] == pytest.approx(1.2348 * 7 / 12)
    # Jensen = 1.26 - [0.0252 + 12/7 * (0.945 - 0.0252)] = 1.26 - 0.0252 - 1.5768 = -0.342
    assert out["jensens_alpha"] == pytest.approx(-0.342)


def test_sortino_divides_by_all_periods_not_down_days():
    returns = pd.Series([0.01, -0.02, 0.01, 0.0])
    # DD = sqrt(0.0004 / 4) = 0.01 (not sqrt(0.0004 / 1)); mean 0 -> Sortino 0
    assert sortino_ratio(returns) == pytest.approx(0.0)
    assert sortino_ratio(pd.Series([0.01, 0.02])) is None  # no downside


def test_no_benchmark_makes_benchmark_ratios_na():
    for bench in (None, pd.Series(dtype=float)):
        out = performance_ratios(_prices(R), bench, rf=0.0, min_obs=2)
        assert out["beta"] is None and out["treynor"] is None
        assert out["jensens_alpha"] is None and out["information_ratio"] is None
        assert out["sortino"] is not None and out["calmar"] is not None


def test_negative_beta_makes_treynor_na():
    out = performance_ratios(_prices(R), _prices([-r for r in M]), rf=0.0, min_obs=2)
    assert out["beta"] == pytest.approx(-12 / 7)
    assert out["treynor"] is None
    assert out["jensens_alpha"] is not None


def test_no_drawdown_makes_calmar_na():
    out = performance_ratios(_prices([0.01, 0.02, 0.005]), None, min_obs=2)
    assert out["calmar_max_drawdown"] == 0.0 and out["calmar"] is None


def test_calmar_uses_trailing_36_months():
    # 800 returns: first 44 days crash -50 %, then steady +0.1 %/day. The trailing 756-day window excludes the crash.
    returns = [-0.0156] * 44 + [0.001] * 755 + [-0.01]
    out = performance_ratios(_prices(returns), None, min_obs=2)
    assert out["calmar_window"] == "36 months"
    assert out["calmar_max_drawdown"] == pytest.approx(-0.01)
    # window = last 757 prices: 755 days of +0.1 % then -1 %: CAGR = (1.001^755 * 0.99)^(252/756) - 1
    assert out["calmar_cagr"] == pytest.approx((1.001 ** 755 * 0.99) ** (252 / 756) - 1)


def test_too_short_history_is_empty():
    assert performance_ratios(_prices(R), _prices(M)) == {}


def test_risk_tab_shows_ratios_without_warnings():
    at = AppTest.from_file(MAIN, default_timeout=600)
    at.run()
    at.sidebar.radio(key=nav.NAV_KEY).set_value(nav.COMPANY_ANALYSIS).run()
    at.sidebar.selectbox(key="ca_popular").set_value("AAPL").run()
    at.sidebar.button(key="ca_analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    labels = {m.label for m in at.metric}
    assert {"Sharpe ratio", "Sortino ratio", "Treynor ratio", "Calmar ratio", "Information ratio",
            "Jensen's alpha"} <= labels
    assert any("Sortino = (Rp - rf) / downside deviation" in c.value for c in at.caption)
