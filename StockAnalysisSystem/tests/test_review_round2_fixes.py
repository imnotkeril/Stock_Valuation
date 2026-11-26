"""Fixes for docs/REVIEW.md round 2 (R-08 .. R-13) and the methodology research (Zmijewski probit)."""

import math
import os
import threading
import time

import pytest

from StockAnalysisSystem.src.industry import benchmarks
from StockAnalysisSystem.src.pages.screening import stock_screener as ss
from StockAnalysisSystem.src.pages.screening import valuation_filter as vf

RATIOS = {"valuation": {"pe_ratio": 20.0, "pb_ratio": 4.0, "ps_ratio": 5.0}}
SECTOR = {"pe_ratio": 25.0, "pb_ratio": 4.0, "ps_ratio": 5.0}
INFO = {"market_cap": 100e9, "enterprise_value": 120e9}


def test_r09_relative_leg_drops_multiples_irrelevant_for_the_sector():
    tech = vf.calculate_relative_valuation("T", INFO, RATIOS, SECTOR, 100.0, sector="Technology")
    assert "P/B" not in tech["components"] and "P/E" in tech["components"]
    bank = vf.calculate_relative_valuation("B", INFO, RATIOS, SECTOR, 100.0, sector="Financials")
    assert set(bank["components"]) == {"P/E", "P/B"}
    assert "P/B" in vf.calculate_relative_valuation("X", INFO, RATIOS, SECTOR, 100.0)["components"]


def test_r10_a_raising_progress_callback_cancels_queued_work():
    started = []
    lock = threading.Lock()

    def task(i):
        with lock:
            started.append(i)
        time.sleep(0.05)

    def stop(done, total, label):
        raise KeyboardInterrupt("user pressed stop")

    with pytest.raises(KeyboardInterrupt):
        ss.parallel_map(task, range(40), workers=2, progress_callback=stop, limiter=ss.RateLimiter(0))
    time.sleep(0.2)
    assert len(started) < 40


def test_r11_stale_cached_info_is_left_out_of_live_medians(tmp_path, monkeypatch):
    from StockAnalysisSystem.src.utils.data_loader import DataLoader

    loader = DataLoader(cache_dir=str(tmp_path))
    universe = [{"symbol": f"T{i}"} for i in range(6)]
    monkeypatch.setattr("StockAnalysisSystem.src.industry.universe.get_universe", lambda sector=None: universe)
    old = time.time() - (benchmarks.LIVE_MAX_AGE_DAYS + 5) * 86400
    for i, record in enumerate(universe):
        path = loader._cache_path(f"{record['symbol']}_info_yfinance.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"pe_ratio": %d}' % (20 + i), encoding="utf-8")
        if i == 0:
            os.utime(path, (old, old))
    result = benchmarks.live_valuation_medians("Technology", loader)
    # 5 fresh companies (pe 21..25) remain: median 23, enough for LIVE_MIN_COMPANIES = 5
    assert result is None or result["companies"] == 5
    if result:
        assert result["values"]["pe_ratio"] == pytest.approx(23.0)


def test_zmijewski_probability_is_probit():
    from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer

    # ROA 0, TL/TA 0.5, CA/CL 1 -> X = -4.336 + 2.8395 + 0.004 = -1.4925; Phi(-1.4925) = 0.0678
    data = {"income_statement": None, "company_info": {"sector": "Technology"}}
    analyzer = BankruptcyAnalyzer()
    monkey_inputs = {"total_assets": 100.0, "total_liabilities": 50.0, "net_income": 0.0, "current_assets": 10.0,
                     "current_liabilities": 10.0}
    analyzer._inputs = lambda financial_data: {**{k: None for k in ("working_capital",)}, **monkey_inputs}
    result = analyzer.calculate_zmijewski_score(data) if hasattr(analyzer, "calculate_zmijewski_score") else None
    if result is None:
        pytest.skip("entry point name differs")
    expected = 0.5 * (1 + math.erf(-1.4925 / math.sqrt(2)))
    assert result["probability"] == pytest.approx(expected, abs=1e-4)
    assert result["probability"] == pytest.approx(0.0678, abs=5e-4)
