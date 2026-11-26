"""Parallel per-ticker fetch for the screener and the valuation filter (review R11 / U11)."""

import threading
import time

import pytest

from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.pages.screening import stock_screener as ss
from StockAnalysisSystem.src.pages.screening import valuation_filter as vf
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "NEE", "PLD", "JNJ", "CAT"]


def test_parallel_map_keeps_input_order_and_reports_every_item():
    seen = []

    def slow_square(x):
        time.sleep(0.02 * (5 - x))  # later items finish first
        return x * x

    out = ss.parallel_map(slow_square, [1, 2, 3, 4], workers=4,
                          progress_callback=lambda done, total, label: seen.append((done, total)),
                          limiter=ss.RateLimiter(0))
    assert out == [1, 4, 9, 16]
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_parallel_map_turns_a_failure_into_none():
    def boom(x):
        if x == 2:
            raise ValueError("bad ticker")
        return x

    assert ss.parallel_map(boom, [1, 2, 3], workers=3) == [1, None, 3]
    assert ss.parallel_map(boom, [1, 2, 3], workers=1) == [1, None, 3]


def test_parallel_map_uses_several_threads():
    names = set()

    def record(_):
        names.add(threading.current_thread().name)
        time.sleep(0.05)

    ss.parallel_map(record, range(6), workers=3, limiter=ss.RateLimiter(0))
    assert len(names) > 1


def test_rate_limiter_spaces_task_starts():
    # Six starts 20 ms apart cannot all happen in less than 5 x 20 ms, however many workers run.
    begin = time.monotonic()
    ss.parallel_map(lambda _: None, range(6), workers=6, limiter=ss.RateLimiter(0.02))
    assert time.monotonic() - begin >= 0.095


def test_screener_parallel_matches_sequential(data_loader):
    analyzer = FinancialRatioAnalyzer()
    seq_fail, par_fail = [], []
    sequential = ss.screen_stocks(TICKERS, data_loader, analyzer, workers=1, failures=seq_fail)
    parallel = ss.screen_stocks(TICKERS, data_loader, analyzer, workers=6, failures=par_fail)
    assert [r["symbol"] for r in parallel] == [r["symbol"] for r in sequential]
    assert parallel == sequential
    assert par_fail == seq_fail
    assert sequential, "fixtures should give at least one row"


@pytest.mark.parametrize("approach", ["Sector model", "Relative", "Combined"])
def test_valuation_filter_parallel_matches_sequential(data_loader, approach):
    analyzer = FinancialRatioAnalyzer()
    factory = ValuationFactory(data_loader)
    universe = [{"symbol": t, "name": t} for t in TICKERS]

    def run(workers):
        return vf.perform_valuation_analysis(universe, data_loader, analyzer, factory, approach, "All",
                                             margin_of_safety=-10.0, include_outliers=False, workers=workers)

    sequential, parallel = run(1), run(6)
    key = [(r["symbol"], round(r["fair_value"], 6)) for r in sequential]
    assert [(r["symbol"], round(r["fair_value"], 6)) for r in parallel] == key
    assert parallel.hidden_outliers == sequential.hidden_outliers
    if approach != "Relative":  # offline fixtures carry no sector multiples for the relative leg
        assert key, "margin of safety -1000 % keeps every valued stock"
