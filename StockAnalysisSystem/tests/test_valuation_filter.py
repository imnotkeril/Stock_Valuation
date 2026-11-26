"""Tests for the relative valuation used by the valuation filter page."""

import pandas as pd
import pytest

from StockAnalysisSystem.src.pages.screening import valuation_filter as vf
from StockAnalysisSystem.src.pages.screening.valuation_filter import calculate_relative_valuation

RATIOS = {"valuation": {"pe_ratio": 20.0, "pb_ratio": 4.0, "ps_ratio": 5.0, "ev_ebitda": 12.0}}
SECTOR = {"pe_ratio": 25.0, "pb_ratio": 4.0, "ps_ratio": 5.0, "ev_ebitda": 12.0}
INFO = {"market_cap": 100e9, "enterprise_value": 120e9}


def test_no_discount_uses_sector_multiples():
    result = calculate_relative_valuation("TEST", INFO, RATIOS, SECTOR, 100.0)
    assert result is not None


@pytest.mark.parametrize("discount", [0.0, 0.2])
def test_discount_lowers_fair_values(discount):
    plain = calculate_relative_valuation("TEST", INFO, RATIOS, SECTOR, 100.0)
    discounted = calculate_relative_valuation("TEST", INFO, RATIOS, SECTOR, 100.0,
                                              discounts={"P/E": discount, "P/B": discount, "P/S": discount,
                                                         "EV/EBITDA": discount})
    if discount == 0:
        assert discounted == plain
    else:
        assert discounted["fair_value"] < plain["fair_value"]


def test_ev_ebitda_bridges_through_net_debt():
    result = calculate_relative_valuation("TEST", INFO, {"valuation": {"ev_ebitda": 12.0}},
                                          {"ev_ebitda": 12.0}, 100.0)
    # Same multiple as the sector: implied equity equals market cap -> value equals price
    assert result["fair_value"] == pytest.approx(100.0)


# --- sanity layer (review R4 / U2) --------------------------------------------------------------
def _result(model, methods, weights):
    return {"model": model, "value_per_share": sum(v * weights[k] for k, (_, v) in methods.items() if k in weights),
            "valuation_methods": {k: {"method": label, "value_per_share": v} for k, (label, v) in methods.items()},
            "valuation_summary": {"weights": weights}, "assumptions": {"discount_rate": 0.09}}


BANK = _result("FinancialSectorValuation",
               {"excess_return": ("Residual income (excess return)", 330.0), "pe_multiple": ("P/E multiple", 240.0),
                "asset": ("Asset-based", None)},
               {"excess_return": 0.6, "pe_multiple": 0.4})


def test_method_label_names_model_and_heaviest_method():
    assert vf.method_label(BANK) == "Financial: Residual income (excess return)"
    assert vf.method_label({"model": "CommunicationValuation"}) == "Communication"


def test_method_values_skip_unweighted_and_missing():
    assert vf.method_values_of(BANK) == {"Residual income (excess return)": 330.0, "P/E multiple": 240.0}


@pytest.mark.parametrize("fair, price, flagged", [(250.0, 100.0, True), (40.0, 100.0, False), (210.0, 100.0, True),
                                                  (150.0, 100.0, False), (0.0, 100.0, False)])
def test_outliers_beyond_100_percent_are_flagged(fair, price, flagged):
    check = vf.sanity_check(fair, price, {"a": fair})
    assert (check["flag"] == "check inputs") is flagged
    if flagged:
        assert check["confidence"] == "Low"


def test_spread_drives_confidence():
    assert vf.sanity_check(100.0, 90.0, {"a": 95.0, "b": 105.0})["confidence"] == "High"
    assert vf.sanity_check(100.0, 90.0, {"a": 80.0, "b": 120.0})["confidence"] == "Medium"
    wide = vf.sanity_check(100.0, 90.0, {"a": 50.0, "b": 150.0})
    assert wide["confidence"] == "Low" and wide["range"] == (50.0, 150.0) and wide["spread"] == pytest.approx(1.0)


class _Loader:
    def __init__(self, prices):
        self.prices = prices

    def get_company_info(self, symbol):
        return {"name": symbol, "sector": "Financials", "market_cap": 1e10}

    def get_historical_prices(self, symbol, start, end):
        return pd.DataFrame({"Close": [self.prices[symbol]]})

    def get_financial_statements(self, symbol, kind, period):
        return pd.DataFrame({"2024": [1.0]}, index=["Total Revenue"])


class _Ratios:
    def get_sector_benchmarks(self, sector):
        return {}

    def calculate_ratios(self, data):
        return {}


class _Factory:
    def get_company_valuation(self, symbol, sector=None, financial_data=None, **kwargs):
        return BANK


def _screen(**params):
    universe = [{"symbol": "CHEAP"}, {"symbol": "WILD"}]
    loader = _Loader({"CHEAP": 220.0, "WILD": 90.0})  # fair value 294: +34 % and +227 %
    return vf.perform_valuation_analysis(universe, loader, _Ratios(), _Factory(), "Sector model", "All",
                                         margin_of_safety=0.25, **params)


def test_outliers_hidden_by_default_and_method_is_the_real_model():
    rows = _screen()
    assert [r["symbol"] for r in rows] == ["CHEAP"] and rows.hidden_outliers == ["WILD"]
    row = rows[0]
    assert row["valuation_method"] == "Financial: Residual income (excess return)"
    assert row["method_range"] == (240.0, 330.0) and row["flag"] is None
    assert row["valuation_components"] == {"Residual income (excess return)": 330.0, "P/E multiple": 240.0}


def test_outliers_listed_and_flagged_on_request():
    rows = _screen(include_outliers=True)
    wild = next(r for r in rows if r["symbol"] == "WILD")
    assert wild["flag"] == "check inputs" and wild["confidence_level"] == "Low" and "Check inputs" in wild["notes"]
    assert rows.hidden_outliers == []


def test_dcf_approach_is_an_alias_of_the_sector_model():
    rows = vf.perform_valuation_analysis([{"symbol": "CHEAP"}], _Loader({"CHEAP": 220.0}), _Ratios(), _Factory(),
                                         "DCF", "All", margin_of_safety=0.25)
    assert rows[0]["valuation_method"].startswith("Financial")
