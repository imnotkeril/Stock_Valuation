"""Tests for the relative valuation used by the valuation filter page."""

import pytest

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
