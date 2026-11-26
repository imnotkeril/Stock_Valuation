"""Confidence on the recommendation badge (review R6)."""

import pandas as pd
import pytest

from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation, assess_confidence, get_recommendation
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory


def _summary(values, price=100.0, weights=None):
    return BaseValuation.summarize_valuations({k: {"value_per_share": v} for k, v in values.items()}, weights, price)


def _methods(values, **extra):
    return {k: {"method": k.upper(), "value_per_share": v, **extra.get(k, {})} for k, v in values.items()}


def test_agreeing_methods_give_high_confidence():
    values = {"dcf": 110.0, "pe": 120.0, "ev_ebitda": 115.0}
    confidence = assess_confidence(_methods(values), _summary(values), {k: 1 for k in values})
    assert confidence == {"level": "High", "reasons": []}


def test_disagreement_terminal_share_and_missing_methods_lower_confidence():
    values = {"dcf": 60.0, "pe": 150.0}
    weights = {"dcf": 0.4, "pe": 0.3, "ev_ebitda": 0.2, "asset": 0.1}
    methods = _methods(values, dcf={"terminal_value_share": 0.85})
    confidence = assess_confidence(methods, _summary(values, weights=weights), weights)
    assert confidence["level"] == "Low"
    text = " ".join(confidence["reasons"])
    assert "methods disagree" in text and "terminal value" in text and "2 of 4 weighted methods" in text


def test_single_method_and_short_history_are_medium_or_low():
    values = {"dcf": 105.0}
    assert assess_confidence(_methods(values), _summary(values), {"dcf": 1})["level"] == "Medium"
    short = {"income_statement": pd.DataFrame({"2024": [1.0], "2023": [1.0]})}
    assert assess_confidence(_methods(values), _summary(values), {"dcf": 1}, short)["level"] == "Low"


def test_implausible_upside_is_low():
    values = {"dcf": 300.0, "pe": 310.0}
    confidence = assess_confidence(_methods(values), _summary(values), {"dcf": 1, "pe": 1})
    assert confidence["level"] == "Low" and "implausibly large" in confidence["reasons"][0]


@pytest.mark.parametrize("upside, label", [(40, "Below model value"), (-40, "Above model value"), (5, "Hold")])
def test_low_confidence_softens_strong_labels(upside, label):
    rec = get_recommendation(upside, {"level": "Low", "reasons": ["methods disagree"]})
    assert rec["label"] == label and rec["confidence"] == "Low"
    if label != "Hold":
        assert rec["model_label"] in ("Strong Buy", "Sell") and "methods disagree" in rec["rationale"]


def test_high_confidence_keeps_label():
    rec = get_recommendation(30, {"level": "High", "reasons": []})
    assert rec["label"] == "Strong Buy" and rec["confidence"] == "High"
    assert "confidence" not in get_recommendation(30)


def test_every_sector_result_carries_confidence(any_synthetic_company):
    company = any_synthetic_company
    result = ValuationFactory().get_company_valuation(company["ticker"], sector=company["company_info"]["sector"],
                                                      financial_data=company["financial_data"])
    rec = result["recommendation"]
    if result.get("value_per_share"):
        assert rec["confidence"] in ("High", "Medium", "Low")
