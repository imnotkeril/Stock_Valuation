"""Sum of the parts for every sector model (offline, synthetic companies)."""

import pytest

from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

#: synthetic kind -> sector model class name
KINDS = {"saas": "TechnologySectorValuation", "bank": "FinancialSectorValuation",
         "industrial": "ManufacturingSectorValuation", "reit": "RealEstateValuation",
         "pharma": "HealthcareValuation", "telecom": "CommunicationValuation",
         "retail": "RetailSectorValuation", "oil_gas": "EnergySectorValuation"}
NEW_SOTP = ["saas", "bank", "industrial", "reit", "pharma", "telecom"]


@pytest.fixture
def factory(data_loader):
    return ValuationFactory(data_loader)


def _value(factory, company, **kwargs):
    return factory.get_company_valuation(company["ticker"], financial_data=company["financial_data"], **kwargs)


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_sector_model_declares_sotp(factory, make_company, kind):
    company = make_company(kind)
    info = company["company_info"]
    model = factory.get_valuation_model(info["sector"], info["industry"])
    assert type(model).__name__ == KINDS[kind]
    assert "sum_of_parts" in model.METHODS


@pytest.mark.parametrize("kind", NEW_SOTP)
def test_sotp_off_by_default_and_runs_when_share_set(factory, make_company, kind):
    company = make_company(kind)
    base = _value(factory, company)
    sotp = base["valuation_methods"].get("sum_of_parts")
    assert sotp is not None and sotp["value_per_share"] is None  # share 0 -> not run
    assert "share" in (sotp.get("note") or sotp.get("error") or "")

    on = _value(factory, company, sotp_second_share=0.3)
    sotp_on = on["valuation_methods"]["sum_of_parts"]
    assert sotp_on["value_per_share"] and sotp_on["value_per_share"] > 0, sotp_on
    assert len(sotp_on["parts"]) == 2
    # Default fair value unchanged by a method that did not run
    assert base["value_per_share"] == pytest.approx(_value(factory, company, sotp_second_share=0.0)["value_per_share"])


@pytest.mark.parametrize("kind", NEW_SOTP)
def test_higher_second_segment_multiple_raises_value(factory, make_company, kind):
    company = make_company(kind)
    low = _value(factory, company, sotp_second_share=0.4, sotp_multiple_second=5.0)
    high = _value(factory, company, sotp_second_share=0.4, sotp_multiple_second=30.0)
    v_low = low["valuation_methods"]["sum_of_parts"]["value_per_share"]
    v_high = high["valuation_methods"]["sum_of_parts"]["value_per_share"]
    if v_low is None or v_high is None:  # outside the plausibility band for this synthetic company
        pytest.skip("value outside plausibility band")
    assert v_high > v_low


def test_parametric_sotp_matches_hand_calculation(factory, make_company):
    company = make_company("industrial")
    result = _value(factory, company, sotp_second_share=0.25, sotp_multiple_first=10.0, sotp_multiple_second=14.0)
    method = result["valuation_methods"]["sum_of_parts"]
    parts = method["parts"]
    ebitda = parts[0]["metric_value"] + parts[1]["metric_value"]
    assert parts[1]["metric_value"] == pytest.approx(ebitda * 0.25)
    expected_ev = ebitda * (0.75 * 10.0 + 0.25 * 14.0)
    assert method["enterprise_value"] == pytest.approx(expected_ev)


def test_explicit_segments_override_sliders(factory, make_company):
    company = make_company("telecom")
    result = _value(factory, company, segments=[{"name": "Towers", "value": 5e9}, {"name": "Wireless", "value": 20e9}])
    method = result["valuation_methods"]["sum_of_parts"]
    assert [p["name"] for p in method["parts"]] == ["Towers", "Wireless"]
    assert method["enterprise_value"] == pytest.approx(25e9)
