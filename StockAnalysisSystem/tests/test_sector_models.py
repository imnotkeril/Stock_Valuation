"""Contract and behaviour tests for every sector-specific valuation model."""

import math

import pytest

from StockAnalysisSystem.src.valuation.sector_factor import MODEL_CLASSES, ValuationFactory

SECTOR_MODELS = [k for k in MODEL_CLASSES if k != "Base"]

#: synthetic profile -> (expected model class, expected detected subsector)
DETECTION = {
    "bank": ("FinancialSectorValuation", "bank"),
    "oil_gas": ("EnergySectorValuation", "integrated"),
    "utility": ("EnergySectorValuation", "utilities"),
    "retail": ("RetailSectorValuation", "specialty_retail"),
    "staples": ("RetailSectorValuation", "consumer_staples"),
    "industrial": ("ManufacturingSectorValuation", "industrial_machinery"),
    "materials": ("ManufacturingSectorValuation", "chemicals"),
    "reit": ("RealEstateValuation", "industrial"),
    "pharma": ("HealthcareValuation", "pharmaceuticals"),
    "biotech_loss": ("HealthcareValuation", "biotechnology"),
    "telecom": ("CommunicationValuation", "telecom"),
    "saas": ("TechnologySectorValuation", "saas"),
}

REQUIRED_KEYS = {"company", "sector", "subsector", "model", "current_price", "value_per_share", "potential_upside",
                 "recommendation", "valuation_summary", "valuation_methods", "assumptions", "notes", "timestamp"}


@pytest.fixture
def factory(data_loader):
    return ValuationFactory(data_loader)


def _valuate(factory, company, **kwargs):
    return factory.get_company_valuation(company["ticker"], financial_data=company["financial_data"], **kwargs)


@pytest.mark.parametrize("model_key", SECTOR_MODELS)
def test_contract_declarations(data_loader, model_key):
    model = MODEL_CLASSES[model_key](data_loader)
    assert model.SUBSECTORS and model.METHODS
    for subsector in model.SUBSECTORS:
        methods = model.get_available_methods(subsector)
        assert methods and set(methods) <= set(model.METHODS)
        for name, spec in model.get_parameters(subsector).items():
            assert spec["min"] <= spec["default"] <= spec["max"], (model_key, subsector, name)
            assert spec["kind"] in {"percent", "number", "int", "currency"}


@pytest.mark.parametrize("kind", sorted(DETECTION))
def test_model_selection_and_subsector_detection(factory, make_company, kind):
    company = make_company(kind)
    info = company["company_info"]
    model_name, subsector = DETECTION[kind]
    assert factory.get_valuation_model(info["sector"], info["industry"]).__class__.__name__ == model_name
    assert factory.get_default_subsector(company["ticker"], info["sector"], company["financial_data"], info) == subsector


@pytest.mark.parametrize("kind", sorted(DETECTION))
def test_every_subsector_produces_a_valid_result(factory, make_company, kind):
    company = make_company(kind)
    info = company["company_info"]
    model = factory.get_valuation_model(info["sector"], info["industry"])
    price = company["profile"].price
    for subsector in model.SUBSECTORS:
        result = _valuate(factory, company, subsector=subsector)
        assert REQUIRED_KEYS <= set(result), subsector
        assert "error" not in result, (subsector, result.get("error"))
        assert set(result["valuation_methods"]) == set(model.get_available_methods(subsector))
        for method in result["valuation_methods"].values():
            value = method["value_per_share"]
            assert value is None or (math.isfinite(value) and value > 0)
            assert value is None or value < price * 20
        if result["value_per_share"] is not None:
            assert 0.05 * price < result["value_per_share"] < 10 * price, subsector


def test_methods_argument_restricts_output(factory, make_company):
    result = _valuate(factory, make_company("oil_gas"), subsector="upstream", methods=["reserve_nav"])
    assert list(result["valuation_methods"]) == ["reserve_nav"]


@pytest.mark.parametrize("kind, subsector, method, param, low, high", [
    ("oil_gas", "upstream", "reserve_nav", "oil_price", 50, 90),
    ("utility", "utilities", "regulated_asset_base", "allowed_roe", 0.08, 0.11),
    ("reit", "industrial", "nav", "cap_rate", 0.07, 0.04),           # higher cap rate -> lower NAV
    ("industrial", "industrial_machinery", "dcf", "discount_rate", 0.12, 0.08),
    ("oil_gas", "renewables", "project_dcf", "capacity_factor", 0.25, 0.45),
])
def test_parameters_move_values_in_expected_direction(factory, make_company, kind, subsector, method, param,
                                                      low, high):
    company = make_company(kind)
    values = [_valuate(factory, company, subsector=subsector, methods=[method], **{param: v})
              ["valuation_methods"][method]["value_per_share"] for v in (low, high)]
    assert values[0] is not None and values[1] is not None
    assert values[0] < values[1]


def test_subscriber_method_needs_user_metrics(factory, make_company):
    company = make_company("telecom")
    without = _valuate(factory, company, subsector="telecom", methods=["subscriber"])
    assert without["valuation_methods"]["subscriber"]["value_per_share"] is None
    low_churn = _valuate(factory, company, subsector="telecom", methods=["subscriber"], subscribers=200,
                         arpu_monthly=56, churn_rate=0.08)["valuation_methods"]["subscriber"]["value_per_share"]
    high_churn = _valuate(factory, company, subsector="telecom", methods=["subscriber"], subscribers=200,
                          arpu_monthly=56, churn_rate=0.15)["valuation_methods"]["subscriber"]["value_per_share"]
    assert low_churn and high_churn and high_churn < low_churn


def test_pipeline_rnpv_adds_risk_adjusted_value(factory, make_company):
    company = make_company("pharma")
    base = _valuate(factory, company, subsector="pharmaceuticals", methods=["rd_adjusted_dcf"])
    pipeline = [{"name": "Drug A", "phase": "phase 3", "peak_sales": 5e9, "years_to_launch": 2},
                {"name": "Drug B", "phase": "phase 1", "peak_sales": 3e9, "years_to_launch": 6}]
    result = _valuate(factory, company, subsector="pharmaceuticals", methods=["pipeline_rnpv"],
                      pipeline_data=pipeline)
    method = result["valuation_methods"]["pipeline_rnpv"]
    assert method["pipeline_rnpv"] > 0
    assert method["programmes"][0]["probability"] > method["programmes"][1]["probability"]
    assert method["value_per_share"] > base["valuation_methods"]["rd_adjusted_dcf"]["value_per_share"]


def test_loss_making_biotech_gets_no_fake_dcf(factory, make_company):
    result = _valuate(factory, make_company("biotech_loss"))
    assert result["valuation_methods"]["rd_adjusted_dcf"]["value_per_share"] is None


def test_banks_do_not_use_fcf_dcf(factory, make_company):
    result = _valuate(factory, make_company("bank"))
    assert "dcf" not in result["valuation_methods"] and "fcf_dcf" not in result["valuation_methods"]


@pytest.mark.parametrize("kind", ["oil_gas", "reit", "pharma", "telecom", "industrial"])
def test_empty_statements_return_error_result(factory, make_company, kind):
    import pandas as pd

    company = make_company(kind)
    data = {**company["financial_data"], "income_statement": pd.DataFrame(), "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame()}
    result = factory.get_company_valuation(company["ticker"], financial_data=data)
    assert result["value_per_share"] is None and result.get("error")


def test_real_companies(factory, aapl, msft):
    for company in (aapl, msft):
        result = _valuate(factory, company)
        price = company["financial_data"]["market_data"]["share_price"]
        assert result["model"] == "TechnologySectorValuation"
        assert 0.2 * price < result["value_per_share"] < 5 * price


def test_legacy_entry_points_still_work(data_loader, make_company):
    from StockAnalysisSystem.src.valuation.sector_specific.energy_sector import EnergySectorValuation
    from StockAnalysisSystem.src.valuation.sector_specific.real_estate import RealEstateValuation

    oil, reit = make_company("oil_gas"), make_company("reit")
    upstream = EnergySectorValuation(data_loader).value_upstream_company(oil["ticker"], oil["financial_data"])
    assert upstream["subsector"] == "upstream" and upstream["values_by_method"]
    premium = RealEstateValuation(data_loader).calculate_premium_discount_to_nav(reit["ticker"],
                                                                               reit["financial_data"])
    assert premium["nav_per_share"] > 0 and premium["premium_to_nav"] is not None
