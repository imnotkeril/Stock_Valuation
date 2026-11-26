"""
Real estate valuation (REITs, developers and real estate services).

Methods (keys of ``RealEstateValuation.METHODS``):

* ``nav`` - net asset value: NOI capitalised at the cap rate, plus cash, less debt.
* ``ffo_multiple`` - P/FFO and P/AFFO against subsector benchmarks
  (FFO = net income + real estate D&A; AFFO = FFO - maintenance capex).
* ``dcf`` - FCFF DCF on the normalised free cash flow margin.
* ``dividend_discount`` - two-stage DDM (REITs distribute most of their income).
* ``relative`` - EV/EBITDA, P/E and EV/Sales (real estate services).
"""

import logging
from typing import Any, Dict, Optional

from StockAnalysisSystem.src.utils.financial_data import safe_divide
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.sector_specific.common import (
    enable_parametric_sotp,
    SectorValuationMixin,
    ValuationContext,
    core_parameters,
    dividend_discount,
    equity_bridge,
    fcff_dcf,
    latest_metric,
    merge_defaults,
    multiple_parameters,
    not_applicable,
    number_param,
    percent_param,
    series_by_period,
    standard_multiples,
    subsector_multiple_defaults,
)

logger = logging.getLogger(__name__)

_REITS = ["industrial", "residential", "retail", "office", "specialty", "diversified"]

#: Typical cap rates and P/FFO multiples by property type.
_CAP_RATES = {"industrial": 0.050, "residential": 0.050, "retail": 0.065, "office": 0.075, "specialty": 0.055,
              "diversified": 0.060}
_PFFO = {"industrial": 22.0, "residential": 19.0, "retail": 13.0, "office": 9.0, "specialty": 20.0,
         "diversified": 14.0}
_BENCHMARK_MULTIPLES = {"real_estate_services": (14.0, 25.0, 2.0)}
#: Maintenance capex as a share of real estate D&A when deriving AFFO.
_MAINTENANCE_SHARE_OF_DA = 0.25


class RealEstateValuation(SectorValuationMixin, BaseValuation):
    """Valuation model for REITs and real estate companies."""

    SECTOR = "Real Estate"
    SUBSECTORS = {
        "industrial": "Industrial & Logistics REIT",
        "residential": "Residential REIT",
        "retail": "Retail REIT",
        "office": "Office REIT",
        "specialty": "Specialty REIT (data centres, towers, storage)",
        "diversified": "Diversified REIT",
        "real_estate_services": "Real Estate Services & Development",
    }
    METHODS = {
        "nav": "Net Asset Value (cap rate)",
        "ffo_multiple": "P/FFO & P/AFFO",
        "dcf": "Real Estate DCF",
        "dividend_discount": "Dividend Discount Model",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
    }
    SUBSECTOR_METHODS = {**{key: ["nav", "ffo_multiple", "dcf", "dividend_discount"] for key in _REITS},
                         "real_estate_services": ["dcf", "relative"]}
    DEFAULT_METHOD_WEIGHTS = {"nav": 0.35, "ffo_multiple": 0.3, "dcf": 0.2, "dividend_discount": 0.15,
                              "relative": 0.4}
    SUBSECTOR_WEIGHTS = {"real_estate_services": {"dcf": 0.6, "relative": 0.4}}
    PARAMETERS = {
        **core_parameters(discount_default=0.08, years_default=7),
        **multiple_parameters(["real_estate_services"]),
        "cap_rate": percent_param("Cap rate", 0.03, 0.12, 0.06,
                                  "Market capitalisation rate applied to net operating income.", _REITS, step=0.0025),
        "ffo_multiple": number_param("Benchmark P/FFO", 5.0, 35.0, 15.0, "Peer price / FFO multiple.", _REITS),
        "cost_of_equity": percent_param("Cost of equity", 0.04, 0.20, 0.08,
                                        "Discount rate for dividends (CAPM default).", _REITS),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(
        {key: {"cap_rate": rate} for key, rate in _CAP_RATES.items()},
        {key: {"ffo_multiple": multiple} for key, multiple in _PFFO.items()},
        subsector_multiple_defaults(_BENCHMARK_MULTIPLES),
    )
    SUBSECTOR_KEYWORDS = (
        ("industrial", ("reit - industrial", "industrial")),
        ("residential", ("residential",)),
        ("retail", ("reit - retail", "retail")),
        ("office", ("office",)),
        ("specialty", ("specialty", "data center", "tower", "storage", "healthcare facilities")),
        ("diversified", ("diversified", "reit")),
        ("real_estate_services", ("real estate services", "development", "real estate - ", "real estate")),
    )
    SUBSECTOR_ALIASES = {"Mixed": "diversified", "mixed": "diversified", "data_center": "specialty",
                         "tower": "specialty", "services": "real_estate_services", "REIT": "diversified"}

    def _fallback_subsector(self, company_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        return "diversified"

    def _method_runners(self):
        return {"nav": self._nav, "ffo_multiple": self._ffo_multiple,
                "dcf": lambda ctx: fcff_dcf(ctx, margin_years=3), "dividend_discount": dividend_discount,
                "relative": standard_multiples}

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    @staticmethod
    def _noi(ctx: ValuationContext) -> Optional[float]:
        """NOI ~ operating income + depreciation (property-level cash earnings before G&A add-back)."""
        ebitda = latest_metric(ctx, "ebitda")
        if ebitda is not None:
            return ebitda
        ebit, da = latest_metric(ctx, "ebit"), latest_metric(ctx, "da")
        return ebit + (da or 0.0) if ebit is not None else None

    def _nav(self, ctx: ValuationContext) -> Dict[str, Any]:
        noi = self._noi(ctx)
        if not noi or noi <= 0:
            return not_applicable("Net operating income is not positive.")
        cap_rate = ctx.params["cap_rate"]
        property_value = noi / cap_rate
        bridge = equity_bridge(property_value, ctx.capital)
        nav_per_share = ctx.per_share(bridge["equity_value"])
        return {**bridge, "value_per_share": nav_per_share, "net_operating_income": noi, "cap_rate": cap_rate,
                "gross_property_value": property_value,
                "premium_to_nav": safe_divide(ctx.price, nav_per_share) - 1
                if ctx.price and nav_per_share and nav_per_share > 0 else None}

    @staticmethod
    def _ffo(ctx: ValuationContext) -> Dict[str, Any]:
        """
        Nareit FFO (2018 restatement) = net income to common - gains on sale of property
        + impairments + depreciation and amortisation, all from the latest net-income period.

        Gains: income-statement row (positive = gain), else the cash-flow reconciliation row (sign
        flipped); impairments: cash-flow "Asset Impairment Charge", else the income-statement row.
        Missing rows count as 0 and are listed in ``ffo_note``. Yahoo has no real-estate-only D&A,
        so total D&A is used (slightly overstates FFO for REITs with non-property assets).
        """
        income, cash_flow = ctx.financial_data.get("income_statement"), ctx.financial_data.get("cash_flow")
        net_income_series = series_by_period(income, "net_income_common")
        da = latest_metric(ctx, "da")
        if net_income_series.empty or da is None:
            return {"ffo": None, "affo": None}
        period = net_income_series.index[-1]
        net_income = float(net_income_series.iloc[-1])

        def at_period(frame: Any, item: str) -> Optional[float]:
            series = series_by_period(frame, item)
            return float(series[period]) if period in series.index else None

        gains = at_period(income, "gain_on_sale_of_assets")
        if gains is None:
            cash_flow_gain = at_period(cash_flow, "gain_loss_on_sale_of_ppe")
            gains = -cash_flow_gain if cash_flow_gain is not None else None
        impairments = at_period(cash_flow, "asset_impairment_charge")
        if impairments is None:
            impairments = at_period(income, "impairment")
        missing = [name for name, value in (("gains on sale", gains), ("impairments", impairments)) if value is None]
        ffo = net_income - (gains or 0.0) + abs(impairments or 0.0) + da
        return {"ffo": ffo, "affo": ffo - da * _MAINTENANCE_SHARE_OF_DA, "net_income_common": net_income,
                "gains_on_sale": gains, "impairments": impairments,
                "ffo_note": ("FFO = net income + D&A without " + " or ".join(missing) + " (not reported).")
                if missing else None}

    def _ffo_multiple(self, ctx: ValuationContext) -> Dict[str, Any]:
        values = self._ffo(ctx)
        ffo, affo = values["ffo"], values["affo"]
        if not ffo or ffo <= 0:
            return not_applicable("FFO is not positive or depreciation data is missing.")
        multiple = ctx.params["ffo_multiple"]
        ffo_ps, affo_ps = ctx.per_share(ffo), ctx.per_share(affo)
        # AFFO trades at roughly a 15 % higher multiple than FFO (maintenance capex removed).
        value_ffo = ffo_ps * multiple if ffo_ps else None
        value_affo = affo_ps * multiple * 1.15 if affo_ps and affo_ps > 0 else None
        candidates = [v for v in (value_ffo, value_affo) if v]
        return {"value_per_share": sum(candidates) / len(candidates) if candidates else None,
                **({"note": values["ffo_note"]} if values.get("ffo_note") else {}),
                "gains_on_sale": values.get("gains_on_sale"), "impairments": values.get("impairments"),
                "ffo": ffo, "affo": affo, "ffo_per_share": ffo_ps, "affo_per_share": affo_ps,
                "ffo_multiple": multiple, "affo_multiple": multiple * 1.15,
                "value_on_ffo": value_ffo, "value_on_affo": value_affo,
                "current_p_ffo": safe_divide(ctx.price, ffo_ps) if ctx.price and ffo_ps else None}

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def nav_valuation(self, ticker: str, financial_data: Optional[Dict] = None, **kwargs) -> Dict:
        return self.run_method("nav", ticker, financial_data, kwargs.pop("subsector", None), **kwargs)

    def ffo_valuation(self, ticker: str, financial_data: Optional[Dict] = None, **kwargs) -> Dict:
        return self.run_method("ffo_multiple", ticker, financial_data, kwargs.pop("subsector", None), **kwargs)

    def cap_rate_valuation(self, ticker: str, financial_data: Optional[Dict] = None, property_type: str = "Mixed",
                           **kwargs) -> Dict:
        return self.run_method("nav", ticker, financial_data, property_type, **kwargs)

    def real_estate_dcf(self, ticker: str, financial_data: Optional[Dict] = None, **kwargs) -> Dict:
        return self.run_method("dcf", ticker, financial_data, kwargs.pop("subsector", None), **kwargs)

    def calculate_premium_discount_to_nav(self, ticker: str, financial_data: Optional[Dict] = None,
                                          **kwargs) -> Dict:
        """Share price premium (+) / discount (-) to NAV per share."""
        result = self.nav_valuation(ticker, financial_data, **kwargs)
        nav = result.get("value_per_share")
        return {"company": ticker, "nav_per_share": nav, "premium_to_nav": result.get("premium_to_nav"),
                "error": result.get("error") or (None if nav else result.get("note"))}


enable_parametric_sotp(RealEstateValuation, ("Rental portfolio", "Development & services"), (18.0, 10.0))
