"""
Healthcare valuation (pharma, biotech, medical devices, healthcare services).

Methods (keys of ``HealthcareValuation.METHODS``):

* ``rd_adjusted_dcf`` - FCFF DCF; reports EBIT and ROIC with R&D capitalised
  (R&D asset amortised straight-line over ``rd_life`` years). FCFF itself is
  unaffected by capitalisation, so the DCF value is not inflated.
* ``relative`` - EV/EBITDA, P/E and EV/Sales against subsector benchmarks.
* ``pipeline_rnpv`` - risk-adjusted NPV of development programmes; needs the
  ``pipeline_data`` keyword argument (list of dicts with ``name``, ``phase``,
  ``peak_sales``, ``years_to_launch`` and optional ``probability``,
  ``margin``, ``exclusivity_years``).
* ``dividend_discount`` - two-stage DDM (large pharma).
"""

import logging
from typing import Any, Dict, List, Optional

from StockAnalysisSystem.src.utils.financial_data import get_series, safe_divide, to_float
from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.sector_specific.common import (
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
    standard_multiples,
    subsector_multiple_defaults,
)

logger = logging.getLogger(__name__)

_BENCHMARK_MULTIPLES = {
    "pharmaceuticals": (12.0, 16.0, 4.0),
    "biotechnology": (15.0, 20.0, 6.0),
    "medical_devices": (18.0, 26.0, 5.0),
    "healthcare_services": (11.0, 17.0, 0.8),
}

#: Cumulative probability of reaching the market by development phase (industry averages).
PHASE_PROBABILITIES = {
    "preclinical": 0.07,
    "phase 1": 0.10,
    "phase 2": 0.18,
    "phase 3": 0.55,
    "filed": 0.85,
    "approved": 1.0,
}


class HealthcareValuation(SectorValuationMixin, BaseValuation):
    """Valuation model for healthcare companies."""

    SECTOR = "Healthcare"
    SUBSECTORS = {
        "pharmaceuticals": "Pharmaceuticals",
        "biotechnology": "Biotechnology",
        "medical_devices": "Medical Devices & Instruments",
        "healthcare_services": "Healthcare Services & Insurance",
    }
    METHODS = {
        "rd_adjusted_dcf": "R&D-Adjusted DCF",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
        "pipeline_rnpv": "Pipeline rNPV",
        "dividend_discount": "Dividend Discount Model",
    }
    SUBSECTOR_METHODS = {
        "pharmaceuticals": ["rd_adjusted_dcf", "relative", "pipeline_rnpv", "dividend_discount"],
        "biotechnology": ["rd_adjusted_dcf", "relative", "pipeline_rnpv"],
        "medical_devices": ["rd_adjusted_dcf", "relative"],
        "healthcare_services": ["rd_adjusted_dcf", "relative", "dividend_discount"],
    }
    DEFAULT_METHOD_WEIGHTS = {"rd_adjusted_dcf": 0.5, "relative": 0.3, "pipeline_rnpv": 0.3,
                              "dividend_discount": 0.1}
    SUBSECTOR_WEIGHTS = {"biotechnology": {"rd_adjusted_dcf": 0.3, "relative": 0.2, "pipeline_rnpv": 0.5}}
    PARAMETERS = {
        **core_parameters(discount_default=0.09, years_default=7),
        **multiple_parameters(),
        "rd_life": number_param("R&D amortisation life (years)", 3, 15, 8,
                                "Useful life of capitalised R&D for the adjusted EBIT / ROIC.", step=1, kind="int"),
        "cost_of_equity": percent_param("Cost of equity", 0.04, 0.20, 0.08,
                                        "Discount rate for dividends (CAPM default).",
                                        ["pharmaceuticals", "healthcare_services"]),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(
        subsector_multiple_defaults(_BENCHMARK_MULTIPLES),
        {"biotechnology": {"discount_rate": 0.11, "rd_life": 10}, "healthcare_services": {"rd_life": 5}},
    )
    SUBSECTOR_KEYWORDS = (
        ("biotechnology", ("biotech",)),
        ("pharmaceuticals", ("drug manufacturer", "pharmaceutical", "specialty & generic")),
        ("medical_devices", ("medical device", "medical instrument", "diagnostics", "life sciences tools")),
        ("healthcare_services", ("healthcare plans", "health information", "medical care", "healthcare services",
                                 "medical distribution", "medical facilities")),
    )
    SUBSECTOR_ALIASES = {"Pharmaceuticals": "pharmaceuticals", "Biotechnology": "biotechnology",
                         "Medical Devices": "medical_devices", "Healthcare Services": "healthcare_services",
                         "pharma": "pharmaceuticals", "biotech": "biotechnology", "devices": "medical_devices",
                         "services": "healthcare_services"}
    EXTRA_INPUTS = ("pipeline_data",)

    def _method_runners(self):
        return {"rd_adjusted_dcf": self._rd_adjusted_dcf, "relative": standard_multiples,
                "pipeline_rnpv": self._pipeline_rnpv, "dividend_discount": dividend_discount}

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    @staticmethod
    def _rd_asset(ctx: ValuationContext) -> Dict[str, Optional[float]]:
        """Capitalised R&D asset and amortisation from the available R&D history."""
        rd = get_series(ctx.income, "rd_expense", ascending=True)
        life = ctx.params["rd_life"]
        if rd.empty:
            return {"rd_asset": None, "rd_amortisation": None, "rd_expense": None}
        values = list(rd.values)
        # Assume pre-history years at the earliest observed level.
        history = [values[0]] * max(life - len(values), 0) + values
        recent = history[-life:]
        asset = sum(spend * (index + 1) / life for index, spend in enumerate(recent))
        amortisation = sum(recent) / life
        return {"rd_asset": asset, "rd_amortisation": amortisation, "rd_expense": values[-1]}

    def _rd_adjusted_dcf(self, ctx: ValuationContext) -> Dict[str, Any]:
        result = fcff_dcf(ctx, margin_years=3)
        rd = self._rd_asset(ctx)
        ebit = latest_metric(ctx, "ebit")
        if rd["rd_expense"] is not None and ebit is not None:
            adjusted_ebit = ebit + rd["rd_expense"] - rd["rd_amortisation"]
            equity = ctx.latest("balance_sheet", "total_equity") or 0.0
            invested = equity + ctx.capital["total_debt"] - ctx.capital["cash"]
            result.update({
                "rd_expense": rd["rd_expense"], "capitalised_rd_asset": rd["rd_asset"],
                "rd_amortisation": rd["rd_amortisation"], "rd_life": ctx.params["rd_life"],
                "reported_ebit": ebit, "rd_adjusted_ebit": adjusted_ebit,
                "rd_adjusted_roic": safe_divide(adjusted_ebit * (1 - ctx.tax_rate), invested + rd["rd_asset"])
                if invested + rd["rd_asset"] > 0 else None,
            })
        return result

    def _pipeline_rnpv(self, ctx: ValuationContext) -> Dict[str, Any]:
        pipeline = ctx.inputs.get("pipeline_data")
        if not pipeline:
            return not_applicable("Provide pipeline_data (programmes with phase, peak sales and years to "
                                  "launch) to value the development pipeline.")
        rate = ctx.params["discount_rate"]
        programmes: List[Dict[str, Any]] = []
        total = 0.0
        for item in pipeline:
            peak = to_float(item.get("peak_sales"))
            if not peak or peak <= 0:
                continue
            phase = str(item.get("phase", "phase 2")).lower()
            probability = to_float(item.get("probability")) or PHASE_PROBABILITIES.get(phase, 0.15)
            launch = max(int(to_float(item.get("years_to_launch")) or 3), 0)
            margin = to_float(item.get("margin")) or 0.35
            exclusivity = int(to_float(item.get("exclusivity_years")) or 10)
            npv = 0.0
            for year in range(1, exclusivity + 1):
                ramp = min(year / 4, 1.0)  # 4-year ramp to peak sales
                cash = peak * ramp * margin * (1 - ctx.tax_rate)
                npv += cash / (1 + rate) ** (launch + year)
            rnpv = npv * probability
            total += rnpv
            programmes.append({"name": item.get("name", "Programme"), "phase": phase, "peak_sales": peak,
                               "probability": probability, "years_to_launch": launch, "npv": npv, "rnpv": rnpv})
        if not programmes:
            return not_applicable("pipeline_data contains no programme with positive peak sales.")
        # Pipeline value is incremental to the existing business: add it to the base DCF where available.
        base = fcff_dcf(ctx, margin_years=3)
        base_ev = base.get("enterprise_value") if base.get("value_per_share") is not None else 0.0
        bridge = equity_bridge((base_ev or 0.0) + total, ctx.capital)
        return {**bridge, "value_per_share": ctx.per_share(bridge["equity_value"]), "pipeline_rnpv": total,
                "base_business_value": base_ev, "programmes": programmes, "discount_rate": rate}

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def rd_adjusted_dcf(self, ticker: str, financial_data: Dict, subsector: str = "pharmaceuticals") -> Dict:
        return self.run_method("rd_adjusted_dcf", ticker, financial_data, subsector)

    def healthcare_multiples_valuation(self, ticker: str, financial_data: Dict,
                                       subsector: str = "pharmaceuticals") -> Dict:
        return self.run_method("relative", ticker, financial_data, subsector)

    def pipeline_valuation(self, ticker: str, financial_data: Dict, pipeline_data: Optional[List[Dict]] = None,
                           **kwargs) -> Dict:
        return self.run_method("pipeline_rnpv", ticker, financial_data, kwargs.pop("subsector", None),
                               pipeline_data=pipeline_data, **kwargs)
