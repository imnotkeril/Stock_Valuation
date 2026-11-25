"""
Industrials and Materials valuation (also auto manufacturers).

Methods (keys of ``ManufacturingSectorValuation.METHODS``):

* ``dcf`` - FCFF DCF on the cycle-normalised free cash flow margin (all years).
* ``replacement_value`` - adjusted book value: net PP&E grossed up to
  replacement cost plus working capital, less net debt.
* ``earnings_power`` - Greenwald earnings power value: normalised NOPAT / WACC
  (no growth).
* ``relative`` - EV/EBITDA, P/E and EV/Sales against subsector benchmarks.
* ``dividend_discount`` - two-stage DDM.
"""

import logging
from typing import Any, Dict, Optional

from StockAnalysisSystem.src.config import normalize_sector
from StockAnalysisSystem.src.utils.financial_data import safe_divide
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation
from StockAnalysisSystem.src.valuation.sector_specific.common import (
    enable_parametric_sotp,
    SectorValuationMixin,
    ValuationContext,
    average_margin,
    core_parameters,
    dividend_discount,
    equity_bridge,
    fcff_dcf,
    latest_metric,
    legacy_summary,
    merge_defaults,
    multiple_parameters,
    not_applicable,
    number_param,
    operating_ratios,
    percent_param,
    standard_multiples,
    subsector_multiple_defaults,
)

logger = logging.getLogger(__name__)

#: Benchmark (EV/EBITDA, P/E, EV/Sales) by subsector.
_BENCHMARK_MULTIPLES = {
    "industrial_machinery": (13.0, 20.0, 2.2),
    "aerospace_defense": (15.0, 22.0, 2.0),
    "transportation": (10.0, 17.0, 1.6),
    "construction": (11.0, 18.0, 1.0),
    "automotive": (6.0, 8.0, 0.5),
    "chemicals": (10.0, 18.0, 1.8),
    "metals_mining": (6.0, 11.0, 1.5),
    "paper_packaging": (8.0, 15.0, 1.1),
}
_MATERIALS = ["chemicals", "metals_mining", "paper_packaging"]


class ManufacturingSectorValuation(SectorValuationMixin, AdvancedDCFValuation):
    """Valuation model for industrial, materials and automotive manufacturers."""

    SECTOR = "Industrials"
    SUBSECTORS = {
        "industrial_machinery": "Industrial Machinery & Equipment",
        "aerospace_defense": "Aerospace & Defense",
        "transportation": "Transportation & Logistics",
        "construction": "Construction & Engineering",
        "automotive": "Automobiles & Components",
        "chemicals": "Chemicals",
        "metals_mining": "Metals & Mining",
        "paper_packaging": "Paper & Packaging",
    }
    METHODS = {
        "dcf": "Cycle-Normalised DCF",
        "replacement_value": "Replacement Value",
        "earnings_power": "Earnings Power Value",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
        "dividend_discount": "Dividend Discount Model",
    }
    SUBSECTOR_METHODS = {
        "metals_mining": ["dcf", "replacement_value", "earnings_power", "relative"],
        "automotive": ["dcf", "replacement_value", "earnings_power", "relative"],
    }
    DEFAULT_METHOD_WEIGHTS = {"dcf": 0.4, "relative": 0.3, "earnings_power": 0.2, "replacement_value": 0.1,
                              "dividend_discount": 0.1}
    SUBSECTOR_WEIGHTS = {
        "metals_mining": {"dcf": 0.3, "relative": 0.25, "earnings_power": 0.2, "replacement_value": 0.25},
        "automotive": {"dcf": 0.3, "relative": 0.3, "earnings_power": 0.2, "replacement_value": 0.2},
    }
    PARAMETERS = {
        **core_parameters(discount_default=0.09, years_default=5),
        **multiple_parameters(),
        "replacement_cost_factor": number_param(
            "Replacement cost / book PP&E", 1.0, 3.0, 1.5,
            "Cost to rebuild the asset base relative to its depreciated book value.", step=0.1),
        "normalized_margin": percent_param(
            "Normalised EBIT margin", 0.0, 0.40, 0.12,
            "Mid-cycle operating margin for earnings power value (defaults to the historical average).",
            step=0.005),
        "cost_of_equity": percent_param(
            "Cost of equity", 0.04, 0.20, 0.09, "Discount rate for dividends (CAPM default)."),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(
        subsector_multiple_defaults(_BENCHMARK_MULTIPLES),
        {"metals_mining": {"replacement_cost_factor": 1.8}, "automotive": {"replacement_cost_factor": 1.3}},
    )
    SUBSECTOR_KEYWORDS = (
        ("aerospace_defense", ("aerospace", "defense")),
        ("automotive", ("auto manufacturer", "auto parts", "automobile", "truck")),
        ("transportation", ("railroad", "trucking", "airline", "freight", "logistics", "marine shipping",
                            "integrated freight", "transportation")),
        ("industrial_machinery", ("machinery",)),
        ("construction", ("engineering & construction", "construction", "infrastructure", "building products")),
        ("metals_mining", ("gold", "silver", "copper", "steel", "aluminum", "mining", "metals")),
        ("chemicals", ("chemical", "agricultural inputs", "industrial gases")),
        ("paper_packaging", ("packaging", "paper", "lumber", "containers")),
        ("industrial_machinery", ("machinery", "equipment", "electrical", "conglomerate", "tools", "industrial")),
    )
    SUBSECTOR_ALIASES = {"machinery": "industrial_machinery", "general": "industrial_machinery",
                         "aerospace": "aerospace_defense", "mining": "metals_mining", "auto": "automotive",
                         "packaging": "paper_packaging", "materials": "chemicals"}

    def _fallback_subsector(self, company_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        return "chemicals" if normalize_sector(company_info.get("sector")) == "Materials" else "industrial_machinery"

    def _sector_for_subsector(self, subsector: str) -> str:
        if subsector in _MATERIALS:
            return "Materials"
        return "Consumer Discretionary" if subsector == "automotive" else "Industrials"

    def _method_runners(self):
        return {
            "dcf": lambda ctx: fcff_dcf(ctx, margin_years=None, margin_label="cycle-normalised"),
            "replacement_value": self._replacement_value,
            "earnings_power": self._earnings_power,
            "relative": standard_multiples,
            "dividend_discount": dividend_discount,
        }

    def _parameter_derivations(self, ctx: ValuationContext):
        from StockAnalysisSystem.src.valuation.sector_specific.common import common_derivations

        return {**common_derivations(ctx), "normalized_margin": lambda: average_margin(ctx, "ebit")}

    def _replacement_value(self, ctx: ValuationContext) -> Dict[str, Any]:
        ppe = ctx.latest("balance_sheet", "net_ppe")
        if not ppe or ppe <= 0:
            return not_applicable("Net PP&E is not available.")
        factor = ctx.params["replacement_cost_factor"]
        working_capital = ctx.latest("balance_sheet", "working_capital") or 0.0
        cash = ctx.capital["cash"]
        operating_wc = working_capital - cash  # cash is added back separately in the bridge
        intangibles = ctx.latest("balance_sheet", "goodwill_and_intangibles") or 0.0
        asset_value = ppe * factor + operating_wc + intangibles * 0.5
        bridge = equity_bridge(asset_value, ctx.capital)
        return {**bridge, "value_per_share": ctx.per_share(bridge["equity_value"]), "net_ppe": ppe,
                "replacement_cost_factor": factor, "replacement_cost": ppe * factor,
                "operating_working_capital": operating_wc, "intangibles_at_50pct": intangibles * 0.5,
                "note": "Intangibles are counted at 50 % of book value."}

    def _earnings_power(self, ctx: ValuationContext) -> Dict[str, Any]:
        revenue = latest_metric(ctx, "revenue")
        margin = ctx.params.get("normalized_margin")
        if not revenue or revenue <= 0 or margin is None:
            return not_applicable("Revenue and an operating margin are required.")
        if margin <= 0:
            return not_applicable("Normalised operating margin is not positive.")
        nopat = revenue * margin * (1 - ctx.tax_rate)
        wacc = ctx.params["discount_rate"]
        bridge = equity_bridge(nopat / wacc, ctx.capital)
        return {**bridge, "value_per_share": ctx.per_share(bridge["equity_value"]), "revenue": revenue,
                "normalized_margin": margin, "normalized_nopat": nopat, "discount_rate": wacc,
                "note": "Assumes no growth: value of current, mid-cycle earnings power only."}

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def detect_manufacturing_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None) -> str:
        return self.detect_subsector(ticker, financial_data)

    def manufacturing_dcf_valuation(self, ticker, financial_data=None, subsector=None, **kwargs):
        return self.run_method("dcf", ticker, financial_data, subsector, **kwargs)

    def manufacturing_replacement_value(self, ticker, financial_data=None, subsector=None, **kwargs):
        return self.run_method("replacement_value", ticker, financial_data, subsector, **kwargs)

    def manufacturing_relative_valuation(self, ticker, financial_data=None, subsector=None, **kwargs):
        return self.run_method("relative", ticker, financial_data, subsector, **kwargs)

    def manufacturing_earnings_power_value(self, ticker, financial_data=None, subsector=None, **kwargs):
        return self.run_method("earnings_power", ticker, financial_data, subsector, **kwargs)

    def apply_manufacturing_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                                      subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        result = self.get_valuation(ticker, financial_data, subsector=subsector, **kwargs)
        return legacy_summary(result, "comprehensive_manufacturing_valuation")

    def get_manufacturing_specific_ratios(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                                          subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        try:
            ctx = self._make_context(ticker, financial_data, subsector, kwargs.get("sector"), kwargs)
            ratios = operating_ratios(ctx)
            ppe, revenue = ctx.latest("balance_sheet", "net_ppe"), latest_metric(ctx, "revenue")
            ratios["capital_intensity"] = {"asset_turnover_ppe": safe_divide(revenue, ppe),
                                           "capex_to_revenue": average_margin(ctx, "capex")}
            ratios["general"] = {"subsector": ctx.subsector, "sector": ctx.sector}
            return ratios
        except Exception as exc:
            logger.warning(f"Manufacturing ratios failed for {ticker}: {exc}")
            return {"error": str(exc)}


enable_parametric_sotp(ManufacturingSectorValuation, ("Core manufacturing", "Services & aftermarket"), (10.0, 14.0))
