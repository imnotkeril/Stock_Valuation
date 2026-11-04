"""
Consumer sector valuation (Consumer Discretionary and Consumer Staples).

Methods (keys of ``RetailSectorValuation.METHODS``):

* ``dcf`` - FCFF DCF on the normalised (3-year average) free cash flow margin.
* ``relative`` - EV/EBITDA, P/E and EV/Sales against subsector benchmarks.
* ``unit_economics`` - store / restaurant unit economics; needs ``store_count``.
* ``sum_of_parts`` - segment multiples; needs the ``segments`` keyword argument.
* ``dividend_discount`` - two-stage DDM (staples and grocers).
"""

import logging
from typing import Any, Dict, Optional

from StockAnalysisSystem.src.config import normalize_sector
from StockAnalysisSystem.src.utils.financial_data import safe_divide, to_float
from StockAnalysisSystem.src.valuation.dcf_models import AdvancedDCFValuation
from StockAnalysisSystem.src.valuation.sector_specific.common import (
    SectorValuationMixin,
    ValuationContext,
    common_derivations,
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
    sum_of_parts,
)

logger = logging.getLogger(__name__)

_STORE_BASED = ["general_retail", "specialty_retail", "apparel", "restaurants", "automotive_dealers", "grocery"]
_DIVIDEND_PAYERS = ["consumer_staples", "grocery"]

#: Benchmark (EV/EBITDA, P/E, EV/Sales) by subsector - typical peer medians.
_BENCHMARK_MULTIPLES = {
    "general_retail": (10.0, 18.0, 0.7),
    "specialty_retail": (11.0, 18.0, 1.2),
    "ecommerce": (18.0, 35.0, 2.5),
    "apparel": (12.0, 20.0, 1.8),
    "restaurants": (15.0, 24.0, 3.0),
    "automotive_dealers": (8.0, 11.0, 0.4),
    "consumer_services": (11.0, 18.0, 2.0),
    "household_durables": (8.0, 12.0, 1.0),
    "grocery": (8.0, 15.0, 0.35),
    "consumer_staples": (14.0, 21.0, 2.8),
}


class RetailSectorValuation(SectorValuationMixin, AdvancedDCFValuation):
    """Valuation model for retailers, restaurants, consumer services and consumer staples."""

    SECTOR = "Consumer Discretionary"
    SUBSECTORS = {
        "general_retail": "General Merchandise & Discount Stores",
        "specialty_retail": "Specialty & Home Improvement Retail",
        "ecommerce": "E-commerce / Internet Retail",
        "apparel": "Apparel, Footwear & Luxury",
        "restaurants": "Restaurants",
        "automotive_dealers": "Auto Dealers & Auto Parts Retail",
        "consumer_services": "Leisure, Travel & Consumer Services",
        "household_durables": "Home Furnishings, Durables & Homebuilders",
        "grocery": "Grocery, Drug Stores & Food Distribution",
        "consumer_staples": "Packaged Food, Beverages & Household Products",
    }
    METHODS = {
        "dcf": "Retail DCF (FCFF)",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
        "unit_economics": "Store / Unit Economics",
        "sum_of_parts": "Sum of the Parts",
        "dividend_discount": "Dividend Discount Model",
    }
    SUBSECTOR_METHODS = {
        "general_retail": ["dcf", "relative", "unit_economics", "sum_of_parts"],
        "specialty_retail": ["dcf", "relative", "unit_economics", "sum_of_parts"],
        "ecommerce": ["dcf", "relative", "sum_of_parts"],
        "apparel": ["dcf", "relative", "unit_economics", "sum_of_parts"],
        "restaurants": ["dcf", "relative", "unit_economics"],
        "automotive_dealers": ["dcf", "relative", "unit_economics"],
        "consumer_services": ["dcf", "relative", "sum_of_parts"],
        "household_durables": ["dcf", "relative"],
        "grocery": ["dcf", "relative", "unit_economics", "dividend_discount"],
        "consumer_staples": ["dcf", "relative", "dividend_discount", "sum_of_parts"],
    }
    DEFAULT_METHOD_WEIGHTS = {"dcf": 0.5, "relative": 0.3, "unit_economics": 0.2, "sum_of_parts": 0.2,
                              "dividend_discount": 0.15}
    SUBSECTOR_WEIGHTS = {
        "ecommerce": {"dcf": 0.6, "relative": 0.4, "sum_of_parts": 0.2},
        "grocery": {"dcf": 0.45, "relative": 0.35, "unit_economics": 0.1, "dividend_discount": 0.1},
        "consumer_staples": {"dcf": 0.45, "relative": 0.35, "dividend_discount": 0.2, "sum_of_parts": 0.2},
    }
    PARAMETERS = {
        **core_parameters(discount_default=0.09, years_default=5),
        **multiple_parameters(),
        "store_count": number_param(
            "Store / unit count", 0, 100000, 0,
            "Number of stores or restaurants (0 = unknown; required for unit economics).",
            _STORE_BASED, step=1, kind="int", optional=True),
        "new_stores_per_year": number_param(
            "New stores per year", 0, 5000, 0, "Planned net openings per year over the forecast horizon.",
            _STORE_BASED, step=1, kind="int"),
        "store_build_cost": number_param(
            "Investment per new store", 0, 2e8, 0,
            "Capex plus inventory to open one store (0 = net PPE and inventory per existing store).",
            _STORE_BASED, step=100000, kind="currency", optional=True),
        "same_store_growth": percent_param(
            "Long-run same-store sales growth", -0.05, 0.08, 0.02,
            "Perpetual growth of the mature store base (defaults to the terminal growth rate).", _STORE_BASED),
        "cost_of_equity": percent_param(
            "Cost of equity", 0.04, 0.20, 0.08, "Discount rate for dividends (defaults to CAPM).", _DIVIDEND_PAYERS),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(subsector_multiple_defaults(_BENCHMARK_MULTIPLES))
    SUBSECTOR_KEYWORDS = (
        ("ecommerce", ("internet retail", "e-commerce", "ecommerce", "online retail")),
        ("restaurants", ("restaurant",)),
        ("automotive_dealers", ("auto & truck dealer", "auto dealer", "dealership", "auto parts", "automotive")),
        ("apparel", ("apparel", "footwear", "luxury", "textile")),
        ("grocery", ("grocery", "food distribution", "pharmaceutical retail", "drug store")),
        ("general_retail", ("discount store", "department store", "general merchandise", "variety store")),
        ("specialty_retail", ("home improvement", "specialty retail")),
        ("household_durables", ("furnishing", "appliance", "residential construction", "homebuild", "home builder")),
        ("consumer_services", ("lodging", "resort", "casino", "gambling", "travel", "leisure", "recreational",
                               "personal services", "education")),
        ("consumer_staples", ("beverage", "food", "confection", "household", "personal product", "tobacco",
                              "farm product", "brewer", "distiller", "winer", "packaged")),
        ("specialty_retail", ("retail",)),
    )
    SUBSECTOR_ALIASES = {
        "traditional_retail": "general_retail", "omnichannel": "general_retail", "discount": "general_retail",
        "e_commerce": "ecommerce", "luxury": "apparel", "specialty": "specialty_retail",
        "retail": "general_retail", "staples": "consumer_staples", "leisure": "consumer_services",
        "automotive": "automotive_dealers",
    }
    EXTRA_INPUTS = ("segments", "corporate_costs", "avg_store_value")

    # ------------------------------------------------------------------
    # Contract hooks
    # ------------------------------------------------------------------
    def _fallback_subsector(self, company_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        if normalize_sector(company_info.get("sector")) == "Consumer Staples":
            return "consumer_staples"
        return "general_retail"

    def _sector_for_subsector(self, subsector: str) -> str:
        return "Consumer Staples" if subsector in ("grocery", "consumer_staples") else "Consumer Discretionary"

    def _method_runners(self):
        return {
            "dcf": self._dcf,
            "relative": self._relative,
            "unit_economics": self._unit_economics,
            "sum_of_parts": self._sum_of_parts,
            "dividend_discount": dividend_discount,
        }

    def _parameter_derivations(self, ctx: ValuationContext):
        return {**common_derivations(ctx), "same_store_growth": lambda: ctx.params.get("terminal_growth")}

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def _dcf(self, ctx: ValuationContext) -> Dict[str, Any]:
        result = fcff_dcf(ctx, margin_years=3)
        if ctx.subsector in _STORE_BASED and not result.get("note"):
            result["note"] = ("FCFF is after store rent; lease liabilities reported in total debt are deducted "
                              "in the equity bridge.")
        return result

    def _relative(self, ctx: ValuationContext) -> Dict[str, Any]:
        return standard_multiples(ctx)

    def _unit_economics(self, ctx: ValuationContext) -> Dict[str, Any]:
        """
        Mature store base valued as a growing perpetuity of store-level NOPAT net
        of the reinvestment needed for same-store growth (FCF = NOPAT (1 - g/ROIC)),
        plus the NPV of planned openings (investment per store vs. value of a mature store).
        """
        stores = ctx.params.get("store_count")
        if not stores:
            return not_applicable("Provide store_count (number of stores / units) to run the unit-economics valuation.")
        revenue, ebit = latest_metric(ctx, "revenue"), latest_metric(ctx, "ebit")
        if not revenue or revenue <= 0 or ebit is None:
            raise ValueError("Revenue and operating income are required for unit economics")
        nopat = ebit * (1 - ctx.tax_rate)
        if nopat <= 0:
            return not_applicable("Store base is loss-making; unit economics do not support a positive value.",
                                  store_count=stores, operating_income=ebit)
        rate, growth = ctx.params["discount_rate"], ctx.params["same_store_growth"]
        if rate <= growth:
            raise ValueError(f"Discount rate ({rate:.2%}) must exceed same-store growth ({growth:.2%})")

        equity = ctx.latest("balance_sheet", "total_equity") or 0.0
        invested_capital = equity + ctx.capital["total_debt"] - ctx.capital["cash"]
        roic = safe_divide(nopat, invested_capital) if invested_capital > 0 else None
        reinvestment_rate = min(max(growth, 0.0) / roic, 1.0) if roic and roic > 0 else 0.0
        nopat_per_store = nopat / stores
        value_per_store = to_float(ctx.inputs.get("avg_store_value")) or (
            nopat_per_store * (1 - reinvestment_rate) * (1 + growth) / (rate - growth))
        mature_value = value_per_store * stores

        build_cost = ctx.params.get("store_build_cost")
        if build_cost is None:
            assets = (ctx.latest("balance_sheet", "net_ppe") or 0.0) + (ctx.latest("balance_sheet", "inventory") or 0.0)
            build_cost = assets / stores if assets > 0 else None
        new_stores = ctx.params.get("new_stores_per_year") or 0
        years = ctx.params["forecast_years"]
        # A store opened during year t is assumed to reach mature economics from year t+1.
        new_store_npv = value_per_store / (1 + rate) - build_cost if build_cost else None
        pipeline_value = (sum(new_stores * new_store_npv / (1 + rate) ** t for t in range(1, years + 1))
                          if new_stores and new_store_npv is not None else 0.0)
        bridge = equity_bridge(mature_value + pipeline_value, ctx.capital)
        ebitda, da = latest_metric(ctx, "ebitda"), latest_metric(ctx, "da") or 0.0
        return {
            **bridge,
            "value_per_share": ctx.per_share(bridge["equity_value"]),
            "store_count": stores,
            "revenue_per_store": revenue / stores,
            "ebitda_per_store": safe_divide(ebitda, stores),
            "nopat_per_store": nopat_per_store,
            "roic": roic,
            "reinvestment_rate": reinvestment_rate,
            "value_per_store": value_per_store,
            "mature_store_value": mature_value,
            "store_build_cost": build_cost,
            "cash_on_cash_return": safe_divide(nopat_per_store, build_cost),
            "payback_years": safe_divide(build_cost, nopat_per_store + da / stores),
            "new_stores_per_year": new_stores,
            "new_store_npv": new_store_npv,
            "store_pipeline_value": pipeline_value,
            "discount_rate": rate,
            "same_store_growth": growth,
        }

    def _sum_of_parts(self, ctx: ValuationContext) -> Dict[str, Any]:
        segments = ctx.inputs.get("segments")
        if not segments:
            return not_applicable("Provide segments=[{'name', 'ebitda', 'ev_ebitda'} | {'name', 'revenue', "
                                  "'ev_sales'} | {'name', 'value'}] for a sum-of-the-parts valuation.")
        return sum_of_parts(ctx, segments, ctx.inputs.get("corporate_costs"))

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def detect_retail_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None) -> str:
        """Subsector key for ``ticker`` (alias of :meth:`detect_subsector`)."""
        return self.detect_subsector(ticker, financial_data)

    def retail_dcf_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                             subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Retail FCFF DCF (method ``dcf``)."""
        return self.run_method("dcf", ticker, financial_data, subsector, **kwargs)

    def retail_relative_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                                  subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Multiples valuation (method ``relative``)."""
        return self.run_method("relative", ticker, financial_data, subsector, **kwargs)

    def store_based_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                              store_count: Optional[int] = None, avg_store_value: Optional[float] = None,
                              **kwargs) -> Dict[str, Any]:
        """Store / unit economics (method ``unit_economics``)."""
        return self.run_method("unit_economics", ticker, financial_data, kwargs.pop("subsector", None),
                               store_count=store_count, avg_store_value=avg_store_value, **kwargs)

    def retail_sum_of_parts(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                            segments: Any = None, **kwargs) -> Dict[str, Any]:
        """Sum of the parts from segment data (method ``sum_of_parts``)."""
        return self.run_method("sum_of_parts", ticker, financial_data, kwargs.pop("subsector", None),
                               segments=segments, **kwargs)

    def apply_retail_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                               subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Full valuation (``get_valuation``) with the legacy summary keys added."""
        result = self.get_valuation(ticker, financial_data, subsector=subsector, **kwargs)
        return legacy_summary(result, "comprehensive_retail_valuation")

    def get_retail_specific_ratios(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                                   subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Retail operating ratios plus subsector benchmark multiples."""
        groups = ("general", "efficiency", "profitability", "growth", "operational", "valuation")
        try:
            ctx = self._make_context(ticker, financial_data, subsector, kwargs.get("sector"), kwargs)
            ratios = operating_ratios(ctx)
            stores = ctx.params.get("store_count")
            revenue = latest_metric(ctx, "revenue")
            return {
                "general": {"subsector": ctx.subsector, "sector": ctx.sector, "revenue": revenue},
                "efficiency": ratios["efficiency"],
                "profitability": ratios["profitability"],
                "growth": ratios["growth"],
                "operational": {"store_count": stores, "revenue_per_store": safe_divide(revenue, stores)}
                if stores else {},
                "valuation": ratios["valuation"],
                "liquidity": ratios["liquidity"],
                "leverage": ratios["leverage"],
                "investment": ratios["investment"],
                "benchmarks": {k: ctx.params.get(k) for k in ("ev_ebitda_multiple", "pe_multiple",
                                                              "ev_sales_multiple")},
            }
        except Exception as exc:
            logger.warning(f"Retail ratios failed for {ticker}: {exc}")
            return {**{group: {} for group in groups}, "error": str(exc)}
