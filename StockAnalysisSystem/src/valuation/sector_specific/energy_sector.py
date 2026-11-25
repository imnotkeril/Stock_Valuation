"""
Energy and utilities valuation.

Subsectors (keys of ``EnergySectorValuation.SUBSECTORS``): ``upstream``,
``integrated``, ``midstream``, ``downstream``, ``utilities`` and
``renewables``. The Utilities sector is routed here with ``utilities``.

Methods (keys of ``METHODS``):

* ``dcf`` - FCFF DCF on the normalised (mid-cycle) free cash flow margin.
* ``reserve_nav`` - upstream NAV: cash flow per barrel at the long-term oil
  price over the remaining reserve life.
* ``sum_of_parts`` - integrated majors: upstream on NAV, the rest on EV/EBITDA.
* ``contracted_dcf`` - midstream: contracted EBITDA over the contract life,
  then re-contracting at a haircut.
* ``regulated_asset_base`` - utilities: value of the rate base earning the
  allowed ROE versus the cost of equity, plus rate base growth.
* ``project_dcf`` - renewables: operating fleet cash flows driven by capacity
  factor and PPA coverage.
* ``relative`` - EV/EBITDA, P/E and EV/Sales against subsector benchmarks.
* ``dividend_discount`` - two-stage DDM (midstream, utilities, integrated).
"""

import logging
from typing import Any, Dict, Optional

from StockAnalysisSystem.src.utils.financial_data import safe_divide
from StockAnalysisSystem.src.valuation.dcf_models import SectorSpecificDCF
from StockAnalysisSystem.src.valuation.sector_specific.common import (
    SectorValuationMixin,
    ValuationContext,
    average_margin,
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
    percent_param,
    standard_multiples,
    subsector_multiple_defaults,
)

logger = logging.getLogger(__name__)

#: Benchmark (EV/EBITDA, P/E, EV/Sales) by subsector.
_BENCHMARK_MULTIPLES = {
    "upstream": (5.5, 11.0, 2.2),
    "integrated": (6.0, 12.0, 1.1),
    "midstream": (10.0, 16.0, 2.8),
    "downstream": (6.0, 10.0, 0.3),
    "utilities": (11.0, 18.0, 3.0),
    "renewables": (13.0, 25.0, 6.0),
}

#: Oil price embedded in reported results, used to rescale upstream cash flows.
_REFERENCE_OIL_PRICE = 75.0
#: Share of integrated-major EBITDA attributed to upstream when no segments are given.
_INTEGRATED_UPSTREAM_SHARE = 0.65
#: Value retained after contract expiry for midstream assets (re-contracting haircut).
_RECONTRACT_RETENTION = 0.85


class EnergySectorValuation(SectorValuationMixin, SectorSpecificDCF):
    """Valuation model for oil & gas, utilities and renewable energy companies."""

    SECTOR = "Energy"
    SUBSECTORS = {
        "upstream": "Oil & Gas - Upstream (E&P)",
        "integrated": "Oil & Gas - Integrated",
        "midstream": "Oil & Gas - Midstream (pipelines, storage)",
        "downstream": "Oil & Gas - Refining & Marketing",
        "utilities": "Utilities (regulated)",
        "renewables": "Renewable Energy",
    }
    METHODS = {
        "dcf": "Energy DCF (mid-cycle FCFF)",
        "reserve_nav": "NAV (Reserve-Based)",
        "sum_of_parts": "Sum of the Parts",
        "contracted_dcf": "Contracted Cash Flow DCF",
        "regulated_asset_base": "Regulated Asset Base",
        "project_dcf": "Renewable Fleet DCF",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
        "dividend_discount": "Dividend Discount Model",
    }
    SUBSECTOR_METHODS = {
        "upstream": ["dcf", "reserve_nav", "relative"],
        "integrated": ["dcf", "sum_of_parts", "relative", "dividend_discount"],
        "midstream": ["dcf", "contracted_dcf", "dividend_discount", "relative"],
        "downstream": ["dcf", "relative"],
        "utilities": ["dcf", "regulated_asset_base", "dividend_discount", "relative"],
        "renewables": ["dcf", "project_dcf", "relative"],
    }
    DEFAULT_METHOD_WEIGHTS = {"dcf": 0.4, "relative": 0.25, "dividend_discount": 0.15, "reserve_nav": 0.35,
                              "sum_of_parts": 0.3, "contracted_dcf": 0.35, "regulated_asset_base": 0.35,
                              "project_dcf": 0.35}
    SUBSECTOR_WEIGHTS = {
        "downstream": {"dcf": 0.6, "relative": 0.4},
        "utilities": {"dcf": 0.25, "regulated_asset_base": 0.4, "dividend_discount": 0.2, "relative": 0.15},
    }
    PARAMETERS = {
        **core_parameters(discount_default=0.09, terminal_default=0.01, years_default=10, growth_low=-0.15),
        **multiple_parameters(),
        "oil_price": number_param(
            "Long-term oil price ($/bbl)", 40, 100, 65,
            "Mid-cycle Brent price; cash flows are rescaled from the reference price of results.",
            ["upstream", "integrated"], step=5, kind="currency"),
        "reserve_life": number_param(
            "Reserve life (years)", 5, 20, 12, "Years of production at current rates (R/P ratio).",
            ["upstream", "integrated"], step=1, kind="int"),
        "contract_coverage": percent_param(
            "Contract coverage", 0.5, 1.0, 0.8, "Share of capacity under long-term, fee-based contracts.",
            ["midstream"], step=0.05),
        "contract_duration": number_param(
            "Average contract duration (years)", 2, 15, 8, "Weighted remaining life of the contracts.",
            ["midstream"], step=1, kind="int"),
        "allowed_roe": percent_param(
            "Allowed return on equity", 0.07, 0.12, 0.095,
            "Regulated return approved by the utility commissions.", ["utilities"], step=0.001),
        "rate_base_growth": percent_param(
            "Rate base growth", 0.01, 0.08, 0.04, "Annual growth of the regulated asset base.",
            ["utilities"], step=0.005),
        "capacity_factor": percent_param(
            "Capacity factor", 0.2, 0.6, 0.35,
            "Actual output as a share of theoretical maximum; scales fleet cash flows.", ["renewables"], step=0.01),
        "ppa_coverage": percent_param(
            "PPA coverage", 0.0, 1.0, 0.8,
            "Share of output sold under power purchase agreements (lower risk premium).", ["renewables"],
            step=0.05),
        "cost_of_equity": percent_param(
            "Cost of equity", 0.04, 0.20, 0.08, "Discount rate for dividends and regulated equity (CAPM default).",
            ["integrated", "midstream", "utilities"]),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(
        subsector_multiple_defaults(_BENCHMARK_MULTIPLES),
        {"utilities": {"discount_rate": 0.07, "terminal_growth": 0.02},
         "midstream": {"discount_rate": 0.08, "terminal_growth": 0.015},
         "renewables": {"terminal_growth": 0.02, "revenue_growth": 0.08}},
    )
    SUBSECTOR_KEYWORDS = (
        ("upstream", ("e&p", "exploration", "oil & gas production")),
        ("integrated", ("integrated",)),
        ("midstream", ("midstream", "pipeline", "storage")),
        ("downstream", ("refining", "marketing")),
        ("renewables", ("renewable", "solar", "wind")),
        ("utilities", ("utilities", "utility", "electric", "gas utilities", "water", "power producer")),
        ("upstream", ("drilling", "equipment & services", "oil & gas", "coal")),
    )
    SUBSECTOR_ALIASES = {
        "oil_gas": "integrated", "oil_gas_upstream": "upstream", "oil_gas_integrated": "integrated",
        "oil_gas_midstream": "midstream", "oil_gas_downstream": "downstream", "utility": "utilities",
        "regulated": "utilities", "renewable": "renewables", "solar": "renewables", "wind": "renewables",
        "refining": "downstream", "coal": "upstream", "Oil & Gas": "integrated",
        "Renewable Energy": "renewables",
    }
    EXTRA_INPUTS = ("upstream_ebitda_share",)

    # ------------------------------------------------------------------
    # Contract hooks
    # ------------------------------------------------------------------
    def _fallback_subsector(self, company_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        from StockAnalysisSystem.src.config import normalize_sector

        return "utilities" if normalize_sector(company_info.get("sector")) == "Utilities" else "integrated"

    def _sector_for_subsector(self, subsector: str) -> str:
        return "Utilities" if subsector == "utilities" else "Energy"

    def _method_runners(self):
        return {
            "dcf": self._dcf,
            "reserve_nav": self._reserve_nav,
            "sum_of_parts": self._sum_of_parts,
            "contracted_dcf": self._contracted_dcf,
            "regulated_asset_base": self._regulated_asset_base,
            "project_dcf": self._project_dcf,
            "relative": standard_multiples,
            "dividend_discount": dividend_discount,
        }

    def _parameter_derivations(self, ctx: ValuationContext):
        derived = common_derivations(ctx)
        derived.pop("revenue_growth")  # energy revenue tracks commodity prices; use the defaults instead
        return derived

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def _dcf(self, ctx: ValuationContext) -> Dict[str, Any]:
        # Use every available year to normalise across the commodity cycle.
        return fcff_dcf(ctx, margin_years=None, margin_label="mid-cycle")

    def _cash_flow_per_year(self, ctx: ValuationContext) -> Optional[float]:
        """Mid-cycle NOPAT: after-tax EBIT, with maintenance capex assumed equal to D&A."""
        ebitda = latest_metric(ctx, "ebitda")
        margin = average_margin(ctx, "ebitda")
        revenue = latest_metric(ctx, "revenue")
        if margin is not None and revenue:
            ebitda = margin * revenue
        if ebitda is None:
            return None
        da = latest_metric(ctx, "da") or 0.0
        return (ebitda - da) * (1 - ctx.tax_rate)

    def _nav_of_production(self, ctx: ValuationContext, share: float) -> Dict[str, Any]:
        annual = self._cash_flow_per_year(ctx)
        if annual is None or annual <= 0:
            return not_applicable("Operating cash flow is not positive; a reserve NAV is not meaningful.")
        price, life = ctx.params["oil_price"], ctx.params["reserve_life"]
        rate = ctx.params["discount_rate"]
        price_factor = price / _REFERENCE_OIL_PRICE
        # Cash margins are levered to the oil price: half of revenue is roughly fixed cost.
        scaled = annual * share * max(1 + 2 * (price_factor - 1), 0.0)
        annuity = (1 - (1 + rate) ** -life) / rate
        return {
            "enterprise_value": scaled * annuity, "annual_cash_flow": scaled, "oil_price": price,
            "reference_oil_price": _REFERENCE_OIL_PRICE, "reserve_life": life, "discount_rate": rate,
            "annuity_factor": annuity,
        }

    def _reserve_nav(self, ctx: ValuationContext) -> Dict[str, Any]:
        nav = self._nav_of_production(ctx, 1.0)
        if "enterprise_value" not in nav:
            return nav
        bridge = equity_bridge(nav["enterprise_value"], ctx.capital)
        return {**nav, **bridge, "value_per_share": ctx.per_share(bridge["equity_value"])}

    def _sum_of_parts(self, ctx: ValuationContext) -> Dict[str, Any]:
        share = ctx.inputs.get("upstream_ebitda_share") or _INTEGRATED_UPSTREAM_SHARE
        upstream = self._nav_of_production(ctx, share)
        if "enterprise_value" not in upstream:
            return upstream
        ebitda = latest_metric(ctx, "ebitda")
        if not ebitda or ebitda <= 0:
            return not_applicable("EBITDA is required for the downstream / chemicals segments.")
        downstream_multiple = _BENCHMARK_MULTIPLES["downstream"][0]
        downstream_ev = ebitda * (1 - share) * downstream_multiple
        bridge = equity_bridge(upstream["enterprise_value"] + downstream_ev, ctx.capital)
        return {
            **bridge, "value_per_share": ctx.per_share(bridge["equity_value"]),
            "segments": [
                {"segment": "Upstream (reserve NAV)", "ebitda_share": share, "value": upstream["enterprise_value"]},
                {"segment": "Downstream & chemicals (EV/EBITDA)", "ebitda_share": 1 - share,
                 "multiple": downstream_multiple, "value": downstream_ev},
            ],
            "oil_price": ctx.params["oil_price"], "reserve_life": ctx.params["reserve_life"],
            "note": "Segment split is estimated; pass upstream_ebitda_share for reported segment data.",
        }

    def _contracted_dcf(self, ctx: ValuationContext) -> Dict[str, Any]:
        ebitda = latest_metric(ctx, "ebitda")
        if not ebitda or ebitda <= 0:
            return not_applicable("Positive EBITDA is required for a contracted cash flow DCF.")
        capex = latest_metric(ctx, "capex") or 0.0
        da = latest_metric(ctx, "da") or 0.0
        maintenance = min(capex, da) if capex else da
        cash = (ebitda - da) * (1 - ctx.tax_rate) + da - maintenance
        coverage, duration = ctx.params["contract_coverage"], ctx.params["contract_duration"]
        rate, growth = ctx.params["discount_rate"], ctx.params["terminal_growth"]
        if rate <= growth:
            raise ValueError(f"Discount rate ({rate:.2%}) must exceed terminal growth ({growth:.2%})")
        # Contracted volumes are near-certain; uncontracted volumes carry a 3 pp risk premium.
        uncontracted_rate = rate + 0.03
        flows, pv = [], 0.0
        for year in range(1, duration + 1):
            contracted = cash * coverage
            merchant = cash * (1 - coverage)
            value = contracted / (1 + rate) ** year + merchant / (1 + uncontracted_rate) ** year
            flows.append(contracted + merchant)
            pv += value
        terminal_cash = cash * _RECONTRACT_RETENTION * (1 + growth)
        terminal = terminal_cash / (rate - growth) / (1 + rate) ** duration
        bridge = equity_bridge(pv + terminal, ctx.capital)
        return {
            **bridge, "value_per_share": ctx.per_share(bridge["equity_value"]),
            "annual_distributable_cash": cash, "contract_coverage": coverage, "contract_duration": duration,
            "discount_rate": rate, "uncontracted_discount_rate": uncontracted_rate,
            "recontracting_retention": _RECONTRACT_RETENTION, "forecast_fcf": flows,
            "present_value_contract_period": pv, "terminal_value_pv": terminal,
        }

    def _regulated_asset_base(self, ctx: ValuationContext) -> Dict[str, Any]:
        """
        Rate base proxied by net PPE (equity share = book equity / (equity + debt)).
        Equity value = regulated equity x (allowed ROE - g) / (ke - g): the justified
        price-to-book of a perpetuity earning the allowed return and growing with
        the rate base.
        """
        rate_base = ctx.latest("balance_sheet", "net_ppe")
        if not rate_base or rate_base <= 0:
            return not_applicable("Net PP&E (rate base proxy) is not available.")
        equity = ctx.latest("balance_sheet", "total_equity")
        debt = ctx.capital["total_debt"]
        if not equity or equity <= 0:
            return not_applicable("Book equity is not positive.")
        equity_ratio = equity / (equity + debt) if equity + debt > 0 else 1.0
        regulated_equity = rate_base * equity_ratio
        roe, growth = ctx.params["allowed_roe"], ctx.params["rate_base_growth"]
        ke = ctx.params.get("cost_of_equity") or ctx.cost_of_capital["cost_of_equity"]
        sustainable_growth = min(growth, ctx.params["terminal_growth"] + 0.02)
        if ke <= sustainable_growth:
            raise ValueError(f"Cost of equity ({ke:.2%}) must exceed rate base growth ({sustainable_growth:.2%})")
        justified_pb = (roe - sustainable_growth) / (ke - sustainable_growth)
        equity_value = regulated_equity * justified_pb
        return {
            "equity_value": equity_value, "value_per_share": ctx.per_share(equity_value),
            "rate_base": rate_base, "equity_ratio": equity_ratio, "regulated_equity": regulated_equity,
            "allowed_roe": roe, "rate_base_growth": growth, "growth_used": sustainable_growth,
            "cost_of_equity": ke, "justified_price_to_book": justified_pb,
            "earned_roe": safe_divide(ctx.latest("income_statement", "net_income"), equity),
        }

    def _project_dcf(self, ctx: ValuationContext) -> Dict[str, Any]:
        ebitda = latest_metric(ctx, "ebitda")
        if not ebitda or ebitda <= 0:
            return not_applicable("Positive EBITDA is required for a fleet DCF.")
        da = latest_metric(ctx, "da") or 0.0
        cf, ppa = ctx.params["capacity_factor"], ctx.params["ppa_coverage"]
        # Reported results reflect a typical 35 % fleet capacity factor.
        scaled = ebitda * cf / 0.35
        cash = (scaled - da) * (1 - ctx.tax_rate) + da * 0.5  # half of D&A reinvested as repowering
        rate = ctx.params["discount_rate"] + (1 - ppa) * 0.025  # merchant exposure premium
        life, growth = ctx.params["forecast_years"], ctx.params["terminal_growth"]
        if rate <= growth:
            raise ValueError(f"Discount rate ({rate:.2%}) must exceed terminal growth ({growth:.2%})")
        flows = [cash * (1 + growth) ** (year - 1) for year in range(1, life + 1)]
        pv = sum(flow / (1 + rate) ** year for year, flow in enumerate(flows, start=1))
        terminal = flows[-1] * (1 + growth) / (rate - growth) / (1 + rate) ** life
        bridge = equity_bridge(pv + terminal, ctx.capital)
        return {
            **bridge, "value_per_share": ctx.per_share(bridge["equity_value"]),
            "capacity_factor": cf, "ppa_coverage": ppa, "risk_adjusted_discount_rate": rate,
            "fleet_ebitda": scaled, "forecast_fcf": flows, "terminal_value_pv": terminal,
        }

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def value_energy_company(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                             subsector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Full valuation with the legacy summary keys."""
        return legacy_summary(self.get_valuation(ticker, financial_data, subsector=subsector, **kwargs),
                              "energy_sector_valuation")

    def value_upstream_company(self, ticker, financial_data=None, **kwargs):
        return self.value_energy_company(ticker, financial_data, "upstream", **kwargs)

    def value_integrated_oil_gas(self, ticker, financial_data=None, **kwargs):
        return self.value_energy_company(ticker, financial_data, "integrated", **kwargs)

    def value_midstream_company(self, ticker, financial_data=None, **kwargs):
        return self.value_energy_company(ticker, financial_data, "midstream", **kwargs)

    def value_utility_company(self, ticker, financial_data=None, **kwargs):
        return self.value_energy_company(ticker, financial_data, "utilities", **kwargs)

    def value_renewable_company(self, ticker, financial_data=None, **kwargs):
        return self.value_energy_company(ticker, financial_data, "renewables", **kwargs)

    def energy_sector_dcf(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Mid-cycle FCFF DCF (method ``dcf``)."""
        return self.run_method("dcf", ticker, financial_data, kwargs.pop("subsector", None), **kwargs)
