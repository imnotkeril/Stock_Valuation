"""
Communication services valuation (telecom, media, interactive media, entertainment).

Methods (keys of ``CommunicationValuation.METHODS``):

* ``dcf`` - FCFF DCF on the normalised free cash flow margin.
* ``relative`` - EV/EBITDA, P/E and EV/Sales against subsector benchmarks.
* ``subscriber`` - customer lifetime value: subscribers x ARPU x margin /
  (churn + discount - growth); needs ``subscribers`` and ``arpu_monthly``.
* ``user_based`` - engaged users x annual revenue per user x margin,
  capitalised; needs ``users`` and ``revenue_per_user``.
* ``dividend_discount`` - two-stage DDM (telecom).
"""

import logging
from typing import Any, Dict, Optional

from StockAnalysisSystem.src.valuation.base_valuation import BaseValuation
from StockAnalysisSystem.src.valuation.sector_specific.common import (
    enable_parametric_sotp,
    SectorValuationMixin,
    ValuationContext,
    average_margin,
    core_parameters,
    dividend_discount,
    equity_bridge,
    fcff_dcf,
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
    "telecom": (7.0, 12.0, 2.0),
    "media": (9.0, 15.0, 1.8),
    "interactive_media": (14.0, 22.0, 6.0),
    "entertainment": (13.0, 25.0, 3.5),
}
_SUBSCRIPTION = ["telecom", "media", "entertainment"]


class CommunicationValuation(SectorValuationMixin, BaseValuation):
    """Valuation model for communication services companies."""

    SECTOR = "Communication Services"
    SUBSECTORS = {
        "telecom": "Telecommunications",
        "media": "Media & Cable",
        "interactive_media": "Interactive Media (search, social)",
        "entertainment": "Entertainment, Streaming & Games",
    }
    METHODS = {
        "dcf": "Discounted Cash Flow (FCFF)",
        "relative": "Relative Valuation (EV/EBITDA, P/E, EV/Sales)",
        "subscriber": "Subscriber Lifetime Value",
        "user_based": "User-Based Valuation",
        "dividend_discount": "Dividend Discount Model",
    }
    SUBSECTOR_METHODS = {
        "telecom": ["dcf", "relative", "subscriber", "dividend_discount"],
        "media": ["dcf", "relative", "subscriber"],
        "interactive_media": ["dcf", "relative", "user_based"],
        "entertainment": ["dcf", "relative", "subscriber", "user_based"],
    }
    DEFAULT_METHOD_WEIGHTS = {"dcf": 0.45, "relative": 0.3, "subscriber": 0.25, "user_based": 0.25,
                              "dividend_discount": 0.15}
    SUBSECTOR_WEIGHTS = {"telecom": {"dcf": 0.45, "relative": 0.25, "subscriber": 0.15, "dividend_discount": 0.15}}
    PARAMETERS = {
        **core_parameters(discount_default=0.09, years_default=6),
        **multiple_parameters(),
        "subscribers": number_param("Subscribers (millions)", 0, 500, 0,
                                    "Paying subscribers; 0 = unknown (subscriber method not run).",
                                    _SUBSCRIPTION, step=0.5, optional=True),
        "arpu_monthly": number_param("Monthly ARPU ($)", 0, 200, 0, "Average revenue per subscriber per month.",
                                     _SUBSCRIPTION, step=0.5, kind="currency", optional=True),
        "churn_rate": percent_param("Annual churn", 0.0, 0.6, 0.12, "Share of subscribers lost per year.",
                                    _SUBSCRIPTION, step=0.005),
        "users": number_param("Engaged users (millions)", 0, 5000, 0,
                              "Monthly active users; 0 = unknown (user-based method not run).",
                              ["interactive_media", "entertainment"], step=10, optional=True),
        "revenue_per_user": number_param("Annual revenue per user ($)", 0, 500, 0,
                                         "Average annual revenue per engaged user.",
                                         ["interactive_media", "entertainment"], step=1, kind="currency",
                                         optional=True),
        "cost_of_equity": percent_param("Cost of equity", 0.04, 0.20, 0.08,
                                        "Discount rate for dividends (CAPM default).", ["telecom"]),
    }
    SUBSECTOR_PARAMETER_DEFAULTS = merge_defaults(subsector_multiple_defaults(_BENCHMARK_MULTIPLES))
    SUBSECTOR_KEYWORDS = (
        ("telecom", ("telecom", "wireless", "communication equipment")),
        ("interactive_media", ("internet content", "interactive media", "social", "search")),
        ("entertainment", ("entertainment", "electronic gaming", "games", "streaming")),
        ("media", ("media", "broadcasting", "publishing", "advertising", "cable")),
    )
    SUBSECTOR_ALIASES = {"Telecommunications": "telecom", "Media": "media", "Social Media": "interactive_media",
                         "Entertainment": "entertainment", "telecommunications": "telecom",
                         "social_media": "interactive_media", "social": "interactive_media"}
    EXTRA_INPUTS = ("user_metrics",)

    def _translate_legacy_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Accept the legacy ``user_metrics`` dict ({subscribers, arpu, churn, users, ...})."""
        metrics = kwargs.get("user_metrics") or {}
        mapping = {"subscribers": "subscribers", "arpu": "arpu_monthly", "churn": "churn_rate",
                   "users": "users", "arpu_annual": "revenue_per_user"}
        for legacy, name in mapping.items():
            if legacy in metrics and name not in kwargs:
                kwargs[name] = metrics[legacy]
        return kwargs

    def _method_runners(self):
        return {"dcf": lambda ctx: fcff_dcf(ctx, margin_years=3), "relative": standard_multiples,
                "subscriber": self._subscriber, "user_based": self._user_based,
                "dividend_discount": dividend_discount}

    def _operating_margin(self, ctx: ValuationContext) -> Optional[float]:
        margin = average_margin(ctx, "ebitda", 3)
        return margin if margin and margin > 0 else None

    def _subscriber(self, ctx: ValuationContext) -> Dict[str, Any]:
        subscribers, arpu = ctx.params.get("subscribers"), ctx.params.get("arpu_monthly")
        if not subscribers or not arpu:
            return not_applicable("Provide subscribers and monthly ARPU to run the subscriber valuation.")
        margin = self._operating_margin(ctx)
        if margin is None:
            return not_applicable("EBITDA margin is not positive.")
        churn, rate, growth = ctx.params["churn_rate"], ctx.params["discount_rate"], ctx.params["terminal_growth"]
        denominator = churn + rate - growth
        if denominator <= 0:
            raise ValueError("Churn + discount rate must exceed the growth rate")
        annual_cash_per_sub = arpu * 12 * margin * (1 - ctx.tax_rate)
        lifetime_value = annual_cash_per_sub / denominator
        customer_base_value = lifetime_value * subscribers * 1e6
        bridge = equity_bridge(customer_base_value, ctx.capital)
        return {**bridge, "value_per_share": ctx.per_share(bridge["equity_value"]),
                "subscribers_millions": subscribers, "arpu_monthly": arpu, "ebitda_margin": margin,
                "churn_rate": churn, "lifetime_value_per_subscriber": lifetime_value,
                "customer_base_value": customer_base_value}

    def _user_based(self, ctx: ValuationContext) -> Dict[str, Any]:
        users, revenue_per_user = ctx.params.get("users"), ctx.params.get("revenue_per_user")
        if not users or not revenue_per_user:
            return not_applicable("Provide engaged users and annual revenue per user to run this method.")
        margin = self._operating_margin(ctx)
        if margin is None:
            return not_applicable("EBITDA margin is not positive.")
        rate, growth = ctx.params["discount_rate"], ctx.params["terminal_growth"]
        if rate <= growth:
            raise ValueError("Discount rate must exceed terminal growth")
        annual_cash = users * 1e6 * revenue_per_user * margin * (1 - ctx.tax_rate)
        value = annual_cash * (1 + growth) / (rate - growth)
        bridge = equity_bridge(value, ctx.capital)
        return {**bridge, "value_per_share": ctx.per_share(bridge["equity_value"]), "users_millions": users,
                "revenue_per_user": revenue_per_user, "ebitda_margin": margin,
                "value_per_user": value / (users * 1e6)}

    # ------------------------------------------------------------------
    # Backward-compatible entry points
    # ------------------------------------------------------------------
    def standard_dcf_valuation(self, ticker: str, financial_data: Dict, subsector: str = "telecom") -> Dict:
        return self.run_method("dcf", ticker, financial_data, subsector)

    def communication_multiples_valuation(self, ticker: str, financial_data: Dict,
                                          subsector: str = "telecom") -> Dict:
        return self.run_method("relative", ticker, financial_data, subsector)

    def subscriber_based_valuation(self, ticker: str, financial_data: Dict, user_metrics: Dict,
                                   subsector: str = "telecom") -> Dict:
        return self.run_method("subscriber", ticker, financial_data, subsector, user_metrics=user_metrics)

    def user_based_valuation(self, ticker: str, financial_data: Dict, user_metrics: Dict) -> Dict:
        return self.run_method("user_based", ticker, financial_data, "interactive_media", user_metrics=user_metrics)


enable_parametric_sotp(CommunicationValuation, ("Connectivity / core", "Media & content"), (7.0, 12.0))
