"""
Bankruptcy / financial distress models.

* Altman Z-Score family (Altman 1968, 1983, 1995):

  - original (public manufacturers):   Z   = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E,
    D = market value of equity / total liabilities; safe > 2.99, distress < 1.81
  - Z' (private firms):                 Z'  = 0.717A + 0.847B + 3.107C + 0.420D + 0.998E,
    D = book equity / total liabilities; safe > 2.90, distress < 1.23
  - Z'' (non-manufacturers, emerging):  Z'' = 6.56A + 3.26B + 6.72C + 1.05D,
    D = book equity / total liabilities; safe > 2.60, distress < 1.10

  with A = working capital / TA, B = retained earnings / TA, C = EBIT / TA,
  E = sales / TA. The Z-Score is not meaningful for banks, insurers and other
  financial institutions (deposits/policy reserves are operating liabilities,
  "working capital" is undefined) and is reported as not applicable.

* Springate (1978): S = 1.03A + 3.07B + 0.66C + 0.4D with A = WC / TA,
  B = EBIT / TA, C = EBT / current liabilities, D = sales / TA; S < 0.862 = distress.

* Zmijewski (1984): X = -4.336 - 4.513 ROA + 5.679 (TL / TA) + 0.004 (CA / CL),
  probability of distress = 1 / (1 + e^-X); probability >= 0.5 = distress.

All inputs come from the same (most recent) fiscal period.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from StockAnalysisSystem.src.config import COLORS
from StockAnalysisSystem.src.industry.sector_mapping import normalize_sector_name
from StockAnalysisSystem.src.models.ratio_analysis import ebitda_value, free_cash_flow, period_value, total_debt
from StockAnalysisSystem.src.utils import financial_data as fd

logger = logging.getLogger(__name__)

Z_VARIANTS: Dict[str, Dict[str, Any]] = {
    "original": {"name": "Altman Z-Score (public manufacturing)", "coefficients": (1.2, 1.4, 3.3, 0.6, 1.0),
                 "safe": 2.99, "grey": 1.81, "equity": "market",
                 "formula": "Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E (D = market cap / total liabilities)"},
    "private": {"name": "Altman Z'-Score (private firms)", "coefficients": (0.717, 0.847, 3.107, 0.420, 0.998),
                "safe": 2.90, "grey": 1.23, "equity": "book",
                "formula": "Z' = 0.717A + 0.847B + 3.107C + 0.420D + 0.998E (D = book equity / total liabilities)"},
    "non_manufacturing": {"name": "Altman Z''-Score (non-manufacturing)", "coefficients": (6.56, 3.26, 6.72, 1.05, 0.0),
                          "safe": 2.60, "grey": 1.10, "equity": "book",
                          "formula": "Z'' = 6.56A + 3.26B + 6.72C + 1.05D (D = book equity / total liabilities)"},
}
_VARIANT_ALIASES = {"modified": "non_manufacturing", "z''": "non_manufacturing", "emerging": "non_manufacturing",
                    "z'": "private", "manufacturing": "original"}

#: Sector -> default Z variant ("not_applicable" for financial institutions)
SECTOR_Z_VARIANT = {
    "Financials": "not_applicable",
    "Real Estate": "non_manufacturing",
    "Utilities": "non_manufacturing",
    "Communication Services": "non_manufacturing",
    "Technology": "original",
    "Healthcare": "original",
    "Industrials": "original",
    "Materials": "original",
    "Energy": "original",
    "Consumer Discretionary": "original",
    "Consumer Staples": "original",
}

#: Industry keywords that identify non-manufacturing businesses inside "manufacturing" sectors.
NON_MANUFACTURING_INDUSTRY_KEYWORDS = (
    "software", "internet", "services", "retail", "restaurant", "lodging", "hotel", "travel", "leisure",
    "entertainment", "media", "broadcast", "publishing", "advertising", "education", "staffing", "consulting",
    "distribution", "stores", "e-commerce", "gambling", "resorts", "airlines", "trucking", "railroads",
)

SPRINGATE_CUTOFF = 0.862
ZMIJEWSKI_CUTOFF = 0.5

FINANCIAL_SECTOR_NOTE = ("Altman, Springate and Zmijewski models were estimated on non-financial companies; "
                         "for banks and insurers deposits and policy reserves are operating liabilities, so the "
                         "scores are not meaningful. Capital and profitability indicators are used instead.")

RISK_LEVELS = (  # (upper bound of risk score, level, overall_assessment)
    (20, "Low", "safe"),
    (40, "Moderate", "grey"),
    (70, "Elevated", "grey"),
    (math.inf, "High", "distress"),
)

ZONE_POINTS = {"safe": 0.0, "grey": 50.0, "distress": 100.0}


def _div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide only by a strictly positive denominator."""
    return fd.safe_divide(numerator, denominator) if denominator is not None and denominator > 0 else None


class BankruptcyAnalyzer:
    """Bankruptcy risk models and an overall distress risk assessment."""

    def __init__(self):
        self.sector_model_mapping = dict(SECTOR_Z_VARIANT)
        self.risk_thresholds = {
            "z_score": {name: {"safe": v["safe"], "grey": v["grey"]} for name, v in Z_VARIANTS.items()},
            "springate": {"threshold": SPRINGATE_CUTOFF},
            "zmijewski": {"threshold": ZMIJEWSKI_CUTOFF},
        }
        self.risk_colors = {"safe": COLORS["success"], "grey": COLORS["warning"], "distress": COLORS["danger"],
                            "elevated": COLORS["accent"], "not_applicable": COLORS["info"]}

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _inputs(financial_data: Dict) -> Dict[str, Optional[float]]:
        data = financial_data or {}
        inc, bs = data.get("income_statement"), data.get("balance_sheet")
        market = data.get("market_data") or {}
        current_assets = period_value(bs, "current_assets")
        current_liabilities = period_value(bs, "current_liabilities")
        working_capital = (current_assets - current_liabilities
                           if current_assets is not None and current_liabilities is not None
                           else period_value(bs, "working_capital"))
        market_cap = fd.to_float(market.get("market_cap"))
        if market_cap is None:
            price, shares = fd.to_float(market.get("share_price")), fd.to_float(market.get("shares_outstanding"))
            market_cap = price * shares if price and shares else None
        return {
            "total_assets": period_value(bs, "total_assets"),
            "total_liabilities": period_value(bs, "total_liabilities"),
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "working_capital": working_capital,
            "retained_earnings": period_value(bs, "retained_earnings"),
            "book_equity": period_value(bs, "total_equity"),
            "market_cap": market_cap if market_cap and market_cap > 0 else None,
            "ebit": period_value(inc, "ebit"),
            "pretax_income": period_value(inc, "pretax_income"),
            "net_income": period_value(inc, "net_income"),
            "revenue": period_value(inc, "revenue"),
        }

    @staticmethod
    def _missing(inputs: Dict[str, Optional[float]], keys: List[str]) -> List[str]:
        return [k for k in keys if inputs.get(k) is None]

    # --------------------------------------------------------------- Z variant
    def select_z_variant(self, sector: Optional[str], industry: Optional[str] = None,
                         has_market_value: bool = True) -> str:
        """
        Altman variant for a company: ``"not_applicable"`` (financials),
        ``"non_manufacturing"`` (Z''), ``"original"`` (Z, public manufacturers) or
        ``"private"`` (Z', manufacturers without a market value of equity).
        """
        variant = self.sector_model_mapping.get(normalize_sector_name(sector), "non_manufacturing")
        if variant == "original" and industry and any(k in str(industry).lower()
                                                      for k in NON_MANUFACTURING_INDUSTRY_KEYWORDS):
            variant = "non_manufacturing"
        if variant == "original" and not has_market_value:
            variant = "private"
        return variant

    # ------------------------------------------------------------------ models
    def calculate_altman_z_score(self, financial_data: Dict, sector: Optional[str] = None,
                                 variant: Optional[str] = None, industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Altman Z-Score with the variant appropriate for the sector.

        Args:
            financial_data: Canonical bundle (uses ``market_data["market_cap"]`` for the original Z).
            sector: Any sector spelling; defaults to ``company_info["sector"]``.
            variant: Force ``"original"``, ``"private"`` or ``"non_manufacturing"``.
            industry: Refines the variant (defaults to ``company_info["industry"]``).

        Returns:
            ``{"score", "components": {"working_capital_to_total_assets",
            "retained_earnings_to_total_assets", "ebit_to_total_assets",
            "equity_to_total_liabilities", "sales_to_total_assets"}, "inputs",
            "assessment": "safe"|"grey"|"distress"|"not_applicable"|None, "risk_level",
            "description", "interpretation", "color", "thresholds": {"safe", "grey"},
            "model", "variant", "formula", "applicable": bool, "missing": [...]}``.
            ``score`` is None when not applicable or when inputs are missing.
        """
        info = (financial_data or {}).get("company_info") or {}
        inputs = self._inputs(financial_data)
        if variant:
            variant = _VARIANT_ALIASES.get(str(variant).lower(), str(variant).lower())
        else:
            variant = self.select_z_variant(sector or info.get("sector"), industry or info.get("industry"),
                                            inputs["market_cap"] is not None)
        if variant == "not_applicable" or variant == "financial":
            return self._not_applicable("altman_z_score", "Altman Z-Score",
                                        "The Altman Z-Score is not meaningful for financial institutions. "
                                        + FINANCIAL_SECTOR_NOTE)
        if variant not in Z_VARIANTS:
            result = self._empty("altman_z_score", "Altman Z-Score")
            result.update(description=f"Unknown Altman Z variant: {variant}", error=f"unknown variant {variant}")
            return result
        spec = Z_VARIANTS[variant]
        equity_key = "market_cap" if spec["equity"] == "market" else "book_equity"
        required = ["total_assets", "total_liabilities", "working_capital", "retained_earnings", "ebit", equity_key]
        if variant != "non_manufacturing":
            required.append("revenue")
        result = self._base_result("altman_z_score", spec["name"], inputs)
        result.update(variant=variant, formula=spec["formula"],
                      thresholds={"safe": spec["safe"], "grey": spec["grey"]})
        missing = self._missing(inputs, required)
        assets, liabilities = inputs["total_assets"], inputs["total_liabilities"]
        if missing or assets <= 0 or liabilities <= 0:
            return self._insufficient(result, missing or ["total_assets/total_liabilities must be positive"])

        components = {
            "working_capital_to_total_assets": inputs["working_capital"] / assets,
            "retained_earnings_to_total_assets": inputs["retained_earnings"] / assets,
            "ebit_to_total_assets": inputs["ebit"] / assets,
            "equity_to_total_liabilities": inputs[equity_key] / liabilities,
            "sales_to_total_assets": _div(inputs["revenue"], assets),
        }
        score = float(sum(coef * value for coef, value in zip(spec["coefficients"], components.values()) if coef))
        zone = "safe" if score > spec["safe"] else "grey" if score >= spec["grey"] else "distress"
        interpretation = {
            "safe": f"Z = {score:.2f} is above the safe threshold of {spec['safe']:.2f}: low probability of "
                    "bankruptcy within two years.",
            "grey": f"Z = {score:.2f} lies in the grey zone ({spec['grey']:.2f}-{spec['safe']:.2f}): elevated "
                    "uncertainty; monitor leverage and profitability.",
            "distress": f"Z = {score:.2f} is below the distress threshold of {spec['grey']:.2f}: the financial "
                        "profile resembles companies that went bankrupt within two years.",
        }[zone]
        return self._scored(result, score, components, zone, interpretation)

    def calculate_springate_score(self, financial_data: Dict, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Springate S-Score.

        Returns:
            ``{"score", "components": {"working_capital_to_total_assets", "ebit_to_total_assets",
            "ebt_to_current_liabilities", "sales_to_total_assets"}, "assessment": "safe"|"distress"|
            "not_applicable"|None, "risk_level", "description", "interpretation", "color",
            "threshold", "model", "formula", "applicable", "missing"}``.
        """
        info = (financial_data or {}).get("company_info") or {}
        if normalize_sector_name(sector or info.get("sector")) == "Financials":
            return self._not_applicable("springate", "Springate", FINANCIAL_SECTOR_NOTE, threshold=SPRINGATE_CUTOFF)
        inputs = self._inputs(financial_data)
        result = self._base_result("springate", "Springate", inputs)
        result.update(threshold=SPRINGATE_CUTOFF, formula="S = 1.03A + 3.07B + 0.66C + 0.4D")
        missing = self._missing(inputs, ["total_assets", "working_capital", "ebit", "pretax_income",
                                         "current_liabilities", "revenue"])
        assets, current_liabilities = inputs["total_assets"], inputs["current_liabilities"]
        if missing or assets <= 0 or current_liabilities <= 0:
            return self._insufficient(result, missing or ["total_assets/current_liabilities must be positive"])
        components = {
            "working_capital_to_total_assets": inputs["working_capital"] / assets,
            "ebit_to_total_assets": inputs["ebit"] / assets,
            "ebt_to_current_liabilities": inputs["pretax_income"] / current_liabilities,
            "sales_to_total_assets": inputs["revenue"] / assets,
        }
        score = float(np.dot((1.03, 3.07, 0.66, 0.4), list(components.values())))
        zone = "safe" if score >= SPRINGATE_CUTOFF else "distress"
        interpretation = (f"S = {score:.2f} is {'above' if zone == 'safe' else 'below'} the {SPRINGATE_CUTOFF} "
                          f"cut-off: the company is classified as {'healthy' if zone == 'safe' else 'failing'}.")
        return self._scored(result, score, components, zone, interpretation)

    def calculate_zmijewski_score(self, financial_data: Dict, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Zmijewski probit-style score and distress probability.

        Returns:
            ``{"score" (X), "probability", "components": {"roa", "leverage", "liquidity"},
            "assessment": "safe"|"distress"|"not_applicable"|None, "risk_level", "description",
            "interpretation", "color", "threshold", "model", "formula", "applicable", "missing"}``.
        """
        info = (financial_data or {}).get("company_info") or {}
        if normalize_sector_name(sector or info.get("sector")) == "Financials":
            result = self._not_applicable("zmijewski", "Zmijewski", FINANCIAL_SECTOR_NOTE, threshold=ZMIJEWSKI_CUTOFF)
            result["probability"] = None
            return result
        inputs = self._inputs(financial_data)
        result = self._base_result("zmijewski", "Zmijewski", inputs)
        result.update(threshold=ZMIJEWSKI_CUTOFF, probability=None,
                      formula="X = -4.336 - 4.513 ROA + 5.679 TL/TA + 0.004 CA/CL; P = 1 / (1 + e^-X)")
        missing = self._missing(inputs, ["total_assets", "total_liabilities", "net_income", "current_assets",
                                         "current_liabilities"])
        assets, current_liabilities = inputs["total_assets"], inputs["current_liabilities"]
        if missing or assets <= 0 or current_liabilities <= 0:
            return self._insufficient(result, missing or ["total_assets/current_liabilities must be positive"])
        components = {
            "roa": inputs["net_income"] / assets,
            "leverage": inputs["total_liabilities"] / assets,
            "liquidity": inputs["current_assets"] / current_liabilities,
        }
        score = -4.336 - 4.513 * components["roa"] + 5.679 * components["leverage"] + 0.004 * components["liquidity"]
        probability = 1.0 / (1.0 + math.exp(-score))
        zone = "safe" if probability < ZMIJEWSKI_CUTOFF else "distress"
        interpretation = (f"Estimated probability of financial distress {probability:.1%} "
                          f"({'below' if zone == 'safe' else 'above'} the 50 % cut-off).")
        result = self._scored(result, score, components, zone, interpretation)
        result["probability"] = probability
        return result

    # --------------------------------------------------------- overall assessment
    def get_comprehensive_risk_assessment(self, financial_data: Dict, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Combine the distress models and red flags into an overall risk level.

        Risk score (0 = no distress signal, 100 = maximal): mean of applicable
        model zones (safe 0, grey 50, distress 100) plus weighted red-flag
        penalties, capped at 100. Financial institutions are scored on capital
        (equity / assets) and profitability instead of the classic models.

        Returns:
            ``{"models": {"altman_z_score", "springate", "zmijewski": model result},
            "overall_assessment": "safe"|"grey"|"distress"|None,
            "overall_risk_level": "Low"|"Moderate"|"Elevated"|"High"|None,
            "risk_score": float or None, "overall_description": str, "overall_color": str,
            "red_flags": [str], "indicators": {name: float}, "models_used": [str],
            "sector": canonical sector, "notes": [str]}``
        """
        data = financial_data or {}
        info = data.get("company_info") or {}
        canonical = normalize_sector_name(sector or info.get("sector"))
        try:
            models = {
                "altman_z_score": self.calculate_altman_z_score(data, canonical),
                "springate": self.calculate_springate_score(data, canonical),
                "zmijewski": self.calculate_zmijewski_score(data, canonical),
            }
            indicators = self._indicators(data)
            red_flags, penalty = self._red_flags(indicators, financial=canonical == "Financials")
            notes = [m["description"] for m in models.values() if m.get("assessment") == "not_applicable"][:1]
            used = [m["model"] for m in models.values() if m.get("assessment") in ZONE_POINTS]

            if canonical == "Financials":
                base = self._financial_institution_score(indicators)
            else:
                zones = [ZONE_POINTS[m["assessment"]] for m in models.values() if m.get("assessment") in ZONE_POINTS]
                base = float(np.mean(zones)) if zones else None
            risk_score = None if base is None else round(min(100.0, base + penalty), 1)
            return {"models": models, **self._overall(risk_score, canonical), "red_flags": red_flags,
                    "indicators": indicators, "models_used": used, "sector": canonical, "notes": notes}
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk assessment: {e}")
            return {"models": {name: self._empty(name, label) for name, label in
                               (("altman_z_score", "Altman Z-Score"), ("springate", "Springate"),
                                ("zmijewski", "Zmijewski"))},
                    **self._overall(None, canonical), "red_flags": [], "indicators": {}, "models_used": [],
                    "sector": canonical, "notes": [], "error": str(e)}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _indicators(financial_data: Dict) -> Dict[str, float]:
        inc = financial_data.get("income_statement")
        bs = financial_data.get("balance_sheet")
        cf = financial_data.get("cash_flow")
        assets = period_value(bs, "total_assets")
        equity = period_value(bs, "total_equity")
        net_income = period_value(inc, "net_income")
        ebit = period_value(inc, "ebit")
        interest = period_value(inc, "interest_expense")
        ebitda = ebitda_value(inc, cf)
        debt = total_debt(bs)
        ocf = period_value(cf, "operating_cash_flow")
        fcf = free_cash_flow(cf)
        cash = period_value(bs, "cash_and_short_term_investments")
        revenue, prior_revenue = period_value(inc, "revenue"), period_value(inc, "revenue", 1)
        values = {
            "net_income": net_income, "ebit": ebit, "operating_cash_flow": ocf, "free_cash_flow": fcf,
            "book_equity": equity,
            "equity_to_assets": _div(equity, assets),
            "return_on_assets": _div(net_income, assets),
            "current_ratio": _div(period_value(bs, "current_assets"), period_value(bs, "current_liabilities")),
            "interest_coverage": _div(ebit, abs(interest) if interest else None),
            "debt_to_ebitda": _div(debt, ebitda),
            "revenue_growth": _div(revenue - prior_revenue, abs(prior_revenue))
            if revenue is not None and prior_revenue else None,
            "cash_runway_years": _div(cash, abs(fcf)) if fcf is not None and fcf < 0 else None,
        }
        return {k: v for k, v in values.items() if fd.to_float(v) is not None}

    @staticmethod
    def _red_flags(ind: Dict[str, float], financial: bool) -> tuple:
        checks = [
            (ind.get("net_income", 0) < 0, 8, "Net loss in the latest fiscal year"),
            (ind.get("book_equity", 0) < 0, 12, "Negative shareholders' equity (liabilities exceed assets)"),
        ]
        if financial:
            checks.append((ind.get("equity_to_assets", 1) < 0.05, 10,
                           f"Thin capital buffer (equity/assets {ind.get('equity_to_assets', 0):.1%})"))
        else:
            checks += [
                (ind.get("ebit", 0) < 0, 8, "Negative operating income (EBIT)"),
                (ind.get("operating_cash_flow", 0) < 0, 8, "Negative operating cash flow"),
                (ind.get("free_cash_flow", 0) < 0, 4, "Negative free cash flow"),
                (0 <= ind.get("interest_coverage", math.inf) < 1.5, 10,
                 f"EBIT covers interest expense only {ind.get('interest_coverage', 0):.1f}x"),
                (ind.get("current_ratio", math.inf) < 1.0, 5,
                 f"Current liabilities exceed current assets (current ratio {ind.get('current_ratio', 0):.2f})"),
                (ind.get("debt_to_ebitda", 0) > 5.0, 8, f"High leverage (debt/EBITDA {ind.get('debt_to_ebitda', 0):.1f}x)"),
                (ind.get("revenue_growth", 0) < -0.10, 6, f"Revenue declined {abs(ind.get('revenue_growth', 0)):.1%}"),
                (ind.get("cash_runway_years", math.inf) < 1.0, 12,
                 f"Cash runway below one year at the current burn ({ind.get('cash_runway_years', 0):.1f} years)"),
            ]
        flags = [message for triggered, _, message in checks if triggered]
        penalty = float(sum(weight for triggered, weight, _ in checks if triggered))
        return flags, penalty

    @staticmethod
    def _financial_institution_score(ind: Dict[str, float]) -> Optional[float]:
        """Base risk score for banks/insurers from capital (equity/assets) and ROA."""
        equity_ratio = ind.get("equity_to_assets")
        if equity_ratio is None:
            return None
        score = float(np.interp(equity_ratio, [0.03, 0.05, 0.07, 0.10], [90, 55, 20, 5]))
        roa = ind.get("return_on_assets")
        if roa is not None:
            score += float(np.interp(roa, [-0.01, 0.0, 0.005, 0.01], [20, 10, 5, 0]))
        return min(score, 100.0)

    def _overall(self, risk_score: Optional[float], sector: str) -> Dict[str, Any]:
        if risk_score is None:
            return {"overall_assessment": None, "overall_risk_level": None, "risk_score": None,
                    "overall_description": "Insufficient financial data for a bankruptcy risk assessment.",
                    "overall_color": COLORS["info"]}
        level, assessment = next((lvl, a) for bound, lvl, a in RISK_LEVELS if risk_score < bound)
        descriptions = {
            "Low": "Low financial distress risk: distress models and red-flag checks show no material warning signs.",
            "Moderate": "Moderate financial distress risk: some indicators warrant monitoring.",
            "Elevated": "Elevated financial distress risk: several models or indicators point to financial strain.",
            "High": "High financial distress risk: the profile resembles companies that later defaulted.",
        }
        description = descriptions[level]
        if sector == "Financials":
            description += " (Assessed on capital and profitability; classic Z-Score models do not apply.)"
        color = {"Low": "safe", "Moderate": "grey", "Elevated": "elevated", "High": "distress"}[level]
        return {"overall_assessment": assessment, "overall_risk_level": level, "risk_score": risk_score,
                "overall_description": description, "overall_color": self.risk_colors[color]}

    @staticmethod
    def _empty(name: str, label: str) -> Dict[str, Any]:
        result = {"score": None, "components": {}, "inputs": {}, "assessment": None, "risk_level": "Insufficient Data",
                  "description": "Insufficient data", "interpretation": "", "color": COLORS["info"],
                  "model": label, "applicable": True, "missing": []}
        if name == "altman_z_score":
            result.update(thresholds=None, variant=None, formula="")
        else:
            result.update(threshold=SPRINGATE_CUTOFF if name == "springate" else ZMIJEWSKI_CUTOFF, formula="")
        if name == "zmijewski":
            result["probability"] = None
        return result

    def _base_result(self, name: str, label: str, inputs: Dict[str, Optional[float]]) -> Dict[str, Any]:
        result = self._empty(name, label)
        result["inputs"] = dict(inputs)
        return result

    def _not_applicable(self, name: str, label: str, reason: str, **extra) -> Dict[str, Any]:
        result = self._empty(name, label)
        result.update(assessment="not_applicable", applicable=False, risk_level="Not Applicable",
                      description=reason, interpretation=reason, color=self.risk_colors["not_applicable"], **extra)
        return result

    @staticmethod
    def _insufficient(result: Dict[str, Any], missing: List[str]) -> Dict[str, Any]:
        result.update(missing=missing, description="Insufficient data: missing " + ", ".join(missing),
                      interpretation="The score cannot be computed from the available statements.")
        return result

    def _scored(self, result: Dict[str, Any], score: float, components: Dict[str, float], zone: str,
                interpretation: str) -> Dict[str, Any]:
        labels = {"safe": ("Low Risk", "Low bankruptcy risk (safe zone)"),
                  "grey": ("Moderate Risk", "Moderate bankruptcy risk (grey zone)"),
                  "distress": ("High Risk", "High bankruptcy risk (distress zone)")}
        result.update(score=score, components=components, assessment=zone, risk_level=labels[zone][0],
                      description=labels[zone][1], interpretation=interpretation, color=self.risk_colors[zone])
        return result
