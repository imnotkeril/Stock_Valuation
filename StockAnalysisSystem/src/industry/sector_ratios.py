"""
Sector-specific ratios and comparisons with sector benchmarks.

Each sector defines a few specialised ratios that are computed from the
canonical ``financial_data`` bundle. KPIs that are only available from
company reports (occupancy, same-store sales, reserves...) are read from
``company_info`` when present (e.g. ``company_info["occupancy_rate"]``) and
skipped otherwise; nothing is ever estimated or randomised.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from StockAnalysisSystem.src.config import SECTOR_DATA_DIR
from StockAnalysisSystem.src.industry.benchmarks import LOWER_IS_BETTER, IndustryBenchmarks
from StockAnalysisSystem.src.industry.sector_mapping import SectorMapper, normalize_sector_name
from StockAnalysisSystem.src.models.ratio_analysis import (
    average_value,
    ebitda_value,
    free_cash_flow,
    growth_rate,
    period_value,
    total_debt,
)
from StockAnalysisSystem.src.utils import financial_data as fd

logger = logging.getLogger(__name__)

DEFINITIONS_FILE = "sector_ratio_definitions.json"
ASSESSMENTS = ("excellent", "good", "neutral", "warning", "poor")

Calculator = Callable[[Dict[str, Any]], Optional[float]]


# ---------------------------------------------------------------------------
# Calculators (financial_data -> value or None)
# ---------------------------------------------------------------------------
def _statements(data: Dict[str, Any]):
    return data.get("income_statement"), data.get("balance_sheet"), data.get("cash_flow")


def _revenue(data):
    revenue = period_value(data.get("income_statement"), "revenue")
    return revenue if revenue is not None and revenue > 0 else None


def _rnd_to_revenue(data):
    rnd = period_value(data.get("income_statement"), "rd_expense")
    return fd.safe_divide(abs(rnd) if rnd is not None else None, _revenue(data))


def _sbc_to_revenue(data):
    sbc = period_value(data.get("cash_flow"), "stock_based_compensation")
    return fd.safe_divide(abs(sbc) if sbc is not None else None, _revenue(data))


def _fcf_margin(data):
    return fd.safe_divide(free_cash_flow(data.get("cash_flow")), _revenue(data))


def _rule_of_40(data):
    inc = data.get("income_statement")
    growth = growth_rate(period_value(inc, "revenue"), period_value(inc, "revenue", 1))
    margin = _fcf_margin(data)
    return growth + margin if growth is not None and margin is not None else None


def _cash_runway_years(data):
    """Years of cash at the current operating + capex burn; None when not burning cash."""
    _, bs, cf = _statements(data)
    fcf = free_cash_flow(cf)
    cash = period_value(bs, "cash_and_short_term_investments")
    if fcf is None or cash is None or fcf >= 0:
        return None
    return cash / abs(fcf)


def _net_interest_margin(data):
    inc, bs, _ = _statements(data)
    nii = period_value(inc, "net_interest_income")
    if nii is None:
        income, expense = period_value(inc, "interest_income"), period_value(inc, "interest_expense")
        nii = income - abs(expense) if income is not None and expense is not None else None
    assets = average_value(bs, "total_assets")
    return fd.safe_divide(nii, assets if assets and assets > 0 else None)


def _bank_efficiency_ratio(data):
    inc = data.get("income_statement")
    expense = period_value(inc, ["Non Interest Expense", "operating_expense", "sga_expense"])
    return fd.safe_divide(abs(expense) if expense is not None else None, _revenue(data))


def _equity_to_assets(data):
    bs = data.get("balance_sheet")
    assets = period_value(bs, "total_assets")
    return fd.safe_divide(period_value(bs, "total_equity"), assets if assets and assets > 0 else None)


def _loan_to_deposit(data):
    bs = data.get("balance_sheet")
    deposits = period_value(bs, ["Total Deposits", "Deposits"])
    return fd.safe_divide(period_value(bs, ["Net Loan", "Gross Loan", "Loans"]),
                          deposits if deposits and deposits > 0 else None)


def _capex_to_ocf(data):
    cf = data.get("cash_flow")
    capex = period_value(cf, "capital_expenditure")
    ocf = period_value(cf, "operating_cash_flow")
    return fd.safe_divide(abs(capex) if capex is not None else None, ocf if ocf and ocf > 0 else None)


def _capex_to_revenue(data):
    capex = period_value(data.get("cash_flow"), "capital_expenditure")
    return fd.safe_divide(abs(capex) if capex is not None else None, _revenue(data))


def _capex_to_depreciation(data):
    inc, _, cf = _statements(data)
    capex = period_value(cf, "capital_expenditure")
    dep = period_value(cf, "cash_flow_depreciation") or period_value(inc, "depreciation")
    return fd.safe_divide(abs(capex) if capex is not None else None, abs(dep) if dep else None)


def _debt_to_ebitda(data):
    inc, bs, cf = _statements(data)
    ebitda = ebitda_value(inc, cf)
    return fd.safe_divide(total_debt(bs), ebitda if ebitda and ebitda > 0 else None)


def _interest_coverage(data):
    inc = data.get("income_statement")
    interest = period_value(inc, "interest_expense")
    return fd.safe_divide(period_value(inc, "ebit"), abs(interest) if interest else None)


def _ebitda_margin(data):
    inc, _, cf = _statements(data)
    return fd.safe_divide(ebitda_value(inc, cf), _revenue(data))


def _payout_ratio(data):
    inc, _, cf = _statements(data)
    dividends = period_value(cf, "dividends_paid")
    net_income = period_value(inc, "net_income")
    return fd.safe_divide(abs(dividends) if dividends is not None else None,
                          net_income if net_income and net_income > 0 else None)


def _fcf_yield(data):
    market_cap = fd.to_float((data.get("market_data") or {}).get("market_cap"))
    return fd.safe_divide(free_cash_flow(data.get("cash_flow")), market_cap if market_cap and market_cap > 0 else None)


def _inventory_to_sales(data):
    return fd.safe_divide(period_value(data.get("balance_sheet"), "inventory"), _revenue(data))


def _days_inventory(data):
    inc, bs, _ = _statements(data)
    cogs = period_value(inc, "cost_of_revenue")
    inventory = average_value(bs, "inventory")
    turnover = fd.safe_divide(abs(cogs) if cogs else None, inventory if inventory and inventory > 0 else None)
    return fd.safe_divide(365, turnover)


def _ffo(data) -> Optional[float]:
    inc, _, cf = _statements(data)
    net_income = period_value(inc, "net_income")
    dep = period_value(cf, "cash_flow_depreciation") or period_value(inc, "depreciation")
    return net_income + abs(dep) if net_income is not None and dep is not None else None


def _ffo_per_share(data):
    shares = fd.to_float((data.get("market_data") or {}).get("shares_outstanding"))
    return fd.safe_divide(_ffo(data), shares if shares and shares > 0 else None)


def _price_to_ffo(data):
    price = fd.to_float((data.get("market_data") or {}).get("share_price"))
    per_share = _ffo_per_share(data)
    return fd.safe_divide(price, per_share if per_share and per_share > 0 else None)


def _debt_to_assets(data):
    bs = data.get("balance_sheet")
    assets = period_value(bs, "total_assets")
    return fd.safe_divide(total_debt(bs), assets if assets and assets > 0 else None)


def _reported(key: str) -> Calculator:
    """Company-reported KPI taken from ``company_info[key]`` when available."""
    return lambda data: fd.to_float((data.get("company_info") or {}).get(key))


CALCULATORS: Dict[str, Calculator] = {
    "rnd_to_revenue": _rnd_to_revenue, "sbc_to_revenue": _sbc_to_revenue, "fcf_margin": _fcf_margin,
    "rule_of_40": _rule_of_40, "cash_runway_years": _cash_runway_years,
    "net_interest_margin": _net_interest_margin, "efficiency_ratio": _bank_efficiency_ratio,
    "equity_to_assets": _equity_to_assets, "loan_to_deposit": _loan_to_deposit,
    "capex_to_ocf": _capex_to_ocf, "capex_to_revenue": _capex_to_revenue,
    "capex_to_depreciation": _capex_to_depreciation, "debt_to_ebitda": _debt_to_ebitda,
    "interest_coverage": _interest_coverage, "ebitda_margin": _ebitda_margin, "payout_ratio": _payout_ratio,
    "fcf_yield": _fcf_yield, "inventory_to_sales": _inventory_to_sales,
    "days_inventory_outstanding": _days_inventory, "funds_from_operations": _ffo_per_share,
    "price_to_ffo": _price_to_ffo, "debt_to_assets": _debt_to_assets,
    "same_store_sales_growth": _reported("same_store_sales_growth"), "occupancy_rate": _reported("occupancy_rate"),
    "reserve_replacement_ratio": _reported("reserve_replacement_ratio"),
    "reserve_life_index": _reported("reserve_life_index"), "arpu": _reported("arpu"),
    "churn_rate": _reported("churn_rate"),
}


def _definition(description: str, formula: str, benchmark: Optional[float], is_good_high: Optional[bool],
                required_data=("income_statement",)) -> Dict[str, Any]:
    return {"description": description, "formula": formula, "benchmark": benchmark,
            "is_good_high": is_good_high, "required_data": list(required_data)}


IS, BS, CF, MD, CI = "income_statement", "balance_sheet", "cash_flow", "market_data", "company_info"

#: Sector -> {ratio_key: definition}. ``is_good_high`` None means an optimal range.
DEFAULT_DEFINITIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Technology": {
        "rnd_to_revenue": _definition("R&D expense as a share of revenue", "R&D / Revenue", 0.15, True),
        "sbc_to_revenue": _definition("Stock-based compensation as a share of revenue", "SBC / Revenue",
                                      0.05, False, (IS, CF)),
        "fcf_margin": _definition("Free cash flow as a share of revenue", "FCF / Revenue", 0.20, True, (IS, CF)),
        "rule_of_40": _definition("Revenue growth plus FCF margin (software health check)",
                                  "Revenue growth + FCF margin", 0.40, True, (IS, CF)),
    },
    "Healthcare": {
        "rnd_to_revenue": _definition("R&D expense as a share of revenue", "R&D / Revenue", 0.15, True),
        "fcf_margin": _definition("Free cash flow as a share of revenue", "FCF / Revenue", 0.15, True, (IS, CF)),
        "cash_runway_years": _definition("Years of cash left at the current free-cash-flow burn",
                                         "(Cash + ST investments) / |FCF|", 2.0, True, (BS, CF)),
    },
    "Financials": {
        "net_interest_margin": _definition("Net interest income relative to average earning assets "
                                           "(total assets proxy)", "Net Interest Income / Avg Assets",
                                           0.03, True, (IS, BS)),
        "efficiency_ratio": _definition("Non-interest expense as a share of revenue",
                                        "Non-Interest Expense / Revenue", 0.60, False),
        "equity_to_assets": _definition("Shareholders' equity relative to total assets (capital buffer)",
                                        "Equity / Total Assets", 0.10, True, (BS,)),
        "loan_to_deposit": _definition("Loans as a share of deposits", "Net Loans / Total Deposits",
                                       0.80, None, (BS,)),
    },
    "Energy": {
        "capex_to_ocf": _definition("Share of operating cash flow reinvested", "|CapEx| / Operating Cash Flow",
                                    0.60, False, (CF,)),
        "debt_to_ebitda": _definition("Total debt relative to EBITDA", "Total Debt / EBITDA", 1.5, False, (IS, BS)),
        "fcf_yield": _definition("Free cash flow relative to market capitalisation", "FCF / Market Cap",
                                 0.06, True, (CF, MD)),
        "reserve_replacement_ratio": _definition("Reserves added relative to production (company-reported)",
                                                 "New Reserves / Production", 1.0, True, (CI,)),
    },
    "Consumer Discretionary": {
        "inventory_to_sales": _definition("Inventory as a share of revenue", "Inventory / Revenue",
                                          0.15, False, (IS, BS)),
        "days_inventory_outstanding": _definition("Average days to sell inventory", "365 / Inventory Turnover",
                                                  60.0, False, (IS, BS)),
        "same_store_sales_growth": _definition("Comparable-store sales growth (company-reported)",
                                               "Current SSS / Previous SSS - 1", 0.03, True, (CI,)),
    },
    "Consumer Staples": {
        "payout_ratio": _definition("Dividends as a share of net income", "Dividends / Net Income",
                                    0.60, None, (IS, CF)),
        "fcf_margin": _definition("Free cash flow as a share of revenue", "FCF / Revenue", 0.12, True, (IS, CF)),
        "inventory_to_sales": _definition("Inventory as a share of revenue", "Inventory / Revenue",
                                          0.10, False, (IS, BS)),
    },
    "Industrials": {
        "capex_to_revenue": _definition("Capital intensity", "|CapEx| / Revenue", 0.04, False, (IS, CF)),
        "days_inventory_outstanding": _definition("Average days to sell inventory", "365 / Inventory Turnover",
                                                  90.0, False, (IS, BS)),
        "fcf_margin": _definition("Free cash flow as a share of revenue", "FCF / Revenue", 0.10, True, (IS, CF)),
    },
    "Materials": {
        "capex_to_depreciation": _definition("Reinvestment relative to asset consumption",
                                             "|CapEx| / Depreciation", 1.2, None, (IS, CF)),
        "ebitda_margin": _definition("EBITDA as a share of revenue", "EBITDA / Revenue", 0.20, True),
        "debt_to_ebitda": _definition("Total debt relative to EBITDA", "Total Debt / EBITDA", 2.0, False, (IS, BS)),
    },
    "Real Estate": {
        "funds_from_operations": _definition("Funds From Operations per share",
                                             "(Net Income + Depreciation & Amortisation) / Shares Outstanding",
                                             None, True, (IS, CF, MD)),
        "price_to_ffo": _definition("Share price relative to FFO per share", "Price / FFO per Share",
                                    16.0, False, (IS, CF, MD)),
        "debt_to_assets": _definition("Total debt relative to total assets", "Total Debt / Total Assets",
                                      0.40, False, (BS,)),
        "occupancy_rate": _definition("Share of leasable space occupied (company-reported)",
                                      "Occupied Space / Total Space", 0.95, True, (CI,)),
    },
    "Communication Services": {
        "ebitda_margin": _definition("EBITDA as a share of revenue", "EBITDA / Revenue", 0.30, True),
        "capex_to_revenue": _definition("Capital intensity", "|CapEx| / Revenue", 0.15, False, (IS, CF)),
        "arpu": _definition("Average revenue per user (company-reported)", "Revenue / Users", None, True, (CI,)),
        "churn_rate": _definition("Customer churn per period (company-reported)", "Lost / Total Customers",
                                  0.02, False, (CI,)),
    },
    "Utilities": {
        "debt_to_ebitda": _definition("Total debt relative to EBITDA", "Total Debt / EBITDA", 5.0, False, (IS, BS)),
        "interest_coverage": _definition("EBIT relative to interest expense", "EBIT / Interest Expense",
                                         3.0, True),
        "payout_ratio": _definition("Dividends as a share of net income", "Dividends / Net Income",
                                    0.70, None, (IS, CF)),
    },
}


def _assess_pct_diff(pct_diff: Optional[float], is_good_high: Optional[bool]) -> str:
    """Five-level assessment of a percentage difference from the benchmark."""
    if pct_diff is None:
        return "neutral"
    if is_good_high is None:  # optimal range: closeness to the benchmark matters
        distance = abs(pct_diff)
        return "good" if distance <= 10 else "neutral" if distance <= 25 else "warning"
    better = pct_diff if is_good_high else -pct_diff
    if better > 20:
        return "excellent"
    if better > 5:
        return "good"
    if better >= -5:
        return "neutral"
    if better >= -20:
        return "warning"
    return "poor"


def _pct_diff(value: Optional[float], benchmark: Optional[float]) -> Optional[float]:
    if value is None or benchmark is None or benchmark == 0:
        return None
    return (value - benchmark) / abs(benchmark) * 100


class SectorRatioAnalyzer:
    """Sector-specific ratios and comparisons with sector averages."""

    def __init__(self,
                 benchmarks: Optional[IndustryBenchmarks] = None,
                 sector_mapper: Optional[SectorMapper] = None,
                 data_dir: str = SECTOR_DATA_DIR):
        """
        Args:
            benchmarks: Benchmark source (default :class:`IndustryBenchmarks`).
            sector_mapper: Sector resolver (default :class:`SectorMapper`).
            data_dir: Directory with optional ``sector_ratio_definitions.json``
                (metadata overrides such as benchmarks; read only).
        """
        self.data_dir = Path(data_dir)
        self.benchmarks = benchmarks or IndustryBenchmarks(data_dir)
        self.sector_mapper = sector_mapper or SectorMapper(data_dir)
        self._definitions: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

    @property
    def sector_ratio_definitions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """``{sector: {ratio_key: {description, formula, benchmark, is_good_high, required_data}}}``."""
        if self._definitions is None:
            self._definitions = self._load_sector_ratio_definitions()
        return self._definitions

    def get_sector_specific_ratios(self, ticker: str, sector: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the specialised ratios of a sector.

        Returns:
            ``{"sector": canonical, "ratios": {key: {"value", "description",
            "formula", "benchmark", "is_good_high", "pct_diff", "assessment"}}}``.
            Ratios without data are omitted. ``{}`` for sectors without
            definitions; ``{"error": str}`` on unexpected failure.
        """
        canonical = normalize_sector_name(sector)
        definitions = self.sector_ratio_definitions.get(canonical, {})
        if not definitions:
            return {}
        try:
            data = financial_data or {}
            overrides = self.benchmarks.get_sector_benchmarks(canonical).get("sector_specific", {}) or {}
            ratios: Dict[str, Dict[str, Any]] = {}
            for key, definition in definitions.items():
                calculator = CALCULATORS.get(key)
                if calculator is None or not self._check_data_available(definition.get("required_data", []), data):
                    continue
                value = fd.to_float(calculator(data))
                if value is None:
                    continue
                benchmark = fd.to_float(overrides.get(key, definition.get("benchmark")))
                is_good_high = definition.get("is_good_high", key not in LOWER_IS_BETTER)
                pct_diff = _pct_diff(value, benchmark)
                ratios[key] = {
                    "value": value,
                    "description": definition.get("description", ""),
                    "formula": definition.get("formula", ""),
                    "benchmark": benchmark,
                    "is_good_high": is_good_high,
                    "pct_diff": pct_diff,
                    "assessment": _assess_pct_diff(pct_diff, is_good_high),
                }
            return {"sector": canonical, "ratios": ratios}
        except Exception as e:
            logger.error(f"Error calculating sector-specific ratios for {ticker}: {e}")
            return {"error": str(e)}

    def compare_with_sector_averages(self, ticker: str, sector: str,
                                     ratios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare company ratios with sector benchmarks.

        Args:
            ratios: ``FinancialRatioAnalyzer.calculate_ratios`` output (floats) or
                ``analyze_ratios`` output (dicts with ``value``).

        Returns:
            ``{"sector", "company", "categories": {category: {ratio: {"company_value",
            "sector_value", "pct_diff", "assessment"}}}, "summary": {...}}``.
        """
        canonical = normalize_sector_name(sector)
        try:
            benchmarks = self.benchmarks.get_sector_benchmarks(canonical)
            categories: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for category, category_ratios in (ratios or {}).items():
                if category in ("sector", "error") or not isinstance(category_ratios, dict):
                    continue
                comparison = {}
                for name, raw in category_ratios.items():
                    value = fd.to_float(raw.get("value") if isinstance(raw, dict) else raw)
                    benchmark = fd.to_float(benchmarks.get(category, {}).get(name))
                    if value is None or benchmark is None:
                        continue
                    is_good_high = raw.get("is_good_high") if isinstance(raw, dict) else None
                    if is_good_high is None:
                        is_good_high = name not in LOWER_IS_BETTER
                    pct_diff = _pct_diff(value, benchmark)
                    comparison[name] = {"company_value": value, "sector_value": benchmark, "pct_diff": pct_diff,
                                        "assessment": _assess_pct_diff(pct_diff, is_good_high)}
                if comparison:
                    categories[category] = comparison
            return {"sector": canonical, "company": ticker, "categories": categories,
                    "summary": self._create_comparison_summary(categories)}
        except Exception as e:
            logger.error(f"Error comparing {ticker} with sector averages: {e}")
            return {"error": str(e)}

    def get_sector_performance_metrics(self, sector: str) -> Dict[str, Any]:
        """``{"sector", "metrics": {"1D", "1W", "1M", "3M", "1Y", "YTD", "date", "source"}}`` or ``{"error"}``."""
        canonical = normalize_sector_name(sector)
        try:
            performance = self.benchmarks.get_sector_performance()
            rows = performance[performance["Sector"] == canonical] if "Sector" in performance else pd.DataFrame()
            if rows.empty:
                return {"error": f"No performance data for {canonical}"}
            metrics = {k: (fd.to_float(v) if k not in ("date", "source") else v)
                       for k, v in rows.iloc[0].to_dict().items() if k != "Sector"}
            return {"sector": canonical, "metrics": metrics}
        except Exception as e:
            logger.error(f"Error getting performance metrics for sector {sector}: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------ helpers
    def _load_sector_ratio_definitions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        definitions = copy.deepcopy(DEFAULT_DEFINITIONS)
        definitions_file = self.data_dir / DEFINITIONS_FILE
        if not definitions_file.exists():
            return definitions
        try:
            file_definitions = json.loads(definitions_file.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable {definitions_file}: {e}")
            return definitions
        for sector, sector_defs in file_definitions.items():
            if not isinstance(sector_defs, dict):
                continue
            target = definitions.setdefault(normalize_sector_name(sector), {})
            for key, definition in sector_defs.items():
                if isinstance(definition, dict):
                    target.setdefault(key, {}).update(definition)
        return definitions

    @staticmethod
    def _check_data_available(required_data, financial_data: Dict[str, Any]) -> bool:
        """True if every required bundle entry is present and non-empty (``company_profile`` == ``company_info``)."""
        for name in required_data:
            name = "company_info" if name == "company_profile" else name
            value = financial_data.get(name)
            if value is None or (isinstance(value, pd.DataFrame) and value.empty) or \
                    (isinstance(value, dict) and not value):
                return False
        return True

    @staticmethod
    def _assess_ratio(value: Optional[float], benchmark: Optional[float], is_good_high: Optional[bool]) -> str:
        return _assess_pct_diff(_pct_diff(fd.to_float(value), fd.to_float(benchmark)), is_good_high)

    @staticmethod
    def _assess_difference(pct_diff: Optional[float], is_good_high: Optional[bool]) -> str:
        return _assess_pct_diff(fd.to_float(pct_diff), is_good_high)

    @staticmethod
    def _create_comparison_summary(categories: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        counts = {assessment: 0 for assessment in ASSESSMENTS}
        for ratios in categories.values():
            for data in ratios.values():
                if data.get("assessment") in counts:
                    counts[data["assessment"]] += 1
        total = sum(counts.values())
        percentages = {k: (v / total * 100 if total else 0.0) for k, v in counts.items()}
        if not total:
            overall = "Insufficient data for comparison"
        else:
            positive = percentages["excellent"] + percentages["good"]
            negative = percentages["warning"] + percentages["poor"]
            if positive > 60:
                overall = "Strong performer relative to sector"
            elif negative > 60:
                overall = "Underperformer relative to sector"
            elif positive > 40:
                overall = "Above average performer in sector"
            elif positive > negative:
                overall = "Average performer in sector"
            elif positive == negative:
                overall = "Mixed performance relative to sector"
            else:
                overall = "Below average performer in sector"
        return {"assessment_counts": counts, "assessment_percentages": percentages,
                "total_ratios_compared": total, "overall_assessment": overall}
