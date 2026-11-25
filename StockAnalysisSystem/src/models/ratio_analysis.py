"""
Financial ratio analysis.

All ratios are computed from the canonical ``financial_data`` bundle (see
:mod:`StockAnalysisSystem.src.utils.financial_data`) using values of the
*same* fiscal period: the most recent period reported by each statement.
Balance-sheet denominators of return and turnover ratios use the average of
the opening and closing balance when the prior period is available.

Margins, returns, yields and growth rates are fractions (0.25 == 25 %).
Ratios that are not meaningful (negative earnings for P/E, negative equity
for ROE / D/E, zero denominators...) are omitted rather than reported as
NaN/inf.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from StockAnalysisSystem.src.config import RATIO_CATEGORIES, SECTOR_SPECIFIC_RATIOS
from StockAnalysisSystem.src.industry.benchmarks import LOWER_IS_BETTER, IndustryBenchmarks
from StockAnalysisSystem.src.industry.sector_mapping import normalize_sector_name
from StockAnalysisSystem.src.utils import financial_data as fd

logger = logging.getLogger(__name__)

CATEGORIES = ("valuation", "profitability", "liquidity", "leverage", "efficiency", "growth")
DEFAULT_TAX_RATE = 0.21
DAYS_PER_YEAR = 365

ASSESSMENT_COLORS = {"positive": "#74f174", "neutral": "#fff59d", "negative": "#faa1a4"}

#: Display name (config.SECTOR_SPECIFIC_RATIOS / RATIO_CATEGORIES) -> (category, key in calculate_ratios)
RATIO_KEYS: Dict[str, tuple] = {
    "P/E": ("Valuation", "pe_ratio"), "Forward P/E": ("Valuation", "forward_pe"), "PEG": ("Valuation", "peg_ratio"),
    "P/S": ("Valuation", "ps_ratio"), "P/B": ("Valuation", "pb_ratio"), "EV/EBITDA": ("Valuation", "ev_ebitda"),
    "EV/Revenue": ("Valuation", "ev_revenue"), "P/CF": ("Valuation", "price_to_cash_flow"),
    "Dividend Yield": ("Valuation", "dividend_yield"), "FFO": ("Valuation", "price_to_ffo"),
    "Gross Margin": ("Profitability", "gross_margin"), "Operating Margin": ("Profitability", "operating_margin"),
    "Net Margin": ("Profitability", "net_margin"), "EBITDA Margin": ("Profitability", "ebitda_margin"),
    "ROE": ("Profitability", "roe"), "ROA": ("Profitability", "roa"), "ROIC": ("Profitability", "roic"),
    "Net Interest Margin": ("Profitability", "net_interest_margin"),
    "Current Ratio": ("Liquidity", "current_ratio"), "Quick Ratio": ("Liquidity", "quick_ratio"),
    "Cash Ratio": ("Liquidity", "cash_ratio"),
    "Debt/Equity": ("Leverage", "debt_to_equity"), "Debt/EBITDA": ("Leverage", "debt_to_ebitda"),
    "Debt/Assets": ("Leverage", "debt_to_assets"), "Interest Coverage": ("Leverage", "interest_coverage"),
    "Asset Turnover": ("Efficiency", "asset_turnover"), "Inventory Turnover": ("Efficiency", "inventory_turnover"),
    "Receivables Turnover": ("Efficiency", "receivables_turnover"), "R&D/Revenue": ("Efficiency", "rnd_to_revenue"),
    "Same-Store Sales Growth": ("Growth", None), "Reserve Replacement Ratio": ("Other", None),
    "NAV": ("Valuation", None), "Occupancy Rate": ("Other", None), "ARPU": ("Other", None),
    "Churn Rate": ("Other", None), "Efficiency Ratio": ("Efficiency", None),
}

RATIO_DESCRIPTIONS: Dict[str, str] = {
    "P/E": "Price to Earnings - share price divided by diluted EPS",
    "Forward P/E": "Forward Price to Earnings - P/E based on analysts' next-year EPS estimate",
    "PEG": "Price/Earnings to Growth - P/E divided by EPS growth rate (in %)",
    "P/S": "Price to Sales - market capitalisation divided by revenue",
    "P/B": "Price to Book - market capitalisation divided by shareholders' equity",
    "P/CF": "Price to Cash Flow - market capitalisation divided by operating cash flow",
    "EV/EBITDA": "Enterprise Value (market cap + debt + minority interest - cash) divided by EBITDA",
    "EV/Revenue": "Enterprise Value divided by revenue",
    "Dividend Yield": "Annual dividends per share divided by share price",
    "FFO": "Funds From Operations (net income + depreciation & amortisation); shown as Price/FFO - key for REITs",
    "NAV": "Net Asset Value - assets minus liabilities at fair value, critical for REITs and financials",
    "ROE": "Return on Equity - net income divided by average shareholders' equity",
    "ROA": "Return on Assets - net income divided by average total assets",
    "ROIC": "Return on Invested Capital - after-tax operating income divided by average invested capital",
    "Gross Margin": "Gross profit divided by revenue - product profitability",
    "Operating Margin": "Operating income divided by revenue - operating efficiency",
    "Net Margin": "Net income divided by revenue - overall profitability",
    "EBITDA Margin": "EBITDA divided by revenue - cash operating profitability",
    "Net Interest Margin": "Net interest income divided by average earning assets (total assets proxy) - key for banks",
    "Efficiency Ratio": "Non-interest expense divided by revenue - cost efficiency of banks (lower is better)",
    "Current Ratio": "Current assets divided by current liabilities - short-term liquidity",
    "Quick Ratio": "(Current assets - inventory) divided by current liabilities - stricter liquidity",
    "Cash Ratio": "Cash and short-term investments divided by current liabilities",
    "Debt/Equity": "Total debt divided by shareholders' equity - financial leverage",
    "Debt/EBITDA": "Total debt divided by EBITDA - years of earnings needed to repay debt",
    "Debt/Assets": "Total debt divided by total assets",
    "Interest Coverage": "EBIT divided by interest expense - ability to service debt",
    "Asset Turnover": "Revenue divided by average total assets - asset efficiency",
    "Inventory Turnover": "Cost of goods sold divided by average inventory",
    "Receivables Turnover": "Revenue divided by average accounts receivable",
    "R&D/Revenue": "Research and development expense divided by revenue - innovation intensity",
    "Same-Store Sales Growth": "Revenue growth of stores open at least a year - company-reported KPI for retail",
    "Reserve Replacement Ratio": "Reserves added relative to production - company-reported KPI for energy",
    "Occupancy Rate": "Share of leasable space that is occupied - company-reported KPI for REITs",
    "ARPU": "Average Revenue Per User - company-reported KPI for subscription businesses",
    "Churn Rate": "Share of customers lost per period - company-reported KPI for subscription businesses",
}


# ---------------------------------------------------------------------------
# Period-aligned statement lookups (reused by other analysis modules)
# ---------------------------------------------------------------------------
def _reference_position(df: pd.DataFrame) -> int:
    """Position (most recent first) of the latest period that has any data."""
    ordered = fd.chronological(df).columns[::-1]
    numeric = df[ordered].apply(pd.to_numeric, errors="coerce")
    has_data = numeric.notna().any(axis=0).tolist()
    return has_data.index(True) if True in has_data else 0


def period_value(df: Optional[pd.DataFrame], item: fd.ItemSpec, offset: int = 0) -> Optional[float]:
    """
    Value of ``item`` in the statement's most recent reported period.

    Unlike :func:`financial_data.get_latest` this does not fall back to an
    older period when the item is missing for the latest one, so ratios never
    mix fiscal years. ``offset=1`` returns the prior period.
    """
    if fd.is_empty(df):
        return None
    series = fd.get_series(df, item, ascending=False, dropna=False)
    position = _reference_position(df) + offset
    if series.empty or position >= len(series):
        return None
    return fd.to_float(series.iloc[position])


def average_value(df: Optional[pd.DataFrame], item: fd.ItemSpec) -> Optional[float]:
    """Average of the latest and prior period balance; the latest balance if the prior is missing."""
    current = period_value(df, item)
    previous = period_value(df, item, 1)
    if current is None:
        return None
    return current if previous is None else (current + previous) / 2


def growth_rate(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Period-over-period growth ``(current - previous) / |previous|``; None for a zero/missing base."""
    if current is None or previous is None or previous == 0:
        return None
    return fd.to_float((current - previous) / abs(previous))


def annualized_growth(series: pd.Series) -> Optional[float]:
    """
    CAGR of a chronological series using the actual time span between the
    first and last observation (falls back to ``len - 1`` years for
    non-date indexes). None for non-positive endpoints or spans < ~1 year.
    """
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return None
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    try:
        years = (pd.Timestamp(series.index[-1]) - pd.Timestamp(series.index[0])).days / 365.25
    except (TypeError, ValueError):
        years = len(series) - 1
    if years < 0.9:
        return None
    return fd.to_float((end / start) ** (1 / years) - 1)


def _positive(value: Optional[float]) -> Optional[float]:
    return value if value is not None and value > 0 else None


def _abs(value: Optional[float]) -> Optional[float]:
    return abs(value) if value is not None else None


def effective_tax_rate(income: Optional[pd.DataFrame]) -> float:
    """Reported tax rate for calcs, else tax provision / pretax income, clipped to [0, 50 %]; 21 % fallback."""
    rate = period_value(income, "tax_rate")
    if rate is None:
        rate = fd.safe_divide(period_value(income, "tax_provision"), _positive(period_value(income, "pretax_income")))
    if rate is None:
        return DEFAULT_TAX_RATE
    return min(max(rate, 0.0), 0.5)


def total_debt(balance: Optional[pd.DataFrame]) -> Optional[float]:
    """Total debt, or long-term + short-term debt when no total is reported."""
    debt = period_value(balance, "total_debt")
    if debt is not None:
        return debt
    parts = [period_value(balance, "long_term_debt"), period_value(balance, "short_term_debt")]
    parts = [p for p in parts if p is not None]
    return sum(parts) if parts else None


def ebitda_value(income: Optional[pd.DataFrame], cash_flow: Optional[pd.DataFrame] = None,
                 offset: int = 0) -> Optional[float]:
    """Reported EBITDA, else operating income + depreciation & amortisation."""
    ebitda = period_value(income, "ebitda", offset)
    if ebitda is not None:
        return ebitda
    operating = period_value(income, "operating_income", offset)
    depreciation = period_value(income, "depreciation", offset)
    if depreciation is None:
        depreciation = period_value(cash_flow, "cash_flow_depreciation", offset)
    if operating is None or depreciation is None:
        return None
    return operating + abs(depreciation)


def free_cash_flow(cash_flow: Optional[pd.DataFrame], offset: int = 0) -> Optional[float]:
    """Reported free cash flow, else operating cash flow - |capital expenditure|."""
    fcf = period_value(cash_flow, "free_cash_flow", offset)
    if fcf is not None:
        return fcf
    ocf = period_value(cash_flow, "operating_cash_flow", offset)
    capex = period_value(cash_flow, "capital_expenditure", offset)
    if ocf is None or capex is None:
        return None
    return ocf - abs(capex)


class FinancialRatioAnalyzer:
    """Calculate financial ratios and compare them with sector benchmarks."""

    def __init__(self, benchmarks: Optional[IndustryBenchmarks] = None):
        """
        Args:
            benchmarks: Source of sector benchmarks (defaults to :class:`IndustryBenchmarks`).
        """
        self.sector_specific_ratios = SECTOR_SPECIFIC_RATIOS
        self.ratio_categories = RATIO_CATEGORIES
        self._benchmarks = benchmarks

    @property
    def benchmarks(self) -> IndustryBenchmarks:
        if self._benchmarks is None:
            self._benchmarks = IndustryBenchmarks()
        return self._benchmarks

    # ------------------------------------------------------------------ ratios
    def calculate_ratios(self, financial_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate financial ratios for the most recent fiscal period.

        Args:
            financial_data: Canonical bundle with ``income_statement``,
                ``balance_sheet``, ``cash_flow`` and optional ``market_data`` /
                ``company_info``.

        Returns:
            ``{category: {ratio_key: float}}`` with every category of
            :data:`CATEGORIES` present (possibly empty). Keys:

            * valuation: pe_ratio, forward_pe, peg_ratio, ps_ratio, pb_ratio, ev_ebitda,
              ev_revenue, price_to_fcf, price_to_cash_flow, dividend_yield, payout_ratio,
              price_to_ffo (Real Estate only)
            * profitability: gross_margin, operating_margin, ebitda_margin, net_margin,
              fcf_margin, roe, roa, roic, net_interest_margin (Financials only)
            * liquidity: current_ratio, quick_ratio, cash_ratio, operating_cash_flow_ratio
            * leverage: debt_to_equity, debt_to_assets, debt_to_ebitda, net_debt_to_ebitda,
              interest_coverage, cash_to_debt
            * efficiency: asset_turnover, inventory_turnover, receivables_turnover,
              days_sales_outstanding, days_inventory_outstanding, rnd_to_revenue
            * growth: revenue_growth, net_income_growth, eps_growth, operating_income_growth,
              ebitda_growth, fcf_growth, dividend_growth (YoY) and revenue_cagr,
              net_income_cagr, eps_cagr

            Ratios that cannot be computed meaningfully are omitted.
        """
        ratios: Dict[str, Dict[str, Optional[float]]] = {category: {} for category in CATEGORIES}
        if not isinstance(financial_data, dict):
            return ratios
        try:
            self._fill_ratios(ratios, financial_data)
        except Exception as e:  # defensive: never break the UI on malformed data
            logger.error(f"Error calculating financial ratios: {e}")
        return {category: {k: v for k, v in values.items() if fd.to_float(v) is not None}
                for category, values in ratios.items()}

    def _fill_ratios(self, ratios: Dict[str, Dict[str, Optional[float]]], financial_data: Dict) -> None:
        inc = financial_data.get("income_statement")
        bs = financial_data.get("balance_sheet")
        cf = financial_data.get("cash_flow")
        info = financial_data.get("company_info") or {}
        market = financial_data.get("market_data") or {}
        sector = normalize_sector_name(info.get("sector"))
        div = fd.safe_divide

        # ----- inputs (same fiscal period) -----
        revenue = period_value(inc, "revenue")
        cogs = _abs(period_value(inc, "cost_of_revenue"))
        gross = period_value(inc, "gross_profit")
        if gross is None and revenue is not None and cogs is not None:
            gross = revenue - cogs
        operating = period_value(inc, "operating_income")
        ebit = period_value(inc, "ebit")
        ebitda = ebitda_value(inc, cf)
        net_income = period_value(inc, "net_income")
        interest = _abs(period_value(inc, "interest_expense"))
        eps = period_value(inc, "diluted_eps")
        if eps is None:
            eps = div(net_income, _positive(period_value(inc, "diluted_shares")))
        rnd = _abs(period_value(inc, "rd_expense"))

        equity = period_value(bs, "total_equity")
        avg_equity = average_value(bs, "total_equity")
        assets = period_value(bs, "total_assets")
        avg_assets = average_value(bs, "total_assets")
        current_assets = period_value(bs, "current_assets")
        current_liabilities = _positive(period_value(bs, "current_liabilities"))
        inventory = period_value(bs, "inventory")
        cash = period_value(bs, "cash_and_short_term_investments")
        debt = total_debt(bs)
        minority = period_value(bs, "minority_interest") or 0.0
        avg_inventory = _positive(average_value(bs, "inventory"))
        avg_receivables = _positive(average_value(bs, "receivables"))
        avg_invested = average_value(bs, "invested_capital")
        if avg_invested is None and debt is not None and equity is not None:
            avg_invested = debt + equity

        ocf = period_value(cf, "operating_cash_flow")
        fcf = free_cash_flow(cf)
        dividends = _abs(period_value(cf, "dividends_paid"))
        depreciation = _abs(period_value(cf, "cash_flow_depreciation") or period_value(inc, "depreciation"))

        price = _positive(fd.to_float(market.get("share_price")))
        shares = _positive(fd.to_float(market.get("shares_outstanding")))
        market_cap = _positive(fd.to_float(market.get("market_cap")))
        if market_cap is None and price and shares:
            market_cap = price * shares

        # ----- valuation -----
        val = ratios["valuation"]
        if price is not None and _positive(eps):
            val["pe_ratio"] = price / eps
        elif market_cap is not None:
            val["pe_ratio"] = div(market_cap, _positive(net_income))
        val["forward_pe"] = _positive(fd.to_float(info.get("forward_pe")))
        val["ps_ratio"] = div(market_cap, _positive(revenue))
        val["pb_ratio"] = div(market_cap, _positive(equity))
        enterprise_value = None
        if market_cap is not None and (debt is not None or cash is not None):
            enterprise_value = market_cap + (debt or 0.0) + minority - (cash or 0.0)
        enterprise_value = _positive(enterprise_value)
        val["ev_ebitda"] = div(enterprise_value, _positive(ebitda))
        val["ev_revenue"] = div(enterprise_value, _positive(revenue))
        val["price_to_fcf"] = div(market_cap, _positive(fcf))
        val["price_to_cash_flow"] = div(market_cap, _positive(ocf))
        dividend_yield = fd.to_float(market.get("dividend_yield"))
        if dividend_yield is None and dividends is not None:
            dividend_yield = div(dividends, market_cap)
        val["dividend_yield"] = dividend_yield
        if dividends is not None:
            val["payout_ratio"] = div(dividends, _positive(net_income))
        if sector == "Real Estate" and net_income is not None and depreciation is not None:
            val["price_to_ffo"] = div(market_cap, _positive(net_income + depreciation))

        # ----- profitability -----
        prof = ratios["profitability"]
        sales = _positive(revenue)
        if sector != "Financials":  # banks report no cost of goods sold
            prof["gross_margin"] = div(gross, sales)
        prof["operating_margin"] = div(operating, sales)
        prof["ebitda_margin"] = div(ebitda, sales)
        prof["net_margin"] = div(net_income, sales)
        prof["fcf_margin"] = div(fcf, sales)
        prof["roe"] = div(net_income, _positive(avg_equity))
        prof["roa"] = div(net_income, _positive(avg_assets))
        if operating is not None:
            prof["roic"] = div(operating * (1 - effective_tax_rate(inc)), _positive(avg_invested))
        if sector == "Financials":
            prof["net_interest_margin"] = div(period_value(inc, "net_interest_income"), _positive(avg_assets))

        # ----- liquidity -----
        liq = ratios["liquidity"]
        liq["current_ratio"] = div(current_assets, current_liabilities)
        if current_assets is not None:
            liq["quick_ratio"] = div(current_assets - (inventory or 0.0), current_liabilities)
        liq["cash_ratio"] = div(cash, current_liabilities)
        liq["operating_cash_flow_ratio"] = div(ocf, current_liabilities)

        # ----- leverage -----
        lev = ratios["leverage"]
        lev["debt_to_equity"] = div(debt, _positive(equity))
        lev["debt_to_assets"] = div(debt, _positive(assets))
        lev["debt_to_ebitda"] = div(debt, _positive(ebitda))
        if debt is not None:
            lev["net_debt_to_ebitda"] = div(debt - (cash or 0.0), _positive(ebitda))
        lev["interest_coverage"] = div(ebit if ebit is not None else operating, _positive(interest))
        lev["cash_to_debt"] = div(cash, _positive(debt))

        # ----- efficiency -----
        eff = ratios["efficiency"]
        eff["asset_turnover"] = div(revenue, _positive(avg_assets))
        inventory_turnover = div(_positive(cogs), avg_inventory)
        receivables_turnover = div(sales, avg_receivables)
        eff["inventory_turnover"] = inventory_turnover
        eff["receivables_turnover"] = receivables_turnover
        eff["days_inventory_outstanding"] = div(DAYS_PER_YEAR, inventory_turnover)
        eff["days_sales_outstanding"] = div(DAYS_PER_YEAR, receivables_turnover)
        eff["rnd_to_revenue"] = div(rnd, sales)

        # ----- growth -----
        gro = ratios["growth"]
        gro["revenue_growth"] = growth_rate(revenue, period_value(inc, "revenue", 1))
        gro["net_income_growth"] = growth_rate(net_income, period_value(inc, "net_income", 1))
        gro["eps_growth"] = growth_rate(eps, period_value(inc, "diluted_eps", 1))
        gro["operating_income_growth"] = growth_rate(operating, period_value(inc, "operating_income", 1))
        gro["ebitda_growth"] = growth_rate(ebitda, ebitda_value(inc, cf, 1))
        gro["fcf_growth"] = growth_rate(fcf, free_cash_flow(cf, 1))
        gro["dividend_growth"] = growth_rate(dividends, _abs(period_value(cf, "dividends_paid", 1)))
        gro["revenue_cagr"] = annualized_growth(fd.get_series(inc, "revenue", ascending=True))
        gro["net_income_cagr"] = annualized_growth(fd.get_series(inc, "net_income", ascending=True))
        gro["eps_cagr"] = annualized_growth(fd.get_series(inc, "diluted_eps", ascending=True))

        # PEG uses multi-year EPS growth when available (in percent, as conventional)
        eps_growth = gro["eps_cagr"] if gro["eps_cagr"] is not None else gro["eps_growth"]
        if val.get("pe_ratio") and eps_growth is not None and eps_growth > 0:
            val["peg_ratio"] = val["pe_ratio"] / (eps_growth * 100)

    # -------------------------------------------------------------- benchmarks
    def get_sector_benchmarks(self, sector: Optional[str]) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Benchmark ratios for a sector (any spelling, e.g. ``"Financial Services"``).

        Returns:
            ``{category: {ratio_key: value or None}}`` with the same categories
            and keys for every sector (see ``industry.benchmarks.BENCHMARK_KEYS``);
            ``None`` marks ratios that are not meaningful for the sector. Unknown
            or missing sectors get generic market benchmarks.
        """
        return self.benchmarks.get_sector_benchmarks(sector)

    def analyze_ratios(self, company_ratios: Dict[str, Dict[str, float]],
                       sector: Optional[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compare company ratios with sector benchmarks.

        Returns:
            ``{category: {ratio_key: {"value", "benchmark", "percent_diff",
            "assessment", "color", "higher_is_better"}}}`` where ``assessment``
            is ``"positive"`` / ``"neutral"`` / ``"negative"`` (a ±10 % band around
            the benchmark, oriented by whether higher values are better) and
            ``percent_diff`` is ``(value - benchmark) / |benchmark| * 100``.
            ``benchmark`` / ``percent_diff`` are None without a benchmark.
        """
        benchmarks = self.get_sector_benchmarks(sector)
        analysis: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for category, values in (company_ratios or {}).items():
            if not isinstance(values, dict):
                continue
            analysis[category] = {}
            for name, raw in values.items():
                value = fd.to_float(raw.get("value") if isinstance(raw, dict) else raw)
                if value is None:
                    continue
                higher_is_better = name not in LOWER_IS_BETTER
                benchmark = fd.to_float(benchmarks.get(category, {}).get(name))
                percent_diff = None
                assessment = "neutral"
                if benchmark is not None and benchmark != 0:
                    percent_diff = (value - benchmark) / abs(benchmark) * 100
                    better = percent_diff if higher_is_better else -percent_diff
                    assessment = "positive" if better > 10 else "negative" if better < -10 else "neutral"
                analysis[category][name] = {
                    "value": value,
                    "benchmark": benchmark,
                    "percent_diff": percent_diff,
                    "assessment": assessment,
                    "color": ASSESSMENT_COLORS[assessment],
                    "higher_is_better": higher_is_better,
                }
        return analysis

    def get_key_ratios_for_sector(self, sector: Optional[str]) -> List[Dict[str, Optional[str]]]:
        """
        Most important ratios for a sector (``config.SECTOR_SPECIFIC_RATIOS``).

        Returns:
            List of ``{"ratio", "category", "description", "key"}`` where ``key``
            is the matching key in :meth:`calculate_ratios` output (under
            ``category.lower()``) or None for company-reported KPIs that cannot
            be derived from financial statements.
        """
        result = []
        for ratio in self.sector_specific_ratios.get(normalize_sector_name(sector), []):
            category, key = RATIO_KEYS.get(ratio, (None, None))
            if category is None:
                category = next((cat for cat, names in self.ratio_categories.items() if ratio in names), "Other")
            result.append({"ratio": ratio, "category": category, "description": self._get_ratio_description(ratio),
                           "key": key})
        return result

    @staticmethod
    def _get_ratio_description(ratio: str) -> str:
        return RATIO_DESCRIPTIONS.get(ratio, "No description available")
