"""
Financial statement analysis: trends, growth, common-size statements and a
0-100 financial health score.

Statements use the yfinance layout (index = line items, columns = period
dates); every line item lookup goes through the alias-aware helpers in
:mod:`StockAnalysisSystem.src.utils.financial_data`.

A *trend dict* (used throughout the returned structures) is::

    {"values": {period: float or None},     # most recent period first
     "latest": float or None,               # most recent period
     "direction": "up" | "down" | "flat",   # slope of a linear fit vs. the mean level
     "trend": "improving" | "deteriorating" | "stable"}  # direction oriented by desirability
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from StockAnalysisSystem.src.models.ratio_analysis import annualized_growth
from StockAnalysisSystem.src.utils import financial_data as fd

logger = logging.getLogger(__name__)

#: Relative slope per period (vs. mean absolute level) below which a series is "flat".
TREND_TOLERANCE = 0.02

COMPONENT_WEIGHTS = {"profitability": 0.30, "liquidity": 0.20, "solvency": 0.25, "cash_flow": 0.25}
COMPONENT_LABELS = {"profitability": "Profitability", "liquidity": "Liquidity", "solvency": "Solvency",
                    "cash_flow": "Cash flow"}

# Piecewise-linear scoring curves: (metric anchors, scores)
_SCORE_CURVES = {
    "net_margin": ([-0.10, 0.0, 0.05, 0.10, 0.15, 0.20], [0, 30, 55, 75, 90, 100]),
    "operating_margin": ([-0.10, 0.0, 0.05, 0.10, 0.20, 0.30], [0, 30, 50, 65, 85, 100]),
    "revenue_growth": ([-0.10, 0.0, 0.05, 0.10, 0.20], [0, 40, 60, 80, 100]),
    "current_ratio": ([0.5, 0.8, 1.0, 1.5, 2.0], [0, 30, 50, 80, 100]),
    "cash_ratio": ([0.0, 0.1, 0.2, 0.5, 1.0], [0, 30, 50, 80, 100]),
    "debt_to_equity": ([0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0], [100, 100, 90, 70, 45, 25, 0]),
    "liabilities_to_assets": ([0.3, 0.5, 0.7, 0.9, 1.0], [100, 80, 55, 25, 0]),
    "ocf_to_net_income": ([0.0, 0.6, 0.8, 1.0, 1.2], [0, 40, 70, 90, 100]),
    "interest_coverage": ([1.0, 1.5, 3.0, 8.0, 15.0], [0, 20, 50, 85, 100]),
}


def _score(metric: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    anchors, scores = _SCORE_CURVES[metric]
    return float(np.interp(value, anchors, scores))


def _numeric(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Numeric copy with columns ordered most recent first."""
    if fd.is_empty(df):
        return pd.DataFrame()
    ordered = fd.chronological(df)
    ordered = ordered[ordered.columns[::-1]]
    return ordered.apply(pd.to_numeric, errors="coerce").astype(float)


def _period_label(period: Any) -> str:
    try:
        return pd.Timestamp(period).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(period)


def _clean(value: Any) -> Optional[float]:
    return fd.to_float(value)


def _trend(series: pd.Series, higher_is_better: bool = True) -> Dict[str, Any]:
    """Trend dict for a series ordered most recent first."""
    values = {period: _clean(v) for period, v in series.items()}
    chronological = pd.Series(list(values.values())[::-1], dtype=float).dropna()
    direction = "flat"
    if len(chronological) >= 2:
        slope = np.polyfit(np.arange(len(chronological)), chronological.to_numpy(), 1)[0]
        level = float(np.mean(np.abs(chronological)))
        relative = slope / level if level > 0 else 0.0
        if relative > TREND_TOLERANCE:
            direction = "up"
        elif relative < -TREND_TOLERANCE:
            direction = "down"
    if direction == "flat":
        trend = "stable"
    else:
        trend = "improving" if (direction == "up") == higher_is_better else "deteriorating"
    latest = next(iter(values.values()), None) if values else None
    return {"values": values, "latest": latest, "direction": direction, "trend": trend}


def _row(df: pd.DataFrame, item: fd.ItemSpec) -> Optional[pd.Series]:
    label = fd.find_row(df, item)
    if label is None:
        return None
    row = df.loc[label]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _metric_trend(df: pd.DataFrame, item: fd.ItemSpec, higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
    row = _row(df, item)
    return _trend(row, higher_is_better) if row is not None else None


def _ratio_trend(numerator: Optional[pd.Series], denominator: Optional[pd.Series],
                 higher_is_better: bool = True, positive_denominator: bool = True) -> Optional[Dict[str, Any]]:
    if numerator is None or denominator is None:
        return None
    denominator = denominator.where(denominator > 0) if positive_denominator else denominator.where(denominator != 0)
    ratio = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
    if ratio.dropna().empty:
        return None
    return _trend(ratio, higher_is_better)


def _growth_table(df: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period growth ``(current - previous) / |previous|`` for every row."""
    columns = {}
    for current, previous in zip(df.columns[:-1], df.columns[1:]):
        base = df[previous].where(df[previous] != 0)
        columns[f"{_period_label(current)} vs {_period_label(previous)}"] = (df[current] - base) / base.abs()
    return pd.DataFrame(columns, index=df.index).replace([np.inf, -np.inf], np.nan)


def _growth_trend(growth: pd.DataFrame, df: pd.DataFrame, item: fd.ItemSpec) -> Optional[Dict[str, Any]]:
    label = fd.find_row(df, item)
    if label is None or growth.empty or label not in growth.index:
        return None
    row = growth.loc[label]
    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    return _trend(row)


def _common_size(df: pd.DataFrame, base: Optional[pd.Series]) -> pd.DataFrame:
    if base is None:
        return pd.DataFrame(index=df.index)
    base = base.where(base != 0)
    return df.divide(base, axis=1).replace([np.inf, -np.inf], np.nan)


def _row_cagr(df: pd.DataFrame) -> pd.Series:
    """CAGR of every row over its available periods (NaN when not meaningful)."""
    chronological = df[df.columns[::-1]]
    values = [annualized_growth(row.dropna()) for _, row in chronological.iterrows()]
    return pd.Series([np.nan if v is None else v for v in values], index=df.index, dtype=float)


class FinancialStatementAnalyzer:
    """Analyse income statements, balance sheets and cash flow statements."""

    # ------------------------------------------------------------ income statement
    def analyze_income_statement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse an income statement.

        Returns:
            ``{}`` for empty input, otherwise::

                {"key_metrics": {
                    "revenue" | "gross_profit" | "operating_income" | "net_income": trend dict,
                    "margins": {"gross_margin", "operating_margin", "net_margin": trend dict},
                    "growth_rates": {"revenue_growth", "net_income_growth": trend dict},
                    "cagr": {"revenue_cagr", "net_income_cagr": float or None},
                    "efficiency": {"rnd_to_revenue": trend dict},       # when R&D is reported
                    "coverage": {"interest_coverage": trend dict}},     # when interest is reported
                 "growth_rates": DataFrame (rows = line items, cols = "<period> vs <prior>"),
                 "common_size": DataFrame (each row / revenue),
                 "cagr": Series (per line item),
                 "periods": [period, ...]}                              # most recent first

            Trend dicts are None when the line item is missing.
        """
        data = _numeric(df)
        if data.empty:
            return {}
        try:
            revenue = _row(data, "revenue")
            net_income = _row(data, "net_income")
            growth = _growth_table(data) if data.shape[1] >= 2 else pd.DataFrame(index=data.index)
            key_metrics: Dict[str, Any] = {
                "revenue": _metric_trend(data, "revenue"),
                "gross_profit": _metric_trend(data, "gross_profit"),
                "operating_income": _metric_trend(data, "operating_income"),
                "net_income": _metric_trend(data, "net_income"),
                "margins": {
                    "gross_margin": _ratio_trend(_row(data, "gross_profit"), revenue),
                    "operating_margin": _ratio_trend(_row(data, "operating_income"), revenue),
                    "net_margin": _ratio_trend(net_income, revenue),
                },
                "growth_rates": {
                    "revenue_growth": _growth_trend(growth, data, "revenue"),
                    "net_income_growth": _growth_trend(growth, data, "net_income"),
                },
                "cagr": {
                    "revenue_cagr": annualized_growth(fd.get_series(data, "revenue", ascending=True)),
                    "net_income_cagr": annualized_growth(fd.get_series(data, "net_income", ascending=True)),
                },
            }
            rnd = _row(data, "rd_expense")
            if rnd is not None:
                key_metrics["efficiency"] = {"rnd_to_revenue": _ratio_trend(rnd.abs(), revenue)}
            interest = _row(data, "interest_expense")
            if interest is not None:
                key_metrics["coverage"] = {"interest_coverage": _ratio_trend(_row(data, "ebit"), interest.abs())}
            return {
                "key_metrics": key_metrics,
                "growth_rates": growth,
                "common_size": _common_size(data, revenue),
                "cagr": _row_cagr(data),
                "periods": list(data.columns),
            }
        except Exception as e:
            logger.error(f"Error analyzing income statement: {e}")
            return {}

    # --------------------------------------------------------------- balance sheet
    def analyze_balance_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse a balance sheet.

        Returns:
            ``{}`` for empty input, otherwise::

                {"key_metrics": {
                    "assets": {"total_assets", "current_assets", "cash_equivalents"},
                    "liabilities": {"total_liabilities", "current_liabilities", "long_term_debt", "total_debt"},
                    "equity": {"total_equity"},
                    "ratios": {"current_ratio", "quick_ratio", "cash_ratio",
                               "debt_to_equity",          # total debt / equity
                               "debt_to_assets",          # total debt / total assets
                               "liabilities_to_equity", "liabilities_to_assets"}},   # all trend dicts
                 "growth_rates": DataFrame, "common_size": DataFrame (each row / total assets),
                 "periods": [...]}

            Ratios with a non-positive denominator (e.g. negative equity) are
            missing for that period.
        """
        data = _numeric(df)
        if data.empty:
            return {}
        try:
            assets = _row(data, "total_assets")
            current_assets = _row(data, "current_assets")
            current_liabilities = _row(data, "current_liabilities")
            liabilities = _row(data, "total_liabilities")
            equity = _row(data, "total_equity")
            debt = _row(data, "total_debt")
            if debt is None:
                parts = [p for p in (_row(data, "long_term_debt"), _row(data, "short_term_debt")) if p is not None]
                debt = sum(p.fillna(0) for p in parts) if parts else None
            inventory = _row(data, "inventory")
            quick_assets = None
            if current_assets is not None:
                quick_assets = current_assets - (inventory.fillna(0) if inventory is not None else 0)
            cash = _row(data, "cash_and_short_term_investments")
            key_metrics = {
                "assets": {
                    "total_assets": _metric_trend(data, "total_assets"),
                    "current_assets": _metric_trend(data, "current_assets"),
                    "cash_equivalents": _metric_trend(data, "cash"),
                },
                "liabilities": {
                    "total_liabilities": _metric_trend(data, "total_liabilities", higher_is_better=False),
                    "current_liabilities": _metric_trend(data, "current_liabilities", higher_is_better=False),
                    "long_term_debt": _metric_trend(data, "long_term_debt", higher_is_better=False),
                    "total_debt": _trend(debt, higher_is_better=False) if debt is not None else None,
                },
                "equity": {"total_equity": _metric_trend(data, "total_equity")},
                "ratios": {
                    "current_ratio": _ratio_trend(current_assets, current_liabilities),
                    "quick_ratio": _ratio_trend(quick_assets, current_liabilities),
                    "cash_ratio": _ratio_trend(cash, current_liabilities),
                    "debt_to_equity": _ratio_trend(debt, equity, higher_is_better=False),
                    "debt_to_assets": _ratio_trend(debt, assets, higher_is_better=False),
                    "liabilities_to_equity": _ratio_trend(liabilities, equity, higher_is_better=False),
                    "liabilities_to_assets": _ratio_trend(liabilities, assets, higher_is_better=False),
                },
            }
            return {
                "key_metrics": key_metrics,
                "growth_rates": _growth_table(data) if data.shape[1] >= 2 else pd.DataFrame(index=data.index),
                "common_size": _common_size(data, assets),
                "periods": list(data.columns),
            }
        except Exception as e:
            logger.error(f"Error analyzing balance sheet: {e}")
            return {}

    # ------------------------------------------------------------------ cash flow
    def analyze_cash_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse a cash flow statement.

        Returns:
            ``{}`` for empty input, otherwise::

                {"key_metrics": {
                    "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
                    "capital_expenditure", "free_cash_flow": trend dict or None,
                    "cf_quality": {"ocf_to_net_income": trend dict}},   # when net income is reported
                 "growth_rates": DataFrame, "periods": [...]}

            Free cash flow is the reported row or operating cash flow - |capex|.
        """
        data = _numeric(df)
        if data.empty:
            return {}
        try:
            ocf = _row(data, "operating_cash_flow")
            capex = _row(data, "capital_expenditure")
            fcf = _row(data, "free_cash_flow")
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf - capex.abs()
            key_metrics: Dict[str, Any] = {
                "operating_cash_flow": _trend(ocf) if ocf is not None else None,
                "investing_cash_flow": _metric_trend(data, "investing_cash_flow"),
                "financing_cash_flow": _metric_trend(data, "financing_cash_flow"),
                # capex is reported negative; spending more is neither good nor bad per se
                "capital_expenditure": _trend(capex) if capex is not None else None,
                "free_cash_flow": _trend(fcf) if fcf is not None else None,
            }
            net_income = _row(data, ["net_income", "Net Income From Continuing Operations"])
            if net_income is not None and ocf is not None:
                key_metrics["cf_quality"] = {"ocf_to_net_income": _ratio_trend(ocf, net_income)}
            return {
                "key_metrics": key_metrics,
                "growth_rates": _growth_table(data) if data.shape[1] >= 2 else pd.DataFrame(index=data.index),
                "periods": list(data.columns),
            }
        except Exception as e:
            logger.error(f"Error analyzing cash flow statement: {e}")
            return {}

    # --------------------------------------------------------------- health score
    def calculate_financial_health_score(self, income_data: Dict, balance_data: Dict,
                                         cash_flow_data: Dict) -> Dict[str, Any]:
        """
        Financial health score from the outputs of the three ``analyze_*`` methods.

        Each component is scored 0-100 from piecewise-linear curves:

        * profitability: net margin, operating margin, revenue growth (+/-5 for the net margin trend)
        * liquidity: current ratio, cash ratio
        * solvency: total debt / equity (0 for negative equity with debt), liabilities / assets,
          interest coverage
        * cash_flow: free cash flow sign and trend, operating cash flow / net income

        Returns:
            ``{"components": {"profitability", "liquidity", "solvency", "cash_flow": int or None},
            "overall_score": float or None (weighted 30/20/25/25 over available components),
            "rating": "Excellent" | "Strong" | "Good" | "Moderate" | "Weak" | "Poor" | None,
            "details": {component: {metric: score}}, "strengths": [str], "weaknesses": [str],
            "recommendations": [str], "assessment_summary": str}``
        """
        details: Dict[str, Dict[str, float]] = {name: {} for name in COMPONENT_WEIGHTS}
        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []
        try:
            self._score_profitability(income_data, details["profitability"], strengths, weaknesses, recommendations)
            self._score_liquidity(balance_data, details["liquidity"], strengths, weaknesses, recommendations)
            self._score_solvency(balance_data, income_data, details["solvency"], strengths, weaknesses,
                                 recommendations)
            self._score_cash_flow(cash_flow_data, details["cash_flow"], strengths, weaknesses, recommendations)
        except Exception as e:
            logger.error(f"Error calculating financial health score: {e}")

        components: Dict[str, Optional[int]] = {}
        for name, scores in details.items():
            components[name] = int(round(min(max(np.mean(list(scores.values())), 0), 100))) if scores else None

        available = {k: v for k, v in components.items() if v is not None}
        overall = None
        if available:
            total_weight = sum(COMPONENT_WEIGHTS[k] for k in available)
            overall = round(sum(COMPONENT_WEIGHTS[k] * v for k, v in available.items()) / total_weight, 1)
        rating = self._rating(overall)

        if overall is None:
            summary = "Insufficient financial data to assess financial health."
        else:
            best = max(available, key=available.get)
            worst = min(available, key=available.get)
            summary = f"Overall financial health is {rating.lower()} ({overall:.1f}/100)."
            if best != worst:
                summary += (f" Strongest area: {COMPONENT_LABELS[best].lower()} ({available[best]}/100); "
                            f"weakest area: {COMPONENT_LABELS[worst].lower()} ({available[worst]}/100).")
        return {
            "components": components,
            "overall_score": overall,
            "rating": rating,
            "details": {k: {m: round(s, 1) for m, s in v.items()} for k, v in details.items()},
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "assessment_summary": summary,
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _latest(container: Optional[Dict], *path: str) -> Optional[Dict[str, Any]]:
        node: Any = container or {}
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                return None
        return node if isinstance(node, dict) else None

    def _latest_value(self, container: Optional[Dict], *path: str) -> Optional[float]:
        node = self._latest(container, *path)
        return _clean(node.get("latest")) if node else None

    def _score_profitability(self, income, scores, strengths, weaknesses, recommendations) -> None:
        net_margin = self._latest_value(income, "key_metrics", "margins", "net_margin")
        operating_margin = self._latest_value(income, "key_metrics", "margins", "operating_margin")
        revenue_growth = self._latest_value(income, "key_metrics", "growth_rates", "revenue_growth")
        if net_margin is not None:
            score = _score("net_margin", net_margin)
            trend = (self._latest(income, "key_metrics", "margins", "net_margin") or {}).get("trend")
            score += 5 if trend == "improving" else -5 if trend == "deteriorating" else 0
            scores["net_margin"] = min(max(score, 0), 100)
            if net_margin >= 0.15:
                strengths.append(f"High net profit margin ({net_margin:.1%})")
            elif net_margin < 0:
                weaknesses.append(f"Net loss (net margin {net_margin:.1%})")
                recommendations.append("Monitor the path to profitability and cost structure.")
            elif net_margin < 0.05:
                weaknesses.append(f"Thin net profit margin ({net_margin:.1%})")
        if operating_margin is not None:
            scores["operating_margin"] = _score("operating_margin", operating_margin)
        if revenue_growth is not None:
            scores["revenue_growth"] = _score("revenue_growth", revenue_growth)
            if revenue_growth >= 0.10:
                strengths.append(f"Strong revenue growth ({revenue_growth:.1%} year over year)")
            elif revenue_growth < 0:
                weaknesses.append(f"Declining revenue ({revenue_growth:.1%} year over year)")
                recommendations.append("Investigate the drivers of the revenue decline.")

    def _score_liquidity(self, balance, scores, strengths, weaknesses, recommendations) -> None:
        current_ratio = self._latest_value(balance, "key_metrics", "ratios", "current_ratio")
        cash_ratio = self._latest_value(balance, "key_metrics", "ratios", "cash_ratio")
        if current_ratio is not None:
            scores["current_ratio"] = _score("current_ratio", current_ratio)
            if current_ratio >= 2:
                strengths.append(f"Strong short-term liquidity (current ratio {current_ratio:.2f})")
            elif current_ratio < 1:
                weaknesses.append(f"Current liabilities exceed current assets (current ratio {current_ratio:.2f})")
                recommendations.append("Check short-term funding needs and working capital management.")
        if cash_ratio is not None:
            scores["cash_ratio"] = _score("cash_ratio", cash_ratio)

    def _score_solvency(self, balance, income, scores, strengths, weaknesses, recommendations) -> None:
        equity = self._latest_value(balance, "key_metrics", "equity", "total_equity")
        debt = self._latest_value(balance, "key_metrics", "liabilities", "total_debt")
        debt_to_equity = self._latest_value(balance, "key_metrics", "ratios", "debt_to_equity")
        liabilities_to_assets = self._latest_value(balance, "key_metrics", "ratios", "liabilities_to_assets")
        coverage = self._latest_value(income, "key_metrics", "coverage", "interest_coverage")
        if equity is not None and equity <= 0:
            scores["debt_to_equity"] = 0.0 if debt else 30.0
            weaknesses.append("Negative shareholders' equity (liabilities exceed assets)")
            recommendations.append("Assess balance sheet solvency and refinancing risk.")
        elif debt_to_equity is not None:
            scores["debt_to_equity"] = _score("debt_to_equity", debt_to_equity)
            if debt_to_equity <= 0.5:
                strengths.append(f"Conservative leverage (debt/equity {debt_to_equity:.2f})")
            elif debt_to_equity > 2:
                weaknesses.append(f"High leverage (debt/equity {debt_to_equity:.2f})")
                recommendations.append("Review debt maturities and the cost of debt.")
        if liabilities_to_assets is not None:
            scores["liabilities_to_assets"] = _score("liabilities_to_assets", liabilities_to_assets)
        if coverage is not None:
            scores["interest_coverage"] = _score("interest_coverage", coverage)
            if coverage < 1.5:
                weaknesses.append(f"Operating earnings barely cover interest (coverage {coverage:.1f}x)")

    def _score_cash_flow(self, cash_flow, scores, strengths, weaknesses, recommendations) -> None:
        fcf_node = self._latest(cash_flow, "key_metrics", "free_cash_flow")
        fcf = _clean(fcf_node.get("latest")) if fcf_node else None
        if fcf is not None:
            if fcf > 0:
                trend = fcf_node.get("trend")
                scores["free_cash_flow"] = 75 + (15 if trend == "improving" else -15 if trend == "deteriorating" else 0)
                strengths.append("Positive free cash flow")
            else:
                scores["free_cash_flow"] = 15.0
                weaknesses.append("Negative free cash flow")
                recommendations.append("Evaluate funding needs: the business does not self-fund its investment.")
        ratio = self._latest_value(cash_flow, "key_metrics", "cf_quality", "ocf_to_net_income")
        ocf = self._latest_value(cash_flow, "key_metrics", "operating_cash_flow")
        if ratio is not None:
            scores["ocf_to_net_income"] = _score("ocf_to_net_income", ratio)
            if ratio >= 1.1:
                strengths.append(f"High earnings quality (operating cash flow {ratio:.2f}x net income)")
            elif ratio < 0.8:
                weaknesses.append(f"Earnings not fully backed by cash (operating cash flow {ratio:.2f}x net income)")
        elif ocf is not None and ocf <= 0:
            scores["ocf_to_net_income"] = 0.0
            weaknesses.append("Negative operating cash flow")

    @staticmethod
    def _rating(score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        for threshold, rating in ((90, "Excellent"), (75, "Strong"), (60, "Good"), (45, "Moderate"), (30, "Weak")):
            if score >= threshold:
                return rating
        return "Poor"
