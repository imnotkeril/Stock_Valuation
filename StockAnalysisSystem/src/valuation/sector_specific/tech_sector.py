"""
Valuation of technology companies.

Mature technology companies are valued with a multi-stage DCF on free cash
flow net of stock-based compensation (SBC is a real cost to shareholders),
cross-checked with subsector peer multiples. SaaS adds a Rule of 40-adjusted
EV/Revenue multiple; unprofitable growth companies use a revenue-driven DCF
in which operating margins converge to a target and growth is funded through
a sales-to-capital reinvestment ratio.
"""

import logging
from typing import Any, Dict, List, Optional

from StockAnalysisSystem.src.industry.universe import lookup_company
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.financial_data import get_latest, get_series, get_value, safe_divide
from StockAnalysisSystem.src.valuation.base_valuation import clamp, growth_path
from StockAnalysisSystem.src.valuation.dcf_models import SectorSpecificDCF

logger = logging.getLogger(__name__)

#: Peer multiples by subsector (long-run medians).
PEER_MULTIPLES: Dict[str, Dict[str, float]] = {
    "saas": {"ev_revenue": 7.0, "ev_ebitda": 25.0, "pe": 32.0},
    "hardware": {"ev_revenue": 3.0, "ev_ebitda": 15.0, "pe": 22.0},
    "semiconductors": {"ev_revenue": 6.0, "ev_ebitda": 18.0, "pe": 25.0},
    "internet": {"ev_revenue": 5.0, "ev_ebitda": 16.0, "pe": 24.0},
    "it_services": {"ev_revenue": 2.0, "ev_ebitda": 14.0, "pe": 22.0},
    "early_stage": {"ev_revenue": 4.0, "ev_ebitda": 20.0, "pe": 30.0},
}

#: Industry keywords (yfinance names, lower case) -> subsector, checked in order.
INDUSTRY_KEYWORDS = (
    (("semiconductor",), "semiconductors"),
    (("software", "saas", "cloud"), "saas"),
    (("internet", "social media", "e-commerce", "interactive media"), "internet"),
    (("information technology services", "it services", "consulting"), "it_services"),
    (("consumer electronics", "computer hardware", "communication equipment", "electronic component",
      "scientific & technical instruments", "electronics", "hardware", "solar", "equipment"), "hardware"),
)
UNIVERSE_SUBSECTORS = {"Software": "saas", "Hardware": "hardware", "Semiconductors": "semiconductors",
                       "IT Services": "it_services", "Internet": "internet"}

PROFITABLE_SUBSECTORS = ["saas", "hardware", "semiconductors", "internet", "it_services"]


class TechnologySectorValuation(SectorSpecificDCF):
    """Valuation of software, hardware, semiconductor, internet and IT services companies."""

    SECTOR = "Technology"

    SUBSECTORS: Dict[str, str] = {
        "saas": "Software / SaaS",
        "hardware": "Hardware & Consumer Electronics",
        "semiconductors": "Semiconductors",
        "internet": "Internet & Digital Platforms",
        "it_services": "IT Services & Consulting",
        "early_stage": "Early-Stage / Unprofitable Growth",
    }

    SUBSECTOR_ALIASES = {"software": "saas", "semiconductor": "semiconductors", "startup": "early_stage",
                         "tech_startup": "early_stage", "it services": "it_services"}

    METHODS: Dict[str, str] = {
        "dcf": "DCF (SBC-adjusted FCF)",
        "revenue_dcf": "Revenue-Driven DCF (Margin Convergence)",
        "rule_of_40": "Rule of 40-Adjusted EV/Revenue",
        "ev_revenue": "EV/Revenue Multiple",
        "ev_ebitda": "EV/EBITDA Multiple",
        "pe_multiple": "P/E Multiple",
    }

    SUBSECTOR_METHODS: Dict[str, List[str]] = {
        "saas": ["dcf", "rule_of_40", "ev_revenue", "ev_ebitda"],
        "hardware": ["dcf", "pe_multiple", "ev_ebitda"],
        "semiconductors": ["dcf", "ev_ebitda", "pe_multiple"],
        "internet": ["dcf", "ev_ebitda", "ev_revenue", "pe_multiple"],
        "it_services": ["dcf", "pe_multiple", "ev_ebitda"],
        "early_stage": ["revenue_dcf", "rule_of_40", "ev_revenue"],
    }

    DEFAULT_METHOD_WEIGHTS: Dict[str, float] = {
        "dcf": 0.50, "revenue_dcf": 0.50, "rule_of_40": 0.20, "ev_revenue": 0.10, "ev_ebitda": 0.20,
        "pe_multiple": 0.20,
    }

    SUBSECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
        "saas": {"dcf": 0.45, "rule_of_40": 0.25, "ev_revenue": 0.10, "ev_ebitda": 0.20},
        "hardware": {"dcf": 0.50, "pe_multiple": 0.25, "ev_ebitda": 0.25},
        "semiconductors": {"dcf": 0.45, "ev_ebitda": 0.30, "pe_multiple": 0.25},
        "internet": {"dcf": 0.45, "ev_ebitda": 0.25, "ev_revenue": 0.10, "pe_multiple": 0.20},
        "it_services": {"dcf": 0.50, "pe_multiple": 0.30, "ev_ebitda": 0.20},
        "early_stage": {"revenue_dcf": 0.50, "rule_of_40": 0.30, "ev_revenue": 0.20},
    }

    PARAMETERS: Dict[str, Dict[str, Any]] = {
        "discount_rate": {"label": "Discount rate (WACC)", "min": 0.06, "max": 0.16, "default": 0.10,
                          "step": 0.0025, "kind": "percent", "subsectors": None,
                          "help": "Rate used to discount free cash flow. Derived from CAPM/WACC when not set."},
        "terminal_growth": {"label": "Terminal growth rate", "min": 0.0, "max": 0.05, "default": 0.03,
                            "step": 0.0025, "kind": "percent", "subsectors": None,
                            "help": "Perpetual growth after the forecast; must stay below the discount rate."},
        "forecast_years": {"label": "Forecast horizon (years)", "min": 5, "max": 15, "default": 10, "step": 1,
                           "kind": "int", "subsectors": None,
                           "help": "High-growth years plus the fade to terminal growth (split evenly)."},
        "revenue_growth": {"label": "Near-term growth", "min": -0.10, "max": 0.60, "default": 0.10, "step": 0.01,
                           "kind": "percent", "subsectors": None,
                           "help": "Year-one growth of revenue and cash flow. Derived from history when not set."},
        "sbc_haircut": {"label": "SBC treated as cash cost", "min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05,
                        "kind": "percent", "subsectors": PROFITABLE_SUBSECTORS,
                        "help": "Share of stock-based compensation deducted from free cash flow."},
        "target_operating_margin": {"label": "Target operating margin", "min": 0.05, "max": 0.45, "default": 0.20,
                                    "step": 0.01, "kind": "percent", "subsectors": ["early_stage"],
                                    "help": "Operating margin reached at the end of the forecast."},
        "multiple_adjustment": {"label": "Peer multiple premium / discount", "min": -0.5, "max": 0.5,
                                "default": 0.0, "step": 0.05, "kind": "percent", "subsectors": None,
                                "help": "Applied to all peer multiples (e.g. +20 % for a market leader)."},
    }

    def __init__(self, data_loader: Optional[DataLoader] = None):
        super().__init__(data_loader)

    # ------------------------------------------------------------------
    # Subsector detection
    # ------------------------------------------------------------------
    def detect_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                         company_info: Optional[Dict[str, Any]] = None) -> str:
        """Loss-making growth -> early_stage; otherwise industry keywords, universe, then gross margin."""
        info = company_info or self._company_info(financial_data)
        metrics = self._tech_metrics(financial_data)
        operating_margin, growth = metrics['operating_margin'], metrics['revenue_growth']
        net_income = metrics['net_income']
        if (operating_margin is not None and operating_margin < 0) or \
                (net_income is not None and net_income < 0 and (growth or 0) > 0.25):
            return 'early_stage'

        industry = str(info.get('industry') or '').lower()
        for keywords, subsector in INDUSTRY_KEYWORDS:
            if any(keyword in industry for keyword in keywords):
                return subsector
        record = lookup_company(str(ticker).upper()) or {}
        if record.get('subsector') in UNIVERSE_SUBSECTORS:
            return UNIVERSE_SUBSECTORS[record['subsector']]
        gross_margin = metrics['gross_margin']
        return 'hardware' if gross_margin is not None and gross_margin < 0.5 else 'saas'

    # ------------------------------------------------------------------
    # Valuation entry point
    # ------------------------------------------------------------------
    def get_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                      subsector: Optional[str] = None, sector: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Value a technology company with the methods of its subsector.

        Keyword overrides: ``discount_rate``, ``terminal_growth``,
        ``forecast_years``, ``revenue_growth``, ``sbc_haircut``,
        ``target_operating_margin``, ``multiple_adjustment``, plus ``methods``
        and ``weights``.
        """
        def prepare(data, key, overrides, notes):
            metrics = self._tech_metrics(data)
            params = self._get_dcf_parameters(self.SECTOR)
            wacc = self._wacc_components(data, self.SECTOR)
            discount_rate = overrides.get('discount_rate')
            terminal_growth = overrides.get('terminal_growth')
            forecast_years = overrides.get('forecast_years') or 10
            growth = overrides.get('revenue_growth')
            haircut = overrides.get('sbc_haircut', 1.0)
            adjustment = 1 + overrides.get('multiple_adjustment', 0.0)
            multiples = {k: v * adjustment for k, v in PEER_MULTIPLES[key].items()}
            rate_kwargs = {'discount_rate': discount_rate, 'terminal_growth': terminal_growth,
                           'forecast_years': forecast_years}

            runners = {
                'dcf': lambda: self.tech_sector_dcf(ticker, data, sbc_haircut=haircut, growth_rate=growth,
                                                    average_years=3 if key == 'semiconductors' else 1,
                                                    **rate_kwargs),
                'revenue_dcf': lambda: self._revenue_driven_dcf(
                    ticker, data, metrics, growth_rate=growth,
                    target_margin=overrides.get('target_operating_margin', 0.20), **rate_kwargs),
                'rule_of_40': lambda: self._rule_of_40_valuation(ticker, data, metrics, multiples['ev_revenue']),
                'ev_revenue': lambda: self._multiples_valuation(ticker, data, {'ev_revenue': multiples['ev_revenue']},
                                                                label=self.METHODS['ev_revenue']),
                'ev_ebitda': lambda: self._multiples_valuation(ticker, data, {'ev_ebitda': multiples['ev_ebitda']},
                                                               label=self.METHODS['ev_ebitda']),
                'pe_multiple': lambda: self._multiples_valuation(ticker, data, {'pe': multiples['pe']},
                                                                 label=self.METHODS['pe_multiple']),
            }
            if key != 'early_stage' and metrics['sbc_pct_revenue']:
                notes.append(f"Free cash flow is reduced by {haircut:.0%} of stock-based compensation "
                             f"({metrics['sbc_pct_revenue']:.1%} of revenue).")
            if key == 'early_stage':
                notes.append("Revenue-driven DCF ignores failure risk and tax loss carry-forwards; "
                             "consider a higher discount rate for unproven business models.")
            if key == 'semiconductors':
                notes.append("Semiconductor cash flows are cyclical; the DCF starts from a three-year average.")
            assumptions = {
                'discount_rate': discount_rate if discount_rate is not None else wacc['wacc'],
                'terminal_growth': terminal_growth if terminal_growth is not None else params['terminal_growth_rate'],
                'forecast_years': forecast_years,
                'revenue_growth': growth, 'sbc_haircut': haircut, 'peer_multiples': multiples,
                'wacc_components': wacc, 'metrics': metrics,
            }
            if key == 'early_stage':
                assumptions['target_operating_margin'] = overrides.get('target_operating_margin', 0.20)
            return runners, assumptions

        return self._run_sector_valuation(ticker, financial_data, subsector, sector, kwargs, prepare)

    # Backward-compatible entry points ------------------------------------
    def value_tech_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Value with the detected subsector."""
        return self.get_valuation(ticker, financial_data)

    def value_saas_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='saas')

    def value_hardware_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='hardware')

    def value_semiconductor_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='semiconductors')

    def value_internet_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='internet')

    def value_tech_startup(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='early_stage')

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _tech_metrics(self, financial_data: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Growth, margin and Rule of 40 metrics from the latest statements."""
        income, balance, cash_flow = self._statements(financial_data)
        revenue = get_latest(income, 'revenue')
        previous = get_value(income, 'revenue', 1)
        growth = revenue / previous - 1 if revenue and previous and previous > 0 else None
        fcf_series = self._calculate_historical_fcf(income, cash_flow)
        fcf = float(fcf_series.iloc[0]) if not fcf_series.empty else None
        fcf_margin = safe_divide(fcf, revenue) if revenue and revenue > 0 else None
        sbc = get_latest(cash_flow, 'stock_based_compensation')
        invested = (get_latest(balance, 'total_equity') or 0.0) + self._total_debt(balance) - self._cash(balance)

        def margin(item):
            value = get_latest(income, item)
            return safe_divide(value, revenue) if revenue and revenue > 0 else None

        return {
            'revenue': revenue, 'revenue_growth': growth,
            'revenue_cagr': self._estimate_company_growth(financial_data) if revenue else None,
            'gross_margin': margin('gross_profit'), 'operating_margin': margin('operating_income'),
            'net_income': get_latest(income, 'net_income'), 'fcf_margin': fcf_margin,
            'rd_pct_revenue': margin('rd_expense'),
            'sbc_pct_revenue': abs(sbc) / revenue if sbc and revenue and revenue > 0 else None,
            'rule_of_40_score': growth + fcf_margin if growth is not None and fcf_margin is not None else None,
            'sales_to_capital': clamp(revenue / invested, 0.5, 5.0) if revenue and invested > 0 else None,
        }

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def tech_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None, sbc_haircut: float = 1.0,
                        discount_rate: Optional[float] = None, terminal_growth: Optional[float] = None,
                        forecast_years: Optional[int] = None, growth_rate: Optional[float] = None,
                        average_years: int = 1, **kwargs) -> Dict[str, Any]:
        """SBC-adjusted multi-stage DCF (see SectorSpecificDCF.tech_sector_dcf); optional cycle averaging."""
        label = self.METHODS['dcf']
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            series = self._sbc_adjusted_fcf(financial_data, sbc_haircut)
            if average_years > 1:
                series = series.copy()
                series.iloc[0] = float(series.head(average_years).mean())
            result = self.multi_stage_dcf_valuation(ticker, financial_data, self.SECTOR, discount_rate,
                                                    terminal_growth, forecast_years or 10, growth_rate,
                                                    cash_flows=series, label=label)
            result['sbc_haircut'] = sbc_haircut
            return result
        except Exception as e:
            return self._method_error(label, e, ticker)

    def _revenue_driven_dcf(self, ticker: str, financial_data: Dict[str, Any], metrics: Dict[str, Any],
                            growth_rate: Optional[float] = None, target_margin: float = 0.20,
                            discount_rate: Optional[float] = None, terminal_growth: Optional[float] = None,
                            forecast_years: Optional[int] = None) -> Dict[str, Any]:
        """
        Revenue-driven DCF for unprofitable growth companies.

        Revenue growth fades to terminal growth, the operating margin moves
        linearly from today's level to ``target_margin``, taxes apply only to
        positive EBIT, and reinvestment = change in revenue / sales-to-capital.
        Terminal reinvestment rate = g / ROIC with ROIC = discount rate + 2 %.
        """
        label = self.METHODS['revenue_dcf']
        try:
            revenue = metrics['revenue']
            if not revenue or revenue <= 0:
                raise ValueError("Revenue is not positive")
            params = self._get_dcf_parameters(self.SECTOR)
            rate = discount_rate if discount_rate is not None else self._wacc_components(financial_data)['wacc']
            g_terminal = terminal_growth if terminal_growth is not None else params['terminal_growth_rate']
            years = int(forecast_years or 10)
            growth = growth_rate if growth_rate is not None else clamp(
                metrics['revenue_cagr'] if metrics['revenue_cagr'] is not None else 0.10, -0.10, 0.60)
            margin_now = metrics['operating_margin'] if metrics['operating_margin'] is not None else 0.0
            sales_to_capital = metrics['sales_to_capital'] or 1.5
            tax_rate = self._effective_tax_rate(financial_data)

            rates = growth_path(growth, g_terminal, years)
            margins = growth_path(margin_now, target_margin, years)
            flows, rows, previous = [], [], revenue
            for year, (g, margin) in enumerate(zip(rates, margins), 1):
                current = previous * (1 + g)
                ebit = current * margin
                nopat = ebit * (1 - tax_rate) if ebit > 0 else ebit
                reinvestment = (current - previous) / sales_to_capital
                flows.append(nopat - reinvestment)
                rows.append({'revenue': current, 'operating_margin': margin, 'ebit': ebit, 'nopat': nopat,
                             'reinvestment': reinvestment})
                previous = current
            terminal_roic = rate + 0.02
            terminal_nopat = previous * (1 + g_terminal) * target_margin * (1 - tax_rate)
            terminal_cash_flow = terminal_nopat * (1 - g_terminal / terminal_roic)
            dcf = self._discount_cash_flows(flows, rate, g_terminal, growth_rates=rates,
                                            terminal_cash_flow=terminal_cash_flow)
            for projection, row in zip(dcf['projections'], rows):
                projection.update(row)
            bridge = self._equity_bridge(dcf['enterprise_value'], financial_data)
            return {'company': ticker, 'method': label, **dcf, **bridge, 'discount_rate': rate,
                    'growth_rate': growth, 'terminal_growth': g_terminal, 'forecast_years': years,
                    'current_operating_margin': margin_now, 'target_operating_margin': target_margin,
                    'sales_to_capital': sales_to_capital, 'terminal_roic': terminal_roic, 'tax_rate': tax_rate}
        except Exception as e:
            return self._method_error(label, e, ticker)

    def _rule_of_40_valuation(self, ticker: str, financial_data: Dict[str, Any], metrics: Dict[str, Any],
                              base_multiple: float) -> Dict[str, Any]:
        """
        EV/Revenue scaled by the Rule of 40 score (revenue growth + FCF margin):
        multiple = peer multiple x clamp(score / 40 %, 0.4, 2.0).
        """
        label = self.METHODS['rule_of_40']
        score = metrics['rule_of_40_score']
        if score is None:
            return self._method_error(label, "Revenue growth or FCF margin unavailable", ticker)
        factor = clamp(score / 0.40, 0.4, 2.0)
        result = self._multiples_valuation(ticker, financial_data, {'ev_revenue': base_multiple * factor},
                                           label=label)
        result.update({'rule_of_40_score': score, 'revenue_growth': metrics['revenue_growth'],
                       'fcf_margin': metrics['fcf_margin'], 'peer_multiple': base_multiple,
                       'adjustment_factor': factor, 'applied_multiple': base_multiple * factor})
        return result

    def _historical_revenue(self, financial_data: Dict[str, Any]):
        """Chronological revenue series (oldest first)."""
        income, _, _ = self._statements(financial_data)
        return get_series(income, 'revenue', ascending=True)
