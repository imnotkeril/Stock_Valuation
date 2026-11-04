"""
Valuation of financial companies (banks, insurers, asset managers, payment
networks, diversified financials).

Banks and insurers are balance-sheet businesses: debt and deposits are
operating liabilities, so enterprise-value DCFs are not meaningful. They are
valued on equity: excess return (residual income) model, justified P/B
``(ROE - g) / (COE - g)``, P/E multiples and a dividend discount model.
Capital-light fee businesses (asset managers, payment networks) may also use
a free cash flow DCF and EV/EBITDA.
"""

import logging
from typing import Any, Dict, List, Optional

from StockAnalysisSystem.src.industry.universe import lookup_company
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.financial_data import find_row, get_latest, safe_divide
from StockAnalysisSystem.src.valuation.base_valuation import clamp
from StockAnalysisSystem.src.valuation.dcf_models import SectorSpecificDCF

logger = logging.getLogger(__name__)

EQUITY_SUBSECTORS = ["bank", "insurance", "asset_manager", "diversified"]

#: Peer multiples by subsector (long-run medians).
PEER_PE = {"bank": 11.0, "insurance": 12.0, "asset_manager": 15.0, "payments": 28.0, "diversified": 13.0}
PEER_EV_EBITDA = {"asset_manager": 12.0, "payments": 22.0}

#: Income statement rows that identify insurers.
INSURANCE_ROWS = ["Total Premiums Earned", "Net Premiums Written", "Net Policyholder Benefits And Claims",
                  "Loss Adjustment Expense", "Policyholder Benefits Gross"]


class FinancialSectorValuation(SectorSpecificDCF):
    """Equity-based valuation of banks, insurers and other financial companies."""

    SECTOR = "Financials"

    SUBSECTORS: Dict[str, str] = {
        "bank": "Banks & Lenders",
        "insurance": "Insurance",
        "asset_manager": "Asset Management",
        "payments": "Payments, Exchanges & Fee-Based Services",
        "diversified": "Diversified Financials & Capital Markets",
    }

    SUBSECTOR_ALIASES = {"banking": "bank", "banks": "bank", "asset_management": "asset_manager",
                         "financial": "diversified", "insurer": "insurance"}

    METHODS: Dict[str, str] = {
        "excess_return": "Excess Return (Residual Income)",
        "justified_pb": "Justified P/B (ROE - g) / (COE - g)",
        "pe_multiple": "P/E Multiple",
        "dividend_discount": "Dividend Discount Model",
        "fcf_dcf": "Discounted Cash Flow (FCF)",
        "ev_ebitda": "EV/EBITDA Multiple",
    }

    SUBSECTOR_METHODS: Dict[str, List[str]] = {
        "bank": ["excess_return", "justified_pb", "pe_multiple", "dividend_discount"],
        "insurance": ["excess_return", "justified_pb", "pe_multiple", "dividend_discount"],
        "asset_manager": ["fcf_dcf", "pe_multiple", "ev_ebitda", "dividend_discount", "excess_return"],
        "payments": ["fcf_dcf", "pe_multiple", "ev_ebitda"],
        "diversified": ["excess_return", "justified_pb", "pe_multiple", "dividend_discount"],
    }

    DEFAULT_METHOD_WEIGHTS: Dict[str, float] = {
        "excess_return": 0.35, "justified_pb": 0.25, "pe_multiple": 0.20, "dividend_discount": 0.20,
        "fcf_dcf": 0.0, "ev_ebitda": 0.0,
    }

    SUBSECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
        "bank": DEFAULT_METHOD_WEIGHTS,
        "insurance": DEFAULT_METHOD_WEIGHTS,
        "diversified": DEFAULT_METHOD_WEIGHTS,
        "asset_manager": {"fcf_dcf": 0.35, "pe_multiple": 0.25, "ev_ebitda": 0.15, "dividend_discount": 0.10,
                          "excess_return": 0.15},
        "payments": {"fcf_dcf": 0.50, "pe_multiple": 0.25, "ev_ebitda": 0.25},
    }

    PARAMETERS: Dict[str, Dict[str, Any]] = {
        "cost_of_equity": {"label": "Cost of equity", "min": 0.06, "max": 0.16, "default": 0.10, "step": 0.0025,
                           "kind": "percent", "subsectors": None,
                           "help": "Required return on equity. Derived from CAPM when not set."},
        "sustainable_roe": {"label": "Sustainable ROE", "min": 0.04, "max": 0.30, "default": 0.12, "step": 0.0025,
                            "kind": "percent", "subsectors": EQUITY_SUBSECTORS,
                            "help": "Long-run return on equity the business converges to "
                                    "(defaults to the recent average ROE, capped at 25 %)."},
        "terminal_growth": {"label": "Terminal growth rate", "min": 0.0, "max": 0.05, "default": 0.02,
                            "step": 0.0025, "kind": "percent", "subsectors": None,
                            "help": "Perpetual growth; must stay below the cost of equity."},
        "payout_ratio": {"label": "Dividend payout ratio", "min": 0.0, "max": 1.0, "default": 0.40, "step": 0.05,
                         "kind": "percent", "subsectors": EQUITY_SUBSECTORS,
                         "help": "Share of earnings paid out; the rest compounds book value."},
        "forecast_years": {"label": "Forecast horizon (years)", "min": 3, "max": 15, "default": 10, "step": 1,
                           "kind": "int", "subsectors": None,
                           "help": "Years over which ROE fades to its sustainable level."},
        "discount_rate": {"label": "Discount rate (WACC)", "min": 0.05, "max": 0.16, "default": 0.09,
                          "step": 0.0025, "kind": "percent", "subsectors": ["asset_manager", "payments"],
                          "help": "Rate used to discount free cash flow (fee-based businesses only)."},
    }

    def __init__(self, data_loader: Optional[DataLoader] = None):
        super().__init__(data_loader)

    # ------------------------------------------------------------------
    # Subsector detection
    # ------------------------------------------------------------------
    def detect_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                         company_info: Optional[Dict[str, Any]] = None) -> str:
        """Classify from the industry name, then statement structure, then the curated universe."""
        info = company_info or self._company_info(financial_data)
        industry = str(info.get('industry') or '').lower()
        income, _, _ = self._statements(financial_data)
        revenue = get_latest(income, 'revenue')
        interest_share = safe_divide(get_latest(income, 'net_interest_income'), revenue, 0.0)

        if 'insurance broker' in industry:
            return 'payments'
        if 'insurance' in industry or 'reinsurance' in industry:
            return 'insurance'
        if any(term in industry for term in ('bank', 'mortgage', 'savings', 'thrift')):
            return 'bank'
        if 'credit services' in industry:
            return 'bank' if interest_share > 0.3 else 'payments'
        if any(term in industry for term in ('asset management', 'investment management', 'wealth')):
            return 'asset_manager'
        if any(term in industry for term in ('financial data', 'exchange', 'payment', 'transaction')):
            return 'payments'
        if any(term in industry for term in ('capital markets', 'conglomerate', 'diversified', 'brokerage')):
            return 'diversified'

        if interest_share > 0.3:
            return 'bank'
        if any(find_row(income, row) is not None for row in INSURANCE_ROWS):
            return 'insurance'
        record = lookup_company(str(ticker).upper()) or {}
        universe_map = {"Banks": "bank", "Insurance": "insurance", "Asset Management": "asset_manager",
                        "Diversified Financial Services": "diversified"}
        return universe_map.get(record.get('subsector'), 'diversified')

    # ------------------------------------------------------------------
    # Valuation entry point
    # ------------------------------------------------------------------
    def get_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                      subsector: Optional[str] = None, sector: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Value a financial company with the methods of its subsector.

        Keyword overrides: ``cost_of_equity``, ``sustainable_roe``,
        ``terminal_growth``, ``payout_ratio``, ``forecast_years``,
        ``discount_rate`` (fee-based subsectors), plus ``methods`` / ``weights``.
        """
        def prepare(data, key, overrides, notes):
            profile = self._financial_profile(data, overrides)
            runners = {
                'excess_return': lambda: self._excess_return_method(ticker, data, profile),
                'justified_pb': lambda: self._justified_pb_method(ticker, data, profile),
                'pe_multiple': lambda: self._multiples_valuation(ticker, data, {'pe': PEER_PE[key]},
                                                                 label='P/E Multiple'),
                'dividend_discount': lambda: self._dividend_method(ticker, data, profile, overrides),
                'fcf_dcf': lambda: self.discounted_cash_flow_valuation(
                    ticker, data, self.SECTOR, discount_rate=overrides.get('discount_rate'),
                    terminal_growth=profile['terminal_growth'], forecast_years=profile['forecast_years']),
                'ev_ebitda': lambda: self._multiples_valuation(ticker, data,
                                                               {'ev_ebitda': PEER_EV_EBITDA.get(key, 12.0)},
                                                               label='EV/EBITDA Multiple'),
            }
            if key in ('bank', 'insurance', 'diversified'):
                notes.append("Valued on equity (book value and returns on equity): deposits and debt are "
                             "operating liabilities, so an enterprise-value FCF DCF is not used.")
            if profile['return_on_equity'] is None:
                notes.append("Return on equity could not be derived; excess-return methods may be unavailable.")
            assumptions = {k: profile[k] for k in ('cost_of_equity', 'cost_of_equity_source', 'sustainable_roe',
                                                   'terminal_growth', 'payout_ratio', 'forecast_years')}
            assumptions['peer_pe'] = PEER_PE[key]
            assumptions['metrics'] = {k: profile[k] for k in ('return_on_equity', 'book_value', 'book_value_per_share',
                                                              'eps', 'price_to_book', 'price_to_earnings',
                                                              'net_interest_income_share')}
            return runners, assumptions

        return self._run_sector_valuation(ticker, financial_data, subsector, sector, kwargs, prepare)

    # Backward-compatible entry points ------------------------------------
    def value_financial_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Value with the detected subsector."""
        return self.get_valuation(ticker, financial_data)

    def value_bank(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='bank')

    def value_insurance_company(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='insurance')

    def value_asset_manager(self, ticker: str, financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.get_valuation(ticker, financial_data, subsector='asset_manager')

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _financial_profile(self, financial_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Resolved equity-valuation inputs (user overrides first, then data-derived values)."""
        income, balance, _ = self._statements(financial_data)
        shares = self._shares_outstanding(financial_data)
        price = self._current_price(financial_data)
        book_value = get_latest(balance, 'total_equity')
        net_income = get_latest(income, 'net_income')
        roe = self._return_on_equity(financial_data)

        payout = overrides.get('payout_ratio')
        if payout is None:
            dps = self._dividends_per_share(financial_data)
            payout = safe_divide(dps * shares, net_income) if dps and shares and net_income and net_income > 0 \
                else None
            payout = clamp(payout, 0.0, 1.0) if payout is not None else 0.4

        cost_of_equity = overrides.get('cost_of_equity')
        source = 'user' if cost_of_equity is not None else 'capm'
        if cost_of_equity is None:
            cost_of_equity = self._cost_of_equity(financial_data)

        sustainable = overrides.get('sustainable_roe')
        if sustainable is None:
            sustainable = clamp(roe, 0.0, 0.25) if roe is not None and roe > 0 else cost_of_equity

        params = self._get_dcf_parameters(self.SECTOR)
        terminal_growth = overrides.get('terminal_growth')
        bvps = book_value / shares if book_value and shares else None
        eps = net_income / shares if net_income is not None and shares else None
        return {
            'shares': shares, 'book_value': book_value, 'book_value_per_share': bvps, 'net_income': net_income,
            'eps': eps, 'return_on_equity': roe, 'payout_ratio': payout,
            'cost_of_equity': cost_of_equity, 'cost_of_equity_source': source, 'sustainable_roe': sustainable,
            'terminal_growth': terminal_growth if terminal_growth is not None else params['terminal_growth_rate'],
            'forecast_years': int(overrides.get('forecast_years') or 10),
            'price_to_book': price / bvps if price and bvps and bvps > 0 else None,
            'price_to_earnings': price / eps if price and eps and eps > 0 else None,
            'net_interest_income_share': safe_divide(get_latest(income, 'net_interest_income'),
                                                     get_latest(income, 'revenue')),
        }

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def _excess_return_method(self, ticker: str, financial_data: Dict[str, Any],
                              profile: Dict[str, Any]) -> Dict[str, Any]:
        """Excess return model with ROE fading to the sustainable ROE."""
        label = self.METHODS['excess_return']
        if profile['return_on_equity'] is None:
            return self._method_error(label, "Return on equity unavailable", ticker)
        try:
            model = self._excess_return_model(
                profile['book_value'], clamp(profile['return_on_equity'], -0.5, 0.5), profile['sustainable_roe'],
                profile['cost_of_equity'], profile['payout_ratio'], profile['terminal_growth'],
                profile['forecast_years'])
        except ValueError as e:
            return self._method_error(label, e, ticker)
        value = self._per_share(model['equity_value'], financial_data)
        result = {'company': ticker, 'method': label, **model, 'value_per_share': value,
                  'book_value_per_share': profile['book_value_per_share'],
                  'return_on_equity': profile['return_on_equity'], 'sustainable_roe': profile['sustainable_roe'],
                  'cost_of_equity': profile['cost_of_equity'], 'terminal_growth': profile['terminal_growth'],
                  'payout_ratio': profile['payout_ratio'], 'forecast_years': profile['forecast_years']}
        if value is None:
            result['error'] = 'Equity value is not positive (returns persistently below the cost of equity)'
        return result

    def _justified_pb_method(self, ticker: str, financial_data: Dict[str, Any],
                             profile: Dict[str, Any]) -> Dict[str, Any]:
        """Justified P/B = (ROE - g) / (COE - g) applied to book value per share."""
        label = self.METHODS['justified_pb']
        roe, ke, g = profile['sustainable_roe'], profile['cost_of_equity'], profile['terminal_growth']
        bvps = profile['book_value_per_share']
        if not bvps or bvps <= 0:
            return self._method_error(label, "Book value per share is not positive", ticker)
        if ke <= g:
            return self._method_error(label, f"Cost of equity ({ke:.2%}) must exceed growth ({g:.2%})", ticker)
        justified_pb = (roe - g) / (ke - g)
        result = {'company': ticker, 'method': label, 'justified_pb': justified_pb, 'book_value_per_share': bvps,
                  'sustainable_roe': roe, 'cost_of_equity': ke, 'terminal_growth': g,
                  'current_pb': profile['price_to_book'],
                  'value_per_share': justified_pb * bvps if justified_pb > 0 else None}
        if justified_pb <= 0:
            result['error'] = 'Sustainable ROE is below the growth rate; justified P/B is not positive'
        return result

    def _dividend_method(self, ticker: str, financial_data: Dict[str, Any], profile: Dict[str, Any],
                         overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Two-stage DDM; near-term growth = ROE x retention, stable payout = 1 - g / sustainable ROE."""
        roe = profile['return_on_equity']
        growth = clamp(roe * (1 - profile['payout_ratio']), 0.0, 0.10) if roe is not None else None
        return self.dividend_discount_valuation(
            ticker, financial_data, self.SECTOR, terminal_growth=profile['terminal_growth'],
            forecast_years=profile['forecast_years'], growth_rate=growth,
            cost_of_equity=profile['cost_of_equity'], payout_ratio=overrides.get('payout_ratio'),
            stable_roe=max(profile['sustainable_roe'], profile['terminal_growth'] + 0.01))
