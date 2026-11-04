"""
Advanced discounted cash flow models.

``AdvancedDCFValuation`` adds multi-stage, driver-based (forecast-driven),
Monte Carlo and scenario DCFs to the generic models of ``BaseValuation``.
``SectorSpecificDCF`` adds sector flavoured variants (SBC-adjusted technology
DCF, excess-return model for financials, cycle-normalised energy DCF, ...)
reused by the sector models.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from StockAnalysisSystem.src.config import normalize_sector
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.financial_data import get_latest, get_series, safe_divide
from StockAnalysisSystem.src.valuation.base_valuation import (
    BaseValuation,
    clamp,
    clean_result,
    compound,
    growth_path,
)

logger = logging.getLogger(__name__)

#: Scenario definitions: probability and shifts applied to the base case.
SCENARIOS: Dict[str, Dict[str, float]] = {
    'bull': {'probability': 0.25, 'growth_shift': 0.5, 'discount_shift': -0.01, 'terminal_shift': 0.005},
    'base': {'probability': 0.50, 'growth_shift': 0.0, 'discount_shift': 0.0, 'terminal_shift': 0.0},
    'bear': {'probability': 0.25, 'growth_shift': -0.5, 'discount_shift': 0.01, 'terminal_shift': -0.005},
}


def _split_horizon(forecast_years: Optional[int], high_growth_years: int, transition_years: int):
    """Split a total horizon into (high growth, transition) years."""
    if not forecast_years:
        return max(int(high_growth_years), 1), max(int(transition_years), 0)
    total = max(int(forecast_years), 1)
    high = max(total // 2, 1)
    return high, total - high


class AdvancedDCFValuation(BaseValuation):
    """Multi-stage, driver-based, Monte Carlo and scenario DCF valuations."""

    METHODS: Dict[str, str] = {
        "dcf": "Discounted Cash Flow",
        "multi_stage_dcf": "Multi-Stage DCF",
        "forecast_driven_dcf": "Forecast-Driven DCF",
        "monte_carlo_dcf": "Monte Carlo DCF",
        "scenario_dcf": "Scenario DCF (Bull / Base / Bear)",
        "relative": "Relative Valuation (Multiples)",
        "asset_based": "Asset-Based Valuation",
        "dividend_discount": "Dividend Discount Model",
    }

    DEFAULT_METHOD_WEIGHTS: Dict[str, float] = {
        "dcf": 0.10,
        "multi_stage_dcf": 0.20,
        "forecast_driven_dcf": 0.20,
        "monte_carlo_dcf": 0.10,
        "scenario_dcf": 0.15,
        "relative": 0.20,
        "asset_based": 0.0,
        "dividend_discount": 0.05,
    }

    PARAMETERS: Dict[str, Dict[str, Any]] = {
        **BaseValuation.PARAMETERS,
        "simulations": {"label": "Monte Carlo simulations", "min": 200, "max": 10000, "default": 1000,
                        "step": 100, "kind": "int", "subsectors": None,
                        "help": "Number of Monte Carlo draws of growth, discount rate and terminal growth."},
    }

    def __init__(self, data_loader: Optional[DataLoader] = None):
        super().__init__(data_loader)

    def _method_runners(self, ticker: str, financial_data: Dict[str, Any], sector: Optional[str],
                        rate_kwargs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Callable[[], Dict]]:
        runners = super()._method_runners(ticker, financial_data, sector, rate_kwargs, kwargs)
        simulations = int(kwargs.get('simulations') or 1000)
        runners.update({
            'multi_stage_dcf': lambda: self.multi_stage_dcf_valuation(ticker, financial_data, sector, **rate_kwargs),
            'forecast_driven_dcf': lambda: self.forecast_driven_dcf(ticker, financial_data, sector, **rate_kwargs),
            'monte_carlo_dcf': lambda: self.monte_carlo_dcf(ticker, financial_data, sector, simulations=simulations,
                                                            seed=kwargs.get('seed'), **rate_kwargs),
            'scenario_dcf': lambda: self.scenario_analysis_dcf(ticker, financial_data, sector, **rate_kwargs),
        })
        return runners

    # ------------------------------------------------------------------
    # Multi-stage DCF
    # ------------------------------------------------------------------
    def multi_stage_dcf_valuation(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  sector: str = None, discount_rate: Optional[float] = None,
                                  terminal_growth: Optional[float] = None,
                                  forecast_years: Optional[int] = None, growth_rate: Optional[float] = None,
                                  high_growth_years: int = 5, transition_years: int = 5,
                                  cash_flows: Optional[pd.Series] = None, label: str = 'Multi-Stage DCF',
                                  **kwargs) -> Dict[str, Any]:
        """
        Three-stage DCF: constant high growth, a linear fade to terminal growth,
        then a Gordon growth terminal value.

        ``forecast_years`` (when given) is split evenly between the high-growth
        and transition stages. ``cash_flows`` replaces the unlevered FCF history.
        """
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            high, transition = _split_horizon(forecast_years, high_growth_years, transition_years)
            inputs = self._dcf_inputs(financial_data, sector, discount_rate, terminal_growth, high + transition,
                                      growth_rate, cash_flows=cash_flows)
            g, g_terminal = inputs['growth_rate'], inputs['terminal_growth']
            rates = [g] * high + [g + (g_terminal - g) * t / transition for t in range(1, transition + 1)]
            return self._dcf_result(label, ticker, inputs, rates, financial_data,
                                    initial_growth_rate=g, high_growth_years=high, transition_years=transition)
        except Exception as e:
            return self._method_error(label, e, ticker)

    # ------------------------------------------------------------------
    # Forecast-driven DCF
    # ------------------------------------------------------------------
    def _forecast_drivers(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Historical operating drivers (three-year averages, as fractions of
        revenue): EBIT margin, D&A, capex and operating working capital.
        """
        income, balance, cash_flow = self._statements(financial_data)
        revenue = get_series(income, 'revenue').head(3)
        if revenue.empty or (revenue <= 0).all():
            raise ValueError("Revenue history unavailable")

        def ratio(df, item, absolute=False):
            series = get_series(df, item).reindex(revenue.index)
            series = series.abs() if absolute else series
            values = (series / revenue).dropna()
            return float(values.mean()) if not values.empty else None

        ebit_margin = ratio(income, 'operating_income')
        if ebit_margin is None:
            raise ValueError("Operating income history unavailable")
        depreciation = ratio(cash_flow, 'cash_flow_depreciation', True) or ratio(income, 'depreciation', True) or 0.0
        capex = ratio(cash_flow, 'capital_expenditure', True)
        if capex is None:
            capex = depreciation

        current_assets = get_latest(balance, 'current_assets')
        current_liabilities = get_latest(balance, 'current_liabilities')
        nwc = 0.0
        if current_assets is not None and current_liabilities is not None:
            operating_assets = current_assets - (get_latest(balance, 'cash_and_short_term_investments') or 0.0)
            operating_liabilities = current_liabilities - (get_latest(balance, 'short_term_debt') or 0.0)
            nwc = clamp(safe_divide(operating_assets - operating_liabilities, revenue.iloc[0], 0.0), -0.3, 0.5)

        return {
            'revenue': float(revenue.iloc[0]),
            'ebit_margin': ebit_margin,
            'depreciation_pct_revenue': depreciation,
            'capex_pct_revenue': capex,
            'nwc_pct_revenue': nwc,
            'tax_rate': self._effective_tax_rate(financial_data),
        }

    def _forecast_financial_statements(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame,
                                       cash_flow: pd.DataFrame, forecast_years: int, sector: str = None,
                                       revenue_growth: Optional[float] = None,
                                       terminal_growth: Optional[float] = None) -> Dict[str, Any]:
        """
        Driver-based projection of revenue, EBIT, NOPAT, reinvestment and
        free cash flow to the firm::

            FCFF = EBIT (1 - t) + D&A - capex - change in NWC

        Revenue growth fades from the historical rate to terminal growth.
        Returns ``fcf`` (list), ``projections`` (per-year dicts), ``drivers``
        and a ``summary`` of key lines.
        """
        financial_data = {'income_statement': income_stmt, 'balance_sheet': balance_sheet, 'cash_flow': cash_flow}
        drivers = self._forecast_drivers(financial_data)
        params = self._get_dcf_parameters(sector)
        g_terminal = terminal_growth if terminal_growth is not None else params['terminal_growth_rate']
        if revenue_growth is None:
            revenue_growth = self._estimate_company_growth(financial_data)
        rates = growth_path(revenue_growth, g_terminal, forecast_years)

        projections, previous_revenue = [], drivers['revenue']
        for year, growth in enumerate(rates, 1):
            revenue = previous_revenue * (1 + growth)
            ebit = revenue * drivers['ebit_margin']
            nopat = ebit * (1 - drivers['tax_rate']) if ebit > 0 else ebit
            depreciation = revenue * drivers['depreciation_pct_revenue']
            capex = revenue * drivers['capex_pct_revenue']
            delta_nwc = (revenue - previous_revenue) * drivers['nwc_pct_revenue']
            fcf = nopat + depreciation - capex - delta_nwc
            projections.append({'year': year, 'growth': growth, 'revenue': revenue, 'ebit': ebit, 'nopat': nopat,
                                'depreciation': depreciation, 'capex': capex, 'change_in_nwc': delta_nwc,
                                'fcf': fcf})
            previous_revenue = revenue

        summary = {key: [p[key] for p in projections] for key in ('revenue', 'ebit', 'nopat', 'capex', 'fcf')}
        return {'fcf': [p['fcf'] for p in projections], 'projections': projections, 'drivers': drivers,
                'revenue_growth': revenue_growth, 'summary': summary}

    def forecast_driven_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                            sector: str = None, discount_rate: Optional[float] = None,
                            terminal_growth: Optional[float] = None, forecast_years: Optional[int] = None,
                            growth_rate: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """DCF on driver-based FCFF projections (see ``_forecast_financial_statements``)."""
        label = 'Forecast-Driven DCF'
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            params = self._get_dcf_parameters(sector)
            years = int(forecast_years or params['forecast_years'])
            g_terminal = terminal_growth if terminal_growth is not None else params['terminal_growth_rate']
            income, balance, cash_flow = self._statements(financial_data)
            forecast = self._forecast_financial_statements(income, balance, cash_flow, years, sector,
                                                           revenue_growth=growth_rate, terminal_growth=g_terminal)
            if forecast['fcf'][-1] <= 0:
                raise ValueError("Projected free cash flow is negative at the end of the forecast")
            wacc = self._wacc_components(financial_data, sector)
            rate = discount_rate if discount_rate is not None else wacc['wacc']
            dcf = self._discount_cash_flows(forecast['fcf'], rate, g_terminal,
                                            growth_rates=[p['growth'] for p in forecast['projections']])
            for projection, discounted in zip(forecast['projections'], dcf['projections']):
                discounted.update({k: projection[k] for k in ('revenue', 'ebit', 'nopat', 'depreciation',
                                                              'capex', 'change_in_nwc')})
            bridge = self._equity_bridge(dcf['enterprise_value'], financial_data)
            return {
                'company': ticker, 'method': label, **dcf, **bridge,
                'discount_rate': rate, 'terminal_growth': g_terminal, 'forecast_years': years,
                'growth_rate': forecast['revenue_growth'], 'drivers': forecast['drivers'],
                'forecast_financials': forecast['summary'], 'wacc_components': wacc,
            }
        except Exception as e:
            return self._method_error(label, e, ticker)

    # ------------------------------------------------------------------
    # Monte Carlo DCF
    # ------------------------------------------------------------------
    def monte_carlo_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                        sector: str = None, simulations: int = 1000, seed: Optional[int] = None,
                        discount_rate: Optional[float] = None, terminal_growth: Optional[float] = None,
                        forecast_years: Optional[int] = None, growth_rate: Optional[float] = None,
                        growth_volatility: Optional[float] = None, discount_volatility: float = 0.01,
                        terminal_volatility: float = 0.005, **kwargs) -> Dict[str, Any]:
        """
        Monte Carlo DCF: draws near-term growth, discount rate and terminal
        growth from normal distributions around the base case and reports the
        distribution of value per share (``value_per_share`` = median).

        Draws are clipped so the discount rate always exceeds terminal growth
        by at least 1 %. Pass ``seed`` for reproducible results.
        """
        label = 'Monte Carlo DCF'
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            inputs = self._dcf_inputs(financial_data, sector, discount_rate, terminal_growth,
                                      forecast_years, growth_rate)
            n_sims = int(clamp(int(simulations or 1000), 100, 100_000))
            years = inputs['forecast_years']
            g, r, g_t = inputs['growth_rate'], inputs['discount_rate'], inputs['terminal_growth']
            if r <= g_t:
                raise ValueError(f"Discount rate ({r:.2%}) must exceed terminal growth ({g_t:.2%})")
            sigma_g = growth_volatility if growth_volatility is not None else max(0.02, abs(g) * 0.3)

            rng = np.random.default_rng(seed)
            growth = np.clip(rng.normal(g, sigma_g, n_sims), -0.10, 0.40)
            rates = np.clip(rng.normal(r, discount_volatility, n_sims), 0.03, 0.30)
            terminal = np.minimum(rng.normal(g_t, terminal_volatility, n_sims), rates - 0.01)

            steps = np.arange(years) / max(years - 1, 1)
            path = growth[:, None] + (terminal - growth)[:, None] * steps[None, :]
            flows = inputs['base_fcf'] * np.cumprod(1 + path, axis=1)
            discount = (1 + rates)[:, None] ** -np.arange(1, years + 1)[None, :]
            terminal_value = flows[:, -1] * (1 + terminal) / (rates - terminal)
            enterprise = (flows * discount).sum(axis=1) + terminal_value * discount[:, -1]

            bridge = self._equity_bridge(0.0, financial_data)
            claims = -bridge['equity_value']  # debt - cash + minorities + preferred
            shares = bridge['shares_outstanding']
            if not shares:
                raise ValueError("Shares outstanding unavailable")
            per_share = np.maximum(enterprise - claims, 0.0) / shares

            percentiles = {f'p{p}': float(np.percentile(per_share, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
            counts, edges = np.histogram(per_share, bins=30)
            price = self._current_price(financial_data)
            median = percentiles['p50']
            result = {
                'company': ticker, 'method': label,
                'value_per_share': median if median > 0 else None,
                'mean_value_per_share': float(per_share.mean()),
                'median_value_per_share': median,
                'std_deviation': float(per_share.std()),
                'percentiles': percentiles,
                'percentile_5': percentiles['p5'], 'percentile_25': percentiles['p25'],
                'percentile_75': percentiles['p75'], 'percentile_95': percentiles['p95'],
                'probability_above_price': float((per_share > price).mean()) if price else None,
                'probability_zero_equity': float((per_share <= 0).mean()),
                'distribution': {'bin_edges': edges.tolist(), 'counts': counts.tolist()},
                'mean_enterprise_value': float(enterprise.mean()),
                'median_enterprise_value': float(np.median(enterprise)),
                'enterprise_value': float(np.median(enterprise)),
                'equity_value': median * shares,
                'net_debt': bridge['net_debt'],
                'base_parameters': {'discount_rate': r, 'growth_rate': g, 'terminal_growth': g_t,
                                    'growth_volatility': sigma_g, 'discount_volatility': discount_volatility,
                                    'terminal_volatility': terminal_volatility},
                'discount_rate': r, 'growth_rate': g, 'terminal_growth': g_t, 'forecast_years': years,
                'base_fcf': inputs['base_fcf'], 'simulations': n_sims, 'seed': seed,
            }
            if median <= 0:
                result['error'] = 'Median simulated equity value is not positive'
            return result
        except Exception as e:
            return self._method_error(label, e, ticker)

    # ------------------------------------------------------------------
    # Scenario DCF
    # ------------------------------------------------------------------
    def scenario_analysis_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                              sector: str = None, discount_rate: Optional[float] = None,
                              terminal_growth: Optional[float] = None, forecast_years: Optional[int] = None,
                              growth_rate: Optional[float] = None, cash_flows: Optional[pd.Series] = None,
                              average_years: int = 1, scenarios: Optional[Dict[str, Dict[str, float]]] = None,
                              label: str = 'Scenario DCF', **kwargs) -> Dict[str, Any]:
        """
        Bull / base / bear DCFs and their probability-weighted value.

        Scenarios shift near-term growth (relative, at least 2 points), the
        discount rate and terminal growth (absolute). A scenario whose equity
        value is negative counts as zero (limited liability).
        """
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            inputs = self._dcf_inputs(financial_data, sector, discount_rate, terminal_growth, forecast_years,
                                      growth_rate, cash_flows=cash_flows, average_years=average_years)
            results, weighted, total_probability = {}, 0.0, 0.0
            for name, shift in (scenarios or SCENARIOS).items():
                g = inputs['growth_rate']
                g = g + math.copysign(max(abs(g) * abs(shift['growth_shift']), 0.02), shift['growth_shift']) \
                    if shift['growth_shift'] else g
                scenario_inputs = dict(inputs, growth_rate=clamp(g, -0.10, 0.40),
                                       discount_rate=inputs['discount_rate'] + shift['discount_shift'],
                                       terminal_growth=inputs['terminal_growth'] + shift['terminal_shift'])
                rates = growth_path(scenario_inputs['growth_rate'], scenario_inputs['terminal_growth'],
                                    scenario_inputs['forecast_years'])
                try:
                    outcome = self._dcf_result(name, ticker, scenario_inputs, rates, financial_data)
                except ValueError as e:
                    outcome = {'value_per_share': None, 'error': str(e)}
                outcome = {'probability': shift['probability'],
                           'parameters': {k: scenario_inputs[k] for k in ('growth_rate', 'discount_rate',
                                                                          'terminal_growth')},
                           **{k: outcome.get(k) for k in ('value_per_share', 'enterprise_value', 'equity_value',
                                                           'forecast_fcf', 'terminal_value', 'present_value_fcf',
                                                           'present_value_terminal', 'error')}}
                results[name] = outcome
                if 'error' in outcome and outcome['error'] and outcome.get('equity_value') is None:
                    continue
                weighted += shift['probability'] * (outcome['value_per_share'] or 0.0)
                total_probability += shift['probability']
            if total_probability <= 0:
                raise ValueError(results.get('base', {}).get('error') or "No scenario could be valued")
            value = weighted / total_probability
            base = results.get('base', {})
            safety = inputs['safety_margin']
            return {
                'company': ticker, 'method': label, 'scenarios': results,
                'value_per_share': value if value > 0 else None,
                'base_value_per_share': base.get('value_per_share'),
                'enterprise_value': base.get('enterprise_value'), 'equity_value': base.get('equity_value'),
                'forecast_fcf': base.get('forecast_fcf'), 'terminal_value': base.get('terminal_value'),
                'discount_rate': inputs['discount_rate'], 'growth_rate': inputs['growth_rate'],
                'terminal_growth': inputs['terminal_growth'], 'forecast_years': inputs['forecast_years'],
                'base_fcf': inputs['base_fcf'], 'historical_fcf': clean_result(inputs['historical_fcf']),
                'conservative_value': base['value_per_share'] * (1 - safety) if base.get('value_per_share') else None,
                'safety_margin': safety,
            }
        except Exception as e:
            return self._method_error(label, e, ticker)


class SectorSpecificDCF(AdvancedDCFValuation):
    """DCF variants with sector-specific cash flow definitions and conventions."""

    SECTOR_DCF_MODELS: Dict[str, str] = {
        "Technology": "tech_sector_dcf",
        "Healthcare": "multi_stage_dcf",
        "Financials": "financial_sector_dcf",
        "Consumer Discretionary": "forecast_driven_dcf",
        "Consumer Staples": "forecast_driven_dcf",
        "Energy": "energy_sector_dcf",
        "Industrials": "forecast_driven_dcf",
        "Materials": "scenario_analysis_dcf",
        "Real Estate": "real_estate_dcf",
        "Communication Services": "multi_stage_dcf",
        "Utilities": "dividend_discount",
    }

    def __init__(self, data_loader: Optional[DataLoader] = None):
        super().__init__(data_loader)

    def get_sector_dcf_model(self, sector: str) -> str:
        """Name of the DCF variant used for ``sector`` (any spelling)."""
        return self.SECTOR_DCF_MODELS.get(normalize_sector(sector), "multi_stage_dcf")

    def apply_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                         sector: str = None, **kwargs) -> Dict[str, Any]:
        """Run the DCF variant appropriate for the company's sector."""
        if financial_data is None:
            financial_data = self._load_financial_data(ticker)
        sector = sector or self._company_info(financial_data).get('sector')
        model = self.get_sector_dcf_model(sector)
        if model in ("tech_sector_dcf", "financial_sector_dcf", "energy_sector_dcf", "real_estate_dcf"):
            return getattr(self, model)(ticker, financial_data, **kwargs)
        if model == "dividend_discount":
            result = self.dividend_discount_valuation(ticker, financial_data, sector, **kwargs)
            if result.get('value_per_share') is None:  # non-payers: fall back to cash flows
                return self.multi_stage_dcf_valuation(ticker, financial_data, sector, **kwargs)
            return result
        method = {"forecast_driven_dcf": self.forecast_driven_dcf,
                  "scenario_analysis_dcf": self.scenario_analysis_dcf}.get(model, self.multi_stage_dcf_valuation)
        return method(ticker, financial_data, sector, **kwargs)

    # ------------------------------------------------------------------
    # Financials: excess return (residual income) model
    # ------------------------------------------------------------------
    def _excess_return_model(self, book_value: float, current_roe: float, sustainable_roe: float,
                             cost_of_equity: float, payout_ratio: float, terminal_growth: float,
                             years: int) -> Dict[str, Any]:
        """
        Excess return model: equity = book value + PV of (ROE - Ke) x opening
        book value. ROE fades linearly from ``current_roe`` to
        ``sustainable_roe`` and the payout from ``payout_ratio`` to the stable
        payout ``1 - g / ROE`` so that book value growth converges to
        ``terminal_growth``; the terminal excess return grows at that rate
        (Ke > g required).
        """
        if book_value is None or book_value <= 0:
            raise ValueError("Book value of equity is not positive")
        if cost_of_equity <= terminal_growth:
            raise ValueError(f"Cost of equity ({cost_of_equity:.2%}) must exceed terminal growth "
                             f"({terminal_growth:.2%})")
        payout_ratio = clamp(payout_ratio, 0.0, 1.0)
        stable_payout = clamp(1 - terminal_growth / sustainable_roe, 0.0, 1.0) if sustainable_roe > terminal_growth \
            else 1.0
        years = max(int(years), 1)
        roes = growth_path(current_roe, sustainable_roe, years)
        payouts = growth_path(payout_ratio, stable_payout, years)
        projections, opening = [], float(book_value)
        for year, (roe, payout) in enumerate(zip(roes, payouts), 1):
            net_income = roe * opening
            excess = (roe - cost_of_equity) * opening
            factor = (1 + cost_of_equity) ** -year
            closing = opening + net_income * (1 - payout)
            projections.append({'year': year, 'roe': roe, 'payout_ratio': payout, 'opening_book_value': opening,
                                'net_income': net_income, 'excess_return': excess, 'discount_factor': factor,
                                'present_value': excess * factor, 'closing_book_value': closing})
            opening = closing
        terminal_excess = (sustainable_roe - cost_of_equity) * opening
        terminal_value = terminal_excess / (cost_of_equity - terminal_growth)
        pv_excess = sum(p['present_value'] for p in projections)
        pv_terminal = terminal_value * (1 + cost_of_equity) ** -years
        return {'book_value': float(book_value), 'equity_value': book_value + pv_excess + pv_terminal,
                'present_value_excess_returns': pv_excess, 'terminal_value': terminal_value,
                'present_value_terminal': pv_terminal, 'stable_payout_ratio': stable_payout,
                'projections': projections}

    def financial_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                             cost_of_equity: Optional[float] = None, sustainable_roe: Optional[float] = None,
                             terminal_growth: Optional[float] = None, payout_ratio: Optional[float] = None,
                             forecast_years: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Equity valuation for financials (an EV/FCF DCF is not meaningful for
        balance-sheet businesses): excess return model on book value.
        """
        label = 'Excess Return Model'
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            income, balance, _ = self._statements(financial_data)
            book_value = get_latest(balance, 'total_equity')
            roe = self._return_on_equity(financial_data)
            if roe is None:
                raise ValueError("Return on equity unavailable")
            params = self._get_dcf_parameters("Financials")
            ke = cost_of_equity if cost_of_equity is not None else self._cost_of_equity(financial_data)
            g = terminal_growth if terminal_growth is not None else params['terminal_growth_rate']
            if payout_ratio is None:
                net_income = get_latest(income, 'net_income')
                dps = self._dividends_per_share(financial_data)
                shares = self._shares_outstanding(financial_data)
                payout_ratio = clamp(safe_divide(dps * shares, net_income, 0.4), 0.0, 1.0) \
                    if dps and shares and net_income and net_income > 0 else 0.4
            target_roe = sustainable_roe if sustainable_roe is not None else clamp(roe, 0.0, 0.25)
            model = self._excess_return_model(book_value, clamp(roe, -0.5, 0.5), target_roe, ke, payout_ratio, g,
                                              forecast_years or 10)
            return {'company': ticker, 'method': label, **model,
                    'value_per_share': self._per_share(model['equity_value'], financial_data),
                    'return_on_equity': roe, 'sustainable_roe': target_roe, 'cost_of_equity': ke,
                    'terminal_growth': g, 'payout_ratio': payout_ratio,
                    'forecast_years': len(model['projections'])}
        except Exception as e:
            return self._method_error(label, e, ticker)

    # ------------------------------------------------------------------
    # Technology: SBC-adjusted multi-stage DCF
    # ------------------------------------------------------------------
    def _sbc_adjusted_fcf(self, financial_data: Dict[str, Any], sbc_haircut: float = 1.0) -> pd.Series:
        """Unlevered FCF minus ``sbc_haircut`` x stock-based compensation (a real, non-cash cost)."""
        _, _, cash_flow = self._statements(financial_data)
        fcf = self._unlevered_fcf_series(financial_data)
        sbc = get_series(cash_flow, 'stock_based_compensation').abs().reindex(fcf.index).fillna(0.0)
        return (fcf - clamp(sbc_haircut, 0.0, 1.0) * sbc).rename('SBC-adjusted FCF')

    def tech_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None,
                        sbc_haircut: float = 1.0, discount_rate: Optional[float] = None,
                        terminal_growth: Optional[float] = None, forecast_years: Optional[int] = None,
                        growth_rate: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Multi-stage DCF on SBC-adjusted free cash flow to the firm. Stock-based
        compensation is deducted because it dilutes shareholders as surely as
        a cash wage; the default horizon is 10 years (5 high growth + 5 fade).
        """
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            series = self._sbc_adjusted_fcf(financial_data, sbc_haircut)
            result = self.multi_stage_dcf_valuation(ticker, financial_data, "Technology", discount_rate,
                                                    terminal_growth, forecast_years or 10, growth_rate,
                                                    cash_flows=series, label='SBC-Adjusted DCF')
            result['sbc_haircut'] = sbc_haircut
            return result
        except Exception as e:
            return self._method_error('SBC-Adjusted DCF', e, ticker)

    # ------------------------------------------------------------------
    # Energy and real estate
    # ------------------------------------------------------------------
    def energy_sector_dcf(self, ticker: str, financial_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Scenario DCF on cycle-normalised (three-year average) free cash flow."""
        kwargs.setdefault('average_years', 3)
        return self.scenario_analysis_dcf(ticker, financial_data, kwargs.pop('sector', "Energy"),
                                          label='Cycle-Normalised Scenario DCF', **kwargs)

    def real_estate_dcf(self, ticker: str, financial_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Blend of adjusted book value (60 %) and a multi-stage DCF (40 %)."""
        if financial_data is None:
            financial_data = self._load_financial_data(ticker)
        asset_result = self.asset_based_valuation(ticker, financial_data, "Real Estate")
        dcf_result = self.multi_stage_dcf_valuation(ticker, financial_data, "Real Estate", **kwargs)
        values = [asset_result.get('value_per_share'), dcf_result.get('value_per_share')]
        result = dict(asset_result, method='Real Estate NAV / DCF Blend', dcf_value_per_share=values[1],
                      dcf_parameters={k: dcf_result.get(k) for k in ('discount_rate', 'growth_rate',
                                                                       'terminal_growth')})
        if all(values):
            result['value_per_share'] = 0.6 * values[0] + 0.4 * values[1]
        elif values[1]:
            result['value_per_share'] = values[1]
        result['blended_value_per_share'] = result.get('value_per_share')
        return result

    @staticmethod
    def _sum_present_values(cash_flows: Sequence[float], rate: float, start_year: int = 1) -> List[float]:
        """Present values of year-end ``cash_flows`` starting at ``start_year``."""
        return [cf * (1 + rate) ** -(start_year + i) for i, cf in enumerate(cash_flows)]

    @staticmethod
    def _black_scholes_call(spot: float, strike: float, years: float, rate: float, volatility: float) -> float:
        """Black-Scholes value of a European call (used for real options)."""
        if spot <= 0:
            return 0.0
        if strike <= 0 or years <= 0 or volatility <= 0:
            return max(spot - strike * math.exp(-rate * max(years, 0.0)), 0.0)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * years) / (volatility * math.sqrt(years))
        d2 = d1 - volatility * math.sqrt(years)

        def cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        return spot * cdf(d1) - strike * math.exp(-rate * years) * cdf(d2)
