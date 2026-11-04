"""
Base valuation model and the unified valuation contract.

``BaseValuation`` defines the contract shared by every sector model (class
attributes describing subsectors, methods and user-adjustable parameters, and
the standardised result produced by :meth:`BaseValuation.build_valuation_result`)
plus generic, textbook valuation building blocks:

* CAPM cost of equity and WACC,
* free cash flow history (reported and unlevered),
* a two-stage DCF with growth fading to the terminal rate and a Gordon
  growth terminal value (only when the discount rate exceeds growth),
* the enterprise-to-equity bridge (EV - debt + cash - minorities - preferred),
* peer-multiple, asset-based and dividend discount valuations.

All statement lookups go through :mod:`StockAnalysisSystem.src.utils.financial_data`.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from StockAnalysisSystem.src.config import DCF_PARAMETERS, RISK_FREE_RATE, SECTOR_DCF_PARAMETERS, normalize_sector
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.financial_data import (
    build_financial_data,
    cagr,
    get_latest,
    get_series,
    is_empty,
    safe_divide,
    to_float,
)

logger = logging.getLogger(__name__)

#: Long-run equity risk premium used in CAPM.
EQUITY_RISK_PREMIUM = 0.055
#: Corporate tax rate used when the effective rate cannot be derived.
DEFAULT_TAX_RATE = 0.21
#: Bounds applied to estimated (not user-supplied) rates.
WACC_BOUNDS = (0.05, 0.20)
COST_OF_EQUITY_BOUNDS = (0.045, 0.25)
GROWTH_BOUNDS = (-0.05, 0.25)
#: A method value outside this band around the share price is treated as implausible.
PLAUSIBLE_PRICE_MULTIPLE = (0.01, 20.0)

#: Peer multiples by canonical sector (long-run medians, deliberately conservative).
SECTOR_MULTIPLES: Dict[str, Dict[str, float]] = {
    "Technology": {"pe": 25.0, "ev_ebitda": 18.0, "ev_revenue": 5.0},
    "Healthcare": {"pe": 20.0, "ev_ebitda": 14.0, "ev_revenue": 3.5},
    "Financials": {"pe": 12.0, "pb": 1.3},
    "Consumer Discretionary": {"pe": 20.0, "ev_ebitda": 12.0, "ev_revenue": 1.5},
    "Consumer Staples": {"pe": 20.0, "ev_ebitda": 13.0, "ev_revenue": 2.0},
    "Energy": {"pe": 11.0, "ev_ebitda": 6.0, "ev_revenue": 1.2},
    "Industrials": {"pe": 19.0, "ev_ebitda": 12.0, "ev_revenue": 1.8},
    "Materials": {"pe": 16.0, "ev_ebitda": 9.0, "pb": 2.0},
    "Real Estate": {"ev_ebitda": 18.0, "pb": 1.6},
    "Communication Services": {"pe": 17.0, "ev_ebitda": 9.0, "ev_revenue": 2.5},
    "Utilities": {"pe": 17.0, "ev_ebitda": 11.0, "pb": 1.8},
}
DEFAULT_MULTIPLES: Dict[str, float] = {"pe": 16.0, "ev_ebitda": 11.0, "ev_revenue": 2.0}

MULTIPLE_LABELS = {
    "pe": "P/E", "ev_ebitda": "EV/EBITDA", "ev_revenue": "EV/Revenue", "ps": "P/S", "pb": "P/B",
}


RECOMMENDATION_THRESHOLDS = (
    # (minimum upside in %, label, colour, rationale)
    (25.0, "Strong Buy", "#74f174", "Significant undervaluation suggests strong upside potential."),
    (10.0, "Buy", "#a5d6a7", "Material undervaluation suggests an attractive entry point."),
    (-10.0, "Hold", "#fff59d", "Current valuation appears reasonable relative to intrinsic value."),
    (-25.0, "Reduce", "#ffab91", "Overvaluation suggests reducing position size."),
    (float("-inf"), "Sell", "#faa1a4", "Significant overvaluation indicates potential for material downside."),
)


def get_recommendation(upside_pct: Optional[float]) -> Dict[str, Any]:
    """
    Translate an upside/downside percentage into an investment recommendation.

    Args:
        upside_pct: Upside of fair value over the current price, in percent.

    Returns:
        Dict with ``label``, ``color`` and ``rationale`` keys. ``label`` is
        ``"N/A"`` when the upside is unknown.
    """
    if upside_pct is None or not np.isfinite(upside_pct):
        return {"label": "N/A", "color": "#9e9e9e", "rationale": "Insufficient data to generate a recommendation."}
    for threshold, label, color, rationale in RECOMMENDATION_THRESHOLDS:
        if upside_pct > threshold:
            return {"label": label, "color": color, "rationale": rationale}
    return {"label": "Sell", "color": "#faa1a4", "rationale": RECOMMENDATION_THRESHOLDS[-1][3]}


def _finite_positive(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def growth_path(initial: float, terminal: float, years: int) -> List[float]:
    """Annual growth rates fading linearly from ``initial`` (year 1) to ``terminal`` (final year)."""
    years = max(int(years), 1)
    if years == 1:
        return [float(initial)]
    return [initial + (terminal - initial) * t / (years - 1) for t in range(years)]


def compound(base: float, rates: Sequence[float]) -> List[float]:
    """Apply successive growth ``rates`` to ``base``."""
    values, current = [], float(base)
    for rate in rates:
        current *= 1 + rate
        values.append(current)
    return values


def annuity_factor(rate: float, years: float) -> float:
    """Present value of 1 per year for ``years`` years (end of year) at ``rate``."""
    if years <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(years)
    return (1 - (1 + rate) ** -years) / rate


def clean_result(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to Python types and NaN/inf to None."""
    if isinstance(obj, dict):
        return {_clean_key(k): clean_result(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {_clean_key(k): clean_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_result(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    return obj


def _clean_key(key: Any) -> Any:
    if isinstance(key, (pd.Timestamp, datetime)):
        return key.strftime('%Y-%m-%d')
    if isinstance(key, np.generic):
        return key.item()
    return key


def _chronological(series: pd.Series) -> pd.Series:
    """Return ``series`` ordered oldest -> newest."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    try:
        order = pd.to_datetime(pd.Index(series.index), errors="raise")
        return series.iloc[np.argsort(order.values)]
    except (ValueError, TypeError):
        return series.iloc[::-1]


class BaseValuation:
    """
    Base class for company valuation models.

    Provides the unified valuation contract plus generic DCF, relative,
    asset-based and dividend discount valuations that sector models reuse.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Args:
            data_loader: DataLoader used when ``financial_data`` is not supplied.
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self.risk_free_rate = RISK_FREE_RATE
        self.dcf_parameters = DCF_PARAMETERS
        self.sector_dcf_parameters = SECTOR_DCF_PARAMETERS

    # ------------------------------------------------------------------
    # Unified valuation contract
    # ------------------------------------------------------------------
    #: Canonical sector this model specialises in (see config.SECTOR_MAPPING).
    SECTOR: str = "General"

    #: Subsector / business-model options exposed to the UI as {key: label}.
    #: The first key is the default.
    SUBSECTORS: Dict[str, str] = {}

    #: Valuation methods produced by get_valuation as {key: display label}.
    METHODS: Dict[str, str] = {
        "dcf": "Discounted Cash Flow",
        "relative": "Relative Valuation (Multiples)",
        "asset_based": "Asset-Based Valuation",
        "dividend_discount": "Dividend Discount Model",
    }

    #: Optional per-subsector restriction of METHODS: {subsector: [method keys]}.
    SUBSECTOR_METHODS: Dict[str, List[str]] = {}

    #: Default blending weights for the methods returned by get_valuation.
    DEFAULT_METHOD_WEIGHTS: Dict[str, float] = {
        "dcf": 0.5,
        "relative": 0.3,
        "asset_based": 0.1,
        "dividend_discount": 0.1,
    }

    #: Optional per-subsector blending weights {subsector: {method: weight}};
    #: falls back to DEFAULT_METHOD_WEIGHTS (see get_method_weights).
    SUBSECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {}

    #: User-adjustable assumptions accepted as ``get_valuation`` keyword
    #: arguments, described so the UI can render controls generically::
    #:
    #:     name: {"label": str, "min": float, "max": float, "default": float,
    #:            "step": float, "kind": "percent" | "number" | "int" | "currency",
    #:            "help": str, "subsectors": [keys] | None (None = all)}
    #:
    #: ``percent`` values are fractions (0.10 == 10 %). When a parameter is not
    #: passed, the model derives it from the data (e.g. CAPM-based WACC).
    PARAMETERS: Dict[str, Dict[str, Any]] = {
        "discount_rate": {"label": "Discount rate (WACC)", "min": 0.04, "max": 0.20, "default": 0.10,
                          "step": 0.0025, "kind": "percent", "subsectors": None,
                          "help": "Rate used to discount future cash flows."},
        "terminal_growth": {"label": "Terminal growth rate", "min": 0.0, "max": 0.05, "default": 0.02,
                            "step": 0.0025, "kind": "percent", "subsectors": None,
                            "help": "Perpetual growth after the forecast period; must stay below the discount rate."},
        "forecast_years": {"label": "Forecast horizon (years)", "min": 3, "max": 15, "default": 5,
                           "step": 1, "kind": "int", "subsectors": None,
                           "help": "Number of explicitly forecast years."},
    }

    def get_available_methods(self, subsector: Optional[str] = None) -> Dict[str, str]:
        """Methods applicable to ``subsector`` as {key: label}."""
        keys = self.SUBSECTOR_METHODS.get(subsector) if subsector else None
        if not keys:
            return dict(self.METHODS)
        return {k: self.METHODS[k] for k in keys if k in self.METHODS}

    def get_parameters(self, subsector: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """PARAMETERS entries applicable to ``subsector``."""
        return {
            name: dict(spec) for name, spec in self.PARAMETERS.items()
            if not spec.get("subsectors") or subsector in spec["subsectors"]
        }

    def get_method_weights(self, subsector: Optional[str] = None) -> Dict[str, float]:
        """Default blending weights for ``subsector``."""
        weights = self.SUBSECTOR_WEIGHTS.get(subsector) if subsector else None
        return dict(weights or self.DEFAULT_METHOD_WEIGHTS)

    def detect_subsector(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                         company_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Best-effort guess of the subsector key (one of ``SUBSECTORS``) for a
        company. The base implementation returns the default subsector.
        """
        return next(iter(self.SUBSECTORS), None)

    def get_valuation(self, ticker: str, financial_data: Optional[Dict[str, Any]] = None,
                      subsector: Optional[str] = None, sector: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Run every applicable valuation method and blend the results.

        This is the single entry point used by the UI and ValuationFactory.
        Sector-specific subclasses override it but must keep the signature
        (unknown keyword arguments are ignored) and the result schema produced
        by :meth:`build_valuation_result`.

        Args:
            ticker: Company ticker symbol.
            financial_data: Canonical financial data bundle
                (see ``utils.financial_data.build_financial_data``). Loaded via
                the data loader when omitted.
            subsector: Optional subsector key from ``SUBSECTORS``.
            sector: Optional canonical sector name (for parameter lookup).
            **kwargs: Model specific overrides declared in ``PARAMETERS``
                (e.g. ``discount_rate``, ``terminal_growth``, ``forecast_years``),
                plus ``weights`` ({method: weight}) and ``methods`` (list of
                method keys to run; default: all available for the subsector).

        Returns:
            Standardised valuation result dictionary.
        """
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            self._require_statements(financial_data)
            sector = sector or (financial_data.get('company_info') or {}).get('sector') or self.SECTOR

            overrides = self._parameter_overrides(subsector, kwargs)
            rate_kwargs = {k: overrides.get(k) for k in ('discount_rate', 'terminal_growth', 'forecast_years')}
            runners = self._method_runners(ticker, financial_data, sector, rate_kwargs, kwargs)
            notes: List[str] = []
            methods = self._run_methods(runners, financial_data, subsector, kwargs.get('methods'), notes)

            wacc = self._wacc_components(financial_data, sector)
            params = self._get_dcf_parameters(sector)
            assumptions = {
                'discount_rate': rate_kwargs['discount_rate'] or wacc['wacc'],
                'terminal_growth': (rate_kwargs['terminal_growth'] if rate_kwargs['terminal_growth'] is not None
                                    else params['terminal_growth_rate']),
                'forecast_years': rate_kwargs['forecast_years'] or params['forecast_years'],
                'cost_of_equity': wacc['cost_of_equity'],
                'wacc_components': wacc,
                'user_overrides': overrides,
            }
            return self.build_valuation_result(
                ticker, financial_data, methods,
                weights=kwargs.get('weights') or self.get_method_weights(subsector),
                sector=sector, subsector=subsector, assumptions=assumptions, notes=notes,
            )
        except Exception as e:
            logger.error(f"Error in valuation for {ticker}: {e}")
            return self.build_error_result(ticker, e, sector=sector, subsector=subsector)

    def build_valuation_result(self, ticker: str, financial_data: Optional[Dict[str, Any]],
                               methods: Dict[str, Dict[str, Any]],
                               weights: Optional[Dict[str, float]] = None,
                               sector: Optional[str] = None, subsector: Optional[str] = None,
                               assumptions: Optional[Dict[str, Any]] = None,
                               notes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Assemble the standardised valuation result.

        Schema::

            {
                "company": str, "sector": str, "subsector": str | None,
                "model": str,                       # class name
                "current_price": float | None,
                "value_per_share": float | None,    # weighted fair value
                "potential_upside": float | None,   # percent
                "recommendation": {"label", "color", "rationale"},
                "valuation_summary": {...},         # see summarize_valuations
                "valuation_methods": {key: method_result_dict},
                "assumptions": dict, "notes": [str], "timestamp": str,
            }

        Each method result should contain at least ``method`` (display name)
        and ``value_per_share`` (None when the method is not applicable), and
        ``error`` when it failed.
        """
        market_data = (financial_data or {}).get('market_data') or {}
        current_price = _finite_positive(market_data.get('share_price'))
        methods = {k: clean_result(v) for k, v in (methods or {}).items() if isinstance(v, dict)}
        for key, result in methods.items():
            result.setdefault('method', self.METHODS.get(key, key))
        summary = self.summarize_valuations(methods, weights, current_price)
        return {
            'company': ticker,
            'sector': sector or self.SECTOR,
            'subsector': subsector,
            'model': self.__class__.__name__,
            'current_price': current_price,
            'value_per_share': summary['weighted_value'],
            'potential_upside': summary['potential_upside'],
            'recommendation': get_recommendation(summary['potential_upside']),
            'valuation_summary': summary,
            'valuation_methods': methods,
            'assumptions': clean_result(assumptions or {}),
            'notes': list(notes or []),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def build_error_result(self, ticker: str, error: Any, sector: Optional[str] = None,
                           subsector: Optional[str] = None) -> Dict[str, Any]:
        """Standardised result for a valuation that failed entirely."""
        summary = self.summarize_valuations({}, None, None)
        return {
            'company': ticker,
            'sector': sector or self.SECTOR,
            'subsector': subsector,
            'model': self.__class__.__name__,
            'current_price': None,
            'value_per_share': None,
            'potential_upside': None,
            'recommendation': get_recommendation(None),
            'valuation_summary': summary,
            'valuation_methods': {},
            'assumptions': {},
            'notes': [],
            'error': str(error),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    @staticmethod
    def summarize_valuations(methods: Dict[str, Dict[str, Any]],
                             weights: Optional[Dict[str, float]] = None,
                             current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Blend per-method fair values into a single weighted estimate.

        Methods without a finite, positive ``value_per_share`` are excluded
        (a negative DCF equity value is reported by the method itself but is
        not averaged). Weights of the remaining methods are re-normalised to
        sum to 1; methods missing from ``weights`` get weight 1 before
        normalisation, and if all weights are zero they are treated as equal.

        Returns:
            Dict with ``weighted_value``, ``average`` (alias of weighted value,
            kept for backward compatibility), ``simple_average``, ``median``,
            ``min``, ``max``, ``range``, ``potential_upside`` (percent),
            ``weights`` (normalised), ``methods_used`` and ``excluded``
            ({method: reason}).
        """
        values: Dict[str, float] = {}
        excluded: Dict[str, str] = {}
        for key, result in (methods or {}).items():
            value = _finite_positive(result.get('value_per_share')) if isinstance(result, dict) else None
            if value is None:
                if isinstance(result, dict) and result.get('error'):
                    excluded[key] = str(result['error'])
                elif isinstance(result, dict) and result.get('value_per_share') is not None:
                    excluded[key] = 'non-positive or invalid value'
                else:
                    excluded[key] = 'not applicable'
                continue
            values[key] = value

        empty = {
            'weighted_value': None, 'average': None, 'simple_average': None, 'median': None,
            'min': None, 'max': None, 'range': None, 'potential_upside': None,
            'weights': {}, 'methods_used': [], 'excluded': excluded,
        }
        if not values:
            return empty

        raw = {k: max(float((weights or {}).get(k, 1.0)), 0.0) for k in values}
        total = sum(raw.values())
        if total <= 0:
            raw = {k: 1.0 for k in values}
            total = float(len(values))
        normalized = {k: w / total for k, w in raw.items()}
        used = [k for k in values if normalized[k] > 0]

        weighted = float(sum(values[k] * normalized[k] for k in values))
        series = np.array([values[k] for k in used] or list(values.values()), dtype=float)
        upside = (weighted / current_price - 1) * 100 if _finite_positive(current_price) else None

        return {
            'weighted_value': weighted,
            'average': weighted,
            'simple_average': float(series.mean()),
            'median': float(np.median(series)),
            'min': float(series.min()),
            'max': float(series.max()),
            'range': float(series.max() - series.min()),
            'potential_upside': upside,
            'weights': normalized,
            'methods_used': used,
            'excluded': excluded,
        }

    # ------------------------------------------------------------------
    # Method orchestration helpers (shared by sector models)
    # ------------------------------------------------------------------
    def _method_runners(self, ticker: str, financial_data: Dict[str, Any], sector: Optional[str],
                        rate_kwargs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Callable[[], Dict]]:
        """Callables producing each method result; subclasses extend this mapping."""
        return {
            'dcf': lambda: self.discounted_cash_flow_valuation(ticker, financial_data, sector, **rate_kwargs),
            'relative': lambda: self.relative_valuation(ticker, financial_data, sector),
            'asset_based': lambda: self.asset_based_valuation(ticker, financial_data, sector),
            'dividend_discount': lambda: self.dividend_discount_valuation(ticker, financial_data, sector,
                                                                          **rate_kwargs),
        }

    def _run_methods(self, runners: Dict[str, Callable[[], Dict]], financial_data: Dict[str, Any],
                     subsector: Optional[str], requested: Any, notes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run the requested (or all available) methods, isolating failures.

        Methods not applicable to ``subsector`` are skipped with a note.
        """
        available = [k for k in self.get_available_methods(subsector) if k in runners]
        if requested:
            requested = [requested] if isinstance(requested, str) else list(requested)
            selected = [k for k in requested if k in available]
            skipped = [str(k) for k in requested if k not in available]
            if skipped:
                notes.append(f"Skipped methods not applicable to {subsector or self.SECTOR}: {', '.join(skipped)}.")
        else:
            selected = available

        results = {}
        for key in dict.fromkeys(selected):
            try:
                result = runners[key]()
            except Exception as e:  # each method is isolated
                logger.warning(f"{self.__class__.__name__}: method {key} failed: {e}")
                result = {'value_per_share': None, 'error': str(e)}
            results[key] = self._finalize_method(key, result, financial_data)
        return results

    def _finalize_method(self, key: str, result: Dict[str, Any],
                         financial_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Label a method result and guard against invalid or implausible values."""
        result = clean_result(dict(result or {}))
        result['method'] = self.METHODS.get(key, result.get('method') or key)
        result['method_key'] = key
        raw = result.get('value_per_share')
        value = _finite_positive(raw)
        if value is None:
            if raw is not None:
                result.setdefault('error', 'Valuation produced a non-positive value per share')
            result['value_per_share'] = None
            return result
        price = self._current_price(financial_data)
        if price:
            low, high = PLAUSIBLE_PRICE_MULTIPLE
            if not low * price <= value <= high * price:
                result['implausible_value'] = value
                result['value_per_share'] = None
                result.setdefault('error', f"Implausible result ({value / price:.2f}x the share price)")
        return result

    def _method_error(self, label: str, error: Any, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Standard per-method failure dict."""
        logger.info(f"{label} not available for {ticker}: {error}")
        return {'company': ticker, 'method': label, 'value_per_share': None, 'error': str(error)}

    def _parameter_overrides(self, subsector: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """User-supplied PARAMETERS applicable to ``subsector`` (numeric, ints rounded)."""
        overrides = {}
        for name, spec in self.get_parameters(subsector).items():
            value = to_float(kwargs.get(name))
            if value is None:
                continue
            overrides[name] = int(round(value)) if spec.get('kind') == 'int' else value
        if 'forecast_years' in overrides:
            overrides['forecast_years'] = int(clamp(overrides['forecast_years'], 1, 50))
        return overrides

    def _resolve_parameters(self, subsector: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """PARAMETERS defaults for ``subsector`` updated with user overrides."""
        params = {name: spec.get('default') for name, spec in self.get_parameters(subsector).items()}
        params.update(self._parameter_overrides(subsector, kwargs))
        return params

    @staticmethod
    def _require_statements(financial_data: Any) -> None:
        """Raise ValueError when no financial statement is available."""
        if not isinstance(financial_data, dict):
            raise ValueError("Financial data is unavailable")
        if all(is_empty(financial_data.get(k)) for k in ('income_statement', 'balance_sheet', 'cash_flow')):
            raise ValueError("Financial statements are unavailable")

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _statements(financial_data: Optional[Dict[str, Any]]):
        """(income statement, balance sheet, cash flow) as DataFrames (empty when missing)."""
        data = financial_data or {}

        def frame(key):
            df = data.get(key)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        return frame('income_statement'), frame('balance_sheet'), frame('cash_flow')

    @staticmethod
    def _market_data(financial_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return (financial_data or {}).get('market_data') or {}

    @staticmethod
    def _company_info(financial_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return (financial_data or {}).get('company_info') or {}

    def _current_price(self, financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        return _finite_positive(self._market_data(financial_data).get('share_price'))

    def _shares_outstanding(self, financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        """Shares outstanding from market data, the balance sheet, diluted shares or market cap / price."""
        income, balance, _ = self._statements(financial_data)
        market = self._market_data(financial_data)
        for candidate in (market.get('shares_outstanding'), get_latest(balance, 'shares_outstanding'),
                          get_latest(income, 'diluted_shares')):
            shares = _finite_positive(candidate)
            if shares:
                return shares
        market_cap, price = _finite_positive(market.get('market_cap')), self._current_price(financial_data)
        return market_cap / price if market_cap and price else None

    def _market_cap(self, financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        market_cap = _finite_positive(self._market_data(financial_data).get('market_cap'))
        if market_cap:
            return market_cap
        shares, price = self._shares_outstanding(financial_data), self._current_price(financial_data)
        return shares * price if shares and price else None

    @staticmethod
    def _total_debt(balance_sheet: Optional[pd.DataFrame]) -> float:
        """Total debt (falls back to long-term + current debt); 0 when unavailable."""
        debt = get_latest(balance_sheet, 'total_debt')
        if debt is None:
            parts = [get_latest(balance_sheet, 'long_term_debt'), get_latest(balance_sheet, 'short_term_debt')]
            debt = sum(p for p in parts if p is not None)
        return max(float(debt or 0.0), 0.0)

    @staticmethod
    def _cash(balance_sheet: Optional[pd.DataFrame]) -> float:
        """Cash, equivalents and short-term investments; 0 when unavailable."""
        cash = get_latest(balance_sheet, 'cash_and_short_term_investments')
        if cash is None:
            cash = get_latest(balance_sheet, 'cash')
        return max(float(cash or 0.0), 0.0)

    def _calculate_net_debt(self, balance_sheet: pd.DataFrame) -> float:
        """Net debt = total debt - cash and short-term investments."""
        if is_empty(balance_sheet):
            return 0.0
        return self._total_debt(balance_sheet) - self._cash(balance_sheet)

    def _equity_bridge(self, enterprise_value: float, financial_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enterprise value -> equity value -> value per share.

        Equity = EV - total debt + cash - minority interest - preferred stock.
        ``value_per_share`` is None (with ``error``) for a non-positive equity
        value or unknown share count.
        """
        _, balance, _ = self._statements(financial_data)
        debt, cash = self._total_debt(balance), self._cash(balance)
        minority = max(get_latest(balance, 'minority_interest') or 0.0, 0.0)
        preferred = max(get_latest(balance, 'preferred_stock') or 0.0, 0.0)
        equity = float(enterprise_value) - debt + cash - minority - preferred
        shares = self._shares_outstanding(financial_data)
        result = {
            'enterprise_value': float(enterprise_value), 'total_debt': debt, 'cash': cash,
            'net_debt': debt - cash, 'minority_interest': minority, 'preferred_stock': preferred,
            'equity_value': equity, 'shares_outstanding': shares,
            'value_per_share': equity / shares if shares and equity > 0 else None,
        }
        if equity <= 0:
            result['error'] = 'Equity value is negative: debt and other claims exceed enterprise value'
        elif not shares:
            result['error'] = 'Shares outstanding unavailable'
        return result

    def _per_share(self, equity_value: Optional[float], financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        shares = self._shares_outstanding(financial_data)
        equity_value = to_float(equity_value)
        return equity_value / shares if shares and equity_value is not None and equity_value > 0 else None

    def _ebitda(self, financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        """Latest EBITDA (reported, else operating income + D&A)."""
        income, _, cash_flow = self._statements(financial_data)
        ebitda = get_latest(income, 'ebitda')
        if ebitda is None:
            ebit = get_latest(income, 'operating_income')
            dep = get_latest(income, 'depreciation') or get_latest(cash_flow, 'cash_flow_depreciation')
            ebitda = ebit + dep if ebit is not None and dep is not None else None
        return ebitda

    def _depreciation(self, financial_data: Optional[Dict[str, Any]]) -> float:
        income, _, cash_flow = self._statements(financial_data)
        return abs(get_latest(cash_flow, 'cash_flow_depreciation') or get_latest(income, 'depreciation') or 0.0)

    def _capex(self, financial_data: Optional[Dict[str, Any]], years: int = 1) -> float:
        """Average absolute capital expenditure over the latest ``years`` periods."""
        _, _, cash_flow = self._statements(financial_data)
        series = get_series(cash_flow, 'capital_expenditure').abs().head(max(years, 1))
        return float(series.mean()) if not series.empty else 0.0

    def _effective_tax_rate(self, financial_data: Optional[Dict[str, Any]]) -> float:
        """Average effective tax rate of the last three profitable years, bounded to [0, 35 %]."""
        income, _, _ = self._statements(financial_data)
        tax, pretax = get_series(income, 'tax_provision'), get_series(income, 'pretax_income')
        aligned = pd.concat([tax, pretax], axis=1, join='inner').head(3)
        rates = [t / p for t, p in aligned.to_numpy() if p > 0]
        rate = float(np.mean(rates)) if rates else to_float(get_latest(income, 'tax_rate'))
        if rate is None or not 0 <= rate <= 0.6:
            rate = DEFAULT_TAX_RATE
        return clamp(rate, 0.0, 0.35)

    # ------------------------------------------------------------------
    # Discount rates
    # ------------------------------------------------------------------
    def _get_dcf_parameters(self, sector: Optional[str] = None) -> Dict[str, Any]:
        """DCF defaults (config.DCF_PARAMETERS) with sector-specific overrides."""
        params = dict(self.dcf_parameters)
        if sector:
            params.update(self.sector_dcf_parameters.get(normalize_sector(sector), {}))
        return params

    def _wacc_components(self, financial_data: Optional[Dict[str, Any]], sector: Optional[str] = None) -> Dict[str, Any]:
        """
        CAPM cost of equity and WACC with their inputs.

        Ke = rf + beta * ERP (beta defaults to 1.0); Kd = interest / debt bounded
        to [rf + 0.5 %, 15 %]; weights from market cap (book equity fallback)
        and book debt; WACC bounded to WACC_BOUNDS.
        """
        income, balance, _ = self._statements(financial_data)
        market = self._market_data(financial_data)
        raw_beta = to_float(market.get('beta'))
        if raw_beta is None:
            raw_beta = to_float(self._company_info(financial_data).get('beta'))
        beta = clamp(raw_beta, 0.3, 3.0) if raw_beta is not None else 1.0
        rf = self.risk_free_rate
        cost_of_equity = clamp(rf + beta * EQUITY_RISK_PREMIUM, *COST_OF_EQUITY_BOUNDS)

        debt = self._total_debt(balance)
        interest = abs(get_latest(income, 'interest_expense') or 0.0)
        implied_kd = safe_divide(interest, debt)
        cost_of_debt = clamp(implied_kd, rf + 0.005, 0.15) if implied_kd else rf + 0.02
        tax_rate = self._effective_tax_rate(financial_data)

        equity = self._market_cap(financial_data) or _finite_positive(get_latest(balance, 'total_equity'))
        if equity:
            equity_weight = equity / (equity + debt)
        else:
            equity_weight = 0.7
        debt_weight = 1 - equity_weight
        raw_wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
        return {
            'risk_free_rate': rf, 'beta': beta, 'beta_assumed': raw_beta is None,
            'equity_risk_premium': EQUITY_RISK_PREMIUM, 'cost_of_equity': cost_of_equity,
            'pre_tax_cost_of_debt': cost_of_debt, 'tax_rate': tax_rate,
            'after_tax_cost_of_debt': cost_of_debt * (1 - tax_rate),
            'equity_weight': equity_weight, 'debt_weight': debt_weight,
            'wacc': clamp(raw_wacc, *WACC_BOUNDS),
        }

    def _calculate_discount_rate(self, ticker: str, financial_data: Dict[str, Any],
                                 sector: Optional[str] = None) -> Optional[float]:
        """WACC (CAPM cost of equity, after-tax cost of debt, market-value weights)."""
        try:
            return self._wacc_components(financial_data, sector)['wacc']
        except Exception as e:
            logger.warning(f"WACC calculation failed for {ticker}: {e}")
            return self._get_dcf_parameters(sector)['default_discount_rate']

    def _cost_of_equity(self, financial_data: Optional[Dict[str, Any]],
                        discount_rate: Optional[float] = None) -> float:
        """
        CAPM cost of equity. When the user overrides the WACC, the cost of
        equity is shifted by the same amount so equity methods move with it.
        """
        components = self._wacc_components(financial_data)
        cost_of_equity = components['cost_of_equity']
        if discount_rate is not None:
            cost_of_equity += discount_rate - components['wacc']
        return cost_of_equity

    # ------------------------------------------------------------------
    # Cash flow history and growth
    # ------------------------------------------------------------------
    def _calculate_historical_fcf(self, income_stmt: pd.DataFrame, cash_flow: pd.DataFrame) -> pd.Series:
        """
        Reported free cash flow per period (most recent first).

        Uses the ``Free Cash Flow`` row, else operating cash flow - |capex|,
        else net income (maintenance capex assumed equal to D&A).
        """
        fcf = get_series(cash_flow, 'free_cash_flow')
        if fcf.empty:
            ocf = get_series(cash_flow, 'operating_cash_flow')
            if not ocf.empty:
                capex = get_series(cash_flow, 'capital_expenditure').abs()
                capex = capex.reindex(ocf.index) if not capex.empty else pd.Series(0.0, index=ocf.index)
                fcf = (ocf - capex).dropna()
        if fcf.empty:
            fcf = get_series(income_stmt, 'net_income')
        return fcf.rename('Free Cash Flow')

    def _unlevered_fcf_series(self, financial_data: Optional[Dict[str, Any]]) -> pd.Series:
        """Free cash flow to the firm: reported FCF + after-tax interest expense (most recent first)."""
        income, _, cash_flow = self._statements(financial_data)
        fcf = self._calculate_historical_fcf(income, cash_flow)
        if fcf.empty:
            return fcf
        interest = get_series(income, 'interest_expense').abs().reindex(fcf.index).fillna(0.0)
        return (fcf + interest * (1 - self._effective_tax_rate(financial_data))).rename('Unlevered FCF')

    @staticmethod
    def _base_cash_flow(series: pd.Series, average_years: int = 1) -> float:
        """
        Starting cash flow for a DCF: the latest value (or the average of the
        latest ``average_years``); falls back to the three-year average when
        the latest is not positive. Raises ValueError when not positive.
        """
        values = pd.to_numeric(series, errors='coerce').dropna()
        if values.empty:
            raise ValueError("Free cash flow history unavailable")
        base = float(values.head(max(average_years, 1)).mean())
        if base <= 0:
            base = float(values.head(3).mean())
        if base <= 0:
            raise ValueError("Free cash flow is negative; a cash flow based valuation is not meaningful")
        return base

    def _estimate_growth_rate(self, historical_fcf: pd.Series) -> float:
        """Growth implied by a cash flow history (CAGR, else weighted YoY), bounded to [-5 %, 20 %]."""
        series = _chronological(historical_fcf).tail(5)
        growth = cagr(series)
        if growth is None and len(series) >= 2:
            changes = [(b / a) - 1 for a, b in zip(series.iloc[:-1], series.iloc[1:]) if a > 0 and b > 0]
            if changes:
                weights = np.arange(1, len(changes) + 1)
                growth = float(np.average(changes, weights=weights))
        return clamp(growth, -0.05, 0.20) if growth is not None else 0.03

    def _estimate_company_growth(self, financial_data: Optional[Dict[str, Any]],
                                 fcf_series: Optional[pd.Series] = None) -> float:
        """
        Near-term growth estimate: 60 % revenue CAGR + 40 % FCF CAGR (whichever
        exist), bounded to GROWTH_BOUNDS; 3 % when no history is available.
        """
        income, _, _ = self._statements(financial_data)
        revenue_growth = cagr(get_series(income, 'revenue', ascending=True).tail(5))
        fcf_growth = cagr(_chronological(fcf_series).tail(5)) if fcf_series is not None else None
        if fcf_growth is not None:
            fcf_growth = clamp(fcf_growth, -0.20, 0.40)
        if revenue_growth is not None and fcf_growth is not None:
            growth = 0.6 * revenue_growth + 0.4 * fcf_growth
        else:
            growth = revenue_growth if revenue_growth is not None else fcf_growth
        return clamp(growth, *GROWTH_BOUNDS) if growth is not None else 0.03

    def _forecast_fcf(self, last_fcf: float, forecast_years: int, growth_rate: float,
                      terminal_growth: Optional[float] = None) -> List[float]:
        """Project cash flows at a constant rate, or fading to ``terminal_growth`` when given."""
        if terminal_growth is None:
            rates = [growth_rate] * max(int(forecast_years), 1)
        else:
            rates = growth_path(growth_rate, terminal_growth, forecast_years)
        return compound(last_fcf, rates)

    def _calculate_terminal_value(self, final_fcf: float, terminal_growth: float, discount_rate: float) -> float:
        """Gordon growth terminal value; raises ValueError unless discount rate > growth."""
        if discount_rate <= terminal_growth:
            raise ValueError(f"Discount rate ({discount_rate:.2%}) must exceed terminal growth "
                             f"({terminal_growth:.2%}) for a terminal value")
        final_fcf = to_float(final_fcf)
        if final_fcf is None:
            raise ValueError("Final cash flow is not a finite number")
        return final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

    def _discount_cash_flows(self, cash_flows: Sequence[float], discount_rate: float, terminal_growth: float,
                             growth_rates: Optional[Sequence[float]] = None,
                             terminal_cash_flow: Optional[float] = None) -> Dict[str, Any]:
        """
        Present value of explicit year-end cash flows plus a Gordon terminal value.

        ``terminal_cash_flow`` (the first post-forecast cash flow) defaults to
        the final explicit cash flow grown at ``terminal_growth``.
        """
        if not cash_flows:
            raise ValueError("No cash flows to discount")
        if discount_rate <= -1:
            raise ValueError("Discount rate must be greater than -100%")
        if terminal_cash_flow is None:
            terminal_value = self._calculate_terminal_value(cash_flows[-1], terminal_growth, discount_rate)
        else:
            if discount_rate <= terminal_growth:
                raise ValueError(f"Discount rate ({discount_rate:.2%}) must exceed terminal growth "
                                 f"({terminal_growth:.2%}) for a terminal value")
            terminal_value = terminal_cash_flow / (discount_rate - terminal_growth)

        projections = []
        for year, cash_flow in enumerate(cash_flows, 1):
            factor = (1 + discount_rate) ** -year
            projections.append({
                'year': year, 'growth': growth_rates[year - 1] if growth_rates else None,
                'fcf': float(cash_flow), 'discount_factor': factor, 'present_value': float(cash_flow) * factor,
            })
        pv_fcf = sum(p['present_value'] for p in projections)
        pv_terminal = terminal_value * (1 + discount_rate) ** -len(cash_flows)
        enterprise_value = pv_fcf + pv_terminal
        return {
            'forecast_fcf': [float(c) for c in cash_flows],
            'projections': projections,
            'terminal_value': terminal_value,
            'present_value_fcf': pv_fcf,
            'present_value_terminal': pv_terminal,
            'enterprise_value': enterprise_value,
            'terminal_value_share': pv_terminal / enterprise_value if enterprise_value > 0 else None,
        }

    def _dcf_inputs(self, financial_data: Dict[str, Any], sector: Optional[str] = None,
                    discount_rate: Optional[float] = None, terminal_growth: Optional[float] = None,
                    forecast_years: Optional[int] = None, growth_rate: Optional[float] = None,
                    cash_flows: Optional[pd.Series] = None, average_years: int = 1) -> Dict[str, Any]:
        """Resolve DCF inputs: base cash flow, discount rate, growth, terminal growth and horizon."""
        params = self._get_dcf_parameters(sector)
        series = self._unlevered_fcf_series(financial_data) if cash_flows is None else cash_flows
        base = self._base_cash_flow(series, average_years)
        wacc = self._wacc_components(financial_data, sector)
        return {
            'base_fcf': base,
            'historical_fcf': series,
            'discount_rate': discount_rate if discount_rate is not None else wacc['wacc'],
            'terminal_growth': terminal_growth if terminal_growth is not None else params['terminal_growth_rate'],
            'forecast_years': int(forecast_years or params['forecast_years']),
            'growth_rate': (growth_rate if growth_rate is not None
                            else self._estimate_company_growth(financial_data, series)),
            'wacc_components': wacc,
            'safety_margin': params['default_margin_of_safety'],
        }

    def _dcf_result(self, label: str, ticker: str, inputs: Dict[str, Any], rates: Sequence[float],
                    financial_data: Dict[str, Any], **extra) -> Dict[str, Any]:
        """Run the DCF engine and the equity bridge for resolved ``inputs`` and growth ``rates``."""
        flows = compound(inputs['base_fcf'], rates)
        dcf = self._discount_cash_flows(flows, inputs['discount_rate'], inputs['terminal_growth'], growth_rates=rates)
        bridge = self._equity_bridge(dcf['enterprise_value'], financial_data)
        value = bridge['value_per_share']
        result = {
            'company': ticker, 'method': label, **dcf, **bridge,
            'discount_rate': inputs['discount_rate'], 'growth_rate': inputs['growth_rate'],
            'terminal_growth': inputs['terminal_growth'], 'forecast_years': len(rates),
            'base_fcf': inputs['base_fcf'], 'historical_fcf': clean_result(inputs['historical_fcf']),
            'wacc_components': inputs['wacc_components'], 'safety_margin': inputs['safety_margin'],
            'conservative_value': value * (1 - inputs['safety_margin']) if value else None,
        }
        result.update(extra)
        return result

    # ------------------------------------------------------------------
    # Generic valuation methods
    # ------------------------------------------------------------------
    def calculate_intrinsic_value(self, ticker: str, financial_data: Dict[str, Any] = None,
                                  sector: str = None, method: str = 'dcf') -> Dict[str, Any]:
        """
        Value a company with a single method.

        Args:
            method: 'dcf', 'relative', 'asset-based'/'asset_based' or
                'dividend'/'dividend_discount'.
        """
        try:
            if financial_data is None:
                financial_data = self._load_financial_data(ticker)
            dispatch = {
                'dcf': self.discounted_cash_flow_valuation,
                'relative': self.relative_valuation,
                'asset-based': self.asset_based_valuation,
                'asset_based': self.asset_based_valuation,
                'dividend': self.dividend_discount_valuation,
                'dividend_discount': self.dividend_discount_valuation,
            }
            if method not in dispatch:
                raise ValueError(f"Unsupported valuation method: {method}")
            return dispatch[method](ticker, financial_data, sector)
        except Exception as e:
            logger.error(f"Error calculating intrinsic value for {ticker}: {e}")
            return {'company': ticker, 'method': method, 'value_per_share': None, 'total_value': None,
                    'error': str(e)}

    def discounted_cash_flow_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                       sector: str = None, discount_rate: Optional[float] = None,
                                       terminal_growth: Optional[float] = None,
                                       forecast_years: Optional[int] = None,
                                       growth_rate: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Two-stage unlevered DCF.

        Free cash flow to the firm grows at the estimated (or given) rate,
        fading linearly to ``terminal_growth`` over the forecast horizon, and
        is discounted at the WACC (or ``discount_rate``). Gordon growth
        terminal value; EV -> equity via the standard bridge.
        """
        label = 'Discounted Cash Flow'
        try:
            inputs = self._dcf_inputs(financial_data, sector, discount_rate, terminal_growth,
                                      forecast_years, growth_rate)
            rates = growth_path(inputs['growth_rate'], inputs['terminal_growth'], inputs['forecast_years'])
            return self._dcf_result(label, ticker, inputs, rates, financial_data)
        except Exception as e:
            return self._method_error(label, e, ticker)

    def _get_sector_multiples(self, sector: Optional[str] = None) -> Dict[str, float]:
        """Peer multiples for a sector (any spelling)."""
        return dict(SECTOR_MULTIPLES.get(normalize_sector(sector), DEFAULT_MULTIPLES))

    def _valuation_metrics(self, financial_data: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Latest earnings, EBITDA, revenue and book value used by multiples."""
        income, balance, _ = self._statements(financial_data)
        return {
            'earnings': get_latest(income, 'net_income'),
            'ebitda': self._ebitda(financial_data),
            'revenue': get_latest(income, 'revenue'),
            'book_value': get_latest(balance, 'total_equity'),
        }

    def _multiples_valuation(self, ticker: str, financial_data: Dict[str, Any], multiples: Dict[str, float],
                             label: str = 'Relative Valuation (Multiples)',
                             metrics: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Any]:
        """
        Implied values from peer multiples.

        Supported keys: ``pe``, ``ps``, ``pb`` (equity multiples) and
        ``ev_ebitda``, ``ev_revenue`` (EV multiples, bridged to equity). The
        fair value is the median of the positive implied per-share values.
        """
        try:
            metrics = metrics or self._valuation_metrics(financial_data)
            metric_for = {'pe': 'earnings', 'ps': 'revenue', 'pb': 'book_value',
                          'ev_ebitda': 'ebitda', 'ev_revenue': 'revenue'}
            implied = {}
            for key, multiple in multiples.items():
                multiple = _finite_positive(multiple)
                metric = to_float(metrics.get(metric_for.get(key, '')))
                if key not in metric_for or multiple is None:
                    continue
                entry = {'label': MULTIPLE_LABELS.get(key, key), 'multiple': multiple, 'metric': metric,
                         'value_per_share': None}
                if metric is None or metric <= 0:
                    entry['note'] = f"{metric_for[key].replace('_', ' ')} is not positive"
                elif key.startswith('ev_'):
                    bridge = self._equity_bridge(metric * multiple, financial_data)
                    entry.update(enterprise_value=metric * multiple, equity_value=bridge['equity_value'],
                                 value_per_share=bridge['value_per_share'])
                else:
                    entry.update(equity_value=metric * multiple,
                                 value_per_share=self._per_share(metric * multiple, financial_data))
                implied[key] = entry
            values = [v['value_per_share'] for v in implied.values() if _finite_positive(v['value_per_share'])]
            result = {
                'company': ticker, 'method': label, 'valuations': implied, 'metrics': metrics,
                'value_per_share': float(np.median(values)) if values else None,
                'average_value_per_share': float(np.mean(values)) if values else None,
            }
            if not values:
                result['error'] = 'No positive metric available for the selected multiples'
            return result
        except Exception as e:
            return self._method_error(label, e, ticker)

    def relative_valuation(self, ticker: str, financial_data: Dict[str, Any],
                           sector: str = None, multiples: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Value from sector peer multiples (P/E, EV/EBITDA, EV/Revenue, P/B as applicable)."""
        return self._multiples_valuation(ticker, financial_data, multiples or self._get_sector_multiples(sector))

    def asset_based_valuation(self, ticker: str, financial_data: Dict[str, Any],
                              sector: str = None) -> Dict[str, Any]:
        """
        Book, tangible book (the fair value used) and liquidation value.

        Liquidation value applies haircuts to current and fixed assets
        (cash 100 %, receivables 80 %, inventory 50 %, net PP&E 40 %) and
        deducts total liabilities.
        """
        label = 'Asset-Based Valuation'
        try:
            _, balance, _ = self._statements(financial_data)
            if is_empty(balance):
                raise ValueError("Balance sheet unavailable")
            total_assets = get_latest(balance, 'total_assets')
            total_liabilities = get_latest(balance, 'total_liabilities')
            book_value = get_latest(balance, 'total_equity')
            if book_value is None and total_assets is not None and total_liabilities is not None:
                book_value = total_assets - total_liabilities
            if book_value is None:
                raise ValueError("Unable to determine book value from the balance sheet")

            intangibles = get_latest(balance, 'goodwill_and_intangibles')
            if intangibles is None:
                intangibles = (get_latest(balance, 'goodwill') or 0.0) + (get_latest(balance, 'intangible_assets') or 0.0)
            tangible_book = get_latest(balance, 'tangible_book_value')
            if tangible_book is None:
                tangible_book = book_value - intangibles

            liquidation_value = None
            if total_liabilities is not None:
                recoverable = (self._cash(balance) + 0.8 * (get_latest(balance, 'receivables') or 0.0)
                               + 0.5 * (get_latest(balance, 'inventory') or 0.0)
                               + 0.4 * (get_latest(balance, 'net_ppe') or 0.0))
                liquidation_value = recoverable - total_liabilities

            def per_share(value):
                return self._per_share(value, financial_data)

            result = {
                'company': ticker, 'method': label,
                'book_value': book_value, 'adjusted_book_value': tangible_book,
                'tangible_book_value': tangible_book, 'intangible_assets': intangibles,
                'liquidation_value': liquidation_value,
                'book_value_per_share': per_share(book_value),
                'adjusted_book_value_per_share': per_share(tangible_book),
                'liquidation_value_per_share': per_share(liquidation_value),
                'value_per_share': per_share(tangible_book),
            }
            if result['value_per_share'] is None:
                result['error'] = 'Tangible book value is not positive'
            return result
        except Exception as e:
            return self._method_error(label, e, ticker)

    def _return_on_equity(self, financial_data: Optional[Dict[str, Any]], years: int = 3) -> Optional[float]:
        """Average ROE (net income / average equity) over the latest ``years`` periods."""
        income, balance, _ = self._statements(financial_data)
        net_income = get_series(income, 'net_income', ascending=True)
        equity = get_series(balance, 'total_equity', ascending=True)
        if net_income.empty or equity.empty:
            return None
        average_equity = ((equity + equity.shift(1)) / 2).fillna(equity)
        aligned = pd.concat([net_income, average_equity], axis=1, join='inner').tail(years)
        rates = [ni / eq for ni, eq in aligned.to_numpy() if eq > 0]
        return float(np.mean(rates)) if rates else None

    def _dividends_per_share(self, financial_data: Optional[Dict[str, Any]]) -> Optional[float]:
        """Trailing dividends per share from cash dividends paid, else price x yield."""
        _, _, cash_flow = self._statements(financial_data)
        shares = self._shares_outstanding(financial_data)
        dividends = get_latest(cash_flow, 'dividends_paid')
        if dividends and shares:
            return abs(dividends) / shares
        market = self._market_data(financial_data)
        price, dividend_yield = self._current_price(financial_data), _finite_positive(market.get('dividend_yield'))
        return price * dividend_yield if price and dividend_yield else None

    def _estimate_dividend_growth(self, financial_data: Dict[str, Any], roe: Optional[float] = None,
                                  payout: Optional[float] = None) -> float:
        """
        Near-term dividend growth: median of dividend-per-share CAGR, EPS CAGR
        and sustainable growth (ROE x retention), bounded to [-2 %, 12 %].
        """
        income, _, cash_flow = self._statements(financial_data)
        candidates = []
        dividends = get_series(cash_flow, 'dividends_paid', ascending=True).abs()
        shares = get_series(income, 'diluted_shares', ascending=True)
        if not dividends.empty and not shares.empty:
            dps = (dividends / shares.reindex(dividends.index)).dropna()
            candidates.append(cagr(dps.tail(5)))
        candidates.append(cagr(get_series(income, 'diluted_eps', ascending=True).tail(5)))
        if roe is not None and payout is not None:
            candidates.append(roe * (1 - payout))
        candidates = [c for c in candidates if c is not None]
        return clamp(float(np.median(candidates)), -0.02, 0.12) if candidates else 0.03

    def dividend_discount_valuation(self, ticker: str, financial_data: Dict[str, Any],
                                    sector: str = None, discount_rate: Optional[float] = None,
                                    terminal_growth: Optional[float] = None,
                                    forecast_years: Optional[int] = None,
                                    growth_rate: Optional[float] = None,
                                    cost_of_equity: Optional[float] = None,
                                    payout_ratio: Optional[float] = None,
                                    stable_roe: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Two-stage dividend discount model (Damodaran).

        Dividends grow at the near-term rate fading to ``terminal_growth``;
        in the stable phase the payout rises to ``1 - g / ROE`` (never below
        the current payout) and the terminal value is a Gordon growth value at
        the cost of equity, which must exceed terminal growth.
        """
        label = 'Dividend Discount Model'
        try:
            params = self._get_dcf_parameters(sector)
            income, _, _ = self._statements(financial_data)
            shares = self._shares_outstanding(financial_data)
            net_income = get_latest(income, 'net_income')
            eps = net_income / shares if net_income is not None and shares else None

            dps = self._dividends_per_share(financial_data)
            if payout_ratio is not None and eps and eps > 0:
                dps = eps * clamp(payout_ratio, 0.0, 1.0)
            if not dps or dps <= 0:
                raise ValueError("Company does not pay dividends or dividend data is unavailable")
            payout = clamp(dps / eps, 0.0, 1.0) if eps and eps > 0 else None

            roe = self._return_on_equity(financial_data)
            ke = cost_of_equity if cost_of_equity is not None else self._cost_of_equity(financial_data, discount_rate)
            g_terminal = terminal_growth if terminal_growth is not None else params['terminal_growth_rate']
            if ke <= g_terminal:
                raise ValueError(f"Cost of equity ({ke:.2%}) must exceed terminal growth ({g_terminal:.2%})")
            years = int(forecast_years or 5)
            g_near = growth_rate if growth_rate is not None else self._estimate_dividend_growth(
                financial_data, clamp(roe, 0.0, 0.4) if roe is not None else None, payout)

            rates = growth_path(g_near, g_terminal, years)
            dividends = compound(dps, rates)
            stable_payout = payout
            stable = stable_roe if stable_roe is not None else (clamp(roe, ke, 0.30) if roe is not None else None)
            if eps and eps > 0 and stable and stable > g_terminal:
                stable_payout = max(payout or 0.0, clamp(1 - g_terminal / stable, 0.0, 1.0))
                terminal_eps = compound(eps, rates)[-1] * (1 + g_terminal)
                terminal_dividend = terminal_eps * stable_payout
            else:
                terminal_dividend = dividends[-1] * (1 + g_terminal)
            dcf = self._discount_cash_flows(dividends, ke, g_terminal, growth_rates=rates,
                                            terminal_cash_flow=terminal_dividend)
            value = dcf['enterprise_value']
            return {
                'company': ticker, 'method': label,
                'value_per_share': value,
                'total_value': value * shares if shares else None,
                'current_dividend': dps, 'payout_ratio': payout, 'stable_payout_ratio': stable_payout,
                'return_on_equity': roe, 'dividend_growth': g_near, 'terminal_growth': g_terminal,
                'cost_of_equity': ke, 'discount_rate': ke, 'forecast_years': years,
                'forecast_dividends': dcf['forecast_fcf'], 'projections': dcf['projections'],
                'terminal_value': dcf['terminal_value'], 'present_value_dividends': dcf['present_value_fcf'],
                'present_value_terminal': dcf['present_value_terminal'],
            }
        except Exception as e:
            return self._method_error(label, e, ticker)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Load statements, company info and recent prices into the canonical bundle."""
        try:
            income = self.data_loader.get_financial_statements(ticker, 'income', 'annual')
            balance = self.data_loader.get_financial_statements(ticker, 'balance', 'annual')
            cash_flow = self.data_loader.get_financial_statements(ticker, 'cash', 'annual')
            company_info = self.data_loader.get_company_info(ticker) or {}
        except Exception as e:
            logger.error(f"Error loading financial data for {ticker}: {e}")
            raise ValueError(f"Failed to load required financial data for {ticker}: {e}") from e
        try:
            start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
            prices = self.data_loader.get_historical_prices(ticker, start_date=start)
        except Exception as e:  # price is optional: company_info carries a fallback
            logger.warning(f"Could not load recent prices for {ticker}: {e}")
            prices = None
        return build_financial_data(income, balance, cash_flow, company_info, prices)
