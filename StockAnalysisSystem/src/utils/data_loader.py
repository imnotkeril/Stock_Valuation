"""
Market and fundamental data access with on-disk caching.

Providers:
    * yfinance (default) — prices, company info, financial statements, search
    * Alpha Vantage (optional, needs ``ALPHA_VANTAGE_API_KEY``) — prices,
      company overview and symbol search as fallbacks
    * FRED via pandas-datareader (optional) — macro indicators

Every public method returns an empty DataFrame / dict / list instead of
raising when data is unavailable. Fresh data is cached in ``CACHE_DIR``;
when a provider fails, an expired cache entry is served rather than nothing.
With ``STOCK_ANALYSIS_OFFLINE=1`` no network calls are made at all.

Normalised outputs:
    * prices: tz-naive ``DatetimeIndex`` named ``Date``; columns ``Open``,
      ``High``, ``Low``, ``Close``, ``Adj Close``, ``Volume`` (+ ``Dividends``,
      ``Stock Splits`` when available), sorted ascending.
    * company info: see :meth:`DataLoader.get_company_info`.
    * statements: yfinance layout (line items × period end dates, newest first).
"""

import json
import logging
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from StockAnalysisSystem.src import config
from StockAnalysisSystem.src.config import API_KEYS, CACHE_EXPIRY_DAYS, normalize_sector
from StockAnalysisSystem.src.utils.financial_data import normalize_dividend_yield, to_float

logger = logging.getLogger('data_loader')

PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
STATEMENT_TYPES = ('income', 'balance', 'cash')
STATEMENT_ATTRIBUTES = {
    ('income', 'annual'): 'income_stmt',
    ('income', 'quarterly'): 'quarterly_income_stmt',
    ('balance', 'annual'): 'balance_sheet',
    ('balance', 'quarterly'): 'quarterly_balance_sheet',
    ('cash', 'annual'): 'cashflow',
    ('cash', 'quarterly'): 'quarterly_cashflow',
}
REQUEST_TIMEOUT = 15  # seconds
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Cache lifetimes in days
INFO_CACHE_DAYS = 7
STATEMENT_CACHE_DAYS = 7
SEARCH_CACHE_DAYS = 3


def provider_symbol(ticker: str) -> str:
    """Ticker as expected by Yahoo / Alpha Vantage (``BRK.B`` -> ``BRK-B``)."""
    return ticker.strip().upper().replace('.', '-')


def display_symbol(symbol: str) -> str:
    """Ticker as shown to users (``BRK-B`` -> ``BRK.B`` for share classes)."""
    parts = symbol.split('-')
    if len(parts) == 2 and len(parts[1]) == 1:
        return symbol.replace('-', '.')
    return symbol


def normalize_price_frame(data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Bring a provider price frame into the normalised layout (see module docstring)."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    data = data.copy()

    # yf.download returns (field, ticker) MultiIndex columns even for one ticker
    if isinstance(data.columns, pd.MultiIndex):
        level = 0 if 'Close' in data.columns.get_level_values(0) else -1
        data.columns = data.columns.get_level_values(level)
    data = data.loc[:, ~pd.Index(data.columns).duplicated()]

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
    if data.index.tz is not None:
        data.index = data.index.tz_convert('America/New_York').tz_localize(None)
    data = data[~data.index.isna()]
    data.index = data.index.normalize()
    data.index.name = 'Date'

    rename = {c: c.title() for c in data.columns if c.lower() in {'open', 'high', 'low', 'close', 'volume'}}
    data = data.rename(columns=rename)
    if 'Close' not in data.columns:
        return pd.DataFrame()
    if 'Adj Close' not in data.columns:
        data['Adj Close'] = data['Close']
    for column in PRICE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data = data[~data.index.duplicated(keep='last')].sort_index()
    data = data.dropna(subset=['Close'])
    extra = [c for c in ('Dividends', 'Stock Splits') if c in data.columns]
    return data[PRICE_COLUMNS + extra]


class DataLoader:
    """Fetches financial data from external providers with caching."""

    def __init__(self, cache_dir: Optional[str] = None, cache_expiry_days: int = CACHE_EXPIRY_DAYS,
                 offline: Optional[bool] = None):
        """
        Args:
            cache_dir: Cache directory (defaults to ``config.CACHE_DIR``).
            cache_expiry_days: Expiry for price data caches.
            offline: Disable all network access (defaults to ``config.OFFLINE_MODE``).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(config.CACHE_DIR)
        self.cache_expiry_days = cache_expiry_days
        self.offline = config.OFFLINE_MODE if offline is None else offline
        self.api_keys = API_KEYS
        self.api_call_counts = {'yfinance': 0, 'alpha_vantage': 0}
        self.api_limits = {'alpha_vantage': 25}  # free tier daily limit
        self.providers: Dict[str, Callable[..., pd.DataFrame]] = {
            'yfinance': self._fetch_yfinance,
            'alpha_vantage': self._fetch_alpha_vantage,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"DataLoader initialised (cache: {self.cache_dir}, offline: {self.offline})")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, name: str) -> Path:
        safe = re.sub(r'[^A-Za-z0-9_.=-]', '_', name)
        return self.cache_dir / safe

    @staticmethod
    def _is_fresh(path: Path, max_age_days: float) -> bool:
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=max_age_days)

    def _read_cache(self, path: Path, max_age_days: Optional[float]) -> Any:
        """Read a cache file; ``max_age_days=None`` accepts any age. Returns None on miss."""
        if not path.exists() or (max_age_days is not None and not self._is_fresh(path, max_age_days)):
            return None
        try:
            if path.suffix == '.json':
                return json.loads(path.read_text())
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Ignoring unreadable cache file {path.name}: {e}")
            return None

    def _write_cache(self, path: Path, value: Any) -> None:
        try:
            if path.suffix == '.json':
                path.write_text(json.dumps(value, default=str))
            else:
                with open(path, 'wb') as f:
                    pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Could not write cache file {path.name}: {e}")

    def _cached(self, path: Path, max_age_days: float, fetch: Callable[[], Any], force_refresh: bool,
                is_valid: Callable[[Any], bool]) -> Any:
        """Fresh cache -> fetch -> stale cache, in that order."""
        if not force_refresh:
            cached = self._read_cache(path, max_age_days)
            if cached is not None and is_valid(cached):
                return cached
        if not self.offline:
            try:
                value = fetch()
                if is_valid(value):
                    self._write_cache(path, value)
                    return value
            except Exception as e:
                logger.error(f"Fetching {path.stem} failed: {e}")
        stale = self._read_cache(path, None)
        if stale is not None and is_valid(stale):
            logger.info(f"Serving stale cache for {path.stem}")
            return stale
        return None

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def get_historical_prices(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                              provider: str = 'yfinance', interval: str = '1d',
                              force_refresh: bool = False) -> pd.DataFrame:
        """
        Historical OHLCV prices.

        Args:
            ticker: Stock/ETF ticker (``BRK.B`` style accepted).
            start_date: ``YYYY-MM-DD`` (default: 5 years ago).
            end_date: ``YYYY-MM-DD`` inclusive (default: today).
            provider: ``yfinance`` or ``alpha_vantage``; the other is used as fallback.
            interval: ``1d``, ``1wk`` or ``1mo``.
            force_refresh: Ignore fresh cache entries.
        """
        if provider not in self.providers:
            raise ValueError(f"Unsupported data provider: {provider}")
        ticker = ticker.strip().upper()
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = start_date or (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        order = [provider] + [p for p in self.providers if p != provider]
        for name in order:
            path = self._cache_path(f"{ticker}_{start_date}_{end_date}_{interval}_{name}.pkl")
            data = self._cached(
                path, self.cache_expiry_days,
                lambda: normalize_price_frame(self.providers[name](ticker, start_date, end_date, interval)),
                force_refresh, lambda df: isinstance(df, pd.DataFrame) and not df.empty,
            )
            if data is not None:
                return normalize_price_frame(data)
        return pd.DataFrame()

    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        import yfinance as yf

        self.api_call_counts['yfinance'] += 1
        # yfinance treats ``end`` as exclusive
        end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        data = yf.Ticker(provider_symbol(ticker)).history(start=start_date, end=end_exclusive, interval=interval,
                                                          auto_adjust=False)
        if data is None or data.empty:
            data = yf.download(provider_symbol(ticker), start=start_date, end=end_exclusive, interval=interval,
                               progress=False, auto_adjust=False, threads=False)
        return data if data is not None else pd.DataFrame()

    def _fetch_alpha_vantage(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        functions = {'1d': 'TIME_SERIES_DAILY', '1wk': 'TIME_SERIES_WEEKLY', '1mo': 'TIME_SERIES_MONTHLY'}
        data = self._alpha_vantage_request({"function": functions.get(interval, 'TIME_SERIES_DAILY'),
                                            "symbol": provider_symbol(ticker), "outputsize": "full"})
        series_key = next((k for k in (data or {}) if k.startswith("Time Series")), None)
        if not series_key:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(data[series_key], orient='index')
        df = df.rename(columns=lambda c: {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                                          'adjusted close': 'Adj Close', 'volume': 'Volume'}.get(
            re.sub(r'^\d+\.\s*', '', c), c))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

    def _alpha_vantage_request(self, params: Dict[str, str]) -> Optional[Dict]:
        if not self.api_keys.get('alpha_vantage'):
            return None
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached for this session")
            return None
        self.api_call_counts['alpha_vantage'] += 1
        response = requests.get(ALPHA_VANTAGE_URL, params={**params, "apikey": self.api_keys['alpha_vantage']},
                                timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "Error Message" in data or "Note" in data or "Information" in data:
            logger.warning(f"Alpha Vantage returned no data for {params.get('symbol') or params.get('keywords')}")
            return None
        return data

    def get_batch_data(self, tickers: List[str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, provider: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """Historical prices for several tickers: ``{ticker: DataFrame}`` (missing tickers omitted)."""
        results: Dict[str, pd.DataFrame] = {}
        for ticker in dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()):
            data = self.get_historical_prices(ticker, start_date, end_date, provider)
            if not data.empty:
                results[ticker] = data
        return results

    # ------------------------------------------------------------------
    # Company information
    # ------------------------------------------------------------------
    def get_company_info(self, ticker: str, provider: str = 'yfinance') -> Dict:
        """
        Normalised company information.

        Keys: ``symbol``, ``name``, ``sector`` (canonical, see
        ``config.normalize_sector``), ``sector_raw``, ``industry``, ``country``,
        ``exchange``, ``currency``, ``market_cap``, ``current_price``,
        ``shares_outstanding``, ``pe_ratio``, ``forward_pe``, ``pb_ratio``,
        ``ps_ratio``, ``ev_ebitda``, ``enterprise_value``, ``dividend_yield``
        (fraction), ``beta``, ``description``, ``website``, ``employees``,
        ``type``, ``return_on_equity``, ``return_on_assets``, ``profit_margins``,
        ``operating_margins``, ``gross_margins``, ``fifty_two_week_high``,
        ``fifty_two_week_low``, ``last_updated``. Empty dict when unknown.
        """
        ticker = ticker.strip().upper()
        fetchers = {'yfinance': self._get_yfinance_company_info,
                    'alpha_vantage': self._get_alpha_vantage_company_info}
        if provider not in fetchers:
            raise ValueError(f"Unsupported provider for company info: {provider}")
        order = [provider] + [p for p in fetchers if p != provider]
        for name in order:
            path = self._cache_path(f"{ticker}_info_{name}.json")
            info = self._cached(path, INFO_CACHE_DAYS, lambda: fetchers[name](ticker), False,
                                lambda d: isinstance(d, dict) and bool(d.get('name')))
            if info:
                return self._finalize_info(info, ticker)
        return {}

    @staticmethod
    def _finalize_info(info: Dict, ticker: str) -> Dict:
        """Apply normalisations that also need to hold for older cache entries."""
        info = dict(info)
        info['symbol'] = info.get('symbol') or ticker
        raw_sector = info.get('sector_raw', info.get('sector'))
        info['sector_raw'] = raw_sector
        info['sector'] = normalize_sector(raw_sector)
        info['dividend_yield'] = normalize_dividend_yield(info.get('dividend_yield'))
        for key in ('industry', 'country', 'exchange', 'description', 'website'):
            if info.get(key) in (None, '', 'None'):
                info[key] = 'N/A'
        info.setdefault('currency', 'USD')
        return info

    def _get_yfinance_company_info(self, ticker: str) -> Dict:
        import yfinance as yf

        self.api_call_counts['yfinance'] += 1
        info = yf.Ticker(provider_symbol(ticker)).info or {}
        if not info or not (info.get('longName') or info.get('shortName')):
            return {}
        quote_type = str(info.get('quoteType', '')).lower()
        normalized = {
            'symbol': ticker,
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector') or ('N/A' if quote_type != 'etf' else 'ETF'),
            'industry': info.get('industry') or 'N/A',
            'country': info.get('country') or 'N/A',
            'exchange': info.get('exchange') or 'N/A',
            'currency': info.get('currency') or 'USD',
            'market_cap': to_float(info.get('marketCap')),
            'current_price': to_float(info.get('currentPrice') or info.get('regularMarketPrice')
                                      or info.get('previousClose')),
            'shares_outstanding': to_float(info.get('sharesOutstanding')),
            'pe_ratio': to_float(info.get('trailingPE')),
            'forward_pe': to_float(info.get('forwardPE')),
            'pb_ratio': to_float(info.get('priceToBook')),
            'ps_ratio': to_float(info.get('priceToSalesTrailing12Months')),
            'ev_ebitda': to_float(info.get('enterpriseToEbitda')),
            'enterprise_value': to_float(info.get('enterpriseValue')),
            'dividend_yield': to_float(info.get('dividendYield')),
            'beta': to_float(info.get('beta')),
            'description': info.get('longBusinessSummary') or 'N/A',
            'website': info.get('website') or 'N/A',
            'employees': info.get('fullTimeEmployees'),
            'type': self._determine_asset_type(info),
            'fifty_two_week_high': to_float(info.get('fiftyTwoWeekHigh')),
            'fifty_two_week_low': to_float(info.get('fiftyTwoWeekLow')),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
        }
        for metric in ('returnOnEquity', 'returnOnAssets', 'profitMargins', 'operatingMargins', 'grossMargins'):
            if info.get(metric) is not None:
                normalized[self._camel_to_snake(metric)] = to_float(info[metric])
        return normalized

    def _get_alpha_vantage_company_info(self, ticker: str) -> Dict:
        data = self._alpha_vantage_request({"function": "OVERVIEW", "symbol": provider_symbol(ticker)})
        if not data or not data.get('Name'):
            return {}
        number = lambda key: to_float(data.get(key)) if data.get(key) not in (None, 'None', '-') else None  # noqa: E731
        return {
            'symbol': ticker,
            'name': data.get('Name', ticker),
            'sector': data.get('Sector', 'N/A'),
            'industry': data.get('Industry', 'N/A'),
            'country': data.get('Country', 'N/A'),
            'exchange': data.get('Exchange', 'N/A'),
            'currency': data.get('Currency', 'USD'),
            'market_cap': number('MarketCapitalization'),
            'shares_outstanding': number('SharesOutstanding'),
            'pe_ratio': number('PERatio'),
            'forward_pe': number('ForwardPE'),
            'pb_ratio': number('PriceToBookRatio'),
            'ps_ratio': number('PriceToSalesRatioTTM'),
            'ev_ebitda': number('EVToEBITDA'),
            'dividend_yield': number('DividendYield'),
            'beta': number('Beta'),
            'description': data.get('Description', 'N/A'),
            'type': 'Stock',
            'return_on_equity': number('ReturnOnEquityTTM'),
            'return_on_assets': number('ReturnOnAssetsTTM'),
            'profit_margins': number('ProfitMargin'),
            'operating_margins': number('OperatingMarginTTM'),
            'fifty_two_week_high': number('52WeekHigh'),
            'fifty_two_week_low': number('52WeekLow'),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
        }

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------
    def get_financial_statements(self, ticker: str, statement_type: str = 'income',
                                 period: str = 'annual', force_refresh: bool = False) -> pd.DataFrame:
        """
        Financial statement in yfinance layout.

        Args:
            ticker: Stock ticker.
            statement_type: ``income``, ``balance`` or ``cash``.
            period: ``annual`` or ``quarterly``.
        """
        if (statement_type, period) not in STATEMENT_ATTRIBUTES:
            raise ValueError(f"Unsupported statement: {statement_type}/{period}")
        ticker = ticker.strip().upper()
        path = self._cache_path(f"{ticker}_{statement_type}_{period}.pkl")
        data = self._cached(path, STATEMENT_CACHE_DAYS,
                            lambda: self._fetch_statement(ticker, statement_type, period), force_refresh,
                            lambda df: isinstance(df, pd.DataFrame) and not df.empty)
        return data if data is not None else pd.DataFrame()

    def _fetch_statement(self, ticker: str, statement_type: str, period: str) -> pd.DataFrame:
        import yfinance as yf

        self.api_call_counts['yfinance'] += 1
        df = getattr(yf.Ticker(provider_symbol(ticker)), STATEMENT_ATTRIBUTES[(statement_type, period)])
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = pd.to_datetime(df.columns, errors='coerce')
        df = df.loc[:, ~df.columns.isna()]
        return df[sorted(df.columns, reverse=True)].apply(pd.to_numeric, errors='coerce')

    def get_financial_ratios(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Headline ratios from company info, grouped like
        ``{"valuation": {...}, "profitability": {...}}``. For full ratio
        analysis use ``models.ratio_analysis.FinancialRatioAnalyzer``.
        """
        info = self.get_company_info(ticker)
        if not info:
            return {}
        return {
            "valuation": {
                "pe_ratio": info.get("pe_ratio"), "forward_pe": info.get("forward_pe"),
                "pb_ratio": info.get("pb_ratio"), "ps_ratio": info.get("ps_ratio"),
                "ev_ebitda": info.get("ev_ebitda"), "dividend_yield": info.get("dividend_yield"),
            },
            "profitability": {
                "return_on_equity": info.get("return_on_equity"),
                "return_on_assets": info.get("return_on_assets"),
                "profit_margins": info.get("profit_margins"),
                "operating_margins": info.get("operating_margins"),
                "gross_margins": info.get("gross_margins"),
            },
            "last_updated": info.get("last_updated"),
        }

    # ------------------------------------------------------------------
    # Sector data
    # ------------------------------------------------------------------
    def get_sector_data(self, sector: str, force_refresh: bool = False) -> pd.DataFrame:
        """Three years of daily prices of the sector's benchmark ETF (empty for unknown sectors)."""
        from StockAnalysisSystem.src.industry.universe import get_sector_etf

        etf = get_sector_etf(sector)
        if not etf:
            logger.info(f"No sector ETF defined for {sector}")
            return pd.DataFrame()
        start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
        return self.get_historical_prices(etf, start_date=start_date, force_refresh=force_refresh)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_tickers(self, query: str, limit: int = 10, provider: str = 'alpha_vantage') -> List[Dict]:
        """
        Search companies by ticker or name.

        The curated universe is searched first (instant, offline); provider
        results (Yahoo search, then Alpha Vantage when a key is configured)
        are appended and de-duplicated. Each result has ``symbol``, ``name``,
        ``type``, ``region``, ``currency`` and optionally ``exchange``/``sector``.
        """
        from StockAnalysisSystem.src.industry.universe import search_universe

        query = (query or '').strip()
        if not query:
            return []
        results = [{"symbol": r["symbol"], "name": r["name"], "type": "Equity", "region": "United States",
                    "currency": "USD", "sector": r["sector"]} for r in search_universe(query, limit)]

        if len(results) < limit and not self.offline:
            order = ['yfinance', 'alpha_vantage'] if provider != 'alpha_vantage' or not self.api_keys.get(
                'alpha_vantage') else ['alpha_vantage', 'yfinance']
            searchers = {'yfinance': self._search_yahoo, 'alpha_vantage': self._search_alpha_vantage}
            for name in order:
                path = self._cache_path(f"search_{query.lower()}_{name}.json")
                found = self._cached(path, SEARCH_CACHE_DAYS, lambda: searchers[name](query, limit), False,
                                     lambda r: isinstance(r, list) and bool(r)) or []
                results.extend(found)
                if found:
                    break

        unique: Dict[str, Dict] = {}
        for result in results:
            symbol = str(result.get('symbol', '')).upper()
            if symbol and symbol not in unique:
                unique[symbol] = result
        return list(unique.values())[:limit]

    def _search_yahoo(self, query: str, limit: int = 10) -> List[Dict]:
        import yfinance as yf

        self.api_call_counts['yfinance'] += 1
        quotes = yf.Search(query, max_results=limit, news_count=0).quotes or []
        return [{
            'symbol': display_symbol(q.get('symbol', '')),
            'original_symbol': q.get('symbol', ''),
            'name': q.get('longname') or q.get('shortname') or q.get('symbol', ''),
            'type': q.get('quoteType', ''),
            'region': q.get('region', ''),
            'currency': q.get('currency', ''),
            'exchange': q.get('exchDisp') or q.get('exchange', ''),
        } for q in quotes if q.get('symbol') and q.get('quoteType') in {'EQUITY', 'ETF', None}]

    def _search_alpha_vantage(self, query: str, limit: int = 10) -> List[Dict]:
        data = self._alpha_vantage_request({"function": "SYMBOL_SEARCH", "keywords": query})
        return [{
            'symbol': display_symbol(m.get('1. symbol', '')),
            'original_symbol': m.get('1. symbol', ''),
            'name': m.get('2. name', ''),
            'type': m.get('3. type', ''),
            'region': m.get('4. region', ''),
            'currency': m.get('8. currency', 'USD'),
        } for m in (data or {}).get('bestMatches', [])[:limit] if m.get('1. symbol')]

    # ------------------------------------------------------------------
    # Macro data
    # ------------------------------------------------------------------
    INDICATOR_CODES = {
        'INFLATION': 'CPIAUCSL',
        'GDP': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'INTEREST_RATE': 'FEDFUNDS',
        'TREASURY_10Y': 'DGS10',
        'RETAIL_SALES': 'RSXFS',
        'INDUSTRIAL_PRODUCTION': 'INDPRO',
        'HOUSE_PRICE_INDEX': 'CSUSHPISA',
        'CONSUMER_SENTIMENT': 'UMCSENT',
    }

    def get_macro_indicators(self, indicators: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """FRED macro series ``{indicator: DataFrame}`` (empty frames when unavailable)."""
        results = {indicator: pd.DataFrame() for indicator in indicators}
        if self.offline:
            return results
        try:
            import pandas_datareader.data as web
        except ImportError:
            logger.info("pandas-datareader not installed; macro indicators unavailable")
            return results
        start = pd.Timestamp(start_date) if start_date else pd.Timestamp.now() - pd.DateOffset(years=5)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        for indicator in indicators:
            try:
                results[indicator] = web.DataReader(self.INDICATOR_CODES.get(indicator, indicator), 'fred', start, end)
            except Exception as e:
                logger.error(f"Error loading macro indicator {indicator}: {e}")
        return results

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_asset_type(info: Dict) -> str:
        quote_type = str(info.get('quoteType', '')).lower()
        mapping = {'equity': 'Stock', 'stock': 'Stock', 'etf': 'ETF', 'index': 'Index',
                   'cryptocurrency': 'Crypto', 'mutualfund': 'Mutual Fund'}
        if quote_type:
            return mapping.get(quote_type, quote_type.capitalize())
        return 'Stock' if info.get('industry') else 'Unknown'

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @staticmethod
    def _safe_convert(value: Any, convert_type) -> Optional[Any]:
        try:
            return None if value is None else convert_type(value)
        except (ValueError, TypeError):
            return None

    def clear_cache(self, tickers: Optional[List[str]] = None) -> int:
        """Delete cache files (all, or only those of ``tickers``). Returns the number removed."""
        patterns = ['*.*'] if tickers is None else [f"{t.strip().upper()}_*.*" for t in tickers]
        removed = 0
        for pattern in patterns:
            for file in self.cache_dir.glob(pattern):
                if file.name == '.gitkeep':
                    continue
                try:
                    file.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning(f"Failed to delete cache file {file}: {e}")
        return removed
