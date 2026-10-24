import os
import sys
import time
import logging
import pickle
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Union, Any


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from StockAnalysisSystem.src.config import CACHE_DIR, CACHE_EXPIRY_DAYS, API_KEYS
from StockAnalysisSystem.src.config import RISK_FREE_RATE
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_loader')


class DataLoader:
    """
    Main class for fetching financial data from various sources.
    Supports data caching for performance optimization.
    """

    def __init__(self, cache_dir: str = CACHE_DIR, cache_expiry_days: int = CACHE_EXPIRY_DAYS):
        """
        Initialize data loader with cache settings

        Args:
            cache_dir: Directory for data caching
            cache_expiry_days: Cache expiry period in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_days = cache_expiry_days
        self.api_call_counts = {'yfinance': 0, 'alpha_vantage': 0}
        self.api_limits = {'alpha_vantage': 500}  # API call limits

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load API keys
        self.api_keys = API_KEYS

        # Supported data providers
        self.providers = {
            'yfinance': self._fetch_yfinance,
            'alpha_vantage': self._fetch_alpha_vantage
        }

        logger.info(f"DataLoader initialized with cache in {self.cache_dir}")

    def get_historical_prices(
            self,
            ticker: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            provider: str = 'yfinance',
            interval: str = '1d',
            force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical price data for a given ticker

        Args:
            ticker: Stock/ETF ticker
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Data provider ('yfinance', 'alpha_vantage')
            interval: Data interval ('1d', '1wk', '1mo')
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with historical prices
        """
        # Check if ticker contains a dot and create corrected version for API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(f"Using corrected ticker {corrected_ticker} for {original_ticker} request")

        # Use original ticker for cache key
        cache_key = original_ticker

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            # Default to 5 years of historical data
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        # Check cache expiry
        cache_file = self.cache_dir / f"{cache_key}_{start_date}_{end_date}_{interval}_{provider}.pkl"

        if not force_refresh and cache_file.exists():
            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < self.cache_expiry_days:
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {original_ticker} data from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache for {original_ticker}: {e}")

        # If cache is missing or expired, fetch new data
        if provider in self.providers:
            try:
                # Get data from provider using corrected ticker
                data = self.providers[provider](corrected_ticker, start_date, end_date, interval)

                # Save to cache if data was retrieved
                if data is not None and not data.empty:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"Saved {original_ticker} data to cache")

                return data
            except Exception as e:
                logger.error(f"Error getting data for {original_ticker} from {provider}: {e}")
                # Try fallback provider if current one fails
                fallback_provider = next((p for p in self.providers.keys() if p != provider), None)
                if fallback_provider:
                    logger.info(f"Trying fallback provider: {fallback_provider}")
                    return self.get_historical_prices(
                        ticker, start_date, end_date, fallback_provider, interval, force_refresh
                    )
        else:
            raise ValueError(f"Unsupported data provider: {provider}")

        # If data retrieval failed
        return pd.DataFrame()

    def get_company_info(self, ticker: str, provider: str = 'yfinance') -> Dict:
        """
        Get company information

        Args:
            ticker: Stock/ETF ticker
            provider: Data provider

        Returns:
            Dictionary with company information
        """
        cache_file = self.cache_dir / f"{ticker}_info_{provider}.json"

        # Check cache (company info cache expires after 7 days)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Company info updates less frequently
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} info from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load info cache for {ticker}: {e}")

        info = {}

        try:
            if provider == 'yfinance':
                info = self._get_yfinance_company_info(ticker)
            elif provider == 'alpha_vantage':
                info = self._get_alpha_vantage_company_info(ticker)
            else:
                raise ValueError(f"Unsupported provider for company info: {provider}")

            # Save to cache
            if info:
                with open(cache_file, 'w') as f:
                    json.dump(info, f)
                logger.info(f"Saved {ticker} info to cache")

            return info
        except Exception as e:
            logger.error(f"Error getting information about {ticker}: {e}")

            # Try alternative provider
            if provider != 'yfinance':
                logger.info("Trying to get info via yfinance")
                return self.get_company_info(ticker, 'yfinance')

            return {}

    def get_financial_statements(self, ticker: str, statement_type: str = 'income',
                                 period: str = 'annual', force_refresh: bool = False) -> pd.DataFrame:
        """
        Get financial statements for a company

        Args:
            ticker: Stock ticker
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with financial statement data
        """
        cache_file = self.cache_dir / f"{ticker}_{statement_type}_{period}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Financial statements update less frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} {statement_type} statement from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load financial statement cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get data
            ticker_obj = yf.Ticker(ticker)

            if statement_type == 'income':
                if period == 'annual':
                    df = ticker_obj.income_stmt
                else:
                    df = ticker_obj.quarterly_income_stmt
            elif statement_type == 'balance':
                if period == 'annual':
                    df = ticker_obj.balance_sheet
                else:
                    df = ticker_obj.quarterly_balance_sheet
            elif statement_type == 'cash':
                if period == 'annual':
                    df = ticker_obj.cashflow
                else:
                    df = ticker_obj.quarterly_cashflow
            else:
                raise ValueError(f"Unsupported statement type: {statement_type}")

            # Save to cache if successful
            if df is not None and not df.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                logger.info(f"Saved {ticker} {statement_type} statement to cache")

            return df
        except Exception as e:
            logger.error(f"Error fetching financial statements for {ticker}: {e}")
            return pd.DataFrame()

    def get_financial_ratios(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Get financial ratios for a company

        Args:
            ticker: Stock ticker
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary of financial ratios
        """
        cache_file = self.cache_dir / f"{ticker}_ratios.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Ratios update less frequently
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} ratios from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load ratios cache for {ticker}: {e}")

        try:
            # Try to get company info which has some ratios
            info = self.get_company_info(ticker)

            # Basic ratios from company info
            ratios = {
                "valuation": {
                    "pe_ratio": info.get("pe_ratio"),
                    "forward_pe": info.get("forward_pe"),
                    "pb_ratio": info.get("pb_ratio"),
                    "dividend_yield": info.get("dividend_yield")
                },
                "profitability": {
                    "return_on_equity": info.get("return_on_equity"),
                    "return_on_assets": info.get("return_on_assets"),
                    "profit_margins": info.get("profit_margins"),
                    "operating_margins": info.get("operating_margins"),
                    "gross_margins": info.get("gross_margins")
                },
                "last_updated": datetime.now().strftime('%Y-%m-%d')
            }

            # Try to calculate additional ratios from financial statements
            try:
                # Get financial statements
                income_statement = self.get_financial_statements(ticker, 'income')
                balance_sheet = self.get_financial_statements(ticker, 'balance')
                cash_flow = self.get_financial_statements(ticker, 'cash')

                if not income_statement.empty and not balance_sheet.empty:
                    # Get latest data
                    latest_income = income_statement.iloc[:, 0]
                    latest_balance = balance_sheet.iloc[:, 0]

                    # Calculate additional ratios
                    if 'Total Revenue' in latest_income and 'Total Assets' in latest_balance:
                        ratios["efficiency"] = {
                            "asset_turnover": latest_income['Total Revenue'] / latest_balance['Total Assets']
                        }

                    if 'Total Current Assets' in latest_balance and 'Total Current Liabilities' in latest_balance:
                        ratios["liquidity"] = {
                            "current_ratio": latest_balance['Total Current Assets'] / latest_balance[
                                'Total Current Liabilities']
                        }

                        # Quick ratio if we have inventory data
                        if 'Inventory' in latest_balance:
                            quick_ratio = (latest_balance['Total Current Assets'] - latest_balance['Inventory']) / \
                                          latest_balance['Total Current Liabilities']
                            ratios["liquidity"]["quick_ratio"] = quick_ratio

                    if 'Total Debt' in latest_balance and 'Total Stockholder Equity' in latest_balance:
                        ratios["leverage"] = {
                            "debt_to_equity": latest_balance['Total Debt'] / latest_balance['Total Stockholder Equity']
                        }
            except Exception as e:
                logger.warning(f"Error calculating additional ratios for {ticker}: {e}")

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(ratios, f)
            logger.info(f"Saved {ticker} ratios to cache")

            return ratios
        except Exception as e:
            logger.error(f"Error getting financial ratios for {ticker}: {e}")
            return {}

    def get_sector_data(self, sector: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get aggregated data for a specific sector

        Args:
            sector: Market sector name
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with sector data
        """
        cache_file = self.cache_dir / f"sector_{sector.lower().replace(' ', '_')}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Sector data updates less frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {sector} sector data from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load sector data cache for {sector}: {e}")

        try:
            # Define sector ETFs to use as proxies
            sector_etfs = {
                "Information Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
                "Industrials": "XLI",
                "Communication Services": "XLC"
            }

            if sector not in sector_etfs:
                logger.warning(f"No sector ETF defined for {sector}")
                return pd.DataFrame()

            # Get historical prices for the sector ETF
            etf_ticker = sector_etfs[sector]
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

            etf_data = self.get_historical_prices(etf_ticker, start_date=start_date)

            # Calculate sector performance metrics
            if not etf_data.empty:
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(etf_data, f)
                logger.info(f"Saved {sector} sector data to cache")

                return etf_data

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting sector data for {sector}: {e}")
            return pd.DataFrame()

    def search_tickers(self, query: str, limit: int = 10, provider: str = 'alpha_vantage') -> List[Dict]:
        """
        Search for tickers based on a query

        Args:
            query: Search query
            limit: Maximum number of results
            provider: Data provider

        Returns:
            List of dictionaries with ticker information
        """
        cache_key = f"search_{query.lower()}_{provider}.json"
        cache_file = self.cache_dir / cache_key

        # Check cache (search results expire after 3 days)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 3:
                try:
                    with open(cache_file, 'r') as f:
                        results = json.load(f)
                        logger.info(f"Loading search results for '{query}' from cache")
                        return results[:limit]
                except Exception as e:
                    logger.warning(f"Failed to load search cache: {e}")

        results = []

        try:
            if provider == 'alpha_vantage':
                results = self._search_alpha_vantage(query, limit)
            elif provider == 'yfinance':
                # yfinance doesn't have a direct search API, use alternative
                results = self._search_alternative(query, limit)
            else:
                raise ValueError(f"Unsupported provider for search: {provider}")

            # Save to cache
            if results:
                with open(cache_file, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Saved search results for '{query}' to cache")

            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching for tickers with query '{query}': {e}")

            # Try alternative provider
            if provider != 'alpha_vantage' and self.api_keys['alpha_vantage']:
                return self.search_tickers(query, limit, 'alpha_vantage')

            return []

    def get_macro_indicators(self, indicators: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get macroeconomic indicators

        Args:
            indicators: List of indicators
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary {indicator: DataFrame}
        """
        results = {}

        # Map indicator codes to FRED codes
        indicator_mapping = {
            'INFLATION': 'CPIAUCSL',  # Consumer Price Index
            'GDP': 'GDP',  # Gross Domestic Product
            'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
            'INTEREST_RATE': 'FEDFUNDS',  # Federal Funds Rate
            'RETAIL_SALES': 'RSXFS',  # Retail Sales
            'INDUSTRIAL_PRODUCTION': 'INDPRO',  # Industrial Production Index
            'HOUSE_PRICE_INDEX': 'CSUSHPISA',  # Case-Shiller Home Price Index
            'CONSUMER_SENTIMENT': 'UMCSENT'  # University of Michigan Consumer Sentiment
        }

        try:
            import pandas_datareader.data as web

            # Set up dates
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # Convert date strings to datetime
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            # Get data for each indicator
            for indicator in indicators:
                try:
                    # Convert indicator to FRED code
                    fred_code = indicator_mapping.get(indicator, indicator)

                    # Load data from FRED
                    data = web.DataReader(fred_code, 'fred', start_date_dt, end_date_dt)

                    # Save to results
                    results[indicator] = data

                    logger.info(f"Loaded macroeconomic indicator {indicator} ({fred_code})")
                except Exception as e:
                    logger.error(f"Error when loading indicator {indicator}: {e}")
                    results[indicator] = pd.DataFrame()

            return results
        except ImportError:
            logger.error("pandas_datareader not installed. Install it with pip install pandas-datareader")
            return {indicator: pd.DataFrame() for indicator in indicators}
        except Exception as e:
            logger.error(f"Error when getting macroeconomic indicators: {e}")
            return {indicator: pd.DataFrame() for indicator in indicators}

    def get_batch_data(self, tickers: List[str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, provider: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers at once

        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            provider: Data provider

        Returns:
            Dictionary {ticker: DataFrame}
        """
        results = {}

        # Process special tickers with dots
        ticker_mapping = {}
        corrected_tickers = []

        for ticker in tickers:
            if '.' in ticker:
                # Replace dot with dash for data request
                corrected_ticker = ticker.replace('.', '-')
                ticker_mapping[corrected_ticker] = ticker
                corrected_tickers.append(corrected_ticker)
            else:
                corrected_tickers.append(ticker)
                ticker_mapping[ticker] = ticker

        if ticker_mapping:
            logger.info(f"Ticker substitution for data request: {ticker_mapping}")

        # Implement multi-threaded loading for yfinance
        if provider == 'yfinance' and len(corrected_tickers) > 1:
            try:
                import yfinance as yf

                # Set default dates if not provided
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')

                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

                # Check cache for each ticker (use original tickers for file names)
                tickers_to_download = []
                download_mapping = {}  # Mapping for download: {corrected_ticker: original_ticker}

                for original_ticker in tickers:
                    cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                    if cache_file.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_age.days < self.cache_expiry_days:
                            try:
                                with open(cache_file, 'rb') as f:
                                    results[original_ticker] = pickle.load(f)
                                    logger.info(f"Loading {original_ticker} data from cache")
                                    continue
                            except Exception as e:
                                logger.warning(f"Failed to load cache for {original_ticker}: {e}")

                    # Find corrected ticker for download
                    corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker
                    tickers_to_download.append(corrected_ticker)
                    download_mapping[corrected_ticker] = original_ticker

                # Download missing data
                if tickers_to_download:
                    self.api_call_counts['yfinance'] += 1

                    # Download data in one request
                    data = yf.download(
                        tickers_to_download,
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        show_errors=False
                    )

                    # Process and save data for each ticker
                    for corrected_ticker, original_ticker in download_mapping.items():
                        if len(tickers_to_download) == 1:
                            # If only one ticker, data is not grouped
                            ticker_data = data
                        else:
                            # Extract data for specific ticker
                            ticker_data = data[corrected_ticker].copy() if corrected_ticker in data else pd.DataFrame()

                        # Check and save data
                        if not ticker_data.empty:
                            # Save to cache under original name
                            cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                            with open(cache_file, 'wb') as f:
                                pickle.dump(ticker_data, f)

                            results[original_ticker] = ticker_data
                            logger.info(
                                f"Downloaded and saved data for {original_ticker} (requested as {corrected_ticker})")

                return results
            except ImportError:
                logger.error("yfinance not installed. Install it with pip install yfinance")
            except Exception as e:
                logger.error(f"Error during batch data download: {e}")
                # Continue with sequential loading as fallback

        # Sequential loading for each ticker
        for original_ticker in tickers:
            try:
                # For single loading use the modified get_historical_prices method
                corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker

                # Temporarily replace ticker for request
                data = self.get_historical_prices(corrected_ticker, start_date, end_date, provider)

                if not data.empty:
                    # Save under original ticker
                    results[original_ticker] = data
            except Exception as e:
                logger.error(f"Error loading data for {original_ticker}: {e}")

        return results

    # Methods for specific data providers

    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Fetch data via yfinance"""
        try:
            import yfinance as yf

            # Check if ticker contains a dot and create corrected version for API
            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(f"Using corrected ticker {corrected_ticker} for request {original_ticker}")

            # Increment API call counter
            self.api_call_counts['yfinance'] += 1

            try:
                # First method - use Ticker.history with corrected ticker
                ticker_obj = yf.Ticker(corrected_ticker)
                data = ticker_obj.history(start=start_date, end=end_date, interval=interval)

                if data is None or data.empty:
                    # If that didn't work, try through download
                    data = yf.download(
                        corrected_ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        show_errors=False
                    )

                # Check and handle empty data
                if data is None or data.empty:
                    logger.warning(
                        f"No data found for {original_ticker} (requested as {corrected_ticker}) through yfinance")
                    return pd.DataFrame()

                # Check and correct index if it's not DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        logger.warning(f"Failed to convert index to DatetimeIndex: {e}")

                # Make sure all our columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        if col == 'Volume' and 'volume' in data.columns:
                            data['Volume'] = data['volume']
                        else:
                            data[col] = np.nan

                # If 'Adj Close' is missing, use 'Close'
                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']

                return data

            except Exception as e:
                logger.error(f"Error getting data via yfinance for {ticker} (requested as {corrected_ticker}): {e}")
                return pd.DataFrame()
        except ImportError:
            logger.error("yfinance not installed. Install it with pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting data via yfinance for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Fetch data via Alpha Vantage API"""
        if not self.api_keys['alpha_vantage']:
            logger.error("Alpha Vantage API key not set")
            return pd.DataFrame()

        # Check API limits
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return pd.DataFrame()

        # Check if ticker contains a dot and create corrected version for API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"Using corrected ticker {corrected_ticker} for Alpha Vantage request {original_ticker}")

        # Interval mapping
        interval_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }

        function = f"TIME_SERIES_{interval_map.get(interval, 'daily').upper()}"

        try:
            # Build URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": corrected_ticker,  # Use corrected ticker
                "apikey": self.api_keys['alpha_vantage'],
                "outputsize": "full"
            }

            # Increment API call counter
            self.api_call_counts['alpha_vantage'] += 1

            # Make request
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Check for HTTP errors

            data = response.json()

            # Check for errors in response
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API returned error for {ticker} (requested as {corrected_ticker}): {data['Error Message']}")
                return pd.DataFrame()

            # Determine time series key based on function
            time_series_key = next((k for k in data.keys() if k.startswith("Time Series")), None)

            if not time_series_key:
                logger.error(f"Unexpected Alpha Vantage response format for {original_ticker}: {data.keys()}")
                return pd.DataFrame()

            # Convert data to DataFrame
            time_series_data = data[time_series_key]

            # Create DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')

            # Convert column names
            column_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '5. volume': 'Volume',
                '6. volume': 'Volume'
            }

            df = df.rename(columns=column_map)

            # Convert data types
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Set index as DatetimeIndex
            df.index = pd.to_datetime(df.index)

            # Sort by date
            df = df.sort_index()

            # Filter by specified dates
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error to Alpha Vantage API for {original_ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data for {original_ticker}: {e}")
            return pd.DataFrame()

    def _get_yfinance_company_info(self, ticker: str) -> Dict:
        """Get company information via yfinance"""
        try:
            import yfinance as yf

            # Check if ticker contains a dot and create corrected version for API
            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(
                    f"Using corrected ticker {corrected_ticker} for company info request {original_ticker}")

            # Increment API call counter
            self.api_call_counts['yfinance'] += 1

            # Get data
            ticker_obj = yf.Ticker(corrected_ticker)
            info = ticker_obj.info

            # Process and normalize data
            # Some keys may be missing, add basic checks
            normalized_info = {
                'symbol': original_ticker,  # Keep original ticker
                'name': info.get('longName', info.get('shortName', original_ticker)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', None),
                'shares_outstanding': info.get('sharesOutstanding', None),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', None)),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None) if info.get('dividendYield') else None,
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', 'N/A'),
                'website': info.get('website', 'N/A'),
                'employees': info.get('fullTimeEmployees', None),
                'logo_url': info.get('logo_url', None),
                'type': self._determine_asset_type(info),
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Add financial metrics if available
            for metric in ['returnOnEquity', 'returnOnAssets', 'profitMargins', 'operatingMargins', 'grossMargins']:
                if metric in info:
                    normalized_info[self._camel_to_snake(metric)] = info[metric]

            return normalized_info
        except ImportError:
            logger.error("yfinance not installed. Install it with pip install yfinance")
            return {}
        except Exception as e:
            logger.error(f"Error getting company info via yfinance for {ticker}: {e}")
            return {}

    def _get_alpha_vantage_company_info(self, ticker: str) -> Dict:
        """Get company information via Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("Alpha Vantage API key not set")
            return {}

        # Check API limits
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return {}

        # Check if ticker contains a dot and create corrected version for API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"Using corrected ticker {corrected_ticker} for Alpha Vantage company info request {original_ticker}")

        try:
            # Build URL for overview
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": corrected_ticker,  # Use corrected ticker
                "apikey": self.api_keys['alpha_vantage']
            }

            # Increment API call counter
            self.api_call_counts['alpha_vantage'] += 1

            # Make request
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for data presence
            if not data or ("Error Message" in data) or len(data.keys()) <= 1:
                logger.warning(
                    f"Alpha Vantage returned no data for {original_ticker} (requested as {corrected_ticker})")
                return {}

            # Transform and normalize data
            normalized_info = {
                'symbol': original_ticker,  # Keep original ticker
                'name': data.get('Name', original_ticker),
                'sector': data.get('Sector', 'N/A'),
                'industry': data.get('Industry', 'N/A'),
                'country': data.get('Country', 'N/A'),
                'exchange': data.get('Exchange', 'N/A'),
                'currency': data.get('Currency', 'USD'),
                'market_cap': self._safe_convert(data.get('MarketCapitalization'), float),
                'pe_ratio': self._safe_convert(data.get('PERatio'), float),
                'forward_pe': self._safe_convert(data.get('ForwardPE'), float),
                'pb_ratio': self._safe_convert(data.get('PriceToBookRatio'), float),
                'dividend_yield': self._safe_convert(data.get('DividendYield'), float),
                'beta': self._safe_convert(data.get('Beta'), float),
                'description': data.get('Description', 'N/A'),
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Add financial metrics
            for key in ['ReturnOnEquityTTM', 'ReturnOnAssetsTTM', 'ProfitMargin', 'OperatingMarginTTM',
                        'GrossProfitTTM']:
                if key in data:
                    normalized_key = self._camel_to_snake(key.replace('TTM', ''))
                    normalized_info[normalized_key] = self._safe_convert(data[key], float)

            return normalized_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error to Alpha Vantage API for company info {original_ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage company info for {original_ticker}: {e}")
            return {}

    def _search_alpha_vantage(self, query: str, limit: int = 10) -> List[Dict]:
        """Search tickers via Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("Alpha Vantage API key not set")
            return []

        # Check API limits
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return []

        try:
            # Build URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": query,
                "apikey": self.api_keys['alpha_vantage']
            }

            # Increment API call counter
            self.api_call_counts['alpha_vantage'] += 1

            # Make request
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'bestMatches' in data:
                for match in data['bestMatches'][:limit]:
                    # Get symbol from result
                    symbol = match.get('1. symbol', '')

                    # Check if symbol contains dash which may be representation of a dot
                    # Replace dash with dot for display to user if it matches
                    # known format of tickers with dots (e.g., BRK-B -> BRK.B)
                    display_symbol = symbol
                    if '-' in symbol:
                        # Only for known patterns (e.g., BRK-B, BF-B)
                        known_patterns = ['BRK-B', 'BF-B']
                        if symbol in known_patterns or (len(symbol.split('-')) == 2 and len(symbol.split('-')[1]) == 1):
                            display_symbol = symbol.replace('-', '.')
                            logger.info(f"Transformed ticker in search results: {symbol} -> {display_symbol}")

                    results.append({
                        'symbol': display_symbol,  # Display with dot if appropriate format
                        'original_symbol': symbol,  # Keep original symbol
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', 'USD'),
                        'exchange': match.get('5. marketOpen', '') + '-' + match.get('6. marketClose', ''),
                        'timezone': match.get('7. timezone', '')
                    })

            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error to Alpha Vantage API for search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching tickers via Alpha Vantage: {e}")
            return []

    def _search_alternative(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Alternative method for searching tickers via Yahoo Finance API

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dictionaries with found ticker information
        """
        try:
            import yfinance as yf
            import json
            import requests

            # Use Yahoo Finance API for search
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}&newsCount=0"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Error requesting Yahoo Finance API: {response.status_code}")
                return []

            data = response.json()

            results = []
            if 'quotes' in data and data['quotes']:
                for quote in data['quotes'][:limit]:
                    # Transform format to match Alpha Vantage
                    ticker = quote.get('symbol', '')

                    # Check if ticker contains dash which may be representation of a dot
                    display_ticker = ticker
                    if '-' in ticker:
                        if len(ticker.split('-')) == 2 and len(ticker.split('-')[1]) == 1:
                            display_ticker = ticker.replace('-', '.')

                    results.append({
                        'symbol': display_ticker,
                        'original_symbol': ticker,
                        'name': quote.get('shortname', quote.get('longname', '')),
                        'type': quote.get('quoteType', ''),
                        'region': quote.get('region', 'US'),
                        'currency': quote.get('currency', 'USD'),
                        'exchange': quote.get('exchange', '')
                    })

            return results
        except Exception as e:
            logger.error(f"Error searching tickers via Yahoo Finance: {e}")
            return []

    # Helper methods

    def _determine_asset_type(self, info: Dict) -> str:
        """Determine asset type based on information"""
        if not info:
            return 'Unknown'

        # Determine type from available data
        if 'quoteType' in info:
            quote_type = info['quoteType'].lower()
            if quote_type in ['equity', 'stock']:
                return 'Stock'
            elif quote_type == 'etf':
                return 'ETF'
            elif quote_type == 'index':
                return 'Index'
            elif quote_type in ['cryptocurrency', 'crypto']:
                return 'Crypto'
            elif quote_type == 'mutualfund':
                return 'Mutual Fund'
            else:
                return quote_type.capitalize()

        # If no direct indication, try to determine from other attributes
        if 'fundFamily' in info and info['fundFamily']:
            return 'ETF' if 'ETF' in info.get('longName', '') else 'Mutual Fund'

        if 'industry' in info and info['industry']:
            return 'Stock'

        return 'Unknown'

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _safe_convert(self, value: Any, convert_type) -> Optional[Any]:
        """Safe type conversion with error handling"""
        if value is None:
            return None

        try:
            return convert_type(value)
        except (ValueError, TypeError):
            return None

    def clear_cache(self, tickers: Optional[List[str]] = None):
        """
        Clear data cache

        Args:
            tickers: List of tickers to clear. If None, clears entire cache.
        """
        if tickers is None:
            # Clear entire cache
            for file in self.cache_dir.glob('*.*'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file}: {e}")
            logger.info("Cache completely cleared")
        else:
            # Clear cache only for specified tickers
            for ticker in tickers:
                for file in self.cache_dir.glob(f"{ticker}_*.*"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file}: {e}")
            logger.info(f"Cache cleared for tickers: {tickers}")

    # Add these methods to the DataLoader class in data_loader.py

    def get_industry_peers(self, ticker: str, max_peers: int = 10, force_refresh: bool = False) -> List[Dict]:
        """
        Get list of industry peers for a given company

        Args:
            ticker: Stock ticker
            max_peers: Maximum number of peers to return
            force_refresh: Force data refresh even if cache exists

        Returns:
            List of peer companies with basic information
        """
        cache_file = self.cache_dir / f"{ticker}_peers.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:  # Peer lists don't change frequently
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} peers from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load peers cache for {ticker}: {e}")

        try:
            # First get company info to determine sector and industry
            company_info = self.get_company_info(ticker)
            if not company_info:
                logger.error(f"Failed to get company info for {ticker}")
                return []

            sector = company_info.get('sector', '')
            industry = company_info.get('industry', '')

            if not sector or not industry:
                logger.warning(f"Missing sector or industry info for {ticker}")
                return []

            # Use Financial Modeling Prep API to get peers if API key is available
            if self.api_keys['financial_modeling_prep']:
                try:
                    base_url = "https://financialmodelingprep.com/api/v3/stock/peers"
                    params = {
                        "symbol": ticker,
                        "apikey": self.api_keys['financial_modeling_prep']
                    }

                    response = requests.get(base_url, params=params)
                    response.raise_for_status()

                    peers_data = response.json()

                    if peers_data and isinstance(peers_data, list) and len(peers_data) > 0:
                        # Get the first element which should be a list of peer tickers
                        peer_tickers = peers_data[0][:max_peers] if isinstance(peers_data[0], list) else peers_data[
                                                                                                         :max_peers]

                        # Get basic info for each peer
                        peers_info = []
                        for peer_ticker in peer_tickers:
                            try:
                                peer_info = self.get_company_info(peer_ticker, provider='yfinance')
                                if peer_info:
                                    peers_info.append(peer_info)
                            except Exception as e:
                                logger.warning(f"Error getting info for peer {peer_ticker}: {e}")

                        # Save to cache
                        with open(cache_file, 'w') as f:
                            json.dump(peers_info, f)

                        return peers_info
                except Exception as e:
                    logger.warning(f"Failed to get peers from Financial Modeling Prep API: {e}")

            # Fallback: use yfinance and manual filtering by sector/industry
            import yfinance as yf

            # Get list of companies in same sector using an ETF for that sector
            sector_etfs = {
                "Information Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
                "Industrials": "XLI",
                "Communication Services": "XLC"
            }

            sector_key = next((k for k in sector_etfs.keys() if k.lower() in sector.lower()), None)

            if not sector_key:
                logger.warning(f"No sector ETF found for {sector}")
                return []

            etf_ticker = sector_etfs[sector_key]
            etf = yf.Ticker(etf_ticker)

            try:
                holdings = etf.holdings

                # Filter and sort by weight
                if not holdings.empty:
                    # Get top holdings
                    top_holdings = holdings.sort_values('% Assets', ascending=False).head(30)

                    # Get info for each holding
                    peers_info = []
                    for i, (holding_ticker, _) in enumerate(top_holdings.iterrows()):
                        if i >= max_peers:
                            break

                        if holding_ticker == ticker:
                            continue

                        try:
                            peer_info = self.get_company_info(holding_ticker, provider='yfinance')
                            # Check if peer is in same industry
                            if peer_info and peer_info.get('industry') == industry:
                                peers_info.append(peer_info)
                        except Exception as e:
                            logger.warning(f"Error getting info for peer {holding_ticker}: {e}")

                    # Save to cache
                    with open(cache_file, 'w') as f:
                        json.dump(peers_info, f)

                    return peers_info
            except Exception as e:
                logger.warning(f"Failed to get holdings for ETF {etf_ticker}: {e}")

            return []
        except Exception as e:
            logger.error(f"Error getting peers for {ticker}: {e}")
            return []

    def get_analyst_estimates(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Get analyst estimates for a company (EPS, revenue forecasts, price targets)

        Args:
            ticker: Stock ticker
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with analyst estimates
        """
        cache_file = self.cache_dir / f"{ticker}_estimates.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Analyst estimates update frequently, but 7 days is reasonable
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} analyst estimates from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load analyst estimates cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Initialize estimates dictionary
            estimates = {
                'eps_trend': {},
                'revenue_estimates': {},
                'earnings_estimates': {},
                'price_targets': {},
                'growth_estimates': {},
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Get analyst recommendations
            try:
                recommendations = ticker_obj.recommendations
                if recommendations is not None and not recommendations.empty:
                    # Get latest recommendations
                    recent_recommendations = recommendations.iloc[-10:] if len(
                        recommendations) > 10 else recommendations

                    # Calculate consensus
                    if 'To Grade' in recent_recommendations.columns:
                        grades = recent_recommendations['To Grade'].value_counts().to_dict()
                        estimates['price_targets']['consensus'] = grades
            except Exception as e:
                logger.warning(f"Failed to get recommendations for {ticker}: {e}")

            # Get earnings estimates
            try:
                earnings_trend = ticker_obj.earnings_trend
                if earnings_trend is not None and not earnings_trend.empty:
                    # Convert to dictionary
                    estimates['eps_trend'] = earnings_trend.to_dict()
            except Exception as e:
                logger.warning(f"Failed to get earnings trend for {ticker}: {e}")

            # Get earnings history
            try:
                earnings_history = ticker_obj.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    # Convert to dictionary
                    estimates['earnings_history'] = earnings_history.to_dict()
            except Exception as e:
                logger.warning(f"Failed to get earnings history for {ticker}: {e}")

            # Get quarterly financials for revenue estimates
            try:
                quarterly_financials = ticker_obj.quarterly_financials
                if quarterly_financials is not None and not quarterly_financials.empty:
                    # Use this to create revenue estimates
                    if 'Total Revenue' in quarterly_financials.index:
                        revenues = quarterly_financials.loc['Total Revenue']
                        # Calculate average quarterly growth
                        revenue_values = revenues.values
                        quarterly_growth = []

                        for i in range(1, len(revenue_values)):
                            if revenue_values[i - 1] != 0:
                                growth = (revenue_values[i] - revenue_values[i - 1]) / revenue_values[i - 1]
                                quarterly_growth.append(growth)

                        if quarterly_growth:
                            avg_growth = sum(quarterly_growth) / len(quarterly_growth)
                            # Estimate next quarter
                            next_revenue = revenue_values[-1] * (1 + avg_growth)

                            estimates['revenue_estimates']['next_quarter'] = float(next_revenue)
                            estimates['revenue_estimates']['quarterly_growth'] = float(avg_growth)
            except Exception as e:
                logger.warning(f"Failed to get quarterly financials for {ticker}: {e}")

            # Get financial data for annual estimates
            try:
                financials = ticker_obj.financials
                if financials is not None and not financials.empty:
                    # Use this to create annual estimates
                    if 'Total Revenue' in financials.index:
                        annual_revenues = financials.loc['Total Revenue']
                        # Calculate average annual growth
                        annual_revenue_values = annual_revenues.values
                        annual_growth = []

                        for i in range(1, len(annual_revenue_values)):
                            if annual_revenue_values[i - 1] != 0:
                                growth = (annual_revenue_values[i] - annual_revenue_values[i - 1]) / \
                                         annual_revenue_values[i - 1]
                                annual_growth.append(growth)

                        if annual_growth:
                            avg_annual_growth = sum(annual_growth) / len(annual_growth)
                            # Estimate next year
                            next_annual_revenue = annual_revenue_values[-1] * (1 + avg_annual_growth)

                            estimates['revenue_estimates']['next_year'] = float(next_annual_revenue)
                            estimates['revenue_estimates']['annual_growth'] = float(avg_annual_growth)
            except Exception as e:
                logger.warning(f"Failed to get annual financials for {ticker}: {e}")

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(estimates, f)

            return estimates
        except Exception as e:
            logger.error(f"Error getting analyst estimates for {ticker}: {e}")
            return {}

    def get_insider_trading(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get insider trading data for a company

        Args:
            ticker: Stock ticker
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with insider trading data
        """
        cache_file = self.cache_dir / f"{ticker}_insider_trading.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Update weekly is reasonable
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} insider trading from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load insider trading cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Get insider trades
            insider_trades = ticker_obj.insider_transactions

            # Save to cache if successful
            if insider_trades is not None and not insider_trades.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(insider_trades, f)
                logger.info(f"Saved {ticker} insider trading to cache")

            return insider_trades if insider_trades is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting insider trading for {ticker}: {e}")
            return pd.DataFrame()

    def get_esg_data(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Get Environmental, Social, and Governance (ESG) data for a company

        Args:
            ticker: Stock ticker
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with ESG data
        """
        cache_file = self.cache_dir / f"{ticker}_esg.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:  # ESG data doesn't change frequently
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} ESG data from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load ESG data cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Initialize ESG dictionary
            esg_data = {
                'environmental': {},
                'social': {},
                'governance': {},
                'controversy': None,
                'total_esg': None,
                'percentile': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Get ESG scores
            try:
                sustainability = ticker_obj.sustainability

                if sustainability is not None and not sustainability.empty:
                    # Extract data
                    for index, value in sustainability.iterrows():
                        metric = index.lower()

                        # Classify metrics
                        if 'environment' in metric:
                            esg_data['environmental'][index] = float(value[0]) if not pd.isna(value[0]) else None
                        elif 'social' in metric:
                            esg_data['social'][index] = float(value[0]) if not pd.isna(value[0]) else None
                        elif 'governance' in metric:
                            esg_data['governance'][index] = float(value[0]) if not pd.isna(value[0]) else None
                        elif 'controversy' in metric:
                            esg_data['controversy'] = float(value[0]) if not pd.isna(value[0]) else None
                        elif 'total' in metric and 'esg' in metric:
                            esg_data['total_esg'] = float(value[0]) if not pd.isna(value[0]) else None
                        elif 'percentile' in metric:
                            esg_data['percentile'] = float(value[0]) if not pd.isna(value[0]) else None
            except Exception as e:
                logger.warning(f"Failed to get sustainability data for {ticker}: {e}")

            # Save to cache if we have any data
            if esg_data['environmental'] or esg_data['social'] or esg_data['governance'] or esg_data['total_esg']:
                with open(cache_file, 'w') as f:
                    json.dump(esg_data, f)
                logger.info(f"Saved {ticker} ESG data to cache")

            return esg_data
        except Exception as e:
            logger.error(f"Error getting ESG data for {ticker}: {e}")
            return {
                'environmental': {},
                'social': {},
                'governance': {},
                'controversy': None,
                'total_esg': None,
                'percentile': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

    def get_historical_dividends(self, ticker: str, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get historical dividend data for a company

        Args:
            ticker: Stock ticker
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with dividend history
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            # Default to 10 years of historical data for dividends
            start_date = (datetime.now() - timedelta(days=10 * 365)).strftime('%Y-%m-%d')

        cache_file = self.cache_dir / f"{ticker}_dividends_{start_date}_{end_date}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:  # Dividends don't update very frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} dividend history from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load dividend history cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Get dividends
            dividends = ticker_obj.dividends

            # Filter by date
            if dividends is not None and not dividends.empty:
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)

                dividends = dividends[(dividends.index >= start_date_dt) & (dividends.index <= end_date_dt)]

                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(dividends, f)
                logger.info(f"Saved {ticker} dividend history to cache")

            return dividends if dividends is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting dividend history for {ticker}: {e}")
            return pd.DataFrame()

    def get_stock_screener(self, filters: Dict[str, Dict[str, Union[str, float, int]]],
                           limit: int = 50) -> List[Dict]:
        """
        Screen stocks based on specified filters

        Args:
            filters: Dictionary of filters {metric: {operator: <, >, =, etc.>, value: <threshold>}}
            limit: Maximum number of results to return

        Returns:
            List of dictionaries with stock information that match the filters
        """
        cache_key = f"screener_{hash(str(filters))}.json"
        cache_file = self.cache_dir / cache_key

        # Check cache (screening results expire after 1 day)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 1:
                try:
                    with open(cache_file, 'r') as f:
                        results = json.load(f)
                        logger.info("Loading screener results from cache")
                        return results[:limit]
                except Exception as e:
                    logger.warning(f"Failed to load screener cache: {e}")

        try:
            # Use Financial Modeling Prep API for screening if API key is available
            if self.api_keys['financial_modeling_prep']:
                try:
                    # Base URL for stock screener
                    base_url = "https://financialmodelingprep.com/api/v3/stock-screener"

                    # Map our filter operators to FMP parameters
                    operator_map = {
                        ">": "gt",
                        "<": "lt",
                        "=": "eq",
                        ">=": "gte",
                        "<=": "lte"
                    }

                    # Map our metrics to FMP parameters
                    metric_map = {
                        "Market Cap": "marketCapMoreThan" if filters.get("Market Cap", {}).get("operator") in [">",
                                                                                                               ">="] else "marketCapLowerThan",
                        "P/E": "priceEarningsRatioMoreThan" if filters.get("P/E", {}).get("operator") in [">",
                                                                                                          ">="] else "priceEarningsRatioLowerThan",
                        "EPS": "earningsPerShareMoreThan" if filters.get("EPS", {}).get("operator") in [">",
                                                                                                        ">="] else "earningsPerShareLowerThan",
                        "Dividend Yield": "dividendMoreThan" if filters.get("Dividend Yield", {}).get("operator") in [
                            ">", ">="] else "dividendLowerThan",
                        "Volume": "volumeMoreThan" if filters.get("Volume", {}).get("operator") in [">",
                                                                                                    ">="] else "volumeLowerThan",
                        "Price": "priceMoreThan" if filters.get("Price", {}).get("operator") in [">",
                                                                                                 ">="] else "priceLowerThan",
                        "Beta": "betaMoreThan" if filters.get("Beta", {}).get("operator") in [">",
                                                                                              ">="] else "betaLowerThan"
                    }

                    # Build params
                    params = {
                        "apikey": self.api_keys['financial_modeling_prep'],
                        "limit": limit
                    }

                    # Add filters to params
                    for metric, filter_info in filters.items():
                        if metric in metric_map:
                            param_name = metric_map[metric]
                            params[param_name] = filter_info["value"]

                    # Make request
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()

                    data = response.json()

                    if isinstance(data, list) and data:
                        # Process and normalize the results
                        results = []
                        for item in data:
                            # Map to our standard format
                            normalized_item = {
                                "symbol": item.get("symbol", ""),
                                "name": item.get("companyName", ""),
                                "sector": item.get("sector", "N/A"),
                                "industry": item.get("industry", "N/A"),
                                "country": item.get("country", "N/A"),
                                "market_cap": item.get("marketCap", None),
                                "price": item.get("price", None),
                                "pe_ratio": item.get("pe", None),
                                "eps": item.get("eps", None),
                                "dividend_yield": item.get("dividendYield", None),
                                "beta": item.get("beta", None),
                                "volume": item.get("volume", None)
                            }
                            results.append(normalized_item)

                        # Save to cache
                        with open(cache_file, 'w') as f:
                            json.dump(results, f)
                        logger.info(f"Saved screener results to cache")

                        return results[:limit]
                except Exception as e:
                    logger.warning(f"Failed to get screener results from Financial Modeling Prep API: {e}")

            # Fallback method - load S&P 500 companies and filter manually
            try:
                import pandas as pd
                import numpy as np

                # Get S&P 500 components (or other default universe)
                sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(sp500_url)
                sp500_table = tables[0]

                # Get tickers
                tickers = sp500_table['Symbol'].tolist()

                # Initialize results
                results = []

                # Process tickers in batches
                batch_size = 10
                for i in range(0, len(tickers), batch_size):
                    batch_tickers = tickers[i:i + batch_size]

                    # Skip if we've reached the limit
                    if len(results) >= limit:
                        break

                    # Get basic info for each ticker
                    for ticker in batch_tickers:
                        try:
                            # Get company info and financial ratios
                            company_info = self.get_company_info(ticker)
                            financial_ratios = self.get_financial_ratios(ticker)

                            if not company_info or not financial_ratios:
                                continue

                            # Combine info and ratios
                            stock_data = {**company_info}

                            # Flatten financial ratios
                            for category, ratios in financial_ratios.items():
                                if isinstance(ratios, dict):
                                    for ratio_name, ratio_value in ratios.items():
                                        stock_data[ratio_name] = ratio_value

                            # Apply filters
                            passes_filters = True
                            for metric, filter_info in filters.items():
                                # Map our metrics to data fields
                                field_map = {
                                    "Market Cap": "market_cap",
                                    "P/E": "pe_ratio",
                                    "EPS": "eps",
                                    "Dividend Yield": "dividend_yield",
                                    "Beta": "beta",
                                    "ROE": "return_on_equity",
                                    "ROA": "return_on_assets",
                                    "Debt/Equity": "debt_to_equity",
                                    "Price": "price",
                                    "Volume": "volume"
                                }

                                field = field_map.get(metric, metric.lower().replace(" ", "_"))

                                if field in stock_data and stock_data[field] is not None:
                                    operator = filter_info["operator"]
                                    value = filter_info["value"]

                                    # Apply operator
                                    if operator == ">" and not (stock_data[field] > value):
                                        passes_filters = False
                                        break
                                    elif operator == ">=" and not (stock_data[field] >= value):
                                        passes_filters = False
                                        break
                                    elif operator == "<" and not (stock_data[field] < value):
                                        passes_filters = False
                                        break
                                    elif operator == "<=" and not (stock_data[field] <= value):
                                        passes_filters = False
                                        break
                                    elif operator == "=" and not (stock_data[field] == value):
                                        passes_filters = False
                                        break
                                else:
                                    # Field not available, skip this filter
                                    passes_filters = False
                                    break

                            if passes_filters:
                                results.append(stock_data)

                                # Break if we've reached the limit
                                if len(results) >= limit:
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing {ticker} for screening: {e}")
                            continue

                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Saved screener results to cache")

                return results[:limit]
            except Exception as e:
                logger.error(f"Error in fallback stock screener: {e}")
                return []
        except Exception as e:
            logger.error(f"Error in stock screener: {e}")
            return []

    def get_corporate_events(self, ticker: str, event_type: Optional[str] = None,
                             force_refresh: bool = False) -> List[Dict]:
        """
        Get corporate events for a company (earnings reports, dividends, splits, etc.)

        Args:
            ticker: Stock ticker
            event_type: Type of event to filter by ('earnings', 'dividends', 'splits', etc.)
            force_refresh: Force data refresh even if cache exists

        Returns:
            List of dictionaries with event information
        """
        cache_file = self.cache_dir / f"{ticker}_events.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 14:  # Update every 2 weeks is reasonable
                try:
                    with open(cache_file, 'r') as f:
                        events = json.load(f)
                        logger.info(f"Loading {ticker} events from cache")

                        # Filter by event type if specified
                        if event_type:
                            return [event for event in events if event.get('type') == event_type]
                        return events
                except Exception as e:
                    logger.warning(f"Failed to load events cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Initialize events list
            events = []

            # Get calendar events
            try:
                calendar = ticker_obj.calendar

                if calendar is not None:
                    # Extract earnings dates
                    if 'Earnings Date' in calendar:
                        earnings_dates = calendar['Earnings Date']
                        if isinstance(earnings_dates, list):
                            for date in earnings_dates:
                                events.append({
                                    'type': 'earnings',
                                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                                    'description': f"Earnings Report",
                                    'details': {}
                                })
                        else:
                            events.append({
                                'type': 'earnings',
                                'date': pd.to_datetime(earnings_dates).strftime('%Y-%m-%d'),
                                'description': f"Earnings Report",
                                'details': {}
                            })

                    # Extract dividend dates
                    for event_name, event_type in [
                        ('Dividend Date', 'dividend_payment'),
                        ('Ex-Dividend Date', 'ex_dividend')
                    ]:
                        if event_name in calendar:
                            date = calendar[event_name]
                            events.append({
                                'type': event_type,
                                'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                                'description': f"{event_name}",
                                'details': {}
                            })
            except Exception as e:
                logger.warning(f"Failed to get calendar events for {ticker}: {e}")

            # Get historical dividends
            try:
                dividends = ticker_obj.dividends

                if dividends is not None and not dividends.empty:
                    for date, amount in dividends.items():
                        events.append({
                            'type': 'dividend',
                            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                            'description': f"Dividend Payment: ${amount:.2f}",
                            'details': {
                                'amount': float(amount)
                            }
                        })
            except Exception as e:
                logger.warning(f"Failed to get dividend events for {ticker}: {e}")

            # Get historical splits
            try:
                splits = ticker_obj.splits

                if splits is not None and not splits.empty:
                    for date, ratio in splits.items():
                        numerator, denominator = ratio.as_integer_ratio()
                        events.append({
                            'type': 'split',
                            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                            'description': f"Stock Split: {numerator}:{denominator}",
                            'details': {
                                'ratio': float(ratio),
                                'numerator': numerator,
                                'denominator': denominator
                            }
                        })
            except Exception as e:
                logger.warning(f"Failed to get split events for {ticker}: {e}")

            # Get earnings history
            try:
                earnings = ticker_obj.earnings_dates

                if earnings is not None and not earnings.empty:
                    for date, row in earnings.iterrows():
                        event = {
                            'type': 'earnings',
                            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                            'description': "Earnings Report",
                            'details': {}
                        }

                        # Add earnings details
                        for col in earnings.columns:
                            if pd.notna(row[col]):
                                event['details'][col.lower().replace(' ', '_')] = row[col]

                        events.append(event)
            except Exception as e:
                logger.warning(f"Failed to get earnings events for {ticker}: {e}")

            # Sort events by date
            events.sort(key=lambda x: x['date'], reverse=True)

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(events, f)
            logger.info(f"Saved {ticker} events to cache")

            # Filter by event type if specified
            if event_type:
                return [event for event in events if event.get('type') == event_type]
            return events
        except Exception as e:
            logger.error(f"Error getting corporate events for {ticker}: {e}")
            return []

    def get_financial_statements_enhanced(self, ticker: str, statement_type: str = 'income',
                                          period: str = 'annual', as_reported: bool = False,
                                          force_refresh: bool = False) -> pd.DataFrame:
        """
        Enhanced version of get_financial_statements with more options

        Args:
            ticker: Stock ticker
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            as_reported: Get statements as originally reported (without revisions)
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with financial statement data
        """
        cache_key = f"{ticker}_{statement_type}_{period}_{'as_reported' if as_reported else 'standardized'}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Financial statements update less frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} enhanced financial statements from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load enhanced financial statements cache for {ticker}: {e}")

        try:
            # First try with FMP API if available
            if self.api_keys['financial_modeling_prep']:
                # Determine endpoint
                endpoint_map = {
                    'income': 'income-statement',
                    'balance': 'balance-sheet-statement',
                    'cash': 'cash-flow-statement'
                }

                period_param = 'quarterly' if period == 'quarterly' else 'annual'

                # Build URL
                endpoint = endpoint_map.get(statement_type, 'income-statement')
                base_url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}"
                params = {
                    "period": period_param,
                    "apikey": self.api_keys['financial_modeling_prep']
                }

                # Add as_reported parameter if needed
                if as_reported:
                    params["as_reported"] = "true"

                try:
                    # Make request
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()

                    data = response.json()

                    if isinstance(data, list) and data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)

                        # Set date as index
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            df = df.sort_index(ascending=False)  # Most recent first

                        # Save to cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        logger.info(f"Saved {ticker} enhanced financial statements to cache")

                        return df
                except Exception as e:
                    logger.warning(f"Failed to get enhanced financial statements from FMP API: {e}")

            # Fallback to yfinance
            return self.get_financial_statements(ticker, statement_type, period, force_refresh)
        except Exception as e:
            logger.error(f"Error getting enhanced financial statements for {ticker}: {e}")
            return pd.DataFrame()

    def get_institutional_holdings(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Get institutional holdings information for a company

        Args:
            ticker: Stock ticker
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with institutional holdings data
        """
        cache_file = self.cache_dir / f"{ticker}_institutional_holdings.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:  # Institutional holdings update quarterly
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} institutional holdings from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load institutional holdings cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Get ticker object
            ticker_obj = yf.Ticker(ticker)

            # Initialize holdings dictionary
            holdings = {
                'major_holders': {},
                'institutional_holders': [],
                'mutualfund_holders': [],
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Get major holders
            try:
                major_holders = ticker_obj.major_holders

                if major_holders is not None and not major_holders.empty:
                    # Process and normalize
                    for i, row in major_holders.iterrows():
                        if len(row) >= 2:
                            holdings['major_holders'][row[1]] = float(row[0].strip('%')) / 100
            except Exception as e:
                logger.warning(f"Failed to get major holders for {ticker}: {e}")

            # Get institutional holders
            try:
                institutional_holders = ticker_obj.institutional_holders

                if institutional_holders is not None and not institutional_holders.empty:
                    # Process and normalize
                    for _, row in institutional_holders.iterrows():
                        holder = {
                            'name': row.get('Holder', 'Unknown'),
                            'shares': int(row.get('Shares', 0)),
                            'value': float(row.get('Value', 0)) if 'Value' in row else None,
                            'percentage': float(row.get('% Out', 0)) / 100 if '% Out' in row else None,
                            'change': int(row.get('Change', 0)) if 'Change' in row else None,
                            'date_reported': row.get('Date Reported', None)
                        }

                        # Convert date if available
                        if holder['date_reported'] is not None:
                            holder['date_reported'] = pd.to_datetime(holder['date_reported']).strftime('%Y-%m-%d')

                        holdings['institutional_holders'].append(holder)
            except Exception as e:
                logger.warning(f"Failed to get institutional holders for {ticker}: {e}")

            # Get mutual fund holders
            try:
                mutualfund_holders = ticker_obj.mutualfund_holders

                if mutualfund_holders is not None and not mutualfund_holders.empty:
                    # Process and normalize
                    for _, row in mutualfund_holders.iterrows():
                        holder = {
                            'name': row.get('Holder', 'Unknown'),
                            'shares': int(row.get('Shares', 0)),
                            'value': float(row.get('Value', 0)) if 'Value' in row else None,
                            'percentage': float(row.get('% Out', 0)) / 100 if '% Out' in row else None,
                            'change': int(row.get('Change', 0)) if 'Change' in row else None,
                            'date_reported': row.get('Date Reported', None)
                        }

                        # Convert date if available
                        if holder['date_reported'] is not None:
                            holder['date_reported'] = pd.to_datetime(holder['date_reported']).strftime('%Y-%m-%d')

                        holdings['mutualfund_holders'].append(holder)
            except Exception as e:
                logger.warning(f"Failed to get mutual fund holders for {ticker}: {e}")

            # Check if we have any data
            if holdings['major_holders'] or holdings['institutional_holders'] or holdings['mutualfund_holders']:
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(holdings, f)
                logger.info(f"Saved {ticker} institutional holdings to cache")

            return holdings
        except Exception as e:
            logger.error(f"Error getting institutional holdings for {ticker}: {e}")
            return {
                'major_holders': {},
                'institutional_holders': [],
                'mutualfund_holders': [],
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

    def get_key_metrics_history(self, ticker: str, period: str = 'annual',
                                years: int = 5, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get historical key metrics for a company over multiple years

        Args:
            ticker: Stock ticker
            period: Period ('annual' or 'quarterly')
            years: Number of years of historical data
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with historical key metrics
        """
        cache_file = self.cache_dir / f"{ticker}_key_metrics_{period}_{years}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Key metrics update less frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} key metrics history from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load key metrics history cache for {ticker}: {e}")

        try:
            # Try with FMP API first if available
            if self.api_keys['financial_modeling_prep']:
                # Build URL
                base_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}"
                params = {
                    "period": "quarter" if period == 'quarterly' else "annual",
                    "limit": years * (4 if period == 'quarterly' else 1),  # Adjust limit based on period
                    "apikey": self.api_keys['financial_modeling_prep']
                }

                try:
                    # Make request
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()

                    data = response.json()

                    if isinstance(data, list) and data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)

                        # Set date as index
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            df = df.sort_index(ascending=False)  # Most recent first

                        # Save to cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        logger.info(f"Saved {ticker} key metrics history to cache")

                        return df
                except Exception as e:
                    logger.warning(f"Failed to get key metrics history from FMP API: {e}")

            # Fallback to calculating key metrics from financial statements
            try:
                # Get financial statements
                income_stmt = self.get_financial_statements(ticker, 'income', period)
                balance_sheet = self.get_financial_statements(ticker, 'balance', period)
                cash_flow = self.get_financial_statements(ticker, 'cash', period)

                if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
                    logger.warning(f"Missing financial statements for {ticker}")
                    return pd.DataFrame()

                # Initialize result DataFrame
                result = pd.DataFrame()

                # Extract key metrics
                for date in income_stmt.columns:
                    # Get data for the date
                    income_data = income_stmt[date]
                    balance_data = balance_sheet[date] if date in balance_sheet.columns else None
                    cash_data = cash_flow[date] if date in cash_flow.columns else None

                    if balance_data is None or cash_data is None:
                        continue

                    # Calculate key metrics
                    metrics = {
                        'date': date,
                        'revenue': income_data.get('Total Revenue'),
                        'gross_profit': income_data.get('Gross Profit'),
                        'operating_income': income_data.get('Operating Income'),
                        'net_income': income_data.get('Net Income'),
                        'eps': income_data.get('EPS (Basic)'),
                        'total_assets': balance_data.get('Total Assets'),
                        'total_liabilities': balance_data.get('Total Liabilities Net Minority Interest'),
                        'total_equity': balance_data.get('Total Equity'),
                        'operating_cash_flow': cash_data.get('Operating Cash Flow'),
                        'capital_expenditure': cash_data.get('Capital Expenditure'),
                        'free_cash_flow': cash_data.get('Free Cash Flow')
                    }

                    # Calculate ratios if possible
                    if metrics['total_equity'] and metrics['net_income']:
                        metrics['roe'] = metrics['net_income'] / metrics['total_equity']

                    if metrics['total_assets'] and metrics['net_income']:
                        metrics['roa'] = metrics['net_income'] / metrics['total_assets']

                    if metrics['revenue'] and metrics['net_income']:
                        metrics['net_margin'] = metrics['net_income'] / metrics['revenue']

                    if metrics['revenue'] and metrics['gross_profit']:
                        metrics['gross_margin'] = metrics['gross_profit'] / metrics['revenue']

                    # Add to result
                    result = pd.concat([result, pd.DataFrame([metrics])], ignore_index=True)

                # Convert date to datetime and set as index
                result['date'] = pd.to_datetime(result['date'])
                result = result.set_index('date')
                result = result.sort_index(ascending=False)  # Most recent first

                # Limit to the requested number of years
                start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
                result = result[result.index >= start_date]

                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Saved {ticker} key metrics history (calculated) to cache")

                return result
            except Exception as e:
                logger.warning(f"Failed to calculate key metrics for {ticker}: {e}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting key metrics history for {ticker}: {e}")
            return pd.DataFrame()

    def get_peer_comparison(self, ticker: str, peers: Optional[List[str]] = None,
                            metrics: Optional[List[str]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get comparison data for a company and its peers

        Args:
            ticker: Stock ticker
            peers: List of peer tickers (if None, automatically determined)
            metrics: List of metrics to compare (if None, use default set)
            force_refresh: Force data refresh even if cache exists

        Returns:
            DataFrame with comparison data
        """
        # If peers not provided, get them automatically
        if peers is None:
            peers_data = self.get_industry_peers(ticker, force_refresh=force_refresh)
            peers = [peer_info.get('symbol') for peer_info in peers_data]

        # Add the main ticker to the list if not already there
        if ticker not in peers:
            peers = [ticker] + peers

        # Define default metrics if not provided
        if metrics is None:
            metrics = [
                "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_to_ebitda",
                "return_on_equity", "return_on_assets", "profit_margins", "operating_margins",
                "debt_to_equity", "current_ratio", "quick_ratio", "beta", "dividend_yield"
            ]

        cache_key = f"{ticker}_peer_comparison_{','.join(sorted(peers))}_{','.join(sorted(metrics))}"
        cache_file = self.cache_dir / f"{hash(cache_key)}.pkl"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Peer comparison data updates less frequently
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {ticker} peer comparison from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load peer comparison cache for {ticker}: {e}")

            try:
                # Initialize result DataFrame
                result = pd.DataFrame(index=peers)

                # Add company names
                result['Company Name'] = ""
                result['Sector'] = ""
                result['Industry'] = ""
                result['Market Cap'] = None

                # Get data for each peer
                for peer in peers:
                    try:
                        # Get company info and financial ratios
                        company_info = self.get_company_info(peer)
                        financial_ratios = self.get_financial_ratios(peer)

                        if company_info:
                            result.at[peer, 'Company Name'] = company_info.get('name', peer)
                            result.at[peer, 'Sector'] = company_info.get('sector', 'N/A')
                            result.at[peer, 'Industry'] = company_info.get('industry', 'N/A')
                            result.at[peer, 'Market Cap'] = company_info.get('market_cap', None)

                        # Add metrics from financial ratios
                        if financial_ratios:
                            # Flatten financial ratios
                            flattened_ratios = {}
                            for category, ratios in financial_ratios.items():
                                if isinstance(ratios, dict):
                                    for ratio_name, ratio_value in ratios.items():
                                        flattened_ratios[ratio_name] = ratio_value

                            # Add requested metrics
                            for metric in metrics:
                                # Try different variations of metric names
                                variations = [
                                    metric,
                                    metric.lower(),
                                    metric.lower().replace(' ', '_'),
                                    '_'.join(metric.lower().split('_')[1:])  # Without prefix
                                ]

                                for var in variations:
                                    if var in flattened_ratios:
                                        result.at[peer, metric] = flattened_ratios[var]
                                        break
                                    elif var in company_info:
                                        result.at[peer, metric] = company_info[var]
                                        break
                    except Exception as e:
                        logger.warning(f"Error getting data for peer {peer}: {e}")

                # Clean up column names
                result.columns = [col.replace('_', ' ').title() for col in result.columns]

                # Reorder columns to put company name and sector first
                first_cols = ['Company Name', 'Sector', 'Industry', 'Market Cap']
                other_cols = [col for col in result.columns if col not in first_cols]
                result = result[first_cols + other_cols]

                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Saved {ticker} peer comparison to cache")

                return result
            except Exception as e:
                logger.error(f"Error getting peer comparison for {ticker}: {e}")
                return pd.DataFrame()

    def get_valuation_models(self, ticker: str, models: Optional[List[str]] = None,
                             force_refresh: bool = False) -> Dict:
        """
        Get valuation model outputs for a company

        Args:
            ticker: Stock ticker
            models: List of models to calculate ('DCF', 'DDM', 'NAV', etc.)
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with valuation model outputs
        """
        # Define default models if not provided
        if models is None:
            models = ['DCF', 'MULTIPLES']

        cache_key = f"{ticker}_valuation_{','.join(sorted(models))}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Valuation models update less frequently
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} valuation models from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load valuation models cache for {ticker}: {e}")

        try:
            # Initialize results dictionary
            valuation_results = {
                'ticker': ticker,
                'models': {},
                'price': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Get current price
            price_data = self.get_historical_prices(ticker, interval='1d', force_refresh=force_refresh)
            if not price_data.empty:
                valuation_results['price'] = float(price_data['Close'].iloc[-1])

            # Calculate DCF model if requested
            if 'DCF' in models:
                try:
                    # Get required data for DCF
                    income_stmt = self.get_financial_statements(ticker, 'income', 'annual')
                    balance_sheet = self.get_financial_statements(ticker, 'balance', 'annual')
                    cash_flow = self.get_financial_statements(ticker, 'cash', 'annual')
                    company_info = self.get_company_info(ticker)

                    # Simplified DCF calculation
                    if not income_stmt.empty and not balance_sheet.empty and not cash_flow.empty:
                        # Get most recent data
                        latest_year = income_stmt.columns[0]

                        # Extract required data
                        revenue = income_stmt[latest_year].get('Total Revenue')
                        net_income = income_stmt[latest_year].get('Net Income')
                        fcf = cash_flow[latest_year].get('Free Cash Flow')
                        debt = balance_sheet[latest_year].get('Total Debt', 0)
                        cash = balance_sheet[latest_year].get('Cash and Cash Equivalents', 0)
                        outstanding_shares = company_info.get('shares_outstanding', None)

                        # Make sure we have the required data
                        if revenue and net_income and fcf and outstanding_shares:
                            # Calculate FCF growth rate from historical data
                            growth_rates = []
                            for i in range(1, min(5, len(cash_flow.columns))):
                                if i < len(cash_flow.columns):
                                    current_fcf = cash_flow[cash_flow.columns[i - 1]].get('Free Cash Flow')
                                    previous_fcf = cash_flow[cash_flow.columns[i]].get('Free Cash Flow')

                                    if current_fcf and previous_fcf and previous_fcf > 0:
                                        growth_rate = (current_fcf - previous_fcf) / previous_fcf
                                        growth_rates.append(growth_rate)

                            # Use average growth rate or default to 3%
                            growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0.03

                            # Cap growth rate to be reasonable
                            growth_rate = min(max(growth_rate, 0.01), 0.15)

                            # Determine discount rate based on sector
                            sector = company_info.get('sector', '')
                            if sector in ['Technology', 'Communication Services']:
                                discount_rate = 0.1  # Higher risk
                            elif sector in ['Utilities', 'Consumer Staples']:
                                discount_rate = 0.07  # Lower risk
                            else:
                                discount_rate = 0.09  # Medium risk

                            # Calculate DCF
                            forecast_years = 5
                            terminal_growth_rate = 0.025  # Long-term growth rate

                            # Project FCF for forecast period
                            projected_fcf = []
                            for year in range(1, forecast_years + 1):
                                projected_fcf.append(fcf * (1 + growth_rate) ** year)

                            # Calculate terminal value
                            terminal_value = projected_fcf[-1] * (1 + terminal_growth_rate) / (
                                        discount_rate - terminal_growth_rate)

                            # Discount cash flows to present value
                            present_value_fcf = sum(
                                [cf / (1 + discount_rate) ** year for year, cf in enumerate(projected_fcf, 1)])
                            present_value_terminal = terminal_value / (1 + discount_rate) ** forecast_years

                            # Calculate enterprise value
                            enterprise_value = present_value_fcf + present_value_terminal

                            # Calculate equity value
                            equity_value = enterprise_value + cash - debt

                            # Calculate per share value
                            dcf_value = equity_value / outstanding_shares

                            # Save DCF results
                            valuation_results['models']['DCF'] = {
                                'value': float(dcf_value),
                                'enterprise_value': float(enterprise_value),
                                'equity_value': float(equity_value),
                                'discount_rate': float(discount_rate),
                                'growth_rate': float(growth_rate),
                                'terminal_growth_rate': float(terminal_growth_rate),
                                'forecast_years': forecast_years,
                                'projected_fcf': [float(cf) for cf in projected_fcf]
                            }
                except Exception as e:
                    logger.warning(f"Failed to calculate DCF model for {ticker}: {e}")

            # Calculate multiple-based valuation if requested
            if 'MULTIPLES' in models:
                try:
                    # Get peer comparison data
                    peers_data = self.get_peer_comparison(ticker, metrics=[
                        'pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'ev_to_ebitda'
                    ])

                    # Get financial data for the company
                    company_info = self.get_company_info(ticker)
                    income_stmt = self.get_financial_statements(ticker, 'income', 'annual')

                    if not peers_data.empty and not income_stmt.empty:
                        # Extract metrics
                        multiples = {}
                        for col in ['Pe Ratio', 'Forward Pe', 'Pb Ratio', 'Ps Ratio', 'Ev To Ebitda']:
                            if col in peers_data.columns:
                                # Filter out the ticker itself and null values
                                peer_values = peers_data.loc[peers_data.index != ticker, col].dropna()

                                if not peer_values.empty:
                                    # Calculate median and mean
                                    multiples[col] = {
                                        'median': float(peer_values.median()),
                                        'mean': float(peer_values.mean()),
                                        'min': float(peer_values.min()),
                                        'max': float(peer_values.max())
                                    }

                        # Calculate valuations based on multiples
                        valuations = {}

                        # Get the latest financial data
                        latest_year = income_stmt.columns[0]
                        eps = income_stmt[latest_year].get('Diluted EPS')
                        revenue = income_stmt[latest_year].get('Total Revenue')

                        # P/E based valuation
                        if eps and 'Pe Ratio' in multiples:
                            valuations['PE'] = {
                                'median': eps * multiples['Pe Ratio']['median'],
                                'mean': eps * multiples['Pe Ratio']['mean'],
                                'min': eps * multiples['Pe Ratio']['min'],
                                'max': eps * multiples['Pe Ratio']['max']
                            }

                        # Forward P/E based valuation if available
                        if eps and 'Forward Pe' in multiples:
                            # Assume 10% growth for forward EPS
                            forward_eps = eps * 1.1
                            valuations['Forward PE'] = {
                                'median': forward_eps * multiples['Forward Pe']['median'],
                                'mean': forward_eps * multiples['Forward Pe']['mean'],
                                'min': forward_eps * multiples['Forward Pe']['min'],
                                'max': forward_eps * multiples['Forward Pe']['max']
                            }

                        # P/S based valuation
                        if revenue and 'Ps Ratio' in multiples and outstanding_shares:
                            ps_values = {}
                            for key, value in multiples['Ps Ratio'].items():
                                ps_values[key] = revenue * value / outstanding_shares
                            valuations['PS'] = ps_values

                        # Save multiple valuation results
                        valuation_results['models']['MULTIPLES'] = {
                            'multiples': multiples,
                            'valuations': valuations
                        }
                except Exception as e:
                    logger.warning(f"Failed to calculate multiple-based valuation for {ticker}: {e}")

            # Calculate Dividend Discount Model if requested
            if 'DDM' in models:
                try:
                    # Get dividend history
                    dividend_history = self.get_historical_dividends(ticker)
                    company_info = self.get_company_info(ticker)

                    if not dividend_history.empty and company_info:
                        # Calculate dividend growth rate
                        annual_dividends = dividend_history.resample('Y').sum()

                        if len(annual_dividends) >= 2:
                            # Calculate growth rates
                            growth_rates = []
                            for i in range(1, len(annual_dividends)):
                                current = annual_dividends.iloc[i]
                                previous = annual_dividends.iloc[i - 1]

                                if previous > 0:
                                    growth_rates.append((current - previous) / previous)

                            # Use average growth rate
                            dividend_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0.03

                            # Cap growth rate to be reasonable
                            dividend_growth = min(max(dividend_growth, 0.01), 0.15)

                            # Get latest annual dividend
                            latest_annual_dividend = annual_dividends.iloc[-1]

                            # Determine discount rate based on sector
                            sector = company_info.get('sector', '')
                            if sector in ['Technology', 'Communication Services']:
                                discount_rate = 0.1  # Higher risk
                            elif sector in ['Utilities', 'Consumer Staples']:
                                discount_rate = 0.07  # Lower risk
                            else:
                                discount_rate = 0.09  # Medium risk

                            # Calculate DDM value using Gordon Growth Model
                            ddm_value = latest_annual_dividend * (1 + dividend_growth) / (
                                        discount_rate - dividend_growth)

                            # Save DDM results
                            valuation_results['models']['DDM'] = {
                                'value': float(ddm_value),
                                'latest_annual_dividend': float(latest_annual_dividend),
                                'dividend_growth': float(dividend_growth),
                                'discount_rate': float(discount_rate)
                            }
                except Exception as e:
                    logger.warning(f"Failed to calculate DDM model for {ticker}: {e}")

            # Calculate Net Asset Value (NAV) if requested
            if 'NAV' in models:
                try:
                    # Get balance sheet
                    balance_sheet = self.get_financial_statements(ticker, 'balance', 'annual')
                    company_info = self.get_company_info(ticker)

                    if not balance_sheet.empty and company_info:
                        # Get latest data
                        latest_year = balance_sheet.columns[0]

                        # Extract required data
                        total_assets = balance_sheet[latest_year].get('Total Assets')
                        total_liabilities = balance_sheet[latest_year].get(
                            'Total Liabilities Net Minority Interest')
                        outstanding_shares = company_info.get('shares_outstanding', None)

                        if total_assets and total_liabilities and outstanding_shares:
                            # Calculate book value
                            book_value = total_assets - total_liabilities

                            # Calculate book value per share
                            bvps = book_value / outstanding_shares

                            # Calculate adjusted NAV with premium/discount
                            sector = company_info.get('sector', '')

                            # Sector-specific adjustments
                            if sector in ['Financials', 'Real Estate']:
                                # These sectors often trade based on book value
                                premium = 1.1  # 10% premium
                            elif sector in ['Technology', 'Healthcare']:
                                # These sectors often have significant intangible assets
                                premium = 2.0  # 100% premium
                            else:
                                premium = 1.5  # 50% premium

                            # Adjusted NAV
                            adjusted_nav = bvps * premium

                            # Save NAV results
                            valuation_results['models']['NAV'] = {
                                'value': float(adjusted_nav),
                                'book_value': float(book_value),
                                'book_value_per_share': float(bvps),
                                'premium_factor': float(premium)
                            }
                except Exception as e:
                    logger.warning(f"Failed to calculate NAV model for {ticker}: {e}")

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(valuation_results, f)
            logger.info(f"Saved {ticker} valuation models to cache")

            return valuation_results
        except Exception as e:
            logger.error(f"Error calculating valuation models for {ticker}: {e}")
            return {
                'ticker': ticker,
                'models': {},
                'price': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

    def get_risk_metrics(self, ticker: str, benchmark: str = 'SPY',
                         period: str = '1y', force_refresh: bool = False) -> Dict:
        """
        Get risk metrics for a stock (beta, volatility, VaR, etc.)

        Args:
            ticker: Stock ticker
            benchmark: Benchmark ticker (default is S&P 500)
            period: Time period ('1m', '3m', '6m', '1y', '3y', '5y')
            force_refresh: Force data refresh even if cache exists

        Returns:
            Dictionary with risk metrics
        """
        cache_file = self.cache_dir / f"{ticker}_risk_{benchmark}_{period}.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Risk metrics update weekly
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading {ticker} risk metrics from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load risk metrics cache for {ticker}: {e}")

        try:
            # Determine date range
            end_date = datetime.now().strftime('%Y-%m-%d')

            if period == '1m':
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == '3m':
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif period == '6m':
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif period == '3y':
                start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
            elif period == '5y':
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
            else:  # Default to 1y
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            # Get historical prices
            stock_data = self.get_historical_prices(ticker, start_date=start_date, end_date=end_date)
            benchmark_data = self.get_historical_prices(benchmark, start_date=start_date, end_date=end_date)

            if stock_data.empty or benchmark_data.empty:
                logger.warning(f"Missing price data for {ticker} or {benchmark}")
                return {}

            # Calculate daily returns
            stock_returns = stock_data['Adj Close'].pct_change().dropna()
            benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()

            # Align return series
            stock_returns, benchmark_returns = stock_returns.align(benchmark_returns, join='inner')

            # Initialize risk metrics
            risk_metrics = {
                'ticker': ticker,
                'benchmark': benchmark,
                'period': period,
                'observations': len(stock_returns),
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            # Calculate beta
            cov = np.cov(stock_returns, benchmark_returns)[0, 1]
            var = np.var(benchmark_returns)
            beta = cov / var if var > 0 else 1.0
            risk_metrics['beta'] = float(beta)

            # Calculate alpha
            alpha = (stock_returns.mean() - RISK_FREE_RATE/252) - beta * (benchmark_returns.mean() - RISK_FREE_RATE/252)  # Annualized

            # Calculate volatility
            volatility = stock_returns.std()
            risk_metrics['volatility_daily'] = float(volatility)
            risk_metrics['volatility_annual'] = float(volatility * np.sqrt(252))

            # Calculate Sharpe ratio
            excess_return = stock_returns.mean() - RISK_FREE_RATE / 252
            sharpe = excess_return / volatility if volatility > 0 else 0
            risk_metrics['sharpe_ratio'] = float(sharpe * np.sqrt(252))  # Annualized

            # Calculate downside deviation
            negative_returns = stock_returns[stock_returns < 0]
            downside_deviation = negative_returns.std() if not negative_returns.empty else 0
            risk_metrics['downside_deviation'] = float(downside_deviation)

            # Calculate Sortino ratio
            sortino = excess_return / downside_deviation if downside_deviation > 0 else 0
            risk_metrics['sortino_ratio'] = float(sortino * np.sqrt(252))  # Annualized

            # Calculate Value at Risk (VaR)
            confidence_levels = [0.90, 0.95, 0.99]
            var_metrics = {}

            for level in confidence_levels:
                var_value = np.percentile(stock_returns, (1 - level) * 100)
                var_metrics[f"{int(level * 100)}%"] = float(var_value)

            risk_metrics['value_at_risk'] = var_metrics

            # Calculate maximum drawdown
            cumulative_returns = (1 + stock_returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / rolling_max) - 1
            max_drawdown = drawdowns.min()
            risk_metrics['max_drawdown'] = float(max_drawdown)

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(risk_metrics, f)
            logger.info(f"Saved {ticker} risk metrics to cache")

            return risk_metrics
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {ticker}: {e}")
            return {}