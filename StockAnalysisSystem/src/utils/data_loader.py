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