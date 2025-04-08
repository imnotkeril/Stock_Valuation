import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from StockAnalysisSystem.src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('time_series')


class TimeSeriesForecaster:
    """
    Class for forecasting financial time series data using classical
    statistical methods like ARIMA, Exponential Smoothing, etc.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize time series forecaster

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader or DataLoader()
        logger.info("Initialized TimeSeriesForecaster")

    def forecast_stock_price(self, ticker: str, days_ahead: int = 30,
                             model_type: str = 'auto', **kwargs) -> Dict[str, Any]:
        """
        Forecast future stock prices

        Args:
            ticker: Company ticker symbol
            days_ahead: Number of days to forecast
            model_type: Type of model to use ('arima', 'sarima', 'exp_smoothing', 'auto')
            **kwargs: Additional parameters for specific models

        Returns:
            Dictionary with forecast results, confidence intervals, and model details
        """
        try:
            # Load historical stock price data
            price_data = self.data_loader.get_stock_price(ticker, period='1y')

            if price_data.empty:
                logger.warning(f"No price data found for {ticker}")
                return {'error': 'No price data available'}

            # Prepare data for forecasting
            price_series = price_data['Close']

            # Determine best model if auto
            if model_type == 'auto':
                model_type = self._select_best_model(price_series)
                logger.info(f"Auto-selected model: {model_type}")

            # Perform forecast based on model type
            if model_type == 'arima':
                forecast_result = self._forecast_arima(price_series, days_ahead, **kwargs)
            elif model_type == 'sarima':
                forecast_result = self._forecast_sarima(price_series, days_ahead, **kwargs)
            elif model_type == 'exp_smoothing':
                forecast_result = self._forecast_exp_smoothing(price_series, days_ahead, **kwargs)
            else:
                logger.warning(f"Unknown model type: {model_type}, using ARIMA")
                forecast_result = self._forecast_arima(price_series, days_ahead, **kwargs)

            # Add ticker and model info
            forecast_result['ticker'] = ticker
            forecast_result['model_type'] = model_type

            return forecast_result
        except Exception as e:
            logger.error(f"Error forecasting stock price for {ticker}: {e}")
            return {'error': str(e)}

    def forecast_financials(self, ticker: str, periods_ahead: int = 4,
                            statement_type: str = 'income', frequency: str = 'quarterly',
                            model_type: str = 'auto', **kwargs) -> Dict[str, Any]:
        """
        Forecast future financial statement values

        Args:
            ticker: Company ticker symbol
            periods_ahead: Number of periods to forecast
            statement_type: Type of financial statement ('income', 'balance', 'cash_flow')
            frequency: Data frequency ('quarterly', 'annual')
            model_type: Type of model to use
            **kwargs: Additional parameters for specific models

        Returns:
            Dictionary with forecast results for financial metrics
        """
        try:
            # Load financial statement data
            if statement_type == 'income':
                statement_data = self.data_loader.get_income_statement(ticker, period=frequency)
                key_metrics = ['Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EPS']
            elif statement_type == 'balance':
                statement_data = self.data_loader.get_balance_sheet(ticker, period=frequency)
                key_metrics = ['Total Assets', 'Total Liabilities', 'Total Equity']
            elif statement_type == 'cash_flow':
                statement_data = self.data_loader.get_cash_flow(ticker, period=frequency)
                key_metrics = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow']
            else:
                logger.warning(f"Unknown statement type: {statement_type}")
                return {'error': 'Invalid statement type'}

            if statement_data.empty:
                logger.warning(f"No {statement_type} statement data found for {ticker}")
                return {'error': f'No {statement_type} statement data available'}

            # Initialize results
            forecast_results = {
                'ticker': ticker,
                'statement_type': statement_type,
                'frequency': frequency,
                'forecasts': {}
            }

            # Forecast each key metric
            for metric in key_metrics:
                if metric not in statement_data.index:
                    logger.warning(f"Metric {metric} not found in {statement_type} statement for {ticker}")
                    continue

                # Get time series for this metric
                metric_series = statement_data.loc[metric]

                # Convert index to datetime if not already
                if not isinstance(metric_series.index, pd.DatetimeIndex):
                    try:
                        metric_series.index = pd.to_datetime(metric_series.index)
                    except Exception as e:
                        logger.warning(f"Could not convert index to datetime for {metric}: {e}")
                        continue

                # Sort by date (most recent first typically)
                metric_series = metric_series.sort_index()

                # Determine best model if auto
                if model_type == 'auto':
                    selected_model = self._select_best_model(metric_series)
                else:
                    selected_model = model_type

                # Perform forecast
                try:
                    if selected_model == 'arima':
                        metric_forecast = self._forecast_arima(metric_series, periods_ahead, **kwargs)
                    elif selected_model == 'sarima':
                        metric_forecast = self._forecast_sarima(metric_series, periods_ahead, **kwargs)
                    elif selected_model == 'exp_smoothing':
                        metric_forecast = self._forecast_exp_smoothing(metric_series, periods_ahead, **kwargs)
                    else:
                        metric_forecast = self._forecast_arima(metric_series, periods_ahead, **kwargs)

                    forecast_results['forecasts'][metric] = {
                        'model': selected_model,
                        'forecast_values': metric_forecast.get('forecast_values', []),
                        'confidence_intervals': metric_forecast.get('confidence_intervals', {}),
                        'forecast_dates': metric_forecast.get('forecast_dates', [])
                    }
                except Exception as e:
                    logger.warning(f"Could not forecast {metric} for {ticker}: {e}")
                    continue

            return forecast_results
        except Exception as e:
            logger.error(f"Error forecasting financials for {ticker}: {e}")
            return {'error': str(e)}

    def analyze_seasonality(self, ticker: str, metric: str = 'price',
                            period: str = '5y') -> Dict[str, Any]:
        """
        Analyze seasonality patterns in financial data

        Args:
            ticker: Company ticker symbol
            metric: Metric to analyze ('price', 'volume', 'revenue', etc.)
            period: Historical period to analyze

        Returns:
            Dictionary with seasonality analysis results
        """
        try:
            # Load appropriate data based on metric
            if metric == 'price':
                data = self.data_loader.get_stock_price(ticker, period=period)
                if data.empty:
                    return {'error': 'No price data available'}

                time_series = data['Close']
            elif metric == 'volume':
                data = self.data_loader.get_stock_price(ticker, period=period)
                if data.empty:
                    return {'error': 'No volume data available'}

                time_series = data['Volume']
            elif metric in ['revenue', 'net_income', 'operating_income']:
                # Convert to snake_case to camel case for financial statements
                metric_map = {
                    'revenue': 'Revenue',
                    'net_income': 'Net Income',
                    'operating_income': 'Operating Income'
                }

                statement_data = self.data_loader.get_income_statement(ticker, period='quarterly')
                if statement_data.empty or metric_map[metric] not in statement_data.index:
                    return {'error': f'No {metric} data available'}

                time_series = statement_data.loc[metric_map[metric]]

                # Convert index to datetime if needed
                if not isinstance(time_series.index, pd.DatetimeIndex):
                    try:
                        time_series.index = pd.to_datetime(time_series.index)
                    except Exception as e:
                        logger.warning(f"Could not convert index to datetime: {e}")
                        return {'error': 'Invalid date format in data'}
            else:
                return {'error': f'Unsupported metric: {metric}'}

            # Sort by date
            time_series = time_series.sort_index()

            # Analyze seasonality
            seasonality_result = self._analyze_time_series_seasonality(time_series)

            return {
                'ticker': ticker,
                'metric': metric,
                'seasonality': seasonality_result
            }
        except Exception as e:
            logger.error(f"Error analyzing seasonality for {ticker}: {e}")
            return {'error': str(e)}

    def _forecast_arima(self, time_series: pd.Series, periods_ahead: int,
                        order: Optional[Tuple[int, int, int]] = None, **kwargs) -> Dict[str, Any]:
        """
        Forecast using ARIMA model

        Args:
            time_series: Time series data
            periods_ahead: Number of periods to forecast
            order: ARIMA order (p,d,q)
            **kwargs: Additional parameters for ARIMA

        Returns:
            Dictionary with forecast results
        """
        # Determine model order if not provided
        if order is None:
            order = self._determine_arima_order(time_series)

        # Fit ARIMA model
        model = ARIMA(time_series, order=order)
        fitted_model = model.fit()

        # Generate forecast
        forecast = fitted_model.forecast(steps=periods_ahead)

        # Get confidence intervals (default 95%)
        conf_int = fitted_model.get_forecast(steps=periods_ahead).conf_int()

        # Generate future dates for forecasting
        last_date = time_series.index[-1]

        if isinstance(last_date, (pd.Timestamp, datetime)):
            # For time series with datetime index
            if isinstance(time_series.index, pd.DatetimeIndex) and time_series.index.freq is not None:
                # If the index has a frequency, use it
                forecast_dates = pd.date_range(start=last_date + time_series.index.freq, periods=periods_ahead,
                                               freq=time_series.index.freq)
            else:
                # Try to infer frequency
                freq = pd.infer_freq(time_series.index)
                if freq is not None:
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq=freq)
                else:
                    # Default to daily frequency if can't infer
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq='D')
        else:
            # For numeric index, just continue the sequence
            forecast_dates = range(len(time_series), len(time_series) + periods_ahead)

        # Format result
        return {
            'forecast_values': forecast.tolist(),
            'confidence_intervals': {
                'lower': conf_int.iloc[:, 0].tolist(),
                'upper': conf_int.iloc[:, 1].tolist()
            },
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        }

    def _forecast_sarima(self, time_series: pd.Series, periods_ahead: int,
                         order: Optional[Tuple[int, int, int]] = None,
                         seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Forecast using SARIMA model (seasonal ARIMA)

        Args:
            time_series: Time series data
            periods_ahead: Number of periods to forecast
            order: ARIMA order (p,d,q)
            seasonal_order: Seasonal order (P,D,Q,s)
            **kwargs: Additional parameters for SARIMA

        Returns:
            Dictionary with forecast results
        """
        # Determine model order if not provided
        if order is None:
            order = self._determine_arima_order(time_series)

        # Determine seasonal order if not provided
        if seasonal_order is None:
            # Default seasonal order (commonly used values)
            seasonal_order = (1, 1, 1, 12)  # Assuming monthly seasonality by default

        # Fit SARIMA model
        model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)

        # Generate forecast
        forecast = fitted_model.forecast(steps=periods_ahead)

        # Get confidence intervals (default 95%)
        conf_int = fitted_model.get_forecast(steps=periods_ahead).conf_int()

        # Generate future dates for forecasting
        last_date = time_series.index[-1]

        if isinstance(last_date, (pd.Timestamp, datetime)):
            # For time series with datetime index
            if isinstance(time_series.index, pd.DatetimeIndex) and time_series.index.freq is not None:
                # If the index has a frequency, use it
                forecast_dates = pd.date_range(start=last_date + time_series.index.freq, periods=periods_ahead,
                                               freq=time_series.index.freq)
            else:
                # Try to infer frequency
                freq = pd.infer_freq(time_series.index)
                if freq is not None:
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq=freq)
                else:
                    # Default to daily frequency if can't infer
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq='D')
        else:
            # For numeric index, just continue the sequence
            forecast_dates = range(len(time_series), len(time_series) + periods_ahead)

        # Format result
        return {
            'forecast_values': forecast.tolist(),
            'confidence_intervals': {
                'lower': conf_int.iloc[:, 0].tolist(),
                'upper': conf_int.iloc[:, 1].tolist()
            },
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        }

    def _forecast_exp_smoothing(self, time_series: pd.Series, periods_ahead: int,
                                seasonal: str = 'add', trend: Optional[str] = None,
                                seasonal_periods: int = 12, **kwargs) -> Dict[str, Any]:
        """
        Forecast using Exponential Smoothing

        Args:
            time_series: Time series data
            periods_ahead: Number of periods to forecast
            seasonal: Type of seasonality ('add', 'mul', None)
            trend: Type of trend ('add', 'mul', None)
            seasonal_periods: Number of periods in a season
            **kwargs: Additional parameters for Exponential Smoothing

        Returns:
            Dictionary with forecast results
        """
        # Fit Exponential Smoothing model
        model = ExponentialSmoothing(
            time_series,
            seasonal=seasonal,
            trend=trend,
            seasonal_periods=seasonal_periods
        )

        fitted_model = model.fit()

        # Generate forecast
        forecast = fitted_model.forecast(periods_ahead)

        # Exponential Smoothing doesn't provide confidence intervals directly
        # We'll estimate them based on prediction error variance

        # Generate future dates for forecasting
        last_date = time_series.index[-1]

        if isinstance(last_date, (pd.Timestamp, datetime)):
            # For time series with datetime index
            if isinstance(time_series.index, pd.DatetimeIndex) and time_series.index.freq is not None:
                # If the index has a frequency, use it
                forecast_dates = pd.date_range(start=last_date + time_series.index.freq, periods=periods_ahead,
                                               freq=time_series.index.freq)
            else:
                # Try to infer frequency
                freq = pd.infer_freq(time_series.index)
                if freq is not None:
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq=freq)
                else:
                    # Default to daily frequency if can't infer
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq='D')
        else:
            # For numeric index, just continue the sequence
            forecast_dates = range(len(time_series), len(time_series) + periods_ahead)

        # Format result
        return {
            'forecast_values': forecast.tolist(),
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'seasonal': seasonal,
                'trend': trend,
                'seasonal_periods': seasonal_periods,
                'params': fitted_model.params
            }
        }

    def _select_best_model(self, time_series: pd.Series) -> str:
        """
        Select the best forecasting model for the time series

        Args:
            time_series: Time series data

        Returns:
            String indicating best model type
        """
        # Perform ADF test to check stationarity
        try:
            adf_result = adfuller(time_series.dropna())
            is_stationary = adf_result[1] < 0.05  # p-value less than 0.05
        except Exception:
            is_stationary = False

        # Try to determine if there's seasonality
        try:
            # Calculate autocorrelation
            acf_values = acf(time_series.dropna(), nlags=40)

            # Check for seasonality patterns in ACF
            has_seasonality = False
            for i in range(10, min(40, len(acf_values))):
                if acf_values[i] > 0.3:  # Arbitrary threshold
                    has_seasonality = True
                    break
        except Exception:
            has_seasonality = False

        # Based on characteristics, select model
        if has_seasonality:
            return 'sarima'
        elif not is_stationary:
            return 'exp_smoothing'
        else:
            return 'arima'

    def _determine_arima_order(self, time_series: pd.Series) -> Tuple[int, int, int]:
        """
        Determine the optimal ARIMA order (p,d,q) for a time series

        Args:
            time_series: Time series data

        Returns:
            Tuple (p,d,q) representing ARIMA order
        """
        # Check stationarity to determine d
        try:
            adf_result = adfuller(time_series.dropna())
            is_stationary = adf_result[1] < 0.05  # p-value less than 0.05

            if is_stationary:
                d = 0
            else:
                # Try differencing once
                diff_series = time_series.diff().dropna()
                adf_result = adfuller(diff_series)
                if adf_result[1] < 0.05:
                    d = 1
                else:
                    # Try differencing twice
                    diff2_series = diff_series.diff().dropna()
                    d = 2
        except Exception:
            # Default value if test fails
            d = 1

        # Determine p and q using ACF and PACF
        try:
            if d > 0:
                # Work with differenced series
                diff_series = time_series.diff(d).dropna()
                acf_values = acf(diff_series, nlags=20)
                pacf_values = pacf(diff_series, nlags=20)
            else:
                acf_values = acf(time_series.dropna(), nlags=20)
                pacf_values = pacf(time_series.dropna(), nlags=20)

            # Determine p from PACF
            p = 0
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > 0.2:  # Arbitrary threshold
                    p = i
                else:
                    break

            # Determine q from ACF
            q = 0
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > 0.2:  # Arbitrary threshold
                    q = i
                else:
                    break
        except Exception:
            # Default values if test fails
            p = 1
            q = 1

        # Cap at reasonable values to avoid overfitting
        p = min(p, 5)
        q = min(q, 5)

        return (p, d, q)

    def _analyze_time_series_seasonality(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        Analyze seasonality patterns in a time series

        Args:
            time_series: Time series data

        Returns:
            Dictionary with seasonality analysis results
        """
        # Make sure the series has a datetime index
        if not isinstance(time_series.index, pd.DatetimeIndex):
            try:
                time_series.index = pd.to_datetime(time_series.index)
            except Exception:
                return {'error': 'Series does not have a valid datetime index'}

        # Extract common seasonal components
        try:
            # Add year, month, day of week, etc. to the series
            ts_df = pd.DataFrame(time_series)
            ts_df.columns = ['value']

            ts_df['year'] = ts_df.index.year
            ts_df['month'] = ts_df.index.month
            ts_df['quarter'] = ts_df.index.quarter
            ts_df['day_of_week'] = ts_df.index.dayofweek

            # Analysis by month
            monthly_avg = ts_df.groupby('month')['value'].mean()

            # Analysis by quarter
            quarterly_avg = ts_df.groupby('quarter')['value'].mean()

            # Analysis by day of week (if applicable)
            day_of_week_avg = ts_df.groupby('day_of_week')['value'].mean()

            # Seasonal strength calculation
            if len(ts_df) >= 24:  # At least 2 years of data for decomposition
                try:
                    # Decompose the time series
                    decomposition = sm.tsa.seasonal_decompose(time_series, model='additive')

                    # Calculate seasonal strength
                    seasonal_variance = np.var(decomposition.seasonal)
                    residual_variance = np.var(decomposition.resid.dropna())

                    if (seasonal_variance + residual_variance) > 0:
                        seasonal_strength = seasonal_variance / (seasonal_variance + residual_variance)
                    else:
                        seasonal_strength = 0
                except Exception:
                    seasonal_strength = None
            else:
                seasonal_strength = None

            # Create result dictionary
            result = {
                'seasonal_strength': seasonal_strength,
                'monthly_pattern': monthly_avg.to_dict(),
                'quarterly_pattern': quarterly_avg.to_dict(),
                'day_of_week_pattern': day_of_week_avg.to_dict(),
                'max_month': monthly_avg.idxmax(),
                'min_month': monthly_avg.idxmin(),
                'max_quarter': quarterly_avg.idxmax(),
                'min_quarter': quarterly_avg.idxmin(),
            }

            # Determine if there's significant seasonality
            if seasonal_strength is not None:
                result['has_seasonality'] = seasonal_strength > 0.3  # Arbitrary threshold
            else:
                # Alternative check based on month-to-month variation
                max_min_ratio = monthly_avg.max() / monthly_avg.min() if monthly_avg.min() > 0 else 1
                result['has_seasonality'] = max_min_ratio > 1.2  # Arbitrary threshold

            return result
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {'error': str(e)}