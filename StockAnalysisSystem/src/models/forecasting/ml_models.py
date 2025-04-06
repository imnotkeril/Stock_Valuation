import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Commented out for now - can be enabled if needed
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, LSTM, Dropout
#     from tensorflow.keras.optimizers import Adam
#     TENSORFLOW_AVAILABLE = True
# except ImportError:
#     TENSORFLOW_AVAILABLE = False

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_models')


class MachineLearningForecaster:
    """
    Class for forecasting financial data using machine learning models
    like Random Forest, Gradient Boosting, and LSTM networks.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize ML forecaster

        Args:
            data_loader: DataLoader instance for fetching financial data
        """
        self.data_loader = data_loader or DataLoader()

        # Cache for trained models
        self.trained_models = {}
        logger.info("Initialized MachineLearningForecaster")

    def forecast_stock_price(self, ticker: str, days_ahead: int = 30,
                             model_type: str = 'random_forest',
                             feature_engineering: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Forecast future stock prices using ML models

        Args:
            ticker: Company ticker symbol
            days_ahead: Number of days to forecast
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'lstm')
            feature_engineering: Whether to perform feature engineering
            **kwargs: Additional parameters for specific models

        Returns:
            Dictionary with forecast results, confidence intervals, and model details
        """
        try:
            # Load historical stock price data
            price_data = self.data_loader.get_stock_price(ticker, period='2y')

            if price_data.empty:
                logger.warning(f"No price data found for {ticker}")
                return {'error': 'No price data available'}

            # Feature engineering
            if feature_engineering:
                features_df = self._create_price_features(price_data)
            else:
                features_df = price_data.copy()
                features_df['target'] = features_df['Close'].shift(-1)  # Next day's price

            # Drop NaN values
            features_df = features_df.dropna()

            # Prepare data for training
            X = features_df.drop(['target'], axis=1)
            y = features_df['target']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train model based on type
            if model_type == 'random_forest':
                forecast_result = self._forecast_random_forest(X_train, X_test, y_train, y_test, days_ahead, **kwargs)
            elif model_type == 'gradient_boosting':
                forecast_result = self._forecast_gradient_boosting(X_train, X_test, y_train, y_test, days_ahead,
                                                                   **kwargs)
            elif model_type == 'linear':
                forecast_result = self._forecast_linear(X_train, X_test, y_train, y_test, days_ahead, **kwargs)
            # elif model_type == 'lstm' and TENSORFLOW_AVAILABLE:
            #     forecast_result = self._forecast_lstm(X_train, X_test, y_train, y_test, days_ahead, **kwargs)
            else:
                logger.warning(f"Unknown model type: {model_type}, using random forest")
                forecast_result = self._forecast_random_forest(X_train, X_test, y_train, y_test, days_ahead, **kwargs)

            # Add ticker and model info
            forecast_result['ticker'] = ticker
            forecast_result['model_type'] = model_type

            return forecast_result
        except Exception as e:
            logger.error(f"Error forecasting stock price for {ticker} with ML: {e}")
            return {'error': str(e)}

    def forecast_financials(self, ticker: str, periods_ahead: int = 4,
                            statement_type: str = 'income', frequency: str = 'quarterly',
                            model_type: str = 'random_forest', **kwargs) -> Dict[str, Any]:
        """
        Forecast future financial statement values using ML models

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

            # Load additional features for prediction
            macro_data = self._get_macroeconomic_features(statement_data.index)
            sector_data = self._get_sector_features(ticker, statement_data.index)

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

                # Create features for this metric
                try:
                    features_df = self._create_financial_features(metric_series, macro_data, sector_data)

                    # Prepare data for training
                    if features_df.empty or len(features_df) < 8:  # Need enough data points
                        logger.warning(f"Not enough data points for {metric}")
                        continue

                    X = features_df.drop(['target'], axis=1)
                    y = features_df['target']

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # Train model based on type
                    if model_type == 'random_forest':
                        metric_forecast = self._forecast_random_forest(X_train, X_test, y_train, y_test, periods_ahead,
                                                                       **kwargs)
                    elif model_type == 'gradient_boosting':
                        metric_forecast = self._forecast_gradient_boosting(X_train, X_test, y_train, y_test,
                                                                           periods_ahead, **kwargs)
                    elif model_type == 'linear':
                        metric_forecast = self._forecast_linear(X_train, X_test, y_train, y_test, periods_ahead,
                                                                **kwargs)
                    # elif model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                    #     metric_forecast = self._forecast_lstm(X_train, X_test, y_train, y_test, periods_ahead, **kwargs)
                    else:
                        metric_forecast = self._forecast_random_forest(X_train, X_test, y_train, y_test, periods_ahead,
                                                                       **kwargs)

                    forecast_results['forecasts'][metric] = {
                        'model': model_type,
                        'forecast_values': metric_forecast.get('forecast_values', []),
                        'confidence_intervals': metric_forecast.get('confidence_intervals', {}),
                        'forecast_dates': metric_forecast.get('forecast_dates', [])
                    }
                except Exception as e:
                    logger.warning(f"Could not forecast {metric} for {ticker}: {e}")
                    continue

            return forecast_results
        except Exception as e:
            logger.error(f"Error forecasting financials for {ticker} with ML: {e}")
            return {'error': str(e)}

    def _create_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature set for price prediction

        Args:
            price_data: DataFrame with price data

        Returns:
            DataFrame with engineered features
        """
        df = price_data.copy()

        # Target variable (next day's close price)
        df['target'] = df['Close'].shift(-1)

        # Technical indicators

        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Price differences and ratios
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change()
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close_Diff'] = df['Close'] - df['Open']

        # Volatility measures
        df['Volatility_5d'] = df['Price_Change_Pct'].rolling(window=5).std()
        df['Volatility_10d'] = df['Price_Change_Pct'].rolling(window=10).std()

        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

        # Lagged features
        for i in range(1, 6):
            df[f'Close_Lag{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag{i}'] = df['Volume'].shift(i)

        # Date features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter

        # One-hot encode categorical features
        day_dummies = pd.get_dummies(df['Day_of_Week'], prefix='Day')
        month_dummies = pd.get_dummies(df['Month'], prefix='Month')

        # Combine all features
        df = pd.concat([df, day_dummies, month_dummies], axis=1)

        # Drop original data columns that aren't directly used as features
        df = df.drop(['Day_of_Week', 'Month', 'Quarter', 'Adj Close'], axis=1, errors='ignore')

        return df

    def _create_financial_features(self, metric_series: pd.Series,
                                   macro_data: Optional[pd.DataFrame] = None,
                                   sector_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create feature set for financial metric prediction

        Args:
            metric_series: Series with financial metric
            macro_data: DataFrame with macroeconomic data
            sector_data: DataFrame with sector performance data

        Returns:
            DataFrame with engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame(metric_series)
        df.columns = ['value']

        # Target variable (next period's value)
        df['target'] = df['value'].shift(-1)

        # Add time features
        df['growth_rate'] = df['value'].pct_change()
        df['growth_rate_lag1'] = df['growth_rate'].shift(1)
        df['growth_rate_lag2'] = df['growth_rate'].shift(2)

        # Add seasonal features
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else None

        # One-hot encode quarter if available
        if df['quarter'] is not None:
            quarter_dummies = pd.get_dummies(df['quarter'], prefix='Quarter')
            df = pd.concat([df, quarter_dummies], axis=1)

        # Add macroeconomic features if available
        if macro_data is not None and not macro_data.empty:
            # Align indices
            macro_data = macro_data.reindex(df.index, method='ffill')

            # Join with main DataFrame
            df = df.join(macro_data, how='left')

        # Add sector features if available
        if sector_data is not None and not sector_data.empty:
            # Align indices
            sector_data = sector_data.reindex(df.index, method='ffill')

            # Join with main DataFrame
            df = df.join(sector_data, how='left')

        # Drop rows with NaN values
        df = df.dropna()

        # Drop non-feature columns
        df = df.drop(['year', 'quarter'], axis=1, errors='ignore')

        return df

    def _get_macroeconomic_features(self, dates_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Get macroeconomic features for given dates

        Args:
            dates_index: DatetimeIndex for which to get macro data

        Returns:
            DataFrame with macroeconomic features
        """
        # In a real implementation, this would fetch data from an economic API
        # For now, we'll return sample data

        # Create empty DataFrame with the given index
        macro_df = pd.DataFrame(index=dates_index)

        # Add sample features
        start_date = min(dates_index)
        date_range = pd.date_range(start=start_date, periods=len(dates_index) * 2, freq='D')

        # Create sample GDP growth data
        gdp_growth = pd.Series(
            np.random.normal(2.5, 0.5, len(date_range)),
            index=date_range
        )

        # Create sample inflation data
        inflation = pd.Series(
            np.random.normal(2.0, 0.3, len(date_range)),
            index=date_range
        )

        # Create sample interest rate data
        interest_rate = pd.Series(
            np.random.normal(3.0, 0.2, len(date_range)),
            index=date_range
        )

        # Reindex to match the original dates index
        macro_df['GDP_Growth'] = gdp_growth.reindex(dates_index, method='ffill')
        macro_df['Inflation'] = inflation.reindex(dates_index, method='ffill')
        macro_df['Interest_Rate'] = interest_rate.reindex(dates_index, method='ffill')

        return macro_df

    def _get_sector_features(self, ticker: str, dates_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Get sector performance features for given dates

        Args:
            ticker: Company ticker symbol
            dates_index: DatetimeIndex for which to get sector data

        Returns:
            DataFrame with sector performance features
        """
        # In a real implementation, this would fetch sector data from an API
        # For now, we'll return sample data

        # Create empty DataFrame with the given index
        sector_df = pd.DataFrame(index=dates_index)

        # Add sample features
        start_date = min(dates_index)
        date_range = pd.date_range(start=start_date, periods=len(dates_index) * 2, freq='D')

        # Create sample sector return data
        sector_return = pd.Series(
            np.random.normal(8.0, 2.0, len(date_range)),
            index=date_range
        )

        # Create sample sector volatility data
        sector_volatility = pd.Series(
            np.random.normal(15.0, 3.0, len(date_range)),
            index=date_range
        )

        # Reindex to match the original dates index
        sector_df['Sector_Return'] = sector_return.reindex(dates_index, method='ffill')
        sector_df['Sector_Volatility'] = sector_volatility.reindex(dates_index, method='ffill')

        return sector_df

    def _forecast_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                periods_ahead: int, **kwargs) -> Dict[str, Any]:
        """
        Forecast using Random Forest regressor

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            periods_ahead: Number of periods to forecast
            **kwargs: Additional parameters for Random Forest

        Returns:
            Dictionary with forecast results
        """
        # Extract model parameters
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)

        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate future features for forecasting
        future_features = self._generate_future_features(X_train, X_test, periods_ahead)

        # Forecast future values
        forecast_values = []
        for i in range(periods_ahead):
            if i < len(future_features):
                prediction = model.predict([future_features.iloc[i]])[0]
                forecast_values.append(prediction)
            else:
                # If we can't generate features, use the last prediction
                forecast_values.append(forecast_values[-1])

        # Generate confidence intervals using prediction intervals
        # Note: Random Forest doesn't provide direct prediction intervals
        # We'll estimate them based on the test set errors

        # Calculate standard deviation of errors
        errors = y_test - y_pred
        error_std = np.std(errors)

        # Create confidence intervals (assuming normal distribution of errors)
        lower_bounds = [val - 1.96 * error_std for val in forecast_values]
        upper_bounds = [val + 1.96 * error_std for val in forecast_values]

        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates(X_test.index, periods_ahead)

        # Format result
        return {
            'forecast_values': forecast_values,
            'confidence_intervals': {
                'lower': lower_bounds,
                'upper': upper_bounds
            },
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
        }

    def _forecast_gradient_boosting(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                    y_train: pd.Series, y_test: pd.Series,
                                    periods_ahead: int, **kwargs) -> Dict[str, Any]:
        """
        Forecast using Gradient Boosting regressor

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            periods_ahead: Number of periods to forecast
            **kwargs: Additional parameters for Gradient Boosting

        Returns:
            Dictionary with forecast results
        """
        # Extract model parameters
        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        max_depth = kwargs.get('max_depth', 3)

        # Train Gradient Boosting model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate future features for forecasting
        future_features = self._generate_future_features(X_train, X_test, periods_ahead)

        # Forecast future values
        forecast_values = []
        for i in range(periods_ahead):
            if i < len(future_features):
                prediction = model.predict([future_features.iloc[i]])[0]
                forecast_values.append(prediction)
            else:
                # If we can't generate features, use the last prediction
                forecast_values.append(forecast_values[-1])

        # Generate confidence intervals
        # Gradient Boosting provides quantile regression capabilities
        lower_bounds = []
        upper_bounds = []

        # For simplicity, we'll use the same approach as in Random Forest
        errors = y_test - y_pred
        error_std = np.std(errors)

        lower_bounds = [val - 1.96 * error_std for val in forecast_values]
        upper_bounds = [val + 1.96 * error_std for val in forecast_values]

        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates(X_test.index, periods_ahead)

        # Format result
        return {
            'forecast_values': forecast_values,
            'confidence_intervals': {
                'lower': lower_bounds,
                'upper': upper_bounds
            },
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
        }

    def _forecast_linear(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series,
                         periods_ahead: int, **kwargs) -> Dict[str, Any]:
        """
        Forecast using Linear Regression or Ridge/Lasso

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            periods_ahead: Number of periods to forecast
            **kwargs: Additional parameters for linear models

        Returns:
            Dictionary with forecast results
        """
        # Extract model parameters
        model_variant = kwargs.get('model_variant', 'linear')
        alpha = kwargs.get('alpha', 1.0)  # Regularization strength for Ridge/Lasso

        # Select model based on variant
        if model_variant == 'ridge':
            model = Ridge(alpha=alpha, random_state=42)
        elif model_variant == 'lasso':
            model = Lasso(alpha=alpha, random_state=42)
        else:
            model = LinearRegression()

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate future features for forecasting
        future_features = self._generate_future_features(X_train, X_test, periods_ahead)

        # Forecast future values
        forecast_values = []
        for i in range(periods_ahead):
            if i < len(future_features):
                future_scaled = scaler.transform([future_features.iloc[i]])
                prediction = model.predict(future_scaled)[0]
                forecast_values.append(prediction)
            else:
                # If we can't generate features, use the last prediction
                forecast_values.append(forecast_values[-1])

        # Generate confidence intervals
        # Linear models don't directly provide prediction intervals
        # We'll estimate them based on the test set errors
        errors = y_test - y_pred
        error_std = np.std(errors)

        lower_bounds = [val - 1.96 * error_std for val in forecast_values]
        upper_bounds = [val + 1.96 * error_std for val in forecast_values]

        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates(X_test.index, periods_ahead)

        # Format result
        return {
            'forecast_values': forecast_values,
            'confidence_intervals': {
                'lower': lower_bounds,
                'upper': upper_bounds
            },
            'forecast_dates': [str(date) for date in forecast_dates],
            'model_details': {
                'model_variant': model_variant,
                'alpha': alpha if model_variant in ['ridge', 'lasso'] else None,
                'coefficients': dict(zip(X_train.columns, model.coef_)) if hasattr(model, 'coef_') else {},
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
        }

    # def _forecast_lstm(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
    #                   y_train: pd.Series, y_test: pd.Series,
    #                   periods_ahead: int, **kwargs) -> Dict[str, Any]:
    #     """
    #     Forecast using LSTM neural network
    #
    #     Note: This requires TensorFlow to be installed
    #     """
    #     # Implementation would go here if TENSORFLOW_AVAILABLE
    #     pass

    def _generate_future_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                  periods_ahead: int) -> pd.DataFrame:
        """
        Generate feature sets for future periods

        Args:
            X_train: Training features
            X_test: Testing features
            periods_ahead: Number of periods to generate

        Returns:
            DataFrame with generated future features
        """
        # This is a simplified approach - in real implementation,
        # this would use autoregressive features and handle different
        # feature types appropriately

        # Start with the last test features
        last_features = X_test.iloc[-1:].copy()
        all_future_features = [last_features]

        # Generate features for future periods
        for i in range(periods_ahead - 1):
            next_features = last_features.copy()

            # Update features based on simple rules
            # This is just a placeholder - real implementation would be more sophisticated
            for col in next_features.columns:
                if col.startswith('MA') or col.endswith('MA'):
                    # Keep moving averages constant
                    pass
                elif col.startswith('Price_Change') or col.endswith('Change'):
                    # Set changes to average of last few periods
                    next_features[col] = X_test[col].mean()
                elif col.startswith('Close_Lag') or col.endswith('Lag'):
                    # Update lags
                    lag_num = int(col.split('Lag')[1]) if 'Lag' in col else 1
                    lag_col_base = col.split('Lag')[0] + 'Lag'

                    if lag_num > 1 and f"{lag_col_base}{lag_num - 1}" in next_features.columns:
                        next_features[col] = next_features[f"{lag_col_base}{lag_num - 1}"]
                elif col.startswith('Day_') or col.startswith('Month_'):
                    # One-hot encoded day/month - cycle through days/months
                    # For simplicity, we'll just keep these constant
                    pass
                elif col in ['Volume', 'High', 'Low', 'Open', 'Close']:
                    # Keep these close to the last known values
                    # In a real implementation, these would be predicted more carefully
                    next_features[col] = last_features[col] * (1 + np.random.normal(0, 0.01))

            all_future_features.append(next_features)
            last_features = next_features

        # Combine all future feature sets
        return pd.concat(all_future_features)

    def _generate_forecast_dates(self, test_index: pd.DatetimeIndex, periods_ahead: int) -> List[Any]:
        """
        Generate dates for forecast periods

        Args:
            test_index: Index of test data
            periods_ahead: Number of periods to generate

        Returns:
            List of dates for forecast periods
        """
        if len(test_index) == 0:
            return [f"Period {i + 1}" for i in range(periods_ahead)]

        last_date = test_index[-1]

        if isinstance(last_date, (pd.Timestamp, datetime)):
            # For time series with datetime index
            if isinstance(test_index, pd.DatetimeIndex) and test_index.freq is not None:
                # If the index has a frequency, use it
                forecast_dates = pd.date_range(start=last_date + test_index.freq, periods=periods_ahead,
                                               freq=test_index.freq)
            else:
                # Try to infer frequency
                freq = pd.infer_freq(test_index)
                if freq is not None:
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq=freq)
                else:
                    # Default to daily frequency if can't infer
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_ahead,
                                                   freq='D')
        else:
            # For numeric index, just continue the sequence
            forecast_dates = range(len(test_index), len(test_index) + periods_ahead)

        return forecast_dates