import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import UI_SETTINGS, COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer
from StockAnalysisSystem.src.models.forecasting.time_series import TimeSeriesForecaster
from StockAnalysisSystem.src.models.forecasting.ml_models import MLForecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forecasting')


def run_forecasting_page():
    """Main function to run the forecasting page"""
    st.title("Financial Forecasting")
    st.markdown("Forecast stock prices, financial metrics, and economic indicators to support investment decisions.")

    # Initialize data loader and visualizer
    data_loader = DataLoader()
    visualizer = FinancialVisualizer(theme="dark")
    time_series_forecaster = TimeSeriesForecaster()
    ml_forecaster = MLForecaster()

    # Sidebar for inputs
    with st.sidebar:
        st.header("Forecasting Parameters")

        # What to forecast
        st.subheader("Select Data to Forecast")
        forecast_options = {
            "Stock Price": "Forecast future stock prices",
            "Financial Metrics": "Forecast financial metrics like revenue, EPS, etc.",
            "Economic Indicators": "Forecast economic indicators like GDP, inflation, etc."
        }

        forecast_type = st.selectbox(
            "What would you like to forecast?",
            options=list(forecast_options.keys())
        )

        st.info(forecast_options[forecast_type])

        # Specify forecasting parameters based on selection
        if forecast_type == "Stock Price":
            # Stock selection
            ticker = st.text_input("Enter ticker symbol:", "AAPL").upper()

            # Historical data period
            hist_period_options = {
                "1 Year": 365,
                "2 Years": 2 * 365,
                "3 Years": 3 * 365,
                "5 Years": 5 * 365,
                "10 Years": 10 * 365
            }

            historical_period = st.selectbox(
                "Historical data period:",
                options=list(hist_period_options.keys()),
                index=2  # Default to 3 years
            )

            days_of_history = hist_period_options[historical_period]

            # Forecast horizon
            forecast_horizon_options = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365
            }

            forecast_horizon = st.selectbox(
                "Forecast horizon:",
                options=list(forecast_horizon_options.keys()),
                index=1  # Default to 3 months
            )

            days_to_forecast = forecast_horizon_options[forecast_horizon]

            # Forecast frequency
            forecast_frequency = st.selectbox(
                "Forecast frequency:",
                options=["Daily", "Weekly", "Monthly"],
                index=0  # Default to daily
            )

            # Forecast models
            st.subheader("Forecasting Models")

            use_arima = st.checkbox("ARIMA", value=True)
            use_exponential = st.checkbox("Exponential Smoothing", value=True)
            use_prophet = st.checkbox("Prophet", value=True)
            use_lstm = st.checkbox("LSTM Neural Network", value=False)

            st.markdown("---")

            # Advanced settings
            with st.expander("Advanced Settings", expanded=False):
                # Confidence interval
                confidence_interval = st.slider(
                    "Confidence Interval (%):",
                    min_value=50,
                    max_value=99,
                    value=95,
                    step=5
                )

                # ARIMA parameters (if selected)
                if use_arima:
                    st.subheader("ARIMA Parameters")

                    auto_arima = st.checkbox("Auto-select ARIMA parameters", value=True)

                    if not auto_arima:
                        p_param = st.slider("p (AR order):", 0, 10, 2)
                        d_param = st.slider("d (Differencing):", 0, 3, 1)
                        q_param = st.slider("q (MA order):", 0, 10, 2)
                    else:
                        p_param, d_param, q_param = None, None, None

                # Prophet parameters (if selected)
                if use_prophet:
                    st.subheader("Prophet Parameters")

                    include_seasonality = st.checkbox("Include seasonality", value=True)
                    include_holidays = st.checkbox("Include holidays", value=True)

                # LSTM parameters (if selected)
                if use_lstm:
                    st.subheader("LSTM Parameters")

                    lookback_period = st.slider("Lookback Period (days):", 7, 90, 30)
                    lstm_epochs = st.slider("Training Epochs:", 10, 500, 100)
                    lstm_batch_size = st.slider("Batch Size:", 8, 128, 32)

            # Events to include
            with st.expander("Include Events", expanded=False):
                include_earnings = st.checkbox("Earnings Announcements", value=True)
                include_dividends = st.checkbox("Dividend Payments", value=True)
                include_splits = st.checkbox("Stock Splits", value=True)

        elif forecast_type == "Financial Metrics":
            # Company selection
            ticker = st.text_input("Enter ticker symbol:", "AAPL").upper()

            # Metric selection
            financial_metrics = [
                "Revenue",
                "EBITDA",
                "Net Income",
                "EPS",
                "Operating Margin",
                "Net Margin",
                "ROE",
                "Free Cash Flow"
            ]

            selected_metrics = st.multiselect(
                "Select metrics to forecast:",
                options=financial_metrics,
                default=["Revenue", "EPS"]
            )

            # Forecast horizon
            forecast_periods = st.slider(
                "Number of periods to forecast:",
                min_value=1,
                max_value=12,
                value=4
            )

            # Forecast frequency
            forecast_frequency = st.selectbox(
                "Forecast frequency:",
                options=["Quarterly", "Annual"],
                index=0  # Default to quarterly
            )

            # Forecast models
            st.subheader("Forecasting Models")

            use_linear = st.checkbox("Linear Trend", value=True)
            use_arima = st.checkbox("ARIMA", value=True)
            use_ml = st.checkbox("Machine Learning (XGBoost)", value=True)

            # Include additional factors
            with st.expander("Include Additional Factors", expanded=False):
                include_sector = st.checkbox("Sector Performance", value=True)
                include_macro = st.checkbox("Macroeconomic Indicators", value=True)
                include_sentiment = st.checkbox("Market Sentiment", value=False)

        elif forecast_type == "Economic Indicators":
            # Indicator selection
            economic_indicators = [
                "GDP Growth",
                "Inflation (CPI)",
                "Unemployment Rate",
                "Federal Funds Rate",
                "10-Year Treasury Yield",
                "Consumer Sentiment",
                "Retail Sales",
                "Housing Starts"
            ]

            selected_indicators = st.multiselect(
                "Select indicators to forecast:",
                options=economic_indicators,
                default=["GDP Growth", "Inflation (CPI)"]
            )

            # Forecast horizon
            forecast_periods = st.slider(
                "Number of periods to forecast:",
                min_value=1,
                max_value=8,
                value=4
            )

            # Forecast frequency
            forecast_frequency = st.selectbox(
                "Forecast frequency:",
                options=["Quarterly", "Annual"],
                index=0  # Default to quarterly
            )

            # Forecast models
            st.subheader("Forecasting Models")

            use_arima = st.checkbox("ARIMA", value=True)
            use_var = st.checkbox("Vector Autoregression (VAR)", value=True)
            use_ml = st.checkbox("Machine Learning", value=True)

        # Run forecast button
        forecast_button = st.button("Run Forecast", type="primary")

    # Main content area
    if forecast_type == "Stock Price" and forecast_button:
        with st.spinner(f"Forecasting {ticker} price for the next {forecast_horizon}..."):
            try:
                # Load historical data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_of_history)).strftime('%Y-%m-%d')

                # Get historical prices
                price_data = data_loader.get_historical_prices(ticker, start_date, end_date)

                if price_data.empty:
                    st.error(f"Could not load historical price data for {ticker}.")
                    return

                # Get company info
                company_info = data_loader.get_company_info(ticker)
                company_name = company_info.get('name', ticker) if company_info else ticker

                # Prepare forecasting parameters
                forecast_params = {
                    "frequency": forecast_frequency,
                    "periods": days_to_forecast,
                    "confidence_interval": confidence_interval / 100
                }

                # ARIMA-specific parameters
                if use_arima:
                    forecast_params["arima_params"] = {
                        "auto": auto_arima,
                        "p": p_param,
                        "d": d_param,
                        "q": q_param
                    }

                # Prophet-specific parameters
                if use_prophet:
                    forecast_params["prophet_params"] = {
                        "seasonality": include_seasonality,
                        "holidays": include_holidays
                    }

                # LSTM-specific parameters
                if use_lstm:
                    forecast_params["lstm_params"] = {
                        "lookback": lookback_period,
                        "epochs": lstm_epochs,
                        "batch_size": lstm_batch_size
                    }

                # Create list of models to use
                models_to_use = []
                if use_arima:
                    models_to_use.append("arima")
                if use_exponential:
                    models_to_use.append("ets")
                if use_prophet:
                    models_to_use.append("prophet")
                if use_lstm:
                    models_to_use.append("lstm")

                forecast_params["models"] = models_to_use

                # Generate forecasts
                forecasts = time_series_forecaster.forecast_price(
                    price_data,
                    **forecast_params
                )

                if not forecasts:
                    st.error("Failed to generate forecasts. Please check your parameters and try again.")
                    return

                # Display results header
                st.header(f"Price Forecast for {company_name} ({ticker})")
                st.subheader(f"Forecast Horizon: {forecast_horizon}")

                # Create tabs for different views
                tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Statistics", "Model Details"])

                # Tab 1: Forecast Chart
                with tabs[0]:
                    st.subheader("Price Forecast")

                    # Create ensemble forecast
                    ensemble_forecast = time_series_forecaster.create_ensemble_forecast(forecasts)

                    # Plot forecast
                    fig = visualizer.plot_price_forecast(
                        price_data,
                        ensemble_forecast,
                        ticker,
                        company_name=company_name,
                        title=f"{ticker} Price Forecast - Next {forecast_horizon}",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display key metrics
                    current_price = price_data['Close'].iloc[-1]
                    forecast_price = ensemble_forecast['mean'].iloc[-1] if not ensemble_forecast.empty else None
                    forecast_change = ((forecast_price / current_price) - 1) * 100 if forecast_price else None

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${current_price:.2f}"
                        )

                    with col2:
                        st.metric(
                            f"Forecast ({forecast_horizon})",
                            f"${forecast_price:.2f}" if forecast_price else "N/A",
                            f"{forecast_change:.2f}%" if forecast_change else None,
                            delta_color="normal"
                        )

                    with col3:
                        high_estimate = ensemble_forecast['upper'].iloc[-1] if not ensemble_forecast.empty else None
                        low_estimate = ensemble_forecast['lower'].iloc[-1] if not ensemble_forecast.empty else None

                        st.metric(
                            f"Range ({confidence_interval}% CI)",
                            f"${low_estimate:.2f} - ${high_estimate:.2f}" if low_estimate and high_estimate else "N/A"
                        )

                    # Add forecast interpretation
                    if forecast_price and forecast_change:
                        if forecast_change > 10:
                            sentiment = "strongly bullish"
                            color = COLORS["primary"]  # Green
                        elif forecast_change > 5:
                            sentiment = "moderately bullish"
                            color = COLORS["success"]  # Light green
                        elif forecast_change > 0:
                            sentiment = "slightly bullish"
                            color = COLORS["warning"]  # Yellow
                        elif forecast_change > -5:
                            sentiment = "slightly bearish"
                            color = COLORS["warning"]  # Yellow
                        elif forecast_change > -10:
                            sentiment = "moderately bearish"
                            color = COLORS["accent"]  # Light red
                        else:
                            sentiment = "strongly bearish"
                            color = COLORS["danger"]  # Red

                        st.markdown(
                            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                            f"<h4 style='color: #121212; text-align: center;'>Forecast Outlook: {sentiment.capitalize()}</h4>"
                            f"<p style='color: #121212; text-align: center;'>The model ensemble predicts a {forecast_change:.2f}% "
                            f"change in {ticker} price over the next {forecast_horizon}.</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Tab 2: Model Comparison
                with tabs[1]:
                    st.subheader("Model Comparison")

                    # Plot all models on one chart
                    fig = visualizer.plot_model_comparison(
                        price_data,
                        forecasts,
                        ticker,
                        title=f"{ticker} Price Forecast - Model Comparison",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display model performance metrics
                    st.subheader("Model Performance")

                    # Calculate performance metrics
                    performance_metrics = time_series_forecaster.evaluate_models(
                        price_data,
                        models_to_use
                    )

                    if performance_metrics:
                        # Create DataFrame for display
                        metrics_df = pd.DataFrame(performance_metrics)

                        # Format metrics for display
                        display_df = metrics_df.copy()

                        # Format numbers for better readability
                        for col in display_df.columns:
                            if col != "Model":
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

                        # Add styling to highlight best model
                        if len(metrics_df) > 1:
                            # Find best model (lowest RMSE)
                            best_model_idx = metrics_df['RMSE'].idxmin()

                            def highlight_best(s):
                                """Highlight the best model row"""
                                is_best = [i == best_model_idx for i in range(len(s))]
                                return ['background-color: #74f174; color: #121212' if v else '' for v in is_best]

                            # Apply styling
                            styled_df = display_df.style.apply(highlight_best, axis=0)

                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.dataframe(display_df, use_container_width=True)
                    else:
                        st.warning("Performance metrics not available.")

                    # Tab 3: Forecast Statistics
                with tabs[2]:
                    st.subheader("Forecast Statistics")

                    # Display summary statistics for the ensemble forecast
                    if ensemble_forecast is not None and not ensemble_forecast.empty:
                        # Calculate statistics
                        current_price = price_data['Close'].iloc[-1]
                        forecast_price = ensemble_forecast['mean'].iloc[-1]
                        forecast_change = ((forecast_price / current_price) - 1) * 100
                        min_forecast = ensemble_forecast['lower'].min()
                        max_forecast = ensemble_forecast['upper'].max()
                        range_width = max_forecast - min_forecast
                        range_percent = (range_width / current_price) * 100

                        # Create a DataFrame for statistics
                        stats_data = [
                            {"Statistic": "Current Price", "Value": f"${current_price:.2f}"},
                            {"Statistic": f"Forecast Price ({forecast_horizon})", "Value": f"${forecast_price:.2f}"},
                            {"Statistic": "Forecast Change", "Value": f"{forecast_change:.2f}%"},
                            {"Statistic": "Minimum Forecast", "Value": f"${min_forecast:.2f}"},
                            {"Statistic": "Maximum Forecast", "Value": f"${max_forecast:.2f}"},
                            {"Statistic": "Forecast Range", "Value": f"${range_width:.2f} ({range_percent:.2f}%)"},
                            {"Statistic": "Confidence Interval", "Value": f"{confidence_interval}%"},
                            {"Statistic": "Forecast Models Used", "Value": ", ".join(models_to_use)}
                        ]

                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)

                        # Show probability distribution
                        st.subheader("Forecast Probability Distribution")

                        fig = visualizer.plot_forecast_distribution(
                            ensemble_forecast,
                            current_price=current_price,
                            title=f"{ticker} Price Forecast Distribution",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Add probability insights
                        prob_up = time_series_forecaster.probability_of_price_increase(ensemble_forecast)
                        prob_up_5 = time_series_forecaster.probability_of_price_change(ensemble_forecast, 0.05)
                        prob_down_5 = time_series_forecaster.probability_of_price_change(ensemble_forecast, -0.05)

                        st.markdown("### Probability Analysis")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Probability of Price Increase", f"{prob_up * 100:.1f}%")

                        with col2:
                            st.metric("Probability of >5% Increase", f"{prob_up_5 * 100:.1f}%")

                        with col3:
                            st.metric("Probability of >5% Decrease", f"{prob_down_5 * 100:.1f}%")
                    else:
                        st.warning("Forecast statistics not available.")

                    # Tab 4: Model Details
                with tabs[3]:
                    st.subheader("Model Details")

                    # Create tabs for each model
                    if forecasts:
                        model_tabs = st.tabs([m.upper() for m in models_to_use])

                        for i, model in enumerate(models_to_use):
                            with model_tabs[i]:
                                if model in forecasts:
                                    model_forecast = forecasts[model]

                                    # Display model description
                                    model_descriptions = {
                                        "arima": """
                                                            **ARIMA (AutoRegressive Integrated Moving Average)** is a statistical model for time series forecasting.
                                                            It combines three components:
                                                            - AR (AutoRegressive): Uses the relationship between an observation and a number of lagged observations.
                                                            - I (Integrated): Applies differencing to make the time series stationary.
                                                            - MA (Moving Average): Uses the dependency between an observation and residual errors from a moving average model.

                                                            ARIMA is particularly useful for stock prices as it can capture trends and some patterns in the data.
                                                            """,
                                        "ets": """
                                                            **Exponential Smoothing (ETS)** is a time series forecasting method that gives more weight to recent observations.
                                                            It uses weighted averages where the weights decrease exponentially as observations get older.

                                                            ETS can handle data with trends and seasonal patterns, making it useful for stock price prediction.
                                                            This model is generally good at capturing short to medium-term patterns.
                                                            """,
                                        "prophet": """
                                                            **Prophet** is a forecasting model developed by Facebook that is designed for time series with strong seasonal effects.
                                                            It uses a decomposable time series model with three main components:
                                                            - Trend: Models non-periodic changes
                                                            - Seasonality: Models periodic changes (daily, weekly, yearly)
                                                            - Holidays: Accounts for holiday effects

                                                            Prophet is robust to missing data, shifts in trends, and typically handles outliers well.
                                                            """,
                                        "lstm": """
                                                            **LSTM (Long Short-Term Memory)** is a type of recurrent neural network capable of learning long-term dependencies.
                                                            Unlike other forecasting methods, LSTM can:
                                                            - Remember patterns over long sequences
                                                            - Capture complex non-linear relationships
                                                            - Learn from a large amount of historical data

                                                            LSTMs are particularly powerful for financial time series that may have complex patterns not easily captured by traditional statistical methods.
                                                            """
                                    }

                                    st.markdown(model_descriptions.get(model, "No description available."))

                                    # Display model parameters
                                    st.subheader("Model Parameters")

                                    if model == "arima" and "parameters" in model_forecast:
                                        arima_params = model_forecast["parameters"]
                                        st.write(f"- p (AR order): {arima_params.get('p', 'Auto')}")
                                        st.write(f"- d (Differencing): {arima_params.get('d', 'Auto')}")
                                        st.write(f"- q (MA order): {arima_params.get('q', 'Auto')}")

                                    elif model == "prophet" and "parameters" in model_forecast:
                                        prophet_params = model_forecast["parameters"]
                                        st.write(f"- Seasonality: {prophet_params.get('seasonality', True)}")
                                        st.write(f"- Change points: {prophet_params.get('n_changepoints', 25)}")
                                        st.write(f"- Included holidays: {prophet_params.get('holidays', False)}")

                                    elif model == "lstm" and "parameters" in model_forecast:
                                        lstm_params = model_forecast["parameters"]
                                        st.write(f"- Lookback period: {lstm_params.get('lookback', 30)} days")
                                        st.write(f"- Epochs: {lstm_params.get('epochs', 100)}")
                                        st.write(f"- Batch size: {lstm_params.get('batch_size', 32)}")
                                        st.write(f"- Layers: {lstm_params.get('layers', [50, 50])}")

                                    # Plot individual model forecast
                                    st.subheader("Model Forecast")

                                    if "forecast" in model_forecast:
                                        model_data = model_forecast["forecast"]

                                        fig = visualizer.plot_individual_forecast(
                                            price_data,
                                            model_data,
                                            ticker,
                                            model_name=model.upper(),
                                            title=f"{ticker} - {model.upper()} Forecast",
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"No forecast data available for {model.upper()} model.")
                                else:
                                    st.warning(f"{model.upper()} model did not produce a valid forecast.")

                except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")
                logger.error(f"Error in stock price forecasting: {str(e)}")

            elif forecast_type == "Financial Metrics" and forecast_button:
            with st.spinner(f"Forecasting financial metrics for {ticker}..."):
                try:
                    # Get company info
                    company_info = data_loader.get_company_info(ticker)
                    company_name = company_info.get('name', ticker) if company_info else ticker

                    # Load financial statements
                    income_stmt = data_loader.get_financial_statements(ticker, 'income', forecast_frequency.lower())
                    balance_sheet = data_loader.get_financial_statements(ticker, 'balance', forecast_frequency.lower())
                    cash_flow = data_loader.get_financial_statements(ticker, 'cash', forecast_frequency.lower())

                    if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
                        st.error(f"Could not load financial statement data for {ticker}.")
                        return

                    # Prepare financial data for forecasting
                    financial_data = {}

                    # Map selected metrics to financial statements
                    statement_mapping = {
                        "Revenue": {"statement": "income", "field": "Total Revenue"},
                        "EBITDA": {"statement": "income", "field": "EBITDA"},
                        "Net Income": {"statement": "income", "field": "Net Income"},
                        "EPS": {"statement": "income", "field": "Diluted EPS"},
                        "Operating Margin": {"statement": "calculated",
                                             "fields": ["Operating Income", "Total Revenue"]},
                        "Net Margin": {"statement": "calculated", "fields": ["Net Income", "Total Revenue"]},
                        "ROE": {"statement": "calculated", "fields": ["Net Income", "Total Stockholder Equity"]},
                        "Free Cash Flow": {"statement": "cash", "field": "Free Cash Flow"}
                    }

                    # Extract data for selected metrics
                    for metric in selected_metrics:
                        if metric in statement_mapping:
                            mapping = statement_mapping[metric]

                            if mapping["statement"] == "income":
                                if not income_stmt.empty and mapping["field"] in income_stmt.index:
                                    financial_data[metric] = income_stmt.loc[mapping["field"]]

                            elif mapping["statement"] == "cash":
                                if not cash_flow.empty and mapping["field"] in cash_flow.index:
                                    financial_data[metric] = cash_flow.loc[mapping["field"]]

                            elif mapping["statement"] == "calculated":
                                # Handle calculated ratios
                                if metric == "Operating Margin" and not income_stmt.empty:
                                    if all(field in income_stmt.index for field in mapping["fields"]):
                                        financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / \
                                                                 income_stmt.loc[mapping["fields"][1]]

                                elif metric == "Net Margin" and not income_stmt.empty:
                                    if all(field in income_stmt.index for field in mapping["fields"]):
                                        financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / \
                                                                 income_stmt.loc[mapping["fields"][1]]

                                elif metric == "ROE" and not income_stmt.empty and not balance_sheet.empty:
                                    if mapping["fields"][0] in income_stmt.index and mapping["fields"][
                                        1] in balance_sheet.index:
                                        financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / \
                                                                 balance_sheet.loc[mapping["fields"][1]]

                    # Check if we have data to forecast
                    if not financial_data:
                        st.error("Could not extract financial metric data for forecasting.")
                        return

                    # Convert financial data to DataFrames
                    financial_dfs = {metric: pd.DataFrame(data) for metric, data in financial_data.items()}

                    # Additional forecasting parameters
                    forecast_params = {
                        "frequency": forecast_frequency.lower(),
                        "periods": forecast_periods
                    }

                    # Create list of models to use
                    models_to_use = []
                    if use_linear:
                        models_to_use.append("linear")
                    if use_arima:
                        models_to_use.append("arima")
                    if use_ml:
                        models_to_use.append("xgboost")

                    forecast_params["models"] = models_to_use

                    # Additional factors
                    if include_sector:
                        forecast_params["include_sector"] = True
                        forecast_params["sector"] = company_info.get('sector')

                    if include_macro:
                        forecast_params["include_macro"] = True

                    if include_sentiment:
                        forecast_params["include_sentiment"] = True

                    # Generate forecasts for each metric
                    forecasts = {}
                    for metric, data in financial_dfs.items():
                        metric_forecast = time_series_forecaster.forecast_financial_metric(
                            data,
                            metric,
                            **forecast_params
                        )

                        if metric_forecast:
                            forecasts[metric] = metric_forecast

                    if not forecasts:
                        st.error("Failed to generate forecasts. Please check your parameters and try again.")
                        return

                    # Display results header
                    st.header(f"Financial Metrics Forecast for {company_name} ({ticker})")
                    st.subheader(f"Forecast: Next {forecast_periods} {forecast_frequency} Periods")

                    # Create tabs for each metric
                    metric_tabs = st.tabs(selected_metrics)

                    for i, metric in enumerate(selected_metrics):
                        with metric_tabs[i]:
                            if metric in forecasts:
                                # Display metric forecast
                                st.subheader(f"{metric} Forecast")

                                # Get historical and forecast data
                                historical_data = financial_dfs[metric]
                                forecast_data = forecasts[metric]

                                # Create ensemble forecast if multiple models
                                if len(models_to_use) > 1 and all(model in forecast_data for model in models_to_use):
                                    ensemble_forecast = time_series_forecaster.create_ensemble_financial_forecast(
                                        forecast_data)
                                else:
                                    # Use the first available model
                                    for model in models_to_use:
                                        if model in forecast_data:
                                            ensemble_forecast = forecast_data[model]
                                            break
                                    else:
                                        ensemble_forecast = None

                                if ensemble_forecast is not None:
                                    # Plot forecast
                                    fig = visualizer.plot_financial_metric_forecast(
                                        historical_data,
                                        ensemble_forecast,
                                        metric,
                                        title=f"{ticker} - {metric} Forecast",
                                        frequency=forecast_frequency,
                                        height=500
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Display key statistics
                                    if not historical_data.empty and not ensemble_forecast.empty:
                                        latest_value = historical_data.iloc[:, 0].iloc[-1]
                                        forecast_end = ensemble_forecast['mean'].iloc[-1]
                                        forecast_change = ((forecast_end / latest_value) - 1) * 100

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric(
                                                "Latest Value",
                                                f"${latest_value:.2f}M" if metric in ["Revenue", "EBITDA", "Net Income",
                                                                                      "Free Cash Flow"] else
                                                f"${latest_value:.2f}" if metric == "EPS" else
                                                f"{latest_value * 100:.2f}%" if metric in ["Operating Margin",
                                                                                           "Net Margin", "ROE"] else
                                                f"{latest_value:.2f}"
                                            )

                                        with col2:
                                            st.metric(
                                                f"Forecast (Period {forecast_periods})",
                                                f"${forecast_end:.2f}M" if metric in ["Revenue", "EBITDA", "Net Income",
                                                                                      "Free Cash Flow"] else
                                                f"${forecast_end:.2f}" if metric == "EPS" else
                                                f"{forecast_end * 100:.2f}%" if metric in ["Operating Margin",
                                                                                           "Net Margin", "ROE"] else
                                                f"{forecast_end:.2f}",
                                                f"{forecast_change:.2f}%",
                                                delta_color="normal"
                                            )

                                        with col3:
                                            # Forecast range
                                            high_estimate = ensemble_forecast['upper'].iloc[-1]
                                            low_estimate = ensemble_forecast['lower'].iloc[-1]

                                            if metric in ["Revenue", "EBITDA", "Net Income", "Free Cash Flow"]:
                                                range_str = f"${low_estimate:.2f}M - ${high_estimate:.2f}M"
                                            elif metric == "EPS":
                                                range_str = f"${low_estimate:.2f} - ${high_estimate:.2f}"
                                            elif metric in ["Operating Margin", "Net Margin", "ROE"]:
                                                range_str = f"{low_estimate * 100:.2f}% - {high_estimate * 100:.2f}%"
                                            else:
                                                range_str = f"{low_estimate:.2f} - {high_estimate:.2f}"

                                            st.metric(
                                                "Forecast Range",
                                                range_str
                                            )

                                    # Add forecast interpretation
                                    if forecast_change is not None:
                                        if forecast_change > 20:
                                            sentiment = "strong growth"
                                            color = COLORS["primary"]  # Green
                                        elif forecast_change > 10:
                                            sentiment = "moderate growth"
                                            color = COLORS["success"]  # Light green
                                        elif forecast_change > 0:
                                            sentiment = "slight growth"
                                            color = COLORS["warning"]  # Yellow
                                        elif forecast_change > -10:
                                            sentiment = "slight decline"
                                            color = COLORS["warning"]  # Yellow
                                        elif forecast_change > -20:
                                            sentiment = "moderate decline"
                                            color = COLORS["accent"]  # Light red
                                        else:
                                            sentiment = "significant decline"
                                            color = COLORS["danger"]  # Red

                                        st.markdown(
                                            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                                            f"<h4 style='color: #121212; text-align: center;'>Forecast: {sentiment.capitalize()}</h4>"
                                            f"<p style='color: #121212; text-align: center;'>Models predict a {forecast_change:.2f}% "
                                            f"change in {metric} over the next {forecast_periods} {forecast_frequency.lower()} periods.</p>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )

                                    # Show model comparison if multiple models
                                    if len(models_to_use) > 1 and all(
                                            model in forecast_data for model in models_to_use):
                                        st.subheader("Model Comparison")

                                        fig = visualizer.plot_financial_model_comparison(
                                            historical_data,
                                            forecast_data,
                                            metric,
                                            title=f"{ticker} - {metric} Forecast Model Comparison",
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Performance metrics
                                        st.subheader("Model Performance")

                                        metrics_data = []
                                        for model in models_to_use:
                                            if model in forecast_data and "metrics" in forecast_data[model]:
                                                model_metrics = forecast_data[model]["metrics"]
                                                metrics_data.append({
                                                    "Model": model.upper(),
                                                    "RMSE": model_metrics.get("rmse", "N/A"),
                                                    "MAE": model_metrics.get("mae", "N/A"),
                                                    "MAPE": model_metrics.get("mape", "N/A")
                                                })

                                        if metrics_data:
                                            metrics_df = pd.DataFrame(metrics_data)
                                            st.dataframe(metrics_df, use_container_width=True)
                                else:
                                    st.warning(f"No forecast generated for {metric}.")
                            else:
                                st.warning(f"No forecast data available for {metric}.")

                    # Overall financial outlook
                    st.header("Overall Financial Outlook")

                    # Analyze trends across metrics
                    trends = {}
                    for metric, forecast in forecasts.items():
                        # Get the first available model's forecast
                        for model in models_to_use:
                            if model in forecast:
                                forecast_data = forecast[model]
                                if "forecast" in forecast_data:
                                    last_historical = financial_dfs[metric].iloc[:, 0].iloc[-1]
                                    last_forecast = forecast_data["forecast"]['mean'].iloc[-1]
                                    change_pct = ((last_forecast / last_historical) - 1) * 100
                                    trends[metric] = change_pct
                                    break

                    # Create a summary of financial trends
                    if trends:
                        trends_df = pd.DataFrame(
                            [{"Metric": k, "Forecasted Change (%)": f"{v:.2f}%"} for k, v in trends.items()])

                        # Sort by forecasted change
                        trends_df = trends_df.sort_values("Forecasted Change (%)", ascending=False)

                        st.dataframe(trends_df, use_container_width=True)

                        # Generate a financial outlook summary
                        revenue_trend = trends.get("Revenue")
                        profit_trend = trends.get("Net Income") or trends.get("EPS")
                        margin_trend = trends.get("Operating Margin") or trends.get("Net Margin")

                        if revenue_trend is not None and profit_trend is not None:
                            if revenue_trend > 10 and profit_trend > 15:
                                outlook = "Strong Growth"
                                color = COLORS["primary"]  # Green
                                description = f"The company is projected to experience strong growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                            elif revenue_trend > 5 and profit_trend > 5:
                                outlook = "Moderate Growth"
                                color = COLORS["success"]  # Light green
                                description = f"The company is projected to experience moderate growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                            elif revenue_trend > 0 and profit_trend > 0:
                                outlook = "Stable Growth"
                                color = COLORS["warning"]  # Yellow
                                description = f"The company is projected to maintain stable growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                            elif revenue_trend < 0 and profit_trend < 0:
                                outlook = "Contraction"
                                color = COLORS["accent"]  # Light red
                                description = f"The company is projected to experience contraction with revenue declining by {-revenue_trend:.1f}% and profits by {-profit_trend:.1f}%."
                            else:
                                outlook = "Mixed"
                                color = COLORS["warning"]  # Yellow
                                description = f"The company is projected to show mixed performance with revenue changing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."

                            if margin_trend is not None:
                                if margin_trend > 0:
                                    description += f" Margins are expected to improve by {margin_trend:.1f}%, indicating enhanced operational efficiency."
                                else:
                                    description += f" Margins are expected to decrease by {-margin_trend:.1f}%, suggesting potential cost pressures."

                            st.markdown(
                                f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                                f"<h4 style='color: #121212; text-align: center;'>Financial Outlook: {outlook}</h4>"
                                f"<p style='color: #121212; text-align: center;'>{description}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error(f"An error occurred during financial metric forecasting: {str(e)}")
                    logger.error(f"Error in financial metric forecasting: {str(e)}")

        elif forecast_type == "Economic Indicators" and forecast_button:
        with st.spinner(f"Forecasting economic indicators..."):
            try:
                # Get economic indicator data
                indicators_data = {}

                # Map selected indicators to their sources/codes
                indicator_mapping = {
                    "GDP Growth": "GDP",
                    "Inflation (CPI)": "CPIAUCSL",
                    "Unemployment Rate": "UNRATE",
                    "Federal Funds Rate": "FEDFUNDS",
                    "10-Year Treasury Yield": "DGS10",
                    "Consumer Sentiment": "UMCSENT",
                    "Retail Sales": "RSXFS",
                    "Housing Starts": "HOUST"
                }

                # Load data for selected indicators
                for indicator in selected_indicators:
                    if indicator in indicator_mapping:
                        try:
                            indicator_code = indicator_mapping[indicator]

                            # Get indicator data
                            indicator_data = data_loader.get_macro_indicators([indicator_code])

                            if indicator_code in indicator_data and not indicator_data[indicator_code].empty:
                                indicators_data[indicator] = indicator_data[indicator_code]
                        except Exception as e:
                            logger.error(f"Error loading data for {indicator}: {str(e)}")

                if not indicators_data:
                    st.error("Could not load economic indicator data. Please try again later.")
                    return

                # Prepare forecasting parameters
                forecast_params = {
                    "frequency": forecast_frequency.lower(),
                    "periods": forecast_periods
                }

                # Create list of models to use
                models_to_use = []
                if use_arima:
                    models_to_use.append("arima")
                if use_var and len(selected_indicators) > 1:
                    models_to_use.append("var")
                if use_ml:
                    models_to_use.append("ml")

                forecast_params["models"] = models_to_use

                # Generate forecasts
                forecasts = {}
                if use_var and len(selected_indicators) > 1:
                    # If VAR is selected, perform multivariate forecasting
                    var_forecast = time_series_forecaster.forecast_economic_indicators_var(
                        indicators_data,
                        **forecast_params
                    )

                    if var_forecast:
                        for indicator in selected_indicators:
                            if indicator in var_forecast:
                                forecasts[indicator] = {"var": var_forecast[indicator]}

                # Perform individual forecasts for each indicator
                for indicator, data in indicators_data.items():
                    if indicator not in forecasts:
                        forecasts[indicator] = {}

                    if use_arima:
                        try:
                            arima_forecast = time_series_forecaster.forecast_economic_indicator_arima(
                                data,
                                indicator,
                                **forecast_params
                            )

                            if arima_forecast:
                                forecasts[indicator]["arima"] = arima_forecast
                        except Exception as e:
                            logger.error(f"Error in ARIMA forecast for {indicator}: {str(e)}")

                    if use_ml:
                        try:
                            ml_forecast = time_series_forecaster.forecast_economic_indicator_ml(
                                data,
                                indicator,
                                **forecast_params
                            )

                            if ml_forecast:
                                forecasts[indicator]["ml"] = ml_forecast
                        except Exception as e:
                            logger.error(f"Error in ML forecast for {indicator}: {str(e)}")

                if not any(forecasts.values()):
                    st.error("Failed to generate forecasts. Please check your parameters and try again.")
                    return

                    # Display results header
                    st.header("Economic Indicators Forecast")
                    st.subheader(f"Forecast: Next {forecast_periods} {forecast_frequency} Periods")

                    # Create tabs for each indicator
                    indicator_tabs = st.tabs(selected_indicators)

                    for i, indicator in enumerate(selected_indicators):
                        with indicator_tabs[i]:
                            if indicator in forecasts and forecasts[indicator]:
                                # Display indicator forecast
                                st.subheader(f"{indicator} Forecast")

                                # Get historical data
                                historical_data = indicators_data[indicator]

                                # Create ensemble forecast from available models
                                available_models = list(forecasts[indicator].keys())

                                if len(available_models) > 1:
                                    # Create ensemble from multiple models
                                    models_data = {model: forecasts[indicator][model] for model in available_models}
                                    ensemble_forecast = time_series_forecaster.create_ensemble_indicator_forecast(
                                        models_data)
                                else:
                                    # Use the only available model
                                    model = available_models[0]
                                    ensemble_forecast = forecasts[indicator][model]

                                if ensemble_forecast is not None:
                                    # Plot forecast
                                    fig = visualizer.plot_economic_indicator_forecast(
                                        historical_data,
                                        ensemble_forecast,
                                        indicator,
                                        title=f"{indicator} Forecast",
                                        frequency=forecast_frequency,
                                        height=500
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Display key statistics
                                    if not historical_data.empty and ensemble_forecast is not None:
                                        latest_value = historical_data.iloc[-1, 0]
                                        forecast_end = ensemble_forecast['mean'].iloc[-1]

                                        # Different metrics have different ways to calculate change
                                        if indicator in ["GDP Growth", "Inflation (CPI)"]:
                                            # These are already percentage change metrics
                                            forecast_change = forecast_end - latest_value
                                        else:
                                            # Calculate percentage change
                                            forecast_change = ((forecast_end / latest_value) - 1) * 100

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric(
                                                "Latest Value",
                                                f"{latest_value:.2f}%" if indicator in ["GDP Growth", "Inflation (CPI)",
                                                                                        "Unemployment Rate",
                                                                                        "Federal Funds Rate",
                                                                                        "10-Year Treasury Yield"] else
                                                f"{latest_value:.2f}"
                                            )

                                        with col2:
                                            st.metric(
                                                f"Forecast (Period {forecast_periods})",
                                                f"{forecast_end:.2f}%" if indicator in ["GDP Growth", "Inflation (CPI)",
                                                                                        "Unemployment Rate",
                                                                                        "Federal Funds Rate",
                                                                                        "10-Year Treasury Yield"] else
                                                f"{forecast_end:.2f}",
                                                f"{forecast_change:.2f}%" if indicator not in ["GDP Growth",
                                                                                               "Inflation (CPI)"] else
                                                f"{forecast_change:.2f} pp",
                                                # percentage points for already-percentage metrics
                                                delta_color="normal"
                                            )

                                        with col3:
                                            # Forecast range
                                            high_estimate = ensemble_forecast['upper'].iloc[-1]
                                            low_estimate = ensemble_forecast['lower'].iloc[-1]

                                            range_str = (f"{low_estimate:.2f}% - {high_estimate:.2f}%"
                                                         if indicator in ["GDP Growth", "Inflation (CPI)",
                                                                          "Unemployment Rate", "Federal Funds Rate",
                                                                          "10-Year Treasury Yield"]
                                                         else f"{low_estimate:.2f} - {high_estimate:.2f}")

                                            st.metric(
                                                "Forecast Range",
                                                range_str
                                            )

                                    # Add forecast interpretation
                                    if forecast_change is not None:
                                        interpretation = ""
                                        sentiment = ""
                                        color = ""

                                        # Different interpretations for different indicators
                                        if indicator == "GDP Growth":
                                            if forecast_end > 3:
                                                sentiment = "strong growth"
                                                color = COLORS["primary"]  # Green
                                            elif forecast_end > 2:
                                                sentiment = "solid growth"
                                                color = COLORS["success"]  # Light green
                                            elif forecast_end > 0:
                                                sentiment = "modest growth"
                                                color = COLORS["warning"]  # Yellow
                                            elif forecast_end > -1:
                                                sentiment = "stagnation"
                                                color = COLORS["warning"]  # Yellow
                                            elif forecast_end > -3:
                                                sentiment = "mild contraction"
                                                color = COLORS["accent"]  # Light red
                                            else:
                                                sentiment = "recession"
                                                color = COLORS["danger"]  # Red

                                            interpretation = f"The model predicts {forecast_end:.2f}% GDP growth."

                                        elif indicator == "Inflation (CPI)":
                                            if forecast_end > 5:
                                                sentiment = "high inflation"
                                                color = COLORS["danger"]  # Red
                                            elif forecast_end > 3:
                                                sentiment = "elevated inflation"
                                                color = COLORS["accent"]  # Light red
                                            elif forecast_end > 2:
                                                sentiment = "moderate inflation"
                                                color = COLORS["warning"]  # Yellow
                                            elif forecast_end > 0:
                                                sentiment = "low inflation"
                                                color = COLORS["success"]  # Light green
                                            else:
                                                sentiment = "deflation"
                                                color = COLORS["danger"]  # Red

                                            interpretation = f"The model predicts {forecast_end:.2f}% inflation."

                                        elif indicator == "Unemployment Rate":
                                            if forecast_end > 8:
                                                sentiment = "high unemployment"
                                                color = COLORS["danger"]  # Red
                                            elif forecast_end > 6:
                                                sentiment = "elevated unemployment"
                                                color = COLORS["accent"]  # Light red
                                            elif forecast_end > 4:
                                                sentiment = "moderate unemployment"
                                                color = COLORS["warning"]  # Yellow
                                            else:
                                                sentiment = "low unemployment"
                                                color = COLORS["primary"]  # Green

                                            interpretation = f"The model predicts {forecast_end:.2f}% unemployment rate."

                                        else:
                                            # Generic interpretation for other indicators
                                            if forecast_change > 20:
                                                sentiment = "significant increase"
                                                color = COLORS["primary"]  # Green or red depending on context
                                            elif forecast_change > 10:
                                                sentiment = "moderate increase"
                                                color = COLORS["success"]  # Light green
                                            elif forecast_change > 0:
                                                sentiment = "slight increase"
                                                color = COLORS["warning"]  # Yellow
                                            elif forecast_change > -10:
                                                sentiment = "slight decrease"
                                                color = COLORS["warning"]  # Yellow
                                            elif forecast_change > -20:
                                                sentiment = "moderate decrease"
                                                color = COLORS["accent"]  # Light red
                                            else:
                                                sentiment = "significant decrease"
                                                color = COLORS["danger"]  # Red

                                            interpretation = f"The model predicts a {forecast_change:.2f}% change over the next {forecast_periods} {forecast_frequency.lower()} periods."

                                        st.markdown(
                                            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                                            f"<h4 style='color: #121212; text-align: center;'>Forecast: {sentiment.capitalize()}</h4>"
                                            f"<p style='color: #121212; text-align: center;'>{interpretation}</p>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )

                                    # Show model comparison if multiple models
                                    if len(available_models) > 1:
                                        st.subheader("Model Comparison")

                                        models_data = {model: forecasts[indicator][model] for model in available_models}

                                        fig = visualizer.plot_indicator_model_comparison(
                                            historical_data,
                                            models_data,
                                            indicator,
                                            title=f"{indicator} Forecast Model Comparison",
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning(f"No forecast generated for {indicator}.")
                            else:
                                st.warning(f"No forecast data available for {indicator}.")

                    # Overall economic outlook
                    st.header("Overall Economic Outlook")

                    # Analyze trends across indicators
                    trends = {}
                    for indicator, forecast_models in forecasts.items():
                        if forecast_models:
                            # Use the first available model
                            model = list(forecast_models.keys())[0]
                            forecast_data = forecast_models[model]

                            if forecast_data is not None:
                                last_historical = indicators_data[indicator].iloc[-1, 0]
                                last_forecast = forecast_data['mean'].iloc[-1]

                                if indicator in ["GDP Growth", "Inflation (CPI)"]:
                                    # These are already percentage change metrics
                                    change = last_forecast - last_historical
                                else:
                                    # Calculate percentage change
                                    change = ((last_forecast / last_historical) - 1) * 100

                                trends[indicator] = (last_forecast, change)

                    # Create a summary of economic trends
                    if trends:
                        summary_data = []

                        for indicator, (forecast_value, change) in trends.items():
                            if indicator in ["GDP Growth", "Inflation (CPI)", "Unemployment Rate", "Federal Funds Rate",
                                             "10-Year Treasury Yield"]:
                                value_str = f"{forecast_value:.2f}%"
                            else:
                                value_str = f"{forecast_value:.2f}"

                            if indicator in ["GDP Growth", "Inflation (CPI)"]:
                                change_str = f"{change:.2f} pp"  # percentage points
                            else:
                                change_str = f"{change:.2f}%"

                            summary_data.append({
                                "Indicator": indicator,
                                "Forecasted Value": value_str,
                                "Change": change_str
                            })

                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)

                        # Generate an economic outlook summary
                        gdp_forecast = trends.get("GDP Growth", (None, None))[0]
                        inflation_forecast = trends.get("Inflation (CPI)", (None, None))[0]
                        unemployment_forecast = trends.get("Unemployment Rate", (None, None))[0]

                        # Determine economic scenario
                        if gdp_forecast is not None and inflation_forecast is not None:
                            if gdp_forecast > 3 and inflation_forecast < 3:
                                scenario = "Strong Growth, Low Inflation"
                                color = COLORS["primary"]  # Green
                                description = f"The economy is projected to experience strong growth ({gdp_forecast:.1f}%) with low inflation ({inflation_forecast:.1f}%)."
                            elif gdp_forecast > 2 and inflation_forecast < 4:
                                scenario = "Moderate Growth, Controlled Inflation"
                                color = COLORS["success"]  # Light green
                                description = f"The economy is projected to grow at a moderate pace ({gdp_forecast:.1f}%) with controlled inflation ({inflation_forecast:.1f}%)."
                            elif gdp_forecast > 0 and inflation_forecast < 6:
                                scenario = "Modest Growth, Elevated Inflation"
                                color = COLORS["warning"]  # Yellow
                                description = f"The economy is projected to grow modestly ({gdp_forecast:.1f}%) with somewhat elevated inflation ({inflation_forecast:.1f}%)."
                            elif gdp_forecast > 0 and inflation_forecast >= 6:
                                scenario = "Stagflation Risk"
                                color = COLORS["accent"]  # Light red
                                description = f"The economy is showing signs of stagflation with modest growth ({gdp_forecast:.1f}%) but high inflation ({inflation_forecast:.1f}%)."
                            elif gdp_forecast <= 0 and inflation_forecast > 4:
                                scenario = "Recession with Inflation"
                                color = COLORS["danger"]  # Red
                                description = f"The economy is projected to contract ({gdp_forecast:.1f}%) while experiencing high inflation ({inflation_forecast:.1f}%)."
                            elif gdp_forecast <= 0 and inflation_forecast <= 2:
                                scenario = "Recession with Low Inflation"
                                color = COLORS["danger"]  # Red
                                description = f"The economy is projected to contract ({gdp_forecast:.1f}%) with low inflation ({inflation_forecast:.1f}%)."
                            else:
                                scenario = "Mixed Economic Signals"
                                color = COLORS["warning"]  # Yellow
                                description = f"The economy is showing mixed signals with {gdp_forecast:.1f}% growth and {inflation_forecast:.1f}% inflation."

                            if unemployment_forecast is not None:
                                description += f" Unemployment is projected to be {unemployment_forecast:.1f}%."

                            st.markdown(
                                f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                                f"<h4 style='color: #121212; text-align: center;'>Economic Outlook: {scenario}</h4>"
                                f"<p style='color: #121212; text-align: center;'>{description}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # Add investment implications
                            st.subheader("Investment Implications")

                            if scenario == "Strong Growth, Low Inflation" or scenario == "Moderate Growth, Controlled Inflation":
                                st.markdown("""
                                            **Favorable environment for:**
                                            - Equities, particularly growth stocks
                                            - Cyclical sectors (Technology, Consumer Discretionary, Industrials)
                                            - High-yield corporate bonds

                                            **Less favorable for:**
                                            - Defensive sectors (Utilities, Consumer Staples)
                                            - Long-term government bonds
                                            - Gold and other safe-haven assets
                                            """)
                            elif scenario == "Modest Growth, Elevated Inflation":
                                st.markdown("""
                                            **Favorable environment for:**
                                            - Value stocks over growth stocks
                                            - Sectors with pricing power (Energy, Materials, Financials)
                                            - Inflation-protected securities (TIPS)
                                            - Real assets (commodities, real estate)

                                            **Less favorable for:**
                                            - Long-duration assets (growth stocks, long-term bonds)
                                            - Companies with low pricing power
                                            """)
                            elif scenario == "Stagflation Risk":
                                st.markdown("""
                                            **Favorable environment for:**
                                            - Commodities and real assets
                                            - Energy sector
                                            - Gold and precious metals
                                            - Short-duration TIPS

                                            **Less favorable for:**
                                            - Most equities, especially growth stocks
                                            - Fixed income securities
                                            - Cyclical consumer sectors
                                            """)
                            elif "Recession" in scenario:
                                st.markdown("""
                                            **Favorable environment for:**
                                            - Defensive sectors (Utilities, Consumer Staples, Healthcare)
                                            - Quality companies with strong balance sheets
                                            - Government bonds
                                            - Cash and cash equivalents

                                            **Less favorable for:**
                                            - Cyclical sectors
                                            - Small cap stocks
                                            - High-yield bonds
                                            - Companies with high debt levels
                                            """)
                            else:
                                st.markdown("""
                                            **Consider a balanced approach:**
                                            - Diversification across asset classes
                                            - Quality companies with strong balance sheets
                                            - Mix of growth and value stocks
                                            - Combination of cyclical and defensive sectors
                                            - Laddered bond portfolio
                                            """)

                except Exception as e:
                st.error(f"An error occurred during economic indicator forecasting: {str(e)}")
                logger.error(f"Error in economic indicator forecasting: {str(e)}")

        # For direct execution
        if __name__ == "__main__":
            run_forecasting_page()
            performance_metrics = time_series_forecaster.evaluate_models(
                price_data,
                models_to_use
            )

            if performance_metrics:
                # Create DataFrame for display
                metrics_df = pd.DataFrame(performance_metrics)

                # Format metrics for display
                display_df = metrics_df.copy()

                # Format numbers for better readability
                for col in display_df.columns:
                    if col != "Model":
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

                # Add styling to highlight best model
                if len(metrics_df) > 1:
                    # Find best model (lowest RMSE)
                    best_model_idx = metrics_df['RMSE'].idxmin()

                    def highlight_best(s):
                        """Highlight the best model row"""
                        is_best = [i == best_model_idx for i in range(len(s))]
                        return ['background-color: #74f174; color: #121212' if v else '' for v in is_best]

                    # Apply styling
                    styled_df = display_df.style.apply(highlight_best, axis=0)

                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("Performance metrics not available.")

            # Tab 3: Forecast Statistics
        with tabs[2]:
            st.subheader("Forecast Statistics")

            # Display summary statistics for the ensemble forecast
            if ensemble_forecast is not None and not ensemble_forecast.empty:
                # Calculate statistics
                current_price = price_data['Close'].iloc[-1]
                forecast_price = ensemble_forecast['mean'].iloc[-1]
                forecast_change = ((forecast_price / current_price) - 1) * 100
                min_forecast = ensemble_forecast['lower'].min()
                max_forecast = ensemble_forecast['upper'].max()
                range_width = max_forecast - min_forecast
                range_percent = (range_width / current_price) * 100

                # Create a DataFrame for statistics
                stats_data = [
                    {"Statistic": "Current Price", "Value": f"${current_price:.2f}"},
                    {"Statistic": f"Forecast Price ({forecast_horizon})", "Value": f"${forecast_price:.2f}"},
                    {"Statistic": "Forecast Change", "Value": f"{forecast_change:.2f}%"},
                    {"Statistic": "Minimum Forecast", "Value": f"${min_forecast:.2f}"},
                    {"Statistic": "Maximum Forecast", "Value": f"${max_forecast:.2f}"},
                    {"Statistic": "Forecast Range", "Value": f"${range_width:.2f} ({range_percent:.2f}%)"},
                    {"Statistic": "Confidence Interval", "Value": f"{confidence_interval}%"},
                    {"Statistic": "Forecast Models Used", "Value": ", ".join(models_to_use)}
                ]

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                # Show probability distribution
                st.subheader("Forecast Probability Distribution")

                fig = visualizer.plot_forecast_distribution(
                    ensemble_forecast,
                    current_price=current_price,
                    title=f"{ticker} Price Forecast Distribution",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add probability insights
                prob_up = time_series_forecaster.probability_of_price_increase(ensemble_forecast)
                prob_up_5 = time_series_forecaster.probability_of_price_change(ensemble_forecast, 0.05)
                prob_down_5 = time_series_forecaster.probability_of_price_change(ensemble_forecast, -0.05)

                st.markdown("### Probability Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Probability of Price Increase", f"{prob_up * 100:.1f}%")

                with col2:
                    st.metric("Probability of >5% Increase", f"{prob_up_5 * 100:.1f}%")

                with col3:
                    st.metric("Probability of >5% Decrease", f"{prob_down_5 * 100:.1f}%")
            else:
                st.warning("Forecast statistics not available.")

            # Tab 4: Model Details
        with tabs[3]:
            st.subheader("Model Details")

            # Create tabs for each model
            if forecasts:
                model_tabs = st.tabs([m.upper() for m in models_to_use])

                for i, model in enumerate(models_to_use):
                    with model_tabs[i]:
                        if model in forecasts:
                            model_forecast = forecasts[model]

                            # Display model description
                            model_descriptions = {
                                "arima": """
                                                        **ARIMA (AutoRegressive Integrated Moving Average)** is a statistical model for time series forecasting.
                                                        It combines three components:
                                                        - AR (AutoRegressive): Uses the relationship between an observation and a number of lagged observations.
                                                        - I (Integrated): Applies differencing to make the time series stationary.
                                                        - MA (Moving Average): Uses the dependency between an observation and residual errors from a moving average model.

                                                        ARIMA is particularly useful for stock prices as it can capture trends and some patterns in the data.
                                                        """,
                                "ets": """
                                                        **Exponential Smoothing (ETS)** is a time series forecasting method that gives more weight to recent observations.
                                                        It uses weighted averages where the weights decrease exponentially as observations get older.

                                                        ETS can handle data with trends and seasonal patterns, making it useful for stock price prediction.
                                                        This model is generally good at capturing short to medium-term patterns.
                                                        """,
                                "prophet": """
                                                        **Prophet** is a forecasting model developed by Facebook that is designed for time series with strong seasonal effects.
                                                        It uses a decomposable time series model with three main components:
                                                        - Trend: Models non-periodic changes
                                                        - Seasonality: Models periodic changes (daily, weekly, yearly)
                                                        - Holidays: Accounts for holiday effects

                                                        Prophet is robust to missing data, shifts in trends, and typically handles outliers well.
                                                        """,
                                "lstm": """
                                                        **LSTM (Long Short-Term Memory)** is a type of recurrent neural network capable of learning long-term dependencies.
                                                        Unlike other forecasting methods, LSTM can:
                                                        - Remember patterns over long sequences
                                                        - Capture complex non-linear relationships
                                                        - Learn from a large amount of historical data

                                                        LSTMs are particularly powerful for financial time series that may have complex patterns not easily captured by traditional statistical methods.
                                                        """
                            }

                            st.markdown(model_descriptions.get(model, "No description available."))

                            # Display model parameters
                            st.subheader("Model Parameters")

                            if model == "arima" and "parameters" in model_forecast:
                                arima_params = model_forecast["parameters"]
                                st.write(f"- p (AR order): {arima_params.get('p', 'Auto')}")
                                st.write(f"- d (Differencing): {arima_params.get('d', 'Auto')}")
                                st.write(f"- q (MA order): {arima_params.get('q', 'Auto')}")

                            elif model == "prophet" and "parameters" in model_forecast:
                                prophet_params = model_forecast["parameters"]
                                st.write(f"- Seasonality: {prophet_params.get('seasonality', True)}")
                                st.write(f"- Change points: {prophet_params.get('n_changepoints', 25)}")
                                st.write(f"- Included holidays: {prophet_params.get('holidays', False)}")

                            elif model == "lstm" and "parameters" in model_forecast:
                                lstm_params = model_forecast["parameters"]
                                st.write(f"- Lookback period: {lstm_params.get('lookback', 30)} days")
                                st.write(f"- Epochs: {lstm_params.get('epochs', 100)}")
                                st.write(f"- Batch size: {lstm_params.get('batch_size', 32)}")
                                st.write(f"- Layers: {lstm_params.get('layers', [50, 50])}")

                            # Plot individual model forecast
                            st.subheader("Model Forecast")

                            if "forecast" in model_forecast:
                                model_data = model_forecast["forecast"]

                                fig = visualizer.plot_individual_forecast(
                                    price_data,
                                    model_data,
                                    ticker,
                                    model_name=model.upper(),
                                    title=f"{ticker} - {model.upper()} Forecast",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No forecast data available for {model.upper()} model.")
                        else:
                            st.warning(f"{model.upper()} model did not produce a valid forecast.")

        except Exception as e:
        st.error(f"An error occurred during forecasting: {str(e)}")
        logger.error(f"Error in stock price forecasting: {str(e)}")

    elif forecast_type == "Financial Metrics" and forecast_button:
        with st.spinner(f"Forecasting financial metrics for {ticker}..."):
            try:
                # Get company info
                company_info = data_loader.get_company_info(ticker)
                company_name = company_info.get('name', ticker) if company_info else ticker

                # Load financial statements
                income_stmt = data_loader.get_financial_statements(ticker, 'income', forecast_frequency.lower())
                balance_sheet = data_loader.get_financial_statements(ticker, 'balance', forecast_frequency.lower())
                cash_flow = data_loader.get_financial_statements(ticker, 'cash', forecast_frequency.lower())

                if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
                    st.error(f"Could not load financial statement data for {ticker}.")
                    return

                # Prepare financial data for forecasting
                financial_data = {}

                # Map selected metrics to financial statements
                statement_mapping = {
                    "Revenue": {"statement": "income", "field": "Total Revenue"},
                    "EBITDA": {"statement": "income", "field": "EBITDA"},
                    "Net Income": {"statement": "income", "field": "Net Income"},
                    "EPS": {"statement": "income", "field": "Diluted EPS"},
                    "Operating Margin": {"statement": "calculated", "fields": ["Operating Income", "Total Revenue"]},
                    "Net Margin": {"statement": "calculated", "fields": ["Net Income", "Total Revenue"]},
                    "ROE": {"statement": "calculated", "fields": ["Net Income", "Total Stockholder Equity"]},
                    "Free Cash Flow": {"statement": "cash", "field": "Free Cash Flow"}
                }

                # Extract data for selected metrics
                for metric in selected_metrics:
                    if metric in statement_mapping:
                        mapping = statement_mapping[metric]

                        if mapping["statement"] == "income":
                            if not income_stmt.empty and mapping["field"] in income_stmt.index:
                                financial_data[metric] = income_stmt.loc[mapping["field"]]

                        elif mapping["statement"] == "cash":
                            if not cash_flow.empty and mapping["field"] in cash_flow.index:
                                financial_data[metric] = cash_flow.loc[mapping["field"]]

                        elif mapping["statement"] == "calculated":
                            # Handle calculated ratios
                            if metric == "Operating Margin" and not income_stmt.empty:
                                if all(field in income_stmt.index for field in mapping["fields"]):
                                    financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / income_stmt.loc[
                                        mapping["fields"][1]]

                            elif metric == "Net Margin" and not income_stmt.empty:
                                if all(field in income_stmt.index for field in mapping["fields"]):
                                    financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / income_stmt.loc[
                                        mapping["fields"][1]]

                            elif metric == "ROE" and not income_stmt.empty and not balance_sheet.empty:
                                if mapping["fields"][0] in income_stmt.index and mapping["fields"][
                                    1] in balance_sheet.index:
                                    financial_data[metric] = income_stmt.loc[mapping["fields"][0]] / balance_sheet.loc[
                                        mapping["fields"][1]]

                # Check if we have data to forecast
                if not financial_data:
                    st.error("Could not extract financial metric data for forecasting.")
                    return

                # Convert financial data to DataFrames
                financial_dfs = {metric: pd.DataFrame(data) for metric, data in financial_data.items()}

                # Additional forecasting parameters
                forecast_params = {
                    "frequency": forecast_frequency.lower(),
                    "periods": forecast_periods
                }

                # Create list of models to use
                models_to_use = []
                if use_linear:
                    models_to_use.append("linear")
                if use_arima:
                    models_to_use.append("arima")
                if use_ml:
                    models_to_use.append("xgboost")

                forecast_params["models"] = models_to_use

                # Additional factors
                if include_sector:
                    forecast_params["include_sector"] = True
                    forecast_params["sector"] = company_info.get('sector')

                if include_macro:
                    forecast_params["include_macro"] = True

                if include_sentiment:
                    forecast_params["include_sentiment"] = True

                # Generate forecasts for each metric
                forecasts = {}
                for metric, data in financial_dfs.items():
                    metric_forecast = time_series_forecaster.forecast_financial_metric(
                        data,
                        metric,
                        **forecast_params
                    )

                    if metric_forecast:
                        forecasts[metric] = metric_forecast

                if not forecasts:
                    st.error("Failed to generate forecasts. Please check your parameters and try again.")
                    return

                # Display results header
                st.header(f"Financial Metrics Forecast for {company_name} ({ticker})")
                st.subheader(f"Forecast: Next {forecast_periods} {forecast_frequency} Periods")

                # Create tabs for each metric
                metric_tabs = st.tabs(selected_metrics)

                for i, metric in enumerate(selected_metrics):
                    with metric_tabs[i]:
                        if metric in forecasts:
                            # Display metric forecast
                            st.subheader(f"{metric} Forecast")

                            # Get historical and forecast data
                            historical_data = financial_dfs[metric]
                            forecast_data = forecasts[metric]

                            # Create ensemble forecast if multiple models
                            if len(models_to_use) > 1 and all(model in forecast_data for model in models_to_use):
                                ensemble_forecast = time_series_forecaster.create_ensemble_financial_forecast(
                                    forecast_data)
                            else:
                                # Use the first available model
                                for model in models_to_use:
                                    if model in forecast_data:
                                        ensemble_forecast = forecast_data[model]
                                        break
                                else:
                                    ensemble_forecast = None

                            if ensemble_forecast is not None:
                                # Plot forecast
                                fig = visualizer.plot_financial_metric_forecast(
                                    historical_data,
                                    ensemble_forecast,
                                    metric,
                                    title=f"{ticker} - {metric} Forecast",
                                    frequency=forecast_frequency,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Display key statistics
                                if not historical_data.empty and not ensemble_forecast.empty:
                                    latest_value = historical_data.iloc[:, 0].iloc[-1]
                                    forecast_end = ensemble_forecast['mean'].iloc[-1]
                                    forecast_change = ((forecast_end / latest_value) - 1) * 100

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric(
                                            "Latest Value",
                                            f"${latest_value:.2f}M" if metric in ["Revenue", "EBITDA", "Net Income",
                                                                                  "Free Cash Flow"] else
                                            f"${latest_value:.2f}" if metric == "EPS" else
                                            f"{latest_value * 100:.2f}%" if metric in ["Operating Margin", "Net Margin",
                                                                                       "ROE"] else
                                            f"{latest_value:.2f}"
                                        )

                                    with col2:
                                        st.metric(
                                            f"Forecast (Period {forecast_periods})",
                                            f"${forecast_end:.2f}M" if metric in ["Revenue", "EBITDA", "Net Income",
                                                                                  "Free Cash Flow"] else
                                            f"${forecast_end:.2f}" if metric == "EPS" else
                                            f"{forecast_end * 100:.2f}%" if metric in ["Operating Margin", "Net Margin",
                                                                                       "ROE"] else
                                            f"{forecast_end:.2f}",
                                            f"{forecast_change:.2f}%",
                                            delta_color="normal"
                                        )

                                    with col3:
                                        # Forecast range
                                        high_estimate = ensemble_forecast['upper'].iloc[-1]
                                        low_estimate = ensemble_forecast['lower'].iloc[-1]

                                        if metric in ["Revenue", "EBITDA", "Net Income", "Free Cash Flow"]:
                                            range_str = f"${low_estimate:.2f}M - ${high_estimate:.2f}M"
                                        elif metric == "EPS":
                                            range_str = f"${low_estimate:.2f} - ${high_estimate:.2f}"
                                        elif metric in ["Operating Margin", "Net Margin", "ROE"]:
                                            range_str = f"{low_estimate * 100:.2f}% - {high_estimate * 100:.2f}%"
                                        else:
                                            range_str = f"{low_estimate:.2f} - {high_estimate:.2f}"

                                        st.metric(
                                            "Forecast Range",
                                            range_str
                                        )

                                # Add forecast interpretation
                                if forecast_change is not None:
                                    if forecast_change > 20:
                                        sentiment = "strong growth"
                                        color = COLORS["primary"]  # Green
                                    elif forecast_change > 10:
                                        sentiment = "moderate growth"
                                        color = COLORS["success"]  # Light green
                                    elif forecast_change > 0:
                                        sentiment = "slight growth"
                                        color = COLORS["warning"]  # Yellow
                                    elif forecast_change > -10:
                                        sentiment = "slight decline"
                                        color = COLORS["warning"]  # Yellow
                                    elif forecast_change > -20:
                                        sentiment = "moderate decline"
                                        color = COLORS["accent"]  # Light red
                                    else:
                                        sentiment = "significant decline"
                                        color = COLORS["danger"]  # Red

                                    st.markdown(
                                        f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                                        f"<h4 style='color: #121212; text-align: center;'>Forecast: {sentiment.capitalize()}</h4>"
                                        f"<p style='color: #121212; text-align: center;'>Models predict a {forecast_change:.2f}% "
                                        f"change in {metric} over the next {forecast_periods} {forecast_frequency.lower()} periods.</p>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )

                                # Show model comparison if multiple models
                                if len(models_to_use) > 1 and all(model in forecast_data for model in models_to_use):
                                    st.subheader("Model Comparison")

                                    fig = visualizer.plot_financial_model_comparison(
                                        historical_data,
                                        forecast_data,
                                        metric,
                                        title=f"{ticker} - {metric} Forecast Model Comparison",
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Performance metrics
                                    st.subheader("Model Performance")

                                    metrics_data = []
                                    for model in models_to_use:
                                        if model in forecast_data and "metrics" in forecast_data[model]:
                                            model_metrics = forecast_data[model]["metrics"]
                                            metrics_data.append({
                                                "Model": model.upper(),
                                                "RMSE": model_metrics.get("rmse", "N/A"),
                                                "MAE": model_metrics.get("mae", "N/A"),
                                                "MAPE": model_metrics.get("mape", "N/A")
                                            })

                                    if metrics_data:
                                        metrics_df = pd.DataFrame(metrics_data)
                                        st.dataframe(metrics_df, use_container_width=True)
                            else:
                                st.warning(f"No forecast generated for {metric}.")
                        else:
                            st.warning(f"No forecast data available for {metric}.")

                # Overall financial outlook
                st.header("Overall Financial Outlook")

                # Analyze trends across metrics
                trends = {}
                for metric, forecast in forecasts.items():
                    # Get the first available model's forecast
                    for model in models_to_use:
                        if model in forecast:
                            forecast_data = forecast[model]
                            if "forecast" in forecast_data:
                                last_historical = financial_dfs[metric].iloc[:, 0].iloc[-1]
                                last_forecast = forecast_data["forecast"]['mean'].iloc[-1]
                                change_pct = ((last_forecast / last_historical) - 1) * 100
                                trends[metric] = change_pct
                                break

                # Create a summary of financial trends
                if trends:
                    trends_df = pd.DataFrame(
                        [{"Metric": k, "Forecasted Change (%)": f"{v:.2f}%"} for k, v in trends.items()])

                    # Sort by forecasted change
                    trends_df = trends_df.sort_values("Forecasted Change (%)", ascending=False)

                    st.dataframe(trends_df, use_container_width=True)

                    # Generate a financial outlook summary
                    revenue_trend = trends.get("Revenue")
                    profit_trend = trends.get("Net Income") or trends.get("EPS")
                    margin_trend = trends.get("Operating Margin") or trends.get("Net Margin")

                    if revenue_trend is not None and profit_trend is not None:
                        if revenue_trend > 10 and profit_trend > 15:
                            outlook = "Strong Growth"
                            color = COLORS["primary"]  # Green
                            description = f"The company is projected to experience strong growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                        elif revenue_trend > 5 and profit_trend > 5:
                            outlook = "Moderate Growth"
                            color = COLORS["success"]  # Light green
                            description = f"The company is projected to experience moderate growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                        elif revenue_trend > 0 and profit_trend > 0:
                            outlook = "Stable Growth"
                            color = COLORS["warning"]  # Yellow
                            description = f"The company is projected to maintain stable growth with revenue increasing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."
                        elif revenue_trend < 0 and profit_trend < 0:
                            outlook = "Contraction"
                            color = COLORS["accent"]  # Light red
                            description = f"The company is projected to experience contraction with revenue declining by {-revenue_trend:.1f}% and profits by {-profit_trend:.1f}%."
                        else:
                            outlook = "Mixed"
                            color = COLORS["warning"]  # Yellow
                            description = f"The company is projected to show mixed performance with revenue changing by {revenue_trend:.1f}% and profits by {profit_trend:.1f}%."

                        if margin_trend is not None:
                            if margin_trend > 0:
                                description += f" Margins are expected to improve by {margin_trend:.1f}%, indicating enhanced operational efficiency."
                            else:
                                description += f" Margins are expected to decrease by {-margin_trend:.1f}%, suggesting potential cost pressures."

                        st.markdown(
                            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                            f"<h4 style='color: #121212; text-align: center;'>Financial Outlook: {outlook}</h4>"
                            f"<p style='color: #121212; text-align: center;'>{description}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"An error occurred during financial metric forecasting: {str(e)}")
                logger.error(f"Error in financial metric forecasting: {str(e)}")

    elif forecast_type == "Economic Indicators" and forecast_button:
        with st.spinner(f"Forecasting economic indicators..."):
            try:
                # Get economic indicator data
                indicators_data = {}

                # Map selected indicators to their sources/codes
                indicator_mapping = {
                    "GDP Growth": "GDP",
                    "Inflation (CPI)": "CPIAUCSL",
                    "Unemployment Rate": "UNRATE",
                    "Federal Funds Rate": "FEDFUNDS",
                    "10-Year Treasury Yield": "DGS10",
                    "Consumer Sentiment": "UMCSENT",
                    "Retail Sales": "RSXFS",
                    "Housing Starts": "HOUST"
                }

                # Load data for selected indicators
                for indicator in selected_indicators:
                    if indicator in indicator_mapping:
                        try:
                            indicator_code = indicator_mapping[indicator]

                            # Get indicator data
                            indicator_data = data_loader.get_macro_indicators([indicator_code])

                            if indicator_code in indicator_data and not indicator_data[indicator_code].empty:
                                indicators_data[indicator] = indicator_data[indicator_code]
                        except Exception as e:
                            logger.error(f"Error loading data for {indicator}: {str(e)}")

                if not indicators_data:
                    st.error("Could not load economic indicator data. Please try again later.")
                    return

                # Prepare forecasting parameters
                forecast_params = {
                    "frequency": forecast_frequency.lower(),
                    "periods": forecast_periods
                }

                # Create list of models to use
                models_to_use = []
                if use_arima:
                    models_to_use.append("arima")
                if use_var and len(selected_indicators) > 1:
                    models_to_use.append("var")
                if use_ml:
                    models_to_use.append("ml")

                forecast_params["models"] = models_to_use

                # Generate forecasts
                forecasts = {}
                if use_var and len(selected_indicators) > 1:
                    # If VAR is selected, perform multivariate forecasting
                    var_forecast = time_series_forecaster.forecast_economic_indicators_var(
                        indicators_data,
                        **forecast_params
                    )

                    if var_forecast:
                        for indicator in selected_indicators:
                            if indicator in var_forecast:
                                forecasts[indicator] = {"var": var_forecast[indicator]}

                # Perform individual forecasts for each indicator
                for indicator, data in indicators_data.items():
                    if indicator not in forecasts:
                        forecasts[indicator] = {}

                    if use_arima:
                        try:
                            arima_forecast = time_series_forecaster.forecast_economic_indicator_arima(
                                data,
                                indicator,
                                **forecast_params
                            )

                            if arima_forecast:
                                forecasts[indicator]["arima"] = arima_forecast
                        except Exception as e:
                            logger.error(f"Error in ARIMA forecast for {indicator}: {str(e)}")

                    if use_ml:
                        try:
                            ml_forecast = time_series_forecaster.forecast_economic_indicator_ml(
                                data,
                                indicator,
                                **forecast_params
                            )

                            if ml_forecast:
                                forecasts[indicator]["ml"] = ml_forecast
                        except Exception as e:
                            logger.error(f"Error in ML forecast for {indicator}: {str(e)}")

                if not any(forecasts.values()):
                    st.error("Failed to generate forecasts. Please check your parameters and try again.")
                    return