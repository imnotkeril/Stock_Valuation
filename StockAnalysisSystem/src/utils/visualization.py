import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COLORS, VIZ_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')


class FinancialVisualizer:
    """
    Class for visualizing financial data using Plotly charts.
    Provides functions for common financial visualizations including
    price charts, financial statement trends, ratio analysis, etc.
    """

    def __init__(self, theme: str = "dark"):
        """
        Initialize visualizer with theme settings

        Args:
            theme: 'dark' or 'light' theme for charts
        """
        self.theme = theme
        self.colors = COLORS
        self.settings = VIZ_SETTINGS

        # Set default colors based on theme
        if theme == "dark":
            self.bg_color = "#121212"
            self.grid_color = "#333333"
            self.text_color = "#e0e0e0"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#e0e0e0"
            self.text_color = "#333333"

    def plot_stock_price(self,
                         price_data: pd.DataFrame,
                         ticker: str,
                         company_name: Optional[str] = None,
                         ma_periods: List[int] = [50, 200],
                         volume: bool = True,
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> go.Figure:
        """
        Create a candlestick chart with moving averages and volume

        Args:
            price_data: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            company_name: Company name for chart title
            ma_periods: List of periods for moving averages
            volume: Whether to show volume panel
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Plotly figure object
        """
        try:
            if price_data.empty:
                return self._empty_chart("No financial health data available")

            # Extract data
            overall_score = health_data['overall_score']
            components = health_data.get('components', {})
            rating = health_data.get('rating', '')

            if overall_score is None:
                return self._empty_chart("Financial health score not available")

            # Create figure with subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=["Overall Financial Health", "Profitability", "Liquidity", "Solvency"]
            )

            # Add overall score gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={"text": f"Rating: {rating}"},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": self.text_color},
                        "bar": {"color": self._get_health_score_color(overall_score)},
                        "steps": [
                            {"range": [0, 30], "color": self.colors['danger']},
                            {"range": [30, 45], "color": "#ff8c69"},  # Light red
                            {"range": [45, 60], "color": self.colors['warning']},
                            {"range": [60, 75], "color": "#9acd32"},  # Yellow-green
                            {"range": [75, 100], "color": self.colors['success']}
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.75,
                            "value": overall_score
                        }
                    }
                ),
                row=1, col=1
            )

            # Add component gauges
            if 'profitability' in components:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=components['profitability'],
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": self.text_color},
                            "bar": {"color": self._get_health_score_color(components['profitability'])},
                            "steps": [
                                {"range": [0, 30], "color": self.colors['danger']},
                                {"range": [30, 70], "color": self.colors['warning']},
                                {"range": [70, 100], "color": self.colors['success']}
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.75,
                                "value": components['profitability']
                            }
                        }
                    ),
                    row=1, col=2
                )

            if 'liquidity' in components:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=components['liquidity'],
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": self.text_color},
                            "bar": {"color": self._get_health_score_color(components['liquidity'])},
                            "steps": [
                                {"range": [0, 30], "color": self.colors['danger']},
                                {"range": [30, 70], "color": self.colors['warning']},
                                {"range": [70, 100], "color": self.colors['success']}
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.75,
                                "value": components['liquidity']
                            }
                        }
                    ),
                    row=2, col=1
                )

            if 'solvency' in components:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=components['solvency'],
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": self.text_color},
                            "bar": {"color": self._get_health_score_color(components['solvency'])},
                            "steps": [
                                {"range": [0, 30], "color": self.colors['danger']},
                                {"range": [30, 70], "color": self.colors['warning']},
                                {"range": [70, 100], "color": self.colors['success']}
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.75,
                                "value": components.get('solvency', 0)
                            }
                        }
                    ),
                    row=2, col=2
                )

            # Configure layout
            fig.update_layout(
                title="Financial Health Assessment",
                plot_bgcolor=self.bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(
                    color=self.text_color,
                    family=self.settings['font_family']
                ),
                height=height or 600,
                width=width or self.settings['width'],
                margin=dict(l=50, r=50, t=100, b=50)
            )

            return fig
            except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return self._empty_chart(f"Error creating price chart: {str(e)}")

    def plot_dcf_sensitivity(self,
                             dcf_data: Dict[str, Any],
                             height: Optional[int] = None,
                             width: Optional[int] = None) -> go.Figure:
        """
        Create a heatmap showing DCF valuation sensitivity to growth and discount rates

        Args:
            dcf_data: Dictionary with DCF valuation data including sensitivity analysis
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Plotly figure object
        """
        try:
            # Check for valid DCF data
            if not dcf_data or 'sensitivity' not in dcf_data:
                return self._empty_chart("No DCF sensitivity analysis data available")

            # Extract sensitivity data
            sensitivity = dcf_data['sensitivity']

            if not isinstance(sensitivity,
                              dict) or 'growth_rates' not in sensitivity or 'discount_rates' not in sensitivity or 'values' not in sensitivity:
                return self._empty_chart("Invalid DCF sensitivity data format")

            growth_rates = sensitivity['growth_rates']
            discount_rates = sensitivity['discount_rates']
            values = sensitivity['values']

            # Create heatmap
            fig = go.Figure()

            current_price = dcf_data.get('current_price')

            # Normalize values by current price if available
            if current_price and current_price > 0:
                # Calculate percentage difference from current price
                heatmap_values = [[(val / current_price - 1) * 100 for val in row] for row in values]

                # Create text matrix for hover information
                text_matrix = [[
                                   f"Growth: {growth_rates[i]}%<br>Discount: {discount_rates[j]}%<br>Value: ${values[i][j]:.2f}<br>Diff from current: {heatmap_values[i][j]:.2f}%"
                                   for j in range(len(discount_rates))] for i in range(len(growth_rates))]

                # Set color scale range
                max_diff = max([max(row) for row in heatmap_values])
                min_diff = min([min(row) for row in heatmap_values])
                abs_max = max(abs(max_diff), abs(min_diff))

                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_values,
                        x=discount_rates,
                        y=growth_rates,
                        colorscale='RdBu_r',  # Red for undervalued, blue for overvalued
                        zmin=-abs_max,
                        zmax=abs_max,
                        text=text_matrix,
                        hoverinfo="text",
                        colorbar=dict(
                            title="% Difference<br>From Current Price",
                            titleside="top"
                        )
                    )
                )

            else:
                # Use absolute values
                text_matrix = [
                    [f"Growth: {growth_rates[i]}%<br>Discount: {discount_rates[j]}%<br>Value: ${values[i][j]:.2f}"
                     for j in range(len(discount_rates))] for i in range(len(growth_rates))]

                fig.add_trace(
                    go.Heatmap(
                        z=values,
                        x=discount_rates,
                        y=growth_rates,
                        colorscale=[
                            [0, self.colors['danger']],
                            [0.5, self.colors['warning']],
                            [1, self.colors['success']]
                        ],
                        text=text_matrix,
                        hoverinfo="text",
                        colorbar=dict(
                            title="DCF Value ($)",
                            titleside="top"
                        )
                    )
                )

            # Mark base case
            if 'base_case' in dcf_data:
                base_case = dcf_data['base_case']
                growth_rate = base_case.get('growth_rate')
                discount_rate = base_case.get('discount_rate')

                if growth_rate is not None and discount_rate is not None:
                    # Find closest values in our arrays
                    growth_idx = min(range(len(growth_rates)), key=lambda i: abs(growth_rates[i] - growth_rate))
                    discount_idx = min(range(len(discount_rates)), key=lambda i: abs(discount_rates[i] - discount_rate))

                    fig.add_trace(
                        go.Scatter(
                            x=[discount_rates[discount_idx]],
                            y=[growth_rates[growth_idx]],
                            mode='markers',
                            marker=dict(
                                symbol='star',
                                size=15,
                                color='white',
                                line=dict(width=2, color='black')
                            ),
                            name="Base Case",
                            hoverinfo="name"
                        )
                    )

            # Add current price line if available
            if current_price and current_price > 0:
                for i, growth_rate in enumerate(growth_rates):
                    for j in range(len(discount_rates) - 1):
                        if (values[i][j] >= current_price >= values[i][j + 1]) or (
                                values[i][j] <= current_price <= values[i][j + 1]):
                            # Calculate interpolation point
                            disc1, disc2 = discount_rates[j], discount_rates[j + 1]
                            val1, val2 = values[i][j], values[i][j + 1]

                            # Linear interpolation: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                            x_interp = disc1 + (current_price - val1) * (disc2 - disc1) / (val2 - val1)

                            fig.add_trace(
                                go.Scatter(
                                    x=[x_interp],
                                    y=[growth_rate],
                                    mode='markers',
                                    marker=dict(
                                        symbol='circle',
                                        size=8,
                                        color='black'
                                    ),
                                    showlegend=False
                                )
                            )

            # Configure layout
            fig.update_layout(
                title="DCF Sensitivity Analysis",
                plot_bgcolor=self.bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(
                    color=self.text_color,
                    family=self.settings['font_family']
                ),
                xaxis=dict(
                    title="Discount Rate (%)",
                    showgrid=False
                ),
                yaxis=dict(
                    title="Growth Rate (%)",
                    showgrid=False
                ),
                height=height or self.settings['height'],
                width=width or self.settings['width'],
                margin=dict(l=50, r=50, t=80, b=50)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating DCF sensitivity chart: {e}")
            return self._empty_chart(f"Error creating DCF sensitivity chart: {str(e)}")

    # Helper methods

    def _empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()

        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=14, color=self.text_color)
        )

        fig.update_layout(
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            width=800
        )

        return fig

    def _get_health_score_color(self, score: float) -> str:
        """Get color based on health score"""
        if score >= 75:
            return self.colors['success']
        elif score >= 60:
            return "#9acd32"  # Yellow-green
        elif score >= 45:
            return self.colors['warning']
        elif score >= 30:
            return "#ff8c69"  # Light red
        else:
            return self.colors['danger']


No
price
data
available
")

# Set up figure with secondary y-axis for volume
fig = make_subplots(
    rows=2 if volume else 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.8, 0.2] if volume else [1]
)

# Add main price candlestick
fig.add_trace(
    go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name="Price",
        increasing_line_color=self.colors['success'],
        decreasing_line_color=self.colors['danger']
    ),
    row=1, col=1
)

# Add moving averages
for period in ma_periods:
    if len(price_data) >= period:
        ma = price_data['Close'].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=ma,
                name=f"{period}-day MA",
                line=dict(width=1.5)
            ),
            row=1, col=1
        )

# Add volume bars if requested
if volume and 'Volume' in price_data.columns:
    # Create color array for volume bars
    colors = np.where(price_data['Close'] >= price_data['Open'],
                      self.colors['success'], self.colors['danger'])

    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

# Set chart title
display_name = company_name if company_name else ticker
title = f"{display_name} ({ticker}) - Price Chart"

# Configure layout
fig.update_layout(
    title=title,
    plot_bgcolor=self.bg_color,
    paper_bgcolor=self.bg_color,
    font=dict(
        color=self.text_color,
        family=self.settings['font_family']
    ),
    legend=dict(
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        rangeslider=dict(visible=False),
        type='date',
        showgrid=True,
        gridcolor=self.grid_color
    ),
    yaxis=dict(
        title="Price",
        showgrid=True,
        gridcolor=self.grid_color,
        autorange=True,
        fixedrange=False
    ),
    height=height or self.settings['height'],
    width=width or self.settings['width'],
    margin=dict(l=50, r=50, t=80, b=50)
)

# Configure volume axis if applicable
if volume:
    fig.update_yaxes(
        title="Volume",
        showgrid=True,
        gridcolor=self.grid_color,
        row=2, col=1
    )

return fig
except Exception as e:
logger.error(f"Error creating price chart: {e}")
return self._empty_chart(f"Error creating price chart: {str(e)}")


def plot_financial_statement_trend(self,
                                   statement_data: pd.DataFrame,
                                   statement_type: str,
                                   metrics: List[str],
                                   height: Optional[int] = None,
                                   width: Optional[int] = None) -> go.Figure:
    """
    Plot trends from financial statements

    Args:
        statement_data: DataFrame containing financial statement data
        statement_type: Type of statement ('income', 'balance', 'cash_flow')
        metrics: List of metrics to plot
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        if statement_data.empty:
            return self._empty_chart(f"No {statement_type} statement data available")

        # Create the figure
        fig = go.Figure()

        # Format title based on statement type
        title_map = {
            'income': 'Income Statement Trends',
            'balance': 'Balance Sheet Trends',
            'cash_flow': 'Cash Flow Statement Trends'
        }
        title = title_map.get(statement_type, 'Financial Statement Trends')

        # Transpose if needed - we need periods as index
        if not isinstance(statement_data.index, pd.DatetimeIndex):
            if set(metrics).issubset(statement_data.index):
                # We have metrics in index, columns are dates
                # Extract just the metrics we want
                plot_data = statement_data.loc[metrics]
            else:
                # Try to find metrics that might be present
                available_metrics = [m for m in metrics if m in statement_data.index]
                if available_metrics:
                    plot_data = statement_data.loc[available_metrics]
                else:
                    return self._empty_chart(f"Specified metrics not found in {statement_type} statement")
        else:
            # Dates are in index, transpose to get consistent format
            plot_data = statement_data.T

        # Add traces for each metric
        for metric in metrics:
            if metric in plot_data.index:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.columns,
                        y=plot_data.loc[metric],
                        mode='lines+markers',
                        name=metric
                    )
                )

        # Configure layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Period",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating financial statement trend chart: {e}")
        return self._empty_chart(f"Error creating financial statement trend chart: {str(e)}")


def plot_financial_ratios(self,
                          ratio_data: Dict[str, Dict[str, Any]],
                          category: Optional[str] = None,
                          benchmark_data: Optional[Dict[str, Dict[str, float]]] = None,
                          height: Optional[int] = None,
                          width: Optional[int] = None) -> go.Figure:
    """
    Create a bar chart for financial ratios with benchmark comparison

    Args:
        ratio_data: Dictionary of analyzed ratios by category
        category: Specific category to plot, if None plots all categories
        benchmark_data: Optional benchmark data for comparison
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if ratio data is empty or None
        if not ratio_data:
            return self._empty_chart("No ratio data available")

        # Prepare data for plotting
        ratio_names = []
        ratio_values = []
        benchmark_values = []
        colors = []

        # Select categories to plot
        if category:
            categories = [category] if category in ratio_data else []
        else:
            categories = list(ratio_data.keys())

        # Extract data for each ratio
        for cat in categories:
            if cat in ratio_data:
                for ratio_name, ratio_info in ratio_data[cat].items():
                    ratio_names.append(f"{cat}: {ratio_name}")

                    # Get ratio value
                    if 'value' in ratio_info:
                        ratio_values.append(ratio_info['value'])
                    else:
                        ratio_values.append(None)

                    # Get benchmark value
                    if 'benchmark' in ratio_info and ratio_info['benchmark'] is not None:
                        benchmark_values.append(ratio_info['benchmark'])
                    elif benchmark_data and cat in benchmark_data and ratio_name in benchmark_data[cat]:
                        benchmark_values.append(benchmark_data[cat][ratio_name])
                    else:
                        benchmark_values.append(None)

                    # Set color based on assessment
                    if 'color' in ratio_info:
                        colors.append(ratio_info['color'])
                    else:
                        colors.append(self.colors['primary'])

        # If no data found
        if not ratio_names:
            return self._empty_chart("No ratios found for the specified category")

        # Create figure
        fig = go.Figure()

        # Add ratio bars
        fig.add_trace(
            go.Bar(
                x=ratio_names,
                y=ratio_values,
                name='Company Ratio',
                marker_color=colors
            )
        )

        # Add benchmark bars if available
        if any(benchmark_values):
            fig.add_trace(
                go.Bar(
                    x=ratio_names,
                    y=benchmark_values,
                    name='Industry Benchmark',
                    marker_color=self.colors['secondary'],
                    opacity=0.7
                )
            )

        # Configure layout
        title = f"Financial Ratios - {category}" if category else "Financial Ratios"
        fig.update_layout(
            title=title,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Ratio",
                showgrid=True,
                gridcolor=self.grid_color,
                tickangle=45
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            barmode='group',
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=150)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating financial ratios chart: {e}")
        return self._empty_chart(f"Error creating financial ratios chart: {str(e)}")


def plot_bankruptcy_risk(self,
                         risk_data: Dict[str, Any],
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> go.Figure:
    """
    Create a gauge chart to visualize bankruptcy risk

    Args:
        risk_data: Dictionary with bankruptcy model results
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid risk data
        if not risk_data or 'models' not in risk_data:
            return self._empty_chart("No bankruptcy risk data available")

        # Create subplot with gauges for each model
        models = risk_data['models']
        num_models = len(models)

        # Create figure
        fig = make_subplots(
            rows=num_models,
            cols=1,
            specs=[[{"type": "indicator"}] for _ in range(num_models)],
            subplot_titles=[models[model].get('model', model) for model in models]
        )

        # Add gauge for each model
        row = 1
        for model_name, model_data in models.items():
            if model_data.get('score') is not None:
                # Determine gauge configuration based on the model
                if model_name == 'altman_z_score':
                    # For Z-Score, we use different thresholds based on the model variant
                    thresholds = model_data.get('thresholds', {'safe': 3.0, 'grey': 1.8})
                    min_val = 0
                    max_val = thresholds['safe'] * 1.5  # Give some headroom

                    # Define zones
                    zones = [
                        {'value': thresholds['grey'], 'color': self.colors['danger']},
                        {'value': thresholds['safe'], 'color': self.colors['warning']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Z-Score')}

                elif model_name == 'springate':
                    # For Springate, higher values are better
                    threshold = model_data.get('threshold', 0.862)
                    min_val = 0
                    max_val = threshold * 2  # Give some headroom

                    zones = [
                        {'value': threshold, 'color': self.colors['danger']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Springate Score')}

                elif model_name == 'zmijewski':
                    # For Zmijewski, we use probability (0-1) with threshold
                    probability = model_data.get('probability')
                    if probability is not None:
                        value = probability
                        min_val = 0
                        max_val = 1
                        threshold = model_data.get('threshold', 0.5)

                        zones = [
                            {'value': threshold, 'color': self.colors['success']},
                            {'value': max_val, 'color': self.colors['danger']}
                        ]

                        title = {'text': model_data.get('description', 'Bankruptcy Probability')}
                    else:
                        # Use score instead
                        value = model_data.get('score')
                        min_val = -10
                        max_val = 10

                        zones = [
                            {'value': 0, 'color': self.colors['success']},
                            {'value': max_val, 'color': self.colors['danger']}
                        ]

                        title = {'text': model_data.get('description', 'Zmijewski Score')}
                else:
                    # Generic configuration for other models
                    min_val = 0
                    max_val = 10
                    value = model_data.get('score', 0)

                    zones = [
                        {'value': 5, 'color': self.colors['danger']},
                        {'value': 7.5, 'color': self.colors['warning']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Risk Score')}

                # Create gauge indicator
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=model_data.get(
                            'score') if model_name != 'zmijewski' or probability is None else probability,
                        title=title,
                        gauge={
                            'axis': {'range': [min_val, max_val], 'tickcolor': self.text_color},
                            'bar': {'color': model_data.get('color', self.colors['primary'])},
                            'steps': zones,
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': model_data.get(
                                    'score') if model_name != 'zmijewski' or probability is None else probability
                            }
                        }
                    ),
                    row=row, col=1
                )

            row += 1

        # Add overall assessment
        if risk_data.get('overall_assessment'):
            subtitle = f"Overall Assessment: {risk_data.get('overall_assessment', '').capitalize()}"
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        showarrow=False,
                        text=subtitle,
                        font=dict(size=14, color=risk_data.get('overall_color', self.text_color)),
                        xref="paper",
                        yref="paper"
                    )
                ]
            )

        # Configure layout
        fig.update_layout(
            title="Bankruptcy Risk Assessment",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            height=height or (250 * num_models),  # Adjust height based on number of models
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=100, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating bankruptcy risk chart: {e}")
        return self._empty_chart(f"Error creating bankruptcy risk chart: {str(e)}")


def plot_company_comparison(self,
                            comparison_data: Dict[str, Dict[str, float]],
                            metrics: List[str],
                            height: Optional[int] = None,
                            width: Optional[int] = None) -> go.Figure:
    """
    Create radar chart to compare companies

    Args:
        comparison_data: Dictionary of companies with their metrics
        metrics: List of metrics to include in comparison
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid comparison data
        if not comparison_data:
            return self._empty_chart("No comparison data available")

        # Create radar chart
        fig = go.Figure()

        # Add trace for each company
        colors = list(self.colors['sectors'].values())
        color_idx = 0

        for company, data in comparison_data.items():
            # Filter data to include only specified metrics
            filtered_data = {k: v for k, v in data.items() if k in metrics}

            if not filtered_data:
                continue

            fig.add_trace(
                go.Scatterpolar(
                    r=list(filtered_data.values()),
                    theta=list(filtered_data.keys()),
                    fill='toself',
                    name=company,
                    line=dict(color=colors[color_idx % len(colors)])
                )
            )

            color_idx += 1

        # Configure layout
        fig.update_layout(
            title="Company Comparison",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=self.grid_color
                ),
                angularaxis=dict(
                    gridcolor=self.grid_color
                ),
                bgcolor=self.bg_color
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=80, r=80, t=100, b=80)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating company comparison chart: {e}")
        return self._empty_chart(f"Error creating company comparison chart: {str(e)}")


def plot_sector_performance(self,
                            sector_data: pd.DataFrame,
                            metric: str = '1Y',
                            height: Optional[int] = None,
                            width: Optional[int] = None) -> go.Figure:
    """
    Create a horizontal bar chart showing sector performance

    Args:
        sector_data: DataFrame with sector performance data
        metric: Column to use for comparison (e.g., '1D', '1W', '1M', '1Y')
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid sector data
        if sector_data.empty or metric not in sector_data.columns:
            return self._empty_chart("No sector performance data available")

        # Sort data by performance metric
        sorted_data = sector_data.sort_values(by=metric)

        # Create color array based on positive/negative performance
        colors = []
        for value in sorted_data[metric]:
            if value > 0:
                colors.append(self.colors['success'])
            else:
                colors.append(self.colors['danger'])

        # Create horizontal bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=sorted_data[metric],
                y=sorted_data['Sector'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:.2f}%" for x in sorted_data[metric]],
                textposition='auto'
            )
        )

        # Configure layout
        fig.update_layout(
            title=f"Sector Performance ({metric})",
            xaxis_title="Return (%)",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.grid_color,
                zeroline=True,
                zerolinecolor=self.grid_color
            ),
            yaxis=dict(
                showgrid=False,
                categoryorder='array',
                categoryarray=sorted_data['Sector'].tolist()
            ),
            height=height or (350 + len(sorted_data) * 25),  # Adjust height based on number of sectors
            width=width or self.settings['width'],
            margin=dict(l=150, r=50, t=80, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating sector performance chart: {e}")
        return self._empty_chart(f"Error creating sector performance chart: {str(e)}")


def plot_valuation_heatmap(self,
                           valuation_data: Dict[str, Dict[str, float]],
                           height: Optional[int] = None,
                           width: Optional[int] = None) -> go.Figure:
    """
    Create a heatmap showing valuation multiples compared to sector averages

    Args:
        valuation_data: Dictionary with company and sector valuation metrics
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid valuation data
        if not valuation_data or 'company' not in valuation_data or 'sector_avg' not in valuation_data:
            return self._empty_chart("No valuation comparison data available")

        # Extract data
        company_vals = valuation_data['company']
        sector_vals = valuation_data['sector_avg']

        # Calculate percentage differences
        metrics = []
        diff_values = []
        company_values = []
        sector_values = []

        for metric in company_vals:
            if metric in sector_vals and sector_vals[metric] != 0:
                metrics.append(metric)

                company_val = company_vals[metric]
                sector_val = sector_vals[metric]
                company_values.append(company_val)
                sector_values.append(sector_val)

                # Calculate percentage difference
                pct_diff = (company_val / sector_val - 1) * 100
                diff_values.append(pct_diff)

        # Create custom scale with center at zero
        max_abs_diff = max(abs(min(diff_values)), abs(max(diff_values))) if diff_values else 50
        scale_bound = min(max_abs_diff, 50)  # Cap at ±50% for better visualization

        # Create heatmap
        fig = go.Figure()

        # Add heatmap for percentage differences
        fig.add_trace(
            go.Heatmap(
                z=[diff_values],
                x=metrics,
                colorscale=[
                    [0, self.colors['success']],
                    [0.5, self.colors['info']],
                    [1, self.colors['danger']]
                ],
                zmin=-scale_bound,
                zmax=scale_bound,
                showscale=True,
                colorbar=dict(
                    title="% Difference<br>From Sector Avg",
                    titleside="top",
                    x=1.05
                ),
                text=[[
                          f"Company: {company_values[i]:.2f}<br>Sector Avg: {sector_values[i]:.2f}<br>Diff: {diff_values[i]:.2f}%"
                          for i in range(len(metrics))]],
                hoverinfo="text"
            )
        )

        # Add annotations with values
        annotations = []
        for i, metric in enumerate(metrics):
            annotations.append(
                dict(
                    x=i,
                    y=0,
                    text=f"{diff_values[i]:.1f}%",
                    showarrow=False,
                    font=dict(color="white", size=12)
                )
            )

        # Configure layout
        fig.update_layout(
            title="Valuation Multiples vs. Sector Averages",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            annotations=annotations,
            xaxis=dict(
                title="Valuation Metrics",
                showgrid=False,
                tickangle=45
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False
            ),
            height=height or 300,
            width=width or self.settings['width'],
            margin=dict(l=50, r=80, t=80, b=100)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating valuation heatmap: {e}")
        return self._empty_chart(f"Error creating valuation heatmap: {str(e)}")


def plot_financial_health_score(self,
                                health_data: Dict[str, Any],
                                height: Optional[int] = None,
                                width: Optional[int] = None) -> go.Figure:
    """
    Create a gauge chart showing overall financial health score

    Args:
        health_data: Dictionary with financial health scores and components
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid health data
        if not health_data or 'overall_score' not in health_data:
            return self._empty_chart("                    row=2, col=1
                                     )

        # Set chart title
        display_name = company_name if company_name else ticker
        title = f"{display_name} ({ticker}) - Price Chart"

        # Configure layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            legend=dict(
                font=dict(size=10),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                showgrid=True,
                gridcolor=self.grid_color
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor=self.grid_color,
                autorange=True,
                fixedrange=False
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Configure volume axis if applicable
        if volume:
            fig.update_yaxes(
                title="Volume",
                showgrid=True,
                gridcolor=self.grid_color,
                row=2, col=1
            )

        return fig
    except Exception as e:
        logger.error(f"Error creating price chart: {e}")
        return self._empty_chart(f"Error creating price chart: {str(e)}")


def plot_financial_statement_trend(self,
                                   statement_data: pd.DataFrame,
                                   statement_type: str,
                                   metrics: List[str],
                                   height: Optional[int] = None,
                                   width: Optional[int] = None) -> go.Figure:
    """
    Plot trends from financial statements

    Args:
        statement_data: DataFrame containing financial statement data
        statement_type: Type of statement ('income', 'balance', 'cash_flow')
        metrics: List of metrics to plot
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        if statement_data.empty:
            return self._empty_chart(f"No {statement_type} statement data available")

        # Create the figure
        fig = go.Figure()

        # Format title based on statement type
        title_map = {
            'income': 'Income Statement Trends',
            'balance': 'Balance Sheet Trends',
            'cash_flow': 'Cash Flow Statement Trends'
        }
        title = title_map.get(statement_type, 'Financial Statement Trends')

        # Transpose if needed - we need periods as index
        if not isinstance(statement_data.index, pd.DatetimeIndex):
            if set(metrics).issubset(statement_data.index):
                # We have metrics in index, columns are dates
                # Extract just the metrics we want
                plot_data = statement_data.loc[metrics]
            else:
                # Try to find metrics that might be present
                available_metrics = [m for m in metrics if m in statement_data.index]
                if available_metrics:
                    plot_data = statement_data.loc[available_metrics]
                else:
                    return self._empty_chart(f"Specified metrics not found in {statement_type} statement")
        else:
            # Dates are in index, transpose to get consistent format
            plot_data = statement_data.T

        # Add traces for each metric
        for metric in metrics:
            if metric in plot_data.index:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.columns,
                        y=plot_data.loc[metric],
                        mode='lines+markers',
                        name=metric
                    )
                )

        # Configure layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Period",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating financial statement trend chart: {e}")
        return self._empty_chart(f"Error creating financial statement trend chart: {str(e)}")


def plot_financial_ratios(self,
                          ratio_data: Dict[str, Dict[str, Any]],
                          category: Optional[str] = None,
                          benchmark_data: Optional[Dict[str, Dict[str, float]]] = None,
                          height: Optional[int] = None,
                          width: Optional[int] = None) -> go.Figure:
    """
    Create a bar chart for financial ratios with benchmark comparison

    Args:
        ratio_data: Dictionary of analyzed ratios by category
        category: Specific category to plot, if None plots all categories
        benchmark_data: Optional benchmark data for comparison
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if ratio data is empty or None
        if not ratio_data:
            return self._empty_chart("No ratio data available")

        # Prepare data for plotting
        ratio_names = []
        ratio_values = []
        benchmark_values = []
        colors = []

        # Select categories to plot
        if category:
            categories = [category] if category in ratio_data else []
        else:
            categories = list(ratio_data.keys())

        # Extract data for each ratio
        for cat in categories:
            if cat in ratio_data:
                for ratio_name, ratio_info in ratio_data[cat].items():
                    ratio_names.append(f"{cat}: {ratio_name}")

                    # Get ratio value
                    if 'value' in ratio_info:
                        ratio_values.append(ratio_info['value'])
                    else:
                        ratio_values.append(None)

                    # Get benchmark value
                    if 'benchmark' in ratio_info and ratio_info['benchmark'] is not None:
                        benchmark_values.append(ratio_info['benchmark'])
                    elif benchmark_data and cat in benchmark_data and ratio_name in benchmark_data[cat]:
                        benchmark_values.append(benchmark_data[cat][ratio_name])
                    else:
                        benchmark_values.append(None)

                    # Set color based on assessment
                    if 'color' in ratio_info:
                        colors.append(ratio_info['color'])
                    else:
                        colors.append(self.colors['primary'])

        # If no data found
        if not ratio_names:
            return self._empty_chart("No ratios found for the specified category")

        # Create figure
        fig = go.Figure()

        # Add ratio bars
        fig.add_trace(
            go.Bar(
                x=ratio_names,
                y=ratio_values,
                name='Company Ratio',
                marker_color=colors
            )
        )

        # Add benchmark bars if available
        if any(benchmark_values):
            fig.add_trace(
                go.Bar(
                    x=ratio_names,
                    y=benchmark_values,
                    name='Industry Benchmark',
                    marker_color=self.colors['secondary'],
                    opacity=0.7
                )
            )

        # Configure layout
        title = f"Financial Ratios - {category}" if category else "Financial Ratios"
        fig.update_layout(
            title=title,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Ratio",
                showgrid=True,
                gridcolor=self.grid_color,
                tickangle=45
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridcolor=self.grid_color
            ),
            barmode='group',
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=150)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating financial ratios chart: {e}")
        return self._empty_chart(f"Error creating financial ratios chart: {str(e)}")


def plot_bankruptcy_risk(self,
                         risk_data: Dict[str, Any],
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> go.Figure:
    """
    Create a gauge chart to visualize bankruptcy risk

    Args:
        risk_data: Dictionary with bankruptcy model results
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid risk data
        if not risk_data or 'models' not in risk_data:
            return self._empty_chart("No bankruptcy risk data available")

        # Create subplot with gauges for each model
        models = risk_data['models']
        num_models = len(models)

        # Create figure
        fig = make_subplots(
            rows=num_models,
            cols=1,
            specs=[[{"type": "indicator"}] for _ in range(num_models)],
            subplot_titles=[models[model].get('model', model) for model in models]
        )

        # Add gauge for each model
        row = 1
        for model_name, model_data in models.items():
            if model_data.get('score') is not None:
                # Determine gauge configuration based on the model
                if model_name == 'altman_z_score':
                    # For Z-Score, we use different thresholds based on the model variant
                    thresholds = model_data.get('thresholds', {'safe': 3.0, 'grey': 1.8})
                    min_val = 0
                    max_val = thresholds['safe'] * 1.5  # Give some headroom

                    # Define zones
                    zones = [
                        {'value': thresholds['grey'], 'color': self.colors['danger']},
                        {'value': thresholds['safe'], 'color': self.colors['warning']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Z-Score')}

                elif model_name == 'springate':
                    # For Springate, higher values are better
                    threshold = model_data.get('threshold', 0.862)
                    min_val = 0
                    max_val = threshold * 2  # Give some headroom

                    zones = [
                        {'value': threshold, 'color': self.colors['danger']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Springate Score')}

                elif model_name == 'zmijewski':
                    # For Zmijewski, we use probability (0-1) with threshold
                    probability = model_data.get('probability')
                    if probability is not None:
                        value = probability
                        min_val = 0
                        max_val = 1
                        threshold = model_data.get('threshold', 0.5)

                        zones = [
                            {'value': threshold, 'color': self.colors['success']},
                            {'value': max_val, 'color': self.colors['danger']}
                        ]

                        title = {'text': model_data.get('description', 'Bankruptcy Probability')}
                    else:
                        # Use score instead
                        value = model_data.get('score')
                        min_val = -10
                        max_val = 10

                        zones = [
                            {'value': 0, 'color': self.colors['success']},
                            {'value': max_val, 'color': self.colors['danger']}
                        ]

                        title = {'text': model_data.get('description', 'Zmijewski Score')}
                else:
                    # Generic configuration for other models
                    min_val = 0
                    max_val = 10
                    value = model_data.get('score', 0)

                    zones = [
                        {'value': 5, 'color': self.colors['danger']},
                        {'value': 7.5, 'color': self.colors['warning']},
                        {'value': max_val, 'color': self.colors['success']}
                    ]

                    title = {'text': model_data.get('description', 'Risk Score')}

                # Create gauge indicator
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=model_data.get(
                            'score') if model_name != 'zmijewski' or probability is None else probability,
                        title=title,
                        gauge={
                            'axis': {'range': [min_val, max_val], 'tickcolor': self.text_color},
                            'bar': {'color': model_data.get('color', self.colors['primary'])},
                            'steps': zones,
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': model_data.get(
                                    'score') if model_name != 'zmijewski' or probability is None else probability
                            }
                        }
                    ),
                    row=row, col=1
                )

            row += 1

        # Add overall assessment
        if risk_data.get('overall_assessment'):
            subtitle = f"Overall Assessment: {risk_data.get('overall_assessment', '').capitalize()}"
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        showarrow=False,
                        text=subtitle,
                        font=dict(size=14, color=risk_data.get('overall_color', self.text_color)),
                        xref="paper",
                        yref="paper"
                    )
                ]
            )

        # Configure layout
        fig.update_layout(
            title="Bankruptcy Risk Assessment",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            height=height or (250 * num_models),  # Adjust height based on number of models
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=100, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating bankruptcy risk chart: {e}")
        return self._empty_chart(f"Error creating bankruptcy risk chart: {str(e)}")


def plot_company_comparison(self,
                            comparison_data: Dict[str, Dict[str, float]],
                            metrics: List[str],
                            height: Optional[int] = None,
                            width: Optional[int] = None) -> go.Figure:
    """
    Create radar chart to compare companies

    Args:
        comparison_data: Dictionary of companies with their metrics
        metrics: List of metrics to include in comparison
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid comparison data
        if not comparison_data:
            return self._empty_chart("No comparison data available")

        # Create radar chart
        fig = go.Figure()

        # Add trace for each company
        colors = list(self.colors['sectors'].values())
        color_idx = 0

        for company, data in comparison_data.items():
            # Filter data to include only specified metrics
            filtered_data = {k: v for k, v in data.items() if k in metrics}

            if not filtered_data:
                continue

            fig.add_trace(
                go.Scatterpolar(
                    r=list(filtered_data.values()),
                    theta=list(filtered_data.keys()),
                    fill='toself',
                    name=company,
                    line=dict(color=colors[color_idx % len(colors)])
                )
            )

            color_idx += 1

        # Configure layout
        fig.update_layout(
            title="Company Comparison",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=self.grid_color
                ),
                angularaxis=dict(
                    gridcolor=self.grid_color
                ),
                bgcolor=self.bg_color
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=80, r=80, t=100, b=80)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating company comparison chart: {e}")
        return self._empty_chart(f"Error creating company comparison chart: {str(e)}")


def plot_sector_performance(self,
                            sector_data: pd.DataFrame,
                            metric: str = '1Y',
                            height: Optional[int] = None,
                            width: Optional[int] = None) -> go.Figure:
    """
    Create a horizontal bar chart showing sector performance

    Args:
        sector_data: DataFrame with sector performance data
        metric: Column to use for comparison (e.g., '1D', '1W', '1M', '1Y')
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid sector data
        if sector_data.empty or metric not in sector_data.columns:
            return self._empty_chart("No sector performance data available")

        # Sort data by performance metric
        sorted_data = sector_data.sort_values(by=metric)

        # Create color array based on positive/negative performance
        colors = []
        for value in sorted_data[metric]:
            if value > 0:
                colors.append(self.colors['success'])
            else:
                colors.append(self.colors['danger'])

        # Create horizontal bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=sorted_data[metric],
                y=sorted_data['Sector'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:.2f}%" for x in sorted_data[metric]],
                textposition='auto'
            )
        )

        # Configure layout
        fig.update_layout(
            title=f"Sector Performance ({metric})",
            xaxis_title="Return (%)",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=self.grid_color,
                zeroline=True,
                zerolinecolor=self.grid_color
            ),
            yaxis=dict(
                showgrid=False,
                categoryorder='array',
                categoryarray=sorted_data['Sector'].tolist()
            ),
            height=height or (350 + len(sorted_data) * 25),  # Adjust height based on number of sectors
            width=width or self.settings['width'],
            margin=dict(l=150, r=50, t=80, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating sector performance chart: {e}")
        return self._empty_chart(f"Error creating sector performance chart: {str(e)}")


def plot_valuation_heatmap(self,
                           valuation_data: Dict[str, Dict[str, float]],
                           height: Optional[int] = None,
                           width: Optional[int] = None) -> go.Figure:
    """
    Create a heatmap showing valuation multiples compared to sector averages

    Args:
        valuation_data: Dictionary with company and sector valuation metrics
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid valuation data
        if not valuation_data or 'company' not in valuation_data or 'sector_avg' not in valuation_data:
            return self._empty_chart("No valuation comparison data available")

        # Extract data
        company_vals = valuation_data['company']
        sector_vals = valuation_data['sector_avg']

        # Calculate percentage differences
        metrics = []
        diff_values = []
        company_values = []
        sector_values = []

        for metric in company_vals:
            if metric in sector_vals and sector_vals[metric] != 0:
                metrics.append(metric)

                company_val = company_vals[metric]
                sector_val = sector_vals[metric]
                company_values.append(company_val)
                sector_values.append(sector_val)

                # Calculate percentage difference
                pct_diff = (company_val / sector_val - 1) * 100
                diff_values.append(pct_diff)

        # Create custom scale with center at zero
        max_abs_diff = max(abs(min(diff_values)), abs(max(diff_values))) if diff_values else 50
        scale_bound = min(max_abs_diff, 50)  # Cap at ±50% for better visualization

        # Create heatmap
        fig = go.Figure()

        # Add heatmap for percentage differences
        fig.add_trace(
            go.Heatmap(
                z=[diff_values],
                x=metrics,
                colorscale=[
                    [0, self.colors['success']],
                    [0.5, self.colors['info']],
                    [1, self.colors['danger']]
                ],
                zmin=-scale_bound,
                zmax=scale_bound,
                showscale=True,
                colorbar=dict(
                    title="% Difference<br>From Sector Avg",
                    titleside="top",
                    x=1.05
                ),
                text=[[
                          f"Company: {company_values[i]:.2f}<br>Sector Avg: {sector_values[i]:.2f}<br>Diff: {diff_values[i]:.2f}%"
                          for i in range(len(metrics))]],
                hoverinfo="text"
            )
        )

        # Add annotations with values
        annotations = []
        for i, metric in enumerate(metrics):
            annotations.append(
                dict(
                    x=i,
                    y=0,
                    text=f"{diff_values[i]:.1f}%",
                    showarrow=False,
                    font=dict(color="white", size=12)
                )
            )

        # Configure layout
        fig.update_layout(
            title="Valuation Multiples vs. Sector Averages",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            annotations=annotations,
            xaxis=dict(
                title="Valuation Metrics",
                showgrid=False,
                tickangle=45
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False
            ),
            height=height or 300,
            width=width or self.settings['width'],
            margin=dict(l=50, r=80, t=80, b=100)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating valuation heatmap: {e}")
        return self._empty_chart(f"Error creating valuation heatmap: {str(e)}")


def plot_financial_health_score(self,
                                health_data: Dict[str, Any],
                                height: Optional[int] = None,
                                width: Optional[int] = None) -> go.Figure:
    """
    Create a gauge chart showing overall financial health score

    Args:
        health_data: Dictionary with financial health scores and components
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        if not health_data:
            return self._empty_chart("No financial health data available")

        # Extract data
        overall_score = health_data.get('overall_score')
        components = health_data.get('components', {})
        rating = health_data.get('rating', '')

        if overall_score is None:
            return self._empty_chart("Financial health score not available")

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ],
            subplot_titles=["Overall Financial Health", "Profitability", "Liquidity", "Solvency"]
        )

        # Add overall score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score,
                title={"text": f"Rating: {rating}"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": self.text_color},
                    "bar": {"color": self._get_health_score_color(overall_score)},
                    "steps": [
                        {"range": [0, 30], "color": self.colors['danger']},
                        {"range": [30, 45], "color": "#ff8c69"},  # Light red
                        {"range": [45, 60], "color": self.colors['warning']},
                        {"range": [60, 75], "color": "#9acd32"},  # Yellow-green
                        {"range": [75, 100], "color": self.colors['success']}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": overall_score
                    }
                }
            ),
            row=1, col=1
        )

        # Add component gauges
        if 'profitability' in components:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=components['profitability'],
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": self.text_color},
                        "bar": {"color": self._get_health_score_color(components['profitability'])},
                        "steps": [
                            {"range": [0, 30], "color": self.colors['danger']},
                            {"range": [30, 70], "color": self.colors['warning']},
                            {"range": [70, 100], "color": self.colors['success']}
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.75,
                            "value": components['solvency']
                        }
                    }
                ),
                row=2, col=2
            )

        # Configure layout
        fig.update_layout(
            title="Financial Health Assessment",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            height=height or 600,
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=100, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating financial health chart: {e}")
        return self._empty_chart(f"Error creating financial health chart: {str(e)}")


def plot_dcf_sensitivity(self,
                         dcf_data: Dict[str, Any],
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> go.Figure:
    """
    Create a heatmap showing DCF valuation sensitivity to growth and discount rates

    Args:
        dcf_data: Dictionary with DCF valuation data including sensitivity analysis
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check for valid DCF data
        if not dcf_data or 'sensitivity' not in dcf_data:
            return self._empty_chart("No DCF sensitivity analysis data available")

        # Extract sensitivity data
        sensitivity = dcf_data['sensitivity']

        if not isinstance(sensitivity,
                          dict) or 'growth_rates' not in sensitivity or 'discount_rates' not in sensitivity or 'values' not in sensitivity:
            return self._empty_chart("Invalid DCF sensitivity data format")

        growth_rates = sensitivity['growth_rates']
        discount_rates = sensitivity['discount_rates']
        values = sensitivity['values']

        # Create heatmap
        fig = go.Figure()

        current_price = dcf_data.get('current_price')

        # Normalize values by current price if available
        if current_price and current_price > 0:
            # Calculate percentage difference from current price
            heatmap_values = [[(val / current_price - 1) * 100 for val in row] for row in values]

            # Create text matrix for hover information
            text_matrix = [[
                               f"Growth: {growth_rates[i]}%<br>Discount: {discount_rates[j]}%<br>Value: ${values[i][j]:.2f}<br>Diff from current: {heatmap_values[i][j]:.2f}%"
                               for j in range(len(discount_rates))] for i in range(len(growth_rates))]

            # Set color scale range
            max_diff = max([max(row) for row in heatmap_values])
            min_diff = min([min(row) for row in heatmap_values])
            abs_max = max(abs(max_diff), abs(min_diff))

            fig.add_trace(
                go.Heatmap(
                    z=heatmap_values,
                    x=discount_rates,
                    y=growth_rates,
                    colorscale='RdBu_r',  # Red for undervalued, blue for overvalued
                    zmin=-abs_max,
                    zmax=abs_max,
                    text=text_matrix,
                    hoverinfo="text",
                    colorbar=dict(
                        title="% Difference<br>From Current Price",
                        titleside="top"
                    )
                )
            )

        else:
            # Use absolute values
            text_matrix = [
                [f"Growth: {growth_rates[i]}%<br>Discount: {discount_rates[j]}%<br>Value: ${values[i][j]:.2f}"
                 for j in range(len(discount_rates))] for i in range(len(growth_rates))]

            fig.add_trace(
                go.Heatmap(
                    z=values,
                    x=discount_rates,
                    y=growth_rates,
                    colorscale=[
                        [0, self.colors['danger']],
                        [0.5, self.colors['warning']],
                        [1, self.colors['success']]
                    ],
                    text=text_matrix,
                    hoverinfo="text",
                    colorbar=dict(
                        title="DCF Value ($)",
                        titleside="top"
                    )
                )
            )

        # Mark base case
        if 'base_case' in dcf_data:
            base_case = dcf_data['base_case']
            growth_rate = base_case.get('growth_rate')
            discount_rate = base_case.get('discount_rate')

            if growth_rate is not None and discount_rate is not None:
                # Find closest values in our arrays
                growth_idx = min(range(len(growth_rates)), key=lambda i: abs(growth_rates[i] - growth_rate))
                discount_idx = min(range(len(discount_rates)), key=lambda i: abs(discount_rates[i] - discount_rate))

                fig.add_trace(
                    go.Scatter(
                        x=[discount_rates[discount_idx]],
                        y=[growth_rates[growth_idx]],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=15,
                            color='white',
                            line=dict(width=2, color='black')
                        ),
                        name="Base Case",
                        hoverinfo="name"
                    )
                )

        # Add current price line if available
        if current_price and current_price > 0:
            for i, growth_rate in enumerate(growth_rates):
                for j in range(len(discount_rates) - 1):
                    if (values[i][j] >= current_price >= values[i][j + 1]) or (
                            values[i][j] <= current_price <= values[i][j + 1]):
                        # Calculate interpolation point
                        disc1, disc2 = discount_rates[j], discount_rates[j + 1]
                        val1, val2 = values[i][j], values[i][j + 1]

                        # Linear interpolation: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        x_interp = disc1 + (current_price - val1) * (disc2 - disc1) / (val2 - val1)

                        fig.add_trace(
                            go.Scatter(
                                x=[x_interp],
                                y=[growth_rate],
                                mode='markers',
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color='black'
                                ),
                                showlegend=False
                            )
                        )

        # Configure layout
        fig.update_layout(
            title="DCF Sensitivity Analysis",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(
                color=self.text_color,
                family=self.settings['font_family']
            ),
            xaxis=dict(
                title="Discount Rate (%)",
                showgrid=False
            ),
            yaxis=dict(
                title="Growth Rate (%)",
                showgrid=False
            ),
            height=height or self.settings['height'],
            width=width or self.settings['width'],
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating DCF sensitivity chart: {e}")
        return self._empty_chart(f"Error creating DCF sensitivity chart: {str(e)}")

    # Helper methods


def _empty_chart(self, message: str) -> go.Figure:
    """Create an empty chart with a message"""
    fig = go.Figure()

    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=message,
        showarrow=False,
        font=dict(size=14, color=self.text_color)
    )

    fig.update_layout(
        plot_bgcolor=self.bg_color,
        paper_bgcolor=self.bg_color,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        width=800
    )

    return fig


def _get_health_score_color(self, score: float) -> str:
    """Get color based on health score"""
    if score >= 75:
        return self.colors['success']
    elif score >= 60:
        return "#9acd32"  # Yellow-green
    elif score >= 45:
        return self.colors['warning']
    elif score >= 30:
        return "#ff8c69"  # Light red
    else:
        return self.colors['danger']


']},
{"range": [70, 100], "color": self.colors['success']}
],
"threshold": {
"line": {"color": "white", "width": 4},
"thickness": 0.75,
"value": components['profitability']
}
}
),
row = 1, col = 2
)

if 'liquidity' in components:
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=components['liquidity'],
            gauge={
                "axis": {"range": [0, 100], "tickcolor": self.text_color},
                "bar": {"color": self._get_health_score_color(components['liquidity'])},
                "steps": [
{"range": [0, 30], "color": self.colors['danger']},
{"range": [30, 70], "color": self.colors['warning']},
{"range": [70, 100], "color": self.colors['success']}
],
"threshold": {
"line": {"color": "white", "width": 4},
"thickness": 0.75,
"value": components['liquidity']
}
}
),
row = 2, col = 1
)

if 'solvency' in components:
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=components['solvency'],
            gauge={
                "axis": {"range": [0, 100], "tickcolor": self.text_color},
                "bar": {"color": self._get_health_score_color(components['solvency'])},
                "steps": [
{"range": [0, 30], "color": self.colors['danger']},
{"range": [30, 70], "color": self.colors['warningimport os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COLORS, VIZ_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')


class FinancialVisualizer:
    """
    Class for visualizing financial data using Plotly charts.
    Provides functions for common financial visualizations including
    price charts, financial statement trends, ratio analysis, etc.
    """

    def __init__(self, theme: str = "dark"):
        """
        Initialize visualizer with theme settings

        Args:
            theme: 'dark' or 'light' theme for charts
        """
        self.theme = theme
        self.colors = COLORS
        self.settings = VIZ_SETTINGS

        # Set default colors based on theme
        if theme == "dark":
            self.bg_color = "#121212"
            self.grid_color = "#333333"
            self.text_color = "#e0e0e0"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#e0e0e0"
            self.text_color = "#333333"

    def plot_stock_price(self,
                         price_data: pd.DataFrame,
                         ticker: str,
                         company_name: Optional[str] = None,
                         ma_periods: List[int] = [50, 200],
                         volume: bool = True,
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> go.Figure:
        """
        Create a candlestick chart with moving averages and volume

        Args:
            price_data: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            company_name: Company name for chart title
            ma_periods: List of periods for moving averages
            volume: Whether to show volume panel
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Plotly figure object
        """
        try:
            if price_data.empty:
                return self._empty_chart("No price data available")

            # Set up figure with secondary y-axis for volume
            fig = make_subplots(
                rows=2 if volume else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.8, 0.2] if volume else [1]
            )

            # Add main price candlestick
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name="Price",
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )

            # Add moving averages
            for period in ma_periods:
                if len(price_data) >= period:
                    ma = price_data['Close'].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=price_data.index,
                            y=ma,
                            name=f"{period}-day MA",
                            line=dict(width=1.5)
                        ),
                        row=1, col=1
                    )

            # Add volume bars if requested
            if volume and 'Volume' in price_data.columns:
                # Create color array for volume bars
                colors = np.where(price_data['Close'] >= price_data['Open'],
                                  self.colors['success'], self.colors['danger'])

                fig.add_trace(
                    go.Bar(
                        x=price_data.index,
                        y=price_data['Volume'],
                        name="Volume",
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )

            # Set chart title
            display_name = company_name if company_name else ticker
            title = f"{display_name} ({ticker}) - Price Chart"

            # Configure layout
            fig.update_layout(
                title=title,
                plot_bgcolor=self.bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(
                    color=self.text_color,
                    family=self.settings['font_family']
                ),
                legend=dict(
                    font=dict(size=10),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    type='date',
                    showgrid=True,
                    gridcolor=self.grid_color
                ),
                yaxis=dict(
                    title="Price",
                    showgrid=True,
                    gridcolor=self.grid_color,
                    autorange=True,
                    fixedrange=False
                ),
                height=height or self.settings['height'],
                width=width or self.settings['width'],
                margin=dict(l=50, r=50, t=80, b=50)
            )

            # Configure volume axis if applicable
            if volume:
                fig.update_yaxes(
                    title="Volume",
                    showgrid=True,
                    gridcolor=self.grid_color,
                    row=2, col=1
                )

            return fig
            except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return self._empty_chart(f"Error creating price chart: {str(e)}")