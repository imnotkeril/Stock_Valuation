"""Tests for utils.visualization.FinancialVisualizer (offline, AAPL fixtures + edge cases)."""

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from StockAnalysisSystem.src.config import VIZ_SETTINGS
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer


@pytest.fixture
def viz() -> FinancialVisualizer:
    return FinancialVisualizer(theme="dark")


def is_empty_chart(fig: go.Figure) -> bool:
    return bool((fig.layout.meta or {}).get("empty_chart"))


def assert_chart(fig, empty: bool = False):
    assert isinstance(fig, go.Figure)
    assert is_empty_chart(fig) is empty, (fig.layout.meta, [a.text for a in fig.layout.annotations])
    if not empty:
        assert fig.data, "chart has no traces"
    fig.to_json()  # must be serialisable for st.plotly_chart


def empty_message(fig) -> str:
    return fig.layout.meta["message"]


# ---------------------------------------------------------------------------
# Inputs built from real AAPL data
# ---------------------------------------------------------------------------
@pytest.fixture
def ratio_data():
    return {
        "valuation": {"pe_ratio": {"value": 29.4, "benchmark": 30.0, "color": "#74f174"},
                      "ps_ratio": {"value": 7.1, "benchmark": None}},
        "profitability": {"net_margin": 0.24, "roe": float("nan")},
    }


@pytest.fixture
def returns_by_ticker(aapl, msft):
    out = {}
    for company in (aapl, msft):
        close = company["price_data"]["Close"].iloc[-250:]
        out[company["ticker"]] = (close / close.iloc[0] - 1) * 100
    return out


def call_all_methods(viz, aapl, ratio_data, returns_by_ticker):
    """Every public plot method called with meaningful AAPL-derived data."""
    prices = aapl["price_data"]
    income = aapl["financial_data"]["income_statement"]
    metrics_df = pd.DataFrame({"Company": ["AAPL", "MSFT", "TSAS"], "P/E": [29.4, 35.1, 22.0],
                               "P/S": [7.1, 12.0, 8.3], "ROE": [1.5, 0.35, 0.12]}).set_index("Company")
    return {
        "plot_stock_price": viz.plot_stock_price(prices, "AAPL", company_name="Apple Inc."),
        "plot_price_comparison": viz.plot_price_comparison(returns_by_ticker, {"AAPL": "Apple"},
                                                           primary_ticker="AAPL"),
        "plot_price_vs_value": viz.plot_price_vs_value(prices, "AAPL", fair_value=180.0, margin_of_safety=0.2),
        "plot_financial_statement_trend": viz.plot_financial_statement_trend(
            income, "income", ["Total Revenue", "net_income"]),
        "plot_financial_ratios": viz.plot_financial_ratios(ratio_data),
        "plot_metrics_bar_chart": viz.plot_metrics_bar_chart(
            pd.DataFrame({"Metric": ["Gross Margin", "Net Margin"], "Value": [46.2, 24.0]}), is_percent=True),
        "plot_bar_comparison": viz.plot_bar_comparison({"net_margin": [0.24, 0.36], "roe": [1.5, 0.35]},
                                                       ["Apple", "Microsoft"], primary_index=0),
        "plot_radar_comparison": viz.plot_radar_comparison(
            {"AAPL": {"pe_ratio": 40, "roe": 90, "net_margin": 70}, "MSFT": {"pe_ratio": 60, "roe": 50}},
            primary_ticker="AAPL", radial_range=(0, 100)),
        "plot_company_comparison": viz.plot_company_comparison(
            {"AAPL": {"a": 1.0, "b": 2.0, "c": 3.0}, "MSFT": {"a": 2.0, "b": 1.0, "c": 1.5}}, ["a", "b", "c"]),
        "plot_metrics_heatmap": viz.plot_metrics_heatmap(metrics_df, higher_is_better={"P/E": False}),
        "plot_sector_performance": viz.plot_sector_performance(
            pd.DataFrame({"Sector": ["Technology", "Energy"], "1Y": [12.5, -3.0]})),
        "plot_sector_distribution": viz.plot_sector_distribution(
            pd.DataFrame({"Sector": ["Technology", "Energy"], "Count": [3, 1]})),
        "plot_method_distribution": viz.plot_method_distribution(
            pd.DataFrame({"Method": ["Blend", "Relative"], "Count": [4, 2]})),
        "plot_market_cap_treemap": viz.plot_market_cap_treemap(
            pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Name": ["Apple", "Microsoft"],
                          "Market Cap": [2.76e12, 2.9e12], "Total Return": [5.0, -2.0]})),
        "plot_valuation_heatmap": viz.plot_valuation_heatmap(
            {"company": {"pe_ratio": 29.4, "pb_ratio": 41.4}, "sector_avg": {"pe_ratio": 30.0, "pb_ratio": 6.0}}),
        "plot_valuation_sector_heatmap": viz.plot_valuation_sector_heatmap(
            pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "P/E": [29.4, 35.0], "P/S": [7.1, 12.0]})),
        "plot_valuation_upside": viz.plot_valuation_upside({"AAPL": -16.2, "MSFT": 8.0}, primary_ticker="AAPL"),
        "plot_fair_value_scatter": viz.plot_fair_value_scatter(
            pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Price": [184.4, 380.0], "Fair Value": [154.5, 410.0]}),
            margin_of_safety=0.25),
        "plot_upside_histogram": viz.plot_upside_histogram([-16.2, 8.0, 30.0, 12.0]),
        "plot_valuation_breakdown": viz.plot_valuation_breakdown(
            pd.DataFrame({"Method": ["DCF", "Relative"], "Value": [150.0, 200.0]}), current_price=184.4),
        "plot_dcf_sensitivity": viz.plot_dcf_sensitivity(
            {"sensitivity": {"growth_rates": [0.02, 0.04], "discount_rates": [0.08, 0.10, 0.12],
                             "values": [[200, 150, 120], [240, 180, 140]]},
             "current_price": 184.4, "base_case": {"growth_rate": 0.04, "discount_rate": 0.10}}),
        "plot_bankruptcy_risk": viz.plot_bankruptcy_risk(
            {"models": {"altman_z_score": {"score": 7.1, "model": "Altman Z-Score",
                                           "thresholds": {"safe": 2.99, "grey": 1.81}},
                        "springate": {"score": 2.3}, "zmijewski": {"score": -3.0, "probability": 0.05}},
             "overall_assessment": "safe", "overall_color": "#74f174"}),
        "plot_financial_health_score": viz.plot_financial_health_score(
            {"overall_score": 72, "rating": "Good", "components": {"profitability": 90, "liquidity": 40,
                                                                    "solvency": None}}),
    }


PUBLIC_METHODS = sorted(name for name, _ in inspect.getmembers(FinancialVisualizer, inspect.isfunction)
                        if name.startswith("plot_"))


def test_every_public_method_is_covered(viz, aapl, ratio_data, returns_by_ticker):
    figures = call_all_methods(viz, aapl, ratio_data, returns_by_ticker)
    assert sorted(figures) == PUBLIC_METHODS


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_method_renders_real_data(theme, aapl, ratio_data, returns_by_ticker):
    viz = FinancialVisualizer(theme=theme)
    for name, fig in call_all_methods(viz, aapl, ratio_data, returns_by_ticker).items():
        assert_chart(fig), name
        assert fig.layout.paper_bgcolor == viz.bg_color, name


def test_dark_theme_comes_from_config(viz):
    assert viz.bg_color == VIZ_SETTINGS["background"]
    assert viz.text_color == VIZ_SETTINGS["text_color"]


# ---------------------------------------------------------------------------
# Empty / NaN inputs never raise and return the placeholder chart
# ---------------------------------------------------------------------------
NAN_PRICES = pd.DataFrame({"Open": [np.nan] * 3, "High": [np.nan] * 3, "Low": [np.nan] * 3,
                           "Close": [np.nan] * 3, "Volume": [np.nan] * 3},
                          index=pd.date_range("2024-01-01", periods=3))

EMPTY_CALLS = {
    "plot_stock_price": [(pd.DataFrame(), "X"), (NAN_PRICES, "X"), (None, "X")],
    "plot_price_comparison": [({},), ({"A": pd.Series(dtype=float)},), ({"A": pd.Series([np.nan, np.inf])},),
                              (None,)],
    "plot_price_vs_value": [(pd.DataFrame(), "X"), (NAN_PRICES, "X", np.nan)],
    "plot_financial_statement_trend": [(pd.DataFrame(), "income", ["Total Revenue"]),
                                       (pd.DataFrame({"2024": [1.0]}, index=["Other"]), "income", ["revenue"]),
                                       (pd.DataFrame({"2024": [np.nan]}, index=["Total Revenue"]), "income",
                                        ["Total Revenue"])],
    "plot_financial_ratios": [({},), (None,), ({"valuation": {"pe_ratio": np.nan}},),
                              ({"valuation": {"pe_ratio": 1.0}}, "liquidity")],
    "plot_metrics_bar_chart": [(pd.DataFrame(),), (pd.DataFrame({"Metric": ["a"], "Value": [np.nan]}),), ({},)],
    "plot_bar_comparison": [({}, []), ({"roe": [np.nan, None]}, ["A", "B"]), (None, None)],
    "plot_radar_comparison": [({},), ({"A": {"x": np.nan, "y": None}},), ({"A": {"x": 1.0}},)],
    "plot_company_comparison": [({}, []), ({"A": {"x": np.nan}}, ["x"])],
    "plot_metrics_heatmap": [(pd.DataFrame(),), (pd.DataFrame({"P/E": [np.nan, np.inf]}),), (None,)],
    "plot_sector_performance": [(pd.DataFrame(),), (pd.DataFrame({"Sector": ["Tech"], "1Y": [np.nan]}),),
                                (pd.DataFrame({"Sector": ["Tech"]}), "1Y"), (None,)],
    "plot_sector_distribution": [(pd.DataFrame(),), ({},), (pd.DataFrame({"Sector": ["T"], "Count": [0]}),)],
    "plot_method_distribution": [(pd.DataFrame(),), (pd.DataFrame({"Method": ["a"], "Count": [np.nan]}),)],
    "plot_market_cap_treemap": [(pd.DataFrame(),), (pd.DataFrame({"Symbol": ["A"], "Market Cap": [np.nan]}),),
                                (pd.DataFrame({"Symbol": ["A"]}),)],
    "plot_valuation_heatmap": [({},), (None,), ({"company": {"pe_ratio": np.nan}, "sector_avg": {"pe_ratio": 0}},)],
    "plot_valuation_sector_heatmap": [(pd.DataFrame(),), (pd.DataFrame({"Symbol": ["A"], "P/E": [-5.0]}),),
                                      (pd.DataFrame({"Symbol": ["A"], "Foo": [1.0]}),)],
    "plot_valuation_upside": [({},), ({"A": np.nan},), (None,)],
    "plot_fair_value_scatter": [(pd.DataFrame(),), (pd.DataFrame({"Price": [np.nan], "Fair Value": [1.0]}),),
                                (pd.DataFrame({"Price": [1.0]}),)],
    "plot_upside_histogram": [([],), ([np.nan, np.inf],), (None,)],
    "plot_valuation_breakdown": [(pd.DataFrame(),), (pd.DataFrame({"Method": ["DCF"], "Value": [np.nan]}), 10.0),
                                 ({},)],
    "plot_dcf_sensitivity": [({},), ({"sensitivity": {}},), ({"sensitivity": {"growth_rates": [0.1],
                                                                              "discount_rates": [0.1, 0.2],
                                                                              "values": [[1, 2, 3]]}},),
                             ({"sensitivity": {"growth_rates": [0.1], "discount_rates": [0.1],
                                               "values": [[np.nan]]}},)],
    "plot_bankruptcy_risk": [({},), ({"models": {}},), ({"models": {"altman_z_score": {"score": None}}},)],
    "plot_financial_health_score": [({},), ({"overall_score": None},), ({"overall_score": np.nan},)],
}


def test_empty_cases_cover_every_public_method():
    assert sorted(EMPTY_CALLS) == PUBLIC_METHODS


@pytest.mark.parametrize("name, args", [(n, a) for n, cases in EMPTY_CALLS.items() for a in cases])
def test_empty_and_nan_inputs_return_placeholder(viz, name, args):
    fig = getattr(viz, name)(*args)
    assert_chart(fig, empty=True)
    assert empty_message(fig)
    assert fig.layout.annotations[0].text == empty_message(fig)


def test_unexpected_errors_are_logged_not_raised(viz, caplog):
    class Exploding(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __bool__(self):
            return True

    fig = viz.plot_financial_health_score(Exploding())
    assert_chart(fig, empty=True)
    assert "boom" not in empty_message(fig)  # no raw error text for users
    assert any("financial health" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Behaviour of individual charts
# ---------------------------------------------------------------------------
def test_stock_price_moving_averages_and_volume(viz, aapl):
    fig = viz.plot_stock_price(aapl["price_data"], "AAPL", ma_periods=[20, 50])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "20-day MA", "50-day MA", "Volume"]
    assert isinstance(fig.data[0], go.Candlestick)
    assert fig.layout.yaxis.tickprefix == "$"
    assert fig.layout.xaxis.rangeslider.visible is False


@pytest.mark.parametrize("tz", [None, "UTC", "America/New_York"])
def test_stock_price_handles_tz_naive_and_aware_index(viz, aapl, tz):
    prices = aapl["price_data"].iloc[-60:].copy()
    if tz:
        prices.index = prices.index.tz_localize(tz)
    fig = viz.plot_stock_price(prices, "AAPL")
    assert_chart(fig)
    x = pd.DatetimeIndex(fig.data[0].x)
    assert x.tz is None
    assert x[-1].date() == aapl["price_data"].index[-1].date()


def test_stock_price_line_fallback_without_ohlc(viz, aapl):
    close_only = aapl["price_data"][["Close"]].iloc[-30:]
    fig = viz.plot_stock_price(close_only, "AAPL", ma_periods=[200], volume=True)
    assert [type(t) for t in fig.data] == [go.Scatter]  # no MA (too short), no volume


def test_price_comparison_normalize(viz, aapl):
    fig = viz.plot_price_comparison({"AAPL": aapl["price_data"]}, normalize=True)
    y = np.asarray(fig.data[0].y, dtype=float)
    assert y[0] == pytest.approx(0.0)
    close = aapl["price_data"]["Close"]
    assert y[-1] == pytest.approx((close.iloc[-1] / close.iloc[0] - 1) * 100)
    assert fig.layout.yaxis.ticksuffix == "%"


def test_statement_trend_is_chronological_and_scaled(viz, aapl):
    income = aapl["financial_data"]["income_statement"]
    fig = viz.plot_financial_statement_trend(income, "income", ["Total Revenue"])
    trace = fig.data[0]
    assert list(trace.x) == sorted(trace.x)
    assert trace.x[-1] == "2024"
    assert trace.y[-1] == pytest.approx(391.035)  # $391.035B
    assert fig.layout.yaxis.ticksuffix == "B" and fig.layout.yaxis.tickprefix == "$"


def test_statement_trend_accepts_transposed_frame(viz, aapl):
    income = aapl["financial_data"]["income_statement"]
    fig = viz.plot_financial_statement_trend(income.T, "income", ["revenue"])
    assert_chart(fig)


def test_bar_comparison_percent_inference(viz):
    fig = viz.plot_bar_comparison({"net_margin": [0.24, 0.36]}, ["A", "B"])
    assert fig.data[0].y[0] == pytest.approx(24.0)
    assert fig.layout.yaxis.ticksuffix == "%"
    fig = viz.plot_bar_comparison({"pe_ratio": [30.0, 20.0]}, ["A", "B"])
    assert fig.data[0].y[0] == pytest.approx(30.0)


def test_sector_performance_accepts_series_mapping(viz, returns_by_ticker):
    fig = viz.plot_sector_performance(returns_by_ticker, {"AAPL": "Apple"}, title="vs SPY")
    assert_chart(fig)
    assert len(fig.data) == 2


def test_sector_performance_tolerates_non_numeric_height(viz, aapl):
    # reports.py passes (prices, ticker, sector) positionally
    fig = viz.plot_sector_performance(aapl["price_data"], "XLK", "Technology")
    assert_chart(fig)
    assert fig.layout.height == 500  # default height of the comparison chart


def test_valuation_heatmap_premium_values(viz):
    fig = viz.plot_valuation_heatmap({"company": {"pe_ratio": 30.0, "pb_ratio": 3.0, "bad": "x"},
                                      "sector_avg": {"pe_ratio": 20.0, "pb_ratio": 6.0}})
    heatmap = fig.data[0]
    assert list(heatmap.x) == ["P/E", "P/B"]
    assert list(heatmap.z[0]) == pytest.approx([50.0, -50.0])


def test_metrics_heatmap_inverts_lower_is_better(viz):
    df = pd.DataFrame({"P/E": [10.0, 30.0], "ROE": [0.1, 0.3]}, index=["A", "B"])
    fig = viz.plot_metrics_heatmap(df, higher_is_better={"P/E": False})
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z[0, 0] > 0 > z[1, 0]  # low P/E is good
    assert z[1, 1] > 0 > z[0, 1]  # high ROE is good
    assert fig.data[0].text[0][0] == "10.00"


class TestDcfSensitivity:
    DATA = {
        "sensitivity": {"growth_rates": [0.02, 0.04, 0.06], "discount_rates": [0.08, 0.10],
                        "values": [[100.0, 80.0], [120.0, 90.0], [150.0, 110.0]]},
        "current_price": 100.0,
        "base_case": {"growth_rate": 0.041, "discount_rate": 0.099},
    }

    def test_axes_and_values(self, viz):
        fig = viz.plot_dcf_sensitivity(self.DATA)
        heatmap = fig.data[0]
        assert list(heatmap.x) == ["8.0%", "10.0%"]  # discount rates on x
        assert list(heatmap.y) == ["2.0%", "4.0%", "6.0%"]  # growth rates on y
        z = np.asarray(heatmap.z, dtype=float)
        assert z.shape == (3, 2)
        # z[i][j]: % difference of value(growth i, discount j) from the current price
        assert z[0, 0] == pytest.approx(0.0)
        assert z[1, 1] == pytest.approx(-10.0)
        assert z[2, 0] == pytest.approx(50.0)
        assert heatmap.text[2][1] == "$110.00"
        assert fig.layout.xaxis.title.text == "Discount Rate"
        assert fig.layout.yaxis.title.text == "Growth Rate"

    def test_base_case_marker_snaps_to_nearest_cell(self, viz):
        fig = viz.plot_dcf_sensitivity(self.DATA)
        marker = fig.data[1]
        assert marker.name == "Base case"
        assert list(marker.x) == ["10.0%"] and list(marker.y) == ["4.0%"]

    def test_transposed_matrix_and_percent_units(self, viz):
        data = {"sensitivity": {"growth_rates": [2, 4, 6], "discount_rates": [8, 10],
                                "values": [[100.0, 120.0, 150.0], [80.0, 90.0, 110.0]]},
                "base_case": {"growth_rate": 0.06, "discount_rate": 0.08}}
        fig = viz.plot_dcf_sensitivity(data)
        heatmap = fig.data[0]
        assert list(heatmap.x) == ["8.0%", "10.0%"]
        assert np.asarray(heatmap.z, dtype=float)[2, 1] == pytest.approx(110.0)  # absolute values, no price
        assert list(fig.data[1].x) == ["8.0%"] and list(fig.data[1].y) == ["6.0%"]

    def test_nan_cells_are_gaps(self, viz):
        data = {"sensitivity": {"growth_rates": [0.02, 0.04], "discount_rates": [0.08, 0.10],
                                "values": [[100.0, None], [np.nan, 90.0]]}, "current_price": 100.0}
        fig = viz.plot_dcf_sensitivity(data)
        assert_chart(fig)
        assert fig.data[0].text[0][1] == "N/A"


def test_radar_closes_polygon(viz):
    fig = viz.plot_radar_comparison({"A": {"x": 10, "y": 20, "z": 30}})
    trace = fig.data[0]
    assert list(trace.r) == [10, 20, 30, 10]
    assert trace.theta[0] == trace.theta[-1]


def test_upside_bars_sorted_and_coloured(viz):
    fig = viz.plot_valuation_upside({"A": 10.0, "B": -5.0, "C": None})
    bar = fig.data[0]
    assert list(bar.x) == [-5.0, 10.0]
    assert bar.marker.color[0] != bar.marker.color[1]
