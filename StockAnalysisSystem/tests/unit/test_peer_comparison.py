from StockAnalysisSystem.src.comparison.peer_comparison import PeerComparison


def test_helper_methods_are_class_members():
    # regression: these were nested inside other methods and therefore unreachable
    for name in ("compare_specific_metrics", "compare_valuation_multiples", "compare_financial_strength",
                 "compare_growth_metrics", "_analyze_category_comparison", "_calculate_percentile",
                 "_is_lower_better", "_interpret_metric_comparison"):
        assert hasattr(PeerComparison, name), name
