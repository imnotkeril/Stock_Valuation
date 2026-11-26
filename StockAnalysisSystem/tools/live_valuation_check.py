"""
Live valuation check: values a few tickers with real provider data and prints one line each.

Not a test (needs network). Used after every valuation change, see CLAUDE.md:
    .venv\\Scripts\\python.exe StockAnalysisSystem\\tools\\live_valuation_check.py [TICKER ...]
Exit code 1 when any ticker fails to produce a fair value.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from StockAnalysisSystem.src.utils.data_loader import DataLoader  # noqa: E402
from StockAnalysisSystem.src.utils.financial_data import build_financial_data  # noqa: E402
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM"]


def _pct(value):
    return "n/a" if value is None else f"{value:+.1f}%"


def check(ticker: str, loader: DataLoader, factory: ValuationFactory) -> bool:
    end = date.today()
    info = loader.get_company_info(ticker) or {}
    prices = loader.get_historical_prices(ticker, (end - timedelta(days=400)).isoformat(), end.isoformat())
    statements = {
        "income_statement": loader.get_financial_statements(ticker, "income", "annual"),
        "balance_sheet": loader.get_financial_statements(ticker, "balance", "annual"),
        "cash_flow": loader.get_financial_statements(ticker, "cash", "annual"),
    }
    data = build_financial_data(company_info=info, price_data=prices, **statements)
    result = factory.get_company_valuation(ticker, sector=info.get("sector"), financial_data=data)
    fair, price = result.get("value_per_share"), result.get("current_price")
    rec = (result.get("recommendation") or {})
    methods = result.get("valuation_methods") or {}
    weights = (result.get("valuation_summary") or {}).get("weights") or {}
    per_method = ", ".join(
        f"{k}={m.get('value_per_share'):.0f}" for k, m in methods.items()
        if k in weights and isinstance(m.get("value_per_share"), (int, float))
    )
    assumptions = result.get("assumptions") or {}
    wacc = assumptions.get("discount_rate") or assumptions.get("wacc")
    rf = assumptions.get("risk_free_rate")
    print(f"{ticker:5} {result.get('model', '?'):28} price={price or 0:8.2f} fair={fair or 0:8.2f} "
          f"upside={_pct(result.get('potential_upside'))} rec={rec.get('label')}"
          f"{' conf=' + str(rec.get('confidence')) if rec.get('confidence') else ''}"
          f" wacc={wacc if wacc is None else f'{wacc:.2%}'} rf={rf if rf is None else f'{rf:.2%}'} | {per_method}")
    return bool(fair)


def main(argv: list[str]) -> int:
    tickers = [t.upper() for t in argv] or DEFAULT_TICKERS
    loader = DataLoader()
    factory = ValuationFactory(loader)
    ok = True
    for ticker in tickers:
        try:
            ok &= check(ticker, loader, factory)
        except Exception as exc:  # report and continue with the next ticker
            print(f"{ticker:5} ERROR {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
