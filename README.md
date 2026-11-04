# Stock Analysis System

A Streamlit application for fundamental analysis and **sector-specific valuation** of publicly traded companies: financial statement analysis, ratio benchmarking, multi-method fair value estimates, bankruptcy-risk models, peer and sector comparison, screening, forecasting and exportable reports.

> For research and education only — nothing in this application is investment advice.

## Features

| Area | What you get |
|---|---|
| **Company analysis** | Price chart with moving averages and volume, key statistics, business profile, performance vs. the sector ETF |
| **Financials** | Income statement, balance sheet and cash-flow analysis, margins, growth/CAGR, common-size statements, financial health score |
| **Valuation** | Weighted fair value from several methods (DCF, multiples, asset-based, dividend discount), sensitivity analysis, recommendation |
| **Sector models** | A dedicated model for every sector with subsector-specific methods and adjustable assumptions (see below) |
| **Risk** | Altman Z-score (variant chosen by sector), Springate, Zmijewski, liquidity/solvency, volatility, drawdown, VaR |
| **Comparison** | Peer comparison (performance, ratios, valuation), sector analysis vs. the market |
| **Screening** | Ratio-based stock screener and a valuation filter for under-/over-valued stocks |
| **Forecasting** | ARIMA / exponential smoothing / ML price forecasts with confidence intervals, financial-metric forecasts, probability analysis |
| **Reports** | Company, valuation, comparison and sector reports as Excel, CSV, JSON or PDF downloads |

### Sector-specific valuation

`ValuationFactory` maps each company to a specialised model. Every model declares its subsectors, methods and user-adjustable assumptions, and the UI builds its controls from that declaration — so all sectors get the same complete interface.

| Sector (GICS) | Model | Examples of methods |
|---|---|---|
| Technology | `TechnologySectorValuation` | SBC-adjusted DCF, EV/Revenue & EV/EBITDA multiples, Rule-of-40 adjusted multiples |
| Financials | `FinancialSectorValuation` | Residual income / excess return, justified P/B, P/E, dividend discount |
| Energy, Utilities | `EnergySectorValuation` | DCF, reserve-based NAV, sum-of-the-parts, regulated asset base, contracted cash flows |
| Consumer Discretionary, Consumer Staples | `RetailSectorValuation` | Retail DCF, multiples, store-based / unit economics, sum-of-the-parts |
| Industrials, Materials | `ManufacturingSectorValuation` | Cycle-normalised DCF, replacement value, multiples, earnings power value |
| Real Estate | `RealEstateValuation` | NAV (cap rate on NOI), P/FFO & P/AFFO, real estate DCF |
| Healthcare | `HealthcareValuation` | R&D-capitalised DCF, healthcare multiples, pipeline rNPV |
| Communication Services | `CommunicationValuation` | DCF, multiples, subscriber/user-based valuation, content library value |

All models return the same result schema (`BaseValuation.build_valuation_result`): per-method values, a weighted fair value, upside versus the current price and a recommendation.

## Quick start

Requirements: Python 3.11+.

```bash
git clone https://github.com/imnotkeril/Stock_Valuation.git
cd Stock_Valuation
python -m venv .venv && source .venv/bin/activate
pip install -r StockAnalysisSystem/requirements.txt
python StockAnalysisSystem/run.py
```

The app opens at http://localhost:8501. Choose **Company Analysis** in the sidebar, pick a company and click **Analyze company**.

### Docker

```bash
docker build -f StockAnalysisSystem/Dockerfile -t stock-analysis .
docker run -p 8501:8501 --env-file .env stock-analysis
```

## Configuration

Copy `.env.example` to `.env`. Every setting is optional — Yahoo Finance works without a key.

| Variable | Purpose |
|---|---|
| `ALPHA_VANTAGE_API_KEY` | Fallback provider for prices, company overview and symbol search |
| `FINANCIAL_MODELING_PREP_API_KEY` | Reserved for future use |
| `STOCK_ANALYSIS_OFFLINE` | `1` = never call external APIs, serve cached data only |

Downloaded data is cached in `StockAnalysisSystem/src/data/cache` (git-ignored). When a provider is unavailable, the last cached copy is served. Use **Data & settings → Clear cached data** in the sidebar to force a refresh.

Optional packages: `prophet` (Prophet forecasts) and `pandas-datareader` (FRED macro indicators).

## Project structure

```
StockAnalysisSystem/
├── run.py                         # Launcher (python StockAnalysisSystem/run.py)
├── Dockerfile
├── requirements.txt / requirements-dev.txt
├── src/
│   ├── main.py                    # App shell: navigation, company selection, data loading
│   ├── navigation.py              # Page names and cross-page navigation helpers
│   ├── config.py                  # Settings, sector names (normalize_sector), DCF parameters, colours
│   ├── utils/
│   │   ├── data_loader.py         # yfinance / Alpha Vantage / FRED access with caching
│   │   ├── financial_data.py      # Robust statement line-item lookup, canonical data bundle
│   │   ├── visualization.py       # Shared Plotly charts
│   │   └── export.py              # Excel / CSV / JSON / PDF exports
│   ├── models/                    # Ratio analysis, statement analysis, bankruptcy models, forecasting
│   ├── valuation/
│   │   ├── base_valuation.py      # Unified valuation contract + generic methods
│   │   ├── dcf_models.py          # Multi-stage, forecast-driven, Monte Carlo and scenario DCF
│   │   ├── sector_factor.py       # ValuationFactory: sector -> model
│   │   └── sector_specific/       # One model per sector
│   ├── industry/                  # Company universe, sector benchmarks, sector ratios
│   └── pages/                     # Streamlit pages (company analysis, comparison, screening, ...)
└── tests/                         # Offline test suite (real AAPL/MSFT fixtures + synthetic sector profiles)
```

## Development

```bash
pip install -r StockAnalysisSystem/requirements.txt -r StockAnalysisSystem/requirements-dev.txt
ruff check StockAnalysisSystem
pytest
```

The test suite never touches the network: it uses financial statements and prices recorded from Yahoo Finance for Apple and Microsoft, plus internally consistent synthetic companies for every sector (bank, insurer, REIT, oil & gas, utility, retailer, …). The Streamlit pages are exercised end-to-end with `streamlit.testing.v1.AppTest`.

### Using the valuation models from Python

```python
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

factory = ValuationFactory(DataLoader())
result = factory.get_company_valuation("AAPL")
print(result["value_per_share"], result["potential_upside"], result["recommendation"]["label"])
for key, method in result["valuation_methods"].items():
    print(method["method"], method["value_per_share"])
```

## Data sources and limitations

- Market data and fundamentals come from Yahoo Finance via `yfinance` (unofficial API — availability and field coverage vary by company).
- Statement history is limited to what the provider returns (typically 4–5 annual periods).
- Peer and screening universes are a curated list of ~130 large-cap US companies (`src/industry/universe.py`).
- Valuation outputs depend heavily on assumptions; always review the per-method details and adjust them.

## License

MIT — see [LICENSE](LICENSE).
