# Stock Analysis System — Specification (unified)

Version: 2026-11-23. Single source of truth for scope. Merges the original feature list
(`docs/sources/in-development/ORIGINAL_README.md`), the readiness audit of 2026-10-24
(`docs/sources/in-development/AUDIT_2026-10-24.md`), the In Development handoff, and what is
actually built on branch `claude/peaceful-euler-aq2j08`.

Status legend: **Done** = implemented and covered by offline tests · **Partial** = works with
limits · **Not built** = backlog or out of scope (see §5).

## 1. Product

A local Streamlit application for fundamental analysis and sector-specific valuation of US
large-cap equities. One user, runs on their own machine (or in Docker). Research and education
only, never investment advice.

Users: an individual investor/analyst who wants a fair-value estimate with transparent,
adjustable assumptions, risk checks and comparisons, and an exportable report.

Constraints:
- Python 3.11–3.13; Streamlit UI; works with **no API keys** (Yahoo Finance via `yfinance`).
- All external data is cached; an offline mode serves cached data only.
- No database, no accounts, no server-side state beyond the local cache.

## 2. Architecture (as built)

- Entry: `StockAnalysisSystem/run.py` → `src/main.py` (app shell: sidebar navigation,
  company selection, data loading). Page routing in `src/navigation.py`.
- Data: `utils/data_loader.py` (yfinance, Alpha Vantage fallback, FRED; file cache in
  `src/data/cache`), `utils/financial_data.py` (robust line-item lookup, canonical bundle).
- Analysis: `models/` (ratios, statements, bankruptcy, forecasting engine).
- Valuation: `valuation/base_valuation.py` (unified result contract), `dcf_models.py`,
  `sector_factor.py` (`ValuationFactory`), `sector_specific/*` (one model per sector).
- Universe and benchmarks: `industry/` (~130 curated US large caps, sector ratios).
- Output: `utils/visualization.py` (Plotly), `utils/export.py` (Excel/CSV/JSON/PDF).
- Config: `src/config.py`, `.env` (all optional), `.streamlit/config.toml` (dark theme).
- Quality: 280 offline tests (`pytest`, real AAPL/MSFT fixtures + synthetic sector companies,
  pages via `streamlit.testing.v1.AppTest`), `ruff`, GitHub Actions CI (3.11/3.12/3.13),
  `StockAnalysisSystem/Dockerfile`.

## 3. Features

### 3.1 App shell — Done
- Sidebar navigation: Home, Company Analysis, Peer Comparison, Sector Analysis, Stock Screener,
  Valuation Filter, Forecasting, Reports. Cross-page jumps keep the active ticker.
- Company search by ticker or name; quick-start tickers per sector on Home.
- Home: feature cards, sector ETF snapshot.
- "Data & settings": clear cached data.
- Results survive the next interaction (session state).

AC: every page opens without error for AAPL; selecting a company on one page carries it to
the others; clearing the cache forces a fresh download.

### 3.2 Company analysis — Done
Tabs:
- **Overview**: price chart with moving averages and volume, key statistics, business profile,
  performance vs. sector ETF, change since first available price.
- **Financials**: income statement, balance sheet, cash flow; margins; growth/CAGR;
  common-size statements; financial health score.
- **Valuation**: weighted fair value summary; interactive DCF with sensitivity grid;
  multiples vs. sector; asset-based value; sector model; recommendation.
- **Risk**: health score; Altman Z (variant by sector), Springate, Zmijewski; red flags;
  liquidity/solvency vs. sector; revenue stress test; volatility, drawdown, VaR.

AC: for one company per sector (AAPL, JPM, XOM, NEE, AMZN, PG, CAT, PLD, JNJ, GOOGL) all tabs
render; methods without inputs show "not applicable" instead of numbers.

### 3.3 Sector-specific valuation — Done
`ValuationFactory` maps GICS sector → model. Every model declares subsectors, methods and
user-adjustable assumptions; the UI builds controls from the declaration. Shared result
schema (`BaseValuation.build_valuation_result`): per-method values, weighted fair value,
upside vs. price, recommendation.

| Sector | Model | Methods |
|---|---|---|
| Technology | `TechnologySectorValuation` | SBC-adjusted DCF, EV/Revenue, EV/EBITDA, Rule-of-40 multiples |
| Financials | `FinancialSectorValuation` | Residual income / excess return, justified P/B, P/E, DDM |
| Energy, Utilities | `EnergySectorValuation` | DCF, reserve NAV, SOTP, regulated asset base, contracted cash flows |
| Consumer Discr./Staples | `RetailSectorValuation` | Retail DCF, multiples, store/unit economics, SOTP |
| Industrials, Materials | `ManufacturingSectorValuation` | Cycle-normalised DCF, replacement value, multiples, EPV |
| Real Estate | `RealEstateValuation` | NAV (cap rate on NOI), P/FFO, P/AFFO, RE DCF |
| Healthcare | `HealthcareValuation` | R&D-capitalised DCF, multiples, pipeline rNPV |
| Communication | `CommunicationValuation` | DCF, multiples, subscriber value, content library |

Generic DCF (`dcf_models.py`): multi-stage, forecast-driven, Monte Carlo, scenario.

AC: every sector model returns the shared schema for its synthetic test company; changing an
assumption in the UI changes the fair value.

### 3.4 Comparison — Done
- Peer comparison: growth, profitability, valuation, price performance vs. sector peers.
- Sector analysis: sector ETF vs. market, fundamentals of leading companies, sector outlook label.

### 3.5 Screening — Done
- Stock screener: filter the universe by sector, size, profitability, valuation metrics.
- Valuation filter: rank by model fair-value upside; honours user multiple discounts and DCF
  adjustments; EV/EBITDA bridged through net debt.

### 3.6 Forecasting — Done (optional extras Partial)
- Price: ARIMA, exponential smoothing, linear, random walk, ML, ensembles; confidence
  intervals; backtests; Monte Carlo probabilities.
- Fundamentals: statement-metric forecasts; bull/base/bear scenarios.
- Macro: FRED indicators — **Partial**, needs optional `pandas-datareader`.
- Prophet model — **Partial**, needs optional `prophet`.

### 3.7 Reports — Done
Company, valuation, comparison and sector reports as PDF (with matplotlib charts), Excel,
CSV, JSON downloads.

### 3.8 Data and configuration — Done
- Providers: Yahoo Finance (primary, no key), Alpha Vantage (fallback: prices, overview,
  search), FRED (macro). `FINANCIAL_MODELING_PREP_API_KEY` reserved, unused.
- Env: `ALPHA_VANTAGE_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY`, `STOCK_ANALYSIS_OFFLINE`.
  Loaded with `load_dotenv` from `StockAnalysisSystem/.env` or repo-root `.env`. No keys in code.
- Cache: `src/data/cache` (git-ignored); last cached copy served when a provider fails.

## 4. Non-functional requirements
- Works without network in tests (`STOCK_ANALYSIS_OFFLINE=1`); CI green on 3.11–3.13.
- `ruff check` clean; no secrets in code or new commits.
- Pages render in a few seconds for one ticker after first download (cache warm).
- Local-only by default: the server should listen on localhost (see plan).

## 5. Original ideas not built (from the original README)

| Idea | Decision needed |
|---|---|
| Adjusted Present Value (APV) | backlog or drop |
| Precedent transactions method | drop (no free data source) |
| News / sentiment analysis | backlog or drop (needs a news API) |
| SEC EDGAR data source | backlog (free, would add filings depth) |
| Finnhub data source | drop (Alpha Vantage covers the fallback) |
| Deep-learning forecasts (TensorFlow, pytorch-forecasting) | drop (heavy, sklearn ML is in place) |
| Market risk simulation / stress testing beyond revenue stress + VaR | backlog |
| SOTP for every sector | Partial: Energy and Retail only |

Default proposal: keep all of these out of MVP; the owner confirms or picks items for v2.

## 6. Out of scope
User accounts, portfolios, live trading, non-US markets, database storage, hosted
multi-user deployment.
