# Stock Analysis System — Specification (unified)

Version: 2026-11-23. Single source of truth for scope. Merges the original feature list
(`docs/sources/in-development/ORIGINAL_README.md`), the readiness audit of 2026-10-24
(`docs/sources/in-development/AUDIT_2026-10-24.md`), the In Development handoff, and what is
actually built on branch `claude/peaceful-euler-aq2j08`.

Status legend: **Done** = implemented and covered by offline tests · **Partial** = works with
limits · **Not built** = out of scope (see §6).

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
- **Valuation**: weighted fair value summary; interactive DCF with sensitivity grid; APV
  (unlevered FCF at Hamada unlevered cost of equity + PV of interest tax shields; n/a for Financials);
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
- Compare sectors (view on the Sector Analysis page): 2-6 sectors side by side - ETF growth of 100
  vs. SPY, return/volatility/drawdown/Sharpe, median fundamentals of each sector's largest
  universe companies, leader per metric (lowest positive multiple, highest margin/return/growth).

### 3.5 Screening — Done
- Stock screener: filter the universe by sector, size, profitability, valuation metrics;
  built-in strategy presets; save the current filters as a named template, reapply or delete it
  (stored in `src/data/user/screener_templates.json`, git-ignored).
- Valuation filter: rank by model fair-value upside; honours user multiple discounts and DCF
  adjustments; EV/EBITDA bridged through net debt.

### 3.6 Forecasting — Done (Prophet optional)
- Price: ARIMA, exponential smoothing, linear, random walk, ML, ensembles; confidence
  intervals; backtests; Monte Carlo probabilities.
- Fundamentals: statement-metric forecasts; bull/base/bear scenarios.
- Macro: FRED indicators — Done: public fredgraph CSV (no key, no extra package), cached a day,
  ARIMA/ML/ensemble per series and joint VAR.
- Prophet model — optional: shown only when `prophet` is installed (heavy dependency, by choice).

### 3.7 Reports — Done
Company, valuation, comparison and sector reports as PDF (with matplotlib charts), Excel,
CSV, JSON downloads.

### 3.8 Data and configuration — Done
- Providers: Yahoo Finance (primary, no key), Alpha Vantage (fallback: prices, overview,
  search), FRED (macro, CSV), Finnhub (news, metrics, peers), FMP (M&A feed; free plan = 5 latest),
  SEC EDGAR (free; needs `SEC_EDGAR_USER_AGENT` with the user's own
  contact: recent filings with links and 10-year annual history from XBRL facts on Overview).
  Alpha Vantage NEWS_SENTIMENT is a premium endpoint: with a free key the News tab uses Finnhub.
- Env: `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY`,
  `SEC_EDGAR_USER_AGENT`, `STOCK_ANALYSIS_OFFLINE`.
  Loaded with `load_dotenv` from `StockAnalysisSystem/.env` or repo-root `.env`. No keys in code.
- Cache: `src/data/cache` (git-ignored); last cached copy served when a provider fails.

### F20 Review fixes: credible numbers (P1) — source `docs/review/PRODUCT_REVIEW.md` §5 P1
- AC20.1 Given network access, when any valuation runs, then the risk-free rate is FRED DGS10 (as-of date shown), with stale-cache and constant fallbacks and a user override (R1).
- AC20.2 Given one company, when the DCF tab, APV and the sector model compute a discount rate, then they share one cost-of-capital module (same rf, ERP, beta rule, Kd band) and a note explains why their values differ (R2).
- AC20.3 Given AAPL was analysed, when the user switches to JPM, then every Valuation/Risk control shows JPM defaults (R3).
- AC20.4 Given the Valuation Filter results, when an upside exceeds +-100 % or the model is not DCF, then the row is flagged and the real method is shown (R4).
- AC20.5 Given a bank, when the Risk tab and Peer Comparison render, then no FCF penalty, no duplicate Altman text, no N/A method and no 0.00 % rates appear (R5).
- AC20.6 Given a valuation, when the recommendation badge renders, then it carries a confidence level and softens the label when confidence is low (R6).

### F21 Review fixes: the product explains itself (P2)
- AC21.1 Given a reader of the docs, when they open README/USER_GUIDE, then screenshots and `docs/METHODOLOGY.md` explain how each number is made (R7).
- AC21.2 Given the Relative tab, when universe data is cached, then live sector medians (dated) are used and irrelevant multiples are dropped (R8).
- AC21.3 Given the DCF tab, when growth is set, then it fades to terminal growth and SBC can be deducted (R9).
- AC21.4 Given Peer Comparison or Sector Analysis, when the page opens empty, then a hint and a main-area run button are shown; a unique search match is selected by Enter (R10, U6-U9).
- AC21.5 Given a cold cache, when the screener runs, then fetches are parallel within a rate limit (R11).
- AC21.6 Given the Precedents tab, when it is empty, then it explains its status and offers labelled sample data (R12).
- AC21.7 Given the test suite, when it runs, then golden hand-computed values for DCF, APV, WACC and residual income pass (R13).

### F22 Owner knowledge-base gaps, small rows (source `docs/sources/KNOWLEDGE_BASE_GAP.md`, 2026-11-26)
- AC22.1 Given any company with a balance sheet, when the Valuation tab opens, then a **Liquidation value** section shows orderly and forced liquidation value per share from asset-class recovery rates (cash 100 %, receivables, inventory, PP&E, other assets) minus all liabilities, with editable recovery rates; it is a floor, not part of the weighted fair value.
- AC22.2 Given a price history and the sector ETF / S&P 500 benchmark, when the Risk tab opens, then **Sortino, Treynor, Calmar, Information ratio and Jensen's alpha** are shown next to Sharpe with one-line definitions; each is N/A when its input is missing (no benchmark, beta <= 0, no drawdown).
- AC22.3 Given `FINNHUB_API_KEY`, when the Overview tab opens, then recent **insider transactions** (date, name, buy/sell, shares, price) and a 90-day net buy/sell summary are listed; without the key the section explains how to enable it and never raises.
- AC22.4 Given a balance sheet, when Financials > Ratios opens, then the **A1-A4 / P1-P4 liquidity grouping** (assets by liquidity, liabilities by maturity) is shown with the four classic conditions A1>=P1, A2>=P2, A3>=P3, A4<=P4 and a verdict (absolutely liquid or not).
- Candidates left for the owner (not built): value adjustments (control premium, non-core assets, contingent liabilities), own models for Materials / Utilities / Staples (medium); bank capital adequacy CET1 (blocked: not in yfinance); Porter / moat / management (not computable).

## 4. Non-functional requirements
- Works without network in tests (`STOCK_ANALYSIS_OFFLINE=1`); CI green on 3.11–3.13.
- `ruff check` clean; no secrets in code or new commits.
- Pages render in a few seconds for one ticker after first download (cache warm).
- Local-only by default: the server listens on localhost; Docker binds 0.0.0.0.

## 5. Original ideas from the original README (all built in v2)

| Idea | Status |
|---|---|
| Adjusted Present Value (APV) | Done (V2): Valuation > APV - unlevered value + PV of tax shields (n/a for Financials) |
| Precedent transactions method | Done (V7): Valuation > Precedents - user-maintained deal table (no bundled figures), median/IQR deal multiples, optional FMP M&A feed |
| News / sentiment analysis | Done (V5-V6): News tab - Finnhub news (lexicon-scored) + Alpha Vantage NEWS_SENTIMENT (premium key only), daily sentiment, peers; keys optional |
| SEC EDGAR data source | Done (V1): Overview > SEC - recent filings and 10-year XBRL history; needs `SEC_EDGAR_USER_AGENT` |
| Finnhub data source | Done (V5): `utils/news_sentiment.py` - news, metrics, peers with `FINNHUB_API_KEY` |
| Deep-learning forecasts (TensorFlow, pytorch-forecasting) | Done (V8) as a neural network (MLP, scikit-learn) price model; no TensorFlow/torch dependency |
| Market risk simulation / stress testing beyond revenue stress + VaR | Done (V4): Risk tab scenario analysis - 5 macro shocks on EV/EBITDA with interest cover, 2008/2020/2022 beta-implied and actual drawdowns |
| SOTP for every sector | Done (V3): two-segment SOTP with user-set split and multiples in every sector model; explicit `segments=[...]` override |

Decision (owner, 2026-11-23): build all of these as v2 - see `docs/plan.md` V1-V8. Keyed sources (Finnhub, Alpha Vantage news, FMP) stay optional: missing key = feature explains what to add, never an error.

## 6. Out of scope
User accounts, portfolios, live trading, non-US markets, database storage, hosted
multi-user deployment.
