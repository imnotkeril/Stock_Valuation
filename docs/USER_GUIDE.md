# User guide

Stock Analysis System is a local web app for analysing and valuing US-listed companies.
It is for research and education only - nothing in it is investment advice.

## 1. Start

```
python -m venv .venv
.venv\Scripts\activate            (macOS / Linux: source .venv/bin/activate)
pip install -r StockAnalysisSystem/requirements.txt
python StockAnalysisSystem/run.py
```

Open http://localhost:8501. The app only listens on this computer. Docker:
`docker build -f StockAnalysisSystem/Dockerfile -t stock-analysis .` then
`docker run -p 8501:8501 --env-file .env stock-analysis`.

## 2. Optional keys (`.env`)

Copy `.env.example` to `.env`. Everything works with Yahoo Finance alone; each key unlocks
one extra feature, and a feature without its key tells you what to add.

| Setting | Unlocks | Where to get it |
|---|---|---|
| `SEC_EDGAR_USER_AGENT` | Overview > SEC filings and 10-year history | Your own `Name email@domain` (SEC requires a contact) |
| `FINNHUB_API_KEY` | News tab (headlines, sentiment, peers) | finnhub.io, free |
| `ALPHA_VANTAGE_API_KEY` | News tab per-ticker sentiment; price fallback | alphavantage.co, free |
| `FINANCIAL_MODELING_PREP_API_KEY` | Valuation > Precedents: recent M&A feed | financialmodelingprep.com |
| `STOCK_ANALYSIS_OFFLINE=1` | Never call the internet, use cached data only | - |

Data is cached in `StockAnalysisSystem/src/data/cache`; **Data & settings > Clear cached
data** in the sidebar forces a refresh. Your own data (screener templates, deals) is in
`StockAnalysisSystem/src/data/user`.

## 3. Pages

### Home
Feature overview, company search, quick-start tickers (one per sector) and a sector ETF snapshot.

### Company Analysis
Pick a company in the sidebar (search or popular list) and click **Analyze company**.

| Tab | What you see |
|---|---|
| Overview | Price chart with moving averages, key statistics, business profile, financial snapshot, key ratios vs. sector, performance vs. sector ETF, technical indicators, **SEC filings and long-term history** (button) |
| Financials | Income statement, balance sheet, cash flow, margins, growth, common-size statements, health score |
| Valuation | **Fair Value Summary** (weighted), **DCF** with sensitivity grid, **APV**, **Relative** multiples vs. sector, **Asset-Based**, **Sector Model** (sector-specific methods with adjustable assumptions, including **Sum of the Parts**), **Precedents** (deal multiples) |
| Risk | Health score, Altman Z / Springate / Zmijewski, liquidity and solvency vs. sector, revenue stress test, **Scenario analysis** (macro shocks, 2008/2020/2022), market risk (volatility, drawdown, VaR) |
| News | Headlines with sentiment, daily sentiment, peers (needs Finnhub or Alpha Vantage key) |
| Peers | Comparison with sector peers |

Tips:
- **Sum of the Parts**: open Valuation > Sector Model, enable assumption overrides and set the
  second segment's share (from the segment note of the annual report) and both multiples.
  With share 0 the method is not run and does not affect the fair value.
- **Precedents**: add deals to the table (date, acquirer, target, sector, EV/Revenue and/or
  EV/EBITDA, source) and click **Save deals**, or import a CSV with the same columns. Deal
  multiples include a control premium, so values are usually above trading multiples.
- **APV** and **Scenario analysis** do not apply to banks and insurers; the tabs say so.

### Peer Comparison
Compare a company with its sector peers on growth, profitability, valuation and price performance.

### Sector Analysis
- **Single sector**: sector ETF vs. market, leading companies, metric comparison, valuation heatmap, outlook.
- **Compare sectors**: pick 2-6 sectors - growth of 100 vs. SPY, return / volatility / drawdown /
  Sharpe, median fundamentals of each sector's largest companies, leading sector per metric.

### Stock Screener
Choose a strategy preset or build filters (sector, market cap, ratio ranges) and click **Run
screen**. **Save filters as template** stores the current filters under a name; saved templates
appear in the Strategy list with "(saved)" and can be deleted there. The first live run
downloads the whole universe (~2-3 minutes); later runs use the cache.

### Valuation Filter
Ranks companies by the upside of their model fair value over the market price; adjust
multiple discounts and DCF assumptions.

### Forecasting
Price forecasts (ARIMA, exponential smoothing, linear trend, random walk, machine learning,
**neural network (MLP)**, ensembles) with confidence bands and hold-out backtests;
fundamentals forecasts; macro indicators (needs `pandas-datareader`); Prophet (needs `prophet`).

### Reports
Company, valuation, comparison and sector reports as PDF, Excel, CSV or JSON.

## 4. Limits
- Yahoo Finance is unofficial: coverage varies by company and statements cover 4-5 years
  (SEC EDGAR adds up to 10).
- The peer and screening universe is ~130 large US companies.
- Every valuation depends on its assumptions - check the per-method details and adjust them.
- Sentiment scores are context, not trading signals.
