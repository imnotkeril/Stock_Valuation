# Changelog

All notable changes to Stock Analysis System. Dates are commit dates.

## [Unreleased] - branch `claude/peaceful-euler-aq2j08`

### Added
- **Neural network price forecast** - "Neural Network (MLP)" model on the Forecasting page
  (scikit-learn, two hidden layers, no new dependency). (2026-11-24)
- **Precedent transactions** - Valuation > Precedents: your own deal table (editor + CSV import),
  median and interquartile deal multiples applied to revenue and EBITDA, optional Financial
  Modeling Prep feed of recent M&A announcements. (2026-11-24)
- **News tab** - company news from Finnhub and Alpha Vantage with sentiment scores, daily
  sentiment chart and Finnhub peers; keys optional. (2026-11-24)
- **Scenario analysis** on the Risk tab - five macro shocks (rates +200 bp, recession, margin
  squeeze, de-rating, stagflation) with interest cover, and the 2008 / 2020 / 2022 crises as
  beta-implied and actual drawdowns. (2026-11-24)
- **Sum of the parts in every sector model** - two-segment SOTP with adjustable split and
  multiples (EV/EBITDA; P/E for Financials); off by default. (2026-11-24)
- **APV** - adjusted present value tab on Valuation (unlevered DCF + interest tax shields). (2026-11-24)
- **SEC EDGAR** - recent filings and 10 years of reported financials on the Overview tab
  (needs `SEC_EDGAR_USER_AGENT`). (2026-11-23)
- **Compare sectors** view on Sector Analysis and **saved screener templates**. (2026-11-23)
- Unified specification, plan, QA report and the documents of the earlier `chore/audit` copy
  under `docs/`. (2026-11-23)

### Changed
- The local server listens on localhost only; Docker keeps `0.0.0.0`. (2026-11-23)
- Deprecated Streamlit `use_container_width` replaced by `width="stretch"`. (2026-11-23)
- Financials: negative free cash flow is explained instead of flagged for banks and insurers. (2026-11-23)

### Fixed
- Duplicate sidebar navigation (Streamlit listed `src/pages/` above the app menu). (2026-11-23)
- Empty price history no longer breaks the historical-crisis table. (2026-11-24)

### Security
- Guard test against API keys hard-coded in the source. Keys that were hard-coded in an old
  `config.py` remain in git history and must be rotated by the owner.

## 2026-11-20
- Sector models for every GICS sector on a shared contract, Valuation and Risk tabs,
  forecasting engine (ARIMA, ETS, linear, random walk, ML, ensembles, backtests), page fixes,
  280 offline tests, app shell polish.

## 2026-11-04
- Application shell, unified valuation contract, offline test infrastructure (work in progress).

## 2025-06-06 and earlier (`master`)
- First version of the analysis library and README.
