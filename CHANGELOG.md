# Changelog

All notable changes to Stock Analysis System. Dates are commit dates.

## [Unreleased] - review fixes and round 2 (branch `claude/review-fixes-finish-2bedc3`, 2026-11-26)

### Added
- **Liquidation value** on Valuation > Asset-Based: orderly and forced value per share from asset-class recovery
  rates (cash 100 %, receivables, inventory, long-term investments, PP&E, other; intangibles 0), minus all
  liabilities, floored at zero, with editable rates. A floor only: not part of the weighted fair value. (F22, T22)
- **Risk-adjusted ratios** on Risk > Market risk: Sortino, Treynor, Calmar, information ratio and Jensen's alpha
  next to Sharpe, against the S&P 500 (SPY), each with a one-line definition and N/A when its input is missing. (F22, T23)
- **Insider transactions** on Overview with `FINNHUB_API_KEY`: recent Form 4 trades and a 90-day summary of
  open-market buys and sells; without the key the section explains how to enable it. (F22, T24)
- **A1-A4 / P1-P4 liquidity grouping** on Financials > Ratios with the four absolute-liquidity conditions and a
  verdict (Sheremet / Savitskaya method, mapped to the US GAAP balance sheet). (F22, T25)
- **One cost of capital for every model** (`valuation/capital.py`, ADR-007): risk-free rate from FRED DGS10 with a
  user override (`STOCK_ANALYSIS_RISK_FREE_RATE`), stale-cache and constant fallbacks; ERP 4.4 %; Blume beta;
  cost of debt clamped to rf + 0.5-8 %; WACC clamped to 6-20 %. (R1, R2)
- **Recommendation confidence** (High / Medium / Low) with reasons; Low softens Buy and Sell labels. (R6)
- **Relative valuation on live sector medians** from the cached universe (at least five companies, at most 30 days old,
  dated); irrelevant multiples are dropped per sector. (R8, T12)
- **Precedents sample**: a labelled, illustrative five-deal sample on request; precedents are a cross-check, not weighted. (R12, T16)
- **Golden-value tests** for DCF projection, APV, WACC, residual income and the justified P/B, hand-computed. (R13, T17)
- **Methodology document** `docs/METHODOLOGY.md`: how every number is made, sources, known biases and open items.
- **Docs gate** `StockAnalysisSystem/tools/docs_check.py`: screenshots and METHODOLOGY linked from README, disclaimer
  phrase present in README, USER_GUIDE and METHODOLOGY, every relative link resolves.

### Changed
- **DCF tab** starts from unlevered free cash flow (interest added back after tax); growth fades to terminal growth
  by default; SBC is deducted by default when reported; terminal growth above rf is flagged. (R9, T13)
- **Peer Comparison and Sector Analysis** open with a hint and a main-area run button; results stay when the view
  changes. (R10, T14)
- Enter on a single US match in the sidebar search analyses the company at once; foreign listings are not counted
  as matches. (U6-U9, T14, T19)
- **Sector notes (static)** tab on Sector Analysis, with a caption that the text is static. (T14)
- Screener and valuation-filter fetches run in parallel under a shared rate limit with progress. Cold-cache
  screener run: 3 min 20 s -> 33 s (owner measurement, T15). Valuation filter hides implausible results (|upside| > 100 %)
  unless ticked, and shows the method behind each value. (R4, R11, T15)
- Banks and insurers: no free-cash-flow penalty in the Risk tab and Peer Comparison. (R5)
- Stress tests use `rf + ERP` (not a fixed 9 %) as the discount rate. (ADR-007)
- Risk tab shows the risk-free rate with its source and date.

### Fixed
- **Leases are no longer debt** (leases are after rent in cash flows and EBITDA); debt comes from one balance-sheet period,
  so the WACC, every equity bridge and the multiples share one debt figure. (S-D1)
- **Banks and insurers use common equity** and net income to common stockholders when reported (book value, ROE, P/B,
  excess return, DDM). (W-D8, S-D3)
- **Generic FCFF deducts SBC once** (no deduction when FCF fell back to net income). (W-D3)
- **APV terminal tax shield** is a fixed-debt perpetuity discounted at Kd, not a growing shield at Kd (the value
  had been about double). (W-D2)
- **Telecom subscriber and user value** uses a cash margin (EBIT after tax + D&A - capex), not the EBITDA margin. (S-D4)
- **REIT FFO** follows Nareit: net income less property-sale gains, plus impairments and D&A. (S-D2)
- **One tax rule** (ADR-007) for the WACC, the FCFF add-back and the driver NOPAT. (W-D5b)
- **Terminal growth is capped at the risk-free rate** in the engine and the sector models, with a note. (W-D7)
- **Zmijewski probability** is the probit `Phi(X)`, not a logistic. (distress models)
- **DCF tab SBC window**: the 3-year average base deducts the 3-year average SBC. (W-D9)
- Switching company resets only the company-specific controls; a bank after a tech company shows bank defaults. (R3)
- A one-step pick ignores foreign listings. (T19)
- Confidence counts only methods that ran. (review round 1, R-01)
- Method-difference text no longer uses absolute wording; the Peer Comparison helpers are defined before the main guard. (review round 1, R-05, R-07)
- Filter relative leg drops multiples irrelevant to the sector; a raising progress callback cancels queued work;
  stale cached company info is left out of live medians. (R-08 to R-11, round 2)

### Documentation
- `README.md`: feature table for the current app, screenshots, configuration names from `.env.example`, run / test /
  lint / Docker commands, disclaimer. Reports export HTML, Excel, CSV, JSON and PDF (Excel and PDF when their packages
  are installed).
- `docs/USER_GUIDE.md`: updated for the new tabs, the search, the run buttons, the sector notes, precedents, liquidation,
  the risk ratios, insider transactions with `FINNHUB_API_KEY`, and the A1-A4 / P1-P4 grouping, with screenshots.
- `docs/SPEC.md`: F20, F21 and F22 marked Done.

### Known open items
- Terminal value does not tie reinvestment to growth in the forecast-driven DCF; the debt tax shield uses the effective
  rate, not the marginal rate; the DCF tab does not deduct minority interest and preferred stock; the VZ DCF tab
  (about 235 against a price near 47 at the review date) is driven by the 6 % WACC floor and telecom default growth.
  See `docs/METHODOLOGY.md` section 10.

## Earlier unreleased work - branch `claude/peaceful-euler-aq2j08`

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
