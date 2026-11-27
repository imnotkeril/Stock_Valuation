# QA report — 2026-11-23

Branch `claude/peaceful-euler-aq2j08`, Windows 11, Python 3.11.9, Streamlit 1.67.0, live Yahoo
Finance data (no API keys).

## What was run

| Check | Result |
|---|---|
| Offline suite `pytest` | 282 passed (280 + 2 new secrets tests) |
| `ruff check StockAnalysisSystem` | All checks passed |
| Live smoke (AppTest, real network): Company Analysis for AAPL, JPM, XOM, NEE, AMZN, PG, CAT, PLD, JNJ, GOOGL | 10/10 OK, 15 tabs each, correct sector model picked (Utilities → Energy, Staples → Retail) |
| Live smoke: every other page opened and every button clicked (Home ×17, Peer Comparison, Sector Analysis, Stock Screener, Valuation Filter, Forecasting ×3, Reports) | all OK, no exceptions, no `st.error` |
| Browser: Home, AAPL Company Analysis → Valuation tab | renders; no app errors in console |
| `pip-audit` on the project venv | No known vulnerabilities (after upgrading venv `setuptools` 65.5 → latest) |
| Docker: `docker build -f StockAnalysisSystem/Dockerfile`, `docker run -p 8502:8501` | builds; `/_stcore/health` → `ok` |
| Local server binding | listens on `127.0.0.1:8501` only; log shows a single local URL |

Script for the live smoke run is not part of the repo (it hits the network); the same flows
are covered offline by `tests/test_app.py` and `tests/test_pages_interaction.py`.

## Fixed during QA
- Duplicate sidebar navigation (Streamlit auto-listed `src/pages/`): `showSidebarNavigation = false`.
- 33 × deprecated `use_container_width=True` → `width="stretch"` (Streamlit announced removal).
- Server listened on all interfaces with `enableCORS = false`: now `address = "localhost"`,
  CORS default restored. Docker keeps `--server.address=0.0.0.0`.
- Ported guard test against hard-coded API keys (`tests/test_no_hardcoded_keys.py`).
- Financials (JPM): the Valuation → DCF tab and the Financials → FCF sustainability block no longer
  warn about negative free cash flow; they explain that FCF is not meaningful for banks and insurers.
  Re-checked live: JPM and AAPL render with no warnings.

## Observations (not bugs, tracked in plan)
- Stock Screener first run takes ~150 s live (downloads the whole ~130-company universe);
  later runs use the cache.
- AAPL weighted fair value $193.96 vs price $342.56 (−43%): conservative defaults; values are
  assumption-driven by design (see SPEC §3.3).
- Macro forecast (FRED) needs optional `pandas-datareader`; not installed, button reports it.
  Fixed 2026-11-26: FRED is read from the public CSV, no package needed (see below).

## Not covered
- Alpha Vantage fallback (no key used).
- Optional `prophet` and `pandas-datareader` paths.
- Linux/macOS run (CI covers tests on Ubuntu only).

## v2 re-check — 2026-11-24

| Check | Result |
|---|---|
| Offline suite | 363 passed |
| `ruff check` | clean |
| Live smoke, Company Analysis (XOM, AAPL, JPM, PLD, JNJ, CAT, GOOGL, NEE, AMZN, PG) | 10/10 OK, 18 tabs each (new: APV, Precedents, News) |
| Browser: Sector Analysis > Compare sectors (4 sectors, live) | OK: XLK/XLF/XLE/XLV vs SPY, leaders table |
| Browser: XOM Valuation > APV | renders; default unlevered cost was 3.7 % from a very low beta - now floored at 6 % (APV $109 vs price) |

Not live-checked (need the user's settings): SEC EDGAR (`SEC_EDGAR_USER_AGENT`), News
(`FINNHUB_API_KEY` / `ALPHA_VANTAGE_API_KEY`), FMP M&A feed. All three are covered offline with
responses in the providers' JSON formats. 2026-11-24: owner chose to skip these live checks
and finalize; they stay open until keys are added to `.env`.

## Live check with keys (2026-11-24)
Keys migrated from the old folder's `.env.txt` into `.env` (values never printed).

| Check | Result |
|---|---|
| Finnhub: news / metrics / peers for AAPL | OK: 248 articles, 133 metrics, 10 peers |
| Alpha Vantage NEWS_SENTIMENT | not available: premium endpoint on the free key; News tab works on Finnhub alone |
| FMP M&A feed | was empty: free plan rejects `limit` > 5 with HTTP 402. Fixed: retry with limit 5 -> 5 deals; new test |
| Live AppTest, Company Analysis AAPL | no exceptions, no `st.error`; News tab and "Recent M&A announcements" expander render |
| SEC EDGAR | not checked: old file had only a placeholder User-Agent; needs `SEC_EDGAR_USER_AGENT` from the owner |

364 tests pass, ruff clean.

## FRED without pandas-datareader (2026-11-26)
| Check | Result |
|---|---|
| Offline tests: CSV parser (missing `.` values), one download then cache, offline serves cache | 2 new tests pass |
| Live AppTest, Forecasting > Economic indicators (CPI, unemployment, 10Y Treasury) | data loaded for all 3, ARIMA + ensemble per series, VAR, 6 charts, no exceptions |

## Live check with the owner's new keys (2026-11-26)
| Check | Result |
|---|---|
| SEC EDGAR (`SEC_EDGAR_USER_AGENT` set by owner) | 200; AAPL CIK found, 15 recent filings, 10-year XBRL history |
| Live AppTest, Overview > Load from SEC EDGAR (AAPL) | filings + history tables render, no exceptions or warnings |
| Finnhub news AAPL | 249 articles |
| FMP M&A feed | 5 deals |
| Alpha Vantage GLOBAL_QUOTE | OK (fallback price source) |

All keyed features are now live-checked.


## Review-fix QA pass (T19, 2026-11-26)

Live data (Yahoo, FRED, Finnhub key present), branch `claude/review-fixes-finish-2bedc3`.

**Scripted pass** (Streamlit AppTest, real network, every page; every Company Analysis tab renders on load):

| Page / action | Time | Exceptions / errors | Warnings |
|---|---|---|---|
| Company Analysis AAPL (all tabs) | 6.4 s | none | none |
| Company Analysis JPM (all tabs) | 5.3 s | none | none |
| Company Analysis XOM (all tabs) | 5.7 s | none | none |
| Peer Comparison (main-area run button) | 5.4 s | none | none |
| Sector Analysis (main-area run button) | 5.7 s | none | none |
| Stock Screener run (warm cache) | 34.1 s | none | none |
| Valuation Filter run | 16.6 s | none | none |
| Forecasting run | 6.7 s | none | none |
| Reports generate | 0.8 s | none | none |

New features, live values: AAPL fair value $184.41 (Sell, High), Sortino 1.35, Treynor 40.6 %, Calmar 2.05, IR 0.89,
Jensen's alpha 22.3 %, liquidation $0.00 (liabilities exceed recoveries), insider: 24 open-market sells in 90 days.
JPM $278.97, bank note on liquidation / A1-A4. XOM $87.00, orderly liquidation $14.50, Treynor N/A (1-year regression beta <= 0 rule).

**Browser pass** (port 8502, 1440x900): typing "Apple Inc" + Enter analyses AAPL in one step (Yahoo returns AAPL plus
foreign listings AAPL.SAO, APC.DE; the pick now counts US listings only — found and fixed in this pass, test added).
Overview, Valuation summary, Risk health and Market risk (new ratios with definitions) render; screenshots in `docs/screenshots/`.

Issues: one found (one-step pick with foreign listings), fixed. Known, backlog: VZ DCF tab far above price (WACC floor + telecom default growth).

## Full-system re-test (2026-11-27, branch `claude/system-retest-722b8c` from master 747b6a2)

| Check | Result |
|---|---|
| Offline suite | 533 passed in 194.13s |
| `ruff check` | All checks passed |
| Live providers | FRED DGS10 4.91 % (2026-11-24), CPIAUCSL to 2026-10; SEC EDGAR CIK + filings + XBRL history (O 2016-2025); Finnhub news AAPL 249 / NEE 164 / O 231, insider AAPL 122 / NEE 135 / O 67; FMP M&A 5 deals; Alpha Vantage GLOBAL_QUOTE ok; yfinance prices + 3 statements for AAPL, O, NEE |
| Live valuation | AAPL 184.41 Sell High, JPM 278.97 Reduce Medium, XOM 87.00 Above model value Low, O 50.07 Reduce Medium, NEE 37.46 -> 39.86 (after RT-01) |

Bugs found and fixed (tests in `tests/test_retest_fixes.py`):
- RT-01: subsector defaults for terminal growth / horizon were dead (the static sector table won), so utilities ran at 1 %
  terminal growth while the model declares 2 %; and the Valuation assumptions sliders started at static spec defaults
  (e.g. 9 % discount rate) instead of the values the model used, so turning "Override model assumptions" on without
  moving a slider changed the fair value (synthetic industrial $240.69 -> $205.01). Now subsector defaults beat the
  sector table for those two parameters (WACC still derived), sliders start at the used values and only moved sliders
  are sent as overrides.
- RT-02: the Utilities sector table had 1 % terminal growth, so after RT-01 the DCF and APV tabs (1 %) disagreed with
  the utility sector model (2 %) for NEE. Utilities table set to 2 % (regulated rate base grows with inflation);
  Energy stays at 1 %.
- RT-03: Risk > Market risk showed an unlabelled "Beta" (provider, 5 years monthly; AAPL 1.09) right above ratios
  that use the regression beta over the selected history (AAPL 0.57). Now "Beta (provider)" with help naming all
  three betas (provider, regression, Blume-adjusted in the models).

Live AppTest pass (qa-tester, real network), Company Analysis all 18 tabs for AAPL, JPM, XOM, O, NEE plus Valuation
Filter, Screener (132 of 132), Peer, Sector, Forecasting, Reports (company + valuation for all 5): 0 exceptions,
0 `st.error`, 0 `st.warning`. Rates agree between tabs (WACC, cost of equity, rf 4.91 %, terminal growth) after RT-02.

Observations, not code bugs (backlog):
- NEE $39.86 vs $85.41: rate base proxied by net PP&E, FCFF DCF negative (heavy capex); utility model is conservative.
- O DCF tab $142 and APV $180 (2.4-3x price): default growth = 4-year FCF CAGR 19 %, funded by acquisitions and new
  shares; a per-share growth default would be better (methodology change, owner decision).
- JPM DCF-tab pre-tax cost of debt 12.9 % (interest on deposits / debt); DCF is flagged not meaningful for banks.
- Liquidation $0.00 for AAPL, JPM, O, NEE (liabilities exceed recoveries; floor by design).
- Stress-test caption uses beta 1 (r = rf + ERP) by design (T06).
- Insider rows for grants and gifts have no price or value (provider data). Macro forecast in AppTest showed no
  output; worked in the 2026-11-26 live check, not re-checked in the browser.
- Review of RT-01..03 (reviewer, no P1): RT-01 also activates the declared Energy subsector defaults, midstream 1.5 %
  and renewables 2 % terminal growth (were 1 % from the Energy table; synthetic midstream fair value +3.5 %, renewables
  +8 % or -22 % with negative FCFF). Kept as the models' declared intent and pinned by a test; the generic DCF tab
  still uses the sector table for these subsectors. Memo now refreshes on hit (base run evicted last). Derived values
  outside a slider's range are shown clamped while the model keeps its own value (untouched sliders are not sent).
