# Stock Analysis System — product and documentation review

Reviewer: independent read-only review, 2026-11-26. Code state: tip of `claude/peaceful-euler-aq2j08`
(3d3a3ee). No product code was changed. Evidence: docs and code read; app run locally
(`run.py`, 127.0.0.1:8501) and every page opened for AAPL and JPM in the built-in browser;
`pytest` re-run (**366 passed, 120 s**); `ruff check` clean. Screenshots: `docs/review/screenshots/`
(19 files, referenced as `#NN`). Cold-cache app, live Yahoo/Finnhub data, prices through 2026-11-25.

## 1. Verdict in five lines

- Feature-complete against its own SPEC; every page opens, nothing crashed in the pass.
- Engineering hygiene is good (366 offline tests, CI, cached/offline mode, no keys in code).
- **Credibility of the headline number is the weak point**: AAPL fair value $193.91 / "Sell, -45.5 %"
  while the Valuation Filter shows +130..+230 % "upside" for CHTR, PGR, CB, CRM, CVS, UNH (#03, #16).
- Several inputs are inconsistent between tabs (three different discount rates for one company; two WACC
  implementations; hard-coded 3 % risk-free rate vs ~5.2 % 10Y).
- Docs describe *what exists* but not *how numbers are made*, and have **zero screenshots**.

## 2. Documentation coverage

### 2.1 Page / tab matrix

Legend: **D** = described and clear · **P** = mentioned, how/assumptions missing · **N** = not described · **W** = wrong/stale.
"Shown" = a screenshot exists in the docs (none do: column is always No).

| Page / tab | Docs | Method, inputs, assumptions explained? | What user can do | Notes (code ref) |
|---|---|---|---|---|
| Home | D | n/a | search, quick-start (auto-runs analysis) | cards list 6 pages, not Reports (#01) |
| Company > Overview | P | No: "Key ratios vs sector" uses **static** benchmarks (`industry/benchmarks.py:1`); technical indicators (RSI/MACD/Bollinger) not described | range, chart type, MAs, SEC button | USER_GUIDE lists sections, nothing on definitions |
| Company > Financials (5 sub-tabs) | P | Sub-tabs Income / Balance / Cash Flow / Ratios / Growth not named; "health score" formula not given | Values / common-size / growth switch | `financials.py:138` |
| Valuation > Fair Value Summary | P | **No**: weights (50/30/10/10 default, `base_valuation.py:218`), recommendation thresholds (`:75`) undocumented | read only | key page, #03 |
| Valuation > DCF | P | **No**: base FCF, default growth (= FCF CAGR capped 25 %), default WACC (= static sector value, `config.py:178`) not stated | 5 sliders + sensitivity grid | #04 |
| Valuation > APV | P | Formula only in SPEC §3.2 (Hamada, shield at Kd); not in USER_GUIDE | sliders | defaults differ from DCF tab |
| Valuation > Relative | P | No: "sector benchmark" = hard-coded medians | none | averages P/E, P/S, P/B, EV/EBITDA incl. P/B ($54 for AAPL) |
| Valuation > Asset-Based | P | one-line | haircut slider | cryptic label "P/x 46.10" |
| Valuation > Sector Model | D/P | Tip on SOTP is good; per-sector formulas only in code | subsector, methods, assumptions, SOTP | #05 |
| Valuation > Precedents | D | Good text; but table is **empty out of the box** and excluded from fair value (not stated) | edit table, CSV, FMP feed | FMP free plan = 5 deals |
| Risk (health, Altman/Springate/Zmijewski, liquidity, stress, scenarios, market risk) | P | Z variant by sector, VaR method (historical 95 %), scenario formula `(r-g)/(r+shock-g)` with r=9 % are not in docs | read only | #06 |
| News | D | OK; lexicon scoring caveat present | Load news | needs key |
| Peers (snapshot) | W | Guide says "comparison with sector peers"; it is a 7-row table + link | open full page | #07 |
| Peer Comparison (5 tabs) | P | Tabs, metric pickers, fair-value table not described | pick peers/metrics/period | #12, #13 |
| Sector Analysis > Single (5 tabs) | P | "Sector Outlook" is **static text** (`sector_analysis.py:573`), SPEC calls it an outlook label | pick sector/subsector/period | #14 |
| Sector Analysis > Compare | D | Good | 2-6 sectors | |
| Stock Screener (4 tabs) | D/P | Presets, templates, 2-3 min first run: good. Preset definitions not listed | filters, templates | measured cold run **3 min 20 s** |
| Valuation Filter (4 tabs) | P | One sentence. Approach (DCF / Relative / Combined), discounts, how fair value is built: not explained | adjust discounts and DCF shifts | #16 |
| Forecasting (4 tabs) | P | Price / Financial Metrics / Scenarios / Economic Indicators: guide names 3 of them; "Model comparison", "Probabilities" views absent | models, horizon, confidence | #17, #18 |
| Reports | W | **HTML** export exists (`reports.py:58`) but README, USER_GUIDE and SPEC list only PDF/Excel/CSV/JSON; Valuation/Comparison/Sector report types only one line | generate, download | #19 |

### 2.2 In code but missing or wrong in docs

| Item | Where | Problem |
|---|---|---|
| HTML report format | `pages/reports.py:58` | undocumented everywhere |
| Multi-stage, forecast-driven, Monte Carlo, scenario DCF | `valuation/dcf_models.py:98,209,246,328` | README/SPEC list them as features; **no page calls them** (only inherited by sector classes), user cannot reach them |
| Technical indicators block | `overview.py:466` | not in README/GUIDE/SPEC |
| Forecasting "Model comparison"/"Probabilities"/"Data" views, Scenarios tab | `forecasting.py:465,1035` | partly in Home card, not in guide |
| Session behaviour: results persist, slider state persists across tickers | `valuation.py:176-190` | undocumented (and buggy, see 3.2) |
| README feature table | `README.md:11-21` | omits APV, SOTP, scenario analysis, News, Precedents, SEC EDGAR, MLP; USER_GUIDE is current, README is not |
| SPEC §2 "280 offline tests" | `docs/SPEC.md:37` | stale, now 366 |
| CHANGELOG | `CHANGELOG.md` | no entry for FRED public-CSV macro or for the Alpha Vantage premium-news limit (both in SPEC) |
| Python version | ruff `target-version = "py310"` vs docs "3.11+" | inconsistent |

### 2.3 Visuals
No screenshots in README, USER_GUIDE or repo (confirmed). For a visual analytical product this is the single
cheapest documentation win. Starter set exists now in `docs/review/screenshots/` (01 home, 02 overview,
03 summary, 04 DCF, 05 sector model, 06 risk, 07 peers, 08 news, 09-11 JPM, 12-13 peer comparison,
14 sector, 15 screener, 16 valuation filter, 17-18 forecasting, 19 reports).

### 2.4 What the docs do well
Setup, keys table ("each key unlocks one extra feature"), limits section, honest "not investment advice",
tip boxes for SOTP/Precedents/APV. CHANGELOG has dates and a Security note.

## 3. Hands-on findings (app run)

### 3.1 What works well
- All 8 pages render for AAPL and JPM; banks get explicit "not meaningful" notices on DCF and APV (`valuation.py:163`) instead of garbage numbers.
- Quick-start tickers run the analysis in one click; Overview, Financials, Risk, News load in a few seconds (cache warm).
- Screener shows a live progress bar ("Screening AVB 105/132") and a usable results table (#15).
- News (Finnhub) loaded 249 articles with sentiment; Reports generated a preview + 5 download buttons in ~6 s (#19).
- Price forecast JPM: forecast, band, P(up) in ~8 s (#17).

### 3.2 Problems (ordered by severity)

| # | Where | Problem | Evidence |
|---|---|---|---|
| U1 | Company > Valuation > DCF/APV | **Stale slider state across tickers**: slider keys are `val_dcf_*` not scoped by ticker and `main.py` never clears them. After AAPL then JPM, the JPM DCF shows AAPL's 12 % WACC, 11 % growth, 7 years (Financials default is 10 %/5y). | `valuation.py:176-190`, #10 |
| U2 | Valuation Filter | Implausible outputs presented as a ranking: CHTR +232 %, PGR +195 % (fair $704 vs $239), CB +174 %, CRM +143 %, CVS +140 %, UNH +136 %; no outlier flag or cap; "Method" column says `DCF` for insurers whose model is residual-income. 18 of 132 "undervalued". | #16, `valuation_filter.py:793` |
| U3 | Valuation tab | Three discount rates for one company: DCF tab 12.00 % (static Technology default), APV unlevered Ke 8.91 % (CAPM), Sector Model CAPM-based. Results $162 / $223 / $200 with no reconciliation or note. | #04, APV text, #05 |
| U4 | Risk > Health score (banks) | JPM: score **46 "Moderate"**, cash-flow 8/100, weakness "Negative free cash flow" — contradicts the app's own statement that FCF is not meaningful for banks (and Financials tab was fixed for this in CHANGELOG). "Revenue stress test: Revenue and operating income are required" for a bank. Altman paragraph is printed twice. | #11, `risk.py:93,140,261` |
| U5 | Peer Comparison > Relative Valuation | "Valuation Method" = `N/A` for all rows; summary shows "Discount Rate 0.00 %, Terminal Growth 0.00 %". | #13, `peer_comparison.py:541-613` |
| U6 | Peer Comparison, Sector Analysis | Main area is blank until the user finds the run button at the bottom of a long sidebar (no hint, unlike Screener which has one). | #12 |
| U7 | Sector Analysis > Sector Outlook | Static text from `SECTOR_CONSIDERATIONS`, same as the Sector Model "considerations"; no computed outlook. | `sector_analysis.py:573-600` |
| U8 | Company selection | Two steps (type, pick from "Matches", then "Analyze company"); not obvious that Enter alone does nothing. | main.py sidebar |
| U9 | Labels | "P/x 46.10" (asset-based), "Change since 2025-11-26" (reads like an event, means 1-year change), AAPL "Beta vs sector 0.16 / correlation 0.20" unexplained and surprising. | #02 |
| U10 | Precedents | Empty editable table out of the box; not part of the weighted fair value; FMP free plan returns only 5 deals. Feature is effectively inert until user does data entry. | `valuation.py:323` |
| U11 | Perf | Screener cold run 3 min 20 s (132 sequential fetches); Peer Comparison runs a full valuation for each of 5 peers (~10 s); no "warm cache" action. | #15 |

Screenshot artefact: during page switches Streamlit briefly overlays the previous page (ghost sidebar in one capture); not an app bug.

## 4. Objective assessment

### 4.1 Strengths (concrete)
1. **Unified valuation contract**: all sector models return one schema (`base_valuation.py: build_valuation_result`); UI controls are generated from model `PARAMETERS` (`base_valuation.py:241`) so a new sector costs little UI code.
2. **Honest "not applicable" behaviour** for banks (DCF/APV/Altman) and for missing data instead of fabricated numbers.
3. **Test and delivery discipline**: 366 offline tests incl. `AppTest` page tests, CI 3.11-3.13, Dockerfile, offline mode, guard test against hard-coded keys (`tests/test_no_hardcoded_keys.py`). Re-run green in 2 min.
4. **Data resilience**: file cache, last-good copy served on provider failure, optional keys that degrade to explanatory text.
5. **Breadth within one local app**: DCF, APV, SOTP, multiples, distress models, scenario shocks, screener, forecasts, exports — rare for a single-user tool.
6. **Sector-aware details**: Altman variant by sector, SBC deducted for tech (`Free cash flow is reduced by 100 % of SBC`), cap-rate NAV for REITs, residual income for banks.

### 4.2 Weaknesses and risks

| Area | Point | Where | Impact |
|---|---|---|---|
| Methodology | **Two WACC implementations**: generic uses raw beta clamped 0.3-3, Kd in [rf+0.5 %, 15 %], WACC 5-20 %; sector models use Blume-adjusted beta, Kd in [rf+1 %, rf+8 %], WACC 6-20 %. Same ticker can get different costs of capital depending on tab. | `base_valuation.py:751-790` vs `sector_specific/common.py:247-285` | High |
| Methodology | ERP 5.5 % defined three times (`base_valuation.py:42`, `common.py:47`, `apv.py:15`); rf 3 % hard-coded in `config.py:169` and again as 9 % in `stress_testing.py:49` and Sharpe. With 10Y near 5.2 %, CAPM cost of equity is understated by ~2 pp while DCF tab uses a static 12 %: errors partly cancel by accident. | listed | High |
| Methodology | Generic DCF: one constant growth rate (default FCF CAGR) for N years then jump to terminal — no fade; terminal value is 61-72 % of EV for AAPL; generic DCF does not deduct SBC while the sector DCF does (AAPL $162 vs $200). | `valuation.py:176-215`, `tech_sector` | Medium |
| Methodology | **Relative valuation vs static sector medians** (not live peers); averaging implied prices from P/E, P/S, P/B, EV/EBITDA includes methods that are meaningless for asset-light firms (P/B-implied $54 for AAPL). Structural low bias for quality compounders. | `benchmarks.py:1-12`, `valuation.py:384` | High |
| Methodology | Fixed blend 50/30/10/10 (DCF/relative/asset/DDM) and Asset-based gets weight even when n/a for the business model; recommendation thresholds (Strong Buy at +25 %) are applied without confidence or dispersion check. AAPL range across methods $175-$200 is narrow and all far below price; user sees "Sell". | `base_valuation.py:218,75` | High |
| Methodology | APV default uses 3-year-average FCF while DCF defaults to latest year; not comparable side by side. APV itself (shield discounted at Kd, growing debt) is internally consistent. | `apv.py:44-84` | Low |
| Data quality | Yahoo (unofficial) with 4-5 annual periods; ~132-company curated universe; sector assigned from provider names then mapped by alias table — mis-mapped names fall through as "Unknown". | `config.py:60-95`, `universe.py` | Medium |
| Data quality | Outliers are not sanity-checked anywhere (U2). One bad input yields a +230 % row in a ranking. | `valuation_filter.py` | High |
| UX | Stale state, blank landing canvases, two-step company pick (U1, U6, U8). | see 3.2 | Medium |
| Performance | Sequential per-ticker fetches; cold screener 3.3 min. | stock_screener | Medium |
| Code quality | 25.7 k LOC; giant modules: `export.py` 1888, `visualization.py` 1441, `valuation_filter.py` 1076, `forecasting.py` 1050, `base_valuation.py` 1284, `common.py` 1176; nested function definitions inside page functions (`valuation_filter.py:720`); **125 `except Exception`** in 32 files hide failures (some surface as silent "N/A"). | repo | Medium |
| Code quality | "ruff clean" is a weak signal: rule set is only `E9,F63,F7,F82,F811,F821,F841,F401` (`pyproject.toml`); no style, bugbear or blind-except checks. | `pyproject.toml` | Low |
| Tests | Strong on schema/behaviour/pages; I saw no independent golden-value tests (e.g. textbook DCF/WACC reference) — not every test file audited. Network paths are only live-checked by hand. | `tests/` | Medium |
| Security | Keys from old `config.py` remain in git history (CHANGELOG notes rotation needed; the session brief says keys were rotated). `.env` is git-ignored; guard test exists. App binds localhost only. | `CHANGELOG.md` | Low |
| Scope honesty | Forecast pages and sentiment carry disclaimers, but the **Sell/Buy recommendation badge has no confidence qualifier**. | `base_valuation.py:75-99` | Medium |

## 5. Recommendations

Value: H/M/L. Effort: S ≤ 1 day, M 2-5 days, L > 1 week.

### P1 — fix before the numbers are trusted

| # | Recommendation | Value | Effort | Where |
|---|---|---|---|---|
| R1 | **Live risk-free rate**: fetch FRED `DGS10` via the existing public-CSV path (`DataLoader` already does FRED CSV), cache 1 day, fall back to last cached then a constant; show "rf 4.9 % as of <date>" in assumptions and let the user override. Replace constant `RISK_FREE_RATE` by a function; propagate to WACC, APV, Sharpe, stress `_BASE_DISCOUNT`. **Caveat**: raising rf lowers every fair value; AAPL is already -45 %, so re-baseline sector default rates/ERP together (see R2) and re-run test fixtures. | H | S-M | `config.py:169`, `utils/data_loader.py`, `base_valuation.py:193`, `common.py:256`, `apv.py:26`, `stress_testing.py:49`, `risk.py:294` |
| R2 | **One cost-of-capital module**: single `cost_of_capital()` used by generic and sector models (one beta rule, one Kd range, one ERP, one WACC range); DCF tab defaults to this WACC, not the static sector table; add a "why do DCF, APV and Sector Model differ" box. | H | M | new `valuation/capital.py`; `base_valuation.py:751`, `common.py:247`, `valuation.py:129` |
| R3 | Fix stale slider state: scope widget keys by ticker or clear `val_*`, `risk_*` keys on ticker change. | H | S | `valuation.py:176`, `main.py` |
| R4 | Valuation Filter sanity layer: flag/cap results beyond ±100 %, show the model actually used (residual income vs DCF), show spread across methods, hide Financials from "DCF" label; default to the sector model only. | H | M | `valuation_filter.py:684-930` |
| R5 | Fix bank health score (exclude cash-flow component for Financials), duplicate Altman text, Peer Comparison `N/A` method and 0 % rates, bank stress-test message. | M | S | `risk.py:93,140,261`, `peer_comparison.py:541-613` |
| R6 | Recommendation badge: show confidence (method dispersion, terminal share, data completeness) and soften to "Below/Above model value" when dispersion is low but all methods share a bias. | M | S | `base_valuation.py:75`, `valuation.py:100` |

### P2 — make the product explain and match itself

| # | Recommendation | Value | Effort | Where |
|---|---|---|---|---|
| R7 | **Docs**: add screenshots (start from `docs/review/screenshots/`), a `docs/METHODOLOGY.md` (WACC build, weights, thresholds, each sector model, benchmarks source, known biases), update README feature table, SPEC test count, HTML export, CHANGELOG (FRED CSV). | H | M | `README.md`, `docs/` |
| R8 | Replace static sector medians in Relative valuation by live medians from the universe cache (the screener already downloads it); drop P/B for asset-light and P/S for profitable firms from the average; show benchmark date. | H | M | `valuation.py:384`, `industry/benchmarks.py`, `universe.py` |
| R9 | Generic DCF: linear growth fade to terminal, SBC toggle, expose Monte Carlo / scenario DCF that already exist in `dcf_models.py:246,328` (or delete the dead code and the doc claims). | M | M | `valuation.py:161`, `dcf_models.py` |
| R10 | Empty-state hints / move run buttons into the main area for Peer Comparison and Sector Analysis; single-step company pick (Enter selects first match). | M | S | `peer_comparison.py`, `sector_analysis.py`, `main.py` |
| R11 | Speed: thread-pool (4-8 workers) fetch with rate limit for screener/valuation filter; "Warm cache" button; progress for Peer Comparison. | M | M | `stock_screener.py`, `data_loader.py` |
| R12 | Seed Precedents with a clearly labelled sample CSV (user-owned) or hide the tab behind the FMP key; include it in the weight table when deals exist. | L | S | `valuation.py:323` |
| R13 | Golden-value tests (hand-computed DCF, APV, WACC, residual income) and a test that WACC is identical across generic and sector paths. | M | S-M | `tests/` |

### P3 — hygiene and growth

| # | Recommendation | Value | Effort | Where |
|---|---|---|---|---|
| R14 | Widen ruff (`B`, `BLE001`, `SIM`, `I`) and narrow the 125 broad excepts; split `export.py`, `visualization.py`, `valuation_filter.py`. | M | L | repo |
| R15 | Saved valuations / watchlist and "what changed since last run" using the existing JSON user store. | M | M | `src/data/user` |
| R16 | Sensitivity tornado (WACC, growth, margin) in PDF report; assumptions page in the report. | M | M | `export.py` |
| R17 | Optional second price source for the ~132 universe when Yahoo field is missing (Finnhub metrics already wired). | L | M | `news_sentiment.py`, `data_loader.py` |

### 5.1 Candidate: macro regime and sector tilts from MacroLens

Looked at `E:/Main/Dev/Python/Done/Macrolens` (FastAPI backend + DB + ML; `schemas/navigator.py` defines quadrants
`Q1_GOLDILOCKS … Q4_STAGFLATION` from a growth score and a Fed-policy score, a `SectorAllocation` list
(overweight/neutral/underweight per quadrant), yield-curve context and `recession_prob_12m` in
`services/cycle_engine.py`; `services/regime_composites.py` builds a LEI proxy from Philly/Empire Fed surveys,
claims, spreads). **Do not import that backend** (heavy, stateful, unrelated deployment); reuse the ideas on top of
data this app already fetches (FRED public CSV).

| Piece | Value for this app | Effort | Where it lives |
|---|---|---|---|
| Live rf (DGS10) | Fixes the largest single constant (R1) | S | `data_loader.py`, `config.py` |
| Yield curve 2s10s and 10s3m (`T10Y2Y`, `T10Y3M`, public CSV) card + 1-line "inverted/normal" | Context for sector valuation and bank models; cheap | S | Home + Sector Analysis header, `utils/macro.py` |
| Recession probability (Estrella-Mishkin probit on 10Y-3M) | Adds a defensible, transparent number; label as context | S-M | same widget |
| Regime quadrant (growth proxy from INDPRO/claims/PMI-like series, policy proxy from fed funds change) | Frames the screener ("which sectors does this regime favour?") | M | `utils/macro.py`, new Home block |
| Sector tilt table by quadrant (static map, sourced from MacroLens `SectorAllocation`) shown next to screener / sector comparison, never feeding fair value | Useful for idea generation; risk of false precision | M | `sector_compare.py`, `stock_screener.py` |
| LEI proxy composite (`regime_composites.py`) | Heavy, more series, more maintenance | L | skip for now |

Guard-rails: keep regime as an *informational overlay*, not as an input to the Buy/Sell badge; show date and
source; offline mode must still work with cached CSV. Suggested order: R1 → yield curve card → recession probability → quadrant + tilt table.

## 6. Reproduction notes
- Run: `.venv\Scripts\python.exe StockAnalysisSystem\run.py`; health `curl http://127.0.0.1:8501/_stcore/health` → `ok`.
- U1: Company Analysis, quick-start AAPL, open Valuation > DCF, then search `JPM`, Analyze, Valuation > DCF: sliders still 12.00 % / 11.0 % / 7.
- U2: Valuation Filter, defaults (Combined, 15 % discounts), Run Valuation Analysis.
- Timing: screener cold 3 min 20 s (132 stocks); all other pages < 12 s with warm cache.
- Not covered: PDF/Excel download contents, Prophet, Docker image, quarterly statements mode, non-US tickers (out of scope), rate-limit behaviour of Yahoo under load.
